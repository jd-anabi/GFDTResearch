import numpy as np
import torch
import hdbscan

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch.distributions import TransformedDistribution
from core import config
from core.SBI.reparam import build_box_bijection, clamp_to_box

# Fixed k-means init for the latent GMM. A prior must be reproducible or no posterior trained from it
# can be: sklearn defaults random_state to the global NumPy RNG, which nothing in this pipeline pins.
GMM_RANDOM_STATE = 0


def resolve_sweep_device(device: torch.device) -> torch.device:
    """The device the LOCAL sweep simulates on, degrading to the CPU when there is no accelerator.

    Every subclass used to pin the flood-fill to ``torch.device('cpu')`` with a hardcode -- they were
    ``@staticmethod``, so they could not see ``self.device`` -- while the GLOBAL sweep ran on the
    accelerator. Measured 6.32 s per inner-loop iteration on the CPU against 0.357 s on CUDA (17.7x),
    and the flood-fill is the dominant cost of a prior build, so that hardcode was most of the wait.

    A caller's device is normally already CPU on a machine without CUDA (``config.detect_device``
    sees to it). This exists for the case where one is handed a cuda device anyway: it must DEGRADE
    with a note rather than raise halfway through a sweep.
    """
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[prior] CUDA was requested for the parameter sweep but is not available; "
              "falling back to the CPU.", flush=True)
        return torch.device("cpu")
    return device


class Prior(ABC):
    def __init__(self, dtype: torch.dtype = torch.float32,device: torch.device = torch.device('cpu')):
        self.dtype = dtype
        self.device = device

    # --- PUBLIC METHODS --- #
    def construct_prior(self, t: torch.Tensor, n_params: int, global_batch_size: int, local_batch_size: int,
                        segs: int, prior_bounds: list[tuple], t_global_scale: int = 1, num_iterations: int = 25,
                        steady: bool = True, n_max: int = 200000, step: float = 0.01,
                        state_dep_drift: bool = False, log_mask: torch.Tensor | None = None,
                        min_cluster_size: int | None = None,
                        min_samples: int | None = None) -> TransformedDistribution:
        """
        Build a stability-screened prior over ND parameters.

        Flood-fills the stable manifold (global Sobol sweep + local random-walk sweep),
        clusters the resulting point cloud in LATENT space (via T.inv), fits a GMM on
        those latent points, and returns a TransformedDistribution that pushes the latent
        GMM forward into the physical box via per-parameter scaled sigmoid.

        The resulting prior has support exactly the cell-file box — no tails leaking into
        nonphysical θ. HDBSCAN's island topology and the GMM's covariance structure are
        preserved; they just live in unbounded latent coordinates.
        """
        self.sweep_device = resolve_sweep_device(self.device)

        n_sims = global_batch_size * num_iterations

        # Global sweep: broad Sobol census of the physical box
        stable_params = self._global_map(
            t[:(t.shape[0] // t_global_scale)], n_params, prior_bounds, segs,
            n_sims, num_iterations, steady, state_dep_drift,
        )

        # Local sweep: random-walk flood-fill of the stable manifold
        accepted_params = np.array(self._local_map(
            t, stable_params, local_batch_size, n_params, n_max, step, segs,
            steady, state_dep_drift,
        ))

        # --- Build the ND bijection from the same bounds the sweep used ---
        # log_mask (per-param) places selected dims in geometric/log coords; None => linear box.
        # The latent GMM below is fit in T_nd's coordinate, so it is consistent with whatever box
        # this is — but a saved prior MUST be reloaded with the SAME mask (file_manager persists it).
        lows = torch.tensor([b[0] for b in prior_bounds], dtype=self.dtype, device=self.device)
        highs = torch.tensor([b[1] for b in prior_bounds], dtype=self.dtype, device=self.device)
        if log_mask is not None:
            log_mask = log_mask.to(device=self.device)
        T_nd = build_box_bijection(lows, highs, log_mask)

        # --- Map accepted physical points to latent space before clustering + GMM fit ---
        # eps-clamp handles the degenerate case where a sweep produced a sample exactly on
        # the box boundary: sigmoid^-1(0) = -inf and sigmoid^-1(1) = +inf would blow up.
        # Shared with gen_training_data via reparam.clamp_to_box so the two cannot drift; safe to
        # clamp in place here because accepted_t was just built and nothing else holds a view.
        accepted_t = torch.tensor(accepted_params, dtype=self.dtype, device=self.device)
        latent_params = T_nd.inv(clamp_to_box(accepted_t, T_nd)).cpu().numpy()  # (N, d) unbounded

        # --- Cluster in latent space (still StandardScaled for HDBSCAN's density metric) ---
        scaler = StandardScaler()
        latent_scaled = scaler.fit_transform(latent_params)
        # Both were LITERALS here. They set the GMM's component count -- HDBSCAN's label count
        # is passed straight to n_components -- so they decide how many modes the prior has,
        # which nothing outside this line could influence.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=(config.PRIOR_CLUSTER_MIN_SIZE if min_cluster_size is None
                              else max(2, int(min_cluster_size))),
            min_samples=(config.PRIOR_CLUSTER_MIN_SAMPLES if min_samples is None
                         else max(1, int(min_samples))))
        labels = clusterer.fit_predict(latent_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 1:
            print('No clusters found. Defaulting to 1 cluster')
            n_clusters = 1
        else:
            print(f'Found {n_clusters} clusters (in latent space)')

        if latent_params.shape[0] < n_clusters:
            raise ValueError(
                f"Not enough stable parameter sets ({latent_params.shape[0]}) to fit {n_clusters} GMM components"
            )

        # --- Fit GMM on UNSCALED latent points (the GMM captures raw latent-space density) ---
        progress_bar = tqdm(total=5, desc="Constructing latent prior...")
        # random_state pins the k-means init; without it the fit was seeded from the global NumPy RNG,
        # so a prior (and every posterior trained from it) could not be reproduced even under a fixed
        # torch seed. Paired with the sorted() in the *_prior samplers -- both are needed.
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full',
                              random_state=GMM_RANDOM_STATE).fit(latent_params)
        progress_bar.update()
        means = torch.tensor(gmm.means_, dtype=self.dtype, device=self.device)
        cov = torch.tensor(gmm.covariances_, dtype=self.dtype, device=self.device)
        weights = torch.tensor(gmm.weights_, dtype=self.dtype, device=self.device)
        progress_bar.update()
        comp_dist = torch.distributions.MultivariateNormal(means, covariance_matrix=cov)
        progress_bar.update()
        mix_dist = torch.distributions.Categorical(probs=weights)
        progress_bar.update()
        latent_prior = torch.distributions.MixtureSameFamily(mix_dist, comp_dist)
        progress_bar.update()
        progress_bar.close()

        # --- Wrap as physical-space prior: sample() returns physical θ in the box ---
        return TransformedDistribution(latent_prior, T_nd)

    # --- PRIVATE METHODS --- #
    @abstractmethod
    def _global_map(self, t: torch.Tensor, n_params: int, prior_bounds: list[tuple], segs: int, batch_size: int, num_iterations: int, steady: bool, state_dep_drift: bool) -> list:
        pass

    @abstractmethod
    def _local_map(self, t: torch.Tensor, stable_params: list, batch_size: int, n_params: int,
                   n_max: int, step: float, segs: int, steady: bool, state_dep_drift: bool) -> list:
        """The flood-fill. An INSTANCE method -- it was a @staticmethod in every subclass, which
        is exactly why all four pinned themselves to the CPU: they could not see self.device.
        Implementations must simulate on ``self.sweep_device``."""
        pass
