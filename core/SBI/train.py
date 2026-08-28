"""The sbi training call, split out of pipeline.py (which stays the public facade).

Every consumer -- orchestrator.build_posterior, scripts/retrain_convergence.py, the test suites --
reaches these as ``pipeline.train_nn`` / ``pipeline._capped_zscore_check``: pipeline.py re-imports
them at its bottom, which also keeps monkeypatching ``pipeline.<name>`` effective. Calls back into
the generation machinery go through the pipeline MODULE OBJECT at call time (``_pipeline.``), never
``from`` imports, so a test that rebinds ``pipeline.gen_training_data`` or
``pipeline.winsorize_summary_block`` is honoured here too.
"""
import contextlib
import math
import sys
import warnings

import torch
from tqdm import tqdm
from sbi.inference.posteriors import DirectPosterior
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from torch.distributions.transforms import Transform

from core.SBI import pipeline as _pipeline


_ZSCORE_CHECK_MAX_ROWS = 200_000


@contextlib.contextmanager
def _capped_zscore_check(max_rows: int = _ZSCORE_CHECK_MAX_ROWS):
    """Run sbi's ``warn_if_zscoring_changes_data`` on a SUBSAMPLE for the duration of this block.

    WHY IT HAS TO BE CAPPED AT ALL. append_simulations calls it UNCONDITIONALLY on the full training
    tensor (sbi/inference/trainers/npe/npe_base.py:189). At the chi retrain's size -- 5000 x 2048 =
    10.24M rows x 114 columns float32, 4.35 GiB -- it runs ``torch.unique(x, dim=0)``, then builds
    ``zx = (x - x.mean(0)) / x.std(0)``, then runs ``torch.unique`` AGAIN. That is >= 13 GiB, and
    since unique over dim=0 is a lexicographic sort of 10.24M rows by 114 keys it is also minutes of
    single-threaded work. On a 16 GB card it is a guaranteed OOM at the END of a multi-day generation
    run -- the worst possible moment, because nothing is checkpointed. It fires on ANY successful run,
    chi or not, and no retry can help because it is one indivisible allocation.

    WHY CAPPING RATHER THAN DISABLING. In chi mode the check is inapplicable: train_nn sets
    ``z_score_x="none"`` whenever the embedding owns its standardization, and sbi's own warning text
    ends "if you have already set z_score_x=False, this warning will still be displayed, but you can
    ignore it." But spontaneous and forced modes DO train under ``z_score_x="independent"``, where a
    collapse under z-scoring is a real finding worth hearing about. It is also a PROPORTION test with
    a 10 % tolerance (``duplicate_tolerance``), so a 200k-row sample answers it to well within its own
    resolution. STRIDED, not a prefix: rows are grouped by batch and each batch shares one
    (t_scale, T) stratum, so ``x[:200_000]`` would sample ~98 of 5000 strata while ``x[::n]`` spans
    all of them.

    PATCHED BY REBINDING IN EVERY MODULE THAT HOLDS THE NAME, not just in sbiutils. npe_base does
    ``from sbi.utils import ... warn_if_zscoring_changes_data``, so its call resolves against
    npe_base's OWN globals -- patching sbi.utils.sbiutils alone changes nothing. Rather than hard-code
    that module path (sbi has already moved it once: sbi.inference.snpe -> sbi.inference.trainers.npe)
    this finds every module whose attribute IS the original function object. A few ms, once per round,
    and it cannot rot.

    SCOPED AND RESTORING, including on an exception: the patch is live only across append_simulations,
    so nothing else in the process -- another panel, a script that imported sbi, a later round -- ever
    sees a monkeypatched sbi.
    """
    import sbi.utils.sbiutils as _sbiutils
    original = _sbiutils.warn_if_zscoring_changes_data

    def _capped(x, duplicate_tolerance: float = 0.1):
        stride = max(1, math.ceil(x.shape[0] / max_rows))
        # .contiguous() explicitly: unique would materialise the strided view anyway, and doing it
        # here keeps the copy's size obvious (<= max_rows x n_features) rather than implicit.
        return original(x[::stride].contiguous(), duplicate_tolerance)

    targets = []
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "warn_if_zscoring_changes_data", None) is original:
                targets.append(mod)
        except Exception:            # noqa: BLE001 -- a lazy module's __getattr__ must not break this
            continue
    try:
        for mod in targets:
            mod.warn_if_zscoring_changes_data = _capped
        yield
    finally:
        for mod in targets:
            mod.warn_if_zscoring_changes_data = original



def train_nn(training_params: dict, model: str, prior: torch.distributions.Distribution, embedding_net: torch.nn.Module,
             forcing_prior: torch.distributions.Distribution, nd_dim: int, forcing_idx: dict, rescale_idx: dict,
             x_obs: torch.Tensor = None, theta_obs: torch.Tensor = None, num_rounds: int = 1, return_diagnostics: bool = False, theta_transform: Transform | None = None,
             fixed_dict: dict = None,
             hidden_features: int = 50, num_transforms: int = 5, num_bins: int = 10,
             learning_rate: float = 5e-4, stop_after_epochs: int = 20, max_num_epochs: int = 2_147_483_647,
             show_train_summary: bool = False,
             batch_size: int = 128, device: torch.device = torch.device('cpu')) -> DirectPosterior | tuple[DirectPosterior, dict]:
    """
    Generate training data and fit the NPE flow (SNPE when ``num_rounds > 1``).

    :param training_params: the dict ``gen_training_data`` is driven from -- see that signature for
        each key's meaning; the .get() defaults below are the compatibility story for older callers.
    :param model: sbi density-estimator name (e.g. "nsf"); NOT the simulation model, which rides in
        ``training_params["model"]``.
    :param embedding_net: the conditioning net; ``owns_standardization`` decides z_score_x below.
    :param x_obs: observed data, required for SNPE (``num_rounds > 1``).
    :param theta_obs: ground-truth parameters, required only for ``return_diagnostics``.
    :param hidden_features: hidden units per flow transform; ``num_transforms`` the coupling-layer
        count; ``num_bins`` the spline bins (NSF only); ``learning_rate`` /
        ``stop_after_epochs`` / ``max_num_epochs`` / ``batch_size`` sbi's fit-loop knobs;
        ``show_train_summary`` prints sbi's per-epoch loss table.
    :return: the trained ``DirectPosterior`` -- ``(posterior, diagnostics)`` when
        ``return_diagnostics``.
    """
    if num_rounds > 1 and x_obs is None:
        raise ValueError("x_obs must be specified for SNPE algorithm")
    if num_rounds > 1 and training_params.get("checkpoint") is not None:
        # Refused loudly rather than half-supported. Rounds >= 2 sample from a PROPOSAL -- a trained
        # DirectPosterior -- whose identity a checkpoint would have to capture and re-validate, which
        # is a separate problem from the one C-11 solves. TRAINING_NUM_ROUNDS is 1 (amortized NPE), so
        # this costs nothing today and closes the hole rather than leaving it to be discovered.
        raise ValueError(
            f"Training-data checkpointing is not supported for SNPE (num_rounds={num_rounds}); the "
            f"per-round proposal is not part of the checkpoint's identity. Use num_rounds=1, or "
            f"drop training_params['checkpoint'].")

    # sbi's default z_score_x="independent" fits a PER-COLUMN affine over the conditioning vector.
    # Under the chi SET layout that is permutation-BREAKING (two orderings of one probe set would be
    # scaled differently, destroying the encoder's whole guarantee), and the near-constant mask column
    # becomes a ~1e7 amplifier under sbi's 1e-7 min-std clamp. The chi net standardizes itself instead
    # -- per channel, over live probes only. Derived from the net, never a separate argument, so the
    # standardizer and the encoder cannot be configured apart.
    _owns = getattr(embedding_net, "owns_standardization", False)
    neural_posterior = posterior_nn(model=model, embedding_net=embedding_net,
                                    z_score_x=("none" if _owns else "independent"),
                                    hidden_features=hidden_features, num_transforms=num_transforms,
                                    num_bins=num_bins)
    infer = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    proposal = None # set up initial proposal distribution
    posterior = None

    # diagnostics storage
    diagnostics = {
        "log_prob_true": [],
        "posterior_means": [],
        "posterior_stds": [],
    }

    for _ in tqdm(range(num_rounds), desc=f"Training neural posterior", leave=False):
        # train the density estimator
        data, thetas = _pipeline.gen_training_data(
            training_params["model"], training_params["prior"], forcing_prior, training_params["t"],
            training_params["run_size"], training_params["num_runs"],
            training_params["steady_idx"], training_params["dt_nd_min"],
            nd_dim, forcing_idx, rescale_idx,
            dt_exp=training_params["dt_exp"], t_min_exp=training_params["t_min_exp"],
            t_max_exp=training_params["t_max_exp"], t_scale_bounds=training_params["t_scale_bounds"],
            proposal=proposal,
            theta_transform=theta_transform,
            fixed_dict=fixed_dict,
            state_dep_drift=training_params.get("state_dep_drift", False),
            spontaneous_only=training_params.get("spontaneous_only", False),
            chi_mode=training_params.get("chi_mode", False),
            chi_f0=training_params.get("chi_f0", None),
            chi_freq_bounds=training_params.get("chi_freq_bounds", None),
            chi_k_pad=training_params.get("chi_k_pad", None),
            chi_k_fixed=training_params.get("chi_k_fixed", None),
            chi_max_cycles=training_params.get("chi_max_cycles", None),
            n_vars=training_params.get("n_vars", None),
            checkpoint=training_params.get("checkpoint", None),
            nd_idx=training_params.get("nd_idx", None),
            k_b_cell=training_params.get("k_b_cell", None),
            dtype=training_params["dtype"], device=training_params["device"],
        )

        # Filter the data -- and the TARGETS. thetas is the LATENT target (a logit); its row can in
        # principle go non-finite while the corresponding data row stays perfectly finite, and
        # filtering thetas only BY data would then feed the flow a +-inf target and NaN the loss for
        # the whole round with no diagnostic. On torch 2.9 the box round-trip cannot actually produce
        # one (SigmoidTransform._inverse clamps internally -- see gen_training_data), so this should
        # never fire; it exists so that if the transform stack ever changes, the failure arrives as a
        # message naming the offending columns instead of as a silently poisoned multi-hour run.
        nan_mask = torch.isfinite(data).all(dim=1)
        theta_finite_mask = torch.isfinite(thetas).all(dim=1)
        # WINSORISATION REPLACES THE OLD `abs(data) < 1e15` ROW FILTER, and
        # the change of instrument is the point. A row filter answers one bad channel by discarding
        # that row's other 113 good values, and at a 1e15 threshold it caught 10 rows in 10.24M while
        # A1_mean still reached -1.7e29 -- three decades of outlier sat under the threshold, and that
        # is what dragged A1_mean's fitted std to 4.19e11 and made the channel invisible to the flow.
        # Clipping each COLUMN at its own 0.1/99.9 percentile removes the leverage without removing
        # any row, and it protects sbi's own z-scoring on the non-chi paths too, which is why it is
        # not conditioned on who owns the standardisation.
        n_sum = getattr(embedding_net, "input_dim", None)
        if n_sum is None:
            # An embedding net that does not declare its summary width (a stub, a hand-built net in a
            # test) gets the pre-2026-08-26 behaviour exactly: there is no safe column split to clip
            # on, and guessing one could clip the chi block. See the block comment on
            # winsorize_summary_block.
            valid_idx = nan_mask & (torch.abs(data) < 1e15).all(dim=1) & theta_finite_mask
        else:
            valid_idx = nan_mask & theta_finite_mask
        n_bad_theta = int((~theta_finite_mask).sum())
        if n_bad_theta:
            bad_cols = torch.nonzero(~torch.isfinite(thetas).all(dim=0)).flatten().tolist()
            warnings.warn(
                f"train_nn: dropped {n_bad_theta}/{thetas.shape[0]} training rows with non-finite "
                f"LATENT targets (columns {bad_cols}). The box round-trip is supposed to make this "
                f"impossible -- treat it as a bug in the transform stack, not as expected attrition.",
                stacklevel=2,
            )
        # Only pay for the gather when it actually drops something. `data[valid_idx]` is a boolean
        # gather: it allocates a SECOND full-size tensor while the first is still live, which at the
        # production shape is another 4.35 GiB and reinstates exactly the 8.7 GiB host peak
        # gen_training_data's preallocated accumulators were introduced to remove. The mask is
        # all-true in practice (the box round-trip cannot produce a non-finite latent on torch 2.9;
        # SigmoidTransform._inverse clamps internally), so this is behaviour-identical and is what
        # makes that preallocation pay.
        if not bool(valid_idx.all()):
            thetas = thetas[valid_idx]
            data = data[valid_idx]

        if n_sum is not None:
            data = _pipeline.winsorize_summary_block(data, n_sum)

        # Fit OUR standardizers, once, on the post-filter data. posterior_nn was already called (it is
        # constructed before any data exists), so this has to happen here -- before append_simulations,
        # because build_nsf/get_numel run the net inside infer.train(). Unfitted, 41 raw statistics
        # spanning log-variances, nm-scale means and PLVs would reach the flow unscaled and the only
        # symptom would be a worse loss curve, days later; the encoder raises rather than allow it.
        if _owns and not embedding_net.standardization_fitted:
            embedding_net.fit_standardization(data)
            assert embedding_net.standardization_fitted, "chi standardization silently did not fit"

        # data_device="cpu": sbi otherwise defaults it to the TRAINING device (npe_base.py:174-175)
        # and moves the whole conditioning tensor onto the GPU -- 10.24M x 114 float32 = 4.35 GiB
        # resident, an 8.7 GiB transient for the `x = x[is_valid_x]` filter, two (10.24M,) bool masks
        # from handle_invalid_x, and later a ~3.9 GiB gather when it slices the training split. All of
        # that on the same card that has to hold the flow, and all of it AFTER a multi-day generation
        # run that is not checkpointed. The data is ALREADY on the CPU here (gen_training_data appends
        # .cpu() tensors), so this also silences validate_theta_and_x's "Moving x to the data_device"
        # warning, which describes a 4.35 GiB copy nobody asked for.
        #
        # Training does not care: sbi's loop moves each minibatch to the training device itself, so a
        # CPU-resident dataset costs one host-to-device copy of (training_batch_size, 114) float32 --
        # tens of KB -- against a flow forward+backward that dominates it by orders of magnitude. It
        # hands 4.35 GiB of VRAM back to the flow, which is a straight win.
        with _capped_zscore_check():
            infer.append_simulations(thetas, data, proposal=proposal, data_device="cpu")
        density_estimator = infer.train(
            training_batch_size=batch_size, learning_rate=learning_rate,
            stop_after_epochs=stop_after_epochs, max_num_epochs=max_num_epochs,
            show_train_summary=show_train_summary,
        )
        posterior = infer.build_posterior(density_estimator)
        assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"

        # compute diagnostics after each round
        if return_diagnostics and x_obs is not None:
            x_obs_device = x_obs.to(device)

            # log probability of ground truth
            if theta_obs is not None:
                theta_true_device = theta_obs.to(device)
                if theta_true_device.dim() == 1:
                    theta_true_device = theta_true_device.unsqueeze(0)
                log_prob = posterior.log_prob(theta_true_device, x=x_obs_device).item()
                diagnostics["log_prob_true"].append(log_prob)

            # posterior mean and std from samples
            samples = posterior.sample((10000,), x=x_obs_device)
            diagnostics["posterior_means"].append(samples.mean(dim=0).cpu())
            diagnostics["posterior_stds"].append(samples.std(dim=0).cpu())

        # need to now check if num_runs > 1: if so, then that is equivalent to SNPE, and if not, that is equivalent to NPE
        if num_rounds > 1:
            assert x_obs is not None, "x_obs must be specified for SNPE algorithm"
            proposal = posterior.set_default_x(x_obs.to(device)) # if SNPE, the user has to specify x_obs

    assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"

    # Capture the per-epoch train/validation loss curve from the sbi trainer so the
    # convergence ("under-fit vs converged") check is reproducible. sbi keeps these in
    # infer._summary; they are otherwise discarded when this function returns.
    if return_diagnostics:
        summary = getattr(infer, "_summary", {}) or {}
        diagnostics["training_loss"] = list(summary.get("training_loss", []))
        diagnostics["validation_loss"] = list(summary.get("validation_loss", []))
        best_val = summary.get("best_validation_loss", [])
        diagnostics["best_validation_loss"] = best_val[-1] if len(best_val) else None
        epochs = summary.get("epochs_trained", [])
        diagnostics["epochs_trained"] = epochs[-1] if len(epochs) else None
        diagnostics["stop_after_epochs"] = stop_after_epochs
        return posterior, diagnostics
    return posterior
