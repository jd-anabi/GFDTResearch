"""Stability-screened prior for user-defined models (SBI v2).

The three built-in priors (nadrowski/hopf/bp) are ~95% identical: only the concrete Simulator class,
the initial-condition shape, and the force-channel count differ. This ONE generic prior parametrizes
those over a registry.ModelSpec, reusing ``prior.Prior.construct_prior`` (the model-agnostic Sobol
stability screen -> HDBSCAN cluster -> GMM fit -> box bijection) unchanged.

Differences from the built-ins, by design:
  * The Simulator comes from ``registry.make_user_simulator`` (a UserSimulator around the compiled
    torch model), not a hardcoded class.
  * Initial conditions are the model's DECLARED inits (broadcast across the batch), not
    ``randint(0, 10)`` -- a nondimensional user model can live on a unit scale that random 0..9 inits
    would blow straight past the divergence guard. Tradeoff: every batch member screens from the same
    init, so there is less init diversity; acceptable because stability is a parameter property and
    the transient washes inits out.
  * Forcing is a zero ``(batch, n_vars, T)`` tensor -- SBI-eligible user models have no drive
    (spontaneous dynamics; see registry.is_sbi_user_model).
  * The dead ``BoxUniform`` list and the (never-taken) ``steady`` branch from the built-ins are dropped.

``_local_map`` overrides the base's @staticmethod as an INSTANCE method so it can read ``self.spec``;
any override satisfies the abstractmethod, and ``construct_prior`` calls it as ``self._local_map(...)``.
"""
from collections import deque

import torch
from tqdm import tqdm

from core import registry
from core.SBI.Priors import prior


def declared_inits(spec) -> torch.Tensor:
    """The user model's declared initial conditions as a (1, n_vars) tensor (variable order)."""
    return torch.tensor([[float(v["init"]) for v in spec.variables]], dtype=torch.float32)


class UserPrior(prior.Prior):
    def __init__(self, spec, dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        super().__init__(dtype, device)
        self.spec = spec

    def _global_map(self, t: torch.Tensor, n_params: int, prior_bounds: list, segs: int,
                    batch_size: int, num_iterations: int, steady: bool, state_dep_drift: bool) -> list:
        t = t.to(dtype=self.dtype, device=self.device)
        if batch_size % num_iterations != 0:
            raise ValueError('batch_size must be divisible by num_iterations')
        curr_batch_size = batch_size // num_iterations

        lows = torch.tensor([b[0] for b in prior_bounds], dtype=self.dtype, device=self.device)
        highs = torch.tensor([b[1] for b in prior_bounds], dtype=self.dtype, device=self.device)
        engine = torch.quasirandom.SobolEngine(dimension=len(prior_bounds), scramble=True)
        unit_samples = engine.draw(batch_size).to(dtype=self.dtype, device=self.device)
        thetas = lows + unit_samples * (highs - lows)

        inits_tensor = declared_inits(self.spec).to(dtype=self.dtype, device=self.device).expand(
            curr_batch_size, -1)
        force = torch.zeros((curr_batch_size, self.spec.n_vars, t.shape[0]),
                            dtype=self.dtype, device=self.device)

        stable_params, num_added = [], 0
        bar = tqdm(total=num_iterations, leave=False,
                   desc=f"Added {num_added} sets during global sweep")
        with torch.no_grad():
            for i in range(num_iterations):
                curr_thetas = thetas[i * curr_batch_size:(i + 1) * curr_batch_size]
                sim = registry.make_user_simulator(
                    self.spec, curr_thetas, force, inits_tensor, t,
                    segs=segs, batch_size=curr_batch_size, device=self.device)
                x = sim.simulate(state_dep_drift=state_dep_drift)[0, 0, :, :]   # (curr_batch, len(t))
                is_valid = torch.isfinite(x).all(dim=1)
                valid_params = curr_thetas[is_valid]
                num_added += int(valid_params.shape[0])
                bar.update()
                bar.set_description(f"Added {num_added} sets during global sweep")
                del x
                stable_params.extend(valid_params.detach().cpu().tolist())
        bar.close()
        return stable_params

    def _local_map(self, t: torch.Tensor, stable_params: list, batch_size: int, n_params: int,
                   n_max: int, step: float, segs: int, steady: bool, state_dep_drift: bool) -> list:
        # self.sweep_device, not a hardcoded CPU. This method used to be a @staticmethod pinned to
        # torch.device('cpu'); measured 6.32 s per iteration there against 0.357 s on CUDA (17.7x),
        # and the flood-fill is the dominant cost of a prior build. construct_prior resolves the
        # device once and falls back to the CPU when there is no accelerator.
        dtype, device = self.dtype, self.sweep_device
        t = t.to(dtype=dtype, device=device)

        queue = deque(stable_params)
        accepted_params = set(tuple(p) for p in stable_params)

        inits = declared_inits(self.spec).to(dtype=dtype, device=device).expand(batch_size, -1)
        force = torch.zeros((batch_size, self.spec.n_vars, t.shape[0]), dtype=dtype, device=device)

        num_added = 0
        bar = tqdm(total=batch_size, leave=False,
                   desc=f"Local sweep: {len(accepted_params)} accepted, {len(queue)} to check")
        with torch.no_grad():
            while len(queue) != 0 and len(accepted_params) <= n_max:
                thetas = (torch.tensor(queue.popleft(), dtype=dtype, device=device)
                          + torch.randn((batch_size, n_params), dtype=dtype, device=device) * step)
                sim = registry.make_user_simulator(
                    self.spec, thetas, force, inits, t,
                    segs=segs, batch_size=batch_size, device=device)
                x = sim.simulate(state_dep_drift=state_dep_drift)[0, 0, :, :]
                is_valid = torch.isfinite(x).all(dim=1)
                # ONE host transfer per iteration, not 2048. Indexing `is_valid[i]` and calling
                # `thetas[i].tolist()` per row forces a device-to-host SYNC per row -- free on the
                # CPU, and on CUDA it would have handed back most of the 17.7x the device move buys.
                # Gathering the valid rows first makes the transfer a single contiguous copy.
                for row in thetas[is_valid].detach().cpu().tolist():
                    stable_point = tuple(row)
                    if stable_point not in accepted_params:
                        accepted_params.add(stable_point)
                        queue.append(stable_point)
                        num_added += 1
                bar.update(bar.total or 0)
                bar.reset()
                bar.set_description(
                    f"Local sweep: {len(accepted_params)} accepted, {len(queue)} to check")
                del x
        bar.close()
        return list(accepted_params)
