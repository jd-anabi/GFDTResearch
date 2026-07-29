from collections import deque

import numpy as np
import torch
from sbi import utils
from tqdm import tqdm

from core.Simulator import hopf_simulator
from core.SBI.Priors import prior

class HopfPrior(prior.Prior):
    def __init__(self, dtype: torch.dtype = torch.float32,device: torch.device = torch.device('cpu')):
        super().__init__(dtype, device)

    # --- PRIVATE METHODS --- #
    def _global_map(self, t: torch.Tensor, n_params: int, prior_bounds: list[tuple], segs: int,
                    batch_size: int, num_iterations: int, steady: bool, state_dep_drift: bool) -> list:
        t = t.to(dtype=self.dtype, device=self.device)
        if batch_size % num_iterations != 0:
            raise ValueError('batch_size must be divisible by num_iterations')
        curr_batch_size = batch_size // num_iterations

        priors = []
        for curr_bounds in prior_bounds:
            curr_prior = utils.BoxUniform(low=torch.ones(1) * curr_bounds[0], high=torch.ones(1) * curr_bounds[1])
            priors.append(curr_prior)

        # Sobol sampling scaled to prior bounds
        lows = torch.tensor([b[0] for b in prior_bounds], dtype=self.dtype, device=self.device)
        highs = torch.tensor([b[1] for b in prior_bounds], dtype=self.dtype, device=self.device)
        engine = torch.quasirandom.SobolEngine(dimension=len(prior_bounds), scramble=True)
        unit_samples = engine.draw(batch_size).to(dtype=self.dtype, device=self.device)
        thetas = lows + unit_samples * (highs - lows)

        inits = np.random.randint(0, 10, size=(curr_batch_size, 2))
        inits_tensor = torch.tensor(inits, dtype=self.dtype, device=self.device)
        # Dual-channel forcing for ND Hopf: shape (batch, 2, T). Channel 0 -> x, channel 1 -> y.
        force = torch.zeros((curr_batch_size, 2, t.shape[0]), dtype=self.dtype, device=self.device)
        stable_params = []

        num_added = 0
        added_params_progress_bar = tqdm(total=num_iterations, desc=f"Added {num_added} sets to accepted parameters during global sweep", leave=False)
        with torch.no_grad():
            for i in range(num_iterations):
                curr_thetas = thetas[i*curr_batch_size:(i+1)*curr_batch_size]
                sim = hopf_simulator.HopfSimulator(curr_thetas, force, inits_tensor, t, segs=segs, batch_size=curr_batch_size, device=self.device)
                x = sim.simulate(state_dep_drift=state_dep_drift)[0, 0, :, :] # shape: (curr_batch_size, len(t))
                is_valid = torch.isfinite(x).all(dim=1)
                valid_params = curr_thetas[is_valid]
                if valid_params.shape[0] > 0:
                    num_added += valid_params.shape[0]
                added_params_progress_bar.update()
                added_params_progress_bar.set_description(f"Added {num_added} sets to accepted parameters during global sweep")
                del x
                stable_params.extend(valid_params.detach().cpu().tolist())
        added_params_progress_bar.close()
        return stable_params

    @staticmethod
    def _local_map(t: torch.Tensor, stable_params: list, batch_size: int,
                   n_params: int, n_max: int, step: float, segs: int, steady: bool, state_dep_drift: bool) -> list:
        # cpu variables
        dtype = torch.float32
        device = torch.device('cpu')
        t = t.to(dtype=dtype, device=device)

        # algorithm variables
        queue = deque(stable_params)
        accepted_params = set([tuple(p) for p in stable_params])

        # check if steady-state
        if steady:
            n_params -= 1

        # SDE variable
        inits = np.random.randint(0, 10, size=(batch_size, 2))  # size: (BATCH_SIZE, 2)
        inits = torch.tensor(inits, dtype=dtype, device=device)
        # Dual-channel forcing for ND Hopf: shape (batch, 2, T). Channel 0 -> x, channel 1 -> y.
        force = torch.zeros((batch_size, 2, t.shape[0]), dtype=dtype, device=device)

        # begin algorithm
        num_added = 0
        added_params_progress_bar = tqdm(total=batch_size, desc=f"Added {num_added} sets to accepted parameters during local sweep. Total parameter sets: {len(accepted_params)}. Number of parameter sets to check: {len(queue)}", leave=False)
        with torch.no_grad():
            while len(queue) != 0 and len(accepted_params) <= n_max:
                thetas = torch.tensor(queue.popleft(), dtype=dtype, device=device) + torch.randn((batch_size, n_params), dtype=dtype, device=device) * step
                sim = hopf_simulator.HopfSimulator(thetas, force, inits, t, segs=segs, batch_size=batch_size, device=device)
                x = sim.simulate(state_dep_drift=state_dep_drift)[0, 0, :, :] # shape: (batch_size, len(t))
                is_valid = torch.isfinite(x).all(dim=1)
                for i in range(batch_size):
                    if is_valid[i]:
                        stable_point = tuple(thetas[i].tolist())
                        if not stable_point in accepted_params:
                            accepted_params.add(stable_point)
                            queue.append(stable_point)
                            num_added += 1
                    added_params_progress_bar.update()
                added_params_progress_bar.reset()
                added_params_progress_bar.set_description(f"Added {num_added} sets to accepted parameters during local sweep. Total parameter sets: {len(accepted_params)}. Number of parameter sets to check: {len(queue)}")
                del x
        added_params_progress_bar.close()

        # SORTED, not list(): see nadrowski_prior for the rationale -- set iteration order over float
        # tuples is unspecified, and this feeds the clustering + GMM fit in prior.py.
        return sorted(accepted_params)