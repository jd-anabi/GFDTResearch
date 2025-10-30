import torch

from core.Helpers import gen_helpers as helpers
from core.Models import model, steady_model
from core.Solvers import sdeint

class Simulator(torch.nn.Module):
    def __init__(self, params: torch.Tensor, t: torch.Tensor, inits: torch.Tensor, force: torch.Tensor,
                 freqs_per_batch: int = 1, segs: int = 1, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        super().__init__()
        # device initialization
        self._device = device
        self._dtype = inits.dtype
        self.batch_size = batch_size

        # rest of the constructor
        self.params = params.to(self._device, dtype=self._dtype)
        self.t = t.to(self._device, dtype=self._dtype)
        self.inits = inits.to(self._device, dtype=self._dtype)
        self.force = force.to(self._device, dtype=self._dtype)
        self.freqs_per_batch = freqs_per_batch
        self.segs = segs

    def simulate(self) -> torch.Tensor:
        """
        Simulates the model with the given constructor parameters
        :return: simulated solution with shape (number of variables, frequencies per batch, ensemble size, length of time series)
        """
        ensemble_size = self.batch_size // self.freqs_per_batch
        time_seg_ids = helpers.get_even_ids(self.t.shape[0], self.segs + 1)

        n_vars = self.inits.shape[-1] if self.params[0, 3] != 0 else self.inits.shape[-1] - 1
        curr_inits = self.inits
        sol = torch.zeros((n_vars, self.batch_size, self.t.shape[0]), dtype=self._dtype, device=self._device)
        for tid in range(len(time_seg_ids) - 1):
            print(f"Time segment {tid + 1}:")
            curr_time = self.t[time_seg_ids[tid]:time_seg_ids[tid + 1]]
            results = self.__sols(curr_time, curr_inits)  # shape: (len(curr_time), BATCH_SIZE, number of variables)

            # update initial conditions
            curr_inits = results[-1, :, :]

            # extract position data
            sol[:, :, time_seg_ids[tid]:time_seg_ids[tid + 1]] = torch.transpose(results, 0, 2)  # shape: (number of variables, BATCH_SIZE, len(curr_time))
        sol = sol.reshape(n_vars, self.freqs_per_batch, ensemble_size, self.t.shape[0])  # shape: (number of variables, frequencies per batch, ensemble size, length of time series)
        return sol

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        self.params = self.params.to(self.device)
        self.t = self.t.to(self.device)
        self.inits = self.inits.to(self.device)
        self.force = self.force.to(self.device)

    @property
    def dtype(self):
        return self._device

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self._dtype = dtype
        self.params = self.params.to(dtype=dtype)
        self.t = self.t.to(dtype=dtype)
        self.t = self.t.to(dtype=dtype)
        self.inits = self.inits.to(dtype=dtype)
        self.force = self.force.to(dtype=dtype)

    def __sols(self, t: torch.Tensor, inits: torch.Tensor, explicit: bool = True) -> torch.Tensor:
        """
        Returns sde solution for a hair bundle given a set of parameters and initial conditions
        :param t: time array
        :param explicit: whether to use the explicit Euler-Maruyama method
        :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
        """
        # check if we are using the steady-state solution
        if self.params[0, 3] == 0:
            inits = inits[:, :4]
            sde = steady_model.HairBundleSDE(*torch.unbind(self.params, dim=1), self.force, batch_size=self.batch_size, device=self._device, dtype=self._dtype)
        else:
            sde = model.HairBundleSDE(*torch.unbind(self.params, dim=1), self.force, batch_size=self.batch_size, device=self._device, dtype=self._dtype)
        # time array
        n = t.shape[0]
        ts = (t[0], t[-1])

        # solving a system of SDEs
        solver = sdeint.Solver()
        sol = torch.zeros((n, self.batch_size, inits.shape[1]), dtype=self._dtype, device=self._device)
        with torch.no_grad():
            try:
                if explicit:
                    sol = solver.euler(sde, inits, ts, n)  # only keep the last solution
                else:
                    sol = solver.implicit_euler(sde, inits, ts, n)
            except (Warning, Exception) as e:
                print(f'Warning or Exception occurred: {e}')
                exit()
        return sol