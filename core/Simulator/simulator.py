import torch
from tqdm import tqdm

from core.Helpers import gen_helpers as helpers
from core.Models import model, steady_model
from core.Solvers import sdeint

class Simulator(torch.nn.Module):
    def __init__(self, params: torch.Tensor, force: torch.Tensor, inits: torch.Tensor, t: torch.Tensor,
                 freqs_per_batch: int = 1, segs: int = 1, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        super().__init__()
        # device initialization
        self._device = device
        self._dtype = inits.dtype
        self._batch_size = batch_size

        # rest of the constructor
        self._params = params
        self._force = force
        self.inits = inits
        self.t = t
        self.freqs_per_batch = freqs_per_batch
        self.segs = segs

        # check if we are using the steady-state solution (all zeros for the 4th parameter)
        self.sde = None
        self.__set_up_model()

    # ------------- PUBLIC METHODS ------------- #
    def simulate(self) -> torch.Tensor:
        """
        Simulates the model with the given constructor parameters
        :return: simulated solution with shape (number of variables, frequencies per batch, ensemble size, length of time series)
        """
        ensemble_size = self.batch_size // self.freqs_per_batch
        time_seg_ids = helpers.get_even_ids(self.t.shape[0], self.segs + 1)

        n_vars = self.inits.shape[-1] if isinstance(self.sde, steady_model.HairBundleSDE) else self.inits.shape[-1] - 1
        curr_inits = self.inits
        sol = torch.zeros((n_vars, self.batch_size, self.t.shape[0]), dtype=self.t.dtype, device=self.t.device)
        overall_progress = tqdm(range(len(time_seg_ids) - 1), desc="Overall Progress")
        for tid in overall_progress:
            overall_progress.set_description(f"Running time segment {tid + 1}")
            curr_time = self.t[time_seg_ids[tid]:time_seg_ids[tid + 1]]
            results = self.__sols(curr_time, curr_inits)  # shape: (len(curr_time), BATCH_SIZE, number of variables)

            # update initial conditions
            curr_inits = results[-1, :, :]

            # extract position data
            sol[:, :, time_seg_ids[tid]:time_seg_ids[tid + 1]] = torch.transpose(results, 0, 2).to(device=sol.device)  # shape: (number of variables, BATCH_SIZE, len(curr_time))
        sol = sol.reshape(n_vars, self.freqs_per_batch, ensemble_size, self.t.shape[0])  # shape: (number of variables, frequencies per batch, ensemble size, length of time series)
        return sol

    # -------------------- GETTERS AND SETTERS -------------------- #
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        self.__set_up_model()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self._dtype = dtype
        self.__set_up_model()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size
        self.__set_up_model()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: torch.Tensor):
        self._params = params
        self.__set_up_model()

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, force: torch.Tensor):
        self._force = force
        self.__set_up_model()

    # ----------------- PRIVATE METHODS ----------------- #
    def __sols(self, t: torch.Tensor, inits: torch.Tensor, explicit: bool = True) -> torch.Tensor:
        """
        Returns sde solution for a hair bundle given a set of parameters and initial conditions
        :param t: time array
        :param explicit: whether to use the explicit Euler-Maruyama method
        :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
        """
        # time array
        n = t.shape[0]
        ts = (t[0], t[-1])

        # solving a system of SDEs
        solver = sdeint.Solver()
        sol = torch.zeros((n, self.batch_size, inits.shape[1]), dtype=self.t.dtype, device=self.t.device)
        with torch.no_grad():
            try:
                if explicit:
                    sol = solver.euler(self.sde, inits, ts, n).to(device=sol.device)  # only keep the last solution
                else:
                    sol = solver.implicit_euler(self.sde, inits, ts, n).to(device=sol.device)
            except (Warning, Exception) as e:
                print(f'Warning or Exception occurred: {e}')
                exit()
        return sol

    def __set_up_model(self):
        try:
            if not torch.any(self.params[:, 3]):
                self.inits = self.inits[:, :4]
                self.sde = steady_model.HairBundleSDE(*torch.unbind(self.params, dim=1), self.force, batch_size=self.batch_size, device=self._device, dtype=self._dtype)
            else:
                if torch.all(self.params[:, 3]):
                    self.sde = model.HairBundleSDE(*torch.unbind(self.params, dim=1), self.force, batch_size=self.batch_size, device=self._device, dtype=self._dtype)
                else:
                    raise ValueError("Can't not mix and match steady and non-steady models; finite time constant in the parameter batch must all be zero or all non-zero")
        except (Warning, Exception) as e:
            print(f"{e}")
            exit()