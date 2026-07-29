import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from core import config
from core.Helpers import helpers
from core.Solvers import sdeint


class SimulationError(RuntimeError):
    """A solver failure inside ``Simulator.simulate``, carrying the geometry that produced it.

    Always raised ``from`` the underlying error, so the original traceback survives as ``__cause__``
    and Python prints both. Callers that need to branch on the cause can test it directly, e.g.
    ``isinstance(err.__cause__, torch.OutOfMemoryError)``.

    This used to be ``print(...); exit()``, which hard-killed the interpreter: a CUDA OOM arrived with
    no traceback (so the failing call site was unknowable), and any caller -- the test runners, the
    GUI's QThreadPool worker, the scripts -- died mid-run with nothing reported. Note ``exit()`` is the
    ``site`` builtin and is not guaranteed to exist at all under ``python -O`` or an embedded
    interpreter.
    """


class Simulator(ABC):
    def __init__(self, params: torch.Tensor, force: torch.Tensor, inits: torch.Tensor, t: torch.Tensor,
                 freqs_per_batch: int = 1, segs: int = 1, batch_size: int = 1, device: torch.device = torch.device('cpu'),
                 use_compile: bool | None = None):
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

        # Auto-enable the torch.compile path on CUDA when the model exposes
        # `compiled_step`. Explicitly pass False to force the eager loop.
        self._use_compile = use_compile

        # check if we are using the steady-state solution (all zeros for the 4th parameter)
        self._set_up_model()

    # --- PUBLIC METHODS --- #
    def simulate(self, state_dep_drift: bool = False) -> torch.Tensor:
        """
        Simulates the model with the given constructor parameters
        :return: simulated solution with shape (N, FPB, B / FPB, T)
        """
        ensemble_size = self._batch_size // self.freqs_per_batch
        time_seg_ids = helpers.get_even_ids(self.t.shape[0], self.segs + 1)

        n_vars = self.inits.shape[-1]
        curr_inits = self.inits
        sol = torch.zeros((n_vars, self._batch_size, self.t.shape[0]), dtype=self.t.dtype, device=self.t.device)

        # The SDE model indexes force with the solver's local step index (0..n_seg-1),
        # not the absolute step across the full simulation. When segs > 1 with
        # non-constant forcing, we slice force to the current segment so the local
        # index lookup picks up the right values; restore the full reference after.
        full_force = self._force
        # `disable` is passed only under the GUI (config.QUIET_SEGMENT_BAR). It must be a conditional
        # splat, not `disable=config.QUIET_SEGMENT_BAR`: tqdm.__init__ is @envwrap("TQDM_")-decorated
        # (tqdm/std.py:951) and a call kwarg outranks the environment, so an explicit False would shadow
        # a TQDM_DISABLE override that reaches this call today. Omitting it keeps the CLI byte-identical.
        for tid in tqdm(range(len(time_seg_ids) - 1), desc="Running time segments", leave=False,
                        **({"disable": True} if config.QUIET_SEGMENT_BAR else {})):
            curr_time = self.t[time_seg_ids[tid]:time_seg_ids[tid + 1]]
            self.sde.force = full_force[:, :, time_seg_ids[tid]:time_seg_ids[tid + 1]]
            results = self.__sols(curr_time, curr_inits, state_dep_drift)  # shape: (len(curr_time), BATCH_SIZE, number of variables)

            # update initial conditions. The .clone() is load-bearing for MEMORY, not correctness:
            # results[-1, :, :] is a VIEW into this segment's (len(curr_time), batch, n_vars) solver
            # buffer, so carrying it into the next iteration would pin that whole buffer -- 2.5 GB at
            # CHUNK_LEN=100k x batch=2048 x 3 vars -- alive for the whole of the next segment's
            # integration. Copying one (batch, n_vars) row costs ~24 KB and lets the buffer go.
            curr_inits = results[-1, :, :].clone()

            # extract position data
            #
            # SEAM DUPLICATION -- know this before building anything phase-sensitive on `sol`.
            # sdeint writes xs[0] = x0 and then integrates range(n-1) steps, and x0 here is the
            # PREVIOUS segment's final state. So sol[..., ids[k]] == sol[..., ids[k]-1] at every
            # segment boundary: the trajectory advances len(t) - segs steps, not len(t) - 1, and `t`
            # and `sol` are NOT exactly co-indexed. Negligible for the spectral/ACF features here at
            # segs <= 3, but a feature that reads instantaneous phase or a finite difference ACROSS a
            # seam would see a zero-length step. Note this also makes Simulator.simulate and a direct
            # sdeint call disagree on effective length for identical inputs.
            sol[:, :, time_seg_ids[tid]:time_seg_ids[tid + 1]] = torch.transpose(results, 0, 2)  # shape: (number of variables, BATCH_SIZE, len(curr_time))
            del results          # release the segment buffer BEFORE the next __sols allocates its own
        self.sde.force = full_force
        sol = sol.reshape(n_vars, self.freqs_per_batch, ensemble_size, self.t.shape[0])  # shape: (number of variables, frequencies per batch, ensemble size, length of time series)
        return sol

    # --- GETTERS AND SETTERS --- #
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        self._set_up_model()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self._dtype = dtype
        self._set_up_model()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size
        self._set_up_model()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: torch.Tensor):
        self._params = params
        self._set_up_model()

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, force: torch.Tensor):
        self._force = force
        self._set_up_model()

    # --- PRIVATE METHODS --- #
    def __sols(self, t: torch.Tensor, inits: torch.Tensor, state_dep_drift: bool) -> torch.Tensor:
        """
        Returns sde solution for a hair bundle given a set of parameters and initial conditions
        :param t: time array
        :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
        """
        # time array
        n = t.shape[0]
        ts = (t[0], t[-1])

        # Solving a system of SDEs. Constructed per call ON PURPOSE, despite this running once per
        # time segment: Solver builds its three methods as closures in __init__, so resolving
        # sdeint.Solver at CALL time is the seam that lets a caller swap the solver out (see
        # tests/test_user_sbi.py::test_solver_failure_raises_instead_of_killing_the_process, which
        # patches the class to make every method raise). Hoisting this to a module-level singleton
        # saves ~3 closure allocations per ~10-second segment -- far too little to be worth losing
        # the seam over.
        solver = sdeint.Solver()

        # Pick eager vs compiled. Auto: CUDA + model exposes compiled_step.
        use_compile = self._use_compile
        if use_compile is None:
            use_compile = (self._device.type == "cuda"
                           and hasattr(self.sde, "compiled_step"))

        with torch.no_grad():
            try:
                if use_compile:
                    sol = solver.euler_compiled(self.sde, inits, ts, n, state_dep_drift=state_dep_drift)
                else:
                    sol = solver.euler(self.sde, inits, ts, n, state_dep_drift=state_dep_drift)
            # NOT BaseException: streams.WorkerCancelled derives from BaseException precisely so a
            # cooperative cancel sails through handlers like this one to reach Worker.run. Widening
            # here would turn every GUI cancel into a spurious simulation failure.
            except Exception as e:
                method = "euler_compiled" if use_compile else "euler"
                raise SimulationError(
                    f"{type(self.sde).__name__} {method} failed after {n} steps "
                    f"(batch={self._batch_size}, segs={self.segs}, device={self._device}, "
                    f"dtype={self._dtype}): {type(e).__name__}: {e}"
                ) from e
        return sol

    @abstractmethod
    def _set_up_model(self):
        self.sde = None