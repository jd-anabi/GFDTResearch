import torch

from core.Models import bp_model, bp_model_steady
from core.Simulator import simulator

class BPSimulator(simulator.Simulator):
    def __init__(self, params: torch.Tensor, force: torch.Tensor, inits: torch.Tensor, t: torch.Tensor,
                 freqs_per_batch: int = 1, segs: int = 1, batch_size: int = 1, device: torch.device = torch.device('cpu'),
                 use_compile: bool | None = None):
        super().__init__(params, force, inits, t, freqs_per_batch, segs, batch_size, device, use_compile=use_compile)
        self._set_up_model()

    # --- PRIVATE METHOD --- #
    def _set_up_model(self):
        try:
            if self._params.shape[-1] != 17:
                self.inits = self.inits[:, :4]
                self.sde = bp_model_steady.BPModelSteady(*torch.unbind(self._params, dim=1), self._force, batch_size=self._batch_size, device=self._device, dtype=self._dtype)
            else:
                if torch.all(self._params[:, 3]):
                    self.sde = bp_model.BPModel(*torch.unbind(self._params, dim=1), self._force, batch_size=self._batch_size, device=self._device, dtype=self._dtype)
                else:
                    raise ValueError("Can't not mix and match steady and non-steady models; finite time constant in the parameter batch must all be zero or all non-zero")
        # NOT BaseException: a cooperative streams.WorkerCancelled must sail through (see
        # simulator.SimulationError for why this is a raise and no longer a print + exit()).
        except Exception as e:
            raise simulator.SimulationError(
                f"BP model construction failed (batch={self._batch_size}, "
                f"device={self._device}): {type(e).__name__}: {e}") from e