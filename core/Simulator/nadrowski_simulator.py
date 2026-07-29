import torch

from core.Models import nadrowski_model
from core.Simulator import simulator

class NadrowskiSimulator(simulator.Simulator):
    def __init__(self, params: torch.Tensor, force: torch.Tensor, inits: torch.Tensor, t: torch.Tensor,
                 freqs_per_batch: int = 1, segs: int = 1, batch_size: int = 1, device: torch.device = torch.device('cpu'),
                 use_compile: bool | None = None):
        super().__init__(params, force, inits, t, freqs_per_batch, segs, batch_size, device, use_compile=use_compile)
        self._set_up_model()

    # --- PRIVATE METHOD --- #
    def _set_up_model(self):
        try:
            self.sde = nadrowski_model.NadrowskiModel(*torch.unbind(self._params, dim=1), self._force, batch_size=self._batch_size, device=self._device, dtype=self._dtype)
        # NOT BaseException: a cooperative streams.WorkerCancelled must sail through (see
        # simulator.SimulationError for why this is a raise and no longer a print + exit()).
        except Exception as e:
            raise simulator.SimulationError(
                f"NadrowskiModel construction failed (batch={self._batch_size}, "
                f"device={self._device}): {type(e).__name__}: {e}") from e