"""Generic Simulator for user-defined models (core/Models/user_model.py).

Mirrors the concrete subclasses' positional ``Model(*torch.unbind(params, dim=1), force, ...)``
construction, raising RuntimeError on failure. This was the pattern the built-in subclasses were
later brought in line with -- they used to ``print + exit()`` and now raise
``simulator.SimulationError`` (itself a RuntimeError). Unlike them it does NOT repeat the redundant
second ``_set_up_model()`` call -- the base ``Simulator.__init__`` already runs it once.
"""
import torch

from core.Models.user_model import CompiledUserModel, UserModel
from core.Simulator import simulator


class UserSimulator(simulator.Simulator):
    def __init__(self, compiled: CompiledUserModel, params: torch.Tensor, force: torch.Tensor,
                 inits: torch.Tensor, t: torch.Tensor, freqs_per_batch: int = 1, segs: int = 1,
                 batch_size: int = 1, device: torch.device = torch.device('cpu'),
                 use_compile: bool | None = None):
        # Set BEFORE super().__init__: the base constructor calls _set_up_model() as its last step.
        self._compiled = compiled
        super().__init__(params, force, inits, t, freqs_per_batch=freqs_per_batch, segs=segs,
                         batch_size=batch_size, device=device, use_compile=use_compile)

    def _set_up_model(self):
        try:
            self.sde = UserModel(self._compiled, torch.unbind(self._params, dim=1), self._force,
                                 batch_size=self._batch_size, device=self._device, dtype=self._dtype)
        except Exception as e:                                 # noqa: BLE001 -- surface, never exit()
            raise RuntimeError(f"User model construction failed: {e}") from e
