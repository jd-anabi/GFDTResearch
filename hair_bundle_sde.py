import numpy as np
import torch

from hair_bundle import HairBundle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

class HairBundleSDE(torch.nn.Module):
    def __init__(self, params: list, force_params: list, p_t_steady_state: bool, noise_type: str, sde_type: str):
        super().__init__()
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.hb = HairBundle(*params, *force_params, p_t_steady_state)

    def f(self, t, x) -> torch.Tensor:
        force = self.hb.sin_driving_force(t.item())
        sys_vars = np.zeros(x.shape, dtype=float)
        dx = []
        for i in range(x.shape[0]):
            var_index = 0
            for var in torch.split(x[i], 1, dim=-1):
                sys_vars[i][var_index] = var.item()
                var_index = var_index + 1
            dx.append(self.hb.sde_sym_lambda_func(*(sys_vars[i])))
            dx[i][0] = dx[i][0] + force
        return torch.tensor(dx, dtype=DTYPE, device=DEVICE)

    def g(self, t, x) -> torch.Tensor:
        dsigma = [self.hb.hb_noise, self.hb.a_noise, 0, 0, 0]
        dsigma = dsigma[0:x.shape[-1]]
        return torch.tile(torch.tensor(dsigma, dtype=DTYPE, device=DEVICE), (x.shape[0], 1))