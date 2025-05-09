import torch
import torchsde

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
        sys_vars = []
        for var in torch.split(x, 1, dim=-1):
            sys_vars.append(var.item())
        print(sys_vars)
        dx_sys = self.hb.sde_sym_lambda_func(*sys_vars)
        dx_sys[0] = dx_sys[0] + force
        torch_vars = []
        for dx in dx_sys:
            torch_vars.append(torch.tensor(dx, dtype=DTYPE, device=DEVICE))
        print(torch.stack(torch_vars, dim=-1).reshape(x.shape))
        return torch.stack(torch_vars, dim=-1).reshape(x.shape)

    def g(self, t, x) -> torch.Tensor:
        dsigma_sys = [self.hb.hb_noise, self.hb.a_noise, 0, 0, 0]
        if self.hb.p_t_steady:
            dsigma_sys = dsigma_sys[0:4]
        torch_noise = []
        for dsigma in dsigma_sys:
            torch_noise.append(torch.tensor(dsigma, dtype=DTYPE, device=DEVICE))
        return torch.stack(torch_noise, dim=-1).reshape(x.shape)