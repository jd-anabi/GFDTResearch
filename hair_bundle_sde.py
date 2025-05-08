import torch

from hair_bundle import HairBundle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

class HairBundleSDE(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'
    def __init__(self, params: list, force_params: list, p_t_steady_state: bool):
        super().__init__()
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
        print(torch_vars)
        print(torch.stack(torch_vars, dim=-1))
        return torch.stack(torch_vars, dim=-1)

    def g(self, t, x) -> torch.Tensor:
        dsigma = [self.hb.hb_noise, self.hb.a_noise, 0, 0, 0]
        if self.hb.p_t_steady:
            dsigma = dsigma[0:4]
        print(torch.diag(torch.tensor(dsigma, dtype=DTYPE, device=DEVICE)))
        return torch.diag(torch.tensor(dsigma, dtype=DTYPE, device=DEVICE))