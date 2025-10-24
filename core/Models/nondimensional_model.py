from typing import Union

import numpy as np
import torch

class HairBundleSDE(torch.nn.Module):
    def __init__(self, tau_hb: Union[float, np.ndarray], tau_m: Union[float, np.ndarray], tau_gs: Union[float, np.ndarray], tau_t: Union[float, np.ndarray],
                 c_min: Union[float, np.ndarray], s_min: Union[float, np.ndarray], s_max: Union[float, np.ndarray], ca2_m: Union[float, np.ndarray],
                 ca2_gs: Union[float, np.ndarray], u_gs_max: Union[float, np.ndarray], delta_e: Union[float, np.ndarray], k_gs_ratio: Union[float, np.ndarray],
                 chi_hb: Union[float, np.ndarray], chi_a: Union[float, np.ndarray], x_c: Union[float, np.ndarray], eta_hb: Union[float, np.ndarray],
                 eta_a: Union[float, np.ndarray], force: np.ndarray, batch_size: int, device: torch.device = 'cuda', dtype: torch.dtype = torch.float64):
        super().__init__()
        # parameters
        self.tau_hb = torch.tensor(tau_hb, dtype=dtype, device=device)  # finite time constant for hair-bundle
        self.tau_m = torch.tensor(tau_m, dtype=dtype, device=device)  # finite time constant for adaptation motor
        self.tau_gs = torch.tensor(tau_gs, dtype=dtype, device=device)  # finite time constant for gating spring
        self.tau_t = torch.tensor(tau_t, dtype=dtype, device=device)  # finite time constant for open channel probability
        self.c_min = torch.tensor(c_min, dtype=dtype, device=device)  # min climbing rate
        self.s_min = torch.tensor(s_min, dtype=dtype, device=device)  # min slipping rate
        self.s_max = torch.tensor(s_max, dtype=dtype, device=device)  # max slipping rate
        self.ca2_m = torch.tensor(ca2_m, dtype=dtype, device=device)  # calcium ion concentration near adaptation motor
        self.ca2_gs = torch.tensor(ca2_gs, dtype=dtype, device=device)  # calcium ion concentration near gating spring
        self.u_gs_max = torch.tensor(u_gs_max, dtype=dtype, device=device)  # max gating spring potential
        self.delta_e = torch.tensor(delta_e, dtype=dtype, device=device)  # intrinsic energy difference between the transduction channel's two states
        self.k_gs_ratio = torch.tensor(k_gs_ratio, dtype=dtype, device=device)  # gating spring stiffness ratio
        self.chi_hb = torch.tensor(chi_hb, dtype=dtype, device=device)  # hair bundle conversion factor
        self.chi_a = torch.tensor(chi_a, dtype=dtype, device=device)  # adaptation motor conversion factor
        self.x_c = torch.tensor(x_c, dtype=dtype, device=device)  # average equilibrium position of the adaptation motors
        self.eta_hb = torch.tensor(eta_hb, dtype=dtype, device=device)  # hair bundle diffusion constant
        self.eta_a = torch.tensor(eta_a, dtype=dtype, device=device)

        # force parameters
        self.force = torch.tensor(force, dtype=dtype, device=device)

        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # subsuming parameters
        self.c_max = 1 - self.s_max
        self.c_offset = 1 - self.c_min
        self.s_offset = 1 - self.s_min
        self.k_gs_offset = 1 - self.k_gs_ratio
        self.E_exp = torch.exp(self.u_gs_max * self.delta_e * torch.ones(self.batch_size, dtype=self.dtype, device=self.device))

    def f(self, x, t, t_id) -> torch.Tensor:
        dx_hb = self.__x_hb_dot(x[:, 0], x[:, 1], x[:, 3], x[:, 4])
        dx_a = self.__x_a_dot(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4])
        dp_m = self.__p_m_dot(x[:, 2], x[:, 4])
        dp_gs = self.__p_gs_dot(x[:, 3], x[:, 4])
        dp_t = self.__p_t_dot(x[:, 0], x[:, 1], x[:, 3], x[:, 4])
        dx_hb = dx_hb + self.force[:, t_id] / self.tau_hb
        dx = torch.stack((dx_hb, dx_a, dp_m, dp_gs, dp_t), dim=1)
        return dx

    def g(self, x = None, t = None) -> torch.Tensor:
        dsigma = torch.tensor([self.__hb_noise(), self.__a_noise(), 0, 0, 0], dtype=self.dtype, device=self.device)
        return torch.tile(torch.diag(dsigma), (self.batch_size, 1, 1))

    # -------------------------------- PDEs (begin) ----------------------------------
    def __x_hb_dot(self, x_hb, x_a, p_gs, p_t) -> torch.Tensor:
        x_gs = self.chi_hb * x_hb - self.chi_a * x_a + self.x_c
        k_gs = 1 - p_gs * self.k_gs_offset
        f_gs = k_gs * (x_gs - p_t)
        return -1 * (f_gs + x_hb) / self.tau_hb

    def __x_a_dot(self, x_hb, x_a, p_m, p_gs, p_t) -> torch.Tensor:
        c = 1 - p_m * self.c_offset
        s = self.s_min + p_m * self.s_offset
        x_gs = self.chi_hb * x_hb - self.chi_a * x_a + self.x_c
        k_gs = 1 - p_gs * self.k_gs_offset
        f_gs = k_gs * (x_gs - p_t)
        return self.s_max * s * (f_gs - x_a) - self.c_max * c

    def __p_m_dot(self, p_m, p_t) -> torch.Tensor:
        return (self.ca2_m * p_t * (1 - p_m) - p_m) / self.tau_m

    def __p_gs_dot(self, p_gs, p_t) -> torch.Tensor:
        return (self.ca2_gs * p_t * (1 - p_gs) - p_gs) / self.tau_gs

    def __p_t_dot(self, x_hb, x_a, p_gs, p_t) -> torch.Tensor:
        k_gs = 1 - p_gs * self.k_gs_offset
        x_gs = self.chi_hb * x_hb - self.chi_a * x_a + self.x_c
        arg = -1 * self.u_gs_max * k_gs * (x_gs - 0.5)
        p_t0 = 1 / (1 + self.E_exp * torch.exp(arg))
        return (p_t0 - p_t) / self.tau_t
    # -------------------------------- PDEs (end) ----------------------------------
    # noise
    def __hb_noise(self) -> float:
        return self.eta_hb / self.tau_hb

    def __a_noise(self) -> float:
        return -1 * self.s_max * self.s_min * self.eta_a