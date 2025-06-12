import numpy as np
import torch

class HairBundleSDE(torch.nn.Module):
    def __init__(self, tau_hb: float, tau_m: float, tau_gs: float, tau_t: float,
                 c_min: float, s_min: float, s_max: float, ca2_m: float, ca2_gs: float,
                 u_gs_max: float, delta_e: float, k_gs_ratio: float, chi_hb: float, chi_a: float,
                 x_c: float, eta_hb: float, eta_a: float, omega: np.ndarray, amp: float,
                 batch_size: int, device: torch.device = 'cuda', dtype: torch.dtype = torch.float64,
                 noise_type: str = 'general', sde_type: str = 'ito'):
        super().__init__()
        # parameters
        self.tau_hb = tau_hb  # finite time constant for hair-bundle
        self.tau_m = tau_m  # finite time constant for adaptation motor
        self.tau_gs = tau_gs  # finite time constant for gating spring
        self.tau_t = tau_t  # finite time constant for open channel probability
        self.c_min = c_min  # min climbing rate
        self.s_min = s_min  # min slipping rate
        self.s_max = s_max  # max slipping rate
        self.ca2_m = ca2_m  # calcium ion concentration near adaptation motor
        self.ca2_gs = ca2_gs  # calcium ion concentration near gating spring
        self.u_gs_max = u_gs_max  # max gating spring potential
        self.delta_e = delta_e  # intrinsic energy difference between the transduction channel's two states
        self.k_gs_ratio = k_gs_ratio  # gating spring stiffness ratio
        self.chi_hb = chi_hb  # hair bundle conversion factor
        self.chi_a = chi_a  # adaptation motor conversion factor
        self.x_c = x_c  # average equilibrium position of the adaptation motors
        self.eta_hb = eta_hb  # hair bundle diffusion constant
        self.eta_a = eta_a  # adaptation motor diffusion constant

        # force parameters
        self.amp = amp  # amplitude of stimulus force
        self.omega = torch.tensor(omega, dtype=dtype, device=device)

        # sde model parameters
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # subsuming parameters
        self.c_max = 1 - self.s_max
        self.c_offset = 1 - self.c_min
        self.s_offset = 1 - self.s_min
        self.k_gs_offset = 1 - self.k_gs_ratio
        self.E_exp = torch.exp(self.u_gs_max * self.delta_e * torch.ones(self.batch_size, dtype=self.dtype, device=self.device))

    def f(self, x, t) -> torch.Tensor:
        p_t0 = self.__p_t0(x[:, 0], x[:, 1], x[:, 3])
        dx_hb = self.__x_hb_dot(x[:, 0], x[:, 1], x[:, 3], p_t0)
        dx_a = self.__x_a_dot(x[:, 0], x[:, 1], x[:, 2], x[:, 3], p_t0)
        dp_m = self.__p_m_dot(x[:, 2], p_t0)
        dp_gs = self.__p_gs_dot(x[:, 3], p_t0)
        dx_hb = dx_hb + self.__sin_sf(t) / self.tau_hb
        dx = torch.stack((dx_hb, dx_a, dp_m, dp_gs), dim=1)
        return dx

    def g(self, x: torch.tensor = None, t: torch.tensor = None) -> torch.Tensor:
        dsigma = torch.tensor([self.__hb_noise(), self.__a_noise(), 0, 0], dtype=self.dtype, device=self.device)
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
    # -------------------------------- PDEs (end) ----------------------------------
    def __p_t0(self, x_hb, x_a, p_gs) -> torch.Tensor:
        k_gs = 1 - p_gs * self.k_gs_offset
        x_gs = self.chi_hb * x_hb - self.chi_a * x_a + self.x_c
        arg = -1 * self.u_gs_max * k_gs * (x_gs - 0.5)
        return 1 / (1 + self.E_exp * torch.exp(arg))

    # stimulus force
    def __sin_sf(self, t):
        return self.amp * torch.sin(self.omega * t)
    # noise
    def __hb_noise(self) -> float:
        return self.eta_hb / self.tau_hb

    def __a_noise(self) -> float:
        return -1 * self.s_max * self.s_min * self.eta_a