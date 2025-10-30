import torch


class HairBundleSDE(torch.nn.Module):
    def __init__(self, tau_hb: torch.Tensor, tau_m: torch.Tensor, tau_gs: torch.Tensor, tau_t: torch.Tensor,
                 c_min: torch.Tensor, s_min: torch.Tensor, s_max: torch.Tensor, ca2_m: torch.Tensor,
                 ca2_gs: torch.Tensor, u_gs_max: torch.Tensor, delta_e: torch.Tensor, k_gs_ratio: torch.Tensor,
                 chi_hb: torch.Tensor, chi_a: torch.Tensor, x_c: torch.Tensor, eta_hb: torch.Tensor,
                 eta_a: torch.Tensor, force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # parameters
        self.tau_hb = tau_hb.to(self.device)  # finite time constant for hair-bundle
        self.tau_m = tau_m.to(self.device)  # finite time constant for adaptation motor
        self.tau_gs = tau_gs.to(self.device)  # finite time constant for gating spring
        self.tau_t = tau_t.to(self.device)  # finite time constant for open channel probability
        self.c_min = c_min.to(self.device)  # min climbing rate
        self.s_min = s_min.to(self.device)  # min slipping rate
        self.s_max = s_max.to(self.device)  # max slipping rate
        self.ca2_m = ca2_m.to(self.device)  # calcium ion concentration near adaptation motor
        self.ca2_gs = ca2_gs.to(self.device)  # calcium ion concentration near gating spring
        self.u_gs_max = u_gs_max.to(self.device)  # max gating spring potential
        self.delta_e = delta_e.to(self.device)  # intrinsic energy difference between the transduction channel's two states
        self.k_gs_ratio = k_gs_ratio.to(self.device)  # gating spring stiffness ratio
        self.chi_hb = chi_hb.to(self.device)  # hair bundle conversion factor
        self.chi_a = chi_a.to(self.device)  # adaptation motor conversion factor
        self.x_c = x_c.to(self.device)  # average equilibrium position of the adaptation motors
        self.eta_hb = eta_hb.to(self.device)  # hair bundle diffusion constant
        self.eta_a = eta_a.to(self.device)  # adaptation motor diffusion constant

        # force parameters
        self.force = force.to(self.device)

        # subsuming parameters
        self.c_max = 1 - self.s_max
        self.c_offset = 1 - self.c_min
        self.s_offset = 1 - self.s_min
        self.k_gs_offset = 1 - self.k_gs_ratio
        self.E_exp = torch.exp(self.u_gs_max * self.delta_e)

    def f(self, x, t, t_id: int = 0) -> torch.Tensor:
        p_t0 = self.__p_t0(x[:, 0], x[:, 1], x[:, 3])
        dx_hb = self.__x_hb_dot(x[:, 0], x[:, 1], x[:, 3], p_t0)
        dx_a = self.__x_a_dot(x[:, 0], x[:, 1], x[:, 2], x[:, 3], p_t0)
        dp_m = self.__p_m_dot(x[:, 2], p_t0)
        dp_gs = self.__p_gs_dot(x[:, 3], p_t0)
        dx_hb = dx_hb + self.force[:, t_id] / self.tau_hb
        dx = torch.stack((dx_hb, dx_a, dp_m, dp_gs), dim=1)
        return dx

    def g(self, x=None, t=None) -> torch.Tensor:
        hb_noise = self.__hb_noise()
        a_noise = self.__a_noise()
        dsigma = torch.stack((hb_noise, a_noise, torch.zeros_like(hb_noise), torch.zeros_like(hb_noise)), dim=0)
        dsigma = torch.atleast_2d(torch.transpose(dsigma, -1, 0))
        return torch.diag_embed(dsigma)

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

    # noise
    def __hb_noise(self) -> torch.Tensor:
        return self.eta_hb / self.tau_hb

    def __a_noise(self) -> torch.Tensor:
        return -1 * self.s_max * self.s_min * self.eta_a
