import numpy as np
import torch

from scipy import constants

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

SCALE: float = 1e3 # g m^2 s^-2 K^-1
K_B: float = SCALE * constants.k # kg m^2 s^-2 K^-1

class HairBundleSDE(torch.nn.Module):
    def __init__(self, tau_t: float, c_min: float, c_max: float, s_min: float, s_max: float,
                 delta_e: float, k_m_plus: float, k_m_minus: float, k_gs_plus: float, k_gs_minus: float,
                 k_gs_min: float, k_gs_max: float, x_c: float, temp: float, z_ca: float, d_ca: float,
                 r_m: float, r_gs: float, v_m: float, e_t_ca: float, p_t_ca: float, ca2_hb_in: float,
                 ca2_hb_ext: float, gamma: float, n: float, lambda_hb: float, lambda_a: float, k_sp: float, x_sp: float,
                 k_es: float, x_es: float, d: float, epsilon: float, omega: float, amp: float, vis_amp: float,
                 noise_type: str = 'diagonal', sde_type: str = 'ito', batch_size: int = 3):
        super().__init__()
        # parameters
        self.tau_t = tau_t  # finite time constant
        self.c_min = c_min  # min climbing rate
        self.c_max = c_max  # max climbing rate
        self.s_min = s_min  # min slipping rate
        self.s_max = s_max  # max slipping rate
        self.delta_e = delta_e  # intrinsic energy difference between the transduction channel's two states
        self.k_m_plus = k_m_plus  # association constant for adaptation motors
        self.k_m_minus = k_m_minus  # dissociation constant for adaptation motors
        self.k_gs_plus = k_gs_plus  # association constant for gating springs
        self.k_gs_minus = k_gs_minus  # dissociation constant for gating springs
        self.k_gs_min = k_gs_min  # min gating spring stiffness
        self.k_gs_max = k_gs_max  # max gating spring stiffness
        self.x_c = x_c  # average equilibrium position of the adaptation motors
        self.temp = temp  # temperature
        self.z_ca = z_ca  # valence number
        self.d_ca = d_ca  # diffusion constant
        self.r_m = r_m  # distance of adaptation motor
        self.r_gs = r_gs  # distance of gating spring
        self.v_m = v_m  # membrane voltage
        self.e_t_ca = e_t_ca  # reversal voltage
        self.p_t_ca = p_t_ca  # transduction-channel permeability fo calcium
        self.ca2_hb_in = ca2_hb_in  # intracellular concentration
        self.ca2_hb_ext = ca2_hb_ext  # extracellular concentration
        self.gamma = gamma  # geometric conversion factor
        self.n = n  # number of gating springs
        self.lambda_hb = lambda_hb  # viscous drag coefficient
        self.lambda_a = lambda_a  # viscous drag coefficient for adaptation motor
        self.k_sp = k_sp  # stereocilliary pivot stiffness
        self.x_sp = x_sp  # stereocilliary pivot displacement
        self.k_es = k_es  # extent spring stiffness
        self.x_es = x_es  # extent spring displacement
        self.d = d  # channel gate opening distance
        self.epsilon = epsilon  # noise scale
        self.omega = omega  # frequency of stimulus force

        # force parameters
        self.amp = amp  # amplitude of stimulus force
        self.vis_amp = vis_amp  # amplitude of the viscous portion of the stimulus force

        # sde model parameters
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.batch_size = batch_size

    def f(self, t, x) -> torch.Tensor:
        dx_hb = self.__x_hb_dot(x[:, 0], x[:, 1], x[:, 3], x[:, 4]).to(DEVICE)
        dx_a = self.__x_a_dot(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]).to(DEVICE)
        dp_m = self.__p_m_dot(x[:, 2], x[:, 4]).to(DEVICE)
        dp_gs = self.__p_gs_dot(x[:, 3], x[:, 4]).to(DEVICE)
        dp_t = self.__p_t_dot(x[:, 0], x[:, 1], x[:, 3], x[:, 4]).to(DEVICE)
        dx_hb = dx_hb.to(DEVICE) + self.__sin_force(t).to(DEVICE)
        dx = torch.stack((dx_hb, dx_a, dp_m, dp_gs, dp_t), dim=1).to(DEVICE)
        return dx

    def g(self, t, x) -> torch.Tensor:
        zero_noise = torch.zeros(self.batch_size, dtype=DTYPE, device=DEVICE)
        dsigma = torch.stack((self.__hb_noise(), self.__a_noise(), zero_noise, zero_noise, zero_noise), dim=1).to(DEVICE)
        return dsigma

    # -------------------------------- PDEs (begin) ----------------------------------
    def __x_hb_dot(self, x_hb, x_a, p_gs, p_t) -> torch.Tensor:
        k_gs = self.k_gs_max - p_gs * (self.k_gs_max - self.k_gs_min)
        f_gs = k_gs * (self.gamma * x_hb - x_a + self.x_c - p_t * self.d)
        return -1 * (self.gamma * self.n * f_gs + self.k_sp * (x_hb - self.x_sp)) / self.lambda_hb

    def __x_a_dot(self, x_hb, x_a, p_m, p_gs, p_t) -> torch.Tensor:
        c = self.c_max - p_m * (self.c_max - self.c_min)
        s = self.s_min + p_m * (self.s_max - self.s_min)
        k_gs = self.k_gs_max - p_gs * (self.k_gs_max - self.k_gs_min)
        f_gs = k_gs * (self.gamma * x_hb - x_a + self.x_c - p_t * self.d)
        return -1 * c + s * ((f_gs + f_gs.abs()) / 2 - self.k_es * (x_a - self.x_es))

    def __p_m_dot(self, p_m, p_t) -> torch.Tensor:
        arg = self.z_ca * constants.elementary_charge * self.v_m / (K_B * self.temp) * torch.ones(self.batch_size, dtype=DTYPE, device=DEVICE)
        arg = torch.clamp(arg, max=100).to(DEVICE)
        exp = torch.exp(arg).to(DEVICE)
        denom = 1 - exp
        denom = torch.where(denom.abs() < 1e-6, torch.tensor([1e-6] * self.batch_size, device=DEVICE, dtype=DTYPE), denom).to(DEVICE)
        g_t_ca_max = self.p_t_ca * self.z_ca ** 2 * constants.elementary_charge / (K_B * self.temp) * (self.ca2_hb_in - self.ca2_hb_ext) / denom
        g_t_ca = p_t * g_t_ca_max
        i_t_ca = g_t_ca * (self.v_m - self.e_t_ca)
        ca2_m = -1 * i_t_ca / (2 * constants.pi * self.z_ca * constants.elementary_charge * self.d_ca * self.r_m)
        return self.k_m_plus * ca2_m * (1 - p_m) - self.k_m_minus * p_m

    def __p_gs_dot(self, p_gs, p_t) -> torch.Tensor:
        arg = self.z_ca * constants.elementary_charge * self.v_m / (K_B * self.temp) * torch.ones(self.batch_size, dtype=DTYPE, device=DEVICE)
        arg = torch.clamp(arg, max=100).to(DEVICE)
        exp = torch.exp(arg).to(DEVICE)
        denom = 1 - exp
        denom = torch.where(denom.abs() < 1e-6, torch.tensor([1e-6] * self.batch_size, device=DEVICE, dtype=DTYPE), denom).to(DEVICE)
        g_t_ca_max = self.p_t_ca * self.z_ca ** 2 * constants.elementary_charge / (K_B * self.temp) * (self.ca2_hb_in - self.ca2_hb_ext) / denom
        g_t_ca = p_t * g_t_ca_max
        i_t_ca = g_t_ca * (self.v_m - self.e_t_ca)
        ca2_m = -1 * i_t_ca / (2 * constants.pi * self.z_ca * constants.elementary_charge * self.d_ca * self.r_m)
        return self.k_gs_plus * ca2_m * (1 - p_gs) - self.k_gs_minus * p_gs

    def __p_t_dot(self, x_hb, x_a, p_gs, p_t) -> torch.Tensor:
        k_gs = self.k_gs_max - p_gs * (self.k_gs_max - self.k_gs_min)
        a = torch.exp(self.delta_e / (K_B * self.temp) * torch.ones(self.batch_size, dtype=DTYPE, device=DEVICE))
        arg = -1 * k_gs * self.d / (K_B * self.temp) * (self.gamma * x_hb - x_a + self.x_c - self.d / 2)
        b = torch.exp(arg)
        p_t0 = 1 / (1 + a * b)
        return (p_t0 - p_t) / self.tau_t
    # -------------------------------- PDEs (end) ----------------------------------

    # stimulus force
    def __sin_force(self, t):
        return -1 * self.amp * torch.sin(self.omega * t * torch.ones(self.batch_size, dtype=DTYPE, device=DEVICE)) + self.vis_amp * self.omega * torch.cos(self.omega * t * torch.ones(self.batch_size, dtype=DTYPE, device=DEVICE))

    # noise
    def __hb_noise(self) -> torch.Tensor:
        return self.epsilon * torch.sqrt(2 * K_B * self.temp / self.lambda_hb * torch.ones(self.batch_size, dtype=DTYPE, device=DEVICE))

    def __a_noise(self) -> torch.Tensor:
        return self.epsilon * torch.sqrt(2 * K_B * self.temp / self.lambda_a * torch.ones(self.batch_size, dtype=DTYPE, device=DEVICE))