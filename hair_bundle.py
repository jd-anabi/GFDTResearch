import numpy as np
import sympy as sym
import scipy.constants as constants

SCALE: float = 10**30 # ng nm^2 s^-2 MK^-1
K_B: float = SCALE * constants.k

class HairBundle:
    # -------------------------------- Helper functions (begin) --------------------------------
    @staticmethod
    def __k_gs(p_gs: float, k_gs_min: float, k_gs_max: float) -> float:
        """
        Gating spring stiffness
        :param p_gs: calcium binding probability for gating spring
        :param k_gs_min: min gating spring stiffness
        :param k_gs_max: max gating spring stiffness
        :return: equation for the gating spring stiffness
        """
        return k_gs_max - p_gs * (k_gs_max - k_gs_min)

    @staticmethod
    def __f_gs(k_gs: float, p_t: float, x_hb: float, x_a: float, x_c: float, d: float, gamma: float) -> float:
        """
        Gating spring force
        :param k_gs: stiffness of gating spring
        :param p_t: open channel probability
        :param x_hb: hair bundle displacement
        :param x_a: adaptation motor displacement
        :param x_c: average equilibrium position of adaptation motor
        :param d: channel gate opening distance
        :param gamma: geometric conversion factor
        :return: equation for the gating spring force
        """
        return -1 * k_gs * (gamma * x_hb - x_a + x_c - p_t * d)

    @staticmethod
    def __s(s_min: float, s_max: float, p_m: float) -> float:
        """
        Slipping rate
        :param s_min: minimum slipping rate
        :param s_max: maximum slipping rate
        :param p_m: calcium binding probability for adaptation motors
        :return: equation for the slipping rate
        """
        return s_min + p_m * (s_max - s_min)

    @staticmethod
    def __c(c_min: float, c_max: float, p_m: float) -> float:
        """
        Climbing rate
        :param c_min: minimum climbing rate
        :param c_max: maximum climbing rate
        :param p_m: calcium binding probability for adaptation motors
        :return: equation for the climbing rate
        """
        return c_max - p_m * (c_max - c_min)

    @staticmethod
    def __p_t0(delta_e: float, temp: float, k_gs: float, d: float, x_hb: float, x_a: float, x_c: float) -> float:
        """
        Open channel probability at equilibrium
        :param delta_e: intrinsic energy difference between the transduction channel's two states
        :param temp: temperature
        :param k_gs: gating spring stiffness
        :param d: channel gate opening distance
        :param x_hb: hair bundle displacement
        :param x_a: adaptation motor displacement
        :param x_c: average equilibrium position of adaptation motor
        :return: equation fot the open channel probability at equilibrium
        """
        return 1 / (1 + sym.exp(delta_e / (K_B * temp) - k_gs * d / (K_B * temp) * (x_hb - x_a + x_c - d / 2)))

    @staticmethod
    def __ca2_m(z_ca: float, d_ca: float, r_m: float, v_m: float, e_t_ca: float, p_t_ca: float, temp: float,
                ca2_hb_in: float, ca2_hb_ext: float, p_t: float) -> float:
        """
        Calcium ion concentration near adaptation motor
        :param z_ca: valence number
        :param d_ca: diffusion constant
        :param r_m: distance of adaptation motor
        :param v_m: voltage
        :param e_t_ca: reversal voltage for calcium
        :param p_t_ca: transduction-channel permeability fo calcium
        :param temp: temperature
        :param ca2_hb_in: intracellular concentration
        :param ca2_hb_ext: extracellular concentration
        :param p_t: open channel probability
        :return: equation for the calcium ion concentration near adaptation motor
        """
        exp = sym.exp(-1 * z_ca * constants.elementary_charge * v_m / (K_B * temp))
        g_t_ca_max = p_t_ca * z_ca**2 * constants.elementary_charge**2 / (K_B * temp) * (ca2_hb_in - ca2_hb_ext * exp) / (1 - exp)
        g_t_ca = p_t * g_t_ca_max
        i_t_ca = g_t_ca * (v_m - e_t_ca)
        return -1 * i_t_ca / (2 * constants.pi * z_ca * constants.elementary_charge * d_ca * r_m)

    @staticmethod
    def __ca2_gs(z_ca: float, d_ca: float, r_gs: float, v_m: float, e_t_ca: float, p_t_ca: float, temp: float,
                ca2_hb_in: float, ca2_hb_ext: float, p_t: float) -> float:
        """
        Calcium ion concentration near gating spring
        :param z_ca: valence number
        :param d_ca: diffusion constant
        :param r_gs: distance of gating spring
        :param v_m: voltage
        :param e_t_ca: reversal voltage for calcium
        :param p_t_ca: transduction-channel permeability fo calcium
        :param temp: temperature
        :param ca2_hb_in: intracellular concentration
        :param ca2_hb_ext: extracellular concentration
        :param p_t: open channel probability
        :return: equation for the calcium ion concentration near gating spring
        """
        exp = sym.exp(-1 * z_ca * constants.elementary_charge * v_m / (K_B * temp))
        g_t_ca_max = p_t_ca * z_ca ** 2 * constants.elementary_charge ** 2 / (K_B * temp) * (
                    ca2_hb_in - ca2_hb_ext * exp) / (1 - exp)
        g_t_ca = p_t * g_t_ca_max
        i_t_ca = g_t_ca * (v_m - e_t_ca)
        return -1 * i_t_ca / (2 * constants.pi * z_ca * constants.elementary_charge * d_ca * r_gs)

    @staticmethod
    def __hb_noise(epsilon: float, temp: float, lam: float) -> float:
        """
        White noise for hair bundle
        :param epsilon: noise control variable
        :param temp: temperature
        :param lam: drag coefficient
        :return: equation for the hair bundle noise
        """
        return epsilon * np.sqrt(2 * K_B * temp * lam)

    @staticmethod
    def __a_noise(epsilon: float, temp: float, lam: float) -> float:
        """
        Time-independent white noise for the adaptation motor
        :param epsilon: noise control variable
        :param temp: temperature
        :param lam: drag coefficient
        :return: equation for the adaptation motor noise
        """
        return epsilon * np.sqrt(2 * K_B * temp * lam)
    # -------------------------------- ODEs --------------------------------
    @staticmethod
    def __x_hb_dot(gamma: float, n: float, f_gs: float, lambda_hb: float, k_sp: float, x_sp: float, x_hb: float) -> float:
        """
        Hair bundle displacement dynamics
        :param gamma: geometric conversion factor
        :param n: number of gating springs
        :param f_gs: gating spring force
        :param lambda_hb: hair bundle drag coefficient
        :param k_sp: stiffness of stereocilliary pivot
        :param x_sp: displacement of stereocilliary pivot
        :param x_hb: hair bundle displacement`
        :return: time derivative of hair bundle displacement
        """
        return -1 * (gamma * n * f_gs + k_sp * (x_hb - x_sp)) / lambda_hb

    @staticmethod
    def __x_a_dot(s: float, c: float, f_gs: float, k_es: float, x_es: float, x_a: float) -> float:
        """
        Adaptation motor displacement dynamics
        :param s: slipping rate
        :param c: climbing rate
        :param f_gs: gating spring force
        :param k_es: stiffness of the extent spring
        :param x_es: displacement of the extent spring
        :param x_a: adaptation motor displacement
        :return: time derivative of adaptation motor displacement
        """
        return -1 * c + s * (f_gs - k_es * (x_a + x_es))

    @staticmethod
    def __p_m_dot(k_m_plus: float, k_m_minus: float, ca2_m: float, p_m: float) -> float:
        """
        Calcium binding probability for adaptation motors dynamics
        :param k_m_plus: association constant for adaptation motors
        :param k_m_minus: dissociation constant for adaptation motors
        :param ca2_m: calcium ion concentration near adaptation motor
        :param p_t: open channel probability
        :param p_m: calcium binding probability for adaptation motors
        :return: time derivative of calcium binding probability for adaptation motors
        """
        return k_m_plus * ca2_m * (1 - p_m) - k_m_minus * p_m

    @staticmethod
    def __p_gs_dot(k_gs_plus: float, k_gs_minus: float, ca2_gs: float, p_gs: float) -> float:
        """
        Calcium binding probability for gating spring dynamics
        :param k_gs_plus: association constant for gating spring
        :param k_gs_minus: dissociation constant for gating spring
        :param ca2_gs: calcium ion concentration near gating spring
        :param p_t: open channel probability
        :param p_gs: calcium binding probability for gating spring
        :return: time derivative of calcium binding probability for gating spring
        """
        return k_gs_plus * ca2_gs * (1 - p_gs) - k_gs_minus * p_gs

    @staticmethod
    def __p_t_dot(tau_t: float, p_t0: float, p_t: float) -> float:
        """
        Open channel probability dynamics
        :param tau_t: finite time constant for open channel
        :param p_t0: open channel probability at equilibrium
        :param p_t: open channel probability
        :return: time derivative of open channel probability
        """
        return (p_t0 - p_t) / tau_t
    # -------------------------------- Helper functions (end) ----------------------------------

    def __init__(self, tau_t: float, c_min: float, c_max: float, s_min: float, s_max: float,
                 delta_e: float, k_m_plus: float, k_m_minus: float, k_gs_plus: float, k_gs_minus: float,
                 k_gs_min: float, k_gs_max: float, x_c: float, temp: float, z_ca: float, d_ca: float,
                 r_m: float, r_gs: float, v_m: float, e_t_ca: float, p_t_ca: float, ca2_hb_in: float,
                 ca2_hb_ext: float, gamma: float, n: float, lambda_hb: float, lambda_a: float, k_sp: float, x_sp: float,
                 k_es: float, x_es: float, d: float, epsilon: float, omega: float, amp: float, vis_amp: float, p_t_steady: bool):
        # parameters
        self.tau_t = tau_t # finite time constant
        self.c_min = c_min # min climbing rate
        self.c_max = c_max # max climbing rate
        self.s_min = s_min # min slipping rate
        self.s_max = s_max # max slipping rate
        self.delta_e = delta_e # intrinsic energy difference between the transduction channel's two states
        self.k_m_plus = k_m_plus # association constant for adaptation motors
        self.k_m_minus = k_m_minus # dissociation constant for adaptation motors
        self.k_gs_plus = k_gs_plus # association constant for gating springs
        self.k_gs_minus = k_gs_minus # dissociation constant for gating springs
        self.k_gs_min = k_gs_min # min gating spring stiffness
        self.k_gs_max = k_gs_max # max gating spring stiffness
        self.x_c = x_c # average equilibrium position of the adaptation motors
        self.temp = temp # temperature
        self.z_ca = z_ca # valence number
        self.d_ca = d_ca # diffusion constant
        self.r_m = r_m # distance of adaptation motor
        self.r_gs = r_gs # distance of gating spring
        self.v_m = v_m # membrane voltage
        self.e_t_ca = e_t_ca # reversal voltage
        self.p_t_ca = p_t_ca # transduction-channel permeability fo calcium
        self.ca2_hb_in = ca2_hb_in # intracellular concentration
        self.ca2_hb_ext = ca2_hb_ext # extracellular concentration
        self.gamma = gamma # geometric conversion factor
        self.n = n # number of gating springs
        self.lambda_hb = lambda_hb # viscous drag coefficient
        self.lambda_a = lambda_a # viscous drag coefficient for adaptation motor
        self.k_sp = k_sp # stereocilliary pivot stiffness
        self.x_sp = x_sp # stereocilliary pivot displacement
        self.k_es = k_es # extent spring stiffness
        self.x_es = x_es # extent spring displacement
        self.d = d # channel gate opening distance
        self.epsilon = epsilon # noise scale
        self.omega = omega  # frequency of stimulus force
        self.amp = amp # amplitude of stimulus force
        self.vis_amp = vis_amp # amplitude of the viscous portion of the stimulus force
        self.p_t_steady = p_t_steady # binary variable dictating whether p_t should be equal to the steady-state solution

        # hair bundle variables
        self.x_hb = sym.symbols('x_hb') # hair bundle displacement
        self.x_a = sym.symbols('x_a') # adaptation motor displacement
        self.p_m = sym.symbols('p_m') # calcium binding probability for adaptation motor
        self.p_gs = sym.symbols('p_gs') # calcium binding probability for gating spring
        self.p_t = sym.symbols('p_t') # open channel probability

        hb_symbols = sym.Tuple(self.x_hb, self.x_a, self.p_m, self.p_gs) # hb sympy symbols
        sdes = sym.Tuple(self.x_hb_dot, self.x_a_dot, self.p_m_dot, self.p_gs_dot)  # SDEs

        # check if we don't want to use the steady state solution for the open channel probability
        if not self.p_t_steady:
            hb_symbols = hb_symbols + sym.Tuple(self.p_t)
            sdes = sdes + sym.Tuple(self.p_t_dot)

        self.sde_sym_lambda_func = sym.lambdify(tuple(hb_symbols), list(sdes), modules=["math"], cse=True)  # lambdify ode system

        def sin_driving_force(t: float) -> float:
            """
            Sinusoidal stimulus force
            :param t: time
            :return: equation for a sinusoidal stimulus
            """
            return -1 * self.amp * np.sin(self.omega * t) + self.vis_amp * self.omega * np.cos(self.omega * t)

        self.sin_driving_force = sin_driving_force

    # -------------------------------- Varying parameters (begin) ----------------------------------
    @property
    def k_gs(self) -> float:
        return sym.simplify(self.__k_gs(self.p_gs, self.k_gs_min, self.k_gs_max))

    @property
    def f_gs(self) -> float:
        if self.p_t_steady:
            return sym.simplify(self.__f_gs(self.k_gs, self.p_t0, self.x_hb, self.x_a, self.x_c, self.d, self.gamma))
        return sym.simplify(self.__f_gs(self.k_gs, self.p_t, self.x_hb, self.x_a, self.x_c, self.d, self.gamma))

    @property
    def s(self) -> float:
        return sym.simplify(self.__s(self.s_min, self.s_max, self.p_m))

    @property
    def c(self) -> float:
        return sym.simplify(self.__c(self.c_min, self.c_max, self.p_m))

    @property
    def p_t0(self) -> float:
        return sym.simplify(self.__p_t0(self.delta_e, self.temp, self.k_gs, self.d, self.x_hb, self.x_a, self.x_c))

    @property
    def ca2_m(self) -> float:
        if self.p_t_steady:
            return sym.simplify(self.__ca2_m(self.z_ca, self.d_ca, self.r_m, self.v_m, self.e_t_ca, self.p_t_ca, self.temp, self.ca2_hb_in, self.ca2_hb_ext, self.p_t0))
        return sym.simplify(self.__ca2_m(self.z_ca, self.d_ca, self.r_m, self.v_m, self.e_t_ca, self.p_t_ca, self.temp, self.ca2_hb_in, self.ca2_hb_ext, self.p_t))

    @property
    def ca2_gs(self) -> float:
        if self.p_t_steady:
            return sym.simplify(self.__ca2_m(self.z_ca, self.d_ca, self.r_gs, self.v_m, self.e_t_ca, self.p_t_ca, self.temp, self.ca2_hb_in, self.ca2_hb_ext, self.p_t0))
        return sym.simplify(self.__ca2_m(self.z_ca, self.d_ca, self.r_gs, self.v_m, self.e_t_ca, self.p_t_ca, self.temp, self.ca2_hb_in, self.ca2_hb_ext, self.p_t))

    @property
    def hb_noise(self) -> float:
        return sym.simplify(self.__hb_noise(self.epsilon, self.temp, self.lambda_hb))

    @property
    def a_noise(self) -> float:
        return sym.simplify(self.__a_noise(self.epsilon, self.temp, self.lambda_a))
    # -------------------------------- Varying parameters (end) ----------------------------------

    # -------------------------------- ODEs (begin) ----------------------------------
    @property
    def x_hb_dot(self) -> float:
        return sym.simplify(self.__x_hb_dot(self.gamma, self.n, self.f_gs, self.lambda_hb, self.k_sp, self.x_sp, self.x_hb))
    @property
    def x_a_dot(self) -> float:
        return sym.simplify(self.__x_a_dot(self.s, self.c, self.f_gs, self.k_es, self.x_es, self.x_a))
    @property
    def p_m_dot(self) -> float:
        return sym.simplify(self.__p_m_dot(self.k_m_plus, self.k_m_minus, self.ca2_m, self.p_m))
    @property
    def p_gs_dot(self) -> float:
        return sym.simplify(self.__p_gs_dot(self.k_gs_plus, self.k_gs_minus, self.ca2_gs, self.p_gs))
    @property
    def p_t_dot(self) -> float:
        return sym.simplify(self.__p_t_dot(self.tau_t, self.p_t0, self.p_t))
    # -------------------------------- ODEs (end) ----------------------------------