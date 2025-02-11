import sympy as sym
import numpy as np

class HairBundleNonDimensional:
    # -------------------------------- Helper functions (begin) --------------------------------
    @staticmethod
    def __k_gs(p_gs: float, k_gs_min: float) -> float:
        """
        Gating spring stiffness
        :param p_gs: calcium binding probability for gating spring
        :param k_gs_min: min gating spring stiffness
        :return: equation for the gating spring stiffness
        """
        return 1 - p_gs * (1 - k_gs_min)

    @staticmethod
    def __f_gs(k_gs: float, p_t: float, x_gs: float) -> float:
        """
        Gating spring force
        :param k_gs: stiffness of gating spring
        :param p_t: open channel probability
        :param x_gs: position of gating spring
        :return: equation for the gating spring force
        """
        return k_gs * (x_gs - p_t)

    @staticmethod
    def __x_gs(chi_hb: float, chi_a: float, x_hb: float, x_a: float, x_c: float) -> float:
        """
        Gating spring displacement
        :param chi_hb: hair bundle conversion factor
        :param chi_a: adaptation motor conversion factor
        :param x_hb: hair bundle displacement
        :param x_a: adaptation motor displacement
        :param x_c: average equilibrium position of the adaptation motors
        :return: equation for the gating spring displacement
        """
        return chi_hb * x_hb - chi_a * x_a + x_c

    @staticmethod
    def __s(s_min: float, p_m: float) -> float:
        """
        Slipping rate
        :param s_min: minimum slipping rate
        :param p_m: calcium binding probability for adaptation motors
        :return: equation for the slipping rate
        """
        return s_min + p_m * (1 - s_min)

    @staticmethod
    def __c(c_min: float, p_m: float) -> float:
        """
        Climbing rate
        :param c_min: minimum climbing rate
        :param p_m: calcium binding probability for adaptation motors
        :return: equation for the climbing rate
        """
        return 1 - p_m * (1 - c_min)

    @staticmethod
    def __p_t0(u_gs_max: float, delta_e: float, k_gs: float, x_gs: float) -> float:
        """
        Open channel probability at equilibrium
        :param u_gs_max: max gating spring potential
        :param delta_e: intrinsic energy difference between the transduction channel's two states
        :param k_gs: gating spring stiffness
        :param x_gs: gating spring displacement
        :return: equation for the open channel probability at equilibrium
        """
        return 1 / (1 + sym.exp(u_gs_max * (delta_e - k_gs * (x_gs - 0.5))))

    @staticmethod
    def __hb_noise(eta: float, tau: float) -> float:
        """
        White noise for hair bundle
        :param eta: noise control variable
        :param tau: finite time constant
        :return: equation for the hair bundle noise
        """
        return eta / tau

    @staticmethod
    def __a_noise0(eta: float, s_max: float, s_min: float) -> float:
        """
        Time-independent white noise for the adaptation motor
        :param eta: noise control variable
        :param s_max: max slipping rate
        :param s_min: min slipping rate
        :return: equation for the adaptation motor noise
        """
        return -1 * s_max * s_min * eta

    @staticmethod
    def __f_hb0(k_gs: float, tau_hb: float, x_c: float) -> float:
        """
        Time-independent external force on the hair bundle
        :param k_gs: gating spring stiffness
        :param tau_hb: finite time constant
        :param x_c: average equilibrium position of the adaptation motors
        :return: time-independent external force on the hair bundle
        """
        return -1 * k_gs * x_c / tau_hb

    # -------------------------------- ODEs --------------------------------
    @staticmethod
    def __x_hb_dot(tau_hb: float, f_gs: float, x_hb: float) -> float:
        """
        Hair bundle displacement dynamics
        :param tau_hb: finite time constant for hair bundle
        :param f_gs: gating spring force
        :param x_hb: hair bundle displacement
        :return: time derivative of hair bundle displacement
        """
        return -1 * (f_gs + x_hb) / tau_hb

    @staticmethod
    def __x_a_dot(s_max: float, s: float, c: float, f_gs: float, x_a: float) -> float:
        """
        Adaptation motor displacement dynamics
        :param s_max: max slipping rate
        :param s: slipping rate
        :param c: climbing rate
        :param f_gs: gating spring force
        :param x_a: adaptation motor displacement
        :return: time derivative of adaptation motor displacement
        """
        c_max = 1 - s_max
        return s_max * s * (f_gs - x_a) - c_max * c

    @staticmethod
    def __p_m_dot(tau_m: float, ca2_m: float, p_t: float, p_m: float) -> float:
        """
        Calcium binding probability for adaptation motors dynamics
        :param tau_m: finite time constant for adaptation motor
        :param ca2_m: calcium ion concentration near adaptation motor
        :param p_t: open channel probability
        :param p_m: calcium binding probability for adaptation motors
        :return: time derivative of calcium binding probability for adaptation motors
        """
        return (ca2_m * p_t * (1 - p_m) - p_m) / tau_m

    @staticmethod
    def __p_gs_dot(tau_gs: float, ca2_gs: float, p_t: float, p_gs: float) -> float:
        """
        Calcium binding probability for gating spring dynamics
        :param tau_gs: finite time constant for gating spring
        :param ca2_gs: calcium ion concentration near gating spring
        :param p_t: open channel probability
        :param p_gs: calcium binding probability for gating spring
        :return: time derivative of calcium binding probability for gating spring
        """
        return (ca2_gs * p_t * (1 - p_gs) - p_gs) / tau_gs

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

    def __init__(self, tau_hb: float, tau_m: float, tau_gs: float, tau_t: float,
                 c_min: float, s_min: float, s_max: float, ca2_m: float, ca2_gs: float,
                 u_gs_max: float, delta_e: float, k_gs_min: float, chi_hb: float, chi_a: float,
                 x_c: float, eta_hb: float, eta_a: float, omega: float, p_t_steady: bool):
        # parameters
        self.tau_hb = tau_hb # finite time constant for hair bundle
        self.tau_m = tau_m # finite time constant for adaptation motor
        self.tau_gs = tau_gs # finite time constant for gating spring
        self.tau_t = tau_t # finite time constant for open channel probability
        self.c_min = c_min # min climbing rate
        self.s_min = s_min # min slipping rate
        self.s_max = s_max # max slipping rate
        self.ca2_m = ca2_m # calcium ion concentration near adaptation motor
        self.ca2_gs = ca2_gs # calcium ion concentration near gating spring
        self.u_gs_max = u_gs_max # max gating spring potential
        self.delta_e = delta_e # intrinsic energy difference between the transduction channel's two states
        self.k_gs_min = k_gs_min # min gating spring stiffness
        self.chi_hb = chi_hb # hair bundle conversion factor
        self.chi_a = chi_a # adaptation conversion factor
        self.x_c = x_c # average equilibrium position of the adaptation motors
        self.eta_hb = eta_hb # hair bundle diffusion constant
        self.eta_a = eta_a # adaptation motor diffusion constant
        self.omega = omega # frequency of stimulus force
        self.p_t_steady = p_t_steady # binary variable dictating whether p_t should be equal to the steady-state solution

        def sin(t: float) -> float:
            """
            Sinusoidal stimulus force
            :param t: time
            :return: equation for a sinusoidal stimulus
            """
            return np.sin(self.omega * t)

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

        self.sde_sym_lambda_func = sym.lambdify(tuple(hb_symbols), list(sdes))  # lambdify ode system

        def f(x: list, t: float) -> np.ndarray:
            x_sf = sin(t)
            if self.p_t_steady:
                sde_sys = np.array(self.sde_sym_lambda_func(x_sf + x[0], x[1], x[2], x[3]))
                return sde_sys
            sde_sys = np.array(self.sde_sym_lambda_func(x_sf + x[0], x[1], x[2], x[3], x[4]))
            return sde_sys

        def g(x: list, t: float) -> np.ndarray:
            a_noise = self.a_noise0 - x[2] * (1 - self.s_min) * self.eta_a
            if self.p_t_steady:
                return np.diag([self.hb_noise, a_noise, 0, 0])
            return np.diag([self.hb_noise, a_noise, 0, 0, 0])

        self.f = f
        self.g = g

    # -------------------------------- Varying parameters (begin) ----------------------------------
    @property
    def k_gs(self) -> float:
        return sym.simplify(self.__k_gs(self.p_gs, self.k_gs_min))

    @property
    def x_gs(self) -> float:
        return sym.simplify(self.__x_gs(self.chi_hb, self.chi_a, self.x_hb, self.x_a, self.x_c))

    @property
    def f_gs(self) -> float:
        if self.p_t_steady:
            return sym.simplify(self.__f_gs(self.k_gs, self.p_t0, self.x_gs))
        return sym.simplify(self.__f_gs(self.k_gs, self.p_t, self.x_gs))

    @property
    def s(self) -> float:
        return sym.simplify(self.__s(self.s_min, self.p_m))

    @property
    def c(self) -> float:
        return sym.simplify(self.__c(self.c_min, self.p_m))

    @property
    def p_t0(self) -> float:
        return sym.simplify(self.__p_t0(self.u_gs_max, self.delta_e, self.k_gs, self.x_gs))

    @property
    def hb_noise(self) -> float:
        return sym.simplify(self.__hb_noise(self.eta_hb, self.tau_hb))

    @property
    def a_noise0(self) -> float:
        return sym.simplify(self.__a_noise0(self.eta_a, self.s_max, self.s_min))

    @property
    def f_hb0(self) -> float:
        return sym.simplify(self.__f_hb0(self.k_gs, self.tau_hb, self.x_c))
    # -------------------------------- Varying parameters (end) ----------------------------------

    # -------------------------------- ODEs (begin) ----------------------------------
    @property
    def x_hb_dot(self) -> float:
        return sym.simplify(self.__x_hb_dot(self.tau_hb, self.f_gs, self.x_hb))
    @property
    def x_a_dot(self) -> float:
        return sym.simplify(self.__x_a_dot(self.s_max, self.s, self.c, self.f_gs, self.x_a))
    @property
    def p_m_dot(self) -> float:
        if self.p_t_steady:
            return sym.simplify(self.__p_m_dot(self.tau_m, self.ca2_m, self.p_t0, self.p_m))
        return sym.simplify(self.__p_m_dot(self.tau_m, self.ca2_m, self.p_t, self.p_m))
    @property
    def p_gs_dot(self) -> float:
        if self.p_t_steady:
            return sym.simplify(self.__p_gs_dot(self.tau_gs, self.ca2_gs, self.p_t0, self.p_gs))
        return sym.simplify(self.__p_gs_dot(self.tau_gs, self.ca2_gs, self.p_t, self.p_gs))
    @property
    def p_t_dot(self) -> float:
        return sym.simplify(self.__p_t_dot(self.tau_t, self.p_t0, self.p_t))
    # -------------------------------- ODEs (end) ----------------------------------