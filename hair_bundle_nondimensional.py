import sympy as sym
from jedi.inference.gradual.typing import Callable

class HairBundleNonDimensional:
    # -------------------------------- Helper functions (begin) --------------------------------
    # -------------------------------- variable parameters --------------------------------
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
        :return: equation fot the open channel probability at equilibrium
        """
        return 1 / (1 + sym.exp(u_gs_max * (delta_e - k_gs * (x_gs - 0.5))))

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
    def __p_t(tau_t: float, p_t0: float, p_t: float) -> float:
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
                 u_gs_max: float, delta_e: float, k_gs_min: float, chi_hb: float, chi_a: float, x_c: float):
        # parameters
        """
        self.tau_hb = sym.Symbol('tau_hb') # finite time constant for hair bundle
        self.tau_m = sym.Symbol('tau_m') # finite time constant for adaptation motor
        self.tau_gs = sym.Symbol('tau_gs') # finite time constant for gating spring
        self.tau_t = sym.Symbol('tau_t') # finite time constant
        self.c_min = sym.Symbol('c_min') # min climbing rate
        self.s_min = sym.Symbol('s_min') # min slipping rate
        self.s_max = sym.Symbol('s_max') # max slipping rate
        self.ca2_m = sym.Symbol('ca2_m') # calcium ion concentration near adaptation motor
        self.ca2_gs = sym.Symbol('ca2_gs') # calcium ion concentration near gating spring
        self.u_gs_max = sym.Symbol('u_gs_max') # max gating spring potential
        self.delta_e = sym.Symbol('delta_e') # intrinsic energy difference between the transduction channel's two states
        self.k_gs_min = sym.Symbol('k_gs_min') # min gating spring stiffness
        self.chi_hb = sym.Symbol('chi_hb') # hair bundle conversion factor
        self.chi_a = sym.Symbol('chi_a') # adaptation conversion factor
        self.x_c = sym.Symbol('x_c') # average equilibrium position of the adaptation motors
        """

        # hair bundle variables
        x_hb = sym.symbols('x_hb') # hair bundle displacement
        x_a = sym.symbols('x_a') # adaptation motor displacement
        p_m = sym.symbols('p_m') # calcium binding probability for adaptation motor
        p_gs = sym.symbols('p_gs') # calcium binding probability for gating spring
        p_t = sym.symbols('p_t') # open channel probability
        self.hb_symbols = sym.Tuple(x_hb, x_a, p_m, p_gs, p_t)

        # varying parameters
        k_gs = sym.simplify(self.__k_gs(p_gs, k_gs_min))
        x_gs = sym.simplify(self.__x_gs(chi_hb, chi_a, x_hb, x_a, x_c))
        f_gs = sym.simplify(self.__f_gs(k_gs, p_t, x_gs))
        s = sym.simplify(self.__s(s_min, p_m))
        c = sym.simplify(self.__c(c_min, p_m))
        p_t0 = sym.simplify(self.__p_t0(u_gs_max, delta_e, k_gs, x_gs))
        self.var_params = sym.Tuple(k_gs, x_gs, f_gs, s, c, p_t0)

        # ODEs
        x_hb_dot = sym.simplify(self.__x_hb_dot(tau_hb, f_gs, x_hb))
        x_a_dot = sym.simplify(self.__x_a_dot(s_max, s, c, f_gs, x_a))
        p_m_dot = sym.simplify(self.__p_m_dot(tau_m, ca2_m, p_t, p_m))
        p_gs_dot = sym.simplify(self.__p_gs_dot(tau_gs, ca2_gs, p_t, p_gs))
        p_t_dot = sym.simplify(self.__p_t(tau_t, p_t0, p_t))
        self.odes = sym.Tuple(x_hb_dot, x_a_dot, p_m_dot, p_gs_dot, p_t_dot)

        self.ode_sym_lambda_func = sym.lambdify(tuple(self.hb_symbols), list(self.odes)) # lambdify ode system

        def odes_for_solver(t: list, z: tuple) -> Callable:
            x_hb_var, x_a_var, p_m_var, p_gs_var, p_t_var = z
            return self.ode_sym_lambda_func(x_hb_var, x_a_var, p_m_var, p_gs_var, p_t_var)

        self.odes_for_solver = odes_for_solver