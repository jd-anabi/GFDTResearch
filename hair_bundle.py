import scipy.constants as sc
import numpy as np

class HairBundle:
    k_B = sc.k
    pi = sc.pi
    q_e = sc.elementary_charge
    def __init__(self, lambda_hb=130e-9, k_sp=200e-6, k_gs=1000e-6, k_es=140e-6,
                 n=35, gamma=0.14, x_sp=2.46e-7, x_c=15e-9, x_es=0.0,
                 d=7e-9, c_max=0.04e-6, c_min=0.0e-6, s_max=0.2e6, s_min=0.0, k_m_plus=1e9, k_m_minus=200e3, k_gs_plus=2e9, k_gs_minus=35e3,
                 zca=2, dca=800e-12, pca=1e-18, r_m=20e-9, r_gs=10e-9, v_m=-55e-3, cca_in=0, cca_ext=0.25e-3, e_t=0, temp=295.0, p_t_inf=0.5, tau_t=1.0):
        # constant parameters
        self.lambda_hb = lambda_hb # hair bundle drag coefficient
        self.k_sp = k_sp # stiffness of the stereociliary pivot
        self.k_gs = k_gs # stiffness of gating spring
        self.k_es = k_es # stiffness of extent spring
        self.n = n # number of gating springs
        self.gamma = gamma # geometric conversion factor
        self.x_sp = x_sp # equilibrium position of the stereociliary pivot
        self.x_c = x_c # average equilibrium position of the adaptation motors
        self.x_es = x_es # equilibrium position of extent spring
        self.d = d # gating spring swing
        self.c_max = c_max # max climbing rate
        self.c_min = c_min # min climbing rate
        self.s_max = s_max # max slipping rate
        self.s_min = s_min # min slipping rate
        self.k_m_plus = k_m_plus # association constant of motor
        self.k_m_minus = k_m_minus # dissociation constant of motor
        self.k_gs_plus = k_gs_plus # association constant of gating spring
        self.k_gs_minus = k_gs_minus # dissociation constant of gating spring
        self.zca = zca # valence number for calcium ion
        self.dca = dca # diffusion constant for calcium ion
        self.pca = pca # calcium ion permeability of the channel
        self.r_m = r_m # distance of adaptation motor from corresponding transduction channel
        self.r_gs = r_gs # distance of gating spring from corresponding transduction chanel
        self.v_m = v_m # resting membrane potential of the hair cell
        self.cca_in = cca_in # intracellular calcium ion concentration
        self.cca_ext = cca_ext # extracellular calcium ion concentration
        self.e_t = e_t # reversal voltage for calcium
        self.temp = temp # temperature
        self.p_t_inf = p_t_inf # channel open probability at equilibrium
        self.tau_t = tau_t # finite time constant

        # initial conditions
        #self.ca_plus_m0 = (self.p_t_inf * self.pca * self.zca ** 2 * self.q_e**2 * self.v_m * self.cca_ext / (self.k_B * self.temp * (1 - np.exp(self.zca * self.q_e * self.v_m / (self.k_B * self.temp))))) / (2 * self.pi * self.zca * self.q_e * self.dca * self.r_m)

        def model(t, w):
            """
            Set of differential equations to describe the hair-bundle dynamics
            :param w: array of state variables
            :param t: time
            :return: solution array
            """
            x_hb, x_a, p_gs = w

            # variables to tidy things up and make it more readable
            #ghk_exp = np.exp(self.zca*self.q_e*self.v_m/(self.k_B*self.temp))
            #g_t_ca_max = self.pca * self.zca**2 * self.q_e**2 * (self.cca_in - self.cca_ext)/(self.k_B * self.temp * (1 - ghk_exp))
            #e_conc_den = 2*self.pi*self.zca*self.q_e**self.dca*self.r_m
            ca_plus_m = (self.p_t_inf * self.pca * self.zca ** 2 * self.q_e ** 2 * self.v_m * self.cca_ext / (self.k_B * self.temp * (1 - np.exp(self.zca * self.q_e * self.v_m / (self.k_B * self.temp))))) / (2 * self.pi * self.zca * self.q_e * self.dca * self.r_m)
            #A = np.exp(np.log(1 / self.p_t_inf - 1) + (self.k_gs_minus + (self.k_gs_plus - self.k_gs_minus) * p_gs) * self.d * (self.x_c - self.d / 2) / (self.k_B * self.temp))
            p_m = 1 / (1 + self.k_m_minus / (self.k_m_plus * ca_plus_m))

            # create array f = [x_hb'(t), x_a'(t), p_m'(t), p_t'(t)]
            x_hb_dot = (-1 * self.gamma * self.n * (self.k_gs_minus + (self.k_gs_plus - self.k_gs_minus) * p_gs) * (self.gamma * x_hb - x_a + self.x_c - self.d / (1 + (np.exp(np.log(1 / self.p_t_inf - 1) + (self.k_gs_minus + (self.k_gs_plus - self.k_gs_minus) * p_gs) * self.d * (self.x_c - self.d / 2) / (self.k_B * self.temp))) * np.exp(-1 * self.k_gs * self.d * (x_hb - x_a + self.x_c - self.d / 2))) - self.k_sp * (x_hb - self.x_sp))) / self.lambda_hb
            x_a_dot = (self.c_max - self.c_min) * p_m - self.c_max + ((self.s_max - self.s_min) * p_m + self.s_min) * ((self.k_gs_minus + (self.k_gs_plus - self.k_gs_minus) * p_gs) * (self.gamma * x_hb - x_a + self.x_c - self.d / (1 + (np.exp(np.log(1 / self.p_t_inf - 1) + (self.k_gs_minus + (self.k_gs_plus - self.k_gs_minus) * p_gs) * self.d * (self.x_c - self.d / 2) / (self.k_B * self.temp))) * np.exp(-1 * self.k_gs * self.d * (x_hb - x_a + self.x_c - self.d / 2)))) - self.k_es * (x_a - self.k_es))
            p_gs_dot = self.k_gs_plus * ca_plus_m * (1 - p_gs) - self.k_gs_minus * p_gs
            #x_hb_dot = (-1 * self.gamma * self.n * (self.gamma*x_hb - x_a + self.x_c - ((1/(1 + A*np.exp((-1*self.k_gs*self.d * (x_hb - x_a + self.x_c - self.d/2)/(self.k_B*self.temp))))))*self.d) - self.k_sp * (x_hb - self.x_sp)) / self.lambda_hb
            #x_a_dot = (-1*self.k_gs*(self.c_max - p_m*(self.c_max - self.c_min))) + (self.s_min + p_m*(self.s_max - self.s_min)) * ((self.gamma*x_hb - x_a + self.x_c - ((1/(1 + A*np.exp((-1*self.k_gs*self.d * (x_hb - x_a + self.x_c - self.d/2)/(self.k_B*self.temp))))))*self.d) - self.k_es * (x_a + self.x_es))
            #p_m_dot = -1*self.k_m_plus * ((((1/(1 + A*np.exp((-1*self.k_gs*self.d * (x_hb - x_a + self.x_c - self.d/2)/(self.k_B*self.temp))))))*g_t_ca_max*(self.v_m - self.e_t))/e_conc_den) * (1 - p_m) - self.k_m_minus * p_m
            #p_t_dot = ((1/(1 + A*np.exp((-1*self.k_gs*self.d * (x_hb - x_a + self.x_c - self.d/2)/(self.k_B*self.temp))))) - p_t) / self.tau_t
            #p_t_dot = ((-self.k_gs*self.d*A*np.exp((-1*self.k_gs*self.d * (x_hb - x_a + self.x_c - self.d/2)/(self.k_B*self.temp)))*(x_hb_dot - x_a_dot))/(A + np.exp((-1*self.k_gs*self.d * (x_hb - x_a + self.x_c - self.d/2)/(self.k_B*self.temp))))**2)
            f = [x_hb_dot, x_a_dot, p_gs_dot]
            return f

        self.model = model