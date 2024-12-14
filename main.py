import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hair_bundle_nondimensional as hb_nd

if __name__ == '__main__':
    # initial conditions
    x_hb0 = 0.0
    x_a0 = 0.0
    p_m0 = 0
    p_gs0 = 0
    p_t0 = 1
    z0 = (x_hb0, x_a0, p_m0, p_gs0, p_t0)

    # fixed parameters
    tau_hb = 1
    tau_gs = 1
    tau_t = 1e-9
    c_min = 1
    s_min = 1e-9
    s_max = 0.5
    ca2_m = 1
    ca2_gs = 1000
    chi_hb = 1
    x_c = 1e-9

    # varying parameters
    tau_m = 10
    u_gs_max = 10
    delta_e = 1
    k_gs_min = 1
    chi_a = 1

    # time interval
    t_interval = [250, 1000]
    t = np.linspace(t_interval[0], t_interval[1], 1000000)

    # initial
    hair_bundle_nd = hb_nd.HairBundleNonDimensional(tau_hb, tau_m, tau_gs, tau_t, c_min, s_min, s_max, ca2_m, ca2_gs, u_gs_max, delta_e, k_gs_min, chi_hb, chi_a, x_c)
    w_sol = sp.integrate.solve_ivp(hair_bundle_nd.odes_for_solver, t_interval, z0, t_eval=t, method='LSODA', dense_output=True)
    plt.plot(t, w_sol.y[0])
    plt.xlim(t_interval[0] + 250, t_interval[1] - 400)
    plt.ylim(np.min(w_sol.y[0][int(len(t) / 2):]) - 0.25, np.max(w_sol.y[0][int(len(t) / 2):]) + 0.25)
    plt.show()