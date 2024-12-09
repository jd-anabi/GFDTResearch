import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hair_bundle_nondimensional as hb_nd

if __name__ == '__main__':
    # initial conditions
    x_hb0 = 0.0
    x_a0 = 0.0
    p_m0 = 0.0
    p_gs0 = 0.0
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
    tau_m = np.linspace(1e-9, 1000, 50)
    u_gs_max = np.linspace(1e-9, 1000, 50)
    delta_e = np.linspace(1e-9, 10, 10)
    k_gs_min = np.linspace(1e-9, 1, 2)
    chi_a = np.linspace(1e-9, 10, 10)

    # time interval
    t_interval = [0, 2000]
    t = np.linspace(t_interval[0], t_interval[1], 1000000)

    # initial
    hair_bundle_nd = hb_nd.HairBundleNonDimensional(tau_hb, tau_m[0], tau_gs, tau_t, c_min, s_min, s_max, ca2_m, ca2_gs, u_gs_max[0], delta_e[0], k_gs_min[0], chi_hb, chi_a[0], x_c)
    w_sol = sp.integrate.solve_ivp(hair_bundle_nd.odes_for_solver, t_interval, z0, t_eval=t, method='LSODA', dense_output=True)
    plt.plot(t, w_sol.y[0])
    plt.xlim(t_interval[0] + t_interval[1] / 2, t_interval[1] / 2 + 250)
    plt.ylim(np.min(w_sol.y[0][int(len(t) / 2):]) - 0.25, np.max(w_sol.y[0][int(len(t) / 2):]) + 0.25)
    plt.show()
    # rest of the varying parameters
    for i0 in range(1, len(tau_m)):
        for i1 in range(1, len(u_gs_max)):
            for i2 in range(1, len(delta_e)):
                for i3 in range(1, len(k_gs_min)):
                    for i4 in range(1, len(chi_a)):
                        hair_bundle_nd.tau_m = tau_m[i0]
                        hair_bundle_nd.u_gs_max = u_gs_max[i1]
                        hair_bundle_nd.delta_e = delta_e[i2]
                        hair_bundle_nd.k_gs_min = k_gs_min[i3]
                        hair_bundle_nd.chi_a = chi_a[i4]
                        w_sol = sp.integrate.solve_ivp(hair_bundle_nd.odes_for_solver, t_interval, z0, t_eval=t,
                                                       method='LSODA', dense_output=True)
                        plt.plot(t, w_sol.y[0])
                        plt.xlim(t_interval[0] + t_interval[1] / 2, t_interval[1] / 2 + 250)
                        plt.ylim(np.min(w_sol.y[0][int(len(t) / 2):]) - 0.25,
                                 np.max(w_sol.y[0][int(len(t) / 2):]) + 0.25)
                        plt.show()