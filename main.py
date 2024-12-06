import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hair_bundle_nondimensional as hb_nd

if __name__ == '__main__':
    hair_bundle_nd = hb_nd.HairBundleNonDimensional(1e-9, 100, 1, 1, 1, 0.0001, 0.5, 1, 1000, 700, 7, 0.9, 1, 7, 1e-9)
    t_interval = [0, 2000]
    t = np.linspace(t_interval[0], t_interval[1], 1000000)
    x_hb0 = 0.0
    x_a0 = 0.0
    p_m0 = 0.0
    p_gs0 = 0.0
    p_t0 = 1
    z0 = (x_hb0, x_a0, p_m0, p_gs0, p_t0)
    w_sol = sp.integrate.solve_ivp(hair_bundle_nd.odes_for_solver, t_interval, z0, t_eval=t, method='LSODA', dense_output=True)
    print(w_sol)
    plt.plot(t, w_sol.y[0])
    plt.xlim(t_interval[0] + t_interval[1] / 2, t_interval[1] / 2 + 250)
    plt.ylim(np.min(w_sol.y[0][int(len(t) / 2):]) - 0.25, np.max(w_sol.y[0][int(len(t) / 2):]) + 0.25)
    plt.show()