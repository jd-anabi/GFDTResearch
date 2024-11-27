import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hair_bundle_nondimensional as hb_nd

if __name__ == '__main__':
    hair_bundle_nd = hb_nd.HairBundleNonDimensional(1, 10, 0.0001, 1, 1, 0.0001, 0.5, 1, 1000, 10, 1, 1, 1, 1, 0.0001)
    t_interval = [0, 1500]
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
    plt.xlim(50, 100)
    plt.show()