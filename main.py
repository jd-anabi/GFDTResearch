import csv
import sdeint
import numpy as np
import matplotlib.pyplot as plt
import hair_bundle_nondimensional as hb_nd

if __name__ == '__main__':
    # read non-dimensional hair cell from csv file
    with open('nd_hair_cell_0.csv', newline='') as csvfile:
        params = csv.reader(csvfile, delimiter=',')
        rows = [row for row in params]

    x0 = np.array([float(i) for i in rows[0]]) # initial conditions

    # time interval
    t_interval = [0, 500]
    dt = 1e-3
    t = np.arange(t_interval[0], t_interval[1], dt)

    # hair bundle
    hair_bundle_nd = hb_nd.HairBundleNonDimensional(*[float(i) for i in rows[1]])
    hb_sol = sdeint.itoEuler(hair_bundle_nd.f, hair_bundle_nd.g, x0, t)
    hb_pos = hb_sol[:, 0]

    '''
    # number of simulations
    sim_num = 50
    hb_sims = np.zeros((sim_num, num_steps))

    # initial
    hair_bundle_nd = hb_nd.HairBundleNonDimensional(*[float(i) for i in rows[1]], t_interval[1], num_steps)
    for i in range(sim_num):
        while time_step < num_steps:
            w_sol = sp.integrate.solve_ivp(hair_bundle_nd.odes_for_solver, t_interval, z0, t_eval=t, method='LSODA', dense_output=True)
            if len(w_sol.y[0]) == num_steps:
                hb_pos[time_step] = w_sol.y[0][time_step]
                time_step += 1
            hair_bundle_nd = hb_nd.HairBundleNonDimensional(*[float(i) for i in rows[1]], t_interval[1], num_steps)
        hb_sims[i] = hb_pos
    '''

    plt.plot(t, hb_pos)
    plt.xlim(t_interval[0] + 400, t_interval[1])
    plt.ylim(np.min(hb_pos[int(len(t) / 2):]) - 0.25, np.max(hb_pos[int(len(t) / 2):]) + 0.25)
    plt.show()