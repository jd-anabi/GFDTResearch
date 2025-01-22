import csv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hair_bundle as hb
import hair_bundle_nondimensional as hb_nd

if __name__ == '__main__':
    nd = True
    if nd:
        # read non-dimensional hair cell from csv file
        with open('nd_hair_cell_0.csv', newline='') as csvfile:
            params = csv.reader(csvfile, delimiter=',')
            rows = [row for row in params]

        z0 = tuple([float(i) for i in rows[0]])

        # time interval
        t_interval = [0, 1]
        num_steps = 1000
        max_time = 1000
        t = np.linspace(t_interval[0], t_interval[1], num_steps)
        hb_pos = np.zeros((max_time, num_steps))

        # initial
        hair_bundle_nd = hb_nd.HairBundleNonDimensional(*[float(i) for i in rows[1]], t_interval[1], num_steps)
        for i in range(max_time):
            w_sol = sp.integrate.solve_ivp(hair_bundle_nd.odes_for_solver, t_interval, z0, t_eval=t, method='LSODA', dense_output=True)
            for j in range(num_steps):
                hb_pos[i][j] = w_sol.y[0][j]
            t_interval = [t_interval[0] + 1, t_interval[1] + 1]
            t = np.linspace(t_interval[0], t_interval[1], num_steps)

        hb_pos = hb_pos.flatten()
        t = np.linspace(0, max_time, max_time * num_steps)
        plt.plot(t, hb_pos)
        plt.xlim(990, max_time)
        #plt.ylim(np.min(hb_pos[int(len(t) / 2):]) - 0.25, np.max(hb_pos[int(len(t) / 2):]) + 0.25)
        plt.show()
    else:
        # read non-dimensional hair cell from csv file
        with open('hair_cell_0.csv', newline='') as csvfile:
            params = csv.reader(csvfile, delimiter=',')
            rows = [row for row in params]

        z0 = tuple([float(i) for i in rows[0]])

        # time interval
        t_interval = [250, 1000]
        t = np.linspace(t_interval[0], t_interval[1], 1000000)

        # initial
        hair_bundle = hb.HairBundle(*[float(i) for i in rows[1]])
        w_sol = sp.integrate.solve_ivp(hair_bundle.odes_for_solver, t_interval, z0, t_eval=t, method='LSODA',
                                       dense_output=True)
        plt.plot(t, w_sol.y[0])
        plt.xlim(t_interval[0] + 250, t_interval[1] - 400)
        plt.ylim(np.min(w_sol.y[0][int(len(t) / 2):]) - 0.25, np.max(w_sol.y[0][int(len(t) / 2):]) + 0.25)
        plt.show()