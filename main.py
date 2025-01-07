import csv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hair_bundle_nondimensional as hb_nd

if __name__ == '__main__':
    # read hair cell from csv file
    row_num = 0
    with open('nd_hair_cell_0.csv', newline='') as csvfile:
        params = csv.reader(csvfile, delimiter=',')
        for row in params:
            match row_num:
                case 0:
                    # initial conditions
                    x_hb0 = float(row[0])
                    x_a0 = float(row[1])
                    p_m0 = float(row[2])
                    p_gs0 = float(row[3])
                    p_t0 = float(row[4])
                    z0 = (x_hb0, x_a0, p_m0, p_gs0, p_t0)

                case 1:
                    # fixed parameters
                    tau_hb = float(row[0])
                    tau_gs = float(row[1])
                    tau_t = float(row[2])
                    c_min = float(row[3])
                    s_min = float(row[4])
                    s_max = float(row[5])
                    ca2_m = float(row[6])
                    ca2_gs = float(row[7])
                    chi_hb = float(row[8])
                    x_c = float(row[9])

                case 2:
                    # varying parameters
                    tau_m = float(row[0])
                    u_gs_max = float(row[1])
                    delta_e = float(row[2])
                    k_gs_min = float(row[3])
                    chi_a = float(row[4])

            row_num += 1

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