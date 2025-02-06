import csv

import sdeint
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import hair_bundle_nondimensional as hb_nd
import fdt_helper_functions as fdt_hf

if __name__ == '__main__':
    # read non-dimensional hair cell from csv file
    with open('nd_hair_cell_0.csv', newline='') as csvfile:
        params = csv.reader(csvfile, delimiter=',')
        rows = [row for row in params]

    x0 = np.array([float(i) for i in rows[0]]) # initial conditions

    # time interval
    t_interval = [0, 300]
    dt = 1e-3
    t = np.arange(t_interval[0], t_interval[1], dt)

    # hair bundle
    hair_bundle_nd = hb_nd.HairBundleNonDimensional(*([float(i) for i in rows[1]] + [True]))

    # run multiple trials
    num_trials = 1
    num_vars = 5
    hb_sols = np.zeros((num_trials, len(t), num_vars))
    for n in range(num_trials):
        hb_sols[n, :, :] = sdeint.itoEuler(hair_bundle_nd.f, hair_bundle_nd.g, x0, t)

    # write to files so we don't have to keep rerunning it
    n = 0
    hb_file_names = ['hb_pos_nd0.csv', 'a_pos_nd0.csv', 'p_m_nd0.csv', 'p_gs_nd0.csv', 'p_t_nd0.csv']
    for file_name in hb_file_names:
        with open(file_name, 'w+', newline='') as csvfile:
            row = csv.writer(csvfile, delimiter=',')
            row.writerows(hb_sols[:, :, n])
        n += 1
    hb_pos = hb_sols[:, :, 0]
    p_t = hb_sols[:, :, 4]

    '''
    with open('hb_pos_nd0.csv', newline='') as csvfile:
        params = csv.reader(csvfile, delimiter=',')
        rows = [row for row in params]
    hb_pos1 = [float(i) for i in rows[0]]
    '''
    hb_pos1 = hb_pos[0]

    plt.plot(t, hb_pos1)
    plt.plot(t, p_t[0])
    plt.xlim(t_interval[0] + 200, t_interval[1])
    plt.ylim(np.min(hb_pos1[int(len(t) / 2):]) - 0.25, np.max(hb_pos1[int(len(t) / 2):]) + 0.25)
    plt.show()

    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]
    hb_pos1_freq = sp.fft.fftshift(sp.fft.fft(hb_pos1 - np.mean(hb_pos1)))[len(t) // 2:]
    plt.plot(freq, np.abs(hb_pos1_freq) / len(t))
    plt.xlim(0, 0.5)
    plt.show()

    spon_osc_freq = freq[np.where(np.abs(hb_pos1_freq) == np.max(np.abs(hb_pos1_freq)))[0][0]]
    print(f'Frequency of spontaneous oscillations: {spon_osc_freq}')