import csv

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing as mp

import helpers
import fdt_helper_functions as fdt_hf

if __name__ == '__main__':
    # read non-dimensional hair cell from csv file
    with open('nd_hair_cell_0.csv', newline='') as csvfile:
        params = csv.reader(csvfile, delimiter=',')
        rows = [row for row in params]
    x0 = np.array([float(i) for i in rows[0]][:4]) # initial conditions
    params = [float(i) for i in rows[1]]

    dt = 1e-3
    t = np.arange(0, 300, dt)

    # solve sdes of the non-dimensional hair bundle
    num_trials = 50
    args = (t, True, params, x0)
    with mp.Pool() as pool:
        hb_sols = pool.starmap(helpers.nd_hb_sols, [args for i in range(num_trials)])

    # write to files so we don't have to keep rerunning it
    '''
    n = 0
    hb_file_names = ['hb_pos_nd0.csv', 'a_pos_nd0.csv', 'p_m_nd0.csv', 'p_gs_nd0.csv']
    for file_name in hb_file_names:
        with open(file_name, 'w+', newline='') as csvfile:
            row = csv.writer(csvfile, delimiter=',')
            row.writerows(hb_sols[:, :, n])
        n += 1
    hb_pos = hb_sols[:, :, 0]
    with open('hb_pos_nd0.csv', newline='') as csvfile:
        params = csv.reader(csvfile, delimiter=',')
        rows = [row for row in params]
    hb_pos1 = [float(i) for i in rows[0]]
    '''

    hb_pos0 = hb_sols[0][:, 0]
    plt.plot(t, hb_pos0)
    plt.xlim(t[0] + 200, t[-1])
    plt.ylim(np.min(hb_pos0[int(len(t) / 2):]) - 0.25, np.max(hb_pos0[int(len(t) / 2):]) + 0.25)
    plt.show()

    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]
    hb_pos1_freq = sp.fft.fftshift(sp.fft.fft(hb_pos0 - np.mean(hb_pos0)))[len(t) // 2:]
    plt.plot(freq, np.abs(hb_pos1_freq) / len(t))
    plt.xlim(0, 0.5)
    plt.show()

    spon_osc_freq = freq[np.where(np.abs(hb_pos1_freq) == np.max(np.abs(hb_pos1_freq)))[0][0]]
    print(f'Frequency of spontaneous oscillations: {spon_osc_freq}')