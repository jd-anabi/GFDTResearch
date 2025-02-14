import csv

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing as mp

import helpers

if __name__ == '__main__':
    # read non-dimensional hair cell from csv file
    with open('nd_hair_cell_0.csv', newline='') as csvfile:
        params = csv.reader(csvfile, delimiter=',')
        rows = [row for row in params]
    x0 = np.array([float(i) for i in rows[0]][:4]) # initial conditions
    s_osc = 2 * sp.constants.pi * 0.07 # spontaneous oscillation frequency for this hair bundle
    params = [float(i) for i in rows[1]]

    dt = 1e-3
    t = np.arange(0, 1000, dt)

    # solve sdes of the non-dimensional hair bundle
    num_trials = 50
    omegas = np.zeros(2 * num_trials, dtype=float)
    args_list = np.zeros(2 * num_trials, dtype=tuple)
    with mp.Pool() as pool:
        for i in range(2 * num_trials):
            s_osc_curr = s_osc + (i - num_trials + 10) * s_osc / num_trials
            omegas[i] = s_osc_curr
            args_list[i] = (t, True, s_osc_curr, params, x0)
        hb_sols = pool.starmap(helpers.nd_hb_sols, args_list)

    hb_pos_omegas = np.zeros(2 * num_trials, dtype=np.ndarray)
    for i in range(2 * num_trials):
        hb_pos_omegas[i] = hb_sols[i][:, 0]
    hb_pos0 = hb_pos_omegas[num_trials - 10]
    plt.plot(t, hb_pos0)
    plt.xlim(t[0] + 950, t[-1])
    plt.ylim(np.min(hb_pos0[int(len(t) / 2):]) - 0.25, np.max(hb_pos0[int(len(t) / 2):]) + 0.25)
    plt.xlabel('Time')
    plt.ylabel('Hair-bundle position')
    plt.show()

    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]
    hb_pos1_freq = sp.fft.fftshift(sp.fft.fft(hb_pos0 - np.mean(hb_pos0)))[len(t) // 2:]
    plt.plot(freq, np.abs(hb_pos1_freq) / len(t))
    plt.xlim(0, 0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.show()

    spon_osc_freq = freq[np.where(np.abs(hb_pos1_freq) == np.max(np.abs(hb_pos1_freq)))[0][0]]
    print(f'Frequency of spontaneous oscillations: {spon_osc_freq}')

    x_sfs = np.zeros(2 * num_trials, dtype=np.ndarray)
    for i in range(len(x_sfs)):
        x_sfs[i] = np.array([np.sin(omegas[i] * j) for j in t])
    thetas = [helpers.fdt_ratio(float(omegas[i]), np.array([hb_pos_omegas[i]]), x_sfs[i], dt) for i in range(2 * num_trials)]
    p_opt = sp.optimize.curve_fit(helpers.log, omegas, thetas)[0]
    print(p_opt)

    plt.plot(omegas, helpers.log(omegas, *p_opt))
    plt.scatter(omegas, thetas)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\frac{1}{\theta}$')
    plt.show()

    plt.plot(omegas, [1 / helpers.log(omegas, *p_opt)[i] for i in range(len(omegas))])
    plt.scatter(omegas, [1 / thetas[i] for i in range(len(thetas))])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\theta$')
    plt.show()