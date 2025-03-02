import csv
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing as mp

import helpers

if __name__ == '__main__':
    # read hair cell from txt file
    line = 0
    pattern = re.compile(r'[\s=]+([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)$') # use pattern matching to extract values

    x0 = np.zeros(5, dtype=float)
    nd = False # whether we want to use the non-dimensional model or not
    if nd:
        file = 'nd_hair_cell_0.txt'
        params = np.zeros(17, dtype=float)
    else:
        file = 'hair_cell_0.txt'
        params = np.zeros(35, dtype=float)
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            if line < 5:
                x0[line] = val
            else:
                params[line - 5] = val
            line = line + 1
    s_osc = 2 * sp.constants.pi * 0.07 # spontaneous oscillation frequency for this hair bundle
    pt0_params = params[9:15]
    pt_steady = True
    if pt_steady:
        x0 = x0[:4]

    dt = 1e-3
    t = np.arange(0, 1000, dt)
    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]

    # solve sdes of the non-dimensional hair bundle
    num_trials = 5
    omegas = np.zeros(2 * num_trials, dtype=float)
    domega = 0
    args_list = np.zeros(2 * num_trials, dtype=tuple)
    with mp.Pool() as pool:
        for i in range(2 * num_trials):
            s_osc_curr = s_osc + (i - num_trials + domega) * s_osc / num_trials
            omegas[i] = s_osc_curr
            args_list[i] = (t, True, s_osc_curr, params, x0, nd)
        hb_sols = pool.starmap(helpers.hb_sols, args_list)

    # creating figure and subplots
    omega_indices = [0, num_trials - 2, num_trials - 1, num_trials, num_trials + 1, num_trials + 2, 2 * num_trials - 1]
    n = len(omega_indices) # n <= 2 * num_trials
    num_vars = 6
    fig, axes = plt.subplots(n, num_vars)
    fig_f, axes_f = plt.subplots(n, num_vars) # for frequency domain plots
    titles = [r'$x_{hb}(t)$', r'$x_a(t)$', r'$p_m(t)$', r'$p_{gs}(t)$', r'$p_t^{(0)}(t)$', r'$k_{gs}(t)$']
    titles_f = [r'$x_{hb}(\omega)$', r'$x_a(\omega)$', r'$p_m(\omega)$', r'$p_{gs}(\omega)$', r'$p_t^{(0)}(\omega)$', r'$k_{gs}(\omega)$']

    # plotting
    for i in range(n):
        axes[i][0].set_ylabel(r'$\omega$ = {}'.format(round(omegas[omega_indices[i]], 5)))
        axes_f[i][0].set_ylabel(r'$\omega$ = {}'.format(round(omegas[omega_indices[i]], 5)))
        for j in range(num_vars):
            if i == 0:
                axes[i][j].set_xlim([t[0] + 950, t[-1]])
            else:
                axes[i][j].set_xlim([t[0] + 750, t[-1]])
                axes_f[i][j].set_xlim([0, 0.5])
            if i == 0:
                axes[i][j].set_title(titles[j])
                axes_f[i][j].set_title(titles_f[j])
            if not (j == 4 or j == 5):
                freq_ij_plot = sp.fft.fftshift(sp.fft.fft(hb_sols[i][:, j] - np.mean(hb_sols[i][:, j])))[len(t) // 2:]
                axes[i][j].plot(t, hb_sols[i][:, j])
                axes_f[i][j].plot(freq, np.abs(freq_ij_plot) / len(t))
                continue
            elif j == 4:
                var = helpers.p_t0(hb_sols[i][:, 0], hb_sols[i][:, 1], hb_sols[i][:, 3], *pt0_params)
            else:
                var = helpers.k_gs(hb_sols[i][:, 3], pt0_params[2])

            freq_ij_plot = sp.fft.fftshift(sp.fft.fft(var - np.mean(var)))[len(t) // 2:]
            axes[i][j].plot(t, var)
            axes_f[i][j].plot(freq, np.abs(freq_ij_plot) / len(t))
    fig.set_figwidth(20)
    fig.set_figheight(18)
    fig_f.set_figwidth(20)
    fig_f.set_figheight(18)
    plt.show()

    hb_pos0 = hb_sols[0][:, 0] # data with no driving force
    hb_pos_omegas = np.zeros(2 * num_trials - 1, dtype=np.ndarray) # rest of the data
    for i in range(2 * num_trials - 1):
        hb_pos_omegas[i] = hb_sols[i + 1][:, 0]

    hb_pos0_freq = sp.fft.fftshift(sp.fft.fft(hb_pos0 - np.mean(hb_pos0)))[len(t) // 2:]
    spon_osc_freq = freq[np.where(np.abs(hb_pos0_freq) == np.max(np.abs(hb_pos0_freq)))[0][0]]
    print(f'Frequency of spontaneous oscillations: {spon_osc_freq}')

    # fdt ratio
    omegas_driven = omegas[1:]
    x_sfs = np.zeros(2 * num_trials - 1, dtype=np.ndarray)
    for i in range(len(x_sfs)):
        x_sfs[i] = np.array([np.sin(omegas_driven[i] * j) for j in t])
    fdt_vars = [helpers.fdt_ratio(float(omegas_driven[i]), np.array([hb_pos_omegas[i]]), x_sfs[i], dt, True) for i in range(2 * num_trials - 1)]
    thetas = [fdt_vars[i][0] for i in range(2 * num_trials - 1)]
    autocorr = [fdt_vars[i][1] for i in range(2 * num_trials - 1)]
    lin_resp_omegas = [fdt_vars[i][2] for i in range(2 * num_trials - 1)]

    #p_opt = sp.optimize.curve_fit(helpers.log, omegas_driven, thetas)[0]
    #theta_fit = helpers.log(omegas_driven, *p_opt)

    plt.plot(t, hb_pos0)
    plt.xlim(t[0] + 975, t[-1])
    plt.show()

    plt.plot(t, autocorr[0])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$C_0$')
    plt.show()

    plt.scatter(omegas_driven, np.real(lin_resp_omegas))
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\tilde{\chi}_R$')
    plt.show()

    plt.scatter(omegas_driven, np.imag(lin_resp_omegas))
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\tilde{\chi}_I$')
    plt.show()

    #plt.plot(omegas_driven, theta_fit)
    plt.scatter(omegas_driven, thetas)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\theta$')
    plt.show()

    #plt.plot(omegas_driven, [1 / theta_fit[i] for i in range(len(theta_fit))])
    plt.scatter(omegas_driven, [1 / thetas[i] for i in range(len(thetas))])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\frac{1}{\theta}$')
    plt.show()