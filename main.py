import re

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing as mp

import helpers

if __name__ == '__main__':
    # time and frequency arrays
    dt = 1e-4
    t = np.arange(0, 500, dt)
    lims = [t[-1] - 150, t[-1] - 50]

    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]

    # whether to use the non-dimensional model or not
    user_input = input('Would you like to use the non-dimensional model (y/n): ').lower()
    if user_input in ['y', 'yes', 'true', 't', '1']:
        nd = True
    else:
        nd = False
    if nd:
        params = np.zeros(17, dtype=float)
    else:
        params = np.zeros(33, dtype=float)

    # read hair cell from txt file
    line = 0
    file = input('Hair cell file: ')
    pattern = re.compile(r'[\s=]+([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)$')  # use pattern matching to extract values
    x0 = np.zeros(5, dtype=float)
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            if line < 5:
                x0[line] = val
            else:
                params[line - 5] = val
            line = line + 1

    # parameters for redimensionalization
    tau_gs = params[2]
    chi_hb = params[12]
    tau_gs_hat = 1 / 35e3
    d = 7e-9
    gamma = 0.14
    x_sp = 2.46e-7
    temp = 295
    # redimensionalize scaling parameters
    alpha = d / gamma * chi_hb
    beta = x_sp / gamma
    a = tau_gs_hat / tau_gs
    t0 = 0
    alpha = 1
    beta = 0
    a = 1

    # read user input for spontaneous oscilaltion frequency and whether to use the steady-state solution for the open-channel probability
    osc_freq_center = 2 * sp.constants.pi * float(input("Frequency to center driving at (Hz): "))
    user_input = input('Want to use the steady-state solution for the open-channel probability (y/n): ').lower()
    pt0_params = []
    if user_input in ['y', 'yes', 'true', 't', '1']:
        x0 = x0[:4] # slice out p_t variable
        pt_steady = True
        if nd:
            pt0_params = params[9:15]
        else:
            pt0_params = [params[5], *params[10:14], params[31]]
    else:
        pt_steady = False

    # solve sdes of the hair bundle
    num_trials = int(input('Number of trials less than or equal to frequency center (total number of trials is twice this values): '))
    omegas = np.zeros(2 * num_trials, dtype=float)
    amp = 0.05
    amp_vis = 0.0
    args_list = np.zeros(2 * num_trials, dtype=tuple)
    for i in range(2 * num_trials):
        s_osc_curr = i * osc_freq_center / num_trials
        omegas[i] = s_osc_curr
        args_list[i] = (t, pt_steady, s_osc_curr, params, x0, nd, amp, amp_vis)
    # multiprocessing
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(helpers.hb_sols, args_list)
    print(results)
    hb_sols = [[results[j][:, k] for j in range(2 * num_trials)] for k in range(len(x0))]

    # separate driven and not driven data
    hb_pos = np.zeros(2 * num_trials, dtype=np.ndarray)
    for i in range(len(hb_pos)):
        hb_pos[i] = alpha * hb_sols[0][i] + beta # redimensionalize
    hb_pos0 = hb_pos[0]  # data with no driving force
    hb_pos_omegas = np.zeros(2 * num_trials - 1, dtype=np.ndarray)  # rest of the data
    for i in range(2 * num_trials - 1):
        hb_pos_omegas[i] = hb_pos[i + 1]
    t = a * t + t0 # redimensionalize time
    omegas = a * omegas # redimensionalize angular frequency

    # get frequency of spontaneous oscillations
    hb_pos0_freq = sp.fft.fftshift(sp.fft.fft(hb_pos0 - np.mean(hb_pos0)))[len(t) // 2:]  # fft for non-driven data
    spon_osc_freq = freq[np.where(np.abs(hb_pos0_freq) == np.max(np.abs(hb_pos0_freq)))[0][0]]  # frequency of spontaneous oscillations
    print(f'Frequency of spontaneous oscillations: {spon_osc_freq} Hz. Angular frequency: {2 * np.pi * spon_osc_freq} rad/s. Dimensionless angular frequency: {2 * np.pi * spon_osc_freq / a}')

    # creating figure and subplots
    if num_trials < 3:
        if num_trials == 1:
            omega_indices = [0, num_trials]
        else:
            omega_indices = [0, num_trials - 1, num_trials, num_trials + 1]
    else:
        omega_indices = [0, num_trials - 2, num_trials - 1, num_trials, num_trials + 1, num_trials + 2]
    n = len(omega_indices) # n <= 2 * num_trials
    num_vars = 5
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
                axes[i][j].set_xlim(lims[0], lims[1])
            else:
                axes[i][j].set_xlim(lims[0], lims[1])
                axes_f[i][j].set_xlim([0, 0.5])
            if i == 0:
                axes[i][j].set_title(titles[j])
                axes_f[i][j].set_title(titles_f[j])
            if pt_steady:
                if j != 4:
                    freq_ij_plot = sp.fft.fftshift(sp.fft.fft(hb_sols[j][i] - np.mean(hb_sols[j][i])))[len(t) // 2:]
                    axes[i][j].plot(t, hb_sols[j][i])
                    axes_f[i][j].plot(freq, np.abs(freq_ij_plot) / len(t))
                    continue
                else:
                    var = helpers.p_t0(hb_sols[0][i], hb_sols[1][i], hb_sols[3][i], pt0_params, nd)
            else:
                var = hb_sols[j][i]
            freq_ij_plot = sp.fft.fftshift(sp.fft.fft(var - np.mean(var)))[len(t) // 2:]
            axes[i][j].plot(t, var)
            axes_f[i][j].plot(freq, np.abs(freq_ij_plot) / len(t))
    fig.set_figwidth(20)
    fig.set_figheight(18)
    fig_f.set_figwidth(20)
    fig_f.set_figheight(18)
    plt.show()

    # fdt ratio
    omegas_driven = omegas[1:]
    x_sfs = np.zeros(2 * num_trials - 1, dtype=np.ndarray)
    args_list = np.zeros(2 * num_trials - 1, dtype=tuple)
    for i in range(2 * num_trials - 1):
        x_sfs[i] = np.array([-1 * amp * np.sin(omegas_driven[i] * j) + amp_vis * omegas_driven[i] * np.cos(omegas_driven[i] * j) for j in t])
        args_list[i] = (float(omegas_driven[i]), hb_pos0, hb_pos_omegas[i], x_sfs[i], dt)
    thetas = np.zeros(2 * num_trials - 1, dtype=float)
    for i in range(len(thetas)):
        thetas[i] = helpers.fdt_ratio(*args_list[i])
    autocorr = helpers.auto_corr(hb_pos0)
    lin_resp_omegas = np.array([helpers.lin_resp_freq(float(omegas_driven[i]), np.array(hb_pos_omegas[i], ndmin=2), x_sfs[i], dt) for i in range(2 * num_trials - 1)])

    #p_opt = sp.optimize.curve_fit(helpers.log, omegas_driven, thetas)[0]
    #theta_fit = helpers.log(omegas_driven, *p_opt)
    #lims = [t[-1] - 250, t[-1] - 50]

    plt.figure(figsize=(20, 18))
    plt.plot(t, hb_pos0)
    plt.xlabel(r'Time')
    plt.ylabel(r'$x_{hb}$')
    plt.xlim((0.005, 0.015))
    plt.show()

    plt.plot(t, autocorr)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$C_0$')
    plt.show()

    plt.scatter(omegas_driven, np.real(lin_resp_omegas))
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Re\{\tilde{\chi}\}$')
    plt.show()

    plt.scatter(omegas_driven, np.imag(lin_resp_omegas))
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Im\{\tilde{\chi}\}$')
    plt.show()

    plt.scatter(omegas_driven, thetas)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\theta$')
    plt.show()

    plt.scatter(omegas_driven, [1 / thetas[i] for i in range(len(thetas))])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\frac{1}{\theta}$')
    plt.show()