import re

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing as mp

import helpers

if __name__ == '__main__':
    # time and frequency arrays
    dt = 1e-4
    t = np.arange(0, 5, dt)
    lims = [t[-1] - 150, t[-1] - 50]
    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]

    # parameters
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

    # read user input for spontaneous oscilaltion frequency and whether to use the steady-state solution for the open-channel probability
    osc_freq_center = 2 * sp.constants.pi * float(input("Frequency to center driving at (Hz): "))
    user_input = input('Want to use the steady-state solution for the open-channel probability (y/n): ').lower()
    pt0_params = []
    if user_input in ['y', 'yes', 'true', 't', '1']:
        x0 = x0[:4] # slice out p_t variable
        pt_steady = True
        pt0_params = [params[5], *params[10:14], params[31]]
    else:
        pt_steady = False

    # solve sdes of the hair bundle
    num_trials = int(input('Number of trials less than or equal to frequency center (total number of trials is twice this values): '))
    omegas = np.zeros(2 * num_trials, dtype=float)
    amp = 0.0
    vis_amp = 0.0
    args_list = np.zeros(2 * num_trials, dtype=tuple)
    for i in range(2 * num_trials):
        s_osc_curr = i * osc_freq_center / num_trials
        omegas[i] = s_osc_curr
        args_list[i] = (t, x0, params, [s_osc_curr, amp, vis_amp], pt_steady)
    # multiprocessing
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(helpers.hb_sols, args_list)
    hb_sols = [[results[j][k] for j in range(2 * num_trials)] for k in range(len(x0))]
    print(hb_sols)

    # separate driven and not driven data
    hb_pos0 = hb_sols[0][0]  # data with no driving force
    hb_pos_omegas = np.zeros(2 * num_trials - 1, dtype=np.ndarray)  # rest of the data
    for i in range(2 * num_trials - 1):
        hb_pos_omegas[i] = hb_sols[0][i + 1]

    # get frequency of spontaneous oscillations
    hb_pos0_freq = sp.fft.fftshift(sp.fft.fft(hb_pos0 - np.mean(hb_pos0)))[len(t) // 2:]  # fft for non-driven data
    spon_osc_freq = freq[np.argmax(hb_pos0_freq)]  # frequency of spontaneous oscillations
    print(f'Frequency of spontaneous oscillations: {spon_osc_freq} Hz. Angular frequency: {2 * np.pi * spon_osc_freq} rad/s')

    # preliminary plotting
    plt.plot(t, hb_pos0)
    plt.xlabel(r'Time')
    plt.ylabel(r'$x_{hb}$')
    plt.xlim((0.005, 0.015))
    plt.show()

    plt.plot(freq, hb_pos0_freq)
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'$\hat{x}_{hb}$')
    plt.xlim((0.005, 0.015))
    plt.show()

    # fdt ratio
    omegas_driven = omegas[1:]
    x_sfs = np.zeros(2 * num_trials - 1, dtype=np.ndarray)
    args_list = np.zeros(2 * num_trials - 1, dtype=tuple)
    for i in range(2 * num_trials - 1):
        x_sfs[i] = np.array([-1 * amp * np.sin(omegas_driven[i] * j) + vis_amp * omegas_driven[i] * np.cos(omegas_driven[i] * j) for j in t])
        args_list[i] = (float(omegas_driven[i]), hb_pos0, hb_pos_omegas[i], x_sfs[i], dt)
    thetas = np.zeros(2 * num_trials - 1, dtype=float)
    for i in range(len(thetas)):
        thetas[i] = helpers.fdt_ratio(*args_list[i])
    autocorr = helpers.auto_corr(hb_pos0)
    lin_resp_omegas = np.array([helpers.lin_resp_freq(float(omegas_driven[i]), np.array(hb_pos_omegas[i], ndmin=2), x_sfs[i], dt) for i in range(2 * num_trials - 1)])

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