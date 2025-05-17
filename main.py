import math
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import torch.multiprocessing as mp

import helpers

if __name__ == '__main__':
    # time and frequency arrays
    t = np.linspace(0, 250, int(1e6))
    dt = float(t[1] - t[0])
    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]

    # parameters
    params = np.zeros(33, dtype=float)

    # read hair cell from txt file
    line = 0
    file = input('Hair cell file: ')
    pattern = re.compile(r'[\s=]+([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)$')  # use pattern matching to extract values
    x0 = []
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            if line < 5:
                x0.append(val)
            else:
                params[line - 5] = val
            line = line + 1

    # read user input for spontaneous oscilaltion frequency and whether to use the steady-state solution for the open-channel probability
    osc_freq_center = 2 * sp.constants.pi * float(input("Frequency to center driving at (Hz): "))
    pt0_params = []
    # check if we want to use the steady-state solution
    if params[0] == 0:
        x0 = x0[:4] # slice out p_t variable
        pt0_params = [params[5], *params[10:14], params[31]]

    # set up forcing params
    num_trials = int(input('Number of trials less than or equal to frequency center (total number of trials is twice this values): '))
    omegas = np.zeros(2 * num_trials, dtype=float)
    file = 'sin_force_params.txt'
    forcing_amps = []
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            forcing_amps.append(val)
    amp = forcing_amps[0]
    vis_amp = forcing_amps[1]
    args_list = np.zeros(2 * num_trials, dtype=tuple)
    for i in range(2 * num_trials):
        curr_osc = i * osc_freq_center / num_trials
        omegas[i] = curr_osc
        args_list[i] = (t, x0, list(params), [curr_osc, amp, vis_amp])

    # multiprocessing and solve sdes
    #mp.set_start_method('spawn')
    #with mp.Pool(processes=mp.cpu_count()) as pool:
    #    results = pool.starmap(helpers.hb_sols, args_list)
    results = helpers.hb_sols(*args_list[0]) # shape: (T, BATCH_SIZE, d)
    print(results)

    # separate driven and not driven data
    hb_pos0 = results[:, 0, 0]  # data of the hair bundle position from the first batch
    #hb_pos_omegas = np.zeros(2 * num_trials - 1, dtype=np.ndarray)
    #for i in range(2 * num_trials - 1):
    #    hb_pos_omegas[i] = hb_sols[0][i + 1]

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