import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import helpers

if __name__ == '__main__':
    # time and frequency arrays
    dt = 1e-4
    ts = (0, 75)
    n = int((ts[-1] - ts[0]) / dt)
    t = np.linspace(ts[0], ts[-1], n)
    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))

    # parameters and initial conditions
    file_str = 'Non-dimensional'
    params = np.zeros(17, dtype=float)
    x0 = []
    hb_rescale_params = np.zeros(8, dtype=float)
    hb_nd_rescale_params = np.zeros(3, dtype=float) # 6, 12, 13

    # pattern matching for file reading
    pattern = re.compile(r'[\s=]+([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)$')  # use pattern matching to extract values

    # read hair cell,  force info, and rescaling parameters from txt files
    line = 0
    if sys.platform == 'win32':
        hair_cell_file_path = '\\Hair Cells\\' + file_str + '\\hair_cell_'
        rescale_file_path = '\\Rescaling\\rescaling_params_'
        force_file_path = '\\Force Info\\sin_force_params.txt'
    else:
        hair_cell_file_path = '/Hair Cells/' + file_str + '/hair_cell_'
        rescale_file_path = '/Rescaling/rescaling_params_'
        force_file_path = '/Force Info/sin_force_params.txt'

    # hair cell parameters
    file = os.getcwd() + hair_cell_file_path + input('Hair cell file number: ') + '.txt'
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            if line < 5:
                x0.append(val)
            else:
                params[line - 5] = val
            line = line + 1

    # rescaling parameters
    line = 0
    file = os.getcwd() + rescale_file_path + input('Rescaling file number: ') + '.txt'
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            hb_rescale_params[line] = val
            line = line + 1
    # need to add in non-dimensional parameters for rescaling too
    hb_nd_rescale_params[0] = params[6]
    hb_nd_rescale_params[1] = params[12]
    hb_nd_rescale_params[2] = params[13]

    # set up forcing params
    file = os.getcwd() + force_file_path
    forcing_amps = []
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            forcing_amps.append(val)
    amp = forcing_amps[0]
    k_sf = forcing_amps[1]
    # read user input for spontaneous oscilaltion frequency and whether to use the steady-state solution for the open-channel probability
    osc_freq_center = 2 * sp.constants.pi * float(input("Frequency to center driving at (Hz): "))
    args_list = (t, x0, list(params), [osc_freq_center, amp, k_sf])

    # multiprocessing and solve sdes
    results = helpers.hb_sols(*args_list) # shape: (T, BATCH_SIZE, d)

    # separate driven and not driven data
    hb_pos_undriven = results[:, 0, 0]  # data of the hair bundle position from the first batch
    hb_pos_driven_data = np.zeros((results.shape[1] - 1, n))
    for i in range(len(hb_pos_driven_data)):
        hb_pos_driven_data[i] = results[:, i + 1, 0]

    # rescale
    sf_pos = helpers.sf_pos(t, amp, osc_freq_center)
    hb_pos_undriven, sf_pos, t = helpers.rescale(hb_pos_undriven, sf_pos, t, *hb_rescale_params, *hb_nd_rescale_params)

    # get frequency of spontaneous oscillations
    hb_pos0_freq = sp.fft.fftshift(sp.fft.fft(hb_pos_undriven - np.mean(hb_pos_undriven)))[len(t) // 2:]  # fft for non-driven data
    spon_osc_freq = freq[np.argmax(hb_pos0_freq)]  # frequency of spontaneous oscillations
    print(f'Frequency of spontaneous oscillations: {spon_osc_freq} Hz. Angular frequency: {2 * np.pi * spon_osc_freq} rad/s')

    # preliminary plotting
    plt.plot(t, hb_pos_undriven)
    plt.xlabel(r'Time (ms)')
    plt.ylabel(r'$x_{hb}$ (nm)')
    plt.show()

    plt.plot(t[int(n / 4):int(3 * n / 4)], hb_pos_undriven[int(n / 4):int(3 * n / 4)])
    plt.xlabel(r'Time (ms)')
    plt.ylabel(r'$x_{hb}$ (nm)')
    plt.show()

    # autocorrelation function
    auto_correlation = helpers.auto_corr(hb_pos_undriven)
    plt.plot(t, auto_correlation)
    plt.xlabel(r'Time (ms)')
    plt.ylabel(r'Autocorrelation')
    plt.show()