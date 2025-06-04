import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import helpers

if __name__ == '__main__':
    # time arrays
    dt = 1e-3
    ts = (0, 100)
    n = int((ts[-1] - ts[0]) / dt)
    t = np.linspace(ts[0], ts[-1], n)
    time_rescale = 1e-3 # ms -> s

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
    amp = (hb_rescale_params[0] * hb_rescale_params[4]) / (hb_nd_rescale_params[1] * (hb_rescale_params[3] + hb_rescale_params[4]) * hb_rescale_params[1])  * amp # non-dimensional amplitude
    k_sf = forcing_amps[1]

    # read user input for spontaneous oscilaltion frequency and whether to use the steady-state solution for the open-channel probability
    osc_freq_center = 2 * np.pi * float(input("Frequency to center driving at (Non-dimensional): "))
    args_list = (t, x0, list(params), [osc_freq_center, amp, k_sf])

    # multiprocessing and solve sdes
    results = helpers.hb_sols(*args_list) # shape: (T, BATCH_SIZE, d)

    # separate driven and not driven data
    hb_pos_data = np.zeros((results.shape[1], n))
    for i in range(results.shape[1]):
        hb_pos_data[i] = results[:, i, 0]

    # rescale
    sf_pos, omegas = helpers.sf_pos(t, amp, osc_freq_center)
    t_nd = t
    hb_pos, sf_pos, t = helpers.rescale(hb_pos_data, sf_pos, t, *hb_rescale_params, *hb_nd_rescale_params) # rescale hb_pos, sf_pos, and t
    omegas = omegas * t_nd[1] / t[1] # rescale angular frequencies
    dt = float(t[1] - t[0]) # rescale dt

    # get undriven data
    hb_pos_undriven = hb_pos[0, :]

    # get frequency of spontaneous oscillations
    if n % 2 == 0:
        upper_bound = int(n / 2)
    else:
        upper_bound = int((n - 1) / 2) + 1
    freqs = sp.fft.fftfreq(n, dt)
    freqs = freqs[:upper_bound]
    hb_pos0_freq = sp.fft.fft(hb_pos_undriven - np.mean(hb_pos_undriven)) / len(hb_pos_undriven)  # fft for non-driven data
    pos_mags = np.abs(hb_pos0_freq)[:upper_bound]
    peak_index = np.argmax(pos_mags)
    freqs = freqs / time_rescale # rescale from kHz -> Hz
    s_osc_freq = freqs[peak_index] # frequency of spontaneous oscillations
    print(f'Frequency of spontaneous oscillations: {s_osc_freq} Hz. Angular frequency: {2 * np.pi * s_osc_freq} rad/s')

    # get stimulus force
    sf = hb_rescale_params[4] * (hb_pos - sf_pos)

    # get linear response function
    lin_resp_ft = helpers.lin_resp_ft(hb_pos, sf)[:upper_bound]

    # calculate linear response at each driving frequency
    omegas = omegas / time_rescale # rescale back to rad/s
    lin_resp_driving_freq = np.zeros(len(omegas), dtype=complex)
    for i in range(len(lin_resp_driving_freq)):
        index = np.argmin(2 * np.pi * freqs - omegas[i])
        lin_resp_driving_freq[i] = lin_resp_ft[i, index]

    # preliminary plotting
    plt.plot(t, hb_pos_undriven)
    plt.xlabel(r'Time (ms)')
    plt.ylabel(r'$x_{hb, 0}$ (nm)')
    plt.show()

    plt.plot(t, hb_pos[31, :])
    plt.xlabel(r'Time (ms)')
    plt.ylabel(r'$x_{hb, 1}$ (nm)')
    plt.show()

    plt.plot(t[int(n / 4):int(3 * n / 4)], hb_pos_undriven[int(n / 4):int(3 * n / 4)])
    plt.xlabel(r'Time (ms)')
    plt.ylabel(r'$x_{\text{hb}}$ (nm)')
    plt.show()

    plt.plot(freqs[:n // 700], pos_mags[:n // 700])
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'$\tilde{x}_{\text{hb}}(\omega)$')
    plt.show()

    # autocorrelation function
    auto_correlation = helpers.auto_corr(hb_pos_undriven)
    plt.plot(t, auto_correlation)
    plt.xlabel(r'Time (ms)')
    plt.ylabel(r'Autocorrelation')
    plt.show()

    # linear response function
    plt.scatter(omegas / (2 * np.pi), np.real(lin_resp_driving_freq))
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Re{\chi}_{\text{hb}}$')
    plt.show()

    plt.scatter(omegas / (2 * np.pi), np.imag(lin_resp_driving_freq))
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Im{\chi}_{\text{hb}}$')
    plt.show()