import os
import re
import sys
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import helpers

if __name__ == '__main__':
    # time arrays
    dt = 1e-3
    ts = (0, 100)
    n = int((ts[-1] - ts[0]) / dt)
    t_nd = np.linspace(ts[0], ts[-1], n)
    time_rescale = 1e-3 # ms -> s

    # parameters and initial conditions
    file_str = 'Non-dimensional'
    params = np.zeros(17, dtype=float)
    x0 = []
    hb_rescale_arr = np.zeros(8, dtype=float)

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
            hb_rescale_arr[line] = val
            line = line + 1
    hb_rescale_params: Dict[str, float] = {'gamma': hb_rescale_arr[0].item(), 'd': hb_rescale_arr[1].item(),
                                           'x_sp': hb_rescale_arr[2].item(), 'k_sp': hb_rescale_arr[3].item(),
                                           'k_sf': hb_rescale_arr[4].item(), 'k_gs_max': hb_rescale_arr[5].item(),
                                           's_max': hb_rescale_arr[6].item(), 't_0': hb_rescale_arr[7].item()}
    # need to add in non-dimensional parameters for rescaling too
    hb_nd_rescale_params: Dict[str, float] = {'s_max': params[6].item(), 'chi_hb': params[12].item(), 'chi_a': params[13].item()}

    # set up forcing params
    file = os.getcwd() + force_file_path
    forcing_amps = []
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            forcing_amps.append(val)
    amp = forcing_amps[0]

    # read user input for spontaneous oscillation frequency
    sosc = 2 * np.pi * float(input("Frequency to center driving at (Hz): "))

    # rescale time
    t = helpers.rescale_t(t_nd, hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'], hb_rescale_params['t_0'],
                          hb_nd_rescale_params['s_max'], hb_nd_rescale_params['chi_a'])

    # calculate stimulus force position (both models)
    x_sf = helpers.sf_pos(t, amp, sosc)
    x_sf_nd = helpers.inv_rescale_f(x_sf, hb_rescale_params['gamma'], hb_rescale_params['d'],
                                    hb_rescale_params['k_sp'], hb_rescale_params['k_sf'],
                                    hb_nd_rescale_params['chi_hb'])

    # find which index in the array of driving frequencies corresponds to sosc
    xsf_nd_means = np.mean(x_sf_nd, axis=1)
    omegas = helpers.driving_freqs(sosc)
    sosc_index = np.argmax(omegas == sosc)
    amp_nd = np.max(x_sf_nd[sosc_index]) - xsf_nd_means[sosc_index]
    sosc_nd = np.arcsin(x_sf_nd[sosc_index, 1] / amp_nd) / t[1]

    # rescaling time and making an array of driving frequencies
    t = time_rescale * t # rescale from ms -> s
    dt = float(t[1] - t[0]) # rescale dt

    # solve sdes
    args_list = (t_nd, x0, list(params), [helpers.driving_freqs(sosc_nd), amp_nd])
    results = helpers.hb_sols(*args_list) # shape: (T, BATCH_SIZE, d)

    # separate driven and not driven data
    hb_pos_data = np.zeros((results.shape[1], n))
    for i in range(results.shape[1]):
        hb_pos_data[i] = results[:, i, 0]

    # rescale hb_pos
    hb_pos = helpers.rescale_x(hb_pos_data, hb_rescale_params['gamma'], hb_rescale_params['d'],
                               hb_rescale_params['x_sp'], hb_rescale_params['k_sp'], hb_rescale_params['k_sf'],
                               hb_nd_rescale_params['chi_hb'])
    mean = np.mean(hb_pos, axis=1)
    for i in range(len(hb_pos)):
        hb_pos[i] = hb_pos[i] - mean[i]

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
    s_osc_freq = freqs[peak_index] # frequency of spontaneous oscillations
    print(f'Frequency of spontaneous oscillations: {s_osc_freq} Hz. Angular frequency: {2 * np.pi * s_osc_freq} rad/s')

    # get stimulus force
    sf = hb_rescale_params['k_sf'] * (hb_pos - x_sf)

    # get linear response function
    lin_resp_ft = helpers.lin_resp_ft(hb_pos, sf)[:upper_bound]

    # calculate linear response at each driving frequency
    lin_resp_driving_freq = np.zeros(len(omegas), dtype=complex)
    for i in range(len(lin_resp_driving_freq)):
        diff = 2 * np.pi * freqs - omegas[i]
        diff = np.where(diff < 0, np.nan, diff)
        index = np.nanargmin(diff)
        lin_resp_driving_freq[i] = lin_resp_ft[i, index]

    # preliminary plotting
    plt.plot(t, hb_pos_undriven)
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{hb, \omega = \omega_0}$ (nm)')
    plt.show()

    plt.plot(t, hb_pos[-1, :])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{hb, \omega = \omega_D}$ (nm)')
    plt.show()

    plt.plot(t[int(n / 4):int(3 * n / 4)], hb_pos_undriven[int(n / 4):int(3 * n / 4)])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{\text{hb, \omega = \omega_0}}$ (nm)')
    plt.show()

    plt.plot(freqs[:n // 700], pos_mags[:n // 700])
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'$\tilde{x}_{\text{hb}}(\omega)$')
    plt.show()

    # autocorrelation function
    auto_correlation = helpers.auto_corr(hb_pos_undriven)
    plt.plot(t, auto_correlation)
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Autocorrelation')
    plt.show()

    # linear response function
    plt.scatter(omegas / (2 * np.pi), lin_resp_driving_freq.real)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Re{\chi}_{\text{hb}}$')
    plt.show()

    plt.scatter(omegas / (2 * np.pi), lin_resp_driving_freq.imag)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Im{\chi}_{\text{hb}}$')
    plt.show()