import os
import re
import sys
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import helpers

if __name__ == '__main__':
    # ------------- BEGIN SETUP ------------- #
    # time arrays
    dt = 1e-3
    ts = (0, 75)
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
                                           'k_gs_max': hb_rescale_arr[4].item(), 's_max': hb_rescale_arr[5].item(),
                                           't_0': hb_rescale_arr[6].item()}
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
    # ------------- END SETUP ------------- #

    # ------------- BEGIN RESCALING AND DIMENSIONAL FORCE CALCULATIONS ------------- #
    # rescale time
    t = helpers.rescale_t(t_nd, hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'], hb_rescale_params['t_0'],
                          hb_nd_rescale_params['s_max'], hb_nd_rescale_params['chi_a'])

    # calculate stimulus force position (both models)
    sf = helpers.sf(t, amp, sosc)
    sf_nd = helpers.inv_rescale_f(sf, hb_rescale_params['gamma'], hb_rescale_params['d'],
                                  hb_rescale_params['k_sp'], hb_nd_rescale_params['chi_hb'])

    # find which index in the array of driving frequencies corresponds to sosc
    sf_nd_means = np.mean(sf_nd, axis=1)
    omegas = helpers.driving_freqs(sosc)
    print("Driving frequencies: \n", omegas / (2 * np.pi))
    sosc_index = np.argmax(omegas == sosc)
    amp_nd = np.max(sf_nd[sosc_index]) - sf_nd_means[sosc_index]
    sosc_nd = np.arcsin(sf_nd[sosc_index, 1] / amp_nd) / t_nd[1]

    # rescaling time and making an array of driving frequencies
    t = time_rescale * t # rescale from ms -> s
    dt = float(t[1] - t[0]) # rescale dt
    # ------------- END RESCALING AND DIMENSIONAL FORCE CALCULATIONS ------------- #

    # ------------- BEGIN SDE SOLVING AND RETRIEVING NEEDED DATA ------------- #
    # solve sdes
    args_list = (t_nd, x0, list(params), [helpers.driving_freqs(sosc_nd), amp_nd])
    results = helpers.hb_sols(*args_list) # shape: (T, BATCH_SIZE, d)

    # separate driven and not driven data
    hb_pos_data = np.zeros((results.shape[1], n))
    for i in range(results.shape[1]):
        hb_pos_data[i] = results[:, i, 0]

    # rescale hb_pos
    hb_pos = helpers.rescale_x(hb_pos_data, hb_rescale_params['gamma'], hb_rescale_params['d'],
                               hb_rescale_params['x_sp'], hb_nd_rescale_params['chi_hb'])
    mean = np.mean(hb_pos, axis=1)
    for i in range(len(hb_pos)):
        hb_pos[i] = hb_pos[i] - mean[i]

    # get undriven and driven data
    hb_pos_undriven = hb_pos[0, :]
    hb_pos_driven = hb_pos[1:, :]

    # get frequency of spontaneous oscillations
    if n % 2 == 0:
        upper_bound = int(n / 2)
    else:
        upper_bound = int((n - 1) / 2) + 1
    freqs = sp.fft.fftfreq(n, dt)
    freqs = freqs[:upper_bound]
    hb_pos_undriven_freq = sp.fft.fft(hb_pos_undriven - np.mean(hb_pos_undriven)) / len(hb_pos_undriven)  # fft for non-driven data
    hb_pos_driven_freq = sp.fft.fft(hb_pos_driven - np.mean(hb_pos_driven), axis=1) / len(hb_pos_driven) # fft for driven data
    undriven_pos_mags = np.abs(hb_pos_undriven_freq)[:upper_bound]
    driven_pos_mags = np.abs(hb_pos_driven_freq)[:, :upper_bound]
    peak_index = np.argmax(undriven_pos_mags)
    s_osc_freq = freqs[peak_index] # frequency of spontaneous oscillations
    print(f'Frequency of spontaneous oscillations: {s_osc_freq} Hz. Angular frequency: {2 * np.pi * s_osc_freq} rad/s')
    # ------------- END SDE SOLVING AND RETRIEVING NEEDED DATA ------------- #

    # ------------- BEGIN FDT CALCULATIONS ------------- #
    # get linear response function
    # only want the driven data
    omegas = omegas[1:]
    sf = sf[1:, :]
    lin_resp_ft = helpers.lin_resp_ft(hb_pos_driven, sf)[:, :upper_bound]

    # do the same for the autocorrelation function
    autocorr = helpers.auto_corr(hb_pos_undriven)
    autocorr_ft = sp.fft.fft(autocorr - np.mean(autocorr)) / len(autocorr)

    # calculate the autocorrelation function and the linear response at each driving frequency
    autocorr_driving_freq = np.zeros(len(omegas), dtype=complex)
    lin_resp_driving_freq = np.zeros(len(omegas), dtype=complex)
    print('Frequencies of oscillations:')
    for i in range(len(lin_resp_driving_freq)):
        diff = np.abs(2 * np.pi * freqs - omegas[i])
        index = np.argmin(diff)
        print(freqs[index])
        autocorr_driving_freq[i] = autocorr[index]
        lin_resp_driving_freq[i] = lin_resp_ft[i, index]

    # calculate fluctuation response
    k_b = 1.380649e-23 # m^2 kg s^-2 K^-1
    boltzmann_rescale = 1e27 # nm^2 ug s^-2 K^-1
    temp = hb_rescale_params['k_gs_max'] * hb_rescale_params['d']**2 / (boltzmann_rescale * k_b * params[9].item())
    theta = helpers.fluc_resp(autocorr_driving_freq[1:], lin_resp_driving_freq[1:], omegas[1:], temp, boltzmann_rescale)
    # ------------- END FDT CALCULATIONS ------------- #

    # ------------- BEGIN PLOTTING ------------- #
    # preliminary plotting
    plt.plot(t, hb_pos_undriven)
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{\omega = \omega_0}$ (nm)')
    plt.show()

    plt.plot(t, hb_pos_driven[0, :])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{\omega = \omega_D}$ (nm)')
    plt.show()

    plt.plot(t[int(n / 4):int(3 * n / 4)], hb_pos_undriven[int(n / 4):int(3 * n / 4)])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{\omega = \omega_0}$ (nm)')
    plt.show()

    plt.plot(t, sf[0])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$F(t) (\text{ug nm} text{ ms}^{-2})$')
    plt.show()

    plt.plot(freqs[:n // 700], undriven_pos_mags[:n // 700])
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'$\tilde{x}_0(\omega)$')
    plt.show()

    # autocorrelation function
    plt.plot(t, autocorr)
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Autocorrelation')
    plt.show()

    # linear response function
    plt.scatter(omegas / (2 * np.pi), lin_resp_driving_freq.real)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Re\{\chi_x\}$')
    plt.show()

    plt.scatter(omegas / (2 * np.pi), lin_resp_driving_freq.imag)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Im\{\chi_x\}$')
    plt.show()

    plt.scatter(omegas[1:] / (2 * np.pi), theta)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\theta(\omega)$')
    plt.show()
    # ------------- END PLOTTING ------------- #