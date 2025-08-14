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
    # damped harmonic oscillator parameters
    parameters = [1, 0.1, 1, 3] # mass, gamma, omega_0, temperature
    # time arrays
    dt = 1e-2
    t_equilibrium = 50 / parameters[1]
    t_max = 2 * np.pi * 100 / parameters[2] + t_equilibrium
    ts = (0, t_max)
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
    phase = forcing_amps[1]
    offset = forcing_amps[2]

    # read user input for spontaneous oscillation frequency
    #sosc = 2 * np.pi * float(input("Frequency to center driving at (Hz): "))
    sosc = parameters[2]
    # ------------- END SETUP ------------- #

    # ------------- BEGIN RESCALING AND DIMENSIONAL FORCE CALCULATIONS ------------- #
    # rescale time
    t = helpers.rescale_t(t_nd, hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'], hb_rescale_params['t_0'],
                          hb_nd_rescale_params['s_max'], hb_nd_rescale_params['chi_a'])

    # rescaling time and making an array of driving frequencies
    t = time_rescale * t # rescale from ms -> s
    t = t_nd
    dt = float(t[1] - t[0]) # rescale dt

    steady_id = int(0.7 * len(t))
    steady_id = int(t_equilibrium / dt)
    t = t[steady_id:]  # take last 30% for steady state
    t = t - t[0]
    n = len(t)

    # frequency arrays
    fs = 70
    freqs = sp.fft.fftfreq(n, dt)
    if n % 2 == 0:
        upper_bound = int(n / 2)
    else:
        upper_bound = int((n - 1) / 2) + 1
    pos_freqs = freqs[1:upper_bound]

    nperseg_needed = int(1 / (dt * pos_freqs[0]))
    nperseg = min(nperseg_needed, n // 4)

    ensemble_size_per_batch = 10
    rep_num = int(helpers.BATCH_SIZE / ensemble_size_per_batch)

    # calculate stimulus force position (both models)
    sf = helpers.sf(t, amp, sosc, phase, offset, rep_num)
    sf_nd = helpers.inv_rescale_f(sf, hb_rescale_params['gamma'], hb_rescale_params['d'],
                                  hb_rescale_params['k_sp'], hb_nd_rescale_params['chi_hb'])

    # find which index in the array of driving frequencies corresponds to sosc
    omegas = helpers.driving_freqs(sosc, rep_num)
    #print("Driving frequencies: \n", omegas / (2 * np.pi))
    sosc_index = np.argmax(omegas == sosc)
    force_params_nd = helpers.rescale_force_params(amp, omegas, phase, offset,
                                                   hb_rescale_params['gamma'], hb_rescale_params['d'], hb_rescale_params['k_sp'],
                                                   hb_nd_rescale_params['chi_hb'], hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'],
                                                   hb_nd_rescale_params['s_max'], hb_nd_rescale_params['chi_a'], hb_rescale_params['t_0'])
    amp_nd, omegas_nd, phases_nd, offset_nd = force_params_nd[0], force_params_nd[1], force_params_nd[2], force_params_nd[3]

    # ------------- END RESCALING AND DIMENSIONAL FORCE CALCULATIONS ------------- #

    # let's do ensemble size of 10 each iteration; so total batch size = rep_num * ensemble_batch_size = 400 * 10 = 4000
    tiled_omegas = np.tile(omegas, ensemble_size_per_batch)

    # solve sdes
    x0 = [0.1, 0.0]
    num_iterations = 1
    num_vars = 2
    args_list = (t_nd, x0, list(parameters), [tiled_omegas, amp, phase, offset])
    avg_results = np.zeros((n, rep_num, num_vars))
    avg_psd_at_omegas = None
    avg_chis = None
    pos_magnitudes = None
    avg_auto_corr = None
    avg_psd = None
    for iteration in range(num_iterations):
        # ------------- BEGIN SDE SOLVING AND RETRIEVING NEEDED DATA ------------- #
        results = helpers.hb_sols(*args_list) # shape: (T, BATCH_SIZE, d)
        #results = results + temp_results.reshape(temp_results.shape[0], -1, ensemble_batch_size, temp_results.shape[2]).mean(axis=2)

        x_data = results[:, :, 0].T # (BATCH_SIZE, T)

        # get steady-state portion of data and stimulus force
        x_data = x_data[:, steady_id:]
        sf = sf[:, steady_id:]

        # subtract off the average
        for i in range(x_data.shape[0]):
            x_data[i] = x_data[i] - np.mean(x_data[i])

        # seperate undriven and driven data (noting every rep_num of batches is a new simulation)
        x0 = x_data[::rep_num, :] # every rep_num repetitions is a new simulation for the same frequency
        id0 = np.arange(0, x_data.shape[0], rep_num) # indices of the undriven simulations
        x = np.delete(x_data, id0, axis=0)
        # ------------- END SDE SOLVING AND RETRIEVING NEEDED DATA ------------- #

        # frequency space
        x0_f = sp.fft.fft(x0, axis=1) / len(x0)
        avg_x0_f = np.mean(x0_f, axis=0) # average the fourier transforms
        magnitudes = np.abs(avg_x0_f)
        pos_magnitudes = magnitudes[1:upper_bound]
        spon_osc_freq = pos_freqs[np.argmax(pos_magnitudes)]

        # ------------- BEGIN AVERAGING CALCULATIONS ------------- #
        # calculate the autocorrelations then average it
        auto_corrs = np.zeros_like(x0)
        for i in range(auto_corrs.shape[0]):
            auto_corrs[i] = helpers.auto_corr(x0[i])
        avg_auto_corr = np.mean(auto_corrs, axis=0)

        # spectral density
        psd = np.zeros((x0.shape[0], pos_freqs.shape[0]))
        for i in range(psd.shape[0]):
            psd[i] = helpers.psd(x0[i], dt, pos_freqs, nperseg)
        avg_psd = np.mean(psd, axis=0)

        psd_at_omegas = np.zeros((x0.shape[0], omegas.shape[0]))
        for i in range(psd_at_omegas.shape[0]):
            psd_at_omegas[i] = helpers.psd(x0[i], dt, omegas / (2 * np.pi), nperseg) # in units of rad / s
        avg_psd_at_omegas = np.mean(psd_at_omegas, axis=0)

        # stimulus force and frequencies information
        f_driven = sf[1:, :]
        tiled_f_driven = np.tile(f_driven, (ensemble_size_per_batch, 1))
        omegas_driven = omegas[1:]

        # linear response
        chis = helpers.chi_ft(x, tiled_f_driven)[:, 1:upper_bound]
        avg_chis = chis.reshape(-1, rep_num, chis.shape[1]).mean(axis=1)
        # ------------- END AVERAGING CALCULATIONS ------------- #
        '''
        # separate driven and not driven data
        hb_pos_data = np.zeros((results.shape[1], n))
        for i in range(results.shape[1]):
            hb_pos_data[i] = results[:, i, 0]

        # rescale hb_pos
        hb_pos = helpers.rescale_x(hb_pos_data, hb_rescale_params['gamma'], hb_rescale_params['d'],
                                   hb_rescale_params['x_sp'], hb_nd_rescale_params['chi_hb'])
        hb_pos = hb_pos_data
        hb_pos = hb_pos[:, steady_id:]
        sf = sf[:, idx:]
        mean = np.mean(hb_pos, axis=1)
        for i in range(len(hb_pos)):
            hb_pos[i] = hb_pos[i] - mean[i]

        # get undriven and driven data
        hb_pos_undriven = hb_pos[0, :]
        x0 = hb_pos[::rep_num]
        hb_pos_driven = hb_pos[1:, :]
        xd = np.zeros((results.shape[1] - ensemble_size_per_batch, n))
        for i in range(xd.shape[0]):
            xd[i] = hb_pos[(1 + i)::rep_num]

        # get frequency of spontaneous oscillations
        hb_pos_undriven_freq = sp.fft.fft(hb_pos_undriven - np.mean(hb_pos_undriven)) / len(hb_pos_undriven)  # fft for non-driven data
        hb_pos_driven_freq = sp.fft.fft(hb_pos_driven - np.mean(hb_pos_driven), axis=1) / len(hb_pos_driven) # fft for driven data
        undriven_pos_mags = np.abs(hb_pos_undriven_freq)[1:upper_bound]
        driven_pos_mags = np.abs(hb_pos_driven_freq)[:, 1:upper_bound]
        peak_index = np.argmax(undriven_pos_mags)
        s_osc_freq = pos_freqs[peak_index] # frequency of spontaneous oscillations
        print(f'Frequency of spontaneous oscillations: {s_osc_freq} Hz. Angular frequency: {2 * np.pi * s_osc_freq} rad/s')
        # ------------- END SDE SOLVING AND RETRIEVING NEEDED DATA ------------- #

        # ------------- BEGIN FDT CALCULATIONS ------------- #
        # get autocorrelation function
        autocorr = helpers.auto_corr(hb_pos_undriven)

        # angular frequencies and stimulus force
        omegas = omegas[1:]
        sf = sf[1:, :]

        #psd = sp.fft.fft(autocorr - np.mean(autocorr)) / len(autocorr)
        #psd = psd[:upper_bound]
        nperseg = min(nperseg_needed, len(hb_pos_undriven) // 4)
        chi = helpers.chi_ft(hb_pos_driven, sf)[:, 1:upper_bound]
        psd = helpers.psd(hb_pos_undriven, dt, pos_freqs, nperseg)

        # calculate the autocorrelation function and the linear response at each driving frequency
        psd_df = helpers.psd(hb_pos_undriven, dt, omegas / (2 * np.pi), nperseg) / (2 * np.pi) # in units of rad/s
        chi_df = np.zeros(len(omegas), dtype=complex)
        for i in range(len(chi_df)):
            diff = np.abs(2 * np.pi * pos_freqs - omegas[i])
            index = np.argmin(diff)
            chi_df[i] = chi[i, index]
        '''

        print("Number of ensembles done: ", iteration + 1)

    chi_at_omegas = np.zeros(len(omegas), dtype=complex)
    for i in range(avg_chis.shape[0]):
        diff = np.abs(2 * np.pi * pos_freqs - omegas[i])
        index = np.argmin(diff)
        chi_at_omegas[i] = avg_chis[i, index]

    # ------------- BEGIN FDT CALCULATIONS ------------- #
    # calculate fluctuation response
    k_b = 1.380649e-23 # m^2 kg s^-2 K^-1
    boltzmann_rescale = 1e18 # nm^2 mg ms^-2 K^-1
    temp = hb_rescale_params['k_gs_max'] * hb_rescale_params['d']**2 / (boltzmann_rescale * k_b * params[9].item())
    theta = helpers.fluc_resp(avg_psd_at_omegas[1:], chi_at_omegas[1:], omegas[1:], temp, boltzmann_rescale)
    # ------------- END FDT CALCULATIONS ------------- #

    # ------------- BEGIN PLOTTING ------------- #
    # preliminary plotting
    plt.plot(t, x0[0])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{0}$ (nm)')
    plt.tight_layout()
    plt.show()

    # entrainment plotting
    r'''
    fig, ax = plt.subplots(3, 3, figsize=(56, 40))
    for i in range(3):
        for j in range(3):
            iteration = 3*i + j
            index = sosc_index + iteration
            if index >= len(omegas):
                index = sosc_index
            curr_omega = round(omegas[index].item(), 3)
            ax[i, j].plot(t, x[index, :], color='b')
            ax[i, j].set_xlabel(r'Time (s)')
            ax[i, j].set_ylabel(rf'$x$ (nm)', color='b')
            ax[i, j].tick_params(axis='y')
            ax[i, j].set_title(rf'$\omega = {curr_omega}$ rad/s')
            #ax[i, j].set_xlim(0.02, 0.05)
            axf = ax[i, j].twinx()
            axf.plot(t, sf[index], color='r')
            axf.set_ylabel(rf'$F(t) = {amp} \sin({curr_omega} \ t + {phase}) + {offset}$ (mg nm ms$^-2$)', color='r')
            axf.tick_params(axis='y')
    plt.tight_layout()
    plt.show()
    '''

    plt.plot(pos_freqs, pos_magnitudes)
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'$\tilde{x}_0(\omega)$')
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.show()

    # autocorrelation function
    plt.plot(t, avg_auto_corr)
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Autocorrelation')
    plt.tight_layout()
    plt.show()

    # Power spectral density
    plt.plot(pos_freqs, avg_psd)
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'Power spectral density')
    plt.xlim(0, 0.7)
    plt.tight_layout()
    plt.show()

    # linear response function
    plt.scatter(omegas / (2 * np.pi), chi_at_omegas.real)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Re\{\chi_x\}$')
    plt.tight_layout()
    plt.show()

    plt.scatter(omegas / (2 * np.pi), chi_at_omegas.imag)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Im\{\chi_x\}$')
    plt.tight_layout()
    plt.show()

    plt.scatter(omegas[1:] / (2 * np.pi), theta)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\theta(\omega)$')
    plt.hlines(1, omegas[1] / (2 * np.pi), omegas[-1] / (2 * np.pi), linestyle='--', color='r')
    plt.tight_layout()
    plt.show()

    plt.scatter(omegas[1:] / (2 * np.pi), theta)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\theta(\omega)$')
    plt.tight_layout()
    plt.show()
    # ------------- END PLOTTING ------------- #