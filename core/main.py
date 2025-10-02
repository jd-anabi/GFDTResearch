import os
import re
import sys
from typing import Dict
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import fdt_helpers as fh
import hair_model_helpers as hmh

if __name__ == '__main__':
    # ------------- BEGIN SETUP ------------- #
    # time arrays
    dt = 1e-3
    t_max = int(input("Number of cycles: "))
    ts = (0, t_max)
    n = int((ts[-1] - ts[0]) / dt)
    t_nd = np.linspace(ts[0], ts[-1], n)
    dt = t_nd[1] - t_nd[0]

    # parameters and initial conditions
    file_str = 'Non-dimensional'
    params = np.zeros(17, dtype=float)
    x0 = []
    rescale_file = np.zeros(8, dtype=float)

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
            rescale_file[line] = val
            line = line + 1
    hb_rescale_params: Dict[str, float] = {'gamma': rescale_file[0].item(), 'd': rescale_file[1].item(),
                                           'x_sp': rescale_file[2].item(), 'k_sp': rescale_file[3].item(),
                                           'k_gs_max': rescale_file[4].item(), 's_max': rescale_file[5].item(),
                                           't_0': rescale_file[6].item()}
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
    sosc = 2 * np.pi * float(input("Frequency to center driving at (Hz): "))
    # ------------- END SETUP ------------- #

    # ------------- BEGIN RESCALING AND DIMENSIONAL FORCE CALCULATIONS ------------- #
    # rescale time
    t = hmh.rescale_t(t_nd, hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'], hb_rescale_params['t_0'],
                          hb_nd_rescale_params['s_max'], hb_nd_rescale_params['chi_a'])

    # rescaling time and making an array of driving frequencies
    time_rescale = 1e-3 # ms -> ms
    t_s = time_rescale * t # rescale from ms -> s
    dt_s = float(t[1] - t[0]) # rescale dt
    fs = 1 / dt_s

    steady_id = int(0.3 * len(t))
    n = len(t)
    n_steady = len(t[steady_id:])

    # frequency arrays
    freqs = sp.fft.fftfreq(n_steady, dt)
    if n % 2 == 0:
        upper_bound = int(n / 2)
    else:
        upper_bound = int((n - 1) / 2) + 1
    pos_freqs = sp.fft.rfftfreq(n_steady, dt)[1:]

    nperseg = min(int(1 / (dt * pos_freqs[0])), n_steady // 4)

    num_unique = 40
    ensemble_size_per_iter = fh.BATCH_SIZE // num_unique

    # calculate stimulus force position (both models)
    f = fh.sf(t, amp, sosc, phase, offset, num_unique)
    f_nd = hmh.inv_rescale_f(f, hb_rescale_params['gamma'], hb_rescale_params['d'],
                                 hb_rescale_params['k_sp'], hb_nd_rescale_params['chi_hb'])
    f = f[:, steady_id:] # steady-state portion of force
    f_driven = f[1:, :] # ignore undriven force

    # find which index in the array of driving frequencies corresponds to sosc
    omegas = fh.driving_freqs(sosc, num_unique)
    sosc_index = np.argmax(omegas == sosc)
    force_params_nd = hmh.rescale_force_params(omegas / time_rescale, amp, phase, offset,
                                                   hb_rescale_params['gamma'], hb_rescale_params['d'], hb_rescale_params['k_sp'],
                                                   hb_nd_rescale_params['chi_hb'], hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'],
                                                   hb_nd_rescale_params['s_max'], hb_nd_rescale_params['chi_a'], hb_rescale_params['t_0'])
    omegas_nd, amp_nd, phases_nd, offset_nd = force_params_nd[0], force_params_nd[1], force_params_nd[2], force_params_nd[3]

    # ------------- END RESCALING AND DIMENSIONAL FORCE CALCULATIONS ------------- #

    # let's do ensemble size of 10 each iteration; so total batch size = num_iterations x ensemble_size_per_iter = 400 x 10 = 4000
    tiled_phases = np.tile(phases_nd, ensemble_size_per_iter)
    tiled_omegas = np.tile(omegas_nd, ensemble_size_per_iter)

    # instantiate needed values
    avg_psd = np.zeros(pos_freqs.shape[0])
    avg_psd_at_omegas = np.zeros(omegas.shape[0])
    avg_real_chi = np.zeros((num_unique - 1, pos_freqs.shape[0]))
    avg_imag_chi = np.zeros((num_unique - 1, pos_freqs.shape[0]))
    pos_magnitudes = np.zeros(pos_freqs.shape[0])
    #avg_auto_corr = np.zeros(t[steady_id:].shape[0])

    # solve ensemble of SDEs
    x0 = np.random.randint(0, 1, size=(fh.BATCH_SIZE, 5)) # initial conditions
    num_iterations = 3 # total ensemble size = ensemble_size_per_iter x num_iterations
    args_list = (t_nd, x0, list(params), [tiled_omegas, amp_nd, tiled_phases, offset_nd]) # parameters
    welch = False
    onesided = True
    angular = False
    for iteration in range(num_iterations):
        # ------------- BEGIN SDE SOLVING AND RETRIEVING NEEDED DATA ------------- #
        results = fh.hb_sols(*args_list) # shape: (T, BATCH_SIZE, d)

        x_data = results[:, :, 0].T # (BATCH_SIZE, T)
        x_data = hmh.rescale_x(x_data, hb_rescale_params['gamma'], hb_rescale_params['d'], hb_rescale_params['x_sp'], hb_nd_rescale_params['chi_hb'])

        # get steady-state portion of data
        x_data = x_data[:, steady_id:]

        x_data = x_data.reshape(ensemble_size_per_iter, num_unique, -1) # (ensemble_size_per_iter, num_unique, T)

        # seperate undriven and driven data (noting every num_unique of batches is a new simulation)
        x0 = x_data[:, 0] # (ensemble_size_per_iter, T)
        x = x_data[:, 1:] # (ensemble_size_per_iter, num_unique - 1, T)
        # ------------- END SDE SOLVING AND RETRIEVING NEEDED DATA ------------- #

        # ------------- BEGIN AVERAGING CALCULATIONS ------------- #
        with mp.Pool(processes=mp.cpu_count()//2) as pool:
            #avg_auto_corr += np.mean(np.array(pool.starmap(helpers.auto_corr, zip(x0))), axis=0)
            avg_psd += np.mean(np.array(pool.starmap(fh.psd, [(x0[i], n_steady, dt_s, fs, pos_freqs, welch, nperseg, onesided, angular) for i in range(x0.shape[0])])), axis=0)
            avg_psd_at_omegas += np.mean(np.array(pool.starmap(fh.psd, [(x0[i], n_steady, dt_s, fs, omegas / (2 * np.pi), welch, nperseg, onesided, angular) for i in range(x0.shape[0])])), axis=0)
            chis = np.mean(np.array(pool.starmap(fh.chi_ft, [(x[i], f_driven) for i in range(x.shape[0])])), axis=0)[:, 1:]
            avg_real_chi += np.real(chis)
            avg_imag_chi += np.imag(chis)
        # ------------- END AVERAGING CALCULATIONS ------------- #

        # frequency space
        # subtract off the average
        # for i in range(x_data.shape[0]):
        #    x_data[i] = x_data[i] - np.mean(x_data[i])
        print("Number of ensembles done: ", iteration + 1)

    avg_psd_at_omegas /= num_iterations
    avg_real_chi /= num_iterations
    avg_imag_chi /= num_iterations

    real_chi_at_omegas = np.zeros(len(omegas) - 1, dtype=float)
    imag_chi_at_omegas = np.zeros(len(omegas) - 1, dtype=float)
    for i in range(len(omegas) - 1):
        diff = np.abs(2 * np.pi * pos_freqs - omegas[i + 1])
        index = np.argmin(diff)
        real_chi_at_omegas[i] = avg_real_chi[i, index]
        imag_chi_at_omegas[i] = avg_imag_chi[i, index]

    # ------------- BEGIN FDT CALCULATIONS ------------- #
    # calculate fluctuation response
    print(omegas[1:])
    k_b = 1.380649e-23 # m^2 kg s^-2 K^-1
    boltzmann_rescale = 1e24 # nm^2 mg s^-2 K^-1
    temp = hb_rescale_params['k_gs_max'] * hb_rescale_params['d']**2 / (boltzmann_rescale * k_b * params[9].item() * time_rescale**2)
    theta = fh.fluc_resp(avg_psd_at_omegas[1:], imag_chi_at_omegas, omegas[1:], temp, onesided=onesided)
    # ------------- END FDT CALCULATIONS ------------- #

    # ------------- BEGIN PLOTTING ------------- #
    t = t[steady_id:]
    # preliminary plotting
    plt.plot(t, x0[0])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{0}$ (nm)')
    plt.tight_layout()
    plt.show()

    #plt.plot(pos_freqs, pos_magnitudes)
    #plt.xlabel(r'Frequency (Hz)')
    #plt.ylabel(r'$\tilde{x}_0(\omega)$')
    #plt.xlim(0, 5)
    #plt.tight_layout()
    #plt.show()

    # autocorrelation function
    #plt.plot(t, avg_auto_corr)
    #plt.xlabel(r'Time (s)')
    #plt.ylabel(r'Autocorrelation')
    #plt.tight_layout()
    #plt.show()

    # Power spectral density
    plt.plot(pos_freqs, avg_psd)
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel(r'Power spectral density')
    plt.xlim(0, 3)
    plt.tight_layout()
    plt.show()

    plt.plot(omegas, avg_psd_at_omegas)
    plt.xlabel(r'Angular frequency (rad/s)')
    plt.ylabel(r'Power spectral density')
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.show()

    # linear response function
    plt.scatter(omegas[1:] / (2 * np.pi), real_chi_at_omegas)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Re\{\chi_x\}$')
    plt.tight_layout()
    plt.show()

    plt.scatter(omegas[1:] / (2 * np.pi), imag_chi_at_omegas)
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

    y_scale_range = 0
    while True:
        try:
            y_scale_range = float(input("Enter the range of y scale: "))
        except ValueError:
            print("Invalid input")
            break
        plt.scatter(omegas[1:] / (2 * np.pi), theta)
        plt.xlabel(r'Driving Frequency (Hz)')
        plt.ylabel(r'$\theta(\omega)$')
        plt.hlines(1, omegas[1] / (2 * np.pi), omegas[-1] / (2 * np.pi), linestyle='--', color='r')
        plt.ylim(-y_scale_range, y_scale_range)
        plt.tight_layout()
        plt.show()
    # ------------- END PLOTTING ------------- #