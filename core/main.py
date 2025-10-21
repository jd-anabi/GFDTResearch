import os
import re
import sys
from typing import Dict
import multiprocessing as mp

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from core.Helpers import fdt_helpers as fh, gen_helpers as gh, hair_model_helpers as hmh
from core.Simulator import simulator

TIME_RS = 1  # s -> s
PATTERN = re.compile(r'[\s=]+([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)$')  # use pattern matching to extract values (scientific notation)

if __name__ == '__main__':
    # -------------------------- BEGIN SETUP -------------------------- #
    # parameters and initial conditions
    file_str = 'Non-dimensional'
    params = np.zeros(17, dtype=float)
    x0 = []
    rescale_file = np.zeros(8, dtype=float)

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
            val = float(re.findall(PATTERN, row.strip())[0])
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
            val = float(re.findall(PATTERN, row.strip())[0])
            rescale_file[line] = val
            line = line + 1
    hb_rescale_params: Dict[str, float] = {'gamma': rescale_file[0].item(), 'd': rescale_file[1].item(),
                                           'x_sp': rescale_file[2].item(), 'k_sp': rescale_file[3].item(),
                                           'k_gs_max': rescale_file[4].item(), 's_max': rescale_file[5].item(),
                                           't_0': rescale_file[6].item()}
    # need to add in non-dimensional parameters for rescaling too
    hb_rescale_params.update({'s_max_nd': params[6].item(), 'chi_hb': params[12].item(), 'chi_a': params[13].item()})

    # set up forcing params
    file = os.getcwd() + force_file_path
    forcing_amps = []
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(PATTERN, row.strip())[0])
            forcing_amps.append(val)
    amp = forcing_amps[0]
    phase = forcing_amps[1]
    offset = forcing_amps[2]

    # -------------------------- END SETUP -------------------------- #

    # ------------- BEGIN RESCALING CALCULATIONS ------------- #
    # non-dimensional time
    dt = 1e-3
    t_max_nd = int(input("Max time: "))
    ts = (0, t_max_nd)
    n = int((ts[-1] - ts[0]) / dt)
    t_nd = np.linspace(ts[0], ts[-1], n)
    n_time_segs = 5
    time_seg_ids = gh.get_even_ids(len(t_nd), n_time_segs + 1)

    # recaling parameters needed for time and data
    t_rescale_params = [hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'], hb_rescale_params['t_0'],
                        hb_rescale_params['s_max_nd'], hb_rescale_params['chi_a']]
    x_rescale_params = hb_rescale_params['gamma'], hb_rescale_params['d'], hb_rescale_params['x_sp'], hb_rescale_params['chi_hb']

    # rescale time to dimensional
    t = hmh.rescale_t(t_nd, *t_rescale_params)
    t_s = TIME_RS * t  # rescale time
    dt = float(t[1] - t[0]) # rescale dt

    # steady-state index for analysis later
    steady_id = int(0.7 * len(t))
    n = len(t)
    n_steady = len(t[steady_id:])

    # frequency arrays
    freqs = sp.fft.fftfreq(n_steady, dt)
    if n % 2 == 0:
        upper_bound = int(n / 2)
    else:
        upper_bound = int((n - 1) / 2) + 1
    pos_freqs = sp.fft.rfftfreq(n_steady, dt)[1:]
    nperseg = min(int(1 / (dt * pos_freqs[0])), n_steady // 4) # use for welch's method if used later
    # ------------- END RESCALING CALCULATIONS ------------- #

    # ------------- BEGIN FORCE AND FREQUENCY CALCULATIONS ------------- #
    # get frequency of spontaneous oscillation
    low_pos = hmh.irescale_x(0, *x_rescale_params)
    high_pos = hmh.irescale_x(10, *x_rescale_params)
    init_pos = np.random.randint(0, 10, size=(1, 2))
    init_probs = np.random.randint(0, 1, size=(1, 3))
    inits = gh.concat(init_pos, init_probs)  # size: (1, 5)

    dt_0 = 1e-3
    n_0 = int((ts[-1] - ts[0]) / dt_0)
    t_0 = np.linspace(ts[0], ts[-1], n_0)
    #steady_id_0 = int(0.7 * len(t_0))
    args_list = (t_0, inits, list(params), x_rescale_params, t_rescale_params, steady_id)
    omega_center = 2 * np.pi * fh.get_sosc_freq(*args_list)
    print(f'Frequency of spontaneous oscillations: {omega_center / (2 * np.pi)} Hz')
    #omega_center = 2 * np.pi * float(input("Frequency to center driving at (Hz): "))

    # ensemble variables needed
    num_uniq_freqs = 50 # number of unique frequencies
    ensemble_size = 200 # ensemble size for each frequency
    freqs_per_batch = simulator.BATCH_SIZE // ensemble_size # number of frequencies per batch
    iterations = int(num_uniq_freqs / freqs_per_batch)

    # calculate stimulus force position (both models)
    f = fh.force(t, amp, omega_center, phase, offset, num_uniq_freqs)
    f_nd = hmh.irescale_f(f, hb_rescale_params['gamma'], hb_rescale_params['d'],
                          hb_rescale_params['k_sp'], hb_rescale_params['chi_hb'])
    f = f[:, steady_id:]  # steady-state portion of force
    f_driven = f[1:, :]  # ignore undriven force

    # find which index in the array of driving frequencies corresponds to omega_center
    omegas = fh.gen_freqs(omega_center, num_uniq_freqs) # generate frequencies
    omega_center_id = np.argmax(omegas == omega_center)
    nd_f_params = hmh.irescale_f_params(omegas, amp, phase, offset,
                                        hb_rescale_params['gamma'], hb_rescale_params['d'],
                                        hb_rescale_params['k_sp'],
                                        hb_rescale_params['chi_hb'], hb_rescale_params['k_gs_max'],
                                        hb_rescale_params['s_max'],
                                        hb_rescale_params['s_max_nd'], hb_rescale_params['chi_a'],
                                        hb_rescale_params['t_0'])
    omegas_nd, amp_nd, phases_nd, offset_nd = nd_f_params[0], nd_f_params[1], nd_f_params[2], nd_f_params[3]
    # ------------- END FORCE AND FREQUENCY CALCULATIONS ------------- #
    welch = False # don't use Welch's method for the PSD calculations
    onesided = True # use the one-sided PSD
    angular = False # don't divide the PSD by 2 pi

    # instantiate needed values
    x0 = np.zeros(t[steady_id:].shape[0])
    avg_psd = np.zeros(pos_freqs.shape[0])
    avg_psd_at_omegas = np.zeros(omegas.shape[0])
    avg_real_chi = np.zeros(num_uniq_freqs - 1)
    avg_imag_chi = np.zeros(num_uniq_freqs - 1)
    avg_auto_corr = np.zeros(t[steady_id:].shape[0])

    tiled_phases = np.tile(phases_nd, simulator.BATCH_SIZE)  # set up array of phases for each simulation (tiled since total batch size = ensemble size x freqs per batch)
    tiled_omegas = np.tile(omegas_nd, simulator.BATCH_SIZE)  # set up array of omegas for each simulation
    args_list = (t_nd, x0, list(params), tiled_omegas, amp_nd, tiled_phases, offset_nd)  # parameters

    # initial conditions for first batch
    init_pos = np.random.randint(0, 10, size=(simulator.BATCH_SIZE, 2))
    init_probs = np.random.randint(0, 1, size=(simulator.BATCH_SIZE, 3))
    inits = gh.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)

    for iteration in range(iterations):
        print(f"\nIteration: {iteration + 1}")
        low_freq = 0
        chi_id_offset = -1
        #x = np.zeros((fh.BATCH_SIZE, len(t_nd)))
        curr_batch_phases = gh.sde_tile(phases_nd[iteration * freqs_per_batch:(iteration + 1) * freqs_per_batch], ensemble_size, simulator.BATCH_SIZE)
        curr_batch_omegas = gh.sde_tile(omegas_nd[iteration * freqs_per_batch:(iteration + 1) * freqs_per_batch], ensemble_size, simulator.BATCH_SIZE)
        force_params = [curr_batch_omegas, curr_batch_phases, amp_nd, offset_nd]
        x = simulator.sim(t_nd, inits, list(params), force_params, n_time_segs, simulator.BATCH_SIZE, freqs_per_batch)[0]
        '''for tid in range(len(time_seg_ids) - 1):
            curr_time = t_nd[time_seg_ids[tid]:time_seg_ids[tid + 1]]
            args_list = (curr_time, inits, list(params), curr_batch_omegas, amp_nd, curr_batch_phases, offset_nd)
            results = fh.sols(*args_list) # shape: (len(curr_time), BATCH_SIZE, number of variables)

            # update initial conditions
            inits = results[-1, :, :]

            # extract position data
            x[:, time_seg_ids[tid]:time_seg_ids[tid + 1]] = results[:, :, 0].T # shape: (BATCH_SIZE, len(curr_time))
            print(f"Time segment {tid + 1} for iteration {iteration + 1} done")'''
        # rescale position data for later
        x = x.reshape(freqs_per_batch, ensemble_size, len(t_nd)) # shape: (freqs_per_batch, ensemble_size, len(curr_time))
        x = hmh.rescale_x(x, *x_rescale_params)
        x = x[:, :, steady_id:]  # only want to use the steady-state solution
        if iteration == 0:
            x0 = x[0, :, :]
            avg_auto_corr = fh.auto_corr(x0, d=ensemble_size)
            avg_psd = fh.psd(x0, n_steady, dt, int_freqs=pos_freqs, d=ensemble_size)
            avg_psd_at_omegas = fh.psd(x0, n_steady, dt, int_freqs=(omegas / (2 * np.pi)), d=ensemble_size)
            low_freq = 1
            chi_id_offset = 0
        # arguments needed for multiprocessing
        chi_args = [(x[freq, :, :], f[iteration * freqs_per_batch + freq, :], ensemble_size, omegas[iteration * freqs_per_batch + freq - 1].item(), dt) for freq in range(low_freq, freqs_per_batch)]
        #chi_args = [(np.max(x[freq, :, :], axis=1, keepdims=True) - np.mean(x[freq, :, :], axis=1, keepdims=True), amp) for freq in range(low_freq, freqs_per_batch)]
        #chi_args = [(t, x[freq, :, :], f[iteration * freqs_per_batch + freq, :], omegas[iteration * freqs_per_batch + freq - 1].item(), t[-1]) for freq in range(low_freq, freqs_per_batch)]
        with mp.Pool(int(0.75 * mp.cpu_count())) as pool:
            chis = pool.starmap(fh.chi, chi_args)
        chi_id_start = iteration * len(chis) + chi_id_offset
        for chi_id in range(len(chis)):
            avg_real_chi[chi_id_start + chi_id] = np.real(chis[chi_id])
            avg_imag_chi[chi_id_start + chi_id] = np.imag(chis[chi_id])
        inits = gh.concat(init_pos, init_probs)

    # ------------- BEGIN FDT CALCULATIONS ------------- #
    # calculate fluctuation response
    k_b = 1.380649e-23 # m^2 kg s^-2 K^-1
    boltzmann_rescale = 1e24 # nm^2 mg s^-2 K^-1
    temp = hb_rescale_params['k_gs_max'] * hb_rescale_params['d']**2 / (boltzmann_rescale * k_b * params[9].item() * TIME_RS**2)
    theta = fh.fluc_resp(avg_psd_at_omegas[1:], avg_imag_chi, omegas[1:], temp, boltzmann_scale=boltzmann_rescale, onesided=onesided)
    # ------------- END FDT CALCULATIONS ------------- #

    # ------------- BEGIN PLOTTING ------------- #
    t = t[steady_id:]
    # preliminary plotting
    plt.plot(t, x0[0, :])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'$x_{0}$ (nm)')
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
    plt.ylabel(r'Power spectral density ($\frac{\text{nm}^2}{Hz}$)')
    plt.xlim(0.25 * omega_center / (2 * np.pi), 2 * omega_center / (2 * np.pi))
    plt.tight_layout()
    plt.show()

    plt.plot(omegas, avg_psd_at_omegas)
    plt.xlabel(r'Angular frequency (rad/s)')
    plt.ylabel(r'Power spectral density ($\frac{\text{nm}^2}{rad/s}$)')
    plt.xlim(0, omegas[-1])
    plt.tight_layout()
    plt.show()

    # linear response function
    plt.scatter(omegas[1:] / (2 * np.pi), avg_real_chi)
    plt.xlabel(r'Driving Frequency (Hz)')
    plt.ylabel(r'$\Re\{\chi_x\}$')
    plt.tight_layout()
    plt.show()

    plt.scatter(omegas[1:] / (2 * np.pi), avg_imag_chi)
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