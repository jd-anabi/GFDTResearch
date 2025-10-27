import os
import re
import sys
import math
from typing import Dict
import multiprocessing as mp

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pint
import torch
import numpy as np
import scipy as sp

from core.Helpers import fdt_helpers as fh, gen_helpers as gh, hair_model_helpers as hmh
from core.Simulator import simulator

SN_PATTERN = re.compile(r'[\s=]+([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)$')  # use pattern matching to extract values (scientific notation)
PAR_PATTERN = re.compile(r'\((.*?)\)') # use pattern matching to extract value within parentheses
UNIT_PATTERN = re.compile(r'[a-zA-Z]+') # use pattern matching to extract units

if __name__ == '__main__':
    # -------------------------- BEGIN SETUP -------------------------- #
    # parameters and initial conditions
    params = np.zeros(17, dtype=float)
    rescale_file = np.zeros(8, dtype=float)
    forcing_params = np.zeros(3, dtype=float)
    units = []
    x0 = []

    # construct OS dependent directory for model parameters directory
    if sys.platform == 'win32':
        model_params_dir = '\\Model Parameters\\'
    else:
        model_params_dir = '/Model Parameters/'

    # list files in directory
    model_files = ['']
    file_num = 1
    direct = os.getcwd() + '\\Model Parameters' if sys.platform == 'win32' else os.getcwd() + '/Model Parameters'
    for root, dirs, files in os.walk(direct):
        level = root.replace(direct, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            model_files.append(file)
            print(f"{subindent}({file_num}) {file}")
            file_num += 1
    model_files.pop(0)

    # read in model parameters
    line = 0
    invalid_lines = (0, 23, 24, 33, 34)
    file_num = int(input('File number for model parameters: '))
    file = os.getcwd() + model_params_dir + model_files[file_num - 1]
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            if line not in invalid_lines:
                # non-dimensional parameters
                if line <= 22:
                    val = float(re.findall(SN_PATTERN, row.strip())[0])
                    if line <= 5:
                        x0.append(val)
                    else:
                        params[line - 6] = val
                # dimensional parameters
                elif line <= 32:
                    val = float(re.findall(SN_PATTERN, row.strip())[0])
                    curr_units = [unit for par in re.findall(PAR_PATTERN, row.strip()) for unit in
                                  re.findall(UNIT_PATTERN, par)]
                    rescale_file[line - 25] = val
                    for unit in curr_units:
                        if unit not in units:
                            units.append(unit)
                # forcing parameters
                else:
                    val = float(re.findall(SN_PATTERN, row.strip())[0])
                    forcing_params[line - 35] = val
            line = line + 1
    amp = forcing_params[0]
    phase = forcing_params[1]
    offset = forcing_params[2]

    # set up dictionary for model parameters and rescaling parameters
    parameters: Dict[str, float] = {'tau_hb': params[0].item(), 'tau_m': params[1].item(), 'tau_gs': params[2].item(), 'tau_t': params[3].item(),
                                    'c_min': params[4].item(), 's_min': params[5].item(), 's_max': params[6].item(), 'ca2_m': params[7].item(),
                                    'ca2_gs': params[8].item(), 'u_gs_max': params[9].item(), 'delta_e': params[10].item(), 'k_gs_ratio': params[11].item(),
                                    'chi_hb': params[12].item(), 'chi_a': params[13].item(), 'x_c': params[14].item(), 'eta_hb': params[15].item(),
                                    'eta_a': params[16].item()}
    hb_rescale_params: Dict[str, float] = {'gamma': rescale_file[0].item(), 'd': rescale_file[1].item(),
                                           'x_sp': rescale_file[2].item(), 'k_sp': rescale_file[3].item(),
                                           'k_gs_max': rescale_file[4].item(), 's_max': rescale_file[5].item(),
                                           't_0': rescale_file[6].item(), 'alpha': rescale_file[7].item()}
    hb_rescale_params.update({'s_max_nd': params[6].item(), 'chi_hb': params[12].item(), 'chi_a': params[13].item()}) # need to add in non-dimensional parameters for rescaling too

    # need to construct dictionary now that converts current units to SI units
    ureg = pint.UnitRegistry()
    try:
        factors = [ureg(unit).to_base_units().magnitude for unit in units]
        rescale_factors: Dict[str, float] = {'distance': factors[0], 'mass': factors[1], 'time': factors[2]}
    except pint.UndefinedUnitError as e:
        print(f'Error: {e}. Unrecognized units.')
        exit()
    # -------------------------- END SETUP -------------------------- #

    # ------------- BEGIN RESCALING CALCULATIONS ------------- #
    # non-dimensional time
    dt = 1e-3
    t_max_nd = int(input("Max time: "))
    ts = (0, t_max_nd)
    n = int((ts[-1] - ts[0]) / dt)
    t_nd = np.linspace(ts[0], ts[-1], n)
    n_time_segs = math.ceil(t_max_nd / 200)
    time_seg_ids = gh.get_even_ids(len(t_nd), n_time_segs + 1)

    # recaling parameters needed for time and data
    t_rescale_params = [hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'], hb_rescale_params['t_0'],
                        hb_rescale_params['s_max_nd'], hb_rescale_params['chi_a']]
    x_rescale_params = [hb_rescale_params['gamma'], hb_rescale_params['d'], hb_rescale_params['x_sp'],
                        hb_rescale_params['k_sp'], hb_rescale_params['alpha'], hb_rescale_params['chi_hb']]

    # rescale time to dimensional
    t = hmh.rescale_t(t_nd, *t_rescale_params)
    dt = float(t[1] - t[0]) # rescale dt

    # steady-state index for analysis later
    steady_id = int(0.4 * len(t))
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
    args_list = (t_0, inits, list(params), x_rescale_params, t_rescale_params, steady_id)
    omega_center = 2 * np.pi * fh.get_sosc_freq(*args_list)
    print(f'Frequency of spontaneous oscillations: {omega_center / (2 * np.pi * rescale_factors['time'])} Hz')

    # ensemble variables needed
    num_uniq_freqs = 64 # number of unique frequencies
    ensemble_size = 128 # ensemble size for each frequency
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
        print(f"\nIteration {iteration + 1}: ")
        low_freq = 0
        chi_id_offset = -1
        curr_f_batch = gh.repeat2d_r(f_nd[iteration * freqs_per_batch:(iteration + 1) * freqs_per_batch, :], ensemble_size, simulator.BATCH_SIZE)
        x = simulator.sim(t_nd, inits, list(params), curr_f_batch, n_time_segs, simulator.BATCH_SIZE, freqs_per_batch)[0]

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
        chi_args = [(x[freq, :, :], f[iteration * freqs_per_batch + freq, :], ensemble_size, omegas[iteration * freqs_per_batch + freq - 1].item(), dt)
                    for freq in range(low_freq, freqs_per_batch)]
        with mp.Pool(int(0.75 * mp.cpu_count())) as pool:
            chis = pool.starmap(fh.chi, chi_args)
        chi_id_start = iteration * len(chis) + chi_id_offset
        for chi_id in range(len(chis)):
            avg_real_chi[chi_id_start + chi_id] = np.real(chis[chi_id])
            avg_imag_chi[chi_id_start + chi_id] = np.imag(chis[chi_id])
        inits = gh.concat(init_pos, init_probs)

    # rescale everything to SI units
    t = rescale_factors['time'] * t
    pos_freqs = pos_freqs / rescale_factors['time']
    omegas = omegas / rescale_factors['time']
    x0 = rescale_factors['distance'] * x0
    avg_psd = rescale_factors['distance']**2 * rescale_factors['time'] * avg_psd
    avg_psd_at_omegas = rescale_factors['distance']**2 * rescale_factors['time'] * avg_psd_at_omegas
    avg_real_chi = rescale_factors['time']**2 / rescale_factors['mass'] * avg_real_chi
    avg_imag_chi = rescale_factors['time']**2 / rescale_factors['mass'] * avg_imag_chi

    # ------------- BEGIN FDT CALCULATIONS ------------- #
    # calculate fluctuation response
    k_b = 1.380649e-23 # m^2 kg s^-2 K^-1
    temp = hb_rescale_params['k_gs_max'] * hb_rescale_params['d']**2 / (k_b * parameters['u_gs_max'])
    temp = (rescale_factors['mass'] * rescale_factors['distance']**3 / rescale_factors['time']**2) * temp
    theta = fh.fluc_resp(avg_psd_at_omegas[1:], avg_imag_chi, omegas[1:], temp, onesided=onesided)
    # ------------- END FDT CALCULATIONS ------------- #

    # ------------- BEGIN PLOTTING ------------- #
    t = t[steady_id:]

    # preliminary plotting
    gh.plot(t, x0[0, :], labels=(r'Time (s)', r'$x_{0}$ (m)'))

    # autocorrelation function
    gh.plot(t, avg_auto_corr, labels=(r'Time (s)', r'$\langle \frac{C(t)}{C(0)} \rangle \text{m}^2$'))

    # Power spectral density
    gh.plot(pos_freqs, avg_psd, labels=(r'Frequency (Hz)', r'Power spectral density $\left(\frac{\text{m}^2}{Hz}\right)$'), lims=[(pos_freqs[0], pos_freqs[len(pos_freqs)//2])])
    gh.plot(omegas, avg_psd_at_omegas, labels=(r'Angular frequency (rad/s)', r'Power spectral density $\left(\frac{\text{m}^2}{rad/s}\right)$'), lims=[(0, omegas[-1])])

    # linear response function
    gh.plot(omegas[1:] / (2 * np.pi), avg_real_chi, scatter=True, labels=(r'Driving Frequency (Hz)', r'$\Re\{\chi_x\}$'))
    gh.plot(omegas[1:] / (2 * np.pi), avg_imag_chi, scatter=True, labels=(r'Driving Frequency (Hz)', r'$\Im\{\chi_x\}$'))

    # theta
    gh.plot(omegas[1:], theta, scatter=True, labels=(r'Angular frequency (rad/s)', r'$\theta(\omega)$'), hlines=(1, omegas[1] / (2 * np.pi), omegas[-1] / (2 * np.pi)))
    gh.plot(omegas[1:], theta, scatter=True, labels=(r'Angular frequency (rad/s)', r'$\theta(\omega)$'))

    y_scale_range = 0
    while True:
        try:
            y_scale_range = float(input("Enter the range of y scale: "))
        except ValueError:
            print("Invalid input")
            break
        gh.plot(omegas[1:], theta, scatter=True, labels=(r'Angular frequency (rad/s)', r'$\theta(\omega)$'),
                lims=[(omegas[1], omegas[-1]), (-y_scale_range, y_scale_range)], hlines=(1, omegas[1] / (2 * np.pi), omegas[-1] / (2 * np.pi)))
    # ------------- END PLOTTING ------------- #