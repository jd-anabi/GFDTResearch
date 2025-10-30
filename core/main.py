import os
import sys
import math
from typing import Dict
import multiprocessing as mp

from torch import dtype

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pint
import torch
import numpy as np
from sbi import utils as utils
from sbi.inference import SNPE
from sbi.analysis import pairplot

from core.Helpers import fdt_helpers as fdt, gen_helpers as helpers, hair_model_helpers as model_helpers
from core.Simulator import simulator, simulator_helpers

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    if (torch.cuda.get_device_properties(DEVICE).major, torch.cuda.get_device_properties(DEVICE).minor) < (8, 0):
        DEVICE = torch.device('cpu')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 2**11 if DEVICE.type == 'cuda' else 2**6

# ensemble variables needed
UNIQUE_FREQS = 2**6 # number of unique frequencies
ENSEMBLE_SIZE = 2**7 if DEVICE.type == 'cuda' else 2**5 # ensemble size for each frequency
FPB = BATCH_SIZE // ENSEMBLE_SIZE # number of frequencies per batch
ITERATIONS = int(UNIQUE_FREQS / FPB)

K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

if __name__ == '__main__':
    # -------------------------- BEGIN SETUP -------------------------- #
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
    file_num = int(input('File number for model parameters: '))
    file = os.getcwd() + model_params_dir + model_files[file_num - 1]
    x0, params, rescale_params, forcing_params, units = helpers.read_model_file(file)
    amp, phase, offset = forcing_params

    # need to construct dictionary now that converts current units to SI units
    ureg = pint.UnitRegistry()
    try:
        factors = [ureg(unit).to_base_units().magnitude for unit in units]
        units_rescale: Dict[str, float] = {'distance': factors[0], 'mass': factors[1], 'time': factors[2]}
    except pint.UndefinedUnitError as e:
        print(f'Error: {e}. Unrecognized units.')
        exit()
    # -------------------------- END SETUP -------------------------- #

    # ------------- BEGIN RESCALING CALCULATIONS ------------- #
    # set up dictionary for model parameters and rescaling parameters
    parameters: Dict[str, float] = {'tau_hb': params[0], 'tau_m': params[1], 'tau_gs': params[2],
                                    'tau_t': params[3],
                                    'c_min': params[4], 's_min': params[5], 's_max': params[6], 'ca2_m': params[7],
                                    'ca2_gs': params[8], 'u_gs_max': params[9], 'delta_e': params[10],
                                    'k_gs_ratio': params[11],
                                    'chi_hb': params[12], 'chi_a': params[13], 'x_c': params[14],
                                    'eta_hb': params[15],
                                    'eta_a': params[16]}
    hb_rescale_params: Dict[str, float] = {'gamma': rescale_params[0], 'd': rescale_params[1],
                                           'x_sp': rescale_params[2], 'k_sp': rescale_params[3],
                                           'k_gs_max': rescale_params[4], 's_max': rescale_params[5],
                                           't_0': rescale_params[6], 'alpha': rescale_params[7]}
    hb_rescale_params.update({'s_max_nd': params[6], 'chi_hb': params[12],
                              'chi_a': params[13]})  # need to add in non-dimensional parameters for rescaling too

    # rescaling parameters needed for time and data
    t_rescale_params = [hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'], hb_rescale_params['t_0'],
                        hb_rescale_params['s_max_nd'], hb_rescale_params['chi_a']]
    x_rescale_params = [hb_rescale_params['gamma'], hb_rescale_params['d'], hb_rescale_params['x_sp'],
                        hb_rescale_params['k_sp'], hb_rescale_params['alpha'], hb_rescale_params['chi_hb']]
    # ------------- END RESCALING CALCULATIONS ------------- #

    # ------------- BEGIN TIME AND FREQUENCY ARRAY CALCULATIONS ------------- #
    # non-dimensional time
    dt = 1e-3
    t_max = int(input("Max time: "))
    ts = (0, t_max)
    n = int((ts[-1] - ts[0]) / dt)
    t_nd = torch.linspace(ts[0], ts[-1], n, dtype=DTYPE, device=DEVICE)
    steady_id = int(0.4 * len(t_nd))
    segs = math.ceil(t_max / 200)
    time_seg_ids = helpers.get_even_ids(t_nd.shape[0], segs + 1)

    # rescale time to dimensional
    t = model_helpers.rescale_t(t_nd, *t_rescale_params)
    dt = float(t[1] - t[0])  # rescale dt

    # steady-state index for analysis later
    n = t.shape[0]
    n_steady = t[steady_id:].shape[0]

    # frequency arrays (only positive values)
    pos_freqs = torch.fft.rfftfreq(n_steady, dt)
    # ------------- END TIME AND FREQUENCY ARRAY CALCULATIONS ------------- #

    # -------------------- BEGIN NATURAL FREQUENCY CALCULATION -------------------- #
    # set up simulator for spontaneous oscillations
    # initial conditions
    init_pos = np.random.randint(0, 10, size=(BATCH_SIZE, 2))
    init_probs = np.random.randint(0, 1, size=(BATCH_SIZE, 3))
    inits = helpers.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)
    inits = torch.tensor(inits, dtype=DTYPE, device=DEVICE)
    inits_0 = inits[0, :].unsqueeze(0)

    sim = simulator.Simulator(torch.tensor(params, dtype=torch.float32, device=torch.device('cpu')).unsqueeze(0),
                              t_nd.to(torch.device('cpu'), dtype=torch.float32), inits_0.to(torch.device('cpu'), dtype=torch.float32),
                              fdt.force(t_nd, 0, 0, 0, 0, 1, device=torch.device('cpu')),
                              segs=segs, device=torch.device('cpu'))
    x0 = sim.simulate()[0, 0, 0]
    helpers.plot(units_rescale['time'] * t.cpu().detach().numpy()[steady_id:], units_rescale['distance'] * x0.cpu().detach().numpy()[steady_id:], labels=(r'Time (s)', r'$x_{0}$ (m)'))
    omega_center = 2 * np.pi * pos_freqs[torch.argmax(torch.abs(torch.fft.rfft(x0 - torch.mean(x0))))]
    print(f'Frequency of spontaneous oscillations: {omega_center / (2 * np.pi * units_rescale['time'])} Hz')

    prior = utils.BoxUniform(low=torch.zeros(17), high=torch.ones(17) * 5, device=str(DEVICE))
    thetas = prior.sample((BATCH_SIZE,)).to(dtype=DTYPE)
    for i in range(BATCH_SIZE):
        thetas[i, 3] = 0
    print(thetas)
    sim.batch_size = BATCH_SIZE
    sim.params = thetas
    sim.force = fdt.force(t_nd, 0, omega_center, 0, 0, BATCH_SIZE, device=DEVICE)
    sim.inits = inits
    sim.device = DEVICE
    sim.dtype = DTYPE
    x = sim.simulate()[0, 0, :, :]
    print(x)
    inference = SNPE(prior=prior)
    exit()
    # -------------------- END NATURAL FREQUENCY CALCULATION -------------------- #

    # calculate stimulus force
    f = fdt.force(t, amp, omega_center, phase, offset, UNIQUE_FREQS)
    f_nd = model_helpers.irescale_f(f, hb_rescale_params['gamma'], hb_rescale_params['d'],
                                    hb_rescale_params['k_sp'], hb_rescale_params['chi_hb'])
    f = f[:, steady_id:]  # steady-state portion of force
    f_driven = f[1:, :]  # ignore undriven force

    # find which index in the array of driving frequencies corresponds to omega_center
    omegas = fdt.gen_freqs(omega_center, UNIQUE_FREQS) # generate frequencies
    omega_center_id = np.argmax(omegas == omega_center)
    nd_f_params = model_helpers.irescale_f_params(omegas, amp, phase, offset,
                                                  hb_rescale_params['gamma'], hb_rescale_params['d'],
                                                  hb_rescale_params['k_sp'],
                                                  hb_rescale_params['chi_hb'], hb_rescale_params['k_gs_max'],
                                                  hb_rescale_params['s_max'],
                                                  hb_rescale_params['s_max_nd'], hb_rescale_params['chi_a'],
                                                  hb_rescale_params['t_0'])
    omegas_nd, amp_nd, phases_nd, offset_nd = nd_f_params[0], nd_f_params[1], nd_f_params[2], nd_f_params[3]

    # instantiate needed values
    x0 = np.zeros(t[steady_id:].shape[0])
    avg_psd = np.zeros(pos_freqs.shape[0])
    avg_psd_at_omegas = np.zeros(omegas.shape[0])
    avg_real_chi = np.zeros(UNIQUE_FREQS - 1)
    avg_imag_chi = np.zeros(UNIQUE_FREQS - 1)
    avg_auto_corr = np.zeros(t[steady_id:].shape[0])

    tiled_phases = np.tile(phases_nd, BATCH_SIZE)  # set up array of phases for each simulation (tiled since total batch size = ensemble size x freqs per batch)
    tiled_omegas = np.tile(omegas_nd, BATCH_SIZE)  # set up array of omegas for each simulation
    args_list = (t_nd, x0, list(params), tiled_omegas, amp_nd, tiled_phases, offset_nd)  # parameters

    for iteration in range(ITERATIONS):
        print(f"\nIteration {iteration + 1}: ")
        low_freq = 0
        chi_id_offset = -1
        curr_f_batch = helpers.repeat2d_r(f_nd[iteration * FPB:(iteration + 1) * FPB, :], ENSEMBLE_SIZE, BATCH_SIZE)
        x = simulator_helpers.sim(t_nd, inits, list(params), curr_f_batch, segs, BATCH_SIZE, FPB)[0]

        # rescale position data for later
        x = x.reshape(FPB, ENSEMBLE_SIZE, len(t_nd)) # shape: (freqs_per_batch, ensemble_size, len(curr_time))
        x = model_helpers.rescale_x(x, *x_rescale_params)
        x = x[:, :, steady_id:]  # only want to use the steady-state solution
        if iteration == 0:
            x0 = x[0, :, :]
            avg_auto_corr = fdt.auto_corr(x0, d=ENSEMBLE_SIZE)
            avg_psd = fdt.psd(x0, n_steady, dt, int_freqs=pos_freqs, d=ENSEMBLE_SIZE)
            avg_psd_at_omegas = fdt.psd(x0, n_steady, dt, int_freqs=(omegas / (2 * np.pi)), d=ENSEMBLE_SIZE)
            low_freq = 1
            chi_id_offset = 0

        # arguments needed for multiprocessing
        chi_args = [(x[freq, :, :], f[iteration * FPB + freq, :], ENSEMBLE_SIZE, omegas[iteration * FPB + freq - 1].item(), dt)
                    for freq in range(low_freq, FPB)]
        with mp.Pool(int(0.9 * mp.cpu_count())) as pool:
            chis = pool.starmap(fdt.chi, chi_args)
        chi_id_start = iteration * len(chis) + chi_id_offset
        for chi_id in range(len(chis)):
            avg_real_chi[chi_id_start + chi_id] = np.real(chis[chi_id])
            avg_imag_chi[chi_id_start + chi_id] = np.imag(chis[chi_id])
        inits = helpers.concat(init_pos, init_probs)

    # rescale everything to SI units
    t = units_rescale['time'] * t
    pos_freqs = pos_freqs / units_rescale['time']
    omegas = omegas / units_rescale['time']
    x0 = units_rescale['distance'] * x0
    avg_psd = units_rescale['distance'] ** 2 * units_rescale['time'] * avg_psd
    avg_psd_at_omegas = units_rescale['distance'] ** 2 * units_rescale['time'] * avg_psd_at_omegas
    avg_real_chi = units_rescale['time'] ** 2 / units_rescale['mass'] * avg_real_chi
    avg_imag_chi = units_rescale['time'] ** 2 / units_rescale['mass'] * avg_imag_chi

    # ------------- BEGIN FDT CALCULATIONS ------------- #
    temp = hb_rescale_params['k_gs_max'] * hb_rescale_params['d']**2 / (K_B * parameters['u_gs_max'])
    temp = (units_rescale['mass'] * units_rescale['distance'] ** 3 / units_rescale['time'] ** 2) * temp
    theta = fdt.fluc_resp(avg_psd_at_omegas[1:], avg_imag_chi, omegas[1:], temp, onesided=False)
    # ------------- END FDT CALCULATIONS ------------- #

    # ------------- BEGIN PLOTTING ------------- #
    t = t[steady_id:]

    # preliminary plotting
    helpers.plot(t, x0[0, :], labels=(r'Time (s)', r'$x_{0}$ (m)'))

    # autocorrelation function
    helpers.plot(t, avg_auto_corr, labels=(r'Time (s)', r'$\langle \frac{C(t)}{C(0)} \rangle \text{m}^2$'))

    # Power spectral density
    helpers.plot(pos_freqs, avg_psd, labels=(r'Frequency (Hz)', r'Power spectral density $\left(\frac{\text{m}^2}{Hz}\right)$'), lims=[(pos_freqs[0], pos_freqs[len(pos_freqs) // 2])])
    helpers.plot(omegas, avg_psd_at_omegas, labels=(r'Angular frequency (rad/s)', r'Power spectral density $\left(\frac{\text{m}^2}{rad/s}\right)$'), lims=[(0, omegas[-1])])

    # linear response function
    helpers.plot(omegas[1:] / (2 * np.pi), avg_real_chi, scatter=True, labels=(r'Driving Frequency (Hz)', r'$\Re\{\chi_x\}$'))
    helpers.plot(omegas[1:] / (2 * np.pi), avg_imag_chi, scatter=True, labels=(r'Driving Frequency (Hz)', r'$\Im\{\chi_x\}$'))

    # theta
    helpers.plot(omegas[1:], theta, scatter=True, labels=(r'Angular frequency (rad/s)', r'$\theta(\omega)$'), hlines=(1, omegas[1] / (2 * np.pi), omegas[-1] / (2 * np.pi)))
    helpers.plot(omegas[1:], theta, scatter=True, labels=(r'Angular frequency (rad/s)', r'$\theta(\omega)$'))

    y_scale_range = 0
    while True:
        try:
            y_scale_range = float(input("Enter the range of y scale: "))
        except ValueError:
            print("Invalid input")
            break
        helpers.plot(omegas[1:], theta, scatter=True, labels=(r'Angular frequency (rad/s)', r'$\theta(\omega)$'),
                     lims=[(omegas[1], omegas[-1]), (-y_scale_range, y_scale_range)], hlines=(1, omegas[1] / (2 * np.pi), omegas[-1] / (2 * np.pi)))
    # ------------- END PLOTTING ------------- #