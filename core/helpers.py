import scipy as sp
import torch
import numpy as np

import sdeint as sdeint
import nondimensional_model as nd_model
import steady_nondimensional_model as steady_nd_model
import examples.harmonic_oscillator as harmonic_oscillator

if torch.cuda.is_available():
    DEVICE = torch.device('cpu')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 2048 if DEVICE.type == 'cuda' else 24
SDE_TYPES = ['ito', 'stratonovich']
K_B = 1.380649e-23 # m^2 kg s^-2 K^-1
SOSC_MAX_RANGE = 7

def hb_sols(t: np.ndarray, x0: list, params: list, force_params: list) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time array
    :param x0: the initial conditions of the hair bundle
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param force_params: the parameters to use in the stimulus force
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    if DEVICE.type == 'cuda' or DEVICE.type == 'mps':
        print("Using GPU")
    else:
        print("Using CPU")

    # check if we are using the steady-state solution
    if params[3] == 0:
        sde = harmonic_oscillator.HarmonicOscillator(*params, *force_params, batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        #x0 = x0[:4]
        #sde = steady_nd_model.HairBundleSDE(*params, *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        print("Using the steady-state solution for the open-channel probability")
    else:
        sde = harmonic_oscillator.HarmonicOscillator(*params, *force_params, batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        #sde = nd_model.HairBundleSDE(*params, *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        print("Hair bundle model has been set up")

    # setting up initial conditions
    x0s = torch.tensor(x0, dtype=DTYPE, device=DEVICE)
    x0s = torch.tile(x0s, (BATCH_SIZE, 1)) # size: (BATCH_SIZE, len(x0))

    # time array
    n = len(t)
    ts = [t[0], t[-1]]

    # solving a system of SDEs and implementing a progress bar (this is cool fyi)
    solver = sdeint.Solver()
    hb_sol = torch.zeros((n, BATCH_SIZE, len(x0)), dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        try:
            hb_sol = solver.euler(sde, x0s, ts, n) # only keep the last solution
        except (Warning, Exception) as e:
            print(e)
            exit()

    print("SDEs have been solved")
    return hb_sol.cpu().detach().numpy()

def rescale_x(nd_hb_pos: np.ndarray, gamma: float, d: float, x_sp: float, chi_hb: float) -> np.ndarray:
    """
    Rescaling the hair-bundle displacement
    :param nd_hb_pos: the hair bundle position
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param x_sp: resting deflection of stereociliary pivots
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :return: the rescaled hair-bundle displacement
    """
    hb_pos = chi_hb * d / gamma * nd_hb_pos + x_sp
    return hb_pos

def rescale_t(nd_t: np.ndarray, k_gs_max: float, s_max: float, t_0: float, s_max_nd: float, chi_a: float) -> np.ndarray:
    """
    Rescaling the time array
    :param nd_t: the time array
    :param k_gs_max: maximum stiffness of gating spring
    :param s_max: maximum slipping rate
    :param t_0: time offset
    :param s_max_nd: non-dimensional maximum slipping rate
    :param chi_a: non-dimensional parameter for non-dimensional adaptation motor displacement
    :return:  the rescaled time
    """
    t = chi_a * s_max_nd / (k_gs_max * s_max) * nd_t - t_0
    return t

def inv_rescale_f(force: np.ndarray, gamma: float, d: float, k_sp: float, chi_hb: float) -> np.ndarray:
    """
    Rescaling the stimulus force from dimensional -> non-dimensional
    :param force: the stimulus force position
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param k_sp: stiffness of stereociliary pivots
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :return: the rescaled stimulus force
    """
    force_nd = gamma / (chi_hb * k_sp * d) * force
    return force_nd

def rescale_force_params(amp: float, omegas: np.ndarray, phase: float, offset: float,
                         gamma: float, d: float, k_sp: float, chi_hb: float,
                         k_gs_max: float, s_max: float, s_max_nd: float, chi_a: float, t_0: float) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Rescale the stimulus force parameters from dimensional -> non-dimensional
    :param amp: amplitude
    :param omegas: frequencies
    :param phase: phase
    :param offset: offset
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param k_sp: stiffness of stereociliary pivots
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :param k_gs_max: maximum stiffness of gating spring
    :param s_max: maximum slipping rate
    :param s_max_nd: non-dimensional maximum slipping rate
    :param chi_a: non-dimensional parameter for non-dimensional adaptation motor displacement
    :param t_0: time offset
    :return: the rescaled stimulus force parameters
    """
    alpha = gamma / (chi_hb * k_sp * d)
    t_prime = chi_a * s_max_nd / (k_gs_max * s_max)
    amp_nd = alpha * amp
    offset_nd = alpha * offset
    omega_nd = t_prime * omegas
    phases_nd = phase - t_0 * omegas
    return amp_nd, omega_nd, phases_nd, offset_nd

def driving_freqs(omega_0: float) -> np.ndarray:
    """
    Returns an array of driving frequencies around omega_0
    :param omega_0: the frequency to generate the array around
    :return: an array of driving frequencies around omega_0
    """
    omegas = np.linspace(0, SOSC_MAX_RANGE * omega_0, BATCH_SIZE - 1)
    delta = omegas[1] - omegas[0]
    if np.any(omegas == omega_0):
        omegas[omegas == omega_0] = omega_0 + delta / 2
    omegas = np.append(omegas, omega_0)
    omegas = np.sort(omegas)
    return np.unique(omegas)

def sf(t: np.ndarray, amp: float, omega_0: float, phase: float, offset: float) -> np.ndarray:
    """
    Returns the stimulus force position at times t
    :param t: time
    :param amp: amplitude
    :param omega_0: angular frequency of spontaneous oscillations
    :param phase: phase of the stimulus force
    :param offset: offset of the stimulus force
    :return: the stimulus force position
    """
    sf_batches = np.zeros((BATCH_SIZE, len(t)))
    omegas = driving_freqs(omega_0)
    for i in range(BATCH_SIZE):
        sf_batches[i] = amp * np.sin(omegas[i] * t + phase) + offset
    return sf_batches

def auto_corr(hb_pos: np.ndarray) -> np.ndarray:
    """
    Returns the (normalized) auto-correlation function <X(t) X(0)> for the position of a hair bundle
    :param hb_pos: the position of a hair bundle
    :return: the auto-correlation function
    """
    hb_pos = hb_pos - np.mean(hb_pos)
    c = np.correlate(hb_pos, hb_pos, mode='full')
    c = c[len(c) // 2:]
    return c / (len(hb_pos) * c[0])

def lin_resp_ft(hb_pos: np.ndarray, force: np.ndarray, norm: bool = True) -> np.ndarray:
    """
    Returns the linear response function (in frequency space) for the position of a hair bundle in response to a stimulus force
    :param hb_pos: the position of a hair bundle
    :param force: the stimulus forces
    :param norm: whether to normalize the response function
    :return: the linear response function (in frequency space)
    """
    # compute the Fourier Transform
    if norm:
        hb_pos_ft = sp.fft.fft(hb_pos - np.mean(hb_pos), axis=1) / len(hb_pos)
        sf_pos_ft = sp.fft.fft(force - np.mean(force), axis=1) / len(force)
    else:
        hb_pos_ft = sp.fft.fft(hb_pos - np.mean(hb_pos), axis=1)
        sf_pos_ft = sp.fft.fft(force - np.mean(force), axis=1)
    return hb_pos_ft / sf_pos_ft

def fluc_resp(autocorr_ft: np.ndarray, linresp_ft: np.ndarray, omegas: np.ndarray,
              temp: float, boltzmann_scale: float = 1.0) -> np.ndarray:
    """
    Returns the fluctuation response (theta(omega) = omega C(omega) / [2 k_B T chi_I(omega)]) at different driving frequencies omega
    :param autocorr_ft: the auto-correlation function in frequency space (specifically at the driving frequencies)
    :param linresp_ft: the linear response function in frequency space (specifically at the driving frequencies)
    :param omegas: the driving frequencies
    :param temp: the temperature
    :param boltzmann_scale: the scale factor to apply in front of the boltzmann constant to ensure consistent units
    :return: the fluctuation response function
    """
    # convert to angular frequency
    autocorr_ft = 2 * np.pi * autocorr_ft
    linresp_ft = 2 * np.pi * linresp_ft
    theta = np.zeros_like(omegas)
    for i in range(len(theta)):
        theta[i] = (omegas[i] * autocorr_ft[i] / (2 * boltzmann_scale * K_B * temp * linresp_ft[i].imag)).real
    return theta