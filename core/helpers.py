import scipy as sp
import torch
import numpy as np

import sdeint as sdeint
import nondimensional_model as nd_model
import steady_nondimensional_model as steady_nd_model
import examples.harmonic_oscillator as harmonic_oscillator

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 5000 if DEVICE.type == 'cuda' else 64
SDE_TYPES = ['ito', 'stratonovich']
K_B = 1.380649e-23 # m^2 kg s^-2 K^-1
SOSC_MAX_RANGE = 1.3
SOSC_MIN_RANGE = 0.7

def hb_sols(t: np.ndarray, x0: np.ndarray, params: list, force_params: list) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time array
    :param x0: the initial conditions of each simulation
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
    #x0s = torch.tile(x0s, (BATCH_SIZE, 1)) # size: (BATCH_SIZE, len(x0))

    # time array
    n = len(t)
    ts = [t[0], t[-1]]

    # solving a system of SDEs and implementing a progress bar (this is cool fyi)
    solver = sdeint.Solver()
    hb_sol = torch.zeros((n, BATCH_SIZE, x0.shape[1]), dtype=DTYPE, device=DEVICE)
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

def driving_freqs(omega_0: float, nfreqs: int = BATCH_SIZE) -> np.ndarray:
    """
    Returns an array of driving frequencies around omega_0
    :param omega_0: the frequency to generate the array around
    :return: an array of driving frequencies around omega_0
    """
    omegas = np.linspace(SOSC_MIN_RANGE * omega_0, SOSC_MAX_RANGE * omega_0, nfreqs - 2)
    delta = omegas[1] - omegas[0]
    if np.any(omegas == omega_0):
        omegas[omegas == omega_0] = omega_0 + delta / 2
    omegas = np.append(omegas, omega_0)
    omegas = np.append(omegas, 0)
    omegas = np.sort(omegas)
    return np.unique(omegas)

def sf(t: np.ndarray, amp: float, omega_0: float, phase: float, offset: float, n: int = BATCH_SIZE) -> np.ndarray:
    """
    Returns the stimulus force position at times t
    :param t: time
    :param amp: amplitude
    :param omega_0: angular frequency of spontaneous oscillations
    :param phase: phase of the stimulus force
    :param offset: offset of the stimulus force
    :return: the stimulus force position
    """
    sf_batches = np.zeros((n, len(t)))
    omegas = driving_freqs(omega_0, n)
    for i in range(n):
        sf_batches[i] = amp * np.sin(omegas[i] * t + phase) + offset
    return sf_batches

def auto_corr(x: np.ndarray, norm: bool = True) -> np.ndarray:
    """
    Returns the (normalized) auto-correlation function <X(t) X(0)> for the time series data
    :param x: the time series data
    :return: the auto-correlation function
    """
    x = x - np.mean(x)
    c = np.correlate(x, x, mode='full')
    c = c[len(c) // 2:]
    if norm:
        return c / c[0]
    else:
        return c

def psd(x: np.ndarray, dt: float, ifreqs: np.ndarray, welch: bool = False, nperseg: float = None) -> np.ndarray:
    """
    Returns the power spectral density (PSD) of the input signal x
    :param x: the time series input signal
    :param dt: the time step
    :param ifreqs: the frequencies to interpolate the PSD with
    :param welch: whether to use Welch's method
    :param nperseg: length of each segment if using Welch's method
    :return: the power spectral density
    """
    fs = 1 / dt
    if welch:
        if nperseg is None:
            nperseg = len(x) // 4
        freqs, psd = sp.signal.welch(x, fs=fs, nperseg=nperseg, scaling='density', return_onesided=True)
        psd = np.interp(ifreqs, freqs, psd)
    else:
        corr = auto_corr(x, norm=False) / len(x)
        psd_gen = np.real(sp.fft.fft(corr)) * dt
        psd = np.zeros(len(ifreqs), dtype=float)
        freqs = sp.fft.fftfreq(corr.shape[0], dt)
        for i in range(len(ifreqs)):
            index = np.argmin(np.abs(freqs - ifreqs[i]))
            psd[i] = psd_gen[index]
    return psd

def chi_ft(x: np.ndarray, force: np.ndarray) -> np.ndarray:
    """
    Returns the linear response function (in frequency space) for the time series data in response to a stimulus force at different frequencies
    :param x: the time series data; n x m array: n = number of simulations, m = number of time steps
    :param force: the stimulus forces; n x m array: n = number of simulations, m = number of time steps
    :return: the linear response function (in frequency space)
    """
    # compute the Fourier Transform
    x_ft = sp.fft.fft(x - np.mean(x, axis=1, keepdims=True), axis=1)
    force_ft = sp.fft.fft(force - np.mean(force, axis=1, keepdims=True), axis=1)
    chi = x_ft / force_ft # n x m array
    return chi

def fluc_resp(psd: np.ndarray, imag_chi: np.ndarray, omegas: np.ndarray, temp: float, boltzmann_scale: float = 1.0, one_sided: bool = True) -> np.ndarray:
    """
    Returns the fluctuation response (theta(omega) = omega C(omega) / [2 k_B T chi_I(omega)]) at different driving frequencies omega
    :param psd: the power spectral density
    :param imag_chi: the linear response function (imaginary component) in frequency space (specifically at the driving frequencies)
    :param omegas: the driving frequencies (angular frequencies)
    :param temp: the temperature
    :param boltzmann_scale: the scale factor to apply in front of the boltzmann constant to ensure consistent units
    :param one_sided: whether the PSD is one-sided or not
    :return: the fluctuation response function
    """
    one_sided_factor = 1 if one_sided else 2
    theta = np.zeros_like(omegas)
    for i in range(len(theta)):
        theta[i] = omegas[i] * psd[i] / (boltzmann_scale * K_B * temp * np.abs(imag_chi[i]))
    return theta