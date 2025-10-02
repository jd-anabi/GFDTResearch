import scipy as sp
import torch
import numpy as np

import sdeint as sdeint
import nondimensional_model as nd_model
import steady_nondimensional_model as steady_nd_model

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 4000 if DEVICE.type == 'cuda' else 64
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
        x0 = x0[:, :4]
        sde = steady_nd_model.HairBundleSDE(*params, *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        print("Using the steady-state solution for the open-channel probability")
        print("Hair bundle model has been set up")
    else:
        sde = nd_model.HairBundleSDE(*params, *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        print("Hair bundle model has been set up")

    # setting up initial conditions
    x0s = torch.tensor(x0, dtype=DTYPE, device=DEVICE)

    # time array
    n = len(t)
    ts = [t[0], t[-1]]

    # solving a system of SDEs and implementing a progress bar (this is cool fyi)
    solver = sdeint.Solver()
    sol = torch.zeros((n, BATCH_SIZE, x0.shape[1]), dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        try:
            sol = solver.euler(sde, x0s, ts, n) # only keep the last solution
        except (Warning, Exception) as e:
            print(e)
            exit()

    print("SDEs have been solved")
    return sol.cpu().detach().numpy()

def driving_freqs(omega_0: float, n: int = BATCH_SIZE) -> np.ndarray:
    """
    Returns an array of driving frequencies around omega_0
    :param omega_0: the frequency to generate the array around
    :param n: number of frequencies to generate
    :return: an array of driving frequencies around omega_0
    """
    omegas = np.linspace(SOSC_MIN_RANGE * omega_0, SOSC_MAX_RANGE * omega_0, n - 2)
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
    :param n: number of stimulus forces
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
    :param norm: whether to normalize the auto-correlation function
    :return: the auto-correlation function
    """
    x = x - np.mean(x)
    c = np.correlate(x, x, mode='full')
    c = c[len(c) // 2:]
    if norm:
        return c / c[0]
    else:
        return c

def psd(x: np.ndarray, n: int, dt: float, fs: float, int_freqs: np.ndarray, welch: bool = False, nperseg: float = None, onesided: bool = True, angular: bool = True) -> np.ndarray:
    """
    Returns the power spectral density (PSD) of the input signal x
    :param x: the time series input signal
    :param n: the number of sampling points
    :param dt: the time step
    :param fs: frequency spacing
    :param int_freqs: the frequencies to interpolate the PSD with
    :param welch: whether to use Welch's method
    :param nperseg: length of each segment if using Welch's method
    :param onesided: whether to use one-sided PSD or not
    :param angular: whether to use angular PSD or not
    :return: the power spectral density
    """
    angular_factor = 2 * np.pi if angular else 1
    if welch:
        if nperseg is None:
            nperseg = len(x) // 4
        freqs, s = sp.signal.welch(x, fs=fs, nperseg=nperseg, scaling='density', return_onesided=onesided)
        s = np.interp(int_freqs, freqs, s) / angular_factor
    else:
        onesided_factor = 2 if onesided else 1
        x_fft = sp.fft.rfft(x - np.mean(x))
        s_gen = onesided_factor * np.abs(x_fft)**2 * dt / (angular_factor * x.shape[0])
        s_gen[0] /= 2
        if len(x) % 2 == 0:
            s_gen[-1] /= 2
        s = np.zeros(len(int_freqs), dtype=float)
        freqs = sp.fft.rfftfreq(n, 1 / fs)
        #psd = np.interp(ifreqs, freqs, psd_gen) / angular_factor
        for i in range(len(int_freqs)):
            index = np.argmin(np.abs(freqs - int_freqs[i]))
            s[i] = s_gen[index]
    return s

def chi_ft(x: np.ndarray, force: np.ndarray) -> np.ndarray:
    """
    Returns the linear response function (in frequency space) for the time series data in response to a stimulus force at different frequencies
    :param x: the time series data; n x m array: n = number of simulations, m = number of time steps
    :param force: the stimulus forces; n x m array: n = number of simulations, m = number of time steps
    :return: the linear response function (in frequency space)
    """
    # compute the Fourier Transform
    x_ft = sp.fft.rfft(x - np.mean(x, axis=1, keepdims=True), axis=1)
    force_ft = sp.fft.rfft(force - np.mean(force, axis=1, keepdims=True), axis=1)
    chi = x_ft / force_ft # n x m array
    return chi

def fluc_resp(psd: np.ndarray, imag_chi: np.ndarray, omegas: np.ndarray, temp: float, boltzmann_scale: float = 1.0, onesided: bool = True) -> np.ndarray:
    """
    Returns the fluctuation response (theta(omega) = omega C(omega) / [2 k_B T chi_I(omega)]) at different driving frequencies omega
    :param psd: the power spectral density
    :param imag_chi: the linear response function (imaginary component) in frequency space (specifically at the driving frequencies)
    :param omegas: the driving frequencies (angular frequencies)
    :param temp: the temperature
    :param boltzmann_scale: the scale factor to apply in front of the boltzmann constant to ensure consistent units
    :param onesided: whether the PSD is one-sided or not
    :return: the fluctuation response function
    """
    onesided_factor = 4 if onesided else 2
    theta = np.zeros_like(omegas)
    for i in range(len(theta)):
        theta[i] = omegas[i] * psd[i] / (onesided_factor * boltzmann_scale * K_B * temp * np.abs(imag_chi[i]))
    return theta