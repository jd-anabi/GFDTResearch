from typing import Union

import scipy as sp
import numpy as np
import torch

import sdeint as sdeint
from core.Models import nondimensional_model as nd_model, steady_nondimensional_model as steady_nd_model
import hair_model_helpers as hmh

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 5000 if DEVICE.type == 'cuda' else 64
SDE_TYPES = ['ito', 'stratonovich']
K_B = 1.380649e-23 # m^2 kg s^-2 K^-1
SOSC_MAX_RANGE = 1.3
SOSC_MIN_RANGE = 0.7

def sols(t: np.ndarray, x0: np.ndarray, params: list,
         omegas: np.ndarray, amp: float, phases: np.ndarray, offset: float,
         explicit: bool = True) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time array
    :param x0: the initial conditions of each simulation
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param omegas: the frequencies to drive the simulations at
    :param amp: the amplitude to drive the simulations at
    :param phases: the phases to drive the simulations at
    :param offset: the offset to drive the simulations at
    :param explicit: whether to use the explicit Euler-Maruyama method
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    if DEVICE.type == 'cuda' or DEVICE.type == 'mps':
        print("Using GPU")
    else:
        print("Using CPU")

    # check if we are using the steady-state solution
    if params[3] == 0:
        x0 = x0[:, :4]
        sde = steady_nd_model.HairBundleSDE(*params, omegas, amp, phases, offset, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        print("Using the steady-state solution for the open-channel probability")
    else:
        sde = nd_model.HairBundleSDE(*params, omegas, amp, phases, offset, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)

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
            if explicit:
                sol = solver.euler(sde, x0s, ts, n) # only keep the last solution
            else:
                sol = solver.implicit_euler(sde, x0s, ts, n)
        except (Warning, Exception) as e:
            print(e)
            exit()

    return sol.cpu().detach().numpy()

def get_sosc_freq(t: np.ndarray, x0: np.ndarray, params: list,
                  x_rescale_params: list, t_rescale_params: list,
                  steady_id: float, explicit: bool = True) -> float:
    """
    Finds the frequency of spontaneous oscillations for an undriven hair bundle given a set of parameters and initial conditions
    :param t: time array
    :param x0: the initial conditions of each simulation
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param x_rescale_params: the parameters to use for scaling the data
    :param t_rescale_params: the parameters to use for scaling the time
    :param steady_id: the index of the steady-state solution slice
    :param explicit: whether to use the explicit Euler-Maruyama method
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    device = torch.device('cpu')

    # check if we are using the steady-state solution
    if params[3] == 0:
        x0 = x0[:, :4]
        sde = steady_nd_model.HairBundleSDE(*params, np.zeros(1), 0, np.zeros(1), 0, sde_type=SDE_TYPES[0], batch_size=1, device=device, dtype=DTYPE).to(device)
        print("Using the steady-state solution for the open-channel probability")
    else:
        sde = nd_model.HairBundleSDE(*params, np.zeros(1), 0, np.zeros(1), 0, sde_type=SDE_TYPES[0], batch_size=1, device=device, dtype=DTYPE).to(device)

    # setting up initial conditions
    x0s = torch.tensor(x0, dtype=DTYPE, device=device)

    # time array
    n = len(t)
    ts = [t[0], t[-1]]

    # solving a system of SDEs and implementing a progress bar (this is cool fyi)
    solver = sdeint.Solver()
    sol = torch.zeros((n, 1, x0.shape[1]), dtype=DTYPE, device=device)
    with torch.no_grad():
        try:
            if explicit:
                sol = solver.euler(sde, x0s, ts, n) # only keep the last solution
            else:
                sol = solver.implicit_euler(sde, x0s, ts, n)
        except (Warning, Exception) as e:
            print(e)
            exit()

    result = sol.cpu().detach().numpy() # shape: (len(t), 1, x0.shape[1])
    x = result[:, 0, 0]
    x = hmh.rescale_x(x, *x_rescale_params)
    t = hmh.rescale_t(t, *t_rescale_params)
    x = x[steady_id:]
    t = t[steady_id:]
    dt = t[1] - t[0] if t.shape[0] != 1 else 0
    freqs = sp.fft.rfftfreq(x.shape[0], dt)
    xf = sp.fft.rfft(x - np.mean(x), x.shape[0])
    sosc_id = np.argmax(xf)
    return freqs[sosc_id]

def gen_freqs(omega_0: float, n: int = BATCH_SIZE) -> np.ndarray:
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

def force(t: np.ndarray, amp: float, omega_0: float, phase: float, offset: float, n: int = BATCH_SIZE) -> np.ndarray:
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
    omegas = gen_freqs(omega_0, n)
    for i in range(n):
        sf_batches[i] = amp * np.sin(omegas[i] * t + phase) + offset
    return sf_batches

def auto_corr(x: np.ndarray, d: int = 1) -> np.ndarray:
    """
    Returns the (normalized) auto-correlation function <X(t) X(0)> for the time series data
    :param x: the time series data
    :param d: the dimension of the time series data; if d > 1, returns the average auto-correlation function
    :return: the auto-correlation function
    """
    if d == 1:
        xf = sp.fft.rfft(x - np.mean(x), n=2*x.shape[1])
        acf = sp.fft.irfft(np.abs(xf)**2)[:x.shape[1]]
    else:
        xf = sp.fft.rfft(x - np.mean(x, axis=1, keepdims=True), n=2*x.shape[1], axis=1)
        acf = sp.fft.irfft(np.abs(xf)**2, axis=1)[:, :x.shape[1]]
        acf = np.mean(acf, axis=0)
    return acf / acf[0]

def psd(x: np.ndarray, n: int, dt: float, int_freqs: np.ndarray = None, d: int = 1, onesided: bool = True, angular: bool = False) -> np.ndarray:
    """
    Returns the power spectral density (PSD) of the input signal x; if d > 1 then returns the average PSD
    :param x: the input signal (shape = (n, d))
    :param int_freqs: the interpolation frequencies
    :param n: number of sampling points (i.e. len(x))
    :param dt: the time step
    :param d: the number of input signals
    :param onesided: whether to return onesided PSD
    :param angular: whether to return angular PSD
    :return: the PSD
    """
    angular_factor = 2 * np.pi if angular else 1
    onesided_factor = 2 if onesided else 1
    if d == 1:
        x_fft = sp.fft.rfft(x - np.mean(x))
        s_gen = onesided_factor * np.abs(x_fft) ** 2 * dt / (angular_factor * x.shape[-1])
        s_gen[0] /= 2
        if len(x) % 2 == 0:
            s_gen[-1] /= 2
    else:
        x_fft = sp.fft.rfft(x - np.mean(x, axis=1, keepdims=True), axis=1)
        s_gen = onesided_factor * np.abs(x_fft) ** 2 * dt / (angular_factor * x.shape[-1])
        s_gen[0] /= 2
        if len(x) % 2 == 0:
            s_gen[-1] /= 2
        s_gen = np.mean(s_gen, axis=0)
    s = np.zeros(len(int_freqs), dtype=float)
    freqs = sp.fft.rfftfreq(n, dt)
    for i in range(len(int_freqs)):
        index = np.argmin(np.abs(freqs - int_freqs[i]))
        s[i] = s_gen[index]
    return s

def chi(x: np.ndarray, f: np.ndarray, d: int = 1, omega: float = None, dt: float = None) -> Union[np.ndarray, complex]:
    """
    Returns the linear response function for an input signal x in response to a stimulus force f
    :param x: input signal
    :param f: the stimulus forces
    :param d: the number of input signals
    :param omega: the frequency to evaluate chi at
    :param dt: the time step (needed if omega is not None)
    :return: the linear response function
    """
    if f.ndim != 1:
        raise ValueError('f must be a 1D array')
    if d == 1:
        x_ft = sp.fft.rfft(x - np.mean(x))
        force_ft = sp.fft.rfft(f - np.mean(f))
        chi_ft = x_ft / force_ft
    else:
        x_ft = sp.fft.rfft(x - np.mean(x, axis=1, keepdims=True), axis=1)
        force_ft = sp.fft.rfft(f - np.mean(f))
        chi_ft = x_ft / force_ft
        chi_ft = np.mean(chi_ft, axis=0)
    if omega is not None:
        if dt is None:
            raise ValueError('must provide a time step if provided a frequency to evaluate at')
        freqs = sp.fft.rfftfreq(x.shape[-1], dt)
        index = np.argmin(np.abs(2 * np.pi * freqs - omega))
        return chi_ft[index]
    return chi_ft

def chi_theory(x_amp: np.ndarray, f_amp: float) -> complex:
    """
    Returns the theoretical linear response function for an input signal x in response to a stimulus force f
    :param x_amp: the amplitude of the input signal; if x_amp.shape[0] > 0, this is an array of the amplitudes for each signal in the ensemble
    :param f_amp: the amplitude of the stimulus force
    :return: the theoretical linear response function; averaged if x_amp.shape[0] > 1
    """
    chi_val = np.mean(np.array([-1j * x_amp[i] / f_amp for i in range(x_amp.shape[0])], dtype=complex), dtype=complex)
    return chi_val

def chi_lock(t: np.ndarray, x: np.ndarray, f: np.ndarray, omega: float, t_max: float) -> complex:
    """
    Returns the linear response function for an input signal x in response to a stimulus force f using the lock-in method
    :param t: time
    :param x: input signal
    :param f: the stimulus forces
    :param omega: the frequency to evaluate chi at
    :param t_max: the simulation time
    :return: the linear response function evaluated at omega
    """
    delta_t = t[1] - t[0]
    delta_x = x - np.mean(x, axis=-1, keepdims=True)
    delta_f = f - np.mean(f)
    f_0 = np.max(delta_f)
    chi_real = np.zeros(x.shape[0], dtype=float)
    chi_imag = np.zeros(x.shape[0], dtype=float)
    for i in range(x.shape[0]):
        for j in range(t.shape[0]):
            chi_real[i] += delta_x[j, i] * np.cos(omega * t[i]) * delta_t
            chi_imag[i] += delta_x[j, i] * np.sin(omega * t[i]) * delta_t
    chi_real *= 2 / (f_0 * t_max)
    chi_imag *= 2 / (f_0 * t_max)
    chi_real = np.mean(chi_real, dtype=float)
    chi_imag = np.mean(chi_imag, dtype=float)
    return chi_real + 1j * chi_imag

def fluc_resp(s: np.ndarray, imag_chi: np.ndarray, omegas: np.ndarray, temp: float, boltzmann_scale: float = 1.0, onesided: bool = True) -> np.ndarray:
    """
    Returns the fluctuation response (theta(omega) = omega C(omega) / [2 k_B T chi_I(omega)]) at different driving frequencies omega
    :param s: the power spectral density
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
        theta[i] = omegas[i] * s[i] / (onesided_factor * boltzmann_scale * K_B * temp * np.abs(imag_chi[i]))
    return theta