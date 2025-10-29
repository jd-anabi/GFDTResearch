from typing import Union

import torch
import scipy as sp
import numpy as np
from numpy.ma.core import shape

FREQ_MAX_RANGE = 1.5
FREQ_MIN_RANGE = 0.5

def gen_freqs(omega_0: float, n: int = 5000) -> torch.Tensor:
    """
    Returns an array of driving frequencies around omega_0
    :param omega_0: the frequency to generate the array around
    :param n: number of frequencies to generate
    :return: an array of driving frequencies around omega_0
    """
    if n < 2:
        omegas = torch.linspace(FREQ_MIN_RANGE * omega_0, FREQ_MAX_RANGE * omega_0, n, device=torch.device('cpu'))
        return torch.unique(omegas)
    else:
        omegas = torch.linspace(FREQ_MIN_RANGE * omega_0, FREQ_MAX_RANGE * omega_0, n - 2, device=torch.device('cpu'))
    delta = omegas[1] - omegas[0]
    if np.any(omegas == omega_0):
        omegas[omegas == omega_0] = omega_0 + delta / 2
    omegas = torch.append(omegas, omega_0)
    omegas = torch.append(omegas, 0)
    omegas = torch.sort(omegas)
    return torch.unique(omegas)

def force(t: torch.Tensor, amp: float, omega_0: float, phase: float, offset: float, batch_size: int = 1, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Returns the force at times t
    :param t: time
    :param amp: amplitude
    :param omega_0: angular frequency of spontaneous oscillations
    :param phase: phase of the force
    :param offset: offset of the force
    :param batch_size: batch size of forces
    :param device: device to compute forces on
    :return: the force
    """
    forces = torch.zeros((batch_size, t.shape[0]), dtype=t.dtype, device=device)
    omegas = gen_freqs(omega_0, batch_size).to(device)
    for i in range(batch_size):
        forces[i] = amp * torch.cos(omegas[i] * t + phase) + offset
    return forces

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

def fluc_resp(s: np.ndarray, imag_chi: np.ndarray, omegas: np.ndarray, temp: float, onesided: bool = True) -> np.ndarray:
    """
    Returns the fluctuation response (theta(omega) = omega C(omega) / [2 k_B T chi_I(omega)]) at different driving frequencies omega
    :param s: the power spectral density
    :param imag_chi: the linear response function (imaginary component) in frequency space (specifically at the driving frequencies)
    :param omegas: the driving frequencies (angular frequencies)
    :param temp: the temperature
    :param onesided: whether the PSD is one-sided or not
    :return: the fluctuation response function
    """
    k_b = 1.380649e-23  # m^2 kg s^-2 K^-1
    onesided_factor = 4 if onesided else 2
    theta = np.zeros_like(omegas)
    for i in range(len(theta)):
        theta[i] = omegas[i] * s[i] / (onesided_factor * k_b * temp * np.abs(imag_chi[i]))
    return theta