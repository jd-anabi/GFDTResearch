from typing import Any, Callable

import torch
import torchsde
import numpy as np
import scipy.fft as ffts
import scipy.constants as constants
from numpy import ndarray, dtype

import hair_bundle_sde as hb_sde

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
BATCH_SIZE = 3

def hb_sols(t_span: tuple, dt: float, x0: list, params: list, force_params: list) -> list:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t_span: time span to solve sdes for
    :param dt: time step
    :param x0: the initial conditions of the hair bundle
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param force_params: the parameters to use in the stimulus force
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    num_time_steps = int((t_span[1] - t_span[0]) / dt)
    #init_conditions = torch.tensor(x0, dtype=DTYPE, device=DEVICE)
    #init_conditions = torch.tile(init_conditions, (BATCH_SIZE, 1))
    init_conditions = torch.rand(BATCH_SIZE, len(x0), dtype=DTYPE, device=DEVICE)
    t = torch.linspace(t_span[0], t_span[1], num_time_steps, dtype=DTYPE, device=DEVICE)
    sde = hb_sde.HairBundleSDE(params, force_params, 'diagonal', 'ito').to(DEVICE)
    print("SDE set up")
    with torch.no_grad():
        hb_sol = torchsde.sdeint(sde, init_conditions, t, method='euler', dt=dt, adaptive=True)
    print(hb_sol)
    print("SDE solved")
    hb_sols_to_np = []
    for i in range(len(x0)):
        hb_sols_to_np.append(hb_sol[:, 0, i].cpu().numpy())
    return hb_sols_to_np

def auto_corr(hb_pos: np.ndarray) -> np.ndarray:
    """
    Returns the auto-correlation function <X(t) X(0)> for the position of a hair bundle
    :param hb_pos: the position of a hair bundle
    :return: the auto-correlation function
    """
    hb_pos = hb_pos - np.mean(hb_pos)
    c = np.correlate(hb_pos, hb_pos, mode='full')
    c = c[len(hb_pos) - 1:]
    return c / (len(hb_pos) * c[0])

def lin_resp_freq(omega: float, hb_driven: np.ndarray, x_sf: np.ndarray, dt: float) -> ndarray[Any, dtype[Any]]:
    """
    Returns the linear response function, at a specific frequency, of the hair bundle (ran over multiple trials) with a stimulating force applied
    :param omega: the frequency tin calculate the response function at
    :param hb_trials: trials of hair bundle position
    :param x_sf: stimulating force
    :param dt: time step, otherwise known as the sample spacing
    :return: the linear response function
    """
    # transfer to the frequency domain
    hb_freq = ffts.fft(hb_driven - np.mean(hb_driven))
    x_sf_freq = ffts.fft(x_sf - np.mean(x_sf))
    # shift ffts
    hb_freq = ffts.fftshift(hb_freq) # type: np.ndarray
    x_sf_freq = ffts.fftshift(x_sf_freq) # type: np.ndarray
    # generate frequency array and shift it
    freqs = ffts.fftfreq(len(x_sf), d=dt)
    freqs = ffts.fftshift(freqs)
    # now for the fun (hard) part, extracting the values at a given frequency
    index = np.argmin(np.abs(freqs - omega / (2 * constants.pi))) # calculate the index closest to the desired frequency
    return hb_freq[index] / x_sf_freq[index]

def fdt_ratio(omega: float, hb_pos_undriven: np.ndarray, hb_driven: np.ndarray, x_sf: np.ndarray, dt: float) -> np.ndarray:
    """
    Returns the fluctuation-response ratio, at a given frequency, given a set of an ensemble of hair-bundle positions
    :param omega: the frequency to calculate the ratio at
    :param hb_pos_undriven: the undriven hair bundle position
    :param hb_driven: set of an ensemble of hair-bundle positions at different frequencies
    :param x_sf: applied stimulus force at a given frequency omega
    :param dt: time stept
    :return: the fluctuation-response ratio at a specific frequencies
    """
    # generate auto-correlation function in frequency space
    autocorr = auto_corr(hb_pos_undriven)
    autocorr_freq = ffts.fft(autocorr)
    autocorr_freq = ffts.fftshift(autocorr_freq) # type: np.ndarray
    # generate frequency array and shift it
    freqs = ffts.fftfreq(len(autocorr), d=dt)
    freqs = ffts.fftshift(freqs)
    # get auto-correlation function and linear response function at a specific frequency
    index = np.argmin(np.abs(freqs - omega / (2 * constants.pi)))  # calculate the index closest to the desired frequency
    autocorr_omega = autocorr_freq[index]
    lin_resp_omega = lin_resp_freq(omega, hb_driven, x_sf, dt)
    theta = omega * autocorr_omega / np.imag(lin_resp_omega)
    return theta

def p_t0(x_hb: np.ndarray, x_a: np.ndarray, p_gs: np.ndarray, params: list, nd: bool) -> np.ndarray:
    """
    Steady-state solution for the open-channel probability
    :param x_hb: hair bundle displacement
    :param x_a: adaptation motor displacement
    :param p_gs: calcium binding probability for gating spring
    :param params: list of parameters for steady-state solution; [u_gs_max, delta_e, k_gs_min, chi_hb, chi_a, x_c] if non-dimensional model, [delta_e, k_gs_min, k_gs_max, x_c, temp, d] otherwise
    :param nd: whether to use the non-dimensional model or not
    :return: steady-state solution for the open-channel probability
    """
    if nd:
        k_gs_var = 1 - p_gs * (1 - params[2])
        x_gs = params[3] * x_hb - params[4] * x_a + params[5]
        return 1 / (1 + np.exp(params[0] * (params[1] - k_gs_var * (x_gs - 0.5))))
    else:
        k_gs_var = params[2] - p_gs * (params[2] - params[1])
        x_gs = k_gs_var * params[5] / (constants.k * params[4]) * (x_hb - x_a + params[3] - params[5] / 2)
        return 1 / (1 + np.exp(params[0] / (constants.k * params[4]) - x_gs))

def itoEuler(f: Callable[[float, ndarray], ndarray], G: ndarray, y0: ndarray, t: ndarray, generator=None, nan_index: float = 10):
    if generator is None:
        generator = np.random.default_rng()
    t_size = t.size
    y_count = y0.shape[0]
    dt = (t[t_size - 1] - t[0]) / (t_size - 1)
    y = np.zeros((t_size, y_count), dtype=y0.dtype)
    eta = G * generator.normal(0, np.sqrt(dt), (t_size, y_count))
    y[0] = y0
    yn = y0
    for n in range(nan_index):
        yn = yn + f(yn, t[n]) * dt + eta[n, :]
        y[n + 1, :] = yn
    if np.any(np.isnan(y[:nan_index, :])):
        y[nan_index:, :] = np.nan
        return y
    for n in range(nan_index, t_size - 1):
        yn = yn + f(yn, t[n]) * dt + eta[n, :]
        y[n + 1, :] = yn
    return y.T