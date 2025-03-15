from typing import Any

import sdeint
import numpy as np
import scipy.fft as ffts
import scipy.signal as signal
import scipy.constants as constants
from numpy import ndarray, dtype

import hair_bundle as hb
import hair_bundle_nondimensional as hb_nd

def hb_sols(t: np.ndarray, pt_steady_state: bool, s_osc: float, params: list, x0: list, nd: bool, a, b) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time to solve sdes at
    :param pt_steady_state: determines whether to use the steady state solution for the open channel probability
    :param s_osc: spontaneous oscillation frequency
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param x0: the initial conditions of the hair bundle
    :param nd: whether to use the non-dimensional model or not
    :param a: the amplitude of the driving force
    :param b: the amplitude of the viscous part of the driving force
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    if nd:
        hb_mod = hb_nd.HairBundleNonDimensional(*params, s_osc, pt_steady_state, a, b)
    else:
        hb_mod = hb.HairBundle(*params, s_osc, pt_steady_state, a, b)
    hb_sol = sdeint.itoEuler(hb_mod.f, hb_mod.g, x0, t)
    return hb_sol

def auto_corr(hb_pos: np.ndarray) -> np.ndarray:
    """
    Returns the auto-correlation function <X(t) X(0)> for the position of a hair bundle
    :param hb_pos: the position of a hair bundle
    :return: the auto-correlation function
    """
    c = signal.correlate(hb_pos, hb_pos, mode='full', method='auto')
    return c[len(hb_pos) - 1:]

def lin_resp_freq(omega: float, hb_trials: np.ndarray, x_sf: np.ndarray, dt: float) -> ndarray[Any, dtype[complex]]:
    """
    Returns the linear response function, at a specific frequency, of the hair bundle (ran over multiple trials) with a stimulating force applied
    :param omega: the frequency tin calculate the response function at
    :param hb_trials: trials of hair bundle position
    :param x_sf: stimulating force
    :param dt: time step, otherwise known as the sample spacing
    :return: the linear response function
    """
    # transfer to the frequency domain
    hb_avg_pos = np.mean(hb_trials, axis=0)
    hb_freq = ffts.fft(hb_avg_pos - np.mean(hb_avg_pos))
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

def fdt_ratio(omega: float, hb_pos_undriven: np.ndarray, hb_trials: np.ndarray, x_sf: np.ndarray, dt: float, inc: bool) -> list:
    """
    Returns the fluctuation-response ratio, at a given frequency, given a set of an ensemble of hair-bundle positions
    :param omega: the frequency to calculate the ratio at
    :param hb_pos_undriven: the undriven hair bundle position
    :param hb_trials: set of an ensemble of hair-bundle positions at different frequencies
    :param x_sf: applied stimulus force at a given frequency omega
    :param dt: time step
    :param inc: whether to include the autocorrelation function and linear response in return statement
    :return: the fluctuation-response ratio at different frequencies (and possibly the autocorrelation function and linear response); [ratio,
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
    lin_resp_omega = lin_resp_freq(omega, hb_trials, x_sf, dt)
    theta = omega * autocorr_omega / np.imag(lin_resp_omega)
    if inc:
        return [theta, autocorr, lin_resp_omega]
    return [theta]

def log(x, a, b, c):
    """
    Natural logarithm of a * ln(b * x) + c; used for fitting function
    :param x: input value
    :param a: multiplicative log coefficient
    :param b: multiplicative log argument coefficient
    :param c: additive constant
    :return: natural logarithm
    """
    return a * np.log(b * x) + c

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

def k_gs(p_gs: np.ndarray, k_gs_min: float) -> np.ndarray:
    """
    Gating spring stiffness
    :param p_gs: calcium binding probability for gating spring
    :param k_gs_min: min gating spring stiffness
    :return: gating spring stiffness
    """
    return 1 - p_gs * (1 - k_gs_min)