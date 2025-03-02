from typing import Any

import sdeint
import numpy as np
import scipy.fft as ffts
import scipy.signal as signal
import scipy.constants as constants
from numpy import ndarray, dtype

import hair_bundle as hb
import hair_bundle_nondimensional as hb_nd

def hb_sols(t: np.ndarray, pt_steady_state: bool, s_osc: float, params: list, x0: list, nd: bool) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time to solve sdes at
    :param pt_steady_state: determines whether to use the steady state solution for the open channel probability
    :param s_osc: spontaneous oscillation frequency
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param x0: the initial conditions of the hair bundle
    :param nd: whether to use the non-dimensional model or not
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    if nd:
        hb_mod = hb_nd.HairBundleNonDimensional(*params, s_osc, pt_steady_state)
    else:
        hb_mod = hb.HairBundle(*params, s_osc, pt_steady_state)
    hb_sol = sdeint.itoEuler(hb_mod.f, hb_mod.g, x0, t)
    return hb_sol

def auto_corr(hb_pos: list) -> np.ndarray:
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

def fdt_ratio(omega: float, hb_trials: np.ndarray, x_sf: np.ndarray, dt: float, inc: bool) -> list:
    """
    Returns the fluctuation-response ratio, at a given frequency, given a set of an ensemble of hair-bundle positions
    :param omega: the frequency to calculate the ratio at
    :param hb_trials: set of an ensemble of hair-bundle positions at different frequencies
    :param x_sf: applied stimulus force at a given frequency omega
    :param dt: time step
    :param inc: whether to include the autocorrelation function and linear response in return statement
    :return: the fluctuation-response ratio at different frequencies (and possibly the autocorrelation function and linear response); [ratio,
    """
    # generate auto-correlation function in frequency space
    autocorr = auto_corr(np.mean(hb_trials, axis=0))
    autocorr_freq = ffts.fft(autocorr)
    autocorr_freq = ffts.fftshift(autocorr_freq) # type: np.ndarray
    # generate frequency array and shift it
    freqs = ffts.fftfreq(len(autocorr), d=dt)
    freqs = ffts.fftshift(freqs)
    # get auto-correlation function and linear response function at a specific frequency
    index = np.argmin(np.abs(freqs - omega / (2 * constants.pi)))  # calculate the index closest to the desired frequency
    autocorr_omega = autocorr_freq[index]
    lin_resp_omega = lin_resp_freq(omega, hb_trials, x_sf, dt)
    theta = omega * np.abs(autocorr_omega) / np.imag(lin_resp_omega)
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

def p_t0(x_hb: np.ndarray, x_a: np.ndarray, p_gs: np.ndarray,
         u_gs_max: float, delta_e: float, k_gs_min: float,
         chi_hb: float, chi_a: float, x_c: float) -> np.ndarray:
    """
    Steady-state solution for the open-channel probability
    :param x_hb: hair bundle displacement
    :param x_a: adaptation motor displacement
    :param p_gs: calcium binding probability for gating spring
    :param u_gs_max: max gating spring potential
    :param delta_e: intrinsic energy difference between the transduction channel's two states
    :param k_gs_min: min gating spring stiffness
    :param chi_hb: hair bundle conversion factor
    :param chi_a: adaptation motor conversion factor
    :param x_c: average equilibrium position of the adaptation motors
    :return: steady-state solution for the open-channel probability
    """
    k_gs_var = 1 - p_gs * (1 - k_gs_min)
    x_gs = chi_hb * x_hb - chi_a * x_a + x_c
    return 1 / (1 + np.exp(u_gs_max * (delta_e - k_gs_var * (x_gs - 0.5))))

def k_gs(p_gs: np.ndarray, k_gs_min: float) -> np.ndarray:
    """
    Gating spring stiffness
    :param p_gs: calcium binding probability for gating spring
    :param k_gs_min: min gating spring stiffness
    :return: gating spring stiffness
    """
    return 1 - p_gs * (1 - k_gs_min)