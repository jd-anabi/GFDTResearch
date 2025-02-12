import sdeint
import numpy as np
import scipy.fft as ffts
import scipy.signal as signal
from sympy import Symbol, symbols

import hair_bundle_nondimensional as hb_nd

def nd_hb_sols(t: np.ndarray, pt_steady_state: bool, s_osc: float, params: list, x0: list) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time to solve sdes at
    :param pt_steady_state: determines whether to use the steady state solution for the open channel probability
    :param s_osc: spontaneous oscillation frequency
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param x0: the initial conditions of the hair bundle
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    nd_hb = hb_nd.HairBundleNonDimensional(*params, s_osc, pt_steady_state)
    hb_sols = sdeint.itoEuler(nd_hb.f, nd_hb.g, x0, t)
    return hb_sols

def auto_corr(hb_pos: list) -> np.ndarray:
    """
    Returns the auto-correlation function <X(t) X(0)> for the position of a hair bundle
    :param hb_pos: the position of a hair bundle
    :return: the auto-correlation function
    """
    c = signal.correlate(hb_pos, hb_pos, mode='full', method='auto')
    return c[len(hb_pos) - 1:]

def lin_resp_freq(hb_pos: list, x_sf: list) -> complex:
    """
    Returns the linear response function of the hair bundle with a stimulating force applied
    :param hb_pos: hair bundle position
    :param x_sf: stimulating force
    :return: the linear response function
    """
    hb_pos_freq = ffts.fftshift(ffts.fft(hb_pos - np.mean(hb_pos))) # type: np.ndarray
    x_sf_freq = ffts.fftshift(ffts.fft(x_sf - np.mean(x_sf))) # type: np.ndarray
    hb_pos_freq_mean = np.mean(np.abs(hb_pos_freq))
    x_sf_freq_max = np.max(np.abs(x_sf_freq))
    return hb_pos_freq_mean / x_sf_freq_max

def fdt_ratio(hb_pos_trials: list, x_sfs: list) -> np.ndarray:
    """
    Returns the fluctuation-response ratio given a set of hair-bundle positions (driven at different frequencies)
    :param hb_pos_trials: set of hair-bundle positions at different frequencies
    :param x_sfs: set of applied stimulus force to the hair bundle at different frequencies
    :return: the fluctuation-response ratio at different frequencies
    """
    auto_corrs = np.zeros(len(hb_pos_trials), dtype=np.ndarray)
    lin_resp_freqs = np.zeros(len(hb_pos_trials), dtype=complex)
    for i in range(len(hb_pos_trials)):
        auto_corrs[i] = auto_corr(hb_pos_trials[i])
        lin_resp_freqs[i] = lin_resp_freq(hb_pos_trials[i], x_sfs[i])
    auto_corrs_freqs = ffts.fftshift(ffts.fft(auto_corrs, axis=1), axes=1) # type: np.ndarray
    theta = np.zeros(len(hb_pos_trials), dtype=float)
    for i in range(len(hb_pos_trials)):
        omega_index = np.where(np.abs(auto_corrs_freqs[i]) == np.max(np.abs(auto_corrs_freqs[i])))
        theta[i] = auto_corrs_freqs[i][omega_index] / np.imag(lin_resp_freqs[i])
    return theta