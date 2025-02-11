import sdeint
import numpy as np
import scipy.integrate as integrate
import scipy.fft as ffts
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

def corr(hb_pos_trials: list) -> np.ndarray:
    """
    Returns the autocorrelation function <X(t) X(0)> for an ensemble of hair bundle positions
    :param hb_pos_trials: set of hair bundle positions
    :return: the autocorrelation function
    """
    c = np.zeros(len(hb_pos_trials[0]))
    for i in range(len(hb_pos_trials[0])):
        for j in range(len(hb_pos_trials)):
            c[i] = hb_pos_trials[j][i] * hb_pos_trials[j][0]
    c = c / len(hb_pos_trials)
    return c

def fdt_ratio(hb_pos_trials: list, x_sf: Symbol, omegas: list, t: list, dt: float) -> list:
    """
    Returns the fluctuation-response ratio given a set of hair-bundle positions (driven at different frequencies)
    :param hb_pos_trials: set of hair-bundle positions at different frequencies
    :param x_sf: applied stimulus force to the hair bundle
    :param omegas: driving frequencies
    :param t: time that the hb positions are at
    :param dt: time step
    :return: the fluctuation-response ratio at different frequencies
    """
    x_sfs = np.zeros((len(omegas), len(t)))
    x_sf_freq = np.zeros((len(omegas), len(t)))
    for i in range(len(omegas)):
        for j in range(len(t)):
            x_sfs[i][j] = x_sf.subs([symbols('omega'), omegas[i]], [symbols('t'), t[j]])
    x_sfs_freq = ffts.fft(x_sfs, axis=1)
    avg_hb_pos_freq = np.mean(ffts.fft(hb_pos_trials, axis=1), axis=0)
