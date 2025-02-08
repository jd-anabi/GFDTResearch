import sdeint
import numpy as np

import hair_bundle_nondimensional as hb_nd

def nd_hb_sols(t: np.ndarray, pt_steady_state: bool, params: list, x0: list) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time to solve sdes at
    :param pt_steady_state: determines whether to use the steady state solution for the open channel probability
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param x0: the initial conditions of the hair bundle
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    nd_hb = hb_nd.HairBundleNonDimensional(*params, pt_steady_state)
    hb_sols = sdeint.itoEuler(nd_hb.f, nd_hb.g, x0, t)
    return hb_sols