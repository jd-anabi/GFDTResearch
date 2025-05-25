import warnings
from typing import Any, Callable

import torch
import numpy as np
import scipy.fft as ffts
import scipy.constants as constants
from numpy import ndarray, dtype
from tqdm import tqdm

from core import dimensinal_model as dim_model
from core import dimensional_model_steady as dim_model_steady
from core import sdeint as sdeint
from core import nondimensional_model as nd_model

warnings.filterwarnings('error')

if torch.cuda.is_available():
    DEVICE = torch.device('cpu')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 256 if DEVICE.type == 'cuda' else 5
SDE_TYPES = ['ito', 'stratonovich']

def hb_sols(ts: tuple, n: int, x0: list, params: list, force_params: list, nd: bool) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param ts: time span to evaluate the solution over
    :param n: number of time steps
    :param x0: the initial conditions of the hair bundle
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param force_params: the parameters to use in the stimulus force
    :param nd: whether to use non-dimensional model
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    if DEVICE.type == 'cuda' or DEVICE.type == 'mps':
        print("Using GPU")
    else:
        print("Using CPU")

    # check if we are using the steady-state solution
    if params[0] == 0:
        x0 = x0[:4]
        if nd:
            sde = nd_model.HairBundleSDE(*(params[1:]), *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        else:
            sde = dim_model_steady.HairBundleSDE(*(params[1:]), *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        print("Using the steady-state solution for the open-channel probability")
    else:
        if nd:
            sde = nd_model.HairBundleSDE(*params, *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        else:
            sde = dim_model.HairBundleSDE(*params, *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
    print("Hair bundle model has been set up")

    # setting up initial conditions
    x0s = torch.tensor(x0, dtype=DTYPE, device=DEVICE)
    x0s = torch.tile(x0s, (BATCH_SIZE, 1)) # size: (BATCH_SIZE, len(x0))

    # time array
    chunk_size = int((n - 1) / 1e3)
    t = np.linspace(*ts, n)

    # solving a system of SDEs and implementing a progress bar (this is cool fyi)
    solver = sdeint.Solver()
    hb_sol = torch.zeros((n, BATCH_SIZE, len(x0)), dtype=DTYPE, device=DEVICE)
    hb_sol[0] = x0s
    for i in tqdm(range(n-1, chunk_size), desc=f"Simulating {BATCH_SIZE} batches of hair-bundles", mininterval=0.1):
        with torch.no_grad():
            try:
                hb_sol[i+1] = solver.euler(sde, x0s, (t[i], t[i+1]), 2)[-1] # only keep the last solution
                x0s = hb_sol[i+1]
            except (Warning, Exception) as e:
                print(e)
                exit()

    print("SDEs have been solved")
    print(hb_sol)
    return hb_sol.cpu().numpy()

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