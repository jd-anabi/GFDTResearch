import warnings

import scipy as sp
import torch
import numpy as np

import sdeint as sdeint
import nondimensional_model as nd_model
import steady_nondimensional_model as steady_nd_model

warnings.filterwarnings('error')

if torch.cuda.is_available():
    DEVICE = torch.device('cpu')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 256 if DEVICE.type == 'cuda' else 12
SDE_TYPES = ['ito', 'stratonovich']

def hb_sols(t: np.ndarray, x0: list, params: list, force_params: list) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time array
    :param x0: the initial conditions of the hair bundle
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
        x0 = x0[:4]
        sde = steady_nd_model.HairBundleSDE(*params, *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
        print("Using the steady-state solution for the open-channel probability")
    else:
        sde = nd_model.HairBundleSDE(*params, *force_params, sde_type=SDE_TYPES[0], batch_size=BATCH_SIZE,device=DEVICE, dtype=DTYPE).to(DEVICE)
        print("Hair bundle model has been set up")

    # setting up initial conditions
    x0s = torch.tensor(x0, dtype=DTYPE, device=DEVICE)
    x0s = torch.tile(x0s, (BATCH_SIZE, 1)) # size: (BATCH_SIZE, len(x0))

    # time array
    n = len(t)
    ts = [t[0], t[-1]]

    # solving a system of SDEs and implementing a progress bar (this is cool fyi)
    solver = sdeint.Solver()
    hb_sol = torch.zeros((n, BATCH_SIZE, len(x0)), dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        try:
            hb_sol = solver.euler(sde, x0s, ts, n) # only keep the last solution
        except (Warning, Exception) as e:
            print(e)
            exit()

    print("SDEs have been solved")
    return hb_sol.cpu().detach().numpy()

def rescale(nd_hb_pos: np.ndarray, nd_sf_pos: np.ndarray, nd_t: np.ndarray,
            gamma: float, d: float, x_sp: float, k_sp: float, k_sf: float, k_gs_max: float,
            s_max: float, t_0: float, s_max_nd: float, chi_hb: float, chi_a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rescaling the hair-bundle displacement, the stimulus force, and the time
    :param nd_hb_pos: the hair bundle position
    :param nd_sf_pos: the stimulus force position
    :param nd_t: the time array
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param x_sp: resting deflection of stereociliary pivots
    :param k_sp: stiffness of stereociliary pivots
    :param k_sf: stiffness of stimulus force
    :param k_gs_max: maximum stiffness of gating spring
    :param s_max: maximum slipping rate
    :param t_0: time offset
    :param s_max_nd: non-dimensional maximum slipping rate
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :param chi_a: non-dimensional parameter for non-dimensional adaptation motor displacement
    :return: tuple containing the rescaled hair-bundle displacement, stimulus force, and time
    """
    hb_pos = chi_hb * d / gamma * nd_hb_pos + k_sp / (k_sp + k_sf) * x_sp
    sf_pos = chi_hb * (k_sp + k_sf) * d / (gamma * k_sf) * nd_sf_pos
    t = chi_a * s_max_nd / (k_gs_max * s_max) * nd_t - t_0
    return hb_pos, sf_pos, t

def sf_pos(t: np.ndarray, amp: float, omega_0: float) -> np.ndarray:
    """
    Returns the stimulus force position at times t
    :param t: time
    :param amp: amplitude
    :param omega_0: angular frequency of spontaneous oscillations
    :return: the stimulus force position
    """
    sf_pos_data = np.zeros((BATCH_SIZE, len(t)))
    for i in range(BATCH_SIZE):
        omega = i * omega_0 / round(BATCH_SIZE / 2)
        sf_pos_data[i] = amp * np.sin(omega * t)
    return sf_pos_data

def auto_corr(hb_pos: np.ndarray) -> np.ndarray:
    """
    Returns the (normalized) auto-correlation function <X(t) X(0)> for the position of a hair bundle
    :param hb_pos: the position of a hair bundle
    :return: the auto-correlation function
    """
    hb_pos = hb_pos - np.mean(hb_pos)
    c = np.correlate(hb_pos, hb_pos, mode='full')
    c = c[len(hb_pos) - 1:]
    return c / (len(hb_pos) * c[0])

def lin_resp_ft(hb_pos: np.ndarray, sf: np.ndarray, shift: bool = True) -> np.ndarray:
    """
    Returns the linear response function (in frequency space) for the position of a hair bundle in response to a stimulus force
    :param hb_pos: the position of a hair bundle
    :param sf: the stimulus forces
    :param shift: whether to perform a shift on the response function before returning it
    :return: the linear response function (in frequency space)
    """
    # compute the Fourier Transform
    hb_pos_ft = sp.fft.fft2(hb_pos - np.mean(hb_pos))
    sf_pos_ft = sp.fft.fft2(sf - np.mean(sf))
    if shift:
        return sp.fft.fftshift(hb_pos_ft / sf_pos_ft)
    return hb_pos_ft / sf_pos_ft