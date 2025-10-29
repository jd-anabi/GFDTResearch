import math
from typing import Union

import numpy as np
import torch

from core.Solvers import sdeint
from core.Models import model, steady_model
from core.Helpers import gen_helpers as helpers, hair_model_helpers as model_helpers, fdt_helpers as fdt

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    if (torch.cuda.get_device_properties(DEVICE).major, torch.cuda.get_device_properties(DEVICE).minor) < (8, 0):
        DEVICE = torch.device('cpu')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 2**11 if DEVICE.type == 'cuda' else 2**6

def sols(t: np.ndarray, x0: np.ndarray, params: Union[list, np.ndarray], force: np.ndarray, explicit: bool = True) -> np.ndarray:
    """
    Returns sde solution for a hair bundle given a set of parameters and initial conditions
    :param t: time array
    :param x0: the initial conditions of each simulation
    :param params: the parameters to use in the for the non-dimensional hair bundle constructor
    :param force: the force to use in the non-dimensional hair bundle constructor
    :param explicit: whether to use the explicit Euler-Maruyama method
    :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
    """
    # check if we are using the steady-state solution
    if params[0, 3] == 0:
        x0 = x0[:, :4]
        sde = steady_model.HairBundleSDE(*params, force, batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
    else:
        sde = model.HairBundleSDE(*params, force, batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)

    # setting up initial conditions
    x0s = torch.tensor(x0, dtype=DTYPE, device=DEVICE)

    # time array
    n = len(t)
    ts = [t[0], t[-1]]

    # solving a system of SDEs
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

def sim(t: np.ndarray, inits: np.ndarray, params: Union[list, np.ndarray], force: np.ndarray, n_time_segs: int, batch_size: int, freqs_per_batch: int) -> np.ndarray:
    ensemble_size = batch_size // freqs_per_batch
    time_seg_ids = helpers.get_even_ids(len(t), n_time_segs + 1)

    init = inits
    n_vars = init.shape[1] if params[0, 3] != 0 else init.shape[1] - 1
    sol = np.zeros((n_vars, batch_size, len(t)))
    for tid in range(len(time_seg_ids) - 1):
        print(f"Time segment {tid + 1}:")
        curr_time = t[time_seg_ids[tid]:time_seg_ids[tid + 1]]
        args_list = (curr_time, init, list(params), force)
        results = sols(*args_list) # shape: (len(curr_time), BATCH_SIZE, number of variables)

        # update initial conditions
        init = results[-1, :, :]

        # extract position data
        sol[:, :, time_seg_ids[tid]:time_seg_ids[tid + 1]] = results.T # shape: (number of variables, BATCH_SIZE, len(curr_time))
    # rescale position data for later
    sol = sol.reshape(init.shape[1], freqs_per_batch, ensemble_size, len(t)) # shape: (number of variables, freqs_per_batch, ensemble_size, len(curr_time))
    return sol

def get_moments(x: torch.Tensor) -> torch.Tensor:
    # handle shape (n, t) -> (1, t)
    if x.dim() == 2:
        x_obs = x[0, :].unsqueeze(0)
    # handle shape (n, b, t) -> (b, t)
    elif x.dim() == 3:
        x_obs = x[0, :, :]
    # handle shape (t) for real data -> (1, t)
    else:
        x_obs = x.unsqueeze(0)
    mean = x_obs.mean(dim=-1)
    var = x_obs.var(dim=-1)
    std = var.sqrt()
    norm = (x_obs - mean.unsqueeze(-1)) / std.unsqueeze(-1)
    skew = (norm**3).mean(dim=-1)
    kurt = (norm**4).mean(dim=-1)
    return torch.stack((mean, var, skew, kurt), dim=-1) # shape: (4, b, 1)

def sim_wrapper(params: torch.Tensor, context: np.ndarray, ts: tuple = (0, 500), dt: float = 1e-2) -> torch.Tensor:
    sim_params = params[:17].cpu().detach().numpy()
    rescale_params = params[17:].cpu().detach().numpy()
    batch_size = params.shape[0] # extract batch size
    inits = np.random.randint(0, 1, size=(batch_size, 5)) # set up some arbitrary default initial conditions
    t = np.linspace(ts[0], ts[1], num=int((ts[-1] - ts[0]) / dt))
    x = sim(t, inits, sim_params, context, math.ceil(ts[1] / 200), batch_size, batch_size).squeeze() # shape: (number of variables, batch size, len(curr_time))
    args = [rescale_params[0], rescale_params[1], rescale_params[2], rescale_params[3], rescale_params[7], sim_params[12]]
    x = model_helpers.rescale_x(x, *args)
    x = torch.tensor(x, dtype=DTYPE, device=DEVICE)
    return get_moments(x)
