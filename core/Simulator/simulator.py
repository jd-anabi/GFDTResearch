import numpy as np
import torch

from core.Solvers import sdeint
from core.Models import nondimensional_model as nd_model, steady_nondimensional_model as steady_nd_model
from core.Helpers import gen_helpers as gh

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 2048 if DEVICE.type == 'cuda' else 64
SDE_TYPES = ['ito', 'stratonovich']

def sols(t: np.ndarray, x0: np.ndarray, params: list, force: np.ndarray, explicit: bool = True) -> np.ndarray:
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
    if params[3] == 0:
        x0 = x0[:, :4]
        sde = steady_nd_model.HairBundleSDE(*params, force, batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)
    else:
        sde = nd_model.HairBundleSDE(*params, force, batch_size=BATCH_SIZE, device=DEVICE, dtype=DTYPE).to(DEVICE)

    # setting up initial conditions
    x0s = torch.tensor(x0, dtype=DTYPE, device=DEVICE)

    # time array
    n = len(t)
    ts = [t[0], t[-1]]

    # solving a system of SDEs and implementing a progress bar (this is cool fyi)
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

def sim(t: np.ndarray, init_conds: np.ndarray, params: list, force: np.ndarray,
              n_time_segs: int, batch_size: int, freqs_per_batch: int) -> np.ndarray:
    ensemble_size = batch_size // freqs_per_batch
    time_seg_ids = gh.get_even_ids(len(t), n_time_segs + 1)

    inits = init_conds
    n_vars = inits.shape[1] if params[3] != 0 else inits.shape[1] - 1
    sol = np.zeros((n_vars, BATCH_SIZE, len(t)))
    #curr_force_batch = gh.repeat1d(force[freqs_per_batch:freqs_per_batch, :], ensemble_size, BATCH_SIZE)
    #curr_batch_phases = gh.sde_tile1d(phases[:freqs_per_batch], ensemble_size, BATCH_SIZE)
    #curr_batch_omegas = gh.sde_tile1d(omegas[freqs_per_batch:freqs_per_batch], ensemble_size, BATCH_SIZE)
    for tid in range(len(time_seg_ids) - 1):
        print(f"Time segment {tid + 1}:")
        curr_time = t[time_seg_ids[tid]:time_seg_ids[tid + 1]]
        args_list = (curr_time, inits, list(params), force)
        results = sols(*args_list) # shape: (len(curr_time), BATCH_SIZE, number of variables)

        # update initial conditions
        inits = results[-1, :, :]

        # extract position data
        sol[:, :, time_seg_ids[tid]:time_seg_ids[tid + 1]] = results.T # shape: (number of variables, BATCH_SIZE, len(curr_time))
    # rescale position data for later
    sol = sol.reshape(inits.shape[1], freqs_per_batch, ensemble_size, len(t)) # shape: (number of variables, freqs_per_batch, ensemble_size, len(curr_time))
    return sol