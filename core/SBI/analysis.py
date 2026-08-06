import math
import warnings

import torch
from torch.distributions.transforms import Transform

from core.SBI import pipeline
from core.config import CAL_N_SCALES, CAL_RUN_SIZE, CAL_RUN_SIZE_MAX
from core.SBI.reparam import _transform_device

# === POSTERIOR PREDICTIVE CHECK ===
def posterior_predictive_check(s_obs: torch.Tensor, s_simulated: torch.Tensor) -> dict:
    """
    Performs a posterior predictive check by comparing observed statistics with simulated statistics,
    providing metrics such as z-scores, absolute z-score statistics, and coverage of observed values
    within a confidence interval.

    The method calculates normalized z-scores for observed statistics (`s_obs`) based on the mean and
    standard deviation of simulated statistics (`s_simulated`). It also computes diagnostic metrics,
    such as the fraction of observations within a 90% confidence interval derived from the simulated data,
    as well as counts of invalid statistics (zero variance) and those outside the confidence interval.

    :param s_obs: Observed statistics tensor.
    :param s_simulated: Simulated statistics tensor with samples along the first dimension.
    :return: A dictionary containing calculated z-scores, absolute z-statistics, coverage fraction, and counts of
        invalid observations or those outside the interval.
    """
    # per-statistic mean and std
    s_mean = s_simulated.mean(dim=0)
    s_std = s_simulated.std(dim=0)

    # handle zero variance statistics
    valid_mask = s_std > 1e-10
    z_scores = torch.full_like(s_obs, float('nan'))
    z_scores[valid_mask] = (s_obs[valid_mask] - s_mean[valid_mask]) / s_std[valid_mask]

    # compute metrics only on valid statistics
    valid_z = z_scores[valid_mask]

    lower = torch.quantile(s_simulated, 0.05, dim=0)
    upper = torch.quantile(s_simulated, 0.95, dim=0)
    within_interval = (s_obs >= lower) & (s_obs <= upper)
    coverage_fraction = within_interval.float().mean().item()

    return {
        "z_scores": z_scores,
        "mean_abs_z": valid_z.abs().mean().item(),
        "max_abs_z": valid_z.abs().max().item(),
        "coverage_90": coverage_fraction,
        "num_outside": (~within_interval).sum().item(),
        "num_invalid": (~valid_mask).sum().item()
    }

# === COVERAGE CHECKS ===
def gen_cal_data(model: str, prior: torch.distributions.Distribution,
                 forcing_prior: torch.distributions.Distribution,
                 t: torch.Tensor, steady_idx: int, dt_nd_min: float, n_cal: int,
                 nd_dim: int, forcing_idx: dict, rescale_idx: dict,
                 dt_exp: float = None, t_min_exp: float = None, t_max_exp: float = None,
                 t_scale_bounds: tuple[float, float] = None, theta_transform: Transform | None = None,
                 fixed_dict: dict = None, state_dep_drift: bool = False,
                 spontaneous_only: bool = False, chi_mode: bool = False,
                 chi_f0: float | None = None,
                 chi_freq_bounds: tuple | None = None, chi_k_pad: int | None = None,
                 chi_k_fixed: int | None = None,
                 n_vars: int | None = None,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates calibration data and filtered parameters for model training based on the provided input parameters.

    If theta_transform is provided, `prior` is LATENT. Internally gen_training_data samples
    z, simulates via T(z), and returns latent z as theta_star. This function then applies
    theta_transform to convert theta_star back to physical coordinates, so SBC/coverage
    can compare it directly against the physical-space TransformedPosterior.

    :param model: Name of the model to evaluate. Must be provided as a string.
    :param prior: Distribution object representing the prior over the model parameters.
    :param t: Pre-simulated ND time tensor at finest resolution, provided as a tensor.
    :param steady_idx: Index defining the steady-state position in the simulation points.
    :param dt_nd_min: Finest ND time step of the pre-simulated trajectory.
    :param n_cal: Number of calibration data samples to generate.
    :param dt_exp: Fixed experimental sampling interval (seconds).
    :param t_min_exp: Shortest experimental recording duration (seconds).
    :param t_max_exp: Longest experimental recording duration (seconds).
    :param t_scale_bounds: (lo, hi) bounds on the t_scale rescaling parameter.
    :param fixed_dict: Dictionary containing fixed parameter values for simulation (default is None).
    :param dtype: Data type for tensor computations (default is torch.float32).
    :param device: Device where computations will be performed (default is CPU).
    :return: A tuple containing filtered calibration data (torch.Tensor) and corresponding parameters
             (torch.Tensor) that exclude invalid simulations.
    """
    # Spread n_cal over at most CAL_N_SCALES batches -- batch COUNT is the simulation cost and the
    # (t_scale, T) diversity, while batch SIZE is nearly free (the solver is kernel-launch-bound).
    # See the CAL_N_SCALES block in config.py.
    #
    # CAL_RUN_SIZE is the FLOOR, so every n_cal at or below CAL_N_SCALES x CAL_RUN_SIZE behaves
    # exactly as it always did (n_cal=2000 -> 200 x 10; the suites' n_cal=40/60 -> 4/6 x 10). Only a
    # LARGER n_cal starts growing the batch instead of the batch count, which is what makes extra
    # statistical power essentially free.
    cal_run_size = min(n_cal, max(CAL_RUN_SIZE,
                                  min(CAL_RUN_SIZE_MAX, math.ceil(n_cal / max(1, CAL_N_SCALES)))))
    cal_run_size = max(1, cal_run_size)
    cal_n_runs = max(1, math.ceil(n_cal / cal_run_size))

    cal_data, theta_star = pipeline.gen_training_data(
        model, prior, forcing_prior, t, cal_run_size, cal_n_runs,
        steady_idx, dt_nd_min, nd_dim, forcing_idx, rescale_idx,
        dt_exp=dt_exp, t_min_exp=t_min_exp, t_max_exp=t_max_exp,
        t_scale_bounds=t_scale_bounds, proposal=None,
        theta_transform=theta_transform,  # <-- NEW
        fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
        spontaneous_only=spontaneous_only, chi_mode=chi_mode,
        chi_f0=chi_f0, chi_freq_bounds=chi_freq_bounds, chi_k_pad=chi_k_pad,
        chi_k_fixed=chi_k_fixed, n_vars=n_vars,
        dtype=dtype, device=device,
    )

    # theta_star is the LATENT truth (a logit); a non-finite one would make every SBC rank for that
    # row meaningless, and filtering targets only BY data would not catch it. Cannot currently fire
    # (see gen_training_data's note on SigmoidTransform's internal clamp) -- this is the tripwire for
    # a future change to the transform stack, same as train_nn's.
    theta_finite = torch.isfinite(theta_star).all(dim=1)
    valid = (torch.isfinite(cal_data).all(dim=1) & (torch.abs(cal_data) < 1e15).all(dim=1)
             & theta_finite)
    n_bad_theta = int((~theta_finite).sum())
    if n_bad_theta:
        warnings.warn(
            f"gen_cal_data: dropped {n_bad_theta}/{theta_star.shape[0]} calibration rows with "
            f"non-finite latent truths. The box clamp in gen_training_data should have prevented "
            f"this -- treat it as a bug, not as expected attrition.",
            stacklevel=2,
        )
    cal_data = cal_data[valid]
    theta_star_latent = theta_star[valid]

    if theta_transform is not None:
        # Convert latent theta_star to physical for downstream comparison.
        # The transform lives on cfg.hw.device; move cpu tensor there, apply, move back.
        t_device = _transform_device(theta_transform)
        theta_star_phys = theta_transform(theta_star_latent.to(t_device)).cpu()
        return cal_data, theta_star_phys
    else:
        return cal_data, theta_star_latent


