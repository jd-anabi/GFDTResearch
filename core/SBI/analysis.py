import math
import warnings

import torch
from torch.distributions.transforms import Transform

from core.SBI import pipeline
from core.config import CAL_N_SCALES, CAL_RUN_SIZE, CAL_RUN_SIZE_MAX
from core.SBI.reparam import _transform_device

# === POSTERIOR PREDICTIVE CHECK ===
def invalid_breakdown(valid_mask: torch.Tensor, layout: dict | None) -> dict | None:
    """Split the ZERO-VARIANCE statistics by origin: base features vs chi probe slots.

    WHY THIS EXISTS. The PPC's "Invalid stats: 48/114" is alarming and unactionable -- it reads as
    "42% of the conditioning is broken". Decomposed, the 2026-08-25 retrain's 48 is not a defect at
    all but a description of the experiment: 6 of 12 chi probe slots were never filled (the run
    supplied 6 probes into a network padded for 12), and 2 more were masked, leaving **4 live probes
    out of 12 slots**. That is a sentence you can act on -- it says the conditioning is mostly padding,
    which is a question about K and the probe planner, not about the estimator.

    A slot counts as DEAD when every one of its ``chi_elem_w`` elements has zero variance. Padded vs
    masked is not recoverable from variance alone (both are zeroed by the packer), so it is inferred
    from ``chi_n_freqs``: slots at or beyond the number supplied are padding, and any remaining dead
    slot below that was supplied and then masked.

    :param valid_mask: per-statistic boolean, True where the statistic has non-zero variance.
    :param layout: ``{"input_dim", "chi_k_pad", "chi_elem_w", "chi_n_freqs"}``; None disables this.
    :return: the breakdown dict, or None when no layout was supplied.
    """
    if not layout:
        return None
    inv = ~valid_mask
    base_n = int(layout.get("input_dim") or 0)
    out = {"total": int(inv.sum()), "base_width": base_n, "base": int(inv[:base_n].sum())}
    elem_w, k_pad = int(layout.get("chi_elem_w") or 0), int(layout.get("chi_k_pad") or 0)
    if not (elem_w and k_pad):
        return out
    dead = sum(1 for j in range(k_pad)
               if int(inv[base_n + j * elem_w: base_n + (j + 1) * elem_w].sum()) == elem_w)
    supplied = int(layout.get("chi_n_freqs") or 0)
    pad_slots = max(0, k_pad - supplied)
    out.update(chi_width=k_pad * elem_w, chi_elem_w=elem_w, chi_k_pad=k_pad,
               chi_supplied=supplied, chi_dead_slots=dead, chi_pad_slots=pad_slots,
               chi_masked_slots=max(0, dead - pad_slots), chi_live_slots=k_pad - dead)
    return out


def describe_invalid(bd: dict | None) -> str:
    """One human line for a breakdown, for the PPC figure and the run log."""
    if not bd:
        return ""
    if "chi_k_pad" not in bd:
        return f"{bd['total']} zero-variance stats ({bd['base']} in the {bd['base_width']} base features)"
    return (f"chi probes: {bd['chi_live_slots']} live / {bd['chi_k_pad']} slots "
            f"({bd['chi_pad_slots']} never filled, {bd['chi_masked_slots']} masked); "
            f"{bd['base']} of {bd['base_width']} base features flat")


def posterior_predictive_check(s_obs: torch.Tensor, s_simulated: torch.Tensor,
                               layout: dict | None = None) -> dict:
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
    :param layout: optional conditioning layout (see :func:`invalid_breakdown`) so the zero-variance
                   count can be split by origin instead of reported as one alarming fraction.
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
        "num_invalid": (~valid_mask).sum().item(),
        "invalid_breakdown": invalid_breakdown(valid_mask, layout),
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
                 chi_k_fixed: int | None = None, chi_max_cycles: float | None = None,
                 n_vars: int | None = None,
                 nd_idx: dict | None = None, k_b_cell: float | None = None,
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
        chi_k_fixed=chi_k_fixed, chi_max_cycles=chi_max_cycles, n_vars=n_vars,
        nd_idx=nd_idx, k_b_cell=k_b_cell,
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




# === INFORMATIVENESS (PRISM_HANDOFF section 11.4) ================================================
# Every other diagnostic in this file measures CALIBRATION -- SBC, TARP, PPC coverage all ask whether
# the posterior's stated uncertainty is honest. None of them asks whether it is USEFUL, and a
# posterior that simply returns the prior passes all three perfectly. `posterior_08232026` is exactly
# that: SBC flat on all 13, TARP on the diagonal, and PPC coverage 99.1% at a nominal 90% because the
# intervals are far too wide.
#
# The scalar is the expected prior-to-posterior KL:
#
#     I = E_x[ KL( p(theta|x) || p(theta) ) ]   estimated as   mean over (theta*, x) pairs of
#         log q(theta* | x) - log p(theta*)
#
# drawn from the calibration set validate_calibration already simulates, so it costs no simulation.
#
# ⚠ IT IS NOT BOUNDED BELOW BY ZERO, AND A NEGATIVE VALUE IS A REAL READING RATHER THAN A BUG.
# E[log q - log p] >= 0 holds only when q IS the true posterior. A flow that is under-trained, or
# trained on a different distribution than the one being scored, assigns LOWER density to the truth
# than the prior does and the estimate goes negative. Measured -23.1 nats on a 5-epoch 40-row smoke
# train, which is the honest answer for that posterior: it is worse than the prior. So read the sign
# first -- negative means "this run learned nothing usable yet", not "the estimator is broken".
# The estimator is unbiased for the true posterior because (theta*, x) is drawn from the joint --
# the same property SBC relies on. It is in NATS: 0 means the data said nothing, and log(2) = 0.69
# means the posterior is on average half the prior's width in one direction.
#
# ⚠ WHY THE DECOMPOSITION IS SAMPLE-BASED WHILE THE TOTAL IS NOT. A normalising flow has no
# closed-form marginals, so per-parameter and per-direction figures cannot be read off log_prob. They
# are estimated from draws instead, which is noisier and is why the JOINT number is the one to quote.
#
# ⚠ AND THEY ARE AN ENTROPY REDUCTION, NOT A MARGINAL KL. `per_param[j]` is
# `H(prior_j) - E_x[H(posterior_j | x)]`, which measures how much NARROWER the marginal got. It is
# not KL(posterior_j || prior_j): a marginal that shifts without narrowing has positive KL and ZERO
# entropy reduction. That is the right quantity here on purpose -- section 4.6's complaint is width
# (PPC coverage 99.1% at a nominal 90%, intervals 2-3x the observation's envelope) -- but do not
# quote it as a KL, and note it can go negative for a marginal the flow widened.
# The per-direction figures are computed in the flow's own LATENT coordinate, which under
# REPARAM_ROTATE *is* the Fisher eigenbasis -- so direction j here is column j of the V that
# scripts/posterior_identifiability.py decomposes, and the two tables line up row for row.


def _entropy_1d(x: torch.Tensor, m: int = None) -> torch.Tensor:
    """Vasicek spacing estimate of 1-D differential entropy, over the LAST dim. Returns (...,).

    Chosen over a KDE because it has one integer knob rather than a bandwidth, and a bandwidth chosen
    on the prior would flatter the posterior (or the reverse). Both sides here use the same m.
    """
    n = x.shape[-1]
    m = max(1, int(round(n ** 0.5 / 2))) if m is None else m
    if n <= 2 * m:
        # The spacing window is wider than the sample; s[..., 2m:] would be EMPTY and the mean of an
        # empty tensor is NaN, which would poison the whole decomposition silently.
        raise ValueError(f"_entropy_1d needs more than 2*m={2 * m} samples, got {n}")
    s = x.sort(dim=-1).values
    hi = s[..., 2 * m:]
    lo = s[..., :-2 * m]
    return (torch.log((n / (2.0 * m)) * (hi - lo).clamp(min=1e-30))).mean(dim=-1)


def prior_log_prob(prior, theta: torch.Tensor) -> torch.Tensor:
    """``prior.log_prob(theta)``, surviving a MIXED-DEVICE product prior.

    The inferred prior is ``ProductPrior([nd_gmm, rescale])`` and the two halves do not have to live
    on the same device -- the ND GMM is built on ``cfg.hw.device`` while the rescale bijection is a
    CPU transform. ``ProductPrior.log_prob`` aligns devices when it SUMS, but each block still gets
    the slice on whatever device the caller handed in, so one plain call raises on a GPU box. That is
    why the rest of the pipeline's standing rule for this object is "sample-only, never .log_prob"
    (see ``check_observation_in_distribution``).

    So: try it whole, and on a device error evaluate block by block, discovering each block's device
    by SAMPLING it -- the one operation that rule guarantees works.
    """
    try:
        return prior.log_prob(theta)
    except RuntimeError:
        pass
    dists, dims = getattr(prior, "distributions", None), getattr(prior, "dims", None)
    if not dists or not dims:
        raise
    out, idx = None, 0
    for d, dim in zip(dists, dims):
        dev = d.sample((1,)).device
        lp = d.log_prob(theta[..., idx:idx + dim].to(dev)).to(theta.device)
        out = lp if out is None else out + lp
        idx += dim
    return out


def informativeness(posterior, theta_star: torch.Tensor, x_cal: torch.Tensor,
                    prior, *, param_names: list | None = None,
                    n_decompose: int = 256, n_samples: int = 512,
                    chunk: int = 256) -> dict:
    """Expected prior-to-posterior KL for a trained posterior, plus its decomposition.

    :param posterior: the PHYSICAL-space posterior (a TransformedPosterior, or a bare DirectPosterior).
    :param theta_star: (N, P) physical ground truths from the calibration set.
    :param x_cal: (N, W) their conditioning vectors.
    :param prior: the physical inferred prior theta_star was drawn from.
    :param n_decompose: calibration points used for the sample-based decomposition (0 disables it).
    :param n_samples: posterior draws per point in that decomposition.
    :return: ``{"total_nats", "per_param", "per_direction", ...}``.
    """
    device = x_cal.device
    theta_star = theta_star.to(device)
    lp_post = []
    with torch.no_grad():
        for a in range(0, x_cal.shape[0], chunk):
            th = theta_star[a:a + chunk].unsqueeze(0)          # (1, b, P)
            xb = x_cal[a:a + chunk]                            # (b, W)
            lp_post.append(posterior.log_prob_batched(th, x=xb).reshape(-1).cpu())
        lp_post = torch.cat(lp_post)
        lp_prior = prior_log_prob(prior, theta_star).reshape(-1).cpu()

    ok = torch.isfinite(lp_post) & torch.isfinite(lp_prior)
    per_row = (lp_post - lp_prior)[ok]
    out = {
        "total_nats": float(per_row.mean()),
        "sem_nats": float(per_row.std() / max(per_row.numel() ** 0.5, 1.0)),
        "n_used": int(ok.sum()),
        "n_dropped": int((~ok).sum()),
        "per_param": None,
        "per_direction": None,
        # Present on BOTH return paths, so a caller never has to ask which branch produced the dict.
        "param_names": list(param_names) if param_names else None,
        "n_decompose": 0,
    }
    if not n_decompose:
        return out

    # --- the decomposition: marginal entropies, prior against posterior ---------------------------
    with torch.no_grad():
        n_d = min(int(n_decompose), x_cal.shape[0])
        # EVENLY SPACED, not the first n_d. gen_cal_data returns rows in generation order and every
        # row in a batch shares one (t_scale, T) operating point -- so `range(n_d)` would compute the
        # decomposition over a handful of t_scale values rather than across the prior, which is the
        # same stratification trap trap X5 records for SBC. Deterministic, so two runs compare.
        idx = torch.linspace(0, x_cal.shape[0] - 1, n_d).round().long()
        prior_draw = prior.sample((max(n_samples * 8, 4096),)).cpu()
        h_prior = _entropy_1d(prior_draw.T)                              # (P,)
        T = getattr(posterior, "T", None)
        h_prior_dir = _entropy_1d(T.inv(prior_draw.to(device)).T.cpu()) if T is not None else None
        h_post = torch.zeros_like(h_prior)
        h_post_dir = None if h_prior_dir is None else torch.zeros_like(h_prior_dir)
        for i in idx.tolist():
            s = posterior.sample((n_samples,), x=x_cal[i], show_progress_bars=False)
            h_post += _entropy_1d(s.T.cpu())
            if h_post_dir is not None:
                h_post_dir += _entropy_1d(T.inv(s).T.cpu())
        h_post /= n_d
        out["per_param"] = (h_prior - h_post).tolist()
        if h_post_dir is not None:
            out["per_direction"] = (h_prior_dir - h_post_dir / n_d).tolist()
    out["n_decompose"] = n_d
    return out


def describe_informativeness(info: dict) -> str:
    """The report block, for validate_calibration's stdout and the run log."""
    if not info:
        return ""
    lines = [f"Informativeness (expected prior->posterior KL): "
             f"{info['total_nats']:.4f} +/- {info['sem_nats']:.4f} nats "
             f"over {info['n_used']} calibration points"]
    if info.get("n_dropped"):
        lines.append(f"  ({info['n_dropped']} dropped for a non-finite log-prob)")
    names = info.get("param_names")
    pp, pd = info.get("per_param"), info.get("per_direction")
    if pp:
        lines.append(f"  per parameter (nats, marginal, n={info.get('n_decompose')}):")
        for j, v in sorted(enumerate(pp), key=lambda kv: -kv[1]):
            lines.append(f"    {(names[j] if names else f'p{j}'):>10s} {v:+.4f}")
    if pd:
        lines.append("  per Fisher direction (best-constrained first, same order as V's columns):")
        lines.append("    " + "  ".join(f"{j}:{v:+.3f}" for j, v in enumerate(pd)))
    return "\n".join(lines)
