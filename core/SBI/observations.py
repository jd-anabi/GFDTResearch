"""Building conditioning observations from REAL bench recordings, split out of orchestrator.

The three builders mirror generate_observations' branches for data that was measured rather than
simulated: forced (passive + one driven recording), spontaneous (one passive recording), and chi
(passive + any number of single-tone recordings at the frequencies they were actually driven at).
Re-exported by orchestrator under the same names for the GUI runners and the scripts.
"""
import math
import warnings

import torch

from core import config
from core.config import SimConfig
from core.SBI import chi, pipeline, statistics


# ── Step 5: Inference on real experimental data ────────────────────────────
def build_experiment_obs(
    cfg: SimConfig,
    X_obs_spont: torch.Tensor,
    X_obs_forced: torch.Tensor,
    T_obs_s: float,
    forcing_params_si: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the conditioning vector + observed trajectory from a real experimental recording, and set the
    observation context on cfg (T_obs + forcing values, in cell-file units) so infer_and_visualize's
    PPC / eye-test can simulate. Posterior sampling + the corner plot are done by infer_and_visualize.

    The recording must be sampled at 1/cfg.dt_exp (the camera frame rate the network was trained on).
    T_obs and the forcing params are given in SI units and converted to cell-file units here.

    :param X_obs_spont: 1D spontaneous (unforced) recording, shape (N_obs,), at 1/cfg.dt_exp.
    :param X_obs_forced: 1D forced (driven) recording, shape (N_obs,), at 1/cfg.dt_exp.
    :param T_obs_s: Observation duration in SECONDS.
    :param forcing_params_si: Dict with keys "amp" (N), "freq" (Hz), "phase" (rad), "offset" (N).
    :return: (obs_stats, obs_data, t_dim): the [S | log(T) | forcing] conditioning vector (1, D), the
             forced recording (1, N_obs) for the eye-test, and the dimensional time axis (1, N_obs).
    """
    dtype = cfg.hw.dtype

    # Unit conversions: SI -> cell file units.
    # Known forcing param SI units; fall back to no conversion (dimensionless) if unknown.
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell

    # Consistency check: X_obs must be sampled at 1/dt_exp with duration T_obs.
    expected_N = int(T_obs / cfg.dt_exp)
    if X_obs_spont.shape[-1] != X_obs_forced.shape[-1]:
        raise ValueError(
            f"Spontaneous and forced recordings must be the same length "
            f"({X_obs_spont.shape[-1]} vs {X_obs_forced.shape[-1]})."
        )
    if abs(X_obs_forced.shape[-1] - expected_N) > 1:
        warnings.warn(
            f"Recording length ({X_obs_forced.shape[-1]}) doesn't match expected from T_obs_s={T_obs_s:.4f}s "
            f"at 1/dt_exp sampling (expected ~{expected_N} points). "
            f"Check that both recordings are sampled at dt_exp={cfg.dt_exp:.6f} (cell units).",
            stacklevel=2,
        )

    # Out-of-distribution warning: compute the minimum feasible t_scale for this T_obs.
    # The NN was only trained on batches where n_fine_total <= N_ND_MAX, i.e.
    #   steady_idx + (T_k / dt_exp) * (t_scale_hi / t_scale_k) <= N_ND_MAX
    # Solving for t_scale_k given T_k = T_obs:
    t_scale_lo_prior, t_scale_hi = cfg.t_scale_bounds
    budget = N_ND_MAX - cfg.steady_idx
    if budget > 0:
        t_scale_min_feasible = (T_obs / cfg.dt_exp) * t_scale_hi / budget
        if t_scale_min_feasible > t_scale_lo_prior:
            warnings.warn(
                f"Inference out-of-distribution risk: for T_obs={T_obs_s:.2f}s, the NN was "
                f"only trained on t_scale >= {t_scale_min_feasible:.3f} (in cell file units). "
                f"If the true t_scale is below this, the posterior may extrapolate poorly.",
                stacklevel=2,
            )

    # Build forcing tensor generically: iterate cfg.force_params_dict and apply the appropriate SI->cell
    # conversion per parameter name (config.FORCING_SI_UNITS is the single source of truth; the CLI/GUI
    # display hints derive from it). Unknown names raise. "Hz" is SPECIAL-CASED below -- see that map.
    _FORCING_SI_UNITS = FORCING_SI_UNITS
    forcing_t = torch.empty((1, len(cfg.force_params_dict)), dtype=dtype)
    for name in cfg.force_params_dict.keys():
        if name not in forcing_params_si:
            raise KeyError(f"forcing_params_si missing required key '{name}' "
                           f"(cell file expects: {list(cfg.force_params_dict.keys())})")
        if name not in _FORCING_SI_UNITS:
            raise ValueError(f"Unknown forcing parameter '{name}'. Known: {list(_FORCING_SI_UNITS)}. "
                             f"Add an entry to _FORCING_SI_UNITS in build_experiment_obs.")
        si_unit = _FORCING_SI_UNITS[name]
        si_val = forcing_params_si[name]
        if si_unit is None:                    # dimensionless (phase, in radians)
            cell_val = si_val
        elif si_unit == "Hz":                  # inverse cell time -- derived from the TIME unit
            cell_val = si_val * cfg.freq_si_to_cell
        else:
            cell_val = si_val * cfg.get_unit_conversion_factor(si_unit)
        forcing_t[0, cfg.forcing_idx[name]] = cell_val

    # Reshape both recordings to (1, N_obs) and compute summary statistics with dt_exp
    X_spont_batched = X_obs_spont.to(dtype=dtype).unsqueeze(0)
    X_forced_batched = X_obs_forced.to(dtype=dtype).unsqueeze(0)
    obs_stats = pipeline.gen_stats(
        X_spont_batched, X_forced_batched, cfg.dt_exp,
        forcing_t[:, cfg.forcing_idx["amp"]], forcing_t[:, cfg.forcing_idx["freq"]],
        forcing_t[:, cfg.forcing_idx["phase"]], device=cfg.hw.device,
    )

    # Build conditioning vector: [S(X_obs), log(T_obs), forcing]
    obs_stats = statistics.conditioning_rows(obs_stats, T_obs, forcing_t)

    # Record the observation context so infer_and_visualize's PPC / eye-test can simulate.
    forcing_vals = {name: float(forcing_t[0, cfg.forcing_idx[name]]) for name in cfg.force_params_dict}
    cfg.set_observation_context(T_obs, forcing_vals)

    # Dimensional time axis (in SECONDS) + observed (forced) trajectory for the eye-test.
    N_obs = X_forced_batched.shape[-1]
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")   # cell time unit -> seconds
    t_dim = ((torch.arange(N_obs, dtype=dtype) * cfg.dt_exp) * s_per_cell).unsqueeze(0)
    return obs_stats, X_forced_batched, t_dim


def build_experiment_obs_spontaneous(
    cfg: SimConfig, X_obs: torch.Tensor, T_obs_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Passive-recording variant of build_experiment_obs for a NO-FORCING model: a SINGLE unforced
    recording, no forced file, and no forcing/frequency SI units (the drive machinery is not entered).
    Conditioning is [S(A-F, Group G zeroed) | log(T_obs)], matching generate_observations' no-forcing path.

    :param X_obs: 1D passive recording, shape (N_obs,), sampled at 1/cfg.dt_exp.
    :param T_obs_s: observation duration in SECONDS.
    :return: (obs_stats, obs_data, t_dim): the [S | log(T)] conditioning vector, the recording (1, N_obs),
             and the dimensional (seconds) time axis (1, N_obs).
    """
    dtype = cfg.hw.dtype
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell

    expected_N = int(T_obs / cfg.dt_exp)
    if abs(X_obs.shape[-1] - expected_N) > 1:
        warnings.warn(
            f"Recording length ({X_obs.shape[-1]}) doesn't match expected from T_obs_s={T_obs_s:.4f}s "
            f"at 1/dt_exp sampling (expected ~{expected_N} points).", stacklevel=2)

    X_batched = X_obs.to(dtype=dtype).unsqueeze(0)
    obs_stats = pipeline.gen_stats(X_batched, None, cfg.dt_exp, None, None, None,
                                   device=cfg.hw.device, spontaneous_only=True)
    obs_stats = statistics.conditioning_rows(obs_stats, T_obs)

    cfg.set_observation_context(T_obs, {})
    N_obs = X_batched.shape[-1]
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")
    t_dim = ((torch.arange(N_obs, dtype=dtype) * cfg.dt_exp) * s_per_cell).unsqueeze(0)
    return obs_stats, X_batched, t_dim


def build_experiment_obs_chi(
    cfg: SimConfig, X_spont: torch.Tensor, X_forced_list: list[torch.Tensor],
    T_obs_s: float, F0_si: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    chi(omega) experimental path: ONE passive recording + ANY NUMBER of single-tone FORCED recordings,
    each locked in at THE FREQUENCY IT WAS ACTUALLY DRIVEN AT. Builds the conditioning
    [S(A-F) | log(T) | padded probe set], mirroring generate_observations' chi branch.

    This used to compute ``freq_k = mult_k * peak_freq(X_spont)`` itself and never asked what you
    drove at. Two ways that went wrong on the bench, both silent: the frequencies you can actually
    achieve are not exactly ``mult_k * Omega_0``, and even aiming for them, your Omega_0 estimate is
    not ``chi.peak_freq``'s (different trace length, windowing, bin resolution). A lock-in at the
    wrong frequency decays like a sinc -- a mismatch of a fraction of 1/T_obs destroys the estimate.
    It also demanded exactly K recordings, with no substitution if one failed.

    chi = response/drive is drive-amplitude-independent in the linear regime, so any linear physical
    drive the experiment used works -- it is reported (F0_si) only so the lock-in divides by it to yield
    the true physical susceptibility (x_scale/f_scale)*chi_nd, matching training.

    :param X_spont: 1D passive recording (N_obs,), sampled at 1/cfg.dt_exp.
    :param X_forced_list: the forced recordings. Either 1-D tensors -- legacy, assumed driven at
        ``chi.chi_multipliers_for(cfg)`` -- or ``(recording, drive_frequency_Hz)`` pairs, which is the
        form to use for real data. Any count from 1 to ``cfg.chi_k_pad``.
    :param T_obs_s: observation duration (seconds).
    :param F0_si: physical drive amplitude used (SI force, N); converted to cell force units.
    :return: (obs_stats, obs_data=X_spont as (1,N), t_dim in seconds).
    """
    dtype = cfg.hw.dtype
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell
    F0 = F0_si * cfg.get_unit_conversion_factor("N")           # SI force -> cell force units

    X_spont_b = X_spont.to(dtype=dtype).unsqueeze(0)          # (1, N)
    N = X_spont_b.shape[-1]
    expected_N = int(T_obs / cfg.dt_exp)
    if abs(N - expected_N) > 1:
        warnings.warn(
            f"Passive recording length ({N}) doesn't match T_obs_s={T_obs_s:.4f}s at 1/dt_exp "
            f"(expected ~{expected_N} points).", stacklevel=2)

    f_peak = chi.peak_freq(X_spont_b, cfg.dt_exp)             # (1,) Omega_0/2pi (cell freq units)
    n_probes = len(X_forced_list)
    if not (1 <= n_probes <= cfg.chi_k_pad):
        raise ValueError(
            f"chi-mode accepts 1 to {cfg.chi_k_pad} forced recordings (CHI_K_PAD), got {n_probes}.")

    # Legacy positional form: no frequency supplied, so fall back to the nominal grid.
    paired = bool(X_forced_list) and isinstance(X_forced_list[0], (tuple, list))
    if not paired:
        mults = chi.chi_multipliers(dtype=dtype, device=torch.device("cpu"),
                                    n_freqs=n_probes, bounds=cfg.chi_freq_bounds)

    chis, u_list, logcyc_list, valid = [], [], [], []
    for k, item in enumerate(X_forced_list):
        if paired:
            Xf, freq_hz = item[0], float(item[1])
        else:
            Xf = item
            # The legacy grid, expressed in Hz so ONE predicate path serves both forms.
            freq_hz = float(mults[k] * f_peak) / cfg.freq_si_to_cell
        Xf_b = Xf.to(dtype=dtype).unsqueeze(0)               # (1, N_k)
        N_k = Xf_b.shape[-1]

        # EVERY predicate lives in chi.probe_verdict, which the GUI's probe planner also calls. They
        # used to be written out here and nowhere else, so "what will be refused" could only be
        # discovered by running the thing -- after a bench session rather than before it.
        # Refusal is still raised HERE: a planner has to be able to describe a bad probe without
        # dying on it, so the shared function returns a verdict and the runtime decides it is fatal.
        v = chi.probe_verdict(cfg, float(f_peak), freq_hz, N_k)
        if v.action == "refuse":
            raise ValueError(f"chi probe {k}: {v.reason}.")
        if v.action == "truncate":
            # TRUNCATE rather than mask -- the recording is fine, only its tail is unusable, and the
            # leading prefix is exactly what training measured. Warned, not silent: the user recorded
            # that length on purpose and is entitled to know only part of it was used. Above the
            # ceiling |chi| stops being reproducible at fixed parameters (trap CHI9) and logcyc would
            # report a cycle count no training row ever carried.
            warnings.warn(f"chi probe {k}: {v.reason}.", stacklevel=2)
        elif v.action == "mask":
            # UNDER-RESOLVED IS MASKED, NOT REFUSED, and the distinction is train/eval consistency.
            # Training masks a sub-cycle probe and keeps the row, so the network has learned to
            # condition on sets with absent probes; refusing here would reject an observation it
            # handles perfectly well, and at the band's low edge that is common.
            warnings.warn(f"chi probe {k}: {v.reason}.", stacklevel=2)

        f_cell = torch.tensor([freq_hz * cfg.freq_si_to_cell], dtype=dtype)
        f_val = float(f_cell)
        Xf_b, N_k = Xf_b[:, :v.n_use], v.n_use
        T_k = N_k * cfg.dt_exp
        resolved = v.action != "mask"
        chis.append(chi.lock_in_batched(Xf_b, 2.0 * math.pi * f_cell, F0, T_k, cfg.dt_exp))
        u_list.append(torch.tensor([math.log(f_val / float(f_peak))], dtype=dtype))
        logcyc_list.append(torch.tensor([math.log(f_val * T_k)], dtype=dtype))
        valid.append(resolved)

    chi_block, chi_mask = chi.pack_probe_block(
        torch.stack(chis, dim=1), torch.stack(u_list, dim=1), torch.stack(logcyc_list, dim=1),
        torch.tensor([valid], dtype=torch.bool), k_pad=cfg.chi_k_pad, bounds=cfg.chi_freq_bounds)
    if not bool(chi_mask.any()):
        # Every probe masked. The encoder handles an empty set (it returns its empty-set constant),
        # but then the posterior is conditioned on the passive trace alone -- so this would silently
        # answer a spontaneous question with a chi posterior. The user supplied recordings expecting
        # them to be used; say that none were.
        raise ValueError(
            f"chi: none of the {n_probes} supplied recordings produced a usable probe (all were "
            f"below the {config.CHI_MIN_CYCLES:g}-cycle floor or had a non-finite lock-in). The "
            f"conditioning would carry no susceptibility information at all.")

    obs_stats = pipeline.gen_stats(X_spont_b, None, cfg.dt_exp, None, None, None,
                                   device=cfg.hw.device, spontaneous_only=True)
    obs_stats = statistics.conditioning_rows(obs_stats, T_obs, chi_block.cpu())

    cfg.set_observation_context(T_obs, {})
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")
    t_dim = ((torch.arange(N, dtype=dtype) * cfg.dt_exp) * s_per_cell).unsqueeze(0)
    return obs_stats, X_spont_b, t_dim
