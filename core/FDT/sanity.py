"""
Sanity checks for the FDT analysis pipeline.

Each check runs a mini measurement and asserts a numerical property that would
catch a common bug class. Run them BEFORE the production sweep.
"""
import math
import warnings

import numpy as np
import torch

from core.config import FDTConfig
from core.FDT.campaigns import (
    run_campaign1_psd, run_campaign2_chi,
    _make_simulator, _n_force_channels, _pick_n_segs, observable_noise_prefactor,
)
from core.FDT.spectral import (
    psd_welch, lock_in_chi, eff_temp_ratio, gen_freqs_log, find_spectral_peak,
)
from core.FDT.plots import plot_eff_temp_ratio


def _interp_log(x_new: torch.Tensor, x_old: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
    """1-D linear INTERPOLATION in log-x, NaN outside the grid. Aligns PSD onto the chi grid.

    The clamp below bounds the INDEX, not the VALUE. Left as it was, a frequency outside the Welch
    grid's span got ``frac < 0`` or ``frac > 1`` -- a linear EXTRAPOLATION off the two edge bins,
    returned silently as if it were data. ``eff_temp_ratio`` then divides by it, so widening
    ``freq_bounds`` past the PSD resolution grew a smooth, plausible-looking, entirely fabricated
    T_eff/T tail. Out-of-range now yields NaN and the callers report how many points that cost.
    """
    x_new = x_new.to(torch.float64)
    x_old = x_old.to(torch.float64).clamp(min=1e-30)
    y_old = y_old.to(torch.float64)
    log_x_old = torch.log(x_old)
    log_x_new = torch.log(x_new.clamp(min=1e-30))
    idx = torch.searchsorted(log_x_old, log_x_new).clamp(1, len(log_x_old) - 1)
    x0, x1 = log_x_old[idx - 1], log_x_old[idx]
    y0, y1 = y_old[idx - 1], y_old[idx]
    frac = (log_x_new - x0) / (x1 - x0)
    out = y0 + frac * (y1 - y0)
    in_range = (log_x_new >= log_x_old[0]) & (log_x_new <= log_x_old[-1])
    return torch.where(in_range, out, torch.full_like(out, float("nan")))


def check_passive_baseline(cfg: FDTConfig, save_plot_path=None) -> tuple[bool, dict]:
    """
    Pipeline with temp=1.0, s=0.0 (tau_c left at its cell value): true
    thermal-equilibrium baseline. Expect T_eff/T ~ 1 across the grid.

    Setting s=0 makes the motor force f_c = f_max*(1 - s*c) -> f_max (constant), so
    _y_dot no longer depends on c. Calcium then decouples entirely: it becomes a
    pure slave variable driven by (x, y) with zero back-action, so its noise (tau_c)
    cannot reach the measured x. tau_c therefore does NOT need to be zeroed. temp=1
    makes the motor noise thermal, leaving the (x, y) subsystem in genuine
    equilibrium with a Boltzmann potential -> FDT holds.

    This is the convention-correctness gate -- PSD/lock-in/noise prefactor bugs all
    show up as a uniform deviation from 1.0. Optionally saves a passive FDT-ratio
    plot (the Martin et al. 2001 Fig 3C control-bundle analogue: a flat line at 1).

    Pass criterion: median(|ratio - 1|) < 0.2 AND max(|ratio - 1|) < 0.4.

    :param save_plot_path: if provided, save the passive FDT-ratio plot there.
    """
    M = min(128, cfg.ensemble_M)
    passive = cfg.with_overrides(temp=1.0, s=0.0, ensemble_M=M,
                                  psd_T_obs_nd=min(4000.0, cfg.psd_T_obs_nd))

    # The passive cell's resonance is near the linearized estimate (s=0 removes the
    # Hopf shift), so derive its characteristic frequency from the passive PSD,
    # searching near cfg.omega_0 to avoid latching onto the low-freq noise floor.
    freqs_psd, G = run_campaign1_psd(passive)
    passive_omega_0 = find_spectral_peak(
        freqs_psd, G, search_band=(cfg.omega_0 * 0.2, cfg.omega_0 * 5.0))

    n_check = 20
    omegas = gen_freqs_log(passive_omega_0, n_check, cfg.freq_bounds,
                            cfg.hw.device, cfg.hw.dtype)
    chis = run_campaign2_chi(passive, omegas, show_progress=False)

    G_at_om = _interp_log(omegas, freqs_psd, G)
    prefactor = observable_noise_prefactor(passive)
    ratio = eff_temp_ratio(G_at_om, chis.imag, omegas.to(torch.float64), prefactor).cpu().numpy()

    # _interp_log returns NaN off the PSD grid rather than extrapolating (it used to fabricate a
    # plausible tail). Judge on the frequencies that are actually covered, and SAY how many were not
    # -- a check that quietly evaluated fewer points than it printed would be its own version of the
    # bug this replaces.
    devs = np.abs(ratio - 1.0)
    n_off = int(np.isnan(devs).sum())
    if n_off:
        warnings.warn(
            f"check_passive_baseline: {n_off}/{devs.size} probe frequencies fall outside the Welch "
            f"PSD grid ({float(freqs_psd.min()):g}..{float(freqs_psd.max()):g}) and were EXCLUDED. "
            f"Narrow cfg.freq_bounds, or lengthen the passive run so the PSD resolves them.",
            stacklevel=2)
    covered = devs[~np.isnan(devs)]
    if covered.size == 0:
        raise ValueError(
            "check_passive_baseline: every probe frequency lies outside the PSD grid, so the FDT "
            "ratio is unmeasurable here. Narrow cfg.freq_bounds or lengthen the passive run.")
    med_dev, max_dev = float(np.median(covered)), float(np.max(covered))
    passed = (med_dev < 0.2) and (max_dev < 0.4)

    if save_plot_path is not None:
        plot_eff_temp_ratio(
            omegas.cpu().numpy(), ratio,
            save_path=save_plot_path,
            title=fr"FDT ratio: PASSIVE baseline ($s=0$, $T_a=T$) -- ND {cfg.model}",
            omega_natural=passive_omega_0,
        )
        print(f"Saved passive FDT-ratio plot to: {save_plot_path}")

    return passed, {"median_dev": med_dev, "max_dev": max_dev,
                    "passive_omega_0": passive_omega_0,
                    "ratios": ratio.tolist(), "omegas": omegas.cpu().tolist()}


def check_high_freq_fdt(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    Natural cell at ambient motor temperature (temp=1) with the active feedback and
    channel noise left at their cell values (s, tau_c unchanged), evaluated only at
    the top three frequencies of the production grid.

    At omega >> 1/tau (tau is the calcium RELAXATION time in _c_dot = (P_t - c)/tau,
    NOT the noise prefactor tau_c), calcium can't follow the bundle. Both the
    deterministic feedback (s*c motor modulation) and the channel-noise injection then
    become quasi-static and stop doing frequency-dependent work, so FDT recovers. This
    validates that the calcium adaptation timescale is the locus of FDT violation in
    the Nadrowski model (as opposed to some other non-equilibrium source).

    temp=1 is required: with temp>1 the motor is a second, hotter thermostat, making
    (x, y) a two-temperature non-equilibrium steady state whose T_eff/T would NOT
    recover to 1 even at high omega. Channel noise (tau_c) is left at its cell value
    so this tests the real cell's high-omega behavior, not an isolated mechanism.

    Pass criterion: median(|ratio - 1|) < 0.15 AND max(|ratio - 1|) < 0.3.
    """
    M = min(128, cfg.ensemble_M)
    high_freq_cfg = cfg.with_overrides(temp=1.0, ensemble_M=M,
                                        psd_T_obs_nd=min(4000.0, cfg.psd_T_obs_nd))

    all_omegas = gen_freqs_log(cfg.omega_0, 7, cfg.freq_bounds,
                                cfg.hw.device, cfg.hw.dtype)
    omegas = all_omegas[-3:]   # top 3 freqs of the 7-point check grid

    freqs_psd, G = run_campaign1_psd(high_freq_cfg)
    chis = run_campaign2_chi(high_freq_cfg, omegas, show_progress=False)

    G_at_om = _interp_log(omegas, freqs_psd, G)
    prefactor = observable_noise_prefactor(high_freq_cfg)
    ratio = eff_temp_ratio(G_at_om, chis.imag, omegas.to(torch.float64), prefactor).cpu().numpy()

    # This check reads all_omegas[-3:] -- the TOP of the probe grid, i.e. precisely where the PSD is
    # most likely to run out. It was therefore the most exposed to _interp_log's old silent
    # extrapolation: it could pass or fail on fabricated numbers. Excluded points are now reported.
    devs = np.abs(ratio - 1.0)
    n_off = int(np.isnan(devs).sum())
    if n_off:
        warnings.warn(
            f"check_high_freq_fdt: {n_off}/{devs.size} of the top probe frequencies fall outside the "
            f"Welch PSD grid ({float(freqs_psd.min()):g}..{float(freqs_psd.max()):g}) and were "
            f"EXCLUDED. This check previously extrapolated there, so a past pass at these "
            f"frequencies was not evidence. Lower cfg.freq_bounds' upper edge or raise psd_T_obs_nd.",
            stacklevel=2)
    covered = devs[~np.isnan(devs)]
    if covered.size == 0:
        return False, {"median_dev": float("nan"), "max_dev": float("nan"), "n_off_grid": n_off,
                       "error": "all high-frequency probes lie outside the PSD grid"}
    med_dev, max_dev = float(np.median(covered)), float(np.max(covered))
    passed = (med_dev < 0.15) and (max_dev < 0.3)
    return passed, {"median_dev": med_dev, "max_dev": max_dev, "n_off_grid": n_off,
                    "ratios": ratio.tolist(), "omegas": omegas.cpu().tolist()}


def check_linearity(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    Campaign-2 at omega_0 with F0, F0/2, F0/4: |chi| must be amplitude-invariant within 5%.
    Failure means the drive is outside the linear regime; lower F0.
    """
    omega = torch.tensor([cfg.omega_0], dtype=cfg.hw.dtype, device=cfg.hw.device)
    F0s = [cfg.F0, cfg.F0 / 2.0, cfg.F0 / 4.0]
    mags = []
    for F0 in F0s:
        chi = run_campaign2_chi(cfg, omega, F0=F0, show_progress=False)[0]
        mags.append(float(chi.abs()))
    mags_t = torch.tensor(mags)
    rel_spread = float(mags_t.std(unbiased=False) / mags_t.mean())
    passed = rel_spread < 0.05
    return passed, {"F0_values": F0s, "|chi|_values": mags, "rel_spread": rel_spread}


def check_ensemble_convergence(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    At omega_0, recompute chi'' from the first {32, 64, 128, 256} trajectories
    of a single M=256 simulation. Pass if |chi''_256 - chi''_128|/|chi''_256| < 0.1.

    Runs the simulator once; slices the (M, n_steady) trajectory tensor by prefix.
    """
    omega = cfg.omega_0
    M_max = 256
    dt, burn = cfg.dt_nd, cfg.burn_in_nd
    device, dtype = cfg.hw.device, cfg.hw.dtype

    burn_idx = int(round(burn / dt))
    n_obs = int(round(cfg.T_obs_periods * 2.0 * math.pi / omega / dt))
    n_steps = burn_idx + n_obs
    t = torch.arange(n_steps, dtype=dtype, device=device) * dt
    n_force = _n_force_channels(cfg)
    force = torch.zeros((M_max, n_force, n_steps), dtype=dtype, device=device)
    force[:, 0, :] = cfg.F0 * torch.cos(omega * t)

    inits = cfg.inits_for_M(M_max)
    params = cfg.params_for_M(M_max)
    n_segs = _pick_n_segs(n_steps, M_max)
    sim = _make_simulator(cfg, params, force, inits, t,
                          freqs_per_batch=1, segs=n_segs, batch_size=M_max, device=device)
    sol = sim.simulate(state_dep_drift=cfg.state_dep_drift)
    x_steady = sol[0, 0, :, burn_idx:]    # (M_max, n_obs)
    t_steady = t[burn_idx:]
    T_obs_used = n_obs * dt

    chi_imags = {}
    for M in (32, 64, 128, 256):
        x_mean = x_steady[:M].mean(dim=0)
        chi = lock_in_chi(t_steady, x_mean, omega, cfg.F0, T_obs_used)
        chi_imags[M] = float(chi.imag)
    rel_change = abs(chi_imags[256] - chi_imags[128]) / max(abs(chi_imags[256]), 1e-30)
    passed = rel_change < 0.1
    return passed, {"chi_imag_by_M": chi_imags, "rel_change_128_256": rel_change}


def check_psd_window(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    Compute Campaign-1; split steady-state into halves; PSD each; compare at 5 freqs.
    Pass if max relative diff < 0.25.
    """
    M = min(128, cfg.ensemble_M)
    half_cfg = cfg.with_overrides(ensemble_M=M, psd_T_obs_nd=min(4000.0, cfg.psd_T_obs_nd))
    dt, burn = half_cfg.dt_nd, half_cfg.burn_in_nd
    device, dtype = half_cfg.hw.device, half_cfg.hw.dtype

    burn_idx = int(round(burn / dt))
    n_obs = int(round(half_cfg.psd_T_obs_nd / dt))
    n_steps = burn_idx + n_obs
    t = torch.arange(n_steps, dtype=dtype, device=device) * dt
    n_force = _n_force_channels(half_cfg)
    force = torch.zeros((M, n_force, n_steps), dtype=dtype, device=device)
    inits = half_cfg.inits_for_M(M)
    params = half_cfg.params_for_M(M)

    n_segs = _pick_n_segs(n_steps, M)
    sim = _make_simulator(half_cfg, params, force, inits, t,
                          freqs_per_batch=1, segs=n_segs, batch_size=M, device=device)
    sol = sim.simulate(state_dep_drift=half_cfg.state_dep_drift)
    x_steady = sol[0, 0, :, burn_idx:]
    n_half = n_obs // 2
    x_first, x_second = x_steady[:, :n_half], x_steady[:, n_half:2 * n_half]

    nperseg = min(2 ** 13, n_half)
    nperseg = 1 << int(math.log2(nperseg))

    omegas1, G1 = psd_welch(x_first, dt=dt, nperseg=nperseg)
    _, G2 = psd_welch(x_second, dt=dt, nperseg=nperseg)

    n_freqs = G1.shape[-1]
    sample_idx = torch.linspace(1, n_freqs - 1, 5).long()
    rel_diff = ((G1[sample_idx] - G2[sample_idx]).abs() / (G1[sample_idx].abs() + 1e-30)).cpu().numpy()
    max_rel = float(rel_diff.max())
    passed = max_rel < 0.25
    return passed, {"max_rel_diff": max_rel, "rel_diffs": rel_diff.tolist(),
                    "omegas_sampled": omegas1[sample_idx].cpu().tolist()}


def run_all_sanity(cfg: FDTConfig, passive_plot_path=None) -> dict:
    """
    Run all five checks; print summary; return dict[name -> (passed, metrics)].

    :param passive_plot_path: if provided, the passive-baseline check saves a
                              passive FDT-ratio plot (Martin Fig 3C analogue) there.
    """
    print("\n" + "=" * 60)
    print("FDT Sanity Checks")
    print("=" * 60)

    checks = [
        ("passive_baseline",      lambda c: check_passive_baseline(c, save_plot_path=passive_plot_path),
                                                               "true equilibrium (s=0): T_eff/T ~ 1"),
        ("high_freq_fdt",         check_high_freq_fdt,         "high omega (s != 0): T_eff/T ~ 1"),
        ("linearity",             check_linearity,             "|chi(omega_0)| F0-invariant"),
        ("ensemble_convergence",  check_ensemble_convergence,  "chi'' stable by M=256"),
        ("psd_window",            check_psd_window,            "PSD halves agree"),
    ]
    # passive_baseline / high_freq_fdt reason about Nadrowski-specific physics (the s-feedback and the
    # motor thermostat via with_overrides(s=, temp=)), which only exist as params_dict keys for
    # Nadrowski. For any other model run only the model-agnostic checks (linearity / convergence / PSD).
    _NADROWSKI_ONLY = {"passive_baseline", "high_freq_fdt"}
    if cfg.model.lower() != "nadrowski":
        checks = [c for c in checks if c[0] not in _NADROWSKI_ONLY]
        print(f"Note: the passive-baseline / high-frequency FDT checks are Nadrowski-specific and are "
              f"skipped for {cfg.model}; running the model-agnostic checks only.")
    results = {}
    for name, fn, desc in checks:
        print(f"\n[{name}] {desc}")
        passed, metrics = fn(cfg)
        results[name] = (passed, metrics)
        print(f"  {'PASS' if passed else 'FAIL'}  metrics: {metrics}")

    print("\n" + "=" * 60 + "\nSummary:")
    for name, (passed, _) in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("=" * 60 + "\n")
    return results
