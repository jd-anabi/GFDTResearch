"""
FDT parameter-sweep study.

Sweeps a single NWK parameter (others held fixed) and runs the FDT campaigns at
each value to map how the effective-temperature ratio T_eff/T responds. Two
canonical sweeps probe FDT restoration:

  - S sweep (T_a/T = 1 fixed): the calcium feedback is the only non-equilibrium
    source. As S -> 0 calcium decouples and FDT is restored (T_eff/T -> 1).
  - T sweep (S = 0 fixed): calcium is decoupled, so the hot motor (T_a > T) is the
    only non-equilibrium source. As T_a/T -> 1, FDT is restored.

Pure FDT measurement -- no reduction map. Output: an HDF5 file per sweep, one
group per swept value. Downstream plotting renders T_eff/T vs (omega/omega_0,
param). The omega/omega_0 grid is identical across operating points by
construction (Campaign 2's grid is omega_0 x fixed log-ratios), so rows stack
with no interpolation.
"""
from __future__ import annotations
import warnings
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from ..config import FDTConfig
from .campaigns import run_campaign1_psd, run_campaign2_chi, observable_noise_prefactor
from .spectral import gen_freqs_log, eff_temp_ratio
from .sanity import _interp_log
from .fdt_pipeline import _estimate_omega_0


_OUT_DIR = Path("Resources/CrossValidation")


def _detect_resonance(omegas, G, omega_0_lin: float) -> tuple[float, bool]:
    """
    Locate the spontaneous-oscillation frequency robustly.

    Picks the highest interior local maximum of the PSD (via scipy.signal.find_peaks
    with a prominence threshold to ignore noise wiggles), skipping the DC bin. If the
    spectrum is monotonic (near-equilibrium / overdamped -- no resonance), falls back
    to the linearized estimate. This avoids the floor-latching that plain argmax
    suffers at low S / T_a/T=1, where the lowest PSD bin dominates.

    :param omegas: (n,) angular frequencies (torch or numpy).
    :param G: (n,) one-sided PSD (torch or numpy).
    :param omega_0_lin: linearized natural-frequency estimate, used as fallback.
    :return: (omega_0, is_resonant). is_resonant is False when the fallback was used.
    """
    from scipy.signal import find_peaks  # lazy import: avoids OpenMP load-order conflicts

    om = omegas.detach().cpu().numpy() if hasattr(omegas, "detach") else np.asarray(omegas)
    g = G.detach().cpu().numpy() if hasattr(G, "detach") else np.asarray(G)
    om, g = om[1:], g[1:]   # skip DC
    if g.size < 3 or not np.any(np.isfinite(g)):
        return float(omega_0_lin), False

    g_range = float(np.nanmax(g) - np.nanmin(g))
    prominence = max(0.05 * g_range, 1e-30)   # 5% of dynamic range
    peaks, _props = find_peaks(g, prominence=prominence)
    if peaks.size == 0:
        return float(omega_0_lin), False
    best = int(peaks[np.argmax(g[peaks])])
    return float(om[best]), True


def _campaign2_ratio(cfg: FDTConfig, omegas: torch.Tensor,
                     freqs_psd: torch.Tensor, G: torch.Tensor):
    """
    Forced response (Campaign 2) on a SUPPLIED frequency grid, plus T_eff/T using a
    precomputed Campaign-1 PSD. Split out so a sweep can run all operating points on
    one common grid.

    :return: (chis, ratio) -- complex susceptibility and T_eff/T, both on `omegas`.
    """
    chis = run_campaign2_chi(cfg, omegas)
    G_at = _interp_log(omegas, freqs_psd, G)
    prefactor = observable_noise_prefactor(cfg)
    ratio = eff_temp_ratio(G_at, chis.imag, omegas.to(torch.float64), prefactor)
    return chis, ratio


def _fdt_measure(cfg: FDTConfig) -> dict:
    """
    Single-operating-point FDT: Campaign 1 -> robust omega_0 -> Campaign 2 on a grid
    centered on that omega_0 -> ratio. Used for one-off measurements; the parameter
    sweep uses the two-phase path in run_fdt_param_sweep (shared grid across rows).

    :returns: dict with omega_grid, omega_norm, T_eff_over_T, chi_prime,
              chi_double_prime, PSD_omegas, PSD_G, omega_0_empirical, omega_0_linearized.
    """
    omega_0_lin, _ = _estimate_omega_0(cfg)
    freqs_psd, G = run_campaign1_psd(cfg)
    omega_0_emp, _is_res = _detect_resonance(freqs_psd, G, omega_0_lin)
    cfg.omega_0 = omega_0_emp

    omegas = gen_freqs_log(cfg.omega_0, cfg.n_freqs, cfg.freq_bounds,
                           cfg.hw.device, cfg.hw.dtype)
    chis, ratio = _campaign2_ratio(cfg, omegas, freqs_psd, G)

    omega_np = omegas.cpu().numpy().astype(np.float64)
    return {
        "omega_grid": omega_np,
        "omega_norm": omega_np / omega_0_emp,
        "T_eff_over_T": ratio.cpu().numpy().astype(np.float64),
        "chi_prime": chis.real.cpu().numpy().astype(np.float64),
        "chi_double_prime": chis.imag.cpu().numpy().astype(np.float64),
        "PSD_omegas": freqs_psd.cpu().numpy().astype(np.float64),
        "PSD_G": G.cpu().numpy().astype(np.float64),
        "omega_0_empirical": omega_0_emp,
        "omega_0_linearized": float(omega_0_lin),
    }


def _build_common_grid(cfg: FDTConfig, omega0s: list[float], is_res: list[bool]
                       ) -> tuple[torch.Tensor, float]:
    """
    Build ONE absolute frequency grid covering every operating point's resonance band,
    plus the single reference omega_0 used to normalize the x-axis.

    The grid spans [freq_bounds[0]*min(omega0), freq_bounds[1]*max(omega0)] over the
    *resonant* operating points (so passive fallbacks don't needlessly widen it). If
    no point is resonant (fully near-equilibrium sweep), the linearized estimates are
    used. Point count scales with the log-span to preserve per-decade density.

    :return: (omegas_common, omega_0_ref). omega_0_ref = max resonant omega_0.
    """
    resonant = [w for w, r in zip(omega0s, is_res) if r]
    basis = resonant if resonant else omega0s
    lo_mult, hi_mult = cfg.freq_bounds
    lo = lo_mult * min(basis)
    hi = hi_mult * max(basis)

    ref_decades = math.log10(hi_mult / lo_mult)
    span_decades = math.log10(hi / lo)
    n = max(cfg.n_freqs, int(round(cfg.n_freqs * span_decades / ref_decades)))

    omegas = torch.exp(torch.linspace(math.log(lo), math.log(hi), n,
                                      device=cfg.hw.device, dtype=cfg.hw.dtype))
    omega_0_ref = float(max(basis))
    return omegas, omega_0_ref


def run_fdt_param_sweep(
    cfg: FDTConfig,
    sweep_param: str,
    sweep_grid: np.ndarray,
    fixed_overrides: dict | None = None,
    *,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Sweep one NWK parameter, running FDT at each value, and save to HDF5.

    Two-phase so all rows share ONE frequency grid that covers every operating
    point's resonance:
      Phase A: Campaign 1 (spontaneous PSD) per row -> robust omega_0(value).
      Phase B: Campaign 2 on the common grid per row -> chi, T_eff/T.

    The HDF5 is written incrementally (PSD in Phase A, Campaign-2 data added in
    Phase B), so an interrupt still leaves a partially-populated, readable file.

    :param cfg: baseline FDTConfig.
    :param sweep_param: NWK param name to vary (e.g. "s" or "temp"); a key in params_dict.
    :param sweep_grid: 1D array of values for sweep_param.
    :param fixed_overrides: other params pinned for the whole sweep
                            (e.g. {"temp": 1.0} for the S sweep, {"s": 0.0} for the T sweep).
    :param output_path: target .h5. Defaults to Resources/CrossValidation/sweep_<param>_<stamp>.h5.
    :returns: the HDF5 output path.
    """
    fixed_overrides = fixed_overrides or {}
    if output_path is None:
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _OUT_DIR / f"sweep_{sweep_param}_{stamp}.h5"

    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_overrides.items())

    with h5py.File(output_path, "w") as h5:
        h5.attrs["timestamp"] = datetime.now().isoformat()
        h5.attrs["model"] = cfg.model
        h5.attrs["sweep_param"] = sweep_param
        h5.attrs["fixed_overrides"] = json.dumps(fixed_overrides)
        h5.attrs["n_freqs"] = cfg.n_freqs
        h5.attrs["ensemble_M"] = cfg.ensemble_M
        h5.attrs["F0"] = cfg.F0
        h5.attrs["psd_T_obs_nd"] = cfg.psd_T_obs_nd
        h5.attrs["n_operating_points"] = int(len(sweep_grid))
        h5.attrs["sweep_grid"] = np.asarray(sweep_grid, dtype=np.float64)
        ops = h5.create_group("operating_points")

        # --- Phase A: spontaneous PSD + robust omega_0 per operating point ---
        print(f"\n--- Phase A ({sweep_param} sweep): spontaneous PSD + omega_0 detection ---")
        cfg_ops, psds, omega0s, is_res, omega0_lins = [], [], [], [], []
        for idx, value in enumerate(sweep_grid):
            print(f"  [A {idx+1}/{len(sweep_grid)}] {sweep_param}={value:.6g} ({fixed_str})")
            cfg_op = cfg.with_overrides(**{sweep_param: float(value), **fixed_overrides})
            omega_0_lin, _ = _estimate_omega_0(cfg_op)
            freqs_psd, G = run_campaign1_psd(cfg_op)
            w0, res = _detect_resonance(freqs_psd, G, omega_0_lin)

            cfg_ops.append(cfg_op); psds.append((freqs_psd, G))
            omega0s.append(w0); is_res.append(res); omega0_lins.append(float(omega_0_lin))

            grp = ops.create_group(f"{idx:03d}")
            grp.attrs["param_value"] = float(value)
            grp.attrs["sweep_param"] = sweep_param
            for k, v in fixed_overrides.items():
                grp.attrs[f"fixed_{k}"] = float(v)
            grp.attrs["omega_0_resonance"] = float(w0)
            grp.attrs["omega_0_linearized"] = float(omega_0_lin)
            grp.attrs["is_resonant"] = bool(res)
            grp.attrs["failed"] = True   # flipped to False once Campaign 2 lands in Phase B
            grp.create_dataset("PSD_omegas", data=freqs_psd.cpu().numpy().astype(np.float64),
                               compression="gzip")
            grp.create_dataset("PSD_G", data=G.cpu().numpy().astype(np.float64),
                               compression="gzip")
            tag = "" if res else " [no resonance -> linearized fallback]"
            print(f"      omega_0 = {w0:.4f}{tag}")
            h5.flush()

        # --- Common grid covering every row's resonance band ---
        omegas_common, omega_0_ref = _build_common_grid(cfg, omega0s, is_res)
        omega_grid_np = omegas_common.cpu().numpy().astype(np.float64)
        omega_norm_np = (omega_grid_np / omega_0_ref).astype(np.float64)
        h5.attrs["omega_0_ref"] = omega_0_ref
        h5.attrs["common_grid_n"] = int(omegas_common.shape[0])
        h5.attrs["common_grid_span"] = np.array([omega_grid_np[0], omega_grid_np[-1]],
                                                dtype=np.float64)
        print(f"\nCommon grid: {omegas_common.shape[0]} pts spanning "
              f"[{omega_grid_np[0]:.4f}, {omega_grid_np[-1]:.4f}] (omega_0_ref={omega_0_ref:.4f})")

        # --- Phase B: forced response on the common grid per operating point ---
        print(f"\n--- Phase B ({sweep_param} sweep): forced response on common grid ---")
        n_failed = 0
        for idx, (cfg_op, (freqs_psd, G)) in enumerate(zip(cfg_ops, psds)):
            print(f"  [B {idx+1}/{len(sweep_grid)}] {sweep_param}={sweep_grid[idx]:.6g}")
            grp = ops[f"{idx:03d}"]
            try:
                chis, ratio = _campaign2_ratio(cfg_op, omegas_common, freqs_psd, G)
            except Exception as e:
                # Recorded per point, and COUNTED. A systematic failure (a CUDA OOM, say) fails every
                # point identically, so an overnight sweep could "complete" with nothing in it -- the
                # per-point note scrolled past hours ago and the summary said nothing.
                print(f"      Campaign 2 FAILED: {e}")
                grp.attrs["error"] = str(e)
                n_failed += 1
                continue

            grp.attrs["omega_0_ref"] = omega_0_ref
            grp.create_dataset("omega_grid", data=omega_grid_np, compression="gzip")
            grp.create_dataset("omega_norm", data=omega_norm_np, compression="gzip")
            grp.create_dataset("T_eff_over_T",
                               data=ratio.cpu().numpy().astype(np.float64), compression="gzip")
            grp.create_dataset("chi_prime",
                               data=chis.real.cpu().numpy().astype(np.float64), compression="gzip")
            grp.create_dataset("chi_double_prime",
                               data=chis.imag.cpu().numpy().astype(np.float64), compression="gzip")
            grp.attrs["failed"] = False
            print(f"      T_eff/T peak = {np.nanmax(ratio.cpu().numpy()):.3g}")
            h5.flush()

    n_points = len(sweep_grid)
    if n_failed == n_points:
        raise RuntimeError(
            f"{sweep_param} sweep produced NO usable points: all {n_points} operating points failed "
            f"in Campaign 2 (per-point reasons are in the HDF5 'error' attrs). This is almost always "
            f"one systematic cause -- a CUDA OOM repeating identically at every point -- not "
            f"{n_points} independent failures. {output_path} contains the PSDs but no response data.")
    if n_failed:
        warnings.warn(
            f"{sweep_param} sweep: {n_failed}/{n_points} operating points failed in Campaign 2 and "
            f"carry no response data. See their 'error' attrs in {output_path}.", stacklevel=2)
    print(f"\n{sweep_param} sweep complete ({n_points - n_failed}/{n_points} points). "
          f"Saved to: {output_path}")
    return output_path


def run_param_study_cli(cfg: FDTConfig, s_grid: np.ndarray, temp_grid: np.ndarray) -> tuple[Path, Path]:
    """
    CLI entry: run the S sweep (T_a/T=1), plot it, then run the T sweep (S=0) and
    plot it. Each sweep's 3D plot is saved as soon as that sweep finishes -- so a
    long study gives you the S-sweep plot at its midpoint rather than only at the
    very end. Returns (s_h5_path, temp_h5_path).
    """
    from .cross_validation_plots import plot_fdt_3d_vs_param

    # --- S sweep, then plot immediately ---
    print("\n" + "#" * 64)
    print("# S sweep:  vary S, hold T_a/T = 1   (FDT restored as S -> 0)")
    print("#" * 64)
    s_path = run_fdt_param_sweep(cfg, sweep_param="s", sweep_grid=s_grid,
                                 fixed_overrides={"temp": 1.0})
    print("Plotting S sweep...")
    p1 = plot_fdt_3d_vs_param(load_param_sweep(s_path), param_symbol=r"$S$",
                              title=r"FDT ratio vs $(\tilde\omega/\Omega_0,\ S)$  ($T_a/T=1$)",
                              filename_tag="fdt3d_vs_S")
    if p1:
        print(f"  Saved S-sweep plot: {p1}")

    # --- T sweep, then plot immediately ---
    print("\n" + "#" * 64)
    print("# T sweep:  vary T_a/T, hold S = 0   (FDT restored as T_a/T -> 1)")
    print("#" * 64)
    temp_path = run_fdt_param_sweep(cfg, sweep_param="temp", sweep_grid=temp_grid,
                                    fixed_overrides={"s": 0.0})
    print("Plotting T sweep...")
    p2 = plot_fdt_3d_vs_param(load_param_sweep(temp_path), param_symbol=r"$T_a/T$",
                              title=r"FDT ratio vs $(\tilde\omega/\Omega_0,\ T_a/T)$  ($S=0$)",
                              filename_tag="fdt3d_vs_T")
    if p2:
        print(f"  Saved T-sweep plot: {p2}")

    return s_path, temp_path


def load_param_sweep(path: Path) -> list[dict]:
    """
    Read a param-sweep HDF5 back into a list of per-operating-point dicts, sorted by
    param_value. Each dict has: param_value, sweep_param, omega_0_resonance,
    omega_0_ref, omega_0_linearized, is_resonant, failed, and (when Campaign 2
    completed) the FDT arrays. A row is `failed` if Campaign 2 didn't finish (e.g.
    interrupted after Phase A) -- detected by a missing T_eff_over_T dataset.
    """
    records: list[dict] = []
    with h5py.File(path, "r") as h5:
        omega_0_ref = float(h5.attrs.get("omega_0_ref", math.nan))
        ops = h5["operating_points"]
        for key in sorted(ops.keys()):
            grp = ops[key]
            complete = ("T_eff_over_T" in grp) and not bool(grp.attrs.get("failed", True))
            rec = {
                "param_value": float(grp.attrs["param_value"]),
                "sweep_param": grp.attrs.get("sweep_param", "?"),
                "failed": not complete,
                "omega_0_resonance": float(grp.attrs.get("omega_0_resonance", math.nan)),
                "omega_0_ref": float(grp.attrs.get("omega_0_ref", omega_0_ref)),
                "omega_0_linearized": float(grp.attrs.get("omega_0_linearized", math.nan)),
                "is_resonant": bool(grp.attrs.get("is_resonant", False)),
            }
            for k in ("omega_grid", "omega_norm", "T_eff_over_T", "chi_prime",
                      "chi_double_prime", "PSD_omegas", "PSD_G"):
                if k in grp:
                    rec[k] = grp[k][...]
            records.append(rec)
    records.sort(key=lambda r: r["param_value"])
    return records
