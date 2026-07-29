"""
Part B sweep + top-level CLI entry point for the reduction map.

`sweep_f_max` produces a DataFrame of analytical Hopf predictions as `f_max`
varies; `run_reduction_map` is the interactive entry point that does Part A
(report at cell-file params) then Part B (sweep + save + plot).
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FDTConfig
from .fixed_point import solve_fixed_point
from .reduce import reduce_nwk_to_hopf, ReductionRecord, ReductionFailure, NWK_KEYS


# Continuity threshold: relative jump in P_o* (or x̃*) flagged as a fold crossing.
_CONTINUITY_REL_TOL = 0.05


def _nwk_params_from_cfg(cfg: FDTConfig) -> dict:
    """Extract the 10 NWK ND params from cfg.params_dict using cell-file naming."""
    return {k: cfg.params_dict[k][0] for k in NWK_KEYS}


def _t_x_scale(cfg: FDTConfig) -> tuple[float, float]:
    """Pull t_scale (= λ/K_gs) and x_scale (= D) from cfg.rescale_params."""
    return cfg.rescale_params["t_scale"][0], cfg.rescale_params["x_scale"][0]


def sweep_f_max(
    cfg: FDTConfig,
    f_max_grid: np.ndarray,
    F_amplitude: float = 1.0,
) -> pd.DataFrame:
    """
    Sweep over f_max values, evaluating the analytical reduction map at each.

    :param cfg: FDTConfig providing the baseline NWK params + dimensional factors.
                All params except f_max are held at their cell-file values.
    :param f_max_grid: 1D array of f_max values to evaluate.
    :param F_amplitude: NWK forcing amplitude passed through to Phase B1.

    :returns: pandas DataFrame, one row per f_max value. Sorted in input order.
              Failed points (ReductionFailure) appear as rows with NaNs and a
              `failed` flag set to True; continuity flags compare adjacent
              successful points.
    """
    base_params = _nwk_params_from_cfg(cfg)
    t_scale_nwk, x_scale_nwk = _t_x_scale(cfg)

    rows: list[dict] = []
    prev_record: ReductionRecord | None = None

    for f_max in f_max_grid:
        params = dict(base_params)
        params["f_max"] = float(f_max)

        try:
            rec = reduce_nwk_to_hopf(
                params, t_scale_nwk=t_scale_nwk, x_scale_nwk=x_scale_nwk,
                F_amplitude=F_amplitude,
            )
            row = rec.to_flat_dict()
            row["f_max"] = float(f_max)
            row["failed"] = False

            if prev_record is not None:
                d_Po = abs(rec.P_o_star - prev_record.P_o_star)
                d_x = abs(rec.x_star - prev_record.x_star)
                ref = max(abs(prev_record.P_o_star), 1e-12)
                refx = max(abs(prev_record.x_star), 1e-12)
                row["continuity_flag"] = (d_Po / ref < _CONTINUITY_REL_TOL
                                          and d_x / refx < _CONTINUITY_REL_TOL)
            else:
                row["continuity_flag"] = True

            prev_record = rec
        except ReductionFailure as e:
            # Partial diagnostic: keep fixed-point data even when the Hopf cubic
            # has no admissible (complex-pair) root — that's a meaningful
            # off-Hopf-manifold signal we want to see in the table.
            row = {"f_max": float(f_max), "failed": True, "error": str(e),
                   "continuity_flag": False}
            try:
                fp = solve_fixed_point(
                    k=params["k"], f_max=params["f_max"], s=params["s"],
                    beta=params["beta"], delta_E=params["delta_E"],
                )
                row["P_o_star"] = fp.P_o_star
                row["xi_star"] = fp.xi_star
                row["x_star"] = fp.x_star
                row["y_star"] = fp.y_star
                row["c_star"] = fp.c_star
                row["g"] = fp.g
            except ValueError as e:
                # RECORD it. A bare `pass` left the fixed-point columns simply absent from this row,
                # so a sweep with every fixed-point solve failing looked identical to one that had
                # not been asked for them.
                row["fixed_point_error"] = f"{type(e).__name__}: {e}"
            prev_record = None

        rows.append(row)

    return pd.DataFrame(rows)


def _default_f_max_grid(cell_value: float, n_points: int = 13,
                       frac: float = 0.10) -> np.ndarray:
    """
    Symmetric grid spanning [(1-frac)·cell, (1+frac)·cell] with n_points samples.

    Default frac=0.10 (±10%) bracket: the Hopf manifold tends to be narrow in
    parameter space, so ±30% often falls off-manifold (Jacobian goes purely real)
    for most of the sweep. Callers should adjust frac based on Part A's μ_H^(N):
    if |μ_H^(N)| is small, the manifold is close and ±5% may suffice; if large,
    extend the bracket to catch the regime transition.
    """
    return np.linspace(cell_value * (1.0 - frac), cell_value * (1.0 + frac), n_points)


def _print_part_a_report(rec: ReductionRecord) -> None:
    print("\n" + "=" * 60)
    print("Part A: reduction map at cell-file parameters")
    print("=" * 60)
    print(f"  Fixed point:")
    print(f"    P_o*   = {rec.P_o_star:.6f}")
    print(f"    ξ*     = {rec.xi_star:.6f}")
    print(f"    x̃*     = {rec.x_star:.6f}")
    print(f"    ỹ*     = {rec.y_star:.6f}")
    print(f"    c̃*     = {rec.c_star:.6f}")
    print(f"    g      = {rec.g:.6f}")
    print(f"  Phase A linear (NWK-ND units):")
    print(f"    a_1, a_2, a_3 = {rec.a_1:.6f}, {rec.a_2:.6f}, {rec.a_3:.6f}")
    print(f"    μ_H^(N) = {rec.mu_N:+.6f}   (regime: {rec.regime})")
    print(f"    Ω_0^(N) = {rec.Omega0_N:.6f}")
    print(f"    ν       = {rec.nu:.6f}")
    print(f"  Phase A cubic:")
    print(f"    α_H^(N) = {rec.alpha_H_N:+.6e}")
    print(f"    β_H^(N) = {rec.beta_H_N:+.6e}")
    print(f"    g_21^eff = {rec.g21_eff.real:+.6e} + {rec.g21_eff.imag:+.6e}i")
    print(f"  Phase B noise (NWK-ND units):")
    print(f"    Σ̂_{{z̄z}}    = {rec.Sigma_hat_barzz:.6e}")
    print(f"    |Σ̂_{{zz}}|   = {rec.Sigma_hat_zz_mag:.6e}")
    print(f"    σ̃_x^(N)    = {rec.sigma_x_N:.6e}")
    print(f"    σ̃_y^(N)    = {rec.sigma_y_N:.6e}")
    print(f"    θ_rot      = {rec.theta_rot:+.6f} rad")
    print(f"  Stage 2 (fully-ND Hopf + dim factors):")
    print(f"    μ̃          = {rec.mu_tilde:+.6e}")
    print(f"    β̃          = {rec.beta_tilde:+.6e}")
    print(f"    σ̃_x^Hopf   = {rec.sigma_x_Hopf:.6e}")
    print(f"    σ̃_y^Hopf   = {rec.sigma_y_Hopf:.6e}")
    print(f"    t_scale    = {rec.t_scale_Hopf:.6e}   (Hopf-side, NWK_t_scale / Ω_0^(N))")
    print(f"    x_scale    = {rec.x_scale_Hopf:.6e}   (Hopf-side, NWK_x_scale · √(Ω_0 / α_H))")
    print("=" * 60 + "\n")


def run_reduction_map(cfg: FDTConfig) -> ReductionRecord:
    """
    CLI entry point. Runs Part A (cell-file report) then Part B (f_max sweep),
    saves the sweep table to Resources/ReductionMap/, and returns the Part A
    record so callers can chain further work.
    """
    nwk_params = _nwk_params_from_cfg(cfg)
    t_scale_nwk, x_scale_nwk = _t_x_scale(cfg)

    part_a = reduce_nwk_to_hopf(
        nwk_params, t_scale_nwk=t_scale_nwk, x_scale_nwk=x_scale_nwk,
        F_amplitude=cfg.F0,
    )
    _print_part_a_report(part_a)

    # Part B sweep
    cell_f_max = float(cfg.params_dict["f_max"][0])
    grid = _default_f_max_grid(cell_f_max)
    print(f"Part B: sweeping f_max over {len(grid)} values in [{grid[0]:.3f}, {grid[-1]:.3f}]"
          f"  (cell value: {cell_f_max:.3f})\n")
    df = sweep_f_max(cfg, grid, F_amplitude=cfg.F0)

    # Save table
    out_dir = Path("Resources/ReductionMap")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sweep_{stamp}.parquet"
    df.to_parquet(out_path)
    print(f"Saved sweep table to: {out_path}")
    print(df[["f_max", "mu_N", "Omega0_N", "alpha_H_N", "beta_H_N", "regime",
              "continuity_flag", "failed"]].to_string(index=False))

    # Diagnostic plot
    from .plots import plot_sweep_summary
    fig_path = plot_sweep_summary(df, save=True, show=False)
    if fig_path is not None:
        print(f"Saved sweep diagnostic plot to: {fig_path}")

    return part_a
