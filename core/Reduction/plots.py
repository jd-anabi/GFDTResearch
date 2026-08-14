"""
Diagnostic plots for the reduction map.

Two kinds:
  - Sweep summary plot (analytical only): μ_H, Ω_0, α_H, β_H vs f_max.
  - 3D cross-validation plot: T_eff(ω)/T from FDT vs (ω, μ_H), with the
    predicted resonance line Ω_0^(N)(μ_H^(N)) overlaid. Requires an FDT
    measurement dict supplied separately by the caller.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PLOT_DIR = Path("Resources/Plots")


def _overlay_halo() -> list:
    """A themed outline for a WHITE line drawn on a colormap (B-c's last residue).

    White is the right colour for the line itself and must NOT follow the page theme -- its backdrop
    is viridis, not the figure, so a themed colour would be near-black on dark purple under the light
    theme. That is the same call ``cross_validation_plots`` makes with ``darkorange`` and
    ``simulate_export`` makes by pinning a white video figure.

    Its LEGEND swatch is the part that was actually broken: the legend sits on ``legend.facecolor``,
    which is ``#FFFFFF`` under the light theme and ``inherit`` -> white under plain matplotlib, so the
    entry for the predicted-Omega_0 line was an invisible white line on white. A stroke in
    ``text.color`` fixes the swatch on any background -- the token pair is contrast-guaranteed by
    construction -- and on the surface it only adds a thin halo, which helps in light and is neutral
    in dark.

    Read at CALL time, never hoisted to a module constant: rcParams is global and re-applied on a
    theme flip (``gui.mpl_theme``), so an import-time snapshot would pin the theme the process started
    in.
    """
    return [pe.Stroke(linewidth=3.5, foreground=plt.rcParams["text.color"]), pe.Normal()]


def plot_sweep_summary(df: pd.DataFrame, save: bool = True, show: bool = False) -> Path | None:
    """
    Plot the four NWK-ND Hopf parameters as functions of f_max.

    :returns: path to saved figure, or None if save=False.
    """
    ok = df[df["failed"] == False]  # noqa: E712 — explicit boolean

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    ax_mu, ax_Omega = axes[0]
    ax_alpha, ax_beta = axes[1]

    ax_mu.plot(ok["f_max"], ok["mu_N"], "o-", color="tab:blue")
    ax_mu.axhline(0.0, color=plt.rcParams['axes.edgecolor'], lw=0.5, ls="--")
    ax_mu.set_ylabel(r"$\mu_H^{(N)}$")
    ax_mu.set_title("Bifurcation distance")

    ax_Omega.plot(ok["f_max"], ok["Omega0_N"], "o-", color="tab:green")
    ax_Omega.set_ylabel(r"$\Omega_0^{(N)}$")
    ax_Omega.set_title("Intrinsic frequency")

    ax_alpha.plot(ok["f_max"], ok["alpha_H_N"], "o-", color="tab:orange")
    ax_alpha.axhline(0.0, color=plt.rcParams['axes.edgecolor'], lw=0.5, ls="--")
    ax_alpha.set_ylabel(r"$\alpha_H^{(N)}$")
    ax_alpha.set_xlabel(r"$f_{\max}$")
    ax_alpha.set_title("Cubic real part")

    ax_beta.plot(ok["f_max"], ok["beta_H_N"], "o-", color="tab:red")
    ax_beta.axhline(0.0, color=plt.rcParams['axes.edgecolor'], lw=0.5, ls="--")
    ax_beta.set_ylabel(r"$\beta_H^{(N)}$")
    ax_beta.set_xlabel(r"$f_{\max}$")
    ax_beta.set_title("Cubic imaginary part")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    # Highlight subcritical/supercritical regions by background shading
    if not ok.empty:
        sub = ok[ok["regime"] == "subcritical"]
        sup = ok[ok["regime"] == "supercritical"]
        for ax in axes.flat:
            if not sub.empty:
                ax.axvspan(sub["f_max"].min(), sub["f_max"].max(),
                           color="tab:blue", alpha=0.05, zorder=0)
            if not sup.empty:
                ax.axvspan(sup["f_max"].min(), sup["f_max"].max(),
                           color="tab:red", alpha=0.05, zorder=0)

    fig.suptitle(r"Reduction-map sweep: NWK-ND Hopf parameters vs $f_{\max}$", y=1.0)
    fig.tight_layout()

    if save:
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _PLOT_DIR / f"reduction_sweep_{stamp}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return path
    if show:
        plt.show()
    return None


def plot_cross_validation_3d(
    df: pd.DataFrame,
    fdt_measurements: list[dict],
    save: bool = True,
    show: bool = False,
) -> Path | None:
    """
    3D surface of T_eff(ω̃)/T vs (ω̃, μ_H^(N)) with the predicted resonance
    line Ω_0^(N)(μ_H^(N)) overlaid.

    :param df: sweep table with at least columns f_max, mu_N, Omega0_N, regime.
    :param fdt_measurements: list of per-operating-point dicts. Each dict must have:
        - 'f_max':         float, matches a row in df
        - 'omega_grid':    1D np.ndarray, NWK-ND frequencies (shared length expected)
        - 'T_eff_over_T':  1D np.ndarray, same length as omega_grid
    """
    if not fdt_measurements:
        raise ValueError("fdt_measurements is empty; nothing to plot.")

    # Align rows by f_max
    df_indexed = df.set_index("f_max")
    rows = []
    for m in fdt_measurements:
        fm = float(m["f_max"])
        if fm not in df_indexed.index:
            continue
        rows.append((fm, df_indexed.loc[fm, "mu_N"], df_indexed.loc[fm, "Omega0_N"],
                     np.asarray(m["omega_grid"]), np.asarray(m["T_eff_over_T"])))

    if not rows:
        raise ValueError("No fdt_measurements aligned to df rows.")

    # Assume shared ω-grid across operating points; pick first
    omega = rows[0][3]
    mu_axis = np.array([r[1] for r in rows])
    Omega_axis = np.array([r[2] for r in rows])
    T_eff_matrix = np.stack([r[4] for r in rows], axis=0)   # (n_op, n_omega)

    # Sort rows by mu_N
    order = np.argsort(mu_axis)
    mu_axis = mu_axis[order]
    Omega_axis = Omega_axis[order]
    T_eff_matrix = T_eff_matrix[order]

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    halo = _overlay_halo()
    OMEGA, MU = np.meshgrid(omega, mu_axis)
    ax3d.plot_surface(OMEGA, MU, T_eff_matrix, cmap="viridis",
                      edgecolor="none", alpha=0.85)
    ax3d.plot(Omega_axis, mu_axis,
              [T_eff_matrix[i, np.argmin(np.abs(omega - Omega_axis[i]))]
               for i in range(len(mu_axis))],
              color="white", lw=2, path_effects=halo,
              label=r"predicted $\Omega_0^{(N)}(\mu_H)$")
    ax3d.set_xlabel(r"$\tilde\omega$")
    ax3d.set_ylabel(r"$\mu_H^{(N)}$")
    ax3d.set_zlabel(r"$T_{\rm eff}/T$")
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.set_title("FDT response vs bifurcation distance")

    im = ax2d.pcolormesh(OMEGA, MU, T_eff_matrix, cmap="viridis", shading="auto")
    ax2d.plot(Omega_axis, mu_axis, "w-", lw=2, path_effects=halo,
              label=r"predicted $\Omega_0^{(N)}(\mu_H)$")
    ax2d.axhline(0.0, color="r", lw=0.7, ls="--", label="Hopf manifold")
    ax2d.set_xlabel(r"$\tilde\omega$")
    ax2d.set_ylabel(r"$\mu_H^{(N)}$")
    ax2d.set_title("2D projection")
    ax2d.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax2d, label=r"$T_{\rm eff}/T$")

    fig.tight_layout()

    if save:
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _PLOT_DIR / f"reduction_crossval_{stamp}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return path
    if show:
        plt.show()
    return None
