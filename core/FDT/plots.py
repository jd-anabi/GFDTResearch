"""Plotting helpers for FDT analysis output."""
from os import PathLike

import numpy as np
from matplotlib import pyplot as plt


def _normalized_freq_axis(omegas: np.ndarray, omega_natural: float):
    """
    Build a frequency axis normalized by the characteristic frequency Omega_0.

    :param omegas: raw angular frequencies (ND).
    :param omega_natural: characteristic frequency Omega_0 (PSD peak). If None or
                          non-positive, the raw omega axis is returned instead.
    :return: (x_values, x_label, resonance_x) where resonance_x is the x-position
             of Omega_0 (1.0 when normalized, None otherwise).
    """
    if omega_natural is not None and omega_natural > 0:
        return omegas / omega_natural, r'$\tilde\omega / \Omega_0$, log scale', 1.0
    return omegas, r'$\tilde\omega$ (ND), log scale', None


def plot_eff_temp_ratio(omegas: np.ndarray, ratio: np.ndarray,
                        save_path: str | PathLike[str] = None,
                        title: str = "FDT violation diagnostic",
                        omega_natural: float = None,
                        linthresh: float = 1.0) -> None:
    """
    Plot T_eff/T vs normalized driving frequency omega/Omega_0.

    Linear x-axis (omega/Omega_0), symlog y-axis so the divergence is visible on
    BOTH sides of zero (the chi'' sign change manifests as a +inf -> -inf crossing
    through y=0 near omega/Omega_0 = 1). The linear region |y| < linthresh keeps the
    equilibrium reference y=1 readable.

    Non-finite values (NaN/inf) are dropped with a warning; negative ratios are kept.

    :param omegas: (n_freqs,) driving angular frequencies (ND).
    :param ratio:  (n_freqs,) T_eff(omega)/T values.
    :param save_path: optional path to save the figure.
    :param title: plot title.
    :param omega_natural: characteristic frequency Omega_0 (PSD peak). The x-axis is
                          omega/Omega_0; if None, raw omega is used.
    :param linthresh: half-width of the linear region around y=0 in the symlog y-axis.
                      Defaults to 1.0 (the equilibrium value).
    """
    omegas = np.asarray(omegas)
    ratio = np.asarray(ratio)

    n_bad = int((~np.isfinite(ratio)).sum())
    if n_bad > 0:
        print(f"plot_eff_temp_ratio: dropping {n_bad}/{len(ratio)} non-finite (NaN/inf) points.")
        mask = np.isfinite(ratio)
        omegas, ratio = omegas[mask], ratio[mask]

    x, xlabel, x_res = _normalized_freq_axis(omegas, omega_natural)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, ratio, marker='o', linestyle='none', markersize=4, color='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=linthresh)
    ax.axhline(1.0, color=plt.rcParams['axes.edgecolor'], linestyle='--', linewidth=0.8,
               label=r'$T_{\rm eff}/T = 1$ (equilibrium)')
    # axes.edgecolor, not 'gray': this is figure CHROME and must follow the theme, exactly like the
    # equilibrium line above it. Deliberately the SAME token as that line -- the two are separated by
    # dash pattern and width, which is matplotlib's own convention for reference lines and survives a
    # theme flip. `grid.color` was the obvious "dimmer neutral" and is WRONG: it is the border token
    # (#3D3D3D) and the dark theme paints axes on #2B2B2B, so the line and its legend swatch both
    # disappear -- worse than the hardcoded gray it replaced. (B-c's last residue.)
    ax.axhline(0.0, color=plt.rcParams['axes.edgecolor'], linestyle=':', linewidth=0.6,
                label=r"$T_{\rm eff}/T = 0$ ($\chi''=0$ crossing)")
    if x_res is not None:
        ax.axvline(x_res, color='darkorange', linestyle=':', linewidth=1.2,
                    label=fr'$\omega/\Omega_0 = 1$ ($\Omega_0 \approx {omega_natural:.3f}$)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$T_{\rm eff}(\tilde\omega) / T$, symlog scale')
    ax.set_title(title)
    ax.grid(False)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_spontaneous_trajectory(t: np.ndarray, x_mean: np.ndarray,
                                  save_path: str | PathLike[str] = None,
                                  title: str = "Spontaneous trajectory (Campaign 1)",
                                  burn_in: float = None) -> None:
    """
    Plot the ensemble-mean unforced trajectory <X>(t) from Campaign 1.

    For a stationary equilibrium system this should: (a) follow the deterministic
    transient during burn-in, (b) decorrelate as independent noise averages out,
    (c) settle to fluctuations of amplitude ~ sigma_x / sqrt(M) around the
    equilibrium mean. Useful as a visual stationarity / burn-in check.

    :param t: (T,) time axis in ND units.
    :param x_mean: (T,) ensemble-mean bundle position trajectory.
    :param save_path: optional path to save the figure.
    :param title: plot title.
    :param burn_in: optional ND time at which to draw a vertical line marking the
                    transient/steady-state boundary (where PSD computation starts).
    """
    t = np.asarray(t)
    x_mean = np.asarray(x_mean)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x_mean, linewidth=0.8, color='steelblue')
    ax.axhline(0.0, color=plt.rcParams['axes.edgecolor'], linestyle='--', linewidth=0.8)
    if burn_in is not None:
        ax.axvline(burn_in, color='darkorange', linestyle=':', linewidth=1.2,
                    label=fr'burn-in ends: $\tilde t = {burn_in:.0f}$')
        ax.legend()
    ax.set_xlabel(r'$\tilde t$ (ND)')
    ax.set_ylabel(r'$\langle \tilde X \rangle(\tilde t)$ (ND)')
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_psd(omegas: np.ndarray, G: np.ndarray,
             save_path: str | PathLike[str] = None,
             title: str = "Spontaneous PSD (Campaign 1)",
             omega_natural: float = None,
             plot_band: tuple = None) -> None:
    """
    Linear plot of one-sided PSD G(omega) vs normalized frequency omega/Omega_0.

    Used to visually verify the spontaneous-oscillation frequency that
    find_spectral_peak picks out (the peak lands at omega/Omega_0 = 1). Non-finite
    PSD values (numerical artifacts) are dropped with a warning.

    :param omegas: (n_freqs,) angular frequencies (ND) from psd_welch.
    :param G: (n_freqs,) one-sided PSD values from psd_welch.
    :param save_path: optional path to save the figure.
    :param title: plot title.
    :param omega_natural: characteristic frequency Omega_0; x-axis is omega/Omega_0.
    :param plot_band: optional (lo, hi) in RAW omega units to restrict the x-range,
                      e.g. matching the production grid. Applied before normalization.
    """
    omegas = np.asarray(omegas)
    G = np.asarray(G)

    # Skip DC bin
    mask = omegas > 0
    omegas, G = omegas[mask], G[mask]

    n_bad = int((~np.isfinite(G)).sum())
    if n_bad > 0:
        print(f"plot_psd: dropping {n_bad}/{len(G)} non-finite PSD points.")
        good = np.isfinite(G)
        omegas, G = omegas[good], G[good]

    if plot_band is not None:
        in_band = (omegas >= plot_band[0]) & (omegas <= plot_band[1])
        omegas, G = omegas[in_band], G[in_band]

    x, xlabel, x_res = _normalized_freq_axis(omegas, omega_natural)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, G, marker='.', linestyle='none', markersize=3, color='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if x_res is not None:
        ax.axvline(x_res, color='darkorange', linestyle=':', linewidth=1.2,
                    label=fr'$\omega/\Omega_0 = 1$ ($\Omega_0 \approx {omega_natural:.3f}$)')
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$G(\tilde\omega)$, log scale')
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_chi_components(omegas: np.ndarray, chis: np.ndarray,
                        save_path: str | PathLike[str] = None,
                        title: str = "Susceptibility components",
                        omega_natural: float = None) -> None:
    """
    Two-panel plot of chi'(omega) and chi''(omega) vs normalized frequency omega/Omega_0.

    Linear axes, comparable to Martin et al. 2001 Fig 2. Negative chi'' below the
    natural frequency indicates active force generation (energy gain from the drive);
    positive chi'' indicates dissipation. chi' (in-phase) peaks at the effective
    resonance.

    :param omegas: (n_freqs,) driving angular frequencies (ND).
    :param chis: (n_freqs,) complex chi values.
    :param save_path: optional path to save the figure.
    :param title: plot title (applied to the top panel).
    :param omega_natural: characteristic frequency Omega_0; x-axis is omega/Omega_0.
    """
    omegas = np.asarray(omegas)
    chis = np.asarray(chis)
    chi_real = chis.real
    chi_imag = chis.imag

    x, xlabel, x_res = _normalized_freq_axis(omegas, omega_natural)

    fig, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_real.set_xscale('log')   # shared with ax_imag via sharex=True

    # chi' (real / in-phase)
    ax_real.plot(x, chi_real, marker='o', linestyle='none', markersize=4, color='steelblue')
    ax_real.axhline(0.0, color=plt.rcParams['axes.edgecolor'], linestyle='--', linewidth=0.8)
    if x_res is not None:
        ax_real.axvline(x_res, color='darkorange', linestyle=':', linewidth=1.2,
                         label=fr'$\omega/\Omega_0 = 1$ ($\Omega_0 \approx {omega_natural:.3f}$)')
        ax_real.legend()
    ax_real.set_ylabel(r"$\chi'(\tilde\omega)$ (in-phase)")
    ax_real.set_title(title)
    ax_real.grid(False)

    # chi'' (imaginary / dissipative)
    ax_imag.plot(x, chi_imag, marker='o', linestyle='none', markersize=4, color='steelblue')
    ax_imag.axhline(0.0, color=plt.rcParams['axes.edgecolor'], linestyle='--', linewidth=0.8)
    if x_res is not None:
        ax_imag.axvline(x_res, color='darkorange', linestyle=':', linewidth=1.2)
    ax_imag.set_xlabel(xlabel)
    ax_imag.set_ylabel(r"$\chi''(\tilde\omega)$ (dissipative)")
    ax_imag.grid(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()
