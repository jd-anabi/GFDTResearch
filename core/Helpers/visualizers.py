from os import PathLike

import corner
import numpy as np
import torch
from matplotlib import pyplot as plt

# === GENERAL VISUALIZERS ===
def plot(x: np.ndarray, y: np.ndarray, scatter: bool = False, title: str = None, labels: tuple = None, lims: list = None, hlines: tuple = None, tight: bool = True, sink=None) -> "plt.Figure":
    """Line/scatter plot. Builds an explicit Figure and returns it. ``sink``, if given, is a callable
    ``(title, fig) -> None`` that handles display (e.g. a GUI canvas); when None, falls back to the
    legacy blocking ``plt.show()`` so the CLI is unchanged."""
    fig, ax = plt.subplots()
    if scatter:
        ax.scatter(x, y)
    else:
        ax.plot(x, y)

    if title is not None:
        ax.set_title(title)

    if labels is not None:
        ax.set_xlabel(labels[0])
        if len(labels) > 1:
            ax.set_ylabel(labels[1])

    if lims is not None:
        ax.set_xlim(*lims[0])
        if len(lims) > 1:
            ax.set_ylim(*lims[1])

    if hlines is not None:
        ax.hlines(*hlines, linestyle='--', color='r')

    if tight:
        fig.tight_layout()
    if sink is not None:
        sink(title or "Plot", fig)
    else:
        plt.show()
    return fig

# === DISTRIBUTION VISUALIZERS ===
def visualize_dist(dist: torch.distributions.Distribution, labels: list, n_samples: int = 10000, save_path: str | PathLike[str] = None, title: str = "Distribution", sink=None) -> "plt.Figure":
    """Corner plot of a distribution. Returns the Figure. ``sink`` (title, fig)->None handles display
    for a GUI; when None, falls back to blocking ``plt.show()`` (CLI-unchanged)."""
    # sample from distribution
    samples = dist.sample((n_samples,)).cpu().numpy()

    # generate the corner plot
    figure = corner.corner(samples, labels=labels, show_titles=True, title_fmt=".2f", plot_datapoints=False, plot_density=True, fill_contours=True, smooth=1.0, color=plt.rcParams["text.color"])

    # Size + de-clutter by PARAMETER COUNT. At corner's default a 10-13 parameter grid renders each panel
    # small enough that the tick numbers and the per-column titles overlap into an unreadable smear.
    n_p = samples.shape[-1]
    side = min(24.0, max(8.0, 1.35 * n_p))
    figure.set_size_inches(side, side)
    from matplotlib.ticker import MaxNLocator
    for ax in figure.axes:
        try:
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.tick_params(axis="both", labelsize=7)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_horizontalalignment("right")
            if ax.title.get_text():
                ax.title.set_fontsize(8)
        except Exception:                          # noqa: BLE001 -- cosmetic only, never fatal
            continue

    # save distribution visualization, then display it (sink for GUI, else blocking show)
    if save_path is not None:
        figure.savefig(save_path)
    if sink is not None:
        sink(title, figure)
    else:
        plt.show()
    return figure

# === POSTERIOR ANALYSIS VISUALIZERS ===
def plot_ppc(ppc_results: dict, ground_truth: list = None, param_names: list = None,
             n_samples: int = None, fig_size: tuple = (16, 7)) -> plt.Figure:
    """
    Plot posterior predictive check z-scores.

    :param ppc_results: Dictionary returned by analysis.posterior_predictive_check().
    :param ground_truth: Ground truth parameter values (for subtitle display).
    :param param_names: LaTeX-formatted parameter names (for subtitle display).
    :param n_samples: Number of posterior samples used to generate simulated statistics.
    :param fig_size: Figure size.
    :return: matplotlib Figure.
    """
    z_scores = ppc_results["z_scores"]
    if isinstance(z_scores, torch.Tensor):
        z_scores = z_scores.cpu().detach().numpy()

    n_stats = len(z_scores)
    indices = np.arange(n_stats)

    valid_mask = np.isfinite(z_scores)
    valid_z = z_scores[valid_mask]
    abs_z = np.abs(valid_z)

    # classify points: blue (|z| <= 1), orange (1 < |z| <= 2), red (|z| > 2)
    outside_mask = valid_mask & (np.abs(z_scores) > 2)
    warning_mask = valid_mask & (np.abs(z_scores) > 1) & (np.abs(z_scores) <= 2)
    safe_mask = valid_mask & (np.abs(z_scores) <= 1)
    invalid_mask = ~valid_mask

    # The ground-truth values go in their OWN axes, never the title: 13 parameters joined into a
    # suptitle ran off both edges of the figure and collided with the summary box.
    show_gt = ground_truth is not None and param_names is not None
    n_p = len(param_names) if show_gt else 0
    if show_gt:
        fig = plt.figure(figsize=(fig_size[0], max(fig_size[1], 2.4 + 0.26 * n_p)),
                         constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.05])
        ax = fig.add_subplot(gs[0, 0])
    else:
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        gs, ax = None, fig.add_subplot(1, 1, 1)

    # shaded |z| < 2 region
    ax.axhspan(-2, 2, alpha=0.1, color='green', label=r'$|z| < 2$ region')

    # reference lines
    ax.axhline(0, color=plt.rcParams["axes.edgecolor"], linewidth=0.8)
    ax.axhline(2, color='red', linestyle='--', linewidth=0.8, label=r'$|z| = 2$')
    ax.axhline(-2, color='red', linestyle='--', linewidth=0.8)

    # plot points by category
    ax.scatter(indices[safe_mask], z_scores[safe_mask], c='steelblue',
               s=40, alpha=0.7, edgecolors='none', zorder=3)
    ax.scatter(indices[warning_mask], z_scores[warning_mask], c='orange',
               s=40, alpha=0.8, edgecolors='none', zorder=3)
    ax.scatter(indices[outside_mask], z_scores[outside_mask], c='red',
               s=50, alpha=0.9, edgecolors='none', zorder=4)
    if invalid_mask.any():
        ax.scatter(indices[invalid_mask], np.zeros(invalid_mask.sum()), c='gray',
                   s=50, marker='x', linewidths=1.5, zorder=4, label='Invalid (zero variance)')

    ax.set_xlabel('Statistic Index')
    ax.set_ylabel('Z-Score')
    ax.set_ylim(-3.5, 3.5)
    ax.legend(loc='upper left')

    title = "Posterior Predictive Check"
    if n_samples is not None:
        title += f" (N = {n_samples} samples)"
    ax.set_title(title)

    # Summary box. The heading is the box's FIRST LINE rather than a separate label pinned at y=1.0 --
    # that label sat exactly where the title lives and the two overlapped.
    num_total = n_stats
    textstr = (
        "Summary statistics\n"
        f"Mean $|z|$: {abs_z.mean():.3f}\n"
        f"Max $|z|$: {abs_z.max():.3f}\n"
        f"Coverage (90%): {ppc_results['coverage_90'] * 100:.1f}%\n"
        f"Outside interval: {ppc_results['num_outside']}/{num_total}\n"
        f"Invalid stats: {ppc_results['num_invalid']}/{num_total}"
    )
    props = dict(boxstyle='round', facecolor=plt.rcParams["axes.facecolor"],
                 edgecolor=plt.rcParams["axes.edgecolor"], alpha=0.9)
    ax.text(0.99, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            family='monospace')

    if show_gt:
        _param_table(fig.add_subplot(gs[0, 1]), param_names, ground_truth,
                     value_header="ground truth")
    return fig


def plot_posterior_vs_truth(t: np.ndarray, x_true: np.ndarray,
                           x_mean: np.ndarray = None, x_median: np.ndarray = None,
                           x_samples: np.ndarray = None, n_show: int = 10,
                           fig_size: tuple = (14, 5),
                           xlabel: str = r"$t$ (s)", ylabel: str = r"$x$ (nm)") -> plt.Figure:
    """
    Overlay posterior-simulated trajectories on top of ground truth data.

    :param t: Time array (steady-state portion), shape (T,).
    :param x_true: Ground truth x-position time series, shape (T,).
    :param x_mean: Posterior-mean-parameter trajectory, shape (T,). Optional.
    :param x_median: Posterior-median-parameter trajectory, shape (T,). Optional.
    :param x_samples: Posterior sample trajectories, shape (N, T).
    :param n_show: Number of individual sample trajectories to display.
    :param fig_size: Figure size.
    :return: matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # Pointwise 95% PERCENTILE band across samples -- NOT mean +- 2*std. These trajectories are
    # oscillations with independent (noise-set) phases, so a pointwise MEAN destructively cancels and a
    # mean-centred band collapses to a flat ribbon around the mean level, saying nothing about the
    # predicted signal. Percentiles are bounded by real sample values, so the band reads as the envelope
    # the posterior actually predicts. (Phase-aware comparisons live in the dedicated overlay figures.)
    if x_samples is not None and len(x_samples) > 1:
        lo, hi = np.percentile(x_samples, [2.5, 97.5], axis=0)
        ax.fill_between(t, lo, hi, alpha=0.15, color='steelblue', label='Posterior 95% band')

        # Individual sample trajectories. Seeded via a LOCAL RandomState so the figure is reproducible
        # across re-renders without disturbing the global numpy RNG the pipeline draws from.
        rng = np.random.RandomState(0)
        show_idx = rng.choice(len(x_samples), size=min(n_show, len(x_samples)), replace=False)
        for i, idx in enumerate(show_idx):
            ax.plot(t, x_samples[idx], color='steelblue', alpha=0.25, linewidth=0.5,
                    label='Posterior samples' if i == 0 else None)

    ax.plot(t, x_true, color=plt.rcParams["text.color"], linewidth=1.2, label='Ground truth')
    if x_median is not None:
        ax.plot(t, x_median, color='red', linewidth=1.0, linestyle='--', label='Posterior median')
    if x_mean is not None:
        ax.plot(t, x_mean, color='darkorange', linewidth=1.0, linestyle='-.', label='Posterior mean')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Posterior vs Ground Truth')
    ax.legend()
    plt.tight_layout()
    return fig


# ── Posterior-overlay figures (phase-aware; see core/SBI/overlay.py for why) ──────────────────────
def _param_table(ax, labels, values, ground_truth=None, value_header: str = "best fit"):
    """Render a parameter table into its own axes.

    Its own axes on purpose: 13 parameters crammed into a title runs off both edges of the figure (the
    failure mode of the older PPC plot), and a table stays readable as the parameter count grows."""
    ax.axis("off")
    show_truth = ground_truth is not None
    header = ["parameter", value_header] + (["truth", "% err"] if show_truth else [])
    rows = []
    for i, name in enumerate(labels):
        v = float(values[i])
        row = [name, f"{v:.4g}"]
        if show_truth:
            g = float(ground_truth[i])
            err = (v - g) / abs(g) * 100.0 if abs(g) > 1e-30 else float("nan")
            row += [f"{g:.4g}", "—" if err != err else f"{err:+.1f}"]
        rows.append(row)
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.18)
    fg = plt.rcParams["text.color"]
    for cell in table.get_celld().values():
        cell.set_edgecolor(plt.rcParams["axes.edgecolor"])
        cell.get_text().set_color(fg)
        cell.set_facecolor("none")
    return table


def plot_best_fit_overlay(t: np.ndarray, x_true: np.ndarray, x_fit: np.ndarray, *,
                          param_labels: list = None, param_values=None, ground_truth=None,
                          criterion: str = "", score_text: str = "",
                          xlabel: str = r"$t$ (s)", ylabel: str = r"$x$ (nm)") -> plt.Figure:
    """A single posterior draw, PHASE-ALIGNED, over the observation, plus its parameter table.

    Absolute phase is set by the noise realisation rather than by theta, so it is aligned away before
    plotting: what the eye should be judging is frequency, amplitude, mean and waveform shape.
    """
    n_params = len(param_labels) if param_labels else 0
    height = max(4.6, 2.2 + 0.26 * n_params)
    fig = plt.figure(figsize=(15, height), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, x_true, color=plt.rcParams["text.color"], linewidth=1.2, label="Observation")
    ax.plot(t, x_fit, color="crimson", linewidth=1.0, alpha=0.9, label="Best-fit draw (aligned)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Best-fit posterior draw — {criterion}" + (f"   ({score_text})" if score_text else ""))
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    if n_params:
        _param_table(fig.add_subplot(gs[0, 1]), param_labels, param_values, ground_truth)
    return fig


def plot_overlay_band(t: np.ndarray, x_true: np.ndarray, lo: np.ndarray, med: np.ndarray,
                      hi: np.ndarray, *, n_used: int = 0, pct: tuple = (5, 95),
                      xlabel: str = r"$t$ (s)", ylabel: str = r"$x$ (nm)") -> plt.Figure:
    """The top-N best-matching draws, each individually phase-aligned, as a percentile band."""
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.fill_between(t, lo, hi, alpha=0.25, color="steelblue",
                    label=f"best {n_used} draws, {pct[0]}–{pct[1]}%")
    ax.plot(t, med, color="steelblue", linewidth=1.0, label="median of best draws")
    ax.plot(t, x_true, color=plt.rcParams["text.color"], linewidth=1.2, label="Observation")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Posterior overlay — best draws, phase-aligned")
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    return fig


def plot_psd_overlay(freqs: np.ndarray, gt_power: np.ndarray, lo: np.ndarray, med: np.ndarray,
                     hi: np.ndarray, *, pct: tuple = (5, 95), freq_unit: str = "Hz") -> plt.Figure:
    """Observation PSD vs the posterior-predictive PSD band. Phase-invariant by construction, so this is
    the honest "do frequency, amplitude and harmonic content agree?" check -- nothing is aligned away."""
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    m = freqs > 0                                        # log axes: drop DC
    ax.fill_between(freqs[m], lo[m], hi[m], alpha=0.25, color="steelblue",
                    label=f"posterior predictive, {pct[0]}–{pct[1]}%")
    ax.plot(freqs[m], med[m], color="steelblue", linewidth=1.0, label="posterior median")
    ax.plot(freqs[m], gt_power[m], color=plt.rcParams["text.color"], linewidth=1.1,
            label="Observation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"frequency ({freq_unit})")
    ax.set_ylabel("power spectral density")
    ax.set_title("Power spectrum — observation vs posterior predictive")
    ax.legend(loc="lower left", fontsize=9)
    return fig


def plot_cycle_average(phase: np.ndarray, gt_mean: np.ndarray, sim_mean: np.ndarray,
                       sim_lo: np.ndarray, sim_hi: np.ndarray, *,
                       ylabel: str = r"$x$ (nm)") -> plt.Figure:
    """Observation vs posterior predictive, folded onto ONE oscillation cycle.

    Shows whether the waveform SHAPE agrees (the asymmetric hair-bundle spike) without depending on
    absolute phase at all -- the complement to the aligned time-domain overlays."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.fill_between(phase, sim_lo, sim_hi, alpha=0.25, color="steelblue",
                    label="posterior predictive, 25–75%")
    ax.plot(phase, sim_mean, color="steelblue", linewidth=1.2, label="posterior mean cycle")
    ax.plot(phase, gt_mean, color=plt.rcParams["text.color"], linewidth=1.4, label="Observation")
    ax.set_xlabel("cycle phase (rad)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_title("Cycle-averaged waveform (phase-invariant)")
    ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_training_loss(diagnostics: dict, save_path: str | PathLike[str] = None,
                       fig_size: tuple = (10, 5)) -> plt.Figure | None:
    """
    Plot the NPE per-epoch training/validation loss curve for a convergence check.

    Distinguishes "under-fit" (validation loss still descending near the best epoch ->
    train longer / raise capacity) from "converged" (clean plateau well before the best
    epoch -> remaining wide marginals are a data/identifiability limit, not under-fit).

    :param diagnostics: dict from train_nn carrying 'training_loss', 'validation_loss',
                        and optionally 'best_validation_loss', 'epochs_trained',
                        'stop_after_epochs'.
    :param save_path: optional path to save the figure.
    :param fig_size: figure size.
    :return: the matplotlib Figure, or None if no validation curve is present.
    """
    val = diagnostics.get("validation_loss") or []
    train = diagnostics.get("training_loss") or []
    if len(val) == 0:
        print("plot_training_loss: no validation_loss curve in diagnostics; nothing to plot.")
        return None

    fig, ax = plt.subplots(figsize=fig_size)
    if len(train):
        ax.plot(np.arange(1, len(train) + 1), train, color='steelblue', label='training loss')
    ax.plot(np.arange(1, len(val) + 1), val, color='darkorange', label='validation loss')

    best_epoch = int(np.argmin(val)) + 1
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=1.0,
               label=f'best epoch ({best_epoch})')
    sae = diagnostics.get("stop_after_epochs")
    if sae:
        ax.axvspan(best_epoch, len(val), alpha=0.08, color='red',
                   label=f'early-stop window ({sae}-epoch patience)')

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (lower = better)')
    title = 'NPE training / validation loss'
    et, bvl = diagnostics.get("epochs_trained"), diagnostics.get("best_validation_loss")
    if et is not None and bvl is not None:
        title += f'  (epochs={et}, best val={bvl:.4f})'
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fig