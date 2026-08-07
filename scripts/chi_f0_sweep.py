"""
Gate the chi(omega) probe BAND and drive amplitude on the axes that actually vary in training:
probe frequency, drive amplitude F0, and -- since 2026-08-06 -- OBSERVATION LENGTH T_obs.

WHY. chi mode drives every probe at a fixed ND amplitude (config.CHI_F0) over a band of multipliers
of the measured Omega_0 (config.CHI_FREQ_BOUNDS). Three independent failure modes bracket those
choices:

  TOO SMALL F0  -> |chi| stops being REPRODUCIBLE. Same theta, different noise seed, different chi.
  TOO LARGE F0  -> the drive ENTRAINS the bundle. It abandons its own rhythm and follows the drive,
                   so chi reports the drive back to itself and the SHAPE information chi mode exists
                   to extract is gone.
  WRONG BAND    -> above ~0.25x Omega_0 |chi| is irreproducible at ANY amplitude or recording length
                   (measured 2026-08-05: high-multiplier CV does not fall from T_obs 5 s to 25 s, so
                   the variability is SYSTEMATIC, not statistical). At K=10 the retired (0.1, 10.0)
                   band put 8 of 10 probes there, which is what made posterior_chi_08042026
                   uninformative.

THE T_obs GATE (PRISM_HANDOFF.md section 4.1 step 1 / backlog C-1). The band and F0 were settled on a
SINGLE T_obs = 5 s slice, but training draws T ~ logU[T_MIN_EXP_S, T_MAX_EXP_S] and |chi| is strongly
T-dependent -- 8.03 / 7.73 / 6.97 / 1.16 across T = 1 / 2 / 5 / 20 s at FIXED theta. A band chosen on
one slice of an axis the training distribution sweeps is not gated. So T is now the OUTER loop, and
every criterion below must hold at EVERY T, not on average.

Faithfulness: each sample is probed at ITS OWN measured Omega_0 via chi.peak_freq, exactly as
pipeline.gen_chi_raw does, so the CV reported here includes the estimator jitter the network actually
sees -- not an idealized fixed-frequency CV.

THE FOUR CRITERIA, per (T_obs, multiplier, F0):
  |chi| CV     reproducibility across noise seeds at fixed theta        (<= CV_MAX)
  phase sd     CIRCULAR scatter of arg(chi) across seeds, radians       (<= PHASE_MAX)
  SNR          mean|chi_driven| / mean|chi_undriven| at the SAME probe frequency. The denominator is
               the lock-in FLOOR: what the estimator returns on a passive trace, i.e. spontaneous
               1/f content plus noise leakage. Free -- it reuses the passive ensemble. This is the
               ratio section 4.1 nominates to eventually replace CHI_MIN_CYCLES, so it is reported
               beside the drive-cycle count the production resolution_filter currently gates on.
  own peak     entrainment: driven power at the UNDRIVEN Omega_0, over its undriven level (>= SUP_MIN)

Env knobs:
  M          ensemble size (independent noise realizations per point)   (default 24)
  TOBS_GRID  observation durations in SECONDS       (default: log-spaced over the REACHABLE range,
                                                     see the training-ceiling note printed at start)
  MULTS      probe multipliers of Omega_0           (default: log-spaced across config.CHI_FREQ_BOUNDS
                                                     plus ONE above-band control at 2x the high edge)
  F0S        ND drive amplitudes                    (default: just config.CHI_F0 -- the amplitude
                                                     window was settled on 2026-08-05, so a bare run
                                                     is the T_obs gate. Pass a list to re-open it.)
  SEED       RNG seed                                                   (default 0)
  CV_MAX     reproducibility ceiling on |chi| CV                        (default 0.20)
  PHASE_MAX  ceiling on circular phase scatter, radians                 (default 0.50)
             ⚠ the one screen here with NO empirical basis. It was chosen, so a verdict that turns
             on it (today: whether 0.3x belongs in the band) is not evidence -- see backlog C-5
  SNR_MIN    floor on driven/undriven |chi|  -- PROPOSED, not settled   (default 3.0)
  SUP_MIN    entrainment floor: own peak must retain this fraction      (default 0.50)
  CYCLE_CAP  the prefix length shown in the per-point table and the plot. This is the control that
             tells a bad FREQUENCY from a bad DURATION                          (default 20)
  CYCLE_CAPS every prefix length to evaluate, for the WALL BRACKET. All of them run on the same
             traces, so this sweep costs lock-ins, not simulations -- never re-run the script per
             cap                                       (default 8 12 16 20 26 32 40 56, + CYCLE_CAP)
  PEAK_BW    half-bandwidth of the entrainment integral, as a fraction of Omega_0   (default 0.10)
  N_TOBS     points in the default T grid                               (default 5)
  N_MULTS    in-band points in the default multiplier grid              (default 4)

Run:
  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/chi_f0_sweep.py
"""
import math
import os
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt

import _common
import build_master_cells as bmc
from core import config
from core.config import PLOT_PATH
from core.SBI import chi as chi_mod

_common.enable_warnings()

M = int(os.environ.get("M", "24"))
SEED = int(os.environ.get("SEED", "0"))
CV_MAX = float(os.environ.get("CV_MAX", "0.20"))
# ADVISORY BY DEFAULT (inf), and that is a measured decision, not a loosening. C-5 swept 11
# multipliers through the band's high edge: under the duration cap, circular phase scatter grows
# SMOOTHLY from 0.13 to 1.52 rad with no knee anywhere (step ratios settle to ~1.15 per grid point),
# so any threshold on it is a choice reported back as a finding -- and the choice is worth a 2.5x
# difference in band width (0.5 rad would put the edge at 0.12x, entrainment puts it at 0.35-0.4x).
# The column is still printed and still worth reading: a probe whose phase is irreproducible has lost
# its cos/sin channels, so it is a HALF-USEFUL probe. It is not a corrupt one -- its |chi| CV stays
# ~0.08 and its SNR ~12 -- which is why it should not be a pass/fail gate. Entrainment is the
# criterion that discriminates, because it corrupts: a captured bundle reports the drive back to
# itself. Set PHASE_MAX to re-enable the screen.
PHASE_MAX = float(os.environ.get("PHASE_MAX", "inf"))
SNR_MIN = float(os.environ.get("SNR_MIN", "3.0"))
SUP_MIN = float(os.environ.get("SUP_MIN", "0.50"))
N_TOBS = int(os.environ.get("N_TOBS", "5"))          # points in the default T grid
N_MULTS = int(os.environ.get("N_MULTS", "4"))        # in-band points in the default multiplier grid
# Half-bandwidth of the "power at Omega_0" integral, as a FRACTION of Omega_0. Fractional, not a bin
# count: bin width is 1/T, so a fixed bin count integrates a physical bandwidth that SHRINKS as T
# grows, and the entrainment ratio would then drift across the very axis this script sweeps -- a
# T-dependence with no physics in it. (bmc._power_near's halfwidth_bins=2 default is fine at one T.)
PEAK_BW = float(os.environ.get("PEAK_BW", "0.10"))
# Lock in over a PREFIX spanning at most this many drive cycles, and report the result alongside the
# full-length one. Free -- the samples already exist, and it is exactly what gen_chi_raw's
# duration_frac does in production. It separates the two readings of a band that fails at long T:
#   "this FREQUENCY is unusable"  -> the capped column fails too, and the band must narrow;
#   "this DURATION is unusable"   -> the capped column recovers, and the fix is a per-probe duration
#                                    cap, which costs nothing and leaves the band intact.
# Those imply completely different changes, and the full-length column alone cannot tell them apart.
#
# EVERY cap in CYCLE_CAPS is evaluated on the SAME simulations -- a cap is just a shorter prefix of an
# already-existing trace, so bracketing the wall costs lock-ins, not simulations. Do not "sweep" this
# by re-running the script per cap: that is N x the simulation cost for the same numbers.
# Defaults to the ceiling ACTUALLY IN FORCE, so the capped column is what production measures rather
# than a literal that can drift away from it.
CYCLE_CAP = float(os.environ.get("CYCLE_CAP", config.CHI_MAX_CYCLES))

PLOT_NAME = "chi_tobs_gate.png"


class Capped(NamedTuple):
    """The same four metrics, re-measured over a shorter prefix of the same trace."""
    cv: float
    phase: float
    snr: float
    cycles: float      # cycles the prefix actually spans (== the full count if the trace is shorter)


class Point(NamedTuple):
    """One measured (T_obs, multiplier, F0) cell of the grid."""
    cv: float          # |chi| coefficient of variation across noise seeds
    phase: float       # circular sd of arg(chi), radians
    snr: float         # mean|chi_driven| / mean|chi_undriven| at the same probe frequency
    sup: float         # driven power at the UNDRIVEN Omega_0, over its undriven level
    cycles: float      # drive cycles the lock-in saw -- what CHI_MIN_CYCLES currently gates on
    by_cap: dict       # cap (cycles) -> Capped.  Always contains CYCLE_CAP.

    def cap(self, c=None):
        return self.by_cap[CYCLE_CAP if c is None else c]


def _grid(name, default):
    raw = os.environ.get(name)
    return default if not raw else [float(v) for v in raw.replace(",", " ").split()]


# Probe multipliers. Derived from the band ACTUALLY IN FORCE rather than hard-coded, so this script
# can never quietly test a band the pipeline no longer uses -- which is how the retired (0.1, 10.0)
# grid outlived the band it was written for. One above-band control keeps the "nothing above ~0.25x
# is recoverable" finding visible rather than assumed.
def _default_mults():
    lo, hi = config.CHI_FREQ_BOUNDS
    in_band = torch.exp(torch.linspace(math.log(lo), math.log(hi), max(2, N_MULTS), dtype=torch.float64))
    return [round(float(v), 4) for v in in_band] + [round(2.0 * hi, 4)]


MULTS = _grid("MULTS", _default_mults())
MULTS_ARE_DEFAULT = not os.environ.get("MULTS")
F0_GRID = _grid("F0S", [config.CHI_F0])
# Prefix lengths to bracket the reproducibility wall. Spans the two numbers on record -- 20 cycles is
# known to work and ~31 is the shortest observed failure -- with room either side, so the knee is
# inside the grid rather than at its edge. CYCLE_CAP is always included so the table, the plot and
# the sweep agree.
CYCLE_CAPS = sorted(set(_grid("CYCLE_CAPS", [8.0, 12.0, 16.0, 20.0, 26.0, 32.0, 40.0, 56.0])
                        + [CYCLE_CAP]))


def _reachable_t_seconds(cfg, hz):
    """(T_cap_s, n_fine_max, subsample) -- the longest recording TRAINING can actually draw here.

    gen_training_data pre-filters its Sobol (t_scale, T) draw on
    ``steady_idx + N_points * subsample <= min(N_ND_MAX, len(t))``. At this cell's t_scale that caps
    T well below T_MAX_EXP_S (measured: ~27 s against a nominal 60 s), and a T beyond the cap is a
    geometry NO training batch can contain. Sweeping past it would report a reproducibility number
    about a recording the network is never trained on -- the same class of mistake as fixing the band
    from one T slice, one level up.
    """
    t_scale = bmc.SHARED_RESCALE["t_scale"]
    subsample = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    n_fine_max = min(config.N_ND_MAX, len(cfg.t))
    n_obs_cap = max(1, (n_fine_max - cfg.steady_idx) // subsample)
    return n_obs_cap * cfg.dt_exp / hz, n_fine_max, subsample


def _default_tobs(t_cap_s):
    lo = config.T_MIN_EXP_S
    hi = min(config.T_MAX_EXP_S, t_cap_s)
    if hi <= lo:
        return [lo]
    pts = torch.exp(torch.linspace(math.log(lo), math.log(hi), max(2, N_TOBS), dtype=torch.float64))
    # FLOOR, never round: the top point sits exactly on the pre-filter ceiling, and rounding it up by
    # 0.01 s adds ~10 fine steps -- enough to put the longest recording just outside the training
    # distribution it is meant to represent. (The OOD flag in the T loop is what caught this.)
    return [math.floor(float(v) * 100) / 100 for v in pts]


def _peak_bins(freqs, f0):
    """PEAK_BW * f0 expressed in FFT bins at this T_obs -- see the PEAK_BW note above."""
    df = float(freqs[1] - freqs[0]) if freqs.numel() > 1 else 1.0
    return max(2, int(round(PEAK_BW * f0 / max(df, 1e-30))))


def _circ_std(z):
    """Circular standard deviation of arg(z) across the ensemble, in radians.

    CIRCULAR, not a plain std: arg(chi) is wrapped, so a linear std over an ensemble straddling +-pi
    reports ~1.8 rad of scatter for a perfectly concentrated phase. R is the mean resultant length;
    sqrt(-2 ln R) is its standard circular-dispersion form (0 = identical phases, large = uniform).
    """
    r = float((z / z.abs().clamp(min=1e-30)).mean().abs())
    return float(math.sqrt(max(0.0, -2.0 * math.log(max(r, 1e-30)))))


def _verdict(cv, phase_sd, snr, sup):
    bad = []
    if cv > CV_MAX:
        bad.append("noisy")
    if phase_sd > PHASE_MAX:
        bad.append("phase")
    if snr < SNR_MIN:
        bad.append("lowSNR")
    if sup < SUP_MIN:
        bad.append("entrained")
    return "OK" if not bad else " ".join(bad)


def _save_plot(tobs, mults, res, f0):
    """CV and SNR over T_obs x multiplier, at FULL length (top) and under the cycle cap (bottom).

    Both rows on one figure because the comparison IS the result: if the failure boundary runs along
    constant ``mult x T`` in the top row and vanishes in the bottom, the binding variable is the
    number of drive cycles integrated, not the probe frequency -- and the two call for opposite
    changes to CHI_FREQ_BOUNDS. Cycle counts are overlaid on the top row so the boundary can be read
    directly rather than inferred from the axes.
    """
    panels = (
        ([[res[(t, m, f0)].cv for m in mults] for t in tobs],
         f"|chi| CV, full length  (pass <= {CV_MAX:g})", "viridis_r", True),
        ([[res[(t, m, f0)].snr for m in mults] for t in tobs],
         f"SNR, full length  (pass >= {SNR_MIN:g})", "viridis", False),
        ([[res[(t, m, f0)].cap().cv for m in mults] for t in tobs],
         f"|chi| CV, capped at {CYCLE_CAP:g} cycles", "viridis_r", False),
        ([[res[(t, m, f0)].cap().snr for m in mults] for t in tobs],
         f"SNR, capped at {CYCLE_CAP:g} cycles", "viridis", False),
    )
    # ONE colour scale per metric across BOTH rows. Per-panel autoscaling would give the capped row
    # its own range, so its worst cell -- comfortably passing -- would render as dark as the
    # full-length row's genuine failures, and the figure would argue against its own result.
    cv_max = max(max(r) for g, _, _, _ in panels[::2] for r in g)
    snr_max = max(max(r) for g, _, _, _ in panels[1::2] for r in g)
    limits = (cv_max, snr_max, cv_max, snr_max)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    for ax, (grid, title, cmap, show_cycles), vmax in zip(axes.ravel(), panels, limits):
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(mults)), [f"{m:g}" for m in mults])
        ax.set_yticks(range(len(tobs)), [f"{t:g}" for t in tobs])
        ax.set_xlabel(r"probe frequency ($\times\,\Omega_0$)")
        ax.set_ylabel(r"$T_{\mathrm{obs}}$ (s)")
        ax.set_title(title, fontsize=10)
        for i, t in enumerate(tobs):
            for j, m in enumerate(mults):
                lbl = (f"{grid[i][j]:.2f}\n{res[(t, m, f0)].cycles:.0f}cyc" if show_cycles
                       else f"{grid[i][j]:.2f}")
                ax.text(j, i, lbl, ha="center", va="center", fontsize=6.5, color="w")
        fig.colorbar(im, ax=ax)
    lo, hi = config.CHI_FREQ_BOUNDS
    fig.suptitle(f"chi(omega) $T_{{obs}}$ gate -- $F_0$={f0:g} ND, M={M}, "
                 f"band in force = ({lo:g}, {hi:g})")
    out = PLOT_PATH / PLOT_NAME
    fig.savefig(str(out), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {out}")


def main():
    torch.manual_seed(SEED)
    cfg = bmc._provisional_cfg()
    hz = cfg.get_unit_conversion_factor("s")
    dtype, device = cfg.hw.dtype, cfg.hw.device
    f_scale = bmc.SHARED_RESCALE["f_scale"]

    t_cap_s, n_fine_max, subsample = _reachable_t_seconds(cfg, hz)
    TOBS = _grid("TOBS_GRID", _default_tobs(t_cap_s))

    print(f"[cfg] model={cfg.model} device={device} M={M} seed={SEED}")
    print(f"[cfg] criteria: |chi| CV <= {CV_MAX}, phase sd <= {PHASE_MAX} rad, "
          f"SNR >= {SNR_MIN}, own peak >= {SUP_MIN:.0%} of undriven")
    print(f"[cfg] band in force: CHI_FREQ_BOUNDS={config.CHI_FREQ_BOUNDS}  CHI_F0={config.CHI_F0:g}  "
          f"CHI_MIN_CYCLES={config.CHI_MIN_CYCLES:g}")
    print(f"[cfg] multipliers: {MULTS}"
          + ("   (derived from the band in force; last one is the ABOVE-BAND control)"
             if MULTS_ARE_DEFAULT else "   (from MULTS -- NOT derived from the band in force)"))
    print(f"[cfg] F0 grid:     {F0_GRID}" + ("" if len(F0_GRID) > 1 else "   (amplitude settled; T is the axis under test)"))
    print(f"[cfg] T_obs grid:  {TOBS} s")
    print(f"[cfg] training ceiling: at t_scale={bmc.SHARED_RESCALE['t_scale']:g} the Sobol pre-filter "
          f"(n_fine <= {n_fine_max}, subsample={subsample}) admits T_obs <= {t_cap_s:.2f} s, "
          f"NOT the nominal T_MAX_EXP_S={config.T_MAX_EXP_S:g} s.")
    print(f"[cfg] cost: {len(TOBS)} x (1 passive + {len(MULTS) * len(F0_GRID)} driven) = "
          f"{len(TOBS) * (1 + len(MULTS) * len(F0_GRID))} simulations\n", flush=True)

    res: dict[tuple, Point] = {}

    for T_s in TOBS:
        cfg.T_obs = T_s * hz
        geom = bmc._geometry(cfg)
        _, n_obs, n_fine, _ = geom
        achieved_s = n_obs * cfg.dt_exp / hz
        in_train = n_fine <= n_fine_max
        flag = ("IN training distribution" if in_train else
                f"*** OUT of the training distribution (n_fine={n_fine} > {n_fine_max}) ***")
        # More than ONE sample short means _geometry hit the pre-simulated ND grid and truncated;
        # less is just int() dropping a partial sample, which every path in the pipeline also does.
        clipped = "" if (T_s - achieved_s) <= 1.5 * cfg.dt_exp / hz else f"  *** CLIPPED from {T_s:g}s"
        print(f"=== T_obs = {achieved_s:.3f} s   n_obs={n_obs} n_fine={n_fine}{clipped}   {flag} ===",
              flush=True)

        # Passive reference. Three jobs: per-sample Omega_0 (what the probes are placed against), the
        # undriven power at Omega_0 (the entrainment denominator), and -- via a lock-in at each probe
        # frequency -- the NOISE FLOOR of the estimator itself. All three move with T.
        x_spont = bmc._simulate(cfg, geom, batch=M)
        f_peak = chi_mod.peak_freq(x_spont, cfg.dt_exp)                 # (M,) per-sample, as production
        freqs, power = bmc._psd(x_spont, cfg.dt_exp)
        f0_ens = float(freqs[int(torch.argmax(power[1:])) + 1])
        peak_bins = _peak_bins(freqs, f0_ens)
        p_spont = bmc._power_near(freqs, power, f0_ens, halfwidth_bins=peak_bins)
        T_lockin = n_obs * cfg.dt_exp
        nyq = 0.5 / cfg.dt_exp
        print(f"  Omega_0 = {f0_ens * hz:.4g} Hz (ensemble)   per-sample median "
              f"{float(f_peak.median()) * hz:.4g} Hz, spread {float(f_peak.std()) * hz:.3g} Hz")
        print(f"  peak band = +-{PEAK_BW:.0%} of Omega_0 = +-{peak_bins} bins at this T")
        print(f"  {'mult':>6} {'f(Hz)':>8} {'cycles':>7} {'F0(ND)':>7} {'|chi| CV':>9} "
              f"{'phase sd':>9} {'SNR':>8} {'own peak':>9} | {'cyc@cap':>7} {'CV@cap':>8} "
              f"{'SNR@cap':>8}  verdict", flush=True)

        for mult in MULTS:
            freq_b = torch.clamp(mult * f_peak, max=0.9 * nyq)          # (M,) per-sample, as gen_chi_raw
            omega_b = 2 * math.pi * freq_b.to(torch.float64)
            f_mean = float(freq_b.mean())
            cycles = f_mean * T_lockin
            masked = cycles < config.CHI_MIN_CYCLES
            # One prefix length per cap for the whole batch, as production's duration_frac does. Ceil
            # so a cap is never undershot into a sub-cycle window by rounding.
            def _prefix(c):
                n = min(n_obs, max(1, int(math.ceil(c / max(f_mean, 1e-30) / cfg.dt_exp))))
                return n, n * cfg.dt_exp
            for f0 in F0_GRID:
                amp_b = torch.full((M,), f0 * f_scale, dtype=dtype, device=device)
                # The floor: the SAME lock-in, the SAME nominal amplitude, on the UNDRIVEN ensemble.
                # Using the same amp_b makes the ratio a pure |chi| ratio (lock_in divides by F0).
                chi_floor = chi_mod.lock_in_batched(x_spont, omega_b, amp_b, T_lockin, cfg.dt_exp)
                x_f = bmc._simulate(cfg, geom, amp_dim=amp_b, freq_cell=freq_b,
                                    phase=math.pi / 2, batch=M)
                chi_v = chi_mod.lock_in_batched(x_f, omega_b, amp_b, T_lockin, cfg.dt_exp)
                mag = chi_v.abs()
                cv = float(mag.std() / mag.mean().clamp(min=1e-30))
                snr = float(mag.mean() / chi_floor.abs().mean().clamp(min=1e-30))
                phase_sd = _circ_std(chi_v)
                # Every cap, on the SAME two traces. A cap is a prefix, so this is lock-ins on
                # tensors that already exist -- bracketing the wall costs no extra simulation.
                by_cap = {}
                for c in CYCLE_CAPS:
                    n_c, T_c = _prefix(c)
                    fl_c = chi_mod.lock_in_batched(x_spont[:, :n_c], omega_b, amp_b, T_c, cfg.dt_exp)
                    dr_c = chi_mod.lock_in_batched(x_f[:, :n_c], omega_b, amp_b, T_c, cfg.dt_exp)
                    m_c = dr_c.abs()
                    by_cap[c] = Capped(
                        cv=float(m_c.std() / m_c.mean().clamp(min=1e-30)),
                        phase=_circ_std(dr_c),
                        snr=float(m_c.mean() / fl_c.abs().mean().clamp(min=1e-30)),
                        cycles=f_mean * T_c)
                fr, po = bmc._psd(x_f, cfg.dt_exp)
                # Entrainment is a property of the WHOLE driven trace's spectrum, not of the lock-in
                # window, so there is no capped counterpart -- `sup` applies to every column.
                sup = bmc._power_near(fr, po, f0_ens, halfwidth_bins=peak_bins) / max(p_spont, 1e-30)
                p = Point(cv=cv, phase=phase_sd, snr=snr, sup=sup, cycles=cycles, by_cap=by_cap)
                res[(T_s, mult, f0)] = p
                note = _verdict(cv, phase_sd, snr, sup)
                if masked:
                    note += f"  [masked: {cycles:.2f} < CHI_MIN_CYCLES]"
                pc = p.cap()
                print(f"  {mult:6g} {f_mean * hz:8.4g} {cycles:7.2f} {f0:7.3g} "
                      f"{cv:9.4g} {phase_sd:9.4g} {snr:8.4g} {sup:9.4g} | {pc.cycles:7.2f} "
                      f"{pc.cv:8.4g} {pc.snr:8.4g}  {note}", flush=True)
                del x_f
        del x_spont
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(flush=True)

    _band_verdict(TOBS, res)
    if len(F0_GRID) > 1:
        _f0_verdict(TOBS, res, f_scale)
    else:
        print(f"=== F0 window: not swept (F0S={F0_GRID[0]:g} only). Pass F0S=\"...\" to re-open the "
              f"amplitude question. ===\n")
    _cycles_verdict(res)
    _cap_verdict(TOBS, res)
    _wall_verdict(res)
    _edge_profile(TOBS, res)
    _save_plot(TOBS, MULTS, res, F0_GRID[0])


def _band_verdict(TOBS, res):
    """Which multipliers survive at EVERY T_obs -- the payload of the T gate.

    Reported TWICE: at full length, and under the CYCLE_CAP prefix. Only the second is a
    recommendation, because a duration cap is something the pipeline can actually do (gen_chi_raw's
    duration_frac) while "this frequency is unusable" is a permanent loss of span. Reporting only the
    full-length column is how a cycle-count limit gets read as a frequency limit -- which is exactly
    what happened when the band was fixed from one T_obs = 5 s slice.
    """
    print("=== BAND VERDICT: every criterion must hold at EVERY T_obs ===")
    print(f"  {'mult':>6}  {'worst CV':>9} {'at T':>6}   {'worst SNR':>10} {'at T':>6}   "
          f"{'worst phase':>11} {'at T':>6}   {'worst peak':>10} {'at T':>6}   full | @cap")
    passing, passing_cap = [], []
    for mult in MULTS:
        keys = [(t, mult, f0) for t in TOBS for f0 in F0_GRID]
        wc = max(keys, key=lambda k: res[k].cv)
        ws = min(keys, key=lambda k: res[k].snr)
        wp = max(keys, key=lambda k: res[k].phase)
        wu = min(keys, key=lambda k: res[k].sup)
        note = _verdict(res[wc].cv, res[wp].phase, res[ws].snr, res[wu].sup)
        # Same election over the capped columns. `sup` has no capped counterpart -- entrainment is a
        # spectral property of the driven trace, not of the lock-in window.
        note_cap = _verdict(max(res[k].cap().cv for k in keys), max(res[k].cap().phase for k in keys),
                            min(res[k].cap().snr for k in keys), res[wu].sup)
        if note == "OK":
            passing.append(mult)
        if note_cap == "OK":
            passing_cap.append(mult)
        print(f"  {mult:6g}  {res[wc].cv:9.4g} {wc[0]:6g}   {res[ws].snr:10.4g} {ws[0]:6g}   "
              f"{res[wp].phase:11.4g} {wp[0]:6g}   {res[wu].sup:10.4g} {wu[0]:6g}   "
              f"{'USABLE' if note == 'OK' else note} | "
              f"{'USABLE' if note_cap == 'OK' else note_cap}")

    lo_cfg, hi_cfg = config.CHI_FREQ_BOUNDS
    print()
    print(f"  Survives every T_obs at FULL length:        {passing or 'NONE'}")
    print(f"  Survives every T_obs under a {CYCLE_CAP:g}-cycle cap: {passing_cap or 'NONE'}")
    if len(passing_cap) > len(passing):
        print(f"  => the full-length column understates the usable band. A frequency is only unusable")
        print(f"     if it fails in BOTH columns; the rest is a duration problem. Recommend from the")
        print(f"     capped column and add the cap -- see the LOCK-IN DURATION CAP section below.")
    print()
    if not passing_cap:
        print("  !! NO multiplier satisfies every criterion at every T_obs, even capped.")
        print("     Options, in rough order of effort:")
        print("       * relax SNR_MIN / CV_MAX -- a noisier chi is still informative if the network")
        print("         sees the noise consistently, since it trains on the same distribution;")
        print("       * sweep CYCLE_CAP -- 20 is a guess, and the wall may sit lower;")
        print("       * make F0 PER-PROBE rather than one fixed ND amplitude.")
        return
    # A contiguous run is what a (lo, hi) band can express; a gap means the band cannot be stated as
    # an interval and the failure should be read, not averaged over.
    idx = [MULTS.index(m) for m in passing_cap]
    contiguous = idx == list(range(idx[0], idx[0] + len(idx)))
    lo_new, hi_new = min(passing_cap), max(passing_cap)
    if not contiguous:
        print(f"  !! NOT CONTIGUOUS -- the passing multipliers have a gap, so no single (lo, hi) band")
        print(f"     describes them. Read the table above rather than taking the range below.")
    print(f"  RECOMMENDED CHI_FREQ_BOUNDS = ({lo_new:g}, {hi_new:g})   [under a {CYCLE_CAP:g}-cycle cap]")
    print(f"  config.CHI_FREQ_BOUNDS is currently ({lo_cfg:g}, {hi_cfg:g})"
          f"{' -- CHANGE IT' if (abs(lo_new - lo_cfg) > 1e-9 or abs(hi_new - hi_cfg) > 1e-9) else ' -- already correct'}.")
    print(f"  The recommendation is bounded by the GRID, not by physics: the edges can only be the")
    print(f"  multipliers actually swept ({MULTS}). If an edge multiplier passes, the band may extend")
    print(f"  further -- re-run with MULTS reaching past it before concluding otherwise.")
    print(f"  CHI_K_PAD is frozen into every trained artifact, but the BAND is not -- changing it")
    print(f"  invalidates existing chi posteriors all the same (the sidecar records and checks it).\n")


def _f0_verdict(TOBS, res, f_scale):
    """The original amplitude window, now required to hold at every (T_obs, multiplier)."""
    print("=== usable F0 window (BOTH bounds, at EVERY T_obs and EVERY probe frequency) ===")
    print(f"  {'F0(ND)':>7}  {'worst CV':>9} {'at':>13}   {'worst peak':>11} {'at':>13}   verdict")
    viable = []
    for f0 in F0_GRID:
        keys = [(t, m, f0) for t in TOBS for m in MULTS]
        wc = max(keys, key=lambda k: res[k].cv)
        ws = min(keys, key=lambda k: res[k].sup)
        ok = res[wc].cv <= CV_MAX and res[ws].sup >= SUP_MIN
        if ok:
            viable.append(f0)
        why = []
        if res[wc].cv > CV_MAX:
            why.append(f"noisy at {wc[1]:g}x/{wc[0]:g}s")
        if res[ws].sup < SUP_MIN:
            why.append(f"entrains at {ws[1]:g}x/{ws[0]:g}s")
        print(f"  {f0:7.3g}  {res[wc].cv:9.4g} {f'{wc[1]:g}x/{wc[0]:g}s':>13}   "
              f"{res[ws].sup:11.4g} {f'{ws[1]:g}x/{ws[0]:g}s':>13}   "
              f"{'USABLE' if ok else ', '.join(why)}")
    print()
    if not viable:
        print("  !! NO F0 satisfies both criteria. Every amplitude that produces a reproducible |chi|")
        print("     also captures the oscillator, at some (T_obs, probe) in the grid.")
    else:
        best = max(viable)
        print(f"  RECOMMENDED CHI_F0 = {best:g} ND (= {best * f_scale:g} pN at this cell's f_scale). "
              f"Largest of {viable}:")
        print(f"  reproducibility improves with amplitude, so the top of the window is the best SNR")
        print(f"  that still leaves the bundle running free.")
        print(f"  config.CHI_F0 is currently {config.CHI_F0:g}"
              f"{' -- CHANGE IT' if abs(config.CHI_F0 - best) > 1e-9 else ' -- already correct'}.")
    print()


def _edge_profile(TOBS, res):
    """Where does the band's HIGH EDGE actually break? -- backlog C-5.

    The band verdict answers "does this multiplier cross the thresholds". That is the wrong question
    at the edge, because ``PHASE_MAX`` was CHOSEN, not measured: a verdict that turns on it reports
    the threshold back to itself. So this prints the CAPPED metrics as a monotone progression in
    multiplier and looks for a KNEE -- a place where a metric's behaviour changes character, which is
    a property of the cell rather than of a constant in this file.

    Read it as: if phase scatter and own-peak degrade SMOOTHLY across the whole range, no threshold
    here is defensible from this data and the edge has to be settled by training (C-5's expensive
    branch). If they turn sharply somewhere, that turn IS the edge and the thresholds should be moved
    to bracket it.
    """
    print("=== EDGE PROFILE: capped metrics vs probe frequency (backlog C-5) ===")
    print(f"  Worst across T_obs, all at the {CYCLE_CAP:g}-cycle cap. 'step' is the ratio to the row")
    print(f"  above -- a knee shows up there, not in the absolute values.")
    print(f"  {'mult':>7} {'CV':>8} {'step':>6} {'SNR':>8} {'step':>6} {'phase':>8} {'step':>6} "
          f"{'own peak':>9} {'step':>6}")
    prev = None
    for mult in MULTS:
        keys = [(t, mult, f0) for t in TOBS for f0 in F0_GRID]
        cv = max(res[k].cap().cv for k in keys)
        snr = min(res[k].cap().snr for k in keys)
        ph = max(res[k].cap().phase for k in keys)
        sup = min(res[k].sup for k in keys)
        cur = (cv, snr, ph, sup)
        if prev is None:
            print(f"  {mult:7g} {cv:8.4g} {'-':>6} {snr:8.4g} {'-':>6} {ph:8.4g} {'-':>6} "
                  f"{sup:9.4g} {'-':>6}")
        else:
            r = [c / p if abs(p) > 1e-30 else float('inf') for c, p in zip(cur, prev)]
            print(f"  {mult:7g} {cv:8.4g} {r[0]:6.2f} {snr:8.4g} {r[1]:6.2f} {ph:8.4g} {r[2]:6.2f} "
                  f"{sup:9.4g} {r[3]:6.2f}")
        prev = cur
    print()
    print(f"  Thresholds for reference: CV <= {CV_MAX:g}, SNR >= {SNR_MIN:g}, "
          f"phase <= {PHASE_MAX:g} rad, own peak >= {SUP_MIN:g}")
    print(f"  ⚠ PHASE_MAX and SNR_MIN were CHOSEN, not measured. Where a multiplier falls relative to")
    print(f"    them is not evidence; where the STEP column turns is. If no step turns sharply, this")
    print(f"    grid cannot settle the edge and C-5 needs the training comparison instead.\n")


def _wall_verdict(res):
    """WHERE is the reproducibility wall? -- the measurement C-4 needs before it can pick a cap.

    Every cap is evaluated on the same traces, so this is a prefix-length sweep, not a re-run. Two
    populations are reported per cap, and the distinction is the whole point:

      TRUNCATED  points the cap actually shortened. These are the evidence about the wall: if their
                 worst CV degrades as the cap grows, the cap is past it.
      UNTOUCHED  points whose full trace is already shorter than the cap. They cannot say anything
                 about a cap they never bound, and pooling them in would dilute exactly the signal
                 being looked for -- at a large cap most of the grid is untouched, and the average
                 would drift toward "fine" for a reason that has nothing to do with the cap.

    A cap is only meaningful where it BINDS, so the recommendation below comes from the truncated
    population and states how many points it rests on.
    """
    print("=== WALL BRACKET: worst metrics vs lock-in prefix length ===")
    print("  (truncated = points the cap actually shortened; the only ones it says anything about)")
    print(f"  {'cap':>6} {'n_trunc':>8} {'worst CV':>9} {'worst SNR':>10} {'worst phase':>12} "
          f"{'n_fail':>7}   verdict")
    rows = []
    for c in CYCLE_CAPS:
        trunc = [p for p in res.values() if p.cap(c).cycles < p.cycles - 1e-9]
        if not trunc:
            print(f"  {c:6g} {0:8d} {'-':>9} {'-':>10} {'-':>12} {'-':>7}   "
                  f"binds nothing in this grid")
            continue
        w_cv = max(p.cap(c).cv for p in trunc)
        w_snr = min(p.cap(c).snr for p in trunc)
        w_ph = max(p.cap(c).phase for p in trunc)
        n_fail = sum(1 for p in trunc
                     if _verdict(p.cap(c).cv, p.cap(c).phase, p.cap(c).snr, p.sup) != "OK")
        rows.append((c, len(trunc), w_cv, w_snr, w_ph, n_fail))
        print(f"  {c:6g} {len(trunc):8d} {w_cv:9.4g} {w_snr:10.4g} {w_ph:12.4g} {n_fail:7d}   "
              f"{'clean' if n_fail == 0 else f'{n_fail} point(s) fail'}")

    print()
    if not rows:
        print("  No cap in CYCLE_CAPS binds any point -- every trace is shorter than every cap.")
        print("  Extend TOBS_GRID upward or lower CYCLE_CAPS before reading anything into this.\n")
        return
    # The wall is the SMALLEST failing cap, not the largest clean one. Scanning from the top and
    # taking max(clean) would jump straight over an interior failure -- and these metrics are NOT
    # monotonic in the cap (a CV over M seeds is itself an estimate), so an interior failure with
    # clean caps above it is a normal outcome and a signal about precision, not about the wall.
    failing = [r[0] for r in rows if r[5] > 0]
    if not failing:
        print(f"  NO cap in the grid fails, so the wall is ABOVE {max(r[0] for r in rows):g} cycles and")
        print(f"  this grid does not bracket it. Raise CYCLE_CAPS (and TOBS_GRID, so the longer caps")
        print(f"  actually bind) before choosing CHI_MAX_CYCLES.\n")
        return
    first_fail = min(failing)
    below = [r for r in rows if r[0] < first_fail and r[5] == 0]
    clean_above = [(r[0], r[1]) for r in rows if r[5] == 0 and r[0] > first_fail]
    if not below:
        print(f"  !! The SMALLEST cap that binds ({first_fail:g} cycles) already fails, so this grid")
        print(f"     does not bracket the wall from below -- it may be lower still, or the failures")
        print(f"     may not be duration-driven at all. Lower CYCLE_CAPS, and check the LOCK-IN")
        print(f"     DURATION CAP table above for which points survive.\n")
        return
    best = max(below, key=lambda r: r[0])
    print(f"  First failing cap: {first_fail:g} cycles. Largest clean cap below it: {best[0]:g} "
          f"({best[1]} truncated points, worst CV {best[2]:.3g}, worst SNR {best[3]:.3g}).")
    print(f"  => the wall sits between {best[0]:g} and {first_fail:g} cycles. Set CHI_MAX_CYCLES at or")
    print(f"     below {best[0]:g}, with margin: this is ONE cell at ONE F0, and the estimate is only")
    print(f"     as fine as the CYCLE_CAPS grid.")
    if clean_above:
        print(f"  ⚠ NON-MONOTONIC: caps {[c for c, _ in clean_above]} are clean despite sitting ABOVE")
        print(f"    a failing one, on {[n for _, n in clean_above]} truncated point(s) respectively.")
        print(f"    A large cap BINDS FEWER POINTS -- only traces longer than it -- so the top of this")
        print(f"    table is thin evidence by construction, and each cell is still an M={M} CV")
        print(f"    estimate. Read the trend over the caps with many points, not the last row.")
    print()


def _cycles_verdict(res):
    """Is the drive-cycle count a good proxy for informativeness? -- the CHI_MIN_CYCLES question.

    Production masks a probe below config.CHI_MIN_CYCLES drive cycles. Section 4.1 nominates the
    driven/undriven ratio as the replacement gate. Both are measured on the same points here, so the
    comparison is a confusion count rather than an argument.
    """
    pts = [(p.cycles, p.snr) for p in res.values()]
    if not pts:
        return
    good = [c for c, s in pts if s >= SNR_MIN]
    bad = [c for c, s in pts if s < SNR_MIN]
    print(f"=== CHI_MIN_CYCLES cross-check ({len(pts)} points; informative := SNR >= {SNR_MIN:g}) ===")
    if not good:
        print(f"  No point in the grid reaches SNR {SNR_MIN:g}; the cycle gate cannot be calibrated here.")
        print(f"  Either SNR_MIN is too strict for this cell or the band is unusable -- read the band table.\n")
        return
    if not bad:
        print(f"  Every point reaches SNR {SNR_MIN:g}, over {min(c for c, _ in pts):.2f}"
              f"-{max(c for c, _ in pts):.2f} cycles. This grid does not constrain the gate.\n")
        return
    print(f"  informative points span {min(good):.2f}-{max(good):.2f} cycles; "
          f"uninformative span {min(bad):.2f}-{max(bad):.2f}")
    if min(good) <= max(bad):
        print(f"  OVERLAP: cycles alone cannot separate them, which is itself the finding -- "
              f"informativeness")
        print(f"  is not a function of cycle count on this cell, so a cycle threshold will always")
        print(f"  both over- and under-mask.")
    # Balanced accuracy over every candidate threshold sitting between two observed cycle counts.
    cands = sorted({c for c, _ in pts})
    best_c, best_score = None, -1.0
    for c in cands:
        tp = sum(1 for cc, s in pts if cc >= c and s >= SNR_MIN)
        fn = sum(1 for cc, s in pts if cc < c and s >= SNR_MIN)
        tn = sum(1 for cc, s in pts if cc < c and s < SNR_MIN)
        fp = sum(1 for cc, s in pts if cc >= c and s < SNR_MIN)
        score = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
        if score > best_score:
            best_c, best_score = c, score
    cur = config.CHI_MIN_CYCLES
    over = sum(1 for c, s in pts if c < cur and s >= SNR_MIN)     # masked but informative
    under = sum(1 for c, s in pts if c >= cur and s < SNR_MIN)    # kept but uninformative
    print(f"  best separating threshold: {best_c:.2f} cycles (balanced accuracy {best_score:.2f})")
    print(f"  CHI_MIN_CYCLES = {cur:g} masks {over} informative point(s) and keeps "
          f"{under} uninformative one(s).")
    print(f"  Masking an informative probe is a lost recording, not a wrong answer; KEEPING an")
    print(f"  uninformative one feeds the flow a row that carries no susceptibility. The second is")
    print(f"  the expensive error, and it is the one a cycle gate cannot see.\n")


def _cap_verdict(TOBS, res):
    """Does capping the lock-in duration rescue the probes that fail at full length?

    This is the fork. A band that fails only at long T_obs has two readings, and they call for
    opposite changes: narrow CHI_FREQ_BOUNDS (losing the frequency span chi(omega) exists to measure)
    or cap each probe's lock-in duration (free -- gen_chi_raw already takes duration_frac, and
    training already jitters it). The capped columns settle it on the SAME simulations.
    """
    # The SAME four criteria _band_verdict elects on, or the two blocks disagree about one point.
    # `sup` has no capped counterpart on purpose: entrainment is a spectral property of the driven
    # trace, so an entrained probe is correctly NOT rescuable by shortening the lock-in.
    failed = [(k, p) for k, p in res.items() if _verdict(p.cv, p.phase, p.snr, p.sup) != "OK"]
    print(f"=== LOCK-IN DURATION CAP: does {CYCLE_CAP:g} cycles rescue the failures? ===")
    if not failed:
        print(f"  Nothing failed at full length, so there is nothing to rescue.\n")
        return

    def _capped(p):
        """The cap actually shortened this point AND the shortened lock-in passes."""
        return (p.cap().cycles < p.cycles - 1e-9
                and _verdict(p.cap().cv, p.cap().phase, p.cap().snr, p.sup) == "OK")

    rescued = [(k, p) for k, p in failed if _capped(p)]
    print(f"  {'T(s)':>6} {'mult':>7} {'cycles':>8} {'CV':>8} {'SNR':>8}  ->  "
          f"{'cyc@cap':>7} {'CV@cap':>8} {'SNR@cap':>8}   ")
    for k, p in sorted(failed):
        tag = ("RESCUED" if _capped(p) else
               "no cap applied" if p.cap().cycles >= p.cycles - 1e-9 else "still fails")
        print(f"  {k[0]:6g} {k[1]:7g} {p.cycles:8.2f} {p.cv:8.4g} {p.snr:8.4g}  ->  "
              f"{p.cap().cycles:7.2f} {p.cap().cv:8.4g} {p.cap().snr:8.4g}   {tag}")
    n_capable = sum(1 for _, p in failed if p.cap().cycles < p.cycles - 1e-9)
    print()
    if n_capable and len(rescued) == n_capable:
        print(f"  ALL {len(rescued)} over-long failures RECOVER under a {CYCLE_CAP:g}-cycle cap.")
        print(f"  => the binding variable is DURATION, not frequency. Do NOT narrow CHI_FREQ_BOUNDS;")
        print(f"     cap each probe's lock-in instead. gen_chi_raw already accepts duration_frac and")
        print(f"     training already jitters it, so this is a ceiling on that draw, not new machinery.")
        print(f"     The experimental path needs the same ceiling, or a bench recording longer than")
        print(f"     the training cap becomes an input the network was never trained on.")
    elif rescued:
        print(f"  MIXED: {len(rescued)} of {n_capable} over-long failures recover under the cap.")
        print(f"  Read the table -- a duration cap helps but does not fully explain the band's edge,")
        print(f"  so the two effects are superposed and both need a decision.")
    else:
        print(f"  NONE recover. The failure is a property of the probe FREQUENCY, not of how long it")
        print(f"  was integrated, so a duration cap buys nothing and the band itself must move.")
    print()


if __name__ == "__main__":
    main()
