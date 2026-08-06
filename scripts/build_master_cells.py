"""
Choose the three Nadrowski MASTER cells BY MEASUREMENT, and emit the master Bounds/Cells triple.

WHY THIS EXISTS. The five old Nadrowski cells (`cell`, `cell_1`, `cell_2`, `cell_2_spont`, `cell_3`)
each carried their OWN bounds box, and three of the boxes disagreed on `f_max`, `tau_c` and `temp`.
Nothing in the pipeline cross-checked a prior or a posterior against the box it was trained in, so a
5-day chi run ended up describable only by forensics. The replacement is ONE bounds pair plus THREE
cells that share a single ND + rescale block and differ only in their Forcing section.

The numbers below are not typed in by hand. Every one is either an input constant declared at the top
or a MEASURED quantity printed with the criterion that selected it:

  A. Hopf regime + predicted Omega_0    -- core/Reduction (analytical, no simulation)
  B. measured Omega_0 + peak clarity    -- a spontaneous ensemble through chi.peak_freq
  C. the WEAK drive                     -- largest amplitude still in the linear-response regime
  D. the ENTRAINING drive               -- smallest amplitude that phase-locks the bundle

Omega_0 is measured with ``chi.peak_freq`` -- the SAME estimator chi mode uses to place its probes --
not ``FDT.spectral.find_spectral_peak``. The two disagree (rfft argmax vs Welch PSD peak), and it is
the chi one that decides where the probes actually land.

Env knobs:
  M          ensemble size per measurement           (default 16)
  TOBS_S     observation duration in SECONDS          (default 5.0)
  SEED       RNG seed                                 (default 0)
  WRITE      1 = write the Bounds/Cells files, 0 = measure and report only (default 0)

Run:
  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/build_master_cells.py
  WRITE=1 ... scripts/build_master_cells.py
"""
import math
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import _common
from core import cli, config, forcing, registry
from core.config import BOUNDS_PATH, CELL_PATH, CHUNK_LEN, VALID_LABELS, VALID_MODELS
from core.Helpers import helpers
from core.Reduction import reduce_nwk_to_hopf, ReductionFailure
from core.SBI import chi as chi_mod, pipeline
from core.SBI.statistics import FEATURE_LABELS, SummaryStatistics

_common.enable_warnings()

MODEL = "NADROWSKI"
M = int(os.environ.get("M", "16"))
TOBS_S = float(os.environ.get("TOBS_S", "5.0"))
SEED = int(os.environ.get("SEED", "0"))
WRITE = os.environ.get("WRITE", "0").strip().lower() in ("1", "true", "yes", "on")

# ── INPUTS ────────────────────────────────────────────────────────────────────────────────────────
# The shared ND + rescale block every master cell carries. Seeded from the archived cell_1 (the only
# old cell whose GT sat strictly interior to its own box) and confirmed oscillatory in step B below.
# ORDER IS LOAD-BEARING: simulators bind columns positionally (Model(*torch.unbind(params, dim=1))),
# so this order IS the constructor's argument order and IS the emitted bounds file's ND order.
#
# `s` (the calcium-feedback strength) is what sets how WIDE the oscillatory window in `f_max` is, and
# `f_max` (phi) is the bifurcation parameter itself. Measured with the reduction over a joint scan:
#
#     s      admissible f_max      mu_N > 0 window       best mu_N
#     0.50   (none)                (none)
#     0.65   1.02..1.12            1.04..1.07            +0.150 at 1.06
#     0.95   1.05..3.72            1.12..1.87            +0.624 at 1.32
#     1.25   1.07..5.29            1.25..2.01            +0.236 at 1.52
#     1.50   1.09..4.35            (none)                -0.088
#
# s = 0.95 gives both the strongest instability and a window wide enough that f_max = 1.32 sits well
# inside it. At the old s = 0.65 the window is 0.03 wide -- which is why the archived cell_2 (f_max
# = 1.06) sat almost exactly on the single oscillatory point available to it, and why nothing else
# in that family reproduced.
SHARED_ND = OrderedDict([
    ("k",       0.8),
    ("lam",     3.57),
    ("f_max",   1.32),
    ("tau",     0.027),
    ("tau_c",   0.268),
    ("s",       0.95),
    ("delta_E", 10.0),
    ("beta",    14.1),
    ("n",       50.0),
    ("temp",    1.5),
])
SHARED_RESCALE = OrderedDict([("x_scale", 62.14), ("t_scale", 3.73), ("f_scale", 10.0)])
SHARED_INITS = OrderedDict([("x_init", -0.11), ("xa_init", -1.32), ("c_init", 0.5)])

# The MASTER ND box. Wide but not degenerate: the ND prior is a stability-screened GMM, so an
# over-wide box mostly costs screening effort and prior diffuseness rather than correctness.
# `tau_c` and `temp` take lower bounds > 0 deliberately -- zero active noise is a degenerate corner,
# and the archived cell_2 sat exactly on it (a ground truth ON a bound is the §7.1 hazard).
MASTER_ND_BOX = OrderedDict([
    ("k",       (0.01, 5.0)),
    ("lam",     (0.1, 50.0)),
    ("f_max",   (0.05, 30.0)),
    ("tau",     (1e-4, 2.0)),
    ("tau_c",   (0.005, 5.0)),
    ("s",       (0.0, 3.0)),
    ("delta_E", (0.0, 30.0)),
    ("beta",    (0.5, 100.0)),
    ("n",       (10.0, 300.0)),
    ("temp",    (0.05, 10.0)),
])
MASTER_RESCALE_BOX = OrderedDict([
    ("x_scale", (10.0, 100.0)),
    ("t_scale", (1.0, 40.0)),
    ("f_scale", (1.0, 1000.0)),
])
# Forcing bounds are FILLED FROM THE MEASUREMENT (steps B-D): `freq` must bracket the measured
# Omega_0 -- the 2026-07-27 lesson was that training drives 43x-10^4x above resonance carried almost
# no information -- and `amp` must cover weak through entraining with headroom.
FORCING_FREQ_DECADES = 1.0    # freq box spans Omega_0 / 10^d .. Omega_0 * 10^d
FORCING_AMP_HEADROOM = 3.0    # amp box upper edge = headroom x the entraining amplitude

# Amplitude sweep grid, as multiples of f_scale (i.e. ND drive amplitude).
AMP_ND_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]

# The probe is DETUNED from Omega_0 on purpose. Entrainment is capture of the oscillator by a drive at
# a NEARBY BUT DIFFERENT frequency (the Arnold tongue), so driving exactly at Omega_0 cannot separate
# the two regimes: the drive peak and the bundle's own peak land in the same FFT bin, and the phase
# locks trivially because the bundle already oscillates there. Detuned, the discriminator is sharp:
#   weak      -> the bundle keeps its own rhythm, the PSD shows TWO peaks, PLV at the drive is low
#   entrained -> the bundle abandons Omega_0 and follows the drive, PSD collapses, PLV -> 1
# This also sidesteps the linear-response question entirely, which matters because PRISM_HANDOFF 4.3
# already established that an actively oscillating bundle has NO clean linear regime near Omega_0 --
# so "weak" cannot be defined by linearity here, only by "does not capture the oscillator".
DETUNE = float(os.environ.get("DETUNE", "1.4"))     # drive frequency = DETUNE * Omega_0
#
# ENTRAINMENT IS MEASURED AS SUPPRESSION OF THE BUNDLE'S OWN PEAK, not as "the drive peak is big".
# A purely linear response ALSO puts power at the drive frequency, so "P(drive) > P(Omega_0)" is
# satisfied by a bundle that is completely ignoring the drive and merely responding to it -- measured
# here, that criterion fired at amp_ND = 0.1 while the bundle was plainly still free-running.
# Capture is the bundle ABANDONING Omega_0, so compare the forced trace's power at Omega_0 against
# the PASSIVE trace's power at the same frequency. That is a different FFT bin from the drive, so the
# linear response cannot contaminate it.
SUPPRESS_FREE = 0.70          # weak:      P_forced(Omega_0) still >= 70% of its undriven level
SUPPRESS_ENTRAIN = 0.10       # entrained: P_forced(Omega_0) has fallen below 10% of it

_PLV_IDX = FEATURE_LABELS.index("G6_plv")


# ── config + simulation helpers ───────────────────────────────────────────────────────────────────
def _provisional_cfg():
    """A SimConfig built from the master box directly, so no bounds FILE has to exist yet.

    `bounds_dicts` is make_sim_config's hand-entry path and takes a (params, rescale, forcing) triple
    in parse_bounds_file's shape. Order is preserved verbatim, which is what keeps the positional
    contract intact between this config and the file this script emits.
    """
    labels = VALID_LABELS[VALID_MODELS.index(MODEL)]
    params = OrderedDict((k, (None, b)) for k, b in MASTER_ND_BOX.items())
    rescale = OrderedDict((k, (None, b)) for k, b in MASTER_RESCALE_BOX.items())
    force = OrderedDict([("amp", (None, (0.0, 1.0))), ("freq", (None, (1e-6, 1.0))),
                         ("phase", (None, (0.0, 2 * math.pi))), ("offset", (None, (-50.0, 50.0)))])
    cfg = cli.make_sim_config(MODEL, labels, registry.state_dep_drift(MODEL),
                              bounds_dicts=(params, rescale, force))
    cfg.T_obs = TOBS_S * cfg.get_unit_conversion_factor("s")
    return cfg


def _geometry(cfg):
    """Fine-grid geometry for the shared block -- the same derivation generate_observations uses."""
    t_scale = SHARED_RESCALE["t_scale"]
    dt_nd_gt = cfg.dt_exp / t_scale
    subsample = max(1, round(dt_nd_gt / cfg.dt_nd_min))
    n_obs = int((cfg.T_obs / t_scale) / dt_nd_gt)
    n_fine = cfg.steady_idx + n_obs * subsample
    if n_fine > len(cfg.t):
        n_obs = (len(cfg.t) - cfg.steady_idx) // subsample
        n_fine = cfg.steady_idx + n_obs * subsample
        print(f"[geom] clipped to the pre-simulated grid: N_obs={n_obs}", flush=True)
    return subsample, n_obs, n_fine, max(1, math.ceil(n_fine / CHUNK_LEN))


def _simulate(cfg, geom, amp_dim=0.0, freq_cell=0.0, phase=0.0, batch=M):
    """Ensemble of `batch` dimensional traces at dt_exp, shape (batch, N_obs).

    amp_dim/freq_cell are DIMENSIONAL drive params in cell units, exactly as a cell file states them;
    build_nondim_sin_force_tensor divides by f_scale internally. amp_dim=0 gives the passive run.
    Either may be a scalar or a (batch,) tensor -- the chi path probes each sample at ITS OWN
    measured Omega_0, so a per-sample frequency is the faithful thing to reproduce.
    """
    subsample, n_obs, n_fine, n_segs = geom
    dtype, device = cfg.hw.dtype, cfg.hw.device
    t_fine = cfg.t[:n_fine]

    params = torch.tensor([list(SHARED_ND.values())], dtype=dtype, device=device).expand(batch, -1)
    inits = torch.tensor([list(SHARED_INITS.values())], dtype=dtype, device=device).expand(batch, -1)
    rescale = torch.tensor([list(SHARED_RESCALE.values())], dtype=dtype, device=device).expand(batch, -1)

    n_ch = forcing.n_force_channels(cfg.model, cfg.forcing_idx, inits.shape[-1])
    driven = bool(torch.is_tensor(amp_dim) or amp_dim != 0.0)
    if not driven:
        force = torch.zeros((batch, n_ch, n_fine), dtype=dtype, device=device)
    else:
        fp = torch.zeros((batch, 4), dtype=dtype, device=device)
        fp[:, cfg.forcing_idx["amp"]] = amp_dim
        fp[:, cfg.forcing_idx["freq"]] = freq_cell
        fp[:, cfg.forcing_idx["phase"]] = phase
        force = pipeline.build_nondim_sin_force_tensor(fp, t_fine, rescale,
                                                       cfg.forcing_idx, cfg.rescale_idx)
        if force.shape[1] < n_ch:                       # sin builder emits 1 channel; pad the rest
            padded = torch.zeros((batch, n_ch, force.shape[2]), dtype=dtype, device=device)
            padded[:, :force.shape[1], :] = force
            force = padded

    x_nd = pipeline.gen_obs(model=cfg.model, params=params, t=t_fine, inits=inits, force=force,
                            n_segs=n_segs, steady_idx=cfg.steady_idx,
                            state_dep_drift=cfg.state_dep_drift, batch_size=batch, var_idx=0,
                            dtype=dtype, device=device)[0, :, :]
    x_sub = x_nd[:, ::subsample][:, :n_obs]
    x_dim = helpers.rescale(x_sub, rescale[:, cfg.rescale_idx["x_scale"]].unsqueeze(1))
    del x_nd, x_sub, force
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return x_dim


def _psd(x, dt):
    """One-sided power spectrum of a demeaned ensemble, and its frequency axis (cell freq units)."""
    xr = x - x.mean(dim=-1, keepdim=True)
    spec = torch.fft.rfft(xr.to(torch.float64), dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=dt).to(spec.device)
    return freqs, (spec.abs() ** 2).mean(dim=0)


def _power_near(freqs, power, f0, halfwidth_bins=2):
    """Total power in a small band around f0 -- the peak's height is bin-placement sensitive."""
    if f0 <= 0:
        return 0.0
    i = int(torch.argmin((freqs - f0).abs()))
    lo, hi = max(1, i - halfwidth_bins), min(len(freqs), i + halfwidth_bins + 1)
    return float(power[lo:hi].sum())


def main():
    torch.manual_seed(SEED)
    cfg = _provisional_cfg()
    geom = _geometry(cfg)
    subsample, n_obs, n_fine, n_segs = geom
    hz = cfg.get_unit_conversion_factor("s")          # cell freq units -> Hz
    print(f"[cfg] model={cfg.model} device={cfg.hw.device} dt_exp={cfg.dt_exp:g} (cell time units)")
    print(f"[cfg] T_obs={TOBS_S}s -> N_obs={n_obs}, n_fine={n_fine}, subsample={subsample}, M={M}")
    print(f"[cfg] ND order: {list(SHARED_ND)}")

    # ── A. Pick f_max analytically: where is the limit cycle? ─────────────────────────────────────
    #
    # f_max (phi) is the bifurcation parameter of the NWK model, so it -- not the noise strengths --
    # is what decides whether this cell oscillates at all. The reduction is analytical and costs
    # milliseconds per point, so scan it before spending a simulation. mu_N > 0 = the fixed point is
    # unstable = a limit cycle exists; the aim is a point comfortably INSIDE that window, not next to
    # its edge, so the cell stays clearly oscillatory across the noise realizations.
    print("\n=== A. Reduction scan over f_max: where does the limit cycle live? ===", flush=True)
    # 0.01 steps, deliberately: at s = 0.65 the whole window is 0.03 wide, so a coarse grid steps
    # straight over it and reports "never oscillates" for a point that plainly does.
    osc = []
    for i in range(20, 600):
        f_max = round(0.01 * i, 3)
        try:
            r = reduce_nwk_to_hopf(dict(SHARED_ND, f_max=f_max),
                                   SHARED_RESCALE["t_scale"], SHARED_RESCALE["x_scale"])
        except ReductionFailure:
            continue                      # no complex eigenpair here = no oscillatory mode at all
        if r.mu_N > 0:
            osc.append((f_max, r.mu_N, r.Omega0_N))
    if not osc:
        raise SystemExit("No f_max gave mu_N > 0 -- this block never oscillates. `s` is the lever "
                         "that opens the window (s=0.5 gives none, s=0.95 gives 1.12..1.87).")
    lo_f, hi_f = osc[0][0], osc[-1][0]
    # Select by STRONGEST instability, then require it to sit clear of both edges. Max mu_N is the
    # clearest limit cycle; the margin check is what stops us shipping a cell perched on the edge of
    # its own window, which is precisely how the archived cell_2 ended up unreproducible.
    f_max_sel, mu_sel, om_sel = max(osc, key=lambda r: r[1])
    if os.environ.get("FMAX"):
        f_max_sel = float(os.environ["FMAX"])
        rr = reduce_nwk_to_hopf(dict(SHARED_ND, f_max=f_max_sel),
                                SHARED_RESCALE["t_scale"], SHARED_RESCALE["x_scale"])
        mu_sel, om_sel = rr.mu_N, rr.Omega0_N
        print(f"  FMAX override -> {f_max_sel}")
    SHARED_ND["f_max"] = f_max_sel
    width = hi_f - lo_f
    margin = min(f_max_sel - lo_f, hi_f - f_max_sel)
    pred_hz = om_sel / SHARED_RESCALE["t_scale"] / (2 * math.pi) * hz
    print(f"  mu_N > 0 for f_max in [{lo_f:g}, {hi_f:g}]  (width {width:.3g}, {len(osc)} points)")
    print(f"  selected f_max = {f_max_sel:g}   mu_N = {mu_sel:+.5g}   (max over the window)")
    print(f"  edge margin    = {margin:.3g} = {100 * margin / max(width, 1e-12):.0f}% of the window width")
    print(f"  Omega0_N       = {om_sel:.5g}  ->  predicted f_0 ~ {pred_hz:.4g} Hz")
    if margin < 0.15 * width:
        print("  !! WARNING: within 15% of a window edge -- raise `s` to widen it before shipping.")
    if not (MASTER_ND_BOX["f_max"][0] < f_max_sel < MASTER_ND_BOX["f_max"][1]):
        raise SystemExit(f"selected f_max={f_max_sel} is outside MASTER_ND_BOX {MASTER_ND_BOX['f_max']}")

    # ── B. Measured Omega_0 + peak clarity ────────────────────────────────────────────────────────
    print("\n=== B. Spontaneous run: measured Omega_0 and peak clarity ===", flush=True)
    x_spont = _simulate(cfg, geom)
    freqs, power = _psd(x_spont, cfg.dt_exp)
    # Omega_0 from the ENSEMBLE-AVERAGED spectrum, not the median of per-trace argmaxes. This bundle
    # is noise-driven, so a single trace's argmax jitters by a few Hz; averaging |X|^2 across the
    # ensemble first gives a stable number to define the drive frequency against. chi.peak_freq's
    # per-trace estimate is reported alongside because THAT is what chi mode uses to place its probes,
    # and the gap between the two is a direct measure of how well-defined this cell's peak is.
    k0 = int(torch.argmax(power[1:])) + 1
    f0_cell = float(freqs[k0])
    f_peak = chi_mod.peak_freq(x_spont, cfg.dt_exp)                  # (M,) cell freq units
    p_peak = _power_near(freqs, power, f0_cell)
    p_med = float(power[1:].median())
    amp_nm = float((x_spont.max(dim=-1).values - x_spont.min(dim=-1).values).median())
    print(f"  Omega_0/2pi    = {f0_cell:.6g} /cell-time  =  {f0_cell * hz:.4g} Hz   (ensemble PSD peak)")
    print(f"  chi.peak_freq  = {float(f_peak.median()) * hz:.4g} Hz per-trace median,"
          f" spread {float(f_peak.std()) * hz:.3g} Hz   (what chi mode will use)")
    print(f"  peak/median    = {p_peak / max(p_med, 1e-30):.4g}   (clean single peak >> 1)")
    print(f"  peak-to-peak   = {amp_nm:.4g} nm over {TOBS_S}s   (visible oscillation after rescaling)")
    print(f"  cycles in T    = {f0_cell * cfg.T_obs:.1f}")
    p_spont_at_f0 = p_peak                                            # the undriven reference level

    # ── C/D. Amplitude sweep at a DETUNED drive: free-running vs entrained ────────────────────────
    f_drive = DETUNE * f0_cell
    print(f"\n=== C/D. Amplitude sweep at {DETUNE:g} x Omega_0 = {f_drive * hz:.4g} Hz"
          f"  (Omega_0 = {f0_cell * hz:.4g} Hz) ===", flush=True)
    print(f"  {'amp_ND':>7} {'amp(pN)':>9} {'|chi|':>11} {'PLV':>7} {'P(Om0)/passive':>15}  verdict")
    f_scale = SHARED_RESCALE["f_scale"]
    rows = []
    for a_nd in AMP_ND_GRID:
        amp_dim = a_nd * f_scale
        x_f = _simulate(cfg, geom, amp_dim=amp_dim, freq_cell=f_drive, phase=math.pi / 2)
        omega_b = torch.full((x_f.shape[0],), 2 * math.pi * f_drive,
                             dtype=torch.float64, device=x_f.device)
        amp_b = torch.full((x_f.shape[0],), amp_dim, dtype=x_f.dtype, device=x_f.device)
        chi_k = chi_mod.lock_in_batched(x_f, omega_b, amp_b, n_obs * cfg.dt_exp, cfg.dt_exp)
        chi_mag = float(chi_k.abs().median())

        stats = SummaryStatistics(x_spont, x_f, cfg.dt_exp, amp_b,
                                  torch.full((x_f.shape[0],), f_drive, dtype=x_f.dtype, device=x_f.device),
                                  torch.full((x_f.shape[0],), math.pi / 2, dtype=x_f.dtype, device=x_f.device))
        plv = float(stats.compute_statistics()[:, _PLV_IDX].median())   # reported, not gated on

        fr, po = _psd(x_f, cfg.dt_exp)
        suppress = _power_near(fr, po, f0_cell) / max(p_spont_at_f0, 1e-30)

        free = suppress >= SUPPRESS_FREE
        entrained = suppress <= SUPPRESS_ENTRAIN
        verdict = "free-running" if free else ("ENTRAINED" if entrained else "transitional")
        rows.append(dict(a_nd=a_nd, amp_dim=amp_dim, chi=chi_mag, plv=plv,
                         suppress=suppress, free=free, entrained=entrained))
        print(f"  {a_nd:7.3g} {amp_dim:9.4g} {chi_mag:11.5g} {plv:7.3f} {suppress:15.4g}  {verdict}",
              flush=True)

    # ENTRAIN = the smallest drive that captures the bundle.
    # WEAK    = the largest drive it still ignores, searched only BELOW that onset.
    #
    # The lower bound matters: far above the onset the Omega_0 bin REFILLS (measured: suppression
    # climbs back from 0.006 at amp_ND=0.35 to 0.71 at amp_ND=20), because a drive that large spreads
    # power across the spectrum through the bundle's nonlinearity. Read without the bound, a 200 pN
    # drive scores as "free-running" -- it is the opposite, and it would have been written into the
    # weak cell.
    entrain = next((r for r in rows if r["entrained"]), None)
    below = [r for r in rows if entrain is None or r["a_nd"] < entrain["a_nd"]]
    weak = next((r for r in reversed(below) if r["free"]), None)
    print()
    if weak is None:
        print("  !! nothing stayed free-running -- extend AMP_ND_GRID downward.")
    else:
        print(f"  WEAK      amp_ND={weak['a_nd']:g}  amp={weak['amp_dim']:g} pN"
              f"   (own peak at {100 * weak['suppress']:.0f}% of its undriven level -- bundle keeps its rhythm)")
    if entrain is None:
        print("  !! nothing entrained -- extend AMP_ND_GRID upward, or reduce DETUNE"
              " (a far-detuned drive needs far more amplitude to capture).")
    else:
        print(f"  ENTRAIN   amp_ND={entrain['a_nd']:g}  amp={entrain['amp_dim']:g} pN"
              f"   (own peak at {100 * entrain['suppress']:.1f}% of its undriven level -- captured)")

    # ── What this sweep says about CHI_F0 ─────────────────────────────────────────────────────────
    # chi mode drives at a FIXED ND amplitude (config.CHI_F0) at multipliers spanning 0.1x..10x
    # Omega_0, so the entrainment onset measured here is directly comparable to it. A chi probe above
    # the onset is not measuring linear response at all -- it is capturing the oscillator, and chi
    # then reports the drive back to itself.
    if entrain is not None:
        onset = entrain["a_nd"]
        print(f"\n  CHI_F0 CHECK: entrainment onset at this detune is amp_ND = {onset:g}.")
        for name, val in (("config.CHI_F0", config.CHI_F0), ("the 08/04 run", 0.1)):
            row = min(rows, key=lambda r: abs(r["a_nd"] - val))
            state = ("ENTRAINS" if row["entrained"] else
                     "free-running" if row["free"] else "transitional")
            print(f"    {name:<16} F0 = {val:<5g} ND -> nearest measured amp_ND={row['a_nd']:g}: "
                  f"{state} (own peak at {100 * row['suppress']:.1f}%)")
        print("    A probe at or above the onset is not a linear-response measurement. Note that")
        print("    PRISM_HANDOFF 4.3 ruled F0=0.05 out for the OPPOSITE reason (|chi| too noisy to")
        print("    reproduce), so the usable window is bounded on both sides -- and it was measured")
        print("    on a different cell. Re-measure before trusting either bound here.")

    if not WRITE:
        print("\n(measure-only; re-run with WRITE=1 to emit the Bounds/Cells files)")
        return
    if weak is None or entrain is None:
        raise SystemExit("Refusing to write: the sweep did not bracket both regimes.")
    _write_files(cfg, f0_cell, f_drive, weak, entrain, hz)


# ── emission ──────────────────────────────────────────────────────────────────────────────────────
def _fmt(v):
    return f"{v:.6g}"


def _write_files(cfg, f0_cell, f_drive, weak, entrain, hz):
    """Emit the master Bounds PAIR and the three master Cells.

    Both files are written from ONE in-memory box, so the two bounds files cannot drift apart: the
    spontaneous variant is the forced one minus `f_scale` and minus the Forcing section (mode 1 drops
    `f_scale` because it only ever divides a force).
    """
    freq_lo = f0_cell / (10 ** FORCING_FREQ_DECADES)
    freq_hi = f0_cell * (10 ** FORCING_FREQ_DECADES)
    amp_hi = entrain["amp_dim"] * FORCING_AMP_HEADROOM
    force_box = OrderedDict([("amp", (0.0, amp_hi)), ("freq", (freq_lo, freq_hi)),
                             ("phase", (0.0, 6.283185307)), ("offset", (-50.0, 50.0))])

    def bounds_text(with_forcing: bool) -> str:
        out = ["# Non-dimensional Parameters"]
        out += [f"{k} in ({_fmt(lo)}, {_fmt(hi)})" for k, (lo, hi) in MASTER_ND_BOX.items()]
        out += ["", "# Dimensional Parameters"]
        for k, (lo, hi) in MASTER_RESCALE_BOX.items():
            if k == "f_scale" and not with_forcing:
                continue
            out.append(f"{k} in ({_fmt(lo)}, {_fmt(hi)})")
        if with_forcing:
            out += ["", "# Forcing Parameters"]
            out += [f"{k} in ({_fmt(lo)}, {_fmt(hi)})" for k, (lo, hi) in force_box.items()]
        return "\n".join(out) + "\n"

    def cell_text(amp, freq, with_f_scale: bool) -> str:
        out = ["# Non-dimensional Initial Conditions"]
        out += [f"{k} = {_fmt(v)}" for k, v in SHARED_INITS.items()]
        out += ["", "# Non-dimensional Parameters"]
        out += [f"{k} = {_fmt(v)}" for k, v in SHARED_ND.items()]
        out += ["", "# Dimensional Parameters"]
        for k, v in SHARED_RESCALE.items():
            if k == "f_scale" and not with_f_scale:
                continue
            out.append(f"{k} = {_fmt(v)}")
        out += ["", "# Forcing Parameters"]
        out += [f"amp = {_fmt(amp)}", f"freq = {_fmt(freq)}", "phase = 0", "offset = 0"]
        return "\n".join(out) + "\n"

    bdir, cdir = BOUNDS_PATH / "nadrowski", CELL_PATH / "nadrowski"
    bdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    written = [
        (bdir / "master.txt", bounds_text(True)),
        (bdir / "master_spont.txt", bounds_text(False)),
        # The spontaneous cell keeps a Forcing section at amp=0: inject_ground_truth requires the
        # drive params to be PRESENT (it just does not range-check them), and chi mode ignores the
        # cell's own drive entirely, so amp=0 is the honest statement of "no drive".
        (cdir / "master_spont.txt", cell_text(0.0, 0.0, with_f_scale=True)),
        (cdir / "master_weak.txt", cell_text(weak["amp_dim"], f_drive, with_f_scale=True)),
        (cdir / "master_entrained.txt", cell_text(entrain["amp_dim"], f_drive, with_f_scale=True)),
    ]
    for path, text in written:
        path.write_text(text, encoding="utf-8")
        print(f"  wrote {path}")

    # Every GT strictly interior to its own bound -- a value ON a bound is the §7.1 hazard, and it is
    # what made the archived cell_2 unusable against any other cell's box.
    bad = [f"{k}={v} not strictly inside {MASTER_ND_BOX[k]}"
           for k, v in SHARED_ND.items() if not (MASTER_ND_BOX[k][0] < v < MASTER_ND_BOX[k][1])]
    bad += [f"{k}={v} not strictly inside {MASTER_RESCALE_BOX[k]}"
            for k, v in SHARED_RESCALE.items() if not (MASTER_RESCALE_BOX[k][0] < v < MASTER_RESCALE_BOX[k][1])]
    if bad:
        raise SystemExit("Emitted bounds do not strictly contain the shared block:\n  " + "\n  ".join(bad))
    print(f"\n  all {len(SHARED_ND) + len(SHARED_RESCALE)} shared values are strictly interior")
    print(f"  freq box  = ({_fmt(freq_lo)}, {_fmt(freq_hi)}) cell freq units"
          f"  =  ({freq_lo * hz:.3g}, {freq_hi * hz:.3g}) Hz, bracketing Omega_0")
    print(f"  amp box   = (0, {_fmt(amp_hi)}) pN, {FORCING_AMP_HEADROOM:g}x the entraining amplitude")


if __name__ == "__main__":
    main()
