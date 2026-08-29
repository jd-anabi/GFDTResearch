"""
Degeneracy / sloppiness map over ALL 16 inferred params.

Generalizes the archived diagnose_fmax.py Part B. At the cell-file ground truth it builds the
standardized feature-Jacobian

    J[j, p] = d<feature_j> / d(param_p)  /  noise_std_j

(CRN central differences of the ensemble-mean 41 features; noise_std_j = single-trajectory
feature std at GT). J is in signal-to-noise units -> an identifiability map. Then:
  - pairwise |cos| (degenerate pairs), SVD spectrum (sloppy directions), unique-handle frac.

ROBUSTNESS:
  - per-member validity filter (finite + |x| < adaptive CAP); features over valid members,
  - ONE-SIDED difference when a perturbation side destabilizes (kept, flagged); a column is
    'unmeasurable' only if BOTH sides destabilize,
  - t_scale handled correctly (its perturbation re-derives subsample / fine-grid length),
  - redimensionalization + features in float64 (so tiny offset perturbations don't underflow),
  - SVD / sloppiest-direction computed over STIFF columns (measurable, ||g|| > tol) so zero
    columns can't dominate.

Env: CELL BOUNDS MODEL TOBS_S CHI CHI_K CHI_F0 CHI_LO CHI_HI M M_NOISE REL SEED MIN_VALID ZERO_TOL
     NOISE_EPS

Run:  TEE IT. Every table below is stdout-only except the .npz, and the [cfg]/[mode]/[noise]/probe
      banners plus the OOD warnings _common.enable_warnings() deliberately un-suppresses are NOT in
      the .npz. Use *>&1 (all streams), not 2>&1 -- PS 5.1 wraps a native exe's stderr lines in
      NativeCommandError records and sets $? false even on exit 0.

      $py = "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe"
      $env:TOBS_S=4.5; $env:CHI=0; $env:CELL="Resources/Cells/nadrowski/master_weak.txt"
      & $py scripts/degeneracy_map.py *>&1 | Tee-Object Resources/Plots/degeneracy_forced_T4.5.log

Outputs are suffixed by MODE **and T_obs** -- both, because a map at another T_obs is another
measurement, not a redo (see MODE_TAG). Compare only equal-T runs.

Diff the two modes by LABEL, never by index -- forced has 41 feature rows and chi has 30+3K, so the
.npz arrays are not element-wise comparable:

      a = np.load("Resources/Plots/degeneracy_map_forced_T4.5.npz")
      b = np.load("Resources/Plots/degeneracy_map_chi_T4.5.npz")
      top = lambda z, p: [z["feat_labels"][i] for i in np.argsort(-np.abs(z["J"][:, p]))[:5]]
      for p, nm in enumerate(a["param_names"]):
          print(f"{nm:11s} forced {top(a, p)}"); print(f"{'':11s} chi    {top(b, p)}")
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt

import _common
from core import forcing
from core.config import CHI_MIN_CYCLES, CHUNK_LEN, PLOT_PATH
from core.SBI import chi as chi_mod, pipeline

_common.enable_warnings()

M = int(os.environ.get("M", "32"))
M_NOISE = int(os.environ.get("M_NOISE", "128"))
REL = float(os.environ.get("REL", "0.02"))
SEED = int(os.environ.get("SEED", "0"))
MIN_VALID = float(os.environ.get("MIN_VALID", "0.5"))
ZERO_TOL = float(os.environ.get("ZERO_TOL", "0.05"))     # ||g||_std below this = "no local info"
NOISE_EPS = float(os.environ.get("NOISE_EPS", "1e-6"))   # fnoise/|feature| below this = dead channel
SF, SS, SC = 1, 2, 3                                     # CRN seeds: forced / spontaneous / chi block
torch.manual_seed(SEED)
print(f"[cfg] M={M} M_NOISE={M_NOISE} REL={REL} MIN_VALID={MIN_VALID} ZERO_TOL={ZERO_TOL} "
      f"NOISE_EPS={NOISE_EPS:g} SEED={SEED}", flush=True)

cfg = _common.script_cfg()
_common.assert_nadrowski(cfg, "the printed ND parameter names are Nadrowski's")
dtype, device = cfg.hw.dtype, cfg.hw.device
N_obs = int(cfg.T_obs / cfg.dt_exp)                       # physical length (t_scale-independent)

gt_nd = cfg.params_tensor[0].clone()
gt_rescale = torch.tensor([v for v, _ in cfg.rescale_params.values()], dtype=dtype, device=device)

# ---- Which information set is this map about? ----------------------------------------------------
# The Jacobian MUST be built from the features the posterior actually conditions on, or the answer is
# about a different experiment than the one being run.
#
#   forced : [S(41) incl. Group G from a single-tone drive at the cell's own frequency]
#   chi    : [S(41) with Group G ZEROED | chi(3K)]  -- so Group G's 11 columns are dropped and 3K
#            chi columns take their place. The FISHER set is (log|chi|, cos, sin), see
#            chi.CHI_FISHER_CHANNELS -- `u`, `mask` and `logcyc` are all excluded
#            because a standardized Jacobian divides by an ensemble std, so a barely-varying channel
#            is an amplifier. The conditioning block is a
#            different, WIDER thing (6 per slot over CHI_K_PAD slots, adding `u` and `mask`) and is
#            not what a Jacobian may be built over.
#
# Left unchanged, this script reported the FORCED Jacobian regardless of the chi toggle: it kept the
# 11 Group-G columns chi mode zeroes and omitted the chi columns entirely. Its |cos| pairs were
# therefore literally independent of chi, so it would have reported kappa~x_scale and lambda~t_scale
# as strong as ever and falsely refuted the whole chi hypothesis it was being used to test.
_G_MASK = _common.summary_keep_idx()
FEAT_LABELS = _common.feature_labels(cfg)
_N_FORCE_CH = forcing.n_force_channels(cfg.model, cfg.forcing_idx, cfg.inits_tensor.shape[-1])
_common.describe_features(cfg)

if cfg.chi_mode:
    _MULTS = chi_mod.chi_multipliers_for(cfg)
    print(f"[mode] probe multipliers of Omega_0: {[round(v, 4) for v in _MULTS.tolist()]}", flush=True)
else:
    # chi probes at its own frequencies and ignores the cell's drive, so only THIS branch needs one.
    _common.assert_forced(cfg, "degeneracy_map in forced mode")
    forcing_gt = torch.tensor([[v for v, _ in cfg.force_params_dict.values()]],
                              dtype=dtype, device=device)
    amp_v = forcing_gt[:, cfg.forcing_idx["amp"]]
    freq_v = forcing_gt[:, cfg.forcing_idx["freq"]]
    phase_v = forcing_gt[:, cfg.forcing_idx["phase"]]

# Suffixes every output. MODE alone is not enough: `J` is in signal-to-noise units and `fnoise` falls
# ~1/sqrt(T), so the SAME mode at two T_obs is two different measurements -- and the forced arm has to
# be run at a second T_obs on purpose (the Group-G non-stationarity control). Tagged by
# mode alone, that control silently overwrote the primary forced result it exists to be compared
# against. Exactly the defect the mode suffix was added to fix, on an axis added later.
MODE_TAG = f"{cfg.observation_mode}_T{cfg.T_obs / cfg.get_unit_conversion_factor('s'):g}"
ND_NAMES = list(cfg.params_dict.keys())
RESCALE_NAMES = list(cfg.rescale_params.keys())
nd_bounds = [b for _, b in cfg.params_dict.values()]


def _raw(pvec, rescale_vec, m, crn):
    """(feats (m, n_feat) float64 np, x_for_validity, xs_dim). Grid re-derived from rescale_vec."""
    t_scale = float(rescale_vec[cfg.rescale_idx["t_scale"]])
    subs = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    n_fine = min(cfg.steady_idx + N_obs * subs, len(cfg.t))
    t_fine = cfg.t[:n_fine]
    n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))
    p = pvec.unsqueeze(0).expand(m, -1).contiguous()
    rv = rescale_vec.unsqueeze(0).expand(m, -1).contiguous()
    inits_m = cfg.inits_tensor.expand(m, -1).contiguous()

    def sim(f):
        return pipeline.gen_obs(model=cfg.model, params=p, t=t_fine,
                                inits=inits_m, force=f,
                                n_segs=n_segs, steady_idx=cfg.steady_idx,
                                state_dep_drift=cfg.state_dep_drift, batch_size=m, dtype=dtype,
                                device=device)[0][:, ::subs][:, :N_obs]

    xsc = rescale_vec[cfg.rescale_idx["x_scale"]].double()
    xof = rescale_vec[cfg.rescale_idx["x_offset"]].double() if "x_offset" in cfg.rescale_idx else 0.0

    # fork_rng so the fixed CRN seeds below do not leak out and pin the caller's global RNG (the
    # same defect that was fixed in SBI/decorrelate.feats).
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        if cfg.chi_mode:
            zero = torch.zeros((m, _N_FORCE_CH, t_fine.shape[0]), dtype=dtype, device=device)
            if crn:
                torch.manual_seed(SS)
            xs_d = xsc * sim(zero).double() + xof
            # SEED AGAIN, right here. gen_chi_block runs K MORE simulations whose noise is otherwise
            # completely unseeded -- so the +delta and -delta arms of the central difference would
            # see different chi noise realisations and the derivative would be swamped. Omitting this
            # produces a plausible-looking, meaningless map.
            if crn:
                torch.manual_seed(SC)
            # The FISHER feature set (3 channels per probe), not the conditioning block: this builds a
            # Jacobian, and `fnoise = max(std, 1e-9)` is a DENOMINATOR, so any barely-varying channel
            # is an amplifier. `u` is theta-independent under a fixed multiplier grid (log(mult_k)
            # plus ~2.5e-8 of float32 rounding -- twenty-five times ABOVE the 1e-9 clamp, so the clamp
            # protects nothing) and `logcyc` is either an exact duplicate of A3_log_fpeak or, where
            # the ceiling binds, floor() quantization. Neither is in CHI_FISHER_CHANNELS any more.
            # resolution_filter=False for the same reason the rotation uses it: a probe crossing the
            # cycle threshold between arms is a step of 1 over that same floor.
            #
            # ALL FOUR NAMED, NOTHING RE-SLICED. gen_chi_raw returns (chi, u, logcyc, valid). This
            # call read `[:2]`, which bound `logcyc_v = u` -- handing fisher_features, under the name
            # `logcyc`, the exact channel the paragraph above exists to keep out. The trap was
            # documented here and then walked into six lines below the documentation, from 5d7e965
            # until it was measured: at TOBS_S=4.5 the channels came out as log(0.03)..log(0.30), all
            # NEGATIVE, against a correct log(cycles) of +1.12..+3.00. `fisher_features` now takes ONE
            # argument, which makes that whole class of mistake a TypeError rather than a rebind --
            # but keep naming every element here anyway, because the next tuple change will not be
            # protected by that signature.
            chi_v, _u_v, _logcyc_v, valid_v = pipeline.gen_chi_raw(
                model=cfg.model, params_nd=p, rescale=rv, x_spont_dim=xs_d.to(dtype),
                t_fine=t_fine, inits=inits_m, rescale_idx=cfg.rescale_idx, n_segs=n_segs,
                steady_idx=cfg.steady_idx, subsample=subs, N_points=N_obs, dt_exp=cfg.dt_exp,
                multipliers=_MULTS, f0_nd=cfg.chi_f0, state_dep_drift=cfg.state_dep_drift,
                # Ceiling ON, filter OFF -- see the note at the same call in SBI/decorrelate.feats.
                # Without it this measures the information in a lock-in longer than any the network
                # will ever be given, which is a different experiment from the one being mapped.
                max_cycles=cfg.chi_max_cycles, resolution_filter=False,
                dtype=dtype, device=device)
            # `valid` is not consulted by fisher_features, so a probe that failed the finite /
            # positive / sub-0.9-Nyquist screen would be packed into J as though it were a
            # measurement. With resolution_filter OFF that screen is the only one left, and on this
            # script's fixed grid nothing should ever trip it -- so if it does, the labels no longer
            # describe the feature vector and every table below is void. Refuse rather than report.
            if not bool(valid_v.all()):
                raise SystemExit(
                    f"chi: {int((~valid_v).sum())} of {valid_v.numel()} probe measurements failed "
                    f"gen_chi_raw's finite / positive / 0.9x-Nyquist screen. fisher_features does "
                    f"not consult `valid`, so those entries would enter the Jacobian as if measured.")
            chi_block = chi_mod.fisher_features(chi_v)
            spont = pipeline.gen_stats_features(xs_d, None, cfg.dt_exp, None, None, None,
                                       device=device, spontaneous_only=True).numpy()
            feats = np.concatenate([spont[:, _G_MASK], chi_block.double().cpu().numpy()], axis=1)
            return feats, xs_d, xs_d

        force = pipeline.build_nondim_sin_force_tensor(forcing_gt.expand(m, -1), t_fine, rv,
                                                       cfg.forcing_idx, cfg.rescale_idx)
        if crn:
            torch.manual_seed(SF)
        xf = sim(force)
        if crn:
            torch.manual_seed(SS)
        xs = sim(torch.zeros_like(force))
        xf_d, xs_d = xsc * xf.double() + xof, xsc * xs.double() + xof   # float64 redim
        feats = pipeline.gen_stats_features(xs_d, xf_d, cfg.dt_exp, amp_v.expand(m).double(),
                                   freq_v.expand(m).double(), phase_v.expand(m).double(),
                                   device=device).numpy()
        return feats, xf_d, xs_d


def _valid(xf_d, xs_d):
    fin = torch.isfinite(xf_d).all(1) & torch.isfinite(xs_d).all(1)
    mag = (xf_d.abs().amax(1) < CAP) & (xs_d.abs().amax(1) < CAP)
    return (fin & mag).cpu().numpy()


# ---- GT noise ensemble: adaptive CAP + single-traj feature noise floor ----
feats0, xf0, xs0 = _raw(gt_nd, gt_rescale, M_NOISE, crn=False)
fin0 = (torch.isfinite(xf0).all(1) & torch.isfinite(xs0).all(1)).cpu().numpy()
amax0 = torch.maximum(xf0.abs().amax(1), xs0.abs().amax(1)).cpu().numpy()
CAP = 100.0 * float(np.median(amax0[fin0]))
keep0 = fin0 & (amax0 < CAP)
fnoise = np.maximum(feats0[keep0].std(0), 1e-9)
print(f"[noise] CAP={CAP:.4g}  GT valid frac={keep0.mean():.2f}  median feature noise={np.median(fnoise):.4g}", flush=True)

# ---- PROBE BUDGET: does each probe MEASURE anything at this T_obs? --------------------------------
# Runs BEFORE the dead-channel test because it feeds it: which probes are PINNED is arithmetic, and
# arithmetic sees a failure mode no statistic can (below).
#
# resolution_filter is OFF here -- mandatory, see the call site -- which means NOTHING masks a probe
# that saw less than CHI_MIN_CYCLES drive cycles. A sub-cycle lock-in returns the demeaned trace's
# residual drift: finite, in range, REPRODUCIBLE (healthy std, healthy-looking gradient) and not a
# susceptibility. No noise-based guard below can see it, because there is nothing anomalous about its
# statistics -- only about its physics. At the other end a probe over CHI_MAX_CYCLES has its segment
# truncated to exactly that many cycles, which PINS its `logcyc` to log(max_cycles) up to the floor()
# quantization of N_row -- see the pinning note in the dead-channel block for why that is worse than
# it sounds.
#
# BOTH edges cannot be cleared with margin at once, and that is arithmetic rather than a tuning
# failure: the band's dynamic range (hi/lo = 10 at the configured band) EQUALS the cycle window's
# (CHI_MAX_CYCLES/CHI_MIN_CYCLES = 10), so exactly one T_obs puts the low edge on the floor and the
# high edge on the ceiling simultaneously. chi.resolvable_multipliers records the same collision from
# the training side. Err ABOVE it: under the floor a probe is not a measurement AT ALL, whereas over
# the ceiling it is simply integrated over fewer cycles and comes back noisier.
#
# Since 2026-08-10 the ceiling is no longer a CORRECTNESS issue for this map, only a precision one.
# It used to pin `logcyc` to log(max_cycles) plus floor() quantization, which a 1e-9-floored fnoise
# amplified into the largest entry in the whole Jacobian (max|J| = 2.0e4 against 289 for the biggest
# real feature -- it set sigma[0] and a condition number of 56000 single-handedly). `logcyc` has since
# left CHI_FISHER_CHANNELS entirely, and the surviving channels -- log|chi|, cos, sin --
# are genuine measurements over a shorter window. So PINNED below is now INFORMATIONAL. Keep printing
# it: it is what tells you the top of the band is being measured over 20 cycles rather than 30.
if cfg.chi_mode:
    hz = cfg.get_unit_conversion_factor("s")                       # cell freq units -> Hz
    f_pk = chi_mod.peak_freq(xs0.to(dtype), cfg.dt_exp).cpu().numpy()[keep0]
    f0_gt = float(np.median(f_pk))
    T_full = N_obs * cfg.dt_exp
    n_sp = len(_G_MASK)
    print(f"\n=== probe budget: T_obs={cfg.T_obs:g} cell-time = {cfg.T_obs / hz:g} s, "
          f"Omega_0={f0_gt * hz:.4g} Hz (ensemble median; p5..p95 "
          f"{np.percentile(f_pk, 5) * hz:.3g}..{np.percentile(f_pk, 95) * hz:.3g}) ===")
    _nch = len(chi_mod.CHI_FISHER_CHANNELS)
    print(f"  {'xOmega_0':>9} {'f (Hz)':>9} {'cycles':>8} {'floor':>7} {'ceiling':>8} "
          f"{'mean log|chi|':>14} {'cos^2+sin^2':>12}")
    _bad = 0
    for _j, _mv in enumerate(_MULTS.tolist()):
        _cyc = _mv * f0_gt * T_full
        # A CHANNEL-IDENTITY check, and it is here because the previous one had to be retired: this
        # used to compare a predicted log(cycles) against the 4th Fisher channel, which is what caught
        # `[:2]` binding `u` into `logcyc`. `logcyc` has since left the Fisher set, so that column no
        # longer exists -- but the class of bug (a caller feeding the wrong channel into
        # fisher_features) deserves a standing check, not just the one-argument signature.
        #
        # `cos^2 + sin^2 == 1` is the strongest invariant available and it costs nothing: they are the
        # cosine and sine of one angle, so any mis-wiring that puts something else in either slot
        # breaks it. It is exactly the identity the packer uses to tell a real probe from a phantom
        # (see chi.pack_probe_block). log|chi| is printed beside it as a magnitude sanity read.
        _lm = float(feats0[keep0][:, n_sp + _nch * _j + 0].mean())
        _c = feats0[keep0][:, n_sp + _nch * _j + 1]
        _s = feats0[keep0][:, n_sp + _nch * _j + 2]
        _unit = float(np.mean(_c ** 2 + _s ** 2))
        _bad += abs(_unit - 1.0) > 1e-3
        print(f"  {_mv:9.4f} {_mv * f0_gt * hz:9.4g} {_cyc:8.2f} "
              f"{'ok' if _cyc >= CHI_MIN_CYCLES else 'DRIFT':>7} "
              f"{'PINNED' if _cyc > cfg.chi_max_cycles else 'ok':>8} {_lm:14.4g} {_unit:12.6f}")
    if _bad:
        raise SystemExit(
            f"chi: {_bad} probe(s) violate cos^2 + sin^2 == 1, so channels 1 and 2 of the Fisher "
            f"block are not the cosine and sine of one phase. Check what is being passed to "
            f"chi.fisher_features and the gen_chi_raw unpack above (trap CHI10).")
    _lo_b, _hi_b = cfg.chi_freq_bounds
    print(f"  low edge {_lo_b:g}x clears {CHI_MIN_CYCLES:g} cycles at T_obs >= "
          f"{CHI_MIN_CYCLES / (_lo_b * f0_gt) / hz:.3g} s; high edge {_hi_b:g}x stays under the "
          f"{cfg.chi_max_cycles:g}-cycle ceiling below T_obs = "
          f"{cfg.chi_max_cycles / (_hi_b * f0_gt) / hz:.3g} s.")
    print(f"  adapt_placement (OFF here, see the gen_chi_raw call) would be a NO-OP iff the first of "
          f"those two numbers is <= this T_obs of {cfg.T_obs / hz:g} s.", flush=True)

# ---- DEAD FEATURE CHANNELS -----------------------------------------------------------------------
# fnoise is a DENOMINATOR. A feature row with no ensemble spread does not become a zero row of J -- it
# becomes whatever float32 rounding survived the central difference divided by whatever float32
# rounding survived the ensemble std, and that ratio is order 1 to 50: the magnitude of a real
# standardized feature. It then leads ||g||, the |cos| matrix, the SVD and the top-features table with
# nothing in the numbers to mark it as an artifact. This is not hypothetical -- it is what the `[:2]`
# unpack at the gen_chi_raw call below produced for four years' worth of K columns (`u`, magnitude
# ~3.5, ensemble std ~2.5e-8), and it is NOT specific to that bug: any channel the pipeline happens to
# pin arrives the same way.
#
# THE TEST IS RELATIVE, and that is the load-bearing choice. `1e-9` cannot be the criterion even
# though the clamp on the line above is 1e-9: `u`'s std is 2.5e-8, twenty-five times ABOVE the clamp,
# so an absolute test at the clamp would have missed the very bug that motivated this block. Against
# the channel's own magnitude, NOISE_EPS = 1e-6 is ~8 float32 ulps -- a spread that small is
# representation noise whatever the channel means.
#
# DROPPED, not fatal, and the pattern is ZERO_TOL's: that removes uninformative COLUMNS so they cannot
# dominate the SVD; this is the identical statement about ROWS. Fatal would be wrong -- the run costs
# hours and a benignly constant summary feature in one mode should not abort the comparison. Rows are
# ZEROED in J rather than deleted so every index still lines up with FEAT_LABELS: a zero row
# contributes nothing to a norm, a cosine, a singular value or an argsort, so no consumer below has to
# know this happened.
# This detector finds a channel pinned at float32 resolution -- `u` (ratio ~1e-8) is the worked
# example. It is kept as a STANDING guard rather than a fix for any current channel: after logcyc left the Fisher set
# no member of CHI_FISHER_CHANNELS pins, so on a healthy run it prints "no dead channels" and that
# line is the evidence, not decoration.
#
# It deliberately does NOT try to catch quantization-pinning, and the reason is worth keeping: the
# ceiling-pinned `logcyc` that used to sit here came in at ratio 2.7e-5, twenty-seven times ABOVE
# NOISE_EPS -- this gate passed it, and it still produced max|J| = 2.0e4 against 289 for the largest
# real feature. Raising NOISE_EPS until it caught would be threshold-guessing that starts eating
# quiet real features instead. That failure was fixed at the source by removing the channel, which is
# the right shape of fix: a statistic cannot separate quantization from a genuinely quiet feature, so
# do not ask it to.
fscale = np.maximum(np.abs(feats0[keep0]).max(0), 1e-30)
dead = fnoise <= NOISE_EPS * fscale
if dead.any():
    print(f"\n!! DEAD FEATURE CHANNELS: {int(dead.sum())}/{len(fnoise)} rows ZEROED in J -- their "
          f"standardized entries would be amplified representation or quantization noise, not signal.")
    for i in np.flatnonzero(dead):
        print(f"     {FEAT_LABELS[i]:18s} std={fnoise[i]:9.3g}  |feat|={fscale[i]:9.3g}  "
              f"ratio={fnoise[i] / fscale[i]:.2g}")
else:
    print(f"[noise] no dead channels (min std/|feat| = {float((fnoise / fscale).min()):.2g} vs "
          f"NOISE_EPS={NOISE_EPS:g})", flush=True)


def feats_valid(pvec, rescale_vec, m):
    feats, xf_d, xs_d = _raw(pvec, rescale_vec, m, crn=True)
    v = _valid(xf_d, xs_d)
    return (feats[v].mean(0) if v.any() else np.full(feats.shape[1], np.nan)), float(v.mean())


def grad(perturb, base, d):
    fp, vp = feats_valid(*perturb(+d))
    fm, vm = feats_valid(*perturb(-d))
    if vp >= MIN_VALID and vm >= MIN_VALID:
        return (fp - fm) / (2 * d) / fnoise, min(vp, vm), "central"
    f0, _ = feats_valid(*base)
    if vp >= MIN_VALID:
        return (fp - f0) / d / fnoise, vp, "1-sided+"
    if vm >= MIN_VALID:
        return (f0 - fm) / d / fnoise, vm, "1-sided-"
    return np.full_like(fnoise, np.nan), max(vp, vm), "UNMEAS"


cols, names, vfr, kinds = [], [], [], []
for i, nm in enumerate(ND_NAMES):
    lo, hi = nd_bounds[i]
    d = max(REL * (hi - lo), 1e-5 * abs(float(gt_nd[i])))
    g, vf, kind = grad(lambda dd, _i=i: (gt_nd.clone().index_put_((torch.tensor([_i], device=device),),
                       (gt_nd[_i] + dd).reshape(1)), gt_rescale, M),
                       (gt_nd, gt_rescale, M), d)
    cols.append(g); names.append(nm); vfr.append(vf); kinds.append(kind)
for nm in RESCALE_NAMES:
    r = cfg.rescale_idx[nm]; lo, hi = cfg.rescale_params[nm][1]
    d = max(REL * (hi - lo), 1e-5 * abs(float(gt_rescale[r])))
    g, vf, kind = grad(lambda dd, _r=r: (gt_nd, gt_rescale.clone().index_put_((torch.tensor([_r], device=device),),
                       (gt_rescale[_r] + dd).reshape(1)), M),
                       (gt_nd, gt_rescale, M), d)
    cols.append(g); names.append(nm); vfr.append(vf); kinds.append(kind)

J = np.stack(cols, axis=1)
# See the DEAD-CHANNEL block above. Zeroed, not deleted, so row indices still match FEAT_LABELS. The
# amplification is printed because it is the EVIDENCE that the guard did something: a dead row whose
# largest entry rivals the largest live one is a row that would have led the payload table.
if dead.any():
    _fin = np.isfinite(J)
    print(f"[noise] zeroed {int(dead.sum())} dead rows of J; their largest standardized entry was "
          f"{np.abs(J[dead][_fin[dead]]).max(initial=0.0):.3g}, against "
          f"{np.abs(J[~dead][_fin[~dead]]).max(initial=0.0):.3g} over the live rows.", flush=True)
    J[dead, :] = 0.0
P = J.shape[1]
norms_std = np.array([np.linalg.norm(J[:, p]) if np.isfinite(J[:, p]).all() else np.nan for p in range(P)])
norms_raw = np.array([np.linalg.norm(J[:, p] * fnoise) if np.isfinite(J[:, p]).all() else np.nan for p in range(P)])

print("\n=== per-param gradient ===")
print(f"{'param':11s} {'kind':9s} {'||g||_std':>10s} {'||g||_raw':>10s} {'valid':>6s}")
for p in range(P):
    print(f"{names[p]:11s} {kinds[p]:9s} {norms_std[p]:10.3f} {norms_raw[p]:10.4g} {vfr[p]:6.2f}")

# ---- WHICH ROWS ARE DRIVING J ---------------------------------------------------------------------
# Advisory, no threshold, and deliberately so. Between "float32 rounding" (dropped above) and "a real
# feature" sits a band nothing can adjudicate by magnitude: a `logcyc` whose CHI_MAX_CYCLES ceiling
# binds varies only through the floor() quantization of N_row, giving a std ~1e-4 -- four orders above
# the dead test and two below a healthy feature. Any constant drawn there would be a guess. So rank
# rather than gate, and put fnoise next to the entry it produced: a row leading this table on a std
# 1000x under the median is quantization, and the probe budget above says which probe it is.
_absJ = np.abs(np.nan_to_num(J, nan=0.0))
_rowmax, _fmed = _absJ.max(1), float(np.median(fnoise))
print(f"\n=== feature rows dominating J (median fnoise {_fmed:.3g}; check it before believing one) ===")
print(f"  {'row':18s} {'fnoise':>10s} {'/median':>9s} {'max|J|':>9s}  at param")
for _i in np.argsort(-_rowmax)[:8]:
    print(f"  {FEAT_LABELS[_i]:18s} {fnoise[_i]:10.3g} {fnoise[_i] / max(_fmed, 1e-30):9.3g} "
          f"{_rowmax[_i]:9.3g}  {names[int(np.argmax(_absJ[_i]))]}")

# ---- THE DELIVERABLE, AS DATA ---------------------------------------------------------------------
# The payoff is a forced-vs-chi DIFF of tables that take hours to produce and, until now,
# existed only in a terminal buffer -- anything nobody thought to print in advance was simply gone.
# J and its labels ARE the map; every table above and below is a pure function of them. Saved beside
# the mode-suffixed figures, under the same suffix and for the same reason, so comparing the two runs
# is a five-line numpy script (see the module docstring) instead of a re-run. `fnoise` and `dead` ride
# along on purpose: without the denominator a standardized Jacobian cannot be re-interpreted later.
npz = str(PLOT_PATH / f"degeneracy_map_{MODE_TAG}.npz")
np.savez(npz, J=J, fnoise=fnoise, dead=dead, norms_std=norms_std, norms_raw=norms_raw,
         feat_labels=np.array([str(s) for s in FEAT_LABELS]),
         param_names=np.array([str(s) for s in names]),
         kinds=np.array(kinds), valid_frac=np.array(vfr),
         mults=(_MULTS.cpu().numpy() if cfg.chi_mode else np.zeros(0)),
         # `mode` is the OBSERVATION MODE, not MODE_TAG -- the tag also carries T_obs, which `meta`
         # already reports separately, and a field named `mode` holding "forced_T4.5" is the kind of
         # small lie that a later reader builds a filter on.
         meta=np.array([f"mode={cfg.observation_mode}", f"tag={MODE_TAG}",
                        f"T_obs={cfg.T_obs}", f"dt_exp={cfg.dt_exp}",
                        f"N_obs={N_obs}", f"M={M}", f"M_NOISE={M_NOISE}", f"REL={REL}",
                        f"SEED={SEED}", f"NOISE_EPS={NOISE_EPS}", f"ZERO_TOL={ZERO_TOL}",
                        f"chi_max_cycles={cfg.chi_max_cycles}", f"band={cfg.chi_freq_bounds}",
                        f"K={cfg.chi_n_freqs if cfg.chi_mode else 0}"]))
print(f"\nsaved: {npz}", flush=True)

measurable = np.array([kinds[p] != "UNMEAS" for p in range(P)])
stiff = measurable & (np.nan_to_num(norms_std) > ZERO_TOL)
mi = [p for p in range(P) if measurable[p]]
si = [p for p in range(P) if stiff[p]]
print(f"\nunmeasurable (both sides destabilize): {[names[p] for p in range(P) if not measurable[p]] or 'none'}")
print(f"no local info (||g||_std<{ZERO_TOL}): {[names[p] for p in range(P) if measurable[p] and not stiff[p]] or 'none'}")

# ---- cosines over measurable columns ----
ns = [names[p] for p in mi]
Jm = J[:, mi]
Jn = Jm / np.maximum(np.linalg.norm(Jm, axis=0), 1e-12)
C = np.abs(Jn.T @ Jn)
print("\n=== |cos(grad_p, grad_q)| over measurable params (|cos|->1 = degenerate) ===")
print("            " + " ".join(f"{n[:7]:>7s}" for n in ns))
for i in range(len(mi)):
    print(f"{ns[i]:11s} " + " ".join(f"{C[i, j]:7.2f}" for j in range(len(mi))))
hot = [(ns[i], ns[j], C[i, j]) for i in range(len(mi)) for j in range(i + 1, len(mi)) if C[i, j] > 0.9]
print("\ndegenerate pairs (|cos|>0.90): " + (", ".join(f"{a}~{b} ({c:.2f})" for a, b, c in hot) or "none"))

# ---- SVD over stiff columns ----
nss = [names[p] for p in si]
Js = J[:, si]
U, S, Vt = np.linalg.svd(Js, full_matrices=False)
Sn = S / S[0]
print(f"\n=== SVD over stiff columns {nss} ===")
for k in range(len(S)):
    print(f"  sigma[{k}] = {S[k]:9.3f}  (norm {Sn[k]:.4f})")
print(f"  condition number = {S[0] / max(S[-1], 1e-12):.1f}")
print("\n=== sloppiest stiff direction (smallest singular value) loadings ===")
for j in np.argsort(-np.abs(Vt[-1])):
    print(f"  {nss[j]:11s} {Vt[-1][j]:+.3f}")

# ---- unique-handle over measurable columns ----
print("\n=== unique-handle ||g_p ⟂ span(others)|| / ||g_p|| (low = degenerate) ===")
uniq = {}
for p in range(len(mi)):
    others = np.delete(Jm, p, axis=1)
    coef, *_ = np.linalg.lstsq(others, Jm[:, p], rcond=None)
    uniq[ns[p]] = np.linalg.norm(Jm[:, p] - others @ coef) / max(np.linalg.norm(Jm[:, p]), 1e-12)
for nm in sorted(uniq, key=lambda k: uniq[k]):
    print(f"  {nm:11s} unique={uniq[nm]:.3f}   ||g||_std={np.linalg.norm(Jm[:, ns.index(nm)]):.3f}")

# ---- plots ----
fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(C, vmin=0, vmax=1, cmap="magma")
ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns, rotation=45, ha="right")
ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns)
for i in range(len(ns)):
    for j in range(len(ns)):
        ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                color="white" if C[i, j] < 0.6 else "black", fontsize=6)
ax.set_title(f"|cos| between standardized feature-gradients — {MODE_TAG.upper()} "
             f"({len(FEAT_LABELS)} features)")
fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
# Suffixed by MODE. The payoff is a forced-vs-chi COMPARISON, so unsuffixed names meant the
# second run silently overwrote the first and you compared a figure with itself.
heat = str(PLOT_PATH / f"degeneracy_cosine_{MODE_TAG}.png"); fig.savefig(heat, dpi=130)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.bar(range(len(Sn)), Sn, color="steelblue"); ax2.set_yscale("log")
ax2.set_xlabel("singular index"); ax2.set_ylabel("sigma / sigma_max (log)")
ax2.set_title(f"Jacobian singular spectrum over stiff cols — {MODE_TAG.upper()} (small = sloppy)")
fig2.tight_layout()
spec = str(PLOT_PATH / f"degeneracy_singular_spectrum_{MODE_TAG}.png"); fig2.savefig(spec, dpi=130)
print("\nsaved:", heat); print("saved:", spec)

# ---- which FEATURES carry each parameter -------------------------------------------------------
# The actual scientific payload of the forced-vs-chi comparison: not merely whether an alias weakened,
# but whether the CHI features are what broke it. Compare the two runs' tables side by side.
print(f"\n=== top features per parameter ({MODE_TAG.upper()}) ===")
for p in range(P):
    col = J[:, p]
    if not np.isfinite(col).all():
        print(f"  {names[p]:11s} (unmeasurable)")
        continue
    top = np.argsort(-np.abs(col))[:5]
    print(f"  {names[p]:11s} " + ", ".join(f"{FEAT_LABELS[i]}={col[i]:+.2f}" for i in top))

print(f"\n[mode] this map describes the {MODE_TAG.upper()} information set. Re-run with the other "
      f"mode and diff the tables above.", flush=True)
print("DEGENERACY_MAP_DONE", flush=True)
