"""
Step 4 (PRISM_HANDOFF.md, Appendix A): degeneracy / sloppiness map over ALL 16 inferred params.

Generalizes scripts/diagnose_fmax.py Part B. At the cell-file ground truth it builds the
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

Env: CELL TOBS_S M M_NOISE REL SEED MIN_VALID ZERO_TOL
Run:  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/degeneracy_map.py
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
from core.config import CHUNK_LEN, PLOT_PATH
from core.SBI import chi as chi_mod, pipeline
from core.SBI.statistics import FEATURE_LABELS

_common.enable_warnings()

M = int(os.environ.get("M", "32"))
M_NOISE = int(os.environ.get("M_NOISE", "128"))
REL = float(os.environ.get("REL", "0.02"))
SEED = int(os.environ.get("SEED", "0"))
MIN_VALID = float(os.environ.get("MIN_VALID", "0.5"))
ZERO_TOL = float(os.environ.get("ZERO_TOL", "0.05"))     # ||g||_std below this = "no local info"
SF, SS, SC = 1, 2, 3                                     # CRN seeds: forced / spontaneous / chi block
torch.manual_seed(SEED)
print(f"[cfg] M={M} M_NOISE={M_NOISE} REL={REL} MIN_VALID={MIN_VALID} ZERO_TOL={ZERO_TOL} "
      f"SEED={SEED}", flush=True)

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
#   chi    : [S(41) with Group G ZEROED | chi(3K)]  -- so Group G's 11 columns are dropped and the
#            3K chi columns take their place.
#
# Left unchanged, this script reported the FORCED Jacobian regardless of the chi toggle: it kept the
# 11 Group-G columns chi mode zeroes and omitted the 3K chi columns entirely. Its |cos| pairs were
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
    forcing_gt = torch.tensor([[v for v, _ in cfg.force_params_dict.values()]],
                              dtype=dtype, device=device)
    amp_v = forcing_gt[:, cfg.forcing_idx["amp"]]
    freq_v = forcing_gt[:, cfg.forcing_idx["freq"]]
    phase_v = forcing_gt[:, cfg.forcing_idx["phase"]]

MODE_TAG = cfg.observation_mode                           # suffixes the outputs, see the plots below
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
            chi_block = pipeline.gen_chi_block(
                model=cfg.model, params_nd=p, rescale=rv, x_spont_dim=xs_d.to(dtype),
                t_fine=t_fine, inits=inits_m, rescale_idx=cfg.rescale_idx, n_segs=n_segs,
                steady_idx=cfg.steady_idx, subsample=subs, N_points=N_obs, dt_exp=cfg.dt_exp,
                multipliers=_MULTS, f0_nd=cfg.chi_f0, state_dep_drift=cfg.state_dep_drift,
                dtype=dtype, device=device)
            spont = pipeline.gen_stats(xs_d, None, cfg.dt_exp, None, None, None,
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
        feats = pipeline.gen_stats(xs_d, xf_d, cfg.dt_exp, amp_v.expand(m).double(),
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
P = J.shape[1]
norms_std = np.array([np.linalg.norm(J[:, p]) if np.isfinite(J[:, p]).all() else np.nan for p in range(P)])
norms_raw = np.array([np.linalg.norm(J[:, p] * fnoise) if np.isfinite(J[:, p]).all() else np.nan for p in range(P)])

print("\n=== per-param gradient ===")
print(f"{'param':11s} {'kind':9s} {'||g||_std':>10s} {'||g||_raw':>10s} {'valid':>6s}")
for p in range(P):
    print(f"{names[p]:11s} {kinds[p]:9s} {norms_std[p]:10.3f} {norms_raw[p]:10.4g} {vfr[p]:6.2f}")

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
# Suffixed by MODE. Step 3 of the handoff is a forced-vs-chi COMPARISON, so unsuffixed names meant the
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
