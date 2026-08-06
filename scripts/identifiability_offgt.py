"""
Step 5 decisive check: is local identifiability a GT-only artifact, or does it hold across
the prior?

The Laplace metric (feature_candidate_test.py) showed every param except t_offset/f_scale is
locally pinned at the cell-file GT -> the SBC failures look like a FLOW-CALIBRATION problem,
not an information deficit. But SBC averages over the whole prior, so this re-runs the same
41-feature Laplace marginal-SD analysis at K points: GT + (K-1) draws from the ACTUAL training
prior (extracted from posterior.prior, so it matches what the network saw).

If kappa/lambda/... stay pinned (SD << 1) across the prior -> confirmed: information is
sufficient prior-wide, the SBC failures are flow calibration -> commit to Track A (reparam).
If identifiability degrades off-GT (SD -> 1 at many points) -> information matters after all.

Method per point: CRN central diffs (per-member stability filter, one-sided fallback, float64
redim), standardized by single-traj feature noise, columns scaled to prior range (log-range for
log-uniform scales); marginal SD = sqrt(diag((J^T J + I)^-1)) in prior-range units.

Env: CELL TOBS_S M M_NOISE REL SEED K MIN_VALID SD_ID
Run:  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/identifiability_offgt.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import _common
from core import orchestrator
from core.config import CHUNK_LEN, POSTERIOR_PATH
from core.SBI import pipeline
from core.SBI.reparam import build_inferred_bijection, load_eval_bijection

_common.enable_warnings()

POST = os.environ.get("POST", "posterior_3d.pt")
M = int(os.environ.get("M", "32"))
M_NOISE = int(os.environ.get("M_NOISE", "128"))
REL = float(os.environ.get("REL", "0.02"))
SEED = int(os.environ.get("SEED", "0"))
K = int(os.environ.get("K", "6"))                  # total points incl. GT
MIN_VALID = float(os.environ.get("MIN_VALID", "0.5"))
SD_ID = float(os.environ.get("SD_ID", "0.3"))      # SD below this = "identified"
SF, SS = 1, 2
torch.manual_seed(SEED)
print(f"[cfg] POST={POST} K={K} M={M} M_NOISE={M_NOISE} REL={REL} SD_ID={SD_ID} SEED={SEED}",
      flush=True)

cfg = _common.script_cfg(default_cell=_common.FORCED_DEFAULT_CELL)   # CELL env var wins
# The Laplace marginal-SD verdict below is computed over the single-frequency 41-feature Fisher. A
# chi posterior conditions on a different feature set entirely, so the same arithmetic would produce
# a confident, meaningless "identified / not identified" call. Refuse rather than mislead.
_common.assert_not_chi(cfg, "identifiability_offgt")
_common.assert_forced(cfg, "identifiability_offgt")
_common.describe_features(cfg)
dtype, device = cfg.hw.dtype, cfg.hw.device
N_obs = int(cfg.T_obs / cfg.dt_exp)
nd_dim = len(cfg.params_dict)

ND_LBL = ["kappa", "lambda", "phi", "tau", "tau_c", "S", "dG", "beta", "N", "temp"]
res_names = list(cfg.rescale_params.keys())
names = ND_LBL + res_names
P = len(names)
bounds = [b for _, b in cfg.params_dict.values()] + [cfg.rescale_params[n][1] for n in res_names]
is_log = [False] * nd_dim + ["scale" in n for n in res_names]
amp_i, freq_i, phase_i = cfg.forcing_idx["amp"], cfg.forcing_idx["freq"], cfg.forcing_idx["phase"]


def _raw(nd, res, force, m, crn):
    t_scale = float(res[cfg.rescale_idx["t_scale"]])
    subs = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    n_fine = min(cfg.steady_idx + N_obs * subs, len(cfg.t))
    t_fine = cfg.t[:n_fine]
    n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))
    p = nd.unsqueeze(0).expand(m, -1).contiguous()
    rv = res.unsqueeze(0).expand(m, -1).contiguous()
    fv = force.unsqueeze(0)
    forcef = pipeline.build_nondim_sin_force_tensor(fv.expand(m, -1), t_fine, rv, cfg.forcing_idx, cfg.rescale_idx)

    def sim(f):
        return pipeline.gen_obs(model=cfg.model, params=p, t=t_fine,
                                inits=cfg.inits_tensor.expand(m, -1).contiguous(), force=f,
                                n_segs=n_segs, steady_idx=cfg.steady_idx,
                                state_dep_drift=cfg.state_dep_drift, batch_size=m, dtype=dtype,
                                device=device)[0][:, ::subs][:, :N_obs]
    if crn:
        torch.manual_seed(SF)
    xf = sim(forcef)
    if crn:
        torch.manual_seed(SS)
    xs = sim(torch.zeros_like(forcef))
    xsc = res[cfg.rescale_idx["x_scale"]].double()
    xof = res[cfg.rescale_idx["x_offset"]].double() if "x_offset" in cfg.rescale_idx else 0.0
    xf_d, xs_d = xsc * xf.double() + xof, xsc * xs.double() + xof
    feats = pipeline.gen_stats(xs_d, xf_d, cfg.dt_exp, fv[:, amp_i].expand(m).double(),
                               fv[:, freq_i].expand(m).double(), fv[:, phase_i].expand(m).double(),
                               device=device).numpy()
    return feats, xf_d, xs_d


def _valid(xf_d, xs_d, CAP):
    fin = torch.isfinite(xf_d).all(1) & torch.isfinite(xs_d).all(1)
    mag = (xf_d.abs().amax(1) < CAP) & (xs_d.abs().amax(1) < CAP)
    return (fin & mag).cpu().numpy()


def measure(nd, res, force, m, CAP):
    feats, xf_d, xs_d = _raw(nd, res, force, m, crn=True)
    v = _valid(xf_d, xs_d, CAP)
    if not v.any():
        return np.full(_common.n_features(cfg), np.nan), 0.0
    return feats[v].mean(0), float(v.mean())


def grad(perturb, base, d, CAP):
    fp, vp = measure(*perturb(+d), CAP)
    fm, vm = measure(*perturb(-d), CAP)
    if vp >= MIN_VALID and vm >= MIN_VALID:
        return (fp - fm) / (2 * d)
    f0, _ = measure(*base, CAP)
    if vp >= MIN_VALID:
        return (fp - f0) / d
    if vm >= MIN_VALID:
        return (f0 - fm) / d
    return np.full(_common.n_features(cfg), np.nan)


def analyze_point(nd, res, force):
    f0, xf0, xs0 = _raw(nd, res, force, M_NOISE, crn=False)
    fin0 = (torch.isfinite(xf0).all(1) & torch.isfinite(xs0).all(1)).cpu().numpy()
    if fin0.sum() < 10:
        return np.full(P, np.nan), 0
    amax0 = torch.maximum(xf0.abs().amax(1), xs0.abs().amax(1)).cpu().numpy()
    CAP = 100.0 * float(np.median(amax0[fin0]))
    keep = fin0 & (amax0 < CAP)
    fnoise = np.maximum(f0[keep].std(0), 1e-9)
    cols, meas = [], 0
    for p in range(P):
        lo, hi = bounds[p]
        if p < nd_dim:
            base_val = float(nd[p]); d = max(REL * (hi - lo), 1e-5 * abs(base_val))
            perturb = lambda dd, _i=p: (nd.clone().index_put_((torch.tensor([_i], device=device),), (nd[_i] + dd).reshape(1)), res, force, M)
        else:
            r = p - nd_dim; base_val = float(res[r]); d = max(REL * (hi - lo), 1e-5 * abs(base_val))
            perturb = lambda dd, _r=r: (nd, res.clone().index_put_((torch.tensor([_r], device=device),), (res[_r] + dd).reshape(1)), force, M)
        g = grad(perturb, (nd, res, force, M), d, CAP)
        if np.isfinite(g).all():
            meas += 1
        g = np.nan_to_num(g) / fnoise
        fac = base_val * math.log(hi / lo) if (is_log[p] and base_val > 0 and lo > 0) else (hi - lo)
        cols.append(g * fac)
    J = np.stack(cols, axis=1)
    cov = np.linalg.inv(J.T @ J + np.eye(P))
    return np.sqrt(np.clip(np.diag(cov), 0, None)), meas


# ---- build the K evaluation points: GT + (K-1) prior draws ----
gt_nd = cfg.params_tensor[0].clone()
gt_res = torch.tensor([v for v, _ in cfg.rescale_params.values()], dtype=dtype, device=device)
gt_force = torch.tensor([v for v, _ in cfg.force_params_dict.values()], dtype=dtype, device=device)

base = torch.load(str(POSTERIOR_PATH / POST), weights_only=False)
latent_prior = base.prior.gen_dist                          # RotatedLatentPrior if trained rotated
T = build_inferred_bijection(cfg)
# Reconstruct POST's exact training bijection (log box + optional rotation) from its sidecar,
# so latent draws map to physical θ consistently; falls back to plain linear T for legacy posteriors.
T_eval = load_eval_bijection(cfg, POST, POSTERIOR_PATH)
force_prior = orchestrator._build_forcing_prior(cfg)
z = latent_prior.sample((K - 1,))
theta = T_eval(z.to(device))
force_s = force_prior.sample((K - 1,)).to(device)

points = [("GT", gt_nd, gt_res, gt_force)]
for k in range(K - 1):
    points.append((f"prior{k+1}", theta[k, :nd_dim].clone(), theta[k, nd_dim:].clone(), force_s[k].clone()))

# ---- run ----
SD = np.full((len(points), P), np.nan)
for j, (tag, nd, res, force) in enumerate(points):
    sd, meas = analyze_point(nd, res, force)
    SD[j] = sd
    print(f"[{tag:8s}] measurable params={meas}/{P}", flush=True)

print("\n=== marginal posterior SD per param across points (prior-range units) ===")
print(f"{'param':9s} " + " ".join(f"{p[0][:7]:>7s}" for p in points) + f" {'median':>8s} {'frac<'+str(SD_ID):>8s}")
for p in range(P):
    row = SD[:, p]
    med = np.nanmedian(row)
    frac = np.mean(row[np.isfinite(row)] < SD_ID) if np.isfinite(row).any() else float("nan")
    print(f"{names[p]:9s} " + " ".join(f"{v:7.3f}" if np.isfinite(v) else f"{'nan':>7s}" for v in row)
          + f" {med:8.3f} {frac:8.2f}")

# ---- verdict ----
focus = ["kappa", "lambda", "x_scale", "t_scale", "tau", "tau_c"]
print("\n=== VERDICT (focus params: identified = SD < {:.2f}) ===".format(SD_ID))
all_id = True
for nm in focus:
    p = names.index(nm)
    row = SD[:, p][np.isfinite(SD[:, p])]
    frac = np.mean(row < SD_ID) if row.size else float("nan")
    if not (frac >= 0.8):
        all_id = False
    print(f"  {nm:9s} median SD={np.nanmedian(SD[:, p]):.3f}  identified in {frac*100:.0f}% of points")
if all_id:
    print("  => Identifiability HOLDS across the prior -> info is sufficient prior-wide ->")
    print("     the SBC failures are FLOW CALIBRATION -> commit to Track A (reparam).")
else:
    print("  => Identifiability DEGRADES off-GT for some params -> information matters there ->")
    print("     reconsider Track B (features) / longer T_obs for those params.")
print("IDENTIFIABILITY_OFFGT_DONE", flush=True)
