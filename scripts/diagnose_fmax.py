"""
Part A diagnostic: empirical sensitivity of the 41 summary features to f_max (phi).

Holds all NWK params at the cell-file ground truth, sweeps f_max across a grid, and at
each point simulates an ensemble of spontaneous + forced trajectories at the GT rescale,
computes the 41 features, and ranks each feature by its rank-correlation / dynamic range
vs f_max. Purely empirical (direct simulation) -- no analytical reduction-map quantities.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import spearmanr

import _common
from core.config import CHUNK_LEN
from core.SBI import pipeline
from core.SBI.statistics import FEATURE_LABELS

_common.enable_warnings()

M = int(os.environ.get("M", "16"))      # ensemble size per f_max point
N_GRID = int(os.environ.get("NGRID", "13"))
torch.manual_seed(0)

cfg = _common.script_cfg(default_cell=_common.FORCED_DEFAULT_CELL)   # CELL env var wins
# The whole point of this script is which of the 41 single-frequency features tracks f_max -- and
# Group G, where much of that signal lives, is ZEROED in chi mode. Reporting a Group-G feature as the
# best handle for a chi run would be actively misleading, so refuse rather than mislead.
_common.assert_not_chi(cfg, "diagnose_fmax")
_common.assert_forced(cfg, "diagnose_fmax")
_common.assert_nadrowski(cfg, "FMAX_IDX below is a fixed Nadrowski column index")
_common.describe_features(cfg)
dtype, device = cfg.hw.dtype, cfg.hw.device

FMAX_IDX = 2                            # phi = f_max is the 3rd ND parameter
fmax_key = list(cfg.params_dict.keys())[FMAX_IDX]
fmax_gt = cfg.params_dict[fmax_key][0]
fmax_lo, fmax_hi = cfg.params_dict[fmax_key][1]
print(f"f_max key='{fmax_key}'  GT={fmax_gt}  bounds=({fmax_lo}, {fmax_hi})", flush=True)

# derived sim sizes at GT rescale (mirror generate_observations)
t_scale_gt = cfg.rescale_params["t_scale"][0]
subsample = max(1, round((cfg.dt_exp / t_scale_gt) / cfg.dt_nd_min))
N_obs = int(cfg.T_obs / cfg.dt_exp)
n_fine = cfg.steady_idx + N_obs * subsample
t_fine = cfg.t[:n_fine]
n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))

rescale_gt = torch.tensor([[v for v, _ in cfg.rescale_params.values()]], dtype=dtype, device=device)
forcing_gt = torch.tensor([[v for v, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
x_scale = rescale_gt[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
x_offset = (rescale_gt[:, cfg.rescale_idx["x_offset"]].unsqueeze(1)
            if "x_offset" in cfg.rescale_idx else torch.zeros((1, 1), dtype=dtype, device=device))
amp_v = forcing_gt[:, cfg.forcing_idx["amp"]]
freq_v = forcing_gt[:, cfg.forcing_idx["freq"]]
phase_v = forcing_gt[:, cfg.forcing_idx["phase"]]


def features_at(fmax: float, m: int):
    p = cfg.params_tensor.clone()
    p[0, FMAX_IDX] = fmax
    p = p.expand(m, -1).contiguous()
    inits_m = cfg.inits_tensor.expand(m, -1).contiguous()
    force = pipeline.build_nondim_sin_force_tensor(
        forcing_gt.expand(m, -1), t_fine, rescale_gt.expand(m, -1), cfg.forcing_idx, cfg.rescale_idx)

    def sim(f):
        x_nd = pipeline.gen_obs(model=cfg.model, params=p, t=t_fine, inits=inits_m, force=f,
                                n_segs=n_segs, steady_idx=cfg.steady_idx, state_dep_drift=cfg.state_dep_drift,
                                batch_size=m, dtype=dtype, device=device)[0]
        return x_nd[:, ::subsample][:, :N_obs]

    x_forced = x_scale * sim(force) + x_offset
    x_spont = x_scale * sim(torch.zeros_like(force)) + x_offset
    stable = (torch.isfinite(x_spont).all(1) & torch.isfinite(x_forced).all(1)).float().mean().item()
    feats = pipeline.gen_stats(x_spont, x_forced, cfg.dt_exp,
                               amp_v.expand(m), freq_v.expand(m), phase_v.expand(m), device=device)
    return feats.mean(0).cpu().numpy(), stable


grid = np.geomspace(0.4, 2.6, N_GRID)   # ~0.4x .. 2.5x of GT (1.06)
print(f"\nSweeping f_max over {grid.round(3).tolist()} (M={M} ensemble each)\n", flush=True)
rows, stab = [], []
for fmax in grid:
    mean_feats, stable = features_at(float(fmax), M)
    rows.append(mean_feats); stab.append(stable)
    print(f"  f_max={fmax:6.3f}  stable_frac={stable:.2f}", flush=True)

F = np.asarray(rows)                     # (N_GRID, 41)
rho = np.array([abs(spearmanr(grid, F[:, j]).correlation) for j in range(len(FEATURE_LABELS))])
rho = np.nan_to_num(rho)
norm_range = (F.max(0) - F.min(0)) / np.maximum(np.abs(F).mean(0), 1e-9)

group = np.array([lab[0] for lab in FEATURE_LABELS])   # 'A'..'G'
order = np.argsort(-rho)

print("\n=== Feature sensitivity to f_max (|Spearman rho| over the sweep) ===")
print("(|rho|>~0.55 is significant for n=13; A1/A2 are the scale/offset-degenerate anchors)\n")
print(f"{'feature':26s} {'|rho|':>6s} {'norm_range':>11s}")
for j in order:
    print(f"{FEATURE_LABELS[j]:26s} {rho[j]:6.3f} {norm_range[j]:11.3f}")

print("\n=== Summary by group (max |rho| within each group) ===")
for g in "ABCDEFG":
    mask = group == g
    jmax = np.where(mask)[0][np.argmax(rho[mask])]
    print(f"  group {g}: max |rho|={rho[mask].max():.3f}  (best: {FEATURE_LABELS[jmax]})")

shape_mask = np.isin(group, list("BCDEF"))
print(f"\nScale-invariant SHAPE features (B-F): max |rho|={rho[shape_mask].max():.3f}"
      f"  ({FEATURE_LABELS[np.where(shape_mask)[0][np.argmax(rho[shape_mask])]]})")
print(f"Anchor A2_log_var |rho|={rho[FEATURE_LABELS.index('A2_log_var')]:.3f}, "
      f"A1_mean |rho|={rho[FEATURE_LABELS.index('A1_mean')]:.3f}")
print(f"min stable_frac across sweep: {min(stab):.2f}")
print("DIAGNOSE_A_DONE")

if os.environ.get("RUN_B", "1") != "1":
    sys.exit(0)

# ===================== Part B: degeneracy directions =====================
# Feature-gradients at GT w.r.t. {f_max, s, x_scale, x_offset}, standardized by the
# across-sweep feature std, using common random numbers (fixed seeds) so the central
# differences isolate the parameter effect rather than ensemble noise.
print("\n=== Part B: feature-gradient alignment at GT (common random numbers) ===", flush=True)
fscale = np.maximum(F.std(0), 1e-9)
gt = cfg.params_tensor[0].clone()
S_IDX = 5  # s is the 6th ND parameter
xscale_gt, xoffset_gt = float(x_scale.item()), float(x_offset.item())
MB, SF, SS = 24, 1, 2

def feats_crn(pvec, xsc, xof, m):
    p = pvec.unsqueeze(0).expand(m, -1).contiguous()
    inits_m = cfg.inits_tensor.expand(m, -1).contiguous()
    force = pipeline.build_nondim_sin_force_tensor(
        forcing_gt.expand(m, -1), t_fine, rescale_gt.expand(m, -1), cfg.forcing_idx, cfg.rescale_idx)
    torch.manual_seed(SF)
    xf = pipeline.gen_obs(model=cfg.model, params=p, t=t_fine, inits=inits_m, force=force, n_segs=n_segs,
                          steady_idx=cfg.steady_idx, state_dep_drift=cfg.state_dep_drift, batch_size=m,
                          dtype=dtype, device=device)[0][:, ::subsample][:, :N_obs]
    torch.manual_seed(SS)
    xs = pipeline.gen_obs(model=cfg.model, params=p, t=t_fine, inits=inits_m, force=torch.zeros_like(force),
                          n_segs=n_segs, steady_idx=cfg.steady_idx, state_dep_drift=cfg.state_dep_drift,
                          batch_size=m, dtype=dtype, device=device)[0][:, ::subsample][:, :N_obs]
    feats = pipeline.gen_stats(xsc * xs + xof, xsc * xf + xof, cfg.dt_exp,
                               amp_v.expand(m), freq_v.expand(m), phase_v.expand(m), device=device)
    return feats.mean(0).cpu().numpy()

def grad_nwk(idx, rel=0.1, m=MB):
    d = rel * abs(float(gt[idx]))
    pp, pm = gt.clone(), gt.clone(); pp[idx] += d; pm[idx] -= d
    return (feats_crn(pp, xscale_gt, xoffset_gt, m) - feats_crn(pm, xscale_gt, xoffset_gt, m)) / (2 * d) / fscale

g_fmax = grad_nwk(FMAX_IDX)
g_s = grad_nwk(S_IDX)
dxs = 0.1 * xscale_gt
g_xscale = (feats_crn(gt, xscale_gt + dxs, xoffset_gt, MB) - feats_crn(gt, xscale_gt - dxs, xoffset_gt, MB)) / (2 * dxs) / fscale
dxo = 0.1 * xscale_gt
g_xoffset = (feats_crn(gt, xscale_gt, xoffset_gt + dxo, MB) - feats_crn(gt, xscale_gt, xoffset_gt - dxo, MB)) / (2 * dxo) / fscale

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print("  ||grad|| (standardized feature units / unit param):")
for nm, g in [("f_max", g_fmax), ("s", g_s), ("x_scale", g_xscale), ("x_offset", g_xoffset)]:
    print(f"    {nm:8s}: {np.linalg.norm(g):.3f}")
print(f"  |cos(grad_fmax, grad_x_offset)| = {abs(cos(g_fmax, g_xoffset)):.3f}")
print(f"  |cos(grad_fmax, grad_x_scale)|  = {abs(cos(g_fmax, g_xscale)):.3f}")
print(f"  |cos(grad_fmax, grad_s)|        = {abs(cos(g_fmax, g_s)):.3f}")

A_anchor = np.stack([g_xscale, g_xoffset], axis=1)
coef = np.linalg.lstsq(A_anchor, g_fmax, rcond=None)[0]
resid = g_fmax - A_anchor @ coef
print(f"  ||g_fmax ⟂ (x_scale,x_offset)|| / ||g_fmax|| = {np.linalg.norm(resid) / (np.linalg.norm(g_fmax) + 1e-12):.3f}")
print("  largest f_max feature-components orthogonal to the anchors (the usable handles, if any):")
for j in np.argsort(-np.abs(resid))[:6]:
    print(f"    {FEATURE_LABELS[j]:24s} {resid[j]:+.3f}")
print("DIAGNOSE_B_DONE")
