"""
Step 5 / Track B: candidate-feature tester (Laplace identifiability metric).

For each CANDIDATE summary statistic, build the standardized feature-Jacobian over ALL 16
inferred params at the cell-file GT (CRN central diffs, per-member stability filter, float64
redim -- same robustness as degeneracy_map.py), then ask the question that actually predicts
SBC: does appending the candidate SHRINK a param's marginal posterior width?

Metric: a local Gaussian (Laplace) identifiability model. Columns of the Jacobian are scaled to
"per prior range" (linear range for uniform params; gt*log(hi/lo) for the log-uniform scale
params). Fisher F = J^T J; posterior cov ~ (F + I)^-1 with a unit prior in prior-range units.
Marginal SD = sqrt(diag(cov)) in prior-range units: ~1 = unidentified (posterior ~ prior),
<<1 = pinned. The reduction in a param's marginal SD when a candidate is appended is the real
value of that candidate -- this is the CONDITIONAL info gain the pairwise cosine misses.

(The pairwise alias cosine is norm-dominated -- kappa's 41-feature gradient has norm ~110 while
a scalar candidate is ~6, so cosine barely moves even for a useful feature. Hence the Laplace
metric instead.)

First batch (all scale-invariant): tau_int/tau0, acf_at_2tau0 (lambda); bicoh_2f, embed_ecc,
rise_fall (kappa).

Env: CELL TOBS_S M M_NOISE REL SEED MIN_VALID
Run:  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/feature_candidate_test.py
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
from core.config import CHUNK_LEN, PLOT_PATH
from core.SBI import pipeline

_common.enable_warnings()

M = int(os.environ.get("M", "48"))
M_NOISE = int(os.environ.get("M_NOISE", "192"))
REL = float(os.environ.get("REL", "0.02"))
SEED = int(os.environ.get("SEED", "0"))
MIN_VALID = float(os.environ.get("MIN_VALID", "0.5"))
SF, SS = 1, 2
torch.manual_seed(SEED)
print(f"[cfg] M={M} M_NOISE={M_NOISE} REL={REL} SEED={SEED}", flush=True)

cfg = _common.script_cfg()
# This ranks candidate features by how much they shrink a marginal ON TOP OF the 41-feature baseline.
# In chi mode the baseline is a different feature set (Group G zeroed, 3K chi features added), so the
# measured "gain" of every candidate would be inflated -- chi already supplies much of it -- and the
# ranking would be wrong with no sign of trouble.
_common.assert_not_chi(cfg, "feature_candidate_test")
_common.describe_features(cfg)
dtype, device = cfg.hw.dtype, cfg.hw.device
DT = cfg.dt_exp
N_obs = int(cfg.T_obs / cfg.dt_exp)

gt_nd = cfg.params_tensor[0].clone()
gt_rescale = torch.tensor([v for v, _ in cfg.rescale_params.values()], dtype=dtype, device=device)
forcing_gt = torch.tensor([[v for v, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
amp_v, freq_v, phase_v = (forcing_gt[:, cfg.forcing_idx[k]] for k in ("amp", "freq", "phase"))

CAND_LABELS = ["tau_int/tau0(L)", "acf_at_2tau0(L)", "bicoh_2f(K)", "embed_ecc(K)", "rise_fall(K)"]
NC = len(CAND_LABELS)


def _acf(x):
    n = x.shape[-1]
    s = x - x.mean(-1, keepdim=True)
    sf = torch.fft.rfft(s, n=2 * n, dim=-1)
    ac = torch.fft.irfft(sf.abs() ** 2, n=2 * n, dim=-1)[:, :n]
    return ac / ac[:, :1].clamp_min(1e-30)


def _fpeak(x):
    n = x.shape[-1]
    xf = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1)
    psd = xf.abs() ** 2
    psd[:, 0] = 0.0
    fr = torch.fft.rfftfreq(n, d=DT, device=x.device).to(x.dtype)
    return float(fr[psd.argmax(-1)].clamp_min(fr[1]).median())


def _tau0_idx(ac):
    n = ac.shape[-1]
    below = ac < math.exp(-1.0)
    return torch.where(below.any(-1), below.int().argmax(-1),
                       torch.full((ac.shape[0],), n - 1, device=ac.device)).clamp(min=1)


def candidate_matrix(xs):
    x = xs - xs.mean(-1, keepdim=True)
    B, n = x.shape
    ar = torch.arange(n, device=x.device)
    ac = _acf(x)
    i0 = _tau0_idx(ac)
    tau0 = i0.double() * DT
    neg = ac < 0
    firstneg = torch.where(neg.any(-1), neg.int().argmax(-1), torch.full((B,), n, device=x.device))
    mask = ar.unsqueeze(0) < firstneg.unsqueeze(1)
    tau_int = (ac.clamp_min(0) * mask).sum(-1) * DT
    c_tauratio = torch.log((tau_int / tau0.clamp_min(DT)).clamp_min(1e-6))
    i2 = (2 * i0).clamp(max=n - 1)
    c_acf2 = ac.gather(1, i2.unsqueeze(1)).squeeze(1)

    fpk = _fpeak(x)
    df = 1.0 / (n * DT)
    S = 8
    L = n // S
    if L >= 16:
        seg = (x[:, :S * L].reshape(B, S, L)) * torch.hann_window(L, device=x.device, dtype=x.dtype)
        Xf = torch.fft.rfft(seg, dim=-1)
        dfL = 1.0 / (L * DT)
        kf = max(1, int(round(fpk / dfL))); k2 = 2 * kf
        if k2 < Xf.shape[-1]:
            A, A2 = Xf[..., kf], Xf[..., k2]
            bisp = (A * A * torch.conj(A2)).mean(1)
            den = torch.sqrt((A * A).abs().pow(2).mean(1) * A2.abs().pow(2).mean(1)).clamp_min(1e-30)
            c_bicoh = (bisp.abs() / den)
        else:
            c_bicoh = torch.zeros(B, device=x.device, dtype=x.dtype)
    else:
        c_bicoh = torch.zeros(B, device=x.device, dtype=x.dtype)

    d = max(1, int(round(0.25 / max(fpk, df) / DT)))
    d = d if d < n - 8 else 1
    a, b = x[:, :-d], x[:, d:]
    am, bm = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    c00, c11, c01 = (am * am).mean(-1), (bm * bm).mean(-1), (am * bm).mean(-1)
    tr = c00 + c11
    disc = (tr * tr / 4 - (c00 * c11 - c01 * c01)).clamp_min(0).sqrt()
    l1, l2 = tr / 2 + disc, (tr / 2 - disc).clamp_min(1e-30)
    c_ecc = torch.log((l1 / l2).clamp_min(1.0))

    dx = torch.diff(x, dim=-1)
    up, dn = dx.clamp_min(0), (-dx).clamp_min(0)
    mu = up.sum(-1) / (up > 0).sum(-1).clamp_min(1)
    md = dn.sum(-1) / (dn > 0).sum(-1).clamp_min(1)
    c_rf = torch.log((mu / md.clamp_min(1e-30)).clamp_min(1e-6))

    return torch.stack([c_tauratio, c_acf2, c_bicoh, c_ecc, c_rf], dim=-1).cpu().numpy()


def _raw(pvec, rescale_vec, m, crn):
    t_scale = float(rescale_vec[cfg.rescale_idx["t_scale"]])
    subs = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    n_fine = min(cfg.steady_idx + N_obs * subs, len(cfg.t))
    t_fine = cfg.t[:n_fine]
    n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))
    p = pvec.unsqueeze(0).expand(m, -1).contiguous()
    rv = rescale_vec.unsqueeze(0).expand(m, -1).contiguous()
    force = pipeline.build_nondim_sin_force_tensor(forcing_gt.expand(m, -1), t_fine, rv,
                                                   cfg.forcing_idx, cfg.rescale_idx)

    def sim(f):
        return pipeline.gen_obs(model=cfg.model, params=p, t=t_fine,
                                inits=cfg.inits_tensor.expand(m, -1).contiguous(), force=f,
                                n_segs=n_segs, steady_idx=cfg.steady_idx,
                                state_dep_drift=cfg.state_dep_drift, batch_size=m, dtype=dtype,
                                device=device)[0][:, ::subs][:, :N_obs]
    if crn:
        torch.manual_seed(SF)
    xf = sim(force)
    if crn:
        torch.manual_seed(SS)
    xs = sim(torch.zeros_like(force))
    xsc = rescale_vec[cfg.rescale_idx["x_scale"]].double()
    xof = rescale_vec[cfg.rescale_idx["x_offset"]].double() if "x_offset" in cfg.rescale_idx else 0.0
    xf_d, xs_d = xsc * xf.double() + xof, xsc * xs.double() + xof
    feats = pipeline.gen_stats(xs_d, xf_d, cfg.dt_exp, amp_v.expand(m).double(),
                               freq_v.expand(m).double(), phase_v.expand(m).double(),
                               device=device).numpy()
    return feats, xf_d, xs_d


def _valid(xf_d, xs_d):
    fin = torch.isfinite(xf_d).all(1) & torch.isfinite(xs_d).all(1)
    mag = (xf_d.abs().amax(1) < CAP) & (xs_d.abs().amax(1) < CAP)
    return (fin & mag).cpu().numpy()


feats0, xf0, xs0 = _raw(gt_nd, gt_rescale, M_NOISE, crn=False)
fin0 = (torch.isfinite(xf0).all(1) & torch.isfinite(xs0).all(1)).cpu().numpy()
amax0 = torch.maximum(xf0.abs().amax(1), xs0.abs().amax(1)).cpu().numpy()
CAP = 100.0 * float(np.median(amax0[fin0]))
keep0 = fin0 & (amax0 < CAP)
fnoise41 = np.maximum(feats0[keep0].std(0), 1e-9)
cand0 = candidate_matrix(xs0[torch.tensor(keep0, device=device)])
fnoiseC = np.maximum(cand0.std(0), 1e-9)
print(f"[noise] CAP={CAP:.4g} GT valid={keep0.mean():.2f}", flush=True)


def measure(pvec, rescale_vec, m):
    feats, xf_d, xs_d = _raw(pvec, rescale_vec, m, crn=True)
    v = _valid(xf_d, xs_d)
    if not v.any():
        return np.full(_common.n_features(cfg), np.nan), np.full(NC, np.nan), 0.0
    cand = candidate_matrix(xs_d[torch.tensor(v, device=device)])
    return feats[v].mean(0), cand.mean(0), float(v.mean())


def grad(perturb, base, d):
    f41p, fCp, vp = measure(*perturb(+d))
    f41m, fCm, vm = measure(*perturb(-d))
    if vp >= MIN_VALID and vm >= MIN_VALID:
        return (f41p - f41m) / (2 * d), (fCp - fCm) / (2 * d)
    f410, fC0, _ = measure(*base)
    if vp >= MIN_VALID:
        return (f41p - f410) / d, (fCp - fC0) / d
    if vm >= MIN_VALID:
        return (f410 - f41m) / d, (fC0 - fCm) / d
    return np.full(_common.n_features(cfg), np.nan), np.full(NC, np.nan)


# ---- 16-param Jacobian (41 feats + candidates), standardized & prior-range-scaled ----
ND_LBL = ["kappa", "lambda", "phi", "tau", "tau_c", "S", "dG", "beta", "N", "temp"]
specs, G41, GC = [], [], []
for i, nm in enumerate(ND_LBL):
    lo, hi = list(cfg.params_dict.values())[i][1]
    d = max(REL * (hi - lo), 1e-5 * abs(float(gt_nd[i])))
    perturb = lambda dd, _i=i: (gt_nd.clone().index_put_((torch.tensor([_i], device=device),), (gt_nd[_i] + dd).reshape(1)), gt_rescale, M)
    r41, rC = grad(perturb, (gt_nd, gt_rescale, M), d)
    G41.append(np.nan_to_num(r41 / fnoise41)); GC.append(np.nan_to_num(rC / fnoiseC))
    specs.append((nm, lo, hi, float(gt_nd[i]), False)); print(f"  grad {nm} done", flush=True)
for nm in cfg.rescale_params.keys():
    idx = cfg.rescale_idx[nm]; lo, hi = cfg.rescale_params[nm][1]
    d = max(REL * (hi - lo), 1e-5 * abs(float(gt_rescale[idx])))
    perturb = lambda dd, _r=idx: (gt_nd, gt_rescale.clone().index_put_((torch.tensor([_r], device=device),), (gt_rescale[_r] + dd).reshape(1)), M)
    r41, rC = grad(perturb, (gt_nd, gt_rescale, M), d)
    G41.append(np.nan_to_num(r41 / fnoise41)); GC.append(np.nan_to_num(rC / fnoiseC))
    specs.append((nm, lo, hi, float(gt_rescale[idx]), "scale" in nm)); print(f"  grad {nm} done", flush=True)

P = len(specs); names = [s[0] for s in specs]
fac = np.array([(sp[3] * math.log(sp[2] / sp[1]) if sp[4] else (sp[2] - sp[1])) for sp in specs])
J41 = np.stack(G41, axis=1) * fac[None, :]          # (41, P) per prior-range
JC = np.stack(GC, axis=1) * fac[None, :]            # (NC, P)


def marg_sd(Jstack):
    cov = np.linalg.inv(Jstack.T @ Jstack + np.eye(P))
    return np.sqrt(np.clip(np.diag(cov), 0, None))


sd_base = marg_sd(J41)
sd_all = marg_sd(np.vstack([J41, JC]))

ik, il = names.index("kappa"), names.index("lambda")
ix, it = names.index("x_scale"), names.index("t_scale")
print("\n=== candidate sensitivities (standardized dC/dparam) ===")
print(f"{'candidate':16s} {'kappa':>8s} {'x_scale':>8s} {'lambda':>8s} {'t_scale':>8s}")
GCu = np.stack(GC, axis=1)  # per-unit standardized (NC, P) before fac-scaling
for c in range(NC):
    print(f"{CAND_LABELS[c]:16s} {GCu[c, ik]:8.2f} {GCu[c, ix]:8.2f} {GCu[c, il]:8.2f} {GCu[c, it]:8.2f}")

print("\n=== Laplace marginal posterior SD (prior-range units; ~1=unidentified, <<1=pinned) ===")
print(f"{'param':9s} {'SD(41)':>8s} {'SD(+all)':>9s} {'reduction':>10s}")
for p in range(P):
    red = (1 - sd_all[p] / max(sd_base[p], 1e-12)) * 100
    print(f"{names[p]:9s} {sd_base[p]:8.3f} {sd_all[p]:9.3f} {red:9.1f}%")

print("\n=== per-candidate marginal-SD reduction (the real value of each feature) ===")
print(f"{'candidate':16s} {'kappa':>8s} {'lambda':>8s} {'best other':>22s}")
for c in range(NC):
    sd_c = marg_sd(np.vstack([J41, JC[c:c + 1]]))
    red = (1 - sd_c / np.maximum(sd_base, 1e-12)) * 100
    others = [(names[p], red[p]) for p in range(P) if p not in (ik, il)]
    bo = max(others, key=lambda kv: kv[1])
    print(f"{CAND_LABELS[c]:16s} {red[ik]:7.1f}% {red[il]:7.1f}%   {bo[0]:>10s} {bo[1]:6.1f}%")

# plot baseline vs +all marginal SDs
fig, ax = plt.subplots(figsize=(10, 4.5))
xp = np.arange(P)
ax.bar(xp - 0.2, sd_base, 0.4, label="41 features", color="steelblue")
ax.bar(xp + 0.2, sd_all, 0.4, label="+ all candidates", color="darkorange")
ax.set_xticks(xp); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylabel("marginal posterior SD (prior-range units)")
ax.set_title("Laplace identifiability: lower = better pinned (1 = prior-wide)")
ax.legend(); fig.tight_layout()
out = str(PLOT_PATH / "feature_candidate_test.png"); fig.savefig(out, dpi=130)
print("\nsaved:", out)
print("FEATURE_CANDIDATE_TEST_DONE", flush=True)
