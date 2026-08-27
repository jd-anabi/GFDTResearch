"""
Self-test for the decorrelating reparameterization machinery (NO retrain, NO orchestrator changes).

Gate before wiring into build_posterior + retraining. Validates:
  1. Fisher eigenbasis V: orthogonal (V^T V = I) and diagonalizes F (V^T F V is diagonal).
  2. Rotated bijection T_new = build_rotated_bijection(box, V): round-trips (T_new.inv(T_new(w))=w),
     maps into the physical box, and its log-det equals the box-only log-det (orthogonal adds 0).
  3. RotatedLatentPrior: sample/log_prob consistent with the base latent prior.
  4. Decorrelation sanity: V^T F V off-diagonal << diagonal (the whole point).

F is the simulation-based latent Fisher (feature-Jacobian wrt the latent coordinate z at GT),
so this also exercises the exact R-derivation the production path will use.

Run:  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/reparam_selftest.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import _common
from core import cli
from core.config import (SimConfig, DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S, detect_device,
                         NADROWSKI_LABELS, CHUNK_LEN, POSTERIOR_PATH)
from core.SBI import pipeline
from core.SBI.reparam import (build_inferred_bijection, build_rotated_bijection,
                              fisher_eigenbasis, RotatedLatentPrior)

CELL = os.environ.get("CELL", "Resources/Cells/nadrowski/master_spont.txt")
POST = os.environ.get("POST", "posterior_3d.pt")
M = int(os.environ.get("M", "32"))
M_NOISE = int(os.environ.get("M_NOISE", "128"))
DZ = float(os.environ.get("DZ", "0.1"))
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

_common.enable_warnings()
cfg = _common.script_cfg(default_cell=_common.FORCED_DEFAULT_CELL)   # CELL env var wins
_common.assert_forced(cfg, "reparam_selftest")
dtype, device = cfg.hw.dtype, cfg.hw.device
nd_dim = len(cfg.params_dict)
P = nd_dim + len(cfg.rescale_params)
N_obs = int(cfg.T_obs / cfg.dt_exp)
T = build_inferred_bijection(cfg)
gt_phys = cfg.ground_truth_tensor
z_gt = T.inv(gt_phys)

forcing_gt = torch.tensor([[v for v, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
amp_v, freq_v, phase_v = (forcing_gt[:, cfg.forcing_idx[k]] for k in ("amp", "freq", "phase"))


def sim_feats(theta_row, m):
    """(m, 41) features for a physical theta (16,) — CRN seeds (member spread = single-traj noise)."""
    nd = theta_row[:nd_dim].unsqueeze(0).expand(m, -1).contiguous()
    res = theta_row[nd_dim:]
    t_scale = float(res[cfg.rescale_idx["t_scale"]])
    subs = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    n_fine = min(cfg.steady_idx + N_obs * subs, len(cfg.t))
    t_fine = cfg.t[:n_fine]
    n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))
    rv = res.unsqueeze(0).expand(m, -1).contiguous()
    force = pipeline.build_nondim_sin_force_tensor(forcing_gt.expand(m, -1), t_fine, rv, cfg.forcing_idx, cfg.rescale_idx)

    def s(f):
        return pipeline.gen_obs(model=cfg.model, params=nd, t=t_fine,
                                inits=cfg.inits_tensor.expand(m, -1).contiguous(), force=f,
                                n_segs=n_segs, steady_idx=cfg.steady_idx,
                                state_dep_drift=cfg.state_dep_drift, batch_size=m, dtype=dtype,
                                device=device)[0][:, ::subs][:, :N_obs]
    torch.manual_seed(1); xf = s(force)
    torch.manual_seed(2); xs = s(torch.zeros_like(force))
    xsc = res[cfg.rescale_idx["x_scale"]].double()
    xof = res[cfg.rescale_idx["x_offset"]].double() if "x_offset" in cfg.rescale_idx else 0.0
    return pipeline.gen_stats_features(xsc * xs.double() + xof, xsc * xf.double() + xof, cfg.dt_exp,
                              amp_v.expand(m).double(), freq_v.expand(m).double(),
                              phase_v.expand(m).double(), device=device).numpy()


# ---- simulation-based latent Fisher F = J^T J ----
print("[fisher] computing latent feature-Jacobian at GT ...", flush=True)
fnoise = np.maximum(sim_feats(gt_phys, M_NOISE).std(0), 1e-9)
J = np.zeros((_common.n_features(cfg), P))   # the MODE's feature count, not a literal 41
for i in range(P):
    zp = z_gt.clone(); zp[i] += DZ
    zm = z_gt.clone(); zm[i] -= DZ
    J[:, i] = (sim_feats(T(zp), M).mean(0) - sim_feats(T(zm), M).mean(0)) / (2 * DZ) / fnoise
F = torch.tensor(J.T @ J, dtype=torch.float64, device=device)
V = fisher_eigenbasis(F).to(dtype)

# ---- check 1: orthogonality + diagonalization ----
I_ = torch.eye(P, dtype=V.dtype, device=V.device)
ortho_err = float((V.transpose(-1, -2) @ V - I_).abs().max())
FVV = V.transpose(-1, -2) @ F.to(V.dtype) @ V
off = float((FVV - torch.diag(torch.diag(FVV))).abs().max())
diag = float(torch.diag(FVV).abs().max())
F_off = float((F - torch.diag(torch.diag(F))).abs().max().to(V.dtype))
F_diag = float(torch.diag(F).abs().max().to(V.dtype))

# ---- check 2: rotated bijection ----
Tnew = build_rotated_bijection(T, V)
w = torch.randn(2000, P, dtype=dtype, device=device)
theta = Tnew(w)
lows = torch.tensor([b[0] for _, b in cfg.params_dict.values()] + [b[0] for _, b in cfg.rescale_params.values()], dtype=dtype, device=device)
highs = torch.tensor([b[1] for _, b in cfg.params_dict.values()] + [b[1] for _, b in cfg.rescale_params.values()], dtype=dtype, device=device)
in_box = bool((theta > lows - 1e-3 * (highs - lows)).all() and (theta < highs + 1e-3 * (highs - lows)).all())
rt = float((Tnew.inv(theta) - w).abs().max())

def _reduce(ld, ndim):
    while ld.dim() > ndim:
        ld = ld.sum(dim=-1)
    return ld
z = w @ V.transpose(-1, -2)
ld_new = _reduce(Tnew.log_abs_det_jacobian(w, theta), 1)
ld_box = _reduce(T.log_abs_det_jacobian(z, theta), 1)
ld_err = float((ld_new - ld_box).abs().max())
ld_finite = bool(torch.isfinite(ld_new).all())

# ---- check 3: RotatedLatentPrior plumbing on a clean single-device base ----
gbase = torch.distributions.Independent(
    torch.distributions.Normal(torch.zeros(P, device=device, dtype=dtype),
                               torch.ones(P, device=device, dtype=dtype)), 1)
rp = RotatedLatentPrior(gbase, V)
ws = rp.sample((2000,))
lp = rp.log_prob(ws)
lp_consist = float((lp - gbase.log_prob(ws @ V.transpose(-1, -2))).abs().max())
prior_finite = bool(torch.isfinite(lp).all()) and tuple(ws.shape) == (2000, P)

# ---- check 3b: real loaded prior — sample path only (its log_prob has a known mixed-device
#      layout: cpu rescale transform + cuda nd GMM; the pipeline only ever samples it) ----
try:
    base = torch.load(str(POSTERIOR_PATH / POST), weights_only=False).prior.gen_dist
    rs = RotatedLatentPrior(base, V).sample((8,))
    real_sample_ok = bool(torch.isfinite(rs).all()) and tuple(rs.shape) == (8, P)
except Exception as e:
    real_sample_ok = False
    print("  [3b] real-prior sample error:", repr(e)[:140])

# ---- report ----
def ok(b):
    return "PASS" if b else "*** FAIL ***"

print("\n=== reparam self-test ===")
print(f"1. V orthogonal:           ||V^T V - I||_max = {ortho_err:.2e}   {ok(ortho_err < 1e-4)}")
print(f"   V diagonalizes F:       off/diag = {off/max(diag,1e-30):.2e}  (F off/diag before = {F_off/max(F_diag,1e-30):.2e})  {ok(off/max(diag,1e-30) < 1e-5)}")
print(f"2. bijection round-trip:   ||T_new.inv(T_new(w)) - w||_max = {rt:.2e}   {ok(rt < 1e-3)}")
print(f"   maps into physical box: {ok(in_box)}")
print(f"   log-det == box log-det: |Δ|_max = {ld_err:.2e}, finite={ld_finite}   {ok(ld_err < 1e-3 and ld_finite)}")
print(f"3. rotated prior log_prob: consistency |Δ|_max = {lp_consist:.2e}, finite={prior_finite}   {ok(lp_consist < 1e-3 and prior_finite)}")
print(f"   real prior sample path: {ok(real_sample_ok)}  (its log_prob is mixed-device by")
print(f"                           construction -> wiring uses the sample path only)")
all_ok = (ortho_err < 1e-4 and off/max(diag,1e-30) < 1e-5 and rt < 1e-3 and in_box
          and ld_err < 1e-3 and ld_finite and lp_consist < 1e-3 and prior_finite and real_sample_ok)
print(f"\nOVERALL: {ok(all_ok)}")
print("REPARAM_SELFTEST_DONE", flush=True)
