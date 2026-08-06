"""
Diagnose the NEW f_scale marginal-SBC tilt in the 13-dim keeper (posterior_07012026).

f_scale passed SBC in every prior run but mild-fails one-sided in the 13-dim linear+rotation
run (KS med 0.004, 8/10). f_scale is NOT a degeneracy param (those are kappa/lambda/x_scale/
t_scale), so this is an in-scope flow-calibration tilt, not the information ceiling. This tool
tests the two candidate causes CHEAPLY (no retrain, no simulation):

  H1  ROTATION over-mixing: does the decorrelating rotation V smear the f_scale axis into the
      degeneracy block {kappa,lambda,x_scale,t_scale}, so the flow's errors on the degenerate
      ridge leak into f_scale? -> analyze the deployed V's f_scale loadings.
  H2  BAD BOX COORDINATE: f_scale is a strictly-positive ~3-decade scale param. With
      REPARAM_LOG_PARAMS=[] it uses a LINEAR box, so its (log-uniform) prior mass and its GT
      pile up at the flat low EDGE of the box -- a latent region the fixed-resolution spline
      flow mis-calibrates. -> compare the LINEAR box vs a LOG-on-f_scale box in latent coords.

Prints a verdict + the targeted fix. NB: f_scale is a RESCALE param, so log-ing it only touches
the (analytic) rescale bijection, NOT the ND GMM -> build_posterior's guard passes with the
existing linear ND prior (no rebuild). A retrain is still needed to CONFIRM the fix.

Env: POST (default posterior_07012026.pt), CELL (default nadrowski/master_spont.txt), NS (prior samples).
Run:  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/diagnose_fscale.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import _common
from core import cli, orchestrator
from core.config import (SimConfig, DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S, detect_device,
                         NADROWSKI_LABELS, POSTERIOR_PATH)
from core.SBI.reparam import build_inferred_bijection, build_rescale_bijection

POST = os.environ.get("POST", "posterior_07012026.pt")
NS = int(os.environ.get("NS", "20000"))

_common.enable_warnings()
cfg = _common.script_cfg()
dtype, device = cfg.hw.dtype, cfg.hw.device
nd_dim = len(cfg.params_dict)
rnames = list(cfg.rescale_params.keys())               # e.g. [x_scale, t_scale, f_scale]
inames = list(cfg.params_dict.keys()) + rnames         # 13 inferred keys (ND cell-keys + rescale)
if "f_scale" not in rnames:
    print("f_scale not in this cell's rescale params — nothing to diagnose."); sys.exit(0)
FS = nd_dim + rnames.index("f_scale")
DEGEN = [inames.index(k) for k in ("k", "lam") if k in inames] \
      + [nd_dim + rnames.index(k) for k in ("x_scale", "t_scale") if k in rnames]
print(f"[cfg] POST={POST}  inferred={len(inames)}D  f_scale idx={FS}  degeneracy idx={DEGEN}", flush=True)


# ============ H1: ROTATION — is f_scale mixed into the degeneracy block? ============
print("\n" + "=" * 74 + "\nH1  ROTATION MIXING (deployed V)\n" + "=" * 74)
base = POST[:-3] if POST.endswith(".pt") else POST
rot_path = POSTERIOR_PATH / (base + ".rot.pt")
V = None
if rot_path.exists():
    obj = torch.load(str(rot_path), map_location="cpu", weights_only=False)
    Vt = obj.get("V") if isinstance(obj, dict) else obj
    V = torch.as_tensor(Vt).float().numpy() if Vt is not None else None
if V is None:
    print("  No rotation V in sidecar (linear, unrotated). H1 is vacuous — the rotation cannot"
          "\n  be mixing f_scale. Skip to H2.")
    h1_mixes = False
else:
    # V is orthogonal; z = V w. Row V[p,:] shows how box-axis p spreads across the w eigen-modes.
    # self-conc = max|row| (~1 => axis rides its own w-coordinate = clean). degen-overlap =
    # sum_m |row_m| * (fraction of mode m that is degeneracy) => how much p leaks into the ridge.
    degcontent = np.sqrt((V[DEGEN, :] ** 2).sum(0))
    def mix(idx):
        row = V[idx, :]
        return float(np.max(np.abs(row))), float((np.abs(row) * degcontent).sum())
    print("  %-8s self-conc  degen-overlap   (1.0/0.0 = clean own-axis; low/high = smeared into ridge)"
          % "axis")
    for idx in [FS] + DEGEN:
        c, o = mix(idx)
        tag = "  <== f_scale" if idx == FS else ("  (degeneracy param)" if idx in DEGEN else "")
        print("  %-8s %.3f      %.3f%s" % (inames[idx], c, o, tag))
    fc, fo = mix(FS)
    h1_mixes = (fc < 0.85 or fo > 0.30)
    print("  => f_scale is %s by the rotation (self-conc %.3f, degen-overlap %.3f)."
          % ("MIXED into the degeneracy block" if h1_mixes else "NOT mixed — rides its own axis", fc, fo))


# ============ H2: BOX COORDINATE — does the linear box push f_scale to the edge? ============
print("\n" + "=" * 74 + "\nH2  BOX COORDINATE (linear vs log-on-f_scale, latent geometry)\n" + "=" * 74)
lo, hi = cfg.rescale_params["f_scale"][1]
gt_fs = cfg.rescale_params["f_scale"][0]
print(f"  f_scale bounds=({lo}, {hi})  span={hi/lo:.0f}x ({np.log10(hi/lo):.1f} decades)  GT={gt_fs}")

# GT latent position (unbounded z) under each box: |z| large => GT sits in the flat sigmoid tail.
gt = cfg.ground_truth_tensor
z_lin = float(build_inferred_bijection(cfg, log_params=[]).inv(gt)[FS])
z_log = float(build_inferred_bijection(cfg, log_params=["f_scale"]).inv(gt)[FS])
u_lin = 1.0 / (1.0 + np.exp(-z_lin)); u_log = 1.0 / (1.0 + np.exp(-z_log))
print(f"  GT f_scale latent z:   LINEAR box z={z_lin:+.2f} (box fraction u={u_lin:.3f})"
      f"   |   LOG box z={z_log:+.2f} (u={u_log:.3f})")
print(f"     (|z|>~3 = deep in the flat sigmoid tail, where the fixed-knot flow mis-resolves.)")

# Prior mass: map the (log-uniform) f_scale prior to latent under each box; edge pile-up = pathology.
rp = orchestrator._build_rescale_prior(cfg)
rs = rp.sample((NS,)).cpu()                              # (NS, n_rescale) physical
fs_col = rnames.index("f_scale")
zr_lin = build_rescale_bijection(cfg, log_params=[]).inv(rs)[:, fs_col].numpy()
zr_log = build_rescale_bijection(cfg, log_params=["f_scale"]).inv(rs)[:, fs_col].numpy()
def q(a): return np.percentile(a, [5, 50, 95])
print(f"  f_scale PRIOR latent [5/50/95 pct], frac(|z|>3):")
print(f"     LINEAR box: {q(zr_lin).round(2)}   frac|z|>3 = {np.mean(np.abs(zr_lin) > 3):.2f}")
print(f"     LOG box:    {q(zr_log).round(2)}   frac|z|>3 = {np.mean(np.abs(zr_log) > 3):.2f}")
h2_edge = (abs(z_lin) > 3) or (np.mean(np.abs(zr_lin) > 3) > 0.25)


# ============ VERDICT ============
print("\n" + "=" * 74 + "\nVERDICT\n" + "=" * 74)
if V is not None and not h1_mixes and h2_edge:
    print("  ROTATION is NOT the cause (f_scale rides its own axis, unmixed from the degeneracy).")
    print("  The LINEAR box IS the cause: f_scale is a strictly-positive multi-decade scale param,")
    print("  so its GT and log-uniform prior pile up at the flat low edge of the linear box — a")
    print("  latent region the fixed-resolution spline flow mis-calibrates, producing the mild")
    print("  one-sided tilt. A LOG box centers f_scale's latent geometry (see u/z above).")
    print("\n  TARGETED FIX (confirm by retrain): config.REPARAM_LOG_PARAMS = ['f_scale']")
    print("    - log ONLY f_scale; keep kappa/lambda/x_scale/t_scale LINEAR (log OVER-MIXED those).")
    print("    - f_scale is a RESCALE param => NO ND prior rebuild (build_posterior guard passes on")
    print("      the existing linear ND prior). Reuse prior_forcing_no_forcing.pt, TRAIN-NEW.")
    print("    - Re-check with sbc_characterize.py; header should read 'log-box dims=1/13'.")
elif h1_mixes:
    print("  ROTATION appears to mix f_scale into the degeneracy block. Consider rotating only the")
    print("  degeneracy sub-block, or lowering REPARAM_FISHER_POINTS, then retrain to confirm.")
else:
    print("  Neither hypothesis is clearly supported: f_scale is unmixed AND the linear box does not")
    print("  push it to the edge. The tilt is likely retrain/flow-fit variance (~1.5%, safe direction)")
    print("  — acceptable under a calibrated-joint-posterior bar; no config change indicated.")
print("DIAGNOSE_FSCALE_DONE", flush=True)
