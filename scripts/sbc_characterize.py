"""
Full characterization of the t_offset SBC anomaly.

t_offset's SBC KS p-value swung 0.762 -> 0.001 between training runs, unlike the
persistently-bad cluster (kappa/lambda/beta). This repeats SBC K times at a raised
n_cal on a saved posterior and reports the run-to-run DISTRIBUTION of the per-parameter
KS p-values plus a t_offset rank-histogram deep-dive, to disambiguate:

  - sampling NOISE      : t_offset KS p is high/variable across repeats; pooled rank
                          histogram is ~flat.
  - REAL miscalibration : KS p stays low across repeats; pooled hist shows a systematic
                          shape (cap = overconfident, U = underconfident, slope = biased).
  - phase-wrap ARTIFACT : KS p low but the t_offset posterior is multimodal/periodic.
                          t_offset enters ONLY through the drive phase (mod 1/f), so a
                          wrapped posterior can break SBC's linear rank statistic without
                          the posterior being miscalibrated in the usual sense.

The training prior is extracted directly from the saved posterior
(posterior.prior = SBIPriorWrapper(latent ProductPrior)), so the SBC proposal is
guaranteed to match training and no separate (possibly mismatched) prior file is needed.

Env knobs:
  CELL   cell file                                  (default Resources/Cells/nadrowski/master_spont.txt)
  POST   posterior filename under Resources/Posteriors (default posterior_3d.pt)
  K      number of SBC repeats                      (default 10)
  N_CAL  calibration datasets per repeat            (default 2000)
  NPS    posterior samples per calibration point    (default 1000)
  SEED   base RNG seed                              (default 0)
  CHI_K_FIXED  hold the chi probe COUNT at this value instead of pooling over the training
               mixture -- run per stratum (2 / 6 / CHI_K_PAD) as well as pooled  (default: pooled)

Outputs are suffixed by STRATUM (`_pooled` or `_k<N>`), because the stratified SBC protocol is a FOUR-RUN
comparison and unsuffixed names meant each run overwrote the last -- so the workflow the suffix
serves left only its final stratum on disk. Same defect degeneracy_map.py's MODE_TAG fixed, on a
different axis. Compare strata by label across the four .npz files.

Run:
  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/sbc_characterize.py

  # the full stratified sweep (chi mode, a trained chi posterior):
  $env:CHI=1; $env:POST="<name>.pt"
  foreach ($k in 2, 6, 12) { $env:CHI_K_FIXED=$k; & $py scripts/sbc_characterize.py }
  Remove-Item Env:CHI_K_FIXED;                    & $py scripts/sbc_characterize.py   # pooled
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")          # headless: savefig only, no display
from matplotlib import pyplot as plt
from sbi.diagnostics import run_sbc, check_sbc

import _common
from core import orchestrator
from core.config import POSTERIOR_PATH, PLOT_PATH
from core.SBI import analysis
from core.SBI.reparam import UnitToBoxTransform, OrthogonalTransform

_common.enable_warnings()

# ---- knobs ---- (CELL / BOUNDS / MODEL / TOBS_S / CHI* are handled by _common.script_cfg)
POST = os.environ.get("POST", "posterior_3d.pt")
K = int(os.environ.get("K", "10"))
N_CAL = int(os.environ.get("N_CAL", "2000"))
NPS = int(os.environ.get("NPS", "1000"))
SEED = int(os.environ.get("SEED", "0"))
# Stratify the calibration set by PROBE COUNT instead of pooling over the training mixture. A pooled
# SBC over a mixture of counts can be flat while each count is miscalibrated in compensating
# directions -- posterior_chi_08042026 is the standing proof that flat SBC is not by itself evidence
# of a working conditioning path. Run once per stratum (2 / 6 / CHI_K_PAD) AND once pooled, and
# compare. Ignored outside chi mode.
CHI_K_FIXED = int(os.environ["CHI_K_FIXED"]) if os.environ.get("CHI_K_FIXED") else None
print(f"[cfg] POST={POST}  K={K}  N_CAL={N_CAL}  NPS={NPS}  SEED={SEED}", flush=True)

# T_obs is irrelevant for SBC; the mode (spontaneous / forced / chi) is NOT -- it decides the width of
# the conditioning vector gen_cal_data must produce, and a mismatch with POST used to surface only
# after the whole calibration set had been simulated.
cfg = _common.script_cfg()
if CHI_K_FIXED is not None and not cfg.chi_mode:
    raise SystemExit("CHI_K_FIXED only means something in chi(omega) mode; this config is "
                     f"{cfg.observation_mode.upper()}. Set CHI=1, or unset CHI_K_FIXED.")
# The stratum is stated on its own line, not buried in a knob dump: a stratified run and a pooled run
# produce identically-shaped reports, so nothing else on screen distinguishes them on a later read.
print(f"[cfg] probe-count stratum: "
      f"{'POOLED over the training mixture' if CHI_K_FIXED is None else f'FIXED K = {CHI_K_FIXED}'}",
      flush=True)
dtype, device = cfg.hw.dtype, cfg.hw.device
nd_dim = len(cfg.params_dict)
labels = cfg.inferred_labels
_has_toff = "t_offset" in cfg.rescale_idx
t_off_idx = nd_dim + cfg.rescale_idx["t_offset"] if _has_toff else -1
if _has_toff:
    print(f"[cfg] device={device}  nd_dim={nd_dim}  n_inferred={len(labels)}  "
          f"t_offset global idx={t_off_idx} ('{labels[t_off_idx]}')", flush=True)
else:
    print(f"[cfg] device={device}  nd_dim={nd_dim}  n_inferred={len(labels)}  "
          f"(no t_offset in this cell; skipping t_offset-specific diagnostics)", flush=True)

# ---- load posterior + extract its EXACT training (latent) prior ----
# load_posterior verifies POST's observation mode against cfg FIRST and exits loudly on a mismatch.
# Without that check the run generated K x N_CAL simulations and only then died on a raw matrix-shape
# RuntimeError inside EmbeddedNet's first Linear.
post_latent, T_eval, posterior, _sidecar = _common.load_posterior(POST, cfg)
latent_inferred_prior = post_latent.prior.gen_dist          # latent prior; RotatedLatentPrior if trained rotated
_rot = any(isinstance(p, OrthogonalTransform) for p in T_eval.parts)
_box = next((p for p in T_eval.parts if isinstance(p, UnitToBoxTransform)), None)
_nlog = int(_box.log_mask.sum()) if _box is not None else 0
print(f"[reparam] POST={POST}: rotation={'on' if _rot else 'off'}, log-box dims={_nlog}/{len(labels)}",
      flush=True)
# None for a SPONTANEOUS or CHI config -- neither samples a drive prior. gen_cal_data is told which
# branch to take below, so it never dereferences this.
force_prior = orchestrator.build_forcing_prior(cfg)

_z = latent_inferred_prior.sample((2,))
assert _z.shape[-1] == len(labels), f"latent prior dim {_z.shape[-1]} != n_inferred {len(labels)}"
print(f"[prior] extracted latent inferred prior from posterior.prior; sample dim={_z.shape[-1]}", flush=True)

t = cfg.t
# Outputs are suffixed by STRATUM. Handoff 4.1 step 6 is a four-run comparison -- CHI_K_FIXED at
# 2 / 6 / CHI_K_PAD, then pooled -- and unsuffixed names meant each run silently overwrote the last,
# so the workflow left only the final stratum on disk and you compared a file with itself. This is
# exactly the defect degeneracy_map.py's MODE_TAG was added to fix, on a different axis.
STRATUM_TAG = "pooled" if CHI_K_FIXED is None else f"k{CHI_K_FIXED}"
csv_path = str(PLOT_PATH / f"sbc_characterization_pvals_{STRATUM_TAG}.csv")
npz_path = str(PLOT_PATH / f"sbc_characterization_ranks_{STRATUM_TAG}.npz")

# ---- SBC repeat loop ----
ks_matrix = np.full((K, len(labels)), np.nan)               # (K, n_inferred) KS p-values
c2st_matrix = np.full((K, len(labels)), np.nan)
ranks_all = []                                              # list of (n_valid, n_inferred) per repeat

for r in range(K):
    torch.manual_seed(SEED + r)
    t0 = time.time()
    x_cal, theta_star = analysis.gen_cal_data(
        model=cfg.model, prior=latent_inferred_prior, forcing_prior=force_prior,
        t=t, steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min, n_cal=N_CAL,
        nd_dim=nd_dim, forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds, theta_transform=T_eval,
        state_dep_drift=cfg.state_dep_drift,
        # Thread the OBSERVATION MODE through. Omitting these let gen_cal_data default to the forced
        # branch, so a spontaneous config hit `forcing_prior.sample` on a None and a chi config built
        # a 46-wide conditioning vector for a network expecting 42+3K.
        spontaneous_only=(cfg.observation_mode == "spontaneous"),
        chi_mode=cfg.chi_mode, chi_f0=cfg.chi_f0,
        # chi_k_pad from the CONFIG, not gen_training_data's module fallback: the pad width is frozen
        # into the trained artifact (trap CHI7), so a config carrying a different one than the live
        # config.CHI_K_PAD would silently calibrate against a differently-shaped block.
        chi_freq_bounds=cfg.chi_freq_bounds, chi_k_pad=cfg.chi_k_pad, chi_k_fixed=CHI_K_FIXED,
        # chi_max_cycles from the CONFIG for the same reason as chi_k_pad, and its absence here used
        # to be the one field this call did not thread while orchestrator.validate_calibration did.
        # It is the lock-in duration ceiling, so it sets the `logcyc` every probe reports -- the one
        # channel whose job is telling the encoder how much to trust a probe. Falling back to the
        # live config.CHI_MAX_CYCLES calibrates against a value the network may never have been
        # trained on; benign only for as long as the load-time ceiling guard forces the two to agree.
        chi_max_cycles=cfg.chi_max_cycles,
        n_vars=cfg.inits_tensor.shape[-1],
        dtype=dtype, device=device,
    )
    n_valid = theta_star.shape[0]
    ranks, dap = run_sbc(
        thetas=theta_star.to(device), xs=x_cal.to(device), posterior=posterior,
        num_posterior_samples=NPS, reduce_fns="marginals",
        use_batched_sampling=True, show_progress_bar=False,
    )
    prior_samples = T_eval(latent_inferred_prior.sample((n_valid,)).to(device)).cpu()
    stats = check_sbc(ranks=ranks.cpu(), prior_samples=prior_samples,
                      dap_samples=dap.cpu(), num_posterior_samples=NPS)
    ks_matrix[r] = np.asarray(stats["ks_pvals"])
    c2st_matrix[r] = np.asarray(stats["c2st_ranks"])
    ranks_all.append(ranks.cpu().numpy())

    # incremental saves so a killed background run still leaves usable results
    np.savetxt(csv_path, ks_matrix, delimiter=",",
               header=",".join(str(l) for l in labels), comments="")
    np.savez(npz_path, ks=ks_matrix, c2st=c2st_matrix, t_off_idx=t_off_idx,
             labels=np.array([str(l) for l in labels]),
             ranks=np.concatenate(ranks_all, axis=0), nps=NPS)
    if _has_toff:
        print(f"[run {r+1}/{K}] n_valid={n_valid}  t_offset KS p={ks_matrix[r, t_off_idx]:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
    else:
        print(f"[run {r+1}/{K}] n_valid={n_valid}  ({time.time() - t0:.1f}s)", flush=True)

# ---- aggregate report ----
def col_summary(j):
    col = ks_matrix[:, j][np.isfinite(ks_matrix[:, j])]
    if col.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return float(np.median(col)), float(col.min()), float(col.max()), float((col < 0.05).mean())

print("\n=== KS p-value distribution over K repeats (sorted by median; low = miscalibrated) ===")
print(f"{'param':16s} {'median':>8s} {'min':>8s} {'max':>8s} {'frac<.05':>9s}")
for j in sorted(range(len(labels)), key=lambda k: col_summary(k)[0]):
    med, lo, hi, frac = col_summary(j)
    mark = "   <== t_offset (target)" if (_has_toff and j == t_off_idx) else ""
    print(f"{str(labels[j]):16s} {med:8.3f} {lo:8.3f} {hi:8.3f} {frac:9.2f}{mark}")

# ---- t_offset deep-dive: pooled rank histogram + ECDF ----
if not _has_toff:
    print("\n(t_offset not in this cell's rescale params — skipping the t_offset deep-dive, "
          "marginals, and verdict; the general KS study above still stands.)", flush=True)
    print("SBC_CHARACTERIZE_DONE", flush=True)
    sys.exit(0)
toff_pooled = np.concatenate([rk[:, t_off_idx] for rk in ranks_all])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(toff_pooled, bins=30, density=True, color="steelblue", alpha=0.85)
axes[0].axhline(1.0 / NPS, color="k", ls="--", label="uniform")
axes[0].set_title(f"t_offset pooled rank histogram (K={K}, n={toff_pooled.size})")
axes[0].set_xlabel("rank"); axes[0].set_ylabel("density"); axes[0].legend()
xs = np.sort(toff_pooled); ecdf = np.arange(1, xs.size + 1) / xs.size
axes[1].plot(xs, ecdf, color="steelblue", label="empirical")
axes[1].plot([0, NPS], [0, 1], color="k", ls="--", label="uniform")
axes[1].set_title("t_offset rank ECDF vs uniform")
axes[1].set_xlabel("rank"); axes[1].set_ylabel("CDF"); axes[1].legend()
fig.tight_layout()
hist_png = str(PLOT_PATH / f"sbc_toffset_characterization_{STRATUM_TAG}.png")
fig.savefig(hist_png, dpi=130); print("\nsaved:", hist_png, flush=True)

# ---- phase-wrap diagnostic: t_offset posterior marginals for a few calibration points ----
n_show = min(6, x_cal.shape[0])
sel = np.random.default_rng(SEED).choice(x_cal.shape[0], size=n_show, replace=False)
figm, axm = plt.subplots(2, 3, figsize=(13, 7))
for ax, i in zip(axm.ravel(), sel):
    s = posterior.sample((2000,), x=x_cal[i:i + 1].to(device))
    ax.hist(s[:, t_off_idx].cpu().numpy(), bins=40, color="darkorange", alpha=0.85)
    ax.axvline(theta_star[i, t_off_idx].item(), color="k", ls="--")
    ax.set_title(f"cal #{int(i)}"); ax.set_xlabel(str(labels[t_off_idx]))
figm.suptitle("t_offset posterior marginals (dashed = truth) — multimodal/periodic => phase-wrap artifact")
figm.tight_layout()
marg_png = str(PLOT_PATH / f"sbc_toffset_marginals_{STRATUM_TAG}.png")
figm.savefig(marg_png, dpi=130); print("saved:", marg_png, flush=True)

# ---- verdict heuristic ----
med, lo, hi, frac = col_summary(t_off_idx)
print("\n=== VERDICT (t_offset) ===")
print(f"  KS p over {K} repeats: median={med:.3f}  range=[{lo:.3f}, {hi:.3f}]  frac<0.05={frac:.2f}")
if med > 0.05 and frac < 0.5:
    print("  => Consistent with SAMPLING NOISE (high/variable KS p; the 0.001 was likely an unlucky draw).")
elif med < 0.05 and frac > 0.8:
    print("  => Looks REAL (KS p stays low). Read the pooled rank histogram shape:")
    print("     cap=overconfident, U=underconfident, slope=biased, multimodal=phase-wrap artifact.")
else:
    print("  => BORDERLINE: raise K and/or N_CAL, and inspect the histogram shape.")
print("SBC_CHARACTERIZE_DONE", flush=True)
