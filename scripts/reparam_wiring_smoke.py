"""
Wiring smoke test for REPARAM_ROTATE (tiny train, ~1 min; NOT a real posterior).

Exercises the exact rotated training path that orchestrator.build_posterior now uses:
  decorrelate.build_latent_fisher_rotation -> RotatedLatentPrior + build_rotated_bijection
  -> pipeline.train_nn (rotated prior + rotated theta_transform) -> TransformedPosterior
  -> save V sidecar -> reload + rebuild rotated T -> sample (in-box, finite).
Confirms the integration runs before a full retrain. Writes/removes temp files _smoke_rot*.

Run:  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/reparam_wiring_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import _common
from core import cli, orchestrator
from core.config import (SimConfig, DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S, detect_device, NADROWSKI_LABELS,
                         POSTERIOR_PATH, DENSITY_ESTIMATOR, NSF_HIDDEN_FEATURES, NSF_NUM_TRANSFORMS,
                         NSF_NUM_BINS, TRAINING_LEARNING_RATE, TRAINING_BATCH_SIZE)
from core.SBI import pipeline, embedded_network, decorrelate, statistics
from core.SBI.Priors import sbi_prior_wrapper
from core.SBI.reparam import (build_inferred_bijection, build_rotated_bijection, RotatedLatentPrior,
                              TransformedPosterior)

torch.manual_seed(0)
_common.enable_warnings()
cfg = _common.script_cfg("Resources/Cells/nadrowski/cell_2.txt")
dtype, device = cfg.hw.dtype, cfg.hw.device
nd_dim = len(cfg.params_dict)
T = build_inferred_bijection(cfg)
force_prior = orchestrator._build_forcing_prior(cfg)

# latent inferred prior (inherit the structure from posterior_3d, as build_posterior would have it)
base = torch.load(str(POSTERIOR_PATH / "posterior_3d.pt"), weights_only=False).prior.gen_dist
print("[smoke] computing Fisher rotation V ...", flush=True)
V = decorrelate.build_latent_fisher_rotation(cfg, T)
print(f"[smoke] V shape={tuple(V.shape)}  orthogonality err={(V.T @ V - torch.eye(V.shape[0], device=device, dtype=dtype)).abs().max():.2e}", flush=True)

rotated_prior = RotatedLatentPrior(base, V)
T_train = build_rotated_bijection(T, V)

forcing_dim = orchestrator.expected_forcing_dim(cfg)   # shared width rule (chi-aware)
input_dim = len(statistics.FEATURE_LABELS) + 1
net = embedded_network.EmbeddedNet(input_dim, 3 * input_dim // 2, (5 * input_dim // 2, 2 * input_dim),
                                   forcing_dim=forcing_dim, forcing_layer_dims=(forcing_dim * 4, forcing_dim * 2),
                                   merge_layer_dim=2 * input_dim)
training_params = {"model": cfg.model, "prior": rotated_prior, "t": cfg.t, "run_size": 256, "num_runs": 2,
                   "steady_idx": cfg.steady_idx, "dt_nd_min": cfg.dt_nd_min, "dt_exp": cfg.dt_exp,
                   "t_min_exp": cfg.t_min_exp, "t_max_exp": cfg.t_max_exp, "t_scale_bounds": cfg.t_scale_bounds,
                   "state_dep_drift": cfg.state_dep_drift, "dtype": dtype, "device": device}
print("[smoke] tiny train (num_runs=2, run_size=256) on the rotated prior ...", flush=True)
post_latent, _ = pipeline.train_nn(
    training_params, model=DENSITY_ESTIMATOR, prior=sbi_prior_wrapper.SBIPriorWrapper(rotated_prior),
    embedding_net=net, forcing_prior=force_prior, nd_dim=nd_dim, forcing_idx=cfg.forcing_idx,
    rescale_idx=cfg.rescale_idx, x_obs=None, theta_obs=None, num_rounds=1, return_diagnostics=True,
    theta_transform=T_train, hidden_features=NSF_HIDDEN_FEATURES, num_transforms=NSF_NUM_TRANSFORMS,
    num_bins=NSF_NUM_BINS, learning_rate=TRAINING_LEARNING_RATE, stop_after_epochs=3, max_num_epochs=5,
    show_train_summary=False, batch_size=TRAINING_BATCH_SIZE, device=device)

# save + reload (mirrors build_posterior save / load paths)
torch.save(post_latent, str(POSTERIOR_PATH / "_smoke_rot.pt"))
torch.save(V, str(POSTERIOR_PATH / "_smoke_rot.rot.pt"))
pl2 = torch.load(str(POSTERIOR_PATH / "_smoke_rot.pt"), weights_only=False)
V2 = torch.load(str(POSTERIOR_PATH / "_smoke_rot.rot.pt"), weights_only=False)
post2 = TransformedPosterior(pl2, build_rotated_bijection(T, V2))

# sample at a fabricated conditioning vector (plumbing only): [41 stats | logT | forcing]
# Width from the MODE, not from the drive params: in chi mode the block is 3K wide.
x = torch.randn(1, input_dim + forcing_dim, dtype=dtype, device=device)
s = post2.sample((16,), x=x)
lows = torch.tensor([b[0] for _, b in cfg.params_dict.values()] + [b[0] for _, b in cfg.rescale_params.values()], dtype=dtype, device=device)
highs = torch.tensor([b[1] for _, b in cfg.params_dict.values()] + [b[1] for _, b in cfg.rescale_params.values()], dtype=dtype, device=device)
in_box = bool((s > lows - 1e-3 * (highs - lows)).all() and (s < highs + 1e-3 * (highs - lows)).all())
finite = bool(torch.isfinite(s).all())
print(f"\n[smoke] reloaded rotated posterior sampled {tuple(s.shape)}  in_box={in_box}  finite={finite}")
print(f"[smoke] V reload matches: {bool(torch.equal(V.cpu(), V2.cpu()))}")

for f in ("_smoke_rot.pt", "_smoke_rot.rot.pt"):
    p = POSTERIOR_PATH / f
    if p.exists():
        os.remove(str(p))
ok = in_box and finite and tuple(s.shape) == (16, nd_dim + len(cfg.rescale_params))
print(f"\nSMOKE: {'PASS' if ok else '*** FAIL ***'}")
print("REPARAM_WIRING_SMOKE_DONE", flush=True)
