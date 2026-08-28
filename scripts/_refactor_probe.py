"""Bit-identity probe for the refactor. TEMPORARY -- deleted in the refactor's final commit.

Prints a SHA-256 per output tensor of the pipeline's hot path, fully seeded, tiny, CPU-only.
Run it before and after every structural commit; every hash must match byte-for-byte. Any
mismatch means the "pure move" changed arithmetic, RNG consumption, or tensor layout, and the
commit must not land.

Seeds numpy AND torch: ``inits`` comes from ``np.random.randint``, which ``torch.manual_seed``
does not touch, so a torch seed alone makes every bit-identity claim vacuous (measured: two runs
of identical code differed by max|diff| 30.6 until numpy was seeded too). That is also why
``smoke_train.py`` cannot serve as this instrument.

Coverage: ``gen_training_data`` in all three modes (which drives ``gen_obs``, ``gen_stats``,
``gen_chi_raw``/``gen_chi_block`` and the forcing builder end to end), plus direct calls to
``gen_stats``, ``build_nondim_sin_force_tensor``, ``winsorize_summary_block`` and
``count_pathological`` so their extractions are pinned in isolation. ``gen_prior`` is
deliberately NOT probed: even a tiny stability sweep costs minutes and can legitimately find no
clusters; its extraction is a pure move gated by its signature/body test and the
``orchestrator.pipeline.gen_prior`` monkeypatch tests. ``train_nn`` is not probed: sbi's fit
loop is not run-to-run deterministic and its extraction is gated by the suites.

Usage (from the repo root):
    KMP_DUPLICATE_LIB_OK=TRUE C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe scripts\\_refactor_probe.py
"""
import hashlib
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # orchestrator imports pyplot; never let a console run pick a GUI backend

import numpy as np
import torch

from core import cli, config, orchestrator, registry
from core.SBI import pipeline

SEED = 1234
_digests: list[str] = []


def _report(name: str, t: torch.Tensor) -> None:
    h = hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    _digests.append(f"{name}:{h}")
    print(f"PROBE {name:34s} shape={tuple(t.shape)!s:20s} dtype={t.dtype} sha256={h[:16]}")


def _cfg():
    model = "NADROWSKI"
    cfg = cli.make_sim_config(model, config.VALID_LABELS[config.VALID_MODELS.index(model)],
                              registry.state_dep_drift(model),
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()
    return cfg


class _FixedPrior:
    """Every draw is the ground-truth row: geometry-independent and free of prior RNG."""

    def __init__(self, theta):
        self.theta = theta

    def sample(self, shape):
        return self.theta.expand(shape[0], -1).clone()


def _probe_training_data(mode: str) -> None:
    cfg = _cfg()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_grid, steady_idx = 12_000, 500
    t = torch.linspace(0, n_grid * cfg.dt_nd_min, n_grid, dtype=cfg.hw.dtype)
    force_prior = orchestrator.build_forcing_prior(cfg) if mode == "forced" else None
    kw = dict(
        run_size=4, n_runs=2, steady_idx=steady_idx, dt_nd_min=cfg.dt_nd_min,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds, state_dep_drift=cfg.state_dep_drift,
        spontaneous_only=(mode == "spontaneous"), chi_mode=(mode == "chi"),
        n_vars=cfg.inits_tensor.shape[-1], dtype=cfg.hw.dtype, device=cfg.hw.device)
    if mode == "chi":
        kw.update(chi_f0=config.CHI_F0, chi_freq_bounds=config.CHI_FREQ_BOUNDS,
                  chi_k_pad=4, chi_max_cycles=config.CHI_MAX_CYCLES)
    x, th = pipeline.gen_training_data(
        cfg.model, _FixedPrior(cfg.ground_truth_tensor.reshape(1, -1)), force_prior, t, **kw)
    _report(f"gen_training_data[{mode}].x", x)
    _report(f"gen_training_data[{mode}].theta", th)


def _probe_components() -> None:
    cfg = _cfg()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_spont = torch.randn(6, 4000, dtype=torch.float32).cumsum(dim=-1) * 0.01
    stats = pipeline.gen_stats(x_spont, None, cfg.dt_exp, None, None, None,
                               device=cfg.hw.device, spontaneous_only=True)
    _report("gen_stats[spontaneous]", stats)

    n_forcing, n_rescale = len(cfg.forcing_idx), len(cfg.rescale_idx)
    forcing_params = torch.rand(5, n_forcing, dtype=torch.float64) + 0.1
    rescale_params = torch.rand(5, n_rescale, dtype=torch.float64) + 0.1
    t_nd = torch.linspace(0, 30.0, 2000, dtype=torch.float64)
    force = pipeline.build_nondim_sin_force_tensor(
        forcing_params, t_nd, rescale_params, cfg.forcing_idx, cfg.rescale_idx)
    _report("build_nondim_sin_force_tensor", force)

    block = torch.randn(400, 60, dtype=torch.float32) * torch.linspace(0.1, 5.0, 60)
    wins = pipeline.winsorize_summary_block(block.clone(), n_summary=50, pct=(0.001, 0.999))
    _report("winsorize_summary_block", wins)

    patho = torch.randn(8, 100, dtype=torch.float32)
    patho[1] = float("nan")
    patho[3] = 2.5
    patho[5] = 1e16
    acc = {"rows": 0, "nonfinite": 0, "constant": 0, "overflow": 0}
    pipeline.count_pathological(patho, acc)
    counts = torch.tensor([acc["rows"], acc["nonfinite"], acc["constant"], acc["overflow"]])
    _report("count_pathological", counts)


def main() -> None:
    torch.set_num_threads(1)  # rule out thread-count-dependent CPU reduction orders across runs
    print(f"[probe] torch {torch.__version__}  numpy {np.__version__}  seed {SEED}")
    _probe_components()
    for mode in ("spontaneous", "forced", "chi"):
        _probe_training_data(mode)
    combined = hashlib.sha256("\n".join(_digests).encode()).hexdigest()
    print(f"PROBE COMBINED sha256={combined}")


if __name__ == "__main__":
    main()
