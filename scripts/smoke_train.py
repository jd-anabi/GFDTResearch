"""
The pre-flight run: all five pipeline stages end-to-end at tiny sizes, non-interactively.

WHY THIS EXISTS. PRISM_HANDOFF 4.1 has called for "a smoke train before the long run" since the
retrain path was written, with no way to do one: `retrain_convergence.py` INHERITS its prior from an
existing posterior, and Resources/Posteriors/ is empty, so the very situation a smoke train is for --
about to spend days on the first posterior -- is the one situation that script cannot serve.

What this covers that the test suite does not: the suite pins contracts on stubs and tiny synthetic
inputs. This runs the REAL chain on the REAL bounds/cell files -- stability-screened prior build,
NPE training, SBC/TARP calibration, PPC and the eye test -- so plumbing that only appears when the
stages are wired together (mode widths, sidecar round-trip, the chi block's shape agreeing with the
network's input layer) fails here, in minutes, instead of hours into a multi-day run.

WHAT TO WATCH, beyond "it finished":

  * `chi: N/M probes masked` -- the count, not the presence. Some masking is by design (a probe under
    CHI_MIN_CYCLES on a short recording). The reference figure is ~37 % of TRAINING probes
    (PRISM_HANDOFF 4.3.6: 36.8 %, reproduced at 36.7 % after the OOM work); materially above that
    means something in the chi path regressed. Note the PPC's fraction is much higher by design and
    is a readout of the POSTERIOR, not of the probe machinery -- see 4.3.6 before reading it as a bug.
      HISTORICAL, and no longer true: this bullet used to say the duration ceiling "is keyed on the
      batch's fastest f_peak, so a batch spanning a wide range of f_peak gives its slowest rows
      proportionally fewer cycles". That WAS the defect -- it is what drove the 77 % masking this
      script first found -- and C-8 fixed it: the ceiling is now applied PER ROW ((B,) N_row) in
      gen_chi_raw. Trap CHI9: a scalar T_k anywhere in that path is now a regression.
  * out-of-distribution warnings from check_observation_in_distribution -- these are the guards that
    were invisible for months before _common.enable_warnings().
  * the mode banner. A width mismatch between the config and the trained net is exactly what this
    run exists to catch before the long one.

This is NOT a calibration measurement. SBC at these sizes has no power -- CAL_N_SCALES is t_scale's
effective sample size (trap X5) and it is tiny here. A flat rank histogram from this run means
nothing at all; only a CRASH means anything.

Env knobs (CELL / BOUNDS / MODEL / TOBS_S / CHI* are handled by _common.script_cfg):
  NUM_RUNS   training batches                                  (default 4)
  RUN_SIZE   simulations per batch                             (default 32)
  N_CAL      calibration datasets for SBC/TARP                 (default 40)
  EPOCHS     max training epochs                               (default 5)
  SEED       RNG seed                                          (default 0)
  SAVE       "1" to persist prior/posterior artifacts          (default 0 -- nothing touches disk)
  STAGES     comma list to run a subset, e.g. "prior,posterior"
             (default prior,posterior,validate,infer)
  CHECKPOINT "1" to exercise the C-11 training-data checkpoint into a fresh temp dir (default 0;
             see the note at the rebinding below for why OFF is the default and why you should
             nonetheless run it ONCE on the GPU before a record run)

Run (chi mode, the case this was written for):
  $env:CHI=1; & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/smoke_train.py
"""
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt

import _common
from core import config, orchestrator
from core.SBI.statistics import FEATURE_LABELS

_common.enable_warnings()

NUM_RUNS = int(os.environ.get("NUM_RUNS", "4"))
RUN_SIZE = int(os.environ.get("RUN_SIZE", "32"))
N_CAL = int(os.environ.get("N_CAL", "40"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
SEED = int(os.environ.get("SEED", "0"))
SAVE = os.environ.get("SAVE", "0") == "1"
CHECKPOINT = os.environ.get("CHECKPOINT", "0") == "1"
STAGES = [s.strip() for s in
          os.environ.get("STAGES", "prior,posterior,validate,infer").split(",") if s.strip()]


def _sink(title, fig):
    """Swallow every figure. A smoke run is about whether the stages COMPLETE, and plt.show() would
    block forever under a headless interpreter."""
    try:
        plt.close(fig)
    except Exception:
        pass


def main():
    torch.manual_seed(SEED)
    cfg = _common.script_cfg()
    print(f"[smoke] NUM_RUNS={NUM_RUNS} RUN_SIZE={RUN_SIZE} N_CAL={N_CAL} EPOCHS={EPOCHS} "
          f"SEED={SEED} SAVE={SAVE} CHECKPOINT={CHECKPOINT}")
    print(f"[smoke] stages: {STAGES}")
    if cfg.chi_mode:
        print(f"[smoke] chi ceiling: <= {cfg.chi_max_cycles:g} drive cycles per probe "
              f"(floor {config.CHI_MIN_CYCLES:g})")
    print(f"[smoke] conditioning width = {len(FEATURE_LABELS)} + 1 + "
          f"{orchestrator.expected_forcing_dim(cfg)} = "
          f"{len(FEATURE_LABELS) + 1 + orchestrator.expected_forcing_dim(cfg)}", flush=True)

    # Shrink the pipeline by rebinding the module constants the stages read. Done here rather than by
    # threading arguments because these are exactly the knobs the long run turns up, so a smoke run
    # differs from a record run in these values ONLY -- nothing else about the code path changes.
    # On ORCHESTRATOR, not on config: orchestrator does `from .config import TRAINING_MAX_NUM_EPOCHS`
    # at import, which snapshots the value -- rebinding config.X afterwards would change nothing and
    # the "smoke" run would quietly train to convergence. (Same class of trap as the old
    # `from .config import REPARAM_ROTATE`, Appendix A 2026-07-27.)
    orchestrator.TRAINING_NUM_RUNS = NUM_RUNS
    orchestrator.SBC_N_CAL = N_CAL
    orchestrator.TRAINING_MAX_NUM_EPOCHS = EPOCHS
    # Training-data checkpointing OFF by default, and this one is not merely tidiness. The checkpoint
    # directory is keyed on a digest of the config, and a COMPLETE checkpoint short-circuits generation
    # and returns its stored rows -- exactly right for the multi-day record run it exists for, and
    # exactly wrong here: the second smoke run of a given config would skip the simulation path this
    # script exists to exercise and report a cheerful pass without having run it.
    #
    # CHECKPOINT=1 turns it back on into a FRESH TEMP DIRECTORY, so the path is exercised and nothing
    # is ever reused. Worth doing on the GPU before a record run: the C-11 tests are CPU-only, and a
    # CPU test cannot catch a tensor on the wrong device -- a CPU-built probe grid meeting the CUDA
    # rotation matrix is what this script caught on 2026-08-12, an hour into the Fisher.
    if CHECKPOINT:
        import tempfile
        config.CHECKPOINT_PATH = pathlib.Path(tempfile.mkdtemp(prefix="prism_smoke_ckpt_"))
        orchestrator.TRAINING_CHECKPOINT_EVERY = max(1, NUM_RUNS // 2)
        print(f"[smoke] checkpointing ON into {config.CHECKPOINT_PATH} "
              f"(every {orchestrator.TRAINING_CHECKPOINT_EVERY} batches; a temp dir, so never reused)",
              flush=True)
    else:
        orchestrator.TRAINING_CHECKPOINT_EVERY = 0
    # NOT here: cfg.hw.batch_size is ALSO build_prior's stability-sweep batch (global_batch_size),
    # and the sweep is iteration-bounded, so shrinking it does not shorten the prior build -- it just
    # accepts fewer points per iteration and makes the prior WORSE for the same wall-clock. Applied
    # inside the posterior stage instead, where it means what it says.
    print(f"[smoke] prior sweep batch stays at {cfg.hw.batch_size} (hardware default); "
          f"RUN_SIZE={RUN_SIZE} applies from the posterior stage on.", flush=True)

    t0, done = time.time(), {}

    def _stage(name, fn):
        if name not in STAGES:
            print(f"[skip] {name}", flush=True)
            return None
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        out = fn()
        done[name] = time.time() - t
        print(f"[ok] {name} in {done[name]:.1f}s", flush=True)
        return out

    prior = _stage("prior", lambda: orchestrator.build_prior(
        cfg, None, True, save=SAVE, save_name="_smoke_prior" if SAVE else None, fig_sink=_sink))
    if prior is None:
        print("\n[smoke] nothing further to run without a prior.")
        return 0
    inferred_prior, force_prior = prior

    def _posterior():
        cfg.hw.batch_size = RUN_SIZE          # training/calibration batch only -- see the note above
        return orchestrator.build_posterior(
            cfg, inferred_prior, force_prior, None, True,
            save=SAVE, save_name="_smoke_posterior" if SAVE else None, fig_sink=_sink)

    post = _stage("posterior", _posterior)
    if post is None:
        print("\n[smoke] prior only; stopping before training.")
        return 0
    posterior, _diag = post

    _stage("validate", lambda: orchestrator.validate_calibration(
        cfg, posterior, inferred_prior, force_prior, fig_sink=_sink))

    def _infer():
        x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)
        want = len(FEATURE_LABELS) + 1 + orchestrator.expected_forcing_dim(cfg)
        assert obs_stats.shape[-1] == want, \
            f"observation width {obs_stats.shape[-1]} != the mode's {want}"
        assert torch.isfinite(obs_stats).all(), "the observation carries non-finite conditioning"
        orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim,
                                         show_truth=True, fig_sink=_sink)

    _stage("infer", _infer)

    print(f"\n[smoke] ALL STAGES COMPLETED in {time.time() - t0:.1f}s "
          f"({', '.join(f'{k} {v:.0f}s' for k, v in done.items())})")
    print("[smoke] This says the chain RUNS. It says nothing about calibration -- SBC at these "
          "sizes has no power (trap X5). Read the masked-probe and OOD warnings above.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        # A smoke run's whole purpose is to surface the failure, so print it and exit non-zero
        # rather than let a traceback scroll past a long simulation log unnoticed.
        traceback.print_exc()
        print("\n[smoke] *** FAILED *** -- the stage banner above says which stage.", flush=True)
        raise SystemExit(1)
