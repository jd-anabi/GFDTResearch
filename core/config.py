"""
Configuration constants, device detection, and data carriers for the SBI pipeline.
"""
import os
from dataclasses import dataclass, field, replace
from collections import OrderedDict
from functools import cached_property, lru_cache
from pathlib import Path

import torch

# === DEVICE DETECTION ===
@dataclass
class DeviceConfig:
    """Hardware configuration: device, dtype, and batch size."""
    device: torch.device
    dtype: torch.dtype
    batch_size: int

def detect_device() -> DeviceConfig:
    """Detect the best available compute device and set dtype / batch size accordingly."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        major, minor = torch.cuda.get_device_properties(dev).major, torch.cuda.get_device_properties(dev).minor
        if (major, minor) < (8, 0):
            dev = torch.device("cpu")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    dtype = torch.float32

    if dev.type == "cuda" and dtype == torch.float32:
        batch_size = 2 ** 11
    elif dev.type == "cuda" and dtype == torch.float64:
        batch_size = 2 ** 10
    else:
        batch_size = 2 ** 6

    return DeviceConfig(device=dev, dtype=dtype, batch_size=batch_size)


def cpu_device() -> DeviceConfig:
    """
    Force a CPU DeviceConfig.

    Used by the FDT and parameter-sweep branches. Their Euler-Maruyama solver is a
    sequential Python time loop over small ensembles (M ~ 256, state dim 3-5), so
    each step is a handful of tiny tensor ops. On GPU this is kernel-launch-bound
    (per-step time is ~constant regardless of M) and benchmarks ~3.4x SLOWER than
    CPU at M=256; the CPU<->GPU crossover is near M ~ 4096, far above FDT ensemble
    sizes. SBI (large batch_size, huge simulation volume) is left on detect_device().
    """
    return DeviceConfig(device=torch.device("cpu"), dtype=torch.float32, batch_size=2 ** 6)


# Fraction of currently-FREE device memory a single simulation batch may plan to occupy. The rest
# absorbs PyTorch internals, allocator fragmentation, and the intermediates that are not counted in
# the caller's per-sample estimate. 0.6 matches the value the FDT campaigns have used all along.
CUDA_MEM_FRACTION = 0.6


def memory_budget_elements(device: torch.device, dtype: torch.dtype,
                           fraction: float = CUDA_MEM_FRACTION) -> int:
    """
    How many tensor ELEMENTS one batch may plan to hold on ``device``.

    On CUDA this reads the actually-free memory, so it adapts to whatever else is resident -- which
    matters on a desktop GPU where the compositor and browsers can hold 1-2 GB. CPU/MPS get fixed
    conservative caps.

    Originally FDT-only (core/FDT/campaigns.py); lifted here because the SBI path needs the same
    budget and was instead relying on CHUNK_LEN / N_ND_MAX, which are batch-size-blind STEP counts
    and so cannot bound bytes at all.
    """
    bytes_per_elem = 4 if dtype == torch.float32 else 8
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        # mem_get_info reports the DRIVER's view, in which every block PyTorch has cached counts as
        # used -- even the ones it has already freed and will hand straight back. Add that reusable
        # pool back, or the budget collapses as soon as the caching allocator has warmed up, and a
        # loop that re-plans against it degenerates to a batch of one (slow enough to look hung).
        reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        budget_bytes = int((free_bytes + max(0, reusable)) * fraction)
    elif device.type == "cpu":
        budget_bytes = 4 * 1024 ** 3   # 4 GB conservative cap for CPU
    else:
        budget_bytes = 1 * 1024 ** 3   # 1 GB for MPS / other
    return max(1, budget_bytes // bytes_per_elem)

# === PATHS ===
# Resources live at <repo-root>/Resources. The run scripts (run.bat/run.sh) cd to the repo root, so the
# cwd-relative form is correct in normal use; the __file__ fallback keeps paths valid if the app is ever
# launched from another directory (config.py is core/config.py, so parent.parent is the repo root).
_ROOT = Path(os.getcwd()) / "Resources"
if not _ROOT.exists():
    _ROOT = Path(__file__).resolve().parent.parent / "Resources"
CELL_PATH    = _ROOT / "Cells"
BOUNDS_PATH  = _ROOT / "Bounds"
UNITS_PATH   = _ROOT / "Units"
PRIOR_PATH   = _ROOT / "Priors"
POSTERIOR_PATH = _ROOT / "Posteriors"
PLOT_PATH    = _ROOT / "Plots"
# Training-data checkpoints (C-11). Its OWN directory, not a subfolder of Priors/ or Posteriors/:
# file_manager.list_dir walks recursively and the GUI's posterior picker keeps any *.pt that is not
# *.rot.pt, so a checkpoint shard under Posteriors/ would appear in the dropdown as a loadable
# posterior and fail an isinstance assert on selection.
CHECKPOINT_PATH = _ROOT / "Checkpoints"
# Observations persisted at INFERENCE time (section 11.6 guardrail 1). Amortized NPE has no
# observation when it is SAVED -- which is why `default_x` is None on posterior_08232026 and why
# the posterior behind those figures cannot be re-sampled from the artifacts alone. TSNPE needs
# one, so it is recorded where it first exists: at inference.
OBSERVATION_PATH = _ROOT / "Observations"
MODELS_PATH  = _ROOT / "Models"      # user-defined model definitions (see core/registry.py)

# === PARAMETER LABELS (for plotting) ===
HOPF_LABELS = [r"$\mu$", r"$\beta$", r"$\sigma_x$", r"$\sigma_y$"]
BP_LABELS = [r"$\tau_{hb}$", r"$\tau_m$", r"$\tau_{gs}$", r"$\tau_t$",
             r"$C_{min}$", r"$S_{min}$", r"$S_{max}$", r"$Ca^2_m$", r"$Ca^2_{gs}$",
             r"$U_{gs,\ max}$", r"$\Delta G$", r"$k_{gs, \text{ ratio}}$",
             r"$\chi_{hb}$", r"$\chi_a$", r"$x_c$", r"$\eta_{hb}$", r"$\eta_{a}$"]
NADROWSKI_LABELS = [r"$\kappa$", r"$\tilde{\lambda}$", r"$\varphi$", r"$\tilde{\tau}$", r"$\tilde{\tau}_c$",
                    r"$S$", r"$\Delta \tilde{G}$", r"$\beta$", r"$N$", r"$\tilde{T}$"]

VALID_MODELS = ["BP", "NADROWSKI", "HOPF"]
VALID_LABELS = [BP_LABELS, NADROWSKI_LABELS, HOPF_LABELS]

# === FORCING PARAMETER UNITS (single source of truth) ===
# SI unit per forcing-parameter name, used to convert an experimenter's values into cell-file units.
# "Hz" is SPECIAL-CASED at the conversion site: a drive frequency is INVERSE CELL TIME by construction
# (forcing.py evaluates sin(2*pi*freq*t_dim) with t_dim in cell time units), so it converts via
# SimConfig.freq_si_to_cell -- never by matching a declared frequency token, which resolves to 1.0
# against an `ms` cell. None = dimensionless (phase, in radians).
# The display map is DERIVED from this one so the prompt/label hints can never drift from the
# authoritative conversion table (they were separately maintained and had already started to).
FORCING_SI_UNITS = {"amp": "N", "amp_y": "N", "freq": "Hz", "phase": None, "offset": "N"}
FORCING_DISPLAY_UNITS = {n: ("rad" if u is None else u) for n, u in FORCING_SI_UNITS.items()}

# === ENSEMBLE CONSTANTS ===
UNIQUE_FREQS = 2 ** 6
K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

# === EXPERIMENTAL CONSTANTS (in seconds, converted to cell file units during setup) ===
DT_EXP_S = 1e-3        # 1000 FPS camera frame interval
T_MIN_EXP_S = 1.0      # shortest expected recording (1 s)
T_MAX_EXP_S = 60.0     # longest expected recording (1 min)

# === SIMULATION COST CONSTANTS ===
# CUDA Graphs for the Euler-Maruyama step loop. ~88% of solver wall-clock was CPU kernel-LAUNCH
# overhead: measured on the real Nadrowski step, 54.87 us/step eager against 6.65 us/step replayed
# from a captured graph at batch 2048 (8.25x), 7.80x at 8192. The solver is the whole run -- training
# generation alone is ~97% of a production retrain -- so this is the single largest lever in the
# project. Read LIVE via `config.SOLVER_CUDA_GRAPHS` (never `from .config import`, which snapshots),
# so a test or a debugging session can turn it off in-process.
#
# Set False to force the eager TorchScript loop. Behaviour is otherwise identical; the graph path
# falls back to eager on its own if capture fails for any reason.
SOLVER_CUDA_GRAPHS = True
SOLVER_GRAPH_CHUNK = 50     # Euler steps captured per graph. Amortises replay overhead without
                            # making the captured region large enough to matter for memory: the
                            # static output block is (CHUNK, batch, d) = ~1.2 MB at batch 2048.
SOLVER_GRAPH_CACHE_MAX = 8  # distinct (step, shape, dt) graphs kept alive. The pipeline uses one
                            # width plus the OOM halving ladder, so ~5 in practice. Graph memory
                            # lives in a PRIVATE pool that torch.cuda.empty_cache() cannot reclaim,
                            # which is why this is bounded rather than unlimited.
CHUNK_LEN = 100_000    # fine integration steps per segment (per-chunk memory cap)
N_ND_MAX = 300_000     # max total fine integration steps per batch (pre-filter ceiling)
PPC_BIN_SIZE = 50      # samples per mini-batch for posterior-predictive-check simulation
# --- SBC calibration batching -------------------------------------------------------------------
# These two are INDEPENDENT knobs and it matters which one you turn.
#
# The SDE solver is a kernel-launch-bound sequential time loop, so a batch of 10 and a batch of 256
# cost the SAME wall-clock. Calibration cost is therefore driven by the NUMBER OF BATCHES, not the
# number of samples:   sim cost  ~  CAL_N_SCALES x (1 + K in chi mode)
#                      samples   =  CAL_N_SCALES x cal_run_size      (run_size is nearly free)
#                      SBC cost  ~  n_cal                            (run_sbc draws + check_sbc C2ST)
#
# CAL_N_SCALES is also the (t_scale, T) DIVERSITY count: gen_training_data draws one Sobol pair per
# batch and OVERRIDES t_scale to it for every row, so all rows in a batch share one t_scale truth and
# their SBC ranks are not independent. t_scale's effective sample size is CAL_N_SCALES, NOT n_cal.
# Lowering it buys wall-clock at the direct expense of the parameter chi(omega) mode exists to pin.
#
# Defaults below reproduce the historical behaviour EXACTLY at SBC_N_CAL=2000 (200 pairs x 10), so
# results stay comparable with the keeper posterior's K=10 x n_cal=2000 characterization. To spend
# the free GPU capacity, raise SBC_N_CAL and leave CAL_N_SCALES alone (200 x 64 = 12800 samples costs
# the same simulation time as today, only more downstream SBC). To make a chi-mode validate tractable
# -- it pays CAL_N_SCALES x (K+1) simulations, ~7x at K=6 -- lower CAL_N_SCALES and say so in the
# write-up, because that is a different measurement, not a faster one.
CAL_N_SCALES = 200     # (t_scale, T) pairs per calibration set == number of batches == sim cost
CAL_RUN_SIZE = 10      # FLOOR on samples per pair (the historical fixed value)
CAL_RUN_SIZE_MAX = 256 # ceiling on samples per pair; the solver is flat in batch size up to ~2048
SBC_N_CAL = 2000       # calibration datasets for SBC in validate(). n_cal=1000 was under-powered:
                       # the K=10 repeat study (scripts/sbc_characterize.py) showed mild marginal
                       # miscalibration only surfaces reliably at n_cal>=2000 (KS power grows with n_cal).
TRAINING_NUM_RUNS = 5000  # number of (t_scale_k, T_k) batches per training round (data budget)

# --- Training-data checkpointing (C-11) -----------------------------------------------------------
# Write a resumable checkpoint every N training batches; 0 disables it entirely.
#
# Deliberately NOT sharing pipeline._MEM_LOG_EVERY (250). That one is a DISPLAY volume, tuned to give
# ~21 [mem] lines over a 5000-batch run; tying durability to it means a later decision to log less
# often silently multiplies the crash-loss window. Two meanings, two knobs.
#
# The cost model at 5000 x 2048 rows, chi width 114 (+13 latent targets), i.e. the retrain's shape:
#     per checkpoint   50 x 2048 x 127 x 4 B          ~=  52 MB written, well under 1 s on NVMe
#     overhead         against 50 batches x ~20 s     ~=  under 0.1 % of wall-clock
#     expected loss    half an interval on a crash    ~=  8 minutes of simulation
#     whole run        100 checkpoints                ~=  4.9 GiB on disk
# Anything from 25 to 100 is defensible; 50 sits in the flat part of both curves. Checkpointing every
# batch is 100x the write volume to buy back 8 minutes on a multi-day run, which is not a trade.
TRAINING_CHECKPOINT_EVERY = 50

TRAINING_RUN_SIZE = 0   # CEILING on simulations per training batch; 0 = follow DeviceConfig.batch_size.
                        #
                        # DEFAULTS TO OFF, and should normally stay off. Batch width is nearly free in
                        # wall-clock -- the SDE solver is a kernel-launch-bound sequential time loop, so
                        # a batch costs about the same whatever its width (measured on a 5070 Ti at
                        # n_fine=100k: 7.37 s at 2048 against 7.74 s at 1024, i.e. the SMALLER batch is
                        # slightly slower). Lowering this therefore does NOT speed anything up; it trades
                        # training rows for peak VRAM at roughly 1:1, and TRAINING_NUM_RUNS has to rise to
                        # compensate, which DOES cost wall-clock proportionally.
                        #
                        # It is an ESCAPE HATCH, not the memory fix. The memory fix is per-geometry: a
                        # training batch's cost is width x n_fine, and n_fine swings from a median ~40k to
                        # a p99 ~283k, so at a fixed width the tail is ~7x the median. pipeline's
                        # _max_sim_batch already sizes each batch against its OWN geometry, and its
                        # learned budget plus _gen_obs_retry handle the tail by SPLITTING -- which costs
                        # k x wall-clock on the few percent of batches that need it rather than on all of
                        # them. Reach for this only if the retry notices show splitting on a large
                        # fraction of batches, i.e. the card is tighter than the split machinery can
                        # absorb.
                        #
                        # A CEILING, NOT A REPLACEMENT (unlike PRIOR_SWEEP_BATCH, which replaces).
                        # scripts/smoke_train.py and three pipeline tests shrink a run by writing
                        # cfg.hw.batch_size directly; a replacing knob would override them and quietly
                        # drive the CPU test suite at this width -- landing as "the tests got slow", not
                        # as an error. The asymmetry is principled: the prior sweep is iteration-bounded,
                        # so a LARGER batch there is free accuracy and worth allowing; training is never
                        # helped by a batch wider than the hardware default, since that is the thing that
                        # OOMs.

# === NEURAL POSTERIOR & TRAINING HYPERPARAMETERS ===
# Capacity / convergence knobs for the SBI posterior. Raise the flow capacity and/or the
# training budget to address broad SBC under-calibration; defaults match sbi's own.
DENSITY_ESTIMATOR = "nsf"                # flow family: "nsf" (neural spline flow) or "maf"
NSF_HIDDEN_FEATURES = 128                 # hidden units per flow transform (sbi default 50)
NSF_NUM_TRANSFORMS = 8                   # number of flow transforms (sbi default 5)
NSF_NUM_BINS = 10                        # spline bins per transform, NSF only (sbi default 10)
TRAINING_NUM_ROUNDS = 1                  # 1 = amortized NPE; >1 = sequential NPE near the observation
TRAINING_BATCH_SIZE = 512                # density-estimator minibatch size
TRAINING_LEARNING_RATE = 1e-3            # Adam learning rate (sbi default)
TRAINING_STOP_AFTER_EPOCHS = 20          # early-stopping patience in epochs (sbi default)
TRAINING_MAX_NUM_EPOCHS = 2_147_483_647  # hard epoch cap (sbi default: effectively unbounded)
TRAINING_SHOW_SUMMARY = True             # print sbi's train/validation-loss summary (check convergence)

# === PROGRESS BARS ===
# The per-time-segment bar (core/Simulator/simulator.py) wraps segs in {1,2,3} -- a three-step bar that
# tells a user nothing, while nesting a whole extra level under the training-data bar.
# core.gui.app.build_app() sets this True; the CLI and scripts/ never touch it, so `python -m core`
# renders exactly the bars it always has.
# Read this through the MODULE (`from core import config; config.QUIET_SEGMENT_BAR`) -- a
# `from core.config import QUIET_SEGMENT_BAR` snapshots the value at import and would freeze it False.
QUIET_SEGMENT_BAR = False

# The SDE solver's per-step bar (core/Solvers/sdeint.py) is f"{SOLVER_BAR_DESC} (batch={batch_size})".
# The GUI finds it by this desc prefix -- keyed on the DESC, never on the row, because the bar's tqdm
# `pos` is 0, 1 or 2 depending on which phase and which panel is running.
#
# WHAT THE GUI DOES WITH IT: exclude it, and only that. Its total is in the tens of thousands, so it
# would win the overall bar's election every time and sweep it 0->100% every second (trap S3). It is
# NOT the "Solver Performance" meter's source any more -- that number comes from core.progress.SOLVER,
# because a solver call shorter than its own bar's mininterval never paints a rate at all, which is
# exactly what CUDA graphs made happen (§10.5).
#
# The bar nevertheless stays ENABLED under the GUI, unlike QUIET_SEGMENT_BAR above: its redraws are the
# cooperative cancel's most frequent checkpoint and feed the stall detector's heartbeat through a long
# batch. See core/gui/widgets/progress_pane.py.
SOLVER_BAR_DESC = "step"

# === DECORRELATING REPARAMETERIZATION (Track A: flow calibration via latent rotation) ===
# When the inferred params are well-identified but strongly correlated (e.g. kappa~x_scale at
# |cos|=0.95), the flow mis-calibrates the thin diagonal ridge. Rotating the flow's latent
# coordinate into the simulation-based Fisher eigenbasis makes that posterior axis-aligned so the
# flow can calibrate it -- no information loss, no model/stats change. REPARAM_ROTATE=False (V=I)
# is exactly the current pipeline, so the rotation is fully optional and model-agnostic.
REPARAM_ROTATE = True   # True = rotate into the Fisher eigenbasis; False = plain pipeline.
REPARAM_FISHER_M = 48    # ensemble per latent-perturbation for the simulation-based Fisher estimate.
REPARAM_FISHER_DZ = 0.1  # latent-space central-difference step for the Fisher Jacobian.
# Operating points (GT + prior draws) over which the simulation Fisher is AVERAGED to build the
# rotation V. >1 makes the single linear rotation valid prior-wide, not just at GT (a GT-only V
# re-correlates the curved degeneracies off-GT). 1 = GT-only (the original behavior).
REPARAM_FISHER_POINTS = 8

# === LOG-SPACE BOX (linearize the multiplicative degeneracies before rotating) ===
# ND/rescale params (by cell-file key) whose box bijection is GEOMETRIC (log) instead of linear.
# In log coords the products kappa*x_scale (amplitude) and lambda*t_scale (timescale) become SUMS,
# so the single linear Fisher rotation can decorrelate them across the whole prior. Only params with
# a strictly positive lower bound are eligible (others fall back to linear with a warning). Empty
# list = pure linear box (legacy). The chosen mask is persisted beside each posterior (<name>.rot.pt)
# so eval reconstructs the exact training box regardless of this setting. REBUILD the ND prior after
# changing this (the latent GMM is fit in the box's coordinate).
REPARAM_LOG_PARAMS = []   # ALL-LINEAR box (the keeper posterior_07012026's coordinate). Log-scaling
                          # f_scale (REPARAM_LOG_PARAMS=["f_scale"]) was TRIED as a fix for its mild
                          # linear-box SBC tilt (GT=10 at box-fraction 0.009 = flat sigmoid tail; see
                          # scripts/diagnose_fscale.py), but the posterior trained under it was WORSE --
                          # bad TARP / expected-coverage and a worse f_scale SBC rank -- so it was
                          # discarded and this was reverted to []. Keep the DEGENERACY params
                          # (k, lam, x_scale, t_scale) LINEAR too (log OVER-MIXED those in posterior_6302026).
                          # f_scale is a RESCALE param, so toggling it here does NOT rebuild the ND prior:
                          # nd_log_mask stays all-False, and the existing linear ND prior
                          # (prior_forcing_no_forcing.pt) + posterior_07012026 already match this box.

# === CONDITIONING REPAIR (PRISM_HANDOFF section 11.3) ============================================
# Knots in the per-channel rank-Gaussian standardizer EmbeddedNet fits over the summary block.
# The transform IS the (knot, probit) pair, so this is its resolution: 1024 knots put the finest
# quantile step at ~0.1%, which resolves every point mass measured on the 10.24M-row cache (the
# smallest flagged one is E2_log_h2 at 2.8%) with two decades of margin, for 42x1024 floats.
RANK_GAUSS_KNOTS = 1024

# Per-column winsorisation of the SUMMARY BLOCK before the flow sees it, replacing train_nn's global
# `abs(data) < 1e15` ROW filter. A row filter is the wrong instrument: one pathological channel threw
# away all 114 of that row's values, and at 1e15 it caught 10 rows in 10.24M while A1_mean still
# reached -1.7e29 -- three decades of outlier under the threshold, which is what dragged its fitted
# std to 4.19e11.
#   ⚠ THE SUMMARY BLOCK ONLY, NEVER THE CHI BLOCK. A pad slot is exactly 0.0 in all six channels and
#   is required to be BITWISE inert (section 3.6, with a test). Clipping a probe column whose 0.1th
#   percentile is non-zero would move that 0.0 and silently turn every pad into a phantom probe.
WINSOR_PCT = (0.001, 0.999)

# === MULTI-FREQUENCY SUSCEPTIBILITY chi(omega) MODE (breaks the information ceiling) ===
# When CHI_MODE is on, the forced conditioning is a K-frequency susceptibility CURVE chi(omega)
# instead of a single-frequency Group-G lock-in. Per observation the drive is K SINGLE-TONE
# recordings at omega_k = CHI_FREQ_BOUNDS-spaced multipliers * Omega_0, where Omega_0 is the
# spontaneous-oscillation peak measured from the passive trace (mirrors the FDT pipeline's
# data-driven grid; see core/FDT/spectral.gen_freqs_log / find_spectral_peak). Each chi(omega_k)
# enters as [log|chi|, cos(arg chi), sin(arg chi)] -> 3K features, routed through the EmbeddedNet's
# second pathway (forcing_dim = 3K). A single passive trace only sees the products D*A_nd (amplitude)
# and (lambda_hb/k_gs)*tau_nd (timescale); the chi(omega) SHAPE over frequency separates
# kappa/lambda/x_scale/t_scale INDIVIDUALLY -- the only lever on the information ceiling + the
# x_scale location bias (KEEPER CAVEAT 1). CHI_MODE=False = the exact current pipeline
# (single-frequency forcing, or spontaneous-only), so this is fully optional and additive.
CHI_MODE = False
CHI_N_FREQS = 6                # K: number of single-tone drive frequencies (recordings) per observation.
CHI_FREQ_BOUNDS = (0.03, 0.3)  # log-spaced multipliers of the measured spontaneous peak Omega_0 spanned
                               # by the K-frequency grid (mirrors FDTConfig.freq_bounds).
                               #
                               # SUB-RESONANCE ONLY, and that is a MEASUREMENT, not a preference.
                               # scripts/chi_f0_sweep.py on the master cell (M=24 seeds), sweeping drive
                               # amplitude against probe frequency, found |chi| reproducible ONLY below
                               # ~0.25x Omega_0:
                               #     0.05x  CV 0.026     0.1x  CV 0.029     0.2x  CV 0.055   (usable)
                               #     0.3x   CV 0.22      0.5x  CV 0.21      0.7x  CV 0.47    (not)
                               #     1x / 2x / 10x: CV 0.36-0.73 at EVERY amplitude tried (0.01 .. 0.3)
                               # and -- the decisive part -- the high-multiplier CV does NOT improve from
                               # T_obs 5 s to 25 s. A noise-limited lock-in would fall by sqrt(5) ~ 2.2x;
                               # it does not move. So that variability is SYSTEMATIC, not statistical:
                               # same theta, different noise seed, genuinely different chi. Neither a
                               # stronger drive nor a longer recording can recover those probes.
                               #
                               # The old (0.1, 10.0) put 8 of 10 probes at K=10 in that regime -- each
                               # costing a full simulation per observation. That is the direct explanation
                               # for posterior_chi_08042026 (archived): flat SBC and a clean PPC, because
                               # the flow correctly learned those features carry nothing, while every ND
                               # marginal stayed at the prior.
                               #
                               # OPEN: the sub-resonance branch is close to the static compliance, so it may
                               # carry chi's MAGNITUDE (x_scale/f_scale, already well identified) without the
                               # SHAPE that was supposed to separate kappa/lambda -- the shape lives near and
                               # above resonance, which is exactly the unusable region. Check with
                               # scripts/degeneracy_map.py before spending another training run.
CHI_K_MAX = 24         # upper bound on CHI_K_PAD accepted by the GUI -- a CAPACITY knob. It used to
                       # bound K itself; under the set layout K is a property of an OBSERVATION and is
                       # bounded by the pad, not by this.

# === chi(omega) SET CONDITIONING (layout 2) ===
# The chi block is a PADDED SET of probes, not a fixed 3K grid. Probe j occupies pad slot j as six
# channels (u, log|chi|, cos, sin, logcyc, mask); the probe's FREQUENCY is carried explicitly in
# channel 0 rather than being implied by its slot index. That is the whole point: the number of
# probes and where they sit in frequency both become free, so a bench session that achieved 7
# recordings at whatever frequencies it could manage conditions the same trained network as a
# simulated 12-probe sweep.
#
# CHI_K_PAD IS FROZEN INTO EVERY ARTIFACT. sbi's reshape_to_batch_event bakes condition_shape into the
# saved posterior, so raising it later invalidates every chi posterior -- which is why the sidecar
# records it and the load path refuses a mismatch (a message, not a shape assert hours into a run).
# The encoder's parameter count does NOT depend on it (phi/rho are per-element and pooled), so a
# generous pad costs only 6*K_PAD input columns. Choose once.
CHI_LAYOUT = 2         # layout version, written to the sidecar. 1 = the retired fixed-3K grid.
CHI_ELEM_W = 6         # channels per pad slot. A LITERAL -- never derive it from the channel tuple.
CHI_K_PAD = 12         # pad capacity -> block width 72, conditioning width 42 + 72 = 114
CHI_K_MIN_TRAIN = 2    # floor of the per-batch probe-count draw
CHI_MIN_CYCLES = 2.0   # a probe is MASKED (never moved, never dropped) below this many drive cycles
                       # inside the segment it was locked in over. A lock-in over a fraction of a cycle
                       # returns the demeaned trace's residual drift plus spontaneous 1/f content:
                       # finite, in range, and REPRODUCIBLE -- which is exactly why it survived the
                       # chi_f0_sweep CV screen at 0.05x. It is not a susceptibility. 2.0 rather than a
                       # larger floor because measurement refuses one: at T=5s, mult=0.03 the probe has
                       # 3.39 cycles and the BEST |chi| CV in the sweep (0.024), so a floor of 8 would
                       # delete the best probe in the experiment.
CHI_MAX_CYCLES = 20.0  # CEILING on the drive cycles a probe is locked in over. The counterpart to
                       # CHI_MIN_CYCLES above, and the less obvious of the two: every instinct about
                       # integration says a longer lock-in is a better one, and above ~30 cycles on
                       # this model it is not. Measured 2026-08-06 (handoff 4.3.1 / trap CHI9): at
                       # FIXED theta, |chi| CV runs 0.03 -> 0.63 and driven/undriven SNR 26 -> 2.3 as
                       # the window grows past the wall, and re-locking the SAME trace over a shorter
                       # prefix reverses it completely. A stationary noise-limited estimator cannot do
                       # that, so the response is non-stationary on the scale of tens of drive cycles
                       # and the lock-in accumulates that wander instead of averaging it away.
                       # This is NOT a filter: no probe is masked or dropped by it. It shortens the
                       # SEGMENT the lock-in runs over, which is a property of the measurement, so it
                       # lives in gen_chi_raw where every caller -- training, the Fisher rotation, the
                       # PPC and the experimental path -- goes through it. Applying it in one caller
                       # would make the network condition on a different observable than it was
                       # trained on, which is silent.
                       # WHY 20. scripts/chi_f0_sweep.py brackets the wall by re-locking the same
                       # traces over every prefix length (M=48, in-band probes only so frequency
                       # effects cannot confound it). Worst |chi| CV by cap:
                       #   8 -> 0.042   12 -> 0.039   16 -> 0.047   20 -> 0.062
                       #   24 -> 0.086  28 -> 0.123   32 -> 0.198   36 -> 0.456  (first failure)
                       # A steady climb, not a cliff, so there is no "correct" value -- only a
                       # trade-off. 20 sits in the flat part with ~3x margin to the 0.2 CV screen and
                       # 10x above CHI_MIN_CYCLES, and it is the value the 4.3.1 rescue table already
                       # validated end-to-end on every failing point. 12-16 reproduce slightly better;
                       # they were not chosen because NOTHING here measures the other side of the
                       # trade -- a shorter lock-in is also less frequency-selective, and no
                       # experiment in this repo has yet priced that.
                       # It is frozen into the artifact for the same reason the band is: a posterior
                       # trained at one ceiling and evaluated at another sees different logcyc values
                       # for the same recording. The sidecar carries it and the load path checks it.
CHI_UHAT_MAX = 1.25    # band-normalised |u_hat| beyond which a probe is masked (packer) or refused
                       # (experimental path). Replaces clamping, which silently moved probes.
# Encoder geometry. Functions of CHI_ELEM_W and design choice ONLY -- never of CHI_K_PAD, or the
# parameter count would change with the pad and no two pads could share a checkpoint.
CHI_PHI_DIM = 64
CHI_BIN_DIM = 16
CHI_SET_OUT = 64
CHI_RHO_HIDDEN = 128
CHI_KNOTS = (-1.0, 0.0, 1.0)   # fixed band-normalised quadrature knots (Nadaraya-Watson)
CHI_KNOT_SIGMA = 0.6
CHI_KNOT_SHRINK = 0.5          # shrinkage of an under-covered knot toward zero
CHI_F0 = 0.15                  # ND drive amplitude for every chi probe. Driving at a FIXED ND amplitude
                               # (dimensional amp = CHI_F0 * f_scale, which build_nondim divides back to
                               # CHI_F0) keeps the lock-in SNR uniform across the f_scale prior, and models
                               # an experimentalist who scales the physical drive to the cell. chi =
                               # redimensionalized response / dimensional drive, so it still carries
                               # x_scale/f_scale.
                               #
                               # CHOSEN BY MEASUREMENT, not by a linearity argument. An ACTIVE (spontaneously
                               # oscillating) bundle has NO clean linear-response regime near its own
                               # frequency -- a weak drive is not "more linear", it is simply swamped by the
                               # spontaneous oscillation, and the lock-in then measures noise divided by a
                               # small number. The criterion that matters is REPRODUCIBILITY: chi must be a
                               # stable function of theta, not of the noise seed. Measured on cell_2 at
                               # T_obs ~ 8 s, M = 8 seeds (|chi| coefficient of variation):
                               #     ND 0.05 -> 0.21 at Omega_0, 0.17 at 0.3x, but 0.62 at 3x  (unusable
                               #                high-frequency probes -- and the grid runs to 10x)
                               #     ND 0.2  -> 0.04 at Omega_0, 0.04 at 0.3x, 0.17 at 3x       (usable
                               #                everywhere; retains a ~10x |chi| range across frequency,
                               #                which IS the shape information the flow conditions on)
                               #     ND >=0.5 -> even steadier, but entrainment saturates |chi| (9.2 -> 1.4
                               #                at Omega_0 from 0.05 -> 1.0), compressing its theta-dependence
                               #
                               # RE-MEASURED 2026-08-05 on the master cell (scripts/chi_f0_sweep.py), over
                               # the SUB-RESONANCE band this grid now spans. Two bounds, not one:
                               #   too small -> CV rises (0.05x: CV 0.090 at F0=0.05 vs 0.026 at F0=0.15)
                               #   too large -> the drive ENTRAINS the bundle, which abandons its own rhythm
                               #                and follows the drive, so chi reports the drive back to
                               #                itself. Onset at 1.4x detune is F0 = 0.2 -- the OLD default.
                               # 0.15 is the largest amplitude that is still reproducible everywhere in the
                               # band while leaving the bundle running free (own peak >= 84% of undriven at
                               # every probe from 0.05x to 0.2x). TUNABLE per config in the Config tab;
                               # re-measure for a cell with a very different Q or noise level.

# Cycles of the observation's own oscillation shown in the time-domain posterior-overlay figures. The
# window is derived per observation from its measured peak frequency, so this stays meaningful whatever
# t_scale is: enough cycles to judge frequency and waveform, few enough that individual cycles are legible.
EYE_TEST_CYCLES = 15

# === TRANSIENT (Case A: clip initial conditions settling) ===
TRANSIENT_ND_UNITS = 100  # ND time units of transient to discard; ~20 e-folds of the slowest
                          # bounded mode (tau_c up to ~5.0) in ND Nadrowski cell files.

# === PRIOR STABILITY SCREENING ===
STABILITY_SWEEP_ND_UNITS = 1000  # ND time units used to screen parameter stability during
                                # prior construction (global + local sweeps). Short enough
                                # to be cheap, long enough for instabilities to manifest.

PRIOR_SWEEP_ITERATIONS = 50     # sweep ROUNDS inside gen_prior's global stability map. Total
                                # candidates screened = PRIOR_SWEEP_BATCH x this, and each round pays
                                # a full STABILITY_SWEEP_ND_UNITS trajectory whatever the batch is --
                                # so rounds cost wall-clock and batch costs memory. Was a bare literal
                                # at the gen_prior call site until C-7.
                                # (Prior.construct_prior passes batch*iterations down as `batch_size`,
                                # so the subclasses' `batch_size % num_iterations` guard is vacuous by
                                # construction -- do not rely on it to catch a bad value here.)

PRIOR_SWEEP_BATCH = 0           # candidates per prior sweep; 0 = follow HardwareConfig.batch_size
                                # (the historical behaviour, and still the right default -- the sweep
                                # wants the largest batch that fits).
                                #
                                # IT EXISTS BECAUSE SHARING hw.batch_size WITH TRAINING IS A TRAP.
                                # That one number used to drive both, so shrinking it for a quick run
                                # made the PRIOR worse without making it faster: the sweep is
                                # iteration-bounded, so a smaller batch runs the same 50 rounds and
                                # merely accepts fewer points each. Measured (backlog C-7): 527 s at
                                # batch 2048, versus >70 min and STILL UNFINISHED at batch 32. Set
                                # this only to bound prior-sweep MEMORY; to make a smoke run cheap,
                                # shrink the training batch instead and leave this at 0.

@lru_cache(maxsize=1)
def unit_registry():
    """The process-wide pint UnitRegistry.

    Constructing one parses pint's full unit-definition file (~100-300 ms). Every config builder and
    every diagnostic script parses units at least once per cell, and cli._parse_cell /
    cli._units_to_factors used to mint a fresh registry on each call. Quantities from different
    registries cannot be combined, so a single shared instance is also the safer arrangement.
    """
    import pint
    return pint.UnitRegistry()


# === SIMULATION CONFIG DATACLASS ===
@dataclass
class SimConfig:
    """
    Carries all state needed to run the SBI pipeline.
    Replaces the 9-element tuple that was threaded through setup() -> run().
    """
    # Model selection
    model: str
    labels: list[str]
    state_dep_drift: bool

    # Parsed from cell file
    inits_dict: OrderedDict # {name: val}
    params_dict: OrderedDict # {name: (val, (lo, hi))}
    rescale_params: OrderedDict # {name: (val, (lo, hi))}
    force_params_dict: OrderedDict # {name: (val, (lo, hi))}
    units_dict: tuple
    # DEPRECATED and unread. si_factors is built by iterating units_dict, which is SET-derived and
    # therefore has non-deterministic order, so it could never be safely indexed -- and nothing in the
    # repo reads it. Kept only so the existing constructor calls (cli + 8 scripts) keep working; use
    # get_unit_conversion_factor / freq_si_to_cell / the *_unit properties instead, which match by
    # DIMENSION rather than position. Do not add new readers.
    si_factors: list[float]

    # Multi-frequency susceptibility mode: replace the single-frequency forced conditioning with a
    # K-frequency chi(omega) curve (see config.CHI_MODE). Threads like has_forcing; set at build time
    # from config.CHI_MODE. Default False = the existing single-frequency / spontaneous pipeline.
    # The three knobs are carried ON THE CONFIG (not read from the module at use time) so a config --
    # and therefore a posterior trained from it -- is SELF-DESCRIBING: changing the global afterwards
    # cannot silently reinterpret an existing run. default_factory reads the module value LIVE at
    # construction, so the CLI still picks up config.CHI_* edits without passing anything.
    chi_mode: bool = False
    # chi_n_freqs is now "probes this OBSERVATION supplies" -- it no longer sets any width. chi_k_pad
    # is the NETWORK's slot capacity and is what every width is derived from, so a posterior trained
    # with K drawn over 2..12 loads against a config at chi_n_freqs = 4 or 9 with no guard loosened.
    chi_n_freqs: int = field(default_factory=lambda: CHI_N_FREQS)
    chi_k_pad: int = field(default_factory=lambda: CHI_K_PAD)
    chi_f0: float = field(default_factory=lambda: CHI_F0)
    chi_freq_bounds: tuple = field(default_factory=lambda: CHI_FREQ_BOUNDS)
    # The lock-in duration ceiling, in drive cycles. On the config for the same reason as the band:
    # it changes the logcyc a given recording reports, so a posterior trained under one ceiling and
    # evaluated under another is reading a different observable.
    chi_max_cycles: float = field(default_factory=lambda: CHI_MAX_CYCLES)

    # Decorrelating Fisher rotation, carried per-config for the same reason as the chi knobs: consumers
    # used to `from .config import REPARAM_ROTATE`, which snapshots at import and cannot be toggled at
    # runtime. Reading it from the config makes a trained posterior self-describing about whether the
    # rotation was intended (the <name>.rot.pt sidecar stores the resulting V).
    reparam_rotate: bool = field(default_factory=lambda: REPARAM_ROTATE)

    # Time / segmentation (legacy fallback fields; primary time setup uses dt_exp + T_obs)
    t_max: float = None
    dt: float = None

    # Experimental observation parameters (in cell file time units, set during setup)
    dt_exp: float = None          # camera frame interval
    t_min_exp: float = None       # shortest expected recording
    t_max_exp: float = None       # longest expected recording
    T_obs: float = None           # ground-truth observation duration (user input)
    # Resolved observation length in SAMPLES, written by orchestrator.generate_observations after any
    # cost-ceiling clipping. Downstream consumers (PPC in infer_and_visualize, the overlay figures)
    # MUST read this rather than recomputing int(T_obs/dt_exp): the two expressions are algebraically
    # equal but not numerically, and the ceiling branch writes T_obs back as N_obs*dt_exp, whose
    # re-truncation can land on N_obs-1. A one-sample disagreement there silently deleted all five
    # posterior-overlay figures. None => generate_observations has not run (the experimental paths,
    # which take their length from the recording itself).
    n_obs: int = None

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)

    def __post_init__(self):
        """Reject a chi geometry that cannot be represented, AT CONFIG BUILD.

        K > chi_k_pad has no valid packing, and the natural failure without this is a raise from
        deep inside pack_probe_block on the FIRST batch -- i.e. after the prior has been built and
        the training loop has started. Both numbers are named so the message says which knob to move.
        """
        if not self.chi_mode:
            return
        if not (2 <= int(self.chi_k_pad) <= CHI_K_MAX):
            raise ValueError(
                f"chi_k_pad={self.chi_k_pad} is out of range: it must be at least 2 and at most "
                f"CHI_K_MAX={CHI_K_MAX}. It is the network's probe-slot capacity and is frozen into "
                f"every posterior trained with it.")
        if not (1 <= int(self.chi_n_freqs) <= int(self.chi_k_pad)):
            raise ValueError(
                f"chi_n_freqs={self.chi_n_freqs} probes cannot be packed into chi_k_pad="
                f"{self.chi_k_pad} slots. Lower the probe count, or raise the pad (which invalidates "
                f"posteriors trained at the current pad).")
        # The floor masks a probe; the ceiling shortens it. If they cross, the ceiling truncates
        # every probe to under the floor and the packer masks the entire set -- an all-masked
        # observation, which the experimental path refuses outright. That would surface as "every
        # probe was masked" with nothing pointing at the two constants that closed on each other.
        if not (float(self.chi_max_cycles) > CHI_MIN_CYCLES):
            raise ValueError(
                f"chi_max_cycles={self.chi_max_cycles} does not clear CHI_MIN_CYCLES={CHI_MIN_CYCLES}. "
                f"The ceiling truncates a probe's lock-in and the floor masks it below that many "
                f"cycles, so a ceiling at or under the floor masks every probe in every observation.")

    # --- Derived properties ---
    @property
    def t_scale_bounds(self) -> tuple[float, float]:
        """(lo, hi) bounds on the t_scale rescaling parameter (λ/K_gs)."""
        _, (lo, hi) = self.rescale_params["t_scale"]
        return lo, hi

    @property
    def dt_nd_min(self) -> float:
        """Finest ND time step needed: dt_exp / t_scale_max."""
        _, t_scale_hi = self.t_scale_bounds
        return self.dt_exp / t_scale_hi

    @property
    def t_nd_max(self) -> float:
        """Longest ND duration needed: t_max_exp / t_scale_min."""
        t_scale_lo, _ = self.t_scale_bounds
        return self.t_max_exp / t_scale_lo

    @cached_property
    def t(self) -> torch.Tensor:
        """
        Pre-simulated ND time vector at finest resolution and longest duration.

        Cached: SimConfig is effectively immutable after build_sim_config(), so we
        allocate the 2.4M-point tensor once per config lifetime.
        """
        if self.dt_exp is not None:
            n_steps = int(self.t_nd_max / self.dt_nd_min)
            return torch.linspace(0, self.t_nd_max, n_steps,
                                  dtype=self.hw.dtype, device=self.hw.device)
        # fallback for legacy usage
        return torch.linspace(0, self.t_max, int(self.t_max / self.dt),
                              dtype=self.hw.dtype, device=self.hw.device)

    @property
    def steady_idx(self) -> int:
        """
        Index where transient ends and steady-state begins.

        Fixed number of fine integration steps corresponding to TRANSIENT_ND_UNITS
        ND time units — model-intrinsic, independent of prior bounds on T or t_scale.
        """
        steady_idx = int(TRANSIENT_ND_UNITS / self.dt_nd_min)
        # Safety check: transient must leave budget for at least the minimum output batch
        assert steady_idx < N_ND_MAX, (
            f"TRANSIENT_ND_UNITS={TRANSIENT_ND_UNITS} produces steady_idx={steady_idx} "
            f">= N_ND_MAX={N_ND_MAX}. Reduce TRANSIENT_ND_UNITS or raise N_ND_MAX."
        )
        return steady_idx

    @property
    def has_ground_truth(self) -> bool:
        """True once ground-truth VALUES are loaded (a bounds-built config has None value slots)."""
        rows = list(self.params_dict.values()) + list(self.rescale_params.values())
        return len(rows) > 0 and all(row[0] is not None for row in rows)

    def _require_ground_truth(self) -> None:
        if not self.has_ground_truth:
            raise ValueError(
                "SimConfig was built from a bounds file (no ground-truth values). Load a cell file via "
                "inject_ground_truth(...) before generating a synthetic observation, or use experimental data."
            )

    @property
    def ground_truth(self) -> list[float]:
        """Ground-truth values for all inferred params (ND + rescale). Requires a loaded cell."""
        self._require_ground_truth()
        nd = [row[0] for row in self.params_dict.values()]
        rescale = [row[0] for row in self.rescale_params.values()]
        return nd + rescale

    @property
    def ground_truth_tensor(self) -> torch.Tensor:
        return torch.tensor(self.ground_truth, dtype=self.hw.dtype, device=self.hw.device)

    @property
    def nd_params_bounds(self) -> list[tuple]:
        """Parameter bounds for prior construction."""
        return [row[1] for row in self.params_dict.values()]

    @property
    def inits_tensor(self) -> torch.Tensor:
        """Initial conditions as a (1, n_vars) tensor. A bounds-built config has no inits until a cell loads."""
        if not self.inits_dict:
            raise ValueError(
                "SimConfig has no initial conditions (built from bounds + units only). Load a ground-truth "
                "cell file via inject_ground_truth(...) before generating an observation."
            )
        return torch.tensor(list(self.inits_dict.values()), dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @property
    def params_tensor(self) -> torch.Tensor:
        """ND-only ground-truth parameters as a (1, n_params) tensor for the simulator. Requires a loaded cell."""
        self._require_ground_truth()
        nd = [row[0] for row in self.params_dict.values()]
        return torch.tensor(nd, dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @staticmethod
    def _fill_checked(label: str, cell_vals: dict, cfg_dict: OrderedDict, check_bounds: bool) -> list:
        """Validate a cell values dict against a config (val,(lo,hi)) dict, then fill in the values.

        MISSING is fatal -- the bounds file declares a parameter the cell cannot supply, and a None left
        in slot 0 would crash later in params_tensor. EXTRA cell values are IGNORED and returned, so the
        caller can report them: the BOUNDS file is the single source of truth for which parameters are
        inferred, so a cell carrying more than the bounds declare is merely over-specified. That is what
        lets one cell serve several bounds files -- e.g. a forced cell (f_scale + a Forcing section) used
        with a spontaneous bounds file that declares neither. cli._merge_vals_bounds already drops
        bounds-absent params exactly this way; being strict only here made the two paths disagree.
        """
        missing = sorted(set(cfg_dict) - set(cell_vals))
        if missing:
            raise ValueError(
                f"Cell file is missing {label} required by the bounds file: {missing}."
            )
        if check_bounds:
            oob = [f"{n}={cell_vals[n]} not in ({lo}, {hi})"
                   for n, (_, (lo, hi)) in cfg_dict.items() if not (lo <= cell_vals[n] <= hi)]
            if oob:
                raise ValueError(f"Cell file {label} outside the bounds file's bounds: " + "; ".join(oob))
        for n in cfg_dict:
            cfg_dict[n] = (cell_vals[n], cfg_dict[n][1])
        return sorted(set(cell_vals) - set(cfg_dict))

    def inject_ground_truth(self, inits: dict, param_vals: dict,
                            rescale_vals: dict, forcing_vals: dict) -> list:
        """
        Fill ground-truth VALUES + initial conditions from a cell file into a bounds-built config.

        SAFEGUARD: every ND and rescale (inferred) parameter the BOUNDS file declares must be present in
        the cell and lie within its bounds — else a clear ValueError listing the offenders. Forcing is the
        known DRIVE (conditioning, not an inferred param): it must be present, but its range is not
        enforced (a spontaneous cell legitimately uses amp=0/freq=0 outside the drive prior's range).

        Values the cell carries that the bounds file does NOT declare are IGNORED and returned, so the
        caller can note them — see _fill_checked. This is what lets one cell be used across the three
        observation modes (a forced cell against a spontaneous bounds file drops f_scale + the drive).

        :return: names of cell values that were ignored, tagged by section (may be empty).
        """
        ignored = [f"{n} (ND)" for n in
                   self._fill_checked("ND parameters", param_vals, self.params_dict, check_bounds=True)]
        ignored += [f"{n} (rescale)" for n in
                    self._fill_checked("rescale parameters", rescale_vals, self.rescale_params,
                                       check_bounds=True)]
        ignored += [f"{n} (forcing)" for n in
                    self._fill_checked("forcing parameters", forcing_vals, self.force_params_dict,
                                       check_bounds=False)]
        self.inits_dict = OrderedDict(inits)
        return ignored

    def set_observation_context(self, T_obs: float, forcing_vals: dict | None = None) -> None:
        """
        Set the observation duration (and optionally forcing VALUES) for the experimental-data branch,
        so PPC / eye-test simulators that read cfg.T_obs and cfg.force_params_dict values work.
        """
        self.T_obs = T_obs
        if forcing_vals is not None:
            for name, v in forcing_vals.items():
                if name in self.force_params_dict:
                    self.force_params_dict[name] = (v, self.force_params_dict[name][1])

    @property
    def inferred_labels(self) -> list[str]:
        """LaTeX labels (with units) for all inferred params (ND + rescale) for plotting.

        ND params keep their model LaTeX (self.labels); rescale params are rendered via
        Helpers.labels.rescale_axis_label so a corner/SBC plot shows e.g. ``$x_{\\mathrm{scale}}$ (nm/ND)``
        instead of the raw string ``x_scale``."""
        from .Helpers import labels as _labels
        rescale_labels = [
            _labels.rescale_axis_label(name, length_unit=self.length_unit,
                                       time_unit=self.time_unit, force_unit=self.force_unit)
            for name in self.rescale_params
        ]
        return self.labels + rescale_labels

    @property
    def has_forcing(self) -> bool:
        """Whether this model carries any external forcing parameters. False for a spontaneous model
        (a no-forcing user model, or BP whose bounds file has no forcing section). The SBI pipeline
        branches on this: a no-forcing config skips the forced run + Group-G lock-in and drops the
        forcing conditioning block, so its forcing_idx has none of amp/freq/phase to key on."""
        return len(self.force_params_dict) > 0

    @property
    def observation_mode(self) -> str:
        """Which of the THREE observation protocols this config describes.

          "spontaneous"  chi off, no drive     -- ONE passive trace. Groups A-F, Group G zero-padded;
                                                  conditioning [S(41) | log T]. f_scale is inert here
                                                  (it only ever divides a force) so it should not be in
                                                  the inferred set -- give such a cell a bounds file with
                                                  neither a Forcing section nor f_scale.
          "forced"       chi off, has_forcing  -- passive + ONE forced trace at the cell's own drive.
                                                  Conditioning [S(41) | log T | forcing].
          "chi"          chi on                -- passive + K single-tone forced traces. Conditioning
                                                  [S(41, G=0) | log T | chi(3K)]. The cell's own drive is
                                                  IGNORED (chi probes at mult_k * measured Omega_0), so
                                                  this is independent of has_forcing.

        The mode is chosen by which BOUNDS file is picked (has_forcing == "it declares a Forcing
        section") plus the chi toggle. The three conditioning widths cannot collide (K >= 2 is enforced),
        so loading a posterior trained in a different mode fails loudly on shape rather than silently.
        """
        if self.chi_mode:
            return "chi"
        return "forced" if self.has_forcing else "spontaneous"

    @property
    def forcing_idx(self) -> dict[str, int]:
        """Maps forcing param names to column indices, e.g. {"amp": 0, "freq": 1, ...}."""
        return {name: i for i, name in enumerate(self.force_params_dict.keys())}

    @property
    def rescale_idx(self) -> dict[str, int]:
        """Maps rescale param names to column indices, e.g. {"x_offset": 0, "x_scale": 1, ...}."""
        return {name: i for i, name in enumerate(self.rescale_params.keys())}

    @property
    def nd_idx(self) -> dict[str, int]:
        """Maps ND param names to column indices. The counterpart to rescale_idx/forcing_idx, added
        for the tier-1 constraint, which needs ``n`` and ``beta`` BY NAME -- a bounds file's ORDER is
        the source of truth for columns, so only its names are safe to reference."""
        return {name: i for i, name in enumerate(self.params_dict.keys())}

    @property
    def sim_rescale_idx(self) -> dict[str, int]:
        """The rescale index the SIMULATOR sees. Identical to ``rescale_idx`` unless this box declares
        ``T`` instead of ``f_scale``, in which case T's column is renamed to f_scale -- see
        core/SBI/derived.py, which owns that rule."""
        from core.SBI import derived
        return derived.sim_rescale_idx(self.rescale_idx)

    @property
    def tier1_args(self) -> tuple:
        """``(nd_idx, k_b_cell)`` for a tier-1 box; ``(None, None)`` for every other config.

        ⚠ LAZY ON PURPOSE, and it was NOT lazy first time round. ``k_b_cell`` needs a FORCE unit in
        the units file and raises without one -- and Python evaluates call arguments eagerly, so
        writing ``to_sim_rescale(..., cfg.nd_idx, cfg.k_b_cell)`` blew up on every config with no
        force token, even though that function returns early for exactly those. It surfaced as
        `test_no_forcing_user_model_full_sbi_pipeline` dying inside the Fisher rotation with
        "No unit with dimensionality [mass] * [length] / [time] ** 2 found in the units file."

        So: every caller passes ``*cfg.tier1_args`` and nothing evaluates the constant unless the box
        actually declares T.
        """
        from core.SBI import derived
        if not derived.uses_derived_f_scale(self.rescale_idx):
            return None, None
        return self.nd_idx, self.k_b_cell

    @property
    def k_b_cell(self) -> float:
        """Boltzmann's constant in CELL units: (cell force) x (cell length) per kelvin.

        Derived from the units file rather than hard-coded, for the same reason ``freq_si_to_cell``
        is: a constant baked for nm/pN silently mis-scales a cell declared in um/nN by nine orders of
        magnitude. For the nm/ms/pN/kHz master cell this is 1.380649e-2 pN*nm/K, so k_B*T at 300 K is
        4.14 pN*nm -- the number to sanity-check a derived f_scale against.
        """
        return (K_B * self.get_unit_conversion_factor("N")
                * self.get_unit_conversion_factor("m"))

    def get_unit_conversion_factor(self, si_unit: str) -> float:
        """
        SI unit -> cell file equivalent unit conversion factor.

        Finds which unit in the cell file has the same dimensionality as si_unit,
        and returns the multiplicative factor to convert from SI value to cell value.

        Examples:
          - get_unit_conversion_factor("s")  -> 1000.0 if cell uses ms
          - get_unit_conversion_factor("N")  -> 1e12 if cell uses pN

        NOTE: do NOT use this for a drive FREQUENCY -- use ``freq_si_to_cell``, which derives the
        factor from the TIME unit. See that property for why.

        :param si_unit: SI unit string (e.g. "s", "N", "rad").
        :return: Conversion factor: cell_value = si_value * factor.
        :raises ValueError: If no unit in the cell file matches the given dimensionality.
        """
        ureg = self._ureg          # cached -- a fresh UnitRegistry per call is expensive and this is hot
        target_dim = ureg.Quantity(1, si_unit).dimensionality
        for unit_str in self.units_dict:
            try:
                if ureg.Quantity(1, unit_str).dimensionality == target_dim:
                    return ureg.Quantity(1, si_unit).to(unit_str).magnitude
            except Exception:                  # noqa: BLE001 -- undefined token; skip
                continue
        raise ValueError(f"No unit with dimensionality {target_dim} found in the units file.")

    @property
    def freq_si_to_cell(self) -> float:
        """Drive-frequency conversion factor: ``freq_cell = freq_Hz * freq_si_to_cell``.

        Frequency is INVERSE CELL TIME *by construction*, not an independently declared unit:
        core/forcing.py builds ``t_dim`` in cell time units and evaluates ``sin(2*pi*freq*t_dim)``, and
        statistics.py / chi.py index their FFTs with ``dt`` in cell time units. So a 30 Hz drive in an
        ``ms`` cell is 0.03 cycles/ms, NOT 30.

        Deriving the factor from the TIME unit is what makes it correct regardless of what (if anything)
        the units file declares for frequency. Matching a declared "Hz" token instead resolves to 1.0
        against an ``ms`` cell and inflates every experimental drive by 1000x -- the bug this replaces.
        Use ``check_unit_consistency()`` to surface a units file whose frequency token disagrees.
        """
        return 1.0 / self.get_unit_conversion_factor("s")

    def check_unit_consistency(self) -> list[str]:
        """Human-readable warnings where the DECLARED units disagree with how the pipeline uses them.

        Units *declare* what the numbers in the bounds/cell files mean; they are never auto-converted.
        So a declaration that contradicts the pipeline's own convention silently mis-scales real data.

        Check: the declared FREQUENCY token must be the reciprocal of the declared TIME token (Hz with s,
        kHz with ms, ...), because drive frequency is consumed as inverse cell time (see freq_si_to_cell).
        """
        msgs: list[str] = []
        t_tok, f_tok = self.time_unit, self.freq_unit
        if t_tok and f_tok:
            try:
                declared = self._ureg.Quantity(1, f_tok).to(f"1/{t_tok}").magnitude
            except Exception:                  # noqa: BLE001 -- unconvertible pair; nothing to assert
                return msgs
            if abs(declared - 1.0) > 1e-9:
                msgs.append(
                    f"Units file declares frequency in '{f_tok}' but time in '{t_tok}'. The pipeline "
                    f"consumes drive frequency as INVERSE CELL TIME (1/{t_tok}), so a '{f_tok}' value is "
                    f"off by {1.0 / declared:g}x. Declare the reciprocal of the time unit (e.g. kHz for "
                    f"ms) or drop the frequency token entirely -- it is display-only, and conversions "
                    f"derive from the time unit."
                )
        return msgs

    @cached_property
    def _ureg(self):
        return unit_registry()

    def _resolve_unit(self, si_unit: str) -> "str | None":
        """The cell's unit TOKEN whose dimensionality matches ``si_unit`` (e.g. "s" -> "ms"), or None.

        units_dict is set-derived (unordered), so match by DIMENSIONALITY, never by index."""
        ureg = self._ureg
        try:
            target = ureg.Quantity(1, si_unit).dimensionality
        except Exception:                      # noqa: BLE001
            return None
        for tok in self.units_dict:
            try:
                if ureg.Quantity(1, tok).dimensionality == target:
                    return tok
            except Exception:                  # noqa: BLE001 -- undefined token; skip
                continue
        return None

    @cached_property
    def length_unit(self) -> "str | None":
        """Cell length unit token (e.g. "nm") for displacement axis labels."""
        return self._resolve_unit("m")

    @cached_property
    def time_unit(self) -> "str | None":
        """Cell time unit token (e.g. "ms"). Note: trace TIME axes are shown in seconds; this is only for
        the rescale-param label t_scale (ms/ND)."""
        return self._resolve_unit("s")

    @cached_property
    def force_unit(self) -> "str | None":
        """Cell force unit token (e.g. "pN"); None for BP, which declares no force unit.

        CAVEAT for a model with NO f_scale (Hopf): forcing.py then falls back to the Hopf-style nondim
        f_scale = x_scale / t_scale, so the EFFECTIVE force unit is length/time (nm/ms) regardless of
        what the units file declares. The declared token is still used to convert an experimental drive,
        so for such a model the declaration is a labelling convention, not a derived quantity."""
        return self._resolve_unit("N")

    @cached_property
    def freq_unit(self) -> "str | None":
        """Cell frequency unit token (e.g. "Hz"); None for BP."""
        return self._resolve_unit("Hz")


# === FDT CONFIG DATACLASS ===
@dataclass
class FDTConfig:
    """
    Carries all state needed to run the FDT analysis pipeline.
    Parallel to SimConfig but minimal: no prior/posterior/inference plumbing.
    """
    # Shared with SimConfig (model identity + parsed cell file)
    model: str
    state_dep_drift: bool
    inits_dict: OrderedDict           # {name: val}
    params_dict: OrderedDict          # {name: (val, (lo, hi))}
    rescale_params: OrderedDict       # {name: (val, (lo, hi))}
    force_params_dict: OrderedDict    # {name: (val, (lo, hi))}
    units_dict: tuple
    si_factors: list[float]

    # FDT-specific knobs (sensible defaults; overrideable in build_fdt_config)
    n_freqs: int = 60
    # Multipliers of omega_0 for the Campaign-2 production grid.
    # Asymmetric in log space by design: below = 1 decade, above = 1.5 decades
    # (=> ~50% more drive frequencies above omega_0, to capture FDT recovery
    # at the high-frequency end while still resolving the active band below).
    freq_bounds: tuple = (0.1, 30.0)
    ensemble_M: int = 256              # trajectories per Campaign-2 frequency
    freqs_per_batch: int = 1           # frequencies packed per simulator call in Campaign 2
    F0: float = 0.05                   # ND forcing amplitude (within linear regime)
    burn_in_nd: float = 100.0
    T_obs_periods: int = 30
    dt_nd: float = 0.01
    psd_T_obs_nd: float = 8000.0       # Campaign-1 steady-state duration

    # Filled in by run_fdt after cfg is built (from params_dict["k"])
    omega_0: float = None

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)

    # --- Derived ---
    @property
    def inits_tensor(self) -> torch.Tensor:
        """(1, n_vars) tensor of initial conditions."""
        return torch.tensor(list(self.inits_dict.values()),
                            dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @property
    def params_tensor(self) -> torch.Tensor:
        """(1, n_params) Nadrowski ND params."""
        nd = [row[0] for row in self.params_dict.values()]
        return torch.tensor(nd, dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    def params_for_M(self, M: int) -> torch.Tensor:
        """Tile ND params to shape (M, n_params) for ensemble batching."""
        return self.params_tensor.expand(M, -1).contiguous()

    def inits_for_M(self, M: int) -> torch.Tensor:
        """Tile initial conditions to shape (M, n_vars)."""
        return self.inits_tensor.expand(M, -1).contiguous()

    def with_overrides(self, **kwargs) -> "FDTConfig":
        """
        Return a shallow copy with overridden values.

        Keys may be:
          - ND parameter names from params_dict (overrides value, preserves bounds);
            used by passive-baseline sanity check (temp=1.0, tau_c=0.0)
          - any top-level FDTConfig field (n_freqs, F0, ensemble_M, ...)
        """
        nd_keys = set(self.params_dict.keys())
        top_kwargs = {k: v for k, v in kwargs.items() if k not in nd_keys}
        nd_kwargs = {k: v for k, v in kwargs.items() if k in nd_keys}

        new_params = OrderedDict(self.params_dict)
        for k, v in nd_kwargs.items():
            _, bounds = new_params[k]
            new_params[k] = (v, bounds)

        return replace(self, params_dict=new_params, **top_kwargs)
