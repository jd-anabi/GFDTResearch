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
# It stays ON under the GUI: its it/s IS the "Solver Performance" meter, and its percentage is the only
# thing that moves during a ~10s training iteration. The GUI does NOT render it as a progress row (a
# posterior build constructs 10k-30k of these bars, one per time segment) -- it feeds a dedicated widget,
# found by this desc prefix. Keyed on the DESC, never on the row: the bar's tqdm `pos` is 0, 1 or 2
# depending on which phase and which panel is running. See core/gui/widgets/progress_pane.py.
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
CHI_FREQ_BOUNDS = (0.1, 10.0)  # log-spaced multipliers of the measured spontaneous peak Omega_0 spanned
                               # by the K-frequency grid (mirrors FDTConfig.freq_bounds).
CHI_K_MAX = 24         # upper bound on K accepted by the GUI. Cost is linear in K (each probe
                       # is another full simulation per observation) and the Infer tab grows one
                       # file-picker row per probe frequency.
CHI_F0 = 0.2                   # ND drive amplitude for every chi probe. Driving at a FIXED ND amplitude
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
                               # So 0.2 balances lock-in SNR against saturation. TUNABLE per config in the
                               # Config tab; re-measure for a cell with a very different Q or noise level.

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
    chi_n_freqs: int = field(default_factory=lambda: CHI_N_FREQS)
    chi_f0: float = field(default_factory=lambda: CHI_F0)
    chi_freq_bounds: tuple = field(default_factory=lambda: CHI_FREQ_BOUNDS)

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
