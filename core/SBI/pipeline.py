import contextlib
import math
import shutil
import sys
import warnings
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sbi.inference.posteriors import DirectPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from torch.distributions.transforms import Transform

from core import forcing as _forcing
from core.Helpers import helpers
from core import config
from core.config import CHUNK_LEN, N_ND_MAX
from .Priors import bp_prior, hopf_prior, nadrowski_prior
from core.Simulator import bp_simulator, nadrowski_simulator, hopf_simulator
from core.SBI import statistics, chi, reparam

VALID_SIMS: dict = {"bp":        bp_simulator.BPSimulator,
                    "nadrowski": nadrowski_simulator.NadrowskiSimulator,
                    "hopf":      hopf_simulator.HopfSimulator}

VALID_PRIORS: dict = {"bp":        bp_prior.BPPrior,
                      "nadrowski": nadrowski_prior.NadrowskiPrior,
                      "hopf":      hopf_prior.HopfPrior}

INIT_SHAPES: dict = {"bp":        (2, 3),
                     "nadrowski": (2, 1),
                     "hopf":      (2, 0)}

def build_nondim_sin_force_tensor(
    forcing_params: torch.Tensor,
    t_nd: torch.Tensor,
    rescale_params: torch.Tensor,
    forcing_idx: dict,
    rescale_idx: dict,
) -> torch.Tensor:
    """
    Build a batch of non-dimensional sinusoidal force tensors.

    Constructs F_dim(t_dim) = amp * sin(2pi * freq * t_dim + phase) + offset
    in dimensional space, then nondimensionalizes via
    F_nd = (F_dim - f_offset) / f_scale.

    :param forcing_params: Forcing parameter values, shape (batch, n_forcing).
    :param t_nd: Non-dimensional time vector, shape (T,).
    :param rescale_params: Rescaling parameter values, shape (batch, n_rescale).
    :param forcing_idx: Maps forcing param names to column indices in forcing_params,
                        e.g. {"amp": 0, "freq": 1, "phase": 2, "offset": 3}. If "amp_y"
                        is present, a second forcing channel is built sharing freq, phase,
                        and offset with the x-channel but using its own amplitude.
    :param rescale_idx: Maps rescale param names to column indices in rescale_params,
                        e.g. {"t_scale": 3, "t_offset": 2, "f_scale": 7, "f_offset": 6}.
                        If "f_scale" is absent (Hopf-style nondim), f_scale is derived
                        as x_scale / t_scale and f_offset is taken as 0 — both follow
                        algebraically from F_ND = F_dim / (l * omega_0) with l = x_scale
                        and 1/omega_0 = t_scale.
    :return: Non-dimensional force tensor, shape (batch, n_force_channels, T) where
             n_force_channels = 2 if "amp_y" in forcing_idx else 1.
    """
    # The math now lives in core/forcing.py (shared with the new step/triangular/exponential kinds);
    # kind="sin" is numerically identical to the original body here (pinned by a golden test).
    return _forcing.build_nondim_force_tensor(
        forcing_params, t_nd, rescale_params, forcing_idx, rescale_idx, kind="sin")


def _sim_class(model: str):
    """The Simulator class for a BUILT-IN model name, with a clear error for anything else (a user
    model leaking past the Simulate-only gate would otherwise surface as a bare KeyError)."""
    cls = VALID_SIMS.get(model.lower())
    if cls is None:
        raise ValueError(
            f"No simulator is registered for model '{model}' (valid: {list(VALID_SIMS)}). "
            "User-defined models are Simulate-only in this version.")
    return cls

# Fraction of the available pool one simulation batch may plan to use. Higher than the FDT default
# (0.6) because the estimate below counts the major tensors explicitly rather than guessing, and
# because splitting is EXPENSIVE here: the SDE solver is a sequential kernel-launch-bound time loop,
# so a batch of 256 costs the same wall-clock as a batch of 2048 (measured: ~22 s either way at
# n_fine=300k) and k chunks therefore cost k x the time. The guard exists to convert a hard OOM into
# a slowdown, so it should engage only when the batch genuinely will not fit.
_SIM_MEM_FRACTION = 0.85

# Smallest sub-batch the guard will plan. At batch 2048 this caps the slowdown at 8x; anything
# tighter is treated as "this geometry does not fit" and left to fail loudly, because grinding
# through a 5000-batch round at that width would take days.
_MIN_SIM_CHUNK = 256


# ── The LEARNED memory budget ─────────────────────────────────────────────────────────────────────
#
# WHY A LEARNED CAP EXISTS AT ALL: on Windows, the free-memory reading is a lie, and it is the input
# _max_sim_batch plans from. config.memory_budget_elements reads torch.cuda.mem_get_info(), and under
# WDDM the OS virtualises VRAM -- other processes' surfaces are EVICTABLE, so the driver reports them
# to you as free. Measured at one instant on a 16 GB RTX 5070 Ti with an ordinary desktop running
# (2026-08-10): mem_get_info said 15037 MiB free while nvidia-smi said 5814 MiB. Optimistic by 9.2 GiB,
# which is exactly the desktop.
#
# The planner then green-lights a batch the driver can only satisfy by EVICTING the browser, and
# returns cudaErrorMemoryAllocation only when eviction cannot keep up -- which is why that failure
# arrives as a raw driver AcceleratorError rather than torch.OutOfMemoryError, and why it lands hours
# into a run rather than immediately. No amount of tuning _SIM_MEM_FRACTION fixes a wrong input.
#
# So: stop treating the reading as authoritative and learn the real ceiling from OUTCOMES. AIMD, on
# the budget in ELEMENTS rather than on the batch width -- width is the wrong variable, because
# whether a batch fits is width x GEOMETRY and _max_sim_batch already handles geometry correctly (a
# 2048-row batch at n_fine=40k fits comfortably; the same width at 283k does not). Adapting width
# would fight the planner and oscillate; adapting its budget composes with it.
#
# Deliberately process-local and NOT persisted: the right cap depends on what else is on the card
# right now, which is a property of this run, not of the machine.
_BUDGET_CAP_ELEMENTS = None          # None = no learned cap yet; trust memory_budget_elements alone
_BUDGET_OOM_BACKOFF = 0.8            # on an OOM at N elements, cap to 0.8*N -- we KNOW N did not fit
_BUDGET_RECOVER_AFTER = 32           # consecutive clean batches before probing upward again
_BUDGET_RECOVER_STEP = 1.1           # ...and how far. Additive-ish increase, multiplicative decrease.
_budget_clean_runs = 0


def _budget_cap() -> float:
    """The learned element cap, or +inf before anything has failed."""
    return math.inf if _BUDGET_CAP_ELEMENTS is None else _BUDGET_CAP_ELEMENTS


def _budget_note_oom(elements: int) -> None:
    """Record that an allocation of ``elements`` did NOT fit, and tighten the cap below it."""
    global _BUDGET_CAP_ELEMENTS, _budget_clean_runs
    _budget_clean_runs = 0
    if elements <= 0:
        return
    cap = int(elements * _BUDGET_OOM_BACKOFF)
    _BUDGET_CAP_ELEMENTS = cap if _BUDGET_CAP_ELEMENTS is None else min(_BUDGET_CAP_ELEMENTS, cap)


def _budget_note_ok() -> None:
    """Record a clean batch; after enough of them, probe the cap upward.

    The recovery half matters as much as the backoff: a desktop that was holding 6 GB when the first
    OOM landed may have closed a browser since, and without this the run would stay throttled to that
    moment for days. It probes multiplicatively but is re-clamped by memory_budget_elements on every
    call to _max_sim_batch, so it can never climb past what the (optimistic) reading allows anyway --
    the cap only ever makes the plan MORE conservative than that reading, never less.
    """
    global _BUDGET_CAP_ELEMENTS, _budget_clean_runs
    if _BUDGET_CAP_ELEMENTS is None:
        return
    _budget_clean_runs += 1
    if _budget_clean_runs >= _BUDGET_RECOVER_AFTER:
        _budget_clean_runs = 0
        _BUDGET_CAP_ELEMENTS = int(_BUDGET_CAP_ELEMENTS * _BUDGET_RECOVER_STEP)


# ── Where are we? ─────────────────────────────────────────────────────────────────────────────────
# A training round is thousands of batches over hours, and its warnings and failures are one line
# each in a log the GUI shows WITHOUT the traceback (Worker.run keeps that for the error dialog). The
# 2026-08-11 chi retrain died with a bare "CUDA error: out of memory" and the only clue to where was
# that the preceding line happened to be gen_chi_block's mask warning -- which narrowed it to "inside
# the per-batch loop" and no further. One f-string per batch, microseconds against a ~20 s batch,
# removes that entire class of forensics.
#
# A module global rather than a parameter because the consumers are gen_chi_block and the two OOM
# retries, none of which has any business taking a batch index. Single-writer by construction (one
# gen_training_data at a time in this process -- BasePanel._running is class-level for exactly that
# reason), and the only consumer is a log string, so the worst a concurrent run could do is mislabel.
_BATCH_TAG = ""


def _batch_tag() -> str:
    """The batch currently being generated, or a neutral label outside gen_training_data."""
    return _BATCH_TAG or "simulation"


_MEM_LOG_EVERY = 250          # 21 lines over a 5000-batch run


def _log_memory(device: torch.device, tag: str) -> None:
    """One memory line, and RESET the peak so the next one describes the next interval.

    PEAK allocated is the number that predicts an OOM. An instantaneous reading taken between batches
    is always low, because the batch's big tensors are already gone by then -- so a series of those
    would look flat right up until the run died. The allocator maintains the peak unconditionally, so
    reading it is free, and resetting it turns the series into "worst batch in the last 250", which is
    the thing that trends upward before a card runs out.

    Reported-free is printed with its health warning attached: under WDDM other processes' evictable
    surfaces count as free -- measured 15037 MiB against nvidia-smi's 5814 on this machine (trap X6).
    """
    if device.type != "cuda":
        return
    free_b, total_b = torch.cuda.mem_get_info(device)
    cap = ("none" if _BUDGET_CAP_ELEMENTS is None
           else f"{_BUDGET_CAP_ELEMENTS * 4 / 2 ** 30:.2f} GiB")
    print(f"[mem] {tag}: peak allocated {torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f} GiB, "
          f"peak reserved {torch.cuda.max_memory_reserved(device) / 2 ** 30:.2f} GiB, "
          f"{free_b / 2 ** 30:.2f}/{total_b / 2 ** 30:.2f} GiB reported free (optimistic on Windows), "
          f"learned cap {cap}", file=sys.stderr, flush=True)
    torch.cuda.reset_peak_memory_stats(device)


_ZSCORE_CHECK_MAX_ROWS = 200_000


@contextlib.contextmanager
def _capped_zscore_check(max_rows: int = _ZSCORE_CHECK_MAX_ROWS):
    """Run sbi's ``warn_if_zscoring_changes_data`` on a SUBSAMPLE for the duration of this block.

    WHY IT HAS TO BE CAPPED AT ALL. append_simulations calls it UNCONDITIONALLY on the full training
    tensor (sbi/inference/trainers/npe/npe_base.py:189). At the chi retrain's size -- 5000 x 2048 =
    10.24M rows x 114 columns float32, 4.35 GiB -- it runs ``torch.unique(x, dim=0)``, then builds
    ``zx = (x - x.mean(0)) / x.std(0)``, then runs ``torch.unique`` AGAIN. That is >= 13 GiB, and
    since unique over dim=0 is a lexicographic sort of 10.24M rows by 114 keys it is also minutes of
    single-threaded work. On a 16 GB card it is a guaranteed OOM at the END of a multi-day generation
    run -- the worst possible moment, because nothing is checkpointed. It fires on ANY successful run,
    chi or not, and no retry can help because it is one indivisible allocation.

    WHY CAPPING RATHER THAN DISABLING. In chi mode the check is inapplicable: train_nn sets
    ``z_score_x="none"`` whenever the embedding owns its standardization, and sbi's own warning text
    ends "if you have already set z_score_x=False, this warning will still be displayed, but you can
    ignore it." But spontaneous and forced modes DO train under ``z_score_x="independent"``, where a
    collapse under z-scoring is a real finding worth hearing about. It is also a PROPORTION test with
    a 10 % tolerance (``duplicate_tolerance``), so a 200k-row sample answers it to well within its own
    resolution. STRIDED, not a prefix: rows are grouped by batch and each batch shares one
    (t_scale, T) stratum, so ``x[:200_000]`` would sample ~98 of 5000 strata while ``x[::n]`` spans
    all of them.

    PATCHED BY REBINDING IN EVERY MODULE THAT HOLDS THE NAME, not just in sbiutils. npe_base does
    ``from sbi.utils import ... warn_if_zscoring_changes_data``, so its call resolves against
    npe_base's OWN globals -- patching sbi.utils.sbiutils alone changes nothing. Rather than hard-code
    that module path (sbi has already moved it once: sbi.inference.snpe -> sbi.inference.trainers.npe)
    this finds every module whose attribute IS the original function object. A few ms, once per round,
    and it cannot rot.

    SCOPED AND RESTORING, including on an exception: the patch is live only across append_simulations,
    so nothing else in the process -- another panel, a script that imported sbi, a later round -- ever
    sees a monkeypatched sbi.
    """
    import sbi.utils.sbiutils as _sbiutils
    original = _sbiutils.warn_if_zscoring_changes_data

    def _capped(x, duplicate_tolerance: float = 0.1):
        stride = max(1, math.ceil(x.shape[0] / max_rows))
        # .contiguous() explicitly: unique would materialise the strided view anyway, and doing it
        # here keeps the copy's size obvious (<= max_rows x n_features) rather than implicit.
        return original(x[::stride].contiguous(), duplicate_tolerance)

    targets = []
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "warn_if_zscoring_changes_data", None) is original:
                targets.append(mod)
        except Exception:            # noqa: BLE001 -- a lazy module's __getattr__ must not break this
            continue
    try:
        for mod in targets:
            mod.warn_if_zscoring_changes_data = _capped
        yield
    finally:
        for mod in targets:
            mod.warn_if_zscoring_changes_data = original


def _is_oom(err: BaseException) -> bool:
    """Is this failure -- or anything it was raised FROM -- a device out-of-memory?

    THREE FORMS ARRIVE HERE AND THEY ARE NOT ONE CLASS:
      * ``torch.OutOfMemoryError``  -- PyTorch's caching allocator could not serve a request.
      * ``torch.AcceleratorError``  -- a RAW driver cudaErrorMemoryAllocation, from an allocation
        PyTorch does not own (a cuFFT plan, a cuBLAS workspace, the allocator's own cudaMalloc for a
        new segment) or, on Windows/WDDM, a lost eviction race against the desktop. This is the form
        the 2026-08 chi retrain died with and the form the 2026-07-28 cuFFT plan-cache leak produced.
      * either of the above wrapped in ``SimulationError`` by Simulator.__sols, which is how anything
        raised inside the solver actually reaches this module.

    ``torch.OutOfMemoryError`` and ``torch.AcceleratorError`` both derive DIRECTLY from RuntimeError
    and neither subclasses the other, so no single isinstance() covers both -- hence the message test
    as well as the type test. Being loose is the SAFE direction here: the only thing a caller does
    with a True is retry smaller, which costs a little wall-clock on a false positive, whereas a false
    negative costs the whole run. ``seen`` guards a cause cycle, which would otherwise hang.
    """
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if isinstance(err, torch.OutOfMemoryError):
            return True
        if "out of memory" in str(err).lower():
            return True
        err = err.__cause__ or err.__context__
    return False


def sim_keep_elements(n_fine: int, steady_idx: int, n_out: int) -> int:
    """Elements ONE simulated row keeps until gen_obs returns: the post-transient output copy."""
    return n_out * max(0, n_fine - steady_idx)


def peak_sim_elements(batch_size: int, n_fine: int, steady_idx: int, n_vars: int, n_ch: int,
                      n_out: int) -> int:
    """Device elements a simulation batch holds at its PEAK, for one geometry.

    PUBLIC because a front-end that wants to show a user what a batch costs must read the same
    formula the planner does. A second copy in the GUI would drift from `_max_sim_batch` the first
    time either is tuned, and a display that reassures you about a number the planner does not use is
    worse than no display.

    The (n_vars, T) solution buffer and the (n_ch, T) drive are live throughout. The solver's
    (seg, n_vars) buffer and the (n_out, T - steady_idx) copy are NOT concurrent with each other --
    the copy is taken after the last segment is released -- so the peak takes their max, not their
    sum. Summing over-counts by ~20% at the training geometry.
    """
    seg = min(n_fine, CHUNK_LEN)
    return (n_vars * n_fine + n_ch * n_fine
            + max(n_vars * seg, sim_keep_elements(n_fine, steady_idx, n_out))) * batch_size


def sim_memory_budget_elements(device: torch.device, dtype: torch.dtype) -> int:
    """The element budget `_max_sim_batch` actually plans against -- free-memory reading, the 0.85
    headroom fraction, and the LEARNED cap, all folded in.

    Public for the same reason as `peak_sim_elements`: a front-end showing "will this fit?" must
    compare against the planner's budget, not against a raw `mem_get_info` reading. ⚠ That reading
    still overstates free VRAM on Windows by roughly the size of the desktop (measured 15037 MiB
    against nvidia-smi's 5814), which is why the learned cap exists -- so treat anything derived from
    this as an UPPER bound on what is really available.
    """
    return min(config.memory_budget_elements(device, dtype, _SIM_MEM_FRACTION), _budget_cap())


def _max_sim_batch(batch_size: int, n_fine: int, steady_idx: int, n_vars: int, n_ch: int,
                   n_out: int, dtype: torch.dtype, device: torch.device) -> int:
    """
    Largest simulation batch whose major device tensors fit the free-memory budget.

    The (n_vars, T) solution buffer and the (n_ch, T) drive are live throughout. The solver's
    (seg, n_vars) buffer and the (n_out, T - steady_idx) copy are NOT concurrent with each other --
    the copy is taken after the last segment is released -- so the peak takes their max, not their
    sum. Summing over-counts by ~20% at the training geometry, which is enough to make the guard
    split batches that would have fit.

    The FULL result is resident no matter how the work is split, so it is reserved off the top;
    otherwise the plan is re-derived against an ever-shrinking pool and collapses into tiny chunks.

    Returns ``batch_size`` unchanged whenever the whole batch already fits, so the common case is
    untouched and the split path costs nothing to have.
    """
    if device.type != "cuda" or batch_size <= 1:
        return batch_size
    n_keep = sim_keep_elements(n_fine, steady_idx, n_out)         # per sample, held until we return
    per_chunk_sample = peak_sim_elements(1, n_fine, steady_idx, n_vars, n_ch, n_out)
    if per_chunk_sample <= 0:
        return batch_size
    budget = sim_memory_budget_elements(device, dtype)
    if per_chunk_sample * batch_size <= budget:
        return batch_size            # the whole batch fits; splitting would only cost wall-clock
    # It does not fit. Now the previous chunks' results ARE extra, so reserve the full output.
    budget -= n_keep * batch_size
    if budget < per_chunk_sample * _MIN_SIM_CHUNK:
        # Not even a floor-sized chunk fits alongside the result. Splitting cannot rescue this, and
        # grinding through it a handful of rows at a time takes hours -- run as asked and fail loudly.
        return batch_size
    fits = max(_MIN_SIM_CHUNK, min(batch_size, int(budget // per_chunk_sample)))
    if fits >= batch_size:
        return batch_size
    # Quantize DOWN to a power of two. The free-memory estimate drifts a little between calls, so an
    # exact quotient yields a different chunk size almost every time -- and the solver specializes on
    # the batch dimension, so each new size pays a fresh compile. Powers of two collapse a whole run
    # onto a handful of shapes and divide the (power-of-two) training batch evenly, leaving no odd
    # remainder chunk. Never rounds up, so the budget still holds.
    return max(_MIN_SIM_CHUNK, 1 << (int(fits).bit_length() - 1))


def gen_obs(model: str, params: torch.Tensor, t: torch.Tensor, inits: torch.Tensor, force: torch.Tensor,
            n_segs: int, steady_idx: int, fixed_dict: dict = None, state_dep_drift: bool = False,
            batch_size: int = 1, var_idx: int | None = None,
            dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
    """
    Generates observations based on specified simulation type, parameters, and other input data.

    This function initializes a simulator based on the chosen simulation type and configuration. It
    validates the batch size of input tensors and ensures that the simulation type is supported.
    The specified simulator is used to simulate observations, and the processed observation data
    is returned.

    :param model: The type of model to use. Must be one of ["bp", "nadrowski", "hopf"].
    :param params: Tensor containing simulation parameters. The first dimension must match the given batch size.
    :param t: Tensor specifying the time points for the simulation. Its data type and device are set during processing.
    :param inits: Tensor containing initial conditions for the simulation. The first dimension must match the batch size.
    :param force: Tensor specifying the forces acting during the simulation.
    :param n_segs: The number of segments in the simulation. Used for configuration of the simulator.
    :param steady_idx: The index representing steady-state time points for slicing simulation results.
    :param fixed_dict: Dictionary of fixed parameters for the model.
    :param state_dep_drift: Whether to use state-dependent drift for the simulator.
    :param batch_size: Number of simulation batches to process. Default is 1.
    :param var_idx: If given, copy out ONLY this state variable, returning shape (1, batch, steady
        time points) instead of (n_vars, ...). Pure memory: the solution buffer is n_vars deep and
        the copy below has to coexist with it, so at the training batch size cloning all channels
        for a caller that only ever reads ``[0, :, :]`` doubles the peak of the largest allocation
        in the pipeline. Leading dim is kept so ``[0, :, :]`` indexes the same variable either way.
        Default None preserves the full multi-variable contract.
    :param dtype: Data type of tensors during processing. Default is `torch.float32`.
    :param device: The device on which simulations are run, such as "cpu" or "cuda". Default is "cpu".

    :return: Tensor containing simulated observations after processing using the selected simulator. Shape: (number of variables, batch size, steady state time points), or (1, batch, ...) when ``var_idx`` is given.
    :rtype: torch.Tensor

    :raises ValueError: If the batch size of input tensors does not match the first dimension of the parameters tensor or initial conditions tensor.
    :raises ValueError: If the specified model is not supported.
    """
    if params.shape[0] != batch_size or inits.shape[0] != batch_size:
        raise ValueError(f"Batch size: {batch_size} cannot differ from dim 0 of parameters tensor or initial conditions tensor")

    from core import registry
    if VALID_SIMS.get(model.lower()) is None and not registry.is_user_model(model):
        raise ValueError(f"Invalid simulator: {model}")

    # --- memory guard: split the batch if this geometry would not fit ---
    # The simulator's tensors are all linear in the batch: the (n_vars, batch, T) solution buffer,
    # the solver's per-segment (seg, batch, n_vars) buffer, the (batch, n_ch, T) drive, and the
    # copy taken at the end. CHUNK_LEN / N_ND_MAX bound STEPS, not bytes, so they cannot see this --
    # and a run sweeps (t_scale, T) over a wide range, so a few percent of batches are far larger
    # than the median and are what actually exhausts the card. Splitting over the batch is safe:
    # rows are independent, and params/inits/force are all row-indexed, so one slice keeps them
    # aligned. It does re-draw the SDE noise in smaller blocks, which is distributionally identical
    # (still iid) but not bit-reproducible against an unsplit run.
    max_b = _max_sim_batch(batch_size, t.shape[0], steady_idx, inits.shape[-1],
                           force.shape[1] if force.dim() > 2 else 1,
                           1 if var_idx is not None else inits.shape[-1], dtype, device)
    if max_b < batch_size:
        # PREALLOCATE the stitched result and have every chunk write STRAIGHT into its slice.
        #
        # This used to collect chunks in a list and torch.cat them. That cat is a DOUBLE RESIDENCY: at
        # the moment it allocates, every chunk is still live, so a 2.29 GiB result costs 4.58 GiB --
        # and it runs ONLY on split batches, i.e. exactly the ones already known not to fit. Wrong
        # direction, and it sat outside every handler so it could not even be retried.
        #
        # It does NOT make the plan more optimistic: _max_sim_batch already reserves the full
        # n_keep * batch_size off the budget for the whole split (the `budget -= ...` line above), so
        # it has always assumed the result is resident throughout. This makes that true. Measured
        # peaks at run_size=2048 / n_fine=280k (result 2.29 GiB, solver buffer 6.87 GiB):
        #     split k     list+cat     this
        #        2          5.73       5.73   (equal -- at k=2 the solver, not the cat, binds)
        #        4          4.58       4.01
        #        8          4.58       3.15
        # torch.empty rather than zeros: on CUDA it touches nothing, and every element is written.
        n_out = 1 if var_idx is not None else inits.shape[-1]
        out = torch.empty((n_out, batch_size, max(0, t.shape[0] - steady_idx)),
                          dtype=dtype, device=device)
        for s in range(0, batch_size, max_b):               # plain range: the tqdm nest is already 4 deep
            e = min(s + max_b, batch_size)
            _gen_obs_retry(
                model, params[s:e], t, inits[s:e],
                force[s:e] if force.shape[0] == batch_size else force,
                n_segs, steady_idx, fixed_dict, state_dep_drift, e - s, var_idx, dtype, device,
                out=out[:, s:e, :])
        # Counted as CLEAN even though it split. What _budget_note_ok tracks is "no OOM happened",
        # not "no split happened" -- a predictive split is the guard working, not failing. Keying
        # recovery on un-split batches instead would deadlock the cap: after one OOM tightens it, the
        # tighter cap makes _max_sim_batch split every subsequent batch, so no batch would ever be
        # counted clean and the cap could never climb back for the rest of the run.
        _budget_note_ok()
        return out
    out = _gen_obs_retry(model, params, t, inits, force, n_segs, steady_idx, fixed_dict,
                         state_dep_drift, batch_size, var_idx, dtype, device)
    _budget_note_ok()
    return out


def _gen_obs_retry(model, params, t, inits, force, n_segs, steady_idx, fixed_dict,
                   state_dep_drift, batch_size, var_idx, dtype, device, out=None):
    """One simulation batch, halving and re-running if it hits an out-of-memory.

    THE COMPANION TO _max_sim_batch, NOT A REPLACEMENT. That guard is PREDICTIVE, and on a shared card
    a prediction cannot be made reliable: it budgets from a free-memory reading that Windows overstates
    by the size of the desktop (see the learned-budget block above), and even where the reading is
    honest it is stale by the time the allocation lands. So the plan is a HINT and this is the
    MECHANISM: fail, shed half the rows, try again. _max_sim_batch's job shrinks to keeping the common
    case off this path, and _budget_note_oom's job is to make sure the next plan is wiser.

    Splitting here is licensed by exactly the argument gen_obs already makes for the predictive split:
    rows are independent and params/inits/force are all row-indexed, so a slice keeps them aligned. It
    re-draws the SDE noise in smaller blocks -- distributionally identical (still iid), not
    bit-reproducible against an unsplit run.

    BOUNDED IN BOTH DIRECTIONS. Halving stops at _MIN_SIM_CHUNK (the same floor the predictive guard
    uses, so there is one number and one meaning), giving at most log2(batch/256)+1 widths. A NON-OOM
    failure re-raises on the first attempt with its traceback intact, so a genuine model blow-up cannot
    become a retry loop -- and a CUDA context that a sticky error has already killed costs a few fast
    failures rather than a hang.

    STAYS ON POWERS OF TWO for a power-of-two input, because the solver specializes on the batch
    dimension and every new width pays a fresh compile -- the same reason _max_sim_batch quantizes.
    Testing ``< 2 * _MIN_SIM_CHUNK`` rather than ``<= _MIN_SIM_CHUNK`` is what stops a
    non-power-of-two batch halving BELOW the floor.

    NOT gated on device.type == "cuda": the halving costs nothing on CPU and gating it would make this
    path untestable without a GPU. Only the cache clears are CUDA-only.
    """
    try:
        return _gen_obs_one(model, params, t, inits, force, n_segs, steady_idx, fixed_dict,
                            state_dep_drift, batch_size, var_idx, dtype, device, out=out)
    except RuntimeError as err:
        # RuntimeError, NOT Exception: every OOM form derives from it (SimulationError,
        # torch.OutOfMemoryError, torch.AcceleratorError), and staying this narrow is also what keeps
        # streams.WorkerCancelled -- a BaseException, on purpose -- sailing straight through to
        # Worker.run. Widening here would turn every GUI cancel into a spurious retry storm.
        # _MIN_SIM_CHUNK is read from the MODULE rather than captured in a default argument, so a test
        # can lower the floor by rebinding it.
        if batch_size < 2 * _MIN_SIM_CHUNK or not _is_oom(err):
            raise                    # a real failure, or the floor: fail loudly, traceback intact
        note = f"{type(err).__name__}: {str(err).splitlines()[0][:200]}"
        n_keep = (1 if var_idx is not None else inits.shape[-1]) * max(0, t.shape[0] - steady_idx)
        _budget_note_oom(batch_size * (inits.shape[-1] * t.shape[0]
                                       + (force.shape[1] if force.dim() > 2 else 1) * t.shape[0]
                                       + max(inits.shape[-1] * min(t.shape[0], CHUNK_LEN), n_keep)))

    # EVERYTHING BELOW IS OUTSIDE THE HANDLER, AND THAT IS LOAD-BEARING. `err` owns a traceback, which
    # owns the frames of the failed attempt, which own its tensors -- Simulator.simulate's entire
    # (n_vars, batch, T) buffer among them, plus whatever the chained __cause__'s solver frames hold.
    # Calling empty_cache() while `err` is still bound frees NOTHING, so the retry OOMs again at half
    # the width and the whole mechanism reads as "the retry does not work". Python drops `err` at the
    # end of the except clause; only then is a release worth asking for. Hence `note` is a STRING.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        # The plan cache lives OUTSIDE the caching allocator, so empty_cache() cannot touch it -- see
        # the per-batch clear in gen_training_data. It is not this call's memory, but at the one moment
        # we are provably short it is the cheapest few hundred MB on the card and every plan is
        # re-mintable. This is also the real defragmentation step: handing every cached segment back to
        # the driver is what expandable_segments would buy, and cannot buy on Windows.
        torch.backends.cuda.cufft_plan_cache.clear()

    half = batch_size // 2
    # NOT SILENT, and on stderr rather than warnings.warn. This repo has no silent caps: a run that
    # halves on a large fraction of its batches has its geometry or its card wrong, and the only way
    # anyone learns that is if it says so EVERY time. warnings.warn cannot -- the default filter
    # ("once per location") would collapse hundreds of events into one line, and parts of
    # gen_training_data run under simplefilter("ignore"). stderr also lands in the GUI log as a
    # WARNING row, which is the right weight for an event that costs 2x on this batch.
    free = (f", {torch.cuda.mem_get_info(device)[0] / 2 ** 30:.2f} GiB reported free (optimistic on "
            f"Windows -- see the learned-budget note)") if device.type == "cuda" else ""
    print(f"{_batch_tag()}: OOM at simulation batch {batch_size}; retrying in chunks of "
          f"{half}{free}. Original: {note}", file=sys.stderr, flush=True)

    # Same preallocation as gen_obs' predictive split, for the same reason -- and note it happens
    # AFTER the empty_cache above, i.e. at the one moment in this function when the card is least
    # full. When a splitting caller already handed us a destination, `out` is non-None and this level
    # merely sub-divides that view, so the full result is allocated exactly ONCE however deep the
    # halving recursion goes.
    if out is None:
        n_out = 1 if var_idx is not None else inits.shape[-1]
        out = torch.empty((n_out, batch_size, max(0, t.shape[0] - steady_idx)),
                          dtype=dtype, device=device)
    for s in range(0, batch_size, half):
        e = min(s + half, batch_size)
        _gen_obs_retry(
            model, params[s:e], t, inits[s:e],
            force[s:e] if force.shape[0] == batch_size else force,
            n_segs, steady_idx, fixed_dict, state_dep_drift, e - s, var_idx, dtype, device,
            out=out[:, s:e, :])
    return out


def _rows_with_oom_retry(fn, lo: int, hi: int, *, per_row_elements: int,
                         device: torch.device) -> torch.Tensor:
    """Produce training rows [lo, hi) via ``fn(lo, hi)``, halving the ROW BLOCK on an out-of-memory.

    THE OUTER OF TWO RETRIES, AND DELIBERATELY THE EXPENSIVE ONE. _gen_obs_retry sits INSIDE fn and
    fires first, so by the time an OOM reaches here it is one the simulator-level split could not see:
    the reassembly buffer in gen_obs, the (rows, n_force_ch, t_fine) zero-drive tensor, the per-probe
    force rebuilt K times inside gen_chi_raw, the (rows, N_points) int64 gather index held across the
    whole K loop, x_spont_dim, gen_stats' sub-batches, pack_probe_block. At run_size=2048 /
    n_fine=280k those are ~2.29, ~2.29 x K, ~0.92 and ~0.46 GiB -- none of them inside _gen_obs_one,
    all of them linear in the rows. Halving the rows halves all of them at once. This is the gap that
    killed the 2026-08-11 retrain: a bare "CUDA error: out of memory" right after gen_chi_block's mask
    warning, i.e. inside the batch but outside the simulator.

    THE TRAINING BATCH IS THE RIGHT RETRY UNIT, and right in a way that lowering run_size is not. The
    batch's (t_scale_k, T_k) pair is an INDEX into the pre-filtered Sobol array, not a draw, and every
    row in the batch shares it -- as do the probe count, the multipliers and the durations, all drawn
    once above the seam. So re-running the batch as two half-width halves yields the SAME rows, in the
    SAME stratum, with the SAME probe set; the concatenation is the batch that would have been
    produced, differing only in that the SDE noise was drawn in smaller blocks (distributionally
    identical, not bit-reproducible -- the same licence _gen_obs_retry already takes). Shrinking
    run_size globally instead would change the NUMBER of (t_scale, T) strata in the run, which is a
    change to the training DISTRIBUTION rather than to its memory profile.

    THE FLOOR IS _MIN_SIM_CHUNK -- the same number the predictive guard and _gen_obs_retry use, so
    there is one number and one meaning.

    PREFER THE INNER RETRY. A halve here re-runs EVERYTHING for those rows: the spontaneous run, the
    summary statistics, and all K probe simulations -- about a dozen simulations plus a full gen_stats
    at K=11, against the inner retry's one. Do NOT lower _MIN_SIM_CHUNK to make this fire more often.

    NOT SHORT-CIRCUITED when the inner retry floored out, which is deliberate rather than an
    oversight: an inner OOM even at 256 rows can be caused entirely by memory held OUTSIDE the
    simulator (idx_c, x_spont_dim and the current probe's force are all live across the K loop), and
    halving the rows sheds all of it. So the outer halve genuinely can rescue what the inner could not.

    ``fn`` must be a pure function of its row range. It may read batch-level state, but it must not
    consume randomness anything OUTSIDE the batch depends on. _subset_probe_rows does draw from
    chi_gen inside fn -- harmless, because torch's CPU generator consumes one word per element, so
    rand(n) and two rand(n/2) leave the stream in the same position; only the within-batch pairing of
    draws to rows is re-permuted, which is a re-permutation of an iid draw.
    """
    n_rows = hi - lo
    try:
        return fn(lo, hi)
    except RuntimeError as err:
        # RuntimeError, NOT Exception -- the same reasoning as _gen_obs_retry: every OOM form derives
        # from it, and staying this narrow is what keeps streams.WorkerCancelled (a BaseException, on
        # purpose) sailing through to Worker.run rather than becoming a retry storm on every cancel.
        if n_rows < 2 * _MIN_SIM_CHUNK or not _is_oom(err):
            raise                    # a real failure, or the floor: fail loudly, traceback intact
        note = f"{type(err).__name__}: {str(err).splitlines()[0][:200]}"
        # Feed the learned budget even though this OOM may not have come from a simulator allocation.
        # It is still literally true that a block of this WIDTH at this GEOMETRY did not fit, and that
        # is the currency _max_sim_batch plans in -- so the next plan is wiser for the right reason.
        _budget_note_oom(n_rows * per_row_elements)

    # OUTSIDE THE HANDLER, AND EVEN MORE LOAD-BEARING HERE THAN IN _gen_obs_retry. `err` owns a
    # traceback owning every frame between here and the failure -- fn's own (force0, x_spont_dim),
    # gen_chi_raw's (idx_c, this probe's force, the accumulating chis list), gen_obs' and the
    # solver's. That is several GiB in the chi path at run_size=2048. Calling empty_cache() while
    # `err` is still bound frees NONE of it, the retry OOMs again at half the width, and the whole
    # mechanism reads as "the retry does not work". Python drops `err` at the end of the except
    # clause; only then is a release worth asking for. Hence `note` is a STRING.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        # The cuFFT plan cache lives OUTSIDE the caching allocator, so empty_cache() cannot touch it.
        # gen_stats mints several plan signatures per batch and chi.peak_freq another; at the one
        # moment we are provably short they are the cheapest few hundred MB on the card, and every
        # plan is re-mintable.
        torch.backends.cuda.cufft_plan_cache.clear()

    half = n_rows // 2
    free = (f", {torch.cuda.mem_get_info(device)[0] / 2 ** 30:.2f} GiB reported free (optimistic on "
            f"Windows -- see the learned-budget note)") if device.type == "cuda" else ""
    print(f"{_batch_tag()}: OOM with {n_rows} rows OUTSIDE the simulator retry; re-running this batch "
          f"in halves of {half}{free}. Original: {note}", file=sys.stderr, flush=True)

    parts = []
    for s in range(lo, hi, half):
        parts.append(_rows_with_oom_retry(fn, s, min(s + half, hi),
                                          per_row_elements=per_row_elements, device=device))
    # Cheap by construction: the parts are the CPU-side conditioning rows, (rows, 42 + 6*K_PAD)
    # float32 -- under a megabyte at run_size=2048 -- so this cat is nothing like the device-side one
    # gen_obs goes out of its way to avoid.
    return torch.cat(parts, dim=0)


def _gen_obs_one(model, params, t, inits, force, n_segs, steady_idx, fixed_dict,
                 state_dep_drift, batch_size, var_idx, dtype, device, out=None):
    """One un-split simulation batch. Split planning lives in gen_obs; this is the body.

    ``out``, when given, is the ``(n_out, batch, T')`` DESTINATION these rows belong in -- normally an
    ``out[:, s:e, :]`` view owned by a splitting caller. Writing into it instead of returning a fresh
    clone is what makes splitting memory-NEUTRAL rather than memory-POSITIVE; see gen_obs.
    """
    from core import registry

    full_params = params
    if fixed_dict is not None:
        n_full = params.shape[1] + len(fixed_dict)
        full_params = torch.empty((params.shape[0], n_full), dtype=params.dtype, device=params.device)
        free_idx = 0
        for i in range(n_full):
            if i in fixed_dict:
                full_params[:, i] = fixed_dict[i]
            else:
                full_params[:, i] = params[:, free_idx]
                free_idx += 1
        del params

    # move to the specified device
    t = t.to(dtype=dtype, device=device)

    if registry.is_user_model(model):
        simulator = registry.make_user_simulator(
            registry.get(model), full_params, force, inits, t,
            segs=n_segs, batch_size=batch_size, device=device)
    else:
        simulator_cls = VALID_SIMS[model.lower()]
        simulator = simulator_cls(full_params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)

    sol = simulator.simulate(state_dep_drift=state_dep_drift)
    # Slice BEFORE the copy: the clone has to coexist with the solver's full (n_vars, batch, T)
    # buffer, so narrowing to the one variable the caller reads is the difference between two
    # n_vars-deep tensors and one. Slicing keeps dim 0, so [0, :, :] means the same thing either way.
    sel = slice(None) if var_idx is None else slice(var_idx, var_idx + 1)
    src = sol[sel, 0, :, steady_idx:]
    if out is None:
        obs = src.clone()            # un-split call: nobody owns a destination, so materialise one
    else:
        # THE COPY THAT REPLACES A CLONE, and the whole point of the `out` parameter. A splitting
        # caller has already reserved the full result, so cloning here would make this chunk resident
        # TWICE -- once inside `out`'s span, once as the return value -- for the lifetime of the
        # return. Measured at run_size=2048 / n_fine=280k that double is 2.29/k GiB per chunk, which
        # at k=2 is 1.15 GiB, and it is exactly what makes a naive "preallocate and drop the cat" a
        # REGRESSION rather than a fix. `out` is a strided view (contiguous in the last dim,
        # batch-strided), which copy_ handles as a batch of contiguous row copies -- no staging buffer.
        out.copy_(src)
        obs = out
    del sol, src
    return obs

def gen_stats(x_spont: torch.Tensor, x_forced: torch.Tensor, dt: float | torch.Tensor,
              drive_amp, drive_freq, drive_phase,
              band_halfwidth: int = 2, bp_lo: float = 0.5, bp_hi: float = 1.5, slow_env_frac: float = 0.15,
              device: torch.device = torch.device('cpu'), stats_batch_size: int = 256,
              spontaneous_only: bool = False) -> torch.Tensor:
    """
    Generate statistical features from input data using the given parameters.

    Computes statistics in sub-batches on the target device to keep GPU FFT
    performance while avoiding OOM on large datasets. Each sub-batch result
    is moved to CPU immediately.

    :param x_spont: Unforced (spontaneous) trajectories for Groups A-F, shape (B, n), on CPU.
    :param x_forced: Forced (driven) trajectories for Group G, shape (B, n), on CPU.
    :param dt: The time step resolution for the input data (scalar, cell time units).
    :type dt: float
    :param drive_amp: Per-sample drive amplitude (dimensional), scalar or (B,).
    :param drive_freq: Per-sample drive frequency (dimensional), scalar or (B,).
    :param drive_phase: Per-sample drive phase (dimensional), scalar or (B,).
    :param band_halfwidth: Spectral band half-width in FFT bins (B7 / E2 harmonic powers). Default 2.
    :param bp_lo: Envelope band-pass lower edge as a fraction of the centre frequency. Default 0.5.
    :param bp_hi: Envelope band-pass upper edge as a fraction of the centre frequency. Default 1.5.
    :param slow_env_frac: Slow-envelope low-pass cutoff as a fraction of f_peak. Default 0.15.
    :param device: The device on which to compute statistics. Defaults to torch.device('cpu').
    :type device: torch.device
    :param stats_batch_size: Number of samples to process per sub-batch on GPU. Defaults to 256.
    :type stats_batch_size: int
    :param spontaneous_only: If True (a no-forcing model), skip the forced-response Group G and zero-pad
        it to the full feature width. ``x_forced``/``drive_*`` may then be None -- the spontaneous run
        is reused as the (unused) forced input. Keeps the output width == len(FEATURE_LABELS).

    :return: A tensor containing the computed statistical features. Shape: (batch size, number of statistics).
    :rtype: torch.Tensor
    """
    def _sub(v, s, e):
        if torch.is_tensor(v) and v.dim() > 0:
            return v[s:e].to(device)
        return v

    if spontaneous_only:
        # Group G is skipped, but SummaryStatistics.__init__ still coerces the drive params via
        # float(v) -> None would crash. The forced trajectory is likewise unused; reuse the spontaneous.
        if x_forced is None:
            x_forced = x_spont
        drive_amp = 0.0 if drive_amp is None else drive_amp
        drive_freq = 0.0 if drive_freq is None else drive_freq
        drive_phase = 0.0 if drive_phase is None else drive_phase

    total = x_spont.shape[0]
    results = []
    for start in range(0, total, stats_batch_size):
        end = min(start + stats_batch_size, total)
        xs_sub = x_spont[start:end].to(device)
        xf_sub = x_forced[start:end].to(device)
        dt_sub = dt[start:end].to(device) if isinstance(dt, torch.Tensor) and dt.dim() > 0 else dt
        stats = statistics.SummaryStatistics(
            xs_sub, xf_sub, dt_sub,
            _sub(drive_amp, start, end), _sub(drive_freq, start, end), _sub(drive_phase, start, end),
            band_halfwidth=band_halfwidth, bp_lo=bp_lo, bp_hi=bp_hi, slow_env_frac=slow_env_frac,
        )
        result = stats.compute_statistics(spontaneous_only=spontaneous_only)
        results.append(result.cpu())
        del stats, xs_sub, xf_sub, result
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

def gen_prior(model: str, t: torch.Tensor, global_batch_size: int, local_batch_size: int, segs: int, prior_bounds: list,
              state_dep_drift: bool = False, num_iterations: int = 25, log_mask: torch.Tensor | None = None,
              dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> torch.distributions.MixtureSameFamily:
    """
    Generates a prior distribution based on the given model and parameters.

    The function constructs a prior distribution using the specified model type
    and parameters. It supports different models, including "BP", "Nadrowski",
    and "Hopf". For any invalid model input, it raises a ValueError. The prior
    generation process involves a series of calculations and iterations executed
    without gradient computation.

    :param model: Specifies the type of model to use for prior generation. Accepted
                  values include "BP", "Nadrowski", and "Hopf".
    :param t: A tensor representing the input time vector used in the prior
              construction process.
    :param global_batch_size: Global batch size to be considered during the prior
                              generation.
    :param local_batch_size: Local batch size to be used in the computation.
    :param segs: Number of segmentation points for prior construction.
    :param prior_bounds: A list of bounding values defining the range of the prior
                         parameters.
    :param state_dep_drift: Boolean flag indicating whether to include state-dependent drift in the prior.
    :param num_iterations: Number of iterations to be performed in the process.
                           Defaults to 25.
    :param dtype: Data type to be used for tensor computations.
                  Defaults to torch.float32.
    :param device: Device on which the computation should run.
                   Defaults to torch.device('cpu').

    :return: A torch.distributions.MixtureSameFamily object representing the
             constructed prior distribution.
    :rtype: torch.distributions.MixtureSameFamily

    :raises ValueError: If the specified model is not supported.
    """
    from core import registry
    if registry.is_user_model(model):
        if not registry.is_sbi_user_model(model):
            raise ValueError(
                f"'{model}' is a user-defined model that is Simulate-only (it has forcing or no free "
                "parameters). Parameter inference supports user models with no forcing and >=1 parameter.")
        from core.SBI.Priors.user_prior import UserPrior
        prior = UserPrior(registry.get(model), dtype, device)
    elif VALID_PRIORS.get(model.lower()) is None:
        raise ValueError(f"Invalid simulator: {model}")
    else:
        prior = VALID_PRIORS[model.lower()](dtype, device)

    n_params = len(prior_bounds)

    with torch.no_grad():
        prior = prior.construct_prior(t, n_params, global_batch_size, local_batch_size, segs, prior_bounds,
                                      t_global_scale=2, num_iterations=num_iterations, n_max=175000, steady=False,
                                      state_dep_drift=state_dep_drift, log_mask=log_mask)

    return prior

def _subset_probe_rows(block: torch.Tensor, mask: torch.Tensor, k_pad: int, generator) -> torch.Tensor:
    """Randomly keep a PREFIX of each row's live probes, in place, and re-zero what is dropped.

    Half the rows keep their full set; the rest keep Uniform{1..n} of them. The drive set is shared
    across the batch (one simulation per probe serves every row), so this costs nothing and is the
    only way to decouple the per-row probe count from the batch's (t_scale, T) stratum -- otherwise
    the flow could read K off the batch's other conditioning and the encoder would never have to
    generalise. Dropping a PREFIX is safe because pack_probe_block already ordered the live probes
    contiguously; the survivors stay contiguous and frequency-ordered.
    """
    B = block.shape[0]
    e = block.reshape(B, k_pad, config.CHI_ELEM_W).clone()
    n = mask.sum(dim=1)                                                   # (B,) live probes per row
    full = torch.rand(B, generator=generator) < 0.5
    frac = torch.rand(B, generator=generator).to(n.device)
    keep = torch.where(full.to(n.device), n, (frac * n.to(frac.dtype)).floor().long() + 1)
    keep = torch.minimum(keep, n).clamp(min=0)
    slots = torch.arange(k_pad, device=block.device).unsqueeze(0)         # (1, k_pad)
    alive = slots < keep.unsqueeze(1).to(slots.device)                    # (B, k_pad)
    return (e * alive.unsqueeze(-1).to(e.dtype)).reshape(B, k_pad * config.CHI_ELEM_W)


def gen_chi_raw(model: str, params_nd: torch.Tensor, rescale: torch.Tensor, x_spont_dim: torch.Tensor,
                t_fine: torch.Tensor, inits: torch.Tensor, rescale_idx: dict,
                n_segs: int, steady_idx: int, subsample, N_points: int, dt_exp: float,
                multipliers: torch.Tensor, f0_nd: float, state_dep_drift: bool = False,
                fixed_dict: dict = None,
                absolute_freqs: bool = False, resolution_filter: bool = True,
                duration_frac=None, max_cycles: float | None = None,
                adapt_placement: bool = False, bounds: tuple | None = None,
                dtype: torch.dtype = torch.float32,
                device: torch.device = torch.device('cpu')) -> tuple:
    """
    K single-tone forced runs -> the RAW probe measurements
    ``(chi (B,K) complex, u (B,K), logcyc (B,K), valid (B,K) bool)``. Generalizes the single-frequency
    Group-G lock-in to a susceptibility CURVE (see config.CHI_MODE + core/SBI/chi.py). One forced
    simulation per probe = the "single-tone x K recordings" protocol.

    Exactly K simulations run, never k_pad -- padding is free.

    Drives at a FIXED ND amplitude f0_nd by passing dimensional amp = f0_nd * f_scale to
    build_nondim_sin_force_tensor (which divides it back to f0_nd), so lock-in SNR is uniform across
    the f_scale prior. chi = redimensionalized response / dimensional drive = (x_scale/f_scale)*chi_nd
    carries the physical scale magnitude (like Group-G's gain); its SHAPE over omega carries the ND
    resonance.

    :param params_nd: (B, n_nd) ND params (the inferred ND block).
    :param rescale: (B, n_rescale) PHYSICAL rescale params (x_scale/t_scale/f_scale...).
    :param x_spont_dim: (B, N_points) physical spontaneous trace -> Omega_0 per sample.
    :param t_fine: (T_full,) fine ND time grid the drive/sim use.
    :param inits: (B, n_vars) initial conditions.
    :param subsample: fine->dt_exp downsample factor. A scalar int (uniform t_scale: training batches,
                      a single GT) OR a (B,) per-sample tensor (posterior samples in PPC, whose t_scale
                      differs per sample); applied via gather so both cases share one code path.
    :param multipliers: (K,) or (B, K). Relative multipliers of each row's own measured Omega_0, or --
                        with ``absolute_freqs`` -- frequencies in cell freq units. ABSOLUTE is what the
                        experimental path and the PPC use: the experiment fixed the drive frequencies,
                        and re-deriving them per posterior sample from that sample's own f_peak would
                        simulate a different experiment and make the PPC agree for the wrong reason.
    :param f0_nd: ND drive amplitude (config.CHI_F0).
    :param resolution_filter: mark probes below config.CHI_MIN_CYCLES drive cycles invalid. **Pass
                              False for the Fisher.** The filter depends on f_peak, which depends on
                              theta, so a probe can CROSS the threshold between the +dz and -dz arms
                              of a central difference -- a step of 1 divided by fnoise's 1e-9 floor
                              puts ~1e9 into the Jacobian, and V becomes that discontinuity.
    :param duration_frac: (K,) fractions of N_points to lock each probe in over. None = full length.
    :param max_cycles: CEILING on the drive cycles each probe is locked in over; None reads
                       ``config.CHI_MAX_CYCLES``, ``math.inf`` disables it. Applied AFTER
                       ``duration_frac``, so it is a ceiling on that draw rather than a replacement.
                       This is not a filter -- nothing is masked or dropped, the SEGMENT is shortened
                       -- which is why it lives here rather than in a caller: training, the Fisher
                       rotation, the PPC and the experimental path must all measure the same
                       observable, and a ceiling applied in only one of them is silent. See
                       config.CHI_MAX_CYCLES for the measurement behind it.
    :param adapt_placement: lift each ROW's multipliers into the sub-band its own Omega_0 can resolve
                            (:func:`core.SBI.chi.resolvable_multipliers`). **TRAINING ONLY.** The
                            experimental path drove at frequencies the experiment chose and the PPC
                            must reproduce the observation's, so both pass ``absolute_freqs`` and are
                            never adapted -- moving a probe there would answer a different experiment
                            than the one that was run. Ignored when ``absolute_freqs`` is set.
    :param bounds: the chi band, for ``adapt_placement``; None reads ``config.CHI_FREQ_BOUNDS``.
    :return: (chi (B,K) complex, u (B,K), logcyc (B,K), valid (B,K) bool). Use
             :func:`gen_chi_block` for the padded conditioning block.
    """
    max_cycles = config.CHI_MAX_CYCLES if max_cycles is None else float(max_cycles)
    B = params_nd.shape[0]
    f_peak = chi.peak_freq(x_spont_dim, dt_exp)                         # (B,) cell freq units
    x_scale = rescale[:, rescale_idx["x_scale"]].unsqueeze(1)
    x_offset = rescale[:, rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in rescale_idx else 0.0
    if "f_scale" in rescale_idx:
        f_scale_eff = rescale[:, rescale_idx["f_scale"]]                # (B,)
    else:  # Hopf-style: build_nondim uses f_scale = x_scale / t_scale
        f_scale_eff = rescale[:, rescale_idx["x_scale"]] / rescale[:, rescale_idx["t_scale"]]
    amp_dim = f0_nd * f_scale_eff                                       # (B,) dimensional; ND drive == f0_nd
    T_obs = N_points * dt_exp
    nyq = 0.5 / dt_exp                                                  # dt_exp-sampling Nyquist (cell freq units)
    # Fine -> dt_exp downsampling. gen_obs solves on t_fine and returns [..., steady_idx:], so x_nd's
    # width is this same value for every one of the K runs -- the choice below is loop-invariant and
    # is made ONCE, and the index tensor (when needed at all) is built ONCE.
    #   * uniform int subsample AND a fine grid long enough that the clamp cannot bind -> plain
    #     strided slicing, exactly what the non-chi branches of gen_training_data do. This builds NO
    #     (B, N_points) int64 index at all; the old code kept two of them live, ~2 GB at run_size=2048.
    #   * (B,) per-sample subsample (the PPC path, whose rows have different strides), or a fine grid
    #     that ran out -> keep the gather. `t_fine = t[:n_fine_total]` SILENTLY CLIPS, which happens
    #     for ~20% of accepted draws on model-builder bounds (t_scale in (v/2, v*2) makes len(t)
    #     shorter than the N_ND_MAX filter allows). There the clamp REPLICATES the last sample, where
    #     slicing would quietly return fewer than N_points columns -- desynchronising the trace from
    #     the T_obs that normalises chi below, a bias that would show up only in that corner.
    n_avail = t_fine.shape[0] - steady_idx
    s_int = None if torch.is_tensor(subsample) else max(1, int(subsample))
    idx_c = None
    if s_int is None or s_int * (N_points - 1) >= n_avail:
        subs = (subsample.to(device=device).long().clamp(min=1) if torch.is_tensor(subsample)
                else torch.full((B,), s_int, device=device, dtype=torch.long))
        idx_c = (subs.unsqueeze(1)
                 * torch.arange(N_points, device=device, dtype=torch.long).unsqueeze(0)
                 ).clamp_(max=n_avail - 1)                              # (B, N_points), clamped in place
    fidx = {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
    n_force_ch = _forcing.n_force_channels(model, fidx, inits.shape[-1])

    # Resolve the probe frequencies ONCE, before the loop. `multipliers` may be relative (the usual
    # case: mult_k * the passive trace's own peak) or ABSOLUTE cell-frequency values, which is what
    # the experimental path and the PPC need -- there the drive frequencies were fixed by the
    # experiment, and re-deriving them per posterior sample from that sample's own f_peak would
    # simulate a different experiment.
    mults = multipliers if torch.is_tensor(multipliers) else torch.as_tensor(multipliers)
    mults = mults.to(device=device, dtype=f_peak.dtype)
    if mults.dim() == 1:
        mults = mults.unsqueeze(0)                                      # (1, K) -> broadcast over B
    if adapt_placement and not absolute_freqs:
        # Per-ROW placement: one shared multiplier set cannot resolve across a prior spanning ~4
        # decades of Omega_0 (trap CHI10 / handoff 4.3.4). Uses the FULL duration as the budget --
        # duration_frac and the CHI_MAX_CYCLES ceiling below only ever SHORTEN the window, so a
        # multiplier chosen against the full length is the most permissive honest choice; the floor
        # check below still has the last word on the duration actually used.
        mults = chi.resolvable_multipliers(mults, f_peak, N_points * dt_exp, bounds=bounds)
    freqs = mults if absolute_freqs else mults * f_peak.unsqueeze(1)    # (B, K) or (1, K)
    freqs = freqs.expand(B, -1)
    K = freqs.shape[1]

    chis, u_list, logcyc_list = [], [], []
    # Per-probe validity. A probe is never MOVED and never silently dropped -- it is masked, and the
    # caller is told how many. Clamping to Nyquist (what this used to do) relabels a probe as a
    # different frequency than the one requested, which is invisible downstream.
    valid = torch.ones((B, K), dtype=torch.bool, device=device)
    for k in range(K):
        freq_k = freqs[:, k].contiguous()                              # (B,) absolute, cell freq units
        valid[:, k] &= torch.isfinite(freq_k) & (freq_k > 0) & (freq_k < 0.9 * nyq)
        # Per-probe duration: lock in over a PREFIX of the trace. Free (the samples already exist) and
        # it makes the (duration, frequency) trade-off -- what a real session actually varies -- an
        # axis of the training distribution rather than a constant.
        N_k = N_points if duration_frac is None else max(1, int(round(float(duration_frac[k]) * N_points)))
        # THE DURATION CEILING (config.CHI_MAX_CYCLES), applied PER ROW.
        #
        # It used to be one scalar keyed on the batch's FASTEST row, because lock_in_batched took a
        # scalar T_obs. That cost was real and measured: Omega_0 spans ~4 decades inside a training
        # batch, so keying on the fastest truncated the slow rows to a fraction of a cycle and masked
        # them -- ~48 % of rows carried no live probe at all (handoff 4.3.4/4.3.5). lock_in_batched
        # now takes an (B,) n_samples, so each row gets exactly the prefix its own frequency needs.
        # Rows whose full length is already under the ceiling are untouched.
        #
        # Computed over the ALREADY-VALIDATED rows: freq_k still holds the non-finite / out-of-Nyquist
        # entries of rows masked on the line above, and dividing by those would poison the length. An
        # invalid row keeps the full N_k -- it is masked anyway, so its length changes nothing.
        N_row = torch.full((B,), N_k, dtype=torch.long, device=device)
        if math.isfinite(max_cycles):
            ok_k = valid[:, k] & (freq_k > 0)
            if bool(ok_k.any()):
                cap_row = torch.floor(max_cycles / freq_k.clamp(min=1e-30).double() / dt_exp)
                cap_row = cap_row.clamp(min=1.0, max=float(N_k)).long()
                N_row = torch.where(ok_k, cap_row, N_row)
        # T_row is what each row was ACTUALLY integrated over: it normalises that row's chi, sets its
        # cycle count for the floor below, and is what logcyc reports. There is deliberately no
        # scalar counterpart any more -- keeping one around is how logcyc would come to describe a
        # duration the lock-in did not use.
        T_row = N_row.to(torch.float64) * dt_exp
        if resolution_filter:
            # A lock-in over a fraction of a cycle returns the demeaned trace's residual drift plus
            # spontaneous 1/f content: finite, in range, and REPRODUCIBLE -- which is exactly why it
            # survived a CV screen -- but it is not a susceptibility.
            # Against the row's OWN duration -- the whole point of C-8 is that these differ.
            valid[:, k] &= (freq_k.double() * T_row) >= config.CHI_MIN_CYCLES
        forcing_params = torch.zeros((B, 4), dtype=dtype, device=device)
        forcing_params[:, 0] = amp_dim
        forcing_params[:, 1] = freq_k
        forcing_params[:, 2] = math.pi / 2.0                           # phase -> cos drive (FDT convention)
        force = build_nondim_sin_force_tensor(forcing_params, t_fine, rescale, fidx, rescale_idx)
        if force.shape[1] < n_force_ch:
            # The sin builder emits ONE channel (fidx above declares no "amp_y"), but the model's
            # drift may index more: HopfModel reads force_step[:, 1] unconditionally, and a user
            # model reads one channel per state variable -- so chi mode used to die with an
            # IndexError on anything but Nadrowski/BP. Probe channel 0 and leave the rest at zero,
            # which is the same convention the FDT campaigns drive (see forcing.n_force_channels).
            padded = torch.zeros((B, n_force_ch, force.shape[2]), dtype=force.dtype, device=force.device)
            padded[:, :force.shape[1], :] = force
            force = padded
        x_nd = gen_obs(model=model, params=params_nd, t=t_fine, inits=inits, force=force,
                       n_segs=n_segs, steady_idx=steady_idx, fixed_dict=fixed_dict,
                       state_dep_drift=state_dep_drift, batch_size=B, var_idx=0,
                       dtype=dtype, device=device)[0, :, :]
        x_sub = x_nd[:, ::s_int][:, :N_points] if idx_c is None else torch.gather(x_nd, 1, idx_c)
        x_dim = helpers.rescale(x_sub[:, :N_k], x_scale, x_offset)      # (B, N_k), a FRESH tensor
        # Release the simulation BEFORE the lock-in: x_nd is a view and so pins its whole base, and
        # force is a (B, n_force_ch, T_fine) tensor of its own. helpers.rescale has already
        # materialised x_dim, so nothing below reads any of them. (idx_c is loop-invariant --
        # do NOT drop it.)
        del force, x_nd, x_sub
        # n_samples/T_row, not the scalar: each row is integrated over its own prefix of x_dim.
        chis.append(chi.lock_in_batched(x_dim, 2.0 * math.pi * freq_k, amp_dim, T_row, dt_exp,
                                        n_samples=N_row))
        # The frequency ACTUALLY locked in at, and the cycles actually seen -- both carried as
        # features rather than implied by slot index, which is what makes placement free. logcyc uses
        # T_row for the same reason the filter above does: it is the encoder's record of how much
        # evidence this probe rests on, so it must describe the integration that really happened.
        u_list.append(torch.log(freq_k / f_peak.clamp(min=1e-30)))
        logcyc_list.append(torch.log(torch.clamp(freq_k.double() * T_row, min=1e-30)).to(freq_k.dtype))
        del x_dim
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (torch.stack(chis, dim=1), torch.stack(u_list, dim=1),
            torch.stack(logcyc_list, dim=1), valid)


def gen_chi_block(*args, k_pad: int = None, bounds: tuple = None, **kwargs) -> tuple:
    """``gen_chi_raw`` + :func:`core.SBI.chi.pack_probe_block` -> the padded CONDITIONING block.

    Split from the raw lock-in deliberately: the Fisher (``SBI/decorrelate.feats``) needs the
    susceptibilities WITHOUT the frequency and mask channels, because both are theta-independent
    there and poison the Jacobian -- see CHI_FISHER_CHANNELS. Sharing the simulation loop keeps the
    two feature sets provably from drifting apart.

    :return: ((B, CHI_ELEM_W*k_pad) block, (B, k_pad) bool mask).
    """
    # `bounds` goes to BOTH: the packer normalises u_hat by it, and adapt_placement compresses into
    # it. Forwarding to only one would let a probe be placed against one band and screened against
    # another -- the same class of mismatch the sidecar band check exists to catch.
    chi_stack, u, logcyc, valid = gen_chi_raw(*args, bounds=bounds, **kwargs)
    block, mask = chi.pack_probe_block(chi_stack, u, logcyc, valid, k_pad=k_pad, bounds=bounds)
    B, K = chi_stack.shape
    dropped = int((~mask[:, :K]).sum())
    if dropped:
        # Silent attrition is what made the first chi posterior inexplicable. Make it a number.
        warnings.warn(
            f"{_batch_tag()}: chi: {dropped}/{B * K} probes masked "
            f"(below {config.CHI_MIN_CYCLES} drive cycles, "
            f"at/above Nyquist, out of band, or a non-finite lock-in).", stacklevel=2)
    return block, mask


def gen_training_data(model: str, prior: torch.distributions.Distribution, forcing_prior: torch.distributions.Distribution,
                      t: torch.Tensor, run_size: int, n_runs: int, steady_idx: int, dt_nd_min: float,
                      nd_dim: int, forcing_idx: dict, rescale_idx: dict,
                      dt_exp: float = None, t_min_exp: float = None, t_max_exp: float = None,
                      t_scale_bounds: tuple[float, float] = None,
                      proposal: DirectPosterior = None, theta_transform: Transform | None = None,
                      fixed_dict: dict = None, state_dep_drift: bool = False,
                      spontaneous_only: bool = False, chi_mode: bool = False,
                      chi_f0: float | None = None,
                      chi_freq_bounds: tuple | None = None, chi_k_pad: int | None = None,
                      chi_k_fixed: int | None = None, chi_max_cycles: float | None = None,
                      n_vars: int | None = None, checkpoint: dict | None = None,
                      dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> tuple:
    """
    Generate synthetic training data for the SBI posterior using batch-by-scale strategy.

    Each batch shares a single (t_scale_k, T_k) pair sampled via Sobol sequence over the
    2D space [t_scale_lo, t_scale_hi] x [t_min_exp, t_max_exp]. Within a batch, the 11 ND
    parameters and (D, K_gs*D) vary per-simulation, but t_scale is overridden to the
    batch-level value. The pre-simulated ND trajectory is subsampled to dt_nd_k = dt_exp / t_scale_k
    and truncated to T_nd_k = T_k / t_scale_k points, so that after rescaling every simulation
    has physical duration T_k at sampling rate 1/dt_exp. Summary statistics are computed with
    the fixed dt_exp, and log(T_k) is appended to the conditioning vector.

    If theta_transform is provided, `prior` is interpreted as a LATENT prior. Samples z
    from it, applies theta_transform(z) to get physical θ for the simulator, and stores
    the latent z as the training target. The override of t_scale to the batch-level value
    is performed in physical space, after which the latent is recomputed via
    theta_transform.inv so the stored z corresponds exactly to what the simulator saw.

    If theta_transform is None, `prior` is physical and the legacy path is taken.

    :param model: Name of the simulation model (e.g. "nadrowski", "hopf").
    :param prior: Prior distribution over inferred parameters (ND x rescale product prior).
    :param forcing_prior: Prior distribution over dimensional forcing parameters, sampled
                          independently every batch regardless of SNPE round.
    :param t: Pre-simulated ND time tensor at finest resolution (dt_nd_min), shape (T_full,).
    :param run_size: Number of simulations per batch.
    :param n_runs: Number of batches to generate.
    :param steady_idx: Index where transient ends and steady-state begins (at full resolution).
    :param dt_nd_min: Finest ND time step of the pre-simulated trajectory.
    :param nd_dim: Number of ND model parameters; used to split inferred params into
                   theta_nd [:nd_dim] and theta_rescale [nd_dim:].
    :param forcing_idx: Maps forcing param names to column indices,
                        e.g. {"amp": 0, "freq": 1, "phase": 2, "offset": 3}.
    :param rescale_idx: Maps rescale param names to column indices,
                        e.g. {"t_scale": 3, "t_offset": 2, "f_scale": 7, "f_offset": 6}.
    :param dt_exp: Fixed experimental sampling interval (seconds).
    :param t_min_exp: Shortest experimental recording duration (seconds).
    :param t_max_exp: Longest experimental recording duration (seconds).
    :param t_scale_bounds: (lo, hi) bounds on the t_scale rescaling parameter.
    :param proposal: Proposal distribution for SNPE rounds 2+. If None, samples from prior.
    :param theta_transform: Optional transformation function for physical parameters.
    :param fixed_dict: Optional dict mapping ND parameter indices to fixed values for
                       conditional posterior estimation.
    :param state_dep_drift: Whether the model uses state-dependent drift.
    :param checkpoint: None (the default) disables checkpointing entirely -- no disk access, and the
                       function behaves exactly as it did before C-11, which is what keeps
                       analysis.gen_cal_data and every existing test call site unchanged. Otherwise a
                       dict:
                         dir       directory to write to (the caller owns naming; see
                                   SBI.training_checkpoint.resolve_dir)
                         identity  the config fields a resume must match, checked field by field
                         probe     bijection_probe(theta_transform, P) -- catches a changed box
                         V         the rotation to store, so a resume can reuse it rather than
                                   recompute it (trap X10: V is NOT reproducible across processes)
                         every     batches between writes; None/absent => config.TRAINING_CHECKPOINT_EVERY
                         resume    "auto" (default) | "never" | "require"
    :param chi_k_fixed: hold the probe COUNT at this value instead of drawing it per batch, and skip
                        the per-row subsetting. **For a STRATIFIED CALIBRATION, not for training** --
                        training's whole point is that K varies, and fixing it would train a network
                        that has only ever seen one probe count. A pooled SBC over a mixture of
                        counts can be flat while each count is miscalibrated in compensating
                        directions, which is why validating one count at a time needs a lever at all.
                        Placement stays jittered (``chi.sample_multipliers``) so only the count
                        differs from the training distribution. Note there is deliberately NO
                        ``chi_n_freqs`` here: that is the count an OBSERVATION supplies, and honouring
                        it during data generation would destroy the K-agnosticism the padded probe-set
                        layout exists to provide.
    :param chi_max_cycles: lock-in duration CEILING in drive cycles; None reads
                           ``config.CHI_MAX_CYCLES``. Bounds the ``duration_frac`` draw below --
                           without it the draw is a FRACTION of the recording, which does not bound
                           cycles at all, so a long recording walks its probes past the
                           reproducibility wall (config.CHI_MAX_CYCLES, trap CHI9).
    :param dtype: Tensor data type. Defaults to torch.float32.
    :param device: Computation device. Defaults to CPU.
    :return: Tuple of (training_data, thetas) where training_data has shape
             (n_runs * run_size, n_stats + n_forcing + 1) and thetas has shape
             (n_runs * run_size, nd_dim + rescale_dim).
    """
    from core import registry
    is_user = registry.is_user_model(model)
    if model.lower() not in VALID_SIMS and not is_user:
        raise ValueError(f"Invalid simulator: {model}")

    if is_user:
        # User models declare per-variable inits (a nondimensional model may live on a unit scale that
        # randint(0, 10) would blow past); broadcast them across the run. n_vars comes from the caller.
        from core.SBI.Priors.user_prior import declared_inits
        inits = declared_inits(registry.get(model)).to(dtype=dtype, device=device).expand(run_size, -1)
    else:
        n_pos, n_prob = INIT_SHAPES[model.lower()]
        if n_prob > 0:
            # Probability-like channels start at 0. (This was np.random.randint(0, 1, ...), which is
            # ALWAYS 0 -- numpy's `high` is exclusive -- so the behaviour is unchanged; it just no
            # longer reads as a random draw that someone might later "fix" into a real one.)
            inits = torch.tensor(
                helpers.concat(np.array(np.random.randint(0, 10, size=(run_size, n_pos))),
                               np.zeros((run_size, n_prob), dtype=int)),
                dtype=dtype, device=device)
        else:
            inits = torch.tensor(np.random.randint(0, 10, size=(run_size, n_pos)), dtype=dtype, device=device)

    # n_vars was ACCEPTED AND IGNORED: the real count comes from inits above, so the argument was a
    # dead input that three callers dutifully computed. Rather than drop it, use it -- a caller whose
    # idea of the state width disagrees with the model's declared inits has a real bug (a stale cell
    # file, a user model edited since the config was built), and it would otherwise surface much
    # later as a shape error inside the solver.
    if n_vars is not None and int(n_vars) != inits.shape[-1]:
        raise ValueError(
            f"n_vars={n_vars} disagrees with the model's initial conditions, which are "
            f"{inits.shape[-1]}-wide for '{model}'. One of the two is stale.")

    # move to the specified device
    t = t.to(dtype=dtype, device=device)

    # Width of the zero-force tensor the driveless runs below pass to gen_obs. This used to be n_vars,
    # which over-allocates the single largest tensor of a training batch 3x for Nadrowski and 5x for BP
    # (their drifts read channel 0 only) -- at run_size=2048 and the longest admissible fine grid that
    # is 7.4 GB where 2.5 GB is needed. forcing.n_force_channels is the shared per-model rule.
    n_force_ch = _forcing.n_force_channels(model, forcing_idx, inits.shape[-1])

    # PREALLOCATED accumulators, not lists. Both are sized on the first batch, because the
    # conditioning width W is a function of the observation mode and is not known here.
    #
    # The lists they replace held ~4.35 GiB of finished rows at the production shape (5000 x 2048 x
    # 114) and then `torch.cat` allocated another 4.35 GiB for the result while the list was still
    # referenced -- an 8.7 GiB host peak at the very END of a multi-day run, which is the worst
    # possible moment to discover it. Filling a buffer in place also makes a checkpoint shard a
    # contiguous slice copy and a resume a slice fill rather than a list rebuild, which is what C-11
    # needs; the checkpoint's whole memory story rests on this.
    #
    # torch.empty, not zeros: every row is written before it is read, and only [0, batches_done) is
    # ever serialised or returned, so zeroing 4.35 GiB would be pure cost.
    x_buf = None
    th_buf = None

    sampling_dist = prior if proposal is None else proposal

    # chi(omega) mode: precompute the relative-frequency multipliers + drive amplitude once. K / bounds /
    # F0 come from the CALLER (carried on the SimConfig) so a run is self-describing; None falls back to
    # the live module defaults for direct/CLI callers.
    chi_gen = None
    if chi_mode:
        from core import config as _config
        chi_f0 = _config.CHI_F0 if chi_f0 is None else chi_f0
        chi_freq_bounds = _config.CHI_FREQ_BOUNDS if chi_freq_bounds is None else chi_freq_bounds
        chi_k_pad = _config.CHI_K_PAD if chi_k_pad is None else int(chi_k_pad)
        if chi_k_fixed is not None:
            chi_k_fixed = int(chi_k_fixed)
            if not (1 <= chi_k_fixed <= chi_k_pad):
                raise ValueError(
                    f"chi_k_fixed={chi_k_fixed} is outside 1..chi_k_pad={chi_k_pad}. A calibration "
                    f"stratum cannot ask for more probes than the network has slots, and 0 probes is "
                    f"an all-masked observation the experimental path refuses outright.")
        # A DEDICATED generator for the probe draw. Never the global RNG: the chi block is surrounded
        # by deliberate manual_seed() calls (trap X3's common-random-numbers), and a placement drawn
        # from the global stream would be re-randomised -- or worse, frozen -- by them.
        chi_gen = torch.Generator(device="cpu")
        chi_gen.manual_seed(20260805)

    # --- Checkpointing (C-11): decide RESUME before anything expensive ---------------------------
    # Resolved here, above the Sobol schedule, because a resume must take that schedule from the
    # checkpoint rather than rebuild it: SobolEngine(scramble=True) consumes the torch global RNG at
    # CONSTRUCTION and _draw_and_filter's accept count depends on the geometry, so it cannot be
    # re-derived from a seed. Rebuilding it would silently re-stratify the second half of the run.
    _ck_dir = _ck_every = _ck_resumed = None
    _start_k = 0
    if checkpoint is not None:
        from core.SBI import training_checkpoint as _tc
        _ck_dir = Path(checkpoint["dir"])
        _ck_every = checkpoint.get("every")
        _ck_every = config.TRAINING_CHECKPOINT_EVERY if _ck_every is None else int(_ck_every)
        _ck_mode = checkpoint.get("resume", "auto")
        _state = _tc.peek(_ck_dir)
        _have = bool(_state and _state.get("batches_done"))
        if _ck_mode == "never" and _have:
            raise ValueError(
                f"A training checkpoint with {_state['batches_done']} completed batches already "
                f"exists at {_ck_dir} and resume='never' would overwrite it. Resume instead, or "
                f"delete that directory deliberately.")
        if _ck_mode == "require" and not _have:
            raise ValueError(f"resume='require' but there is no resumable checkpoint at {_ck_dir}.")
        if _have and _ck_mode != "never":
            _ck_resumed = _tc.verify(_ck_dir, checkpoint["identity"], checkpoint.get("probe"))
            _start_k = int(_state["batches_done"])
            # The schedule and the initial conditions come from the header, never from a redraw.
            # inits especially: it is drawn from NUMPY's RNG (trap X8), which nothing else here
            # restores, so a redraw would quietly change the initial conditions mid-run.
            batch_t_scales = _ck_resumed["batch_t_scales"]
            batch_Ts = _ck_resumed["batch_Ts"]
            inits = _ck_resumed["inits"].to(dtype=dtype, device=device)
            _x_prev, _th_prev = _tc.load_rows(_ck_dir, _start_k, run_size)
            if _x_prev is not None:
                x_buf = torch.empty((n_runs * run_size, _x_prev.shape[-1]), dtype=_x_prev.dtype)
                th_buf = torch.empty((n_runs * run_size, _th_prev.shape[-1]), dtype=_th_prev.dtype)
                x_buf[:_x_prev.shape[0]] = _x_prev
                th_buf[:_th_prev.shape[0]] = _th_prev
                del _x_prev, _th_prev
            # LAST, so nothing above (Sobol is skipped, but prior construction elsewhere may have
            # drawn) leaves the streams anywhere other than where batch _start_k found them.
            _tc.rng_restore(_state.get("rng") or {}, device, chi_gen)
            print(f"[checkpoint] resuming at batch {_start_k}/{n_runs} from {_ck_dir} "
                  f"({'reusing the stored rotation V' if _ck_resumed.get('V') is not None else 'no rotation'})",
                  flush=True)
        else:
            note = _tc.describe_siblings(checkpoint["identity"], _ck_dir.parent)
            if note:
                print(note, flush=True)

    # --- Stratified sampling of batch-level (t_scale, T) pairs with pre-filter ---
    t_scale_lo, t_scale_hi = t_scale_bounds
    log_t_scale_lo, log_t_scale_hi = math.log(t_scale_lo), math.log(t_scale_hi)
    log_T_lo, log_T_hi = math.log(t_min_exp), math.log(t_max_exp)

    # A batch must fit BOTH ceilings. N_ND_MAX is the cost cap; len(t) is the hard length of the ND
    # grid every batch slices with `t_fine = t[:n_fine_total]`, which CLIPS SILENTLY when it is
    # exceeded. An over-long draw then produces a self-inconsistent training row: the spontaneous
    # trace is built by SLICING (so it comes back short) while gen_chi_block GATHERS to N_points
    # (its clamp replicating the last sample), so chi and the summary statistics describe different
    # trace lengths, and log(T_k) records a duration neither of them actually has. Filtering the
    # draw is the honest fix -- these geometries cannot be simulated at the requested resolution, so
    # they must not enter the training set at all.
    #   Built-in bounds are unaffected: t_scale in (1, 40) puts len(t) at ~2.4M against a 300k cap,
    #   so nothing was ever clipped. It bites model-builder bounds, where t_scale in (v/2, v*2)
    #   makes len(t) = 240k -- SHORTER than N_ND_MAX -- and ~20% of accepted draws truncated.
    n_fine_max = min(N_ND_MAX, t.shape[0])

    def _draw_and_filter(n_candidates: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw Sobol candidates, filter by the fine-grid ceiling, return (t_scales, Ts) that fit."""
        pts = sobol.draw(n_candidates)
        cand_t_scales = torch.exp(log_t_scale_lo + pts[:, 0] * (log_t_scale_hi - log_t_scale_lo))
        cand_Ts = torch.exp(log_T_lo + pts[:, 1] * (log_T_hi - log_T_lo))
        dt_nd_cand = dt_exp / cand_t_scales
        subsample_cand = torch.clamp(torch.round(dt_nd_cand / dt_nd_min), min=1).long()
        N_points_cand = (cand_Ts / dt_exp).long()
        n_fine_cand = steady_idx + N_points_cand * subsample_cand
        valid = n_fine_cand <= n_fine_max
        return cand_t_scales[valid], cand_Ts[valid]

    # SKIPPED ENTIRELY on a resume: batch_t_scales/batch_Ts already came from the checkpoint header.
    # Not merely redundant -- rebuilding the engine would consume the torch global RNG (scramble=True
    # draws at construction) between here and the RNG restore, and re-deriving a schedule that the
    # accept/reject filter makes geometry-dependent is precisely the "silently non-uniform
    # stratification" C-11 warns is worse than crashing.
    if _ck_resumed is None:
        sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        oversample = 3
        valid_t_scales, valid_Ts = _draw_and_filter(n_runs * oversample)
        # Fallback: keep drawing more candidates until we have enough valid ones. A whole draw coming
        # back empty means NO (t_scale, T) in the declared bounds fits the grid, so redrawing would
        # spin forever -- say what is wrong instead of hanging.
        while valid_t_scales.shape[0] < n_runs:
            more_t_scales, more_Ts = _draw_and_filter(n_runs * oversample)
            if more_t_scales.numel() == 0:
                raise ValueError(
                    f"No (t_scale, T) pair in the declared bounds fits the fine-grid ceiling of "
                    f"{n_fine_max} steps (steady_idx={steady_idx}, dt_exp={dt_exp}, "
                    f"dt_nd_min={dt_nd_min}, t_scale in {t_scale_bounds}, T in "
                    f"[{t_min_exp}, {t_max_exp}]). Shorten the recording range, widen t_scale, or "
                    f"raise N_ND_MAX / the model's t_nd_max.")
            valid_t_scales = torch.cat([valid_t_scales, more_t_scales])
            valid_Ts = torch.cat([valid_Ts, more_Ts])
        batch_t_scales = valid_t_scales[:n_runs]
        batch_Ts = valid_Ts[:n_runs]

        if _ck_dir is not None:
            # The PREFLIGHT write, before the first simulation. A read-only Resources/, a permissions
            # problem or a full disk then surfaces in the first seconds instead of at the first
            # cadence write, twenty minutes in -- and this is also where the schedule becomes durable.
            _tc.create(_ck_dir, checkpoint["identity"],
                       schedule_t_scales=batch_t_scales, schedule_Ts=batch_Ts, inits=inits,
                       V=checkpoint.get("V"), probe=checkpoint.get("probe"),
                       run_size=run_size, n_runs=n_runs)
            _free = shutil.disk_usage(_ck_dir).free
            # Conditioning width is [S(41) | log T | forcing-or-chi]; the exact forcing width is not
            # resolved until the first batch returns, so bound it here -- this is a disk-space sanity
            # check, not an accounting figure. +8 covers the latent targets.
            _w = len(statistics.FEATURE_LABELS) + 1 + (
                config.CHI_ELEM_W * chi_k_pad if chi_mode else 8)
            _need = n_runs * run_size * (_w + 8) * 4
            print(f"[checkpoint] writing to {_ck_dir} every {_ck_every} batches "
                  f"(~{_need / 2 ** 30:.1f} GiB total, {_free / 2 ** 30:.1f} GiB free)", flush=True)
            if _free < _need:
                warnings.warn(
                    f"Only {_free / 2 ** 30:.1f} GiB free where the training checkpoint needs about "
                    f"{_need / 2 ** 30:.1f} GiB. The run will fail partway through a checkpoint "
                    f"write; free space now.", stacklevel=2)

    global _BATCH_TAG
    _pending_rng = None          # RNG as of the TOP of batch_k -- see the checkpoint write below
    _ck_from = _start_k          # first batch not yet committed to disk
    batch_k = _start_k           # bound up front: the except handler reads it, and an exception
    try:                         # before the first iteration would otherwise raise NameError there
      with torch.no_grad():
        for batch_k in tqdm(range(_start_k, n_runs), desc="Generating training data", leave=False,
                            initial=_start_k, total=n_runs):
            # The RNG snapshot for THIS batch, taken before the first draw below (sampling_dist.sample)
            # and therefore describing the state batch_k started from. A checkpoint committing batches
            # [0, k) stores the snapshot taken at the top of k, so restoring it puts every stream
            # exactly where the interrupted run's batch k began.
            #
            # Taken EVERY iteration rather than only on a cadence boundary, because a cancel can
            # arrive at any batch and must be able to commit the same way. A few KB of memcpy against
            # a ~20 s batch. Snapshot-and-restore, never replay: the OOM retries redraw SDE noise, so
            # per-batch RNG consumption is a function of what the desktop was doing.
            if _ck_dir is not None:
                _pending_rng = _tc.rng_snapshot(device, chi_gen)
            # --- Batch-level scale and duration (unchanged) ---
            t_scale_k = batch_t_scales[batch_k].item()
            T_k = batch_Ts[batch_k].item()
            T_nd_k = T_k / t_scale_k
            dt_nd_k = dt_exp / t_scale_k
            subsample_factor = max(1, round(dt_nd_k / dt_nd_min))
            N_points_k = int(T_nd_k / dt_nd_k)
            n_fine_total = steady_idx + N_points_k * subsample_factor
            t_fine = t[:n_fine_total]
            n_segs_k = max(1, math.ceil(n_fine_total / CHUNK_LEN))
            # Everything that determines this batch's memory footprint, in one string, set BEFORE the
            # first allocation that could fail. K is deliberately absent: the probe count is drawn
            # below, and the tag has to exist before force0 -- the largest single allocation in a chi
            # batch and a thoroughly plausible place to die.
            _BATCH_TAG = (f"training batch {batch_k + 1}/{n_runs} "
                          f"[t_scale={t_scale_k:.4g}, T={T_k:.4g}, n_fine={n_fine_total}, "
                          f"N_points={N_points_k}, rows={run_size}]")

            # 1. Sample inferred params. If theta_transform given, sampling_dist is latent.
            curr_thetas_raw = sampling_dist.sample((run_size,)).to(device=device, dtype=dtype)
            if theta_transform is not None:
                # prior is latent; lift to physical for the simulator
                curr_thetas_phys = theta_transform(curr_thetas_raw)
            else:
                curr_thetas_phys = curr_thetas_raw

            curr_thetas_nd      = curr_thetas_phys[:, :nd_dim]
            curr_thetas_rescale = curr_thetas_phys[:, nd_dim:]
            curr_thetas_forcing = (None if (spontaneous_only or chi_mode)
                                   else forcing_prior.sample((run_size,)).to(device=device, dtype=dtype))

            # Override t_scale to the batch-level value (in PHYSICAL space)
            curr_thetas_rescale[:, rescale_idx["t_scale"]] = t_scale_k

            # Recompute the latent to reflect the override; this is the training target.
            #
            # NOTE on the "non-finite training targets" concern (handoff 7.1): on torch 2.9 this
            # round-trip CANNOT produce +-inf. SigmoidTransform._inverse clamps its argument to
            # [tiny, 1-eps] internally, and sigmoid() saturates at 0.9999998807907104 rather than
            # exactly 1.0, so a theta on -- or even outside -- a box bound still inverts to a finite
            # latent (+-15.94 / -87.34). Verified for the linear box, the log box, out-of-box values
            # and the rotated transform. Deliberately NOT clamping here: it would buy nothing and
            # would perturb the parameters the simulator actually runs. train_nn's filter checks the
            # targets for finiteness and warns loudly, so if the transform stack ever changes the
            # invariant, it surfaces as a message rather than as a silently poisoned run.
            if theta_transform is not None:
                curr_thetas_latent = theta_transform.inv(curr_thetas_phys)
            else:
                curr_thetas_latent = curr_thetas_phys

            # --- THE PROBE SET FOR THIS BATCH, drawn ABOVE the retry seam ---
            # K is drawn per batch and the placement is stratified-jittered, so the encoder is trained
            # ACROSS probe counts and frequencies rather than memorising one grid. A fixed grid would
            # leave `u` taking only K distinct values in the entire training set, and an
            # experimentalist's 0.07x recording would then be an out-of-distribution input to an MLP
            # that extrapolates linearly and confidently.
            #
            # HOISTED OUT OF THE chi BRANCH so the two halves of a retried batch share one probe set.
            # That is what makes a halve a REPARTITION rather than a resample: the halves differ only
            # in SDE noise, so their concatenation is the batch that would have been produced. It also
            # keeps chi_gen advancing by a FIXED amount per batch, so a seeded run reproduces whether
            # or not the card happened to be short -- otherwise the training distribution would be a
            # function of what the desktop was doing. RNG-neutral: these three already preceded
            # _subset_probe_rows within a batch, so the stream is byte-identical to before the hoist.
            if chi_mode:
                k_b = (chi_k_fixed if chi_k_fixed is not None else
                       int(torch.randint(config.CHI_K_MIN_TRAIN, chi_k_pad + 1, (1,),
                                         generator=chi_gen).item()))
                b_mults = chi.sample_multipliers(k_b, chi_freq_bounds, generator=chi_gen,
                                                 dtype=dtype, device=device)
                # Per-probe duration: free (the samples already exist) and it makes the
                # (duration, frequency) trade-off an axis of the training distribution.
                dfrac = torch.where(torch.rand(k_b, generator=chi_gen) < 0.6,
                                    torch.ones(k_b),
                                    torch.exp(torch.rand(k_b, generator=chi_gen)
                                              * (math.log(1.0) - math.log(0.3)) + math.log(0.3)))

            def _rows(lo: int, hi: int):
                """This batch's rows [lo, hi) -> (hi-lo, W) CPU conditioning tensor.

                A CLOSURE, not a module-level function, because the body reads about thirty names from
                this scope -- the geometry, the thetas, the probe set, every mode flag. A thirty-
                argument function is a standing invitation to a call-site drift bug in code where a
                wrong subsample_factor produces a PLAUSIBLE training row rather than a crash. The
                logic worth testing -- the halving, the floor, the cancel pass-through -- lives in
                _rows_with_oom_retry, which is module-level and takes a fake fn. Called synchronously
                within this iteration and never stored, so the usual late-binding hazard of defining a
                closure in a loop does not apply.
                """
                n = hi - lo
                resc = curr_thetas_rescale[lo:hi]
                nd = curr_thetas_nd[lo:hi]
                init_rows = inits[lo:hi]
                # None in the chi and spontaneous branches, where there is no drive to slice.
                fparams = None if curr_thetas_forcing is None else curr_thetas_forcing[lo:hi]
                # Derived HERE rather than at batch level so the "this model has no x_offset" case
                # needs no branch: slicing the SOURCE works for both, slicing the float 0.0 does not.
                x_scale  = resc[:, rescale_idx["x_scale"]].unsqueeze(1)
                x_offset = (resc[:, rescale_idx["x_offset"]].unsqueeze(1)
                            if "x_offset" in rescale_idx else 0.0)

                if chi_mode:
                    # chi(omega) mode: spontaneous run (Groups A-F + Omega_0) + K single-tone forced runs.
                    # Conditioning [S(41, Group G zeroed) | log(T) | chi(3K)] -- chi replaces the forcing block.
                    force0 = _forcing.zero_force(n, n_force_ch, t_fine.shape[0], dtype, device)
                    x_nd_spont_fine = gen_obs(
                        model=model, params=nd, t=t_fine, inits=init_rows,
                        force=force0, n_segs=n_segs_k, steady_idx=steady_idx,
                        fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                        batch_size=n, var_idx=0, dtype=dtype, device=device,
                    )[0, :, :]
                    x_spont_dim = helpers.rescale(
                        x_nd_spont_fine[:, ::subsample_factor][:, :N_points_k], x_scale, x_offset)
                    del x_nd_spont_fine, force0
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        training_stats = gen_stats(x_spont_dim.cpu(), None, dt_exp, None, None, None,
                                                   device=device, spontaneous_only=True)   # (run, 41), G zeroed
                    chi_block, chi_mask = gen_chi_block(
                        # init_rows, NOT inits: this one is passed POSITIONALLY, which is exactly how
                        # it survived the row-slicing sweep -- gen_obs' calls name it `inits=` and were
                        # caught. It surfaced as "Batch size: 2 cannot differ from dim 0 of parameters
                        # tensor", i.e. loudly, only because the simulator validates the two against
                        # each other; a tensor that happened to broadcast would have gone unnoticed.
                        model, nd, resc, x_spont_dim, t_fine, init_rows, rescale_idx,
                        n_segs_k, steady_idx, subsample_factor, N_points_k, dt_exp,
                        b_mults, chi_f0, state_dep_drift=state_dep_drift, fixed_dict=fixed_dict,
                        k_pad=chi_k_pad, bounds=chi_freq_bounds, duration_frac=dfrac,
                        max_cycles=chi_max_cycles,
                        # THE ONLY adapt_placement=True in the codebase. Training is the one path that
                        # gets to choose where it probes; every other caller is reproducing an experiment
                        # whose frequencies are already fixed.
                        adapt_placement=True,
                        dtype=dtype, device=device)
                    # Per-ROW subsetting of the SAME drive set. Free -- the simulation is shared with the
                    # rows that keep the probe -- and it is the only way to decouple the probe count from
                    # the batch's (t_scale, T) stratum. It also hands the flow pairs of rows with the same
                    # drive set and different subsets, which is a direct regulariser toward K-agnosticism.
                    #   SKIPPED under chi_k_fixed: subsetting is exactly what makes the per-row count
                    #   vary, so leaving it on would silently turn a "K = 6" calibration stratum into a
                    #   mixture over 1..6 -- the pooled measurement the stratification exists to avoid.
                    if chi_k_fixed is None:
                        chi_block = _subset_probe_rows(chi_block, chi_mask, chi_k_pad, chi_gen)
                    log_T_k_tensor = torch.full((n, 1), math.log(T_k), dtype=dtype)
                    training_stats = torch.cat((training_stats, log_T_k_tensor, chi_block.cpu()), dim=-1)
                    return training_stats
                elif spontaneous_only:
                    # No drive: one spontaneous run (Groups A-F; Group G is zero-padded), no forcing block.
                    force = _forcing.zero_force(n, n_force_ch, t_fine.shape[0], dtype, device)
                    x_nd_spont_fine = gen_obs(
                        model=model, params=nd, t=t_fine, inits=init_rows,
                        force=force, n_segs=n_segs_k, steady_idx=steady_idx,
                        fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                        batch_size=n, var_idx=0, dtype=dtype, device=device,
                    )[0, :, :]
                    # Rescale STRAIGHT off the strided view: helpers.rescale materialises a fresh
                    # contiguous tensor, so nothing keeps a reference to the fine buffer and the `del`
                    # genuinely releases it. Binding the view to a name first (as this used to) pins the
                    # whole (n, n_fine) storage until that name dies -- the `del` frees nothing.
                    x_spont_dim = helpers.rescale(
                        x_nd_spont_fine[:, ::subsample_factor][:, :N_points_k], x_scale, x_offset)
                    del x_nd_spont_fine, force
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        training_stats = gen_stats(x_spont_dim.cpu(), None, dt_exp, None, None, None,
                                                   device=device, spontaneous_only=True)
                        log_T_k_tensor = torch.full((n, 1), math.log(T_k), dtype=dtype)
                        # Conditioning [S | log(T)] -- no forcing block (forcing_dim = 0).
                        training_stats = torch.cat((training_stats, log_T_k_tensor), dim=-1)
                        return training_stats
                else:
                    # 2. Build nondimensional force tensor at fine resolution (uses PHYSICAL rescale)
                    force = build_nondim_sin_force_tensor(
                        fparams, t_fine, resc, forcing_idx, rescale_idx
                    )

                    # 3. Simulate the FORCED run (drive on) -> Group G
                    x_nd_fine = gen_obs(
                        model=model, params=nd, t=t_fine, inits=init_rows,
                        force=force, n_segs=n_segs_k, steady_idx=steady_idx,
                        fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                        batch_size=n, var_idx=0, dtype=dtype, device=device,
                    )[0, :, :]
                    # 4a. Redimensionalize the forced run IMMEDIATELY (uses PHYSICAL rescale).
                    # Order matters: helpers.rescale materialises a fresh contiguous tensor, so the `del`
                    # below actually releases the fine buffer. Holding the strided VIEW in a name instead
                    # (as this used to) pinned the entire (n, n_fine) storage right across the
                    # second gen_obs call below -- two full fine trajectories resident where one
                    # subsampled slice was needed, several GB at n=2048. That also made
                    # _max_sim_batch split batches it did not need to, and k chunks cost k x wall-clock.
                    x_dim = helpers.rescale(
                        x_nd_fine[:, ::subsample_factor][:, :N_points_k], x_scale, x_offset)
                    del x_nd_fine

                    # 3b. Simulate the SPONTANEOUS run (zero force) -> Groups A-F
                    x_nd_spont_fine = gen_obs(
                        model=model, params=nd, t=t_fine, inits=init_rows,
                        force=torch.zeros_like(force), n_segs=n_segs_k, steady_idx=steady_idx,
                        fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                        batch_size=n, var_idx=0, dtype=dtype, device=device,
                    )[0, :, :]
                    # 4b. Same treatment for the spontaneous run.
                    x_spont_dim = helpers.rescale(
                        x_nd_spont_fine[:, ::subsample_factor][:, :N_points_k], x_scale, x_offset)
                    del x_nd_spont_fine, force

                    # 5. Stats (A-F from spontaneous, G from forced) + conditioning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        drive_amp = fparams[:, forcing_idx["amp"]].cpu()
                        drive_freq = fparams[:, forcing_idx["freq"]].cpu()
                        drive_phase = fparams[:, forcing_idx["phase"]].cpu()
                        training_stats = gen_stats(x_spont_dim.cpu(), x_dim.cpu(), dt_exp, drive_amp, drive_freq, drive_phase, device=device)
                        log_T_k_tensor = torch.full((n, 1), math.log(T_k), dtype=dtype)
                        # Canonical conditioning layout: [S(x_dim) | log(T) | theta_force].
                        # log(T) rides with the summary pathway; theta_force is a separate block.
                        # The embedding split in build_posterior depends on this exact order, so
                        # keep it in sync with generate_observations / validate / infer_from_experiment.
                        training_stats = torch.cat((training_stats, log_T_k_tensor, fparams.cpu()), dim=-1)
                        return training_stats

            # Per-ROW element cost at THIS geometry, in the same currency _max_sim_batch plans in, so
            # an OOM here tightens the SAME learned cap the predictive guard reads rather than
            # inventing a second, incompatible notion of "too big".
            _seg = min(n_fine_total, CHUNK_LEN)
            _n_keep = max(0, n_fine_total - steady_idx)              # var_idx=0 on every path here
            _per_row = (inits.shape[-1] * n_fine_total + n_force_ch * n_fine_total
                        + max(inits.shape[-1] * _seg, _n_keep))
            _rows_out = _rows_with_oom_retry(
                _rows, 0, run_size, per_row_elements=_per_row, device=device)

            # 6. Collect LATENT targets (not physical). OUTSIDE the retry on purpose: the targets are
            # computed before any simulation and are already at full width, so a retry neither
            # recomputes nor re-draws them -- the halves REPARTITION the batch's rows, they do not
            # resample it.
            _th_out = curr_thetas_latent.cpu()
            if x_buf is None:
                x_buf = torch.empty((n_runs * run_size, _rows_out.shape[-1]), dtype=_rows_out.dtype)
                th_buf = torch.empty((n_runs * run_size, _th_out.shape[-1]), dtype=_th_out.dtype)
            _lo, _hi = batch_k * run_size, (batch_k + 1) * run_size
            x_buf[_lo:_hi] = _rows_out
            th_buf[_lo:_hi] = _th_out
            del _rows_out, _th_out
            if device.type == "cuda":
                torch.cuda.empty_cache()
                # cuFFT caches a plan per distinct transform SHAPE, outside PyTorch's caching
                # allocator -- so empty_cache() above cannot touch it and it surfaces as a RAW
                # driver cudaErrorMemoryAllocation rather than torch.cuda.OutOfMemoryError.
                # N_points_k changes every batch, so cross-batch plan reuse is exactly zero while
                # each batch mints ~7 new signatures (6 from SummaryStatistics, 1 from
                # chi.peak_freq) at ~2 MB apiece; the default 4096-entry cache would saturate
                # around batch ~585 of 5000 and hold ~8.6 GB hostage. Clearing per batch costs
                # nothing (the intra-batch reuse across gen_stats' sub-batches already happened)
                # and is preferable to shrinking cufft_plan_cache.max_size, which WOULD thrash it.
                torch.backends.cuda.cufft_plan_cache.clear()
            # Batch 0 gives an immediate baseline (so a run that is doomed says so in the first
            # minute rather than the fifth hour); after that, one line per _MEM_LOG_EVERY batches.
            if batch_k == 0 or (batch_k + 1) % _MEM_LOG_EVERY == 0:
                _log_memory(device, _BATCH_TAG)

            # --- checkpoint the completed batches [_ck_from, batch_k + 1) ---
            # _pending_rng is this batch's OPENING state, so the write records batches [0, k+1) with
            # the state batch k+1 will start from -- which is the snapshot the NEXT iteration takes.
            # Hence the write below uses the snapshot taken at the top of the following iteration; we
            # take a fresh one here for exactly that reason.
            if _ck_dir is not None and _ck_every and (batch_k + 1) % _ck_every == 0:
                _tc.save(_ck_dir, from_batch=_ck_from, batch_k=batch_k + 1,
                         rng=_tc.rng_snapshot(device, chi_gen),
                         x_buf=x_buf, th_buf=th_buf, run_size=run_size)
                _ck_from = batch_k + 1

    except BaseException:
        # WorkerCancelled (a BaseException by design, so `except Exception` would miss it) and
        # KeyboardInterrupt both land here, and a GUI cancel is the MOST likely way a multi-day run
        # ends -- MainWindow.closeEvent reaches request_cancel_all(), so closing the window stops it.
        # Before this, that discarded every completed batch.
        if _ck_dir is not None and _pending_rng is not None and batch_k > _ck_from:
            try:
                # Announced BEFORE the write, so a multi-second flush is not an unexplained hang after
                # Cancel. Safe to print here even under a cancel: CancelToken.fired is a one-shot
                # latch, so the raise has already happened and later writes pass through. Nothing is
                # printed BETWEEN the shard fsync and the state replace -- see training_checkpoint.
                print(f"[checkpoint] stopping: saving {batch_k - _ck_from} completed batches "
                      f"({_ck_from} -> {batch_k}) before unwinding…", flush=True)
                _tc.save(_ck_dir, from_batch=_ck_from, batch_k=batch_k,
                         rng=_pending_rng, x_buf=x_buf, th_buf=th_buf, run_size=run_size)
            except Exception as _e:              # noqa: BLE001
                # A failed rescue write must never REPLACE the cancel/crash with an I/O error.
                print(f"[checkpoint] could not save on the way out: {_e}", file=sys.stderr, flush=True)
        raise                                    # UNCONDITIONAL: never swallow a cancel
    finally:
        # Cleared however we leave -- return, OOM, or a cooperative cancel. A stale tag would make the
        # NEXT failure anywhere in the process claim a batch that finished hours ago, which is worse
        # than no tag at all.
        _BATCH_TAG = ""

    if x_buf is None:                       # n_runs == 0: nothing was generated, and nothing to size from
        return torch.empty((0, 0)), torch.empty((0, 0))
    if _ck_dir is not None:
        # Commit whatever the last cadence boundary left, then mark it done. A COMPLETE checkpoint is
        # deliberately not deleted: it is a several-GiB cache of a multi-day simulation, and it is what
        # lets the flow be retrained (different capacity, learning rate, epochs) without re-simulating.
        if n_runs > _ck_from:
            _tc.save(_ck_dir, from_batch=_ck_from, batch_k=n_runs,
                     rng=_tc.rng_snapshot(device, chi_gen),
                     x_buf=x_buf, th_buf=th_buf, run_size=run_size)
        _tc.mark_complete(_ck_dir, n_runs)
        print(f"[checkpoint] complete: {n_runs} batches in {_ck_dir}. Safe to delete once the "
              f"posterior is saved; keeping it lets you retrain the flow without re-simulating.",
              flush=True)
    return x_buf, th_buf

def train_nn(training_params: dict, model: str, prior: torch.distributions.Distribution, embedding_net: torch.nn.Module,
             forcing_prior: torch.distributions.Distribution, nd_dim: int, forcing_idx: dict, rescale_idx: dict,
             x_obs: torch.Tensor = None, theta_obs: torch.Tensor = None, num_rounds: int = 1, return_diagnostics: bool = False, theta_transform: Transform | None = None,
             fixed_dict: dict = None,
             hidden_features: int = 50, num_transforms: int = 5, num_bins: int = 10,
             learning_rate: float = 5e-4, stop_after_epochs: int = 20, max_num_epochs: int = 2_147_483_647,
             show_train_summary: bool = False,
             batch_size: int = 128, device: torch.device = torch.device('cpu')) -> DirectPosterior | tuple[DirectPosterior, dict]:
    """
    Trains a neural posterior distribution using either Neural Posterior Estimation (NPE) or Sequential Neural Posterior
    Estimation (SNPE), depending on the number of training runs specified. The method automates simulation-based
    learning by generating synthetic data, training a density estimator, and refining a posterior iteratively if multiple
    training runs are performed.

    :param training_params: A dictionary of parameters required to generate training data. These parameters are used as input
        for the data generation function. Check @gen_training_data for details of the order of the parameters.
    :param model: The type of neural density estimator to use, specified as a string. It determines the architecture of the
        neural network approximating the posterior distribution.
    :param prior: The prior distribution over parameters, given as a `torch.distributions.Distribution` object.
    :param embedding_net: A neural network module that is used to compute embeddings of the data.
    :param x_obs: Observed data given as a `torch.Tensor`. Required when performing SNPE (i.e., `num_runs > 1`). Defaults
        to None.
    :param theta_obs: Observed parameters given as a `torch.Tensor`. Required when returning diagnostics. Defaults to None.
    :param num_rounds: The number of sequential training runs. If greater than 1, Sequential Neural Posterior Estimation (SNPE)
        is performed. Defaults to 1.
    :param return_diagnostics: Whether to return additional diagnostics such as loss values during training. Defaults to False.
    :param fixed_dict: Dictionary of fixed parameters for the model. Defaults to None.
    :param hidden_features: Hidden units per flow transform (density-estimator capacity).
    :param num_transforms: Number of flow transforms / coupling layers (capacity).
    :param num_bins: Spline bins per transform (NSF only).
    :param learning_rate: Adam learning rate for training.
    :param stop_after_epochs: Early-stopping patience in epochs.
    :param max_num_epochs: Hard cap on the number of training epochs.
    :param show_train_summary: If True, print sbi's per-epoch train/validation-loss summary.
    :param batch_size: Batch size for training the density estimator during each run. Defaults to 128.
    :param device: Device on which the computations should be performed (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
    :return: A `NeuralPosterior` object representing the trained posterior distribution. If 'return_diagnostics = True', return a tuple containing
        the posterior and diagnostics.
    """
    if num_rounds > 1 and x_obs is None:
        raise ValueError("x_obs must be specified for SNPE algorithm")
    if num_rounds > 1 and training_params.get("checkpoint") is not None:
        # Refused loudly rather than half-supported. Rounds >= 2 sample from a PROPOSAL -- a trained
        # DirectPosterior -- whose identity a checkpoint would have to capture and re-validate, which
        # is a separate problem from the one C-11 solves. TRAINING_NUM_ROUNDS is 1 (amortized NPE), so
        # this costs nothing today and closes the hole rather than leaving it to be discovered.
        raise ValueError(
            f"Training-data checkpointing is not supported for SNPE (num_rounds={num_rounds}); the "
            f"per-round proposal is not part of the checkpoint's identity. Use num_rounds=1, or "
            f"drop training_params['checkpoint'].")

    # sbi's default z_score_x="independent" fits a PER-COLUMN affine over the conditioning vector.
    # Under the chi SET layout that is permutation-BREAKING (two orderings of one probe set would be
    # scaled differently, destroying the encoder's whole guarantee), and the near-constant mask column
    # becomes a ~1e7 amplifier under sbi's 1e-7 min-std clamp. The chi net standardizes itself instead
    # -- per channel, over live probes only. Derived from the net, never a separate argument, so the
    # standardizer and the encoder cannot be configured apart.
    _owns = getattr(embedding_net, "owns_standardization", False)
    neural_posterior = posterior_nn(model=model, embedding_net=embedding_net,
                                    z_score_x=("none" if _owns else "independent"),
                                    hidden_features=hidden_features, num_transforms=num_transforms,
                                    num_bins=num_bins)
    infer = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    proposal = None # set up initial proposal distribution
    posterior = None

    # diagnostics storage
    diagnostics = {
        "log_prob_true": [],
        "posterior_means": [],
        "posterior_stds": [],
    }

    for _ in tqdm(range(num_rounds), desc=f"Training neural posterior", leave=False):
        # train the density estimator
        data, thetas = gen_training_data(
            training_params["model"], training_params["prior"], forcing_prior, training_params["t"],
            training_params["run_size"], training_params["num_runs"],
            training_params["steady_idx"], training_params["dt_nd_min"],
            nd_dim, forcing_idx, rescale_idx,
            dt_exp=training_params["dt_exp"], t_min_exp=training_params["t_min_exp"],
            t_max_exp=training_params["t_max_exp"], t_scale_bounds=training_params["t_scale_bounds"],
            proposal=proposal,
            theta_transform=theta_transform,
            fixed_dict=fixed_dict,
            state_dep_drift=training_params.get("state_dep_drift", False),
            spontaneous_only=training_params.get("spontaneous_only", False),
            chi_mode=training_params.get("chi_mode", False),
            chi_f0=training_params.get("chi_f0", None),
            chi_freq_bounds=training_params.get("chi_freq_bounds", None),
            chi_k_pad=training_params.get("chi_k_pad", None),
            chi_k_fixed=training_params.get("chi_k_fixed", None),
            chi_max_cycles=training_params.get("chi_max_cycles", None),
            n_vars=training_params.get("n_vars", None),
            checkpoint=training_params.get("checkpoint", None),
            dtype=training_params["dtype"], device=training_params["device"],
        )

        # Filter the data -- and the TARGETS. thetas is the LATENT target (a logit); its row can in
        # principle go non-finite while the corresponding data row stays perfectly finite, and
        # filtering thetas only BY data would then feed the flow a +-inf target and NaN the loss for
        # the whole round with no diagnostic. On torch 2.9 the box round-trip cannot actually produce
        # one (SigmoidTransform._inverse clamps internally -- see gen_training_data), so this should
        # never fire; it exists so that if the transform stack ever changes, the failure arrives as a
        # message naming the offending columns instead of as a silently poisoned multi-hour run.
        nan_mask = torch.isfinite(data).all(dim=1)
        safe_magnitude_mask = (torch.abs(data) < 1e15).all(dim=1)
        theta_finite_mask = torch.isfinite(thetas).all(dim=1)
        valid_idx = nan_mask & safe_magnitude_mask & theta_finite_mask
        n_bad_theta = int((~theta_finite_mask).sum())
        if n_bad_theta:
            bad_cols = torch.nonzero(~torch.isfinite(thetas).all(dim=0)).flatten().tolist()
            warnings.warn(
                f"train_nn: dropped {n_bad_theta}/{thetas.shape[0]} training rows with non-finite "
                f"LATENT targets (columns {bad_cols}). The box round-trip is supposed to make this "
                f"impossible -- treat it as a bug in the transform stack, not as expected attrition.",
                stacklevel=2,
            )
        # Only pay for the gather when it actually drops something. `data[valid_idx]` is a boolean
        # gather: it allocates a SECOND full-size tensor while the first is still live, which at the
        # production shape is another 4.35 GiB and reinstates exactly the 8.7 GiB host peak
        # gen_training_data's preallocated accumulators were introduced to remove. The mask is
        # all-true in practice (the box round-trip cannot produce a non-finite latent on torch 2.9 --
        # trap X4), so this is behaviour-identical and is what makes that preallocation pay.
        if not bool(valid_idx.all()):
            thetas = thetas[valid_idx]
            data = data[valid_idx]

        # Fit OUR standardizers, once, on the post-filter data. posterior_nn was already called (it is
        # constructed before any data exists), so this has to happen here -- before append_simulations,
        # because build_nsf/get_numel run the net inside infer.train(). Unfitted, 41 raw statistics
        # spanning log-variances, nm-scale means and PLVs would reach the flow unscaled and the only
        # symptom would be a worse loss curve, days later; the encoder raises rather than allow it.
        if _owns and not embedding_net.standardization_fitted:
            embedding_net.fit_standardization(data)
            assert embedding_net.standardization_fitted, "chi standardization silently did not fit"

        # data_device="cpu": sbi otherwise defaults it to the TRAINING device (npe_base.py:174-175)
        # and moves the whole conditioning tensor onto the GPU -- 10.24M x 114 float32 = 4.35 GiB
        # resident, an 8.7 GiB transient for the `x = x[is_valid_x]` filter, two (10.24M,) bool masks
        # from handle_invalid_x, and later a ~3.9 GiB gather when it slices the training split. All of
        # that on the same card that has to hold the flow, and all of it AFTER a multi-day generation
        # run that is not checkpointed. The data is ALREADY on the CPU here (gen_training_data appends
        # .cpu() tensors), so this also silences validate_theta_and_x's "Moving x to the data_device"
        # warning, which describes a 4.35 GiB copy nobody asked for.
        #
        # Training does not care: sbi's loop moves each minibatch to the training device itself, so a
        # CPU-resident dataset costs one host-to-device copy of (training_batch_size, 114) float32 --
        # tens of KB -- against a flow forward+backward that dominates it by orders of magnitude. It
        # hands 4.35 GiB of VRAM back to the flow, which is a straight win.
        with _capped_zscore_check():
            infer.append_simulations(thetas, data, proposal=proposal, data_device="cpu")
        density_estimator = infer.train(
            training_batch_size=batch_size, learning_rate=learning_rate,
            stop_after_epochs=stop_after_epochs, max_num_epochs=max_num_epochs,
            show_train_summary=show_train_summary,
        )
        posterior = infer.build_posterior(density_estimator)
        assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"

        # compute diagnostics after each round
        if return_diagnostics and x_obs is not None:
            x_obs_device = x_obs.to(device)

            # log probability of ground truth
            if theta_obs is not None:
                theta_true_device = theta_obs.to(device)
                if theta_true_device.dim() == 1:
                    theta_true_device = theta_true_device.unsqueeze(0)
                log_prob = posterior.log_prob(theta_true_device, x=x_obs_device).item()
                diagnostics["log_prob_true"].append(log_prob)

            # posterior mean and std from samples
            samples = posterior.sample((10000,), x=x_obs_device)
            diagnostics["posterior_means"].append(samples.mean(dim=0).cpu())
            diagnostics["posterior_stds"].append(samples.std(dim=0).cpu())

        # need to now check if num_runs > 1: if so, then that is equivalent to SNPE, and if not, that is equivalent to NPE
        if num_rounds > 1:
            assert x_obs is not None, "x_obs must be specified for SNPE algorithm"
            proposal = posterior.set_default_x(x_obs.to(device)) # if SNPE, the user has to specify x_obs

    assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"

    # Capture the per-epoch train/validation loss curve from the sbi trainer so the
    # convergence ("under-fit vs converged") check is reproducible. sbi keeps these in
    # infer._summary; they are otherwise discarded when this function returns.
    if return_diagnostics:
        summary = getattr(infer, "_summary", {}) or {}
        diagnostics["training_loss"] = list(summary.get("training_loss", []))
        diagnostics["validation_loss"] = list(summary.get("validation_loss", []))
        best_val = summary.get("best_validation_loss", [])
        diagnostics["best_validation_loss"] = best_val[-1] if len(best_val) else None
        epochs = summary.get("epochs_trained", [])
        diagnostics["epochs_trained"] = epochs[-1] if len(epochs) else None
        diagnostics["stop_after_epochs"] = stop_after_epochs
        return posterior, diagnostics
    return posterior