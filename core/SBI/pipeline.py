import dataclasses
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sbi.inference.posteriors import DirectPosterior
from torch.distributions.transforms import Transform

from core import forcing as _forcing
from core.Helpers import helpers
from core import config
from core.config import CHUNK_LEN, N_ND_MAX
from core.Simulator import bp_simulator, nadrowski_simulator, hopf_simulator
from core.SBI import statistics, chi, derived

VALID_SIMS: dict = {"bp":        bp_simulator.BPSimulator,
                    "nadrowski": nadrowski_simulator.NadrowskiSimulator,
                    "hopf":      hopf_simulator.HopfSimulator}


INIT_SHAPES: dict = {"bp":        (2, 3),
                     "nadrowski": (2, 1),
                     "hopf":      (2, 0)}

def sim_class(model: str):
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

# Peak device residency of forcing.build_nondim_force_tensor, as a multiple of the (batch, n_ch, T)
# tensor it RETURNS. Derived by counting the eager allocations on the "sin" path (core/forcing.py):
# t_dim and sin_term stay live to the end and the four elementwise ops each hold one result-sized
# transient, so the high-water mark is 4R (the returned unsqueeze(1) is a view). MEASURED 4.10x on
# the 5070 Ti (2026-08-27, B=64/T=20000, max_memory_allocated around the call; independently a
# 2.16 GiB result was seen with an 8.64 GiB transient). The 0.10 is (batch, 1) parameter columns
# that do not scale with T, so 4 is charged, not 5 -- over-charging would split batches that fit.
# tests/test_user_sbi.py re-measures this on any box with CUDA.
_FORCE_BUILD_PEAK_MULTIPLE = 4


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


def _budget_note_ok(*, batch_level: bool = False) -> None:
    """Record a clean unit of work; after enough of them, probe the cap upward.

    The recovery half matters as much as the backoff: a desktop that was holding 6 GB when the first
    OOM landed may have closed a browser since, and without this the run would stay throttled to that
    moment for days. It probes multiplicatively but is re-clamped by memory_budget_elements on every
    call to _max_sim_batch, so it can never climb past what the (optimistic) reading allows anyway --
    the cap only ever makes the plan MORE conservative than that reading, never less.

    THE UNIT IS THE TRAINING BATCH, NOT THE gen_obs CALL: a chi batch makes 1 + K gen_obs calls
    (spontaneous plus one per probe), so counting calls probed the cap upward every ~3 batches
    instead of every _BUDGET_RECOVER_AFTER=32 -- the 0.8x backoff unwound in three batches and the
    throttle never held, which is why a busy card produced repeated OOMs instead of settling into a
    slower surviving state. Hence: inside gen_training_data (``_BATCH_TAG`` non-empty) only the
    batch-level call counts, one per completed batch. Outside it (the PPC, decorrelate, the prior
    sweeps) ``_BATCH_TAG`` is "" and every call counts -- those callers have no batch to speak of.
    """
    global _BUDGET_CAP_ELEMENTS, _budget_clean_runs
    if _BATCH_TAG and not batch_level:
        return                   # inside a training batch: the batch tail does the accounting
    if _BUDGET_CAP_ELEMENTS is None:
        return
    _budget_clean_runs += 1
    if _budget_clean_runs >= _BUDGET_RECOVER_AFTER:
        _budget_clean_runs = 0
        _BUDGET_CAP_ELEMENTS = int(_BUDGET_CAP_ELEMENTS * _BUDGET_RECOVER_STEP)


# ── Where are we? ─────────────────────────────────────────────────────────────────────────────────
# The GUI log shows warnings/failures without tracebacks, and the 2026-08-11 chi retrain died with a
# bare "CUDA error: out of memory" that could not be placed beyond "inside the per-batch loop". One
# f-string per batch (microseconds against a ~20 s batch) removes that class of forensics. A module
# global rather than a parameter because the consumers (gen_chi_block, the OOM retries) have no
# business taking a batch index; single-writer by construction (one gen_training_data at a time per
# process), and the only consumer is a log string, so the worst a concurrent run could do is mislabel.
_BATCH_TAG = ""


def _batch_tag() -> str:
    """The batch currently being generated, or a neutral label outside gen_training_data."""
    return _BATCH_TAG or "simulation"


# 21 lines over a 5000-batch run. PRISM_MEM_LOG_EVERY overrides it for a diagnostic run: when a run
# is dying at batch 93, a line every 250 batches has told you nothing at all.
_MEM_LOG_EVERY = int(os.environ.get("PRISM_MEM_LOG_EVERY") or 250)


def _log_memory(device: torch.device, tag: str) -> None:
    """One memory line, and RESET the peak so the next one describes the next interval.

    PEAK allocated is the number that predicts an OOM. An instantaneous reading taken between batches
    is always low, because the batch's big tensors are already gone by then -- so a series of those
    would look flat right up until the run died. The allocator maintains the peak unconditionally, so
    reading it is free, and resetting it turns the series into "worst batch in the last 250", which is
    the thing that trends upward before a card runs out.

    Reported-free is printed with its health warning attached: under WDDM other processes' evictable
    surfaces count as free -- measured 15037 MiB against nvidia-smi's 5814 on this machine.

    ⚠ A DIAGNOSTIC MUST NEVER BE ABLE TO KILL THE RUN. All four device calls here can raise on a
    starved card, and this runs on the success path right after a batch may have clawed through all
    three OOM ladders -- the card at its most degraded. Best-effort; a failure degrades to a note.
    """
    if device.type != "cuda":
        return
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        cap = ("none" if _BUDGET_CAP_ELEMENTS is None
               else f"{_BUDGET_CAP_ELEMENTS * 4 / 2 ** 30:.2f} GiB")
        line = (f"[mem] {tag}: peak allocated "
                f"{torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f} GiB, peak reserved "
                f"{torch.cuda.max_memory_reserved(device) / 2 ** 30:.2f} GiB, "
                f"{free_b / 2 ** 30:.2f}/{total_b / 2 ** 30:.2f} GiB reported free "
                f"(optimistic on Windows), learned cap {cap}")
    except Exception as e:                   # noqa: BLE001 -- see the docstring
        line = f"[mem] {tag}: memory statistics unavailable ({_short_err(e, 120)})"
    print(line, file=sys.stderr, flush=True)
    # Outside the try on purpose: if the reads above failed, the peak was never reported, so
    # resetting it would discard the interval's high-water mark and the NEXT line would understate.
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:                        # noqa: BLE001
        pass


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


def _free_gib_note(device: torch.device) -> str:
    """", 3.71 GiB reported free (optimistic on Windows)" -- or "" if the reading is unavailable.

    Decoration on a log line, and nothing more, so it must never be the thing that raises. On a card
    that is already refusing allocations ``mem_get_info`` is itself a driver call that can fail, and
    it sits in the OOM notices -- i.e. on the one path where an extra exception is most expensive.
    The health warning rides along because the reading OVERSTATES free VRAM on Windows by roughly the
    size of the desktop (measured 15037 MiB reported against nvidia-smi's 5814 at the same instant).
    """
    if device.type != "cuda":
        return ""
    try:
        return (f", {torch.cuda.mem_get_info(device)[0] / 2 ** 30:.2f} GiB reported free "
                f"(optimistic on Windows -- see the learned-budget note)")
    except Exception:                        # noqa: BLE001 -- see the docstring
        return ", free memory unreadable"


def _release_device_memory(device: torch.device, *, plans: bool = True,
                           graphs: bool = True) -> None:
    """Hand back everything reclaimable on ``device``. BEST EFFORT: this function NEVER raises.

    Called across module lines (orchestrator's PPC loop) and named as a literal needle by the
    order-pinned retry tests in tests/test_user_sbi.py -- do not rename it.

    ``plans`` and ``graphs`` select the two resources that live OUTSIDE the caching allocator, and
    they default to on because the callers that matter are RECOVERY paths. Turn them off in a hot
    loop -- see "NOT EVERY CALLER WANTS ALL THREE" below.

    WHY THE GUARD EXISTS. On 2026-08-27 the retrain died at batch 351/5000 with a raw
    ``AcceleratorError: CUDA error: out of memory`` raised BY ``torch.cuda.empty_cache()`` inside
    _rows_with_oom_retry -- the OOM was caught correctly and the run was killed by its own recovery.
    A release that cannot free anything is INFORMATION, not a reason to abandon a multi-day run.
    And because the release sits OUTSIDE the ``except`` block (see below), Python has already
    cleared the exception context -- the secondary failure arrives with ``__context__`` None and a
    traceback saying nothing about the OOM that started it, which is the traceback that crash
    produced.

    ``except Exception``, NEVER ``BaseException``: streams.WorkerCancelled derives from BaseException
    precisely so a cooperative cancel sails through handlers like this one to reach Worker.run.

    THREE RESOURCES, THREE SEPARATE GUARDS, because they fail independently and each is worth having
    even if the others could not run:
      * the caching allocator's cached-but-unused segments (``empty_cache``);
      * the cuFFT plan cache, which lives OUTSIDE that allocator so empty_cache cannot touch it --
        ~2 MB per distinct transform shape, ~7 new signatures per training batch, zero cross-batch
        reuse (the 2026-07-28 leak);
      * the captured CUDA graphs, whose memory lives in PRIVATE pools that empty_cache also cannot
        reclaim. At an OOM up to SOLVER_GRAPH_CACHE_MAX of them are pinned and the halving retry is
        about to capture ANOTHER at the reduced width, since the batch shape is part of the graph
        key. Dropping them is the one release that addresses the retry's own next allocation.

    CALL IT OUTSIDE THE ``except`` BLOCK, always. While the caught error is still bound its traceback
    owns the frames of the failed attempt, which own their tensors -- the solver's (n_vars, batch, T)
    buffer among them -- so releasing there frees nothing. Python drops the error when the clause
    ends; only then is this worth asking for.

    NOT EVERY CALLER WANTS ALL THREE, and the defaults are tuned for the recovery paths:
      * a hot loop (gen_stats' sub-batches, gen_chi_raw's probes) passes ``plans=False,
        graphs=False``. Clearing the plan cache there would destroy the INTRA-batch cuFFT reuse
        those loops exist to get, and dropping graphs would force a recapture per probe -- against
        a solver whose graph replay is an ~8x speedup, that is a large regression bought for
        nothing, because a loop that is not failing is not short of memory.
      * gen_training_data's per-batch tail passes ``graphs=False`` for the same reason but keeps
        ``plans=True``: N_points_k changes every batch so cross-batch plan reuse is exactly zero,
        and the cache would otherwise saturate mid-run.
      * the OOM retries take the default, all three, because the card is provably short and the
        retry's own next allocation is a graph capture at the halved width.
    """
    if device.type != "cuda":
        return

    def _try(fn, what):
        try:
            fn()
        except Exception as e:               # noqa: BLE001 -- see the docstring
            print(f"{_batch_tag()}: {what} failed during recovery and was ignored "
                  f"({_short_err(e, 120)})", file=sys.stderr, flush=True)

    _try(torch.cuda.empty_cache, "empty_cache()")
    if plans:
        _try(torch.backends.cuda.cufft_plan_cache.clear, "cuFFT plan-cache clear")
    if graphs:
        def _drop():
            from core.Solvers import sdeint as _sdeint
            _sdeint.drop_graph_cache()
        _try(_drop, "CUDA-graph cache drop")


def _we_are_the_holder(device: torch.device) -> bool:
    """True when THIS process's own reserved pool is more than everyone else's usage combined.

    ⚠ WAITING ONLY HELPS IF SOMEBODY ELSE HOLDS THE MEMORY. The batch-level retry was written for the
    documented failure -- a card momentarily full of the desktop's evictable surfaces -- where pausing
    is exactly right. It is exactly WRONG when we are the holder: on 2026-08-28 a stuck run sat at
    batch 93 repeating "waiting for device memory", holding 15310 MB of a 16303 MB card while every
    other process on the machine held ~270 MB combined. It was waiting for itself, and no delay could
    ever have satisfied it.

    Measured against the DRIVER's totals rather than our own bookkeeping alone, because the question
    is whose memory it is. `reserved` counts what PyTorch has taken from the driver, cached blocks
    included -- which is the right quantity: if empty_cache could have returned it, it already did,
    since the release runs before this is consulted.
    """
    if device.type != "cuda":
        return False
    try:
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
    except Exception:                        # noqa: BLE001 -- unknowable => fall back to waiting
        return False
    others = max(0, total - free - reserved)
    return reserved > others


def _cancellable_wait(seconds: float, why: str) -> None:
    """Sleep ``seconds``, in slices, staying responsive to a GUI cancel and saying why we are idle.

    A PLAIN time.sleep() CANNOT BE CANCELLED HERE. Cancellation in this app is cooperative and is
    raised from CancelToken.check() inside the redirected stream's write() on the worker thread
    (core/gui/streams.py) -- so a run that is sleeping is a run that cannot notice the Cancel button
    until it wakes. Sleeping in ~1 s slices and PRINTING between them gives the latch its chance,
    and a multi-minute silent pause in a run that has already logged an OOM would otherwise read as
    a hang at precisely the moment the user is most likely to reach for Cancel.
    """
    end = time.monotonic() + max(0.0, seconds)
    last_note = 0.0
    while True:
        left = end - time.monotonic()
        if left <= 0:
            break
        time.sleep(min(1.0, left))
        # One line every ~5 s: enough to drive the cancel check and to show progress, few enough
        # that a 180 s wait costs 36 log rows rather than 180.
        now = time.monotonic()
        if now - last_note >= 5.0:
            last_note = now
            print(f"{_batch_tag()}: {why} -- {max(0.0, end - now):.0f}s remaining",
                  file=sys.stderr, flush=True)


def _short_err(err: BaseException, limit: int = 200) -> str:
    """``"TypeName: first line of the message"``, safely, for a log line.

    ``str(err).splitlines()[0]`` RAISES IndexError ON AN EMPTY MESSAGE, because "".splitlines() is
    [] rather than [""]. That is not hypothetical here: _is_oom returns True on the TYPE test alone
    (torch.OutOfMemoryError), so a zero-message OOM reaches the retry ladders' note lines -- where an
    IndexError would REPLACE the out-of-memory with a confusing traceback from inside the handler,
    and where _release_device_memory's own error path would violate its "never raises" contract.
    """
    # ⚠ THE REGRESS STOPS HERE. This function is called from inside the `except` clause of every
    # guard in this module -- _release_device_memory._try, _try_rng_snapshot, _try_rng_restore,
    # _log_memory -- so if IT can raise, all of them can, and the whole best-effort layer is a
    # fiction. `str(err)` is not free: an exception whose __str__ raises (or whose args hold an
    # object with a broken __repr__) takes the guard down with it. So this one has no error path of
    # its own: it is total, and the type name is always available without touching the payload.
    try:
        text = str(err)
    except Exception:                        # noqa: BLE001 -- see above
        return type(err).__name__
    lines = text.splitlines()
    first = lines[0] if lines else ""
    return f"{type(err).__name__}: {first[:limit]}" if first else type(err).__name__


def _try_rng_snapshot(tc, device, chi_gen):
    """``rng_snapshot`` that returns None instead of raising. See _try_rng_restore for why.

    ``torch.cuda.get_rng_state_all()`` enters a device context and reads each generator's state, so
    it is a driver call like any other and it fails like any other on a starved card -- and it runs
    at the TOP OF EVERY BATCH. A run must not die because it could not record where its random
    streams were.
    """
    try:
        return tc.rng_snapshot(device, chi_gen)
    except Exception as e:                   # noqa: BLE001
        print(f"{_batch_tag()}: could not snapshot the RNG ({_short_err(e, 120)}); this batch has no "
              f"restore point and will not be checkpointed until the next successful snapshot",
              file=sys.stderr, flush=True)
        return None


def _try_rng_restore(tc, rng, device, chi_gen) -> bool:
    """``rng_restore`` that reports failure instead of raising. Returns whether it restored.

    WHY THIS IS SAFE TO SKIP, which is not what an earlier version of this comment claimed.
    Restoring before a retry makes the re-run reproduce the batch exactly; NOT restoring makes it a
    different but equally valid iid draw -- the same licence _rows_with_oom_retry already takes when
    it re-draws SDE noise in smaller blocks. It does NOT desynchronise the checkpoint, because the
    checkpoint always stores the ACTUAL state at a batch boundary (the cadence write snapshots
    fresh, the rescue write uses that batch's own opening snapshot), and rows already on disk are
    never regenerated. Seed-level reproducibility is out of reach here anyway: the inits come from
    numpy's global RNG, which torch seeds do not touch.

    WHY IT MUST NOT RAISE. ``torch.cuda.set_rng_state_all`` copies each generator's state into
    device memory, so it is an ALLOCATION -- small, but an allocation. On 2026-08-28 the retrain
    died here at batch 3990/10000: the recovery block had just released every cached block and slept
    15 s on a contended card, so the few KB this needs was the one request the driver could not
    serve -- the second time a recovery step killed the run it was rescuing (the first was
    empty_cache itself, 2026-08-27).
    """
    if not rng:
        return False
    try:
        tc.rng_restore(rng, device, chi_gen)
        return True
    except Exception as e:                   # noqa: BLE001 -- see the docstring
        print(f"{_batch_tag()}: could not restore the RNG before re-running ({_short_err(e, 120)}); "
              f"the re-run proceeds with a fresh draw -- statistically equivalent, not bit-identical",
              file=sys.stderr, flush=True)
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


VRAM_CEILING_ENV = "PRISM_VRAM_CEILING_GIB"


def vram_ceiling_gib() -> float:
    """The hard per-batch VRAM ceiling in GiB: the env override if set, else the config constant.

    0 means off. See sim_memory_budget_elements for what it does and does not buy.
    """
    raw = os.environ.get(VRAM_CEILING_ENV)
    if raw is not None and raw.strip():
        try:
            return max(0.0, float(raw))
        except ValueError:
            print(f"{VRAM_CEILING_ENV}={raw!r} is not a number; ignoring it and using "
                  f"config.SIM_VRAM_CEILING_GIB instead", file=sys.stderr, flush=True)
    # Guarded for the same reason as the env branch: this runs inside the PLANNER, on every batch,
    # and a config.py edited to a non-numeric value would otherwise raise there rather than at the
    # point of the mistake.
    try:
        return max(0.0, float(getattr(config, "SIM_VRAM_CEILING_GIB", 0.0) or 0.0))
    except (TypeError, ValueError):
        print(f"config.SIM_VRAM_CEILING_GIB is not a number; treating the ceiling as off",
              file=sys.stderr, flush=True)
        return 0.0


def sim_memory_budget_elements(device: torch.device, dtype: torch.dtype) -> int:
    """The element budget `_max_sim_batch` actually plans against -- free-memory reading, the 0.85
    headroom fraction, and the LEARNED cap, all folded in.

    Public for the same reason as `peak_sim_elements`: a front-end showing "will this fit?" must
    compare against the planner's budget, not against a raw `mem_get_info` reading. ⚠ That reading
    still overstates free VRAM on Windows by roughly the size of the desktop (measured 15037 MiB
    against nvidia-smi's 5814), which is why the learned cap exists -- so treat anything derived from
    this as an UPPER bound on what is really available.

    THREE TERMS, AND EACH ANSWERS A DIFFERENT QUESTION. The reading says what the driver claims is
    free (optimistic). The LEARNED cap says what has actually failed (reactive -- it can only know
    after something died). The CEILING says what the OPERATOR knows in advance: on a day the desktop
    will be busy, it makes the planner split from batch 0 rather than discover the ceiling hours in.
    Read live off the module, never imported: ``from .config import NAME`` binds at import, so an
    importer would never see a per-run assignment to config.SIM_VRAM_CEILING_GIB.

    ⚠ THE CEILING NEEDS HEADROOM TO CAP TO, and it is not a substitute for freeing VRAM. On a card
    with 115 MiB actually free it changes nothing: _max_sim_batch finds that not even a floor-sized
    chunk fits alongside the result and runs the batch as asked. What it buys is stopping a run that
    HAS headroom from silently spilling into WDDM's shared-memory pool, where a batch does not fail
    -- it pages, at up to a 9x wall-clock penalty (measured 2026-08-27: 21.67 GiB completed on this
    15.92 GiB card). Free the memory first, then set this to about (nvidia-smi free) - 1 GiB.

    ``PRISM_VRAM_CEILING_GIB`` overrides the config constant, so a single run can be throttled
    without editing a tracked file -- the same shape as PRISM_CHI_OVERRIDE. An unparsable value is
    ignored with a note rather than crashing a multi-day run on a typo.
    """
    budget = min(config.memory_budget_elements(device, dtype, _SIM_MEM_FRACTION), _budget_cap())
    ceiling_gib = vram_ceiling_gib()
    if ceiling_gib > 0:
        bytes_per_elem = 4 if dtype == torch.float32 else 8
        budget = min(budget, int(ceiling_gib * 2 ** 30) // bytes_per_elem)
    return budget


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
    Simulate one batch of observations, splitting it whenever the geometry would not fit on device.

    :param model: built-in model name (one of VALID_SIMS) or a registered user model.
    :param params: (batch, n_params) simulation parameters.
    :param t: ND time grid; dtype/device are applied during processing.
    :param inits: (batch, n_vars) initial conditions.
    :param force: (batch, n_ch, T) drive tensor (or a broadcastable single-row drive).
    :param n_segs: time segments the simulator integrates over.
    :param steady_idx: transient cutoff; only points past it are returned.
    :param var_idx: if given, copy out ONLY this state variable, returning (1, batch, steady points)
        instead of (n_vars, ...). Pure memory: the solution buffer is n_vars deep and the copy has
        to coexist with it, so at the training batch size cloning all channels for a caller that
        only ever reads ``[0, :, :]`` doubles the peak of the largest allocation in the pipeline.
        The leading dim is kept so ``[0, :, :]`` indexes the same variable either way.
    :return: (n_vars, batch, steady points) observations -- (1, batch, ...) when ``var_idx`` is set.
    :raises ValueError: batch-size mismatch against params/inits, or an unknown model.
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
        note = _short_err(err)
        n_keep = (1 if var_idx is not None else inits.shape[-1]) * max(0, t.shape[0] - steady_idx)
        _budget_note_oom(batch_size * (inits.shape[-1] * t.shape[0]
                                       + (force.shape[1] if force.dim() > 2 else 1) * t.shape[0]
                                       + max(inits.shape[-1] * min(t.shape[0], CHUNK_LEN), n_keep)))

    # Outside the except: while `err` is bound its traceback pins the failed attempt's tensors
    # (the solver's whole (n_vars, batch, T) buffer among them), so releasing there frees nothing.
    # Hence `note` is a STRING.
    half = batch_size // 2
    # Notice BEFORE release: the release can itself raise on a starved card, outside the except
    # clause the exception context is already cleared, and on 2026-08-27 a run died exactly there
    # with `note` never seen. Printed on stderr, not warnings.warn -- the "once per location" filter
    # would collapse hundreds of events into one line, and parts of gen_training_data run under
    # simplefilter("ignore"); stderr also lands in the GUI log as a WARNING row.
    print(f"{_batch_tag()}: OOM at simulation batch {batch_size}; retrying in chunks of "
          f"{half}{_free_gib_note(device)}. Original: {note}", file=sys.stderr, flush=True)
    _release_device_memory(device)

    # Same preallocation as gen_obs' predictive split, for the same reason -- and note it happens
    # AFTER the release above, i.e. at the one moment in this function when the card is least
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
        note = _short_err(err)
        # Feed the learned budget even though this OOM may not have come from a simulator allocation.
        # It is still literally true that a block of this WIDTH at this GEOMETRY did not fit, and that
        # is the currency _max_sim_batch plans in -- so the next plan is wiser for the right reason.
        _budget_note_oom(n_rows * per_row_elements)

    # Outside the except: while `err` is bound its traceback pins the whole failed chi path --
    # several GiB at run_size=2048 -- so releasing there frees nothing; hence `note` is a STRING.
    # This is where the 2026-08-27 retrain died: empty_cache() raised its own AcceleratorError with
    # the context already cleared. The release is now best-effort and the notice prints BEFORE it,
    # so the original failure is on the record whatever the recovery manages to free.
    half = n_rows // 2
    print(f"{_batch_tag()}: OOM with {n_rows} rows OUTSIDE the simulator retry; re-running this batch "
          f"in halves of {half}{_free_gib_note(device)}. Original: {note}", file=sys.stderr, flush=True)
    _release_device_memory(device)

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


def retry_on_oom(fn, *, what: str, device: torch.device, attempts: int | None = None,
                 delays: tuple | None = None, extra_retryable: tuple = ()):
    """Run ``fn()``, waiting and retrying on a device out-of-memory. For NEW call sites.

    One implementation of the recovery protocol the three training ladders inline: notice BEFORE
    the release (the release can itself raise on a starved card, and once the except clause has
    closed the exception context is gone -- printing first puts the original on the record
    whatever the recovery manages), then release, one [mem] line, then the holder gate --
    waiting only helps when somebody ELSE holds the memory, so when this process is the holder
    the retry is immediate and the caller's own splitting is the remedy. The three existing
    ladders do NOT call this: their statement order inside gen_training_data's own source is
    pinned by tests, so the protocols are kept in step by hand instead.

    ``attempts``/``delays`` default to the live config retry knobs (read via getattr, so a per-run
    assignment to config is honoured). ``extra_retryable`` is a tuple of message substrings treated
    as retryable alongside a genuine OOM: the Fisher passes "cudaErrorUnknown", the error that
    actually killed a rotation on a starved card (2026-08-28) -- after one the context may be
    poisoned, in which case the attempts exhaust quickly and the caller degrades instead of dying
    inside its own recovery.

    Narrow on RuntimeError, like the ladders: streams.WorkerCancelled is a BaseException so a GUI
    cancel sails straight through, and a non-OOM RuntimeError re-raises immediately, traceback
    intact.
    """
    n_attempts = (int(getattr(config, "TRAINING_BATCH_RETRY_ATTEMPTS", 0) or 0)
                  if attempts is None else int(attempts))
    delay_seq = (tuple(getattr(config, "TRAINING_BATCH_RETRY_DELAYS_S", ()) or (60.0,))
                 if delays is None else tuple(delays))
    for attempt in range(n_attempts + 1):
        try:
            return fn()
        except RuntimeError as err:
            retryable = _is_oom(err) or any(s in str(err) for s in extra_retryable)
            if attempt >= n_attempts or not retryable:
                raise
            note = _short_err(err)
        # Outside the except: while `err` is bound its traceback pins the failed attempt's tensors.
        delay = delay_seq[min(attempt, len(delay_seq) - 1)]
        print(f"{_batch_tag()}: {what} hit a device error ({attempt + 1}/{n_attempts + 1})"
              f"{_free_gib_note(device)}. Original: {note}", file=sys.stderr, flush=True)
        _release_device_memory(device)
        _log_memory(device, f"after OOM in {what}")
        if _we_are_the_holder(device):
            print(f"{_batch_tag()}: THIS process holds most of the card, so waiting cannot free "
                  f"anything -- retrying {what} immediately.", file=sys.stderr, flush=True)
        else:
            _cancellable_wait(delay, f"waiting for device memory before retrying {what}")


@dataclasses.dataclass(frozen=True)
class _BatchGeometry:
    """One training batch's shared state, captured once so the three mode branches can be plain
    module-level functions instead of arms of a closure reading ~30 enclosing names (a thirty-
    argument function is a standing invitation to a call-site drift bug -- a wrong
    subsample_factor produces a PLAUSIBLE training row, not a crash)."""
    model: str
    t_fine: torch.Tensor
    n_segs_k: int
    steady_idx: int
    subsample_factor: int
    N_points_k: int
    T_k: float
    n_force_ch: int
    dt_exp: float
    forcing_idx: dict
    sim_ridx: dict
    fixed_dict: dict
    state_dep_drift: bool
    dtype: torch.dtype
    device: torch.device
    # chi mode only; None elsewhere
    chi_k_pad: int = None
    chi_f0: float = None
    chi_freq_bounds: tuple = None
    chi_max_cycles: float = None
    chi_k_fixed: int = None
    chi_gen: object = None
    b_mults: torch.Tensor = None
    dfrac: torch.Tensor = None


def _chi_rows(g: _BatchGeometry, nd, resc, init_rows, x_scale, x_offset, _patho) -> torch.Tensor:
    """One chi-mode row block: spontaneous run + K single-tone probes -> [S | log T | chi block].
    gen_obs / gen_stats / gen_chi_block / _subset_probe_rows are read as MODULE names on purpose --
    the test harness patches them on this module and must be honoured."""
    (model, t_fine, n_segs_k, steady_idx, subsample_factor, N_points_k, T_k, n_force_ch,
     dt_exp, fixed_dict, state_dep_drift, sim_ridx, forcing_idx, dtype, device) = (
        g.model, g.t_fine, g.n_segs_k, g.steady_idx, g.subsample_factor, g.N_points_k, g.T_k,
        g.n_force_ch, g.dt_exp, g.fixed_dict, g.state_dep_drift, g.sim_ridx, g.forcing_idx,
        g.dtype, g.device)
    n = nd.shape[0]
    chi_k_pad, chi_f0, chi_freq_bounds = g.chi_k_pad, g.chi_f0, g.chi_freq_bounds
    chi_max_cycles, chi_k_fixed, chi_gen = g.chi_max_cycles, g.chi_k_fixed, g.chi_gen
    b_mults, dfrac = g.b_mults, g.dfrac
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
    count_pathological(x_spont_dim, _patho)
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
        model, nd, resc, x_spont_dim, t_fine, init_rows, sim_ridx,
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
    return statistics.conditioning_rows(training_stats, T_k, chi_block.cpu())



def _spontaneous_rows(g: _BatchGeometry, nd, init_rows, x_scale, x_offset, _patho) -> torch.Tensor:
    """One spontaneous-mode row block: a single unforced run -> [S | log T]."""
    (model, t_fine, n_segs_k, steady_idx, subsample_factor, N_points_k, T_k, n_force_ch,
     dt_exp, fixed_dict, state_dep_drift, sim_ridx, forcing_idx, dtype, device) = (
        g.model, g.t_fine, g.n_segs_k, g.steady_idx, g.subsample_factor, g.N_points_k, g.T_k,
        g.n_force_ch, g.dt_exp, g.fixed_dict, g.state_dep_drift, g.sim_ridx, g.forcing_idx,
        g.dtype, g.device)
    n = nd.shape[0]
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
    count_pathological(x_spont_dim, _patho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        training_stats = gen_stats(x_spont_dim.cpu(), None, dt_exp, None, None, None,
                                   device=device, spontaneous_only=True)
        return statistics.conditioning_rows(training_stats, T_k)



def _forced_rows(g: _BatchGeometry, nd, resc, init_rows, fparams, x_scale, x_offset, _patho) -> torch.Tensor:
    """One forced-mode row block: driven + spontaneous runs -> [S | log T | theta_force]."""
    (model, t_fine, n_segs_k, steady_idx, subsample_factor, N_points_k, T_k, n_force_ch,
     dt_exp, fixed_dict, state_dep_drift, sim_ridx, forcing_idx, dtype, device) = (
        g.model, g.t_fine, g.n_segs_k, g.steady_idx, g.subsample_factor, g.N_points_k, g.T_k,
        g.n_force_ch, g.dt_exp, g.fixed_dict, g.state_dep_drift, g.sim_ridx, g.forcing_idx,
        g.dtype, g.device)
    n = nd.shape[0]
    # 2. Build nondimensional force tensor at fine resolution (uses PHYSICAL rescale)
    force = build_nondim_sin_force_tensor(
        fparams, t_fine, resc, forcing_idx, sim_ridx
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
    count_pathological(x_spont_dim, _patho)
    count_pathological(x_dim, _patho)

    # 5. Stats (A-F from spontaneous, G from forced) + conditioning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drive_amp = fparams[:, forcing_idx["amp"]].cpu()
        drive_freq = fparams[:, forcing_idx["freq"]].cpu()
        drive_phase = fparams[:, forcing_idx["phase"]].cpu()
        training_stats = gen_stats(x_spont_dim.cpu(), x_dim.cpu(), dt_exp, drive_amp, drive_freq, drive_phase, device=device)
        return statistics.conditioning_rows(training_stats, T_k, fparams.cpu())



def _training_inits(model: str, run_size: int, n_vars, dtype, device) -> torch.Tensor:
    """Initial conditions for a training run: the model's declared inits (user models) or the
    randint-pos/zero-prob synthesis (built-ins). Drawn from NUMPY's global RNG -- which torch seeds
    do not touch -- so callers claiming seeded reproducibility must seed numpy too.

    ``n_vars`` is a consistency CHECK, not an input: the real count comes from the inits, and a
    caller whose state width disagrees with the model's declared inits has a real bug (a stale cell
    file, a user model edited since the config was built) that would otherwise surface much later
    as a shape error inside the solver.
    """
    from core import registry
    is_user = registry.is_user_model(model)
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

    if n_vars is not None and int(n_vars) != inits.shape[-1]:
        raise ValueError(
            f"n_vars={n_vars} disagrees with the model's initial conditions, which are "
            f"{inits.shape[-1]}-wide for '{model}'. One of the two is stale.")
    return inits


def _batch_schedule(n_runs: int, t: torch.Tensor, t_scale_bounds, t_min_exp, t_max_exp,
                    dt_exp, dt_nd_min, steady_idx) -> tuple[torch.Tensor, torch.Tensor]:
    """The run's stratified (t_scale, T) schedule: Sobol pairs, log-spaced in both axes, pre-filtered
    by the fine-grid ceiling. Consumes the torch global RNG (SobolEngine(scramble=True) draws at
    construction), so a resume must reuse the stored schedule rather than call this again.
    """
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
    return batch_t_scales, batch_Ts


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
                      nd_idx: dict | None = None, k_b_cell: float | None = None,
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
                                   recompute it (V is NOT reproducible across processes: its
                                   operating points come from the caller's unseeded global RNG)
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
                           cycles at all, so a long recording walks its probes past the ~31-cycle
                           reproducibility wall (see config.CHI_MAX_CYCLES for the measurement).
    :param nd_idx: ND parameter name -> column. Required only when the box declares ``T`` instead
                   of ``f_scale`` (tier-1 physical consistency, core/SBI/derived.py); ignored
                   otherwise, so every pre-tier-1 caller is unaffected.
    :param k_b_cell: Boltzmann's constant in cell units (``SimConfig.k_b_cell``). Same condition.
    :param dtype: Tensor data type. Defaults to torch.float32.
    :param device: Computation device. Defaults to CPU.
    :return: Tuple of (training_data, thetas) where training_data has shape
             (n_runs * run_size, n_stats + n_forcing + 1) and thetas has shape
             (n_runs * run_size, nd_dim + rescale_dim).
    """
    from core import registry
    is_user = registry.is_user_model(model)
    # The index the SIMULATOR reads. Identical to rescale_idx unless this box declares T instead
    # of f_scale, in which case T's column is renamed -- see core/SBI/derived.py.
    sim_ridx = derived.sim_rescale_idx(rescale_idx)
    if model.lower() not in VALID_SIMS and not is_user:
        raise ValueError(f"Invalid simulator: {model}")

    inits = _training_inits(model, run_size, n_vars, dtype, device)

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
        # A DEDICATED generator for the probe draw. Never the global RNG: common-random-number
        # schemes (the Fisher, degeneracy_map) surround the chi block with deliberate manual_seed()
        # calls, and a placement drawn from the global stream would be re-randomised -- or worse,
        # frozen -- by them.
        chi_gen = torch.Generator(device="cpu")
        chi_gen.manual_seed(20260805)

    # --- Checkpointing (C-11): decide RESUME before anything expensive ---------------------------
    # Resolved here, above the Sobol schedule, because a resume must take that schedule from the
    # checkpoint rather than rebuild it: SobolEngine(scramble=True) consumes the torch global RNG at
    # CONSTRUCTION and _draw_and_filter's accept count depends on the geometry, so it cannot be
    # re-derived from a seed. Rebuilding it would silently re-stratify the second half of the run.
    # Imported unconditionally: the checkpoint needs it, and so does the batch-level OOM retry
    # below, which snapshots the RNG whether or not anything is being written to disk.
    from core.SBI import training_checkpoint as _tc

    _ck_dir = _ck_every = _ck_resumed = None
    _start_k = 0
    if checkpoint is not None:
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
            # inits especially: it is drawn from NUMPY's RNG (which torch seeds do not touch), and
            # nothing here restores that, so a redraw would quietly change the inits mid-run.
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

    # SKIPPED ENTIRELY on a resume: batch_t_scales/batch_Ts already came from the checkpoint header.
    # Not merely redundant -- rebuilding the engine would consume the torch global RNG (scramble=True
    # draws at construction) between here and the RNG restore, and re-deriving a schedule that the
    # accept/reject filter makes geometry-dependent is precisely the "silently non-uniform
    # stratification" C-11 warns is worse than crashing.
    if _ck_resumed is None:
        batch_t_scales, batch_Ts = _batch_schedule(
            n_runs, t, t_scale_bounds, t_min_exp, t_max_exp, dt_exp, dt_nd_min, steady_idx)

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
            _w = statistics.SUMMARY_WIDTH + 1 + (
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
    _patho = dict.fromkeys(("rows", "nonfinite", "constant", "overflow"), 0)
    _patho_seen = 0              # count already reported, so each line is NEW rows
    _pending_rng = None          # RNG as of the TOP of batch_k -- see the checkpoint write below
    _pending_rng_at = -1         # ...and WHICH batch it describes; see the rescue write
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
                # PAIRED WITH ITS BATCH INDEX, and set together. `batch_k` is bound by the `for`
                # before this runs, so a snapshot that FAILS here would otherwise leave the previous
                # batch's state sitting in `_pending_rng` while the rescue write records `batch_k` --
                # committing rows [.., k) with a restore point describing the top of k-1, so a resume
                # would restart batch k with the streams where k-1 began. Recording nothing is
                # correct; recording the wrong thing is not.
                _pending_rng = _try_rng_snapshot(_tc, device, chi_gen)
                _pending_rng_at = batch_k
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
            # NOTE on the "non-finite training targets" concern: on torch 2.9 this
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

            # TIER 1 (a box that declares T instead of f_scale): the SIMULATOR runs at the DERIVED
            # f_scale; the training
            # TARGET keeps T. Applied after the latent is taken, and that order is the whole
            # point -- T is the inferred parameter, so the target must record T while the
            # simulation runs at the force scale T implies. Deriving first would train the flow
            # to predict a quantity nobody asked about. A no-op (the same object, not a copy)
            # for a box that declares f_scale, so nothing pre-tier-1 changes.
            sim_thetas_rescale = derived.to_sim_rescale(
                curr_thetas_nd, curr_thetas_rescale, rescale_idx, nd_idx, k_b_cell)

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

            geom = _BatchGeometry(
                model=model, t_fine=t_fine, n_segs_k=n_segs_k, steady_idx=steady_idx,
                subsample_factor=subsample_factor, N_points_k=N_points_k, T_k=T_k,
                n_force_ch=n_force_ch, dt_exp=dt_exp, forcing_idx=forcing_idx, sim_ridx=sim_ridx,
                fixed_dict=fixed_dict, state_dep_drift=state_dep_drift, dtype=dtype, device=device,
                **(dict(chi_k_pad=chi_k_pad, chi_f0=chi_f0, chi_freq_bounds=chi_freq_bounds,
                        chi_max_cycles=chi_max_cycles, chi_k_fixed=chi_k_fixed, chi_gen=chi_gen,
                        b_mults=b_mults, dfrac=dfrac) if chi_mode else {}))

            def _rows(lo: int, hi: int):
                """Slice this batch's rows [lo, hi) and dispatch to the mode branch.

                Still a CLOSURE, deliberately: the batch state is captured once, here, into a
                _BatchGeometry, and the retry seam _rows_with_oom_retry(_rows, ...) stays
                byte-identical. The logic worth testing -- the halving, the floor, the cancel
                pass-through -- lives in _rows_with_oom_retry, which takes a fake fn.
                """
                resc = sim_thetas_rescale[lo:hi]
                nd = curr_thetas_nd[lo:hi]
                init_rows = inits[lo:hi]
                # None in the chi and spontaneous branches, where there is no drive to slice.
                fparams = None if curr_thetas_forcing is None else curr_thetas_forcing[lo:hi]
                # Derived HERE rather than at batch level so the "this model has no x_offset" case
                # needs no branch: slicing the SOURCE works for both, slicing the float 0.0 does not.
                x_scale  = resc[:, sim_ridx["x_scale"]].unsqueeze(1)
                x_offset = (resc[:, sim_ridx["x_offset"]].unsqueeze(1)
                            if "x_offset" in sim_ridx else 0.0)
                if chi_mode:
                    return _chi_rows(geom, nd, resc, init_rows, x_scale, x_offset, _patho)
                elif spontaneous_only:
                    return _spontaneous_rows(geom, nd, init_rows, x_scale, x_offset, _patho)
                return _forced_rows(geom, nd, resc, init_rows, fparams, x_scale, x_offset, _patho)

            # Per-ROW element cost at THIS geometry, in the same currency _max_sim_batch plans in, so
            # an OOM here tightens the SAME learned cap the predictive guard reads rather than
            # inventing a second, incompatible notion of "too big".
            #
            # THE DRIVE IS CHARGED AT ITS BUILD PEAK, NOT AT ITS RESULT SIZE. This term used to be a
            # bare `n_force_ch * n_fine_total`, i.e. the tensor the builder returns -- but building it
            # costs _FORCE_BUILD_PEAK_MULTIPLE times that, and in chi mode the builder runs again for
            # every probe, inside the K loop, alongside x_spont_dim and idx_c. That 4x under-count is
            # a plausible source of the historical UNWRAPPED OOMs (a bare CUDA OOM outside
            # SimulationError means the batch's own tensors, not the solver), and this is the place
            # it actually bites: `_per_row` is what _budget_note_oom charges the learned cap
            # with, so under-counting here taught the cap a number smaller than what really failed.
            _seg = min(n_fine_total, CHUNK_LEN)
            _n_keep = max(0, n_fine_total - steady_idx)              # var_idx=0 on every path here
            _per_row = (inits.shape[-1] * n_fine_total
                        + _FORCE_BUILD_PEAK_MULTIPLE * n_force_ch * n_fine_total
                        + max(inits.shape[-1] * _seg, _n_keep))
            # THE OUTERMOST OF THREE RETRIES, AND THE ONLY ONE THAT DOES NOT SHRINK THE WORK.
            #
            # _gen_obs_retry halves the simulator batch; _rows_with_oom_retry halves this batch's
            # rows. Both answer "this is too big". They cannot answer the other failure mode, which
            # is what actually kills runs on a desktop card: the batch is a perfectly reasonable
            # size and the card is momentarily full of somebody ELSE's surfaces -- a browser opening
            # a video, a game launcher waking up, the compositor after an unlock. Under WDDM those
            # are evictable, so mem_get_info reported them to us as free and the driver then lost
            # the eviction race. Shrinking does not help; WAITING does.
            #
            # THE RE-RUN REPRODUCES THE BATCH, which is worth having but is NOT a correctness gate
            # -- an earlier version of this comment claimed it was, and that was wrong. Restoring the
            # RNG makes the re-run consume randomness identically, so it produces the batch that
            # would have been produced. Skipping it makes the re-run a DIFFERENT but equally valid
            # iid draw, which is the same licence _rows_with_oom_retry already takes when it re-draws
            # SDE noise in smaller blocks. It does not desynchronise the checkpoint: the checkpoint
            # records the ACTUAL state at each batch boundary, and rows already on disk are never
            # regenerated. So this is best-effort -- see _try_rng_restore.
            #
            # ⚠ IT IS *THIS* SNAPSHOT, NOT `_pending_rng`. `_pending_rng` is the state at the TOP of
            # the iteration -- before the theta draw, the chi multipliers and the durations -- which
            # is what a RESUME needs, because a resume re-runs all of those. The retry does not: the
            # thetas are already drawn and are reused as they stand, so restoring `_pending_rng` here
            # would rewind past draws the retry never repeats and feed `_rows` the noise the THETA
            # draw should have consumed. Snapshot immediately before the call instead.
            #
            # A few KB of memcpy against a ~20 s batch, so it is taken unconditionally rather than
            # only when a retry is configured -- one code path, and no way for the two to disagree.
            _rows_rng = _try_rng_snapshot(_tc, device, chi_gen)
            _attempts = int(getattr(config, "TRAINING_BATCH_RETRY_ATTEMPTS", 0) or 0)
            _delays = tuple(getattr(config, "TRAINING_BATCH_RETRY_DELAYS_S", ()) or (60.0,))
            for _attempt in range(_attempts + 1):
                try:
                    _rows_out = _rows_with_oom_retry(
                        _rows, 0, run_size, per_row_elements=_per_row, device=device)
                    break
                # RuntimeError, NOT Exception -- the same narrowing the two inner ladders use, and
                # for the same reason: streams.WorkerCancelled is a BaseException so a GUI cancel
                # sails through, and a non-OOM RuntimeError is a real bug that must not be retried
                # into a loop.
                except RuntimeError as _err:
                    if _attempt >= _attempts or not _is_oom(_err):
                        raise
                    _note = _short_err(_err)
                    _budget_note_oom(run_size * _per_row)
                # OUTSIDE THE HANDLER, for the reason spelled out in _rows_with_oom_retry: while
                # `_err` is bound its traceback owns every frame of the failed attempt and the
                # tensors they hold, so releasing here is what makes the release mean anything.
                _delay = _delays[min(_attempt, len(_delays) - 1)]
                print(f"{_batch_tag()}: batch FAILED after both halving retries "
                      f"({_attempt + 1}/{_attempts + 1}){_free_gib_note(device)}. Waiting "
                      f"{_delay:.0f}s and re-running the whole batch. Original: {_note}",
                      file=sys.stderr, flush=True)
                # ⚠ RESTORE FIRST, RELEASE SECOND, WAIT LAST. The order is the fix, not decoration.
                #
                # This block used to run release -> wait -> restore, and on 2026-08-28 that killed
                # the run at batch 3990/10000. The release hands every cached block back to the
                # driver; we then sleep for up to three minutes on a CONTENDED card, during which the
                # desktop takes the memory; and the restore -- which needs only a few KB, but needs
                # them on the device -- is left asking for a fresh cudaMalloc at the exact moment we
                # have the weakest claim on the card. We donated our working set and then asked for
                # it back.
                #
                # Restoring first inverts that. The failed attempt's tensors were dropped when the
                # except clause closed, so the allocator is holding them as CACHED BLOCKS and a
                # KB-sized request is served without touching the driver at all. Only then is it
                # worth handing that cache back, and only then worth sleeping.
                _try_rng_restore(_tc, _rows_rng, device, chi_gen)
                _release_device_memory(device)
                # ONE [mem] LINE PER OOM, unconditionally. The cadence line is every
                # _MEM_LOG_EVERY batches, which is far too coarse to diagnose a run that dies at
                # batch 93 -- and peak RESERVED against peak ALLOCATED is the number that separates
                # "this geometry is too big" from "the allocator is fragmented and cannot hand the
                # memory back". It is printed AFTER the release, so it describes what we could not
                # give up.
                _log_memory(device, f"after OOM on {_batch_tag()}")
                if _we_are_the_holder(device):
                    # Waiting cannot help: we are what is full. Go straight back to the retry, where
                    # the two halving ladders will shrink the work instead.
                    print(f"{_batch_tag()}: THIS process holds most of the card, so waiting cannot "
                          f"free anything -- retrying immediately at a smaller size instead of "
                          f"pausing {_delay:.0f}s. If this repeats, the allocator is fragmented: "
                          f"restart the run (it resumes from its checkpoint) and consider setting "
                          f"the VRAM ceiling on the Config tab.", file=sys.stderr, flush=True)
                else:
                    _cancellable_wait(_delay, "waiting for device memory before re-running this batch")

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
            # plans ON: cuFFT caches a plan per distinct transform SHAPE, outside PyTorch's
            # caching allocator -- so empty_cache() cannot touch it and it surfaces as a RAW driver
            # cudaErrorMemoryAllocation rather than torch.cuda.OutOfMemoryError. N_points_k changes
            # every batch, so cross-batch plan reuse is exactly zero while each batch mints ~7 new
            # signatures (6 from SummaryStatistics, 1 from chi.peak_freq) at ~2 MB apiece; the
            # default 4096-entry cache would saturate around batch ~585 of 5000 and hold ~8.6 GB
            # hostage. Clearing per batch costs nothing (the intra-batch reuse across gen_stats'
            # sub-batches already happened) and is preferable to shrinking
            # cufft_plan_cache.max_size, which WOULD thrash it.
            #
            # graphs OFF: the batch SUCCEEDED. Dropping the captured graphs here would recapture on
            # every batch of a 5000-batch run and give back the solver's ~8x replay speedup to
            # reclaim a few MiB we are not short of. Graph pools are dropped only on the OOM paths.
            _release_device_memory(device, graphs=False)
            # THE one recovery credit for this batch. gen_obs' own calls are suppressed while
            # _BATCH_TAG is set precisely so that a chi batch's 1 + K of them cannot count as
            # 1 + K clean batches -- see _budget_note_ok.
            _budget_note_ok(batch_level=True)
            # Batch 0 gives an immediate baseline (so a run that is doomed says so in the first
            # minute rather than the fifth hour); after that, one line per _MEM_LOG_EVERY batches.
            if batch_k == 0 or (batch_k + 1) % _MEM_LOG_EVERY == 0:
                _log_memory(device, _BATCH_TAG)
            # Pathological trajectories, reported the batch they appear in rather than only as a
            # run total: they cluster in particular (t_scale, T) strata, so WHICH batch is the
            # diagnostic. Silent while there are none, which is the normal case.
            _bad = _patho["nonfinite"] + _patho["constant"] + _patho["overflow"]
            if _bad > _patho_seen:
                print(f"[patho] batch {batch_k}: {_bad - _patho_seen} new pathological "
                      f"trajectorie(s) -- {_patho['nonfinite']} non-finite, "
                      f"{_patho['constant']} exactly constant, {_patho['overflow']} over "
                      f"{_PATHO_MAG:g} in magnitude, of {_patho['rows']:,} simulated", flush=True)
                _patho_seen = _bad

            # --- checkpoint the completed batches [_ck_from, batch_k + 1) ---
            # _pending_rng is this batch's OPENING state, so the write records batches [0, k+1) with
            # the state batch k+1 will start from -- which is the snapshot the NEXT iteration takes.
            # Hence the write below uses the snapshot taken at the top of the following iteration; we
            # take a fresh one here for exactly that reason.
            if _ck_dir is not None and _ck_every and (batch_k + 1) % _ck_every == 0:
                # The snapshot is a driver call and can fail on a starved card. DEFER rather than
                # die: the rows are already in x_buf, _ck_from is not advanced, and the next cadence
                # boundary writes the whole span. The only cost is a longer crash window.
                _ck_rng = _try_rng_snapshot(_tc, device, chi_gen)
                if _ck_rng is None:
                    print(f"[checkpoint] deferring the write at batch {batch_k + 1}: no RNG "
                          f"snapshot. The rows are still held and go out at the next boundary.",
                          file=sys.stderr, flush=True)
                else:
                    _tc.save(_ck_dir, from_batch=_ck_from, batch_k=batch_k + 1, rng=_ck_rng,
                             x_buf=x_buf, th_buf=th_buf, run_size=run_size)
                    _ck_from = batch_k + 1

    except BaseException:
        # WorkerCancelled (a BaseException by design, so `except Exception` would miss it) and
        # KeyboardInterrupt both land here, and a GUI cancel is the MOST likely way a multi-day run
        # ends -- MainWindow.closeEvent reaches request_cancel_all(), so closing the window stops it.
        # Before this, that discarded every completed batch.
        # THE ROWS MATTER MORE THAN THE RESTORE POINT. If the snapshot for THIS batch failed or
        # belongs to another one, commit the completed batches with no RNG rather than skipping the
        # write: a checkpoint that resumes without restoring the streams draws fresh noise from
        # batch_k onward -- statistically equivalent, the same licence the OOM ladders take -- while
        # a skipped write throws away hours of simulation outright.
        if _ck_dir is not None and batch_k > _ck_from:
            _rescue_rng = _pending_rng if _pending_rng_at == batch_k else None
            if _rescue_rng is None:
                print(f"[checkpoint] no valid RNG snapshot for batch {batch_k}; saving the rows "
                      f"without a restore point (a resume will draw fresh noise from there)",
                      file=sys.stderr, flush=True)
            try:
                # Announced BEFORE the write, so a multi-second flush is not an unexplained hang after
                # Cancel. Safe to print here even under a cancel: CancelToken.fired is a one-shot
                # latch, so the raise has already happened and later writes pass through. Nothing is
                # printed BETWEEN the shard fsync and the state replace -- see training_checkpoint.
                print(f"[checkpoint] stopping: saving {batch_k - _ck_from} completed batches "
                      f"({_ck_from} -> {batch_k}) before unwinding…", flush=True)
                _tc.save(_ck_dir, from_batch=_ck_from, batch_k=batch_k,
                         rng=_rescue_rng, x_buf=x_buf, th_buf=th_buf, run_size=run_size)
            except Exception as _e:              # noqa: BLE001
                # A failed rescue write must never REPLACE the cancel/crash with an I/O error.
                print(f"[checkpoint] could not save on the way out: {_e}", file=sys.stderr, flush=True)
        raise                                    # UNCONDITIONAL: never swallow a cancel
    finally:
        # Cleared however we leave -- return, OOM, or a cooperative cancel. A stale tag would make the
        # NEXT failure anywhere in the process claim a batch that finished hours ago, which is worse
        # than no tag at all.
        _BATCH_TAG = ""

    if _patho["rows"]:
        _bad = _patho["nonfinite"] + _patho["constant"] + _patho["overflow"]
        print(f"[patho] run total: {_bad:,} pathological of {_patho['rows']:,} simulated "
              f"trajectories ({100.0 * _bad / _patho['rows']:.4f}%) -- "
              f"{_patho['nonfinite']:,} non-finite, {_patho['constant']:,} exactly constant, "
              f"{_patho['overflow']:,} over {_PATHO_MAG:g}", flush=True)
    if x_buf is None:                       # n_runs == 0: nothing was generated, and nothing to size from
        return torch.empty((0, 0)), torch.empty((0, 0))
    if _ck_dir is not None:
        # Commit whatever the last cadence boundary left, then mark it done. A COMPLETE checkpoint is
        # deliberately not deleted: it is a several-GiB cache of a multi-day simulation, and it is what
        # lets the flow be retrained (different capacity, learning rate, epochs) without re-simulating.
        if n_runs > _ck_from:
            # ⚠ THIS IS AFTER THE `finally`, i.e. OUTSIDE the try -- nothing rescues a failure here,
            # and what would be lost is the whole run's product minus the last cadence boundary. The
            # snapshot is therefore best-effort: a COMPLETE checkpoint short-circuits generation and
            # returns its stored rows, so its restore point is never read and None costs nothing.
            _tc.save(_ck_dir, from_batch=_ck_from, batch_k=n_runs,
                     rng=_try_rng_snapshot(_tc, device, chi_gen),
                     x_buf=x_buf, th_buf=th_buf, run_size=run_size)
        _tc.mark_complete(_ck_dir, n_runs)
        print(f"[checkpoint] complete: {n_runs} batches in {_ck_dir}. Safe to delete once the "
              f"posterior is saved; keeping it lets you retrain the flow without re-simulating.",
              flush=True)
    return x_buf, th_buf

# Extracted seams, re-imported so every existing consumer -- orchestrator, the scripts, the
# test suites -- keeps reaching them as pipeline.* attributes, and so monkeypatching
# pipeline.<name> still lands on the object read at call time. Bottom of the file on purpose:
# the extracted modules call back into this one through the module object, which is fully
# populated by this line.
from core.SBI.train import (train_nn, TrainingPlan, _capped_zscore_check,  # noqa: E402
                            _ZSCORE_CHECK_MAX_ROWS)
from core.SBI.prior_screen import gen_prior, VALID_PRIORS  # noqa: E402
from core.forcing import build_nondim_sin_force_tensor  # noqa: E402
from core.SBI.chi_probes import gen_chi_raw, gen_chi_block, _subset_probe_rows  # noqa: E402
from core.SBI.summaries import (gen_stats, gen_stats_features, winsorize_summary_block,  # noqa: E402
                                count_pathological, _PATHO_MAG)
