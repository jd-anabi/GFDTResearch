"""Summary statistics glue, split out of pipeline.py (which stays the public facade).

Consumers reach everything here as ``pipeline.gen_stats`` / ``gen_stats_features`` /
``winsorize_summary_block`` / ``count_pathological`` -- pipeline.py re-imports the names at its
bottom, which keeps monkeypatching ``pipeline.<name>`` effective (the kill-at test harness rebinds
``pipeline.gen_stats`` and gen_training_data's ``_rows`` reads it by bare name from that
namespace). Calls back into pipeline machinery go through the module object (``_pipeline.``) at
call time for the same reason.
"""
import warnings

import torch

from core import config
from core.SBI import statistics
from core.SBI import pipeline as _pipeline


_PATHO_MAG = 1e15          # |x| beyond which a trajectory's features cannot be trusted


def count_pathological(x: torch.Tensor, acc: dict) -> None:
    """Tally trajectories that are non-finite, exactly constant, or of overflow magnitude.

    WHY THIS EXISTS. One population of pathological simulations was silently damaging three
    unrelated things at once, and NOTHING in the pipeline counted them:

      * an EXACTLY CONSTANT trace makes `_group_d`'s `std.clamp(1e-12)` fire, so `z == 0`,
        `kurt == 0`, and D3_bimodality comes back as exactly 1/1e-12 -- a flatlined simulation, not
        an underflow;
      * a ~1e29-magnitude trace drags A1_mean's fitted std to 4.19e11, which is what made the
        channel invisible to the flow;
      * divergent draws erased the posterior-predictive PSD band entirely (torch.quantile
        propagates a single non-finite entry across the whole reduction, so one bad draw in a
        thousand blanked the figure).

    Three cheap reductions over tensors that are already resident, so this is the cheapest item in
    the conditioning-repair programme and should have existed from the start.

    Counted per SIMULATED trajectory, so an OOM retry that re-simulates a half-batch counts those
    rows twice -- the row denominator is accumulated the same way, so the FRACTION stays honest even
    though the absolute count can exceed the training set size on a heavily-retried run.
    """
    if x is None or x.numel() == 0:
        return
    acc["rows"] += int(x.shape[0])
    # TWO REDUCTIONS AND NOTHING ELSE. The obvious spelling -- isfinite(x).all(-1), nan_to_num(x),
    # safe.abs() -- allocates three tensors the size of the trajectory block, which at the production
    # shape is 2048 x 60000 float32 = 492 MB EACH, on a card that has already died of OOM twice
    # (traps X6/X7). amax/amin PROPAGATE NaN and +-inf, so the row-level (rows,) reductions below
    # answer all three questions exactly: verified equal to isfinite(x).all(-1) on NaN, +inf, -inf,
    # constant and ordinary rows.
    mx, mn = x.amax(dim=-1), x.amin(dim=-1)
    finite = torch.isfinite(mx) & torch.isfinite(mn)
    acc["nonfinite"] += int((~finite).sum())
    # A non-finite row is counted ONLY as non-finite -- otherwise an all-NaN batch reports a flatline
    # population that is not there. The two remaining categories DO overlap on purpose: an all-1e29
    # row is genuinely both constant and overflow.
    # amax == amin, not std == 0: a 1e29-magnitude row squares to inf inside a variance, so a
    # std-based test would answer the overflow case rather than the constant one.
    acc["constant"] += int(((mx == mn) & finite).sum())
    acc["overflow"] += int(((torch.maximum(mx.abs(), mn.abs()) > _PATHO_MAG) & finite).sum())


def gen_stats(x_spont: torch.Tensor, x_forced: torch.Tensor, dt: float | torch.Tensor,
              drive_amp, drive_freq, drive_phase,
              band_halfwidth: int = 2, bp_lo: float = 0.5, bp_hi: float = 1.5, slow_env_frac: float = 0.15,
              device: torch.device = torch.device('cpu'), stats_batch_size: int = 256,
              spontaneous_only: bool = False) -> torch.Tensor:
    """
    Compute the summary-statistics block, in sub-batches on ``device`` (GPU FFT throughput without
    OOM on large inputs; each sub-batch result moves to CPU immediately).

    :param x_spont: unforced (spontaneous) trajectories for Groups A-F, shape (B, n), on CPU.
    :param x_forced: forced (driven) trajectories for Group G, shape (B, n), on CPU.
    :param dt: time-step resolution in cell time units, scalar or (B,) tensor.
    :param drive_amp: per-sample drive amplitude (dimensional), scalar or (B,); likewise
        ``drive_freq`` / ``drive_phase``.
    :param band_halfwidth: spectral band half-width in FFT bins (B7 / E2 harmonic powers).
    :param bp_lo: envelope band-pass lower edge as a fraction of the centre frequency; ``bp_hi``
        the upper edge; ``slow_env_frac`` the slow-envelope low-pass cutoff as a fraction of f_peak.
    :param spontaneous_only: if True (a no-forcing model), skip the forced-response Group G and
        zero-pad it to the full feature width. ``x_forced``/``drive_*`` may then be None -- the
        spontaneous run is reused as the (unused) forced input.
    :return: ``[features | valid flags]``, shape (batch, statistics.SUMMARY_WIDTH) -- the
        len(FEATURE_LABELS) features followed by the binary companion channels saying which are
        real measurements rather than substituted sentinels (statistics.derive_valid_flags).
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
        # plans/graphs OFF: this loop's whole point is intra-batch cuFFT plan reuse across
        # sub-batches, and a graph recapture per sub-batch would cost far more than the segments
        # this hands back. Guarded all the same -- an empty_cache() that raises must not take the
        # run down (2026-08-27).
        _pipeline._release_device_memory(device, plans=False, graphs=False)
    feats = torch.cat(results, dim=0)
    # [features | valid flags]. Appended HERE rather than at each call site: every conditioning
    # vector in the project is assembled by statistics.conditioning_rows over this block (training,
    # calibration, PPC, the simulated and experimental observation paths). Widening the summary
    # block at its single source means every consumer stays in step by construction -- a flag set
    # that reached training but not the PPC would be invisible until the two disagreed about what
    # a row means.
    return torch.cat([feats, statistics.derive_valid_flags(feats, dt)], dim=-1)

def gen_stats_features(*args, **kwargs) -> torch.Tensor:
    """``gen_stats`` WITHOUT the trailing valid-flag block: the 41 features and nothing else.

    THE ONE NAME FOR "features, not conditioning", and every Jacobian in the project must use it.
    A diagnostic that standardises by a locally-measured `fnoise = max(std, 1e-9)` turns a BINARY
    channel into an amplifier the moment it steps between two operating points -- constant almost
    everywhere (harmless) and then 1/1e-9 at the one place it moves. The same defect class that
    removed `logcyc` from the Fisher channel set (C-9/C-10): decide deliberately, for every channel
    added to gen_stats, whether each caller is a conditioning vector (wants it) or a Jacobian
    (does not); the tell is whether the result is cat-ed with log_T.

    It is also simply the right width: these callers index their results against FEATURE_LABELS, and
    the flags are not features -- they are a statement about the OBSERVATION, carrying no gradient in
    theta at all.
    """
    return statistics.split_summary_block(_pipeline.gen_stats(*args, **kwargs))[0]



def winsorize_summary_block(data: torch.Tensor, n_summary: int,
                            pct: tuple[float, float] = None) -> torch.Tensor:
    """Clip each SUMMARY column to its own [lo, hi] percentile. Returns a new tensor.

    ⚠ THE CHI BLOCK IS NEVER TOUCHED, and that is a correctness requirement rather than a scoping
    convenience. A padded probe slot is exactly 0.0 in all six channels and is required to stay
    BITWISE inert (section 3.6, pinned by tests/test_chi_set_encoder.py). Under chi the mask column
    is ~0.28 zeros, so its 0.1th percentile is 0.0 and clipping would be a no-op there -- but
    `logmag`, `cos` and `sin` are dense over live probes, so THEIR 0.1th percentile is non-zero and
    clipping would push every pad off 0.0 and turn it into a phantom probe. That is the exact defect
    the packer's `nan_to_num` removal fixed once already.

    log(T) rides in the summary block and is clipped with it; that is harmless (it is bounded by
    t_min_exp/t_max_exp by construction, so its percentiles are inside its own range).
    """
    lo_p, hi_p = config.WINSOR_PCT if pct is None else pct
    s = data[:, :n_summary]
    q = torch.tensor([lo_p, hi_p], dtype=torch.float32)
    lo = torch.empty(n_summary, dtype=data.dtype)
    hi = torch.empty(n_summary, dtype=data.dtype)
    for c in range(n_summary):
        # Sort-based, for the same reason EmbeddedNet fits its knots that way: torch.quantile carries
        # a 2**24-element input ceiling that 10.24M rows sit just under today.
        col = s[:, c].to(torch.float32).sort().values
        n = col.numel()
        idx = (q * (n - 1)).round().long().clamp(0, max(n - 1, 0))
        lo[c], hi[c] = col[idx[0]].to(data.dtype), col[idx[1]].to(data.dtype)
    # IN PLACE, and at the production shape that is the difference between a 5 GiB step and a 12 GiB
    # one. `data` here is 10.24M x 122 float32 = 5.0 GiB; building a clipped COPY of the summary
    # block and then torch.cat-ing it back would hold the original, the copy and the concatenation
    # alive together -- reinstating exactly the host peak gen_training_data's preallocated
    # accumulators were introduced to remove, at exactly the same point in the run (right before
    # append_simulations, at the end of a multi-day generation). `data` is owned by train_nn and is
    # not aliased by the caller, so mutating it is safe; the same tensor is returned for readability.
    n_moved = int(((s < lo) | (s > hi)).sum())
    s.clamp_(min=lo, max=hi)
    if n_moved:
        print(f"[winsor] clipped {n_moved:,} of {s.numel():,} summary elements "
              f"({100.0 * n_moved / max(s.numel(), 1):.3f}%) to their per-column "
              f"{lo_p:.1%}/{hi_p:.1%} percentiles", flush=True)
    return data

