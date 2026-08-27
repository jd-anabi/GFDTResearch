"""Machinery for comparing posterior-predictive traces against the observation.

WHY THIS EXISTS. For spontaneous dynamics the oscillation PHASE is set by the noise realisation, not by
theta: simulating the exact ground truth still yields a different phase. So a posterior draw can never
line up with the observation pointwise, and pointwise-averaging draws is worse than useless (different
phases cancel, leaving a flat line at the mean level -- which is why the eye test simulates central
PARAMETERS rather than averaging trajectories). What genuinely can agree is frequency, amplitude, mean
and waveform shape. This module therefore provides:

  * explicit PHASE ALIGNMENT by FFT cross-correlation, so a draw can be laid over the observation;
  * TWO best-fit rankings, which answer different questions:
      - by SUMMARY STATISTICS: closest in the space the posterior actually conditioned on (the
        statistically principled pick), free because those statistics are already computed for the PPC;
      - by TRACE (after alignment): closest in waveform, i.e. the draw that literally looks most like
        the observation -- the right pick for an eye test. Aligning FIRST is essential; raw trace L2 is
        dominated by phase offset, which carries no parameter information here;
  * two PHASE-INVARIANT summaries -- a PSD band and a cycle-averaged waveform -- which compare the
    things that should agree without any alignment step to argue about.

Everything is batched torch and float64 internally (lock-in-style sums over long records).
"""
import math

import torch

_EPS = 1e-12


def roll_rows(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Per-row circular LEFT shift: ``out[i, j] = x[i, (j + shifts[i]) % n]``.

    torch.roll only takes a scalar shift, but each trace needs its own lag."""
    n = x.shape[-1]
    idx = (torch.arange(n, device=x.device).unsqueeze(0) + shifts.to(x.device).long().unsqueeze(1)) % n
    return torch.gather(x, 1, idx)


def align_lag(reference: torch.Tensor, traces: torch.Tensor) -> torch.Tensor:
    """Per-trace lag (in samples) that best aligns each trace to ``reference``.

    Circular cross-correlation via FFT, following the SummaryStatistics._acf pattern but with
    ``conj(A) * B`` instead of ``|A|^2``. Zero-padded to 2n so the correlation is linear rather than
    wrap-around, then lags above n are folded to negative shifts.

    :param reference: (n,) the observation.
    :param traces: (B, n) candidate traces.
    :return: (B,) integer lags, for use with ``roll_rows``.
    """
    n = traces.shape[-1]
    a = (reference - reference.mean()).to(torch.float64)
    b = (traces - traces.mean(dim=-1, keepdim=True)).to(torch.float64)
    L = 2 * n
    A = torch.fft.rfft(a, n=L)
    B = torch.fft.rfft(b, n=L)
    xc = torch.fft.irfft(A.conj().unsqueeze(0) * B, n=L)      # (B, 2n); xc[:, k] = sum_t a[t] b[t+k]
    lag = xc.argmax(dim=-1)
    return torch.where(lag > n, lag - L, lag)                 # fold to the shorter (signed) shift


def align_to(reference: torch.Tensor, traces: torch.Tensor) -> tuple:
    """Phase-align ``traces`` to ``reference``. Returns (aligned, lags)."""
    lags = align_lag(reference, traces)
    return roll_rows(traces, lags), lags


def rank_by_stats(sim_stats: torch.Tensor, obs_stats: torch.Tensor) -> tuple:
    """Rank posterior draws by standardized distance between their summary stats and the observation's.

    The statistically principled ranking: these are exactly the features the posterior conditioned on.
    Costs nothing -- ``sim_stats`` is already materialised for the posterior-predictive check.

    Constant columns are dropped (the conditioning block -- log T and the forcing/chi values -- is
    identical for every draw by construction, so it carries no discriminating information and would
    divide by ~0). Mirrors the zero-variance mask in analysis.posterior_predictive_check.

    :return: (order, distance) with ``order[0]`` the best-matching draw.
    """
    s = sim_stats.detach().to(torch.float64)
    o = obs_stats.detach().reshape(-1).to(torch.float64)
    sd = s.std(dim=0)
    keep = sd > 1e-10
    if not bool(keep.any()):
        d = torch.zeros(s.shape[0], dtype=torch.float64)
        return torch.arange(s.shape[0]), d
    z = (s[:, keep] - o[keep]) / sd[keep]
    d = z.norm(dim=1) / math.sqrt(int(keep.sum()))            # per-feature RMS z, comparable across runs
    return torch.argsort(d), d


def rank_by_trace(reference: torch.Tensor, traces: torch.Tensor) -> tuple:
    """Rank posterior draws by RMSE against the observation AFTER phase alignment.

    The visually optimal ranking -- it directly optimises "looks like the observation". Alignment first
    is not cosmetic: without it the score is dominated by an arbitrary phase offset that carries no
    parameter information for spontaneous dynamics.

    :return: (order, rmse, aligned) with ``order[0]`` the best-matching draw.
    """
    aligned, _lags = align_to(reference, traces)
    ref = reference.to(torch.float64).unsqueeze(0)
    rmse = ((aligned.to(torch.float64) - ref) ** 2).mean(dim=-1).sqrt()
    return torch.argsort(rmse), rmse, aligned


def psd(traces: torch.Tensor, dt: float, nperseg: int = None) -> tuple:
    """One-sided WELCH PSD of each trace (batched), Hann window, 50% overlap.

    Segment-averaged rather than a raw periodogram: a single-record periodogram has ~100% variance at
    every bin, which is far too jagged to compare against a band by eye (and the eye then reads noise
    spikes as structure). Averaging trades frequency resolution for a readable estimate -- the same
    trade core/FDT/spectral.psd_welch makes.

    :return: (freqs, power) with power shape (B, nfreq).
    """
    x = (traces - traces.mean(dim=-1, keepdim=True)).to(torch.float64)
    n = x.shape[-1]
    if nperseg is None:
        # Quarter-length segments (=> ~8 with 50% overlap): enough averaging to read, while keeping the
        # frequency resolution needed to separate the fundamental from DC on a short record.
        nperseg = max(64, min(n, 1 << int(math.log2(max(64, n // 4)))))
    nperseg = int(min(nperseg, n))
    step = max(1, nperseg // 2)
    starts = list(range(0, n - nperseg + 1, step)) or [0]
    window = torch.hann_window(nperseg, dtype=torch.float64, device=x.device)
    win_norm = (window ** 2).sum()
    acc = torch.zeros(x.shape[0], nperseg // 2 + 1, dtype=torch.float64, device=x.device)
    for s in starts:
        seg = x[:, s:s + nperseg]
        seg = (seg - seg.mean(dim=-1, keepdim=True)) * window
        acc += torch.fft.rfft(seg, dim=-1).abs() ** 2
    acc /= len(starts)
    power = 2.0 * acc * float(dt) / win_norm
    power[:, 0] /= 2                                          # DC is not doubled
    if nperseg % 2 == 0:
        power[:, -1] /= 2                                     # nor is Nyquist
    freqs = torch.fft.rfftfreq(nperseg, d=float(dt)).to(torch.float64)
    return freqs, power


# torch.quantile refuses an input above 2**24 elements. A posterior-predictive PSD is
# (draws, nfreq) and clears that at ~1000 draws x 16k bins, so the band this module exists to draw is
# exactly the shape that trips it -- and it raises rather than degrading, which costs the whole figure.
_QUANTILE_MAX_ELEMS = 1 << 24


def _column_quantile(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``torch.quantile(x, q, dim=0)``, chunked over columns to stay under its size limit."""
    n_rows = max(1, x.shape[0])
    max_cols = max(1, _QUANTILE_MAX_ELEMS // n_rows)
    if x.shape[1] <= max_cols:
        return torch.quantile(x, q, dim=0)
    return torch.cat([torch.quantile(x[:, i:i + max_cols], q, dim=0)
                      for i in range(0, x.shape[1], max_cols)], dim=1)


def psd_band(traces: torch.Tensor, dt: float, lo_pct: float = 5.0, hi_pct: float = 95.0) -> tuple:
    """Median + percentile band of the posterior-predictive PSD. Phase-invariant by construction.

    ⚠ NON-FINITE ROWS ARE DROPPED, AND THE COUNT IS RETURNED SO THE CALLER CAN SAY SO. This is not
    defensive coding for its own sake -- it is the fix for a figure that came back as a bare
    observation line with no band and no median, and said nothing about why. A broad posterior samples
    parameter sets that do not produce a stable oscillator; ``|rfft|**2`` of such a trace overflows to
    ``inf``, and ``torch.quantile`` propagates a single non-finite entry across the WHOLE column set,
    so one bad draw in a thousand silently erases the entire band. The two sibling figures survived it
    (the overlay band takes only the 50 best draws, and cycle_average confines the damage to one phase
    bin), which is precisely why the failure looked like a plotting bug rather than a data one.

    Reporting the count matters as much as dropping the rows: "the band is missing" is a bug report,
    "17 of 1000 draws were non-finite" is a finding about the posterior.

    :return: (freqs, lo, median, hi, n_dropped). lo/median/hi are all-NaN if nothing finite remains.
    """
    freqs, power = psd(traces, dt)
    ok = torch.isfinite(power).all(dim=-1)
    n_dropped = int((~ok).sum())
    if n_dropped:
        power = power[ok]
    if power.shape[0] == 0:
        nan = torch.full_like(freqs, float("nan"))
        return freqs, nan, nan, nan, n_dropped
    q = torch.tensor([lo_pct / 100.0, 0.5, hi_pct / 100.0], dtype=torch.float64)
    band = _column_quantile(power, q)
    return freqs, band[0], band[1], band[2], n_dropped


def _analytic_phase(x: torch.Tensor, dt: float, f_center: float, bp_lo: float = 0.5,
                    bp_hi: float = 1.5) -> torch.Tensor:
    """Instantaneous phase of each trace, band-passed around ``f_center`` (Hilbert via FFT).

    Same construction as SummaryStatistics._analytic_bandpass: keep only positive frequencies inside the
    band and double them, so the inverse transform is the analytic signal."""
    # float32 throughout, and a REAL-input FFT. This is a diagnostic phase, consumed only by
    # cycle_average's binning (48 bins over 2*pi -- ~0.13 rad each), so float64 bought nothing while
    # costing a lot: the old form held a float64 copy of x, a complex128 fft, a complex128 product
    # and a complex128 ifft, ~2.4 GB peak at 1000 draws x 60000 samples. It also sat inside
    # _emit_overlay_figures' try, so an OOM here degraded to a one-line warning and you lost figures
    # without learning why.
    #
    # rfft/irfft rather than fft/ifft: the input is real, so the negative-frequency half is redundant
    # -- and this filter DISCARDS it anyway (keep is `ff > 0`). Doubling the positive bins of an
    # rfft and taking a complex irfft is the same analytic signal, from half the spectrum.
    x = (x - x.mean(dim=-1, keepdim=True)).to(torch.float32)
    n = x.shape[-1]
    xf = torch.fft.rfft(x, dim=-1)                                   # (B, n//2+1) complex64
    ff = torch.fft.rfftfreq(n, d=float(dt)).to(torch.float32)
    keep = (ff > 0) & (ff >= bp_lo * f_center) & (ff <= bp_hi * f_center)
    xf = xf * (keep.to(torch.float32) * 2.0).unsqueeze(0)
    # Rebuild the full analytic spectrum: positive bins as-is, negatives zero (that IS the analytic
    # signal), so a plain ifft over the full length recovers the complex envelope.
    full = torch.zeros((*xf.shape[:-1], n), dtype=xf.dtype, device=xf.device)
    full[..., :xf.shape[-1]] = xf
    z = torch.fft.ifft(full, dim=-1)
    return torch.angle(z)


def cycle_average(traces: torch.Tensor, dt: float, f_center: float, n_bins: int = 48,
                  lo_pct: float = 25.0, hi_pct: float = 75.0) -> tuple:
    """Fold traces onto ONE oscillation cycle and average -- waveform shape without any phase reference.

    Each sample is binned by its instantaneous phase (from the analytic signal around ``f_center``), so
    the result is the mean cycle shape: the asymmetric spike of a hair-bundle oscillation survives, while
    the arbitrary absolute phase does not.

    ⚠ NON-FINITE SAMPLES ARE MASKED OUT BEFORE BINNING, and the count is returned. Without this a
    single divergent draw poisons a bin rather than being excluded: a NaN phase goes through
    ``.long()`` as an implementation-defined garbage integer, which ``clamp`` then parks in bin 0 --
    so the damage is silent, confined to one end of the curve, and easy to read as a real feature.
    Same root cause as the dropped rows in :func:`psd_band`; see the note there.

    Bit-identical to the previous implementation whenever everything is finite: the mask is then
    all-True and boolean indexing flattens in the same row-major order the old ``reshape(-1)`` did, so
    each bin still sums its samples in the original sequence.

    :return: (phase_bin_centres in [0, 2pi), mean, lo, hi, n_dropped), the first four (n_bins,).
             Empty bins are NaN.
    """
    phase = _analytic_phase(traces, dt, f_center)                     # (B, n) in (-pi, pi]
    x = (traces - traces.mean(dim=-1, keepdim=True)).to(torch.float64)
    finite = torch.isfinite(phase) & torch.isfinite(x)
    n_dropped = int((~finite).sum())
    flat_x = x[finite]
    flat_i = ((((phase[finite].to(torch.float64) + math.pi) / (2 * math.pi)) * n_bins)
              .long().clamp(0, n_bins - 1))
    centres = (torch.arange(n_bins, dtype=torch.float64) + 0.5) * (2 * math.pi / n_bins)
    mean = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    lo = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    hi = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    # ONE stable sort, then slice per bin. The old form evaluated `flat_x[flat_i == b]` inside the
    # loop -- a full boolean pass plus a gather over ALL B*n elements for EVERY bin, i.e. 48 complete
    # sweeps (~2.9 billion comparisons at 1000 draws x 60k samples) to produce one diagnostic figure.
    # stable=True keeps each bin's samples in their original order, so the per-bin mean sums in the
    # same sequence as before and the results stay bit-identical.
    order = torch.argsort(flat_i, stable=True)
    sorted_x = flat_x[order]
    counts = torch.bincount(flat_i, minlength=n_bins)
    starts = torch.cumsum(counts, 0) - counts
    for b in range(n_bins):
        c = int(counts[b])
        if c:
            s = int(starts[b])
            vals = sorted_x[s:s + c]
            mean[b] = vals.mean()
            lo[b] = torch.quantile(vals, lo_pct / 100.0)
            hi[b] = torch.quantile(vals, hi_pct / 100.0)
    return centres, mean, lo, hi, n_dropped


def cycle_window(n_total: int, dt: float, f_peak: float, n_cycles: int) -> int:
    """How many samples span ``n_cycles`` periods of ``f_peak`` (clamped to the record length)."""
    if f_peak <= 0:
        return n_total
    return int(min(n_total, max(16, round(n_cycles / (f_peak * float(dt))))))
