"""Multi-frequency susceptibility chi(omega) features for the SBI pipeline (CHI_MODE).

Pure math helpers only -- NO simulation and NO dependency on core.SBI.pipeline (the K single-tone
forced runs live in pipeline.gen_chi_block, which owns gen_obs / build_nondim_sin_force_tensor). This
module supplies: the relative-frequency grid, a batched spontaneous peak-frequency estimator, a
batched lock-in, the [log|chi|, cos, sin] feature packing, and the feature labels.

chi(omega) generalizes the single-frequency Group-G lock-in (statistics.SummaryStatistics._group_g)
to a K-frequency CURVE. The drive is K single-tone recordings at omega_k = mult_k * Omega_0, with the
multipliers log-spaced over config.CHI_FREQ_BOUNDS and Omega_0 the measured spontaneous peak (so the
resonance sits in a canonical frame regardless of where t_scale puts it -- mirrors the FDT pipeline's
data-driven grid). See config.CHI_MODE for the full rationale.

Unit convention (matches the forced path + core/FDT/spectral.lock_in_chi): chi is measured on the
REDIMENSIONALIZED response x_dim with the DIMENSIONAL drive amplitude F0, so chi = response/drive
carries the physical x_scale/f_scale magnitude (like Group-G's gain), while its SHAPE over omega
carries the ND resonance (kappa/lambda/...). Frequencies are in cell frequency units; times in cell
time units (dt_exp). Accumulations are done in float64 (long low-omega lock-in sums), then cast back.
"""
import math
from typing import NamedTuple

import torch

from core import config

_EPS = 1e-12

# Time-chunk length for lock_in_batched's float64 accumulation, and sample sub-batch for peak_freq's
# FFT. Both are pure memory knobs with no effect on the returned values. 8192 keeps the lock-in's
# working set at ~0.5 GB even at the largest training batch; 256 mirrors pipeline.gen_stats'
# stats_batch_size, which is the same treatment for the same reason.
_LOCK_IN_CHUNK = 8192
_PEAK_FREQ_BATCH = 256


# The THREE chi feature sets. They are different widths for different consumers and conflating them is
# silent, so they are named once here and every consumer asks for one by name.
#
#   CONDITIONING  6 channels x K_PAD slots  -> the network, expected_forcing_dim, the sidecar
#   FISHER        3 channels x K probes     -> decorrelate.feats, scripts/degeneracy_map
#   (the diagnostic labels in scripts/_common are the FISHER set minus the 11 Group-G columns)
#
# `u`, `mask` and `logcyc` are deliberately ABSENT from the Fisher set. The reason is one property with
# three faces: a standardized Jacobian DIVIDES by an ensemble std (`fnoise = max(std, 1e-9)`), so a
# channel that barely varies with theta is an AMPLIFIER, not a quiet row. Note the asymmetry that makes
# this so hard to see -- an EXACTLY constant channel is harmless (0/1e-9 = 0, which is why chi mode's
# 11 zeroed Group-G columns cost nothing), while a NEARLY constant one writes order-1-to-1e4 entries
# into the matrix that defines the flow's coordinate system, with V still orthogonal to 1e-4 and every
# test passing.
#
#   `u = log(f_k/f_peak)` is theta-INDEPENDENT wherever the Fisher probes a deterministic multiplier
#   grid: across an ensemble it is log(mult_k) plus one float32 multiply-and-divide of rounding, std
#   ~2.5e-8. (That is 25x ABOVE the 1e-9 clamp, so the clamp protects nothing -- any guard for this
#   has to be RELATIVE to the channel's own magnitude.)
#
#   `mask` is theta-dependent but DISCONTINUOUS -- a step of 1 over the same floor. See trap CHI2.
#
#   `logcyc` was kept here until 2026-08-10 on the grounds that it "genuinely varies with theta through
#   f_peak". True, and insufficient: it varies EXACTLY as a row already in the feature set does. With
#   the ceiling clear, logcyc_j = log(mult_j) + log(f_peak) + log(T_obs), whose first and third terms
#   are constants that vanish under standardization -- so the standardized row IS `A3_log_fpeak`'s, and
#   K probes contribute K exact duplicates that weight that one direction K-fold in J^T J. Measured on
#   a real rotation: four of six rows agreed to 6 significant figures. With the ceiling BINDING,
#   freq*T_row -> CHI_MAX_CYCLES and all that remains is the sawtooth of floor(), std ~1e-4 -- the
#   amplifier case, measured at max|J| = 2.0e4 against 289 for the largest real feature in
#   scripts/degeneracy_map, and at a harmless 6.0 in a rotation whose +-dz arms happened not to
#   straddle a quantization step. Intermittent, not benign: a rotation averages 8 operating points over
#   a ~4-decade Omega_0 prior. Duplicate or quantization, never independent -- so it leaves. Backlog
#   C-9/C-10.
#
# This is about the FISHER set ONLY. In the CONDITIONING block `logcyc` is genuinely informative,
# because training varies placement AND duration per row (C-6/C-8), so the constants above are not
# constant there. Do not "simplify" the two lists toward each other.
CHI_COND_CHANNELS = ("u", "logmag", "cos", "sin", "logcyc", "mask")
CHI_FISHER_CHANNELS = ("logmag", "cos", "sin")


def n_chi_features(k_pad: int | None = None) -> int:
    """Width of the chi(omega) conditioning block. **K-INDEPENDENT** -- it is a function of the pad.

    This one line is what buys the payoff: a posterior trained with K drawn over 2..K_PAD loads
    against a config declaring a different probe count with no width guard loosened anywhere.
    """
    kp = config.CHI_K_PAD if k_pad is None else int(k_pad)
    return config.CHI_ELEM_W * kp


def chi_labels(k: int | None = None, channels: tuple = CHI_COND_CHANNELS) -> list[str]:
    """Ordered labels for one of the chi feature sets; ``channels`` picks which.

    Conditioning labels say ``chiS{j}`` -- S for SLOT. A slot is not a probe identity: which probe
    lands in slot j depends on the observation, so a per-column diagnostic table keyed by slot must
    not be read as "probe j".
    """
    n = config.CHI_K_PAD if k is None else int(k)
    stem = "chiS" if channels is CHI_COND_CHANNELS else "chi"
    return [f"{stem}{j}_{ch}" for j in range(n) for ch in channels]


def band_norm(bounds: tuple | None = None) -> tuple[float, float]:
    """(u_mid, u_half) mapping the log-frequency band onto u_hat in [-1, 1].

    Fixed from the BAND, never fitted from data: it is baked into a trained encoder, so the load path
    compares it and refuses a posterior whose band differs.
    """
    lo, hi = config.CHI_FREQ_BOUNDS if bounds is None else bounds
    return 0.5 * (math.log(lo) + math.log(hi)), 0.5 * (math.log(hi) - math.log(lo))


class ProbeVerdict(NamedTuple):
    """What will happen to one experimental chi probe, decided WITHOUT simulating anything.

    ``action`` is one of:
      ``"use"``       the probe contributes in full.
      ``"truncate"``  the recording is longer than the CHI_MAX_CYCLES ceiling; the leading prefix is
                      used and the tail discarded. The recording is fine -- only its tail is unusable.
      ``"mask"``      under CHI_MIN_CYCLES drive cycles, so the probe is kept in the set but
                      contributes nothing. NOT an error: training masks these too, so the network has
                      learned to condition on sets with absent probes.
      ``"refuse"``    a structural mistake -- a bad, aliased or out-of-band frequency, which means the
                      user recorded something this posterior cannot interpret at all.

    The mask/refuse split is about train/eval CONSISTENCY, not severity. See build_experiment_obs_chi.
    """
    action: str
    reason: str            # "" when action == "use"
    cycles: float          # drive cycles actually locked in over, after any truncation
    n_use: int             # samples actually locked in over
    min_seconds: float     # recording length needed to clear the floor at this frequency
    max_seconds: float     # length beyond which the ceiling truncates


def probe_verdict(cfg, f_peak_cell: float, freq_hz: float, n_samples: int) -> ProbeVerdict:
    """Classify one experimental probe against the SAME predicates build_experiment_obs_chi applies.

    THIS IS THE SHARED SOURCE OF TRUTH, and that is the whole point of it existing. The GUI's probe
    planner (backlog C-3) has to tell a user what is in band and how long to record BEFORE a bench
    session; the experimental path has to refuse/mask/truncate the same way AFTER it. Those two
    re-deriving the same five predicates separately is precisely how a diagnostic comes to disagree
    with the thing it is diagnosing -- so the planner and the runtime call this, and neither owns a
    copy. It simulates nothing and raises nothing: a REFUSAL is returned as a verdict, because a
    planner must be able to describe a bad probe without dying on it. build_experiment_obs_chi is what
    turns ``action == "refuse"`` into a ValueError.

    :param cfg: SimConfig (chi band, ceiling, dt_exp, unit conversions).
    :param f_peak_cell: measured Omega_0 in CELL frequency units (chi.peak_freq's output).
    :param freq_hz: the frequency this recording was actually driven at, in Hz.
    :param n_samples: length of that forced recording, in samples at 1/cfg.dt_exp.
    """
    s_to_cell = cfg.get_unit_conversion_factor("s")
    lo_b, hi_b = cfg.chi_freq_bounds
    dt = cfg.dt_exp
    nyq = 0.5 / dt

    def _sec(cell_freq, cycles):
        return cycles / cell_freq / s_to_cell if cell_freq > 0 else float("inf")

    if not (math.isfinite(freq_hz) and freq_hz > 0):
        return ProbeVerdict("refuse", f"drive frequency must be finite and positive, got {freq_hz} Hz",
                            0.0, 0, float("inf"), float("inf"))

    f_val = freq_hz * cfg.freq_si_to_cell
    min_s, max_s = _sec(f_val, config.CHI_MIN_CYCLES), _sec(f_val, cfg.chi_max_cycles)

    if f_val >= 0.9 * nyq:
        return ProbeVerdict(
            "refuse", f"{freq_hz:g} Hz is at or above the recording's Nyquist limit "
                      f"({0.9 * nyq / cfg.freq_si_to_cell:g} Hz at dt_exp={dt:g})",
            0.0, 0, min_s, max_s)

    u_mid, u_half = band_norm(cfg.chi_freq_bounds)
    if f_peak_cell <= 0 or not math.isfinite(f_peak_cell):
        return ProbeVerdict("refuse", f"Omega_0 came back as {f_peak_cell}; the passive recording "
                                      f"has no usable spectral peak", 0.0, 0, min_s, max_s)
    if abs((math.log(f_val / f_peak_cell) - u_mid) / u_half) > config.CHI_UHAT_MAX:
        band = (lo_b * f_peak_cell / cfg.freq_si_to_cell, hi_b * f_peak_cell / cfg.freq_si_to_cell)
        return ProbeVerdict(
            "refuse", f"{freq_hz:g} Hz is outside the band this posterior was trained over. For this "
                      f"cell (Omega_0 = {f_peak_cell / cfg.freq_si_to_cell:g} Hz) that is "
                      f"{band[0]:g}-{band[1]:g} Hz",
            0.0, 0, min_s, max_s)

    n_cap = max(1, int(math.floor(cfg.chi_max_cycles / f_val / dt)))
    n_use = min(int(n_samples), n_cap)
    cycles = f_val * n_use * dt
    if n_use < int(n_samples):
        return ProbeVerdict(
            "truncate", f"{n_samples} samples give {f_val * n_samples * dt:.1f} drive cycles, above "
                        f"the {cfg.chi_max_cycles:g}-cycle ceiling; only the first {n_use} samples "
                        f"({n_use * dt / s_to_cell:.3g} s) are used",
            cycles, n_use, min_s, max_s)
    if cycles < config.CHI_MIN_CYCLES:
        return ProbeVerdict(
            "mask", f"{n_samples} samples give only {cycles:.2f} drive cycles, below the "
                    f"{config.CHI_MIN_CYCLES:g}-cycle floor, so this probe is MASKED and contributes "
                    f"nothing. Record >= {min_s:.3g} s at this frequency to use it",
            cycles, n_use, min_s, max_s)
    return ProbeVerdict("use", "", cycles, n_use, min_s, max_s)


def band_hz(cfg, f_peak_cell: float) -> tuple[float, float]:
    """(lo, hi) of the trained band in Hz for a cell whose Omega_0 is ``f_peak_cell`` (cell units).

    The band is RELATIVE to each cell's own Omega_0, so "what is in band" is not a property of the
    posterior alone -- it cannot be answered until a passive recording has been taken. That is why the
    planner needs one before it can say anything useful.
    """
    lo_b, hi_b = cfg.chi_freq_bounds
    return (lo_b * f_peak_cell / cfg.freq_si_to_cell, hi_b * f_peak_cell / cfg.freq_si_to_cell)


def sample_multipliers(k: int, bounds: tuple | None = None, *, generator=None,
                       dtype: torch.dtype = torch.float32,
                       device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """(k,) STRATIFIED-JITTERED log-spaced multipliers spanning ``bounds``, sorted ascending.

    Training placement. One draw per stratum rather than k iid draws: quadrature variance falls as
    O(k^-3) instead of O(k^-1), and every row spans the band with no holes and no clusters -- which
    matters precisely because k can be 1 or 2.

    Deliberately NOT the deterministic grid used for observations. A fixed grid would leave the
    encoder's frequency channel taking only k distinct values across the entire training set, so an
    experimentalist's 0.07x recording would be an out-of-distribution input to an MLP that
    extrapolates linearly and confidently.

    Draws from ``generator``, never the global RNG -- the chi block is bracketed by deliberate
    manual_seed() calls for common random numbers, which would otherwise re-randomise or freeze it.
    """
    lo, hi = config.CHI_FREQ_BOUNDS if bounds is None else bounds
    u_lo, u_hi = math.log(lo), math.log(hi)
    xi = torch.rand(int(k), generator=generator)
    a = u_lo + (u_hi - u_lo) * (torch.arange(int(k), dtype=torch.float64) + xi.double()) / int(k)
    return torch.exp(a).to(dtype=dtype, device=device)


def resolvable_multipliers(mults: torch.Tensor, f_peak: torch.Tensor, T_obs: float,
                           bounds: tuple | None = None,
                           min_cycles: float | None = None) -> torch.Tensor:
    """Lift each ROW's multipliers into the sub-band its own Omega_0 can actually resolve -> (B, K).

    THE PROBLEM (backlog C-6, measured in PRISM_HANDOFF 4.3.4). Probes sit at ``mult * Omega_0``, so
    a probe resolves only if ``mult * Omega_0 * T >= CHI_MIN_CYCLES``. The prior spans ~4 decades of
    Omega_0 while the band is a FIXED window of multipliers, so with one shared multiplier set the
    slow rows cannot resolve at any placement in the band: measured, 55 % of training rows carried
    ZERO live probes, going 0 % live below 3 Hz to 98 % above 30 Hz. Those rows are genuine
    oscillators (spectral prominence in the thousands), so masking them is correct -- the fix is to
    stop ASKING them for probes they cannot deliver.

    WHAT THIS DOES. Per row, the usable sub-band is ``[max(lo, min_cycles/(f*T)), hi]``. The drawn
    multipliers are mapped onto it by an affine transform in LOG space, which preserves their
    ordering, their relative spacing and the stratified jitter that
    :func:`sample_multipliers` produced -- the placement is compressed, not redrawn. Costs nothing:
    the frequencies were already per-row (``freqs = mults * f_peak`` is ``(B, K)``, each row driven
    at its own frequency) and only the multipliers were shared, so this changes no simulation count.

    WHAT IT DOES NOT DO, and this must be stated wherever the result is used: a slow row ends up
    probed near the band's TOP, so it gains resolution but loses frequency SPREAD -- and spread is
    what chi(omega) exists to measure. This converts inert rows into weakly-informative ones, not
    into good ones. A row too slow even for the band's top edge (``lo_row >= hi``) is left at ``hi``
    and will still be masked; that is the honest outcome, not a failure of this function.

    ONLY the floor is a placement bound. ``max_cycles`` is deliberately NOT one, and the reason is
    arithmetic: the band's dynamic range (``hi/lo`` = 10 at the configured band) is exactly the cycle
    window's (``CHI_MAX_CYCLES/CHI_MIN_CYCLES`` = 10), so requiring a row to satisfy both leaves it a
    single feasible multiplier for all but one value of ``Omega_0 * T`` -- every probe on one
    frequency, resolving perfectly and measuring no SHAPE at all. Tried, and reverted for that reason.
    The asymmetry that decides it: falling under the floor MASKS a probe outright, while exceeding the
    ceiling only makes it noisier, and gen_chi_raw's duration ceiling still truncates it afterwards.
    A hard bound belongs on the failure that loses the probe, not on the one that degrades it.

    *That leaves a known limit, and it is not this function's to fix:* the duration ceiling is keyed
    on the batch's FASTEST row, so a slow row it just rescued can still be truncated back under the
    floor. Measured, that holds the rescue to ~47 % live. The real fix is a per-ROW lock-in duration
    (``lock_in_batched`` takes a scalar ``T_obs`` today), which is a change to the estimator itself.

    NEVER leaves the band: every returned multiplier is within ``[lo, hi]``, so ``|u_hat| <= 1`` and
    the packer's CHI_UHAT_MAX filter cannot fire on placement alone.

    :param mults: (K,) or (B, K) relative multipliers as drawn.
    :param f_peak: (B,) per-row measured Omega_0, cell frequency units.
    :param T_obs: the lock-in duration available, cell time units (same units as 1/f_peak).
    :return: (B, K) multipliers, per row.
    """
    lo, hi = config.CHI_FREQ_BOUNDS if bounds is None else bounds
    floor = config.CHI_MIN_CYCLES if min_cycles is None else float(min_cycles)
    m = mults if torch.is_tensor(mults) else torch.as_tensor(mults)
    m = m.to(device=f_peak.device, dtype=f_peak.dtype)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    m = m.expand(f_peak.shape[0], -1)                                   # (B, K)

    u_lo, u_hi = math.log(lo), math.log(hi)
    fT = (f_peak.clamp(min=1e-30) * max(T_obs, 1e-30)).unsqueeze(1)     # (B, 1) cycles per unit mult
    # The row's usable window: lifted off the floor, still capped by the band. A row fast enough that
    # the floor is already below `lo` keeps the FULL band and is returned untouched.
    u_lo_row = torch.clamp(torch.log((floor / fT).clamp(min=1e-30)), min=u_lo)

    span = u_hi - u_lo_row
    ok = span > 0
    u = torch.log(m.clamp(min=1e-30))
    scaled = u_lo_row + (u - u_lo) * (span / max(u_hi - u_lo, 1e-30))
    # A row too slow even for the band's top edge: park every probe there -- the most resolvable
    # placement that exists -- and let the floor mask them. Honest, and unchanged from before.
    out = torch.where(ok, scaled, torch.full_like(scaled, u_hi))
    # clamp is belt-and-braces against float drift at the edges; the algebra already lands inside.
    return torch.exp(out).clamp(min=lo, max=hi)


def chi_multipliers(dtype: torch.dtype = torch.float32,
                    device: torch.device = torch.device("cpu"),
                    n_freqs: int | None = None,
                    bounds: tuple | None = None) -> torch.Tensor:
    """(K,) log-spaced multipliers of the spontaneous peak Omega_0 spanning ``bounds``.

    ``n_freqs``/``bounds`` default to the module values, but callers in the pipeline pass the values
    carried on the SimConfig so a run is self-describing (see SimConfig.chi_n_freqs)."""
    k = config.CHI_N_FREQS if n_freqs is None else int(n_freqs)
    lo, hi = config.CHI_FREQ_BOUNDS if bounds is None else bounds
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), k, dtype=dtype, device=device))


def chi_multipliers_for(cfg, dtype=None, device=None) -> torch.Tensor:
    """``chi_multipliers`` using the K / bounds carried on ``cfg``. The one call the pipeline should use."""
    return chi_multipliers(dtype=cfg.hw.dtype if dtype is None else dtype,
                           device=cfg.hw.device if device is None else device,
                           n_freqs=cfg.chi_n_freqs, bounds=cfg.chi_freq_bounds)


def peak_freq(x: torch.Tensor, dt: float, batch: int = _PEAK_FREQ_BATCH) -> torch.Tensor:
    """
    Per-sample spontaneous peak frequency Omega_0/2pi (cell freq units) from a batch of traces.

    Mirrors statistics.SummaryStatistics._build_spectral: rfft of the demeaned trace, drop the DC bin,
    take the argmax, clamp to the first non-zero bin. One noisy trace gives a noisy peak (as the real
    passive recording would); the network also sees A3 (log f_peak) so it is not blind to the estimate.

    Sub-batched over SAMPLES, the same treatment (and for the same reason) as pipeline.gen_stats'
    stats_batch_size: every row is independent, so this is numerically identical, but it keeps the
    rfft off the full training batch. That matters twice over -- the (B, n/2+1) complex spectrum is
    the allocation, and n = N_points is an arbitrary integer drawn from a continuous distribution, so
    roughly half of all batches land on a length with large prime factors where cuFFT falls back to
    Bluestein and needs ~2.4x the memory.

    :param x: (B, n) trajectories (physical, sampled at dt).
    :param dt: sampling interval (cell time units).
    :param batch: samples per FFT sub-batch. Memory only -- numerically irrelevant.
    :return: (B,) peak frequencies (cell freq units).
    """
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=float(dt), device=x.device).to(x.dtype)  # (nfr,)
    df = (freqs[1] - freqs[0]) if freqs.numel() > 1 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
    step = max(1, int(batch))
    idx = []
    for s in range(0, x.shape[0], step):
        xs = x[s:s + step]
        xs = xs - xs.mean(dim=-1, keepdim=True)
        psd = (torch.fft.rfft(xs, dim=-1).abs().clamp(max=1e15) ** 2)
        psd[:, 0] = 0.0                                         # kill DC so the peak is the resonance
        idx.append(psd.argmax(dim=-1))                          # (b,)
        del xs, psd
    peak_idx = (torch.cat(idx) if idx else
                torch.zeros(0, dtype=torch.long, device=x.device))          # (B,)
    return freqs[peak_idx].clamp(min=df)


def lock_in_batched(x: torch.Tensor, omega: torch.Tensor, F0, T_obs, dt: float,
                    chunk: int = _LOCK_IN_CHUNK, n_samples: torch.Tensor | None = None) -> torch.Tensor:
    """
    Batched complex susceptibility by lock-in detection (per-sample omega), mirroring
    core/FDT/spectral.lock_in_chi:

        chi_b = (2 / (F0_b * T_obs)) * sum_n (x_b[n] - <x_b>) * exp(i*omega_b*t_n) * dt

    Accumulated in float64 over TIME CHUNKS, with the real and imaginary sums built separately so no
    (B, n) complex128 tensor is ever materialized. The naive form holds ~64 bytes per (B, n) element
    concurrently (x as f64, the omega (x) t phase matrix, e_iwt as c128, x recast to c128, and their
    product); chunking drops that to 64 bytes per (B, chunk) element, which at the training batch
    size is the difference between ~10 GB and ~0.2 GB. Results agree with the naive form to float64
    round-off; see tests/test_user_sbi.py::test_lock_in_chunking_matches_full_batch.

    TWO INVARIANTS, both of which fail SILENTLY (finite, in-range, wrong):
      * the mean is over the trace ACTUALLY LOCKED IN, computed in pass 1. Demeaning per chunk would
        be a high-pass filter with corner ~1/(chunk*dt), which preferentially eats the LOW-multiplier
        probes -- and CHI_FREQ_BOUNDS is entirely SUB-RESONANCE, so EVERY probe is a low-multiplier
        one. Under ``n_samples`` this means the mean is over each row's OWN prefix: taking it over the
        full width instead would subtract a level the row's samples never had, and the residual would
        land at DC where a low-frequency lock-in is most sensitive to it.
      * the phase in a chunk starting at s uses the ABSOLUTE times arange(s, e)*dt, never
        arange(0, e-s)*dt. (The wrong form is normally an order-unity error, but it is invisible when
        omega*chunk*dt happens to be a multiple of 2*pi -- so test with incommensurate omega.)

    :param x: (B, n) response traces (physical), sampled at dt.
    :param omega: (B,) drive ANGULAR frequencies (cell freq units, rad/cell-time).
    :param F0: drive amplitude the response was driven with -- a scalar OR a (B,) per-sample tensor
               (chi is drive-amplitude-independent in the linear regime, but F0 must MATCH the drive
               so the units are physical).
    :param T_obs: observation duration covered by the trace, cell time units. Scalar, or (B,) to go
                  with a per-row ``n_samples``.
    :param dt: sampling interval (scalar, cell time units).
    :param chunk: time-chunk length. Memory only -- numerically irrelevant.
    :param n_samples: (B,) leading sample count per row, or None for "all n". This is what lets ONE
                      batched call lock in rows over DIFFERENT durations, which chi mode needs
                      because Omega_0 spans ~4 decades within a training batch: a single shared
                      duration must be keyed on the fastest row to respect CHI_MAX_CYCLES, and that
                      truncates the slow rows below CHI_MIN_CYCLES and masks them (backlog C-6/C-8,
                      handoff 4.3.5). Rows are MASKED rather than sliced -- the tensor stays
                      rectangular, so the chunked float64 accumulation is unchanged and the memory
                      bound grows by one BOOL mask per chunk: +16 MiB at B=2048, chunk=8192, against
                      a 553.7 MiB unmasked peak. (C-8 claimed "unchanged" and shipped a float64 mask,
                      which was +128 MiB. See _mask.)
    :return: (B,) complex128 susceptibilities.
    """
    B, n = x.shape[0], x.shape[-1]
    dev, dt = x.device, float(dt)
    step = max(1, int(chunk))

    # Per-row live length. None -> every row uses all n, and every mask below is all-True, so the
    # scalar path is bit-identical to the pre-C-8 code (pinned by the chunking test).
    if n_samples is None:
        n_live = torch.full((B,), n, dtype=torch.float64, device=dev)
        limit = None
    else:
        limit = n_samples.to(device=dev).reshape(-1).long().clamp(min=0, max=n)
        n_live = limit.to(torch.float64)

    def _mask(s, e):
        """(B, e-s) BOOL -- whether column j is inside that row's prefix.

        BOOL, NOT float64, and the distinction is 112 MiB. This mask is only ever multiplied into a
        float64 tensor, where torch promotes it to exactly 1.0 / 0.0 -- so every result below is
        BIT-IDENTICAL to a float64 mask (verified with torch.equal), while the mask itself costs one
        byte per element instead of eight. At the training batch (B=2048, chunk=8192) that is 16 MiB
        rather than 128 MiB.

        C-8 shipped this as float64, and the claim in this function's docstring that per-row masking
        leaves "the chunked float64 accumulation and its memory bound ... unchanged" was wrong by one
        full (B, chunk) float64 tensor. Measured peak for the whole call at that batch: 553.7 MiB
        unmasked (pre-C-8), 681.7 MiB as C-8 shipped it, 569.7 MiB now. The regression was invisible
        because the only end-to-end exercise was the smoke train at RUN_SIZE=32, where 128 MiB is 2 MB.
        """
        if limit is None:
            return None
        cols = torch.arange(s, e, device=dev).unsqueeze(0)                    # (1, e-s)
        return cols < limit.unsqueeze(1)

    # pass 1: the float64 row mean over each row's OWN prefix (see invariant 1 above)
    acc = torch.zeros(B, dtype=torch.float64, device=dev)
    for s in range(0, n, step):
        e = min(s + step, n)
        xs = x[..., s:e].to(torch.float64)
        m = _mask(s, e)
        acc += (xs if m is None else xs * m).sum(dim=-1)
    mean = (acc / n_live.clamp(min=1.0)).unsqueeze(-1)                        # (B, 1)

    # pass 2: separate real / imaginary lock-in sums, one time chunk at a time
    w = omega.to(torch.float64).reshape(-1, 1)                                # (B, 1)
    re = torch.zeros(B, dtype=torch.float64, device=dev)
    im = torch.zeros(B, dtype=torch.float64, device=dev)
    for s in range(0, n, step):
        e = min(s + step, n)
        t = torch.arange(s, e, device=dev, dtype=torch.float64) * dt          # ABSOLUTE times
        phase = w * t.unsqueeze(0)                                            # (B, e-s)
        xc = x[..., s:e].to(torch.float64) - mean
        m = _mask(s, e)
        if m is not None:
            # Mask AFTER demeaning: zeroing first would leave -mean in the dead columns, which is a
            # step function at the prefix boundary -- exactly the DC-adjacent artefact invariant 1
            # exists to prevent, and it grows with how much of the row is dead.
            #
            # IN PLACE, which is safe HERE and only here. `xc` is the result of `... - mean`, so it is
            # always a freshly allocated tensor that cannot alias the caller's `x`. Pass 1's `xs` is
            # NOT: when `x` already arrives as float64, `.to(torch.float64)` returns the input VIEW,
            # and mutating it would corrupt the caller's tensor between the two calls
            # test_lock_in_per_row_durations_match_locking_each_row_alone makes on one array -- which
            # would surface as "the rows are coupled", pointing at the wrong bug entirely.
            xc.mul_(m)
        re += (xc * torch.cos(phase)).sum(dim=-1)
        im += (xc * torch.sin(phase)).sum(dim=-1)

    F0 = F0.to(torch.float64).reshape(-1) if torch.is_tensor(F0) else float(F0)
    T = T_obs.to(torch.float64).reshape(-1) if torch.is_tensor(T_obs) else float(T_obs)
    return torch.complex(re, im) * (2.0 / (F0 * T)) * dt


def _mag_phase(chi_stack: torch.Tensor):
    """(log|chi|, cos arg, sin arg) from a complex stack. log for the positive, unbounded magnitude;
    a (cos, sin) pair for the phase so there is no 2pi wrap -- the statistics.py convention."""
    ang = torch.angle(chi_stack)
    return torch.log(torch.clamp(chi_stack.abs(), min=_EPS)), torch.cos(ang), torch.sin(ang)


def fisher_features(chi_stack: torch.Tensor) -> torch.Tensor:
    """(B, K) complex -> (B, 3K) float32: [log|chi|, cos, sin] per probe.

    The FISHER set: no pad slots, no mask, no frequency channel and no cycle count -- see
    CHI_FISHER_CHANNELS for why each of those is an AMPLIFIER rather than a quiet row once the
    Jacobian divides by an ensemble std.

    TAKES ONE ARGUMENT ON PURPOSE. It used to take `logcyc` too, and `scripts/degeneracy_map` passed
    it `gen_chi_raw(...)[:2]` -- i.e. `u`, the one channel this docstring warned about -- for three
    commits, silently, because a 4-tuple sliced to 2 still unpacks into 2 names. A one-argument
    signature makes that entire class of mistake a TypeError. Do not add a second parameter back
    without reading trap CHI10 first.
    """
    logmag, cos, sin = _mag_phase(chi_stack)
    feats = torch.stack([logmag, cos, sin], dim=-1)                            # (B, K, 3)
    out = feats.reshape(feats.shape[0], -1)
    return torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)


def pack_probe_block(chi_stack: torch.Tensor, u: torch.Tensor, logcyc: torch.Tensor,
                     valid: torch.Tensor, k_pad: int | None = None,
                     bounds: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack K probes into the padded conditioning block.

    :param chi_stack: (B, K) complex susceptibilities.
    :param u:         (B, K) log(f_probe / f_peak) -- the frequency ACTUALLY locked in at.
    :param logcyc:    (B, K) log(f_probe * T_probe) -- drive cycles inside the locked-in segment.
    :param valid:     (B, K) bool, the caller's verdict (Nyquist, resolution floor, ...).
    :return: ((B, CHI_ELEM_W*k_pad) float32 block, (B, k_pad) bool mask)

    THREE properties the rest of the design leans on:

    * **A failed probe is MASKED, never a phantom.** The old packer ran nan_to_num over the whole
      block, so a non-finite lock-in became a live-looking (0, 0, 0) triple -- and cos^2+sin^2 = 1
      says no real probe can produce that. Here `mask` is the single verdict: simulated AND finite
      AND resolvable AND in band.
    * **Pads are EXACTLY 0.0 in all six channels.** Deterministic, so every downstream constant-column
      filter (posterior_predictive_check's s_std > 1e-10, overlay.rank_by_stats) drops them; and
      finite, so train_nn's isfinite/|x|<1e15 filter and gen_cal_data's validity mask drop ZERO rows.
      Never torch.empty, never NaN.
    * **Valid probes are packed contiguously into slots 0..n-1, ascending in frequency.** The encoder
      is provably slot-invariant so this is free for learning, but it makes the layout deterministic
      across generate_observations / gen_training_data / the PPC / the experimental path, which is
      what lets a test compare them byte-for-byte.

    Raises rather than truncating when K > k_pad: silently dropping probes the caller paid to
    simulate is the kind of attrition that shows up as a mysteriously uninformative posterior.
    """
    kp = config.CHI_K_PAD if k_pad is None else int(k_pad)
    B, K = chi_stack.shape
    if K > kp:
        raise ValueError(
            f"chi: {K} probes cannot be packed into {kp} slots (CHI_K_PAD). The pad is frozen into "
            f"every posterior trained with it, so raise it deliberately and retrain, or probe less.")

    u_mid, u_half = band_norm(bounds)
    m = (valid.to(torch.bool)
         & torch.isfinite(chi_stack.abs()) & (chi_stack.abs() > 0)
         & torch.isfinite(u) & torch.isfinite(logcyc)
         & (((u - u_mid) / u_half).abs() <= config.CHI_UHAT_MAX))

    # Canonical order: valid slots first, ASCENDING IN FREQUENCY among them. Sorting here rather than
    # trusting the caller matters for the experimental path, where the frequencies are whatever the
    # operator typed in whatever order they typed them -- the layout must not depend on that. The
    # encoder is permutation-invariant so this is free for learning; it exists so the simulated and
    # experimental paths produce byte-comparable blocks for the same probe set.
    big = torch.finfo(u.dtype).max
    order = torch.argsort(torch.where(m, u, torch.full_like(u, big)), dim=1, stable=True)
    logmag, cos, sin = _mag_phase(chi_stack)
    elem = torch.stack([u, logmag, cos, sin, logcyc.to(u.dtype), m.to(u.dtype)], dim=-1)  # (B,K,6)
    elem = torch.gather(elem, 1, order.unsqueeze(-1).expand(-1, -1, config.CHI_ELEM_W))
    m_sorted = torch.gather(m, 1, order)
    # Zero the whole slot wherever the mask is off, so a dead slot is 0.0 in every channel including
    # any non-finite u/logcyc that came with it.
    elem = torch.nan_to_num(elem, nan=0.0, posinf=0.0, neginf=0.0) * m_sorted.unsqueeze(-1).to(elem.dtype)

    out = torch.zeros((B, kp, config.CHI_ELEM_W), dtype=torch.float32, device=chi_stack.device)
    out[:, :K, :] = elem.to(torch.float32)
    mask = torch.zeros((B, kp), dtype=torch.bool, device=chi_stack.device)
    mask[:, :K] = m_sorted
    return out.reshape(B, config.CHI_ELEM_W * kp), mask
