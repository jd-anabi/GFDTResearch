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
#   FISHER        4 channels x K probes     -> decorrelate.feats, scripts/degeneracy_map
#   (the diagnostic labels in scripts/_common are the FISHER set minus the 11 Group-G columns)
#
# `u` and `mask` are deliberately ABSENT from the Fisher set. `u = log(f_k/f_peak)` is theta-INDEPENDENT
# wherever the Fisher probes a deterministic multiplier grid, so across an ensemble it takes ~two
# distinct float32 values with std ~2.5e-8 -- and `fnoise = max(std, 1e-9)` does NOT protect against
# that. The central difference then writes entries of order 1, the magnitude of a real standardized
# feature, into up to K x P cells of the Jacobian that defines the flow's coordinate system, while V
# stays orthogonal to 1e-4 and every existing test passes.
CHI_COND_CHANNELS = ("u", "logmag", "cos", "sin", "logcyc", "mask")
CHI_FISHER_CHANNELS = ("logmag", "cos", "sin", "logcyc")


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


def lock_in_batched(x: torch.Tensor, omega: torch.Tensor, F0, T_obs: float, dt: float,
                    chunk: int = _LOCK_IN_CHUNK) -> torch.Tensor:
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
      * the mean is over the FULL trace, computed in pass 1. Demeaning per chunk would be a high-pass
        filter with corner ~1/(chunk*dt), which preferentially eats the LOW-multiplier probes -- and
        CHI_FREQ_BOUNDS is now entirely SUB-RESONANCE (0.03 .. 0.3 * Omega_0, measured: everything
        above ~0.25x is irreproducible), so EVERY probe is a low-multiplier one. This invariant went
        from "protects the probes chi-mode exists for" to "protects all of them".
      * the phase in a chunk starting at s uses the ABSOLUTE times arange(s, e)*dt, never
        arange(0, e-s)*dt. (The wrong form is normally an order-unity error, but it is invisible when
        omega*chunk*dt happens to be a multiple of 2*pi -- so test with incommensurate omega.)

    :param x: (B, n) response traces (physical), sampled at dt.
    :param omega: (B,) drive ANGULAR frequencies (cell freq units, rad/cell-time).
    :param F0: drive amplitude the response was driven with -- a scalar OR a (B,) per-sample tensor
               (chi is drive-amplitude-independent in the linear regime, but F0 must MATCH the drive
               so the units are physical).
    :param T_obs: observation duration covered by the trace (scalar, cell time units).
    :param dt: sampling interval (scalar, cell time units).
    :param chunk: time-chunk length. Memory only -- numerically irrelevant.
    :return: (B,) complex128 susceptibilities.
    """
    B, n = x.shape[0], x.shape[-1]
    dev, dt = x.device, float(dt)
    step = max(1, int(chunk))

    # pass 1: the float64 row mean over the WHOLE trace (see invariant 1 above)
    acc = torch.zeros(B, dtype=torch.float64, device=dev)
    for s in range(0, n, step):
        acc += x[..., s:min(s + step, n)].to(torch.float64).sum(dim=-1)
    mean = (acc / n if n > 0 else acc).unsqueeze(-1)                          # (B, 1)

    # pass 2: separate real / imaginary lock-in sums, one time chunk at a time
    w = omega.to(torch.float64).reshape(-1, 1)                                # (B, 1)
    re = torch.zeros(B, dtype=torch.float64, device=dev)
    im = torch.zeros(B, dtype=torch.float64, device=dev)
    for s in range(0, n, step):
        e = min(s + step, n)
        t = torch.arange(s, e, device=dev, dtype=torch.float64) * dt          # ABSOLUTE times
        phase = w * t.unsqueeze(0)                                            # (B, e-s)
        xc = x[..., s:e].to(torch.float64) - mean
        re += (xc * torch.cos(phase)).sum(dim=-1)
        im += (xc * torch.sin(phase)).sum(dim=-1)

    F0 = F0.to(torch.float64).reshape(-1) if torch.is_tensor(F0) else float(F0)
    return torch.complex(re, im) * (2.0 / (F0 * float(T_obs))) * dt


def _mag_phase(chi_stack: torch.Tensor):
    """(log|chi|, cos arg, sin arg) from a complex stack. log for the positive, unbounded magnitude;
    a (cos, sin) pair for the phase so there is no 2pi wrap -- the statistics.py convention."""
    ang = torch.angle(chi_stack)
    return torch.log(torch.clamp(chi_stack.abs(), min=_EPS)), torch.cos(ang), torch.sin(ang)


def fisher_features(chi_stack: torch.Tensor, logcyc: torch.Tensor) -> torch.Tensor:
    """(B, K) complex + (B, K) logcyc -> (B, 4K) float32: [log|chi|, cos, sin, logcyc] per probe.

    The FISHER set: no pad slots, no mask, and NO frequency channel -- see CHI_FISHER_CHANNELS for why
    `u` must not appear here. `logcyc` is kept because it genuinely varies with theta (through
    f_peak), so it carries signal rather than float32 rounding.
    """
    logmag, cos, sin = _mag_phase(chi_stack)
    feats = torch.stack([logmag, cos, sin, logcyc.to(logmag.dtype)], dim=-1)   # (B, K, 4)
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
