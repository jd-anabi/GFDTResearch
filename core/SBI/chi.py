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


def n_chi_features(n_freqs: int | None = None) -> int:
    """Width of the chi(omega) conditioning block: 3 features (log|chi|, cos, sin) per frequency."""
    k = config.CHI_N_FREQS if n_freqs is None else n_freqs
    return 3 * k


def chi_labels(n_freqs: int | None = None) -> list[str]:
    """Ordered feature labels for the chi block, matching chi_features()'s [log|chi|, cos, sin] packing."""
    k = config.CHI_N_FREQS if n_freqs is None else n_freqs
    labels: list[str] = []
    for i in range(k):
        labels += [f"chi{i}_logmag", f"chi{i}_cos", f"chi{i}_sin"]
    return labels


# Default-K labels, analogous to statistics.FEATURE_LABELS. Recompute via chi_labels(K) for a custom K.
CHI_LABELS = chi_labels()


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
        CHI_FREQ_BOUNDS starts at 0.1*Omega_0, so those are exactly the probes chi-mode exists for.
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


def chi_features(chi_stack: torch.Tensor) -> torch.Tensor:
    """
    Pack a (B, K) complex susceptibility curve into (B, 3K) real features, ordered per frequency as
    [log|chi_k|, cos(arg chi_k), sin(arg chi_k)] -- log for the (positive, unbounded) magnitude and a
    (cos, sin) pair for the phase (no 2pi wrap), matching the statistics.py conventions. Output width
    equals len(chi_labels(K)). NaN/Inf -> 0 for a clean conditioning vector.

    :param chi_stack: (B, K) complex susceptibilities.
    :return: (B, 3K) float32 features.
    """
    mag = chi_stack.abs()
    ang = torch.angle(chi_stack)
    logmag = torch.log(torch.clamp(mag, min=_EPS))
    feats = torch.stack([logmag, torch.cos(ang), torch.sin(ang)], dim=-1)     # (B, K, 3)
    out = feats.reshape(feats.shape[0], -1)                                   # (B, 3K), per-freq blocks
    return torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
