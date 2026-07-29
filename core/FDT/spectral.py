"""
Spectral primitives for FDT analysis.

All accumulations are done in float64 to avoid roundoff error at the low-ω end
where Campaign-2 lock-in sums span ~10^5 timesteps x 256 ensemble members.
"""
import math
import torch


def psd_welch(x: torch.Tensor, dt: float, nperseg: int, overlap: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One-sided PSD G(omega) of an ensemble of trajectories, scipy.signal.welch-compatible.

    The normalization matches scipy's default (variance/Hz), with the frequency
    axis converted to angular omega = 2*pi*f for direct use in FDT formulas. With the
    convention used in this pipeline (see fdt_pipeline docstring), G(omega) here
    equals 2*C(omega) for omega > 0, where C(omega) is the user's two-sided correlator
    C(omega) = integral d(tau) exp(-i*omega*tau) <X(tau) X(0)>.

    :param x: (M, T) ensemble of trajectories, M independent realizations.
    :param dt: timestep (in same units as the trajectory's time axis).
    :param nperseg: segment length (samples). Power of 2 recommended.
    :param overlap: fractional segment overlap. SciPy default is 0.5.
    :return: (omegas, G) -- omegas shape (nperseg//2 + 1,), G one-sided PSD shape (nperseg//2 + 1,).
    """
    # NOT `x = x.to(torch.float64)`. Promoting the WHOLE ensemble up front doubles a tensor that is
    # ~1.6 GB at Campaign-1 defaults, when only the (M, nperseg) slice inside the loop is ever
    # arithmetic -- ~33 MB. FDT runs on host RAM (cpu_device()), so that was 1.6 GB of resident
    # working set for a diagnostic. The promotion moves onto `seg` below; every accumulation,
    # the window and the output stay float64, so the result is unchanged.
    M, T_total = x.shape
    step = max(1, int(nperseg * (1 - overlap)))
    n_segs = 1 + (T_total - nperseg) // step
    if n_segs < 1:
        raise ValueError(f"trajectory length {T_total} < nperseg {nperseg}; cannot compute PSD")

    window = torch.hann_window(nperseg, dtype=torch.float64, device=x.device)
    win_norm = (window ** 2).sum()   # scipy's "S2"

    psd_accum = torch.zeros(nperseg // 2 + 1, dtype=torch.float64, device=x.device)
    for i in range(n_segs):
        seg = x[:, i * step : i * step + nperseg].to(torch.float64)   # promote the SLICE, not x
        seg = seg - seg.mean(dim=-1, keepdim=True)   # detrend each segment
        seg = seg * window
        X = torch.fft.rfft(seg, dim=-1)              # (M, nperseg//2 + 1)
        psd_accum += (X.abs() ** 2).mean(dim=0)      # ensemble-average per segment
    psd_accum /= n_segs

    # SciPy-equivalent one-sided density: Sxx(f) = 2 |X|^2 * dt / S2
    G = 2.0 * psd_accum * dt / win_norm
    G[0] /= 2                          # DC not doubled
    if nperseg % 2 == 0:
        G[-1] /= 2                     # Nyquist (only for even nperseg) not doubled

    freqs_hz = torch.fft.rfftfreq(nperseg, dt).to(x.device, dtype=torch.float64)
    omegas = 2.0 * math.pi * freqs_hz
    return omegas, G


def lock_in_chi(t: torch.Tensor, x_mean: torch.Tensor, omega: float, F0: float, T_obs: float) -> torch.Tensor:
    """
    Complex susceptibility chi(omega) by lock-in detection.

        chi(omega) = (2 / (F0 * T_obs)) * sum_n <X>(t_n) * exp(i*omega*t_n) * dt

    Subtracts the mean of x_mean to remove residual DC drift (forcing in
    Campaign-2 is zero-mean, so any non-zero mean is a transient/burn-in
    residue or numerical artifact).

    :param t: (T,) time axis, already trimmed by burn-in. MUST be aligned with x_mean.
    :param x_mean: (T,) ensemble-mean response, already trimmed by burn-in.
    :param omega: drive angular frequency.
    :param F0: drive amplitude (same units as x_mean / drive).
    :param T_obs: observation duration covered by t (in same time units).
    :return: complex scalar tensor (complex128).
    """
    t = t.to(torch.float64)
    x = x_mean.to(torch.float64)
    x = x - x.mean()

    dt = (t[-1] - t[0]) / (t.shape[0] - 1)

    phase = omega * t
    e_iwt = torch.complex(torch.cos(phase), torch.sin(phase))
    chi = (2.0 / (F0 * T_obs)) * (x.to(torch.complex128) * e_iwt).sum() * dt
    return chi


def eff_temp_ratio(G: torch.Tensor, chi_imag: torch.Tensor, omega: torch.Tensor, prefactor: float) -> torch.Tensor:
    """
    T_eff / T = prefactor * omega * G(omega) / (4 * chi''(omega))   -- one-sided PSD convention.

    ``prefactor`` is the per-model normalization coupling/D_x (the inverse observable diffusion, times
    the drive coupling); it is ``n * beta`` for Nadrowski. See
    ``core.FDT.campaigns.observable_noise_prefactor``. Vectorized over omega; same shape as omega.
    """
    return prefactor * omega * G / (4.0 * chi_imag)


def gen_freqs_log(omega_0: float, n: int, bounds: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Log-spaced angular-frequency grid in [bounds[0]*omega_0, bounds[1]*omega_0].

    :param omega_0: characteristic / natural angular frequency.
    :param n: number of grid points.
    :param bounds: (lo_mult, hi_mult); both multiplied by omega_0 to set the span.
    """
    lo = math.log(bounds[0] * omega_0)
    hi = math.log(bounds[1] * omega_0)
    return torch.exp(torch.linspace(lo, hi, n, device=device, dtype=dtype))


def find_spectral_peak(omegas: torch.Tensor, G: torch.Tensor,
                        search_band: tuple = None) -> float:
    """
    Locate the angular frequency at the PSD peak.

    Useful for identifying the spontaneous-oscillation frequency from Campaign 1
    in a model-agnostic way (works for any simulator without needing the linearized
    Jacobian). Bracket the search to avoid the DC bin or non-resonant noise-floor
    artifacts.

    :param omegas: (n,) angular frequencies from psd_welch.
    :param G: (n,) one-sided PSD values from psd_welch.
    :param search_band: optional (omega_lo, omega_hi) to restrict the search.
                        Defaults to skipping the DC bin only.
    :return: angular frequency at the peak (float).
    """
    if search_band is not None:
        mask = (omegas >= search_band[0]) & (omegas <= search_band[1])
        omegas_s, G_s = omegas[mask], G[mask]
    else:
        omegas_s, G_s = omegas[1:], G[1:]   # skip DC
    return float(omegas_s[G_s.argmax()])
