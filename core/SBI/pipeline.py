import math
import warnings

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
    seg = min(n_fine, CHUNK_LEN)
    n_keep = n_out * max(0, n_fine - steady_idx)                  # per sample, held until we return
    per_chunk_sample = n_vars * n_fine + n_ch * n_fine + max(n_vars * seg, n_keep)
    if per_chunk_sample <= 0:
        return batch_size
    budget = config.memory_budget_elements(device, dtype, _SIM_MEM_FRACTION)
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
        outs = []
        for s in range(0, batch_size, max_b):               # plain range: the tqdm nest is already 4 deep
            e = min(s + max_b, batch_size)
            outs.append(_gen_obs_one(
                model, params[s:e], t, inits[s:e],
                force[s:e] if force.shape[0] == batch_size else force,
                n_segs, steady_idx, fixed_dict, state_dep_drift, e - s, var_idx, dtype, device))
        return torch.cat(outs, dim=1)                      # dim 1 is the batch: (n_out, batch, T')
    return _gen_obs_one(model, params, t, inits, force, n_segs, steady_idx, fixed_dict,
                        state_dep_drift, batch_size, var_idx, dtype, device)


def _gen_obs_one(model, params, t, inits, force, n_segs, steady_idx, fixed_dict,
                 state_dep_drift, batch_size, var_idx, dtype, device):
    """One un-split simulation batch. Split planning lives in gen_obs; this is the body."""
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
    obs = sol[sel, 0, :, steady_idx:].clone()
    del sol
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
            f"chi: {dropped}/{B * K} probes masked (below {config.CHI_MIN_CYCLES} drive cycles, "
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
                      n_vars: int | None = None,
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

    training_data = []
    thetas = []

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

    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
    oversample = 3
    valid_t_scales, valid_Ts = _draw_and_filter(n_runs * oversample)
    # Fallback: keep drawing more candidates until we have enough valid ones. A whole draw coming
    # back empty means NO (t_scale, T) in the declared bounds fits the grid, so redrawing would spin
    # forever -- say what is wrong instead of hanging.
    while valid_t_scales.shape[0] < n_runs:
        more_t_scales, more_Ts = _draw_and_filter(n_runs * oversample)
        if more_t_scales.numel() == 0:
            raise ValueError(
                f"No (t_scale, T) pair in the declared bounds fits the fine-grid ceiling of "
                f"{n_fine_max} steps (steady_idx={steady_idx}, dt_exp={dt_exp}, "
                f"dt_nd_min={dt_nd_min}, t_scale in {t_scale_bounds}, T in "
                f"[{t_min_exp}, {t_max_exp}]). Shorten the recording range, widen t_scale, or raise "
                f"N_ND_MAX / the model's t_nd_max.")
        valid_t_scales = torch.cat([valid_t_scales, more_t_scales])
        valid_Ts = torch.cat([valid_Ts, more_Ts])
    batch_t_scales = valid_t_scales[:n_runs]
    batch_Ts = valid_Ts[:n_runs]

    with torch.no_grad():
        for batch_k in tqdm(range(n_runs), desc="Generating training data", leave=False):
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

            x_scale  = curr_thetas_rescale[:, rescale_idx["x_scale"]].unsqueeze(1)
            x_offset = curr_thetas_rescale[:, rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in rescale_idx else 0.0

            if chi_mode:
                # chi(omega) mode: spontaneous run (Groups A-F + Omega_0) + K single-tone forced runs.
                # Conditioning [S(41, Group G zeroed) | log(T) | chi(3K)] -- chi replaces the forcing block.
                force0 = torch.zeros((run_size, n_force_ch, t_fine.shape[0]), dtype=dtype, device=device)
                x_nd_spont_fine = gen_obs(
                    model=model, params=curr_thetas_nd, t=t_fine, inits=inits,
                    force=force0, n_segs=n_segs_k, steady_idx=steady_idx,
                    fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                    batch_size=run_size, var_idx=0, dtype=dtype, device=device,
                )[0, :, :]
                x_spont_dim = helpers.rescale(
                    x_nd_spont_fine[:, ::subsample_factor][:, :N_points_k], x_scale, x_offset)
                del x_nd_spont_fine, force0
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    training_stats = gen_stats(x_spont_dim.cpu(), None, dt_exp, None, None, None,
                                               device=device, spontaneous_only=True)   # (run, 41), G zeroed
                # --- THE PROBE SET FOR THIS BATCH ---
                # K is drawn per batch and the placement is stratified-jittered, so the encoder is
                # trained ACROSS probe counts and frequencies rather than memorising one grid. A fixed
                # grid would leave `u` taking only K distinct values in the entire training set, and an
                # experimentalist's 0.07x recording would then be an out-of-distribution input to an
                # MLP that extrapolates linearly and confidently.
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
                chi_block, chi_mask = gen_chi_block(
                    model, curr_thetas_nd, curr_thetas_rescale, x_spont_dim, t_fine, inits, rescale_idx,
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
                log_T_k_tensor = torch.full((run_size, 1), math.log(T_k), dtype=dtype)
                training_stats = torch.cat((training_stats, log_T_k_tensor, chi_block.cpu()), dim=-1)
                training_data.append(training_stats)
                del x_spont_dim
            elif spontaneous_only:
                # No drive: one spontaneous run (Groups A-F; Group G is zero-padded), no forcing block.
                force = torch.zeros((run_size, n_force_ch, t_fine.shape[0]), dtype=dtype, device=device)
                x_nd_spont_fine = gen_obs(
                    model=model, params=curr_thetas_nd, t=t_fine, inits=inits,
                    force=force, n_segs=n_segs_k, steady_idx=steady_idx,
                    fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                    batch_size=run_size, var_idx=0, dtype=dtype, device=device,
                )[0, :, :]
                # Rescale STRAIGHT off the strided view: helpers.rescale materialises a fresh
                # contiguous tensor, so nothing keeps a reference to the fine buffer and the `del`
                # genuinely releases it. Binding the view to a name first (as this used to) pins the
                # whole (run_size, n_fine) storage until that name dies -- the `del` frees nothing.
                x_spont_dim = helpers.rescale(
                    x_nd_spont_fine[:, ::subsample_factor][:, :N_points_k], x_scale, x_offset)
                del x_nd_spont_fine, force
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    training_stats = gen_stats(x_spont_dim.cpu(), None, dt_exp, None, None, None,
                                               device=device, spontaneous_only=True)
                    log_T_k_tensor = torch.full((run_size, 1), math.log(T_k), dtype=dtype)
                    # Conditioning [S | log(T)] -- no forcing block (forcing_dim = 0).
                    training_stats = torch.cat((training_stats, log_T_k_tensor), dim=-1)
                    training_data.append(training_stats)
            else:
                # 2. Build nondimensional force tensor at fine resolution (uses PHYSICAL rescale)
                force = build_nondim_sin_force_tensor(
                    curr_thetas_forcing, t_fine, curr_thetas_rescale, forcing_idx, rescale_idx
                )

                # 3. Simulate the FORCED run (drive on) -> Group G
                x_nd_fine = gen_obs(
                    model=model, params=curr_thetas_nd, t=t_fine, inits=inits,
                    force=force, n_segs=n_segs_k, steady_idx=steady_idx,
                    fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                    batch_size=run_size, var_idx=0, dtype=dtype, device=device,
                )[0, :, :]
                # 4a. Redimensionalize the forced run IMMEDIATELY (uses PHYSICAL rescale).
                # Order matters: helpers.rescale materialises a fresh contiguous tensor, so the `del`
                # below actually releases the fine buffer. Holding the strided VIEW in a name instead
                # (as this used to) pinned the entire (run_size, n_fine) storage right across the
                # second gen_obs call below -- two full fine trajectories resident where one
                # subsampled slice was needed, several GB at run_size=2048. That also made
                # _max_sim_batch split batches it did not need to, and k chunks cost k x wall-clock.
                x_dim = helpers.rescale(
                    x_nd_fine[:, ::subsample_factor][:, :N_points_k], x_scale, x_offset)
                del x_nd_fine

                # 3b. Simulate the SPONTANEOUS run (zero force) -> Groups A-F
                x_nd_spont_fine = gen_obs(
                    model=model, params=curr_thetas_nd, t=t_fine, inits=inits,
                    force=torch.zeros_like(force), n_segs=n_segs_k, steady_idx=steady_idx,
                    fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                    batch_size=run_size, var_idx=0, dtype=dtype, device=device,
                )[0, :, :]
                # 4b. Same treatment for the spontaneous run.
                x_spont_dim = helpers.rescale(
                    x_nd_spont_fine[:, ::subsample_factor][:, :N_points_k], x_scale, x_offset)
                del x_nd_spont_fine, force

                # 5. Stats (A-F from spontaneous, G from forced) + conditioning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    drive_amp = curr_thetas_forcing[:, forcing_idx["amp"]].cpu()
                    drive_freq = curr_thetas_forcing[:, forcing_idx["freq"]].cpu()
                    drive_phase = curr_thetas_forcing[:, forcing_idx["phase"]].cpu()
                    training_stats = gen_stats(x_spont_dim.cpu(), x_dim.cpu(), dt_exp, drive_amp, drive_freq, drive_phase, device=device)
                    log_T_k_tensor = torch.full((run_size, 1), math.log(T_k), dtype=dtype)
                    # Canonical conditioning layout: [S(x_dim) | log(T) | theta_force].
                    # log(T) rides with the summary pathway; theta_force is a separate block.
                    # The embedding split in build_posterior depends on this exact order, so
                    # keep it in sync with generate_observations / validate / infer_from_experiment.
                    training_stats = torch.cat((training_stats, log_T_k_tensor, curr_thetas_forcing.cpu()), dim=-1)
                    training_data.append(training_stats)

            # 6. Collect LATENT targets (not physical)
            thetas.append(curr_thetas_latent.cpu())
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

    training_data_tensor = torch.cat(training_data, dim=0)
    thetas_tensor = torch.cat(thetas, dim=0)
    return training_data_tensor, thetas_tensor

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

        infer.append_simulations(thetas, data, proposal=proposal)
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