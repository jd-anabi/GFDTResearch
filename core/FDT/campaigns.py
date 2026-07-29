"""
Simulation campaigns for FDT analysis.

Campaign 1: spontaneous fluctuations -> Welch PSD G(omega).
Campaign 2: forced response at each driving frequency -> chi(omega) via lock-in.
"""
import math
import torch
from tqdm import tqdm

from core.config import FDTConfig
from core.Simulator.nadrowski_simulator import NadrowskiSimulator
from core.Simulator.hopf_simulator import HopfSimulator
from core.Simulator.bp_simulator import BPSimulator
from core.FDT.spectral import psd_welch, lock_in_chi

# Model -> simulator class. Matches core/SBI/pipeline.py:VALID_SIMS so FDT and SBI
# stay consistent.
VALID_SIMS = {
    "nadrowski": NadrowskiSimulator,
    "hopf":      HopfSimulator,
    "bp":        BPSimulator,
}


class FDTModelError(ValueError):
    """A model/cell cannot run the FDT pipeline (missing FDT parameter, multiplicative or zero
    observable noise, or an unregistered model). Carries a user-facing message the GUI shows plainly."""

# Per-segment element budget for the solver's xs buffer. Sized to keep the per-segment
# allocation under ~800 MB at float32, matching the SBI side's CHUNK_LEN x reference
# batch (~100k steps x 2048 batch). The simulator's full `sol` tensor lives outside
# this budget; segs > 1 only shrinks the per-segment xs buffer.
FDT_MAX_ELEMENTS_PER_SEG = 200_000_000

# Fraction of currently-free CUDA memory to allocate as the per-batch budget. The
# remaining headroom absorbs PyTorch internals, fragmentation, and intermediate
# tensors during integration. 0.6 leaves ~40% buffer.
FDT_CUDA_MEM_FRACTION = 0.6


def _pick_n_segs(n_steps: int, batch_size: int) -> int:
    """Pick segs to keep the solver's per-segment xs buffer under the element budget."""
    max_steps_per_seg = max(1, FDT_MAX_ELEMENTS_PER_SEG // batch_size)
    return max(1, math.ceil(n_steps / max_steps_per_seg))


def _memory_budget_elements(device: torch.device, dtype: torch.dtype) -> int:
    """
    Per-batch element budget for the simulator's major device-resident tensors (sol + force).
    Delegates to config.memory_budget_elements, which the SBI path shares.
    """
    from core.config import memory_budget_elements
    return memory_budget_elements(device, dtype, FDT_CUDA_MEM_FRACTION)


def _plan_adaptive_batches(omegas_list: list, fpb_max: int, M: int, n_vars: int,
                            n_force: int, dt: float, burn_idx: int, n_periods: int,
                            budget_elements: int) -> list:
    """
    Walk omegas left-to-right, greedily packing as many consecutive frequencies into
    each batch as the memory budget allows (up to fpb_max).

    The per-group n_steps is determined by the LOWEST omega in the group (since that
    one needs the longest t-grid). So memory cost = (n_vars + n_force) * fpb * M * n_steps.

    :return: list of (start, count) tuples.
    """
    batches = []
    start = 0
    n = len(omegas_list)
    while start < n:
        omega_low = omegas_list[start]
        T_target = n_periods * 2.0 * math.pi / omega_low
        n_obs = int(round(T_target / dt))
        n_steps = burn_idx + n_obs

        elements_per_fpb_unit = (n_vars + n_force) * M * n_steps
        max_fpb_by_memory = max(1, budget_elements // elements_per_fpb_unit)
        actual_fpb = min(fpb_max, n - start, max_fpb_by_memory)
        batches.append((start, actual_fpb))
        start += actual_fpb
    return batches


def _make_simulator(cfg: FDTConfig, params, force, inits, t, *, freqs_per_batch=1, segs=1,
                    batch_size=1, device):
    """Build the model simulator for an FDT campaign. User models dispatch through the registry
    (a UserSimulator around the compiled torch model), mirroring pipeline.gen_obs's dispatch;
    built-ins use their VALID_SIMS class. Case-insensitive on the model name."""
    from core import registry
    if registry.is_user_model(cfg.model):
        return registry.make_user_simulator(
            registry.get(cfg.model), params, force, inits, t,
            freqs_per_batch=freqs_per_batch, segs=segs, batch_size=batch_size, device=device)
    cls = VALID_SIMS.get(cfg.model.lower())
    if cls is None:
        raise FDTModelError(f"Invalid model for FDT: {cfg.model}. Valid: {list(VALID_SIMS)}.")
    return cls(params, force, inits, t, freqs_per_batch=freqs_per_batch, segs=segs,
               batch_size=batch_size, device=device)


def observable_noise_prefactor(cfg: FDTConfig) -> float:
    """The per-model FDT normalization prefactor: T_eff/T = prefactor * omega * G / (4 * chi'').

    Physically prefactor = coupling / D_x, where D_x = g_0^2 / 2 is the diffusion coefficient of the
    OBSERVABLE (state column 0; the solver writes dx = drift*dt + g*dW*sqrt(dt)) and `coupling` is the
    coefficient the Campaign-2 drive enters the observable's drift with (1 for every model except BP).
    For Nadrowski this reduces to the historical n*beta. Raises FDTModelError when a model/cell cannot
    supply it (missing param, or a user model with multiplicative/zero observable noise)."""
    m, pd = cfg.model.lower(), cfg.params_dict
    try:
        if m == "nadrowski":       # x_noise = sqrt(2/(n*beta)) -> D_x = 1/(n*beta); coupling 1
            return float(pd["n"][0] * pd["beta"][0])
        if m == "hopf":            # x_noise = sigma_x -> D_x = sigma_x^2/2; coupling 1
            return float(2.0 / pd["sigma_x"][0] ** 2)
        if m == "bp":              # EXPERIMENTAL: drive enters as force/tau_hb (coupling 1/tau_hb),
            # x_noise = eta_hb/tau_hb -> D_x = (eta_hb/tau_hb)^2/2, so prefactor = 2*tau_hb/eta_hb^2.
            # No passive baseline exists to adjudicate the magnitude -- treat BP numbers as unverified.
            return float(2.0 * pd["tau_hb"][0] / pd["eta_hb"][0] ** 2)
    except KeyError as e:
        raise FDTModelError(f"The {cfg.model} cell is missing the FDT parameter {e}.") from e

    # User model: g = sqrt(2 D) so D_x = D_0 (the observable's white-noise strength); coupling 1.
    from core import registry
    spec = registry.get(cfg.model)
    if spec is None or spec.compiled is None:
        raise FDTModelError(f"User model '{cfg.model}' has no compiled definition for FDT.")
    c = spec.compiled
    if {str(s) for s in c.diff_exprs[0].free_symbols} & set(c.var_names):
        raise FDTModelError(f"Observable '{c.var_names[0]}' has state-dependent (multiplicative) noise; "
                            "FDT supports additive-noise observables only.")
    try:
        param_vals = [pd[name][0] for name in c.param_names]
    except KeyError as e:
        raise FDTModelError(f"The {cfg.model} cell is missing parameter {e}.") from e
    args = tuple(0.0 for _ in c.var_names) + tuple(param_vals)     # states irrelevant for additive D
    D0 = float(c.diff_fns[0](args))
    if not math.isfinite(D0) or D0 <= 0.0:
        raise FDTModelError(f"Observable '{c.var_names[0]}' has non-positive/zero noise (D0={D0}); "
                            "FDT requires a stochastic observable.")
    return 1.0 / D0


def _n_force_channels(cfg: FDTConfig) -> int:
    """Number of forcing channels the model's simulator expects; see forcing.n_force_channels, which
    is the single source of truth (shared with the SBI pipeline and the GUI's Simulate planner)."""
    from core import registry
    from core.forcing import n_force_channels
    # Resolve inits_tensor ONLY for user models (which need one channel per state variable). A
    # built-in's channel count is a property of its drift, so reading inits_tensor eagerly here would
    # break callers that never build one -- pinned by test_n_force_channels_user_vs_builtin.
    n_vars = cfg.inits_tensor.shape[1] if registry.is_user_model(cfg.model) else None
    return n_force_channels(cfg.model, cfg.force_params_dict, n_vars)


def run_campaign1_psd(cfg: FDTConfig, M: int = None, T_obs_nd: float = None,
                      nperseg: int = None, return_trajectory: bool = False):
    """
    Spontaneous-fluctuation PSD G(omega) of bundle deflection x.

    :param cfg: FDTConfig (defaults below pulled from cfg unless overridden).
    :param M: ensemble size; default cfg.ensemble_M.
    :param T_obs_nd: PSD observation duration in ND time; default cfg.psd_T_obs_nd.
    :param nperseg: Welch segment length; default min(2**14, steady-state length)
                    rounded down to nearest power of 2.
    :param return_trajectory: if True, also returns the full time axis and the
                              ensemble-mean trajectory (including burn-in) for
                              diagnostic plotting.
    :return: (omegas, G) if return_trajectory=False; otherwise
             (omegas, G, t_full, x_mean_full).
    """
    M = M if M is not None else cfg.ensemble_M
    T_obs_nd = T_obs_nd if T_obs_nd is not None else cfg.psd_T_obs_nd
    dt, burn = cfg.dt_nd, cfg.burn_in_nd
    device, dtype = cfg.hw.device, cfg.hw.dtype

    burn_idx = int(round(burn / dt))
    n_obs = int(round(T_obs_nd / dt))
    n_steps = burn_idx + n_obs
    t = torch.arange(n_steps, dtype=dtype, device=device) * dt

    n_force = _n_force_channels(cfg)
    force = torch.zeros((M, n_force, n_steps), dtype=dtype, device=device)
    inits = cfg.inits_for_M(M)
    params = cfg.params_for_M(M)

    n_segs = _pick_n_segs(n_steps, M)
    sim = _make_simulator(cfg, params, force, inits, t,
                          freqs_per_batch=1, segs=n_segs, batch_size=M, device=device)
    sol = sim.simulate(state_dep_drift=cfg.state_dep_drift)  # (n_vars, 1, M, n_steps)
    # sol[0, 0] is a VIEW, so holding it pins the WHOLE (n_vars, 1, M, n_steps) solution -- ~2.5 GB
    # for Nadrowski -- alive through psd_welch, for a diagnostic that only ever reads channel 0.
    # Materialise the one channel we need, take the ensemble mean while the buffer is still around
    # (it is the only other thing anyone wants from it), then drop the full solution.
    x_steady = sol[0, 0, :, burn_idx:].contiguous()   # (M, n_obs) post-burn-in, its own storage
    x_mean_full = sol[0, 0, :, :].mean(dim=0) if return_trajectory else None   # (n_steps,)
    del sol

    if nperseg is None:
        nperseg = min(2 ** 14, x_steady.shape[-1])
        nperseg = 1 << int(math.log2(nperseg))  # nearest power of 2 <= nperseg

    omegas, G = psd_welch(x_steady, dt=dt, nperseg=nperseg)

    if return_trajectory:
        return omegas, G, t, x_mean_full
    return omegas, G


def run_campaign2_chi(cfg: FDTConfig, omegas: torch.Tensor, M: int = None,
                      F0: float = None, freqs_per_batch: int = None,
                      show_progress: bool = True) -> torch.Tensor:
    """
    Forced-response chi(omega) by lock-in detection.

    Packs `freqs_per_batch` consecutive frequencies into each simulator call by
    using the simulator's native (n_vars, freqs_per_batch, ensemble_size, T) layout.
    Within a group, all frequencies share a single t-grid of length matching the
    longest required T_obs in the group; per-frequency lock-in then slices its own
    integer-period window from the result so leakage stays minimized.

    Trade-off: log-spaced freqs grouped by fpb=8 waste ~30% compute on the high-omega
    end of each group (their T_obs is smaller than the group max). This is dominated
    by the GPU-parallelism win when fpb > 1.

    Per group:
      - T_obs_target_k = T_obs_periods * 2*pi / omega_k for each k in group
      - n_obs_k        = round(T_obs_target_k / dt_nd)
      - n_steps_group  = burn_idx + max(n_obs_k)
      - force[k*M:(k+1)*M, 0, :] = F0 * cos(omega_k * t)
      - simulate, then per-k slice sol[0, k, :, burn_idx:burn_idx+n_obs_k].mean(dim=0)

    :param cfg: FDTConfig.
    :param omegas: (n_freqs,) driving angular frequencies.
    :param M: ensemble size; default cfg.ensemble_M.
    :param F0: forcing amplitude; default cfg.F0.
    :param freqs_per_batch: # of frequencies per simulator call; default cfg.freqs_per_batch.
    :param show_progress: tqdm bar across batch groups.
    :return: (n_freqs,) complex tensor of chi(omega_k).
    """
    M = M if M is not None else cfg.ensemble_M
    F0 = F0 if F0 is not None else cfg.F0
    fpb_max = freqs_per_batch if freqs_per_batch is not None else cfg.freqs_per_batch
    dt, burn = cfg.dt_nd, cfg.burn_in_nd
    n_periods = cfg.T_obs_periods
    device, dtype = cfg.hw.device, cfg.hw.dtype

    burn_idx = int(round(burn / dt))
    n_freqs = len(omegas)
    chis = torch.zeros(n_freqs, dtype=torch.complex128, device=device)

    # Plan batches adaptively so each fits in the memory budget. Low-omega groups
    # (long t-grid) may shrink to fpb=1; high-omega groups will use the requested fpb_max.
    omegas_list = omegas.tolist()
    n_force = _n_force_channels(cfg)
    n_vars = cfg.inits_tensor.shape[1]
    budget_elements = _memory_budget_elements(device, dtype)
    batches = _plan_adaptive_batches(omegas_list, fpb_max, M, n_vars, n_force,
                                       dt, burn_idx, n_periods, budget_elements)

    reduced = [(s, c) for s, c in batches if c < fpb_max]
    if reduced:
        print(f"Adaptive batching: {len(reduced)}/{len(batches)} groups capped below fpb_max={fpb_max} "
              f"due to memory budget. Smallest group: fpb={min(c for _, c in batches)}.")

    iterator = batches
    if show_progress:
        iterator = tqdm(batches, desc=f"Campaign 2 (chi sweep, fpb<={fpb_max})")

    for start, actual_fpb in iterator:
        omegas_batch = omegas_list[start:start + actual_fpb]

        # Per-frequency integer-period window length
        n_obs_per_freq = [int(round(n_periods * 2.0 * math.pi / w / dt)) for w in omegas_batch]
        n_obs_max = max(n_obs_per_freq)
        n_steps = burn_idx + n_obs_max

        t = torch.arange(n_steps, dtype=dtype, device=device) * dt
        batch_size = actual_fpb * M
        force = torch.zeros((batch_size, n_force, n_steps), dtype=dtype, device=device)
        for k, omega in enumerate(omegas_batch):
            # Drive only channel 0 (bundle position); other channels stay zero
            force[k * M:(k + 1) * M, 0, :] = F0 * torch.cos(omega * t)

        inits = cfg.inits_tensor.expand(batch_size, -1).contiguous()
        params = cfg.params_tensor.expand(batch_size, -1).contiguous()

        n_segs = _pick_n_segs(n_steps, batch_size)
        sim = _make_simulator(cfg, params, force, inits, t,
                              freqs_per_batch=actual_fpb, segs=n_segs,
                              batch_size=batch_size, device=device)
        sol = sim.simulate(state_dep_drift=cfg.state_dep_drift)  # (n_vars, actual_fpb, M, n_steps)

        for k, omega in enumerate(omegas_batch):
            n_obs_k = n_obs_per_freq[k]
            x_mean = sol[0, k, :, burn_idx : burn_idx + n_obs_k].mean(dim=0)   # (n_obs_k,)
            t_slice = t[burn_idx : burn_idx + n_obs_k]
            T_obs_used = n_obs_k * dt
            chis[start + k] = lock_in_chi(t_slice, x_mean, omega, F0, T_obs_used)

        # Free fragmented allocations between batches; sol/force/t go out of scope here.
        del sol, force, t, sim, inits, params
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return chis
