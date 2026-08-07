"""
Pipeline orchestration for the SBI pipeline.

No input() calls live here -- all user interaction is delegated to cli.py.
This module owns the pipeline flow: observe -> prior -> posterior -> validate.
"""
import hashlib
import importlib
import math
import time
import warnings

import torch
import numpy as np
from matplotlib import pyplot as plt
from sbi.analysis import pairplot, sbc_rank_plot, plot_tarp
from sbi.diagnostics import run_sbc, check_sbc, run_tarp, check_tarp
from sbi.inference import DirectPosterior
from torch.distributions import Distribution, MixtureSameFamily
from tqdm import tqdm

from .config import (
    SimConfig, PRIOR_PATH, POSTERIOR_PATH, PLOT_PATH, T_MIN_EXP_S, T_MAX_EXP_S,
    CHUNK_LEN, N_ND_MAX, PPC_BIN_SIZE, SBC_N_CAL, STABILITY_SWEEP_ND_UNITS, TRAINING_NUM_RUNS,
    DENSITY_ESTIMATOR, NSF_HIDDEN_FEATURES, NSF_NUM_TRANSFORMS, NSF_NUM_BINS,
    TRAINING_NUM_ROUNDS, TRAINING_BATCH_SIZE, TRAINING_LEARNING_RATE,
    TRAINING_STOP_AFTER_EPOCHS, TRAINING_MAX_NUM_EPOCHS, TRAINING_SHOW_SUMMARY, FORCING_SI_UNITS,
    EYE_TEST_CYCLES,
)
from . import cli, config, forcing
from .Helpers import helpers, visualizers, file_manager, labels
from .SBI import embedded_network, pipeline, analysis, decorrelate, chi, overlay
from .SBI.Priors import sbi_prior_wrapper
from .SBI.reparam import (
    build_inferred_bijection, TransformedPosterior, build_rescale_bijection,
    build_rotated_bijection, RotatedLatentPrior, OrthogonalTransform, load_eval_bijection,
    nd_log_mask, resolved_log_params, read_sidecar, posterior_mode as reparam_posterior_mode,
)

# Directories have spaces in their names, so use importlib for these imports
_scaling_mod = importlib.import_module("core.SBI.Priors.Scaling Priors.scaling_prior")
ScalingPrior = _scaling_mod.ScalingPrior

_forcing_mod = importlib.import_module("core.SBI.Priors.Forcing Priors.forcing_prior")
ForcingPrior = _forcing_mod.ForcingPrior

_product_mod = importlib.import_module("core.SBI.Priors.Product Prior.product_prior")
ProductPrior = _product_mod.ProductPrior


# ── Pipeline entry point ────────────────────────────────────────────────────
def run(cfg: SimConfig):
    """
    Execute the SBI pipeline:
      1. Build or load the prior (ND x rescale x forcing product prior).
      2. Train or load the posterior (amortized NPE — ground-truth-free).
      3. Calibration diagnostics (SBC + expected coverage) — no chosen observation needed.
      4. Optionally infer on a chosen observation: a simulated cell (ground truth), experimental
         data, or neither. Only this step shows observation-dependent plots (GT trace, corner, PPC,
         eye test).
    """
    # 1. Prior
    prior_choice, build_new = cli.select_or_build_prior()
    inf_prior, force_prior = build_prior(cfg, prior_choice, build_new)

    # 2. Posterior (training is amortized and observation-independent)
    pos_choice, train_new = cli.select_or_train_posterior()
    posterior, pos_diagnostics = build_posterior(cfg, inf_prior, force_prior, pos_choice, train_new)
    helpers.clear_screen()

    # 3. Calibration (data-free): SBC + expected coverage
    validate_calibration(cfg, posterior, inf_prior, force_prior)

    # 4. Optional inference on a chosen observation
    mode = cli.select_inference_mode()
    if mode == "simulated":
        cell_file = cli.select_cell_file()
        ignored = cli.load_and_validate_gt(cfg, cell_file)   # inject GT + inits, validated vs bounds
        if ignored:
            print(f"Note: the bounds file does not declare {', '.join(ignored)} — those cell values "
                  f"were ignored.")
        for msg in check_observation_in_distribution(cfg, inf_prior, force_prior):
            warnings.warn(msg, stacklevel=2)
        T_obs_s = cli.get_time_params()
        cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
        if T_obs_s < T_MIN_EXP_S:
            warnings.warn(
                f"T_obs={T_obs_s:.2f}s is below the training range minimum T_MIN_EXP_S="
                f"{T_MIN_EXP_S:.2f}s; the posterior may extrapolate poorly.", stacklevel=2)
        elif T_obs_s > T_MAX_EXP_S:
            warnings.warn(
                f"T_obs={T_obs_s:.2f}s exceeds the training range maximum T_MAX_EXP_S="
                f"{T_MAX_EXP_S:.2f}s; the posterior may extrapolate poorly.", stacklevel=2)
        x_dim, obs_stats, t_dim = generate_observations(cfg)
        visualizers.plot(
            t_dim.squeeze(0).cpu().detach().numpy(),
            x_dim[0, :].cpu().detach().numpy(),
            title="Ground-truth trace",
            labels=(labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)),
        )
        infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True)
    elif mode == "experimental" and cfg.chi_mode:
        # chi(omega): one passive recording + K single-tone forced recordings (one per relative freq).
        spont_path, forced_paths, T_obs_s, F0_si = cli.get_inference_inputs_chi()
        X_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
        X_forced = [file_manager.load_experimental_data(p, dtype=cfg.hw.dtype) for p in forced_paths]
        obs_stats, obs_data, t_dim = build_experiment_obs_chi(cfg, X_spont, X_forced, T_obs_s, F0_si)
        infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False)
    elif mode == "experimental" and not cfg.has_forcing:
        # Passive recording: a single unforced trace, no drive.
        path, T_obs_s = cli.get_inference_inputs_spontaneous()
        X_obs = file_manager.load_experimental_data(path, dtype=cfg.hw.dtype)
        obs_stats, obs_data, t_dim = build_experiment_obs_spontaneous(cfg, X_obs, T_obs_s)
        infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False)
    elif mode == "experimental":
        spont_path, forced_path, T_obs_s, forcing_params_si = cli.get_inference_inputs(
            list(cfg.force_params_dict.keys()))
        X_obs_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
        X_obs_forced = file_manager.load_experimental_data(forced_path, dtype=cfg.hw.dtype)
        obs_stats, obs_data, t_dim = build_experiment_obs(
            cfg, X_obs_spont, X_obs_forced, T_obs_s, forcing_params_si)
        infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False)
    # mode == "none": stop after calibration


# ── Step 1: Synthetic data ──────────────────────────────────────────────────
def generate_observations(cfg: SimConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate a ground-truth observation matching experimental conditions.

    Simulates at fine ND resolution (dt_nd_min, stable for EM), then downsamples
    to match the physical sampling rate dt_exp and duration T_obs — exactly
    mirroring what the training loop produces.

    :return: (obs_data, obs_stats, t_dim) where obs_data has shape (1, N_obs),
             obs_stats has shape (1, n_stats + n_forcing + 1), and t_dim is the
             dimensional time vector.
    """
    t = cfg.t  # full pre-simulated ND time vector at dt_nd_min

    # Ground-truth rescale and forcing params as (1, n) tensors
    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=cfg.hw.dtype, device=cfg.hw.device)
    rescale_gt = torch.tensor([[val for val, _ in cfg.rescale_params.values()]], dtype=cfg.hw.dtype, device=cfg.hw.device)

    # Ground-truth t_scale for this observation
    t_scale_gt = rescale_gt[:, cfg.rescale_idx["t_scale"]].item()

    # Compute ND quantities for this observation (same logic as training loop)
    dt_nd_gt = cfg.dt_exp / t_scale_gt
    T_nd_obs = cfg.T_obs / t_scale_gt
    subsample_factor = max(1, round(dt_nd_gt / cfg.dt_nd_min))
    N_obs = int(T_nd_obs / dt_nd_gt)

    # Fine-resolution time vector: transient + enough to downsample into N_obs points
    n_fine_total = cfg.steady_idx + N_obs * subsample_factor

    # OOD warning: NN was only trained on combinations with n_fine_total <= N_ND_MAX
    if n_fine_total > N_ND_MAX:
        warnings.warn(
            f"Synthetic GT observation out-of-distribution: n_fine_total={n_fine_total} "
            f"> N_ND_MAX={N_ND_MAX}. Network was trained only on combinations with "
            f"n_fine_total <= {N_ND_MAX}. Posterior may extrapolate poorly.",
            stacklevel=2,
        )

    # Cost ceiling: if simulation exceeds the pre-simulated grid, clip and update T_obs
    # so that log(T_obs) conditioning matches the actual trajectory length downstream.
    if n_fine_total > len(t):
        N_obs = (len(t) - cfg.steady_idx) // subsample_factor
        n_fine_total = cfg.steady_idx + N_obs * subsample_factor
        actual_T_obs = N_obs * cfg.dt_exp
        warnings.warn(
            f"Observation cost ceiling hit: requested T_obs={cfg.T_obs:.4f} exceeds "
            f"pre-simulated grid. Clipping N_obs to {N_obs} (actual T_obs={actual_T_obs:.4f}). "
            f"cfg.T_obs updated so downstream code sees the consistent value.",
            stacklevel=2,
        )
        cfg.T_obs = actual_T_obs  # keep log(T) conditioning consistent across pipeline

    # Publish the RESOLVED length (post-clipping) so downstream stages use this exact count instead
    # of re-deriving it. infer_and_visualize used to recompute int(cfg.T_obs / cfg.dt_exp), which is
    # algebraically the same but can differ by one sample -- and after the clip above it re-truncates
    # cfg.T_obs = N_obs*dt_exp back to N_obs-1. The observation and the PPC traces then had different
    # widths, and _emit_overlay_figures' shape guard dropped all five overlay figures without a word.
    cfg.n_obs = N_obs

    t_fine = t[:n_fine_total]
    n_vars = cfg.inits_tensor.shape[-1]

    # Auto-derive n_segs based on CHUNK_LEN (per-chunk memory cap)
    n_segs_gt = max(1, math.ceil(n_fine_total / CHUNK_LEN))

    x_scale = rescale_gt[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
    x_offset = rescale_gt[:, cfg.rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in cfg.rescale_idx else 0.0
    t_offset = rescale_gt[:, cfg.rescale_idx["t_offset"]].item() if "t_offset" in cfg.rescale_idx else 0.0

    def _spont_run(force_tensor):
        x_fine = pipeline.gen_obs(
            model=cfg.model, params=cfg.params_tensor, t=t_fine, inits=cfg.inits_tensor,
            force=force_tensor, n_segs=n_segs_gt, steady_idx=cfg.steady_idx,
            state_dep_drift=cfg.state_dep_drift, var_idx=0,
            dtype=cfg.hw.dtype, device=cfg.hw.device,
        )[0, :, :]
        return x_fine[:, ::subsample_factor][:, :N_obs]

    if cfg.has_forcing and not cfg.chi_mode:
        force = pipeline.build_nondim_sin_force_tensor(forcing_gt, t_fine, rescale_gt, cfg.forcing_idx, cfg.rescale_idx)
        x_nd = _spont_run(force)                                 # forced run -> Group G
        x_nd_spont = _spont_run(torch.zeros_like(force))         # spontaneous -> Groups A-F
        x_dim = helpers.rescale(x_nd, x_scale, x_offset)
        x_spont_dim = helpers.rescale(x_nd_spont, x_scale, x_offset)
        del x_nd, x_nd_spont, force
    else:
        # No single-frequency drive (spontaneous, or chi-mode which drives its own K-freq probe):
        # a single spontaneous run; the base observation IS the passive trace.
        zero_force = torch.zeros((1, forcing.n_force_channels(cfg.model, cfg.forcing_idx, n_vars),
                                  n_fine_total), dtype=cfg.hw.dtype, device=cfg.hw.device)
        x_spont_dim = helpers.rescale(_spont_run(zero_force), x_scale, x_offset)
        x_dim = x_spont_dim                                      # the observation IS the passive trace

    # Dimensional time vector for plotting, in SECONDS (N_obs points at dt_exp spacing). t_dim is
    # display-only (never fed to gen_stats), so converting cell-time-units -> s here makes every
    # downstream trace plot seconds without per-site conversion.
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")   # cell time unit (e.g. ms) -> seconds
    t_dim = (torch.arange(N_obs, dtype=cfg.hw.dtype) * cfg.dt_exp + t_offset) * s_per_cell
    t_dim = t_dim.unsqueeze(0)  # (1, N_obs), seconds

    # Summary statistics + conditioning vector. Layout: [S | log(T) | forcing]; log(T) is grouped
    # with the summary pathway. Keep this order in sync with gen_training_data and build_posterior.
    log_T_obs = torch.tensor([[math.log(cfg.T_obs)]], dtype=cfg.hw.dtype)
    if cfg.chi_mode:
        # [S(41, Group G zeroed) | log(T) | padded probe SET] -- probes at mult_k * Omega_0.
        # An OBSERVATION uses the deterministic grid, not the training sampler's jitter: this is a
        # specific measurement, and the PPC has to be able to reproduce its exact drive frequencies.
        obs_stats = pipeline.gen_stats(x_spont_dim, None, cfg.dt_exp, None, None, None,
                                       device=cfg.hw.device, spontaneous_only=True)
        obs_mults = chi.chi_multipliers_for(cfg)
        chi_block, _chi_mask = pipeline.gen_chi_block(
            cfg.model, cfg.params_tensor, rescale_gt, x_spont_dim, t_fine, cfg.inits_tensor,
            cfg.rescale_idx, n_segs_gt, cfg.steady_idx, subsample_factor, N_obs, cfg.dt_exp,
            obs_mults, cfg.chi_f0, k_pad=cfg.chi_k_pad, bounds=cfg.chi_freq_bounds,
            max_cycles=cfg.chi_max_cycles,
            state_dep_drift=cfg.state_dep_drift, dtype=cfg.hw.dtype, device=cfg.hw.device)
        # Record the ABSOLUTE probe frequencies this observation was measured at, so the PPC drives
        # the same experiment rather than re-deriving frequencies from each posterior sample's own
        # f_peak -- which would simulate a different experiment and make the PPC agree for the wrong
        # reason.
        cfg.chi_obs_freqs = (obs_mults.to(cfg.hw.device)
                             * chi.peak_freq(x_spont_dim, cfg.dt_exp).median()).detach()
        obs_stats = torch.cat([obs_stats, log_T_obs, chi_block.cpu()], dim=-1)
    elif cfg.has_forcing:
        obs_stats = pipeline.gen_stats(
            x_spont_dim, x_dim, cfg.dt_exp,
            forcing_gt[:, cfg.forcing_idx["amp"]], forcing_gt[:, cfg.forcing_idx["freq"]],
            forcing_gt[:, cfg.forcing_idx["phase"]], device=cfg.hw.device,
        )
        obs_stats = torch.cat([obs_stats, log_T_obs, forcing_gt.cpu()], dim=-1)
    else:
        obs_stats = pipeline.gen_stats(x_spont_dim, None, cfg.dt_exp, None, None, None,
                                       device=cfg.hw.device, spontaneous_only=True)
        obs_stats = torch.cat([obs_stats, log_T_obs], dim=-1)
    return x_dim, obs_stats, t_dim


# ── Step 2: Prior construction ──────────────────────────────────────────────
def _find_nd_gmm(obj, _depth: int = 0):
    """The latent ND MixtureSameFamily inside any of the prior wrappers, or None.

    The same GMM is reachable by several paths depending on how it was built -- ProductPrior ->
    TransformedDistribution -> base_dist (the PHYSICAL prior), SBIPriorWrapper -> ProductPrior
    (the posterior's stored TRAINING prior), and RotatedLatentPrior around either -- so walk the
    known containers rather than hard-coding one route that silently returns None for the others.
    """
    if isinstance(obj, torch.distributions.MixtureSameFamily):
        return obj
    if obj is None or _depth > 5:
        return None
    for attr in ("gen_dist", "base_dist", "base"):
        found = _find_nd_gmm(getattr(obj, attr, None), _depth + 1)
        if found is not None:
            return found
    for seq in ("distributions", "transforms"):
        for item in (getattr(obj, seq, None) or []):
            found = _find_nd_gmm(item, _depth + 1)
            if found is not None:
                return found
    return None


def _gmm_fingerprint(obj) -> str | None:
    """Stable digest of an ND prior's GMM (component means + weights), or None if not found.

    Identifies WHICH prior an artifact was built from. Component count alone is not enough -- two
    runs over the same box produce different fits -- and the means are latent, so they cannot be
    eyeballed. The digest is over float64 bytes, so it is exact rather than tolerance-based: this
    answers "is this the same prior object", not "are these priors similar".
    """
    gmm = _find_nd_gmm(obj)
    if gmm is None:
        return None
    means = gmm.component_distribution.loc.detach().cpu().to(torch.float64).contiguous()
    weights = gmm.mixture_distribution.probs.detach().cpu().to(torch.float64).contiguous()
    h = hashlib.sha256()
    h.update(means.numpy().tobytes())
    h.update(weights.numpy().tobytes())
    return h.hexdigest()[:16]


def _assert_prior_used_matches_posterior(posterior, inferred_prior, what: str) -> None:
    """Refuse to run a posterior against a prior it was not trained with.

    SBC draws theta* from the TRAINING prior; run against a different one it is not a calibration
    measurement of this posterior at all, just a plot. The posterior carries its own training prior
    (sbi stores it on the DirectPosterior), so this needs no sidecar and works for a posterior that
    was trained moments ago and never saved. Unverifiable on either side => silence, not a false
    alarm: legacy posteriors and hand-built stand-in priors both land there.
    """
    latent = getattr(posterior, "latent", posterior)
    trained = _gmm_fingerprint(getattr(latent, "prior", None))
    supplied = _gmm_fingerprint(inferred_prior)
    if trained is None or supplied is None or trained == supplied:
        return
    raise ValueError(
        f"{what}: the prior supplied is not the one this posterior was trained with "
        f"(prior {supplied} vs posterior's {trained}). Load the prior that belongs to this "
        f"posterior -- results computed against a different prior describe neither.")


def _assert_prior_matches(cfg: SimConfig, path: str, choice: str) -> None:
    """Fail LOUDLY when a saved ND prior does not belong to this config.

    The latent GMM is fit in its box's OWN coordinate, so a prior is meaningful only against the
    exact (model, parameter set + ORDER, box) it was built for. None of that was checked here, and
    the consequences are silent rather than loud: the box edges rescale every sample the flow is
    trained on, and a reordered parameter set mis-binds columns positionally. The one guard that did
    exist lives in ``build_posterior`` and covers only the log-mask.

    Legacy priors carry no ``model``/``param_keys``; those WARN rather than raise, because the box
    comparison below is still exact and is the part that actually rescales the samples.
    """
    meta = file_manager.read_prior_metadata(path)
    if not meta:
        return                                    # pre-reparam file: nothing recorded to check

    def _bad(what, got, want):
        raise ValueError(
            f"Prior '{choice}' does not match this configuration: {what} differs.\n"
            f"  prior:  {got}\n  config: {want}\n"
            f"A prior's GMM is fit in its own box coordinate, so loading it here would train the "
            f"flow against a different distribution than the one the samples came from. Build a new "
            f"prior for this bounds file, or pick the prior that belongs to it.")

    if "model" in meta and str(meta["model"]) != cfg.model:
        _bad("the model", meta["model"], cfg.model)
    keys = list(cfg.params_dict.keys())
    if "param_keys" in meta and list(meta["param_keys"]) != keys:
        _bad("the ND parameter set or ORDER", list(meta["param_keys"]), keys)
    if "lows" in meta and "highs" in meta:
        want_lo = torch.tensor([b[0] for _, b in cfg.params_dict.values()], dtype=torch.float64)
        want_hi = torch.tensor([b[1] for _, b in cfg.params_dict.values()], dtype=torch.float64)
        got_lo = meta["lows"].detach().cpu().to(torch.float64)
        got_hi = meta["highs"].detach().cpu().to(torch.float64)
        if got_lo.shape != want_lo.shape:
            _bad("the ND parameter COUNT", tuple(got_lo.shape), tuple(want_lo.shape))
        if not (torch.allclose(got_lo, want_lo) and torch.allclose(got_hi, want_hi)):
            diff = [f"{n}: prior ({lo:g}, {hi:g}) vs config ({wl:g}, {wh:g})"
                    for n, lo, hi, wl, wh in zip(keys, got_lo.tolist(), got_hi.tolist(),
                                                 want_lo.tolist(), want_hi.tolist())
                    if lo != wl or hi != wh]
            _bad("the ND box", "; ".join(diff), "the bounds file in use")
    if "model" not in meta or "param_keys" not in meta:
        warnings.warn(
            f"Prior '{choice}' predates model/param_keys recording, so only its box could be "
            f"verified. Re-save it to make it fully self-describing.", stacklevel=2)


def build_prior(cfg: SimConfig, choice: str | None, build_new: bool,
                *, save: bool = True, save_name: str | None = None, fig_sink=None) -> tuple[Distribution, Distribution]:
    """
    Load an existing prior from disk, or construct a new product prior:
        ProductPrior = ND parameter prior x rescaling prior x forcing prior

    :param cfg: Pipeline configuration.
    :param choice: Filename of a saved prior, or None to build from scratch.
    :param build_new: True to construct from scratch.
    :param save: When building new, persist the ND prior (+corner PNG). Defaults True (CLI behavior).
                 Pass False to defer saving (e.g. a GUI that saves via an explicit control).
    :param save_name: Name to save under; when None (and save=True) the CLI prompt is used.
    :param fig_sink: Optional (title, fig) -> None display callback for the corner plot; None => plt.show().
    :return: A Distribution that can be sampled and scored.
    """
    # User-model guard: the bounds ND section order MUST equal the compiled param order (torch.unbind
    # binds columns positionally). A hand-edited JSON over a stale Bounds file would mis-bind silently.
    from core import registry
    if registry.is_user_model(cfg.model):
        spec = registry.get(cfg.model)
        expected = list(spec.compiled.param_names)
        actual = list(cfg.params_dict.keys())
        if actual != expected:
            raise ValueError(
                f"Model '{cfg.model}' is out of sync with its bounds file: definition uses {expected}, "
                f"bounds file lists {actual}. Re-save the model from the Settings model builder.")

    # 1. Forcing prior
    force_prior = _build_forcing_prior(cfg)

    # 2. Rescaling prior
    rescale_prior = _build_rescale_prior(cfg)

    if not build_new and choice is not None:
        _assert_prior_matches(cfg, str(PRIOR_PATH / choice), choice)
        nd_prior = file_manager.load_mix_dist(str(PRIOR_PATH / choice), device=cfg.hw.device)
        visualizers.visualize_dist(nd_prior, labels=cfg.labels, title="Prior (loaded)", sink=fig_sink)
        nd_dim = len(cfg.params_dict)
        rescale_dim = len(cfg.rescale_params)
        inferred_prior = ProductPrior(
            distributions=[nd_prior, rescale_prior],
            dims=[nd_dim, rescale_dim],
        )
        return inferred_prior, force_prior

    # --- Build from scratch ---
    print("No prior found. Going to construct prior from scratch.")
    time.sleep(5)
    helpers.clear_screen()

    # 3. ND parameter prior (stability-filtered GMM)
    # Stability is a per-parameter property — screen on a short fixed-length trajectory
    # (STABILITY_SWEEP_ND_UNITS) rather than the full master grid. Global sweep uses
    # half this (t_global_scale=2 inside gen_prior), local sweep uses the full t_stab.
    n_stab_fine = int(STABILITY_SWEEP_ND_UNITS / cfg.dt_nd_min)
    t_stab = cfg.t[:n_stab_fine]
    prior_segs = max(1, math.ceil(n_stab_fine / CHUNK_LEN))
    nd_prior = pipeline.gen_prior(
        model=cfg.model, t=t_stab,
        global_batch_size=cfg.hw.batch_size,
        local_batch_size=(cfg.hw.batch_size // 2),
        segs=prior_segs,
        prior_bounds=cfg.nd_params_bounds,
        state_dep_drift=cfg.state_dep_drift,
        num_iterations=50,
        log_mask=nd_log_mask(cfg),   # geometric/log box on the configured ND scale params (REPARAM_LOG_PARAMS)
        dtype=cfg.hw.dtype, device=cfg.hw.device,
    )

    # Save the ND prior (GMM) with the existing serializer -- or defer (GUI) and just display it.
    if save:
        nd_name = save_name if save_name is not None else cli.prompt_save_name("ND parameter prior")
        save_prior_artifacts(nd_name, nd_prior, cfg, fig_sink=fig_sink)
    else:
        visualizers.visualize_dist(nd_prior, labels=cfg.labels, title="Prior", sink=fig_sink)

    # 4. Compose into product prior
    nd_dim = len(cfg.params_dict)
    rescale_dim = len(cfg.rescale_params)

    inferred_prior = ProductPrior(
        distributions=[nd_prior, rescale_prior],
        dims=[nd_dim, rescale_dim],
    )

    return inferred_prior, force_prior


def _build_rescale_prior(cfg: SimConfig) -> Distribution:
    """
    Construct the rescaling-parameter prior from cell file bounds.

    Scale parameters (names containing 'scale') use log-uniform — they're positive
    and span orders of magnitude, so uniform would over-weight the high end.
    Offset parameters use uniform — they can be negative or zero.
    """
    bounds = [row[1] for row in cfg.rescale_params.values()]
    types = tuple(
        "log-uni" if "scale" in name else "uniform"
        for name in cfg.rescale_params.keys()
    )
    scaling = ScalingPrior(cfg.hw.dtype, cfg.hw.device)
    return scaling.construct_prior(bounds, types)


def _build_forcing_prior(cfg: SimConfig) -> Distribution:
    """
    Construct the forcing-parameter prior from cell file bounds.

    'freq' uses log-uniform — hair bundle resonances span decades of Hz, and uniform
    over-weights the high end. All other forcing params (amp, phase, offset) use
    uniform — amp bound can include 0 (log-uniform would fail), phase is a bounded
    angle, offset can be negative.
    """
    if not cfg.has_forcing:
        return None                                   # no drive -> no forcing prior (spontaneous model)
    bounds = [row[1] for row in cfg.force_params_dict.values()]
    types = tuple(
        "log-uni" if name == "freq" else "uniform"
        for name in cfg.force_params_dict.keys()
    )
    forcing = ForcingPrior(cfg.hw.dtype, cfg.hw.device)
    return forcing.construct_prior(bounds, types)


# ── Step 3: Posterior construction ──────────────────────────────────────────
def build_posterior(
    cfg: SimConfig,
    prior: Distribution,                 # physical inferred prior from build_prior
    force_prior: Distribution,
    choice: str | None,
    train_new: bool,
    *, save: bool = True, save_name: str | None = None, fig_sink=None,
) -> tuple[TransformedPosterior, dict | None]:
    """
    Load an existing latent DirectPosterior from disk and wrap with T, or train a new one
    via NPE in latent space. Returns a TransformedPosterior whose .sample/.log_prob operate
    in physical-parameter coordinates for downstream code.

    :param save: When training new, persist <name>.pt / .rot.pt / .loss.npz / _loss.png. Defaults
                 True (CLI behavior). Pass False to defer saving (GUI saves via an explicit control).
    :param save_name: Name to save under; when None (and save=True) the CLI prompt is used.
    :param fig_sink: Optional (title, fig) -> None display callback for the training-loss curve
                     (a GUI embeds it); None keeps the CLI behavior (loss saved to PNG, not shown).
    """
    T = build_inferred_bijection(cfg)

    if not train_new and choice is not None:
        # map_location rehomes every stored tensor onto this machine's device, so a posterior trained
        # on a CUDA box (e.g. a Windows GPU) loads on a CPU/MPS-only Mac instead of raising
        # "Attempting to deserialize object on a CUDA device". sbi caches the training device in two
        # scalar attributes it does NOT refresh on load, so repoint both: .device drives sampling and
        # ._device drives log_prob (sbi DirectPosterior.log_prob builds tensors on ._device).
        posterior_latent = torch.load(str(POSTERIOR_PATH / choice),
                                      map_location=cfg.hw.device, weights_only=False)
        assert isinstance(posterior_latent, DirectPosterior)
        posterior_latent.device = posterior_latent._device = cfg.hw.device
        _assert_mode_matches(cfg, posterior_latent, choice)
        # Reconstruct the exact training box (+ rotation) from the <name>.rot.pt sidecar — log-mask
        # and V are self-describing, so eval is correct regardless of the current config (single
        # source of truth shared with the offline diagnostic scripts).
        T_load = load_eval_bijection(cfg, choice, POSTERIOR_PATH)
        return TransformedPosterior(posterior_latent, T_load), None

    # --- Build a LATENT product prior for SBI to train on ---
    # Physical prior layout: ProductPrior([nd_prior_physical, rescale_prior_physical]).
    # Extract latent ND (the MixtureSameFamily inside the TransformedDistribution):
    nd_prior_physical      = prior.distributions[0]      # TransformedDistribution(latent_gmm, T_nd)
    if not isinstance(nd_prior_physical, torch.distributions.TransformedDistribution):
        raise ValueError(
            "Loaded ND prior is not a TransformedDistribution — it was saved with the pre-reparameterization "
            "pipeline. Regenerate the prior with the current `gen_prior` before training a new posterior."
        )
    rescale_prior_physical = prior.distributions[1]      # MultipleIndependent
    latent_nd = nd_prior_physical.base_dist              # the raw latent MixtureSameFamily

    # The latent ND GMM was fit in its box's coordinate. If we now train with a different ND log
    # box (REPARAM_LOG_PARAMS changed since this prior was built), physical training samples would
    # be drawn from the wrong prior. Require the loaded prior's box mask to match the config mask.
    from torch.distributions.transforms import ComposeTransform as _Compose
    from .SBI.reparam import UnitToBoxTransform as _Box
    _nd_box = next((inner for tr in nd_prior_physical.transforms
                    for inner in (tr.parts if isinstance(tr, _Compose) else [tr])
                    if isinstance(inner, _Box)), None)
    if _nd_box is not None:
        _want = nd_log_mask(cfg).to(_nd_box.log_mask.device)
        if not torch.equal(_nd_box.log_mask, _want):
            raise ValueError(
                "Loaded ND prior's log-box mask does not match config.REPARAM_LOG_PARAMS "
                f"(prior log dims={_nd_box.log_mask.tolist()}, config wants={_want.tolist()}). "
                "The latent GMM was fit in a different coordinate — REBUILD the ND prior "
                "(construct a new prior) before training a new posterior."
            )

    # Pushforward the physical rescale prior through T_rescale.inv (Issue 2a).
    T_rescale = build_rescale_bijection(cfg)
    latent_rescale = torch.distributions.TransformedDistribution(rescale_prior_physical, T_rescale.inv)

    latent_inferred_prior = ProductPrior(
        distributions=[latent_nd, latent_rescale],
        dims=[len(cfg.params_dict), len(cfg.rescale_params)],
    )

    # Optional decorrelating reparameterization (Track A): rotate the flow's latent coordinate
    # into the simulation-based Fisher eigenbasis so the well-identified-but-correlated posterior
    # is axis-aligned and the flow can calibrate it. REPARAM_ROTATE=False => V=None => plain.
    # The Fisher rotation probes a representative drive (decorrelate reads forcing_idx["amp"/…]); a
    # no-forcing model has no such params, so rotation is disabled for it. V=None is the plain pipeline.
    # Read the flag off the CONFIG, not the module: `from .config import REPARAM_ROTATE` snapshots at
    # import, so a GUI toggle could never have taken effect. Works in ALL THREE observation modes --
    # decorrelate.feats builds its Jacobian over whichever feature set the mode conditions on.
    #
    # chi mode used to be excluded here, because "chi(omega) already attacks the degeneracy the
    # rotation targets". That was never measured, and it is false: on the master cell k~x_scale is
    # 0.98 forced vs 0.95 chi (scripts/degeneracy_map.py, 2026-08-05), i.e. chi leaves the dominant
    # alias essentially intact while improving nearly everything else. The rotation exists for that
    # alias, so chi gets one too. Cost note: the Fisher pays (1 + K) simulations per evaluation in chi
    # mode instead of 2, so a rotation costs ~(K+1)/2 x what it does in forced mode -- REPARAM_FISHER_M
    # and REPARAM_FISHER_POINTS are the knobs if that is too slow.
    rotate = cfg.reparam_rotate
    if rotate:
        print("Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)...")
        # Average the Fisher over the prior (not just GT) so the linear rotation is valid prior-wide.
        # GT-free: the rotation anchors on the prior median with a representative drive (force_prior).
        V = decorrelate.build_latent_fisher_rotation(
            cfg, T, latent_prior=latent_inferred_prior, force_prior=force_prior)
        T_train = build_rotated_bijection(T, V)
        train_prior = RotatedLatentPrior(latent_inferred_prior, V)
    else:
        V, T_train, train_prior = None, T, latent_inferred_prior

    training_params = {
        "model": cfg.model,
        "prior": train_prior,                         # <-- latent (rotated if REPARAM_ROTATE)
        "t": cfg.t,
        "run_size": cfg.hw.batch_size,
        "num_runs": TRAINING_NUM_RUNS,
        "steady_idx": cfg.steady_idx,
        "dt_nd_min": cfg.dt_nd_min,
        "dt_exp": cfg.dt_exp,
        "t_min_exp": cfg.t_min_exp,
        "t_max_exp": cfg.t_max_exp,
        "t_scale_bounds": cfg.t_scale_bounds,
        "state_dep_drift": cfg.state_dep_drift,
        "spontaneous_only": not cfg.has_forcing,
        "chi_mode": cfg.chi_mode,
        # No "chi_n_freqs" here on purpose. It is the count an OBSERVATION supplies; training draws K
        # per batch over [CHI_K_MIN_TRAIN, chi_k_pad] and subsets again per row, which is what makes
        # one posterior serve any probe count. It used to be threaded in and silently ignored -- an
        # invitation to "fix" gen_training_data into honouring it and destroy exactly that property.
        "chi_f0": cfg.chi_f0,
        "chi_freq_bounds": cfg.chi_freq_bounds,
        "chi_k_pad": cfg.chi_k_pad,
        "chi_max_cycles": cfg.chi_max_cycles,
        # _observation_inits, NOT cfg.inits_tensor: training is ground-truth-free, so a config built from
        # bounds alone (no cell loaded) has an empty inits_dict and cfg.inits_tensor would RAISE. The
        # fallback synthesizes the same model-default inits the training loop itself uses.
        "n_vars": _observation_inits(cfg).shape[-1],
        "dtype": cfg.hw.dtype,
        "device": cfg.hw.device,
    }

    # Conditioning layout is [S(x) | log(T) | forcing]. log(T) rides with the summary
    # pathway, so input_dim (the leading summary block) includes it; only the forcing
    # params form the separate forcing pathway.
    # chi-mode routes the padded probe SET through the EmbeddedNet's second pathway in place of the
    # single-frequency forcing block, as a permutation-invariant set encoder.
    forcing_dim = expected_forcing_dim(cfg)        # shared with the sidecar + the load-side mode guard
    from .SBI.statistics import FEATURE_LABELS
    input_dim = len(FEATURE_LABELS) + 1            # n_summary_stats + log(T); observation-independent

    embedded_net = build_embedding_net(cfg, input_dim, forcing_dim)

    sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(train_prior)

    # Training is amortized (TRAINING_NUM_ROUNDS=1) and observation-independent: x_obs/theta_obs only
    # feed training-time diagnostics, so we pass None — no ground-truth observation needed to train.
    theta_obs_latent = None

    posterior_latent, pos_diagnostics = pipeline.train_nn(
        training_params, model=DENSITY_ESTIMATOR, prior=sbi_prior,
        embedding_net=embedded_net, forcing_prior=force_prior,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        x_obs=None, theta_obs=theta_obs_latent, num_rounds=TRAINING_NUM_ROUNDS,
        return_diagnostics=True,
        theta_transform=T_train,
        hidden_features=NSF_HIDDEN_FEATURES, num_transforms=NSF_NUM_TRANSFORMS, num_bins=NSF_NUM_BINS,
        learning_rate=TRAINING_LEARNING_RATE, stop_after_epochs=TRAINING_STOP_AFTER_EPOCHS,
        max_num_epochs=TRAINING_MAX_NUM_EPOCHS, show_train_summary=TRAINING_SHOW_SUMMARY,
        batch_size=TRAINING_BATCH_SIZE, device=cfg.hw.device,
    )

    if save:
        name = save_name if save_name is not None else cli.prompt_save_name("posterior")
        save_posterior_artifacts(name, posterior_latent, V, pos_diagnostics, cfg)

    # Display the training-loss curve (a GUI embeds it via the sink; the CLI historically saved it to
    # PNG without showing, so with no sink we do nothing here to preserve that behavior).
    if fig_sink is not None and pos_diagnostics is not None and pos_diagnostics.get("validation_loss"):
        fig_loss = visualizers.plot_training_loss(pos_diagnostics)
        if fig_loss is not None:
            fig_sink("Training loss", fig_loss)

    assert isinstance(posterior_latent, DirectPosterior)
    return TransformedPosterior(posterior_latent, T_train), pos_diagnostics


def save_prior_artifacts(name: str, nd_prior, cfg: SimConfig, *, fig_sink=None) -> None:
    """
    Persist an ND prior GMM to Resources/Priors/<name>.pt and its corner PNG to Resources/Plots.
    Shared by build_prior (CLI, save=True) and a GUI's explicit "Save prior" control. With no
    fig_sink the corner plot falls back to plt.show() (a no-op under the GUI's Agg backend).
    """
    # model + the ND parameter ORDER travel with the file so _assert_prior_matches can refuse a
    # cross-config load. Without them a prior is identifiable only by its box edges, which several
    # cells happened to share.
    file_manager.save_mix_dist(nd_prior, str(PRIOR_PATH / (name + ".pt")),
                               model=cfg.model, param_keys=list(cfg.params_dict.keys()))
    visualizers.visualize_dist(nd_prior, labels=cfg.labels,
                               save_path=str(PLOT_PATH / (name + ".png")), title="Prior", sink=fig_sink)


def expected_forcing_dim(cfg: SimConfig) -> int:
    """Width of the conditioning vector's forcing/chi block for this config. Single source of truth,
    shared by build_posterior's EmbeddedNet, the save-side sidecar and the load-side mode guard.

    Under chi this is a function of the PAD, not of the probe count -- which is the one line that buys
    K-agnosticism. A posterior trained with K drawn over 2..K_PAD loads against a config declaring any
    other probe count with NO width guard loosened anywhere.
    """
    return chi.n_chi_features(cfg.chi_k_pad) if cfg.chi_mode else len(cfg.force_params_dict)


def build_embedding_net(cfg: SimConfig, input_dim: int = None, forcing_dim: int = None):
    """The ONE construction site for the conditioning network.

    The sizing arithmetic used to be duplicated in two scripts as well as here, so a layout change
    had three places to be wrong in. Under chi the forcing pathway's hidden dims are CONSTANTS
    (the set encoder's geometry is independent of the pad, or no two pads could share a checkpoint);
    everything else keeps the original forcing_dim-derived sizing byte-for-byte.
    """
    from .SBI.statistics import FEATURE_LABELS
    input_dim = (len(FEATURE_LABELS) + 1) if input_dim is None else input_dim
    forcing_dim = expected_forcing_dim(cfg) if forcing_dim is None else forcing_dim
    if cfg.chi_mode:
        return embedded_network.EmbeddedNet(
            input_dim, 3 * input_dim // 2, (5 * input_dim // 2, 2 * input_dim),
            forcing_dim=forcing_dim,
            forcing_layer_dims=(config.CHI_PHI_DIM, config.CHI_SET_OUT),
            merge_layer_dim=2 * input_dim,
            chi_k_pad=cfg.chi_k_pad, chi_band=cfg.chi_freq_bounds,
        )
    return embedded_network.EmbeddedNet(
        input_dim, 3 * input_dim // 2, (5 * input_dim // 2, 2 * input_dim),
        forcing_dim=forcing_dim,
        forcing_layer_dims=(forcing_dim * 4, forcing_dim * 2),
        merge_layer_dim=2 * input_dim,
    )


def _assert_mode_matches(cfg: SimConfig, posterior_latent, choice: str) -> None:
    """
    Fail LOUDLY and IMMEDIATELY when a saved posterior's observation mode disagrees with this config.

    Without this the mismatch surfaced as a raw matrix-shape RuntimeError from inside EmbeddedNet's
    first Linear -- but only at the FIRST SAMPLE, i.e. after an entire calibration set had already
    been simulated. The three conditioning widths (42 / 42+n_f / 42+6*K_PAD) cannot collide, so the
    check is exact; it just needs to happen before the simulation spend rather than after it.
    """
    sidecar = read_sidecar(choice, POSTERIOR_PATH, map_location=cfg.hw.device)

    # LAYOUT GATE FIRST -- ahead of the decode below, whose `except ValueError: warn; return` would
    # otherwise let a layout-1 posterior through on a decode failure. Keyed on the SIDECAR's own mode,
    # not cfg's: a forced posterior loaded against a chi config must be told it is forced, not that it
    # was "trained under chi layout 1".
    sc_mode = (sidecar or {}).get("mode")
    if sc_mode == "chi" or cfg.observation_mode == "chi":
        if sc_mode == "chi":
            got_layout = (sidecar or {}).get("chi_layout")
            if got_layout != config.CHI_LAYOUT:
                raise ValueError(
                    f"Posterior '{choice}' was trained under chi layout {got_layout or 1} -- the "
                    f"retired fixed-3K grid, where the probe's frequency was implied by its slot "
                    f"index. This build writes layout {config.CHI_LAYOUT} (a padded probe set, "
                    f"{config.CHI_ELEM_W} channels per slot, frequency carried explicitly). The two "
                    f"are not interchangeable and their widths can collide exactly "
                    f"(6*5 == 3*10 == 30), so this cannot be auto-detected. Retrain.")
            for key, want, what in (("chi_k_pad", cfg.chi_k_pad, "probe-slot capacity"),
                                    ("chi_elem_w", config.CHI_ELEM_W, "channels per slot")):
                got = (sidecar or {}).get(key)
                if got is not None and int(got) != int(want):
                    raise ValueError(
                        f"Posterior '{choice}' has {what} {got}, but this config declares {want}. "
                        f"It is frozen into the trained network's input shape, so retrain or set "
                        f"{key} back to {got}.")
            got_band = (sidecar or {}).get("chi_freq_bounds")
            if got_band is not None and tuple(got_band) != tuple(cfg.chi_freq_bounds):
                raise ValueError(
                    f"Posterior '{choice}' was trained over chi band {tuple(got_band)}, but this "
                    f"config declares {tuple(cfg.chi_freq_bounds)}. The band fixes the encoder's "
                    f"frequency normalization and is baked into its weights.")
            got_cyc = (sidecar or {}).get("chi_max_cycles")
            if got_cyc is not None and abs(float(got_cyc) - float(cfg.chi_max_cycles)) > 1e-9:
                raise ValueError(
                    f"Posterior '{choice}' was trained with a {float(got_cyc):g}-cycle lock-in "
                    f"ceiling, but this config declares {float(cfg.chi_max_cycles):g}. The ceiling "
                    f"decides how much of each recording is integrated, so the same bench data "
                    f"produces different |chi| AND a different logcyc under the two -- and logcyc is "
                    f"how the encoder weighs a probe. Set chi_max_cycles back to {float(got_cyc):g}, "
                    f"or retrain.")

    try:
        mode, forcing_dim, k = reparam_posterior_mode(posterior_latent, sidecar)
    except ValueError as e:                                  # undecodable: warn, do not block a load
        warnings.warn(f"Could not verify the observation mode of '{choice}': {e}", stacklevel=2)
        return
    # Identity checks first: mode + width agreeing says only that the conditioning vectors are the
    # same SHAPE, which several different configs satisfy. The model, the parameter ORDER and the
    # training box are what make a posterior's numbers mean anything, and none of them were checked
    # -- which is how a posterior trained on one cell's bounds was evaluated against another's.
    if sidecar:
        if sidecar.get("model") is not None and str(sidecar["model"]) != cfg.model:
            raise ValueError(
                f"Posterior '{choice}' was trained for model {sidecar['model']}, but this config is "
                f"for {cfg.model}.")
        want_keys = list(cfg.params_dict.keys()) + list(cfg.rescale_params.keys())
        got_keys = list(sidecar.get("param_keys") or [])
        if got_keys and got_keys != want_keys:
            raise ValueError(
                f"Posterior '{choice}' was trained over a different inferred parameter set or ORDER.\n"
                f"  posterior: {got_keys}\n  config:    {want_keys}\n"
                f"Columns bind positionally, so every reported value would refer to the wrong "
                f"parameter. Pick the bounds file this posterior was trained with.")

    want_mode, want_dim = cfg.observation_mode, expected_forcing_dim(cfg)
    if mode == want_mode and forcing_dim == want_dim:
        return
    from .SBI.statistics import FEATURE_LABELS
    summary_w = len(FEATURE_LABELS) + 1
    detail = f" (K={k})" if k else ""
    raise ValueError(
        f"Posterior '{choice}' was trained in {mode.upper()} mode{detail} with a forcing/chi block of "
        f"{forcing_dim} features, but this config is {want_mode.upper()} mode expecting {want_dim}. "
        f"Conditioning widths {summary_w + forcing_dim} vs {summary_w + want_dim} are incompatible. "
        f"Pick a posterior trained in this mode, or rebuild the config to match "
        f"(the chi toggle and the bounds file's Forcing section are what select the mode)."
    )


def save_posterior_artifacts(name: str, posterior_latent, V, diagnostics: dict | None, cfg: SimConfig) -> None:
    """
    Persist a trained posterior and its companions: <name>.pt (raw latent DirectPosterior), the
    <name>.rot.pt reparam sidecar (rotation V + log params, when either is active), and the
    <name>.loss.npz curve + <name>_loss.png. Shared by build_posterior (CLI) and a GUI's explicit
    "Save posterior" control.
    """
    torch.save(posterior_latent, str(POSTERIOR_PATH / (name + ".pt")))
    # Self-describing sidecar so eval reconstructs the exact training box (log-mask + rotation V) AND
    # knows which observation mode produced this posterior.
    #
    # Written UNCONDITIONALLY. It used to be skipped when V was None and no log params were active --
    # which is exactly the chi case (chi is deliberately unrotated, and REPARAM_LOG_PARAMS is []), so
    # a multi-hour chi posterior landed on disk BYTE-INDISTINGUISHABLE from the legacy forced
    # posteriors sitting beside it, with nothing on the load path checking width or mode. A missing
    # sidecar still means "pre-reparam, linear box" for the old artifacts; from here on, absence is
    # only ever a legacy signal, never an ambiguous new one.
    log_params_used = resolved_log_params(cfg)
    from .SBI.statistics import FEATURE_LABELS
    torch.save({
        "V": V,
        "log_params": log_params_used,
        # Observation mode + conditioning geometry -- see SBI/reparam.posterior_mode, which prefers
        # these over decoding the trained net (that decoding cannot distinguish chi at K=2 from a
        # hypothetical 6-parameter drive).
        "mode": cfg.observation_mode,
        "input_dim": len(FEATURE_LABELS) + 1,
        "forcing_dim": expected_forcing_dim(cfg),
        # LAYOUT is what the load path gates on. Width cannot be trusted to identify it: 6*K_PAD at
        # K_PAD=5 is exactly 30, an exact collision with the retired layout-1 3*K at K=10.
        "chi_layout": config.CHI_LAYOUT if cfg.chi_mode else None,
        "chi_k_pad": cfg.chi_k_pad if cfg.chi_mode else None,
        "chi_elem_w": config.CHI_ELEM_W if cfg.chi_mode else None,
        # A TRAINING RECORD only -- never read as "the K this posterior needs". That is the payoff.
        "chi_n_freqs": cfg.chi_n_freqs if cfg.chi_mode else None,
        "chi_f0": cfg.chi_f0 if cfg.chi_mode else None,
        "chi_freq_bounds": tuple(cfg.chi_freq_bounds) if cfg.chi_mode else None,
        # The lock-in duration ceiling. Recorded for the same reason as the band: it sets the logcyc
        # a given recording reports, and logcyc is the channel the encoder uses to decide how much to
        # trust a probe. Evaluating at a different ceiling feeds it a value the training set never
        # contained, on the one channel whose job is calibration.
        "chi_max_cycles": float(cfg.chi_max_cycles) if cfg.chi_mode else None,
        # Parameter ORDER is load-bearing (simulators bind columns positionally), so record it.
        "param_keys": list(cfg.params_dict.keys()) + list(cfg.rescale_params.keys()),
        # THE TRAINING BOX. The flow learns a density over the LATENT coordinate, so the box is what
        # turns its output back into physical parameters -- and eval used to rebuild that box from
        # whatever config happened to be loaded rather than from the posterior. Two configs sharing a
        # mode and a conditioning width therefore looked interchangeable while decoding the same
        # latent sample to different physical values. Recorded here, load_eval_bijection can
        # reconstruct the box the flow was actually trained in.
        "model": cfg.model,
        "nd_lows": torch.tensor([b[0] for _, b in cfg.params_dict.values()], dtype=torch.float64),
        "nd_highs": torch.tensor([b[1] for _, b in cfg.params_dict.values()], dtype=torch.float64),
        "rescale_lows": torch.tensor([b[0] for _, b in cfg.rescale_params.values()], dtype=torch.float64),
        "rescale_highs": torch.tensor([b[1] for _, b in cfg.rescale_params.values()], dtype=torch.float64),
    }, str(POSTERIOR_PATH / (name + ".rot.pt")))
    # Loss curve: persisted so the convergence check is reproducible (sbi keeps it only in the trainer).
    if diagnostics is not None and diagnostics.get("validation_loss"):
        np.savez(
            str(PLOT_PATH / (name + ".loss.npz")),
            training_loss=np.asarray(diagnostics.get("training_loss", []), dtype=float),
            validation_loss=np.asarray(diagnostics.get("validation_loss", []), dtype=float),
            best_validation_loss=float(diagnostics.get("best_validation_loss") or float("nan")),
            epochs_trained=int(diagnostics.get("epochs_trained") or -1),
            stop_after_epochs=int(diagnostics.get("stop_after_epochs") or -1),
        )
        fig_loss = visualizers.plot_training_loss(diagnostics, save_path=str(PLOT_PATH / (name + "_loss.png")))
        if fig_loss is not None:
            plt.close(fig_loss)


def check_observation_in_distribution(cfg: SimConfig, inferred_prior, force_prior,
                                      n_samples: int = 2000,
                                      lo_pct: float = 1.0, hi_pct: float = 99.0) -> list:
    """Warn when the chosen ground truth / drive sits outside the region the network was TRAINED on.

    Bounds-checking is NOT enough, and this is the check that would have caught the 2026-07-27 retrain:
      * the ND prior is a stability-SCREENED GMM, so a value can sit inside the box yet in a near-empty
        corner the training data never visited; and
      * the forcing block is deliberately not range-checked at all, so a cell with amp=freq=0 was being
        conditioned against a log-uniform drive prior that *cannot produce 0* -- a point of zero
        probability under training, where the flow can only revert to the prior.

    SAMPLING-based on purpose: the latent prior is mixed-device (cpu rescale bijection + cuda ND GMM) and
    the pipeline rule is sample-only, never ``.log_prob``.

    :return: human-readable warnings; empty when everything is comfortably in-distribution.
    """
    if not cfg.has_ground_truth:
        return []
    msgs = []

    def _flag(kind, names, values, samples):
        if samples is None or samples.numel() == 0:
            return
        s = samples.detach().to("cpu", torch.float64)
        lo = torch.quantile(s, lo_pct / 100.0, dim=0)
        hi = torch.quantile(s, hi_pct / 100.0, dim=0)
        for i, name in enumerate(names):
            # PHASE is circular: its prior is uniform over a full turn, so 0 and 2*pi are the same point
            # and a percentile band says nothing about being in-distribution. Every phase is reachable.
            if name.split("_")[0] == "phase":
                continue
            v = float(values[i])
            if not (float(lo[i]) <= v <= float(hi[i])):
                msgs.append(
                    f"{kind} '{name}' = {v:g} lies outside the training prior's {lo_pct:g}-{hi_pct:g}% "
                    f"range [{float(lo[i]):g}, {float(hi[i]):g}]. The posterior will extrapolate here "
                    f"and may simply revert to the prior for this parameter.")

    with torch.no_grad():
        try:
            theta = inferred_prior.sample((n_samples,))
        except Exception:                              # noqa: BLE001 -- a diagnostic must never break a run
            theta = None
        _flag("Ground-truth parameter", list(cfg.params_dict) + list(cfg.rescale_params),
              cfg.ground_truth, theta)

        if cfg.has_forcing and not cfg.chi_mode:
            # chi-mode ignores the cell's own drive entirely, so it cannot be out-of-distribution there.
            try:
                drive = force_prior.sample((n_samples,)) if force_prior is not None else None
            except Exception:                          # noqa: BLE001
                drive = None
            _flag("Drive parameter", list(cfg.force_params_dict),
                  [v for v, _ in cfg.force_params_dict.values()], drive)
    return msgs


def _observation_inits(cfg: SimConfig) -> torch.Tensor:
    """
    (1, n_vars) initial conditions for observation-side simulation (PPC / eye-test): the loaded cell's
    inits when present (simulated branch), else the model-default the training loop synthesizes
    (experimental branch has no cell; the transient washes these out).
    """
    if cfg.inits_dict:
        return cfg.inits_tensor
    from core import registry
    if registry.is_user_model(cfg.model):
        from core.SBI.Priors.user_prior import declared_inits
        return declared_inits(registry.get(cfg.model)).to(dtype=cfg.hw.dtype, device=cfg.hw.device)
    n_pos, n_prob = pipeline.INIT_SHAPES[cfg.model.lower()]
    rng = np.random.RandomState(0)
    arr = np.concatenate([rng.randint(0, 10, size=(1, n_pos)), np.zeros((1, n_prob))], axis=1)
    return torch.tensor(arr, dtype=cfg.hw.dtype, device=cfg.hw.device)


# ── Step 4a: Calibration diagnostics (data-free — no chosen observation) ─────
def _thin_ticks(fig, max_ticks: int = 3, rotation: int = 30) -> None:
    """Keep a dense grid of small panels legible: few ticks, rotated, small labels.

    A 13x13 corner at default tick density draws overlapping numbers on every panel, which is what makes
    the plot unreadable rather than the panel size itself."""
    from matplotlib.ticker import MaxNLocator
    for ax in fig.axes:
        try:
            ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="both"))
            ax.tick_params(axis="both", labelsize=7)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(rotation)
                lbl.set_horizontalalignment("right")
        except Exception:                              # noqa: BLE001 -- cosmetic only, never fatal
            continue


def _emit(fig_sink, title: str, fig) -> None:
    """Display a figure: hand it to fig_sink (a GUI canvas) when given, else fall back to the legacy
    blocking plt.show() (CLI). This keeps orchestrator.run's CLI behavior unchanged when fig_sink is None."""
    if fig_sink is not None:
        fig_sink(title, fig)
    else:
        plt.show()


def validate_calibration(cfg: SimConfig, posterior: DirectPosterior | TransformedPosterior,
                         inferred_prior: Distribution, force_prior: Distribution,
                         *, fig_sink=None) -> None:
    """
    Data-free posterior calibration: SBC (Talts 2018, marginals) + expected coverage (TARP, Lemos
    2023). Both draw their calibration set from the PRIOR (theta_star ~ prior, x_cal simulated), so
    this runs right after training with no chosen observation.

    :param inferred_prior: the actual training prior (ND x rescale product prior) — SBC draws
                           theta_star from it, not from the posterior.
    """
    _assert_prior_used_matches_posterior(posterior, inferred_prior, "SBC/TARP calibration")
    t = cfg.t
    device = cfg.hw.device
    dtype = cfg.hw.dtype
    # Posterior's actual transform (rotated if REPARAM_ROTATE) so the cal prior + theta_transform match.
    T = posterior.T if isinstance(posterior, TransformedPosterior) else build_inferred_bijection(cfg)

    # Critical: draw theta_star from the PRIOR (not the posterior) for valid SBC.
    val_latent_prior = _build_latent_prior_for_validation(cfg, inferred_prior)
    # If the posterior uses a decorrelating rotation, rotate the calibration prior to match it.
    if hasattr(T, "parts") and len(T.parts) and isinstance(T.parts[0], OrthogonalTransform):
        val_latent_prior = RotatedLatentPrior(val_latent_prior, T.parts[0].M.transpose(-1, -2))
    x_cal, theta_star = analysis.gen_cal_data(
        model=cfg.model, prior=val_latent_prior,
        forcing_prior=force_prior,
        t=t, steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min, n_cal=SBC_N_CAL,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds,
        theta_transform=T,
        state_dep_drift=cfg.state_dep_drift,
        # _observation_inits: SBC/TARP draw theta from the PRIOR and need no ground truth, so this must
        # work on a cell-free config (cfg.inits_tensor would raise). See build_posterior.
        spontaneous_only=not cfg.has_forcing, chi_mode=cfg.chi_mode,
        chi_f0=cfg.chi_f0, chi_freq_bounds=cfg.chi_freq_bounds,
        chi_k_pad=cfg.chi_k_pad, chi_max_cycles=cfg.chi_max_cycles,
        # chi_k_fixed stays None here: validate_calibration's SBC is the POOLED one, over the same
        # mixture of probe counts training saw. Stratifying by count is scripts/sbc_characterize.py's
        # CHI_K_FIXED, run per stratum -- see section 4.1 step 5.
        chi_k_fixed=None,
        n_vars=_observation_inits(cfg).shape[-1],
        dtype=dtype, device=device,
    )
    x_cal_dev = x_cal.to(device)
    theta_star_dev = theta_star.to(device)

    # --- SBC (Talts 2018, marginals) via sbi.diagnostics ---
    ranks, dap_samples = run_sbc(
        thetas=theta_star_dev, xs=x_cal_dev, posterior=posterior,
        num_posterior_samples=1000, reduce_fns="marginals",
        use_batched_sampling=True, show_progress_bar=True,
    )
    prior_samples = inferred_prior.sample((theta_star.shape[0],)).cpu()
    sbc_stats = check_sbc(
        ranks=ranks.cpu(), prior_samples=prior_samples, dap_samples=dap_samples.cpu(),
        num_posterior_samples=1000,
    )
    print("SBC uniformity checks:")
    for j, label in enumerate(cfg.inferred_labels):
        print(f"  {label}: KS p={sbc_stats['ks_pvals'][j]:.3f}  "
              f"c2st_ranks={sbc_stats['c2st_ranks'][j]:.3f}  "
              f"c2st_dap={sbc_stats['c2st_dap'][j]:.3f}")

    # Both SBC figures are grids of small panels, so give each ROW enough height for its own x-label --
    # at the previous 2.75 in/row the per-panel "posterior rank <param>" label was clipped by the row
    # beneath it -- and add explicit vertical spacing rather than relying on tight_layout alone.
    n_sbc_rows = math.ceil(len(cfg.inferred_labels) / 4)
    f_cdf, _ = sbc_rank_plot(ranks=ranks, num_posterior_samples=1000, plot_type="cdf",
                             parameter_labels=cfg.inferred_labels, figsize=(16, 3.4 * n_sbc_rows))
    f_cdf.subplots_adjust(hspace=0.75, wspace=0.3)
    _thin_ticks(f_cdf, max_ticks=4, rotation=0)
    f_hist, _ = sbc_rank_plot(ranks=ranks, num_posterior_samples=1000, plot_type="hist",
                              parameter_labels=cfg.inferred_labels, figsize=(16, 3.4 * n_sbc_rows))
    f_hist.subplots_adjust(hspace=0.75, wspace=0.3)
    _thin_ticks(f_hist, max_ticks=4, rotation=0)
    if fig_sink is not None:
        fig_sink("SBC ranks (CDF)", f_cdf)
        fig_sink("SBC ranks (histogram)", f_hist)
    else:
        plt.show()   # CLI: a single blocking show for both open SBC figures (unchanged)

    # --- Expected coverage (TARP, Lemos 2023) via sbi.diagnostics ---
    ecp, alpha_grid = run_tarp(
        thetas=theta_star_dev, xs=x_cal_dev, posterior=posterior,
        num_posterior_samples=1000, use_batched_sampling=True,
        z_score_theta=True, show_progress_bar=True,
    )
    atc, tarp_kspval = check_tarp(ecp.cpu(), alpha_grid.cpu())
    print(f"TARP: ATC={atc:.3f}  KS p={tarp_kspval:.3f}")
    plot_tarp(ecp.cpu(), alpha_grid.cpu(),
              title=f"TARP (ATC={atc:.3f}, KS p={tarp_kspval:.3f})")
    _emit(fig_sink, "TARP coverage", plt.gcf())


# ── Step 4b: Inference visualization (requires a chosen observation) ─────────
def infer_and_visualize(cfg: SimConfig, posterior: DirectPosterior | TransformedPosterior,
                        obs_stats: torch.Tensor, obs_data: torch.Tensor, t_dim: torch.Tensor,
                        show_truth: bool, *, fig_sink=None) -> None:
    """
    Observation-dependent posterior plots for a chosen observation (a simulated ground-truth cell or
    experimental data): corner plot, posterior-predictive check (PPC), and the eye test. show_truth
    overlays the ground truth (simulated branch) or omits it (experimental branch).

    :param fig_sink: Optional (title, fig) -> None display callback (a GUI embeds the figures); when
                     None each plot falls back to the legacy blocking plt.show() (CLI unchanged).
    """
    t = cfg.t
    device = cfg.hw.device
    dtype = cfg.hw.dtype
    T_obs = cfg.T_obs
    inits = _observation_inits(cfg)

    # Corner plot
    samples = posterior.sample((1000,), x=obs_stats.to(device))
    # Size the corner by the PARAMETER COUNT: a 13x13 grid at pairplot's default is cramped enough that
    # tick labels overlap and axis titles clip. Thin the ticks for the same reason.
    n_p = len(cfg.inferred_labels)
    fig, ax = pairplot(
        samples.cpu().numpy(),
        points=(np.array([cfg.ground_truth]) if show_truth else None),
        labels=cfg.inferred_labels,
        figsize=(min(24, max(8, 1.35 * n_p)), min(24, max(8, 1.35 * n_p))),
    )
    _thin_ticks(fig)
    _emit(fig_sink, "Posterior corner", fig)

    # PPC - Option B: sort posterior samples by t_scale, process in mini-batches
    # Each sample gets its own subsample_factor based on its t_scale; all samples
    # share physical duration T_obs at dt_exp sampling (matching the observation).
    nd_dim = len(cfg.params_dict)
    samples_nd = samples[:, :nd_dim]
    samples_rescale = samples[:, nd_dim:]
    n_samples = samples.shape[0]
    # Same for all samples. Prefer the length generate_observations actually resolved (post
    # cost-ceiling clip); fall back to the formula only on the experimental paths, which never call
    # generate_observations and take their length from the recording itself.
    N_points_obs = cfg.n_obs if cfg.n_obs is not None else int(cfg.T_obs / cfg.dt_exp)

    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
    forcing_gt_expanded = forcing_gt.expand(n_samples, -1)  # (n_samples, n_forcing); empty if no forcing
    n_vars = inits.shape[-1]
    n_force_ch = forcing.n_force_channels(cfg.model, cfg.forcing_idx, n_vars)

    # Sort by t_scale (ascending) so each bin contains similar-scale samples
    t_scales_all = samples_rescale[:, cfg.rescale_idx["t_scale"]]
    sort_idx = torch.argsort(t_scales_all)
    inv_sort_idx = torch.argsort(sort_idx)
    samples_nd_sorted = samples_nd[sort_idx]
    samples_rescale_sorted = samples_rescale[sort_idx]

    x_dim_sorted = torch.empty((n_samples, N_points_obs), dtype=dtype, device=device)
    x_spont_sorted = torch.empty((n_samples, N_points_obs), dtype=dtype, device=device)
    arange_out = torch.arange(N_points_obs, device=device, dtype=torch.long)
    n_bins = math.ceil(n_samples / PPC_BIN_SIZE)
    # chi-mode: per-sample chi(omega) block, filled bin-by-bin alongside the spontaneous run.
    chi_block_sorted = (torch.empty((n_samples, expected_forcing_dim(cfg)),
                                    dtype=dtype, device=device) if cfg.chi_mode else None)

    with torch.no_grad():
        for b in tqdm(range(n_bins), desc="PPC simulations", leave=False):
            start = b * PPC_BIN_SIZE
            end = min(start + PPC_BIN_SIZE, n_samples)
            bs = end - start

            bin_nd = samples_nd_sorted[start:end]
            bin_rescale = samples_rescale_sorted[start:end]
            bin_t_scales = bin_rescale[:, cfg.rescale_idx["t_scale"]]

            # Smallest t_scale in the bin determines the finest resolution needed
            # (largest subsample_factor, hence largest n_fine_total)
            bin_t_scale_min = bin_t_scales.min().item()
            max_subsample_bin = max(1, round((cfg.dt_exp / bin_t_scale_min) / cfg.dt_nd_min))
            n_fine_bin = min(cfg.steady_idx + N_points_obs * max_subsample_bin, len(t))
            t_fine_bin = t[:n_fine_bin]
            n_segs_bin = max(1, math.ceil(n_fine_bin / CHUNK_LEN))

            # Per-sample downsample indices (each row uses its own subsample_factor)
            subsample_factors = torch.clamp(
                torch.round((cfg.dt_exp / bin_t_scales) / cfg.dt_nd_min), min=1
            ).long()  # (bs,)
            idx = subsample_factors.unsqueeze(1) * arange_out.unsqueeze(0)  # (bs, N_points_obs)

            x_scale_col = bin_rescale[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
            x_offset_col = bin_rescale[:, cfg.rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in cfg.rescale_idx else 0.0

            # Forced run (Group G) then spontaneous run (Groups A-F); no-forcing / chi-mode = spontaneous
            # only (chi-mode's driven info is the separate K-freq chi block computed below).
            if cfg.has_forcing and not cfg.chi_mode:
                force_bin = pipeline.build_nondim_sin_force_tensor(
                    forcing_gt.expand(bs, -1), t_fine_bin, bin_rescale, cfg.forcing_idx, cfg.rescale_idx)
                run_specs = ((force_bin, x_dim_sorted), (torch.zeros_like(force_bin), x_spont_sorted))
            else:
                force_bin = torch.zeros((bs, n_force_ch, t_fine_bin.shape[0]), dtype=dtype, device=device)
                run_specs = ((force_bin, x_spont_sorted),)
            for force_run, dest in run_specs:
                x_nd_bin = pipeline.gen_obs(
                    model=cfg.model, params=bin_nd, t=t_fine_bin,
                    inits=inits.expand(bs, -1),
                    force=force_run, n_segs=n_segs_bin, steady_idx=cfg.steady_idx,
                    state_dep_drift=cfg.state_dep_drift,
                    batch_size=bs, var_idx=0, dtype=dtype, device=device,
                )[0, :, :]  # (bs, n_fine_bin - steady_idx)
                idx_c = torch.clamp(idx, max=x_nd_bin.shape[1] - 1)  # safety for OOD samples
                x_nd_ds = torch.gather(x_nd_bin, dim=1, index=idx_c)  # (bs, N_points_obs)
                dest[start:end] = x_scale_col * x_nd_ds + x_offset_col
                del x_nd_bin, x_nd_ds

            del force_bin
            if cfg.chi_mode:
                # Single-tone probes for this bin (per-sample t_scale -> subsample_factors).
                #
                # Driven at the OBSERVATION'S ABSOLUTE FREQUENCIES, not at each sample's own
                # mult_k*f_peak. The experiment fixed those frequencies; a PPC that re-derives them
                # per posterior sample simulates a DIFFERENT experiment for every sample, and its chi
                # z-scores then come out small for the wrong reason.
                obs_freqs = getattr(cfg, "chi_obs_freqs", None)
                if obs_freqs is None:
                    probe, absolute = chi.chi_multipliers_for(cfg, dtype=dtype, device=device), False
                else:
                    probe, absolute = obs_freqs.to(device=device, dtype=dtype), True
                chi_block_sorted[start:end] = pipeline.gen_chi_block(
                    cfg.model, bin_nd, bin_rescale, x_spont_sorted[start:end], t_fine_bin,
                    inits.expand(bs, -1), cfg.rescale_idx, n_segs_bin, cfg.steady_idx,
                    subsample_factors, N_points_obs, cfg.dt_exp,
                    probe, cfg.chi_f0, k_pad=cfg.chi_k_pad, bounds=cfg.chi_freq_bounds,
                    absolute_freqs=absolute, max_cycles=cfg.chi_max_cycles,
                    state_dep_drift=cfg.state_dep_drift, dtype=dtype, device=device)[0]
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Restore original sample order
    x_spont = x_spont_sorted[inv_sort_idx]
    log_T_obs = torch.full((n_samples, 1), math.log(T_obs), dtype=dtype)
    # Layout [S | log(T) | forcing|chi] — must match the observation in generate_observations.
    if cfg.chi_mode:
        x_dim = x_spont                                 # PPC "sample trajectories" = passive spontaneous trace
        sim_stats = pipeline.gen_stats(x_spont, None, cfg.dt_exp, None, None, None,
                                       device=device, spontaneous_only=True)
        sim_stats = torch.cat([sim_stats, log_T_obs, chi_block_sorted[inv_sort_idx].cpu()], dim=-1)
    elif cfg.has_forcing:
        x_dim = x_dim_sorted[inv_sort_idx]
        n_drive = x_dim.shape[0]
        sim_stats = pipeline.gen_stats(
            x_spont, x_dim, cfg.dt_exp,
            forcing_gt[:, cfg.forcing_idx["amp"]].expand(n_drive),
            forcing_gt[:, cfg.forcing_idx["freq"]].expand(n_drive),
            forcing_gt[:, cfg.forcing_idx["phase"]].expand(n_drive),
            device=device,
        )
        sim_stats = torch.cat([sim_stats, log_T_obs, forcing_gt_expanded.cpu()], dim=-1)
    else:
        x_dim = x_spont                                 # the PPC "sample trajectories" are spontaneous
        sim_stats = pipeline.gen_stats(x_spont, None, cfg.dt_exp, None, None, None,
                                       device=device, spontaneous_only=True)
        sim_stats = torch.cat([sim_stats, log_T_obs], dim=-1)
    results = analysis.posterior_predictive_check(obs_stats.squeeze(), sim_stats)
    fig_ppc = visualizers.plot_ppc(
        results,
        ground_truth=(cfg.ground_truth if show_truth else None),
        param_names=cfg.inferred_labels,
        n_samples=n_samples,
    )
    _emit(fig_sink, "Posterior predictive check", fig_ppc)

    # Eye test: central-estimate trajectories (posterior mean & median) vs ground truth.
    # The MAP (argmax-log-prob sample) is a poor summary of a wide posterior, so instead we
    # simulate the trajectories of the posterior MEAN and MEDIAN parameter vectors. Averaging
    # the sample trajectories pointwise would destructively cancel the oscillation (samples
    # differ in freq/phase), so we simulate the central PARAMETERS and keep a coherent drive
    # response. Each central vector is simulated on the same physical grid as the observation
    # (T_obs at dt_exp), mirroring one row of the per-sample PPC path above.
    def _simulate_central_trajectory(theta_central: torch.Tensor) -> np.ndarray:
        """Forced-run trajectory of a single (nd + rescale) param vector, on the obs grid."""
        theta_central = theta_central.unsqueeze(0)                       # (1, n_inferred)
        central_nd = theta_central[:, :nd_dim]
        central_rescale = theta_central[:, nd_dim:]
        t_scale_c = central_rescale[0, cfg.rescale_idx["t_scale"]].item()
        subsample_c = max(1, round((cfg.dt_exp / t_scale_c) / cfg.dt_nd_min))
        n_fine_c = min(cfg.steady_idx + N_points_obs * subsample_c, len(t))
        t_fine_c = t[:n_fine_c]
        n_segs_c = max(1, math.ceil(n_fine_c / CHUNK_LEN))
        if cfg.has_forcing and not cfg.chi_mode:
            force_c = pipeline.build_nondim_sin_force_tensor(
                forcing_gt, t_fine_c, central_rescale, cfg.forcing_idx, cfg.rescale_idx)
        else:
            force_c = torch.zeros((1, n_force_ch, t_fine_c.shape[0]), dtype=dtype, device=device)
        x_nd_c = pipeline.gen_obs(
            model=cfg.model, params=central_nd, t=t_fine_c, inits=inits,
            force=force_c, n_segs=n_segs_c, steady_idx=cfg.steady_idx,
            state_dep_drift=cfg.state_dep_drift, var_idx=0, dtype=dtype, device=device,
        )[0, :, :]                                                       # (1, n_fine_c - steady_idx)
        idx_c = torch.clamp(
            torch.arange(N_points_obs, device=device) * subsample_c, max=x_nd_c.shape[1] - 1
        )
        x_nd_c_ds = x_nd_c[:, idx_c]                                     # (1, N_points_obs)
        x_scale_c = central_rescale[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
        x_offset_c = central_rescale[:, cfg.rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in cfg.rescale_idx else 0.0
        return (x_scale_c * x_nd_c_ds + x_offset_c)[0].cpu().numpy()     # (N_points_obs,)

    with torch.no_grad():
        x_mean = _simulate_central_trajectory(samples.mean(dim=0))
        x_median = _simulate_central_trajectory(samples.median(dim=0).values)

    # ── Posterior-overlay figures ────────────────────────────────────────────────────────────────
    # Phase is set by the noise realisation, not by theta, so a draw can never match the observation
    # pointwise; these figures either align that away explicitly or avoid depending on it. See
    # core/SBI/overlay.py.
    _emit_overlay_figures(cfg, obs_data, x_dim, sim_stats, obs_stats, samples, show_truth, fig_sink)

    t_plot = t_dim.squeeze(0).cpu().numpy()
    fig = visualizers.plot_posterior_vs_truth(
        t=t_plot,
        x_true=obs_data[0, :].cpu().numpy(),
        x_mean=x_mean,
        x_median=x_median,
        x_samples=x_dim.cpu().numpy(),
        n_show=10,
        xlabel=labels.axis_label("t", "s"),
        ylabel=labels.axis_label("x", cfg.length_unit),
    )
    _emit(fig_sink, "Eye test", fig)

def _emit_overlay_figures(cfg, obs_data, x_samples, sim_stats, obs_stats, samples, show_truth, fig_sink):
    """The five posterior-overlay figures. Isolated + best-effort: they are diagnostics, so a failure
    here must never lose the corner/PPC/eye-test the caller already produced."""
    try:
        gt = obs_data[0, :].detach().cpu()
        traces = x_samples.detach().cpu()
        if traces.ndim != 2 or traces.shape[-1] != gt.shape[-1] or traces.shape[0] < 2:
            # Say so. This used to be a bare `return` that also bypassed the except-clause's warning
            # below, so the five overlay figures looked unimplemented rather than skipped -- and a
            # one-sample width disagreement (see cfg.n_obs) made this the EXPECTED path, not an edge
            # case.
            warnings.warn(
                f"Posterior-overlay figures skipped: need 2-D posterior traces at least 2 deep whose "
                f"length matches the observation, got traces {tuple(traces.shape)} vs observation "
                f"length {gt.shape[-1]}.",
                stacklevel=2,
            )
            return
        dt_s = 1.0 / cfg.get_unit_conversion_factor("s") * cfg.dt_exp    # sample spacing in SECONDS
        labels_ = cfg.inferred_labels
        truth = cfg.ground_truth if show_truth else None
        xlab, ylab = labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)

        # window the time-domain overlays to a readable number of cycles
        f_pk = float(chi.peak_freq(gt.unsqueeze(0), dt_s)[0])
        w = overlay.cycle_window(gt.shape[-1], dt_s, f_pk, EYE_TEST_CYCLES)
        t_w = np.arange(w) * dt_s

        # The five figures are emitted in four independent groups. (2)+(3) share rank_by_trace's
        # alignment so they stand or fall together; the rest are independent. One shared try used to
        # mean a failure in the first silently cost the other four -- notably cycle_average's
        # quantile, which can raise on a large enough draw x sample product.
        def _group(name, fn):
            try:
                fn()
            except Exception as e:                        # noqa: BLE001 -- diagnostics, never fatal
                warnings.warn(f"Posterior-overlay figure(s) '{name}' could not be produced: {e}",
                              stacklevel=2)

        def _best_fit_stats():
            # (1) best fit by SUMMARY STATISTICS -- the space the posterior conditioned on
            order_s, dist_s = overlay.rank_by_stats(sim_stats, obs_stats)
            i_s = int(order_s[0])
            fit_s, _ = overlay.align_to(gt, traces[i_s:i_s + 1])
            _emit(fig_sink, "Best fit — summary stats", visualizers.plot_best_fit_overlay(
                t_w, gt[:w].numpy(), fit_s[0, :w].numpy(), param_labels=labels_,
                param_values=samples[i_s].detach().cpu(), ground_truth=truth,
                criterion="closest summary statistics",
                score_text=f"RMS z = {float(dist_s[i_s]):.3f}", xlabel=xlab, ylabel=ylab))

        def _best_fit_trace_and_band():
            # (2) best fit by TRACE, after alignment -- the draw that literally looks most like the data
            order_t, rmse_t, aligned = overlay.rank_by_trace(gt, traces)
            i_t = int(order_t[0])
            _emit(fig_sink, "Best fit — trace", visualizers.plot_best_fit_overlay(
                t_w, gt[:w].numpy(), aligned[i_t, :w].numpy(), param_labels=labels_,
                param_values=samples[i_t].detach().cpu(), ground_truth=truth,
                criterion="closest waveform (phase-aligned)",
                score_text=f"RMSE = {float(rmse_t[i_t]):.3g}", xlabel=xlab, ylabel=ylab))

            # (3) band over the top-N best draws, each aligned independently
            n_best = int(min(50, max(5, traces.shape[0] // 20)))
            best = aligned[order_t[:n_best]].to(torch.float64)
            q = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
            band = torch.quantile(best, q, dim=0)
            _emit(fig_sink, "Posterior overlay (band)", visualizers.plot_overlay_band(
                t_w, gt[:w].numpy(), band[0, :w].numpy(), band[1, :w].numpy(), band[2, :w].numpy(),
                n_used=n_best, xlabel=xlab, ylabel=ylab))

        def _psd_band():
            # (4) PSD band -- phase-invariant
            freqs, lo_p, med_p, hi_p = overlay.psd_band(traces, dt_s)
            _, gt_p = overlay.psd(gt.unsqueeze(0), dt_s)
            _emit(fig_sink, "Power spectrum", visualizers.plot_psd_overlay(
                freqs.numpy(), gt_p[0].numpy(), lo_p.numpy(), med_p.numpy(), hi_p.numpy()))

        def _cycle_avg():
            # (5) cycle-averaged waveform -- phase-invariant shape comparison
            if f_pk > 0:
                ph, sim_m, sim_lo, sim_hi = overlay.cycle_average(traces, dt_s, f_pk)
                _, gt_m, _, _ = overlay.cycle_average(gt.unsqueeze(0), dt_s, f_pk)
                _emit(fig_sink, "Cycle-averaged waveform", visualizers.plot_cycle_average(
                    ph.numpy(), gt_m.numpy(), sim_m.numpy(), sim_lo.numpy(), sim_hi.numpy(),
                    ylabel=ylab))

        _group("best fit (summary stats)", _best_fit_stats)
        _group("best fit (trace) + band", _best_fit_trace_and_band)
        _group("power spectrum", _psd_band)
        _group("cycle-averaged waveform", _cycle_avg)
    except Exception as e:                                # noqa: BLE001 -- diagnostics, never fatal
        warnings.warn(f"Posterior-overlay figures could not be produced: {e}", stacklevel=2)


def _build_latent_prior_for_validation(cfg, inferred_prior):
    """Mirror of the latent-prior construction in build_posterior, for gen_cal_data in validate."""
    nd_prior_physical = inferred_prior.distributions[0]
    if not isinstance(nd_prior_physical, torch.distributions.TransformedDistribution):
        raise ValueError(
            "Loaded ND prior is not a TransformedDistribution — it was saved with the pre-reparameterization "
            "pipeline. Regenerate the prior with the current `gen_prior` before running validate."
        )
    latent_nd = nd_prior_physical.base_dist
    T_rescale = build_rescale_bijection(cfg)
    latent_rescale = torch.distributions.TransformedDistribution(inferred_prior.distributions[1], T_rescale.inv)
    return ProductPrior(
        distributions=[latent_nd, latent_rescale],
        dims=[len(cfg.params_dict), len(cfg.rescale_params)],
    )


# ── Step 5: Inference on real experimental data ────────────────────────────
def build_experiment_obs(
    cfg: SimConfig,
    X_obs_spont: torch.Tensor,
    X_obs_forced: torch.Tensor,
    T_obs_s: float,
    forcing_params_si: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the conditioning vector + observed trajectory from a real experimental recording, and set the
    observation context on cfg (T_obs + forcing values, in cell-file units) so infer_and_visualize's
    PPC / eye-test can simulate. Posterior sampling + the corner plot are done by infer_and_visualize.

    The recording must be sampled at 1/cfg.dt_exp (the camera frame rate the network was trained on).
    T_obs and the forcing params are given in SI units and converted to cell-file units here.

    :param X_obs_spont: 1D spontaneous (unforced) recording, shape (N_obs,), at 1/cfg.dt_exp.
    :param X_obs_forced: 1D forced (driven) recording, shape (N_obs,), at 1/cfg.dt_exp.
    :param T_obs_s: Observation duration in SECONDS.
    :param forcing_params_si: Dict with keys "amp" (N), "freq" (Hz), "phase" (rad), "offset" (N).
    :return: (obs_stats, obs_data, t_dim): the [S | log(T) | forcing] conditioning vector (1, D), the
             forced recording (1, N_obs) for the eye-test, and the dimensional time axis (1, N_obs).
    """
    dtype = cfg.hw.dtype

    # Unit conversions: SI -> cell file units.
    # Known forcing param SI units; fall back to no conversion (dimensionless) if unknown.
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell

    # Consistency check: X_obs must be sampled at 1/dt_exp with duration T_obs.
    expected_N = int(T_obs / cfg.dt_exp)
    if X_obs_spont.shape[-1] != X_obs_forced.shape[-1]:
        raise ValueError(
            f"Spontaneous and forced recordings must be the same length "
            f"({X_obs_spont.shape[-1]} vs {X_obs_forced.shape[-1]})."
        )
    if abs(X_obs_forced.shape[-1] - expected_N) > 1:
        warnings.warn(
            f"Recording length ({X_obs_forced.shape[-1]}) doesn't match expected from T_obs_s={T_obs_s:.4f}s "
            f"at 1/dt_exp sampling (expected ~{expected_N} points). "
            f"Check that both recordings are sampled at dt_exp={cfg.dt_exp:.6f} (cell units).",
            stacklevel=2,
        )

    # Out-of-distribution warning: compute the minimum feasible t_scale for this T_obs.
    # The NN was only trained on batches where n_fine_total <= N_ND_MAX, i.e.
    #   steady_idx + (T_k / dt_exp) * (t_scale_hi / t_scale_k) <= N_ND_MAX
    # Solving for t_scale_k given T_k = T_obs:
    t_scale_lo_prior, t_scale_hi = cfg.t_scale_bounds
    budget = N_ND_MAX - cfg.steady_idx
    if budget > 0:
        t_scale_min_feasible = (T_obs / cfg.dt_exp) * t_scale_hi / budget
        if t_scale_min_feasible > t_scale_lo_prior:
            warnings.warn(
                f"Inference out-of-distribution risk: for T_obs={T_obs_s:.2f}s, the NN was "
                f"only trained on t_scale >= {t_scale_min_feasible:.3f} (in cell file units). "
                f"If the true t_scale is below this, the posterior may extrapolate poorly.",
                stacklevel=2,
            )

    # Build forcing tensor generically: iterate cfg.force_params_dict and apply the appropriate SI->cell
    # conversion per parameter name (config.FORCING_SI_UNITS is the single source of truth; the CLI/GUI
    # display hints derive from it). Unknown names raise. "Hz" is SPECIAL-CASED below -- see that map.
    _FORCING_SI_UNITS = FORCING_SI_UNITS
    forcing_t = torch.empty((1, len(cfg.force_params_dict)), dtype=dtype)
    for name in cfg.force_params_dict.keys():
        if name not in forcing_params_si:
            raise KeyError(f"forcing_params_si missing required key '{name}' "
                           f"(cell file expects: {list(cfg.force_params_dict.keys())})")
        if name not in _FORCING_SI_UNITS:
            raise ValueError(f"Unknown forcing parameter '{name}'. Known: {list(_FORCING_SI_UNITS)}. "
                             f"Add an entry to _FORCING_SI_UNITS in build_experiment_obs.")
        si_unit = _FORCING_SI_UNITS[name]
        si_val = forcing_params_si[name]
        if si_unit is None:                    # dimensionless (phase, in radians)
            cell_val = si_val
        elif si_unit == "Hz":                  # inverse cell time -- derived from the TIME unit
            cell_val = si_val * cfg.freq_si_to_cell
        else:
            cell_val = si_val * cfg.get_unit_conversion_factor(si_unit)
        forcing_t[0, cfg.forcing_idx[name]] = cell_val

    # Reshape both recordings to (1, N_obs) and compute summary statistics with dt_exp
    X_spont_batched = X_obs_spont.to(dtype=dtype).unsqueeze(0)
    X_forced_batched = X_obs_forced.to(dtype=dtype).unsqueeze(0)
    obs_stats = pipeline.gen_stats(
        X_spont_batched, X_forced_batched, cfg.dt_exp,
        forcing_t[:, cfg.forcing_idx["amp"]], forcing_t[:, cfg.forcing_idx["freq"]],
        forcing_t[:, cfg.forcing_idx["phase"]], device=cfg.hw.device,
    )

    # Build conditioning vector: [S(X_obs), log(T_obs), forcing]
    log_T_obs = torch.tensor([[math.log(T_obs)]], dtype=dtype)
    obs_stats = torch.cat([obs_stats, log_T_obs, forcing_t], dim=-1)

    # Record the observation context so infer_and_visualize's PPC / eye-test can simulate.
    forcing_vals = {name: float(forcing_t[0, cfg.forcing_idx[name]]) for name in cfg.force_params_dict}
    cfg.set_observation_context(T_obs, forcing_vals)

    # Dimensional time axis (in SECONDS) + observed (forced) trajectory for the eye-test.
    N_obs = X_forced_batched.shape[-1]
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")   # cell time unit -> seconds
    t_dim = ((torch.arange(N_obs, dtype=dtype) * cfg.dt_exp) * s_per_cell).unsqueeze(0)
    return obs_stats, X_forced_batched, t_dim


def build_experiment_obs_spontaneous(
    cfg: SimConfig, X_obs: torch.Tensor, T_obs_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Passive-recording variant of build_experiment_obs for a NO-FORCING model: a SINGLE unforced
    recording, no forced file, and no forcing/frequency SI units (the drive machinery is not entered).
    Conditioning is [S(A-F, Group G zeroed) | log(T_obs)], matching generate_observations' no-forcing path.

    :param X_obs: 1D passive recording, shape (N_obs,), sampled at 1/cfg.dt_exp.
    :param T_obs_s: observation duration in SECONDS.
    :return: (obs_stats, obs_data, t_dim): the [S | log(T)] conditioning vector, the recording (1, N_obs),
             and the dimensional (seconds) time axis (1, N_obs).
    """
    dtype = cfg.hw.dtype
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell

    expected_N = int(T_obs / cfg.dt_exp)
    if abs(X_obs.shape[-1] - expected_N) > 1:
        warnings.warn(
            f"Recording length ({X_obs.shape[-1]}) doesn't match expected from T_obs_s={T_obs_s:.4f}s "
            f"at 1/dt_exp sampling (expected ~{expected_N} points).", stacklevel=2)

    X_batched = X_obs.to(dtype=dtype).unsqueeze(0)
    obs_stats = pipeline.gen_stats(X_batched, None, cfg.dt_exp, None, None, None,
                                   device=cfg.hw.device, spontaneous_only=True)
    log_T_obs = torch.tensor([[math.log(T_obs)]], dtype=dtype)
    obs_stats = torch.cat([obs_stats, log_T_obs], dim=-1)      # [S | log(T)], no forcing block

    cfg.set_observation_context(T_obs, {})
    N_obs = X_batched.shape[-1]
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")
    t_dim = ((torch.arange(N_obs, dtype=dtype) * cfg.dt_exp) * s_per_cell).unsqueeze(0)
    return obs_stats, X_batched, t_dim


def build_experiment_obs_chi(
    cfg: SimConfig, X_spont: torch.Tensor, X_forced_list: list[torch.Tensor],
    T_obs_s: float, F0_si: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    chi(omega) experimental path: ONE passive recording + ANY NUMBER of single-tone FORCED recordings,
    each locked in at THE FREQUENCY IT WAS ACTUALLY DRIVEN AT. Builds the conditioning
    [S(A-F) | log(T) | padded probe set], mirroring generate_observations' chi branch.

    This used to compute ``freq_k = mult_k * peak_freq(X_spont)`` itself and never asked what you
    drove at. Two ways that went wrong on the bench, both silent: the frequencies you can actually
    achieve are not exactly ``mult_k * Omega_0``, and even aiming for them, your Omega_0 estimate is
    not ``chi.peak_freq``'s (different trace length, windowing, bin resolution). A lock-in at the
    wrong frequency decays like a sinc -- a mismatch of a fraction of 1/T_obs destroys the estimate.
    It also demanded exactly K recordings, with no substitution if one failed.

    chi = response/drive is drive-amplitude-independent in the linear regime, so any linear physical
    drive the experiment used works -- it is reported (F0_si) only so the lock-in divides by it to yield
    the true physical susceptibility (x_scale/f_scale)*chi_nd, matching training.

    :param X_spont: 1D passive recording (N_obs,), sampled at 1/cfg.dt_exp.
    :param X_forced_list: the forced recordings. Either 1-D tensors -- legacy, assumed driven at
        ``chi.chi_multipliers_for(cfg)`` -- or ``(recording, drive_frequency_Hz)`` pairs, which is the
        form to use for real data. Any count from 1 to ``cfg.chi_k_pad``.
    :param T_obs_s: observation duration (seconds).
    :param F0_si: physical drive amplitude used (SI force, N); converted to cell force units.
    :return: (obs_stats, obs_data=X_spont as (1,N), t_dim in seconds).
    """
    dtype = cfg.hw.dtype
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell
    F0 = F0_si * cfg.get_unit_conversion_factor("N")           # SI force -> cell force units

    X_spont_b = X_spont.to(dtype=dtype).unsqueeze(0)          # (1, N)
    N = X_spont_b.shape[-1]
    expected_N = int(T_obs / cfg.dt_exp)
    if abs(N - expected_N) > 1:
        warnings.warn(
            f"Passive recording length ({N}) doesn't match T_obs_s={T_obs_s:.4f}s at 1/dt_exp "
            f"(expected ~{expected_N} points).", stacklevel=2)

    f_peak = chi.peak_freq(X_spont_b, cfg.dt_exp)             # (1,) Omega_0/2pi (cell freq units)
    nyq = 0.5 / cfg.dt_exp
    n_probes = len(X_forced_list)
    if not (1 <= n_probes <= cfg.chi_k_pad):
        raise ValueError(
            f"chi-mode accepts 1 to {cfg.chi_k_pad} forced recordings (CHI_K_PAD), got {n_probes}.")

    # Legacy positional form: no frequency supplied, so fall back to the nominal grid.
    paired = bool(X_forced_list) and isinstance(X_forced_list[0], (tuple, list))
    if not paired:
        mults = chi.chi_multipliers(dtype=dtype, device=torch.device("cpu"),
                                    n_freqs=n_probes, bounds=cfg.chi_freq_bounds)

    lo_b, hi_b = cfg.chi_freq_bounds
    u_mid, u_half = chi.band_norm(cfg.chi_freq_bounds)
    chis, u_list, logcyc_list, valid = [], [], [], []
    for k, item in enumerate(X_forced_list):
        if paired:
            Xf, freq_hz = item[0], float(item[1])
            if not (math.isfinite(freq_hz) and freq_hz > 0):
                raise ValueError(f"chi probe {k}: drive frequency must be finite and positive, "
                                 f"got {freq_hz} Hz.")
            # freq_si_to_cell, NOT get_unit_conversion_factor("Hz") -- the latter returns 1.0 against
            # an ms cell, a silent 1000x error landing as a wildly off-resonance but valid-looking chi.
            f_cell = torch.tensor([freq_hz * cfg.freq_si_to_cell], dtype=dtype)
        else:
            Xf, f_cell = item, (mults[k] * f_peak)
        Xf_b = Xf.to(dtype=dtype).unsqueeze(0)               # (1, N_k)
        N_k = Xf_b.shape[-1]
        T_k = N_k * cfg.dt_exp
        f_val = float(f_cell)
        # Refuse per row, naming the row -- the old code's only guard was a count check, and deleting
        # that without replacement would let a 2-probe set run clean against a 12-slot posterior and
        # return a near-prior answer: the first chi posterior's exact signature from another cause.
        if f_val >= 0.9 * nyq:
            raise ValueError(
                f"chi probe {k}: {freq_hz if paired else f_val:g} is at or above the recording's "
                f"Nyquist limit ({0.9 * nyq / cfg.freq_si_to_cell:g} Hz at dt_exp={cfg.dt_exp:g}).")
        u_k = math.log(f_val / float(f_peak))
        if abs((u_k - u_mid) / u_half) > config.CHI_UHAT_MAX:
            in_band = (lo_b * float(f_peak) / cfg.freq_si_to_cell,
                       hi_b * float(f_peak) / cfg.freq_si_to_cell)
            raise ValueError(
                f"chi probe {k}: {f_val / cfg.freq_si_to_cell:g} Hz is outside the band this "
                f"posterior was trained over. For this cell (Omega_0 = "
                f"{float(f_peak) / cfg.freq_si_to_cell:g} Hz) that is {in_band[0]:g}-{in_band[1]:g} Hz.")
        # THE DURATION CEILING (config.CHI_MAX_CYCLES) -- the same one gen_chi_raw applies to every
        # training row. A bench recording is routinely far longer than the ceiling, and locking in
        # over all of it would be wrong twice over: |chi| goes past the reproducibility wall (trap
        # CHI9), and logcyc reports a cycle count no training row ever carried, so the encoder is
        # extrapolating on the one channel that tells it how much to trust the probe. TRUNCATE rather
        # than mask -- the recording is fine, only its tail is unusable, and the leading prefix is
        # exactly what training measured. Warned, not silent: the user recorded that length on
        # purpose and is entitled to know only part of it was used.
        n_cap = max(1, int(math.floor(cfg.chi_max_cycles / f_val / cfg.dt_exp)))
        if n_cap < N_k:
            warnings.warn(
                f"chi probe {k}: {N_k} samples give {f_val * T_k:.1f} drive cycles, above the "
                f"{cfg.chi_max_cycles:g}-cycle ceiling; locking in over the first {n_cap} samples "
                f"({n_cap * cfg.dt_exp / s_to_cell:.3g} s) only. Above the ceiling |chi| stops being "
                f"reproducible at fixed parameters, so the extra recording carries no information "
                f"the network can use.", stacklevel=2)
            Xf_b, N_k = Xf_b[:, :n_cap], n_cap
            T_k = N_k * cfg.dt_exp
        # UNDER-RESOLVED IS MASKED, NOT REFUSED -- unlike the structural errors above.
        #
        # The distinction is train/eval consistency. Training MASKS a sub-cycle probe and keeps the
        # row, so the network has learned to condition on sets with absent probes; refusing here
        # would reject an observation the network handles perfectly well, and at the band's low edge
        # that is common (at Omega_0 = 7.6 Hz the 0.03x probe has a 4.4 s period, so a 1 s recording
        # cannot resolve it however carefully it was made). The other checks above are different in
        # kind: a bad frequency, an aliased probe or an out-of-band one indicate a mistake the user
        # must fix, not a limitation of the recording.
        resolved = f_val * T_k >= config.CHI_MIN_CYCLES
        if not resolved:
            need_s = config.CHI_MIN_CYCLES / f_val / s_to_cell
            warnings.warn(
                f"chi probe {k}: {N_k} samples give only {f_val * T_k:.2f} drive cycles, below the "
                f"{config.CHI_MIN_CYCLES:g}-cycle floor, so it is MASKED and contributes nothing. "
                f"Record >= {need_s:.3g} s at this frequency to use it.", stacklevel=2)
        chis.append(chi.lock_in_batched(Xf_b, 2.0 * math.pi * f_cell, F0, T_k, cfg.dt_exp))
        u_list.append(torch.tensor([u_k], dtype=dtype))
        logcyc_list.append(torch.tensor([math.log(f_val * T_k)], dtype=dtype))
        valid.append(resolved)

    chi_block, chi_mask = chi.pack_probe_block(
        torch.stack(chis, dim=1), torch.stack(u_list, dim=1), torch.stack(logcyc_list, dim=1),
        torch.tensor([valid], dtype=torch.bool), k_pad=cfg.chi_k_pad, bounds=cfg.chi_freq_bounds)
    if not bool(chi_mask.any()):
        # Every probe masked. The encoder handles an empty set (it returns its empty-set constant),
        # but then the posterior is conditioned on the passive trace alone -- so this would silently
        # answer a spontaneous question with a chi posterior. The user supplied recordings expecting
        # them to be used; say that none were.
        raise ValueError(
            f"chi: none of the {n_probes} supplied recordings produced a usable probe (all were "
            f"below the {config.CHI_MIN_CYCLES:g}-cycle floor or had a non-finite lock-in). The "
            f"conditioning would carry no susceptibility information at all.")

    obs_stats = pipeline.gen_stats(X_spont_b, None, cfg.dt_exp, None, None, None,
                                   device=cfg.hw.device, spontaneous_only=True)
    log_T_obs = torch.tensor([[math.log(T_obs)]], dtype=dtype)
    obs_stats = torch.cat([obs_stats, log_T_obs, chi_block.cpu()], dim=-1)

    cfg.set_observation_context(T_obs, {})
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")
    t_dim = ((torch.arange(N, dtype=dtype) * cfg.dt_exp) * s_per_cell).unsqueeze(0)
    return obs_stats, X_spont_b, t_dim
