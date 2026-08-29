"""
Pipeline orchestration for the SBI pipeline.

No input() calls live here -- all user interaction is delegated to cli.py.
This module owns the pipeline flow: observe -> prior -> posterior -> validate.
"""
import hashlib
import importlib
import math
import os
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
    SimConfig, PRIOR_PATH, POSTERIOR_PATH, PLOT_PATH, OBSERVATION_PATH,
    T_MIN_EXP_S, T_MAX_EXP_S,
    CHUNK_LEN, N_ND_MAX, SBC_N_CAL, STABILITY_SWEEP_ND_UNITS, TRAINING_NUM_RUNS,
    PRIOR_SWEEP_ITERATIONS, PRIOR_SWEEP_BATCH, TRAINING_RUN_SIZE, TRAINING_CHECKPOINT_EVERY,
    DENSITY_ESTIMATOR, NSF_HIDDEN_FEATURES, NSF_NUM_TRANSFORMS, NSF_NUM_BINS,
    TRAINING_NUM_ROUNDS, TRAINING_BATCH_SIZE, TRAINING_LEARNING_RATE,
    TRAINING_STOP_AFTER_EPOCHS, TRAINING_MAX_NUM_EPOCHS, TRAINING_SHOW_SUMMARY, FORCING_SI_UNITS,
    EYE_TEST_CYCLES,
)
from . import cli, config, forcing
from .Helpers import helpers, visualizers, file_manager, labels
from .Helpers.visualizers import emit_figure as _emit, thin_ticks as _thin_ticks
from .SBI.overlay import emit_overlay_figures as _emit_overlay_figures
from .SBI.run_guards import (CHI_OVERRIDE_ENV, _find_nd_gmm, _gmm_fingerprint,  # noqa: E402
                             _assert_prior_used_matches_posterior, _assert_prior_matches,
                             _assert_chi_config_is_deliberate,
                             _assert_amortization_understood, _log_params_for)
from .SBI import (embedded_network, pipeline, analysis, decorrelate, chi, derived, overlay, ppc,
                  truncate,
                  statistics, training_checkpoint)
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
    # TIER 1 (a box that declares T instead of f_scale): substitute the DERIVED f_scale into T's column before anything
    # simulates. A no-op for a box that declares f_scale. `sim_rescale_idx` is what the force
    # builders and gen_chi_raw must then be given -- handed the INFERRED index they would not
    # find 'f_scale', would fall into the Hopf-style x_scale/t_scale branch, and would drive
    # at a silently wrong amplitude.
    rescale_gt = derived.to_sim_rescale(cfg.params_tensor, rescale_gt, cfg.rescale_idx,
                                       *cfg.tier1_args)

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
        force = pipeline.build_nondim_sin_force_tensor(forcing_gt, t_fine, rescale_gt,
                                                      cfg.forcing_idx, cfg.sim_rescale_idx)
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
    if cfg.chi_mode:
        # [S(41, Group G zeroed) | log(T) | padded probe SET] -- probes at mult_k * Omega_0.
        # An OBSERVATION uses the deterministic grid, not the training sampler's jitter: this is a
        # specific measurement, and the PPC has to be able to reproduce its exact drive frequencies.
        obs_stats = pipeline.gen_stats(x_spont_dim, None, cfg.dt_exp, None, None, None,
                                       device=cfg.hw.device, spontaneous_only=True)
        obs_mults = chi.chi_multipliers_for(cfg)
        chi_block, _chi_mask = pipeline.gen_chi_block(
            cfg.model, cfg.params_tensor, rescale_gt, x_spont_dim, t_fine, cfg.inits_tensor,
            cfg.sim_rescale_idx, n_segs_gt, cfg.steady_idx, subsample_factor, N_obs, cfg.dt_exp,
            obs_mults, cfg.chi_f0, k_pad=cfg.chi_k_pad, bounds=cfg.chi_freq_bounds,
            max_cycles=cfg.chi_max_cycles,
            state_dep_drift=cfg.state_dep_drift, dtype=cfg.hw.dtype, device=cfg.hw.device)
        # Record the ABSOLUTE probe frequencies this observation was measured at, so the PPC drives
        # the same experiment rather than re-deriving frequencies from each posterior sample's own
        # f_peak -- which would simulate a different experiment and make the PPC agree for the wrong
        # reason.
        cfg.chi_obs_freqs = (obs_mults.to(cfg.hw.device)
                             * chi.peak_freq(x_spont_dim, cfg.dt_exp).median()).detach()
        obs_stats = statistics.conditioning_rows(obs_stats, cfg.T_obs, chi_block.cpu())
    elif cfg.has_forcing:
        obs_stats = pipeline.gen_stats(
            x_spont_dim, x_dim, cfg.dt_exp,
            forcing_gt[:, cfg.forcing_idx["amp"]], forcing_gt[:, cfg.forcing_idx["freq"]],
            forcing_gt[:, cfg.forcing_idx["phase"]], device=cfg.hw.device,
        )
        obs_stats = statistics.conditioning_rows(obs_stats, cfg.T_obs, forcing_gt.cpu())
    else:
        obs_stats = pipeline.gen_stats(x_spont_dim, None, cfg.dt_exp, None, None, None,
                                       device=cfg.hw.device, spontaneous_only=True)
        obs_stats = statistics.conditioning_rows(obs_stats, cfg.T_obs)
    return x_dim, obs_stats, t_dim


# ── Step 2: Prior construction ──────────────────────────────────────────────


_UNSAVED_PRIOR_MIN_RUNS = 100


def _saved_prior_fingerprints() -> dict:
    """``{fingerprint: filename}`` over every saved ND prior in Resources/Priors.

    Reads the stored ``means``/``weights`` directly rather than rebuilding a distribution: those are
    exactly the two tensors `_gmm_fingerprint` digests (file_manager.save_prior writes them), so the
    digests are comparable by construction, and this stays cheap enough to run before every training
    round -- the files are tens of kilobytes.
    """
    out = {}
    try:
        candidates = sorted(PRIOR_PATH.glob("*.pt"))
    except Exception:                        # noqa: BLE001 -- a missing directory is "none saved"
        return out
    for f in candidates:
        try:
            d = torch.load(str(f), map_location="cpu", weights_only=False)
            if not (isinstance(d, dict) and "means" in d and "weights" in d):
                continue
            h = hashlib.sha256()
            h.update(d["means"].detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
            h.update(d["weights"].detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
            out.setdefault(h.hexdigest()[:16], f.name)
        except Exception:                    # noqa: BLE001 -- a stale or foreign .pt is not our problem
            continue
    return out


def _assert_prior_is_saved(prior, n_runs: int, run_size: int) -> None:
    """Refuse to start a long generation run from a prior that exists only in memory.

    WHY THIS IS A HARD ERROR AND NOT A WARNING. `training_identity` fingerprints the prior's GMM, and
    that fingerprint names the checkpoint DIRECTORY. A prior that was fitted but never written to
    disk therefore produces a directory nobody can ever resolve again: the moment the process ends,
    the fingerprint is unreproducible, so the checkpoint it has been faithfully writing for hours can
    never be resumed by anything. It is not a degraded resume -- it is a guaranteed total loss of the
    run, discovered only when you try to recover from a crash.

    That is not hypothetical. On 2026-08-27 a run reached 884 committed batches under fingerprint
    bd307c079d14db0b, for which no file in Resources/Priors exists; those rows are unrecoverable, and
    a second run started minutes later under a third fingerprint. The cost of the check is reading a
    few 30 KB files; the cost of not having it is a day of simulation.

    Silent for short runs (see _UNSAVED_PRIOR_MIN_RUNS) and for anything with checkpointing off,
    which is where the tests and the smoke train live.
    """
    fp = _gmm_fingerprint(prior)
    if fp is None:
        return                               # no GMM to identify (a stub or hand-built prior)
    saved = _saved_prior_fingerprints()
    if fp in saved:
        return
    raise ValueError(
        f"This prior (fingerprint {fp}) is not saved anywhere in {PRIOR_PATH}. Training would "
        f"write a {n_runs}-batch checkpoint ({n_runs * run_size:,} rows) into a directory named "
        f"after that fingerprint -- and because the fingerprint is computed from the fitted GMM, "
        f"nothing could ever reproduce it once this process exits. The checkpoint would be "
        f"unresumable and a crash would cost the whole run.\n"
        f"Save the prior first (it is what SBC later draws theta* from in any case), then train. "
        f"Saved priors currently on disk: "
        f"{', '.join(sorted(saved.values())) if saved else '(none)'}.")








def training_identity(cfg: SimConfig, prior, run_size: int, n_runs: int) -> dict:
    """The config fields a training-data checkpoint must agree with before it can be resumed (C-11).

    PUBLIC (it was `_training_identity`) because the GUI's Posterior tab computes it to tell the user,
    before they press Train, whether their current Batches / rows-per-batch settings resume an
    existing checkpoint or silently start a new run -- §9.6's private-name-across-boundaries rule.

    Deliberately the SAME key names save_posterior_artifacts writes into the .rot.pt sidecar, plus the
    training geometry the sidecar has no reason to carry. One vocabulary for "which run is this",
    checked in two places, so a checkpoint and the posterior it eventually produces cannot describe
    different things.

    Everything here is known BEFORE the Fisher rotation runs, which it has to be: the digest names the
    directory the rotation's own V is stored in, so including V would make the naming circular.

    ``prior_fingerprint`` is the one entry that is NOT derivable from the config, and it is the one
    that matters most: the box bounds do not identify a prior. ``_gmm_fingerprint``'s own docstring
    says it -- *"two runs over the same box produce different fits"* -- and the training rows are
    drawn FROM the prior, so resuming under a rebuilt one would mix rows from two different
    distributions while every declared field still matched. It is also what makes "save your prior and
    reuse it" a guard rather than merely advice.
    """
    return {
        "format": "training-rows",
        "model": cfg.model,
        "prior_fingerprint": _gmm_fingerprint(prior),
        "mode": cfg.observation_mode,
        "param_keys": list(cfg.params_dict) + list(cfg.rescale_params),
        "nd_lows": [b[0] for _, b in cfg.params_dict.values()],
        "nd_highs": [b[1] for _, b in cfg.params_dict.values()],
        "rescale_lows": [b[0] for _, b in cfg.rescale_params.values()],
        "rescale_highs": [b[1] for _, b in cfg.rescale_params.values()],
        "log_params": resolved_log_params(cfg, log_params=_log_params_for(cfg)),
        "reparam_rotate": bool(cfg.reparam_rotate),
        "run_size": int(run_size),
        "n_runs": int(n_runs),
        "steady_idx": int(cfg.steady_idx),
        "dt_nd_min": float(cfg.dt_nd_min),
        "dt_exp": float(cfg.dt_exp),
        "t_min_exp": float(cfg.t_min_exp),
        "t_max_exp": float(cfg.t_max_exp),
        "t_scale_bounds": list(cfg.t_scale_bounds),
        "n_grid": int(cfg.t.shape[0]),
        "spontaneous_only": not cfg.has_forcing,
        # The FEATURE SET, not just its width. A checkpoint stores conditioning ROWS, so a run
        # whose summary block means something different must not resume onto them -- and width
        # alone would not catch a reordered or substituted flag set of equal length. Naming the
        # flags makes the digest change when the feature set does, which is what orphans the
        # pre-flag checkpoints and sends scripts/migrate_checkpoint_flags.py to a NEW directory
        # rather than splicing rows that mean two different things.
        "summary_flags": list(statistics.VALID_FLAG_LABELS),
        "chi_mode": bool(cfg.chi_mode),
        "chi_layout": config.CHI_LAYOUT if cfg.chi_mode else None,
        "chi_k_pad": cfg.chi_k_pad if cfg.chi_mode else None,
        "chi_elem_w": config.CHI_ELEM_W if cfg.chi_mode else None,
        "chi_f0": cfg.chi_f0 if cfg.chi_mode else None,
        "chi_freq_bounds": list(cfg.chi_freq_bounds) if cfg.chi_mode else None,
        "chi_max_cycles": float(cfg.chi_max_cycles) if cfg.chi_mode else None,
        "device": cfg.hw.device.type,
        "dtype": str(cfg.hw.dtype),
    }


PERSIST_OBSERVATIONS = True





def build_prior(cfg: SimConfig, choice: str | None, build_new: bool,
                *, save: bool = True, save_name: str | None = None, fig_sink=None,
                num_iterations: int | None = None, sweep_batch: int | None = None,
                max_sets: int | None = None, walk_step: float | None = None,
                stability_units: float | None = None,
                min_cluster_size: int | None = None,
                min_samples: int | None = None) -> tuple[Distribution, Distribution]:
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
    :param num_iterations: GLOBAL sweep rounds; None = config.PRIOR_SWEEP_ITERATIONS.
    :param sweep_batch: candidates per global round; None = config.PRIOR_SWEEP_BATCH (0 = follow the
                     hardware batch). ⚠ NOT a speed knob -- see the note at the constant (527 s at
                     batch 2048 against >70 min unfinished at 32; the sweep is iteration-bounded).
    :param max_sets: accepted sets that stop the LOCAL flood-fill; None = config.PRIOR_SWEEP_MAX_SETS.
    :param walk_step: flood-fill random-walk stride; None = config.PRIOR_SWEEP_STEP.
    :param stability_units: ND time units the stability screen integrates over; None =
                     config.STABILITY_SWEEP_ND_UNITS. This defines what "stable" MEANS, so changing
                     it changes the prior's support, not just how long the sweep takes.
    :param min_cluster_size: HDBSCAN floor on an island; None = config.PRIOR_CLUSTER_MIN_SIZE.
    :param min_samples: HDBSCAN density conservatism; None = config.PRIOR_CLUSTER_MIN_SAMPLES.
                     ⚠ These two are the CLUSTERING stage, not the sweep: HDBSCAN's label count is
                     handed straight to the GMM's n_components, so they set how many modes the
                     prior has. A prior with a different component count is a different prior.
    :return: A Distribution that can be sampled and scored.

    ⚠ WHY THESE ARE PARAMETERS AND NOT "JUST SET THE CONFIG CONSTANT" -- the same reason
    build_posterior's budget is: this module does `from .config import PRIOR_SWEEP_ITERATIONS, ...`,
    which SNAPSHOTS them at import, so a caller writing `config.PRIOR_SWEEP_ITERATIONS = 10` is a
    silent no-op and the sweep runs at 50 anyway with nothing to say otherwise.
    """
    # FIRST, before the ~9-minute stability sweep: is this chi configuration the one you meant? The
    # prior itself is chi-independent, so this is here purely to fail at the START of a session
    # rather than after its first expensive stage.
    _assert_chi_config_is_deliberate(cfg)

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
    force_prior = build_forcing_prior(cfg)

    # 2. Rescaling prior
    rescale_prior = build_rescale_prior(cfg)

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
    stab_units = STABILITY_SWEEP_ND_UNITS if stability_units is None else float(stability_units)
    n_stab_fine = int(stab_units / cfg.dt_nd_min)
    t_stab = cfg.t[:n_stab_fine]
    prior_segs = max(1, math.ceil(n_stab_fine / CHUNK_LEN))
    # The sweep's batch is its OWN knob (C-7), not the training batch. They were the same number,
    # and because the sweep is ITERATION-bounded rather than accept-bounded, shrinking that number for
    # a cheap run made the prior worse WITHOUT making it faster -- 527 s at batch 2048 against >70 min
    # and unfinished at batch 32. PRIOR_SWEEP_BATCH = 0 keeps the historical behaviour (follow the
    # hardware batch), which is still what a real run wants; see config for when to set it.
    sweep_batch = (PRIOR_SWEEP_BATCH if sweep_batch is None else int(sweep_batch)) or cfg.hw.batch_size
    n_iter = PRIOR_SWEEP_ITERATIONS if num_iterations is None else int(num_iterations)
    if n_iter < 1:
        raise ValueError(f"num_iterations must be at least 1, got {n_iter}")
    nd_prior = pipeline.gen_prior(
        model=cfg.model, t=t_stab,
        global_batch_size=sweep_batch,
        local_batch_size=(sweep_batch // 2),
        segs=prior_segs,
        prior_bounds=cfg.nd_params_bounds,
        state_dep_drift=cfg.state_dep_drift,
        num_iterations=n_iter,
        n_max=max_sets, step=walk_step,
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        # geometric/log box on the ND params that asked for one: a user model's own per-parameter
        # choice, else config.REPARAM_LOG_PARAMS. See _log_params_for.
        log_mask=nd_log_mask(cfg, log_params=_log_params_for(cfg)),
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


def build_rescale_prior(cfg: SimConfig) -> Distribution:
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


def build_forcing_prior(cfg: SimConfig) -> Distribution:
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
    num_runs: int | None = None, run_size_cap: int | None = None,
    truncation=None, x_obs_digest: str | None = None,
    hidden_features: int | None = None, num_transforms: int | None = None,
    learning_rate: float | None = None, stop_after_epochs: int | None = None,
    fisher_m: int | None = None, fisher_dz: float | None = None,
    fisher_points: int | None = None,
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
    :param num_runs: Training BATCHES to simulate; None (the default) = config.TRAINING_NUM_RUNS,
                     which is the CLI's behaviour and what every script and test gets.
    :param run_size_cap: CEILING on simulations per batch, 0 = follow the hardware default; None =
                     config.TRAINING_RUN_SIZE.
    :param truncation: a ``SBI.truncate.TruncationRegion`` to restrict the PRIOR to (TSNPE round 2+).
                     None = ordinary amortized NPE. The resulting artifact is marked NON-AMORTIZED in
                     its sidecar and the load path refuses it for general inference.
    :param x_obs_digest: the observation the region was drawn around (``observation_digest``), so the
                     artifact records what it is valid near.
    :param hidden_features: flow width per transform; None = config.NSF_HIDDEN_FEATURES.
    :param num_transforms: flow depth; None = config.NSF_NUM_TRANSFORMS.
    :param learning_rate: Adam LR; None = config.TRAINING_LEARNING_RATE.
    :param stop_after_epochs: early-stopping patience; None = config.TRAINING_STOP_AFTER_EPOCHS.
    :param fisher_m: ensemble per latent perturbation for the rotation; None =
                     config.REPARAM_FISHER_M.
    :param fisher_dz: latent central-difference step; None = config.REPARAM_FISHER_DZ.
    :param fisher_points: operating points the Fisher is averaged over; None =
                     config.REPARAM_FISHER_POINTS. n_points=1 is GT-only, which re-correlates
                     off-GT -- averaging is what makes ONE linear rotation valid prior-wide.
                     ⚠ A RESUMED run reuses the checkpoint's stored V and ignores all three
                     : V is not reproducible across processes, so a fresh one would
                     put the reused rows in a different coordinate than their stored targets.

    ⚠ THESE FOUR ARE WHAT A COMPLETE C-11 CHECKPOINT IS FOR. Its own docstring says a finished
    checkpoint "is a cache of the whole simulation run, so you can retrain the flow at a different
    capacity/learning rate without re-simulating" -- and until 2026-08-27 there was no way to do that
    without editing config.py. Re-trying capacity costs ~46 h against ~57 h for a full run.
    ⚠ And note the 2026-08-25 characterisation ruled out more capacity on the BROKEN conditioning
    (a clean loss plateau well before the best epoch); that verdict is worth re-testing on the
    repaired conditioning, not inherited.

    ⚠ WHY THESE ARE PARAMETERS AND NOT "JUST SET THE CONFIG CONSTANT". This module does
    `from .config import TRAINING_NUM_RUNS, TRAINING_RUN_SIZE`, which SNAPSHOTS both at import -- so a
    caller writing `config.TRAINING_NUM_RUNS = 2000` is a silent no-op and the run uses 5000 anyway,
    with nothing to say otherwise. (`scripts/smoke_train.py` gets this right by assigning to
    `orchestrator.TRAINING_NUM_RUNS`; a GUI mutating a module global per run would also leak across
    runs.) Passing them keeps the CLI byte-identical and makes the override explicit and testable.

    ⚠ AND THEY ARE NOT INTERCHANGEABLE BUDGET KNOBS. Each batch shares ONE Sobol (t_scale_k, T_k)
    pair, overridden for every row in it -- so `num_runs` is the (t_scale, T) DIVERSITY count and the
    run size is rows per operating point. 5000x2048 and 10000x1024 have equal totals and different
    statistics (calibration has the same property). Batch WIDTH is also nearly free in
    wall-clock -- the solver is kernel-launch-bound; measured 7.37 s at 2048 against 7.74 s at 1024 --
    so narrowing it does not speed anything up, it trades training rows for peak VRAM about 1:1.
    """
    # Tier 1: announce the DERIVED force scale before the first simulation, for the
    # same reason the chi banner exists -- a training distribution that changed silently is what cost
    # the 2026-08-19 run. Reports rather than refuses: whether ~1e4 pN is reasonable is a judgement
    # about the preparation, not something a threshold in this file should decide.
    if derived.uses_derived_f_scale(cfg.rescale_idx):
        try:
            _s = prior.sample((4096,)).to("cpu")
            print(derived.describe_derived_f_scale(
                _s[:, :len(cfg.params_dict)], _s[:, len(cfg.params_dict):],
                cfg.rescale_idx, cfg.nd_idx, cfg.k_b_cell,
                chi_f0=cfg.chi_f0 if cfg.chi_mode else None), flush=True)
        except Exception as _e:                  # noqa: BLE001 -- a banner must never stop a run
            print(f"[tier1] could not describe the derived f_scale: {_e}", flush=True)

    # Resolved before anything reads them: both are part of the checkpoint identity below.
    n_runs = TRAINING_NUM_RUNS if num_runs is None else int(num_runs)
    size_cap = TRAINING_RUN_SIZE if run_size_cap is None else int(run_size_cap)
    if n_runs < 1:
        raise ValueError(f"num_runs must be at least 1, got {n_runs}")
    if size_cap < 0:
        raise ValueError(
            f"run_size_cap must be >= 0 (0 = follow the hardware default), got {size_cap}")
    # Above BOTH branches, and the load branch is the subtle half. _assert_mode_matches compares the
    # posterior against cfg -- so a STALE cfg loading the posterior trained under that same stale cfg
    # agrees with itself and says nothing, while every inference it serves is at a retired band. This
    # check is against config.py, which is the only party to the comparison that cannot go stale.
    _assert_chi_config_is_deliberate(cfg)

    # The TRAINING bijection. Its log box must be the one gen_prior fitted the latent GMM in, hence
    # the same resolver the guard below and the sidecar use (_log_params_for).
    T = build_inferred_bijection(cfg, log_params=_log_params_for(cfg))

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
        _assert_amortization_understood(choice)
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
        _want = nd_log_mask(cfg, log_params=_log_params_for(cfg)).to(_nd_box.log_mask.device)
        if not torch.equal(_nd_box.log_mask, _want):
            _src = ("this user model's per-parameter box settings"
                    if _log_params_for(cfg) is not None else "config.REPARAM_LOG_PARAMS")
            raise ValueError(
                f"Loaded ND prior's log-box mask does not match {_src} "
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
    # The training batch's OWN ceiling -- not hw.batch_size, and deliberately not PRIOR_SWEEP_BATCH's
    # twin. A CEILING rather than a replacement, because smoke_train.py and three pipeline tests shrink
    # runs by writing cfg.hw.batch_size directly and a replacing knob would silently override them.
    # Announced when it binds: a cap that changes the shape of a multi-day run is not allowed to be
    # silent, and the printed row count is also the check that TRAINING_NUM_RUNS was moved to match.
    # Resolved HERE, above the rotation, because the checkpoint's identity includes it.
    run_size = cfg.hw.batch_size
    if size_cap and size_cap < run_size:
        print(f"Training batch capped at {size_cap} (hardware default {run_size}) — "
              f"{n_runs} batches x {size_cap} = {n_runs * size_cap:,} training rows.")
        run_size = size_cap
    if n_runs != TRAINING_NUM_RUNS:
        # Announced for the same reason the cap is: a batch count that changes the shape (and the
        # (t_scale, T) diversity) of a multi-day run is not allowed to be silent.
        print(f"Training batch COUNT overridden: {n_runs} batches (config default "
              f"{TRAINING_NUM_RUNS}) — {n_runs * run_size:,} training rows.")

    # --- training-data checkpoint (C-11): resolved BEFORE the rotation, because a resume REUSES V ---
    # This ordering is the whole reason the resume works with rotation ON, which is how the retrain is
    # specified to run. `build_latent_fisher_rotation` seeds its noise under fork_rng but draws its
    # OPERATING POINTS from the caller's global RNG (decorrelate's z_med/z_samp), which nothing seeds
    # -- so a restarted process computes a DIFFERENT V, hence a different T_train, hence different
    # LATENT targets for every batch after the seam, silently mixed with the pre-crash rows. Trap X10.
    # Reusing the stored V also skips the Fisher entirely on a resume, which is the single largest
    # pre-training cost.
    ckpt_dir = ckpt_resumed = None
    if TRAINING_CHECKPOINT_EVERY and train_new:
        # BEFORE the digest is computed, because the digest is the thing an unsaved prior poisons.
        if n_runs >= _UNSAVED_PRIOR_MIN_RUNS:
            _assert_prior_is_saved(prior, n_runs, run_size)
        ident = training_identity(cfg, prior, run_size, n_runs)
        ckpt_dir = training_checkpoint.resolve_dir(ident)
        _st = training_checkpoint.peek(ckpt_dir)
        # A COMPLETE checkpoint counts too, and deliberately so. Its rows are already expressed in the
        # V they were generated under, so recomputing V here would make gen_training_data's probe
        # check refuse the very rows it is about to reuse -- turning the "died during NN training,
        # don't re-simulate for days" path into a hard failure. Any committed batch pins V.
        if _st and _st.get("batches_done"):
            ckpt_resumed = training_checkpoint.read_header(ckpt_dir)

    rotate = cfg.reparam_rotate
    # Only the freshly-computed branch below knows the eigenvalues; a resumed checkpoint carries V but
    # not them, and an unrotated run has no Fisher at all. None is recorded honestly in the sidecar.
    fisher_evals = None
    if ckpt_resumed is not None and rotate:
        # Rehomed onto this run's device/dtype. The checkpoint stores V on the CPU so it is portable,
        # but build_latent_fisher_rotation returns it on cfg.hw.device -- and OrthogonalTransform does
        # `x @ M`, which is a hard device error, not a silent promotion. Without this the FIRST GPU
        # resume with rotation ON would crash, i.e. exactly the run this feature exists to rescue.
        # Same defect the bijection probe had; found by looking for its siblings after the smoke train
        # surfaced the first one.
        V = ckpt_resumed.get("V")
        if V is not None:
            V = V.to(device=cfg.hw.device, dtype=cfg.hw.dtype)
        _done = _st["batches_done"]
        print(f"Reusing the Fisher rotation stored with the training checkpoint "
              f"({_done}/{n_runs} batches"
              f"{' — COMPLETE, so generation will be skipped' if _st.get('complete') else ''}) — NOT "
              f"recomputing it: the rotation's operating points are not reproducible across "
              f"processes, so a fresh V would put the reused rows in a different coordinate than the "
              f"targets stored beside them.")
        T_train = build_rotated_bijection(T, V) if V is not None else T
        train_prior = RotatedLatentPrior(latent_inferred_prior, V) if V is not None else latent_inferred_prior
    elif rotate:
        print("Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)...")
        # Average the Fisher over the prior (not just GT) so the linear rotation is valid prior-wide.
        # GT-free: the rotation anchors on the prior median with a representative drive (force_prior).
        V, fisher_evals = decorrelate.build_latent_fisher_rotation(
            cfg, T, latent_prior=latent_inferred_prior, force_prior=force_prior, with_values=True,
            m=fisher_m, dz=fisher_dz, n_points=fisher_points)
        # The eigenvalues ride into the sidecar with V. Without them the saved rotation only says
        # WHICH direction is least constrained, never BY HOW MUCH -- and recovering them afterwards
        # costs a full Fisher re-run. See scripts/identifiability.py.
        _spread = float(fisher_evals[0] / fisher_evals[-1]) if float(fisher_evals[-1]) > 0 else float("inf")
        print(f"[fisher] eigenvalue spread (best/worst direction): {_spread:.3g}", flush=True)
        T_train = build_rotated_bijection(T, V)
        train_prior = RotatedLatentPrior(latent_inferred_prior, V)
    else:
        V, T_train, train_prior = None, T, latent_inferred_prior

    # ── TSNPE ─────────────────────────────────────────────────────────────────────────────────────
    # ⚠ THE PROPOSAL IS THE TRUNCATED PRIOR, NEVER THE POSTERIOR. Wrapped around `train_prior`, which
    # is the ROTATED latent prior when the rotation is on -- so the region's axes are the flow's own
    # latent axes, i.e. V's columns, i.e. the Fisher directions. Wrapping the UNROTATED prior instead
    # would silently truncate along physical-ish axes and cut the flat directions on noise, which is
    # exactly what guardrail 3 exists to prevent.
    #
    # No proposal correction is applied, and that is correct rather than an omission: truncation is a
    # RESTRICTION, not a reweighting, which is the property that distinguishes TSNPE from SNPE-A/B/C.
    if truncation is not None:
        train_prior = truncate.TruncatedLatentPrior(train_prior, truncation)
        print(f"[tsnpe] training on the PRIOR RESTRICTED to {truncation!r}", flush=True)
        print(f"[tsnpe] this artifact will be marked NON-AMORTIZED; it is valid only near the "
              f"observation its region was drawn around (digest {x_obs_digest}).", flush=True)

    training_params = pipeline.TrainingPlan(
        model=cfg.model,
        prior=train_prior,                         # <-- latent (rotated if REPARAM_ROTATE)
        t=cfg.t,
        run_size=run_size,
        num_runs=n_runs,
        steady_idx=cfg.steady_idx,
        dt_nd_min=cfg.dt_nd_min,
        dt_exp=cfg.dt_exp,
        t_min_exp=cfg.t_min_exp,
        t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds,
        state_dep_drift=cfg.state_dep_drift,
        spontaneous_only=not cfg.has_forcing,
        chi_mode=cfg.chi_mode,
        # No "chi_n_freqs" here on purpose. It is the count an OBSERVATION supplies; training draws K
        # per batch over [CHI_K_MIN_TRAIN, chi_k_pad] and subsets again per row, which is what makes
        # one posterior serve any probe count. It used to be threaded in and silently ignored -- an
        # invitation to "fix" gen_training_data into honouring it and destroy exactly that property.
        chi_f0=cfg.chi_f0,
        chi_freq_bounds=cfg.chi_freq_bounds,
        chi_k_pad=cfg.chi_k_pad,
        chi_max_cycles=cfg.chi_max_cycles,
        # _observation_inits, NOT cfg.inits_tensor: training is ground-truth-free, so a config built from
        # bounds alone (no cell loaded) has an empty inits_dict and cfg.inits_tensor would RAISE. The
        # fallback synthesizes the same model-default inits the training loop itself uses.
        n_vars=_observation_inits(cfg).shape[-1],
        # Tier 1. Both are None-safe downstream and are simply ignored by a box
        # that declares f_scale, so this costs pre-tier-1 runs nothing.
        nd_idx=cfg.tier1_args[0],
        k_b_cell=cfg.tier1_args[1],
        dtype=cfg.hw.dtype,
        device=cfg.hw.device,
    )
    if ckpt_dir is not None:
        # V and the probe go in AFTER the rotation, so a fresh run stores the V it just computed and
        # a resumed one stores nothing new (create() is not called on a resume).
        training_params.checkpoint = {
            "dir": ckpt_dir, "identity": ident,
            # device= is load-bearing: T_train holds the rotation V on cfg.hw.device, and a CPU grid
            # into a CUDA matmul is a hard RuntimeError inside build_posterior.
            "probe": training_checkpoint.bijection_probe(
                T_train, len(cfg.params_dict) + len(cfg.rescale_params), device=cfg.hw.device),
            "V": V, "every": TRAINING_CHECKPOINT_EVERY, "resume": "auto",
        }

    # Conditioning layout is [S(x) | log(T) | forcing]. log(T) rides with the summary
    # pathway, so input_dim (the leading summary block) includes it; only the forcing
    # params form the separate forcing pathway.
    # chi-mode routes the padded probe SET through the EmbeddedNet's second pathway in place of the
    # single-frequency forcing block, as a permutation-invariant set encoder.
    forcing_dim = expected_forcing_dim(cfg)        # shared with the sidecar + the load-side mode guard
    from .SBI.statistics import SUMMARY_WIDTH
    input_dim = SUMMARY_WIDTH + 1            # n_summary_stats + log(T); observation-independent

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
        hidden_features=NSF_HIDDEN_FEATURES if hidden_features is None else int(hidden_features),
        num_transforms=NSF_NUM_TRANSFORMS if num_transforms is None else int(num_transforms),
        num_bins=NSF_NUM_BINS,
        learning_rate=TRAINING_LEARNING_RATE if learning_rate is None else float(learning_rate),
        stop_after_epochs=(TRAINING_STOP_AFTER_EPOCHS if stop_after_epochs is None
                           else int(stop_after_epochs)),
        max_num_epochs=TRAINING_MAX_NUM_EPOCHS, show_train_summary=TRAINING_SHOW_SUMMARY,
        batch_size=TRAINING_BATCH_SIZE, device=cfg.hw.device,
    )

    if save:
        name = save_name if save_name is not None else cli.prompt_save_name("posterior")
        save_posterior_artifacts(name, posterior_latent, V, pos_diagnostics, cfg,
                                 fisher_eigenvalues=fisher_evals, truncation=truncation,
                                 x_obs_digest=x_obs_digest)

    # Display the training-loss curve (a GUI embeds it via the sink; the CLI historically saved it to
    # PNG without showing, so with no sink we do nothing here to preserve that behavior).
    if fig_sink is not None and pos_diagnostics is not None and pos_diagnostics.get("validation_loss"):
        fig_loss = visualizers.plot_training_loss(pos_diagnostics)
        if fig_loss is not None:
            fig_sink("Training loss", fig_loss)

    if truncation is not None and hasattr(train_prior, "acceptance_rate"):
        # GUARDRAIL 5: the fraction of prior mass this round threw away, measured rather than
        # assumed. Deleted support is a one-way ratchet -- no later round can recover it -- so the
        # number belongs in the run log next to the artifact it produced.
        _acc = train_prior.acceptance_rate
        print(f"[tsnpe] the truncation kept {_acc:.3%} of the prior's mass "
              f"({1 - _acc:.3%} of the support is now permanently unavailable to later rounds).",
              flush=True)

    assert isinstance(posterior_latent, DirectPosterior)
    return TransformedPosterior(posterior_latent, T_train), pos_diagnostics


def observation_digest(x_obs: torch.Tensor) -> str:
    """Stable 16-hex digest of a conditioning vector. Same shape as _gmm_fingerprint, and exact
    rather than tolerance-based for the same reason: this answers "is this the same observation",
    not "are these observations similar"."""
    b = x_obs.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def save_observation(cfg: SimConfig, x_obs: torch.Tensor, *, tag: str = "") -> tuple:
    """Persist the observation an inference was actually run against. Returns (path, digest).

    SECTION 11.6 GUARDRAIL 1, and the timing is the whole point. Amortized NPE has NO observation
    when the posterior is saved -- which is exactly why ``default_x`` is None on
    ``posterior_08232026``, and why the posterior behind that run's figures cannot be re-sampled from
    the artifacts alone. The fix therefore belongs at INFERENCE time, not at save time; bolting it
    onto save_posterior_artifacts would record a None.

    TSNPE then refuses to build a truncation region unless the stored digest matches the dataset
    currently loaded -- a region drawn around one recording and applied to another deletes prior
    support on the strength of the wrong data, and truncation is a one-way ratchet.
    """
    dig = observation_digest(x_obs)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    name = f"obs_{stamp}_{dig}{('_' + tag) if tag else ''}.pt"
    path = OBSERVATION_PATH / name
    file_manager.atomic_torch_save({
        "x_obs": x_obs.detach().cpu(),
        "digest": dig,
        "mode": cfg.observation_mode,
        "input_dim": statistics.SUMMARY_WIDTH + 1,
        "forcing_dim": expected_forcing_dim(cfg),
        "param_keys": list(cfg.params_dict) + list(cfg.rescale_params),
        "chi_k_pad": cfg.chi_k_pad if cfg.chi_mode else None,
        "chi_n_freqs": cfg.chi_n_freqs if cfg.chi_mode else None,
        "model": cfg.model,
    }, path)
    return path, dig


def load_observation(path) -> dict:
    """Read a record written by :func:`save_observation`."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def build_truncation_region(posterior, obs_record: dict, x_obs: torch.Tensor, *,
                            n_directions: int = None, level: float = None):
    """The TSNPE region for the NEXT round, with guardrail 1 enforced.

    ⚠ REFUSES unless the observation currently loaded is BITWISE the one the stored record describes.
    A region drawn around one recording and applied to another deletes prior support on the strength
    of the wrong data -- and truncation is a one-way ratchet, so a later round cannot undo it. This is
    the check that makes "persist x_obs at inference time" worth doing at all.

    :param posterior: the TransformedPosterior (or DirectPosterior) to draw the region from.
    :param obs_record: the dict from :func:`load_observation`.
    :param x_obs: the conditioning vector currently loaded.
    """
    want, got = obs_record.get("digest"), observation_digest(x_obs)
    if want != got:
        raise ValueError(
            f"The stored observation (digest {want}) is not the one currently loaded (digest {got}), "
            f"so a truncation region built from it would delete prior support on the strength of a "
            f"DIFFERENT recording -- permanently, because truncation is one-way. Re-run inference on "
            f"this dataset first so its observation is the one on record.")
    latent = getattr(posterior, "latent", posterior)
    return truncate.region_from_posterior(
        latent, x_obs,
        n_directions=truncate.DEFAULT_N_DIRECTIONS if n_directions is None else int(n_directions),
        level=truncate.DEFAULT_HPD if level is None else float(level))


def _refuse_to_orphan_a_checkpoint(name: str, nd_prior) -> None:
    """Refuse to overwrite a prior file that an existing checkpoint is the only copy of.

    ⚠ THIS IS HOW 3989 BATCHES WERE LOST. A checkpoint's directory is named after a digest of the
    prior's fitted GMM, so the prior FILE is the only thing that can reproduce it. On 2026-08-28
    ``prior_08282026.pt`` was overwritten, under the same name, with a different distribution -- and
    the 3989-batch run that had been training against the old contents for six and a half hours
    became unreachable. No error, no warning, one click. The same mechanism cost 884 batches the day
    before.

    Only fires when ALL of: the file exists, its current contents differ from what is being written,
    and a COMMITTED checkpoint depends on those contents. Saving under a new name, re-saving the
    same distribution, or overwriting a prior nothing references are all untouched.
    """
    path = PRIOR_PATH / (name + ".pt")
    if not path.exists():
        return
    try:
        existing = torch.load(str(path), map_location="cpu", weights_only=False)
        if not (isinstance(existing, dict) and "means" in existing and "weights" in existing):
            return
        h = hashlib.sha256()
        h.update(existing["means"].detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
        h.update(existing["weights"].detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
        old_fp = h.hexdigest()[:16]
        new_fp = _gmm_fingerprint(nd_prior)
    except Exception:                        # noqa: BLE001 -- an unreadable existing file is not ours to judge
        return
    if new_fp is None or old_fp == new_fp:
        return                               # same distribution: overwriting changes nothing
    users = training_checkpoint.checkpoints_using_prior(old_fp)
    if not users:
        return
    listed = "; ".join(f"{n} ({b:,} batches)" for n, b in users)
    raise ValueError(
        f"Refusing to overwrite {path.name}: it is the only copy of the prior that "
        f"{len(users)} checkpoint(s) were generated against -- {listed}.\n"
        f"  A checkpoint's directory is named after a digest of the prior's fitted GMM, so replacing "
        f"this file makes those runs UNRESUMABLE -- the simulation is still on disk but nothing can "
        f"ever match it again. That is how 3989 batches (6.5 h) were lost on 2026-08-28.\n"
        f"  Save under a different name. If you really mean to discard those checkpoints, delete "
        f"them first and the save will go through.")


def save_prior_artifacts(name: str, nd_prior, cfg: SimConfig, *, fig_sink=None) -> None:
    """
    Persist an ND prior GMM to Resources/Priors/<name>.pt and its corner PNG to Resources/Plots.
    Shared by build_prior (CLI, save=True) and a GUI's explicit "Save prior" control. With no
    fig_sink the corner plot falls back to plt.show() (a no-op under the GUI's Agg backend).

    The .pt write is ATOMIC (file_manager.save_mix_dist -> atomic_torch_save). The PNG beside it is
    not, deliberately: a half-written PNG is loud and free to regenerate, whereas a half-written prior
    is the file a checkpointed resume fingerprints and SBC later draws theta* from.
    """
    _refuse_to_orphan_a_checkpoint(name, nd_prior)
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
    from .SBI.statistics import SUMMARY_WIDTH
    input_dim = (SUMMARY_WIDTH + 1) if input_dim is None else input_dim
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
    from .SBI.statistics import SUMMARY_WIDTH
    summary_w = SUMMARY_WIDTH + 1
    detail = f" (K={k})" if k else ""
    raise ValueError(
        f"Posterior '{choice}' was trained in {mode.upper()} mode{detail} with a forcing/chi block of "
        f"{forcing_dim} features, but this config is {want_mode.upper()} mode expecting {want_dim}. "
        f"Conditioning widths {summary_w + forcing_dim} vs {summary_w + want_dim} are incompatible. "
        f"Pick a posterior trained in this mode, or rebuild the config to match "
        f"(the chi toggle and the bounds file's Forcing section are what select the mode)."
    )


def save_posterior_artifacts(name: str, posterior_latent, V, diagnostics: dict | None, cfg: SimConfig,
                            fisher_eigenvalues=None, truncation=None, x_obs_digest=None) -> None:
    """
    Persist a trained posterior and its companions: <name>.pt (raw latent DirectPosterior), the
    <name>.rot.pt reparam sidecar (rotation V + log params, when either is active), and the
    <name>.loss.npz curve + <name>_loss.png. Shared by build_posterior (CLI) and a GUI's explicit
    "Save posterior" control.

    Every write here is ATOMIC (file_manager._atomic_write: sibling tmp -> fsync -> os.replace). These
    are one-shot end-of-run writes, so the window is narrow -- but the ``.pt`` and its ``.rot.pt`` are
    the product of a multi-day run, the GUI's Save button can be pressed twice over the same name, and
    a torn artifact does not announce itself: it is an unpickling error hours later, or a sidecar that
    loads with half its keys and silently decodes every latent sample through a default box.
    """
    file_manager.atomic_torch_save(posterior_latent, POSTERIOR_PATH / (name + ".pt"))
    # Self-describing sidecar so eval reconstructs the exact training box (log-mask + rotation V) AND
    # knows which observation mode produced this posterior.
    #
    # Written UNCONDITIONALLY. It used to be skipped when V was None and no log params were active --
    # which is exactly the chi case (chi is deliberately unrotated, and REPARAM_LOG_PARAMS is []), so
    # a multi-hour chi posterior landed on disk BYTE-INDISTINGUISHABLE from the legacy forced
    # posteriors sitting beside it, with nothing on the load path checking width or mode. A missing
    # sidecar still means "pre-reparam, linear box" for the old artifacts; from here on, absence is
    # only ever a legacy signal, never an ambiguous new one.
    # Same resolver as build_prior/build_posterior, so what the sidecar records is what the flow was
    # trained in. load_eval_bijection rebuilds the box from THIS list, not from the live config, so a
    # divergence here would be invisible until the posterior evaluated in the wrong coordinate.
    log_params_used = resolved_log_params(cfg, log_params=_log_params_for(cfg))
    from .SBI.statistics import SUMMARY_WIDTH
    file_manager.atomic_torch_save({
        "V": V,
        # GUARDRAIL 2. An amortized posterior serves any observation; a TRUNCATED one
        # is valid only near the observation its region was drawn around, and outside it the flow has
        # never seen a single training row. With both workflows live the two sit side by side in one
        # ArtifactPicker -- the same class of confusion as the retired-band posterior that already
        # cost a five-day run -- so the distinction is recorded rather than left to a filename.
        "amortized": truncation is None,
        "truncation": None if truncation is None else truncation.to_dict(),
        "x_obs_digest": x_obs_digest,
        # The eigenvalues V's columns were sorted by, descending. None when the rotation came from a
        # resumed training checkpoint (which stores V but not them) or when the rotation is off.
        # Without these the sidecar records an ORDERING of directions but no scale, and the scale is
        # the question -- see reparam.fisher_eigenbasis and scripts/identifiability.py.
        "fisher_eigenvalues": (fisher_eigenvalues.detach().cpu()
                               if hasattr(fisher_eigenvalues, "detach") else fisher_eigenvalues),
        "log_params": log_params_used,
        # Observation mode + conditioning geometry -- see SBI/reparam.posterior_mode, which prefers
        # these over decoding the trained net (that decoding cannot distinguish chi at K=2 from a
        # hypothetical 6-parameter drive).
        "mode": cfg.observation_mode,
        "input_dim": SUMMARY_WIDTH + 1,
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
    }, POSTERIOR_PATH / (name + ".rot.pt"))
    # Loss curve: persisted so the convergence check is reproducible (sbi keeps it only in the trainer).
    if diagnostics is not None and diagnostics.get("validation_loss"):
        file_manager.atomic_savez(
            PLOT_PATH / (name + ".loss.npz"),
            dict(
                training_loss=np.asarray(diagnostics.get("training_loss", []), dtype=float),
                validation_loss=np.asarray(diagnostics.get("validation_loss", []), dtype=float),
                best_validation_loss=float(diagnostics.get("best_validation_loss") or float("nan")),
                epochs_trained=int(diagnostics.get("epochs_trained") or -1),
                stop_after_epochs=int(diagnostics.get("stop_after_epochs") or -1),
            ),
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
def validate_calibration(cfg: SimConfig, posterior: DirectPosterior | TransformedPosterior,
                         inferred_prior: Distribution, force_prior: Distribution,
                         *, fig_sink=None, n_cal: int | None = None,
                         cal_n_scales: int | None = None) -> None:
    """
    Data-free posterior calibration: SBC (Talts 2018, marginals) + expected coverage (TARP, Lemos
    2023). Both draw their calibration set from the PRIOR (theta_star ~ prior, x_cal simulated), so
    this runs right after training with no chosen observation.

    :param inferred_prior: the actual training prior (ND x rescale product prior) — SBC draws
                           theta_star from it, not from the posterior.
    :param n_cal: calibration datasets for SBC/TARP; None = config.SBC_N_CAL.
    :param cal_n_scales: (t_scale, T) operating points the calibration set is spread over; None =
                     config.CAL_N_SCALES.
                     ⚠ TRAP X5: this is `t_scale`'s EFFECTIVE SAMPLE SIZE, not a speed dial. Lowering
                     it is a DIFFERENT measurement, not a faster one -- "SBC flat on all 13" is
                     strong for 11 of them and materially weaker for `t_scale` and anything the probe
                     design controls, and this number is why.
    """
    _assert_prior_used_matches_posterior(posterior, inferred_prior, "SBC/TARP calibration")
    t = cfg.t
    device = cfg.hw.device
    dtype = cfg.hw.dtype
    # Posterior's actual transform (rotated if REPARAM_ROTATE) so the cal prior + theta_transform match.
    T = (posterior.T if isinstance(posterior, TransformedPosterior)
         else build_inferred_bijection(cfg, log_params=_log_params_for(cfg)))

    # Critical: draw theta_star from the PRIOR (not the posterior) for valid SBC.
    val_latent_prior = _build_latent_prior_for_validation(cfg, inferred_prior)
    # If the posterior uses a decorrelating rotation, rotate the calibration prior to match it.
    if hasattr(T, "parts") and len(T.parts) and isinstance(T.parts[0], OrthogonalTransform):
        val_latent_prior = RotatedLatentPrior(val_latent_prior, T.parts[0].M.transpose(-1, -2))
    x_cal, theta_star = analysis.gen_cal_data(
        model=cfg.model, prior=val_latent_prior,
        forcing_prior=force_prior,
        t=t, steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min,
        n_cal=SBC_N_CAL if n_cal is None else int(n_cal),
        cal_n_scales=cal_n_scales,
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
        # CHI_K_FIXED, run per stratum (a pooled SBC over a mixture of counts can be flat while
        # each count is miscalibrated in compensating directions).
        chi_k_fixed=None,
        n_vars=_observation_inits(cfg).shape[-1],
        nd_idx=cfg.tier1_args[0], k_b_cell=cfg.tier1_args[1],
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

    # --- Informativeness --------------------------------------------------------------------------
    # Everything above measures CALIBRATION, and a posterior that simply returns the prior passes all
    # of it. This is the scalar that says whether the run learned anything, on the calibration set
    # just simulated, so it costs nothing extra. Reported alongside rather than instead: a run wants
    # both numbers, and the pair is what distinguishes "honest and useful" from "honest and vacuous".
    try:
        info = analysis.informativeness(
            posterior, theta_star_dev, x_cal_dev, inferred_prior,
            param_names=list(cfg.params_dict) + list(cfg.rescale_params))
        print(analysis.describe_informativeness(info))
    except Exception as _e:                      # noqa: BLE001
        # A diagnostic must never be the thing that loses a multi-day run's other results. The
        # sample-based decomposition in particular reaches into the posterior's transform stack.
        warnings.warn(f"informativeness could not be computed ({type(_e).__name__}: {_e}); the "
                      f"calibration results above are unaffected.", stacklevel=2)


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

    # GUARDRAIL 1: record the observation this inference actually ran against, here,
    # where it first exists. Written before the figures, so an interrupted or crashed inference still
    # leaves behind the thing needed to reproduce or extend it.
    if PERSIST_OBSERVATIONS:
        try:
            _obs_path, _obs_dig = save_observation(cfg, obs_stats)
            print(f"[obs] observation persisted as {_obs_path.name} (digest {_obs_dig})", flush=True)
        except Exception as _e:                  # noqa: BLE001 -- never lose the inference over this
            warnings.warn(f"could not persist the observation ({type(_e).__name__}: {_e}); "
                          f"inference continues, but TSNPE will have nothing to key on.",
                          stacklevel=2)

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
    # TIER 1 (a box that declares T instead of f_scale): substitute the DERIVED f_scale into T's column before anything
    # simulates. A no-op for a box that declares f_scale. `sim_rescale_idx` is what the force
    # builders and gen_chi_raw must then be given -- handed the INFERRED index they would not
    # find 'f_scale', would fall into the Hopf-style x_scale/t_scale branch, and would drive
    # at a silently wrong amplitude.
    samples_rescale = derived.to_sim_rescale(samples_nd, samples_rescale, cfg.rescale_idx,
                                            *cfg.tier1_args)
    n_samples = samples.shape[0]
    # Same for all samples. Prefer the length generate_observations actually resolved (post
    # cost-ceiling clip); fall back to the formula only on the experimental paths, which never call
    # generate_observations and take their length from the recording itself.
    N_points_obs = cfg.n_obs if cfg.n_obs is not None else int(cfg.T_obs / cfg.dt_exp)

    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
    forcing_gt_expanded = forcing_gt.expand(n_samples, -1)  # (n_samples, n_forcing); empty if no forcing
    n_vars = inits.shape[-1]
    n_force_ch = forcing.n_force_channels(cfg.model, cfg.forcing_idx, n_vars)

    (x_dim_sorted, x_spont_sorted, chi_block_sorted,
     inv_sort_idx) = ppc.simulate_ppc_bins(cfg, t, inits, samples_nd, samples_rescale,
                                           forcing_gt, N_points_obs, expected_forcing_dim(cfg),
                                           dtype, device)

    # Restore original sample order
    x_spont = x_spont_sorted[inv_sort_idx]
    # Layout [S | log(T) | forcing|chi] — must match the observation in generate_observations.
    if cfg.chi_mode:
        x_dim = x_spont                                 # PPC "sample trajectories" = passive spontaneous trace
        sim_stats = pipeline.gen_stats(x_spont, None, cfg.dt_exp, None, None, None,
                                       device=device, spontaneous_only=True)
        sim_stats = statistics.conditioning_rows(sim_stats, T_obs, chi_block_sorted[inv_sort_idx].cpu())
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
        sim_stats = statistics.conditioning_rows(sim_stats, T_obs, forcing_gt_expanded.cpu())
    else:
        x_dim = x_spont                                 # the PPC "sample trajectories" are spontaneous
        sim_stats = pipeline.gen_stats(x_spont, None, cfg.dt_exp, None, None, None,
                                       device=device, spontaneous_only=True)
        sim_stats = statistics.conditioning_rows(sim_stats, T_obs)
    # Conditioning layout, so the zero-variance count can be split by origin rather than reported as
    # one number. See analysis.invalid_breakdown: most of a big "invalid" count is normally empty chi
    # probe slots, which is a fact about the run's K, not a defect.
    from .SBI.statistics import SUMMARY_WIDTH
    ppc_layout = {"input_dim": SUMMARY_WIDTH + 1,
                  "chi_k_pad": cfg.chi_k_pad if cfg.chi_mode else None,
                  "chi_elem_w": config.CHI_ELEM_W if cfg.chi_mode else None,
                  "chi_n_freqs": cfg.chi_n_freqs if cfg.chi_mode else None}
    results = analysis.posterior_predictive_check(obs_stats.squeeze(), sim_stats, layout=ppc_layout)
    _note = analysis.describe_invalid(results.get("invalid_breakdown"))
    if _note:
        print(f"[ppc] {_note}", flush=True)
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
        central_rescale = derived.to_sim_rescale(central_nd, central_rescale, cfg.rescale_idx,
                                                *cfg.tier1_args)              # tier 1, as above
        t_scale_c = central_rescale[0, cfg.rescale_idx["t_scale"]].item()
        subsample_c = max(1, round((cfg.dt_exp / t_scale_c) / cfg.dt_nd_min))
        n_fine_c = min(cfg.steady_idx + N_points_obs * subsample_c, len(t))
        t_fine_c = t[:n_fine_c]
        n_segs_c = max(1, math.ceil(n_fine_c / CHUNK_LEN))
        if cfg.has_forcing and not cfg.chi_mode:
            force_c = pipeline.build_nondim_sin_force_tensor(
                forcing_gt, t_fine_c, central_rescale, cfg.forcing_idx, cfg.sim_rescale_idx)
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


# The experimental observation builders live in SBI/observations.py; re-exported here because
# the GUI runners and the diagnostic scripts call them as orchestrator.build_experiment_obs*.
from .SBI.observations import (build_experiment_obs, build_experiment_obs_spontaneous,  # noqa: E402
                               build_experiment_obs_chi)
