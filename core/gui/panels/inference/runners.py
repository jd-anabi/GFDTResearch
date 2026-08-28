"""Worker-callable inference runners: module-level so a Worker can call them with an injected
fig_sink, and free of Qt so they stay testable headless."""
from core import cli, orchestrator
from core.Helpers import file_manager, labels, visualizers


# ── worker-callable runners (module-level so a Worker can call them with an injected fig_sink) ─────
def _run_simulated_inference(cfg, posterior, cell_path, T_obs_s, *, gt_dicts=None, inferred_prior=None,
                             force_prior=None, fig_sink=None):
    """Mirror orchestrator.run's simulated branch: inject GT + T_obs, simulate, show GT trace + infer.

    ``gt_dicts`` is the hand-entered alternative to ``cell_path``: an (inits, params, rescale, forcing)
    tuple in parse_values_file's shape. It goes through the SAME inject_ground_truth validation, so
    typed values are bounds-checked exactly like a file's."""
    ignored = (cfg.inject_ground_truth(*gt_dicts) if gt_dicts is not None
               else cli.load_and_validate_gt(cfg, cell_path))
    if ignored:
        print(f"Note: the bounds file does not declare {', '.join(ignored)} — those cell values were "
              f"ignored (the bounds file defines the inferred set).")
    cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
    # Is this observation actually in the region the network trained on? Bounds-checking cannot tell.
    if inferred_prior is not None:
        for msg in orchestrator.check_observation_in_distribution(cfg, inferred_prior, force_prior):
            print(f"WARNING: {msg}")
    x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)
    visualizers.plot(t_dim.squeeze(0).cpu().detach().numpy(), x_dim[0, :].cpu().detach().numpy(),
                     title="Ground-truth trace",
                     labels=(labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)),
                     sink=fig_sink)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True, fig_sink=fig_sink)


def _run_experimental_inference(cfg, posterior, spont_path, forced_path, T_obs_s, forcing_si, *, fig_sink=None):
    """Mirror orchestrator.run's experimental branch."""
    x_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
    x_forced = file_manager.load_experimental_data(forced_path, dtype=cfg.hw.dtype)
    obs_stats, obs_data, t_dim = orchestrator.build_experiment_obs(cfg, x_spont, x_forced, T_obs_s, forcing_si)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False, fig_sink=fig_sink)


def _run_experimental_inference_chi(cfg, posterior, spont_path, forced_pairs, T_obs_s, F0_si,
                                    *, fig_sink=None):
    """chi(omega) experimental inference: ONE passive recording (which sets Omega_0) plus ANY NUMBER
    of single-tone forced recordings, each locked in at THE FREQUENCY IT WAS ACTUALLY DRIVEN AT.

    ``forced_pairs`` is a list of ``(path, drive_frequency_Hz)``. It used to be a bare list of paths
    whose frequencies were assumed to be ``chi.chi_multipliers_for(cfg)``: the core has accepted
    per-probe frequencies at any count for some time, and the GUI was the only thing still forcing
    a fixed grid on it."""
    x_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
    x_forced = [(file_manager.load_experimental_data(p, dtype=cfg.hw.dtype), float(f))
                for p, f in forced_pairs]
    obs_stats, obs_data, t_dim = orchestrator.build_experiment_obs_chi(
        cfg, x_spont, x_forced, T_obs_s, F0_si)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False,
                                     fig_sink=fig_sink)


def _run_experimental_inference_spontaneous(cfg, posterior, path, T_obs_s, *, fig_sink=None):
    """Passive-recording inference for a no-forcing model: a single unforced recording, no drive."""
    x_obs = file_manager.load_experimental_data(path, dtype=cfg.hw.dtype)
    obs_stats, obs_data, t_dim = orchestrator.build_experiment_obs_spontaneous(cfg, x_obs, T_obs_s)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False, fig_sink=fig_sink)


def _run_tsnpe_round(cfg, posterior, inferred_prior, force_prior, obs_path, n_directions, level,
                     num_runs, run_size_cap, *, fig_sink=None):
    """One TSNPE round: region from the posterior -> prior RESTRICTED to it -> simulate -> retrain.

    The proposal is the TRUNCATED PRIOR and never the posterior -- see core/SBI/truncate.py, which
    owns that rule, and tests/test_conditioning_repair.py, which pins it. Nothing here reimplements
    it; this function only carries the GUI's choices into orchestrator.
    """
    rec = orchestrator.load_observation(obs_path)
    x_obs = rec["x_obs"].to(cfg.hw.device)
    region = orchestrator.build_truncation_region(posterior, rec, x_obs,
                                                  n_directions=n_directions, level=level)
    print(f"[tsnpe] region from {getattr(obs_path, 'name', obs_path)}: {region!r}", flush=True)
    out = orchestrator.build_posterior(
        cfg, inferred_prior, force_prior, choice=None, train_new=True, save=False,
        fig_sink=fig_sink, num_runs=num_runs, run_size_cap=run_size_cap,
        truncation=region, x_obs_digest=rec.get("digest"))
    # The region and digest ride back with the posterior. save=False here because the GUI saves from
    # a button, and a deferred save that does not know about the region writes an artifact marked
    # amortized -- see TSNPEPanel._on_round.
    return out, region, rec.get("digest")

