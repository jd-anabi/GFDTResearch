"""End-to-end SBI test for a no-forcing user-defined model (v2).

Drives the WHOLE spontaneous inference path at tiny sizes: build a stability-screened UserPrior, train a
short NPE posterior, run SBC/TARP calibration, infer on a simulated observation, and infer on a passive
recording. The stability sweep's production constants (50 iterations x batch, n_max=175000 flood-fill)
are far too slow for a test, so ``pipeline.gen_prior`` is monkeypatched to a tiny UserPrior screen -- the
same construct_prior call, small sizes. Everything else runs for real.

Also pins the built-in (Nadrowski) forcing path: generate_observations still yields the full-width,
Group-G-populated conditioning vector, so the spontaneous-only branching did not perturb it.

Run:  python tests/test_user_sbi.py      (or under pytest)
"""
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")

import torch                                                      # noqa: E402

from core import config, registry, orchestrator, cli, forcing    # noqa: E402
from core.Helpers import model_store                             # noqa: E402
from core.SBI import chi as chi_mod, pipeline as pipeline_mod    # noqa: E402
from core.SBI.Priors.user_prior import UserPrior                 # noqa: E402
from core.SBI.statistics import FEATURE_LABELS                   # noqa: E402
from core.config import VALID_MODELS, VALID_LABELS               # noqa: E402

_N_GROUP_G = 11
_N_SPONT = len(FEATURE_LABELS) - _N_GROUP_G   # 30


def _tiny_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds,
                    state_dep_drift=False, num_iterations=25, log_mask=None,
                    dtype=torch.float32, device=torch.device("cpu")):
    """A tiny stand-in for pipeline.gen_prior: the same UserPrior.construct_prior, small sizes."""
    p = UserPrior(registry.get(model), dtype, device)
    return p.construct_prior(t, len(prior_bounds), 32, 8, segs, prior_bounds,
                             t_global_scale=2, num_iterations=2, n_max=120, steady=False,
                             state_dep_drift=state_dep_drift, log_mask=log_mask)


def test_no_forcing_user_model_full_sbi_pipeline():
    """build_prior -> build_posterior -> generate_observations -> infer -> validate -> passive-infer."""
    name = "SBITEST"
    doc = {"schema_version": 1, "name": name,
           "variables": [{"name": "x", "drift": "-k*x", "D": "d0", "init": 0.5, "forcing": None}],
           "params": {"k": 1.0, "d0": 0.05}, "rescale": {"x_scale": 10.0, "t_scale": 0.01}}
    saved_gen_prior = orchestrator.pipeline.gen_prior
    saved_runs, saved_ncal = orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL
    sink = lambda title, fig: None                                # noqa: E731
    try:
        model_store.save_user_model(doc)
        registry.load_user_models()
        assert registry.is_sbi_user_model(name) is True

        cfg = cli.make_sim_config(name, registry.get(name).labels, registry.state_dep_drift(name),
                                  str(config.BOUNDS_PATH / name.lower() / "default.txt"))
        cli.load_and_validate_gt(cfg, str(config.CELL_PATH / name.lower() / "default.txt"))
        cfg.hw = config.cpu_device()
        cfg.hw.batch_size = 8
        cfg.T_obs = 1.0
        assert cfg.has_forcing is False

        orchestrator.pipeline.gen_prior = _tiny_gen_prior
        orchestrator.TRAINING_NUM_RUNS = 2
        orchestrator.SBC_N_CAL = 60

        inferred_prior, force_prior = orchestrator.build_prior(cfg, None, True, save=False, fig_sink=sink)
        assert force_prior is None                               # no drive -> no forcing prior

        posterior, _ = orchestrator.build_posterior(cfg, inferred_prior, force_prior, None, True,
                                                    save=False, fig_sink=sink)

        x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)
        assert obs_stats.shape[-1] == len(FEATURE_LABELS) + 1    # [S | log(T)], no forcing block
        assert torch.allclose(obs_stats[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
        assert torch.isfinite(obs_stats).all()

        orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True,
                                         fig_sink=sink)
        orchestrator.validate_calibration(cfg, posterior, inferred_prior, force_prior, fig_sink=sink)

        # passive experimental path: a single unforced recording, no drive / force units
        obs_stats_e, obs_data_e, t_dim_e = orchestrator.build_experiment_obs_spontaneous(
            cfg, x_dim[0].clone(), 1.0)
        assert obs_stats_e.shape[-1] == len(FEATURE_LABELS) + 1
        assert torch.allclose(obs_stats_e[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
        orchestrator.infer_and_visualize(cfg, posterior, obs_stats_e, obs_data_e, t_dim_e,
                                         show_truth=False, fig_sink=sink)
    finally:
        orchestrator.pipeline.gen_prior = saved_gen_prior
        orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL = saved_runs, saved_ncal
        try:
            model_store.delete_user_model(name)
        except Exception:                                        # noqa: BLE001
            pass
        registry.unregister(name)


def test_builtin_forcing_path_unperturbed():
    """The spontaneous-only branching must leave the Nadrowski forcing path byte-compatible: a full-width
    conditioning vector [S(41) | log(T) | forcing] with Group G populated by the drive response."""
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, True,
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()
    cfg.T_obs = 1000.0                                            # ms units -> 1 s of data
    assert cfg.has_forcing is True
    _, obs_stats, _ = orchestrator.generate_observations(cfg)
    n_forcing = len(cfg.force_params_dict)
    assert obs_stats.shape[-1] == len(FEATURE_LABELS) + 1 + n_forcing
    assert not torch.allclose(obs_stats[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
    assert torch.isfinite(obs_stats).all()


def _tiny_nadrowski_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds,
                              state_dep_drift=False, num_iterations=25, log_mask=None,
                              dtype=torch.float32, device=torch.device("cpu")):
    """Tiny stand-in for pipeline.gen_prior on the built-in Nadrowski: same construct_prior, small sizes."""
    from core.SBI.Priors import nadrowski_prior
    p = nadrowski_prior.NadrowskiPrior(dtype, device)
    return p.construct_prior(t, len(prior_bounds), 32, 8, segs, prior_bounds,
                             t_global_scale=2, num_iterations=2, n_max=120, steady=False,
                             state_dep_drift=state_dep_drift, log_mask=log_mask)


def test_train_and_validate_without_a_loaded_cell():
    """Training + calibration are GROUND-TRUTH-FREE, so they must work on a config built from BOUNDS
    ALONE (no cell file). Pins the _observation_inits fallback: cfg.inits_tensor RAISES on an empty
    inits_dict, and the inference Simulate tab used to be the only pre-Posterior path that populated it,
    so this scenario was silently unreachable rather than supported."""
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    saved_gen_prior = orchestrator.pipeline.gen_prior
    saved_runs, saved_ncal = orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL
    sink = lambda title, fig: None                                # noqa: E731
    try:
        cfg = cli.make_sim_config("NADROWSKI", labels, True,
                                  str(config.BOUNDS_PATH / "nadrowski" / "master.txt"),
                                  reparam_rotate=False)           # the Fisher rotation is far too slow here
        cfg.hw = config.cpu_device()
        cfg.hw.batch_size = 8
        assert cfg.observation_mode == "forced"
        assert not cfg.inits_dict and not cfg.has_ground_truth     # a genuinely bounds-only config
        try:                                                       # the raise this fallback works around
            cfg.inits_tensor
            raise AssertionError("expected inits_tensor to raise on a cell-free config")
        except ValueError:
            pass

        orchestrator.pipeline.gen_prior = _tiny_nadrowski_gen_prior
        orchestrator.TRAINING_NUM_RUNS = 2
        orchestrator.SBC_N_CAL = 40

        inferred_prior, force_prior = orchestrator.build_prior(cfg, None, True, save=False, fig_sink=sink)
        posterior, _ = orchestrator.build_posterior(cfg, inferred_prior, force_prior, None, True,
                                                    save=False, fig_sink=sink)
        orchestrator.validate_calibration(cfg, posterior, inferred_prior, force_prior, fig_sink=sink)
    finally:
        orchestrator.pipeline.gen_prior = saved_gen_prior
        orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL = saved_runs, saved_ncal


def test_out_of_distribution_drive_is_flagged():
    """The check that would have caught the 2026-07-27 retrain: a cell whose DRIVE the training prior
    cannot produce (freq=0 against a log-uniform freq prior) must be flagged, even though bounds-checking
    passes -- forcing is deliberately not range-checked. An in-distribution drive must NOT be flagged."""
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, True,
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()
    force_prior = orchestrator._build_forcing_prior(cfg)      # cheap: no simulation involved

    class _BoxPrior:                                          # stands in for the (expensive) ND prior
        def __init__(self, bounds):
            self.lo = torch.tensor([b[0] for b in bounds], dtype=torch.float64)
            self.hi = torch.tensor([b[1] for b in bounds], dtype=torch.float64)

        def sample(self, shape):
            n = shape[0] if isinstance(shape, (tuple, torch.Size)) else int(shape)
            return self.lo + torch.rand(n, self.lo.numel(), dtype=torch.float64) * (self.hi - self.lo)

    inf_prior = _BoxPrior([row[1] for row in cfg.params_dict.values()]
                          + [row[1] for row in cfg.rescale_params.values()])

    # master_weak carries an in-distribution DRIVE -> no drive warning
    msgs = orchestrator.check_observation_in_distribution(cfg, inf_prior, force_prior)
    assert not any("Drive" in m for m in msgs), msgs
    # ...and NO parameter sits on a box edge either. The archived cell_2 had tau_c=0 and temp=0 exactly
    # on their lower bounds, which is why those two marginals could only ever come out one-sided; the
    # master cells are built strictly interior, so a quantile flag here is now a real regression.
    assert not any("'tau_c'" in m for m in msgs), msgs
    assert not any("'temp'" in m for m in msgs), msgs
    # phase must NOT be flagged despite sitting at 0: it is circular, so every value is reachable.
    assert not any("'phase'" in m for m in msgs), msgs

    # The edge detector itself must still fire -- pin it by pushing a parameter onto its bound.
    lo_tau_c = cfg.params_dict["tau_c"][1][0]
    saved = cfg.params_dict["tau_c"]
    cfg.params_dict["tau_c"] = (lo_tau_c, saved[1])
    assert any("'tau_c'" in m for m in
               orchestrator.check_observation_in_distribution(cfg, inf_prior, force_prior))
    cfg.params_dict["tau_c"] = saved

    # zero it out the way it used to be -> the drive must be flagged
    for name in cfg.force_params_dict:
        cfg.force_params_dict[name] = (0.0, cfg.force_params_dict[name][1])
    msgs = orchestrator.check_observation_in_distribution(cfg, inf_prior, force_prior)
    assert any("freq" in m for m in msgs), msgs

    # ...but chi-mode ignores the cell's drive entirely, so it must NOT be flagged there
    cfg.chi_mode = True
    assert not any("Drive" in m for m in
                   orchestrator.check_observation_in_distribution(cfg, inf_prior, force_prior))


def test_observation_modes_and_conditioning_widths():
    """The three observation protocols must resolve from (chi_mode, has_forcing) and produce three
    DISTINCT conditioning widths, so a posterior trained in one mode cannot silently be used in another.
    Also pins that a forced cell can be paired with a spontaneous bounds file (extra cell values are
    ignored, not fatal) and that mode 1 carries no f_scale."""
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    saved_mode, saved_k = config.CHI_MODE, config.CHI_N_FREQS
    S = len(FEATURE_LABELS)
    try:
        config.CHI_N_FREQS = 3

        def build(bounds, cell, **kw):
            cfg = cli.make_sim_config("NADROWSKI", labels, True,
                                      str(config.BOUNDS_PATH / "nadrowski" / f"{bounds}.txt"), **kw)
            cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / f"{cell}.txt"))
            cfg.hw = config.cpu_device()
            cfg.T_obs = 500.0
            return cfg

        # mode 1 -- spontaneous: no drive, and f_scale is correctly absent (it could not act on anything)
        spont = build("master_spont", "master_spont")
        assert spont.observation_mode == "spontaneous"
        assert "f_scale" not in spont.rescale_params
        assert len(spont.params_dict) + len(spont.rescale_params) == 12
        assert orchestrator.generate_observations(spont)[1].shape[-1] == S + 1

        # mode 2 -- forced: the cell's own drive, f_scale identified through Group G's gain
        forced = build("master", "master_weak")
        assert forced.observation_mode == "forced" and "f_scale" in forced.rescale_params
        assert orchestrator.generate_observations(forced)[1].shape[-1] == S + 1 + len(forced.force_params_dict)

        # mode 3 -- chi: K probes; the cell's own drive is ignored
        chi_cfg = build("master", "master_weak", chi_mode=True, chi_n_freqs=3)
        assert chi_cfg.observation_mode == "chi"
        # Width is a function of the PAD, not the probe count -- that is what lets one posterior
        # serve any number of probes. Asserted via the shared rule, never a fresh literal.
        assert (orchestrator.generate_observations(chi_cfg)[1].shape[-1]
                == S + 1 + orchestrator.expected_forcing_dim(chi_cfg))
        assert orchestrator.expected_forcing_dim(chi_cfg) == config.CHI_ELEM_W * chi_cfg.chi_k_pad

        # a FORCING-BEARING cell against SPONTANEOUS bounds: extras dropped, not an error.
        # This is what lets ONE master cell serve every mode -- master_spont.txt carries f_scale and a
        # zeroed Forcing section precisely so chi mode (which needs f_scale inferred) can use it too.
        mixed = cli.make_sim_config("NADROWSKI", labels, True,
                                    str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
        ignored = cli.load_and_validate_gt(mixed, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
        assert mixed.observation_mode == "spontaneous"
        assert "f_scale (rescale)" in ignored and any(n.endswith("(forcing)") for n in ignored)

        # ...but a genuinely MISSING parameter is still fatal. Every master cell deliberately carries
        # the full superset, so this is exercised at the guard rather than with a deficient file.
        vals = {k: v for k, v in ((n, r[0]) for n, r in forced.rescale_params.items())}
        vals.pop("f_scale")
        try:
            forced.inject_ground_truth(dict(forced.inits_dict),
                                       {n: r[0] for n, r in forced.params_dict.items()},
                                       vals,
                                       {n: r[0] for n, r in forced.force_params_dict.items()})
            raise AssertionError("expected a missing-parameter ValueError")
        except ValueError as e:
            assert "missing" in str(e).lower() and "f_scale" in str(e)
    finally:
        config.CHI_MODE, config.CHI_N_FREQS = saved_mode, saved_k


def test_chi_fisher_rotation_builds_over_the_chi_feature_set():
    """chi mode now gets a decorrelating rotation too, and its Fisher must be built over the features
    a chi posterior actually conditions on -- 41 summary + 3K chi, not the 41-feature single-frequency
    set.

    WHY THIS EXISTS. build_posterior used to skip the rotation entirely under chi, on the untested
    assumption that "chi already attacks the degeneracy the rotation targets". Measured on the master
    cell, chi leaves k~x_scale at 0.95 (0.98 forced), so that was wrong. Re-enabling it has two ways
    to fail SILENTLY, both pinned here: gen_chi_block never being called (a Fisher over the wrong
    experiment), and J being allocated at len(FEATURE_LABELS) rows, which would truncate the chi block
    out of the Jacobian without any error."""
    from core.SBI import decorrelate
    from core.SBI import pipeline as pipeline_mod
    from core.SBI.reparam import build_inferred_bijection
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    K = 2
    cfg = cli.make_sim_config("NADROWSKI", labels, True,
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"),
                              chi_mode=True, chi_n_freqs=K)
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()
    cfg.T_obs = 100.0
    assert cfg.observation_mode == "chi"
    P = len(cfg.params_dict) + len(cfg.rescale_params)

    calls, saved = [], pipeline_mod.gen_chi_raw

    def _spy(*a, **kw):
        out = saved(*a, **kw)
        # (chi, u, logcyc, valid). Record the probe count and whether the resolution filter was off.
        calls.append((out[0].shape[-1], kw.get("resolution_filter", True)))
        return out

    pipeline_mod.gen_chi_raw = _spy
    try:
        V = decorrelate.build_latent_fisher_rotation(cfg, build_inferred_bijection(cfg),
                                                     m=2, n_points=1)
    finally:
        pipeline_mod.gen_chi_raw = saved

    assert calls, "the chi probes were never simulated: the Fisher was built over the wrong features"
    assert {c[0] for c in calls} == {K}, calls
    # The Fisher must use the RAW lock-ins, so its feature width is 4 per probe (log|chi|, cos, sin,
    # logcyc) -- NOT the 6-channel conditioning block. `u` and `mask` are theta-independent here and
    # would write float32 rounding into the Jacobian at the magnitude of a real feature.
    assert len(chi_mod.CHI_FISHER_CHANNELS) == 4
    assert "u" not in chi_mod.CHI_FISHER_CHANNELS and "mask" not in chi_mod.CHI_FISHER_CHANNELS
    # The resolution filter MUST be off: it depends on f_peak, hence on theta, so a probe crossing
    # the cycle threshold between the +dz and -dz arms is a mask step of 1 over fnoise's 1e-9 floor.
    assert all(rf is False for _, rf in calls), "the Fisher must disable the resolution filter"
    # one baseline + a +dz/-dz pair per latent dimension, each evaluating the chi block
    assert len(calls) == 2 * P + 1, f"expected {2 * P + 1} chi evaluations, got {len(calls)}"
    assert V.shape == (P, P)
    assert float((V.T @ V - torch.eye(P, dtype=V.dtype)).abs().max()) < 1e-4


def test_spontaneous_fisher_rotation_is_orthogonal():
    """The decorrelating rotation used to require a drive to probe; it must now also work for a
    driveless (mode 1) config, where the Fisher is driven purely by Groups A-F."""
    from core.SBI import decorrelate
    from core.SBI.reparam import build_inferred_bijection
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, True,
                              str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_spont.txt"))
    cfg.hw = config.cpu_device()
    cfg.T_obs = 500.0
    assert cfg.observation_mode == "spontaneous"
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    V = decorrelate.build_latent_fisher_rotation(cfg, build_inferred_bijection(cfg), m=4, n_points=1)
    assert V.shape == (P, P)
    assert float((V.T @ V - torch.eye(P, dtype=V.dtype)).abs().max()) < 1e-4


def test_chi_mode_observation_width():
    """CHI_MODE: generate_observations yields [S(41, Group G zeroed) | log(T) | chi(3K)], all finite.
    Uses the Nadrowski cell (which HAS a forcing section) to pin that chi-mode ignores it and uses the
    passive trajectory + the K-frequency chi block instead."""
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    saved_mode, saved_k = config.CHI_MODE, config.CHI_N_FREQS
    try:
        config.CHI_MODE, config.CHI_N_FREQS = True, 3
        cfg = cli.make_sim_config("NADROWSKI", labels, True,
                                  str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
        assert cfg.chi_mode is True
        cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
        cfg.hw = config.cpu_device()
        cfg.T_obs = 1000.0                                        # ms units -> 1 s of data
        _, obs_stats, _ = orchestrator.generate_observations(cfg)
        assert obs_stats.shape[-1] == len(FEATURE_LABELS) + 1 + orchestrator.expected_forcing_dim(cfg)
        assert torch.allclose(obs_stats[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
        assert torch.isfinite(obs_stats).all()
    finally:
        config.CHI_MODE, config.CHI_N_FREQS = saved_mode, saved_k


def test_chi_mode_full_sbi_pipeline():
    """CHI_MODE end-to-end at tiny sizes: prior -> posterior -> observe -> infer -> validate, plus the
    experimental chi path. Pins the chi(omega) branch across gen_training_data / gen_cal_data / PPC."""
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    saved_gen_prior = orchestrator.pipeline.gen_prior
    saved_runs, saved_ncal = orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL
    saved_mode, saved_k = config.CHI_MODE, config.CHI_N_FREQS
    sink = lambda title, fig: None                                # noqa: E731
    try:
        config.CHI_MODE, config.CHI_N_FREQS = True, 3
        cfg = cli.make_sim_config("NADROWSKI", labels, True,
                                  str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
        cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
        cfg.hw = config.cpu_device()
        cfg.hw.batch_size = 8
        cfg.T_obs = 1000.0
        assert cfg.chi_mode is True

        orchestrator.pipeline.gen_prior = _tiny_nadrowski_gen_prior
        orchestrator.TRAINING_NUM_RUNS = 2
        orchestrator.SBC_N_CAL = 40

        inferred_prior, force_prior = orchestrator.build_prior(cfg, None, True, save=False, fig_sink=sink)
        posterior, _ = orchestrator.build_posterior(cfg, inferred_prior, force_prior, None, True,
                                                    save=False, fig_sink=sink)

        K3 = orchestrator.expected_forcing_dim(cfg)
        x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)
        assert obs_stats.shape[-1] == len(FEATURE_LABELS) + 1 + K3
        assert torch.isfinite(obs_stats).all()

        orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True,
                                         fig_sink=sink)
        orchestrator.validate_calibration(cfg, posterior, inferred_prior, force_prior, fig_sink=sink)

        # experimental chi path: 1 passive + K forced recordings (GT passive trace as stand-ins).
        forced = [x_dim[0].clone() for _ in range(config.CHI_N_FREQS)]
        obs_stats_e, obs_data_e, t_dim_e = orchestrator.build_experiment_obs_chi(
            cfg, x_dim[0].clone(), forced, 1.0, 1.0)
        assert obs_stats_e.shape[-1] == len(FEATURE_LABELS) + 1 + K3
        assert torch.isfinite(obs_stats_e).all()
        orchestrator.infer_and_visualize(cfg, posterior, obs_stats_e, obs_data_e, t_dim_e,
                                         show_truth=False, fig_sink=sink)
    finally:
        orchestrator.pipeline.gen_prior = saved_gen_prior
        orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL = saved_runs, saved_ncal
        config.CHI_MODE, config.CHI_N_FREQS = saved_mode, saved_k


def _lock_in_reference(x, omega, F0, T_obs, dt):
    """The pre-chunking lock_in_batched, kept verbatim as the numerical reference."""
    x = x.to(torch.float64)
    x = x - x.mean(dim=-1, keepdim=True)
    n = x.shape[-1]
    t = torch.arange(n, device=x.device, dtype=torch.float64) * float(dt)
    phase = omega.to(torch.float64).reshape(-1, 1) * t.unsqueeze(0)
    e_iwt = torch.complex(torch.cos(phase), torch.sin(phase))
    F0 = F0.to(torch.float64).reshape(-1) if torch.is_tensor(F0) else float(F0)
    return (2.0 / (F0 * float(T_obs))) * (x.to(torch.complex128) * e_iwt).sum(dim=-1) * float(dt)


def test_lock_in_chunking_matches_full_batch():
    """chi.lock_in_batched accumulates over time chunks to keep the float64/complex128 working set off
    the full training batch. Pinned against the pre-chunking formula, on the case built to expose the
    two failure modes that are otherwise SILENT (finite, in-range, wrong):

      * demeaning per chunk instead of over the whole trace -- a high-pass filter that eats the LOW
        multipliers, which is exactly the part of chi(omega) the mode exists for;
      * using within-chunk times arange(0, e-s) instead of absolute arange(s, e).

    The trace therefore carries a large DC offset AND a slow trend, and omega is deliberately
    INCOMMENSURATE with chunk*dt (a commensurate omega hides the phase bug entirely).
    """
    torch.manual_seed(7)
    n, dt, L = 1234, 0.01, 256                      # n % L != 0 on purpose
    x = torch.randn(5, n) * 1e-3 + 40.0 + torch.linspace(0, 3, n).unsqueeze(0)
    omega = torch.full((5,), 2 * math.pi * 0.1 * 1.7)
    T_obs = n * dt

    for F0 in (1.0, torch.rand(5) + 0.5):
        ref = _lock_in_reference(x, omega, F0, T_obs, dt)
        got = chi_mod.lock_in_batched(x, omega, F0, T_obs, dt, chunk=L)
        rel = ((ref - got).abs() / ref.abs().clamp(min=1e-300)).max().item()
        assert rel < 1e-12, f"chunked lock-in drifted from the reference: max rel {rel:.3e}"

    # chunk length is a memory knob only -- every L must agree, including L=1 and L > n
    base = chi_mod.lock_in_batched(x, omega, 1.0, T_obs, dt, chunk=n)
    for L2 in (1, 7, 255, 4096, 99999):
        got = chi_mod.lock_in_batched(x, omega, 1.0, T_obs, dt, chunk=L2)
        rel = ((base - got).abs() / base.abs().clamp(min=1e-300)).max().item()
        assert rel < 1e-12, f"chunk={L2} changed the result (max rel {rel:.3e}); it must not"

    # A pure cosine of amplitude A driven at F0 has |chi| = A/F0, independent of chunking.
    n2, dt2, A, F0a, w = 100_000, 1e-3, 2.5, 0.7, 2 * math.pi * 13.0
    tt = torch.arange(n2, dtype=torch.float64) * dt2
    got = chi_mod.lock_in_batched((A * torch.cos(w * tt)).unsqueeze(0),
                                  torch.tensor([w]), F0a, n2 * dt2, dt2, chunk=8192)[0]
    assert abs(abs(got) - A / F0a) < 1e-6, f"analytic cosine: |chi|={abs(got)} expected {A / F0a}"


def test_peak_freq_sub_batching_is_bit_identical():
    """chi.peak_freq sub-batches its rfft over SAMPLES (rows are independent), so it must be
    bit-identical to the full-batch form -- including when the batch does not divide evenly."""
    torch.manual_seed(11)
    for B, n in ((700, 4001), (257, 4096), (1, 999), (5, 1), (300, 2)):
        x = torch.randn(B, n)
        full = chi_mod.peak_freq(x, 1e-3, batch=max(B, 1))
        subbed = chi_mod.peak_freq(x, 1e-3)          # default _PEAK_FREQ_BATCH
        assert full.shape == subbed.shape == (B,), f"B={B} n={n}: shape {full.shape}/{subbed.shape}"
        assert torch.equal(full, subbed), f"B={B} n={n}: sub-batching changed peak_freq"


def test_chi_downsample_slice_matches_clamped_gather():
    """gen_chi_block downsamples fine -> dt_exp without materialising a (B, N_points) int64 index when
    the subsample is a uniform int AND the fine grid is long enough. That fast path must be exactly the
    clamped gather it replaces -- and the guard must FALL BACK to the gather when the grid was clipped,
    because slicing truncates there while the gather replicates the last sample."""
    def selected(x_nd, subsample, N_points):
        """Mirrors the branch in pipeline.gen_chi_block (n_avail == x_nd's width)."""
        B, n_avail = x_nd.shape[0], x_nd.shape[1]
        s_int = None if torch.is_tensor(subsample) else max(1, int(subsample))
        if s_int is not None and s_int * (N_points - 1) < n_avail:
            return x_nd[:, ::s_int][:, :N_points], True
        subs = (subsample.long().clamp(min=1) if torch.is_tensor(subsample)
                else torch.full((B,), s_int, dtype=torch.long))
        idx = subs.unsqueeze(1) * torch.arange(N_points, dtype=torch.long).unsqueeze(0)
        return torch.gather(x_nd, 1, idx.clamp_(max=n_avail - 1)), False

    def old_gather(x_nd, subsample, N_points):
        B = x_nd.shape[0]
        subs = (subsample.long().clamp(min=1) if torch.is_tensor(subsample)
                else torch.full((B,), int(subsample), dtype=torch.long))
        idx = subs.unsqueeze(1) * torch.arange(N_points, dtype=torch.long).unsqueeze(0)
        return torch.gather(x_nd, 1, idx.clamp(max=x_nd.shape[1] - 1))

    torch.manual_seed(3)
    # Training geometry: gen_obs returns exactly N_points * subsample columns, so the clamp is dead.
    for B in (1, 3, 50):
        for s in (1, 2, 5, 17, 40):
            for N in (1, 2, 101, 1000):
                x = torch.randn(B, N * s)
                got, fast = selected(x, s, N)
                assert fast, f"B={B} s={s} N={N}: should have taken the no-index fast path"
                assert torch.equal(got, old_gather(x, s, N)), f"B={B} s={s} N={N}: fast path differs"

    # PPC passes a (B,) per-sample subsample; rows have different strides -> must keep the gather.
    x = torch.randn(50, 4000)
    subs = torch.randint(1, 6, (50,))
    got, fast = selected(x, subs, 700)
    assert not fast, "a (B,) subsample must not take the strided fast path"
    assert torch.equal(got, old_gather(x, subs, 700)), "tensor-subsample path differs from the gather"

    # Clipped fine grid (t_fine = t[:n] runs out; ~20% of accepted draws on model-builder bounds).
    s, N = 4, 10
    x = torch.arange(2 * 25, dtype=torch.float32).reshape(2, 25)     # 25 < s*(N-1)+1 == 37
    got, fast = selected(x, s, N)
    assert not fast, "a clipped fine grid must fall back to the clamped gather"
    assert got.shape == (2, N), f"clipped grid must still yield N_points columns, got {tuple(got.shape)}"
    assert torch.equal(got, old_gather(x, s, N)), "clipped-grid result differs from the clamped gather"
    assert x[:, ::s][:, :N].shape != got.shape, "test is toothless: unguarded slicing did not truncate"


def test_zero_force_tensor_channel_width():
    """The driveless runs in gen_training_data / generate_observations size their zero-force tensor with
    forcing.n_force_channels, not n_vars -- that is the single largest tensor in a training batch and
    n_vars over-allocates it 3x for Nadrowski and 5x for BP. The width must still be wide enough for
    every channel the model's DRIFT indexes, which is not the same as the cell's declared force params:
    HopfModel reads force_step[:, 1] unconditionally, so a driveless Hopf still needs 2."""
    assert forcing.n_force_channels("NADROWSKI", {"amp": 0, "freq": 1}, 3) == 1
    assert forcing.n_force_channels("NADROWSKI", {}, 3) == 1, "Nadrowski drift reads channel 0 only"
    assert forcing.n_force_channels("BP", {}, 5) == 1, "BP drift reads channel 0 only"
    assert forcing.n_force_channels("HOPF", {"amp": 0, "amp_y": 1}, 2) == 2
    assert forcing.n_force_channels("HOPF", {}, 2) == 2, "Hopf drift reads force_step[:, 1] regardless"

    # A width that is too NARROW is an IndexError inside the drift, so simulate for real (tiny).
    for model, sub, b_stem, c_stem in (("NADROWSKI", "nadrowski", "master", "master_weak"),
                                       ("HOPF", "hopf", "cell", "cell")):
        sdd = registry.state_dep_drift(model)
        cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)], sdd,
                                  str(config.BOUNDS_PATH / sub / f"{b_stem}.txt"))
        cli.load_and_validate_gt(cfg, str(config.CELL_PATH / sub / f"{c_stem}.txt"))
        cfg.hw = config.cpu_device()
        n_vars = cfg.inits_tensor.shape[-1]
        n_ch = forcing.n_force_channels(model, cfg.forcing_idx, n_vars)
        assert n_ch <= n_vars, f"{model}: {n_ch} channels is not a saving over n_vars={n_vars}"
        t = torch.linspace(0, 1.0, 60, dtype=cfg.hw.dtype)
        try:
            pipeline_mod.gen_obs(model=model, params=cfg.params_tensor, t=t, inits=cfg.inits_tensor,
                                 force=torch.zeros((1, n_ch, 60), dtype=cfg.hw.dtype),
                                 n_segs=1, steady_idx=10, state_dep_drift=sdd,
                                 dtype=cfg.hw.dtype, device=cfg.hw.device)
        except IndexError as e:
            raise AssertionError(f"{model}: {n_ch}-channel zero-force is too narrow for the drift ({e})")


def test_chi_mode_drives_every_channel_the_model_reads():
    """chi mode must hand the simulator as many force channels as the model's drift INDEXES.

    gen_chi_block builds its probe with the sinusoidal builder and a fidx that declares no "amp_y",
    so the builder emits ONE channel. HopfModel reads ``force_step[:, 1]`` unconditionally
    (hopf_model.py:15, :49) and a user model reads one channel per state variable, so chi mode used
    to die with an IndexError on anything but Nadrowski/BP. The probe drives channel 0 and leaves
    the rest at zero -- the same convention the FDT campaigns use.
    """
    saved_k = config.CHI_N_FREQS
    try:
        config.CHI_N_FREQS = 3
        labels = VALID_LABELS[VALID_MODELS.index("HOPF")]
        cfg = cli.make_sim_config("HOPF", labels, registry.state_dep_drift("HOPF"),
                                  str(config.BOUNDS_PATH / "hopf" / "cell.txt"),
                                  chi_mode=True, chi_n_freqs=3)
        cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "hopf" / "cell.txt"))
        cfg.hw = config.cpu_device()
        cfg.T_obs = 200.0
        assert cfg.observation_mode == "chi" and cfg.inits_tensor.shape[-1] == 2

        stats = orchestrator.generate_observations(cfg)[1]
        assert stats.shape[-1] == len(FEATURE_LABELS) + 1 + orchestrator.expected_forcing_dim(cfg), (
            f"hopf chi conditioning is the wrong width: {tuple(stats.shape)}")
        assert torch.isfinite(stats).all(), "hopf chi conditioning has non-finite entries"

        # The channel rule is what makes that work -- pin it, and pin that a 1-channel drive really
        # would break (so this test cannot pass for the wrong reason).
        fidx = {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
        assert forcing.n_force_channels("HOPF", fidx, 2) == 2
        assert forcing.n_force_channels("NADROWSKI", fidx, 3) == 1, "Nadrowski must not be widened"
        from core.Simulator.simulator import SimulationError
        saved_rule = pipeline_mod._forcing.n_force_channels
        try:
            pipeline_mod._forcing.n_force_channels = lambda *a, **k: 1     # the pre-fix behaviour
            try:
                orchestrator.generate_observations(cfg)
                raise AssertionError("a 1-channel drive should have failed for Hopf; "
                                     "this test would pass even unfixed")
            except SimulationError as e:
                # The drift's IndexError, wrapped by Simulator.__sols and chained as __cause__.
                assert isinstance(e.__cause__, IndexError), (
                    f"expected the drift's out-of-bounds channel read, got {e.__cause__!r}")
        finally:
            pipeline_mod._forcing.n_force_channels = saved_rule
    finally:
        config.CHI_N_FREQS = saved_k


def test_chi_batch_never_outruns_the_nd_time_grid():
    """Every training batch must fit the ND grid it slices, so the chi block and the summary
    statistics describe the SAME trace.

    ``t_fine = t[:n_fine_total]`` clips silently. The spontaneous trace is then built by SLICING (so
    it comes back short) while gen_chi_block GATHERS to N_points with a clamp that replicates the
    last sample -- chi.peak_freq and chi.lock_in_batched end up seeing different lengths for the same
    batch, and log(T_k) records a duration neither has. The Sobol pre-filter used to bound only
    N_ND_MAX; it now also bounds len(t). Built-in bounds never tripped this (len(t) ~ 2.4M vs a 300k
    cap); model-builder bounds, where len(t) is SHORTER than N_ND_MAX, did.
    """
    from core.SBI import chi as chi_module

    model = "NADROWSKI"
    cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)],
                              registry.state_dep_drift(model),
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()

    class _FixedPrior:                      # stands in for the trained prior
        def __init__(self, theta): self.theta = theta
        def sample(self, shape): return self.theta.expand(shape[0], -1).clone()

    n_grid, steady_idx = 12_000, 500        # deliberately SHORTER than N_ND_MAX
    t = torch.linspace(0, n_grid * cfg.dt_nd_min, n_grid, dtype=cfg.hw.dtype)

    # Key every lock-in to ITS batch explicitly. K is drawn per batch under the set layout, so
    # inferring it by dividing call counts (as this used to) is no longer even well-defined.
    seen = {"spont": [], "chi": []}
    real_peak, real_lock = chi_module.peak_freq, chi_module.lock_in_batched

    def _peak(x, dt, *a, **k):
        seen["spont"].append(x.shape[-1])
        return real_peak(x, dt, *a, **k)

    def _lock(x, *a, **k):
        seen["chi"].append((len(seen["spont"]) - 1, x.shape[-1]))
        return real_lock(x, *a, **k)

    chi_module.peak_freq, chi_module.lock_in_batched = _peak, _lock
    try:
        pipeline_mod.gen_training_data(
            model, _FixedPrior(cfg.ground_truth_tensor.reshape(1, -1)), None, t,
            run_size=2, n_runs=4, steady_idx=steady_idx, dt_nd_min=cfg.dt_nd_min,
            nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
            dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
            t_scale_bounds=cfg.t_scale_bounds, state_dep_drift=cfg.state_dep_drift,
            chi_mode=True, chi_k_fixed=2, chi_f0=config.CHI_F0,
            chi_freq_bounds=config.CHI_FREQ_BOUNDS, n_vars=cfg.inits_tensor.shape[-1],
            dtype=cfg.hw.dtype, device=cfg.hw.device)
    finally:
        chi_module.peak_freq, chi_module.lock_in_batched = real_peak, real_lock

    assert seen["spont"], "the chi branch never ran"
    assert seen["chi"], "no probes were locked in"
    # A probe may see FEWER samples than the summary statistics -- per-probe durations are a
    # deliberate axis of the training distribution, and the count is recorded in the probe's own
    # logcyc channel so the encoder can discount a short one. It must never see MORE: that is the
    # 2026-07-28 defect, where the spontaneous trace was built by SLICING a clipped fine grid (so it
    # came back short) while the chi path GATHERED to N_points with a clamp replicating the last
    # sample -- the two then described different trace lengths and log(T) recorded a third.
    for batch_i, n_chi in seen["chi"]:
        n_spont = seen["spont"][batch_i]
        assert 0 < n_chi <= n_spont, (
            f"batch {batch_i}: a chi probe saw {n_chi} samples but the summary statistics saw "
            f"{n_spont} -- the fine grid was clipped and the conditioning row is inconsistent")


def test_chi_k_fixed_holds_the_probe_count_for_a_stratified_calibration():
    """``chi_k_fixed`` must pin the per-ROW live-probe count, and the default must NOT.

    WHY THIS EXISTS. Section 4.1 step 5 wants SBC stratified by probe count, because a pooled SBC
    over a mixture of counts can be flat while each count is miscalibrated in compensating
    directions. There was no lever: ``chi_n_freqs`` was accepted by gen_training_data and never read.

    TWO ways to get this wrong, both silent:
      * fixing k_b but leaving ``_subset_probe_rows`` on. The drive SET would then have K probes while
        individual ROWS kept a random prefix of them, so a "K = 3" stratum would quietly be a mixture
        over 1..3 -- exactly the pooled measurement the stratification exists to avoid.
      * honouring ``chi_n_freqs`` during TRAINING, which is the obvious "fix" for the dead parameter
        and would train a network that has only ever seen one probe count.

    Asserted against the PACKER's own mask rather than against a bare count, because after
    pack_probe_block a masked probe and a pad slot are bitwise identical (a dead slot is exactly 0.0
    in all six channels). Some probes ARE legitimately masked here -- the training draw includes short
    recordings where a 0.03x probe cannot complete two drive cycles -- so "the row has exactly K live
    slots" is not a true invariant at any K. What IS invariant: K probes are simulated, and the row
    keeps every probe the packer marked valid.
    """
    model = "NADROWSKI"
    cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)],
                              registry.state_dep_drift(model),
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()

    class _FixedPrior:
        def __init__(self, theta): self.theta = theta
        def sample(self, shape): return self.theta.expand(shape[0], -1).clone()

    n_grid, steady_idx = 12_000, 500
    t = torch.linspace(0, n_grid * cfg.dt_nd_min, n_grid, dtype=cfg.hw.dtype)
    k_pad, k_fixed = 6, 3

    def _run(**kw):
        """-> (probes simulated per batch, packer-valid count per row, live slots per emitted row)."""
        simulated, valid_rows = [], []
        real_raw, real_block = pipeline_mod.gen_chi_raw, pipeline_mod.gen_chi_block

        def _spy_raw(*a, **k):
            out = real_raw(*a, **k)
            simulated.append(out[0].shape[-1])              # (B, K) chi stack -> K probes simulated
            return out

        def _spy_block(*a, **k):
            block, mask = real_block(*a, **k)
            valid_rows.append(mask.sum(dim=1).clone())      # (B,) live probes BEFORE any subsetting
            return block, mask

        pipeline_mod.gen_chi_raw, pipeline_mod.gen_chi_block = _spy_raw, _spy_block
        try:
            data, _ = pipeline_mod.gen_training_data(
                model, _FixedPrior(cfg.ground_truth_tensor.reshape(1, -1)), None, t,
                run_size=4, n_runs=4, steady_idx=steady_idx, dt_nd_min=cfg.dt_nd_min,
                nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
                dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
                t_scale_bounds=cfg.t_scale_bounds, state_dep_drift=cfg.state_dep_drift,
                chi_mode=True, chi_f0=config.CHI_F0, chi_freq_bounds=config.CHI_FREQ_BOUNDS,
                chi_k_pad=k_pad, n_vars=cfg.inits_tensor.shape[-1],
                dtype=cfg.hw.dtype, device=cfg.hw.device, **kw)
        finally:
            pipeline_mod.gen_chi_raw, pipeline_mod.gen_chi_block = real_raw, real_block
        block = data[:, -k_pad * config.CHI_ELEM_W:].reshape(-1, k_pad, config.CHI_ELEM_W)
        live = (block != 0).any(dim=-1).sum(dim=-1)                      # (rows,)
        # Batches are appended in order and concatenated, so these line up row-for-row.
        return simulated, torch.cat(valid_rows), live

    simulated, valid, live = _run(chi_k_fixed=k_fixed)
    assert live.numel(), "no training rows were produced"
    assert int(valid.sum()) > 0, (
        "every probe was masked, so the equality below holds trivially -- this test's geometry is "
        "wrong, not the code's")
    assert set(simulated) == {k_fixed}, (
        f"chi_k_fixed={k_fixed} simulated {sorted(set(simulated))} probes per batch. The count must "
        f"not be drawn: a calibration stratum is the whole point of the flag.")
    assert torch.equal(live, valid), (
        f"chi_k_fixed dropped probes the packer had marked valid: live={live.tolist()} vs "
        f"valid={valid.tolist()}. The per-row subsetting is still running, so a 'K = {k_fixed}' "
        f"stratum is silently a mixture over 1..{k_fixed} -- the pooled measurement it exists to avoid.")

    simulated_p, valid_p, live_p = _run()
    assert len(set(simulated_p)) > 1, (
        "the DEFAULT path simulated one probe count for every batch. K must vary across the training "
        "set -- a fixed count would leave the encoder's K-agnosticism untrained rather than merely "
        "untested.")
    assert not torch.equal(live_p, valid_p), (
        "the DEFAULT path kept every valid probe in every row, so _subset_probe_rows did nothing. "
        "Per-row subsetting is what decouples the probe count from the batch's (t_scale, T) stratum.")

    for bad in (0, k_pad + 1):
        try:
            _run(chi_k_fixed=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"chi_k_fixed={bad} was accepted. Out of 1..chi_k_pad it either asks for more probes "
                f"than the network has slots or for an all-masked observation.")


def test_solver_failure_raises_instead_of_killing_the_process():
    """A solver failure must raise SimulationError, not hard-exit.

    ``Simulator.__sols`` used to answer every failure with ``print(...); exit()``. That killed the
    interpreter outright: a CUDA OOM arrived with no traceback, and every caller -- these test
    runners, the GUI's QThreadPool worker, the scripts -- died mid-run with nothing reported.

    Two properties, and the second is the subtle one: an ORDINARY failure must be wrapped (with the
    original preserved as __cause__ so the real traceback survives), while a cooperative cancel must
    still sail straight through. streams.WorkerCancelled derives from BaseException exactly so it
    skips handlers like this one; widening the except would turn every GUI cancel into a spurious
    simulation failure. See test_gui_progress.test_worker_cancelled_passes_through_except_exception
    for the generic version of that contract.
    """
    from core.Solvers import sdeint
    from core.Simulator.simulator import SimulationError
    from core.gui.streams import WorkerCancelled

    model = "NADROWSKI"
    sdd = registry.state_dep_drift(model)
    cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)], sdd,
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()
    n_ch = forcing.n_force_channels(model, cfg.forcing_idx, cfg.inits_tensor.shape[-1])

    def run():
        return pipeline_mod.gen_obs(
            model=model, params=cfg.params_tensor, t=torch.linspace(0, 1.0, 50, dtype=cfg.hw.dtype),
            inits=cfg.inits_tensor, force=torch.zeros((1, n_ch, 50), dtype=cfg.hw.dtype),
            n_segs=1, steady_idx=5, state_dep_drift=sdd, var_idx=0,
            dtype=cfg.hw.dtype, device=cfg.hw.device)

    def solver_raising(exc):
        """Stand-in for sdeint.Solver whose methods all raise. The real solver builds its methods as
        closures in __init__, so they are instance attributes -- patch the CLASS, not the function."""
        class _Stub:
            def euler(self, *a, **k): raise exc
            def euler_compiled(self, *a, **k): raise exc
        return _Stub

    assert run().shape == (1, 1, 45), "the unpatched path must still simulate normally"

    saved = sdeint.Solver
    try:
        sdeint.Solver = solver_raising(torch.OutOfMemoryError("CUDA error: out of memory"))
        try:
            run()
            raise AssertionError("a solver failure must raise, but gen_obs returned normally")
        except SimulationError as e:
            assert isinstance(e.__cause__, torch.OutOfMemoryError), (
                f"the original error must survive as __cause__, got {e.__cause__!r}")
            for token in ("NadrowskiModel", "batch=", "device=", "out of memory"):
                assert token in str(e), f"SimulationError message is missing {token!r}: {e}"

        # ...and a cancel must NOT be converted into one.
        sdeint.Solver = solver_raising(WorkerCancelled())
        try:
            run()
            raise AssertionError("WorkerCancelled was swallowed; the GUI cancel path is broken")
        except WorkerCancelled:
            pass
        except SimulationError as e:
            raise AssertionError(f"a cancel was mis-reported as a simulation failure: {e}")
    finally:
        sdeint.Solver = saved

    # The same wart lived in the built-in subclasses' _set_up_model, which answered a bad cell with
    # print + exit(). The GUI's Simulate panel used to carry an `except SystemExit` translation for
    # exactly that; it has been retired, so construction must raise on its own.
    from core.Simulator.nadrowski_simulator import NadrowskiSimulator
    try:
        NadrowskiSimulator(torch.zeros((1, 3)),              # far too few params -> unbind mismatch
                           torch.zeros((1, n_ch, 2)), cfg.inits_tensor,
                           torch.zeros(2, dtype=cfg.hw.dtype), segs=1, batch_size=1)
        raise AssertionError("a bad simulator construction must raise, not exit()")
    except SimulationError as e:
        assert e.__cause__ is not None, "construction failure must chain the original error"
        assert "construction failed" in str(e), f"unexpected message: {e}"


def test_sim_batch_planning():
    """pipeline._max_sim_batch decides whether a simulation batch has to be split. It is pure
    arithmetic and it is where the subtle bugs live, so pin the behaviour directly."""
    plan = pipeline_mod._max_sim_batch
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    f32 = torch.float32
    kw = dict(n_fine=300_000, steady_idx=4_000, n_vars=3, n_ch=1, n_out=1, dtype=f32)

    # CPU (and a batch of one) never splits -- the guard is a CUDA memory guard.
    assert plan(batch_size=2048, device=cpu, **kw) == 2048
    assert plan(batch_size=1, device=cpu, **kw) == 1
    if not torch.cuda.is_available():
        return

    got = plan(batch_size=2048, device=cuda, **kw)
    assert 1 <= got <= 2048, f"planned chunk {got} out of range"
    assert got == 2048 or (got & (got - 1)) == 0, (
        f"a split chunk must be a power of two so the solver reuses shapes, got {got}")
    assert got == 2048 or got >= 256, f"a split chunk must not go below the floor, got {got}"

    # A geometry that cannot possibly fit must NOT be ground down to a sliver: splitting cannot
    # rescue it, so the planner hands the batch back unchanged and lets it fail loudly.
    huge = plan(batch_size=2048, device=cuda, n_fine=300_000, steady_idx=0,
                n_vars=64, n_ch=64, n_out=64, dtype=f32)
    assert huge == 2048, f"an unfittable geometry must be returned unsplit, got {huge}"

    # A tiny geometry always fits whole -- the guard must be invisible in the common case.
    assert plan(batch_size=2048, device=cuda, n_fine=1_000, steady_idx=100,
                n_vars=3, n_ch=1, n_out=1, dtype=f32) == 2048


def test_split_gen_obs_concatenates_correctly():
    """When the guard does split, gen_obs must stitch the chunks back into one batch -- including a
    ragged final chunk. Forced on CPU (where the guard never fires on its own) by patching the
    planner, so the concatenation logic is exercised without needing a GPU."""
    saved = pipeline_mod._max_sim_batch
    try:
        pipeline_mod._max_sim_batch = lambda batch_size, *a, **k: 3      # 8 -> 3 + 3 + 2
        model, B, n = "NADROWSKI", 8, 80
        sdd = registry.state_dep_drift(model)
        cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)], sdd,
                                  str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
        cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
        cfg.hw = config.cpu_device()
        params = cfg.params_tensor.expand(B, -1).contiguous()
        inits = cfg.inits_tensor.expand(B, -1).contiguous()
        n_ch = forcing.n_force_channels(model, cfg.forcing_idx, inits.shape[-1])
        out = pipeline_mod.gen_obs(model=model, params=params,
                                   t=torch.linspace(0, 1.0, n, dtype=cfg.hw.dtype), inits=inits,
                                   force=torch.zeros((B, n_ch, n), dtype=cfg.hw.dtype),
                                   n_segs=1, steady_idx=10, state_dep_drift=sdd, batch_size=B,
                                   var_idx=0, dtype=cfg.hw.dtype, device=cfg.hw.device)
        assert out.shape == (1, B, n - 10), f"split gen_obs returned {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "split gen_obs produced non-finite values"
    finally:
        pipeline_mod._max_sim_batch = saved


def test_cufft_plan_cache_is_cleared_between_training_batches():
    """cuFFT caches one plan per distinct transform SHAPE, allocated OUTSIDE PyTorch's caching
    allocator -- so torch.cuda.empty_cache() cannot reclaim it and exhaustion surfaces as a raw driver
    cudaErrorMemoryAllocation. N_points_k changes every training batch, so cross-batch reuse is zero
    and the default 4096-entry cache would accumulate ~2 MB apiece until it OOMs mid-run."""
    import inspect
    src = inspect.getsource(pipeline_mod.gen_training_data)
    assert "cufft_plan_cache.clear()" in src, (
        "gen_training_data must clear the cuFFT plan cache per batch; empty_cache() does NOT free it")

    if not torch.cuda.is_available():
        return                                       # mechanism check below is CUDA-only
    cache = torch.backends.cuda.cufft_plan_cache
    cache.clear()
    for n in (4096, 4097, 5003, 6151):               # distinct lengths -> distinct plans
        chi_mod.peak_freq(torch.randn(8, n, device="cuda"), 1e-3)
    assert cache.size > 0, "expected distinct transform lengths to mint distinct cuFFT plans"
    cache.clear()
    assert cache.size == 0, "cufft_plan_cache.clear() did not release the cached plans"


def test_summary_statistics_rejects_a_non_uniform_dt():
    """A per-sample dt must be REFUSED, not silently averaged.

    Every frequency axis, ACF decay time and Group-G lock-in phase shares ONE grid built from a single
    scalar. `gen_stats` genuinely accepts and sub-batches a (B,) dt, so the feature looked supported
    while `SummaryStatistics` quietly took the mean -- a value correct for no row in the batch.
    A UNIFORM tensor is still accepted (that is what the live callers pass) and must be identical to
    passing the scalar.
    """
    from core.SBI.statistics import SummaryStatistics

    B, n, dt = 4, 256, 1e-3
    torch.manual_seed(0)
    x = torch.randn(B, n)
    zero = torch.zeros(B)

    scalar = SummaryStatistics(x, x, dt, zero, zero, zero)
    uniform = SummaryStatistics(x, x, torch.full((B,), dt), zero, zero, zero)
    # Approximate, not exact: a float32 tensor cannot hold 1e-3 exactly, so the tensor path yields
    # 0.0010000000474974513 where the Python scalar is 0.001. That gap is the dtype, not the logic
    # (the previous .mean().item() had it too).
    assert abs(uniform.dt - scalar.dt) <= 1e-7 * scalar.dt, \
        f"a uniform dt tensor must match the scalar: {uniform.dt!r} vs {scalar.dt!r}"

    bad = torch.full((B,), dt)
    bad[2] = dt * 2.0
    try:
        SummaryStatistics(x, x, bad, zero, zero, zero)
        raise AssertionError("a non-uniform per-sample dt must raise, but was accepted")
    except ValueError as e:
        assert "non-uniform" in str(e), f"unexpected message: {e}"


def test_interp_log_returns_nan_off_the_psd_grid():
    """Off-grid frequencies must come back NaN, not a linear extrapolation off the edge bins.

    `eff_temp_ratio` divides by this, so widening freq_bounds past the PSD resolution used to grow a
    smooth, plausible, entirely fabricated T_eff/T tail -- and `check_high_freq_fdt` probes exactly
    the top of the grid, so it could pass on invented numbers.
    """
    from core.FDT.sanity import _interp_log

    x_old = torch.logspace(0, 2, 32, dtype=torch.float64)          # 1 .. 100
    y_old = torch.linspace(1.0, 10.0, 32, dtype=torch.float64)
    probe = torch.tensor([0.1, 1.0, 10.0, 100.0, 500.0], dtype=torch.float64)
    out = _interp_log(probe, x_old, y_old)

    assert torch.isnan(out[0]), "below the grid must be NaN, not extrapolated"
    assert torch.isnan(out[-1]), "above the grid must be NaN, not extrapolated"
    assert torch.isfinite(out[1:4]).all(), "in-range points must still interpolate"
    mid = _interp_log(torch.tensor([10.0], dtype=torch.float64), x_old, y_old)
    assert 1.0 <= float(mid) <= 10.0, f"in-range interpolation left the data range: {float(mid)}"


def test_nadrowski_compiled_step_matches_eager_and_keeps_its_arity():
    """The JIT fast path must stay in lockstep with the eager model.

    euler_compiled splats compiled_params() POSITIONALLY into compiled_step, and every entry is a
    same-shaped tensor -- so an inserted, dropped or reordered parameter mis-binds silently: wrong
    physics, no error. That path is CUDA-only, so nothing else in this suite exercises it on a CPU
    box and a break would ship straight into a multi-hour GPU run. torch.jit.script compiles on CPU,
    so the contract is checkable here.
    """
    import math as _math
    from core.Models.nadrowski_model import NadrowskiModel, _nadrowski_compiled_step

    B, d, T = 6, 3, 4
    col = lambda v: torch.full((B,), float(v))                     # noqa: E731
    m = NadrowskiModel(k=col(0.3), lam=col(10.0), f=col(0.5), tau=col(0.01), tau_c=col(0.001),
                       s=col(0.6), delta_e=col(1.0), beta=col(1.5), n=col(50.0), temp=col(1.2),
                       force=torch.zeros(B, 1, T), batch_size=B)

    # compiled_step(x, force_step, dW, *params, dt, sqrt_dt) -> 5 non-param inputs
    n_in = len(list(_nadrowski_compiled_step.graph.inputs()))
    assert len(m.compiled_params()) == n_in - 5, (
        f"compiled_params() has {len(m.compiled_params())} entries but compiled_step takes "
        f"{n_in - 5} between dW and dt -- the positional splat would mis-bind parameters")

    torch.manual_seed(1)
    x, dW = torch.randn(B, d), torch.randn(B, d)
    dt = 1e-4
    force_step = torch.zeros(B, 1)
    got = _nadrowski_compiled_step(x, force_step, dW, *m.compiled_params(), dt, _math.sqrt(dt))
    want = x + m.f_pure(x, force_step) * dt + m.g(x) * dW * _math.sqrt(dt)
    assert torch.equal(got, want), \
        f"compiled step diverged from eager by {(got - want).abs().max().item():g}"


def test_box_roundtrip_never_yields_a_nonfinite_latent_target():
    """The latent training target must stay finite for ANY physical theta, however extreme.

    gen_training_data records theta_transform.inv(theta) as the flow's target. A non-finite one
    would NaN the NPE loss for a whole training round, and the data-side filter cannot see it
    (the corresponding data row stays perfectly finite).

    The invariant currently holds because torch's SigmoidTransform._inverse clamps its argument to
    [tiny, 1-eps], so even a theta sitting exactly on -- or outside -- a box bound inverts finitely.
    That is a property of TORCH, not of this repo, so it is pinned here: if a version bump or a
    transform-stack change removes it, this fires instead of a silently poisoned multi-hour run.
    Covers the linear box, the log box, out-of-box values, and the rotated composition.
    """
    from core.SBI.reparam import build_box_bijection, build_rotated_bijection

    lows = torch.tensor([1.0, 1e-3, -5.0])
    highs = torch.tensor([1000.0, 1.0, 5.0])
    extreme_z = torch.tensor([[1e4, -1e4, 1e4], [40.0, -40.0, 0.0], [0.0, 0.0, 0.0]])
    outside = torch.stack([lows - 1.0, highs + 1.0, torch.tensor([0.0, -1.0, 0.0])])

    for label, mask in (("linear", None), ("log-box", torch.tensor([True, True, False]))):
        T = build_box_bijection(lows, highs, mask)
        assert torch.isfinite(T.inv(T(extreme_z))).all(), \
            f"{label} box: a saturating latent round-tripped to a non-finite target"
        assert torch.isfinite(T.inv(outside)).all(), \
            f"{label} box: an out-of-box physical value inverted to a non-finite target"

    V, _ = torch.linalg.qr(torch.randn(3, 3))
    TR = build_rotated_bijection(build_box_bijection(lows, highs), V)
    assert torch.isfinite(TR.inv(TR(extreme_z))).all(), \
        "rotated box: a saturating latent round-tripped to a non-finite target"


def test_clamp_to_box_lands_strictly_inside_and_mutates_in_place():
    """Pins the shared clamp helper that Priors/prior.py relies on before its own T_nd.inv().

    In place is part of the contract: prior.py clamps the tensor it then inverts, and any caller
    holding a VIEW (gen_training_data slices its theta into _nd/_rescale views) must observe the
    clamped values, not the originals. A helper that rebound instead of mutating would silently
    desync a caller's views from the tensor it inverted.
    """
    from core.SBI.reparam import build_box_bijection, clamp_to_box

    lows, highs = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
    T = build_box_bijection(lows, highs)
    theta = torch.tensor([[2.0, -1.0]])                            # deliberately outside the box
    view = theta[:, 1:]                                            # stand-in for the _rescale view
    returned = clamp_to_box(theta, T)

    assert returned.data_ptr() == theta.data_ptr(), "clamp_to_box returned a different tensor"
    assert float(view[0, 0]) > 0.0, \
        "the aliased view still sees the UNCLAMPED value -- clamp_to_box stopped being in place"
    assert (theta > lows).all() and (theta < highs).all(), "clamp did not land strictly inside"


def test_posterior_mode_decodes_all_three_observation_modes():
    """A saved posterior must be able to say which mode it was trained in, by sidecar or by net.

    Nothing used to read this, so a chi posterior (deliberately unrotated, hence no sidecar at all
    before this change) was byte-indistinguishable on disk from a legacy forced one.
    """
    from core.SBI import embedded_network
    from core.SBI.reparam import posterior_mode

    class _Stub:                                                   # minimal DirectPosterior stand-in
        def __init__(self, net):
            self.posterior_estimator = type("E", (), {"embedding_net": torch.nn.Sequential(net),
                                                      "condition_shape": None})()

    summary_w = len(FEATURE_LABELS) + 1
    k_pad = 6
    chi_dim = config.CHI_ELEM_W * k_pad
    for want_mode, fdim, want_k in (("spontaneous", 0, None), ("forced", 4, None),
                                    ("chi", chi_dim, k_pad)):
        kw = ({"chi_k_pad": k_pad, "chi_band": config.CHI_FREQ_BOUNDS} if want_mode == "chi" else {})
        net = embedded_network.EmbeddedNet(summary_w, 8, (16, 12), forcing_dim=fdim,
                                           forcing_layer_dims=(max(fdim * 4, 4), max(fdim * 2, 2)),
                                           merge_layer_dim=16, **kw)
        mode, got_dim, got_k = posterior_mode(_Stub(net))
        assert (mode, got_dim, got_k) == (want_mode, fdim, want_k), \
            f"decoded {(mode, got_dim, got_k)} from the trained net, expected {(want_mode, fdim, want_k)}"

        # Tier 1: an explicit sidecar always wins over decoding the net.
        side = {"mode": want_mode, "forcing_dim": fdim, "chi_k_pad": want_k}
        assert posterior_mode(_Stub(net), side) == (want_mode, fdim, want_k)

    # A chi posterior is identified by its net's own attributes or by a chi_layout sidecar -- NEVER by
    # width arithmetic. The old `fdim >= 6 and fdim % 3 == 0 -> chi` rule decoded any 6-parameter
    # drive as chi, and under the set layout width cannot identify a layout at all: 6*5 == 3*10 == 30.
    # A wide forcing_dim with neither marker must SAY it cannot tell, not guess.
    class _Bare:
        def __init__(self, fdim):
            self.posterior_estimator = type(
                "E", (), {"embedding_net": None, "condition_shape": (summary_w + fdim,)})()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posterior_mode(_Bare(chi_dim))
        raise AssertionError("an unidentifiable wide forcing_dim must raise, not guess 'chi'")
    except ValueError as e:
        assert "chi_layout" in str(e), str(e)


def test_overlay_figures_warn_instead_of_vanishing_on_a_width_mismatch():
    """The shape guard must SAY something. It used to `return` silently, bypassing even the

    surrounding except-clause's warning -- so the five overlay figures looked unimplemented rather
    than skipped, and a one-sample width disagreement made that the expected path, not an edge case.
    """
    import warnings as _w

    emitted = []
    cfg = object()                                                 # never reached: the guard fires first
    obs = torch.zeros((1, 100))
    traces = torch.zeros((8, 99))                                  # one sample short -- the real bug
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        orchestrator._emit_overlay_figures(cfg, obs, traces, None, None, None, False,
                                           lambda title, fig: emitted.append(title))
    assert not emitted, "no figure should be emitted when the widths disagree"
    assert any("overlay" in str(c.message).lower() for c in caught), \
        "the width mismatch was swallowed silently -- that is the bug this pins"
    assert any("99" in str(c.message) and "100" in str(c.message) for c in caught), \
        "the warning must report the ACTUAL shapes, else it cannot be diagnosed"


if __name__ == "__main__":
    failures = 0
    for test_name, fn in sorted(globals().items()):
        if test_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {test_name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {test_name}\n      {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
