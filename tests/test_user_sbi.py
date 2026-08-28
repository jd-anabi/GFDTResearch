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
import ast
import io
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
from core.Solvers import sdeint as _sdeint_mod              # noqa: E402
from core.SBI.Priors.user_prior import UserPrior                 # noqa: E402
from core.SBI.statistics import FEATURE_LABELS, SUMMARY_WIDTH    # noqa: E402
from core.config import VALID_MODELS, VALID_LABELS               # noqa: E402

_N_GROUP_G = 11
_N_SPONT = len(FEATURE_LABELS) - _N_GROUP_G   # 30

# Training-data checkpointing OFF for the whole suite (C-11). Rebound on ORCHESTRATOR, not config,
# because orchestrator does `from .config import ...` at import and would otherwise keep its snapshot.
#
# This is a TEST-INTEGRITY guard, not housekeeping. Left on, the full-pipeline tests write real
# checkpoints into Resources/Checkpoints/ keyed on a digest of their config -- and a COMPLETE
# checkpoint short-circuits generation and returns its stored rows. So the FIRST run would create
# them and every run after that would silently skip gen_training_data entirely, and the suite would
# stay green while testing nothing. Tests that want checkpointing pass an explicit `checkpoint=` dict
# with a tmpdir, which is unaffected by this.
orchestrator.TRAINING_CHECKPOINT_EVERY = 0

# Observation records OFF for the same reason (section 11.6 guardrail 1). The full-pipeline tests
# call infer_and_visualize, which records the observation it ran against -- correct for a real run,
# and litter here. Nothing else in the suite writes into Resources/; keep it that way.
orchestrator.PERSIST_OBSERVATIONS = False

# Snapshot of the checkpoint directories that existed BEFORE this suite ran, so the guard below can
# tell "the suite created one" from "the user has a real retrain checkpoint on disk". Checking for
# their mere existence would fail on any machine that has actually run a retrain -- which is every
# machine this matters on.
_CKPT_DIRS_AT_IMPORT = frozenset(
    p.name for p in config.CHECKPOINT_PATH.glob("train_*")) if config.CHECKPOINT_PATH.exists() else frozenset()


def _tiny_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds,
                    state_dep_drift=False, num_iterations=25, log_mask=None,
                    dtype=torch.float32, device=torch.device("cpu"), **_kw):
    """A tiny stand-in for pipeline.gen_prior: the same UserPrior.construct_prior, small sizes.

    ``**_kw`` IS LOAD-BEARING AND THERE ARE TWO OF THESE STUBS. A stub installed over a function must
    tolerate arguments added to that function later, or the suite dies ~40 tests in with a TypeError
    raised deep inside build_prior -- which names only the stub it hit first, so fixing that one
    reveals the second on the next run. Adding n_max/step upstream cost an hour this way on
    2026-08-27. Deliberately NOT a hand-mirrored signature: that never checked anything (it failed as
    a TypeError, not an assertion), and gen_prior's real signature is asserted directly by
    test_n_max_and_step_are_no_longer_hidden_inside_gen_prior.
    """
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
        assert obs_stats.shape[-1] == SUMMARY_WIDTH + 1    # [S | log(T)], no forcing block
        assert torch.allclose(obs_stats[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
        assert torch.isfinite(obs_stats).all()

        orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True,
                                         fig_sink=sink)
        orchestrator.validate_calibration(cfg, posterior, inferred_prior, force_prior, fig_sink=sink)

        # passive experimental path: a single unforced recording, no drive / force units
        obs_stats_e, obs_data_e, t_dim_e = orchestrator.build_experiment_obs_spontaneous(
            cfg, x_dim[0].clone(), 1.0)
        assert obs_stats_e.shape[-1] == SUMMARY_WIDTH + 1
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
    assert obs_stats.shape[-1] == SUMMARY_WIDTH + 1 + n_forcing
    assert not torch.allclose(obs_stats[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
    assert torch.isfinite(obs_stats).all()


def _tiny_nadrowski_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds,
                              state_dep_drift=False, num_iterations=25, log_mask=None,
                              dtype=torch.float32, device=torch.device("cpu"), **_kw):
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
    force_prior = orchestrator.build_forcing_prior(cfg)      # cheap: no simulation involved

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
    S = SUMMARY_WIDTH
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
    # The Fisher must use the RAW lock-ins, so its feature width is 3 per probe (log|chi|, cos, sin)
    # -- NOT the 6-channel conditioning block. `u`, `mask` and `logcyc` are all excluded because
    # fnoise is a DENOMINATOR: a channel that barely varies with theta is an amplifier, not a quiet
    # row. (`logcyc` left on 2026-08-10, C-9/C-10 -- it is an exact duplicate of A3_log_fpeak with the
    # ceiling clear, and floor() quantization with it binding.)
    assert chi_mod.CHI_FISHER_CHANNELS == ("logmag", "cos", "sin")
    for banned in ("u", "mask", "logcyc"):
        assert banned not in chi_mod.CHI_FISHER_CHANNELS, f"`{banned}` is back (trap CHI10)"
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
        assert obs_stats.shape[-1] == SUMMARY_WIDTH + 1 + orchestrator.expected_forcing_dim(cfg)
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
        assert obs_stats.shape[-1] == SUMMARY_WIDTH + 1 + K3
        assert torch.isfinite(obs_stats).all()

        orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True,
                                         fig_sink=sink)
        orchestrator.validate_calibration(cfg, posterior, inferred_prior, force_prior, fig_sink=sink)

        # experimental chi path: 1 passive + K forced recordings (GT passive trace as stand-ins).
        forced = [x_dim[0].clone() for _ in range(config.CHI_N_FREQS)]
        obs_stats_e, obs_data_e, t_dim_e = orchestrator.build_experiment_obs_chi(
            cfg, x_dim[0].clone(), forced, 1.0, 1.0)
        assert obs_stats_e.shape[-1] == SUMMARY_WIDTH + 1 + K3
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


def test_lock_in_per_row_durations_match_locking_each_row_alone():
    """``n_samples`` must make ONE batched call equal B separate calls, each over its own prefix.

    WHY (backlog C-8, handoff 4.3.5). Omega_0 spans ~4 decades inside a training batch, so a single
    shared lock-in duration has to be keyed on the FASTEST row to respect CHI_MAX_CYCLES -- which
    truncated the slow rows below CHI_MIN_CYCLES and masked them. ~48 % of training rows carried no
    live probe. Per-row durations are the fix, and the reference here is the only unambiguous one:
    lock each row in on its own, with no batching to get wrong.

    The subtle half is the MEAN. It has to be over each row's OWN prefix -- the samples that row
    actually contributes -- and the mask has to be applied AFTER demeaning. Zeroing first leaves
    ``-mean`` standing in every dead column, a step function at the prefix boundary whose energy
    lands at DC, which is exactly where a sub-resonance lock-in is most sensitive. Both wrong forms
    return finite, plausible numbers; the incommensurate omega below is what makes them differ.
    """
    torch.manual_seed(4)
    B, n, dt = 5, 3000, 0.011
    x = torch.randn(B, n, dtype=torch.float64) + torch.linspace(0, 3, n).unsqueeze(0)  # + a DC ramp
    omega = torch.tensor([0.31, 0.77, 1.19, 2.03, 3.47], dtype=torch.float64)          # incommensurate
    F0 = torch.tensor([1.0, 2.0, 0.5, 1.5, 3.0], dtype=torch.float64)
    n_samples = torch.tensor([n, 2500, 1200, 640, 137])                                # incl. full-length
    T_row = n_samples.to(torch.float64) * dt

    got = chi_mod.lock_in_batched(x, omega, F0, T_row, dt, n_samples=n_samples)
    for b in range(B):
        nb = int(n_samples[b])
        want = _lock_in_reference(x[b:b + 1, :nb], omega[b:b + 1], F0[b:b + 1], nb * dt, dt)
        assert torch.allclose(got[b], want[0], rtol=1e-9, atol=1e-12), (
            f"row {b} (n={nb}) got {got[b]} want {want[0]} -- one batched call must equal locking "
            f"that row alone over its own prefix")

    # The rows must not influence each other: re-run with a different row-0 length and every OTHER
    # row is unchanged. Without the mask this couples them through the shared mean and sum.
    other = n_samples.clone()
    other[0] = 900
    got2 = chi_mod.lock_in_batched(x, omega, F0, other.to(torch.float64) * dt, dt, n_samples=other)
    assert torch.allclose(got[1:], got2[1:], rtol=1e-12, atol=1e-14), \
        "changing one row's duration moved another row's chi -- the rows are coupled"
    assert not torch.allclose(got[0], got2[0]), "row 0's own chi did not respond to its length"


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
        assert stats.shape[-1] == SUMMARY_WIDTH + 1 + orchestrator.expected_forcing_dim(cfg), (
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


def test_lock_in_duration_is_capped_at_chi_max_cycles():
    """No probe may be locked in over more than ``chi_max_cycles`` drive cycles -- in the SIMULATED
    path and in the EXPERIMENTAL one, which must agree or the network is fed an observable it was
    not trained on.

    WHY THIS EXISTS. Every instinct about integration says a longer lock-in is a better one; measured
    (handoff 4.3.1, trap CHI9), |chi| CV runs 0.03 -> 0.63 and driven/undriven SNR 26 -> 2.3 as the
    window grows past ~30 cycles, and re-locking the SAME trace over a shorter prefix reverses it.
    The ceiling is therefore part of the measurement's definition, and it is silent when missing:
    nothing errors, the run simply reproduces posterior_chi_08042026's uninformative failure.

    Asserted on ``logcyc``, the probe's own record of the cycles it saw, because that is both the
    thing the ceiling changes and the channel the encoder uses to weigh a probe -- so a cap that
    failed to move it would be a cap the network cannot see. ``absolute_freqs`` pins the probe
    frequency instead of deriving it from a measured Omega_0, so "cycles" here is arithmetic rather
    than an estimate.
    """
    model = "NADROWSKI"
    cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)],
                              registry.state_dep_drift(model),
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()
    dtype = cfg.hw.dtype

    B, N_points, cap = 2, 4000, 20.0
    steady_idx, subsample = 200, 1
    t_fine = torch.linspace(0, (steady_idx + N_points) * cfg.dt_nd_min,
                            steady_idx + N_points, dtype=dtype)
    nd = cfg.params_tensor[0].reshape(1, -1).expand(B, -1).contiguous()
    rescale = torch.tensor([[v for v, _ in cfg.rescale_params.values()]],
                           dtype=dtype).expand(B, -1).contiguous()
    inits = cfg.inits_tensor.reshape(1, -1).expand(B, -1).contiguous()
    # A DETERMINISTIC Omega_0. peak_freq is the argmax of the rfft, so a pure tone on a bin centre
    # pins it; torch.randn would make Omega_0 the argmax of NOISE -- a different value every run, and
    # the experimental leg below (which checks the probe is in band, a band defined relative to
    # Omega_0) would then pass or fail by coin flip. It did: this test passed standalone and took the
    # suite down on the next full run.
    tt = torch.arange(N_points, dtype=dtype) * cfg.dt_exp
    f0_cell = 668.0 / (N_points * cfg.dt_exp)                    # exactly on rfft bin 668
    x_spont = torch.sin(2 * math.pi * f0_cell * tt).unsqueeze(0).expand(B, -1).contiguous()
    # Half a BIN, not an epsilon: the claim is "peak_freq landed on the intended bin", and the
    # frequency axis is float32, so an exact comparison fails on representation alone.
    assert abs(float(chi_mod.peak_freq(x_spont, cfg.dt_exp)[0]) - f0_cell) < 0.5 / (N_points * cfg.dt_exp), \
        "the synthetic passive trace must pin Omega_0, or the band below is not what it says"
    # Probe at the band's TOP edge: in band by construction, and the most cycles available in band,
    # so the ceiling binds hard (200 cycles at full length against a 20-cycle cap).
    f_abs = float(cfg.chi_freq_bounds[1]) * f0_cell
    assert f_abs < 0.9 * (0.5 / cfg.dt_exp), "the test probe must stay under Nyquist"

    def _cycles(max_cycles):
        _chi, _u, logcyc, _valid = pipeline_mod.gen_chi_raw(
            model=model, params_nd=nd, rescale=rescale, x_spont_dim=x_spont, t_fine=t_fine,
            inits=inits, rescale_idx=cfg.rescale_idx, n_segs=1, steady_idx=steady_idx,
            subsample=subsample, N_points=N_points, dt_exp=cfg.dt_exp,
            multipliers=torch.tensor([f_abs], dtype=dtype), f0_nd=cfg.chi_f0,
            absolute_freqs=True, resolution_filter=False, max_cycles=max_cycles,
            state_dep_drift=cfg.state_dep_drift, dtype=dtype, device=cfg.hw.device)
        return float(torch.exp(logcyc).max())

    uncapped = _cycles(math.inf)
    assert uncapped > cap, (
        f"the test geometry gives only {uncapped:.1f} cycles uncapped, at or below the {cap:g} "
        f"ceiling -- the assertion below would hold with no cap at all")
    capped = _cycles(cap)
    assert capped <= cap + 1e-6, (
        f"a probe was locked in over {capped:.2f} cycles against a {cap:g}-cycle ceiling. The "
        f"duration ceiling is not being applied in gen_chi_raw, so every caller past it -- training, "
        f"the Fisher rotation, the PPC -- measures a different observable than the ceiling defines.")

    # The EXPERIMENTAL path must apply the same ceiling to a supplied recording, and say that it did:
    # a bench recording is routinely far longer than the ceiling, and silently using a prefix would
    # leave the experimenter believing the whole recording was measured.
    cfg.T_obs = N_points * cfg.dt_exp
    hz = cfg.freq_si_to_cell
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obs_stats, _data, _t = orchestrator.build_experiment_obs_chi(
            cfg, x_spont[0].clone(), [(x_spont[0].clone(), f_abs / hz)],
            float(cfg.T_obs / cfg.get_unit_conversion_factor("s")), 1.0)
    assert torch.isfinite(obs_stats).all()
    assert any("ceiling" in str(c.message) for c in caught), \
        f"the experimental path truncated silently, or not at all: {[str(c.message) for c in caught]}"


def test_chi_max_cycles_must_clear_the_min_cycles_floor():
    """The floor MASKS a probe and the ceiling SHORTENS it, so a ceiling at or under the floor
    truncates every probe to below the floor and masks the entire set.

    The natural failure without this is `build_experiment_obs_chi` refusing with "none of the
    supplied recordings produced a usable probe" -- true, unhelpful, and pointing at the recordings
    rather than at the two constants that closed on each other. In training it is worse: nothing
    refuses, every chi block is all-pad, and the run trains happily on no susceptibility at all.
    """
    import dataclasses
    model = "NADROWSKI"
    cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)],
                              registry.state_dep_drift(model),
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"),
                              chi_mode=True)
    for bad in (config.CHI_MIN_CYCLES, config.CHI_MIN_CYCLES / 2):
        try:
            dataclasses.replace(cfg, chi_max_cycles=bad)
        except ValueError as e:
            assert "CHI_MIN_CYCLES" in str(e), f"the message must name both constants, got: {e}"
        else:
            raise AssertionError(
                f"chi_max_cycles={bad} was accepted against CHI_MIN_CYCLES="
                f"{config.CHI_MIN_CYCLES}: every probe would be masked in every observation.")
    dataclasses.replace(cfg, chi_max_cycles=config.CHI_MIN_CYCLES * 2)      # must not raise


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


def _oom_gen_obs_setup():
    """(kwargs for gen_obs at batch 8) -- the same CPU config the split test uses."""
    model, B, n = "NADROWSKI", 8, 80
    sdd = registry.state_dep_drift(model)
    cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)], sdd,
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()
    inits = cfg.inits_tensor.expand(B, -1).contiguous()
    n_ch = forcing.n_force_channels(model, cfg.forcing_idx, inits.shape[-1])
    return dict(model=model, params=cfg.params_tensor.expand(B, -1).contiguous(),
                t=torch.linspace(0, 1.0, n, dtype=cfg.hw.dtype), inits=inits,
                force=torch.zeros((B, n_ch, n), dtype=cfg.hw.dtype),
                n_segs=1, steady_idx=10, state_dep_drift=sdd, batch_size=B,
                var_idx=0, dtype=cfg.hw.dtype, device=cfg.hw.device)


def _flaky_gen_obs_one(fail_at_or_above, widths):
    """Stand-in for _gen_obs_one that raises a REALISTIC OOM above a width and delegates below it.

    Reports the width it was asked for via params.shape[0] rather than the positional batch_size arg:
    gen_obs' own contract is that dim 0 of params IS the batch, so this cannot drift with the
    signature. The raised error is shaped exactly like the real one -- a torch.AcceleratorError (the
    RAW DRIVER form, which is what the 2026-08 retrain actually died with) wrapped in SimulationError
    by Simulator.__sols -- so the test exercises _is_oom's chain walk, not a convenient stub.
    """
    from core.Simulator.simulator import SimulationError
    real = pipeline_mod._gen_obs_one

    def stub(*a, **k):
        b = a[1].shape[0]
        widths.append(b)
        if b >= fail_at_or_above:
            try:
                raise torch.AcceleratorError("CUDA error: out of memory")
            except RuntimeError as e:
                raise SimulationError(
                    f"NadrowskiModel euler_compiled failed after 50 steps (batch={b}, segs=1, "
                    f"device=cpu, dtype=torch.float32): AcceleratorError: CUDA error: out of memory"
                ) from e
        return real(*a, **k)
    return stub


def test_gen_obs_halves_the_batch_on_an_oom_instead_of_dying():
    """A CUDA OOM must be answered by RE-RUNNING that chunk at half the batch, not by killing a run.

    The predictive guard cannot prevent this on a shared card: it budgets from
    torch.cuda.mem_get_info(), and on Windows/WDDM that reports other processes' EVICTABLE surfaces as
    free -- measured 15037 MiB against nvidia-smi's 5814 MiB at one instant on the 2026-08 box, an
    overstatement of 9.2 GiB. The plan is a hint; this retry is the mechanism. Forced on CPU by
    patching _gen_obs_one, so halving, stitching and reporting are all exercised without a GPU.
    """
    widths, saved, floor = [], pipeline_mod._gen_obs_one, pipeline_mod._MIN_SIM_CHUNK
    try:
        pipeline_mod._MIN_SIM_CHUNK = 1              # module-level, so _gen_obs_retry re-reads it
        pipeline_mod._gen_obs_one = _flaky_gen_obs_one(4, widths)     # 8 and 4 fail; 2 succeeds
        out = pipeline_mod.gen_obs(**_oom_gen_obs_setup())
    finally:
        pipeline_mod._gen_obs_one, pipeline_mod._MIN_SIM_CHUNK = saved, floor

    assert out.shape == (1, 8, 70), f"retried gen_obs returned {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "retried gen_obs produced non-finite values"
    # Depth-first halving, and chunks that ALREADY SUCCEEDED are never recomputed.
    assert widths == [8, 4, 2, 2, 4, 2, 2], f"unexpected retry ladder {widths}"


def test_gen_obs_stops_halving_at_the_floor_and_re_raises_the_original():
    """The retry is BOUNDED. At the floor it must give up and re-raise the real SimulationError with
    its __cause__ intact -- grinding a 5000-batch round out a few rows at a time takes days, and a
    CUDA context that a sticky error has already killed must cost a few fast failures, not a hang."""
    from core.Simulator.simulator import SimulationError
    widths, saved, floor = [], pipeline_mod._gen_obs_one, pipeline_mod._MIN_SIM_CHUNK
    try:
        pipeline_mod._MIN_SIM_CHUNK = 2
        pipeline_mod._gen_obs_one = _flaky_gen_obs_one(0, widths)     # every width OOMs
        try:
            pipeline_mod.gen_obs(**_oom_gen_obs_setup())
            raise AssertionError("an unrecoverable OOM must still raise")
        except SimulationError as e:
            assert isinstance(e.__cause__, torch.AcceleratorError), \
                f"the driver error must survive as __cause__, got {e.__cause__!r}"
    finally:
        pipeline_mod._gen_obs_one, pipeline_mod._MIN_SIM_CHUNK = saved, floor
    assert widths == [8, 4, 2], f"halving must stop at the floor, got {widths}"


def test_oom_detection_covers_the_allocator_and_the_raw_driver_form():
    """torch.OutOfMemoryError and torch.AcceleratorError both derive DIRECTLY from RuntimeError and
    NEITHER subclasses the other, so no single isinstance() covers both. The 2026-07-28 cuFFT leak and
    the 2026-08 chi retrain BOTH produced the driver form -- a predicate that only knew
    torch.OutOfMemoryError would have retried on neither."""
    from core.Simulator.simulator import SimulationError
    is_oom = pipeline_mod._is_oom
    assert is_oom(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_oom(torch.AcceleratorError("CUDA error: out of memory"))
    try:                                              # how it actually arrives: wrapped by the solver
        try:
            raise torch.AcceleratorError("CUDA error: out of memory")
        except RuntimeError as e:
            raise SimulationError("NadrowskiModel euler_compiled failed after 81616 steps") from e
    except SimulationError as e:
        assert is_oom(e), "an OOM wrapped by Simulator.__sols must still be recognised"
    # ...and must NOT fire on ordinary failures, or a real bug becomes a silent retry loop.
    assert not is_oom(RuntimeError("shape '[2, 3]' is invalid for input of size 5"))
    assert not is_oom(ValueError("bounds are out of order"))


def test_a_cancel_is_not_mistaken_for_an_oom_and_retried():
    """A cooperative cancel must reach Worker.run untouched -- not be read as a resource problem and
    re-run at ever-smaller widths. WorkerCancelled derives from BaseException exactly so it slips
    past `except RuntimeError`; this pins that the RETRY layer honours that, not just __sols."""
    from core.gui.streams import WorkerCancelled
    calls, saved = [], pipeline_mod._gen_obs_one

    def cancelling(*a, **k):
        calls.append(a[1].shape[0])
        raise WorkerCancelled()
    try:
        pipeline_mod._gen_obs_one = cancelling
        try:
            pipeline_mod.gen_obs(**_oom_gen_obs_setup())
            raise AssertionError("WorkerCancelled was swallowed; the GUI cancel path is broken")
        except WorkerCancelled:
            pass
    finally:
        pipeline_mod._gen_obs_one = saved
    assert calls == [8], f"a cancel must not be retried, but the batch was re-run: {calls}"


def test_the_memory_budget_learns_from_an_oom_and_recovers():
    """The learned cap is the fix for a free-memory reading that Windows overstates by the size of the
    desktop. It must tighten below what just failed, and must climb back afterwards -- a card that was
    busy when the first OOM landed may be idle an hour later, and a run throttled to that one moment
    for days would be the wrong kind of safe."""
    saved_cap, saved_clean = pipeline_mod._BUDGET_CAP_ELEMENTS, pipeline_mod._budget_clean_runs
    try:
        pipeline_mod._BUDGET_CAP_ELEMENTS, pipeline_mod._budget_clean_runs = None, 0
        assert pipeline_mod._budget_cap() == math.inf, "no cap until something has actually failed"

        pipeline_mod._budget_note_oom(1_000_000)
        assert pipeline_mod._budget_cap() == 800_000, pipeline_mod._budget_cap()
        pipeline_mod._budget_note_oom(2_000_000)      # a LARGER failure must not loosen the cap
        assert pipeline_mod._budget_cap() == 800_000, "the cap must be a running minimum"
        pipeline_mod._budget_note_oom(500_000)
        assert pipeline_mod._budget_cap() == 400_000, "a smaller failure must tighten it further"

        for _ in range(pipeline_mod._BUDGET_RECOVER_AFTER - 1):
            pipeline_mod._budget_note_ok()
        assert pipeline_mod._budget_cap() == 400_000, "must not probe upward before the run of successes"
        pipeline_mod._budget_note_ok()
        assert pipeline_mod._budget_cap() > 400_000, "the cap must recover once the card settles"

        pipeline_mod._budget_note_oom(100_000)        # a fresh OOM resets the success counter
        assert pipeline_mod._budget_clean_runs == 0

        # A batch that SPLIT must still count as clean, and this is the deadlock the first cut had:
        # once an OOM tightens the cap, the tighter cap makes _max_sim_batch split every subsequent
        # batch, so if recovery were keyed on un-split batches no batch would ever count and the cap
        # could never climb back for the remaining days of the run.
        saved_planner = pipeline_mod._max_sim_batch
        try:
            pipeline_mod._max_sim_batch = lambda batch_size, *a, **k: 3      # force a split on CPU
            pipeline_mod._budget_clean_runs = 0
            pipeline_mod.gen_obs(**_oom_gen_obs_setup())
            assert pipeline_mod._budget_clean_runs == 1, \
                "a split batch must count toward recovery, or the cap deadlocks once it tightens"
        finally:
            pipeline_mod._max_sim_batch = saved_planner
    finally:
        pipeline_mod._BUDGET_CAP_ELEMENTS, pipeline_mod._budget_clean_runs = saved_cap, saved_clean


def _oom_at(width, seen):
    """A fake `fn(lo, hi)` for _rows_with_oom_retry: OOMs at or above `width`, else returns its rows.

    Raises the realistic shape -- a torch.AcceleratorError (the RAW DRIVER form, which is what both
    the 2026-08-10 and 2026-08-11 retrains actually produced) wrapped in SimulationError -- so the
    chain walk in _is_oom is exercised rather than a convenient stub.
    """
    from core.Simulator.simulator import SimulationError

    def fn(lo, hi):
        seen.append((lo, hi))
        if hi - lo >= width:
            try:
                raise torch.AcceleratorError("CUDA error: out of memory")
            except RuntimeError as e:
                raise SimulationError("chi batch failed: AcceleratorError: CUDA error: out of "
                                      "memory") from e
        return torch.arange(lo, hi, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    return fn


def test_batch_level_retry_halves_the_rows_and_reconstructs_the_batch():
    """The OUTER retry, for OOMs that never reach the simulator-level one.

    This is the gap that killed the 2026-08-11 retrain: a bare "CUDA error: out of memory" right
    after gen_chi_block's mask warning -- inside the batch, outside _gen_obs_one. Everything in a chi
    batch that is NOT the solver is unguarded and linear in rows (the zero-drive tensor, the per-probe
    force rebuilt K times, the (rows, N_points) int64 gather index, x_spont_dim, gen_stats), so
    halving the ROWS is what sheds them.

    Reconstruction is the load-bearing assertion, not the ladder: a retried batch must be the batch
    that would have been produced, in the same order, or the training set silently acquires a
    duplicated or reordered stratum.
    """
    seen = []
    saved = pipeline_mod._MIN_SIM_CHUNK
    try:
        pipeline_mod._MIN_SIM_CHUNK = 1
        out = pipeline_mod._rows_with_oom_retry(
            _oom_at(4, seen), 0, 8, per_row_elements=10, device=torch.device("cpu"))
    finally:
        pipeline_mod._MIN_SIM_CHUNK = saved
    assert seen == [(0, 8), (0, 4), (0, 2), (2, 4), (4, 8), (4, 6), (6, 8)], seen
    assert torch.equal(out, torch.arange(0, 8, dtype=torch.float32).unsqueeze(1).repeat(1, 3)), \
        "a retried batch must reconstruct the unsplit batch exactly, in order"


def test_batch_level_retry_stops_at_the_floor_and_lets_a_cancel_through():
    """Bounded, and narrow. Two failure modes in one test because they share a harness.

    FLOOR: at _MIN_SIM_CHUNK it must re-raise the real error with __cause__ intact rather than grind
    a multi-thousand-batch round out a handful of rows at a time.

    CANCEL: WorkerCancelled derives from BaseException precisely so it slips past `except
    RuntimeError`. If the outer retry swallowed it, a GUI cancel would become a retry storm that
    re-runs the batch at ever-smaller widths instead of stopping.
    """
    from core.Simulator.simulator import SimulationError
    from core.gui.streams import WorkerCancelled

    seen, saved = [], pipeline_mod._MIN_SIM_CHUNK
    try:
        pipeline_mod._MIN_SIM_CHUNK = 2
        try:
            pipeline_mod._rows_with_oom_retry(_oom_at(0, seen), 0, 8, per_row_elements=10,
                                              device=torch.device("cpu"))
            raise AssertionError("an unrecoverable OOM must still raise")
        except SimulationError as e:
            assert isinstance(e.__cause__, torch.AcceleratorError), repr(e.__cause__)
    finally:
        pipeline_mod._MIN_SIM_CHUNK = saved
    assert seen == [(0, 8), (0, 4), (0, 2)], f"halving must stop at the floor, got {seen}"

    calls = []

    def cancelling(lo, hi):
        calls.append((lo, hi))
        raise WorkerCancelled()
    try:
        pipeline_mod._rows_with_oom_retry(cancelling, 0, 8, per_row_elements=10,
                                          device=torch.device("cpu"))
        raise AssertionError("WorkerCancelled was swallowed; the GUI cancel path is broken")
    except WorkerCancelled:
        pass
    assert calls == [(0, 8)], f"a cancel must not be retried, but the batch was re-run: {calls}"


def test_gen_training_data_recovers_from_an_oom_outside_the_simulator():
    """The WIRING of the batch-level retry, end to end through a real gen_training_data call.

    The driver is tested separately with a fake fn; this is the half where a bug would actually live,
    because the closure has to slice ~30 batch-level names down to its row range and any one of them
    left at full width breaks the batch. That is not hypothetical: `inits` is passed POSITIONALLY to
    gen_chi_block, so it survived the keyword-based slicing sweep and surfaced here as "Batch size: 2
    cannot differ from dim 0 of parameters tensor" -- loudly, but only because the simulator happens
    to cross-validate those two. A name that broadcast instead would have produced plausible, wrong
    training rows.

    gen_stats is the thing patched to fail because it sits INSIDE the closure and OUTSIDE
    _gen_obs_one, which is precisely the gap the 2026-08-11 retrain died in.
    """
    from core.Simulator.simulator import SimulationError
    model = "NADROWSKI"
    cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)],
                              registry.state_dep_drift(model),
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()

    class _FixedPrior:
        def __init__(self, theta): self.theta = theta
        def sample(self, shape): return self.theta.expand(shape[0], -1).clone()

    n_grid, steady_idx, run_size = 12_000, 500, 4
    t = torch.linspace(0, n_grid * cfg.dt_nd_min, n_grid, dtype=cfg.hw.dtype)
    real_stats, saved_floor = pipeline_mod.gen_stats, pipeline_mod._MIN_SIM_CHUNK

    def flaky(x_spont, *a, **k):
        if x_spont.shape[0] > run_size // 2:
            try:
                raise torch.AcceleratorError("CUDA error: out of memory")
            except RuntimeError as e:
                raise SimulationError("gen_stats: AcceleratorError: CUDA error: out of memory") from e
        return real_stats(x_spont, *a, **k)

    try:
        pipeline_mod.gen_stats, pipeline_mod._MIN_SIM_CHUNK = flaky, 1
        data, thetas = pipeline_mod.gen_training_data(
            model, _FixedPrior(cfg.ground_truth_tensor.reshape(1, -1)), None, t,
            run_size=run_size, n_runs=2, steady_idx=steady_idx, dt_nd_min=cfg.dt_nd_min,
            nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
            dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
            t_scale_bounds=cfg.t_scale_bounds, state_dep_drift=cfg.state_dep_drift,
            chi_mode=True, chi_f0=config.CHI_F0, chi_freq_bounds=config.CHI_FREQ_BOUNDS,
            chi_k_pad=4, chi_max_cycles=config.CHI_MAX_CYCLES,
            n_vars=cfg.inits_tensor.shape[-1], dtype=cfg.hw.dtype, device=cfg.hw.device)
    finally:
        pipeline_mod.gen_stats, pipeline_mod._MIN_SIM_CHUNK = real_stats, saved_floor

    assert data.shape[0] == 2 * run_size, f"the retry lost or duplicated rows: {tuple(data.shape)}"
    assert thetas.shape[0] == 2 * run_size, tuple(thetas.shape)
    assert torch.isfinite(data).all(), "a retried batch produced non-finite conditioning"


def test_zscore_check_is_capped_and_reaches_sbis_own_binding():
    """sbi calls warn_if_zscoring_changes_data UNCONDITIONALLY on the full training tensor.

    At the retrain's size (10.24M x 114 float32) that is torch.unique(x, dim=0), then a full z-score,
    then torch.unique AGAIN -- >= 13 GiB, on the GPU, at the END of a multi-day generation run that is
    not checkpointed. Capping it to a strided subsample keeps the diagnostic (it is a PROPORTION test
    with a 10% tolerance) at a fraction of the cost.

    The patch has to reach npe_base's OWN binding: it does `from sbi.utils import
    warn_if_zscoring_changes_data`, so its call resolves against npe_base's globals and patching
    sbi.utils.sbiutils alone would change nothing. That is the half of this most likely to rot -- sbi
    has already moved the module once -- so it is asserted directly.
    """
    import sbi.utils.sbiutils as sbiutils
    from sbi.inference.trainers.npe import npe_base

    original = sbiutils.warn_if_zscoring_changes_data
    assert npe_base.warn_if_zscoring_changes_data is original, \
        "npe_base no longer holds its own binding; re-check how the patch must reach it"

    # Stand a recorder in for the real function in BOTH modules before entering the block. Both,
    # because the contextmanager finds its targets by object IDENTITY against sbiutils' current
    # binding -- so a recorder installed in only one place would make the scan miss the other and the
    # test would fail for its own reasons rather than the code's.
    seen = []
    recorder = lambda x, *a, **k: seen.append(x.shape[0])            # noqa: E731
    sbiutils.warn_if_zscoring_changes_data = recorder
    npe_base.warn_if_zscoring_changes_data = recorder
    try:
        with pipeline_mod._capped_zscore_check(max_rows=1000):
            assert npe_base.warn_if_zscoring_changes_data is not original, \
                "the patch did not reach npe_base's binding -- it would be a no-op in the real call"
            npe_base.warn_if_zscoring_changes_data(torch.zeros(10_000, 4))
        assert seen == [1000], f"expected a 1000-row subsample, got {seen}"
        # Restored even when the block raises -- otherwise one failed run leaves sbi monkeypatched
        # for every later panel and script in the process.
        try:
            with pipeline_mod._capped_zscore_check(max_rows=1000):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert npe_base.warn_if_zscoring_changes_data is sbiutils.warn_if_zscoring_changes_data
    finally:
        sbiutils.warn_if_zscoring_changes_data = original
        npe_base.warn_if_zscoring_changes_data = original


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
    import ast as _ast, inspect as _inspect, textwrap as _textwrap

    # The clear moved into pipeline._release_device_memory (2026-08-27), so this asks the question
    # structurally rather than by grepping for a literal that a refactor can move: gen_training_data
    # must reach the release, and must NOT be the caller that turns the plan clear off. That flag
    # exists for the HOT loops -- gen_stats' sub-batches and gen_chi_raw's probes, where the whole
    # point is intra-batch plan reuse -- and using it here would silently reinstate the leak.
    fn = _ast.parse(_textwrap.dedent(_inspect.getsource(pipeline_mod.gen_training_data)))
    calls = [n for n in _ast.walk(fn)
             if isinstance(n, _ast.Call) and getattr(n.func, "id", None) == "_release_device_memory"]
    assert calls, ("gen_training_data must release device memory per batch through "
                   "_release_device_memory; empty_cache() alone does NOT free the cuFFT plans")
    assert any(not any(k.arg == "plans" and getattr(k.value, "value", None) is False
                       for k in c.keywords) for c in calls), (
        "gen_training_data's per-batch release passes plans=False, so the cuFFT plan cache is never "
        "cleared: N_points_k changes every batch, cross-batch reuse is zero, and ~2 MB per signature "
        "accumulates until the run dies on a raw driver cudaErrorMemoryAllocation")

    if not torch.cuda.is_available():
        return                                       # mechanism check below is CUDA-only
    cache = torch.backends.cuda.cufft_plan_cache
    cache.clear()
    for n in (4096, 4097, 5003, 6151):               # distinct lengths -> distinct plans
        chi_mod.peak_freq(torch.randn(8, n, device="cuda"), 1e-3)
    assert cache.size > 0, "expected distinct transform lengths to mint distinct cuFFT plans"
    # END TO END: the helper itself must really free them, which is the property the structural
    # check above can only point at. This is what the old literal-string assertion stood in for.
    pipeline_mod._release_device_memory(torch.device("cuda"))
    assert cache.size == 0, "_release_device_memory did not release the cached cuFFT plans"


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

    summary_w = SUMMARY_WIDTH + 1
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


# ── C-11: reproducibility harness + the resume seam ──────────────────────────────────────────────
_TD_MODES = ("chi", "forced", "spontaneous")


def _td_cfg():
    """A tiny CPU Nadrowski config for gen_training_data. Same shape the OOM wiring test uses."""
    model = "NADROWSKI"
    cfg = cli.make_sim_config(model, VALID_LABELS[VALID_MODELS.index(model)],
                              registry.state_dep_drift(model),
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.hw = config.cpu_device()
    return cfg


class _FixedTdPrior:
    def __init__(self, theta): self.theta = theta
    def sample(self, shape): return self.theta.expand(shape[0], -1).clone()


def _gen_td(mode, *, seed=0, n_runs=3, run_size=4, **over):
    """``gen_training_data`` under a FULLY pinned RNG.

    Seeds numpy as well as torch, and that is not belt-and-braces: ``inits`` comes from
    ``np.random.randint`` (trap X8), which ``torch.manual_seed`` does not touch, so without the numpy
    seed two runs of identical code differ and every bit-identity claim below would be vacuous.
    """
    import numpy as np
    cfg = _td_cfg()
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_grid, steady_idx = 12_000, 500
    t = torch.linspace(0, n_grid * cfg.dt_nd_min, n_grid, dtype=cfg.hw.dtype)
    force_prior = orchestrator.build_forcing_prior(cfg) if mode == "forced" else None
    kw = dict(
        run_size=run_size, n_runs=n_runs, steady_idx=steady_idx, dt_nd_min=cfg.dt_nd_min,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds, state_dep_drift=cfg.state_dep_drift,
        spontaneous_only=(mode == "spontaneous"), chi_mode=(mode == "chi"),
        n_vars=cfg.inits_tensor.shape[-1], dtype=cfg.hw.dtype, device=cfg.hw.device)
    if mode == "chi":
        kw.update(chi_f0=config.CHI_F0, chi_freq_bounds=config.CHI_FREQ_BOUNDS,
                  chi_k_pad=4, chi_max_cycles=config.CHI_MAX_CYCLES)
    kw.update(over)
    return pipeline_mod.gen_training_data(
        cfg.model, _FixedTdPrior(cfg.ground_truth_tensor.reshape(1, -1)), force_prior, t, **kw)


def test_the_suite_does_not_write_checkpoints_into_the_real_resources_tree():
    """A guard on the guard, because the failure mode is silent and severe.

    The full-pipeline tests call orchestrator.build_posterior for real. With checkpointing at its
    production default they write into Resources/Checkpoints/ keyed on a digest of their config --
    and a COMPLETE checkpoint short-circuits generation and returns its stored rows. So the first
    suite run would create them and EVERY RUN AFTER would skip gen_training_data entirely while
    reporting a pass: the suite would be green and testing nothing. Observed, not theorised -- three
    such directories (spontaneous / forced / chi, run_size=8, n_runs=2) appeared in the tree the
    first time this suite ran with C-11 enabled.
    """
    from core import orchestrator as _orch
    assert _orch.TRAINING_CHECKPOINT_EVERY == 0, (
        "this suite must disable training-data checkpointing at import (see the module header); "
        f"it is {_orch.TRAINING_CHECKPOINT_EVERY}")
    ck = config.CHECKPOINT_PATH
    now = frozenset(p.name for p in ck.glob("train_*")) if ck.exists() else frozenset()
    # NEW ones only. A user who has run a retrain has a train_* directory sitting there legitimately,
    # and failing on its existence would make this suite un-runnable on exactly the machines that
    # matter. What must never happen is the suite ADDING one.
    created = sorted(now - _CKPT_DIRS_AT_IMPORT)
    assert not created, (
        f"the suite created training checkpoints in {ck}: {created}. Later runs would reuse those "
        f"rows instead of generating them, and the suite would go green without testing anything.")


def test_gen_training_data_is_reproducible_from_a_seed_in_every_mode():
    """THE GATE for any change to gen_training_data's loop, and the reason C-11 could be built at all.

    Two runs of identical code, identical seeds, must be bit-identical in all three branches. This is
    the same harness the 2026-08-11 batch-retry refactor was held to; it is what makes "the resume is
    bit-identical to an uninterrupted run" a meaningful claim rather than a hopeful one.

    If this ever fails, do NOT chase the resume tests -- the function stopped being reproducible and
    every downstream bit-identity assertion below is measuring nothing.
    """
    for mode in _TD_MODES:
        a_x, a_th = _gen_td(mode, seed=7)
        b_x, b_th = _gen_td(mode, seed=7)
        assert torch.equal(a_x, b_x), f"{mode}: conditioning differs between two identical runs"
        assert torch.equal(a_th, b_th), f"{mode}: targets differ between two identical runs"
        c_x, _ = _gen_td(mode, seed=8)
        assert not torch.equal(a_x, c_x), f"{mode}: the seed does not affect the output at all"


class _KillRun(BaseException):
    """Stands in for WorkerCancelled: a BaseException, so it passes through the OOM retries'
    deliberately narrow `except RuntimeError` exactly as a real GUI cancel does."""


def _kill_at(batch_index):
    """Patch gen_stats to abort partway through batch `batch_index`. gen_stats is the right seam: it
    sits INSIDE the batch closure and OUTSIDE _gen_obs_one, which is where the 2026-08-11 retrain
    actually died."""
    real = pipeline_mod.gen_stats
    seen = {"n": -1, "last": None}

    def _spy(x_spont, *a, **k):
        if seen["last"] is not pipeline_mod._BATCH_TAG:
            seen["last"] = pipeline_mod._BATCH_TAG
            seen["n"] += 1
        if seen["n"] >= batch_index:
            raise _KillRun(f"killed during batch {seen['n']}")
        return real(x_spont, *a, **k)
    return real, _spy


def _ck(tmp, **over):
    d = dict(dir=tmp, identity={"model": "NADROWSKI", "run_size": 4, "n_runs": 6, "mode": "chi"},
             probe=torch.linspace(0, 1, 21, dtype=torch.float64).reshape(7, 3), V=None, every=2)
    d.update(over)
    return d


def test_a_resumed_training_run_is_bit_identical_to_an_uninterrupted_one():
    """THE test C-11 exists to pass.

    Three runs at the same seeds: one straight through, one killed partway, and a resume of that
    second one. The resume's output must equal the uninterrupted output BIT FOR BIT -- not merely
    'the right shape' and not 'statistically similar'. Anything less means the second half of a
    resumed multi-day run was drawn from a different distribution than the first, which is the exact
    failure C-11's own backlog entry calls worse than crashing.

    Exercises both write paths: the cadence write (every=2) and the on-the-way-out write in the
    BaseException handler, which is what a GUI cancel takes.
    """
    import tempfile
    from core.SBI import training_checkpoint as tc
    tmp = Path(tempfile.mkdtemp())
    n_runs, run_size, kill = 6, 4, 3

    ref_x, ref_th = _gen_td("chi", seed=11, n_runs=n_runs, run_size=run_size)

    real, spy = _kill_at(kill)
    try:
        pipeline_mod.gen_stats = spy
        try:
            _gen_td("chi", seed=11, n_runs=n_runs, run_size=run_size,
                    checkpoint=_ck(tmp / "a", resume="never"))
        except _KillRun:
            pass
        else:
            raise AssertionError("the injected kill did not propagate -- it was swallowed")
    finally:
        pipeline_mod.gen_stats = real

    st = tc.peek(tmp / "a")
    assert st and 0 < st["batches_done"] < n_runs, st
    assert st["batches_done"] == kill, (
        f"committed {st['batches_done']} batches, expected {kill}: the cadence write covered "
        f"[0,2) and the cancel handler should have added batch 2")

    got_x, got_th = _gen_td("chi", seed=11, n_runs=n_runs, run_size=run_size,
                            checkpoint=_ck(tmp / "a", resume="require"))
    assert got_x.shape == ref_x.shape, (tuple(got_x.shape), tuple(ref_x.shape))
    assert torch.equal(got_x, ref_x), (
        "a resumed run is NOT bit-identical to an uninterrupted one; "
        f"max|diff| = {float((got_x - ref_x).abs().max())}")
    assert torch.equal(got_th, ref_th), "the latent TARGETS diverged across the resume"
    assert tc.peek(tmp / "a")["complete"] is True


def test_a_resume_keeps_the_stratification_schedule_identical():
    """The (t_scale, T) schedule must come from the header, not a redraw.

    SobolEngine(scramble=True) consumes the torch global RNG AT CONSTRUCTION and _draw_and_filter's
    accept count is geometry-dependent, so a rebuilt schedule is a DIFFERENT stratification -- and
    because every row's t_scale is overridden to its batch's value, that silently changes the
    training distribution rather than crashing. This is the assertion that would catch someone
    'simplifying' the resume by re-deriving the schedule from the seed.
    """
    import tempfile
    from core.SBI import training_checkpoint as tc
    tmp = Path(tempfile.mkdtemp())
    n_runs, run_size = 6, 4
    _gen_td("chi", seed=5, n_runs=n_runs, run_size=run_size, checkpoint=_ck(tmp / "s"))
    h = tc.read_header(tmp / "s")
    before_ts, before_T = h["batch_t_scales"].clone(), h["batch_Ts"].clone()

    real, spy = _kill_at(2)
    try:
        pipeline_mod.gen_stats = spy
        try:
            _gen_td("chi", seed=5, n_runs=n_runs, run_size=run_size,
                    checkpoint=_ck(tmp / "k", resume="never"))
        except _KillRun:
            pass
    finally:
        pipeline_mod.gen_stats = real
    _gen_td("chi", seed=5, n_runs=n_runs, run_size=run_size,
            checkpoint=_ck(tmp / "k", resume="require"))

    h2 = tc.read_header(tmp / "k")
    assert torch.equal(h2["batch_t_scales"], before_ts), "the schedule changed across the seam"
    assert torch.equal(h2["batch_Ts"], before_T)
    # write-once: a resume must not rewrite the header at all
    assert torch.equal(tc.read_header(tmp / "k")["inits"], h["inits"])


def test_resume_modes_never_and_require_refuse_the_wrong_situation():
    """'never' must not clobber a live checkpoint -- days of simulation behind a typo -- and
    'require' must not silently start from zero when the thing it was told to resume is absent."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    try:
        _gen_td("chi", seed=3, n_runs=2, run_size=4, checkpoint=_ck(tmp / "none", resume="require"))
    except ValueError as e:
        assert "no resumable checkpoint" in str(e), e
    else:
        raise AssertionError("resume='require' accepted a missing checkpoint")

    _gen_td("chi", seed=3, n_runs=2, run_size=4,
            checkpoint=_ck(tmp / "live", identity={"model": "NADROWSKI", "run_size": 4,
                                                   "n_runs": 2, "mode": "chi"}, every=1))
    try:
        _gen_td("chi", seed=3, n_runs=2, run_size=4,
                checkpoint=_ck(tmp / "live", identity={"model": "NADROWSKI", "run_size": 4,
                                                       "n_runs": 2, "mode": "chi"},
                               every=1, resume="never"))
    except ValueError as e:
        assert "already exists" in str(e), e
    else:
        raise AssertionError("resume='never' silently overwrote a completed checkpoint")


def test_a_complete_checkpoint_short_circuits_generation_entirely():
    """The 'died during NN TRAINING' path, which is the expensive one to get wrong.

    Data generation is the multi-day part; the flow fit that follows is hours. If training dies after
    generation completed -- an OOM in append_simulations, a bad hyperparameter, a reboot -- re-running
    must reuse the finished rows rather than re-simulate for days. A complete checkpoint therefore
    runs ZERO batches and returns exactly what it stored.
    """
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    ref_x, ref_th = _gen_td("chi", seed=31, n_runs=3, run_size=4, checkpoint=_ck(tmp / "c", every=1))

    real = pipeline_mod.gen_stats
    calls = []

    def _count(x_spont, *a, **k):
        calls.append(1)
        return real(x_spont, *a, **k)
    try:
        pipeline_mod.gen_stats = _count
        got_x, got_th = _gen_td("chi", seed=31, n_runs=3, run_size=4,
                                checkpoint=_ck(tmp / "c", every=1))
    finally:
        pipeline_mod.gen_stats = real

    assert not calls, f"a completed checkpoint re-simulated {len(calls)} time(s) instead of loading"
    assert torch.equal(got_x, ref_x) and torch.equal(got_th, ref_th)


def test_checkpointing_off_writes_nothing_and_changes_nothing():
    """checkpoint=None is the whole backward-compatibility story: analysis.gen_cal_data,
    scripts/chi_mask_audit and every pre-C-11 call site pass nothing and must be untouched -- same
    bytes out, and no disk written."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    from core import config as _cfg
    saved, _cfg.CHECKPOINT_PATH = _cfg.CHECKPOINT_PATH, tmp
    try:
        a_x, a_th = _gen_td("chi", seed=21, n_runs=2, run_size=4)
        b_x, b_th = _gen_td("chi", seed=21, n_runs=2, run_size=4)
        assert torch.equal(a_x, b_x) and torch.equal(a_th, b_th)
        assert not any(tmp.iterdir()), f"checkpointing was off but something was written: "\
                                       f"{[p.name for p in tmp.iterdir()]}"
    finally:
        _cfg.CHECKPOINT_PATH = saved


# ── C-11: the atomic write and the checkpoint store (pure, no simulation) ────────────────────────
def _ckpt_ident(**over):
    base = {"model": "NADROWSKI", "run_size": 4, "n_runs": 6, "chi_mode": True, "chi_k_pad": 12,
            "t_scale_bounds": (1.0, 40.0), "nd_dim": 10}
    base.update(over)
    return base


def test_atomic_torch_save_never_leaves_a_partial_file():
    """The destination is either the whole old file or the whole new one.

    A failure mid-write must leave the PREVIOUS content readable, because the checkpoint's commit
    point is a single small file that is rewritten every interval -- if that write can be torn, the
    thing meant to survive a crash is itself the thing a crash corrupts.
    """
    import tempfile
    from core.Helpers import file_manager
    d = Path(tempfile.mkdtemp())
    p = d / "state.pt"
    file_manager.atomic_torch_save({"batches_done": 1}, p)
    assert torch.load(str(p), weights_only=False)["batches_done"] == 1

    real_save = torch.save

    def _boom(obj, f, *a, **k):
        real_save(obj, f, *a, **k)          # the tmp file really is written...
        raise OSError("disk full")          # ...and then the write fails before the rename
    torch.save = _boom
    try:
        try:
            file_manager.atomic_torch_save({"batches_done": 2}, p)
        except OSError:
            pass
        else:
            raise AssertionError("the injected failure did not propagate")
    finally:
        torch.save = real_save
    assert torch.load(str(p), weights_only=False)["batches_done"] == 1, \
        "a failed write clobbered the previous checkpoint -- the one thing it must never do"
    file_manager.atomic_torch_save({"batches_done": 3}, p)
    assert torch.load(str(p), weights_only=False)["batches_done"] == 3
    # The partial temp is cleaned up. Harmless if left (nothing reads a .tmp), but a checkpoint
    # writes every N batches, so a recurring failure would litter one per attempt beside the file it
    # failed to replace -- precisely where someone is trying to read a failing run.
    assert not (d / "state.pt.tmp").exists(), "a failed write left its temp file behind"


def test_checkpoint_identity_digest_is_stable_and_field_sensitive():
    """The digest routes a config to its directory, so it must not depend on dict order or on
    tuple-vs-list, and must change when any identity field changes -- otherwise two different runs
    share a directory and their rows interleave."""
    from core.SBI import training_checkpoint as tc
    a = _ckpt_ident()
    assert tc.identity_digest(a) == tc.identity_digest(dict(reversed(list(a.items()))))
    assert tc.identity_digest(a) == tc.identity_digest(_ckpt_ident(t_scale_bounds=[1.0, 40.0]))
    for key, val in (("run_size", 8), ("n_runs", 123), ("chi_k_pad", 6), ("model", "HOPF"),
                     ("chi_mode", False)):
        assert tc.identity_digest(a) != tc.identity_digest(_ckpt_ident(**{key: val})), key
    assert len(tc.identity_digest(a)) == 12


def test_a_rebuilt_prior_routes_to_a_different_checkpoint():
    """The box does not identify a PRIOR, and the training rows are drawn from the prior.

    `_gmm_fingerprint`'s own docstring says two runs over the same box produce different fits. So
    without the fingerprint in the identity, rebuilding the prior and restarting would resume into
    the SAME directory and splice rows drawn from two different distributions -- with every declared
    field still matching, so nothing would complain. This is also what makes handoff 4.1's "save your
    prior and reuse it" a guard rather than advice: without it that instruction would be untrue.
    """
    from core import cli as _cli, registry as _reg
    from core.SBI import training_checkpoint as tc
    m = "NADROWSKI"
    cfg = _cli.make_sim_config(m, VALID_LABELS[VALID_MODELS.index(m)], _reg.state_dep_drift(m),
                               str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))

    def _gmm(seed):                       # same box, different fit -- exactly the rebuild case
        torch.manual_seed(seed)
        mix = torch.distributions.Categorical(probs=torch.rand(3))
        comp = torch.distributions.MultivariateNormal(torch.randn(3, 13),
                                                      torch.eye(13).expand(3, 13, 13))
        return torch.distributions.MixtureSameFamily(mix, comp)

    a, b = _gmm(1), _gmm(2)
    ia = orchestrator.training_identity(cfg, a, 2048, 5000)
    ib = orchestrator.training_identity(cfg, b, 2048, 5000)
    assert ia["prior_fingerprint"] and ia["prior_fingerprint"] != ib["prior_fingerprint"]
    assert tc.resolve_dir(ia) != tc.resolve_dir(ib), \
        "a rebuilt prior resolved to the SAME checkpoint directory; a resume would mix two priors"
    assert tc.resolve_dir(ia) == tc.resolve_dir(orchestrator.training_identity(cfg, a, 2048, 5000)), \
        "the same prior must resolve to the same directory, or a resume can never find its rows"




def test_fisher_eigenbasis_can_return_the_eigenvalues_its_columns_are_sorted_by():
    """V alone is an ORDERING of directions; the eigenvalues are the SCALE, and the scale is the
    question. "direction 12 is the least constrained" is compatible both with an experiment that
    measures everything tolerably (spread ~3x) and with one that measures four things and returns the
    prior for the rest (spread 1e6). They used to be computed here and thrown away, and recovering
    them afterwards costs a full Fisher re-run -- so the 2026-08-25 posterior can be decomposed only
    up to an ordering. See scripts/posterior_identifiability.py.
    """
    from core.SBI.reparam import fisher_eigenbasis

    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))
    planted = torch.tensor([100.0, 10.0, 1.0, 0.1, 1e-9], dtype=torch.float64)
    F = Q @ torch.diag(planted) @ Q.T

    V, ev = fisher_eigenbasis(F, with_values=True)
    assert torch.allclose(ev, planted, rtol=1e-6), f"eigenvalues not recovered: {ev}"
    assert bool((ev[:-1] >= ev[1:]).all()), "eigenvalues are not descending"
    for j in range(5):
        assert torch.allclose(F @ V[:, j], ev[j] * V[:, j], atol=1e-6), \
            f"column {j} is not the eigenvector of eigenvalue {float(ev[j])}"
    # The near-null direction must be LAST -- every reader of V depends on that ordering.
    assert int(ev.argmin()) == 4

    # and the one-argument form is unchanged, because build_rotated_bijection and the scripts use it
    assert torch.equal(fisher_eigenbasis(F), V)

def test_the_training_budget_routes_to_a_different_checkpoint():
    """THE HAZARD THAT MAKES THE GUI's BUDGET FIELDS DANGEROUS, pinned.

    `build_posterior`'s `num_runs` / `run_size_cap` are now user-reachable from the Posterior tab, and
    BOTH land in the checkpoint identity, which is digested into the checkpoint's DIRECTORY NAME. So
    nudging either does not adjust a running job -- it silently routes to a directory that does not
    exist yet and starts from zero, with no error anywhere. The Posterior tab states this inline
    (PosteriorPanel._budget_checkpoint) rather than in a tooltip; this is the premise it rests on.
    """
    from core import cli as _cli, registry as _reg
    from core.SBI import training_checkpoint as tc
    m = "NADROWSKI"
    cfg = _cli.make_sim_config(m, VALID_LABELS[VALID_MODELS.index(m)], _reg.state_dep_drift(m),
                               str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    torch.manual_seed(0)
    mix = torch.distributions.Categorical(probs=torch.rand(3))
    comp = torch.distributions.MultivariateNormal(torch.randn(3, 13),
                                                  torch.eye(13).expand(3, 13, 13))
    prior = torch.distributions.MixtureSameFamily(mix, comp)

    base = tc.resolve_dir(orchestrator.training_identity(cfg, prior, 2048, 5000))
    narrower = tc.resolve_dir(orchestrator.training_identity(cfg, prior, 1024, 5000))
    fewer = tc.resolve_dir(orchestrator.training_identity(cfg, prior, 2048, 2500))

    assert base != narrower, "halving the batch width kept the SAME checkpoint directory"
    assert base != fewer, "halving the batch count kept the SAME checkpoint directory"
    assert narrower != fewer
    assert base == tc.resolve_dir(orchestrator.training_identity(cfg, prior, 2048, 5000)), \
        "the same budget must resolve to the same directory, or nothing could ever resume"


def test_build_posterior_takes_the_budget_as_arguments_because_the_constants_are_snapshotted():
    """orchestrator does `from .config import TRAINING_NUM_RUNS, TRAINING_RUN_SIZE`, so both are bound
    at IMPORT. A caller that "configures" a run by writing config.TRAINING_NUM_RUNS = 200 changes
    nothing and gets 5000 batches with no warning -- days of simulation, silently.

    This is why the budget is a pair of parameters. The signature check is the cheap half; the second
    assertion is the one that would catch someone "simplifying" it back to reading the module.
    """
    import inspect
    sig = inspect.signature(orchestrator.build_posterior).parameters
    for name in ("num_runs", "run_size_cap"):
        assert name in sig, f"build_posterior lost its {name} parameter"
        assert sig[name].default is None, f"{name} must default to None (= follow the config constant)"
        assert sig[name].kind is inspect.Parameter.KEYWORD_ONLY, f"{name} must be keyword-only"

    src = inspect.getsource(orchestrator.build_posterior)
    body = src[src.index('"""', src.index('"""') + 3):]      # past the docstring
    for frozen in ("TRAINING_NUM_RUNS if num_runs is None", "TRAINING_RUN_SIZE if run_size_cap is None"):
        assert frozen in body, \
            f"build_posterior no longer resolves the budget from its arguments ({frozen!r} missing)"


def test_peak_sim_elements_is_the_formula_the_memory_planner_actually_uses():
    """The GUI shows a peak-memory estimate, and it must not carry its own copy of the cost model --
    a second copy drifts the first time either is tuned, and a display that reassures you about a
    number the planner does not use is worse than no display. So `_max_sim_batch` was refactored onto
    the public `peak_sim_elements`; this pins that they are still one formula."""
    from core.SBI import pipeline as pl

    n_fine, steady, n_vars, n_ch, n_out = 283_000, 500, 3, 1, 1
    seg = min(n_fine, config.CHUNK_LEN)
    n_keep = n_out * max(0, n_fine - steady)
    by_hand = n_vars * n_fine + n_ch * n_fine + max(n_vars * seg, n_keep)

    assert pl.peak_sim_elements(1, n_fine, steady, n_vars, n_ch, n_out) == by_hand
    assert pl.sim_keep_elements(n_fine, steady, n_out) == n_keep
    # Linear in the batch, which is the whole reason splitting the batch is the lever.
    assert pl.peak_sim_elements(2048, n_fine, steady, n_vars, n_ch, n_out) == by_hand * 2048
    # The post-transient copy is what `steady_idx` removes; a config with no transient keeps more.
    assert pl.sim_keep_elements(n_fine, 0, n_out) > pl.sim_keep_elements(n_fine, steady, n_out)

def test_a_mismatched_sibling_checkpoint_is_reported_by_name():
    """The message that stands between the user and a silent restart from zero.

    The digest routes a changed config to a NEW directory, so a rebuilt prior does not corrupt
    anything -- it just quietly starts over, days of simulation sitting unused one directory away
    with nothing on screen saying so. This is the line that names the reason, and `prior_fingerprint`
    is the field it will almost always be. It runs on the fresh-start path of every checkpointed run,
    so it must also degrade to silence rather than raise when there is nothing to report.
    """
    import tempfile
    from core.SBI import training_checkpoint as tc
    root = Path(tempfile.mkdtemp())
    base = {"model": "NADROWSKI", "run_size": 2048, "prior_fingerprint": "aaaa", "chi_k_pad": 12}
    sib = {**base, "prior_fingerprint": "bbbb"}          # same everything, prior rebuilt
    d = tc.resolve_dir(sib, root)
    tc.create(d, sib, schedule_t_scales=torch.zeros(2), schedule_Ts=torch.zeros(2),
              inits=torch.zeros(2, 3), V=None, probe=torch.zeros(0, dtype=torch.float64),
              run_size=2, n_runs=2)
    tc.save(d, from_batch=0, batch_k=2, rng={"cpu": torch.get_rng_state(), "cuda": None,
                                             "chi_gen": None},
            x_buf=torch.zeros(4, 5), th_buf=torch.zeros(4, 2), run_size=2)

    msg = tc.describe_siblings(base, root)
    assert "prior_fingerprint" in msg, msg          # names the FIELD, not just "a mismatch"
    assert d.name in msg and "2 batches" in msg, msg
    # silence when there is nothing to say -- this runs on every fresh checkpointed start
    assert tc.describe_siblings(base, Path(tempfile.mkdtemp())) == ""
    assert tc.describe_siblings(base, root / "does-not-exist") == ""
    assert tc.describe_siblings(sib, root) == "", "a run must not report ITSELF as a mismatch"


def test_checkpoint_shards_do_not_serialize_the_whole_accumulator():
    """A shard must cost its own rows, not the whole preallocated buffer.

    torch.save of a slice VIEW serialises the entire underlying storage -- measured 8,001,492 bytes
    for a 100-row view of a 200k-row buffer against 5,566 for the same rows cloned, and
    .contiguous() does NOT help because a row-slice of a contiguous 2-D tensor is already contiguous
    and stays a view. At the production shape that is the difference between ~47 MB and ~4.35 GiB per
    checkpoint, i.e. hundreds of GiB over a run, and it presents as "checkpointing is slow" rather
    than as a correctness failure, so nothing else would catch it.
    """
    import tempfile
    from core.SBI import training_checkpoint as tc
    # A buffer big enough that "rows" and "whole buffer" cannot be confused by torch.save's ~1.5 KB
    # of fixed zip/pickle overhead: 8.2 MB of buffer against 8 KB of rows.
    run_size, n_runs, width = 4, 2000, 256
    d = Path(tempfile.mkdtemp()) / "ck"
    ident = _ckpt_ident(run_size=run_size, n_runs=n_runs)
    tc.create(d, ident, schedule_t_scales=torch.zeros(n_runs), schedule_Ts=torch.zeros(n_runs),
              inits=torch.zeros(run_size, 3), V=None, probe=torch.zeros(0, dtype=torch.float64),
              run_size=run_size, n_runs=n_runs)
    x = torch.zeros(n_runs * run_size, width)
    th = torch.zeros(n_runs * run_size, 3)
    tc.save(d, from_batch=0, batch_k=2, rng={"cpu": torch.get_rng_state(), "cuda": None,
                                             "chi_gen": None}, x_buf=x, th_buf=th, run_size=run_size)
    shard = next((d / "shards").glob("x_000000_000002.pt"))
    size = shard.stat().st_size
    rows_bytes = 2 * run_size * width * 4                        # 8,192
    whole_bytes = x.numel() * 4                                  # 8,192,000
    assert size < rows_bytes + 16_384, (
        f"shard is {size} bytes for {rows_bytes} bytes of rows -- it serialised the whole "
        f"{whole_bytes}-byte buffer (missing .clone() in training_checkpoint.save)")
    assert size < whole_bytes // 100, (size, whole_bytes)        # the failure mode, stated directly


def test_a_checkpoint_round_trips_its_rows_and_ignores_orphan_shards():
    """load_rows walks the COMMITTED ranges, not the directory, so shards written just before a crash
    (step 1 of the commit order, with the state commit in step 3 never reached) are ignored rather
    than silently appended as extra training rows."""
    import tempfile
    from core.SBI import training_checkpoint as tc
    run_size, n_runs, width = 4, 6, 5
    d = Path(tempfile.mkdtemp()) / "ck"
    ident = _ckpt_ident(run_size=run_size, n_runs=n_runs)
    tc.create(d, ident, schedule_t_scales=torch.arange(float(n_runs)),
              schedule_Ts=torch.arange(float(n_runs)) * 2, inits=torch.zeros(run_size, 3), V=None,
              probe=torch.zeros(0, dtype=torch.float64), run_size=run_size, n_runs=n_runs)
    x = torch.arange(n_runs * run_size * width, dtype=torch.float32).reshape(n_runs * run_size, width)
    th = torch.arange(n_runs * run_size * 2, dtype=torch.float32).reshape(n_runs * run_size, 2)
    rng = {"cpu": torch.get_rng_state(), "cuda": None, "chi_gen": None}
    tc.save(d, from_batch=0, batch_k=2, rng=rng, x_buf=x, th_buf=th, run_size=run_size)
    tc.save(d, from_batch=2, batch_k=4, rng=rng, x_buf=x, th_buf=th, run_size=run_size)
    assert tc.peek(d)["batches_done"] == 4

    # An ORPHAN: rows on disk for batches [4,6) whose state commit never landed.
    from core.Helpers.file_manager import atomic_torch_save
    atomic_torch_save(x[16:24].clone(), d / "shards" / "x_000004_000006.pt")
    atomic_torch_save(th[16:24].clone(), d / "shards" / "th_000004_000006.pt")

    xr, thr = tc.load_rows(d, tc.peek(d)["batches_done"], run_size)
    assert xr.shape[0] == 16, f"orphan shard was picked up: {xr.shape}"
    assert torch.equal(xr, x[:16]) and torch.equal(thr, th[:16])

    tc.mark_complete(d, n_runs)
    assert tc.peek(d)["complete"] is True and tc.peek(d)["batches_done"] == n_runs


def test_a_checkpoint_refuses_a_config_it_was_not_written_for():
    """Every identity field, named individually, plus the parameter-transform probe. The digest
    already routes a changed config elsewhere; this is what catches a hand-moved directory, and its
    message has to say WHICH field so the fix is obvious."""
    import tempfile
    from core.SBI import training_checkpoint as tc
    run_size, n_runs = 4, 6
    d = Path(tempfile.mkdtemp()) / "ck"
    ident = _ckpt_ident(run_size=run_size, n_runs=n_runs)
    probe = torch.linspace(0, 1, 21, dtype=torch.float64).reshape(7, 3)
    tc.create(d, ident, schedule_t_scales=torch.zeros(n_runs), schedule_Ts=torch.zeros(n_runs),
              inits=torch.zeros(run_size, 3), V=None, probe=probe,
              run_size=run_size, n_runs=n_runs)
    tc.verify(d, ident, probe)                                   # the matching case must NOT raise

    for key, val in (("run_size", 8), ("chi_k_pad", 6), ("t_scale_bounds", (1.0, 20.0)),
                     ("n_runs", 12), ("nd_dim", 11)):
        try:
            tc.verify(d, {**ident, key: val}, probe)
        except ValueError as e:
            assert key in str(e), f"the message must name the field that differs: {e}"
        else:
            raise AssertionError(f"a changed '{key}' was accepted")

    try:
        tc.verify(d, ident, probe + 1.0)                          # same box? no.
    except ValueError as e:
        assert "transform" in str(e) and "max|diff|" in str(e), e
    else:
        raise AssertionError("a changed parameter transform was accepted")


def test_checkpoint_peek_survives_a_torn_state_file():
    """peek falls back to state.prev.pt rather than raising, so a crash DURING the commit costs one
    interval instead of the run. It must never raise: the caller's next move is 'start fresh', and an
    exception there would turn a recoverable checkpoint into a dead one."""
    import tempfile
    from core.SBI import training_checkpoint as tc
    run_size, n_runs = 4, 6
    d = Path(tempfile.mkdtemp()) / "ck"
    ident = _ckpt_ident(run_size=run_size, n_runs=n_runs)
    tc.create(d, ident, schedule_t_scales=torch.zeros(n_runs), schedule_Ts=torch.zeros(n_runs),
              inits=torch.zeros(run_size, 3), V=None, probe=torch.zeros(0, dtype=torch.float64),
              run_size=run_size, n_runs=n_runs)
    x = torch.zeros(n_runs * run_size, 5)
    rng = {"cpu": torch.get_rng_state(), "cuda": None, "chi_gen": None}
    tc.save(d, from_batch=0, batch_k=2, rng=rng, x_buf=x, th_buf=x, run_size=run_size)
    tc.save(d, from_batch=2, batch_k=4, rng=rng, x_buf=x, th_buf=x, run_size=run_size)

    (d / "state.pt").write_bytes(b"\x00\x01\x02 not a torch file")
    st = tc.peek(d)
    assert st is not None and st["_state_file"] == "state.prev.pt", st
    assert st["batches_done"] == 2, "fell back to the wrong generation"

    (d / "state.pt").unlink()
    (d / "state.prev.pt").unlink()
    assert tc.peek(d) is None, "no usable state must read as None, not raise"


def test_the_bijection_probe_detects_a_changed_rotation():
    """The probe is what stops a resume under a different coordinate, so it must actually SEE the
    rotation. If it were insensitive to V the guard would pass vacuously -- and V is precisely the
    thing that is not reproducible across processes (trap X10), so that is the case it exists for.

    This is also why build_posterior reuses a COMPLETE checkpoint's V rather than recomputing it: a
    fresh V would fail this very check against the rows it is about to reuse.
    """
    from core.SBI import training_checkpoint as tc
    from core.SBI.reparam import build_rotated_bijection
    from torch.distributions.transforms import AffineTransform, ComposeTransform
    dim = 4
    base = ComposeTransform([AffineTransform(loc=torch.zeros(dim), scale=torch.ones(dim) * 2.0)])
    p0 = tc.bijection_probe(base, dim)
    assert p0.shape == (7, dim) and torch.isfinite(p0).all()
    assert torch.allclose(tc.bijection_probe(base, dim), p0), "the probe must be deterministic"

    torch.manual_seed(0)
    q1, _ = torch.linalg.qr(torch.randn(dim, dim))
    torch.manual_seed(1)
    q2, _ = torch.linalg.qr(torch.randn(dim, dim))
    pa = tc.bijection_probe(build_rotated_bijection(base, q1), dim)
    pb = tc.bijection_probe(build_rotated_bijection(base, q2), dim)
    assert not torch.allclose(pa, pb, rtol=1e-6, atol=1e-6), \
        "the probe cannot tell two different rotations apart -- the resume guard would be vacuous"
    assert not torch.allclose(pa, p0, rtol=1e-6, atol=1e-6), \
        "the probe cannot tell a rotated box from an unrotated one"


def test_the_bijection_probe_works_when_the_rotation_lives_on_the_gpu():
    """The probe grid must be built on the TRANSFORM's device.

    A rotated transform holds V in OrthogonalTransform.M. With V on CUDA and the grid on the CPU,
    `x @ M` raises "Expected all tensors to be on the same device" -- a hard error inside
    build_posterior, not a silent promotion. That is the retrain's exact configuration (rotation ON,
    CUDA), it is unreachable from any CPU-only test, and it is what the chi smoke train caught after
    the Fisher rotation had already run for over an hour. Hence a GPU test.

    Skipped with a note off-GPU: the suite's runner has no skip mechanism, and a CPU box still gets
    the device-agnostic coverage from test_the_bijection_probe_detects_a_changed_rotation.
    """
    if not torch.cuda.is_available():
        print("      (no CUDA -- skipping the cross-device probe check)")
        return
    from core.SBI import training_checkpoint as tc
    from core.SBI.reparam import build_rotated_bijection
    from torch.distributions.transforms import AffineTransform, ComposeTransform
    dev, dim = torch.device("cuda"), 4
    base = ComposeTransform([AffineTransform(loc=torch.zeros(dim, device=dev),
                                             scale=torch.ones(dim, device=dev) * 2.0)])
    torch.manual_seed(0)
    V = torch.linalg.qr(torch.randn(dim, dim, device=dev))[0]
    rotated = build_rotated_bijection(base, V)          # V lives on cuda, like build_posterior's

    p = tc.bijection_probe(rotated, dim, device=dev)    # must not raise
    assert p.device.type == "cpu", "the probe must come back on the CPU so it is storable/comparable"
    assert p.shape == (7, dim) and torch.isfinite(p).all()
    # and it must equal the same computation done wholly on the CPU, so a checkpoint written on one
    # machine still verifies on another
    base_cpu = ComposeTransform([AffineTransform(loc=torch.zeros(dim), scale=torch.ones(dim) * 2.0)])
    p_cpu = tc.bijection_probe(build_rotated_bijection(base_cpu, V.cpu()), dim)
    assert torch.allclose(p, p_cpu, rtol=1e-5, atol=1e-5), (p - p_cpu).abs().max()

    # The RESUME half of the same defect. A checkpoint stores V on the CPU (portable, by design), so
    # build_posterior has to rehome it before rebuilding the rotated bijection -- otherwise the first
    # GPU resume with rotation ON crashes in the same matmul, which is the run this feature exists to
    # rescue. This asserts the round trip a resume actually performs.
    import tempfile
    d = Path(tempfile.mkdtemp()) / "ck"
    ident = {"model": "X", "run_size": 4, "n_runs": 2}
    tc.create(d, ident, schedule_t_scales=torch.zeros(2), schedule_Ts=torch.zeros(2),
              inits=torch.zeros(4, 3), V=V, probe=p, run_size=4, n_runs=2)
    V_back = tc.read_header(d)["V"]
    assert V_back.device.type == "cpu", "V must be stored on the CPU so a checkpoint is portable"
    V_home = V_back.to(device=dev, dtype=torch.float32)          # what build_posterior now does
    p_resumed = tc.bijection_probe(build_rotated_bijection(base, V_home), dim, device=dev)
    assert torch.allclose(p_resumed, p, rtol=1e-5, atol=1e-5), \
        "the rehomed rotation does not reproduce the probe it was stored with"


def test_checkpoint_rng_snapshot_round_trips_the_streams_it_owns():
    """The CPU stream and the dedicated chi generator come back bit-identical, and a CUDA state is
    refused rather than half-applied on a CPU run -- silently skipping it would leave the SDE noise
    coming from a different stream for every batch after the resume."""
    from core.SBI import training_checkpoint as tc
    cpu = torch.device("cpu")
    chi_gen = torch.Generator(device="cpu")
    chi_gen.manual_seed(20260805)
    torch.manual_seed(1234)

    snap = tc.rng_snapshot(cpu, chi_gen)
    a_global = torch.rand(5)
    a_chi = torch.rand(5, generator=chi_gen)

    torch.manual_seed(999)                       # move both streams somewhere else
    _ = torch.rand(3)
    chi_gen.manual_seed(7)
    _ = torch.rand(3, generator=chi_gen)

    tc.rng_restore(snap, cpu, chi_gen)
    assert torch.equal(torch.rand(5), a_global), "global CPU stream not restored"
    assert torch.equal(torch.rand(5, generator=chi_gen), a_chi), "chi_gen not restored"

    fake_cuda = {"cpu": torch.get_rng_state(), "cuda": [torch.zeros(16, dtype=torch.uint8)],
                 "chi_gen": None}
    try:
        tc.rng_restore(fake_cuda, cpu, None)
    except ValueError as e:
        assert "same device type" in str(e), e
    else:
        raise AssertionError("a CUDA RNG state was accepted onto a CPU run")


# ── the CUDA-graph solver fast path ───────────────────────────────────────────────────────────────
def _graph_test_model(B, T, dev):
    """A Nadrowski model with EVERY noise channel zeroed, so the dynamics are deterministic.

    That is what makes a bitwise graph-vs-eager comparison possible at all: with noise live the two
    paths draw in a different order and can only be compared statistically, which would not catch an
    off-by-one in the force index or the output slice. The drive is time-VARYING on purpose -- a
    constant force is indistinguishable under an indexing error.
    """
    from core.Models.nadrowski_model import NadrowskiModel
    o = lambda v: torch.full((B,), float(v), device=dev)
    f = torch.zeros((B, 1, T), device=dev)
    f[:, 0, :] = torch.sin(torch.linspace(0, 20, T, device=dev)).unsqueeze(0) * 0.3
    m = NadrowskiModel(o(0.8), o(3.57), o(1.32), o(0.027), o(0.0), o(0.95), o(10.), o(14.1),
                       o(50.), o(0.0), f, batch_size=B, device=dev, dtype=torch.float32)
    m._x_noise_const = torch.zeros_like(m._x_noise_const)
    m._y_noise_const = torch.zeros_like(m._y_noise_const)
    return m


def test_the_cuda_graph_step_matches_the_eager_step_bitwise():
    """The graphed solver must integrate the SAME trajectory as the eager loop.

    ~88% of solver wall-clock was CPU kernel-launch overhead (measured 54.87 -> 6.65 us/step at batch
    2048), so the step loop is now replayed from a captured CUDA graph. The graph bakes in the force
    slice offsets and writes a whole chunk of outputs at once, which is exactly where an off-by-one
    would hide -- and it would present as slightly-wrong physics, never as an error.

    T is chosen so the run is NOT a whole number of chunks: it must exercise the eager tail too.

    ⚠ THE WARM-UP BELOW IS LOAD-BEARING, and finding out why was most of the work. TorchScript's
    PROFILING EXECUTOR runs the first invocations of a scripted function unoptimised, then
    specialises and fuses -- and the fused kernel differs from the unfused one by ~1 ULP (a fused
    multiply-add against separate ops). Measured: three consecutive EAGER runs of this same
    deterministic model gave run1 != run2 (differing from row 1, max|diff| 7.5e-09) and
    run2 == run3. So "graphed == eager bitwise" is not even well posed until both are warm --
    the eager baseline is not bitwise reproducible against ITSELF. Warm first, and then all four
    comparisons (eager/eager, graph/graph, graph/eager, replayed-region) are exactly 0.0.

    This is CUDA-only and TorchScript-only, so it does not touch the CPU suite's seeded-reproducibility
    gates: `euler_compiled` is selected only on CUDA, and every CPU test runs the plain eager `euler`.
    """
    from core import config as _cfg
    from core.Solvers import sdeint as _sd
    if not torch.cuda.is_available():
        print("      (no CUDA -- skipping the graph/eager equivalence check)")
        return
    dev = torch.device("cuda")
    B, T = 128, 2 * _cfg.SOLVER_GRAPH_CHUNK + 3        # 2 full chunks + a 2-step tail
    x0 = torch.linspace(-0.2, 0.2, B, device=dev).unsqueeze(1).repeat(1, 3).contiguous()
    ts = (0.0, (T - 1) * 0.001)

    prev = _cfg.SOLVER_CUDA_GRAPHS
    try:
        _cfg.SOLVER_CUDA_GRAPHS = False
        for _ in range(3):                             # specialise the scripted step; see above
            _sd.Solver().euler_compiled(_graph_test_model(B, T, dev), x0.clone(), ts, T)
        eager = _sd.Solver().euler_compiled(_graph_test_model(B, T, dev), x0.clone(), ts, T)
        _cfg.SOLVER_CUDA_GRAPHS = True
        _sd._GRAPH_CACHE.clear()
        graphed = _sd.Solver().euler_compiled(_graph_test_model(B, T, dev), x0.clone(), ts, T)
    finally:
        _cfg.SOLVER_CUDA_GRAPHS = prev

    n_full = (T - 1) // _cfg.SOLVER_GRAPH_CHUNK * _cfg.SOLVER_GRAPH_CHUNK
    assert torch.equal(graphed[:n_full + 1], eager[:n_full + 1]), (
        "the GRAPH-REPLAYED region disagrees with the eager loop -- this is the region the capture "
        f"owns, max|diff|={float((graphed[:n_full + 1] - eager[:n_full + 1]).abs().max()):.3e}")

    assert graphed.shape == eager.shape == (T, B, 3), (graphed.shape, eager.shape)
    assert torch.equal(graphed[0], x0), "row 0 must be the initial condition, untouched"
    assert torch.isfinite(graphed).all(), "graphed trajectory went non-finite"
    assert float((graphed[-1] - graphed[0]).abs().max()) > 1e-3, \
        "the trajectory did not move -- the test would pass trivially"
    assert torch.equal(graphed, eager), \
        f"graphed and eager disagree, max|diff|={float((graphed - eager).abs().max()):.3e}"


def test_the_cuda_graph_preserves_the_rng_contract_c11_depends_on():
    """C-11's resume restores the CUDA RNG state and expects the noise stream to continue from there.

    A captured graph could plausibly have broken this in two ways: by FREEZING the noise (replaying
    identical draws, which would silently collapse every ensemble to one trajectory), or by keeping a
    private offset that torch.cuda.set_rng_state_all cannot rewind (which would break the
    bit-identical resume). Neither happens, and both are cheap to keep pinned.
    """
    from core import config as _cfg
    from core.Solvers import sdeint as _sd
    if not torch.cuda.is_available():
        print("      (no CUDA -- skipping the graph RNG contract check)")
        return
    dev = torch.device("cuda")
    B, T = 128, 2 * _cfg.SOLVER_GRAPH_CHUNK + 1
    x0 = torch.zeros((B, 3), device=dev)
    ts = (0.0, (T - 1) * 0.001)

    def run():
        from core.Models.nadrowski_model import NadrowskiModel
        o = lambda v: torch.full((B,), float(v), device=dev)
        f = torch.zeros((B, 1, T), device=dev)
        m = NadrowskiModel(o(0.8), o(3.57), o(1.32), o(0.027), o(0.268), o(0.95), o(10.), o(14.1),
                           o(50.), o(1.5), f, batch_size=B, device=dev, dtype=torch.float32)
        return _sd.Solver().euler_compiled(m, x0.clone(), ts, T)

    prev = _cfg.SOLVER_CUDA_GRAPHS
    try:
        _cfg.SOLVER_CUDA_GRAPHS = True
        _sd._GRAPH_CACHE.clear()
        a = run()
        b = run()
        assert not torch.equal(a, b), \
            "consecutive graphed runs produced IDENTICAL noise -- the RNG is frozen inside the graph"

        state = torch.cuda.get_rng_state_all()
        c = run()
        torch.cuda.set_rng_state_all(state)
        d = run()
        assert torch.equal(c, d), \
            "restoring the CUDA RNG state did not reproduce the run -- C-11's resume would break"
    finally:
        _cfg.SOLVER_CUDA_GRAPHS = prev


def test_the_graph_cache_is_not_hung_off_the_solver_class():
    """Trap X1 in a new costume.

    The graph cache is module-level, NOT a Solver attribute or a module-level Solver singleton,
    because `sdeint.Solver` must stay resolvable at CALL time for
    test_solver_failure_raises_instead_of_killing_the_process to patch it. It is also bounded --
    graph memory lives in a private pool that torch.cuda.empty_cache() cannot reclaim.
    """
    import inspect
    from core import config as _cfg
    from core.Solvers import sdeint as _sd
    src = inspect.getsource(_sd)
    assert "_GRAPH_CACHE" in src and isinstance(_sd._GRAPH_CACHE, dict)
    assert not hasattr(_sd.Solver, "_GRAPH_CACHE"), "the cache must not live on the Solver class"
    for bad in ("_SOLVER_SINGLETON", "_SOLVER = Solver()"):
        assert bad not in src, f"{bad}: a module-level Solver singleton re-opens trap X1"
    assert _cfg.SOLVER_GRAPH_CACHE_MAX >= 1
    if not torch.cuda.is_available():
        print("      (no CUDA -- cache-bound check is structural only)")
        return
    assert len(_sd._GRAPH_CACHE) <= _cfg.SOLVER_GRAPH_CACHE_MAX, "graph cache exceeded its bound"


def _strip_docstrings(tree):
    """Remove every docstring from a parsed tree, in place, and return it.

    ⚠ ``ast.unparse`` DROPS COMMENTS BUT KEEPS DOCSTRINGS -- they are real string expressions in the
    AST, not trivia. An earlier version of _unparsed claimed otherwise, and the claim went unnoticed
    because the checks that used it happened to forbid strings that appeared only in comments. It
    stopped being harmless the moment a check forbade `mem_get_info` in a function whose DOCSTRING
    explains why it does not use mem_get_info: the assertion matched the prose, exactly the
    false positive the parse was supposed to prevent (the same shape as the _local_map lesson).
    """
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and isinstance(body, list) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.fix_missing_locations(tree)


def _unparsed(obj) -> str:
    """Source for a function/method with comments AND docstrings removed -- the executable code only.

    IF YOU ASSERT ON SOURCE TEXT, PARSE IT FIRST. A check that "_local_map no longer hardcodes the
    CPU" failed on its first run against the very COMMENT that documents the fix, because the comment
    necessarily contains the string it forbids. The same false positive had already cost time once on
    the TSNPE runner check, and a THIRD time on 2026-08-28 -- see _strip_docstrings for why
    ast.unparse alone is not enough.
    """
    import inspect as _inspect, textwrap as _textwrap
    return ast.unparse(_strip_docstrings(ast.parse(_textwrap.dedent(_inspect.getsource(obj)))))


def test_n_max_and_step_are_no_longer_hidden_inside_gen_prior():
    """Two of gen_prior's four "constants" were not constants at all.

    ``n_max`` was the literal 175000 written inside the function body, silently overriding
    construct_prior's own n_max=200000 default -- nothing named it, so nothing could change it. Worse,
    ``step`` (the flood-fill's random-walk stride) was never threaded through gen_prior at all, so
    construct_prior's default won no matter what any caller asked for. Both are arguments now, and
    both default to a named config constant.

    HDBSCAN's two come with them, and they are NOT a tuning preference: the label count is handed
    straight to the GMM's n_components, so min_cluster_size decides how many MODES the prior has
    (measured on one small build: 5 -> 15 components, 60 -> 1). A prior with a different component
    count is a different prior, not a faster one.
    """
    import inspect as _inspect
    params = _inspect.signature(pipeline_mod.gen_prior).parameters
    for knob in ("n_max", "step", "min_cluster_size", "min_samples"):
        assert knob in params, f"gen_prior must accept {knob!r} rather than hiding it in the body"
        assert params[knob].default is None, (
            f"gen_prior's {knob!r} must default to None so the CALLER resolves it against config -- "
            f"a literal default here is the hardcode this test exists to prevent")

    body = _unparsed(pipeline_mod.gen_prior)
    assert "175000" not in body and "175_000" not in body, (
        "the 175000 max-sets literal is back inside gen_prior; it belongs in "
        "config.PRIOR_SWEEP_MAX_SETS, reached through the n_max argument")
    for knob in ("n_max", "step", "min_cluster_size", "min_samples"):
        assert body.count(knob) >= 2, (
            f"gen_prior accepts {knob!r} but never uses it -- accepted-and-dropped is exactly the "
            f"failure a signature check alone would miss")


def test_the_local_sweep_is_not_pinned_to_the_cpu_in_any_prior():
    """THE FLOOD-FILL HAD BEEN ON THE CPU IN ALL FOUR PRIORS while the global census ran on the GPU --
    which is backwards, because the flood-fill is the larger half of the work.

    ``_local_map`` was a @staticmethod in NadrowskiPrior, BPPrior and HopfPrior, so it could not see
    self.device even in principle, and every one of them opened by hardcoding torch.device('cpu').
    UserPrior's was an instance method and hardcoded it anyway. Measured on the real inner loop
    (1024 trajectories x 40,000 steps): 6.32 s per iteration on the CPU against 0.357 s on CUDA, a
    17.7x difference; end to end on a small build, 9.01 s -> 2.83 s.

    Asserted on PARSED source, never raw text -- see _unparsed.
    """
    from core.SBI.Priors import bp_prior, hopf_prior, nadrowski_prior
    from core.SBI.Priors.user_prior import UserPrior as _UserPrior
    import inspect as _inspect

    for cls in (nadrowski_prior.NadrowskiPrior, bp_prior.BPPrior,
                hopf_prior.HopfPrior, _UserPrior):
        fn = cls.__dict__.get("_local_map")
        assert fn is not None, f"{cls.__name__} has no _local_map of its own"
        assert not isinstance(fn, staticmethod), (
            f"{cls.__name__}._local_map is a @staticmethod, so it cannot see self.sweep_device -- "
            f"that is how the CPU hardcode survived in three priors at once")
        assert "self" in _inspect.signature(fn).parameters, \
            f"{cls.__name__}._local_map must be an instance method"
        body = _unparsed(fn)
        for bad in ("torch.device('cpu')", 'torch.device("cpu")'):
            assert bad not in body, (
                f"{cls.__name__}._local_map hardcodes {bad} again -- it must simulate on "
                f"self.sweep_device, which resolve_sweep_device already degrades to the CPU")
        assert "sweep_device" in body, (
            f"{cls.__name__}._local_map must simulate on self.sweep_device")


def test_the_sweep_device_degrades_to_the_cpu_instead_of_raising():
    """A caller's device is normally already CPU on a machine without CUDA, so this is for the case
    where one is handed a cuda device anyway: it must DEGRADE with a printed note rather than raise
    halfway through a multi-minute sweep."""
    from core.SBI.Priors import prior as _prior_mod
    import contextlib as _ctx

    assert _prior_mod.resolve_sweep_device(torch.device("cpu")).type == "cpu"

    saved = torch.cuda.is_available
    buf = io.StringIO()
    try:
        torch.cuda.is_available = lambda: False
        with _ctx.redirect_stdout(buf):
            got = _prior_mod.resolve_sweep_device(torch.device("cuda"))
    finally:
        torch.cuda.is_available = saved
    assert got.type == "cpu", "a cuda device with no CUDA must fall back, not raise"
    assert "falling back" in buf.getvalue().lower(), (
        "the fallback must SAY so -- a sweep silently running 17.7x slower than asked is the "
        "failure this whole change removed")


def test_the_sweep_and_flow_knobs_are_ARGUMENTS_because_the_constants_are_snapshotted():
    """EVERY KNOB MUST BE AN ARGUMENT, and it must reach the thing it configures.

    orchestrator does `from .config import PRIOR_SWEEP_ITERATIONS, NSF_HIDDEN_FEATURES, ...`, which
    binds all of them at IMPORT (trap X12). So a panel that "configured" a run by assigning to
    config.PRIOR_SWEEP_ITERATIONS would change nothing, the run would use the default, and there
    would be nothing in the log to say so. That is the trap the training budget was made a parameter
    for, and it applies to every knob the GUI now exposes.

    THE SIGNATURE CHECK ALONE IS NOT ENOUGH. A parameter that is accepted and then never read is
    indistinguishable from the bug, from the caller's side -- so each knob is also required to be
    USED in the body it was added to. That is the accepted-and-dropped failure mode.
    """
    import inspect as _inspect

    for fn, knobs in (
        (orchestrator.build_prior,
         ("num_iterations", "sweep_batch", "max_sets", "walk_step",
          "min_cluster_size", "min_samples")),
        (orchestrator.build_posterior,
         ("hidden_features", "num_transforms", "learning_rate", "stop_after_epochs",
          "fisher_m", "fisher_dz", "fisher_points")),
    ):
        params = _inspect.signature(fn).parameters
        body = _unparsed(fn)
        for knob in knobs:
            assert knob in params, (
                f"{fn.__name__} must accept {knob!r} as an ARGUMENT -- assigning to the config "
                f"constant is a silent no-op, because orchestrator snapshots it at import (X12)")
            assert params[knob].default is None, (
                f"{fn.__name__}'s {knob!r} must default to None, so 'not supplied' is distinct from "
                f"a value and the config constant is resolved inside the function")
            # >= 2: once binding it in the signature, at least once reading it in the body.
            assert body.count(knob) >= 2, (
                f"{fn.__name__} accepts {knob!r} and never reads it -- accepted-and-dropped, the "
                f"failure a signature check alone misses")


def test_the_prior_sweep_constants_are_named_in_config():
    """The literals that used to be buried now have names, and the GUI binds to these."""
    for name in ("PRIOR_SWEEP_MAX_SETS", "PRIOR_SWEEP_STEP", "PRIOR_SWEEP_ON_ACCELERATOR",
                 "PRIOR_CLUSTER_MIN_SIZE", "PRIOR_CLUSTER_MIN_SAMPLES"):
        assert hasattr(config, name), f"config.{name} is missing; the GUI field has nothing to bind to"
    assert config.PRIOR_SWEEP_MAX_SETS == 175_000, "the historical max-sets value must be preserved"
    assert config.PRIOR_CLUSTER_MIN_SIZE == 50 and config.PRIOR_CLUSTER_MIN_SAMPLES == 10, (
        "HDBSCAN's two decide the GMM's component count, so changing their defaults changes what "
        "prior a default build produces -- keep them at the measured historical values")


# ── 2026-08-27: the recovery path was the thing that killed the run ──────────────────────────────
def _raising_empty_cache(calls):
    """A torch.cuda.empty_cache stand-in that fails the way the real one did on 2026-08-27.

    A RAW driver torch.AcceleratorError, not torch.OutOfMemoryError -- that distinction is the whole
    of trap X6/X7 and the reason `_is_oom` carries a message test as well as a type test.
    """
    def fake():
        calls.append(1)
        raise torch.AcceleratorError("CUDA error: out of memory")
    return fake


def _cuda_release_probe(fn):
    """Run ``fn`` with device.type == 'cuda' faked past the guards in _release_device_memory.

    _release_device_memory returns early on a non-CUDA device, so on a CPU test box the guarded body
    would never execute and the regression would go unpinned. A tiny stand-in device object gets the
    body to run while the torch entry points it calls are themselves monkeypatched.
    """
    class _FakeDevice:
        type = "cuda"
    return fn(_FakeDevice())


def test_the_release_path_survives_a_failing_empty_cache():
    """THE 2026-08-27 REGRESSION. The retrain caught a real OOM at batch 351/5000, exited its except
    block, and was then killed by `torch.cuda.empty_cache()` raising an AcceleratorError of its own
    from OUTSIDE any handler -- so the run died on its own RECOVERY, and because the except clause
    had already closed, Python attached no context and the traceback said nothing about the OOM.

    A release is best-effort by definition: every caller is already recovering and its next act is to
    retry smaller. This asserts the whole ladder still completes when all three releases fail."""
    # torch.backends.cuda.cufft_plan_cache is a DEVICE-PROXYING descriptor: assigning to its .clear
    # raises before the test starts. It is left real, which costs nothing -- whatever it does, the
    # guard under test absorbs it, and that is precisely the behaviour being asserted.
    saved_empty = torch.cuda.empty_cache
    saved_drop = _sdeint_mod.drop_graph_cache
    saved_floor = pipeline_mod._MIN_SIM_CHUNK
    calls = []
    try:
        torch.cuda.empty_cache = _raising_empty_cache(calls)
        _sdeint_mod.drop_graph_cache = _raising_empty_cache(calls)
        # 1. the helper absorbs every failing release and returns normally
        _cuda_release_probe(pipeline_mod._release_device_memory)
        assert len(calls) == 2, f"both patched releases must be attempted, got {len(calls)}"
        # 2. and the retry ladder it sits inside still reconstructs the batch.
        #    _MIN_SIM_CHUNK is lowered so 8 rows can actually halve -- at the production floor of 256
        #    an 8-row range is below 2*floor and re-raises by design.
        pipeline_mod._MIN_SIM_CHUNK = 1
        seen = []
        out = pipeline_mod._rows_with_oom_retry(
            _oom_at(4, seen), 0, 8, per_row_elements=10, device=torch.device("cpu"))
        assert torch.equal(out, torch.arange(0, 8, dtype=torch.float32).unsqueeze(1).repeat(1, 3)), \
            "a release that fails throughout must still leave the batch exactly reconstructed"
    finally:
        torch.cuda.empty_cache = saved_empty
        _sdeint_mod.drop_graph_cache = saved_drop
        pipeline_mod._MIN_SIM_CHUNK = saved_floor


def test_the_oom_notice_is_printed_before_the_release():
    """The notice has to reach the log even when the release that follows it explodes.

    On 2026-08-27 the order was the other way round: `note` was captured, the release raised, and the
    only record of the ORIGINAL failure died with it. Ordering, not wording, is what this pins -- so
    it asserts the notice is present after a release that raises."""
    import contextlib as _ctx
    saved_empty = torch.cuda.empty_cache
    saved_floor = pipeline_mod._MIN_SIM_CHUNK
    buf = io.StringIO()
    try:
        torch.cuda.empty_cache = _raising_empty_cache([])
        pipeline_mod._MIN_SIM_CHUNK = 1
        seen = []
        with _ctx.redirect_stderr(buf):
            pipeline_mod._rows_with_oom_retry(
                _oom_at(4, seen), 0, 8, per_row_elements=10, device=torch.device("cpu"))
    finally:
        torch.cuda.empty_cache = saved_empty
        pipeline_mod._MIN_SIM_CHUNK = saved_floor
    text = buf.getvalue()
    assert "OUTSIDE the simulator retry" in text, f"the OOM notice was not printed:\n{text}"
    assert "out of memory" in text, f"the original error text was not carried into the notice:\n{text}"


def test_the_budget_credits_once_per_training_batch_not_once_per_gen_obs():
    """_BUDGET_RECOVER_AFTER is 32 and every description of it says "32 clean BATCHES". But gen_obs
    is what called _budget_note_ok, and a chi batch makes 1 + K of those -- one spontaneous run plus
    one per probe -- so at the production K the cap probed upward every ~3 batches instead of every
    32. An 0.8x backoff unwound in three batches and kept climbing: the throttle never held, which is
    why a busy card produced repeated OOMs rather than settling into a slower surviving state."""
    saved_cap, saved_clean = pipeline_mod._BUDGET_CAP_ELEMENTS, pipeline_mod._budget_clean_runs
    saved_tag = pipeline_mod._BATCH_TAG
    try:
        pipeline_mod._BUDGET_CAP_ELEMENTS, pipeline_mod._budget_clean_runs = 1_000_000, 0
        pipeline_mod._BATCH_TAG = "training batch 1/5000 [t_scale=1, T=1, n_fine=1, N_points=1, rows=1]"
        for _ in range(1 + 11):                     # a chi batch at K=11
            pipeline_mod._budget_note_ok()
        assert pipeline_mod._budget_clean_runs == 0, (
            f"gen_obs' own calls must not count inside a batch, got "
            f"{pipeline_mod._budget_clean_runs} credits from 12 calls")
        pipeline_mod._budget_note_ok(batch_level=True)
        assert pipeline_mod._budget_clean_runs == 1, "the batch tail must credit exactly once"

        # Outside gen_training_data there is no batch, so the historical per-call credit stands.
        pipeline_mod._BATCH_TAG = ""
        pipeline_mod._budget_note_ok()
        assert pipeline_mod._budget_clean_runs == 2, (
            "outside a training batch every call must still count -- the PPC and the prior sweeps "
            "have no batch to key on and would otherwise never recover the cap")
    finally:
        pipeline_mod._BUDGET_CAP_ELEMENTS = saved_cap
        pipeline_mod._budget_clean_runs = saved_clean
        pipeline_mod._BATCH_TAG = saved_tag


def test_dropping_the_graph_cache_is_available_and_empties_it():
    """The OOM path drops captured graphs because their memory lives in a PRIVATE pool that
    empty_cache() cannot reclaim -- and the halving retry that follows is about to capture ANOTHER
    at the reduced width, since the batch shape is part of the graph key."""
    _sdeint_mod._GRAPH_CACHE[("fake", (1, 2), 1, 50, "f32", "cuda", 0.1)] = {"graph": object()}
    assert len(_sdeint_mod._GRAPH_CACHE) >= 1
    dropped = _sdeint_mod.drop_graph_cache()
    assert dropped >= 1, "drop_graph_cache must report what it dropped"
    assert len(_sdeint_mod._GRAPH_CACHE) == 0, "the cache must be empty afterwards"


def test_the_vram_ceiling_bounds_the_plan_and_stays_out_of_the_identity():
    """SIM_VRAM_CEILING_GIB is the PROACTIVE half of the memory budget: the learned cap can only
    tighten after something has already died, and on a shared Windows card that can be hours in.

    It must NOT reach training_identity. The identity digest names the checkpoint DIRECTORY, so a
    memory knob inside it would rename the directory whenever it moved and silently restart a
    resumable multi-day run from zero -- which is exactly how 884 batches were orphaned on
    2026-08-27. A split batch is row-aligned and produces the same training distribution, so the
    knob genuinely does not describe the data."""
    dev, dt = torch.device("cpu"), torch.float32
    saved = getattr(config, "SIM_VRAM_CEILING_GIB", 0.0)
    # The LEARNED cap is a module global and the OOM tests above leave one behind; it is the other
    # term in the same min(), so without neutralising it this measures that instead of the ceiling.
    saved_cap = pipeline_mod._BUDGET_CAP_ELEMENTS
    try:
        pipeline_mod._BUDGET_CAP_ELEMENTS = None
        config.SIM_VRAM_CEILING_GIB = 0.0
        base = pipeline_mod.sim_memory_budget_elements(dev, dt)
        config.SIM_VRAM_CEILING_GIB = 2.0
        capped = pipeline_mod.sim_memory_budget_elements(dev, dt)
        assert capped == (2.0 * 2 ** 30) // 4, f"ceiling not applied in elements: {capped}"
        assert capped < base, "a 2 GiB ceiling must bind below the CPU default budget"
        config.SIM_VRAM_CEILING_GIB = 0.0
        assert pipeline_mod.sim_memory_budget_elements(dev, dt) == base, "0 must mean off"
    finally:
        config.SIM_VRAM_CEILING_GIB = saved
        pipeline_mod._BUDGET_CAP_ELEMENTS = saved_cap

    import inspect as _inspect
    src = _inspect.getsource(orchestrator.training_identity)
    assert "SIM_VRAM_CEILING" not in src, (
        "the VRAM ceiling must never enter the checkpoint identity -- it would rename the "
        "checkpoint directory and silently restart a resumable run")


def test_the_drive_is_charged_at_its_build_peak():
    """Section 8.2 recorded the planner under-counting the drive 4x -- a 2.16 GiB result with an
    8.64 GiB transient -- and named it a plausible source of trap X7's unwrapped OOMs. `_per_row` is
    where that bites: it is the number _budget_note_oom teaches the learned cap, so under-counting
    taught the cap something smaller than what actually failed.

    The multiple is derived by counting the builder's eager allocations (see the constant's comment);
    on CUDA this measures it instead."""
    assert pipeline_mod._FORCE_BUILD_PEAK_MULTIPLE >= 2, "the drive costs more than its result"
    import inspect as _inspect
    src = _inspect.getsource(pipeline_mod.gen_training_data)
    assert "_FORCE_BUILD_PEAK_MULTIPLE * n_force_ch * n_fine_total" in src, (
        "_per_row must charge the drive at its build peak, not at its result size")

    if not torch.cuda.is_available():
        print("      (no CUDA -- the 4x multiple is derived, not measured, on this box)")
        return
    dev = torch.device("cuda")
    B, T = 64, 20_000
    fparams = torch.tensor([[1.0, 0.5, 0.0, 0.0]], device=dev).expand(B, -1).contiguous()
    rparams = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device=dev).expand(B, -1).contiguous()
    t_nd = torch.linspace(0, 1.0, T, device=dev)
    fidx = {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
    ridx = {"t_scale": 0, "t_offset": 1, "f_scale": 2, "f_offset": 3}
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    before = torch.cuda.memory_allocated(dev)
    out = pipeline_mod.build_nondim_sin_force_tensor(fparams, t_nd, rparams, fidx, ridx)
    peak = torch.cuda.max_memory_allocated(dev) - before
    measured = peak / max(1, out.numel() * out.element_size())
    print(f"      measured drive build peak = {measured:.2f}x the result")
    assert measured <= pipeline_mod._FORCE_BUILD_PEAK_MULTIPLE + 0.5, (
        f"the builder now peaks at {measured:.2f}x, above the planner's charged "
        f"{pipeline_mod._FORCE_BUILD_PEAK_MULTIPLE}x -- raise the constant")


def test_an_unsaved_prior_cannot_start_a_long_checkpointed_run():
    """training_identity fingerprints the prior's fitted GMM, and that fingerprint names the
    checkpoint DIRECTORY. A prior that was never written to disk therefore produces a directory
    nothing can ever resolve again once the process exits -- so the checkpoint it spends hours
    writing is unresumable by construction, and a crash costs the entire run.

    That happened: on 2026-08-27 a run reached 884 committed batches under fingerprint
    bd307c079d14db0b, for which no file in Resources/Priors exists. Those rows are unrecoverable."""
    # A REAL MixtureSameFamily, because _find_nd_gmm isinstance-checks for one and
    # component_distribution is a read-only property. Random means guarantee it collides with
    # nothing on disk.
    _k, _d = 4, 3
    unsaved = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=torch.rand(_k, dtype=torch.float64)),
        torch.distributions.MultivariateNormal(
            torch.randn(_k, _d, dtype=torch.float64),
            covariance_matrix=torch.eye(_d, dtype=torch.float64).expand(_k, _d, _d)))
    fp = orchestrator._gmm_fingerprint(unsaved)
    assert fp is not None, "the probe prior must be fingerprintable, or the test proves nothing"
    assert fp not in orchestrator._saved_prior_fingerprints(), "random prior collided with a saved one"
    try:
        orchestrator._assert_prior_is_saved(unsaved, n_runs=5000, run_size=2048)
    except ValueError as e:
        assert "not saved" in str(e) and "unresumable" in str(e), f"unhelpful message: {e}"
    else:
        raise AssertionError("an unsaved prior must be refused before a long checkpointed run")

    # A prior that IS on disk must pass, or the guard blocks the very run it exists to protect.
    saved = orchestrator._saved_prior_fingerprints()
    if saved:
        import core.Helpers.file_manager as _fm
        name = sorted(saved.values())[0]
        dist = _fm.load_mix_dist(str(config.PRIOR_PATH / name), device=torch.device("cpu"))
        orchestrator._assert_prior_is_saved(dist, n_runs=5000, run_size=2048)


def test_the_batch_retry_waits_releases_and_restores_the_rng():
    """The outermost retry does not shrink the work -- it waits and runs the SAME batch again,
    because the failure the halving ladders cannot fix is a card that is momentarily full of
    somebody else's surfaces. Under WDDM those are evictable, so mem_get_info reported them as free
    and the driver lost the eviction race (trap X6); shrinking does not help, waiting does.

    The re-run must be EXACT. gen_training_data restores the batch's opening RNG snapshot first, so
    the retried batch is the batch that would have been produced. That is worth having but is NOT a
    correctness gate -- an earlier version of this docstring said it was, and that was wrong. The
    checkpoint records the ACTUAL state at every batch boundary and rows already on disk are never
    regenerated, so a re-run that skips the restore is simply a different valid draw. Which is why
    the restore is best-effort; see test_the_batch_retry_survives_a_failing_rng_restore.

    ⚠ IT MUST BE THE SNAPSHOT TAKEN IMMEDIATELY BEFORE `_rows`, NOT `_pending_rng`. `_pending_rng` is
    the state at the TOP of the iteration -- before the theta draw and the chi multipliers -- which
    is what a RESUME needs, because a resume repeats those. The retry does not: it reuses the thetas
    as drawn. Restoring `_pending_rng` would therefore hand `_rows` the noise the theta draw should
    have consumed, and the batch would differ from the one an uninterrupted run produces -- the exact
    defect the restore exists to prevent.

    This pins the structure of that loop, which a full end-to-end OOM cannot be staged for on a CPU
    box."""
    import inspect as _inspect
    src = _inspect.getsource(pipeline_mod.gen_training_data)
    for needle, why in (
        ("TRAINING_BATCH_RETRY_ATTEMPTS", "the retry must be configurable, and disable-able"),
        ("except RuntimeError as _err", "narrow, so a GUI cancel (a BaseException) still escapes"),
        ("not _is_oom(_err)", "a non-OOM RuntimeError is a bug and must not be retried in a loop"),
        ("_release_device_memory(device)", "the full release, graphs included, on the OOM path"),
        ("_cancellable_wait(", "a plain sleep cannot be cancelled -- see core/gui/streams.py"),
        ("_rows_rng = _try_rng_snapshot(", "snapshot the state _rows itself starts from"),
        ("_try_rng_restore(_tc, _rows_rng", "the re-run must resume THAT state, best-effort"),
    ):
        assert needle in src, f"batch retry: {needle!r} missing -- {why}"
    # The release must follow the notice, not precede it (the 2026-08-27 ordering bug).
    # ORDER, not adjacency: anything pinning these two as neighbours goes stale the moment
    # a step is inserted between them -- round 2 moved the RNG restore ahead of the release
    # and round 4 put the _we_are_the_holder branch before the wait. This line DID go stale,
    # and took the whole suite down with it, because str.index raises ValueError rather than
    # AssertionError and the runner below caught only the latter: the 26 tests after it never
    # ran. Both halves of that are fixed.
    assert src.index("Waiting") < src.index("_release_device_memory(device)"), \
        "the batch-retry notice must be printed BEFORE the release that may itself fail"


def test_cancellable_wait_returns_and_stays_short():
    """It sleeps in slices so the cooperative cancel -- which is raised from a stream write on the
    worker thread -- gets a chance to fire, and prints so a multi-minute pause is not read as a hang."""
    import contextlib as _ctx, time as _time
    buf = io.StringIO()
    t0 = _time.monotonic()
    with _ctx.redirect_stderr(buf):
        pipeline_mod._cancellable_wait(0.2, "unit test")
    assert 0.15 <= _time.monotonic() - t0 < 3.0, "wait did not sleep about the requested time"
    pipeline_mod._cancellable_wait(0.0, "zero")          # must not hang or raise


# ── 2026-08-28: the recovery step that fixed round 1 became the next failure point ───────────────
def test_short_err_survives_an_empty_message():
    """`str(err).splitlines()[0]` raises IndexError when the message is empty, because "".splitlines()
    is [] and not [""]. That is reachable: _is_oom returns True on the TYPE test alone, so a
    zero-message torch.OutOfMemoryError reaches the ladders' note lines -- and reached the error path
    of _release_device_memory, the one function documented never to raise."""
    assert pipeline_mod._short_err(torch.OutOfMemoryError("")) == "OutOfMemoryError"
    assert pipeline_mod._short_err(RuntimeError("boom")) == "RuntimeError: boom"
    assert pipeline_mod._short_err(RuntimeError("first\nsecond")) == "RuntimeError: first"
    assert pipeline_mod._short_err(RuntimeError("x" * 500), 10) == "RuntimeError: xxxxxxxxxx"
    # ⚠ AND IT MUST BE TOTAL, because it is called from inside the `except` clause of every guard in
    # the module -- _release_device_memory._try, _try_rng_snapshot, _try_rng_restore, _log_memory.
    # If _short_err can raise, all of them can, and the whole best-effort layer is a fiction. An
    # exception whose __str__ raises is the case that would do it.
    class _Nasty(RuntimeError):
        def __str__(self):
            raise ValueError("this exception's __str__ is broken")

    assert pipeline_mod._short_err(_Nasty()) == "_Nasty", (
        "_short_err must never raise -- it is the terminal primitive every other guard calls")

    # And the guard that uses it must survive an exception with no message at all.
    saved = torch.cuda.empty_cache
    try:
        def _blank():
            raise torch.AcceleratorError("")
        torch.cuda.empty_cache = _blank
        _cuda_release_probe(pipeline_mod._release_device_memory)   # must not raise
    finally:
        torch.cuda.empty_cache = saved


def test_log_memory_can_never_kill_a_run():
    """A DIAGNOSTIC MUST NOT BE ABLE TO END A MULTI-DAY RUN. _log_memory makes four device calls --
    mem_get_info, max_memory_allocated, max_memory_reserved, reset_peak_memory_stats -- and runs on
    the SUCCESS path every _MEM_LOG_EVERY batches, which is exactly the moment after a batch has
    fought its way through all three OOM ladders and the card is at its most degraded."""
    import contextlib as _ctx

    class _FakeDevice:
        type = "cuda"

    saved = (torch.cuda.mem_get_info, torch.cuda.max_memory_allocated,
             torch.cuda.reset_peak_memory_stats)
    buf = io.StringIO()
    try:
        def _boom(*a, **k):
            raise torch.AcceleratorError("CUDA error: out of memory")
        torch.cuda.mem_get_info = _boom
        torch.cuda.max_memory_allocated = _boom
        torch.cuda.reset_peak_memory_stats = _boom
        with _ctx.redirect_stderr(buf):
            pipeline_mod._log_memory(_FakeDevice(), "training batch 1/5000")
    finally:
        (torch.cuda.mem_get_info, torch.cuda.max_memory_allocated,
         torch.cuda.reset_peak_memory_stats) = saved
    assert "unavailable" in buf.getvalue(), (
        f"a failed memory read must degrade to a note, not an exception:\n{buf.getvalue()}")


def test_the_batch_retry_survives_a_failing_rng_restore():
    """THE 2026-08-28 REGRESSION. The retrain reached batch 3990/10000 and died inside the batch-level
    retry added the day before: `rng_restore` -> `torch.cuda.set_rng_state_all` copies each
    generator's state into DEVICE memory, so it is an allocation, and it was unguarded.

    Skipping the restore is safe -- the re-run becomes a different but equally valid iid draw, the
    same licence _rows_with_oom_retry already takes, and the checkpoint still records the ACTUAL
    state at every batch boundary. Dying is not safe. So this asserts the helper reports failure
    rather than raising, and says so."""
    import contextlib as _ctx

    class _FakeTC:
        def rng_restore(self, rng, device, chi_gen):
            raise torch.AcceleratorError("CUDA error: out of memory")

        def rng_snapshot(self, device, chi_gen):
            raise torch.AcceleratorError("CUDA error: out of memory")

    buf = io.StringIO()
    with _ctx.redirect_stderr(buf):
        ok = pipeline_mod._try_rng_restore(_FakeTC(), {"cpu": b"x"}, torch.device("cpu"), None)
    assert ok is False, "a failed restore must report failure, not raise"
    assert "fresh draw" in buf.getvalue(), f"the fallback must be announced:\n{buf.getvalue()}"

    buf2 = io.StringIO()
    with _ctx.redirect_stderr(buf2):
        snap = pipeline_mod._try_rng_snapshot(_FakeTC(), torch.device("cpu"), None)
    assert snap is None, "a failed snapshot must return None, not raise"
    assert "could not snapshot" in buf2.getvalue()

    # An empty/None rng is "nothing to restore", not a failure to shout about.
    assert pipeline_mod._try_rng_restore(_FakeTC(), None, torch.device("cpu"), None) is False


def test_the_rng_restore_happens_before_the_release_and_the_wait():
    """ORDER IS THE FIX, not the guard alone.

    The block used to run release -> wait -> restore. The release hands every cached block back to
    the driver; the wait then sleeps up to three minutes on a contended card while the desktop takes
    the memory; and the restore -- which needs only a few KB, but needs them on the device -- was
    left asking for a fresh cudaMalloc at the moment we had the weakest claim on the card.

    Restoring first serves that request from the allocator's own cache: the failed attempt's tensors
    were dropped when the except clause closed, so they are sitting there as free blocks.

    Asserted on PARSED source -- the comments in this region name all three calls, so a raw-text
    check would match the prose rather than the code (the lesson recorded for _local_map)."""
    src = _unparsed(pipeline_mod.gen_training_data)
    i_restore = src.find("_try_rng_restore(")
    i_release = src.find("_release_device_memory(device)")
    i_wait = src.find("_cancellable_wait(")
    assert i_restore != -1 and i_release != -1 and i_wait != -1, "the recovery block changed shape"
    assert i_restore < i_release, (
        "the RNG restore must happen BEFORE the release, while the allocator still holds the failed "
        "attempt's blocks -- releasing first is what killed the 2026-08-28 run")
    assert i_release < i_wait, (
        "release before the wait: sleeping while holding a cache we are not using starves the "
        "process we are waiting for")


def test_a_failed_snapshot_never_writes_a_stale_restore_point():
    """`batch_k` is bound by the `for` before the snapshot is taken, so a snapshot that fails at the
    top of batch k would leave batch k-1's state in `_pending_rng` while the rescue write records
    `batch_k = k`. A resume would then restart batch k with the streams positioned where k-1 began.

    Recording nothing is correct; recording the wrong thing is not -- but the ROWS must still be
    written either way, because they are hours of simulation and a checkpoint that resumes without
    restoring streams merely draws fresh noise from that point."""
    src = _unparsed(pipeline_mod.gen_training_data)
    assert "_pending_rng_at = batch_k" in src, (
        "the snapshot must be paired with the batch index it describes")
    assert "_rescue_rng = _pending_rng if _pending_rng_at == batch_k else None" in src, (
        "the rescue write must validate the snapshot against the batch it is committing")
    # The rows are saved regardless: the guard on the rescue write must NOT require an rng.
    assert "if _ck_dir is not None and batch_k > _ck_from:" in src, (
        "the rescue write must not be conditional on having an RNG snapshot -- that would throw "
        "away the completed batches to avoid an imperfect restore point")


def test_the_vram_ceiling_env_override_wins_and_tolerates_junk():
    """A per-run throttle has to be settable without editing a tracked file, so it follows
    PRISM_CHI_OVERRIDE's shape. A typo in an env var must not end a multi-day run."""
    import os as _os
    saved_env = _os.environ.get(pipeline_mod.VRAM_CEILING_ENV)
    saved_cfg = config.SIM_VRAM_CEILING_GIB
    try:
        _os.environ.pop(pipeline_mod.VRAM_CEILING_ENV, None)
        config.SIM_VRAM_CEILING_GIB = 3.0
        assert pipeline_mod.vram_ceiling_gib() == 3.0, "the config constant is the fallback"
        _os.environ[pipeline_mod.VRAM_CEILING_ENV] = "6.5"
        assert pipeline_mod.vram_ceiling_gib() == 6.5, "the env override must win"
        _os.environ[pipeline_mod.VRAM_CEILING_ENV] = "not-a-number"
        assert pipeline_mod.vram_ceiling_gib() == 3.0, "junk must fall back, not raise"
        _os.environ[pipeline_mod.VRAM_CEILING_ENV] = ""
        assert pipeline_mod.vram_ceiling_gib() == 3.0, "empty means unset"
    finally:
        _os.environ.pop(pipeline_mod.VRAM_CEILING_ENV, None)
        if saved_env is not None:
            _os.environ[pipeline_mod.VRAM_CEILING_ENV] = saved_env
        config.SIM_VRAM_CEILING_GIB = saved_cfg


def test_the_planner_budget_survives_an_unreadable_card():
    """Every OOM retry re-enters _max_sim_batch to re-plan, so config.memory_budget_elements' driver
    reads run on a card that is already refusing service. pipeline._free_gib_note guards the
    identical mem_get_info call; this one was bare. Falling back to a small budget is the safe
    direction -- it makes the planner split more, which costs wall-clock and cannot lose data."""
    class _FakeDevice:
        type = "cuda"

    saved = torch.cuda.mem_get_info
    try:
        def _boom(*a, **k):
            raise torch.AcceleratorError("CUDA error: out of memory")
        torch.cuda.mem_get_info = _boom
        got = config.memory_budget_elements(_FakeDevice(), torch.float32)
    finally:
        torch.cuda.mem_get_info = saved
    assert got == (1 * 1024 ** 3) // 4, f"expected the conservative fallback budget, got {got}"


def _gmm_from(means, weights):
    """A real MixtureSameFamily over the given means/weights -- what _gmm_fingerprint digests."""
    k, d = means.shape
    return torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=weights),
        torch.distributions.MultivariateNormal(
            means, covariance_matrix=torch.eye(d, dtype=means.dtype).expand(k, d, d)))


def test_saving_a_prior_cannot_orphan_a_checkpoint():
    """⚠ THIS IS HOW 3989 BATCHES (6.5 h) WERE LOST ON 2026-08-28.

    A checkpoint's directory is named after a digest of the prior's fitted GMM, so the prior FILE is
    the only thing that can reproduce it. `prior_08282026.pt` was overwritten, under the same name,
    with a different distribution -- and the run that had been training against the old contents all
    morning became unreachable. No error, no warning, one click. The same mechanism cost 884 batches
    the day before, and `3d_master_08102026.pt` currently backs THREE 5000-batch checkpoints.

    Narrow by construction: it fires only when the file exists, its contents would actually change,
    AND a committed checkpoint depends on the old contents."""
    import tempfile, hashlib
    from pathlib import Path as _P
    from core.SBI import training_checkpoint as tc

    def fp_of(means, weights):
        h = hashlib.sha256()
        h.update(means.detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
        h.update(weights.detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
        return h.hexdigest()[:16]

    torch.manual_seed(0)
    # NORMALISED, because torch.distributions.Categorical normalises `probs` on construction: a file
    # holding raw weights would fingerprint differently from the distribution rebuilt out of it, and
    # the test would then "pass" by accident on a mismatch that production never sees (save_mix_dist
    # writes `mixture_distribution.probs`, which is already normalised).
    def _w(n):
        w = torch.rand(n, dtype=torch.float64)
        return w / w.sum()

    m_old, w_old = torch.randn(4, 3, dtype=torch.float64), _w(4)
    m_new, w_new = torch.randn(4, 3, dtype=torch.float64), _w(4)

    with tempfile.TemporaryDirectory() as tmp:
        priors, ckpts = _P(tmp) / "Priors", _P(tmp) / "Checkpoints"
        priors.mkdir(); ckpts.mkdir()
        # Build the distribution FIRST and save what it exposes, exactly as save_mix_dist does.
        # Constructing a Categorical normalises `probs` again, so saving the raw weights would make
        # the file and the rebuilt distribution differ in the last bits -- a mismatch production
        # never has, which would make this test assert the wrong thing.
        gmm_old = _gmm_from(m_old, w_old)
        f_means = gmm_old.component_distribution.loc.detach().clone()
        f_weights = gmm_old.mixture_distribution.probs.detach().clone()
        torch.save({"means": f_means, "weights": f_weights}, priors / "p.pt")

        ident = {"format": "training-rows", "n_runs": 10000,
                 "prior_fingerprint": fp_of(f_means, f_weights)}
        d = tc.resolve_dir(ident, ckpts); (d / "shards").mkdir(parents=True)
        torch.save({"identity": ident}, d / "header.pt")
        torch.save({"batches_done": 3989, "complete": False, "rng": None}, d / "state.pt")

        saved_pp, saved_cp = orchestrator.PRIOR_PATH, config.CHECKPOINT_PATH
        try:
            orchestrator.PRIOR_PATH = priors
            config.CHECKPOINT_PATH = ckpts

            try:
                orchestrator._refuse_to_orphan_a_checkpoint("p", _gmm_from(m_new, w_new))
            except ValueError as e:
                assert "3,989" in str(e), f"the message must name what would be lost: {e}"
                assert "UNRESUMABLE" in str(e).upper(), f"and why it matters: {e}"
            else:
                raise AssertionError(
                    "overwriting a prior that a 3989-batch checkpoint depends on must be refused")

            # Re-saving the SAME distribution changes nothing, so it must go through.
            orchestrator._refuse_to_orphan_a_checkpoint("p", gmm_old)
            # A name nothing depends on must go through.
            orchestrator._refuse_to_orphan_a_checkpoint("something_else", _gmm_from(m_new, w_new))
            # And a prior no COMMITTED checkpoint uses must go through.
            torch.save({"means": m_new, "weights": w_new}, priors / "unused.pt")
            orchestrator._refuse_to_orphan_a_checkpoint("unused", gmm_old)
        finally:
            orchestrator.PRIOR_PATH, config.CHECKPOINT_PATH = saved_pp, saved_cp


def test_the_retry_does_not_wait_when_THIS_process_holds_the_card():
    """WAITING ONLY HELPS IF SOMEBODY ELSE HOLDS THE MEMORY.

    The batch retry was written for the documented failure -- a card momentarily full of the
    desktop's evictable surfaces -- where pausing is exactly right. It is exactly wrong when we are
    the holder. On 2026-08-28 a run sat at batch 93 repeating "waiting for device memory" while
    holding 15310 MB of a 16303 MB card, every other process on the machine accounting for ~270 MB
    combined. It was waiting for itself, and no delay could ever have satisfied it."""
    class _Dev:
        type = "cuda"

    saved = (torch.cuda.mem_get_info, torch.cuda.memory_reserved)
    try:
        total = 16303 * 2 ** 20
        # The real numbers from the stuck run: we hold 15310 MB, everyone else ~270 MB.
        torch.cuda.mem_get_info = lambda *a, **k: (373 * 2 ** 20, total)
        torch.cuda.memory_reserved = lambda *a, **k: 15310 * 2 ** 20
        assert pipeline_mod._we_are_the_holder(_Dev()) is True, (
            "holding 15310 of 16303 MB must be recognised as us, so the retry stops waiting")

        # And the case the wait WAS designed for: the desktop holds it, we hold almost nothing.
        torch.cuda.memory_reserved = lambda *a, **k: 200 * 2 ** 20
        torch.cuda.mem_get_info = lambda *a, **k: (300 * 2 ** 20, total)
        assert pipeline_mod._we_are_the_holder(_Dev()) is False, (
            "when another process holds the card, waiting is the right response and must happen")

        # Unknowable => fall back to waiting rather than skipping it.
        def _boom(*a, **k):
            raise torch.AcceleratorError("CUDA error: out of memory")
        torch.cuda.mem_get_info = _boom
        assert pipeline_mod._we_are_the_holder(_Dev()) is False
    finally:
        torch.cuda.mem_get_info, torch.cuda.memory_reserved = saved

    # ...and the retry must actually consult it rather than always sleeping.
    src = _unparsed(pipeline_mod.gen_training_data)
    assert "_we_are_the_holder(device)" in src, "the retry must ask whose memory it is"
    i_hold = src.find("_we_are_the_holder(device)")
    i_wait = src.find("_cancellable_wait(_delay")
    assert i_hold < i_wait, "the check must gate the wait, not follow it"


def test_the_mem_line_is_printed_on_every_oom_and_its_cadence_is_overridable():
    """Peak RESERVED against peak ALLOCATED is what separates 'this geometry is too big' from 'the
    allocator is fragmented and cannot hand the memory back'. A line every 250 batches told us
    nothing about a run that died at batch 93."""
    import os as _os
    src = _unparsed(pipeline_mod.gen_training_data)
    assert "_log_memory(device, f'after OOM on " in src or "after OOM on" in src, (
        "every OOM must emit a [mem] line -- that is the moment the numbers matter")
    assert isinstance(pipeline_mod._MEM_LOG_EVERY, int) and pipeline_mod._MEM_LOG_EVERY >= 1
    mod_src = _strip_docstrings(ast.parse(io.open(pipeline_mod.__file__, encoding="utf-8").read()))
    assert "PRISM_MEM_LOG_EVERY" in ast.unparse(mod_src), (
        "the cadence must be overridable for a diagnostic run without editing a tracked file")


if __name__ == "__main__":
    failures = 0
    for test_name, fn in sorted(globals().items()):
        if test_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {test_name}")
            # Exception, NOT AssertionError. A test that raises anything else -- a ValueError
            # from a stale str.index, a CUDA error from a hostile card -- used to abort the
            # ENTIRE run at that point, silently losing every test after it. That cost 26
            # tests twice on 2026-08-28. A crash is a failure of THAT test, not of the suite.
            except Exception as e:
                failures += 1
                print(f"FAIL  {test_name}\n      {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
