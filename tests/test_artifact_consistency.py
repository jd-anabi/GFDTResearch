"""Artifact-identity tests: the master Bounds/Cells triple, and the guards that stop a prior or a
posterior being used against a configuration it does not belong to.

WHAT THESE LOCK DOWN
    A posterior is only meaningful against the exact (model, parameter set + ORDER, box) it was
    trained in -- the flow learns a density over the LATENT coordinate, so the box is what turns its
    output back into physical values. None of that used to be checked. `build_prior` validated
    NOTHING on its load path, `build_posterior` checked only the log-mask, `_assert_mode_matches`
    checked only the observation mode and the conditioning WIDTH, and `load_eval_bijection` rebuilt
    the box from whatever config happened to be loaded rather than from the posterior.

    The cost of that was paid once already: a chi posterior trained on one cell's bounds could be
    loaded, validated and inferred against another's, with every reported parameter silently decoded
    through the wrong box edges, and the only way to establish what had actually happened was to
    compare GMM component counts against the prior files on disk after the fact.

    Also pinned here: the master Bounds/Cells triple that replaced the five per-cell boxes. Its whole
    point is that ONE box serves every cell, so "the two bounds files agree" and "every cell sits
    strictly inside" are invariants, not incidental facts.

Run:  python tests/test_artifact_consistency.py
"""
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core import cli, config, orchestrator, registry
from core.config import BOUNDS_PATH, CELL_PATH, POSTERIOR_PATH, PRIOR_PATH, VALID_LABELS, VALID_MODELS
from core.Helpers import file_manager
from core.SBI import reparam
from core.SBI.statistics import FEATURE_LABELS, SUMMARY_WIDTH

_NAD = "nadrowski"
_LABELS = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
_MASTER_CELLS = ("master_spont", "master_weak", "master_entrained")


def _cfg(bounds="master.txt", **kw):
    cfg = cli.make_sim_config("NADROWSKI", _LABELS, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / _NAD / bounds), **kw)
    cfg.hw = config.cpu_device()
    return cfg


# ── the master Bounds/Cells triple ────────────────────────────────────────────────────────────────
def test_master_bounds_pair_share_one_nd_section():
    """master_spont.txt exists ONLY to drop f_scale and the Forcing section (mode 1 drops f_scale
    because it only ever divides a force). Its ND block must be byte-for-byte the same box, or the
    two modes are quietly inferring over different parameter spaces."""
    forced = file_manager.parse_bounds_file(str(BOUNDS_PATH / _NAD / "master.txt"))
    spont = file_manager.parse_bounds_file(str(BOUNDS_PATH / _NAD / "master_spont.txt"))
    assert list(forced[0].items()) == list(spont[0].items()), "ND sections differ"
    assert list(forced[1]) == ["x_scale", "t_scale", "f_scale"], list(forced[1])
    assert list(spont[1]) == ["x_scale", "t_scale"], list(spont[1])
    assert spont[2] == {} and forced[2], "only the forced file may declare a Forcing section"


def test_every_master_cell_is_strictly_interior_to_the_master_box():
    """A ground truth ON a bound is the hazard the archived cell_2 shipped with (tau_c = 0, temp = 0
    sat exactly on their lower edges), which is why those marginals could only ever be one-sided.
    Strictly interior, with no exceptions, is the property the master cells were built to have."""
    for cell in _MASTER_CELLS:
        cfg = _cfg()
        cli.load_and_validate_gt(cfg, str(CELL_PATH / _NAD / f"{cell}.txt"))
        for name, (val, (lo, hi)) in list(cfg.params_dict.items()) + list(cfg.rescale_params.items()):
            assert lo < val < hi, f"{cell}: {name}={val} is not strictly inside ({lo}, {hi})"


def test_every_master_cell_injects_under_every_mode():
    """One cell set, three observation modes. The spontaneous cell deliberately keeps f_scale and a
    zeroed Forcing section so chi mode -- which needs f_scale inferred -- can use it too; extras are
    ignored, never fatal."""
    seen = set()
    for bounds, chi in (("master.txt", False), ("master.txt", True), ("master_spont.txt", False)):
        for cell in _MASTER_CELLS:
            cfg = _cfg(bounds, chi_mode=chi, chi_n_freqs=4)
            problems = cli.validate_gt_file(cfg, str(CELL_PATH / _NAD / f"{cell}.txt"))
            assert not problems, f"{bounds} + {cell}: {problems}"
            cli.load_and_validate_gt(cfg, str(CELL_PATH / _NAD / f"{cell}.txt"))
            seen.add(cfg.observation_mode)
    assert seen == {"spontaneous", "forced", "chi"}, seen


def test_chi_probe_band_is_sub_resonance():
    """CHI_FREQ_BOUNDS was retargeted to the band where |chi| is actually reproducible: measured on
    the master cell, everything above ~0.25x Omega_0 has CV 0.2-0.7 at EVERY drive amplitude, and
    does not improve from T_obs 5 s to 25 s (so it is systematic, not statistical). The old
    (0.1, 10.0) put 8 of 10 probes at K=10 in that regime. This pins the band against a well-meaning
    revert to 'cover more frequency'."""
    lo, hi = config.CHI_FREQ_BOUNDS
    assert 0 < lo < hi <= 0.3, f"chi probes must stay sub-resonance, got {config.CHI_FREQ_BOUNDS}"
    assert 0 < config.CHI_F0 < 0.2, (
        f"CHI_F0={config.CHI_F0}: 0.2 is the measured entrainment onset -- at or above it the drive "
        f"captures the bundle and chi reports the drive back to itself")


# ── bounds resolution ─────────────────────────────────────────────────────────────────────────────
def test_bounds_resolution_prefers_a_sibling_then_falls_back_to_master():
    """Three cells now share one bounds file, which the same-named-sibling rule cannot express. The
    sibling still WINS where it exists, so every pre-existing cell resolves exactly as before."""
    assert cli.resolve_bounds_for_cell(str(CELL_PATH / _NAD / "master_spont.txt")).name == "master_spont.txt"
    for cell in ("master_weak", "master_entrained"):
        got = cli.resolve_bounds_for_cell(str(CELL_PATH / _NAD / f"{cell}.txt"))
        assert got.name == cli.MASTER_BOUNDS_NAME, f"{cell} -> {got}"
    # a cell that exists in neither place still resolves, because master.txt governs the folder
    assert cli.resolve_bounds_for_cell(str(CELL_PATH / _NAD / "no_such_cell.txt")).name == "master.txt"
    # ...but a model folder with no master.txt and no sibling resolves to nothing, not to a guess
    assert cli.resolve_bounds_for_cell(str(CELL_PATH / "hopf" / "no_such_cell.txt")) is None


# ── prior identity ────────────────────────────────────────────────────────────────────────────────
def _write_prior(path, lows, highs, keys, model="NADROWSKI"):
    d = len(lows)
    base = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=torch.ones(2)),
        torch.distributions.MultivariateNormal(torch.zeros(2, d),
                                               covariance_matrix=torch.eye(d).expand(2, d, d)))
    T = reparam.build_box_bijection(torch.tensor(lows), torch.tensor(highs),
                                    torch.zeros(d, dtype=torch.bool))
    file_manager.save_mix_dist(torch.distributions.TransformedDistribution(base, T), str(path),
                               model=model, param_keys=keys)


def test_a_prior_from_another_config_is_refused():
    """build_prior's load path used to validate NOTHING. The GMM is fit in its box's own coordinate,
    so a prior from a different box trains the flow against a different distribution than the one its
    samples came from -- silently, because the means are latent and cannot be eyeballed."""
    cfg = _cfg()
    keys = list(cfg.params_dict)
    lo = [b[0] for _, b in cfg.params_dict.values()]
    hi = [b[1] for _, b in cfg.params_dict.values()]
    cases = {
        "ok": (lo, hi, keys, "NADROWSKI"),
        "box": (lo, [hi[0] * 2] + hi[1:], keys, "NADROWSKI"),
        "order": (lo, hi, keys[1:] + keys[:1], "NADROWSKI"),
        "model": (lo, hi, keys, "HOPF"),
    }
    for tag, (l, h, k, m) in cases.items():
        path = PRIOR_PATH / f"_ptest_{tag}.pt"
        try:
            _write_prior(path, l, h, k, m)
            if tag == "ok":
                orchestrator._assert_prior_matches(cfg, str(path), path.name)   # must not raise
            else:
                try:
                    orchestrator._assert_prior_matches(cfg, str(path), path.name)
                    raise AssertionError(f"a prior with the wrong {tag} was accepted")
                except ValueError:
                    pass
        finally:
            path.unlink(missing_ok=True)


def test_a_prior_the_posterior_was_not_trained_with_is_refused():
    """SBC draws theta* from the TRAINING prior. Run against a different one it is not a calibration
    measurement of that posterior at all. The posterior carries its own training prior, so this needs
    no sidecar and works for a posterior trained moments ago and never saved."""
    d = 4
    def gmm(seed):
        torch.manual_seed(seed)
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(probs=torch.rand(3)),
            torch.distributions.MultivariateNormal(torch.randn(3, d),
                                                   covariance_matrix=torch.eye(d).expand(3, d, d)))
    g1, g2 = gmm(1), gmm(2)
    T = reparam.build_box_bijection(torch.zeros(d), torch.ones(d), torch.zeros(d, dtype=torch.bool))
    phys1 = torch.distributions.TransformedDistribution(g1, T)
    phys2 = torch.distributions.TransformedDistribution(g2, T)

    class _Product:                      # stands in for ProductPrior
        def __init__(self, ds): self.distributions = ds

    class _Wrapper:                      # stands in for SBIPriorWrapper
        def __init__(self, g): self.gen_dist = g

    class _Post:
        def __init__(self, p): self.prior = p

    post = _Post(_Wrapper(_Product([g1])))
    # the fingerprint must see the SAME gmm through every wrapper shape it can arrive in
    assert len({orchestrator._gmm_fingerprint(x)
                for x in (g1, phys1, _Product([phys1]), _Wrapper(_Product([g1])))}) == 1
    orchestrator._assert_prior_used_matches_posterior(post, _Product([phys1]), "t")   # must not raise
    try:
        orchestrator._assert_prior_used_matches_posterior(post, _Product([phys2]), "t")
        raise AssertionError("a foreign prior was accepted")
    except ValueError:
        pass
    # unverifiable on either side => silence, not a false alarm (legacy artifacts land here)
    orchestrator._assert_prior_used_matches_posterior(_Post(None), _Product([phys1]), "t")


# ── posterior identity ────────────────────────────────────────────────────────────────────────────
def _sidecar(cfg, **over):
    """A sidecar that a CURRENT build would actually write, so an override tests what it names.

    Every field the chi block of ``_assert_mode_matches`` gates on has to be here and has to be
    RIGHT. It is checked in order and the first mismatch raises, so one missing key makes every test
    built on this helper pass on that key instead of on its own override -- which is what happened:
    ``chi_layout``/``chi_k_pad``/``chi_elem_w`` were absent, so the baseline sidecar was rejected as
    "trained under chi layout 1" and the model / param-order cases below were never reached.
    """
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    d = dict(
        V=None, log_params=[], mode="chi", chi_n_freqs=4,
        # DERIVED, never literals. These were `input_dim=42, forcing_dim=12` -- 12 being 3K at K=4
        # under the retired layout-1 grid, stale since the probe set landed and silently wrong ever
        # since. Deriving them from the same helpers the writer uses is what keeps a fixture honest.
        input_dim=SUMMARY_WIDTH + 1, forcing_dim=orchestrator.expected_forcing_dim(cfg),
        chi_layout=config.CHI_LAYOUT, chi_k_pad=cfg.chi_k_pad, chi_elem_w=config.CHI_ELEM_W,
        chi_max_cycles=float(cfg.chi_max_cycles),
        chi_f0=cfg.chi_f0, chi_freq_bounds=tuple(cfg.chi_freq_bounds), param_keys=keys,
        model="NADROWSKI",
        nd_lows=torch.tensor([b[0] for _, b in cfg.params_dict.values()], dtype=torch.float64),
        nd_highs=torch.tensor([b[1] for _, b in cfg.params_dict.values()], dtype=torch.float64),
        rescale_lows=torch.tensor([b[0] for _, b in cfg.rescale_params.values()], dtype=torch.float64),
        rescale_highs=torch.tensor([b[1] for _, b in cfg.rescale_params.values()], dtype=torch.float64),
    )
    d.update(over)
    return d


def test_eval_box_comes_from_the_posterior_not_the_config():
    """THE fix. load_eval_bijection's docstring always claimed eval was self-describing, but the box
    was rebuilt from cfg -- so a posterior trained against one bounds file and evaluated against
    another decoded every latent sample through the wrong edges, changing the physical value of every
    reported parameter with nothing raised."""
    cfg = _cfg(chi_mode=True, chi_n_freqs=4)
    path = POSTERIOR_PATH / "_ptest.rot.pt"
    n = len(cfg.params_dict) + len(cfg.rescale_params)
    try:
        torch.save(_sidecar(cfg), str(path))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            T_match = reparam.load_eval_bijection(cfg, "_ptest.pt", POSTERIOR_PATH)
        assert not w, "a matching box must not warn"

        # same posterior, config box widened: the SIDECAR must win, and it must say so
        torch.save(_sidecar(cfg, nd_highs=_sidecar(cfg)["nd_highs"] * 2), str(path))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            T_other = reparam.load_eval_bijection(cfg, "_ptest.pt", POSTERIOR_PATH)
        assert any("DIFFERENT box" in str(x.message) for x in w), [str(x.message) for x in w]
        z = torch.zeros(1, n)
        assert not torch.allclose(T_match(z), T_other(z)), \
            "the two boxes decode the same latent identically -- the sidecar box was ignored"
    finally:
        path.unlink(missing_ok=True)


def test_a_posterior_over_different_parameters_is_refused():
    """Mode + conditioning width agreeing says only that the vectors are the same SHAPE, which many
    configs satisfy. Columns bind positionally, so a reordered parameter set makes every reported
    value refer to the wrong parameter."""
    cfg = _cfg(chi_mode=True, chi_n_freqs=4)
    path = POSTERIOR_PATH / "_ptest.rot.pt"
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    try:
        # THE BASELINE, and the reason this test means anything: an unmodified sidecar must be
        # ACCEPTED. Without it every case below can pass on some unrelated field being wrong.
        torch.save(_sidecar(cfg), str(path))
        orchestrator._assert_mode_matches(cfg, object(), "_ptest.pt")
        for tag, over in (("model", {"model": "HOPF"}),
                          ("param order", {"param_keys": keys[1:] + keys[:1]})):
            torch.save(_sidecar(cfg, **over), str(path))
            try:
                orchestrator._assert_mode_matches(cfg, object(), "_ptest.pt")
                raise AssertionError(f"a posterior with the wrong {tag} was accepted")
            except ValueError:
                pass
    finally:
        path.unlink(missing_ok=True)


def test_a_posterior_trained_at_a_different_cycle_ceiling_is_refused():
    """chi_max_cycles is frozen into an artifact for the same reason the band is.

    It decides how much of each recording is integrated, so the SAME bench data yields a different
    |chi| and -- the part that bites -- a different ``logcyc`` under two ceilings. logcyc is the
    channel the encoder uses to decide how much to trust a probe, so evaluating at a foreign ceiling
    feeds it a value the training set never contained, on precisely the channel whose job is
    calibration. Nothing about the shapes disagrees, so without this check it loads clean.

    An ABSENT ceiling must stay silent: posteriors written before 2026-08-06 have no such field, and
    turning those into a hard failure would strand every existing artifact.
    """
    cfg = _cfg(chi_mode=True, chi_n_freqs=4)
    path = POSTERIOR_PATH / "_ptest.rot.pt"
    try:
        torch.save(_sidecar(cfg, chi_max_cycles=float(cfg.chi_max_cycles) * 2), str(path))
        try:
            orchestrator._assert_mode_matches(cfg, object(), "_ptest.pt")
            raise AssertionError("a posterior trained at a different cycle ceiling was accepted")
        except ValueError as e:
            assert "ceiling" in str(e), f"the message must name the ceiling, got: {e}"

        d = _sidecar(cfg)
        d.pop("chi_max_cycles")
        torch.save(d, str(path))
        orchestrator._assert_mode_matches(cfg, object(), "_ptest.pt")   # legacy: must not raise
    finally:
        path.unlink(missing_ok=True)


# ── the end-of-run artifact writes are atomic ─────────────────────────────────────────────────────
class _WriteFailed(RuntimeError):
    """Injected mid-write failure. Not OSError, so a handler that swallows disk errors cannot hide it."""


def _failing(real):
    """Wrap a serializer so it writes its bytes and THEN fails -- the tear that atomicity must absorb.

    Failing before writing anything would pass against a plain `torch.save` too: the destination is
    only clobbered once the writer has begun. The bytes have to land first for the test to mean
    anything.

    Signature-agnostic (`*a, **k`) because the two serialisers order their arguments differently --
    ``torch.save(obj, file)`` against ``np.savez(file, **arrays)``.
    """
    def _boom(*a, **k):
        real(*a, **k)
        raise _WriteFailed("disk full")
    return _boom



def test_the_sidecar_records_the_fisher_eigenvalues_beside_V():
    """The rotation is saved so a later reader can say WHICH directions the run constrained. Without
    the eigenvalues it can only say which is worst, never by how much -- and the gap between "3x" and
    "1e6" is the gap between "uneven" and "nine of thirteen parameters are prior".

    The None case is asserted too, and deliberately: the key must be PRESENT-and-None when the
    rotation came from a resumed checkpoint (which stores V but not them), so a reader can tell "not
    recorded" from "this artifact predates the field".
    """
    cfg = _cfg(chi_mode=True, chi_n_freqs=4)
    name = "_ptest_evals"
    pt, rot = POSTERIOR_PATH / f"{name}.pt", POSTERIOR_PATH / f"{name}.rot.pt"
    evals = torch.tensor([9.0, 3.0, 0.5], dtype=torch.float64)
    try:
        orchestrator.save_posterior_artifacts(name, {"generation": 1}, torch.eye(3), None, cfg,
                                              fisher_eigenvalues=evals)
        d = torch.load(str(rot), weights_only=False)
        assert "fisher_eigenvalues" in d, "the sidecar dropped the eigenvalues"
        assert torch.allclose(d["fisher_eigenvalues"], evals)

        orchestrator.save_posterior_artifacts(name, {"generation": 2}, None, None, cfg)
        d2 = torch.load(str(rot), weights_only=False)
        assert "fisher_eigenvalues" in d2 and d2["fisher_eigenvalues"] is None, \
            "an un-rotated run must record the field as None, not omit it"
    finally:
        for q in (pt, rot):
            q.unlink(missing_ok=True)

def test_a_torn_posterior_write_leaves_the_previous_artifact_intact():
    """The posterior and its .rot.pt sidecar are the product of a multi-day run, and the GUI's Save
    button can be pressed a second time over the same name. A bare torch.save truncates the
    destination before it writes, so a failure there leaves a file that exists, has a plausible size,
    and cannot be unpickled -- discovered whenever someone next tries to load it.

    Routed through save_posterior_artifacts rather than the helper directly, because what regresses is
    not the helper (it has its own test) but a call site quietly reverting to torch.save."""
    cfg = _cfg(chi_mode=True, chi_n_freqs=4)
    name = "_ptest_atomic"
    pt, rot = POSTERIOR_PATH / f"{name}.pt", POSTERIOR_PATH / f"{name}.rot.pt"
    real_save = torch.save
    try:
        orchestrator.save_posterior_artifacts(name, {"generation": 1}, None, None, cfg)
        assert torch.load(str(pt), weights_only=False)["generation"] == 1
        assert torch.load(str(rot), weights_only=False)["mode"] == "chi", "no sidecar was written"

        torch.save = _failing(real_save)
        try:
            orchestrator.save_posterior_artifacts(name, {"generation": 2}, None, None, cfg)
            raise AssertionError("the injected failure did not propagate")
        except _WriteFailed:
            pass
        finally:
            torch.save = real_save

        assert torch.load(str(pt), weights_only=False)["generation"] == 1, \
            "a torn write clobbered the posterior it was replacing"
        assert not (POSTERIOR_PATH / f"{name}.pt.tmp").exists(), "a failed write left its temp behind"
    finally:
        torch.save = real_save
        for p in (pt, rot, POSTERIOR_PATH / f"{name}.pt.tmp", POSTERIOR_PATH / f"{name}.rot.pt.tmp"):
            p.unlink(missing_ok=True)


def test_a_torn_prior_write_leaves_the_previous_prior_intact():
    """A prior is not just a file: it is what the training checkpoint's identity fingerprints and what
    SBC draws theta* from. Half-replacing one does not produce a broken run, it produces a run that
    resumes against a distribution nobody can name (2026-08-12: prior_fingerprint is in the checkpoint
    identity for exactly this reason)."""
    path = PRIOR_PATH / "_ptest_atomic.pt"
    real_save = torch.save
    try:
        _write_prior(path, [0.0, 0.0], [1.0, 1.0], ["a", "b"])
        first = file_manager.read_prior_metadata(str(path))["param_keys"]
        assert first == ["a", "b"], first

        torch.save = _failing(real_save)
        try:
            _write_prior(path, [0.0, 0.0], [1.0, 1.0], ["c", "d"])
            raise AssertionError("the injected failure did not propagate")
        except _WriteFailed:
            pass
        finally:
            torch.save = real_save

        assert file_manager.read_prior_metadata(str(path))["param_keys"] == ["a", "b"], \
            "a torn write clobbered the prior it was replacing"
        assert not (PRIOR_PATH / "_ptest_atomic.pt.tmp").exists(), "a failed write left its temp behind"
    finally:
        torch.save = real_save
        path.unlink(missing_ok=True)
        (PRIOR_PATH / "_ptest_atomic.pt.tmp").unlink(missing_ok=True)


def test_atomic_savez_round_trips_and_cannot_be_torn():
    """The .loss.npz is a zip, so a truncated one raises BadZipFile rather than reading short -- and it
    is the file scripts/retrain_convergence.py reads back for its convergence verdict.

    The round-trip half is load-bearing on its own: np.savez appends '.npz' when handed a NAME but not
    when handed a HANDLE, which is the difference between landing on <name>.loss.npz and on
    <name>.loss.npz.tmp.npz."""
    import numpy as np
    path = PRIOR_PATH / "_ptest_atomic.npz"
    real_savez = np.savez
    try:
        file_manager.atomic_savez(path, dict(validation_loss=np.arange(3.0), epochs_trained=7))
        assert path.exists(), f"nothing landed at {path} -- np.savez rewrote the name"
        with np.load(str(path)) as z:
            assert list(z["validation_loss"]) == [0.0, 1.0, 2.0] and int(z["epochs_trained"]) == 7

        np.savez = _failing(real_savez)
        try:
            file_manager.atomic_savez(path, dict(validation_loss=np.arange(99.0), epochs_trained=99))
            raise AssertionError("the injected failure did not propagate")
        except _WriteFailed:
            pass
        finally:
            np.savez = real_savez

        with np.load(str(path)) as z:
            assert int(z["epochs_trained"]) == 7, "a torn write clobbered the previous curve"
        assert not (PRIOR_PATH / "_ptest_atomic.npz.tmp").exists(), "a failed write left its temp behind"
    finally:
        np.savez = real_savez
        path.unlink(missing_ok=True)
        (PRIOR_PATH / "_ptest_atomic.npz.tmp").unlink(missing_ok=True)


# ── the chi band/drive preflight (2026-08-19 regression) ─────────────────────────────────────────
def test_a_chi_run_at_a_non_default_band_is_refused_before_the_simulation_spend():
    """A ~5-day retrain was spent at the RETIRED band (0.1, 10.0) because QSettings restored a value
    saved before C-5 changed it. `_assert_mode_matches` catches that disagreement only when a
    posterior is LOADED, i.e. after the days are gone.

    The subtle half is the LOAD path: it compares the posterior against cfg, so a stale cfg loading
    the posterior trained under that same stale cfg agrees with itself and stays silent. This guard
    compares against config.py, the one party that cannot go stale.

    Scope matters as much as existence: chi_n_freqs must NOT be an error. It is the count an
    OBSERVATION supplies, training draws its own K per batch, and failing on it would refuse a
    perfectly good 7-recording experiment.
    """
    import os as _os
    cfg = _cfg(chi_mode=True, chi_n_freqs=4)

    orchestrator._assert_chi_config_is_deliberate(cfg)          # at the defaults: must not raise

    for field, bad in (("chi_freq_bounds", (0.1, 10.0)), ("chi_f0", cfg.chi_f0 * 2)):
        stale = _cfg(chi_mode=True, chi_n_freqs=4)
        setattr(stale, field, bad)
        try:
            orchestrator._assert_chi_config_is_deliberate(stale)
            raise AssertionError(f"a chi run with a non-default {field} was accepted")
        except ValueError as e:
            assert field in str(e), f"the message must name {field}, got: {e}"
            assert "PRISM.ini" in str(e) or "QSettings" in str(e), \
                "the message must point at the persisted-settings cause, which is what bit"

    # K alone is legitimate -- one posterior serves any probe count (build_posterior omits it from
    # training_params on purpose). Refusing it would break the K-agnosticism the set encoder buys.
    k_only = _cfg(chi_mode=True, chi_n_freqs=4)
    k_only.chi_n_freqs = int(config.CHI_N_FREQS) + 3
    orchestrator._assert_chi_config_is_deliberate(k_only)       # must not raise

    # A non-chi run is never gated by a chi knob.
    spont = _cfg("master_spont.txt")
    spont.chi_freq_bounds = (0.1, 10.0)
    orchestrator._assert_chi_config_is_deliberate(spont)        # must not raise

    # The escape hatch works, and is explicit -- a band sweep is a real activity.
    prev = _os.environ.get(orchestrator.CHI_OVERRIDE_ENV)
    _os.environ[orchestrator.CHI_OVERRIDE_ENV] = "1"
    try:
        override = _cfg(chi_mode=True, chi_n_freqs=4)
        override.chi_freq_bounds = (0.1, 10.0)
        orchestrator._assert_chi_config_is_deliberate(override)  # must not raise
    finally:
        if prev is None:
            _os.environ.pop(orchestrator.CHI_OVERRIDE_ENV, None)
        else:
            _os.environ[orchestrator.CHI_OVERRIDE_ENV] = prev


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
