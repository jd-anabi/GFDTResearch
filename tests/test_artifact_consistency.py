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
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    d = dict(
        V=None, log_params=[], mode="chi", input_dim=42, forcing_dim=12, chi_n_freqs=4,
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
