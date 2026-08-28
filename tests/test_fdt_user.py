"""FDT-for-user-models unit tests (FEATURE 1 v3 + B-d), Qt-free.

Locks down the generalized effective-temperature normalization and its gate:
  * campaigns.observable_noise_prefactor computes the per-model coupling/D_x -- n*beta for NADROWSKI,
    2/sigma_x^2 for HOPF, 2*tau_hb/eta_hb^2 for BP (experimental), 1/D_0 for an additive-noise user
    model -- and raises FDTModelError for a user model whose observable noise is multiplicative, zero,
    or negative.
  * registry.fdt_support admits the built-ins and additive-noise user models; it rejects user models
    with multiplicative / zero observable noise or intrinsic forcing.
  * campaigns._n_force_channels returns one channel per state variable for a user model.

No cell files / QApplication needed: a tiny fake cfg supplies only .model / .params_dict (and, for the
force-channel test, .inits_tensor / .force_params_dict), which is all these functions read.

Run:  python tests/test_fdt_user.py
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                       # noqa: E402

from core import registry                                         # noqa: E402
from core.FDT.campaigns import (observable_noise_prefactor, _n_force_channels,   # noqa: E402
                                FDTModelError)
from core.Models.user_model import parse_user_model               # noqa: E402


class _FakeCfg:
    """The subset of FDTConfig the FDT model helpers read."""
    def __init__(self, model, params=None, n_vars=None, force_params=None):
        self.model = model
        self.params_dict = {k: (v, None) for k, v in (params or {}).items()}   # name -> (value, bounds)
        if n_vars is not None:
            self.inits_tensor = torch.zeros(1, n_vars)
        self.force_params_dict = force_params or {}


def _register_user(name, variables):
    compiled = parse_user_model(variables)
    registry.register(registry.ModelSpec(
        name=name, labels=list(compiled.param_names), state_dep_drift=compiled.state_dep_noise,
        is_user_model=True, n_vars=len(compiled.var_names), variables=variables, compiled=compiled))
    return compiled


def test_observable_noise_prefactor_builtins():
    """The built-in prefactors reduce to the coded per-model formulas (Nadrowski = the historical value)."""
    assert observable_noise_prefactor(_FakeCfg("NADROWSKI", {"n": 50.0, "beta": 14.1})) == 50.0 * 14.1
    assert observable_noise_prefactor(_FakeCfg("HOPF", {"sigma_x": 0.1})) == 2.0 / 0.1 ** 2
    # BP (experimental): coupling 1/tau_hb, so prefactor = 2*tau_hb/eta_hb^2
    assert observable_noise_prefactor(_FakeCfg("BP", {"tau_hb": 1.0, "eta_hb": 0.5})) == 2.0 * 1.0 / 0.5 ** 2
    # a missing FDT parameter is a readable FDTModelError, not a bare KeyError
    try:
        observable_noise_prefactor(_FakeCfg("HOPF", {}))
    except FDTModelError as e:
        assert "sigma_x" in str(e), e
    else:
        raise AssertionError("missing HOPF param not reported")


def test_observable_noise_prefactor_user_additive():
    """An additive-noise user model's prefactor is 1/D_0 evaluated at the cell param values."""
    name = "FDTUADD"
    try:
        _register_user(name, [{"name": "x", "drift": "-k*x", "D": "Dx"}])
        pref = observable_noise_prefactor(_FakeCfg(name, {"k": 1.0, "Dx": 0.5}))
        assert pref == 1.0 / 0.5
    finally:
        registry.unregister(name)


def test_observable_noise_prefactor_user_rejects_multiplicative_zero_negative():
    name = "FDTUBAD"
    # multiplicative observable noise -> refused
    try:
        _register_user(name, [{"name": "x", "drift": "-k*x", "D": "0.5*x^2"}])
        try:
            observable_noise_prefactor(_FakeCfg(name, {"k": 1.0}))
        except FDTModelError as e:
            assert "multiplicative" in str(e), e
        else:
            raise AssertionError("multiplicative observable noise not refused")
    finally:
        registry.unregister(name)
    # zero and negative constant D_0 -> refused at runtime (D0 <= 0)
    try:
        _register_user(name, [{"name": "x", "drift": "-x", "D": "d0"}])
        for d0 in (0.0, -1.0):
            try:
                observable_noise_prefactor(_FakeCfg(name, {"d0": d0}))
            except FDTModelError as e:
                assert "non-positive" in str(e) or "zero" in str(e), e
            else:
                raise AssertionError(f"D0={d0} not refused")
    finally:
        registry.unregister(name)


def test_fdt_support_gate():
    """Built-ins are supported; user models are gated on additive, non-zero, unforced observable noise."""
    for m in ("NADROWSKI", "HOPF", "BP"):
        assert registry.fdt_support(m) == (True, ""), m
    assert registry.fdt_support("NOPE")[0] is False

    cases = [
        ("FDTGOK",  [{"name": "x", "drift": "-k*x", "D": "d0"}],                       True,  ""),
        ("FDTGMUL", [{"name": "x", "drift": "-k*x", "D": "0.5*x^2"}],                  False, "multiplicative"),
        ("FDTGZ",   [{"name": "x", "drift": "-x", "D": "0"}],                          False, "deterministic"),
        ("FDTGF",   [{"name": "x", "drift": "-k*x", "D": "d0",
                      "forcing": {"kind": "sin", "params": {}}}],                      False, "forcing"),
    ]
    for name, variables, ok, needle in cases:
        try:
            _register_user(name, variables)
            got_ok, reason = registry.fdt_support(name)
            assert got_ok is ok, (name, got_ok, reason)
            assert needle in reason, (name, reason)
        finally:
            registry.unregister(name)


def test_n_force_channels_user_vs_builtin():
    """A user model needs one force channel per state variable; built-ins keep the 1-or-2 convention."""
    name = "FDTNCH"
    try:
        _register_user(name, [{"name": "x", "drift": "v", "D": "0"},
                              {"name": "v", "drift": "-k*x", "D": "d0"}])
        assert _n_force_channels(_FakeCfg(name, n_vars=2)) == 2
    finally:
        registry.unregister(name)
    assert _n_force_channels(_FakeCfg("BP", force_params={})) == 1
    assert _n_force_channels(_FakeCfg("HOPF", force_params={"amp_y": 0.0})) == 2


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
