"""Runtime model registry: the three built-ins plus user-defined models (Resources/Models/*.json).

The many existing consumers of ``config.VALID_MODELS`` / ``config.VALID_LABELS`` index the two lists
positionally, so user models are APPENDED IN PLACE to those live lists -- the built-ins keep indices
0-2 and nothing downstream changes. Loading is an explicit call (``load_user_models()``, from
core.gui.app.build_app and core.__main__), never an import side effect: the GUI test suite constructs
MainWindow directly and must keep seeing pristine registries.

User models are Simulate-only in v1: they carry no Prior/INIT_SHAPES entries and the FDT/Inference
panels gate them out; ``is_user_model`` is the single source of truth for that gate.
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch

from core import config


@dataclass
class ModelSpec:
    name: str                          # canonical (uppercase) name as shown in every model dropdown
    labels: list                       # per-ND-param plot labels (built-ins: LaTeX; user models: names)
    state_dep_drift: bool = False      # user models: True iff any D references state (multiplicative noise)
    is_user_model: bool = False
    n_vars: int = 0
    variables: list = field(default_factory=list)   # user models: [{name, drift, D, init, forcing}]
    compiled: object = None            # user models: user_model.CompiledUserModel
    spec_path: Path | None = None      # user models: the Resources/Models/<name>.json source
    # ND parameter NAMES this model asks to be boxed in log space -- the per-model counterpart of
    # config.REPARAM_LOG_PARAMS, read by orchestrator.build_prior. Carried on the spec because the
    # saved doc is otherwise read and discarded at load time, leaving nothing downstream to ask.
    # Empty for every built-in: theirs is a global setting, not a per-model one.
    log_params: list = field(default_factory=list)


_BUILTINS = ("BP", "NADROWSKI", "HOPF")
_SPECS: dict = {
    "BP":        ModelSpec("BP", config.BP_LABELS, state_dep_drift=False, n_vars=5),
    "NADROWSKI": ModelSpec("NADROWSKI", config.NADROWSKI_LABELS, state_dep_drift=True, n_vars=3),
    "HOPF":      ModelSpec("HOPF", config.HOPF_LABELS, state_dep_drift=False, n_vars=2),
}

# (path, message) per model file that failed to load -- shown on the Settings screen, never raised.
load_errors: list = []


def get(name: str) -> "ModelSpec | None":
    return _SPECS.get(str(name).upper())


def is_user_model(name: str) -> bool:
    spec = get(name)
    return spec is not None and spec.is_user_model


def user_model_has_forcing(name: str) -> bool:
    """True if any variable of a user model declares a forcing entry (-> Simulate-only in v2)."""
    spec = get(name)
    return bool(spec and spec.is_user_model and any(v.get("forcing") for v in spec.variables))


def is_sbi_user_model(name: str) -> bool:
    """A user model eligible for the SBI (Parameter Inference) path in v2: user-defined, NO forcing
    (spontaneous dynamics only), and at least one ND parameter to infer. A forced model keeps the
    sinusoid machinery out of scope; a zero-parameter model (e.g. pure SHM) has nothing to infer and
    can't build a stability-screened GMM prior -- both stay Simulate-only."""
    spec = get(name)
    return bool(
        spec and spec.is_user_model
        and not user_model_has_forcing(name)
        and spec.compiled is not None
        and len(spec.compiled.param_names) >= 1
    )


def fdt_support(name: str) -> tuple:
    """Whether the FDT effective-temperature pipeline can run for this model, and why not.

    Built-ins are always supported (their observable-noise prefactor is coded per model in
    ``campaigns.observable_noise_prefactor``). A user model is supported only when its OBSERVABLE
    (variable index 0) has ADDITIVE, non-zero white noise and the model carries NO intrinsic forcing --
    FDT overwrites the force tensor to probe chi(omega), so a user's own drive would be silently
    dropped. Returns (ok, reason); ``reason`` is '' when ok. The single source of truth for the FDT
    GUI gate (mirrors ``is_sbi_user_model`` for the SBI gate)."""
    spec = get(name)
    if spec is None:
        return False, f"Unknown model '{name}'."
    if not spec.is_user_model:
        return True, ""
    if spec.compiled is None:
        return False, f"User model '{name}' has no compiled definition."
    if user_model_has_forcing(name):
        return False, (f"FDT can't run '{name}': it has intrinsic forcing, but FDT drives the "
                       "observable itself to measure the response. Remove the forcing to use FDT.")
    c = spec.compiled
    obs, d0 = c.var_names[0], c.diff_exprs[0]
    if {str(s) for s in d0.free_symbols} & set(c.var_names):
        return False, (f"FDT can't run '{name}': observable '{obs}' has state-dependent "
                       "(multiplicative) noise. FDT supports additive-noise observables only.")
    if d0.is_zero:
        return False, (f"FDT can't run '{name}': observable '{obs}' is deterministic (D=0). "
                       "FDT needs a stochastic observable.")
    return True, ""


def state_dep_drift(name: str) -> bool:
    """The model's state-dependent-diffusion flag; falls back to the legacy name test for unknowns."""
    spec = get(name)
    if spec is not None:
        return spec.state_dep_drift
    return "nadrowski" in str(name).lower()


def register(spec: ModelSpec) -> None:
    """Register (or re-register) a spec and keep the live VALID_MODELS/VALID_LABELS lists in sync.

    Append-only for new names: the built-ins MUST keep indices 0-2 (several call sites and tests index
    the parallel lists positionally).
    """
    _SPECS[spec.name] = spec
    if spec.name in config.VALID_MODELS:
        config.VALID_LABELS[config.VALID_MODELS.index(spec.name)] = spec.labels
    else:
        config.VALID_MODELS.append(spec.name)
        config.VALID_LABELS.append(spec.labels)
    assert list(config.VALID_MODELS[:3]) == list(_BUILTINS), \
        "VALID_MODELS built-in order changed -- positional label lookups would silently break"


def unregister(name: str) -> None:
    """Remove a USER model from the registry + the live lists. Built-ins cannot be removed."""
    name = str(name).upper()
    spec = _SPECS.get(name)
    if spec is None or not spec.is_user_model:
        return
    del _SPECS[name]
    if name in config.VALID_MODELS:
        idx = config.VALID_MODELS.index(name)
        config.VALID_MODELS.pop(idx)
        config.VALID_LABELS.pop(idx)


def user_model_names() -> list:
    return [s.name for s in _SPECS.values() if s.is_user_model]


def load_user_models() -> None:
    """Scan config.MODELS_PATH (at CALL time -- tests monkeypatch the path) and register every valid
    saved model; collect per-file failures into ``load_errors``. Idempotent; never raises."""
    from core.Helpers import model_store
    from core.Models.user_model import parse_user_model, build_compiled_step

    load_errors.clear()
    for path in model_store.list_user_models():
        try:
            doc = model_store.load_user_model(path)
            compiled = parse_user_model(doc["variables"])
            # Best-effort JIT fast path (used only on CUDA; None -> eager euler). Kept OUT of the
            # builder's validate (correctness-only) so the GUI thread never pays the torch.jit.script cost.
            compiled.compiled_step_fn = build_compiled_step(compiled)
            register(ModelSpec(
                name=doc["name"],
                labels=list(compiled.param_names),
                state_dep_drift=compiled.state_dep_noise,   # multiplicative noise -> solver g(x) per step
                is_user_model=True,
                n_vars=len(compiled.var_names),
                variables=doc["variables"],
                compiled=compiled,
                spec_path=Path(path),
                log_params=model_store.nd_log_params(doc),
            ))
        except Exception as e:                                  # noqa: BLE001 -- one bad file must not
            load_errors.append((Path(path), str(e)))            # brick startup (CrossValPanel precedent)


def make_user_simulator(spec: ModelSpec, params: torch.Tensor, force: torch.Tensor,
                        inits: torch.Tensor, t: torch.Tensor, *, freqs_per_batch: int = 1,
                        segs: int = 1, batch_size: int = 1,
                        device: torch.device = torch.device('cpu')):
    """Construct a UserSimulator for a registered user-model spec (lazy import keeps registry light).
    ``freqs_per_batch`` supports the FDT Campaign-2 frequency packing; it defaults to 1, so existing
    SBI callers are unaffected."""
    from core.Simulator.user_simulator import UserSimulator
    if spec.compiled is None:
        raise RuntimeError(f"User model '{spec.name}' has no compiled definition.")
    return UserSimulator(spec.compiled, params, force, inits, t, freqs_per_batch=freqs_per_batch,
                         segs=segs, batch_size=batch_size, device=device)
