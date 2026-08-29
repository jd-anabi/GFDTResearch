"""Shared setup for the diagnostic scripts in this folder.

WHY THIS EXISTS. Nine scripts each carried a copy of the same five-line ``SimConfig(...)`` literal,
built by hand from ``cli.parse_cell``. That literal hard-coded ``model="NADROWSKI"`` and
``state_dep_drift=True``, and -- the part that actually bit -- it set **no chi(omega) fields at all**.
``SimConfig.chi_mode`` is a plain ``= False`` default rather than a ``default_factory`` reading
``config.CHI_MODE``, and only ``cli.make_sim_config`` bridges the module constants onto a config. So
a hand-built config was PERMANENTLY non-chi, no matter what the global said, and every script
silently measured the single-frequency information set even when pointed at a chi run.

Routing through ``cli.make_sim_config`` + ``cli.load_and_validate_gt`` -- the pair the GUI's
``simulate_runner.build_stream_config`` already uses -- fixes chi, spontaneous configs, non-Nadrowski
models and user models in one move, and drops the deprecated ``si_factors`` argument from nine call
sites for free.

MODE DETECTION is three tiers, in strict precedence (see ``core.SBI.reparam.posterior_mode``):
  1. the posterior's ``.rot.pt`` sidecar (authoritative; always written since 2026-07-28),
  2. the trained network's own ``forcing_dim``,
  3. arithmetic on ``condition_shape`` (warns; breaks if the summary width ever changes).
Env knobs feed the CONFIG only. When a script also loads a posterior, the posterior VALIDATES the
config via :func:`require_mode`, which fails before the simulation spend rather than as a raw matrix
shape error thousands of simulations later.

Env knobs honoured here (each script documents its own on top of these):
  CELL      cell file path            (default Resources/Cells/nadrowski/master_spont.txt)
  BOUNDS    bounds file path          (default: the cell's sibling in Resources/Bounds/<model>/,
                                      else that model's shared master.txt)
  MODEL     override the model name   (default: derived from the cell's parent folder)
  TOBS_S    observation duration in SECONDS
  CHI       1/0 -- enable chi(omega) mode
  CHI_K     number of probe frequencies K
  CHI_F0    ND drive amplitude
  CHI_LO/CHI_HI   chi frequency bounds, as multiples of the measured Omega_0

⚠ CHI_F0 and CHI_LO/CHI_HI now require PRISM_CHI_OVERRIDE=1 for anything that reaches
``orchestrator.build_prior``/``build_posterior``. They set the BAND and DRIVE the network is trained
on, both baked into the encoder's weights, and a silent disagreement with ``config.CHI_*`` cost a
~5-day retrain once (Appendix A, 2026-08-19). Reading them is still free -- the guard fires only on
the two expensive entry points, so the measurement scripts that merely build a config are unaffected.
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core import cli, config, orchestrator, registry
from core.config import (SimConfig, BOUNDS_PATH, CELL_PATH, POSTERIOR_PATH, T_MIN_EXP_S,
                         VALID_LABELS, VALID_MODELS)
from core.SBI.reparam import (TransformedPosterior, load_eval_bijection, posterior_mode,
                              read_sidecar)

DEFAULT_CELL = str(CELL_PATH / "nadrowski" / "master_spont.txt")
# Several diagnostics read cfg.forcing_idx["amp"] unconditionally, so they need a bounds file that
# DECLARES a Forcing section -- which is a property of the CELL you point them at.
# master_weak resolves (sibling-first, then the folder's master.txt) to the forced 13-dim box; the
# spontaneous default above resolves to master_spont.txt and would KeyError on 'amp'.
FORCED_DEFAULT_CELL = str(CELL_PATH / "nadrowski" / "master_weak.txt")


def enable_warnings() -> None:
    """Undo the blanket ``warnings.filterwarnings("ignore")`` these scripts used to open with.

    EVERY out-of-distribution guard in the pipeline is a ``warnings.warn`` -- the N_ND_MAX ceiling,
    the prior-quantile flags, the recording-length checks, and the one that silently rewrites
    ``cfg.T_obs`` when the cost ceiling is hit. Suppressing everything meant the diagnostics ran
    blind to the exact safety net that was built for them. Narrow the filter to the noisy
    third-party categories instead.

    EXPECT NEW OUTPUT after adopting this. Warnings that were always being generated simply become
    visible; they are findings, not regressions.
    """
    warnings.resetwarnings()
    warnings.simplefilter("default")
    for category in (DeprecationWarning, PendingDeprecationWarning, ImportWarning, ResourceWarning):
        warnings.filterwarnings("ignore", category=category)
    # torch/sbi/sklearn chatter that carries no signal for these scripts.
    for module in (r"torch\..*", r"sbi\..*", r"sklearn\..*", r"pyro\..*", r"pytensor\..*"):
        warnings.filterwarnings("ignore", category=UserWarning, module=module)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_opt(name: str, cast):
    raw = os.environ.get(name)
    return None if raw is None else cast(raw)


def model_for_cell(cell: str, override: str | None = None) -> str:
    """The model a cell belongs to, from its PARENT FOLDER (Resources/Cells/<model>/<cell>.txt).

    Same rule ``cli.parse_cell`` uses -- the layout is the single source of truth, so a script never
    has to hard-code a model name.
    """
    if override:
        return override.upper()
    name = Path(cell).parent.name.upper()
    if name not in VALID_MODELS and not registry.is_user_model(name):
        raise SystemExit(
            f"Cannot tell which model '{cell}' belongs to: its parent folder is '{name}', which is "
            f"neither a built-in {VALID_MODELS} nor a registered user model. Pass MODEL=<name>.")
    return name


def default_bounds_for(cell: str, model: str) -> str:
    """The bounds file governing ``cell``. Bounds define the parameter SET, ORDER and ranges -- and
    hence the observation mode -- so this pairing is what makes a script's config match the cell.

    Delegates to ``cli.resolve_bounds_for_cell`` so scripts and the CLI/GUI agree on the answer:
    same-named sibling first, then the model's shared ``master.txt``. Duplicating the rule here is
    exactly how the two would drift.
    """
    path = cli.resolve_bounds_for_cell(cell, model)
    if path is None:
        raise SystemExit(
            f"No bounds file for cell '{cell}': tried the sibling "
            f"{BOUNDS_PATH / model.lower() / Path(cell).name} and the shared "
            f"{BOUNDS_PATH / model.lower() / cli.MASTER_BOUNDS_NAME}. Bounds declare which "
            f"parameters are inferred, in what order, so one is required. Pass BOUNDS=<path>.")
    return str(path)


def script_cfg(cell: str | None = None, *, default_cell: str | None = None,
               bounds: str | None = None, model: str | None = None,
               t_obs_s: float | None = None, chi: bool | None = None, chi_k: int | None = None,
               chi_f0: float | None = None, chi_bounds: tuple | None = None,
               load_gt: bool = True, quiet: bool = False) -> SimConfig:
    """Build a SimConfig the way the CLI and GUI do, with the chi knobs actually threaded through.

    Explicit arguments win over env vars, which win over the ``config.CHI_*`` module defaults (that
    last fallback is ``make_sim_config``'s own -- deliberately not duplicated here).

    :param cell:         an explicit cell path. WINS over the ``CELL`` env var -- so a script that
                         wants a different *default* must use ``default_cell``, not this. Passing a
                         literal here is what made the since-archived ``reparam_wiring_smoke``
                         unrunnable when its cell file was archived: ``CELL=`` could not override it.
    :param default_cell: the fallback when ``CELL`` is unset. For the forced-only diagnostics, see
                         :data:`FORCED_DEFAULT_CELL`.
    :param load_gt: inject the cell's ground-truth VALUES. Off for the handful of checks that only
                    need the bounds geometry.
    :param quiet:   suppress the resolved-configuration banner.
    """
    cell = cell or os.environ.get("CELL", default_cell or DEFAULT_CELL)
    model = model_for_cell(cell, model or os.environ.get("MODEL"))
    bounds = bounds or os.environ.get("BOUNDS") or default_bounds_for(cell, model)

    labels = (VALID_LABELS[VALID_MODELS.index(model)] if model in VALID_MODELS
              else registry.get(model).labels)
    chi_on = chi if chi is not None else _env_flag("CHI", None) if os.environ.get("CHI") else chi
    lo, hi = _env_opt("CHI_LO", float), _env_opt("CHI_HI", float)
    if chi_bounds is None and lo is not None and hi is not None:
        chi_bounds = (lo, hi)

    cfg = cli.make_sim_config(
        model, labels, registry.state_dep_drift(model), bounds,
        chi_mode=chi_on,
        chi_n_freqs=chi_k if chi_k is not None else _env_opt("CHI_K", int),
        chi_f0=chi_f0 if chi_f0 is not None else _env_opt("CHI_F0", float),
        chi_freq_bounds=chi_bounds,
    )

    ignored = cli.load_and_validate_gt(cfg, cell) if load_gt else []
    t_obs_s = t_obs_s if t_obs_s is not None else _env_opt("TOBS_S", float)
    cfg.T_obs = (t_obs_s if t_obs_s is not None else T_MIN_EXP_S) * cfg.get_unit_conversion_factor("s")

    if not quiet:
        describe(cfg, cell=cell, bounds=bounds, ignored=ignored)
    return cfg


def describe(cfg: SimConfig, *, cell: str = None, bounds: str = None, ignored=()) -> None:
    """Print the resolved configuration, including the PARAMETER ORDER.

    The order is printed on every run on purpose. Simulators bind parameter columns POSITIONALLY
    (``Model(*torch.unbind(params, dim=1), ...)``), so a bounds file whose order differs from what a
    script assumes mis-binds values with no error and no crash -- just wrong physics. Several scripts
    still index specific columns by hand, and nothing in the test suite imports ``scripts/``, so this
    banner is the only thing standing between that and a silently wrong result.
    """
    print(f"[cfg] model={cfg.model} mode={cfg.observation_mode.upper()} "
          f"device={cfg.hw.device} T_obs={cfg.T_obs:g} (cell time units)", flush=True)
    if cell:
        print(f"[cfg] cell={cell}", flush=True)
    if bounds:
        print(f"[cfg] bounds={bounds}", flush=True)
    if cfg.chi_mode:
        print(f"[cfg] chi: K={cfg.chi_n_freqs} F0={cfg.chi_f0} range={cfg.chi_freq_bounds} "
              f"x Omega_0  -> {cfg.chi_k_pad} probe slots, conditioning block = "
              f"{orchestrator.expected_forcing_dim(cfg)} features", flush=True)
    print(f"[cfg] ND order:      {list(cfg.params_dict.keys())}", flush=True)
    print(f"[cfg] rescale order: {list(cfg.rescale_params.keys())}", flush=True)
    if cfg.force_params_dict:
        note = " (IGNORED in chi mode -- chi probes at multiples of the measured Omega_0)" \
               if cfg.chi_mode else ""
        print(f"[cfg] forcing order: {list(cfg.force_params_dict.keys())}{note}", flush=True)
    if ignored:
        print(f"[cfg] cell values the bounds file does not declare, so IGNORED: {sorted(ignored)}",
              flush=True)


def load_posterior(name: str, cfg: SimConfig, *, check_mode: bool = True):
    """Load a saved posterior + its exact evaluation bijection.

    Collapses the ``torch.load`` / ``load_eval_bijection`` / ``TransformedPosterior`` boilerplate that
    four scripts carried near-verbatim, and -- by default -- refuses a posterior whose observation
    mode disagrees with ``cfg`` BEFORE any simulation is paid for.

    :return: ``(posterior_latent, T_eval, posterior_physical, sidecar)``.
    """
    path = POSTERIOR_PATH / name
    if not path.exists():
        raise SystemExit(f"No such posterior: {path}")
    posterior_latent = torch.load(str(path), map_location=cfg.hw.device, weights_only=False)
    sidecar = read_sidecar(name, POSTERIOR_PATH, map_location=cfg.hw.device)
    if check_mode:
        require_mode(cfg, posterior_latent, sidecar, name=name)
    T_eval = load_eval_bijection(cfg, name, POSTERIOR_PATH)
    return posterior_latent, T_eval, TransformedPosterior(posterior_latent, T_eval), sidecar


def require_mode(cfg: SimConfig, posterior_latent, sidecar: dict | None = None,
                 name: str = "posterior") -> tuple:
    """Fail LOUDLY and EARLY when a posterior's observation mode disagrees with ``cfg``.

    Without this, ``sbc_characterize`` generated an entire calibration set (K x N_CAL simulations)
    and only then died on a raw matrix-shape ``RuntimeError`` from inside ``EmbeddedNet``'s first
    ``Linear``. The three conditioning widths cannot collide, so the check is exact -- it just has to
    happen before the spend. Mirrors ``retrain_convergence``'s existing rotation guard.

    :return: ``(mode, forcing_dim, K or None)``.
    """
    try:
        mode, forcing_dim, k = posterior_mode(posterior_latent, sidecar)
    except ValueError as e:
        print(f"[mode] WARNING: cannot verify '{name}': {e}", flush=True)
        return cfg.observation_mode, None, None

    want_mode = cfg.observation_mode
    # ONE width rule, shared with build_posterior and the sidecar -- this used to be a third copy.
    want_dim = orchestrator.expected_forcing_dim(cfg)
    detail = f", K={k}" if k else ""
    print(f"[mode] {name}: {mode.upper()}{detail}, forcing/chi block = {forcing_dim} features",
          flush=True)
    if mode != want_mode or forcing_dim != want_dim:
        raise SystemExit(
            f"MODE MISMATCH: '{name}' was trained in {mode.upper()} mode{detail} with a "
            f"{forcing_dim}-feature forcing/chi block, but this config is {want_mode.upper()} "
            f"expecting {want_dim}. Their conditioning vectors are different widths, so nothing "
            f"downstream is meaningful.\n"
            f"  -> point CHI/CHI_K at the mode this posterior was trained in, or pick another "
            f"posterior. The chi toggle and the bounds file's Forcing section select the mode.")
    return mode, forcing_dim, k


# ── The mode's FEATURE SET ────────────────────────────────────────────────────────────────────────
# Every Jacobian / identifiability diagnostic in this folder must be built from the features the
# posterior ACTUALLY conditions on, or it answers a question about a different experiment:
#
#   spontaneous / forced : the full 41 summary features
#   chi                  : 30 summary features (Group G is ZEROED in this mode, so its 11 columns
#                          carry no information) + 3K chi features
#
# Left on the 41-feature assumption, these scripts kept Group G and omitted chi entirely, so their
# results were literally independent of the chi toggle -- they would report the kappa~x_scale /
# lambda~t_scale aliases as strong as ever and falsely refute the hypothesis chi mode exists to test.
_GROUP_G_PREFIX = "G"


def summary_keep_idx() -> list:
    """Indices of the summary features that survive in chi mode (everything outside Group G)."""
    from core.SBI.statistics import FEATURE_LABELS
    keep = [i for i, lbl in enumerate(FEATURE_LABELS) if not lbl.startswith(_GROUP_G_PREFIX)]
    n_dropped = len(FEATURE_LABELS) - len(keep)
    assert n_dropped == 11, f"expected 11 Group-G features, found {n_dropped}"
    return keep


def feature_labels(cfg: SimConfig) -> list:
    """Ordered labels for the Jacobian ROWS this config's diagnostics should be built over."""
    from core.SBI import chi as _chi
    from core.SBI.statistics import FEATURE_LABELS
    if not cfg.chi_mode:
        return list(FEATURE_LABELS)
    # The FISHER channel set (log|chi|, cos, sin), not the conditioning one: these diagnostics
    # build a Jacobian, and the conditioning block's `u` and `mask` columns are theta-independent
    # there -- see chi.CHI_FISHER_CHANNELS for what that does to a central difference.
    return ([FEATURE_LABELS[i] for i in summary_keep_idx()]
            + _chi.chi_labels(cfg.chi_n_freqs, _chi.CHI_FISHER_CHANNELS))


def n_features(cfg: SimConfig) -> int:
    """Row count of the mode's feature vector. Replaces the hardcoded 41 these scripts carried."""
    return len(feature_labels(cfg))


def describe_features(cfg: SimConfig) -> None:
    """One banner line so a result can never be read without knowing which information set made it."""
    if cfg.chi_mode:
        from core.SBI import chi as _chi
        n_sp, n_ch = len(summary_keep_idx()), len(_chi.CHI_FISHER_CHANNELS)
        print(f"[mode] CHI: feature rows = {n_sp} spontaneous + {n_ch * cfg.chi_n_freqs} chi "
              f"= {n_features(cfg)}  (Group G dropped: it is zeroed in this mode)", flush=True)
        print("[mode] NOTE f_scale is informative here (chi drives at amp = F0 * f_scale).", flush=True)
    else:
        print(f"[mode] {cfg.observation_mode.upper()}: feature rows = {n_features(cfg)} "
              f"(the full single-frequency feature set)", flush=True)


def assert_not_chi(cfg: SimConfig, what: str) -> None:
    """Refuse to run a diagnostic that has not been generalised past the single-frequency layout.

    Better a loud refusal than a plausible-looking number computed over the wrong feature set --
    which is precisely how these scripts failed before.
    """
    if cfg.chi_mode:
        from core.SBI import chi as _chi
        raise SystemExit(
            f"{what} has not been generalised to chi(omega) mode: it measures the single-frequency "
            f"41-feature information set, while a chi posterior conditions on "
            f"{n_features(cfg)} different features (Group G zeroed, {len(_chi.CHI_FISHER_CHANNELS) * cfg.chi_n_freqs} chi "
            f"features added). Running it here would produce a confident, meaningless answer.\n"
            f"  -> unset CHI to analyse the forced information set, or use scripts/degeneracy_map.py, "
            f"which is chi-aware.")


def assert_nadrowski(cfg: SimConfig, why: str = "") -> None:
    """Guard for scripts whose SCIENCE is Nadrowski-specific (hardcoded parameter roles, fixed
    column indices). Better a loud refusal than a plausible-looking wrong answer for another model."""
    if cfg.model != "NADROWSKI":
        raise SystemExit(
            f"This diagnostic is Nadrowski-specific{(' (' + why + ')') if why else ''}, but the "
            f"config is for {cfg.model}. It would run and produce meaningless numbers.")


def assert_forced(cfg: SimConfig, what: str) -> None:
    """Guard for diagnostics that read the cell's own drive (``cfg.forcing_idx["amp"]`` and friends).

    Whether a drive EXISTS is a property of the bounds file, not the cell values, so
    pointing one of these at a spontaneous cell used to surface as a bare ``KeyError: 'amp'`` twenty
    lines below the config banner. Say which file to point at instead.
    """
    if not cfg.has_forcing:
        raise SystemExit(
            f"{what} measures the response to the cell's OWN drive, but this config is "
            f"{cfg.observation_mode.upper()}: its bounds file declares no Forcing section, so there "
            f"is no amp/freq/phase to read.\n"
            f"  -> point it at a forced cell, e.g. CELL={FORCED_DEFAULT_CELL}")
