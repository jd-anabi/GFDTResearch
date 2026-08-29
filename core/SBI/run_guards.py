"""Run guards, split out of orchestrator: the checks that stop an expensive run before the spend.

Prior identity (a GMM's box does not identify it -- two sweeps over one box produce different
fits), the load-side prior/posterior agreement checks, the chi band/drive deliberateness gate, the
amortization gate for TSNPE artifacts, and the log-box resolution rule. orchestrator re-imports
every name, so the suites keep calling them as orchestrator._* and every existing call site is
unchanged. The guards that scan Resources through a test-sandboxable path (PRIOR_PATH rebinds on
orchestrator) stay in orchestrator: _saved_prior_fingerprints, _assert_prior_is_saved,
_refuse_to_orphan_a_checkpoint, and the width-computing _assert_mode_matches.
"""
import hashlib
import os

import torch

from core import config, registry
from core.config import POSTERIOR_PATH, SimConfig
from core.Helpers import file_manager
from core.SBI.reparam import read_sidecar

CHI_OVERRIDE_ENV = "PRISM_CHI_OVERRIDE"


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



# Below this many batches a generation run is short enough that losing it is an inconvenience rather
# than a day, so the unsaved-prior guard stays out of the way of smoke runs and experiments.


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


def _assert_chi_config_is_deliberate(cfg: SimConfig) -> None:
    """Refuse a chi run whose BAND or DRIVE silently disagrees with config.py's defaults.

    WHY THIS EXISTS, and it is not hypothetical. The 2026-08-19 retrain spent ~5 days and 10.24M rows
    training at ``chi_freq_bounds=(0.1, 10.0)``, ``chi_f0=0.1`` -- the band RETIRED on 2026-08-06,
    and verbatim the configuration of ``posterior_chi_08042026``, on record as
    "well-calibrated but uninformative ... 8 of its 10 probes measured noise". The values came from
    QSettings: ``inference_tabs`` seeds the chi widgets from ``config.CHI_*`` and then ``_restore``
    overwrites them from ``PRISM.ini``, so a value saved before the band changed was reapplied on
    every launch afterwards with nothing to say so. A persisted preference is the right behaviour;
    a persisted MEASUREMENT DEFINITION needs comparing against the module default before the spend.

    ``_assert_mode_matches`` already catches the same disagreement -- but only when a posterior is
    LOADED, i.e. after the days are spent. This fires before the first simulation.

    SCOPE IS DELIBERATELY NARROW. Only the band and the drive amplitude are checked, because only
    they shape the TRAINING distribution and are baked into the encoder's weights.
    ``chi_n_freqs`` is deliberately NOT an error: it is the count an OBSERVATION supplies, training
    draws K per batch so one posterior serves any count (see build_posterior's training_params, which
    omits it on purpose), and failing on it would refuse a perfectly good 7-recording experiment. It
    is reported alongside a real mismatch as context, never as the cause.

    :raises ValueError: on a band/drive mismatch, unless ``PRISM_CHI_OVERRIDE=1``. Deliberate band
        exploration is a real activity -- ``scripts/chi_f0_sweep.py`` exists for it -- so the escape
        hatch is explicit rather than absent.
    """
    if not cfg.chi_mode:
        return
    checks = (("chi_freq_bounds", tuple(float(v) for v in cfg.chi_freq_bounds),
               tuple(float(v) for v in config.CHI_FREQ_BOUNDS)),
              ("chi_f0", float(cfg.chi_f0), float(config.CHI_F0)))
    diffs = [(n, got, want) for n, got, want in checks
             if (got != want if isinstance(got, tuple) else abs(got - want) > 1e-12)]
    if not diffs:
        return
    detail = "\n".join(f"    {n:<16} this run: {got!r:<16} config.py: {want!r}" for n, got, want in diffs)
    k_note = ""
    if int(cfg.chi_n_freqs) != int(config.CHI_N_FREQS):
        k_note = (f"\n  (FYI, not an error: chi_n_freqs is {cfg.chi_n_freqs} against config's "
                  f"{config.CHI_N_FREQS}. K is per-observation and training draws its own, so it is "
                  f"legitimate -- but if you did not choose it either, it points at the same source.)")
    if os.environ.get(CHI_OVERRIDE_ENV) == "1":
        print(f"[chi] {CHI_OVERRIDE_ENV}=1 -- proceeding with a NON-DEFAULT chi configuration:\n"
              f"{detail}{k_note}", flush=True)
        return
    raise ValueError(
        f"This chi run's configuration does not match config.py, and the difference decides what the "
        f"network is trained on:\n{detail}{k_note}\n\n"
        f"  The band and drive amplitude fix the encoder's frequency normalization and are baked into "
        f"its weights, so a run at the wrong values cannot be reinterpreted afterwards -- it has to be "
        f"redone. This has cost a ~5-day run once already (Appendix A, 2026-08-19).\n"
        f"  MOST LIKELY CAUSE: stale persisted GUI settings. The Config tab seeds these from config.py "
        f"and then restores them from QSettings, so a value saved before a config change wins silently."
        f" Check the [inference_config] chi_lo / chi_hi / chi_f0 keys in PRISM.ini.\n"
        f"  If the difference is DELIBERATE (a band sweep, say), re-run with {CHI_OVERRIDE_ENV}=1.")


def _assert_amortization_understood(choice: str) -> None:
    """Refuse a TRUNCATED (non-amortized) posterior unless the caller opted into one.

    SECTION 11.6 GUARDRAIL 2. A truncated posterior is valid only near the observation its region was
    drawn around: outside that region the flow saw ZERO training rows, so it does not return the
    prior there, it returns whatever the flow extrapolates -- confidently. Amortized and truncated
    artifacts sit side by side in one ArtifactPicker, with nothing in the filename to tell them
    apart, which is precisely how the retired-band posterior cost a five-day run.

    A missing or amortized sidecar passes silently, so every existing artifact is unaffected.
    """
    side = read_sidecar(choice, POSTERIOR_PATH, map_location="cpu")
    if not side or side.get("amortized", True):
        return
    tr = side.get("truncation") or {}
    dims = tr.get("dims", [])
    raise ValueError(
        f"Posterior '{choice}' is NOT AMORTIZED: it was trained by TSNPE on a prior truncated to a "
        f"{tr.get('level', '?')}-HPD region along Fisher direction(s) {dims}, drawn around the "
        f"observation with digest {side.get('x_obs_digest')}. It is only valid for observations in "
        f"that region -- outside it the flow has never seen a training row and will extrapolate "
        f"confidently rather than return the prior. Use it from the TSNPE tab, which checks the "
        f"observation against that digest, or pick an amortized posterior for general inference.")


# Whether infer_and_visualize records the observation it ran against.
# A MODULE global so the suites can rebind it, exactly as they rebind TRAINING_CHECKPOINT_EVERY and
# for the same reason: the full-pipeline tests call infer_and_visualize, and left on they scatter a
# record into Resources/Observations on every run. Nothing else in the suite writes into Resources,
# and that property is worth keeping. It is NOT a user-facing switch -- a real inference always
# records, because TSNPE keys on the digest and an amortized posterior has no observation at save
# time.


def _log_params_for(cfg: SimConfig):
    """Which ND parameter names go in a LOG box, for this config's model.

    A user model carries its own choice per parameter (model_store's ``"box"`` field, surfaced as
    ``registry.ModelSpec.log_params``); a built-in has no per-model source and follows the global
    ``config.REPARAM_LOG_PARAMS``. Returning None for the built-in case is deliberate -- it is the
    "no override" sentinel every reparam entry point already understands (``_resolve_log_params``),
    so the built-in path is byte-identical to what it did before user models could express this.

    ONE resolver, called at every site that decides a box coordinate: build_prior's gen_prior mask,
    build_posterior's loaded-prior mask GUARD, the training bijection, and the sidecar. If two of
    those disagreed, the mismatch would surface as build_posterior's "REBUILD the ND prior" error
    against a prior that is in fact correct -- or, worse, not surface at all and train the flow in a
    different coordinate than the GMM was fitted in.
    """
    from core import registry                       # lazy: registry imports config, config imports us
    spec = registry.get(cfg.model)
    if spec is None or not spec.is_user_model:
        return None                                 # built-in -> config.REPARAM_LOG_PARAMS
    return list(spec.log_params)

