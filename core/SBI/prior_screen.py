"""Prior construction (the stability screen), split out of pipeline.py.

Consumers -- orchestrator.build_prior via ``pipeline.gen_prior`` (which tests monkeypatch on the
pipeline module), and nothing else -- keep reaching this through the facade's bottom re-import.
"""
import torch

from core import config
from .Priors import bp_prior, hopf_prior, nadrowski_prior

VALID_PRIORS: dict = {"bp":        bp_prior.BPPrior,
                      "nadrowski": nadrowski_prior.NadrowskiPrior,
                      "hopf":      hopf_prior.HopfPrior}


def gen_prior(model: str, t: torch.Tensor, global_batch_size: int, local_batch_size: int, segs: int, prior_bounds: list,
              state_dep_drift: bool = False, num_iterations: int = 25, log_mask: torch.Tensor | None = None,
              n_max: int | None = None, step: float | None = None,
              min_cluster_size: int | None = None, min_samples: int | None = None,
              dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> torch.distributions.MixtureSameFamily:
    """
    Construct the stability-screened GMM prior for ``model`` over ``prior_bounds``.

    :param model: a built-in prior owner ("BP" / "NADROWSKI" / "HOPF") or an SBI-capable user model.
    :param t: ND time grid the stability sweep integrates over.
    :param global_batch_size: candidates per global sweep round; ``local_batch_size`` the local
        flood-fill's batch; ``segs`` the integration segments; ``num_iterations`` the global rounds.
    :param prior_bounds: per-parameter (lo, hi) list; its length is the parameter count.
    :param n_max: accepted parameter sets that STOP the local flood-fill; None reads
                  ``config.PRIOR_SWEEP_MAX_SETS``. This is the point cloud HDBSCAN clusters and the
                  GMM is fitted to, so it buys COVERAGE of the stable manifold, not precision.
    :param step: random-walk stride for the flood-fill's perturbation, in PHYSICAL parameter units;
                 None reads ``config.PRIOR_SWEEP_STEP``.
    :param min_cluster_size: HDBSCAN's floor on an island of stable parameters; None reads
                 ``config.PRIOR_CLUSTER_MIN_SIZE``. Its label count IS the GMM's component count.
    :param min_samples: HDBSCAN's density conservatism; None reads
                 ``config.PRIOR_CLUSTER_MIN_SAMPLES``. Higher declares more points noise.
    :return: the fitted ``torch.distributions.MixtureSameFamily``.
    :raises ValueError: unknown model, or a user model that is Simulate-only.
    """
    from core import registry
    if registry.is_user_model(model):
        if not registry.is_sbi_user_model(model):
            raise ValueError(
                f"'{model}' is a user-defined model that is Simulate-only (it has forcing or no free "
                "parameters). Parameter inference supports user models with no forcing and >=1 parameter.")
        from core.SBI.Priors.user_prior import UserPrior
        prior = UserPrior(registry.get(model), dtype, device)
    elif VALID_PRIORS.get(model.lower()) is None:
        raise ValueError(f"Invalid simulator: {model}")
    else:
        prior = VALID_PRIORS[model.lower()](dtype, device)

    n_params = len(prior_bounds)

    with torch.no_grad():
        # n_max and step were INVISIBLE before 2026-08-27: n_max was the literal 175000 right here,
        # silently overriding construct_prior's own default, and `step` was not threaded at all so
        # that default always won whatever a caller asked for. Both are config constants now and both
        # arrive as arguments -- the same fix C-7 applied to num_iterations.
        prior = prior.construct_prior(t, n_params, global_batch_size, local_batch_size, segs, prior_bounds,
                                      t_global_scale=2, num_iterations=num_iterations,
                                      n_max=config.PRIOR_SWEEP_MAX_SETS if n_max is None else int(n_max),
                                      step=config.PRIOR_SWEEP_STEP if step is None else float(step),
                                      steady=False,
                                      min_cluster_size=min_cluster_size, min_samples=min_samples,
                                      state_dep_drift=state_dep_drift, log_mask=log_mask)

    return prior

