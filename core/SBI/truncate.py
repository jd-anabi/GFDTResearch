"""TSNPE: truncated sequential NPE (Deistler, Goncalves & Macke 2022). Section 11.6.

    posterior -> HPD region A -> sample theta from the PRIOR RESTRICTED TO A -> simulate -> retrain

⚠⚠ THE ONE THING THAT MUST NOT BE GOT WRONG, and it is why this module exists rather than a few lines
inside build_posterior:

    THE PROPOSAL IS THE TRUNCATED PRIOR. IT IS NEVER THE POSTERIOR.

The posterior only ever says WHERE TO LOOK. Fitting a density to the posterior and proposing from it
gives ``p_L ∝ L^(L+1) q`` -- tempering. Credible intervals then contract as ``(L+1)^(-1/2)`` with NO
new information entering: at L = 4 the posterior is 2.2x narrower than the data supports. And SBC
comes out FLAT anyway, because SBC validates the flow against the proposal it was trained on. No
diagnostic in this project catches that, which is why ``tests/`` carries a dedicated pinning test
asserting the round-1 credible width does not contract by sqrt(2).

Truncation is a RESTRICTION, not a REWEIGHTING, which is the property that makes TSNPE need no
proposal correction where SNPE-A/B/C each need one: within A the proposal IS the prior, up to the
constant 1/P(A), so the NPE loss is unchanged there and zero outside.

WHY THE REGION LIVES IN THE FISHER EIGENBASIS (guardrail 3). ``k``, ``delta_E`` and ``temp`` sit at or
near prior on ``posterior_08232026``, and ``k`` in particular is FLAT rather than aliased -- 99.9% of
its weight on one direction loading -1.00*k. An HPD box drawn in 13-D physical space would cut those
axes on NOISE, and deleted support is permanent: truncation is a one-way ratchet, and a round-2 run
cannot recover a region round 1 threw away. So the region is expressed along the flow's own latent
axes, which under REPARAM_ROTATE *are* V's columns -- truncate directions 0..K-1, leave the flat ones
full width. This also dissolves the addendum's own section 8.3 flaw 2 outright: in that basis the
region is approximately axis-aligned, so there is no curved ridge to fragment and no clustering
needed at all.
"""
import torch

# 99.9%, not 95%: guardrail 5. The cost of an over-wide region is simulations; the cost of a narrow
# one is deleted support that no later round can recover.
DEFAULT_HPD = 0.999
# How many of the best-constrained directions to truncate. The rest keep full prior width.
DEFAULT_N_DIRECTIONS = 5

# The rejection sampler's blind first guess at P(A), and the ceiling on any single proposal draw.
# Both exist to keep a tight region from turning into one enormous allocation before there is any
# measurement to size it from.
_FIRST_PASS_RATE = 0.25
_MAX_DRAW = 1_000_000


class TruncationRegion:
    """An axis-aligned box in the flow's LATENT coordinate, over a subset of directions.

    Latent, not physical, and that is load-bearing -- see the module docstring. ``dims`` are indices
    into the latent vector; under REPARAM_ROTATE, dim j is column j of the Fisher rotation V, so it
    is the same "direction j" that scripts/posterior_identifiability.py reports.
    """

    def __init__(self, dims, lo, hi, *, level: float = DEFAULT_HPD, n_latent: int | None = None):
        self.dims = [int(d) for d in dims]
        self.lo = torch.as_tensor(lo, dtype=torch.float64).reshape(-1)
        self.hi = torch.as_tensor(hi, dtype=torch.float64).reshape(-1)
        if not (len(self.dims) == self.lo.numel() == self.hi.numel()):
            raise ValueError(f"TruncationRegion: {len(self.dims)} dims but {self.lo.numel()} lo / "
                             f"{self.hi.numel()} hi bounds.")
        if bool((self.hi <= self.lo).any()):
            raise ValueError("TruncationRegion: every interval must have hi > lo.")
        self.level = float(level)
        self.n_latent = None if n_latent is None else int(n_latent)

    def contains(self, z: torch.Tensor) -> torch.Tensor:
        """(N, P) latent -> (N,) bool. Untruncated directions are unconstrained by construction."""
        lo = self.lo.to(z.device, z.dtype)
        hi = self.hi.to(z.device, z.dtype)
        sel = z[:, self.dims]
        return ((sel >= lo) & (sel <= hi)).all(dim=1)

    def to_dict(self) -> dict:
        return {"basis": "fisher-latent", "dims": list(self.dims), "level": self.level,
                "lo": self.lo.clone(), "hi": self.hi.clone(), "n_latent": self.n_latent}

    @staticmethod
    def from_dict(d: dict) -> "TruncationRegion":
        return TruncationRegion(d["dims"], d["lo"], d["hi"],
                                level=d.get("level", DEFAULT_HPD), n_latent=d.get("n_latent"))

    def __repr__(self) -> str:
        parts = ", ".join(f"d{d}:[{float(a):.3g},{float(b):.3g}]"
                          for d, a, b in zip(self.dims, self.lo, self.hi))
        return f"TruncationRegion(level={self.level:.4g}, {parts})"


def region_from_posterior(posterior_latent, x_obs: torch.Tensor, *,
                          n_directions: int = DEFAULT_N_DIRECTIONS,
                          level: float = DEFAULT_HPD, n_samples: int = 20000) -> TruncationRegion:
    """Draw from the posterior at ``x_obs`` and take a per-direction HPD interval in LATENT space.

    ⚠ GUARDRAIL 4: UNWEIGHTED draws. Not "the M best fits". Selecting on goodness of fit applies a
    second, undeclared likelihood with the discrepancy metric as a hidden hyperparameter --
    ``overlay.posterior_overlay`` takes the best 50, which is right for a figure and wrong as the seed
    of a prior.

    The interval is a marginal quantile range per direction, which is what makes the region a box.
    That is deliberately conservative: a box containing the 99.9% marginal of every truncated
    direction contains AT LEAST the 99.9% joint HPD, never less.
    """
    if x_obs is None:
        raise ValueError(
            "region_from_posterior needs the observation the region is being drawn around. An "
            "amortized posterior has no default_x -- persist x_obs at INFERENCE time and pass it "
            "here (section 11.6 guardrail 1).")
    with torch.no_grad():
        z = posterior_latent.sample((int(n_samples),), x=x_obs)
    z = z.detach().to(torch.float64).cpu()
    p = z.shape[-1]
    k = max(1, min(int(n_directions), p))
    tail = (1.0 - float(level)) / 2.0
    q = torch.tensor([tail, 1.0 - tail], dtype=torch.float64)
    dims = list(range(k))                       # latent axes are already sorted best-constrained first
    bounds = torch.quantile(z[:, dims], q, dim=0)
    return TruncationRegion(dims, bounds[0], bounds[1], level=level, n_latent=p)


class TruncatedLatentPrior:
    """The latent prior RESTRICTED to a region. Sampling is rejection; nothing is reweighted.

    Deliberately a thin wrapper with the same duck-type ``gen_training_data`` already expects of a
    prior (``sample``, ``log_prob``), rather than a torch Distribution: the latent prior it wraps is
    itself a hand-built ProductPrior/RotatedLatentPrior, and the pipeline's standing rule for it is
    sample-only.

    ``log_prob`` returns the BASE log-density inside the region and -inf outside, i.e. it is off by
    the constant log P(A). NPE never needs that constant -- the loss is over q(theta|x) and, because
    truncation is a restriction rather than a reweighting, TSNPE applies no proposal correction at
    all. Anything that does need a normalised density must estimate P(A) itself; ``acceptance_rate``
    is the estimator.
    """

    def __init__(self, base, region: TruncationRegion, *, max_tries: int = 64):
        self.base = base
        self.region = region
        self.max_tries = int(max_tries)
        self._accepted = 0
        self._proposed = 0

    def sample(self, sample_shape=torch.Size()):
        n = int(torch.Size(sample_shape).numel()) if len(torch.Size(sample_shape)) else 1
        out, got, tries = [], 0, 0
        while got < n and tries < self.max_tries:
            # Over-draw by the MEASURED acceptance rate, and only once there is one to measure.
            # Seeding the first pass with a 1e-3 floor asked for (n / 1e-3) * 1.3 draws before any
            # evidence -- 2.66 MILLION rows for a 2048-row batch, out of a 13-D GMM, on the very
            # first call. The first pass is a probe: draw a modest multiple, then let the observed
            # rate size the rest. _MAX_DRAW caps any single allocation so a pathologically tight
            # region fails through max_tries with a message rather than through an OOM.
            rate = self.acceptance_rate if self._proposed else _FIRST_PASS_RATE
            want = int((n - got) / max(rate, 1e-6) * 1.3) if rate else (n - got) * 4
            draw = self.base.sample((min(max(want, n - got, 64), _MAX_DRAW),))
            keep = draw[self.region.contains(draw)]
            self._proposed += draw.shape[0]
            self._accepted += keep.shape[0]
            if keep.shape[0]:
                out.append(keep[: n - got])
                got += out[-1].shape[0]
            tries += 1
        if got < n:
            raise RuntimeError(
                f"TruncatedLatentPrior: only {got} of {n} draws landed inside the truncation region "
                f"after {tries} attempts (acceptance ~{self.acceptance_rate:.2e}). The region is far "
                f"out in the prior's tail, which usually means the posterior it came from disagrees "
                f"with the prior rather than sharpening it -- check the observation before widening "
                f"the HPD level.")
        z = torch.cat(out, dim=0)
        return z if len(torch.Size(sample_shape)) else z[0]

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        lp = self.base.log_prob(theta)
        inside = self.region.contains(theta.reshape(-1, theta.shape[-1]))
        return torch.where(inside.reshape(lp.shape), lp, torch.full_like(lp, float("-inf")))

    @property
    def acceptance_rate(self) -> float:
        """Measured P(A) under the prior. Also guardrail 5's honest failure rate: 1 - this is the
        fraction of prior mass the round threw away."""
        return (self._accepted / self._proposed) if self._proposed else 0.0

    def __getattr__(self, name):
        # Everything else (device, dims, event_shape, ...) is the base prior's business.
        # KeyError would be WRONG here: __getattr__ is consulted before __init__ has run during
        # copy/pickle, and Python requires an AttributeError to treat the attribute as absent --
        # a KeyError escapes and breaks copying instead of falling through.
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(base, name)
