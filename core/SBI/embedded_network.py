from torch import nn
import torch

from core import config


def _probit(p: torch.Tensor) -> torch.Tensor:
    """Phi^-1, the standard-normal quantile function."""
    return torch.erfinv(2.0 * p - 1.0) * (2.0 ** 0.5)


def _column_quantiles(col: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``torch.quantile``'s linear interpolation, without its 2**24-element input ceiling.

    Deliberately sort-based rather than a call to ``torch.quantile``: at the production shape a
    column is 10.24M rows, which is UNDER that ceiling today and would silently stop fitting the
    moment TRAINING_NUM_RUNS or the batch width grew. One sort per channel is ~40 MB and runs once
    per training run, so there is nothing to buy by being clever here.
    """
    s = col.detach().reshape(-1).to(torch.float32).sort().values
    n = s.numel()
    if n == 1:
        return s.expand(p.numel()).clone()
    pos = p.to(s.dtype) * (n - 1)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = pos.ceil().long().clamp(0, n - 1)
    w = pos - lo.to(s.dtype)
    return s[lo] + w * (s[hi] - s[lo])


class EmbeddedNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer_dims: tuple,
                 forcing_dim: int = 0, forcing_layer_dims: tuple = None,
                 merge_layer_dim: int = None,
                 chi_k_pad: int = None, chi_band: tuple = None):
        """
        Embedding network with optional conditioning on forcing parameters.

        :param input_dim: width of the leading conditioning block routed to the summary
            pathway (summary statistics; in this pipeline log(T) is grouped here too)
        :param output_dim: final output dimension
        :param layer_dims: hidden layer dims for the summary pathway
        :param forcing_dim: width of the trailing conditioning block routed to the forcing
            pathway, e.g. the forcing parameters (0 = unconditioned)
        :param forcing_layer_dims: hidden layer dims for the forcing pathway
        :param merge_layer_dim: hidden layer dim for the merge pathway
        :param chi_k_pad: chi SET mode. The trailing block is a PADDED PROBE SET of ``chi_k_pad``
            slots, so the forcing pathway becomes a permutation-invariant ChiSetEncoder and this
            module takes over standardization of the WHOLE conditioning vector (see below).
            None = the original dense pathway, byte-unchanged.
        :param chi_band: (lo, hi) chi frequency bounds; fixes the encoder's frequency normalization.

        WHY CHI OWNS ITS STANDARDIZATION. sbi's ``posterior_nn(z_score_x=...)`` defaults to
        ``"independent"``, a PER-COLUMN affine over the whole conditioning vector. Over probe-slot
        columns that is inherently permutation-BREAKING -- two orderings of one probe set would be
        scaled differently, destroying the invariance the encoder exists to provide -- and the
        near-constant mask column becomes a ~1e7 amplifier under sbi's 1e-7 min-std clamp. So chi
        trains with ``z_score_x="none"`` and standardizes here instead: per-channel inside the
        encoder, and RANK-GAUSSIANISED for the summary block, both fitted over real data only.

        WHY RANK-GAUSSIANISATION AND NOT MEAN/STD. The affine this replaces
        was measured on the 2026-08-25 retrain's own artifact: ``A1_mean`` was fitted at
        std = 4.19e11 against a physical range of ~1e3, and ``D3_bimodality`` at std = 4.42e8
        against a range of (0, 1]. Both were driven there by a handful of pathological trajectories
        -- ~1e29-magnitude traces for A1, exactly-constant ones for D3 (which make ``_group_d``'s
        clamp fire and return exactly 1/1e-12). The consequence is that sweeping either channel
        across its ENTIRE physical range moved the embedding by 1.8e-7 / 8.9e-8, against ~1.4 for a
        healthy channel: less than one float32 ulp, i.e. the flow could not see two of its own
        conditioning channels at all.

        Rank-Gaussianisation fixes that by construction and buys three more things:

        * it is MONOTONE and invertible, so no information is lost;
        * being invariant to any monotone transform, it settles the log-versus-linear question for
          every channel at once (see REPARAM_LOG_PARAMS' history);
        * a sentinel point mass becomes a point mass at a KNOWN quantile the flow can key on, rather
          than a scale factor -- which matters because a large fraction of several ``_logp`` channels
          sits on exactly ``log(1e-12)``.

        LOADING AN OLDER POSTERIOR. A posterior trained before this change is a pickled
        DirectPosterior holding an EmbeddedNet whose buffers are ``sum_mean``/``sum_std``.
        Unpickling restores THOSE buffers into an instance of THIS class, so ``forward`` dispatches
        on which buffers are present rather than assuming the new ones. Without that branch every
        pre-2026-08-26 artifact becomes unloadable -- including ``posterior_08232026``, which is the
        baseline every conditioning-repair gate is measured against.
        """
        super().__init__()

        self.input_dim = input_dim
        self.forcing_dim = forcing_dim
        self.conditioned = forcing_dim > 0
        self.owns_standardization = chi_k_pad is not None

        # Summary pathway
        self.summary_net = nn.Sequential(
            nn.Linear(input_dim, layer_dims[0]),
            nn.LayerNorm(layer_dims[0]),
            nn.GELU(),

            nn.Linear(layer_dims[0], layer_dims[1]),
            nn.LayerNorm(layer_dims[1]),
            nn.GELU(),
        )

        if self.conditioned:
            if forcing_layer_dims is None or merge_layer_dim is None:
                raise ValueError(
                    "forcing_layer_dims and merge_layer_dim must be provided when forcing_dim > 0"
                )

            if self.owns_standardization:
                from core.SBI import chi as _chi
                from core.SBI.chi_encoder import ChiSetEncoder
                if forcing_dim != config.CHI_ELEM_W * chi_k_pad:
                    raise ValueError(
                        f"chi set mode: forcing_dim={forcing_dim} does not match "
                        f"CHI_ELEM_W*chi_k_pad={config.CHI_ELEM_W * chi_k_pad}.")
                u_mid, u_half = _chi.band_norm(chi_band)
                self.forcing_net = ChiSetEncoder(chi_k_pad, u_mid, u_half,
                                                 out_dim=forcing_layer_dims[1])
                # Decoded by reparam.posterior_mode tier 2 straight off the trained net.
                self.chi_k_pad = chi_k_pad
                self.chi_layout = config.CHI_LAYOUT
                # Summary-block standardization, ours because z_score_x is "none" under chi.
                # rg_knots[c] is channel c's empirical quantile function sampled at the mid-point
                # levels below; rg_z is the shared probit of those levels. Monotone piecewise-linear
                # between them, so the pair IS the transform -- there is no fitted scale to go wrong.
                q = int(config.RANK_GAUSS_KNOTS)
                p = (torch.arange(q, dtype=torch.float64) + 0.5) / q
                self.register_buffer("rg_knots", torch.zeros(input_dim, q))
                self.register_buffer("rg_z", _probit(p).to(torch.float32))
                self.register_buffer("rg_keep", torch.zeros(input_dim, dtype=torch.uint8))
            else:
                self.forcing_net = nn.Sequential(
                    nn.Linear(forcing_dim, forcing_layer_dims[0]),
                    nn.LayerNorm(forcing_layer_dims[0]),
                    nn.GELU(),

                    nn.Linear(forcing_layer_dims[0], forcing_layer_dims[1]),
                    nn.LayerNorm(forcing_layer_dims[1]),
                    nn.GELU(),
                )

            self.merge_net = nn.Sequential(
                nn.Linear(layer_dims[1] + forcing_layer_dims[1], merge_layer_dim),
                nn.LayerNorm(merge_layer_dim),
                nn.GELU(),

                nn.Linear(merge_layer_dim, output_dim),
            )
        else:
            # No forcing: just project summary output to final dimension
            self.output_net = nn.Linear(layer_dims[1], output_dim)

    # --- summary-block standardization ------------------------------------------------------
    @staticmethod
    def rank_gaussianize(x: torch.Tensor, knots: torch.Tensor, z: torch.Tensor,
                         keep: torch.Tensor) -> torch.Tensor:
        """(B, C) -> (B, C), monotone per channel. Public (and static) so the tests and the ablation
        script can exercise the transform without a trained net.

        THE TIE RULE IS THE PART THAT MATTERS. A channel with a large sentinel mass produces a long
        run of IDENTICAL knots, and a query landing on that value must map to the MID-RANK of the
        run. Taking whichever end ``searchsorted`` happens to return would put the whole point mass
        at one edge of its own rank interval, so the sign of the resulting jump would depend on a
        float comparison rather than on the data. The tie branch is applied LAST, so a value equal to
        the first or last knot is resolved as a tie rather than by the out-of-range clamps.
        """
        q = knots.shape[1]
        xt = x.transpose(0, 1).contiguous()                              # (C, B)
        lo = torch.searchsorted(knots, xt, right=False)                  # first knot >= x
        hi = torch.searchsorted(knots, xt, right=True)                   # first knot >  x

        j = lo.clamp(1, q - 1)
        x0 = torch.gather(knots, 1, j - 1)
        x1 = torch.gather(knots, 1, j)
        z0, z1 = z[j - 1], z[j]
        w = ((xt - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)
        out = z0 + w * (z1 - z0)

        # Outside the fitted range: clamp to the extreme knot rather than extrapolating a probit.
        out = torch.where(lo == 0, z[0].expand_as(out), out)
        out = torch.where(lo >= q, z[q - 1].expand_as(out), out)
        # Exactly ON a knot (or a run of them): the mid-rank of that run. LAST, see the docstring.
        mid = 0.5 * (z[lo.clamp(max=q - 1)] + z[(hi - 1).clamp(min=0)])
        out = torch.where(hi > lo, mid, out)

        out = out.transpose(0, 1)
        return torch.where(keep.to(torch.bool), out, torch.zeros_like(out))

    @torch.no_grad()
    def fit_standardization(self, x: torch.Tensor) -> None:
        """Fit both standardizers from the post-filter training tensor. chi set mode only."""
        if not self.owns_standardization:
            return
        s = x[:, :self.input_dim]
        # Pass a near-constant column THROUGH rather than ranking it. sbi's affine equivalent clamps
        # at min_std=1e-7, which turns such a column into a ~1e7 amplifier of its own rounding --
        # and under chi, Group G's 11 columns are identically zero BY CONSTRUCTION. Ranking a
        # constant is undefined in-distribution and actively harmful out of it: an observation whose
        # dead channel is not exactly the training constant would clamp to +-z_max and inject a
        # full-scale signal from a channel that carries nothing.
        keep = s.std(0) > 1e-6
        q = self.rg_knots.shape[1]
        p = (torch.arange(q, dtype=torch.float64) + 0.5) / q
        for c in range(self.input_dim):
            if bool(keep[c]):
                self.rg_knots[c] = _column_quantiles(s[:, c], p).to(self.rg_knots.dtype)
        self.rg_keep.copy_(keep.to(torch.uint8))
        self.forcing_net.fit(x[:, self.input_dim:])

    def standardize_summary(self, s: torch.Tensor) -> torch.Tensor:
        """The summary block's standardizer, whichever this instance carries.

        A pre-2026-08-26 artifact unpickles with ``sum_mean``/``sum_std`` and no rank buffers; it
        keeps the affine it was trained under, because re-standardising a trained flow's input is
        not a fix, it is a different network. Anything else is a net that was never fitted.
        """
        if "rg_knots" in self._buffers:
            return self.rank_gaussianize(s, self.rg_knots, self.rg_z, self.rg_keep)
        if "sum_mean" in self._buffers:                     # legacy affine, see the class docstring
            return (s - self._buffers["sum_mean"]) / self._buffers["sum_std"]
        raise RuntimeError(
            "EmbeddedNet.standardize_summary(): this instance carries neither the rank-Gaussian "
            "buffers nor the legacy sum_mean/sum_std pair, so the summary block has no standardizer "
            "at all. A net built by build_embedding_net always has one; this is a hand-constructed "
            "or partially-unpickled module.")

    @property
    def standardization_fitted(self) -> bool:
        return bool(getattr(self.forcing_net, "fitted", torch.ones(()))) if self.owns_standardization else True

    def forward(self, x):
        if self.conditioned:
            s = x[:, :self.input_dim]
            f = x[:, self.input_dim:]
            if self.owns_standardization:
                s = self.standardize_summary(s)
            return self.merge_net(torch.cat([
                self.summary_net(s),
                self.forcing_net(f)
            ], dim=-1))
        else:
            return self.output_net(self.summary_net(x))
