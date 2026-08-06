"""Permutation-invariant encoder for the chi(omega) probe SET (layout 2).

Replaces the dense forcing_net branch of EmbeddedNet, which consumed a fixed 3K vector whose slot
index carried the probe's frequency implicitly. Here each probe is a 6-tuple
``(u, log|chi|, cos, sin, logcyc, mask)`` carrying its own frequency in channel 0, so the number of
probes and where they sit are both free.

WHY IT OWNS ITS OWN STANDARDIZATION. sbi's ``posterior_nn(z_score_x=...)`` defaults to
``"independent"``, which fits a PER-COLUMN affine over the whole conditioning vector. Over 6*K_PAD
probe columns that is inherently permutation-BREAKING: two orderings of the same probe set would be
scaled differently, destroying the invariance this module exists to provide. Worse, the mask column
is near-constant, and sbi's ``min_std`` clamp of 1e-7 turns a near-constant column into a ~1e7
amplifier. So chi is trained with ``z_score_x="none"`` and this module standardizes PER CHANNEL
(shape (5,), shared across slots) over LIVE probes only.

WHY THERE IS NO MAX POOL. ``E[max of n iid N(0,1)]`` is 0.564 / 1.267 / 1.629 at n = 2 / 6 / 12, so a
masked max writes a ~1.07 sigma LOCATION shift into every pooled channel purely as a function of the
probe count -- exactly the K-dependence this design removes. The set is samples of a smooth function
chi(u) at scattered abscissae, so the right summary is a fixed-knot Nadaraya-Watson quadrature, which
carries no n-dependent location shift.
"""
import torch
from torch import nn

from core import config


class ChiSetEncoder(nn.Module):
    """(B, CHI_ELEM_W*k_pad) padded probe set -> (B, out_dim) permutation-invariant embedding."""

    def __init__(self, k_pad: int, u_mid: float, u_half: float,
                 phi_dim: int = None, bin_dim: int = None,
                 rho_hidden: int = None, out_dim: int = None):
        super().__init__()
        phi_dim = config.CHI_PHI_DIM if phi_dim is None else phi_dim
        bin_dim = config.CHI_BIN_DIM if bin_dim is None else bin_dim
        rho_hidden = config.CHI_RHO_HIDDEN if rho_hidden is None else rho_hidden
        self.k_pad = int(k_pad)
        self.out_dim = config.CHI_SET_OUT if out_dim is None else out_dim
        # Read by reparam.posterior_mode tier 2 to decode a posterior's mode off the trained net.
        self.forcing_dim = config.CHI_ELEM_W * self.k_pad
        self.chi_k_pad = self.k_pad
        self.chi_layout = config.CHI_LAYOUT

        # Per-CHANNEL statistics over the 5 real channels (mask excluded). Channel 0 (u) is fixed from
        # the BAND rather than fitted, so the frequency coordinate means the same thing in every run
        # and the load path can compare it.
        self.register_buffer("elem_mean", torch.tensor([u_mid, 0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("elem_std", torch.tensor([u_half, 1.0, 1.0, 1.0, 1.0]))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.uint8))
        self.register_buffer("knots", torch.tensor(config.CHI_KNOTS))

        self.phi = nn.Sequential(
            nn.Linear(5, phi_dim), nn.LayerNorm(phi_dim), nn.GELU(),
            nn.Linear(phi_dim, phi_dim), nn.LayerNorm(phi_dim), nn.GELU(),
        )
        self.to_bin = nn.Linear(phi_dim, bin_dim)
        n_scalar = 4 + len(config.CHI_KNOTS)
        self.rho = nn.Sequential(
            nn.Linear(phi_dim + bin_dim * len(config.CHI_KNOTS) + n_scalar, rho_hidden),
            nn.LayerNorm(rho_hidden), nn.GELU(),
            nn.Linear(rho_hidden, self.out_dim), nn.LayerNorm(self.out_dim), nn.GELU(),
        )
        # No BatchNorm anywhere: sbi's get_numel runs the net on ONE cpu row at build time, and a
        # single-observation inference must equal the corresponding row of a batched one.

    @torch.no_grad()
    def fit(self, f_block: torch.Tensor) -> None:
        """Fit the per-channel statistics over LIVE probes only, from the full training block."""
        e = f_block.reshape(-1, self.k_pad, config.CHI_ELEM_W)
        live = e[..., 1:5][e[..., 5] > 0.5]                    # pads never enter the statistics
        if live.numel() == 0:
            self.fitted.fill_(1)
            return
        mu, sd = live.mean(0), live.std(0)
        # Pass a near-constant channel THROUGH rather than dividing by ~0. sbi's equivalent clamps at
        # min_std=1e-7, which turns such a channel into a ~1e7 amplifier of its own rounding.
        keep = sd > 1e-6
        self.elem_mean[1:] = torch.where(keep, mu, torch.zeros_like(mu))
        self.elem_std[1:] = torch.where(keep, sd, torch.ones_like(sd))
        self.fitted.fill_(1)

    def pool(self, f: torch.Tensor):
        """(curve, sampling) -- the two halves of the pooled representation, before rho.

        SPLIT DELIBERATELY, and the distinction is the design's central claim:

        * ``curve`` (p_mean, p_bin) describes the SUSCEPTIBILITY. It is what must be stable when the
          same physical chi(omega) is sampled at different probe counts, and it is what the tests
          pin -- a masked mean over probes and a coverage-normalised quadrature both approximate
          functionals of the curve rather than of the sample.
        * ``sampling`` (g) describes HOW MUCH DATA THERE WAS: log1p(n) and the per-knot coverage.
          These are K-DEPENDENT ON PURPOSE. An observation with 2 probes really is less informative
          than one with 12, and a posterior conditioned on it should be wider. Hiding that would not
          make the network K-agnostic, it would make it overconfident on sparse observations.

        So "K-agnostic" means the network can CONSUME any probe count, not that it must return the
        same answer regardless of how many it got.
        """
        e = f.reshape(-1, self.k_pad, config.CHI_ELEM_W)
        m = (e[..., 5] > 0.5).to(e.dtype).unsqueeze(-1)
        z = (e[..., :5] - self.elem_mean) / self.elem_std
        n = m.sum(1)
        n_safe = n.clamp(min=1.0)

        lbar = (z[..., 1:2] * m).sum(1) / n_safe
        lt = z[..., 1:2] - lbar.unsqueeze(1)

        elem = torch.cat([z[..., 0:1], lt, z[..., 2:5]], dim=-1) * m
        h = self.phi(elem) * m
        p_mean = h.sum(1) / n_safe

        hb = self.to_bin(h) * m
        uh = z[..., 0]
        w = torch.exp(-(((uh.unsqueeze(1) - self.knots.view(1, -1, 1)) / config.CHI_KNOT_SIGMA) ** 2))
        w = w * m.squeeze(-1).unsqueeze(1)
        c = w.sum(-1)
        p_bin = ((w.unsqueeze(-1) * hb.unsqueeze(1)).sum(2)
                 / (c + config.CHI_KNOT_SHRINK).unsqueeze(-1)).flatten(1)

        ubar = (uh.unsqueeze(-1) * m).sum(1) / n_safe
        cycbar = (z[..., 4:5] * m).sum(1) / n_safe
        g = torch.cat([torch.log1p(n), lbar, ubar, cycbar, torch.log1p(c)], dim=-1)
        return torch.cat([p_mean, p_bin], dim=-1), g

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError(
                "ChiSetEncoder.forward() before fit_standardization(): chi trains with "
                "z_score_x='none', so this module owns the scaling. Unfitted, the summary block and "
                "the probe channels both reach the flow unscaled and the only symptom is a worse "
                "loss curve, days later.")
        curve, sampling = self.pool(f)
        return self.rho(torch.cat([curve, sampling], dim=-1))
