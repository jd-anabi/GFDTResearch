from torch import nn
import torch

from core import config


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
        encoder, and per-column for the summary block, both fitted over real data only.
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
                self.register_buffer("sum_mean", torch.zeros(input_dim))
                self.register_buffer("sum_std", torch.ones(input_dim))
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

    @torch.no_grad()
    def fit_standardization(self, x: torch.Tensor) -> None:
        """Fit both standardizers from the post-filter training tensor. chi set mode only."""
        if not self.owns_standardization:
            return
        s = x[:, :self.input_dim]
        mu, sd = s.mean(0), s.std(0)
        # Pass a near-constant column THROUGH rather than dividing by ~0. sbi's equivalent clamps at
        # min_std=1e-7, which turns such a column into a ~1e7 amplifier of its own rounding -- and
        # under chi, Group G's 11 columns are identically zero by construction.
        keep = sd > 1e-6
        self.sum_mean.copy_(torch.where(keep, mu, torch.zeros_like(mu)))
        self.sum_std.copy_(torch.where(keep, sd, torch.ones_like(sd)))
        self.forcing_net.fit(x[:, self.input_dim:])

    @property
    def standardization_fitted(self) -> bool:
        return bool(getattr(self.forcing_net, "fitted", torch.ones(()))) if self.owns_standardization else True

    def forward(self, x):
        if self.conditioned:
            s = x[:, :self.input_dim]
            f = x[:, self.input_dim:]
            if self.owns_standardization:
                s = (s - self.sum_mean) / self.sum_std
            return self.merge_net(torch.cat([
                self.summary_net(s),
                self.forcing_net(f)
            ], dim=-1))
        else:
            return self.output_net(self.summary_net(x))