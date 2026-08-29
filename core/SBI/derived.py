"""Tier-1 physical consistency: f_scale derived from the gating-spring energy.

THE BUG THIS FIXES, AND IT IS A BUG RATHER THAN A NARROWING. The box samples the gating-spring energy
TWICE and never requires the two copies to agree:

  * once through the ND block, as ``N * beta`` (beta is the ND gating-spring energy in units of k_B T);
  * once through the rescale block, as ``f_scale * x_scale`` (a force scale times a length scale, i.e.
    an energy).

Their ratio IS an implied bath temperature, and nothing constrained it. Measured over the 10.24M rows
of the 2026-08-25 run, the implied T spans 14 / 80 / 286 / 939 / 6292 K at the 5/25/50/75/95th
percentiles -- the median is room temperature, so this is a VARIANCE problem, not a bias one -- and
only **2.27%** of rows land in 280-310 K. A multi-day training run spent ~97.7% of its budget on
configurations whose two copies of one energy disagree by orders of magnitude.

    f_scale = N * beta * k_B * T / x_scale

WHY f_scale AND NEVER beta. ``beta`` is a DRIFT parameter and the prior's stability screen operates on
the ND block alone. Deriving beta would make the ND block depend on the rescale block and contaminate
that screen -- every ND group would have to be re-swept. Deriving f_scale confines the change to the
3-D rescale block and leaves all ten ND groups sampled exactly as they are today. The
nondimensionalisation itself is untouched: T is not one of the four scales, ``NadrowskiModel`` never
sees it, and only the prior-sampling layer knows it exists.

HOW IT IS WIRED, and the shape was chosen to keep the prior a product of INDEPENDENTS. The inferred
parameter is ``T``, not ``f_scale``: the bounds file declares ``T in (lo, hi)`` in the column f_scale
used to occupy, so the prior keeps a proper 13-D density (making f_scale a deterministic function of
the other twelve would make that density singular, and SBC/TARP/NPE all need it non-degenerate).
``f_scale`` is then computed from theta at the one seam where theta is split into (nd, rescale) for
simulation, and it lands in the SAME COLUMN -- so every downstream reader of
``rescale_idx["f_scale"]`` is untouched.

⚠ TWO HAZARDS, both real:
  1. The derived f_scale reaches ~1e4 pN, and the chi drive amplitude is ``CHI_F0 * f_scale``. Any
     feasibility guard must read the DERIVED value. This is a real change to the training
     distribution, not a relabelling.
  2. T's posterior will be its prior, because the data says nothing on that axis. Its SBC histogram
     will be flat and VACUOUS. Report T as a fixed input, never as an inferred quantity, or you are
     quoting a credible interval on something you assumed.
"""
import torch

# The inferred parameter that replaces f_scale in the bounds file. A bounds file declaring this in
# the rescale section opts the whole run into the tier-1 constraint; one declaring f_scale does not.
TEMPERATURE_PARAM = "T"

# Which ND parameters carry the gating-spring energy. Named here rather than by column, because a
# bounds file's ORDER is the source of truth for columns and only its NAMES are stable.
_N_CHANNELS_PARAM = "n"
_BETA_PARAM = "beta"


def uses_derived_f_scale(rescale_idx: dict) -> bool:
    """True when this box declares T instead of f_scale, i.e. tier 1 is ON for this run."""
    return TEMPERATURE_PARAM in rescale_idx and "f_scale" not in rescale_idx


def sim_rescale_idx(rescale_idx: dict) -> dict:
    """The rescale index the SIMULATOR sees: T's column renamed to f_scale, in place.

    Derived from the inferred index rather than declared beside it, so the two cannot drift -- there
    is exactly one statement anywhere about which column the derived value occupies.
    """
    if not uses_derived_f_scale(rescale_idx):
        return dict(rescale_idx)
    out = dict(rescale_idx)
    out["f_scale"] = out.pop(TEMPERATURE_PARAM)
    return out


def to_sim_rescale(nd: torch.Tensor, rescale: torch.Tensor, rescale_idx: dict,
                   nd_idx: dict | None = None, k_b_cell: float | None = None) -> torch.Tensor:
    """(nd, rescale-with-T) -> rescale-with-f_scale. A no-op for a box that declares f_scale.

    :param nd: (B, n_nd) physical ND parameters.
    :param rescale: (B, n_rescale) physical rescale parameters, T in f_scale's column.
    :param rescale_idx: the INFERRED index (the one carrying T).
    :param nd_idx: name -> column for the ND block; required when tier 1 is on.
    :param k_b_cell: Boltzmann's constant in CELL units (force x length / K); see
        ``SimConfig.k_b_cell``. Required when tier 1 is on -- there is no safe default, because the
        value depends on what the units file declares.
    """
    if not uses_derived_f_scale(rescale_idx):
        return rescale
    if nd_idx is None or k_b_cell is None:
        # Loud, because the silent alternative is simulating with a TEMPERATURE in newtons. A caller
        # that has not been taught about tier 1 must stop, not guess.
        raise ValueError(
            "This box declares 'T' instead of 'f_scale' (tier-1 physical consistency), "
            "so f_scale must be DERIVED as N*beta*k_B*T/x_scale -- but nd_idx and/or k_b_cell were "
            "not supplied, so it cannot be. Pass both, or use a bounds file that declares f_scale.")
    for name in (_N_CHANNELS_PARAM, _BETA_PARAM):
        if name not in nd_idx:
            raise ValueError(
                f"The tier-1 constraint needs the ND parameter '{name}' (f_scale = N*beta*k_B*T/"
                f"x_scale) and this model's ND block does not declare it: {sorted(nd_idx)}. Tier 1 is "
                f"specific to the Nadrowski gating-spring parameterisation.")
    n_ch = nd[:, nd_idx[_N_CHANNELS_PARAM]]
    beta = nd[:, nd_idx[_BETA_PARAM]]
    temp_k = rescale[:, rescale_idx[TEMPERATURE_PARAM]]
    x_scale = rescale[:, rescale_idx["x_scale"]]
    f_scale = n_ch * beta * k_b_cell * temp_k / x_scale
    out = rescale.clone()
    out[:, rescale_idx[TEMPERATURE_PARAM]] = f_scale.to(out.dtype)
    return out


def implied_temperature(nd: torch.Tensor, rescale: torch.Tensor, rescale_idx: dict,
                        nd_idx: dict, k_b_cell: float) -> torch.Tensor:
    """The bath temperature an UNCONSTRAINED (f_scale, x_scale, N, beta) draw implies, in K.

    The inverse of the relation above, and the measurement behind the 2.27% figure. Kept here beside
    the constraint so the diagnostic and the fix cannot disagree about the formula.
    """
    n_ch = nd[:, nd_idx[_N_CHANNELS_PARAM]]
    beta = nd[:, nd_idx[_BETA_PARAM]]
    f_scale = rescale[:, rescale_idx["f_scale"]]
    x_scale = rescale[:, rescale_idx["x_scale"]]
    return f_scale * x_scale / (n_ch * beta * k_b_cell).clamp(min=1e-30)


def describe_derived_f_scale(nd_samples: torch.Tensor, rescale_samples: torch.Tensor,
                             rescale_idx: dict, nd_idx: dict, k_b_cell: float,
                             chi_f0: float | None = None) -> str:
    """What force scale does this prior actually imply? A PREFLIGHT line, not a check.

    HAZARD 1 OF SECTION 11.5, answered the way the chi banner answers its own: by putting the number
    on screen before the run rather than by adding a threshold nobody chose. Under tier 1 f_scale is
    no longer bounded by a declared box -- it follows from (N, beta, x_scale, T) -- and on the master
    box that range reaches ~1.2e4 pN at the corner, against the (1, 1000) the retired box declared.
    The chi drive is ``CHI_F0 * f_scale``, so that corner drives at ~1.9e3 pN. Whether that is
    physically reasonable is a judgement about the preparation, which is exactly why this reports
    rather than refuses.
    """
    f = to_sim_rescale(nd_samples, rescale_samples, rescale_idx,
                       nd_idx, k_b_cell)[:, rescale_idx[TEMPERATURE_PARAM]]
    q = [float(torch.quantile(f.double(), p)) for p in (0.01, 0.5, 0.99)]
    line = (f"[tier1] f_scale is DERIVED as N*beta*k_B*T/x_scale: implied range over the prior "
            f"p1/p50/p99 = {q[0]:.3g} / {q[1]:.3g} / {q[2]:.3g} (cell force units), "
            f"min {float(f.min()):.3g} max {float(f.max()):.3g}")
    if chi_f0:
        line += (f"\n[tier1] chi drive amplitude = CHI_F0 * f_scale = "
                 f"{chi_f0 * q[0]:.3g} / {chi_f0 * q[1]:.3g} / {chi_f0 * q[2]:.3g} at those quantiles")
    return line
