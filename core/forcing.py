"""Time-driven external forcing tensors, shared by the SBI pipeline and user-defined models.

``build_nondim_force_tensor`` generalizes the original ``pipeline.build_nondim_sin_force_tensor``
(which now delegates here with kind="sin" -- its numerical behaviour is pinned by a golden test) to
four carrier shapes. Every kind follows the same recipe: build F_dim on the dimensional time grid,
then nondimensionalize as F_nd = (F_dim - f_offset) / f_scale with the identical rescale logic
(f_scale/f_offset from the rescale block if present, else Hopf-style f_scale = x_scale / t_scale).

``build_user_force_tensor`` assembles the per-variable force tensor for a user model: ONE row per
state variable, zeros where unforced -- the UserModel adds force[:, j, t] to variable j.

Kept light on purpose (torch/numpy/helpers only, no sbi import) so the GUI and headless tests can
import it without pulling the SBI stack.
"""
import numpy as np
import torch

from core.Helpers import helpers

FORCE_KINDS = ("sin", "step", "triangular", "exponential")

# Force channels each BUILT-IN drift actually indexes (see core/Models/*.f_pure). This is a property
# of the MODEL, not of the cell's forcing params: HopfModel reads force_step[:, 1] unconditionally
# (hopf_model.py:15, :49), so a driveless Hopf config still needs 2 channels even though no "amp_y"
# is declared. Nadrowski (nadrowski_model.py:18, :68) and BP (bp_model.py:50) read channel 0 only.
_BUILTIN_FORCE_CHANNELS = {"nadrowski": 1, "bp": 1, "hopf": 2}

# Forcing parameter names per kind (the <name>_<var> suffix convention is applied by the caller).
FORCING_PARAM_NAMES = {
    "sin":         ("amp", "freq", "phase", "offset"),
    "triangular":  ("amp", "freq", "phase", "offset"),
    "step":        ("amp", "t0", "offset"),
    "exponential": ("amp", "tau", "offset"),
}


def zero_force(batch: int, n_channels: int, n_steps: int, dtype, device) -> torch.Tensor:
    """A driveless force tensor that costs ``batch x n_channels`` elements instead of ``x n_steps``.

    Every consumer only ever READS ``force[:, ch, t]``, so a stride-0 expansion along time is
    indistinguishable from a materialised block of zeros -- verified bit-identical end-to-end through
    ``gen_obs`` (identical checksum, peak 2.29 -> 1.82 GiB). At the production geometry the
    materialised version is ~2.3 GiB per spontaneous run, allocated and zeroed once per training
    batch and once per Fisher evaluation, to hold nothing but zeros.

    ⚠ The result is a VIEW with a zero stride: it must never be written into. Every hand-built force
    that IS written (FDT's ``campaigns``/``sanity`` cosine drives) builds its own real tensor and does
    not come through here. Slicing (``[s:e]``, ``[:, :, a:b]``) is fine and stays expanded.
    """
    return torch.zeros((batch, n_channels, 1), dtype=dtype, device=device).expand(batch, n_channels, n_steps)


def n_force_channels(model: str, forcing_idx: dict | None = None, n_vars: int | None = None) -> int:
    """
    Number of force channels the model's simulator expects -- the single source of truth for the
    width of a zero-force (or any hand-built) force tensor.

    This matters for memory, not just correctness: the SBI pipeline used to build its zero-force
    tensors ``n_vars`` wide, which for Nadrowski (n_vars=3) and BP (n_vars=5) over-allocates the
    single largest tensor in a training batch by 3x and 5x respectively.

    The channel count is a property of the MODEL's drift, not of the cell's declared forcing params:
    a driveless Hopf config declares no "amp_y" but HopfModel still indexes ``force_step[:, 1]``.
    Unknown built-ins fall back to ``n_vars``, which is always wide enough.

    :param model: model name (built-in or user-defined).
    :param forcing_idx: forcing-param name -> column map. Only consulted to widen a built-in to the
                        legacy dual-channel Hopf convention when "amp_y" is present.
    :param n_vars: number of state variables. Required for user models (which index force[:, j, t]
                   per variable, see build_user_force_tensor) and for the unknown-built-in fallback.
    :return: channel count for a (batch, n_channels, T) force tensor.
    """
    from core import registry            # local: forcing.py stays importable without the registry

    if registry.is_user_model(model):
        if n_vars is None:
            raise ValueError(f"n_vars is required to size the force tensor for user model '{model}'.")
        return n_vars
    n = _BUILTIN_FORCE_CHANNELS.get(model.lower())
    if n is None:                        # unknown built-in: keep the conservative full-width tensor
        if n_vars is None:
            raise ValueError(f"Unknown model '{model}' and no n_vars given to size the force tensor.")
        return n_vars
    if forcing_idx and "amp_y" in forcing_idx:
        n = max(n, 2)                    # legacy dual-channel sinusoidal convention (ND Hopf)
    return n


def build_nondim_force_tensor(
    forcing_params: torch.Tensor,
    t_nd: torch.Tensor,
    rescale_params: torch.Tensor,
    forcing_idx: dict,
    rescale_idx: dict,
    kind: str = "sin",
    *,
    exp_sign: float = 1.0,
    name_suffix: str = "",
) -> torch.Tensor:
    """
    Build a batch of non-dimensional force tensors for one carrier ``kind``.

    Carriers (all built in dimensional time t_dim, then nondimensionalized):
        sin         : amp * sin(2*pi*freq*t_dim + phase) + offset
        triangular  : amp * (2/pi) * asin(sin(2*pi*freq*t_dim + phase)) + offset
        step        : offset + amp * (t_dim >= t0)
        exponential : amp * exp(exp_sign * t_dim / tau) + offset      (exp_sign = +1 grow / -1 decay)

    :param forcing_params: forcing parameter values, shape (batch, n_forcing).
    :param t_nd: non-dimensional time vector, shape (T,).
    :param rescale_params: rescaling parameter values, shape (batch, n_rescale).
    :param forcing_idx: maps forcing param names to columns of forcing_params. Parameter names are
                        looked up with ``name_suffix`` appended (user models name theirs amp_<var> etc.).
                        For kind="sin" with no suffix, an "amp_y" entry builds the legacy second (Hopf)
                        channel sharing freq/phase/offset.
    :param rescale_idx: maps rescale param names to columns of rescale_params. If "f_scale" is absent,
                        f_scale = x_scale / t_scale and f_offset = 0 (Hopf-style nondim).
    :param kind: one of FORCE_KINDS.
    :param exp_sign: +1.0 or -1.0; the exponential's grow/decay sign (spec metadata, not a parameter).
    :param name_suffix: appended to every forcing param name before the forcing_idx lookup.
    :return: non-dimensional force tensor, shape (batch, n_channels, T); n_channels = 2 only for the
             legacy un-suffixed sin + "amp_y" case, else 1.
    """
    if kind not in FORCE_KINDS:
        raise ValueError(f"Unknown forcing kind '{kind}'. Valid: {FORCE_KINDS}.")

    def fp(name: str) -> torch.Tensor:
        key = name + name_suffix
        if key not in forcing_idx:
            raise KeyError(f"Forcing parameter '{key}' missing for kind '{kind}'.")
        return forcing_params[:, forcing_idx[key]].unsqueeze(1)          # (batch, 1)

    # rescale params as (batch, 1) -- identical logic to the original sinusoidal builder
    t_scale = rescale_params[:, rescale_idx["t_scale"]].unsqueeze(1)
    t_offset = rescale_params[:, rescale_idx["t_offset"]].unsqueeze(1) if "t_offset" in rescale_idx else 0.0
    if "f_scale" in rescale_idx:
        f_scale = rescale_params[:, rescale_idx["f_scale"]].unsqueeze(1)
        f_offset = (rescale_params[:, rescale_idx["f_offset"]].unsqueeze(1)
                    if "f_offset" in rescale_idx else torch.zeros_like(f_scale))
    else:
        # Hopf-style nondim: F_ND = F_dim / (l * omega_0) -> f_scale = x_scale / t_scale, f_offset = 0.
        x_scale = rescale_params[:, rescale_idx["x_scale"]].unsqueeze(1)
        f_scale = x_scale / t_scale
        f_offset = torch.zeros_like(f_scale)

    # nd -> dim time; (T,) -> (1, T) for broadcasting against (batch, 1)
    t_dim = helpers.rescale(t_nd.unsqueeze(0), t_scale, t_offset)        # (batch, T)

    amp = fp("amp")
    offset = fp("offset")

    if kind in ("sin", "triangular"):
        freq = fp("freq")
        phase = fp("phase")
        sin_term = torch.sin(2 * np.pi * freq * t_dim + phase)           # (batch, T)
        if kind == "sin":
            carrier = sin_term
        else:
            carrier = (2.0 / np.pi) * torch.asin(sin_term)
        f_x_nd = (amp * carrier + offset - f_offset) / f_scale
        if kind == "sin" and not name_suffix and "amp_y" in forcing_idx:
            # Legacy second channel (ND Hopf): shares freq/phase/offset/f_scale/f_offset, own amplitude.
            amp_y = forcing_params[:, forcing_idx["amp_y"]].unsqueeze(1)
            f_y_nd = (amp_y * carrier + offset - f_offset) / f_scale
            return torch.stack([f_x_nd, f_y_nd], dim=1)                  # (batch, 2, T)
        return f_x_nd.unsqueeze(1)                                       # (batch, 1, T)

    if kind == "step":
        t0 = fp("t0")
        f_dim = offset + amp * (t_dim >= t0).to(t_dim.dtype)
    else:  # exponential
        tau = fp("tau")
        f_dim = amp * torch.exp(exp_sign * t_dim / tau) + offset
    return ((f_dim - f_offset) / f_scale).unsqueeze(1)                   # (batch, 1, T)


def build_user_force_tensor(
    spec,
    forcing_params: torch.Tensor,
    t_nd: torch.Tensor,
    rescale_params: torch.Tensor,
    forcing_idx: dict,
    rescale_idx: dict,
) -> torch.Tensor:
    """
    The (batch, n_vars, T) force tensor for a user model spec (registry.ModelSpec): one row per state
    variable in declared order, built from that variable's forcing entry (params named <name>_<var>),
    zeros for unforced variables. All-zeros when nothing is forced.
    """
    batch = rescale_params.shape[0]
    n_t = t_nd.shape[0]
    zeros = None
    rows = []
    for v in spec.variables:
        forcing = v.get("forcing") or None
        if forcing:
            row = build_nondim_force_tensor(
                forcing_params, t_nd, rescale_params, forcing_idx, rescale_idx,
                kind=forcing["kind"], exp_sign=float(forcing.get("sign", 1.0)),
                name_suffix=f"_{v['name']}")
            rows.append(row[:, 0, :])
        else:
            if zeros is None:
                zeros = torch.zeros((batch, n_t), dtype=t_nd.dtype, device=t_nd.device)
            rows.append(zeros)
    return torch.stack(rows, dim=1)
