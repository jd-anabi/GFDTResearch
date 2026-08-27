"""Single source of truth for plot/GUI variable-name symbols and unit annotations.

Two audiences, two renderers:
  * PLOTS (matplotlib) use true LaTeX mathtext ($...$) -- `axis_label` / `rescale_axis_label` / `SWEEP_LATEX`.
  * The Qt GUI cannot render LaTeX in a QLabel, so config-option labels use Qt rich text (Unicode Greek +
    <sub>/<sup>) -- `pretty_gui` / `gui_forcing_label`. Qt's AutoText auto-detects the HTML, and using no
    explicit color keeps it dark-mode safe (the label inherits palette text colour).

Pure strings only: this module imports nothing from the project (so config.py / help_badge.py can import
it without a cycle).
"""

# ── matplotlib LaTeX (bare symbols; $...$ added by the helpers) ───────────────────────────────────
RESCALE_LATEX = {
    "x_scale": r"x_{\mathrm{scale}}", "t_scale": r"t_{\mathrm{scale}}", "f_scale": r"f_{\mathrm{scale}}",
    "x_offset": r"x_{\mathrm{off}}",  "t_offset": r"t_{\mathrm{off}}",  "f_offset": r"f_{\mathrm{off}}",
    # Tier-1 boxes declare T in f_scale's place (section 11.5). It is a bath temperature in
    # KELVIN, which is not one of the four cell-file unit tokens, so _RESCALE_KIND deliberately
    # has no entry for it and the label degrades to a bare symbol rather than inventing a unit.
    "T": r"T",
}
_RESCALE_KIND = {"x_scale": "length", "x_offset": "length", "t_scale": "time",
                 "t_offset": "time", "f_scale": "force", "f_offset": "force"}

# Full LaTeX (with $) for the FDT sweep parameters -- centralizes the literals in cross_validation.py.
SWEEP_LATEX = {"s": r"$S$", "temp": r"$T_a/T$"}


def axis_label(symbol_latex: str, unit: str | None = None) -> str:
    """A matplotlib axis label ``"$sym$ (unit)"`` -- or ``"$sym$ (ND)"`` for a dimensionless axis."""
    return f"${symbol_latex}$ ({unit if unit else 'ND'})"


def rescale_axis_label(name: str, *, length_unit: str | None = None, time_unit: str | None = None,
                       force_unit: str | None = None) -> str:
    """A LaTeX label (with unit) for a rescale parameter, for ``SimConfig.inferred_labels``.

    Scales carry ``unit/ND`` (e.g. nm per ND unit), offsets carry ``unit``; if the symbol is unknown or
    the cell declares no matching unit token, degrade to a bare ``$sym$`` rather than guess.
    """
    sym = RESCALE_LATEX.get(name)
    if sym is None:
        return f"${name.replace('_', r'\_')}$"      # escape '_' so mathtext doesn't read it as a subscript
    unit = {"length": length_unit, "time": time_unit,
            "force": force_unit}.get(_RESCALE_KIND.get(name))
    if unit is None:
        return f"${sym}$"
    return f"${sym}$ ({unit}/ND)" if name.endswith("scale") else f"${sym}$ ({unit})"


# ── Qt GUI rich text (Unicode + <sub>) ────────────────────────────────────────────────────────────
# Keyed by the EXACT label strings the panels pass to add_help_row (grep add_help_row to keep in sync).
# Non-math labels (Model, Cell, Bounds, Preset, "Steps / frame", "Max FPS", …) are passthrough.
GUI_HTML = {
    "T_obs (s)":                 "T<sub>obs</sub> (s)",
    "n_freqs":                   "n<sub>freqs</sub>",
    "ensemble_M":                "M<sub>ensemble</sub>",
    "freqs_per_batch":           "freqs / batch",
    "F0 (ND forcing amplitude)": "F<sub>0</sub> (ND forcing amplitude)",
    "S grid  (T_a/T = 1)":       "S grid  (T<sub>a</sub>/T = 1)",
    "T_a/T grid  (S = 0)":       "T<sub>a</sub>/T grid  (S = 0)",
}

# Pretty symbols for the (dynamically built) forcing-parameter labels in the Infer tab.
GUI_FORCING = {"amp": "A", "amp_y": "A<sub>y</sub>", "freq": "f", "phase": "φ", "offset": "offset"}


def pretty_gui(text: str) -> str:
    """The Qt-rich-text form of a config-option label; passthrough for labels that need no math."""
    return GUI_HTML.get(text, text)


def gui_forcing_label(name: str, unit: str = "") -> str:
    """Pretty label for a forcing parameter, e.g. ``gui_forcing_label("phase", "rad") -> "φ (rad)"``."""
    sym = GUI_FORCING.get(name, name)
    return f"{sym} ({unit})" if unit else sym
