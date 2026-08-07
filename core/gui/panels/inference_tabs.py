"""Parameter Inference, split into five tabs over ONE shared SbiSession owned by the InferenceScreen.

    Config -> Prior -> Posterior -> Validate -> Infer

Config records the MODEL-level choices as a ConfigDraft; the PRIOR tab picks the bounds file and turns
that draft into the SimConfig, because the bounds file declares the inferred parameter set (and hence the
observation mode), so a config cannot exist before it.

Each tab is its own BasePanel (its own FigureStack/ProgressPane/LogPane), so dispatch()/cancel/the
figure-sink/progress plumbing are reused verbatim; BasePanel._running is class-level, so the tabs can
never run concurrently. The screen owns the SbiSession and greys tabs via setTabEnabled; every tab
reads/writes the session through ``self._screen`` and calls ``self._screen.refresh_gates()`` after a
stage completes.
"""
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from core import cli, config, orchestrator, registry
from core.Helpers import file_manager, labels, visualizers
from core.config import (VALID_MODELS, VALID_LABELS, BOUNDS_PATH, CELL_PATH, PRIOR_PATH,
                         POSTERIOR_PATH, T_MIN_EXP_S, CHI_K_MAX)

from .base_panel import BasePanel
from .. import settings
from ..session import ConfigDraft
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.help_badge import add_help_row, with_badge
from ..widgets.labeled_inputs import FloatField, IntField, PathField
from ..widgets.param_grid import BoundsGrid, ValuesGrid
from ..widgets.source_toggle import SourceToggle
from ..widgets.field_row import LabeledFieldRow
from ..widgets.forms import make_form
from ..widgets.adaptive_stack import AdaptiveStack

# Help text shown by the "?" badge next to each option. Drafted from the code/science; user reviews.
HELP = {
    "model": "Which model to fit. NADROWSKI is the state-dependent-drift model the pipeline is tuned "
             "for; HOPF and BP are alternatives. User-defined models are inferable too if they have no "
             "forcing (spontaneous dynamics) and at least one parameter — but their calibration is not "
             "pre-tuned, so validate SBC/TARP per model.",
    "bounds": "Parameter-bounds file (Resources/Bounds/<model>) defining the inference box: which "
              "parameters are inferred and the prior range of each.",
    "cell": "A cell file (Resources/Cells/<model>) whose parameter values are the ground truth — the "
            "simulator uses them to generate a synthetic observation.",
    "tobs": "Observation duration in seconds. Longer traces carry more information but cost more to "
            "simulate.",
    "prior": "Load a saved prior (.pt), or choose “(from scratch)” to construct a new "
             "stability-screened parameter prior.",
    "posterior": "Load a trained posterior (.pt), or “(from scratch)” to train a new one. Training "
                 "from scratch needs a prior; loading an existing posterior does not.",
    "infer_mode": "Simulated: infer on a synthetic observation from a cell’s ground truth. "
                  "Experimental: infer on your own recording (a driven spontaneous+forced pair, or — "
                  "for a no-forcing model — a single passive recording).",
    "spont": "Path to the recorded spontaneous/passive (undriven) trace (.csv or .npy; last column "
             "= values).",
    "forced": "Path to the recorded forced (driven) hair-bundle trace (.csv or .npy; last column = "
              "values).",
    "forcing": "The value of this sinusoidal-drive parameter used in the forced recording, in the "
               "shown units.",
    "chi_mode": "Multi-frequency susceptibility χ(ω). Instead of conditioning on ONE drive, each "
                "observation is a passive recording plus K single-tone driven recordings, and the "
                "conditioning carries the χ(ω) curve. This is the only lever on the information "
                "ceiling: a single passive trace sees only the PRODUCTS D·A_nd and (λ/k)·τ, whereas "
                "the shape of χ(ω) separates κ, λ, x_scale and t_scale individually. Costs about "
                "(K+1)/2× the training time. Training and inference must both use it.",
    "chi_k": "How many drive frequencies THIS observation is measured at, and therefore how many "
             "forced recordings an experiment must supply. More probes resolve the curve better but "
             "cost linearly more simulation. It does NOT have to match the posterior: the network "
             "conditions on a probe SET, so it accepts any count up to the slot capacity below.",
    "chi_k_pad": "Probe SLOTS the network reserves — its capacity, not a probe count. Training draws "
                 "probe counts from 2 up to this, so the encoder learns to handle any of them. It is "
                 "FROZEN into every posterior trained with it (it fixes the input width), so raising "
                 "it later means retraining; pick generously. Costs only input columns — the "
                 "encoder's parameter count does not depend on it.",
    "chi_f0": "Non-dimensional drive amplitude for every χ probe. χ = response/drive is independent of "
              "amplitude in the linear regime, so this only needs to be small enough to stay linear "
              "(≲0.1) and large enough for the lock-in to beat the noise.",
    "chi_max_cycles": "Longest lock-in, in drive cycles, used for any one probe. A longer lock-in is "
                      "NOT a better one here: past roughly 30 cycles χ stops being reproducible at "
                      "fixed parameters, and the extra recording adds noise rather than signal. "
                      "Recordings longer than this are truncated, not rejected — the leading part is "
                      "exactly what the network was trained on. Frozen into any posterior trained "
                      "with it, so changing it means retraining.",
    "chi_range": "The K probe frequencies are placed log-spaced across this range, as MULTIPLES of each "
                 "observation's own measured spontaneous peak Ω₀ — so the probes track the resonance "
                 "wherever t_scale puts it, instead of sitting at fixed absolute frequencies.",
    "chi_passive": "The passive (undriven) recording. Its power spectrum sets Ω₀, which anchors the "
                   "frequency of every driven recording below.",
    "chi_forced": "The driven recording for this multiple of Ω₀ (.csv or .npy; last column = values).",
    "chi_f0_si": "The physical drive amplitude used for the forced recordings. χ cancels the amplitude "
                 "in the linear regime, so this only sets the lock-in normalisation.",
    "bounds_source": "Pick a bounds file, or edit the numbers directly. Direct entry starts FROM the "
                     "selected file, because the parameter names and their order are fixed by the model "
                     "(simulators bind parameter columns by position) — only the numbers are yours to "
                     "change. Switching to “Edit values” reloads from whichever file is selected.",
    "cell_source": "Pick a cell file, or edit its ground-truth values directly. As with bounds, direct "
                   "entry starts from the selected file and lets you change only the numbers.",
    "units": "The units the numbers in your bounds and cell files are written in. This DECLARES what "
             "those numbers mean — it never converts them, so changing it re-interprets your files "
             "rather than rescaling them. Plot axes and unit labels follow this choice. Frequency is "
             "special: the pipeline consumes a drive frequency as inverse cell TIME, so for an `ms` cell "
             "the frequency unit must be kHz — a mismatch is reported when the config is built.",
    "units_mode": "Take the units from the model's units file (Resources/Units/<model>/units.txt), or "
                  "type them directly as space-separated tokens.",
    "units_text": "Space-separated unit tokens, one per physical dimension — e.g. “nm ms pN kHz”. Any "
                  "unit pint understands works; they are matched to quantities by DIMENSION, not order.",
    "reparam_rotate": "Rotate the flow's latent coordinate into the simulation-Fisher eigenbasis, so a "
                      "strongly correlated posterior (κ↔x_scale, λ↔t_scale) becomes axis-aligned and the "
                      "flow can calibrate it. The rotation is orthogonal, so it adds and removes no "
                      "information — off is exactly the plain pipeline. Cost: computing the rotation runs "
                      "extra simulations at several operating points BEFORE training starts. Results so "
                      "far have been a redistribution rather than a clean win, so it is worth comparing "
                      "SBC with it on and off. Available in all three observation modes; under χ(ω) the "
                      "Fisher is built over the χ features, which costs (K+1)/2× a forced-mode rotation.",
}


class _ChiRangeRow(LabeledFieldRow):
    """lo / hi multipliers of the measured spontaneous peak Ω₀ bounding the χ(ω) probe grid."""

    def __init__(self, lo: float, hi: float, parent=None):
        self.lo, self.hi = FloatField(lo), FloatField(hi)
        super().__init__((("×Ω₀ from", self.lo), ("to", self.hi)), parent=parent)

    def value(self) -> tuple:
        return self.values()


# ── worker-callable runners (module-level so a Worker can call them with an injected fig_sink) ─────
def _run_simulated_inference(cfg, posterior, cell_path, T_obs_s, *, gt_dicts=None, inferred_prior=None,
                             force_prior=None, fig_sink=None):
    """Mirror orchestrator.run's simulated branch: inject GT + T_obs, simulate, show GT trace + infer.

    ``gt_dicts`` is the hand-entered alternative to ``cell_path``: an (inits, params, rescale, forcing)
    tuple in parse_values_file's shape. It goes through the SAME inject_ground_truth validation, so
    typed values are bounds-checked exactly like a file's."""
    ignored = (cfg.inject_ground_truth(*gt_dicts) if gt_dicts is not None
               else cli.load_and_validate_gt(cfg, cell_path))
    if ignored:
        print(f"Note: the bounds file does not declare {', '.join(ignored)} — those cell values were "
              f"ignored (the bounds file defines the inferred set).")
    cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
    # Is this observation actually in the region the network trained on? Bounds-checking cannot tell.
    if inferred_prior is not None:
        for msg in orchestrator.check_observation_in_distribution(cfg, inferred_prior, force_prior):
            print(f"WARNING: {msg}")
    x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)
    visualizers.plot(t_dim.squeeze(0).cpu().detach().numpy(), x_dim[0, :].cpu().detach().numpy(),
                     title="Ground-truth trace",
                     labels=(labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)),
                     sink=fig_sink)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True, fig_sink=fig_sink)


def _run_experimental_inference(cfg, posterior, spont_path, forced_path, T_obs_s, forcing_si, *, fig_sink=None):
    """Mirror orchestrator.run's experimental branch."""
    x_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
    x_forced = file_manager.load_experimental_data(forced_path, dtype=cfg.hw.dtype)
    obs_stats, obs_data, t_dim = orchestrator.build_experiment_obs(cfg, x_spont, x_forced, T_obs_s, forcing_si)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False, fig_sink=fig_sink)


def _run_experimental_inference_chi(cfg, posterior, spont_path, forced_paths, T_obs_s, F0_si,
                                    *, fig_sink=None):
    """chi(omega) experimental inference: ONE passive recording (which sets Omega_0) plus K single-tone
    forced recordings, the k-th driven at the k-th multiple of Omega_0."""
    x_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
    x_forced = [file_manager.load_experimental_data(p, dtype=cfg.hw.dtype) for p in forced_paths]
    obs_stats, obs_data, t_dim = orchestrator.build_experiment_obs_chi(
        cfg, x_spont, x_forced, T_obs_s, F0_si)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False,
                                     fig_sink=fig_sink)


def _run_experimental_inference_spontaneous(cfg, posterior, path, T_obs_s, *, fig_sink=None):
    """Passive-recording inference for a no-forcing model: a single unforced recording, no drive."""
    x_obs = file_manager.load_experimental_data(path, dtype=cfg.hw.dtype)
    obs_stats, obs_data, t_dim = orchestrator.build_experiment_obs_spontaneous(cfg, x_obs, T_obs_s)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False, fig_sink=fig_sink)


class _StagePanel(BasePanel):
    """Common base for the five inference tabs: holds a back-reference to the owning InferenceScreen and
    reads/writes the shared session through it (never caching the session object, which Config replaces
    wholesale on each build)."""

    def __init__(self, screen, parent=None):
        super().__init__(parent)
        self._screen = screen

    @property
    def session(self):
        return self._screen.session


# ── 1. Config ─────────────────────────────────────────────────────────────────
class ConfigPanel(_StagePanel):
    """Tab 1. Records the MODEL-level choices as a ``ConfigDraft`` -- it does NOT build the SimConfig.

    A SimConfig cannot exist without a bounds file, because the bounds file declares which parameters
    are inferred and hence the observation mode; the Prior tab owns that. Validates the chi knobs
    here (2 <= K <= CHI_K_MAX, F0 > 0, 0 < lo < hi), since this is where they are entered.

    Persists (group "inference_config"): model, units source/text, and the chi knobs.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Config")
        form = make_form(box)
        self.model_combo = QComboBox()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.btn_config = QPushButton("Apply model & options")
        self.btn_config.setProperty("accent", True)       # primary CTA (Fluent accent)
        self.btn_config.clicked.connect(self._build_config)
        add_help_row(form, "Model", self.model_combo, HELP["model"])

        # Units DECLARE what the numbers in the bounds/cell files mean; they never convert them.
        self.units_default = QLabel("—")
        self.units_default.setProperty("type", "caption")
        self.units_text = QLineEdit()
        self.units_text.setPlaceholderText("e.g. nm ms pN kHz")
        self.units_toggle = SourceToggle(self.units_default, self.units_text,
                                         file_label="Model's units file", direct_label="Type units")
        add_help_row(form, "Units", self.units_toggle, HELP["units"])

        # chi(omega) mode. Captured onto the config at build time (SimConfig carries K/F0/range), so a
        # posterior is self-describing and toggling this later cannot reinterpret an existing run.
        self.chi_check = QCheckBox("Multi-frequency χ(ω) conditioning")
        self.chi_k = IntField(config.CHI_N_FREQS)
        self.chi_f0 = FloatField(config.CHI_F0)
        self.chi_range = _ChiRangeRow(*config.CHI_FREQ_BOUNDS)
        self.chi_pad = IntField(config.CHI_K_PAD)
        self.chi_cycles = FloatField(config.CHI_MAX_CYCLES)
        self.chi_check.toggled.connect(lambda _on: self._sync_chi_enabled())
        form.addRow(with_badge(self.chi_check, HELP["chi_mode"]))
        add_help_row(form, "χ probes per observation", self.chi_k, HELP["chi_k"])
        add_help_row(form, "χ probe slots (capacity)", self.chi_pad, HELP["chi_k_pad"])
        add_help_row(form, "χ drive F₀ (ND)", self.chi_f0, HELP["chi_f0"])
        add_help_row(form, "χ frequency range", self.chi_range, HELP["chi_range"])
        add_help_row(form, "χ lock-in ceiling (cycles)", self.chi_cycles, HELP["chi_max_cycles"])

        self.rot_check = QCheckBox("Decorrelating Fisher rotation")
        form.addRow(with_badge(self.rot_check, HELP["reparam_rotate"]))

        form.addRow(self.btn_config)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())
        self._sync_chi_enabled()

    def _sync_chi_enabled(self):
        """The three χ knobs are meaningless unless χ-mode is on. The rotation is available in ALL
        THREE observation modes -- it used to be greyed out under χ, on the assumption that χ already
        decorrelated what the rotation targets; measured on the master cell, χ leaves k~x_scale at
        0.95 (vs 0.98 forced), so that assumption was wrong and the exclusion is gone."""
        on = self.chi_check.isChecked()
        for w in (self.chi_k, self.chi_pad, self.chi_f0, self.chi_range, self.chi_cycles):
            w.setEnabled(on)
        self.rot_check.setEnabled(True)
        self.rot_check.setToolTip(
            "In χ(ω) mode the Fisher is built over the χ feature set, which costs (K+1)/2× what a "
            "forced-mode rotation does." if on else "")

    def _show_model_units(self, model: str):
        """Reflect the model's declared units, and seed the direct-entry box from them so switching to
        'Type units' starts from the truth rather than an empty field."""
        try:
            tokens = file_manager.parse_units_file(cli.resolve_units_file(model))
        except Exception as e:                         # noqa: BLE001 -- a missing units file is the
            self.units_default.setText(f"(no units file for {model}: {e})")   # model's problem, not a crash
            return
        self.units_default.setText(" ".join(tokens) + f"   —  Resources/Units/{model.lower()}/units.txt")
        if not self.units_text.text().strip():
            self.units_text.setText(" ".join(tokens))

    def _units_override(self):
        """None => use the model's units file; a token tuple => the user typed them."""
        if not self.units_toggle.is_direct():
            return None
        tokens = tuple(self.units_text.text().split())
        return tokens or None

    def _on_model_changed(self, model: str):
        self._show_model_units(model)
        # The bounds picker lives on the PRIOR tab now (bounds are what build the config), and it is
        # repointed via InferenceScreen.on_draft_set when the model is APPLIED -- not on every combo
        # change, so a half-changed selection cannot leave the two tabs disagreeing.
        # User models are inferable only when SBI-eligible: no forcing (spontaneous dynamics) AND at
        # least one ND parameter. Forced / zero-parameter user models stay Simulate-only.
        ineligible_user = registry.is_user_model(model) and not registry.is_sbi_user_model(model)
        self.btn_config.setEnabled(not ineligible_user)
        if ineligible_user:
            reason = ("has external forcing" if registry.user_model_has_forcing(model)
                      else "has no free parameters to infer")
            self.log_pane.append_line(
                f"'{model}' {reason}, so it is Simulate-only. Parameter inference supports "
                "user-defined models with no forcing and at least one parameter.", "warning")

    def _build_config(self):
        model = self.model_combo.currentText()
        if registry.is_user_model(model) and not registry.is_sbi_user_model(model):   # backstop
            self.log_pane.append_line(
                "This user-defined model is Simulate-only (needs no forcing + ≥1 parameter for "
                "inference).", "warning")
            return
        # model_labels, NOT `labels`: this module imports the core.Helpers.labels MODULE at the top
        # and calls labels.axis_label(...) / labels.gui_forcing_label(...) elsewhere in this same
        # file. Binding a local of that name shadowed it for the whole function, so any future line
        # added here that touched the module would raise AttributeError on a list -- a crash sitting
        # one edit away, and invisible until someone made that edit.
        model_labels = VALID_LABELS[VALID_MODELS.index(model)]
        state_dep_drift = registry.state_dep_drift(model)
        chi_on = self.chi_check.isChecked()
        if chi_on:                                   # FloatField/IntField return 0 on unparseable text
            lo, hi = self.chi_range.value()
            # K has an UPPER bound too. Cost is linear in K (each probe is a whole extra simulation
            # per observation, so training and calibration both scale as K+1), and the Infer tab
            # grows one file-picker row per probe frequency -- K=500 would mean 500 rows.
            problem = ("K must be at least 2 to resolve a χ(ω) curve." if self.chi_k.value() < 2 else
                       f"K must be at most {CHI_K_MAX}: every probe frequency is another full "
                       f"simulation per observation, and the Infer tab needs one recording per "
                       f"frequency." if self.chi_k.value() > CHI_K_MAX else
                       "χ drive F₀ must be > 0." if self.chi_f0.value() <= 0 else
                       "χ frequency range must satisfy 0 < from < to." if not (0 < lo < hi) else
                       # Caught here rather than by SimConfig.__post_init__ so it reads as a form
                       # error next to the box, not as a traceback on "Apply model & options".
                       f"χ lock-in ceiling must exceed the {config.CHI_MIN_CYCLES:g}-cycle floor, "
                       f"or every probe is truncated below it and masked."
                       if self.chi_cycles.value() <= config.CHI_MIN_CYCLES else None)
            if problem:
                self.log_pane.append_line(problem, "warning")
                return
        units = self._units_override()
        if self.units_toggle.is_direct():
            if units is None:
                self.log_pane.append_line("Enter at least one unit token, or switch back to the "
                                          "model's units file.", "warning")
                return
            try:                                       # reject unresolvable tokens HERE, not mid-run
                cli._units_to_factors(units)
            except Exception as e:                     # noqa: BLE001
                self.log_pane.append_line(f"Those units are not usable: {e}", "warning")
                return
        draft = ConfigDraft(
            model=model, labels=model_labels, state_dep_drift=state_dep_drift, units_override=units,
            chi_mode=chi_on, chi_n_freqs=self.chi_k.value(), chi_f0=self.chi_f0.value(),
            chi_freq_bounds=self.chi_range.value(), chi_k_pad=self.chi_pad.value(),
            chi_max_cycles=self.chi_cycles.value(),
            reparam_rotate=self.rot_check.isChecked())
        self._screen.new_draft(draft)                # replaces the session + repoints Prior + re-gates
        extras = []
        if chi_on:
            lo, hi = draft.chi_freq_bounds
            extras.append(f"χ(ω) on — {draft.chi_n_freqs} frequencies over {lo:g}–{hi:g}×Ω₀ at ND "
                          f"amplitude {draft.chi_f0:g}, ≤{draft.chi_max_cycles:g} cycles per probe, "
                          f"so expect ~{(draft.chi_n_freqs + 1) / 2:.1f}× the "
                          f"usual training time and train a NEW posterior")
        elif draft.reparam_rotate:
            extras.append("decorrelating Fisher rotation on")
        self.log_pane.append_line(
            f"Model applied: {model}" + (f" ({'; '.join(extras)})" if extras else "")
            + ". Now pick a bounds file on the Prior tab — that builds the config and selects the "
              "observation mode.")
        if registry.is_user_model(model):
            self.log_pane.append_line(
                "Note: user-model inference runs the full pipeline (spontaneous dynamics only), but "
                "calibration is NOT pre-tuned — check the Validate tab's SBC/TARP results for this model.",
                "warning")

    def save_settings(self, qs):
        qs.beginGroup("inference_config")
        qs.setValue("model", self.model_combo.currentText())
        qs.setValue("units_mode", self.units_toggle.key())
        settings.save_field(qs, "units_text", self.units_text)
        settings.set_bool(qs, "chi_mode", self.chi_check.isChecked())
        settings.set_bool(qs, "reparam_rotate", self.rot_check.isChecked())
        settings.save_field(qs, "chi_k", self.chi_k)
        settings.save_field(qs, "chi_k_pad", self.chi_pad)
        settings.save_field(qs, "chi_f0", self.chi_f0)
        settings.save_field(qs, "chi_lo", self.chi_range.lo)
        settings.save_field(qs, "chi_hi", self.chi_range.hi)
        settings.save_field(qs, "chi_max_cycles", self.chi_cycles)
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_config")
        # Explicit _on_model_changed: currentTextChanged won't fire if the value already equals the
        # default. (The bounds picker moved to the Prior tab, which restores its own key in on_draft_set,
        # so the old restore-order trap no longer applies here.)
        self.model_combo.setCurrentText(settings.get_str(qs, "model", self.model_combo.currentText()))
        # units_text BEFORE _on_model_changed: the latter seeds the box from the model's file only when
        # it is empty, so a restored custom value must already be in place to survive.
        settings.restore_field(qs, "units_text", self.units_text)
        self._on_model_changed(self.model_combo.currentText())
        self.units_toggle.restore_key(settings.get_str(qs, "units_mode", "file"))
        self.chi_check.setChecked(settings.get_bool(qs, "chi_mode", False))
        self.rot_check.setChecked(settings.get_bool(qs, "reparam_rotate", config.REPARAM_ROTATE))
        settings.restore_field(qs, "chi_k", self.chi_k)
        settings.restore_field(qs, "chi_k_pad", self.chi_pad)
        settings.restore_field(qs, "chi_f0", self.chi_f0)
        settings.restore_field(qs, "chi_lo", self.chi_range.lo)
        settings.restore_field(qs, "chi_hi", self.chi_range.hi)
        settings.restore_field(qs, "chi_max_cycles", self.chi_cycles)
        qs.endGroup()


class _CellPreviewMixin:
    """Shared cell-picker handling for Simulate + Infer: the cell folder follows the BUILT config's
    model (there is no live model combo in these tabs), so the picker is repointed in on_config_built
    and the saved key is re-applied there (it could not resolve at __init__, before any config)."""

    def _init_cell_picker(self):
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self._saved_cell_key = ""

    def on_config_built(self, cfg):
        self.cell_picker.base_path = CELL_PATH / cfg.model.lower()
        self.cell_picker.refresh()
        self.cell_picker.restore_key(self._saved_cell_key)   # -1 guard leaves default if not in folder


# ── 2. Prior (also picks the BOUNDS file, which is what builds the config) ────
class PriorPanel(_StagePanel):
    """Tab 2. Picks the BOUNDS file -- which is what turns the draft into a real SimConfig -- then
    builds or loads the parameter prior.

    Installs the config IN PLACE (``install_config``), not as a new session: building the prior is
    the first step of the existing session, so replacing it would discard the draft.

    Persists (group "inference_prior"): the bounds and prior picker selections. NOT the bounds grid --
    parameter names and order belong to the model, so hand-entry only ever edits numbers.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        self._saved_bounds_key = ""              # resolved in on_draft_set (needs the chosen model)
        box = QGroupBox("Prior")
        v = QVBoxLayout(box)
        form = make_form()
        self.bounds_picker = ArtifactPicker(BOUNDS_PATH / "nadrowski")
        self.bounds_grid = BoundsGrid()
        self.bounds_source = SourceToggle(self.bounds_picker, self.bounds_grid,
                                          file_label="Use file", direct_label="Edit values")
        self.bounds_source.changed.connect(self._on_bounds_source_changed)
        add_help_row(form, "Bounds", self.bounds_source, HELP["bounds_source"])
        self.prior_picker = ArtifactPicker(PRIOR_PATH, keep=lambda fn: fn.endswith(".pt"), allow_new=True)
        add_help_row(form, "Prior", self.prior_picker, HELP["prior"])
        v.addLayout(form)
        self.btn_prior = QPushButton("Build / Load prior")
        self.btn_prior.setProperty("accent", True)        # primary CTA (Fluent accent)
        self.btn_prior.clicked.connect(self._build_prior)
        v.addWidget(self.btn_prior)
        self.prior_name = QLineEdit()
        self.prior_name.setPlaceholderText("name to save prior as…")
        self.btn_save_prior = QPushButton("Save")
        self.btn_save_prior.clicked.connect(self._save_prior)
        row = QHBoxLayout()
        row.addWidget(self.prior_name, 1)
        row.addWidget(self.btn_save_prior)
        v.addLayout(row)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def on_draft_set(self, draft):
        """Config applied: repoint the bounds picker at the new model's folder and re-apply the saved
        key (it could not resolve at __init__, before any model was chosen)."""
        self.bounds_picker.base_path = BOUNDS_PATH / draft.model.lower()
        self.bounds_picker.refresh()
        self.bounds_picker.restore_key(self._saved_bounds_key)   # -1 guard leaves default if absent
        if self.bounds_source.is_direct():        # a different model means a different parameter set
            self._on_bounds_source_changed()

    def _on_bounds_source_changed(self):
        """Entering direct-entry mode seeds the grid FROM the selected file: the parameter names and
        their order belong to the model, so hand-entry edits numbers rather than inventing a schema."""
        if not self.bounds_source.is_direct():
            return
        path = self.bounds_picker.selected_path()
        if not path:
            self.log_pane.append_line("Select a bounds file first — direct entry starts from it.",
                                      "warning")
            self.bounds_source.set_direct(False)
            return
        try:
            params, rescale, forcing, _ = file_manager.parse_bounds_file(path)
        except Exception as e:                       # noqa: BLE001
            self._config_error(e)
            self.bounds_source.set_direct(False)
            return
        self.bounds_grid.load(params, rescale, forcing)

    def _build_prior(self):
        """Build the SimConfig from (Config draft + this tab's bounds file), then build/load the prior.

        The config is built HERE because the bounds file is what defines the inferred parameter set --
        and therefore the observation mode -- so it cannot exist until this tab has been used."""
        draft = self.session.draft
        if draft is None:
            return
        if self.bounds_source.is_direct():
            problems = self.bounds_grid.problems()
            if problems:
                self.log_pane.append_line("Fix the bounds first: " + "; ".join(problems), "warning")
                return
            source = dict(bounds_dicts=self.bounds_grid.to_dicts())
        else:
            bounds_path = self.bounds_picker.selected_path()
            if not bounds_path:
                self.log_pane.append_line("Select a bounds file first.", "warning")
                return
            source = dict(bounds_path=bounds_path)
        try:
            cfg = draft.make_config(**source)
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return
        for msg in cfg.check_unit_consistency():      # a units declaration that contradicts the pipeline
            self.log_pane.append_line(msg, "warning")
        self.session.reset_downstream("prior")
        self._screen.install_config(cfg)             # sets session.cfg + repoints the Infer tab + re-gates
        _MODE_BLURB = {
            "spontaneous": "one passive trace, no drive anywhere",
            "forced": "passive + one forced trace at the cell's drive",
            "chi": "passive + K single-tone forced traces (the cell's own drive is ignored)",
        }
        self.log_pane.append_line(
            f"Config built: {cfg.model} — {len(cfg.params_dict)} ND + {len(cfg.rescale_params)} rescale "
            f"params. Observation mode: {cfg.observation_mode.upper()} "
            f"({_MODE_BLURB[cfg.observation_mode]}).")
        if cfg.observation_mode == "spontaneous" and "f_scale" in cfg.rescale_params:
            self.log_pane.append_line(
                "This config infers f_scale but has no drive anywhere, so f_scale cannot affect the "
                "observable — its marginal will just return the prior. Use a bounds file without a "
                "Forcing section AND without f_scale for spontaneous inference.", "warning")
        if cfg.chi_mode:
            lo, hi = cfg.chi_freq_bounds
            # Width from the SHARED rule, not 3*K: that was layout 1, where a probe's frequency was
            # implied by its slot. Under the padded probe set it is CHI_ELEM_W * chi_k_pad and does
            # not depend on K at all -- which is the entire point of the layout, so reporting the old
            # formula here told the user the opposite of what the mode now does.
            self.log_pane.append_line(
                f"χ(ω) mode: {cfg.chi_n_freqs} drive frequencies over {lo:g}–{hi:g}×Ω₀ at ND amplitude "
                f"{cfg.chi_f0:g}, each locked in over at most {cfg.chi_max_cycles:g} drive cycles; "
                f"conditioning is [S(41) | log T | χ({orchestrator.expected_forcing_dim(cfg)})] over "
                f"{cfg.chi_k_pad} probe slots. Train a NEW posterior (the width differs from a non-χ one).")

        entry, is_new = self.prior_picker.selected()
        self.dispatch(orchestrator.build_prior, cfg, entry, is_new, save=False,
                      provide_fig_sink=True, on_result=self._on_prior)

    def _on_prior(self, payload):
        self.session.inf_prior, self.session.force_prior = payload
        self.log_pane.append_line("Prior ready.")
        self._screen.refresh_gates()

    def _save_prior(self):
        name = self.prior_name.text().strip()
        if not name or self.session.inf_prior is None:
            self.log_pane.append_line("Build a prior and enter a name first.", "warning")
            return
        nd_prior = self.session.inf_prior.distributions[0]
        self.dispatch(orchestrator.save_prior_artifacts, name, nd_prior, self.session.cfg,
                      on_finished=lambda: (self.prior_picker.refresh(),
                                           self.log_pane.append_line(f"Saved prior '{name}'.")))

    def refresh_local_gates(self):
        self.btn_prior.setEnabled(self.session.draft is not None)
        self.btn_save_prior.setEnabled(self.session.inf_prior is not None)

    def save_settings(self, qs):
        qs.beginGroup("inference_prior")
        qs.setValue("prior", self.prior_picker.key())
        qs.setValue("bounds", self.bounds_picker.key())
        qs.setValue("bounds_source", self.bounds_source.key())
        qs.endGroup()
        # The bounds GRID is not persisted: it is seeded from whichever file is selected, so restoring a
        # stale hand-edited grid against a different model/bounds would silently mis-bind parameters.

    def restore_settings(self, qs):
        qs.beginGroup("inference_prior")
        self.prior_picker.restore_key(settings.get_str(qs, "prior"))
        # The bounds picker points at CONFIG's model, which is not known at __init__ -- stash the key and
        # re-apply it in on_draft_set (the same deferred-restore trap the cell pickers have).
        self._saved_bounds_key = settings.get_str(qs, "bounds")
        # Always start in FILE mode: direct entry has to be seeded from a file, and no file is selected
        # until on_draft_set runs. The saved mode is deliberately not restored for that reason.
        self.bounds_source.set_direct(False)
        qs.endGroup()


# ── 3. Posterior ──────────────────────────────────────────────────────────────
class PosteriorPanel(_StagePanel):
    """Tab 3. Trains a new neural posterior, or loads a saved one.

    A loaded posterior is checked against the config's observation mode before anything else runs --
    the three conditioning widths cannot collide, so a cross-mode load is caught immediately rather
    than as a matrix-shape error deep inside the embedding net.

    Persists (group "inference_posterior"): the posterior picker selection and the save name.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Posterior")
        v = QVBoxLayout(box)
        form = make_form()
        self.post_picker = ArtifactPicker(
            POSTERIOR_PATH, keep=lambda fn: fn.endswith(".pt") and not fn.endswith(".rot.pt"), allow_new=True)
        self.post_picker.combo.currentIndexChanged.connect(lambda _i: self._sync_train_button())
        add_help_row(form, "Posterior", self.post_picker, HELP["posterior"])
        v.addLayout(form)
        self.btn_post = QPushButton("Train / Load posterior")
        self.btn_post.setProperty("accent", True)         # primary CTA (Fluent accent)
        self.btn_post.clicked.connect(self._build_posterior)
        v.addWidget(self.btn_post)
        self.post_name = QLineEdit()
        self.post_name.setPlaceholderText("name to save posterior as…")
        self.btn_save_post = QPushButton("Save")
        self.btn_save_post.clicked.connect(self._save_posterior)
        row = QHBoxLayout()
        row.addWidget(self.post_name, 1)
        row.addWidget(self.btn_save_post)
        v.addLayout(row)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _build_posterior(self):
        cfg = self.session.cfg
        if cfg is None:
            return
        entry, is_new = self.post_picker.selected()
        if is_new and self.session.inf_prior is None:
            self.log_pane.append_line("Build or load a prior first to train a new posterior.", "warning")
            return
        self.session.reset_downstream("posterior")
        self._screen.refresh_gates()
        self.dispatch(orchestrator.build_posterior, cfg, self.session.inf_prior,
                      self.session.force_prior, entry, is_new, save=False,
                      provide_fig_sink=True, on_result=self._on_posterior)

    def _on_posterior(self, payload):
        self.session.posterior, self.session.diagnostics = payload
        self.session.posterior_latent = getattr(self.session.posterior, "latent", None)
        self.session.V = self._extract_rotation(self.session.posterior)
        self.log_pane.append_line("Posterior ready.")
        self._screen.refresh_gates()

    def _save_posterior(self):
        name = self.post_name.text().strip()
        if not name or self.session.posterior_latent is None:
            self.log_pane.append_line("Train a posterior and enter a name first.", "warning")
            return
        self.dispatch(orchestrator.save_posterior_artifacts, name, self.session.posterior_latent,
                      self.session.V, self.session.diagnostics, self.session.cfg,
                      on_finished=lambda: (self.post_picker.refresh(),
                                           self.log_pane.append_line(f"Saved posterior '{name}'.")))

    def _sync_train_button(self):
        """Disable the Train button when the "(from scratch)" option is selected but no prior exists --
        loading an existing posterior is always allowed; training a new one needs a prior."""
        _entry, is_new = self.post_picker.selected()
        blocked = is_new and self.session.inf_prior is None
        self.btn_post.setEnabled(self.session.cfg is not None and not blocked)
        self.btn_post.setToolTip("Build or load a prior first to train a new posterior." if blocked else "")

    def refresh_local_gates(self):
        self._sync_train_button()
        self.btn_save_post.setEnabled(self.session.posterior_latent is not None)

    @staticmethod
    def _extract_rotation(posterior):
        """Recover the decorrelating rotation V from the posterior's transform (for a deferred save)."""
        try:
            from core.SBI.reparam import OrthogonalTransform
            parts = getattr(getattr(posterior, "T", None), "parts", [])
            if parts and isinstance(parts[0], OrthogonalTransform):
                return parts[0].M
        except Exception:
            pass
        return None

    def save_settings(self, qs):
        qs.beginGroup("inference_posterior")
        qs.setValue("posterior", self.post_picker.key())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_posterior")
        self.post_picker.restore_key(settings.get_str(qs, "posterior"))
        qs.endGroup()


# ── 4. Validate ───────────────────────────────────────────────────────────────
class ValidatePanel(_StagePanel):
    """Tab 4. Runs the calibration battery (SBC / TARP / PPC) on the current posterior.

    Gated on a posterior AND ``inf_prior`` -- deliberately not on ``force_prior``, which is None for
    every no-forcing model and once made this tab permanently unreachable for exactly those.

    Persists: nothing. It has no configurable inputs.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Validate (SBC + TARP)")
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Data-free calibration. Needs a posterior and the prior it was trained against."))
        self.btn_validate = QPushButton("Run calibration")
        self.btn_validate.setProperty("accent", True)     # primary CTA (Fluent accent)
        self.btn_validate.clicked.connect(self._validate)
        v.addWidget(self.btn_validate)
        self.controls_layout.addWidget(box)

    def _validate(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None:   # force_prior is legitimately None (no drive)
            return
        self.dispatch(orchestrator.validate_calibration, s.cfg, s.posterior,
                      s.inf_prior, s.force_prior, provide_fig_sink=True)

    def refresh_local_gates(self):
        s = self.session
        self.btn_validate.setEnabled(s.posterior is not None and s.inf_prior is not None)


# ── 5. Infer ──────────────────────────────────────────────────────────────────
class InferPanel(_StagePanel, _CellPreviewMixin):
    """Tab 5. Infers on a simulated observation (from a cell's ground truth) or on real recordings.

    Its cell picker follows the BUILT config's model rather than a live combo -- there isn't one in
    this tab. In chi mode the experimental page grows one file-picker row per probe frequency, which
    is why K is bounded at ``config.CHI_K_MAX``.

    Persists (group "inference_infer"): the mode, the cell picker, and the experimental file paths.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        self._init_cell_picker()
        self._cell_problems = []             # why the picked cell can't be used (empty = usable)
        self.cell_picker.combo.currentIndexChanged.connect(lambda _i: self._on_cell_changed())
        self._forcing_fields = {}            # name -> FloatField (experimental drive)
        box = QGroupBox("Infer")
        v = QVBoxLayout(box)

        self.infer_mode = QComboBox()
        self.infer_mode.addItems(["Simulated (cell ground truth)", "Experimental data"])
        self.infer_mode.currentIndexChanged.connect(
            lambda _i: (self._sync_infer_page(), self.refresh_local_gates()))
        mode_form = make_form()
        add_help_row(mode_form, "Mode", self.infer_mode, HELP["infer_mode"])
        v.addLayout(mode_form)

        # AdaptiveStack: the simulated page is two rows, the chi page is K+3, so a plain stack left a
        # large dead gap under the short pages.
        self.infer_stack = AdaptiveStack()
        # simulated inputs
        sim_w = QWidget(); sim_f = make_form(sim_w)
        self.sim_tobs = FloatField(T_MIN_EXP_S)
        self.values_grid = ValuesGrid()
        self.cell_source = SourceToggle(self.cell_picker, self.values_grid,
                                        file_label="Use file", direct_label="Edit values")
        self.cell_source.changed.connect(self._on_cell_source_changed)
        add_help_row(sim_f, "Cell", self.cell_source, HELP["cell_source"])
        add_help_row(sim_f, "T_obs (s)", self.sim_tobs, HELP["tobs"])
        self.infer_stack.addWidget(sim_w)
        # experimental inputs
        exp_w = QWidget(); self.exp_form = make_form(exp_w)
        self.exp_spont = PathField()
        self.exp_forced = PathField()
        self.exp_tobs = FloatField(T_MIN_EXP_S)
        add_help_row(self.exp_form, "Spontaneous", self.exp_spont, HELP["spont"])
        add_help_row(self.exp_form, "Forced", self.exp_forced, HELP["forced"])
        add_help_row(self.exp_form, "T_obs (s)", self.exp_tobs, HELP["tobs"])
        self._forcing_anchor = QLabel("(build config to list drive params)")
        self.exp_form.addRow(self._forcing_anchor)
        self.infer_stack.addWidget(exp_w)
        # page 2: experimental, chi(omega) -- one passive recording + K single-tone forced recordings
        chi_w = QWidget(); self.chi_form = make_form(chi_w)
        self.chi_spont = PathField()
        self.chi_tobs = FloatField(T_MIN_EXP_S)
        self.chi_f0_si = FloatField(1.0)
        self._chi_forced_fields = []
        add_help_row(self.chi_form, "Passive", self.chi_spont, HELP["chi_passive"])
        add_help_row(self.chi_form, "T_obs (s)", self.chi_tobs, HELP["tobs"])
        add_help_row(self.chi_form, "Drive F₀ (N)", self.chi_f0_si, HELP["chi_f0_si"])
        self._chi_anchor = QLabel("(build a χ config to list drive frequencies)")
        self.chi_form.addRow(self._chi_anchor)
        self.infer_stack.addWidget(chi_w)
        v.addWidget(self.infer_stack)

        self.btn_infer = QPushButton("Run inference")
        self.btn_infer.setProperty("accent", True)        # primary CTA (Fluent accent)
        self.btn_infer.clicked.connect(self._infer)
        v.addWidget(self.btn_infer)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _on_cell_source_changed(self):
        """Entering direct-entry seeds the grid from the picked cell -- same rule as the bounds grid:
        the parameter schema belongs to the model, only the numbers are the user's."""
        if not self.cell_source.is_direct():
            self._on_cell_changed()                  # back to file mode: re-validate the picked file
            return
        path = self.cell_picker.selected_path()
        if not path:
            self.log_pane.append_line("Select a cell file first — direct entry starts from it.",
                                      "warning")
            self.cell_source.set_direct(False)
            return
        try:
            inits, params, rescale, forcing = file_manager.parse_values_file(path)
        except Exception as e:                       # noqa: BLE001
            self._config_error(e)
            self.cell_source.set_direct(False)
            return
        self.values_grid.load(inits, params, rescale, forcing)
        self._cell_problems = []                     # hand-entered values are validated at Run instead
        self.refresh_local_gates()

    def _on_cell_changed(self):
        """Validate the picked cell against the bounds file ON THE GUI THREAD, the moment it is chosen.

        The check already existed inside inject_ground_truth, but it only fired inside the worker -- so a
        mismatched cell surfaced as a mid-run error dialog after the user had already committed. Here it
        is immediate, and the Run button stays disabled until a usable cell is selected."""
        self._cell_problems = []
        cfg, path = self.session.cfg, self.cell_picker.selected_path()
        if cfg is not None and path:
            try:
                self._cell_problems = cli.validate_gt_file(cfg, path)
            except Exception:                          # noqa: BLE001 -- a pre-flight check must never
                self._cell_problems = []               # break the panel; the worker still validates
            if self._cell_problems:
                self.log_pane.append_line(
                    "This cell does not fit the bounds file used to build the config: "
                    + "; ".join(self._cell_problems) + ". Choose another cell (or rebuild the config "
                    "against matching bounds).", "warning")
        self.refresh_local_gates()

    def on_config_built(self, cfg):
        _CellPreviewMixin.on_config_built(self, cfg)
        self._on_cell_changed()                       # re-validate against the newly built config
        self._rebuild_forcing_fields(cfg)
        # A no-forcing (passive) model has no forced recording and no drive params: hide the forced row.
        # _rebuild_forcing_fields already produces no forcing fields for an empty force_params_dict.
        self.exp_form.setRowVisible(self.exp_forced, cfg.has_forcing)
        self._rebuild_chi_fields(cfg)
        self._sync_infer_page()

    def _sync_infer_page(self):
        """Experimental mode shows the χ page when the config is χ-mode. Load-bearing: a χ observation
        needs K forced recordings, and falling through to the ordinary experimental branch would build a
        silently wrong-width conditioning vector rather than failing."""
        if self.infer_mode.currentIndex() == 0:
            self.infer_stack.setCurrentIndex(0)
            return
        cfg = self.session.cfg
        mode = cfg.observation_mode if cfg is not None else "forced"
        self.infer_stack.setCurrentIndex(2 if mode == "chi" else 1)

    def _rebuild_chi_fields(self, cfg):
        """One forced-recording PathField per χ probe frequency, labelled with its multiple of Ω₀."""
        for fld in self._chi_forced_fields:
            self.chi_form.removeRow(fld)
        self._chi_forced_fields = []
        if self._chi_anchor is not None:
            self.chi_form.removeRow(self._chi_anchor)
            self._chi_anchor = None
        if not cfg.chi_mode:
            self._chi_anchor = QLabel("(build a χ config to list drive frequencies)")
            self.chi_form.addRow(self._chi_anchor)
            return
        from core.SBI import chi as _chi
        for mult in _chi.chi_multipliers_for(cfg).tolist():
            fld = PathField()
            self._chi_forced_fields.append(fld)
            add_help_row(self.chi_form, f"Forced @ {mult:.3g}×Ω₀", fld, HELP["chi_forced"])

    def _rebuild_forcing_fields(self, cfg):
        for fld in self._forcing_fields.values():
            self.exp_form.removeRow(fld)
        self._forcing_fields = {}
        if self._forcing_anchor is not None:
            self.exp_form.removeRow(self._forcing_anchor)
            self._forcing_anchor = None
        for name in cfg.force_params_dict:
            unit = cli._INFERENCE_PROMPT_UNITS.get(name, "")
            fld = FloatField(0.0)
            self._forcing_fields[name] = fld
            add_help_row(self.exp_form, labels.gui_forcing_label(name, unit), fld, HELP["forcing"])

    def _infer(self):
        cfg, post = self.session.cfg, self.session.posterior
        if post is None:
            return
        if self.infer_mode.currentIndex() == 0:      # simulated
            gt_dicts, cell = None, None
            if self.cell_source.is_direct():
                problems = self.values_grid.problems()
                if problems:
                    self.log_pane.append_line("Fix the values first: " + "; ".join(problems), "warning")
                    return
                gt_dicts = self.values_grid.to_dicts()
            else:
                cell = self.cell_picker.selected_path()
                if not cell:
                    self.log_pane.append_line("Select a cell file first.", "warning")
                    return
                if self._cell_problems:
                    self.log_pane.append_line(
                        "Fix the cell selection first: " + "; ".join(self._cell_problems), "warning")
                    return
            self.dispatch(_run_simulated_inference, cfg, post, cell, self.sim_tobs.value(),
                          gt_dicts=gt_dicts,
                          inferred_prior=self.session.inf_prior, force_prior=self.session.force_prior,
                          provide_fig_sink=True)
        elif cfg.observation_mode == "chi":          # experimental, χ(ω): 1 passive + K forced
            paths = [f.value() for f in self._chi_forced_fields]
            if not self.chi_spont.value() or not all(paths):
                self.log_pane.append_line(
                    f"Select the passive recording and all {len(self._chi_forced_fields)} forced "
                    "recordings first.", "warning")
                return
            self.dispatch(_run_experimental_inference_chi, cfg, post, self.chi_spont.value(), paths,
                          self.chi_tobs.value(), self.chi_f0_si.value(), provide_fig_sink=True)
        elif not cfg.has_forcing:                    # experimental, passive (no drive)
            if not self.exp_spont.value():
                self.log_pane.append_line("Select a passive recording first.", "warning")
                return
            self.dispatch(_run_experimental_inference_spontaneous, cfg, post,
                          self.exp_spont.value(), self.exp_tobs.value(), provide_fig_sink=True)
        else:                                        # experimental, driven
            forcing_si = {name: fld.value() for name, fld in self._forcing_fields.items()}
            self.dispatch(_run_experimental_inference, cfg, post, self.exp_spont.value(),
                          self.exp_forced.value(), self.exp_tobs.value(), forcing_si, provide_fig_sink=True)

    def refresh_local_gates(self):
        simulated = self.infer_mode.currentIndex() == 0
        # _cell_problems only describes the PICKED FILE; hand-entered values are validated at Run.
        blocked = (simulated and not self.cell_source.is_direct()
                   and bool(getattr(self, "_cell_problems", [])))
        self.btn_infer.setEnabled(self.session.posterior is not None and not blocked)
        self.btn_infer.setToolTip(
            "The selected cell does not fit the bounds file used to build the config." if blocked else "")

    def save_settings(self, qs):
        qs.beginGroup("inference_infer")
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("infer_mode", self.infer_mode.currentIndex())
        settings.save_field(qs, "sim_tobs", self.sim_tobs)
        settings.save_field(qs, "exp_tobs", self.exp_tobs)
        settings.save_field(qs, "exp_spont", self.exp_spont)
        settings.save_field(qs, "exp_forced", self.exp_forced)
        settings.save_field(qs, "chi_spont", self.chi_spont)
        settings.save_field(qs, "chi_tobs", self.chi_tobs)
        settings.save_field(qs, "chi_f0_si", self.chi_f0_si)
        qs.endGroup()
        # The forcing fields and the per-frequency χ forced-recording fields don't exist until
        # "Build config" runs (and their COUNT depends on K), so they are not persisted.

    def restore_settings(self, qs):
        qs.beginGroup("inference_infer")
        self._saved_cell_key = settings.get_str(qs, "cell")     # re-applied in on_config_built
        try:
            self.infer_mode.setCurrentIndex(int(settings.get_str(qs, "infer_mode", "0")))
        except ValueError:
            pass
        settings.restore_field(qs, "sim_tobs", self.sim_tobs)
        settings.restore_field(qs, "exp_tobs", self.exp_tobs)
        settings.restore_field(qs, "exp_spont", self.exp_spont)
        settings.restore_field(qs, "exp_forced", self.exp_forced)
        settings.restore_field(qs, "chi_spont", self.chi_spont)
        settings.restore_field(qs, "chi_tobs", self.chi_tobs)
        settings.restore_field(qs, "chi_f0_si", self.chi_f0_si)
        qs.endGroup()
