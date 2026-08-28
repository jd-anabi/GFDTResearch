"""Parameter Inference, split into six tabs over ONE shared SbiSession owned by the InferenceScreen.

    Config -> Prior -> Posterior -> Validate -> Infer -> TSNPE

Config records the MODEL-level choices as a ConfigDraft; the PRIOR tab picks the bounds file and turns
that draft into the SimConfig, because the bounds file declares the inferred parameter set (and hence the
observation mode), so a config cannot exist before it.

Each tab is its own BasePanel (its own FigureStack/ProgressPane/LogPane), so dispatch()/cancel/the
figure-sink/progress plumbing are reused verbatim; BasePanel._running is class-level, so the tabs can
never run concurrently. The screen owns the SbiSession and greys tabs via setTabEnabled; every tab
reads/writes the session through ``self._screen`` and calls ``self._screen.refresh_gates()`` after a
stage completes.
"""
import math
import os
import subprocess

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox,
                               QLineEdit, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from core import cli, config, forcing, orchestrator, registry
from core.Helpers import file_manager, labels, visualizers
from core.SBI import pipeline, training_checkpoint
from core.config import (VALID_MODELS, VALID_LABELS, BOUNDS_PATH, CELL_PATH, PRIOR_PATH, OBSERVATION_PATH,
                         POSTERIOR_PATH, T_MIN_EXP_S, CHI_K_MAX)

from .base_panel import BasePanel
from .. import icons, settings
from ..session import ConfigDraft
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.help_badge import add_help_row, with_badge
from ..widgets.labeled_inputs import FloatField, IntField, PathField
from ..widgets.param_grid import BoundsGrid, ValuesGrid
from ..widgets.source_toggle import SourceToggle
from ..widgets.field_row import LabeledFieldRow
from ..widgets.forms import make_form
from ..widgets.adaptive_stack import AdaptiveStack

from .inference.help_text import HELP
from .inference.rows import _ChiRangeRow, _ChiProbeRow
from .inference.runners import (_run_simulated_inference, _run_experimental_inference,
                                _run_experimental_inference_chi,
                                _run_experimental_inference_spontaneous, _run_tsnpe_round)
from .inference.base import (_StagePanel, _CellPreviewMixin, _TrainingBudgetMixin,
                             _hw_batch, _nvidia_smi_free_gib)

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

        # -- hardware: not a science knob. It changes how a batch is PLANNED, never what is trained,
        # which is exactly why it is kept out of the checkpoint identity (a memory knob in the
        # digest would rename the checkpoint directory and silently restart a resumable multi-day
        # run). It lives on Config rather than Prior/Posterior because it applies to every stage
        # that simulates.
        hw = QGroupBox("Hardware")
        hv = QVBoxLayout(hw)
        hform = make_form()
        self.vram_ceiling = FloatField(str(pipeline.vram_ceiling_gib()))
        add_help_row(hform, "VRAM ceiling per batch (GiB, 0 = off)", self.vram_ceiling,
                     HELP["vram_ceiling"])
        hv.addLayout(hform)
        self.vram_note = _TrainingBudgetMixin._derived_label()
        hv.addWidget(self.vram_note)
        self.vram_ceiling.textChanged.connect(lambda _t: self._apply_vram_ceiling())
        self.controls_layout.addWidget(hw)
        self._apply_vram_ceiling()

        self.restore_settings(settings.settings())
        self._sync_chi_enabled()

    def _apply_vram_ceiling(self):
        """Push the field into ``config.SIM_VRAM_CEILING_GIB`` and say what will ACTUALLY take effect.

        ASSIGNING THE CONSTANT IS ENOUGH HERE, and that is not true of the sweep and flow knobs
        beside it. Those had to become ARGUMENTS because `orchestrator` does `from .config import ...`
        and binds them at import, so assigning to the constant is a silent no-op. This one
        is read LIVE -- `pipeline.vram_ceiling_gib()` does a `getattr` on the module every time the
        planner asks -- so a plain assignment reaches every stage that simulates, with no plumbing.

        NOT PERSISTED, ON PURPOSE, and it is the only field on this tab that is not. Stale QSettings
        have already cost this project a ~5-day run (the 2026-08-19 retrain trained on the retired
        band because a saved value silently won over config.py). A ceiling fails the same way but
        more quietly: a forgotten 2 GiB would not error, it would just make every future run split
        from batch 0 and take several times longer, with nothing in the log to explain it. Starting
        each session from config.py's 0.0 means the throttle is always a decision someone just made.

        The env override still wins if it is set -- said out loud here rather than left to puzzle
        over, because a field that silently does nothing is worse than no field.
        """
        config.SIM_VRAM_CEILING_GIB = max(0.0, self.vram_ceiling.value())
        effective = pipeline.vram_ceiling_gib()
        env = os.environ.get(pipeline.VRAM_CEILING_ENV)
        free = _nvidia_smi_free_gib()
        free_txt = (f"nvidia-smi reports {free:.2f} GiB free"
                    if free is not None else "free VRAM unreadable (no nvidia-smi)")
        if env is not None and env.strip():
            self.vram_note.setText(
                f"⚠ {pipeline.VRAM_CEILING_ENV}={env} is set and OVERRIDES this field — "
                f"planning to {effective:.2f} GiB. {free_txt}.")
        elif effective <= 0:
            self.vram_note.setText(
                f"Off: batches are planned from the free-memory reading and the learned cap alone. "
                f"{free_txt} — set a ceiling near that, minus ~1 GiB, only if you need the desktop.")
        else:
            self.vram_note.setText(
                f"Batches will be planned to fit {effective:.2f} GiB. {free_txt}. Above the real "
                f"free figure this does nothing; below it, expect more splitting and more wall-clock.")

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
                cli.units_to_factors(units)
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

        # -- the stability sweep: what actually builds the prior ---------------------------------
        sweep = QGroupBox("Stability sweep")
        sv = QVBoxLayout(sweep)
        sform = make_form()
        self.sweep_iters = IntField(str(config.PRIOR_SWEEP_ITERATIONS))
        self.sweep_batch = IntField(str(config.PRIOR_SWEEP_BATCH))
        self.sweep_max_sets = IntField(str(config.PRIOR_SWEEP_MAX_SETS))
        self.sweep_step = FloatField(str(config.PRIOR_SWEEP_STEP))
        self.sweep_units = FloatField(str(config.STABILITY_SWEEP_ND_UNITS))
        add_help_row(sform, "Global rounds", self.sweep_iters, HELP["sweep_iters"])
        add_help_row(sform, "Candidates per round (0 = auto)", self.sweep_batch, HELP["sweep_batch"])
        add_help_row(sform, "Max accepted sets", self.sweep_max_sets, HELP["sweep_max_sets"])
        add_help_row(sform, "Random-walk step", self.sweep_step, HELP["sweep_step"])
        add_help_row(sform, "Stability duration (ND units)", self.sweep_units, HELP["sweep_units"])
        sv.addLayout(sform)
        self.sweep_note = _TrainingBudgetMixin._derived_label()
        sv.addWidget(self.sweep_note)
        for fld in (self.sweep_iters, self.sweep_batch, self.sweep_max_sets, self.sweep_units):
            fld.textChanged.connect(lambda _t: self._sync_sweep())
        self.controls_layout.addWidget(sweep)
        self._sync_sweep()

        # -- clustering: a different STAGE from the sweep. The sweep maps the stable manifold;
        # this decides how many MODES the prior has, because HDBSCAN's label count becomes the
        # GMM's n_components.
        clust = QGroupBox("Clustering / GMM")
        cv = QVBoxLayout(clust)
        cform = make_form()
        self.cluster_size = IntField(str(config.PRIOR_CLUSTER_MIN_SIZE))
        self.cluster_samples = IntField(str(config.PRIOR_CLUSTER_MIN_SAMPLES))
        add_help_row(cform, "Min cluster size", self.cluster_size, HELP["cluster_size"])
        add_help_row(cform, "Min samples", self.cluster_samples, HELP["cluster_samples"])
        cv.addLayout(cform)
        self.controls_layout.addWidget(clust)

        self.restore_settings(settings.settings())

    def on_draft_set(self, draft):
        """Config applied: repoint the bounds picker at the new model's folder and re-apply the saved
        key (it could not resolve at __init__, before any model was chosen)."""
        self.bounds_picker.repoint(BOUNDS_PATH / draft.model.lower(), self._saved_bounds_key)
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
        # Passed, never written to config: orchestrator does `from .config import
        # PRIOR_SWEEP_ITERATIONS, ...`, so assigning to the constants here would be a silent no-op.
        self.dispatch(orchestrator.build_prior, cfg, entry, is_new, save=False,
                      provide_fig_sink=True, on_result=self._on_prior,
                      num_iterations=max(1, self.sweep_iters.value()),
                      sweep_batch=max(0, self.sweep_batch.value()),
                      max_sets=max(1, self.sweep_max_sets.value()),
                      walk_step=self.sweep_step.value() or config.PRIOR_SWEEP_STEP,
                      stability_units=self.sweep_units.value() or config.STABILITY_SWEEP_ND_UNITS,
                      min_cluster_size=max(2, self.cluster_size.value()),
                      min_samples=max(1, self.cluster_samples.value()))

    def _sync_sweep(self) -> None:
        """The one derived line: how many candidates the GLOBAL census screens, and where the time
        goes. Pure and cheap, so it is safe on every keystroke; wrapped because a status line must
        never be able to raise into refresh_gates and take the tab down."""
        try:
            cfg = self.session.cfg
            hw_batch = getattr(getattr(cfg, "hw", None), "batch_size", None) or config.detect_device().batch_size
            per_round = self.sweep_batch.value() or hw_batch
            rounds = max(1, self.sweep_iters.value())
            units = self.sweep_units.value() or config.STABILITY_SWEEP_ND_UNITS
            dt = getattr(cfg, "dt_nd_min", None)
            steps = f"{int(units / dt):,}" if dt else "?"
            self.sweep_note.setText(
                f"Global census screens {rounds * per_round:,} candidates ({rounds:,} rounds x "
                f"{per_round:,}), each integrated over {steps} steps.\n"
                f"The LOCAL flood-fill then runs until {max(1, self.sweep_max_sets.value()):,} sets "
                f"are accepted — that is the dominant cost of a prior build, and it now runs on the "
                f"same device as the global sweep (falling back to the CPU when there is no "
                f"accelerator).")
        except Exception as e:                    # noqa: BLE001 -- never break the tab over a label
            self.sweep_note.setText(f"Sweep summary unavailable: {type(e).__name__}: {e}")

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
        for name in ("sweep_iters", "sweep_batch", "sweep_max_sets", "sweep_step", "sweep_units",
                     "cluster_size", "cluster_samples"):
            qs.setValue(name, str(getattr(self, name).value()))
        qs.endGroup()
        # The bounds GRID is not persisted: it is seeded from whichever file is selected, so restoring a
        # stale hand-edited grid against a different model/bounds would silently mis-bind parameters.

    def restore_settings(self, qs):
        qs.beginGroup("inference_prior")
        self.prior_picker.restore_key(settings.get_str(qs, "prior"))
        # The bounds picker points at CONFIG's model, which is not known at __init__ -- stash the key and
        # re-apply it in on_draft_set (the same deferred-restore trap the cell pickers have).
        self._saved_bounds_key = settings.get_str(qs, "bounds")
        # str + cast, because settings has no get_float; a blank or unparseable value falls back to
        # the config constant rather than to FloatField.value()'s 0.0 -- and a 0 here would mean a
        # sweep with no rounds, or a flood-fill that stops at zero accepted sets.
        for name, default, cast in (("sweep_iters", config.PRIOR_SWEEP_ITERATIONS, int),
                                    ("sweep_batch", config.PRIOR_SWEEP_BATCH, int),
                                    ("sweep_max_sets", config.PRIOR_SWEEP_MAX_SETS, int),
                                    ("sweep_step", config.PRIOR_SWEEP_STEP, float),
                                    ("sweep_units", config.STABILITY_SWEEP_ND_UNITS, float),
                                    ("cluster_size", config.PRIOR_CLUSTER_MIN_SIZE, int),
                                    ("cluster_samples", config.PRIOR_CLUSTER_MIN_SAMPLES, int)):
            try:
                getattr(self, name).setText(str(cast(settings.get_str(qs, name, str(default)))))
            except (TypeError, ValueError):
                getattr(self, name).setText(str(default))
        # Always start in FILE mode: direct entry has to be seeded from a file, and no file is selected
        # until on_draft_set runs. The saved mode is deliberately not restored for that reason.
        self.bounds_source.set_direct(False)
        qs.endGroup()


# ── the training budget, shared by the Posterior and TSNPE tabs ───────────────────────────────
# ── 3. Posterior ──────────────────────────────────────────────────────────────
class PosteriorPanel(_TrainingBudgetMixin, _StagePanel):
    """Tab 3. Trains a new neural posterior, or loads a saved one.

    A loaded posterior is checked against the config's observation mode before anything else runs --
    the three conditioning widths cannot collide, so a cross-mode load is caught immediately rather
    than as a matrix-shape error deep inside the embedding net.

    Also owns the TRAINING BUDGET (batches x rows-per-batch), which was previously reachable only by
    editing config.py. Both fields are passed to build_posterior as arguments rather than written to
    config: orchestrator snapshots those constants at import, so assigning to them would be a silent
    no-op. See _sync_budget for the three derived lines and why the checkpoint one is not a tooltip.

    Persists (group "inference_posterior"): the posterior picker selection, the batch count and the
    rows-per-batch cap.
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

        # -- training budget: what a new posterior will actually simulate, and what it costs -------
        budget = QGroupBox("Training budget")
        bv = QVBoxLayout(budget)
        bform = make_form()
        self.num_runs = IntField(config.TRAINING_NUM_RUNS)
        self.run_size_cap = IntField(config.TRAINING_RUN_SIZE)
        add_help_row(bform, "Batches", self.num_runs, HELP["num_runs"])
        add_help_row(bform, "Max rows per batch (0 = auto)", self.run_size_cap, HELP["run_size"])
        bv.addLayout(bform)
        self.budget_total = self._derived_label()
        self.budget_mem = self._derived_label()
        self.budget_ckpt = self._derived_label()
        for lab in (self.budget_total, self.budget_mem, self.budget_ckpt):
            bv.addWidget(lab)
        # After the labels exist: textChanged fires during restore_settings below.
        for fld in (self.num_runs, self.run_size_cap):
            fld.textChanged.connect(lambda _t: self._sync_budget())
        self.controls_layout.addWidget(budget)

        # -- flow capacity: re-tryable against a COMPLETE checkpoint without re-simulating --------
        flow = QGroupBox("Density estimator")
        fv = QVBoxLayout(flow)
        fform = make_form()
        self.flow_hidden = IntField(str(config.NSF_HIDDEN_FEATURES))
        self.flow_transforms = IntField(str(config.NSF_NUM_TRANSFORMS))
        self.flow_lr = FloatField(str(config.TRAINING_LEARNING_RATE))
        self.flow_patience = IntField(str(config.TRAINING_STOP_AFTER_EPOCHS))
        add_help_row(fform, "Hidden features", self.flow_hidden, HELP["flow_hidden"])
        add_help_row(fform, "Transforms", self.flow_transforms, HELP["flow_transforms"])
        add_help_row(fform, "Learning rate", self.flow_lr, HELP["flow_lr"])
        add_help_row(fform, "Early-stop patience", self.flow_patience, HELP["flow_patience"])
        fv.addLayout(fform)
        self.controls_layout.addWidget(flow)

        fisher = QGroupBox("Fisher rotation")
        rv = QVBoxLayout(fisher)
        rform = make_form()
        self.fisher_m = IntField(str(config.REPARAM_FISHER_M))
        self.fisher_dz = FloatField(str(config.REPARAM_FISHER_DZ))
        self.fisher_points = IntField(str(config.REPARAM_FISHER_POINTS))
        add_help_row(rform, "Ensemble per perturbation", self.fisher_m, HELP["fisher_m"])
        add_help_row(rform, "Central-difference step", self.fisher_dz, HELP["fisher_dz"])
        add_help_row(rform, "Operating points", self.fisher_points, HELP["fisher_points"])
        rv.addLayout(rform)
        self.controls_layout.addWidget(fisher)

        self.restore_settings(settings.settings())
        self._sync_budget()

    def _build_posterior(self):
        cfg = self.session.cfg
        if cfg is None:
            return
        entry, is_new = self.post_picker.selected()
        if is_new and self.session.inf_prior is None:
            self.log_pane.append_line("Build or load a prior first to train a new posterior.", "warning")
            return
        n_runs, cap = self._budget_values()
        if n_runs < 1:
            self.log_pane.append_line("Batches must be at least 1.", "warning")
            return
        if cap < 0:
            self.log_pane.append_line("Max rows per batch cannot be negative (0 = auto).", "warning")
            return
        if is_new and not self._confirm_fresh_run(cfg, cap or _hw_batch(cfg), n_runs):
            return
        self.session.reset_downstream("posterior")
        self._screen.refresh_gates()
        # Passed, never written to config: orchestrator does `from .config import TRAINING_NUM_RUNS`,
        # so setting the constant here would be a silent no-op and the run would use the default.
        self.dispatch(orchestrator.build_posterior, cfg, self.session.inf_prior,
                      self.session.force_prior, entry, is_new, save=False,
                      num_runs=n_runs, run_size_cap=cap,
                      hidden_features=max(1, self.flow_hidden.value()),
                      num_transforms=max(1, self.flow_transforms.value()),
                      learning_rate=self.flow_lr.value() or config.TRAINING_LEARNING_RATE,
                      stop_after_epochs=max(1, self.flow_patience.value()),
                      fisher_m=max(1, self.fisher_m.value()),
                      fisher_dz=self.fisher_dz.value() or config.REPARAM_FISHER_DZ,
                      fisher_points=max(1, self.fisher_points.value()),
                      provide_fig_sink=True, on_result=self._on_posterior)

    def _confirm_fresh_run(self, cfg, width: int, n_runs: int) -> bool:
        """Ask before starting from zero when a checkpoint is ONE FIELD away. True = go ahead.

        THE STATUS LINE WAS NOT ENOUGH, and this is the evidence. `_budget_checkpoint` already says
        "these settings match no checkpoint, so this starts a NEW run" and names the differing field
        -- but it is a passive label, on a tab the user has usually scrolled past by the time they
        press Train, and it has now failed to prevent three restarts: 884 batches lost outright on
        2026-08-27 (a prior rebuilt rather than loaded, and never saved, so unrecoverable), and a
        3989-batch checkpoint nearly abandoned twice on 2026-08-28 because a prior was selected in
        the picker but never loaded. A modal costs one click on the rare occasion it fires.

        DELIBERATELY NARROW. It asks only when a committed sibling differs in EXACTLY ONE field --
        the signature of an accident rather than of a different experiment. A genuinely new run,
        with no near-miss, is never interrupted.

        FAILS OPEN. A status line must never block a run: if the identity cannot be computed (no
        prior yet, an unreadable header) this returns True and the run proceeds, exactly as before.
        """
        if not config.TRAINING_CHECKPOINT_EVERY or self.session.inf_prior is None:
            return True
        try:
            ident = orchestrator.training_identity(cfg, self.session.inf_prior, width, n_runs)
            if (training_checkpoint.peek(training_checkpoint.resolve_dir(ident)) or {}).get("batches_done"):
                return True                  # this IS a resume; nothing to warn about
            near = training_checkpoint.near_miss_siblings(ident)
        except Exception as e:               # noqa: BLE001 -- never block a run over a warning
            self.log_pane.append_line(
                f"Could not check for resumable checkpoints ({type(e).__name__}: {e}); "
                f"continuing.", "warning")
            return True
        if not near:
            return True

        lines = [f"  • {r['name']}: {r['batches']:,} batches — differs only in {r['field']}\n"
                 f"      this run: {str(r['mine'])[:60]}\n"
                 f"      that one: {str(r['theirs'])[:60]}" for r in near[:3]]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("This starts a NEW run")
        box.setText(f"This will simulate {n_runs:,} batches from zero.")
        box.setInformativeText(
            "A checkpoint that is ONE setting away already exists:\n\n" + "\n".join(lines) +
            "\n\nIf you meant to continue that run, cancel and change the setting named above "
            "— for a prior, remember that choosing it in the picker does nothing until you press "
            "\"Build / Load prior\".")
        go = box.addButton("Start a new run anyway", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        return box.clickedButton() is go

    def _on_posterior(self, payload):
        self.session.posterior, self.session.diagnostics = payload
        self.session.posterior_latent = getattr(self.session.posterior, "latent", None)
        self.session.V = self._extract_rotation(self.session.posterior)
        # CLEARED, not merely left alone. Training an amortized posterior after a TSNPE round would
        # otherwise inherit that round's region and be saved marked non-amortized -- the mislabelling
        # runs in both directions, and this is the direction that is easy to miss.
        self.session.truncation = self.session.x_obs_digest = None
        self.log_pane.append_line("Posterior ready.")
        self._screen.refresh_gates()

    def _save_posterior(self):
        name = self.post_name.text().strip()
        if not name or self.session.posterior_latent is None:
            self.log_pane.append_line("Train a posterior and enter a name first.", "warning")
            return
        self.dispatch(orchestrator.save_posterior_artifacts, name, self.session.posterior_latent,
                      self.session.V, self.session.diagnostics, self.session.cfg,
                      truncation=self.session.truncation,
                      x_obs_digest=self.session.x_obs_digest,
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
        # A config or a prior arriving changes every derived line, and the checkpoint line cannot be
        # computed without both.
        self._sync_budget()

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
        qs.setValue("num_runs", self.num_runs.value())
        qs.setValue("run_size_cap", self.run_size_cap.value())
        for name in ("flow_hidden", "flow_transforms", "flow_lr", "flow_patience",
                     "fisher_m", "fisher_dz", "fisher_points"):
            qs.setValue(name, str(getattr(self, name).value()))
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_posterior")
        self.post_picker.restore_key(settings.get_str(qs, "posterior"))
        for name, default, cast in (("flow_hidden", config.NSF_HIDDEN_FEATURES, int),
                                    ("flow_transforms", config.NSF_NUM_TRANSFORMS, int),
                                    ("flow_lr", config.TRAINING_LEARNING_RATE, float),
                                    ("flow_patience", config.TRAINING_STOP_AFTER_EPOCHS, int),
                                    ("fisher_m", config.REPARAM_FISHER_M, int),
                                    ("fisher_dz", config.REPARAM_FISHER_DZ, float),
                                    ("fisher_points", config.REPARAM_FISHER_POINTS, int)):
            try:
                getattr(self, name).setText(str(cast(settings.get_str(qs, name, str(default)))))
            except (TypeError, ValueError):
                getattr(self, name).setText(str(default))
        # Defaults are the config constants, so a fresh install and a wiped QSettings both land on
        # exactly the CLI's behaviour.
        self.num_runs.setText(str(settings.get_int(qs, "num_runs", config.TRAINING_NUM_RUNS)))
        self.run_size_cap.setText(str(settings.get_int(qs, "run_size_cap", config.TRAINING_RUN_SIZE)))
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
        cform = make_form()
        self.cal_n = IntField(str(config.SBC_N_CAL))
        self.cal_scales = IntField(str(config.CAL_N_SCALES))
        add_help_row(cform, "Calibration datasets", self.cal_n, HELP["cal_n"])
        add_help_row(cform, "(t_scale, T) operating points", self.cal_scales, HELP["cal_scales"])
        v.addLayout(cform)
        self.btn_validate = QPushButton("Run calibration")
        self.btn_validate.setProperty("accent", True)     # primary CTA (Fluent accent)
        self.btn_validate.clicked.connect(self._validate)
        v.addWidget(self.btn_validate)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _validate(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None:   # force_prior is legitimately None (no drive)
            return
        self.dispatch(orchestrator.validate_calibration, s.cfg, s.posterior,
                      s.inf_prior, s.force_prior, provide_fig_sink=True,
                      n_cal=max(1, self.cal_n.value()),
                      cal_n_scales=max(1, self.cal_scales.value()))

    def refresh_local_gates(self):
        s = self.session
        self.btn_validate.setEnabled(s.posterior is not None and s.inf_prior is not None)

    def save_settings(self, qs):
        qs.beginGroup("inference_validate")
        qs.setValue("cal_n", self.cal_n.value())
        qs.setValue("cal_scales", self.cal_scales.value())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_validate")
        self.cal_n.setText(str(settings.get_int(qs, "cal_n", config.SBC_N_CAL)))
        self.cal_scales.setText(str(settings.get_int(qs, "cal_scales", config.CAL_N_SCALES)))
        qs.endGroup()


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
        # The probe table. Rows live in their OWN container rather than as form rows, so
        # adding and removing one is a local layout edit that cannot disturb the fields above it.
        self._chi_probe_host = QWidget()
        self._chi_probe_layout = QVBoxLayout(self._chi_probe_host)
        self._chi_probe_layout.setContentsMargins(0, 0, 0, 0)
        add_help_row(self.chi_form, "Forced probes", self._chi_probe_host, HELP["chi_forced"])
        self._chi_buttons = chi_btns = QWidget(); chi_btns_l = QHBoxLayout(chi_btns)
        chi_btns_l.setContentsMargins(0, 0, 0, 0)
        self.btn_chi_add = QPushButton("+ Add probe")
        self.btn_chi_add.clicked.connect(lambda: self._add_chi_probe())
        self.btn_chi_plan = QPushButton("Plan probes…")
        self.btn_chi_plan.setToolTip("Measure Ω₀ from the passive recording and report what is in "
                                     "band, and how long each probe must be recorded.")
        self.btn_chi_plan.clicked.connect(self._plan_chi_probes)
        chi_btns_l.addWidget(self.btn_chi_add)
        chi_btns_l.addWidget(self.btn_chi_plan)
        chi_btns_l.addStretch(1)
        self.chi_form.addRow(chi_btns)
        self._chi_anchor = QLabel("(build a χ config to enable the probe table)")
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

    def _add_chi_probe(self, freq_hz: float = 0.0):
        """Append one probe row, up to the posterior's slot capacity."""
        cfg = self.session.cfg
        cap = cfg.chi_k_pad if cfg is not None and cfg.chi_mode else config.CHI_K_PAD
        if len(self._chi_forced_fields) >= cap:
            self.log_pane.append_line(
                f"This posterior reserves {cap} probe slots (CHI_K_PAD), which is frozen into the "
                f"trained artifact — it cannot take more probes than that.", "warning")
            return None
        row = _ChiProbeRow(self._remove_chi_probe, freq_hz)
        self._chi_forced_fields.append(row)
        self._chi_probe_layout.addWidget(row)
        return row

    def _remove_chi_probe(self, row):
        if row not in self._chi_forced_fields:
            return
        self._chi_forced_fields.remove(row)
        self._chi_probe_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def _rebuild_chi_fields(self, cfg):
        """Enable/disable the probe table for the built config, PRESERVING every existing row.

        Rows are never destroyed on a rebuild, and that is deliberate. They carry hand-typed drive
        frequencies and browsed recording paths -- neither of which this method could regenerate, and
        both of which represent a bench session that already happened. Rebuilding the config (to fix
        a bounds file, say) must not silently discard them. Contrast _rebuild_forcing_fields, whose
        rows ARE derivable from the config's forcing schema and so are rebuilt freely.

        Count and placement are both free: the encoder is permutation-invariant and
        carries each probe's frequency explicitly, so the table seeds a suggested number of rows and
        then gets out of the way. `cfg.chi_n_freqs` is a suggestion, NOT a requirement -- the core
        accepts 1..chi_k_pad probes at whatever frequencies the experiment achieved.
        """
        if self._chi_anchor is not None:
            self.chi_form.removeRow(self._chi_anchor)
            self._chi_anchor = None
        on = bool(cfg.chi_mode)
        self.btn_chi_add.setEnabled(on)
        self.btn_chi_plan.setEnabled(on)
        # setRowVisible, not setVisible: hiding the widget alone strands its form LABEL, so a
        # non-chi config would show a "Forced probes" caption with nothing under it.
        self.chi_form.setRowVisible(self._chi_probe_host, on)
        self.chi_form.setRowVisible(self._chi_buttons, on)
        if not on:
            self._chi_anchor = QLabel("(build a χ config to enable the probe table)")
            self.chi_form.addRow(self._chi_anchor)
            return
        # Seed only when EMPTY -- never top up, never trim. A user who deleted rows meant it.
        if not self._chi_forced_fields:
            for _ in range(max(1, min(int(cfg.chi_n_freqs), cfg.chi_k_pad))):
                self._add_chi_probe()

    def _plan_chi_probes(self):
        """Backlog C-3: say what is in band for THIS cell, and how long each probe must be recorded.

        Every predicate comes from chi.probe_verdict, the same function build_experiment_obs_chi
        refuses and masks on -- so this cannot tell the user one thing and the run another. That is
        the point of the exercise: these answers were previously only discoverable by running the
        inference, i.e. after the bench session rather than before it.

        The band is RELATIVE to the cell's own Ω₀, so nothing useful can be said until a passive
        recording exists. Measuring it needs one load and one FFT, which is why this is a button
        rather than something recomputed on every keystroke.
        """
        cfg = self.session.cfg
        if cfg is None or not cfg.chi_mode:
            return
        path = self.chi_spont.value()
        if not path:
            self.log_pane.append_line(
                "Select the passive recording first — Ω₀ is measured from it, and the χ band is "
                "defined relative to Ω₀, so there is nothing to plan without it.", "warning")
            return
        try:
            from core.SBI import chi as _chi
            x = file_manager.load_experimental_data(path, dtype=cfg.hw.dtype)
            f_peak = float(_chi.peak_freq(x.unsqueeze(0), cfg.dt_exp))
        except Exception as e:                                  # noqa: BLE001 -- a planner must never
            self.log_pane.append_line(f"Could not measure Ω₀ from {path}: {e}", "error")   # break the panel
            return
        hz = cfg.get_unit_conversion_factor("s")
        lo_hz, hi_hz = _chi.band_hz(cfg, f_peak)
        n_samp = max(1, int(round(self.chi_tobs.value() * hz / cfg.dt_exp)))
        self.log_pane.append_line(
            f"Ω₀ = {f_peak * hz:.4g} Hz for this recording. In band for this cell: "
            f"{lo_hz:.3g}–{hi_hz:.3g} Hz "
            f"({cfg.chi_freq_bounds[0]:g}–{cfg.chi_freq_bounds[1]:g}×Ω₀).")
        self.log_pane.append_line(
            f"At the band's low edge a probe needs ≥ {config.CHI_MIN_CYCLES / lo_hz:.3g} s to clear "
            f"the {config.CHI_MIN_CYCLES:g}-cycle floor; above "
            f"{cfg.chi_max_cycles / hi_hz:.3g} s the high edge is truncated to the "
            f"{cfg.chi_max_cycles:g}-cycle ceiling (which is fine — only the tail is dropped).")
        # Fill blank frequency boxes with the nominal in-band grid so the table is usable immediately.
        # Only BLANK ones: a typed frequency is a record of what the bench actually did.
        blanks = [r for r in self._chi_forced_fields if r.freq.value() <= 0]
        if blanks:
            grid = _chi.chi_multipliers(n_freqs=len(blanks), bounds=cfg.chi_freq_bounds).tolist()
            for row, mult in zip(blanks, grid):
                row.freq.setText(f"{mult * f_peak * hz:.4g}")
            self.log_pane.append_line(
                f"Filled {len(blanks)} blank frequency box(es) with a nominal log-spaced in-band "
                f"grid. These are SUGGESTIONS — replace each with the frequency you actually drove "
                f"at, because a lock-in decays like a sinc and a small mismatch destroys it.")
        # Now report each row's verdict against the T_obs entered.
        for i, row in enumerate(self._chi_forced_fields):
            f = row.freq.value()
            if not (math.isfinite(f) and f > 0):
                self.log_pane.append_line(f"  probe {i + 1}: no frequency entered.", "warning")
                continue
            v = _chi.probe_verdict(cfg, f_peak, f, n_samp)
            if v.action == "use":
                self.log_pane.append_line(
                    f"  probe {i + 1}: {f:g} Hz — OK, {v.cycles:.1f} drive cycles at "
                    f"T_obs = {self.chi_tobs.value():g} s.")
            else:
                self.log_pane.append_line(f"  probe {i + 1}: {f:g} Hz — {v.action.upper()}: "
                                          f"{v.reason}.", "warning" if v.action != "refuse" else "error")

    def _rebuild_forcing_fields(self, cfg):
        for fld in self._forcing_fields.values():
            self.exp_form.removeRow(fld)
        self._forcing_fields = {}
        if self._forcing_anchor is not None:
            self.exp_form.removeRow(self._forcing_anchor)
            self._forcing_anchor = None
        for name in cfg.force_params_dict:
            unit = cli.INFERENCE_PROMPT_UNITS.get(name, "")
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
            if not self.chi_spont.value():
                self.log_pane.append_line("Select the passive recording first — it sets Ω₀.",
                                          "warning")
                return
            if not self._chi_forced_fields:
                self.log_pane.append_line(
                    "Add at least one forced probe. χ mode conditions on a passive recording plus "
                    "any number of single-tone forced ones, but zero probes is a spontaneous "
                    "observation wearing a χ conditioning vector.", "warning")
                return
            problems = [p for i, r in enumerate(self._chi_forced_fields) for p in r.problems(i)]
            if problems:
                self.log_pane.append_line("Fix the probe table first: " + "; ".join(problems),
                                          "warning")
                return
            # (recording, drive frequency in Hz) PAIRS, never a bare path list. The core locks in at
            # the frequency it is TOLD, rather than assuming mult_k * Omega_0 -- the frequencies a
            # bench achieves are not exactly that, and a lock-in at the wrong frequency decays like a
            # sinc. Pairs come straight off each row widget, so they cannot be mismatched by an
            # add/remove in the middle of the table.
            pairs = [r.pair() for r in self._chi_forced_fields]
            self.dispatch(_run_experimental_inference_chi, cfg, post, self.chi_spont.value(), pairs,
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


# ── 6. TSNPE (truncated sequential NPE) ───────────────────────────────────────
class TSNPEPanel(_TrainingBudgetMixin, _StagePanel):
    """Tab 6. Refines the current posterior on ONE observation, without giving up the amortized one.

    ⚠⚠ WHAT THIS TAB MUST NOT BECOME. TSNPE proposes from the PRIOR RESTRICTED to an HPD region. It
    does NOT propose from the posterior. Fitting a density to the posterior and proposing from that
    gives ``p_L ∝ L^(L+1) q`` -- tempering -- and credible intervals then contract as
    ``(L+1)^(-1/2)`` with NO new information entering. SBC comes out flat anyway, because it validates
    the flow against the proposal it was trained on, so nothing on the Validate tab would catch it.
    The rule lives in ``core/SBI/truncate.py`` and is pinned by ``tests/test_conditioning_repair.py``.

    Gated on a posterior, the prior it was trained against, AND a persisted observation.
    An amortized posterior has no observation at SAVE time (``default_x`` is None on
    posterior_08232026), so the Infer tab records one at INFERENCE time and this tab keys on that.

    The budget group is the Posterior tab's, through ``_TrainingBudgetMixin``: a round is a simulation
    campaign, not a click, and the number belongs on screen before the button.

    Persists (group "inference_tsnpe"): the observation, the HPD level, the direction count and the
    two budget fields.
    """

    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        from core.SBI import truncate as _tr

        box = QGroupBox("TSNPE round")
        v = QVBoxLayout(box)
        warn = QLabel("Restricts the PRIOR to the posterior's credible region and retrains there. It "
                      "never proposes from the posterior itself. The result is NON-AMORTIZED and is "
                      "marked as such in its sidecar, so the load path will refuse it for general "
                      "inference.")
        warn.setWordWrap(True)
        warn.setTextFormat(Qt.PlainText)
        v.addWidget(warn)

        form = make_form()
        self.obs_picker = ArtifactPicker(OBSERVATION_PATH, keep=lambda fn: fn.endswith(".pt"))
        add_help_row(form, "Observation", self.obs_picker, HELP["tsnpe_obs"])
        self.hpd = FloatField(str(_tr.DEFAULT_HPD))
        self.n_dirs = IntField(str(_tr.DEFAULT_N_DIRECTIONS))
        add_help_row(form, "HPD level", self.hpd, HELP["tsnpe_hpd"])
        add_help_row(form, "Directions truncated", self.n_dirs, HELP["tsnpe_dirs"])
        v.addLayout(form)

        self.btn_round = QPushButton("Run TSNPE round")
        self.btn_round.setProperty("accent", True)         # primary CTA (Fluent accent)
        self.btn_round.clicked.connect(self._round)
        v.addWidget(self.btn_round)
        self.controls_layout.addWidget(box)

        # The SAME budget group the Posterior tab shows, and the same code behind it.
        budget = QGroupBox("Simulation budget for this round")
        bv = QVBoxLayout(budget)
        bform = make_form()
        self.num_runs = IntField(str(config.TRAINING_NUM_RUNS))
        self.run_size_cap = IntField(str(config.TRAINING_RUN_SIZE))
        add_help_row(bform, "Batches", self.num_runs, HELP["num_runs"])
        add_help_row(bform, "Max rows per batch (0 = auto)", self.run_size_cap, HELP["run_size"])
        bv.addLayout(bform)
        self.budget_total = self._derived_label()
        self.budget_mem = self._derived_label()
        self.budget_ckpt = self._derived_label()
        for lab in (self.budget_total, self.budget_mem, self.budget_ckpt):
            bv.addWidget(lab)
        for fld in (self.num_runs, self.run_size_cap):
            fld.textChanged.connect(lambda _t: self._sync_budget())
        self.controls_layout.addWidget(budget)
        self._sync_budget()

    def _round(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None or not self.obs_picker.key():
            return
        level, n_dirs = self.hpd.value(), self.n_dirs.value()
        if not (0.0 < level < 1.0):
            self.log_pane.append_line("HPD level must be strictly between 0 and 1.", "warning")
            return
        if n_dirs < 1:
            self.log_pane.append_line("At least one direction must be truncated.", "warning")
            return
        if level < 0.99:
            # A judgement, so it warns rather than refuses -- but deleted support is a ONE-WAY
            # ratchet, and a region that is too TIGHT is the expensive mistake, not the cheap one.
            self.log_pane.append_line(
                f"HPD {level:g} is tighter than the recommended {0.999:g}. Truncation permanently "
                f"deletes prior support; no later round can recover it.", "warning")
        n_runs, cap = self._budget_values()
        self.dispatch(_run_tsnpe_round, s.cfg, s.posterior, s.inf_prior, s.force_prior,
                      OBSERVATION_PATH / self.obs_picker.key(), n_dirs, level,
                      max(1, n_runs), max(0, cap), provide_fig_sink=True,
                      on_result=self._on_round)

    def _on_round(self, payload):
        """Install the round's posterior AND the region that makes it non-amortized.

        Without an on_result the round trains for hours and the result is discarded -- and without
        the region travelling with it, the deferred Save writes it marked amortized. Both halves
        matter; the second is the one that produces a wrong artifact rather than no artifact.
        """
        (posterior, diagnostics), region, digest = payload
        s = self.session
        s.posterior, s.diagnostics = posterior, diagnostics
        s.posterior_latent = getattr(posterior, "latent", None)
        s.V = PosteriorPanel._extract_rotation(posterior)
        s.truncation, s.x_obs_digest = region, digest
        self.log_pane.append_line(
            f"TSNPE round complete. This posterior is NON-AMORTIZED: it is valid near the "
            f"observation {digest}, and its sidecar will say so.", "warning")
        self._screen.refresh_gates()

    def refresh_local_gates(self):
        s = self.session
        self.obs_picker.refresh()
        self.btn_round.setEnabled(s.posterior is not None and s.inf_prior is not None
                                  and bool(self.obs_picker.key()))
        self._sync_budget()

    def save_settings(self, qs):
        qs.beginGroup("inference_tsnpe")
        qs.setValue("observation", self.obs_picker.key())
        qs.setValue("hpd", str(self.hpd.value()))
        qs.setValue("n_dirs", self.n_dirs.value())
        qs.setValue("num_runs", self.num_runs.value())
        qs.setValue("run_size_cap", self.run_size_cap.value())
        qs.endGroup()

    def restore_settings(self, qs):
        from core.SBI import truncate as _tr
        qs.beginGroup("inference_tsnpe")
        self.obs_picker.restore_key(settings.get_str(qs, "observation"))
        # get_str + float, because settings has no get_float and inventing one for a single caller
        # would be a wider change than this needs. A blank or unparseable value falls back to the
        # module default rather than to 0.0, which FloatField.value() would otherwise hand back --
        # and an HPD of 0 would truncate the prior to a point.
        try:
            self.hpd.setText(str(float(settings.get_str(qs, "hpd", str(_tr.DEFAULT_HPD)))))
        except ValueError:
            self.hpd.setText(str(_tr.DEFAULT_HPD))
        self.n_dirs.setText(str(settings.get_int(qs, "n_dirs", _tr.DEFAULT_N_DIRECTIONS)))
        self.num_runs.setText(str(settings.get_int(qs, "num_runs", config.TRAINING_NUM_RUNS)))
        self.run_size_cap.setText(str(settings.get_int(qs, "run_size_cap", config.TRAINING_RUN_SIZE)))
        qs.endGroup()
