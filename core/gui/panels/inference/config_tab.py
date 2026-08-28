import os

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout)

from core import cli, config, registry
from core.Helpers import file_manager
from core.SBI import pipeline
from core.config import CHI_K_MAX, VALID_LABELS, VALID_MODELS

from ... import icons, settings
from ...session import ConfigDraft
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from ...widgets.source_toggle import SourceToggle
from .rows import _ChiRangeRow
from .base import _StagePanel, _TrainingBudgetMixin, _nvidia_smi_free_gib
from .help_text import HELP


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
