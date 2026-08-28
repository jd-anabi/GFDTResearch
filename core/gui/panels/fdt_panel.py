"""FDT mode: the fluctuation-dissipation-theorem analysis.

Mirrors cli.build_fdt_config (cli.make_fdt_config) with widgets, then runs FDT.fdt_pipeline.run_fdt on a
worker.

The two checkboxes are load-bearing. run_fdt's `skip_sanity` / `confirm_production` default to None,
which means "ask via input()" -- that is the CLI path (FDT.fdt_pipeline.run_fdt). A GUI MUST pass
explicit booleans; leaving them None would block the worker forever on an input() nobody can answer.
"""
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QPushButton

from core import cli, registry
from core.config import CELL_PATH, PLOT_PATH, VALID_MODELS
from core.FDT.campaigns import FDTModelError
from core.FDT.fdt_pipeline import run_fdt

from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.help_badge import add_help_row, with_badge
from ..widgets.labeled_inputs import FloatField, IntField
from ..widgets.forms import make_form

HELP = {
    "model": "Which model to analyse. FDT supports NADROWSKI, HOPF, BP (experimental), and "
             "additive-noise user-defined models. The frequency grid and burn-in are Nadrowski-tuned, "
             "so treat other models' calibration as your responsibility.",
    "cell": "A cell file whose parameters define the system whose fluctuation-dissipation relation is tested.",
    "n_freqs": "Number of (log-spaced) frequencies at which G(ω) and χ(ω) are evaluated.",
    "ensemble_M": "Ensemble size: independent trajectories averaged per frequency. More reduces noise at higher cost.",
    "freqs_per_batch": "How many frequencies to simulate per batch (a memory/speed trade-off; results are identical).",
    "f0": "Non-dimensional forcing amplitude used to probe the susceptibility χ(ω).",
    "skip_sanity": "Skip the passive-baseline sanity checks and go straight to the production sweep.",
    "confirm_production": "After the sanity checks, proceed to the (long) production sweep automatically.",
}


def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production):
    """Translate a model/cell FDT incompatibility into a readable message. FDTModelError (a missing FDT
    parameter, or a user model with multiplicative/zero observable noise) is already user-facing; the
    KeyError net is a defensive backstop for a malformed cell (it should no longer fire for HOPF/BP)."""
    try:
        return run_fdt(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production)
    except FDTModelError as e:
        raise RuntimeError(str(e)) from e
    except KeyError as e:
        raise RuntimeError(
            f"The FDT pipeline needs the parameter {e}, which the selected {cfg.model} cell does not "
            f"define."
        ) from e


class FdtPanel(BasePanel):
    """Drives a single-cell FDT analysis (``FDT.fdt_pipeline.run_fdt``).

    Passes explicit booleans for run_fdt's ``skip_sanity``/``confirm_production``: their None default
    means "ask via input()", which on a worker thread would block forever with no prompt to answer.

    Persists (group "fdt"): model, cell picker, and the campaign knobs. Restore order matters --
    model FIRST, then the pickers, or the model's refresh() wipes the restored picker.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_controls()
        self.restore_settings(settings.settings())

    def _build_controls(self):
        box = QGroupBox("FDT analysis")
        form = make_form(box)

        self.model_combo = QComboBox()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self.n_freqs = IntField(60)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)

        self.skip_sanity = QCheckBox("Skip sanity checks")
        self.confirm_production = QCheckBox("Proceed to the production sweep after sanity")
        self.confirm_production.setChecked(True)
        self.skip_sanity.toggled.connect(
            lambda on: self.confirm_production.setEnabled(not on))   # only consulted when sanity runs

        self.btn_run = QPushButton("Run FDT analysis")
        self.btn_run.setProperty("accent", True)          # primary CTA (Fluent accent)
        self.btn_run.clicked.connect(self._run)

        add_help_row(form, "Model", self.model_combo, HELP["model"])
        add_help_row(form, "Cell", self.cell_picker, HELP["cell"])
        add_help_row(form, "n_freqs", self.n_freqs, HELP["n_freqs"])
        add_help_row(form, "ensemble_M", self.ensemble_m, HELP["ensemble_M"])
        add_help_row(form, "freqs_per_batch", self.freqs_per_batch, HELP["freqs_per_batch"])
        add_help_row(form, "F0 (ND forcing amplitude)", self.f0, HELP["f0"])
        form.addRow(with_badge(self.skip_sanity, HELP["skip_sanity"]))
        form.addRow(with_badge(self.confirm_production, HELP["confirm_production"]))
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)

    def _on_model_changed(self, model: str):
        self.cell_picker.repoint(CELL_PATH / model.lower())
        # FDT supports the built-ins + additive-noise, no-forcing user models. Gate the CTA on
        # registry.fdt_support and show the reason (multiplicative/zero-noise or forced user models).
        ok, reason = registry.fdt_support(model)
        self.btn_run.setEnabled(ok)
        if not ok:
            self.log_pane.append_line(reason, "warning")

    def _run(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            self.log_pane.append_line("Select a cell file first.", "warning")
            return
        model = self.model_combo.currentText()
        ok, reason = registry.fdt_support(model)          # backstop; the CTA is already disabled
        if not ok:
            self.log_pane.append_line(reason, "warning")
            return
        try:
            cfg = cli.make_fdt_config(
                model, registry.state_dep_drift(model), cell,
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value())
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return

        # Explicit bools, never None -- see the module docstring.
        self.dispatch(_run_fdt_guarded, cfg, watch_dir=PLOT_PATH,
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_finished=lambda: self.log_pane.append_line("FDT run finished."))

    def save_settings(self, qs):
        qs.beginGroup("fdt")
        qs.setValue("model", self.model_combo.currentText())
        qs.setValue("cell", self.cell_picker.key())
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.save_field(qs, name, fld)
        settings.set_bool(qs, "skip_sanity", self.skip_sanity.isChecked())
        settings.set_bool(qs, "confirm_production", self.confirm_production.isChecked())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("fdt")
        # Model FIRST, explicit _on_model_changed, THEN the picker key -- ArtifactPicker.repoint owns
        # the rationale. Feed the combo's ACCEPTED text, not the raw saved string: a corrupt saved
        # model a non-editable combo rejects would otherwise point the picker at a nonexistent folder.
        self.model_combo.setCurrentText(settings.get_str(qs, "model", self.model_combo.currentText()))
        self._on_model_changed(self.model_combo.currentText())
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.restore_field(qs, name, fld)
        self.skip_sanity.setChecked(settings.get_bool(qs, "skip_sanity", False))
        self.confirm_production.setChecked(settings.get_bool(qs, "confirm_production", True))
        qs.endGroup()
