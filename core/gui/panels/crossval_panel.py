"""CROSSVAL mode: the FDT parameter-sweep study.

Two sweeps probe FDT restoration on the Nadrowski model (see cli.make_param_sweep_config):
  * S sweep  (T_a/T held at 1): vary S;      FDT is restored as S -> 0.
  * T sweep  (S held at 0):     vary T_a/T;  FDT is restored as T_a/T -> 1.

Mirrors cli.build_param_sweep_config with widgets, then runs the prompt-free
FDT.cross_validation.run_param_study_cli on a worker. Model is fixed to NADROWSKI.
"""
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QWidget)

from core import cli
from core.config import CELL_PATH, PLOT_PATH
from core.FDT.cross_validation import run_param_study_cli

from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.help_badge import add_help_row
from ..widgets.labeled_inputs import FloatField, IntField
from ..widgets.field_row import LabeledFieldRow
from ..widgets.forms import make_form

_MODEL = "NADROWSKI"

HELP = {
    "cell": "A cell whose S and T_a/T set the far end of each sweep (the FDT-violating extreme).",
    "preset": "exploratory = quick/coarse; production = finer grids and more frequencies.",
    "s_grid": "S sweep (T_a/T fixed at 1): FDT is restored as S → 0. min / max / number of points.",
    "t_grid": "T_a/T sweep (S fixed at 0): FDT is restored as T_a/T → 1. min / max / number of points.",
    "n_freqs": "Number of (log-spaced) frequencies evaluated per sweep point.",
    "ensemble_M": "Ensemble size: independent trajectories averaged per frequency.",
    "freqs_per_batch": "How many frequencies to simulate per batch (memory/speed; results identical).",
    "f0": "Non-dimensional forcing amplitude used to probe χ(ω) at each sweep point.",
}


class _GridRow(LabeledFieldRow):
    """min / max / n_points for one sweep axis."""

    def __init__(self, lo: float, hi: float, points: int, parent=None):
        self.lo, self.hi, self.points = FloatField(lo), FloatField(hi), IntField(points)
        super().__init__((("min", self.lo), ("max", self.hi), ("n", self.points)), parent=parent)

    def spec(self) -> tuple:
        return self.values()


class CrossValPanel(BasePanel):
    """Drives the FDT parameter-sweep study (``FDT.cross_validation.run_param_study_cli``).

    Model is fixed to NADROWSKI. Two sweeps -- S with T_a/T held at 1, and T_a/T with S held at 0 --
    each running from the FDT-restoring limit to the selected cell's own value.

    Persists (group "crossval"): the cell picker, the preset, the free knobs (n_freqs, ensemble_M,
    freqs_per_batch, F0) and each grid's point COUNT. Deliberately NOT the grids' lo/hi: those are
    re-derived from the cell, so a value saved against a different cell would be a stale bound.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_controls()
        self._on_preset_changed(self.preset_combo.currentText())
        self._on_cell_changed()
        self.restore_settings(settings.settings())

    def _build_controls(self):
        box = QGroupBox("FDT parameter-sweep study")
        form = make_form(box)

        self.cell_picker = ArtifactPicker(CELL_PATH / _MODEL.lower())
        self.cell_picker.combo.currentIndexChanged.connect(self._on_cell_changed)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(cli.SWEEP_PRESETS))          # exploratory | production
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

        self.cell_values = QLabel("—")
        # This label receives an UNBOUNDED str(e) from _on_cell_changed's broad except. Without wrap
        # it forced the whole controls column wider than any sensible width, which used to trigger
        # the permanent horizontal scrollbar on a panel that otherwise fit.
        self.cell_values.setWordWrap(True)
        self.s_grid = _GridRow(0.0, 1.0, 8)
        self.t_grid = _GridRow(1.0, 1.0, 8)
        self.n_freqs = IntField(30)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)

        self.btn_run = QPushButton("Run sweep study")
        self.btn_run.setProperty("accent", True)          # primary CTA (Fluent accent)
        self.btn_run.clicked.connect(self._run)

        form.addRow(QLabel(f"Model is fixed to {_MODEL}."))
        add_help_row(form, "Cell", self.cell_picker, HELP["cell"])
        form.addRow("Cell values", self.cell_values)          # an output display, not a configurable input
        add_help_row(form, "Preset", self.preset_combo, HELP["preset"])
        add_help_row(form, "S grid  (T_a/T = 1)", self.s_grid, HELP["s_grid"])
        add_help_row(form, "T_a/T grid  (S = 0)", self.t_grid, HELP["t_grid"])
        add_help_row(form, "n_freqs", self.n_freqs, HELP["n_freqs"])
        add_help_row(form, "ensemble_M", self.ensemble_m, HELP["ensemble_M"])
        add_help_row(form, "freqs_per_batch", self.freqs_per_batch, HELP["freqs_per_batch"])
        add_help_row(form, "F0 (ND forcing amplitude)", self.f0, HELP["f0"])
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)

    # ── prefill, exactly as the CLI does (cli.build_param_sweep_config's prefill) ───────────────
    def _on_cell_changed(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            return
        try:
            _i, params, _r, _f, _u, _si, _s = cli.parse_cell(cell, model=_MODEL)
        except Exception as e:                       # noqa: BLE001
            # Deliberately broad. __init__ calls this, so ANY exception here escapes CrossValPanel()
            # -> MainWindow() -> build_app() and the whole GUI fails to launch -- and app.py installs
            # its excepthook only AFTER MainWindow() is built, so nothing would even show it. A cell
            # with no sibling Bounds/<model>/<name>.txt is enough to trigger it: parse_cell then takes
            # the legacy branch and raises a plain ValueError, not UnitParseError. A bad cell file must
            # degrade this one label, never brick the app.
            self.cell_values.setText(f"(could not read cell: {e})")
            return
        cell_s = float(params["s"][0])
        cell_temp = float(params["temp"][0])
        self.cell_values.setText(f"S = {cell_s:.4f},  T_a/T = {cell_temp:.4f}")
        # The sweeps run FROM the FDT-restoring limit TO the cell's own value.
        self.s_grid.lo.setText("0.0")
        self.s_grid.hi.setText(f"{cell_s:g}")
        self.t_grid.lo.setText("1.0")
        self.t_grid.hi.setText(f"{cell_temp:g}")

    def _on_preset_changed(self, name: str):
        preset = cli.SWEEP_PRESETS[name]
        self.n_freqs.setText(str(preset["n_freqs"]))
        self.ensemble_m.setText(str(preset["ensemble_M"]))
        self.s_grid.points.setText(str(preset["points"]))
        self.t_grid.points.setText(str(preset["points"]))

    def _run(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            self.log_pane.append_line("Select a cell file first.", "warning")
            return
        preset = dict(cli.SWEEP_PRESETS[self.preset_combo.currentText()])
        try:
            cfg, s_grid, temp_grid = cli.make_param_sweep_config(
                cell, preset=preset, s_spec=self.s_grid.spec(), t_spec=self.t_grid.spec(),
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value())
        except Exception as e:                       # noqa: BLE001 -- see _config_error
            self._config_error(e)
            return

        # run_param_study_cli returns the two HDF5 DATA paths, not the figures -- the plots are saved
        # to disk (the S-sweep one at the study's midpoint, deliberately) and arrive via the watcher.
        self.dispatch(run_param_study_cli, cfg, s_grid, temp_grid, watch_dir=PLOT_PATH,
                      on_result=self._on_result)

    def _on_result(self, paths):
        if not paths:
            return
        for path in paths:
            self.log_pane.append_line(f"Sweep data: {path}")

    def save_settings(self, qs):
        qs.beginGroup("crossval")
        qs.setValue("preset", self.preset_combo.currentText())
        qs.setValue("cell", self.cell_picker.key())
        settings.save_field(qs, "f0", self.f0)
        settings.save_field(qs, "freqs_per_batch", self.freqs_per_batch)
        settings.save_field(qs, "s_points", self.s_grid.points)
        settings.save_field(qs, "t_points", self.t_grid.points)
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("crossval")
        # Order: preset first (it overwrites n_freqs/ensemble_M and the grid `points`), then cell (its
        # currentIndexChanged fires _on_cell_changed, which sets the grid lo/hi from the cell file).
        self.preset_combo.setCurrentText(settings.get_str(qs, "preset", self.preset_combo.currentText()))
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        # Restore ONLY the freely-set knobs -- and after the cell, so a saved `points` survives. Do NOT
        # restore the grid lo/hi: those are re-derived from the cell (cli.build_param_sweep_config's prefill), and a saved
        # value from a DIFFERENT cell would be a stale, wrong bound.
        settings.restore_field(qs, "f0", self.f0)
        settings.restore_field(qs, "freqs_per_batch", self.freqs_per_batch)
        settings.restore_field(qs, "s_points", self.s_grid.points)
        settings.restore_field(qs, "t_points", self.t_grid.points)
        qs.endGroup()
