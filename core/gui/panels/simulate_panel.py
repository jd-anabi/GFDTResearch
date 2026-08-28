"""SIMULATE mode: a real-time streaming view of a cell's hair-bundle motion.

Pick a model + cell + observation time and press Start; a worker-thread chunked Euler-Maruyama loop
(``simulate_runner.run_simulation_stream``) streams the redimensionalized hair-bundle displacement one
frame at a time, and a pyqtgraph ``LiveHairBundleView`` renders a scrolling trace + a top-down heatmap
on the GUI thread. The stream is finite (bounded by T_obs) and the shared Cancel button stops it early.

This panel reuses BasePanel's dispatch rails verbatim; the only new machinery is dispatch(provide_stream)
(which injects the chunk emitter + a stop-flag) and the live view mounted above the (hidden) figure
stack. Like every panel, a run here holds the app-wide single-task guard, so all controls lock until it
ends or is cancelled.
"""
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QGroupBox, QLabel, QPushButton)

from core.config import CELL_PATH, DT_EXP_S, T_MIN_EXP_S, VALID_MODELS

from .base_panel import BasePanel
from .simulate_export import (estimate_frame_count, export_animation, export_stride, ffmpeg_available)
from .simulate_runner import build_stream_config, run_simulation_stream
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.help_badge import add_help_row
from ..widgets.labeled_inputs import FloatField, IntField
from ..widgets.live_hair_bundle import LiveHairBundleView
from ..widgets.forms import make_form

HELP = {
    "model": "Which hair-cell model to simulate. The cell list follows this choice.",
    "cell": "A cell file (Resources/Cells/<model>) whose ground-truth parameters drive the simulation. "
            "Its sibling bounds file (Resources/Bounds/<model>) is used to validate the values.",
    "tobs": "How much observation time (seconds) to stream before the run stops. Longer = a longer run.",
    "frame": "How far the simulation jumps per on-screen frame. Bigger = faster but choppier playback "
             "and more compute per frame; smaller = smoother but slower. (Snapped to a whole number of "
             "display samples.)",
    "fps": "Maximum render frames per second. The worker paces itself to this so the trace scrolls at a "
           "watchable rate.",
}


class SimulatePanel(BasePanel):
    """Drives a live streaming simulation, plotted frame-by-frame, with video export.

    The only panel that replaces the static figure stack with its own primary view: a live pyqtgraph
    trace, mounted through ``BasePanel.insert_result_widget`` so it sits inside the results splitter.
    Runs batch-1 on CPU (``cfg.hw = cpu_device()``) -- the per-frame loop is CPU-optimal and every
    tensor must share one device.

    Persists (group "simulate"): model, cell picker, and the frame/duration knobs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._record = []          # emitted (k,2) [t,x] frames of the last/current run, for video export
        # The live view is this panel's primary surface: mount it above the (unused) static figure stack.
        self.live_view = LiveHairBundleView()
        # Via the BasePanel seam: the results column is a QSplitter now, so inserting into
        # right_layout would put this outside it (and give it no drag handle).
        self.insert_result_widget(0, self.live_view, stretch=5)
        self.figure_stack.setVisible(False)
        self._build_controls()
        self.restore_settings(settings.settings())

    def _build_controls(self):
        box = QGroupBox("Live simulation")
        form = make_form(box)

        self.model_combo = QComboBox()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.tobs = FloatField(T_MIN_EXP_S)
        self.frame_steps = IntField(2000)
        self.fps = IntField(30)
        self.btn_start = QPushButton("Start streaming")
        self.btn_start.setProperty("accent", True)        # primary CTA (Fluent accent)
        self.btn_start.clicked.connect(self._start)
        self.btn_save_video = QPushButton("Save video…")
        self.btn_save_video.setToolTip("Save the last run as an animation (.mp4 or .gif)")
        self.btn_save_video.setEnabled(False)
        self.btn_save_video.clicked.connect(self._save_video)

        add_help_row(form, "Model", self.model_combo, HELP["model"])
        add_help_row(form, "Cell", self.cell_picker, HELP["cell"])
        add_help_row(form, "T_obs (s)", self.tobs, HELP["tobs"])
        add_help_row(form, "Steps / frame", self.frame_steps, HELP["frame"])
        add_help_row(form, "Max FPS", self.fps, HELP["fps"])
        form.addRow(QLabel("Streaming holds the app single-task until it finishes or you Cancel."))
        form.addRow(self.btn_start)
        form.addRow(self.btn_save_video)
        self.controls_layout.addWidget(box)

    def _on_model_changed(self, model: str):
        self.cell_picker.repoint(CELL_PATH / model.lower())

    def _start(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            self.log_pane.append_line("Select a cell file first.", "warning")
            return
        model = self.model_combo.currentText()
        try:
            cfg = build_stream_config(model, cell)
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return
        self.live_view.set_displacement_unit(cfg.length_unit or "nm")   # label the trace y-axis in cell units
        self.live_view.reset()
        self._record = []                                    # a new run: drop the previous recording
        self.log_pane.append_line(
            f"Streaming {model} — {Path(cell).name} for {self.tobs.value():g} s of observation…")
        self.dispatch(run_simulation_stream, cfg, self.tobs.value(),
                      max(1, self.frame_steps.value()), float(max(1, self.fps.value())),
                      provide_stream=True, on_chunk=self._on_chunk,
                      on_result=lambda _r: self.log_pane.append_line("Simulation complete."))

    def _on_chunk(self, chunk):
        """Render the frame AND record it (a fresh (k,2) array per frame) for a later video export."""
        self.live_view.push(chunk)
        self._record.append(np.asarray(chunk))

    def _save_video(self):
        if not self._record:
            self.log_pane.append_line("Run a stream first, then save it as a video.", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save animation", "", "MP4 video (*.mp4);;GIF (*.gif)")
        if not path:
            return
        if not path.lower().endswith((".mp4", ".gif")):
            path += ".mp4"
        if path.lower().endswith(".mp4") and not ffmpeg_available():
            self._config_error(RuntimeError(
                "Saving MP4 needs an ffmpeg binary, which isn't available here. Save as .gif instead, "
                "or install ffmpeg (or set the IMAGEIO_FFMPEG_EXE environment variable)."))
            return

        series = np.concatenate(self._record, axis=0)
        fps = float(max(1, self.fps.value()))
        n_frames = estimate_frame_count(len(series), export_stride(1.0 / DT_EXP_S, fps))
        if n_frames > 600:
            self.log_pane.append_line(
                f"Exporting ~{n_frames} frames — this may be slow"
                f"{' and produce a large GIF' if path.lower().endswith('.gif') else ''}.", "warning")

        v = self.live_view
        self.log_pane.append_line(f"Exporting {n_frames} frames to {Path(path).name}…")
        self.dispatch(export_animation, series, path,
                      window_pts=v._w, grid_x=v._grid_x, grid_y=v._grid_y,
                      sigma_x=v._sig_x, sigma_y=v._sig_y, aspect=v._aspect, margin=v._margin,
                      video_fps=fps, x_unit=v._x_unit,
                      on_result=lambda p: self.log_pane.append_line(f"Saved animation to {p}."))

    def refresh_local_gates(self):
        self.btn_start.setEnabled(self.cell_picker.has_entries())
        self.btn_save_video.setEnabled(bool(self._record))

    def save_settings(self, qs):
        qs.beginGroup("simulate")
        qs.setValue("model", self.model_combo.currentText())
        qs.setValue("cell", self.cell_picker.key())
        settings.save_field(qs, "tobs", self.tobs)
        settings.save_field(qs, "frame_steps", self.frame_steps)
        settings.save_field(qs, "fps", self.fps)
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("simulate")
        # Model FIRST, explicit _on_model_changed, THEN the picker key -- ArtifactPicker.repoint
        # owns the rationale.
        self.model_combo.setCurrentText(settings.get_str(qs, "model", self.model_combo.currentText()))
        self._on_model_changed(self.model_combo.currentText())
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        settings.restore_field(qs, "tobs", self.tobs)
        settings.restore_field(qs, "frame_steps", self.frame_steps)
        settings.restore_field(qs, "fps", self.fps)
        qs.endGroup()
