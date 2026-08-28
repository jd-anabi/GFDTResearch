"""Composite row widgets for the inference tabs."""
import math

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ... import icons
from ...widgets.field_row import LabeledFieldRow
from ...widgets.labeled_inputs import FloatField, PathField


class _ChiRangeRow(LabeledFieldRow):
    """lo / hi multipliers of the measured spontaneous peak Ω₀ bounding the χ(ω) probe grid."""

    def __init__(self, lo: float, hi: float, parent=None):
        self.lo, self.hi = FloatField(lo), FloatField(hi)
        super().__init__((("×Ω₀ from", self.lo), ("to", self.hi)), parent=parent)

    def value(self) -> tuple:
        return self.values()


class _ChiProbeRow(QWidget):
    """ONE forced recording and the frequency it was ACTUALLY driven at, as a SINGLE widget.

    One object per probe is the entire point, not a layout convenience. Parallel path/frequency lists
    let a middle deletion pair recording *k* with frequency *k+1*, and that failure is invisible: a
    lock-in decays like a sinc, so a mismatch of a fraction of 1/T_obs destroys the estimate while
    every number on screen still looks reasonable. Deleting this widget deletes the pair, so the two
    cannot drift apart by construction.

    The frequency is entered, never derived. The frequencies a bench can actually achieve are not
    exactly ``mult_k * Omega_0``, and even aiming for them your Omega_0 estimate is not
    ``chi.peak_freq``'s -- different trace length, windowing, bin resolution. See
    orchestrator.build_experiment_obs_chi, which stopped guessing them for the same reason.
    """

    def __init__(self, on_remove, freq_hz: float = 0.0, parent=None):
        super().__init__(parent)
        self.path = PathField()
        self.freq = FloatField(freq_hz)
        self.freq.setMaximumWidth(96)
        self.btn_remove = QPushButton()
        self.btn_remove.setObjectName("iconButton")     # the QSS that owns icon-button size/colour
        icons.apply_icon(self.btn_remove, "close")      # bundled icon font; falls back to "✕"
        self.btn_remove.setMaximumWidth(32)
        self.btn_remove.setToolTip("Remove this probe")
        self.btn_remove.clicked.connect(lambda: on_remove(self))
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.path, 1)
        row.addWidget(QLabel("at"))
        row.addWidget(self.freq)
        row.addWidget(QLabel("Hz"))
        row.addWidget(self.btn_remove)

    def pair(self) -> tuple:
        return self.path.value(), self.freq.value()

    def problems(self, index: int) -> list:
        """Why this row cannot be run, phrased for a user. Structural only -- whether the probe is in
        band or long enough is chi.probe_verdict's job, and the planner reports it."""
        out = []
        if not self.path.value():
            out.append(f"probe {index + 1}: no recording selected")
        # FloatField.value() returns 0.0 on unparseable text, so a BLANK box is indistinguishable from
        # a deliberate zero unless it is checked here -- and 0 Hz is a genuine DC probe the lock-in
        # would happily attempt. This is the check that stops a typo becoming a measurement.
        f = self.freq.value()
        if not (math.isfinite(f) and f > 0):
            out.append(f"probe {index + 1}: drive frequency must be a positive number (got {f:g})")
        return out

