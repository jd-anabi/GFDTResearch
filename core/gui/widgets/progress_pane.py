"""Live progress: a solver line, an overall bar with a spinner, and one row per active tqdm bar.

    Solver Performance: ++++  (13.3k it/s)   [▓▓▓▓▓▓▓░░░░░░]
    [███████░░░░░░░░░░░░░  38%]  ⠹
        Training neural posterior
        Generating training data  —  1902/5000 [05:12<13:41]

Two things this widget exists to solve.

WHY PROGRESS DOES NOT SHARE THE LOG PANE. The pipeline nests bars up to three deep, and inlining
several redrawing bars into a scrolling text widget is what produced the append storm this replaces.
Each bar owns a row, keyed by its tqdm `pos`, and simply overwrites it.

WHY THE SOLVER GETS ITS OWN LINE INSTEAD OF A ROW. A top-level iteration ("Generating training data")
takes ~10 seconds, so the overall bar sits still and the GUI reads as frozen. The thing that IS moving
is the SDE solver underneath it (core/Solvers/sdeint.py), and its `it/s` is the number the user wants.
But it cannot be a row: a posterior build constructs 10k-30k of those bars -- one per time segment,
each alive 1-10s -- so a row would mean creating and destroying a widget every few seconds. It gets one
fixed line instead, whose rate is HELD across the gaps between bars.
"""
import math
import time
from collections import deque

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QPainter, QPalette, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget)

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPIN_MS = 100

# A solver bar shorter than its mininterval (1.0s) paints only a "?it/s" opening frame, and with
# leave=False there is no final 100% frame either -- so rate samples arrive in gaps. Hold the last one
# for this long before calling the solver idle. Comfortably longer than the ~1s between rate frames,
# and short enough that neural-network training (where the solver genuinely is not running) reads idle.
SOLVER_IDLE_S = 45.0

# No output of ANY kind for this long means the run is probably wedged. Say so, instead of spinning
# cheerfully at a corpse.
STALL_S = 45.0


def plus_meter(rate: float | None) -> str:
    """One '+' per order of magnitude of iterations/sec: 10 -> '+', 10_000 -> '++++'.

    Under 10 it/s there is no order of magnitude to show, so render a '·' rather than an empty string --
    otherwise the line reads as broken rather than as slow.
    """
    if rate is None or rate <= 0:
        return "—"
    n = max(0, min(9, int(math.floor(math.log10(rate)))))
    return "+" * n if n else "·"


def format_rate(rate: float | None) -> str:
    if rate is None or rate <= 0:
        return "idle"
    if rate >= 1000:
        return f"{rate / 1000:.1f}k it/s"
    if rate >= 10:
        return f"{rate:.0f} it/s"
    return f"{rate:.1f} it/s"


class _Sparkline(QWidget):
    """A tiny live trend line of the SDE solver's throughput -- the primary at-a-glance solver readout.

    Fed one sample per 100 ms tick with the HELD solver rate; ``None`` marks a gap (the solver is idle or
    the run has stalled), which breaks the line so a pause reads as a pause rather than a flat crawl. Drawn
    on a log10 scale, so a 10 it/s -> 10k it/s ramp is a gentle slope, not a cliff. Colours come from the
    palette (Highlight for the trace, Mid for the baseline), so it reads correctly in light and dark.
    """

    def __init__(self, capacity: int = 64, parent=None):
        super().__init__(parent)
        self._capacity = capacity
        self._samples: deque = deque([None] * capacity, maxlen=capacity)
        self.setFixedSize(88, 18)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setToolTip("Recent SDE solver throughput (log it/s)")

    def push(self, rate) -> None:
        """Append one sample (a positive it/s, or None for a gap) and repaint just this widget."""
        self._samples.append(rate if (rate is not None and rate > 0) else None)
        self.update()

    def clear(self) -> None:
        self._samples = deque([None] * self._capacity, maxlen=self._capacity)
        self.update()

    def paintEvent(self, _event) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        vals = list(self._samples)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        pad = 2.0
        mid_y = h - pad
        # A baseline so an all-gap (idle) sparkline still reads as "present, nothing to show".
        p.setPen(QPen(self.palette().color(QPalette.Mid), 1.0))
        p.drawLine(QPointF(pad, mid_y), QPointF(w - pad, mid_y))

        finite = [v for v in vals if v is not None]
        if len(finite) < 2:
            p.end()
            return

        logs = [math.log10(v) for v in finite]
        lo, hi = min(logs), max(logs)
        span = hi - lo
        n = len(vals)
        plot_w = w - 2 * pad
        plot_h = h - 2 * pad

        def point(i: int, v: float) -> QPointF:
            x = pad + plot_w * (i / (n - 1))
            frac = 0.5 if span < 1e-9 else (math.log10(v) - lo) / span
            return QPointF(x, pad + plot_h * (1.0 - frac))

        pen = QPen(self.palette().color(QPalette.Highlight))
        pen.setWidthF(1.5)
        p.setPen(pen)
        prev = None                                  # break the polyline across None gaps
        for i, v in enumerate(vals):
            if v is None:
                prev = None
                continue
            cur = point(i, v)
            if prev is not None:
                p.drawLine(prev, cur)
            prev = cur
        p.end()


class _BarRow(QWidget):
    """One tqdm bar: a label for the description and a bar for the percentage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel()
        # Pin the size policy: QLabel.setText() calls updateGeometry() unconditionally, so an
        # unpinned label re-lays-out the whole right-hand pane on every frame (~60/s across all
        # rows), visibly jittering the figure/log splitter as the frame text changes width.
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.label.setTextFormat(Qt.PlainText)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(1)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)

    def update_from(self, state) -> None:
        depth = "    " * state.row
        stats = f"  —  {state.stats}" if state.stats else ""
        self.label.setText(f"{depth}{state.desc}{stats}")
        if state.pct is None:
            self.bar.setRange(0, 0)              # indeterminate: this bar has no total
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(state.pct)


class ProgressPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: dict[tuple, _BarRow] = {}
        self._rate: float | None = None
        self._rate_at = 0.0
        self._beat_at = 0.0
        self._frame = 0

        # ── solver line: the rate meter + the live step strip ────────────────
        self.solver_label = QLabel()
        self.solver_label.setTextFormat(Qt.PlainText)
        self.solver_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # The sparkline is the primary at-a-glance readout; the '+'-meter in solver_label is kept as a
        # compact secondary. Must be constructed before _reset_solver() below (it clears the sparkline).
        self.sparkline = _Sparkline()

        self.solver_strip = QProgressBar()
        self.solver_strip.setTextVisible(False)
        self.solver_strip.setFixedHeight(8)
        self.solver_strip.setToolTip("Progress through the current SDE integration segment")

        solver_line = QWidget()
        solver_layout = QHBoxLayout(solver_line)
        solver_layout.setContentsMargins(0, 0, 0, 0)
        solver_layout.setSpacing(8)
        solver_layout.addWidget(self.solver_label)
        solver_layout.addWidget(self.sparkline)
        solver_layout.addWidget(self.solver_strip, 1)

        # ── overall bar + spinner ───────────────────────────────────────────
        self.overall = QProgressBar()
        self.overall.setRange(0, 0)              # indeterminate until something reports a percentage
        self.overall.setTextVisible(True)        # show the top-level % right on the bar
        self.overall.setFormat("%p%")
        self.overall.setMinimumHeight(18)

        self.spinner = QLabel()
        self.spinner.setTextFormat(Qt.PlainText)
        # Fixed width: the spinner cycles glyphs and swaps to a stall message, and an unpinned label
        # would re-lay-out the pane on every 100ms tick.
        #
        # setFIXEDWidth, not setMinimumWidth. The comment above always claimed a pinned width, but a
        # MINIMUM is not one: with QSizePolicy.Fixed the widget takes its sizeHint(), which grows
        # when _tick swaps in the longer stall message -- so the pane re-laid-out exactly as the
        # comment said it must not. setFixedWidth makes the claim true.
        self.spinner.setFixedWidth(150)
        self.spinner.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        overall_line = QWidget()
        overall_layout = QHBoxLayout(overall_line)
        overall_layout.setContentsMargins(0, 0, 0, 0)
        overall_layout.setSpacing(8)
        overall_layout.addWidget(self.overall, 1)
        overall_layout.addWidget(self.spinner)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.setSpacing(2)
        self._layout.addWidget(solver_line)
        self._layout.addWidget(overall_line)

        # Drives the spinner AND the two timeouts -- staleness has to be noticed precisely when no
        # events are arriving, so it cannot be evaluated from set_rows() alone.
        self._timer = QTimer(self)
        self._timer.setInterval(_SPIN_MS)
        self._timer.timeout.connect(self._tick)

        self._reset_solver()
        self.setVisible(False)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def begin(self) -> None:
        self.end()
        self.heartbeat()
        self.setVisible(True)
        self._timer.start()

    def end(self) -> None:
        """Authoritative teardown. Deletes every row regardless of whether its bar reported a close,
        so a row leaked by a crashed worker cannot survive into the next dispatch."""
        self._timer.stop()
        for row in self._rows.values():
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self.overall.setRange(0, 0)
        self.spinner.setText("")
        self._reset_solver()
        self.setVisible(False)

    def heartbeat(self) -> None:
        """Mark 'the run produced output just now'. Called on every rows snapshot AND on every batch of
        log lines -- a run that is printing but not drawing bars is still alive."""
        self._beat_at = time.monotonic()

    # ── the worker's rows signal lands here ──────────────────────────────────
    def set_rows(self, snapshot) -> None:
        """`snapshot` is the pump's full set of live rows (a tuple[RowState]), already sorted."""
        self.heartbeat()

        solver = next((s for s in snapshot if s.is_solver), None)
        rows = [s for s in snapshot if not s.is_solver]

        seen = set()
        for state in rows:
            seen.add(state.key)
            row = self._rows.get(state.key)
            if row is None:
                row = _BarRow(self)
                self._rows[state.key] = row
                self._layout.addWidget(row)
            row.update_from(state)

        for key in [k for k in self._rows if k not in seen]:
            row = self._rows.pop(key)
            self._layout.removeWidget(row)
            row.deleteLater()

        self._update_solver(solver)
        self._retarget(rows)

    # ── the solver line ──────────────────────────────────────────────────────
    def _reset_solver(self) -> None:
        self._rate = None
        self._rate_at = 0.0
        self.solver_strip.setRange(0, 100)
        self.solver_strip.setValue(0)
        self.solver_strip.setVisible(False)
        self.sparkline.clear()
        self._paint_solver()

    def _update_solver(self, solver) -> None:
        if solver is None:
            self.solver_strip.setVisible(False)      # no solver running (e.g. NN training, Reduction)
        else:
            self.solver_strip.setVisible(True)
            if solver.pct is None:
                self.solver_strip.setRange(0, 0)
            else:
                self.solver_strip.setRange(0, 100)
                self.solver_strip.setValue(solver.pct)
            if solver.rate is not None:              # None on the opening frame of every bar
                self._rate, self._rate_at = solver.rate, time.monotonic()
        self._paint_solver()

    def _paint_solver(self) -> None:
        rate = self._rate
        if rate is not None and time.monotonic() - self._rate_at > SOLVER_IDLE_S:
            rate = None                              # held sample has gone stale: the solver stopped
        self.solver_label.setText(f"Solver Performance: {plus_meter(rate)}  ({format_rate(rate)})")
        if rate is None:
            self.solver_label.setToolTip("The SDE solver is not running right now.")
        else:
            self.solver_label.setToolTip(
                f"SDE solver: {rate:,.0f} integration steps/sec.\nOne '+' per order of magnitude.")

    # ── the spinner + stall detection ────────────────────────────────────────
    def _tick(self) -> None:
        idle = time.monotonic() - self._beat_at
        if idle > STALL_S:
            # Freeze the spinner rather than animate it. A spinner that keeps twirling on a wedged run
            # is worse than none: it actively asserts progress that is not happening.
            stall = f"⏳ no output for {int(idle)}s"
            self.spinner.setText(stall)
            self.spinner.setToolTip(stall)      # the width is pinned now, so long text can clip
            self.sparkline.push(None)                # and drain the trend line: nothing is happening
            return
        self._frame = (self._frame + 1) % len(_SPINNER)
        self.spinner.setText(_SPINNER[self._frame])
        self._paint_solver()                         # re-evaluate the solver's own idle timeout
        self._sample_rate()                          # feed the sparkline on the same 100ms cadence

    def _sample_rate(self) -> None:
        """Feed the sparkline the current HELD solver rate -- None once it has gone stale, i.e. a gap."""
        rate = self._rate
        if rate is not None and time.monotonic() - self._rate_at > SOLVER_IDLE_S:
            rate = None
        self.sparkline.push(rate)

    def _retarget(self, rows) -> None:
        """Drive the overall bar from the DEEPEST live row that reports an informative percentage.

        `rows` EXCLUDES the solver bar, and must: the solver has a total in the tens of thousands and is
        the deepest bar there is, so it would win this election every time and drag the overall bar
        through a full 0->100% sweep every second or two. The overall bar's job is the top-level count.

        Nor the outermost, and nor a sticky first-seen "driver" -- both of those peg the bar:
          * the pos-0 bar ("Training neural posterior", pipeline.train_nn) wraps
            range(TRAINING_NUM_ROUNDS) and that is 1 (config.TRAINING_NUM_ROUNDS), so it reads 0% for the
            entire multi-hour build -- hence RowState.informative excludes total<=1 bars;
          * sbi's neural-network training emits no tqdm bar at all (only a printed epoch counter), so
            a driver latched onto "Generating training data" would sit at 100% through the longest
            phase, which reads as finished or hung.
        """
        live = sorted((s for s in rows if s.informative), key=lambda s: s.row, reverse=True)
        if not live:
            self.overall.setRange(0, 0)
        else:
            self.overall.setRange(0, 100)
            self.overall.setValue(live[0].pct)
