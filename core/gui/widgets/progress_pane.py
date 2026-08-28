"""Live progress: ONE overall bar, one caption naming the stage, and a solver rate with no bar.

    Solver Performance: +++++  (142.4k it/s)   [sparkline]
    [███████░░░░░░░░░░░░░  38%]  ⠹
    Generating training data  —  1902/5000 [05:12<13:41]

WHY PROGRESS DOES NOT SHARE THE LOG PANE. The pipeline nests bars up to three deep, and inlining
several redrawing bars into a scrolling text widget is what produced the append storm this replaces.

WHY THERE IS ONE BAR AND NOT ONE ROW PER TQDM BAR. There are 17 tqdm sites in core/, in seven
families, and this pane used to render a row for each live one -- a stack of bars mostly saying the
same thing at different granularities, plus rows like `Training neural posterior -- 0/1` that carry
no information at all. The nest is still parsed in full (core/gui/vt.py); it is just rendered as one
bar plus one caption. The whole live set is on the overall bar's TOOLTIP, one hover away.

    ⚠ THE CAPTION IS NOT DECORATION -- IT IS WHAT KEEPS THE LONGEST PHASE OF A RUN VISIBLE.
    sbi's neural-network training emits NO tqdm bar at all: it prints "\\r" + "Training neural
    network. Epochs trained: N", which vt turns into an overwrite-mode row with pct=None.
    During that phase -- hours of a multi-day build -- the only live rows are that status line and a
    degenerate total=1 bar. With no caption the pane would show an indeterminate bar and nothing
    else. Hence the fallback in _paint_caption: when nothing reports a percentage, show the deepest
    row whose total is not 1.

WHY THE SOLVER GETS ITS OWN LINE, AND WHERE ITS NUMBER COMES FROM. The thing that is really moving
during a build is the SDE solver (core/Solvers/sdeint.py), and its it/s is the number the user wants.
It cannot be a bar in here: a posterior build constructs 10k-30k of those bars, one per time segment.

    ⚠ THE RATE IS NO LONGER SCRAPED FROM THE BAR'S RENDERED TEXT, AND MUST NOT GO BACK TO BEING.
    It was, and that coupling broke the moment the solver got fast: tqdm paints a frame carrying a
    rate only once `mininterval` has elapsed, so a solver call SHORTER than its own bar's mininterval
    renders "?it/s" and nothing else. When CUDA graphs took a 100k-step call from ~10 s to ~0.7 s the
    meter read "-- (idle)" for entire runs -- a 10x speedup made a progress bar too fast to render.
    The rate now comes from `core.progress.SOLVER`, a monotonic step count the solver publishes,
    differenced here on the 100 ms tick. That is correct at any speed, including the next speedup.

    The solver bar still exists and is still found by `config.SOLVER_BAR_DESC` -- but only to EXCLUDE
    it from the election that drives the overall bar (its total is in the tens of thousands,
    so it would win every time and sweep the bar 0->100% every second).
"""
import math
import time
from collections import deque

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QPainter, QPalette, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget)

from core import progress

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPIN_MS = 100

# How long to keep showing the last rate after the step counter stops advancing. This bridges the
# NON-solver work between solver calls (force construction, gen_stats) so the meter does not flicker
# between segments; it is not there to bridge a bar's mininterval any more.
#
# It was 45.0, sized for ~10 s solver calls whose rate frames arrived about a second apart. Samples
# now arrive every 100 ms, so 45 s only meant displaying a stale number long after the solver stopped
# -- through the whole start of neural-network training, where the solver genuinely is not running.
SOLVER_IDLE_S = 8.0

# EMA weight for the sampled rate. 0.3 is tqdm's own `smoothing` default, so the number reads the way
# the CLI's bar does. Raw 100 ms deltas are jumpy: the graphed path submits work to the GPU faster
# than the GPU completes it, so the counter advances in bursts.
_RATE_SMOOTH = 0.3

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

    Fed one sample per 100 ms tick with the current solver rate; ``None`` marks a gap (the solver is
    idle or the run has stalled), which breaks the line so a pause reads as a pause rather than a flat
    crawl. Drawn on a log10 scale, so a 10 it/s -> 10k it/s ramp is a gentle slope, not a cliff. Colours
    come from the palette (Highlight for the trace, Mid for the baseline), so it reads correctly in
    light and dark.
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


class ProgressPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: tuple = ()          # the last snapshot, solver bar excluded
        self._rate: float | None = None
        self._rate_at = 0.0
        self._steps = 0                 # last sampled core.progress.SOLVER.steps
        self._steps_at = 0.0
        self._beat_at = 0.0
        self._frame = 0

        # -- solver line: the rate meter + the live trend ---------------------
        self.solver_label = QLabel()
        self.solver_label.setTextFormat(Qt.PlainText)
        self.solver_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # The sparkline is the primary at-a-glance readout; the '+'-meter in solver_label is kept as a
        # compact secondary. Must be constructed before _reset_solver() below (it clears the sparkline).
        self.sparkline = _Sparkline()

        solver_line = QWidget()
        solver_layout = QHBoxLayout(solver_line)
        solver_layout.setContentsMargins(0, 0, 0, 0)
        solver_layout.setSpacing(8)
        solver_layout.addWidget(self.solver_label)
        solver_layout.addWidget(self.sparkline)
        solver_layout.addStretch(1)      # took over from the deleted per-segment strip

        # -- the one overall bar + spinner ------------------------------------
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

        # -- the caption: which stage the bar is measuring ---------------------
        self.caption = QLabel()
        self.caption.setTextFormat(Qt.PlainText)
        # Ignored horizontally, exactly as the deleted _BarRow's label was: QLabel.setText() calls
        # updateGeometry() unconditionally, so an unpinned label re-lays-out the whole right-hand pane
        # on every pump tick, visibly jittering the figure/log splitter. It also lets the prior sweeps'
        # very long mutating desc CLIP instead of widening the pane. Left in place (never hidden) when
        # empty, so the pane's height does not change as stages come and go.
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.setSpacing(2)
        self._layout.addWidget(solver_line)
        self._layout.addWidget(overall_line)
        self._layout.addWidget(self.caption)

        # Drives the spinner, the solver rate AND the two timeouts -- staleness has to be noticed
        # precisely when no events are arriving, so it cannot be evaluated from set_rows() alone.
        self._timer = QTimer(self)
        self._timer.setInterval(_SPIN_MS)
        self._timer.timeout.connect(self._tick)

        self._reset_solver()
        self.setVisible(False)

    # -- lifecycle -------------------------------------------------------------
    def begin(self) -> None:
        self.end()
        self.heartbeat()
        self.setVisible(True)
        self._timer.start()

    def end(self) -> None:
        """Authoritative teardown. Clears the whole surface regardless of whether any bar reported a
        close, so nothing leaked by a crashed worker can survive into the next dispatch."""
        self._timer.stop()
        self._rows = ()
        self.overall.setRange(0, 0)
        self.overall.setToolTip("")
        self.caption.setText("")
        self.spinner.setText("")
        self.spinner.setToolTip("")
        self._reset_solver()
        self.setVisible(False)

    def heartbeat(self) -> None:
        """Mark 'the run produced output just now'. Called on every rows snapshot, on every batch of
        log lines, and on any tick where the solver's step counter advanced -- a run that is computing
        but not printing is still alive."""
        self._beat_at = time.monotonic()

    # -- the worker's rows signal lands here -----------------------------------
    def set_rows(self, snapshot) -> None:
        """`snapshot` is the pump's full set of live rows (a tuple[RowState]), already sorted.

        The solver bar is dropped here and nowhere else: it is the one bar whose total (tens of
        thousands) and its largest-in-the-nest total would otherwise decide the overall bar. Its rate does not come
        from this snapshot -- see the module docstring.
        """
        self.heartbeat()
        self._rows = tuple(s for s in snapshot if not s.is_solver)
        self._retarget(self._rows)

    # -- the solver line -------------------------------------------------------
    def _reset_solver(self) -> None:
        self._rate = None
        self._rate_at = 0.0
        # Re-baseline the counter, so steps burned by a PREVIOUS run are never attributed to this one.
        # The counter itself is monotonic and process-wide and must not be reset (core/progress.py).
        self._steps = progress.SOLVER.steps
        self._steps_at = time.monotonic()
        self.sparkline.clear()
        self._paint_solver()

    def _sample_solver(self) -> None:
        """Difference the solver's step counter over this 100 ms tick and smooth it into a rate.

        Runs on the GUI thread; the counter is written on the worker thread. One writer, so no update
        can be lost, and a stale read just moves those steps into the next sample (core/progress.py).
        """
        now = time.monotonic()
        steps = progress.SOLVER.steps
        dn, dt = steps - self._steps, now - self._steps_at
        self._steps, self._steps_at = steps, now

        if dn > 0 and dt > 0:
            inst = dn / dt
            self._rate = inst if self._rate is None else \
                (1.0 - _RATE_SMOOTH) * self._rate + _RATE_SMOOTH * inst
            self._rate_at = now
            self.heartbeat()               # a moving counter is evidence of life, printing or not
        elif self._rate is not None and now - self._rate_at > SOLVER_IDLE_S:
            self._rate = None              # the solver stopped: holding the last number would lie

        self.sparkline.push(self._rate)
        self._paint_solver()

    def _paint_solver(self) -> None:
        rate = self._rate
        self.solver_label.setText(f"Solver Performance: {plus_meter(rate)}  ({format_rate(rate)})")
        if rate is None:
            self.solver_label.setToolTip("The SDE solver is not running right now.")
        else:
            self.solver_label.setToolTip(
                f"SDE solver: {rate:,.0f} integration steps/sec.\nOne '+' per order of magnitude.")

    # -- the spinner + stall detection -----------------------------------------
    def _tick(self) -> None:
        # BEFORE the stall check, and that order matters: a moving step counter heartbeats, so a run
        # that is integrating hard while printing nothing cannot be declared wedged.
        self._sample_solver()
        idle = time.monotonic() - self._beat_at
        if idle > STALL_S:
            # Freeze the spinner rather than animate it. A spinner that keeps twirling on a wedged run
            # is worse than none: it actively asserts progress that is not happening.
            stall = f"⏳ no output for {int(idle)}s"
            self.spinner.setText(stall)
            self.spinner.setToolTip(stall)      # the width is pinned now, so long text can clip
            return
        self._frame = (self._frame + 1) % len(_SPINNER)
        self.spinner.setText(_SPINNER[self._frame])

    # -- the one bar, and the caption under it ---------------------------------
    def _retarget(self, rows) -> None:
        """Drive the overall bar from the non-degenerate bar with the LARGEST total.

        `rows` EXCLUDES the solver bar, and must: its tens-of-thousands total would win the
        largest-total election and sweep the overall bar 0->100% every second.

        Largest-total, not deepest and not outermost:
          * DEEPEST is what this used to do, and it is one config flip away from being wrong -- the
            per-time-segment bar (core/Simulator/simulator.py) wraps segs in {1,2,3}, so with
            config.QUIET_SEGMENT_BAR False it is both deeper and far coarser than "Generating
            training data", and would sweep the overall bar 0->100% every couple of seconds. Largest
            total is immune to that whole class. In every nest actually observed the two agree.
          * OUTERMOST pegs the bar: the pos-0 bar ("Training neural posterior", pipeline.train_nn)
            wraps range(TRAINING_NUM_ROUNDS) and that is 1 (config.TRAINING_NUM_ROUNDS), so it reads
            0% for the entire multi-hour build -- hence RowState.informative excludes total<=1.
          * A STICKY first-seen driver also pegs it: sbi's neural-network training emits no tqdm bar
            at all (only a printed epoch counter), so a driver latched onto "Generating training
            data" would sit at 100% through the longest phase, reading as finished or hung.

        Ties on total break on depth, so the reading is at least stable rather than arbitrary.
        """
        driver = max((s for s in rows if s.informative),
                     key=lambda s: (s.total, s.row), default=None)
        if driver is None:
            self.overall.setRange(0, 0)
        else:
            self.overall.setRange(0, 100)
            self.overall.setValue(driver.pct)
        self._paint_caption(driver, rows)
        self._paint_detail(rows)

    def _paint_caption(self, driver, rows) -> None:
        """Name the stage the bar is measuring -- or, when nothing reports a percentage, the deepest
        thing that is still saying something.

        ⚠ The fallback is load-bearing, not a nicety: see the caption warning in the module docstring.
        `total != 1` drops the degenerate `Training neural posterior -- 0/1` bar, which leaves sbi's
        printed epoch counter (an overwrite-mode row, pct and total both None) as the only candidate
        during neural-network training -- which is precisely what should be on screen for those hours.
        """
        if driver is not None:
            self.caption.setText(f"{driver.desc}  —  {driver.stats}" if driver.stats else driver.desc)
            return
        speaking = [s for s in rows if (s.total or 0) != 1 and s.desc]
        self.caption.setText(max(speaking, key=lambda s: s.row).desc if speaking else "")

    def _paint_detail(self, rows) -> None:
        """The full live nest, on the overall bar's tooltip -- the detail this pane stopped rendering
        as rows, kept one hover away rather than deleted. setToolTip stores a string and triggers no
        relayout, so it is safe to refresh on every snapshot."""
        detail = "\n".join(
            f"{'    ' * s.row}{s.desc}{('  —  ' + s.stats) if s.stats else ''}" for s in rows)
        self.overall.setToolTip(detail or "Nothing is reporting progress right now.")
