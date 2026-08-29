"""The tqdm/VT100 router, the Qt progress stack, the solver meter and the overall-bar election. Split from test_gui_progress.py; run directly: python tests/test_vt_progress.py"""
"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  python -m pytest tests/test_gui_progress.py -v
      (or just: python tests/test_gui_progress.py)
"""
import ast
import inspect
import textwrap
import os
import tempfile
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

import torch                                                      # noqa: E402
from PySide6.QtWidgets import QApplication                        # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from core.gui.panels.base_panel import BasePanel                  # noqa: E402
from core.gui.streams import redirect_streams                     # noqa: E402
from core.gui.vt import StreamRouter, parse_bar                   # noqa: E402
from core.gui.widgets.log_pane import LogPane                     # noqa: E402
from core.gui.widgets.progress_pane import ProgressPane           # noqa: E402
from core.gui.worker import WorkerSignals                         # noqa: E402
import contextlib                                                  # noqa: E402

def _app():
    return QApplication.instance() or QApplication([])
def _pump(app, seconds=0.5):
    """Drive the event loop without app.exec(), so the pump's queued signals get delivered."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
# ── the router, driven by REAL tqdm (no Qt) ──────────────────────────────────────────────────────
def _drive(fn, level="warning"):
    """Run `fn` with sys.stderr routed through a StreamRouter; return its event list."""
    events = []

    class Stream:
        def write(self, s):
            if s:
                router.feed(s)
            return len(s)

        def flush(self):
            pass

        def isatty(self):
            return False

    router = StreamRouter("err", lambda k, p: events.append((k, p)), level=level)
    real, sys.stderr = sys.stderr, Stream()
    try:
        fn()
    finally:
        router.close()
        sys.stderr = real
    return events
def _live(events):
    rows = {}
    peak = 0
    for kind, payload in events:
        if kind == "row":
            rows[payload.key] = payload
        elif kind == "retire":
            rows.pop(payload, None)
        peak = max(peak, len(rows))
    return rows, peak


def test_four_deep_nest_emits_no_log_lines():
    """The regression: a 4-deep nest must produce ZERO log lines and exactly 4 concurrent rows."""
    def nest():
        for _ in tqdm(range(2), desc="Training neural posterior", leave=False):
            for _ in tqdm(range(3), desc="Generating training data", leave=False):
                for _ in tqdm(range(2), desc="Running time segments", leave=False):
                    for _ in tqdm(range(2), desc="step (batch=64)", leave=False):
                        pass

    events = _drive(nest)
    logs = [p for k, p in events if k == "log"]
    rows, peak = _live(events)

    assert logs == [], f"nested bars leaked {len(logs)} log line(s): {logs[:3]}"
    assert peak == 4, f"expected 4 concurrent rows, got {peak}"
    assert rows == {}, f"{len(rows)} row(s) survived close()"
    assert {p.row for k, p in events if k == "row"} == {0, 1, 2, 3}

def test_no_ansi_reaches_the_log():
    """colorama is installed, so tqdm's moveto(-n) emits real '\\x1b[A'. It is motion, not text."""
    def nest():
        for _ in tqdm(range(2), desc="outer", leave=False):
            for _ in tqdm(range(2), desc="inner", leave=False):
                pass

    for kind, payload in _drive(nest):
        text = payload[0] if kind == "log" else getattr(payload, "raw", "")
        assert "\x1b" in text is False or "\x1b" not in text, f"ANSI leaked: {text!r}"

def test_leave_true_bar_persists_to_log_and_leaves_no_ghost_row():
    """close(leave=True)'s trailing '\\n' (tqdm/std.py:1303) is byte-identical to a moveto(+1), so a
    new pos-0 bar opening right after one is initially GUESSED onto row 1. The absence of a following
    up-move then proves it belongs on row 0 and the router rekeys it.

    What must hold is the settled state: the sequence ends with exactly one row, at index 0, and the
    closed bar's final frame graduates into the log (as it does on a real terminal). The transient
    row-1 guess is invisible because the pump coalesces -- see the peak-row assertion below, which is
    what a widget actually sees."""
    def bars():
        for _ in tqdm(range(2), desc="Constructing latent prior..."):
            pass
        for _ in tqdm(range(2), desc="Campaign 2 (chi sweep)"):
            pass

    events = _drive(bars)
    rows, peak = _live(events)
    logs = [t for k, (t, _lvl) in ((k, p) for k, p in events if k == "log")]

    assert rows == {}, "a leave=True bar left a ghost row"
    assert peak == 1, f"the two sequential pos-0 bars were live at once ({peak} rows) -- the " \
                      f"leave=True finalizer newline was mistaken for a moveto"
    assert sum("100%" in t for t in logs) == 2, f"final frames not persisted to the log: {logs}"

def test_pump_never_exposes_the_transient_row_guess():
    """The user-visible guarantee: through the pump, two sequential pos-0 bars never render as two
    rows. The row-1 guess and its correction land inside one 15 Hz tick and are coalesced away."""
    app = _app()
    signals = WorkerSignals()
    snapshots = []
    signals.rows.connect(snapshots.append)
    signals.log_batch.connect(lambda _b: None)

    with redirect_streams(signals):
        for _ in tqdm(range(2), desc="Constructing latent prior..."):
            time.sleep(0.05)
        for _ in tqdm(range(2), desc="Campaign 2 (chi sweep)"):
            time.sleep(0.05)
    _pump(app)

    worst = max((len(s) for s in snapshots), default=0)
    assert worst <= 1, f"the pump exposed {worst} concurrent rows for two sequential pos-0 bars"
    assert all(r.row == 0 for s in snapshots for r in s), \
        "a pos-0 bar was painted on row 1"

def test_mutating_description_does_not_mint_a_row_per_iteration():
    """core/SBI/Priors/{bp,hopf,nadrowski}_prior.py call set_description() with a live counter on
    EVERY iteration. Rows are keyed by tqdm `pos`, and the digit-normalised ident keeps the counter
    from reading as 'a different bar took this slot'."""
    def sweep():
        bar = tqdm(total=20, desc="Added 0 sets to accepted parameters", leave=False)
        for i in range(20):
            bar.set_description(f"Added {i} sets to accepted parameters")
            bar.update(1)
        bar.close()

    events = _drive(sweep)
    rows, peak = _live(events)
    logs = [p for k, p in events if k == "log"]

    assert peak == 1, f"a mutating desc minted {peak} concurrent rows"
    assert rows == {}
    assert logs == [], f"a mutating desc leaked log lines: {logs[:3]}"

def test_total_one_bar_is_not_informative():
    """core/SBI/pipeline.py:517 wraps range(TRAINING_NUM_ROUNDS) and that is 1, so it reads 0% for
    the whole build. It must not be allowed to drive the overall bar."""
    degenerate = parse_bar(("err", 0), 0, "Training neural posterior:   0%|  | 0/1 [00:00<?, ?it/s]")
    real = parse_bar(("err", 1), 1, "Generating training data:  42%|## | 2100/5000 [00:12<00:16]")
    assert degenerate.total == 1 and not degenerate.informative
    assert real.total == 5000 and real.informative and real.pct == 42

# ── the full Qt stack ────────────────────────────────────────────────────────────────────────────
def test_end_to_end_log_pane_gains_zero_blocks():
    """The whole stack: real nested tqdm -> redirect_streams -> pump -> ProgressPane / LogPane."""
    app = _app()
    log, prog = LogPane(), ProgressPane()
    signals = WorkerSignals()
    signals.log.connect(log.append_line)
    signals.log_batch.connect(log.append_lines)
    signals.rows.connect(prog.set_rows)

    peak_rows = 0

    def watch(snapshot):
        nonlocal peak_rows
        peak_rows = max(peak_rows, len(snapshot))

    signals.rows.connect(watch)

    prog.begin()
    with redirect_streams(signals):
        for _ in tqdm(range(2), desc="Training neural posterior", leave=False):
            for _ in tqdm(range(4), desc="Generating training data", leave=False):
                for _ in tqdm(range(3), desc="Running time segments", leave=False):
                    time.sleep(0.02)          # let the 15 Hz pump actually tick
    _pump(app)

    assert log.blockCount() == 1 and not log.toPlainText().strip(), \
        f"log pane gained {log.blockCount()} block(s):\n{log.toPlainText()[:500]}"
    assert peak_rows == 3, f"expected 3 concurrent progress rows, got {peak_rows}"

    prog.end()
    assert not prog._rows, "ProgressPane.end() left rows behind"
    assert not prog.caption.text(), "ProgressPane.end() left the caption behind"
    assert prog.overall.maximum() == 0, "ProgressPane.end() left the overall bar determinate"

def test_print_output_still_reaches_the_log():
    """The bars must not swallow ordinary pipeline output."""
    app = _app()
    log = LogPane()
    signals = WorkerSignals()
    signals.log.connect(log.append_line)
    signals.log_batch.connect(log.append_lines)
    signals.rows.connect(lambda _s: None)

    with redirect_streams(signals):
        print("Config built: NADROWSKI")
        for _ in tqdm(range(3), desc="Generating training data", leave=False):
            time.sleep(0.02)
        print("Prior ready.")
    _pump(app)

    text = log.toPlainText()
    assert "Config built: NADROWSKI" in text
    assert "Prior ready." in text
    assert "\x1b" not in text
    assert "it/s]" not in text, f"a bar frame leaked into the log:\n{text}"

def test_sbi_epoch_counter_becomes_a_progress_row_not_log_spam():
    """sbi's training loop has NO tqdm bar. It prints, on STDOUT:
           print("\\r", f"Training neural network. Epochs trained: {epoch}", end="")
       (sbi/inference/trainers/base.py:1024) -- a LEADING '\\r' and no terminator, so it is an
       overwrite-mode status line. It must render as one updating row, not one log line per epoch,
       and its final value must not be stranded (the old reader dropped it with its buffer)."""
    app = _app()
    log = LogPane()
    signals = WorkerSignals()
    snapshots = []
    signals.log.connect(log.append_line)
    signals.log_batch.connect(log.append_lines)
    signals.rows.connect(snapshots.append)

    with redirect_streams(signals):
        for epoch in range(1, 6):
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")
            time.sleep(0.03)
        print("\nNeural network successfully converged after 5 epochs.")
    _pump(app)

    text = log.toPlainText()
    assert text.count("Epochs trained") == 1, \
        f"the epoch counter was appended per-epoch instead of overwriting:\n{text}"
    assert "Epochs trained: 5" in text, "the FINAL epoch was stranded and never shown"
    assert "successfully converged" in text
    assert any(r.desc.startswith("Training neural network") for s in snapshots for r in s), \
        "the epoch counter never rendered as a progress row"
    assert all(r.pct is None for s in snapshots for r in s), "a status line faked a percentage"

def test_plain_prints_are_not_eaten_by_the_cursor_logic():
    """A print() writes its text and its '\\n' as TWO chunks, so the '\\n' arrives alone -- byte-identical
    to a tqdm moveto(+1). Treating it as cursor motion strands the line forever and shifts the next
    bar down a phantom row. Ordering and line boundaries must survive a print/bar/print interleave."""
    app = _app()
    log = LogPane()
    signals = WorkerSignals()
    signals.log.connect(log.append_line)
    signals.log_batch.connect(log.append_lines)
    signals.rows.connect(lambda _s: None)

    with redirect_streams(signals):
        print("Starting fake stage")
        print()                                   # a bare '\n' chunk, with nothing pending
        for _ in tqdm(range(3), desc="Generating training data", leave=False):
            time.sleep(0.02)
        for epoch in range(1, 4):
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")
            time.sleep(0.02)
        print("\nNeural network successfully converged.")
        print("Prior ready.")
    _pump(app)

    lines = [ln for ln in log.toPlainText().splitlines() if ln.strip()]
    assert lines == [
        "Starting fake stage",
        "Training neural network. Epochs trained: 3",
        "Neural network successfully converged.",
        "Prior ready.",
    ], f"log lines mangled:\n{lines}"

def test_quiet_segment_bar_collapses_the_nest():
    """config.QUIET_SEGMENT_BAR (set by core.gui.app.build_app) drops the per-time-segment bar, taking
    the nest from 4 deep to 3 -- and a disabled bar must SURRENDER its slot (tqdm/std.py:985-992 removes
    it from _instances), not merely hide, or the solver would still sit at pos 3.

    The solver bar itself stays ON: its it/s is the Solver Performance meter. It is hidden at the widget
    layer, not the tqdm layer -- see test_solver_bar_is_not_rendered_as_a_row."""
    from core import config

    def nest():
        for _ in tqdm(range(2), desc="Training neural posterior", leave=False):
            for _ in tqdm(range(2), desc="Generating training data", leave=False):
                for _ in tqdm(range(2), desc="Running time segments", leave=False,
                              **({"disable": True} if config.QUIET_SEGMENT_BAR else {})):
                    for _ in tqdm(range(2), desc="step (batch=64)", leave=False):
                        pass

    assert config.QUIET_SEGMENT_BAR is False, "the CLI default must be False"
    _, loud_peak = _live(_drive(nest))

    config.QUIET_SEGMENT_BAR = True
    try:
        rows, quiet_peak = _live(_drive(nest))
    finally:
        config.QUIET_SEGMENT_BAR = False

    assert loud_peak == 4, f"expected a 4-deep nest when loud, got {loud_peak}"
    assert quiet_peak == 3, f"QUIET_SEGMENT_BAR should leave 3 bars, got {quiet_peak}"
    assert rows == {}

# ── the solver-performance meter ─────────────────────────────────────────────────────────────────
def test_solver_rate_is_parsed_from_all_three_tqdm_renderings():
    """tqdm renders its rate three ways (std.py:550-559). The s/it form is the trap: it is SECONDS PER
    ITERATION, so " 2.50s/it" is 0.4 it/s, not 2.5 -- read naively, a crawling solver reads as a fast
    one and the meter would show MORE plus signs the slower it got."""
    fast = parse_bar(("err", 2), 2, "step (batch=32):  88%|## | 13269/14999 [00:01<00:00, 13267.85it/s]")
    slow = parse_bar(("err", 2), 2, "step (batch=32):  10%|#  | 3/30 [00:07<01:07,  2.50s/it]")
    fresh = parse_bar(("err", 2), 2, "step (batch=32):   0%|   | 0/83190 [00:00<?, ?it/s]")

    assert fast.rate == 13267.85 and fast.is_solver
    assert slow.rate == 0.4, f"s/it must be inverted, got {slow.rate}"
    assert fresh.rate is None, "'?it/s' means no measurement yet -- it must not read as 0"

def test_only_the_solver_bar_is_identified_as_the_solver():
    """The meter keys on the desc prefix, never the row: the solver's tqdm `pos` is 0, 1 or 2 depending
    on the phase and the panel."""
    for row in (0, 1, 2):
        assert parse_bar(("err", row), row, f"step (batch=2048):  50%|# | 1/2 [00:00<00:00, 9.0it/s]").is_solver
    for desc in ("Generating training data", "Training neural posterior", "PPC simulations",
                 "Campaign 2 (chi sweep, fpb<=64)", "Constructing latent prior..."):
        state = parse_bar(("err", 1), 1, f"{desc}:  50%|# | 1/2 [00:00<00:00, 9.0it/s]")
        assert not state.is_solver, f"{desc!r} must not be mistaken for the solver bar"

def test_plus_meter_is_one_sign_per_order_of_magnitude():
    from core.gui.widgets.progress_pane import plus_meter

    assert plus_meter(10_000) == "++++"       # the user's worked example
    assert plus_meter(13_267.85) == "++++"
    assert plus_meter(1_000) == "+++"
    assert plus_meter(999) == "++"
    assert plus_meter(10) == "+"
    assert plus_meter(5) == "·"               # under one order of magnitude: not an empty string
    assert plus_meter(0.4) == "·"
    assert plus_meter(None) == "—"

def test_the_solver_bar_never_drives_the_overall_bar_and_is_not_the_caption():
    """The solver bar must not decide the overall bar: its total is in the tens of thousands and it is
    the deepest bar there is, so it would win the election every time and drag the bar through a full
    0->100% sweep every second instead of showing the top-level count. It is also not a row
    any more -- nothing is -- so it must not surface as the caption either."""
    _app()
    prog = ProgressPane()
    prog.begin()

    top = parse_bar(("err", 1), 1, "Generating training data:  38%|### | 1902/5000 [05:12<13:41,  6.1it/s]")
    solver = parse_bar(("err", 2), 2, "step (batch=32):  88%|####| 13269/14999 [00:01<00:00, 13267.85it/s]")
    prog.set_rows((top, solver))

    assert prog._rows == (top,), f"the solver bar survived into the pane's row set: {prog._rows}"
    assert prog.overall.maximum() == 100 and prog.overall.value() == 38, \
        "the overall bar must track the top-level count, not the solver"
    assert prog.caption.text().startswith("Generating training data"), prog.caption.text()
    assert "step (batch=" not in prog.caption.text(), prog.caption.text()
    assert "step (batch=" not in prog.overall.toolTip(), prog.overall.toolTip()
    assert "1902/5000" in prog.overall.toolTip(), "the detail tooltip lost the live nest"
    prog.end()

def test_the_solver_meter_reads_the_step_counter_not_a_rendered_bar():
    """THE REGRESSION THIS DESIGN EXISTS TO KILL. The meter used to be scraped from the rendered text
    of the solver's tqdm bar, so a solver call SHORTER than that bar's own `mininterval` produced no
    rate at all -- and CUDA graphs made every call shorter than it, so the meter read "-- (idle)" for
    whole multi-day runs.

    The property asserted here is the one the scrape could never have: a correct rate with ZERO tqdm
    frames painted. Not one bar is created below.
    """
    from core import progress
    from core.gui.widgets import progress_pane as pp

    _app()
    prog = ProgressPane()
    prog.begin()
    prog.set_rows(())                       # no bars at all, anywhere

    assert "idle" in prog.solver_label.text(), "a fresh pane should not claim a rate"

    # One second of solver, 120k steps. The clock is nudged rather than slept on: time.monotonic()
    # has ~15ms granularity on Windows, so back-to-back ticks can measure dt == 0.
    prog._steps_at = time.monotonic() - 1.0
    progress.SOLVER.add(120_000)
    prog._tick()
    assert "120.0k it/s" in prog.solver_label.text(), prog.solver_label.text()
    assert "+++++" in prog.solver_label.text(), prog.solver_label.text()

    # A tick with no new steps must NOT wipe the rate -- the gaps between solver calls are real work.
    prog._tick()
    assert "120.0k it/s" in prog.solver_label.text(), \
        f"one quiet tick clobbered the held rate: {prog.solver_label.text()}"

    # ...but once the counter has been still for SOLVER_IDLE_S, the meter must admit it is idle.
    prog._rate_at -= pp.SOLVER_IDLE_S + 1
    prog._tick()
    assert "idle" in prog.solver_label.text(), prog.solver_label.text()
    prog.end()

def test_the_solver_bar_paints_a_rate_even_when_the_call_is_under_a_second():
    """The CLI half of the same regression, pinned at the tqdm layer.

    The GUI no longer reads this bar, but `python -m core` has nothing else: with mininterval=1.0 a
    graphed 100k-step call (~0.7s) rendered its opening "?it/s" frame and never a rate -- measured 123
    chars of stderr containing none. Drives the REAL settings from sdeint._bar_kwargs, and checks the
    counterfactual so the test cannot go vacuous if someone puts mininterval back.
    """
    import re
    from io import StringIO

    from core.Solvers import sdeint

    n, chunk, seconds = 100_000, 50, 0.4

    def render(**override):
        buf = StringIO()

        class Sink:
            def write(self, text):
                buf.write(text)
                return len(text)

            def flush(self):
                pass

            def isatty(self):
                return False

        kw = sdeint._bar_kwargs(n, 2048)
        kw.update(override)
        bar = tqdm(total=n - 1, file=Sink(), **kw)
        steps = (n - 1) // chunk
        t0 = time.perf_counter()
        for k in range(steps):
            while time.perf_counter() - t0 < (k + 1) * seconds / steps:
                pass
            bar.update(chunk)
        bar.close()
        return re.findall(r"[0-9.]+(?:it/s|s/it)", buf.getvalue())

    assert sdeint._bar_kwargs(n, 2048)["mininterval"] <= 0.2, \
        "the solver bar's mininterval is back above a short call's duration"
    assert render(), "the solver bar rendered NO rate over a 0.4s call -- the CLI meter is blind again"
    assert not render(mininterval=1.0), \
        "the counterfactual rendered a rate, so this test no longer proves anything"

def test_spinner_animates_and_then_reports_a_stall():
    """A spinner that keeps twirling on a wedged run asserts progress that is not happening."""
    from core.gui.widgets import progress_pane as pp

    _app()
    prog = ProgressPane()
    prog.begin()

    frames = set()
    for _ in range(4):
        prog._tick()
        frames.add(prog.spinner.text())
    assert len(frames) == 4, f"the spinner did not advance: {frames}"

    prog._beat_at -= pp.STALL_S + 62
    prog._tick()
    assert "no output for" in prog.spinner.text(), prog.spinner.text()

    prog.heartbeat()                      # output resumes -> back to spinning
    prog._tick()
    assert "no output for" not in prog.spinner.text()
    prog.end()

def test_the_solver_step_iterator_closes_its_bar_when_the_consumer_raises():
    """`sdeint._step_iter` is a GENERATOR wrapping the tqdm bar, so it adds a frame to the cancel's tqdm-lock unwind
    path -- a cancel raises from inside a bar redraw, and tqdm's own `finally: self.close()` has to run
    anyway or its global write lock leaks and the NEXT `tqdm.__new__` DEADLOCKS. (C1's own test hangs
    rather than failing, which is why this cheap structural guard is worth having in front of it.)

    Not a proof that the generator changed anything -- it is a guard on the property the generator put
    at risk.
    """
    import gc

    from core.Solvers import sdeint

    class Quiet:
        def write(self, text):
            return len(text)

        def flush(self):
            pass

        def isatty(self):
            return False

    real, sys.stderr = sys.stderr, Quiet()
    try:
        before = len(tqdm._instances)
        try:
            for i in sdeint._step_iter(5000, 32):
                if i == 10:
                    raise RuntimeError("the consumer blew up mid-integration")
        except RuntimeError:
            pass
        gc.collect()
        after = len(tqdm._instances)
    finally:
        sys.stderr = real

    assert after == before, \
        f"the solver bar was stranded by the generator wrapper ({before} -> {after} live tqdm instances)"

def test_the_caption_falls_back_to_the_sbi_epoch_counter():
    """THE REASON THE CAPTION EXISTS. sbi's neural-network training -- hours of a multi-day build --
    emits no tqdm bar at all, only a printed epoch counter (an overwrite-mode row, pct and total both
    None). The only other live row is `Training neural posterior -- 0/1`, which is degenerate. Render
    "no rows, just the bar" literally and that whole phase is a blank indeterminate bar."""
    _app()
    prog = ProgressPane()
    prog.begin()

    degenerate = parse_bar(("err", 0), 0, "Training neural posterior:   0%|  | 0/1 [00:00<?, ?it/s]")
    epochs = parse_bar(("out", 0), 0, " Training neural network. Epochs trained: 812")
    prog.set_rows((degenerate, epochs))

    assert prog.overall.maximum() == 0, "a total=1 bar was allowed to drive the overall bar"
    assert "Epochs trained: 812" in prog.caption.text(), \
        f"the longest phase of the run has no caption: {prog.caption.text()!r}"
    prog.end()

def test_the_overall_bar_follows_the_largest_non_degenerate_total():
    """Largest total, not deepest. The per-time-segment bar wraps segs in {1,2,3}, so it is BOTH
    deeper and far coarser than "Generating training data" -- picking the deepest row would sweep the
    overall bar 0->100% every couple of seconds. (config.QUIET_SEGMENT_BAR hides that bar under the
    GUI today; the election should not depend on it still being set.)"""
    _app()
    prog = ProgressPane()
    prog.begin()

    rounds = parse_bar(("err", 0), 0, "Training neural posterior:   0%|  | 0/1 [00:00<?, ?it/s]")
    data = parse_bar(("err", 1), 1, "Generating training data:  38%|### | 1902/5000 [05:12<13:41]")
    segs = parse_bar(("err", 2), 2, "Running time segments:  66%|##  | 2/3 [00:01<00:00]")
    solver = parse_bar(("err", 3), 3, "step (batch=32):  88%|####| 13269/14999 [00:01<00:00, 13267.85it/s]")
    prog.set_rows((rounds, data, segs, solver))

    assert prog.overall.value() == 38, \
        f"the overall bar followed the deepest bar ({prog.overall.value()}%), not the largest total"
    assert prog.caption.text().startswith("Generating training data"), prog.caption.text()
    prog.end()

def test_leave_true_pos0_bar_is_retired_not_left_pegged_at_100():
    """close(leave=True) at pos 0 paints the final frame and then writes a bare '\\n' -- which is
    byte-identical to a moveto(+1). Assuming "moveto" leaves the finished bar sitting in the pane at
    100% AND, because it is `informative`, pegging the overall bar at 100% while the pipeline is still
    working. Both real leave=True bars (core/SBI/Priors/prior.py:88 "Constructing latent prior...",
    core/FDT/campaigns.py:214 "Campaign 2") are pos 0, so this is the common case, not a corner."""
    def bar_then_work():
        for _ in tqdm(range(5), desc="Constructing latent prior..."):
            pass
        print("Prior constructed.")          # the pipeline carries on after the bar closes

    events = _drive(bar_then_work)
    rows, _ = _live(events)
    logs = [t for _k, (t, _lvl) in ((k, p) for k, p in events if k == "log")]

    assert rows == {}, f"the finished leave=True bar was never retired: {rows}"
    assert any("100%" in t for t in logs), "its final frame should graduate into the log"

    # and the overall bar must not still be reading 100%
    app = _app()
    prog = ProgressPane()
    live = {}
    for kind, payload in events:
        if kind == "row":
            live[payload.key] = payload
        elif kind == "retire":
            live.pop(payload, None)
        prog.set_rows(tuple(live.values()))
    assert prog.overall.maximum() == 0, "the overall bar is still determinate after the bar finished"


if __name__ == "__main__":
    _app()
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            # Exception, NOT AssertionError: a crash is a failure of THAT test, not of the suite.
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}\n      {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
