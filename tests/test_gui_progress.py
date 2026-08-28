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
    0->100% sweep every second instead of showing the top-level count (trap S3). It is also not a row
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
    """`sdeint._step_iter` is a GENERATOR wrapping the tqdm bar, so it adds a frame to trap C1's unwind
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


# -- the training budget (Posterior tab) ----------------------------------------------------------
def _budget_cfg():
    """Stub SimConfig carrying exactly the fields the budget lines read.

    A stub, not a real build: the arithmetic under test is geometry -> elements -> GiB, and a real
    make_sim_config drags in bounds files and a 300k-point time grid for nothing.
    """
    from core import config

    class Cfg:
        hw = config.detect_device()
        t = type("T", (), {"shape": (250_000,)})()
        inits_dict = {"x": 0.0, "xa": 0.0, "f": 0.0}
        steady_idx = 500
        forcing_idx = {}
        model = "NADROWSKI"
        chi_mode = False
        chi_k_pad = 12
        observation_mode = "spontaneous"
    return Cfg()


def _budget_panel(cfg=None, prior=None):
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession()
    inf.session.cfg = cfg
    inf.session.inf_prior = prior
    inf.refresh_gates()
    return inf, inf.tabs.widget(2)          # Config Prior Posterior Validate Infer


def test_the_training_budget_shows_the_simulation_count_and_the_cap_trade():
    """The count was invisible: 5000 x 2048 = 10,240,000 simulations lived only in config.py. The line
    also has to say what the cap really trades, because batch width is NOT a speed knob -- the solver
    is kernel-launch-bound (measured 7.37 s at 2048 against 7.74 s at 1024, the SMALLER batch being
    slightly slower), so halving it halves the training rows for the same wall-clock.

    ⚠ ISOLATES THE SETTINGS FILE, and must. The panel seeds these fields from config.py and then
    RESTORES them from QSettings, so without isolation the assertions read whatever the developer
    last left in their real PRISM.ini. This test passed all morning and then failed the moment a run
    was configured with 10000 batches -- nothing to do with the code under test. It is the same
    restore-wins-over-config mechanism that cost a ~5-day run on 2026-08-19 (the retired chi band),
    so a test asserting the DEFAULT has to start from a file with no saved value.
    """
    from core import config
    import core.gui.settings as st

    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    os.unlink(path)                      # QSettings creates it; an empty file is not the same thing
    st.use_ini_file(path)
    try:
        cfg = _budget_cfg()
        _inf, panel = _budget_panel(cfg)
        width = cfg.hw.batch_size

        assert panel.num_runs.value() == config.TRAINING_NUM_RUNS, (
            f"the field must seed from config.py, not from a saved session: "
            f"{panel.num_runs.value()} != {config.TRAINING_NUM_RUNS}")
        assert panel.run_size_cap.value() == config.TRAINING_RUN_SIZE
        assert f"{config.TRAINING_NUM_RUNS * width:,} simulations" in panel.budget_total.text(), \
            panel.budget_total.text()
        assert "diversity" in panel.budget_total.text(), \
            "the line must say batch COUNT is the (t_scale, T) diversity, not just a budget"

        panel.run_size_cap.setText(str(width // 4))
        assert f"{config.TRAINING_NUM_RUNS * (width // 4):,} simulations" in panel.budget_total.text()
        assert "capped from" in panel.budget_total.text(), panel.budget_total.text()

        # ...and quadrupling the batch COUNT is what buys those rows back, at 4x the wall-clock.
        panel.num_runs.setText(str(config.TRAINING_NUM_RUNS * 4))
        assert f"{config.TRAINING_NUM_RUNS * width:,} simulations" in panel.budget_total.text()
    finally:
        st.use_ini_file(None)
        if os.path.exists(path):
            os.unlink(path)


def test_the_training_budget_reaches_build_posterior_as_arguments_not_via_config():
    """THE WIRING THAT MAKES THE FIELDS DO ANYTHING AT ALL.

    orchestrator binds TRAINING_NUM_RUNS / TRAINING_RUN_SIZE at import, so a panel that "applied" the
    user's numbers by assigning to core.config would be a silent no-op -- the run would simulate 5000
    x 2048 anyway and nothing would say otherwise. Both halves are asserted: the values arrive as
    call kwargs, AND the config constants are untouched.
    """
    from core import config

    before = (config.TRAINING_NUM_RUNS, config.TRAINING_RUN_SIZE)
    _inf, panel = _budget_panel(_budget_cfg(), prior=object())
    panel.num_runs.setText("777")
    panel.run_size_cap.setText("256")

    seen = {}
    panel.dispatch = lambda fn, *a, **kw: seen.update(fn=fn, args=a, kwargs=kw)
    panel._build_posterior()

    assert seen, "the Train button dispatched nothing"
    assert seen["kwargs"].get("num_runs") == 777, seen["kwargs"]
    assert seen["kwargs"].get("run_size_cap") == 256, seen["kwargs"]
    assert (config.TRAINING_NUM_RUNS, config.TRAINING_RUN_SIZE) == before, \
        "the panel mutated the config constants, which orchestrator has already snapshotted"


def test_the_budget_refuses_a_batch_count_below_one():
    """0 batches is a whole run that simulates nothing and then trains on an empty tensor."""
    _inf, panel = _budget_panel(_budget_cfg(), prior=object())
    panel.num_runs.setText("0")
    seen = []
    panel.dispatch = lambda fn, *a, **kw: seen.append(kw)
    panel._build_posterior()
    assert not seen, "a zero batch count was dispatched"


def test_the_budget_memory_line_reads_pipelines_own_cost_model():
    """The estimate must come from pipeline.peak_sim_elements, not a second copy of the formula, and
    it must be quoted at the WORST geometry the Sobol pre-filter admits -- n_fine swings from a median
    ~40k to a p99 ~283k, so a width that fits the median still OOMs on a few percent of batches, which
    is how two retrains actually died."""
    from core import config
    from core.SBI import pipeline

    cfg = _budget_cfg()
    _inf, panel = _budget_panel(cfg)
    width = cfg.hw.batch_size

    if cfg.hw.device.type != "cuda":
        assert "CUDA-only" in panel.budget_mem.text(), panel.budget_mem.text()
        return

    n_fine = min(config.N_ND_MAX, cfg.t.shape[0])
    need = pipeline.peak_sim_elements(width, n_fine, cfg.steady_idx, len(cfg.inits_dict), 1, 1)
    gib = need * cfg.hw.dtype.itemsize / float(1 << 30)
    assert f"{gib:.2f} GiB" in panel.budget_mem.text(), panel.budget_mem.text()
    assert f"{n_fine:,}" in panel.budget_mem.text(), "the line must name the geometry it assumed"
    assert "upper bound" in panel.budget_mem.text().lower(), \
        "free-VRAM readings overstate what is available; the line must not present one as fact"

    # Halving the width must halve the estimate -- the peak is linear in the batch.
    panel.run_size_cap.setText(str(width // 2))
    assert f"{gib / 2:.2f} GiB" in panel.budget_mem.text(), panel.budget_mem.text()


def test_the_budget_lines_never_raise_on_a_config_they_do_not_understand():
    """_sync_budget runs from refresh_gates(), so an exception in a STATUS LINE would take down the
    whole tab. The gate tests set session.cfg to a bare object(); so could any future stub."""
    _inf, panel = _budget_panel(cfg=object(), prior=object())
    panel._sync_budget()                                   # must not raise
    assert panel.budget_total.text(), "the total line went blank on an unknown config"
    assert "config" in panel.budget_ckpt.text().lower(), panel.budget_ckpt.text()


def test_the_training_budget_round_trips_through_settings():
    """"I have to retype it every launch" is the complaint L1 already answered for splitters."""
    from core.gui import settings as st

    path = _temp_settings()
    try:
        _inf, panel = _budget_panel(_budget_cfg())
        panel.num_runs.setText("1234")
        panel.run_size_cap.setText("512")
        panel.save_settings(st.settings())

        _inf2, fresh = _budget_panel(_budget_cfg())
        fresh.restore_settings(st.settings())
        assert fresh.num_runs.value() == 1234, fresh.num_runs.text()
        assert fresh.run_size_cap.value() == 512, fresh.run_size_cap.text()
        # And the derived line followed the restored values, not the defaults.
        assert "1,234 batches" in fresh.budget_total.text(), fresh.budget_total.text()
    finally:
        st.use_ini_file(None) if hasattr(st, "use_ini_file") else None
        os.unlink(path)


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


def test_worker_payload_is_released_after_the_run():
    """setAutoDelete(False) + the _finished closure keep the Worker shell alive forever. That is fine
    for the shell, but NOT for what it points at -- without an explicit release every dispatch pins its
    cfg / prior / posterior / CUDA tensors for the life of the process."""
    import gc
    import weakref

    app = _app()

    class Big:
        pass

    class P(BasePanel):
        pass

    panel = P()
    big = Big()
    ref = weakref.ref(big)

    panel.dispatch(lambda payload: None, big)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and panel._busy:
        app.processEvents()
        time.sleep(0.01)
    _pump(app, 0.2)

    del big
    gc.collect()
    assert ref() is None, "the worker still pins its argument after the run finished"


# ── Phase-2 panels ───────────────────────────────────────────────────────────────────────────────
def test_fdt_panel_guard_translates_model_error_and_gate_admits_builtins():
    """FDT now supports HOPF/BP + additive-noise user models. The guard turns an FDTModelError (a
    missing FDT parameter, or a user model with multiplicative/zero observable noise) into a readable
    RuntimeError instead of a bare traceback; the registry gate admits every built-in."""
    import core.gui.panels.fdt_panel as fdt_panel
    from core.FDT.campaigns import FDTModelError
    from core import registry

    class Cfg:
        model = "HOPF"

    def boom(cfg, *, skip_sanity, confirm_production):
        raise FDTModelError("Observable 'x' has state-dependent (multiplicative) noise; FDT supports "
                            "additive-noise observables only.")

    real, fdt_panel.run_fdt = fdt_panel.run_fdt, boom
    try:
        try:
            fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False)
        except RuntimeError as e:
            assert "multiplicative" in str(e), str(e)
        except FDTModelError:
            raise AssertionError("the FDTModelError escaped the guard unwrapped")
        else:
            raise AssertionError("the guard swallowed the failure entirely")
    finally:
        fdt_panel.run_fdt = real

    # HOPF / BP / NADROWSKI are no longer rejected by the FDT gate.
    for m in ("NADROWSKI", "HOPF", "BP"):
        assert registry.fdt_support(m) == (True, ""), m


def test_an_unparseable_cell_does_not_brick_the_gui():
    """Dropping a cell into Resources/Cells/<model>/ that cli._parse_cell cannot read makes it raise a
    bare ValueError (NOT a UnitParseError). CrossValPanel prefills from _parse_cell in __init__, so
    that exception used to escape CrossValPanel() -> MainWindow() -> build_app(), and
    `python -m core.gui` died before the window ever appeared -- before app.py's excepthook was even
    installed.

    The probe used to be a cell with no sibling bounds file, which is no longer unparseable: a cell
    without a sibling now resolves to the model's shared master.txt (cli.resolve_bounds_for_cell), so
    the natural 'add my cell' action WORKS rather than merely degrading. The failure mode this test
    guards still exists though -- a cell that omits a parameter its bounds file declares -- so probe
    with that instead."""
    import shutil
    from pathlib import Path

    from core.config import CELL_PATH
    from core.gui import settings as st

    _app()
    src = Path(CELL_PATH) / "nadrowski" / "master_weak.txt"
    if not src.exists():
        return                                   # nothing to probe with
    probe = src.with_name("aaa_probe_unparseable.txt")   # sorts first => the picker selects it
    # Drop a declared ND parameter: bounds resolve fine, but the merge cannot fill the value.
    kept = [ln for ln in src.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith("beta ")]
    probe.write_text("\n".join(kept) + "\n", encoding="utf-8")
    try:
        # Isolate from the developer's real QSettings store. MainWindow() -> CrossValPanel.__init__
        # restores its saved cell selection; a cell saved from a previous GUI session would be reloaded
        # OVER our probe, so the picker would land on a valid cell and the prefill would parse fine --
        # masking the degrade path this test exists to check. A clean temp .ini defaults the picker to
        # the alphabetically-first entry (the probe), regardless of the machine's state.
        _temp_settings()
        from core.gui.main_window import MainWindow
        from core.gui.panels.crossval_panel import CrossValPanel
        window = MainWindow()                    # must not raise
        xval = window.panel(CrossValPanel)       # CrossVal now lives inside the FDT Analysis section
        assert "could not read cell" in xval.cell_values.text().lower(), \
            f"the bad cell should degrade the prefill label, got: {xval.cell_values.text()!r}"
    finally:
        probe.unlink(missing_ok=True)
        st.use_ini_file(None)


def test_plot_watcher_only_reports_pngs_written_after_start():
    """The FDT/Reduction/CrossVal runners never return their figure paths, so the panels pick them up
    off disk. Pre-existing figures must not be re-shown, and each new one must be emitted once."""
    import tempfile
    from pathlib import Path

    from core.gui.plot_watcher import NewPngWatcher

    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "old_plot_20260101_000000.png").write_bytes(b"stale")

        seen = []
        watcher = NewPngWatcher(d)
        watcher.png_ready.connect(lambda title, path: seen.append((title, Path(path).name)))
        watcher.start()

        (d / "fdt3d_vs_S_20260714_120301.png").write_bytes(b"new")
        watcher.stop()                      # stop() forces a final scan, ignoring the settle delay
        app.processEvents()

        assert seen == [("fdt3d vs S", "fdt3d_vs_S_20260714_120301.png")], seen


# ── Phase 3: cancellation ────────────────────────────────────────────────────────────────────────
def test_worker_cancelled_passes_through_except_exception():
    """The cancel exception must be a BaseException so the pipeline's many `except Exception` handlers
    (sbi, cross_validation, Worker.run itself) do not swallow it."""
    from core.gui.streams import WorkerCancelled

    try:
        try:
            raise WorkerCancelled()
        except Exception:  # noqa: BLE001
            raise AssertionError("`except Exception` caught the cancel -- it is not BaseException-derived")
    except WorkerCancelled:
        pass


def test_cancel_token_latches_and_leaves_tqdm_usable():
    """The token raises exactly ONCE (so tqdm's teardown write can finish), and after the cancel a NEW
    tqdm bar must be creatable on a fresh thread.

    That last part is the sharp edge: tqdm's refresh() manual-acquires its global write lock and
    manual-releases it (tqdm/std.py:1346-1349), so a raise from inside a redraw's write() skips the
    release and leaks the lock -- after which the next tqdm.__new__ deadlocks. redirect_streams' cancel
    teardown resets the lock; without that reset THIS TEST HANGS (which is exactly the production bug:
    cancel a run, start another, hang)."""
    import threading

    from core.gui.streams import CancelToken, WorkerCancelled, redirect_streams

    signals = WorkerSignals()
    token = CancelToken()
    token.requested.set()                      # cancel already requested when the run starts

    with redirect_streams(signals, token):
        try:
            for _ in tqdm(range(5), desc="Generating training data", leave=False):
                for _ in tqdm(range(5), desc="step (batch=8)", leave=False):
                    pass
        except WorkerCancelled:
            pass
    assert token.fired, "the token never fired"
    assert len(tqdm._instances) == 0, f"tqdm left {len(tqdm._instances)} stale bar(s)"

    # tqdm must not be wedged: make a bar on a fresh thread with a bounded join.
    made = []

    def _probe():
        b = tqdm(range(2), desc="probe", file=open(os.devnull, "w"))
        b.close()
        made.append(True)

    th = threading.Thread(target=_probe, daemon=True)
    th.start()
    th.join(3.0)
    assert made == [True], "tqdm deadlocked after a cancel -- the write lock was not recovered"


def test_dispatched_run_cancels_cleanly_and_a_later_run_still_works():
    """End to end: a run cancelled mid-flight emits `cancelled` (not `error`), ends not-busy with rows
    dropped and stray figures closed, clears the active token, and a fresh run afterwards completes."""
    import matplotlib.pyplot as plt

    app = _app()

    class P(BasePanel):
        pass

    panel = P()
    started = {"v": False}
    outcome = {"cancelled": 0, "error": 0, "result": []}

    def heavy(fig_sink=None):
        plt.figure()                          # a stray figure the cancel path must close
        for i in range(2000):
            for _ in tqdm(range(20), desc="step (batch=8)", leave=False):
                pass
            started["v"] = True
            print(f"epoch {i}")               # a write() checkpoint
            time.sleep(0.01)
        return "COMPLETED"

    panel.dispatch(heavy, on_result=lambda r: outcome["result"].append(r))
    for w in panel._workers:
        w.signals.cancelled.connect(lambda: outcome.__setitem__("cancelled", outcome["cancelled"] + 1))
        w.signals.error.connect(lambda *_a: outcome.__setitem__("error", outcome["error"] + 1))

    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and not started["v"]:
        app.processEvents()
        time.sleep(0.005)
    assert panel._busy and BasePanel._active_cancel is not None

    panel._request_cancel()
    assert panel._cancel.requested.is_set()
    assert panel.btn_cancel.text() == "Cancelling…" and not panel.btn_cancel.isEnabled()

    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and panel._busy:
        app.processEvents()
        time.sleep(0.005)
    _pump(app, 0.3)

    assert not panel._busy, "panel stuck busy after cancel"
    assert outcome["result"] == [], "the run COMPLETED instead of cancelling"
    assert outcome["cancelled"] == 1 and outcome["error"] == 0, outcome
    assert not panel.progress_pane._rows, "rows leaked after cancel"
    assert plt.get_fignums() == [], "stray figures not closed on cancel"
    assert BasePanel._active_cancel is None and not BasePanel._running
    assert "Run cancelled." in panel.log_pane.toPlainText()

    later = []
    panel.dispatch(lambda: "SECOND OK", on_result=later.append)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and (panel._busy or not later):
        app.processEvents()
        time.sleep(0.005)
    assert later == ["SECOND OK"], f"a run after a cancel did not complete: {later}"


def test_cancel_is_not_consumed_by_a_non_worker_thread():
    """tqdm's TMonitor daemon force-refreshes a quiet bar, writing to our stream from ITS thread. If
    that write consumed the cancel latch, it would raise where nobody catches it and leave the worker
    to sail past a fired latch -- silently losing the cancel. Only the armed (worker) thread may raise."""
    import threading
    from core.gui.streams import CancelToken, WorkerCancelled

    token = CancelToken()
    token.requested.set()
    out = {}

    def worker():
        token.arm()                            # redirect_streams arms on the worker thread
        # a non-worker ("monitor") write happens first and must NOT consume the latch
        other = threading.Thread(target=token.check)
        other.start()
        other.join()
        try:
            token.check()                      # the worker's own next write MUST still raise
        except WorkerCancelled:
            out["worker_raised"] = True

    th = threading.Thread(target=worker)
    th.start()
    th.join(3.0)
    assert out.get("worker_raised") is True, "the cancel was consumed by a non-worker thread and lost"


def test_inference_config_restore_with_a_stale_model_does_not_desync_the_bounds_picker():
    """A corrupt/version-skewed .ini with an unknown model must not leave the (Prior-tab) bounds picker
    pointing at a nonexistent folder while the Config combo shows a real default. The picker is repointed
    when the model is APPLIED, so drive that path."""
    from core.gui import settings as st
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    _temp_settings()
    try:
        qs = st.settings()
        qs.beginGroup("inference_config")
        qs.setValue("model", "NOT_A_REAL_MODEL")
        qs.endGroup()
        qs.sync()

        screen = InferenceScreen()
        model = screen.config_panel.model_combo.currentText()
        assert model in ("BP", "NADROWSKI", "HOPF"), model
        screen.config_panel._build_config()                 # apply -> new_draft -> repoint the picker
        picker = screen.prior_panel.bounds_picker
        assert picker.base_path.name == model.lower()
        assert picker.combo.count() > 0, "the bounds picker was left empty by a stale model"
    finally:
        st.use_ini_file(None)


# ── Phase 3: QSettings persistence ───────────────────────────────────────────────────────────────
def _temp_settings():
    import tempfile
    from core.gui import settings as st
    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    st.use_ini_file(path)
    return path


def test_settings_round_trip_reduction_and_fdt():
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.panels.fdt_panel import FdtPanel

    _app()
    _temp_settings()
    try:
        red = ReductionPanel()
        red.f0.setText("0.123")
        if red.cell_picker.combo.count():
            red.cell_picker.combo.setCurrentIndex(red.cell_picker.combo.count() - 1)
        want_cell = red.cell_picker.key()

        fdt = FdtPanel()
        fdt.n_freqs.setText("77")
        fdt.skip_sanity.setChecked(True)
        fdt.confirm_production.setChecked(False)

        qs = st.settings()
        red.save_settings(qs)
        fdt.save_settings(qs)
        qs.sync()

        red2 = ReductionPanel()
        fdt2 = FdtPanel()
        assert red2.f0.value() == 0.123
        assert red2.cell_picker.key() == want_cell
        assert fdt2.n_freqs.value() == 77
        assert fdt2.skip_sanity.isChecked() is True
        assert fdt2.confirm_production.isChecked() is False
    finally:
        st.use_ini_file(None)


def test_missing_picker_key_restores_to_default_not_blank():
    """A saved selection whose file is gone must leave the picker at its default, never blank it via
    setCurrentIndex(-1)."""
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel

    _app()
    _temp_settings()
    try:
        qs = st.settings()
        qs.beginGroup("reduction")
        qs.setValue("cell", "nadrowski/does_not_exist.txt")
        qs.setValue("f0", "0.05")
        qs.endGroup()
        qs.sync()

        red = ReductionPanel()
        assert red.cell_picker.combo.currentIndex() >= 0, "a stale key blanked the combo"
    finally:
        st.use_ini_file(None)


def test_crossval_does_not_persist_cell_derived_bounds():
    """The S/T grid lo/hi are re-derived from the cell file; a saved value from a different cell would
    be a stale, wrong bound. Only the free knobs (points, f0, freqs_per_batch, preset, cell) persist."""
    from core.gui import settings as st
    from core.gui.panels.crossval_panel import CrossValPanel

    _app()
    _temp_settings()
    try:
        xv = CrossValPanel()
        derived_hi = xv.s_grid.hi.text()          # set by _on_cell_changed from the cell file
        xv.s_grid.hi.setText("999.0")             # user 'edits' it to a bogus value
        xv.s_grid.points.setText("13")
        xv.f0.setText("0.077")

        qs = st.settings()
        xv.save_settings(qs)
        qs.sync()
        # the bogus hi must NOT have been written
        qs.beginGroup("crossval")
        assert qs.value("s_hi") is None, "cell-derived s_grid.hi was persisted -- it must not be"
        qs.endGroup()

        xv2 = CrossValPanel()
        assert xv2.s_grid.points.text() == "13", "the free `points` knob was not restored"
        assert xv2.f0.value() == 0.077
        assert xv2.s_grid.hi.text() == derived_hi, "the grid bound must be RE-DERIVED, not restored"
    finally:
        st.use_ini_file(None)


# ── Phase 3: error dialogs ───────────────────────────────────────────────────────────────────────
def test_on_error_puts_the_traceback_in_details_not_the_body(monkeypatch=None):
    """A run failure's traceback belongs in a collapsible Details panel, not pasted whole into the
    dialog body."""
    from PySide6.QtWidgets import QMessageBox

    _app()

    class P(BasePanel):
        pass

    panel = P()
    captured = {}
    orig_exec = QMessageBox.exec

    def fake_exec(self):
        captured["text"] = self.text()
        captured["detail"] = self.detailedText()
        return 0

    QMessageBox.exec = fake_exec
    try:
        panel._on_error("Something failed", "Traceback (most recent call last):\n  ...\nValueError: x")
    finally:
        QMessageBox.exec = orig_exec

    assert captured["text"] == "Something failed"
    assert "Traceback" in captured["detail"], "the traceback was not routed to Details"
    assert "Traceback" not in captured["text"], "the traceback leaked into the body"


# ── interactive "Pop out" for figures ────────────────────────────────────────────────────────────
def _tiny_fig():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig


def _png_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def test_fig_sink_emits_png_and_a_reloadable_pickle():
    """The worker sink must emit BOTH the PNG thumbnail and a pickle that reloads to a real Figure on
    the GUI thread -- that pickle is what "Pop out" rebuilds into an interactive window."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.panels.base_panel import _png_fig_sink

    _app()
    sig = WorkerSignals()
    events = []
    sig.figure.connect(lambda title, png, fp: events.append((title, png, fp)))

    _png_fig_sink(sig.figure)("Corner", _tiny_fig())

    assert len(events) == 1, events
    title, png, fp = events[0]
    assert title == "Corner"
    assert png[:4] == b"\x89PNG", "the PNG thumbnail is missing / not a PNG"
    assert fp is not None, "no pickle was shipped for the pop-out"
    assert len(pickle.loads(fp).axes) == 1, "the pickle did not reload to the figure"
    plt.close("all")


def test_fig_sink_pickle_failure_still_emits_the_png():
    """If a figure will not pickle, the run must not break: emit fig_pickle=None and keep the PNG."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.panels.base_panel import _png_fig_sink

    _app()
    sig = WorkerSignals()
    events = []
    sig.figure.connect(lambda title, png, fp: events.append((title, png, fp)))

    real = pickle.dumps
    pickle.dumps = lambda *a, **k: (_ for _ in ()).throw(TypeError("cannot pickle this"))
    try:
        _png_fig_sink(sig.figure)("X", _tiny_fig())
    finally:
        pickle.dumps = real

    assert len(events) == 1
    _title, png, fp = events[0]
    assert fp is None, "a pickle failure should degrade to None, not raise or drop the event"
    assert png[:4] == b"\x89PNG", "the PNG must still be emitted when pickling fails"
    plt.close("all")


def test_add_figure_creates_an_interactive_capable_tab():
    import pickle
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QPushButton
    from core.gui.widgets.figure_stack import FigureStack

    _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    assert fs.count() == 1
    container = fs.widget(0)
    assert getattr(container, "_fig_pickle", None) is not None, "the pickle was not stored on the tab"
    assert any(b.text() == "Pop out" for b in container.findChildren(QPushButton)), "no Pop out button"


def test_pop_out_of_a_pickle_builds_a_qtagg_canvas_and_keeps_pyplot_clean():
    """Popping a pickled figure builds a live FigureCanvasQTAgg on the GUI thread, and -- the linchpin
    -- must NOT leave the reconstructed figure registered in pyplot's Gcf (else the worker's
    plt.close("all") would later close it out from under the user)."""
    import pickle
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from core.gui.widgets.figure_stack import FigureStack

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    before = plt.get_fignums()
    fs._pop_out(fs.widget(0))
    assert len(fs._windows) == 1
    win = next(iter(fs._windows))
    assert isinstance(win.canvas, FigureCanvasQTAgg)
    assert plt.get_fignums() == before, "the unpickled figure leaked into pyplot's Gcf"

    win.close()
    _pump(app, 0.1)
    assert len(fs._windows) == 0, "the window ref was not dropped on close"
    plt.close("all")


def test_pop_out_without_a_pickle_uses_the_image_viewer():
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QGraphicsView
    from core.gui.widgets.figure_stack import FigureStack
    from core.gui.widgets.figure_window import ImageZoomWindow

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("NoPickle", _png_bytes(fig), fig_pickle=None)
    plt.close("all")

    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert isinstance(win, ImageZoomWindow), "a pickle-less figure should open the image viewer"
    assert isinstance(win.view, QGraphicsView)
    win.close()
    _pump(app, 0.1)


def test_disk_png_pop_out_is_an_image_viewer():
    import tempfile
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QGraphicsView
    from core.gui.widgets.figure_stack import FigureStack
    from core.gui.widgets.figure_window import ImageZoomWindow

    app = _app()
    png = Path(tempfile.mkdtemp()) / "sweep.png"
    fig = _tiny_fig()
    fig.savefig(str(png), format="png")
    plt.close("all")

    fs = FigureStack()
    fs.add_png("Sweep", str(png))
    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert isinstance(win, ImageZoomWindow)
    assert isinstance(win.view, QGraphicsView), "the disk-PNG pop-out has no zoom/pan view"
    win.close()
    _pump(app, 0.1)


def test_a_popped_out_figure_survives_a_worker_plt_close_all():
    """Worker.run runs plt.close("all") after every run and every cancel. A figure the user has popped
    out must NOT be torn down by it -- the Gcf detach in build_interactive_window is the guarantee."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.widgets.figure_stack import FigureStack

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert len(win._fig.axes) == 1

    plt.close("all")     # the worker's teardown, fired process-wide
    assert len(win._fig.axes) == 1, "plt.close('all') destroyed a figure being viewed in a pop-out"

    win.close()
    _pump(app, 0.1)
    plt.close("all")


# ── PRISM navigation redesign ────────────────────────────────────────────────────────────────────
def test_greeting_maps_hours_to_time_of_day():
    from core.gui.screens.home_screen import greeting
    assert greeting(5) == greeting(11) == "Good morning"
    assert greeting(8) == "Good morning"
    assert greeting(12) == greeting(16) == "Good afternoon"
    assert greeting(14) == "Good afternoon"
    assert greeting(17) == greeting(23) == "Good evening"
    assert greeting(0) == greeting(4) == "Good evening"


def test_nav_shell_back_arrow_tracks_the_screen():
    from PySide6.QtWidgets import QWidget
    from core.gui.screens.nav_shell import NavShell

    _app()
    nav = NavShell()
    for _ in range(3):
        nav.add_screen(QWidget())
    nav.go_home()
    assert nav.btn_back.isHidden(), "back arrow should be hidden on home"
    nav.go_to(2)
    assert not nav.btn_back.isHidden(), "back arrow should show on a section"
    nav.go_home()
    assert nav.btn_back.isHidden(), "back arrow should hide again on home"


def test_main_window_always_opens_on_home():
    from core.gui import settings as st
    _app()
    _temp_settings()
    try:
        qs = st.settings()
        qs.setValue("window/tab", 2)          # a stale key from the old flat-tab layout
        qs.sync()
        from core.gui.main_window import MainWindow
        w = MainWindow()
        assert w.nav.stack.currentIndex() == 0, "the app must always open on the home screen"
    finally:
        st.use_ini_file(None)


def test_inference_tab_gates_follow_the_session():
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    _app()
    inf = InferenceScreen()

    def enabled():
        return [inf.tabs.isTabEnabled(i) for i in range(5)]   # Config Prior Posterior Validate Infer

    inf.session = SbiSession(); inf.refresh_gates()
    assert enabled() == [True, False, False, False, False]
    # A DRAFT (model applied) unlocks Prior -- which is where the bounds file builds the config.
    inf.session.draft = object(); inf.refresh_gates()
    assert enabled() == [True, True, False, False, False]
    inf.session.cfg = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, False, False]
    inf.session.posterior = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, False, True], "Infer needs only a posterior; Validate needs a prior"
    # force_prior stays None on purpose: it is None for every no-forcing model, and requiring it used to
    # make Validate permanently unreachable for them.
    inf.session.inf_prior = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, True, True]


def test_tsnpe_tab_is_gated_and_never_proposes_from_the_posterior():
    """The TSNPE tab needs a posterior, its prior AND an observation on disk -- and its round must go
    through orchestrator.build_posterior with a TRUNCATION, never by fitting the posterior.

    The second half is the one worth a test: proposing from the posterior instead of the truncated
    prior is TEMPERING, it contracts credible intervals with no new information, and SBC comes out
    flat anyway because it validates the flow against the proposal it was trained on. Nothing on the
    Validate tab would catch it, so the wiring is pinned here and the maths in
    tests/test_conditioning_repair.py.
    """
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.gui.panels import inference_tabs

    _app()
    inf = InferenceScreen()
    assert inf.tabs.count() == 6 and inf.tabs.tabText(5) == "TSNPE"

    inf.session = SbiSession(cfg=object(), posterior=object()); inf.refresh_gates()
    assert not inf.tabs.isTabEnabled(5), "TSNPE must not open on a posterior alone"
    inf.session.inf_prior = object(); inf.refresh_gates()
    assert inf.tabs.isTabEnabled(5), "TSNPE opens once a posterior and its prior exist"

    # The observation gate, driven through the picker's own accessor rather than through the real
    # Resources/Observations directory. Clearing the combo does NOT work: refresh_local_gates calls
    # obs_picker.refresh(), which repopulates it from disk -- so the assertion passed only while that
    # directory happened to be empty, and any suite run that had exercised infer_and_visualize left a
    # record behind and flipped it. A gate test must not depend on what an earlier test wrote.
    panel = inf.tsnpe_panel
    panel.obs_picker.key = lambda: ""                    # nothing recorded yet
    panel.refresh_local_gates()
    assert not panel.btn_round.isEnabled(), "a round must be impossible without an observation"
    panel.obs_picker.key = lambda: "obs_20260101T000000_deadbeefdeadbeef.pt"
    panel.refresh_local_gates()
    assert panel.btn_round.isEnabled(), "with a posterior, its prior and an observation, a round is allowed"

    # The runner's contract: it hands build_posterior a truncation region and nothing else refits.
    # Checked against the CODE, with the docstring stripped -- that docstring necessarily contains the
    # word "proposal" while explaining what must not happen, and a naive text search on the whole
    # source flags the very comment that documents the rule.
    tree = ast.parse(textwrap.dedent(inspect.getsource(inference_tabs._run_tsnpe_round)))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str)):
        fn.body = fn.body[1:]
    code = ast.unparse(tree)
    assert "build_truncation_region" in code and "truncation=region" in code,         "the TSNPE runner does not build a truncation region and pass it to build_posterior"
    for banned in ("set_default_x", "proposal"):
        assert banned not in code, (
            f"the TSNPE runner's CODE references '{banned}' -- it must sample the truncated PRIOR, "
            f"never the posterior; that is tempering, and SBC cannot detect it")


def test_a_tsnpe_posterior_cannot_be_saved_as_amortized():
    """⚠ SECTION 11.6 GUARDRAIL 2, at the seam where it is easiest to lose.

    The GUI trains with save=False and saves LATER from a button, so the truncation region has to
    survive on the session or the deferred save writes the artifact marked `amortized: True` --
    indistinguishable, in the same ArtifactPicker, from a genuinely amortized posterior. That is the
    class of confusion the retired-band posterior already cost a five-day run for.

    Three things, and the third is the one that is easy to miss: the round INSTALLS the region, the
    save PASSES it, and training an ordinary posterior afterwards CLEARS it.
    """
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.gui.panels import inference_tabs

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=object())
    panel = inf.tsnpe_panel

    region = object()
    panel._on_round(((object(), {"loss": []}), region, "deadbeefdeadbeef"))
    assert inf.session.truncation is region, "the round did not install its truncation region"
    assert inf.session.x_obs_digest == "deadbeefdeadbeef"
    assert inf.session.posterior is not None, "the round's posterior never reached the session"

    # the deferred save must forward both
    captured = {}
    pp = inf.posterior_panel
    pp.dispatch = lambda fn, *a, **k: captured.update(args=a, kwargs=k)
    pp.post_name.setText("some_name")
    inf.session.posterior_latent = object()
    pp._save_posterior()
    assert captured.get("kwargs", {}).get("truncation") is region,         "save_posterior_artifacts was called WITHOUT the truncation -- it would be marked amortized"
    assert captured["kwargs"].get("x_obs_digest") == "deadbeefdeadbeef"

    # and an ordinary train afterwards must CLEAR it, or the mislabelling runs the other way
    pp._on_posterior((object(), {"loss": []}))
    assert inf.session.truncation is None and inf.session.x_obs_digest is None,         "an amortized posterior inherited the previous round's truncation"


def test_the_new_tab_knobs_are_forwarded_and_not_written_to_config():
    """Prior, Posterior and Validate all gained fields. Each must reach its orchestrator function as
    an ARGUMENT -- orchestrator snapshots those constants at import, so writing them would be a
    silent no-op and the run would use the defaults with nothing to say so."""
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core import config as _cfg

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession(draft=object(), cfg=object(), inf_prior=object(), posterior=object())

    cap = {}
    for panel in (inf.prior_panel, inf.posterior_panel, inf.validate_panel):
        panel.dispatch = lambda fn, *a, **k: cap.update(kwargs=k)

    inf.validate_panel.cal_n.setText("77")
    inf.validate_panel.cal_scales.setText("11")
    inf.validate_panel._validate()
    assert cap["kwargs"]["n_cal"] == 77 and cap["kwargs"]["cal_n_scales"] == 11, cap["kwargs"]
    assert _cfg.SBC_N_CAL != 77, "the panel wrote the config constant instead of passing an argument"

    inf.posterior_panel.flow_hidden.setText("64")
    inf.posterior_panel.flow_transforms.setText("3")
    inf.posterior_panel.flow_patience.setText("7")
    inf.posterior_panel.post_picker.combo.setCurrentIndex(0)
    inf.posterior_panel._build_posterior()
    k = cap["kwargs"]
    assert (k["hidden_features"], k["num_transforms"], k["stop_after_epochs"]) == (64, 3, 7), k
    assert _cfg.NSF_HIDDEN_FEATURES != 64, "the panel wrote NSF_HIDDEN_FEATURES"
    # the Fisher knobs ride along on the same call
    assert (k["fisher_m"], k["fisher_points"]) == (_cfg.REPARAM_FISHER_M, _cfg.REPARAM_FISHER_POINTS)

    inf.prior_panel.cluster_size.setText("9")
    inf.prior_panel.cluster_samples.setText("4")
    inf.prior_panel.sweep_iters.setText("3")
    inf.prior_panel.bounds_source.set_direct(False)
    inf.prior_panel.bounds_picker.combo.clear()
    inf.prior_panel.bounds_picker.combo.addItem("master.txt")
    inf.session.draft = type("D", (), {"make_config": lambda self, **kw: inf.session.cfg})()
    try:
        inf.prior_panel._build_prior()
    except Exception:
        pass                                  # config construction is stubbed; the dispatch is the point
    if "min_cluster_size" in cap.get("kwargs", {}):
        assert cap["kwargs"]["min_cluster_size"] == 9 and cap["kwargs"]["min_samples"] == 4, cap["kwargs"]
        assert _cfg.PRIOR_CLUSTER_MIN_SIZE != 9, "the panel wrote PRIOR_CLUSTER_MIN_SIZE"

def test_posterior_from_scratch_is_gated_on_a_prior():
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object()); inf.refresh_gates()
    pp = inf.posterior_panel
    pp.post_picker.combo.setCurrentIndex(0)               # the "(from scratch)" sentinel (allow_new adds it first)
    assert pp.post_picker.selected()[1] is True, "index 0 should be the from-scratch sentinel"
    pp.refresh_local_gates()
    assert not pp.btn_post.isEnabled(), "training from scratch must be disabled without a prior"
    inf.session.inf_prior = object(); inf.session.force_prior = object()
    pp.refresh_local_gates()
    assert pp.btn_post.isEnabled(), "with a prior, training from scratch is allowed"


def test_inference_pickers_repoint_from_draft_and_config():
    """The Prior tab's BOUNDS picker follows the applied model (new_draft), and the Infer tab's CELL
    picker follows the BUILT config's model (install_config) -- neither tab has its own model combo."""
    from core import config
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import ConfigDraft

    _app()
    inf = InferenceScreen()

    inf.new_draft(ConfigDraft(model="HOPF", labels=[], state_dep_drift=False))
    assert inf.prior_panel.bounds_picker.base_path.name == "hopf"
    assert inf.session.cfg is None, "a draft alone must not produce a config"

    class Cfg:
        model = "HOPF"
        params_dict = {}      # the Infer tab validates a picked cell against these
        rescale_params = {}
        force_params_dict = {}
        has_forcing = False   # mirrors SimConfig.has_forcing (empty force_params_dict)
        chi_mode = False      # mirrors SimConfig.chi_mode
        chi_k_pad = config.CHI_K_PAD    # mirrors SimConfig.chi_k_pad (probe-slot capacity)
        observation_mode = "spontaneous"

    inf.install_config(Cfg())
    assert inf.infer_panel.cell_picker.base_path.name == "hopf"
    assert inf.session.draft is not None, "install_config must not wipe the draft/session"


def _chi_cfg(k=3, pad=12):
    """Stub config that puts the Infer tab on its chi page (mirrors the SimConfig fields it reads)."""
    from core import config

    class Cfg:
        model = "NADROWSKI"
        params_dict = {}
        rescale_params = {}
        force_params_dict = {}
        has_forcing = False
        chi_mode = True
        chi_n_freqs = k
        chi_k_pad = pad
        chi_freq_bounds = config.CHI_FREQ_BOUNDS
        chi_max_cycles = config.CHI_MAX_CYCLES
        observation_mode = "chi"
    return Cfg()


def test_chi_probe_table_is_variable_length_and_capped_by_the_posteriors_slots():
    """Backlog C-2. The core has always accepted 1..chi_k_pad probes at arbitrary frequencies; the GUI
    was the only thing forcing a fixed grid. Rows must be addable and removable, and the cap must be
    chi_k_pad -- which is FROZEN into the trained artifact, so exceeding it is not a soft limit."""
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=3, pad=5))
    panel = inf.infer_panel
    assert len(panel._chi_forced_fields) == 3, "should seed chi_n_freqs rows when empty"

    panel._add_chi_probe()
    panel._add_chi_probe()
    assert len(panel._chi_forced_fields) == 5
    panel._add_chi_probe()                                   # over capacity
    assert len(panel._chi_forced_fields) == 5, "must refuse to exceed chi_k_pad slots"

    panel._remove_chi_probe(panel._chi_forced_fields[0])
    assert len(panel._chi_forced_fields) == 4


def test_chi_probe_rows_keep_each_recording_paired_with_its_own_frequency():
    """The one-widget-per-row invariant. Parallel path/frequency lists let a MIDDLE deletion pair
    recording k with frequency k+1 -- silent, because a lock-in at the wrong frequency decays like a
    sinc and simply returns a smaller number. Delete from the middle and check the survivors."""
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=4))
    panel = inf.infer_panel
    for i, row in enumerate(panel._chi_forced_fields):
        row.path.edit.setText(f"/tmp/rec{i}.csv")
        row.freq.setText(str(1.0 + i))

    panel._remove_chi_probe(panel._chi_forced_fields[1])      # delete from the MIDDLE
    pairs = [r.pair() for r in panel._chi_forced_fields]
    assert pairs == [("/tmp/rec0.csv", 1.0), ("/tmp/rec2.csv", 3.0), ("/tmp/rec3.csv", 4.0)], pairs


def test_chi_probe_table_survives_a_config_rebuild_and_rejects_a_blank_frequency():
    """Two C-2 constraints in one place, because both are about data the GUI cannot regenerate.

    PRESERVATION: rows carry hand-typed drive frequencies and browsed paths -- a record of a bench
    session that already happened. Rebuilding the config (to fix a bounds file, say) must not discard
    them, unlike the forcing rows, which ARE derivable from the config.

    BLANK FREQUENCY: FloatField.value() returns 0.0 on unparseable text, so an empty box is
    indistinguishable from a deliberate zero -- and 0 Hz is a genuine DC probe the lock-in would
    happily attempt. It has to be caught before the run, not after.
    """
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=2))
    panel = inf.infer_panel
    panel._chi_forced_fields[0].path.edit.setText("/tmp/keep.csv")
    panel._chi_forced_fields[0].freq.setText("2.5")

    inf.install_config(_chi_cfg(k=2))                         # rebuild
    assert len(panel._chi_forced_fields) == 2, "a rebuild must not add or drop rows"
    assert panel._chi_forced_fields[0].pair() == ("/tmp/keep.csv", 2.5), \
        "a rebuild destroyed hand-entered probe data"

    # Row 1 still has a blank frequency -> 0.0 -> must be reported, naming the row.
    probs = [p for i, r in enumerate(panel._chi_forced_fields) for p in r.problems(i)]
    assert any("probe 2" in p and "positive" in p for p in probs), probs
    assert not any("probe 1" in p for p in probs), probs


def test_config_units_control_declares_units_and_validates_them():
    """Units DECLARE what the numbers in the files mean (never converting them). Typed units must reach
    the built config, and unusable tokens must be rejected when the model is applied -- not mid-run."""
    from core.config import BOUNDS_PATH
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    screen = InferenceScreen()
    cfgp = screen.config_panel
    cfgp.model_combo.setCurrentText("NADROWSKI")
    cfgp._on_model_changed("NADROWSKI")

    cfgp.units_toggle.set_direct(False)                       # model's units file
    assert cfgp._units_override() is None

    cfgp.units_toggle.set_direct(True)                        # typed
    cfgp.units_text.setText("nm s pN Hz")
    assert cfgp._units_override() == ("nm", "s", "pN", "Hz")
    cfgp._build_config()
    draft = screen.session.draft
    assert draft is not None and draft.units_override == ("nm", "s", "pN", "Hz")

    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    if bounds.exists():
        cfg = draft.make_config(str(bounds))
        assert (cfg.time_unit, cfg.freq_unit) == ("s", "Hz")
        # Hz alongside s IS self-consistent (unlike Hz alongside ms), so no warning
        assert cfg.check_unit_consistency() == []
        assert abs(30.0 * cfg.freq_si_to_cell - 30.0) < 1e-12, "in an `s` cell, 30 Hz stays 30"

    before = screen.session.draft
    cfgp.units_text.setText("notaunit")                       # must be refused, draft left alone
    cfgp._build_config()
    assert screen.session.draft is before


def test_direct_entry_grids_round_trip_their_files():
    """Hand-entered bounds/values must reproduce the parsed file exactly (same names, same ORDER --
    simulators bind parameter columns positionally), and must refuse unparseable or inverted input."""
    from collections import OrderedDict
    from core.config import BOUNDS_PATH, CELL_PATH
    from core.Helpers import file_manager
    from core.gui.widgets.param_grid import BoundsGrid, ValuesGrid

    _app()
    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    cell = CELL_PATH / "nadrowski" / "master_weak.txt"
    if not (bounds.exists() and cell.exists()):
        return                                                   # environment without Resources: skip

    p, r, f, _ = file_manager.parse_bounds_file(str(bounds))
    grid = BoundsGrid()
    assert grid.problems(), "an unloaded grid must report a problem rather than build nothing"
    grid.load(p, r, f)
    assert grid.problems() == []
    gp, gr, gf = grid.to_dicts()
    for got, want in ((gp, p), (gr, r), (gf, f)):
        assert list(got) == list(want), "parameter ORDER must survive the round trip"
        for name in want:
            assert got[name][1] == want[name][1], name

    grid._rows[("PARAM", "k")].lo.setText("")                    # unparseable -> refused, not 0.0
    assert any("k" in m for m in grid.problems())
    grid._rows[("PARAM", "k")].lo.setText("9")                   # min > max -> refused
    grid._rows[("PARAM", "k")].hi.setText("1")
    assert any("less than" in m for m in grid.problems())

    inits, vp, vr, vf = file_manager.parse_values_file(str(cell))
    vgrid = ValuesGrid()
    vgrid.load(inits, vp, vr, vf)
    assert vgrid.problems() == []
    gi, gvp, gvr, gvf = vgrid.to_dicts()
    assert list(gvp) == list(vp) and list(gi) == list(inits)
    assert all(abs(gvp[n] - vp[n]) < 1e-12 for n in vp)
    assert isinstance(gvp, OrderedDict)


def test_overlay_alignment_and_best_fit_rankings():
    """Phase alignment must recover a known lag; both best-fit rankings must find a planted match; and
    the trace ranking must be invariant to rolling the reference (absolute phase carries no info)."""
    import math
    import numpy as np
    import torch
    from core.SBI import overlay

    torch.manual_seed(0)
    n, dt, f = 2048, 1e-3, 7.5
    t = torch.arange(n, dtype=torch.float32) * dt

    # unambiguous (non-periodic) reference -> the lag is exact
    chirp = torch.sin(2 * math.pi * (2.0 + 6.0 * t) * t) * torch.hann_window(n)
    shifts = torch.tensor([0, 37, -91])
    rolled = torch.stack([torch.roll(chirp, int(s)) for s in shifts])
    aligned, lags = overlay.align_to(chirp, rolled)
    assert torch.equal(lags, shifts), (lags, shifts)
    assert float((aligned - chirp.unsqueeze(0)).abs().max()) < 1e-5

    gt = torch.sin(2 * math.pi * f * t) + 0.05 * torch.randn(n)
    cand = torch.stack([torch.sin(2 * math.pi * (f * 1.3) * t),      # wrong frequency
                        0.4 * torch.sin(2 * math.pi * f * t),        # wrong amplitude
                        torch.roll(gt, 123)])                        # the planted match, phase-shifted
    order, _rmse, _al = overlay.rank_by_trace(gt, cand)
    assert int(order[0]) == 2, "the planted draw must win on waveform"
    order2, _, _ = overlay.rank_by_trace(torch.roll(gt, -500), cand)
    assert int(order2[0]) == 2, "ranking must not depend on the reference's absolute phase"

    # summary-stat ranking: exact match wins, and the CONSTANT conditioning column is ignored
    obs = torch.tensor([[1.0, 2.0, 3.0, 42.0]])
    sim = torch.tensor([[9.0, 9.0, 9.0, 42.0], [1.0, 2.0, 3.0, 42.0], [1.5, 2.5, 2.0, 42.0]])
    o, d = overlay.rank_by_stats(sim, obs)
    assert int(o[0]) == 1 and float(d[1]) < 1e-9

    # phase-invariant summaries are well-formed
    freqs, lo, med, hi, dropped = overlay.psd_band(cand, dt)
    assert bool((lo <= hi).all()) and abs(float(freqs[med.argmax()]) - f) < 2.0
    centres, mean, clo, chi_, c_dropped = overlay.cycle_average(gt.unsqueeze(0), dt, f)
    assert dropped == 0 and c_dropped == 0, "clean traces must drop nothing"
    good = torch.isfinite(mean)
    assert int(good.sum()) > 0.8 * len(mean)
    assert 1.5 < float(mean[good].max() - mean[good].min()) < 2.5, "should recover the unit amplitude"
    assert overlay.cycle_window(n, dt, f, 15) == 2000



def test_a_divergent_draw_does_not_erase_the_whole_psd_band():
    """THE 2026-08-25 FAILURE, pinned.

    The power-spectrum figure came back as a bare observation line: no band, no median, no message.
    Cause: a broad posterior samples parameter sets that do not integrate stably, `|rfft|**2` of such a
    trace overflows to inf, and `torch.quantile` propagates one non-finite entry across every column --
    so ONE bad draw in a thousand silently erased the band for all of them.

    It went unnoticed for so long because the two sibling figures survive it: the overlay band takes
    only the 50 best draws, and cycle_average confines the damage to a single phase bin. So the
    symptom looked like a plotting bug in one figure rather than a property of the posterior.
    """
    import math
    import torch
    from core.SBI import overlay

    n, dt, f = 4096, 1.0 / 500.0, 12.0
    t = torch.arange(n, dtype=torch.float64) * dt
    good = torch.stack([torch.sin(2 * math.pi * f * t) * a for a in (0.9, 1.0, 1.1, 1.05, 0.95)])

    clean_freqs, clean_lo, clean_med, clean_hi, clean_drop = overlay.psd_band(good, dt)
    assert clean_drop == 0 and torch.isfinite(clean_med).all()

    for label, bad_row in (("inf", torch.full((n,), float("inf"), dtype=torch.float64)),
                           ("nan", torch.full((n,), float("nan"), dtype=torch.float64)),
                           ("overflow", torch.sin(2 * math.pi * f * t) * 1e300)):
        traces = torch.cat([good, bad_row.unsqueeze(0)], dim=0)
        freqs, lo, med, hi, dropped = overlay.psd_band(traces, dt)
        assert dropped == 1, f"{label}: expected 1 dropped draw, got {dropped}"
        assert torch.isfinite(med).all(), f"{label}: one bad draw still poisoned the median"
        assert torch.isfinite(lo).all() and torch.isfinite(hi).all(), f"{label}: band not finite"
        assert bool((lo <= hi).all())
        # and the surviving band is the clean one, not some rescued average of the wreckage
        assert torch.allclose(med, clean_med), f"{label}: the good draws' band changed"


def test_the_psd_band_reports_when_nothing_survives():
    """All-bad input must produce NaNs and a count, not an exception and not a silent empty plot."""
    import torch
    from core.SBI import overlay

    traces = torch.full((4, 1024), float("nan"), dtype=torch.float64)
    freqs, lo, med, hi, dropped = overlay.psd_band(traces, 1.0 / 500.0)
    assert dropped == 4
    assert not torch.isfinite(med).any() and len(med) == len(freqs)


def test_the_cycle_average_masks_non_finite_samples_instead_of_binning_them():
    """A NaN phase goes through `.long()` as a garbage integer that `clamp` parks in bin 0, so an
    unmasked divergent draw silently corrupts one end of the cycle -- which reads as a real feature."""
    import math
    import torch
    from core.SBI import overlay

    n, dt, f = 4096, 1.0 / 500.0, 12.0
    t = torch.arange(n, dtype=torch.float64) * dt
    good = torch.sin(2 * math.pi * f * t).unsqueeze(0)

    _, m_clean, _, _, d_clean = overlay.cycle_average(good, dt, f)
    assert d_clean == 0

    dirty = torch.cat([good, torch.full((1, n), float("nan"), dtype=torch.float64)], dim=0)
    _, m_dirty, lo_d, hi_d, d_dirty = overlay.cycle_average(dirty, dt, f)
    assert d_dirty == n, f"expected the whole bad row masked, got {d_dirty}"
    live = torch.isfinite(m_clean)
    assert torch.allclose(m_dirty[live], m_clean[live]), "the good row's cycle changed"
    assert torch.isfinite(m_dirty[live]).all(), "bin 0 was poisoned by the NaN row"


def test_an_empty_predictive_band_is_annotated_on_the_psd_figure():
    """An absent band must not be mistakable for a rendering glitch -- which is exactly what happened.
    When nothing finite survives, the figure has to say so in the axes."""
    import numpy as np
    from core.Helpers import visualizers

    _app()
    freqs = np.linspace(0.1, 100, 64)
    nan = np.full(64, np.nan)
    fig = visualizers.plot_psd_overlay(freqs, np.ones(64), nan, nan, nan, n_dropped=7)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("no finite" in t for t in texts), f"no explanation drawn: {texts}"
    assert any("7" in t for t in texts), f"the dropped count is not on the figure: {texts}"

    # and with a healthy band there must be no such annotation
    ok = visualizers.plot_psd_overlay(freqs, np.ones(64), np.ones(64) * .5, np.ones(64),
                                      np.ones(64) * 2)
    assert not [t.get_text() for t in ok.axes[0].texts], "annotated a perfectly good band"


def test_overlay_figures_render_and_are_picklable():
    """Each new Infer-tab figure must build and survive pickling (the 'Pop out' path unpickles it)."""
    import pickle
    import numpy as np
    from core.Helpers import visualizers

    _app()
    t = np.linspace(0, 2, 400)
    y = np.sin(2 * np.pi * 7.5 * t)
    labels_ = [f"$p_{{{i}}}$" for i in range(13)]
    vals = list(range(13))
    figs = [
        visualizers.plot_best_fit_overlay(t, y, y * 1.05, param_labels=labels_, param_values=vals,
                                          ground_truth=vals, criterion="closest summary statistics",
                                          score_text="RMS z = 0.4"),
        visualizers.plot_overlay_band(t, y, y - 0.2, y, y + 0.2, n_used=20),
        visualizers.plot_psd_overlay(np.linspace(0.1, 100, 50), np.ones(50), np.ones(50) * 0.5,
                                     np.ones(50), np.ones(50) * 2),
        visualizers.plot_cycle_average(np.linspace(0, 2 * np.pi, 48), np.zeros(48), np.zeros(48),
                                       -np.ones(48), np.ones(48)),
    ]
    for fig in figs:
        assert fig.axes, "figure has no axes"
        pickle.loads(pickle.dumps(fig))                  # must not raise
    # the 13-row parameter table gets its own axes, never the title
    assert len(figs[0].axes) == 2


def test_help_badge_carries_its_text():
    from core.gui.widgets.help_badge import HelpBadge
    _app()
    assert HelpBadge("what this does").toolTip() == "what this does"


def test_simulated_inference_runner_emits_the_ground_truth_figure():
    """The simulated-inference runner shows the 'Ground-truth trace' figure before inferring (the old
    Simulate tab did only the first half; the tab is gone, the figure is not). A real SDE sim is too slow
    for a unit test, so stub the heavy pieces and assert the fig_sink wiring."""
    import torch
    from core import cli, orchestrator
    from core.gui.panels import inference_tabs

    _app()

    class Cfg:
        length_unit = "nm"                       # trace y-axis unit (round-4 labels)

        def get_unit_conversion_factor(self, _unit):
            return 1.0

    seen = []
    real_gt, real_go = cli.load_and_validate_gt, orchestrator.generate_observations
    real_iv = orchestrator.infer_and_visualize
    cli.load_and_validate_gt = lambda cfg, path: []
    orchestrator.generate_observations = lambda cfg: (
        torch.zeros(1, 5), None, torch.linspace(0, 1, 5).unsqueeze(0))
    orchestrator.infer_and_visualize = lambda *a, **k: None
    try:
        inference_tabs._run_simulated_inference(
            Cfg(), object(), "cell.txt", 0.1, fig_sink=lambda title, fig: seen.append(title))
    finally:
        cli.load_and_validate_gt = real_gt
        orchestrator.generate_observations = real_go
        orchestrator.infer_and_visualize = real_iv

    assert seen == ["Ground-truth trace"], seen


# ── Simulate section (real-time streaming) ───────────────────────────────────────────────────────
def test_simulate_frame_time_grid_preserves_dt_and_is_continuous():
    """The streaming loop advances one frame at a time; the frame grid must keep the fine EM step exactly
    dt_nd (so stability/timescale don't drift) and hand off continuously to the next frame."""
    from core.gui.panels.simulate_runner import frame_time_grid

    g = frame_time_grid(0.0, 100, 0.025)
    assert g.shape[0] == 101, "a frame of m steps needs m+1 points (dt = (t1-t0)/(n-1))"
    assert abs((g[1] - g[0]).item() - 0.025) < 1e-6
    assert abs((g[-1] - g[-2]).item() - 0.025) < 1e-6
    g2 = frame_time_grid(g[-1].item(), 40, 0.025)          # the next frame starts where this one ended
    assert abs(g2[0].item() - g[-1].item()) < 1e-6, "frames must be time-continuous"
    assert abs((g2[1] - g2[0]).item() - 0.025) < 1e-6


def test_simulate_gaussian_field_is_an_ellipse_perpendicular_to_the_motion():
    """The heatmap blob must peak at its center, stay in [0, 1], and be an ellipse whose MAJOR axis is
    perpendicular to the (horizontal) oscillation -- i.e. it decays faster along the motion axis than
    across it (sigma_par < sigma_perp)."""
    import numpy as np
    from core.gui.widgets.live_hair_bundle import gaussian_field

    gx = np.linspace(0, 1, 33)                             # 33 points -> 0.5 lands exactly on index 16
    gy = np.linspace(0, 1, 33)
    sig_par, sig_perp = 0.10, 0.20                         # along motion (x) < perpendicular (y)
    f = gaussian_field(0.5, 0.5, gx, gy, sig_par, sig_perp)
    assert f.shape == (33, 33)
    assert abs(float(f.max()) - 1.0) < 1e-5
    ix, iy = np.unravel_index(int(np.argmax(f)), f.shape)
    assert gx[ix] == 0.5 and gy[iy] == 0.5
    # moving the center along the motion axis must shift the blob
    ix2, _ = np.unravel_index(int(np.argmax(gaussian_field(0.75, 0.5, gx, gy, sig_par, sig_perp))), f.shape)
    assert gx[ix2] == 0.75

    # ellipse orientation: for a fixed offset the field is SMALLER along the motion axis than across it.
    d = 0.15
    along = gaussian_field(0.5, 0.5, np.array([0.5 + d]), np.array([0.5]), sig_par, sig_perp)[0, 0]
    across = gaussian_field(0.5, 0.5, np.array([0.5]), np.array([0.5 + d]), sig_par, sig_perp)[0, 0]
    assert along < across, "the ellipse major axis must be perpendicular to the oscillation"


def test_simulate_heatmap_center_stays_off_the_edge_when_oscillating():
    """The blob center must be mapped inside a horizontal margin, so an extreme displacement (x0_norm at
    0 or 1) does not clip at the field edge -- the on-display 'cut off when oscillating' bug."""
    from core.gui.widgets.live_hair_bundle import LiveHairBundleView

    _app()
    v = LiveHairBundleView()
    assert v._margin > 0.0
    assert v._cx(0.0) >= v._margin - 1e-9
    assert v._cx(1.0) <= v._aspect - v._margin + 1e-9
    assert v._aspect > 1.0, "the heatmap field must be a wide rectangle"


def test_simulate_plan_stream_matches_generate_observations_arithmetic():
    """plan_stream must reproduce the pipeline's subsample/steady/total arithmetic so a streamed trace
    matches the observation the SBI pipeline would build from the same cell."""
    from core.cli import resolve_bounds_for_cell
    from core.config import CELL_PATH
    from core.gui.panels.simulate_runner import build_stream_config, plan_stream

    _app()
    cdir = CELL_PATH / "nadrowski"
    # Resolve through the shared rule, not a same-named-sibling glob: the master cells deliberately
    # share ONE bounds file, so a sibling-only filter would silently skip most of them.
    cells = [c for c in sorted(cdir.glob("*.txt"))
             if resolve_bounds_for_cell(str(c)) is not None] if cdir.exists() else []
    if not cells:
        return                                             # environment without Resources: skip, don't fail

    cfg = build_stream_config("NADROWSKI", str(cells[0]))
    assert cfg.hw.device.type == "cpu", "streaming config must be forced onto CPU (device coherence)"
    plan = plan_stream(cfg, 0.05)

    t_scale = cfg.rescale_params["t_scale"][0]
    assert plan.subsample_factor == max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    assert plan.steady_steps == cfg.steady_idx
    assert plan.total_steps == plan.steady_steps + plan.n_obs * plan.subsample_factor
    assert plan.dt_nd == cfg.dt_nd_min
    assert plan.n_channels == 1 and plan.state_dep_drift is True
    # x_scale rides on cfg.hw.dtype (float32), matching how generate_observations builds rescale_gt --
    # so compare with a float32-scale tolerance, not float64-exact.
    x_scale_gt = cfg.rescale_params["x_scale"][0]
    assert abs(plan.x_scale - x_scale_gt) <= 1e-5 * abs(x_scale_gt)


def test_simulate_dispatch_streams_chunks_and_a_cancel_is_not_an_error():
    """dispatch(provide_stream=True) must inject the chunk emitter + stop flag, deliver frames to
    on_chunk, and a cancel of a streaming run must land as `cancelled` (not `error`)."""
    import numpy as np
    from core.gui.streams import WorkerCancelled

    app = _app()

    class P(BasePanel):
        pass

    panel = P()
    chunks = []
    started = {"v": False}
    outcome = {"cancelled": 0, "error": 0}

    def streamer(emit_chunk=None, should_stop=None):
        i = 0
        while True:
            if should_stop is not None and should_stop():
                raise WorkerCancelled()
            emit_chunk(np.array([[float(i), float(i)]], dtype=np.float64))
            started["v"] = True
            print(f"frame {i}")                            # a write() checkpoint + lets the pump tick
            time.sleep(0.01)
            i += 1

    panel.dispatch(streamer, provide_stream=True, on_chunk=chunks.append)
    for w in panel._workers:
        w.signals.cancelled.connect(lambda: outcome.__setitem__("cancelled", outcome["cancelled"] + 1))
        w.signals.error.connect(lambda *_a: outcome.__setitem__("error", outcome["error"] + 1))

    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and not (started["v"] and chunks):
        app.processEvents()
        time.sleep(0.005)
    assert chunks, "no streamed chunks were delivered to on_chunk"
    assert panel._busy and BasePanel._active_cancel is not None

    panel._request_cancel()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and panel._busy:
        app.processEvents()
        time.sleep(0.005)
    _pump(app, 0.3)

    assert not panel._busy, "panel stuck busy after cancelling a stream"
    assert outcome["cancelled"] == 1 and outcome["error"] == 0, outcome
    assert "Run cancelled." in panel.log_pane.toPlainText()


def test_simulate_panel_is_wired_and_navigable():
    """The 4th home button is live: the Simulate section is registered, navigable, and its panel is in
    the persistence sweep."""
    from core.gui.main_window import MainWindow
    from core.gui.panels.simulate_panel import SimulatePanel

    _app()
    w = MainWindow()
    assert w.panel(SimulatePanel) is not None
    assert "Simulate" in w._section_index
    w.nav.go_to(w._section_index["Simulate"])
    assert w.nav.stack.currentIndex() == w._section_index["Simulate"]
    assert any(isinstance(p, SimulatePanel) for p in w._all_panels())


def test_simulate_settings_round_trip():
    from core.gui import settings as st
    from core.gui.panels.simulate_panel import SimulatePanel

    _app()
    _temp_settings()
    try:
        sp = SimulatePanel()
        sp.tobs.setText("2.5")
        sp.fps.setText("24")
        sp.frame_steps.setText("1234")
        if sp.cell_picker.combo.count():
            sp.cell_picker.combo.setCurrentIndex(sp.cell_picker.combo.count() - 1)
        want_cell = sp.cell_picker.key()
        want_model = sp.model_combo.currentText()

        qs = st.settings()
        sp.save_settings(qs)
        qs.sync()

        sp2 = SimulatePanel()
        assert sp2.tobs.value() == 2.5
        assert sp2.fps.value() == 24
        assert sp2.frame_steps.value() == 1234
        assert sp2.model_combo.currentText() == want_model
        assert sp2.cell_picker.key() == want_cell
    finally:
        st.use_ini_file(None)


# ── Simulate section: "Save video…" export ───────────────────────────────────────────────────────
def _tiny_series():
    import numpy as np
    t = np.arange(600) * 1e-3                                   # 0.6 s -> several video frames at 30 fps
    x = np.sin(2 * np.pi * 8 * t) * 1.5 + 0.2
    return np.column_stack((t, x))


def _export_kwargs():
    import numpy as np
    return dict(window_pts=2000, grid_x=np.linspace(0, 2.6, 60), grid_y=np.linspace(0, 1, 24),
                sigma_x=0.10, sigma_y=0.20, aspect=2.6, margin=0.35, video_fps=30)


def test_export_stride_maps_sample_rate_to_video_fps():
    from core.gui.panels.simulate_export import estimate_frame_count, export_stride
    assert export_stride(1000.0, 30.0) == 33
    assert export_stride(1000.0, 10.0) == 100
    assert export_stride(1000.0, 0.0) == 1                      # zero/neg fps guard -> stride 1
    assert estimate_frame_count(300, 100) == 3                  # range(99, 300, 100) -> 99,199,299


def test_export_animation_writes_a_readable_gif():
    """A real GIF round-trip: render a tiny series, then read it back with imageio."""
    import os
    import tempfile
    # import the app module (torch/pyqtgraph/matplotlib) BEFORE imageio -- the OMP-safe order.
    from core.gui.panels.simulate_export import export_animation
    path = os.path.join(tempfile.mkdtemp(), "anim.gif")
    export_animation(_tiny_series(), path, **_export_kwargs())
    assert os.path.getsize(path) > 0
    import imageio
    frames = imageio.mimread(path)
    assert len(frames) >= 2, "expected a multi-frame gif"
    assert frames[0].shape[0] % 2 == 0 and frames[0].shape[1] % 2 == 0, "exported frame dims must be even"


def test_export_animation_writes_a_readable_mp4_when_ffmpeg_is_available():
    import os
    import tempfile
    from core.gui.panels.simulate_export import export_animation, ffmpeg_available
    if not ffmpeg_available():
        return                                                 # skip, don't fail, on a bare-pip env
    path = os.path.join(tempfile.mkdtemp(), "anim.mp4")
    export_animation(_tiny_series(), path, **_export_kwargs())
    assert os.path.getsize(path) > 0
    import imageio
    r = imageio.get_reader(path)
    try:
        frame = r.get_next_data()
    finally:
        r.close()
    assert frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0, "H.264 needs even frame dims"


def test_export_animation_removes_the_partial_file_on_failure():
    """A cancel/error mid-export must not leave a half-written file (cleanup is in a finally)."""
    import os
    import tempfile
    import core.gui.panels.simulate_export as se
    path = os.path.join(tempfile.mkdtemp(), "bad.gif")

    real = se.gaussian_field
    calls = {"n": 0}

    def boom(*a, **k):                                          # 1st call = field0 (ok); 2nd = 1st frame -> raise
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("boom mid-loop")
        return real(*a, **k)

    se.gaussian_field = boom
    try:
        se.export_animation(_tiny_series(), path, **_export_kwargs())
    except RuntimeError:
        pass
    finally:
        se.gaussian_field = real
    assert not os.path.exists(path), "a failed export must not leave a partial file"


def test_simulate_panel_records_chunks_and_gates_the_save_button():
    import numpy as np
    from core.gui.panels.simulate_panel import SimulatePanel

    _app()
    p = SimulatePanel()
    assert not p.btn_save_video.isEnabled(), "save must be disabled before any recording"
    p._on_chunk(np.array([[0.0, 0.1], [1e-3, 0.2]]))
    p._on_chunk(np.array([[2e-3, 0.3]]))
    assert len(p._record) == 2
    p.refresh_local_gates()
    assert p.btn_save_video.isEnabled(), "save must enable once a recording exists"
    p._record = []
    p.refresh_local_gates()
    assert not p.btn_save_video.isEnabled(), "save must disable again when the recording is cleared"


# ── Labels + units (round 4) ─────────────────────────────────────────────────────────────────────
def test_labels_axis_and_rescale_render_latex_with_units():
    from core.Helpers import labels as L
    assert L.axis_label("x", "nm") == "$x$ (nm)"
    assert L.axis_label(r"\tilde\omega") == r"$\tilde\omega$ (ND)"          # unit=None -> ND
    assert "ms/ND" in L.rescale_axis_label("t_scale", time_unit="ms")
    assert "nm/ND" in L.rescale_axis_label("x_scale", length_unit="nm")
    assert "pN/ND" in L.rescale_axis_label("f_scale", force_unit="pN")
    assert L.rescale_axis_label("x_offset", length_unit="nm") == r"$x_{\mathrm{off}}$ (nm)"
    assert L.rescale_axis_label("t_scale") == r"$t_{\mathrm{scale}}$"       # missing token -> bare symbol


def test_labels_pretty_gui_and_forcing():
    from core.Helpers import labels as L
    assert L.pretty_gui("F0 (ND forcing amplitude)") == "F<sub>0</sub> (ND forcing amplitude)"
    assert "<sub>obs</sub>" in L.pretty_gui("T_obs (s)")
    assert "<sub>a</sub>/T" in L.pretty_gui("T_a/T grid  (S = 0)")
    assert L.pretty_gui("Model") == "Model"                                # passthrough for non-math
    assert L.gui_forcing_label("phase", "rad") == "φ (rad)"
    assert L.gui_forcing_label("amp") == "A"


def test_simconfig_units_and_inferred_labels_are_latex():
    from pathlib import Path
    from core import cli
    from core.config import BOUNDS_PATH, CELL_PATH, VALID_LABELS, VALID_MODELS

    cell = CELL_PATH / "nadrowski" / "master_weak.txt"
    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    if not (cell.exists() and bounds.exists()):
        return                                                             # environment without Resources: skip
    cfg = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")], True, str(bounds))
    cli.load_and_validate_gt(cfg, str(cell))
    # freq is declared kHz, NOT Hz: drive frequency is consumed as INVERSE CELL TIME (1/ms = kHz), so a
    # declared "Hz" against an `ms` cell is a 1000x lie. See SimConfig.freq_si_to_cell.
    assert (cfg.length_unit, cfg.time_unit, cfg.force_unit, cfg.freq_unit) == ("nm", "ms", "pN", "kHz")
    assert cfg.check_unit_consistency() == [], "nadrowski units must be self-consistent"
    assert abs(30.0 * cfg.freq_si_to_cell - 0.03) < 1e-12, "30 Hz must be 0.03 cycles/ms in an ms cell"
    labels = cfg.inferred_labels
    assert all(l.startswith("$") for l in labels), labels
    assert any("nm/ND" in l for l in labels), "x_scale should carry nm/ND"

    bp_bounds = BOUNDS_PATH / "bp" / "cell_1.txt"
    if bp_bounds.exists():                                                 # BP declares no force/freq unit
        bp = cli.make_sim_config("BP", VALID_LABELS[VALID_MODELS.index("BP")], False, str(bp_bounds))
        assert bp.force_unit is None and bp.length_unit == "nm"


def test_plot_posterior_vs_truth_default_labels_have_units():
    import numpy as np
    from core.Helpers import visualizers
    fig = visualizers.plot_posterior_vs_truth(np.arange(5) * 1.0, np.zeros(5))
    ax = fig.axes[0]
    assert ax.get_xlabel() == "$t$ (s)" and ax.get_ylabel() == "$x$ (nm)"
    fig2 = visualizers.plot_posterior_vs_truth(np.arange(5) * 1.0, np.zeros(5),
                                               xlabel="$t$ (ms)", ylabel="$x$ (µm)")
    assert fig2.axes[0].get_xlabel() == "$t$ (ms)"


def test_gui_form_labels_are_prettified():
    from PySide6.QtWidgets import QLabel
    from core.gui.widgets.help_badge import help_label

    _app()
    holder = help_label("F0 (ND forcing amplitude)", "help text")
    lbl = holder.findChild(QLabel)
    assert lbl is not None and "F<sub>0</sub>" in lbl.text()
    # panels still build with the prettify hook in place
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.simulate_panel import SimulatePanel
    FdtPanel(); CrossValPanel(); SimulatePanel()


# ── dark-theme matplotlib figures (B-c) ──────────────────────────────────────────────────────────
import contextlib                                                  # noqa: E402


@contextlib.contextmanager
def _rcparams_guard():
    """Snapshot + restore matplotlib rcParams (process-global) so a theming test never bleeds its dark
    colours into the plot-producing tests that expect matplotlib's white defaults."""
    import matplotlib
    saved = dict(matplotlib.rcParams)
    try:
        yield
    finally:
        matplotlib.rcParams.update(saved)


def _fake_appearance(dark):
    """A minimal Appearance stand-in (is_dark + a real theme_changed Signal) so the theming test never
    touches theming._ACTIVE / the app styleHints wiring a real Appearance installs."""
    from PySide6.QtCore import QObject, Signal

    class _FakeAppearance(QObject):
        theme_changed = Signal(bool)

        def __init__(self, d):
            super().__init__()
            self._dark = d

        def is_dark(self):
            return self._dark

        def flip(self, d):
            self._dark = d
            self.theme_changed.emit(d)

    return _FakeAppearance(dark)


def test_a_theme_flip_between_build_and_save_cannot_split_a_figure():
    """THE 2026-08-25 RENDERING FAILURE, pinned, with its own counterfactual.

    matplotlib bakes artist colours at BUILD time but reads `savefig.facecolor` at SAVE time, and
    mpl_theme rewrites rcParams GLOBALLY on every appearance change. A multi-day run straddling one
    flip therefore saved dark axes onto a light page with `text.color` still light -- every tick
    label, axis label, title and table cell at ~1.06:1 contrast, i.e. invisible. 14 of 15 figures.
    """
    import io as _io
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from core.gui import mpl_theme
    from core.Helpers import visualizers

    _app()
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme(True)                 # build under DARK
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("x")
        try:
            mpl_theme.apply_mpl_theme(False)            # ...then the theme flips to LIGHT

            def corner_luminance(save):
                buf = _io.BytesIO()
                save(buf)
                buf.seek(0)
                return float(np.array(Image.open(buf).convert("RGB")).astype(int)[:6, :6].mean())

            pinned = corner_luminance(
                lambda b: visualizers.save_figure(fig, b, format="png", dpi=50))
            unpinned = corner_luminance(
                lambda b: fig.savefig(b, format="png", dpi=50))
        finally:
            plt.close(fig)

    assert pinned < 120, \
        f"save_figure put dark artists on a light page after a theme flip (corner {pinned:.0f})"
    assert unpinned > 200, \
        f"the counterfactual did not reproduce the bug (corner {unpinned:.0f}); this test proves nothing"


def test_the_parameter_table_is_readable_against_its_own_cells():
    """The best-fit parameter table came back completely blank. Transparent cells show the FIGURE
    background while the text is `text.color`, and those two only contrast while the theme is
    self-consistent -- so the 13 numbers you most want to read are the first thing to vanish."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgb
    from core.gui import mpl_theme
    from core.Helpers import visualizers

    def relative_luminance(c):
        r, g, b = (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in to_rgb(c))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    _app()
    for dark in (True, False):
        with _rcparams_guard():
            mpl_theme.apply_mpl_theme(dark)
            t = np.linspace(0, 1, 32)
            fig = visualizers.plot_best_fit_overlay(
                t, np.sin(t), np.sin(t) * 1.02,
                param_labels=[f"p{i}" for i in range(13)], param_values=list(range(13)))
            try:
                tables = [c for a in fig.axes for c in a.tables]
                assert tables, "the parameter table is gone"
                cells = list(tables[0].get_celld().values())
                for cell in cells:
                    bg = cell.get_facecolor()
                    assert bg[3] > 0.5, "a transparent cell: the text falls back to the FIGURE colour"
                    l1 = relative_luminance(bg[:3])
                    l2 = relative_luminance(cell.get_text().get_color())
                    ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
                    assert ratio >= 3.0, \
                        f"dark={dark}: table text at {ratio:.2f}:1 against its cell (was 1.06:1)"
            finally:
                plt.close(fig)


def test_apply_mpl_theme_sets_design_tokens():
    """apply_mpl_theme drives figure/axes/savefig facecolor + text.color from the DARK/LIGHT tokens."""
    import matplotlib
    from core.gui import design, mpl_theme
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme(True)
        d = design.tokens(True)
        assert matplotlib.rcParams["figure.facecolor"] == d["window"]
        assert matplotlib.rcParams["axes.facecolor"] == d["base"]
        assert matplotlib.rcParams["savefig.facecolor"] == d["window"]
        assert matplotlib.rcParams["text.color"] == d["text"]
        mpl_theme.apply_mpl_theme(False)
        assert matplotlib.rcParams["figure.facecolor"] == design.tokens(False)["window"]


def test_mpl_theme_install_none_is_a_noop():
    """No Appearance (the test path) -> rcParams untouched, so figures keep matplotlib's default white."""
    import matplotlib
    from core.gui import mpl_theme
    with _rcparams_guard():
        before = matplotlib.rcParams["figure.facecolor"]
        mpl_theme.install(None)
        assert matplotlib.rcParams["figure.facecolor"] == before


def test_mpl_theme_follows_the_appearance_signal():
    """install(appearance) applies now AND subscribes: flipping the theme re-applies the rcParams."""
    import matplotlib
    from core.gui import design, mpl_theme
    _app()
    with _rcparams_guard():
        ap = _fake_appearance(True)
        mpl_theme.install(ap)
        assert matplotlib.rcParams["figure.facecolor"] == design.tokens(True)["window"]
        ap.flip(False)
        assert matplotlib.rcParams["figure.facecolor"] == design.tokens(False)["window"]


def test_plot_ppc_summary_box_follows_the_dark_theme():
    """The two PPC summary boxes read plt.rcParams, so under a dark theme they are NOT the old hardcoded
    white -- locks the B-c hardcoded-white regression."""
    import numpy as np
    import matplotlib.pyplot as plt
    from core.gui import mpl_theme
    from core.Helpers.visualizers import plot_ppc
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme(True)
        ppc = {"z_scores": np.array([0.1, 0.5, 2.5, np.nan]),
               "coverage_90": 0.9, "num_outside": 1, "num_invalid": 1}
        fig = plot_ppc(ppc)
        try:
            fig.canvas.draw()
            boxes = [t.get_bbox_patch() for t in fig.axes[0].texts if t.get_bbox_patch() is not None]
            assert boxes, "expected summary boxes with a bbox patch"
            assert all(max(p.get_facecolor()[:3]) < 0.5 for p in boxes), \
                [p.get_facecolor() for p in boxes]
        finally:
            plt.close(fig)


def test_export_animation_background_stays_white_under_dark_theme():
    """The Simulate video export is insulated from the app's matplotlib theme: its background stays
    white even when the global rcParams are dark, so exported videos look the same in any theme."""
    import os
    import tempfile
    from core.gui import mpl_theme
    from core.gui.panels.simulate_export import export_animation
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme(True)                            # global dark theme active
        path = os.path.join(tempfile.mkdtemp(), "dark.gif")
        export_animation(_tiny_series(), path, **_export_kwargs())
        import imageio
        frame0 = imageio.mimread(path)[0]
        corner = frame0[0, 0][:3]                                  # top-left = figure background margin
        assert min(int(c) for c in corner) > 230, corner          # near-white, not the dark theme bg


def test_panel_splitter_is_sized_and_not_collapsible():
    """Every panel opens with a usable controls column that cannot be dragged to nothing.

    BasePanel builds the app's only QSplitter and there are nine live instances. It used to be a
    LOCAL with no setSizes and no setChildrenCollapsible, so every launch started at the minimum and
    one slip past the left edge collapsed the controls to zero width, recoverable only by finding a
    5px handle at x=0.
    """
    app = _app()
    from core.gui.panels.base_panel import BasePanel

    class P(BasePanel):
        pass

    panel = P()
    panel.resize(1300, 820)
    _pump(app, 0.15)
    assert not panel.splitter.childrenCollapsible(), \
        "the controls column can still be collapsed to zero width"
    assert all(s > 0 for s in panel.splitter.sizes()), \
        f"splitter opened with a zero-width pane: {panel.splitter.sizes()}"
    # The old hard 460px cap could not be escaped by widening the window, so any wider form got a
    # permanent horizontal scrollbar in the left column.
    assert panel.controls_scroll.maximumWidth() > 1000, \
        f"controls column is still hard-capped at {panel.controls_scroll.maximumWidth()}px"


def test_panel_layout_round_trips_through_settings():
    """The splitter position must survive a restart -- 'I have to re-drag it every launch' is the
    complaint. Uses saveState/restoreState, which work before the widget is shown or polished."""
    app = _app()
    from core.gui import settings as st
    from core.gui.panels.base_panel import BasePanel

    class P(BasePanel):
        pass

    path = _temp_settings()
    try:
        a = P()
        a.resize(1300, 820)
        _pump(app, 0.15)
        a.splitter.setSizes([500, 700])
        _pump(app, 0.05)
        want = a.splitter.sizes()
        qs = st.settings()
        a.save_layout(qs)
        qs.sync()

        b = P()                                   # restore_layout runs in __init__
        b.resize(1300, 820)
        _pump(app, 0.15)
        assert b.splitter.sizes() == want, \
            f"layout did not round-trip: saved {want}, restored {b.splitter.sizes()}"
    finally:
        st.use_ini_file(None)
        os.unlink(path)


def test_forms_grow_their_fields_and_numeric_boxes_have_a_floor():
    """Pins both halves of the 'input boxes are cut off' fix, across every form in every panel.

    Qt's Windows default is FieldsStayAtSizeHint: the field takes its size hint and stops, so numeric
    boxes rendered 3-6 characters wide and typing "0.033333" scrolled inside the box. Repo-wide there
    were ZERO setFieldGrowthPolicy calls over 18 QFormLayout sites. Asserting over discovered forms
    (not a fixed list) means a newly added form is covered too.
    """
    from PySide6.QtWidgets import QFormLayout, QLineEdit
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.widgets.labeled_inputs import FloatField, IntField

    _app()
    path = _temp_settings()
    try:
        panels = [CrossValPanel(), FdtPanel(), ReductionPanel()]
        forms = [f for p in panels for f in p.findChildren(QFormLayout)]
        assert forms, "no QFormLayouts discovered — the test is not exercising anything"
        bad = [f for f in forms
               if f.fieldGrowthPolicy() != QFormLayout.AllNonFixedFieldsGrow]
        assert not bad, f"{len(bad)}/{len(forms)} forms do not grow their fields"

        narrow = [w for p in panels for w in p.findChildren(QLineEdit)
                  if isinstance(w, (FloatField, IntField)) and w.minimumWidth() < 80]
        assert not narrow, \
            f"{len(narrow)} numeric field(s) have no usable minimum width (e.g. " \
            f"{narrow[0].minimumWidth()}px)"
    finally:
        st_mod = __import__("core.gui.settings", fromlist=["settings"])
        st_mod.use_ini_file(None)
        os.unlink(path)


def test_long_diagnostics_are_readable_without_horizontal_scrolling():
    """Panels emit long single-line diagnostics; the log pane must wrap them, and the crossval cell
    label (which receives an unbounded str(e)) must not force the whole column wider."""
    from PySide6.QtWidgets import QPlainTextEdit
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.widgets.log_pane import LogPane

    _app()
    assert LogPane().lineWrapMode() == QPlainTextEdit.WidgetWidth, \
        "log pane still truncates long lines instead of wrapping them"
    path = _temp_settings()
    try:
        assert CrossValPanel().cell_values.wordWrap(), \
            "the crossval cell-values label does not wrap, so a long error widens the whole column"
    finally:
        st_mod = __import__("core.gui.settings", fromlist=["settings"])
        st_mod.use_ini_file(None)
        os.unlink(path)


def _strip_docstrings(tree):
    """Remove every docstring from a parsed tree, in place, and return it.

    ⚠ ``ast.unparse`` DROPS COMMENTS BUT KEEPS DOCSTRINGS -- they are real string expressions in the
    AST, not trivia. An earlier version of _unparsed claimed otherwise, and the claim went unnoticed
    because the checks that used it happened to forbid strings that appeared only in comments. It
    stopped being harmless the moment a check forbade `mem_get_info` in a function whose DOCSTRING
    explains why it does not use mem_get_info: the assertion matched the prose, exactly the
    false positive the parse was supposed to prevent (the same shape as the _local_map lesson).
    """
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and isinstance(body, list) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.fix_missing_locations(tree)


def _code_only(obj) -> str:
    """Executable source of ``obj`` -- comments and docstrings both gone. See _strip_docstrings."""
    return ast.unparse(_strip_docstrings(ast.parse(textwrap.dedent(inspect.getsource(obj)))))


def test_the_vram_ceiling_is_on_config_live_and_NOT_persisted():
    """The VRAM ceiling is a HARDWARE knob, and it is the one field on the Config tab that is
    deliberately forgotten between sessions.

    NOT PERSISTED, because stale QSettings have already cost this project a ~5-day run: the
    2026-08-19 retrain trained on the RETIRED chi band because a saved value silently won over
    config.py. A ceiling fails the same way but far more quietly -- a forgotten 2 GiB does not
    error, it just makes every future run split from batch 0 and take several times longer, with
    nothing in the log to explain it. Starting each session at config.py's 0.0 keeps the throttle a
    decision somebody just made.

    A PLAIN ASSIGNMENT IS ENOUGH, unlike every other knob the GUI exposes. The sweep and flow fields
    had to become ARGUMENTS threaded into build_prior/build_posterior, because orchestrator does
    `from .config import ...` and binds them at import, so writing to the constant is a silent no-op
    (trap X12). pipeline._vram_ceiling_gib() does a getattr on the module every time the planner
    asks, so this one genuinely takes effect -- and this test would catch it if that ever changed.
    """
    from core.gui.panels import inference_tabs as it
    from core import config as _cfg
    from core.SBI import pipeline as _pipe

    saved = _cfg.SIM_VRAM_CEILING_GIB
    saved_env = os.environ.pop(_pipe.VRAM_CEILING_ENV, None)
    try:
        panel = it.ConfigPanel(None)
        assert hasattr(panel, "vram_ceiling"), "the Config tab must carry the VRAM ceiling field"
        assert panel.vram_ceiling.value() == 0.0, "it must default to OFF"

        # Live: typing must reach the planner with no plumbing in between.
        panel.vram_ceiling.setText("6.5")
        assert _cfg.SIM_VRAM_CEILING_GIB == 6.5, "the field must assign the config constant"
        assert _pipe._vram_ceiling_gib() == 6.5, "and the planner must read it live"
        # 2 GiB, not 6.5: the budget is a min() and the CPU branch of memory_budget_elements caps at
        # 4 GiB, so a ceiling above that would not bind and the assertion would prove nothing.
        panel.vram_ceiling.setText("2")
        dev, dt = torch.device("cpu"), torch.float32
        saved_cap = _pipe._BUDGET_CAP_ELEMENTS
        try:
            _pipe._BUDGET_CAP_ELEMENTS = None       # the other term in the same min()
            assert _pipe.sim_memory_budget_elements(dev, dt) == (2 * 2 ** 30) // 4, \
                "the ceiling must bind the budget the planner actually uses"
        finally:
            _pipe._BUDGET_CAP_ELEMENTS = saved_cap

        # Not persisted: neither direction may mention it.
        src_save = _code_only(it.ConfigPanel.save_settings)
        src_restore = _code_only(it.ConfigPanel.restore_settings)
        for what, src in (("save_settings", src_save), ("restore_settings", src_restore)):
            assert "vram" not in src.lower(), (
                f"{what} must NOT touch the VRAM ceiling -- a stale ceiling throttles every future "
                f"run silently, which is the failure mode the 2026-08-19 band trap already cost us")

        # And the env override must ANNOUNCE that it wins, rather than leaving a dead field.
        os.environ[_pipe.VRAM_CEILING_ENV] = "2.0"
        panel.vram_ceiling.setText("9")
        assert _pipe._vram_ceiling_gib() == 2.0, "the env override wins"
        assert _pipe.VRAM_CEILING_ENV in panel.vram_note.text(), (
            "a field that silently does nothing is worse than no field -- the note must say the "
            "environment is overriding it")
    finally:
        os.environ.pop(_pipe.VRAM_CEILING_ENV, None)
        if saved_env is not None:
            os.environ[_pipe.VRAM_CEILING_ENV] = saved_env
        _cfg.SIM_VRAM_CEILING_GIB = saved


def test_the_free_vram_readout_does_not_use_mem_get_info():
    """⚠ The readout beside the ceiling must come from nvidia-smi, never torch.cuda.mem_get_info.

    That reading overstates free VRAM on Windows by roughly the size of the desktop -- measured
    15037 MiB against nvidia-smi's 5814 at the same instant (trap X6) -- and it is the number that
    green-lit the batch which killed the first chi retrain. Printing it next to a field whose entire
    purpose is to bound VRAM would hand the user the exact lie the field defends against."""
    from core.gui.panels import inference_tabs as it
    src = _code_only(it._nvidia_smi_free_gib)
    assert "mem_get_info" not in src, "the readout must not use the optimistic driver reading"
    assert "nvidia-smi" in src, "the readout must come from nvidia-smi"
    got = it._nvidia_smi_free_gib()
    assert got is None or got >= 0.0, f"unexpected reading {got!r}"


def test_a_one_field_near_miss_blocks_a_fresh_run_until_confirmed():
    """A passive status line was not enough, three times over.

    `_budget_checkpoint` already said "these settings match no checkpoint, so this starts a NEW run"
    and named the differing field -- but it is a label on a tab the user has scrolled past by the
    time they press Train. It failed to prevent 884 batches being lost outright on 2026-08-27, and a
    3989-batch checkpoint being abandoned twice on 2026-08-28. So a run that would start from zero
    while a committed checkpoint sits ONE identity field away now has to be confirmed.

    Narrow on purpose: exactly one differing field is the signature of an accident. Two or more is
    usually a genuinely different experiment, and warning there would make this noise."""
    from core.SBI import training_checkpoint as tc

    base = {"format": "training-rows", "model": "NADROWSKI", "n_runs": 10000,
            "run_size": 2048, "prior_fingerprint": "aaaa", "chi_mode": True}
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        # A committed sibling that differs in exactly ONE field.
        sib = dict(base, prior_fingerprint="bbbb")
        d = tc.resolve_dir(sib, root); (d / "shards").mkdir(parents=True)
        torch.save({"identity": sib}, d / "header.pt")
        torch.save({"batches_done": 3989, "complete": False, "rng": None}, d / "state.pt")

        near = tc.near_miss_siblings(base, root)
        assert len(near) == 1, f"expected the one-field sibling, got {near}"
        assert near[0]["field"] == "prior_fingerprint" and near[0]["batches"] == 3989, near[0]

        # TWO differing fields is a different experiment -- it must NOT warn.
        far = dict(base, prior_fingerprint="cccc", n_runs=5000)
        d2 = tc.resolve_dir(far, root); (d2 / "shards").mkdir(parents=True)
        torch.save({"identity": far}, d2 / "header.pt")
        torch.save({"batches_done": 500, "complete": False, "rng": None}, d2 / "state.pt")
        names = {r["name"] for r in tc.near_miss_siblings(base, root)}
        assert d2.name not in names, "a two-field difference must not be reported as a near miss"

        # An identity that MATCHES a checkpoint is a resume, so there is nothing to confirm.
        assert tc.near_miss_siblings(sib, root) == [] or             all(r["name"] != d.name for r in tc.near_miss_siblings(sib, root)),             "a checkpoint must never be a near miss of itself"


def test_the_confirmation_is_reached_and_can_refuse():
    """The dialog must actually gate the dispatch, and Cancel must mean cancel."""
    from core.gui.panels import inference_tabs as it

    src = _code_only(it.PosteriorPanel._build_posterior)
    assert "_confirm_fresh_run" in src, "the Train handler must consult the confirmation"
    i_confirm = src.find("_confirm_fresh_run")
    i_dispatch = src.find("self.dispatch(")
    assert i_confirm < i_dispatch, "the confirmation must gate the dispatch, not follow it"
    assert "return" in src[i_confirm:i_dispatch], "a refusal must return instead of training"

    # Fails open rather than blocking a run it cannot assess.
    body = _code_only(it.PosteriorPanel._confirm_fresh_run)
    assert body.count("return True") >= 3, (
        "the check must fail OPEN -- no prior, an unreadable identity, and no near miss must all "
        "proceed; a warning that can block a run is worse than no warning")


if __name__ == "__main__":
    _app()
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            # Exception, NOT AssertionError. A test that raises anything else -- a ValueError
            # from a stale str.index, a CUDA error from a hostile card -- used to abort the
            # ENTIRE run at that point, silently losing every test after it. That cost 26
            # tests twice on 2026-08-28. A crash is a failure of THAT test, not of the suite.
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}\n      {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
