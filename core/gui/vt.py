"""Route tqdm's terminal protocol into structured progress rows + clean log lines.

Qt-free on purpose: everything here is plain Python, so it can be unit-tested by replaying the
literal chunks tqdm writes (see the classifier tests in tests/test_gui_progress.py).

WHY THIS EXISTS
    The naive "split on \\r and \\n" reader that this replaces mistook tqdm's bar frames for log
    lines and appended one row per redraw. tqdm redraws a bar at pos>0 as THREE writes
    (tqdm/std.py:1493-1497 display(); :451-460 print_status; :1441-1443 moveto):

        write("\\n" * pos)                 # moveto(pos)
        write("\\r" + frame + padding)     # the bar itself
        write("\\x1b[A" * pos)             # moveto(-pos)     (colorama present => real ANSI)

    The third write has no terminator, so `frame + "\\x1b[A"` stranded in the reader's buffer and
    was flushed by the NEXT redraw's leading "\\n" -- through the newline branch, i.e. as a log line.

HOW WE CLASSIFY
    Every chunk tqdm writes is atomic and self-describing, so we classify per chunk, not by
    scanning for terminators. There is no cursor to track and therefore nothing to desync.

    The row index of a paint is NOT simply "newlines in the preceding chunk": moveto(+1) and
    close(leave=True)'s trailing fp_write('\\n') (std.py:1303) are byte-identical. The authority is
    the chunk that FOLLOWS the paint -- display() guards moveto(pos) and moveto(-pos) behind the
    same `if pos:`, so:

        a paint sits at row n  <=>  it is followed by a pure up-move of n     (else row 0)

    A paint preceded by no newline is therefore UNAMBIGUOUSLY row 0, and we publish it at once. A
    paint preceded by newlines is only a GUESS (real moveto, or a stale finalizer newline), so we
    hold it for exactly one chunk. That costs nothing when the guess is right: display() writes
    moveto / paint / moveto-back as three back-to-back writes, so the confirming up-move lands
    microseconds later. When the guess is wrong, the bar simply never appears on the wrong row.

INVARIANT -- DO NOT GIVE THE REDIRECTED STREAM A `fileno()` OR AN `encoding` ATTRIBUTE.
    Without them tqdm's screen-shape probe fails, so ncols is None (frames are never disp_trim'd,
    which `_PCT` relies on) and nrows is None (`pos < (nrows or 20)`, so every nested bar displays).
    Adding `encoding` also flips `ascii` (std.py:1038) and changes the frame glyphs.
"""
import re
from dataclasses import dataclass

from core.config import SOLVER_BAR_DESC

_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")      # any CSI escape
_UP = re.compile(r"\A(?:\x1b\[[0-9]*A)+\Z")          # a pure moveto(-n): motion, never text
_DOWN = re.compile(r"\A\n+\Z")                       # a pure moveto(+n) -- or a leave=True finalizer
_PCT = re.compile(r"(?P<pct>\d{1,3})%\|(?P<glyphs>[^|]*)\|\s?(?P<stats>.*)\Z")
_TOTAL = re.compile(r"\A\d+/(?P<total>\d+)")
_DIGITS = re.compile(r"\d+")

# tqdm renders its rate in exactly three forms (tqdm/std.py:550-559):
#   "13267.85it/s"  rate >= 1        (note the :5.2f -- values under 100 carry a leading space)
#   " 2.50s/it"     rate <  1        SECONDS PER ITERATION -- must be inverted, not read as 2.5 it/s
#   "?it/s"         rate unknown     the opening frame, and any bar shorter than its mininterval
_RATE = re.compile(r"(?P<n>\d+(?:\.\d+)?)(?P<unit>it/s|s/it)")

# The solver bar is identified by its DESC, never by its row: its tqdm `pos` is 0, 1 or 2 depending on
# the phase and the panel (during a posterior build it sits under two other bars; during FDT Campaign 1
# it is the only bar there is). This prefix is unique across the repo -- core/Solvers/sdeint.py is its
# sole producer.
#
# WHAT THIS IS FOR HAS CHANGED. It used to identify the bar whose rendered `it/s` was scraped for the
# GUI's Solver Performance meter; that number now comes from core.progress.SOLVER, because a bar
# shorter than its own mininterval never paints a rate (see progress_pane's module docstring). The
# prefix survives purely to EXCLUDE this bar from the overall-bar election -- trap S3 -- and that use
# is if anything more load-bearing now: the election is by largest total, and the solver's is in the
# tens of thousands.
_SOLVER_PREFIX = f"{SOLVER_BAR_DESC} (batch="


@dataclass(frozen=True, slots=True)
class RowState:
    """One live progress row. Frozen: instances cross the worker->GUI thread boundary and stay
    referenced by the pump on the writer side afterwards."""
    key: tuple          # (stream_name, row) -- stdout row 0 and stderr row 0 are different rows
    row: int            # == tqdm's `pos`
    desc: str           # the bar's label
    ident: str          # digit-normalised desc: the "is this still the same bar?" identity
    pct: int | None     # None => indeterminate (total=None, or a bare status line)
    total: int | None   # the bar's total; 1 means the bar is degenerate and carries no information
    rate: float | None  # ITERATIONS PER SECOND (already inverted if tqdm reported s/it); None if unknown
                        # -- part of a faithful frame parse; NOT the Solver Performance meter's source
    stats: str          # "1234/5000 [00:12<01:03, 59.4it/s]"
    raw: str            # the whole frame, ANSI-stripped

    @property
    def informative(self) -> bool:
        """Can this row drive an overall percentage? A total=1 bar goes 0% -> 100% with nothing in
        between -- e.g. pipeline.train_nn's outer bar wraps range(TRAINING_NUM_ROUNDS) and that is 1
        (config.TRAINING_NUM_ROUNDS), so it reads 0% for the whole multi-hour posterior build."""
        return self.pct is not None and (self.total or 0) > 1

    @property
    def is_solver(self) -> bool:
        """The SDE solver's per-step bar, which the pane must never let drive the overall bar.

        It used to also mean "this is the row the Solver Performance meter reads". It no longer does:
        the meter differences core.progress.SOLVER instead, because scraping a rendered bar cannot
        work for a call shorter than that bar's mininterval. This predicate is now purely an
        exclusion (trap S3).
        """
        return self.desc.startswith(_SOLVER_PREFIX)


def normalise(desc: str) -> str:
    """Identity key for a bar, immune to a mutating counter in its description.

    core/SBI/Priors/{bp,hopf,nadrowski}_prior.py call set_description() with a running count on
    EVERY iteration, so the literal desc is not a stable identity -- keying rows off it would mint
    a new row per iteration, which is the very bug this module exists to kill.
    """
    return _DIGITS.sub("#", desc).strip()[:80]


def parse_rate(stats: str) -> float | None:
    """Iterations per second from a tqdm stats field, or None if the bar has not measured one yet.

    tqdm flips to `s/it` below 1 it/s (tqdm/std.py:557), so " 2.50s/it" means 0.4 it/s -- NOT 2.5. Read
    naively, a crawling bar would report as a fast one.

    KEPT ON PURPOSE even though the Solver Performance meter stopped reading it: parse_bar's contract
    is a FAITHFUL parse of a tqdm frame, this is one regex against a string already in hand, and the
    s/it inversion is knowledge that cost a bug to learn (trap S4).
    """
    m = _RATE.search(stats)
    if not m:
        return None                                  # "?it/s": no measurement yet
    value = float(m["n"])
    if m["unit"] == "s/it":
        return 1.0 / value if value else None
    return value


def parse_bar(key: tuple, row: int, raw: str) -> RowState:
    """Turn one tqdm frame into a RowState. A frame with no `NN%|...|` is indeterminate.

    strip(), not rstrip(): a status line built by `print("\\r", text, end="")` (sbi's epoch counter,
    base.py:1024) arrives with print's separator as a leading space.
    """
    text = _CSI.sub("", raw).strip()
    m = _PCT.search(text)
    if not m:
        return RowState(key, row, text, normalise(text), None, None, None, "", text)
    desc = text[:m.start()].rstrip().removesuffix(":").rstrip()
    stats = m["stats"].strip()
    t = _TOTAL.match(stats)
    return RowState(key, row, desc, normalise(desc), min(100, int(m["pct"])),
                    int(t["total"]) if t else None, parse_rate(stats), stats, text)


class StreamRouter:
    """Consumes one redirected stream's write() chunks and reports structured events to `sink`.

    `sink(kind, payload)` is called synchronously on the WRITER thread and must do no Qt work:
        ("row",    RowState)        upsert a progress row
        ("retire", key)             remove a progress row
        ("log",    (text, level))   a completed log line
    """

    def __init__(self, name: str, sink, level: str = "info"):
        self.name = name                 # "out" | "err"
        self.sink = sink
        self.level = level
        self.rows: dict[int, RowState] = {}
        self._pending_down = 0           # newlines in the immediately preceding chunk
        self._held: tuple[int, str] | None = None   # a row>0 paint awaiting its confirming up-move
        self._closing: int | None = None  # a row a bare '\n' MAY have just finalised (leave=True)
        self._over: str | None = None    # overwrite buffer, opened by a bare '\r' (sbi's epoch line)
        self._over_row = 0
        self._line = ""                  # ordinary-text buffer, split into log lines on '\n'

    # ── event helpers ────────────────────────────────────────────────────────
    def _key(self, row: int) -> tuple:
        return (self.name, row)

    def _retire(self, row: int) -> None:
        if self.rows.pop(row, None) is not None:
            self.sink("retire", self._key(row))

    def _log(self, text: str) -> None:
        # strip(), not rstrip(): a status line graduating into the log carries print()'s separator as
        # a leading space (`print("\\r", text, end="")`).
        clean = _CSI.sub("", text).strip()
        if clean:
            self.sink("log", (clean, self.level))

    def _install(self, row: int, raw: str) -> None:
        state = parse_bar(self._key(row), row, raw)
        old = self.rows.get(row)
        if old is not None and old.ident != state.ident:
            # A different bar took this slot. Persist the old bar's final frame to the log (this is
            # what a leave=True bar does on a real terminal) and kill whatever was nested under it.
            self._log(old.raw)
            for deeper in [r for r in self.rows if r > row]:
                self._retire(deeper)
        self.rows[row] = state
        self.sink("row", state)

    def _confirm(self, up_n: int) -> None:
        """Settle a held paint. `up_n` is the depth an up-move just proved, or 0 when the chunk that
        followed the paint was not an up-move -- which means no moveto ever happened and the paint
        belongs on row 0 (the newlines that preceded it were a leave=True bar's finalizer)."""
        held, self._held = self._held, None
        if held is not None:
            self._install(up_n, held[1])

    def _settle_closing(self) -> None:
        """No moveto materialised after that bare '\\n', so it really was close(leave=True) finalising
        row 0 (tqdm/std.py:1302-1303). Persist the bar's final frame to the log -- which is what a
        terminal shows -- and retire the row.

        Without this a finished leave=True bar (Priors.prior "Constructing latent
        prior...", FDT.campaigns "Campaign 2") would sit in the progress pane at 100% for
        the rest of the run AND, being `informative`, would peg the overall bar at 100% -- reading as
        "done" or "hung" while the pipeline is still working.
        """
        row, self._closing = self._closing, None
        if row is not None and row in self.rows:
            self._log(self.rows[row].raw)
            self._retire(row)

    # ── overwrite mode (a bare '\r' followed by ordinary text) ───────────────
    def _publish_over(self) -> None:
        if self._over and self._over.strip():
            self._install(self._over_row, self._over)

    def _finalise_over(self) -> None:
        """The status line is DONE (a newline arrived, or we are tearing down): graduate it into the
        log and drop its row.

        Not to be confused with a status line being OVERWRITTEN by the next bare '\\r', which just
        drops the text buffer and leaves the row to be re-published. Conflating the two logs one line
        per update -- exactly what sbi's per-epoch counter would do (base.py:1024 reprints itself
        every epoch with a leading '\\r').
        """
        if self._over is None:
            return
        text, self._over = self._over, None
        self._retire(self._over_row)
        self._log(text)

    # ── ordinary text ────────────────────────────────────────────────────────
    def _text(self, chunk: str) -> None:
        if not chunk:
            return
        self._line += chunk
        while "\n" in self._line:
            line, _, self._line = self._line.partition("\n")
            self._log(line)

    # ── the state machine ────────────────────────────────────────────────────
    def feed(self, chunk: str) -> None:
        if not chunk:
            return

        if _UP.match(chunk):                       # moveto(-n): pure motion, never text
            self._confirm(chunk.count("\x1b["))
            self._closing = None                   # a real moveto happened, so row 0 is alive
            self._pending_down = 0
            return

        if _DOWN.match(chunk):
            # A bare newline is THREE different things depending on what is pending:
            #   * the terminator of a status line   (sbi's epoch counter, then its summary print)
            #   * the terminator of a plain print() (by far the most common -- print writes its text
            #     and its '\n' as two separate chunks, and the '\n' arrives here alone)
            #   * a real moveto(+n), or a leave=True bar's finalizer (tqdm/std.py:1303)
            # Only the last is cursor motion. Mistaking a print()'s terminator for a moveto strands
            # the line forever and shifts the next bar down a row.
            self._confirm(0)
            if self._over is not None:
                self._finalise_over()
                self._pending_down = 0
            elif self._line:
                self._text(chunk)
                self._pending_down = 0
            else:
                # moveto(+n) and close(leave=True)'s finalizer are byte-identical here. Assume moveto
                # (so a nested bar lands on the right row), but remember that row 0 may have just died
                # -- _settle_closing() retires it once the next chunk proves no moveto was coming.
                self._pending_down = len(chunk)
                if 0 in self.rows:
                    self._closing = 0
            return

        self._confirm(0)                           # any other chunk: no up-move came, so row 0

        if chunk.startswith("\r"):
            body = chunk[chunk.rindex("\r") + 1:]
            row, self._pending_down = self._pending_down, 0
            # An in-place overwrite, NOT a finalisation: drop the text buffer but leave the row alone
            # (a bare '\r' re-publishes it; a bar frame replaces it via _install).
            self._over = None
            if not body:
                # A bare '\r' is ALWAYS row 0: tqdm never precedes one with a moveto (close() emits it
                # only `if not pos`, std.py:1307), so any pending_down here is stale -- e.g. from a
                # blank print() -- and honouring it would push the status line onto a phantom row.
                self._settle_closing()
                self._over, self._over_row = "", 0
            elif not _CSI.sub("", body).strip():   # a blank paint IS close(leave=False) (std.py:1306)
                self._retire(row)                  # always preceded by a real moveto, so row is exact
            elif row == 0:
                self._closing = None               # _install replaces row 0 (and logs the old frame)
                self._install(0, body)             # no preceding newline => unambiguously pos 0
            else:
                self._held = (row, body)           # a guess: hold for the confirming up-move
            return

        self._pending_down = 0
        if self._over is not None:                 # ordinary text inside overwrite mode
            self._over += chunk
            if "\n" in self._over:
                head, _, tail = self._over.partition("\n")
                self._over = head
                self._finalise_over()              # the status line graduates into the log
                self._text(tail)
            else:
                self._publish_over()               # eager: the row is current, never a frame stale
            return

        self._settle_closing()                     # plain text, so no moveto was coming after all
        self._text(chunk)

    def close(self) -> None:
        """Teardown: settle everything, and persist any still-live bar's final frame to the log."""
        self._confirm(0)
        self._settle_closing()
        self._finalise_over()
        if self._line:
            self._log(self._line)
            self._line = ""
        for row in sorted(self.rows, reverse=True):
            self._log(self.rows[row].raw)
            self._retire(row)
