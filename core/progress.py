"""A front-end-agnostic step counter the SDE solver publishes, and any front-end may sample.

WHY THIS EXISTS -- and why it is a COUNTER rather than anything cleverer.
    The GUI's "Solver Performance" meter used to be obtained by PARSING THE RENDERED TEXT of the
    solver's tqdm bar (core/Solvers/sdeint.py -> core/gui/vt.py -> widgets/progress_pane.py). That
    coupling has one failure mode and it fired: tqdm only paints a frame carrying a rate once
    `mininterval` has elapsed, so a solver call SHORTER than its own bar's mininterval renders
    "?it/s" and nothing else, forever. When CUDA graphs took a 100k-step call from ~10 s to ~0.7 s
    (§8.3) the meter read "-- (idle)" for entire multi-day runs. A 10x speedup made a progress bar
    too fast to render.

    A monotonically increasing count of completed steps has no such failure mode. It is correct at
    any speed, including speeds that make tqdm paint nothing at all, and it will still be correct
    after the next speedup. Sampling policy -- smoothing, idle timeouts, units -- belongs to the
    reader, not here.

THREAD MODEL -- ONE WRITER, MANY READERS. DO NOT ADD A SECOND WRITER.
    `add()` is called only from the thread running the solver: the CLI's main thread, or the GUI's
    QThreadPool worker thread. Readers (the GUI thread's 100 ms tick) only ever read. `steps += n`
    is NOT atomic under the GIL -- it is LOAD_ATTR / BINARY_OP / STORE_ATTR -- but with a single
    writer no update can be lost, and a reader that catches a stale value simply attributes those
    steps to its next sample. A second writer would silently drop steps, and no test would see it.

NEVER RESET IT. Readers take DIFFERENCES between successive samples, so a reset would present as a
    large negative delta -- i.e. a negative rate -- at whatever moment it happened. `ProgressPane`
    snapshots the current value when a run begins instead, which is the same thing without the race.
    Python ints are arbitrary precision, so there is nothing to overflow.
"""


class SolverSteps:
    """Completed SDE integration steps, since process start. See the module docstring."""

    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps = 0

    def add(self, n: int = 1) -> None:
        """Publish `n` completed steps. Called from the solver's hot loop -- keep it this cheap."""
        self.steps += n


#: The process-wide solver counter. `core/Solvers/sdeint.py` writes it; front-ends read it.
SOLVER = SolverSteps()
