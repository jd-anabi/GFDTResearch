"""Pick up PNGs that a runner saved to disk and show them in the panel's FigureStack.

WHY NOT just take them from the runner's return value: the FDT / Reduction / CrossVal runners don't
hand their figures back. run_fdt returns None, run_reduction_map returns a ReductionRecord, and
run_param_study_cli returns the two HDF5 *data* paths -- the figure paths are only print()ed
(FDT.fdt_pipeline.run_fdt, Reduction.sweep, FDT.cross_validation.run_fdt_param_sweep).

WHY NOT scrape those prints: four modules, four different formats, and it would weld the GUI to
print() text inside core.

So instead we snapshot the plot directory when a run starts and pick up whatever appears. That needs
no core change at all, and -- the real payoff -- it shows figures INCREMENTALLY: the FDT sanity plot
lands before the production sweep starts, and the S-sweep plot lands at the study's midpoint (which
core/FDT/cross_validation.py saves early precisely so a long run shows you something). It also
still surfaces whatever a half-failed run managed to produce.

Runs entirely on the GUI thread (a QTimer). The worker is never involved.
"""
import re
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

_POLL_MS = 1200
_SETTLE_S = 1.0                                     # savefig may still be writing; let the file age
_STAMP = re.compile(r"_\d{8}_\d{6}$")               # the ..._20260714_120301 suffix every writer adds


def _title(name: str) -> str:
    """'fdt3d_vs_S_20260714_120301.png' -> 'fdt3d vs S'."""
    return _STAMP.sub("", Path(name).stem).replace("_", " ") or Path(name).stem


class NewPngWatcher(QObject):
    """Emits (title, absolute path) for each PNG that appears in `directory` after start()."""

    png_ready = Signal(str, str)

    def __init__(self, directory, parent=None):
        super().__init__(parent)
        self._dir = Path(directory)
        self._baseline: set[str] = set()
        self._emitted: set[str] = set()
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._scan)

    def start(self) -> None:
        self._baseline = self._pngs()               # whatever is already on disk is not ours
        self._emitted.clear()
        self._timer.start()

    def stop(self) -> None:
        """Stop polling, but scan once more first -- the run's last figure is usually written in the
        moment between the final poll and the worker finishing."""
        self._timer.stop()
        self._scan(final=True)

    # ── internals ────────────────────────────────────────────────────────────
    def _pngs(self) -> set[str]:
        if not self._dir.is_dir():
            return set()
        return {p.name for p in self._dir.glob("*.png")}

    def _scan(self, final: bool = False) -> None:
        import time

        for name in sorted(self._pngs() - self._baseline - self._emitted):
            path = self._dir / name
            try:
                if not final and time.time() - path.stat().st_mtime < _SETTLE_S:
                    continue                        # still being written; catch it next poll
            except OSError:
                continue
            self._emitted.add(name)
            self.png_ready.emit(_title(name), str(path))
