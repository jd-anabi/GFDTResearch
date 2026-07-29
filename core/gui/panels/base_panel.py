"""Shared panel scaffolding: a left controls column and a right results area (a figure stack over a
progress pane + log pane), plus ``dispatch()`` to run a callable on a background worker with its
output wired to those widgets."""
import weakref

from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtWidgets import (QHBoxLayout, QMessageBox, QPushButton, QScrollArea, QSplitter,
                               QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from ..design import CONTROLS_MIN_W, DEFAULT_RESULTS_SPLIT, DEFAULT_SPLIT
from ..plot_watcher import NewPngWatcher
from ..streams import CancelToken
from ..widgets.figure_stack import FigureStack
from ..widgets.log_pane import LogPane
from ..widgets.progress_pane import ProgressPane
from ..worker import Worker


def _png_fig_sink(figure_signal):
    """Return a ``(title, fig) -> None`` sink that renders a matplotlib Figure to PNG bytes ON THE
    WORKER THREAD and emits them. The UI thread then shows a QPixmap -- it never paints a live canvas
    created on the worker thread, which deadlocks on matplotlib's global lock. The figure is closed
    after rendering to free memory.

    It ALSO pickles the figure (best-effort) and ships the bytes alongside the PNG, so the panel can
    rebuild an interactive copy on the GUI thread for the "Pop out" button (FigureStack). Pickling is
    done here on the worker because the Figure is closed right after -- but pickling never renders
    (it does not touch Agg's renderer lock), so it is safe, unlike painting a live canvas. A pickle
    failure must never break the run, so it degrades to None and the pop-out falls back to the image
    viewer; the PNG thumbnail is emitted unconditionally either way.
    """
    def _sink(title, fig):
        import io
        import pickle
        import matplotlib.pyplot as plt
        try:
            fig_pickle = pickle.dumps(fig)      # before savefig -> the pristine figure the stage built
        except Exception:                       # noqa: BLE001 -- pop-out is optional; keep the run alive
            fig_pickle = None
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        finally:
            plt.close(fig)
        figure_signal.emit(title, buf.getvalue(), fig_pickle)
    return _sink


class BasePanel(QWidget):
    """Shared scaffolding for every run-a-thing panel: a controls column, a results column, and
    ``dispatch()`` to run work on a background thread with its output wired to both.

    Nine of these exist (Reduction, FDT, CrossVal, Simulate + the five inference tabs). Three class
    attributes carry app-wide state and are class-level ON PURPOSE -- see their comments: ``_running``
    (only one panel may run at a time, because stream redirection is process-wide), ``_active_cancel``
    and ``_instances``.

    Persists: nothing here. Subclasses override ``save_settings``/``restore_settings`` for their own
    selections; the splitter geometry goes through the separate ``save_layout``/``restore_layout``
    pair, because 8 of the 9 subclasses override save_settings without calling super().
    """
    # Class-level, deliberately: redirect_streams swaps sys.stdout/stderr PROCESS-WIDE (see
    # core/gui/streams.py), so only ONE panel may run at a time -- a per-panel guard would let the FDT
    # tab start a run while the SBI tab is training, and the two would fight over the console. The
    # _REDIRECT lock in streams.py is the backstop; this is the thing that keeps us away from it.
    _running = False
    # The one live run's cancel token (there is only ever one, per _running). MainWindow.closeEvent
    # reaches it through request_cancel_all() to stop a run before quitting.
    _active_cancel: "CancelToken | None" = None
    # Every live panel, so a run in ANY panel can lock the controls of ALL of them (see _set_busy).
    # A WeakSet so panels torn down in tests (or a future dynamic UI) drop out without bookkeeping.
    _instances: "weakref.WeakSet" = weakref.WeakSet()

    @classmethod
    def request_cancel_all(cls) -> None:
        """Ask the currently-running task (if any) to stop at its next checkpoint."""
        if cls._active_cancel is not None:
            cls._active_cancel.requested.set()

    def __init__(self, parent=None):
        super().__init__(parent)
        BasePanel._instances.add(self)
        self._busy = False
        self._cancel: "CancelToken | None" = None
        self._workers = set()   # keep workers alive until 'finished' (else Qt purges its queued signals)

        # Left: subclasses fill self.controls (inside a scroll area so long forms stay usable).
        self.controls = QWidget()
        self.controls_layout = QVBoxLayout(self.controls)
        self.controls_layout.setAlignment(Qt.AlignTop)
        # Kept on self so save_layout/restore_layout can reach them (both used to be locals, which is
        # why nothing could persist or even inspect the splitter).
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(self.controls)
        self.controls_scroll.setMinimumWidth(CONTROLS_MIN_W)
        # No maximum. The old hard 460px cap could not be escaped by widening the window, so ANY form
        # wider than that got a permanent horizontal scrollbar in the left column -- one of the two
        # halves of the "input boxes are cut off" report. Width is now governed by the splitter, which
        # the user can drag and which now remembers where they put it.

        # Right: figures over a progress pane over a log. Progress lives in its own widget -- one row
        # per live tqdm bar -- and never touches the log, which only ever appends completed lines.
        self.figure_stack = FigureStack()
        self.progress_pane = ProgressPane()
        self.log_pane = LogPane()

        # Cancel sits on the progress row, hidden until a run starts. It is a "please stop": it sets a
        # flag the pipeline's next print/redraw checks, so the run unwinds at its next checkpoint.
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._request_cancel)
        progress_row = QWidget()
        progress_layout = QHBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.addWidget(self.progress_pane, 1)
        progress_layout.addWidget(self.btn_cancel)

        # Stored as attributes so a subclass can insert its own primary view above the figure stack
        # (e.g. SimulatePanel mounts a live pyqtgraph view here and hides the static figure stack).
        # The results column is a VERTICAL SPLITTER, not a fixed 3:0:1 stretch. The log pane used to
        # hold ~25% of the height whether it had one line in it or a thousand, with no handle to drag
        # -- that is the other half of "panes have to be dragged to show everything": here there was
        # nothing to drag at all. The progress row stays outside the splitter (it is a fixed-height
        # strip and a handle around it would be noise).
        self.results_split = QSplitter(Qt.Vertical)
        self.results_split.addWidget(self.figure_stack)
        self.results_split.addWidget(self.log_pane)
        self.results_split.setStretchFactor(0, 3)
        self.results_split.setStretchFactor(1, 1)
        self.results_split.setChildrenCollapsible(False)
        self.results_split.setSizes(list(DEFAULT_RESULTS_SPLIT))

        self.right = QWidget()
        self.right_layout = QVBoxLayout(self.right)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.addWidget(self.results_split, 1)
        self.right_layout.addWidget(progress_row)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.controls_scroll)
        self.splitter.addWidget(self.right)
        self.splitter.setStretchFactor(1, 1)
        # A drag past the left edge used to collapse the controls column to zero width, recoverable
        # only by finding a 5px handle at x=0.
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes(list(DEFAULT_SPLIT))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self.splitter)

        # Persist the split as the user drags it, not only on a clean quit: save_settings runs from
        # MainWindow.closeEvent alone, so a crash -- or answering "don't quit" -- lost everything, and
        # "I have to re-drag it every launch" is the actual complaint.
        self._layout_save_timer = QTimer(self)
        self._layout_save_timer.setSingleShot(True)
        self._layout_save_timer.setInterval(1500)
        self._layout_save_timer.timeout.connect(self._persist_layout)
        self.splitter.splitterMoved.connect(lambda *_: self._layout_save_timer.start())
        self.results_split.splitterMoved.connect(lambda *_: self._layout_save_timer.start())

        self.restore_layout()

    # ── background dispatch ──────────────────────────────────────────────────
    def dispatch(self, fn, *args, provide_fig_sink: bool = False, provide_stream: bool = False,
                 on_chunk=None, watch_dir=None, on_result=None, on_finished=None, **kwargs):
        """Run ``fn`` on a worker thread. Its print()s and warnings stream to the log pane and its tqdm
        bars to the progress pane; figures (when ``provide_fig_sink``) embed in the figure stack; the
        return value goes to ``on_result``.

        ONE TASK AT A TIME APP-WIDE, not merely per panel -- redirect_streams swaps sys.stdout/stderr
        process-wide, so two concurrent runs would fight over the console (see GOTCHA #4).

        ``watch_dir`` is for the FDT / Reduction / CrossVal runners, which save their figures to disk
        instead of handing them back: any PNG appearing there during the run is picked up and shown
        (see core/gui/plot_watcher.py).
        """
        if BasePanel._running:
            where = "in this tab" if self._busy else "in another tab"
            self.log_pane.append_line(
                f"A task is already running ({where}); please wait for it to finish.", "warning")
            return

        watcher = None
        if watch_dir is not None:
            watcher = NewPngWatcher(watch_dir, self)
            watcher.png_ready.connect(self.figure_stack.add_png)
            watcher.start()

        self._cancel = CancelToken()
        BasePanel._active_cancel = self._cancel
        worker = Worker(fn, *args, cancel=self._cancel, **kwargs)
        # Retain the worker (and thus its WorkerSignals sender) until it reports finished, and stop the
        # thread pool from auto-deleting the C++ QRunnable underneath it -- otherwise the sender is
        # destroyed as soon as run() returns and Qt discards its still-queued result/finished events.
        worker.setAutoDelete(False)
        self._workers.add(worker)
        if provide_fig_sink:
            worker.kwargs["fig_sink"] = _png_fig_sink(worker.signals.figure)
        if provide_stream:
            # A long-lived streaming runner (e.g. SimulatePanel) emits numpy frames through the `chunk`
            # signal and polls should_stop() to unwind cooperatively -- the same injection trick as
            # provide_fig_sink, but for a continuous stream rather than one-shot figures. should_stop is
            # the cancel token's flag: the runner raises WorkerCancelled between frames when it flips.
            worker.kwargs["emit_chunk"] = worker.signals.chunk.emit
            worker.kwargs["should_stop"] = self._cancel.requested.is_set
            if on_chunk is not None:
                worker.signals.chunk.connect(on_chunk)
        worker.signals.log.connect(self.log_pane.append_line)
        worker.signals.log_batch.connect(self.log_pane.append_lines)
        worker.signals.log_batch.connect(lambda _b: self.progress_pane.heartbeat())
        worker.signals.rows.connect(self.progress_pane.set_rows)
        worker.signals.figure.connect(self.figure_stack.add_figure)
        worker.signals.error.connect(self._on_error)
        worker.signals.cancelled.connect(lambda: self.log_pane.append_line("Run cancelled.", "warning"))
        if on_result is not None:
            worker.signals.result.connect(on_result)

        def _finished():
            if watcher is not None:
                watcher.stop()          # one last scan: the final figure often lands right at the end
                watcher.deleteLater()
            self._set_busy(False)
            self._workers.discard(worker)
            # Release the PAYLOAD explicitly. discard() is not a lifetime bound: QThreadPool.start()
            # hands the QRunnable to C++, and setAutoDelete(False) (above) means C++ never frees it, so
            # the Worker shell outlives the run -- as does the _finished closure, which captures it and
            # is never disconnected. Both are tiny; what is NOT tiny is what the Worker still points at
            # (cfg, prior, posterior, CUDA tensors). Without this, every posterior you build stays
            # pinned for the life of the process even after SbiSession.reset_downstream drops it.
            worker.fn = None
            worker.args = ()
            worker.kwargs = {}
            self._cancel = None
            BasePanel._active_cancel = None
            if on_finished is not None:
                on_finished()

        worker.signals.finished.connect(_finished)
        self._set_busy(True)
        QThreadPool.globalInstance().start(worker)

    def _request_cancel(self):
        if self._cancel is None:
            return
        self._cancel.requested.set()
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Cancelling…")
        # A cancel is cooperative: it lands at the next print/tqdm redraw. That is ~1s almost
        # everywhere, but a neural-network training epoch and the SBC C2ST block are silent for longer.
        self.log_pane.append_line(
            "Cancelling — will stop at the next checkpoint (up to ~1 min during training).", "warning")

    def _set_busy(self, busy: bool):
        self._busy = busy
        BasePanel._running = busy
        self.btn_cancel.setVisible(busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_cancel.setText("Cancel")
        if busy:
            self.progress_pane.begin()
        else:
            self.progress_pane.end()   # authoritative: drops any row a crashed worker left behind
        # Lock EVERY panel's controls while a run is live, not just this one: redirect_streams swaps
        # sys.stdout/stderr process-wide, so another panel's ArtifactPicker refresh (which wraps
        # list_dir in redirect_stdout) or a model combo could swallow / corrupt the running worker's
        # stream -- the hazard set_controls_enabled documents, now spread across sibling inference tabs.
        # Only this panel keeps its Cancel button live (it lives outside `controls`).
        for panel in list(BasePanel._instances):
            panel.set_controls_enabled(not busy)

    def set_controls_enabled(self, enabled: bool):
        """Lock the whole left-hand column while a task runs.

        The WHOLE column, not just the run button: ArtifactPicker.refresh() (its ⟳ button, and the
        model combos that call it) wraps file_manager.list_dir in contextlib.redirect_stdout, which
        reassigns the PROCESS-WIDE sys.stdout -- i.e. the very stream redirect_streams installed for
        the running worker. Leaving a picker live mid-run lets a click swallow the worker's output,
        and if the worker's teardown restores sys.stdout inside that window, redirect_stdout.__exit__
        then reinstates the dead _SignalStream as the process's stdout permanently.
        """
        self.controls.setEnabled(enabled)
        if enabled:
            self.refresh_local_gates()   # re-apply this panel's own widget gating after a run frees it

    def refresh_local_gates(self) -> None:
        """Re-apply widget-level gating within this panel (which buttons/options are enabled). Base is a
        no-op; the inference sub-panels override it, and an owning screen may call it after a stage
        completes. Distinct from the tab-level greying an InferenceScreen does via setTabEnabled."""

    # ── persistence (subclasses override; keys are namespaced under group() by MainWindow) ──────────
    def insert_result_widget(self, index: int, widget, stretch: int = 1) -> None:
        """Mount a panel-specific primary view in the results column, above the figure stack.

        SimulatePanel puts a live pyqtgraph view here and hides the static figure stack. It used to
        reach into ``right_layout`` with ``insertWidget(0, view, 5)`` directly; now that the results
        column is a QSplitter that call would land the widget in the wrong parent, so this is the
        supported seam. Going through it also means the live view gets a drag handle like everything
        else in that column.
        """
        self.results_split.insertWidget(index, widget)
        self.results_split.setStretchFactor(index, stretch)

    # ── layout persistence ───────────────────────────────────────────────────
    # DELIBERATELY SEPARATE from save_settings/restore_settings. Two reasons, and the first is fatal:
    #   1. save_settings is overridden by 8 of the 9 panels WITHOUT calling super(), so anything
    #      hooked there would silently never run for them. This pair is driven from BasePanel itself
    #      (restore) and MainWindow._save_state (save), so a subclass cannot break it by forgetting.
    #   2. Layout is not a user *selection*. A panel that resets its pickers should not lose the
    #      column widths the user set.
    def layout_key(self) -> str:
        """Settings key for this panel's layout. Distinct per panel -- there are nine independent
        splitters. Subclasses that share a class name would override this; today all nine differ."""
        return type(self).__name__

    def save_layout(self, qs) -> None:
        """Persist the splitter geometry. Called by MainWindow._save_state and the debounce timer."""
        qs.setValue(f"layout/{self.layout_key()}/splitter", self.splitter.saveState())
        qs.setValue(f"layout/{self.layout_key()}/results", self.results_split.saveState())

    def restore_layout(self, qs=None) -> None:
        """Restore the splitter geometry, if any was stored.

        saveState/restoreState rather than sizes(): restoreState works before the widget is shown or
        polished, which setSizes does not do reliably. The return value MUST be checked -- it is
        False when the stored state does not match the current child count, and silently ignoring
        that is exactly the bug L9 describes for restoreGeometry.
        """
        from .. import settings as _settings
        qs = qs or _settings.settings()
        blob = qs.value(f"layout/{self.layout_key()}/splitter")
        if blob is not None and not self.splitter.restoreState(blob):
            self.splitter.setSizes(list(DEFAULT_SPLIT))     # stale/incompatible -> sane default
        # The results split has a variable child count (SimulatePanel inserts a live view), so
        # restoreState legitimately fails after such a change -- fall back rather than ignore.
        rblob = qs.value(f"layout/{self.layout_key()}/results")
        if rblob is not None and not self.results_split.restoreState(rblob):
            self.results_split.setSizes(list(DEFAULT_RESULTS_SPLIT))

    def _persist_layout(self) -> None:
        """Debounced write from splitterMoved. Touches ONLY this panel's layout key."""
        from .. import settings as _settings
        qs = _settings.settings()
        self.save_layout(qs)
        qs.sync()

    def save_settings(self, qs) -> None:
        """Persist this panel's user selections. Base is a no-op; subclasses override."""

    def restore_settings(self, qs) -> None:
        """Restore what save_settings wrote. Called at the END of a subclass __init__, after signals
        are connected -- restore order matters (a picker restored before its model gets wiped by the
        model's refresh())."""

    def _config_error(self, exc: Exception):
        """Report a failed config build as user-input trouble, not a crash.

        Deliberately catches broadly at the call sites: cli's builders raise a bare ValueError (NOT
        UnitParseError) for the two most plausible user mistakes -- a cell with no sibling bounds file
        (cli._parse_cell) and a cell missing a param the bounds file requires
        (cli.load_and_validate_gt -> SimConfig.inject_ground_truth). A
        narrow `except cli.UnitParseError` lets those escape the clicked slot and surface as a raw
        traceback in app.py's last-resort excepthook, with nothing in the panel's own log.
        """
        msg = str(exc)
        self.log_pane.append_line(f"Could not build the config: {msg}", "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)                  # user input, not a crash
        box.setWindowTitle("Check your inputs")
        box.setText("The configuration could not be built.")
        box.setInformativeText(msg)
        box.exec()

    def _on_error(self, message: str, tb: str):
        """Show a run failure. The traceback goes in a collapsible Details panel, not pasted whole into
        the body (which produced an unscrollable, un-copyable wall of text stretched to the widest stack
        frame)."""
        self.log_pane.append_line(message, "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Error")
        box.setText(message)
        if tb:
            box.setDetailedText(tb)
        box.exec()
