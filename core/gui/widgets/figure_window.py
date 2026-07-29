"""Pop-out figure windows: a figure a stage produced, re-opened in its own top-level window where it
can be zoomed, panned and saved.

Two window types, one per figure source (see FigureStack):

  * InteractiveFigureWindow -- for SBI-panel figures that arrived as a pickled matplotlib Figure. The
    figure is UNPICKLED ON THE GUI THREAD and embedded in a live FigureCanvasQTAgg with matplotlib's
    navigation toolbar (zoom-rect / pan / home / save). This is the whole reason the sink ships the
    pickle: a figure BUILT on the worker thread must never be painted by a live canvas -- it deadlocks
    on matplotlib's global lock (PRISM_HANDOFF.md trap G2). A figure UNPICKLED here is a fresh
    main-thread object, so painting it is safe.

  * ImageZoomWindow -- for figures we only have as a PNG on disk (FDT / Reduction / CrossVal, whose
    runners save PNGs to disk rather than hand a Figure back), and as the fallback when a pickle is
    missing or fails to load. A QGraphicsView gives wheel-zoom, drag-pan and fit-to-window over the
    static image.

Both are parentless top-level windows. FigureStack holds the ONLY reference (a set) and drops it in
each window's closeEvent -- without that pin, Python/Qt would GC the window the moment show() returns
(the same lifetime trap BasePanel.dispatch handles for its workers).
"""
import pickle

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

_MIN_SCALE = 0.05
_MAX_SCALE = 40.0
_WHEEL_STEP = 1.15


def build_interactive_window(fig_pickle, title, on_close=None, parent=None):
    """Unpickle a figure on the GUI thread and wrap it in an InteractiveFigureWindow.

    THE Gcf DETACH -- do not remove. Every stage figure is pyplot-managed (visualizers.py uses
    plt.subplots; corner / sbi-pairplot use plt.figure), so matplotlib bakes `_restore_to_pylab=True`
    into the pickle and `pickle.loads` RE-REGISTERS the figure into the process-global pyplot registry
    (matplotlib._pylab_helpers.Gcf). Left there it (a) leaks for the life of the process and (b) is
    destroyed by Worker.run's `plt.close("all")` -- which fires on EVERY run and every cancel -- tearing
    the manager out from under a figure the user is currently viewing. So we snapshot Gcf around the
    load and destroy whatever it registered: the returned figure is then owned solely by our Qt canvas,
    and pyplot can never touch it. Verified: plt.get_fignums() is unchanged across a pop-out.
    """
    import matplotlib._pylab_helpers as pylab_helpers
    before = set(pylab_helpers.Gcf.figs)
    fig = pickle.loads(fig_pickle)
    for num in set(pylab_helpers.Gcf.figs) - before:
        pylab_helpers.Gcf.destroy(num)
    return InteractiveFigureWindow(fig, title=title, on_close=on_close, parent=parent)


class InteractiveFigureWindow(QWidget):
    """A live matplotlib canvas for a (main-thread) figure, with the standard navigation toolbar."""

    def __init__(self, fig, title="Figure", on_close=None, parent=None):
        super().__init__(parent)
        # The Qt-agg backend is imported HERE -- lazily, and only when a figure is actually popped out --
        # so app start and the headless, non-pop-out tests never pull it in, and an import failure is
        # catchable at click time (FigureStack falls back to the image viewer) instead of breaking the
        # whole app. Importing the module does NOT switch pyplot's backend: the app stays on Agg and we
        # simply embed a canvas of our own. Never call plt.show() / matplotlib.use() from here.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        self._on_close = on_close
        self._fig = fig
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(900, 650)

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.canvas.draw_idle()

    def closeEvent(self, event):
        if self._on_close is not None:
            self._on_close(self)
        # Drop our refs so the (Gcf-detached) figure and its canvas can be collected.
        self._fig = None
        self.canvas = None
        self.toolbar = None
        super().closeEvent(event)


class _ZoomView(QGraphicsView):
    """A QGraphicsView that wheel-zooms about the cursor and drag-pans, clamped to a sane scale range."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)   # zoom toward the cursor
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    def zoom(self, factor):
        target = self.transform().m11() * factor
        if target < _MIN_SCALE or target > _MAX_SCALE:
            return
        self.scale(factor, factor)

    def wheelEvent(self, event):
        self.zoom(_WHEEL_STEP if event.angleDelta().y() > 0 else 1 / _WHEEL_STEP)
        event.accept()


class ImageZoomWindow(QWidget):
    """A pan/zoom viewer over a static image (a QPixmap). Used for disk-PNG figures and as the fallback
    when a figure could not be unpickled."""

    def __init__(self, pixmap, title="Figure", on_close=None, parent=None):
        super().__init__(parent)
        self._on_close = on_close
        self._pixmap = pixmap
        self._fitted = False
        self.view = None
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(900, 700)

        layout = QVBoxLayout(self)

        if pixmap is None or pixmap.isNull():
            layout.addWidget(QLabel(f"(could not load figure: {title})"))
            return

        self.scene = QGraphicsScene(self)
        self.item = QGraphicsPixmapItem(pixmap)
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self.item)
        self.view = _ZoomView(self.scene, self)

        bar = QHBoxLayout()
        for label, slot in (("Fit", self._fit), ("100%", self._actual_size),
                            ("Zoom in", lambda: self.view.zoom(_WHEEL_STEP)),
                            ("Zoom out", lambda: self.view.zoom(1 / _WHEEL_STEP)),
                            ("Save As…", self._save)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            bar.addWidget(button)
        bar.addStretch(1)

        layout.addLayout(bar)
        layout.addWidget(self.view, 1)

    def showEvent(self, event):
        super().showEvent(event)
        # Fit once, after the view has its real geometry -- but never again, so a later window resize
        # does not fight the zoom level the user has dialled in. The Fit button re-fits on demand.
        if self.view is not None and not self._fitted:
            self._fitted = True
            self._fit()

    def _fit(self):
        if self.view is not None:
            self.view.resetTransform()
            self.view.fitInView(self.item, Qt.KeepAspectRatio)

    def _actual_size(self):
        if self.view is not None:
            self.view.resetTransform()      # identity => one screen pixel per image pixel

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save figure as", "", "PNG image (*.png)")
        if path:
            self._pixmap.save(path)

    def closeEvent(self, event):
        if self._on_close is not None:
            self._on_close(self)
        super().closeEvent(event)
