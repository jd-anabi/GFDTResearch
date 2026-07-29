"""A tabbed container of figure images. Figures are rendered to PNG on the worker thread (see
BasePanel's fig_sink) and shown here as QPixmaps -- deliberately NOT live FigureCanvasQTAgg widgets:
painting a matplotlib figure that was created on a worker thread deadlocks on matplotlib's global
lock. Static PNGs sidestep that entirely and unify with the FDT/Reduction/CrossVal runners, which
already save PNGs to disk.

Each tab carries a "Pop out" button that re-opens the figure in its own window where it CAN be zoomed,
panned and saved (core/gui/widgets/figure_window.py):
  * SBI-panel figures also arrive as a pickled Figure -> a true interactive matplotlib window,
    reconstructed on the GUI thread (never painting the worker's original -- see GOTCHA #2).
  * disk-PNG figures (and any figure whose pickle is missing / unloadable) -> a pan/zoom image viewer.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QScrollArea, QTabWidget, QVBoxLayout,
                               QWidget)


class _FitLabel(QLabel):
    """A QLabel that scales its pixmap DOWN to the available width, keeping aspect ratio.

    Never scales UP past 1:1 -- enlarging a raster figure only blurs it. The full-resolution pixmap
    is retained, so the "Pop out" window still gets the original.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self.setMinimumSize(1, 1)

    def set_source(self, pix: QPixmap) -> None:
        self._source = pix
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        avail = max(1, self.width())
        if self._source.width() <= avail:
            super().setPixmap(self._source)          # already fits: show it 1:1
            return
        super().setPixmap(self._source.scaledToWidth(avail, Qt.SmoothTransformation))


class FigureStack(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabCloseRequested.connect(self._close_tab)
        # Pop-out windows are parentless top-levels; hold them or Python/Qt GCs them the instant show()
        # returns (the lifetime trap BasePanel.dispatch handles for its workers). Dropped on close.
        self._windows = set()

    def add_figure(self, title: str, png_bytes: bytes, fig_pickle=None):
        """Show a figure rendered to PNG bytes on a worker thread. ``fig_pickle`` (when present) is the
        pickled Figure, so "Pop out" can rebuild an interactive copy on the GUI thread."""
        pix = QPixmap()
        if png_bytes:
            pix.loadFromData(png_bytes, "PNG")
        self._add_tab(title, pix, fig_pickle=fig_pickle, png_path=None)

    def add_png(self, title: str, path):
        """Show a PNG that a runner saved to disk (FDT/Reduction/CrossVal modes). No Figure exists for
        these, so "Pop out" is a pan/zoom image viewer over the file."""
        self._add_tab(title, QPixmap(str(path)), fig_pickle=None, png_path=str(path))

    def _add_tab(self, title: str, pix: QPixmap, *, fig_pickle, png_path):
        # FIT TO WIDTH. Stage figures are rendered at dpi=110 with bbox_inches="tight", routinely
        # 1200-1800px wide, and were shown at full resolution in a scroll area with no scaling -- so
        # an embedded figure was ALWAYS scrolled in both axes, while only the pop-out window had
        # Fit/100% controls. Downscale-only (never enlarge past 1:1, which would just blur it).
        label = _FitLabel()
        label.setAlignment(Qt.AlignCenter)
        if pix.isNull():
            label.setText(f"(could not render figure: {title})")
        else:
            label.set_source(pix)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)

        pop = QPushButton("Pop out")
        pop.setToolTip("Open an interactive, zoomable copy in a new window" if fig_pickle is not None
                       else "Open a zoomable copy in a new window")

        container = QWidget()
        # Stash everything the pop-out needs directly on the tab widget, so it travels with the tab
        # (tabs are movable) and is freed when the tab closes.
        container._fig_pickle = fig_pickle
        container._png_path = png_path
        container._pix = pix
        container._title = title
        pop.clicked.connect(lambda: self._pop_out(container))

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addStretch(1)
        top.addWidget(pop)
        layout.addLayout(top)
        layout.addWidget(scroll, 1)

        idx = self.addTab(container, title)
        self.setCurrentIndex(idx)

    def _pop_out(self, container):
        from .figure_window import ImageZoomWindow, build_interactive_window

        title = getattr(container, "_title", "Figure")
        window = None
        if getattr(container, "_fig_pickle", None) is not None:
            try:
                window = build_interactive_window(container._fig_pickle, title, on_close=self._drop_window)
            except Exception:      # noqa: BLE001 -- a version-skewed / unloadable pickle: fall back
                window = None
        if window is None:
            pix = container._pix
            if container._png_path:                       # prefer the crisp full-res file on disk
                disk = QPixmap(container._png_path)
                if not disk.isNull():
                    pix = disk
            window = ImageZoomWindow(pix, title, on_close=self._drop_window)
        self._windows.add(window)
        window.show()
        window.raise_()

    def _drop_window(self, window):
        self._windows.discard(window)

    def _close_tab(self, index: int):
        widget = self.widget(index)
        self.removeTab(index)
        widget.deleteLater()

    def clear_all(self):
        while self.count():
            self._close_tab(0)
