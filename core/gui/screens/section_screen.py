"""A section screen: a heading plus a tab widget of the section's panels.

Used for the simple sections -- Reduction Map (one tab) and FDT Analysis (FDT + Cross-validation). The
Parameter Inference section needs cross-tab gating over a shared session, so it has its own screen
(inference_screen.py) rather than using this generic host.
"""
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ..widgets.anim import crossfade_tab


class SectionScreen(QWidget):
    """Generic section host: a heading over a tab widget of independent panels.

    Drives nothing itself and persists nothing -- the panels it hosts own both.
    """

    def __init__(self, title: str, tabs, parent=None):
        """`tabs` is a list of ``(label, panel)`` pairs, added left to right."""
        super().__init__(parent)
        heading = QLabel(title)
        heading.setProperty("type", "heading")     # Fluent type ramp (global QSS)

        self.tabs = QTabWidget()
        for label, panel in tabs:
            self.tabs.addTab(panel, label)
        # Track the outgoing page + connect AFTER the loop (the first addTab already emitted
        # currentChanged(0) mid-construction, and the handler dereferences _prev_tab).
        self._prev_tab = self.tabs.currentWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(heading)
        layout.addWidget(self.tabs, 1)

    def _on_tab_changed(self, _index):
        crossfade_tab(self.tabs, self._prev_tab)          # no-op offscreen / single-tab / not-yet-visible
        self._prev_tab = self.tabs.currentWidget()

    def panels(self):
        """The section's panels (for MainWindow persistence + tests)."""
        return [self.tabs.widget(i) for i in range(self.tabs.count())]
