"""A two-way source selector: take a value from a FILE, or type it in DIRECTLY.

Used wherever the inference flow accepts either an artifact on disk or hand-entered values (units,
parameter bounds, cell values). It owns nothing domain-specific -- it just swaps between the two widgets
its caller supplies and reports which one is live -- so the same control works for a units line edit, a
grid of (lo, hi) bound rows, or a grid of parameter values.

Follows the panel convention of setting the initial state BEFORE connecting signals, so constructing one
never fires a handler back into a half-built panel.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QRadioButton, QVBoxLayout, QWidget)

from .adaptive_stack import AdaptiveStack


class SourceToggle(QWidget):
    """Radio pair over a QStackedWidget. ``changed`` fires whenever the live source switches."""

    changed = Signal()

    def __init__(self, file_widget: QWidget, direct_widget: QWidget, *,
                 file_label: str = "From file", direct_label: str = "Enter directly",
                 direct: bool = False, parent=None):
        super().__init__(parent)
        self.file_widget = file_widget
        self.direct_widget = direct_widget

        self._radio_file = QRadioButton(file_label)
        self._radio_direct = QRadioButton(direct_label)
        self._group = QButtonGroup(self)
        self._group.addButton(self._radio_file, 0)
        self._group.addButton(self._radio_direct, 1)

        # AdaptiveStack: the file page is one picker row, the direct page is a 320px-capped grid, so
        # a plain QStackedWidget reserved the grid's height even while showing the picker.
        self._stack = AdaptiveStack()
        self._stack.addWidget(file_widget)
        self._stack.addWidget(direct_widget)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._radio_file)
        row.addWidget(self._radio_direct)
        row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(row)
        layout.addWidget(self._stack)

        # State first, THEN the connection: constructing this must not call back into the owner.
        self.set_direct(direct)
        self._group.idToggled.connect(self._on_toggled)

    def _on_toggled(self, _id, checked):
        if checked:                                    # idToggled fires twice (off then on); act once
            self._stack.setCurrentIndex(1 if self.is_direct() else 0)
            self.changed.emit()

    def is_direct(self) -> bool:
        return self._radio_direct.isChecked()

    def set_direct(self, direct: bool) -> None:
        """Set the live source without emitting ``changed`` (for restore paths)."""
        blocked = self._group.blockSignals(True)
        (self._radio_direct if direct else self._radio_file).setChecked(True)
        self._group.blockSignals(blocked)
        self._stack.setCurrentIndex(1 if direct else 0)

    def key(self) -> str:
        """Persistable state: which source is live."""
        return "direct" if self.is_direct() else "file"

    def restore_key(self, key: str) -> None:
        self.set_direct(str(key) == "direct")
