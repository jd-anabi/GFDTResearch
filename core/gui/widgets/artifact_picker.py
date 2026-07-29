"""A combo box populated from ``file_manager.list_dir(base, keep=...)`` -- the GUI equivalent of the
CLI file pickers (cells, bounds, priors, posteriors). Optionally offers a '(from scratch)' sentinel."""
import contextlib
import io
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

_TOOLTIP_ROLE = Qt.ToolTipRole

from core.Helpers import file_manager
from core.gui import icons


class ArtifactPicker(QWidget):
    NEW_LABEL = "➕  (from scratch)"

    def __init__(self, base_path, keep=None, allow_new: bool = False, parent=None):
        super().__init__(parent)
        self.base_path = Path(base_path)
        self._keep = keep
        self._allow_new = allow_new

        self.combo = QComboBox()
        refresh = QPushButton()                 # keep QPushButton -> QSS selector QPushButton#iconButton
        refresh.setObjectName("iconButton")     # compact square button -> small QSS padding (glyph fits)
        icons.apply_icon(refresh, "refresh")    # bundled icon font (falls back to "⟳")
        refresh.setFixedWidth(32)
        refresh.setToolTip("Rescan the folder")
        refresh.clicked.connect(self.refresh)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)
        layout.addWidget(refresh)
        self.refresh()

    def refresh(self):
        self.combo.clear()
        if self._allow_new:
            self.combo.addItem(self.NEW_LABEL, userData=None)
        entries = []
        if self.base_path.exists():
            with contextlib.redirect_stdout(io.StringIO()):   # list_dir prints a tree; suppress it
                entries = file_manager.list_dir(str(self.base_path), keep=self._keep)
        for entry in entries:
            # display forward-slashed for subfoldered layouts; keep the raw relpath as data
            self.combo.addItem(entry.replace("\\", "/"), userData=entry)
            # Per-item tooltip: AdjustToContentsOnFirstShow sizes the combo to whatever was present
            # at first show, and refresh() adds entries LATER (after a save, or a model change), so
            # those get silently elided with no way to read them.
            self.combo.setItemData(self.combo.count() - 1, entry.replace("\\", "/"), _TOOLTIP_ROLE)

    def selected(self):
        """Return (entry_or_None, is_new). ``entry`` is the path relative to base_path."""
        data = self.combo.currentData()
        return data, (data is None)

    def selected_path(self):
        """Full filesystem path of the selected entry, or None for the '(from scratch)' sentinel."""
        data = self.combo.currentData()
        return None if data is None else str(self.base_path / data)

    def has_entries(self) -> bool:
        return any(self.combo.itemData(i) is not None for i in range(self.combo.count()))

    # ── persistence ──────────────────────────────────────────────────────────
    def key(self) -> str:
        """A stable string identifying the current selection, for QSettings. The relpath (userData),
        not the index (order shifts as files are added), and "" for the '(from scratch)' sentinel."""
        data = self.combo.currentData()
        return "" if data is None else str(data)

    def restore_key(self, key: str) -> None:
        """Reselect a previously saved key. If the file is gone (findData == -1), leave the current
        selection alone rather than blanking the combo with setCurrentIndex(-1)."""
        if not key:
            return
        i = self.combo.findData(key)
        if i >= 0:
            self.combo.setCurrentIndex(i)
