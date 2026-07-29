"""The Settings + Help screen (reached from the bottom-left gear's "Full settings…").

Two parts: an Appearance chooser (Follow-system / Light / Dark / Auto) that calls back to MainWindow to
apply + persist the mode, and a per-section Help reference assembled from the user-facing panel module
docstrings and their per-control ``HELP`` dicts -- filling the gap where the app had no section-level
"what is this for" text. Reached only from the gear; the back arrow still returns Home.
"""
import html
import importlib
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QRadioButton, QScrollArea, QTextBrowser, QVBoxLayout, QWidget)

from core import registry

from ..design import HELP_VIEWER_MIN_H
from ..theming import MODE_LABELS

# (module path, human title) for the five user-facing sections, in display order. Each module opens with
# a "<MODE> mode: ..." docstring line and (most) carry a per-control HELP dict.
_HELP_SECTIONS = (
    ("core.gui.panels.reduction_panel", "Reduction Map"),
    ("core.gui.panels.fdt_panel", "FDT Analysis"),
    ("core.gui.panels.crossval_panel", "Cross-validation (sweep study)"),
    ("core.gui.panels.inference_tabs", "Parameter Inference"),
    ("core.gui.panels.simulate_panel", "Simulate"),
)


def _first_paragraph(doc: str) -> str:
    """The first paragraph of a docstring (up to the first blank line) -- the user-facing summary."""
    out = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        out.append(line.strip())
    return " ".join(out)


class SettingsScreen(QWidget):
    def __init__(self, on_mode, current_mode="system", parent=None, *,
                 on_open_builder=None, on_edit_model=None, on_delete_model=None,
                 on_system_accent=None, system_accent=False,
                 on_force_inter=None, force_inter=False):
        """``on_mode(mode)`` is called when the user picks an appearance; ``current_mode`` pre-selects.

        The keyword callbacks wire the "User-defined models" group and the accent/font checkboxes to
        MainWindow. All default to None/False so tests can construct the screen with the historical
        two-argument signature -- the extra controls are then inert.
        """
        super().__init__(parent)
        self._on_mode = on_mode
        self._on_open_builder = on_open_builder
        self._on_edit_model = on_edit_model
        self._on_delete_model = on_delete_model
        self._on_system_accent = on_system_accent
        self._on_force_inter = on_force_inter

        heading = QLabel("Settings")
        heading.setProperty("type", "heading")     # Fluent type ramp (global QSS)

        appearance = QGroupBox("Appearance")
        av = QVBoxLayout(appearance)
        self._group = QButtonGroup(self)
        self._radios = {}
        for mode, label in MODE_LABELS:
            rb = QRadioButton(label)
            self._group.addButton(rb)
            self._radios[mode] = rb
            av.addWidget(rb)
        # Select the current mode BEFORE connecting, so construction never fires on_mode (which would
        # re-enter MainWindow before it has finished wiring the gear menu + this screen).
        self._radios.get(current_mode, self._radios["system"]).setChecked(True)
        for mode, rb in self._radios.items():
            rb.toggled.connect(lambda checked, m=mode: checked and self._on_mode(m))

        # Accent + font toggles: state set BEFORE connecting (same construction-never-fires rule).
        self.accent_check = QCheckBox("Use the Windows accent colour")
        self.accent_check.setChecked(bool(system_accent))
        self.accent_check.toggled.connect(
            lambda on: self._on_system_accent and self._on_system_accent(on))
        if sys.platform != "win32":                    # the OS accent read is Windows-only
            self.accent_check.setVisible(False)
        av.addWidget(self.accent_check)

        self.inter_check = QCheckBox("Use the Inter font everywhere")
        self.inter_check.setChecked(bool(force_inter))
        self.inter_check.toggled.connect(
            lambda on: self._on_force_inter and self._on_force_inter(on))
        av.addWidget(self.inter_check)
        inter_note = QLabel("Font changes apply immediately; some views refresh fully after a restart.")
        inter_note.setProperty("type", "caption")
        av.addWidget(inter_note)

        models_box = QGroupBox("User-defined models")
        mv = QVBoxLayout(models_box)
        self._models_list = QVBoxLayout()          # rebuilt by refresh_models()
        mv.addLayout(self._models_list)
        btn_new = QPushButton("Open model builder")
        btn_new.setProperty("accent", True)
        btn_new.clicked.connect(lambda: self._on_open_builder and self._on_open_builder())
        mv.addWidget(btn_new)
        self.refresh_models()

        help_box = QGroupBox("Help — what each section does")
        hv = QVBoxLayout(help_box)
        help_label = self._build_help()
        # A word-wrapped QLabel does not propagate heightForWidth through the intervening layouts of a
        # setWidgetResizable QScrollArea, so this blob (five module docstrings plus every HELP dict,
        # several thousand characters) got a height computed for the wrong width and the BOTTOM WAS
        # UNREACHABLE. A QTextBrowser lays out its own rich text and scrolls itself, which sidesteps
        # the propagation problem entirely -- and it makes the help selectable, which a QLabel is not.
        viewer = QTextBrowser()
        viewer.setHtml(help_label.text())
        viewer.setOpenExternalLinks(True)
        viewer.setMinimumHeight(HELP_VIEWER_MIN_H)
        hv.addWidget(viewer)

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.addWidget(heading)
        iv.addWidget(appearance)
        iv.addWidget(models_box)
        iv.addWidget(help_box)
        iv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def set_mode(self, mode: str) -> None:
        """Reflect a mode change that came from the gear popover (without re-firing ``on_mode``)."""
        rb = self._radios.get(mode)
        if rb is not None and not rb.isChecked():
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)

    def refresh_models(self) -> None:
        """Rebuild the user-model list (one row per registered model + any startup load failures)."""
        while self._models_list.count():
            item = self._models_list.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        names = registry.user_model_names()
        if not names and not registry.load_errors:
            empty = QLabel("No user-defined models yet. Build one to add it to the Simulate model list.")
            empty.setProperty("type", "caption")
            self._models_list.addWidget(empty)
        for name in names:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(name))
            h.addStretch(1)
            btn_edit = QPushButton("Edit")
            btn_edit.clicked.connect(lambda _=False, n=name: self._on_edit_model and self._on_edit_model(n))
            btn_del = QPushButton("Delete")
            btn_del.clicked.connect(lambda _=False, n=name: self._on_delete_model and self._on_delete_model(n))
            h.addWidget(btn_edit)
            h.addWidget(btn_del)
            self._models_list.addWidget(row)
        for path, msg in registry.load_errors:
            err = QLabel(f"Failed to load {path.name}: {msg}")
            err.setProperty("type", "caption")
            err.setWordWrap(True)
            self._models_list.addWidget(err)

    def _build_help(self) -> QLabel:
        parts = []
        for modpath, title in _HELP_SECTIONS:
            try:
                mod = importlib.import_module(modpath)     # already imported by MainWindow -> cache hit
            except Exception:                              # noqa: BLE001 -- Help is best-effort
                continue
            parts.append(f"<h3>{html.escape(title)}</h3>")
            summary = _first_paragraph(mod.__doc__ or "")
            if summary:
                parts.append(f"<p>{html.escape(summary)}</p>")
            help_dict = getattr(mod, "HELP", None)
            if isinstance(help_dict, dict) and help_dict:
                items = "".join(
                    f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>"
                    for k, v in help_dict.items())
                parts.append(f"<ul>{items}</ul>")
        label = QLabel("".join(parts))
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop)
        return label
