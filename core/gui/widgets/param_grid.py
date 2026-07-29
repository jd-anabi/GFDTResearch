"""Editable grids for entering parameter BOUNDS and cell VALUES by hand instead of picking a file.

Both grids are LOADED FROM a parsed file first and then edited: the parameter NAMES and their ORDER are
not the user's to invent -- they are fixed by the model, and the order is load-bearing (simulators bind
parameter columns positionally via ``torch.unbind``, and build_prior asserts the ND order matches a user
model's compiled signature). So these widgets only ever let you change NUMBERS, never the schema, which
is what makes hand-entry safe.

Validation is explicit because ``FloatField.value()`` silently returns 0.0 on unparseable text -- a blank
"min" box would otherwise become a real bound of 0. ``problems()`` returns human-readable strings and the
callers refuse to build anything while it is non-empty.
"""
from collections import OrderedDict

from PySide6.QtWidgets import (QFormLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget)

from .field_row import LabeledFieldRow
from .labeled_inputs import FloatField
from .forms import make_form

_MAX_VISIBLE_HEIGHT = 320          # keep a 13-parameter grid from swallowing the controls column


def _num(field) -> "float | None":
    """The field's value, or None when it does not parse (FloatField.value() would return 0.0)."""
    try:
        return float(field.text().strip())
    except (TypeError, ValueError):
        return None


class _PassThroughScrollArea(QScrollArea):
    """A scroll area that hands the wheel back to its parent once it cannot scroll any further.

    This is the ONLY nested scroll area in the app -- a bounds grid sits inside a SourceToggle inside
    BasePanel's controls scroll area. Qt's default is for the inner area to swallow every wheel event
    it receives, so scrolling over the grid moved the grid and then simply stopped: you could not
    reach the "Build / Load prior" button below it by scrolling from there, which reads as the page
    being stuck.

    Ignoring the event at the limit lets it propagate to the outer area, which is the behaviour
    people expect from nested scrollers everywhere else.
    """

    def wheelEvent(self, event):
        bar = self.verticalScrollBar()
        going_up = event.angleDelta().y() > 0
        at_limit = (bar.value() == bar.minimum()) if going_up else (bar.value() == bar.maximum())
        if at_limit or bar.maximum() == 0:
            event.ignore()               # -> outer scroll area handles it
            return
        super().wheelEvent(event)


def _scrollable(inner: QWidget) -> QScrollArea:
    area = _PassThroughScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    # Sized to CONTENT up to a ceiling, rather than a flat 320px for everything: a 3-parameter grid
    # and a 13-parameter grid used to reserve exactly the same box.
    area.setMaximumHeight(_MAX_VISIBLE_HEIGHT)
    inner.adjustSize()
    area.setMinimumHeight(min(inner.sizeHint().height() + 4, _MAX_VISIBLE_HEIGHT))
    return area


class _BoundsRow(LabeledFieldRow):
    """min / max for one parameter."""

    def __init__(self, lo: float, hi: float, parent=None):
        self.lo, self.hi = FloatField(lo), FloatField(hi)
        super().__init__((("min", self.lo), ("max", self.hi)), parent=parent)


class BoundsGrid(QWidget):
    """A (min, max) row per parameter, across the ND / Dimensional / Forcing sections.

    Produces exactly what ``file_manager.parse_bounds_file`` produces -- ``{name: (None, (lo, hi))}``
    per section, in the loaded order -- so downstream code cannot tell the difference."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = OrderedDict()                 # (section, name) -> _BoundsRow
        self._sections = OrderedDict()             # section -> [names] (preserves file order)
        inner = QWidget()
        self._form = make_form(inner)
        self._form.setContentsMargins(0, 0, 0, 0)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))
        self._placeholder = QLabel("(load a bounds file to edit its values)")
        self._placeholder.setProperty("type", "caption")
        self._form.addRow(self._placeholder)

    def load(self, params: OrderedDict, rescale: OrderedDict, forcing: OrderedDict) -> None:
        """Populate from parsed bounds dicts. Rebuilds the whole grid, preserving section + name order."""
        while self._form.rowCount():
            self._form.removeRow(0)
        self._rows, self._sections = OrderedDict(), OrderedDict()
        for section, d, title in (("PARAM", params, "Non-dimensional"),
                                  ("RESCALE", rescale, "Dimensional"),
                                  ("FORCING", forcing, "Forcing")):
            if not d:
                continue
            header = QLabel(title)
            header.setProperty("type", "caption")
            self._form.addRow(header)
            self._sections[section] = list(d)
            for name, (_v, (lo, hi)) in d.items():
                row = _BoundsRow(lo, hi)
                self._rows[(section, name)] = row
                self._form.addRow(name, row)

    def is_loaded(self) -> bool:
        return bool(self._rows)

    def problems(self) -> list:
        if not self.is_loaded():
            return ["no bounds loaded — pick a file to start from, then edit it"]
        out = []
        for (section, name), row in self._rows.items():
            lo, hi = _num(row.lo), _num(row.hi)
            if lo is None or hi is None:
                out.append(f"{name}: min and max must both be numbers")
            elif not lo < hi:
                out.append(f"{name}: min ({lo:g}) must be less than max ({hi:g})")
        return out

    def to_dicts(self) -> tuple:
        """(params, rescale, forcing) in parse_bounds_file's shape. Call only when problems() is empty."""
        out = {s: OrderedDict() for s in ("PARAM", "RESCALE", "FORCING")}
        for section, names in self._sections.items():
            for name in names:
                row = self._rows[(section, name)]
                out[section][name] = (None, (_num(row.lo), _num(row.hi)))
        return out["PARAM"], out["RESCALE"], out["FORCING"]


class ValuesGrid(QWidget):
    """One value field per initial condition and per parameter -- a hand-editable cell file.

    Produces ``file_manager.parse_values_file``'s shape: (inits, params, rescale, forcing) plain
    ``{name: value}`` dicts, ready for ``SimConfig.inject_ground_truth``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields = OrderedDict()               # (section, name) -> FloatField
        self._sections = OrderedDict()
        inner = QWidget()
        self._form = make_form(inner)
        self._form.setContentsMargins(0, 0, 0, 0)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))
        self._placeholder = QLabel("(load a cell file to edit its values)")
        self._placeholder.setProperty("type", "caption")
        self._form.addRow(self._placeholder)

    def load(self, inits: dict, params: dict, rescale: dict, forcing: dict) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._fields, self._sections = OrderedDict(), OrderedDict()
        for section, d, title in (("INITS", inits, "Initial conditions"),
                                  ("PARAM", params, "Non-dimensional"),
                                  ("RESCALE", rescale, "Dimensional"),
                                  ("FORCING", forcing, "Forcing")):
            if not d:
                continue
            header = QLabel(title)
            header.setProperty("type", "caption")
            self._form.addRow(header)
            self._sections[section] = list(d)
            for name, v in d.items():
                fld = FloatField(float(v))
                self._fields[(section, name)] = fld
                self._form.addRow(name, fld)

    def is_loaded(self) -> bool:
        return bool(self._fields)

    def problems(self) -> list:
        if not self.is_loaded():
            return ["no values loaded — pick a cell file to start from, then edit it"]
        return [f"{name}: must be a number" for (_s, name), fld in self._fields.items()
                if _num(fld) is None]

    def to_dicts(self) -> tuple:
        """(inits, params, rescale, forcing). Call only when problems() is empty."""
        out = {s: OrderedDict() for s in ("INITS", "PARAM", "RESCALE", "FORCING")}
        for section, names in self._sections.items():
            for name in names:
                out[section][name] = _num(self._fields[(section, name)])
        return out["INITS"], out["PARAM"], out["RESCALE"], out["FORCING"]
