"""One labelled-fields-in-a-row widget, replacing four near-identical copies.

`_GridRow` (crossval: min/max/n), `_ChiRangeRow` (inference: lo/hi multipliers), `_BoundsRow`
(param_grid: min/max -- whose own docstring admitted it was "the crossval _GridRow pattern") and
`_ParamRow` (model builder: value/min/max + an auto checkbox) were the same widget four times: a
zero-margin QHBoxLayout alternating a caption QLabel with a stretch-1 field.

Collapsing them is not only tidiness -- it is the single place the composite rows' sizing lives.
Those rows split an already-narrow controls column N ways, so this is where a minimum width or a
wrap policy has to go, and having it in four places meant it went in none.
"""
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget


class LabeledFieldRow(QWidget):
    """A row of ``(caption, widget)`` pairs, captions at their natural width, fields sharing the rest.

    :param pairs: iterable of ``(caption, widget)``. A falsy caption omits the label, so a row can
                  hold an unlabelled control (the model builder's "auto" checkbox) alongside
                  labelled ones.
    :param spacing: inter-item spacing; the default matches the hand-rolled rows this replaces.
    """

    def __init__(self, pairs, *, spacing: int = 6, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        self.fields = []
        for caption, widget in pairs:
            if caption:
                layout.addWidget(QLabel(caption))
            # stretch 1: captions keep their size hint, the fields absorb the remaining width. With
            # the field classes' own minimum widths this is what stops a 3-field row from squeezing
            # every field to a few characters.
            layout.addWidget(widget, 1)
            self.fields.append(widget)

    def values(self) -> tuple:
        """Each field's ``value()``, in construction order."""
        return tuple(f.value() for f in self.fields)
