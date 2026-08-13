"""Small typed input widgets mirroring the CLI's numeric/path prompts (_prompt_int/_prompt_float and
the experimental-data file prompts)."""
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget

from ..design import FIELD_MIN_W, PATH_FIELD_MIN_W

# These classes set a validator and, now, a MINIMUM WIDTH. Qt's default QFormLayout policy on Windows
# is FieldsStayAtSizeHint, so without a floor a numeric field rendered 3-6 characters wide: typing
# "0.033333" scrolled inside the box and the value could not be read back without clicking into it.
# The matching half of the fix is the field-growth policy (see widgets/forms.make_form).


class FloatField(QLineEdit):
    def __init__(self, default: float = 0.0, parent=None):
        super().__init__(str(default), parent)
        self.setValidator(QDoubleValidator())
        self.setMinimumWidth(FIELD_MIN_W)

    def value(self) -> float:
        try:
            return float(self.text())
        except ValueError:
            return 0.0

    def value_or_none(self) -> "float | None":
        """The field's value, or None when it does not parse.

        Use this ANYWHERE 0.0 is a legal value the user could also have meant, because ``value()``
        cannot tell "0" from "" or from "-" mid-typing -- a blank "min" box silently becomes a real
        bound of 0. That is not hypothetical: it is why widgets/param_grid validates explicitly (see
        its module docstring), and the model builder's own parameter rows had the bug it warns about.
        """
        try:
            return float(self.text().strip())
        except (TypeError, ValueError):
            return None


class IntField(QLineEdit):
    def __init__(self, default: int = 0, parent=None):
        super().__init__(str(default), parent)
        self.setValidator(QIntValidator())
        self.setMinimumWidth(FIELD_MIN_W)

    def value(self) -> int:
        try:
            return int(self.text())
        except ValueError:
            return 0


class PathField(QWidget):
    def __init__(self, file_filter: str = "Data (*.csv *.npy);;All files (*)", parent=None):
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setMinimumWidth(PATH_FIELD_MIN_W)
        # Absolute paths elide at the FRONT, so the filename stays visible; the tooltip carries the
        # whole thing (a per-frequency chi row leaves only ~120px of line edit).
        self.edit.textChanged.connect(lambda t: self.edit.setToolTip(t))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", self._filter)
        if path:
            self.edit.setText(path)

    def value(self) -> str:
        return self.edit.text().strip()
