"""The one place a QFormLayout is constructed.

Qt's default field-growth policy on Windows is ``FieldsStayAtSizeHint``: the label column takes the
widest label and every field gets its size hint and stops growing. Repo-wide there were ZERO calls to
setFieldGrowthPolicy or setRowWrapPolicy across 18 QFormLayout sites, which is the direct cause of
the "input boxes are cut off" report -- widening the window made the results column bigger and left
every numeric field at 3-6 characters.

``AllNonFixedFieldsGrow`` lets fields take the available width; ``WrapLongRows`` drops a field onto
its own line when the label plus field cannot fit, instead of squeezing the field to nothing. Going
through a factory means the 19th form gets this too, rather than reintroducing the bug.
"""
from PySide6.QtWidgets import QFormLayout


def make_form(parent=None) -> QFormLayout:
    """A QFormLayout whose fields grow with the column and whose long rows wrap."""
    form = QFormLayout(parent) if parent is not None else QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    form.setLabelAlignment(_label_alignment())
    return form


def _label_alignment():
    from PySide6.QtCore import Qt
    return Qt.AlignLeft | Qt.AlignVCenter


def apply_form_policy(form: QFormLayout) -> QFormLayout:
    """Retrofit an existing QFormLayout. For call sites that cannot use the factory directly."""
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    return form
