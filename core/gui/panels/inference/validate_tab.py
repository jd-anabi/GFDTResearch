from PySide6.QtWidgets import (QGroupBox, QLabel, QPushButton, QVBoxLayout)

from core import config, orchestrator

from ... import icons, settings
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from .base import _StagePanel
from .help_text import HELP


# ── 4. Validate ───────────────────────────────────────────────────────────────
class ValidatePanel(_StagePanel):
    """Tab 4. Runs the calibration battery (SBC / TARP / PPC) on the current posterior.

    Gated on a posterior AND ``inf_prior`` -- deliberately not on ``force_prior``, which is None for
    every no-forcing model and once made this tab permanently unreachable for exactly those.

    Persists: nothing. It has no configurable inputs.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Validate (SBC + TARP)")
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Data-free calibration. Needs a posterior and the prior it was trained against."))
        cform = make_form()
        self.cal_n = IntField(str(config.SBC_N_CAL))
        self.cal_scales = IntField(str(config.CAL_N_SCALES))
        add_help_row(cform, "Calibration datasets", self.cal_n, HELP["cal_n"])
        add_help_row(cform, "(t_scale, T) operating points", self.cal_scales, HELP["cal_scales"])
        v.addLayout(cform)
        self.btn_validate = QPushButton("Run calibration")
        self.btn_validate.setProperty("accent", True)     # primary CTA (Fluent accent)
        self.btn_validate.clicked.connect(self._validate)
        v.addWidget(self.btn_validate)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _validate(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None:   # force_prior is legitimately None (no drive)
            return
        self.dispatch(orchestrator.validate_calibration, s.cfg, s.posterior,
                      s.inf_prior, s.force_prior, provide_fig_sink=True,
                      n_cal=max(1, self.cal_n.value()),
                      cal_n_scales=max(1, self.cal_scales.value()))

    def refresh_local_gates(self):
        s = self.session
        self.btn_validate.setEnabled(s.posterior is not None and s.inf_prior is not None)

    def save_settings(self, qs):
        qs.beginGroup("inference_validate")
        qs.setValue("cal_n", self.cal_n.value())
        qs.setValue("cal_scales", self.cal_scales.value())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_validate")
        self.cal_n.setText(str(settings.get_int(qs, "cal_n", config.SBC_N_CAL)))
        self.cal_scales.setText(str(settings.get_int(qs, "cal_scales", config.CAL_N_SCALES)))
        qs.endGroup()
