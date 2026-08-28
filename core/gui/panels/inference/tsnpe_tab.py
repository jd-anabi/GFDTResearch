from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGroupBox, QLabel, QPushButton, QVBoxLayout)

from core import config
from core.config import OBSERVATION_PATH

from ... import icons, settings
from ...widgets.artifact_picker import ArtifactPicker
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from .posterior_tab import PosteriorPanel
from .runners import (_run_tsnpe_round)
from .base import _StagePanel, _TrainingBudgetMixin
from .help_text import HELP


# ── 6. TSNPE (truncated sequential NPE) ───────────────────────────────────────
class TSNPEPanel(_TrainingBudgetMixin, _StagePanel):
    """Tab 6. Refines the current posterior on ONE observation, without giving up the amortized one.

    ⚠⚠ WHAT THIS TAB MUST NOT BECOME. TSNPE proposes from the PRIOR RESTRICTED to an HPD region. It
    does NOT propose from the posterior. Fitting a density to the posterior and proposing from that
    gives ``p_L ∝ L^(L+1) q`` -- tempering -- and credible intervals then contract as
    ``(L+1)^(-1/2)`` with NO new information entering. SBC comes out flat anyway, because it validates
    the flow against the proposal it was trained on, so nothing on the Validate tab would catch it.
    The rule lives in ``core/SBI/truncate.py`` and is pinned by ``tests/test_conditioning_repair.py``.

    Gated on a posterior, the prior it was trained against, AND a persisted observation.
    An amortized posterior has no observation at SAVE time (``default_x`` is None on
    posterior_08232026), so the Infer tab records one at INFERENCE time and this tab keys on that.

    The budget group is the Posterior tab's, through ``_TrainingBudgetMixin``: a round is a simulation
    campaign, not a click, and the number belongs on screen before the button.

    Persists (group "inference_tsnpe"): the observation, the HPD level, the direction count and the
    two budget fields.
    """

    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        from core.SBI import truncate as _tr

        box = QGroupBox("TSNPE round")
        v = QVBoxLayout(box)
        warn = QLabel("Restricts the PRIOR to the posterior's credible region and retrains there. It "
                      "never proposes from the posterior itself. The result is NON-AMORTIZED and is "
                      "marked as such in its sidecar, so the load path will refuse it for general "
                      "inference.")
        warn.setWordWrap(True)
        warn.setTextFormat(Qt.PlainText)
        v.addWidget(warn)

        form = make_form()
        self.obs_picker = ArtifactPicker(OBSERVATION_PATH, keep=lambda fn: fn.endswith(".pt"))
        add_help_row(form, "Observation", self.obs_picker, HELP["tsnpe_obs"])
        self.hpd = FloatField(str(_tr.DEFAULT_HPD))
        self.n_dirs = IntField(str(_tr.DEFAULT_N_DIRECTIONS))
        add_help_row(form, "HPD level", self.hpd, HELP["tsnpe_hpd"])
        add_help_row(form, "Directions truncated", self.n_dirs, HELP["tsnpe_dirs"])
        v.addLayout(form)

        self.btn_round = QPushButton("Run TSNPE round")
        self.btn_round.setProperty("accent", True)         # primary CTA (Fluent accent)
        self.btn_round.clicked.connect(self._round)
        v.addWidget(self.btn_round)
        self.controls_layout.addWidget(box)

        # The SAME budget group the Posterior tab shows, and the same code behind it.
        budget = QGroupBox("Simulation budget for this round")
        bv = QVBoxLayout(budget)
        bform = make_form()
        self.num_runs = IntField(str(config.TRAINING_NUM_RUNS))
        self.run_size_cap = IntField(str(config.TRAINING_RUN_SIZE))
        add_help_row(bform, "Batches", self.num_runs, HELP["num_runs"])
        add_help_row(bform, "Max rows per batch (0 = auto)", self.run_size_cap, HELP["run_size"])
        bv.addLayout(bform)
        self.budget_total = self._derived_label()
        self.budget_mem = self._derived_label()
        self.budget_ckpt = self._derived_label()
        for lab in (self.budget_total, self.budget_mem, self.budget_ckpt):
            bv.addWidget(lab)
        for fld in (self.num_runs, self.run_size_cap):
            fld.textChanged.connect(lambda _t: self._sync_budget())
        self.controls_layout.addWidget(budget)
        self._sync_budget()

    def _round(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None or not self.obs_picker.key():
            return
        level, n_dirs = self.hpd.value(), self.n_dirs.value()
        if not (0.0 < level < 1.0):
            self.log_pane.append_line("HPD level must be strictly between 0 and 1.", "warning")
            return
        if n_dirs < 1:
            self.log_pane.append_line("At least one direction must be truncated.", "warning")
            return
        if level < 0.99:
            # A judgement, so it warns rather than refuses -- but deleted support is a ONE-WAY
            # ratchet, and a region that is too TIGHT is the expensive mistake, not the cheap one.
            self.log_pane.append_line(
                f"HPD {level:g} is tighter than the recommended {0.999:g}. Truncation permanently "
                f"deletes prior support; no later round can recover it.", "warning")
        n_runs, cap = self._budget_values()
        self.dispatch(_run_tsnpe_round, s.cfg, s.posterior, s.inf_prior, s.force_prior,
                      OBSERVATION_PATH / self.obs_picker.key(), n_dirs, level,
                      max(1, n_runs), max(0, cap), provide_fig_sink=True,
                      on_result=self._on_round)

    def _on_round(self, payload):
        """Install the round's posterior AND the region that makes it non-amortized.

        Without an on_result the round trains for hours and the result is discarded -- and without
        the region travelling with it, the deferred Save writes it marked amortized. Both halves
        matter; the second is the one that produces a wrong artifact rather than no artifact.
        """
        (posterior, diagnostics), region, digest = payload
        s = self.session
        s.posterior, s.diagnostics = posterior, diagnostics
        s.posterior_latent = getattr(posterior, "latent", None)
        s.V = PosteriorPanel._extract_rotation(posterior)
        s.truncation, s.x_obs_digest = region, digest
        self.log_pane.append_line(
            f"TSNPE round complete. This posterior is NON-AMORTIZED: it is valid near the "
            f"observation {digest}, and its sidecar will say so.", "warning")
        self._screen.refresh_gates()

    def refresh_local_gates(self):
        s = self.session
        self.obs_picker.refresh()
        self.btn_round.setEnabled(s.posterior is not None and s.inf_prior is not None
                                  and bool(self.obs_picker.key()))
        self._sync_budget()

    def save_settings(self, qs):
        qs.beginGroup("inference_tsnpe")
        qs.setValue("observation", self.obs_picker.key())
        qs.setValue("hpd", str(self.hpd.value()))
        qs.setValue("n_dirs", self.n_dirs.value())
        qs.setValue("num_runs", self.num_runs.value())
        qs.setValue("run_size_cap", self.run_size_cap.value())
        qs.endGroup()

    def restore_settings(self, qs):
        from core.SBI import truncate as _tr
        qs.beginGroup("inference_tsnpe")
        self.obs_picker.restore_key(settings.get_str(qs, "observation"))
        # get_str + float, because settings has no get_float and inventing one for a single caller
        # would be a wider change than this needs. A blank or unparseable value falls back to the
        # module default rather than to 0.0, which FloatField.value() would otherwise hand back --
        # and an HPD of 0 would truncate the prior to a point.
        try:
            self.hpd.setText(str(float(settings.get_str(qs, "hpd", str(_tr.DEFAULT_HPD)))))
        except ValueError:
            self.hpd.setText(str(_tr.DEFAULT_HPD))
        self.n_dirs.setText(str(settings.get_int(qs, "n_dirs", _tr.DEFAULT_N_DIRECTIONS)))
        self.num_runs.setText(str(settings.get_int(qs, "num_runs", config.TRAINING_NUM_RUNS)))
        self.run_size_cap.setText(str(settings.get_int(qs, "run_size_cap", config.TRAINING_RUN_SIZE)))
        qs.endGroup()
