"""The Parameter Inference section: five tabs over ONE shared SbiSession, with cross-tab gating.

    Config -> Prior -> Posterior -> Validate -> Infer

Config records the MODEL-level choices (model, units, chi/rotation knobs) as a ConfigDraft. The Prior tab
picks the BOUNDS file and turns that draft into the SimConfig, because the bounds file is what declares
which parameters are inferred, in what order and over what range -- a SimConfig cannot exist without it,
and the choice of bounds is also what selects the observation mode (a Forcing section or not).

The screen owns the session (the single source of truth); tabs read it through their ``_screen``
back-reference and never cache it. ``refresh_gates`` is the truth table that greys tabs via
setTabEnabled after every stage.
"""
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ..panels.inference_tabs import (ConfigPanel, InferPanel, PosteriorPanel, PriorPanel,
                                     TSNPEPanel, ValidatePanel)
from ..session import SbiSession
from ..widgets.anim import crossfade_tab


class InferenceScreen(QWidget):
    """The Parameter Inference section: six stage tabs over one shared SbiSession.

    Drives the Config -> Prior -> Posterior -> Validate -> Infer -> TSNPE workflow and owns the
    cross-tab gating (refresh_gates' setTabEnabled truth table, re-run after every stage). Persists
    nothing itself -- each panel saves its own settings group.
    """

    def __init__(self, title="Parameter Inference", parent=None):
        super().__init__(parent)
        self.session = SbiSession()

        heading = QLabel(title)
        heading.setProperty("type", "heading")     # Fluent type ramp (global QSS)

        self.tabs = QTabWidget()
        # Size to the CURRENT tab. The six pages differ wildly -- Validate is one label and one
        # button, Infer in chi mode is 10-25 rows -- and QTabWidget's default is to size to the
        # largest, so switching tabs visibly reflowed the whole results column and the short pages
        # carried a large dead gap.
        self.tabs.currentChanged.connect(self._size_to_current_tab)
        self.config_panel = ConfigPanel(self)
        self.prior_panel = PriorPanel(self)
        self.posterior_panel = PosteriorPanel(self)
        self.validate_panel = ValidatePanel(self)
        self.infer_panel = InferPanel(self)
        # TSNPE sits AFTER Infer, and the order is the workflow: a round needs an observation, and
        # the Infer tab is what records one (an amortized posterior has none at save time).
        self.tsnpe_panel = TSNPEPanel(self)
        for label, panel in (("Config", self.config_panel), ("Prior", self.prior_panel),
                             ("Posterior", self.posterior_panel), ("Validate", self.validate_panel),
                             ("Infer", self.infer_panel), ("TSNPE", self.tsnpe_panel)):
            self.tabs.addTab(panel, label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(heading)
        layout.addWidget(self.tabs, 1)

        self.refresh_gates()
        # Track the outgoing page + connect AFTER refresh_gates: its programmatic setCurrentIndex(0) fires
        # currentChanged, and the handler dereferences _prev_tab (which must exist by then).
        self._prev_tab = self.tabs.currentWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, _index):
        crossfade_tab(self.tabs, self._prev_tab)          # no-op under offscreen / not-yet-visible
        self._prev_tab = self.tabs.currentWidget()

    def _size_to_current_tab(self, index: int) -> None:
        """Let only the visible tab contribute to the QTabWidget's size hint (see __init__)."""
        from PySide6.QtWidgets import QSizePolicy
        for i in range(self.tabs.count()):
            page = self.tabs.widget(i)
            if page is None:
                continue
            policy = page.sizePolicy()
            policy.setVerticalPolicy(QSizePolicy.Preferred if i == index else QSizePolicy.Ignored)
            page.setSizePolicy(policy)
        current = self.tabs.widget(index)
        if current is not None:
            current.updateGeometry()

    def panels(self):
        return [self.config_panel, self.prior_panel, self.posterior_panel,
                self.validate_panel, self.infer_panel, self.tsnpe_panel]

    def new_draft(self, draft):
        """Config applied: replace the WHOLE session (a different model or unit system invalidates every
        artifact) and repoint the Prior tab's bounds picker at the new model's folder."""
        self.session = SbiSession(draft=draft)
        self.prior_panel.on_draft_set(draft)
        self.refresh_gates()

    def install_config(self, cfg):
        """Prior stage: bounds chosen, so the SimConfig now exists. Set it IN PLACE and fan out to the
        tabs whose pickers/fields depend on it.

        Deliberately NOT a new session: the Prior stage installs the config as the first step of building
        the prior, so replacing the session here would wipe the artifact it is about to store."""
        self.session.cfg = cfg
        self.infer_panel.on_config_built(cfg)
        self.refresh_gates()

    def refresh_gates(self):
        s = self.session
        has_draft = s.draft is not None
        has_cfg = s.cfg is not None
        # NOT force_prior: build_forcing_prior returns None for any NO-FORCING model (spontaneous /
        # BP / no-forcing user models / a chi-mode config), so requiring it made Validate permanently
        # unreachable for exactly the models the CLI validates fine. The inferred prior is the one
        # validate_calibration actually consumes; force_prior is passed through and may legitimately be None.
        can_validate = s.posterior is not None and s.inf_prior is not None
        can_infer = s.posterior is not None                        # infer_and_visualize needs no prior

        self.tabs.setTabEnabled(1, has_draft)        # Prior      (picks bounds -> builds the config)
        self.tabs.setTabEnabled(2, has_cfg)          # Posterior
        self.tabs.setTabEnabled(3, can_validate)     # Validate
        self.tabs.setTabEnabled(4, can_infer)        # Infer
        # TSNPE needs what Validate needs, PLUS an observation on disk -- and the observation gate
        # is the panel's own (refresh_local_gates), because it depends on Resources/Observations
        # rather than on the session. Enabling the TAB on the session alone keeps the tooltip
        # useful: "no observation yet" is a different message from "no posterior yet".
        self.tabs.setTabEnabled(5, can_validate)     # TSNPE

        self.tabs.setTabToolTip(1, "" if has_draft else "Apply a model in Config first.")
        self.tabs.setTabToolTip(2, "" if has_cfg else
                                "Choose a bounds file and build/load a prior first — that is what builds "
                                "the config.")
        self.tabs.setTabToolTip(3, "" if can_validate else
                                "Needs a posterior AND its prior — build/load a prior, then a posterior.")
        self.tabs.setTabToolTip(4, "" if can_infer else "Train or load a posterior first.")
        self.tabs.setTabToolTip(5, "" if can_validate else
                                "Needs a posterior AND its prior. It also needs an observation, "
                                "which the Infer tab records when it runs.")

        for panel in self.panels():
            panel.refresh_local_gates()

        # If the visible tab just got disabled (e.g. re-running the prior greys Validate/Infer), Qt would
        # jump to an arbitrary neighbour; make it deterministic -- fall back to Config.
        if not self.tabs.isTabEnabled(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(0)
