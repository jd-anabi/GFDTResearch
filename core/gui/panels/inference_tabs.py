"""Parameter Inference, split into six tabs over ONE shared SbiSession owned by the InferenceScreen.

    Config -> Prior -> Posterior -> Validate -> Infer -> TSNPE

Config records the MODEL-level choices as a ConfigDraft; the PRIOR tab picks the bounds file and turns
that draft into the SimConfig, because the bounds file declares the inferred parameter set (and hence the
observation mode), so a config cannot exist before it.

Each tab is its own BasePanel (its own FigureStack/ProgressPane/LogPane), so dispatch()/cancel/the
figure-sink/progress plumbing are reused verbatim; BasePanel._running is class-level, so the tabs can
never run concurrently. The screen owns the SbiSession and greys tabs via setTabEnabled; every tab
reads/writes the session through ``self._screen`` and calls ``self._screen.refresh_gates()`` after a
stage completes.
"""
# THE IMPORT SURFACE, kept deliberately: settings_screen reads this module's HELP and docstring BY
# STRING PATH ("core.gui.panels.inference_tabs"), inference_screen imports the six panels from here,
# and the test suites reach _ChiProbeRow, the _run_* runners and _nvidia_smi_free_gib through it.
# The implementations live in the inference/ package, one module per tab plus the shared matter.
from .inference.help_text import HELP  # noqa: F401
from .inference.rows import _ChiRangeRow, _ChiProbeRow  # noqa: F401
from .inference.runners import (_run_simulated_inference, _run_experimental_inference,  # noqa: F401
                                _run_experimental_inference_chi,
                                _run_experimental_inference_spontaneous, _run_tsnpe_round)
from .inference.base import (_StagePanel, _CellPreviewMixin, _TrainingBudgetMixin,  # noqa: F401
                             _hw_batch, _nvidia_smi_free_gib)
from .inference.config_tab import ConfigPanel  # noqa: F401
from .inference.prior_tab import PriorPanel  # noqa: F401
from .inference.posterior_tab import PosteriorPanel  # noqa: F401
from .inference.validate_tab import ValidatePanel  # noqa: F401
from .inference.infer_tab import InferPanel  # noqa: F401
from .inference.tsnpe_tab import TSNPEPanel  # noqa: F401
