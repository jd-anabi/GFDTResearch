"""Shared base classes and budget machinery for the inference tabs."""
import subprocess

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from core import config, forcing, orchestrator
from core.config import CELL_PATH
from core.SBI import pipeline, training_checkpoint

from ..base_panel import BasePanel
from ...widgets.artifact_picker import ArtifactPicker


class _StagePanel(BasePanel):
    """Common base for the five inference tabs: holds a back-reference to the owning InferenceScreen and
    reads/writes the shared session through it (never caching the session object, which Config replaces
    wholesale on each build)."""

    def __init__(self, screen, parent=None):
        super().__init__(parent)
        self._screen = screen

    @property
    def session(self):
        return self._screen.session



class _CellPreviewMixin:
    """Cell-picker handling for the Infer tab (its only user): the cell folder follows the BUILT
    config's model (there is no live model combo in that tab), so the picker is repointed in
    on_config_built and the saved key re-applied there (it could not resolve at __init__, before
    any config exists)."""

    def _init_cell_picker(self):
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self._saved_cell_key = ""

    def on_config_built(self, cfg):
        self.cell_picker.repoint(CELL_PATH / cfg.model.lower(), self._saved_cell_key)



def _hw_batch(cfg) -> int:
    """The rows-per-batch the run will actually use when the cap field is 0 (= auto)."""
    return (getattr(getattr(cfg, "hw", None), "batch_size", None)
            or config.detect_device().batch_size)


def _nvidia_smi_free_gib() -> "float | None":
    """Free VRAM in GiB according to ``nvidia-smi``, or None if it cannot be read.

    ⚠ DELIBERATELY NOT ``torch.cuda.mem_get_info``. That reading overstates free VRAM on Windows by
    roughly the size of the desktop -- measured 15037 MiB against nvidia-smi's 5814 at the same
    instant -- and it is the number that green-lit the batch which killed the first chi
    retrain. Showing it next to a field whose whole purpose is to bound VRAM would be handing the
    user the exact lie the field exists to defend against.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0)
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:                        # noqa: BLE001 -- no driver, no binary, a timeout: all "unknown"
        return None


class _TrainingBudgetMixin:
    """Batches x rows-per-batch, and the three derived lines that say what it will cost.

    EXTRACTED rather than duplicated when the TSNPE tab arrived (each round is a simulation
    campaign, not a click, so its cost belongs on screen). A second copy of _budget_memory would
    have been a second place for pipeline's cost model to be restated wrongly, and the whole
    point of that method is that it reads the planner's own numbers instead of restating them.

    A user of this mixin must create ``num_runs``, ``run_size_cap`` (FloatField-likes with
    ``.value()``) and the three ``budget_*`` labels, then call ``_sync_budget()``.
    """
    @staticmethod
    def _derived_label() -> QLabel:
        """A read-only derived line under the budget fields.

        Word-wrapped and PlainText on purpose: these carry generated strings that can be long (a
        checkpoint directory name and the field it differs in), and an unwrapped label widens the
        whole controls column -- the same unwrapped-label defect that once put a permanent
        horizontal scrollbar on the crossval panel.
        """
        lab = QLabel()
        lab.setWordWrap(True)
        lab.setTextFormat(Qt.PlainText)
        return lab

    def _hardware(self):
        """The DeviceConfig a config would be built with. Memoised: detect_device() probes the
        accelerator and _sync_budget runs on every keystroke in the two fields."""
        if getattr(self, "_hw_cache", None) is None:
            self._hw_cache = config.detect_device()
        return self._hw_cache

    def _budget_values(self) -> tuple:
        return self.num_runs.value(), self.run_size_cap.value()

    def _effective_width(self, cfg, cap: int) -> tuple:
        """(rows actually simulated per batch, the hardware default it was capped from)."""
        hw = cfg.hw if cfg is not None else self._hardware()
        return (min(hw.batch_size, cap) if cap else hw.batch_size), hw

    def _sync_budget(self) -> None:
        """Recompute the three derived lines. Pure and cheap -- safe on every keystroke.

        ⚠ Treats anything without a `.hw` as no-config-yet rather than trusting `session.cfg` to be a
        SimConfig. Two reasons, and the second is the real one: the gate tests set `session.cfg` to a
        bare `object()` sentinel, and more importantly a derived STATUS LINE must never be able to
        raise into refresh_gates() and take the whole tab down with it.
        """
        cfg = self.session.cfg
        if getattr(cfg, "hw", None) is None:
            cfg = None
        n_runs, cap = self._budget_values()
        width, hw = self._effective_width(cfg, max(0, cap))
        n_runs = max(1, n_runs)

        capped = "" if width == hw.batch_size else f" (capped from {hw.batch_size:,})"
        chi = ""
        if cfg is not None and cfg.chi_mode:
            chi = (f"\nIn chi mode each row costs 1+K solver passes, K up to {cfg.chi_k_pad}.")
        self.budget_total.setText(
            f"{n_runs * width:,} simulations = {n_runs:,} batches x {width:,} rows{capped}."
            f"\nBatches is also the (t_scale, T) diversity count: every row in a batch shares one "
            f"operating point, so batch COUNT is the statistics and batch WIDTH is not.{chi}")
        self.budget_mem.setText(self._budget_memory(cfg, hw, width))
        self.budget_ckpt.setText(self._budget_checkpoint(cfg, width, n_runs))

    def _budget_memory(self, cfg, hw, width: int) -> str:
        """Peak device memory for ONE batch at the worst geometry the Sobol pre-filter admits.

        The WORST case is the one worth showing: n_fine swings from a median ~40k to a p99 ~283k, so a
        width that fits the median still OOMs on a few percent of batches -- which is how the
        2026-08-10 and 2026-08-11 retrains both died. gen_training_data rejects any (t_scale, T) whose
        n_fine exceeds min(N_ND_MAX, len(t)), so that IS the ceiling, by construction.

        Reads pipeline's own cost model rather than restating it, so this cannot drift from what the
        planner does (pipeline.peak_sim_elements / sim_memory_budget_elements).
        """
        if hw.device.type != "cuda":
            return f"Peak-memory estimate is CUDA-only; this config runs on {hw.device.type}."
        n_fine = min(config.N_ND_MAX, cfg.t.shape[0]) if cfg is not None else config.N_ND_MAX
        n_vars = len(cfg.inits_dict) if cfg is not None else 3
        steady = cfg.steady_idx if cfg is not None else 0
        n_ch = 1
        if cfg is not None:
            try:
                n_ch = forcing.n_force_channels(cfg.model, cfg.forcing_idx, n_vars)
            except Exception:                 # noqa: BLE001 -- a display must not break the tab
                n_ch = 1
        try:
            # var_idx=0 in the training path, so exactly one variable is kept.
            need = pipeline.peak_sim_elements(width, n_fine, steady, n_vars, n_ch, 1)
            have = pipeline.sim_memory_budget_elements(hw.device, hw.dtype)
        except Exception as e:                # noqa: BLE001
            return f"Peak-memory estimate unavailable: {type(e).__name__}: {e}"
        gib = hw.dtype.itemsize / float(1 << 30)
        verdict = ("fits in one piece" if need <= have else
                   "does NOT fit -- the planner will split this batch, costing wall-clock")
        return (f"Worst-case peak ~{need * gib:.2f} GiB per batch (n_fine <= {n_fine:,}, {n_vars} "
                f"state vars); planner budget right now ~{have * gib:.2f} GiB, so it {verdict}.\n"
                f"That budget is an UPPER bound: free-VRAM readings overstate what is really "
                f"available by roughly the size of the desktop, so closing browsers is a bigger lever "
                f"than lowering the cap." + ("" if cfg is not None else
                                             "\n(Estimated from hardware defaults until a config is built.)"))

    def _budget_checkpoint(self, cfg, width: int, n_runs: int) -> str:
        """THE GUARD THAT MAKES THESE TWO FIELDS SAFE TO EXPOSE AT ALL.

        Both are inside the training-checkpoint identity, which is DIGESTED into the checkpoint's
        directory name. Change either and a resumable multi-day run silently resolves to a different
        directory -- there is no error, just a run that starts from zero. A tooltip is not a strong
        enough guard for that, so the state is stated inline and re-evaluated as you type;
        describe_siblings even names the field that differs.
        """
        if not config.TRAINING_CHECKPOINT_EVERY:
            return "Checkpointing is off (config.TRAINING_CHECKPOINT_EVERY = 0): a crash loses the run."
        if cfg is None or self.session.inf_prior is None:
            return "Checkpoint status needs a config and a prior -- the prior is part of the identity."
        try:
            ident = orchestrator.training_identity(cfg, self.session.inf_prior, width, n_runs)
            state = training_checkpoint.peek(training_checkpoint.resolve_dir(ident))
            siblings = training_checkpoint.describe_siblings(ident)
        except Exception as e:                # noqa: BLE001 -- never let a status line break the tab
            return f"Checkpoint status unavailable: {type(e).__name__}: {e}"
        if state and state.get("batches_done"):
            done = int(state["batches_done"])
            if state.get("complete"):
                return (f"Resumes a COMPLETE checkpoint ({done:,} batches) -- simulation will be "
                        f"skipped entirely and only the flow retrained.")
            return f"Resumes an existing checkpoint: {done:,}/{n_runs:,} batches already done."
        if siblings:
            return ("WARNING: these settings match no checkpoint, so this starts a NEW run.\n"
                    + siblings)
        return "No checkpoint exists yet; this starts a new run."

