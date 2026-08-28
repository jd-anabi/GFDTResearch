from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout)

from core import config, forcing, orchestrator
from core.Helpers import file_manager
from core.config import BOUNDS_PATH, PRIOR_PATH

from ... import icons, settings
from ...widgets.artifact_picker import ArtifactPicker
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from ...widgets.param_grid import BoundsGrid, ValuesGrid
from ...widgets.source_toggle import SourceToggle
from .base import _StagePanel, _TrainingBudgetMixin
from .help_text import HELP


# ── 2. Prior (also picks the BOUNDS file, which is what builds the config) ────
class PriorPanel(_StagePanel):
    """Tab 2. Picks the BOUNDS file -- which is what turns the draft into a real SimConfig -- then
    builds or loads the parameter prior.

    Installs the config IN PLACE (``install_config``), not as a new session: building the prior is
    the first step of the existing session, so replacing it would discard the draft.

    Persists (group "inference_prior"): the bounds and prior picker selections. NOT the bounds grid --
    parameter names and order belong to the model, so hand-entry only ever edits numbers.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        self._saved_bounds_key = ""              # resolved in on_draft_set (needs the chosen model)
        box = QGroupBox("Prior")
        v = QVBoxLayout(box)
        form = make_form()
        self.bounds_picker = ArtifactPicker(BOUNDS_PATH / "nadrowski")
        self.bounds_grid = BoundsGrid()
        self.bounds_source = SourceToggle(self.bounds_picker, self.bounds_grid,
                                          file_label="Use file", direct_label="Edit values")
        self.bounds_source.changed.connect(self._on_bounds_source_changed)
        add_help_row(form, "Bounds", self.bounds_source, HELP["bounds_source"])
        self.prior_picker = ArtifactPicker(PRIOR_PATH, keep=lambda fn: fn.endswith(".pt"), allow_new=True)
        add_help_row(form, "Prior", self.prior_picker, HELP["prior"])
        v.addLayout(form)
        self.btn_prior = QPushButton("Build / Load prior")
        self.btn_prior.setProperty("accent", True)        # primary CTA (Fluent accent)
        self.btn_prior.clicked.connect(self._build_prior)
        v.addWidget(self.btn_prior)
        self.prior_name = QLineEdit()
        self.prior_name.setPlaceholderText("name to save prior as…")
        self.btn_save_prior = QPushButton("Save")
        self.btn_save_prior.clicked.connect(self._save_prior)
        row = QHBoxLayout()
        row.addWidget(self.prior_name, 1)
        row.addWidget(self.btn_save_prior)
        v.addLayout(row)
        self.controls_layout.addWidget(box)

        # -- the stability sweep: what actually builds the prior ---------------------------------
        sweep = QGroupBox("Stability sweep")
        sv = QVBoxLayout(sweep)
        sform = make_form()
        self.sweep_iters = IntField(str(config.PRIOR_SWEEP_ITERATIONS))
        self.sweep_batch = IntField(str(config.PRIOR_SWEEP_BATCH))
        self.sweep_max_sets = IntField(str(config.PRIOR_SWEEP_MAX_SETS))
        self.sweep_step = FloatField(str(config.PRIOR_SWEEP_STEP))
        self.sweep_units = FloatField(str(config.STABILITY_SWEEP_ND_UNITS))
        add_help_row(sform, "Global rounds", self.sweep_iters, HELP["sweep_iters"])
        add_help_row(sform, "Candidates per round (0 = auto)", self.sweep_batch, HELP["sweep_batch"])
        add_help_row(sform, "Max accepted sets", self.sweep_max_sets, HELP["sweep_max_sets"])
        add_help_row(sform, "Random-walk step", self.sweep_step, HELP["sweep_step"])
        add_help_row(sform, "Stability duration (ND units)", self.sweep_units, HELP["sweep_units"])
        sv.addLayout(sform)
        self.sweep_note = _TrainingBudgetMixin._derived_label()
        sv.addWidget(self.sweep_note)
        for fld in (self.sweep_iters, self.sweep_batch, self.sweep_max_sets, self.sweep_units):
            fld.textChanged.connect(lambda _t: self._sync_sweep())
        self.controls_layout.addWidget(sweep)
        self._sync_sweep()

        # -- clustering: a different STAGE from the sweep. The sweep maps the stable manifold;
        # this decides how many MODES the prior has, because HDBSCAN's label count becomes the
        # GMM's n_components.
        clust = QGroupBox("Clustering / GMM")
        cv = QVBoxLayout(clust)
        cform = make_form()
        self.cluster_size = IntField(str(config.PRIOR_CLUSTER_MIN_SIZE))
        self.cluster_samples = IntField(str(config.PRIOR_CLUSTER_MIN_SAMPLES))
        add_help_row(cform, "Min cluster size", self.cluster_size, HELP["cluster_size"])
        add_help_row(cform, "Min samples", self.cluster_samples, HELP["cluster_samples"])
        cv.addLayout(cform)
        self.controls_layout.addWidget(clust)

        self.restore_settings(settings.settings())

    def on_draft_set(self, draft):
        """Config applied: repoint the bounds picker at the new model's folder and re-apply the saved
        key (it could not resolve at __init__, before any model was chosen)."""
        self.bounds_picker.repoint(BOUNDS_PATH / draft.model.lower(), self._saved_bounds_key)
        if self.bounds_source.is_direct():        # a different model means a different parameter set
            self._on_bounds_source_changed()

    def _on_bounds_source_changed(self):
        """Entering direct-entry mode seeds the grid FROM the selected file: the parameter names and
        their order belong to the model, so hand-entry edits numbers rather than inventing a schema."""
        if not self.bounds_source.is_direct():
            return
        path = self.bounds_picker.selected_path()
        if not path:
            self.log_pane.append_line("Select a bounds file first — direct entry starts from it.",
                                      "warning")
            self.bounds_source.set_direct(False)
            return
        try:
            params, rescale, forcing, _ = file_manager.parse_bounds_file(path)
        except Exception as e:                       # noqa: BLE001
            self._config_error(e)
            self.bounds_source.set_direct(False)
            return
        self.bounds_grid.load(params, rescale, forcing)

    def _build_prior(self):
        """Build the SimConfig from (Config draft + this tab's bounds file), then build/load the prior.

        The config is built HERE because the bounds file is what defines the inferred parameter set --
        and therefore the observation mode -- so it cannot exist until this tab has been used."""
        draft = self.session.draft
        if draft is None:
            return
        if self.bounds_source.is_direct():
            problems = self.bounds_grid.problems()
            if problems:
                self.log_pane.append_line("Fix the bounds first: " + "; ".join(problems), "warning")
                return
            source = dict(bounds_dicts=self.bounds_grid.to_dicts())
        else:
            bounds_path = self.bounds_picker.selected_path()
            if not bounds_path:
                self.log_pane.append_line("Select a bounds file first.", "warning")
                return
            source = dict(bounds_path=bounds_path)
        try:
            cfg = draft.make_config(**source)
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return
        for msg in cfg.check_unit_consistency():      # a units declaration that contradicts the pipeline
            self.log_pane.append_line(msg, "warning")
        self.session.reset_downstream("prior")
        self._screen.install_config(cfg)             # sets session.cfg + repoints the Infer tab + re-gates
        _MODE_BLURB = {
            "spontaneous": "one passive trace, no drive anywhere",
            "forced": "passive + one forced trace at the cell's drive",
            "chi": "passive + K single-tone forced traces (the cell's own drive is ignored)",
        }
        self.log_pane.append_line(
            f"Config built: {cfg.model} — {len(cfg.params_dict)} ND + {len(cfg.rescale_params)} rescale "
            f"params. Observation mode: {cfg.observation_mode.upper()} "
            f"({_MODE_BLURB[cfg.observation_mode]}).")
        if cfg.observation_mode == "spontaneous" and "f_scale" in cfg.rescale_params:
            self.log_pane.append_line(
                "This config infers f_scale but has no drive anywhere, so f_scale cannot affect the "
                "observable — its marginal will just return the prior. Use a bounds file without a "
                "Forcing section AND without f_scale for spontaneous inference.", "warning")
        if cfg.chi_mode:
            lo, hi = cfg.chi_freq_bounds
            # Width from the SHARED rule, not 3*K: that was layout 1, where a probe's frequency was
            # implied by its slot. Under the padded probe set it is CHI_ELEM_W * chi_k_pad and does
            # not depend on K at all -- which is the entire point of the layout, so reporting the old
            # formula here told the user the opposite of what the mode now does.
            self.log_pane.append_line(
                f"χ(ω) mode: {cfg.chi_n_freqs} drive frequencies over {lo:g}–{hi:g}×Ω₀ at ND amplitude "
                f"{cfg.chi_f0:g}, each locked in over at most {cfg.chi_max_cycles:g} drive cycles; "
                f"conditioning is [S(41) | log T | χ({orchestrator.expected_forcing_dim(cfg)})] over "
                f"{cfg.chi_k_pad} probe slots. Train a NEW posterior (the width differs from a non-χ one).")

        entry, is_new = self.prior_picker.selected()
        # Passed, never written to config: orchestrator does `from .config import
        # PRIOR_SWEEP_ITERATIONS, ...`, so assigning to the constants here would be a silent no-op.
        self.dispatch(orchestrator.build_prior, cfg, entry, is_new, save=False,
                      provide_fig_sink=True, on_result=self._on_prior,
                      num_iterations=max(1, self.sweep_iters.value()),
                      sweep_batch=max(0, self.sweep_batch.value()),
                      max_sets=max(1, self.sweep_max_sets.value()),
                      walk_step=self.sweep_step.value() or config.PRIOR_SWEEP_STEP,
                      stability_units=self.sweep_units.value() or config.STABILITY_SWEEP_ND_UNITS,
                      min_cluster_size=max(2, self.cluster_size.value()),
                      min_samples=max(1, self.cluster_samples.value()))

    def _sync_sweep(self) -> None:
        """The one derived line: how many candidates the GLOBAL census screens, and where the time
        goes. Pure and cheap, so it is safe on every keystroke; wrapped because a status line must
        never be able to raise into refresh_gates and take the tab down."""
        try:
            cfg = self.session.cfg
            hw_batch = getattr(getattr(cfg, "hw", None), "batch_size", None) or config.detect_device().batch_size
            per_round = self.sweep_batch.value() or hw_batch
            rounds = max(1, self.sweep_iters.value())
            units = self.sweep_units.value() or config.STABILITY_SWEEP_ND_UNITS
            dt = getattr(cfg, "dt_nd_min", None)
            steps = f"{int(units / dt):,}" if dt else "?"
            self.sweep_note.setText(
                f"Global census screens {rounds * per_round:,} candidates ({rounds:,} rounds x "
                f"{per_round:,}), each integrated over {steps} steps.\n"
                f"The LOCAL flood-fill then runs until {max(1, self.sweep_max_sets.value()):,} sets "
                f"are accepted — that is the dominant cost of a prior build, and it now runs on the "
                f"same device as the global sweep (falling back to the CPU when there is no "
                f"accelerator).")
        except Exception as e:                    # noqa: BLE001 -- never break the tab over a label
            self.sweep_note.setText(f"Sweep summary unavailable: {type(e).__name__}: {e}")

    def _on_prior(self, payload):
        self.session.inf_prior, self.session.force_prior = payload
        self.log_pane.append_line("Prior ready.")
        self._screen.refresh_gates()

    def _save_prior(self):
        name = self.prior_name.text().strip()
        if not name or self.session.inf_prior is None:
            self.log_pane.append_line("Build a prior and enter a name first.", "warning")
            return
        nd_prior = self.session.inf_prior.distributions[0]
        self.dispatch(orchestrator.save_prior_artifacts, name, nd_prior, self.session.cfg,
                      on_finished=lambda: (self.prior_picker.refresh(),
                                           self.log_pane.append_line(f"Saved prior '{name}'.")))

    def refresh_local_gates(self):
        self.btn_prior.setEnabled(self.session.draft is not None)
        self.btn_save_prior.setEnabled(self.session.inf_prior is not None)

    def save_settings(self, qs):
        qs.beginGroup("inference_prior")
        qs.setValue("prior", self.prior_picker.key())
        qs.setValue("bounds", self.bounds_picker.key())
        qs.setValue("bounds_source", self.bounds_source.key())
        for name in ("sweep_iters", "sweep_batch", "sweep_max_sets", "sweep_step", "sweep_units",
                     "cluster_size", "cluster_samples"):
            qs.setValue(name, str(getattr(self, name).value()))
        qs.endGroup()
        # The bounds GRID is not persisted: it is seeded from whichever file is selected, so restoring a
        # stale hand-edited grid against a different model/bounds would silently mis-bind parameters.

    def restore_settings(self, qs):
        qs.beginGroup("inference_prior")
        self.prior_picker.restore_key(settings.get_str(qs, "prior"))
        # The bounds picker points at CONFIG's model, which is not known at __init__ -- stash the key and
        # re-apply it in on_draft_set (the same deferred-restore trap the cell pickers have).
        self._saved_bounds_key = settings.get_str(qs, "bounds")
        # str + cast, because settings has no get_float; a blank or unparseable value falls back to
        # the config constant rather than to FloatField.value()'s 0.0 -- and a 0 here would mean a
        # sweep with no rounds, or a flood-fill that stops at zero accepted sets.
        for name, default, cast in (("sweep_iters", config.PRIOR_SWEEP_ITERATIONS, int),
                                    ("sweep_batch", config.PRIOR_SWEEP_BATCH, int),
                                    ("sweep_max_sets", config.PRIOR_SWEEP_MAX_SETS, int),
                                    ("sweep_step", config.PRIOR_SWEEP_STEP, float),
                                    ("sweep_units", config.STABILITY_SWEEP_ND_UNITS, float),
                                    ("cluster_size", config.PRIOR_CLUSTER_MIN_SIZE, int),
                                    ("cluster_samples", config.PRIOR_CLUSTER_MIN_SAMPLES, int)):
            try:
                getattr(self, name).setText(str(cast(settings.get_str(qs, name, str(default)))))
            except (TypeError, ValueError):
                getattr(self, name).setText(str(default))
        # Always start in FILE mode: direct entry has to be seeded from a file, and no file is selected
        # until on_draft_set runs. The saved mode is deliberately not restored for that reason.
        self.bounds_source.set_direct(False)
        qs.endGroup()


# ── the training budget, shared by the Posterior and TSNPE tabs ───────────────────────────────
