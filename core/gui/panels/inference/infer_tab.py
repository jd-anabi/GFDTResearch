import math

from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

from core import cli, config, forcing
from core.Helpers import file_manager, labels
from core.config import T_MIN_EXP_S

from ... import icons, settings
from ...widgets.adaptive_stack import AdaptiveStack
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from ...widgets.param_grid import BoundsGrid, ValuesGrid
from ...widgets.source_toggle import SourceToggle
from .rows import _ChiProbeRow
from .runners import (_run_experimental_inference, _run_experimental_inference_chi, _run_experimental_inference_spontaneous, _run_simulated_inference)
from .base import _CellPreviewMixin, _StagePanel
from .help_text import HELP


# ── 5. Infer ──────────────────────────────────────────────────────────────────
class InferPanel(_StagePanel, _CellPreviewMixin):
    """Tab 5. Infers on a simulated observation (from a cell's ground truth) or on real recordings.

    Its cell picker follows the BUILT config's model rather than a live combo -- there isn't one in
    this tab. In chi mode the experimental page grows one file-picker row per probe frequency, which
    is why K is bounded at ``config.CHI_K_MAX``.

    Persists (group "inference_infer"): the mode, the cell picker, and the experimental file paths.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        self._init_cell_picker()
        self._cell_problems = []             # why the picked cell can't be used (empty = usable)
        self.cell_picker.combo.currentIndexChanged.connect(lambda _i: self._on_cell_changed())
        self._forcing_fields = {}            # name -> FloatField (experimental drive)
        box = QGroupBox("Infer")
        v = QVBoxLayout(box)

        self.infer_mode = QComboBox()
        self.infer_mode.addItems(["Simulated (cell ground truth)", "Experimental data"])
        self.infer_mode.currentIndexChanged.connect(
            lambda _i: (self._sync_infer_page(), self.refresh_local_gates()))
        mode_form = make_form()
        add_help_row(mode_form, "Mode", self.infer_mode, HELP["infer_mode"])
        v.addLayout(mode_form)

        # AdaptiveStack: the simulated page is two rows, the chi page is K+3, so a plain stack left a
        # large dead gap under the short pages.
        self.infer_stack = AdaptiveStack()
        # simulated inputs
        sim_w = QWidget(); sim_f = make_form(sim_w)
        self.sim_tobs = FloatField(T_MIN_EXP_S)
        self.values_grid = ValuesGrid()
        self.cell_source = SourceToggle(self.cell_picker, self.values_grid,
                                        file_label="Use file", direct_label="Edit values")
        self.cell_source.changed.connect(self._on_cell_source_changed)
        add_help_row(sim_f, "Cell", self.cell_source, HELP["cell_source"])
        add_help_row(sim_f, "T_obs (s)", self.sim_tobs, HELP["tobs"])
        self.infer_stack.addWidget(sim_w)
        # experimental inputs
        exp_w = QWidget(); self.exp_form = make_form(exp_w)
        self.exp_spont = PathField()
        self.exp_forced = PathField()
        self.exp_tobs = FloatField(T_MIN_EXP_S)
        add_help_row(self.exp_form, "Spontaneous", self.exp_spont, HELP["spont"])
        add_help_row(self.exp_form, "Forced", self.exp_forced, HELP["forced"])
        add_help_row(self.exp_form, "T_obs (s)", self.exp_tobs, HELP["tobs"])
        self._forcing_anchor = QLabel("(build config to list drive params)")
        self.exp_form.addRow(self._forcing_anchor)
        self.infer_stack.addWidget(exp_w)
        # page 2: experimental, chi(omega) -- one passive recording + K single-tone forced recordings
        chi_w = QWidget(); self.chi_form = make_form(chi_w)
        self.chi_spont = PathField()
        self.chi_tobs = FloatField(T_MIN_EXP_S)
        self.chi_f0_si = FloatField(1.0)
        self._chi_forced_fields = []
        add_help_row(self.chi_form, "Passive", self.chi_spont, HELP["chi_passive"])
        add_help_row(self.chi_form, "T_obs (s)", self.chi_tobs, HELP["tobs"])
        add_help_row(self.chi_form, "Drive F₀ (N)", self.chi_f0_si, HELP["chi_f0_si"])
        # The probe table. Rows live in their OWN container rather than as form rows, so
        # adding and removing one is a local layout edit that cannot disturb the fields above it.
        self._chi_probe_host = QWidget()
        self._chi_probe_layout = QVBoxLayout(self._chi_probe_host)
        self._chi_probe_layout.setContentsMargins(0, 0, 0, 0)
        add_help_row(self.chi_form, "Forced probes", self._chi_probe_host, HELP["chi_forced"])
        self._chi_buttons = chi_btns = QWidget(); chi_btns_l = QHBoxLayout(chi_btns)
        chi_btns_l.setContentsMargins(0, 0, 0, 0)
        self.btn_chi_add = QPushButton("+ Add probe")
        self.btn_chi_add.clicked.connect(lambda: self._add_chi_probe())
        self.btn_chi_plan = QPushButton("Plan probes…")
        self.btn_chi_plan.setToolTip("Measure Ω₀ from the passive recording and report what is in "
                                     "band, and how long each probe must be recorded.")
        self.btn_chi_plan.clicked.connect(self._plan_chi_probes)
        chi_btns_l.addWidget(self.btn_chi_add)
        chi_btns_l.addWidget(self.btn_chi_plan)
        chi_btns_l.addStretch(1)
        self.chi_form.addRow(chi_btns)
        self._chi_anchor = QLabel("(build a χ config to enable the probe table)")
        self.chi_form.addRow(self._chi_anchor)
        self.infer_stack.addWidget(chi_w)
        v.addWidget(self.infer_stack)

        self.btn_infer = QPushButton("Run inference")
        self.btn_infer.setProperty("accent", True)        # primary CTA (Fluent accent)
        self.btn_infer.clicked.connect(self._infer)
        v.addWidget(self.btn_infer)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _on_cell_source_changed(self):
        """Entering direct-entry seeds the grid from the picked cell -- same rule as the bounds grid:
        the parameter schema belongs to the model, only the numbers are the user's."""
        if not self.cell_source.is_direct():
            self._on_cell_changed()                  # back to file mode: re-validate the picked file
            return
        path = self.cell_picker.selected_path()
        if not path:
            self.log_pane.append_line("Select a cell file first — direct entry starts from it.",
                                      "warning")
            self.cell_source.set_direct(False)
            return
        try:
            inits, params, rescale, forcing = file_manager.parse_values_file(path)
        except Exception as e:                       # noqa: BLE001
            self._config_error(e)
            self.cell_source.set_direct(False)
            return
        self.values_grid.load(inits, params, rescale, forcing)
        self._cell_problems = []                     # hand-entered values are validated at Run instead
        self.refresh_local_gates()

    def _on_cell_changed(self):
        """Validate the picked cell against the bounds file ON THE GUI THREAD, the moment it is chosen.

        The check already existed inside inject_ground_truth, but it only fired inside the worker -- so a
        mismatched cell surfaced as a mid-run error dialog after the user had already committed. Here it
        is immediate, and the Run button stays disabled until a usable cell is selected."""
        self._cell_problems = []
        cfg, path = self.session.cfg, self.cell_picker.selected_path()
        if cfg is not None and path:
            try:
                self._cell_problems = cli.validate_gt_file(cfg, path)
            except Exception:                          # noqa: BLE001 -- a pre-flight check must never
                self._cell_problems = []               # break the panel; the worker still validates
            if self._cell_problems:
                self.log_pane.append_line(
                    "This cell does not fit the bounds file used to build the config: "
                    + "; ".join(self._cell_problems) + ". Choose another cell (or rebuild the config "
                    "against matching bounds).", "warning")
        self.refresh_local_gates()

    def on_config_built(self, cfg):
        _CellPreviewMixin.on_config_built(self, cfg)
        self._on_cell_changed()                       # re-validate against the newly built config
        self._rebuild_forcing_fields(cfg)
        # A no-forcing (passive) model has no forced recording and no drive params: hide the forced row.
        # _rebuild_forcing_fields already produces no forcing fields for an empty force_params_dict.
        self.exp_form.setRowVisible(self.exp_forced, cfg.has_forcing)
        self._rebuild_chi_fields(cfg)
        self._sync_infer_page()

    def _sync_infer_page(self):
        """Experimental mode shows the χ page when the config is χ-mode. Load-bearing: a χ observation
        needs K forced recordings, and falling through to the ordinary experimental branch would build a
        silently wrong-width conditioning vector rather than failing."""
        if self.infer_mode.currentIndex() == 0:
            self.infer_stack.setCurrentIndex(0)
            return
        cfg = self.session.cfg
        mode = cfg.observation_mode if cfg is not None else "forced"
        self.infer_stack.setCurrentIndex(2 if mode == "chi" else 1)

    def _add_chi_probe(self, freq_hz: float = 0.0):
        """Append one probe row, up to the posterior's slot capacity."""
        cfg = self.session.cfg
        cap = cfg.chi_k_pad if cfg is not None and cfg.chi_mode else config.CHI_K_PAD
        if len(self._chi_forced_fields) >= cap:
            self.log_pane.append_line(
                f"This posterior reserves {cap} probe slots (CHI_K_PAD), which is frozen into the "
                f"trained artifact — it cannot take more probes than that.", "warning")
            return None
        row = _ChiProbeRow(self._remove_chi_probe, freq_hz)
        self._chi_forced_fields.append(row)
        self._chi_probe_layout.addWidget(row)
        return row

    def _remove_chi_probe(self, row):
        if row not in self._chi_forced_fields:
            return
        self._chi_forced_fields.remove(row)
        self._chi_probe_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def _rebuild_chi_fields(self, cfg):
        """Enable/disable the probe table for the built config, PRESERVING every existing row.

        Rows are never destroyed on a rebuild, and that is deliberate. They carry hand-typed drive
        frequencies and browsed recording paths -- neither of which this method could regenerate, and
        both of which represent a bench session that already happened. Rebuilding the config (to fix
        a bounds file, say) must not silently discard them. Contrast _rebuild_forcing_fields, whose
        rows ARE derivable from the config's forcing schema and so are rebuilt freely.

        Count and placement are both free: the encoder is permutation-invariant and
        carries each probe's frequency explicitly, so the table seeds a suggested number of rows and
        then gets out of the way. `cfg.chi_n_freqs` is a suggestion, NOT a requirement -- the core
        accepts 1..chi_k_pad probes at whatever frequencies the experiment achieved.
        """
        if self._chi_anchor is not None:
            self.chi_form.removeRow(self._chi_anchor)
            self._chi_anchor = None
        on = bool(cfg.chi_mode)
        self.btn_chi_add.setEnabled(on)
        self.btn_chi_plan.setEnabled(on)
        # setRowVisible, not setVisible: hiding the widget alone strands its form LABEL, so a
        # non-chi config would show a "Forced probes" caption with nothing under it.
        self.chi_form.setRowVisible(self._chi_probe_host, on)
        self.chi_form.setRowVisible(self._chi_buttons, on)
        if not on:
            self._chi_anchor = QLabel("(build a χ config to enable the probe table)")
            self.chi_form.addRow(self._chi_anchor)
            return
        # Seed only when EMPTY -- never top up, never trim. A user who deleted rows meant it.
        if not self._chi_forced_fields:
            for _ in range(max(1, min(int(cfg.chi_n_freqs), cfg.chi_k_pad))):
                self._add_chi_probe()

    def _plan_chi_probes(self):
        """Backlog C-3: say what is in band for THIS cell, and how long each probe must be recorded.

        Every predicate comes from chi.probe_verdict, the same function build_experiment_obs_chi
        refuses and masks on -- so this cannot tell the user one thing and the run another. That is
        the point of the exercise: these answers were previously only discoverable by running the
        inference, i.e. after the bench session rather than before it.

        The band is RELATIVE to the cell's own Ω₀, so nothing useful can be said until a passive
        recording exists. Measuring it needs one load and one FFT, which is why this is a button
        rather than something recomputed on every keystroke.
        """
        cfg = self.session.cfg
        if cfg is None or not cfg.chi_mode:
            return
        path = self.chi_spont.value()
        if not path:
            self.log_pane.append_line(
                "Select the passive recording first — Ω₀ is measured from it, and the χ band is "
                "defined relative to Ω₀, so there is nothing to plan without it.", "warning")
            return
        try:
            from core.SBI import chi as _chi
            x = file_manager.load_experimental_data(path, dtype=cfg.hw.dtype)
            f_peak = float(_chi.peak_freq(x.unsqueeze(0), cfg.dt_exp))
        except Exception as e:                                  # noqa: BLE001 -- a planner must never
            self.log_pane.append_line(f"Could not measure Ω₀ from {path}: {e}", "error")   # break the panel
            return
        hz = cfg.get_unit_conversion_factor("s")
        lo_hz, hi_hz = _chi.band_hz(cfg, f_peak)
        n_samp = max(1, int(round(self.chi_tobs.value() * hz / cfg.dt_exp)))
        self.log_pane.append_line(
            f"Ω₀ = {f_peak * hz:.4g} Hz for this recording. In band for this cell: "
            f"{lo_hz:.3g}–{hi_hz:.3g} Hz "
            f"({cfg.chi_freq_bounds[0]:g}–{cfg.chi_freq_bounds[1]:g}×Ω₀).")
        self.log_pane.append_line(
            f"At the band's low edge a probe needs ≥ {config.CHI_MIN_CYCLES / lo_hz:.3g} s to clear "
            f"the {config.CHI_MIN_CYCLES:g}-cycle floor; above "
            f"{cfg.chi_max_cycles / hi_hz:.3g} s the high edge is truncated to the "
            f"{cfg.chi_max_cycles:g}-cycle ceiling (which is fine — only the tail is dropped).")
        # Fill blank frequency boxes with the nominal in-band grid so the table is usable immediately.
        # Only BLANK ones: a typed frequency is a record of what the bench actually did.
        blanks = [r for r in self._chi_forced_fields if r.freq.value() <= 0]
        if blanks:
            grid = _chi.chi_multipliers(n_freqs=len(blanks), bounds=cfg.chi_freq_bounds).tolist()
            for row, mult in zip(blanks, grid):
                row.freq.setText(f"{mult * f_peak * hz:.4g}")
            self.log_pane.append_line(
                f"Filled {len(blanks)} blank frequency box(es) with a nominal log-spaced in-band "
                f"grid. These are SUGGESTIONS — replace each with the frequency you actually drove "
                f"at, because a lock-in decays like a sinc and a small mismatch destroys it.")
        # Now report each row's verdict against the T_obs entered.
        for i, row in enumerate(self._chi_forced_fields):
            f = row.freq.value()
            if not (math.isfinite(f) and f > 0):
                self.log_pane.append_line(f"  probe {i + 1}: no frequency entered.", "warning")
                continue
            v = _chi.probe_verdict(cfg, f_peak, f, n_samp)
            if v.action == "use":
                self.log_pane.append_line(
                    f"  probe {i + 1}: {f:g} Hz — OK, {v.cycles:.1f} drive cycles at "
                    f"T_obs = {self.chi_tobs.value():g} s.")
            else:
                self.log_pane.append_line(f"  probe {i + 1}: {f:g} Hz — {v.action.upper()}: "
                                          f"{v.reason}.", "warning" if v.action != "refuse" else "error")

    def _rebuild_forcing_fields(self, cfg):
        for fld in self._forcing_fields.values():
            self.exp_form.removeRow(fld)
        self._forcing_fields = {}
        if self._forcing_anchor is not None:
            self.exp_form.removeRow(self._forcing_anchor)
            self._forcing_anchor = None
        for name in cfg.force_params_dict:
            unit = cli.INFERENCE_PROMPT_UNITS.get(name, "")
            fld = FloatField(0.0)
            self._forcing_fields[name] = fld
            add_help_row(self.exp_form, labels.gui_forcing_label(name, unit), fld, HELP["forcing"])

    def _infer(self):
        cfg, post = self.session.cfg, self.session.posterior
        if post is None:
            return
        if self.infer_mode.currentIndex() == 0:      # simulated
            gt_dicts, cell = None, None
            if self.cell_source.is_direct():
                problems = self.values_grid.problems()
                if problems:
                    self.log_pane.append_line("Fix the values first: " + "; ".join(problems), "warning")
                    return
                gt_dicts = self.values_grid.to_dicts()
            else:
                cell = self.cell_picker.selected_path()
                if not cell:
                    self.log_pane.append_line("Select a cell file first.", "warning")
                    return
                if self._cell_problems:
                    self.log_pane.append_line(
                        "Fix the cell selection first: " + "; ".join(self._cell_problems), "warning")
                    return
            self.dispatch(_run_simulated_inference, cfg, post, cell, self.sim_tobs.value(),
                          gt_dicts=gt_dicts,
                          inferred_prior=self.session.inf_prior, force_prior=self.session.force_prior,
                          provide_fig_sink=True)
        elif cfg.observation_mode == "chi":          # experimental, χ(ω): 1 passive + K forced
            if not self.chi_spont.value():
                self.log_pane.append_line("Select the passive recording first — it sets Ω₀.",
                                          "warning")
                return
            if not self._chi_forced_fields:
                self.log_pane.append_line(
                    "Add at least one forced probe. χ mode conditions on a passive recording plus "
                    "any number of single-tone forced ones, but zero probes is a spontaneous "
                    "observation wearing a χ conditioning vector.", "warning")
                return
            problems = [p for i, r in enumerate(self._chi_forced_fields) for p in r.problems(i)]
            if problems:
                self.log_pane.append_line("Fix the probe table first: " + "; ".join(problems),
                                          "warning")
                return
            # (recording, drive frequency in Hz) PAIRS, never a bare path list. The core locks in at
            # the frequency it is TOLD, rather than assuming mult_k * Omega_0 -- the frequencies a
            # bench achieves are not exactly that, and a lock-in at the wrong frequency decays like a
            # sinc. Pairs come straight off each row widget, so they cannot be mismatched by an
            # add/remove in the middle of the table.
            pairs = [r.pair() for r in self._chi_forced_fields]
            self.dispatch(_run_experimental_inference_chi, cfg, post, self.chi_spont.value(), pairs,
                          self.chi_tobs.value(), self.chi_f0_si.value(), provide_fig_sink=True)
        elif not cfg.has_forcing:                    # experimental, passive (no drive)
            if not self.exp_spont.value():
                self.log_pane.append_line("Select a passive recording first.", "warning")
                return
            self.dispatch(_run_experimental_inference_spontaneous, cfg, post,
                          self.exp_spont.value(), self.exp_tobs.value(), provide_fig_sink=True)
        else:                                        # experimental, driven
            forcing_si = {name: fld.value() for name, fld in self._forcing_fields.items()}
            self.dispatch(_run_experimental_inference, cfg, post, self.exp_spont.value(),
                          self.exp_forced.value(), self.exp_tobs.value(), forcing_si, provide_fig_sink=True)

    def refresh_local_gates(self):
        simulated = self.infer_mode.currentIndex() == 0
        # _cell_problems only describes the PICKED FILE; hand-entered values are validated at Run.
        blocked = (simulated and not self.cell_source.is_direct()
                   and bool(getattr(self, "_cell_problems", [])))
        self.btn_infer.setEnabled(self.session.posterior is not None and not blocked)
        self.btn_infer.setToolTip(
            "The selected cell does not fit the bounds file used to build the config." if blocked else "")

    def save_settings(self, qs):
        qs.beginGroup("inference_infer")
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("infer_mode", self.infer_mode.currentIndex())
        settings.save_field(qs, "sim_tobs", self.sim_tobs)
        settings.save_field(qs, "exp_tobs", self.exp_tobs)
        settings.save_field(qs, "exp_spont", self.exp_spont)
        settings.save_field(qs, "exp_forced", self.exp_forced)
        settings.save_field(qs, "chi_spont", self.chi_spont)
        settings.save_field(qs, "chi_tobs", self.chi_tobs)
        settings.save_field(qs, "chi_f0_si", self.chi_f0_si)
        qs.endGroup()
        # The forcing fields and the per-frequency χ forced-recording fields don't exist until
        # "Build config" runs (and their COUNT depends on K), so they are not persisted.

    def restore_settings(self, qs):
        qs.beginGroup("inference_infer")
        self._saved_cell_key = settings.get_str(qs, "cell")     # re-applied in on_config_built
        try:
            self.infer_mode.setCurrentIndex(int(settings.get_str(qs, "infer_mode", "0")))
        except ValueError:
            pass
        settings.restore_field(qs, "sim_tobs", self.sim_tobs)
        settings.restore_field(qs, "exp_tobs", self.exp_tobs)
        settings.restore_field(qs, "exp_spont", self.exp_spont)
        settings.restore_field(qs, "exp_forced", self.exp_forced)
        settings.restore_field(qs, "chi_spont", self.chi_spont)
        settings.restore_field(qs, "chi_tobs", self.chi_tobs)
        settings.restore_field(qs, "chi_f0_si", self.chi_f0_si)
        qs.endGroup()
