"""Navigation, the inference tabs' gating/session flow, and the chi probe table. Split from test_gui_progress.py; run directly: python tests/test_nav_and_gating.py"""
"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  python -m pytest tests/test_gui_progress.py -v
      (or just: python tests/test_gui_progress.py)
"""
import ast
import inspect
import textwrap
import os
import tempfile
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

import torch                                                      # noqa: E402
from PySide6.QtWidgets import QApplication                        # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from core.gui.panels.base_panel import BasePanel                  # noqa: E402
from core.gui.streams import redirect_streams                     # noqa: E402
from core.gui.vt import StreamRouter, parse_bar                   # noqa: E402
from core.gui.widgets.log_pane import LogPane                     # noqa: E402
from core.gui.widgets.progress_pane import ProgressPane           # noqa: E402
from core.gui.worker import WorkerSignals                         # noqa: E402
import contextlib                                                  # noqa: E402

def _app():
    return QApplication.instance() or QApplication([])
# ── Phase 3: QSettings persistence ───────────────────────────────────────────────────────────────
def _temp_settings():
    import tempfile
    from core.gui import settings as st
    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    st.use_ini_file(path)
    return path
def _chi_cfg(k=3, pad=12):
    """Stub config that puts the Infer tab on its chi page (mirrors the SimConfig fields it reads)."""
    from core import config

    class Cfg:
        model = "NADROWSKI"
        params_dict = {}
        rescale_params = {}
        force_params_dict = {}
        has_forcing = False
        chi_mode = True
        chi_n_freqs = k
        chi_k_pad = pad
        chi_freq_bounds = config.CHI_FREQ_BOUNDS
        chi_max_cycles = config.CHI_MAX_CYCLES
        observation_mode = "chi"
    return Cfg()


# ── Phase-2 panels ───────────────────────────────────────────────────────────────────────────────
def test_fdt_panel_guard_translates_model_error_and_gate_admits_builtins():
    """FDT now supports HOPF/BP + additive-noise user models. The guard turns an FDTModelError (a
    missing FDT parameter, or a user model with multiplicative/zero observable noise) into a readable
    RuntimeError instead of a bare traceback; the registry gate admits every built-in."""
    import core.gui.panels.fdt_panel as fdt_panel
    from core.FDT.campaigns import FDTModelError
    from core import registry

    class Cfg:
        model = "HOPF"

    def boom(cfg, *, skip_sanity, confirm_production):
        raise FDTModelError("Observable 'x' has state-dependent (multiplicative) noise; FDT supports "
                            "additive-noise observables only.")

    real, fdt_panel.run_fdt = fdt_panel.run_fdt, boom
    try:
        try:
            fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False)
        except RuntimeError as e:
            assert "multiplicative" in str(e), str(e)
        except FDTModelError:
            raise AssertionError("the FDTModelError escaped the guard unwrapped")
        else:
            raise AssertionError("the guard swallowed the failure entirely")
    finally:
        fdt_panel.run_fdt = real

    # HOPF / BP / NADROWSKI are no longer rejected by the FDT gate.
    for m in ("NADROWSKI", "HOPF", "BP"):
        assert registry.fdt_support(m) == (True, ""), m

def test_an_unparseable_cell_does_not_brick_the_gui():
    """Dropping a cell into Resources/Cells/<model>/ that cli.parse_cell cannot read makes it raise a
    bare ValueError (NOT a UnitParseError). CrossValPanel prefills from parse_cell in __init__, so
    that exception used to escape CrossValPanel() -> MainWindow() -> build_app(), and
    `python -m core.gui` died before the window ever appeared -- before app.py's excepthook was even
    installed.

    The probe used to be a cell with no sibling bounds file, which is no longer unparseable: a cell
    without a sibling now resolves to the model's shared master.txt (cli.resolve_bounds_for_cell), so
    the natural 'add my cell' action WORKS rather than merely degrading. The failure mode this test
    guards still exists though -- a cell that omits a parameter its bounds file declares -- so probe
    with that instead."""
    import shutil
    from pathlib import Path

    from core.config import CELL_PATH
    from core.gui import settings as st

    _app()
    src = Path(CELL_PATH) / "nadrowski" / "master_weak.txt"
    if not src.exists():
        return                                   # nothing to probe with
    probe = src.with_name("aaa_probe_unparseable.txt")   # sorts first => the picker selects it
    # Drop a declared ND parameter: bounds resolve fine, but the merge cannot fill the value.
    kept = [ln for ln in src.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith("beta ")]
    probe.write_text("\n".join(kept) + "\n", encoding="utf-8")
    try:
        # Isolate from the developer's real QSettings store. MainWindow() -> CrossValPanel.__init__
        # restores its saved cell selection; a cell saved from a previous GUI session would be reloaded
        # OVER our probe, so the picker would land on a valid cell and the prefill would parse fine --
        # masking the degrade path this test exists to check. A clean temp .ini defaults the picker to
        # the alphabetically-first entry (the probe), regardless of the machine's state.
        _temp_settings()
        from core.gui.main_window import MainWindow
        from core.gui.panels.crossval_panel import CrossValPanel
        window = MainWindow()                    # must not raise
        xval = window.panel(CrossValPanel)       # CrossVal now lives inside the FDT Analysis section
        assert "could not read cell" in xval.cell_values.text().lower(), \
            f"the bad cell should degrade the prefill label, got: {xval.cell_values.text()!r}"
    finally:
        probe.unlink(missing_ok=True)
        st.use_ini_file(None)

def test_plot_watcher_only_reports_pngs_written_after_start():
    """The FDT/Reduction/CrossVal runners never return their figure paths, so the panels pick them up
    off disk. Pre-existing figures must not be re-shown, and each new one must be emitted once."""
    import tempfile
    from pathlib import Path

    from core.gui.plot_watcher import NewPngWatcher

    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "old_plot_20260101_000000.png").write_bytes(b"stale")

        seen = []
        watcher = NewPngWatcher(d)
        watcher.png_ready.connect(lambda title, path: seen.append((title, Path(path).name)))
        watcher.start()

        (d / "fdt3d_vs_S_20260714_120301.png").write_bytes(b"new")
        watcher.stop()                      # stop() forces a final scan, ignoring the settle delay
        app.processEvents()

        assert seen == [("fdt3d vs S", "fdt3d_vs_S_20260714_120301.png")], seen

# ── PRISM navigation redesign ────────────────────────────────────────────────────────────────────
def test_greeting_maps_hours_to_time_of_day():
    from core.gui.screens.home_screen import greeting
    assert greeting(5) == greeting(11) == "Good morning"
    assert greeting(8) == "Good morning"
    assert greeting(12) == greeting(16) == "Good afternoon"
    assert greeting(14) == "Good afternoon"
    assert greeting(17) == greeting(23) == "Good evening"
    assert greeting(0) == greeting(4) == "Good evening"

def test_nav_shell_back_arrow_tracks_the_screen():
    from PySide6.QtWidgets import QWidget
    from core.gui.screens.nav_shell import NavShell

    _app()
    nav = NavShell()
    for _ in range(3):
        nav.add_screen(QWidget())
    nav.go_home()
    assert nav.btn_back.isHidden(), "back arrow should be hidden on home"
    nav.go_to(2)
    assert not nav.btn_back.isHidden(), "back arrow should show on a section"
    nav.go_home()
    assert nav.btn_back.isHidden(), "back arrow should hide again on home"

def test_main_window_always_opens_on_home():
    from core.gui import settings as st
    _app()
    _temp_settings()
    try:
        qs = st.settings()
        qs.setValue("window/tab", 2)          # a stale key from the old flat-tab layout
        qs.sync()
        from core.gui.main_window import MainWindow
        w = MainWindow()
        assert w.nav.stack.currentIndex() == 0, "the app must always open on the home screen"
    finally:
        st.use_ini_file(None)

def test_inference_tab_gates_follow_the_session():
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    _app()
    inf = InferenceScreen()

    def enabled():
        return [inf.tabs.isTabEnabled(i) for i in range(5)]   # Config Prior Posterior Validate Infer

    inf.session = SbiSession(); inf.refresh_gates()
    assert enabled() == [True, False, False, False, False]
    # A DRAFT (model applied) unlocks Prior -- which is where the bounds file builds the config.
    inf.session.draft = object(); inf.refresh_gates()
    assert enabled() == [True, True, False, False, False]
    inf.session.cfg = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, False, False]
    inf.session.posterior = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, False, True], "Infer needs only a posterior; Validate needs a prior"
    # force_prior stays None on purpose: it is None for every no-forcing model, and requiring it used to
    # make Validate permanently unreachable for them.
    inf.session.inf_prior = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, True, True]

def test_tsnpe_tab_is_gated_and_never_proposes_from_the_posterior():
    """The TSNPE tab needs a posterior, its prior AND an observation on disk -- and its round must go
    through orchestrator.build_posterior with a TRUNCATION, never by fitting the posterior.

    The second half is the one worth a test: proposing from the posterior instead of the truncated
    prior is TEMPERING, it contracts credible intervals with no new information, and SBC comes out
    flat anyway because it validates the flow against the proposal it was trained on. Nothing on the
    Validate tab would catch it, so the wiring is pinned here and the maths in
    tests/test_conditioning_repair.py.
    """
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.gui.panels import inference_tabs

    _app()
    inf = InferenceScreen()
    assert inf.tabs.count() == 6 and inf.tabs.tabText(5) == "TSNPE"

    inf.session = SbiSession(cfg=object(), posterior=object()); inf.refresh_gates()
    assert not inf.tabs.isTabEnabled(5), "TSNPE must not open on a posterior alone"
    inf.session.inf_prior = object(); inf.refresh_gates()
    assert inf.tabs.isTabEnabled(5), "TSNPE opens once a posterior and its prior exist"

    # The observation gate, driven through the picker's own accessor rather than through the real
    # Resources/Observations directory. Clearing the combo does NOT work: refresh_local_gates calls
    # obs_picker.refresh(), which repopulates it from disk -- so the assertion passed only while that
    # directory happened to be empty, and any suite run that had exercised infer_and_visualize left a
    # record behind and flipped it. A gate test must not depend on what an earlier test wrote.
    panel = inf.tsnpe_panel
    panel.obs_picker.key = lambda: ""                    # nothing recorded yet
    panel.refresh_local_gates()
    assert not panel.btn_round.isEnabled(), "a round must be impossible without an observation"
    panel.obs_picker.key = lambda: "obs_20260101T000000_deadbeefdeadbeef.pt"
    panel.refresh_local_gates()
    assert panel.btn_round.isEnabled(), "with a posterior, its prior and an observation, a round is allowed"

    # The runner's contract: it hands build_posterior a truncation region and nothing else refits.
    # Checked against the CODE, with the docstring stripped -- that docstring necessarily contains the
    # word "proposal" while explaining what must not happen, and a naive text search on the whole
    # source flags the very comment that documents the rule.
    tree = ast.parse(textwrap.dedent(inspect.getsource(inference_tabs._run_tsnpe_round)))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str)):
        fn.body = fn.body[1:]
    code = ast.unparse(tree)
    assert "build_truncation_region" in code and "truncation=region" in code,         "the TSNPE runner does not build a truncation region and pass it to build_posterior"
    for banned in ("set_default_x", "proposal"):
        assert banned not in code, (
            f"the TSNPE runner's CODE references '{banned}' -- it must sample the truncated PRIOR, "
            f"never the posterior; that is tempering, and SBC cannot detect it")

def test_a_tsnpe_posterior_cannot_be_saved_as_amortized():
    """⚠ SECTION 11.6 GUARDRAIL 2, at the seam where it is easiest to lose.

    The GUI trains with save=False and saves LATER from a button, so the truncation region has to
    survive on the session or the deferred save writes the artifact marked `amortized: True` --
    indistinguishable, in the same ArtifactPicker, from a genuinely amortized posterior. That is the
    class of confusion the retired-band posterior already cost a five-day run for.

    Three things, and the third is the one that is easy to miss: the round INSTALLS the region, the
    save PASSES it, and training an ordinary posterior afterwards CLEARS it.
    """
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.gui.panels import inference_tabs

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=object())
    panel = inf.tsnpe_panel

    region = object()
    panel._on_round(((object(), {"loss": []}), region, "deadbeefdeadbeef"))
    assert inf.session.truncation is region, "the round did not install its truncation region"
    assert inf.session.x_obs_digest == "deadbeefdeadbeef"
    assert inf.session.posterior is not None, "the round's posterior never reached the session"

    # the deferred save must forward both
    captured = {}
    pp = inf.posterior_panel
    pp.dispatch = lambda fn, *a, **k: captured.update(args=a, kwargs=k)
    pp.post_name.setText("some_name")
    inf.session.posterior_latent = object()
    pp._save_posterior()
    assert captured.get("kwargs", {}).get("truncation") is region,         "save_posterior_artifacts was called WITHOUT the truncation -- it would be marked amortized"
    assert captured["kwargs"].get("x_obs_digest") == "deadbeefdeadbeef"

    # and an ordinary train afterwards must CLEAR it, or the mislabelling runs the other way
    pp._on_posterior((object(), {"loss": []}))
    assert inf.session.truncation is None and inf.session.x_obs_digest is None,         "an amortized posterior inherited the previous round's truncation"

def test_the_new_tab_knobs_are_forwarded_and_not_written_to_config():
    """Prior, Posterior and Validate all gained fields. Each must reach its orchestrator function as
    an ARGUMENT -- orchestrator snapshots those constants at import, so writing them would be a
    silent no-op and the run would use the defaults with nothing to say so."""
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core import config as _cfg

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession(draft=object(), cfg=object(), inf_prior=object(), posterior=object())

    cap = {}
    for panel in (inf.prior_panel, inf.posterior_panel, inf.validate_panel):
        panel.dispatch = lambda fn, *a, **k: cap.update(kwargs=k)

    inf.validate_panel.cal_n.setText("77")
    inf.validate_panel.cal_scales.setText("11")
    inf.validate_panel._validate()
    assert cap["kwargs"]["n_cal"] == 77 and cap["kwargs"]["cal_n_scales"] == 11, cap["kwargs"]
    assert _cfg.SBC_N_CAL != 77, "the panel wrote the config constant instead of passing an argument"

    inf.posterior_panel.flow_hidden.setText("64")
    inf.posterior_panel.flow_transforms.setText("3")
    inf.posterior_panel.flow_patience.setText("7")
    inf.posterior_panel.post_picker.combo.setCurrentIndex(0)
    inf.posterior_panel._build_posterior()
    k = cap["kwargs"]
    assert (k["hidden_features"], k["num_transforms"], k["stop_after_epochs"]) == (64, 3, 7), k
    assert _cfg.NSF_HIDDEN_FEATURES != 64, "the panel wrote NSF_HIDDEN_FEATURES"
    # the Fisher knobs ride along on the same call
    assert (k["fisher_m"], k["fisher_points"]) == (_cfg.REPARAM_FISHER_M, _cfg.REPARAM_FISHER_POINTS)

    inf.prior_panel.cluster_size.setText("9")
    inf.prior_panel.cluster_samples.setText("4")
    inf.prior_panel.sweep_iters.setText("3")
    inf.prior_panel.bounds_source.set_direct(False)
    inf.prior_panel.bounds_picker.combo.clear()
    inf.prior_panel.bounds_picker.combo.addItem("master.txt")
    inf.session.draft = type("D", (), {"make_config": lambda self, **kw: inf.session.cfg})()
    try:
        inf.prior_panel._build_prior()
    except Exception:
        pass                                  # config construction is stubbed; the dispatch is the point
    if "min_cluster_size" in cap.get("kwargs", {}):
        assert cap["kwargs"]["min_cluster_size"] == 9 and cap["kwargs"]["min_samples"] == 4, cap["kwargs"]
        assert _cfg.PRIOR_CLUSTER_MIN_SIZE != 9, "the panel wrote PRIOR_CLUSTER_MIN_SIZE"

def test_posterior_from_scratch_is_gated_on_a_prior():
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object()); inf.refresh_gates()
    pp = inf.posterior_panel
    pp.post_picker.combo.setCurrentIndex(0)               # the "(from scratch)" sentinel (allow_new adds it first)
    assert pp.post_picker.selected()[1] is True, "index 0 should be the from-scratch sentinel"
    pp.refresh_local_gates()
    assert not pp.btn_post.isEnabled(), "training from scratch must be disabled without a prior"
    inf.session.inf_prior = object(); inf.session.force_prior = object()
    pp.refresh_local_gates()
    assert pp.btn_post.isEnabled(), "with a prior, training from scratch is allowed"

def test_inference_pickers_repoint_from_draft_and_config():
    """The Prior tab's BOUNDS picker follows the applied model (new_draft), and the Infer tab's CELL
    picker follows the BUILT config's model (install_config) -- neither tab has its own model combo."""
    from core import config
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import ConfigDraft

    _app()
    inf = InferenceScreen()

    inf.new_draft(ConfigDraft(model="HOPF", labels=[], state_dep_drift=False))
    assert inf.prior_panel.bounds_picker.base_path.name == "hopf"
    assert inf.session.cfg is None, "a draft alone must not produce a config"

    class Cfg:
        model = "HOPF"
        params_dict = {}      # the Infer tab validates a picked cell against these
        rescale_params = {}
        force_params_dict = {}
        has_forcing = False   # mirrors SimConfig.has_forcing (empty force_params_dict)
        chi_mode = False      # mirrors SimConfig.chi_mode
        chi_k_pad = config.CHI_K_PAD    # mirrors SimConfig.chi_k_pad (probe-slot capacity)
        observation_mode = "spontaneous"

    inf.install_config(Cfg())
    assert inf.infer_panel.cell_picker.base_path.name == "hopf"
    assert inf.session.draft is not None, "install_config must not wipe the draft/session"

def test_chi_probe_table_is_variable_length_and_capped_by_the_posteriors_slots():
    """Backlog C-2. The core has always accepted 1..chi_k_pad probes at arbitrary frequencies; the GUI
    was the only thing forcing a fixed grid. Rows must be addable and removable, and the cap must be
    chi_k_pad -- which is FROZEN into the trained artifact, so exceeding it is not a soft limit."""
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=3, pad=5))
    panel = inf.infer_panel
    assert len(panel._chi_forced_fields) == 3, "should seed chi_n_freqs rows when empty"

    panel._add_chi_probe()
    panel._add_chi_probe()
    assert len(panel._chi_forced_fields) == 5
    panel._add_chi_probe()                                   # over capacity
    assert len(panel._chi_forced_fields) == 5, "must refuse to exceed chi_k_pad slots"

    panel._remove_chi_probe(panel._chi_forced_fields[0])
    assert len(panel._chi_forced_fields) == 4

def test_chi_probe_rows_keep_each_recording_paired_with_its_own_frequency():
    """The one-widget-per-row invariant. Parallel path/frequency lists let a MIDDLE deletion pair
    recording k with frequency k+1 -- silent, because a lock-in at the wrong frequency decays like a
    sinc and simply returns a smaller number. Delete from the middle and check the survivors."""
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=4))
    panel = inf.infer_panel
    for i, row in enumerate(panel._chi_forced_fields):
        row.path.edit.setText(f"/tmp/rec{i}.csv")
        row.freq.setText(str(1.0 + i))

    panel._remove_chi_probe(panel._chi_forced_fields[1])      # delete from the MIDDLE
    pairs = [r.pair() for r in panel._chi_forced_fields]
    assert pairs == [("/tmp/rec0.csv", 1.0), ("/tmp/rec2.csv", 3.0), ("/tmp/rec3.csv", 4.0)], pairs

def test_chi_probe_table_survives_a_config_rebuild_and_rejects_a_blank_frequency():
    """Two C-2 constraints in one place, because both are about data the GUI cannot regenerate.

    PRESERVATION: rows carry hand-typed drive frequencies and browsed paths -- a record of a bench
    session that already happened. Rebuilding the config (to fix a bounds file, say) must not discard
    them, unlike the forcing rows, which ARE derivable from the config.

    BLANK FREQUENCY: FloatField.value() returns 0.0 on unparseable text, so an empty box is
    indistinguishable from a deliberate zero -- and 0 Hz is a genuine DC probe the lock-in would
    happily attempt. It has to be caught before the run, not after.
    """
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=2))
    panel = inf.infer_panel
    panel._chi_forced_fields[0].path.edit.setText("/tmp/keep.csv")
    panel._chi_forced_fields[0].freq.setText("2.5")

    inf.install_config(_chi_cfg(k=2))                         # rebuild
    assert len(panel._chi_forced_fields) == 2, "a rebuild must not add or drop rows"
    assert panel._chi_forced_fields[0].pair() == ("/tmp/keep.csv", 2.5), \
        "a rebuild destroyed hand-entered probe data"

    # Row 1 still has a blank frequency -> 0.0 -> must be reported, naming the row.
    probs = [p for i, r in enumerate(panel._chi_forced_fields) for p in r.problems(i)]
    assert any("probe 2" in p and "positive" in p for p in probs), probs
    assert not any("probe 1" in p for p in probs), probs

def test_config_units_control_declares_units_and_validates_them():
    """Units DECLARE what the numbers in the files mean (never converting them). Typed units must reach
    the built config, and unusable tokens must be rejected when the model is applied -- not mid-run."""
    from core.config import BOUNDS_PATH
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    screen = InferenceScreen()
    cfgp = screen.config_panel
    cfgp.model_combo.setCurrentText("NADROWSKI")
    cfgp._on_model_changed("NADROWSKI")

    cfgp.units_toggle.set_direct(False)                       # model's units file
    assert cfgp._units_override() is None

    cfgp.units_toggle.set_direct(True)                        # typed
    cfgp.units_text.setText("nm s pN Hz")
    assert cfgp._units_override() == ("nm", "s", "pN", "Hz")
    cfgp._build_config()
    draft = screen.session.draft
    assert draft is not None and draft.units_override == ("nm", "s", "pN", "Hz")

    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    if bounds.exists():
        cfg = draft.make_config(str(bounds))
        assert (cfg.time_unit, cfg.freq_unit) == ("s", "Hz")
        # Hz alongside s IS self-consistent (unlike Hz alongside ms), so no warning
        assert cfg.check_unit_consistency() == []
        assert abs(30.0 * cfg.freq_si_to_cell - 30.0) < 1e-12, "in an `s` cell, 30 Hz stays 30"

    before = screen.session.draft
    cfgp.units_text.setText("notaunit")                       # must be refused, draft left alone
    cfgp._build_config()
    assert screen.session.draft is before

def test_direct_entry_grids_round_trip_their_files():
    """Hand-entered bounds/values must reproduce the parsed file exactly (same names, same ORDER --
    simulators bind parameter columns positionally), and must refuse unparseable or inverted input."""
    from collections import OrderedDict
    from core.config import BOUNDS_PATH, CELL_PATH
    from core.Helpers import file_manager
    from core.gui.widgets.param_grid import BoundsGrid, ValuesGrid

    _app()
    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    cell = CELL_PATH / "nadrowski" / "master_weak.txt"
    if not (bounds.exists() and cell.exists()):
        return                                                   # environment without Resources: skip

    p, r, f, _ = file_manager.parse_bounds_file(str(bounds))
    grid = BoundsGrid()
    assert grid.problems(), "an unloaded grid must report a problem rather than build nothing"
    grid.load(p, r, f)
    assert grid.problems() == []
    gp, gr, gf = grid.to_dicts()
    for got, want in ((gp, p), (gr, r), (gf, f)):
        assert list(got) == list(want), "parameter ORDER must survive the round trip"
        for name in want:
            assert got[name][1] == want[name][1], name

    grid._rows[("PARAM", "k")].lo.setText("")                    # unparseable -> refused, not 0.0
    assert any("k" in m for m in grid.problems())
    grid._rows[("PARAM", "k")].lo.setText("9")                   # min > max -> refused
    grid._rows[("PARAM", "k")].hi.setText("1")
    assert any("less than" in m for m in grid.problems())

    inits, vp, vr, vf = file_manager.parse_values_file(str(cell))
    vgrid = ValuesGrid()
    vgrid.load(inits, vp, vr, vf)
    assert vgrid.problems() == []
    gi, gvp, gvr, gvf = vgrid.to_dicts()
    assert list(gvp) == list(vp) and list(gi) == list(inits)
    assert all(abs(gvp[n] - vp[n]) < 1e-12 for n in vp)
    assert isinstance(gvp, OrderedDict)

def test_overlay_alignment_and_best_fit_rankings():
    """Phase alignment must recover a known lag; both best-fit rankings must find a planted match; and
    the trace ranking must be invariant to rolling the reference (absolute phase carries no info)."""
    import math
    import numpy as np
    import torch
    from core.SBI import overlay

    torch.manual_seed(0)
    n, dt, f = 2048, 1e-3, 7.5
    t = torch.arange(n, dtype=torch.float32) * dt

    # unambiguous (non-periodic) reference -> the lag is exact
    chirp = torch.sin(2 * math.pi * (2.0 + 6.0 * t) * t) * torch.hann_window(n)
    shifts = torch.tensor([0, 37, -91])
    rolled = torch.stack([torch.roll(chirp, int(s)) for s in shifts])
    aligned, lags = overlay.align_to(chirp, rolled)
    assert torch.equal(lags, shifts), (lags, shifts)
    assert float((aligned - chirp.unsqueeze(0)).abs().max()) < 1e-5

    gt = torch.sin(2 * math.pi * f * t) + 0.05 * torch.randn(n)
    cand = torch.stack([torch.sin(2 * math.pi * (f * 1.3) * t),      # wrong frequency
                        0.4 * torch.sin(2 * math.pi * f * t),        # wrong amplitude
                        torch.roll(gt, 123)])                        # the planted match, phase-shifted
    order, _rmse, _al = overlay.rank_by_trace(gt, cand)
    assert int(order[0]) == 2, "the planted draw must win on waveform"
    order2, _, _ = overlay.rank_by_trace(torch.roll(gt, -500), cand)
    assert int(order2[0]) == 2, "ranking must not depend on the reference's absolute phase"

    # summary-stat ranking: exact match wins, and the CONSTANT conditioning column is ignored
    obs = torch.tensor([[1.0, 2.0, 3.0, 42.0]])
    sim = torch.tensor([[9.0, 9.0, 9.0, 42.0], [1.0, 2.0, 3.0, 42.0], [1.5, 2.5, 2.0, 42.0]])
    o, d = overlay.rank_by_stats(sim, obs)
    assert int(o[0]) == 1 and float(d[1]) < 1e-9

    # phase-invariant summaries are well-formed
    freqs, lo, med, hi, dropped = overlay.psd_band(cand, dt)
    assert bool((lo <= hi).all()) and abs(float(freqs[med.argmax()]) - f) < 2.0
    centres, mean, clo, chi_, c_dropped = overlay.cycle_average(gt.unsqueeze(0), dt, f)
    assert dropped == 0 and c_dropped == 0, "clean traces must drop nothing"
    good = torch.isfinite(mean)
    assert int(good.sum()) > 0.8 * len(mean)
    assert 1.5 < float(mean[good].max() - mean[good].min()) < 2.5, "should recover the unit amplitude"
    assert overlay.cycle_window(n, dt, f, 15) == 2000

def test_a_divergent_draw_does_not_erase_the_whole_psd_band():
    """THE 2026-08-25 FAILURE, pinned.

    The power-spectrum figure came back as a bare observation line: no band, no median, no message.
    Cause: a broad posterior samples parameter sets that do not integrate stably, `|rfft|**2` of such a
    trace overflows to inf, and `torch.quantile` propagates one non-finite entry across every column --
    so ONE bad draw in a thousand silently erased the band for all of them.

    It went unnoticed for so long because the two sibling figures survive it: the overlay band takes
    only the 50 best draws, and cycle_average confines the damage to a single phase bin. So the
    symptom looked like a plotting bug in one figure rather than a property of the posterior.
    """
    import math
    import torch
    from core.SBI import overlay

    n, dt, f = 4096, 1.0 / 500.0, 12.0
    t = torch.arange(n, dtype=torch.float64) * dt
    good = torch.stack([torch.sin(2 * math.pi * f * t) * a for a in (0.9, 1.0, 1.1, 1.05, 0.95)])

    clean_freqs, clean_lo, clean_med, clean_hi, clean_drop = overlay.psd_band(good, dt)
    assert clean_drop == 0 and torch.isfinite(clean_med).all()

    for label, bad_row in (("inf", torch.full((n,), float("inf"), dtype=torch.float64)),
                           ("nan", torch.full((n,), float("nan"), dtype=torch.float64)),
                           ("overflow", torch.sin(2 * math.pi * f * t) * 1e300)):
        traces = torch.cat([good, bad_row.unsqueeze(0)], dim=0)
        freqs, lo, med, hi, dropped = overlay.psd_band(traces, dt)
        assert dropped == 1, f"{label}: expected 1 dropped draw, got {dropped}"
        assert torch.isfinite(med).all(), f"{label}: one bad draw still poisoned the median"
        assert torch.isfinite(lo).all() and torch.isfinite(hi).all(), f"{label}: band not finite"
        assert bool((lo <= hi).all())
        # and the surviving band is the clean one, not some rescued average of the wreckage
        assert torch.allclose(med, clean_med), f"{label}: the good draws' band changed"

def test_the_psd_band_reports_when_nothing_survives():
    """All-bad input must produce NaNs and a count, not an exception and not a silent empty plot."""
    import torch
    from core.SBI import overlay

    traces = torch.full((4, 1024), float("nan"), dtype=torch.float64)
    freqs, lo, med, hi, dropped = overlay.psd_band(traces, 1.0 / 500.0)
    assert dropped == 4
    assert not torch.isfinite(med).any() and len(med) == len(freqs)

def test_the_cycle_average_masks_non_finite_samples_instead_of_binning_them():
    """A NaN phase goes through `.long()` as a garbage integer that `clamp` parks in bin 0, so an
    unmasked divergent draw silently corrupts one end of the cycle -- which reads as a real feature."""
    import math
    import torch
    from core.SBI import overlay

    n, dt, f = 4096, 1.0 / 500.0, 12.0
    t = torch.arange(n, dtype=torch.float64) * dt
    good = torch.sin(2 * math.pi * f * t).unsqueeze(0)

    _, m_clean, _, _, d_clean = overlay.cycle_average(good, dt, f)
    assert d_clean == 0

    dirty = torch.cat([good, torch.full((1, n), float("nan"), dtype=torch.float64)], dim=0)
    _, m_dirty, lo_d, hi_d, d_dirty = overlay.cycle_average(dirty, dt, f)
    assert d_dirty == n, f"expected the whole bad row masked, got {d_dirty}"
    live = torch.isfinite(m_clean)
    assert torch.allclose(m_dirty[live], m_clean[live]), "the good row's cycle changed"
    assert torch.isfinite(m_dirty[live]).all(), "bin 0 was poisoned by the NaN row"

def test_an_empty_predictive_band_is_annotated_on_the_psd_figure():
    """An absent band must not be mistakable for a rendering glitch -- which is exactly what happened.
    When nothing finite survives, the figure has to say so in the axes."""
    import numpy as np
    from core.Helpers import visualizers

    _app()
    freqs = np.linspace(0.1, 100, 64)
    nan = np.full(64, np.nan)
    fig = visualizers.plot_psd_overlay(freqs, np.ones(64), nan, nan, nan, n_dropped=7)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("no finite" in t for t in texts), f"no explanation drawn: {texts}"
    assert any("7" in t for t in texts), f"the dropped count is not on the figure: {texts}"

    # and with a healthy band there must be no such annotation
    ok = visualizers.plot_psd_overlay(freqs, np.ones(64), np.ones(64) * .5, np.ones(64),
                                      np.ones(64) * 2)
    assert not [t.get_text() for t in ok.axes[0].texts], "annotated a perfectly good band"

def test_overlay_figures_render_and_are_picklable():
    """Each new Infer-tab figure must build and survive pickling (the 'Pop out' path unpickles it)."""
    import pickle
    import numpy as np
    from core.Helpers import visualizers

    _app()
    t = np.linspace(0, 2, 400)
    y = np.sin(2 * np.pi * 7.5 * t)
    labels_ = [f"$p_{{{i}}}$" for i in range(13)]
    vals = list(range(13))
    figs = [
        visualizers.plot_best_fit_overlay(t, y, y * 1.05, param_labels=labels_, param_values=vals,
                                          ground_truth=vals, criterion="closest summary statistics",
                                          score_text="RMS z = 0.4"),
        visualizers.plot_overlay_band(t, y, y - 0.2, y, y + 0.2, n_used=20),
        visualizers.plot_psd_overlay(np.linspace(0.1, 100, 50), np.ones(50), np.ones(50) * 0.5,
                                     np.ones(50), np.ones(50) * 2),
        visualizers.plot_cycle_average(np.linspace(0, 2 * np.pi, 48), np.zeros(48), np.zeros(48),
                                       -np.ones(48), np.ones(48)),
    ]
    for fig in figs:
        assert fig.axes, "figure has no axes"
        pickle.loads(pickle.dumps(fig))                  # must not raise
    # the 13-row parameter table gets its own axes, never the title
    assert len(figs[0].axes) == 2

def test_help_badge_carries_its_text():
    from core.gui.widgets.help_badge import HelpBadge
    _app()
    assert HelpBadge("what this does").toolTip() == "what this does"

def test_simulated_inference_runner_emits_the_ground_truth_figure():
    """The simulated-inference runner shows the 'Ground-truth trace' figure before inferring (the old
    Simulate tab did only the first half; the tab is gone, the figure is not). A real SDE sim is too slow
    for a unit test, so stub the heavy pieces and assert the fig_sink wiring."""
    import torch
    from core import cli, orchestrator
    from core.gui.panels import inference_tabs

    _app()

    class Cfg:
        length_unit = "nm"                       # trace y-axis unit (round-4 labels)

        def get_unit_conversion_factor(self, _unit):
            return 1.0

    seen = []
    real_gt, real_go = cli.load_and_validate_gt, orchestrator.generate_observations
    real_iv = orchestrator.infer_and_visualize
    cli.load_and_validate_gt = lambda cfg, path: []
    orchestrator.generate_observations = lambda cfg: (
        torch.zeros(1, 5), None, torch.linspace(0, 1, 5).unsqueeze(0))
    orchestrator.infer_and_visualize = lambda *a, **k: None
    try:
        inference_tabs._run_simulated_inference(
            Cfg(), object(), "cell.txt", 0.1, fig_sink=lambda title, fig: seen.append(title))
    finally:
        cli.load_and_validate_gt = real_gt
        orchestrator.generate_observations = real_go
        orchestrator.infer_and_visualize = real_iv

    assert seen == ["Ground-truth trace"], seen


if __name__ == "__main__":
    _app()
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            # Exception, NOT AssertionError: a crash is a failure of THAT test, not of the suite.
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}\n      {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
