"""QSettings round-trips, the training-budget group, layout geometry, and the config-tab hardware fields. Split from test_gui_progress.py; run directly: python tests/test_settings_persistence.py"""
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
def _pump(app, seconds=0.5):
    """Drive the event loop without app.exec(), so the pump's queued signals get delivered."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
# -- the training budget (Posterior tab) ----------------------------------------------------------
def _budget_cfg():
    """Stub SimConfig carrying exactly the fields the budget lines read.

    A stub, not a real build: the arithmetic under test is geometry -> elements -> GiB, and a real
    make_sim_config drags in bounds files and a 300k-point time grid for nothing.
    """
    from core import config

    class Cfg:
        hw = config.detect_device()
        t = type("T", (), {"shape": (250_000,)})()
        inits_dict = {"x": 0.0, "xa": 0.0, "f": 0.0}
        steady_idx = 500
        forcing_idx = {}
        model = "NADROWSKI"
        chi_mode = False
        chi_k_pad = 12
        observation_mode = "spontaneous"
    return Cfg()
def _budget_panel(cfg=None, prior=None):
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession()
    inf.session.cfg = cfg
    inf.session.inf_prior = prior
    inf.refresh_gates()
    return inf, inf.tabs.widget(2)          # Config Prior Posterior Validate Infer
# ── Phase 3: QSettings persistence ───────────────────────────────────────────────────────────────
def _temp_settings():
    import tempfile
    from core.gui import settings as st
    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    st.use_ini_file(path)
    return path
def _strip_docstrings(tree):
    """Remove every docstring from a parsed tree, in place, and return it.

    ⚠ ``ast.unparse`` DROPS COMMENTS BUT KEEPS DOCSTRINGS -- they are real string expressions in the
    AST, not trivia. An earlier version of _unparsed claimed otherwise, and the claim went unnoticed
    because the checks that used it happened to forbid strings that appeared only in comments. It
    stopped being harmless the moment a check forbade `mem_get_info` in a function whose DOCSTRING
    explains why it does not use mem_get_info: the assertion matched the prose, exactly the
    false positive the parse was supposed to prevent (the same shape as the _local_map lesson).
    """
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and isinstance(body, list) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.fix_missing_locations(tree)
def _code_only(obj) -> str:
    """Executable source of ``obj`` -- comments and docstrings both gone. See _strip_docstrings."""
    return ast.unparse(_strip_docstrings(ast.parse(textwrap.dedent(inspect.getsource(obj)))))


def test_the_training_budget_shows_the_simulation_count_and_the_cap_trade():
    """The count was invisible: 5000 x 2048 = 10,240,000 simulations lived only in config.py. The line
    also has to say what the cap really trades, because batch width is NOT a speed knob -- the solver
    is kernel-launch-bound (measured 7.37 s at 2048 against 7.74 s at 1024, the SMALLER batch being
    slightly slower), so halving it halves the training rows for the same wall-clock.

    ⚠ ISOLATES THE SETTINGS FILE, and must. The panel seeds these fields from config.py and then
    RESTORES them from QSettings, so without isolation the assertions read whatever the developer
    last left in their real PRISM.ini. This test passed all morning and then failed the moment a run
    was configured with 10000 batches -- nothing to do with the code under test. It is the same
    restore-wins-over-config mechanism that cost a ~5-day run on 2026-08-19 (the retired chi band),
    so a test asserting the DEFAULT has to start from a file with no saved value.
    """
    from core import config
    import core.gui.settings as st

    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    os.unlink(path)                      # QSettings creates it; an empty file is not the same thing
    st.use_ini_file(path)
    try:
        cfg = _budget_cfg()
        _inf, panel = _budget_panel(cfg)
        width = cfg.hw.batch_size

        assert panel.num_runs.value() == config.TRAINING_NUM_RUNS, (
            f"the field must seed from config.py, not from a saved session: "
            f"{panel.num_runs.value()} != {config.TRAINING_NUM_RUNS}")
        assert panel.run_size_cap.value() == config.TRAINING_RUN_SIZE
        assert f"{config.TRAINING_NUM_RUNS * width:,} simulations" in panel.budget_total.text(), \
            panel.budget_total.text()
        assert "diversity" in panel.budget_total.text(), \
            "the line must say batch COUNT is the (t_scale, T) diversity, not just a budget"

        panel.run_size_cap.setText(str(width // 4))
        assert f"{config.TRAINING_NUM_RUNS * (width // 4):,} simulations" in panel.budget_total.text()
        assert "capped from" in panel.budget_total.text(), panel.budget_total.text()

        # ...and quadrupling the batch COUNT is what buys those rows back, at 4x the wall-clock.
        panel.num_runs.setText(str(config.TRAINING_NUM_RUNS * 4))
        assert f"{config.TRAINING_NUM_RUNS * width:,} simulations" in panel.budget_total.text()
    finally:
        st.use_ini_file(None)
        if os.path.exists(path):
            os.unlink(path)

def test_the_training_budget_reaches_build_posterior_as_arguments_not_via_config():
    """THE WIRING THAT MAKES THE FIELDS DO ANYTHING AT ALL.

    orchestrator binds TRAINING_NUM_RUNS / TRAINING_RUN_SIZE at import, so a panel that "applied" the
    user's numbers by assigning to core.config would be a silent no-op -- the run would simulate 5000
    x 2048 anyway and nothing would say otherwise. Both halves are asserted: the values arrive as
    call kwargs, AND the config constants are untouched.
    """
    from core import config

    before = (config.TRAINING_NUM_RUNS, config.TRAINING_RUN_SIZE)
    _inf, panel = _budget_panel(_budget_cfg(), prior=object())
    panel.num_runs.setText("777")
    panel.run_size_cap.setText("256")

    seen = {}
    panel.dispatch = lambda fn, *a, **kw: seen.update(fn=fn, args=a, kwargs=kw)
    panel._build_posterior()

    assert seen, "the Train button dispatched nothing"
    assert seen["kwargs"].get("num_runs") == 777, seen["kwargs"]
    assert seen["kwargs"].get("run_size_cap") == 256, seen["kwargs"]
    assert (config.TRAINING_NUM_RUNS, config.TRAINING_RUN_SIZE) == before, \
        "the panel mutated the config constants, which orchestrator has already snapshotted"

def test_the_budget_refuses_a_batch_count_below_one():
    """0 batches is a whole run that simulates nothing and then trains on an empty tensor."""
    _inf, panel = _budget_panel(_budget_cfg(), prior=object())
    panel.num_runs.setText("0")
    seen = []
    panel.dispatch = lambda fn, *a, **kw: seen.append(kw)
    panel._build_posterior()
    assert not seen, "a zero batch count was dispatched"

def test_the_budget_memory_line_reads_pipelines_own_cost_model():
    """The estimate must come from pipeline.peak_sim_elements, not a second copy of the formula, and
    it must be quoted at the WORST geometry the Sobol pre-filter admits -- n_fine swings from a median
    ~40k to a p99 ~283k, so a width that fits the median still OOMs on a few percent of batches, which
    is how two retrains actually died."""
    from core import config
    from core.SBI import pipeline

    cfg = _budget_cfg()
    _inf, panel = _budget_panel(cfg)
    width = cfg.hw.batch_size

    if cfg.hw.device.type != "cuda":
        assert "CUDA-only" in panel.budget_mem.text(), panel.budget_mem.text()
        return

    n_fine = min(config.N_ND_MAX, cfg.t.shape[0])
    need = pipeline.peak_sim_elements(width, n_fine, cfg.steady_idx, len(cfg.inits_dict), 1, 1)
    gib = need * cfg.hw.dtype.itemsize / float(1 << 30)
    assert f"{gib:.2f} GiB" in panel.budget_mem.text(), panel.budget_mem.text()
    assert f"{n_fine:,}" in panel.budget_mem.text(), "the line must name the geometry it assumed"
    assert "upper bound" in panel.budget_mem.text().lower(), \
        "free-VRAM readings overstate what is available; the line must not present one as fact"

    # Halving the width must halve the estimate -- the peak is linear in the batch.
    panel.run_size_cap.setText(str(width // 2))
    assert f"{gib / 2:.2f} GiB" in panel.budget_mem.text(), panel.budget_mem.text()

def test_the_budget_lines_never_raise_on_a_config_they_do_not_understand():
    """_sync_budget runs from refresh_gates(), so an exception in a STATUS LINE would take down the
    whole tab. The gate tests set session.cfg to a bare object(); so could any future stub."""
    _inf, panel = _budget_panel(cfg=object(), prior=object())
    panel._sync_budget()                                   # must not raise
    assert panel.budget_total.text(), "the total line went blank on an unknown config"
    assert "config" in panel.budget_ckpt.text().lower(), panel.budget_ckpt.text()

def test_the_training_budget_round_trips_through_settings():
    """"I have to retype it every launch" is the complaint L1 already answered for splitters."""
    from core.gui import settings as st

    path = _temp_settings()
    try:
        _inf, panel = _budget_panel(_budget_cfg())
        panel.num_runs.setText("1234")
        panel.run_size_cap.setText("512")
        panel.save_settings(st.settings())

        _inf2, fresh = _budget_panel(_budget_cfg())
        fresh.restore_settings(st.settings())
        assert fresh.num_runs.value() == 1234, fresh.num_runs.text()
        assert fresh.run_size_cap.value() == 512, fresh.run_size_cap.text()
        # And the derived line followed the restored values, not the defaults.
        assert "1,234 batches" in fresh.budget_total.text(), fresh.budget_total.text()
    finally:
        st.use_ini_file(None) if hasattr(st, "use_ini_file") else None
        os.unlink(path)

def test_settings_round_trip_reduction_and_fdt():
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.panels.fdt_panel import FdtPanel

    _app()
    _temp_settings()
    try:
        red = ReductionPanel()
        red.f0.setText("0.123")
        if red.cell_picker.combo.count():
            red.cell_picker.combo.setCurrentIndex(red.cell_picker.combo.count() - 1)
        want_cell = red.cell_picker.key()

        fdt = FdtPanel()
        fdt.n_freqs.setText("77")
        fdt.skip_sanity.setChecked(True)
        fdt.confirm_production.setChecked(False)

        qs = st.settings()
        red.save_settings(qs)
        fdt.save_settings(qs)
        qs.sync()

        red2 = ReductionPanel()
        fdt2 = FdtPanel()
        assert red2.f0.value() == 0.123
        assert red2.cell_picker.key() == want_cell
        assert fdt2.n_freqs.value() == 77
        assert fdt2.skip_sanity.isChecked() is True
        assert fdt2.confirm_production.isChecked() is False
    finally:
        st.use_ini_file(None)

def test_missing_picker_key_restores_to_default_not_blank():
    """A saved selection whose file is gone must leave the picker at its default, never blank it via
    setCurrentIndex(-1)."""
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel

    _app()
    _temp_settings()
    try:
        qs = st.settings()
        qs.beginGroup("reduction")
        qs.setValue("cell", "nadrowski/does_not_exist.txt")
        qs.setValue("f0", "0.05")
        qs.endGroup()
        qs.sync()

        red = ReductionPanel()
        assert red.cell_picker.combo.currentIndex() >= 0, "a stale key blanked the combo"
    finally:
        st.use_ini_file(None)

def test_crossval_does_not_persist_cell_derived_bounds():
    """The S/T grid lo/hi are re-derived from the cell file; a saved value from a different cell would
    be a stale, wrong bound. Only the free knobs (points, f0, freqs_per_batch, preset, cell) persist."""
    from core.gui import settings as st
    from core.gui.panels.crossval_panel import CrossValPanel

    _app()
    _temp_settings()
    try:
        xv = CrossValPanel()
        derived_hi = xv.s_grid.hi.text()          # set by _on_cell_changed from the cell file
        xv.s_grid.hi.setText("999.0")             # user 'edits' it to a bogus value
        xv.s_grid.points.setText("13")
        xv.f0.setText("0.077")

        qs = st.settings()
        xv.save_settings(qs)
        qs.sync()
        # the bogus hi must NOT have been written
        qs.beginGroup("crossval")
        assert qs.value("s_hi") is None, "cell-derived s_grid.hi was persisted -- it must not be"
        qs.endGroup()

        xv2 = CrossValPanel()
        assert xv2.s_grid.points.text() == "13", "the free `points` knob was not restored"
        assert xv2.f0.value() == 0.077
        assert xv2.s_grid.hi.text() == derived_hi, "the grid bound must be RE-DERIVED, not restored"
    finally:
        st.use_ini_file(None)

def test_panel_splitter_is_sized_and_not_collapsible():
    """Every panel opens with a usable controls column that cannot be dragged to nothing.

    BasePanel builds the app's only QSplitter and there are nine live instances. It used to be a
    LOCAL with no setSizes and no setChildrenCollapsible, so every launch started at the minimum and
    one slip past the left edge collapsed the controls to zero width, recoverable only by finding a
    5px handle at x=0.
    """
    app = _app()
    from core.gui.panels.base_panel import BasePanel

    class P(BasePanel):
        pass

    panel = P()
    panel.resize(1300, 820)
    _pump(app, 0.15)
    assert not panel.splitter.childrenCollapsible(), \
        "the controls column can still be collapsed to zero width"
    assert all(s > 0 for s in panel.splitter.sizes()), \
        f"splitter opened with a zero-width pane: {panel.splitter.sizes()}"
    # The old hard 460px cap could not be escaped by widening the window, so any wider form got a
    # permanent horizontal scrollbar in the left column.
    assert panel.controls_scroll.maximumWidth() > 1000, \
        f"controls column is still hard-capped at {panel.controls_scroll.maximumWidth()}px"

def test_panel_layout_round_trips_through_settings():
    """The splitter position must survive a restart -- 'I have to re-drag it every launch' is the
    complaint. Uses saveState/restoreState, which work before the widget is shown or polished."""
    app = _app()
    from core.gui import settings as st
    from core.gui.panels.base_panel import BasePanel

    class P(BasePanel):
        pass

    path = _temp_settings()
    try:
        a = P()
        a.resize(1300, 820)
        _pump(app, 0.15)
        a.splitter.setSizes([500, 700])
        _pump(app, 0.05)
        want = a.splitter.sizes()
        qs = st.settings()
        a.save_layout(qs)
        qs.sync()

        b = P()                                   # restore_layout runs in __init__
        b.resize(1300, 820)
        _pump(app, 0.15)
        assert b.splitter.sizes() == want, \
            f"layout did not round-trip: saved {want}, restored {b.splitter.sizes()}"
    finally:
        st.use_ini_file(None)
        os.unlink(path)

def test_forms_grow_their_fields_and_numeric_boxes_have_a_floor():
    """Pins both halves of the 'input boxes are cut off' fix, across every form in every panel.

    Qt's Windows default is FieldsStayAtSizeHint: the field takes its size hint and stops, so numeric
    boxes rendered 3-6 characters wide and typing "0.033333" scrolled inside the box. Repo-wide there
    were ZERO setFieldGrowthPolicy calls over 18 QFormLayout sites. Asserting over discovered forms
    (not a fixed list) means a newly added form is covered too.
    """
    from PySide6.QtWidgets import QFormLayout, QLineEdit
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.widgets.labeled_inputs import FloatField, IntField

    _app()
    path = _temp_settings()
    try:
        panels = [CrossValPanel(), FdtPanel(), ReductionPanel()]
        forms = [f for p in panels for f in p.findChildren(QFormLayout)]
        assert forms, "no QFormLayouts discovered — the test is not exercising anything"
        bad = [f for f in forms
               if f.fieldGrowthPolicy() != QFormLayout.AllNonFixedFieldsGrow]
        assert not bad, f"{len(bad)}/{len(forms)} forms do not grow their fields"

        narrow = [w for p in panels for w in p.findChildren(QLineEdit)
                  if isinstance(w, (FloatField, IntField)) and w.minimumWidth() < 80]
        assert not narrow, \
            f"{len(narrow)} numeric field(s) have no usable minimum width (e.g. " \
            f"{narrow[0].minimumWidth()}px)"
    finally:
        st_mod = __import__("core.gui.settings", fromlist=["settings"])
        st_mod.use_ini_file(None)
        os.unlink(path)

def test_long_diagnostics_are_readable_without_horizontal_scrolling():
    """Panels emit long single-line diagnostics; the log pane must wrap them, and the crossval cell
    label (which receives an unbounded str(e)) must not force the whole column wider."""
    from PySide6.QtWidgets import QPlainTextEdit
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.widgets.log_pane import LogPane

    _app()
    assert LogPane().lineWrapMode() == QPlainTextEdit.WidgetWidth, \
        "log pane still truncates long lines instead of wrapping them"
    path = _temp_settings()
    try:
        assert CrossValPanel().cell_values.wordWrap(), \
            "the crossval cell-values label does not wrap, so a long error widens the whole column"
    finally:
        st_mod = __import__("core.gui.settings", fromlist=["settings"])
        st_mod.use_ini_file(None)
        os.unlink(path)

def test_the_vram_ceiling_is_on_config_live_and_NOT_persisted():
    """The VRAM ceiling is a HARDWARE knob, and it is the one field on the Config tab that is
    deliberately forgotten between sessions.

    NOT PERSISTED, because stale QSettings have already cost this project a ~5-day run: the
    2026-08-19 retrain trained on the RETIRED chi band because a saved value silently won over
    config.py. A ceiling fails the same way but far more quietly -- a forgotten 2 GiB does not
    error, it just makes every future run split from batch 0 and take several times longer, with
    nothing in the log to explain it. Starting each session at config.py's 0.0 keeps the throttle a
    decision somebody just made.

    A PLAIN ASSIGNMENT IS ENOUGH, unlike every other knob the GUI exposes. The sweep and flow fields
    had to become ARGUMENTS threaded into build_prior/build_posterior, because orchestrator does
    `from .config import ...` and binds them at import, so writing to the constant is a silent no-op
    (an imported name is a snapshot). pipeline.vram_ceiling_gib() does a getattr on the module every time the planner
    asks, so this one genuinely takes effect -- and this test would catch it if that ever changed.
    """
    from core.gui.panels import inference_tabs as it
    from core import config as _cfg
    from core.SBI import pipeline as _pipe

    saved = _cfg.SIM_VRAM_CEILING_GIB
    saved_env = os.environ.pop(_pipe.VRAM_CEILING_ENV, None)
    try:
        panel = it.ConfigPanel(None)
        assert hasattr(panel, "vram_ceiling"), "the Config tab must carry the VRAM ceiling field"
        assert panel.vram_ceiling.value() == 0.0, "it must default to OFF"

        # Live: typing must reach the planner with no plumbing in between.
        panel.vram_ceiling.setText("6.5")
        assert _cfg.SIM_VRAM_CEILING_GIB == 6.5, "the field must assign the config constant"
        assert _pipe.vram_ceiling_gib() == 6.5, "and the planner must read it live"
        # 2 GiB, not 6.5: the budget is a min() and the CPU branch of memory_budget_elements caps at
        # 4 GiB, so a ceiling above that would not bind and the assertion would prove nothing.
        panel.vram_ceiling.setText("2")
        dev, dt = torch.device("cpu"), torch.float32
        saved_cap = _pipe._BUDGET_CAP_ELEMENTS
        try:
            _pipe._BUDGET_CAP_ELEMENTS = None       # the other term in the same min()
            assert _pipe.sim_memory_budget_elements(dev, dt) == (2 * 2 ** 30) // 4, \
                "the ceiling must bind the budget the planner actually uses"
        finally:
            _pipe._BUDGET_CAP_ELEMENTS = saved_cap

        # Not persisted: neither direction may mention it.
        src_save = _code_only(it.ConfigPanel.save_settings)
        src_restore = _code_only(it.ConfigPanel.restore_settings)
        for what, src in (("save_settings", src_save), ("restore_settings", src_restore)):
            assert "vram" not in src.lower(), (
                f"{what} must NOT touch the VRAM ceiling -- a stale ceiling throttles every future "
                f"run silently, which is the failure mode the 2026-08-19 band trap already cost us")

        # And the env override must ANNOUNCE that it wins, rather than leaving a dead field.
        os.environ[_pipe.VRAM_CEILING_ENV] = "2.0"
        panel.vram_ceiling.setText("9")
        assert _pipe.vram_ceiling_gib() == 2.0, "the env override wins"
        assert _pipe.VRAM_CEILING_ENV in panel.vram_note.text(), (
            "a field that silently does nothing is worse than no field -- the note must say the "
            "environment is overriding it")
    finally:
        os.environ.pop(_pipe.VRAM_CEILING_ENV, None)
        if saved_env is not None:
            os.environ[_pipe.VRAM_CEILING_ENV] = saved_env
        _cfg.SIM_VRAM_CEILING_GIB = saved

def test_the_free_vram_readout_does_not_use_mem_get_info():
    """⚠ The readout beside the ceiling must come from nvidia-smi, never torch.cuda.mem_get_info.

    That reading overstates free VRAM on Windows by roughly the size of the desktop -- measured
    15037 MiB against nvidia-smi's 5814 at the same instant -- and it is the number that
    green-lit the batch which killed the first chi retrain. Printing it next to a field whose entire
    purpose is to bound VRAM would hand the user the exact lie the field defends against."""
    from core.gui.panels import inference_tabs as it
    src = _code_only(it._nvidia_smi_free_gib)
    assert "mem_get_info" not in src, "the readout must not use the optimistic driver reading"
    assert "nvidia-smi" in src, "the readout must come from nvidia-smi"
    got = it._nvidia_smi_free_gib()
    assert got is None or got >= 0.0, f"unexpected reading {got!r}"

def test_a_one_field_near_miss_blocks_a_fresh_run_until_confirmed():
    """A passive status line was not enough, three times over.

    `_budget_checkpoint` already said "these settings match no checkpoint, so this starts a NEW run"
    and named the differing field -- but it is a label on a tab the user has scrolled past by the
    time they press Train. It failed to prevent 884 batches being lost outright on 2026-08-27, and a
    3989-batch checkpoint being abandoned twice on 2026-08-28. So a run that would start from zero
    while a committed checkpoint sits ONE identity field away now has to be confirmed.

    Narrow on purpose: exactly one differing field is the signature of an accident. Two or more is
    usually a genuinely different experiment, and warning there would make this noise."""
    from core.SBI import training_checkpoint as tc

    base = {"format": "training-rows", "model": "NADROWSKI", "n_runs": 10000,
            "run_size": 2048, "prior_fingerprint": "aaaa", "chi_mode": True}
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        # A committed sibling that differs in exactly ONE field.
        sib = dict(base, prior_fingerprint="bbbb")
        d = tc.resolve_dir(sib, root); (d / "shards").mkdir(parents=True)
        torch.save({"identity": sib}, d / "header.pt")
        torch.save({"batches_done": 3989, "complete": False, "rng": None}, d / "state.pt")

        near = tc.near_miss_siblings(base, root)
        assert len(near) == 1, f"expected the one-field sibling, got {near}"
        assert near[0]["field"] == "prior_fingerprint" and near[0]["batches"] == 3989, near[0]

        # TWO differing fields is a different experiment -- it must NOT warn.
        far = dict(base, prior_fingerprint="cccc", n_runs=5000)
        d2 = tc.resolve_dir(far, root); (d2 / "shards").mkdir(parents=True)
        torch.save({"identity": far}, d2 / "header.pt")
        torch.save({"batches_done": 500, "complete": False, "rng": None}, d2 / "state.pt")
        names = {r["name"] for r in tc.near_miss_siblings(base, root)}
        assert d2.name not in names, "a two-field difference must not be reported as a near miss"

        # An identity that MATCHES a checkpoint is a resume, so there is nothing to confirm.
        assert tc.near_miss_siblings(sib, root) == [] or             all(r["name"] != d.name for r in tc.near_miss_siblings(sib, root)),             "a checkpoint must never be a near miss of itself"

def test_the_confirmation_is_reached_and_can_refuse():
    """The dialog must actually gate the dispatch, and Cancel must mean cancel."""
    from core.gui.panels import inference_tabs as it

    src = _code_only(it.PosteriorPanel._build_posterior)
    assert "_confirm_fresh_run" in src, "the Train handler must consult the confirmation"
    i_confirm = src.find("_confirm_fresh_run")
    i_dispatch = src.find("self.dispatch(")
    assert i_confirm < i_dispatch, "the confirmation must gate the dispatch, not follow it"
    assert "return" in src[i_confirm:i_dispatch], "a refusal must return instead of training"

    # Fails open rather than blocking a run it cannot assess.
    body = _code_only(it.PosteriorPanel._confirm_fresh_run)
    assert body.count("return True") >= 3, (
        "the check must fail OPEN -- no prior, an unreadable identity, and no near miss must all "
        "proceed; a warning that can block a run is worse than no warning")


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
