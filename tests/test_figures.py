"""Figure sinks, pop-out windows, dark-theme matplotlib and label rendering. Split from test_gui_progress.py; run directly: python tests/test_figures.py"""
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
# ── interactive "Pop out" for figures ────────────────────────────────────────────────────────────
def _tiny_fig():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig
def _png_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()
# ── Simulate section: "Save video…" export ───────────────────────────────────────────────────────
def _tiny_series():
    import numpy as np
    t = np.arange(600) * 1e-3                                   # 0.6 s -> several video frames at 30 fps
    x = np.sin(2 * np.pi * 8 * t) * 1.5 + 0.2
    return np.column_stack((t, x))
def _export_kwargs():
    import numpy as np
    return dict(window_pts=2000, grid_x=np.linspace(0, 2.6, 60), grid_y=np.linspace(0, 1, 24),
                sigma_x=0.10, sigma_y=0.20, aspect=2.6, margin=0.35, video_fps=30)
@contextlib.contextmanager
def _rcparams_guard():
    """Snapshot + restore matplotlib rcParams (process-global) so a theming test never bleeds its dark
    colours into the plot-producing tests that expect matplotlib's white defaults."""
    import matplotlib
    saved = dict(matplotlib.rcParams)
    try:
        yield
    finally:
        matplotlib.rcParams.update(saved)
def _fake_appearance(dark):
    """A minimal Appearance stand-in (is_dark + a real theme_changed Signal) so the theming test never
    touches theming._ACTIVE / the app styleHints wiring a real Appearance installs."""
    from PySide6.QtCore import QObject, Signal

    class _FakeAppearance(QObject):
        theme_changed = Signal(bool)

        def __init__(self, d):
            super().__init__()
            self._dark = d

        def is_dark(self):
            return self._dark

        def flip(self, d):
            self._dark = d
            self.theme_changed.emit(d)

    return _FakeAppearance(dark)


def test_fig_sink_emits_png_and_a_reloadable_pickle():
    """The worker sink must emit BOTH the PNG thumbnail and a pickle that reloads to a real Figure on
    the GUI thread -- that pickle is what "Pop out" rebuilds into an interactive window."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.panels.base_panel import _png_fig_sink

    _app()
    sig = WorkerSignals()
    events = []
    sig.figure.connect(lambda title, png, fp: events.append((title, png, fp)))

    _png_fig_sink(sig.figure)("Corner", _tiny_fig())

    assert len(events) == 1, events
    title, png, fp = events[0]
    assert title == "Corner"
    assert png[:4] == b"\x89PNG", "the PNG thumbnail is missing / not a PNG"
    assert fp is not None, "no pickle was shipped for the pop-out"
    assert len(pickle.loads(fp).axes) == 1, "the pickle did not reload to the figure"
    plt.close("all")

def test_fig_sink_pickle_failure_still_emits_the_png():
    """If a figure will not pickle, the run must not break: emit fig_pickle=None and keep the PNG."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.panels.base_panel import _png_fig_sink

    _app()
    sig = WorkerSignals()
    events = []
    sig.figure.connect(lambda title, png, fp: events.append((title, png, fp)))

    real = pickle.dumps
    pickle.dumps = lambda *a, **k: (_ for _ in ()).throw(TypeError("cannot pickle this"))
    try:
        _png_fig_sink(sig.figure)("X", _tiny_fig())
    finally:
        pickle.dumps = real

    assert len(events) == 1
    _title, png, fp = events[0]
    assert fp is None, "a pickle failure should degrade to None, not raise or drop the event"
    assert png[:4] == b"\x89PNG", "the PNG must still be emitted when pickling fails"
    plt.close("all")

def test_add_figure_creates_an_interactive_capable_tab():
    import pickle
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QPushButton
    from core.gui.widgets.figure_stack import FigureStack

    _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    assert fs.count() == 1
    container = fs.widget(0)
    assert getattr(container, "_fig_pickle", None) is not None, "the pickle was not stored on the tab"
    assert any(b.text() == "Pop out" for b in container.findChildren(QPushButton)), "no Pop out button"

def test_pop_out_of_a_pickle_builds_a_qtagg_canvas_and_keeps_pyplot_clean():
    """Popping a pickled figure builds a live FigureCanvasQTAgg on the GUI thread, and -- the linchpin
    -- must NOT leave the reconstructed figure registered in pyplot's Gcf (else the worker's
    plt.close("all") would later close it out from under the user)."""
    import pickle
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from core.gui.widgets.figure_stack import FigureStack

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    before = plt.get_fignums()
    fs._pop_out(fs.widget(0))
    assert len(fs._windows) == 1
    win = next(iter(fs._windows))
    assert isinstance(win.canvas, FigureCanvasQTAgg)
    assert plt.get_fignums() == before, "the unpickled figure leaked into pyplot's Gcf"

    win.close()
    _pump(app, 0.1)
    assert len(fs._windows) == 0, "the window ref was not dropped on close"
    plt.close("all")

def test_pop_out_without_a_pickle_uses_the_image_viewer():
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QGraphicsView
    from core.gui.widgets.figure_stack import FigureStack
    from core.gui.widgets.figure_window import ImageZoomWindow

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("NoPickle", _png_bytes(fig), fig_pickle=None)
    plt.close("all")

    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert isinstance(win, ImageZoomWindow), "a pickle-less figure should open the image viewer"
    assert isinstance(win.view, QGraphicsView)
    win.close()
    _pump(app, 0.1)

def test_disk_png_pop_out_is_an_image_viewer():
    import tempfile
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QGraphicsView
    from core.gui.widgets.figure_stack import FigureStack
    from core.gui.widgets.figure_window import ImageZoomWindow

    app = _app()
    png = Path(tempfile.mkdtemp()) / "sweep.png"
    fig = _tiny_fig()
    fig.savefig(str(png), format="png")
    plt.close("all")

    fs = FigureStack()
    fs.add_png("Sweep", str(png))
    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert isinstance(win, ImageZoomWindow)
    assert isinstance(win.view, QGraphicsView), "the disk-PNG pop-out has no zoom/pan view"
    win.close()
    _pump(app, 0.1)

def test_a_popped_out_figure_survives_a_worker_plt_close_all():
    """Worker.run runs plt.close("all") after every run and every cancel. A figure the user has popped
    out must NOT be torn down by it -- the Gcf detach in build_interactive_window is the guarantee."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.widgets.figure_stack import FigureStack

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert len(win._fig.axes) == 1

    plt.close("all")     # the worker's teardown, fired process-wide
    assert len(win._fig.axes) == 1, "plt.close('all') destroyed a figure being viewed in a pop-out"

    win.close()
    _pump(app, 0.1)
    plt.close("all")

# ── Labels + units (round 4) ─────────────────────────────────────────────────────────────────────
def test_labels_axis_and_rescale_render_latex_with_units():
    from core.Helpers import labels as L
    assert L.axis_label("x", "nm") == "$x$ (nm)"
    assert L.axis_label(r"\tilde\omega") == r"$\tilde\omega$ (ND)"          # unit=None -> ND
    assert "ms/ND" in L.rescale_axis_label("t_scale", time_unit="ms")
    assert "nm/ND" in L.rescale_axis_label("x_scale", length_unit="nm")
    assert "pN/ND" in L.rescale_axis_label("f_scale", force_unit="pN")
    assert L.rescale_axis_label("x_offset", length_unit="nm") == r"$x_{\mathrm{off}}$ (nm)"
    assert L.rescale_axis_label("t_scale") == r"$t_{\mathrm{scale}}$"       # missing token -> bare symbol

def test_labels_pretty_gui_and_forcing():
    from core.Helpers import labels as L
    assert L.pretty_gui("F0 (ND forcing amplitude)") == "F<sub>0</sub> (ND forcing amplitude)"
    assert "<sub>obs</sub>" in L.pretty_gui("T_obs (s)")
    assert "<sub>a</sub>/T" in L.pretty_gui("T_a/T grid  (S = 0)")
    assert L.pretty_gui("Model") == "Model"                                # passthrough for non-math
    assert L.gui_forcing_label("phase", "rad") == "φ (rad)"
    assert L.gui_forcing_label("amp") == "A"

def test_simconfig_units_and_inferred_labels_are_latex():
    from pathlib import Path
    from core import cli
    from core.config import BOUNDS_PATH, CELL_PATH, VALID_LABELS, VALID_MODELS

    cell = CELL_PATH / "nadrowski" / "master_weak.txt"
    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    if not (cell.exists() and bounds.exists()):
        return                                                             # environment without Resources: skip
    cfg = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")], True, str(bounds))
    cli.load_and_validate_gt(cfg, str(cell))
    # freq is declared kHz, NOT Hz: drive frequency is consumed as INVERSE CELL TIME (1/ms = kHz), so a
    # declared "Hz" against an `ms` cell is a 1000x lie. See SimConfig.freq_si_to_cell.
    assert (cfg.length_unit, cfg.time_unit, cfg.force_unit, cfg.freq_unit) == ("nm", "ms", "pN", "kHz")
    assert cfg.check_unit_consistency() == [], "nadrowski units must be self-consistent"
    assert abs(30.0 * cfg.freq_si_to_cell - 0.03) < 1e-12, "30 Hz must be 0.03 cycles/ms in an ms cell"
    labels = cfg.inferred_labels
    assert all(l.startswith("$") for l in labels), labels
    assert any("nm/ND" in l for l in labels), "x_scale should carry nm/ND"

    bp_bounds = BOUNDS_PATH / "bp" / "cell_1.txt"
    if bp_bounds.exists():                                                 # BP declares no force/freq unit
        bp = cli.make_sim_config("BP", VALID_LABELS[VALID_MODELS.index("BP")], False, str(bp_bounds))
        assert bp.force_unit is None and bp.length_unit == "nm"

def test_plot_posterior_vs_truth_default_labels_have_units():
    import numpy as np
    from core.Helpers import visualizers
    fig = visualizers.plot_posterior_vs_truth(np.arange(5) * 1.0, np.zeros(5))
    ax = fig.axes[0]
    assert ax.get_xlabel() == "$t$ (s)" and ax.get_ylabel() == "$x$ (nm)"
    fig2 = visualizers.plot_posterior_vs_truth(np.arange(5) * 1.0, np.zeros(5),
                                               xlabel="$t$ (ms)", ylabel="$x$ (µm)")
    assert fig2.axes[0].get_xlabel() == "$t$ (ms)"

def test_gui_form_labels_are_prettified():
    from PySide6.QtWidgets import QLabel
    from core.gui.widgets.help_badge import help_label

    _app()
    holder = help_label("F0 (ND forcing amplitude)", "help text")
    lbl = holder.findChild(QLabel)
    assert lbl is not None and "F<sub>0</sub>" in lbl.text()
    # panels still build with the prettify hook in place
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.simulate_panel import SimulatePanel
    FdtPanel(); CrossValPanel(); SimulatePanel()

def test_a_theme_flip_between_build_and_save_cannot_split_a_figure():
    """THE 2026-08-25 RENDERING FAILURE, pinned, with its own counterfactual.

    matplotlib bakes artist colours at BUILD time but reads `savefig.facecolor` at SAVE time, and
    mpl_theme rewrites rcParams GLOBALLY on every appearance change. A multi-day run straddling one
    flip therefore saved dark axes onto a light page with `text.color` still light -- every tick
    label, axis label, title and table cell at ~1.06:1 contrast, i.e. invisible. 14 of 15 figures.
    """
    import io as _io
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from core.gui import mpl_theme
    from core.Helpers import visualizers

    _app()
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme(True)                 # build under DARK
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("x")
        try:
            mpl_theme.apply_mpl_theme(False)            # ...then the theme flips to LIGHT

            def corner_luminance(save):
                buf = _io.BytesIO()
                save(buf)
                buf.seek(0)
                return float(np.array(Image.open(buf).convert("RGB")).astype(int)[:6, :6].mean())

            pinned = corner_luminance(
                lambda b: visualizers.save_figure(fig, b, format="png", dpi=50))
            unpinned = corner_luminance(
                lambda b: fig.savefig(b, format="png", dpi=50))
        finally:
            plt.close(fig)

    assert pinned < 120, \
        f"save_figure put dark artists on a light page after a theme flip (corner {pinned:.0f})"
    assert unpinned > 200, \
        f"the counterfactual did not reproduce the bug (corner {unpinned:.0f}); this test proves nothing"

def test_the_parameter_table_is_readable_against_its_own_cells():
    """The best-fit parameter table came back completely blank. Transparent cells show the FIGURE
    background while the text is `text.color`, and those two only contrast while the theme is
    self-consistent -- so the 13 numbers you most want to read are the first thing to vanish."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgb
    from core.gui import mpl_theme
    from core.Helpers import visualizers

    def relative_luminance(c):
        r, g, b = (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in to_rgb(c))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    _app()
    for dark in (True, False):
        with _rcparams_guard():
            mpl_theme.apply_mpl_theme(dark)
            t = np.linspace(0, 1, 32)
            fig = visualizers.plot_best_fit_overlay(
                t, np.sin(t), np.sin(t) * 1.02,
                param_labels=[f"p{i}" for i in range(13)], param_values=list(range(13)))
            try:
                tables = [c for a in fig.axes for c in a.tables]
                assert tables, "the parameter table is gone"
                cells = list(tables[0].get_celld().values())
                for cell in cells:
                    bg = cell.get_facecolor()
                    assert bg[3] > 0.5, "a transparent cell: the text falls back to the FIGURE colour"
                    l1 = relative_luminance(bg[:3])
                    l2 = relative_luminance(cell.get_text().get_color())
                    ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
                    assert ratio >= 3.0, \
                        f"dark={dark}: table text at {ratio:.2f}:1 against its cell (was 1.06:1)"
            finally:
                plt.close(fig)

def test_apply_mpl_theme_sets_design_tokens():
    """apply_mpl_theme drives figure/axes/savefig facecolor + text.color from the DARK/LIGHT tokens."""
    import matplotlib
    from core.gui import design, mpl_theme
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme(True)
        d = design.tokens(True)
        assert matplotlib.rcParams["figure.facecolor"] == d["window"]
        assert matplotlib.rcParams["axes.facecolor"] == d["base"]
        assert matplotlib.rcParams["savefig.facecolor"] == d["window"]
        assert matplotlib.rcParams["text.color"] == d["text"]
        mpl_theme.apply_mpl_theme(False)
        assert matplotlib.rcParams["figure.facecolor"] == design.tokens(False)["window"]

def test_mpl_theme_install_none_is_a_noop():
    """No Appearance (the test path) -> rcParams untouched, so figures keep matplotlib's default white."""
    import matplotlib
    from core.gui import mpl_theme
    with _rcparams_guard():
        before = matplotlib.rcParams["figure.facecolor"]
        mpl_theme.install(None)
        assert matplotlib.rcParams["figure.facecolor"] == before

def test_mpl_theme_follows_the_appearance_signal():
    """install(appearance) applies now AND subscribes: flipping the theme re-applies the rcParams."""
    import matplotlib
    from core.gui import design, mpl_theme
    _app()
    with _rcparams_guard():
        ap = _fake_appearance(True)
        mpl_theme.install(ap)
        assert matplotlib.rcParams["figure.facecolor"] == design.tokens(True)["window"]
        ap.flip(False)
        assert matplotlib.rcParams["figure.facecolor"] == design.tokens(False)["window"]

def test_plot_ppc_summary_box_follows_the_dark_theme():
    """The two PPC summary boxes read plt.rcParams, so under a dark theme they are NOT the old hardcoded
    white -- locks the B-c hardcoded-white regression."""
    import numpy as np
    import matplotlib.pyplot as plt
    from core.gui import mpl_theme
    from core.Helpers.visualizers import plot_ppc
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme(True)
        ppc = {"z_scores": np.array([0.1, 0.5, 2.5, np.nan]),
               "coverage_90": 0.9, "num_outside": 1, "num_invalid": 1}
        fig = plot_ppc(ppc)
        try:
            fig.canvas.draw()
            boxes = [t.get_bbox_patch() for t in fig.axes[0].texts if t.get_bbox_patch() is not None]
            assert boxes, "expected summary boxes with a bbox patch"
            assert all(max(p.get_facecolor()[:3]) < 0.5 for p in boxes), \
                [p.get_facecolor() for p in boxes]
        finally:
            plt.close(fig)

def test_export_animation_background_stays_white_under_dark_theme():
    """The Simulate video export is insulated from the app's matplotlib theme: its background stays
    white even when the global rcParams are dark, so exported videos look the same in any theme."""
    import os
    import tempfile
    from core.gui import mpl_theme
    from core.gui.panels.simulate_export import export_animation
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme(True)                            # global dark theme active
        path = os.path.join(tempfile.mkdtemp(), "dark.gif")
        export_animation(_tiny_series(), path, **_export_kwargs())
        import imageio
        frame0 = imageio.mimread(path)[0]
        corner = frame0[0, 0][:3]                                  # top-left = figure background margin
        assert min(int(c) for c in corner) > 230, corner          # near-white, not the dark theme bg


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
