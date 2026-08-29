"""The Simulate section's streaming loop and its video export. Split from test_gui_progress.py; run directly: python tests/test_simulate.py"""
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
# ── Phase 3: QSettings persistence ───────────────────────────────────────────────────────────────
def _temp_settings():
    import tempfile
    from core.gui import settings as st
    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    st.use_ini_file(path)
    return path
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


# ── Simulate section (real-time streaming) ───────────────────────────────────────────────────────
def test_simulate_frame_time_grid_preserves_dt_and_is_continuous():
    """The streaming loop advances one frame at a time; the frame grid must keep the fine EM step exactly
    dt_nd (so stability/timescale don't drift) and hand off continuously to the next frame."""
    from core.gui.panels.simulate_runner import frame_time_grid

    g = frame_time_grid(0.0, 100, 0.025)
    assert g.shape[0] == 101, "a frame of m steps needs m+1 points (dt = (t1-t0)/(n-1))"
    assert abs((g[1] - g[0]).item() - 0.025) < 1e-6
    assert abs((g[-1] - g[-2]).item() - 0.025) < 1e-6
    g2 = frame_time_grid(g[-1].item(), 40, 0.025)          # the next frame starts where this one ended
    assert abs(g2[0].item() - g[-1].item()) < 1e-6, "frames must be time-continuous"
    assert abs((g2[1] - g2[0]).item() - 0.025) < 1e-6

def test_simulate_gaussian_field_is_an_ellipse_perpendicular_to_the_motion():
    """The heatmap blob must peak at its center, stay in [0, 1], and be an ellipse whose MAJOR axis is
    perpendicular to the (horizontal) oscillation -- i.e. it decays faster along the motion axis than
    across it (sigma_par < sigma_perp)."""
    import numpy as np
    from core.gui.widgets.live_hair_bundle import gaussian_field

    gx = np.linspace(0, 1, 33)                             # 33 points -> 0.5 lands exactly on index 16
    gy = np.linspace(0, 1, 33)
    sig_par, sig_perp = 0.10, 0.20                         # along motion (x) < perpendicular (y)
    f = gaussian_field(0.5, 0.5, gx, gy, sig_par, sig_perp)
    assert f.shape == (33, 33)
    assert abs(float(f.max()) - 1.0) < 1e-5
    ix, iy = np.unravel_index(int(np.argmax(f)), f.shape)
    assert gx[ix] == 0.5 and gy[iy] == 0.5
    # moving the center along the motion axis must shift the blob
    ix2, _ = np.unravel_index(int(np.argmax(gaussian_field(0.75, 0.5, gx, gy, sig_par, sig_perp))), f.shape)
    assert gx[ix2] == 0.75

    # ellipse orientation: for a fixed offset the field is SMALLER along the motion axis than across it.
    d = 0.15
    along = gaussian_field(0.5, 0.5, np.array([0.5 + d]), np.array([0.5]), sig_par, sig_perp)[0, 0]
    across = gaussian_field(0.5, 0.5, np.array([0.5]), np.array([0.5 + d]), sig_par, sig_perp)[0, 0]
    assert along < across, "the ellipse major axis must be perpendicular to the oscillation"

def test_simulate_heatmap_center_stays_off_the_edge_when_oscillating():
    """The blob center must be mapped inside a horizontal margin, so an extreme displacement (x0_norm at
    0 or 1) does not clip at the field edge -- the on-display 'cut off when oscillating' bug."""
    from core.gui.widgets.live_hair_bundle import LiveHairBundleView

    _app()
    v = LiveHairBundleView()
    assert v._margin > 0.0
    assert v._cx(0.0) >= v._margin - 1e-9
    assert v._cx(1.0) <= v._aspect - v._margin + 1e-9
    assert v._aspect > 1.0, "the heatmap field must be a wide rectangle"

def test_simulate_plan_stream_matches_generate_observations_arithmetic():
    """plan_stream must reproduce the pipeline's subsample/steady/total arithmetic so a streamed trace
    matches the observation the SBI pipeline would build from the same cell."""
    from core.cli import resolve_bounds_for_cell
    from core.config import CELL_PATH
    from core.gui.panels.simulate_runner import build_stream_config, plan_stream

    _app()
    cdir = CELL_PATH / "nadrowski"
    # Resolve through the shared rule, not a same-named-sibling glob: the master cells deliberately
    # share ONE bounds file, so a sibling-only filter would silently skip most of them.
    cells = [c for c in sorted(cdir.glob("*.txt"))
             if resolve_bounds_for_cell(str(c)) is not None] if cdir.exists() else []
    if not cells:
        return                                             # environment without Resources: skip, don't fail

    cfg = build_stream_config("NADROWSKI", str(cells[0]))
    assert cfg.hw.device.type == "cpu", "streaming config must be forced onto CPU (device coherence)"
    plan = plan_stream(cfg, 0.05)

    t_scale = cfg.rescale_params["t_scale"][0]
    assert plan.subsample_factor == max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    assert plan.steady_steps == cfg.steady_idx
    assert plan.total_steps == plan.steady_steps + plan.n_obs * plan.subsample_factor
    assert plan.dt_nd == cfg.dt_nd_min
    assert plan.n_channels == 1 and plan.state_dep_drift is True
    # x_scale rides on cfg.hw.dtype (float32), matching how generate_observations builds rescale_gt --
    # so compare with a float32-scale tolerance, not float64-exact.
    x_scale_gt = cfg.rescale_params["x_scale"][0]
    assert abs(plan.x_scale - x_scale_gt) <= 1e-5 * abs(x_scale_gt)

def test_simulate_dispatch_streams_chunks_and_a_cancel_is_not_an_error():
    """dispatch(provide_stream=True) must inject the chunk emitter + stop flag, deliver frames to
    on_chunk, and a cancel of a streaming run must land as `cancelled` (not `error`)."""
    import numpy as np
    from core.gui.streams import WorkerCancelled

    app = _app()

    class P(BasePanel):
        pass

    panel = P()
    chunks = []
    started = {"v": False}
    outcome = {"cancelled": 0, "error": 0}

    def streamer(emit_chunk=None, should_stop=None):
        i = 0
        while True:
            if should_stop is not None and should_stop():
                raise WorkerCancelled()
            emit_chunk(np.array([[float(i), float(i)]], dtype=np.float64))
            started["v"] = True
            print(f"frame {i}")                            # a write() checkpoint + lets the pump tick
            time.sleep(0.01)
            i += 1

    panel.dispatch(streamer, provide_stream=True, on_chunk=chunks.append)
    for w in panel._workers:
        w.signals.cancelled.connect(lambda: outcome.__setitem__("cancelled", outcome["cancelled"] + 1))
        w.signals.error.connect(lambda *_a: outcome.__setitem__("error", outcome["error"] + 1))

    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and not (started["v"] and chunks):
        app.processEvents()
        time.sleep(0.005)
    assert chunks, "no streamed chunks were delivered to on_chunk"
    assert panel._busy and BasePanel._active_cancel is not None

    panel._request_cancel()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and panel._busy:
        app.processEvents()
        time.sleep(0.005)
    _pump(app, 0.3)

    assert not panel._busy, "panel stuck busy after cancelling a stream"
    assert outcome["cancelled"] == 1 and outcome["error"] == 0, outcome
    assert "Run cancelled." in panel.log_pane.toPlainText()

def test_simulate_panel_is_wired_and_navigable():
    """The 4th home button is live: the Simulate section is registered, navigable, and its panel is in
    the persistence sweep."""
    from core.gui.main_window import MainWindow
    from core.gui.panels.simulate_panel import SimulatePanel

    _app()
    w = MainWindow()
    assert w.panel(SimulatePanel) is not None
    assert "Simulate" in w._section_index
    w.nav.go_to(w._section_index["Simulate"])
    assert w.nav.stack.currentIndex() == w._section_index["Simulate"]
    assert any(isinstance(p, SimulatePanel) for p in w._all_panels())

def test_simulate_settings_round_trip():
    from core.gui import settings as st
    from core.gui.panels.simulate_panel import SimulatePanel

    _app()
    _temp_settings()
    try:
        sp = SimulatePanel()
        sp.tobs.setText("2.5")
        sp.fps.setText("24")
        sp.frame_steps.setText("1234")
        if sp.cell_picker.combo.count():
            sp.cell_picker.combo.setCurrentIndex(sp.cell_picker.combo.count() - 1)
        want_cell = sp.cell_picker.key()
        want_model = sp.model_combo.currentText()

        qs = st.settings()
        sp.save_settings(qs)
        qs.sync()

        sp2 = SimulatePanel()
        assert sp2.tobs.value() == 2.5
        assert sp2.fps.value() == 24
        assert sp2.frame_steps.value() == 1234
        assert sp2.model_combo.currentText() == want_model
        assert sp2.cell_picker.key() == want_cell
    finally:
        st.use_ini_file(None)

def test_export_stride_maps_sample_rate_to_video_fps():
    from core.gui.panels.simulate_export import estimate_frame_count, export_stride
    assert export_stride(1000.0, 30.0) == 33
    assert export_stride(1000.0, 10.0) == 100
    assert export_stride(1000.0, 0.0) == 1                      # zero/neg fps guard -> stride 1
    assert estimate_frame_count(300, 100) == 3                  # range(99, 300, 100) -> 99,199,299

def test_export_animation_writes_a_readable_gif():
    """A real GIF round-trip: render a tiny series, then read it back with imageio."""
    import os
    import tempfile
    # import the app module (torch/pyqtgraph/matplotlib) BEFORE imageio -- the OMP-safe order.
    from core.gui.panels.simulate_export import export_animation
    path = os.path.join(tempfile.mkdtemp(), "anim.gif")
    export_animation(_tiny_series(), path, **_export_kwargs())
    assert os.path.getsize(path) > 0
    import imageio
    frames = imageio.mimread(path)
    assert len(frames) >= 2, "expected a multi-frame gif"
    assert frames[0].shape[0] % 2 == 0 and frames[0].shape[1] % 2 == 0, "exported frame dims must be even"

def test_export_animation_writes_a_readable_mp4_when_ffmpeg_is_available():
    import os
    import tempfile
    from core.gui.panels.simulate_export import export_animation, ffmpeg_available
    if not ffmpeg_available():
        return                                                 # skip, don't fail, on a bare-pip env
    path = os.path.join(tempfile.mkdtemp(), "anim.mp4")
    export_animation(_tiny_series(), path, **_export_kwargs())
    assert os.path.getsize(path) > 0
    import imageio
    r = imageio.get_reader(path)
    try:
        frame = r.get_next_data()
    finally:
        r.close()
    assert frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0, "H.264 needs even frame dims"

def test_export_animation_removes_the_partial_file_on_failure():
    """A cancel/error mid-export must not leave a half-written file (cleanup is in a finally)."""
    import os
    import tempfile
    import core.gui.panels.simulate_export as se
    path = os.path.join(tempfile.mkdtemp(), "bad.gif")

    real = se.gaussian_field
    calls = {"n": 0}

    def boom(*a, **k):                                          # 1st call = field0 (ok); 2nd = 1st frame -> raise
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("boom mid-loop")
        return real(*a, **k)

    se.gaussian_field = boom
    try:
        se.export_animation(_tiny_series(), path, **_export_kwargs())
    except RuntimeError:
        pass
    finally:
        se.gaussian_field = real
    assert not os.path.exists(path), "a failed export must not leave a partial file"

def test_simulate_panel_records_chunks_and_gates_the_save_button():
    import numpy as np
    from core.gui.panels.simulate_panel import SimulatePanel

    _app()
    p = SimulatePanel()
    assert not p.btn_save_video.isEnabled(), "save must be disabled before any recording"
    p._on_chunk(np.array([[0.0, 0.1], [1e-3, 0.2]]))
    p._on_chunk(np.array([[2e-3, 0.3]]))
    assert len(p._record) == 2
    p.refresh_local_gates()
    assert p.btn_save_video.isEnabled(), "save must enable once a recording exists"
    p._record = []
    p.refresh_local_gates()
    assert not p.btn_save_video.isEnabled(), "save must disable again when the recording is cleared"


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
