"""
Generate a labelled dataset of top-down hair-bundle "videos" (multi-page TIFF) over a lambda sweep.

One TIFF stack per sampled ND parameter `lam`, filed as `lam_<value>.tif`. Each page is the SAME
top-down elliptical-blob field the Simulate section renders live (core/gui/widgets/live_hair_bundle.py),
as raw uint8 grayscale -- no axes, no trace subplot, no matplotlib. Everything else (k, f_max, tau, ...)
is held at the cell file's ground truth; only `lam` moves.

WHY THIS IS NOT JUST `export_animation` IN A LOOP
  1. AMPLITUDE. Both the live view (live_hair_bundle.py:157-160) and the video exporter
     (simulate_export.py:136-139) renormalize x against the CURRENT SLIDING WINDOW's min/max every
     frame, so every run's blob sweeps the full field no matter its real amplitude. For a
     lam-comparison dataset that erases the physics. Here ONE global nm->position map, calibrated
     across every sample of every lam (see `calibrate`), is shared by all videos, so amplitude and
     DC offset survive alongside frequency.
  2. SPEED. `NadrowskiSimulator` binds params via `torch.unbind(self._params, dim=1)`
     (nadrowski_simulator.py:16) and `Solver.euler` returns (n_steps, B, d) (sdeint.py:48), so ALL
     lam values integrate in one batched loop -- the same shape contract NadrowskiPrior._global_map
     already uses. Measured on an M1 Max over the full 59k-step run: 5.76 s/lam at B=1 vs 0.0135
     s/lam at B=512 (~430x). `run_simulation_stream` itself is batch-1 AND wall-clock-paced by
     `fps` (simulate_runner.py:246), so this file reuses its PIECES (_make_simulator,
     frame_time_grid, plan_stream) rather than the streaming entry point.
  3. TRANSIENT. The streaming plan SHOWS the settling transient (simulate_runner.py:80); a dataset
     wants steady state, so the first `plan.steady_steps` fine steps are integrated and dropped.

WHAT LAMBDA ACTUALLY DOES (measured, 512-point pilot over the full bounds)
  Peak-to-peak stays ~37-49 nm while the dominant frequency runs 41.2 Hz at lam=0.1 down to 1.0 Hz
  at lam=50 -- i.e. ~1/lam, since `dy = (...)/lam` sets the adaptation timescale. So the dominant
  learnable signal here is oscillation RATE, not swing. At lam=50 a 5 s clip is only ~5 cycles.

RENDERING IS A TABLE LOOKUP
  `gaussian_field` is separable and its cy is pinned to 0.5, so a frame depends on ONE scalar (the
  horizontal blob center). NQ quantized frames are precomputed once (0.14 s, 21.8 MB) and every
  video is then a fancy-index into that table -- 5.7 ms for 2500 frames. Writing the TIFF is the
  only real per-video cost (~4.1 s deflate).

COST (measured, defaults below): simulate ~70 s + render/write ~4.1 s/video over WORKERS processes
  => ~45 min and ~16.4 GB for 5000 videos.

Env: N_VIDEOS T_OBS_S VIDEO_FPS FRAME_H BATCH CHUNK SEED WORKERS OUT_DIR NQ COMPRESSION LAM_LIST
Run:  /opt/homebrew/Caskroom/miniforge/base/envs/biophys-arm/bin/python scripts/generate_bundle_videos.py
      N_VIDEOS=8 LAM_LIST=0.1,0.5,1,3.57,10,25,40,50 ... scripts/generate_bundle_videos.py   # smoke
"""
import csv
import os
import sys
import time
import _common; _common.enable_warnings()

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np
from PIL import Image

# --- CONFIG (env-overridable; see the Env: line above) --------------------------------------------
MODEL       = "NADROWSKI"
CELL        = "Resources/Cells/nadrowski/cell_2.txt"
N_VIDEOS    = int(os.environ.get("N_VIDEOS", 5000))
T_OBS_S     = float(os.environ.get("T_OBS_S", 5.0))       # seconds of steady state per video
VIDEO_FPS   = float(os.environ.get("VIDEO_FPS", 500.0))   # TIFF pages per second of signal
FRAME_H     = int(os.environ.get("FRAME_H", 64))          # field height in px; width = H * ASPECT
BATCH       = int(os.environ.get("BATCH", 512))           # lam values integrated per euler loop
CHUNK       = int(os.environ.get("CHUNK", 5900))          # fine steps per euler call (memory cap)
SEED        = int(os.environ.get("SEED", 0))
WORKERS     = int(os.environ.get("WORKERS", 8))
NQ          = int(os.environ.get("NQ", 2048))             # quantized blob positions in the LUT
COMPRESSION = os.environ.get("COMPRESSION", "tiff_deflate") or None
LAM_LIST    = os.environ.get("LAM_LIST", "")              # explicit comma-separated lams (smoke tests)
OUT_DIR     = os.path.abspath(os.path.expanduser(
    os.environ.get("OUT_DIR", "~/Desktop/videos")))

# Geometry, copied from LiveHairBundleView.__init__ so the dataset matches the on-screen view.
ASPECT, SIG_X, SIG_Y = 2.6, 0.10, 0.20
MARGIN  = min(ASPECT / 2.0 - 1e-3, 3.5 * SIG_X)
FRAME_W = max(2, round(FRAME_H * ASPECT))

# Percentiles (not min/max) for the global nm->position map, so one noise spike can't compress the
# whole dataset into the middle of the field.
CAL_PCT = (0.05, 99.95)

# core/config.py builds every Resources/ path from os.getcwd() at import time (config.py:61), with a
# __file__ fallback only if that path is missing. chdir so a run from anywhere resolves the repo's.
os.chdir(_REPO)


# --- SIMULATION -----------------------------------------------------------------------------------
def build_config():
    """The ground-truth SimConfig + StreamPlan for the cell, on CPU.

    `build_stream_config` is the only existing headless one-call GT builder for the real-time backend:
    it resolves the sibling bounds file, runs make_sim_config + load_and_validate_gt, and pins
    cfg.hw = cpu_device() (simulate_runner.py:66) -- required, since detect_device() returns MPS here
    and the batch-sequential Euler loop is CPU-optimal AND every tensor must share one device.
    `plan_stream` is pure and never touches cfg.t (the ~2.4M-point cached grid).
    """
    from core.gui.panels.simulate_runner import build_stream_config, plan_stream
    cfg = build_stream_config(MODEL, CELL)
    return cfg, plan_stream(cfg, T_OBS_S)


def sample_lambdas(cfg, n):
    """`n` lam values, scrambled-Sobol uniform over the BOUNDS FILE's (lo, hi).

    Matches NadrowskiPrior._global_map (nadrowski_prior.py:31-33), which Sobol-samples and scales
    linearly into the box -- config.REPARAM_LOG_PARAMS is deliberately [] so lam stays linear. Sobol
    rather than iid uniform so 5000 draws tile the range without clumps and gaps.

    NOT drawn from the persisted ND prior (file_manager.load_mix_dist): that is a stability-screened
    GMM whose lam marginal is heavily non-uniform.
    """
    import torch
    if LAM_LIST.strip():                       # explicit list wins (smoke tests / targeted reruns)
        return np.array([float(v) for v in LAM_LIST.split(",")], dtype=np.float64)
    lo, hi = cfg.params_dict["lam"][1]         # parse_bounds_file yields {name: (None, (lo, hi))}
    eng = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=SEED)
    u = eng.draw(n).squeeze(1).double().numpy()
    return lo + u * (hi - lo)


def simulate_all(cfg, plan, lams):
    """Integrate every lam and return steady-state displacement in nm, shape (n_obs, len(lams)).

    Mirrors run_simulation_stream's chunked loop (carry state forward, rebuild the force per chunk,
    globally-phased decimation) but batched over lam and with the transient dropped instead of shown.
    Returns (x_nm, ok) where `ok` masks members that stayed finite and inside the ND blow-up limit.
    """
    import copy
    import torch
    from core.Helpers import helpers
    from core.SBI import pipeline
    from core.Solvers import sdeint
    from core.gui.panels.simulate_runner import BLOWUP_ND_LIMIT, _make_simulator, frame_time_grid

    torch.manual_seed(SEED)
    dtype, device = cfg.hw.dtype, cfg.hw.device
    solver = sdeint.Solver()
    n_all = len(lams)
    lam_idx = list(cfg.params_dict.keys()).index("lam")

    x_nm = np.empty((plan.n_obs, n_all), dtype=np.float32)
    ok = np.ones(n_all, dtype=bool)

    for start in range(0, n_all, BATCH):
        sl = slice(start, min(start + BATCH, n_all))
        b = sl.stop - sl.start

        params = plan.params_tensor.repeat(b, 1).clone()
        params[:, lam_idx] = torch.tensor(lams[sl], dtype=dtype, device=device)
        inits = plan.inits_tensor.repeat(b, 1).clone()

        # _make_simulator reads params/inits off the plan; shallow-copy so the shared plan is intact.
        bplan = copy.copy(plan)
        bplan.params_tensor, bplan.inits_tensor = params, inits
        sim = _make_simulator(bplan, dtype, device)

        forcing_b = plan.forcing_gt.repeat(b, 1)
        rescale_b = plan.rescale_gt.repeat(b, 1)

        curr, t_now, fine_idx, filled = inits, 0.0, 0, 0
        finite = torch.ones(b, dtype=torch.bool)
        with torch.no_grad():
            for step in range(0, plan.total_steps, CHUNK):
                m = min(CHUNK, plan.total_steps - step)
                t_chunk = frame_time_grid(t_now, m, plan.dt_nd, dtype, device)
                # cell_2 is unforced (amp=freq=0) so this is all zeros -- built anyway to keep the
                # (B, n_channels, T) shape contract honest for forced cells.
                sim.sde.force = pipeline.build_nondim_sin_force_tensor(
                    forcing_b, t_chunk, rescale_b, plan.forcing_idx, plan.rescale_idx)
                res = solver.euler(sim.sde, curr, (t_chunk[0].item(), t_chunk[-1].item()), m + 1,
                                   state_dep_drift=plan.state_dep_drift)
                curr = res[-1]

                # Per-member validity: one divergent lam must not poison the whole batch.
                bad = ~torch.isfinite(res).all(dim=0).all(dim=-1) | (res.abs().amax(dim=(0, 2)) > BLOWUP_ND_LIMIT)
                finite &= ~bad

                # res[0] duplicates `curr`; res[1:] are the m new fine samples. Keep only the tail
                # past the transient, decimated on the GLOBAL phase so chunks splice seamlessly.
                hb = res[1:, :, 0]
                first = (-fine_idx) % plan.subsample_factor
                keep = max(first, plan.steady_steps - step - 1)
                if keep < m and filled < plan.n_obs:
                    dec = hb[keep::plan.subsample_factor]
                    take = min(dec.shape[0], plan.n_obs - filled)
                    if take > 0:
                        x_nm[filled:filled + take, sl] = helpers.rescale(
                            dec[:take], plan.x_scale, plan.x_offset).cpu().numpy()
                        filled += take
                t_now += m * plan.dt_nd
                fine_idx += m

        if filled < plan.n_obs:                # short batch -> mark unusable rather than ship zeros
            finite[:] = False
        ok[sl] = finite.numpy()
    return x_nm, ok


def calibrate(x_nm):
    """The ONE global nm -> [0,1] position map shared by every video. Returns (lo, hi) in nm."""
    lo, hi = np.percentile(x_nm, CAL_PCT)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


# --- RENDERING ------------------------------------------------------------------------------------
def build_lut():
    """(NQ, FRAME_H, FRAME_W) uint8 table of the top-down field at NQ quantized blob positions.

    `gaussian_field` is separable and its cy is pinned to 0.5, so the frame is a function of the
    single scalar `hb_center(x0n, ...)`. Precomputing it turns per-frame rendering into an array
    index. NQ=2048 levels across FRAME_W px is sub-pixel, so the quantization is invisible.

    Imported lazily: live_hair_bundle pulls pyqtgraph (-> Qt), which the render workers must not pay
    for -- the parent builds this once and hands the array down.
    """
    from core.gui.widgets.live_hair_bundle import gaussian_field, hb_center
    gx = np.linspace(0.0, ASPECT, FRAME_W)
    gy = np.linspace(0.0, 1.0, FRAME_H)
    qs = np.linspace(0.0, 1.0, NQ)
    # .T + the (H, W) orientation matches the ImageItem/imshow convention (simulate_export.py:107).
    return np.stack([(gaussian_field(hb_center(q, ASPECT, MARGIN), 0.5, gx, gy, SIG_X, SIG_Y).T
                      * 255.0).round().astype(np.uint8) for q in qs])


_W = {}          # per-worker state, populated by _init_worker (spawn-safe: no torch, no Qt)


def _init_worker(lut, out_dir, x_lo, x_hi, stride, compression, sample_hz):
    _W.update(lut=lut, out_dir=out_dir, x_lo=x_lo, x_hi=x_hi, stride=stride,
              compression=compression, sample_hz=sample_hz)


def _render_one(job):
    """Write one lam's TIFF stack and return its manifest row."""
    lam, x = job                                  # x: (n_obs,) float32 nm at sample_hz
    lut, nq = _W["lut"], _W["lut"].shape[0]
    lo, hi = _W["x_lo"], _W["x_hi"]

    frames_x = x[_W["stride"] - 1::_W["stride"]]  # decimate 1000 Hz -> VIDEO_FPS
    x0n = np.clip((frames_x - lo) / (hi - lo), 0.0, 1.0)     # GLOBAL map -- never per-video
    frames = lut[np.rint(x0n * (nq - 1)).astype(np.int32)]

    name = f"lam_{lam:09.6f}.tif"
    path = os.path.join(_W["out_dir"], name)
    n = 1
    while os.path.exists(path):                   # Sobol makes this ~impossible; never overwrite
        n += 1
        path = os.path.join(_W["out_dir"], f"lam_{lam:09.6f}_{n}.tif")
        name = os.path.basename(path)

    ims = [Image.fromarray(f) for f in frames]
    kw = {"compression": _W["compression"]} if _W["compression"] else {}
    ims[0].save(path, save_all=True, append_images=ims[1:], **kw)

    d = x - x.mean()                              # dominant frequency off the full-rate series
    mag = np.abs(np.fft.rfft(d))
    freqs = np.fft.rfftfreq(len(d), 1.0 / _W["sample_hz"])
    return dict(filename=name, lam=f"{lam:.6f}",
                p2p_nm=f"{float(x.max() - x.min()):.4f}",
                dom_freq_hz=f"{float(freqs[mag.argmax()]):.4f}",
                x_min_nm=f"{float(x.min()):.4f}", x_max_nm=f"{float(x.max()):.4f}",
                n_frames=len(frames), bytes=os.path.getsize(path))


# --- MAIN -----------------------------------------------------------------------------------------
def main():
    import multiprocessing as mp
    from tqdm import tqdm
    from core.config import DT_EXP_S
    from core.gui.panels.simulate_export import export_stride

    os.makedirs(OUT_DIR, exist_ok=True)
    sample_hz = 1.0 / DT_EXP_S
    stride = export_stride(sample_hz, VIDEO_FPS)

    cfg, plan = build_config()
    lams = sample_lambdas(cfg, N_VIDEOS)
    n_frames = len(range(stride - 1, plan.n_obs, stride))
    print(f"[cfg]   {MODEL} {os.path.basename(CELL)} | T_obs {T_OBS_S:g}s | "
          f"{plan.total_steps} fine steps (burn-in {plan.steady_steps}) | "
          f"subsample {plan.subsample_factor} -> {plan.n_obs} samples @ {sample_hz:g} Hz")
    print(f"[out]   {len(lams)} videos -> {OUT_DIR}")
    print(f"[video] {n_frames} pages @ {VIDEO_FPS:g} fps (stride {stride}) | "
          f"{FRAME_H}x{FRAME_W} px uint8 | compression={COMPRESSION}")

    t0 = time.perf_counter()
    x_nm, ok = simulate_all(cfg, plan, lams)
    print(f"[sim]   {time.perf_counter() - t0:.1f}s | {int(ok.sum())}/{len(lams)} usable")
    if not ok.all():
        bad = ", ".join(f"{v:.4f}" for v in lams[~ok][:12])
        print(f"[sim]   dropped (diverged / non-finite): {bad}{' ...' if (~ok).sum() > 12 else ''}")

    x_lo, x_hi = calibrate(x_nm[:, ok])
    print(f"[scale] global map {x_lo:.2f} .. {x_hi:.2f} nm -> blob position (pct {CAL_PCT})")

    lut = build_lut()
    print(f"[lut]   {lut.shape} {lut.nbytes / 1e6:.1f} MB")

    jobs = [(float(lams[i]), x_nm[:, i]) for i in range(len(lams)) if ok[i]]
    rows = []
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS, initializer=_init_worker,
                  initargs=(lut, OUT_DIR, x_lo, x_hi, stride, COMPRESSION, sample_hz)) as pool:
        for row in tqdm(pool.imap_unordered(_render_one, jobs, chunksize=4),
                        total=len(jobs), desc="Writing TIFFs"):
            rows.append(row)
    dt = time.perf_counter() - t0
    total = sum(r["bytes"] for r in rows)
    print(f"[write] {dt:.1f}s | {len(rows)} files | {total / 1e9:.2f} GB "
          f"({total / max(1, len(rows)) / 1e6:.2f} MB/video)")

    rows.sort(key=lambda r: float(r["lam"]))
    man = os.path.join(OUT_DIR, "manifest.csv")
    with open(man, "w", newline="") as f:
        f.write(f"# model={MODEL} cell={CELL} seed={SEED} T_obs_s={T_OBS_S:g}\n")
        f.write(f"# video_fps={VIDEO_FPS:g} stride={stride} n_frames={n_frames} "
                f"frame_hw={FRAME_H}x{FRAME_W} compression={COMPRESSION}\n")
        f.write(f"# global_scale_nm=[{x_lo:.6f},{x_hi:.6f}] calib_pct={CAL_PCT} "
                f"aspect={ASPECT} sigma_par={SIG_X} sigma_perp={SIG_Y} margin={MARGIN:.6f} nq={NQ}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[done]  manifest -> {man}")


if __name__ == "__main__":
    main()
