"""Worker-thread side of the real-time Simulate section.

Builds a ground-truth ``SimConfig`` from a model + cell pick, then drives a *chunked* Euler-Maruyama
loop that streams the redimensionalized hair-bundle displacement (state index 0 in every model) to the
GUI thread one frame at a time.

Deliberately does NOT call ``orchestrator.generate_observations``: that allocates ``cfg.t`` (the cached
~2.4M-point ND grid, core/config.py) and simulates the whole horizon in one shot. Instead this mirrors
the per-segment loop in ``core/Simulator/simulator.py`` incrementally -- carrying state forward and
rebuilding the sinusoidal force per frame -- so memory stays flat and frames arrive continuously.

The runner is driven directly on ``sdeint.Solver().euler`` rather than through ``Simulator.simulate`` /
``__sols`` so it can advance a fixed ``frame_steps`` of fine EM steps per frame and yield between
them -- ``simulate`` runs the whole horizon in one call. (It also used to dodge the ``exit()`` those
called on a solver exception; that became simulator.SimulationError on 2026-07-28, so it is no longer
a reason -- see _make_simulator below.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core import cli, forcing, registry
from core.SBI import pipeline
from core.Solvers import sdeint
from core.config import BOUNDS_PATH, DT_EXP_S, VALID_LABELS, VALID_MODELS, cpu_device

# ND magnitude ceiling for the divergence guard: built-in ND states are O(1)-O(1e2); an arbitrary
# user-typed drift can blow up under fixed-step Euler-Maruyama, and NaN/inf silently flatlines the plot.
BLOWUP_ND_LIMIT = 1e6

from ..streams import WorkerCancelled


def build_stream_config(model: str, cell_path: str):
    """Build a ground-truth ``SimConfig`` from a model + cell pick, forced onto CPU.

    Mirrors the inference simulated path (``inference_tabs._run_simulated_inference``): the bounds file
    is RESOLVED from the cell by ``cli.resolve_bounds_for_cell`` (same-named sibling, else the model's
    shared ``master.txt``), so no separate bounds pick is needed. ``make_sim_config`` defaults ``hw``
    to ``detect_device()`` (CUDA on a capable box); the batch-1 sequential Euler loop is CPU-optimal
    AND every tensor here must share one device, so force CPU right after building. Raises on an
    unresolvable bounds file / out-of-bounds cell -- the panel catches it as a config error.
    """
    bounds_file = cli.resolve_bounds_for_cell(cell_path, model)
    if bounds_file is None:
        raise ValueError(
            f"No bounds file governs '{Path(cell_path).name}': tried the sibling "
            f"{BOUNDS_PATH / model.lower() / Path(cell_path).name} and the shared "
            f"{BOUNDS_PATH / model.lower() / cli.MASTER_BOUNDS_NAME}. Bounds declare which "
            f"parameters are inferred and in what order, so one is required.")
    spec = registry.get(model)
    if spec is not None and spec.is_user_model:
        labels = list(spec.labels)
    else:
        labels = VALID_LABELS[VALID_MODELS.index(model)]
    state_dep_drift = registry.state_dep_drift(model)
    cfg = cli.make_sim_config(model, labels, state_dep_drift, str(bounds_file))
    if spec is not None and spec.is_user_model and spec.compiled is not None:
        # The JSON (-> spec.compiled) and the emitted Bounds file must agree on the parameter NAME
        # ORDER -- torch.unbind binds columns positionally, so a hand-edited JSON over a stale Bounds
        # file would otherwise mis-bind values silently (wrong physics, no error).
        expected, actual = list(spec.compiled.param_names), list(cfg.params_dict.keys())
        if actual != expected:
            raise ValueError(
                f"Model '{model}' is out of sync with its bounds file: the definition uses "
                f"parameters {expected} but the bounds file lists {actual}. Re-save the model "
                "from the Settings model builder to regenerate its files.")
    cli.load_and_validate_gt(cfg, cell_path)
    cfg.hw = cpu_device()
    return cfg


@dataclass
class StreamPlan:
    """Everything the streaming loop needs, computed once up front from ``cfg`` + ``T_obs``.

    ``total_steps`` bounds the (finite) stream; it and ``subsample_factor`` mirror the arithmetic in
    ``orchestrator.generate_observations`` exactly, so a streamed trace matches the observation the SBI
    pipeline would build from the same cell.
    """
    dt_nd: float                    # fine ND step (== cfg.dt_nd_min); preserved across every chunk
    subsample_factor: int           # fine steps per displayed sample (fine ND -> experimental rate)
    steady_steps: int               # transient fine steps (cfg.steady_idx); shown, not discarded
    n_obs: int                      # steady-state display samples implied by T_obs
    total_steps: int                # steady_steps + n_obs * subsample_factor (finite stream length)
    n_channels: int                 # force channels (2 for Hopf's x/y, else 1)
    state_dep_drift: bool
    model: str
    x_scale: float
    x_offset: float
    forcing_gt: torch.Tensor        # (1, n_forcing)
    rescale_gt: torch.Tensor        # (1, n_rescale)
    forcing_idx: dict
    rescale_idx: dict
    params_tensor: torch.Tensor     # (1, n_params)
    inits_tensor: torch.Tensor      # (1, n_vars)
    user_spec: object = None        # registry.ModelSpec for user-defined models, else None


def plan_stream(cfg, t_obs_s: float) -> StreamPlan:
    """Pure: derive the streaming plan from a ground-truth ``cfg`` and an observation time (seconds).

    Does not touch ``cfg.t`` (the heavy allocation) and does not mutate ``cfg``.
    """
    factor = cfg.get_unit_conversion_factor("s")     # SI seconds -> cell time units
    t_obs_cell = t_obs_s * factor

    rescale_idx = cfg.rescale_idx
    forcing_idx = cfg.forcing_idx
    dtype, device = cfg.hw.dtype, cfg.hw.device

    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]],
                              dtype=dtype, device=device)
    rescale_gt = torch.tensor([[val for val, _ in cfg.rescale_params.values()]],
                              dtype=dtype, device=device)

    t_scale_gt = rescale_gt[:, rescale_idx["t_scale"]].item()
    dt_nd_min = cfg.dt_nd_min
    dt_nd_gt = cfg.dt_exp / t_scale_gt
    subsample_factor = max(1, round(dt_nd_gt / dt_nd_min))
    n_obs = int((t_obs_cell / t_scale_gt) / dt_nd_gt)             # == generate_observations' N_obs
    steady_steps = cfg.steady_idx
    total_steps = steady_steps + n_obs * subsample_factor

    x_scale = rescale_gt[:, rescale_idx["x_scale"]].item()
    x_offset = (rescale_gt[:, rescale_idx["x_offset"]].item() if "x_offset" in rescale_idx else 0.0)

    # User models drive ONE force channel per state variable (zeros where unforced); built-ins keep
    # the legacy 1-or-2-channel sinusoidal convention. forcing.n_force_channels is the shared rule.
    spec = registry.get(cfg.model)
    user_spec = spec if (spec is not None and spec.is_user_model) else None
    n_channels = forcing.n_force_channels(cfg.model, forcing_idx, cfg.inits_tensor.shape[1])

    return StreamPlan(
        dt_nd=dt_nd_min, subsample_factor=subsample_factor, steady_steps=steady_steps, n_obs=n_obs,
        total_steps=total_steps, n_channels=n_channels,
        state_dep_drift=cfg.state_dep_drift, model=cfg.model, x_scale=x_scale, x_offset=x_offset,
        forcing_gt=forcing_gt, rescale_gt=rescale_gt, forcing_idx=forcing_idx, rescale_idx=rescale_idx,
        params_tensor=cfg.params_tensor, inits_tensor=cfg.inits_tensor, user_spec=user_spec,
    )


def frame_time_grid(t_now: float, m: int, dt_nd: float, dtype=torch.float32,
                    device=torch.device("cpu")) -> torch.Tensor:
    """The ND time grid for one frame: ``m+1`` points spanning ``[t_now, t_now + m*dt_nd]``.

    The point count is load-bearing: ``sdeint.euler`` derives ``dt = (t1-t0)/(n-1)`` from ``linspace(t0,
    t1, n)`` (sdeint.py:42-43), so ``m+1`` points make the Euler step exactly ``dt_nd``; and the grid's
    last point equals the next frame's ``t_now``, so state carried forward stays time-continuous.
    """
    return torch.linspace(t_now, t_now + m * dt_nd, m + 1, dtype=dtype, device=device)


def _make_simulator(plan: StreamPlan, dtype, device):
    """Construct the per-model Simulator ONCE to get its ``.sde`` (reuses each subclass's positional
    ``*torch.unbind(params)`` construction, incl. BP steady-vs-full selection + init slicing).

    Every Simulator now raises on a construction failure (SimulationError for the built-ins,
    RuntimeError for UserSimulator), both ordinary Exceptions that ``Worker.run`` catches and turns
    into an error dialog. This used to need a ``except SystemExit`` translation, because the built-in
    subclasses answered a bad cell with ``print(); exit()`` -- a BaseException that Worker.run does
    NOT catch, so it killed the worker thread silently.
    """
    placeholder_force = torch.zeros((1, plan.n_channels, 2), dtype=dtype, device=device)
    placeholder_t = torch.zeros(2, dtype=dtype, device=device)
    if plan.user_spec is not None:
        return registry.make_user_simulator(plan.user_spec, plan.params_tensor, placeholder_force,
                                            plan.inits_tensor, placeholder_t, segs=1, batch_size=1,
                                            device=device)
    sim_cls = pipeline._sim_class(plan.model)
    return sim_cls(plan.params_tensor, placeholder_force, plan.inits_tensor, placeholder_t,
                   segs=1, batch_size=1, device=device)


def run_simulation_stream(cfg, t_obs_s: float, frame_steps: int = 2000, fps: float = 30.0,
                          *, emit_chunk=None, should_stop=None) -> None:
    """Stream the redimensionalized hair-bundle displacement one frame at a time.

    Each frame advances ``frame_steps`` fine EM steps (an integer multiple of ``subsample_factor`` so
    the decimation phase is stable), carries the state forward, decimates to the experimental sample
    rate, redimensionalizes, and emits a fresh ``(k, 2)`` float array of ``[t_seconds, x_displacement]``
    via ``emit_chunk``. ``should_stop`` (the cancel flag) is polled once per frame and unwinds the run
    cooperatively with ``WorkerCancelled`` -- raised BETWEEN frames, never inside a tqdm redraw, so no
    tqdm write-lock leaks. Wall-clock-paced to ``fps`` so it plays at a watchable rate and ends when the
    (finite) horizon is reached.
    """
    dtype, device = cfg.hw.dtype, cfg.hw.device
    plan = plan_stream(cfg, t_obs_s)
    if plan.total_steps <= 0:
        return

    frame_steps = max(plan.subsample_factor, frame_steps)
    frame_steps -= frame_steps % plan.subsample_factor          # snap to a whole number of samples/frame
    frame_dt = 1.0 / fps if fps and fps > 0 else 0.0

    sim = _make_simulator(plan, dtype, device)
    solver = sdeint.Solver()
    curr_inits = sim.inits                                        # correctly shaped (BP-steady sliced to 4)

    t_now = 0.0                 # running ND time (absolute), advanced by exactly frame_steps*dt_nd
    fine_idx = 0                # global fine-step counter, for a phase-stable decimation across frames
    sample_idx = 0              # global emitted-sample counter, for the seconds time axis (dt = DT_EXP_S)

    with torch.no_grad():
        step = 0
        while step < plan.total_steps:
            if should_stop is not None and should_stop():
                raise WorkerCancelled()
            frame_start = time.perf_counter()

            m = min(frame_steps, plan.total_steps - step)        # fine steps this frame
            t_chunk = frame_time_grid(t_now, m, plan.dt_nd, dtype, device)
            if plan.user_spec is not None:
                force_chunk = forcing.build_user_force_tensor(
                    plan.user_spec, plan.forcing_gt, t_chunk, plan.rescale_gt,
                    plan.forcing_idx, plan.rescale_idx)
            else:
                force_chunk = pipeline.build_nondim_sin_force_tensor(
                    plan.forcing_gt, t_chunk, plan.rescale_gt, plan.forcing_idx, plan.rescale_idx)
            sim.sde.force = force_chunk                           # mirror simulator.py:58 (NOT sim.force=)
            res = solver.euler(sim.sde, curr_inits,
                               (t_chunk[0].item(), t_chunk[-1].item()), m + 1,
                               state_dep_drift=plan.state_dep_drift)
            if not bool(torch.isfinite(res).all()) or bool(res.abs().max() > BLOWUP_ND_LIMIT):
                raise RuntimeError(
                    f"Simulation diverged near t ≈ {sample_idx * DT_EXP_S:.3g} s (state left the "
                    f"±{BLOWUP_ND_LIMIT:g} ND range or became NaN). Check the drift/noise "
                    "expressions, initial conditions, or reduce the forcing amplitude.")
            curr_inits = res[-1]
            t_now += m * plan.dt_nd
            step += m

            # res[0] == curr_inits (duplicated boundary); res[1:] are the m NEW fine samples.
            hb_nd = res[1:, 0, 0]                                 # hair-bundle displacement, this frame
            first = (-fine_idx) % plan.subsample_factor          # phase offset to keep decimation global
            hb_nd_dec = hb_nd[first::plan.subsample_factor]
            fine_idx += m

            k = hb_nd_dec.shape[0]
            if k and emit_chunk is not None:
                x_dim = plan.x_scale * hb_nd_dec.cpu().numpy() + plan.x_offset
                t_sec = (sample_idx + np.arange(k)) * DT_EXP_S
                emit_chunk(np.column_stack((t_sec, x_dim)).astype(np.float64))   # a FRESH array per frame
            sample_idx += k

            if frame_dt:
                time.sleep(max(0.0, frame_dt - (time.perf_counter() - frame_start)))
