import math
from typing import Any

import torch
from tqdm import tqdm

from core import config


def _step_iter(n: int, batch_size: int):
    """tqdm wrapper for the per-step loop with overhead-minimizing settings.

    `miniters` controls how often tqdm checks for a display refresh.
    With ~1% of total per check, the per-iteration cost is just a counter
    increment, which is essentially free even at 2.4M iterations.

    This bar runs in the GUI too, not just the CLI: its rendered `it/s` is what drives the GUI's
    "Solver Performance" meter, and its percentage is the only thing that moves during a training
    iteration (which takes ~10s, so the top-level bar looks frozen between ticks). The GUI finds it by
    its desc -- hence config.SOLVER_BAR_DESC rather than a literal -- and shows it in a dedicated widget
    rather than as a progress row. See core/gui/widgets/progress_pane.py.
    """
    return tqdm(
        range(n - 1),
        desc=f"{config.SOLVER_BAR_DESC} (batch={batch_size})",
        leave=False,
        mininterval=1.0,
        miniters=max(1000, n // 100),
    )


def _step_bar(n: int, batch_size: int):
    """The same bar as :func:`_step_iter`, but driven manually via ``.update()``.

    The graphed path advances the time loop a CHUNK at a time, so it cannot wrap an iterable. Same
    desc and settings on purpose -- the GUI finds this bar by ``config.SOLVER_BAR_DESC`` and renders
    its it/s as the "Solver Performance" meter, so a graphed run must advance it by the chunk size or
    the meter reads 1/CHUNK of the truth.
    """
    return tqdm(
        total=n - 1,
        desc=f"{config.SOLVER_BAR_DESC} (batch={batch_size})",
        leave=False,
        mininterval=1.0,
        miniters=max(1000, n // 100),
    )


# --- CUDA Graph step capture ---------------------------------------------------------------------
# Keyed on everything the capture bakes in. NOT hung off the Solver class: trap X1 requires
# `sdeint.Solver` to be resolvable (and patchable) at call time, and a cache on a Solver instance
# would be useless anyway -- Solver is constructed once per TIME SEGMENT, so it would recapture
# constantly. A module-level dict is orthogonal to that seam.
#
# The PARAMETERS are static buffers copied into per call rather than captured by reference, which is
# what lets one graph serve every training batch at a given geometry. Capturing `params` directly
# would bake the address of one model's tensors and silently replay stale physics for the next.
_GRAPH_CACHE: dict = {}
_GRAPH_DISABLED = False          # set once if capture fails, so we do not retry every segment


def _graph_key(step, x0, force, dt, chunk):
    return (id(step), tuple(x0.shape), int(force.shape[1]), chunk,
            str(x0.dtype), str(x0.device), float(dt))


def _acquire_graph(step, params, x0, force, dt, sqrt_dt, chunk):
    """Capture (or fetch) a graph replaying ``chunk`` Euler steps. None => caller runs eager."""
    global _GRAPH_DISABLED
    if _GRAPH_DISABLED or not getattr(config, "SOLVER_CUDA_GRAPHS", True):
        return None
    if x0.device.type != "cuda" or not torch.cuda.is_available():
        return None
    key = _graph_key(step, x0, force, dt, chunk)
    ent = _GRAPH_CACHE.get(key)
    if ent is not None:
        return ent
    try:
        B, d = x0.shape
        n_ch = int(force.shape[1])
        kw = dict(dtype=x0.dtype, device=x0.device)
        x_s = torch.zeros((B, d), **kw)
        dW_s = torch.empty((B, d), **kw)
        f_s = torch.zeros((B, n_ch, chunk), **kw)
        out_s = torch.empty((chunk, B, d), **kw)
        p_s = tuple(p.clone() for p in params)

        # Warm-up on a side stream is required before capture; three passes is the documented
        # minimum that lets lazy allocations and any autotuning settle.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                xx = x_s.clone()
                for j in range(chunk):
                    dW_s.normal_()
                    xx = step(xx, f_s[:, :, j], dW_s, *p_s, dt, sqrt_dt)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cur = x_s
            for j in range(chunk):
                dW_s.normal_()
                cur = step(cur, f_s[:, :, j], dW_s, *p_s, dt, sqrt_dt)
                out_s[j] = cur
            # Chain the state forward INSIDE the graph, so consecutive chunks need no copy between
            # replays -- x_s is both the graph's input and its output.
            x_s.copy_(cur)

        if len(_GRAPH_CACHE) >= getattr(config, "SOLVER_GRAPH_CACHE_MAX", 8):
            _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
        ent = {"graph": g, "x": x_s, "dW": dW_s, "force": f_s, "out": out_s,
               "params": p_s, "chunk": chunk}
        _GRAPH_CACHE[key] = ent
        return ent
    except Exception as e:                     # noqa: BLE001 -- a solver must never die on a speedup
        _GRAPH_DISABLED = True
        print(f"[solver] CUDA graph capture unavailable ({type(e).__name__}: {e}); "
              f"falling back to the eager step loop for the rest of this process.", flush=True)
        return None


class Solver:
    def __init__(self):
        def euler(sde: Any, x0: torch.Tensor, ts: tuple[float, float], n: int, state_dep_drift: bool = False) -> torch.Tensor:
            """
            Explicit Euler-Maruyama SDE solver.

            Noise is assumed diagonal: `sde.g(...)` returns a (batch, d) vector of
            per-channel amplitudes, and the update is `x + f*dt + g*dW` elementwise.
            """
            x0 = x0.to(sde.device)

            t = torch.linspace(*ts, n, device=x0.device)
            dt = t[1].item() - t[0].item()
            sqrt_dt = math.sqrt(dt)

            batch_size, d = x0.shape

            # empty, not zeros: row 0 is assigned immediately below and _step_iter is range(n-1), so
            # the loop writes every remaining row. The zero-fill was a full memset of a buffer that
            # is overwritten in its entirety -- ~2.5 GB per segment, x segments x runs x 5000
            # training batches.
            xs = torch.empty((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0, :, :] = x0

            # Pre-allocated dW buffer reused every step.
            dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)

            if not state_dep_drift:
                g = sde.g()  # (batch, d)
                for i in _step_iter(n, batch_size):
                    x_curr = xs[i, :, :]
                    dW_buf.normal_()
                    eta = g * dW_buf * sqrt_dt
                    xs[i + 1, :, :] = x_curr + sde.f(x_curr, i) * dt + eta
            else:
                for i in _step_iter(n, batch_size):
                    x_curr = xs[i, :, :]
                    g = sde.g(x_curr)  # (batch, d)
                    dW_buf.normal_()
                    eta = g * dW_buf * sqrt_dt
                    xs[i + 1, :, :] = x_curr + sde.f(x_curr, i) * dt + eta

            return xs

        def euler_compiled(sde: Any, x0: torch.Tensor, ts: tuple[float, float], n: int, state_dep_drift: bool = False) -> torch.Tensor:
            """
            Euler-Maruyama via the model's `compiled_step`, replayed from a CUDA Graph where possible.

            The model must expose:
              - `compiled_step(x, force_step, dW, *params, dt, sqrt_dt) -> next_x`, a
                `@torch.jit.script` function
              - `compiled_params()` returning the params tuple, POSITIONALLY load-bearing
              - `f_pure(x, force_step)`

            Diagonal-noise assumption: the compiled step is responsible for computing
            its own g (constant or state-dependent) internally.

            WHY THE GRAPH. TorchScript removes PYTHON overhead; it does not remove kernel-LAUNCH
            overhead, and that is what this loop is bound by -- measured 54.87 us/step eager against
            6.65 us/step replayed at batch 2048, i.e. ~88% of the time was the CPU submitting ~3
            kernels per step to a GPU that was ~6% utilised. The docstring here used to claim
            "torch.compile + CUDA Graphs"; neither existed anywhere in the repo, so the fast path it
            described was never built. It is built now.

            NOT bit-identical to the eager loop, because the noise is drawn inside the captured
            region and the draw order differs. That is the same licence ``gen_obs`` already takes when
            it splits a batch. The ARITHMETIC is identical and is pinned bitwise by
            ``test_the_cuda_graph_step_matches_the_eager_step_bitwise`` (which supplies dW
            externally, so the comparison isolates the maths from the RNG).
            """
            x0 = x0.to(sde.device)

            t = torch.linspace(*ts, n, device=x0.device)
            dt = t[1].item() - t[0].item()
            sqrt_dt = math.sqrt(dt)

            batch_size, d = x0.shape

            # empty, not zeros -- same reasoning as the eager euler above.
            xs = torch.empty((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0, :, :] = x0

            step = sde.compiled_step
            params = sde.compiled_params()

            chunk = int(getattr(config, "SOLVER_GRAPH_CHUNK", 50))
            n_steps = n - 1
            ent = (_acquire_graph(step, params, x0, sde.force, dt, sqrt_dt, chunk)
                   if n_steps >= chunk else None)

            if ent is not None:
                # Copy THIS call's parameters into the captured statics. The graph replays whatever
                # is in these buffers, so this is what makes one capture serve every batch.
                for ps, p in zip(ent["params"], params):
                    ps.copy_(p)
                ent["x"].copy_(x0)
                f_s, out_s, g = ent["force"], ent["out"], ent["graph"]
                n_full = n_steps // chunk
                bar = _step_bar(n, batch_size)
                try:
                    for c in range(n_full):
                        lo = c * chunk
                        f_s.copy_(sde.force[:, :, lo:lo + chunk])
                        g.replay()
                        xs[lo + 1:lo + 1 + chunk, :, :].copy_(out_s)
                        bar.update(chunk)
                    # Tail: the last n_steps % chunk steps, eager, continuing from the graph's state.
                    x = ent["x"].clone()
                    dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)
                    for i in range(n_full * chunk, n_steps):
                        dW_buf.normal_()
                        x = step(x, sde.force[:, :, i], dW_buf, *params, dt, sqrt_dt)
                        xs[i + 1, :, :] = x
                        bar.update(1)
                finally:
                    bar.close()
                return xs

            dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)
            x = x0
            for i in _step_iter(n, batch_size):
                dW_buf.normal_()
                force_step = sde.force[:, :, i]
                x = step(x, force_step, dW_buf, *params, dt, sqrt_dt)
                xs[i + 1, :, :] = x

            return xs

        self.euler = euler
        self.euler_compiled = euler_compiled