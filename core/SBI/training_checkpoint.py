"""Resumable checkpointing for ``pipeline.gen_training_data`` (backlog C-11).

WHY. A production run is ``TRAINING_NUM_RUNS=5000`` batches x ``hw.batch_size=2048`` rows -- days of
simulation accumulating in host RAM with nothing on disk until it finishes. The 2026-08-10/11 OOM work
removed the failure modes that had actually bitten, but a driver reset, a reboot, a power cut or the
user closing the GUI window still cost the whole run.

THE INVARIANT THE WHOLE DESIGN RESTS ON:

    a checkpoint describes batches [0, k), with the RNG state captured at the TOP of iteration k,
    before that iteration has consumed any randomness.

Snapshot-and-restore, never replay: per-batch RNG consumption is NOT a fixed quantity. ``_gen_obs_retry``
and ``_rows_with_oom_retry`` re-draw SDE noise after an OOM and ``gen_obs``'s predictive split re-blocks
it, so how much randomness a batch consumes depends on what else the desktop was doing. Anything that
tries to fast-forward by counting draws is wrong on exactly the runs this exists to rescue.

WHAT IS DELIBERATELY NOT PERSISTED
  * ``pipeline._BUDGET_CAP_ELEMENTS`` / ``_budget_clean_runs``. Process-local by design: the right cap
    depends on what is on the card NOW, and a resumed run must re-learn it against the current desktop.
  * numpy's RNG state. ``np.random`` is used exactly once in this path, for ``inits`` before the loop
    (trap X8 -- ``torch.manual_seed`` does not touch numpy). The TENSOR is persisted instead, which is
    both smaller and immune to anything else in the process having drawn from numpy meanwhile.
  * The Sobol engine. ``SobolEngine(dimension=2, scramble=True)`` consumes the torch global RNG AT
    CONSTRUCTION, and ``_draw_and_filter``'s accept count depends on the geometry, so the schedule
    cannot be re-derived from a seed. The drawn arrays are persisted and the engine is never rebuilt.

LAYOUT.  ``<root>/train_<digest12>/``
    header.pt      write-once: identity + the (t_scale, T) schedule + inits + V + the bijection probe
    state.pt       rewritten atomically; its ``batches_done`` is THE COMMIT POINT
    state.prev.pt  one generation back, a few KB, for the case where state.pt is lost mid-write
    shards/x_<from>_<to>.pt, th_<from>_<to>.pt    write-once row blocks, never mutated after commit

The directory NAME is a digest of the declared identity, so "same config" resolves to the same place by
construction and two different configs can never share one. The digest is only a ROUTER: ``verify``
re-checks the header field by field and names the field that differs, so a hand-moved directory or a
hash collision produces a diagnosable message rather than "digest mismatch".

ORDERING, which is what makes a crash safe:
    1. write + fsync the shards for [prev, k)     (write-once names; a crash leaves orphans)
    2. copy state.pt -> state.prev.pt
    3. atomically replace state.pt with batches_done = k
Resume only ever loads shards covered by ``batches_done``, so orphans from step 1 are ignored. Shards
are durable before the state that references them, so no commit can point at data still in the page
cache.

Do not ``print()`` between steps 1 and 3 under the GUI: every write funnels through
``gui.streams._SignalStream.write``, which calls ``CancelToken.check()`` and would raise mid-commit.
"""
import hashlib
import json
import shutil
from pathlib import Path

import torch

from core import config
from core.Helpers.file_manager import atomic_torch_save

# Bumped when the on-disk layout changes in a way an older/newer PRISM cannot read. It rides in the
# identity, so a bump routes to a fresh directory rather than misreading an existing one.
CHECKPOINT_FORMAT = 1

_HEADER = "header.pt"
_STATE = "state.pt"
_STATE_PREV = "state.prev.pt"
_SHARDS = "shards"


# ── identity ─────────────────────────────────────────────────────────────────────────────────────
def _canonical(obj):
    """JSON-able form of an identity value. Tensors become nested lists, tuples become lists, so the
    digest does not depend on whether a caller passed a tuple or a list for the same numbers."""
    if isinstance(obj, torch.Tensor):
        return _canonical(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return repr(obj)                     # repr, not float: exact and stable across json round-trips
    return str(obj)                          # torch.device, torch.dtype, Path, ...


def identity_digest(identity: dict) -> str:
    """12 hex chars over the canonicalised identity. Same shape as orchestrator._gmm_fingerprint."""
    blob = json.dumps(_canonical(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def resolve_dir(identity: dict, root=None) -> Path:
    """The directory this identity's checkpoint lives in. Pure; creates nothing."""
    root = Path(root) if root is not None else config.CHECKPOINT_PATH
    return root / f"train_{identity_digest(identity)}"


def bijection_probe(theta_transform, dim: int, n: int = 7, device=None) -> torch.Tensor:
    """A fixed latent grid pushed through ``theta_transform``, as float64 on the CPU.

    The identity check for the box + rotation. NOT a hash of V: V is float, so hashing its bytes is
    brittle across a torch build, and V alone would miss a changed BOX. Pushing a deterministic grid
    through the whole transform catches any change to either, whatever the transform is made of
    (ComposeTransform, log box, rotation, anything added later), and reports a real numeric distance
    when it fails instead of "the hashes differ".

    ``device`` MUST be the device the transform's own tensors live on. A rotated transform holds V
    (``reparam.OrthogonalTransform.M``), and `x @ M` with x on the CPU and M on CUDA is a hard
    RuntimeError, not a silent promotion -- so a CPU-built grid crashes `build_posterior` the moment
    the rotation is on and the run is on a GPU. That is precisely the retrain's configuration, and no
    CPU-only test can reach it; this signature exists because the smoke train found it.

    The RESULT always comes back on the CPU, so a probe compares equal across machines and is
    storable as-is.
    """
    if theta_transform is None:
        return torch.zeros(0, dtype=torch.float64)
    z = torch.linspace(-1.5, 1.5, n, dtype=torch.float64).unsqueeze(1).repeat(1, dim)
    with torch.no_grad():
        out = theta_transform(z.to(dtype=torch.float32, device=device or "cpu"))
    return out.detach().to(torch.float64).cpu()


# ── lifecycle ────────────────────────────────────────────────────────────────────────────────────
def peek(path) -> dict | None:
    """The committed state, cheaply: ``{batches_done, complete, ...}``. None when there is no usable
    checkpoint here. Never raises -- a corrupt or half-written state falls back to ``state.prev.pt``,
    and if that is unreadable too the caller simply starts fresh."""
    path = Path(path)
    for name in (_STATE, _STATE_PREV):
        f = path / name
        if not f.exists():
            continue
        try:
            st = torch.load(str(f), map_location="cpu", weights_only=False)
        except Exception:                    # noqa: BLE001 -- torn write, truncation, version skew
            continue
        if isinstance(st, dict) and "batches_done" in st:
            st = dict(st)
            st["_state_file"] = name
            return st
    return None


def create(path, identity: dict, *, schedule_t_scales, schedule_Ts, inits, V, probe,
           run_size: int, n_runs: int) -> None:
    """Write the write-once header and a zeroed state. Called BEFORE the first simulation.

    Doing this up front is the cheapest insurance in the feature: a read-only Resources/, a
    permissions problem or a disk with no room surfaces in the first seconds rather than on day three
    when the first cadence write is attempted.
    """
    path = Path(path)
    (path / _SHARDS).mkdir(parents=True, exist_ok=True)
    atomic_torch_save({
        "format": CHECKPOINT_FORMAT,
        "identity": identity,
        # The stratification schedule. Persisted, never re-derived -- see the module docstring.
        "batch_t_scales": schedule_t_scales.detach().cpu(),
        "batch_Ts": schedule_Ts.detach().cpu(),
        "inits": inits.detach().cpu(),                     # numpy-drawn; trap X8
        "V": None if V is None else V.detach().cpu(),
        "probe": probe,
        "run_size": int(run_size),
        "n_runs": int(n_runs),
    }, path / _HEADER)
    atomic_torch_save({"batches_done": 0, "complete": False, "rng": None}, path / _STATE)


def read_header(path) -> dict:
    return torch.load(str(Path(path) / _HEADER), map_location="cpu", weights_only=False)


def verify(path, identity: dict, probe=None, *, probe_atol: float = 1e-6) -> dict:
    """Validate a checkpoint against the config that wants to resume it; return its header.

    Field by field, naming the field and BOTH values, in the voice of
    ``orchestrator._assert_mode_matches``. The digest already routed us here, so anything caught below
    is a hand-moved directory, a format skew, or a collision -- all of which deserve a message that
    says what is wrong rather than "digest mismatch".
    """
    path = Path(path)
    header = read_header(path)
    if header.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Training checkpoint at {path} was written in format {header.get('format')!r}; this "
            f"PRISM reads format {CHECKPOINT_FORMAT}. Delete the directory to start fresh.")
    stored = header.get("identity", {})
    for key in sorted(set(stored) | set(identity)):
        want, got = _canonical(identity.get(key)), _canonical(stored.get(key))
        if want != got:
            raise ValueError(
                f"Training checkpoint at {path} does not match this configuration: '{key}' was "
                f"{got!r} when the checkpoint was written, and is {want!r} now. Resuming would mix "
                f"rows generated under two different settings. Change it back, or delete the "
                f"directory to start fresh.")
    if probe is not None:
        got = header.get("probe")
        if got is None or got.shape != probe.shape:
            raise ValueError(
                f"Training checkpoint at {path} carries no comparable parameter-transform probe "
                f"(stored {None if got is None else tuple(got.shape)}, expected "
                f"{tuple(probe.shape)}). It cannot be resumed safely.")
        if not torch.allclose(got, probe, rtol=probe_atol, atol=probe_atol):
            raise ValueError(
                f"Training checkpoint at {path} was generated under a DIFFERENT parameter transform "
                f"(max|diff| = {float((got - probe).abs().max()):.3g}). The stored targets are latent, "
                f"so every batch after the resume point would be expressed in another coordinate. "
                f"build_posterior reuses the checkpoint's rotation V, so this means the BOX changed "
                f"(bounds, or a log-box setting) since the run started.")
    return header


def _sibling_diffs(identity: dict, root=None):
    """Yield ``(dir, batches_done, diff_fields, stored_identity)`` for every OTHER committed
    checkpoint under ``root``.

    Extracted so describe_siblings and near_miss_siblings cannot drift: they ask the same question of
    the same directories and differ only in what they do with the answer.
    """
    root = Path(root) if root is not None else config.CHECKPOINT_PATH
    if not root.is_dir():
        return
    mine = resolve_dir(identity, root).name
    for d in sorted(root.glob("train_*")):
        if d.name == mine:
            continue
        st = peek(d)
        if not st or not st.get("batches_done"):
            continue
        try:
            stored = read_header(d).get("identity", {})
        except Exception:                    # noqa: BLE001 -- an unreadable sibling is not our problem
            continue
        diff = [k for k in sorted(set(stored) | set(identity))
                if _canonical(identity.get(k)) != _canonical(stored.get(k))]
        yield d, int(st["batches_done"]), diff, stored


def describe_siblings(identity: dict, root=None) -> str:
    """A one-line account of other checkpoints under ``root``, and the first identity field each one
    differs in. Turns the commonest silent restart -- 'I rebuilt the prior, so the digest changed and
    it started from zero' -- into a message that names the reason."""
    notes = [f"{d.name} ({done} batches, differs in {diff[0] if diff else 'nothing recorded'})"
             for d, done, diff, _ in _sibling_diffs(identity, root)]
    if not notes:
        return ""
    return (f"[checkpoint] {len(notes)} other checkpoint(s) exist and do NOT match this run: "
            + "; ".join(notes))


def checkpoints_using_prior(fingerprint: str, root=None) -> list:
    """``[(name, batches_done)]`` for every committed checkpoint whose prior has ``fingerprint``.

    A checkpoint is keyed on the prior's fitted GMM, so the prior FILE is the only thing that can
    reproduce its directory name. Overwrite that file and the checkpoint becomes unresumable -- there
    is no error and no warning, and several GiB of simulation simply stop being reachable. That has
    now happened twice: 884 batches on 2026-08-27 (a prior built and never saved) and 3989 batches on
    2026-08-28 (prior_08282026.pt overwritten with a different distribution under the same name).
    """
    root = Path(root) if root is not None else config.CHECKPOINT_PATH
    if not fingerprint or not root.is_dir():
        return []
    out = []
    for d in sorted(root.glob("train_*")):
        st = peek(d)
        if not st or not st.get("batches_done"):
            continue
        try:
            stored = read_header(d).get("identity", {})
        except Exception:                    # noqa: BLE001
            continue
        if stored.get("prior_fingerprint") == fingerprint:
            out.append((d.name, int(st["batches_done"])))
    return sorted(out, key=lambda r: -r[1])


def near_miss_siblings(identity: dict, root=None) -> list:
    """Committed checkpoints that differ from ``identity`` in EXACTLY ONE field.

    ``[{"name", "batches", "field", "mine", "theirs"}]``, richest first.

    WHY ONE FIELD SPECIFICALLY. A run that shares every declared property but one is almost never a
    genuinely different experiment -- it is the same experiment with something nudged by accident,
    and the digest turns that nudge into a silent restart from zero. That has now happened three
    times on this project: a prior rebuilt rather than loaded (884 batches, unrecoverable because the
    prior was never saved), and twice more where a 3989-batch checkpoint was one field away from the
    run about to start. Two or more differing fields is a different question -- that usually IS a
    different experiment -- so widening this would make it noise and it would be ignored.
    """
    out = [{"name": d.name, "batches": done, "field": diff[0],
            "mine": identity.get(diff[0]), "theirs": stored.get(diff[0])}
           for d, done, diff, stored in _sibling_diffs(identity, root) if len(diff) == 1]
    return sorted(out, key=lambda r: -r["batches"])


# ── rows ─────────────────────────────────────────────────────────────────────────────────────────
def _shard(path, prefix: str, a: int, b: int) -> Path:
    return Path(path) / _SHARDS / f"{prefix}_{a:06d}_{b:06d}.pt"


def save(path, *, from_batch: int, batch_k: int, rng: dict, x_buf, th_buf, run_size: int) -> None:
    """Commit batches [from_batch, batch_k). See the module docstring for why the order matters.

    ``.clone()`` on the slices is LOAD-BEARING, not tidiness: ``torch.save`` of a slice VIEW serialises
    the entire underlying storage, so saving ``x_buf[a:b]`` directly writes the whole multi-GiB buffer
    every time -- hundreds of GiB over a run, presenting as "checkpointing got slow". ``.contiguous()``
    does not help, because a row-slice of a contiguous 2-D tensor is already contiguous and stays a view.
    """
    path = Path(path)
    (path / _SHARDS).mkdir(parents=True, exist_ok=True)
    lo, hi = from_batch * run_size, batch_k * run_size
    if hi > lo:
        atomic_torch_save(x_buf[lo:hi].clone(), _shard(path, "x", from_batch, batch_k))
        atomic_torch_save(th_buf[lo:hi].clone(), _shard(path, "th", from_batch, batch_k))
    prev = path / _STATE
    if prev.exists():
        shutil.copyfile(prev, path / _STATE_PREV)
    atomic_torch_save({"batches_done": int(batch_k), "complete": False, "rng": rng}, prev)


def mark_complete(path, batch_k: int) -> None:
    path = Path(path)
    st = peek(path) or {}
    atomic_torch_save({"batches_done": int(batch_k), "complete": True,
                       "rng": st.get("rng")}, path / _STATE)


def load_rows(path, batches_done: int, run_size: int):
    """Every committed row, in batch order, as ``(x, thetas)``. Orphan shards past ``batches_done``
    (a crash between the shard write and the state commit) are ignored by construction: this walks
    the recorded ranges, not the directory."""
    path = Path(path)
    xs, ths = [], []
    got = 0
    for f in sorted((path / _SHARDS).glob("x_*.pt")):
        a, b = (int(p) for p in f.stem.split("_")[1:3])
        if b > batches_done:
            continue
        th = _shard(path, "th", a, b)
        if not th.exists():
            raise ValueError(f"Training checkpoint at {path} is missing {th.name}, the targets for "
                             f"batches [{a}, {b}). Delete the directory to start fresh.")
        x_part = torch.load(str(f), map_location="cpu", weights_only=False)
        th_part = torch.load(str(th), map_location="cpu", weights_only=False)
        want = (b - a) * run_size
        if x_part.shape[0] != want or th_part.shape[0] != want:
            raise ValueError(
                f"Training checkpoint shard {f.name} holds {x_part.shape[0]} rows, not the "
                f"{want} its name claims. Delete the directory to start fresh.")
        xs.append(x_part)
        ths.append(th_part)
        got += b - a
    if got != batches_done:
        raise ValueError(f"Training checkpoint at {path} commits {batches_done} batches but only "
                         f"{got} are present on disk. Delete the directory to start fresh.")
    if not xs:
        return None, None
    return torch.cat(xs, dim=0), torch.cat(ths, dim=0)


# ── RNG ──────────────────────────────────────────────────────────────────────────────────────────
def rng_snapshot(device, chi_gen) -> dict:
    """Every stream ``gen_training_data`` consumes, captured at a batch boundary. A few KB of memcpy
    against a ~20 s batch, so it is taken EVERY iteration and the cadence write just uses the last
    one -- one code path for a scheduled checkpoint and for a cancel."""
    return {
        "cpu": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if getattr(device, "type", None) == "cuda" and torch.cuda.is_available() else None),
        "chi_gen": None if chi_gen is None else chi_gen.get_state(),
    }


def rng_restore(rng: dict, device, chi_gen) -> None:
    """Put the streams back exactly as ``rng_snapshot`` found them. A CUDA state cannot be restored
    onto a CPU run or onto a different device count, so that is refused rather than half-applied --
    the SDE noise would silently come from a different stream for the rest of the run."""
    if not rng:
        return
    torch.set_rng_state(rng["cpu"].to(torch.uint8).cpu() if torch.is_tensor(rng["cpu"]) else rng["cpu"])
    cuda = rng.get("cuda")
    if cuda is not None:
        if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
            raise ValueError("This training checkpoint carries CUDA RNG state but the run is on "
                             f"{getattr(device, 'type', device)!r}. Resume on the same device type.")
        if len(cuda) != torch.cuda.device_count():
            raise ValueError(f"This training checkpoint carries CUDA RNG state for {len(cuda)} "
                             f"device(s); this machine has {torch.cuda.device_count()}.")
        torch.cuda.set_rng_state_all(list(cuda))
    elif getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        raise ValueError("This training checkpoint was written on CPU but the run is on CUDA; the "
                         "SDE noise stream cannot be reconstructed. Resume on the same device type.")
    if chi_gen is not None and rng.get("chi_gen") is not None:
        chi_gen.set_state(rng["chi_gen"])
