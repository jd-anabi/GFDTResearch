"""Widen a pre-flag training checkpoint with the derived valid-flag block. One-shot, additive.

WHY THIS EXISTS. Section 11.3 item 1.2 adds a valid-flag channel beside each summary feature whose
value is a substituted sentinel. That is a change to the FEATURE SET, which normally means
re-simulating -- and the run behind `posterior_08232026` was 10.24M simulations over roughly five
days. It does not have to this time, because every flag is an exact equality test on a value the
checkpoint already stores (`_logp`'s sentinel is exactly log(1e-12); B7's no-secondary state is an
exact 0.0 in both of its columns). So the flags are DERIVABLE from the cache, and
`statistics.derive_valid_flags` is the same function the live pipeline calls -- the migrated rows are
bit-for-bit what a re-simulation would have produced for those columns.

WHY A NEW DIRECTORY RATHER THAN AN IN-PLACE EDIT. `summary_flags` is part of the checkpoint identity,
so the digest changes and this writes to a new directory. That is the design working, not an
inconvenience: a checkpoint stores conditioning ROWS, and a run whose summary block means something
different must never resume onto them. The source is left untouched, so an unmigrated re-run still
finds it and `posterior_08232026` remains reproducible.

WHAT IS COPIED VERBATIM, AND WHY IT MATTERS:
  * `V`, the Fisher rotation. Trap X10: its operating points come from an unseeded RNG, so a
    recomputed V would express the reused rows' LATENT targets in a different coordinate than the
    targets stored beside them. Copying it is a correctness requirement.
  * `probe`, the bijection probe. It pins the box the latent targets live in; the box is unchanged,
    so the stored probe stays valid and a later resume can still verify it.
  * the Sobol schedule (`batch_t_scales`, `batch_Ts`) and `inits`, neither of which is re-derivable
    (SobolEngine consumes the global RNG at construction; `inits` is drawn from NUMPY's, which torch seeds do not touch).

Env:
  SRC     source checkpoint directory. Default: the newest COMPLETE one that has no summary_flags.
  DRY     "1" to report what would happen and write nothing (default 0)
  ROWCHK  rows per shard to verify after writing (default 4096; 0 disables)

Run:
  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/migrate_checkpoint_flags.py
"""
import glob
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core.config import CHECKPOINT_PATH
from core.Helpers.file_manager import atomic_torch_save
from core.SBI import statistics, training_checkpoint as tc

DRY = os.environ.get("DRY", "0") == "1"
ROWCHK = int(os.environ.get("ROWCHK", "4096"))
N_FEAT = len(statistics.FEATURE_LABELS)
N_FLAG = len(statistics.VALID_FLAG_LABELS)


def _pick_source() -> Path:
    if os.environ.get("SRC"):
        return Path(os.environ["SRC"])
    cands = []
    for d in sorted(CHECKPOINT_PATH.glob("train_*")):
        st = tc.peek(d)
        if not (st and st.get("complete")):
            continue
        try:
            ident = tc.read_header(d).get("identity", {})
        except Exception:                                        # noqa: BLE001
            continue
        if "summary_flags" not in ident:
            cands.append((os.path.getmtime(d), d))
    if not cands:
        sys.exit(f"no complete pre-flag checkpoint under {CHECKPOINT_PATH}; set SRC=<dir>")
    if len(cands) > 1:
        print(f"[note] {len(cands)} candidates; taking the newest. Set SRC to choose:")
        for _, d in cands:
            print(f"         {d.name}")
    return max(cands)[1]


def _widen(x: torch.Tensor, dt: float) -> torch.Tensor:
    """[feat | logT | chi] -> [feat | flags | logT | chi]. Insert, never append: gen_stats emits the
    flags as part of the summary block, so they must sit where it puts them or the network's
    input_dim split would land mid-block."""
    flags = statistics.derive_valid_flags(x[:, :N_FEAT], dt)
    return torch.cat([x[:, :N_FEAT], flags.to(x.dtype), x[:, N_FEAT:]], dim=1)


def main() -> None:
    src = _pick_source()
    state = tc.peek(src)
    if not state:
        sys.exit(f"{src} holds no readable checkpoint state")
    header = tc.read_header(src)
    ident = dict(header.get("identity", {}))
    if "summary_flags" in ident:
        sys.exit(f"{src.name} already carries summary_flags={ident['summary_flags']}; nothing to do.")

    dt = float(ident.get("dt_exp", 1.0))
    new_ident = dict(ident)
    new_ident["summary_flags"] = list(statistics.VALID_FLAG_LABELS)
    dst = tc.resolve_dir(new_ident)

    x_shards = sorted(glob.glob(str(src / "shards" / "x_*.pt")))
    th_shards = sorted(glob.glob(str(src / "shards" / "th_*.pt")))
    src_bytes = sum(os.path.getsize(f) for f in x_shards + th_shards)
    grow = (N_FEAT + N_FLAG + 1) / (N_FEAT + 1)      # x only; th is copied unchanged

    print(f"[source]  {src.name}  {state['batches_done']} batches, "
          f"complete={bool(state.get('complete'))}, dt_exp={dt:g}")
    print(f"[source]  {len(x_shards)} x-shards + {len(th_shards)} th-shards, "
          f"{src_bytes / 2**30:.2f} GiB")
    print(f"[flags]   {N_FLAG}: {', '.join(statistics.VALID_FLAG_LABELS)}")
    print(f"[target]  {dst.name}   (summary block {N_FEAT + 1} -> {N_FEAT + N_FLAG + 1})")
    free = shutil.disk_usage(CHECKPOINT_PATH).free
    need = int(src_bytes * grow)
    print(f"[disk]    need ~{need / 2**30:.2f} GiB, {free / 2**30:.1f} GiB free")
    if need > free:
        sys.exit("not enough free disk for the migrated copy")
    if dst.exists():
        sys.exit(f"{dst} already exists -- delete it deliberately, or it is already migrated.")
    if DRY:
        print("\nDRY=1, nothing written.")
        return

    (dst / "shards").mkdir(parents=True, exist_ok=True)
    atomic_torch_save({
        "format": header["format"],
        "identity": new_ident,
        "batch_t_scales": header["batch_t_scales"],
        "batch_Ts": header["batch_Ts"],
        "inits": header["inits"],
        "V": header.get("V"),                 # NEVER recomputed -- V is not reproducible across processes; see the module docstring
        "probe": header.get("probe"),
        "run_size": header["run_size"],
        "n_runs": header["n_runs"],
        # Provenance. A migrated checkpoint is not a simulated one, and the next person to read this
        # directory should not have to infer that from a directory name.
        "migrated_from": src.name,
        "flags_synthesised": True,
    }, dst / "header.pt")

    bad = 0
    for i, f in enumerate(x_shards):
        name = os.path.basename(f)
        x = torch.load(f, map_location="cpu", weights_only=False)
        w = _widen(x, dt)
        atomic_torch_save(w, dst / "shards" / name)
        if ROWCHK:
            # Re-read what was actually written, not the tensor still in memory: the point of the
            # check is that the FILE is right.
            back = torch.load(str(dst / "shards" / name), map_location="cpu", weights_only=False)
            n = min(ROWCHK, back.shape[0])
            ok = (torch.equal(back[:n, :N_FEAT], x[:n, :N_FEAT])
                  and torch.equal(back[:n, N_FEAT + N_FLAG:], x[:n, N_FEAT:])
                  and torch.equal(back[:n, N_FEAT:N_FEAT + N_FLAG],
                                  statistics.derive_valid_flags(x[:n, :N_FEAT], dt).to(x.dtype)))
            if not ok:
                bad += 1
                print(f"  !! {name} failed the bitwise check", flush=True)
        del x, w
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(x_shards)} x-shards", flush=True)
    for f in th_shards:
        shutil.copyfile(f, dst / "shards" / os.path.basename(f))

    # State LAST, so an interrupted migration leaves a directory with no usable state rather than one
    # that claims batches it does not have -- the same commit order the checkpoint writer uses.
    atomic_torch_save({"batches_done": int(state["batches_done"]),
                       "complete": bool(state.get("complete")),
                       "rng": state.get("rng")}, dst / "state.pt")

    if bad:
        sys.exit(f"\n{bad} shard(s) failed verification -- do NOT train against {dst.name}")
    print(f"\n[done] {dst}")
    print(f"[done] {len(x_shards)} shards widened and verified, {len(th_shards)} target shards copied.")
    print("[done] Start the same run again (same prior, same config) and it will resume here with "
          "zero simulation.")
    print("\nMIGRATE_CHECKPOINT_DONE", flush=True)


if __name__ == "__main__":
    main()
