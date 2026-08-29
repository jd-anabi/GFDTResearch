"""Which conditioning channels can the flow actually SEE? Reads the artifacts; simulates nothing.

THE TEST HAD TO BE CORRECTED BEFORE IT WOULD ANSWER. The
2026-08-25 addendum nominates this as its decisive test and specifies it as: replace channel i with
its fitted mean, and expect EXACTLY ZERO change from a dead channel. Run that way it gives 5.2e-5 for
A1_mean and 4.6e-4 for D3_bimodality -- not zero -- and you conclude both channels are alive, which
discards the document's best finding.

The reason is that the fitted mean is ITSELF contaminated. A1_mean was fitted at mean = -3.7e7 and
std = 4.19e11 on `posterior_08232026`, against a physical range of order 1e3, so "substitute the
fitted mean" is a move of ~1e7 in standardised units -- a large perturbation, not a null one.

WHAT THIS SCRIPT DOES INSTEAD: sweep each channel across its own real p1-p99 range, every other
channel held at its median, and record the largest embedding displacement that produces. That is the
question worth asking -- "does anything this channel can physically do change the network's output?"
-- and it separates cleanly:

    A1_mean         3.2e-7      <- a MILLIONTH of a typical channel's response (1.09e-6x). The
    D3_bimodality   3.2e-7         entire physical range of these two does essentially nothing.
    A2_log_var      1.228
    B4_spec_entropy 1.277       <- healthy; the median non-constant channel is 0.295
    A3_log_fpeak    2.374

The two dead channels reporting the SAME number to three figures is itself the tell: that value is
the network's own float noise, identical because neither channel moves the output at all.

(Measured on `posterior_08232026` at ROWS=200000, NSWEEP=33, base = the real row nearest the
standardised median -- 2 live probes of 12, a realistic observation. An earlier ad-hoc run recorded 1.8e-7 and
8.9e-8 for the same two channels from an ad-hoc run whose hold-point and resolution were not written
down; the RATIO to a healthy channel, ~1e-6, is the finding, and it reproduces. Verdict: 11
structurally dead, 2 numerically invisible, 29 usable of 42.)

Run it against the OLD posterior and the NEW one and compare: that comparison is the Phase-1
retrain's gate.

Env:
  POST   posterior .pt (bare name resolves against Resources/Posteriors). Default: newest.
  CKPT   training-checkpoint dir supplying the real data ranges. Default: the newest under
         Resources/Checkpoints. Rows are widened with the valid flags automatically when the net
         expects them, so ONE pre-flag checkpoint serves both an old and a new posterior.
  ROWS   rows sampled from the checkpoint for the quantiles (default 200000)
  NSWEEP points per channel sweep (default 33)

Run:
  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/channel_ablation.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core.config import POSTERIOR_PATH, CHECKPOINT_PATH
from core.SBI import statistics
from core.config import CHI_ELEM_W as config_chi_elem_w

ROWS = int(os.environ.get("ROWS", "200000"))
NSWEEP = int(os.environ.get("NSWEEP", "33"))
FLOAT32_EPS = 1.1920929e-07


def _resolve_posterior(name: str | None):
    if not name:
        cands = [p for p in POSTERIOR_PATH.glob("*.pt") if not p.name.endswith(".rot.pt")]
        if not cands:
            sys.exit(f"no posteriors in {POSTERIOR_PATH}; set POST=<path>")
        return max(cands, key=lambda p: p.stat().st_mtime)
    from pathlib import Path
    pt = Path(name)
    if not pt.exists():
        pt = POSTERIOR_PATH / (name if name.endswith(".pt") else name + ".pt")
    if not pt.exists():
        sys.exit(f"posterior not found: {name}")
    return pt


def _find_net(posterior):
    """The EmbeddedNet inside a pickled DirectPosterior, wherever sbi wrapped it."""
    est = getattr(posterior, "posterior_estimator", posterior)
    for m in est.modules():
        if hasattr(m, "standardize_summary") or hasattr(m, "sum_std"):
            return m
    sys.exit("no EmbeddedNet found inside this posterior")


def _load_rows(ckpt: str | None, want_width: int) -> torch.Tensor:
    root = ckpt or max((str(d) for d in CHECKPOINT_PATH.glob("train_*")),
                       key=lambda d: os.path.getmtime(d), default=None)
    if root is None:
        sys.exit(f"no training checkpoints under {CHECKPOINT_PATH}; set CKPT=<dir>")
    shards = sorted(glob.glob(os.path.join(root, "shards", "x_*.pt")))
    if not shards:
        sys.exit(f"no x_*.pt shards under {root}")
    out, got = [], 0
    for f in shards:
        x = torch.load(f, map_location="cpu", weights_only=False)
        out.append(x)
        got += x.shape[0]
        if got >= ROWS:
            break
    data = torch.cat(out, dim=0)[:ROWS]
    print(f"[data] {data.shape[0]:,} rows x {data.shape[1]} from {os.path.basename(root)}")
    n_feat = len(statistics.FEATURE_LABELS)
    if data.shape[1] == want_width - len(statistics.VALID_FLAG_LABELS):
        # A pre-flag checkpoint against a net that expects them: derive the flags and insert them
        # where gen_stats now puts them, i.e. immediately after the feature block.
        flags = statistics.derive_valid_flags(data[:, :n_feat], 1.0)
        data = torch.cat([data[:, :n_feat], flags, data[:, n_feat:]], dim=1)
        print(f"[data] widened to {data.shape[1]} with {flags.shape[1]} derived valid flags")
    if data.shape[1] != want_width:
        sys.exit(f"checkpoint rows are {data.shape[1]} wide; this posterior conditions on "
                 f"{want_width}. Point CKPT at the checkpoint this posterior trained on.")
    return data


def main() -> None:
    pt = _resolve_posterior(os.environ.get("POST"))
    posterior = torch.load(str(pt), map_location="cpu", weights_only=False)
    net = _find_net(posterior)
    net.eval()
    n_sum = int(net.input_dim)
    width = n_sum + int(net.forcing_dim)
    kind = "rank-Gaussian" if "rg_knots" in net._buffers else "legacy mean/std affine"
    print(f"[artifact] {pt.name}   summary {n_sum} + forcing {net.forcing_dim} = {width}")
    print(f"[standardizer] {kind}")

    data = _load_rows(os.environ.get("CKPT"), width)

    # Label from the net's ACTUAL layout, not by truncating the current one: a pre-flag posterior's
    # summary block is [features | logT] and slicing the new label list would silently name its
    # log(T) column "V_B1_Q" -- a mislabelled healthy channel in exactly the table this script exists
    # to read.
    n_feat = len(statistics.FEATURE_LABELS)
    if n_sum == n_feat + 1:
        labels = statistics.FEATURE_LABELS + ["logT"]
    elif n_sum == statistics.SUMMARY_WIDTH + 1:
        labels = statistics.FEATURE_LABELS + statistics.VALID_FLAG_LABELS + ["logT"]
    else:
        labels = [f"col{i}" for i in range(n_sum)]
        print(f"[warn] summary width {n_sum} matches neither the pre-flag layout ({n_feat + 1}) nor "
              f"the current one ({statistics.SUMMARY_WIDTH + 1}); columns are labelled positionally.")
    # THE BASE POINT IS A REAL ROW, not the column-wise median vector.
    #
    # A median VECTOR is not an observation: under chi, most probe-slot columns are zero in most
    # rows, so their medians are zero and the composed "row" has an all-pad probe block -- an
    # observation with no live probes at all, which no recording can produce and which puts the set
    # encoder on its n=0 path. The sweep still measures the summary pathway there, and the numbers do
    # reproduce, but a gate that decides whether to accept a multi-day retrain should not be
    # evaluated at a point outside the data. So: take the real row closest to the median, in
    # per-column-standardised units so no single wide channel picks it.
    med = data.median(dim=0).values
    scale = data.std(dim=0).clamp(min=1e-12)
    row = int((((data - med) / scale) ** 2).sum(dim=1).argmin())
    base = data[row].unsqueeze(0)
    note = ""
    if net.forcing_dim and int(net.forcing_dim) % config_chi_elem_w == 0:
        blk = base[0, n_sum:].reshape(-1, config_chi_elem_w)
        note = (f", {int((blk[:, -1] > 0.5).sum())} live probe(s) of "
                f"{blk.shape[0]} slots")
    print(f"[base] real row {row}{note}")
    q = torch.tensor([0.01, 0.99])
    with torch.no_grad():
        e0 = net(base)
        rows = []
        for j in range(n_sum):
            col = data[:, j].float().sort().values
            idx = (q * (col.numel() - 1)).long()
            lo, hi = float(col[idx[0]]), float(col[idx[1]])
            if hi <= lo:
                rows.append((0.0, labels[j], lo, hi, "CONSTANT"))
                continue
            v = base.repeat(NSWEEP, 1)
            v[:, j] = torch.linspace(lo, hi, NSWEEP, dtype=v.dtype)
            d = float((net(v) - e0).norm(dim=-1).max())
            rows.append((d, labels[j], lo, hi, ""))

    live = [r[0] for r in rows if r[4] != "CONSTANT"]
    med = torch.tensor(live).median().item() if live else 0.0
    print(f"\n=== max ||delta embedding|| over each channel's real p1-p99 range ===")
    print(f"median over non-constant channels: {med:.4g};  float32 eps = {FLOAT32_EPS:.3g}\n")
    print(f"{'channel':<24} {'max|d emb|':>12} {'vs median':>10}   {'p1':>12} {'p99':>12}  verdict")
    print("-" * 92)
    for d, lab, lo, hi, note in sorted(rows):
        if note == "CONSTANT":
            verdict = "constant in training (structurally dead)"
        elif d < FLOAT32_EPS or (med and d < med * 1e-4):
            # Either test alone under-reports. The absolute one misses a channel whose whole range
            # moves the embedding a millionth as far as a typical channel's but still clears an ulp;
            # the relative one would flag a channel on a network whose outputs are all tiny.
            verdict = "*** INVISIBLE -- the whole physical range does nothing ***"
        elif med and d < med / 100:
            verdict = "severely compressed"
        elif med and d < med / 10:
            verdict = "compressed"
        else:
            verdict = "healthy"
        rel = f"{d / med:.3g}x" if med else "-"
        print(f"{lab:<24} {d:12.4g} {rel:>10}   {lo:12.4g} {hi:12.4g}  {verdict}")

    n_invis = sum(1 for d, _, _, _, n in rows
                  if n != "CONSTANT" and (d < FLOAT32_EPS or (med and d < med * 1e-4)))
    n_const = sum(1 for r in rows if r[4] == "CONSTANT")
    print(f"\n{n_const} structurally dead, {n_invis} numerically invisible, "
          f"{n_sum - n_const - n_invis} usable of {n_sum} summary channels")
    print("\nCHANNEL_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
