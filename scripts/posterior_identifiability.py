"""
What did a SAVED posterior actually measure? Reads the artifacts; simulates nothing.

Answers the question a corner plot cannot: a 13-D posterior with four well-constrained directions and
nine prior-like ones looks like "severe degeneracy" on the physical axes, but it is a perfectly good
result as long as you can say WHICH four. The Fisher rotation V in the posterior's `.rot.pt` sidecar
is exactly that basis -- its columns are the eigenvectors of the prior-averaged simulation Fisher
F = J^T J, sorted descending by eigenvalue -- so the decomposition is already on disk and free.

NOT the same thing as scripts/identifiability_offgt.py, which SIMULATES a Laplace metric at K prior
points to ask whether the information exists at all. This one asks what a particular trained artifact
kept, costs no simulation, and runs in a second.

WHAT IT REVEALED ON THE 2026-08-25 RETRAIN, and why the distinction matters: `k` puts 99.9% of its
weight on a single direction whose loading is -1.00*k, and its overlap with `x_scale` across all 13
directions is 0.0002. The handoff had the central problem recorded as the `k`~`x_scale` alias
(|cos| 0.97, measured three times). Both are true, and together they say something sharper than
either: degeneracy_map's |cos| compares the two parameters' gradient DIRECTIONS, while a Fisher
eigenvalue is about gradient MAGNITUDE, and `k`'s gradient is both nearly parallel to `x_scale`'s and
tiny. So `k` is not so much degenerate as UNMEASURED -- and you cannot rotate, reparameterise or
re-prior your way out of a flat direction, which is a different engineering problem from breaking an
alias.

Env:
  POST   path to the posterior .pt (the .rot.pt sidecar is read from beside it), or a bare name
         resolved against Resources/Posteriors. Default: the newest .pt in Resources/Posteriors.
  NBOT   how many of the worst directions to total in the per-parameter table (default 3)
  TOPN   loadings shown per direction (default 4)

Run:  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/posterior_identifiability.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from core.config import POSTERIOR_PATH

NBOT = int(os.environ.get("NBOT", "3"))
TOPN = int(os.environ.get("TOPN", "4"))


def _resolve(post: str | None) -> Path:
    """The posterior .pt. Accepts a full path or a bare name; defaults to the newest in Resources.

    A bare .pt is tried as a path FIRST, because these artifacts are routinely copied out of
    Resources/ into a results folder next to the figures they produced -- which is where you are when
    you want to run this."""
    if not post:
        cands = [p for p in POSTERIOR_PATH.glob("*.pt") if not p.name.endswith(".rot.pt")]
        if not cands:
            sys.exit(f"no posteriors in {POSTERIOR_PATH}; set POST=<path>")
        return max(cands, key=lambda p: p.stat().st_mtime)
    pt = Path(post)
    if not pt.exists():
        pt = POSTERIOR_PATH / (post if post.endswith(".pt") else post + ".pt")
    if not pt.exists():
        sys.exit(f"posterior not found: {post}")
    return pt


def main() -> None:
    pt = _resolve(os.environ.get("POST"))
    side = pt.parent / (pt.name[:-3] + ".rot.pt")
    if not side.exists():
        sys.exit(f"no sidecar beside the posterior: {side}\n"
                 f"Pre-reparameterisation artifacts have none, and without it there is no rotation "
                 f"to decompose.")
    d = torch.load(str(side), map_location="cpu", weights_only=False)
    print(f"[artifact] {pt.name}   mode={d.get('mode')}   model={d.get('model')}")

    names = list(d.get("param_keys") or [])
    V = d.get("V")
    if V is None:
        sys.exit("the sidecar records V=None (the rotation was off for this run); nothing to decompose.")
    V = V.double().numpy()
    P = V.shape[0]
    if len(names) != P:
        names = [f"p{i}" for i in range(P)]

    orth = float(np.abs(V.T @ V - np.eye(P)).max())
    print(f"[V] {P}x{P}, orthogonal to {orth:.1e}")

    # ── the conditioning geometry: how much of the input was ever filled ────────────────────────
    k_pad, elem_w = d.get("chi_k_pad"), d.get("chi_elem_w")
    if k_pad and elem_w:
        supplied, base = d.get("chi_n_freqs") or 0, d.get("input_dim") or 0
        print(f"[conditioning] width {base + k_pad * elem_w} = {base} base + {k_pad} probe slots x "
              f"{elem_w}")
        print(f"[conditioning] {supplied} probe(s) supplied into {k_pad} slots -> "
              f"{(k_pad - supplied) * elem_w} elements are pure padding "
              f"({100.0 * (k_pad - supplied) / k_pad:.0f}% of the probe block never filled)")

    # ── eigenvalues: the half that turns an ordering into a measurement ─────────────────────────
    ev = d.get("fisher_eigenvalues")
    if ev is None:
        print("\n[eigenvalues] NOT STORED for this artifact.")
        print("  Everything below is an ORDERING and a set of LOADINGS -- which direction is worst,")
        print("  and what it is made of -- but NOT how much worse it is. That scale is the question:")
        print("  a 3x spread means the experiment measures everything tolerably; 1e6 means it")
        print("  measures a handful of directions and returns the prior for the rest.")
        print("  Posteriors saved after 2026-08-25 carry them; recovering them for an older artifact")
        print("  costs a full Fisher re-run.")
    else:
        ev = np.asarray(ev, dtype=float)
        pos = ev[ev > 0]
        spread = ev[0] / ev[-1] if ev[-1] > 0 else float("inf")
        # Participation ratio: how many directions carry the information, if they carried it equally.
        pr = (pos.sum() ** 2) / (pos ** 2).sum() if pos.size else 0.0
        print(f"\n[eigenvalues] max {ev[0]:.4g}  min {ev[-1]:.4g}  spread {spread:.4g}")
        print(f"[eigenvalues] participation ratio {pr:.2f} of {P} "
              f"-> ~{pr:.1f} effectively constrained direction(s)")

    # ── the directions ──────────────────────────────────────────────────────────────────────────
    print(f"\n=== Fisher eigen-directions (columns of V), BEST-constrained first ===")
    print("loadings are in box-normalised coordinates, so they are comparable across parameters")
    for j in range(P):
        col = V[:, j]
        top = np.argsort(-np.abs(col))[:TOPN]
        part = "  ".join(f"{col[i]:+.2f}*{names[i]}" for i in top)
        ev_s = f"  [lambda={float(ev[j]):.3g}]" if ev is not None else ""
        tag = "   <-- BEST" if j == 0 else ("   <-- WORST" if j == P - 1 else "")
        print(f"  dir {j:2d}: {part}{ev_s}{tag}")

    # ── per parameter ───────────────────────────────────────────────────────────────────────────
    W = V ** 2                       # rows sum to 1: how parameter i is spread over the directions
    print(f"\n=== per parameter: share of its identifiability in the WORST directions ===")
    print(f"{'param':>10s} {'bottom-' + str(NBOT):>10s} {'top-4':>8s} {'peak dir':>9s}   verdict")
    rows = [(W[i, P - NBOT:].sum(), W[i, :4].sum(), int(np.argmax(W[i])), names[i]) for i in range(P)]
    for bot, top4, peak, nm in sorted(rows, reverse=True):
        if bot > 0.6:
            verdict = "UNMEASURED" if bot > 0.95 else "very poor"
        elif bot > 0.25:
            verdict = "poor"
        elif top4 > 0.5:
            verdict = "good"
        else:
            verdict = "moderate"
        print(f"{nm:>10s} {bot:10.3f} {top4:8.3f} {peak:9d}   {verdict}")

    # ── isolated flat axes: the finding that is easy to miss ────────────────────────────────────
    print("\n=== parameters that are their OWN near-null direction ===")
    print("(a parameter with ~all its weight on one bottom direction is FLAT, not aliased --")
    print(" no reparameterisation or prior rotation reaches it; only a new observable does)")
    flagged = False
    for i in range(P):
        j = int(np.argmax(W[i]))
        if j >= P - NBOT and W[i, j] > 0.9 and abs(V[i, j]) > 0.9:
            partners = [names[q] for q in range(P) if q != i and abs(V[q, j]) > 0.25]
            print(f"  {names[i]:>10s}: {W[i, j]:.3f} of its weight on dir {j}, loading {V[i, j]:+.3f}"
                  + (f", shared with {', '.join(partners)}" if partners else ", ALONE"))
            flagged = True
    if not flagged:
        print("  none -- every poorly-constrained parameter is mixed with others, i.e. aliased "
              "rather than flat")

    print("\nPOSTERIOR_IDENTIFIABILITY_DONE", flush=True)


if __name__ == "__main__":
    main()
