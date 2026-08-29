"""
WHY are chi probes being masked, and which knob would recover them?

The smoke train found 77 % of training probes masked. The runtime warning
lumps every cause into one sentence -- "below CHI_MIN_CYCLES drive cycles, at/above Nyquist, out of
band, or a non-finite lock-in" -- which is the right message for a training log and useless for
deciding what to change. This separates them, on the REAL stability-screened prior, and reports the
one distribution nobody has ever looked at: Omega_0 across prior draws.

The predicates live in two places and it matters which is which:
  gen_chi_raw   non-finite / non-positive frequency, >= 0.9 Nyquist, and the CHI_MIN_CYCLES floor
  pack_probe_block  the BAND filter (|u_hat| > CHI_UHAT_MAX)
so this instruments both and reports the packer's marginal contribution separately.

Read the output as: whichever predicate dominates is the one to attack. If it is the cycle floor,
the fix is about the (band x T) interaction -- either place probes conditional on T, or narrow the
T range, or move the floor. If it is Nyquist or the band, the fix is a different one entirely.

Env knobs (CELL / BOUNDS / MODEL / CHI* are handled by _common.script_cfg):
  PRIOR      prior artifact to reuse; built and saved under this name if absent
                                                    (default _c6_prior)
  N_RUNS     training batches to audit              (default 12)
  RUN_SIZE   rows per batch                         (default 32)
  SEED       RNG seed                               (default 0)

Run:
  $env:CHI=1; $env:BOUNDS="Resources/Bounds/nadrowski/master.txt"
  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/chi_mask_audit.py
"""
import inspect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt

import _common
from core import config, orchestrator
from core.config import PRIOR_PATH
from core.SBI import chi as chi_mod, pipeline as pipeline_mod

_common.enable_warnings()

PRIOR = os.environ.get("PRIOR", "_c6_prior")
N_RUNS = int(os.environ.get("N_RUNS", "12"))
RUN_SIZE = int(os.environ.get("RUN_SIZE", "32"))
SEED = int(os.environ.get("SEED", "0"))

REC = []            # one row per (batch, probe): every quantity the predicates read


def _instrument():
    """Wrap gen_chi_raw + gen_chi_block, re-deriving each predicate SEPARATELY from the same inputs
    the real code sees. Re-derivation rather than refactoring the predicates out: this is a
    diagnostic, and it must not be able to change what production does."""
    real_raw, real_block = pipeline_mod.gen_chi_raw, pipeline_mod.gen_chi_block
    pending = {}

    sig = inspect.signature(real_raw)

    def spy_raw(*a, **kw):
        out = real_raw(*a, **kw)
        chis, u, logcyc, valid = out
        # BIND the signature rather than reading kwargs: gen_chi_block forwards *args positionally,
        # so the training path arrives here with x_spont_dim/N_points/multipliers as positionals and
        # a kwargs lookup KeyErrors. Binding gets them by NAME however the caller passed them.
        bound = sig.bind(*a, **kw)
        bound.apply_defaults()
        args = bound.arguments
        x_spont = args["x_spont_dim"]
        dt_exp, N_points = args["dt_exp"], args["N_points"]
        mults = args["multipliers"]
        f_peak = chi_mod.peak_freq(x_spont, dt_exp)                      # (B,)
        m = mults if torch.is_tensor(mults) else torch.as_tensor(mults)
        m = m.to(device=f_peak.device, dtype=f_peak.dtype)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        freqs = (m if args.get("absolute_freqs") else m * f_peak.unsqueeze(1)).expand(f_peak.shape[0], -1)
        nyq = 0.5 / dt_exp
        cycles = torch.exp(logcyc)
        # Is this row an OSCILLATOR at all? peak_freq is an argmax and always returns something --
        # on a non-oscillatory trace it returns the bottom of a 1/f-ish spectrum, which is
        # indistinguishable from a genuine slow oscillator by frequency alone. Peak power over
        # MEDIAN power separates them: ~1 means there is no peak, only a slope.
        xd = (x_spont - x_spont.mean(dim=-1, keepdim=True)).to(torch.float64)
        psd = (torch.fft.rfft(xd, dim=-1).abs() ** 2)
        psd[:, 0] = 0.0
        prom = psd.max(dim=-1).values / psd[:, 1:].median(dim=-1).values.clamp(min=1e-300)
        pending.update(dict(
            f_peak=f_peak.cpu(), freqs=freqs.cpu(), cycles=cycles.cpu(), valid=valid.cpu(),
            prom=prom.cpu(),
            bad_freq=(~torch.isfinite(freqs) | (freqs <= 0)).cpu(),
            over_nyq=(freqs >= 0.9 * nyq).cpu(),
            under_cyc=(cycles < config.CHI_MIN_CYCLES).cpu(),
            T_full=N_points * dt_exp, mult=m.expand_as(freqs).cpu()))
        return out

    def spy_block(*a, **kw):
        block, mask = real_block(*a, **kw)
        if pending:
            p = dict(pending)
            p["packed_mask"] = mask.cpu()
            REC.append(p)
            pending.clear()
        return block, mask

    pipeline_mod.gen_chi_raw, pipeline_mod.gen_chi_block = spy_raw, spy_block
    return real_raw, real_block


def _pct(n, d):
    return f"{100.0 * n / max(1, d):5.1f}%"


def main():
    torch.manual_seed(SEED)
    cfg = _common.script_cfg()
    if not cfg.chi_mode:
        raise SystemExit("chi_mask_audit only means anything in chi mode. Set CHI=1.")
    print(f"[audit] N_RUNS={N_RUNS} RUN_SIZE={RUN_SIZE} SEED={SEED} PRIOR={PRIOR}")
    print(f"[audit] band={cfg.chi_freq_bounds} floor={config.CHI_MIN_CYCLES:g} cycles "
          f"ceiling={cfg.chi_max_cycles:g} cycles  T~logU[{cfg.t_min_exp / cfg.get_unit_conversion_factor('s'):g}, "
          f"{cfg.t_max_exp / cfg.get_unit_conversion_factor('s'):g}]s", flush=True)

    have = (PRIOR_PATH / f"{PRIOR}.pt").exists()
    print(f"[audit] prior: {'loading ' + PRIOR if have else 'building (saved as ' + PRIOR + ')'}",
          flush=True)
    inferred_prior, force_prior = orchestrator.build_prior(
        cfg, f"{PRIOR}.pt" if have else None, not have,
        save=not have, save_name=None if have else PRIOR, fig_sink=lambda t, f: plt.close(f))

    real_raw, real_block = _instrument()
    cfg.hw.batch_size = RUN_SIZE
    try:
        pipeline_mod.gen_training_data(
            cfg.model, inferred_prior, force_prior, cfg.t,
            run_size=RUN_SIZE, n_runs=N_RUNS, steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min,
            nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
            dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
            t_scale_bounds=cfg.t_scale_bounds, state_dep_drift=cfg.state_dep_drift,
            chi_mode=True, chi_f0=cfg.chi_f0, chi_freq_bounds=cfg.chi_freq_bounds,
            chi_k_pad=cfg.chi_k_pad, chi_max_cycles=cfg.chi_max_cycles,
            n_vars=cfg.inits_tensor.shape[-1], dtype=cfg.hw.dtype, device=cfg.hw.device)
    finally:
        pipeline_mod.gen_chi_raw, pipeline_mod.gen_chi_block = real_raw, real_block

    if not REC:
        raise SystemExit("no chi batches were recorded -- the instrumentation did not fire.")

    hz = cfg.get_unit_conversion_factor("s")
    tot = sum(r["valid"].numel() for r in REC)
    bad = sum(int(r["bad_freq"].sum()) for r in REC)
    nyq = sum(int((r["over_nyq"] & ~r["bad_freq"]).sum()) for r in REC)
    cyc = sum(int((r["under_cyc"] & ~r["over_nyq"] & ~r["bad_freq"]).sum()) for r in REC)
    raw_masked = sum(int((~r["valid"]).sum()) for r in REC)
    print(f"\n=== why probes are masked ({tot} probe-rows over {len(REC)} batches) ===")
    print(f"  non-finite / non-positive frequency : {bad:6d}  {_pct(bad, tot)}")
    print(f"  at or above 0.9 x Nyquist           : {nyq:6d}  {_pct(nyq, tot)}")
    print(f"  under the {config.CHI_MIN_CYCLES:g}-cycle floor          "
          f": {cyc:6d}  {_pct(cyc, tot)}   <- the (band x T) interaction")
    print(f"  ---- gen_chi_raw total              : {raw_masked:6d}  {_pct(raw_masked, tot)}")
    # COUNTS per row, not an element-wise AND: `valid` is (B, K) over the probes as DRAWN while
    # `packed_mask` is (B, k_pad) over slots, and pack_probe_block COMPACTS the live probes to the
    # front ascending in frequency -- so slot j is not probe j and the two cannot be lined up
    # positionally. The difference in live COUNT per row is exactly the packer's marginal kill.
    extra = sum(int((r["valid"].sum(1) - r["packed_mask"].sum(1)).clamp(min=0).sum())
                for r in REC if "packed_mask" in r)
    packed_live = sum(int(r["packed_mask"].sum()) for r in REC if "packed_mask" in r)
    print(f"  band filter |u_hat| > {config.CHI_UHAT_MAX:g} (packer)    : {extra:6d}  {_pct(extra, tot)}")
    print(f"  ==== LIVE after packing             : {packed_live:6d}  {_pct(packed_live, tot)}")

    f_peaks = torch.cat([r["f_peak"] for r in REC]) * hz
    q = torch.quantile(f_peaks.double(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95],
                                                      dtype=torch.float64))
    print(f"\n=== Omega_0 across prior draws (Hz) -- never measured before ===")
    print(f"  p5 {q[0]:.3g}   p25 {q[1]:.3g}   median {q[2]:.3g}   p75 {q[3]:.3g}   p95 {q[4]:.3g}")
    print(f"  For the band's LOW edge to clear the floor a row needs "
          f"Omega_0 * T >= {config.CHI_MIN_CYCLES / cfg.chi_freq_bounds[0]:.0f}.")

    # Both breakdowns below use `valid`, NOT the packed mask: they are per-PROBE questions and only
    # `valid` is aligned with the probe that produced it (see the compaction note above). They
    # therefore exclude the packer's band filter, which the block above reports on its own.
    print(f"\n=== live fraction vs recording length (pre-packer) ===")
    print(f"  {'T (s)':>10} {'batches':>8} {'live':>8}")
    for lo, hi in ((0, 2), (2, 5), (5, 12), (12, 30), (30, 1e9)):
        sel = [r for r in REC if lo <= r["T_full"] / hz < hi]
        if not sel:
            continue
        live = sum(int(r["valid"].sum()) for r in sel)
        n = sum(r["valid"].numel() for r in sel)
        print(f"  {lo:4g}-{hi if hi < 1e9 else 999:<5g} {len(sel):>8} {_pct(live, n):>8}")

    # THE decisive table. Probes sit at mult * Omega_0, so a row's Omega_0 sets every one of its
    # probe frequencies -- masking is a per-ROW property of theta before it is anything else.
    print(f"\n=== live fraction vs the ROW's Omega_0 ===")
    f_all = torch.cat([r["f_peak"].unsqueeze(1).expand_as(r["valid"]).flatten() for r in REC]) * hz
    v_all = torch.cat([r["valid"].flatten() for r in REC])
    print(f"  {'Omega_0 (Hz)':>16} {'probes':>8} {'live':>8}")
    f_edges = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 1e9]
    for i in range(len(f_edges) - 1):
        sel = (f_all >= f_edges[i]) & (f_all < f_edges[i + 1])
        if not bool(sel.any()):
            continue
        print(f"  {f_edges[i]:6.3g}-{f_edges[i+1] if f_edges[i+1] < 1e9 else 999:<8.4g} "
              f"{int(sel.sum()):>8} {_pct(int(v_all[sel].sum()), int(sel.sum())):>8}")

    # THE interpretive question. If the masked rows are non-oscillatory, chi is UNDEFINED for them
    # and masking them is correct -- the problem is then the PRIOR spending its mass there, not the
    # mask. If they are genuine slow oscillators, chi is real and merely unmeasurable in an
    # admissible recording, which is a different (and harder) problem.
    print(f"\n=== is a low-Omega_0 row an oscillator at all? (peak power / median power) ===")
    print(f"  {'Omega_0 (Hz)':>16} {'rows':>7} {'median prominence':>19}")
    fp_row = torch.cat([r["f_peak"] for r in REC]) * hz
    pr_row = torch.cat([r["prom"] for r in REC])
    for i in range(len(f_edges) - 1):
        sel = (fp_row >= f_edges[i]) & (fp_row < f_edges[i + 1])
        if not bool(sel.any()):
            continue
        print(f"  {f_edges[i]:6.3g}-{f_edges[i+1] if f_edges[i+1] < 1e9 else 999:<8.4g} "
              f"{int(sel.sum()):>7} {float(pr_row[sel].median()):>19.1f}")
    print(f"  A flat spectrum sits near 1. A clean limit cycle is orders of magnitude above it.")

    # A row with NO live probe is a spontaneous row wearing a chi conditioning vector. This is the
    # number that decides whether chi mode is doing anything at all on this prior.
    dead = sum(int((r["valid"].sum(1) == 0).sum()) for r in REC)
    rows = sum(r["valid"].shape[0] for r in REC)
    print(f"\n=== rows with ZERO live probes: {dead}/{rows} = {_pct(dead, rows)} ===")
    print(f"  Such a row conditions on the passive trace alone -- chi mode is inert for it.")

    # LIVE COUNT IS NOT THE OBJECTIVE. chi(omega) measures the SHAPE of a curve, so a row whose live
    # probes all sit at one frequency is barely better than a row with none -- and any placement rule
    # tuned on live count alone will happily produce exactly that. Report the span too.
    lo_b, hi_b = cfg.chi_freq_bounds
    spans, singles = [], 0
    for r in REC:
        fr, va = r["freqs"], r["valid"]
        for b in range(fr.shape[0]):
            f_live = fr[b][va[b]]
            if f_live.numel() == 0:
                continue
            singles += int(f_live.numel() == 1)
            spans.append(1.0 if f_live.numel() == 1
                         else float(f_live.max() / f_live.min().clamp(min=1e-30)))
    if spans:
        s = torch.tensor(spans)
        q = torch.quantile(s.double(), torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        print(f"\n=== frequency SPAN of a row's live probes (max/min) ===")
        print(f"  p25 {q[0]:.2f}x   median {q[1]:.2f}x   p75 {q[2]:.2f}x"
              f"   (the band itself spans {hi_b / lo_b:.0f}x)")
        print(f"  rows with exactly ONE live probe: {singles}/{len(spans)} = "
              f"{_pct(singles, len(spans))} -- those carry a point, not a curve.")

    print(f"\n=== live fraction vs probe multiplier (pre-packer) ===")
    mult_all = torch.cat([r["mult"].flatten() for r in REC])
    live_all = torch.cat([r["valid"].flatten() for r in REC])
    edges = torch.tensor([0.0, 0.05, 0.08, 0.12, 0.2, 1e9])
    print(f"  {'x Omega_0':>12} {'probes':>8} {'live':>8}")
    for i in range(len(edges) - 1):
        sel = (mult_all >= edges[i]) & (mult_all < edges[i + 1])
        if not bool(sel.any()):
            continue
        print(f"  {edges[i]:5.3g}-{edges[i+1] if edges[i+1] < 1e9 else 9:<5.3g} "
              f"{int(sel.sum()):>8} {_pct(int(live_all[sel].sum()), int(sel.sum())):>8}")
    print()


if __name__ == "__main__":
    main()
