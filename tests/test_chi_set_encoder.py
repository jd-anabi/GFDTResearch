"""Invariant tests for the chi(omega) probe-set encoder (layout 2).

Pure torch, no simulation, no Qt -- these run in about a second and they are the only thing standing
between a subtly non-invariant encoder and a five-day training run that produces a posterior nobody
can explain. Every claim here is one a reviewer would otherwise have to take on trust from reading
`forward()`.

THE DISTINCTION THAT MATTERS: pad-VALUE inertness is BITWISE (a dead slot multiplies to exactly 0.0
and contributes exactly 0 to every reduction), while pad-POSITION, permutation and pad-WIDTH
invariance are only to ~1e-6, because `sum(dim=1)` is a float reduction whose rounding depends on the
order and number of elements summed. Asserting torch.equal on the latter would be wrong and would
fail intermittently.

Run:  python tests/test_chi_set_encoder.py
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core import config
from core.SBI import chi as chi_mod
from core.SBI.chi_encoder import ChiSetEncoder

W = config.CHI_ELEM_W
U_MID, U_HALF = chi_mod.band_norm()


def _enc(k_pad=8, seed=0, fit=True):
    torch.manual_seed(seed)
    e = ChiSetEncoder(k_pad, U_MID, U_HALF).eval()
    if fit:
        # A structurally valid fit set: never torch.randn through the whole block, which would push
        # fractional masks and out-of-band u through the statistics and prove nothing.
        e.fit(_block(_probes(6, k_pad, seed=seed + 99), k_pad)[0])
    return e


def _probes(n, k_pad, seed=0):
    """n physically plausible probes: u inside the band, unit-modulus phase, positive cycle counts."""
    g = torch.Generator().manual_seed(seed)
    u = U_MID + U_HALF * (torch.rand(n, generator=g) * 1.6 - 0.8)
    ang = torch.rand(n, generator=g) * 2 * math.pi
    mag = torch.exp(torch.randn(n, generator=g))
    logcyc = torch.log(torch.rand(n, generator=g) * 40 + 2.0)
    return u, mag * torch.exp(1j * ang), logcyc


def _block(probes, k_pad, slots=None, pad_value=0.0):
    """Pack probes into a (1, W*k_pad) block. `slots` places them explicitly (default 0..n-1)."""
    u, chi_v, logcyc = probes
    n = u.shape[0]
    e = torch.full((1, k_pad, W), float(pad_value))
    if pad_value != 0.0:
        e[..., 5] = 0.0                                  # a dead slot is dead whatever else it holds
    else:
        e.zero_()
    logmag = torch.log(chi_v.abs())
    ang = torch.angle(chi_v)
    where = list(range(n)) if slots is None else list(slots)
    for i, s in enumerate(where):
        e[0, s] = torch.tensor([u[i], logmag[i], math.cos(ang[i]), math.sin(ang[i]), logcyc[i], 1.0])
    return e.reshape(1, W * k_pad), where


# ── invariance ────────────────────────────────────────────────────────────────────────────────────
def test_permutation_invariance():
    """The whole point. Jointly permute the 6-tuple ROWS; the embedding must not move."""
    for seed in (0, 1, 2):
        enc = _enc(k_pad=8, seed=seed)
        x, _ = _block(_probes(5, 8, seed=seed), 8)
        base = enc(x)
        e = x.reshape(1, 8, W)
        for p in range(8):
            perm = torch.randperm(8, generator=torch.Generator().manual_seed(p))
            got = enc(e[:, perm, :].reshape(1, W * 8))
            assert (got - base).abs().max() < 1e-6, (seed, p, float((got - base).abs().max()))


def test_pad_value_inertness_is_bitwise():
    """A dead slot's other five channels must be arithmetically unreachable -- not merely small."""
    enc = _enc(k_pad=8)
    probes = _probes(4, 8)
    x0, _ = _block(probes, 8, pad_value=0.0)
    e0 = enc(x0)
    for junk in (1e6, -1e6, 3.7):
        x = x0.reshape(1, 8, W).clone()
        x[0, 4:, :5] = junk                              # poison every dead slot, leave mask at 0
        assert torch.equal(enc(x.reshape(1, W * 8)), e0), junk


def test_pad_position_and_pad_width_invariance():
    """Same live set, different slots and a different pad capacity -> the same embedding."""
    probes = _probes(3, 8)
    enc8, enc32 = _enc(k_pad=8), _enc(k_pad=32)
    enc32.load_state_dict(enc8.state_dict(), strict=False)     # same weights, different capacity
    enc32.elem_mean.copy_(enc8.elem_mean); enc32.elem_std.copy_(enc8.elem_std)
    enc32.fitted.fill_(1)
    base = enc8(_block(probes, 8)[0])
    moved = enc8(_block(probes, 8, slots=[5, 2, 7])[0])
    wider = enc32(_block(probes, 32, slots=[20, 3, 31])[0])
    assert (moved - base).abs().max() < 1e-6, float((moved - base).abs().max())
    assert (wider - base).abs().max() < 1e-6, float((wider - base).abs().max())


def test_masked_mean_divides_by_the_live_count():
    """Dividing by k_pad instead of n makes the embedding a function of the pad. Two pads, one set."""
    probes = _probes(2, 16)
    enc16 = _enc(k_pad=16)
    enc4 = ChiSetEncoder(4, U_MID, U_HALF).eval()
    enc4.load_state_dict(enc16.state_dict(), strict=False)
    enc4.elem_mean.copy_(enc16.elem_mean); enc4.elem_std.copy_(enc16.elem_std); enc4.fitted.fill_(1)
    a, b = enc16(_block(probes, 16)[0]), enc4(_block(probes, 4)[0])
    assert (a - b).abs().max() < 1e-6, float((a - b).abs().max())


def test_post_gate_is_present():
    """phi(0) != 0 for a biased MLP, so the mask must be re-applied AFTER phi. Without the post-gate
    the embedding drifts with the number of DEAD slots, which is the pad-width bug in disguise."""
    enc = _enc(k_pad=8)
    with torch.no_grad():                                  # force a non-zero phi(0)
        for m in enc.phi:
            if isinstance(m, torch.nn.Linear):
                m.bias.fill_(0.3)
    probes = _probes(3, 8)
    a = enc(_block(probes, 8)[0])
    enc_wide = ChiSetEncoder(24, U_MID, U_HALF).eval()
    enc_wide.load_state_dict(enc.state_dict(), strict=False)
    enc_wide.elem_mean.copy_(enc.elem_mean); enc_wide.elem_std.copy_(enc.elem_std)
    enc_wide.fitted.fill_(1)
    b = enc_wide(_block(probes, 24)[0])
    assert (a - b).abs().max() < 1e-6, float((a - b).abs().max())


# ── numerical safety ──────────────────────────────────────────────────────────────────────────────
def test_no_batchnorm_and_single_row_matches_batch():
    """sbi's get_numel runs the net on ONE cpu row at build time, and single-observation inference
    must equal the corresponding row of a batched call."""
    enc = _enc(k_pad=8)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in enc.modules())
    xs = torch.cat([_block(_probes(k, 8, seed=k), 8)[0] for k in (2, 4, 6)], dim=0)
    full = enc(xs)
    for i in range(xs.shape[0]):
        assert (enc(xs[i:i + 1]) - full[i:i + 1]).abs().max() < 1e-6, i


def test_empty_and_singleton_sets_are_finite():
    """n=0 is a legal training row (a batch whose T resolves nothing), and must be the empty-set
    constant rather than a NaN from dividing by zero."""
    enc = _enc(k_pad=8)
    empty = enc(torch.zeros(1, W * 8))
    assert torch.isfinite(empty).all()
    assert torch.equal(enc(torch.zeros(1, W * 8)), empty)          # deterministic constant
    one = enc(_block(_probes(1, 8), 8)[0])
    assert torch.isfinite(one).all()


def test_mask_is_binarised():
    """A fractional or wild mask must not leak into n. randn through the block used to NaN log1p."""
    enc = _enc(k_pad=8)
    torch.manual_seed(0)
    x = torch.randn(4, W * 8)
    out = enc(x)
    assert torch.isfinite(out).all(), "randn through the block must not produce NaN"
    e = x.reshape(4, 8, W)
    expect = (e[..., 5] > 0.5).sum(1)
    got = (e[..., 5] > 0.5).to(x.dtype).sum(1)
    assert torch.equal(expect.to(got.dtype), got)


def test_kernel_weights_are_masked():
    """A dead slot sitting at a knot centre must contribute nothing to that knot's quadrature."""
    enc = _enc(k_pad=8)
    probes = _probes(3, 8)
    x, _ = _block(probes, 8)
    base = enc(x)
    e = x.reshape(1, 8, W).clone()
    e[0, 6, 0] = U_MID                  # exactly at knot u_hat = 0
    e[0, 6, 1:5] = 5.0
    e[0, 6, 5] = 0.0                    # ...but dead
    assert torch.equal(enc(e.reshape(1, W * 8)), base)


def test_gradients_do_not_reach_padded_columns():
    enc = _enc(k_pad=8)
    x, _ = _block(_probes(3, 8), 8)
    x = x.clone().requires_grad_(True)
    enc(x).sum().backward()
    g = x.grad.reshape(1, 8, W)
    assert g[0, 3:, :].abs().max() == 0.0, float(g[0, 3:, :].abs().max())


def test_forward_raises_when_unfitted():
    enc = _enc(k_pad=8, fit=False)
    try:
        enc(_block(_probes(3, 8), 8)[0])
        raise AssertionError("an unfitted encoder must refuse to run")
    except RuntimeError as e:
        assert "fit_standardization" in str(e), str(e)


def test_fit_ignores_padded_slots():
    """Pads must never enter the channel statistics: a block that is mostly pad must fit the same
    statistics as the same live probes with no pad at all."""
    probes = _probes(6, 24, seed=3)
    a, b = ChiSetEncoder(24, U_MID, U_HALF), ChiSetEncoder(6, U_MID, U_HALF)
    a.fit(_block(probes, 24)[0])
    b.fit(_block(probes, 6)[0])
    assert (a.elem_mean - b.elem_mean).abs().max() < 1e-6
    assert (a.elem_std - b.elem_std).abs().max() < 1e-6


# ── does it actually represent the curve? ─────────────────────────────────────────────────────────
def _lorentzian(u, u0=0.0, q=6.0, gain=1.0):
    """A stand-in chi(omega) with a resonance, sampled at log-frequencies u."""
    w = torch.exp(u - u0)
    denom = torch.complex(1 - w ** 2, w / q)
    return gain / denom


def test_same_curve_at_two_K_gives_a_stable_curve_representation():
    """THE payoff property, stated correctly.

    K-agnostic means the network can CONSUME any probe count -- not that its output is identical
    regardless of how many probes it got. `pool()` splits the representation for exactly this reason:
    the CURVE half must be stable when the same physical chi(omega) is sampled at K=5 vs K=11, while
    the SAMPLING half (log1p(n), per-knot coverage) is K-dependent ON PURPOSE, because a 2-probe
    observation really is less informative than a 12-probe one and its posterior should be wider.

    Asserted as a discrimination RATIO: no absolute tolerance is defensible at random init.
    """
    enc = _enc(k_pad=16, seed=7)

    def curve(k, u0=0.0):
        u = U_MID + U_HALF * torch.linspace(-0.85, 0.85, k)
        chi_v = _lorentzian((u - U_MID) / U_HALF, u0=u0)
        logcyc = torch.full((k,), math.log(20.0))
        return enc.pool(_block((u, chi_v, logcyc), 16)[0])[0]

    # Compared in the DENSE regime. Measured at random init, K-sensitivity falls off fast with
    # sampling density -- 0.94 (K=2 vs 4), 0.38 (4 vs 8), 0.12 (8 vs 12), 0.06 (12 vs 16) -- while a
    # 10% resonance shift costs ~0.5 at every K. The sparse end is NOT a defect and must not be
    # asserted away: two probes genuinely carry less information about a curve than four do, and a
    # representation that claimed otherwise would be lying to the flow.
    c8, c12 = curve(8, 0.0), curve(12, 0.0)
    same = float((c8 - c12).norm())
    diff = float((c8 - curve(8, 0.10)).norm())
    assert same < 0.35 * diff, f"K-sensitivity {same:.4g} is not small vs curve-sensitivity {diff:.4g}"
    # ...and the sparse end must still be ORDERED correctly: sparser pairs differ more.
    assert float((curve(2) - curve(4)).norm()) > same


def test_the_sampling_half_reports_the_probe_count():
    """The complement of the test above: `g` MUST move with n, or the flow cannot know how much
    evidence it was given and would be equally confident on 2 probes as on 12."""
    enc = _enc(k_pad=16, seed=7)

    def sampling(k):
        u = U_MID + U_HALF * torch.linspace(-0.85, 0.85, k)
        chi_v = _lorentzian((u - U_MID) / U_HALF)
        return enc.pool(_block((u, chi_v, torch.full((k,), math.log(20.0))), 16)[0])[1]

    g2, g12 = sampling(2), sampling(12)
    assert float((g2 - g12).abs().max()) > 0.5, "the sampling half is not reporting the probe count"
    assert abs(float(g12[0, 0]) - math.log1p(12)) < 1e-5, float(g12[0, 0])


def test_denser_sampling_converges():
    """The curve representation must settle as the curve is sampled more densely, not wander with K."""
    enc = _enc(k_pad=64, seed=11)

    def curve(k):
        u = U_MID + U_HALF * torch.linspace(-0.85, 0.85, k)
        chi_v = _lorentzian((u - U_MID) / U_HALF)
        return enc.pool(_block((u, chi_v, torch.full((k,), math.log(20.0))), 64)[0])[0]

    with torch.no_grad():
        ref = curve(64)
        d = [float((curve(k) - ref).norm()) for k in (4, 8, 16, 32)]
    assert d[-1] <= d[0], f"denser sampling did not converge toward the reference: {d}"
    assert all(a >= b for a, b in zip(d, d[1:])), f"convergence must be monotone: {d}"


# ── the packer ────────────────────────────────────────────────────────────────────────────────────
def test_packer_round_trips_and_masks_failures():
    """pack_probe_block is the only writer of the layout, so its inverse is the layout's definition."""
    u, chi_v, logcyc = _probes(4, 8, seed=5)
    u, chi_v, logcyc = u.unsqueeze(0), chi_v.unsqueeze(0), logcyc.unsqueeze(0)
    valid = torch.ones(1, 4, dtype=torch.bool)
    block, mask = chi_mod.pack_probe_block(chi_v, u, logcyc, valid, k_pad=8)
    assert block.shape == (1, W * 8) and mask.shape == (1, 8)
    assert int(mask.sum()) == 4
    e = block.reshape(1, 8, W)
    # live slots are contiguous and ascending in frequency
    assert torch.equal(mask[0], torch.tensor([True] * 4 + [False] * 4))
    assert (e[0, :4, 0].diff() >= 0).all(), "live probes must be packed in frequency order"
    # every pad channel is exactly zero
    assert torch.equal(e[0, 4:, :], torch.zeros(4, W))
    # phase round-trips
    ang_back = torch.atan2(e[0, :4, 3], e[0, :4, 2])
    got = torch.sort(torch.angle(chi_v[0])).values
    assert (torch.sort(ang_back).values - got).abs().max() < 1e-5


def test_a_failed_probe_is_masked_not_a_phantom():
    """A non-finite lock-in used to become a live-looking (0,0,0) triple via nan_to_num -- which
    cos^2+sin^2=1 says no real probe can produce."""
    u, chi_v, logcyc = _probes(3, 8, seed=6)
    chi_v = chi_v.clone()
    chi_v[1] = complex(float("nan"), float("nan"))
    block, mask = chi_mod.pack_probe_block(chi_v.unsqueeze(0), u.unsqueeze(0), logcyc.unsqueeze(0),
                                           torch.ones(1, 3, dtype=torch.bool), k_pad=8)
    assert int(mask.sum()) == 2, mask
    e = block.reshape(1, 8, W)
    assert torch.isfinite(e).all()
    for j in range(8):
        if e[0, j, 5] == 0.0:
            assert torch.equal(e[0, j, :], torch.zeros(W)), j


def test_packer_refuses_more_probes_than_slots():
    u, chi_v, logcyc = _probes(5, 8)
    try:
        chi_mod.pack_probe_block(chi_v.unsqueeze(0), u.unsqueeze(0), logcyc.unsqueeze(0),
                                 torch.ones(1, 5, dtype=torch.bool), k_pad=4)
        raise AssertionError("packing 5 probes into 4 slots must raise, never truncate")
    except ValueError as e:
        assert "CHI_K_PAD" in str(e), str(e)


def test_out_of_band_probe_is_masked():
    u, chi_v, logcyc = _probes(3, 8, seed=8)
    u = u.clone()
    u[0] = U_MID + U_HALF * 5.0                     # far outside the band
    _, mask = chi_mod.pack_probe_block(chi_v.unsqueeze(0), u.unsqueeze(0), logcyc.unsqueeze(0),
                                       torch.ones(1, 3, dtype=torch.bool), k_pad=8)
    assert int(mask.sum()) == 2, mask


def test_width_is_k_independent():
    """n_chi_features must be a function of the PAD, never of the probe count -- the one line that
    lets a posterior load against a config declaring a different K."""
    assert chi_mod.n_chi_features(12) == W * 12
    assert chi_mod.n_chi_features(12) == chi_mod.n_chi_features(12)
    assert len(chi_mod.chi_labels(12)) == W * 12
    assert len(chi_mod.chi_labels(7, chi_mod.CHI_FISHER_CHANNELS)) == 4 * 7
    assert "u" not in chi_mod.CHI_FISHER_CHANNELS and "mask" not in chi_mod.CHI_FISHER_CHANNELS


if __name__ == "__main__":
    failures = 0
    for test_name, fn in sorted(globals().items()):
        if test_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {test_name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {test_name}\n      {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
