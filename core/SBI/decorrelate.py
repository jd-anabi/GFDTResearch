"""
Simulation-based Fisher eigenbasis for the decorrelating reparameterization (Track A).

V = eigenvectors of the latent Fisher F = J^T J, where J is the standardized feature-Jacobian
w.r.t. the flow's latent coordinate z at the ground-truth operating point (perturb z -> T(z) ->
simulate -> features). Rotating the flow's coordinate by V decorrelates the (near-degenerate)
posterior so the flow can calibrate it.

The Jacobian is built over THE FEATURES THE POSTERIOR ACTUALLY CONDITIONS ON, which differ by
observation mode:
    spontaneous  41 features (Groups A-F; Group G is zero-padded and contributes zero rows)
    forced       41 features (Group G populated by the drive response)
    chi          41 + 3K -- Groups A-F (Group G zeroed) plus the chi FISHER block, which is
                 CHI_FISHER_CHANNELS (log|chi|, cos, sin) per probe -- NOT the 6-channel
                 conditioning block, and no longer the 4-channel set that included logcyc (C-9/C-10)
Getting that wrong is silent: a Fisher built over the single-frequency feature set describes a
different experiment than the one being run, so V would decorrelate the wrong thing.

No trained posterior is needed -- F comes from the simulator alone, so this generalizes to any
model. V = I (REPARAM_ROTATE=False) recovers the plain pipeline exactly. Validated end-to-end by
scripts/reparam_selftest.py (orthogonality, bijection round-trip, decorrelation).
"""
import math

import numpy as np
import torch

from core import forcing as _forcing
from core.config import CHUNK_LEN, REPARAM_FISHER_M, REPARAM_FISHER_DZ, REPARAM_FISHER_POINTS
from core.SBI import chi as _chi
from core.SBI import pipeline
from core.SBI import statistics
from core.SBI import derived
from core.SBI.reparam import build_inferred_bijection, fisher_eigenbasis


def _default_inits(cfg, dtype, device) -> torch.Tensor:
    """
    (1, n_vars) model-default initial conditions for the GT-free rotation, matching the training loop's
    own synthesis (pipeline.INIT_SHAPES: randint pos + zero prob), seeded for a deterministic Fisher.
    The transient (steady_idx) washes these out, so the specific values do not matter.
    """
    from core import registry
    if registry.is_user_model(cfg.model):        # user models declare their own inits (no INIT_SHAPES row)
        from core.SBI.Priors.user_prior import declared_inits
        return declared_inits(registry.get(cfg.model)).to(dtype=dtype, device=device)
    n_pos, n_prob = pipeline.INIT_SHAPES[cfg.model.lower()]
    rng = np.random.RandomState(0)
    inits = np.concatenate([rng.randint(0, 10, size=(1, n_pos)), np.zeros((1, n_prob))], axis=1)
    return torch.tensor(inits, dtype=dtype, device=device)


def _representative_forcing(cfg, force_prior, dtype, device) -> torch.Tensor:
    """
    (1, n_forcing) representative in-distribution DRIVE for the GT-free rotation: the forcing-prior
    median if available, else the forcing-bounds midpoints. Gives Group-G features something to respond
    to (a GT-free config has no forcing values).
    """
    if force_prior is not None:
        s = force_prior.sample((256,)).to(device=device, dtype=dtype)
        return s.median(dim=0).values.reshape(1, -1)
    mids = [(lo + hi) / 2.0 for _, (lo, hi) in cfg.force_params_dict.values()]
    return torch.tensor([mids], dtype=dtype, device=device)


def build_latent_fisher_rotation(cfg, T=None, m: int = None, dz: float = None,
                                 latent_prior=None, n_points: int = None,
                                 force_prior=None, with_values: bool = False):
    """
    Decorrelating rotation V (P, P), P = ND + rescale dims, from the latent Fisher, AVERAGED over
    n_points operating points (GT + prior draws). Averaging makes the (single, linear) rotation
    valid across the prior rather than only at GT — the multiplicative degeneracies curve away from
    GT, so a GT-only V re-correlates off-GT (see the K=10 SBC redistribution finding). Pairs with
    the log-space box (REPARAM_LOG_PARAMS), which linearizes those degeneracies in the first place.

    :param cfg: SimConfig (provides model, params, rescale, forcing, time grid, device).
    :param T: the box bijection (build_inferred_bijection(cfg)); rebuilt if None.
    :param m: ensemble per latent perturbation (default config.REPARAM_FISHER_M).
    :param dz: latent central-difference step (default config.REPARAM_FISHER_DZ).
    :param latent_prior: latent inferred prior to draw the extra operating points from. If None,
                         only GT is used (original GT-only behavior, regardless of n_points).
    :param n_points: number of operating points GT + (n_points-1) prior draws (default
                     config.REPARAM_FISHER_POINTS). n_points=1 => GT only.
    :param with_values: also return the Fisher eigenvalues (descending, aligned with V's columns).
                        They are what turns V from an ordering into a measurement of how much better
                        constrained one direction is than another -- see reparam.fisher_eigenbasis.
    :return: orthogonal V on cfg.hw.device; w = z @ V are the decorrelated flow coordinates.
             With ``with_values``, ``(V, eigenvalues)``.
    """
    T = T if T is not None else build_inferred_bijection(cfg)
    m = m or REPARAM_FISHER_M
    dz = dz if dz is not None else REPARAM_FISHER_DZ
    n_points = n_points or REPARAM_FISHER_POINTS
    dtype, device = cfg.hw.dtype, cfg.hw.device
    nd_dim = len(cfg.params_dict)
    P = nd_dim + len(cfg.rescale_params)
    # Observation length: a specific T_obs when a cell is loaded (scripts), else a representative
    # training length (a GT-free training config has no T_obs).
    t_rep = cfg.T_obs if cfg.T_obs is not None else cfg.t_min_exp
    N_obs = int(t_rep / cfg.dt_exp)
    # Drive + initial conditions: use the ground-truth cell when present (scripts, preserves the exact
    # legacy V); otherwise (GT-free training) use a representative in-distribution drive and the
    # model-default inits the training loop itself synthesizes, so the rotation reflects training, not GT.
    # A SPONTANEOUS config (no Forcing section in its bounds) has no drive to probe. The rotation is
    # still well-defined there -- the Fisher just measures how Groups A-F respond to a latent
    # perturbation, which is exactly the information such a posterior conditions on -- so branch rather
    # than refuse.
    #
    # chi mode used to be excluded upstream in build_posterior, on the stated grounds that "chi(omega)
    # already attacks the degeneracy the rotation targets". MEASURED FALSE (scripts/degeneracy_map.py,
    # master cell, 2026-08-05): k~x_scale is 0.98 in forced mode and 0.95 in chi mode -- essentially
    # untouched -- and k / x_scale still hold the two worst unique handles (0.102 / 0.147) under chi.
    # The rotation exists for exactly that alias, so chi mode now gets one too, built over the chi
    # feature set. chi IGNORES the cell's own drive, so it is checked BEFORE has_drive below.
    has_drive = cfg.has_forcing
    chi_mults = _chi.chi_multipliers_for(cfg) if cfg.chi_mode else None
    base_inits = cfg.inits_tensor if cfg.has_ground_truth else _default_inits(cfg, dtype, device)
    n_vars = base_inits.shape[-1]
    n_force_ch = _forcing.n_force_channels(cfg.model, cfg.forcing_idx, n_vars)
    if has_drive:
        if cfg.has_ground_truth:
            forcing_gt = torch.tensor([[v for v, _ in cfg.force_params_dict.values()]],
                                      dtype=dtype, device=device)
        else:
            forcing_gt = _representative_forcing(cfg, force_prior, dtype, device)
        amp_v, freq_v, phase_v = (forcing_gt[:, cfg.forcing_idx[k]] for k in ("amp", "freq", "phase"))
    else:
        forcing_gt = amp_v = freq_v = phase_v = None

    # NOTE on the float64 statistics stack below (an audited "performance opportunity", NOT taken).
    #
    # The stated rationale was that float64 here is wasted "for a Fisher that is immediately
    # .numpy()'d and whose eigenvectors are cast back to float32". That does not follow: this is a
    # CENTRAL DIFFERENCE. fisher_at computes (feats(z+dz) - feats(z-dz)) / (2*dz) on features that
    # are O(1) with dz = REPARAM_FISHER_DZ = 0.1, so the difference is small relative to the operands
    # and cancellation is the dominant error term -- exactly the situation where intermediate
    # precision matters even though the RESULT is rounded. Doing the FFT stack in float32 would eat
    # into the signal, and the output of this function is the rotation V: the coordinate system the
    # flow is trained in. A cheaper Fisher that quietly moves V is not a performance win.
    #
    # If this is ever revisited, the honest test is to build V both ways at several operating points
    # and compare the resulting eigenbasis -- not to reason from the output dtype.
    def feats(theta_row, mm):
        nd = theta_row[:nd_dim].unsqueeze(0).expand(mm, -1).contiguous()
        res = theta_row[nd_dim:]
        t_scale = float(res[cfg.rescale_idx["t_scale"]])
        subs = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
        n_fine = min(cfg.steady_idx + N_obs * subs, len(cfg.t))
        t_fine = cfg.t[:n_fine]
        n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))
        rv = res.unsqueeze(0).expand(mm, -1).contiguous()
        # TIER 1 (section 11.5): the Fisher must be built over the SIMULATED experiment, so the
        # drive amplitude here has to be the derived f_scale rather than the temperature sitting
        # in its column. A no-op for a box that declares f_scale. Note the gradient is still taken
        # with respect to the INFERRED coordinates (T among them) -- feats() is differenced in
        # latent space by fisher_at, which is upstream of this substitution, so V comes out in the
        # basis the flow actually trains in.
        rv = derived.to_sim_rescale(nd, rv, cfg.rescale_idx, *cfg.tier1_args)
        sim_ridx = cfg.sim_rescale_idx

        def s(f):
            return pipeline.gen_obs(model=cfg.model, params=nd, t=t_fine,
                                    inits=base_inits.expand(mm, -1).contiguous(), force=f,
                                    n_segs=n_segs, steady_idx=cfg.steady_idx,
                                    state_dep_drift=cfg.state_dep_drift, batch_size=mm, var_idx=0,
                                    dtype=dtype, device=device)[0][:, ::subs][:, :N_obs]

        def stats_no_flags(*a, **kw):
            """gen_stats WITHOUT its trailing valid-flag block.

            The flags are BINARY and near-constant across an ensemble at one operating point, which
            is precisely the shape `fnoise = max(std, 1e-9)` turns into an amplifier: a flag that
            happens to step between the +dz and -dz arms writes 1/1e-9 into J. That is the same
            defect C-9/C-10 removed `logcyc` for, and the reason `mask` is absent from
            CHI_FISHER_CHANNELS -- see the chi branch below, which spells out why a NEARLY constant
            channel is lethal where an exactly constant one is free. The flags say which features
            are real, which is a statement about the OBSERVATION and carries no gradient in theta.
            """
            return pipeline.gen_stats_features(*a, **kw)

        xsc = res[cfg.rescale_idx["x_scale"]].double()
        xof = res[cfg.rescale_idx["x_offset"]].double() if "x_offset" in cfg.rescale_idx else 0.0
        # The fixed seeds below are LOAD-BEARING -- common random numbers, so the zp/zm arms of the
        # central difference in fisher_at() share a noise realisation and the derivative is not
        # swamped by it. But torch.manual_seed is GLOBAL: build_latent_fisher_rotation runs inside
        # build_posterior immediately before train_nn, so on return the process RNG was left pinned
        # at seed 2 and every SDE noise draw of the 5000-batch training run became a deterministic
        # function of it -- two "independent" runs shared their entire noise realisation, and any
        # run-to-run variance study measured nothing. fork_rng keeps the CRN benefit and restores
        # the caller's stream on exit. Fork the CUDA generator too when we are on a GPU, since
        # manual_seed reseeds both and the simulation noise is drawn on `device`.
        fork_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            if cfg.chi_mode:
                # [S(41, Group G zeroed) | chi FISHER features (3K)] -- NOT the conditioning block.
                #
                # The conditioning block carries `u = log(f_k/f_peak)`, `logcyc` and a `mask`, and ALL
                # THREE poison a Fisher, for one reason with three faces: `fnoise = max(std, 1e-9)` is
                # a DENOMINATOR, so a barely-varying channel is an amplifier. Here the placement is a
                # deterministic multiplier grid, so `u` is theta-INDEPENDENT by construction (std
                # ~2.5e-8 of pure rounding) and `logcyc` is either an exact duplicate of A3_log_fpeak
                # or, where the CHI_MAX_CYCLES ceiling binds, floor() quantization. The central
                # difference then writes entries of ORDER 1 to 1e4 -- the magnitude of a real
                # standardized feature or far beyond -- into up to K x P cells of the Jacobian that
                # defines the coordinate system the flow trains in, while V stays orthogonal to 1e-4
                # and every test passes.
                #
                # Note the asymmetry, because it is why this hid so well: the 11 Group-G columns are
                # EXACTLY zero in chi mode and cost nothing (0/1e-9 = 0). It is the NEARLY constant
                # channel that is lethal, not the constant one. See trap CHI10 and backlog C-9/C-10.
                zero = _forcing.zero_force(mm, n_force_ch, t_fine.shape[0], dtype, device)
                torch.manual_seed(2)
                xs_d = xsc * s(zero).double() + xof
                spont = stats_no_flags(xs_d, None, cfg.dt_exp, None, None, None,
                                       device=device, spontaneous_only=True).numpy()
                # SEED AGAIN, right here. gen_chi_block runs K MORE simulations that are otherwise
                # completely unseeded, so the zp/zm arms of the central difference would see different
                # chi noise and the derivative would be swamped -- a plausible-looking, meaningless V.
                # Same trap as scripts/degeneracy_map.py (PRISM_HANDOFF trap X3).
                torch.manual_seed(3)
                # resolution_filter=False is MANDATORY here. The filter depends on f_peak, which
                # depends on theta, so a probe can CROSS the threshold between the +dz and -dz arms --
                # a mask step of 1 divided by fnoise's 1e-9 floor puts ~1e9 into J, and V becomes that
                # discontinuity rather than the Fisher geometry.
                # All four named, none re-sliced: `[:N]` on this return is what produced trap CHI10.
                chi_v, _u_v, _logcyc_v, _valid_v = pipeline.gen_chi_raw(
                    model=cfg.model, params_nd=nd, rescale=rv, x_spont_dim=xs_d.to(dtype),
                    t_fine=t_fine, inits=base_inits.expand(mm, -1).contiguous(),
                    rescale_idx=sim_ridx, n_segs=n_segs, steady_idx=cfg.steady_idx,
                    subsample=subs, N_points=N_obs, dt_exp=cfg.dt_exp,
                    multipliers=chi_mults, f0_nd=cfg.chi_f0,
                    # The duration ceiling STAYS ON here, unlike resolution_filter. The filter is a
                    # theta-dependent MASK and so poisons a central difference; the ceiling shortens
                    # the segment and is theta-independent given the probe frequency. Turning it off
                    # would build the rotation over a longer lock-in than the network ever sees.
                    max_cycles=cfg.chi_max_cycles,
                    state_dep_drift=cfg.state_dep_drift, resolution_filter=False,
                    dtype=dtype, device=device)
                # logcyc is deliberately NOT passed -- fisher_features takes one argument now. It is
                # neither used nor useful here: with the ceiling clear it is an exact duplicate of
                # A3_log_fpeak (measured: four of six rows agreeing to 6 significant figures in a real
                # rotation), and with the ceiling binding it is floor() quantization that a 1e-9-floored
                # fnoise amplifies. C-9/C-10, trap CHI10.
                fisher_block = _chi.fisher_features(chi_v)
                return np.concatenate([spont, fisher_block.double().cpu().numpy()], axis=1)
            if has_drive:
                force = pipeline.build_nondim_sin_force_tensor(forcing_gt.expand(mm, -1), t_fine, rv,
                                                               cfg.forcing_idx, sim_ridx)
                torch.manual_seed(1); xf = s(force)
                torch.manual_seed(2); xs = s(torch.zeros_like(force))
                return stats_no_flags(xsc * xs.double() + xof, xsc * xf.double() + xof, cfg.dt_exp,
                                          amp_v.expand(mm).double(), freq_v.expand(mm).double(),
                                          phase_v.expand(mm).double(), device=device).numpy()
            # Spontaneous: ONE unforced run; Group G is zero-padded, so those 11 columns contribute
            # nothing to J and the Fisher is driven purely by Groups A-F (the same features the
            # posterior sees).
            # n_force_channels, not n_vars: this is the one zero-force site the 2026-07-28 sweep
            # missed. The tensor is the largest single allocation in a Fisher evaluation and there
            # are ~216 of them per rotation, so an n_vars-wide one over-allocates 3x for Nadrowski
            # and 5x for BP. See forcing.n_force_channels -- the channel count is a property of the
            # model's DRIFT, so a driveless Hopf still needs 2 channels.
            zero = _forcing.zero_force(mm, n_force_ch, t_fine.shape[0], dtype, device)
            torch.manual_seed(2); xs = s(zero)
            return stats_no_flags(xsc * xs.double() + xof, None, cfg.dt_exp, None, None, None,
                                      device=device, spontaneous_only=True).numpy()

    def fisher_at(theta_row):
        """Per-point standardized feature-Fisher F_k = J^T J, or None if features are non-finite."""
        f0 = feats(theta_row, max(4 * m, 128))
        if not np.isfinite(f0).all():
            return None
        fnoise = np.maximum(f0.std(0), 1e-9)
        z0 = T.inv(theta_row)
        # Row count from the ACTUAL feature width, not len(FEATURE_LABELS): chi mode returns 41 + 3K,
        # and hardcoding 41 would have silently truncated the chi block out of the Jacobian.
        J = np.zeros((f0.shape[1], P))
        for i in range(P):
            zp = z0.clone(); zp[i] += dz
            zm = z0.clone(); zm[i] -= dz
            J[:, i] = (feats(T(zp), m).mean(0) - feats(T(zm), m).mean(0)) / (2 * dz) / fnoise
        return None if not np.isfinite(J).all() else (J.T @ J)

    # Anchor point, then (n_points-1) prior draws (if a prior was provided). GT-free training anchors
    # on the per-dim PRIOR MEDIAN (transformed to physical) instead of the ground-truth point.
    if cfg.has_ground_truth:
        points = [cfg.ground_truth_tensor]
    elif latent_prior is not None:
        z_med = latent_prior.sample((max(256, n_points),)).median(dim=0).values.to(device)
        points = [T(z_med)]
    else:
        raise RuntimeError("build_latent_fisher_rotation: need a ground-truth cell or a latent_prior anchor.")
    if latent_prior is not None and n_points > 1:
        z_samp = latent_prior.sample((n_points - 1,)).to(device)
        points += [T(z_samp[k]) for k in range(z_samp.shape[0])]

    with torch.no_grad():
        F_accum = np.zeros((P, P)); n_used = 0
        for k, theta_row in enumerate(points):
            Fk = fisher_at(theta_row.to(device))
            if Fk is None:
                print(f"[fisher] operating point {k} gave non-finite features; skipping", flush=True)
                continue
            F_accum += Fk; n_used += 1
    if n_used == 0:
        raise RuntimeError("Fisher rotation: all operating points produced non-finite features.")
    print(f"[fisher] averaged simulation Fisher over {n_used}/{len(points)} operating points "
          f"(GT + {n_used - 1} prior draw(s))", flush=True)
    F = torch.tensor(F_accum / n_used, dtype=torch.float64, device=device)
    if with_values:
        V, evals = fisher_eigenbasis(F, with_values=True)
        return V.to(dtype), evals.to(torch.float64).cpu()
    return fisher_eigenbasis(F).to(dtype)
