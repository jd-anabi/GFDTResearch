"""The SimConfig / FDTConfig data carriers, split out of config.py (which stays the constants
module and re-exports both classes permanently -- ``from core.config import SimConfig`` keeps
working everywhere).

Every module-level constant is read as ``config.NAME`` -- a LIVE read, never a from-import
snapshot -- so the chi default factories keep their read-at-construction semantics and a per-run
assignment to ``core.config`` is still honoured here.
"""
import os
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import cached_property

import torch

from core import config
from core.config import DeviceConfig, detect_device




# === SIMULATION CONFIG DATACLASS ===
@dataclass
class SimConfig:
    """
    Carries all state needed to run the SBI pipeline.
    Replaces the 9-element tuple that was threaded through setup() -> run().
    """
    # Model selection
    model: str
    labels: list[str]
    state_dep_drift: bool

    # Parsed from cell file
    inits_dict: OrderedDict # {name: val}
    params_dict: OrderedDict # {name: (val, (lo, hi))}
    rescale_params: OrderedDict # {name: (val, (lo, hi))}
    force_params_dict: OrderedDict # {name: (val, (lo, hi))}
    units_dict: tuple
    # si_factors was removed here: built by iterating SET-derived units_dict it had non-deterministic
    # order, so it could never be safely indexed, and nothing ever read it. Unit conversions go
    # through get_unit_conversion_factor / freq_si_to_cell / the *_unit properties, which match by
    # DIMENSION rather than position.

    # Multi-frequency susceptibility mode: replace the single-frequency forced conditioning with a
    # K-frequency chi(omega) curve (see config.CHI_MODE). Threads like has_forcing; set at build time
    # from config.CHI_MODE. Default False = the existing single-frequency / spontaneous pipeline.
    # The three knobs are carried ON THE CONFIG (not read from the module at use time) so a config --
    # and therefore a posterior trained from it -- is SELF-DESCRIBING: changing the global afterwards
    # cannot silently reinterpret an existing run. default_factory reads the module value LIVE at
    # construction, so the CLI still picks up config.CHI_* edits without passing anything.
    chi_mode: bool = False
    # chi_n_freqs is now "probes this OBSERVATION supplies" -- it no longer sets any width. chi_k_pad
    # is the NETWORK's slot capacity and is what every width is derived from, so a posterior trained
    # with K drawn over 2..12 loads against a config at chi_n_freqs = 4 or 9 with no guard loosened.
    chi_n_freqs: int = field(default_factory=lambda: config.CHI_N_FREQS)
    chi_k_pad: int = field(default_factory=lambda: config.CHI_K_PAD)
    chi_f0: float = field(default_factory=lambda: config.CHI_F0)
    chi_freq_bounds: tuple = field(default_factory=lambda: config.CHI_FREQ_BOUNDS)
    # The lock-in duration ceiling, in drive cycles. On the config for the same reason as the band:
    # it changes the logcyc a given recording reports, so a posterior trained under one ceiling and
    # evaluated under another is reading a different observable.
    chi_max_cycles: float = field(default_factory=lambda: config.CHI_MAX_CYCLES)

    # Decorrelating Fisher rotation, carried per-config for the same reason as the chi knobs: consumers
    # used to `from .config import REPARAM_ROTATE`, which snapshots at import and cannot be toggled at
    # runtime. Reading it from the config makes a trained posterior self-describing about whether the
    # rotation was intended (the <name>.rot.pt sidecar stores the resulting V).
    reparam_rotate: bool = field(default_factory=lambda: config.REPARAM_ROTATE)

    # Time / segmentation (legacy fallback fields; primary time setup uses dt_exp + T_obs)
    t_max: float = None
    dt: float = None

    # Experimental observation parameters (in cell file time units, set during setup)
    dt_exp: float = None          # camera frame interval
    t_min_exp: float = None       # shortest expected recording
    t_max_exp: float = None       # longest expected recording
    T_obs: float = None           # ground-truth observation duration (user input)
    # Resolved observation length in SAMPLES, written by orchestrator.generate_observations after any
    # cost-ceiling clipping. Downstream consumers (PPC in infer_and_visualize, the overlay figures)
    # MUST read this rather than recomputing int(T_obs/dt_exp): the two expressions are algebraically
    # equal but not numerically, and the ceiling branch writes T_obs back as N_obs*dt_exp, whose
    # re-truncation can land on N_obs-1. A one-sample disagreement there silently deleted all five
    # posterior-overlay figures. None => generate_observations has not run (the experimental paths,
    # which take their length from the recording itself).
    n_obs: int = None

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)

    def __post_init__(self):
        """Reject a chi geometry that cannot be represented, AT CONFIG BUILD.

        K > chi_k_pad has no valid packing, and the natural failure without this is a raise from
        deep inside pack_probe_block on the FIRST batch -- i.e. after the prior has been built and
        the training loop has started. Both numbers are named so the message says which knob to move.
        """
        if not self.chi_mode:
            return
        if not (2 <= int(self.chi_k_pad) <= config.CHI_K_MAX):
            raise ValueError(
                f"chi_k_pad={self.chi_k_pad} is out of range: it must be at least 2 and at most "
                f"CHI_K_MAX={config.CHI_K_MAX}. It is the network's probe-slot capacity and is frozen into "
                f"every posterior trained with it.")
        if not (1 <= int(self.chi_n_freqs) <= int(self.chi_k_pad)):
            raise ValueError(
                f"chi_n_freqs={self.chi_n_freqs} probes cannot be packed into chi_k_pad="
                f"{self.chi_k_pad} slots. Lower the probe count, or raise the pad (which invalidates "
                f"posteriors trained at the current pad).")
        # The floor masks a probe; the ceiling shortens it. If they cross, the ceiling truncates
        # every probe to under the floor and the packer masks the entire set -- an all-masked
        # observation, which the experimental path refuses outright. That would surface as "every
        # probe was masked" with nothing pointing at the two constants that closed on each other.
        if not (float(self.chi_max_cycles) > config.CHI_MIN_CYCLES):
            raise ValueError(
                f"chi_max_cycles={self.chi_max_cycles} does not clear CHI_MIN_CYCLES={config.CHI_MIN_CYCLES}. "
                f"The ceiling truncates a probe's lock-in and the floor masks it below that many "
                f"cycles, so a ceiling at or under the floor masks every probe in every observation.")

    # --- Derived properties ---
    @property
    def t_scale_bounds(self) -> tuple[float, float]:
        """(lo, hi) bounds on the t_scale rescaling parameter (λ/K_gs)."""
        _, (lo, hi) = self.rescale_params["t_scale"]
        return lo, hi

    @property
    def dt_nd_min(self) -> float:
        """Finest ND time step needed: dt_exp / t_scale_max."""
        _, t_scale_hi = self.t_scale_bounds
        return self.dt_exp / t_scale_hi

    @property
    def t_nd_max(self) -> float:
        """Longest ND duration needed: t_max_exp / t_scale_min."""
        t_scale_lo, _ = self.t_scale_bounds
        return self.t_max_exp / t_scale_lo

    @cached_property
    def t(self) -> torch.Tensor:
        """
        Pre-simulated ND time vector at finest resolution and longest duration.

        Cached: SimConfig is effectively immutable after build_sim_config(), so we
        allocate the 2.4M-point tensor once per config lifetime.
        """
        if self.dt_exp is not None:
            n_steps = int(self.t_nd_max / self.dt_nd_min)
            return torch.linspace(0, self.t_nd_max, n_steps,
                                  dtype=self.hw.dtype, device=self.hw.device)
        # fallback for legacy usage
        return torch.linspace(0, self.t_max, int(self.t_max / self.dt),
                              dtype=self.hw.dtype, device=self.hw.device)

    @property
    def steady_idx(self) -> int:
        """
        Index where transient ends and steady-state begins.

        Fixed number of fine integration steps corresponding to TRANSIENT_ND_UNITS
        ND time units — model-intrinsic, independent of prior bounds on T or t_scale.
        """
        steady_idx = int(config.TRANSIENT_ND_UNITS / self.dt_nd_min)
        # Safety check: transient must leave budget for at least the minimum output batch
        assert steady_idx < config.N_ND_MAX, (
            f"TRANSIENT_ND_UNITS={config.TRANSIENT_ND_UNITS} produces steady_idx={steady_idx} "
            f">= N_ND_MAX={config.N_ND_MAX}. Reduce TRANSIENT_ND_UNITS or raise N_ND_MAX."
        )
        return steady_idx

    @property
    def has_ground_truth(self) -> bool:
        """True once ground-truth VALUES are loaded (a bounds-built config has None value slots)."""
        rows = list(self.params_dict.values()) + list(self.rescale_params.values())
        return len(rows) > 0 and all(row[0] is not None for row in rows)

    def _require_ground_truth(self) -> None:
        if not self.has_ground_truth:
            raise ValueError(
                "SimConfig was built from a bounds file (no ground-truth values). Load a cell file via "
                "inject_ground_truth(...) before generating a synthetic observation, or use experimental data."
            )

    @property
    def ground_truth(self) -> list[float]:
        """Ground-truth values for all inferred params (ND + rescale). Requires a loaded cell."""
        self._require_ground_truth()
        nd = [row[0] for row in self.params_dict.values()]
        rescale = [row[0] for row in self.rescale_params.values()]
        return nd + rescale

    @property
    def ground_truth_tensor(self) -> torch.Tensor:
        return torch.tensor(self.ground_truth, dtype=self.hw.dtype, device=self.hw.device)

    @property
    def nd_params_bounds(self) -> list[tuple]:
        """Parameter bounds for prior construction."""
        return [row[1] for row in self.params_dict.values()]

    @property
    def inits_tensor(self) -> torch.Tensor:
        """Initial conditions as a (1, n_vars) tensor. A bounds-built config has no inits until a cell loads."""
        if not self.inits_dict:
            raise ValueError(
                "SimConfig has no initial conditions (built from bounds + units only). Load a ground-truth "
                "cell file via inject_ground_truth(...) before generating an observation."
            )
        return torch.tensor(list(self.inits_dict.values()), dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @property
    def params_tensor(self) -> torch.Tensor:
        """ND-only ground-truth parameters as a (1, n_params) tensor for the simulator. Requires a loaded cell."""
        self._require_ground_truth()
        nd = [row[0] for row in self.params_dict.values()]
        return torch.tensor(nd, dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @staticmethod
    def _fill_checked(label: str, cell_vals: dict, cfg_dict: OrderedDict, check_bounds: bool) -> list:
        """Validate a cell values dict against a config (val,(lo,hi)) dict, then fill in the values.

        MISSING is fatal -- the bounds file declares a parameter the cell cannot supply, and a None left
        in slot 0 would crash later in params_tensor. EXTRA cell values are IGNORED and returned, so the
        caller can report them: the BOUNDS file is the single source of truth for which parameters are
        inferred, so a cell carrying more than the bounds declare is merely over-specified. That is what
        lets one cell serve several bounds files -- e.g. a forced cell (f_scale + a Forcing section) used
        with a spontaneous bounds file that declares neither. cli._merge_vals_bounds already drops
        bounds-absent params exactly this way; being strict only here made the two paths disagree.
        """
        missing = sorted(set(cfg_dict) - set(cell_vals))
        if missing:
            raise ValueError(
                f"Cell file is missing {label} required by the bounds file: {missing}."
            )
        if check_bounds:
            oob = [f"{n}={cell_vals[n]} not in ({lo}, {hi})"
                   for n, (_, (lo, hi)) in cfg_dict.items() if not (lo <= cell_vals[n] <= hi)]
            if oob:
                raise ValueError(f"Cell file {label} outside the bounds file's bounds: " + "; ".join(oob))
        for n in cfg_dict:
            cfg_dict[n] = (cell_vals[n], cfg_dict[n][1])
        return sorted(set(cell_vals) - set(cfg_dict))

    def inject_ground_truth(self, inits: dict, param_vals: dict,
                            rescale_vals: dict, forcing_vals: dict) -> list:
        """
        Fill ground-truth VALUES + initial conditions from a cell file into a bounds-built config.

        SAFEGUARD: every ND and rescale (inferred) parameter the BOUNDS file declares must be present in
        the cell and lie within its bounds — else a clear ValueError listing the offenders. Forcing is the
        known DRIVE (conditioning, not an inferred param): it must be present, but its range is not
        enforced (a spontaneous cell legitimately uses amp=0/freq=0 outside the drive prior's range).

        Values the cell carries that the bounds file does NOT declare are IGNORED and returned, so the
        caller can note them — see _fill_checked. This is what lets one cell be used across the three
        observation modes (a forced cell against a spontaneous bounds file drops f_scale + the drive).

        :return: names of cell values that were ignored, tagged by section (may be empty).
        """
        ignored = [f"{n} (ND)" for n in
                   self._fill_checked("ND parameters", param_vals, self.params_dict, check_bounds=True)]
        ignored += [f"{n} (rescale)" for n in
                    self._fill_checked("rescale parameters", rescale_vals, self.rescale_params,
                                       check_bounds=True)]
        ignored += [f"{n} (forcing)" for n in
                    self._fill_checked("forcing parameters", forcing_vals, self.force_params_dict,
                                       check_bounds=False)]
        self.inits_dict = OrderedDict(inits)
        return ignored

    def set_observation_context(self, T_obs: float, forcing_vals: dict | None = None) -> None:
        """
        Set the observation duration (and optionally forcing VALUES) for the experimental-data branch,
        so PPC / eye-test simulators that read cfg.T_obs and cfg.force_params_dict values work.
        """
        self.T_obs = T_obs
        if forcing_vals is not None:
            for name, v in forcing_vals.items():
                if name in self.force_params_dict:
                    self.force_params_dict[name] = (v, self.force_params_dict[name][1])

    @property
    def inferred_labels(self) -> list[str]:
        """LaTeX labels (with units) for all inferred params (ND + rescale) for plotting.

        ND params keep their model LaTeX (self.labels); rescale params are rendered via
        Helpers.labels.rescale_axis_label so a corner/SBC plot shows e.g. ``$x_{\\mathrm{scale}}$ (nm/ND)``
        instead of the raw string ``x_scale``."""
        from .Helpers import labels as _labels
        rescale_labels = [
            _labels.rescale_axis_label(name, length_unit=self.length_unit,
                                       time_unit=self.time_unit, force_unit=self.force_unit)
            for name in self.rescale_params
        ]
        return self.labels + rescale_labels

    @property
    def has_forcing(self) -> bool:
        """Whether this model carries any external forcing parameters. False for a spontaneous model
        (a no-forcing user model, or BP whose bounds file has no forcing section). The SBI pipeline
        branches on this: a no-forcing config skips the forced run + Group-G lock-in and drops the
        forcing conditioning block, so its forcing_idx has none of amp/freq/phase to key on."""
        return len(self.force_params_dict) > 0

    @property
    def observation_mode(self) -> str:
        """Which of the THREE observation protocols this config describes.

          "spontaneous"  chi off, no drive     -- ONE passive trace. Groups A-F, Group G zero-padded;
                                                  conditioning [S(41) | log T]. f_scale is inert here
                                                  (it only ever divides a force) so it should not be in
                                                  the inferred set -- give such a cell a bounds file with
                                                  neither a Forcing section nor f_scale.
          "forced"       chi off, has_forcing  -- passive + ONE forced trace at the cell's own drive.
                                                  Conditioning [S(41) | log T | forcing].
          "chi"          chi on                -- passive + K single-tone forced traces. Conditioning
                                                  [S(41, G=0) | log T | chi(3K)]. The cell's own drive is
                                                  IGNORED (chi probes at mult_k * measured Omega_0), so
                                                  this is independent of has_forcing.

        The mode is chosen by which BOUNDS file is picked (has_forcing == "it declares a Forcing
        section") plus the chi toggle. The three conditioning widths cannot collide (K >= 2 is enforced),
        so loading a posterior trained in a different mode fails loudly on shape rather than silently.
        """
        if self.chi_mode:
            return "chi"
        return "forced" if self.has_forcing else "spontaneous"

    @property
    def forcing_idx(self) -> dict[str, int]:
        """Maps forcing param names to column indices, e.g. {"amp": 0, "freq": 1, ...}."""
        return {name: i for i, name in enumerate(self.force_params_dict.keys())}

    @property
    def rescale_idx(self) -> dict[str, int]:
        """Maps rescale param names to column indices, e.g. {"x_offset": 0, "x_scale": 1, ...}."""
        return {name: i for i, name in enumerate(self.rescale_params.keys())}

    @property
    def nd_idx(self) -> dict[str, int]:
        """Maps ND param names to column indices. The counterpart to rescale_idx/forcing_idx, added
        for the tier-1 constraint, which needs ``n`` and ``beta`` BY NAME -- a bounds file's ORDER is
        the source of truth for columns, so only its names are safe to reference."""
        return {name: i for i, name in enumerate(self.params_dict.keys())}

    @property
    def sim_rescale_idx(self) -> dict[str, int]:
        """The rescale index the SIMULATOR sees. Identical to ``rescale_idx`` unless this box declares
        ``T`` instead of ``f_scale``, in which case T's column is renamed to f_scale -- see
        core/SBI/derived.py, which owns that rule."""
        from core.SBI import derived
        return derived.sim_rescale_idx(self.rescale_idx)

    @property
    def tier1_args(self) -> tuple:
        """``(nd_idx, k_b_cell)`` for a tier-1 box; ``(None, None)`` for every other config.

        ⚠ LAZY ON PURPOSE, and it was NOT lazy first time round. ``k_b_cell`` needs a FORCE unit in
        the units file and raises without one -- and Python evaluates call arguments eagerly, so
        writing ``to_sim_rescale(..., cfg.nd_idx, cfg.k_b_cell)`` blew up on every config with no
        force token, even though that function returns early for exactly those. It surfaced as
        `test_no_forcing_user_model_full_sbi_pipeline` dying inside the Fisher rotation with
        "No unit with dimensionality [mass] * [length] / [time] ** 2 found in the units file."

        So: every caller passes ``*cfg.tier1_args`` and nothing evaluates the constant unless the box
        actually declares T.
        """
        from core.SBI import derived
        if not derived.uses_derived_f_scale(self.rescale_idx):
            return None, None
        return self.nd_idx, self.k_b_cell

    @property
    def k_b_cell(self) -> float:
        """Boltzmann's constant in CELL units: (cell force) x (cell length) per kelvin.

        Derived from the units file rather than hard-coded, for the same reason ``freq_si_to_cell``
        is: a constant baked for nm/pN silently mis-scales a cell declared in um/nN by nine orders of
        magnitude. For the nm/ms/pN/kHz master cell this is 1.380649e-2 pN*nm/K, so k_B*T at 300 K is
        4.14 pN*nm -- the number to sanity-check a derived f_scale against.
        """
        return (config.K_B * self.get_unit_conversion_factor("N")
                * self.get_unit_conversion_factor("m"))

    def get_unit_conversion_factor(self, si_unit: str) -> float:
        """
        SI unit -> cell file equivalent unit conversion factor.

        Finds which unit in the cell file has the same dimensionality as si_unit,
        and returns the multiplicative factor to convert from SI value to cell value.

        Examples:
          - get_unit_conversion_factor("s")  -> 1000.0 if cell uses ms
          - get_unit_conversion_factor("N")  -> 1e12 if cell uses pN

        NOTE: do NOT use this for a drive FREQUENCY -- use ``freq_si_to_cell``, which derives the
        factor from the TIME unit. See that property for why.

        :param si_unit: SI unit string (e.g. "s", "N", "rad").
        :return: Conversion factor: cell_value = si_value * factor.
        :raises ValueError: If no unit in the cell file matches the given dimensionality.
        """
        ureg = self._ureg          # cached -- a fresh UnitRegistry per call is expensive and this is hot
        target_dim = ureg.Quantity(1, si_unit).dimensionality
        for unit_str in self.units_dict:
            try:
                if ureg.Quantity(1, unit_str).dimensionality == target_dim:
                    return ureg.Quantity(1, si_unit).to(unit_str).magnitude
            except Exception:                  # noqa: BLE001 -- undefined token; skip
                continue
        raise ValueError(f"No unit with dimensionality {target_dim} found in the units file.")

    @property
    def freq_si_to_cell(self) -> float:
        """Drive-frequency conversion factor: ``freq_cell = freq_Hz * freq_si_to_cell``.

        Frequency is INVERSE CELL TIME *by construction*, not an independently declared unit:
        core/forcing.py builds ``t_dim`` in cell time units and evaluates ``sin(2*pi*freq*t_dim)``, and
        statistics.py / chi.py index their FFTs with ``dt`` in cell time units. So a 30 Hz drive in an
        ``ms`` cell is 0.03 cycles/ms, NOT 30.

        Deriving the factor from the TIME unit is what makes it correct regardless of what (if anything)
        the units file declares for frequency. Matching a declared "Hz" token instead resolves to 1.0
        against an ``ms`` cell and inflates every experimental drive by 1000x -- the bug this replaces.
        Use ``check_unit_consistency()`` to surface a units file whose frequency token disagrees.
        """
        return 1.0 / self.get_unit_conversion_factor("s")

    def check_unit_consistency(self) -> list[str]:
        """Human-readable warnings where the DECLARED units disagree with how the pipeline uses them.

        Units *declare* what the numbers in the bounds/cell files mean; they are never auto-converted.
        So a declaration that contradicts the pipeline's own convention silently mis-scales real data.

        Check: the declared FREQUENCY token must be the reciprocal of the declared TIME token (Hz with s,
        kHz with ms, ...), because drive frequency is consumed as inverse cell time (see freq_si_to_cell).
        """
        msgs: list[str] = []
        t_tok, f_tok = self.time_unit, self.freq_unit
        if t_tok and f_tok:
            try:
                declared = self._ureg.Quantity(1, f_tok).to(f"1/{t_tok}").magnitude
            except Exception:                  # noqa: BLE001 -- unconvertible pair; nothing to assert
                return msgs
            if abs(declared - 1.0) > 1e-9:
                msgs.append(
                    f"Units file declares frequency in '{f_tok}' but time in '{t_tok}'. The pipeline "
                    f"consumes drive frequency as INVERSE CELL TIME (1/{t_tok}), so a '{f_tok}' value is "
                    f"off by {1.0 / declared:g}x. Declare the reciprocal of the time unit (e.g. kHz for "
                    f"ms) or drop the frequency token entirely -- it is display-only, and conversions "
                    f"derive from the time unit."
                )
        return msgs

    @cached_property
    def _ureg(self):
        return config.unit_registry()

    def _resolve_unit(self, si_unit: str) -> "str | None":
        """The cell's unit TOKEN whose dimensionality matches ``si_unit`` (e.g. "s" -> "ms"), or None.

        units_dict is set-derived (unordered), so match by DIMENSIONALITY, never by index."""
        ureg = self._ureg
        try:
            target = ureg.Quantity(1, si_unit).dimensionality
        except Exception:                      # noqa: BLE001
            return None
        for tok in self.units_dict:
            try:
                if ureg.Quantity(1, tok).dimensionality == target:
                    return tok
            except Exception:                  # noqa: BLE001 -- undefined token; skip
                continue
        return None

    @cached_property
    def length_unit(self) -> "str | None":
        """Cell length unit token (e.g. "nm") for displacement axis labels."""
        return self._resolve_unit("m")

    @cached_property
    def time_unit(self) -> "str | None":
        """Cell time unit token (e.g. "ms"). Note: trace TIME axes are shown in seconds; this is only for
        the rescale-param label t_scale (ms/ND)."""
        return self._resolve_unit("s")

    @cached_property
    def force_unit(self) -> "str | None":
        """Cell force unit token (e.g. "pN"); None for BP, which declares no force unit.

        CAVEAT for a model with NO f_scale (Hopf): forcing.py then falls back to the Hopf-style nondim
        f_scale = x_scale / t_scale, so the EFFECTIVE force unit is length/time (nm/ms) regardless of
        what the units file declares. The declared token is still used to convert an experimental drive,
        so for such a model the declaration is a labelling convention, not a derived quantity."""
        return self._resolve_unit("N")

    @cached_property
    def freq_unit(self) -> "str | None":
        """Cell frequency unit token (e.g. "Hz"); None for BP."""
        return self._resolve_unit("Hz")


# === FDT CONFIG DATACLASS ===
@dataclass
class FDTConfig:
    """
    Carries all state needed to run the FDT analysis pipeline.
    Parallel to SimConfig but minimal: no prior/posterior/inference plumbing.
    """
    # Shared with SimConfig (model identity + parsed cell file)
    model: str
    state_dep_drift: bool
    inits_dict: OrderedDict           # {name: val}
    params_dict: OrderedDict          # {name: (val, (lo, hi))}
    rescale_params: OrderedDict       # {name: (val, (lo, hi))}
    force_params_dict: OrderedDict    # {name: (val, (lo, hi))}
    units_dict: tuple

    # FDT-specific knobs (sensible defaults; overrideable in build_fdt_config)
    n_freqs: int = 60
    # Multipliers of omega_0 for the Campaign-2 production grid.
    # Asymmetric in log space by design: below = 1 decade, above = 1.5 decades
    # (=> ~50% more drive frequencies above omega_0, to capture FDT recovery
    # at the high-frequency end while still resolving the active band below).
    freq_bounds: tuple = (0.1, 30.0)
    ensemble_M: int = 256              # trajectories per Campaign-2 frequency
    freqs_per_batch: int = 1           # frequencies packed per simulator call in Campaign 2
    F0: float = 0.05                   # ND forcing amplitude (within linear regime)
    burn_in_nd: float = 100.0
    T_obs_periods: int = 30
    dt_nd: float = 0.01
    psd_T_obs_nd: float = 8000.0       # Campaign-1 steady-state duration

    # Filled in by run_fdt after cfg is built (from params_dict["k"])
    omega_0: float = None

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)

    # --- Derived ---
    @property
    def inits_tensor(self) -> torch.Tensor:
        """(1, n_vars) tensor of initial conditions."""
        return torch.tensor(list(self.inits_dict.values()),
                            dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @property
    def params_tensor(self) -> torch.Tensor:
        """(1, n_params) Nadrowski ND params."""
        nd = [row[0] for row in self.params_dict.values()]
        return torch.tensor(nd, dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    def params_for_M(self, M: int) -> torch.Tensor:
        """Tile ND params to shape (M, n_params) for ensemble batching."""
        return self.params_tensor.expand(M, -1).contiguous()

    def inits_for_M(self, M: int) -> torch.Tensor:
        """Tile initial conditions to shape (M, n_vars)."""
        return self.inits_tensor.expand(M, -1).contiguous()

    def with_overrides(self, **kwargs) -> "FDTConfig":
        """
        Return a shallow copy with overridden values.

        Keys may be:
          - ND parameter names from params_dict (overrides value, preserves bounds);
            used by passive-baseline sanity check (temp=1.0, tau_c=0.0)
          - any top-level FDTConfig field (n_freqs, F0, ensemble_M, ...)
        """
        nd_keys = set(self.params_dict.keys())
        top_kwargs = {k: v for k, v in kwargs.items() if k not in nd_keys}
        nd_kwargs = {k: v for k, v in kwargs.items() if k in nd_keys}

        new_params = OrderedDict(self.params_dict)
        for k, v in nd_kwargs.items():
            _, bounds = new_params[k]
            new_params[k] = (v, bounds)

        return replace(self, params_dict=new_params, **top_kwargs)
