# PRISM — Engineering & Science Handoff

**Single source of truth.** Consolidated 2026-07-28 from the former `features_handoff.txt`,
`gui_handoff.txt` and `sbi_calibration_handoff.txt` (deleted; git history preserves them). Those three
overlapped, contradicted each other, and mixed timeless reference with session log.

PRISM is a research application for **simulation-based inference of hair-bundle model parameters**,
with three front-ends over one science core: an interactive CLI, a PySide6 desktop GUI, and a set of
non-interactive diagnostic scripts.

## 0. How to use this document

- **§1–§3 are the reference** a newcomer needs: how to run it, how it fits together, and the
  contracts you must not break.
- **§4 is current state** — read this before starting any run.
- **§5 is the trap list.** Each trap is a bug that was paid for once already. They are grouped by
  subsystem with stable prefixes (`G/P/S/C/Q/F/M/SIM/V/L/U`) so they can be cited and grepped.
- **§6 is open work**; **§7–§10 are audit findings** (bugs, performance, code/doc organisation, GUI
  layout). **Much of §7, §8 and all of §10 were remediated on 2026-07-28** — each section carries its
  own status banner, and the top Appendix A entry is the full account. The original diagnoses are
  kept even where fixed, because they explain why each change is shaped the way it is. **Three
  catalogued items did not survive verification** (§7.1's mechanism, §8.1's "25–200×" figure, and
  §8.1's `Solver()` hoist); each is corrected in place rather than quietly deleted.
- **Appendix A is history.** Nothing there is required to work on the code; it records why decisions
  were made, and which dead ends were already tried.

> **Line numbers drift.** Citations use `file.py` + a function or symbol name wherever the line would
> go stale. Where a line number appears, re-read the file first.

---

## 1. Orientation

### 1.1 The three front-ends

| Front-end | Entry | Notes |
|---|---|---|
| CLI | `python -m core` | The original. Interactive prompts. **Must keep working** — every GUI refactor was non-breaking and defaults to old CLI behaviour. |
| GUI | `run.bat` / `run.sh` (`python -m core.gui`) | PySide6. An *additional* front-end, not a replacement. |
| Scripts | `python scripts/<name>.py` | 11 non-interactive diagnostics. Env knobs are documented in each file's docstring. |

Both run scripts **must be launched from the repo root** — `config.py` builds `Resources/` paths from
`os.getcwd()` (there is a `__file__` fallback, but the run scripts are the supported path).

### 1.2 Environment

- **Interpreter:** the conda env `biophys-env` at `C:\Users\J\anaconda3\envs\biophys-env\python.exe`.
  There is no `.venv`; the default `python` on PATH does **not** have the dependencies.
- **Required env vars** for anything importing torch/PySide6:
  - `QT_QPA_PLATFORM=offscreen` (headless Qt; the test files also `setdefault` it)
  - `KMP_DUPLICATE_LIB_OK=TRUE` (torch+MKL duplicate OpenMP runtime on Windows/conda — without it,
    simulations abort with OMP Error #15)
  - `PYTHONPATH=<repo root>` when running from outside the repo.
- **Syntax check:** `python -m py_compile <path>`
- **GIT: make LOCAL changes only.** The user handles all git and remote operations.

**Version drift — re-verify before upgrading.** The GUI was verified against **sbi 0.25.0**,
PySide6 6.9.3, tqdm 4.67.1. `requirements.txt` pins **sbi==0.26.1** (a planned bump, not installed).
The cooperative-cancel design reaches into sbi's fit loop via its per-epoch print (`npe_base.py` in
0.25.0); if you upgrade, **re-verify that print still fires per epoch** or a mid-training cancel
loses its finest checkpoint. The tqdm traps (C1/C2) are tqdm-version-bound, not sbi-bound.

### 1.3 Tests

**There is no pytest.** Each test module has an `if __name__ == "__main__":` runner that executes
every `test_*` function and prints `PASS`/`FAIL` then `ALL PASSED`. Run each file directly.

| Suite | Tests | Covers |
|---|---|---|
| `tests/test_gui_progress.py` | 76 | tqdm classifier, Qt stack, panels, pop-out, nav/gating, Simulate stream + video, labels, **layout geometry** |
| `tests/test_user_models.py` | 29 | sympy parser/codegen, forcing kinds + the sin golden test, persistence, registry |
| `tests/test_user_sbi.py` | 25 | spontaneous + chi SBI paths, the built-in-path-unperturbed guard, memory/geometry regressions, **the box round-trip invariant, the posterior-mode decoder, the JIT/eager contract, the dt and off-grid guards** |
| `tests/test_fdt_user.py` | 5 | FDT for user models (FEATURE 1 v3 / B-d) |
| **Total** | **135** | |

> **COUNT them; do not trust a number written down.** And the grep **must be unanchored**: tqdm
> leaves a progress bar on the same line as some `PASS` lines, so `grep -c "^PASS"` silently
> undercounts (it once reported a green 29/29 run as 25/29).
>
> ```bash
> tr '\r' '\n' < out.txt | grep -c 'PASS  test_'
> ```
>
> Also pipe runs to a file rather than `| tail -N`, which truncates away most of the PASS lines.

---

## 2. Architecture map

```
core/
  __main__.py     CLI entry. Catches UnitParseError -> clean exit(1).
  cli.py          Prompt-free config builders (make_sim_config / make_fdt_config /
                  make_reduction_config / make_param_sweep_config), _parse_cell, _SWEEP_PRESETS,
                  UnitParseError.
  config.py       Constants + the SimConfig/FDTConfig dataclasses. detect_device()/cpu_device(),
                  paths, VALID_MODELS/VALID_LABELS, TRAINING_*, CHI_*, QUIET_SEGMENT_BAR,
                  SOLVER_BAR_DESC, memory_budget_elements().
  orchestrator.py The 5 stage functions: generate_observations -> build_prior -> build_posterior ->
                  validate_calibration -> infer_and_visualize. Plus save_*_artifacts and _emit/fig_sink.
  registry.py     Runtime model registry (ModelSpec, register, load_user_models, is_user_model,
                  is_sbi_user_model, state_dep_drift, fdt_support).
  forcing.py      Kind-dispatching force builders (sin/step/triangular/exponential),
                  build_user_force_tensor, and n_force_channels (the shared channel rule).
  Models/         nadrowski_model, hopf_model, bp_model(+_steady), user_model (sympy parse + torch).
  Simulator/      Simulator ABC + per-model subclasses + UserSimulator. SimulationError lives here.
  Solvers/sdeint.py  Euler-Maruyama (eager + torch.compile fast path) and an unused implicit solver.
  SBI/            pipeline (gen_obs/gen_stats/gen_training_data/train_nn/gen_chi_block),
                  statistics (the 41 features), chi (chi(omega) helpers), reparam (latent box +
                  rotation), decorrelate (Fisher rotation), analysis (SBC data + PPC), overlay
                  (posterior-overlay diagnostics), Priors/.
  FDT/            campaigns, spectral, fdt_pipeline, cross_validation, sanity.
  Reduction/      NWK->Hopf normal-form map. IRRELEVANT to SBI — do not reason about it here.
  Helpers/        file_manager, helpers, visualizers, labels, model_store.
  gui/            see §2.2
scripts/          11 non-interactive diagnostics.
Resources/        Bounds/ Cells/ Units/ Models/ (inputs) + Plots/ Priors/ Posteriors/
                  CrossValidation/ (generated outputs).
```

### 2.1 SBI data flow

```
Bounds+Cells+Units  ->  SimConfig  ->  build_prior     (stability-screened GMM over the ND box)
                                   ->  build_posterior (gen_training_data -> train_nn -> NPE flow)
                                   ->  validate_calibration (SBC / TARP / PPC)
                                   ->  infer_and_visualize  (corner / PPC / eye-test / overlays)
```

Training is **scale-batched**: each batch shares a Sobol-sampled `(t_scale, T)` pair, simulates at a
fine ND `dt` then downsamples to `dt_exp`, pre-filtered by `min(N_ND_MAX, len(t))`.

### 2.2 GUI package

```
core/gui/
  __main__.py   entry. Forces matplotlib Agg BEFORE any core.* import (trap G3).
  app.py        build_app() -> (QApplication, MainWindow); sets QUIET_SEGMENT_BAR; excepthook.
  main_window.py  NavShell over Home + 5 section screens; always opens on Home.
  screens/      nav_shell, home_screen, section_screen, inference_screen (owns SbiSession +
                refresh_gates), settings_screen, model_builder_screen.
  panels/       base_panel (BasePanel: controls column | results column; dispatch()),
                inference_tabs (the FIVE inference tabs), fdt_panel, reduction_panel,
                crossval_panel, simulate_panel + simulate_runner + simulate_export.
  widgets/      artifact_picker, figure_stack, figure_window, log_pane, progress_pane,
                live_hair_bundle, labeled_inputs, help_badge, param_grid, source_toggle, anim.
  session.py    SbiSession + ConfigDraft.       worker.py  Worker(QRunnable) + WorkerSignals.
  streams.py    redirect_streams + CancelToken + WorkerCancelled.    vt.py  tqdm chunk classifier.
  design.py / fonts.py / theming.py    the Fluent token + QSS layer.
  plot_watcher.py  NewPngWatcher: polls a plot dir and emits new PNGs.
```

**Navigation** is two levels deep: Home → one of Reduction Map / FDT Analysis / Parameter Inference /
Simulate (+ Settings and the model builder, reached from Settings). The app always opens on Home;
only geometry and each panel's own settings persist.

---

## 3. Contracts

These are the invariants. Breaking one is silent, not loud.

### 3.1 The model/solver contract

There is **no abstract base Model class** — `core/Models/__init__.py` is empty. Models are plain
duck-typed torch classes whose interface is defined by how the solver and simulator call them.

A model must expose:

- **`f(self, x, t) -> Tensor`** — the DRIFT. `x` is `(batch, d)`; **`t` is the INTEGER local step
  index, not physical time**. Returns `(batch, d)`. **Forcing is folded in here** as an additive term
  read from `self.force[:, ch, t]`; there is no separate force method.
- **`g(self)` OR `g(self, x)`** — the DIFFUSION: a **`(batch, d)` DIAGONAL amplitude vector**,
  elementwise, no off-diagonal. Zero-pad noiseless channels. Use `g(self, x)` together with
  `state_dep_drift=True` for multiplicative noise.
  > Do **not** copy `bp_model_steady.g()` or the archived `HarmonicOscillator` — they return full
  > `(batch, d, d)` MATRICES, which the eager solver does not consume.
- **`self.force`** — a settable public `(batch, n_channels, T)` tensor. The Simulator overwrites it
  per time segment via a **bare attribute set**, *not* the `.force` property (which rebuilds the model).
- **`self.device`** — the solver does `x0.to(sde.device)`. Accept device/dtype and `.to(...)` all
  param tensors.

**Solver** (`core/Solvers/sdeint.py`, the eager `euler`): Itô Euler-Maruyama,
`x_{i+1} = x_i + f(x_i, i)·dt + g·dW·sqrt(dt)`, `dW ~ N(0,1)`. `dt` is **derived** from `(ts, n)`,
never passed. `state_dep_drift` toggles `g()` once vs `g(x)` each step. `euler_compiled` is an
optional CUDA torch.compile fast path (needs `compiled_step`/`compiled_params`); `__sols` falls back
to eager automatically.

**Construction is POSITIONAL:**
```python
Model(*torch.unbind(params, dim=1), force, batch_size=..., device=..., dtype=...)
```
⇒ **the parameter ORDER in the bounds file MUST equal the constructor's positional arg order.**
`_set_up_model` is `@abstractmethod` and raises `simulator.SimulationError` (a `RuntimeError`) on bad
construction, chained `from` the original.

**OBSERVABLE: state column 0.** Every model's primary observable is variable index 0.

A model does **not** provide: `dt` (solver-owned), state dimension `d` (inferred from
`inits.shape[-1]`), or state names / initial conditions (from the cell file).

Everything in the hot path is torch; numpy appears only in force construction.

### 3.2 The force-channel rule

`forcing.n_force_channels(model, forcing_idx, n_vars)` is the **single source of truth** for how wide
a force tensor must be. It is a property of the model's **drift**, not of the cell's declared forcing
params:

| Model | Channels | Why |
|---|---|---|
| Nadrowski, BP | 1 | drift reads `force_step[:, 0]` only |
| Hopf | 2 | `HopfModel` indexes `force_step[:, 1]` **unconditionally**, even with no `amp_y` declared |
| User model | `n_vars` | `UserModel` adds `force[:, j, t]` to variable `j` |

Built-ins keep the legacy sinusoidal convention (2 channels iff `"amp_y"` in `forcing_idx`); user
models get one row per state variable, zeros where unforced, with forcing params suffixed
`<pname>_<var>` so they flow through the name-keyed `forcing_idx` machinery.

> This matters for memory as well as correctness: the SBI pipeline used to size its zero-force
> tensors `n_vars` wide, over-allocating the single largest tensor of a training batch 3× for
> Nadrowski and 5× for BP.

### 3.3 The Bounds / Cells / Units triple

```
Resources/Bounds/<model>/<cell>.txt   the param SET + ORDER (+ (lo,hi)). THE SOURCE OF TRUTH.
Resources/Cells/<model>/<cell>.txt    VALUES only (a ground-truth point) + initial conditions.
Resources/Units/<model>/units.txt     unit tokens (e.g. "nm ms pN kHz").
```

One bounds file **per cell**. `cli._parse_cell` derives the model from the cell's **parent folder**.
`params_dict` etc. are `OrderedDict{name: (value, (lo, hi))}`.

The bounds file is what declares **which** parameters are inferred, in what order, over what range —
and hence which observation mode you are in. This is why the GUI's Config tab cannot build a
`SimConfig` (see trap M1b).

`inject_ground_truth` is **fatal on MISSING parameters but tolerates EXTRA ones** (returned and
logged), so one cell can serve several bounds files — a forced cell against spontaneous bounds
simply drops `f_scale` and the drive.

### 3.4 Everything is nondimensional

Drift and noise are ND. Dimensional physics enters **only** via the rescale params
`x_scale` / `t_scale` / `f_scale`, which are **inferred, not derived**. Nothing auto-nondimensionalises.

### 3.5 Units

`freq` is **inverse cell-time by construction**: `forcing.py` builds `t_dim` in cell time units and
evaluates `sin(2π·freq·t_dim)`, so `freq` is cycles per cell-time-unit throughout the simulator,
statistics and chi. The conversion is `SimConfig.freq_si_to_cell = 1/factor("s")` — **never** by
matching a declared frequency token. `SimConfig.check_unit_consistency()` warns when the declared
frequency token is not the reciprocal of the declared time token.

`labels.py` is the single source of truth for display: `axis_label` / `rescale_axis_label` produce
LaTeX for matplotlib; `pretty_gui` produces Qt rich text for the GUI (which cannot render LaTeX).

### 3.6 Conditioning layout and the three observation modes

The flow is **never fed raw traces** — it conditions on a summary vector. The layout is
`[ S(41) | log(T_obs) | forcing-or-chi ]`: `log(T)` rides the summary pathway; the forcing/chi block
is a separate `EmbeddedNet` pathway. **Keep this order in sync** across `generate_observations`,
`gen_training_data`, `validate_calibration` and the experimental paths.

The mode is chosen by **which bounds file** is picked (does it declare a Forcing section?) plus the
chi toggle. The three widths cannot collide, so a cross-mode posterior load fails loudly.

| # | Mode | Data | Dims | Conditioning |
|---|---|---|---|---|
| 1 | `spontaneous` | one passive trace | 12 | 42 |
| 2 | `forced` | passive + one forced trace | 13 | 46 |
| 3 | `chi` | passive + K single-tone forced traces | 13 | 42 + 3K |

Mode 1 **drops `f_scale`**: it only ever divides a force, and this mode never builds a force tensor,
so its marginal would just return the prior. Mode 3 **ignores the cell's own drive** (chi probes at
`mult_k · measured Ω₀`), so it is independent of `has_forcing` and cannot be out-of-distribution in
the drive.

The Fisher rotation works for modes 1 and 2; **mode 3 stays unrotated** (chi already attacks the
degeneracy the rotation targets, and rotating it would need the chi features in the Fisher).

---

## 4. Current status

### 4.1 START HERE

**Every existing posterior is stale for the forced path, and no new one has been trained.**

- The 2026-07/26-27 forced retrain is **invalid** — it ran with a 10³ frequency error and an
  out-of-distribution (zeroed) drive. Do not compare anything against it.
- `posterior_07012026` (THE KEEPER) predates the re-scoped `freq`/`amp` bounds, so its forcing prior
  no longer matches the bounds files. It remains the reference for the **calibration story** below,
  but a like-for-like forced comparison needs a retrain under the new bounds.

**Recommended order:**

1. **A chi-mode run** — this is the point of all the recent work. GUI → Config (NADROWSKI, χ ON,
   K=6, F₀=0.2, 0.1–10×) → Prior (bounds `cell_2`, **reuse the existing linear ND prior** — chi does
   not change the ND box) → train a NEW posterior (conditioning is 42+3K wide, so old ones will not
   load) → Validate → Infer. Budget ≈ (K+1)/2 × the usual training time.
2. `scripts/sbc_characterize.py` (K=10 × n_cal=2000) against the keeper. The hypothesis: do
   `kappa`/`lambda`/`x_scale`/`t_scale` **tighten**, and does `x_scale`'s ~3% location bias **shrink**?
3. Cross-check with `scripts/degeneracy_map.py` that the `kappa~x_scale` / `lambda~t_scale` aliases weaken.

> ~~**Caveat:** the `scripts/` diagnostics were not updated for chi-mode.~~ **DONE** (2026-07-28,
> top Appendix A entry). All 11 now build their config through `scripts/_common.py`.
> `sbc_characterize`, `retrain_convergence` and `degeneracy_map` are chi-aware; `diagnose_fmax`,
> `identifiability_offgt` and `feature_candidate_test` **refuse loudly** under `CHI=1` because their
> metrics are defined over the 41-feature single-frequency set. A posterior whose mode disagrees with
> the config now fails in seconds, before the simulation spend, instead of after it.
>
> For step 3, run `degeneracy_map` in **both** modes (`CHI=0` then `CHI=1 CHI_K=6 CHI_F0=0.2`) —
> outputs are mode-suffixed so they no longer overwrite each other — and diff the new
> `=== top features per parameter ===` tables. That table is the actual payload: it says *whether the
> chi features are what broke the alias*, not merely that the alias weakened.
>
> **Budget note for step 1's Validate:** chi calibration costs `CAL_N_SCALES × (K+1)` simulations
> (~1400 at the defaults, K=6) — longer than the training it validates. Smoke it at
> `CAL_N_SCALES=32` first. Lowering it for the record run is a *different measurement*, not a faster
> one: it is `t_scale`'s effective SBC sample size.

### 4.2 The keeper posterior

`posterior_07012026.pt` — run-5, 13-dim, LINEAR box + multi-point averaged Fisher rotation. Sidecar
`{V: 13×13, log_params: []}`. Converged at 179 epochs.

**Verdict: borderline-MEETS the calibrated-joint bar, conservatively. Locked in, with two honest
caveats. Not a clean bill of health.**

K=10 × n_cal=2000 repeat study (`scripts/sbc_characterize.py`), KS-p median (fraction of 10 runs with
p<0.05), worst first:

```
x_scale 0.000(1.0)  f_scale 0.004(0.8)  phi 0.006(0.8)  kappa 0.008(0.6)  S 0.021(0.6)
dG 0.057(0.5)  N 0.077(0.4)
PASS-ish: lambda 0.170, temp 0.187, tau_c 0.259, tau 0.310, beta 0.312, t_scale 0.444
```

Failure mode (pooled 20k ranks): **every histogram ~flat within ~3pp**; max |mean−0.5| = 0.031
(x_scale); every tailTot ≤ 0.20 — slightly WIDE, i.e. safe, **never overconfident**.
TARP(joint) KS p=1.000, ATC +0.322. PPC mean|z| = 0.398, 0/46 outside.

- **CAVEAT 1 (out of scope):** `x_scale` ~3% one-sided **over**-estimate — a LOCATION bias, not width.
  `x_scale` is the chronic information-ceiling parameter (never calibrated in any config). Safe for
  interval coverage: **quote intervals, not point values.**
- **CAVEAT 2 (in scope):** `f_scale` ~1.5% one-sided **under**-estimate. Diagnosed
  (`scripts/diagnose_fscale.py`) as **not** the rotation but the LINEAR box on a 3-decade positive
  scale parameter (bounds 1–1000, GT=10 at box-fraction 0.009 / latent z=−4.7; 57% of the log-uniform
  prior mass sits in the flat sigmoid tail). **Log-scaling is RULED OUT** — it was tried and came out
  worse (see Appendix A).

> Do **not** oversell TARP=1.000 — it is at ceiling partly from the conservative/wide lean, and TARP
> is less sensitive than marginal SBC. The honest claim is *"no detectable overconfidence; coverage
> indistinguishable from ideal"*, **not** *"perfectly calibrated joint"*.

**KEY FINDING:** the SBC failures are **flow calibration, not an information deficit**. The model is
converged (not under-fit); the degeneracies are real (`kappa~x_scale` |cos|=0.94,
`lambda~t_scale` |cos|=0.96) but identifiability is sufficient prior-wide (every param pinned to
~0.3–7% of its prior range). The joint posterior is a thin 0.95-correlated ridge that the flow
mis-calibrates — tight posteriors are hypersensitive to small flow biases in SBC.

### 4.3 chi(ω) mode

**Implemented (machinery + tests); NOT yet a trained, calibrated posterior.** This is the only lever
that adds the missing *information* — a passive trace plus a single-frequency forced trace only see
the products `D·A_nd` and `(lambda_hb/k_gs)·tau_nd`; the **shape of chi(ω) over frequency** separates
`kappa`/`lambda`/`x_scale`/`t_scale` individually.

- **Drive protocol:** single-tone × K recordings. Per observation = 1 spontaneous run (Groups A–F +
  Ω₀) + K single-tone forced runs at `omega_k = mult_k·Ω₀`, where Ω₀ is the spontaneous PSD peak
  **measured from the passive trace**. `mult_k` is log-spaced over `CHI_FREQ_BOUNDS`, clamped below
  the `dt_exp` Nyquist. Cost ≈ (K+1)/2 × the keeper (~3.5× at K=6).
- **Amplitude:** drive at a FIXED **ND** amplitude (dimensional `amp = CHI_F0 · f_scale`) so
  linearity and lock-in SNR are uniform across the `f_scale` prior. chi cancels the drive amplitude
  in the linear regime, so `CHI_F0` is only a linearity knob.
- **`CHI_F0 = 0.2`, set by measurement.** An actively oscillating bundle has no clean linear-response
  regime near its own frequency — a weak drive is not "more linear", it is swamped by the spontaneous
  oscillation. The criterion is **reproducibility**. Measured |chi| coefficient of variation on
  `cell_2`: ND 0.05 → 0.21 / 0.17 / **0.62** (unusable at 3×Ω₀); ND 0.2 → 0.04 / 0.04 / 0.17 (usable
  everywhere, retains a ~10× |chi| range); ND ≥0.5 saturates via entrainment. Re-measure for a cell
  with very different Q or noise.
- **Enabling it:** use the **GUI Config tab** (χ toggle + K / F₀ / frequency range). The knobs are
  carried on the `SimConfig`, so a run is self-describing. (`config.CHI_MODE` is the module default
  the CLI picks up; the GUI passes explicit values.)
- **Tunables:** `CHI_N_FREQS` (linear K× cost), `CHI_FREQ_BOUNDS` span, `CHI_F0` (keep linear),
  rotation-in-chi (generalise `decorrelate.feats()` if a ridge remains), and whether to keep a
  single-frequency Group G alongside chi (likely redundant).

### 4.4 Capability matrix

| Section | Built-ins | User models |
|---|---|---|
| Simulate | ✅ | ✅ |
| Parameter Inference | ✅ | ✅ **no-forcing only** (`registry.is_sbi_user_model`); forced/zero-param stay Simulate-only |
| FDT | ✅ | ✅ (per-model `observable_noise_prefactor`; `registry.fdt_support` gates) |
| Reduction | NADROWSKI only | ❌ **excluded by design** — an intrinsic NWK→Hopf normal-form map |

> The SBI pipeline (summary stats, reparam, calibration) is **Nadrowski-TUNED**. It runs the
> machinery for a user model and produces a posterior + SBC/TARP, but calibration is not pre-tuned
> per model.

---

## 5. Traps — *** THINGS THAT WILL BITE YOU ***

Each of these cost real time once. Each has a regression test unless noted.

### G — Global (threading / figures / paths)

- **G1. WORKER LIFETIME.** A `Worker` created as a local in `dispatch()` is garbage-collected when
  `run()` returns, and Qt then **purges that sender's still-queued result/finished events** — the
  slots never fire and the panel stays busy forever. Fix: `worker.setAutoDelete(False)` **and** keep
  it in `self._workers` until the finished slot discards it. **Never start a raw Worker outside
  `BasePanel.dispatch`.**
- **G2. NEVER PAINT A WORKER-CREATED MATPLOTLIB FIGURE.** Wrapping a worker-built pyplot figure in a
  live `FigureCanvasQTAgg` and letting a shown widget paint it **deadlocks on matplotlib's global
  lock** — silent hang, no traceback. Fix: `BasePanel._png_fig_sink` renders to PNG bytes *on the
  worker thread*; `FigureStack` shows a QPixmap.
- **G3. Agg is forced** in `core/gui/__main__.py` **before any `core.*` import**. That is what makes
  it safe to run the pipeline on a worker thread — every un-refactored `plt.show()` becomes a no-op.
  Do not move or remove that line.
- **G4. `redirect_streams` swaps `sys.stdout`/`stderr` PROCESS-WIDE.** Therefore **only one panel may
  run at a time, app-wide** — `BasePanel._running` is a **class** attribute for exactly this reason.
  `streams._REDIRECT` is the backstop: a second redirect declines and warns rather than corrupting
  `sys.stdout` permanently.
- **G5. `Resources/` layout is per-model subfolders** — see §3.3. Pickers are repointed when a model
  combo changes.

### P — Progress / tqdm classification (`core/gui/vt.py`, `streams.py`)

The original bug: tqdm bars appended hundreds of rows to the log pane instead of redrawing one line.
**Root cause:** tqdm redraws a bar at `pos>0` as THREE atomic writes — `"\n"*pos`, then
`"\r"+frame+padding`, then `"\x1b[A"*pos`. The third has no terminator, so `frame + "\x1b[A"` stranded
in the reader's buffer and the *next* redraw's leading `"\n"` flushed it through the newline branch,
i.e. as a log line.

**Key insight of the fix:** every chunk tqdm writes is atomic and self-describing, so **classify per
chunk, not by scanning for terminators**. A paint's row index is not "newlines in the preceding
chunk" — the authority is the chunk that FOLLOWS: *a paint is at row n ⟺ it is followed by a pure
up-move of n*.

- **P1. Do NOT give `_SignalStream` a `fileno()` or an `encoding` attribute.** Without them tqdm's
  screen-shape probe fails, so `ncols` is None (frames are never `disp_trim`'d, which the frame regex
  relies on) and `nrows` is None (so every nested bar displays). Adding `encoding` also flips `ascii`
  and changes the glyphs.
- **P2. A bare `"\n"` chunk is THREE different things:** the terminator of a status line, the
  terminator of a plain `print()` (print writes text and newline as *two* chunks), or a real
  moveto/finalizer. Only the last is cursor motion.
- **P3. A bare `"\r"` is ALWAYS row 0** — tqdm only emits one when `pos == 0`.
- **P4. The pump's FINAL publish must happen ON THE PUMP THREAD** (`stop()` sets a flag and joins).
  Emitting inline from the GUI thread makes Qt resolve it as a DirectConnection, so those slots run
  *ahead* of ticks still queued in the event loop — silently scrambling the log into teardown-first order.
- **P5. The 15 Hz pump is LOAD-BEARING.** tqdm's `mininterval` does not bound the redraw rate here:
  `set_description()` and `reset()` both call `refresh()` → `display()` with no time gate, and the
  prior sweeps call both on every iteration.
- **P6. Rows are keyed by tqdm `pos`, NOT by desc.** The prior sweeps rewrite desc with a live counter
  every iteration; a desc-keyed map would mint a row per iteration — the original bug, reborn.
- **P7. The overall bar tracks the DEEPEST live row whose total > 1.** Not the outermost (the pos-0
  "Training neural posterior" bar wraps `range(1)` and would read 0% for the whole build), and not a
  sticky first-seen driver (sbi's NN training emits no tqdm bar at all, only a printed epoch counter).
- **P8. `close(leave=True)` at pos 0 paints the final frame then writes a bare `"\n"`** — byte-identical
  to a moveto(+1). Treat it naively and the finished bar stays in the pane at 100% and pegs the overall
  bar while the pipeline is still working. `StreamRouter._settle_closing()` resolves it on later evidence.

### S — Solver performance meter (`widgets/progress_pane.py`)

A top-level iteration takes ~10 s, so the overall bar sits still and the GUI reads as frozen. The
thing that *is* moving is the SDE solver, and its it/s is the number the user wants.

- **S1. The solver bar is found by its DESC PREFIX** (`config.SOLVER_BAR_DESC` → `"step (batch="`),
  **never by its row.** Its tqdm `pos` is 0, 1 or 2 depending on phase and panel.
- **S2. The solver bar must NEVER become a progress row.** A posterior build constructs 10k–30k of
  them — one per time segment — so a row would mean creating and destroying a widget every few seconds.
- **S3. The solver bar must be EXCLUDED from `_retarget()`.** Its total is in the tens of thousands
  and it is the deepest bar, so it would win the "deepest informative row" election every time and
  drag the overall bar through a full 0→100% sweep every second.
- **S4. tqdm flips to `s/it` BELOW 1 it/s**, so `" 2.50s/it"` means 0.4 it/s, not 2.5.
  `vt.parse_rate()` inverts it — read naively, a crawling solver would show *more* plus signs the
  slower it got.

### C — Cancellation (`core/gui/streams.py`)

Cancellation is **cooperative and needs zero core changes**: every `print()` and tqdm redraw funnels
through `_SignalStream.write()` on the worker thread, and sbi prints an epoch counter *inside* its fit
loop — so a flag checked in `write()` is a checkpoint reaching even inside sbi's training loop.
`WorkerCancelled` derives from **`BaseException`** so it sails through the pipeline's many
`except Exception`; it is caught by name only in `Worker.run`.

- **C1. tqdm's `refresh()` does a MANUAL `_lock.acquire()/display()/release()`, not a `with`.** Our
  cancel raises from inside `display()`'s `write()`, so the release is skipped and tqdm's global write
  lock **leaks** — the next `tqdm.__new__` then deadlocks. `redirect_streams`' `finally` calls
  `streams.reset_tqdm_lock()` when `cancel.fired`. **Do not remove it** — the pinning test *hangs*
  without it.
- **C2. tqdm runs a TMonitor DAEMON thread** that force-refreshes a quiet bar, writing to our stream
  *from its thread*. If that write consumed the cancel latch it would raise where nobody catches it
  and leave the worker to sail past a fired latch — silently losing the cancel. So `CancelToken.check()`
  **only raises on the OWNER thread** (`arm()` records it).

Latency: ~1 s almost everywhere, but up to one NN-training epoch (~10–60 s) or the `check_sbc` C2ST
block (~46 s) — those are silent, so the button reads "Cancelling…".

### Q — QSettings restore order (`core/gui/settings.py`)

- **Q1. SBI/FDT: restore the model FIRST** and call `_on_model_changed(model)` explicitly
  (`currentTextChanged` does **not** fire if the value already equals the default), **then** the
  pickers — a picker restored first gets wiped by the model's `refresh()`.
- **Q2. CrossVal: do NOT persist the cell-derived s/t grid lo/hi.** They are re-derived from the cell
  file; a saved value from a *different* cell is a stale, wrong bound. Only the free knobs persist.
- **Q3. Pickers: persist `combo.currentData()`** (the model-namespaced relpath) and restore via
  `findData` with a **−1 guard** (file gone → leave at default, never `setCurrentIndex(-1)`).

### F — Interactive figures (`widgets/figure_window.py`)

- **F1. THE Gcf DETACH — do not remove.** Every stage figure is pyplot-managed, so matplotlib bakes
  `_restore_to_pylab` into the pickle and `pickle.loads` **re-registers** the figure into the
  process-global `Gcf`. Left there it leaks for the process lifetime **and** is destroyed by
  `Worker.run`'s `plt.close("all")` — tearing the manager out from under a figure the user is viewing.
- **F2. Import `backend_qtagg` LAZILY**, inside `InteractiveFigureWindow.__init__`. It does not switch
  the active backend (the app stays on Agg). Eager import would drag Qt into app start and the headless
  tests. **Never call `plt.show()` / `matplotlib.use()` here.**
- **F3. WINDOW LIFETIME** (G1 again). Pop-outs are parentless top-levels; `FigureStack._windows` holds
  the only reference, or Python/Qt GCs the window the instant `show()` returns.

### M — Parameter Inference tabs (`panels/inference_tabs.py`)

**Why Config no longer builds the config:** a `SimConfig` cannot exist without a bounds file — the
bounds file declares which parameters are inferred and hence the observation mode. So Config records a
`ConfigDraft` and the **Prior** tab turns it into the `SimConfig`.

- **M1. GATING is `InferenceScreen.refresh_gates()`** — a `setTabEnabled` truth table re-run after
  every stage: Config always on; Prior once a *draft* exists; Posterior once `session.cfg` exists;
  Validate needs posterior **and** `inf_prior` — **not `force_prior`**, which is None for every
  no-forcing model and had made Validate permanently unreachable for exactly those; Infer needs just a
  posterior. After a re-gate, if the visible tab just got disabled it falls back to Config.
- **M1b. TWO screen entry points; mixing them up wipes your work.** `new_draft(draft)` **REPLACES**
  the whole session (a different model invalidates every artifact). `install_config(cfg)` sets
  `session.cfg` **IN PLACE** — deliberately not a new session, because Prior installs the config as
  the first step of building the prior.
- **M2. The Infer CELL PICKER follows the BUILT cfg's model**, not a live combo (there isn't one in
  that tab). The Prior tab's bounds picker has the same deferred-restore trap.
- **M2b. DIRECT ENTRY always SEEDS FROM the selected file.** Parameter names and ORDER belong to the
  model (simulators bind columns positionally), so hand-entry edits **numbers only** and can never
  invent a schema. Grids are not persisted for the same reason. Validate explicitly —
  `FloatField.value()` returns 0.0 on bad text.
- **M3. APP-WIDE CONTROL LOCK.** A run in *any* panel locks *every* panel's controls
  (`BasePanel._instances` WeakSet). With five sibling inference tabs, a picker ⟳ in another tab
  mid-run would otherwise corrupt the worker's stream.
- **M4. HELP BADGES:** `add_help_row(form, label, widget, help)`. Copy lives in a per-panel `HELP` dict
  and also surfaces in Settings → Help.

### SIM — Simulate section

- **SIM1. Construct the Simulator ONCE, up front.** *(Historical: `_set_up_model` used to call
  `exit()` → `SystemExit`, a BaseException `Worker.run` does not catch, so `_make_simulator` carried
  an `except SystemExit` translation. Fixed 2026-07-28 — every Simulator now raises, and the
  translation is gone.)*
- **SIM2. dt CONTINUITY:** a frame of `m` steps needs **`m+1`** grid points — `euler` derives
  `dt=(t1-t0)/(n-1)`. `m` points would make `dt != dt_nd_min` (silent timescale bug). `res[0]`
  duplicates the boundary, so emit `res[1:]`.
- **SIM3. `torch.no_grad()` around the loop is REQUIRED** — `Solver().euler` has no autograd guard of
  its own (that lived in `Simulator.__sols`, which this path bypasses).
- **SIM4. CANCEL** rides an injected `should_stop`, polled once per frame and raised **between**
  frames — never inside a tqdm redraw, so no write-lock leak.
- **SIM5. `sim.sde.force = force_chunk`** (bare attribute set) — **not** `sim.force=`, which rebuilds
  the whole model.
- **`cfg.hw = cpu_device()` is REQUIRED** in this path: the batch-1 loop is CPU-optimal and every
  tensor must share one device (the bare force set has no `.to(device)`).

### V — Video export (`panels/simulate_export.py`)

- **V1.** `buffer_rgba()[...,:3]` is a **non-contiguous** view; wrap in `np.ascontiguousarray` and crop
  to even dims (H.264).
- **V2.** Cancel rides the per-frame tqdm redraw, not `provide_stream`. Partial-file cleanup is in a
  `finally`, and `writer.close` is itself guarded (closing a 0-frame writer can raise).
- **V3.** GIF duration is **milliseconds**. MP4 needs ffmpeg — the panel pre-checks
  `ffmpeg_available()` on the GUI thread rather than dispatching a doomed job.
- **V4. OMP:** import imageio **after** torch (lazily, inside the writer). Importing it before torch
  loads a second `libiomp5md` → OMP Error #15 abort.

### L — Labels & units

- **L1. TIME axes are SECONDS everywhere.** `t_dim` is display-only and is converted at the SOURCE.
  Do **not** relabel the `t_scale` *parameter* axis to seconds — that plots a value in ms/ND.
- **L2. `SimConfig.inferred_labels` returns LaTeX.** Consumers render mathtext fine; the SBC console
  print shows raw `$...$` (cosmetic).
- **L3. pyqtgraph:** the displacement unit goes in the **label text** (`"x (nm)"`), *not*
  `setLabel(units=)` — pyqtgraph SI-prefixes a `units=` string and would mangle "nm". Time keeps
  `units="s"`.
- **L4.** Most plots are ND and already say "(ND)" — only the dimensional trace axes needed units.

### U — User-defined models

- **U1. PARAM DISCOVERY must ignore numeric literals.** `_identifiers` strips numbers (incl. `1e-3`,
  `0x1F`) **before** the identifier scan, else the mantissa tail becomes a phantom parameter — a dead
  column that desyncs the positional contract.
- **U2. `"E"` is an ORDINARY parameter**, not Euler's number (physics names `E` are common); only
  `"pi"` is a constant. Names shadowing `parse_expr`'s constructors are rejected.
- **U3. `build_stream_config` re-checks** that the compiled `param_names` EQUAL the emitted bounds
  file's ND key order and raises "out of sync" — else `torch.unbind` would silently mis-bind values by
  position (wrong physics, no error).
- **U4. `save_user_model` REFUSES** non-finite values, a `t_scale` ≥ the transient budget ceiling
  (would fail `SimConfig.steady_idx`'s assert *after* a clean save), Windows reserved device names,
  and built-in/existing names. It writes the triple FIRST and the **JSON LAST** (the registry loads
  from the JSON, so that is the commit point).
- **U5. `_refresh_model_combos` re-fires `_on_model_changed` ONLY when the selection actually
  changed** — the hooks reset the pickers, so firing on a mere item-list update would discard the
  user's picker selections app-wide on every save/delete.
- **U6. The builder's Validate/Save run a short smoke integration ON THE GUI THREAD.** Cheap, so no
  dispatch — but its solver's tqdm writes to the process-wide redirected streams, so they **refuse
  while `BasePanel._running`**. The screen is a plain QWidget, so it is not auto-locked by the
  app-wide control lock; this guard is the substitute.

### X — Cross-cutting traps found during the 2026-07-28 remediation

- **X1. `sdeint.Solver` MUST be resolved at CALL time, not hoisted to a module singleton.**
  `Simulator.__sols` constructs `sdeint.Solver()` once per time segment, which looks like an obvious
  thing to hoist — the handoff even listed it under §8.1. It is worth ~0.1 s per training round, and
  `tests/test_user_sbi.py::test_solver_failure_raises_instead_of_killing_the_process` **patches the
  class** to make every solver method raise. A singleton is built at import, so the patch silently
  stops applying. Implemented, caught by the suite, reverted. The reason is at the call site.
- **X2. `SimConfig.chi_mode` is a plain `= False`, NOT a `default_factory`.** The comment above the
  chi block says the knobs read the module value live at construction; that is true of
  `chi_n_freqs`/`chi_f0`/`chi_freq_bounds` and **false of `chi_mode`**. Only `cli.make_sim_config`
  bridges `config.CHI_MODE` onto a config, so anything that builds a `SimConfig(...)` by hand is
  permanently non-chi no matter what the global says. This is what made every diagnostic script
  silently measure the single-frequency information set.
- **X3. `gen_chi_block` runs K UNSEEDED simulations.** Any common-random-number scheme around it must
  seed *immediately before the chi block as well*, not only before the spontaneous run — otherwise
  the ±δ arms of a central difference see different chi noise and the derivative is swamped. The
  result looks plausible and means nothing. See `scripts/degeneracy_map.py`.
- **X4. Torch's `SigmoidTransform._inverse` clamps to `[tiny, 1-eps]`.** Do not write code (or bug
  reports) premised on the box round-trip producing `±inf`; it cannot on torch 2.9. Pinned by
  `test_box_roundtrip_never_yields_a_nonfinite_latent_target`, which exists precisely so a version
  bump that removes that clamp surfaces as a test failure rather than a dead training run.
- **X5. `CAL_N_SCALES` is `t_scale`'s effective SBC sample size.** Every row in a calibration batch is
  *assigned* that batch's `t_scale`, so their ranks are not independent. Lowering the pair count to
  buy wall-clock is a different measurement, and it damages precisely the parameter chi(ω) exists to
  separate. Batch SIZE is nearly free; batch COUNT is the cost and the statistics.

### Core-side contract hooks (Phase 0 refactors — non-breaking, defaults = old CLI behaviour)

- **R1.** `cli.UnitParseError(ValueError)`; `_parse_cell`/`_units_to_factors` **raise** instead of
  `print()+exit()`.
- **R2.** `build_prior`/`build_posterior` gained keyword-only `save=True, save_name=None`;
  `save=False` skips the prompt and all disk writes. Save bodies extracted to `save_*_artifacts`.
- **R3. FIGURE SINK.** The stage functions gained keyword-only `fig_sink(title, fig) -> None`;
  `orchestrator._emit` calls it if given, else the legacy `plt.show()`.
- **R4.** Pure, prompt-free config cores in `cli.py` (`make_sim_config`, `make_fdt_config`,
  `make_reduction_config`, `make_param_sweep_config`).
- **R5.** `run_fdt(cfg, *, skip_sanity=None, confirm_production=None)` — both default None ⇒ the old
  inline `input()` prompts. **A GUI must pass explicit bools**, or the worker blocks forever.

---

## 6. Open backlog

| ID | Item | Status |
|---|---|---|
| **B-c** | **Dark-theme matplotlib figures.** The remaining theming gap: figures/PNGs stay WHITE in dark mode. Means driving matplotlib rcParams/facecolors from the design tokens in the plotting layer. (pyqtgraph already follows the theme.) | OPEN |
| **L-1..18** | ~~GUI layout/sizing (§10).~~ | ✅ **DONE 2026-07-28** — tier 1 + tier 2, pinned by 4 new geometry tests |
| **B-e** | **A proper icon set.** Replace the unicode glyph buttons (⟳ ⚙ ← ?) with real icons (bundled SVGs behind a `.qrc`, or a licensed icon font). Sidesteps per-font glyph coverage quirks. | OPEN |
| **S-1** | **Per-parameter bounds/priors in the model builder.** Today `model_store._nd_bounds` emits placeholder boxes `(v ± max(|v|,1))`, whose negative half is unphysical for SBI (e.g. a noise strength `d0=0.05` gets `(-0.95, 1.05)`). Fine for Simulate (which uses the exact GT value); wasteful for SBI. Let the user set per-parameter `(lo, hi)` + prior type. **Keep the ND-section ORDER == `compiled.param_names`.** | OPEN |
| — | **A trained chi-mode posterior + the calibration payoff.** See §4.1. | OPEN — the priority |
| — | ~~Make the `scripts/` diagnostics chi-aware.~~ | ✅ **DONE 2026-07-28** — `scripts/_common.py`; see §4.1 |
| — | Non-atomic `torch.save`/HDF5 writes against a cancel (narrow window; worst case a partial-but-valid sweep file, not corruption). | OPEN, low |
| — | **Never run for real on a display** — ask before assuming these work: the Parameter-Inference Save buttons, Validate, Infer (simulated and experimental), a full FDT/CrossVal run, a cancel *during* live NN training, the nav click-through, the Simulate trace/heatmap/cancel, the model-builder round-trip, and the accent/Inter checkboxes. Headless tests cover the wiring, not the pixels. | ONGOING |

**Closed since the last handoff** (do not re-open): FEATURE 1 v1/v2/v3 (user models: Simulate → SBI →
FDT), backlog B-a (OS accent), B-b (Inter font), **B-d — FDT for non-Nadrowski, shipped as
`campaigns.observable_noise_prefactor` + `registry.fdt_support`, pinned by `tests/test_fdt_user.py`**,
S-2 (JIT solver fast path), UX1–UX5 (the five UI/UX requests), and the PySide6 pin
(`requirements.txt:37`).

---

## 7. Known bugs and risks

Catalogued 2026-07-28, then largely remediated later the same day — see the top Appendix A entry.

**Status: ALL of 7.1-7.11 are FIXED** (7.1 with an important correction — read its banner).
**7.12 is fixed for `scripts/` only**, via `_common.enable_warnings`; the core-side `warnings.warn`
OOD guards were always correct and are untouched. Each entry keeps its original description so
the reasoning survives; the fix is noted inline.

### 7.1 Non-finite training targets are never filtered — ✅ FIXED, and the diagnosis below is WRONG

> **READ THIS FIRST.** The mechanism described below does **not** occur on torch 2.9.0:
> `SigmoidTransform._inverse` clamps to `[tiny, 1-eps]` internally and `sigmoid` saturates at
> `0.9999998807907104`, so a value on (or outside) a bound inverts to a FINITE `±15.94 / −87.34`.
> The real defect is only that `thetas` was never checked; that check now exists and warns. No clamp
> was added to the hot path. Pinned by `test_box_roundtrip_never_yields_a_nonfinite_latent_target`.
> Severity here was overstated — it was listed first, and it was not the worst item in this section.

`SBI/pipeline.py` `train_nn`, `SBI/analysis.py` `gen_cal_data`

```python
nan_mask = torch.isfinite(data).all(dim=1)
safe_magnitude_mask = (torch.abs(data) < 1e15).all(dim=1)
valid_idx = nan_mask & safe_magnitude_mask
thetas = thetas[valid_idx]      # <- thetas is filtered BY data, never checked itself
```

`thetas` is the **latent** target, produced by `theta_transform.inv(...)` — a logit. Any physical
value landing exactly on a box bound maps to **±inf**. And `gen_training_data` *assigns*
`curr_thetas_rescale[:, rescale_idx["t_scale"]] = t_scale_k` from a Sobol draw over the **closed**
`[t_scale_lo, t_scale_hi]`, so a value on the bound is representable.

**Consequence:** a single ±inf row survives the filter and NaNs the NPE loss for the whole run, with
no diagnostic — either a garbage posterior or an uninformative NaN failure thousands of batches in.
**Fix:** filter `thetas` for finiteness too, and log how many rows were dropped.

### 7.2 `decorrelate` reseeds the global RNG and never restores it — ✅ FIXED (fork_rng; seeds kept)

`SBI/decorrelate.py` — `torch.manual_seed(1)` / `torch.manual_seed(2)` inside `feats`.

`build_latent_fisher_rotation` runs inside `build_posterior` **immediately before `train_nn`**. On
return the process RNG is pinned at seed 2, so every subsequent SDE noise draw in the 5000-batch
training run starts from a fixed state.

**Consequence:** two "independent" training runs share their entire noise realisation, so any
run-to-run variance study measures nothing. **Fix:** save and restore the RNG state around the
Fisher computation (`torch.random.fork_rng()`), or use a local `Generator`.

### 7.3 Two different formulas for the observation length — ✅ FIXED (cfg.n_obs; with 7.4)

`orchestrator.py` — `generate_observations` uses `N_obs = int(T_nd_obs / dt_nd_gt)`;
`infer_and_visualize` uses `N_points_obs = int(cfg.T_obs / cfg.dt_exp)`.

Algebraically equal, numerically distinct float expressions. Worse, the cost-ceiling branch writes
back `cfg.T_obs = N_obs * cfg.dt_exp`, and `int(N_obs*dt_exp/dt_exp)` rounds to `N_obs - 1` whenever
the product lands just below.

**Consequence:** `x_dim` (length `N_obs`) and the PPC traces (length `N_points_obs`) disagree, and
`plot_posterior_vs_truth` raises a raw matplotlib dimension error **at the very end of a multi-hour
run**. **Fix:** compute once and pass it.

### 7.4 A silent early-return drops all five posterior-overlay figures — ✅ FIXED (warns; try split 4 ways)

`orchestrator.py` `_emit_overlay_figures`:

```python
if traces.ndim != 2 or traces.shape[-1] != gt.shape[-1] or traces.shape[0] < 2:
    return          # no warning, no log line
```

Combined with 7.3 this is the *expected* path, not an edge case, so the overlay figures look as
though they were never implemented. The surrounding `except Exception` at least warns; this guard
does not. **Fix:** warn with the actual shapes.

### 7.5 Entering `0` at a CLI prompt silently selects the LAST item — ✅ FIXED (`cli._prompt_index`)

`cli.py` — `model = VALID_MODELS[model_num - 1]`, and the same pattern for cell files, bounds files
and saved artifacts. `0 - 1 = -1` → the last entry. An out-of-range *positive* number raises a bare
`IndexError` that the surrounding `except ValueError` does not catch. Several of these prompts
explicitly invite `0` ("'0' if you want to make from scratch").

**Consequence:** a fat-fingered prompt runs an entire pipeline against the wrong cell. **Fix:** range-check.

### 7.6 Group B's FWHM is a global span, not the peak's width — ✅ FIXED ⚠ CHANGES A CONDITIONING FEATURE

`SBI/statistics.py` `_group_b` — `above = psd > half` is evaluated over the **whole** spectrum, then
`fwhm = last_index_above - first_index_above`. Any second peak or harmonic above half the main peak's
power stretches that span across both lobes.

**Consequence:** `B1_log_Q` reads the span between two lobes rather than the resonance width, so
bimodal/harmonic-rich bundles — exactly the interesting regime — get systematically under-estimated
Q. Not a crash; a quietly wrong conditioning feature. **Fix:** walk outward from the argmax.

### 7.7 `implicit_euler` is dead code that is also wrong — ✅ FIXED (deleted)

`Solvers/sdeint.py` — reachable only via `Simulator.__sols(..., explicit=False)`, and `explicit` is
never passed False anywhere. Inside it: on convergence it stores `x_next` (the **previous** iterate),
not `x_temp`; it ignores `state_dep_drift` entirely (calls `sde.g()` with no state); and it builds
its time grid on **CPU** while every other solver passes `device=x0.device`.

**Consequence:** a trap for whoever enables it. **Fix:** delete it, or fix and test it.

### 7.8 Prior construction is not reproducible — ✅ FIXED (sorted() in all three priors + random_state)

`SBI/Priors/prior.py` + `nadrowski_prior.py` — accepted points come out of a Python **`set`** of
float tuples (iteration order unspecified), and `GaussianMixture(...)` is constructed with **no
`random_state`**.

**Consequence:** you cannot reproduce a prior, and therefore cannot reproduce a posterior, even with
a fixed seed. Note also the dedup guard `if not stable_point in accepted_params` compares float
tuples *after* adding Gaussian noise — it never fires, so it is pure overhead. **Fix:** sort the
accepted points and pass `random_state`.

### 7.9 A per-sample `dt` is silently collapsed to its mean — ✅ FIXED (rejected, not supported)

`SBI/statistics.py` — `self.dt = float(dt.float().mean().item()) if torch.is_tensor(dt) else float(dt)`.

`gen_stats` **explicitly supports** a per-sample `dt` tensor, and every frequency axis, ACF decay time
and Group-G lock-in phase is built from this single scalar. Latent today because all callers pass
`dt_exp` — but wrong for every row the moment anyone uses the feature. **Fix:** support it or reject it loudly.

### 7.10 `_interp_log` extrapolates off the PSD grid without saying so — ✅ FIXED (NaN off-grid; callers exclude and report)

`FDT/sanity.py` — `torch.searchsorted(...).clamp(1, len(log_x_old) - 1)`. A chi frequency outside the
Welch grid's span gets a linear extrapolation in log-ω from the two edge bins, and `eff_temp_ratio`
then divides by it.

**Consequence:** widen `freq_bounds` past the PSD resolution and `T_eff/T` acquires a smooth,
plausible-looking, entirely fabricated tail. **Fix:** return NaN outside the grid.

### 7.11 Smaller items

- **Swallowed exceptions.** `Reduction/sweep.py` (`except ValueError: pass` leaves a ragged row with
  no `error` field); `FDT/cross_validation.py` (`except Exception: print; continue` around a campaign
  — a CUDA OOM is recorded as a string and the loop marches into the next point, which OOMs
  identically, so **a 12-point overnight sweep can "complete" with every row failed**);
  `orchestrator._emit_overlay_figures` wraps **all five** figures in one `try`, so a failure in the
  first silently costs the other four.
- **Accepted-but-ignored parameters.** `pipeline.gen_training_data(n_vars=...)` is threaded from the
  orchestrator through `train_nn` and never referenced in the body (the real count comes from
  `inits.shape[-1]`). `file_manager.parse_bounds_file` returns `collected_units`, which its own
  docstring admits "BOTH callers discard". `config.SimConfig.si_factors` is built from a **set**-derived
  tuple, so it is positionally meaningless, yet it is a required constructor arg threaded through
  `cli.py` and 8 scripts.
- **Dead code.** `helpers.condition_gmm_on_param` allocates on CPU unconditionally (so it would raise
  on a CUDA GMM) **and** has zero callers; likewise `helpers.repeat2d_r`.
- **Segment stitching duplicates a sample.** `simulator.simulate` carries `results[-1]` into the next
  segment while `sdeint` writes `xs[0] = x0`, so the trajectory advances `n_fine - (segs-1)` steps and
  `sol[:, :, boundary]` repeats the previous terminal state. Negligible at `segs ≤ 3`, but `t` and
  `sol` are **not exactly co-indexed** — state this before anyone builds a phase-sensitive feature.
- **Two spellings of the same thing.** `np.random.randint(0, 1, ...)` appears three times and always
  returns zeros (numpy's `high` is exclusive), while `orchestrator._observation_inits` writes the same
  thing honestly as `np.zeros(...)`. The behaviours agree — but this is exactly what gets "fixed" into
  a real bug later.

### 7.12 Cross-cutting: the scripts run blind to their own safety net

Every diagnostic script opens with `warnings.filterwarnings("ignore")`. Meanwhile **every**
out-of-distribution guard in the pipeline is a `warnings.warn` — `generate_observations`' `N_ND_MAX`
warning, `check_observation_in_distribution`'s prior-quantile flags, the recording-length check.
So the scripts deliberately suppress the safety net that was built for them. **Fix:** narrow the
filter to the specific noisy third-party categories.

---

## 8. Performance opportunities

> **Status: §8 is CLOSED.** Everything actionable is done (see the top Appendix A entry).
> **Three of its recommendations were REJECTED after investigation**, each with the reason recorded
> at the code site so it is not re-attempted:
>
> | Rejected | Why |
> |---|---|
> | the headline "25-200x" on `CAL_RUN_SIZE` | Not achievable. `CAL_N_SCALES` is `t_scale`'s effective SBC sample size, so trading pairs for wall-clock damages the parameter chi(omega) exists to separate. Reshaped into independent knobs whose defaults are bit-identical instead. |
> | hoisting the per-segment `Solver()` | ~0.1 s per training round, against a testability seam the suite depends on. Implemented, caught by `test_solver_failure_raises_instead_of_killing_the_process`, reverted. |
> | dropping `_build_spectral`'s PSD clone | `self.psd` is read on its own elsewhere, and the two `psd_nodc` consumers are NOT equivalent to masking at use (the local-maxima scan compares bin 1 against bin 0). Both feed CONDITIONING features; 246 MB is not worth a silent drift in one. |
> | `decorrelate`'s float64 statistics stack | The rationale ("the result is cast to float32 anyway") does not follow for a CENTRAL DIFFERENCE, where intermediate precision is exactly what resists cancellation -- and the output is V, the coordinate the flow trains in. |

### 8.1 Time

**The headline: SBC calibration runs at batch size 10.**
`config.CAL_RUN_SIZE = 10` with `SBC_N_CAL = 2000` gives `cal_n_runs = 200` **sequential full-length
simulations**. But the SDE solver is a **kernel-launch-bound sequential time loop** — measured, a
batch of 256 costs the same wall-clock as a batch of 2048 (~22 s at `n_fine=300k`). So each run pays
the full ~22 s to produce 10 rows. Raising `CAL_RUN_SIZE` toward the training batch size is a
**~25–200× win on `validate_calibration`**, and ×7 again in chi mode (K+1 sims per run). This is by
far the largest single wall-clock win available.

| Site | Issue |
|---|---|
| `Models/nadrowski_model.py` `g()` | Recomputes the **constant** noise amplitudes (`sqrt(2/(n·beta))` etc.) on **every one of ~2.4M steps** — Nadrowski is `state_dep_drift=True`, so `g(x)` is called per step. Pure functions of the parameter tensors. Hoist to `__init__`. |
| `SBI/overlay.py` `cycle_average` | O(n_bins × B × n): 48 full boolean-mask passes over `(1000, N_obs)` — ~2.9 billion comparisons at `N_obs=60000`, for one diagnostic figure. One `argsort` + `bucketize` collapses it to a single pass. |
| `Solvers/sdeint.py` | `torch.zeros` for the per-segment buffer memsets ~2.4 GB that is immediately overwritten (every element except row 0 is written by the loop). `torch.empty` + `xs[0] = x0` is exact and free. Multiply by segs × runs × 5000 batches. |
| `cli.py` `_units_to_factors`, `_parse_cell` | A fresh `pint.UnitRegistry()` per cell parse (~100–300 ms each — it parses the full unit-definition file). Called from every config builder and every script. `SimConfig` already caches one via `@cached_property`; make it a module-level singleton. |
| `Simulator/simulator.py` `__sols` | Constructs a `Solver()` per **time segment** — its `__init__` defines three closures, so ~30k–150k pointless closure constructions per training round. Make them module-level. |
| `SBI/decorrelate.py` | The Fisher rotation runs ~208 full ensemble simulations before training even starts, and each `feats` call re-derives `subs`/`n_fine`/`t_fine`/`n_segs` and rebuilds `base_inits.expand(...).contiguous()` from scratch — identical for the `zp`/`zm` pair. |

### 8.2 Memory

**`del x_nd_fine` frees nothing.** `SBI/pipeline.py`, the forced branch of `gen_training_data`:

```python
x_nd = x_nd_fine[:, ::subsample_factor][:, :N_points_k]   # a strided VIEW
del x_nd_fine                                              # ...so this is a no-op
```

The view pins the whole `(run_size, n_fine - steady_idx)` storage until the rescale ~15 lines later,
while the spontaneous run allocates its own. At `run_size=2048`, `n_fine≈300k`: **~4.3 GB held where
~16 MB is needed.** The **chi branch two blocks up already does it correctly** — rescale first, *then*
`del`. Apply that same shape to the forced and `spontaneous_only` branches, and to the same pattern in
`orchestrator.generate_observations._spont_run` and `decorrelate.py`.

| Site | Issue |
|---|---|
| `SBI/decorrelate.py` | **The one `n_vars`-wide zero-force site the 2026-07-28 fix did not reach** — `forcing.n_force_channels` is imported nowhere in this file. 690 MB where 230 MB is needed, on each of ~216 Fisher calls. |
| `FDT/spectral.py` `psd_welch` | Promotes the **entire ensemble** to float64 up front (~1.6 GB at Campaign-1 defaults) when only the `(M, nperseg)` segment inside the loop needs it (~33 MB). Move the `.to(torch.float64)` onto `seg`. FDT runs on **host RAM** (`cpu_device()`). |
| `FDT/campaigns.py` | Campaign 1 holds the full `(n_vars, 1, M, n_steps)` solution alive through `psd_welch` via a view — ~2.5 GB for Nadrowski, for a diagnostic that only reads channel 0. Same in `sanity.check_ensemble_convergence` / `check_psd_window`. |
| `SBI/statistics.py` `_group_g` | Per harmonic: a `(B,n)` float zeros, a `(B,n)` float, a `(B,n)` complex from `torch.complex`, the `exp` result and the product — five allocations, three complex, to produce three scalars per row. `(x*cos).sum()` / `(x*sin).sum()` is identical at a third the bytes and no complex dtype. |
| `SBI/statistics.py` `_build_spectral` | Clones the whole PSD purely to zero one bin; `psd_nodc` is read in exactly two places, both of which could mask the DC bin at use. |
| `SBI/decorrelate.py` `feats` | Runs the entire statistics stack in **float64**, which makes every FFT **complex128**, across ~216 calls — for a Fisher that is immediately `.numpy()`'d and whose eigenvectors are cast back to float32. |
| `SBI/overlay.py` `_analytic_phase` | ~2.4 GB peak at 1000 draws × 60000 samples (float64 + two complex128 `(B,n)` tensors). Sits inside the `try/except`, so an OOM degrades to a one-line warning and you lose the figures without learning why. |

---

## 9. Code organisation and documentation

> **Status (2026-07-28):** 9.1, 9.2, 9.4, 9.5, 9.6 and the actionable part of 9.7 are DONE.
> **9.3 (file splits) was deliberately NOT done** -- reasoning below. One 9.1 item had already been
> fixed before the sweep: `vt.py` cites `tests/test_gui_progress.py`, not the non-existent
> `tests/test_vt.py`.
>
> **9.3 -- why the splits were left alone.** They are the only §9 item that is pure churn: mechanical
> import rewiring across many call sites, no behaviour change, and the benefit (readability) is
> subjective. `core/config.py` alone is imported by ~50 sites, `orchestrator.py` by most of the
> pipeline. The audit itself frames these as *"Suggested split"*, not defects. Doing them at the tail
> of a long change set -- after every §7/§8/§10 fix had already landed in those same files -- is
> exactly when an import gets mis-rewired and the tests still pass because nothing exercises that
> path. They are worth doing, in their own focused pass, with the test suite as the only moving part.
>
> What DID land from the surrounding items: the four duplicate row widgets are now one
> `widgets/field_row.LabeledFieldRow` (9.4); every panel class has a docstring saying what it drives
> and what it PERSISTS (9.5); every in-repo `file.py:LINE` citation is now a function name (9.5 --
> and all of them were already stale, e.g. `cli.py:331` pointed at a section header); the five
> AST-verified unused imports are gone (9.6); `.gitignore` covers the 6.9 MB root log (9.7).


### 9.1 Four already-wrong facts (cheapest possible fixes)

1. `core/gui/vt.py:4` — "see `tests/test_vt.py`". **That file does not exist.**
2. `core/gui/app.py:67` — the user-facing startup-failure dialog is titled **"GFDT could not start"**.
   The app is PRISM everywhere else. This is visible to users.
3. `core/gui/panels/inference_tabs.py` — the section banners read `── 1. Config`, `── 2. Prior`,
   `── 4. Posterior`, `── 5. Validate`, `── 6. Infer`. **There is no 3**, and the numbers don't match
   the five tabs. `_StagePanel`'s docstring says "the **six** inference tabs" while the module
   docstring says five.
4. `core/Simulator/user_simulator.py` and `core/gui/panels/simulate_runner.py` still carried stale
   `exit()`/`SystemExit` prose after the 2026-07-28 fix — corrected in that pass; **check for more.**

### 9.2 A latent crash: a module shadowed by a local

`core/gui/panels/inference_tabs.py` imports `from core.Helpers import ... labels ...`, then
`_build_config` rebinds it: `labels = VALID_LABELS[VALID_MODELS.index(model)]`. This works only
because that one function never touches the module — but the same file calls `labels.axis_label(...)`
and `labels.gui_forcing_label(...)` elsewhere. **Any future line added to `_build_config` that uses
the module raises `AttributeError` on a `list`.** Rename the local to `model_labels`.

### 9.3 Files that are too large

| File | Lines | Contents | Suggested split |
|---|---|---|---|
| `tests/test_gui_progress.py` | 1872 | 72 tests over **eight** unrelated subjects | `test_vt_progress.py` (restoring the name `vt.py:4` already claims exists), `test_worker_dispatch.py`, `test_settings_persistence.py`, `test_figures.py`, `test_nav_and_gating.py`, `test_simulate.py` |
| `core/orchestrator.py` | 1270 | five stage fns + PPC + overlays + experimental paths | by stage |
| `core/SBI/pipeline.py` | 904 | simulation, stats, training data, training | `simulate.py` / `training_data.py` / `train.py` |
| `core/gui/panels/inference_tabs.py` | 888 | five screens + a 68-line HELP dict + four worker runners | `inference/help_text.py`, `inference/runners.py`, `inference/base.py`, then one file per tab |
| `core/config.py` | 738 | tuning constants **and** path constants **and** the `SimConfig` dataclass with real behaviour | `config/constants.py` + `config/sim_config.py` |

### 9.4 Duplication worth collapsing

- **Four near-identical "labelled fields in a row" widgets:** `_ChiRangeRow` (inference_tabs),
  `_BoundsRow` (param_grid — whose own docstring admits it is "the crossval `_GridRow` pattern"),
  `_GridRow` (crossval_panel), `_ParamRow` (model_builder_screen). Collapse to one
  `widgets/field_row.py::LabeledFieldRow(pairs)` — which is also the single place to add the minimum
  widths from §10.
- **Cell-picker repointing** is byte-identical in `fdt_panel` and `simulate_panel`, and the same logic
  again in `inference_tabs._CellPreviewMixin` — whose docstring claims it is shared by "Simulate +
  Infer", though `SimulatePanel` does not use it.
- **The restore-order rule (Q1)** is re-explained in three multi-line comments. One helper, rationale
  stated once.

### 9.5 Documentation quality

- **Coverage is inverted.** Every panel class is undocumented — `BasePanel`, `MainWindow`, all five
  inference panels, `FdtPanel`, `CrossValPanel`, `ReductionPanel` — as are all 18
  `save_settings`/`restore_settings`/`_build_controls`/`_run` overrides. Meanwhile private helpers
  carry 10–20 line essays. **Highest-value fix:** one docstring per panel class saying what it drives
  and what it persists.
- **No consistent style and zero parameter docs** in `core/gui/`: no `:param:`, no Google `Args:`, no
  type hints on any panel method. The de-facto house style is "summary line + free-form rationale
  paragraphs" — state that and normalise to it.
- **~15–20% of `core/gui/` is rationale prose that belongs here**, not in the source. It is genuinely
  valuable "why" — it just makes the "what" unreadable. Move it to this document and leave a one-line
  pointer.
- **Comments cite line numbers in other actively-edited files** (`base_panel.py` → `cli.py:331`,
  `crossval_panel.py` → `cli.py:523-526`, `progress_pane.py` → `pipeline.py:517`). `cli.py` is 694
  lines and changes often. **Cite function names.**
- **A comment that contradicts its own code:** `widgets/progress_pane.py` says "Fixed width: … an
  unpinned label would re-lay-out the pane on every 100ms tick", then calls `setMinimumWidth(150)`,
  which does **not** pin a width — with `QSizePolicy.Fixed` the widget takes `sizeHint()`, which grows
  when `_tick()` swaps in the stall message. The pane *does* re-lay-out. Fix: `setFixedWidth(150)`.

### 9.6 Naming and boundaries

- `core/gui/vt.py` is an opaque name for the tqdm/VT100 terminal-protocol parser, sitting beside
  descriptive siblings (`plot_watcher.py`, `progress_pane.py`).
- Mixed directory casing inside `core/`: `Helpers/ Models/ SBI/ FDT/ Reduction/ Solvers/ Simulator/`
  vs `gui/ config.py cli.py orchestrator.py`.
- **Private names imported across module boundaries:** `cli._SWEEP_PRESETS`, `cli._parse_cell`,
  `cli._units_to_factors`, `cli._INFERENCE_PROMPT_UNITS`, `model_store._nd_bounds`. Either promote
  them to public API or stop reaching through the underscore — right now it conveys no information.
- **Unused imports** (AST-verified): `gui/session.py` (`field`), `orchestrator.py`
  (`_transform_device`), `SBI/pipeline.py` (`OrderedDict`), `cli.py` (`warnings`),
  `SBI/Priors/hopf_prior.py` (`helpers`), plus vestigial `from __future__ import annotations` in
  several files.

### 9.7 Repo hygiene

- **`sbc_run.log` — 6.9 MB, tracked at the repo root.** `.gitignore` is three lines
  (`/sbi-logs/`, `/archive/`, `/.claude/`).
- **`Resources/` tracks generated outputs** — 7 `.h5` sweeps, `.pt` priors/posteriors, `.png` plots,
  `.parquet` — **interleaved with the hand-written `Bounds/`/`Cells/`/`Units/` files a user is
  supposed to edit.** Inputs and outputs are indistinguishable in the same tree.
- **`.claude/worktrees/*/` hold complete second and third copies of `core/`** on disk. Ignored by git,
  but **not by grep** — every IDE "find in files" returns three hits for everything.

---

## 10. GUI layout and sizing

The user's report: *"user inputted boxes being cut off"* and *"panes in tabs having to be dragged to
fully show everything."* **Three compounding defects explain nearly all of it**, and together they are
roughly 40 lines of code to fix.

> **STATUS: L1–L18 are all FIXED** (2026-07-28, see the top Appendix A entry). The diagnosis below is
> kept because it explains *why* each change is shaped the way it is. What landed, in brief:
>
> | | |
> |---|---|
> | **L1** | `self.splitter` / `self.controls_scroll` promoted from locals; `setChildrenCollapsible(False)`; default sizes; persisted via a `save_layout`/`restore_layout` pair (**not** `save_settings`, which 8 of 9 panels override without `super()`), plus a debounced save on `splitterMoved` so a crash no longer loses it |
> | **L2** | the 460px maximum is gone; `CONTROLS_MIN_W = 360` is the floor |
> | **L3** | new `widgets/forms.make_form()` — `AllNonFixedFieldsGrow` + `WrapLongRows` — used at all 18 sites, so the 19th gets it free |
> | **L4/L5** | `FIELD_MIN_W`/`PATH_FIELD_MIN_W` on the field classes; `help_label`'s inner QLabel wraps and is capped at `LABEL_MAX_W` |
> | **L6** | `config.CHI_K_MAX = 24` bounds K (cost is linear in K *and* the Infer tab grows a picker row per frequency) |
> | **L7** | new `widgets/adaptive_stack.AdaptiveStack` — hidden pages go `Ignored`, so a stack sizes to the page you are looking at; used by `infer_stack`, the builder's `force_stack` and `SourceToggle` |
> | **L8** | `_PassThroughScrollArea` hands the wheel back to the outer area at its limit |
> | **L9** | `setMinimumSize(900, 600)`; initial size clamped to the screen; `restoreGeometry`'s return value checked **and** the result validated against attached screens, re-centring if off-screen |
> | **L10** | the results column is a vertical `QSplitter`; `BasePanel.insert_result_widget()` is the seam `SimulatePanel` uses for its live view |
> | **L11/L13** | log pane wraps; the crossval cell-values label (unbounded `str(e)`) wraps |
> | **L12** | `_FitLabel` scales embedded figures down to the available width (never up past 1:1) |
> | **L14** | per-item tooltips on `ArtifactPicker`, so entries added by a later `refresh()` are readable |
> | **L15** | the Settings help blob is a `QTextBrowser`, which lays out and scrolls its own rich text — a word-wrapped QLabel could not propagate `heightForWidth` there, which is why the bottom was unreachable |
> | **L16** | `design.ui_scale()` / `scaled()` derive from the real application font; control heights, the badge and the whole type ramp now track the OS "make text bigger" setting |
> | **L17** | the inference `QTabWidget` sizes to the current tab |
> | **L18** | the model builder's Validate/Save are a sticky action bar OUTSIDE the scroll area |

### 10.1 The three root causes

**L1. The only splitter in the app never remembers its position, and can be dragged to zero.**
`panels/base_panel.py` creates the **one** `QSplitter` in the entire GUI — and `BasePanel` is
instantiated **nine** times (Reduction, FDT, CrossVal, Simulate + the five inference tabs), so there
are nine independent splitters. Repo-wide there are **zero** calls to `setSizes`,
`setChildrenCollapsible` or `setCollapsible`, and `BasePanel` has no `save_settings` override, so
splitter positions are **never persisted**.
*What the user sees:* every launch, in every one of nine tabs, the form column starts at or near its
340 px minimum and must be re-dragged. One slip past the left edge collapses the controls column to
zero width, recoverable only by finding a 5 px handle at x=0.

**L2. The controls column is hard-capped at 460 px.** `base_panel.py` sets
`setMinimumWidth(340)` / `setMaximumWidth(460)` with `setWidgetResizable(True)`. Because the scroll
area refuses to shrink its inner widget below `minimumSizeHint().width()`, **any form wider than 460 px
gets a permanent horizontal scrollbar in the left column — and widening the window does nothing**,
because the cap is absolute.

**L3. No form-growth policy and no minimum widths, anywhere.** Repo-wide there are **zero**
occurrences of `setFieldGrowthPolicy` or `setRowWrapPolicy`, and no input widget declares a minimum
width (`FloatField`/`IntField`/`PathField` set a validator and nothing else; the stylesheet sets
`min-height` only). Qt's Windows default is `FieldsStayAtSizeHint`: the label column takes the widest
label and each field gets its size hint and stops.
*What the user sees:* numeric boxes 3–6 characters wide. Typing `0.033333` scrolls inside the box and
the value cannot be read back without clicking into it.

The two-line fix for every form is
`setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)` + `setRowWrapPolicy(QFormLayout.WrapLongRows)`.

### 10.2 The rest, in priority order

| ID | Site | What the user sees |
|---|---|---|
| L4 | Composite rows: `_GridRow` (crossval), `_ChiRangeRow` (inference), `_BoundsRow` (param_grid), `_ParamRow` (model builder) | 2–3 fields split the remainder of an already-capped column N ways. `_ParamRow` packs 3 fields + 4 labels + a checkbox on one row. |
| L5 | `widgets/help_badge.py` `help_label`/`add_help_row` (~40 uses) | The badge holder is the *label widget*, so `QFormLayout` sizes the label column to the widest one — and the inner `QLabel` never sets `setWordWrap(True)`. **One long label narrows every field in that form** (e.g. "F0 (ND forcing amplitude)", "χ frequency range"). |
| L6 | `inference_tabs._rebuild_chi_fields` | The χ Infer page grows one file-picker row **per probe frequency**, and Config validates only `K >= 2` — **no upper bound**. Label + badge + `Browse…` leave ~120 px of line edit for an absolute path, with no tooltip and no elision. |
| L7 | Three `QStackedWidget`s (`infer_stack`, the builder's `force_stack`, `source_toggle._stack`) | Each reserves the **tallest** page's height on every page — a large dead gap on the short pages (e.g. the builder's empty "None" forcing page reserves the "exponential" page's height, once per state variable). |
| L8 | `widgets/param_grid.py` (`_MAX_VISIBLE_HEIGHT = 320`, inside `SourceToggle` inside the outer scroll area) | **Nested scroll areas trap the mouse wheel.** Scrolling over a bounds grid scrolls the grid and stops — you cannot reach the "Build / Load prior" button by scrolling from there. Also a hardcoded 320 px, so a 3-parameter and a 13-parameter grid get the same box. |
| L9 | `main_window.py` — `resize(1300, 820)`, `restoreGeometry` unchecked | 820 px exceeds the usable height of a 1366×768 laptop and of 1920×1080 at 150% scaling, so the action buttons start below the fold. Geometry saved on an external monitor restores **off-screen**; `setMinimumSize` is never called. |
| L10 | `base_panel.py` — `addWidget(figure_stack, 3)` / `addWidget(log_pane, 1)` | The log pane permanently occupies ~25% of the results area even when empty, **with no handle to drag**. This is the other half of the complaint: here there is nothing to drag at all. |
| L11 | `widgets/log_pane.py` — `setLineWrapMode(NoWrap)` | Panels write long single-line diagnostics; the user sees the first ~60 characters and must scroll horizontally to read the part that says what to do. |
| L12 | `widgets/figure_stack.py` | Embedded figures are shown at full render resolution (`dpi=110, bbox_inches="tight"` → routinely 1200–1800 px) in a scroll area with no scaling, so they are *always* scrolled in both axes. The pop-out window has Fit / 100% controls; the embedded tab has none. |
| L13 | Only **three** `setWordWrap(True)` calls exist in the whole GUI | `crossval_panel` sets an **unbounded exception string** into a non-wrapping label in a form field — which forces the form past 460 px and triggers L2's permanent scrollbar on a panel that otherwise fits. |
| L14 | `widgets/artifact_picker.py` (7 uses) | No minimum width, no `setSizeAdjustPolicy`, no tooltip. Qt's `AdjustToContentsOnFirstShow` makes the combo as wide as the widest entry *present at first show*; entries added later by `refresh()` are silently elided with no tooltip. |
| L15 | `screens/settings_screen.py` `_build_help` | Concatenates every panel docstring **and** every `HELP` dict (~4,500 chars from `inference_tabs` alone) into **one word-wrapped `QLabel`** inside a resizable `QScrollArea`. A word-wrapped label does not propagate `heightForWidth` correctly there, so **the bottom of the help text is unreachable**. |
| L16 | `help_badge.py` `setFixedSize(16,16)`; `design.py` `CTL_H=32` + a literal `font-size: NNpx` ramp | True fixed sizes: at a larger OS text size the "?" glyph clips, and the 32 px control height directly contributes to forms overflowing the viewport. These ignore the OS "Make text bigger" setting. |
| L17 | `screens/inference_screen.py` | The five tabs differ wildly in content height (Validate is one label + one button; Infer in χ mode is 10–25 rows). `QTabWidget` sizes to the largest page, so switching tabs visibly reflows the whole results column. |
| L18 | `screens/model_builder_screen.py` | There *is* a page-level scroll area (the only correctly-scrolled tall form in the app) — but the Validate / Save buttons are the **last** thing in the scrolled column, so with ~15 parameters they sit a full screen below the fold. **The user reports the Save button as missing.** No sticky action bar. |

### 10.3 This entire defect class is untested — ✅ NO LONGER TRUE

~~**Not one of the 72 GUI tests asserts anything about geometry, minimum sizes, splitter sizes or
scroll ranges.**~~ Four now do:

- `test_panel_splitter_is_sized_and_not_collapsible` — non-zero sizes, not collapsible, no width cap
- `test_panel_layout_round_trips_through_settings` — the split survives a restart
- `test_forms_grow_their_fields_and_numeric_boxes_have_a_floor` — walks **discovered** `QFormLayout`s
  and `QLineEdit`s across three panels, so a newly added form is covered without editing the test
- `test_long_diagnostics_are_readable_without_horizontal_scrolling` — log wrap + the crossval label

None is vacuous: before the fixes, `childrenCollapsible()` defaulted True, `maximumWidth()` was 460,
the growth policy was Qt's `FieldsStayAtSizeHint`, field minimums were 0, and both wrap flags were off.

**Testability constraint** (bit me, will bite you): tests run offscreen and never `show()`, so
`widget.width()` returns a 640×480 placeholder. Assert on `sizeHint()` / `minimumSizeHint()` /
`maximumWidth()`, or `show()` + `resize()` + `_pump()` first.

### 10.4 Suggested order

1. **L1** (splitter sizes + persistence + non-collapsible) → **L3** (form growth policy) → **L4**
   (minimum widths on the three field classes) → **L2** (raise or remove the 460 px cap) → **L11**
   (log word wrap) → **L13** (word wrap on the error/status labels). *This is the ~40 lines that
   answers the complaint.*
2. **L9** (screen-bounds validation) → **L10** (a vertical splitter in the results column) → **L7**
   (stacked-widget heights) → **L8** (nested scroll) → **L12** (fit-to-window) → **L18** (sticky
   action bar).

---

# Appendix A — Change history

Newest first. Nothing here is required to work on the code; it records **why** decisions were made,
and — importantly — **which dead ends were already tried**, so they are not retried.

## 2026-07-28 (later) — §7/§8/§10 remediation sweep; scripts made chi-aware

A pass over the four open tracks in §7, §8, §10 and §4.1. **Tests 124 → 135** (+7 SBI, +4 GUI).
Everything below is
local (uncommitted). Three catalogued items did **not** survive verification and are corrected here
rather than silently dropped.

### Corrections to this document's own bug list

- **§7.1 does not reproduce on the installed PyTorch (2.9.0), and its severity was overstated.**
  The claim was that a physical value on a box bound inverts to `±inf` and NaNs the run.
  Measured: `torch.distributions.SigmoidTransform._inverse` **clamps its argument to
  `[tiny, 1-eps]` internally**, and `sigmoid` saturates at `0.9999998807907104`, never exactly `1.0`.
  A theta on — or outside — a bound inverts to a finite `±15.94 / −87.34`. Verified across the
  linear box, the log box, out-of-box values (the Sobol write-back case) and the rotated
  composition: **no path yields a non-finite latent.**
  The *code* defect is real and narrower: `train_nn` filtered `thetas` by `data` and never checked
  `thetas`. That check now exists in `train_nn` and `gen_cal_data` and warns with the offending
  columns. A clamp in the training hot path was deliberately **not** added — it would buy nothing on
  this torch and would perturb the parameters the simulator actually runs. The invariant is a
  property of *torch*, not of this repo, so it is pinned by
  `test_box_roundtrip_never_yields_a_nonfinite_latent_target`.
- **§8.1's "~25–200× win" on `validate_calibration` is not achievable.** `CAL_RUN_SIZE` is samples
  per `(t_scale, T)` pair and `gen_training_data` draws exactly one pair per batch, so raising it at
  fixed `n_cal` collapses the calibration set toward a single scale. Worse, **every row in a batch is
  *assigned* that batch's `t_scale`**, so their SBC ranks are not independent — `t_scale`'s effective
  sample size is the PAIR COUNT, not `n_cal`. Since chi(ω) exists to separate `λ`/`t_scale`, trading
  pairs for wall-clock damages the exact measurement the mode is for.
- **§8.1's per-segment `Solver()` hoist is a false economy — implemented, then reverted.** The call
  count is real (~30k–150k per training round) but each is three closure allocations against a
  segment that takes ~10 s: ~0.1 s per round. And resolving `sdeint.Solver` at *call* time is a
  deliberate seam — `test_solver_failure_raises_instead_of_killing_the_process` patches the class.
  A module-level singleton is built at import, so the patch silently stopped working and the suite
  caught it. The reasoning is now recorded at the call site.

### Fixed

| Item | What changed |
|---|---|
| **7.1** | Targets-side finiteness check in `train_nn` + `gen_cal_data`, warning with the offending columns. See the correction above for what was *not* done. |
| **7.2** | `decorrelate.feats` wrapped in `torch.random.fork_rng()` (CUDA generator included). The fixed seeds STAY — they are common random numbers keeping the central difference from being swamped; deleting them would break the rotation. |
| **7.3 + 7.4** | One change. `cfg.n_obs` now carries the resolved post-clip observation length and the PPC path reads it instead of re-deriving `int(T_obs/dt_exp)`. No formula's rounding changed, so training geometry — and comparability with the keeper — is untouched. `_emit_overlay_figures`' silent `return` now warns with the actual shapes, and its single `try` became four independent groups. |
| **7.5** | One `cli._prompt_index(n, prompt, allow_zero=)` behind all five prompts. `0`, negatives, out-of-range and non-numeric all re-ask instead of silently selecting the last item or raising an uncaught `IndexError`. The dead `if model not in VALID_MODELS` check (run *after* reading out of that list) is gone. |
| **7.6** | `_group_b` walks outward from the argmax instead of taking a global span. **This changes a conditioning feature — see the warning below.** |
| **7.8** | `sorted()` instead of `list()` on the accepted-point set in **all three** built-in priors, plus `random_state=GMM_RANDOM_STATE` on the GMM fit. Both are needed. The never-firing dedup guard was KEPT: it also guards `queue.append`/`num_added`, so removing it is a behaviour change for no gain. |
| **8.2** | Forced and spontaneous `gen_training_data` branches rescale straight off the strided view, so `del` actually releases (the chi branch was already correct). `decorrelate`'s zero-force tensor now uses `forcing.n_force_channels` — the one site the earlier sweep missed. |
| **8.1** | `CAL_N_SCALES` / `CAL_RUN_SIZE` (floor) / `CAL_RUN_SIZE_MAX` decoupled. Nadrowski's constant noise amplitudes hoisted to `__init__` in **both** the eager `g()` and the JIT step. `torch.empty` for the per-segment buffer. Process-wide `pint` registry (`config.unit_registry()`). `overlay.cycle_average`: 48 masked passes → one stable sort, verified **bit-identical**. |
| **7.7** | `implicit_euler` DELETED, along with `Simulator.__sols`' `explicit=` parameter and its branch. Zero call sites, three independent bugs (stored the pre-convergence iterate, called `g()` with no args so it could not run a `state_dep_drift` model at all, built its time grid on CPU). |
| **7.9** | `statistics._resolve_dt` REJECTS a non-uniform per-sample `dt` instead of averaging it. A uniform `(B,)` tensor is still accepted — `gen_stats` legitimately sub-batches one. Supporting a real per-sample dt would mean per-sample frequency grids, which breaks the batched-FFT design for a caller that does not exist. |
| **7.10** | `_interp_log` returns NaN outside the Welch grid instead of extrapolating off the edge bins. **Both callers now exclude NaNs and report how many** — `check_high_freq_fdt` probes the top of the grid, so it could previously pass or fail on fabricated numbers; `check_passive_baseline` raises if nothing is covered. |
| **7.11** | `Reduction/sweep.py` records `fixed_point_error` instead of `pass`. `FDT/cross_validation` COUNTS failures and raises if a whole sweep produced nothing (a repeating CUDA OOM used to let a 12-point overnight run "complete" empty). `gen_training_data`'s ignored `n_vars` is now a cross-check against the model's declared inits. Dead `helpers.condition_gmm_on_param` / `repeat2d_r` deleted. The five always-zero `np.random.randint(0, 1, ...)` are `np.zeros`. The segment-seam sample duplication is documented at the site. `si_factors` fell from ~10 construction sites to 2 for free, via the script migration. |
| **8.2 (rest)** | `psd_welch` promotes the (M, nperseg) SLICE instead of the whole ensemble (~1.6 GB → ~33 MB resident; **verified bit-identical**). `run_campaign1_psd` materialises channel 0 and drops the full `(n_vars, 1, M, n_steps)` solution instead of pinning it through `psd_welch` via a view (~2.5 GB). `_group_g`'s lock-in is two REAL reductions per harmonic instead of a complex exponential — no `(B,n)` complex tensors at all. `overlay._analytic_phase` is float32 + `rfft` rather than float64 + full complex128 FFT (~2.4 GB peak → a fraction of it). |
| **§9** | 9.1 the four wrong facts ("GFDT could not start" → PRISM; banners renumbered 1-5 from a set that had no 3 and ran to 6; "six inference tabs" → five). 9.2 the shadowed `labels` module — renaming the local IMMEDIATELY exposed that its consumer still read `labels=labels`, so the latent crash was one edit away exactly as described. 9.4 the four duplicate row widgets → `widgets/field_row.LabeledFieldRow`. 9.5 `setFixedWidth` (the comment always claimed a pinned width), a docstring on every panel class stating what it drives and what it PERSISTS, and all 12 in-repo `file.py:LINE` citations → function names (**every one was already stale**). 9.6 five AST-verified unused imports. 9.7 `.gitignore`. **9.3 (file splits) deliberately NOT done** — see the §9 banner. |
| **§10** | See the L-table in §10 — tier 1 and tier 2 both landed. |

### chi(ω): the artifact-safety gate that was missing

`save_posterior_artifacts` only wrote the `.rot.pt` sidecar when `V is not None or log_params`. chi
is deliberately unrotated and `REPARAM_LOG_PARAMS` is `[]`, so **a chi posterior wrote no sidecar at
all** and landed in `Resources/Posteriors/` byte-indistinguishable from the legacy forced posteriors
beside it, with nothing on the load path checking width or mode.

The sidecar is now written **unconditionally** and carries `mode`, `input_dim`, `forcing_dim`,
`chi_n_freqs`, `chi_f0`, `chi_freq_bounds` and `param_keys`. `reparam.posterior_mode()` decodes a
posterior's mode in three tiers (sidecar → the trained net's `forcing_dim` → arithmetic on
`condition_shape`, which warns), and `orchestrator._assert_mode_matches()` fails **loudly and before
any simulation**. A cross-mode load used to surface as a raw `Linear` shape error from inside
`EmbeddedNet` after the entire calibration set had been simulated.

### `scripts/` — all 11 migrated

New `scripts/_common.py` routes every script through `cli.make_sim_config` + `load_and_validate_gt`
(the pair `simulate_runner.build_stream_config` already used). Nine scripts had hand-rolled the same
`SimConfig(...)` literal, hard-coding `model="NADROWSKI"` and setting **no chi fields** — and because
`SimConfig.chi_mode` is a plain `= False` default rather than a `default_factory`, such a config was
*permanently* non-chi however the module constant was set. Verified: the new builder reproduces the
old path's parameter ORDER, values, inits and mode **identically** across all four Nadrowski cells.

- `sbc_characterize` threads the mode into `gen_cal_data` and verifies the posterior first.
- `retrain_convergence` derives `forcing_dim` from the shared width rule, passes the chi keys, and
  saves through `save_posterior_artifacts` so its output carries a sidecar.
- `degeneracy_map` builds its Jacobian from the **mode's own** feature vector (30 spontaneous + 3K
  chi in chi mode; Group G dropped because chi zeroes it), suffixes outputs by mode, and prints a
  per-parameter top-features table. **It also seeds a second time immediately before
  `gen_chi_block`** — those K simulations were otherwise unseeded, so the ±δ arms of the central
  difference saw different chi noise and the derivative was swamped.
- `diagnose_fmax`, `identifiability_offgt`, `feature_candidate_test` **refuse loudly** under `CHI=1`
  (`_common.assert_not_chi`): their metrics are defined over the 41-feature single-frequency set.
- The blanket `warnings.filterwarnings("ignore")` is replaced by `_common.enable_warnings()`.
  **Expect new output** — the OOD guards these scripts were built for have been invisible for months.

> **On the two §8 rewrites that are NOT bit-identical.** `_group_g`'s lock-in and
> `overlay._analytic_phase` both touch numbers that feed features, so they were measured rather than
> assumed. `_group_g`: max RELATIVE difference 3–5× float32 epsilon, and against a float64 reference
> the old and new forms are equally far from the truth (3.8e-9 vs 6.8e-9) — this is summation-order
> rounding, not a semantic change, and it is far below the run-to-run spread of the simulation that
> produced the input. `_analytic_phase`: max circular phase difference 3e-7 rad against
> `cycle_average`'s 0.13 rad bin width, median exactly 0. **Neither is comparable to the 7.6 change
> below**, which moved a derived quantity by a median factor of 5.
>
> **⚠ `B1_log_Q` CHANGED (7.6).** It is one of the 41 conditioning features, so a posterior trained
> BEFORE this fix must **not** be re-evaluated with post-fix statistics — including
> `posterior_07012026`. Measured on simulated GT traces: rows affected 20/48 (`cell_1`), 2/48
> (`cell_2`), 14/48 (`cell_3`); median width ratio 5.0× / 3.0× / 2.0×, max 18×, i.e. log-Q shifts up
> to 2.89. The keeper's §4.2 numbers stand as historical record only. `cell_2` — the cell §4.1
> nominates for the chi run — is the least affected, and the chi posterior will be trained fresh, so
> it picks up the corrected feature from the start.

### Two stale things found, not fixed

- `POST` defaults to `posterior_3d.pt` in three scripts; that file does not exist in this repo.
- `diagnose_fscale`'s verdict recommends `REPARAM_LOG_PARAMS=['f_scale']`, which Appendix A below
  records as **tried → failed → reverted**. The script has never been told.

## 2026-07-28 — Chi-mode OOM, `SimulationError`, and two chi correctness bugs

**A CUDA OOM when retraining with chi-mode on.** Diagnosed as a tail event: the Sobol filter accepts a
few percent of batches far larger than the median, and those exhaust the card. Peak at the worst
geometry went **20.51 GB → 9.61 GB** on a 15.92 GB card. Fixes, by measured contribution:

- The driveless runs sized their zero-force tensor `n_vars` wide (7.4 GB at batch 2048) when
  Nadrowski's drift reads channel 0 only. Now `forcing.n_force_channels` — §3.2. *This was the
  chi-vs-forced asymmetry: the forced branch always built a 1-channel drive.*
- **The cuFFT plan cache leaked outside PyTorch's allocator** (~2 MB/plan, ~7 new plan signatures per
  batch, zero cross-batch reuse, saturating around batch ~585 of 5000). `empty_cache()` cannot touch
  it — which is why the failure arrived as a **raw driver `cudaErrorMemoryAllocation`** rather than
  `torch.cuda.OutOfMemoryError`. Now cleared per batch. *Preferred to shrinking
  `cufft_plan_cache.max_size`, which would thrash the intra-batch reuse and add ~38 min to a run.*
- `simulator.simulate` carried `results[-1]` as a **view**, pinning the previous 2.5 GB solver segment
  through the next one. A `.clone()` of a 24 KB row frees it.
- `chi.lock_in_batched` promoted the whole batch to float64/complex128 (~64 B/element, measured);
  now a time-chunked float64 accumulation of separate re/im sums (~0 B/element amortised).
- `chi.peak_freq` was the only FFT still running at the full batch; now sub-batched like `gen_stats`.
  (Also dodges cuFFT's Bluestein fallback, which costs 2.4× memory and 28× time on the ~half of
  batches whose length has large prime factors.)
- `gen_obs` cloned all `n_vars` channels when every caller reads `[0, :, :]`; added an opt-in
  `var_idx` rather than narrowing the documented contract.
- A free-VRAM-derived batch guard (`config.memory_budget_elements`, promoted from the FDT campaigns).
  It engages only when a batch genuinely will not fit, because **splitting is expensive**: the solver
  is kernel-launch-bound, so *k* chunks cost *k*× wall-clock.

**`Simulator`'s `print + exit()` → `SimulationError`.** All four sites (the solver call plus the three
built-in `_set_up_model`s) now raise a `RuntimeError` subclass chained `from` the original. The old
form hard-killed the interpreter: a CUDA OOM arrived with **no traceback**, and every caller — the
test runners, the GUI's worker thread, the scripts — died mid-run with nothing reported. Bare `exit()`
also returns **exit code 0**, so a crashed simulation reported *success* to the shell. The GUI's
`except SystemExit` workaround was retired. `WorkerCancelled` still sails through (it is a
`BaseException`) — verified explicitly, since widening that handler would turn every cancel into a
spurious failure.

**Two chi correctness bugs** — neither affected a Nadrowski run, and both were silent:
1. **chi mode was broken for Hopf and user models.** `gen_chi_block` builds its probe under a `fidx`
   declaring no `"amp_y"`, so the builder emitted ONE channel while `HopfModel` indexes
   `force_step[:, 1]` unconditionally. Now widened via `n_force_channels`, probing channel 0 and
   zero-filling the rest.
2. **A clipped fine grid made the conditioning row self-inconsistent.** `t_fine = t[:n_fine_total]`
   clips silently; the spontaneous trace was built by *slicing* (short) while `gen_chi_block`
   *gathers* to `N_points` with a clamp that replicates the last sample — so the summary statistics
   and the chi lock-in described different trace lengths, and `log(T_k)` recorded a duration neither
   had. The Sobol filter now bounds `min(N_ND_MAX, len(t))`. Built-in bounds never tripped this
   (`t_scale ∈ (1,40)` puts `len(t)` at ~2.4M against a 300k cap); model-builder bounds did.

**Documentation consolidated.** The three former handoffs became this file. Correcting them turned up
a fourth test suite — `tests/test_fdt_user.py` — documented in **none** of them, so every stated test
total was wrong. Running it immediately caught a regression from the same day's work:
`campaigns._n_force_channels` had been made to delegate to the shared rule, but it resolved
`cfg.inits_tensor` **eagerly**, where the original only touched it inside the user-model branch — so
built-in callers that never build one started failing. Now resolved lazily. The lesson is in the
table in §1.3: **count the suites, and run all of them.**

## 2026-07-27 — Units, drive, observation modes, and the five-tab restructure

**Two units/drive bugs that invalidate the 07/26-27 forced retrain** (it came out well-calibrated but
*uninformative* — flat SBC, clean PPC, yet marginals ≈ prior for most ND params):

1. **Frequency was off by 10³.** `freq` is cycles per cell-time-unit by construction, but
   `units.txt` declared "Hz", so a 30 Hz drive was simulated as 30 kHz. Fixed structurally
   (`freq_si_to_cell` + `check_unit_consistency`), not by editing a token. **Consequence for the
   science:** measured per-cell spontaneous resonance is 7.6–23.2 Hz, so **every** training drive sat
   43×–10⁴× *above* resonance — the bundle cannot follow it, so Group G carried almost no information
   even when the drive was non-zero. Bounds were re-scoped per cell to bracket that cell's own Ω₀.
2. **The observation was out-of-distribution in the drive.** `cell_2` had `amp=freq=phase=offset=0`,
   but training samples `freq ~ log-uniform`, where 0 has zero probability — so the conditioning block
   was a point training could not produce and the flow could only revert to the prior. Fixed by giving
   `cell_2` a measured weak in-distribution drive, narrowing `amp` 500 → 20 pN, and adding
   `check_observation_in_distribution` (sampling-based, skips circular `phase`).

Also: `CHI_F0` raised 0.05 → 0.2 by measurement (§4.3); the three observation modes made first-class
via `SimConfig.observation_mode`; `REPARAM_ROTATE` moved onto the config (the old
`from .config import REPARAM_ROTATE` snapshotted at import); and the Parameter Inference section
restructured from six tabs to **five**, with Config recording a `ConfigDraft` and Prior owning the
bounds picker (§5/M1b).

## 2026-07-23 — chi(ω) mode implemented (v1)

Machinery + tests; not a trained posterior. Drive protocol chosen with the user: single-tone × K
recordings. See §4.3.

## 2026-07-17/18 — User-defined models

v1 Simulate-only, then v2 SBI-ready (no-forcing only) with state-dependent noise, then S-2 (a
TorchScript `compiled_step` fast path — **2.01× on CUDA**, 6.5k → 13.1k it/s, matching Nadrowski).
Three adversarial-review passes found **22 real defects** total; the model-builder pass alone found
12 (phantom-parameter parsing, silent value mis-binding, unstreamable-on-save configs, a picker-reset
regression) — all fixed and pinned, and distilled into traps **U1–U6**.

Also: the MAPIS → PRISM rename; the Fluent design-token + QSS layer (bespoke, not a third-party widget
library — `qfluentwidgets` was rejected as GPLv3 and would have forced a NavShell rewrite); OS accent
colour; Inter-everywhere font toggle; snapshot-based nav transitions.

## 2026-07-16 — UI/UX requests UX1–UX5

Frame-steps tooltip; progress-pane it/s sparkline; screen/tab transitions; settings gear +
Light/Dark/Auto/Follow-system theming + a Settings/Help screen; cross-platform torch pins + hardened
`run.sh`.

**`[TRIED → FAILED → REVERTED]` — log-scaling `f_scale`.** `REPARAM_LOG_PARAMS=['f_scale']` was
trained and came out **worse** (bad TARP/expected-coverage and a worse `f_scale` SBC rank). That
posterior was deleted and the config reverted to `[]`. **Log-scaling is ruled out**; the mild tilt is
an accepted caveat, not a correctness blocker.

## 2026-07-12/13 — The keeper, and the cell-file refactor

`posterior_07012026` (run-5, 13-dim LINEAR box + multi-point averaged rotation) locked in as THE
KEEPER after a K=10×2000 repeat study and an adversarial for/against panel — **borderline-MEETS,
conservatively** (§4.2).

Cell files were stripped to VALUES + INITIAL CONDITIONS only, and `cli._parse_cell` migrated to a
decoupled path (values from the cell, bounds/order from `Bounds/`, units from `Units/`). *The original
"no code change needed" note was WRONG* — the parser had sourced bounds and units *from* the cell, so
a stripped cell silently dropped parameters.

## 2026-07-01 — Offsets removed; the log-box experiment

The additive offsets `x_offset`/`t_offset`/`f_offset` were **removed from the inferred set** (no basis
in the model math; they were pinned to ±1e-6 ≈ 0). 16 → 13 dims. Offset access is now
optional/guarded throughout. Old 16-dim posteriors fail loudly against a 13-dim cell.

**`[TRIED → PARTIAL]` — Track A, the decorrelating rotation.** A redistribution, not a clean win: it
fixed `kappa` and `t_offset` but broke `N` and `dG` and left `lambda` stuck. **Diagnosis:** `V` was
computed at a single point (GT) in a LINEAR box, but the dominant degeneracies are *multiplicative*
(hyperbolae) — a single linear rotation straightens the GT tangent and re-correlates off-GT. Textbook
ceiling of a single-point linear decorrelation.

**`[TRIED → SPLIT RESULT]` — Track A+.** Multi-point Fisher averaging **helped** (recovered N/dG/tau_c,
as intended). **LOG-space box HURT** the degeneracy parameters (`x_scale`/`t_scale`/`kappa`/`lambda`) —
net worse. Artifacts were ruled out on-disk (convergence, eval box, failure mode, prior rebuild + V
geometry): the result is **real but mild** (rank histograms ~flat with a small monotone tilt). The
mechanism: in log coordinates the products did **not** linearise — the rotation over-mixed, scattering
`lambda` with no clean `lambda~t_scale` mode. ⇒ next was 13-dim LINEAR + multi-point, which became the
keeper.

## Conventions worth keeping

- **Trust SBC KS p-values, not `c2st_ranks`** (~0.58 is the c2st finite-sample floor).
- **SBC power: use `n_cal >= 2000`.** A single `n_cal=1000` under-reports. But **pooled rank
  histograms read the severity KS cannot** — ~flat means mild even when KS p is low at n=2000.
- **A saved posterior is described by** (bounds from the cell file) + (log_mask) + (rotation V). The
  last two live in the `<name>.rot.pt` sidecar. **Always** reconstruct the eval box via
  `reparam.load_eval_bijection(cfg, POST, dir)`, never a bare `build_inferred_bijection(cfg)` —
  config may have drifted since training.
- **The latent prior is MIXED-DEVICE** (CPU rescale bijection + CUDA ND GMM). The pipeline only ever
  **samples** it, never `.log_prob` — keep it that way.
- **Changing `REPARAM_LOG_PARAMS` requires rebuilding the ND prior** (the latent GMM is fit in the box
  coordinate); `build_posterior` raises on a mismatch. Only `lo > 0` params are eligible. *Nuance:*
  only **ND** log params force a rebuild — rescale-only ones do not.
- **Signal generally improves with longer `T_obs`**, especially for the weakly-imprinted parameters.
- **A true instant/hard cancel (QProcess subprocess) was explicitly declined** — cooperative was chosen.
