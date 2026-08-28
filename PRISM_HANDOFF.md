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
- **§11 is the informativeness programme — IMPLEMENTED 2026-08-26, NOT YET RETRAINED.** All four
  phases are code, tests and a migrated cache; **§11.9 is the runbook for the two runs that are
  owed.** It carries its own evidence ledger, because it began as a user-written addendum whose
  claims were then verified against the artifacts — **§11.2 lists the four that did not survive, one
  of them the test that addendum nominates as decisive, and a fifth correction added on 2026-08-26
  where §11.2 itself gave the wrong REASON for a right number.** Read §11.2 before acting on it.
  Two of its own instructions were also overtaken by measurement: the valid-flag set is **eight
  channels, not one**, and §11.8's "edit `master.txt`" would have orphaned the cache Phase 1 needs.
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
| Scripts | `python scripts/<name>.py` | 15 non-interactive diagnostics (+ `_common.py`, the shared config builder). Env knobs are documented in each file's docstring. |

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
| `tests/test_gui_progress.py` | 98 | tqdm classifier, Qt stack, panels, pop-out, nav/gating, Simulate stream + video, labels, **layout geometry**, **the χ probe table + planner** (C-2/C-3), and **the one-bar progress display** — the solver meter reading `core.progress.SOLVER` rather than a rendered bar, the largest-total overall-bar election, and the caption's fallback to sbi's epoch counter (§10.5), and **the GUI training budget** — the derived simulation count, the peak-memory estimate against `pipeline`'s own cost model, and the budget reaching `build_posterior` as ARGUMENTS rather than through the config (§6 row 5), and **the TSNPE tab** — its gate, and an `ast`-level check that its runner never references a proposal or `set_default_x` (§11.6) |
| `tests/test_user_models.py` | 39 | sympy parser/codegen, forcing kinds + the sin golden test, persistence, registry, **the v1→v2→v3 param migrations, the log-box validation and its route to the prior mask, physical forcing bounds, the blank-bound guard, icon-glyph coverage, the app icon, and `build_app` itself (which had no test at all)** |
| `tests/test_user_sbi.py` | 66 | spontaneous + chi SBI paths, the built-in-path-unperturbed guard, memory/geometry regressions, the box round-trip invariant, the posterior-mode decoder, the JIT/eager contract, the dt and off-grid guards, the fixed-K calibration lever, the OOM retry ladder + learned memory budget, the batch-level retry and its gen_training_data wiring, the capped z-score check, **the seeded-reproducibility gate, the bit-identical resume, the stratification-across-the-seam check, the checkpoint store + atomic write**, and **the CUDA-graph solver path** (graph/eager equivalence, the RNG contract C-11 depends on, the X1 cache-placement guard), and **every GUI-exposed knob arriving as an ARGUMENT** rather than a config write, checked both in the signature and at the point it is used (§2.3) |
| `tests/test_chi_set_encoder.py` | 23 | the chi probe-set encoder and packer, pure torch and fast — permutation invariance, **bitwise** pad inertness, pad-width invariance, masked-mean-over-live-count, the post-gate, empty/singleton sets, mask binarisation, the packer's round-trip / masked-not-phantom behaviour, and **the Fisher set's one-argument signature + channel identity** (C-9/C-10), and **`probe_verdict`'s refuse/mask/truncate split** (C-3) |
| `tests/test_artifact_consistency.py` | 15 | the master Bounds/Cells triple, bounds resolution, the prior/posterior identity guards — **eval box comes from the sidecar, not the config** — **the end-of-run artifact writes being atomic** (posterior, prior, loss curve), and **the χ band/drive preflight** (trap Q4) |
| `tests/test_fdt_user.py` | 5 | FDT for user models (FEATURE 1 v3 / B-d) |
| `tests/test_conditioning_repair.py` | 25 | **§11**, and pure torch in seconds like the encoder suite: rank-Gaussianisation (monotone, the mid-rank tie rule, the dead-channel pass-through, and the contaminated channel becoming visible), the valid flags (each predicate, the B7 pair sharing one flag, `tau_slow`'s dt dependence), winsorisation leaving the χ block **bitwise** untouched, the pathological-trajectory counter, **the KL scalar's zero point** (a posterior that IS the prior scores exactly 0 nats) and its per-block retry on a mixed-device product prior, tier-1's round trip through the implied temperature and its refusal without `nd_idx`/`k_b_cell`, and **the TSNPE pinning test the whole of §11.6 rests on** — that the proposal has the truncated PRIOR's width, not the posterior's and not the posterior's over √2. Also: **a legacy `sum_mean`/`sum_std` posterior still loads and evaluates**, without which every pre-2026-08-26 artifact — including the §11 baseline — is unopenable |
| **Total** | **271** | |

> The encoder suite is the one to run first when touching chi: it is seconds, needs no simulation, and
> the invariants it pins are the ones whose violation is invisible — a subtly non-invariant encoder
> trains perfectly happily and produces a posterior nobody can explain.

> ⚠ **A green suite does NOT certify the GPU path.** Every suite runs on the CPU (`config.cpu_device()`
> or small CPU configs), so nothing in it can catch a tensor built on the wrong device: a CPU tensor
> into a CUDA matmul is a hard `RuntimeError`, not a silent promotion, and it fires only on the card.
> C-11 shipped exactly that bug into `build_posterior` with all 55 tests passing, and `smoke_train.py`
> on the GPU is what found it — **after the Fisher rotation had already run for over an hour**. Two
> tests now early-return with a printed note when `torch.cuda.is_available()` is False rather than
> pretending to cover it. **After touching anything that moves tensors, run the smoke train on the
> card; the suite is necessary and not sufficient.**

> **Runtimes are wildly uneven — budget for it.** Five of the six suites finish in seconds to a few
> minutes. **`test_user_sbi.py` takes ~1 hour**, nearly all of it inside
> `test_chi_mode_full_sbi_pipeline`, which runs a whole chi pipeline on CPU — and chi calibration
> costs `(K+1)` simulations per batch with `K` drawn up to `CHI_K_PAD`. It is not hung; check for
> movement in the `Running time segments` bars before assuming otherwise.

> ⚠ **DO NOT EDIT A SOURCE FILE WHILE A SUITE IS RUNNING** — it produces a failure that is not real
> and costs an hour to disbelieve. Several tests assert on `inspect.getsource(...)`, which reads the
> file **as it is now** using the line numbers the function was **loaded** with. Insert eight lines
> above such a function mid-run and it extracts a shifted block, so
> `test_build_posterior_takes_the_budget_as_arguments_because_the_constants_are_snapshotted` reports
> that `build_posterior` "no longer resolves the budget from its arguments" when it plainly does.
> Freeze the tree, then run.

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
                  paths, VALID_MODELS/VALID_LABELS, TRAINING_*, CHI_*, SOLVER_CUDA_GRAPHS,
                  QUIET_SEGMENT_BAR, SOLVER_BAR_DESC, memory_budget_elements().
  orchestrator.py The 5 stage functions: generate_observations -> build_prior -> build_posterior ->
                  validate_calibration -> infer_and_visualize. Plus save_*_artifacts and _emit/fig_sink.
  registry.py     Runtime model registry (ModelSpec, register, load_user_models, is_user_model,
                  is_sbi_user_model, state_dep_drift, fdt_support).
  forcing.py      Kind-dispatching force builders (sin/step/triangular/exponential),
                  build_user_force_tensor, and n_force_channels (the shared channel rule).
  Models/         nadrowski_model, hopf_model, bp_model(+_steady), user_model (sympy parse + torch).
  Simulator/      Simulator ABC + per-model subclasses + UserSimulator. SimulationError lives here.
  progress.py     A monotonic count of completed SDE steps, published by the solver and sampled
                  by whichever front-end wants a rate. Qt-free, torch-free (§10.5).
  Solvers/sdeint.py  Euler-Maruyama: an eager loop, and a TorchScript step replayed from a CUDA
                  graph (§8.3). The `torch.compile fast path` and the `unused implicit solver`
                  this line used to name never existed / were deleted (§7.7).
  SBI/            pipeline (gen_obs/gen_stats/gen_training_data/train_nn/gen_chi_raw+gen_chi_block),
                  statistics (the 41 features), chi (chi(omega) math + the probe-set PACKER),
                  chi_encoder (ChiSetEncoder: the permutation-invariant probe-set encoder),
                  embedded_network (the two-pathway conditioning net), reparam (latent box +
                  rotation), decorrelate (Fisher rotation), analysis (SBC data + PPC), overlay
                  (posterior-overlay diagnostics), training_checkpoint (C-11: resumable
                  training-data checkpoints), Priors/.
  FDT/            campaigns, spectral, fdt_pipeline, cross_validation, sanity.
  Reduction/      NWK->Hopf normal-form map. IRRELEVANT to SBI — do not reason about it here.
  Helpers/        file_manager, helpers, visualizers, labels, model_store.
  gui/            see §2.2
scripts/          16 non-interactive diagnostics + _common.py (the shared SimConfig builder).
                  Newest: posterior_identifiability (what a SAVED posterior actually measured --
                  reads the artifacts, simulates nothing; NOT identifiability_offgt, which simulates),
                  smoke_train (the pre-flight five-stage run), chi_mask_audit (why chi probes get
                  masked), chi_f0_sweep (the band/F0/T_obs/cycle-cap measurement).
Resources/        Bounds/ Cells/ Units/ Models/ (inputs) + Plots/ Priors/ Posteriors/
                  CrossValidation/ Checkpoints/ (generated outputs, all gitignored).
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
                inference_tabs (the FIVE inference tabs; Posterior also owns the TRAINING
                BUDGET — batches x rows-per-batch — see §6 row 5), fdt_panel, reduction_panel,
                crossval_panel, simulate_panel + simulate_runner + simulate_export.
  widgets/      artifact_picker, figure_stack, figure_window, log_pane, progress_pane,
                live_hair_bundle, labeled_inputs, help_badge, param_grid, source_toggle, anim.
  session.py    SbiSession + ConfigDraft.       worker.py  Worker(QRunnable) + WorkerSignals.
  streams.py    redirect_streams + CancelToken + WorkerCancelled.    vt.py  tqdm chunk classifier.
  design.py / fonts.py / theming.py    the Fluent token + QSS layer.
  plot_watcher.py  NewPngWatcher: polls a plot dir and emits new PNGs.
  mpl_theme.py  drives matplotlib rcParams from the design tokens (B-c).
  icons.py / app_icon.py  the bundled icon FONT (B-e) and the application/window icon.
  assets/       fonts/ (Inter), icons/ (prism-icons.ttf + its generator), app/ (the PRISM mark:
                prism.svg is the editable source, the PNGs are what ships -- Qt has no svg
                image-format plugin here, so QIcon("x.svg") would be silently NULL).
```

**Navigation** is two levels deep: Home → one of Reduction Map / FDT Analysis / Parameter Inference /
Simulate (+ Settings and the model builder, reached from Settings). The app always opens on Home;
only geometry and each panel's own settings persist.

### 2.3 What the Inference tabs expose, and the rule every one of them follows

Added 2026-08-27. Before that the only tunable in the GUI was the training budget (§6 row 5);
everything else meant editing `config.py`.

> ### ⚠ THE RULE: EVERY KNOB IS AN ARGUMENT, NEVER A CONFIG WRITE
> `orchestrator` does `from .config import PRIOR_SWEEP_ITERATIONS, NSF_HIDDEN_FEATURES, SBC_N_CAL, …`,
> which binds all of them at **import**. A panel that "configured" a run by assigning to the constant
> would change nothing, and the run would use the default with nothing to say so. Every field below
> therefore travels as a keyword argument defaulting to `None` (= follow the config), and
> `test_the_sweep_and_flow_knobs_are_ARGUMENTS_because_the_constants_are_snapshotted` checks both the
> signature **and** that each one reaches the thing it configures — accepted-and-dropped is the
> failure a signature check alone misses.

| tab | field | argument | default |
|---|---|---|---|
| **Prior** — *Stability sweep* | Global rounds | `build_prior(num_iterations=)` | `PRIOR_SWEEP_ITERATIONS` 50 |
| | Candidates per round | `sweep_batch=` | `PRIOR_SWEEP_BATCH` 0 (= hw batch) |
| | Max accepted sets | `max_sets=` | `PRIOR_SWEEP_MAX_SETS` 175 000 |
| | Random-walk step | `walk_step=` | `PRIOR_SWEEP_STEP` 0.01 |
| | Stability duration | `stability_units=` | `STABILITY_SWEEP_ND_UNITS` 1000 |
| **Prior** — *Clustering / GMM* | Min cluster size | `min_cluster_size=` | `PRIOR_CLUSTER_MIN_SIZE` 50 |
| | Min samples | `min_samples=` | `PRIOR_CLUSTER_MIN_SAMPLES` 10 |
| **Posterior** — *Training budget* | Batches / Max rows | `build_posterior(num_runs=, run_size_cap=)` | `TRAINING_NUM_RUNS` 5000 / `TRAINING_RUN_SIZE` 0 |
| **Posterior** — *Density estimator* | Hidden features | `hidden_features=` | `NSF_HIDDEN_FEATURES` 128 |
| | Transforms | `num_transforms=` | `NSF_NUM_TRANSFORMS` 8 |
| | Learning rate | `learning_rate=` | `TRAINING_LEARNING_RATE` 1e-3 |
| | Early-stop patience | `stop_after_epochs=` | `TRAINING_STOP_AFTER_EPOCHS` 20 |
| **Posterior** — *Fisher rotation* | Ensemble per perturbation | `fisher_m=` | `REPARAM_FISHER_M` 48 |
| | Central-difference step | `fisher_dz=` | `REPARAM_FISHER_DZ` 0.1 |
| | Operating points | `fisher_points=` | `REPARAM_FISHER_POINTS` 8 |
| **Validate** | Calibration datasets | `validate_calibration(n_cal=)` | `SBC_N_CAL` 2000 |
| | (t_scale, T) operating points | `cal_n_scales=` | `CAL_N_SCALES` 200 |
| **TSNPE** | HPD level / directions | `build_posterior(truncation=)` | `truncate.DEFAULT_HPD` 0.999 / `DEFAULT_N_DIRECTIONS` 5 |

**Five of these are NOT speed dials, and the tooltips say so:**

- **Candidates per round** — the global sweep is ITERATION-bounded, so shrinking it makes the prior
  worse *without* making it faster (527 s at 2048 against >70 min and unfinished at 32; C-7).
- **Stability duration** — defines what *stable means*, so it moves the prior's support.
- **Min cluster size / min samples** — HDBSCAN's label count IS the GMM's component count. Measured:
  5 → **15 components**, 60 → **1**. A different component count is a different prior.
- **(t_scale, T) operating points** — `t_scale`'s effective sample size (trap **X5**). Lowering it is a
  DIFFERENT measurement, not a faster one.
- **Max accepted sets** — buys COVERAGE of the stable manifold, not statistical precision.

⚠ **The three Fisher knobs do nothing on a RESUMED run**: it reuses the checkpoint's stored `V` and
skips the rotation entirely (trap **X10**), because `V` is not reproducible across processes.


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
`[ S(41) | VALID FLAGS(8) | log(T_obs) | forcing-or-chi ]`: the flags and `log(T)` ride the summary
pathway; the forcing/chi block is a separate `EmbeddedNet` pathway. **Keep this order in sync**
across `generate_observations`, `gen_training_data`, `validate_calibration` and the experimental
paths — in practice you do not have to, because all eight assembly sites are
`cat([gen_stats(...), log_T, forcing-or-chi])` and `gen_stats` is the single place the summary block
is widened.

**The flags say which features are MEASUREMENTS** (1) rather than substituted sentinels (0), the
same convention as the χ probe mask. `statistics.derive_valid_flags` is the one derivation, run by
both the live pipeline and the cache migration, so they cannot drift. Widths quoted below are the
current ones: `statistics.SUMMARY_WIDTH` (= 49) `+ 1`. **Pre-2026-08-26 posteriors are 42 + forcing**
and load through `EmbeddedNet`'s legacy branch; the cross-mode guard turns an old-versus-new load
into a message rather than a shape assert.

The mode is chosen by **which bounds file** is picked (does it declare a Forcing section?) plus the
chi toggle. The three widths cannot collide, so a cross-mode posterior load fails loudly.

| # | Mode | Data | Dims | Conditioning |
|---|---|---|---|---|
| 1 | `spontaneous` | one passive trace | 12 | 50 |
| 2 | `forced` | passive + one forced trace | 13 | 54 |
| 3 | `chi` | passive + **any number** of single-tone forced traces | 13 | 50 + 6·K_PAD |

Mode 1 **drops `f_scale`**: it only ever divides a force, and this mode never builds a force tensor,
so its marginal would just return the prior. Mode 3 **ignores the cell's own drive** (chi probes at
its own frequencies), so it is independent of `has_forcing` and cannot be out-of-distribution in
the drive.

**Mode 3 is a padded SET, not a vector (layout 2).** Probe *j* occupies slot *j* as six channels —
`(u, log|chi|, cos, sin, logcyc, mask)` where `u = log(f_probe/Ω₀)` and `logcyc = log(f·T)` — `T`
being the duration actually **locked in over**, which is bounded on both sides: below by
`CHI_MIN_CYCLES` (under it the probe is masked) and above by `CHI_MAX_CYCLES` (over it the segment is
shortened, §4.3.1). Both bounds are therefore visible to the network in `logcyc`, which is what lets
it discount a short probe — and why evaluating under a different ceiling than training feeds it a
value the training set never contained. The
frequency is carried **explicitly**, not implied by the slot index, and the encoder
(`SBI/chi_encoder.ChiSetEncoder`) is permutation-invariant, so **the number of probes and their
placement are both free**. Width is a function of `CHI_K_PAD`, never of K — which is exactly what
lets one posterior serve an experiment that managed 7 recordings at whatever frequencies it could
achieve. Three properties are load-bearing and each has a test:

- a dead slot is **exactly 0.0** in all six channels, so it is bitwise inert and no downstream
  finite-filter drops a row;
- a failed probe is **masked, never a phantom** — the old packer's `nan_to_num` turned a NaN lock-in
  into a live-looking `(0,0,0)` triple that `cos²+sin²=1` says no real probe can produce;
- probes are packed contiguously, ascending in frequency, so the simulated and experimental paths
  produce byte-comparable blocks.

**chi trains with `z_score_x="none"`.** sbi's default fits a per-COLUMN affine, which over probe
slots is permutation-*breaking*, and the near-constant mask column becomes a ~1e7 amplifier under
sbi's 1e-7 min-std clamp. `EmbeddedNet` standardizes instead: per channel over live probes only.

`CHI_K_PAD` is **frozen into every artifact** (sbi bakes `condition_shape` into the saved posterior).
Raising it invalidates existing chi posteriors — the load path turns that into a message rather than
a shape assert hours into a run. Width alone can never identify a layout: `6·5 == 3·10 == 30`, an
exact collision with the retired layout 1, which is why the sidecar carries `chi_layout`.

The valid flags are **excluded from the Fisher feature set** (`decorrelate.stats_no_flags`), for the
reason C-9/C-10 removed `logcyc` and `mask` is absent from `CHI_FISHER_CHANNELS`: `fnoise` is a
DENOMINATOR floored at 1e-9, so a binary channel that steps between the ±δz arms writes ~1e9 into
J. Trap **CHI10**'s class.

The Fisher rotation works in **all three modes** — see §4.4 for the measurement that overturned the
old chi exclusion. It builds its Jacobian over the *Fisher* channel set (4 per probe, no `u`, no
`mask`), which is a different feature set from the conditioning block for a reason recorded there.

---

## 4. Current status

### 4.1 START HERE

**Everything before 2026-08-05 is archived under `archive/2026-08-05-pre-consolidation/`.**
`08192026_posterior_RETIRED_band_0p1_to_10.pt` trained on the retired χ band and must not be used for
anything.

> ### ▶ THE RETRAIN RAN. `posterior_08232026` EXISTS, AND IT IS CHARACTERISED — SEE §4.6.
>
> Short version: **calibrated, converged, and uninformative in most directions.** SBC flat on all 13,
> TARP on the diagonal, the flow cleanly converged — and PPC coverage 99.1 % at a nominal 90 %, i.e.
> intervals far too wide. It is the same "well calibrated but uninformative" outcome as
> `posterior_chi_08042026`, reached honestly. **§4.6 says which directions it did measure, and why the
> obvious fixes are ruled out.** The Nadrowski inputs were
consolidated to ONE bounds pair
(`Bounds/nadrowski/master.txt` + `master_spont.txt`) and THREE cells (`Cells/nadrowski/master_spont`
/ `master_weak` / `master_entrained`), chosen by measurement — see `scripts/build_master_cells.py`
and the 2026-08-05 Appendix A entry.

- The first chi posterior (`posterior_chi_08042026`, K=10, F₀=0.1, 0.1–10×Ω₀, 5 days of training)
  came out **well-calibrated but uninformative**: SBC flat on 12/13, TARP ATC 0.523 / KS p 1.000,
  PPC mean|z| 0.582 — and every ND marginal at the prior. **Diagnosed, not mysterious:** 8 of its 10
  probes measured noise. *The diagnosis was refined on 2026-08-06 (§4.3.1): those probes failed
  because they ran past a ~31-drive-cycle reproducibility wall, not because their frequencies are
  intrinsically unusable. The remedy is a duration cap (C-4), not a narrower band.*
- The 2026-07/26-27 forced retrain and `posterior_07012026` (the old KEEPER) both predate the
  `B1_log_Q` fix (§7.6) and the consolidation. Their §4.2 numbers stand as historical record only.

**The chi conditioning is now a padded probe SET (layout 2, §3.6) and the Fisher rotation is
available under χ (§4.4). All six suites pass** — the per-suite counts live in §1.3 and only there,
because a total repeated across sections is a number that goes stale twice as fast (this line said
185 while §1.3 said 213). **COUNT them; do not trust either.**

**C-6 is resolved** (§4.3.3–§4.3.6). The first end-to-end smoke train found 77 % of training probes
masked — a typical row conditioning on ~2 live probes, which is `posterior_chi_08042026`'s
uninformative signature reached from a new direction. Diagnosed to the prior's ~4-decade Ω₀ spread
(not `T`, not the band) and fixed in two parts: **per-ROW probe placement** and **per-ROW lock-in
durations**. Live probes 41 % → **64 %**, inert rows 55 % → **33 %**. The remaining third is the
genuinely unmeasurable tail and is a question about the PRIOR, not a blocker.

Confirmed end-to-end: a fresh smoke train puts TRAINING at **36.8 %** masked (the audit said 35.8 %,
so the cheap audit is a good proxy). The PPC's 79.9 % is expected and is a readout of the posterior,
not of the probe machinery — see §4.3.6.

**The lock-in duration ceiling (C-4) is IMPLEMENTED**: `config.CHI_MAX_CYCLES = 20`, applied in
`pipeline.gen_chi_raw` so training, the Fisher rotation, the PPC and the experimental path all
measure the same observable, carried on the `SimConfig`, frozen into the sidecar and checked on load.
Without it, `T ~ logU[1 s, 60 s]` puts a large fraction of probes past the reproducibility wall
(§4.3.1) and a retrain spends days reproducing `posterior_chi_08042026`'s failure mode.

> ### ⚠ THE RETRAIN HAS BEEN ATTEMPTED TWICE AND DIED BOTH TIMES — ON MEMORY, NOT SCIENCE
>
> Neither failure said anything about chi(ω); both were CUDA out-of-memory on a **16 GB card shared
> with a Windows desktop**, and each was a different unguarded allocation. Read traps **X6/X7/X8** and
> the two Appendix entries (2026-08-10, 2026-08-11) before restarting.
>
> | | died in | wrapped? | fix |
> |---|---|---|---|
> | 2026-08-10 | the SDE solver, mid-generation | yes (`SimulationError`) | learned memory budget + `_gen_obs_retry` |
> | 2026-08-11 | a chi batch, **outside** the solver | **no** | `_rows_with_oom_retry` (per-batch, halves ROWS) + `gen_obs` preallocation |
>
> **Plus one that had NOT fired and would have:** `append_simulations` moved the whole 4.35 GiB
> training tensor to the GPU and ran `torch.unique` on it twice (≥13 GiB) — at the END of a
> multi-day run that is not checkpointed. Defused (`data_device="cpu"` + a capped z-score check) and
> **smoke-tested clean 2026-08-11** (all four stages, training masked 36.7 %, zero OOM retries). Note
> the smoke train ran it on **128 rows**, so it proves the path executes, not the memory claim at
> scale — see that Appendix entry.
>
> **The single largest lever is still closing the browsers.** `mem_get_info` overstates free VRAM by
> the size of the desktop (measured 15037 MiB against nvidia-smi's 5814), so a busy card is not just
> slower — it is invisible to the planner. Under 7 GiB of pressure a worst-geometry batch took 136 s
> against 18.6 s.
>
> **Nothing is checkpointed.** A death at batch 4000 of 5000 loses everything. The retries remove the
> failure modes that have actually bitten, but a driver reset or reboot still costs the whole run.

**The retrain is UNBLOCKED as of 2026-08-10.** Step 3 (§4.4.1) answered the information question —
chi buys `f_scale`, `t_scale` and `lam`, does not buy `k`~`x_scale`, and the set encoder lost nothing
versus §4.4. Getting there exposed that a standardized Jacobian amplifies any *nearly*-constant
channel (trap **CHI10**), which also reached `decorrelate` — measured, confirmed, and closed as
C-9/C-10 by removing `logcyc` from the Fisher set. Every C-item that gates a retrain is now done;
what is left open (C-2, C-3, C-7) does not. What remains:

**Recommended order:**

0. **`git status` first.** Everything since `eeaa5c5` is LOCAL and uncommitted — the consolidation,
   the artifact-identity guards, rotation-in-chi, the set-encoder layout, all of C-1/C-4/C-5/C-6/C-8,
   and the 2026-08-08 `degeneracy_map` fix. Nothing has been committed on your behalf. **Commit before
   the retrain** so the posterior traces to a revision.
   *Untracked files that are new since `eeaa5c5` and easy to miss:* `scripts/smoke_train.py`,
   `scripts/chi_mask_audit.py`, `scripts/chi_f0_sweep.py`, `scripts/build_master_cells.py`,
   `core/SBI/chi_encoder.py`, `tests/test_artifact_consistency.py`, `tests/test_chi_set_encoder.py`.
   `Resources/Priors/_c6_prior.pt` is a throwaway from the audit — delete it, do not train against it.
1. ~~**Gate the band on `T_obs` before spending days.**~~ ✅ **DONE 2026-08-06 — and it did not come
   back clean. See §4.3.1.** The band fails on the T axis, but the binding variable is **drive
   cycles**, not probe frequency: every full-length failure sits above ~31 cycles, and re-locking the
   *same trace* over a ≤20-cycle prefix restores every one of them. Under the cap the band's interior
   (`0.03`–`0.14×`) clears outright; its high edge `0.3×` remains marginal on phase coherence, which
   is a frequency effect the cap does not touch, and the retired near-resonance band was re-measured
   and does **not** come back. With the cap in force (step 2) the band was re-measured end to end
   (§4.3.2) and **`(0.03, 0.3)` stands unchanged**. Do not "fix" `CHI_FREQ_BOUNDS` to the script's
   full-length recommendation: that column is the pre-cap picture, and a one-frequency band cannot
   measure a curve's shape at all.
2. ~~**Backlog C-4 — the lock-in duration cap.**~~ ✅ **DONE 2026-08-06.** The wall was bracketed at
   **32–36 cycles** (M=48, in-band probes, worst CV climbing 0.042 → 0.198 across caps 8 → 32 and
   0.456 at 36), and `CHI_MAX_CYCLES = 20` sits in the flat part with ~3× margin. **C-5 then settled
   the band's high edge under that cap (§4.3.2): `CHI_FREQ_BOUNDS = (0.03, 0.3)` stands unchanged.**
   The band moves in neither direction — the retired near-resonance multipliers do not come back, and
   the narrowing §4.3.1 hinted at was an artifact of a chosen phase threshold.
3. ~~**`scripts/degeneracy_map.py`, chi vs forced.**~~ ✅ **DONE 2026-08-08 — §4.4.1.** The set
   conditioning did **not** lose what the retired fixed grid carried: the §4.4 baseline reproduces
   within noise (`k` 0.040→0.092, `x_scale` 0.043→0.127, `t_scale` 0.219→0.371, condition number
   2212→2093, `k`~`x_scale` 0.98→0.96). And the payload table finally answers the question it was
   written for — **which** features break which alias. See §4.4.1 for the reading.

   **It could not have been run as-is.** The script sliced `gen_chi_raw(...)[:2]`, binding `logcyc_v`
   to `u`, so every chi Jacobian it had ever produced was contaminated — trap **CHI10**. Two guards
   now stand where that was, and both fired on first use. Run it with the interpreter, from the repo
   root, **teed** (all tables are stdout-only apart from the `.npz`), and at a **deliberate
   `TOBS_S`** — the default of 1.0 s puts three of six probes under the resolution floor:

   ```bash
   TOBS_S=4.5 SEED=0 M=32 M_NOISE=128 CHI=0 BOUNDS=Resources/Bounds/nadrowski/master.txt CELL=Resources/Cells/nadrowski/master_weak.txt python scripts/degeneracy_map.py
   TOBS_S=4.5 SEED=0 M=32 M_NOISE=128 CHI=1 CHI_K=6 BOUNDS=Resources/Bounds/nadrowski/master.txt CELL=Resources/Cells/nadrowski/master_spont.txt python scripts/degeneracy_map.py
   ```

   `TOBS_S` **must match across the pair** (`J` is in signal-to-noise units and `fnoise` falls ~1/√T,
   so a 1.0 s map against a 4.5 s one is not a chi-vs-forced comparison at all), and forced **must**
   use `master_weak` — `master_spont` has no Forcing section and `assert_forced` exits. Outputs are
   mode-suffixed and now include `degeneracy_map_<mode>.npz` (J, fnoise, dead, labels, run meta), so
   the two runs diff by LABEL without a re-run; the snippet is in the module docstring.

   **Why 4.5 s, and why there is no comfortable answer.** The band's dynamic range (`hi/lo` = 10)
   **exactly equals** the cycle window's (`CHI_MAX_CYCLES/CHI_MIN_CYCLES` = 20/2 = 10), so exactly one
   `T_obs` puts the low edge on the floor and the high edge on the ceiling at once — measured
   `T* = 2.93 s` on this cell (Ω₀ = 22.78 Hz, printed by the run). Both edges cannot be cleared with
   margin. Err ABOVE: under the floor a probe is not a measurement in any of its four channels, over
   the ceiling it loses one of four. 4.5 s clears the floor by 50 % (3.07 cycles) and pins one probe.
   `chi.resolvable_multipliers` records the same collision from the training side.

   **Three things to know before reading the result.** First, the Fisher runs with
   `resolution_filter=False` — mandatory (trap CHI2), but it means the map is built over an
   **unmasked** probe set, while training masks ~37 % (§4.3.6). Its answer is an upper bound on the
   information training actually has. Second, `max_cycles` IS applied there (unlike the filter), so
   the probes are the same length training uses — without that the map would measure a lock-in longer
   than the network ever sees. Third, `adapt_placement` is OFF here and that is deliberate (it would
   make placement theta-dependent, trap CHI2's defect class); at `T_obs ≥ T*` it is a numeric no-op
   anyway, and the run prints the threshold so you can confirm rather than trust it.
4. ~~**A smoke train.**~~ ✅ **DONE 2026-08-06, re-run 2026-08-07 after C-6/C-8** via
   `scripts/smoke_train.py`. All four stages complete (~46 min); training masking 76.7 % → **36.8 %**.
   It is the cheapest possible check on a multi-day commitment — **re-run it after any change to the
   chi path**, and watch the per-stage masked fractions in §4.3.6.
4b. ~~**Verify C-9.**~~ ✅ **DONE 2026-08-10 — measured, reproduced, and fixed together with C-10.**
   `logcyc` left `CHI_FISHER_CHANNELS` (now 3 channels, 3K not 4K) and `fisher_features` takes one
   argument. The rotation's chi block went 24 rows → 18 with **zero** pinned channels, every
   surviving row untouched. **Nothing now stands between here and the retrain.**

4c. ~~**Run `scripts/smoke_train.py` after the OOM work.**~~ ✅ **DONE 2026-08-11** — all four stages
   in 8827 s, training masked **36.7 %** (against §4.3.6's 36.8 %), zero OOM retries, and the
   `append_simulations` fix exercised end to end. Re-run it after any further chi-path change; watch
   the masked fraction, any `OOM ...` lines, and the `[mem]` peak-allocated figure.

5. **The retrain** ⬅ **STILL NEXT** on `master_spont` with `master.txt` bounds, χ ON, **rotation ON**
   (§4.4).

   > ### ⚠ ATTEMPTED 2026-08-19 AND MUST BE REDONE — it trained on the RETIRED band
   >
   > 10.24M rows, ~5 days, completed cleanly and calibrated well — at
   > `chi_freq_bounds = (0.1, 10.0)`, `chi_f0 = 0.1`, `K = 10`, i.e. verbatim
   > `posterior_chi_08042026`'s configuration. Stale QSettings (trap **Q4**); full account in the top
   > Appendix A entry. The artifact is renamed `08192026_posterior_RETIRED_band_0p1_to_10.pt` and its
   > ~5 GB checkpoint is unusable (its identity digest includes the band, so a corrected run routes
   > elsewhere rather than splicing — safe, just wasted).
   >
   > **`_assert_chi_config_is_deliberate` now refuses this before the first simulation**, so the
   > specific failure cannot recur silently. Confirm before starting: the banner must read
   > `chi: K=6 F0=0.15 range=(0.03, 0.3)`.
   >
   > **Post-fix smoke train, 2026-08-20:** all four stages, 7183.6 s (prior 526, posterior 5772,
   > validate 204, infer 682), training masked **35.4 %** (249/704), zero OOM retries — squarely
   > inside the 34.8–38.2 % spread of every prior corrected-band run. The chi path is healthy and the
   > config fix broke nothing. **Cleared to retrain.**
   >
   > **The three hypotheses below are therefore STILL UNTESTED.** Do not read the 2026-08-19 result
   > as evidence about any of them — the band it used destroys exactly the phase channels §4.4.1 says
   > carry `lam`. The one thing it did confirm is the `k`~`x_scale` alias, which came back off by an
   > identical +11.3 % on both.

   C-6 is resolved, step 3 says go, and C-9/C-10 are closed. Budget
   ≈ (1 + E[K]) / 2 × a spontaneous run for the training data, plus (K+1)/2 × a forced-mode Fisher
   for the rotation. At `CHI_K_PAD = 12`, `E[K] = 7`. From the smoke train's timings, expect the prior
   build alone to be ~9 min and the Fisher rotation to dominate the pre-training cost.

   > **⚠ BOTH SENTENCES ABOVE ARE NOW MISLEADING — corrected 2026-08-20.** (a) The solver is **~8×
   > faster** since CUDA graphs landed (§8, 5520 → 698 ms per 100 k-step call), so every wall-clock
   > figure written before that date is an over-estimate. (b) "The Fisher dominates the pre-training
   > cost" is true of a 4-batch SMOKE train and false of a 5000-batch retrain, where training
   > generation is **~97 %** and the Fisher ~1–2 %. Budget against the production denominator.
   **Expect `k`~`x_scale` to stay aliased** — §4.4.1 is the third measurement saying so. The
   hypotheses worth testing on the result are `f_scale` (unmeasured → measurable, 213× on `‖g‖`),
   `t_scale` and `lam`.

   > ### The run is now CHECKPOINTED (C-11) — what that changes for you
   >
   > A resumable checkpoint is written to `Resources/Checkpoints/train_<digest>/` every
   > `TRAINING_CHECKPOINT_EVERY = 50` batches, and also on the way out of a cancel. **To resume after
   > any death, simply start the same run again**: the directory is a digest of the config, so a
   > matching run finds it and reports `[checkpoint] resuming at batch N/5000`. There is no GUI
   > control and nothing to remember.
   >
   > **New priors are fine. Always.** A new prior simply means a new run, which gets its own
   > checkpoint directory and proceeds normally. Nothing here restricts what priors you may build.
   >
   > The one rule is narrower: **you cannot RESUME an interrupted run under a different prior.** The
   > prior identifies the distribution the finished rows were drawn from, so splicing new rows drawn
   > from a different prior onto them would silently corrupt the training set — the checkpoint refuses
   > by routing elsewhere, which is the correct behaviour, not a limitation. So: if a run dies and you
   > want to resume it, keep the prior it started with (save it, and load it rather than rebuilding).
   > If you'd rather start over with a new prior, just do that; nothing is in your way.
   >
   > The run prints
   > `[checkpoint] N other checkpoint(s) exist and do NOT match this run: … differs in <field>`
   > when a checkpoint exists that it cannot use — read that line before letting a "fresh" run
   > proceed, in case the mismatch was accidental.
   >
   > A resume **reuses the stored Fisher rotation `V`** and skips recomputing it. This is not an
   > optimisation, it is required for correctness (trap **X10**), and it also removes the largest
   > pre-training cost from every restart.
   >
   > Watch for: the preflight line naming the directory and the free disk (~5 GiB is needed), the
   > `[checkpoint] resuming at …` line if you expected a resume, and `[checkpoint] complete:` at the
   > end. A completed checkpoint is **not** deleted — it is a cache of the whole simulation run, so
   > you can retrain the flow at a different capacity/learning rate without re-simulating. Delete it
   > once the posterior is saved and you are happy.
   >
   > `smoke_train.py` has checkpointing **OFF** by default (a complete checkpoint would short-circuit
   > the very simulation a smoke run exists to exercise). ~~**Run it once with `CHECKPOINT=1` on the
   > card before the record run**~~ ✅ **DONE 2026-08-13** — chi + rotation + CUDA + `CHECKPOINT=1` +
   > `SAVE=1`, all four stages in 7069 s, training masked 34.8 %, zero OOM retries, the preflight and
   > `complete: 4 batches` lines both present, and every saved artifact opened and checked.
   >
   > **The GPU RESUME has now been exercised too** (same day, second drill): two runs over one
   > `CKPT_DIR`, run 2 in **3.0 s** against run 1's 5125 s, reusing the stored `V` and skipping both
   > the Fisher and generation. That is the first time `build_posterior`'s CPU→CUDA rehoming of the
   > checkpoint's `V` has ever executed — see the top Appendix A entry. Trap **X10**'s guard is no
   > longer reasoned-to; it is observed.
   > ⚠ **Pass `BOUNDS=Resources/Bounds/nadrowski/master.txt`** when you re-run it: the default
   > resolves the 12-dim spontaneous box, which still calls itself `mode=chi` at the same conditioning
   > width and silently drops `f_scale`. The script's docstring now carries the full command.
   >
   > **VRAM, because the first attempt died on it (2026-08-10 Appendix entry, trap X6).** Check
   > `nvidia-smi --query-gpu=memory.used --format=csv` before starting — **not**
   > `torch.cuda.mem_get_info()`, which overstates free VRAM by the size of your desktop (measured:
   > 15037 MiB against nvidia-smi's 5814 MiB). The run now survives a busy card by splitting and
   > retrying, but survival is not speed: under 7 GiB of pressure a worst-geometry batch took **136 s
   > against 18.6 s** unpressured. Closing the browsers is worth more than any constant in this file.
   > Watch for `OOM at simulation batch ...` lines in the log — a few are fine, a steady stream means
   > the card is tighter than the split machinery can absorb and `TRAINING_RUN_SIZE` is the lever.
6. **Run SBC stratified by probe count** (n = 2, 6, `CHI_K_PAD`) as well as pooled. A pooled SBC over
   a mixture of probe counts can be flat while each count is miscalibrated in compensating
   directions — and `posterior_chi_08042026` is the standing proof that flat SBC is not by itself
   evidence of a working conditioning path. **The lever now exists:**
   `CHI_K_FIXED=<n> scripts/sbc_characterize.py` (added 2026-08-06 — before that,
   `gen_training_data` accepted a probe count and ignored it, so this step was not runnable).
7. `scripts/sbc_characterize.py` pooled on the result. The hypothesis to test is now specifically
   whether `k`/`x_scale` **tighten**, since §4.4 shows that is the alias chi does *not* break on its own.

> ~~**Still open (not blocking a retrain):** the Infer tab still submits a fixed probe list, so the
> GUI cannot yet enter per-probe frequencies even though the core accepts them.~~ **STALE — this was
> already false when written down.** C-2 and C-3 shipped 2026-08-10: the Infer tab holds an
> add/remove probe table (`_ChiProbeRow`, one widget per probe carrying its recording AND the
> frequency it was driven at) and a **Plan probes…** button. See §6.

> ~~**Caveat:** the `scripts/` diagnostics were not updated for chi-mode.~~ **DONE** (2026-07-28,
> top Appendix A entry). All of them now build their config through `scripts/_common.py`.
> `sbc_characterize`, `retrain_convergence` and `degeneracy_map` are chi-aware; `diagnose_fmax`,
> `identifiability_offgt` and `feature_candidate_test` **refuse loudly** under `CHI=1` because their
> metrics are defined over the 41-feature single-frequency set. A posterior whose mode disagrees with
> the config now fails in seconds, before the simulation spend, instead of after it.
>
> For step 3, run `degeneracy_map` in **both** modes (`CHI=0` then `CHI=1 CHI_K=6`) —
> outputs are mode-suffixed so they no longer overwrite each other — and diff the new
> `=== top features per parameter ===` tables. That table is the actual payload: it says *whether the
> chi features are what broke the alias*, not merely that the alias weakened.
>
> **Budget note for step 4's Validate:** chi calibration costs `CAL_N_SCALES × (K+1)` simulations
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

**Machinery + tests complete (layout 2, §3.6); NOT yet a trained, calibrated posterior under it.**
The rationale was that a passive trace plus a single-frequency forced trace only see the products
`D·A_nd` and `(lambda_hb/k_gs)·tau_nd`, so the **shape of chi(ω) over frequency** should separate
`kappa`/`lambda`/`x_scale`/`t_scale` individually.

> **That rationale is now PARTLY REFUTED by measurement — read §4.4 before planning around it.**
> chi does improve every parameter's unique handle, but it leaves `k`~`x_scale` at 0.95 (0.98 forced),
> and `lambda`~`t_scale` was never degenerate on the master cell to begin with (0.59 forced). The
> shape information chi was designed to extract lives near and above resonance, which is exactly the
> region where |chi| turned out to be irreproducible at any drive amplitude or recording length —
> hence the sub-resonance band below. Whether the remaining band carries enough shape is the open
> question a retrain answers.

- **Drive protocol:** single-tone × K recordings. Per observation = 1 spontaneous run (Groups A–F +
  Ω₀) + K single-tone forced runs, where Ω₀ is the spontaneous PSD peak **measured from the passive
  trace**. Cost ≈ (K+1)/2 × a spontaneous run.
  **K and placement are free** (§3.6): an observation uses the deterministic log-spaced grid over
  `CHI_FREQ_BOUNDS`, TRAINING draws K per batch over `[CHI_K_MIN_TRAIN, CHI_K_PAD]` and jitters the
  placement, and the EXPERIMENTAL path takes whatever frequencies you actually drove at. A fixed
  training grid would leave the encoder's frequency channel taking only K distinct values across the
  whole run, so a 0.07× bench recording would be an OOD input to an MLP that extrapolates linearly
  and confidently.
  A probe outside Nyquist, out of band, or below `CHI_MIN_CYCLES` drive cycles is **masked and
  counted**, never silently moved (the old code clamped to Nyquist, relabelling a probe as a
  different frequency than the one requested).
- **Mask vs refuse — the distinction is train/eval consistency, not severity.** Training MASKS a
  sub-cycle probe and keeps the row, so the network learns to condition on sets with absent probes.
  The experimental path therefore masks it too: refusing would reject an observation the network
  handles fine, and at the band's low edge that is routine (at Ω₀ = 7.6 Hz the 0.03× probe has a
  4.4 s period, so a 1 s recording cannot resolve it however well it was made). What the experimental
  path DOES refuse is a different kind of thing — a non-finite or non-positive frequency, an aliased
  probe, an out-of-band one, a count over `CHI_K_PAD`, or *every* probe masked. Those indicate a
  mistake to fix, not a limit of the recording. Do not "simplify" these into one behaviour.
- **Amplitude:** drive at a FIXED **ND** amplitude (dimensional `amp = CHI_F0 · f_scale`) so
  linearity and lock-in SNR are uniform across the `f_scale` prior. chi cancels the drive amplitude
  in the linear regime, so `CHI_F0` is only a linearity knob.
- **`CHI_FREQ_BOUNDS = (0.03, 0.3)` — SUB-RESONANCE ONLY.** ⚠ **This entry's DIAGNOSIS was overturned
  on 2026-08-06 by the `T_obs` gate (C-1) — read §4.3.1 before acting on it.** The band is not a
  frequency limit; it is a *drive-cycle* limit that a fixed `T_obs = 5 s` slice made look like one.
  The measurements below are real and reproduce; their interpretation was wrong.

  This is the single most important number
  in the mode, and the old `(0.1, 10.0)` is what made the first chi posterior uninformative.
  `scripts/chi_f0_sweep.py` on the master cell (M=24 seeds) measured |chi| reproducibility against
  probe frequency and drive amplitude:

  | probe | best CV | at F₀ | entrainment |
  |---|---|---|---|
  | 0.05×Ω₀ | **0.026** | 0.15 | none |
  | 0.1× | **0.029** | 0.2 | none |
  | 0.2× | **0.055** | 0.2 | mild |
  | 0.3× | 0.220 | 0.15 | mild |
  | 0.5× | 0.208 | 0.1 | onset |
  | 1× / 2× / 10× | 0.36–0.73 | *any amplitude* | — |

  ~~**The decisive control:** high-multiplier CV does **not** improve from `T_obs` 5 s to 25 s. A
  noise-limited lock-in would fall by √5 ≈ 2.2×; it does not move. So that variability is
  **systematic, not statistical** — same θ, different noise seed, genuinely different chi. Neither a
  stronger drive nor a longer recording recovers those probes; only avoiding them does.~~

  **That control tested the wrong direction and drew the wrong conclusion.** It checked whether a
  longer recording *helps*, found it did not, and inferred the probes were unrecoverable. Measured
  2026-08-06: a longer recording **actively hurts**, and a *shorter lock-in window on the very same
  trace* recovers every one of those probes (§4.3.1). "Systematic, not statistical" was right; "only
  avoiding them does" was not. At K=10 the old band did put 8 of 10 probes in the failing regime —
  but because those frequencies reach the cycle wall soonest at `T_obs = 5 s`, not because the
  frequencies are unusable.
- **`CHI_F0 = 0.15`, bounded from BOTH sides.** Too small and |chi| stops being reproducible
  (0.05× probe: CV 0.090 at F₀=0.05 vs 0.026 at 0.15). Too large and the drive **entrains** the
  bundle, which abandons its own rhythm and follows the drive, so chi reports the drive back to
  itself — onset measured at F₀ = **0.2**, which was the previous default. 0.15 is the largest
  amplitude reproducible across the whole band while leaving the bundle running free (own spectral
  peak ≥ 84 % of undriven at every probe). *Historical: the earlier "ND 0.2, CV 0.04/0.04/0.17"
  figures were measured on the archived `cell_2` across the OLD band, and do not transfer.*
- **Enabling it:** use the **GUI Config tab** (χ toggle + K / F₀ / frequency range). The knobs are
  carried on the `SimConfig`, so a run is self-describing. (`config.CHI_MODE` is the module default
  the CLI picks up; the GUI passes explicit values.)
- **Tunables:** `CHI_N_FREQS` (linear K× cost), `CHI_FREQ_BOUNDS` span, `CHI_F0`, and whether to keep
  a single-frequency Group G alongside chi (likely redundant — chi mode zeroes it).

### 4.3.1 The `T_obs` gate (C-1): inside the band it is a CYCLE limit, not a frequency one

**Measured 2026-08-06.** `scripts/chi_f0_sweep.py` gained `T_obs` as a third axis, plus three metrics
the earlier sweep did not have: a **driven/undriven SNR** (the same lock-in run on the *passive*
ensemble at the same probe frequency — free, since those traces already exist), **circular** phase
scatter, and each point's **drive-cycle count** beside the `CHI_MIN_CYCLES` gate. 5 T_obs × 5
multipliers, M=24, `F₀=0.15`, master cell. Cost: 30 simulations, ~15 min.

**The band does not survive the T axis.** At full length only `0.03×Ω₀` passes at every `T_obs`;
`0.0646` / `0.1392` / `0.3` — all inside the configured band — each collapse (CV 0.23→0.63, SNR →2.3)
at some T. But the failure boundary runs along constant **`multiplier × T_obs`**, not along
multiplier. Every full-length failure in the grid sits above **~31 drive cycles**, and every survivor
below:

| probe fails at | T_obs | drive cycles |
|---|---|---|
| 0.6× | 2.27 s | 31.0 |
| 0.3× | 5.18 s | 35.5 |
| 0.1392× | 11.81 s | 37.4 |
| 0.0646× | 26.9 s | 39.8 |

**The decisive control — same trace, shorter window.** Re-running the lock-in over a *prefix* of the
already-simulated trace spanning at most 20 cycles restores the reproducibility of **every** failing
point, the above-band `0.6×` control included:

```
   T(s)   mult  cycles      CV     SNR   ->  cyc@cap  CV@cap  SNR@cap
  11.81 0.1392   37.38  0.6341    2.27   ->    20.00 0.05635    18.26
   26.9 0.0646   39.77  0.4799    2.66   ->    20.00 0.03942    28.24
   11.81    0.3   80.56  0.5181    2.42   ->    20.00 0.06365    18.25
   26.9     0.6  369.35  0.3460    2.25   ->    20.01 0.08183    11.50
```

No new simulation, no different seeds, no different frequency — only fewer samples entering the sum.
A stationary, noise-limited lock-in cannot behave this way: less integration must mean *more*
variance. So the bundle's response at fixed θ is **non-stationary on the scale of tens of drive
cycles**, and the lock-in accumulates that wander instead of averaging it away. *(The mechanism is
inferred, not established — phase diffusion of the free-running limit cycle is the obvious candidate
and has not been tested. What is established is the effect and its remedy.)*

**The two failure modes are different, and only one is a duration artifact.** A second run over the
*retired* `(0.1, 10.0)` multipliers settles the obvious follow-up — does a duration cap bring the
near-resonance band back? **It does not:**

| probe | full length | capped at 20 cycles | why |
|---|---|---|---|
| 0.1× | USABLE | USABLE | — |
| 0.5× | noisy phase lowSNR entrained | **phase entrained** | entrainment: own peak 0.30 |
| 1× | noisy phase lowSNR | **phase lowSNR** | SNR **0.12** — the driven lock-in is *below* the undriven one |
| 2× | noisy phase lowSNR entrained | **noisy phase entrained** | entrainment: own peak 0.24 |
| 10× | noisy phase lowSNR | **noisy phase lowSNR** | SNR 0.77 |

So §4.3's sub-resonance restriction **stands on its own merits** — and the SNR metric reproduces it
from an independent direction: at and above resonance the response is entrained or simply smaller
than the bundle's own activity at the same frequency, which no amount of shortening fixes.
Entrainment in particular *cannot* be a duration artifact — it is a property of the driven trace's
spectrum, not of the lock-in window, which is why `sup` has no capped counterpart.

**Consequences.**

1. **The band's INTERIOR is a duration problem; its HIGH EDGE is not.** Under a 20-cycle cap,
   `0.03` / `0.0646` / `0.1392×Ω₀` clear all four criteria at every reachable `T_obs` — without the
   cap only `0.03×` does. `0.3×`, the configured high edge, is a different story: capped, its CV
   (0.048–0.065) and SNR (12–18) are fine, but its **phase scatter is 0.52–1.06 rad at every T,
   including at 7 cycles where nothing else is wrong**, and its own-peak retention (0.66–0.84) is the
   worst inside the band. Phase coherence and entrainment degrade *monotonically* with multiplier —
   0.9–1.0 own-peak at ≤0.14×, 0.66–0.84 at 0.3×, 0.12–0.16 at 0.6× — so `0.3×` sits on the same
   slope that becomes outright entrainment above it. **The top edge is where §4.3 always said it was;
   the cap does not move it.**
   ⚠ *`PHASE_MAX = 0.5 rad` is the one screen here with no empirical basis — it was chosen, not
   measured. Whether `0.3×` belongs in the band is therefore not settled by this run.* **→ Settled in
   §4.3.2: it does. The phase screen was measured to be smooth (no knee anywhere), so it discriminates
   nothing and is now advisory; entrainment, which does have a knee, puts the edge at 0.35–0.4×.**
   Do not take the script's `(0.03, 0.03)` from the full-length column either — mechanically correct,
   scientifically absurd, since a one-frequency band cannot measure the *shape* of χ(ω). Its band
   verdict prints full-length and capped columns side by side for exactly this reason.
2. **Cap each probe's lock-in duration instead** — ✅ done, `config.CHI_MAX_CYCLES = 20`. The wall was
   then bracketed on the same machinery by re-locking each trace over every prefix length (M=48,
   in-band probes only, so frequency effects cannot confound it):

   | cap (cycles) | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 |
   |---|---|---|---|---|---|---|---|---|
   | worst \|chi\| CV | .042 | .039 | .047 | .062 | .086 | .123 | .198 | **.456** |
   | worst SNR | 18.8 | 22.0 | 21.9 | 18.3 | 15.9 | 12.8 | 7.9 | **3.6** |

   A steady climb, not a cliff, so there is no "correct" value — only a trade-off, and 20 was chosen
   for margin (~3× to the 0.2 CV screen, 10× above `CHI_MIN_CYCLES`) plus the fact that it is the
   ceiling the rescue table above already validated on every failing point. **12–16 reproduce
   slightly better and were not chosen**, because nothing here measures the other side of the trade:
   a shorter lock-in is also less frequency-selective, and no experiment in this repo has priced that.
   The ceiling lives in `gen_chi_raw`, not in a caller — training, the Fisher, the PPC and the
   experimental path must all measure the same observable, and a ceiling applied in only one of them
   is silent. On the experimental path an over-long recording is **truncated with a warning**, not
   refused: the recording is fine, only its tail is unusable.
3. **`CHI_MIN_CYCLES` is only half a gate.** It is a floor; the data says there is a ceiling ~15×
   higher. At full length no cycle threshold works at all — informative points span 0.70–184.7 cycles
   and uninformative ones 35.5–369.4, so the two *overlap* and any threshold both over- and
   under-masks. That is not an argument for a better threshold: under a 20-cycle cap **every point in
   this grid clears the SNR floor**, so there is nothing left to separate. Bound the cycles and the
   ceiling half of the gate becomes unnecessary rather than better-tuned.
4. **A second measurement risk is now open, not closed.** Everything here is one cell, `F₀ = 0.15`,
   and `CYCLE_CAP = 20` is a guess that happened to work — the wall's location was not bracketed.
   The cap interacts with the band's low edge from the other side, too: `CHI_MIN_CYCLES = 2` is a
   floor and the cap is a ceiling, so a `0.03×` probe on a short recording can be squeezed between
   them. On this grid they do not collide (0.70 to ~40 cycles across the whole T range at 0.03×),
   but that is a fact about this cell's Ω₀ ≈ 23 Hz, not a guarantee.

### 4.3.2 The band's high edge (C-5): `(0.03, 0.3)` stands

**Measured 2026-08-06**, 11 multipliers through the edge region, M=32, all five `T_obs`, with the
duration ceiling in force. Worst-across-`T_obs` metrics **at the cap**:

| ×Ω₀ | 0.03 | 0.05 | 0.08 | 0.12 | 0.18 | 0.25 | 0.30 | 0.35 | 0.40 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| \|chi\| CV | .159 | .094 | .066 | .062 | .076 | .074 | .078 | .087 | .091 | .105 | .101 |
| SNR | 12.2 | 14.2 | 12.9 | 13.9 | 12.0 | 11.9 | 12.8 | 11.0 | 9.7 | 10.0 | 9.0 |
| phase (rad) | .13 | .14 | .23 | .36 | .52 | .69 | .85 | .98 | 1.12 | 1.34 | 1.52 |
| own peak | .93 | .96 | .95 | .87 | .85 | .75 | .67 | .57 | **.44** | **.26** | **.11** |

**Three findings, in order of how much they should be trusted.**

1. **Under the cap, CV and SNR do not discriminate anywhere in 0.03–0.6.** CV stays 0.06–0.10 across
   the whole range (the 0.159 at the low edge is the few-cycles effect, not a frequency one) and SNR
   never falls below 9. The reproducibility collapse C-1 found is **fully solved by the ceiling**,
   across a range twice as wide as the configured band.
2. **Entrainment has a real knee at ~0.35–0.4×.** Own-peak retention erodes gently to 0.35 (step
   ratios 0.86–0.98 per grid point) and then accelerates: **0.77 → 0.60 → 0.43** at 0.4 / 0.5 / 0.6.
   That is a change of character, not a threshold crossing, so it is a property of the cell rather
   than of a constant in the script. **This is the measured ceiling on the band.**
3. **Phase scatter cannot settle anything.** It grows smoothly 0.13 → 1.52 rad with no knee anywhere
   — step ratios settle to ~1.15 per grid point. Where it crosses a threshold is therefore a report
   of the threshold, not of the cell, and the choice is worth a **2.5× difference in band width**:
   0.5 rad puts the edge at 0.12×, entrainment puts it at 0.35–0.4×.

**Verdict: keep `CHI_FREQ_BOUNDS = (0.03, 0.3)`.** It sits below the one measured knee with margin,
and nothing that discriminates argues against it. §4.3.1's "0.3× is marginal" was an artifact of
`PHASE_MAX = 0.5`, exactly as flagged there — `scripts/chi_f0_sweep.py` now defaults that screen to
advisory (`inf`) so it stops being reported as a verdict, with the reasoning at the constant.

> **The judgement inside this, stated plainly.** A probe with irreproducible phase has lost its
> `cos`/`sin` channels, so it is a **half-useful** probe — but its `log|chi|` CV is still ~0.08 and
> its SNR ~12, so it is not a *corrupt* one, and the network can learn to down-weight two noisy
> channels. Entrainment is different in kind: a captured bundle reports the drive back to itself, so
> the probe carries information about the drive instead of about θ. That asymmetry is why entrainment
> gates the band and phase does not. It is **reasoning about the measurement, not a measurement** —
> the experiment that would settle it is a posterior trained at `(0.03, 0.12)` against one at
> `(0.03, 0.3)`, which is two multi-day runs and was never worth it before a first working posterior
> exists.

### 4.3.3 The smoke train's finding: **77 % of training probes were MASKED**

> ✅ **RESOLVED 2026-08-07 — 76.7 % → 36.8 %.** §4.3.4 is the diagnosis, §4.3.5 and §4.3.6 the two
> halves of the fix. This section is the original finding, kept because it is the only record of what
> the pipeline looked like before, and because the *way* it was found — an end-to-end run, after the
> unit tests were all green — is the point.

**Measured 2026-08-06** by `scripts/smoke_train.py` — the first end-to-end chi run on the real
bounds/cell files, real stability-screened prior, `master.txt` (13-dim), rotation on. All four stages
completed (prior 527 s, posterior 1744 s, validate 196 s, infer 195 s), so the **plumbing is sound**:
mode widths, the sidecar round-trip, the duration ceiling and the chi block all agree end to end.

**But the conditioning is nearly empty.** Pooled over the run: **4362 / 5690 probes masked = 76.7 %**,
per batch 31 %–96 %. At `CHI_K_PAD = 12` with K drawn over 2–12, a typical training row therefore
carries **~2 live probes**, and a chi observation conditioned on two probes is barely distinguishable
from a spontaneous one. **That is `posterior_chi_08042026`'s exact failure signature — a
well-calibrated posterior at the prior — reached from a different direction.** Training on this
distribution is how you spend days to rediscover it.

**Attribution (controlled A/B, identical θ and strata, only `max_cycles` differing):**

| arm | masked | median cycles |
|---|---|---|
| ceiling ON (20) | 44.1 % | 2.60 |
| ceiling OFF | 36.7 % | 5.96 |

So the C-4 ceiling contributes **~7 pp and is not the cause** — masking is already 37 % without it.
*(Those absolute figures come from a box-uniform θ stub, not the screened prior, so they are not
comparable to the 76.7 % above; only the DIFFERENCE between the two arms is attributable.)*

### 4.3.4 C-6 diagnosed: it is **Ω₀**, not `T`, and the masking is CORRECT

`scripts/chi_mask_audit.py` separates the predicates the runtime warning lumps together, on the real
screened prior (16 batches, 2752 probe-rows, `master.txt`).

> ⚠ **My first reading of §4.3.3 was wrong and is corrected here.** I diagnosed a "(band × T)
> interaction" in which short-`T` batches mask the band's lower half. The measurement says the live
> fraction is **flat in `T`** (37 / 47 / 38 / 44 / 34 % across `T` buckets from <2 s to >30 s) and
> **flat in multiplier** (38–46 % across the band). Neither is the driver. The plausible mechanism
> was not the actual one.

**Only one predicate is ever active** — the `CHI_MIN_CYCLES` floor accounts for 100 % of masking.
Nyquist, non-finite frequencies and the packer's band filter each contribute **exactly zero**.

**The driver is the row's own Ω₀, and the threshold is sharp:**

| Ω₀ (Hz) | 0–0.3 | 0.3–1 | 1–3 | 3–10 | 10–30 | 30+ |
|---|---|---|---|---|---|---|
| live probes | 0.0 % | 0.0 % | 0.0 % | 14 % | 69 % | **98 %** |
| median peak/median PSD power | *nan* | 6202 | 2378 | 273350 | 143863 | 7516 |

**55 % of training rows have ZERO live probes** — they condition on the passive trace alone, so chi
mode is inert for them.

**And the second row of that table kills the obvious fix.** The natural guess is that low-Ω₀ draws
are non-oscillatory junk — `peak_freq` is an argmax and returns the bottom of a 1/f spectrum when
there is no peak — in which case screening the prior would be right. **Measured: they are genuine
oscillators**, spectral peaks thousands of times the median power. A 0.5 Hz bundle really does
oscillate at 0.5 Hz, and its χ at 0.015–0.15 Hz really is unmeasurable in ≤ 60 s. **The mask is
correct physics.** The prior spans ~4 decades of Ω₀ while the protocol — a band *relative* to Ω₀,
recordings ≤ 60 s — only reaches the fast end.

*(Separately: the 0–0.3 Hz bucket returns a NaN prominence, i.e. those traces are degenerate rather
than merely slow. ~16 % of rows. Worth its own look; it is not what C-6 turns on.)*

**So the options are not the three I first listed.** Corrected, with the constraint that makes one of
them cheap: probe frequencies are **already per-row** (`freqs = mults × f_peak` is `(B, K)`, each row
driven at its own frequency), and `gen_chi_raw` already accepts a `(B, K)` multiplier tensor. Only
the MULTIPLIERS are currently shared across the batch.

1. **Per-ROW multipliers, chosen so the probes resolve** — draw each row's multipliers conditional on
   its own `Ω₀ · T`. Costs **nothing**: same K simulations per batch, same force-tensor machinery,
   and the encoder already sees placement explicitly via `u` and `logcyc`. Rescues every row down to
   `Ω₀ ≥ 2/(0.3 · T)` ≈ 0.11 Hz at `T = 60 s`. **The trade is real and must be stated:** a slow row
   can then only be probed near the band's TOP, so it gets resolution but little frequency *spread* —
   and spread is what χ(ω) exists to measure. It converts inert rows into weakly-informative ones,
   not into good ones.
2. **Restrict the prior's Ω₀ range.** Defensible on science grounds — §4.3 records real per-cell
   resonance as 7.6–23.2 Hz, so the sub-1 Hz mass may be outside the regime of interest — but it
   changes the question the posterior answers and must be declared in the artifact.
3. **Accept it.** 45 % of rows carry chi; the network learns to use it when present. Correct but
   wasteful, and the waste lands on a multi-day run.
4. ~~Raise `T_MAX_EXP_S`~~ — rescuing Ω₀ = 0.1 Hz needs ~600 s recordings. Outside experimental
   reality and it would dominate the simulation budget.

### 4.3.5 C-6 part 1: per-ROW probe placement — implemented, and not sufficient alone

`chi.resolvable_multipliers` lifts each row's multipliers into the sub-band its own Ω₀ can resolve
(affine in log-space, so the stratified jitter's ordering and relative spacing survive), wired in at
`gen_chi_raw(adapt_placement=True)` — **the training path only**, since every other caller is
reproducing an experiment whose frequencies are already fixed. It costs nothing: probe frequencies
were always per-row (`freqs = mults × f_peak`), only the multipliers were shared.

| | before | after |
|---|---|---|
| live probes | 41.1 % | **46.1 %** |
| rows with ZERO live probes | 55.1 % | **47.9 %** |
| live-probe frequency span, median | — | **4.99×** (band spans 10×) |
| rows with exactly ONE live probe | — | 7.9 % |

The span row is the one that says the rescue is real rather than cosmetic: **live count is not the
objective.** A placement rule tuned on live count alone will happily park every probe on one
frequency — perfectly resolved, and measuring no shape at all. The audit now reports span for exactly
that reason, and it held (median 5× of the band's 10×, only 7.9 % single-probe rows).

**A rejected variant, recorded so it is not retried.** The obvious next move is to bound placement by
`CHI_MAX_CYCLES` as well as the floor, so no row needs the shared duration truncation. It collapses
the band: `hi/lo` = 10 and `CHI_MAX_CYCLES/CHI_MIN_CYCLES` = 10, so requiring both leaves a row a
single feasible multiplier for all but one value of `Ω₀·T`. The asymmetry that settles it — falling
under the floor MASKS a probe, exceeding the ceiling only makes it noisier — is recorded at the
function.

**What still bound after part 1: the SHARED lock-in duration** — one `N_k` per batch keyed on the
fastest row, which re-masked the slow rows placement had just rescued. That is **C-8**, below.

### 4.3.6 C-6 part 2 (C-8): per-ROW lock-in durations — the change that actually moved it

`chi.lock_in_batched` gained an `(B,) n_samples`: each row is integrated over its own prefix, by
MASKING rather than slicing so the tensor stays rectangular and the chunked float64 accumulation and
its memory bound are untouched. `gen_chi_raw` now computes an `(B,) N_row` from each row's own
frequency, and the resolution filter, the chi normalisation and **`logcyc`** all read that per-row
duration — `logcyc` especially, since it is the encoder's record of how much evidence a probe rests
on and must describe the integration that really happened.

| | original | + per-row multipliers (C-6) | + per-row durations (C-8) |
|---|---|---|---|
| live probes | 41.1 % | 46.1 % | **64.2 %** |
| rows with ZERO live probes | 55.1 % | 47.9 % | **32.6 %** |
| live-probe span, median | — | 4.99× | **5.38×** |
| rows with ONE live probe | — | 7.9 % | **1.4 %** |

By Ω₀: the 1–3 Hz band goes 4.4 % → **48.4 %** live, 0.3–1 Hz 0 % → 8.7 %. **Both quality measures
improved alongside the count** — the span went up and single-probe rows nearly vanished — so this is
not the count-for-shape trade the placement change had to be watched for.

Two invariants had to survive the masking, and both fail silently:

- **the mean is over each row's OWN prefix**, not the full width. Taking it over the full width
  subtracts a level the row's samples never had, and the residual lands at DC where a sub-resonance
  lock-in is most sensitive.
- **the mask is applied AFTER demeaning.** Zeroing first leaves `-mean` standing in every dead
  column — a step function at the prefix boundary, again with its energy at DC.

Pinned by `test_lock_in_per_row_durations_match_locking_each_row_alone`, whose reference is the only
unambiguous one available: lock each row in on its own, with no batching to get wrong. It also
asserts the rows are *decoupled* — changing one row's length must not move another's chi, which is
precisely what a shared mean or a shared sum would do. `n_samples=None` remains bit-identical to the
pre-C-8 code, which `test_lock_in_chunking_matches_full_batch` still pins.

**~33 % of rows are still inert**, and that residue is the genuinely unmeasurable tail: a 0.3 Hz
bundle cannot deliver two drive cycles at 0.03–0.3× its Ω₀ inside 60 s at any placement or duration.
Closing it further means changing the *prior*, not the estimator — §4.3.4 option 2.

**Confirmed end-to-end** by re-running the smoke train (all four stages, 2764 s). The masked fraction
differs by stage and the differences are the informative part:

| stage | masked | reading |
|---|---|---|
| posterior (training) | **36.8 %** | the fix working — and it matches `chi_mask_audit`'s 35.8 %, which validates the cheap audit as a proxy for the full chain |

> ⚠ **HOW PRECISE IS THIS NUMBER? Much less than it looks — corrected 2026-08-20.** A smoke train's
> training masked fraction has a run-level SD of **~6.1 pp**, not the ~1.8 pp a binomial on 704
> probes suggests: every row in a batch shares one `(t_scale, T)` stratum *and* one probe set, so
> 704 probes are **4 correlated strata** and the effective n is the batch count. Measured per-batch
> fractions span **13.5–57.3 %** (SD 12.2 pp over 12 batches). Seven corrected-band runs read
> 31.7 / 32.7 / 34.8 / 35.4 / 36.7 / 38.2 / 44.9 %, mean ~36 %. **So compare the MEAN of two or three
> runs against ~37 %; a single run within roughly ±12 pp is uninformative**, and `SEED` does not make
> runs comparable (torch is seeded, `np.random` is not — trap X8). If masking really does look
> moved, the decisive follow-up is a physics A/B on Ω₀ — the quantity masking depends on — which is
> measurable at n = 3000+ in about a minute, rather than another two-hour smoke train.
| validate (SBC calibration) | 47.2 % | same code path as training; 180 probes, so mostly small-sample spread |
| infer (PPC) | **79.9 %** | **unchanged, and correctly so** |

The PPC is the one to understand before reading it as a regression. It drives at the OBSERVATION's
absolute frequencies (`absolute_freqs=True`) against posterior *samples*, so it is deliberately never
adapted — moving those probes would simulate a different experiment than the one being checked. Its
masked fraction is therefore an indirect readout of **how well the posterior constrains Ω₀**: samples
whose Ω₀ is far from the observation's cannot resolve at the observation's frequencies. At this
smoke-train quality (4 batches, 5 epochs) the posterior is essentially the prior, so its samples span
the prior's ~4 decades and mostly cannot. **Expect this number to fall on a real run — and treat it
as a diagnostic of the posterior rather than of the probe machinery.**

### 4.4 What chi(ω) actually buys — measured

`scripts/degeneracy_map.py`, master cell, forced (`master_weak`) vs chi (`master_spont`, retargeted
band), same `T_obs`/M/seed. **Every parameter's unique handle improves:**

| param | forced | chi | | param | forced | chi |
|---|---|---|---|---|---|---|
| `k` | 0.040 | **0.102** | | `lam` | 0.261 | 0.359 |
| `x_scale` | 0.043 | **0.147** | | `delta_E` | 0.085 | 0.126 |
| `t_scale` | 0.224 | **0.501** | | `temp` | 0.218 | 0.310 |

Condition number 2300 → 2034. chi features earn top-5 sensitivity slots: `chi2_cos` is the **top**
feature for `lam` and `f_max`, `chi7/8/6_logmag` the top three for `t_scale`, all five for `f_scale`.

**But the two headline claims in the original rationale do not survive:**

- **`lam`~`t_scale` was never degenerate on this cell** — 0.59 in forced mode before chi does
  anything. The 0.96 in §4.2 was a property of the archived cell and box, not of the model.
- **`k`~`x_scale` survives chi essentially intact: 0.98 forced → 0.95 chi**, and `k`/`x_scale` still
  hold the two worst unique handles. Both are dominated by `A1_mean` — stiffness and displacement
  scale move the mean together, and a sub-resonance susceptibility does not separate them.

⚠ Caveat on the unique-handle table: chi swaps 11 Group-G features for 3K chi features, so part of a
gain can be dimensionality rather than information. The |cos| numbers are immune to that (extra
uninformative dimensions do not change a cosine), which is why the `k`~`x_scale` result is the one to
trust.

**Consequence: the rotation is no longer excluded under chi.** `build_posterior` used to set
`rotate = cfg.reparam_rotate and not cfg.chi_mode`, justified as "chi already attacks the degeneracy
the rotation targets". Measured false. `decorrelate.feats` now builds its Jacobian over whichever
feature set the mode conditions on (41 for spontaneous/forced, 41+3K for chi), seeding a second time
immediately before `gen_chi_block` — trap **X3**, without which the ±δ arms see different chi noise
and V is meaningless. Cost: a chi rotation pays (1+K) simulations per Fisher evaluation instead of 2,
so ≈ (K+1)/2 × a forced one. Pinned by
`tests/test_user_sbi.py::test_chi_fisher_rotation_builds_over_the_chi_feature_set`, which fails both
if `gen_chi_block` is never called and if `J` is allocated at 41 rows.

### 4.4.1 Step 3 re-measured on the set encoder (2026-08-08) — and what the payload table says

**Measured 2026-08-08**, `TOBS_S=4.5 SEED=0 M=32 M_NOISE=128`, forced (`master_weak`) vs chi
(`master_spont`, K=6), `master.txt` bounds, after fixing trap **CHI10** in the script. Ω₀ measured
22.78 Hz. Logs: `Resources/Plots/degeneracy_{forced,chi}_T4.5.log`; data: `degeneracy_map_*_T4.5.npz`.
**The chi arm was re-run on 2026-08-10 under the post-C-9/C-10 Fisher set** (3 channels per probe,
48 feature rows instead of 54), so these numbers describe the feature set the retrain's rotation will
actually use. The forced arm is unaffected — it has no chi features — and was not re-run.

> **Removing the duplicates IMPROVED the result, which is the tell that they were redundant rather
> than informative.** `t_scale` went 0.371 → **0.447** and `lam` 0.411 → **0.472** when the four
> duplicate `logcyc` rows left, because a unique-handle score is a residual against the span of the
> other columns and near-duplicate rows inflate that span. `k`, `x_scale`, `delta_E` and `f_scale`
> moved by ≤0.006 and the condition number by 1. Redundant rows were flattering nothing and
> obscuring `t_scale`, the parameter chi is most supposed to help.

> **Outputs are suffixed by mode AND `T_obs`, and the second half was learned the hard way.** The
> forced control at 1.0 s below shares a mode with the primary run, so under a mode-only suffix it
> silently overwrote the very result it exists to be compared against — the same defect the mode
> suffix was added to fix, on an axis added later. Compare only equal-`T` runs.

**The §4.4 baseline reproduces, so the set encoder lost nothing.** This is the question step 3
existed to answer, and it is answered:

| param | forced | chi | Δ | §4.4 forced | §4.4 chi |
|---|---|---|---|---|---|
| `k` | 0.040 | **0.091** | +0.051 | 0.040 | 0.102 |
| `x_scale` | 0.043 | **0.125** | +0.082 | 0.043 | 0.147 |
| `t_scale` | 0.219 | **0.447** | +0.228 | 0.224 | 0.501 |
| `lam` | 0.278 | **0.472** | +0.194 | 0.261 | 0.359 |
| `delta_E` | 0.091 | 0.102 | +0.011 | 0.085 | 0.126 |
| `f_scale` | 0.871 | 0.695 | −0.176 | — | — |

`k`~`x_scale`: **0.98 forced → 0.97 chi** (§4.4: 0.98 → 0.95). Condition number 2212 → 2092
(§4.4: 2300 → 2034). Different `T_obs`, different band, different K, a rebuilt encoder and
C-4/C-6/C-8 in between — and the numbers land on top of each other.

> **`f_scale`'s unique handle FALLING is the largest gain in the table, not a loss.** Unique-handle is
> a *ratio*: a parameter with no gradient is trivially orthogonal to everything. Forced `‖g‖_std` for
> `f_scale` is **0.018** — it is not measured at all at this weak drive, so its 0.871 is noise being
> unique. Under chi `‖g‖_std` is **3.789, a 213× increase**, and it becomes a real, partially
> correlated handle. Read `‖g‖` beside `unique` or this row reads backwards.

**THE PAYLOAD — which features break which alias.** This table was contaminated in every previous
run (CHI10), so this is the first time it can be read at all:

| param | what carries it under chi |
|---|---|
| `f_scale` | **all five slots are chi**: `chi1_cos`, `chi2_sin`, `chi0_sin`, `chi2_logmag`, `chi0_logmag` |
| `t_scale` | `chi4_logmag` +9.65, `chi3_logmag` +7.81, `chi5_logmag` +6.54 — **three** of the top five, the top two above `A3_log_fpeak` (7.60) |
| `lam` | **`chi2_sin` is the TOP feature** (+7.89), `chi0_sin` fifth |
| `k` | `A1_mean` **+291**, then `A2_log_var` −124; chi_logmag only reaches −46 |
| `x_scale` | `A1_mean` **−3.72**, `A2_log_var` +1.89; chi_logmag only reaches +0.51 |

So the mechanism is specific, and it splits cleanly. **chi's PHASE channels (`cos`/`sin`) carry `lam`
and `f_scale`; its MAGNITUDE channel carries `t_scale`.** And `k`~`x_scale` survives for exactly the
reason §4.4 gave: both are led by `A1_mean` by a factor of 6, and a sub-resonance susceptibility does
not touch the mean. **Three independent measurements now agree that chi does not break that pair.**
Stop expecting it to.

**A control that partly deflates the above — read it before quoting the deltas.** The forced arm's
Group-G lock-in has no `CHI_MAX_CYCLES` counterpart, so at `T_obs = 4.5 s` it runs **142 drive
cycles**, far past the ~31-cycle non-stationarity wall of §4.3.1. That inflates Group-G `fnoise` and
deflates forced's gradients — biasing *in favour* of chi. Re-running forced at `T_obs = 1.0 s`
(31.6 cycles) measures it:

| | forced @1.0 s | forced @4.5 s | chi @4.5 s |
|---|---|---|---|
| `k` | 0.076 | 0.040 | 0.091 |
| `x_scale` | 0.082 | 0.043 | 0.125 |
| `t_scale` | 0.206 | 0.219 | 0.447 |
| `lam` | 0.270 | 0.278 | 0.472 |
| condition number | 1669 | 2212 | 2092 |

**Two conclusions, opposite in sign.**

1. **`k` and `x_scale`'s gains do not survive intact.** Against forced @1.0 s they shrink roughly by
   half (+0.051 → +0.015, +0.082 → +0.043). The forced arm's own `T_obs` sensitivity is the same size
   as the chi gain for these two, so the honest claim is "chi helps `k`/`x_scale` a little, within the
   noise of how long you record", not the +0.05/+0.08 the matched pair suggests.
2. **`t_scale`, `lam` and `f_scale` are untouched by it** (+0.241, +0.202, and 180× on `‖g‖`
   respectively against the 1.0 s arm). Those three gains are real and are chi's actual product.

**And the condition-number claim should be retired.** §4.4 reported 2300 → 2034 as a chi improvement.
The `T_obs` effect alone moves it 1669 → 2212 — *larger, and in the other direction*. That number was
never measuring chi. The `|cos|` pairs and the payload table are what survive this control; the
scalar summaries are not.

**Verdict for the retrain: GO on the information question.** chi buys real, mechanism-identified
information on `f_scale` (from nothing to measurable), `t_scale` and `lam`. It does not buy
`k`~`x_scale`, and no amount of retraining will change that. The blocker is elsewhere — see the two
`decorrelate` items in §6.

### 4.5 Capability matrix

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

### 4.6 The 2026-08-25 retrain (`posterior_08232026`) — what it actually measured

> **What to DO about it is §11.** This section is the characterisation; §11 is the programme, and
> §11.2.1 reconciles the two where they appear to disagree.

Trained on the corrected χ band (`chi_f0 = 0.15`, `(0.03, 0.3) × Ω₀`, `CHI_MAX_CYCLES = 20`,
K=6 supplied into `chi_k_pad = 12`). Analysed with **`scripts/posterior_identifiability.py`**, which
reads the `.pt` + `.rot.pt` and simulates nothing.

**Calibration: excellent, and it is not the problem.** SBC rank histograms flat on all 13; SBC CDFs
on the diagonal; TARP on the diagonal. The posterior is honest about what it does not know.

**Informativeness: poor, and it is the problem.** PPC coverage **99.1 %** at nominal 90 %,
mean |z| 0.490, max |z| 1.893, 1/114 outside. Over-dispersed: the predictions land comfortably inside
intervals that are far too wide. The eye-test 95 % band is 2–3× the observation's envelope.

> **⚠ THE LOSS CURVE RULES OUT EVERY CHEAP FIX, AND IT IS THE MOST IMPORTANT PLOT OF THE RUN.**
> Train and validation loss track each other tightly (no overfitting) and plateau from ~epoch 90;
> best validation −9.6814 at epoch 110, stopped at 130 on 20-epoch patience. `plot_training_loss`'s
> own decision rule: a clean plateau well before the best epoch means the wide marginals are a
> **data/identifiability limit, not under-fit**. **More epochs, more flow capacity and more training
> simulations will not fix this.** Do not spend another multi-day run on any of them.

#### What it measured, by direction

`V`'s columns are the prior-averaged Fisher eigenvectors, best-constrained first:

| | direction | dominant loadings |
|---|---|---|
| best | 0 | −0.67·`lam` +0.54·`tau_c` −0.34·`beta` |
| | 4 | −0.82·`t_scale` +0.55·`f_scale` |
| | 8 | −0.86·`x_scale` +0.49·`f_scale` |
| | 11 | **−1.00·`k`** (everything else ≤0.03) |
| worst | 12 | −0.79·`delta_E` +0.45·`temp` −0.35·`n` |

Per-parameter share of identifiability in the worst three directions: `k` **0.999**, `delta_E` 0.767,
`s` 0.409, `n` 0.337, `temp` 0.303 — against `beta` 0.130, `tau_c` 0.053, `lam` 0.001, and `tau` /
`f_max` / the rescale triple at 0.000. Well measured (top-4 weight): `beta` 0.818, `tau` 0.714,
`f_max` 0.619, `lam` 0.584. **`lam` leading the single best direction is exactly what §4.4.1
predicted chi would buy.**

> ### ⚠ THIS REVISES §4.4/§4.4.1's FRAMING OF THE CENTRAL DEGENERACY
>
> Those sections record the problem as the **`k`~`x_scale` alias** (|cos| 0.97, measured three times).
> The eigenbasis says something sharper: **`k` and `x_scale` are not entangled here at all** — their
> shared weight across all 13 directions is **0.0002**. `k` puts **99.9 %** of its weight on one
> direction loading −0.999·`k`, alone; `x_scale` sits separately at direction 8.
>
> Both are true. `degeneracy_map`'s |cos| compares gradient **directions**; a Fisher eigenvalue is
> about gradient **magnitude**. `k`'s gradient is *both* nearly parallel to `x_scale`'s *and tiny*
> (unique handle 0.091). §4.4.1 already warns about this trap in the other direction — *"a parameter
> with no gradient is trivially orthogonal to everything… read ‖g‖ beside `unique`"*.
>
> **So `k` is not degenerate so much as UNMEASURED, and that changes the remedy.** You cannot rotate,
> reparameterise or re-prior your way out of a flat direction — only a new observable reaches it. The
> drive here is **weak and sub-resonance**, which is close to the worst case for stiffness, since
> stiffness dominates the response *at and above* resonance. A drive nearer Ω₀, a stronger drive, or
> a step/force-clamp protocol are the candidates. `scripts/posterior_identifiability.py` flags this
> class automatically ("parameters that are their OWN near-null direction").

#### The conditioning was mostly padding

`input_dim` 42 + `forcing_dim` 72 = 114, where 72 = 12 slots × 6. **Only 6 probes were supplied into
12 slots**, so 36 elements are pure padding before anything is masked; the PPC's 48/114 zero-variance
count decomposes to roughly **4 live probes out of 12**. `analysis.invalid_breakdown` now prints this
split (`[ppc] chi probes: N live / K slots (…never filled, …masked)`) instead of one alarming
fraction. **Raising K, or lowering `chi_k_pad` to what is actually supplied, is a cheap and untested
lever** — untested because nothing measured it until now.

> **⚠ THE SCALE IS UNKNOWN FOR THIS RUN.** The Fisher **eigenvalues were not stored** with `V`, so
> everything above is an *ordering and loadings* analysis: direction 12 is worst and `k` is alone at
> the bottom, but whether direction 0 is 10× or 10⁶× better constrained is not recoverable without a
> full Fisher re-run. That gap is the difference between "mildly uneven" and "nine of thirteen
> parameters are prior". **Fixed for every future run** — `save_posterior_artifacts` now writes
> `fisher_eigenvalues`, and `build_posterior` prints the spread as it computes the rotation.

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
- **G6. SAVE A FIGURE WITH `visualizers.save_figure`, NEVER `fig.savefig`.** matplotlib bakes artist
  colours at **build** time but reads `savefig.facecolor` at **save** time, and
  `gui/mpl_theme.apply_mpl_theme` rewrites rcParams **globally** on every appearance change. Anything
  that changes theme between the two splits one figure across two palettes — and a multi-day run
  straddles a flip easily (a manual toggle, or Windows switching light/dark on its own schedule).
  **Measured on the 2026-08-25 retrain: dark axes (#2B2B2B) on a light page (#F3F3F3) with
  `text.color` still light — every tick label, axis label, title and table cell at a contrast ratio of
  1.06:1. 14 of 15 figures unreadable, and the parameter tables carrying the best-fit numbers were
  blank.** `save_figure` pins the save to `fig.get_facecolor()`, i.e. the theme the artists were drawn
  in. Related: a matplotlib table cell set to `facecolor="none"` shows the FIGURE background, so
  `_param_table` now pins cells to `axes.facecolor`, which is the surface `text.color` is chosen
  against. Pinned by `test_a_theme_flip_between_build_and_save_cannot_split_a_figure` (which carries
  its own counterfactual) and `test_the_parameter_table_is_readable_against_its_own_cells`.
- **G7. `torch.quantile` propagates a single non-finite entry across the WHOLE reduction, and refuses
  an input over 2**24 elements.** Both bite the posterior-predictive diagnostics, which are
  (draws × features) and routinely clear both thresholds. One divergent draw in a thousand turned the
  power-spectrum figure into a bare observation line with no band, no median and no message — while
  the two sibling figures survived (the overlay band takes only the 50 best draws; `cycle_average`
  confines the damage to one phase bin), so it read as a plotting bug in one figure rather than as a
  property of the posterior. `overlay.psd_band` now drops non-finite rows **and returns the count**,
  `overlay.cycle_average` masks non-finite samples before binning, and `_column_quantile` chunks
  around the size limit. **Report the drop, do not just survive it:** "the band is missing" is a bug
  report, "17 of 1000 draws were non-finite" is a finding about the posterior.

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
- **P7. The overall bar tracks the live row with the LARGEST total, among those whose total > 1.**
  Not the outermost (the pos-0 "Training neural posterior" bar wraps `range(1)` and would read 0%
  for the whole build), and not a sticky first-seen driver (sbi's NN training emits no tqdm bar at
  all, only a printed epoch counter). ⚠ **It was DEEPEST until 2026-08-21** and that was one config
  flip from being wrong: `Running time segments` wraps segs in {1,2,3}, so it is both deeper and far
  coarser than `Generating training data` and would sweep the bar 0→100 % every couple of seconds —
  `QUIET_SEGMENT_BAR` was papering over the election rule. On every nest actually observed the two
  rules agree; the test that separates them is
  `test_the_overall_bar_follows_the_largest_non_degenerate_total` (verified non-vacuous: deepest
  gives 66 %, largest-total gives 38 %).
- **P9. There is exactly ONE bar and ONE caption on screen, and the caption is load-bearing.** The
  full nest is still parsed; it is rendered as one bar plus one line. When nothing reports a
  percentage the caption falls back to the deepest row whose total is not 1 — which during sbi's
  neural-network training is its printed epoch counter, and that is *hours* of a multi-day build.
  Delete that fallback and the longest phase of the run becomes a blank indeterminate bar.
- **P8. `close(leave=True)` at pos 0 paints the final frame then writes a bare `"\n"`** — byte-identical
  to a moveto(+1). Treat it naively and the finished bar stays in the pane at 100% and pegs the overall
  bar while the pipeline is still working. `StreamRouter._settle_closing()` resolves it on later evidence.

### S — Solver performance meter (`widgets/progress_pane.py`)

The overall bar moves slowly, so the GUI can read as frozen. The thing that *is* moving is the SDE
solver, and its it/s is the number the user wants.

> ### ⚠ S0 — THE METER DOES NOT READ THE SOLVER BAR. DO NOT MAKE IT DO SO AGAIN.
>
> It did, until 2026-08-21: the rate was parsed out of the *rendered text* of the solver's tqdm bar.
> That has exactly one failure mode and it fired — tqdm paints a frame carrying a rate only once
> `mininterval` has elapsed, so a solver call **shorter than its own bar's mininterval** renders
> `?it/s` and nothing else, forever. CUDA graphs took a 100 k-step call from ~10 s to ~0.7 s and the
> meter read `— (idle)` for entire runs. **A 10× speedup made a progress bar too fast to render.**
>
> The rate now comes from **`core.progress.SOLVER`**, a monotonic step count the solver publishes and
> the pane differences on its existing 100 ms tick. That is correct at any speed, including the next
> speedup. Anything that reintroduces a dependency on rendered bar text reintroduces this bug.

- **S1. The solver bar is still found by its DESC PREFIX** (`config.SOLVER_BAR_DESC` →
  `"step (batch="`), **never by its row** (its tqdm `pos` is 0, 1 or 2 depending on phase and panel)
  — but now for one purpose only: **exclusion**. See S3.
- **S2. The solver bar must never be rendered.** Nothing is rendered as a row any more (P9), but a
  posterior build constructs 10k–30k of these bars, one per time segment, so this stays the reason
  any future "show me the bars" feature must special-case it.
- **S3. The solver bar must be EXCLUDED from `_retarget()`**, and this is *more* load-bearing than it
  was: the election is now by largest total (P7) and the solver's is in the tens of thousands, so it
  would win outright and drag the overall bar through a full 0→100 % sweep every second.
- **S4. `vt.parse_rate` still inverts tqdm's `s/it`** (`" 2.50s/it"` is 0.4 it/s, not 2.5 — read
  naively a crawling bar reports as a fast one). It is **no longer load-bearing**: nothing acts on
  `RowState.rate`. It is kept because `parse_bar`'s contract is a faithful parse of a frame and the
  inversion is knowledge that cost a bug — not because anything reads it.
- **S5. The solver bar stays ENABLED under the GUI**, deliberately, unlike the segment bar that
  `config.QUIET_SEGMENT_BAR` silences — even though silencing it would delete the desc seam entirely.
  Two things depend on it still writing: the cooperative **cancel**, whose checkpoint is
  `_SignalStream.write()` and for which this is by far the most frequent writer (the next-most is the
  per-batch `Generating training data` redraw, ~2 s at production geometry and longer at large
  `T_obs`); and the **stall detector**, which fires on no output of any kind. Cost of keeping it is
  ~600 chars per 0.7 s call. *(The pane's `_sample_solver` also heartbeats on a moving counter, so
  the stall side is now belt-and-braces — the cancel side is not.)*

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
- **Q4. A PERSISTED SCIENCE CONSTANT OUTLIVES THE CONFIG CHANGE THAT RETIRED IT — this cost a 5-day
  retrain.** `inference_tabs` seeds the χ widgets from `config.CHI_*` (lines ~297-299) and `_restore`
  then overwrites them from QSettings (~465-469). That is correct for a *preference*; it is a trap
  for a **measurement definition**. When C-5 moved `CHI_FREQ_BOUNDS` from `(0.1, 10.0)` to
  `(0.03, 0.3)` on 2026-08-06, every machine with a saved value silently kept training at the retired
  band, and nothing anywhere compared the two. The 2026-08-19 retrain is the bill (Appendix A).
  **The general rule: if a persisted value defines what is measured rather than how it is displayed,
  something must compare it against the module default before the spend.**
  `orchestrator._assert_chi_config_is_deliberate` now does exactly that for the band and F₀, on both
  `build_prior` and `build_posterior`, with `PRISM_CHI_OVERRIDE=1` as the deliberate escape hatch.
  ⚠ It gates the band and drive only — **`chi_n_freqs` is NOT gated**, because K is the count an
  *observation* supplies and training draws its own per batch; failing on it would refuse a
  legitimate 7-recording experiment and break the K-agnosticism the set encoder exists to provide.

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

### CHI — the chi(ω) probe-set layout (2026-08-06)

- **CHI11. NEVER WINSORISE, RESCALE OR CLIP THE χ BLOCK.** A padded probe slot is **exactly 0.0 in
  all six channels** and is required to stay BITWISE inert (§3.6, pinned by `test_chi_set_encoder`).
  Under χ the mask column is ~28 % zeros so its 0.1th percentile is 0.0 and a clip there is a no-op —
  but `logmag`, `cos` and `sin` are dense over live probes, so THEIR 0.1th percentile is non-zero and
  clipping would push every pad off 0.0 and turn it into a phantom probe. That is the exact defect
  the packer's `nan_to_num` removal fixed once already. `pipeline.winsorize_summary_block` therefore
  takes the summary width as an argument and touches nothing past it.
- **CHI12. A NEW CONDITIONING CHANNEL REACHES THE FISHER TOO.** `decorrelate.feats` builds its
  Jacobian by calling `gen_stats`, so anything added there is silently added to `J` — and
  `fnoise = max(f0.std(0), 1e-9)` is a DENOMINATOR. A **binary** channel is the worst case: constant
  across most operating points (harmless, 0/1e-9 = 0) and then stepping by 1 between the ±δz arms at
  some others (~1e9 into J, intermittently). Same class as CHI10 and C-9/C-10. The valid flags go
  through **`pipeline.gen_stats_features`**, which is the ONE name for "features, not conditioning" —
  `decorrelate` and all five Jacobian-building scripts (`degeneracy_map`, `diagnose_fmax`,
  `feature_candidate_test`, `identifiability_offgt`, `reparam_selftest`) call it. **If you add a
  channel to `gen_stats`, every caller of `gen_stats` proper gets it**; decide deliberately which of
  those is a conditioning vector (wants it) and which is a Jacobian (does not). The tell is whether
  the result is `cat`-ed with `log_T`.

Every one of these is silent. The suite that catches them is `tests/test_chi_set_encoder.py`, which
runs in seconds and needs no simulation — **run it first when touching anything chi.**

- **CHI1. THREE feature sets, and conflating them is invisible.** `CHI_COND_CHANNELS` (6/probe,
  padded — the network), `CHI_FISHER_CHANNELS` (4/probe, no pad — `decorrelate.feats` and
  `degeneracy_map`), and the diagnostic labels (the Fisher set minus Group G). `u` and `mask` are
  absent from the Fisher set ON PURPOSE: under a deterministic grid `u` is theta-independent, its
  float32 std is ~2.5e-8, and `fnoise = max(std, 1e-9)` does not protect — the central difference then
  writes entries of order 1 into `J` while `V` stays orthogonal and the tests pass.
  ⚠ *Two corrections, both from measuring it (2026-08-08).* The std is **25× ABOVE** the 1e-9 clamp,
  not below it — the clamp is irrelevant here and any guard keyed on it will miss this; the test has
  to be **relative** to the channel's own magnitude. And this warning was **not enough**:
  `degeneracy_map` carried it in a comment and violated it six lines later for three commits. See
  **CHI10**.
- **CHI2. The Fisher must pass `resolution_filter=False`.** The filter depends on `f_peak`, hence on
  theta, so a probe can CROSS the threshold between the ±dz arms. A mask step of 1 over a 1e-9 floor
  puts ~1e9 into the Jacobian and `V` becomes that discontinuity.
- **CHI3. Never `z_score_x="independent"` under chi.** A per-COLUMN affine over probe slots is
  permutation-BREAKING, and the near-constant mask column becomes a ~1e7 amplifier under sbi's 1e-7
  min-std clamp. The flag is derived from `EmbeddedNet.owns_standardization`, never passed separately,
  so the standardizer and the encoder cannot be configured apart.
- **CHI4. The mask gate must be applied BOTH sides of φ.** `phi(0) != 0` for a biased MLP, so a
  pre-gate alone lets dead slots contribute — and the symptom is an embedding that drifts with the
  PAD WIDTH, which looks like anything but a missing multiply.
- **CHI5. No max pool, and no BatchNorm.** `E[max of n]` grows with n, so a masked max writes a
  probe-count-dependent location shift into every channel. BatchNorm breaks because sbi's `get_numel`
  runs the net on one CPU row at build time and single-observation inference must match the batch.
- **CHI6. Width can never identify a chi layout.** `6·K_PAD = 30` at `K_PAD = 5` collides exactly with
  the retired layout-1 `3·K` at `K = 10`. The sidecar's `chi_layout` is the only safe gate, and it is
  checked BEFORE `posterior_mode`'s decode, keyed on the SIDECAR's mode — otherwise a forced posterior
  loaded against a chi config is told it is "chi layout 1" and the message names the wrong problem.
- **CHI7. `CHI_K_PAD` is frozen into every artifact** (sbi bakes `condition_shape` into the saved
  posterior). The load path turns a change into a message; without it, a later bump surfaces as a
  shape assert hours into a run. The encoder's parameter count does NOT depend on it — a bigger pad
  costs only input columns, so choose generously once.
- **CHI9. A LONGER lock-in is not a better one.** Every instinct about integration says more samples
  means less variance, and for chi(ω) on this model that is **false above ~31 drive cycles**: |chi|
  CV goes 0.03 → 0.63 and driven/undriven SNR 26 → 2.3, and shortening the window on the *same trace*
  reverses it (§4.3.1). `config.CHI_MAX_CYCLES` now bounds it, but the trap survives the fix in three
  forms. First, it makes a longer recording look like a *worse* probe frequency — which is how the
  2026-08-05 band was mis-derived from a single `T_obs` slice, so when reasoning about a chi failure
  always ask for the CYCLE COUNT before the frequency. Second, the ceiling is applied in
  `gen_chi_raw` **on purpose**: it is the definition of the measurement, not a policy of one caller,
  and "simplifying" it up into `gen_training_data` would leave the Fisher, the PPC and the
  experimental path measuring something the network was never trained on, silently. Third, it is
  applied **PER ROW** (`(B,) N_row`, since C-8 / §4.3.6) — it *used* to be one scalar keyed on the
  batch's highest `f_peak`, which truncated the slow rows to a fraction of a cycle and masked them;
  if you find a scalar `T_k` anywhere in this path, it is a regression.
  `CHI_MIN_CYCLES` guards the floor, `CHI_MAX_CYCLES` the ceiling, and `SimConfig.__post_init__`
  rejects a config where they cross (which would mask every probe in every observation).
- **CHI8. Hz → cell frequency is `cfg.freq_si_to_cell`, NOT `get_unit_conversion_factor("Hz")`,**
  which returns 1.0 against an `ms` cell. A 1000× error lands as a wildly off-resonance but
  numerically valid chi.
- **CHI10. A standardized Jacobian DIVIDES by an ensemble std, so any pinned channel is an
  amplifier — and there are three ways to pin one.** `J = ΔF / (2·dz) / max(std, 1e-9)`
  (`decorrelate.fisher_at`, `degeneracy_map`). A channel that does not vary with theta contributes
  rounding-over-rounding, which is **order 1 to 10⁴**, not order zero — it then leads `‖g‖`, the
  `|cos|` matrix, the SVD and the top-features table with nothing in the numbers marking it.
  Measured 2026-08-08 on `master_spont`, `T_obs = 4.5 s`:

  | how it pins | channel | ensemble std | what it did |
  |---|---|---|---|
  | theta-independent by construction | `u` (CHI1) | 2.5e-8 (ratio 1e-8) | `[:2]` unpack fed it in as `logcyc`; **wrong sign**, log(0.03)…log(0.30) against a correct +1.12…+3.00 |
  | `CHI_MAX_CYCLES` ceiling binds | `logcyc` of a pinned probe | 8e-5 (ratio **2.7e-5**) | `max\|J\|` = **2.0e4** against 289 for the largest real feature; `sigma[0]` and a condition number of **56 000**, alone |
  | duplicates an existing row | `logcyc` of an unpinned probe | healthy | `corr(chi_j_logcyc, A3_log_fpeak) = +1.000000`, elementwise ratio exactly 1.0 |

  > ✅ **RESOLVED at the source, 2026-08-10 (C-9/C-10).** `logcyc` left `CHI_FISHER_CHANNELS`
  > (now `("logmag", "cos", "sin")`, 3K) and `fisher_features` takes **one argument**, so rows 1 and 3
  > of that table cannot occur and row 2's mis-wiring is a `TypeError`. What survives is the
  > *principle*, which is why this trap is not deleted: **`fnoise` is a denominator, so a channel that
  > barely varies is an amplifier.** Note the asymmetry that hid it — an *exactly* constant channel is
  > harmless (`0/1e-9 = 0`, which is why chi mode's 11 zeroed Group-G columns cost nothing); it is the
  > *nearly* constant one that is lethal. Apply that test to any channel added to any Fisher here.

  **Each needs a different detector, and one detector cannot cover two.** The relative-std test
  (`std <= 1e-6·|feat|`) catches the first and **provably misses the second** — 2.7e-5 is 27× above
  it, and raising the constant until it catches is threshold-guessing that would start eating quiet
  real features. The second is caught by **arithmetic instead**: `mult·Ω₀·T > CHI_MAX_CYCLES` says
  which probes were capped, deterministically, before any statistics. The third is caught by neither
  and needs no fix in the script (a duplicate row is redundant, not wrong) but does over-weight that
  direction in `Jᵀ J` by K× — see §6.
  > **Why `logcyc` pins, stated once:** `logcyc = log(mult_j) + log(f_peak) + log(T)` when the
  > ceiling is clear — so `log(mult_j)` is a constant offset that vanishes under standardization and
  > every unpinned probe's row is **exactly** `A3_log_fpeak`'s. When the ceiling binds,
  > `freq·T_row → CHI_MAX_CYCLES` and all that is left is the sawtooth of `floor()`. `logcyc` is
  > therefore *either* a duplicate *or* quantization, never independent information — which is the
  > opposite of the justification in `chi.fisher_features`' docstring ("it genuinely varies with
  > theta"). True, and insufficient: it varies *exactly as a row already in the set does*.
  >
  > That reasoning is about the FISHER set only. In the **conditioning** block `logcyc` is genuinely
  > informative, because training varies placement AND duration per row (C-6/C-8) — so do not
  > "simplify" this into a claim about `CHI_COND_CHANNELS`.

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
- **X3. `gen_chi_raw` runs K UNSEEDED simulations.** Any common-random-number scheme around it must
  seed *immediately before the chi block as well*, not only before the spontaneous run — otherwise
  the ±δ arms of a central difference see different chi noise and the derivative is swamped. The
  result looks plausible and means nothing. See `scripts/degeneracy_map.py` and
  `SBI/decorrelate.feats`. *Corollary since the set layout:* the training probe sampler draws from a
  DEDICATED `torch.Generator`, never the global stream — a placement drawn globally would be
  re-randomised (or frozen) by exactly these `manual_seed` calls.
- **X4. Torch's `SigmoidTransform._inverse` clamps to `[tiny, 1-eps]`.** Do not write code (or bug
  reports) premised on the box round-trip producing `±inf`; it cannot on torch 2.9. Pinned by
  `test_box_roundtrip_never_yields_a_nonfinite_latent_target`, which exists precisely so a version
  bump that removes that clamp surfaces as a test failure rather than a dead training run.
- **X13. THE RECOVERY PATH CAN BE THE THING THAT KILLS YOU, and Python will not tell you what it was
  recovering from.** On 2026-08-27 the retrain caught a genuine OOM at batch 351/5000, entered
  `_rows_with_oom_retry`'s handler correctly, and then died on `torch.cuda.empty_cache()` — which
  raised a raw `AcceleratorError: CUDA error: out of memory` of its **own**. Two things made that
  worse than it had to be. The release deliberately sits **outside** the `except` block (while the
  caught error is bound, its traceback pins the failed attempt's tensors and a release frees
  nothing), and Python **clears the exception context when an except clause exits** — so the
  secondary failure arrived with `__context__` of `None` and the traceback contained *nothing* about
  the OOM. The stderr notice naming the original was printed *after* the release, so it never ran.
  The symptom is a bare traceback ending in `empty_cache`, which reads as "empty_cache is broken"
  rather than "the card is full". Both retries now print the notice **before** releasing, and all
  releases go through `pipeline._release_device_memory`, which never raises. **If you add a release
  anywhere, it is best-effort by definition** — its caller is already recovering.
  **ROUND 2 (2026-08-28), and it fired again 3639 batches later.** The retry added to fix the above
  died on `_tc.rng_restore` → `torch.cuda.set_rng_state_all`, which copies each generator's state
  into *device* memory and is therefore an allocation. Two lessons on top of the first:
  **(a) a recovery block must not release and then sleep.** The order was release → wait 15 s →
  restore; the release hands every cached block back to the driver and the wait then sleeps on a
  contended card while the desktop takes it, so the few KB the restore needed was the one request
  the driver could not serve. We donated the working set and asked for it back. Restoring *first*
  serves it from the allocator's own cache, because the failed attempt's blocks are still sitting
  there. **(b) guard the class, not the instance** — this was the second time the newly-guarded step
  simply moved the failure to its neighbour. An audit of every recovery window found five more:
  `_log_memory`'s four device calls (a *diagnostic* able to kill a 5000-batch run),
  `config.memory_budget_elements`' bare `mem_get_info` (re-entered by every retry when it re-plans),
  the per-batch and cadence/final `rng_snapshot`s, and `str(e).splitlines()[0]` — an `IndexError` on
  a zero-message exception, inside the very function documented never to raise.
- **X7. An UNWRAPPED CUDA OOM rules out the solver — and that is the fastest thing to check.**
  `Simulator.__sols` wraps everything raised inside `euler`/`euler_compiled` into a `SimulationError`
  reading `"<Model> <method> failed after N steps (batch=..., segs=..., device=..., dtype=...)"`. So a
  bare `CUDA error: out of memory` did NOT come from the SDE integration, which also means
  `_gen_obs_retry` (which wraps only `_gen_obs_one`) never saw it. Both retrains died this way and the
  distinction is what separated them: 2026-08-10 was wrapped (solver), 2026-08-11 was not (the batch's
  own tensors). **Read the prefix before theorising.** Everything in a chi batch outside the solver is
  linear in ROWS, not in the solver's batch — the zero-drive tensor, the per-probe force rebuilt K
  times, `idx_c`, `x_spont_dim`, `gen_stats` — which is why the outer retry halves rows rather than
  simulation width. And note `sbi`'s `append_simulations` is outside *both* retries: it is one
  indivisible ≥13 GiB allocation that no halving can rescue, which is why it is defused rather than
  retried.
- **X8. `gen_training_data` is NOT reproducible from a torch seed alone.** The initial conditions come
  from **numpy's** global RNG (`np.random.randint`), which `torch.manual_seed` does not touch —
  measured, two runs of identical code differed by max|diff| 30.6 until `np.random.seed` was set too.
  Any claim that "a seeded run reproduces" needs both, and any bit-identity check of this function is
  meaningless without them.
- **X6. `torch.cuda.mem_get_info()` OVERSTATES free VRAM on Windows — by the size of the desktop.**
  Measured on the 16 GB RTX 5070 Ti: `mem_get_info` said **15037 MiB** free while `nvidia-smi` said
  **5814 MiB**, at the same instant. Under WDDM the OS virtualises VRAM, so other processes' surfaces
  are *evictable* and get reported to you as free. Anything that sizes a batch from that number is
  planning against memory it does not have, and the failure it produces is a **raw driver**
  `AcceleratorError: CUDA error: out of memory` (the driver lost an eviction race) rather than
  `torch.OutOfMemoryError` — so a handler that only knows the latter will not catch it. This is what
  killed the first chi retrain, hours in. `config.memory_budget_elements` is therefore a HINT:
  `pipeline._BUDGET_CAP_ELEMENTS` learns the real ceiling from OOMs and `_gen_obs_retry` is what
  actually recovers. **When measuring headroom by hand, read `nvidia-smi`, never `mem_get_info`.**
  ⚠ And the corollary measured on 2026-08-27: because WDDM also provides a **shared-memory spillover
  pool** (half of system RAM — ~31.7 GiB here), a batch that does not fit usually does **not** fail.
  It pages, at up to a 9× wall-clock penalty; 21.67 GiB completed on this 15.92 GiB card. So a
  *slow* run is the symptom to watch for, not an OOM, and the OOMs that do happen are transients
  rather than sizing errors. See the 2026-08-27 (later) entry.
  The learned cap is REACTIVE, though — it cannot know anything until something has already failed,
  which on a shared card can be hours in. `config.SIM_VRAM_CEILING_GIB` (0 = off) is the proactive
  half: state the real budget up front on a day you know the desktop will be busy, and the planner
  splits from batch 0. It is deliberately **not** part of the checkpoint identity — it changes the
  memory plan, not the rows.
- **X5. `CAL_N_SCALES` is `t_scale`'s effective SBC sample size.** Every row in a calibration batch is
  *assigned* that batch's `t_scale`, so their ranks are not independent. Lowering the pair count to
  buy wall-clock is a different measurement, and it damages precisely the parameter chi(ω) exists to
  separate. Batch SIZE is nearly free; batch COUNT is the cost and the statistics.
- **X9. `torch.save` of a slice VIEW serialises the whole underlying storage — and `.contiguous()`
  does not save you.** Measured on torch 2.9.0: a 100-row view of a `(200000, 10)` float32 buffer
  wrote **8,001,492 bytes**; `view.contiguous()` wrote **8,001,633**; `view.clone()` wrote **5,566**.
  A row-slice of a contiguous 2-D tensor *is already contiguous*, so `.contiguous()` returns the same
  view and changes nothing — only `.clone()` detaches the storage. This bites anything that persists
  a window of a preallocated accumulator: the training checkpoint (C-11) writes a ~47 MB shard per
  interval with the clone and would have written the whole **4.35 GiB** buffer 100 times without it,
  ~435 GiB over a run. It presents as *"checkpointing got slow"*, never as a wrong answer, so no
  correctness test catches it — `test_checkpoint_shards_do_not_serialize_the_whole_accumulator`
  exists solely to pin the file SIZE.
- **X13. USING A FITTED POSTERIOR AS THE NEXT ROUND'S PRIOR IS TEMPERING, AND SBC WILL NOT CATCH
  IT.** Fit a density `q_1` to round 0's posterior, use it as the prior at the SAME observation, and
  you get `p_1 ∝ L·q_1 ∝ L²q` — and after L rounds, `L^(L+1)q`. Credible intervals contract as
  `(L+1)^(-1/2)` **with no new information entering**: at L=4 the posterior is 2.2× narrower than the
  data supports. **No current diagnostic sees this**, because SBC and TARP validate the flow against
  the proposal it was trained on, so round ℓ's SBC comes out flat while the reported intervals are
  wrong by `√(ℓ+1)` relative to the real prior. The correct form is TSNPE: **restrict the prior's
  support, never reshape it** — `q|_A / q(A)`, so `L·q|_A ∝ L·q` on `A` and the true posterior is
  recovered exactly, restricted. Restriction is not reweighting, which is why TSNPE needs no proposal
  correction and SNPE-A/B/C each do. See §11.6.
- **X12. `from .config import NAME` SNAPSHOTS the value at import, so writing `config.NAME = x` at
  runtime is a SILENT no-op.** `orchestrator.py` binds `TRAINING_NUM_RUNS`, `TRAINING_RUN_SIZE` and a
  dozen more this way, so a caller that "configures" a run by assigning to `core.config` changes
  nothing, gets the default, and is told nothing — which for the training budget is days of
  simulation at the wrong size. Three shapes of fix are already in the repo and they are not
  interchangeable: read through the MODULE (`config.QUIET_SEGMENT_BAR` — for a process-wide flag),
  carry it ON THE CONFIG (`SimConfig.chi_mode`, `reparam_rotate` — for anything a trained artifact
  must be self-describing about), or take it as a PARAMETER defaulting to the constant
  (`build_posterior(num_runs=…, run_size_cap=…)` — for a per-run override). Assigning to the
  *orchestrator's* attribute (`orchestrator.TRAINING_NUM_RUNS = n`, which `scripts/smoke_train.py`
  does) works but is process-global and leaks into the next run, so it is fine for a one-shot script
  and wrong for a GUI.
- **X11. On CUDA, a TorchScript step is NOT bitwise reproducible until it has warmed up.** The
  profiling executor runs a scripted function's first invocations unoptimised, then specialises and
  fuses; the fused kernel differs by ~1 ULP. **Measured: three consecutive eager runs of a fully
  deterministic model gave run1 ≠ run2 (differing from row 1, max|diff| 7.5e-09) and run2 == run3.**
  So any GPU claim of the form *"this seeded run reproduces bit-for-bit"* must warm up first or it is
  measuring the JIT, not the code. Harmless to the science (1 ULP against noise 8 orders larger) and
  invisible to the CPU suite — `euler_compiled` is CUDA-only and every CPU test runs the plain eager
  `euler` — but it cost a full investigation once, because it presents as *"my change broke bitwise
  equality"*. **Ask whether the BASELINE reproduces against itself before assuming you broke it.**
  Full account in §8.3 and the 2026-08-20 (later) Appendix entry.
- **X10. The Fisher rotation `V` is NOT reproducible across processes.** `build_latent_fisher_rotation`
  seeds its *noise* under `fork_rng`, but its **operating points** come from
  `latent_prior.sample(...)` on the caller's global RNG (`decorrelate.py`, the `z_med` and `z_samp`
  draws), which nothing seeds. Training is ground-truth-free, so both draws run. A restarted process
  therefore computes a different `V` → a different `T_train` → different **latent** targets for every
  batch after the restart, silently mixed with the pre-crash rows. This is why C-11's resume reuses
  the checkpoint's stored `V` and skips the Fisher entirely, and why that reuse lives in
  `build_posterior` rather than inside `gen_training_data`. Any future "just re-run the rotation"
  shortcut re-opens it.
  ✅ **Observed, not merely reasoned to, as of 2026-08-13.** The reuse path — including the CPU→CUDA
  rehoming of the stored `V`, which fires on a GPU resume with rotation and nowhere else — was
  executed on the card by the resume drill in the top Appendix A entry. Exercising it needs
  `smoke_train.py`'s `CKPT_DIR` **and** `PRIOR` together; `CKPT_DIR` alone routes run 2 to a different
  directory on `prior_fingerprint` and tests nothing.

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

> ### ▶ PICK UP HERE (as of 2026-08-21)
>
> **The retrain is DONE** (`posterior_08232026`, 2026-08-25) and characterised in **§4.6**: calibrated,
> converged, and uninformative in most directions.
>
> ### ▶ §11 IS IMPLEMENTED (2026-08-26). WHAT IS LEFT IS THE TWO RUNS THAT NEED THE CARD.
>
> All four phases landed; **nothing has been retrained**. The Phase-1 flow-only retrain needs no
> simulation — the cache is migrated and waiting at `Resources/Checkpoints/train_230ae7cb5fc2` — and
> the Phase-3 retrain is a full multi-day re-simulation on `master_tier1.txt`. See §11.9 for the two
> commands and the gates, and the top Appendix A entry for what was measured on the way.
>
> ### ▶ THE ORIGINAL FRAMING, KEPT BECAUSE IT EXPLAINS THE SHAPE OF THE WORK.
>
> Two causes are now MEASURED, not hypothesised: the flow cannot see part of its own conditioning
> (two channels attenuated **10⁷×**, `B1_log_Q` **69.7 %** sentinel, 11 of 42 channels structurally
> dead), and the prior spends **97.7 %** of its budget on physically inconsistent configurations.
> **Phase 1 needs no re-simulation** — the C-11 checkpoints hold all 10.24 M conditioning vectors.
>
> **Read §11.2 first.** §11 began as a user-written addendum; most of it verified exactly, four
> claims did not, and one of the failures is in the task that addendum ranks as decisive.
>
> §4.6 still rules out the obvious moves — the loss curve rules out more training, and `k` is FLAT
> rather than aliased. §11.2.1 explains why fixing Phase 1 will NOT change that: the Fisher is
> computed on a clean standardisation, so the two findings are independent.
>
> **Rows 0, 1 and 2 are DONE** (row 0 on 2026-08-21, rows 1–2 on 2026-08-13 — see Appendix A).
> **What is left needs a display or a decision, and neither is something a coding session can close
> on its own** — so unless the user asks for something new, the next move is the retrain:
>
> | # | What | Why it is worth doing | Size |
> |---|---|---|---|
> | ~~1~~ | ~~**Migrate the remaining non-atomic writes.**~~ `save_posterior_artifacts` (all three writes), `save_mix_dist` (hence `save_prior_artifacts`) and `retrain_convergence.py` now go through `file_manager.atomic_torch_save` / the new `atomic_savez`. **`FDT/cross_validation` was deliberately NOT migrated** and the reason is recorded at the function: its output path is TIMESTAMPED, so there is no existing file for atomicity to protect, and staging through a temp sibling would only rename "partial file at X" to "partial file at X.tmp" while destroying the incremental readability `load_param_sweep` is built around. | ✅ **DONE 2026-08-13** |
> | ~~2~~ | ~~**Two hardcoded plot colours.**~~ Done, and neither was what the row assumed — read the Appendix entry before "improving" either. `Reduction/plots.py`'s white **is correct and must stay** (its backdrop is viridis, not the page); the real defect was its invisible LEGEND swatch, fixed with a `text.color` halo. `FDT/plots.py` moved to `axes.edgecolor`, **not** `grid.color` — that was tried and is worse. | ✅ **DONE 2026-08-13** |
> | ~~0~~ | ~~**Rework the progress display: ONE overall bar, plus a Solver Performance rate with NO bar.**~~ Both parts done. The meter no longer parses a rendered bar at all — the solver publishes a step count (`core/progress.py`) that the pane differences on its 100 ms tick, which is correct at any speed rather than at speeds slow enough for tqdm to paint. The pane renders one bar + one caption; the nest lives on the bar's tooltip. **Read §10.5 before touching it** — two things there are not obvious: the caption's fallback is what keeps sbi's NN training visible for hours, and the solver's tqdm bar is deliberately left ENABLED under the GUI because it is the cancel's checkpoint. | ✅ **DONE 2026-08-21** |
> | ~~5~~ | ~~**Expose the training budget.**~~ *(Asked for 2026-08-21 while §10.5 was being verified: "how many simulations per batch, and should that be user-configurable?")* The answer was **5000 batches x 2048 rows = 10,240,000 simulations**, invisible outside `config.py`. Now on the Posterior tab as **Batches** + **Max rows per batch**, with three derived lines: the simulation count, a peak-VRAM estimate read from `pipeline.peak_sim_elements`, and the checkpoint status. **Read the Appendix A entry before changing any of it** — batch width is not a speed knob, the two fields are not interchangeable, and both sit in the C-11 checkpoint identity. | ✅ **DONE 2026-08-21** |
> | 3 | **Eyeball the new GUI surfaces on a real display**: the application icon (taskbar + window); the model builder's `_ParamRow`, which became TWO lines when it gained the box selector; **the reworked progress pane (§10.5, new 2026-08-21)** — one bar, one caption, a solver rate with no bar; and **the Posterior tab's new Training-budget group**, which is three wrapped derived lines and is the likeliest thing yet to overflow the controls column. | Headless tests cover the wiring, never the pixels — the standing rule in the last row of this table. Both are new since 2026-08-12 and no human has seen either. **Add to the list: the two figures above** — they were verified by rendering to PNG under both themes plus plain matplotlib, which is stronger than a headless test but still not a human looking at the app. | minutes |
> | 4 | **Decide the ND log-SAMPLING question** (S-1's remaining half). | `"box": "log"` changes the coordinate the GMM is fitted in; it does **not** move prior mass, because `UserPrior._global_map` still samples uniformly in the linear box. Making the sweep geometric is a *science* decision and would have to move the built-ins too. **Not a coding task — needs the user's call first.** | — |
>
> Do **not** re-open: B-c, B-e, S-1's `(lo, hi)` half, C-1 through C-11, or §9.1/§9.2 — all done, and
> the rows below say what landed. §9.3's file splits remain deliberately undone for the reason stated
> there, which the 2026-08-12 change set (a large edit across those same files) only strengthens.

| ID | Item | Status |
|---|---|---|
| **B-c** | ~~**Dark-theme matplotlib figures.**~~ `core/gui/mpl_theme.py` drives 16 rcParams from the design tokens, installed once by `app.build_app` and re-applied on `Appearance.theme_changed` — the same **apply-now + subscribe** shape `widgets/live_hair_bundle._install_theme` uses for pyqtgraph. The core plot modules (`visualizers.py`, `FDT/plots.py`, `Reduction/plots.py`) read `plt.rcParams` rather than importing GUI tokens, so they stay CLI-safe and get themed for free. One deliberate exception: `panels/simulate_export.py` pins its own white figure, because an exported video must not depend on the app's theme. **Known limitation, by design:** a theme flip does not recolour figures ALREADY displayed — they are rasterised to PNG at build time (`figure_stack.py`) and adopt the new theme on the next run. Pinned by 5 tests in `test_gui_progress.py`, incl. `test_plot_ppc_summary_box_follows_the_dark_theme` and the white-video guard. | ✅ **DONE** (shipped in `20a1a52`; this row was simply never updated) |
| **L-1..18** | ~~GUI layout/sizing (§10).~~ | ✅ **DONE 2026-07-28** — tier 1 + tier 2, pinned by 4 new geometry tests |
| **B-e** | ~~**A proper icon set.**~~ Solved with a **bundled icon FONT**, not the `.qrc`/SVG route this row proposed — `core/gui/icons.py` + `assets/icons/prism-icons.ttf`, generated offline by `build_prism_icons.py` (fontTools, original glyphs, MIT). **The reason for the font over `QIcon`s is load-bearing:** buttons render an icon as TEXT, so they recolour for free from the QSS `color:` on every theme flip, with no re-tint machinery. Every glyph keeps its unicode as a fallback, so a missing `.ttf` degrades rather than blanks. Now 5 glyphs — the original `⟳ ⚙ ← ?` plus `close` (U+E004) for the χ probe table's remove button. `artifact_picker`'s `➕` is now plain ASCII: it is a combo ITEM label, which `apply_icon` cannot reach, and a `QIcon` there would be the one baked-in colour that ignores the theme. **Also added, and not in this row: an application/window icon**, which did not exist at all (`setWindowIcon` appeared nowhere, so the taskbar showed the interpreter's mark) — `assets/app/prism.svg` → PNG set via `build_app_icon.py`, plus the Windows `SetCurrentProcessExplicitAppUserModelID` the taskbar needs. ⚠ Qt's `svg` **image-format plugin is absent here**, so `QIcon("x.svg")` is silently NULL; the SVG is source, the PNGs ship. | ✅ **DONE** (the four glyphs in `20a1a52`; the rest since) |
| **S-1** | ~~**Per-parameter bounds/priors in the model builder.**~~ Both halves now done, in two steps. **`(lo, hi)`** landed with `schema_version 2` (`20a1a52`): each ND param persists as `{value, lo, hi}`, validated (`lo < hi`, `lo <= value <= hi`), with a v1 in-memory migration, and the box reaches the Bounds file verbatim. **Prior type** is `schema_version 3`'s per-param **`"box"`** (`"linear"`/`"log"`) → `nd_log_params` → `ModelSpec.log_params` → `orchestrator._log_params_for` → `reparam.nd_log_mask`, the same seam `config.REPARAM_LOG_PARAMS` gives the built-ins, and already persisted into the `.rot.pt` sidecar. ⚠ **`"box"` is named for what it controls and is NOT the rescale/forcing blocks' `"log-uni"`:** it changes the COORDINATE the latent GMM and the flow work in (which is what linearizes a multiplicative degeneracy before rotating); it does **not** move prior mass, because `UserPrior._global_map` still samples uniformly in the linear box. Making the sweep geometric is a separate science decision that would have to move the built-ins too — deliberately not done. **ORDER invariant untouched:** the ND section is still written by `for p in compiled.param_names`. Two live bugs fixed in passing: `_ParamRow` read a blank min/max as a real `0.0` (`FloatField.value()`'s documented hazard, which `param_grid` guarded against and the builder did not — now `value_or_none`), and **forcing** params still took the placeholder box, so `amp=0.05` got `(-0.95, 1.05)` — now `phase` = `(0, 2π)`, `freq`/`tau` geometric and strictly positive, `amp` floored at 0. `_nd_bounds` promoted to public `nd_bounds` (§9.6's private-name-across-boundaries). | ✅ **DONE** |
| — | **A trained chi-mode posterior + the calibration payoff.** See §4.1 for the gated order. | OPEN — the priority |
| **C-1** | ~~Gate the chi band on `T_obs`.~~ | ✅ **DONE 2026-08-06** — §4.3.1. Result: the band's failures are a drive-CYCLE limit, not a frequency one. Superseded by **C-4**. |
| **C-4** | ~~Cap each chi probe's lock-in DURATION at a cycle count.~~ | ✅ **DONE 2026-08-06** — `config.CHI_MAX_CYCLES = 20`, applied in `gen_chi_raw`, on the `SimConfig`, in the sidecar and checked on load. §4.3.1. |
| **C-8** | ~~Per-ROW lock-in duration in `chi.lock_in_batched`.~~ | ✅ **DONE 2026-08-07** — §4.3.6. Live probes 46 % → **64 %**, inert rows 48 % → **33 %**, and the frequency span improved rather than traded away. |
| **C-6** | **~33 % of training rows carry no live chi probe** (was 55 %; §4.3.3–§4.3.6). Both parts done — per-ROW placement (C-6) and per-ROW durations (C-8). The residue is the genuinely unmeasurable tail: a sub-1 Hz bundle cannot complete two drive cycles at 0.03–0.3× its Ω₀ within 60 s at any placement or duration. Closing it further means changing the PRIOR (§4.3.4 option 2), not the estimator — a science decision. **Diagnosis, kept because it explains the shape of the fix:** the `CHI_MIN_CYCLES` floor is the only active predicate, and the driver is the row's own **Ω₀**, not `T` and not the band — live fraction goes 0 % below 3 Hz to 98 % above 30 Hz, while the prior spans ~4 decades of Ω₀. The masked rows are **genuine oscillators** (spectral prominence in the thousands), so the mask is correct physics and screening them out would be wrong. | ✅ **DONE 2026-08-07** — no longer a blocker |
| **C-9** | ~~The Fisher rotation amplifies ceiling-pinned `logcyc`.~~ **MEASURED on a real rotation 2026-08-10, and it does reproduce — but mildly, and INTERMITTENTLY, which is the part that matters.** `chi5_logcyc` pinned exactly as predicted (std 8.7e-05, ratio 2.9e-05, the same signature as the map) yet reached only `max\|J\|` = **6.0** — the *smallest* chi row, against the map's 2.0e4. The amplification depends on whether the ±dz arms straddle a `floor()` step; at that operating point they did not. A production rotation averages **8** operating points at m=48 over a ~4-decade Ω₀ prior, so this is a landmine that fires sometimes, not a constant — the worst kind to leave in. **Fixed with C-10 by the same one-line change.** | ✅ **DONE 2026-08-10** |
| **C-10** | ~~`chi.fisher_features` emits K duplicate rows.~~ **Confirmed hard on the same rotation:** `chi0/1/2/3_logcyc` agreed to **6 significant figures** (`max\|diff\|` ~2e-5 against entries of 37.44), because with the ceiling clear `logcyc_j = log(mult_j) + log(f_peak) + log(T_obs)` and both constants vanish under standardization — so the row **is** `A3_log_fpeak`'s, K times over, weighting that direction K-fold in `Jᵀ J`. **Fix: `logcyc` removed from `CHI_FISHER_CHANNELS`** (now `("logmag", "cos", "sin")`, 3K not 4K) and `fisher_features` reduced to **one argument**, which makes trap CHI10's whole class of mis-wiring a `TypeError`. Every `logcyc` row was a duplicate, a degrading duplicate, or quantization — never independent information, so nothing was lost. Pinned by two updated assertions plus `test_fisher_features_takes_one_argument_and_its_channels_are_what_they_claim`. | ✅ **DONE 2026-08-10** |
| **C-11** | ~~**No checkpointing in `gen_training_data`.**~~ Built, resumable, opt-in. `core/SBI/training_checkpoint.py` + `config.TRAINING_CHECKPOINT_EVERY = 50` + `CHECKPOINT_PATH`. **The invariant:** *a checkpoint describes batches `[0,k)`, with the RNG captured at the TOP of iteration k* — snapshot-and-restore, never replay, because the OOM retries redraw SDE noise so per-batch RNG consumption depends on what the desktop was doing. Persists the Sobol SCHEDULE (not a seed: `SobolEngine(scramble=True)` consumes the global RNG at construction and the accept/reject filter is geometry-dependent), torch CPU + CUDA RNG, `chi_gen`, `inits` (the tensor, not numpy's state — trap X8), and **`V`**. ⚠ **`V` is the part that would have been missed:** it is not reproducible across processes (trap **X10**), so `build_posterior` peeks the checkpoint BEFORE the rotation block and reuses the stored `V` rather than recomputing it — which also skips the Fisher, the dominant pre-training cost. Directory is a digest of a declared identity, so auto-detect is exact and two configs can never mix; `verify` then re-checks field by field and names the field. Commit order is shards → fsync → atomic state replace, so orphan shards from a crash are ignored by construction. A GUI **cancel** (incl. closing the window, via `MainWindow.closeEvent`) commits the completed batches on the way out and re-raises unconditionally. `checkpoint=None` is the whole compatibility story — `gen_cal_data` and every pre-C-11 call site are untouched and write nothing. Also removed the **8.7 GiB** host peak: the accumulators are preallocated and `train_nn`'s filter no longer gathers when the mask is all-true. **Deliberately not built:** auto-resume without an explicit mode, SNPE `num_rounds > 1` (refused loudly), cross-device resume, checkpointing sbi's fit loop, a GUI control, compressing shards, auto-deleting a completed checkpoint (it is a cache of a multi-day run — it lets you retrain the flow without re-simulating). | ✅ **DONE** |
| **C-7** | ~~`build_prior`'s `num_iterations=50` is a hard-coded literal, and its stability sweep shares `cfg.hw.batch_size` with TRAINING.~~ Promoted to **`config.PRIOR_SWEEP_ITERATIONS`** and **`config.PRIOR_SWEEP_BATCH`** (0 = follow `hw.batch_size`, the historical behaviour and still the right default). The trap it removes: the sweep is ITERATION-bounded, so shrinking the shared batch for a quick run made the prior worse *without* making it faster — 527 s at batch 2048 against >70 min and unfinished at 32. Also recorded at the constant: total candidates = batch x iterations, each round pays a full trajectory whatever the batch, and the subclasses' `batch_size % num_iterations` guard is **vacuous** (`construct_prior` passes `batch*iterations` down), so do not rely on it. | ✅ **DONE 2026-08-10** |
| **C-5** | ~~Settle the chi band's HIGH EDGE under the cap.~~ | ✅ **DONE 2026-08-06** — §4.3.2. `(0.03, 0.3)` **stands**: under the cap CV and SNR discriminate nowhere in 0.03–0.6, entrainment has a measured knee at 0.35–0.4×, and phase scatter is smooth so no threshold on it is evidence. The one residual is a judgement, not a gap: whether a phase-incoherent probe is worth including. Settling *that* needs two trained posteriors and is not worth it before a first working one exists. |
| **C-2** | ~~The Infer tab's variable-length probe table.~~ The Infer tab now holds an add/remove probe table; each row is **one `_ChiProbeRow` widget** carrying its recording AND the frequency it was actually driven at, and `_infer` submits `(path, freq_Hz)` PAIRS. Both stated constraints are met and pinned by tests: `_rebuild_chi_fields` **preserves** existing rows (they hold hand-typed frequencies and browsed paths a rebuild cannot regenerate; it seeds only when empty, never tops up or trims), and one-widget-per-row makes the middle-deletion mispairing structurally impossible. A blank frequency box is caught before the run — `FloatField.value()` returns 0.0 on bad text, and 0 Hz is a genuine DC probe the lock-in would attempt. | ✅ **DONE 2026-08-10** |
| **C-3** | ~~A GUI probe planner.~~ A **Plan probes…** button on the Infer tab's χ page measures Ω₀ from the selected passive recording and reports the in-band range in Hz, the minimum seconds to clear the `CHI_MIN_CYCLES` floor at the low edge, the length above which the ceiling truncates, and a per-row verdict at the entered `T_obs`. It also fills BLANK frequency boxes with a nominal in-band grid (blank only — a typed frequency is a record of what the bench did). **The predicates are not reimplemented:** they live in `chi.probe_verdict`, which `orchestrator.build_experiment_obs_chi` was refactored to call, so the planner and the run cannot disagree. | ✅ **DONE 2026-08-10** |
| — | ~~Make the `scripts/` diagnostics chi-aware.~~ | ✅ **DONE 2026-07-28** — `scripts/_common.py`; see §4.1 |
| — | ~~Non-atomic `torch.save`/HDF5 writes against a cancel.~~ The mechanism is now one function — `file_manager._atomic_write` (tmp sibling → fsync → `os.replace`, with a short retry for Windows' transient `PermissionError` from AV/Explorer) — with `atomic_torch_save` and the new **`atomic_savez`** on top of it. Callers: the training checkpoint (C-11), `save_mix_dist` (hence `save_prior_artifacts`), all three writes in `save_posterior_artifacts`, and `scripts/retrain_convergence.py`. `atomic_savez` takes its arrays as a **dict**, not `**kwargs`, so an array named `path`/`retries` cannot collide with the helper's own parameters — and passing a HANDLE rather than a name is what stops numpy appending `.npz` to the temp file's `.tmp` suffix. **`FDT/cross_validation` is a deliberate exception, recorded at the function:** its `output_path` is TIMESTAMPED, so there is no pre-existing file for atomicity to protect, and staging through a temp sibling would only move the partial file to `X.tmp` while destroying the incremental readability `load_param_sweep` and the all-failed error message both depend on. Its real data-loss path is an explicit `output_path` that already exists, since `"w"` truncates up front. Pinned by 3 tests in `test_artifact_consistency.py`, each verified to FAIL against the pre-change writes. | ✅ **DONE 2026-08-13** |
| — | **Never run for real on a display** — ask before assuming these work: the Parameter-Inference Save buttons, Validate, Infer (simulated and experimental), a full FDT/CrossVal run, a cancel *during* live NN training, the nav click-through, the Simulate trace/heatmap/cancel, the model-builder round-trip, and the accent/Inter checkboxes. Headless tests cover the wiring, not the pixels. **Added 2026-08-12 and unseen by anyone: the application icon (taskbar + window) and the model builder's `_ParamRow`, which is now TWO lines because it gained the box selector — `_ParamRow` is §10.2's worst offender for crowding, so the second line is the thing to look at.** | ONGOING |

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

> ### ▶ REOPENED 2026-08-20, and the reason it was ever closed is the lesson
>
> §8 was marked **CLOSED — "everything actionable is done"**, and it was wrong by ~8×. Every item it
> catalogued was a real but *small* one (a memset here, a duplicate FFT there), and the total of all
> of them is under 1 % of a production run. **The thing nobody had measured is that ~88 % of solver
> wall-clock was CPU kernel-LAUNCH overhead** — the GPU was ~6 % utilised and the CPU could not feed
> it. `scripts/profile_sde.py`, batch 2048, 100 000 steps:
>
> | | before | after | |
> |---|---|---|---|
> | end-to-end simulator call | **5520.5 ms** (±75.1) | **698.5 ms** (±1.0) | **7.90×** |
>
> Fixed by replaying the Euler step loop from a captured **CUDA Graph**
> (`config.SOLVER_CUDA_GRAPHS`, `Solvers/sdeint.py`). Note the variance too: ±75 ms → ±1 ms, because
> the jitter *was* the CPU.
>
> **Why it was missed for so long, in one sentence:** `euler_compiled`'s own docstring claimed
> "torch.compile + CUDA Graphs" and required a step "wrapped with
> `@torch.compile(mode='reduce-overhead')`" — so anyone reading the solver concluded the fast path
> already existed. **`torch.compile(` appeared exactly once in the entire repo: inside that
> docstring.** Every `compiled_step` is `@torch.jit.script`, which removes Python overhead and not
> launch overhead. *A docstring describing a design that was never built is worse than no docstring:
> it is a closed door with a sign on it saying the room is already furnished.*
>
> **Rank optimisations against the PRODUCTION denominator, not a smoke train's.** §4.1 step 5 says
> the Fisher rotation dominates the pre-training cost; that is true of a *smoke* train (4 batches)
> and false of the *record* run, and the difference has misdirected effort:
>
> | stage | share of a production retrain |
> |---|---|
> | **training generation** | **~97 %** (5000 batches × (1+K) sims) |
> | validate | ~3 % |
> | Fisher rotation | ~1–2 % |
> | `gen_stats`, all 41 features | ~0.3 % (~7.5 min over the whole run) |
>
> So the Fisher's 216-call structure and the duplicate FFTs in `statistics.py` — the two most
> obvious-looking targets — are together worth **~20 minutes out of ~38 hours**. See §8.3 for what
> is left and what it is gated on.

> **The REJECTED list stands unchanged.** Four recommendations were investigated and refused, each
> with the reason recorded at the code site so they are not re-attempted:
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

### 8.3 What the CUDA-graph work landed, and what is left (2026-08-20)

**Landed.** `Solvers/sdeint.py` captures `SOLVER_GRAPH_CHUNK = 50` Euler steps into a
`torch.cuda.CUDAGraph` and replays it, with the last `n % chunk` steps run eagerly as a tail.
Config: `SOLVER_CUDA_GRAPHS` (live-read, so it can be flipped in-process), `SOLVER_GRAPH_CHUNK`,
`SOLVER_GRAPH_CACHE_MAX`. Also landed, both bit-identical: `Simulator.simulate`'s `sol` is
`torch.empty` not `torch.zeros` (the *larger* of the two buffers, ~6.95 GiB — §8.1 listed the
`sdeint` one and missed this), and `forcing.zero_force` returns a stride-0 expansion instead of a
materialised block of zeros (2.29 → 1.82 GiB, verified identical checksum through `gen_obs`).

**Four things about the graph path that are load-bearing:**

- **The parameters are STATIC buffers copied in per call, not captured by reference.** That is what
  lets one capture serve every training batch. Capturing `compiled_params()` directly would bake one
  model's tensor addresses and silently replay stale physics for the next batch — wrong numbers, no
  error.
- **The cache is module-level, deliberately.** Not on `Solver` (trap **X1**: `sdeint.Solver` must
  stay patchable at call time) and not a module-level `Solver` singleton (same trap). A per-Solver
  cache would also be useless — `Solver` is constructed once per *time segment*, so it would
  recapture constantly. Pinned by `test_the_graph_cache_is_not_hung_off_the_solver_class`.
- **The bar must be advanced by the chunk** — and so must the step counter. Both go through
  `sdeint._advance(bar, k)`, which exists precisely so a caller cannot update one and forget the
  other: a graphed run that ticked once per replay would read 1/50th of the truth. ⚠ **The GUI's
  Solver Performance meter no longer reads this bar** (trap **S0**, 2026-08-21) — it reads
  `core.progress.SOLVER`, because this very speedup made the bar too fast to render a rate. The
  bar's own `it/s` still matters to the CLI, which has nothing else.
- **Capture failure falls back to the eager loop** and disables further attempts for the process,
  with one printed line. A solver that refuses to run is worse than a slow one.

> ### ⚠ NEW TRAP — TorchScript's profiling executor is not bitwise reproducible on its first runs
>
> Chasing a 0.83-ULP discrepancy between the graphed and eager paths turned up something that has
> nothing to do with CUDA graphs: **three consecutive EAGER runs of a fully deterministic model gave
> run1 ≠ run2 (differing from row 1, max|diff| 7.5e-09) and run2 == run3.** TorchScript runs the
> first invocations of a scripted function unoptimised, then specialises and fuses, and the fused
> kernel differs by ~1 ULP (fused multiply-add against separate ops).
>
> So **"graphed == eager bitwise" is not a well-posed assertion until both are warm** — the eager
> baseline is not bitwise reproducible against *itself*. After a 3-run warm-up, all four comparisons
> (eager/eager, graph/graph, graph/eager, replayed-region) are exactly 0.0.
>
> This is **CUDA-only and TorchScript-only**, so it does not touch the CPU suite's
> seeded-reproducibility gates (X8) or C-11's bit-identical resume: `euler_compiled` is selected only
> on CUDA, and every CPU test runs the plain eager `euler`. But any future claim of the form "this
> GPU run reproduces bit-for-bit" has to warm up first, or it is measuring the JIT.

**Still open, in value order, each with its blocker:**

| item | worth | gated on |
|---|---|---|
| **Probe concatenation** — `gen_chi_raw` walks the time loop `1+K` ≈ 8 times per chi batch, once per probe. Width is FREE in time (measured flat 512 → 16384 rows), so running them as one wider call is up to **8× on the dominant cost**, and largely multiplicative with the graph win since the card is still unsaturated. | up to 8× | **The force tensor.** `build_nondim_sin_force_tensor` returns 2.16 GiB with an **8.64 GiB transient** (4×) — `forcing.py` materialises a full `(batch, T)` `t_dim` plus three more `(batch, T)` temporaries. Fix that first (fold `t_scale`/`t_offset` into the phase) or concatenation buys only ~1.25×. ⚠ **`_max_sim_batch` budgets only `n_ch * n_fine` for the drive, under-counting the real peak 4× — a plausible contributor to trap X7's unwrapped OOMs.** |
| **Strided + single-variable solver output** — the solver materialises the full fine trajectory of all variables, then the caller keeps every `subsample_factor`-th sample of variable 0 (mean ratio 9.7, p90 24.3) | ~23 % time, ~6× peak | Keep indices are **global** and segment boundaries are not stride-aligned; the seam duplication (`simulator.py`) must be reproduced; the return has to become `(kept, x_final)` because the segment carry-over needs all variables at the true last step; and the PPC path uses a **per-row** subsample, so `keep` must accept an explicit index tensor. |
| **PPC binning by a step budget** instead of `PPC_BIN_SIZE = 50` | 0.02–0.8 h | Bins are `t_scale`-**sorted** and each bin's `n_fine` is keyed on its *smallest* `t_scale`, so a 5× wider bin is not 5× cheaper (3.6–3.9×) and the worst bin's memory grows 5×. Bin on `rows × per_row_elements`. |
| **Fisher arm stacking** (2×13 sequential `feats` calls per operating point) | **~15–30 min of ~38 h — DROP** | Needs CRN-preserving *tiled* noise, arms grouped by `(n_fine, subs, n_segs)` (the `t_scale` arms differ), and split granularity pinned to multiples of `m` or a mid-arm split silently destroys CRN (trap X3 in a new costume). High risk, negligible payoff. Take only the free part §8.1 lists: hoist `feats`'s per-call re-derivation of `subs`/`n_fine`/`t_fine`/`n_segs`/`base_inits`. |
| **`gen_stats` dedup** — `_acf` and `_analytic_bandpass` each computed twice with identical arguments; `chi.peak_freq` re-does the rfft peak `_build_spectral` already has | **~6 min of ~38 h** | Nothing. Genuine duplicates. Drive-by only if already in `statistics.py`; never a session. |

**A measurement, not a change: `dt_nd_min` is a sampling constraint wearing an integration
constraint's clothes.** `config.dt_nd_min = dt_exp / t_scale_MAX` is derived from a *prior bound*, so
raising `t_scale`'s upper bound from 40 to 80 would silently double every simulation's cost and
accuracy with no change to any physics, and a batch whose own `t_scale` is 1 integrates ~40× finer
than it samples. Nothing has ever established what `dt` the ND dynamics actually need — and the
current step is already 250× larger than the fastest admissible `tau`, so a study could equally
conclude the run is *under*-resolved. **The obvious fix is wrong:** a per-batch `dt_nd_k` would make
integration error a deterministic function of `t_scale`, which is an *inferred parameter* — the
network could learn to read `t_scale` off a numerical artifact present in training and absent in real
data, which is the same train/eval mismatch class as the 2026-08-19 retired-band failure. The
defensible form is to establish `dt_nd_required` by a dt-halving convergence study on **each of the
41 features individually, in `fnoise` units**, make it a constant *independent of `t_scale`*, and set
`subs_k` from it. Science decision; do not bundle it with a performance session.

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


### 9.1 Four already-wrong facts (cheapest possible fixes) — ✅ ALL FIXED

> **Verified fixed (re-checked in full).** Items 2 and 3 were still listed as open long after they had
> been repaired, which is how this section came to need its own audit:
> `vt.py` cites `tests/test_gui_progress.py`; **no `GFDT` string occurs anywhere in `core/gui/`** (the
> dialog reads "PRISM could not start"); and the banners now read `── 1. Config` / `2. Prior` /
> `3. Posterior` / `4. Validate` / `5. Infer` — five, contiguous, matching the five tabs.

1. `core/gui/vt.py:4` — "see `tests/test_vt.py`". **That file does not exist.**
2. `core/gui/app.py:67` — the user-facing startup-failure dialog is titled **"GFDT could not start"**.
   The app is PRISM everywhere else. This is visible to users.
3. `core/gui/panels/inference_tabs.py` — the section banners read `── 1. Config`, `── 2. Prior`,
   `── 4. Posterior`, `── 5. Validate`, `── 6. Infer`. **There is no 3**, and the numbers don't match
   the five tabs. `_StagePanel`'s docstring says "the **six** inference tabs" while the module
   docstring says five.
4. `core/Simulator/user_simulator.py` and `core/gui/panels/simulate_runner.py` still carried stale
   `exit()`/`SystemExit` prose after the 2026-07-28 fix — corrected in that pass; **check for more.**

### 9.2 A latent crash: a module shadowed by a local — ✅ FIXED

> `inference_tabs._build_config` now binds `model_labels`, so nothing shadows the `labels` module.
> The one surviving `labels = VALID_LABELS[...]` is in `panels/simulate_runner.py`, and that file
> does **not** import the `labels` module — so there is nothing there to shadow. Checked, not assumed.

`core/gui/panels/inference_tabs.py` imports `from core.Helpers import ... labels ...`, then
`_build_config` rebinds it: `labels = VALID_LABELS[VALID_MODELS.index(model)]`. This works only
because that one function never touches the module — but the same file calls `labels.axis_label(...)`
and `labels.gui_forcing_label(...)` elsewhere. **Any future line added to `_build_config` that uses
the module raises `AttributeError` on a `list`.** Rename the local to `model_labels`.

### 9.3 Files that are too large

> **Counts refreshed 2026-08-26** — every one of them was stale by 40–130 %, which is the same
> defect §9.1 catalogued. The split suggestions still stand and are still deliberately undone;
> §11 made four of these five files larger, so the case is stronger, not weaker.

| File | Lines | Contents | Suggested split |
|---|---|---|---|
| `tests/test_gui_progress.py` | **2614** | **96** tests over **eight** unrelated subjects | `test_vt_progress.py` (restoring the name `vt.py:4` already claims exists), `test_worker_dispatch.py`, `test_settings_persistence.py`, `test_figures.py`, `test_nav_and_gating.py`, `test_simulate.py` |
| `core/orchestrator.py` | **2211** | five stage fns + PPC + overlays + experimental paths + the observation/truncation helpers | by stage |
| `core/SBI/pipeline.py` | **2072** | simulation, stats, training data, training | `simulate.py` / `training_data.py` / `train.py` |
| `core/gui/panels/inference_tabs.py` | **1530** | **six** screens + the HELP dict + five worker runners + `_TrainingBudgetMixin` | `inference/help_text.py`, `inference/runners.py`, `inference/base.py`, then one file per tab |
| `core/config.py` | **1066** | tuning constants **and** path constants **and** the `SimConfig` dataclass with real behaviour | `config/constants.py` + `config/sim_config.py` |

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

> **Status: `.gitignore` covers all of it; the UNTRACKING is still outstanding and is a deliberate,
> user-owned git step.** 20 files are currently *tracked but ignored* — the 6.9 MB `sbc_run.log` plus
> 19 generated files under `Resources/` (7 `.h5`, 5 `.png`, 1 `.pt`, 6 `.parquet`). Nothing in the
> code references any of them (checked: no `.py` mentions `shm.pt`/`shm.png`). The command is
> self-updating, so it needs no maintained list:
>
> ```bash
> git rm --cached $(git ls-files -i -c --exclude-standard)
> ```
>
> `Resources/Checkpoints/` (C-11) was added to `.gitignore` at birth and was never tracked.

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

### 10.5 Progress display: ONE overall bar + a Solver Performance rate with NO bar — ✅ DONE 2026-08-21

**Requested 2026-08-21, shipped the same day.** What is on screen now:

```
Solver Performance: +++++  (142.4k it/s)   [sparkline]
[███████░░░░░░░░░░░░░  38%]  ⠹
Generating training data  —  1902/5000 [05:12<13:41]
```

One bar. One caption. A solver rate with no bar of its own. The full nest is still *parsed* — nothing
in `core/gui/vt.py` changed — it is just rendered as one line, with the whole live set on the overall
bar's **tooltip**, one hover away rather than deleted.

#### (a) The rate: a published counter, not a parsed bar

**The regression, reproduced exactly before anything was changed** — `tqdm 4.67.1`, the real bar
settings, a synthetic 100 k-step bar paced to 0.7 s (no GPU needed):

| `mininterval` | `miniters` | stderr | rates rendered |
|---|---|---|---|
| **1.0 (as shipped)** | 1000 | **123 chars** | **none** — only the `?it/s` opening frame |
| 0.1 | 1000 | 601 chars | 6, e.g. `142353.52it/s` |
| 0.1 | 100 | 601 chars | 6 |
| 0.05 | 1000 | 1139 chars | 13 |

123 chars and zero rates is byte-for-byte what was measured on the real run, so the reproduction is
faithful. **It also settles the ⚠ this section used to raise: `miniters` is NOT the blocker.** At
`SOLVER_GRAPH_CHUNK = 50`, 1000 steps is 20 `update()` calls ≈ 14 ms — far inside a 100 ms
`mininterval`. Lowering `mininterval` alone is sufficient, and it *was* lowered (1.0 → 0.1, tqdm's own
default) — **but for the CLI, which has nothing but the bar and was equally blind.**

**The GUI does not read the bar at all any more**, because `mininterval` is a *time* gate and so this
whole failure class returns for any call shorter than whatever it is set to. Instead:

- **`core/progress.py`** — `SOLVER`, a monotonic count of completed steps. Qt-free, torch-free,
  CLI-safe. One writer (the thread running the solver), many readers; never reset, because readers
  take *differences* and a reset would present as a negative rate.
- **`sdeint._step_iter`** publishes one step per iteration (a generator wrapping the tqdm object —
  ~50 ns against a measured 54.87 µs/step eager, i.e. under 0.1 %); **`sdeint._advance(bar, k)`**
  publishes the graphed path's chunk *and* advances the bar, so a caller cannot update one and forget
  the other.
- **`ProgressPane._sample_solver`** differences the counter on the 100 ms tick it already ran for the
  spinner, EMA-smoothed at 0.3 (tqdm's own weight, so the number reads the way the CLI's bar does).

That is correct at *any* speed, including the next speedup. `SOLVER_IDLE_S` came down 45 → **8 s**:
45 s was sized to bridge ~1 s gaps between rate frames on ~10 s calls; with samples every 100 ms it
only has to bridge the non-solver work *between* calls, and 45 s meant showing a stale number well
into neural-network training. The `QProgressBar` step strip is gone — that is the "with NO bar" half.

#### (b) One bar, and the caption that is not decoration

> ### ⚠ THE THING THIS SECTION'S ORIGINAL SPEC MISSED
>
> "Render no other rows" taken literally **blacks out the longest phase of the entire run.** sbi's
> neural-network training emits *no tqdm bar at all* — it prints `"\r"` + `Training neural network.
> Epochs trained: N`, which `vt` turns into an overwrite-mode row with `pct is None` (trap P7, pinned
> by `test_sbi_epoch_counter_becomes_a_progress_row_not_log_spam`). During that phase — hours of a
> multi-day build — the only live rows are that status line and a degenerate `total=1` bar. With rows
> deleted and no caption, the pane would show an indeterminate bar and nothing else.
>
> Hence `_paint_caption`'s fallback: **when nothing reports a percentage, show the deepest row whose
> total is not 1.** Do not "simplify" it away.

**The election rule changed too, and this was not cosmetic.** It was *deepest informative row*; it is
now *largest non-degenerate `total`* (ties broken on depth). Deepest was one config flip from being
wrong: `Running time segments` wraps `segs` in {1,2,3}, so it is both deeper and far coarser than
`Generating training data` — `config.QUIET_SEGMENT_BAR` was papering over the election rule rather
than merely tidying the display. The two rules agree on every nest actually observed;
`test_the_overall_bar_follows_the_largest_non_degenerate_total` is the case that separates them, and
was **verified non-vacuous** (deepest → 66 %, largest-total → 38 %).

#### Deliberately NOT done: quieting the solver bar under the GUI

It is tempting — it would delete the `SOLVER_BAR_DESC` seam outright, the way `QUIET_SEGMENT_BAR`
deletes the segment bar. **Do not.** Two things depend on that bar still writing:

- **Cancellation** (trap **C**) is cooperative and its checkpoint is `_SignalStream.write()`. During
  training-data generation the solver bar is by far the most frequent writer; the next-most is the
  per-batch `Generating training data` redraw, ~2 s at production geometry and longer at large
  `T_obs`. Quieting it turns ~0.1 s cancel latency into one batch.
- **The stall detector** fires on no output of *any* kind for `STALL_S`. *(This side is now
  belt-and-braces — `_sample_solver` heartbeats on a moving counter — but the cancel side is not.)*

Cost of keeping it: ~600 chars per 0.7 s call through the router. Negligible.

`vt.parse_rate` / `RowState.rate` were **kept** although nothing acts on them now: `parse_bar`'s
contract is a faithful parse of a tqdm frame, it is one regex against a string already in hand, and
the `s/it` inversion is knowledge that cost a bug (trap **S4**).

#### Verification

`tests/test_gui_progress.py` is 79 → **83**: two solver-meter tests rewritten, four added — the CLI
bar's rate over a sub-second call *with its counterfactual* (`mininterval=1.0` must render none, or
the test proves nothing), the caption's fallback to the epoch counter, and the largest-total election.
Plus an end-to-end smoke of the real solver → `redirect_streams` → pump → a real `ProgressPane` on
CPU: **a live rate on 195 of 196 ticks (99 %)**, against 0 % before. And a cancel raised from *inside*
the solver's step loop: it unwound in well under a second and the next `tqdm` bar did not deadlock —
the generator `_step_iter` became sits inside trap **C1**'s unwind path, so this was run rather than
argued. A cheap structural guard on the same property is now in the suite
(`test_the_solver_step_iterator_closes_its_bar_when_the_consumer_raises`) — C1's own test *hangs*
rather than failing, which is a bad way to find out.

**Measured on the real graphed path (RTX 5070 Ti, batch 2048, `T = 100 000`, `CHUNK = 50`):**

| | wall | steps/s | stderr | rate frames |
|---|---|---|---|---|
| graphs ON | 0.811 s | **123 349** | 678 chars | **7** (first `128420.87it/s`) |
| graphs OFF | 7.912 s | 12 639 | 6059 chars | 78 |

The ON row is the case that used to emit 123 chars and **zero** rate frames. Step accounting is exact
on both paths at a geometry with a partial tail (2 full chunks + 3 leftover steps → 102 published,
102 expected), which is what pins `_advance` across the chunked replay *and* the eager remainder.

**Still owed: a human looking at it.** Headless tests cover the wiring, never the pixels — this is on
§6 row 3's eyeball list with the app icon and `_ParamRow`.

---

## 11. The informativeness programme — ▶ IMPLEMENTED 2026-08-26, NOT YET RETRAINED

> ### ▶ STATUS, and read it before re-planning any of what follows
>
> | phase | code | run |
> |---|---|---|
> | 1 — repair the feature set | ✅ landed | ⬅ **the flow-only retrain is NEXT; it needs no simulation** |
> | 2 — the informativeness scalar | ✅ landed, reported by `validate_calibration` | runs with any validate |
> | 3 — tier-1 prior narrowing | ✅ landed, own box `master_tier1.txt` | needs a full re-simulation |
> | 4 — TSNPE | ✅ landed, six guardrails + the pinning test | needs a Phase-1 posterior first |
>
> **§11.9 has the two commands.** Three things below were changed by measurement while implementing
> them, and each is corrected in place: the valid-flag set is **8 channels, not 1** (§11.3), §11.2's
> correction 3 gives the **wrong reason** for a right number, and §11.8's "edit `master.txt`" would
> have orphaned the very cache Phase 1 needs. The top Appendix A entry is the full account.


**Where this came from.** `posterior_08232026` (§4.6) is calibrated, converged and uninformative. A
user-written analysis, `PRISM_HANDOFF_ADDENDUM_2026-08-25.md` (kept on the Desktop), proposed a
diagnosis and a work queue. **Every checkable claim in it was run against the real artifacts** — the
posterior, the sidecar, and the two complete 5000×2048 C-11 training checkpoints. Most of it holds
exactly; four things do not. §11.1 and §11.2 are that ledger, and **§11.2 must be read before acting
on the addendum**, because one of its errors is in the test it nominates as decisive.

> ### THE TWO CAUSES, BOTH NOW MEASURED
>
> 1. **The flow cannot see part of its own conditioning.** `A1_mean` and `D3_bimodality` are
>    attenuated **10⁷×** by contaminated standardisation; `B1_log_Q` is **69.7 %** sentinel; 11 of 42
>    channels are structurally dead.
> 2. **The prior spends most of its budget on impossible configurations.** Only **2.27 %** of training
>    rows have an implied bath temperature in 280–310 K; **33.7 %** carry zero live χ probes (mean
>    **3.37** live of 12 slots).
>
> Neither is fixed by more training — §4.6's loss curve settles that.

### 11.1 The ledger — what is MEASURED

`Resources/Checkpoints/train_98aebd93ed17` is the checkpoint this posterior trained on. Confirmed:
applying `train_nn`'s `abs(data) < 1e15` guard to its 10.24 M rows reproduces the stored
`sum_mean`/`sum_std` **to every digit**. (The sibling `train_6c80f7d8037d` does not — do not mix them.)

| addendum claim | verified |
|---|---|
| §2.1 all `sum_mean`/`sum_std` | exact to 5 s.f. |
| §2.2 Sarle's coefficient bounded in (0,1] | correct — `statistics.py::_group_d` uses `(z**4).mean()`, the RAW 4th moment, so β₂ ≥ g₁²+1 holds. Had it used *excess* kurtosis the bound would fail; it does not |
| §2.3 `v = σ²/μ = 1.0000e12 = 1/_EPS`, `k = 2.000` | **exactly 2 surviving rows at exactly 1e12** — predicted from σ²/μ alone with no access to the data |
| §2.4 standardised spans ~5e-9 | measured p1–p99: `A1_mean` 3.98e-9, `D3` 1.44e-9 |
| §3.2 `B1_log_Q` sentinel fraction 0.699 | **measured 0.697** over 1.02 M rows |
| §3.4 exactly 11 pass-through Group-G channels | exact — and ablation confirms they are *precisely* zero |
| §4.3 `elem_std` = [1.1513, 2.5148, 0.7096, 0.7025, 0.8398] | exact to 4 d.p. |
| §10 989,312 estimator parameters | exact |
| §6.2 the whole drift/diffusion table | verified line-by-line against `nadrowski_model.py`: `n`, `tau_c`, `temp` appear in **no** drift term; `tau_c` only as `tau_c/n`; `temp` only in `sqrt(2*temp/(n*beta*lam))` |
| §5.3 implied `f_scale` range | measured (0.33, 1.18e4) vs claimed (0.29, 12018) |

**The `1/_EPS` mechanism is sharper than the addendum states.** `_group_d` computes
`z = (s-mu)/std.clamp(1e-12)`. For an **exactly constant trace** `s - mu ≡ 0`, so `z ≡ 0`, `kurt = 0`,
the clamp fires and `bimod = 1/1e-12` exactly. Not "kurtosis underflowed" — **a flatlined simulation**.

> ### ⚠ ONE PHENOMENON, THREE VICTIMS — AND NOTHING REPORTED IT
> `D3`'s contamination is *exactly constant* traces. `A1_mean`'s is ~1e32-magnitude traces (measured
> min −4.59e32). The 2026-08-25 missing PSD band was divergent draws. **That is one population of
> pathological simulations, damaging three unrelated things silently.** The `n_dropped` reporting
> added 2026-08-25 is the first thing in the pipeline that would surface any of it. A per-batch count
> of non-finite / exactly-constant / overflow trajectories, logged during generation, is Phase 1's
> cheapest item and should have existed from the start.

### 11.2 ⚠ The corrections — READ BEFORE ACTING ON THE ADDENDUM

**1. §2.5's "decisive test" is the wrong test, and as written would refute its own §2 finding.**
The addendum's task 0.1 says to replace channel *i* with its fitted mean and expect **exactly zero**
change from a dead channel. Measured: `A1_mean` gives **5.2e-5** and `D3` **4.6e-4** — not zero. The
fitted mean is *itself contaminated* and sits ~10⁷ from the data, so substituting it is a LARGE move
in standardised units. Run task 0.1 as specified and you conclude both channels are alive and discard
the document's best finding.

**The correct test is to sweep the channel across its real p1–p99 range.** Measured that way:

| channel | max ‖Δ embedding‖ over its real range |
|---|---|
| `A1_mean` | **1.8e-7** |
| `D3_bimodality` | **8.9e-8** |
| `A3_log_fpeak` | 1.379 |
| `A2_log_var` | 1.813 |
| `B4_spec_entropy` | 1.159 |

**A factor of ~10⁷.** float32 ε is 1.19e-7, so the entire physical range of `A1_mean` moves the
embedding by less than one float32 ulp. §2's *conclusion* is right; its *test* is not.

**2. §3.2's "the fits are exact, which is itself strong evidence the two-point model is right" is a
logical error.** Two parameters solved against two moments reproduce both **by construction** —
`p = r/(1+r)` with `r = σ²/(μ−S)²`, closed form, no residual. It is an identity, not evidence. And it
produces one **false positive**: `C2_log_env_cv` predicted p = 0.056, **measured 0.000** — its std is
a genuine heavy tail. The addendum's own §10 hedges this correctly; its §3.2 does not.

**3. §5.2's NUMBERS are wrong and its REASON here is wrong too — corrected 2026-08-26.**

> ⚠ This entry used to read *"it assumes `x_scale`/`f_scale` are log-uniform. They are not:
> `REPARAM_LOG_PARAMS = []`, an all-linear box."* **That reason is false.** `REPARAM_LOG_PARAMS`
> governs only the **ND** block's box coordinate. The rescale block is built by
> `orchestrator._build_rescale_prior`, whose rule is `"scale" in name → "log-uni"`; sampling it
> directly gives log-medians of 3.451 / 1.844 / 3.457 against a log-uniform's 3.454 / 1.844 / 3.454.
> **They ARE log-uniform in physical space.** The measured quantiles below stand — they were taken on
> real rows, and `derived.implied_temperature` reproduces them to 14 / 81 / 287 / 945 / 6318 K — and
> the variance argument is untouched. But do not re-derive a survival fraction from the retracted
> cause.

Measured
over 1.13 M real training rows, implied `T` quantiles are **14 / 80 / 286 / 939 / 6292 K**, not
0.33 / 2.5 / 15.2 / 90 / 821. **The median is 286 K — room temperature — not 15 K.**
**The spread claim survives and is what matters**: 5–95 % spans a factor of 450, and only **2.27 %**
of rows fall in 280–310 K. §5 is a **variance** argument, not the bias argument it presents.

**4. §5.3's 60.7 % is 74.3 %** — same linear-vs-log cause. ~26 % of draws imply an `f_scale` the
declared box forbids. Real, less dramatic.

**5. §2.3 under-credits the magnitude guard.** 12 rows exceed `D3`'s ceiling in the raw data; the
`1e15` guard drops 10 of them, leaving exactly the 2 the reconstruction predicts.

**6. §11's verification snippet will not run** — it reads `posterior_08232026_rot.pt`; the file is
`posterior_08232026.rot.pt`.

**Not checkable from this repo:** §5.1's `Nβ = f_scale·x_scale/(k_BT)` and §6.5's centre-manifold
argument rest on journal sections (§1.2, §1.3.2, §2.6) that are not in the repository. Internally
consistent, unverified here.

### 11.2.1 How §11 and §4.6 fit together — the question the addendum raises and answers wrongly

The addendum's §0 argues §4.6's "identifiability limit" verdict is invalid *because* the conditioning
is broken. **That challenge is checkable and the answer is no.** `decorrelate.py` standardises its
Jacobian by `fnoise = np.maximum(f0.std(0), 1e-9)` — the **single-trajectory feature noise measured
locally at each operating point**, NOT the embedding's training-set `sum_std`. So the Fisher, `V`,
and §4.6's entire decomposition run on a **clean** standardisation and are not artifacts of the §2 bug.

Two independent facts, both true:

- **§4.6 (clean features):** `k`'s gradient is genuinely tiny *relative to trajectory noise*.
- **§11 (contaminated features):** the flow additionally cannot see `A1_mean` — which §4.4.1's payload
  table names as `k`'s top feature by 6×.

> **⇒ Fixing Phase 1 should recover real information, but it is bounded above by a Fisher gradient
> already measured as near-zero on clean features. Expect gains in `x_scale`, `delta_E` and the
> absolute-scale directions. Do NOT expect `k` to become well determined.** The two documents are
> consistent once you notice they standardise differently.

### 11.3 Phase 1 — repair the feature set (NO re-simulation)

The checkpoint stores all 10.24 M conditioning vectors, so every item here is a pure function of
cached data. **Retrain the flow only.** Highest evidence, lowest cost — do this first.

| # | change | file |
|---|---|---|
| 1.1 | **Rank-Gaussianisation** replaces mean/std in `EmbeddedNet.fit_standardization`: per-channel empirical CDF → probit, stored as monotone-interpolant buffers | `core/SBI/embedded_network.py` |
| 1.2 | **Valid flags**, mirroring the mask design `ChiSetEncoder` already uses for the probe block. ⚠ **8 channels, not 1** — see below | `core/SBI/statistics.py` |
| 1.3 | **Per-column winsorisation** (0.1/99.9 pct) replacing `train_nn`'s global `abs(data) < 1e15` guard | `core/SBI/pipeline.py` |
| 1.4 | **Pathological-trajectory counter** — per batch, count non-finite / exactly-constant / overflow-magnitude trajectories and log them | `core/SBI/pipeline.py` |

Why rank-Gaussianisation specifically: invertible (no information lost), invariant to any monotone
transform (**which kills the log-versus-linear question for every channel at once** — see
`REPARAM_LOG_PARAMS`' history), outlier-immune by construction (two rows at 1e12 become two rows at
the top rank, not a scale factor), and sentinel mass becomes a point mass at a known quantile the flow
can key on.

**1.2 needs no re-simulation *this time*.** Every predicate is an exact equality on a value the
checkpoint already stores, so the flags are derivable from the cache. There is **one** derivation
(`statistics.derive_valid_flags`), run by the live pipeline and by the migration alike — it derives
from the feature VALUES rather than from the internal booleans that produced them, so the two paths
cannot drift and there is no assertion to maintain because there is no second implementation.

> ### ⚠ IT IS EIGHT CHANNELS, NOT ONE — MEASURED OVER ALL 10.24 M ROWS
> This item was written as though `B1_log_Q` were the case. A point-mass census says otherwise, and
> this is what sets the conditioning width to **50 + 6·K_PAD = 122**:
>
> | flag | substituted | | flag | substituted |
> |---|---|---|---|---|
> | `V_B1_Q` | **69.66 %** | | `V_E1_tau_slow` | 8.01 % |
> | `V_C7_slowenv` | **30.04 %** | | `V_E1_tau_fast` | 7.53 % |
> | `V_B7_secondary` | **14.49 %** | | `V_E2_h3` | 5.51 % |
> | `V_C6_slowenv_tau` | 5.07 % | | `V_E2_h2` | 2.83 % |
>
> Excluded, each by measurement rather than by argument: `C2_log_env_cv` fires on **exactly 0 rows in
> 10.24 M** (the addendum predicted 5.6 %); `B6_log_rolloff_ratio` at 0.199 % is under the 1 %
> threshold; `B2`/`B3`/`C1` never fire; and `A4_log_acf_decay`'s point masses at `0.0` and `log 2` are
> the **discreteness of an integer lag index**, not a substitution. Group G is constant under χ and is
> handled by `EmbeddedNet`'s pass-through instead.
>
> Two pairs share one flag each — `B7`'s two columns, and `E1_log_tau_fast` with `E1_w_fast` — and
> both pairings were checked rather than assumed: **the XOR fired zero times on 10.24 M rows.**
>
> **Compare in the dtype the data is stored in.** The first census cast `float32` rows to `float64`
> before testing against `math.log(1e-12)`; the two are not equal, so every count came back
> `0.000000` and the table read as "no problem here".

> ⚠ **Use the corrected ablation test (§11.2 item 1), not the addendum's.** Sweep each channel across
> its real p1–p99 range. Substituting the fitted mean gives a false negative on exactly the channels
> under investigation.

**Live channel count, for calibration of expectations:** of 42 conditioning channels, 11 are
structurally dead under χ mode, 2 are numerically annihilated, and ~5 are compressed by an order of
magnitude or worse. **The flow is conditioning on roughly 24 usable channels, not 42.**

### 11.4 Phase 2 — an informativeness scalar, BEFORE anything is judged

Every diagnostic in the pipeline (SBC, TARP, PPC coverage) measures **calibration**. There is no
scalar for **informativeness**, so every change in §11 would otherwise be judged by eye on a corner
plot. Use the expected prior-to-posterior KL, on the calibration set `validate_calibration` already
simulates:

$$I = \mathbb{E}_x\big[\log q_\phi(\theta \mid x) - \log p(\theta)\big]$$

> ⚠ **It is NOT bounded below by zero.** `E[log q − log p] ≥ 0` holds only when `q` IS the true
> posterior; an under-trained flow assigns lower density to the truth than the prior does and the
> estimate goes negative. A 5-epoch smoke train measured **−23.1 nats**, which is the honest reading
> for that posterior — worse than the prior. **Read the sign first.** Note also that a figure measured
> on the checkpoint's own TRAINING rows (35.9 nats, while smoke-testing the estimator) is
> optimistically biased and is not comparable to one from a fresh calibration set.

It decomposes per parameter **and per Fisher direction**, which is what makes it the natural partner
to `scripts/posterior_identifiability.py`. **Build it before Phase 3** — without it you cannot tell
whether Phase 1 worked, and the masked-fraction lesson (run-level SD 6.1 pp against a binomial's
~1.8 pp) already showed that eyeballing a difference of this size fails.

`core/SBI/analysis.py`.

### 11.5 Phase 3 — prior narrowing that PRESERVES amortization

The user wants **both** workflows: a broad amortized posterior for screening, then optional per-cell
refinement. Most of what sequential inference promises is available **without** giving up
amortization, because the prior is too wide in ways knowable *before* seeing any recording.

Three tiers, by defensibility, **each recorded by name in the sidecar** — a hand-narrowed box is
unfalsifiable and will be forgotten by the next run:

| tier | constraint | default | measured survival |
|---|---|---|---|
| 1 | **Physical consistency.** Derive `f_scale = N·β·k_B·T/x_scale` (addendum §5.5) | **ON** | 2.27 % of current rows |
| 2 | **Instrument.** `x_scale` from bead/photodiode calibration; `dt_exp`/`T_obs` from the protocol | opt-in | — |
| 3 | **Population/class.** The Ω₀ band the preparation actually produces | opt-in, flagged data-derived | 66.32 % (≥1 live probe) |

**Tier 1 is ON by default and is framed as a bug fix, not a narrowing.** The box currently samples the
gating-spring energy **twice** — once as `Nβ`, once as `f_scale·x_scale` — with no consistency
requirement anywhere, and `T` is the ratio between the two copies. **97.7 % of a multi-day run went to
combinations where those two disagree by orders of magnitude.** That is the largest single efficiency
lever in the project, and it is not conservatism to keep it.

**Derive `f_scale`, never `β`.** `β` is a drift parameter and the stability screen operates on the ND
block alone; deriving `β` would make the ND block depend on the rescale block and contaminate the
screen. Deriving `f_scale` confines the change to the 3-D rescale block and leaves all ten ND groups
sampled exactly as now. The nondimensionalisation is untouched — `T` is not one of the four scales,
`NadrowskiModel` never sees it, only the prior-sampling layer does.

**The UI already exists.** The Prior tab's `SourceToggle` + `BoundsGrid` gives direct box entry
("Direct entry starts FROM the file"). What is missing is *named constraints with provenance*, not
widgets.

> ### ⚠ TWO HAZARDS THAT MUST BE HANDLED BEFORE A FULL RUN
> 1. **The derived `f_scale` reaches ~10⁴ pN**, and the χ drive amplitude is `CHI_F0 × f_scale`. Any
>    feasibility guard must move to the derived value. This is a real change to the training
>    distribution, not a relabelling.
> 2. **`T`'s posterior will be its prior**, because the data adds nothing on that axis. Its SBC
>    histogram will be flat and vacuous. **Report `T` as a fixed input, never as an inferred
>    quantity**, or you will be quoting a credible interval on something you assumed.

**Scope, honestly.** Projecting the constraint-violating direction onto `V`'s columns puts most of its
weight on directions 8–12. So this tightens `N`, `β`, `x_scale`, `f_scale` and their combinations. It
does **nothing** for `k` — direction 11 is `−1.00·k` alone and the constraint has no `k` component.

### 11.6 Phase 4 — TSNPE, with the guardrails the "both" workflow requires

Yes, the addendum's §8.2 is exactly **TSNPE** (Deistler, Gonçalves & Macke 2022), correctly described,
including the property that matters: truncation is a **restriction**, not a **reweighting**, so no
proposal correction is needed — which is why SNPE-A/B/C each need one and TSNPE does not.

> ### ⚠⚠ THE ONE THING THAT MUST NOT BE GOT WRONG
> **TSNPE proposes from the TRUNCATED PRIOR, never from the posterior.**
>
> Posterior → HPD region `A` → sample θ from **prior restricted to A** → simulate → retrain. The
> posterior only ever says *where to look*.
>
> Fitting a density to the posterior and proposing from it gives `p_L ∝ L^(L+1) q` — **tempering**.
> Credible intervals contract as `(L+1)^(−1/2)` with **no new information entering**: at L=4 the
> posterior is 2.2× narrower than the data supports. **And SBC will come out flat anyway**, because it
> validates the flow against the proposal it was trained on. No current diagnostic catches this.

Guardrails, in order of how badly their absence bites:

1. **Persist `x_obs` at INFERENCE time.** Not at save time — amortized NPE has no observation when it
   is saved, which is why `default_x` is `None` on `posterior_08232026`. The TSNPE tab must refuse
   unless the stored observation hash matches the dataset currently loaded.
2. **Mark the artifact NON-AMORTIZED** and record the truncation region in the sidecar; the load path
   refuses (or hard-warns) outside it. With both workflows live, truncated and amortized posteriors
   sit side by side in one `ArtifactPicker` — **the same class of failure as the retired-band
   posterior that already cost a run (§4.1).**
3. **Truncate in the Fisher eigenbasis, not in 13-D physical space.** `k`, `delta_E` and `temp` sit at
   or near prior (§4.6); an HPD box in physical space cuts those axes **on noise**, and deleted
   support is permanent — a one-way ratchet. Truncate along directions 0–4, leave the flat ones
   full-width. This also dissolves the addendum's own §8.3 flaw 2 outright: in that basis the region
   is approximately axis-aligned, so there is no ridge for HDBSCAN to fragment and no clustering
   needed at all.
4. **Use unweighted posterior draws, not "the M best fits."** Selecting on goodness of fit applies a
   second, undeclared likelihood with the discrepancy metric as a hidden hyperparameter.
   `posterior_overlay` takes the best 50 — fine for a figure, wrong as the seed of a prior.
5. **Generous HPD (99.9 %)**, and monitor how often simulated ground truths fall outside `A`. That
   frequency is the honest failure rate.
6. **Show the cost.** Each round is a simulation campaign, not a click — reuse the Posterior tab's
   training-budget widget so the number is on screen before the button.

**The free pre-test (addendum §8.5) works for one truncation and not the other.** Retraining on the
cached rows whose θ falls in `A` is truncated NPE with zero new simulation — **66 % of the cache
survives a probe-based truncation** (viable), but only **1.25 %, ~128 k rows, survives the Tier-1
temperature constraint** — far too few for a 989 k-parameter flow. **Tier 1 cannot be tested by
subsetting the checkpoint; it needs simulation.**

Note the parent doc records `SNPE num_rounds > 1` as "refused loudly" in C-11. That is a checkpoint
compatibility constraint, not a scientific one, and truncation is not the multi-round SNPE that was
refused.

### 11.7 Deferred, and the order is not arbitrary

**§8.6 — M-replicate conditioning. Do this BEFORE designing any experiment.** Posterior width is a
convolution of genuine degeneracy with the sampling noise of single-realisation summary estimators,
and **nothing currently distinguishes them.** Simulate `M` independent passive traces per θ and
condition on the set using the DeepSet machinery already built for χ. Width falling like `M^(−1/2)` →
it was estimator noise, and longer or repeated recordings fix it. Width saturating → genuine
degeneracy, and only a new observable helps. **Run this before §11.8's protocol work**, or you risk
designing an experiment for a parameter whose width was never degeneracy.

**§7 — the absolute thermal reference (Tier 2, needs re-simulation).** Above the bundle's corner
frequency the ND bundle row gives `S_X(ω) → 2k_BT/(λω²)` with **every other parameter absent** — a
direct absolute readout of `f_scale·t_scale/x_scale`. **Every Group-B feature is a ratio**
(normalised by `f_peak` or peak power), so the entire spectral block is scale-invariant by
construction and the one place an absolute energy calibration sits in plain sight is systematically
normalised away. Proposed feature: `⟨log[ω²S_X(ω)]⟩` over ~a decade above the corner, paired with a
valid-flag for rows where the tail is unresolved. Pairs with Tier 1 — the constraint is what makes
the prefactor absolute.

**§4 — the strata hypothesis (cheap test, not yet a finding).** Every row in a batch shares one
`t_scale`, `T_k` and χ probe design, so those axes have effective sample size `TRAINING_NUM_RUNS`
(5000), not 10.24 M — a factor of `run_size` = 2048. A clean loss plateau therefore says nothing about
whether they converged. **Direction 4 sitting high in the ordering argues against it being binding.**
Test: retrain from the checkpoint with `TRAINING_NUM_RUNS` halved and `run_size` doubled (same total
rows, half the strata) and see whether `t_scale`'s SBC and posterior width degrade. Related and
already in `config.py`: `t_scale`'s SBC effective sample size is `CAL_N_SCALES = 200`, not `SBC_N_CAL`
— so "SBC flat on all 13" is strong for 11 of them and materially weaker for `t_scale` and anything
the probe design controls.

**Tier 3 — new experimental observables (§6.4). Only after §8.6.** For `k`: a **step or force-clamp
transient**, which never accumulates the ~31-drive-cycle non-stationarity wall of §4.3.1 because that
wall is a *steady-state* lock-in artifact. For `delta_E`, `S`, `N`, `φ` — the parameters carrying
§4.6's worst directions: **two-tone intermodulation** at `2ω₁ − ω₂` with both drives inside the
reproducible sub-resonance band, which structurally avoids entrainment because the measured frequency
is not driven, while IMD amplitude directly probes the nonlinearity. `chi_f0_sweep` would measure IMD
SNR with a modest extension.

**Also worth doing and nearly free:** raise supplied `K`, or lower `chi_k_pad` 12 → 6. The network was
built for 12 slots and fed 4 live probes.

### 11.8 Files and verification

| file | change |
|---|---|
| `core/SBI/embedded_network.py` | rank-Gaussianisation buffers (1.1) |
| `core/SBI/statistics.py` | valid-flags at each `clamp(min=_EPS)` (1.2); the §7 thermal feature |
| `core/SBI/pipeline.py` | winsorisation (1.3); pathological-trajectory counter (1.4); truncated-prior sampling |
| `core/SBI/analysis.py` | the KL informativeness scalar (Phase 2) |
| `core/SBI/reparam.py` | eigenbasis truncation region |
| `core/orchestrator.py` | derived `f_scale`; persist `x_obs` at inference; sidecar provenance + non-amortized flag |
| `core/SBI/Priors/prior.py` | derived-`f_scale` push-through; ND block untouched |
| `core/gui/panels/inference_tabs.py` | `_TrainingBudgetMixin` (extracted, not copied) + the gated **TSNPE tab** |
| `core/gui/screens/inference_screen.py` | the sixth tab and its gate |
| `core/SBI/derived.py` | ⊕ **new** — the tier-1 relation, both directions, and its preflight |
| `core/SBI/truncate.py` | ⊕ **new** — `TruncationRegion` + `TruncatedLatentPrior` (Phase 4) |
| `core/SBI/decorrelate.py` | `stats_no_flags`: the flags must NOT reach the Fisher (trap CHI12) |
| `core/config.py` | `RANK_GAUSS_KNOTS`, `WINSOR_PCT`, `OBSERVATION_PATH`; `SimConfig.nd_idx` / `sim_rescale_idx` / `k_b_cell` |
| `scripts/channel_ablation.py` | ⊕ **new** — the CORRECTED range-sweep ablation (§11.2 item 1) |
| `scripts/migrate_checkpoint_flags.py` | ⊕ **new** — one-shot cache widening, `V` copied verbatim |
| `tests/test_conditioning_repair.py` | ⊕ **new** — 20 tests incl. the TSNPE pinning test |
| ~~`Resources/Bounds/nadrowski/master.txt`~~ | ⚠ **NOT edited in place — see below.** Tier 1 got its own box: `master_tier1.txt` + `Cells/nadrowski/master_spont_tier1.txt` |

> **Why tier 1 did not edit `master.txt`.** Editing it changes `param_keys`, hence the checkpoint
> digest, hence the directory a Phase-1 retrain looks in — orphaning the cache that was migrated for
> exactly that run, and making `posterior_08232026` no longer reproducible. Two boxes keep Phase 1
> and Phase 3 separately runnable, which they need to be anyway: Phase 1 reuses 10.24 M cached rows
> and Phase 3 cannot (**only 1.25 % of the cache survives the temperature constraint**).

**Sort work by this line** (the addendum's, and it is right): *any change to the feature SET
invalidates the checkpoint and requires re-simulation; changes to standardisation, or to per-channel
transforms that are pure functions of stored features, do not.*

**Verification** — **seven** suites now, run directly (no pytest), `QT_QPA_PLATFORM=offscreen`,
`KMP_DUPLICATE_LIB_OK=TRUE`. §1.3 has the per-suite counts; the new one is
`tests/test_conditioning_repair.py`. **Phase 4's gate is a TEST, not a run** — it is arithmetic on a
case with a known answer and it already passes. The other three need the retrains, and §11.9 is the
runbook.

| # | gate | status |
|---|---|---|
| 1 | the corrected range-sweep ablation moves `A1_mean`/`D3` from ~1e-7 to a healthy channel's order | **baseline captured** (`A1_mean` 3.2e-7, `D3` 3.2e-7 vs healthy 1.2–2.4, 29 usable of 42, via `scripts/channel_ablation.py`); the standardiser-level improvement measures 1.166 / 0.827 on real cached rows. **Owed: the same script against a RETRAINED net** |
| 2 | the KL scalar *increases* on the same calibration set | ⬜ needs both posteriors; `validate_calibration` prints it |
| 3 | implied `T` in 280–310 K goes 2.27 % → ~100 %; mean live probes rises from 3.37/12 | ⬜ needs the Phase-3 run. The 2.27 % baseline is reproduced by `derived.implied_temperature` |
| 4 | ⚠ **the proposal is the prior restricted to `A`, not the posterior** — the round-1 credible width must **not** contract by `√2` | ✅ **passing**, `test_the_proposal_is_the_TRUNCATED_PRIOR_and_not_the_posterior`. **This is the test that catches the tempering error, and SBC cannot** |

---

### 11.9 The runbook — the two runs that are owed, and what to watch

Everything in §11 is code, tests and a migrated cache. **Neither retrain has been done.** Both need
the card, and the VRAM rule has not changed: check `nvidia-smi --query-gpu=memory.used --format=csv`,
**never** `torch.cuda.mem_get_info()`, which overstates free VRAM by the size of the desktop (measured
15037 MiB against nvidia-smi's 5814). Closing the browsers is still worth more than any constant here.

#### Run A — the Phase-1 flow-only retrain. NO SIMULATION.

The cache at `Resources/Checkpoints/train_230ae7cb5fc2` is complete (5000/5000) and already carries
the valid flags, so `gen_training_data` loads its 10.24 M rows and skips generation entirely.

> ### ⚠ THAT SAVES 20 % OF A RETRAIN, NOT "days versus hours" — corrected 2026-08-27
> This line first said "hours, not days". It is wrong, and the file timestamps of the run that
> produced `posterior_08232026` say so: header written 2026-08-21 10:48, generation complete
> 22:14 (**11.4 h**), posterior saved 2026-08-23 20:18 (**46.1 h** of flow training, 130 epochs).
> **Simulation is 20 % of the run; sbi's fit loop is 80 %.** So skipping generation turns ~2.4 days
> into ~1.9 days. It is still worth having — it is free, and it is what makes Phase 1 a clean A/B —
> but budget two days, not an afternoon.
>
> The CUDA-graph speedup (§8.3) is already inside both of those figures: it landed 2026-08-21 and
> that run started the same morning. There is no further solver speedup to bank, and none of §11
> makes training faster.

> **This was verified, not assumed.** Rebuilding the record run's `training_identity` from scratch
> resolves to `train_230ae7cb5fc2`; `training_checkpoint.verify()` — the same field-by-field check a
> resume performs — passes; and `load_rows` returns **10,240,000 x 122**, which is exactly
> `SUMMARY_WIDTH + 1 + expected_forcing_dim`. The stored `V` (13x13) and the bijection probe came
> across from `train_98aebd93ed17` untouched.

**The prior must be LOADED, not rebuilt.** `Resources/Priors/3d_master_08102026.pt` is the one whose
fingerprint (`33ecff5c70c828d8`) the checkpoint identity names; a rebuilt prior gets a different
fingerprint, a different digest, and starts a fresh 5-day simulation run instead. The run prints
`[checkpoint] resuming at batch 5000/5000` when it has found the cache — **if that line is absent,
stop it.**

```
$env:CHI=1; $env:TOBS_S=4.5
$env:BOUNDS="Resources/Bounds/nadrowski/master.txt"
$env:CELL="Resources/Cells/nadrowski/master_spont.txt"
```

Watch for, in order: the χ banner reading `K=6 F0=0.15 range=(0.03, 0.3)`; the
`[checkpoint] resuming at batch 5000/5000 … COMPLETE, so generation will be skipped` line; the
`[winsor] clipped N of M summary elements` line (expect a few tenths of a percent); and at the end the
`Informativeness (expected prior->posterior KL)` block.

> **Do NOT compare that KL against the 35.9 nats measured on cached TRAINING rows while smoke-testing
> the estimator.** That figure is on rows the flow memorised and is optimistically biased by an
> unknown amount. The gate is old-versus-new **on the same fresh calibration set**, which is what
> `validate_calibration` produces.

#### Run B — the Phase-3 retrain. FULL RE-SIMULATION, multi-day.

```
$env:BOUNDS="Resources/Bounds/nadrowski/master_tier1.txt"
$env:CELL="Resources/Cells/nadrowski/master_spont_tier1.txt"
```

Tier 1 cannot be tested by subsetting the cache — **only 1.25 % (~128 k rows) survives the temperature
constraint**, far too few for a 989 k-parameter flow. Before the first simulation the run prints
`[tier1] f_scale is DERIVED as N*beta*k_B*T/x_scale: implied range over the prior p1/p50/p99 = …` and
the χ drive amplitude those imply. **Read that line.** On the master box the corner reaches ~1.2e4 pN
of `f_scale` and ~1.9e3 pN of drive; whether that is physically reasonable is a judgement about the
preparation, which is why the banner reports rather than refuses.

> ⚠ **`T`'s posterior will be its prior, and its SBC histogram will be flat and VACUOUS.** Report `T`
> as a fixed input, never as an inferred quantity, or you are quoting a credible interval on something
> you assumed. (The cleaner alternative — fix `T = 300 K` and drop to 12 inferred dimensions — is a
> larger change and is worth raising once the 13-dim version has been measured.)

#### The gates

| # | gate | how |
|---|---|---|
| 1 | `A1_mean`/`D3` reach the same order as healthy channels | `POST=<new> CKPT=Resources/Checkpoints/train_230ae7cb5fc2 python scripts/channel_ablation.py`, against the baseline (`A1_mean` 3.2e-7, `D3` 3.2e-7, healthy 1.2–2.4, **29 usable of 42**) |
| 2 | the KL scalar **increases** | the `Informativeness` block, old vs new, same calibration set |
| 3 | implied `T` in 280–310 K goes 2.27 % → ~100 %; mean live probes rises from 3.37/12 | the `[tier1]` banner + `[ppc] chi probes: N live / K slots` |
| 4 | the proposal is the truncated prior | `tests/test_conditioning_repair.py` — it is arithmetic, not a run |

Also re-run `scripts/posterior_identifiability.py`. The new artifact carries `fisher_eigenvalues`
(added 2026-08-25), so for the first time the decomposition has a **scale** and not just an ordering —
the difference between "mildly uneven" and "nine of thirteen parameters are prior".

> **Expect `k` to stay unmeasured.** §11.2.1's reconciliation is unchanged by any of this work: the
> Fisher is computed on a **clean** standardisation (`fnoise` is locally measured single-trajectory
> noise, not the embedding's contaminated `sum_std`), so `k`'s gradient was already known to be tiny
> on clean features. Phase 1 recovers what the flow could not SEE; it cannot create information that
> was never in the observable. Gains belong in `x_scale`, `delta_E` and the absolute-scale directions.

---

# Appendix A — Change history

## 2026-08-28 — the retry died in its own recovery, 3639 batches further on

The retrain reached **batch 3990/10000**, against 351 the day before, and stopped with an
`AcceleratorError: CUDA error: out of memory` raised from `_tc.rng_restore` — inside the batch-level
retry added the previous day *to stop exactly this class of failure*.

### The same defect, in the step that fixed the same defect

`rng_restore` → `torch.cuda.set_rng_state_all` copies each generator's state into device memory, so
it is an allocation, and it was the one step of the recovery block left unguarded. But the guard is
only half the fix. **The order was wrong.**

The block ran `release → wait 15 s → restore`. `_release_device_memory` hands every cached block back
to the driver; the wait then sleeps up to three minutes on a *contended* card, during which the
desktop takes the memory; and the restore — a few KB, but a few KB **on the device** — was left
asking for a fresh `cudaMalloc` at the exact moment the process had the weakest claim on the card.
The recovery donated its working set and then asked for it back.

Restoring **first** inverts that. By then the failed attempt's tensors have been dropped (the
`except` clause has closed), so the allocator is holding them as cached blocks and a KB-sized request
never reaches the driver. Only then is it worth handing that cache back, and only then worth
sleeping. `test_the_rng_restore_happens_before_the_release_and_the_wait` pins the ordering on parsed
source.

### What the run got right, and it is worth recording

- The batch-level retry **engaged in production for the first time**: the notice printed, the release
  ran, the wait completed. It failed on its last statement, not its design.
- **`batches_done = 3989`** — the `except BaseException` rescue write fired and cost about one batch.
  That write is CUDA-free precisely because it uses the pre-captured `_pending_rng` rather than
  taking a fresh snapshot; the cadence and final writes did *not* share that property and have been
  fixed to.
- Pace was 5.86 s/batch (3989 batches in 6 h 29 m).

### Five more recovery windows, found by audit rather than by crashing

Guarding one step twice in a row moved the failure to its neighbour twice in a row, so the whole
recovery surface was walked instead:

| site | why it can kill a run |
|---|---|
| `_log_memory` (4 CUDA calls) | a **diagnostic**, on the success path every 250 batches — i.e. right after a batch clawed through all three ladders. Now best-effort; a failed read prints "statistics unavailable". |
| `config.memory_budget_elements` | bare `mem_get_info` / `memory_reserved`, and **every retry re-enters it** to re-plan. `_free_gib_note` guards the identical call. Now falls back to a deliberately small budget — the safe direction, since it only causes more splitting. |
| `rng_snapshot` ×4 (per-batch, before-retry, cadence, final) | all driver calls. The **final** one sits *after* the `finally`, outside the `try` entirely — a failure there would lose the whole run's product. All best-effort now; the cadence write defers to the next boundary rather than dying. |
| `str(e).splitlines()[0]` ×4 | `IndexError` on a zero-message exception — `"".splitlines()` is `[]`. Reachable, because `_is_oom` returns True on the *type* test alone. One of the four was inside `_release_device_memory`, the function whose docstring says it never raises. Now `_short_err`. |
| stale `_pending_rng` | `batch_k` is bound by the `for` before the snapshot, so a snapshot failing at the top of batch *k* left *k-1*'s state paired with `batch_k = k` — a resume would restart *k* with the streams where *k-1* began. Now paired with its index, and the rescue write commits the rows with **no** restore point rather than a wrong one. |

> **The rows outrank the restore point.** Every one of these fixes resolves the same way: a
> checkpoint that resumes without restoring streams draws fresh noise from that point, which is a
> reproducibility loss. A write that is skipped, or a run that dies, throws away hours of simulation.
> Those are not the same size of mistake.

### Three that were found and deliberately NOT fixed

1. **`_gen_obs_retry`'s `out = torch.empty(...)` preallocation** sits in the recovery window and is a
   multi-GiB device allocation on a card that just proved it is short. It is left to propagate, and
   that is correct: if the full result cannot be allocated, the caller must reduce the ROW count,
   which is exactly what `_rows_with_oom_retry` does when it catches this. **But the PPC bin loop
   (`orchestrator.py:1841-1877`) and `decorrelate.py` have no outer ladder**, so there the same
   failure is fatal. That is the real gap, and it is a missing ladder rather than a missing guard.
2. **`decorrelate.py:193`'s `torch.random.fork_rng(devices=[device])`** wraps ~216 Fisher evaluations
   and calls `get_rng_state_all()` on entry and `set_rng_state_all()` on exit — the identical pair
   that crashed here. Its `__exit__` restore runs *while unwinding* any exception raised inside the
   block, so an OOM in `feats()` would be followed by a device call on a starved card and the
   secondary failure would replace the original traceback. Same shape as trap X13, different
   subsystem, and not touched because the Fisher path was not what failed.
3. **`gen_training_data:1550/1560`** (the resume path's `.to(device)` and `rng_restore`) run *before*
   the `try`, so they have no rescue write. Tolerable: nothing has been generated yet, so a failure
   there costs only a restart.

### The ceiling is now settable per run

`PRISM_VRAM_CEILING_GIB` overrides `config.SIM_VRAM_CEILING_GIB` (the `PRISM_CHI_OVERRIDE` shape), so
a single run can be throttled without editing a tracked file; junk values are ignored with a note.
⚠ **It needs headroom to cap to.** With the card at 115 MiB actually free it does nothing —
`_max_sim_batch` finds that not even a floor-sized chunk fits and runs the batch as asked. What it
buys is keeping a run that *has* headroom out of the 9× shared-memory paging regime measured on
2026-08-27. Free the memory first, then set it to about `(nvidia-smi free) − 1 GiB`.

---

## 2026-08-27 (later) — the OOM retry fired for the first time, and died on its own cleanup

The Phase-1 retrain reached batch 351/5000 and stopped with a bare
`torch.AcceleratorError: CUDA error: out of memory` raised from `torch.cuda.empty_cache()` at
`pipeline.py:632`. Two questions were asked of it — "is dynamic memory management on?" and "are we
holding a finished batch's simulations in VRAM?" — and the answer to both was already yes.

### Neither suspected cause was the cause

`x_buf`/`th_buf` are **host** tensors (`torch.empty` with no `device=`), `_rows` returns CPU data,
`gen_stats` does `.cpu()` per sub-batch, and `append_simulations` is pinned to `data_device="cpu"`.
Nothing large is GPU-resident across batches. The learned budget worked too: it caught the OOM and
entered the retry. **The run died on its RECOVERY.**

`empty_cache()` sits outside the `except` block — correctly, because a release while the error is
bound frees nothing — and it raised. The pasted traceback had no *"During handling of the above
exception"* chain, which is the signature of a raise after an except clause closes, and is what
pinned the diagnosis. Worse, the notice naming the original OOM was printed *after* the release, so
the failure that started it all left **no record at all**. That is now trap **X13**.

### Five changes

1. **`pipeline._release_device_memory(device, *, plans, graphs)`** — one guarded place for all three
   reclaimable resources, and it **never raises**. `except Exception`, never `BaseException`, so a
   `WorkerCancelled` still reaches `Worker.run`. The flags exist because not every caller wants all
   three: a hot loop (`gen_stats`' sub-batches, `gen_chi_raw`'s probes) passes `plans=False,
   graphs=False`, since clearing the plan cache there destroys the intra-batch cuFFT reuse those
   loops exist to get and dropping graphs would recapture per probe against an ~8× replay speedup.
   `gen_training_data`'s per-batch tail keeps `plans=True` (cross-batch plan reuse is exactly zero)
   and `graphs=False` (the batch succeeded). Only the OOM paths take all three.
2. **The notice is printed BEFORE the release** in both ladders. Ordering, not wording.
3. **`sdeint.drop_graph_cache()`**, called on the OOM paths only. Graph memory lives in a private
   pool `empty_cache()` cannot reclaim; at an OOM up to `SOLVER_GRAPH_CACHE_MAX` are pinned and the
   halving retry is about to capture *another* at the reduced width, because `tuple(x0.shape)` is
   part of `_graph_key`. Clearing the dict is a request, not a guarantee — the pool returns when the
   `CUDAGraph` is collected — which is one more reason the release belongs outside the handler.
4. **⚠ THE LEARNED BUDGET WAS RECOVERING ~10× TOO FAST IN CHI MODE.** `_BUDGET_RECOVER_AFTER` is 32
   and every description of it, here and in the code, says *"32 clean **batches**"*. But `gen_obs`
   was what called `_budget_note_ok`, and a chi batch makes **1 + K** of those — one spontaneous run
   plus one per probe. At the production K that is ~7–12 calls per batch, so the cap probed upward
   every ~3 batches instead of every 32: an 0.8× backoff fully unwound in three batches and kept
   climbing. **The throttle never held**, which is why a busy card produced repeated OOMs rather than
   settling into a slower surviving state. `_budget_note_ok` now takes `batch_level=`, ignores
   `gen_obs`' calls while `_BATCH_TAG` is set, and `gen_training_data` credits exactly once per
   completed batch. Outside training the per-call credit is unchanged — the PPC and the prior sweeps
   have no batch to key on.
5. **A batch-level retry that WAITS instead of shrinking.** Both existing ladders answer "this is too
   big"; neither can answer the failure that actually kills runs here, which is a reasonable batch on
   a card momentarily full of somebody else's evictable surfaces. It restores the batch's opening RNG
   snapshot and re-runs — so the retried batch is **the batch that would have been produced**.
   ⚠ **Corrected 2026-08-28:** this entry originally called that a *correctness requirement*, and it
   is not. The checkpoint always records the ACTUAL state at a batch boundary (the cadence write
   snapshots fresh, the rescue write uses that batch's own opening snapshot) and rows already on
   disk are never regenerated, so a retry that skips the restore is simply a different valid iid
   draw — the same licence the halving ladders take. It is a reproducibility nicety, which is what
   makes it safe to be best-effort. Delays 15 s / 60 s / 180 s, and the wait is
   sliced and printed, because **cancellation is raised from a stream `write()` on the worker thread
   — a plain `time.sleep()` cannot be cancelled.**

### The drive was under-charged 4×, and it is now measured

§8.2 flagged `_max_sim_batch` as budgeting only `n_ch * n_fine` for the drive. Counting
`build_nondim_force_tensor`'s eager allocations gives 4R (`t_dim` + `sin_term` both live to the end,
plus two elementwise temporaries), and **measured 4.10×** on the 5070 Ti at B=64/T=20000. `_per_row`
now charges `_FORCE_BUILD_PEAK_MULTIPLE`, which matters because `_per_row` is the number
`_budget_note_oom` teaches the cap — under-counting taught it something smaller than what failed.

### What the card actually looked like

`nvidia-smi` at the time of diagnosis: **102 MiB free of 16303 MiB**, ~30 GPU-touching processes
(two Firefox, three Edge WebViews, Steam, Discord, Spotify, Parsec, TeamViewer, PyCharm's JCEF, two
Claude, OneDrive, ProtonDrive…), plus the crashed run still holding its context. The worst admissible
geometry at 2048 rows needs ~11.44 GiB against a ~12.48 GiB budget **on an idle card**. No fix to the
retry machinery changes that arithmetic — **closing the browsers is still the largest single lever**,
and `SIM_VRAM_CEILING_GIB` is how you tell the planner in advance when you cannot.

### ⚠ THE RETRY LADDERS COULD NOT BE MADE TO FIRE, AND THE REASON RESHAPES THE DIAGNOSIS

This entry set out to do the thing §"what this actually buys" says has never been done — stage a
genuine reactive CUDA OOM with a second process holding VRAM — so the ladders would stop being
tested-by-construction. **It could not be done on this machine, and finding out why was worth more
than the test would have been.**

Four attempts on the 5070 Ti, `nvidia-smi` free measured at each start:

| geometry (peak) | competing hog | `mem_get_info` | `nvidia-smi` | outcome |
|---|---|---|---|---|
| 2048 × 100k (5.34 GiB) | 13.5 GiB, idle | 13.77 GiB | **1.02 GiB** | completed 2.5 s, no split |
| 2048 × 280k (10.83 GiB) | 13.6 GiB, **churning** | 6.76 GiB | **0.30 GiB** | planner split 2048→512, completed 42.5 s |
| 2048 × 280k, planner blinded | 6.75 GiB, churning | 6.76 GiB | **0.24 GiB** | completed 60.8 s at full width |
| **4096 × 280k (21.67 GiB)** | none | 14.62 GiB | 0.20 GiB | **completed 168.7 s** |

Read the last row again: **an allocation of 21.67 GiB completed on a 15.92 GiB card**, returning a
(1, 4096, 276000) tensor that is finite throughout. Nothing OOM'd. The first attempt also explains
why an idle hog proves nothing — its pages are cold, so WDDM simply evicts them; the hog has to
*churn* to compete at all, and even then it loses.

**The mechanism is Windows' shared GPU memory.** WDDM exposes up to half of system RAM as a spillover
pool — 63.4 GiB installed here, so ~31.7 GiB of shared memory behind a 15.92 GiB board, ~47 GiB
addressable in total. A batch that cannot fit in VRAM does not fail; it **pages**, and the cost is
wall-clock: 168.7 s for a batch that takes ~18 s unpressured, a 9× penalty. The 3.4× at 42.5 s in row
two is the same effect, smaller.

**So the failure mode is not what the memory work has been assuming.** "This batch is too big" is
close to unreachable on this hardware — which is why two rounds of pressure testing never fired a
reactive retry, and why the ladders have stayed unexercised through every smoke run. What actually
killed the run at batch 351 was a **transient**: a moment when the driver could not complete a
page-in or an eviction in time, which is a function of what the desktop was doing that second, not of
the geometry. Three consequences worth keeping:

1. **Waiting is the right lever, and shrinking is not.** The batch-level retry added in this entry
   pauses and re-runs the same batch; the halving ladders answer a question the hardware rarely asks.
   That ordering is the opposite of what the 2026-08-10 and 2026-08-11 entries assumed.
2. **A "slow run" and a "nearly-OOM run" are the same event here.** If generation is crawling, the
   card is spilling to shared memory — that is the signal to close things, and it arrives long before
   any OOM does. The `[mem]` line's *peak reserved* is what shows it.
3. **`SIM_VRAM_CEILING_GIB` earns its place for a reason other than the one it was added for.** Its
   value is not avoiding a hard OOM — it is keeping the batch inside real VRAM so the run does not
   quietly drop into the 9× paging regime.

**What is and is not verified, stated plainly.** The *predictive* guard is now exercised on real
hardware under real pressure (row two: it split 2048→512 unprompted and the batch completed). The
*reactive* ladders are still exercised only by injected failures — but that injection set now
includes the exact 2026-08-27 shape, a release that raises inside the handler
(`test_the_release_path_survives_a_failing_empty_cache`). Reproducing a real reactive OOM would need
a second process that can defeat WDDM paging, and this one could not.

### ⚠ AN UNSAVED PRIOR ORPHANS ITS OWN CHECKPOINT, PERMANENTLY

Found while checking whether the crashed run could resume. Three checkpoints from that day:

| dir | batches | `prior_fingerprint` | prior file |
|---|---|---|---|
| `train_230ae7cb5fc2` | 5000, complete | `33ecff5c70c828d8` | `3d_master_08102026.pt` (75 comp.) |
| `train_85226b80f5d1` | 884 | `bd307c079d14db0b` | **none — never saved** |
| `train_005ff030d387` | 351 (the crash) | `d33fd505672d7b25` | `prior_08272026.pt` (98 comp.) |

`training_identity` fingerprints the prior's fitted GMM and that fingerprint names the checkpoint
**directory**. A prior that was fitted but never written to disk therefore produces a directory
nothing can resolve again once the process exits — so 884 batches of simulation are unrecoverable,
not merely awkward to resume. `build_posterior` now refuses to start a run of ≥100 batches from a
prior whose fingerprint matches no file in `Resources/Priors` (`_assert_prior_is_saved`). Silent for
short runs and for checkpointing-off, which is where the tests and the smoke train live.

The crashed run *is* resumable: load `prior_08272026.pt` rather than rebuilding it, keep
`n_runs=5000`, and the log prints `[checkpoint] resuming at batch 351/5000`. **If that line is
absent, stop.**

> **A process note, recorded because it cost real work.** Unwinding a bad edit with
> `git checkout -- <file>` discarded ~70 lines of *uncommitted* work in `tests/test_user_sbi.py` that
> the session-start `git status` had listed as modified. They were reconstructed from this document —
> which names the lost tests and what they must assert — and all five pass against the working tree,
> with the `ast.unparse` comment-stripping verified to be load-bearing (`nadrowski_prior.py`'s
> comment about the fix contains the string the check forbids). **Use `git stash`, not
> `git checkout --`, and read `git status` before touching a file you did not create.**

---

## 2026-08-27 — the sweep knobs came out of hiding, and the flood-fill had been on the CPU all along

Asked for two fields on the Prior tab. Finding out what they should be bound to turned up a knob that
was not a constant at all, a second that was never threaded, and a hardcode that made the dominant
cost of every prior build ~18× slower than it needed to be.

### Two of the four "constants" were not constants

- **`n_max` was the literal `175000` inside `pipeline.gen_prior`**, silently overriding
  `construct_prior`'s own `n_max=200000` default. Nothing named it; nothing could change it.
- **`step` (the flood-fill's random-walk stride) was never threaded through `gen_prior` at all**, so
  `construct_prior`'s default won no matter what any caller asked for.

Both are now `config.PRIOR_SWEEP_MAX_SETS` / `PRIOR_SWEEP_STEP` and both arrive as arguments — the
same fix C-7 applied to `num_iterations`, one layer further down.

### ⚠ THE LOCAL SWEEP HAD BEEN PINNED TO THE CPU IN ALL FOUR PRIORS

`_local_map` was a **`@staticmethod`** in `NadrowskiPrior`, `BPPrior` and `HopfPrior` — so it could
not see `self.device` — and every one of them opened with
`dtype = torch.float32; device = torch.device('cpu')`. `UserPrior`'s was an instance method and
hardcoded it anyway. Meanwhile `_global_map` ran on `self.device`, i.e. the GPU. **The global census
was accelerated and the flood-fill it feeds was not**, which is backwards: the flood-fill is the
larger half of the work.

Measured on the real inner loop (1024 trajectories × 40,000 steps, the local sweep's actual shape):

| | per iteration |
|---|---|
| CPU | **6.32 s** |
| CUDA | **0.357 s** |
| | **17.7×** |

End to end on a small prior build: **9.01 s → 2.83 s**. All four are instance methods now and
simulate on `self.sweep_device`, resolved once by `prior.resolve_sweep_device` and **falling back to
the CPU with a printed note** when CUDA is unavailable — verified by forcing
`torch.cuda.is_available` False, which completes on the CPU rather than raising.

> **The accept loop had to be vectorised in the same change, or the move would have paid for
> nothing.** It walked `for i in range(batch_size)` doing `is_valid[i]` and `thetas[i].tolist()` per
> row — free on the CPU, and **2048 device-to-host syncs per iteration** on CUDA. It is now one
> `thetas[is_valid].detach().cpu().tolist()` per iteration.

### What is worth knowing before turning any of these

- **Candidates-per-round is NOT a speed dial.** The global sweep is ITERATION-bounded, so shrinking it
  makes the prior worse *without* making it faster — 527 s at 2048 against >70 min and unfinished at
  32 (C-7). The field carries that warning.
- **Stability duration changes what "stable" MEANS**, so it moves the prior's support, not just the
  wait. A longer screen rejects slow instabilities a short one accepts.
- **`CAL_N_SCALES` is `t_scale`'s effective sample size** (trap X5), so lowering it is a *different*
  measurement, not a faster one. "SBC flat on all 13" is strong for 11 of them and materially weaker
  for `t_scale`, and this number is why.
- **Max accepted sets buys COVERAGE, not precision.** It is the point cloud HDBSCAN clusters and the
  GMM is fitted to; a 10-D GMM with a few components needs nothing like 175,000 points.

### Two more that were asked for after the first pass, and one of them is not a tuning dial

**HDBSCAN's `min_cluster_size=50` and `min_samples=10` were literals in `construct_prior`.** They are
not a clustering preference: HDBSCAN's label count is handed **straight to the GMM's `n_components`**,
so they decide how many MODES the prior has. Measured on a small build, changing nothing else:

| `min_cluster_size` | GMM components |
|---|---|
| 5 | **15** |
| 60 | **1** |

A prior with a different component count is a **different prior**, not a faster one — which is why
they get their own "Clustering / GMM" group on the Prior tab rather than joining the sweep fields.
`min_samples` is the other half: higher declares more points NOISE, which HDBSCAN leaves unassigned
and the GMM therefore never sees, so it thins the cloud as well as splitting it.

**`REPARAM_FISHER_M` / `_DZ` / `_POINTS` needed no plumbing at all** —
`build_latent_fisher_rotation` has accepted `m`, `dz` and `n_points` as optional arguments all along;
`build_posterior` simply never passed them. Now on the Posterior tab beside the flow capacity.

> ⚠ **A RESUMED run ignores all three.** It reuses the checkpoint's stored `V` and skips the Fisher
> entirely — trap **X10**, and that is a correctness requirement, not an optimisation: `V`'s operating
> points are not reproducible across processes, so a fresh one would express the reused rows in a
> different coordinate than the targets stored beside them. Turning these knobs only does anything on
> a run that computes its rotation.

### The flow knobs are on the Posterior tab because the checkpoint is a cache

C-11's own docstring says a completed checkpoint *"is a cache of the whole simulation run, so you can
retrain the flow at a different capacity/learning rate without re-simulating"* — and until now there
was no way to do that without editing `config.py`. Hidden features, transforms, learning rate and
early-stop patience are now fields. The economics: **~46 h to re-train against a cache, against ~57 h
for a full run** (measured from the 2026-08-25 run's own file timestamps — 11.4 h of generation,
46.1 h of training). And §4.6's "more capacity will not help" was measured on the **broken**
conditioning; that verdict is worth re-testing after §11's Phase 1, not inheriting.

### Every knob is an ARGUMENT, and there is a test that says so

`orchestrator` does `from .config import PRIOR_SWEEP_ITERATIONS, NSF_HIDDEN_FEATURES, …`, which binds
all of them at IMPORT — so a panel that "configured" a run by assigning to the constant would change
nothing and the run would use the default with nothing to say so. That is the trap the training
budget was made a parameter for (§6 row 5), and it now applies to eleven more.
`test_the_sweep_and_flow_knobs_are_ARGUMENTS_because_the_constants_are_snapshotted` checks the
signature AND that each knob reaches the thing it configures, because accepted-and-dropped is the
failure mode a signature check alone misses.

> ### ⚠ A STUB INSTALLED OVER A FUNCTION MUST TOLERATE ITS NEW ARGUMENTS
> `tests/test_user_sbi.py` monkeypatches `pipeline.gen_prior` with tiny stand-ins, and there are
> **two** of them. Adding `n_max`/`step` upstream killed the suite with
> `TypeError: _tiny_nadrowski_gen_prior() got an unexpected keyword argument 'n_max'` -- deep inside
> `build_prior`, ~40 tests in. Fixing the one named in the traceback then hit the SECOND stub on the
> next run and cost another hour. Both now take `**_kw`.
>
> That is deliberately NOT a lost contract check: mirroring the signature by hand was never checking
> anything useful (it failed as a TypeError, not an assertion), and `gen_prior`'s real signature is
> asserted directly by `test_n_max_and_step_are_no_longer_hidden_inside_gen_prior`.

> **A test-writing note that has now cost time twice.** The check that `_local_map` no longer
> hardcodes the CPU failed on its first run — against the **comment that documents the fix**, which
> necessarily contains the string it forbids. Same false positive as the TSNPE runner check. Both now
> compare `ast.unparse(ast.parse(source))`, which drops comments and docstrings by construction. If
> you assert on source text, parse it first.

---

## 2026-08-26 — §11 implemented, and the measurement that set its width was not the one §11 assumed

All four phases of the informativeness programme landed. **Nothing has been retrained yet** — the
code, the migrated cache and the gates exist; the two runs that need the card are handed off.

### The premise was checked before any of it was written, and it holds exactly

§11's whole economy rests on "Phase 1 needs no re-simulation". That was verified rather than assumed:

- `Resources/Checkpoints/train_98aebd93ed17` reports `complete: True, 5000/5000 batches` at
  `chi_f0 = 0.15`, band `(0.03, 0.3)` — the run behind `posterior_08232026`.
- `Resources/Priors/3d_master_08102026.pt` reproduces `prior_fingerprint 33ecff5c70c828d8`, the value
  in both checkpoint identities.
- **Rebuilding the retrain's `training_identity` from scratch reproduces the digest `train_98aebd93ed17`
  exactly** — so the cache is genuinely reachable by a real run, not merely present on disk.

The two §11 claims that Phase 1 targets were also reproduced from the artifact itself, off the
trained net's own buffers: `A1_mean` fitted at **std 4.19e11** against a physical range of ~1e3, and
`D3_bimodality` at **std 4.42e8** against a range of (0, 1]. `elem_std` matched §11.1's ledger to four
decimals and the parameter count to the digit.

### ⚠ THE SENTINEL PROBLEM IS SIX CHANNELS, NOT ONE — AND THAT IS WHAT SETS THE NEW WIDTH

§11.3 item 1.2 is written as though `B1_log_Q` were the case. A point-mass census over **all
10.24 M rows** — each column's most frequent value found empirically, then counted exactly — says
otherwise:

| channel | substituted | | channel | substituted |
|---|---|---|---|---|
| `B1_log_Q` | **69.66 %** | | `E1_log_tau_slow` | 8.01 % |
| `C7_log_slowenv_relvar` | **30.04 %** | | `E1_w_fast` (fast decay gone) | 7.53 % |
| `B7` (no secondary peak, both columns) | **14.49 %** | | `E2_log_h3` | 5.51 % |
| `C6_log_slowenv_corrtime` | 5.07 % | | `E2_log_h2` | 2.83 % |

So the valid-flag block is **8 channels**, not 1, and the conditioning goes `42 + 72 = 114` →
`50 + 72 = **122**`. `C2_log_env_cv` — the addendum's predicted 5.6 % sentinel, which §11.2 item 2
already flagged as a false positive — measures **exactly 0 rows in 10.24 M**. `A4_log_acf_decay`'s
point masses at `0.0` (7.7 %) and `log 2` (9.1 %) are the **discreteness of an integer lag index** at
short correlation times, not a substitution, and earn no flag.

> **A measurement bug worth recording, because it fails silently and reads as good news.** The first
> census cast the stored `float32` rows to `float64` before comparing against `math.log(1e-12)`. The
> two are not equal, so **every sentinel count came back 0.000000** — a clean-looking table saying the
> problem does not exist. Compare in the dtype the data is stored in.

### Two co-occurrence assumptions, each checked on all 10.24 M rows rather than argued

One flag covers `B7_log_sec_freq_ratio` and `B7_sec_height_ratio` together, and one covers
`E1_log_tau_fast` and `E1_w_fast`. Both pairings are only legitimate if the members never disagree:
measured, **the XOR fired zero times on either pair**. `E1`'s flag is keyed on `w_fast == 1` rather
than on `log_tau_fast`'s value, because that value also carries the lag index `l1` and is only
*incidentally* constant — the two agreed on every row, and the chosen one stays true if the lag clamp
ever moves.

### The flags must not reach the Fisher, and the reason was already written down

`gen_stats` is what the Fisher's `feats()` calls, so widening it silently widened the Jacobian by 8
**binary** rows. `decorrelate` standardises by `fnoise = max(f0.std(0), 1e-9)`, and a flag that
happens to step between the ±δz arms divides 1 by that floor. That is trap **CHI10** exactly, and the
comment three lines above the call site already says it: *"It is the NEARLY constant channel that is
lethal, not the constant one."* The three Fisher call sites now route through `stats_no_flags`.

### Phase 1, and the gate it has to clear

`scripts/channel_ablation.py` is new, and it is §11.2 item 1's **corrected** test — sweep each channel
across its real p1–p99 range, never substitute the fitted mean (which is itself contaminated and sits
~1e7 from the data, which is why the addendum's own version concludes the dead channels are alive).
Run against `posterior_08232026` it reproduces the finding:

```
A1_mean         3.2e-7      D3_bimodality  3.2e-7      <- 1.09e-6x a typical channel
A2_log_var      1.228       A3_log_fpeak   2.374       <- healthy; median non-constant 0.295
verdict: 11 structurally dead, 2 numerically invisible, 29 usable of 42
```

**The two dead channels reporting the same number to three figures is itself the tell** — that value
is the network's own float noise, identical because neither channel moves the output at all. The base
point is the real row nearest the standardised median (2 live probes of 12), not a median VECTOR: a
median vector's χ block is all-zeros, i.e. an observation with no live probes, which no recording can
produce and which puts the set encoder on its n=0 path.

Under rank-Gaussianisation the same sweep on real cached rows moves `A1_mean` to **1.166** and `D3`
to **0.827**, against `A3_log_fpeak` 1.152 and `B4_spec_entropy` 1.024. **The gate clears by a factor
of ~1e6.** Two caveats, and they matter: that figure is on an untrained net, so the meaningful part is
the *parity across channels*, not the absolute number; and the real comparison is old-trained against
new-trained, which is the retrain that has not been run.

Two defects in the script were found by reading its own output rather than by testing: on a pre-flag
posterior it labelled `logT` as `V_B1_Q` (slicing the new label list against the old layout), and its
"invisible" verdict keyed on an absolute float32 ulp, which misses a channel that moves the embedding
a millionth as far as its neighbours while still clearing an ulp. Both now key on the layout the net
actually carries and on the ratio to the median.

### The cache was migrated rather than the guard loosened

`summary_flags` joins `training_identity`, so the digest changes and the pre-flag checkpoints are
orphaned — which is C-11's identity doing its job, not an obstacle. `scripts/migrate_checkpoint_flags.py`
derives the flags from the cached values with **the same function the live pipeline calls** and writes
a new directory (`train_230ae7cb5fc2`, 5.2 GiB, 102 shards, every one bitwise-verified after the
write). `V` and the bijection probe are copied verbatim: trap **X10** again — a recomputed `V` would
express the reused rows' latent targets in a different coordinate than the targets stored beside them.
The source is untouched, so `posterior_08232026` stays reproducible.

### The migrated cache was then checked the way a resume would check it

Not "the files are there": the record run's identity was rebuilt from scratch, resolved to
`train_230ae7cb5fc2`, and put through `training_checkpoint.verify()` — the field-by-field check the
resume itself performs. It passes, `load_rows` returns **10,240,000 x 122** (exactly
`SUMMARY_WIDTH + 1 + expected_forcing_dim`), and the stored `V` and probe are the source's.

### Phase 3 reproduced §11.2's numbers before changing anything

`derived.implied_temperature` over 1.02 M real cached θ gives quantiles **14 / 81 / 287 / 945 / 6318 K**
against §11.2's 14 / 80 / 286 / 939 / 6292, and **2.273 %** in 280–310 K against 2.27 %, spread 447×
against 450×. The formula and the units are right: `k_B` is derived from the units file rather than
baked, and `k_B·T` at 300 K comes out **4.1419 pN·nm**.

> ### ⚠ §11.2's CORRECTION 3 GIVES THE WRONG REASON, AND THE REASON IS WHAT PHASE 3 WOULD BE BUILT ON
> It says `x_scale`/`f_scale` "are not log-uniform: `REPARAM_LOG_PARAMS = []`". `REPARAM_LOG_PARAMS`
> governs only the **ND** block's box coordinate. The rescale block is built by
> `_build_rescale_prior`, whose rule is `"scale" in name → "log-uni"` — and sampling it directly
> returns log-medians of 3.451 / 1.844 / 3.457 against a log-uniform's 3.454 / 1.844 / 3.454. **They
> ARE log-uniform in physical space.** The measured quantiles and the 2.27 % stand (they were taken on
> real rows) and Phase 3's variance argument is untouched, but anyone re-deriving a survival fraction
> from the stated cause will get it wrong.

**One deliberate deviation from §11.8's file list.** It says to edit `Bounds/nadrowski/master.txt`
in place (`f_scale` → `T`). Doing that would change the Phase-1 retrain's identity and orphan the
cache that was just migrated for it. Tier 1 therefore gets its own box —
`Bounds/nadrowski/master_tier1.txt` + `Cells/nadrowski/master_spont_tier1.txt` — so Phase 1 and
Phase 3 are separately runnable and `posterior_08232026` stays reproducible.

The derived `f_scale` spans **0.2 → 12426 pN** over the box corners, which is §11.5's ~1e4 hazard
confirmed by arithmetic. It is announced by a preflight banner rather than bounded by a new threshold:
whether 1.9e3 pN of drive is reasonable is a judgement about the preparation, not something a constant
in `config.py` should decide.

### Phase 4, and the test SBC cannot do

The whole of TSNPE's risk is one sentence — **the proposal is the truncated prior, never the
posterior** — and the failure is invisible: tempering contracts credible intervals by `(L+1)^(-1/2)`
with no new information, and SBC comes out flat because it validates the flow against the proposal it
was trained on. So the pinning test is arithmetic on a case with a known answer: a wide prior, a
narrow posterior, a region, and an assertion that the proposal's width is the **prior's** over that
region — explicitly not the posterior's (1.0) and not the posterior's over √2 (0.707).

Guardrails 1–6 are all in: `x_obs` persisted at **inference** time (an amortized posterior has none at
save time, which is why `default_x` is None on `posterior_08232026`); the artifact marked
`amortized: False` with its region in the sidecar and a load path that refuses it for general
inference; truncation along the leading Fisher directions only, leaving the flat ones full width;
unweighted draws; a 99.9 % default with the kept-mass fraction printed; and the cost on screen before
the button, via the Posterior tab's budget widget — **extracted into `_TrainingBudgetMixin` rather
than copied**, because a second `_budget_memory` would be a second place for `pipeline`'s cost model
to be restated wrongly.

### Four defects the suites found in the new code, and the two that matter are the last two

- `count_pathological` counted an all-NaN row as **both** non-finite and "exactly constant", because
  `nan_to_num` flattens it to zeros and `amax == amin` then agrees. An all-NaN batch would have
  reported a flatline population that was not there. Non-finite rows are now counted only as
  non-finite; `constant` and `overflow` still overlap on purpose, since an all-1e29 row is genuinely
  both.
- One of the new tests contained `assert ... or True`, which is vacuous — the same defect this project
  caught in its own fixtures on 2026-08-06. Replaced with a bounded assertion on the fraction of
  elements a 0.1/99.9 clip is supposed to move.
- **⚠ PYTHON EVALUATES CALL ARGUMENTS EAGERLY, AND A GUARD INSIDE THE CALLEE DOES NOT HELP.**
  `to_sim_rescale` returns early for a box that declares `f_scale`, so passing
  `cfg.nd_idx, cfg.k_b_cell` into it looked free. It is not: `k_b_cell` needs a FORCE unit in the
  units file and raises without one. `test_no_forcing_user_model_full_sbi_pipeline` died inside the
  Fisher rotation with *"No unit with dimensionality [mass] * [length] / [time] ** 2 found in the
  units file"* — a user model, no force token, tier 1 not even on. Now `SimConfig.tier1_args`
  returns `(None, None)` off-tier and every call site passes `*cfg.tier1_args`. **This is the one
  that a green-looking six-suite run would have shipped**: the five fast suites never build a
  rotation.
- **The observation record turned the test suites into writers.** `infer_and_visualize` persisting
  `x_obs` is right for a real run and litter in a test, and nothing else in the suite writes into
  `Resources/` — a property worth keeping. Now `orchestrator.PERSIST_OBSERVATIONS`, rebound to False
  by the suite exactly as `TRAINING_CHECKPOINT_EVERY` already is, and for the same reason. It is not
  a user-facing switch: a real inference always records, because TSNPE keys on the digest.
  *It also made a new test flaky in the honest way* — the TSNPE gate test cleared the picker's combo,
  but `refresh_local_gates` repopulates it from disk, so the assertion held only while
  `Resources/Observations/` happened to be empty and flipped after any run that had inferred. The
  test now drives the picker's accessor instead of the filesystem.

### The smoke train on the card — clean, and what it says about the masked fraction

All four stages, **624 s** (prior 1 s from a loaded prior, posterior 549 s, validate 21 s, infer 52 s),
**zero OOM retries**, banner reading `chi: K=6 F0=0.15 range=(0.03, 0.3)` and
`conditioning width = 41+8 + 1 + 72 = 122`. The new instruments all reported: `[patho] run total: 0
pathological of 40 simulated trajectories`, an `Informativeness` line, and
`[obs] observation persisted as obs_…` — guardrail 1 working end to end. The PPC's decomposition read
`6 live / 12 slots (6 never filled, 0 masked); 12 of 50 base features flat`.

A smoke train on the tier-1 box was also clean — all four stages, **847 s**, zero OOM,
zero pathological trajectories — and its preflight is the number Phase 3 turns on:

```
[tier1] f_scale is DERIVED as N*beta*k_B*T/x_scale: implied range over the prior
        p1/p50/p99 = 14.2 / 683 / 6.86e+03, min 2.01 max 1.05e+04   (cell force units, pN)
[tier1] chi drive amplitude = CHI_F0 * f_scale = 2.13 / 102 / 1.03e+03 at those quantiles
```

**The median derived `f_scale` is 683 pN against the retired box's log-uniform median of ~32 pN** — a
~20× shift in the drive the training set actually sees. That is §11.5's "a real change to the
training distribution, not a relabelling", quantified.

> ### The masked fraction, and why one run cannot answer it
> **Three** smoke trains, and they are the cleanest demonstration of this rule the project has:
>
> | run | masked | |
> |---|---|---|
> | Phase-1 box, pre-audit | 106/180 | **58.9 %** |
> | tier-1 box | 197/704 | **28.0 %** |
> | Phase-1 box, post-audit | 288/704 | **40.9 %** |
> | **probe-weighted** | **591/1588** | **37.2 %** |
>
> Individually they span 28–59 %. Pooled they land on **37.2 %**, against a ~36–37 % reference. Any
> one of the three, read alone, would have supported a different conclusion.
>
> The 58.9 % reading is a DRAW, not a regression, and there are two independent reasons to say so.
> First, the χ path is byte-unchanged by this work: `chi.py` is untouched, the diff has no hunk inside
> `gen_chi_raw`, and `sim_ridx` is the identical dict when tier 1 is off. Second, the spread is
> exactly what `smoke_train`'s own docstring warns about — the effective n is the BATCH COUNT, here 4,
> and one batch (`t_scale=37.29, T=1516`) masked 36/40 by itself. `SEED` does not make runs comparable
> either, because initial conditions come from numpy's unseeded RNG (trap X8). **Compare a
> probe-weighted mean over a few runs, never a single figure** — this pair is the cleanest
> demonstration of that rule the project has.

### The audit pass, and it found nine more — read this before trusting anything above

Everything above was verified INCREMENTALLY as it was written: compile checks, targeted measurements
against the artifacts, seven suites, two smoke trains. That process caught four defects, which is
already a reason not to stop there. A deliberate read-through of the new code afterwards found
**nine more**, five of them real bugs. The lesson is the boring one: *incremental verification finds
what the tests were pointed at; only re-reading finds what nobody thought to point them at.*

**Three that would have produced wrong results or lost work.**

- **The TSNPE round's result was thrown away.** `TSNPEPanel._round` dispatched with `provide_fig_sink`
  and **no `on_result`** — the round would simulate, train for hours, and the posterior would never
  reach the session. It now installs the posterior, its latent, `V`, and the region.
- **A TSNPE posterior would have been saved marked `amortized: True`** — guardrail 2 failing at the
  one seam it is hardest to see. The GUI trains with `save=False` and saves LATER from a button, so
  the region has to survive on the session; it did not, and `_save_posterior` called
  `save_posterior_artifacts` without it. The result would be a non-amortized artifact sitting in the
  same `ArtifactPicker` as the real ones with nothing to tell them apart — the retired-band failure
  class exactly. **The mislabelling ran in both directions:** training an ordinary posterior after a
  round would have inherited that round's region and been saved non-amortized. `SbiSession` now
  carries `truncation`/`x_obs_digest`, `_on_posterior` CLEARS them, and
  `test_a_tsnpe_posterior_cannot_be_saved_as_amortized` pins all three halves (verified to fail
  against the old call).
- **The informativeness decomposition characterised one or two strata, not the prior.** It looped
  `range(n_d)` over the calibration set — and `gen_cal_data` returns rows in generation order with
  every row in a batch sharing one `(t_scale, T)` operating point. The first 256 rows are a handful
  of `t_scale` values. Trap **X5**'s class, in a new place. Now evenly spaced indices.

**Two that would have hurt the runs this work exists to enable.**

- **`winsorize_summary_block` tripled the host peak.** It built a clipped COPY of the summary block
  and `torch.cat`-ed it back, holding the original, the copy and the concatenation alive together:
  **~12 GiB against a 5 GiB tensor** — reinstating the very peak C-11's preallocated accumulators
  were introduced to remove, at the very point in the run they removed it from (right before
  `append_simulations`, at the end of a multi-day generation). Now `clamp_` in place on the summary
  view, returning the same tensor.
- **`count_pathological` allocated three trajectory-sized tensors per call.** `isfinite(x).all(-1)`,
  `nan_to_num(x)` and `safe.abs()` are each 492 MB at the production shape, twice per batch, on a
  card that has already died of OOM twice (traps X6/X7) — the cheapest item in §11 was the most
  expensive thing in the batch. Now two `(rows,)` reductions: `amax`/`amin` propagate NaN and ±inf,
  so `isfinite(mx) & isfinite(mn)` was **verified equal** to `isfinite(x).all(-1)` on NaN, +inf,
  -inf, constant and ordinary rows.

**One that would have been slow and alarming rather than wrong.** `TruncatedLatentPrior` seeded its
blind first pass at a 1e-3 acceptance floor, so it asked for `(n / 1e-3) × 1.3` draws before it had
measured anything — **2.66 million rows** for a 2048-row batch, out of a 13-D GMM. It now probes at a
sane rate and sizes the rest from the measured one, with a cap on any single draw.

**Three where the code was right and what it CLAIMED was not** — which in this document's terms is
the same kind of defect:

- **The ablation gate's base point was a median VECTOR, not an observation.** Under χ most probe-slot
  columns are zero in most rows, so their medians are zero and the composed row has an all-pad probe
  block — no live probes at all, which no recording can produce and which puts the set encoder on its
  `n=0` path. A gate that decides whether to accept a multi-day retrain should not be evaluated
  outside the data. It now uses the real row nearest the standardised median (2 live probes of 12).
  **The finding is unchanged and sharper**: `A1_mean` and `D3` both come out at **3.2e-7**, and the
  two agreeing to three figures is itself the tell — that number is the network's own float noise,
  identical because neither channel moves the output at all.
- **`per_param` is an entropy REDUCTION, not a marginal KL.** `H(prior_j) − E_x[H(posterior_j|x)]`
  measures narrowing; a marginal that shifts without narrowing has positive KL and zero entropy
  reduction. It is the right quantity for §4.6's complaint, which is width — but it was not named
  that, and it can go negative.
- **The valid flags are meaningless on a pathological row.** `compute_statistics` ends in
  `nan_to_num(..., nan=0.0)`, so by the time `derive_valid_flags` sees a feature, `0.0` may mean
  "genuinely zero" or "was NaN" — a fully non-finite trajectory arrives looking finite and its
  `_logp` channels read 0.0 rather than the sentinel, so their flags say VALID. Not fixable at that
  layer; the defence is `count_pathological`, which is exactly why item 1.4 belongs in the same phase
  as 1.2. **Read the `[patho]` line before trusting a flag histogram.**

Also fixed in passing: `TruncatedLatentPrior.__getattr__` raised `KeyError` where Python requires
`AttributeError` (it is consulted before `__init__` during copy/pickle, so this broke copying);
`_entropy_1d` returned a silent NaN when the spacing window was wider than the sample; and
`informativeness` omitted two keys on one of its two return paths.

> **And fixing the winsorise made two of MY OWN tests vacuous** — both compared `out` against `data`,
> which after an in-place clip is the same object, so both asserted nothing. That is the third time
> this defect class has appeared in this project's fixtures (2026-08-06, and the `or True` above).
> They now snapshot with `.clone()` first, and one of them asserts `out is data` so the in-place
> contract is pinned rather than assumed.

### Verification

`test_gui_progress.py` 95 → **97**, plus a new suite `tests/test_conditioning_repair.py` (**23**,
seconds, pure torch). The other five are unchanged in count and green. The GUI test checks the TSNPE
runner's **code** with its docstring stripped by `ast` — a naive text search for "proposal" flags the
very comment that documents the rule, which is how that test failed first time.

**Re-verified after the audit:** all seven suites green (266), and a THIRD smoke train on the card —
all four stages, 599.1 s, zero OOM, zero pathological trajectories, and an observation digest
byte-identical to the first run's, which is what it should be for the same cell and `T_obs`.

**Not verified, and owed:** the two runs that need the card. The Phase-1 flow-only retrain against
`train_230ae7cb5fc2` (no simulation), and the Phase-3 retrain (full re-simulation, multi-day). Also
not verified: the pixels of the new TSNPE tab — §6 row 3's standing rule.

---

## 2026-08-25 (later) — the addendum, checked against the artifacts rather than read

A user-written analysis (`PRISM_HANDOFF_ADDENDUM_2026-08-25.md`, kept on the Desktop) proposed a
diagnosis for `posterior_08232026`'s uninformativeness and an ordered work queue. It is an unusually
good document *because it is falsifiable*: it states its numbers and says where they came from. So it
was not read and judged — **every checkable claim was run against the artifacts.** The result is §11.

### The thing that made this possible was already on disk and nobody had used it

**Both complete 5000×2048 C-11 training checkpoints survive**, holding all 10.24 M conditioning
vectors and their θ. `train_98aebd93ed17` is the one this posterior trained on — proved by applying
`train_nn`'s `abs<1e15` guard to its rows and reproducing the stored `sum_mean`/`sum_std` **to every
digit**. That turns most of the addendum's §2 and §3 from inference into measurement, and it is the
reason Phase 1 of §11 needs no re-simulation.

### What held

Most of it, exactly. All the embedding buffers; the 11 pass-through channels; `elem_std` to four
decimals; 989,312 parameters; the whole drift/diffusion table in §6.2, verified line-by-line against
`nadrowski_model.py`. Two deserve singling out:

- **`k = 2.000` contaminated rows, predicted from `σ²/μ` alone with no access to the training data.
  Measured: exactly 2 surviving rows, at exactly 1e12.** The mechanism is sharper than the addendum
  states — `_group_d` computes `z = (s-mu)/std.clamp(1e-12)`, so an **exactly constant** trace gives
  `z ≡ 0`, `kurt = 0`, clamp fires, `bimod = 1/_EPS`. A flatlined simulation, not an underflow.
- **`B1_log_Q` sentinel fraction predicted 0.699, measured 0.697** over 1.02 M rows.

### The four that did not, and one of them matters a great deal

**The addendum's own "decisive test" is the wrong test and would have refuted its own best finding.**
Task 0.1 says to replace a channel with its fitted mean and expect exactly zero change from a dead
one. But the fitted mean is *itself contaminated* and sits ~10⁷ from the data, so substituting it is a
LARGE move — measured 5.2e-5 and 4.6e-4, not zero. Run it as written and you conclude the channels are
alive. **Sweeping the real p1–p99 range instead gives 1.8e-7 and 8.9e-8 against ~1.4 for healthy
channels — a factor of 10⁷, and below one float32 ulp.** The conclusion was right; the test was not.

Also: §3.2's "the fits are exact, which is itself strong evidence" is an identity, not evidence (two
parameters, two moments, closed form) — and it produced a false positive, `C2_log_env_cv` predicted
5.6 % sentinel and **measured 0.000 %**. And §5.2 assumed log-uniform sampling where the box is
all-linear: implied median bath temperature is **286 K, not 15 K**. *The spread claim survives and is
what matters* — only **2.27 %** of rows land in 280–310 K.

### The reconciliation that decides what to expect

The addendum argues §4.6's "identifiability limit" verdict is invalid because the conditioning is
broken. Checkable, and the answer is no: `decorrelate.py` standardises its Jacobian by
`fnoise = max(f0.std(0), 1e-9)` — **locally measured single-trajectory noise, not the embedding's
contaminated `sum_std`.** So the Fisher, `V` and §4.6's decomposition run on clean features and are
not artifacts of the bug. **Both findings are true and independent, which bounds what Phase 1 can
buy: expect gains in `x_scale`, `delta_E` and the absolute-scale directions; do not expect `k`.**

### One phenomenon behind three unrelated symptoms

`D3`'s contamination is *exactly constant* traces. `A1_mean`'s is ~1e32-magnitude traces (measured min
−4.59e32). The missing PSD band earlier the same day was divergent draws. **One population of
pathological simulations, damaging three unrelated things silently, and nothing in the pipeline
counted them.** That is now Phase 1's cheapest item.

### Measured while answering "is TSNPE worth losing amortization for?"

**33.68 %** of training rows carry zero live χ probes (mean **3.37** live of 12 slots) — matching
§4.3.6's independent estimate. **66.32 %** survive a probe-based truncation; **2.27 %** survive the
temperature constraint; **1.25 %** survive both. So the addendum's free-subset test works for the
first and **not** the second, and the amortization question is largely a false dilemma: the two big
wins come from a prior narrowing that keeps amortization entirely (§11.5).

**Nothing in §11 has been implemented.** This entry records the verification and the plan only.

---

## 2026-08-25 — the retrain landed, and the plots that were meant to explain it could not be read

The first posterior on the corrected χ band finished. Four questions came back with it: what the
cycle-averaged waveform means, why the power spectrum was a bare white line, how to read the set, and
what to do about the degeneracy. Three of the four turned out to have answers in the artifacts.

### Two of the four questions were rendering defects, and one of them made the run look worse than it is

**The power spectrum really was broken.** Counted pixels rather than eyeballing it: the only
steelblue in the image was the legend swatch — zero band pixels, zero median pixels inside the axes.
The obvious cause (a NaN poisoning `torch.quantile`) is **wrong**, and ruling it out is what found the
real one: `posterior_overlay.png` and `cycle_avg_waveform.png` run the same quantile over the *same*
traces and both rendered. They survive because the overlay band takes only the 50 best draws and
`cycle_average` confines a bad phase to one bin — so a broad posterior drawing a few unstable
parameter sets erases the PSD band and nothing else. **The symptom looked like a plotting bug in one
figure; it is a property of the posterior.** Now trap **G7**: rows are dropped, the count is returned,
and an empty band is annotated in the axes — *"the band is missing"* is a bug report, *"17 of 1000
draws were non-finite"* is a finding.

**And a defect nobody had asked about explained most of "it doesn't look that good".** Every axis
label, tick label, title and both parameter tables were **invisible in 14 of the 15 figures** —
measured contrast **1.06:1**, where readable is 4.5:1. The figures were BUILT under the dark theme
(#2B2B2B axes, #F5F5F5 text — exact `DARK` token matches) and SAVED on a light page (#F3F3F3, the
`LIGHT` token). matplotlib bakes artist colours at build time and reads `savefig.facecolor` at save
time, and `mpl_theme` rewrites rcParams globally; a multi-day run straddles a theme flip easily.
`loss_curve.png` was the one readable figure, fully light — i.e. built on the other side of the flip.
Now trap **G6** + `visualizers.save_figure`, which pins the save to `fig.get_facecolor()`.

### The identifiability decomposition, and the finding that revises §4.4

The `.rot.pt` sidecar already carried the Fisher rotation `V` — eigenvectors of the prior-averaged
`J^T J`, sorted best-first. That is exactly the "which directions does this experiment measure?"
object, sitting on disk, free. `scripts/posterior_identifiability.py` now reads it.

**`k` puts 99.9 % of its weight on a single direction loading −0.999·`k`, and its overlap with
`x_scale` across all 13 directions is 0.0002.** §4.4/§4.4.1 have the central problem recorded as the
`k`~`x_scale` alias (|cos| 0.97, measured three times). Both are true and together they say more:
|cos| compares gradient DIRECTIONS, a Fisher eigenvalue is about gradient MAGNITUDE, and `k`'s
gradient is both nearly parallel to `x_scale`'s and tiny. **`k` is not degenerate so much as
unmeasured** — and a flat direction is a different engineering problem from an alias, because no
rotation, reparameterisation or prior reaches it. Only a new observable does. Recorded in §4.6.

The same read explained the PPC's alarming "Invalid stats: 48/114": `chi_n_freqs` 6 supplied into
`chi_k_pad` 12 slots means 36 elements are padding before anything is masked — **~4 live probes in
12 slots.** `analysis.invalid_breakdown` prints that split now, because "42 % of the conditioning is
flat" is unactionable and "4 live probes of 12" is a lever.

### The gap this left, and it is closed only for future runs

**The eigenvalues were computed and thrown away.** `fisher_eigenbasis` returned V and dropped them,
so the 2026-08-25 posterior can be decomposed only up to an ORDERING — direction 12 is worst, but
whether direction 0 is 10× or 10⁶× better constrained is unrecoverable without a full Fisher re-run.
That gap is the difference between "mildly uneven" and "nine of thirteen parameters are prior".
`save_posterior_artifacts` now writes `fisher_eigenvalues`, and `build_posterior` prints the spread as
it computes the rotation. **The current artifact cannot be retrofitted cheaply; the next one is fine.**

Also noted and not fixed: `default_x` is None on the saved posterior, so the posterior behind those
figures cannot be re-sampled from the artifacts alone. Amortized NPE has no observation at save time
by design, so the fix belongs at *inference* time (persist `x_obs` beside the run's outputs), not at
save time — deliberately left rather than bolted on.

### Verification

`test_gui_progress.py` 89 → **95**, `test_artifact_consistency.py` 14 → **15**,
`test_user_sbi.py` 63 → **64**. Two of the new tests carry their own counterfactual: the theme test
asserts the *unpinned* save still reproduces the bug, and the divergent-draw test asserts the good
draws' band is unchanged rather than merely finite. The identifiability script was run against the
real artifact and reproduces the table in §4.6 exactly.

**Not verified: the pixels.** The figure fixes are pinned by contrast and background assertions, not
by a human looking at the app — §6 row 3's standing rule.

---

## 2026-08-21 (later) — the training budget, and the answer to "how many simulations per batch?"

Asked while §10.5's suites were still running. The factual answer, which was not written down
anywhere and was reachable only by reading `config.py`:

| | | |
|---|---|---|
| simulations per training batch | **2048** | `cfg.hw.batch_size` — `2**11` CUDA/fp32, `2**10` fp64, `2**6` CPU |
| batches | **5000** | `config.TRAINING_NUM_RUNS` |
| **total per training round** | **10 240 000** | and ×(1+K)≈8 solver passes per row in χ mode |

`_max_sim_batch` may still SPLIT a batch's solver call against that batch's own geometry, but the
training batch is 2048 rows either way.

### Why the obvious version of this feature would have been a trap

The request was "expose both, with a memory warning in the tooltip". A tooltip is the weakest
available guard for these two, for three reasons that are all measured or in the code already:

- **Batch width is not a memory-vs-SPEED knob, it is a memory-vs-ROWS knob.** `config.py`'s own note
  records 7.37 s at 2048 against **7.74 s at 1024** on this card — the *smaller* batch is slightly
  *slower*, because the solver is kernel-launch-bound. "Lower this if you run out of memory" silently
  halves the training set at unchanged wall-clock.
- **The two fields are not interchangeable budget knobs, they are statistics.** Each batch shares ONE
  Sobol `(t_scale_k, T_k)` pair, overridden for every row — so batch COUNT is the operating-point
  diversity and width is replication per point. 5000×2048 and 10000×1024 have equal totals and
  different training sets. Trap **X5** already says exactly this for calibration.
- **⚠ Both sit inside the C-11 checkpoint identity, which is digested into the checkpoint's DIRECTORY
  NAME.** Nudge either and a resumable multi-day run resolves to a directory that does not exist:
  no error, just a run that starts from zero. That is not tooltip material.

### So: a read-out first, an escape hatch second, and a hard inline status

The Posterior tab gained **Batches** and **Max rows per batch (0 = auto)** plus three derived lines
that update as you type:

1. `10,240,000 simulations = 5,000 batches x 2,048 rows`, and the sentence that batch count is the
   `(t_scale, T)` diversity — so the trade is on screen, not in a tooltip nobody opens.
2. A peak-VRAM estimate at **the worst geometry the Sobol pre-filter admits** (`n_fine ≤ min(N_ND_MAX,
   len(t))`), because that is the case that OOMs — the median is ~40 k and the p99 ~283 k, and both
   2026-08-10 and 2026-08-11 died in the tail. It reads `pipeline.peak_sim_elements` and
   `sim_memory_budget_elements` rather than restating the formula, so it cannot drift from the
   planner. It also says the budget is an **upper bound**, since the free-VRAM reading overstates by
   roughly the size of the desktop.
3. Checkpoint status: resumes / starts a new run, and when it is a new run,
   `training_checkpoint.describe_siblings` names **the field that differs**.

> **A measurement the read-out produced immediately, and it is worth knowing before the retrain:**
> at the default 2048 rows the worst admissible geometry needs **~11.44 GiB** against a planner
> budget of **~12.48 GiB** on an idle card. That is a ~1 GiB margin against a number that is itself an
> over-estimate of free VRAM — which is a much sharper way of saying what §4.1 says about closing
> browsers, and it is now visible before you press Train rather than inferable after an OOM.

### The wiring detail that would have made the whole feature a no-op

`orchestrator.py` does `from .config import TRAINING_NUM_RUNS, TRAINING_RUN_SIZE` — **snapshotted at
import**. A panel that "applied" the user's numbers by assigning to `core.config` would change
nothing, run 5000×2048 anyway, and say nothing. So the budget is a pair of **keyword-only parameters**
on `build_posterior`, defaulting to `None` = the constants, which keeps the CLI and every script and
test byte-identical. Written up as trap **X12**, because the same shape is one import away in a dozen
other places.

### What was refactored rather than duplicated

- `pipeline.peak_sim_elements` / `sim_keep_elements` / `sim_memory_budget_elements` are extracted
  **public** from `_max_sim_batch`, which now calls them. A second copy of the cost model in the GUI
  would drift the first time either was tuned, and a display that reassures you about a number the
  planner does not use is worse than no display.
- `orchestrator._training_identity` → **`training_identity`** (§9.6's private-name-across-boundaries
  rule): the GUI needs it to answer "will this resume?".
- `settings.get_int`, because QSettings hands back strings on Windows and `TRAINING_RUN_SIZE`'s
  meaningful value is `0` — so a bare `int(qs.value(...) or 0)` would silently swallow a missing key.

### Verification

`test_gui_progress.py` 83 → **89**, `test_user_sbi.py` 60 → **63**. The wiring test asserts both
halves — the values arrive as kwargs *and* the config constants are untouched — and was checked
against **two deliberate mutations**: deleting the kwargs, and "applying" the budget by writing
config. Both were caught, and the second also broke a neighbouring test, which is itself a neat
demonstration of why writing the global is wrong. Defaults are the config constants, so an untouched
GUI is byte-identical to today's behaviour.

**Owed: the eyeball.** Three wrapped derived lines in the controls column is the most likely thing
yet to overflow it (§10.2's L4/L13 class); no human has seen it.

---

## 2026-08-21 — one progress bar, and a solver meter that the next speedup cannot break

Backlog row 0 (§10.5), both parts. The user asked for ONE overall bar plus a Solver Performance rate
with no bar of its own; part (a) was a live regression that had to be fixed to get there at all.

### The regression, and the measurement that told us which knob it was

The Solver Performance meter had been reading `— (idle)` for whole runs since CUDA graphs landed.
Nothing was broken in the solver: the meter was **parsed out of the rendered text of the solver's
tqdm bar**, and a solver call had become *shorter than that bar's own `mininterval=1.0`*, so tqdm
never painted a frame carrying a rate. **A 10× speedup made a progress bar too fast to render.**

§10.5 flagged an uncertainty worth settling before touching anything — `miniters` gates refreshes
too, so lowering only `mininterval` might not be enough. Reproduced on the real settings with a
synthetic bar paced to 0.7 s, no GPU:

| | stderr | rates |
|---|---|---|
| `mininterval=1.0, miniters=1000` (as shipped) | **123 chars** | **0** |
| `mininterval=0.1, miniters=1000` | 601 chars | 6 |
| `mininterval=0.1, miniters=100` | 601 chars | 6 |

123 chars / zero rates is byte-identical to what was measured on the real run, so the reproduction is
trustworthy — and **`miniters` is not the blocker**: 1000 steps is 20 chunked `update()` calls ≈ 14 ms,
far inside 100 ms. One knob, not two.

### The one-line fix was made anyway — for the CLI, which nobody had noticed was blind

`mininterval` went 1.0 → 0.1 (tqdm's own default). Not for the GUI, which no longer reads the bar,
but because **`python -m core` has nothing else**: a CLI user watching a graphed run was seeing an
opening frame and no speed, and §1.1 makes the CLI first-class. Cost is ~10 refreshes/s against a
6.65 µs/step loop.

### But the durable fix is that the GUI stopped reading a rendered bar at all

`mininterval` is a *time* gate, so lowering it shrinks the broken window without closing it — any
call shorter than 0.1 s still reports nothing, and the next speedup re-opens it wider. So the solver
now **publishes**: `core/progress.py` holds a monotonic step count, `_step_iter` bumps it per step
and `_advance(bar, k)` bumps it per graphed chunk (and advances the bar in the same call, so the two
cannot drift), and `ProgressPane` differences it on the 100 ms tick it already ran for the spinner.
Correct at any speed. `SOLVER_IDLE_S` came down 45 → 8 s to match, and the step-progress strip was
deleted — the "NO bar" the request asked for.

### ⚠ The spec's own instruction would have blacked out the longest phase of the run

§10.5 said "render no other rows". Taken literally that is wrong, and not subtly: **sbi's
neural-network training emits no tqdm bar at all.** It prints an epoch counter, which `vt` classifies
as an overwrite-mode row with `pct is None`; the only other live row in that phase is the degenerate
`Training neural posterior — 0/1`. Delete rows and add no caption and *hours* of a multi-day build
show an indeterminate bar and nothing else. The caption's fallback — deepest row whose `total != 1` —
exists for exactly that, and is now pinned by a test that names the reason.

### A second thing found on the way in: `QUIET_SEGMENT_BAR` was hiding a bad election rule

The overall bar tracked the DEEPEST informative row. That is one config flip from wrong: `Running
time segments` wraps segs in {1,2,3}, so it is both deeper and far coarser than `Generating training
data` and would sweep the bar 0→100 % every couple of seconds. `QUIET_SEGMENT_BAR` was added to tidy
the display and was silently load-bearing for correctness. The rule is now **largest non-degenerate
total**, which is immune to that whole class. The two agree on every nest observed, so the test that
separates them was checked for vacuity explicitly: deepest gives 66 %, largest-total gives 38 %.

### Deliberately NOT done, and the reason is cancellation

Quieting the solver's tqdm bar under the GUI would have retired the `SOLVER_BAR_DESC` seam entirely.
It was rejected: the cancel is cooperative and its checkpoint is `_SignalStream.write()`, and during
training-data generation that bar is by far the most frequent writer. Quieting it would turn ~0.1 s
cancel latency into one training batch (~2 s at production geometry, longer at large `T_obs`). The
desc seam therefore survives — but *only* as an exclusion predicate for the overall-bar election
(trap S3), never again as a rate source (trap **S0**, new).

### What was actually verified, and what is still owed

- `test_gui_progress.py` 79 → **83**; the other five suites re-run clean.
- The new CLI-bar test carries its own **counterfactual** (`mininterval=1.0` must render *no* rate),
  so it cannot quietly go vacuous if someone puts the value back.
- End-to-end on CPU — the real `sdeint` solver through `redirect_streams` through the pump into a
  real `ProgressPane`: **a live rate on 195 of 196 ticks (99 %)**, against 0 % before; the overall bar
  tracked `Generating training data` 0→83 %; zero bar frames leaked into the log pane.
- **A cancel raised from inside the solver's step loop**, because `_step_iter` became a generator and
  that puts a frame in trap **C1**'s unwind path: it unwound in well under a second, and the next
  `tqdm` bar did not deadlock (C1's pinning test *hangs* rather than failing, so this was run, not
  reasoned about).
- **On real CUDA** (RTX 5070 Ti, batch 2048, `T = 100 000`): the graphed path now emits **678 chars
  and 7 rate frames** where it emitted 123 chars and none — 0.811 s, **123 349 steps/s**, first frame
  `128420.87it/s`; eager for comparison, 7.912 s and 12 639 steps/s. Step accounting is exact on both
  paths at a geometry with a partial tail (2 full chunks + a 3-step remainder), which is the case that
  pins `_advance`.
- **Owed: a human looking at it.** Headless tests cover the wiring, never the pixels (§10.3) — this
  joins §6 row 3's eyeball list, and it is the only thing about this change that is unverified.

### Two stale facts corrected in passing

§2's architecture map described `Solvers/sdeint.py` as "eager + torch.compile fast path and an unused
implicit solver". Neither half was true: the torch.compile path never existed (§8.3) and
`implicit_euler` was deleted in §7.7. Both were sitting in the one section a newcomer is told to read
first.

---

## 2026-08-20 (later) — the solver was 8× slower than it needed to be, and the docstring said otherwise

Asked to speed up training generation, validation and inference before the retrain. The answer was
none of the things on the list: **~88 % of solver wall-clock was CPU kernel-LAUNCH overhead.** The
GPU was doing ~5 µs of work per Euler step and sitting idle for the other ~50 waiting for the CPU to
submit the next three kernels.

`scripts/profile_sde.py --batch 2048 --steps 100000`, the same command before and after:

| | before | after |
|---|---|---|
| simulator call | **5520.5 ms** (±75.1) | **698.5 ms** (±1.0) |

**7.90× end-to-end on the real path**, not a microbenchmark — and the ±75 ms → ±1 ms collapse is its
own evidence, because the jitter *was* the CPU.

### Why nobody found it

`euler_compiled`'s docstring said the fast path already existed: *"torch.compile + CUDA Graphs"*,
with a stated requirement that the model expose a step *"wrapped with
`@torch.compile(mode='reduce-overhead')`"*. **`torch.compile(` appeared exactly once in the whole
repo — inside that docstring.** Every `compiled_step` is `@torch.jit.script`, which removes *Python*
overhead and not *launch* overhead. §8 was meanwhile marked CLOSED, so the door was shut and labelled
already-furnished. Both are now corrected.

### Two measurements that reordered the whole plan

- **Batch width is free in TIME and linear in MEMORY.** 512 / 2048 / 8192 / 16384 rows at 20 k steps:
  1159 / 1164 / 1106 / 1191 ms — flat 8× beyond the ~2048 the code's comments assume. Every "batch it
  wider" idea is therefore bounded by memory alone, which is why the remaining §8.3 items are all
  gated on a *tensor*, never on the solver.
- **The Fisher does NOT dominate a production run.** §4.1 step 5 says it does; that is true of a
  4-batch smoke train and false of a 5000-batch retrain, where training generation is ~97 % and the
  Fisher ~1–2 %. Ranking optimisations off smoke timings is what makes the Fisher's 216-call
  structure look like the prize when it is worth ~15 minutes out of ~38 hours. **Rank against the
  production denominator.**

### What was verified before trusting it

| check | result |
|---|---|
| graphed vs eager, deterministic model (all noise zeroed), time-VARYING drive, 2 full chunks + a 2-step tail | **bitwise equal**, max diff 0.0 |
| RNG advances across replays | yes — not frozen (consecutive replays differ) |
| `get_rng_state_all` / `set_rng_state_all` round-trip through replay | **reproduces exactly** → C-11's resume is safe |
| graph cache placement | module-level, not on `Solver`, not a `Solver` singleton (trap **X1**) |

The deterministic-model trick is what made the first row possible: with noise live the two paths draw
in a different order and can only be compared statistically, which would never catch an off-by-one in
the force index or the output slice. Zeroing every noise channel and using a *time-varying* drive
makes any indexing error a hard failure.

### The 0.83-ULP detour, which found something unrelated and worth keeping

The first equivalence run failed at **max|diff| 5.8e-11** — sub-ULP, and only in the eager tail after
100 bitwise-identical replayed rows. Chasing it: `ent["x"] == a[100]` exactly, `a[100] == b[100]`
exactly, identical params and force — identical inputs, one eager step, different answers. The cause
is not the graph at all. **Three consecutive EAGER runs of the same deterministic model gave
run1 ≠ run2 (from row 1, max|diff| 7.5e-09) and run2 == run3**: TorchScript's profiling executor runs
the first invocations unoptimised, then specialises and fuses, and the fused kernel differs by ~1 ULP.

So *"graphed == eager bitwise"* is not well posed until both are warm — the eager baseline is not
bitwise reproducible against itself. After a 3-run warm-up every comparison is exactly 0.0. Recorded
as a trap in §8.3, because it is CUDA-only and TorchScript-only and therefore invisible to the CPU
suite, but it bounds any future *"this GPU run reproduces bit-for-bit"* claim.

> **Worth generalising:** the failing assertion was mine, not the code's. When a sub-ULP discrepancy
> appears, the question "is my baseline reproducible with itself?" comes before "what did I break?"

### End to end: the smoke train, and the masked fraction that had to be chased

Same invocation as the 2026-08-20 baseline (chi, `TOBS_S=4.5`, master bounds + `master_spont`):

| stage | 2026-08-20 | with graphs | |
|---|---|---|---|
| prior | 526 s | 455 s | 1.16× — mostly the HDBSCAN/GMM fit, not the solver |
| **posterior** | **5772 s** | **531 s** | **10.9×** |
| validate | 204 s | 25 s | 8.2× |
| infer | 682 s | 63 s | 10.8× |
| **total** | **7183.6 s** | **1073.3 s** | **6.7×** |

Zero OOM retries. Note the prior stage is now the *single largest* item — the profile of the run has
inverted, and any further optimisation should be re-ranked against that rather than against this
document's older numbers.

**Training masking came in at 31.7 %, below the 34.8–38.2 % band** the standing rule says to check.
Treated as a signal rather than waved through, because "the masked fraction moved" is exactly how a
solver change that altered the trajectories would present. Chasing it produced a real correction to
the check itself.

- **A direct physics A/B, graphs ON vs OFF, 3072 rows per arm, both RNGs seeded.** Ω₀ — the quantity
  that *drives* masking — is indistinguishable: two-sample **z = −0.01**, **KS D = 0.0020 against a
  5 % critical value of 0.0347**. Trace mean and std agree to 0.05 % and 0.025 % of one SD. At that
  sample size the test would catch a 0.06-SE shift. **The solver did not move the physics.**
- **Two more smoke trains**, now that one costs 18 min instead of two hours: **44.9 %** and
  **32.7 %**. Three graph-path runs mean **36.4 %**, against the four pre-graph runs' **36.3 %**.

### ⚠ The masked-fraction check has ~10× less power than this document implied

The band was being read as a tolerance on 704 probes, i.e. a ~1.8 pp binomial standard error. That is
the wrong error model. **All rows in a batch share one `(t_scale, T)` stratum AND one probe set**
(both drawn above the seam, by design — see `_rows_with_oom_retry`), so the 704 probes are **4
correlated strata** and the effective sample size is the BATCH count. Measured per-batch masked
fractions across the three runs:

```
seed=0:  44.2%  13.5%  18.8%  39.1%
seed=1:  44.6%  39.6%  35.4%  57.3%
seed=2:  36.2%  27.1%  43.8%  20.3%
   per-batch SD = 12.2 pp over 12 batches  ->  a 4-batch run has SD ~6.1 pp
```

At SD 6.1 pp, 31.7 % is **−0.75 SD** and 44.9 % is **+1.4 SD** — both unremarkable, and the old
band's 3.4 pp tightness across four draws was luck, not precision. **Read the MEAN of a few runs
against ~37 %; treat a single run within roughly ±12 pp as uninformative.** `smoke_train.py`'s own
WHAT-TO-WATCH list now says so, with the numbers.

Compounding it: `smoke_train.py` sets `torch.manual_seed` but not `np.random.seed`, and initial
conditions come from `np.random.randint` (trap **X8**) — so two runs at the same `SEED` were never
comparable anyway.

> **The generalisable bit, and it is the same question twice at different scales.** The 0.83-ULP
> detour above was resolved by asking *"is my baseline reproducible against itself?"*. This one is
> resolved by asking it of a statistic instead of a number: *"what is this check's actual error
> bar?"* A threshold assembled from a handful of unseeded runs is a **distribution**, not a
> tolerance, and quoting a binomial SE over correlated units understates it by the square root of
> the cluster size.

### One regression the speedup caused, found by looking at the GUI during the retrain — ✅ FIXED 2026-08-21

**The Solver Performance meter now reads `— (idle)` for the whole run.** Nothing is broken in the
solver; the meter is scraped from the *rendered text of a tqdm bar*, and a solver call is now shorter
than that bar's own `mininterval=1.0`, so tqdm never paints a frame carrying a rate. Measured on one
100 k-step call: graphs OFF emits 670 chars of stderr containing `14152.92it/s`; graphs ON emits 123
chars containing **no rate at all**, just the `?it/s` opening frame.

The plumbing had anticipated *some* short bars — `vt.py` documents `?it/s` as *"any bar shorter than
its mininterval"* and `progress_pane.SOLVER_IDLE_S` holds the last rate across gaps — but a hold needs
one rate to hold, and there are now none. **A 10× speedup made a progress bar too fast to render.**
Fixed the next day, together with the requested one-bar rework — **§10.5**, and the entry above this
one. The meter no longer reads a rendered bar at all; the solver publishes a step count instead, so
the failure class (a call shorter than its own bar's mininterval) is closed rather than narrowed.

### Also landed, both bit-identical

- `Simulator.simulate`'s `sol` is `torch.empty`, not `torch.zeros` — the *larger* of the two solver
  buffers (~6.95 GiB at production geometry against `sdeint`'s ~2.3 GiB). §8.1 listed the small one
  and missed this one entirely.
- `forcing.zero_force` returns a stride-0 expansion instead of a materialised block of zeros
  (2.29 → 1.82 GiB, identical checksum through `gen_obs`). Four call sites in `pipeline` and
  `decorrelate`. The helper carries the warning that the result must never be written into — every
  hand-built force that IS written (FDT's cosine drives) builds its own real tensor.

### Deliberately NOT done, with the reasons in §8.3

Probe concatenation (gated on `build_nondim_sin_force_tensor`'s **8.64 GiB transient for a 2.16 GiB
result** — and `_max_sim_batch` under-counts that 4×, a plausible contributor to trap X7); strided
single-variable output; PPC budget binning; Fisher arm stacking (**dropped** — ~15 min for a CRN
hazard); `gen_stats` dedup (~6 min). And `dt_nd_min`, which is a *sampling* constraint derived from a
prior bound masquerading as an integration constraint — a science measurement, and one whose obvious
fix (per-batch `dt`) would make integration error a function of an inferred parameter.

## 2026-08-20 — the retrain finished, and it trained on the RETIRED band

The first full chi retrain completed on 2026-08-19: 10.24M rows (2048 × 5000), rotation ON,
~5 days. It has to be redone. **It trained at `chi_freq_bounds = (0.1, 10.0)`, `chi_f0 = 0.1`,
`K = 10`** — the band C-5 retired on 2026-08-06, and verbatim `posterior_chi_08042026`'s
configuration, the one §4.1 records as *"well-calibrated but uninformative… 8 of its 10 probes
measured noise"*.

Read from the artifact, not inferred: the `.rot.pt` sidecar and the checkpoint header both carry
`(0.1, 10.0)` / `0.1` / `10` against `config.py`'s `(0.03, 0.3)` / `0.15` / `6`.

**Cause — trap Q4, newly written down.** `PRISM.ini` held `chi_lo=0.1 chi_hi=10.0 chi_f0=0.1
chi_k=10` from a session predating C-5. `inference_tabs` seeds those widgets from `config.CHI_*` and
`_restore` then overwrites them from QSettings, so the retired band was silently reapplied on every
launch for two weeks. The `_assert_mode_matches` band guard **did** exist — and is useless here,
because it compares the posterior against `cfg`, and a stale `cfg` loading the posterior trained
under that same stale `cfg` agrees with itself.

### It was visible in a plot the whole time

`ppc.png` reports `Invalid stats: 24/114`. Under chi, invalid = 11 zeroed Group-G columns + `log T` +
`(CHI_K_PAD − K) × 6` dead pad channels. At `K_PAD = 12` that is 24 **only at K = 10**; at the
intended K = 6 it would have been 48. The run's own diagnostic encoded its probe count in a number
nobody had a reason to decode. Worth remembering as a general habit: **the invalid-column count is a
free readout of K.**

### Why the failure pattern is the band, and not the "structural degeneracy" it looked like

Probes were log-spaced over (0.1, 10): 0.100, 0.167, 0.278, 0.464, 0.774, 1.29, 2.15, 3.59, 5.99,
10.0. §4.3.2 put the entrainment knee at **0.35–0.4×**, above which *"a captured bundle reports the
drive back to itself"*. **Three probes in band; seven reporting the drive.** Against §4.4.1's payload
table:

| parameter | §4.4.1 says it rides | fate above the knee | observed |
|---|---|---|---|
| `lam` | chi **phase** (`chi2_sin` is its top feature) | phase dies first under capture | **worst of 13**; marginal ramps to the box edge |
| `f_scale` | chi **magnitude** | magnitude survives | sharp, near truth |
| `t_scale` | chi magnitude + `A3_log_fpeak` | survives | sharp, near truth |
| `k`, `x_scale` | `A1_mean`, `A2_log_var` — **spontaneous** | untouched by the band | both off by an *identical* **+11.3%** |

Four for four. The identical +11.3% on `k`/`x_scale` is the §4.4.1 alias (|cos| 0.96–0.98) being
traversed rather than broken — the one prediction §4.1 step 5 made that this run did confirm.

**What is NOT the band, and no retrain will fix:** `n`, `temp` and `tau_c` appear in
`Models/nadrowski_model.py` **only inside noise amplitudes** (`_x_noise_const`/`_y_noise_const`,
`c_noise`) and never in the drift, while §3.1 fixes the observable at state column 0 — so `y` and `c`
are latent and those three are seen only through how latent-variable noise propagates into `x`. That
is provable from the model file with zero simulation, and it accounts for three of the four ramping
marginals. `lam` is the fourth, and `lam` is the one the band destroyed.

### Why the smoke train could never have caught this — and it is not about the masked fraction

The obvious theory is that the retired band flattered the smoke train's headline metric (entrainment
lives at *high* frequency, where probes clear `CHI_MIN_CYCLES` easily, so masking should read low).
**That theory is untested and probably wrong, and it is the less important answer.** Masking across
four corrected-band smoke trains reads 36.7 / 38.2 / 34.8 / 35.4 % — *this entry originally quoted a
±1.8 pp binomial standard error at n = 704, which is **wrong**: the run-level SD is ~6.1 pp, because
the 704 probes are 4 correlated strata (corrected in the 2026-08-20 (later) entry)* — and §4.3.4
already established that the driver of masking is the ROW's own Ω₀, not the band. There is no
retired-band smoke measurement to compare against, because:

> **`scripts/` and the GUI were configured from DIFFERENT SOURCES.** Every script builds its config
> through `_common.script_cfg` → `cli.make_sim_config`, which falls back to `config.CHI_*`. Nothing
> under `scripts/` reads `PRISM.ini` at all. The GUI reads QSettings. So **every smoke train ever run
> — including the ones on 2026-08-12 and 2026-08-13, hours before the retrain — was green at
> (0.03, 0.3) while the GUI was about to train at (0.1, 10.0).** No masked fraction, no OOM count, no
> stage timing could have revealed that, because the pre-flight was not testing the configuration the
> record run would use.

That is the real lesson and it generalises past chi: **a pre-flight that reads its configuration from
a different source than the run it is clearing is not a pre-flight.** It is why
`_assert_chi_config_is_deliberate` lives in `orchestrator` — the one path both front-ends share — and
why it compares against `config.py` rather than against another artifact.

### Two instruments that mislead, and what to read instead

- **Flat SBC is not evidence of success.** All 13 rank CDFs sat on the diagonal, TARP on the
  diagonal, PPC mean |z| 0.396 — and the run is unusable. `posterior_chi_08042026` was likewise flat
  on 12/13 with TARP p = 1.000 *and* every marginal at the prior. A posterior that returns the prior
  is perfectly SBC-flat **by construction**: calibration is a property of the joint, not of any
  conditional. Flat SBC says "not overconfident"; it never says "informative".
- **The best-fit-draw table is not a recovery measurement.** `overlay.rank_by_stats` standardises by
  `s.std(dim=0)` — the spread **across posterior draws**, not measurement noise — so its RMS z says
  "this draw is inside the predictive cloud", not "this draw is indistinguishable from truth". And
  the distance weights all live features equally, so it is nearly blind along degenerate directions
  and the argmin is close to a free draw there. `N` came back at **+4.5 %** while its marginal ramps
  to 300 with truth at 50: luck, not recovery. **Trust that table only where the marginal is sharp;
  read the marginal quantile of the true value otherwise.**

### Fixed

`orchestrator._assert_chi_config_is_deliberate(cfg)` — called at the top of `build_prior` **and**
`build_posterior`, before any simulation. Refuses when `chi_freq_bounds` or `chi_f0` differ from the
`config.CHI_*` defaults, names both values, points at `PRISM.ini` as the likely source, and offers
`PRISM_CHI_OVERRIDE=1` for a deliberate sweep. Placed above the load branch as well as the training
one, for the self-agreeing-stale-config reason above. Keyed on the module defaults, never a literal,
so C-5's successor moves one constant and the guard follows. **`chi_n_freqs` is deliberately not
gated** (K is per-observation; training draws its own). Pinned by
`test_a_chi_run_at_a_non_default_band_is_refused_before_the_simulation_spend`, verified to fail
against the pre-change orchestrator. The retired artifact is renamed
`08192026_posterior_RETIRED_band_0p1_to_10.pt`; `PRISM.ini` is corrected.

> **The lesson, which is bigger than chi.** A constant that defines *what is measured* must not be
> persisted like a UI preference. Every guard in this file compares an artifact against a live
> config; none compared the live config against the code that defines it. That gap is what a
> two-week-old settings file walked through.

Newest first. Nothing here is required to work on the code; it records **why** decisions were made,
and — importantly — **which dead ends were already tried**, so they are not retried.

## 2026-08-13 — the last two non-retrain backlog rows, and both were wrong about themselves

§6's *PICK UP HERE* rows 1 and 2. The retrain is the user's to run, so this session took the two
items that are independent of it. Neither turned out to be the change its row described, and in both
cases the row's own premise is the thing worth recording.

### Row 1: three of the four writes migrated, and the fourth refused for a reason

`file_manager.atomic_torch_save`'s body is now `_atomic_write(path, writer)` — the tmp-sibling →
fsync → `os.replace` mechanism, with the writer passed in — and `atomic_savez` sits beside it for the
`.loss.npz`. Migrated: `save_mix_dist` (so `save_prior_artifacts` comes free), all three writes in
`save_posterior_artifacts`, and `scripts/retrain_convergence.py`'s own loss curve.

**`FDT/cross_validation` was not migrated, and that is the finding.** The row grouped it with the
other two; they are not alike. Atomicity buys exactly one thing — *an existing good file is never
replaced by a partial one* — and `run_fdt_param_sweep`'s `output_path` defaults to a **timestamped**
name, so there is no existing file. Staging through a temp sibling would rename "partial file at X"
to "partial file at X.tmp" and buy nothing, while destroying a property that IS used: the HDF5 is
written incrementally so `load_param_sweep` can mark a row `failed` when a run is interrupted after
Phase A, and the all-points-failed error tells the reader that path holds the PSDs. Its actual
data-loss path is different from the one the row imagined — an explicit `output_path` that already
exists, because `"w"` truncates before Phase A writes anything. All of that is now at the function.

Two details worth keeping:

- **`atomic_savez` takes a dict, not `**kwargs`.** np.savez's own signature is `**kwds`, so
  inheriting it would let an array named `path` or `retries` bind to the helper's own parameters.
- **numpy appends `.npz` to a NAME, not to a HANDLE.** `_atomic_write` hands the writer an open
  handle, which is what stops the file landing at `<name>.loss.npz.tmp.npz`. The round-trip half of
  the new test exists for that specific failure, which is silent — the run reports success and the
  convergence read later finds nothing.

### Row 2: the white line was RIGHT, and the row named the wrong defect

`Reduction/plots.py`'s `color="white"` does not follow the theme **and must not**: its backdrop is a
viridis surface, not the page, so a themed colour would be near-black on dark purple under the light
theme. That is the same call `cross_validation_plots` already makes with `darkorange` and
`simulate_export` makes by pinning a white video figure. Changing it would have been a regression.

The real defect was one line further down: the line carries a **legend** entry, and the legend sits
on `legend.facecolor` — `#FFFFFF` under the light theme, and `inherit` → white under plain
matplotlib. A white swatch on white. So the fix is a `text.color` **halo** (`path_effects.Stroke`)
around the white core: contrast-guaranteed against the legend background by construction of the token
pair, and on the surface it only adds a thin outline. Both the 3D ridge line and the 2D `"w-"` carry
it; only the first was in the row.

`FDT/plots.py`'s `'gray'` was straightforward chrome and moved to a token — but **`grid.color` is the
wrong token and was caught by rendering rather than by reasoning.** It is the border colour
(`#3D3D3D`) and the dark theme paints axes on `#2B2B2B`, so the χ''=0 line *and its legend swatch*
both vanished — worse than the hardcoded gray. It is now `axes.edgecolor`, the same token as the
equilibrium line directly above it, the two separated by dash pattern and width, which is
matplotlib's own convention and survives a theme flip.

Both figures were rendered to PNG under the light theme, the dark theme and plain matplotlib
defaults, and looked at. That is stronger than a headless test and still not a human looking at the
app, so they join §6 row 3's eyeball list.

> **The generalisable bit, since it has now happened twice in three sessions** (the 2026-08-12 entry
> records three backlog rows that were already built): **a backlog row is a hypothesis, not a
> specification.** Row 1 named four call sites of which one should not change; row 2 named two colours
> of which one was already correct and pointed at the wrong line of the other. Both rows were written
> by someone who had just been in that code. Re-derive the defect before implementing the fix.

### What kept it honest

Each of the three new tests was run against a **restored pre-change write** (`atomic_*` rebound to a
plain `torch.save`/`np.savez` at the destination) and confirmed to fail with the message it claims —
a torn-write test that passes either way is the easiest kind to write and guards nothing. The
injected failure writes its bytes and *then* raises, because failing before the writer starts passes
against a bare `torch.save` too: the destination is only clobbered once writing has begun.

Suites: 216 tests, all green (`test_artifact_consistency` 10 → 13).

### The CHECKPOINT=1 smoke train — §4.1 step 5's last un-done pre-flight, now DONE

`scripts/smoke_train.py`, **chi + rotation ON + CUDA + `CHECKPOINT=1` + `SAVE=1`**, `TOBS_S=4.5`,
master bounds + `master_spont`, card otherwise idle. **All four stages, 7069 s** (prior 571,
posterior 5638, validate 206, infer 653) — against 8218 s on 2026-08-12 and 8827 s on 2026-08-11.

| signal | result | reading |
|---|---|---|
| **training masked** | **34.8 %** (245/704) | §4.3.6 records 36.8 %; the runs so far read 36.7 / 38.2 / 34.8 at the same 704 probes. SE at that n is 1.8 pp, so this is 1.1 SE low — noise, and the watch condition is *materially ABOVE* ~37 %, not below |
| validate masked | 40.6 % (73/180) | between §4.3.6's 47.2 % and 2026-08-12's 38.3 %; 180 probes, so spread is expected |
| PPC masked | 61.2 % (2938/4800) | §4.3.6's 79.9 % → 64.6 % → 61.2 %. A readout of the posterior, not the probe machinery |
| OOM retries, either level | **0** | 13 GiB free; not evidence the retries work |
| `[checkpoint]` lines | preflight (dir + free disk) and `complete: 4 batches` | **the point of the run** — see below |
| `[fisher]` | `averaged simulation Fisher over 8/8 operating points` | rotation genuinely ON, i.e. the configuration that shipped the C-11 device bug |
| artifacts | `_smoke_prior.pt`, `_smoke_posterior.pt`, `.rot.pt`, `.loss.npz` all landed; **zero stray `.tmp`** | all four writes migrated today, exercised on the real path |

Every artifact was then opened rather than assumed: the sidecar carries `mode=chi`, `chi_layout=2`,
`k_pad=12`, `elem_w=6`, `max_cycles=20`, the `(0.03, 0.3)` band, 13 `param_keys` and a **(13,13) `V`
on the CPU that is orthogonal to 1e-5**; the posterior unpickles as a `DirectPosterior`; the prior
reports NADROWSKI with a 10-dim box; the `.loss.npz` reads back all five keys. That last one is the
check worth keeping — `atomic_savez`'s failure mode (numpy appending `.npz` to the temp file's
`.tmp`) is silent, and only opening the file at the expected path catches it. All `_smoke_*` files
and the checkpoint temp dir were deleted afterwards; `Resources/Posteriors/` is empty again.

**What this does and does not prove.** It proves the checkpoint's *write* path runs on the card under
the retrain's exact configuration — the thing §4.1 asked for, and the thing no CPU test can certify
(2026-08-12). It did **not** exercise a RESUME. That gap was closed separately, below.

### Then the RESUME drill — the line that exists to rescue the retrain had never run

`build_posterior` rehomes the checkpoint's stored `V` from the CPU onto `cfg.hw.device`, guarded by
`if ckpt_resumed is not None and rotate:`. Read that guard literally: the line executes **only on a
resume, with rotation on** — i.e. only on the run C-11 exists to rescue, and never in any CPU test.
Its own comment says *"Without this the FIRST GPU resume with rotation ON would crash."* Nothing had
ever run it. It was reasoned to, written, reviewed — and unexecuted.

**The trap that would have made the obvious drill a silent pass.** `smoke_train.py` allocated
`tempfile.mkdtemp` per run, so a second run never found the first's checkpoint. A stable directory
knob alone is **not enough**, and this is the part worth remembering: `_training_identity` includes
`prior_fingerprint`, and `_gmm_fingerprint`'s own docstring says two runs over the *same box* produce
different fits. Two runs that each BUILD a prior therefore land in two different directories under
one `CKPT_DIR` — run 2 never resumes, and the drill reports success having tested nothing. So
`smoke_train.py` gained **both** `CKPT_DIR` and `PRIOR` (load a saved prior instead of building), the
second being what pins the fingerprint. A `CKPT_DIR` without a `PRIOR` now prints a warning naming
this exact failure. Note this is the same mechanism as §4.1 step 5's *"keep the prior it started
with"* rule, met from the other direction.

| | run 1 | run 2 (identical command) |
|---|---|---|
| prior | 1.4 s (**loaded**, against 571 s to build) | 1.4 s |
| posterior | 5123.7 s — Fisher + generation + fit | **1.6 s** — Fisher and generation both skipped |
| total | 5125.2 s | **3.0 s** |

Run 2 printed `Reusing the Fisher rotation stored with the training checkpoint (2/2 batches —
COMPLETE, so generation will be skipped)`. That print sits **after** the `.to(device=…, dtype=…)` in
the same block, so emitting it *is* the proof the rehoming ran. Checked rather than inferred: the
stored `V` is a real `(13,13)` **CPU** float32 tensor, orthogonal to 1e-5; `torch.cuda.is_available()`
is True; the flow then trained six epochs through the rotated bijection; and the run contains **zero**
`Expected all tensors to be on the same device` errors. A `V` left on the CPU meeting a CUDA
`OrthogonalTransform` is a hard error, not a silent promotion — which is precisely how this bug class
announced itself on 2026-08-12.

**Still not exercised:** the *partial* splice. Run 1 completed, so run 2 took the
complete-checkpoint short-circuit rather than resuming mid-generation and appending new rows to old.
That case is covered on the CPU, including an out-of-process SIGKILL (2026-08-12). What is now
observed on the GPU is the `V` rehoming and the short-circuit; what remains reasoned-to is the splice.

The scratch `CKPT_DIR` was deleted and `Resources/Checkpoints/` was never created — a real run must
not find a smoke checkpoint.

### Two script-level defects fixed in the same pass

- **`retrain_convergence.py` omitted `chi_max_cycles`** from `training_params`, while both
  `sbc_characterize.py` and `build_posterior` thread it. Under `CHI=1` it fell through to
  `gen_training_data`'s module fallback. Benign *only* while `SimConfig`'s default happens to equal
  `config.CHI_MAX_CYCLES` — which makes it the identical latent class to the bug recorded three lines
  above it in that same dict (*"omitting them silently took the forced branch"*). One line.
- **`diagnose_fscale.py` recommended a config that is known to be worse.** Its docstring said *"A
  retrain is still needed to CONFIRM the fix"* and its H2 verdict printed
  `config.REPARAM_LOG_PARAMS = ['f_scale']`. **That retrain was run, and it disconfirmed** — Appendix
  A 2026-07-16, `[TRIED → FAILED → REVERTED]`, worse TARP/expected coverage *and* a worse `f_scale`
  SBC rank. §9's "two stale things found, not fixed" had already caught this; the script had simply
  never been told. Both now report the finding and cite the entry. **H1 was deliberately left
  intact** — "does the deployed `V` smear `f_scale` into the degeneracy block?" is a generic rotation
  diagnostic and the retrain is rotated, so H1 applies directly to the posterior about to exist. Only
  H2's *recommendation* was stale, not the tool.

### A trap found on the way in: the smoke train's own default box is not the retrain's

`smoke_train.py` had to be given `BOUNDS=…/master.txt` explicitly. Left at its default it resolves
the cell's **same-named sibling**, `Bounds/nadrowski/master_spont.txt` — correct behaviour
(`cli.resolve_bounds_for_cell`, shared with the CLI and GUI, pinned by `test_artifact_consistency`)
and the wrong box for this purpose. Measured both ways:

```
DEFAULT (no BOUNDS)    mode=chi  dims=12  chi block=72  rescale=['x_scale','t_scale']
BOUNDS=master.txt      mode=chi  dims=13  chi block=72  rescale=['x_scale','t_scale','f_scale']
```

**Both say `mode=chi` and both build the same 114-wide conditioning vector**, because the chi block is
a function of `CHI_K_PAD` and not of the parameter set — so no width guard anywhere can see the
difference. What differs is the inferred dimension, and the parameter dropped is **`f_scale`**, which
§4.1 step 5 names as the retrain's headline hypothesis (unmeasured → measurable, 213× on `‖g‖`). A
smoke run at the default therefore exercises a box that omits the thing the retrain exists to measure,
and the only place it shows is one line of the banner. The script's own documented invocation was the
defaulting one; it now carries the full command and a warning, and `[cfg] rescale order` joined its
WHAT-TO-WATCH list. A cross-*load* between the two boxes is still caught, by the `param_keys` guard —
what was uncovered is running the wrong one in the first place.

## 2026-08-12 (later) — the smoke train, re-run after the device fix: clean

`scripts/smoke_train.py`, chi, `TOBS_S=4.5`, master bounds + `master_spont`. **All four stages
completed, 8218 s** (prior 754, posterior 6499, validate 195, infer 771) — against 8827 s on
2026-08-11, so no regression in cost.

| signal | result | reading |
|---|---|---|
| **training masked** | **38.2 %** (269/704) | against §4.3.6's 36.8 % and the 2026-08-11 run's 36.7 %. Well inside one standard error at this sample size — **the chi path survived the C-11 refactor, the preallocated accumulators and the S-1 threading** |
| validate masked | 38.3 % (69/180) | §4.3.6 records 47.2 %; that reference is itself 180 probes, and §4.3.6 already attributes the stage to small-sample spread |
| PPC masked | 64.6 % (3685/5700) | §4.3.6 records 79.9 % and says to expect it to FALL as the posterior improves. It is a readout of the posterior, not of the probe machinery |
| OOM retries, either level | **0** | nothing needed rescuing |
| `[mem]` breadcrumb | present, per batch tag | the forensic line works |
| `[checkpoint]` lines | **0** | correct: smoke runs disable checkpointing by default (see below) |

> The pooled figure across all three stages is 61.1 % and means nothing — the stages have different
> expected values by design. **Always separate them before reading a smoke train.**

## 2026-08-12 — C-11 built; and three backlog items turned out to be already done

The retrain is being driven by hand through the GUI, so this session did not start it. It closed the
last item flagged as *"the main risk left on a long run"* and cleared the non-retrain backlog.

### The thing that would have made a naive C-11 useless

**`V` is not reproducible across processes.** `build_latent_fisher_rotation` seeds its *noise* under
`fork_rng`, but its **operating points** come from `latent_prior.sample(...)` on the caller's global
RNG, which nothing seeds; training is ground-truth-free, so both the `z_med` and `z_samp` draws run.
A restarted process therefore computes a different `V` → a different `T_train` → different **latent**
targets for every batch after the seam, silently mixed with the pre-crash rows. That is exactly the
corruption C-11's own entry called *worse than crashing*, and a checkpoint that saved only RNG and
rows would have produced it **under the retrain's own configuration** (rotation ON).

So the resume decision lives in `build_posterior`, above the rotation block: it peeks the checkpoint,
reuses the stored `V`, and skips the Fisher. Recorded as trap **X10**.

### What was measured rather than assumed

| | measured |
|---|---|
| `torch.save` of a slice **view** | **8,001,492 B** for 100 rows of a `(200000, 10)` buffer; `.contiguous()` **8,001,633 B** (a row-slice is already contiguous, so it is a no-op); `.clone()` **5,566 B**. At the production shape that is ~47 MB vs 4.35 GiB per checkpoint — ~435 GiB over a run, presenting as *"checkpointing is slow"*. Trap **X9**, pinned by a file-SIZE test. |
| Qt's `svg` image-format plugin | **absent** — `QImageReader.supportedImageFormats()` has no `svg`, so `QIcon("x.svg")` is silently NULL. The app icon ships as PNGs rendered offline from the SVG by `QSvgRenderer` (the QtSvg *module*, which is present). |
| `gen_training_data` reproducibility | bit-identical across two seeded runs in **all three modes** — the gate that had to pass before the loop was touched, and the thing that makes "the resume is bit-identical" a real claim. Needs `np.random.seed` as well as `torch.manual_seed` (trap X8). |
| a resumed run vs an uninterrupted one | **`torch.equal` on both returned tensors**, with the kill placed so that both the cadence write and the cancel-path write are exercised. |
| the POWER-CUT case, out of process | a checkpointed run **SIGKILLed** at 12 s (exit 137, so no handler ran at all), resumed in a **fresh process**: the last cadence commit stood at 3/6 batches with its shards intact and `state.pt` readable, and the resumed run's output was **bit-identical** to an uninterrupted reference. This is the one the in-process tests cannot cover, and the one C-11 exists for. |
| the real `build_posterior` on the CARD | chi + rotation ON + CUDA + checkpointing, with `gen_prior` replaced by a same-typed stand-in so the stability sweep does not dominate. First call: returns, writes the checkpoint, stores `V` `(13,13)` on the CPU and a `(7,13)` probe. Second call: *"Reusing the Fisher rotation stored with the training checkpoint (2/2 batches — COMPLETE, so generation will be skipped)"*, and **zero** batches re-simulated. Confirms on the real path both fixes that had been reasoned to rather than observed — the CPU→CUDA rehoming and the complete-checkpoint `V` reuse. |

### The smoke train earned its keep: a CUDA/CPU bug no CPU test could reach

`scripts/smoke_train.py` ran chi + rotation on the GPU and **failed** in `build_posterior`, after the
Fisher rotation had already burned over an hour:

```
File "core/SBI/training_checkpoint.py", in bijection_probe
    out = theta_transform(z.to(torch.float32))
File "core/SBI/reparam.py", in _call
    return x @ self.M
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0,
              different from other tensors on cpu
```

The probe grid was built on the **CPU**; a rotated transform holds `V` in
`OrthogonalTransform.M` on `cfg.hw.device`, and `x @ M` across devices is a hard error, not a silent
promotion. It fires **only** with rotation ON *and* a GPU — the retrain's exact configuration — so
every one of the 55 CPU tests passed while the real path was broken. **The general lesson: a test
suite that runs on the CPU cannot certify a `.to(device)`; only a run on the card can.**

Looking for siblings of it immediately found a second, worse one: the checkpoint stores `V` on the CPU
(deliberately — that is what makes a checkpoint portable), so `build_posterior` must **rehome** it
before rebuilding the rotated bijection. Without that, the *first GPU resume with rotation ON* would
crash in the same matmul — the exact run C-11 exists to rescue. An audit of every tensor the
checkpoint stores or returns confirmed the rest are correct (all come back on the CPU by design;
`inits`, `V` and the RNG states are rehomed by their consumers, the schedule is read via `.item()`,
and the probe is compared CPU-to-CPU). Both are pinned by a CUDA test that early-returns with a note
off-GPU, and whose negative check confirms the old CPU-grid path really did raise.

### The box does not identify the prior

The checkpoint's identity started out as config fields only — and the training rows are drawn from the
**prior**, which the box does not pin: `_gmm_fingerprint`'s own docstring already said *"two runs over
the same box produce different fits"*. So rebuilding the prior and restarting would have resumed into
the **same** directory and spliced rows from two different distributions, with every declared field
matching and nothing to complain. `prior_fingerprint` is now part of the identity, which is also what
turns §4.1's *"save your prior and reuse it"* from advice into a guard — before the fix that
instruction was simply untrue.

### A regression the feature introduced into the TEST SUITE, caught by looking

Turning checkpointing on by default made the full-pipeline tests write real checkpoints into
`Resources/Checkpoints/` — three appeared on the first run (spontaneous / forced / chi, `run_size=8`,
`n_runs=2`). They are marked **complete**, and a complete checkpoint **short-circuits generation and
returns its stored rows**. So the first suite run would create them and every run afterwards would
skip `gen_training_data` entirely *while reporting a pass*: green, and testing nothing. The same
applies to a repeated `smoke_train.py`, whose entire job is to exercise that path.

Both harnesses now rebind `orchestrator.TRAINING_CHECKPOINT_EVERY = 0` — on the orchestrator, not
`config`, for the same snapshot-at-import reason `smoke_train` already rebinds three other constants.
Pinned by `test_the_suite_does_not_write_checkpoints_into_the_real_resources_tree`. **The general
shape is worth remembering: a cache whose hit is invisible turns any harness that shares its key into
a no-op.**

That guard needed a second pass of its own. Written as *"no `train_*` directories exist"* it fails on
any machine that has actually run a retrain — i.e. every machine where it matters — so it now
snapshots the directories present at module import and asserts only that the suite **added** none.
The lesson generalises past this one test: **a guard phrased over global state has to be scoped to
what the code under test controls, or it becomes a tripwire on real use.**

### Also fixed, and the pattern is worth noting

`gen_training_data`'s accumulators were plain lists whose final `torch.cat` allocated a second full
copy while the list was still live — **8.7 GiB** at the end of a multi-day run. They are preallocated
now. That alone would not have helped: `train_nn` immediately did `data = data[valid_idx]`, a boolean
gather that reinstates the same peak, so it is now guarded by `if not valid_idx.all()`
(behaviour-identical, since the box round-trip cannot produce a non-finite latent — trap X4).

### Three backlog items were already built

**B-c**, **B-e** and the `(lo, hi)` half of **S-1** all shipped in `20a1a52` and §6 was never updated;
`mpl_theme.py` and `icons.py` even cite the backlog IDs in their own docstrings. The same was true of
§9.1's items 2 and 3, §9.2, and §4.1's closing "Still open" paragraph about the Infer tab. **The
lesson is about the document, not the code:** an item marked OPEN was taken at face value and nearly
rebuilt. Every one of those rows now carries what actually landed, including where the implementation
diverged from what the row proposed — B-e used a bundled icon FONT rather than `.qrc`/SVGs, because
rendering icons as text is what makes them recolour for free on a theme flip.

What genuinely remained there was small and is now done: the χ probe table's `✕` was the last
unmigrated glyph button, and **PRISM had no application icon at all** (`setWindowIcon` appeared
nowhere, so the taskbar showed the interpreter's mark). S-1's second half — per-parameter prior type —
is `schema_version 3`'s `"box"`, named for what it controls: it changes the COORDINATE the latent GMM
and flow work in, and does **not** move prior mass, because `UserPrior._global_map` still samples
uniformly in the linear box. Making the sweep geometric is a separate science decision and was
deliberately not taken.

### Three bugs on the post-retrain critical path

All three sat on §4.1 steps 6–7, i.e. the first thing that happens after the retrain finishes:
`sbc_characterize.py` wrote un-suffixed outputs, so the four-run stratified sweep the suffix exists to
serve overwrote itself down to one stratum; it also omitted `chi_max_cycles` where its sibling passes
it; and `retrain_convergence.py` decided "was this rotated?" by testing whether the `.rot.pt` file
**exists** — which `save_posterior_artifacts` now writes unconditionally, so it refused every posterior
the current build produces, naming the wrong reason.

## 2026-08-11 — the retrain OOM'd AGAIN, outside everything the first fix guarded

Second failure, and the message shape was the diagnosis: a **bare, unwrapped**
`CUDA error: out of memory`, where the 2026-08-10 one had carried
`NadrowskiModel euler_compiled failed after 81616 steps (batch=2048, ...)`. `Simulator.__sols` wraps
everything raised inside the solver into `SimulationError`, so an unwrapped OOM **rules the solver
out** — and `_gen_obs_retry` wraps only `_gen_obs_one`, so it never saw this one.

The line before it was `chi: 1347/22528 probes masked`, emitted at the END of `gen_chi_block` inside
the per-batch loop. So: during generation, inside a batch, outside the simulator. (22528/2048 ⇒ K=11.)

### Two problems, only one of which had fired

**What killed it:** everything in a chi batch that is not the solver, all unguarded and all linear in
rows — the zero-drive tensor (~2.29 GiB), the per-probe `force` rebuilt K times (~2.29 GiB each),
`idx_c` (0.92 GiB int64, live across the whole K loop), `x_spont_dim`, `gen_stats`, and — worst,
because I introduced it — the `torch.cat(outs, dim=1)` reassembly in `gen_obs`, which runs **only on
split batches** and holds every chunk plus the result at once.

**What had not fired yet, and would have:** `append_simulations` was called with no `data_device`, so
sbi defaults it to **cuda**, moves the whole (10.24M, 114) tensor to the GPU, filters it (8.7 GiB
transient), and calls `warn_if_zscoring_changes_data` **unconditionally** — `torch.unique(x, dim=0)`,
a full z-score, then `torch.unique` again: **≥13 GiB**, at the end of a multi-day run that is not
checkpointed. It fires on **any** successful generation, chi or not. sbi's own warning text says it
can be ignored when z-scoring is off, which chi mode always is (§3.6).

### Four changes

1. **`data_device="cpu"` + a capped z-score check.** Both halves needed: `data_device` alone
   *relocates* the ≥13 GiB to host RAM and costs minutes of lexicographic sort. Capped rather than
   disabled, because non-chi modes do z-score and the diagnostic is real there — and it is a
   *proportion* test with a 10 % tolerance, so a **200k strided** sample answers it. Strided, not a
   prefix: rows are grouped by batch and each batch is one `(t_scale, T)` stratum.
   The patch rebinds the name in **every module that holds it** — `npe_base` does
   `from sbi.utils import warn_if_zscoring_changes_data`, so patching `sbi.utils.sbiutils` alone is a
   no-op. Found by identity scan rather than a hard-coded path, because sbi has already moved that
   module once.
2. **`gen_obs` preallocates** and each chunk writes into its slice via a new `out=` parameter.
   Preallocating *without* also removing `_gen_obs_one`'s clone is a **regression at k=2** — the live
   chunk would exist both in `out`'s span and as the return. Peaks (result 2.29, solver 6.87 GiB):
   k=2 5.73→5.73, k=4 4.58→**4.01**, k=8 4.58→**3.15**.
3. **Breadcrumbs.** `_BATCH_TAG` — batch index, `t_scale`, `T`, `n_fine`, `N_points`, rows — set
   before the first allocation that can fail and consumed by the chi mask warning and both OOM
   messages. Plus a `[mem]` line every 250 batches reporting **peak** allocated/reserved (reset each
   interval, so it reads "worst batch in the last 250" — the series that trends up before a death).
4. **A per-training-batch OOM retry** (`_rows_with_oom_retry`). The batch is the right unit: its
   `(t_scale_k, T_k)` is an *index* into a pre-filtered Sobol array, and the probe count, multipliers
   and durations are now drawn **above the seam** — so halving is a **repartition**, not a resample.
   Shrinking `run_size` globally would instead change the *number of strata*, i.e. the training
   distribution. Prefer the inner retry (a batch halve re-runs ~12 simulations plus a full
   `gen_stats`); the outer is the backstop for what the inner cannot see.

### What made the refactor safe, and what it caught

The loop body became a closure that slices ~30 batch-level names to its row range. **Any one left at
full width silently corrupts the batch**, so the gate was a **bit-identity harness**: seeded
`gen_training_data`, all three branches, `torch.equal` before and after. It passed 6/6 — and building
it turned up that **`gen_training_data` is not reproducible from a torch seed alone**: the initial
conditions come from `np.random.randint`, so two runs of identical code differed by max|diff| 30.6
until numpy was seeded too. Worth knowing for any "seeded run reproduces" claim.

It also caught a real slip: `inits` is passed **positionally** to `gen_chi_block`, so it survived the
keyword-based slicing sweep. That surfaced as `Batch size: 2 cannot differ from dim 0 of parameters
tensor` — loudly, but only because the simulator cross-validates those two. A name that broadcast
instead would have produced plausible, wrong training rows.

## 2026-08-11 (later) — smoke train after the OOM work: clean, with one honest gap

`scripts/smoke_train.py`, chi, `TOBS_S=4.5`, on a card with 13.6 GiB free (browsers closed). **All
four stages completed, 8827 s** (prior 547, posterior 7192, validate 262, infer 826).

| signal | result | reading |
|---|---|---|
| training masked | **36.7 %** | matches §4.3.6's 36.8 % and the pre-refactor 35.7 % — C-6/C-8 survived the batch-retry refactor |
| PPC masked | 59.0 % | expected; a readout of posterior quality at smoke sizes, not of the probe machinery (§4.3.6) |
| OOM retries, either level | **0** | nothing needed rescuing at 13.6 GiB free — which is the point about closing browsers, not evidence the retries work |
| sbi "Moving x to the data_device" | **absent** | `data_device="cpu"` took effect; the tensor was already on the host and stayed there |
| `[mem]` breadcrumb | peak allocated 1.09 GiB on batch 1 | the line works, and its peak-not-instantaneous figure is the one that trends before a death |
| batch tag | `training batch 2/4 [t_scale=5.171, T=3.44e+04, n_fine=279096, N_points=34387, rows=32]: chi: 17/96 probes masked` | exactly the forensic line whose absence made the 2026-08-11 failure take a full investigation to localise |

### The gap this does NOT close, stated plainly

At smoke sizes the training set is `NUM_RUNS=4 x RUN_SIZE=32 = 128 rows`. `append_simulations`
therefore handled a **128-row** tensor, not the 10.24M-row one whose `torch.unique` was the ≥13 GiB
bomb. So this run proves the fixed path **executes correctly** — `data_device="cpu"` is accepted, the
scoped z-score patch applies and restores, nothing downstream breaks — and proves **nothing** about
the memory claim at scale, which rests on arithmetic plus a unit test asserting the check never sees
more than 200k rows.

The same caveat applies to both retries: with 13.6 GiB free, neither fired. Their behaviour is pinned
by CPU tests with injected failures (including a full `gen_training_data` recovery), and by one GPU
run under artificial pressure that exercised the *predictive* split rather than the reactive retry.
**Reproducing a genuine reactive CUDA OOM needs a second process holding VRAM, which has not been
staged.** Treat the retries as tested-by-construction until a real run exercises them.
>
> **UPDATE 2026-08-27: they fired, and the first real firing found a bug in them.** The Phase-1
> retrain hit a genuine reactive OOM at batch 351/5000 and `_rows_with_oom_retry` caught it exactly
> as designed — then died on its own `empty_cache()`. See the 2026-08-27 (later) entry and trap X13.
> The retries are no longer untested by real hardware; the release path they depend on is now
> best-effort and pinned by `test_the_release_path_survives_a_failing_empty_cache`.

## 2026-08-10 (later) — the retrain OOM'd, and the free-memory reading is why

The first real retrain died hours in with
`AcceleratorError: CUDA error: out of memory` at `batch=2048, segs=3` — the RAW DRIVER form, not
`torch.OutOfMemoryError`.

### The guard was fine. Its input was lying.

`_max_sim_batch` budgets from `config.memory_budget_elements` → `torch.cuda.mem_get_info()`. Measured
at one instant on the 16 GB RTX 5070 Ti with an ordinary desktop running:

| source | free VRAM |
|---|---|
| `torch.cuda.mem_get_info()` — what the guard reads | **15037 MiB** |
| `nvidia-smi` — what the card actually has | **5814 MiB** |

**Optimistic by 9.2 GiB, which is exactly the desktop.** Under Windows/WDDM the OS virtualises VRAM:
other processes' surfaces are *evictable*, so the driver reports them to you as free. The planner then
green-lights a batch the driver can only satisfy by **evicting Firefox**, and returns
`cudaErrorMemoryAllocation` only when eviction cannot keep up — which is why the failure wears the raw
driver form and why it lands hours in rather than immediately.

The cost model itself is accurate to <1 % (predicted 5.34 GiB against a measured 5.35 GiB peak at
B=2048/n_fine=100k), and it already sizes **per geometry**, which matters because `n_fine` swings from a
median ~40k to a p99 ~283k. **Only the budget was wrong.**

### Three changes, and one that was deliberately NOT made

1. **A learned budget** (`_BUDGET_CAP_ELEMENTS`, AIMD). On an OOM at N elements, cap to 0.8·N — we now
   *know* N does not fit. After 32 clean batches, probe up 10 %, re-clamped by the reading on every
   call so it can only ever be more conservative than it. Deliberately AIMD **on the budget in bytes,
   not on the batch width**: fitting is width × *geometry*, and the planner already handles geometry.
   Adapting width would fight it and oscillate.
2. **A reactive retry** in `gen_obs` (`_gen_obs_retry`). Catches an OOM and re-runs that chunk at half
   the width, recursively, down to `_MIN_SIM_CHUNK`. `except RuntimeError`, narrow, so `WorkerCancelled`
   still reaches `Worker.run`. Announces every retry on **stderr** — `warnings.warn`'s "once per
   location" filter would collapse hundreds of events into one line, and parts of `gen_training_data`
   run under `simplefilter("ignore")`.
3. **`TRAINING_RUN_SIZE`**, a *ceiling* (0 = off, and it should stay off). A ceiling rather than a
   replacement because `smoke_train.py` and three tests shrink runs by writing `cfg.hw.batch_size`
   directly; a replacing knob would override them and quietly drive the CPU suite at 1024.

**NOT done: lowering the training batch.** An earlier draft dropped it to 1024 and doubled
`TRAINING_NUM_RUNS`. That pays ~2× wall-clock on *every* batch to solve a problem only the ~25 % tail
has, and halves the `(t_scale, T)` stratum count for a fixed row budget. Per-geometry splitting pays
k× only where it is needed. The historical **5000 × 2048** shape stands.

**NOT done: `expandable_segments:True`.** Measured a no-op here — *"expandable_segments not supported
on this platform"*, `is_expandable=False`; the Windows cu130 build does not define
`PYTORCH_C10_DRIVER_API_SUPPORTED`. It would be right on Linux. Note also that torch 2.9 renamed the
variable: `PYTORCH_CUDA_ALLOC_CONF` warns "deprecated, use `PYTORCH_ALLOC_CONF`". Do not re-try this.

### A C-8 memory regression, found on the way

`chi.lock_in_batched._mask` returned a `(B, chunk)` **float64** tensor. At B=2048/chunk=8192 that is
128 MiB per mask; as a **bool** it is 16 MiB, and the results are **bit-identical** (verified with
`torch.equal` at three chunk sizes) because it is only ever multiplied into a float64 tensor, where
torch promotes it to exactly 1.0/0.0. Measured +128 MiB → **+16.1 MiB**. C-8's docstring claimed the
masking left "the memory bound ... unchanged"; it is now corrected to +16 MiB rather than deleted.

Invisible until now because the only end-to-end exercise was the smoke train at `RUN_SIZE=32`, where
128 MiB is 2 MB. **Neither lock-in test would have caught it** — they pin numerics, not allocation.

### What this actually buys, stated honestly

Under 7 GiB of artificial pressure the 245k-step geometry at batch 2048 **completed in 136 s against
18.6 s unpressured** — a 7× slowdown, because the predictive guard split it all the way to the
`_MIN_SIM_CHUNK` floor. It survived, which is the point; it was not fast. **Closing the browsers is
still the largest single lever.** These changes make the run survive a busy desktop, not enjoy one.

> **Measuring headroom: use `nvidia-smi`, never `torch.cuda.mem_get_info()`.** They disagree by 9.2 GiB
> on this machine. Working budget ≈ `15.92 GiB − (what other processes hold) − ~0.8 GiB CUDA context`.

**Known gap:** the prior stability sweep bypasses `gen_obs` — `Priors/*_prior.py` build a `Simulator`
and call `.simulate()` directly — so the retry does **not** protect it. Acceptable today: fixed, small
geometry (`n_stab_fine = 40_000` → ~2.3 GiB at batch 2048) and it completed in 561 s. Not coverage.

## 2026-08-10 — C-9/C-10: `logcyc` leaves the Fisher set, and the retrain is unblocked

C-9 was catalogued on 2026-08-08 as inferred-from-construction, explicitly **not measured**, with a
note that measuring it needed a real rotation. Measured now, on
`build_latent_fisher_rotation` at `m=4, n_points=1` (structurally faithful, ~100× cheaper than the
`REPARAM_FISHER_M=48 / _POINTS=8` defaults), instrumented from outside by spying on
`chi.fisher_features` rather than by editing production.

### The prediction was half right, and the half that was wrong is the more useful half

**C-9 reproduced, but mildly.** `chi5_logcyc` pinned exactly as predicted — std 8.7e-05, ratio
2.9e-05, the same signature the map showed — and then reached `max|J|` = **6.0**, the *smallest* chi
row, against the map's 2.0e4. The amplification turns on whether the ±dz arms happen to straddle a
`floor()` step. At that operating point they did not.

**That is worse news than a clean reproduction, not better.** A deterministic 2.0e4 would announce
itself on the first run. An intermittent one hides through every check and fires on one of the eight
operating points a production rotation averages, over a prior spanning ~4 decades of Ω₀ — and `V` is
frozen into the sidecar. "Measured, mild, intermittent" is the profile of a thing that ships.

**C-10 reproduced hard**, and it is what actually justified the fix: `chi0/1/2/3_logcyc` agreed to
**six significant figures** (`max|diff|` ~2e-5 against entries of 37.44). With the ceiling clear,
`logcyc_j = log(mult_j) + log(f_peak) + log(T_obs)` — both constants vanish under standardization, so
the row **is** `A3_log_fpeak`'s, K times over.

### One change closes both

`CHI_FISHER_CHANNELS` drops to `("logmag", "cos", "sin")` and `fisher_features` takes **one
argument**. Every `logcyc` row was a duplicate, a degrading duplicate, or quantization — never
independent information — so nothing was lost, and the one-argument signature turns trap CHI10's
whole class of mis-wiring into a `TypeError` rather than a rebind.

Verified surgical: after the change the rotation's chi block has 18 rows instead of 24, **zero**
pinned (minimum std/|feat| = 0.0149, four orders clear), and every surviving row's std and `max|J|`
is **identical** to the pre-fix run — it removed exactly the bad rows and perturbed nothing else.

> **No artifact shape changes, which is less obvious than it sounds.** The Fisher's feature width is
> an *internal* dimension: `J` is `(n_features, P)` and only ever leaves as `V = eigenbasis(JᵀJ)`,
> which is `(P, P)` — parameter space. So `chi_layout`, `chi_k_pad`, `chi_elem_w`, `input_dim` and
> every width the sidecar guards are untouched, and this changes the *values* of `V`, never any
> shape. That is exactly why it was free today and would not have been after a posterior existed:
> nothing would have failed loudly, the coordinates would just have been different ones.

### The lesson worth keeping, since the trap survives its own fix

`fnoise` is a **denominator**. A channel that barely varies with theta is an amplifier, not a quiet
row. And the asymmetry is what makes it invisible: an **exactly** constant channel is harmless
(`0/1e-9 = 0` — which is why chi mode's 11 zeroed Group-G columns cost nothing), while a **nearly**
constant one writes order-1-to-1e4 entries into the matrix defining the flow's coordinate system,
with `V` still orthogonal to 1e-4 and every test green. Apply that test to any channel added to any
Fisher in this repo.

### Same day — C-2, C-3 and C-7, which closes the C series

**C-7** promoted two literals: `PRIOR_SWEEP_ITERATIONS` and `PRIOR_SWEEP_BATCH` (0 = follow
`hw.batch_size`, i.e. unchanged behaviour). Worth knowing beyond the refactor — the subclasses'
`batch_size % num_iterations` guard is **vacuous**: `construct_prior` passes `batch*iterations` down
as `batch_size`, so the modulo is always 0. Do not rely on it to catch a bad value.

**C-2 + C-3 were one problem wearing two hats,** and the fix is a shared predicate rather than two
features. The Infer tab now has an add/remove probe table submitting `(recording, frequency_Hz)`
pairs, and a **Plan probes…** button that says what is in band for this cell and how long each probe
must be recorded. Those two must agree, and the only way to guarantee that was to stop them each
owning a copy of the rules: `chi.probe_verdict` is now the single source of the refuse / mask /
truncate split, and `orchestrator.build_experiment_obs_chi` was refactored to call it. It returns a
verdict rather than raising, because a planner has to be able to *describe* a bad probe without dying
on it; the runtime is what turns `"refuse"` into a `ValueError`.

Two design points that are easy to get wrong and are pinned by tests:

- **One widget per row, not parallel lists.** Deleting from the middle of two parallel lists pairs
  recording *k* with frequency *k+1*, and that is invisible: a lock-in at the wrong frequency decays
  like a sinc, so it returns a smaller number rather than an obviously wrong one.
- **A rebuild must PRESERVE the rows.** They hold hand-typed drive frequencies and browsed paths — a
  record of a bench session that already happened, which nothing in the GUI can regenerate. Contrast
  the forcing rows, which *are* derivable from the config and so are rebuilt freely. The table seeds
  only when empty; it never tops up or trims.

And the blank-frequency trap is real rather than theoretical: `FloatField.value()` returns `0.0` on
unparseable text, so an empty box is indistinguishable from a deliberate zero — and 0 Hz is a genuine
DC probe the lock-in would happily attempt.

## 2026-08-08 — step 3 run at last, after fixing the script that produces it

§4.1 step 3 was "the only thing left before the retrain" and had been runnable-looking for three
commits. It was not: `degeneracy_map` sliced `gen_chi_raw(...)[:2]`, binding `logcyc_v` to `u`.
Full account in §4.4.1 and trap **CHI10**; the result itself is a **GO on the information question**
and a **NO-GO pending C-9** on the machinery.

### The trap was written down, then walked into six lines later

The comment at the call site described the `u` hazard precisely and the code below it did exactly
that. Both existed in the same commit. The lesson taken is not "read comments harder" — it is that a
warning a reader must act on is worth less than an assertion the program acts on, so the guard is now
a **printed equality**: predicted `log(cycles integrated)` against the measured 4th Fisher channel,
per probe, `SystemExit` on a mismatch over 0.5. Under the bug it read log(0.03)…log(0.30) — negative
across a sub-resonance band, against a correct +1.12…+3.00. Unmissable, and it costs one array read
of an ensemble already simulated.

### One fix uncovered the next, and only the second matters for the retrain

With the unpack right, the top probe's `logcyc` — pinned by `CHI_MAX_CYCLES` to log(20) plus
`floor()` quantization — became the largest entry in the whole Jacobian: `max|J|` = 2.0e4 against 289
for the largest real feature, setting `sigma[0]` and a condition number of **56 000** by itself.
Removing it: **1033**. The relative-std guard **provably could not catch it** (ratio 2.7e-5, 27×
above the gate), which is why the second detector is arithmetic — the cap already knows which probes
it capped. **`decorrelate.fisher_at` has the identical construction over the identical feature set**
(C-9), and that is the retrain blocker; it was not run here.

### Three measurements, three different lifespans

- **Reproduces:** the §4.4 baseline, on a rebuilt encoder, new band, new K, and C-4/C-6/C-8 in
  between (`k` 0.040→0.092, `x_scale` 0.043→0.127, cond 2212→2093). The set conditioning lost nothing.
- **New:** the payload table, readable for the first time. chi's **phase** channels carry `lam` and
  `f_scale`; its **magnitude** channel carries `t_scale`; `k`~`x_scale` is led by `A1_mean` at 6× any
  chi feature and does not move (0.98 → 0.96 — the third independent confirmation).
- **Retired:** the condition-number claim. §4.4 read 2300 → 2034 as a chi gain; a forced control at
  `T_obs = 1.0 s` moves it 1669 → 2212, *larger and in the other direction*. That number was never
  measuring chi. The same control halves `k`/`x_scale`'s apparent gain but leaves `t_scale`, `lam`
  and `f_scale` untouched — so those three are chi's actual product.

### Two process notes

- **`TOBS_S` defaulted to 1.0 s and that silently voided 12 of 54 feature rows** — three probes under
  `CHI_MIN_CYCLES`, returning demeaned residual drift, with `resolution_filter=False` so nothing
  masked them. Drift is finite, reproducible and has a healthy std, so it passes every *statistical*
  guard; only cycle arithmetic sees it. The band's span (10×) exactly equals the cycle window's
  (10×), so no `T_obs` clears both edges — err above the collision at `T* = 2.93 s`.
- **A ratio needs its numerator quoted beside it.** `f_scale`'s unique handle *falls* 0.871 → 0.701
  under chi, which reads as a regression and is the single largest gain in the table: forced `‖g‖` is
  0.018, i.e. not measured at all, and chi takes it to 3.789.

## 2026-08-07 — the smoke train's finding, and C-6/C-8

The first end-to-end chi run (`scripts/smoke_train.py`, new) completed all four stages and then said
the thing that mattered: **77 % of training probes were masked**, a typical row conditioning on ~2
live probes out of 12 slots. Well-calibrated and at the prior is `posterior_chi_08042026`'s
signature, and this would have reproduced it from a new direction — days of GPU to rediscover a known
failure. Tests 169 → 171. Full account in §4.3.3–§4.3.6.

### Two wrong diagnoses, both mine, both corrected by measuring

1. **"It's a (band × T) interaction — short-`T` batches mask the band's lower half."** Plausible
   arithmetic, and false: the live fraction is **flat in `T`** (37–47 % from <2 s to >30 s) and flat
   in multiplier. `scripts/chi_mask_audit.py` (new) separates the predicates the runtime warning
   lumps together; the `CHI_MIN_CYCLES` floor turned out to be the **only** one ever active, and the
   driver is the row's own **Ω₀** — 0 % live below 3 Hz against 98 % above 30 Hz, over a prior
   spanning ~4 decades.
2. **"Those low-Ω₀ rows are non-oscillatory junk, so screen the prior."** `peak_freq` is an argmax
   and does return the bottom of a 1/f spectrum when there is no peak, so this was worth believing.
   Measured peak-over-median PSD power: **thousands**. They are genuine sharp oscillators. A 0.5 Hz
   bundle really oscillates at 0.5 Hz and its χ at 0.015–0.15 Hz really is unmeasurable in ≤ 60 s, so
   the mask is correct physics and screening them out would have been wrong.

### The fix, in two parts

**C-6, per-ROW placement.** `chi.resolvable_multipliers` lifts each row's multipliers into the
sub-band its own Ω₀ can resolve. Free — frequencies were always per-row, only the multipliers were
shared. 41 % → 46 % live. A variant bounding placement by `CHI_MAX_CYCLES` too was implemented and
**reverted**: `hi/lo` = 10 and `ceiling/floor` = 10, so requiring both leaves one feasible multiplier
for all but a single `Ω₀·T` — every probe on one frequency, measuring no shape.

**C-8, per-ROW durations.** The real one. `lock_in_batched` gained `(B,) n_samples` (masked, not
sliced, so the chunked float64 accumulation is untouched) and `gen_chi_raw` an `(B,) N_row`. 46 % →
**64 %** live, inert rows 55 % → **33 %**.

### What kept the work honest

**Live count is not the objective.** chi(ω) measures the shape of a curve, so a placement rule tuned
on live count alone will happily park every probe on one frequency — perfectly resolved and carrying
nothing. The audit reports the live probes' frequency SPAN for that reason, and it is what would have
caught the rejected variant had the arithmetic not. Both parts improved span (4.99× → 5.38×) and cut
single-probe rows (7.9 % → 1.4 %) *alongside* the count, which is why this is a fix and not a trade.

### Two process notes that cost real time

- **`RUN_SIZE` shrank the PRIOR's stability sweep too**, because `cfg.hw.batch_size` is both. The
  sweep is iteration-bounded, so that made the prior worse without making it faster: **527 s at batch
  2048 versus >70 min and unfinished at batch 32.** `smoke_train.py` now applies `RUN_SIZE` only from
  the posterior stage on; the underlying literal is **C-7**.
- **Do not edit source under a running suite.** `test_cufft_plan_cache_is_cleared_between_training_batches`
  reads its own source with `inspect.getsource`, which re-reads from disk, so an edit mid-run made it
  fail spuriously and cost a full hour-long re-run to clear.

## 2026-08-06 (last) — the band's high edge (C-5)

11 multipliers through the edge region under the cap, M=32. Full result in §4.3.2. **`CHI_FREQ_BOUNDS
= (0.03, 0.3)` stands** — the band moves in neither direction, which is the least eventful possible
outcome and took a measurement to establish rather than a preference.

The useful part is *which* criterion turned out to carry the answer. Under the ceiling, |chi| CV and
SNR discriminate **nowhere** in 0.03–0.6 — the reproducibility problem is entirely a duration problem
and the cap solves it across a range twice the configured band. That leaves two candidate edges with
wildly different answers: circular phase scatter (crossing 0.5 rad at ~0.15×) and entrainment
(accelerating at 0.35–0.4×). **Phase scatter has no knee** — it grows smoothly, ~1.15× per grid
point, so any threshold on it reports the threshold. **Entrainment does** — own-peak step ratios go
0.86 → 0.77 → 0.60 → 0.43, a change of character rather than a crossing. So the band's ceiling is
entrainment's, and `PHASE_MAX` is now advisory (`inf`) rather than a gate.

That change deserves scrutiny, since relaxing a threshold to reach a conclusion is exactly how one
fools oneself: the justification is that the quantity was measured to be smooth, not that the answer
was inconvenient. The remaining judgement — that a phase-incoherent probe is *half-useful* rather
than *corrupt*, since its `log|chi|` CV is still ~0.08 while an entrained probe reports the drive back
to itself — is reasoning, not measurement, and is labelled as such in §4.3.2. The experiment that
would settle it is two multi-day training runs, which is not worth spending before a first working
posterior exists.

## 2026-08-06 (later still) — the lock-in duration ceiling (C-4)

The fix C-1 pointed at, implemented and measured. Tests 166 → 169.

### The wall, bracketed

`scripts/chi_f0_sweep.py` now evaluates a LIST of prefix lengths from one set of simulations — a cap
is a shorter prefix of a trace that already exists, so bracketing costs lock-ins, not simulations.
(The first draft of this entry claimed the sweep was already free; it was not — `CYCLE_CAP` was a
scalar and sweeping it meant N full re-runs. That is what `CYCLE_CAPS` fixes.) At M=48 over in-band
probes, worst |chi| CV climbs 0.042 → 0.198 across caps 8 → 32 and hits 0.456 at 36: the wall is at
**32–36 cycles**, approached as a steady climb rather than a cliff. `CHI_MAX_CYCLES = 20` — the
numbers and the reasoning for not taking the CV-optimal 12–16 are in §4.3.1 and at the constant.

Two things the summary logic got wrong on the first pass, both caught by running it: `max(clean_cap)`
is not the wall — the metrics are **not** monotonic in the cap, so the scan has to stop at the
*first* failing cap; and a large cap BINDS FEWER POINTS by construction (only traces longer than it),
so the top of that table is thin evidence and now says how many points each row rests on.

### Where the ceiling lives, and why

In `pipeline.gen_chi_raw`, not in `gen_training_data`. It is the definition of the measurement rather
than a policy of one caller, and training, `decorrelate.feats`, the PPC and
`build_experiment_obs_chi` must agree or the network conditions on an observable it was not trained
on — with nothing raised. Keyed on the batch's **highest** `f_peak` so no row exceeds it (`N_k` is one
scalar while `freq_k` is per-sample; erring long would put the fastest rows, the ones nearest the
wall, back over it). It is **not** a filter: nothing is masked or dropped, the segment is shortened.
It is carried on the `SimConfig`, written to the sidecar and checked on load, for the same reason as
the band — it sets the `logcyc` a recording reports, and `logcyc` is the channel the encoder uses to
weigh a probe. On the experimental path an over-long recording is **truncated with a warning**: the
recording is fine, only its tail is unusable, and the experimenter is entitled to know.

`SimConfig.__post_init__` now rejects `chi_max_cycles <= CHI_MIN_CYCLES`. The floor masks and the
ceiling shortens, so a crossed pair truncates every probe below the floor and masks the entire set —
which surfaces as "none of the supplied recordings produced a usable probe", true and useless, and in
training as a silent all-pad chi block.

### Two vacuous tests found while extending the fixtures

`tests/test_artifact_consistency.py`'s `_sidecar` fixture omitted `chi_layout` / `chi_k_pad` /
`chi_elem_w` and carried a stale `forcing_dim=12` (3K at K=4 — layout 1, retired). Since
`_assert_mode_matches` checks those first and raises on the first mismatch, **every test built on
that fixture was passing on a missing key rather than on its own override**:
`test_a_posterior_over_different_parameters_is_refused` would have passed with the model and
param-order checks deleted. Fixed by making the fixture write what a current build actually writes,
deriving `input_dim`/`forcing_dim` from the same helpers rather than as literals, and — the part that
matters — asserting the **unmodified** sidecar is ACCEPTED before testing that each override is
refused. That baseline is what makes the override meaningful, and adding it is what exposed the stale
`forcing_dim` immediately.

Also stale and user-visible: the Prior tab reported chi conditioning as `χ(3·K)`, the retired
layout-1 width. Under the probe set it is `CHI_ELEM_W · chi_k_pad` and does not depend on K at all —
so the message told the user the opposite of what the mode now does.

### And one the new test caught in itself

`test_lock_in_duration_is_capped_at_chi_max_cycles` first built its passive trace with `torch.randn`.
`chi.peak_freq` is the argmax of the rfft, so Ω₀ was **the argmax of noise** — a different value on
every run, and the experimental leg's in-band check therefore passed or failed by coin flip. It
passed standalone and took the whole suite down on the next full run. Now a pure tone on a known bin,
with the probe placed at the band's top edge relative to it. **Any chi test that calls `peak_freq` on
synthetic data needs a deterministic Ω₀**, because every band and mask predicate downstream is
defined relative to it.

*Process note, since it cost a full hour-long re-run:* the suite before that was polluted by editing
`core/SBI/pipeline.py` **while `test_user_sbi.py` was running**.
`test_cufft_plan_cache_is_cleared_between_training_batches` reads its own source with
`inspect.getsource`, which re-reads from disk, so the shifted file made it fail spuriously. Do not
edit source under a running suite — this one takes ~1 hour and the failure looks exactly like a
regression.

## 2026-08-06 (later) — the `T_obs` gate (C-1), and a fixed-K lever

Backlog **C-1**, the last measurement standing between here and a multi-day retrain. It did not
confirm the band; it overturned the reasoning behind it. Tests 165 → 166.

### What was measured

`scripts/chi_f0_sweep.py` gained `T_obs` as an outer loop, so every criterion must now hold at every
recording length rather than at the single `T_obs = 5 s` slice the band was fixed from. Four things
were added along with the axis, each because the old sweep could not have caught this:

| added | why |
|---|---|
| **driven/undriven SNR** | the denominator is the lock-in FLOOR — the same estimator run on the *passive* ensemble at the same probe frequency. Free (those traces already exist), and it is the ratio §4.1 nominated to replace `CHI_MIN_CYCLES`. A CV alone cannot distinguish "reproducibly small" from "reproducibly measuring nothing". |
| **circular phase scatter** | `arg(chi)` is wrapped; a linear std reports ~1.8 rad for a perfectly concentrated phase straddling ±π. |
| **cycle count + would-be mask** | printing production's gate beside the measurement is what turned a table into a diagnosis. |
| **a capped-duration column** | lock in over a ≤`CYCLE_CAP`-cycle prefix of the *same* trace. This is the control that separates "this frequency is unusable" from "this duration is unusable" — and they call for opposite changes. |

Two grid defaults were also stopped from drifting: the multipliers are now derived from
`config.CHI_FREQ_BOUNDS` (the old hard-coded `[0.1, 0.5, 1, 2, 10]` was the *retired* band) plus one
above-band control, and the `T` grid is capped at what the Sobol pre-filter can actually draw.

### The finding

**Full length: only `0.03×Ω₀` survives every `T_obs`** — three multipliers *inside* the configured
band collapse. **Capped at 20 cycles: all 9 failures recover**, the above-band `0.6×` control
included. Same simulations, same seeds, same frequencies; fewer samples in the sum. So the band's
*interior* is a duration problem, not a frequency one. The cap is not a universal solvent, though:
the band's high edge `0.3×` stays marginal on phase coherence even capped, and a second run over the
retired `(0.1, 10.0)` multipliers confirms that at ≥0.5×Ω₀ the failures are entrainment and a
response below the bundle's own spontaneous activity — neither of which shortening the window
touches. Details, tables and the mechanism caveat are in **§4.3.1**; the trap is **CHI9**.

The 2026-08-05 entry below reached a partly-right conclusion for a wrong reason. Its control asked
whether a longer recording *helps* at high multipliers, found it did not, and inferred the probes were
unrecoverable at any amplitude or length. The direction it did not test is the one that mattered:
longer is *worse*, and shorter is the fix. At `T_obs = 5 s` every high multiplier is already past the
wall (`0.5×` → 57 cycles, `1×` → 114, `10×` → 1140), so a frequency limit and a cycle limit are
indistinguishable in that slice — and the sub-resonance band inherited an exclusion it did not need
while the durations that actually break it went unbounded.

**A second run over the retired `(0.1, 10.0)` multipliers checked the obvious follow-up — does the cap
bring the near-resonance band back? It does not** (table in §4.3.1): at ≥0.5×Ω₀ the drive entrains the
bundle or the response falls below its own spontaneous activity, and neither is a duration artifact.
So §4.3's sub-resonance band survives; what changes is that it needs a **duration ceiling** to be
usable across the `T_obs` range training draws. That is backlog **C-4**, and it gates the retrain in
C-1's place.

**What was deliberately NOT done:** narrowing `CHI_FREQ_BOUNDS`. The script's own arithmetic
recommends `(0.03, 0.03)` — mechanically correct, and a one-frequency band cannot measure the shape
of a curve. The band verdict now prints full-length and capped columns side by side so the next
reader cannot make the 2026-08-05 inference by accident.

### A T_obs ceiling nobody had written down

At the master cell's `t_scale = 3.73`, `gen_training_data`'s Sobol pre-filter
(`n_fine ≤ min(N_ND_MAX, len(t))`) admits `T_obs ≤ 26.9 s`, not the nominal `T_MAX_EXP_S = 60 s`.
The reachable range is a function of `t_scale` (≈7.4 s at `t_scale = 1`, ≈296 s at 40), so "training
draws `T ~ logU[1 s, 60 s]`" is true of the *draw* and false of the *accepted* set. The script
computes and prints the ceiling and flags any point beyond it — which is what caught its own T grid
rounding 0.01 s past the cap and landing 10 fine steps outside the training distribution.

### The fixed-K lever (§4.1 step 5)

`gen_training_data` accepted `chi_n_freqs` and **never read it** — the §7.11 accepted-but-ignored
class, and a live trap, since the obvious "fix" is to honour it and thereby destroy the
K-agnosticism the probe-set layout exists to provide. It is now removed from the data-generation
signatures (it stays on `SimConfig`, in the sidecar, and in `chi_multipliers_for`, where it genuinely
means "probes this OBSERVATION supplies") and replaced by `chi_k_fixed`, which fixes the per-batch
count **and skips `_subset_probe_rows`** — without that second half a "K = 6" stratum is silently a
mixture over 1..6, i.e. the pooled measurement the stratification exists to avoid.
`scripts/sbc_characterize.py` exposes it as `CHI_K_FIXED` and states the stratum on its own line,
because a stratified and a pooled run produce identically-shaped reports.

Pinned by `test_chi_k_fixed_holds_the_probe_count_for_a_stratified_calibration`, which asserts
against the *packer's own mask* rather than a live-slot count: a masked probe and a pad slot are
bitwise identical after `pack_probe_block`, and the training draw legitimately masks some probes, so
"the row has exactly K live slots" is not a true invariant at any K. The first draft asserted it and
failed for that reason — the test is written the way it is because the naive form is vacuous.

Also fixed while here: `sbc_characterize` and `retrain_convergence` never passed `chi_k_pad`, so both
silently fell back to the live `config.CHI_K_PAD` rather than the config's — and the pad width is
frozen into the artifact (trap CHI7).

### Scripts the 2026-08-05 consolidation broke

Not previously catalogued. `reparam_wiring_smoke.py` passed the archived `cell_2.txt` as a
**positional** argument to `_common.script_cfg`, where explicit args beat the `CELL` env var, so it
could not be run at all; `diagnose_fmax.py` defaulted to the archived `cell.txt`. Repointing them at
`_common.DEFAULT_CELL` then exposed a second layer: that default is `master_spont`, and **five**
diagnostics read `cfg.forcing_idx["amp"]` unconditionally, so they need a bounds file that declares a
Forcing section. Hence `_common.FORCED_DEFAULT_CELL`, `_common.assert_forced()`, and a `default_cell=`
seam on `script_cfg` that changes the fallback without overriding an explicit `CELL`.
`reparam_wiring_smoke` still cannot complete — it inherits its latent prior from a saved posterior and
`Resources/Posteriors/` is empty — but it now says so, with a `BASE_POST` knob, instead of dying on a
raw `FileNotFoundError`.

## 2026-08-06 — chi(ω) conditioning became a K-agnostic probe SET (layout 2)

Same session as the entry below; separated because it is a different kind of change. Tests 145 → 165
(new suite `tests/test_chi_set_encoder.py`, 20). Design was produced by a mapping/judging pass over
the codebase whose adversarial reviewers found three defects that would otherwise have been written;
all three are recorded below because each is invisible at runtime.

### What changed

The chi block was a fixed `3K` vector whose slot index IMPLIED the probe's frequency. It is now a
padded SET of 6-channel probes carrying frequency explicitly (§3.6), consumed by a
permutation-invariant encoder (`core/SBI/chi_encoder.ChiSetEncoder`). Probe count and placement are
both free: `expected_forcing_dim` keys on `CHI_K_PAD` instead of K, which is the single line that
lets one posterior serve any probe count with no width guard loosened. `gen_chi_block` split into
`gen_chi_raw` + a packing wrapper; training draws K per batch, jitters placement, varies per-probe
durations and subsets rows; the experimental path takes `(recording, drive_frequency_Hz)` pairs.

### Three defects the review caught

| defect | why it is invisible |
|---|---|
| **`u` poisons the Fisher.** `log(f_k/f_peak)` is theta-INDEPENDENT under a deterministic multiplier grid, so its float32 std is ~2.5e-8 — below `fnoise`'s 1e-9 floor. A central difference turns pure rounding into `J` entries of ORDER 1, in the matrix that defines the flow's coordinate system. | `V` stays orthogonal to 1e-4 and every existing test passes. Fixed: the Fisher uses `CHI_FISHER_CHANNELS` (4 per probe, no `u`, no `mask`). |
| **A masked max pool is n-biased.** `E[max of n iid N(0,1)]` = 0.564 / 1.267 / 1.629 at n = 2 / 6 / 12, so max-pooling writes a ~1.07σ LOCATION shift into every channel purely as a function of probe count — exactly the K-dependence the change removes. | Looks like a standard DeepSets choice. Fixed: no max pool; a fixed-knot Nadaraya-Watson quadrature, which carries no n-dependent shift. |
| **sbi's `z_score_x` breaks permutation invariance.** The default fits a per-COLUMN affine over the conditioning vector; over probe slots two orderings of one set are scaled differently. The near-constant mask column also becomes a ~1e7 amplifier under sbi's 1e-7 min-std clamp. | Nothing errors; the encoder's guarantee is simply void. Fixed: `z_score_x="none"` under chi, with `EmbeddedNet` standardizing per CHANNEL over live probes only. |

Also fixed: `resolution_filter=False` is mandatory in the Fisher (the filter depends on `f_peak`,
hence on theta, so a probe crossing the threshold between ±dz arms is a step of 1 over a 1e-9 floor);
`reparam.posterior_mode`'s `fdim % 3 == 0 → chi` numerology is gone (it decoded any 6-parameter drive
as chi, and `6·5 == 3·10 == 30` means width cannot identify a layout at all).

### Two places the implementation deliberately DIVERGES from the spec

Both were tested and the spec was wrong; do not "restore" either.

1. **The embedding is NOT fully K-invariant, and must not be.** `ChiSetEncoder.pool()` returns a
   CURVE half and a SAMPLING half. The curve half is K-stable (measured: 0.115 at K = 8 vs 12,
   against ~0.5 for a 10% resonance shift; 0.94 at K = 2 vs 4). The sampling half carries `log1p(n)`
   and per-knot coverage and is K-dependent BY DESIGN — a 2-probe observation genuinely is less
   informative than a 12-probe one, and hiding that would make the flow overconfident on sparse data.
   "K-agnostic" means *can consume any count*, not *returns the same answer regardless*.
2. **An under-resolved probe is MASKED, not refused** (§4.3). Training masks and keeps the row, so
   refusing at eval would reject observations the network handles fine — routine at the band's low
   edge, where a 1 s recording cannot resolve a 0.03× probe however well it was made. Structural
   mistakes (bad/aliased/out-of-band frequency, count over the pad, ALL probes masked) still refuse.

### Three tests changed meaning, not just paths

Each was correct about the old design, so each was rewritten rather than deleted:
`test_chi_batch_never_outruns_the_nd_time_grid` inferred K by dividing call counts (undefined now)
and demanded every probe see the summary's exact sample count — it now keys each lock-in to its batch
and asserts `0 < n_chi <= n_spont`, since fewer is the deliberate per-probe duration and more is still
the 2026-07-28 clipping defect. `test_chi_fisher_rotation_...` now spies `gen_chi_raw` and asserts the
filter is off. `test_posterior_mode_...` built its chi case at `forcing_dim=18` on the retired
numerology, and gained the negative control: a wide `forcing_dim` with no `chi_layout` marker must SAY
it cannot tell rather than guess.

## 2026-08-05 — Consolidation, artifact identity, and the chi(ω) band

Prompted by a day lost to forensics: `posterior_chi_08042026` validated well but was uninformative,
and establishing *what it had even been trained on* meant comparing GMM component counts against the
prior files on disk. Nothing in the software recorded or checked it. Tests 135 → 145.

### The chi(ω) band was wrong, and that explains the uninformative posterior

Two new measurement scripts, both cheap and both worth re-running per cell:

- **`scripts/build_master_cells.py`** — picks a cell's parameters by measurement rather than by hand.
  Found the oscillatory window in `f_max` is only **0.03 wide** at the old `s = 0.65` (which is why
  the archived `cell_2` at 1.06 sat on essentially the only oscillatory point available to it, and
  why nothing else in that family reproduced). `s = 0.95` opens it to `[1.12, 1.87]`.
- **`scripts/chi_f0_sweep.py`** — sweeps drive amplitude × probe frequency against BOTH failure
  modes. Results and the decisive `T_obs` control are in §4.3. Headline: |chi| is reproducible only
  **below ~0.25×Ω₀**, and above that no amplitude or recording length helps.

`CHI_FREQ_BOUNDS` (0.1, 10.0) → **(0.03, 0.3)**; `CHI_F0` 0.2 → **0.15**. At K=10 the old band put
8 of 10 probes in the unusable regime, each costing a full simulation per observation — which is
where most of that run's five days went.

### The rotation exclusion was never measured, and was wrong

`scripts/degeneracy_map.py` run forced-vs-chi on the same cell (§4.4): chi improves every parameter's
unique handle, but leaves `k`~`x_scale` at 0.95 (0.98 forced). So `build_posterior`'s
`not cfg.chi_mode` exclusion — "chi already attacks the degeneracy the rotation targets" — is false.
`decorrelate.feats` now builds its Jacobian over the mode's own feature set and the exclusion is gone.
Also corrected: **`lam`~`t_scale` was never degenerate on this cell** (0.59 forced), so §4.2's 0.96
was a property of the archived cell, not of the model.

### Consolidation

Five per-cell Nadrowski boxes → one `master.txt` + `master_spont.txt` pair (identical ND sections)
and three cells sharing one ND+rescale block. `cli.resolve_bounds_for_cell` resolves sibling-first
then the folder's `master.txt`, so every pre-existing cell resolves exactly as before while three
cells can share one box. Everything else moved to `archive/2026-08-05-pre-consolidation/`.

### Artifacts now describe themselves

| Was | Now |
|---|---|
| `build_prior`'s load path checked **nothing** | refuses on model / param ORDER / box mismatch |
| `build_posterior` checked only the log-mask | sidecar carries model, param_keys and the full box |
| `_assert_mode_matches` checked only mode + width | also model and parameter order |
| `load_eval_bijection` rebuilt the box **from cfg** | rebuilds from the SIDECAR, warns on divergence |
| nothing tied a posterior to its prior | GMM fingerprint, checked in `validate_calibration` |

The `load_eval_bijection` row is the one that mattered: its docstring always claimed eval was
self-describing, but the box came from whatever config happened to be loaded — so a posterior trained
against one bounds file and evaluated against another silently decoded every latent sample through
the wrong edges. New suite `tests/test_artifact_consistency.py` (9 tests) pins all of it.

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
  **Still true**, and `diagnose_fscale`'s own `posterior_07012026.pt` default is gone too. Deliberately
  left: every one of these needs a real name after the retrain anyway, and inventing a default now
  would only be wrong differently. **Set `POST` explicitly.**
- ~~`diagnose_fscale`'s verdict recommends `REPARAM_LOG_PARAMS=['f_scale']`, which Appendix A below
  records as **tried → failed → reverted**. The script has never been told.~~ ✅ **FIXED 2026-08-13** —
  docstring and H2 verdict branch both corrected; H1 kept, because it is a rotation diagnostic and the
  retrain is rotated. It took two sessions to act on a defect this document had already written down,
  which is the same lesson as the 2026-08-12 entry's "three backlog items were already built": **a
  row in this file is only worth writing if someone later acts on it.**

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
