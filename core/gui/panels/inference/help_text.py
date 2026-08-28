"""The inference tabs' "?"-badge help copy, keyed by field. Shared: several keys serve two or
three tabs (tobs, num_runs, run_size), so there is one table rather than one per tab. Also
surfaced in Settings -> Help via the inference_tabs module path."""


# Help text shown by the "?" badge next to each option. Drafted from the code/science; user reviews.
HELP = {
    "model": "Which model to fit. NADROWSKI is the state-dependent-drift model the pipeline is tuned "
             "for; HOPF and BP are alternatives. User-defined models are inferable too if they have no "
             "forcing (spontaneous dynamics) and at least one parameter — but their calibration is not "
             "pre-tuned, so validate SBC/TARP per model.",
    "bounds": "Parameter-bounds file (Resources/Bounds/<model>) defining the inference box: which "
              "parameters are inferred and the prior range of each.",
    "cell": "A cell file (Resources/Cells/<model>) whose parameter values are the ground truth — the "
            "simulator uses them to generate a synthetic observation.",
    "tobs": "Observation duration in seconds. Longer traces carry more information but cost more to "
            "simulate.",
    "prior": "Load a saved prior (.pt), or choose “(from scratch)” to construct a new "
             "stability-screened parameter prior.",
    "posterior": "Load a trained posterior (.pt), or “(from scratch)” to train a new one. Training "
                 "from scratch needs a prior; loading an existing posterior does not.",
    "tsnpe_obs": "The observation to refine around. Written by the Infer tab at INFERENCE time -- "
                 "an amortized posterior has none when it is SAVED, which is why there is a picker "
                 "here rather than an automatic choice. The round refuses unless the stored "
                 "observation is bitwise the one currently loaded.",
    "tsnpe_hpd": "How much of the posterior's credible mass the truncated region must contain. "
                 "0.999 by default and deliberately generous: truncation permanently deletes prior "
                 "support, and no later round can recover it. A region that is too WIDE only costs "
                 "simulations.",
    "tsnpe_dirs": "How many of the best-constrained Fisher directions to truncate; the rest keep "
                  "full prior width. Truncating every axis would cut the FLAT directions (k, "
                  "delta_E, temp sit at or near prior) on noise rather than on information.",
    "sweep_iters": "GLOBAL sweep rounds. Total candidates screened for stability = rounds x "
                   "candidates-per-round, so this is the coverage of the broad Sobol census that "
                   "SEEDS the local flood-fill. The sweep is ITERATION-bounded, which is why the "
                   "next field is not a speed dial.",
    "vram_ceiling": "HARD ceiling on what ONE simulation batch may plan to hold on the GPU. "
                    "0 = off, and off is right on an idle card. It is NOT a substitute for freeing "
                    "VRAM — with nothing free it can do nothing, because not even a floor-sized "
                    "chunk fits. What it buys is keeping a run that HAS headroom inside real VRAM: "
                    "past that, Windows pages the batch into shared system memory rather than "
                    "failing, and it runs up to 9x slower with nothing to say why (measured "
                    "2026-08-27: 21.67 GiB completed on a 15.92 GiB card). Set it to about the free "
                    "VRAM nvidia-smi reports, minus ~1 GiB for the CUDA context. Splitting costs "
                    "wall-clock on the batches it touches. Not remembered between sessions, on "
                    "purpose.",
    "sweep_batch": "Candidates per global round; 0 = follow the hardware batch. NOT a speed knob — "
                   "the sweep is iteration-bounded, so shrinking this makes the prior WORSE without "
                   "making it faster (measured 527 s at 2048 against >70 min and unfinished at 32).",
    "sweep_max_sets": "Accepted parameter sets that STOP the local flood-fill. This is the point "
                      "cloud HDBSCAN clusters and the GMM is fitted to, so it buys COVERAGE of the "
                      "stable manifold rather than statistical precision — a 10-D GMM with a few "
                      "components needs nothing like 175,000 points.",
    "sweep_step": "Random-walk stride for the flood-fill, in PHYSICAL parameter units. Too small and "
                  "the walk never leaves its seed points; too large and it steps ACROSS the stable "
                  "manifold instead of tracing it.",
    "sweep_units": "ND time units the stability screen integrates each candidate over. This defines "
                   "what 'stable' MEANS, so it changes the prior's support and not just how long the "
                   "sweep takes — a longer screen rejects slow instabilities a short one accepts.",
    "cluster_size": "HDBSCAN's floor on what counts as an ISLAND of stable parameters. Its label "
                    "count is handed straight to the GMM's n_components, so this sets how many "
                    "MODES the prior has — a different component count is a different prior, not "
                    "a faster one.",
    "cluster_samples": "How conservative HDBSCAN's density estimate is. Higher declares more "
                       "points NOISE, which it leaves unassigned and the GMM never sees — so this "
                       "thins the cloud the prior is fitted to as well as splitting it.",
    "fisher_m": "Ensemble size per latent perturbation in the Fisher rotation. Cost is linear in "
                "this; under chi each evaluation already pays (1+K) simulations instead of 2.",
    "fisher_dz": "Latent central-difference step for the Fisher Jacobian.",
    "fisher_points": "Operating points the Fisher is AVERAGED over. 1 is ground-truth-only, which "
                     "re-correlates away from it — averaging is what makes one LINEAR rotation "
                     "valid across the whole prior. ⚠ A resumed run reuses the checkpoint's stored "
                     "V and ignores all three of these: the rotation is not reproducible across "
                     "processes, so a resume must reuse the stored one.",
    "flow_hidden": "Flow width: hidden units per spline transform. With a COMPLETE training "
                   "checkpoint on disk this can be re-tried without re-simulating (~46 h against "
                   "~57 h for a full run) — that is what the checkpoint is a cache for.",
    "flow_transforms": "Flow depth: number of spline transforms. Same re-try economics as the width.",
    "flow_lr": "Adam learning rate for the density estimator.",
    "flow_patience": "Early-stopping patience in epochs. The 2026-08-25 run stopped at 130 on a "
                     "patience of 20, with its best validation loss at epoch 110.",
    "cal_n": "Calibration datasets drawn for SBC/TARP.",
    "cal_scales": "(t_scale, T) operating points those datasets are spread over. ⚠ This is "
                  "t_scale's EFFECTIVE SAMPLE SIZE, not a speed dial: lowering it is a DIFFERENT "
                  "measurement, not a faster one. 'SBC flat on all 13' is strong for 11 of them and "
                  "materially weaker for t_scale, and this number is why: every row in a "
                  "calibration batch shares that batch's t_scale, so their ranks are not "
                  "independent samples of it.",
    "num_runs": "How many training BATCHES to simulate. Each batch is one Sobol (t_scale, T) "
                "operating point that every row in it shares — so this is the data budget AND the "
                "timescale/duration diversity of the training set, and it is what wall-clock scales "
                "with. Raising it is the honest way to buy a better posterior. ⚠ It is part of the "
                "training checkpoint's identity: changing it means an in-progress run cannot be "
                "resumed.",
    "run_size": "CEILING on simulations per batch; 0 = follow the hardware default. This is a VRAM "
                "escape hatch, NOT a speed control — the SDE solver is kernel-launch-bound, so a "
                "narrower batch is not faster (measured 7.37 s at 2048 against 7.74 s at 1024, i.e. "
                "the smaller batch is slightly slower). Lowering it trades training rows for peak "
                "memory about 1:1, and you have to raise Batches to get those rows back, which does "
                "cost wall-clock. The per-batch splitter already handles the geometry tail, so reach "
                "for this only if you see splitting on most batches. ⚠ Also part of the checkpoint "
                "identity.",
    "infer_mode": "Simulated: infer on a synthetic observation from a cell’s ground truth. "
                  "Experimental: infer on your own recording (a driven spontaneous+forced pair, or — "
                  "for a no-forcing model — a single passive recording).",
    "spont": "Path to the recorded spontaneous/passive (undriven) trace (.csv or .npy; last column "
             "= values).",
    "forced": "Path to the recorded forced (driven) hair-bundle trace (.csv or .npy; last column = "
              "values).",
    "forcing": "The value of this sinusoidal-drive parameter used in the forced recording, in the "
               "shown units.",
    "chi_mode": "Multi-frequency susceptibility χ(ω). Instead of conditioning on ONE drive, each "
                "observation is a passive recording plus K single-tone driven recordings, and the "
                "conditioning carries the χ(ω) curve. This is the only lever on the information "
                "ceiling: a single passive trace sees only the PRODUCTS D·A_nd and (λ/k)·τ, whereas "
                "the shape of χ(ω) separates κ, λ, x_scale and t_scale individually. Costs about "
                "(K+1)/2× the training time. Training and inference must both use it.",
    "chi_k": "How many drive frequencies THIS observation is measured at, and therefore how many "
             "forced recordings an experiment must supply. More probes resolve the curve better but "
             "cost linearly more simulation. It does NOT have to match the posterior: the network "
             "conditions on a probe SET, so it accepts any count up to the slot capacity below.",
    "chi_k_pad": "Probe SLOTS the network reserves — its capacity, not a probe count. Training draws "
                 "probe counts from 2 up to this, so the encoder learns to handle any of them. It is "
                 "FROZEN into every posterior trained with it (it fixes the input width), so raising "
                 "it later means retraining; pick generously. Costs only input columns — the "
                 "encoder's parameter count does not depend on it.",
    "chi_f0": "Non-dimensional drive amplitude for every χ probe. χ = response/drive is independent of "
              "amplitude in the linear regime, so this only needs to be small enough to stay linear "
              "(≲0.1) and large enough for the lock-in to beat the noise.",
    "chi_max_cycles": "Longest lock-in, in drive cycles, used for any one probe. A longer lock-in is "
                      "NOT a better one here: past roughly 30 cycles χ stops being reproducible at "
                      "fixed parameters, and the extra recording adds noise rather than signal. "
                      "Recordings longer than this are truncated, not rejected — the leading part is "
                      "exactly what the network was trained on. Frozen into any posterior trained "
                      "with it, so changing it means retraining.",
    "chi_range": "The K probe frequencies are placed log-spaced across this range, as MULTIPLES of each "
                 "observation's own measured spontaneous peak Ω₀ — so the probes track the resonance "
                 "wherever t_scale puts it, instead of sitting at fixed absolute frequencies.",
    "chi_passive": "The passive (undriven) recording. Its power spectrum sets Ω₀, which anchors the "
                   "frequency of every driven recording below.",
    "chi_forced": "One row per single-tone forced recording: the file, and the frequency you "
                  "ACTUALLY drove it at in Hz. Type the real frequency rather than the nominal "
                  "one — a lock-in decays like a sinc, so being off by a fraction of 1/T_obs "
                  "destroys the estimate while every number still looks plausible. Any count from 1 "
                  "to the posterior's probe slots works, at any frequencies in band: the encoder is "
                  "permutation-invariant and carries each probe's frequency explicitly. Use "
                  "'Plan probes…' to see what is in band for this cell and how long each must be.",
    "chi_f0_si": "The physical drive amplitude used for the forced recordings. χ cancels the amplitude "
                 "in the linear regime, so this only sets the lock-in normalisation.",
    "bounds_source": "Pick a bounds file, or edit the numbers directly. Direct entry starts FROM the "
                     "selected file, because the parameter names and their order are fixed by the model "
                     "(simulators bind parameter columns by position) — only the numbers are yours to "
                     "change. Switching to “Edit values” reloads from whichever file is selected.",
    "cell_source": "Pick a cell file, or edit its ground-truth values directly. As with bounds, direct "
                   "entry starts from the selected file and lets you change only the numbers.",
    "units": "The units the numbers in your bounds and cell files are written in. This DECLARES what "
             "those numbers mean — it never converts them, so changing it re-interprets your files "
             "rather than rescaling them. Plot axes and unit labels follow this choice. Frequency is "
             "special: the pipeline consumes a drive frequency as inverse cell TIME, so for an `ms` cell "
             "the frequency unit must be kHz — a mismatch is reported when the config is built.",
    "units_mode": "Take the units from the model's units file (Resources/Units/<model>/units.txt), or "
                  "type them directly as space-separated tokens.",
    "units_text": "Space-separated unit tokens, one per physical dimension — e.g. “nm ms pN kHz”. Any "
                  "unit pint understands works; they are matched to quantities by DIMENSION, not order.",
    "reparam_rotate": "Rotate the flow's latent coordinate into the simulation-Fisher eigenbasis, so a "
                      "strongly correlated posterior (κ↔x_scale, λ↔t_scale) becomes axis-aligned and the "
                      "flow can calibrate it. The rotation is orthogonal, so it adds and removes no "
                      "information — off is exactly the plain pipeline. Cost: computing the rotation runs "
                      "extra simulations at several operating points BEFORE training starts. Results so "
                      "far have been a redistribution rather than a clean win, so it is worth comparing "
                      "SBC with it on and off. Available in all three observation modes; under χ(ω) the "
                      "Fisher is built over the χ features, which costs (K+1)/2× a forced-mode rotation.",
}

