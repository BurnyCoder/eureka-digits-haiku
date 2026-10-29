# Experiments: registered comparisons (cycle 001)

**Status: complete for every registered comparison, plus the sensitivity arm, the shuffled-label control, two baselines, and one post-hoc analysis (labelled exploratory).**

- **Primary hypothesis H1 (decay 0.01 improves test accuracy): inconclusive under the registered rule.** The paired interval is degenerate, [0.00, 0.00] pp. No test prediction changed between decay 0.01 and decay 0 in any seed. This is not evidence of equivalence.
- The shuffled-label control, the logistic baseline, and the class-prior floor behaved as preregistered.
- Decay did change the weights and the training trajectory. It did not change any test argmax at this budget.

Predictions are those in `reports/preregistration.md` §4, written before any test-split output existed. This file reports the observations against them. No prediction was edited after the observations.

## 0. Registration chain

| Item | Value | Source |
|---|---|---|
| Pre-registration start time | 2026-10-29T17:57:50.785+00:00, commit `287f40bd` ("Preregister research cycle 1 before results") | `research/registration.json` |
| Protocol SHA-256 | `5edcfb3f…ec257` (matches registration and all 15 records) | `research/protocol.json`, `scripts/write_experiment_cards.py` |
| `study.py` SHA-256 | `34415a87…6d67d39` (matches `research/core-code.json`, recorded at core publication 18:00:40 UTC, and all 15 records) | `research/core-code.json` |
| Core publication | commit `2b10207`, PR #1 | `research/core-publication.json` |
| Pilot publication | commit `0b37f0d`, PR #2; checkpoint on Hugging Face | `research/pilot-publication.json` |

The three SHA-256 values are rechecked by `scripts/write_experiment_cards.py` before any card is written. The protocol, code, and records were not changed in this phase.

## 1. Question, objective, and immediate goal

- **Project goal:** Does AdamW weight decay improve held-out accuracy of a fixed 64-128-10 ReLU MLP on the bundled `sklearn` digits data? (`research/protocol.json`, `question`)
- **Subgoal of this phase:** Run the registered comparisons on the test split, verify each record independently, and report every outcome, including failures.
- **Analysis kind:** H1 and its primary metric are *confirmatory*. Every other analysis here is *exploratory* and labelled per item.

## 2. Methods

**Data.** `sklearn.datasets.load_digits` [S05]: 1797 images, 8×8 pixels, 10 classes, pixels divided by the constant 16 [S05]. Stratified split with `split_seed = 42`: train 1078 / validation 359 / test 360 [S05, `study.py: dataset`]. Disjointness and full coverage are asserted in `study.py`.

**Model and training (identical for every MLP arm).** `Linear(64,128) → ReLU → Linear(128,10)`, cross-entropy loss [`study.py: network`]. `torch.optim.AdamW` [S03] with `lr = 1e-3`, default betas and eps, `batch_size = 64` (17 steps per epoch, 850 steps total), `epochs = 50`, no early stopping, final-epoch model evaluated. Per-epoch reshuffling uses a generator seeded with the run seed. Deterministic algorithms are enabled [`study.py: fit_mlp`]. MLP arms ran on CUDA (NVIDIA GeForce RTX 5070 Laptop GPU); logistic and class-prior ran on CPU. Torch 2.14.1+cu130, NumPy 2.5.4 (per record).

**Hyperparameters.** The constants are the registered ones: learning rate, batch size, epochs, and architecture were fixed for fast feedback on an established task, not tuned on any split. The three decay values form a fixed local check, not a search. The protocol's `hyperparameter_rationale` gives the reasons.

**Primary metric and analysis (registered).** For each seed, `Δ_s = acc_test(decay 0.01) − acc_test(decay 0)`. The primary metric is `mean(Δ_0, Δ_1, Δ_2)`. The 95% CI uses 10,000 paired percentile-bootstrap resamples of the 360 test indices (`numpy.random.default_rng(2026)`, the same index draw for every arm) [`scripts/analyze_experiments.py`]. The bootstrap is conditional on the trained models and does not estimate training-population variance.

## 3. Experiment register

Every test-split record, with its method and analysis kind. Per-record cards: `reports/experiment-<id>.md`.

| ID | Method | Seed | Analysis kind | Weight decay | Test accuracy | Correct / n | Card |
|---|---|---|---|---|---|---|---|
| `regularized_s0` | regularized (MLP) | 0 | confirmatory | 0.01 | 0.96389 | 347 / 360 | [card](experiment-regularized_s0.md) |
| `regularized_s1` | regularized (MLP) | 1 | confirmatory | 0.01 | 0.96944 | 349 / 360 | [card](experiment-regularized_s1.md) |
| `regularized_s2` | regularized (MLP) | 2 | confirmatory | 0.01 | 0.96944 | 349 / 360 | [card](experiment-regularized_s2.md) |
| `unregularized_s0` | unregularized (MLP) | 0 | confirmatory | 0.0 | 0.96389 | 347 / 360 | [card](experiment-unregularized_s0.md) |
| `unregularized_s1` | unregularized (MLP) | 1 | confirmatory | 0.0 | 0.96944 | 349 / 360 | [card](experiment-unregularized_s1.md) |
| `unregularized_s2` | unregularized (MLP) | 2 | confirmatory | 0.0 | 0.96944 | 349 / 360 | [card](experiment-unregularized_s2.md) |
| `sensitivity_s0` | sensitivity (MLP) | 0 | exploratory | 0.001 | 0.96389 | 347 / 360 | [card](experiment-sensitivity_s0.md) |
| `sensitivity_s1` | sensitivity (MLP) | 1 | exploratory | 0.001 | 0.96944 | 349 / 360 | [card](experiment-sensitivity_s1.md) |
| `sensitivity_s2` | sensitivity (MLP) | 2 | exploratory | 0.001 | 0.96944 | 349 / 360 | [card](experiment-sensitivity_s2.md) |
| `shuffled_labels_s0` | shuffled-label control (MLP) | 0 | exploratory | 0.01 | 0.06111 | 22 / 360 | [card](experiment-shuffled_labels_s0.md) |
| `shuffled_labels_s1` | shuffled-label control (MLP) | 1 | exploratory | 0.01 | 0.11667 | 42 / 360 | [card](experiment-shuffled_labels_s1.md) |
| `shuffled_labels_s2` | shuffled-label control (MLP) | 2 | exploratory | 0.01 | 0.10833 | 39 / 360 | [card](experiment-shuffled_labels_s2.md) |
| `logistic_s0` | logistic regression baseline | 0 | exploratory | not applicable | 0.95556 | 344 / 360 | [card](experiment-logistic_s0.md) |
| `class_prior_s0` | majority-class floor | 0 | exploratory | not applicable | 0.10278 | 37 / 360 | [card](experiment-class_prior_s0.md) |

The pilot record `pilot_regularized_s0` (validation split, exploratory) is documented in `reports/pilot.md` and `reports/experiment-pilot_regularized_s0.md`. Its checkpoint is the one reused for `regularized_s0`. All 15 IDs are unique.

## 4. Results

### 4.1 Test accuracy by arm and seed

| Arm | seed 0 | seed 1 | seed 2 | Mean | Seed SD (ddof=1) |
|---|---|---|---|---|---|
| regularized (decay 0.01) | 0.96389 | 0.96944 | 0.96944 | 0.96759 | 0.32 pp |
| unregularized (decay 0) | 0.96389 | 0.96944 | 0.96944 | 0.96759 | 0.32 pp |
| sensitivity (decay 0.001) | 0.96389 | 0.96944 | 0.96944 | 0.96759 | 0.32 pp |
| shuffled labels (decay 0.01) | 0.06111 | 0.11667 | 0.10833 | 0.09537 | 3.00 pp |
| logistic regression (seed 0) | — | — | — | 0.95556 | — |
| class prior (seed 0) | — | — | — | 0.10278 | — |

One flipped test prediction is 1/360 = 0.278 pp. The binomial standard error of one accuracy near 0.968 at n = 360 is about 0.93 pp. The seed SD of 0.32 pp is about one flipped example.

### 4.2 Primary paired comparison (H1)

| Seed | Δ = acc(0.01) − acc(0) | Discordant pairs (reg-only / unreg-only) | McNemar exact p |
|---|---|---|---|
| 0 | 0.00 pp | 0 / 0 | 1.0 (no discordant pairs; defined as 1.0 in code) |
| 1 | 0.00 pp | 0 / 0 | 1.0 |
| 2 | 0.00 pp | 0 / 0 | 1.0 |

- **Primary metric:** mean Δ = **0.00 pp**, paired 95% bootstrap CI **[0.00, 0.00] pp** (`research/experiment_summary.json`, `paired.primary`).
- `y_pred` for the regularized and unregularized arms is element-wise identical in every seed. The bootstrap of an identically zero difference is exactly zero, so the interval is degenerate.
- **Registered decision rule:** the falsification rule (CI wholly below 0) is not triggered. The support rule (CI wholly above 0) is not met. The result is **inconclusive**. It is not an equivalence claim: the interval only says the metric did not move on these 360 examples.
- The McNemar test is reported for completeness. With zero discordant pairs it carries no sign information.

### 4.3 Training-level effect of decay (implementation check and P8)

| Seed | max \|Δ train loss\| over epochs | final train loss, reg − unreg | final validation loss, reg − unreg |
|---|---|---|---|
| 0 | 7.4e-4 | +7.1e-4 | +2.8e-4 |
| 1 | 8.4e-4 | +7.0e-4 | +3.5e-4 |
| 2 | 8.0e-4 | +7.7e-4 | +4.2e-4 |

The regularized and unregularized loss histories differ at every epoch in every seed. Decay is applied, so the zero test effect is not a silent no-op. `tests/test_study.py::test_adamw_decay_is_decoupled_from_gradients` checks the decoupled update directly with a zero-gradient step.

### 4.4 Weight-level effect (`scripts/check_decay_mechanism.py`, `research/decay_mechanism.json`)

Predicted cumulative shrink from the registered constants, if gradients had no effect: `(1 − lr·λ)^850`, which gives a factor of 0.991536 for λ = 0.01 (shrink 0.846%) and 0.999150 for λ = 0.001 (shrink 0.085%). This follows the decoupled update in `torch/optim/adam.py` [S04].

Observed relative change in the whole-vector L2 norm of weights and biases, `1 − ‖θ_reg‖/‖θ_unreg‖`, against the unregularized checkpoint with the same seed:

| Seed | regularized (λ=0.01) | sensitivity (λ=0.001) | fraction of entries changed (reg) |
|---|---|---|---|
| 0 | 0.4065% | 0.0406% | 100.00% |
| 1 | 0.4020% | 0.0392% | 99.99% |
| 2 | 0.4059% | 0.0415% | 100.00% |

The observed shrink is about **0.48×** the decay-only prediction in both arms, and the ratio is the same at both λ values. That is the observation. The likely cause is that gradient updates partly offset the decay. This is a hypothesis; it was not tested by decomposing the update. The zero-gradient unit test checks the decay-only factor, not the full training dynamics.

### 4.5 Predicted agreement across arms

Per seed, the regularized, unregularized, and sensitivity arms produce identical test predictions. Checked directly on `y_pred` for all three seeds; this is the same fact that gives the zero-width CI in §4.2. The shuffled-label arm uses the same test `y_true` ordering as the other arms.

### 4.6 Exploratory analysis: test negative log-likelihood (post hoc, not registered)

`scripts/exploratory_test_nll.py` → `research/exploratory_test_nll.json`. Paired over the same 360 examples, with the same bootstrap RNG seed (2026). **Exploratory.** It was added after the registered primary result, which has zero discordant predictions and so cannot show whether decay changes the output distribution at all. It is not part of the registered conclusion.

| Seed | NLL reg | NLL unreg | Δ NLL (reg − unreg) | Mean max-softmax confidence reg / unreg |
|---|---|---|---|---|
| 0 | 0.124932 | 0.124535 | +0.000397 | 0.93450 / 0.93513 |
| 1 | 0.124639 | 0.124161 | +0.000478 | 0.93377 / 0.93435 |
| 2 | 0.128358 | 0.127784 | +0.000574 | 0.92964 / 0.93026 |

Mean Δ NLL = **+0.00048 nats**, conditional 95% bootstrap CI **[+0.00024, +0.00070]**. The sign is positive in all three seeds: in this view, decay slightly raised test loss and lowered the predicted-class probability by about 0.0006 in every seed. That fits the shrink mechanism (smaller logits give softer probabilities). It is a correlation across three trained models, not an intervention, and it cannot be promoted to a confirmatory result in this cycle. Relative to the mean NLL of about 0.125, the effect is about 0.4%.

### 4.7 Predictions (preregistration §4) versus observations

| ID | Prediction | Observed | Verdict |
|---|---|---|---|
| P1 | \|mean Δ\| ≤ 1.0 pp | 0.00 pp | Held |
| P2 | Paired 95% CI includes 0 | [0.00, 0.00] pp; degenerate | Held (uninformative, see §5) |
| P3 | CI wholly above 0 (supports H1) | Not above 0 | Not observed |
| P4 | Pilot validation (regularized, seed 0) in 0.93–0.99; regularized test in 0.93–0.99; logistic test in 0.94–0.98 | Validation 0.96657 (pilot); regularized test 0.96389–0.96944; logistic 0.95556 | Held |
| P5 | Shuffled-label test accuracy in 0.05–0.16 (≈0.10); >0.20 indicates a bug | 0.06111, 0.11667, 0.10833 (mean 0.09537) | Held |
| P6 | Class-prior test accuracy ≈ majority-class fraction of the test split | 0.10278 = 37/360 (exact) | Held |
| P7 | Seed SD of test accuracy in 0.3–1.5 pp | 0.32 pp | Held (lower edge; one flipped example ≈ 0.28 pp) |
| P8 | Regularized and unregularized loss histories differ for the same seed | Differ at every epoch in every seed (§4.3) | Held |

The outcomes that would have supported the hypothesis were P3 and a positive primary mean. Neither was observed, which is consistent with the small-effect mechanism in §4.4.

## 5. What the test split can and cannot say

- The primary metric is **structurally insensitive** at this scale. Zero flipped test predictions across three seeds means the accuracy test carries no information about the direction of the decay effect. It is not evidence that the effect is zero in the population.
- The preregistered minimum detectable paired difference at an **assumed** ~3% disagreement rate was ≈2.5 pp (`reports/preregistration.md` §12). The observed disagreement rate is 0%, below that assumption, so the study cannot separate "no effect" from "an effect below the detection floor".
- The exploratory NLL result is the only analysis that detects any change in the outputs. Its size (≈0.0005 nats) is tiny, and it rests on three models.
- The registered CI is conditional on the three trained models. It does not include training-seed uncertainty beyond the per-seed SD, which is itself estimated from three seeds.

## 6. Failures, corrections, and how they were handled

**Corrections required from the prior attempt (the handoff listed them).** Each item was checked against the current artifacts:

| Listed correction | Status in this phase |
|---|---|
| Duplicate experiment identifiers | Not reproduced. All 15 record IDs are unique; `scripts/write_experiment_cards.py` asserts uniqueness before writing. |
| Protocol hash mismatch | Not present now. `research/protocol.json` hash matches `research/registration.json` and every record's `protocol_sha256`; asserted by the card script. |
| Missing pre-registration start time | Recorded in §0 from `research/registration.json` (`registered_at`). |
| Experiments require method and analysis kind | Every test-split record has its method and analysis kind in §3, and in its card. |
| Training code differs from published core | Not present now. `study.py` hash matches `research/core-code.json` and every record's `code_sha256`; asserted by the card script. The file is unchanged from the core commit. |

I did not verify the prior attempt's handoff file itself, so I cannot say which exact artifact the earlier check read. The current artifacts satisfy every listed condition.

**Earlier attempt history (preserved).**

1. **The original training run.** `results/experiments_run.log` (18:11 UTC) trained the 13 MLP and baseline records. It reused the pilot checkpoint for `regularized_s0`.
2. **Draft reports.** A previous experiments attempt left draft reports, now at `research/attempts/experiments_attempt1_draft.md` and `research/attempts/highlights_attempt1_draft.md`. The draft highlight 5 misquoted P5's window as 0.08–0.14. The preregistered window is 0.05–0.16. The verdicts in this report come from the preregistration and the records, not from the draft.
3. **Reuse run.** `results/experiments_attempt2_reuse.log` shows `study.py experiments` reusing all 14 records. `results/experiments_reuse_check.log` repeats it in this phase, with the same result. Nothing was overwritten.

**Checks performed in this phase.**

- `study.py check`: passed (train 1078 / validation 359 / test 360).
- `study.py experiments` (reuse path, in the cycle): all 14 records reused; each record's protocol and code hashes were verified by the fixture (`results/experiments_reuse_check.log`).
- `pytest` (13 tests): all passed.
- **From-scratch retraining.** `scripts/rerun_from_scratch.py` copied only the inputs `study.py` needs (fixture, protocol, registration, pilot publication, pilot record and checkpoint) into a fresh temporary directory outside the cycle, then ran `study.py experiments` there. It retrained 13 records from initialization and reused the pilot checkpoint for `regularized_s0`. All 15 records in the scratch directory matched the cycle's records on checkpoint SHA-256, `y_true`, `y_pred`, `example_ids`, `accuracy`, and per-epoch `history` (`research/rerun_comparison.json`, log `results/rerun_from_scratch_repeat.log`). The pilot entry is a copy, so its match is trivial. The earlier scratch log `results/rerun_from_scratch.log` did not save a comparison.
- **Independent record verification** (`scripts/verify_records.py` → `research/record_verification.json`): recomputed accuracy from `y_true`/`y_pred` equals the stored value for every record; each MLP checkpoint, reloaded from `safetensors`, reproduces the stored test predictions exactly.
- The analysis scripts (`analyze_experiments.py`, `check_decay_mechanism.py`, `exploratory_test_nll.py`, `verify_records.py`) were re-run after their output paths moved from `results/` to `research/`. The regenerated JSON files are byte-identical to the previous ones.
- `scripts/write_experiment_cards.py` regenerated 14 cards from the records; the pilot card was left as written in the pilot phase.

**Known issues and limitations of the fixture (not changed).**

1. **Test features pass through the network.** `fit_mlp` runs the model over all 1797 inputs at the reload check, and `experiment` does the same for the final predictions. Test labels are used only for the accuracy and no training gradient touches test rows. The pilot report disclosed this. The protocol's "validation-only pilot" wording is not strictly true for the code. It does not change any reported number.
2. **`checkpoint_reload_verified` is a constant.** `study.py` writes `True` unconditionally, after an `assert_close` that would raise before the record is saved. The independent check is `research/record_verification.json`.
3. **Degenerate CI.** The bootstrap returns exactly zero because the metric has zero variance on this test split. This is reported as such (§4.2), not as precision.

## 7. Learnings

- **The registered intervention is too small to show up in test accuracy at this budget.** Decay changed the weights (≥99.99% of entries in each regularized checkpoint) and the training trajectory (loss histories differ every epoch), but no test argmax flipped. The directional H1 is not supported, and the falsification rule was not triggered. This matches the expectation in `reports/preregistration.md` §3 and the prediction P2 (inconclusive).
- **The primary metric has no power for effects this small.** A metric with zero discordant pairs carries no sign information. The continuous loss did change, but only by about 0.0005 nats. For a future cycle, a continuous primary metric with a registered bootstrap would be better powered. That would need a new preregistration; it cannot be substituted here.
- **The mechanism prediction should be stated as an observed ratio.** Observed weight shrink was 0.48× the decay-only prediction in both arms. The offset by gradient updates is a hypothesis, and this study does not decompose it.
- **Seed noise is about one example.** The three-seed SD of 0.32 pp is small relative to the 2.5 pp detection floor. Three seeds estimate the SD poorly, so more seeds would be needed to say anything about seed-to-seed variance.
- **Process.** The earlier draft's P5 window was not carried forward; each verdict was re-derived from the preregistration and the records. The reproducibility claim rests on a rerun that was actually performed and logged, not on the earlier log.

## 8. Limitations and open gaps

- One random stratified example split (split seed 42). No writer identity, no other dataset, no distribution shift. Results do not generalize beyond this split, architecture, and budget.
- Three seeds. The per-seed SD is a rough estimate.
- Only AdamW decay versus no decay, plus a local point at λ = 0.001. No decoupled-versus-L2 comparison (Loshchilov & Hutter [S01] compare the two; this cycle does not).
- No early stopping and a fixed 50-epoch budget. Under- or over-training is not controlled.
- The shuffled-label arm has no λ = 0 counterpart, so it cannot isolate decay's effect on memorization.
- The percentile-bootstrap method source [S15] was read at abstract level only (preregistration §12). The primary result does not depend on the bootstrap choice, because the bootstrap distribution is a point mass at zero.
- Sources in §9 were taken from the research-phase ledger (`research/sources.json`) and were not re-fetched in this phase.
- The data licence: `load_digits` is distributed under the UCI terms (CC BY 4.0 per the UCI page [S06]; see the research highlights and preregistration §7).

## 9. Sources

- [S01] Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW), ICLR 2019, https://arxiv.org/abs/1711.05101 (abstract only; numeric results not read).
- [S03] PyTorch `torch.optim.AdamW` documentation, https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html
- [S04] PyTorch `torch/optim/adam.py` decoupled update (installed torch 2.14.1), https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adam.py
- [S05] scikit-learn `load_digits` documentation, https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- [S06] UCI Optical Recognition of Handwritten Digits, https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- [S15] Dalitz, Construction of Confidence Intervals, https://arxiv.org/abs/1807.03582 (abstract only)
- safetensors torch API, https://huggingface.co/docs/safetensors/en/api/torch
- `scipy.stats.binomtest`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
- `numpy.random.Generator.integers`, https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
- Python `tempfile`, `subprocess`, `shutil`, `hashlib` standard library documentation (used by `scripts/rerun_from_scratch.py` and `scripts/write_experiment_cards.py`), https://docs.python.org/3/library/

Full source ledger: `research/sources.json`. Predictions: `reports/preregistration.md` §4. Pilot: `reports/pilot.md`.

## 10. Files

- `results/<id>.json` (15 records): `regularized_s{0,1,2}`, `unregularized_s{0,1,2}`, `sensitivity_s{0,1,2}`, `shuffled_labels_s{0,1,2}`, `logistic_s0`, `class_prior_s0`, and the pilot `pilot_regularized_s0`. Each stores `y_true`, `y_pred`, `example_ids`, `history`, checkpoint hashes, device, and library versions.
- `reports/experiment-<id>.md`: one card per record (generated by `scripts/write_experiment_cards.py`).
- `research/experiment_summary.json`: registered analysis (`scripts/analyze_experiments.py`).
- `research/decay_mechanism.json`: weight-level check (`scripts/check_decay_mechanism.py`).
- `research/exploratory_test_nll.json`: exploratory NLL analysis (`scripts/exploratory_test_nll.py`).
- `research/record_verification.json`: independent record and checkpoint verification (`scripts/verify_records.py`).
- `research/rerun_comparison.json`: scratch retraining comparison (`scripts/rerun_from_scratch.py`).
- `results/experiments_run.log` (original training), `results/experiments_attempt2_reuse.log`, `results/experiments_reuse_check.log`, `results/rerun_from_scratch.log`, `results/rerun_from_scratch_repeat.log`, `results/pilot_run.log`: timestamped logs.
- `research/attempts/`: preserved drafts from the earlier attempt.
- `models/*.safetensors`, `models/logistic.npz`, `models/class_prior.npz`: checkpoints.
