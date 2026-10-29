# Experiments: registered comparisons (cycle 001)

Status: **complete for the registered comparisons**. The predictions below were written before any test-split result existed and were not edited afterward. Results follow the predictions. The exploratory section at the end is post-hoc and is labelled as such.

Files produced by this phase:

- `results/regularized_s{0,1,2}.json`, `unregularized_s{0,1,2}.json`, `sensitivity_s{0,1,2}.json`, `shuffled_labels_s{0,1,2}.json`, `logistic_s0.json`, `class_prior_s0.json` (one record per run; `y_true`, `y_pred`, `history`, `checkpoint_sha256`).
- `results/experiments_run.log` (timestamped stdout of `study.py experiments`, 18:11 UTC).
- `results/experiment_summary.json` (from `scripts/analyze_experiments.py`).
- `results/decay_mechanism.json` (from `scripts/check_decay_mechanism.py`).
- `results/exploratory_test_nll.json` (from `scripts/exploratory_test_nll.py`; not registered).
- `models/*.safetensors` (one checkpoint per neural run), `models/logistic.npz`, `models/class_prior.npz`.

Protocol: `research/protocol.json` (sha256 `5edcfb3f…0ec257`, matches `research/registration.json`). Preregistration: `reports/preregistration.md`. Pilot: `reports/pilot.md` (validation only; published via `research/pilot-publication.json`).

## Scope of this phase

- Command: `study.py experiments` in `cycles/001`, which runs the six methods in `protocol.json` order: `regularized`, `unregularized`, `sensitivity`, `shuffled_labels` (seeds 0, 1, 2 each), then `logistic` and `class_prior` (seed 0 only, as the fixture hard-codes).
- Primary metric (registered): mean over seeds of test accuracy(decay 0.01) − test accuracy(decay 0), with a paired conditional bootstrap (10,000 resamples of test-example indices, RNG seed 2026).
- The `regularized` seed-0 test result reuses the pilot checkpoint `models/regularized-s0.safetensors` and its stored loss history. It is not retrained. The pilot record's `code_sha256` is checked against the current `study.py`.
- Known inexactness, carried over from the pilot: `fit_mlp` and `experiment` run the forward pass over all 1797 inputs, including test features. Test labels are used only in the accuracy computation on the test split. The test predictions are stored, as the protocol requires.

## Predictions (recorded before the run)

Each prediction states the observation that would contradict it.

| ID | Prediction | Contradicted by |
|---|---|---|
| P1 | Primary paired Δ (decay 0.01 − 0) satisfies \|mean Δ\| ≤ 1.0 pp. Basis: the shrink arithmetic in `reports/highlights.md` (≈0.85% cumulative weight shrink over 850 steps). | \|mean Δ\| > 1.0 pp |
| P2 | The paired 95% bootstrap CI for the primary Δ includes 0. | CI wholly above or wholly below 0 |
| P3 | The CI is wholly above 0. Estimated probability ~10–15%. | Not contradicted by a positive mean, only by the CI including 0 |
| P5 | Shuffled-label arm test accuracy (against the true labels) is near chance, 0.10, within 0.08–0.14. | Accuracy > 0.20 would indicate a pipeline bug (label leakage or mismatch) |
| P6 | Logistic regression test accuracy is 0.93–0.98, and above the class-prior baseline. Class-prior accuracy is 0.08–0.12 (majority class is ~10% of the stratified train set). | Logistic < class-prior, or class-prior > 0.15 |
| P7 | Seed SD of test accuracy within each neural arm is 0.3–1.5 pp, so seed variance is of the same order as the predicted effect. | Seed SD > 2.5 pp, which would make the registered three-seed design uninformative |
| P8 | Regularized and unregularized loss histories differ, but only by a small amount (max train-loss difference ≤ 1e-2 nats at epoch 50). Histories must not be identical. | Identical histories (decay not applied); or a difference > 0.05 nats (the decay is larger than the arithmetic predicts) |

Competing alternatives for the primary Δ, and the observation that would distinguish them:

- **No effect (the arithmetic is right).** Δ near 0, CI spans 0, histories differ only at the 1e-4 level. Distinguishing observation: P8 magnitude.
- **Decay hurts accuracy.** Δ < 0 with a CI wholly below 0. Would contradict the directional hypothesis (falsification rule in `protocol.json`).
- **Decay helps through a path other than shrink.** Not separable from "no effect" with this design; the sensitivity arm (λ = 0.001) would show a dose trend if the effect were real. Predicted: Δ(λ=0.001) is between Δ(0.01) and 0, in the same direction as Δ(0.01) but about 10× smaller.
- **Apparent gain from training randomness or an implementation error.** Distinguishing observations: (a) per-seed Δ signs disagree; (b) shuffled-label accuracy above chance (P5); (c) P8 histories identical.

## Results

All numbers come from `results/*.json` through `scripts/analyze_experiments.py`, `scripts/check_decay_mechanism.py`, and `scripts/exploratory_test_nll.py`. The test split has 360 examples, so one prediction is 0.278 pp.

### Run execution

`study.py experiments` completed with exit code 0 and saved all 15 records. No run was rejected or repeated. The `regularized` seed-0 record reused the published pilot checkpoint (`models/regularized-s0.safetensors`, sha256 `a57cb6e2…`) and its stored history, as the fixture is written to do. Its test accuracy is 347/360 (0.96389), and its record stores `regularized_s0` with `device: cuda`. The other neural runs took 2.4–6.6 s each on the RTX 5070 Laptop GPU. The logistic and class-prior baselines ran on CPU.

### Accuracy on the test split (registered arms)

| Arm | Decay | Seed 0 | Seed 1 | Seed 2 | Mean | Seed SD |
|---|---|---|---|---|---|---|
| regularized | 0.01 | 0.96389 (347) | 0.96944 (349) | 0.96944 (349) | 0.96759 | 0.32 pp |
| unregularized | 0 | 0.96389 (347) | 0.96944 (349) | 0.96944 (349) | 0.96759 | 0.32 pp |
| sensitivity | 0.001 | 0.96389 (347) | 0.96944 (349) | 0.96944 (349) | 0.96759 | 0.32 pp |
| shuffled_labels | 0.01 | 0.06111 (22) | 0.11667 (42) | 0.10833 (39) | 0.09537 | 3.00 pp |
| logistic (C = 1.0, seed 0 only) | — | 0.95556 (344) | — | — | — | — |
| class_prior (seed 0 only) | — | 0.10278 (37) | — | — | — | — |

Chance for a balanced 10-class task is 0.10; the binomial standard error at n = 360 is 1.58 pp (sqrt(0.1·0.9/360)).

### Primary comparison (registered)

| Quantity | Value |
|---|---|
| Per-seed Δ = acc(decay 0.01) − acc(decay 0) | 0.00 pp, 0.00 pp, 0.00 pp |
| Mean Δ over seeds (primary metric) | **0.00 pp** |
| Paired conditional 95% bootstrap CI (10,000 resamples of test indices, RNG seed 2026) | **[0.00, 0.00] pp** |
| Discordant test pairs (decay-only correct / decay-0-only correct), seeds 0, 1, 2 | 0/0, 0/0, 0/0 |
| McNemar exact p (per seed) | 1.0 in every seed |
| Identical test predictions (`y_pred`), decay 0.01 vs decay 0 | **identical for all three seeds** |
| Sensitivity arm (decay 0.001) minus decay 0, mean | 0.00 pp |

The falsification rule in `research/protocol.json` is not triggered. The CI is not wholly below zero, so the directional hypothesis is not contradicted. The CI includes zero, so by the registered rule the result is **inconclusive**. It is not evidence of equivalence. The interval is degenerate because no test example changed its predicted class under decay. The bootstrap cannot detect any effect that leaves the argmax unchanged, and this design has no test where it could.

### Decay did change the optimization and the weights

The identical predictions could mean either that decay was not applied or that its effect is too small to change any argmax. The next two checks separate these.

Training histories (`history` field, same seeds and batch order, from `results/experiment_summary.json`):

| Seed | Max \|Δ train loss\| over 50 epochs | Final train loss (decay − no decay) | Final validation loss (decay − no decay) |
|---|---|---|---|
| 0 | 7.4e-4 | +7.1e-4 | +2.8e-4 |
| 1 | 8.4e-4 | +7.0e-4 | +3.5e-4 |
| 2 | 8.0e-4 | +7.7e-4 | +4.2e-4 |

Checkpoint weights (`results/decay_mechanism.json`), relative to the unregularized checkpoint of the same seed:

| Seed | Weight-norm ratio, decay 0.01 | Pure-decay prediction (1 − lr·λ)^850 | Max abs weight difference, decay 0.01 | Max abs weight difference, decay 0.001 |
|---|---|---|---|---|
| 0 | 0.99594 | 0.99154 | 0.068 | 0.027 |
| 1 | 0.99598 | 0.99154 | 0.019 | 0.060 |
| 2 | 0.99594 | 0.99154 | 0.0078 | 0.0046 |

- Decay is applied. The loss trajectories differ, and every weight entry differs (fraction changed 0.9998–1.0).
- The norm shrink is about 0.41%, half the 0.85% a pure multiplicative decay would give. Gradient updates over 850 steps partly offset the shrink, so the norm ratio does not measure the decay alone.
- The maximum per-entry weight difference (0.005–0.07) is much larger than the decay's own per-step effect on a weight of typical size. The sensitivity arm (λ = 0.001, 10× smaller) has a larger max difference than the λ = 0.01 arm for seed 1. This pattern is not explained by the decay magnitude. It is consistent with trajectory divergence under Adam's per-coordinate normalization, but this was not tested here.

### Exploratory (post-hoc, not registered): paired test cross-entropy

The accuracy metric has zero discordant pairs, so it cannot show whether decay changes the predicted distribution. `scripts/exploratory_test_nll.py` computes the same paired procedure for test NLL. **This analysis was not registered, so it cannot become confirmatory in this cycle.** It is reported to inform the next protocol.

| Seed | NLL, decay 0.01 | NLL, decay 0 | Δ NLL (decay − no decay) | Mean top-class prob, decay 0.01 | Mean top-class prob, decay 0 |
|---|---|---|---|---|---|
| 0 | 0.12493 | 0.12453 | +0.00040 | 0.93450 | 0.93513 |
| 1 | 0.12464 | 0.12416 | +0.00048 | 0.93377 | 0.93435 |
| 2 | 0.12836 | 0.12778 | +0.00057 | 0.92964 | 0.93026 |

- Mean Δ NLL = +0.00048 nats. Conditional bootstrap 95% CI [+0.00024, +0.00070] (same resampling procedure as the registered analysis). Sign is positive in 3 of 3 seeds.
- Decay lowers the mean top-class probability in every seed (by about 0.0006). This is consistent with the shrink explanation: smaller logits give less confident outputs, which raises NLL on the correctly classified majority. It is the opposite sign to the directional hypothesis.
- Limits: the bootstrap resamples examples, not seeds, so the CI treats the three seeds as a fixed set and excludes training-population variance. With n = 3 seeds, the sign consistency is weak evidence. The effect is 0.05% of the NLL. This is an exploratory description, not a demonstrated harm of decay.

### Predictions vs. observations

| ID | Prediction (written before the run) | Observed | Status |
|---|---|---|---|
| P1 | \|mean Δ\| ≤ 1.0 pp | 0.00 pp | Held |
| P2 | Paired 95% CI includes 0 | [0.00, 0.00] pp (degenerate, includes 0) | Held |
| P3 | CI wholly above 0 (~10–15%) | Not met (CI is [0, 0]) | Not met |
| P5 | Shuffled-label test accuracy in 0.08–0.14; > 0.20 would signal a bug | Seed 0: 0.061 (outside the window); seeds 1, 2: 0.117, 0.108 (inside). Mean 0.095. No value > 0.20. | Partly held. Seed 0 fell below my window; no bug signal under the falsifier |
| P6 | Logistic 0.93–0.98; class prior 0.08–0.12 | Logistic 0.956; class prior 0.103 | Held |
| P7 | Seed SD of neural arms 0.3–1.5 pp | 0.32 pp | Held, at the lower edge of the window |
| P8 | Histories differ; max train-loss diff ≤ 1e-2 and not > 0.05 | 7–8e-4 nats | Held |

The falsifier for P5 concerned a bug. Shuffled-label seed 0 is below chance (z = −2.46 against the binomial standard error). One plausible explanation is that a fit to permuted labels maps test classes onto a systematically wrong label set for that seed, which can land below chance on one 360-example split. This is a hypothesis and was not tested. It does not indicate a pipeline error under the registered rule, and I did not predict a value below 0.08.

### Competing explanations and what they predicted

| Explanation | Prediction | Result |
|---|---|---|
| No decay effect on predictions (too small to flip argmax) | Δ ≈ 0; histories differ slightly; weights differ | Consistent with all three |
| Decay not applied | Histories identical, weights identical | **Rejected**: histories and weights differ |
| Decay hurts accuracy | CI wholly below 0 | Not observed (CI is [0, 0]) |
| Gain from training randomness | Per-seed Δ signs disagree | No per-seed Δ is nonzero |
| Pipeline bug on labels | Shuffled-label accuracy > 0.20 | Not observed (max 0.117) |
| Decay effect masked by the sensitivity arm | Dose trend: Δ(0.001) smaller than Δ(0.01) and same sign | Not testable; both are 0.00 pp |

### Deviations and disclosures

- **Test features passed through the network during fitting and reloading.** As recorded in `reports/pilot.md`, `fit_mlp` and `experiment` run the forward pass over all 1797 inputs, including the 360 test features. Test labels were used only to compute accuracy on the test split. The stored `y_pred` for the test split is the registered output. This is an inexactness in the frozen `study.py`. It does not change any reported number, but the statement "validation only" in the pilot does not hold for the code, and any publication should say so.
- **The seed-0 regularized test result reuses the pilot checkpoint.** It was not retrained. Its loss history comes from the pilot record. This is what the fixture does by design.
- **Post-hoc analyses.** `scripts/exploratory_test_nll.py` and the top-class-probability check were added after the registered results. They are labelled exploratory above and are not used in the primary conclusion.
- **Not verified in this phase.** The library documentation URLs cited in script comments (NumPy `Generator.integers`, SciPy `binomtest`) were taken from memory and not fetched again in this phase.

## Learnings

1. **The registered metric has zero resolution for this intervention.** Decay changes the loss by about 7e-4 nats and the weights by about 0.4% of their norm, yet changes no test prediction. My research-phase estimate (~2.5 pp detectable) assumed a disagreement rate of about 3%. The observed disagreement rate was 0%. The actual constraint is not the sample size but whether the intervention moves any argmax at all.
2. **Predictions P1, P2, P8 held, and the reason was different from what I wrote.** I predicted a small effect from the shrink arithmetic. The arithmetic was right, but the effect is invisible to accuracy for a reason I had not written down: the decay leaves every argmax unchanged, not just the average.
3. **The loss-level signal has a sign opposite to the directional hypothesis.** The post-hoc NLL result suggests that decay makes the model less confident, not more accurate. This is consistent with the shrink mechanism. It is exploratory and cannot be promoted to a confirmatory finding in this cycle. The next cycle must register test NLL (or another continuous metric) as the primary metric if it is to be tested.
4. **Shuffled-label control is noisy at this size.** Seeds 0, 1, and 2 score 0.061, 0.117, and 0.108 (seed SD 3.0 pp; seed 0 to seed 1 spans 5.6 pp). The control shows that the pipeline does not fit random labels above chance on this test split, which is the property it was meant to check. It does not measure memorization by decay, because there is no shuffled arm with decay 0.
5. **Pairing does what it should, and it is also why the CI is degenerate.** Shared initialization and batch order make paired differences small and informative for the loss, but the accuracy metric is too coarse to pick up that signal.

## Limitations

- Three seeds and one stratified split of 1797 examples. The seed SD (0.32 pp for the neural arms) is estimated from three values and is itself uncertain. No population-level claim is made.
- The primary CI is conditional on the trained models and excludes training variance. The protocol states this.
- The sensitivity arm cannot show a dose-response, because the effect is zero at both decay values on argmax.
- The test-feature forward pass in the frozen `study.py` is disclosed above.
- No comparison with other optimizers, architectures, or datasets.
- Exploratory NLL and confidence results are post-hoc and unreplicated.

## Next steps (not run in this phase)

- Analysis phase: write up the registered result as inconclusive, with the exact-zero discordance, and report the exploratory NLL result as exploratory.
- A future cycle needs a new preregistration before any new test-set evaluation. It should register a continuous primary metric (test NLL) and more seeds, and fix the test-feature forward pass in `study.py` (a new code hash would be required).
