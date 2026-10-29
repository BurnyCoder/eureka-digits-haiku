# Interpretation: AdamW weight decay on a fixed digits MLP (cycle 001)

This report interprets the completed registered study. It contains no typed empirical scores. Each result is cited by claim ID (`C#` confirmatory or `E#` exploratory, defined in `research/claims.json`), experiment record ID (the `id` of `results/<id>.json`, for example `regularized_s0`), local artifact ID (`L##`, `research/evidence.json`), or source ID (`S##`, `research/sources.json`). Numbers are in the artifacts those IDs point to.

## 1. Motivation
Decoupled weight decay (AdamW) is a standard regulariser for MLPs. The question is whether a fixed ReLU MLP on the digits fixture gains held-out accuracy from decay 0.01, compared with decay 0 and a 0.001 sensitivity arm. The hypothesis, primary metric and falsification rule were registered before results in `reports/preregistration.md` and `research/protocol.json` (`DOC`-level document, `L10`). This is a small illustrative study, not evidence of broad reliability.

## 2. Context and related work
- The decoupled update applies decay to the weights directly, not through the gradient. The implementation is in torch's `adam.py` (`S04` in `research/sources.json`).
- The paired bootstrap for the primary interval follows the percentile method in `S17` and `S18`. `S15` was read only at abstract level, which is a limitation.
- Lipton & Steinhardt (Troubling Trends) and Lones (arXiv:2108.02497) motivated the pre-registration, controls and the confirmatory/exploratory split. Full list with URLs: `research/sources.json`.

## 3. Methods
- Fixture: train 1078 / validation 359 / test 360 (`study.py check`). The test split is used only for the registered comparisons.
- Arms: regularized (decay 0.01: `regularized_s0..2`), unregularized (decay 0: `unregularized_s0..2`), sensitivity (decay 0.001: `sensitivity_s0..2`), shuffled-label control (`shuffled_labels_s0..2`), logistic baseline (`logistic_s0`), class-prior floor (`class_prior_s0`). Pilot: `pilot_regularized_s0`.
- Primary metric: test accuracy. Paired comparison of regularized vs unregularized predictions per test example, with a paired bootstrap over test examples. The bootstrap does not capture training-seed variance; seed spread is reported separately.
- Derived aggregate: `L01` (exploratory).

## 4. Confirmatory results
- **C1.** The registered primary comparison is null. Across seeds the two arms make identical test argmax predictions, so the paired difference and its interval are degenerate at zero (`regularized_s0..2`, `unregularized_s0..2`). The seed spread of accuracy is the same for both arms (`L01`).
- **C2.** The sensitivity arm (decay 0.001) is also identical to the unregularized arm (`sensitivity_s0..2`, `unregularized_s0..2`).
- **C3.** The shuffled-label control is at near-chance accuracy on every seed (`shuffled_labels_s0..2`). This is a leakage control: the pipeline does not extract signal from relabelled data.
- **C5.** The logistic baseline (`logistic_s0`) is close to, but below, the MLP arms. The class-prior floor (`class_prior_s0`) is near chance. The baseline has one seed.

C1, C2, C3 and C5 are confirmatory in the registered sense. Their conclusions hold only for the registered decay values, learning rate and 850-step budget.

## 5. Exploratory findings
- **E1.** Loss histories differ between decay and no decay at every seed, even though predictions do not (`L01`, `regularized_s0`, `unregularized_s0`). The model changes, but not its test argmax.
- **E2.** The weight-level analysis (`L02`) shows decay shrinking weights in the direction the decoupled update predicts, by a small amount over the run. This is the mechanistic reason C1 is near-null under the registered constants. It was derived after the result was known, and rests on the source code in `L08` and the protocol in `L10`.
- **E3.** The design resolves only accuracy differences above a minimum detectable effect set by n=360 and the seed spread. The figure is an approximation from the research phase (`pilot_regularized_s0`, `L01`, `L10`), not a registered power analysis.
- **E4.** A post-hoc test negative log-likelihood analysis (`L03`) shows the decay arm slightly worse on this continuous metric, with an interval excluding zero. This is the most informative signal in the study, but it is not a registered endpoint and cannot be promoted to confirmatory.
- **C4.** Reproducibility audits (`L04`, `L05`, `L06`) show the stored results are internally consistent and an independent from-scratch retrain matches the arm records.
- **E5.** Implementation checks pass (`L09`, `L07`, `L08`).
- **E6.** Code audit (`L08`): test features reach eval-mode forward passes in the reload and prediction steps. No test label was found in the training or selection paths. This is a procedural disclosure, not a leak into the reported metric.

## 6. Negative and inconclusive findings
- The primary hypothesis (decay improves held-out accuracy) is not supported. The result is a null with a degenerate interval. That is not evidence of equivalence (C1).
- The sensitivity arm (C2) is also null. Neither non-zero decay value changes accuracy here.
- The continuous endpoint suggests a small harm from decay (E4), the opposite sign to the hypothesis. This is exploratory and should be re-tested, not reported as a finding.
- The predictions that the interval would span zero and that the effect would be too small to see held (E3), but only as a degenerate interval.

## 7. Implications
- For this fixture and budget, decay has no detectable effect on the argmax, so the registered comparison cannot recommend decay.
- A future cycle needs a pre-registered continuous primary endpoint, more seeds or a larger test set, and a budget long enough for decay to matter. Accuracy alone cannot say anything about direction here.

## 8. Limitations
- Three seeds, one learning rate, one 850-step budget, one fixture.
- Zero discordant pairs gives no sign information for accuracy (C1).
- The bootstrap resamples test examples only (see Methods).
- The logistic baseline has one seed. The minimum-detectable-effect figure (E3) is approximate.
- Test features pass through eval-mode forward passes (E6). This is not a label leak, but the split is not strictly "validation-only" for those steps.
- E1, E2, E4, C4 and E5 are post-hoc and exploratory.

## 9. Related work
Decoupled weight decay: torch `adam.py` implementation (`S04`). Bootstrap: `S17`, `S18` (`S15` read at abstract level). Methodology guidance: Lipton & Steinhardt and Lones (arXiv:2108.02497), as listed in `research/sources.json`.
