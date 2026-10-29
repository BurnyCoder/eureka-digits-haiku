# Preregistration: AdamW weight decay on a fixed digits MLP (cycle 001)

Written 2026-10-29, before any pilot or comparison run. No model has been trained and no test-split output exists. The machine-readable protocol is `research/protocol.json`, which is the supplied, immutable protocol; this document restates it and adds the predictions recorded in the research phase (`reports/research.md` §3, `research/decisions.jsonl`). If the two disagree, `research/protocol.json` governs and this file is corrected, not the protocol.

Analysis kind: **confirmatory** for H1 and the primary metric; every other comparison is **exploratory** and labeled as such (`study.py` writes `analysis_kind` per record).

## 1. Question and objective

- **Question:** Does AdamW weight decay improve held-out test accuracy of a fixed 64-128-10 ReLU MLP on the bundled `sklearn` digits data?
- **Objective:** Exercise the full research harness on a bounded, reproducible study. Negative and inconclusive outcomes are valid results.
- **Scope:** One random stratified example split (split seed 42), no writer identity, no other datasets. Results do not generalize beyond this split, this architecture, or this training budget.

## 2. Hypotheses and alternatives

**H1 (primary, directional):** The mean paired test-accuracy difference for `weight_decay=0.01` minus `weight_decay=0`, averaged over seeds {0, 1, 2}, is positive.

Alternative explanations, each with a distinguishing check:

| Alternative | Distinguishing prediction / check |
|---|---|
| A1. No effect (decay has no effect at this budget) | Paired CI spans 0; per-seed effects scatter around 0 with no consistent sign. |
| A2. Regularization reduces accuracy | Paired CI wholly below 0 (this falsifies the directional H1). |
| A3. An apparent gain comes from training randomness (seed, init, data order) | Per-seed effects change sign or their SD is comparable to the mean. Arms share seed, initial weights, and batch order (`order_rng`), so pairing removes these as a systematic source, but three seeds estimate seed variance poorly. |
| A4. An apparent gain is an implementation error (decay not applied, wrong index, reload mismatch) | (a) Regularized and unregularized per-epoch loss histories must differ (P8); (b) `torch.testing.assert_close(..., rtol=0, atol=0)` reload check passes; (c) shuffled-label test accuracy near chance (P5); (d) recomputing accuracy from the saved `y_true`/`y_pred` reproduces the stored `accuracy`. |
| A5. The "effect" is a test-split artifact | Bootstrap over test indices is conditional on the models; a CI that spans 0 cannot be read as a positive effect. Validation-split accuracy is recorded for the pilot only and is not used for selection. |

## 3. Mechanistic expectation (recorded before results)

AdamW's decoupled update multiplies weights by `(1 − lr·λ)` per step (`torch/optim/adam.py:417` in torch 2.14.1, the `decoupled_weight_decay` path used by `AdamW`, [S04]). With `lr = 1e-3`, `λ = 0.01`, and 850 optimizer steps (17 batches × 50 epochs; 1078 training examples at batch size 64), the cumulative shrink is about `(1 − 1e-5)^850 ≈ 0.9915`, i.e. ≈0.85%, ignoring gradient updates. The sensitivity arm (`λ = 0.001`) shrinks by ≈0.085%. This is arithmetic from the registered constants, not a measurement; `study.py` does not log weight norms. The expected effect on accuracy is therefore small, and the study is more likely to return an inconclusive result than a clear effect.

## 4. Predictions

Probabilities are my subjective estimates, recorded before results (`research/decisions.jsonl`, research phase). They are not calibrated frequencies.

| ID | Prediction | Est. prob. | Falsified / flagged if |
|---|---|---|---|
| P1 | \|mean paired Δ (decay 0.01 − 0)\| ≤ 1.0 pp | ~70% | \|mean Δ\| > 1.0 pp |
| P2 | Paired 95% bootstrap CI for Δ includes 0 (inconclusive) | ~75% | — |
| P3 | CI lies wholly above 0 (supports H1) | ~10–15% | — (supports H1 only if it occurs) |
| P4 | Pilot validation accuracy (regularized, seed 0) and comparison test accuracy for the real-label MLP both fall in 0.93–0.99; logistic test accuracy 0.94–0.98 | ~80% | Real-label accuracy < 0.85 indicates a training bug |
| P5 | Shuffled-label arm test accuracy is near chance, ≈0.10 (range 0.05–0.16) | ~90% | > 0.20 indicates leakage or a bug (labels are permuted on the training split only) |
| P6 | Class-prior baseline test accuracy ≈ majority-class fraction of the test split (≈0.10) | ~90% | Large deviation indicates an indexing bug |
| P7 | Seed-to-seed SD of test accuracy is 0.3–1.5 pp, comparable to or larger than any decay effect | ~60% | — |
| P8 | Regularized and unregularized arms produce **different** per-epoch loss histories for the same seed | ~99% | Identical histories means decay is not applied (implementation error) |

Checks covered by each stage: the **pilot** (validation only) can verify P4 (validation range), and the reload-equality assertion. It cannot verify P8, which needs the unregularized arm, nor P5 or P6, which belong to the comparison stage.

## 5. Primary metric

For each seed s ∈ {0, 1, 2}:

`Δ_s = acc_test(weight_decay=0.01, seed=s) − acc_test(weight_decay=0, seed=s)`

Primary metric: `Δ̄ = (Δ_0 + Δ_1 + Δ_2) / 3`, in percentage points. Accuracy is `mean(y_pred == y_true)` on the 360 test examples as computed by `study.py` (`record['accuracy']`).

Accuracy differences are quantized in steps of 1/360 ≈ 0.278 pp per flipped example.

## 6. Falsification and decision rules (fixed before results)

- **H1 contradicted:** the paired 95% CI for Δ̄ is wholly below 0.
- **H1 supported (in this design only):** the paired 95% CI for Δ̄ is wholly above 0 **and** the implementation checks (A4 a–d) pass.
- **Inconclusive:** the CI spans 0. This is reported as inconclusive. "No effect" and equivalence are **not** claimed, since an interval spanning 0 is not evidence of equivalence.
- **Implementation error:** any of A4 (a)–(d) fails, or P5 (shuffled test accuracy > 0.20), or a reload mismatch. In that case the affected results are reported as invalid, the failure is recorded, and the fix requires a new cycle; old result files are preserved, not overwritten.
- **Harness integrity:** `study.py` refuses to reuse a result whose `protocol_sha256` or `code_sha256` no longer matches the current files. Any code change after results exist invalidates the reuse path and must be handled by a new preregistration.

## 7. Methods

**Data.** `sklearn.datasets.load_digits` [S05]: 1797 samples, 8×8 grayscale digits, 64 features in {0,…,16}, 10 classes. Features are divided by the constant 16 (no estimated normalization parameters). The data originate from UCI Optical Recognition of Handwritten Digits [S06]; the sklearn BSD license covers the code, while the data is distributed under the UCI terms (CC BY 4.0 per the UCI dataset page; to be verified against the UCI page before publication).

**Split.** Stratified `train_test_split` [S05 and sklearn docs], test fraction 0.4 with `random_state=42`, then the held-out portion is split 50/50 into validation and test with the same seed and stratification. Result: train 1078 / validation 359 / test 360. The study asserts disjointness and full coverage.

**Architecture.** `nn.Sequential(Linear(64,128), ReLU, Linear(128,10))`, cross-entropy loss. Identical for every neural arm.

**Optimizer and training.** `torch.optim.AdamW` [S03] with `lr = 1e-3`, default betas and eps, `batch_size = 64`, `epochs = 50` (no early stopping; the final epoch is evaluated), reshuffled each epoch by `torch.Generator` seeded with the run seed. Deterministic algorithms enabled (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Device: CUDA if available, else CPU; the device is recorded per run.

**Arms.**

| Arm | Model | weight_decay | Seeds | Split evaluated | Analysis kind |
|---|---|---|---|---|---|
| pilot | MLP | 0.01 | 0 | validation | exploratory |
| regularized | MLP | 0.01 | 0, 1, 2 | test | confirmatory |
| unregularized | MLP | 0.0 | 0, 1, 2 | test | confirmatory |
| sensitivity | MLP | 0.001 | 0, 1, 2 | test | exploratory (local sensitivity, not selected) |
| shuffled_labels | MLP | 0.01 | 0, 1, 2 | test | exploratory (labels permuted on training split with `default_rng(seed)`) |
| logistic | `LogisticRegression(C=1.0, max_iter=1000, random_state=seed)` | — | 0 | test | exploratory, untuned reference |
| class_prior | majority training class | — | 0 | test | exploratory, trivial floor |

The regularized seed-0 test run reuses the checkpoint and history from the validation pilot (same code, same weights); no re-training occurs for it. The pilot must be published (`research/pilot-publication.json`) before any comparison runs; `study.py` enforces this.

**Why these arms.** Regularized vs unregularized isolates the registered intervention. The shuffled-label arm tests whether the pipeline can fit noise (a capacity check) and cannot isolate decay, because it has no λ = 0 counterpart (stated limitation). Logistic regression and class prior are references: logistic is a different regularizer (L2 in sklearn's solver) and is not tuned; class prior is a trivial floor (the digit classes are near-balanced, so it sits near chance, ≈0.10). The sensitivity arm is a descriptive check at a smaller λ, not a selected sweep.

**Hyperparameter rationale.** The architecture and budget are small and prespecified to give fast feedback on an established task. The three decay values are a local sensitivity check, not a search. Three seeds expose limited initialization sensitivity and do not support population claims.

## 8. Seeds

Seeds {0, 1, 2} drive `random`, `numpy`, and `torch` seeding, the initial weights, the batch order, and the shuffled-label permutation. Within each seed, the regularized and unregularized arms share initial weights and batch order; only `weight_decay` differs. Split seed 42 is fixed and identical across all arms.

## 9. Data boundaries

- Training uses only the train split. Validation is used for the pilot's reported loss and accuracy only; no model, epoch, or hyperparameter is selected using validation or test results.
- Test labels and predictions are produced only in the comparison stage, after the pilot is published.
- No learned preprocessing (the /16 scaling is a constant), no feature selection, no early stopping.
- Generalization is limited to this random example split. The study does not test new writers, new datasets, or distribution shift.

## 10. Analysis plan

1. **Paired bootstrap CI for the primary metric.** Draw 10,000 bootstrap resamples of the 360 test indices with replacement using `numpy.random.default_rng(2026)`. Each resample uses the **same** index draw for all arms and seeds, which preserves pairing. For each resample, compute `Δ̄*` as the mean over seeds of the per-seed accuracy difference on the resampled indices. The 95% CI is the 2.5th and 97.5th percentiles of `Δ̄*` (percentile bootstrap; method to be verified against the source before this phase computes it, see §12).
2. **Per-seed effects.** Report `Δ_0, Δ_1, Δ_2` and their sample SD (ddof = 1) beside the CI. The CI conditions on the three trained models per arm and does not estimate training-population uncertainty; the per-seed SD is the only direct view of seed variance.
3. **Implementation checks.** Verify A4 (a)–(d) and P5 before interpreting Δ̄.
4. **Predictions.** Report each P1–P8 as confirmed, falsified, or not checkable at this stage, with the observed value.
5. **Exploratory comparisons.** Sensitivity (λ = 0.001) minus unregularized, shuffled-label accuracy vs regularized, regularized vs logistic and class prior. These are descriptive: no multiplicity correction is claimed, no comparison is promoted to primary, and each is labeled exploratory.
6. **Reporting.** Include both positive and negative results, the full per-seed table, and the conditional nature of the CI. Do not describe a CI spanning 0 as evidence of no effect.

## 11. Budget

60 active minutes and 20 CLI invocations for the whole validation run, with no automatic extension. Expected cost, for planning only: 1 pilot invocation, plus 1 comparison invocation that runs 11 MLP fits (2 regularized + 3 unregularized + 3 sensitivity + 3 shuffled, each 850 optimizer steps) and 2 baseline fits (logistic, class prior). Each MLP fit is small, so the budget is dominated by setup and verification, not compute.

## 12. Known gaps and limitations (stated before results)

- Percentile-bootstrap method source [S15] was read at abstract level only. It must be verified before the analysis phase computes the CI.
- The safetensors license was not verified; the API was verified [S16].
- The Loshchilov & Hutter numeric results [S01] were not read; the claim cited is from the abstract only.
- The shuffled-label control has no λ = 0 counterpart, so it cannot isolate decay's effect on memorization.
- The study compares AdamW(λ = 0.01) vs AdamW(λ = 0). It does not test decoupled vs L2 (Adam with L2), which is the comparison in [S01].
- No early stopping and a fixed 50-epoch budget: over- or under-training is not controlled.
- Power is coarse. Under an **assumed** ~3% disagreement between arms (not measured), the minimum detectable paired difference for one comparison is ≈2.5 pp. Effects smaller than this will usually be inconclusive.
- Device: runs may use CUDA; GPU and CPU results can differ. The device is recorded per run, and mixed devices across arms must be reported.

## 13. Sources

- [S01] Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW), ICLR 2019, https://arxiv.org/abs/1711.05101
- [S02] Kingma & Ba, Adam, ICLR 2015, https://arxiv.org/abs/1412.6980
- [S03] PyTorch `torch.optim.AdamW` documentation, https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html
- [S04] PyTorch `torch/optim/adam.py` decoupled update (installed torch 2.14.1; upstream: https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adam.py)
- [S05] scikit-learn `load_digits` documentation, https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- [S06] UCI Optical Recognition of Handwritten Digits, https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- [S07] Zhang et al., Understanding deep learning requires rethinking generalization, ICLR 2017, https://arxiv.org/abs/1611.03530
- [S08] Krogh & Hertz, A Simple Weight Decay Can Improve Generalization, NIPS 1991, https://papers.nips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html
- [S09] Bjorck et al., Understanding Decoupled and Early Weight Decay, 2020, https://consensus.app/papers/details/cfe1ea949a105c5f953b251b2af3701e/
- [S10] Zhou et al., Towards Understanding Convergence and Generalization of AdamW, https://consensus.app/papers/details/c179f549f272597888e9890895eef7f1/
- [S11] Lipton & Steinhardt, Troubling Trends in Machine Learning Scholarship, https://arxiv.org/abs/1807.03341
- [S12] Lones, How to avoid machine learning pitfalls, https://arxiv.org/abs/2108.02497
- [S13] Neel Nanda, How I Think About My Research Process (sequence), https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL
- [S15] Dalitz, Construction of Confidence Intervals, https://arxiv.org/abs/1807.03582
- [S16] safetensors torch API, https://huggingface.co/docs/safetensors/en/api/torch

Full source ledger: `research/sources.json`. Literature review and the predictions' rationale: `reports/research.md`.
