# Pilot: regularized seed-0 model (validation split only)

Cycle 001. Phase: pilot. Protocol: `research/protocol.json` (sha256 `5edcfb3fcda62035…50ec257`, matches `research/registration.json`). Code: `study.py` (sha256 `34415a875e7f6195…a6d67d39`).

Scope: one model, the registered `regularized` arm (AdamW, weight_decay = 0.01), seed 0. Evaluated on the **validation** split only. No test labels or test predictions were stored, computed, or used (see the data-access note below). No comparative arm (`unregularized`, `sensitivity`, `shuffled_labels`, `logistic`, `class_prior`) was run. The pilot's purpose is to confirm that the pipeline trains, reloads, and records a reproducible model before any comparison.

## Command and environment

- Command: `study.py pilot` in `cycles/001`. Log: `results/pilot_run.log` (the original training run, 18:01 UTC).
- In this phase I ran `study.py pilot` again. The code found the verified record and reused it, so no new training occurred in that call (`Reusing verified experiment pilot_regularized_s0`).
- Independent reproduction: I copied `study.py` and `research/` into a fresh scratch directory outside the cycle and ran `study.py pilot` there. It retrained from scratch (log in the scratch directory, not preserved as a project artifact). Result: identical accuracy, identical `y_pred`, identical validation-loss history (max absolute difference 0.0), and identical checkpoint sha256. Details in "Reproduction" below.
- Python: `.venv` (torch 2.14.1+cu130, numpy 2.5.4, scikit-learn 1.9.1).
- Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (`device: cuda`). `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are set in `study.py`.
- Wall-clock for training and evaluation: 6.8 s (original run, from the record's `seconds` field), 8.5 s (scratch rerun, from its log).

## Exact configuration (from `results/pilot_regularized_s0.json` and `study.py`)

| Item | Value |
|---|---|
| Data | `sklearn.datasets.load_digits`, 1797 × 64, pixels divided by 16 (`dataset_sha256` `8b3b6ae9…98772`, a hash over all inputs and labels) |
| Split | Stratified 60/20/20, `split_seed` 42 → train 1078 / validation 359 / test 360 |
| Model | MLP 64 → 128 → 10, ReLU, `nn.Sequential`, 9,610 parameters (64·128+128 + 128·10+10) |
| Optimizer | `torch.optim.AdamW`, lr 1e-3, weight_decay 0.01, default betas and eps |
| Batching | batch size 64, shuffled each epoch by a generator seeded with 0 → 17 steps/epoch (1078/64 rounded up) |
| Epochs | 50 (850 optimizer steps) |
| Loss | cross-entropy, mean over batch |
| Checkpoint | `models/regularized-s0.safetensors`, sha256 `a57cb6e24f5e7d2a3e83da8efd97233a4b7e0e5b8492f89b3cbe6064cb86cc3d`, 38,728 bytes. Reloaded and output-matched exactly (`torch.testing.assert_close`, atol = rtol = 0). |

## Results

Validation accuracy: **0.96657** (347 of 359 correct). Recomputing accuracy from the stored `y_true` and `y_pred` gives the same 347/359.

Training history (50 epochs, from the `history` field of the record):

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 2.2119 | 2.0877 |
| 5 | 1.0456 | 0.9138 |
| 37 | 0.0857 | 0.1344 |
| 50 | 0.0568 | 0.1202 |

- Validation loss was lowest at the final epoch (0.12023). It did not turn upward, so this run shows no overfitting onset within 50 epochs. The pilot does not select an epoch; the protocol uses the final model.
- Validation loss is not monotone (for example, epoch 45 is 0.12282 and epoch 46 is 0.12334). This is minibatch noise at lr 1e-3, not a sign of a broken schedule.
- Predicted-class counts on validation are 33–41 per class. True class counts are 35–36 per class, as expected from the stratified split.
- Per-class validation accuracy, digits 0–9: 0.972, 0.972, 0.917, 1.000, 1.000, 0.944, 1.000, 1.000, 0.886, 0.972. The weakest classes are 8 (0.886) and 2 (0.917). I did not analyse the cause of these errors; the attribution to shared stroke structure is a hypothesis.

## Predictions recorded before the run and their status

These come from `reports/highlights.md` and `reports/preregistration.md`. Only the predictions that a single validation run can speak to are checked here.

| ID | Prediction | Status after pilot |
|---|---|---|
| P4 | Reloaded checkpoint reproduces the reference outputs exactly | **Observed.** `checkpoint_reload_verified: true`; the in-code `assert_close` passed in both runs. |
| — | Training runs and reduces loss on the registered schedule | **Observed.** Train loss 2.21 → 0.057; validation loss 2.09 → 0.120. |
| — | Validation accuracy is well above chance (0.10) | **Observed.** 0.967 on validation. |
| P8 | Regularized and unregularized loss histories differ | **Not testable yet.** Requires the `unregularized` arm, which is out of scope for this phase. |
| P1–P3 | Sign and size of the paired decay effect | **Not testable yet.** Requires paired comparisons on test data in the experiments phase. |
| P5 | Shuffled-label control near chance | **Not testable yet.** |

Pre-run arithmetic: the research phase estimated that decay shrinks weights by about 0.85% over 850 steps, from (1 − lr·λ)^850 = (1 − 10⁻⁵)^850 ≈ 0.9915 (`reports/highlights.md`). The pilot does not measure this shrink, because no undecayed model exists in the pilot. It remains an unverified expectation until the comparison.

## Data-access note (test split)

The evaluation outputs and the stored record cover only the validation split. However, `fit_mlp` in `study.py` runs the trained network over all 1797 inputs to verify the reload (`reference = model(x)` and `reloaded = restored(x)`), and `experiment` also computes `predicted` over all inputs. So the test-split feature vectors passed through the network. Test labels were not used in any computation, and the test predictions were neither stored nor evaluated. The record's `dataset_sha256` hashes all inputs and labels, including test labels, but as a hash it reveals no label information. This is a design-level inexactness in `study.py`, and I record it here rather than leave it implicit. It does not affect the validation numbers. It should be stated in any publication of this run.

## Reproduction

The original record and the scratch rerun were compared with a script outside the project (`C:/Users/Happy/AppData/Local/Temp/eureka_verify/verify_pilot.py`):

| Quantity | Original | Scratch rerun |
|---|---|---|
| Validation accuracy | 0.9665738161559888 | 0.9665738161559888 |
| `y_pred` (validation) | — | identical |
| Max absolute validation-loss difference over 50 epochs | — | 0.0 |
| Checkpoint sha256 (prefix) | `a57cb6e24f5e…` | `a57cb6e24f5e…` |

This shows that the pilot is reproducible bit-for-bit on this machine and software stack (CUDA, torch 2.14.1+cu130). It does not show reproducibility on other hardware, which is not tested.

## Alternative explanations considered

- **A metric or indexing bug that makes the model look better than it is.** Recomputing accuracy from `y_true` and `y_pred` gives 347/359. `example_ids` equals the validation index list exactly. `y_true` and `y_pred` have length 359.
- **A data leakage path.** The split is taken before fitting. Normalization is a constant division by 16, so it has no fitted parameters. `study.py check` confirms disjoint splits covering all 1797 samples. The test-input forward pass described above is a separate, non-label path that does not feed any training or selection step.
- **Stale or mismatched artifacts.** The protocol hash, code hash, and checkpoint hash are recorded and were recomputed. They match `research/registration.json`, the record, and the checkpoint file.
- **Non-determinism.** Measured only for this single seed, by the scratch rerun above. Seed sensitivity is not measured; this is a gap for the experiments phase, where seeds 0, 1, and 2 are run.
- **Dependency on the pilot checkpoint.** In `study.py experiment`, the `regularized`, seed 0 arm in the experiments phase reuses this pilot checkpoint and its stored history rather than retraining. Any later test result for that arm therefore depends on the same artifact that is published here.

## Learnings

1. The fixture trains, saves, and reloads correctly on CUDA in about 7–8 seconds per run. Compute is not a constraint for the registered comparisons.
2. The pilot's validation accuracy (0.967) is one number from one seed on 359 examples. Its binomial standard error, sqrt(p(1−p)/n) with p = 0.9666 and n = 359, is about 0.95 pp. A 1 pp difference between two arms is therefore indistinguishable from noise at this sample size with one seed. This agrees with the research phase's estimate that the protocol resolves only effects of roughly 2.5 pp or larger. That estimate was a planning figure, not a measured power analysis.
3. Validation loss does not rise over the 50 epochs, so the registered fixed-budget schedule does not need early stopping for this model. Whether decay changes final test accuracy is still untested.
4. Reporting only the final epoch, as the protocol does, avoids selecting an epoch on validation. The within-run loss wiggles near the end do not justify any selection.
5. Recording the test-input forward pass (see the data-access note) was a useful catch. It does not leak labels, but the pipeline should not feed test features through the model in a phase that is supposed to be validation-only. The fix belongs in `study.py`, which is frozen for this cycle. I did not change it. I am recording the issue for the next cycle's protocol.

## Issues and fixes

- No blocking failures in the pilot. The original run's assertions passed.
- Earlier handoff attempt: the controller reported four artifacts as missing. The artifacts exist on disk; this phase lists them with project-relative paths (`cycles/001/...`) to match the controller's lookup.
- Test suite: `python -m pytest -q tests` gives `13 passed` (18.9 s) on the current code.
- I did not change `study.py`, `research/protocol.json`, or `research/registration.json`. No data or files were deleted.

## Sources

Sources are taken from `research/sources.json`. Abstract-level access only for the first two, as recorded there.

- [S01] Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW), ICLR 2019, https://arxiv.org/abs/1711.05101. Motivation for decoupled decay. Abstract read.
- [S03] PyTorch `torch.optim.AdamW` docs, https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html. Defaults `lr=1e-3`, `weight_decay=0.01`.
- [S04] PyTorch `torch/optim/adam.py` (installed torch 2.14.1), decoupled update `param.mul_(1 - lr * weight_decay)`. The shrink arithmetic above uses this line.
- [S05] scikit-learn `load_digits` docs, https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html. 1797 × 64, integer pixels 0–16.
- [S11] Lipton & Steinhardt, Troubling Trends in Machine Learning Scholarship, https://arxiv.org/abs/1807.03341. Failure modes: unclear source of gains, conflating explanation with speculation. Cited for the attribution rule. Abstract-level summary only.
- [S12] Lones, How to avoid machine learning pitfalls, https://arxiv.org/abs/2108.02497. Used for its five-stage structure as a checklist. Its leakage recommendations were not checked in full text, so I do not cite it for leakage.
- [S16] safetensors torch API, https://huggingface.co/docs/safetensors/en/api/torch. `save_file` requires contiguous tensors; `load_file` used for the reload check.

## Pointers (project-relative, cycle root `cycles/001`)

- Record: `results/pilot_regularized_s0.json`
- Checkpoint: `models/regularized-s0.safetensors`
- Log: `results/pilot_run.log`
- Pre-registration and predictions: `reports/preregistration.md`, `reports/highlights.md`

## What this pilot does not show

- It does not estimate the effect of weight decay. There is no paired comparison.
- It does not compute test accuracy for this model, and it stores no test labels or test predictions.
- It does not establish seed sensitivity, hyperparameter sensitivity, reproducibility on other hardware, or generalization beyond this random split.

## Next step (not run in this phase)

Publish this pilot and checkpoint (controller). The experiments phase runs the registered comparisons with `study.py experiments`, which is gated on `research/pilot-publication.json`. Test data are used only there.
