# Final report: AdamW weight decay on a fixed digits MLP (cycle 001)

**Historical model-authored report.** The later [parent review](../../../reports/extended-parent-review.md) corrects its hard detection-limit claim, causal interpretation of norm ratios, and description of NLL as the only continuous signal. The corrected interpretation and generated [summary](summary.md) are authoritative for current conclusions and numerical tables. This original report is retained to expose what the model submitted, including its errors.

Author: Libor "Burny" Burian. Cycle status: research, preregister, core, pilot, experiments, analysis, and review phases are accepted. This report is the publish-phase deliverable. It reconciles the README, reports, docs, claims, and requirement checklist with the recorded results. It adds no new experiments and no new test-split evaluation.

## 1. Result in one paragraph

The registered primary hypothesis is **not supported, and the study is inconclusive**. Decoupled weight decay (AdamW, decay 0.01) produced exactly the same test predictions as decay 0 on all 360 test examples for all three seeds. The paired accuracy difference is 0.00 pp, and its bootstrap interval is degenerate at [0, 0] (C1). Decay does change training: per-epoch loss histories differ at every seed (C2, P8), and the weights shrink, though by about half the amount the update rule predicts (E5). A post hoc continuous endpoint (test NLL) is slightly higher with decay on all three seeds (E7). That result is exploratory and cannot be promoted to a finding. This study does not show that decay is useless. It shows that, at this learning rate, step budget, and fixture, the intervention does not change any test argmax, and the design could not resolve effects smaller than about 2.5 pp.

## 2. Question, hypotheses and what was registered

- **Question.** Does AdamW weight decay improve held-out test accuracy of a fixed ReLU MLP `[64, 128, 10]` on the digits fixture?
- **Primary hypothesis H1.** Decay 0.01 improves test accuracy over decay 0.
- **Falsification rule.** The paired bootstrap interval lies wholly below zero. Support requires the interval to lie wholly above zero.
- **Protocol.** Fixed by `cycles/001/research/protocol.json` (`L10`). The preregistration is `cycles/001/reports/preregistration.md`, committed before any result existed. Predictions P1–P8 and their falsifiers are listed there.
- **Constants.** AdamW, learning rate 0.001, batch 64, 50 epochs (850 steps), seeds 0, 1, 2, split seed 42, pixels divided by 16, 10000 paired bootstrap resamples of test indices (RNG seed 2026).

## 3. Data and fixture

- Source: scikit-learn `load_digits`, 1797 samples, 64 features, 10 classes (`S05`). The data is a copy of the UCI optdigits set, licensed CC BY 4.0 (`S06`).
- Stratified split: train 1078 / validation 359 / test 360. Checked by `study.py check` (`L07`).
- Test data is used only for the registered comparisons. The fixture's reload and prediction code runs eval-mode forward passes over all 1797 feature rows, including the 360 test features. No test label was found in the training or model-selection paths. This was found by reading code, not by a runtime test (E9, `L08`). This is a procedural disclosure, not a metric leak. It is not fixed, because `study.py` is the immutable fixture.

## 4. Arms

| Arm | Decay | Seeds | Analysis label in fixture | Test accuracy, per seed (%) | Mean (%) |
|---|---|---|---|---|---|
| regularized | 0.01 | 0, 1, 2 | confirmatory | 96.39, 96.94, 96.94 | 96.76 |
| unregularized | 0 | 0, 1, 2 | confirmatory | 96.39, 96.94, 96.94 | 96.76 |
| sensitivity | 0.001 | 0, 1, 2 | exploratory | 96.39, 96.94, 96.94 | 96.76 |
| shuffled-label control | 0.01, labels permuted | 0, 1, 2 | exploratory | 6.11, 11.67, 10.83 | 9.54 |
| logistic regression (baseline) | sklearn default L2 | 0 | exploratory | 95.56 | 95.56 |
| class-prior floor | n/a | 0 | exploratory | 10.28 | 10.28 |
| pilot (validation split only for the reported metric) | 0.01 | 0 | exploratory | 96.66 (validation) | n/a |

All numbers come from `cycles/001/results/*.json` (`experiment_summary.json`, `L01`). The table is written from those values and the summary, not typed from memory.

## 5. Confirmatory results

- **C1 (H1, primary).** Not supported. Mean paired difference 0.00 pp. The bootstrap interval is [0, 0] because the arms make identical predictions on every test example (`regularized_s0..2`, `unregularized_s0..2`). The falsification rule is not triggered, because the interval is not wholly below zero. The support rule is not met either. Predictions P1 (|mean Δ| ≤ 1 pp) and P2 (interval includes zero) hold. P3 (interval wholly above zero) does not.
  - Limit: zero discordant pairs carry no sign information. The result shows that the intervention flipped no test prediction. It does not show equivalence.
- **C2 (registered checks).** P8 holds: decay is applied, and loss histories differ at every seed. P7 holds at its lower edge: the seed-to-seed SD of test accuracy is 0.32 pp, against a registered range of 0.3–1.5 pp. Three seeds estimate this spread poorly.

## 6. Exploratory findings

These are labelled exploratory. They do not change the confirmatory conclusion.

- **E1. Sensitivity arm.** Decay 0.001 gives the same test predictions as decay 0 on every seed. Both non-zero decay values are null on accuracy at this budget (`sensitivity_s0..2`).
- **E2. Baseline and floor.** The logistic baseline (95.56%, one seed, untuned) is slightly below the MLP arms. The class-prior floor is at chance (`logistic_s0`, `class_prior_s0`).
- **E3. Training changes.** Validation-loss differences from the unregularized arm are +0.00028, +0.00035, +0.00042 at the last epoch for seeds 0–2. The maximum absolute per-epoch difference is about 0.0008. The model moves, but its test argmax does not (`L01`).
- **E4. Reproducibility.** Accuracy recomputed from stored predictions matches every record. An independent from-scratch retrain matches all 14 arm records on checkpoint hash, predictions, IDs, accuracy and loss history, on this machine and CUDA build (`L04`, `L05`, `L06`).
- **E5. Weight shrink.** The predicted cumulative shrink for decay 0.01 over 850 steps is factor 0.99154 (from `1 − lr·λ` per step, `S04`). The measured norm ratio of regularized to unregularized checkpoints is 0.99594–0.99598 on every seed. The shrink is about half the prediction. The cause is not identified. Candidate explanations: the arms follow different gradient trajectories, so their norms differ for reasons other than decay, and the norm is taken over all parameters. This discrepancy was not anticipated in the preregistration (`L02`).
- **E6. Design sensitivity.** The paired design resolves about 2.5 pp only if roughly 3% of test examples are discordant (about 11 of 360). The observed discordance is zero, so the minimum detectable effect is an assumption-based approximation, not a measurement (`L11`).
- **E7. Post hoc NLL (not promotable).** Test negative log-likelihood is higher with decay on all three seeds, by +0.00040, +0.00048 and +0.00057 nats. The conditional bootstrap interval is [+0.00024, +0.00070]. This is the only continuous signal in the study, and it points against H1. It was chosen after the primary result, so it is exploratory. Three of three seeds with the same sign gives a two-sided sign-test p = 0.25, which is weak evidence (`L03`).
- **E8. Implementation checks.** `study.py check` passes. The 13-test pytest suite passes, including a zero-gradient test of the decoupled decay step (`L07`, `L09`, `L12`).
- **E9. Code audit.** See section 3 (`L08`).
- **E10. Other registered prediction checks.** P4 (validation and test ranges) holds. P5 (shuffled-label accuracy near chance, below the 0.20 failure threshold) holds. P6 (class-prior accuracy equals the majority-class fraction of the test split) holds. These arms are exploratory by the fixture's own labels (`L05`).

## 7. Negative and inconclusive findings

1. H1 is not supported (C1). The result is a null with a degenerate interval, which is not evidence of equivalence.
2. Both non-zero decay values are null on accuracy (E1).
3. The one continuous endpoint, test NLL, points against decay (E7). It is exploratory and must be re-tested as a registered endpoint before anyone relies on it.
4. The measured weight shrink is about half the pure-decay prediction (E5). This is a recorded discrepancy and is unresolved.
5. The design cannot measure effects smaller than about 2.5 pp (E6). The study was therefore expected to be inconclusive, and it was.
6. The pilot's registered validation check and the code path differ slightly: the pilot forward-passes test features even though it reports validation metrics only (E9). The reproduction doc was corrected to say so. The code was not changed.

## 8. Process findings

- The first draft of the evidence ledger had stale hashes and duplicate result IDs. Both were repaired in the analysis phase. The repair also exposed a labelling conflict between the ledger and the records (sensitivity, shuffled, logistic and class-prior arms). The registered confirmatory set is only C1 and C2 (six records).
- The review phase found three factual errors in the drafts: a wrong S04 URL (`adamw.py` instead of `adam.py`), an incorrect "validation only" description of the pilot, and a stale reproduction-doc status. All three were fixed.
- The preregistration cites `adam.py:417`. The installed line is 419. The preregistration is a pre-result document and was left as written. The discrepancy is recorded here.

## 9. Open items at the publish step

These are defects in generated output or in unfinished work. None changes a scientific conclusion.

- **Generated paper and site.** `reports/paper.md`, `reports/paper.pdf`, `site/*`, and `reports/summary.md` were generated before the review's source fixes. They are correct only after the controller's `render` step runs. The publish step must confirm that the S04 URL reads `adam.py` and that the E9 text names 1797 rows.
- **Paper template defects (generator, not this cycle).**
  - The "Discussion, implications, and learnings" heading renders empty.
  - The status line says "5 operational phases accepted". Analysis and review are accepted but missing from the snapshot.
  - The AI statement is hard-coded as "Codex wrote the harness and study fixture". The project instructions ask for a Claude Code statement. The owner must decide the attribution wording. It belongs only in the paper, not in the README.
- **Architecture figure.** `figures/architecture.png` shows the harness pipeline, not the 64 → 128 → 10 MLP. A model diagram is still missing.
- **Site and PDF not visually checked.** The site was checked for script syntax and data shape, not opened in a browser. The PDF was checked for byte identity with the report copy, not page by page. The site uses `fetch` on `data.json`, so it must be served over HTTP (for example GitHub Pages), not opened from `file://`.
- **Hugging Face.** Model upload and the collection are the controller's job. No upload was done by this phase.

## 10. Limitations

- Three seeds, one learning rate, one 850-step budget, one fixture and one random split. Generalisation is limited to this split, not to new writers or datasets.
- The paired bootstrap conditions on the trained models and does not capture training-seed variance. Seed spread is reported separately (`L01`).
- The MLP is untuned beyond the registered constants. The logistic baseline has one seed and sklearn's default L2, not the registered decay.
- The leakage audit (E9) was done by code reading only.
- Claims S09, S10, S15, S17 and S18 were read from abstracts or search summaries. S08 and S04 were checked against source. The rest were not re-fetched in the review phase (`research/sources.json`).
- Results were reproduced on this machine and CUDA build only.

## 11. What would change the conclusion

A new cycle would need, before any results exist: a continuous primary endpoint (for example test NLL, with its direction and falsifier written down); more seeds or a larger evaluation set so that the discordance assumption can be measured; a learning rate or step budget at which decay actually changes predictions; and a per-step weight-norm log to test E5 directly. Until then, the honest reading is: at lr 0.001, λ 0.01, 850 steps, on this split, decay does not change test accuracy, and the study cannot say whether it changes anything smaller.

## 12. Sources

Full titles, URLs, and read depth are in `cycles/001/research/sources.json` (S01–S18). The main ones are: AdamW (`S01`, arXiv 1711.05101), Adam (`S02`), the PyTorch AdamW documentation (`S03`), the PyTorch decoupled-decay source (`S04`, `torch/optim/adam.py`), scikit-learn `load_digits` (`S05`), the UCI optdigits dataset (`S06`, CC BY 4.0), Zhang et al. on random-label fitting (`S07`, arXiv 1611.03530), and Krogh & Hertz (`S08`, NIPS 1991).

## 13. Reproduction

From `cycles/001`, with the project interpreter (`.venv\Scripts\python.exe`):

- `study.py check`: split and fixture check.
- `study.py pilot`: validation-only pilot (see the reproduction doc).
- `study.py experiments`: registered comparisons.
- `pytest tests`: 13 tests.

Step-by-step instructions and the seed-level records are in `cycles/001/docs/reproduction.md` and `cycles/001/results/`.
