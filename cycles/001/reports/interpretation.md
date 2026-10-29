# Interpretation: AdamW weight decay on a fixed digits MLP (cycle 001)

This report interprets the completed registered study. It contains no typed empirical scores. Each result is cited by claim ID (`C#` confirmatory, `E#` exploratory; defined in `research/claims.json`), experiment record ID (the `id` of `results/<id>.json`, e.g. `regularized_s0`), local artifact ID (`L##`, `research/evidence.json`), or source ID (`S##`, `research/sources.json`). Numbers live in the artifacts those IDs point to. Protocol and preregistration are in `research/protocol.json` (`L10`) and `reports/preregistration.md`, committed before any result existed.

## 1. Motivation
Decoupled weight decay (AdamW, `S01`) is a standard regulariser for MLPs. The question is narrow: does a fixed ReLU MLP on the digits fixture gain held-out accuracy from decay 0.01, compared with decay 0 and a decay 0.001 sensitivity arm? The study follows the scientific-method loop recommended in the harness contract and in Neel Nanda's research-process notes (`S13`, `S14`): define the question, record predictions, run the registered comparison, then report what was and was not learned. It is a small illustrative study, not evidence of broad reliability.

## 2. Context and related work
- **Mechanism.** Decoupled decay multiplies the weights by a factor `1 - lr*lambda` at each step, independently of the gradient. The PyTorch implementation is in `adam.py` (`S04`; documentation in `S03`). This is the origin of the shrink analysis in E5.
- **Original method.** `S01` argues that L2 regularisation and weight decay differ for Adam and proposes decoupling. Adam itself is `S02`. The claim that decoupling improves generalisation comes from `S01`'s abstract and was not re-tested beyond this fixture.
- **Weight decay history.** Krogh & Hertz (`S08`) was verified from the proceedings; the later analyses of decoupled and early decay (`S09`) and of AdamW convergence (`S10`) were read from abstracts only and are cited as context, not as evidence.
- **Data.** The fixture is scikit-learn's `load_digits` (`S05`), a copy of the UCI optdigits data under CC BY 4.0 (`S06`). `S07` motivates the shuffled-label control: networks can fit random labels, so training accuracy alone does not indicate generalisation.
- **Methodology.** The confirmatory/exploratory split, controls and falsification rules follow `S11` and `S12`. The bootstrap interval uses the percentile method (`S17`, `S18`, both read as search-result summaries, not full text; `S15` read at abstract level only).

## 3. Methods
- **Fixture.** Stratified split train 1078 / validation 359 / test 360, split seed 42, pixels divided by 16 (`research/protocol.json`, `L10`; split check in `L07`). The test split is used only for the registered comparisons; see E9 for a read-only audit of test-feature use in the fixture.
- **Model and training.** ReLU MLP `[64, 128, 10]`, AdamW, learning rate 0.001, batch size 64, 50 epochs (850 optimizer steps), seeds 0, 1, 2 (`research/protocol.json`).
- **Arms.** Regularized (decay 0.01: `regularized_s0..2`); unregularized (decay 0: `unregularized_s0..2`); sensitivity (decay 0.001: `sensitivity_s0..2`); shuffled-label control (training labels permuted, decay 0.01: `shuffled_labels_s0..2`); logistic regression baseline (`logistic_s0`); class-prior floor (`class_prior_s0`). Pilot on validation only: `pilot_regularized_s0`.
- **Primary metric.** Test accuracy. Paired comparison of regularized vs unregularized predictions per test example, with 10000 paired bootstrap resamples of test indices (`protocol.json`). The bootstrap conditions on the trained models and does not measure training-seed variance; seed spread is reported separately (`L01`).
- **Reproducibility.** Study checks, pytest suite and from-scratch rerun are in `L04`, `L05`, `L06`, `L07`, `L09`, `L12`.

## 4. Confirmatory results
- **C1 (primary, H1).** Not supported. The mean paired difference is zero and the bootstrap interval is degenerate at zero, because the arms make identical test predictions for every seed (`regularized_s0..2`, `unregularized_s0..2`, `L01`). The falsification rule is not triggered, since the interval is not wholly below zero, but the directional prediction is not observed. Predictions P1 and P2 hold; P3 does not (`reports/preregistration.md`).
- **C2 (registered checks, confirmatory arms).** Decay is applied: the per-epoch loss histories differ from the unregularized arm for every seed (P8, `regularized_s0..2`, `unregularized_s0..2`). The seed-to-seed SD of test accuracy is 0.32 pp, inside the registered 0.3-1.5 pp range at its lower edge (P7, same records). Three seeds estimate this spread poorly.

## 5. Exploratory findings
- **E1.** The sensitivity arm (decay 0.001) gives the same test predictions as the unregularized arm on every seed (`sensitivity_s0..2`, `unregularized_s0..2`). Both non-zero decay values are null on accuracy.
- **E2.** The logistic baseline (`logistic_s0`) is slightly below the MLP arms, and the class-prior floor (`class_prior_s0`) is at chance. The baseline has one seed and is not tuned.
- **E3.** Decay changes training: loss histories differ from the unregularized arm at every seed (`L01`, `regularized_s0`, `unregularized_s0`). The model moves, but its test argmax does not.
- **E4.** Reproducibility audits pass (`L04`, `L05`, `L06`). This was done after the registered comparison, so it is exploratory.
- **E5.** Weight-level mechanism (`L02`). Decay shrinks the weights in the direction the update rule predicts. The measured shrink is about half of the pure-decay prediction on both decay values. The cause is not identified. Candidate explanations are that the arms follow different gradient trajectories, so their norms also differ for reasons other than decay, and that the norm is taken over all parameters. This explains why the intervention is small, but it does not give an exact magnitude.
- **E6.** Design sensitivity (`L11`). The paired design resolves about 2.5 pp only if roughly 3 percent of test examples are discordant. The observed discordance is zero, so the minimum detectable effect cannot be measured. Effects below that size are expected to be inconclusive. The research-phase value is reproduced from a normal approximation and an assumed discordance, not from data.
- **E7.** Post hoc test negative log-likelihood (`L03`). The decay arm has slightly higher NLL on all three seeds, with a conditional bootstrap interval excluding zero. This is the only continuous signal in the study, and it points against the hypothesis. It was chosen after the primary result, so it cannot be promoted to confirmatory.
- **E8.** Implementation checks pass today: the split check and the 13-test suite (`L07`, `L09`, `L12`).
- **E9.** Code audit (`L08`). The fixture's reload check and prediction step run eval-mode forward passes on all 1797 feature rows, which include the 360 test features (the pilot's code path does this too, although the pilot reports validation metrics only). No test label was found in the training or model-selection paths. This is a procedural disclosure, not a leak into the reported metric. It was found by reading code, not by a runtime test.
- **E10.** Other registered prediction checks (`pilot_regularized_s0`, `shuffled_labels_s0..2`, `logistic_s0`, `class_prior_s0`). The validation and test ranges hold (P4). Shuffled-label accuracy is near chance and under the failure threshold (P5). Class-prior accuracy equals the majority-class fraction of the test split (P6). Accuracy recomputed from stored predictions matches the records (A4(d), `L05`). These arms are labelled exploratory by the fixture, so the outcomes are reported under that label.

## 6. Negative and inconclusive findings
- H1 (decay improves held-out accuracy) is not supported (C1). The result is a null with a degenerate interval, which is not evidence of equivalence.
- The sensitivity arm is also null on accuracy (E1).
- The only continuous endpoint points against decay (E7). This is exploratory and must be re-tested with a registered endpoint, not reported as a finding.
- The mechanism is smaller than the pure-decay arithmetic predicts (E5). This was not anticipated in the preregistration and is recorded as a discrepancy.
- The design cannot measure an effect smaller than about 2.5 pp (E6). The MDE prediction held only in the sense that no effect was resolved.

## 7. Implications
- For this fixture and budget, decay has no detectable effect on test argmax, so the registered comparison cannot recommend decay.
- Accuracy alone cannot give a direction here. A future cycle needs a preregistered continuous primary endpoint (such as the NLL in E7), more seeds or a larger test set, and a budget where decay matters. A preregistered weight-norm measurement would test E5 directly.
- The most useful procedural change is to remove test features from the reload and prediction path, or to document the exception before the next comparison (E9).

## 8. Limitations
- Three seeds, one learning rate, one 850-step budget, one fixture and one random split. Generalisation is limited to this split, not new writers or datasets.
- Zero discordant pairs give no sign information for accuracy (C1).
- The bootstrap resamples test examples only, conditioning on the trained models (Methods).
- The logistic baseline has one seed, and the MDE in E6 is an approximation.
- E1-E10 are descriptive or post hoc and are labelled exploratory. Only C1 and C2 are confirmatory.
- The sign consistency in E7 (three of three seeds) is weak evidence: a two-sided sign test on three seeds gives p = 0.25.

## 9. Related work
Decoupled decay and Adam: `S01`, `S02`, `S03`, `S04`. Weight decay and generalisation: `S08`, `S09`, `S10`. Random-label memorisation: `S07`. Data: `S05`, `S06`. Research methodology: `S11`, `S12`, `S13`, `S14`. Interval construction: `S15`, `S17`, `S18`. Artifact format: `S16`. Full titles and URLs are in `research/sources.json`.
