# All experiments

Recorded harness status: **complete**; 8 operational phases accepted. Measurements below do not imply completion of unaccepted phases.

| Experiment | Method | Seed | Split | Accuracy | Seconds | Device |
|---|---|---:|---|---:|---:|---|
| class_prior_s0 | class_prior | 0 | test | 0.102778 | 0.026 | cpu |
| logistic_s0 | logistic | 0 | test | 0.955556 | 0.107 | cpu |
| pilot_regularized_s0 | regularized | 0 | validation | 0.966574 | 6.805 | cuda |
| regularized_s0 | regularized | 0 | test | 0.963889 | 0.486 | cuda |
| regularized_s1 | regularized | 1 | test | 0.969444 | 6.630 | cuda |
| regularized_s2 | regularized | 2 | test | 0.969444 | 2.553 | cuda |
| sensitivity_s0 | sensitivity | 0 | test | 0.963889 | 2.525 | cuda |
| sensitivity_s1 | sensitivity | 1 | test | 0.969444 | 2.583 | cuda |
| sensitivity_s2 | sensitivity | 2 | test | 0.969444 | 2.481 | cuda |
| shuffled_labels_s0 | shuffled_labels | 0 | test | 0.061111 | 2.465 | cuda |
| shuffled_labels_s1 | shuffled_labels | 1 | test | 0.116667 | 2.470 | cuda |
| shuffled_labels_s2 | shuffled_labels | 2 | test | 0.108333 | 2.367 | cuda |
| unregularized_s0 | unregularized | 0 | test | 0.963889 | 2.426 | cuda |
| unregularized_s1 | unregularized | 1 | test | 0.969444 | 2.393 | cuda |
| unregularized_s2 | unregularized | 2 | test | 0.969444 | 2.501 | cuda |

The mean paired accuracy difference is 0.000000, with conditional 95% bootstrap interval [0.000000, 0.000000]. The registered decision is **inconclusive**. The sample standard deviation across seed effects is 0.000000. The interval uses 10000 paired resamples of test examples and conditions on the trained models.

- **analysis_kind**: "confirmatory"
- **objective**: "Exercise the complete research harness on a bounded reproducible study; negative and inconclusive effects are valid."
- **question**: "Does AdamW weight decay improve held-out accuracy of a small MLP on the bundled digits data?"
- **architecture**: [64, 128, 10]
- **optimizer**: "AdamW"
- **learning_rate**: 0.001
- **batch_size**: 64
- **epochs**: 50
- **methods**: ["regularized", "unregularized", "sensitivity", "shuffled_labels", "logistic", "class_prior"]
- **seeds**: [0, 1, 2]
- **data_boundaries**: "Frozen stratified 60/20/20 train/validation/test partition. Split seed 42. Divide pixels by the documented constant 16. Pilot reports validation only. No learned preprocessing or selection uses test data. Generalization is limited to this random example split, not new writers or datasets."
- **analysis_plan**: "Use 10000 paired bootstrap resamples of test example indices with RNG seed 2026, preserving pairing across methods and all trained seeds. This interval conditions on the trained models and does not estimate training-population uncertainty. Separately show each seed effect and its sample standard deviation. Treat other comparisons as descriptive, with no post-hoc primary selection."
- **hyperparameter_rationale**: "Small prespecified network and fixed training budget provide fast feedback on an established task. The three decay values are a descriptive local sensitivity check, not a test-selected sweep. Three seeds expose limited initialization sensitivity; they do not support broad population claims."

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
- **C2 (registered checks, confirmatory arms).** The loss histories differ across arms, consistent with the intervention changing training (P8, `regularized_s0..2`, `unregularized_s0..2`). Seed spread falls within the registered range (P7); the generated results report its value. The limited seed sample estimates this spread poorly and does not validate the full optimizer implementation.

## 5. Exploratory findings
- **E1.** The sensitivity arm (decay 0.001) gives the same test predictions as the unregularized arm on every seed (`sensitivity_s0..2`, `unregularized_s0..2`). Both non-zero decay values are null on accuracy.
- **E2.** The logistic baseline (`logistic_s0`) is slightly below the MLP arms, and the class-prior floor (`class_prior_s0`) is at chance. The baseline has one seed and is not tuned.
- **E3.** Decay changes training: loss histories differ from the unregularized arm at every seed (`L01`, `regularized_s0`, `unregularized_s0`). The model moves, but its test argmax does not.
- **E4.** Reproducibility audits pass (`L04`, `L05`, `L06`). This was done after the registered comparison, so it is exploratory.
- **E5.** Checkpoint norms (`L02`). Fitted arm norm ratios differ from hypothetical pure-decay multiplication. Gradient trajectories also differ, so this is not an isolated decay measurement or a causal explanation for unchanged accuracy.
- **E6.** Sensitivity scenarios (`L11`). The audit uses assumed discordance rates and a normal approximation. It establishes neither a hard detection limit nor that smaller effects must be inconclusive. Agreement in the observed sample does not prove population agreement or equivalence.
- **E7.** Post hoc test negative log-likelihood (`L03`). The saved analysis reports higher loss with decay and a conditional interval excluding zero. This exploratory finding concerns a separate endpoint and does not test the primary accuracy hypothesis. Validation loss is another continuous measurement that changed.
- **E8.** Implementation checks pass today: the split check and the 13-test suite (`L07`, `L09`, `L12`).
- **E9.** Code audit (`L08`). The fixture's reload check and prediction step run eval-mode forward passes on all 1797 feature rows, which include the 360 test features (the pilot's code path does this too, although the pilot reports validation metrics only). No test label was found in the training or model-selection paths. This is a procedural disclosure, not a leak into the reported metric. It was found by reading code, not by a runtime test.
- **E10.** Other registered prediction checks (`pilot_regularized_s0`, `shuffled_labels_s0..2`, `logistic_s0`, `class_prior_s0`). The validation and test ranges hold (P4). Shuffled-label accuracy is near chance and under the failure threshold (P5). Class-prior accuracy equals the majority-class fraction of the test split (P6). Accuracy recomputed from stored predictions matches the records (A4(d), `L05`). These arms are labelled exploratory by the fixture, so the outcomes are reported under that label.

## 6. Negative and inconclusive findings
- H1 (decay improves held-out accuracy) is not supported (C1). The result is a null with a degenerate interval, which is not evidence of equivalence.
- The sensitivity arm is also null on accuracy (E1).
- The exploratory NLL finding differs from the primary accuracy result (E7); a new confirmatory study would require prospective analysis and untouched evaluation data.
- Checkpoint norms do not isolate the pure-decay mechanism (E5).
- The sensitivity scenarios do not establish a measured detection threshold (E6).

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


## Sources

- [S01: Decoupled Weight Decay Regularization (AdamW), Loshchilov & Hutter, ICLR 2019](https://arxiv.org/abs/1711.05101)
- [S02: Adam: A Method for Stochastic Optimization, Kingma & Ba, ICLR 2015](https://arxiv.org/abs/1412.6980)
- [S03: torch.optim.AdamW documentation (PyTorch 2.15)](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)
- [S04: PyTorch adam.py: decoupled weight decay update (torch 2.14.1 installed locally, plus upstream main)](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adam.py)
- [S05: sklearn.datasets.load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [S06: Optical Recognition of Handwritten Digits (UCI ML Repository), Alpaydin & Kaynak 1998](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [S07: Understanding deep learning requires rethinking generalization, Zhang et al., ICLR 2017](https://arxiv.org/abs/1611.03530)
- [S08: A Simple Weight Decay Can Improve Generalization, Krogh & Hertz, NIPS 1991 (vol. 4)](https://papers.nips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html)
- [S09: Understanding Decoupled and Early Weight Decay, Bjorck et al., 2020](https://arxiv.org/abs/2012.13841)
- [S10: Towards Understanding Convergence and Generalization of AdamW, Zhou et al., IEEE TPAMI 2024](https://panzhous.github.io/assets/pdf/2024-TPAMI-AdamW.pdf)
- [S11: Troubling Trends in Machine Learning Scholarship, Lipton & Steinhardt, 2018](https://arxiv.org/abs/1807.03341)
- [S12: How to avoid machine learning pitfalls: a guide for academic researchers, Lones](https://arxiv.org/abs/2108.02497)
- [S13: How I Think About My Research Process (sequence), Neel Nanda](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL)
- [S14: How I Think About My Research Process: Explore, Understand, Distill, Neel Nanda, 26 Apr 2025](https://www.greaterwrong.com/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand)
- [S15: Construction of Confidence Intervals, Dalitz, arXiv:1807.03582](https://arxiv.org/abs/1807.03582)
- [S16: safetensors torch API documentation (save_file, load_file)](https://huggingface.co/docs/safetensors/en/api/torch)
- [S17: scipy.stats.bootstrap documentation (percentile method, paired option)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [S18: oceCens percci documentation (percentile interval, citing Efron & Tibshirani 1993, p. 168)](https://search.r-project.org/CRAN/refmans/oceCens/html/percci.html)
