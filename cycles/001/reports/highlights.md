# Highlights and anomalies (research phase, cycle 001)

No experimental results exist yet. Everything below comes from reading sources, the fixture's code, and arithmetic on the registered constants. Full detail and citations are in `reports/research.md` and `research/sources.json`.

## Highlights

1. **The registered intervention is probably too small to show up at this budget.** AdamW decay with λ = 0.01 and lr = 0.001 shrinks weights by `(1 − 1e-5)` per step. Over the 850 steps of the run, that is a cumulative shrink of about 0.85% (0.085% for the sensitivity arm). This follows from the installed PyTorch code (`torch/optim/adam.py:419`) and the protocol's constants [S04]. It is an expectation, not a measurement. I predict the effect will be within ±1 pp and the CI will span zero (P1, P2).
2. **Resolution is coarse.** The test split has 360 examples, so one flipped prediction is 0.278 pp. Under an assumed ~3% disagreement rate between arms, the minimum detectable paired difference is about 2.5 pp for a single comparison. The protocol's test-only bootstrap does not capture seed variance, which is estimated from three seeds (see `research.md` §4).
3. **The shuffled-label control is not a clean decay control.** It runs only with λ = 0.01. It cannot show whether decay changes memorization, because there is no shuffled-label arm with λ = 0. The protocol is immutable, so this stays a stated limitation. The control still tests whether the pipeline can fit noise at all (P5).
4. **The data is CC BY 4.0, not BSD.** `load_digits` is a copy of the UCI optdigits test file [S05, S06]. The scikit-learn BSD line covers code and docs only. The paper must attribute UCI and CC BY 4.0.
5. **Prior evidence is mixed in the direction that matters.** Loshchilov & Hutter report that decoupled decay improves Adam's generalization [S01]. Bjorck et al. [S09] attribute much of the benefit to how L2 interacts with Adam's first-moment buffer. Zhang et al. [S07] show that regularization does not change memorization of random labels in large networks. None of these claims was tested here. They motivate the question, not the answer.

## Anomalies and issues found

1. **Citation mismatch (fixed in notes).** A search returned a Nanda post (`sYHrW4wwfoMBxNDcA`, "Real-Time Research Recording") that is not the sequence the protocol cites. It is not cited. The protocol's sequence URL is correct and was verified [S13].
2. **Two sources had only abstract-level access.** The percentile-bootstrap details [S15] and Lones's leakage guidance [S12] were not verified in full text. The analysis phase must source the bootstrap method before computing the CI.
3. **Stage definitions were initially unavailable from the sequence page.** The sequence index lists the post titles but not their text. The explore/understand/distill/ideation definitions came from the LessWrong mirror of the opener post, read directly [S14].
4. **`study.py check` passed** with train 1078 / validation 359 / test 360 and features in [0, 1] after /16 scaling. The check does not train anything.

## Predictions to watch (recorded before any results)

- P1: |mean paired Δ| ≤ 1.0 pp. P2: the 95% CI includes 0. P3: the CI wholly above 0 (~10–15%).
- P5: shuffled-label test accuracy near chance (≈0.10). A value above 0.20 would indicate a bug.
- P8: regularized and unregularized loss histories differ. Identical histories would mean the decay is not applied.

## What I changed in my understanding

- Before reading the fixture, I expected a decay effect of the size usually reported for image models. Reading the update rule and computing the shrink showed that, under these constants, the expected effect is much smaller. The conclusion is inconclusive by design unless the effect is larger than the arithmetic suggests.
- The main open question is therefore not whether decay helps in general, but whether this protocol can detect any decay effect at all. The seed-level results and the per-seed SD will show that more directly than the pooled CI.

## Open items for later phases

- Pilot (validation only) before any comparison. Confirm P4 and the reload check.
- Source the percentile-bootstrap method before the analysis phase.
- Report per-seed effects and their SD next to the CI.
- Report the shuffled-label control's limitation (§5.4 of `research.md`) in the paper.
