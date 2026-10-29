# Research phase: AdamW weight decay on a fixed digits MLP

Cycle 001. Phase: research (ideation → exploration). No models were trained in this phase. Sources are listed in `research/sources.json` (IDs S01–S16) and cited inline as `[S0x]`.

## 1. Domain and question

- **Domain:** supervised image classification on small 8×8 grayscale digits, with a fixed 64-128-10 ReLU MLP.
- **Question (registered, `research/protocol.json`):** Does AdamW weight decay improve held-out accuracy of this MLP on the bundled digits data?
- **Registered primary metric:** mean over seeds {0,1,2} of `test_acc(weight_decay=0.01) − test_acc(weight_decay=0)`.
- **Data:** `sklearn.datasets.load_digits`, 1797 samples, 64 features in 0–16, divided by the constant 16. Stratified split with seed 42 into train 1078 / validation 359 / test 360 (60/20/20). `python study.py check` passed in this phase and printed exactly these sizes.

## 2. Literature review

### 2.1 Prior method: decoupled weight decay

- Adam [S02] adapts per-parameter step sizes using running estimates of gradient moments. Kingma & Ba's abstract describes this as "based on adaptive estimates of lower-order moments".
- Loshchilov & Hutter [S01] argue that L2 regularization and weight decay are equivalent for SGD, but not for Adam, because Adam rescales the L2 gradient term per parameter. Their proposed AdamW applies the decay directly to the weights, outside the adaptive gradient step. The abstract claims this "substantially improves Adam's generalization performance". The paper is cited as ICLR 2019. I read the abstract only and did not verify its CIFAR/ImageNet numbers.
- PyTorch's `torch.optim.AdamW` default is `weight_decay=0.01` and `lr=0.001` [S03]. The protocol overrides `weight_decay` explicitly for each arm, so the default is not used.
- **Implementation check (installed torch 2.14.1):** the decoupled path in `torch/optim/adam.py:419` executes `param.mul_(1 - lr * weight_decay)` when `decoupled_weight_decay=True` [S04]. Decay is therefore a multiplicative shrink of the weights by `(1 − lr·λ)` per optimizer step, applied independently of the gradient. This is the mechanism whose size I compute in §4.
- Follow-up work [S09, S10] reports that the decoupled-vs-L2 distinction is an active research question. Bjorck et al. [S09] attribute the main benefit to avoiding the mixing of the L2 term into Adam's first-moment buffer. Zhou et al. [S10] give a theoretical generalization argument. Both were read at the abstract level only.

### 2.2 Why weight decay might not matter here

- Zhang et al. [S07] show that large image networks can fit random labels and that explicit regularization does not change this qualitatively. A tiny MLP on 1078 examples may memorize less, so the regularizer may have less to act on. This is a hypothesis, not a result from this study.
- Krogh & Hertz [S08] give the classical argument that weight decay can improve feed-forward generalization by suppressing weight components that do not fit the task. Their paper is from the early 1990s and the claim is not tested on this data.

### 2.3 Methodology sources

- Lipton & Steinhardt [S11] list the failure modes this study must avoid: unclear source of gains, mathiness, and conflating speculation with explanation. The protocol's rule that every gain must be attributed to the registered intervention (decay), not to seed, order or tuning, follows directly from this.
- Lones [S12] gives a five-stage pitfall checklist, including fair comparison and reporting. I used its stage structure as a checklist. I did not verify its specific leakage recommendations in full text.
- Nanda's research-process sequence [S13] and the opener post [S14] define the stages the harness follows: ideation (choose a problem), exploration ("Your north star is just to gain information"; check "am I getting enough information per unit time?"), understanding ("Convince yourself of a key hypothesis"), and distillation ("Compress your research findings into concise, rigorous truth"). This phase is exploration. No hypothesis here is mature.
- Bootstrap: the protocol uses a 10,000-resample paired bootstrap of test example indices. I found only abstract-level support [S15] and did **not** verify the percentile-interval details. That is a gap to close in the analysis phase, before the interval is computed.

### 2.4 Dataset provenance and license

- `load_digits` is a copy of the UCI optdigits **test** file; the UCI training file has 3823 examples [S05, S06]. The 1797 samples are therefore a single UCI file, not the full UCI collection.
- The UCI dataset is licensed **CC BY 4.0** [S06]. The scikit-learn BSD line in the docs footer covers its code and documentation, not the data [S05]. The paper must attribute the UCI source and CC BY 4.0 terms.

## 3. Predictions (recorded before any training)

Each prediction states what would confirm or falsify it. None has been tested.

| ID | Prediction | Confidence | Falsified if |
|---|---|---|---|
| P1 | Paired mean Δ (decay 0.01 − 0) is within ±1.0 pp (≈ ±3.6 test examples). | ~70% | \|mean Δ\| > 1.0 pp |
| P2 | The paired 95% bootstrap CI for Δ includes 0. The protocol treats this as "inconclusive". | ~75% | — (this is the expected inconclusive outcome) |
| P3 | The CI lies wholly above 0 (supports H1). | ~10–15% | Not falsified by this alone; supported if CI > 0 |
| P4 | Real-label MLP test accuracy is 0.93–0.99, and the logistic baseline is 0.94–0.98. | ~80% | Real-label test < 0.85 would suggest a training bug |
| P5 | The shuffled-label arm reaches near-chance test accuracy (≈0.10; ~0.05–0.16). | ~90% | Test accuracy > 0.20 would indicate leakage or a bug, since labels are permuted on the training set only |
| P6 | Class-prior baseline test accuracy is ≈ the majority-class fraction of the test split (≈0.10). | ~90% | Any large deviation signals an indexing bug |
| P7 | Seed-to-seed SD of test accuracy is 0.3–1.5 pp, comparable to or larger than any decay effect. | ~60% | — |
| P8 | Regularized and unregularized arms produce **different** per-epoch loss histories at float precision. | ~99% | **Identical** histories for a seed would mean the decay is not being applied. This is an implementation check. |

### 3.1 Mechanistic prediction (computed, not measured)

Using `lr = 1e-3`, `batch_size = 64`, `train = 1078`, and `epochs = 50`:

- Steps per epoch = ⌈1078 / 64⌉ = 17, so total steps = 850.
- For λ = 0.01, the per-step factor is `1 − 1e-5`. The cumulative factor is `(1 − 1e-5)^850 = 0.99154`, i.e. weights shrink by **≈0.85%** over the run, **ignoring** the gradient updates.
- For the sensitivity arm, λ = 0.001 gives a cumulative shrink of **≈0.085%**.

So the registered intervention is small at this budget. Adam's own updates are much larger than the decay term, so the decay effect on predictions is expected to be second-order. This is the main reason for P1 and P2. It is an arithmetic consequence of the protocol's constants, not an empirical result. `study.py` does not log weight norms, so this shrink is not directly observable from the fixture's outputs; it stays an unverified mechanistic expectation unless a future cycle registers a norm measurement.

## 4. Power and resolution (back-of-envelope, stated assumptions)

- With 360 test examples, one correct/incorrect flip changes accuracy by 100/360 = **0.278 pp**. Accuracy differences are quantized in steps of 0.278 pp.
- **Assumption (not measured):** about 3% of test predictions differ between two arms (≈11 examples). The standard error of a paired per-example accuracy difference is then √(p/n) = √(0.03/360) ≈ 0.91 pp.
- Under that assumption, the minimum detectable difference for one comparison at α = 0.05 (two-sided) and 80% power is ≈ 2.8 × 0.91 ≈ **2.5 pp**. Effects below this will mostly be inconclusive, even with a correct analysis.
- **Important limitation:** the registered bootstrap resamples **test examples** only. It quantifies test-sampling noise, conditional on the three trained models per arm. It does **not** capture seed-to-seed training variance, which is estimated from just three seeds. The per-seed effects and their SD must be reported alongside the CI, as the protocol requires.

## 5. Competing explanations and confounders

Each of these could produce an apparent decay effect without decay causing it:

1. **Seed/initialization and data-order variance.** Controlled by pairing. Both arms share the same seed, the same initial weights (same `torch.manual_seed` and architecture), and the same `order_rng` batch order. Only `weight_decay` differs. The paired design isolates decay, but three seeds leave seed variance poorly estimated (see §4).
2. **Nondeterminism.** `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` are set. The GPU is an RTX 5070 Laptop (CUDA available, torch 2.14.1+cu130). Still, GPU results may differ from CPU results. The protocol records the device per run, so this can be checked.
3. **Pilot reuse.** The regularized seed-0 checkpoint from the validation-only pilot is reused for the final test evaluation. This is the same weights and code, so it does not select anything on test, but the pilot must be published first.
4. **Shuffled-label control is not paired with an undecayed control.** It runs only with λ = 0.01 (`shuffled_labels` in the protocol). It shows memorization capacity under decay, but it **cannot** isolate whether decay changes memorization. A shuffled-label arm with λ = 0 is absent, and I am not adding it here: the protocol is immutable and any addition must go through a new cycle. This is a limitation to state in the paper.
5. **Baselines are not tuned.** `logistic` uses sklearn's default solver with C = 1.0 and max_iter = 1000 [S05]. It is a useful reference because its L2 regularization is a different regularizer, but it is not a tuned baseline.
6. **Decoupled vs L2 is not tested.** The study compares AdamW(λ=0.01) with AdamW(λ=0). Loshchilov & Hutter's claim [S01] concerns AdamW vs Adam with L2. A result here does not speak to that claim.
7. **Fixed 50 epochs, no early stopping.** The final epoch is used, so validation is not used for model selection. This avoids selection bias but means over- or under-training is not controlled.

## 6. Falsification criteria (pre-specified)

- **H1 (positive effect) is contradicted** if the paired 95% bootstrap CI for Δ is wholly below 0.
- **H1 is supported** only if the CI is wholly above 0. Otherwise the result is inconclusive, and "no effect" cannot be claimed (an interval spanning 0 is not evidence of equivalence).
- **Implementation error** is indicated by: (a) identical loss histories for regularized and unregularized arms (P8); (b) reload mismatch (the code asserts exact equality, `rtol=0, atol=0`); (c) shuffled-label test accuracy > 0.20 (P5).
- **Harness error:** a change to `study.py` after results exist invalidates them (the code hash is checked). Preserve old runs, do not overwrite.

## 7. Sources I did not verify (gaps)

- Percentile-bootstrap formula and coverage [S15]: abstract-level only. Must be sourced before the analysis phase computes the CI.
- Safetensors license and format guarantees [S16]: API verified; license not verified.
- Lones [S12] and Lipton & Steinhardt [S11]: abstract-level or search summaries only.
- Krogh & Hertz [S08] and Bjorck et al. [S09]: metadata and abstract only.
- The Loshchilov & Hutter numeric results [S01]: not read.

## 8. Process notes

- **Citation correction:** during this phase I found that one of my search results for Nanda's research-process material was a different post (`sYHrW4wwfoMBxNDcA`, "Real-Time Research Recording"). It is **not** cited. The protocol's cited URL, `https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL`, is the correct sequence index, and I verified it [S13].
- The `check` operation passed. No pilot or comparison was run. No test data was accessed.

## 9. Next phase inputs

- Pilot (validation only): `python study.py pilot` publishes the regularized seed-0 validation run and checkpoint. It can check P4 (validation accuracy in range) and the reload-equality assertion. It cannot check P8, because that needs the unregularized run, which belongs to the comparison phase. P5 and P6 are not covered by the pilot either: they are run in the comparison.
- Compare the pilot's validation results with P4 before the comparison runs. Any mismatch goes into highlights, and the prediction is not edited afterwards.
