# regularized_s2

Method: regularized. Analysis: confirmatory.

| Experiment | Method | Seed | Split | Accuracy | Seconds | Device |
|---|---|---:|---|---:|---:|---|
| regularized_s2 | regularized | 2 | test | 0.969444 | 2.553 | cuda |

## Method, provenance, and learning

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

Real-label models should outperform the label-shuffled control; the primary decay effect may be positive, negative, or inconclusive.

One random example split without writer IDs; conditional intervals exclude training-population uncertainty.

Protocol SHA256: `5edcfb3fcda62035f6ea4013f88ebca59b7d75977f49d155af208a2bb50ec257`. Code SHA256: `34415a875e7f61954cf3813486f46a189d6b647f2e9682552d16db88a6d67d39`.
