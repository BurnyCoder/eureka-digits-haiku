# class_prior_s0

Method: class_prior. Analysis: exploratory.

| Experiment | Method | Seed | Split | Accuracy | Seconds | Device |
|---|---|---:|---|---:|---:|---|
| class_prior_s0 | class_prior | 0 | test | 0.102778 | 0.026 | cpu |

## Method, provenance, and learning

Class-prior baseline fitted on training labels. Neural architecture, optimizer, batches and epochs do not apply. The committed study source defines prediction behavior.

Real-label models should outperform the label-shuffled control; the primary decay effect may be positive, negative, or inconclusive.

One random example split without writer IDs; conditional intervals exclude training-population uncertainty.

Protocol SHA256: `5edcfb3fcda62035f6ea4013f88ebca59b7d75977f49d155af208a2bb50ec257`. Code SHA256: `34415a875e7f61954cf3813486f46a189d6b647f2e9682552d16db88a6d67d39`.
