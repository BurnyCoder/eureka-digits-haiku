# logistic_s0

Method: logistic. Analysis: exploratory.

| Experiment | Method | Seed | Split | Accuracy | Seconds | Device |
|---|---|---:|---|---:|---:|---|
| logistic_s0 | logistic | 0 | test | 0.955556 | 0.107 | cpu |

## Method, provenance, and learning

Multinomial logistic regression: C=1.0, max_iter=1000. Neural architecture, AdamW, batches and epochs do not apply. See the [scikit-learn estimator](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Real-label models should outperform the label-shuffled control; the primary decay effect may be positive, negative, or inconclusive.

One random example split without writer IDs; conditional intervals exclude training-population uncertainty.

Protocol SHA256: `5edcfb3fcda62035f6ea4013f88ebca59b7d75977f49d155af208a2bb50ec257`. Code SHA256: `34415a875e7f61954cf3813486f46a189d6b647f2e9682552d16db88a6d67d39`.
