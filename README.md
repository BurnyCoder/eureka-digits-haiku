# eureka-digits-haiku

Execute the fixed digits validation study supplied in this project's cycle directory.

Investigate whether AdamW weight decay improves the held-out accuracy of the fixed ReLU MLP. Follow the immutable supplied protocol. Use the phase-specific tasks supplied by the harness, actual web research, meaningful code checks, real CUDA training when available, pilot publication before sensitivity experiments, all registered comparisons, careful evidence-linked interpretation, and the requirement checklist.

The primary objective of this run is an inspectable completed research cycle, regardless of whether regularization helps, hurts, or has an inconclusive effect. Do not select methods using the test set. Record failures, confounders, alternative explanations, predictions and changes in understanding. Follow Neel Nanda's research stages and prioritize useful learning per unit time. This is a small illustrative study, not evidence of broad scientific reliability or agent superiority.

Only the selected model/backend may be used. The controller handles publication and budgets. Do not launch other LLMs, coding CLIs, or collaborators. Preserve all recorded artifacts.


Run status: not yet executed. See cycles/ for protocols, code, and generated reports.


<!-- EUREKA:RESULTS:START -->
## Recorded results

Recorded harness status: **budget_exhausted**; 5 operational phases accepted. Measurements below do not imply completion of unaccepted phases.

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

See [full report](cycles/001/reports/summary.md).
<!-- EUREKA:RESULTS:END -->
