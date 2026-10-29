# eureka-digits-haiku

A bounded Claude Code Haiku execution of the [Auto Eureka Scientist harness](https://github.com/BurnyCoder/auto-eureka-scientist-agent-harness), studying AdamW weight decay on the bundled [UCI digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). The fixed MLP comparison includes a decay ablation, logistic regression, a class-prior baseline, shuffled-label controls, and seed/sensitivity checks.

The recorded experiments are available, but the run exhausted its active-time budget before analysis, review and final publication were accepted. The generated results below state the observed status and scientific conclusion. Draft claims remain unaccepted; see the [post-run disclosure](reports/post-run-publication.md).

[Interactive explorer](https://burnycoder.github.io/eureka-digits-haiku/) · [Paper PDF](cycles/001/reports/paper.pdf) · [Model bundle](https://huggingface.co/BurnyCoder/eureka-digits-haiku-models) · [Model collection](https://huggingface.co/collections/BurnyCoder/eureka-digits-haiku-6ae38bdd8bdaca1527379a69)

![Interactive filters and recorded accuracies](docs/images/explorer.png)


The [registered protocol](cycles/001/research/protocol.json) fixes the data partition, architecture, optimizer and comparisons before results. The pilot checkpoint was published before sensitivity runs. Saved predictions drive every generated numerical table and figure. PyTorch uses CUDA for neural training when available; scikit-learn supplies the dataset and linear baseline. No test outcome selects a configuration.

Follow the [portable reproduction guide](https://github.com/BurnyCoder/auto-eureka-scientist-agent-harness/blob/main/docs/reproduction.md) to install locked dependencies, download checkpoints, regenerate reports and verify predictions without another model call. The cycle's [original environment and methodology notes](cycles/001/docs/reproduction.md) retain their historical core-phase context; the post-run disclosure corrects their overly strong pilot data-access wording. Elapsed seconds in generated tables include checkpoint reuse and are not a controlled training-speed comparison.

The workflow follows [Neel Nanda's research stages](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL). Its source ledger, protocol history, observations and failures remain available under `cycles/001/`. This particular development trial does not establish broad scientific reliability or agent superiority.


<!-- EUREKA:RESULTS:START -->
## Recorded results

Recorded harness status: **running**; 7 operational phases accepted. Measurements below do not imply completion of unaccepted phases.

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
