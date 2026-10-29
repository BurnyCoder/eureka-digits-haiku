---
title: "eureka-digits-haiku: reproducible research report"
author: 'Libor “Burny” Burian'
geometry: margin=1in
fontsize: 10pt
---

# Abstract

This report records a bounded research cycle executed through Claude Code. The mean paired accuracy difference is 0.000000, with conditional 95% bootstrap interval [0.000000, 0.000000]. The registered decision is **inconclusive**. The sample standard deviation across seed effects is 0.000000. The interval uses 10000 paired resamples of test examples and conditions on the trained models.

Recorded harness status: **budget_exhausted**; 5 operational phases accepted. Measurements below do not imply completion of unaccepted phases.

Post-run repairs or publication work are documented in the [audit disclosure](https://github.com/BurnyCoder/eureka-digits-haiku/blob/main/reports/post-run-publication.md). They are separate from the bounded model execution.

# Context, motivation, and introduction

Does AdamW weight decay improve held-out accuracy of a small MLP on the bundled digits data?

The objective is reproducible evidence and an inspectable research process. This study does not establish broad scientific reliability or superiority of an agent model.

# Background and method

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

![Research architecture](../figures/architecture.png)

*Research architecture.*

# Results

| Method | Seed | Split | Accuracy | Seconds | Device |
|-------------------------|-----:|------------|-----------:|----------:|----------|
| class prior | 0 | test | 0.102778 | 0.026 | cpu |
| logistic | 0 | test | 0.955556 | 0.107 | cpu |
| regularized | 0 | validation | 0.966574 | 6.805 | cuda |
| regularized | 0 | test | 0.963889 | 0.486 | cuda |
| regularized | 1 | test | 0.969444 | 6.630 | cuda |
| regularized | 2 | test | 0.969444 | 2.553 | cuda |
| sensitivity | 0 | test | 0.963889 | 2.525 | cuda |
| sensitivity | 1 | test | 0.969444 | 2.583 | cuda |
| sensitivity | 2 | test | 0.969444 | 2.481 | cuda |
| shuffled labels | 0 | test | 0.061111 | 2.465 | cuda |
| shuffled labels | 1 | test | 0.116667 | 2.470 | cuda |
| shuffled labels | 2 | test | 0.108333 | 2.367 | cuda |
| unregularized | 0 | test | 0.963889 | 2.426 | cuda |
| unregularized | 1 | test | 0.969444 | 2.393 | cuda |
| unregularized | 2 | test | 0.969444 | 2.501 | cuda |

The mean paired accuracy difference is 0.000000, with conditional 95% bootstrap interval [0.000000, 0.000000]. The registered decision is **inconclusive**. The sample standard deviation across seed effects is 0.000000. The interval uses 10000 paired resamples of test examples and conditions on the trained models.

![All recorded accuracies](../figures/accuracy.png)

*All recorded accuracies.*

![Recorded learning curves](../figures/loss.png)

*Recorded learning curves.*

![First saved test examples; no correctness-based selection](../figures/examples.png)

*First saved test examples; no correctness-based selection.*

# Claims and evidence

Claims have not yet been distilled.

# Discussion, implications, and learnings

The analysis phase was not accepted before this run stopped. Its draft interpretation and claim ledger are preserved in the repository, but are excluded from this generated paper because their evidence contracts remain unresolved. The tables and primary calculation above are regenerated from saved predictions; review remains incomplete.

# Limitations

- Conditional test-example intervals do not estimate variation over new training seeds, writers, or datasets.
- A small single-task study does not establish general model or harness superiority.
- No discordant predictions were observed in the primary pairs. The empirical bootstrap interval collapses to zero; this is not an equivalence test or a guarantee about unseen examples.

Exploratory outcomes do not become confirmatory by inclusion in this document. Missing metrics are not zero. Measurements without an applicable recorded objective are omitted; see the requirement checklist for explanations.

# Related work and sources

- [S01: Decoupled Weight Decay Regularization (AdamW), Loshchilov & Hutter, ICLR 2019](https://arxiv.org/abs/1711.05101)
- [S02: Adam: A Method for Stochastic Optimization, Kingma & Ba, ICLR 2015](https://arxiv.org/abs/1412.6980)
- [S03: torch.optim.AdamW documentation (PyTorch 2.15)](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)
- [S04: PyTorch adam.py: decoupled weight decay update (torch 2.14.1 installed locally, plus upstream main)](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adamw.py)
- [S05: sklearn.datasets.load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [S06: Optical Recognition of Handwritten Digits (UCI ML Repository), Alpaydin & Kaynak 1998](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [S07: Understanding deep learning requires rethinking generalization, Zhang et al., ICLR 2017](https://arxiv.org/abs/1611.03530)
- [S08: A Simple Weight Decay Can Improve Generalization, Krogh & Hertz, NIPS 1991 (vol. 4)](https://papers.nips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html)
- [S09: Understanding Decoupled and Early Weight Decay, Bjorck et al., 2020](https://consensus.app/papers/details/cfe1ea949a105c5f953b251b2af3701e/)
- [S10: Towards Understanding Convergence and Generalization of AdamW, Zhou et al., IEEE TPAMI 2024](https://consensus.app/papers/details/c179f549f272597888e9890895eef7f1/)
- [S11: Troubling Trends in Machine Learning Scholarship, Lipton & Steinhardt, 2018](https://arxiv.org/abs/1807.03341)
- [S12: How to avoid machine learning pitfalls: a guide for academic researchers, Lones](https://arxiv.org/abs/2108.02497)
- [S13: How I Think About My Research Process (sequence), Neel Nanda](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL)
- [S14: How I Think About My Research Process: Explore, Understand, Distill, Neel Nanda, 26 Apr 2025](https://www.greaterwrong.com/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand)
- [S15: Construction of Confidence Intervals, Dalitz, arXiv:1807.03582](https://arxiv.org/abs/1807.03582)
- [S16: safetensors torch API documentation (save_file, load_file)](https://huggingface.co/docs/safetensors/en/api/torch)
- [S17: scipy.stats.bootstrap documentation (percentile method, paired option)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [S18: oceCens percci documentation (percentile interval, citing Efron & Tibshirani 1993, p. 168)](https://search.r-project.org/CRAN/refmans/oceCens/html/percci.html)

# Conclusion

The mean paired accuracy difference is 0.000000, with conditional 95% bootstrap interval [0.000000, 0.000000]. The registered decision is **inconclusive**. The sample standard deviation across seed effects is 0.000000. The interval uses 10000 paired resamples of test examples and conditions on the trained models.

# Appendix: reproducibility and provenance

The committed protocol, per-experiment JSON files, saved predictions, loss histories, and checkpoints are the source of all numerical tables and figures. Rebuild using `eureka render`. See `docs/reproduction.md` for execution commands and environment details.

# AI statement

Codex wrote the harness and study fixture; Claude Code performed this cycle and authored its interpretation. The harness generated this paper from recorded artifacts. A human supplied the exact specification. Human cleanup and rewriting are in progress.
