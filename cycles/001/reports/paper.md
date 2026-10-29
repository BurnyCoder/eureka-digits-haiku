---
title: "eureka-digits-haiku: reproducible research report"
author: 'Libor “Burny” Burian'
geometry: margin=1in
fontsize: 10pt
---

# Abstract

This report records a bounded research cycle executed through Claude Code. The registered primary comparison is pending; no confirmatory conclusion is available.

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

# Results

| Experiment | Method | Seed | Split | Accuracy | Seconds | Device |
|---|---|---:|---|---:|---:|---|
| pilot_regularized_s0 | regularized | 0 | validation | 0.966574 | 6.805 | cuda |

The registered primary comparison is pending; no confirmatory conclusion is available.

![All recorded accuracies](../figures/accuracy.png)

![Recorded learning curves](../figures/loss.png)

# Claims and evidence

Claims have not yet been distilled.

# Discussion, implications, and learnings

This is a preliminary artifact generated before final interpretation. All available outcomes are retained; a completed experiment is not evidence that the favored hypothesis is true.

# Limitations

- Conditional test-example intervals do not estimate variation over new training seeds, writers, or datasets.
- A small single-task study does not establish general model or harness superiority.

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

# Conclusion

The registered primary comparison is pending; no confirmatory conclusion is available.

# Appendix: reproducibility and provenance

The committed protocol, per-experiment JSON files, saved predictions, loss histories, and checkpoints are the source of all numerical tables and figures. Rebuild using `eureka render`. See `docs/reproduction.md` for execution commands and environment details.

# AI statement

Codex wrote the harness and supplied study fixture. Claude Code executed this cycle's research phases and authored its interpretation; the harness generated the paper from recorded artifacts. A human provided the exact specification that was executed. Human cleanup and rewriting are in progress.
