# Reproducing cycle 001: AdamW weight decay on a fixed digits MLP

Status: the pilot and the registered experiments have been run (see [`reports/experiments.md`](../reports/experiments.md) and [`reports/pilot.md`](../reports/pilot.md)). The commands below reproduce them; the results are in `results/` and `models/`. Phase-by-phase history is in `reports/`, and the registered definitions are in `reports/preregistration.md`.

Scientific definitions live in [`reports/preregistration.md`](../reports/preregistration.md) and the immutable [`research/protocol.json`](../research/protocol.json) (SHA-256 recorded in [`research/registration.json`](../research/registration.json)). This document only explains how to run things.

## Environment

Versions observed in the core phase (Windows 11, Python from the project `.venv`):

| Package | Version |
|---|---|
| Python | 3.14.2 |
| torch | 2.14.1+cu130 (CUDA available on this machine) |
| scikit-learn | 1.9.1 |
| numpy | 2.5.4 |
| safetensors | installed with the project venv |
| pytest | 9.2.0 (only for `tests/`) |

Dependencies are declared in [`../../pyproject.toml`](../../pyproject.toml) (`numpy`, `scipy`, `scikit-learn`, `torch`, `safetensors`, `matplotlib`). The harness-supplied interpreter is `C:\Users\Happy\Documents\projects\auto-eureka-scientist-agent-harness\.venv\Scripts\python.exe`; the commands below assume it. The dataset is bundled with scikit-learn ([`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)), so no download is needed.

## Commands

Run from `cycles/001/`.

```bash
PY=C:/Users/Happy/Documents/projects/auto-eureka-scientist-agent-harness/.venv/Scripts/python.exe

$PY study.py check                 # fixture sanity: shapes, value range, split sizes (no training)
$PY -m pytest tests -q             # core unit checks (~30 s CPU, writes only to pytest tmp dirs)
$PY study.py pilot                 # regularized seed 0, validation metrics (reuses results/ if the hashes match)
$PY study.py experiments           # all registered test-split arms (needs research/pilot-publication.json)
```

`study.py` prints one timestamped JSON line per epoch and per saved experiment to stdout. It does not write a log file; redirect stdout (for example `... > logs/run.log`) to keep the record.

Outputs: `results/<id>.json` (one record per arm and seed, including `y_true`, `y_pred`, per-epoch `history`, split IDs, and SHA-256 of the protocol, code, dataset, and checkpoint), `models/*.safetensors` for the MLP arms, and `models/*.npz` for the two non-neural references. Existing result files are reused only if their protocol and code hashes still match; otherwise `study.py` stops.

## Data and splits

- 1797 images, 8x8 grayscale, 10 classes ([`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)). Pixels are divided by the constant 16; nothing is learned from data during preprocessing.
- Stratified 60/20/20 split with `train_test_split(..., stratify=y, random_state=42)` ([sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)): train 1078 / validation 359 / test 360.
- Labels of the test split are used only in the registered comparisons (`study.py experiments`). Caveat: the fixture's reload check and prediction step run the model on all 1797 feature rows, including the 360 test features, in the pilot and in every arm. The pilot reports only validation metrics, but its code path does forward-pass test features. This is disclosed in E9 and `reports/pilot.md`; it is not a label leak, and it was not changed because `study.py` is the immutable fixture.
- Limit: one random example split with no writer identity, so results do not measure generalization to new writers or datasets.

## Model and training (registered)

- MLP 64 → 128 → 10 with ReLU (`study.network`); 9610 parameters.
- Loss: cross-entropy. Optimizer: [`torch.optim.AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html), learning rate 1e-3, batch size 64, 50 epochs, no early stopping, no learning-rate schedule.
- Decoupled weight decay: AdamW multiplies parameters by `(1 - lr * weight_decay)` outside the gradient update (installed torch: `torch/optim/adam.py:419`). Loshchilov & Hutter introduce this decoupling in [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
- Seeds 0, 1, 2 set Python, NumPy, and torch RNGs, and seed the epoch shuffle generator. `torch.use_deterministic_algorithms(True)` is enabled.
- Checkpoints are saved with [safetensors](https://huggingface.co/docs/safetensors/en/api/torch) and reloaded; `torch.testing.assert_close(..., rtol=0, atol=0)` must pass before predictions are used.

## Arms: treatment, ablation, controls, baselines

| Arm (`method`) | Model | weight_decay | Seeds | Role |
|---|---|---|---|---|
| `regularized` | MLP | 0.01 | 0, 1, 2 | **Treatment** (the hypothesis: decay improves test accuracy) |
| `unregularized` | MLP | 0.0 | 0, 1, 2 | **Ablation** of decay; paired comparator for the primary metric |
| `sensitivity` | MLP | 0.001 | 0, 1, 2 | Descriptive local sensitivity check (not selected on results) |
| `shuffled_labels` | MLP | 0.01 | 0, 1, 2 | **Negative control**: labels permuted within the training split with `np.random.default_rng(seed)`; test accuracy should be near chance |
| `logistic` | `LogisticRegression(C=1.0, max_iter=1000)` | — | 0 | **Baseline**: linear model, untuned reference |
| `class_prior` | majority training class | — | 0 | **Floor baseline**: predicts the most frequent training label |

Primary metric: mean over seeds of `acc(weight_decay=0.01) − acc(weight_decay=0)` on the test split, with a paired bootstrap over test indices (10000 resamples, RNG seed 2026). Pairing uses the same seed, initialization, and batch order in both arms. The registered falsification and decision rules are in the preregistration §6.

Not included, and named as such: the comparison of decoupled AdamW against L2-regularized Adam (the comparison in [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) and any λ = 0 shuffled-label control.

## Core-phase verification

`research/checks.json` records the core check, `study.py check`, which the controller reruns. Last result observed in this phase:

```
{"status": "passed", "dataset_shape": [1797, 64], "split_sizes": {"train": 1078, "validation": 359, "test": 360}}
```

`tests/test_study.py` (13 tests, all passing) checks:

- split sizes, disjointness, completeness, and stratification (each class count within 1 of its proportional share);
- pixel range and the exact `/16` scaling;
- the registered architecture and parameter count;
- that one AdamW step with zero gradient multiplies weights by `1 - lr*wd` (decoupled decay), and leaves them unchanged when `wd = 0`;
- seed reproducibility of a short training run and checkpoint reload;
- that the regularized and unregularized arms produce different loss trajectories (a code check for registered prediction P8);
- that the logistic and class-prior records recompute their stored accuracy from the saved predictions (registered check A4d);
- that the class-prior baseline predicts only the majority training class;
- that `experiment` refuses a protocol whose hash does not match the registration.

Gap: the shuffled-label permutation is not tested directly, because it is inside `fit_mlp`. It is covered only by the registered prediction P5 (shuffled test accuracy ≈ chance), which is checked in the experiments phase.

The tests use a temporary root and a reduced epoch count (2). They do not write into `results/` or `models/`, and they do not alter the registered protocol.

## Sources

- [S05] scikit-learn `load_digits`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- [S03] PyTorch `torch.optim.AdamW`: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- [S04] PyTorch decoupled update, `torch/optim/adam.py`: https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adam.py
- [S01] Loshchilov & Hutter, AdamW (decoupled weight decay), ICLR 2019: https://arxiv.org/abs/1711.05101
- [S16] safetensors torch API: https://huggingface.co/docs/safetensors/en/api/torch
- Study framing: [S11] Lipton & Steinhardt, https://arxiv.org/abs/1807.03341; [S12] Lones, https://arxiv.org/abs/2108.02497; [S13] Neel Nanda, https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL
