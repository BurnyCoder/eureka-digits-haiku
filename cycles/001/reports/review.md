# Review: cycle 001 (AdamW weight decay on a fixed digits MLP)

Phase: review. Scope: one substantive pass over the science, code, citations, reproducibility, and presentation (paper, site, and reproduction doc). No training was run, and no new experiment or test-split evaluation was made. Interpretation of results is unchanged from the analysis phase.

## What was checked

| Check | Result |
|---|---|
| `study.py check` (cycle directory) | passed: 1797 rows, split 1078 / 359 / 360 |
| `pytest tests` (13 tests) | 13 passed (17 s) |
| Records vs stored predictions (`results/*.json`, 15 records) | accuracy recomputed from `y_pred`/`y_true` matches every record; `regularized_s{0,1,2}` and `unregularized_s{0,1,2}` have identical `y_pred` on all 360 test rows |
| E3 (loss trajectories differ) | recomputed from `history`: validation-loss difference at last epoch is +0.00028, +0.00035, +0.00042 (seeds 0-2); nonzero at every epoch with max |diff| about 0.0008 |
| E5 (weight shrink) | recomputed from `research/decay_mechanism.json`: norm ratio 0.99594 / 0.99598 / 0.99594 vs predicted 0.99154; measured shrink is about half the prediction |
| E6 (MDE) | recomputed from `research/mde_audit.json`: 3% discordance gives 2.56 pp (z_sum 2.80 times SE sqrt(10.8)/360); consistent with the claim |
| E7 (post hoc NLL) | recomputed from `research/exploratory_test_nll.json`: per-seed deltas +0.00040, +0.00048, +0.00057, all positive; reported CI [+0.00024, +0.00070] |
| C2 / P7 seed SD | test accuracies 0.963889, 0.969444, 0.969444 give sample SD 0.32 pp, as stated |
| Paper figures (`cycles/001/figures/*.png`) | viewed directly: accuracy dot plot (axis from 0, pilot point in a distinct color and labelled), loss curves (legend does not cover data), examples grid (labelled prediction/true), architecture diagram (see residual limits) |
| Site (`site/index.html`) | inline script parses (node 24 `new Function`); `site/data.json` has the keys and shapes the page reads (claims with array evidence, limitations, 15 records, 18 sources) |
| `site/paper.md` vs `reports/paper.md` | differ only in image paths (`figures/` vs `../figures/`) |
| `site/paper.pdf` vs `reports/paper.pdf` | byte-identical |
| Citation S08 (Krogh & Hertz, NIPS 1991) | fetched from papers.nips.cc: title, authors, and volume 4 confirmed |
| Citation S04 (PyTorch decoupled decay) | fetched upstream `torch/optim/adam.py`: contains `param.mul_(1 - lr * weight_decay)` under `decoupled_weight_decay` (single-tensor and multi-tensor paths). The installed line is `adam.py:419`. The upstream adamw.py delegation was not confirmed in that fetch. |

## Findings and fixes

1. **Wrong citation URL for S04 (fixed).** `research/sources.json` pointed the S04 link at `torch/optim/adamw.py`, but the decay branch is in `adam.py`, and the text cites `adam.py`. Changed the URL to `.../torch/optim/adam.py`.
2. **Pilot described as reading validation only (fixed).** `docs/reproduction.md` said "The pilot reads validation only." In `study.py`, `fit_mlp` and the reuse path forward-pass the full feature matrix, which includes the 360 test rows, and reload-verify on it. The labels are not used, and the pilot reports validation metrics only. The doc and E9 (`research/claims.json`, `reports/interpretation.md`) now say this precisely.
3. **Stale reproduction doc (fixed).** The status header still said "no pilot, no training, and no test-split evaluation has been run yet," and the commands were labelled "later phase." Both are updated to describe the completed pilot and experiments.
4. **Line-number inconsistency (documented, not edited).** `reports/preregistration.md` cites `torch/optim/adam.py:417`; `research/sources.json`, `docs/reproduction.md`, and the installed file give line 419, which I confirmed. The preregistration is a pre-result document and is kept as written. The discrepancy is recorded here; the code is the same.

## Residual issues (not fixed in this phase)

- **Paper and site are generated and still show the pre-review text.** `reports/paper.md`, `reports/paper.pdf`, `site/*`, and `reports/summary.md` were produced before these source edits. They will reflect the S04 URL and the E9 wording only after the harness `render` runs at publication. I did not run `render` myself: it also rewrites the root README, the model card folder, and evidence snapshots, and the publish step owns that.
- **Generator-level defects in the paper template (not fixable inside the cycle).**
  - The rendered "Discussion, implications, and learnings" heading is empty and is followed directly by the interpretation heading.
  - The status line says "5 operational phases accepted." The run-status snapshot lists 5 completed phases (research, preregister, core, pilot, experiments); the analysis phase was accepted at attempt 18 but is not in that list. The count will lag until the controller updates the snapshot.
  - The AI statement is hard-coded: "Codex wrote the harness and study fixture." This comes from the harness template, not from this run. The CLAUDE.md asks for a statement that Claude Code wrote the codebase and paper, so the attribution sentence needs the owner's decision.
- **Architecture figure does not show the model.** `figures/architecture.png` shows the harness pipeline (task, CLI, evidence checks, reports). It does not show the 64 → 128 → 10 ReLU MLP. The CLAUDE.md asks for architecture visualization; a model diagram is still missing.
- **Site not rendered in a browser.** I checked script syntax and the data shape only. I did not open `index.html` in a browser, so layout, filter behavior, and canvas drawing are unverified. The page also `fetch`es `data.json`, so it works only when served over HTTP (for example GitHub Pages), not from `file://`.
- **PDF not read page by page.** I checked that the PDF is byte-identical to the report copy; I did not inspect its pages for table overflow.
- **Limits of E9.** The leakage audit was done by reading code, not by a runtime test. The fixture was left unchanged, as required.
- **Citation depth.** S09, S10, and S17/S18 were read from abstracts or search summaries, as the paper states. S08 and S04 were checked against the source in this phase; the other sources were not re-fetched here.

## Predictions, alternatives, and what would change them

- Prediction: after `render`, the paper's S04 link resolves to `adam.py` and the E9 text names 1797 rows. Confirmed by grep of the regenerated files at publication.
- Alternative for the zero-difference result: a bug that makes the two arms share predictions. Ruled out by the matching checkpoint hashes, the non-zero loss trajectories, and the from-scratch rerun in the analysis phase.
- Alternative for the E5 gap (measured shrink about half the prediction): gradient-driven norm differences between arms. Not resolved here; it needs a per-step weight-norm log in a future cycle.

## Learning

Re-deriving the numbers from raw records was cheap and caught the one factual error that mattered for the methods section: the pilot does forward-pass test features, which the docs had said it did not. The visual check of figures found no graph-level problems. The remaining defects sit in the generator template and in the missing model diagram, so they belong to the publish step or to a later figure task.

## Decision

Continue to publish. The publish step should run `render`, which regenerates the paper and site from the corrected sources, and the owner should decide the AI-statement attribution before release.
