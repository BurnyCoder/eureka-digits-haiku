"""Minimum-detectable-effect audit for the paired primary comparison.

Global context: the research phase quoted a minimum detectable difference of about 2.5 pp
without showing its derivation. This script recomputes it from first principles so the
number is reproducible and its dependence on the unobserved discordance rate is explicit.

Method: for a paired comparison of two classifiers on the same n test examples, let b and c
be the counts where only one model is correct. The difference in accuracy is (b - c)/n. Treating
each discordant example as an independent sign flip, its variance is (b + c)/n^2 (derived here;
no external source was checked for this formula, so it is a normal approximation).
With discordance fraction d = (b + c)/n, SE = sqrt(d / n). Two-sided alpha 0.05 and 80% power
give MDE = (z_{0.975} + z_{0.80}) * SE, with z_{0.975} = 1.96 and z_{0.80} = 0.8416 (normal quantiles).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # cycles/001
N_TEST = 360  # test size from study.py check (research/checks.json)
Z = 1.959964 + 0.841621  # z_{0.975} + z_{0.80}
# Observed discordance on the registered comparison is zero (research/experiment_summary.json),
# so the MDE cannot be estimated from data. The table shows the range of assumed discordance.
rows = []
for d in (0.0, 0.01, 0.02, 0.03, 0.05, 0.10):
    se = math.sqrt(d / N_TEST)
    rows.append({"discordance_fraction": d, "discordant_examples": round(d * N_TEST, 2),
                 "se_pp": round(100 * se, 3), "mde_pp": round(100 * Z * se, 3)})
out = {"n_test": N_TEST, "z_sum": round(Z, 6), "observed_discordance": 0.0, "table": rows,
       "note": "MDE is 0 only because the observed discordance is 0; it is not evidence of equivalence."}
(ROOT / "research" / "mde_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
