"""EXPLORATORY (not registered): paired test cross-entropy between decay arms.

Purpose: the registered metric (accuracy) has zero discordant predictions, so it cannot show whether decay
changes the output distribution at all. Mean test negative log-likelihood is a finer, continuous signal.
This is post-hoc and does not change the registered primary conclusion (reports/experiments.md).
Uses the same paired seeds, the same test split and the same bootstrap RNG seed (2026) as the primary analysis.
Sources: torch.nn.functional.cross_entropy https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
Output: research/exploratory_test_nll.json
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))  # so `study` (the fixture) resolves from the cycle root
from study import dataset  # noqa: E402
protocol = json.loads((root/'research/protocol.json').read_text(encoding='utf-8'))
x, y, splits = dataset(protocol)
test = splits['test']
inputs = torch.as_tensor(x[test])
targets = torch.as_tensor(y[test])
inputs_dim, hidden, outputs = protocol['architecture']


def per_example_nll(method, seed):
    """Per-example cross-entropy of one saved checkpoint on the test split."""
    model = nn.Sequential(nn.Linear(inputs_dim, hidden), nn.ReLU(), nn.Linear(hidden, outputs))
    model.load_state_dict(load_file(str(root/f'models/{method}-s{seed}.safetensors')))
    model.eval()
    with torch.no_grad():
        return nn.functional.cross_entropy(model(inputs), targets, reduction='none').numpy()


def mean_top_probability(method, seed):
    """Mean max softmax probability on the test split: a confidence measure for the shrink explanation."""
    model = nn.Sequential(nn.Linear(inputs_dim, hidden), nn.ReLU(), nn.Linear(hidden, outputs))
    model.load_state_dict(load_file(str(root/f'models/{method}-s{seed}.safetensors')))
    model.eval()
    with torch.no_grad():
        return float(nn.functional.softmax(model(inputs), dim=1).max(dim=1).values.mean())


n = len(test)
rng = np.random.default_rng(protocol['bootstrap_seed'])
idx = rng.integers(0, n, size=(protocol['bootstrap_resamples'], n))
out = {'metric': 'mean test NLL, decay 0.01 minus decay 0 (negative = decay lowers loss)', 'per_seed': {}}
deltas = []
for seed in protocol['seeds']:
    reg = per_example_nll('regularized', seed)
    unreg = per_example_nll('unregularized', seed)
    d = reg - unreg
    deltas.append(d.mean())
    out['per_seed'][f'seed{seed}'] = {'nll_regularized': float(reg.mean()), 'nll_unregularized': float(unreg.mean()),
                                       'delta': float(d.mean()), 'paired_sd_per_example': float(d.std(ddof=1)),
                                       'mean_top_prob_regularized': mean_top_probability('regularized', seed),
                                       'mean_top_prob_unregularized': mean_top_probability('unregularized', seed)}
boot = np.mean([(per_example_nll('regularized', s)[idx]-per_example_nll('unregularized', s)[idx]).mean(axis=1)
                for s in protocol['seeds']], axis=0)
out['mean_delta'] = float(np.mean(deltas))
out['ci95_conditional'] = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
(root/'research/exploratory_test_nll.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
print(json.dumps(out, indent=2))
