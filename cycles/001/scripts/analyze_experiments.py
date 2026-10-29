"""Registered analysis of the test-split experiments, computed only from results/*.json.

Primary metric (research/protocol.json): mean over seeds of acc(decay 0.01) - acc(decay 0),
with a paired conditional bootstrap over test-example indices (10000 resamples, RNG seed 2026).
Sources: numpy.random.Generator.integers https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
McNemar exact test (binomial on discordant pairs), scipy.stats.binomtest https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
Output: research/experiment_summary.json, printed table.
"""
from pathlib import Path
import json

import numpy as np
from scipy.stats import binomtest

root = Path(__file__).resolve().parents[1]
protocol = json.loads((root/'research/protocol.json').read_text(encoding='utf-8'))
seeds = protocol['seeds']


def record(method, seed):
    """Load one saved experiment record."""
    return json.loads((root/'results'/f'{method}_s{seed}.json').read_text(encoding='utf-8'))


def correct(rec):
    """Boolean correctness vector over the test split, in the stored example order."""
    return np.asarray(rec['y_true']) == np.asarray(rec['y_pred'])


# All records share one test split; verify before pairing by position.
reference_ids = record('regularized', 0)['example_ids']
for method in protocol['methods']:
    for seed in (seeds if method in protocol['weight_decay'] else [0]):
        assert record(method, seed)['example_ids'] == reference_ids, 'Test split order differs'

summary = {'per_seed': {}, 'arms': {}, 'paired': {}, 'history': {}}
for method in protocol['methods']:
    for seed in (seeds if method in protocol['weight_decay'] else [0]):
        rec = record(method, seed)
        summary['arms'][f'{method}_s{seed}'] = {'accuracy': rec['accuracy'], 'n_test': len(rec['y_true']),
                                                 'checkpoint_sha256': rec['checkpoint_sha256'],
                                                 'analysis_kind': rec['analysis_kind']}

# Primary paired comparison, per seed and pooled with a bootstrap over test examples.
reg = {s: correct(record('regularized', s)) for s in seeds}
unreg = {s: correct(record('unregularized', s)) for s in seeds}
deltas = {s: float(reg[s].mean() - unreg[s].mean()) for s in seeds}
for s in seeds:
    b = int((reg[s] & ~unreg[s]).sum())   # regularized right, unregularized wrong
    c = int((~reg[s] & unreg[s]).sum())   # regularized wrong, unregularized right
    summary['per_seed'][f'seed{s}'] = {'acc_regularized': float(reg[s].mean()), 'acc_unregularized': float(unreg[s].mean()),
                                        'delta_pp': 100*deltas[s], 'discordant_reg_only': b, 'discordant_unreg_only': c,
                                        'mcnemar_exact_p': float(binomtest(b, b+c, 0.5).pvalue) if b+c else 1.0}

n = len(reference_ids)
rng = np.random.default_rng(protocol['bootstrap_seed'])
idx = rng.integers(0, n, size=(protocol['bootstrap_resamples'], n))
boot = np.mean([reg[s][idx].mean(axis=1) - unreg[s][idx].mean(axis=1) for s in seeds], axis=0)
mean_delta = float(np.mean(list(deltas.values())))
lo, hi = np.percentile(boot, [2.5, 97.5])
summary['paired']['primary'] = {'mean_delta_pp': 100*mean_delta, 'ci95_pp': [100*lo, 100*hi],
                                 'seed_sd_delta_pp': 100*float(np.std(list(deltas.values()), ddof=1)),
                                 'identical_predictions_all_seeds': all(np.array_equal(record('regularized', s)['y_pred'], record('unregularized', s)['y_pred']) for s in seeds)}

# Sensitivity arm (decay 0.001) against unregularized, same paired procedure, descriptive only.
sens = {s: correct(record('sensitivity', s)) for s in seeds}
summary['paired']['sensitivity_vs_unregularized_pp'] = 100*float(np.mean([sens[s].mean()-unreg[s].mean() for s in seeds]))

# Accuracy summaries for every arm (mean over seeds, sample SD where n>1).
def arm_stats(method):
    accs = [record(method, s)['accuracy'] for s in (seeds if method in protocol['weight_decay'] else [0])]
    sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else None
    return {'mean_acc': float(np.mean(accs)), 'sd_pp': None if sd is None else 100*sd, 'per_seed': accs}

summary['arms'] = {m: arm_stats(m) for m in protocol['methods']}

# Training histories: P8 compares regularized vs unregularized loss trajectories per seed.
for s in seeds:
    h_reg = record('regularized', s)['history']
    h_unreg = record('unregularized', s)['history']
    summary['history'][f'seed{s}'] = {
        'max_abs_train_loss_diff': max(abs(a['train_loss']-b['train_loss']) for a, b in zip(h_reg, h_unreg)),
        'max_abs_validation_loss_diff': max(abs(a['validation_loss']-b['validation_loss']) for a, b in zip(h_reg, h_unreg)),
        'final_train_loss_reg_minus_unreg': h_reg[-1]['train_loss']-h_unreg[-1]['train_loss'],
        'final_validation_loss_reg_minus_unreg': h_reg[-1]['validation_loss']-h_unreg[-1]['validation_loss'],
    }

# Shuffled-label control: test accuracy against true labels vs chance (1/10), binomial z-score.
for s in seeds:
    acc = record('shuffled_labels', s)['accuracy']
    summary.setdefault('shuffled_z', {})[f'seed{s}'] = (acc-0.1)/np.sqrt(0.1*0.9/n)

out = root/'research/experiment_summary.json'
out.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
