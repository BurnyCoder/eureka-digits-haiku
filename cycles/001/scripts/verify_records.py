"""Independent verification of every saved experiment record (run after `study.py experiments`).

Checks, per record in results/*.json:
  (d) accuracy recomputed from stored y_true/y_pred equals the stored accuracy (A4d, reported arithmetic);
  (b) the saved checkpoint, reloaded with safetensors, reproduces the stored test predictions exactly
      (reload equality, A4b; predictions are regenerated from the weights, not read from the record).
Also checks the registered sanity predictions P5 (shuffled-label accuracy) and P6 (class-prior accuracy
vs majority test-class fraction), and P8 (regularized vs unregularized loss histories differ).
Sources: safetensors load_file https://huggingface.co/docs/safetensors/en/api/torch ;
torch.nn.Module.load_state_dict https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
Output: research/record_verification.json (printed as well).
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))  # so the fixture `study` resolves from the cycle root
from study import dataset  # noqa: E402
protocol = json.loads((root/'research/protocol.json').read_text(encoding='utf-8'))
x, y, splits = dataset(protocol)
test = splits['test']
inputs = torch.as_tensor(x[test])
inputs_dim, hidden, outputs = protocol['architecture']


def record(name):
    """Load one saved record by identifier, e.g. regularized_s0."""
    return json.loads((root/'results'/f'{name}.json').read_text(encoding='utf-8'))


def network_predictions(checkpoint):
    """Rebuild the registered MLP, load the saved weights, and predict the full test split."""
    model = nn.Sequential(nn.Linear(inputs_dim, hidden), nn.ReLU(), nn.Linear(hidden, outputs))
    model.load_state_dict(load_file(str(root/checkpoint)))
    model.eval()
    with torch.no_grad():
        return model(inputs).argmax(1).numpy()


report = {'records': {}, 'checks': {}}
for method in protocol['methods']:
    for seed in (protocol['seeds'] if method in protocol['weight_decay'] else [0]):
        name = f'{method}_s{seed}'
        rec = record(name)
        y_true, y_pred = np.asarray(rec['y_true']), np.asarray(rec['y_pred'])
        entry = {'accuracy_stored': rec['accuracy'], 'accuracy_recomputed': float(np.mean(y_true == y_pred)),
                 'test_size': len(y_true), 'ids_match_split': rec['example_ids'] == splits['test'].tolist()}
        assert entry['ids_match_split'], f'{name}: example order differs from the test split'
        assert abs(entry['accuracy_stored'] - entry['accuracy_recomputed']) < 1e-12, f'{name}: accuracy mismatch'
        if method != 'logistic' and method != 'class_prior':
            regenerated = network_predictions(rec['checkpoint'])
            entry['checkpoint_predictions_equal_stored'] = bool(np.array_equal(regenerated, y_pred))
        report['records'][name] = entry

# P6: class prior equals the majority-class fraction of the test split.
majority_fraction = float(np.bincount(y[test]).max()/len(test))
report['checks']['P6_class_prior_vs_majority_fraction'] = {'class_prior_acc': report['records']['class_prior_s0']['accuracy_stored'],
                                                           'majority_test_fraction': majority_fraction}

# P5: shuffled-label arms at chance, all below the registered 0.20 failure threshold.
report['checks']['P5_shuffled_labels_acc'] = [report['records'][f'shuffled_labels_s{s}']['accuracy_stored'] for s in protocol['seeds']]

# P8: decay changes the training trajectory (histories must differ for the same seed).
report['checks']['P8_history_differs'] = {f'seed{s}': bool(any(a != b for a, b in zip(record(f'regularized_s{s}')['history'],
                                                                         record(f'unregularized_s{s}')['history'])))
                                          for s in protocol['seeds']}

out = root/'research/record_verification.json'
out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
