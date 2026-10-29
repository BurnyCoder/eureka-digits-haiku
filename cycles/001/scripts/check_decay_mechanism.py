"""Mechanism check: did decay change the saved weights, and by how much?

Compares the saved safetensors checkpoints per seed (decay 0.01 and 0.001 against decay 0), and
compares the observed relative weight-norm shrink with the prediction in reports/highlights.md:
(1 - lr*lambda)^steps. Also reports the fraction of weight entries that changed.
Sources: safetensors load_file https://huggingface.co/docs/safetensors/en/api/torch
Decoupled decay update: torch/optim/adam.py `param.mul_(1 - lr * weight_decay)` (PyTorch AdamW docs
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
Output: research/decay_mechanism.json
"""
from pathlib import Path
import json

import numpy as np
from safetensors.numpy import load_file

root = Path(__file__).resolve().parents[1]
protocol = json.loads((root/'research/protocol.json').read_text(encoding='utf-8'))
steps = protocol['epochs'] * int(np.ceil(1078/protocol['batch_size']))  # 17 steps/epoch -> 850
out = {'steps': steps, 'predicted_shrink_factor': {}, 'checkpoints': {}}
for name, weight_decay in protocol['weight_decay'].items():
    out['predicted_shrink_factor'][name] = (1 - protocol['learning_rate']*weight_decay)**steps


def weights(path):
    """Flatten all weight and bias tensors of one checkpoint into one vector."""
    tensors = load_file(str(path))
    return np.concatenate([tensors[k].ravel() for k in sorted(tensors)])


for seed in protocol['seeds']:
    base = weights(root/f'models/unregularized-s{seed}.safetensors')
    for method in ('regularized', 'sensitivity'):
        other = weights(root/f'models/{method}-s{seed}.safetensors')
        ratio = float(np.linalg.norm(other)/np.linalg.norm(base))
        out['checkpoints'][f'{method}_s{seed}_vs_unregularized_s{seed}'] = {
            'norm_ratio': ratio, 'relative_shrink': 1-ratio,
            'fraction_entries_changed': float(np.mean(other != base)),
            'max_abs_difference': float(np.max(np.abs(other-base))),
        }
out_path = root/'research/decay_mechanism.json'
out_path.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
print(json.dumps(out, indent=2))
