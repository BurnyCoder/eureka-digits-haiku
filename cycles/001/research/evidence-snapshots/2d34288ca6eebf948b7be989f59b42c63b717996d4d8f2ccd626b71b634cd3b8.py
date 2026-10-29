"""Core checks for the registered digits study. Run from cycles/001: python -m pytest tests

Every test writes only into pytest's tmp_path, never into results/ or models/, so the
registered artifacts are untouched. Epoch counts are reduced for speed; the registered
protocol is never edited (its SHA-256 is asserted in test_registered_protocol_hash).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import study

ROOT = Path(study.__file__).resolve().parent
PROTOCOL = study.load(ROOT/'research/protocol.json')
CPU = torch.device('cpu')


@pytest.fixture(scope='module')
def data():
    return study.dataset(PROTOCOL)


def registered_root(tmp_path):
    """Copy the registered protocol into a temporary root with a matching registration record."""
    (tmp_path/'research').mkdir()
    (tmp_path/'results').mkdir()
    (tmp_path/'models').mkdir()
    protocol_path = tmp_path/'research/protocol.json'
    protocol_path.write_bytes((ROOT/'research/protocol.json').read_bytes())
    (tmp_path/'research/registration.json').write_text(json.dumps({'protocol_sha256': study.sha(protocol_path)}))
    return tmp_path


def test_registered_protocol_hash():
    # Matches research/registration.json, so the protocol has not been edited after registration.
    registration = study.load(ROOT/'research/registration.json')
    assert registration['protocol_sha256'] == study.sha(ROOT/'research/protocol.json')


def test_split_is_disjoint_complete_and_registered_sizes(data):
    _, y, splits = data
    assert {k: len(v) for k, v in splits.items()} == {'train': 1078, 'validation': 359, 'test': 360}
    assert np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(y)))


def test_split_is_stratified(data):
    # Each class count in a split should be within 1 of its proportional share of the split size.
    _, y, splits = data
    overall = np.bincount(y, minlength=10)/len(y)
    for name, idx in splits.items():
        expected = overall*len(idx)
        assert np.abs(np.bincount(y[idx], minlength=10) - expected).max() <= 1, name


def test_inputs_are_constant_scaled_pixels(data):
    # load_digits pixels are integers 0..16, so /16 maps them exactly to multiples of 1/16 in [0, 1].
    x, _, _ = data
    assert x.dtype == np.float32 and x.shape == (1797, 64)
    assert x.min() >= 0 and x.max() <= 1
    assert np.allclose(x*16, np.round(x*16))


def test_network_matches_registered_architecture():
    model = study.network(PROTOCOL)
    assert [tuple(p.shape) for p in model.parameters()] == [(128, 64), (128,), (10, 128), (10,)]
    assert sum(p.numel() for p in model.parameters()) == 9610
    assert isinstance(model[1], torch.nn.ReLU)


def test_adamw_decay_is_decoupled_from_gradients():
    # With zero gradient the Adam moment terms are zero, so one AdamW step must equal theta*(1 - lr*wd).
    # Implementation: torch/optim/adam.py:419 `param.mul_(1 - lr * weight_decay)` (decoupled branch).
    # Docs: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    param = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.01)
    param.grad = torch.zeros(3)
    optimizer.step()
    assert torch.allclose(param.detach(), torch.full((3,), 1 - 1e-5))


def test_zero_weight_decay_leaves_zero_grad_parameters_unchanged():
    # The unregularized arm (weight_decay=0) must not shrink weights.
    param = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
    param.grad = torch.zeros(3)
    optimizer.step()
    assert torch.equal(param.detach(), torch.ones(3))


def test_fit_is_seed_reproducible_and_reloads_checkpoint(tmp_path, data):
    x, y, splits = data
    protocol = {**PROTOCOL, 'epochs': 2}
    root = registered_root(tmp_path)
    predicted_a, history_a, checkpoint = study.fit_mlp(root, protocol, 'regularized', 0, x, y, splits, CPU)
    predicted_b, history_b, _ = study.fit_mlp(root, protocol, 'regularized', 0, x, y, splits, CPU)
    assert checkpoint.exists() and len(history_a) == 2
    assert np.array_equal(predicted_a, predicted_b)
    assert history_a == history_b


def test_decay_changes_training_trajectory(tmp_path, data):
    # P8 code check: the regularized and unregularized arms share seed, init and batch order,
    # so any difference in loss history must come from the decay term.
    x, y, splits = data
    protocol = {**PROTOCOL, 'epochs': 2}
    root = registered_root(tmp_path)
    _, regularized, _ = study.fit_mlp(root, protocol, 'regularized', 0, x, y, splits, CPU)
    _, unregularized, _ = study.fit_mlp(root, protocol, 'unregularized', 0, x, y, splits, CPU)
    assert regularized != unregularized


@pytest.mark.parametrize('method', ['logistic', 'class_prior'])
def test_baseline_records_are_recomputable(tmp_path, data, method):
    # A4(d): accuracy stored in each record must equal the mean agreement of its saved y_true/y_pred.
    _, y, splits = data
    root = registered_root(tmp_path)
    record = study.experiment(root, PROTOCOL, method, 0, 'test', CPU)
    y_true, y_pred = np.array(record['y_true']), np.array(record['y_pred'])
    assert record['accuracy'] == pytest.approx(np.mean(y_true == y_pred))
    assert np.array_equal(record['example_ids'], splits['test'])
    assert np.array_equal(y_true, y[splits['test']])
    assert record['analysis_kind'] == 'exploratory'


def test_class_prior_predicts_majority_training_class(tmp_path, data):
    # Baseline definition: predict the most frequent training-split label for every example.
    _, y, splits = data
    root = registered_root(tmp_path)
    record = study.experiment(root, PROTOCOL, 'class_prior', 0, 'test', CPU)
    majority = np.bincount(y[splits['train']]).argmax()
    assert set(record['y_pred']) == {int(majority)}


def test_experiment_refuses_unregistered_protocol(tmp_path):
    root = registered_root(tmp_path)
    (root/'research/registration.json').write_text(json.dumps({'protocol_sha256': 'bogus'}))
    with pytest.raises(AssertionError, match='Unregistered protocol change'):
        study.experiment(root, PROTOCOL, 'class_prior', 0, 'test', CPU)
