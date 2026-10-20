import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from hot3d_ft import training
from hot3d_ft.common import config
from hot3d_ft.models import freeze, regression_losses


def test_losses_ignore_unmatched_predictions():
    prediction = torch.randn(4, 12, requires_grad=True)
    weights = torch.zeros_like(prediction); weights[1] = 1.
    losses = regression_losses(prediction, torch.zeros_like(prediction), weights, 1)
    sum(losses.values()).backward()
    assert prediction.grad[1].abs().sum() > 0
    assert prediction.grad[[0, 2, 3]].abs().sum() == 0


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.bbox3d_head = torch.nn.Linear(2, 1)


def test_freeze_keeps_detector_fixed():
    model = Tiny(); freeze(model)
    assert model.training and model.bbox3d_head.training
    assert not model.backbone.training
    assert not model.backbone.weight.requires_grad
    assert model.bbox3d_head.weight.requires_grad


def test_interruption_restores_optimizer_rng_and_exact_next_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'verify', lambda cfg: None)
    monkeypatch.setattr(training, 'load', lambda cfg: (
        dict(train_sha256='fixed', benchmark_id='fixed', classes=[]),
        [dict(frame_id=str(i), value=i) for i in range(5)], []))
    monkeypatch.setattr(training, 'training_image', lambda frame, epoch, seed: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU-test')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    class Adapter:
        def close(self):
            pass

    monkeypatch.setattr(training, 'load_adapter', lambda cfg, key, device:
                        (Adapter(), Tiny(), dict(checkpoint_sha256='fixed')))
    interruptions = {'at': None, 'calls': 0}

    class Trainer:
        def __init__(self, adapter, model, classes):
            self.model = model
            freeze(model)

        def __call__(self, frame, image):
            interruptions['calls'] += 1
            if interruptions['calls'] == interruptions['at']:
                # Consume random numbers before interruption to prove that
                # replay uses the last committed RNG state, not current RNG.
                torch.randn(20)
                raise RuntimeError('injected interruption')
            values = torch.tensor([float(frame['value']), 1.]) + torch.randn(2) * .01
            return {'loss': (self.model.bbox3d_head(self.model.backbone(values)) - .5).square().sum()}

    monkeypatch.setattr(training, 'MoodTrainer', Trainer)
    cfg = config(root=tmp_path/'reference'); cfg['device'] = 'cpu'
    cfg['training'].update(accumulate=2, checkpoint_updates=2, threads=1)
    training.train(cfg, '3dmood')
    reference = torch.load(Path(cfg['root'])/'runs/3dmood/checkpoints/final_head.pt', weights_only=False)
    cfg['root'] = str(tmp_path/'interrupted')
    interruptions.update(at=8, calls=0)
    with pytest.raises(RuntimeError, match='injected'):
        training.train(cfg, '3dmood')
    saved = torch.load(Path(cfg['root'])/'runs/3dmood/checkpoints/last.pt', weights_only=False)
    assert (saved['epoch'], saved['next_index'], saved['update']) == (1, 2, 4)
    interruptions.update(at=None, calls=0)
    training.train(cfg, '3dmood')
    resumed = torch.load(Path(cfg['root'])/'runs/3dmood/checkpoints/final_head.pt', weights_only=False)
    assert all(torch.equal(value, resumed['head'][key]) for key, value in reference['head'].items())
