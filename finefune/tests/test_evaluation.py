"""Exercise prediction records -> existing AP -> full report, without a PPU."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from hot3d_ft import evaluation
from hot3d_ft.common import checksum, data_path, digest, jsonlines, write
from test_data import fixture_dataset


def test_text_evaluation_and_report_contract(tmp_path, monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl'))
    cfg, rows, protocol = fixture_dataset(tmp_path)
    root = data_path(cfg)
    for index, frame in enumerate(rows):
        image_path = root/f'image-{index}.png'
        Image.fromarray(np.full((768, 768, 3), index*100, dtype=np.uint8)).save(image_path)
        frame.update(rgb_path=str(image_path), rgb_sha256=checksum(image_path))
    protocol.update(train_sha256=digest(rows[:1]), eval_sha256=digest(rows[1:]))
    jsonlines(root/'train.jsonl', rows[:1]); jsonlines(root/'eval.jsonl', rows[1:])
    write(root/'protocol.json', protocol)
    write(root/'dataset_audit.json', dict(train=dict(frames=1), eval=dict(frames=1)))
    calls = []

    class Adapter:
        def predict(self, sample):
            assert len(sample.prompts) == 25
            assert not hasattr(sample, 'gt_path')
            calls.append(sample.frame_id)
            return []

        def close(self):
            pass

    monkeypatch.setattr(evaluation, 'load_adapter', lambda *args:
                        (Adapter(), torch.nn.Linear(1, 1), dict(checkpoint_sha256='fake-test-only')))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU-test')
    for key in cfg['models']:
        for variant in ('baseline', 'finetuned'):
            result = evaluation.evaluate(cfg, key, variant)
            assert result['complete'] and result['frames'] == 1
            assert result['geometry']['summary']['AP3D@0.25'] == 0.
        journal = Path(cfg['root'])/'runs'/key/'train.jsonl'
        history = [dict(update=i, lr=.0001, seconds=.1, peak_allocated_bytes=0,
                        loss_delta_2d=1., loss_depth=1., loss_dim=1., loss_rot=1.) for i in range(1, 5)]
        jsonlines(journal, history)
        write(journal.parent/'wandb/attempt-0001/status.json',
              dict(stage='complete', mode='online', url=f'https://wandb.ai/test/{key}'))
    result = evaluation.report(cfg)
    assert Path(result['report']).is_file()
    assert len(list((Path(cfg['root'])/'report/visualizations').rglob('*.png'))) == 4
    assert (Path(cfg['root'])/'report/wilddet3d-training.png').is_file()
    assert 'https://wandb.ai/test/3dmood' in Path(result['report']).read_text()
    assert len(calls) == 4
