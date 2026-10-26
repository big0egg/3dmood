import copy
from pathlib import Path

import numpy as np
import pytest

from hot3d_ft.common import (atomic_bytes, bind, config, data_path, digest,
                            inputs_only, jsonlines, write)
from hot3d_ft.dataset import balanced_select, verify
from hot3d_ft.raw import amodal_mask


def test_selection_is_reproducible_and_covers_assets_sequences():
    rows = [dict(frame_id=f'{sequence}:{i}', sequence_id=str(sequence), asset_ids=[i % 5],
                 class_ids=[i % 3]) for sequence in range(4) for i in range(20)]
    result = balanced_select(rows, 24, 123)
    assert result == balanced_select(list(reversed(rows)), 24, 123)
    assert len({r['frame_id'] for r in result}) == 24
    assert {r['sequence_id'] for r in result} == {'0', '1', '2', '3'}
    assert set().union(*(set(r['asset_ids']) for r in result)) == set(range(5))
    with pytest.raises(ValueError):
        balanced_select(rows, 81, 123)


def fixture_dataset(tmp_path):
    cfg = config(root=tmp_path)
    cfg.update(train_count=1, eval_count=1, train_participants=['P1'], eval_quotas={'P2': 1})
    root = data_path(cfg)
    boxes = [dict(uid=str(i), center=[0., 0., 1.], size=[.1, .2, .3], rotation=np.eye(3).tolist(),
                  object_bop_id=i, class_id=(i-1) % 25, ignore=False) for i in range(1, 34)]
    rows = []
    for i, actor in enumerate(['P1', 'P2']):
        path = root/f'{actor}.json'; write(path, boxes)
        rows.append(dict(frame_id=actor+':1', sequence_id=actor+'_seq', participant_id=actor,
                         rgb_sha256=str(i)*64, gt_path=str(path), gt_sha256=digest(boxes),
                         clip_id=actor, frame_key='1', image_hw=[768, 768],
                         intrinsics=np.eye(3).tolist(), rgb_path='/unused.png'))
    protocol = dict(classes=[dict(class_id=i, name=str(i), prompt=str(i)) for i in range(25)],
                    train_sha256=digest(rows[:1]), eval_sha256=digest(rows[1:]), benchmark_id='test')
    jsonlines(root/'train.jsonl', rows[:1]); jsonlines(root/'eval.jsonl', rows[1:])
    write(root/'protocol.json', protocol)
    return cfg, rows, protocol


def test_verify_rejects_leakage_and_modified_gt(tmp_path):
    cfg, rows, protocol = fixture_dataset(tmp_path)
    assert verify(cfg)['train']['objects'] == 33
    root = data_path(cfg)
    rows[1]['rgb_sha256'] = rows[0]['rgb_sha256']
    jsonlines(root/'eval.jsonl', rows[1:])
    protocol['eval_sha256'] = digest(rows[1:]); write(root/'protocol.json', protocol)
    with pytest.raises(ValueError, match='rgb_sha256'):
        verify(cfg)


def test_input_manifest_excludes_truth(tmp_path):
    _, rows, _ = fixture_dataset(tmp_path)
    row = dict(rows[0], labels=['secret'], target_count=33, prompts=['secret'], world_pose='secret')
    result = inputs_only(row)
    assert set(result) == {'frame_id', 'clip_id', 'frame_key', 'image_hw', 'intrinsics', 'rgb_sha256', 'rgb_path'}


def test_verify_rejects_changed_groundtruth(tmp_path):
    cfg, rows, _ = fixture_dataset(tmp_path)
    write(rows[0]['gt_path'], [])
    with pytest.raises(ValueError, match='GT 已变化'):
        verify(cfg)


def test_bind_rejects_changed_identity(tmp_path):
    path = tmp_path/'identity.json'
    assert bind(path, {'a': 1}) == bind(path, {'a': 1})
    with pytest.raises(ValueError):
        bind(path, {'a': 2})


def test_mesh_rasterization_clips_and_unions_triangles():
    k = np.array([[20., 0., 32.], [0., 20., 32.], [0., 0., 1.]])
    front = np.array([[[-.5, -.5, 1.], [.5, -.5, 1.], [0., .5, 1.]]])
    mask = amodal_mask(front, np.eye(4), k, (64, 64))
    assert 150 < mask.sum() < 300
    assert np.array_equal(mask, amodal_mask(np.concatenate([front, front]), np.eye(4), k, (64, 64)))
    partial = amodal_mask(front, np.eye(4), k, (64, 64), minimum_area=64)
    assert partial.sum() >= 64
    assert (partial <= mask).all()
    behind = front.copy(); behind[:, :, 2] *= -1
    assert not amodal_mask(behind, np.eye(4), k, (64, 64)).any()
    crossing = front.copy(); crossing[0, 0, 2] = -.1
    assert amodal_mask(crossing, np.eye(4), k, (64, 64)).sum() > 0
