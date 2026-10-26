"""Run once in each model env: verify 6D rotation and W,L,H conventions."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from ioutest.geometry import corners


@pytest.mark.parametrize('key', ['3dmood', 'wilddet3d'])
def test_native_coder_roundtrip_full_rotation(key):
    if key == '3dmood':
        from opendet3d.op.detect3d.grounding_dino_3d import GroundingDINO3DCoder
        coder = GroundingDINO3DCoder()
    else:
        repo = Path('/mnt/workspace/wilddet3d')
        sys.path.insert(0, str(repo))
        from wilddet3d.head import Det3DCoder
        coder = Det3DCoder(canonical_rotation=True)
    r = Rotation.from_euler('xyz', [.7, -.9, 1.2])
    size = np.array([.13, .07, .22])
    center = np.array([.12, -.04, .8])
    quaternion = r.as_quat()[[3, 0, 1, 2]]
    gt = torch.tensor([[*center, *size[[2, 0, 1]], *quaternion]], dtype=torch.float32)
    boxes = torch.tensor([[380., 300., 500., 430.]])
    k = torch.tensor([[384., 0., 383.5], [0., 384., 383.5], [0., 0., 1.]])
    encoded, weights = coder.encode(boxes, gt, k)
    decoded = coder.decode(boxes, encoded, k)[0].numpy()
    actual = corners(dict(center=decoded[:3], size=decoded[[4, 5, 3]],
                          rotation=Rotation.from_quat(decoded[[7, 8, 9, 6]]).as_matrix()))
    expected = corners(dict(center=center, size=size, rotation=r.as_matrix()))
    distances = np.linalg.norm(actual[:, None] - expected[None], axis=-1)
    assert distances.min(axis=1).max() < 1e-5
    assert (weights > 0).all()


def test_native_head_matched_loss_backward_wilddet3d():
    sys.path.insert(0, '/mnt/workspace/wilddet3d')
    from wilddet3d.head import Det3DHead, Det3DCoder
    from wilddet3d.loss.wilddet3d_loss import WildDet3DLoss, WildDet3DLossConfig
    from wilddet3d.data_types import WildDet3DInput
    torch.set_num_threads(1)
    coder = Det3DCoder(canonical_rotation=True)
    head = Det3DHead(box_coder=coder, use_camera_prompt=False, use_depth_prompt=False)
    prediction, _ = head(torch.randn(6, 1, 3, 256), None, None)
    gt3d = torch.tensor([[[.1, .02, .7, .2, .1, .08, 1., 0., 0., 0.]]])
    k = torch.tensor([[[384., 0., 383.5], [0., 384., 383.5], [0., 0., 1.]]])
    batch = WildDet3DInput(images=torch.zeros(1, 3, 768, 768), intrinsics=k,
                          img_ids=torch.tensor([0]), text_ids=torch.tensor([0]), unique_texts=['mug'],
                          gt_boxes2d=torch.tensor([[[.4, .4, .6, .6]]]), gt_boxes3d=gt3d,
                          num_gts=torch.tensor([1]), query_types=torch.tensor([0]))
    loss = WildDet3DLoss(WildDet3DLossConfig(use_o2m=False, use_3d_conf=False), coder)
    targets = loss._normalize_targets(loss._build_targets_from_batch(batch))
    indices = (torch.tensor([0]), torch.tensor([1]), torch.tensor([0]))
    losses = loss._loss_boxes_3d(torch.zeros(1, 3, 4), prediction[-1], indices, targets, k,
                                loss._get_num_boxes(targets), image_size=(768, 768))
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in head.reg_branches[-2].parameters())
    assert all(p.grad is None for p in head.conf_branches.parameters())
