"""Native text-only forwards with positive matched 3D regression losses.

No classification/presence/3D-confidence negative losses are evaluated. The
geometry backend stays in inference mode, so no fake depth GT is needed.
"""
import importlib
import random
import re
from pathlib import Path

import numpy as np

from ..common import checksum, data_path, digest, read


def head_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.bbox3d_head.state_dict().items()}


def freeze(model, wild=False):
    for param in model.parameters():
        param.requires_grad_(False)
    for name, param in model.bbox3d_head.named_parameters():
        # Keep the native confidence regressor frozen as well.
        param.requires_grad_('conf' not in name)
    # Module flags are deliberately set without recursively re-enabling
    # dropout in the frozen detector. SAM3's root flag must remain True to
    # produce Hungarian matches; its frozen children remain in eval mode.
    for module in model.modules():
        module.training = False
    model.training = True
    model.bbox3d_head.train()
    if wild:
        model.sam3.training = True
    elif hasattr(model, 'dn_query_generator'):
        model.dn_query_generator.train()


def checkpoint_coverage(model, checkpoint):
    """Existing demo loaders use strict=False; reject random missing parameters."""
    import torch
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = raw.get('state_dict', raw.get('model', raw))
    cleaned = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = re.sub(r'^(?:module\.|model\.)+', '', key)
        cleaned[key] = value
    required = dict(model.named_parameters())
    missing = sorted(set(required) - set(cleaned))
    wrong = [key for key in required if key in cleaned and required[key].shape != cleaned[key].shape]
    if missing or wrong:
        raise ValueError(f'checkpoint 不完整；缺失参数 {missing[:12]}，形状不符 {wrong[:12]}')
    return dict(parameter_tensors=len(required), missing_parameters=[], shape_mismatches=[])


def load_adapter(cfg, key, device='cuda', delta=None):
    import torch
    torch.set_num_threads(cfg['training']['threads'])
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('当前进程不可见 PPU。请在 DSW 原生 Terminal 执行；不会自动使用 CPU 训练。')
    name = 'mood3d:Mood3DAdapter' if key == '3dmood' else 'wilddet:WildDetAdapter'
    module, cls = name.split(':')
    options = dict(cfg['models'][key], device=device, max_detections=cfg['max_detections'], score_threshold=0.,
                   environment=cfg.get('model', {}).get('environment', {}))
    options.pop('python')
    adapter = getattr(importlib.import_module('ioutest.adapters.' + module), cls)(options)
    try:
        info = adapter.load()
        model = adapter.model if key == '3dmood' else getattr(adapter.model, 'wilddet3d', adapter.model)
        info['coverage'] = checkpoint_coverage(model, cfg['models'][key]['checkpoint'])
        if delta is not None:
            state = torch.load(delta, map_location='cpu', weights_only=False)
            if (state['model_id'] != key or state['base_sha256'] != info['checkpoint_sha256'] or
                    state.get('epochs_completed') != cfg['training']['epochs'] or
                    state['benchmark_id'] != read(data_path(cfg) / 'protocol.json')['benchmark_id']):
                raise ValueError('微调权重模型/基础权重/训练轮数不一致')
            model.bbox3d_head.load_state_dict(state['head'], strict=True)
            info['finetune_sha256'] = checksum(delta)
            info['training_identity'] = state['identity']
        return adapter, model, info

    except BaseException:
        adapter.close()
        raise


def training_image(frame, epoch, seed):
    from ioutest.hot3d.common import rgb
    image = rgb(frame)
    # Stateless augmentation makes resume independent of loader prefetch/RNG.
    rng = random.Random(int(digest([seed, epoch, frame['frame_id']])[:16], 16))
    contrast, brightness = rng.uniform(.9, 1.1), rng.uniform(.9, 1.1)
    out = image.astype(np.float32)
    mean = out.mean(axis=(0, 1), keepdims=True)
    return np.clip((out - mean) * contrast * brightness + mean * brightness, 0, 255).astype(np.uint8)


def box_tensors(frame, data, device):
    """The two native coders store dimensions W,L,H and local axes L,H,W."""
    import torch
    from scipy.spatial.transform import Rotation
    boxes = [b for b in read(frame['gt_path']) if not b['ignore']]
    if not boxes:
        raise ValueError(f'训练帧没有有效 GT：{frame["frame_id"]}')
    transform = data['intrinsics'].detach().cpu().numpy() @ np.linalg.inv(np.asarray(frame['intrinsics']))
    rectangles, geometry, labels = [], [], []
    for box in boxes:
        x0, y0, x1, y1 = box['projected_bbox']
        points = np.array([[x0, y0, 1.], [x1, y1, 1.]]) @ transform.T
        points = points[:, :2] / points[:, 2:]
        rectangles.append(points.reshape(-1).tolist())
        q = Rotation.from_matrix(box['rotation']).as_quat()[[3, 0, 1, 2]]
        size = np.asarray(box['size'])[[2, 0, 1]]
        geometry.append([*box['center'], *size, *q])
        labels.append(box['class_id'])
    return (torch.tensor(rectangles, dtype=torch.float32, device=device),
            torch.tensor(geometry, dtype=torch.float32, device=device),
            torch.tensor(labels, dtype=torch.long, device=device))


class MoodTrainer:
    def __init__(self, adapter, model, classes):
        from opendet3d.op.detect3d.grounding_dino_3d import GroundingDINO3DLoss
        self.adapter, self.model = adapter, model
        self.classes = classes
        self.loss = GroundingDINO3DLoss(box_coder=model.bbox3d_head.box_coder)
        freeze(model)

    def __call__(self, frame, image):
        import torch
        adapter = self.adapter
        values = dict(images=image.astype(np.float32)[None], original_images=image.astype(np.float32)[None],
                      input_hw=image.shape[:2], original_hw=image.shape[:2],
                      intrinsics=np.asarray(frame['intrinsics'], dtype=np.float32),
                      original_intrinsics=np.asarray(frame['intrinsics'], dtype=np.float32))
        data = adapter.to_tensor(adapter.preprocess([values]))[0]
        boxes, boxes3d, labels = box_tensors(frame, data, adapter.device)
        k = data['intrinsics'].to(adapter.device)
        out = self.model(images=data['images'].to(adapter.device), input_hw=[data['input_hw']],
                         intrinsics=k[None], boxes2d=[boxes], boxes2d_classes=[labels],
                         input_texts=[[c['prompt'] for c in self.classes]], padding=[data['padding']])
        dn = out.dn_meta['num_denoising_queries']
        pred = out.all_layers_bbox_3d_preds[-1]
        targets = self.loss.get_targets_3d(
            [out.all_layers_cls_scores[-1, 0, dn:].detach()],
            [out.all_layers_bbox_preds[-1, 0, dn:].detach()], [pred[0]],
            [data['input_hw']], [boxes], [boxes3d], [labels], [k],
            out.positive_maps, out.text_token_mask)
        encoded, weights = torch.cat(targets[4]), torch.cat(targets[5])
        positives = max(int(targets[6]), 1)
        return regression_losses(pred.reshape(-1, pred.shape[-1]), encoded, weights, positives)


def regression_losses(prediction, target, weights, positives):
    import torch
    if not torch.isfinite(target).all() or not torch.isfinite(weights).all():
        raise ValueError('编码后的三维 GT 存在非有限数值')
    error = (prediction - target).abs() * weights
    return {name: error[:, start:end].sum() / max(positives, 1)
            for name, start, end in [('loss_delta_2d', 0, 2), ('loss_depth', 2, 3),
                                      ('loss_dim', 3, 6), ('loss_rot', 6, prediction.shape[-1])]}


class WildTrainer:
    def __init__(self, adapter, model, classes):
        from wilddet3d.loss.wilddet3d_loss import WildDet3DLoss, WildDet3DLossConfig
        self.adapter, self.model, self.classes = adapter, model, classes
        self.loss = WildDet3DLoss(WildDet3DLossConfig(use_o2m=False, use_3d_conf=False), model.box_coder)
        freeze(model, wild=True)

    def __call__(self, frame, image):
        import torch
        from wilddet3d.data_types import WildDet3DInput
        data = self.adapter.preprocess(image, np.asarray(frame['intrinsics'], dtype=np.float32))
        device = self.adapter.device
        boxes, boxes3d, labels = box_tensors(frame, data, device)
        present = labels.unique(sorted=True)
        groups = [torch.where(labels == label)[0] for label in present]
        max_gt = max(len(group) for group in groups)
        gt2d = torch.zeros(len(groups), max_gt, 4, device=device)
        gt3d = torch.zeros(len(groups), max_gt, 10, device=device)
        height, width = data['images'].shape[-2:]
        for index, group in enumerate(groups):
            gt2d[index, :len(group)] = boxes[group] / boxes.new_tensor([width, height, width, height])
            gt3d[index, :len(group)] = boxes3d[group]
        batch = WildDet3DInput(images=data['images'].to(device), intrinsics=data['intrinsics'].to(device)[None],
                              img_ids=torch.zeros(len(groups), dtype=torch.long, device=device),
                              text_ids=torch.arange(len(groups), device=device),
                              unique_texts=[self.classes[int(label)]['prompt'] for label in present],
                              gt_boxes2d=gt2d, gt_boxes3d=gt3d,
                              num_gts=torch.tensor([len(group) for group in groups], device=device),
                              gt_category_ids=present, query_types=torch.zeros(len(groups), dtype=torch.long, device=device),
                              original_hw=[data['original_hw']], padding=[data['padding']])
        out = self.model(batch)
        if out.indices is None or len(out.indices[0]) == 0:
            raise RuntimeError('有效训练帧未产生三维监督匹配，停止而非静默跳过')
        targets = self.loss._normalize_targets(self.loss._build_targets_from_batch(batch))
        indices = tuple(value.to(device) if value is not None else None for value in out.indices)
        return self.loss._loss_boxes_3d(out.pred_boxes_2d, out.pred_boxes_3d, indices, targets,
                                        batch.intrinsics[batch.img_ids], self.loss._get_num_boxes(targets),
                                        image_size=(height, width))
