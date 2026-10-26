"""Verified base weights and optional strictly matched three-dimensional heads."""
import re
from .common import file_digest

def load_head(adapter, family, variant, info):
    import torch
    model = adapter.model if family == '3dmood' else getattr(adapter.model, 'wilddet3d', adapter.model)
    # Native base loaders can use strict=False. Verify all trainable parameter names/shapes.
    raw = torch.load(adapter.checkpoint, map_location='cpu', weights_only=False) if hasattr(adapter, 'checkpoint') else None
    if raw is not None:
        state = raw.get('state_dict', raw.get('model', raw))
        state = {re.sub(r'^(?:module\.|model\.)+', '', k): v for k, v in state.items() if isinstance(v, torch.Tensor)}
        required = dict(model.named_parameters())
        missing = [k for k, v in required.items() if k not in state or state[k].shape != v.shape]
        if missing: raise ValueError('基础权重参数缺失或形状不符：' + str(missing[:12]))
        info['base_parameter_coverage'] = len(required)
    path = variant.get('head_checkpoint')
    if not path: return info
    if family not in ('3dmood', 'wilddet3d'): raise ValueError('该模型不支持三维头增量权重')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('model_id') != family or state.get('base_sha256') != info['checkpoint_sha256']:
        raise ValueError('三维头与模型／官方基础权重不匹配')
    for field, config_key in (('benchmark_id', 'training_benchmark_id'), ('identity', 'training_identity'), ('epochs_completed', 'epochs_completed')):
        if variant.get(config_key) is not None and state.get(field) != variant[config_key]:
            raise ValueError('三维头训练来源不匹配：' + field)
    if not state.get('identity') or not state.get('benchmark_id') or state.get('epochs_completed', 0) < 1:
        raise ValueError('三维头缺少完整训练来源')
    model.bbox3d_head.load_state_dict(state['head'], strict=True)
    info.update(head_sha256=file_digest(path), training_identity=state['identity'], training_benchmark_id=state['benchmark_id'],
                epochs_completed=state['epochs_completed'])
    return info
