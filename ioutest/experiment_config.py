"""Strict experiment configuration, model capabilities and immutable selections."""
import copy
import math
import os
from pathlib import Path
from .common import read_json, safe_name
from .model_environment import normalize_environment

DEFAULT_CONFIG = Path(__file__).parent / 'configs/robot_boxing_v1.json'
CAPABILITIES = {'3dmood': ('text',), 'wilddet3d': ('text', 'gt-box'),
                'unipose9d': ('text', 'gt-box'), 'cutr': ('text',)}
DEFAULT_METRICS = {'iou_thresholds': [.25, .5], 'center_thresholds': [.02, .05, .1, .25, .5, 1.],
                   'association_iou': .5, 'score_floor': 0.}

def resolve_path(value, base):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(base) / path).resolve())

def interpreter_path(value, base):
    # Resolving venv/bin/python symlinks would silently select the base environment.
    path = Path(value).expanduser()
    return os.path.abspath(path if path.is_absolute() else Path(base) / path)

def load_config(path):
    path = Path(path).expanduser().resolve(); c = copy.deepcopy(read_json(path))
    if c.get('schema_version') != 'ioutest.experiment.v1':
        raise ValueError('config schema_version 必须为 ioutest.experiment.v1')
    c['output']['root'] = resolve_path(c['output']['root'], path.parent)
    c['environment'] = normalize_environment(c.get('environment', {}), path.parent)
    for section in ('datasets', 'models'):
        if not isinstance(c.get(section), dict) or not c[section]:
            raise ValueError(section + ' 必须为非空对象')
        for key in c[section]: safe_name(key)
    c['metrics'] = dict(DEFAULT_METRICS, **c.get('metrics', {})); m = c['metrics']
    for field in ('iou_thresholds', 'center_thresholds'):
        values = m[field]
        if (not values or len(set(values)) != len(values) or
                any(not math.isfinite(v) or v <= 0 or (field == 'iou_thresholds' and v > 1) for v in values)):
            raise ValueError('无效的指标阈值：' + field)
    if not 0 < m['association_iou'] <= 1 or not math.isfinite(m['score_floor']):
        raise ValueError('无效的关联阈值或置信度工作点')
    c['timing'] = dict({'warmup_frames': 1}, **c.get('timing', {}))
    if not isinstance(c['timing']['warmup_frames'], int) or c['timing']['warmup_frames'] < 0:
        raise ValueError('warmup_frames 必须为非负整数')
    c['execution'] = dict({'device': 'cuda', 'seed': 0, 'threads': 4}, **c.get('execution', {}))
    if not isinstance(c['execution']['threads'], int) or c['execution']['threads'] < 1: raise ValueError('threads 必须为正整数')
    if not isinstance(c['execution']['seed'], int) or not 0 <= c['execution']['seed'] < 2**32:
        raise ValueError('seed 必须为 0～2^32-1 的整数')
    if not isinstance(c['execution']['device'], str) or not (c['execution']['device'] == 'cpu' or c['execution']['device'] == 'cuda' or
            (c['execution']['device'].startswith('cuda:') and c['execution']['device'][5:].isdigit())):
        raise ValueError('device 必须为 cpu、cuda 或 cuda:N')
    for d in c['datasets'].values():
        d['root'] = resolve_path(d['root'], path.parent)
        for field in ('manifest', 'protocol', 'image_cache', 'depth_manifest', 'symmetries', 'categories_file'):
            if d.get(field): d[field] = resolve_path(d[field], d['root'])
        if not isinstance(d.get('minimum_visible_pixels', 32), int) or d.get('minimum_visible_pixels', 32) < 0:
            raise ValueError('minimum_visible_pixels 必须为非负整数')
    for model in c['models'].values():
        if model.get('family') not in CAPABILITIES:
            raise ValueError('未知模型 family：' + str(model.get('family')))
        model['python'] = interpreter_path(model['python'], path.parent)
        model['environment'] = normalize_environment(model.get('environment', {}), path.parent)
        for field in ('project_root', 'checkpoint', 'lingbot_checkpoint', 'config'):
            if model.get('options', {}).get(field):
                model['options'][field] = resolve_path(model['options'][field], path.parent)
        for name, variant in model['variants'].items():
            safe_name(name)
            for field in ('checkpoint', 'head_checkpoint', 'training_manifest', 'training_root'):
                if variant.get(field): variant[field] = resolve_path(variant[field], path.parent)
    if c.get('frontend'):
        f = c['frontend']
        f['python'] = interpreter_path(f['python'], path.parent)
        for key in ('project_root', 'sam2_checkpoint', 'moge_checkpoint', 'grounding_model'):
            f[key] = resolve_path(f[key], path.parent)
    return c

def select(c, models=None, variants=None, datasets=None, tracks=None):
    defaults = c.get('selection', {})
    models = models or defaults.get('models') or list(c['models'])
    datasets = datasets or defaults.get('datasets') or list(c['datasets'])
    variants = variants or defaults.get('variants') or ['official']
    tracks = tracks or defaults.get('tracks') or ['text']
    for values, choices, label in ((models, c['models'], 'models'), (datasets, c['datasets'], 'datasets'),
                                    (tracks, ('text', 'gt-box'), 'tracks')):
        if len(values) != len(set(values)) or set(values) - set(choices):
            raise ValueError('未知或重复的 ' + label + '：' + str(values))
    if len(variants) != len(set(variants)): raise ValueError('variants 重复')
    jobs = []
    for dataset in datasets:
        for model in models:
            spec = c['models'][model]
            for variant in variants:
                if variant not in spec['variants']: raise ValueError(f'{model} 未配置权重版本 {variant}')
                for track in tracks:
                    if track not in CAPABILITIES[spec['family']]:
                        raise ValueError(f'{model} 不支持 {track}；请分开选择输入条件')
                    jobs.append(dict(dataset=dataset, model=model, variant=variant, track=track))
    return jobs
