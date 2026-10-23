"""Dependency, configuration and small kernel checks; never loads checkpoints."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os

from .dependencies import dependency_checks


def resolve_model_targets(config):
    """Import lazy target names without running any model constructors."""
    from collections.abc import Mapping
    import attrs
    from omegaconf import ListConfig
    from cosmos_transfer2._src.imaginaire.lazy_config.registry import locate

    targets, seen = set(), set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Mapping):
            if '_target_' in value:
                target = value['_target_']
                resolved = locate(target) if isinstance(target, str) else target
                if not callable(resolved):
                    raise ImportError(f'Configuration target is not callable: {target}')
                targets.add(str(target))
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, ListConfig)):
            for item in value:
                visit(item)
        elif not isinstance(value, type) and attrs.has(type(value)):
            for field in attrs.fields(type(value)):
                visit(getattr(value, field.name))
    visit(config)
    return len(targets)


def model_config_check(controls):
    """Resolve the same experiment as the worker, without constructing a model."""
    from cosmos_transfer2._src.imaginaire.lazy_config import LazyConfig  # noqa: F401
    from cosmos_transfer2._src.imaginaire.utils.config_helper import override
    from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.config import make_config
    from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
    from cosmos_transfer2.config import MODEL_CHECKPOINTS, ModelKey

    if len(controls) == 1:
        checkpoint = MODEL_CHECKPOINTS[ModelKey(variant=controls[0], distilled=False)]
        experiment = EXPERIMENTS[checkpoint.experiment]
        name, options = experiment.registered_exp_name, experiment.command_args.copy()
    else:
        name, options = 'multibranch_720p_t24_spaced_layer4_cr1pt1_rectified_flow_inference', []
    # The native inference pipeline removes the training data group as well.
    options.append('~data_train')
    config = override(make_config(), ['--', f'experiment={name}'] + options)
    # Upstream validate() performs a CUDA broadcast. This phase only checks
    # configuration/imports; the worker runs the full validation on the device.
    if any(not getattr(config.job, field) for field in ('project', 'group', 'name')):
        raise ValueError('Model configuration has an incomplete job identity')
    return {'experiment': name, 'resolved_targets': resolve_model_targets(config.model)}


def probe(*, imports_only=False, controls=('depth',)):
    checks = dependency_checks()
    packages = {}
    for name in ('torch', 'torchvision', 'transformer-engine', 'flash-attn', 'triton',
                 'numpy', 'pydantic', 'cosmos-transfer2', 'cosmos-oss', 'cosmos-cuda'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
        checks.append({'component': name, 'ok': packages[name] is not None})
    # Check each entry independently, so one missing module cannot hide others.
    for name in ('decord', 'shapely.geometry', 'sklearn', 'nltk',
                 'cosmos_transfer2.inference'):
        try:
            importlib.import_module(name)
            checks.append({'component': 'import:' + name, 'ok': True})
        except Exception as error:
            checks.append({'component': 'import:' + name, 'ok': False,
                           'error': f'{type(error).__name__}: {error}'[:3000]})
    try:
        details = model_config_check(controls)
        checks.append({'component': 'cosmos_model_config', 'ok': True, **details})
    except Exception as error:
        checks.append({'component': 'cosmos_model_config', 'ok': False,
                       'error': f'{type(error).__name__}: {error}'[:3000]})
    device = None
    try:
        import torch
        available = not imports_only and torch.cuda.is_available()
        if not imports_only:
            checks.append({'component': 'cuda_device', 'ok': available})
        if available:
            device = {'name': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
                      'memory_bytes': torch.cuda.get_device_properties(0).total_memory}
            q = torch.randn(1, 16, 2, 128, device='cuda', dtype=torch.bfloat16)
            def test(name, fn):
                try:
                    with torch.inference_mode():
                        result = fn()
                        torch.cuda.synchronize()
                    if not torch.isfinite(result).all():
                        raise ValueError('Non-finite kernel output')
                    checks.append({'component': name, 'ok': True})
                except Exception as error:
                    checks.append({'component': name, 'ok': False, 'error': str(error)[:1500]})
            test('bf16_matmul', lambda: q.flatten(2) @ q.flatten(2).transpose(-1, -2))
            test('sdpa', lambda: torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)))
            def norm():
                import transformer_engine.pytorch as te
                return te.RMSNorm(128, params_dtype=torch.bfloat16)(q)
            test('te_rmsnorm', norm)
            def flash():
                from flash_attn import flash_attn_func
                return flash_attn_func(q, q, q)
            test('flash_attention_head128', flash)
    except Exception as error:
        checks.append({'component': 'torch_runtime', 'ok': False, 'error': str(error)[:1500]})
    return {'ready': all(item['ok'] for item in checks), 'checks': checks, 'packages': packages, 'device': device,
            'scope': 'dependencies, inference imports and composed model config' +
                     ('; no kernel tests' if imports_only else '; small kernel tests') +
                     '; checkpoints are not loaded and a real generated image is still required'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imports-only', action='store_true')
    parser.add_argument('--controls', nargs='+', choices=('depth', 'edge'), default=['depth'])
    args = parser.parse_args()
    # Keep configuration checks offline, irrespective of the download settings
    # that will be used by the actual render worker in its separate process.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      COSMOS_TRAINING='1', COSMOS_SMOKE='0', COSMOS_INTERNAL='0')
    result = probe(imports_only=args.imports_only, controls=args.controls)
    print('REALISM_PROBE_JSON=' + json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['ready'] else 2)
