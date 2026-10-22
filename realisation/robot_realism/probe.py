"""Small runtime checks; never downloads checkpoints or generates dummy data."""
import importlib.metadata as metadata
import json


def probe():
    checks = []
    packages = {}
    for name in ('torch', 'torchvision', 'transformer-engine', 'flash-attn', 'triton',
                 'numpy', 'pydantic', 'cosmos-transfer2', 'cosmos-oss', 'cosmos-cuda'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
        checks.append({'component': name, 'ok': packages[name] is not None})
    device = None
    try:
        import torch
        available = torch.cuda.is_available()
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
    try:
        import cosmos_transfer2.config  # Includes checkpoint registrations, not weight loading.
        checks.append({'component': 'cosmos_config_import', 'ok': True})
    except Exception as error:
        checks.append({'component': 'cosmos_config_import', 'ok': False, 'error': str(error)[:1500]})
    return {'ready': all(item['ok'] for item in checks), 'checks': checks, 'packages': packages, 'device': device,
            'scope': 'small operators and imports only; a real generated image is still required'}


if __name__ == '__main__':
    result = probe()
    print('REALISM_PROBE_JSON=' + json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['ready'] else 2)
