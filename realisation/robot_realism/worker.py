"""Single-device persistent Cosmos process. No mock or smoke generation mode."""
import importlib.metadata as metadata
import os
from pathlib import Path
import sys
import traceback

from .common import PROJECT, atomic_json, digest, file_hash, read_json


def stack_identity():
    import torch
    return {'packages': {name: metadata.version(name) for name in
            ('torch', 'torchvision', 'transformer-engine', 'flash-attn', 'triton', 'numpy',
             'pydantic', 'cosmos-transfer2', 'cosmos-oss', 'cosmos-cuda', 'transformers',
             'diffusers', 'huggingface-hub', 'safetensors', 'Pillow', 'av', 'einops', 'megatron-core')},
            'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
            'capability': list(torch.cuda.get_device_capability(0))}


def cached_weights(cache):
    """Record actual HF snapshot content after first inference has loaded models."""
    cache = Path(cache)
    values = {}
    for path in sorted(cache.rglob('*')):
        if path.is_file() and 'snapshots' in path.parts and path.suffix in ('.pt', '.pth', '.bin', '.safetensors', '.json'):
            print(f'记录权重/配置摘要：{path.name}', flush=True)
            values[str(path.relative_to(cache))] = file_hash(path)
    if not values:
        raise RuntimeError('No Hugging Face checkpoint provenance found; refusing unverifiable output')
    return values


def main(request_path):
    from .pipeline import commit_generated, verify_result
    if os.environ.get('COSMOS_SMOKE', '').lower() not in ('0', 'false', ''):
        raise RuntimeError('COSMOS_SMOKE is forbidden')
    request = read_json(request_path)
    config, frozen, profile, jobs = (request[k] for k in ('config', 'plan', 'profile', 'jobs'))
    profile_id = digest(profile)
    from .pipeline import make_profile
    if make_profile(config, frozen) != profile:
        raise ValueError('Source code/configuration changed after worker request')
    stack = stack_identity()
    runtime_path = Path(config['output']['root']) / 'runtime_identity.json'
    previous = read_json(runtime_path) if runtime_path.exists() else None
    if previous:
        if previous['stack'] != stack:
            raise ValueError('Runtime packages/device changed; use a new output and run_dir')
        for relative, expected in previous['cached_files_sha256'].items():
            path = Path(config['runtime']['cache']) / relative
            if path.is_file() and file_hash(path) != expected:
                raise ValueError(f'Cached checkpoint/config changed: {path}')
    from cosmos_oss.init import init_environment, init_output_dir
    from cosmos_transfer2.config import SetupArguments, InferenceArguments
    from cosmos_transfer2.inference import Control2WorldInference
    init_environment()
    native_output = Path(config['output']['scratch']) / frozen['plan_id'] / 'native_output'
    init_output_dir(native_output)
    modes = config['generation']['controls']
    args = SetupArguments(output_dir=native_output, model='depth' if 'depth' in modes else 'edge',
                          disable_guardrails=not config['runtime']['enable_guardrails'],
                          offload_guardrail_models=config['runtime']['offload_guardrails'],
                          context_parallel_size=1, keep_going=False, compile_tokenizer='none')
    inference = Control2WorldInference(args, batch_hint_keys=modes)
    # Capture provenance even when the very first frame is blocked.
    if previous:
        for relative, expected in previous['cached_files_sha256'].items():
            path = Path(config['runtime']['cache']) / relative
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f'Cached checkpoint/config changed or disappeared: {path}')
        runtime = previous
    else:
        runtime = {'stack': stack, 'cached_files_sha256': cached_weights(config['runtime']['cache']),
                   'upstream_model': args.model, 'source_code_sha256': profile['source_code_sha256']}
        atomic_json(runtime_path, runtime)
    process_jobs(config, frozen, profile, jobs, inference, runtime, InferenceArguments)


def process_jobs(config, frozen, profile, jobs, inference, runtime, sample_type):
    """Run real inference in production; callable with isolated fixtures in unit tests."""
    from cosmos_transfer2.guardrail_errors import GuardrailBlockedError
    from .blocked import ProgressTracker, record_blocked, skip_enabled
    from .pipeline import commit_generated
    profile_id = digest(profile)
    tracker = ProgressTracker(config, frozen, profile_id)
    modes = config['generation']['controls']
    for number, job in enumerate(jobs, 1):
        record = job['record']
        if tracker.is_done(record):
            continue
        try:
            generated = {}
            for view in job['views']:
                spec = read_json(view['spec'])
                if file_hash(view['spec']) != view['spec_sha256']:
                    raise ValueError('Prepared spec changed before generation')
                for key, expected in view['controls_sha256'].items():
                    if file_hash(spec[key]['control_path']) != expected:
                        raise ValueError('Control hint changed before generation')
                try:
                    paths = inference.generate([sample_type(**spec)], output_dir=
                        Path(config['output']['scratch']) / frozen['plan_id'] / 'native_output')
                except GuardrailBlockedError as rejection:
                    if not skip_enabled(config):
                        raise
                    record_blocked(config, frozen, profile_id, job, view, rejection, runtime)
                    tracker.rejected(record)
                    tracker.flush()
                    print(f"内容拦截 [{number}/{len(jobs)}] {record['state_id']} "
                          f"stage={rejection.stage}: {rejection.reason}；已记录，继续下一帧", flush=True)
                    break
                if len(paths) != 1 or not Path(paths[0]).is_file():
                    raise RuntimeError('Cosmos did not produce an image; no placeholder will be saved')
                generated[view['camera']] = paths[0]
            if record['state_id'] in tracker.blocked:
                continue
            commit_generated(config, frozen, profile_id, job, generated, runtime)
            tracker.successful(record)
            tracker.flush()
            print(f"真实写实化 [{number}/{len(jobs)}] {record['split']}/{record['scene_id']}/{record['state_id']}：{len(generated)} 张已提交", flush=True)
            # Only cleanup files owned by this request, after complete commit.
            for view in job['views']:
                for key in modes:
                    Path(read_json(view['spec'])[key]['control_path']).unlink(missing_ok=True)
                Path(view['spec']).unlink(missing_ok=True)
            for path in generated.values():
                Path(path).unlink(missing_ok=True)
        except BaseException as error:
            import json
            log_path = Path(config['output']['run_dir']) / 'worker_errors.jsonl'
            with log_path.open('a') as stream:
                stream.write(json.dumps({'state_id': record['state_id'], 'error': str(error),
                                         'traceback': traceback.format_exc()}, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            raise
    return tracker.flush(full=True)


if __name__ == '__main__':
    main(sys.argv[1])
