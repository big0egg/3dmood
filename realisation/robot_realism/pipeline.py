from pathlib import Path
import os
import shutil
import subprocess

from PIL import Image

from .common import PROJECT, atomic_json, digest, file_hash, read_json, safe_child, source_identity, writer_lock
from .controls import prepare_camera
from .config import config_id
from .dataset import bundle, verify_source
from .planning import load_plan, plan, select


def worker_env(config):
    env = os.environ.copy()
    if env.get('COSMOS_SMOKE', '').lower() in ('1', 'true', 'yes', 'y'):
        raise ValueError('COSMOS_SMOKE disables checkpoint loading; it is forbidden for dataset generation')
    # Inference configs also import trainer definitions and LazyConfig.
    # This enables those shared components; the worker only runs inference.
    env.update(COSMOS_SMOKE='0', COSMOS_TRAINING='1', COSMOS_INTERNAL='0',
               HF_HOME=str(Path(config['runtime']['cache']) / 'huggingface'),
               HF_HUB_CACHE=str(Path(config['runtime']['cache']) / 'huggingface/hub'),
               TORCH_HOME=str(Path(config['runtime']['cache']) / 'torch'),
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    env.pop('TRANSFORMERS_CACHE', None)
    env['PYTHONPATH'] = str(PROJECT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def make_profile(config, frozen):
    return {'plan_id': frozen['plan_id'], 'config_id': frozen['config_id'],
            'source_code_sha256': source_identity(), 'upstream_commit': read_json(PROJECT / 'UPSTREAM.json')['commit'],
            'groundtruth_policy': 'inherited_simulation_gt_visual_alignment_requires_review'}


def verify_result(root, record, cameras, profile_id):
    directory = bundle(root, record)
    marker = directory / 'COMMIT.json'
    if not marker.exists():
        return None
    value = read_json(marker)
    if value.get('record') != record or value.get('profile_id') != profile_id or value.get('cameras') != cameras:
        raise ValueError(f'Output bundle belongs to another source/runtime: {directory}')
    required = {'state.npy'} | {f'{cam}/{name}' for cam in cameras
                               for name in ('rgb.png', 'annotation.json', 'depth.npz', 'instance.png', 'render_meta.json')}
    if set(value.get('files_sha256', {})) != required:
        raise ValueError(f'Incomplete output manifest: {directory}')
    for name, expected in value['files_sha256'].items():
        path = safe_child(directory, name)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Committed result missing/corrupt: {path}')
    provenance = value.get('provenance_xml', {})
    p = safe_child(root, provenance.get('path', ''))
    if not p.is_file() or file_hash(p) != provenance.get('sha256'):
        raise ValueError(f'Output scene XML missing/corrupt: {p}')
    return value


def _prepare(config, frozen, records):
    if frozen['config_id'] != config_id(config):
        raise ValueError('Prompt files or configuration changed after planning')
    jobs = []
    scratch = Path(config['output']['scratch']) / frozen['plan_id']
    for index, record in enumerate(records, 1):
        source = bundle(config['input']['root'], record)
        commit = verify_source(config['input']['root'], record, frozen['cameras'])
        if commit.get('plan_id') != frozen['source_plan_id'] or commit.get('profile_id') != frozen['source_profile_id']:
            raise ValueError('Source COMMIT identity differs from frozen realism plan')
        assignment = frozen.get('prompt_assignments', {}).get(record['state_id'], {})
        for name, expected in assignment.get('source_files_sha256', {}).items():
            if commit['files_sha256'].get('agentview/' + name) != expected:
                raise ValueError('Specific prompt source changed after planning')
        views = []
        for camera in frozen['cameras']:
            target = scratch / record['state_id'] / camera
            hints = prepare_camera(source / camera, target, config)
            seed = int(digest([config['selection']['seed'], record['state_id']])[:8], 16)
            spec = {'name': record['state_id'] + '_' + camera, 'video_path': str(source / camera / 'rgb.png'),
                    'max_frames': 1, 'num_video_frames_per_chunk': 1, 'num_conditional_frames': 0,
                    'resolution': config['generation']['resolution'], 'keep_input_resolution': True,
                    'num_steps': config['generation']['steps'], 'guidance': config['generation']['guidance'],
                    'prompt': frozen.get('prompt_assignments', {}).get(record['state_id'], {}).get(
                        'prompt', config['generation']['prompt']),
                    'negative_prompt': config['generation']['negative_prompt'],
                    'seed': seed}
            for key, hint in hints.items():
                spec[key] = {'control_path': hint, 'control_weight': config['controls'][key + '_weight']}
            spec_path = target / 'spec.json'
            atomic_json(spec_path, spec)
            views.append({'camera': camera, 'spec': str(spec_path), 'spec_sha256': file_hash(spec_path),
                          'controls_sha256': {key: file_hash(path) for key, path in hints.items()},
                          'source_directory': str(source / camera)})
        jobs.append({'record': record, 'source_commit_sha256': file_hash(source / 'COMMIT.json'), 'views': views})
        if index % 20 == 0:
            print(f'准备控制图：{index}/{len(records)} 个状态', flush=True)
    atomic_json(Path(config['output']['run_dir']) / 'prepared.json',
                {'plan_id': frozen['plan_id'], 'states': len(jobs), 'images': len(jobs) * len(frozen['cameras']), 'jobs': jobs})
    return jobs


def prepare(config, limit=None):
    with writer_lock(config['output']['run_dir']):
        frozen = plan(config)
        jobs = _prepare(config, frozen, select(frozen, limit))
    return {'prepared_states': len(jobs), 'prepared_images': len(jobs) * len(frozen['cameras']),
            'run_dir': config['output']['run_dir']}


def initialise_output(config, frozen, profile, resume):
    root = Path(config['output']['root'])
    path = root / 'dataset_info.json'
    expected = {'schema_version': 'robot_realism.v1', 'profile_id': digest(profile), 'profile': profile,
                'plan_id': frozen['plan_id'], 'config': config,
                'gt_status': 'inherited_from_simulation_not_certified_for_generated_rgb',
                'multi_view_generation': 'independent_views_shared_state_seed_no_consistency_guarantee'}
    if path.exists():
        value = read_json(path)
        if not resume:
            raise ValueError('Realism dataset already exists; use --resume with the same config')
        if value['profile_id'] != expected['profile_id']:
            raise ValueError('Realism profile differs; keep original version or use a new output and run_dir')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('Output is nonempty and has no realism ownership record')
        atomic_json(path, expected)
        atomic_json(root / 'sampling_plan.json', frozen)
        for name in ('categories.json', 'scene_split.json'):
            source = Path(config['input']['root']) / name
            if source.is_file():
                shutil.copyfile(source, root / name)
    return expected


def write_summary(config, frozen, profile_id, error=None):
    from .blocked import ProgressTracker
    return ProgressTracker(config, frozen, profile_id).flush(error, full=True)


def render(config, limit=None, resume=False):
    from .doctor import doctor
    from .blocked import ProgressTracker
    frozen = load_plan(config)
    selected = select(frozen, limit)
    with writer_lock(config['output']['run_dir']):
        profile = make_profile(config, frozen)
        path = Path(config['output']['root']) / 'dataset_info.json'
        if path.exists():
            info = read_json(path)
            if not resume:
                raise ValueError('Realism dataset already exists; use --resume')
            if info['profile_id'] != digest(profile):
                raise ValueError('Realism profile differs; use original version or a new output and run_dir')
        tracker = ProgressTracker(config, frozen, digest(profile))
        pending = [r for r in selected if not tracker.is_done(r)]
        if not pending:
            summary = write_summary(config, frozen, digest(profile))
            return {k: v for k, v in summary.items() if k not in ('accepted', 'blocked')}
        check = doctor(config)
        if not check['ready']:
            raise ValueError('Cosmos runtime is not ready; inspect doctor.json before rendering')
        jobs = _prepare(config, frozen, pending)
        initialise_output(config, frozen, profile, resume)
        tracker.flush()
        request_path = Path(config['output']['run_dir']) / 'worker_request.json'
        atomic_json(request_path, {'config': config, 'plan': frozen, 'profile': profile, 'jobs': jobs})
        cmd = [config['runtime']['python'], '-B', '-m', 'robot_realism.worker', str(request_path)]
        error = None
        try:
            log_path = Path(config['output']['run_dir']) / 'worker.log'
            with log_path.open('a', encoding='utf-8') as log:
                process = subprocess.Popen(cmd, cwd=PROJECT, env=worker_env(config),
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if code:
                raise RuntimeError(f'Cosmos worker failed (exit {code}); see {log_path}')
            finished = ProgressTracker(config, frozen, digest(profile))
            if any(not finished.is_done(r) for r in selected):
                raise RuntimeError('Worker exited with unaccounted states: neither committed nor explicitly blocked')
        except BaseException as exc:
            error = str(exc) or type(exc).__name__
            raise
        finally:
            summary = write_summary(config, frozen, digest(profile), error)
        return {k: v for k, v in summary.items() if k not in ('accepted', 'blocked')}


def commit_generated(config, frozen, profile_id, job, generated, runtime):
    """Commit all selected cameras together; callable only with real worker outputs."""
    record, cameras = job['record'], frozen['cameras']
    source = bundle(config['input']['root'], record)
    commit = verify_source(config['input']['root'], record, cameras)
    if file_hash(source / 'COMMIT.json') != job['source_commit_sha256']:
        raise ValueError('Source COMMIT changed after prepare')
    for view in job['views']:
        spec = read_json(view['spec'])
        if file_hash(view['spec']) != view['spec_sha256']:
            raise ValueError('Prepared inference spec was modified')
        for key, expected in view['controls_sha256'].items():
            if file_hash(spec[key]['control_path']) != expected:
                raise ValueError('Prepared control was modified')
    target = bundle(config['output']['root'], record)
    if verify_result(config['output']['root'], record, cameras, profile_id):
        return
    owner = {'profile_id': profile_id, 'state_id': record['state_id']}
    if target.exists():
        if not (target / 'PENDING.json').exists() or read_json(target / 'PENDING.json') != owner:
            raise ValueError(f'Refusing to overwrite unowned incomplete output: {target}')
        shutil.rmtree(target)
    atomic_json(target / 'PENDING.json', owner)
    shutil.copyfile(source / 'state.npy', target / 'state.npy')
    hashes = {'state.npy': file_hash(target / 'state.npy')}
    provenance = commit['provenance_xml']
    xml_target = safe_child(config['output']['root'], provenance['path'])
    if xml_target.exists():
        if file_hash(xml_target) != provenance['sha256']:
            raise ValueError('Existing output scene XML differs from source')
    else:
        xml_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_child(config['input']['root'], provenance['path']), xml_target)
    if file_hash(xml_target) != provenance['sha256']:
        raise OSError('Scene XML copy verification failed')
    for view in job['views']:
        camera = view['camera']
        src = source / camera
        folder = target / camera
        folder.mkdir()
        with Image.open(src / 'rgb.png') as original, Image.open(generated[camera]) as image:
            image.load()
            if image.size != original.size:
                raise ValueError('Generated image size differs from source camera calibration')
            image.convert('RGB').save(folder / 'rgb.png')
        for name in ('annotation.json', 'depth.npz', 'instance.png'):
            shutil.copyfile(src / name, folder / name)
        annotation = read_json(folder / 'annotation.json')
        # Keep source geometry, but identify the new RGB as generated appearance.
        annotation['appearance_source'] = 'cosmos_transfer2.5'
        annotation['gt_status'] = 'inherited_simulation_gt_not_revalidated_against_generated_rgb'
        annotation['source_gt_sha256'] = commit['files_sha256'][f'{camera}/annotation.json']
        annotation['files'] = {'rgb': 'rgb.png', 'depth': 'depth.npz', 'instances': 'instance.png'}
        atomic_json(folder / 'annotation.json', annotation)
        atomic_json(folder / 'render_meta.json', {'source_record': record, 'camera': camera,
                    'source_rgb_sha256': commit['files_sha256'][f'{camera}/rgb.png'],
                    'profile_id': profile_id, 'inference': read_json(view['spec']),
                    'control_files_sha256': view['controls_sha256'],
                    'runtime_id': digest(runtime), 'runtime_identity_file': 'runtime_identity.json',
                    'upstream_output_format': 'JPEG; decoded and stored as PNG',
                    'prompt_assignment': frozen.get('prompt_assignments', {}).get(record['state_id']),
                    'prompt_inputs_sha256': frozen.get('prompt_inputs_sha256', {}),
                    'review_status': 'pending_visual_review'})
        for name in ('rgb.png', 'annotation.json', 'depth.npz', 'instance.png', 'render_meta.json'):
            p = folder / name
            hashes[f'{camera}/{name}'] = file_hash(p)
    atomic_json(target / 'COMMIT.json', {'record': record, 'profile_id': profile_id, 'cameras': cameras,
                'files_sha256': hashes, 'source_commit_sha256': job['source_commit_sha256'],
                'provenance_xml': provenance})
    verify_result(config['output']['root'], record, cameras, profile_id)
