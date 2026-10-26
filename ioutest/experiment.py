"""Single public CLI: prepared test data -> predictions -> metrics -> reports."""
import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from .common import digest, file_digest, read_json, write_json, utc_now
from .experiment_config import DEFAULT_CONFIG, load_config, select
from .experiment_data import audit_training, input_frames, load_dataset
from .hot3d.common import bind
from .model_environment import bert_cache_files, child_environment

def runtime_files(model, variant, frontend=None, track='text', environment=None):
    paths = dict(model.get('options', {}))
    if variant.get('checkpoint'): paths['checkpoint'] = variant['checkpoint']
    if variant.get('head_checkpoint'): paths['head_checkpoint'] = variant['head_checkpoint']
    checks = {}
    for key in ('checkpoint', 'lingbot_checkpoint', 'head_checkpoint', 'config'):
        if paths.get(key):
            p = Path(paths[key])
            with p.open('rb') as f: prefix = f.read(160)
            if not prefix or prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError('权重／配置为空或是 Git LFS 指针：' + str(p))
            checks[key] = file_digest(p)
    root = Path(model['options']['project_root'])
    folder = {'3dmood': 'opendet3d', 'wilddet3d': 'wilddet3d', 'unipose9d': 'infer', 'cutr': 'cubifyanything'}[model['family']]
    files = sorted((root / folder).rglob('*.py'))
    if not files: raise FileNotFoundError('模型源码缺失：' + str(root / folder))
    checks['model_source'] = digest({str(p.relative_to(root)): file_digest(p) for p in files})
    if model['family'] == '3dmood':
        checks['bert_cache'] = {name: file_digest(p) for name, p in bert_cache_files(environment or {}).items()}
    if frontend is not None:
        needed = ['sam2_checkpoint'] + ([] if paths.get('depth_model') else ['moge_checkpoint'])
        for key in needed:
            checks['frontend_' + key] = file_digest(frontend[key])
        if track == 'text':
            grounding = Path(frontend['grounding_model'])
            if not (grounding / 'config.json').is_file(): raise FileNotFoundError(grounding / 'config.json')
            checks['grounding'] = digest({str(p.relative_to(grounding)): file_digest(p) for p in sorted(grounding.rglob('*'))
                                         if p.is_file() and '.cache' not in p.parts})
        checks['frontend_source'] = file_digest(Path(frontend['project_root']) / 'infer/moge_model.py')
    return checks

def make_tasks(c, selections, data):
    root = Path(c['output']['root']); tasks = []; assets = {}; audits = {}
    package = Path(__file__).parent
    sources = ['experiment_worker.py', 'frontend_worker.py', 'worker_rpc.py', 'records.py', 'checkpoints.py',
               'timing.py', 'schema.py', 'suite_io.py', 'suite_frontend.py', 'experiment_data.py', 'hot3d/common.py', 'model_environment.py']
    code = {n: file_digest(package / n) for n in sources}
    code.update({str(p.relative_to(package)): file_digest(p) for p in (package / 'adapters').glob('*.py')})
    for s in selections:
        model = c['models'][s['model']]; variant = model['variants'][s['variant']]
        needs_frontend = model['family'] == 'unipose9d'
        front = c.get('frontend') if needs_frontend else None
        environment = dict(c.get('environment', {}), **model.get('environment', {}))
        if needs_frontend and not front: raise ValueError('UniPose9D 需要配置 frontend')
        if model.get('options', {}).get('use_gravity'): raise ValueError('统一入口未声明实测重力输入；请关闭 use_gravity')
        key = (s['model'], s['variant'], s['track'])
        if key not in assets: assets[key] = runtime_files(model, variant, front, s['track'], environment)
        audit_key = (s['dataset'], variant.get('training_manifest'))
        if audit_key not in audits: audits[audit_key] = audit_training(variant, data[s['dataset']])
        options = dict(model['options'])
        if variant.get('checkpoint'): options['checkpoint'] = variant['checkpoint']
        inputs = input_frames(data[s['dataset']], s['track'], options.get('depth_model', False))
        head_variant = {k: variant[k] for k in ('head_checkpoint', 'training_benchmark_id', 'training_identity', 'epochs_completed') if k in variant}
        if variant.get('head_checkpoint') and variant.get('training_manifest'):
            protocol_path = Path(variant.get('training_root', Path(variant['training_manifest']).parent)) / 'protocol.json'
            if protocol_path.is_file():
                protocol = read_json(protocol_path)
                if protocol.get('benchmark_id'): head_variant.setdefault('training_benchmark_id', protocol['benchmark_id'])
        task = dict(dataset=s['dataset'], model_name=s['model'], variant_name=s['variant'], track=s['track'],
            model_id=s['model'] + '/' + s['variant'], model=dict(family=model['family'], python=model['python']),
            variant=head_variant,
            options=options, classes=data[s['dataset']]['classes'], frames=inputs,
            execution=c['execution'], timing=c['timing'], frontend=front,
            environment=environment,
            dataset_id=data[s['dataset']]['dataset_id'], assets=assets[key], inference_code=code,
            training_audit=audits[audit_key])
        task['request_id'] = digest(task)
        task['run_root'] = str(root / 'runs' / s['dataset'] / s['model'] / s['variant'] / s['track'])
        tasks.append(task)
    # Source datasets and training manifests must remain outside the output tree.
    for spec in c['datasets'].values():
        source = Path(spec['root'])
        if root == source or root.is_relative_to(source): raise ValueError('输出目录不能写入源数据集')
    for task in tasks:
        for f in task['frames']:
            p = Path(f['rgb_path'])
            if p == root or p.is_relative_to(root): raise ValueError('输出根目录不能包含输入测试图片')
        manifest = task['training_audit'].get('training_manifest')
        if manifest and Path(manifest).is_relative_to(root): raise ValueError('输出目录不能包含训练来源清单')
    return tasks

def check_environments(tasks):
    checks = []
    required = {'3dmood': ['opendet3d', 'vis4d', 'vis4d_cuda_ops', 'transformers'],
                'wilddet3d': ['wilddet3d', 'vis4d', 'sam3'], 'unipose9d': ['unipose9d_inference'],
                'cutr': ['cubifyanything']}
    requested = set()
    for t in tasks:
        family = t['model']['family']; project = Path(t['options']['project_root'])
        source = project / 'infer' if family == 'unipose9d' else project
        environment = tuple(sorted(t.get('environment', {}).items()))
        requested.add((t['model']['python'], t['execution']['device'], str(source), tuple(required[family]), environment))
        if t.get('frontend'):
            source = Path(t['frontend']['project_root']) / 'infer'
            requested.add((t['frontend']['python'], t['execution']['device'], str(source), ('sam2', 'transformers', 'moge_model'), environment))
    program = """import importlib.util,json,sys,torch
sys.path.insert(0,sys.argv[2])
missing=[n for n in json.loads(sys.argv[3]) if importlib.util.find_spec(n) is None]
device=torch.device(sys.argv[1])
ready=device.type!='cuda' or (torch.cuda.is_available() and (device.index is None or device.index<torch.cuda.device_count()))
print(json.dumps(dict(torch=torch.__version__,device_ready=ready,missing_modules=missing)))
sys.exit(0 if ready and not missing else 2)
"""
    import json
    for python, device, source, modules, environment in sorted(requested):
        try:
            p = subprocess.run([python, '-B', '-c', program, device, source, json.dumps(modules)],
                               env=child_environment(os.environ, dict(environment)), capture_output=True, text=True, timeout=120)
            checks.append(dict(python=python, ready=p.returncode == 0, detail=(p.stdout + p.stderr)[-4000:]))
        except (OSError, subprocess.TimeoutExpired) as exc: checks.append(dict(python=python, ready=False, detail=str(exc)))
    return dict(ready=all(r['ready'] for r in checks), environments=checks)

def run_worker(task, mode):
    root = Path(task['run_root']); task = dict(task, mode=mode)
    path = root / 'logs' / (mode + '_task.json'); write_json(path, task)
    env = dict(child_environment(os.environ, task.get('environment', {})), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', XFORMERS_DISABLED='1')
    project = str(Path(__file__).resolve().parents[1]); env['PYTHONPATH'] = project + os.pathsep + env.get('PYTHONPATH', '')
    with (root / 'logs' / 'worker.log').open('a') as log:
        result = subprocess.run([task['model']['python'], '-B', '-m', 'ioutest.experiment_worker', str(path)], env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"模型进程失败（{result.returncode}）：{root / 'logs/worker.log'}")

def execute(c, selections, stage):
    names = list(dict.fromkeys(s['dataset'] for s in selections))
    data = {}
    for name in names:
        print('读取已处理测试集：' + name, flush=True)
        data[name] = load_dataset(c['datasets'][name])
    tasks = make_tasks(c, selections, data)
    if stage == 'check':
        result = check_environments(tasks)
        result.update(datasets={n: dict(frames=len(d['frames']), dataset_id=d['dataset_id']) for n, d in data.items()},
                      jobs=len(tasks), training_audits=[t['training_audit'] for t in tasks])
        return result
    root = Path(c['output']['root']); root.mkdir(parents=True, exist_ok=True)
    with (root / '.evaluation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / 'resolved_config.json', c)
        write_json(root / 'experiment.json', dict(schema=c['schema_version'], selection=selections,
            datasets={n: d['dataset_id'] for n, d in data.items()}))
        for name, d in data.items():
            bind(root / 'audits' / (name + '_dataset.json'), dict(dataset_id=d['dataset_id'], classes=d['classes'], metadata=d['metadata']))
        try:
            if stage in ('all', 'infer', 'latency'):
                ready = check_environments(tasks); write_json(root / 'readiness.json', ready)
                if not ready['ready']: raise RuntimeError('选中模型的环境／PPU 检查失败，见 readiness.json')
            for task in tasks:
                directory = Path(task['run_root'])
                bind(directory / 'request.json', {k: v for k, v in task.items() if k != 'run_root'})
                write_json(directory / 'training_audit.json', task['training_audit'])
                write_json(root / 'status.json', dict(stage=stage, job={k: task[k] for k in ('dataset', 'model_name', 'variant_name', 'track')}, status='running'))
                if stage in ('all', 'infer'): run_worker(task, 'infer')
                elif stage == 'latency': run_worker(task, 'latency')
                if stage in ('all', 'score', 'latency'):
                    from .experiment_score import score
                    score(task, data[task['dataset']], c['metrics'])
            if stage in ('all', 'report', 'latency'):
                from .experiment_report import assemble
                result = assemble(root, tasks, data, c['metrics'])
            else: result = dict(complete=True, jobs=len(tasks), stage=stage)
            write_json(root / 'status.json', dict(status='complete', stage=stage, updated_utc=utc_now()))
            return result
        except BaseException as exc:
            write_json(root / 'status.json', dict(status='failed', stage=stage, error=str(exc), updated_utc=utc_now()))
            raise

def main(argv=None):
    parser = argparse.ArgumentParser(description='三维框评测：只读取已处理测试集，不训练、不制作数据')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--models', nargs='+', help='配置 models 中的模型 ID')
    parser.add_argument('--variants', nargs='+', help='各模型 variants 中的权重版本')
    parser.add_argument('--datasets', nargs='+', help='配置 datasets 中的测试集 ID')
    parser.add_argument('--tracks', nargs='+', choices=['text', 'gt-box'])
    parser.add_argument('--stage', choices=['check', 'infer', 'score', 'latency', 'report', 'all'], default='all')
    parser.add_argument('--debug', action='store_true', help='显示完整错误堆栈')
    args = parser.parse_args(argv)
    try:
        c = load_config(args.config)
        selections = select(c, args.models, args.variants, args.datasets, args.tracks)
        result = execute(c, selections, args.stage)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get('ready') is False else 0
    except Exception as exc:
        if args.debug: raise
        print('ioutest: ' + str(exc), file=sys.stderr)
        return 1
