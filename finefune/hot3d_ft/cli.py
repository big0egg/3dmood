"""Native-terminal orchestration; each model retains its own Python env."""
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .common import PROJECT, REPO, config, lock, write


def check(cfg):
    checks = {}
    for key, model in cfg['models'].items():
        files = ['python', 'checkpoint', 'project_root']
        if key == 'wilddet3d':
            files.append('lingbot_checkpoint')
        missing = [model[name] for name in files if not Path(model[name]).exists()]
        if missing:
            checks[key] = dict(ready=False, missing=missing)
            continue
        process = subprocess.run([model['python'], '-B', '-c',
            'import json,torch; print(json.dumps(dict(torch=torch.__version__,ppu=torch.cuda.is_available(),'
            'device=torch.cuda.get_device_name() if torch.cuda.is_available() else None)))'],
            capture_output=True, text=True)
        try:
            result = json.loads(process.stdout.splitlines()[-1])
        except (ValueError, IndexError):
            result = dict(error=process.stderr[-1500:])
        checks[key] = dict(ready=process.returncode == 0 and result.get('ppu', False), **result)
    python = Path(cfg['data_python'])
    if python.exists():
        result = subprocess.run([str(python), '-B', '-c',
             'from projectaria_tools.core import data_provider; import cv2,numpy,scipy,PIL,matplotlib'],
             capture_output=True, text=True)
        checks['data'] = dict(ready=result.returncode == 0, error=result.stderr[-1000:])
    else:
        checks['data'] = dict(ready=False, error=f'数据环境不存在：{python}；运行 setup_data_env.sh')
    return dict(ready=all(value['ready'] for value in checks.values()), checks=checks,
                products_root=cfg['root'])


def child(cfg, args, python):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       SAM3_COMPILE='0', XFORMERS_DISABLED='1', MPLBACKEND='Agg',
                       OMP_NUM_THREADS=str(cfg['training']['threads']),
                       OMP_THREAD_LIMIT=str(cfg['training']['threads']), PYTHONFAULTHANDLER='1')
    environment['PYTHONPATH'] = str(REPO) + os.pathsep + str(PROJECT)
    path = Path(cfg['root']) / 'resolved_config.json'
    write(path, cfg)
    command = [str(python), '-B', str(PROJECT / 'legacy_run.py'), *args, '--config', str(path)]
    print('执行：' + ' '.join(command), flush=True)
    subprocess.run(command, check=True, env=environment, cwd=REPO)


def full(cfg):
    readiness = check(cfg)
    if not readiness['ready']:
        raise RuntimeError('环境检查未通过：' + json.dumps(readiness, ensure_ascii=False))
    root = Path(cfg['root'])
    with lock(root / '.suite.lock'):
        try:
            write(root / 'suite_status.json', dict(stage='prepare', complete=False))
            child(cfg, ['prepare'], cfg['data_python'])
            child(cfg, ['verify', '--images'], cfg['data_python'])
            for key, model in cfg['models'].items():
                for stage, args in [
                    ('baseline', ['evaluate', '--model', key, '--variant', 'baseline']),
                    ('training', ['train', '--model', key]),
                    ('finetuned', ['evaluate', '--model', key, '--variant', 'finetuned'])]:
                    write(root / 'suite_status.json', dict(stage=stage, model=key, complete=False))
                    child(cfg, args, model['python'])
            child(cfg, ['report'], cfg['data_python'])
            write(root / 'suite_status.json', dict(stage='complete', complete=True))
        except BaseException as exc:
            write(root / 'suite_status.json', dict(stage='failed', error=str(exc), complete=False,
                                                  resume='再次执行相同命令，完整帧和训练更新自动恢复'))
            raise
    return dict(complete=True, report=str(root / 'report/report.md'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='HOT3D 15k训练/3k评测：两个模型、仅文本、默认四轮训练')
    parser.add_argument('command', choices=['check', 'audit', 'prepare', 'verify', 'train', 'evaluate', 'report', 'run',
                                          'wandb-status', 'wandb-sync'])
    parser.add_argument('--config', type=Path, help='配置文件，默认 configs/default.json')
    parser.add_argument('--root', type=Path, help='产物根目录，默认 /root/hot3d-finetune-v1')
    parser.add_argument('--model', choices=['3dmood', 'wilddet3d'], help='train/evaluate 必填；W&B命令省略则处理两个模型')
    parser.add_argument('--variant', choices=['baseline', 'finetuned'], help='evaluate 必填')
    parser.add_argument('--images', action='store_true', help='verify 同时解码并校验全部图片')
    parser.add_argument('--output', type=Path, help='audit/check 可选保存JSON路径')
    args = parser.parse_args(argv)
    cfg = config(args.config, args.root)
    if args.command in ('train', 'evaluate') and not args.model:
        parser.error('train/evaluate 必须指定 --model')
    if args.command == 'evaluate' and not args.variant:
        parser.error('evaluate 必须指定 --variant')
    if args.command == 'check':
        result = check(cfg)
    elif args.command == 'audit':
        from .dataset import audit_sources
        result = audit_sources(cfg)
    elif args.command in ('prepare', 'verify'):
        from .dataset import prepare, verify
        with lock(Path(cfg['root']) / '.data.lock'):
            result = prepare(cfg) if args.command == 'prepare' else verify(cfg, args.images)
    elif args.command == 'train':
        from .training import train
        def terminate(signum, frame):
            raise KeyboardInterrupt('收到终止信号，保存最近完成的更新')
        signal.signal(signal.SIGTERM, terminate)
        result = train(cfg, args.model)
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        result = evaluate(cfg, args.model, args.variant)
    elif args.command == 'report':
        from .evaluation import report
        result = report(cfg)
    elif args.command in ('wandb-status', 'wandb-sync'):
        from .wandb_log import statuses, sync_history
        result = (statuses if args.command == 'wandb-status' else sync_history)(cfg, args.model)
    else:
        result = full(cfg)
    if args.output:
        write(args.output, result)
    # Source audits can contain many sequences; save full detail to --output.
    display = {k: v for k, v in result.items() if k != 'sequences'} if args.command == 'audit' else result
    print(json.dumps(display, ensure_ascii=False, indent=2))
    return int(args.command == 'check' and not result['ready'])
