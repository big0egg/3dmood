"""Isolated W&B SDK process: metrics never import the SDK into model envs.

An attempt gets a fresh run and replays committed local history. This avoids
duplicate/backwards cloud steps after a hard kill rolls back a checkpoint.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import PROJECT, REPO, digest, lock, read, write


def settings(cfg):
    result = read(PROJECT / 'configs/wandb.json')
    result.update(cfg.get('wandb', {}))
    if 'WANDB_MODE' in os.environ:
        result['mode'] = os.environ['WANDB_MODE']
    if result['mode'] not in ('online', 'offline', 'disabled'):
        raise ValueError('wandb.mode/WANDB_MODE 只能为 online、offline、disabled')
    for name in ('log_every_updates', 'init_timeout'):
        if type(result[name]) != int or result[name] < 1:
            raise ValueError(f'wandb.{name} 必须为正整数')
    return result


def metric_row(row, train_size):
    losses = {key: float(value) for key, value in row.items() if key.startswith('loss_')}
    return {'train/update': row['update'], 'train/loss_total': sum(losses.values()),
            **{'train/' + key: value for key, value in losses.items()},
            'train/epoch': row['epoch'] - 1 + row['images_completed'] / train_size,
            'optim/lr': row['lr'], 'optim/grad_norm': row['grad_norm'],
            'runtime/seconds_per_update': row['seconds'],
            'runtime/peak_allocated_gib': row['peak_allocated_bytes'] / 1024**3}


class Tracker:
    def __init__(self, process=None, directory=None):
        self.process, self.directory = process, directory

    def finish(self, outcome):
        if self.process is None:
            return
        # Metrics are supplementary: a broken upload never invalidates a
        # completed optimizer update or replaces the original training error.
        try:
            write(self.directory / 'stop.json', dict(outcome=outcome))
            code = self.process.wait(timeout=5)
            path = self.directory / 'status.json'
            status = read(path) if path.is_file() else {}
            if code or status.get('stage') == 'failed':
                print(f'W&B 上传未完成，本地训练记录保留；查看 {path}', flush=True)
            elif status.get('mode') == 'offline':
                print(f'W&B 为离线记录，尚未上传网页；登录后可 wandb-sync。状态：{path}', flush=True)
            elif status.get('url'):
                print('W&B 训练曲线：' + status['url'], flush=True)
        except subprocess.TimeoutExpired:
            print(f'W&B 正在后台完成同步，状态：{self.directory / "status.json"}', flush=True)
        except Exception as exc:
            print(f'W&B 收尾未完成，训练记录仍保存在 train.jsonl：{type(exc).__name__}', flush=True)


def start_tracking(cfg, key, directory, signature, train_size, snapshot=False):
    directory = Path(directory)
    attempt = None
    try:
        options = settings(cfg)
        if not options['enabled'] or options['mode'] == 'disabled':
            return Tracker()
        with lock(directory / 'wandb/.attempt.lock'):
            attempts = list((directory / 'wandb').glob('attempt-*'))
            number = max([int(p.name.split('-')[-1]) for p in attempts] or [0]) + 1
            attempt = directory / 'wandb' / f'attempt-{number:04d}'
            run_id = digest([signature, str(directory.resolve()), time.time_ns(), number])[:16]
            task = dict(options=options, model=key, run_id=run_id, attempt=number,
                        directory=str(attempt), journal=str(directory / 'train.jsonl'),
                        parent_pid=os.getpid(), snapshot=snapshot, train_size=train_size,
                        group=options['group'] or Path(cfg['root']).name,
                        public_config=dict(model=key, training=cfg['training'], seed=cfg['seed'],
                                           train_count=train_size, training_identity=signature))
            write(attempt / 'task.json', task)
            write(attempt / 'status.json', dict(stage='starting', model=key, mode=options['mode']))
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                           WANDB_SILENT='true', MPLBACKEND='Agg')
        environment['PYTHONPATH'] = str(PROJECT) + os.pathsep + str(REPO)
        with (attempt / 'uploader.log').open('a') as stream:
            process = subprocess.Popen([options['python'], '-B', '-m', 'hot3d_ft.wandb_log',
                                        str(attempt / 'task.json')], stdout=stream,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=environment)
        print(f'W&B 独立日志进程已启动：{attempt / "status.json"}', flush=True)
        return Tracker(process, attempt)
    except Exception as exc:
        if attempt is not None:
            try:
                write(attempt / 'status.json', dict(stage='failed', error_type=type(exc).__name__))
            except OSError:
                pass
        print(f'W&B 未能启动：{type(exc).__name__}；训练继续，指标保存在 train.jsonl。', flush=True)
        return Tracker()


def worker(task, sdk=None):
    directory = Path(task['directory'])
    options = task['options']
    run = None
    stage = 'failed'
    try:
        if sdk is None:
            import wandb as sdk
        mode = options['mode']
        fallback = None
        # Let the public SDK resolve login credentials. The child has no TTY
        # and stdin=/dev/null, so missing login fails instead of prompting.
        kwargs = dict(project=options['project'], entity=options['entity'], group=task['group'],
                      name=f'{options["name_prefix"]}-{task["model"]}-a{task["attempt"]:04d}',
                      id=task['run_id'], job_type='train', dir=str(directory), config=task['public_config'],
                      settings=sdk.Settings(init_timeout=options['init_timeout'], login_timeout=1,
                                            disable_git=True, save_code=False))
        try:
            run = sdk.init(mode=mode, **kwargs)
        except Exception as exc:
            if mode != 'online' or not options['fallback_offline']:
                raise
            # Never write exception text that could contain a credential/URL.
            fallback = f'在线初始化失败（{type(exc).__name__}），改为离线记录'
            mode = 'offline'
            run = sdk.init(mode=mode, **kwargs)
        run.define_metric('train/update')
        for prefix in ('train/*', 'optim/*', 'runtime/*'):
            run.define_metric(prefix, step_metric='train/update')
        url = getattr(run, 'url', None) if mode == 'online' else None
        status = dict(stage='logging', model=task['model'], mode=mode, url=url,
                      sdk_directory=str(run.dir), run_id=task['run_id'], fallback=fallback,
                      history_policy='new run per attempt; replay committed local journal', last_update=0)
        write(directory / 'status.json', status)
        if url:
            print('W&B 实时曲线：' + url, flush=True)
        if fallback:
            print(fallback, flush=True)
        position, previous, last_logged = 0, None, 0
        stop_seen = False
        while True:
            journal = Path(task['journal'])
            if journal.exists():
                with journal.open() as stream:
                    stream.seek(position)
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith('\n'):
                            position = start
                            break
                        row = json.loads(line)
                        previous = row
                        if row['update'] == 1 or row['update'] % options['log_every_updates'] == 0:
                            run.log(metric_row(row, task['train_size']))
                            last_logged = row['update']
                        position = stream.tell()
            stop = directory / 'stop.json'
            parent_gone = os.getppid() != task['parent_pid']
            stopping = stop.exists() or parent_gone or task['snapshot']
            if stopping:
                # Drain once more after observing stop: the trainer may have
                # appended its final row between our EOF and stop-file check.
                if not stop_seen:
                    stop_seen = True
                    continue
                if previous is not None and previous['update'] != last_logged:
                    run.log(metric_row(previous, task['train_size']))
                    last_logged = previous['update']
                outcome = read(stop)['outcome'] if stop.exists() else ('snapshot' if task['snapshot'] else 'interrupted')
                run.summary['training_outcome'] = outcome
                stage = 'complete' if outcome in ('complete', 'snapshot') else 'interrupted'
                break
            status['last_update'] = previous['update'] if previous else 0
            write(directory / 'status.json', status)
            time.sleep(.5)
        run.finish(exit_code=0 if stage == 'complete' else 1)
        status.update(stage=stage, last_update=last_logged)
        write(directory / 'status.json', status)
        return status
    except Exception as exc:
        write(directory / 'status.json', dict(stage='failed', error_type=type(exc).__name__,
              hint='训练日志保留在 train.jsonl；检查SDK环境、登录和网络后使用 wandb-sync 补传'))
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        raise


def statuses(cfg, key=None):
    result = {}
    for model in ([key] if key else cfg['models']):
        directory = Path(cfg['root']) / 'runs' / model / 'wandb'
        result[model] = [dict(path=str(p), **read(p))
                         for p in sorted(directory.glob('attempt-*/status.json'))]
    return result


def sync_history(cfg, key=None):
    """Replay local metrics using saved training metadata, without model/PPU."""
    results = {}
    for model in ([key] if key else cfg['models']):
        directory = Path(cfg['root']) / 'runs' / model
        # Same lock as train(): no reads during checkpoint rollback/truncation.
        with lock(directory / '.training.lock'):
            binding = read(directory / 'training.json')
            if not (directory / 'train.jsonl').is_file():
                raise FileNotFoundError(directory / 'train.jsonl')
            saved_cfg = dict(cfg, training=binding['identity']['training'], seed=binding['identity']['seed'])
            tracker = start_tracking(saved_cfg, model, directory, binding['signature'], cfg['train_count'], snapshot=True)
            if tracker.process is None:
                raise RuntimeError('W&B 补传未启动；检查配置与终端提示')
            code = tracker.process.wait()
            result = read(tracker.directory / 'status.json')
            results[model] = dict(path=str(tracker.directory), **result)
            if code or result['stage'] != 'complete':
                raise RuntimeError(f'W&B 补传未完成，请查看 {tracker.directory / "uploader.log"}')
    return results


if __name__ == '__main__':
    worker(read(sys.argv[1]))
