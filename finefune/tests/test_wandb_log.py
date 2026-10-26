import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hot3d_ft import wandb_log as logging
from hot3d_ft.common import config, read, write


def row(update):
    return dict(update=update, epoch=1, images_completed=update*8,
                lr=.0001, grad_norm=.2, seconds=3., peak_allocated_bytes=2*1024**3,
                loss_delta_2d=1., loss_depth=2., loss_dim=3., loss_rot=4.)


class FakeRun:
    def __init__(self, directory):
        self.dir = str(directory/'sdk/files')
        self.url = 'https://wandb.ai/test/run'
        self.summary, self.history, self.metrics = {}, [], []
        self.exit_code = None

    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))

    def log(self, values):
        self.history.append(values)

    def finish(self, exit_code):
        self.exit_code = exit_code


def task(tmp_path, mode='offline'):
    cfg = config(root=tmp_path)
    options = logging.settings(cfg)
    options.update(mode=mode, log_every_updates=10)
    journal = tmp_path/'train.jsonl'
    journal.write_text(''.join(json.dumps(row(i))+'\n' for i in range(1, 24)))
    return dict(options=options, model='3dmood', run_id='test', attempt=1,
                directory=str(tmp_path), journal=str(journal), parent_pid=os.getppid(),
                snapshot=True, train_size=15000, group='test', public_config={})


def sdk(run, init):
    return SimpleNamespace(Settings=lambda **kwargs: kwargs, init=init)


def test_snapshot_metric_axes_throttle_and_final_flush(tmp_path):
    work = task(tmp_path)
    run = FakeRun(tmp_path)
    result = logging.worker(work, sdk(run, lambda **kwargs: run))
    assert result['stage'] == 'complete' and result['mode'] == 'offline'
    assert [r['train/update'] for r in run.history] == [1, 10, 20, 23]
    assert run.history[-1]['train/loss_total'] == 10.
    assert run.history[-1]['runtime/peak_allocated_gib'] == 2.
    assert run.history[-1]['train/epoch'] == 23*8/15000
    assert run.exit_code == 0
    assert ('optim/*', {'step_metric': 'train/update'}) in run.metrics
    assert result['url'] is None


def test_online_init_failure_falls_back_without_losing_metrics(tmp_path):
    work = task(tmp_path, mode='online')
    run = FakeRun(tmp_path)
    calls = []

    def initialize(**kwargs):
        calls.append(kwargs['mode'])
        if kwargs['mode'] == 'online':
            raise TimeoutError('private exception details must not enter status')
        return run

    result = logging.worker(work, sdk(run, initialize))
    assert calls == ['online', 'offline']
    assert result['last_update'] == 23
    assert 'private' not in (tmp_path/'status.json').read_text()


def test_failed_sdk_status_preserves_journal(tmp_path):
    work = task(tmp_path)
    before = Path(work['journal']).read_bytes()

    def initialize(**kwargs):
        raise RuntimeError('SDK failed')

    with pytest.raises(RuntimeError, match='SDK failed'):
        logging.worker(work, sdk(None, initialize))
    assert read(tmp_path/'status.json')['stage'] == 'failed'
    assert Path(work['journal']).read_bytes() == before


def test_stop_drains_row_written_between_eof_and_stop(tmp_path, monkeypatch):
    work = task(tmp_path)
    work['snapshot'] = False
    run = FakeRun(tmp_path)
    original = Path.exists

    def exists(path):
        if path == tmp_path/'stop.json' and not original(path):
            with Path(work['journal']).open('a') as stream:
                stream.write(json.dumps(row(24))+'\n')
            write(path, dict(outcome='complete'))
        return original(path)

    monkeypatch.setattr(Path, 'exists', exists)
    logging.worker(work, sdk(run, lambda **kwargs: run))
    assert run.history[-1]['train/update'] == 24


def test_invalid_settings_cannot_abort_training(tmp_path, capsys):
    cfg = config(root=tmp_path)
    cfg['wandb'] = dict(mode='invalid')
    tracker = logging.start_tracking(cfg, '3dmood', tmp_path, 'signature', 15000)
    assert tracker.process is None
    tracker.finish('complete')
    assert '训练继续' in capsys.readouterr().out


def test_attempts_use_fresh_ids_and_preserve_saved_training_metadata(tmp_path, monkeypatch):
    cfg = config(root=tmp_path)
    cfg['wandb'] = dict(mode='offline')
    spawned = []

    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        return SimpleNamespace(wait=lambda timeout=None: 0)

    monkeypatch.setattr(logging.subprocess, 'Popen', popen)
    first = logging.start_tracking(cfg, '3dmood', tmp_path, 'signature', 15000)
    first.finish('failed')
    second = logging.start_tracking(cfg, '3dmood', tmp_path, 'signature', 15000)
    a, b = read(first.directory/'task.json'), read(second.directory/'task.json')
    assert a['run_id'] != b['run_id'] and (a['attempt'], b['attempt']) == (1, 2)
    assert a['group'] == tmp_path.name
    assert spawned[0][0][0] == cfg['models']['3dmood']['python']
    assert spawned[0][1]['stdin'] == logging.subprocess.DEVNULL
    assert read(first.directory/'stop.json')['outcome'] == 'failed'


def test_sync_uses_saved_recipe_and_refuses_live_training(tmp_path, monkeypatch):
    cfg = config(root=tmp_path)
    directory = tmp_path/'runs/3dmood'
    write(directory/'training.json', dict(signature='saved', identity=dict(training={'epochs': 4}, seed=123)))
    (directory/'train.jsonl').write_text(json.dumps(row(1))+'\n')
    calls = []

    def start(saved_cfg, model, path, signature, train_size, snapshot):
        calls.append((saved_cfg, signature, snapshot))
        destination = path/'wandb/attempt-0001'
        write(destination/'status.json', dict(stage='complete', mode='online'))
        return logging.Tracker(SimpleNamespace(wait=lambda: 0), destination)

    monkeypatch.setattr(logging, 'start_tracking', start)
    with logging.lock(directory/'.training.lock'):
        with pytest.raises(RuntimeError, match='已有任务持有锁'):
            logging.sync_history(cfg, '3dmood')
    assert calls == []
    result = logging.sync_history(cfg, '3dmood')
    assert calls[0][0]['training'] == {'epochs': 4}
    assert calls[0][0]['seed'] == 123
    assert calls[0][1:] == ('saved', True)
    assert result['3dmood']['mode'] == 'online'
