"""Progress uses worker records; detailed reports reuse the scored predictions."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ioutest import experiment, experiment_worker
from ioutest.common import read_json, write_json
from ioutest.experiment_config import select
from ioutest.adapters.unipose9d import ObjectFailure
from .test_experiment import prepared, FixtureAdapter, box


@pytest.mark.parametrize('track', ['text', 'gt-box'])
@pytest.mark.parametrize('empty', [False, True])
def test_full_report_units_input_meaning_and_reuse(prepared, monkeypatch, track, empty):
    _, config = prepared
    model = config['models'].pop('3dmood')
    model.update(family='wilddet3d', environment={})
    config['models']['wilddet3d'] = model
    source = Path(model['options']['project_root']) / 'wilddet3d'
    source.mkdir()
    (source / '__init__.py').write_text('fixture = 1\n')

    class Adapter(FixtureAdapter):
        def predict(self, sample):
            return [] if empty else [dict(box(center=(.03, 0., 2.), size=(.22, .4, .3)), score=.9)]

        def predict_object(self, sample, seed):
            if empty:
                raise ObjectFailure('fixture_no_output')
            return dict(box(center=(.03, 0., 2.), size=(.22, .4, .3)), class_id=None, score=.9), {}

    monkeypatch.setattr(experiment_worker, 'create_adapter', lambda task: Adapter())
    monkeypatch.setattr(experiment_worker, 'load_head', lambda adapter, family, variant, info: info)
    monkeypatch.setattr(experiment, 'check_environments', lambda tasks: dict(ready=True, environments=[]))
    monkeypatch.setattr(experiment, 'run_worker', lambda task, mode: experiment_worker.work(dict(task, mode=mode)))
    selections = select(config, models=['wilddet3d'], tracks=[track])
    experiment.execute(config, selections, 'all')
    root = Path(config['output']['root'])
    report = (root / 'reports/report.md').read_text()
    for heading in ['### AP', '### 中心误差', '### 尺寸误差', '### 方向误差', '### 感知流程延迟']:
        assert heading in report
    assert 'AP3D@0.50' in report and 'APcenter@0.02m' in report
    assert 'P90 cm' in report and 'P90 %' in report and 'P90 °' in report
    assert '阶段耗时之和不代替实测延迟' in report
    if track == 'gt-box':
        assert '### 条件定位与输出失败' in report and '输出失败率 %' in report
        assert '不代表自主检测召回率' in report
        assert '### 检出、漏检与关联覆盖' not in report
    else:
        assert '### 检出、漏检与关联覆盖' in report
        assert 'Precision %' in report and '类别与几何联合诊断' in report
        assert '### 条件定位与输出失败' not in report
    run = root / 'runs/test/wilddet3d/official' / track
    metrics = read_json(run / 'metrics.json')
    if empty:
        assert metrics['diagnostics']['errors']['center_error_m']['count'] == 0
        assert '| 三维中心距离 | 0 | — | — | — |' in report
    else:
        assert metrics['diagnostics']['errors']['center_error_m']['median'] == pytest.approx(.03)
        assert '| 三维中心距离 | 2 | 3.0000 | 3.0000 | 3.0000 |' in report
    before = {name: (run / name).read_bytes() for name in ['request.json', 'metrics.json']}
    monkeypatch.setattr(experiment, 'run_worker', lambda *args: pytest.fail('report must not infer'))
    experiment.execute(config, selections, 'report')
    assert all((run / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize('exit_code', [0, 3])
def test_worker_progress_is_visible_and_stale_progress_is_ignored(tmp_path, monkeypatch, capsys, exit_code):
    root = tmp_path / 'run'
    write_json(root / 'progress.json', dict(completed=999, total=999, reused=999))
    real_popen = subprocess.Popen
    script = """import json,os,sys
from pathlib import Path
task=json.loads(Path(sys.argv[1]).read_text())
print('worker log preserved', flush=True)
if int(sys.argv[2]) == 0:
    path=Path(task['run_root'])/'progress.json'
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(completed=3,total=3,reused=2)))
    os.replace(tmp,path)
sys.exit(int(sys.argv[2]))
"""
    def start(command, **kwargs):
        assert command[2:4] == ['-m', 'ioutest.experiment_worker']
        return real_popen([sys.executable, '-B', '-c', script, command[-1], str(exit_code)], **kwargs)
    monkeypatch.setattr(experiment.subprocess, 'Popen', start)
    task = dict(run_root=str(root), model=dict(python=sys.executable))
    if exit_code:
        with pytest.raises(RuntimeError, match='worker.log'):
            experiment.run_worker(task, 'infer')
    else:
        experiment.run_worker(task, 'infer')
    output = capsys.readouterr().out
    assert '999/999' not in output
    if exit_code == 0:
        assert '3/3 帧 100.0%' in output and '复用 2' in output
    else:
        assert '100.0%' not in output and '尚无本次帧进度' in output
    assert 'worker log preserved' in (root / 'logs/worker.log').read_text()


def test_progress_eta_excludes_loading_and_cached_frames(tmp_path, monkeypatch, capsys):
    tick = [0.]
    monkeypatch.setattr(experiment.time, 'monotonic', lambda: tick[0])
    progress = experiment._WorkerProgress(tmp_path, 'infer')
    for now, completed, reused in [(100., 90, 90), (110., 91, 90), (115., 92, 90)]:
        tick[0] = now
        write_json(tmp_path / 'progress.json', dict(completed=completed, total=100, reused=reused))
        progress.show(force=True)
    assert '预计剩余 00:00:40' in capsys.readouterr().out
