import importlib.util
from pathlib import Path

import pytest

from hot3d_ft.common import config, read, source_code


def module():
    path = Path(__file__).parents[1]/'legacy_workflow.py'
    spec = importlib.util.spec_from_file_location('test_hot3d_workflow', path)
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    return workflow


def test_all_baselines_precede_training_and_both_tracks_precede_report(tmp_path, monkeypatch):
    workflow = module();cfg = config(root=tmp_path)
    monkeypatch.setattr(workflow, 'check', lambda cfg: {'ready': True})
    observed = []

    def child(cfg, args, python):
        observed.append((args, python, read(tmp_path/'suite_status.json')))

    monkeypatch.setattr(workflow, 'child', child)
    result = workflow.run(cfg)
    assert result['complete'] and result['tracks'] == ['text', 'gt-box']
    phases = [item[2]['stage'] for item in observed]
    assert phases == ['prepare','verify', 'baseline','baseline','baseline',
                      'training','training', 'finetuned','finetuned','finetuned', 'report']
    for start in (2, 7):
        jobs = observed[start:start+3]
        assert [(j[2]['model'],j[2]['track']) for j in jobs] == [
            ('3dmood','text'), ('wilddet3d','text'), ('wilddet3d','gt-box')]
        assert jobs[-1][1] == cfg['models']['wilddet3d']['python']
    assert read(tmp_path/'suite_status.json')['complete']
    assert read(tmp_path/'workflow.json')['jobs'] == workflow.steps(cfg)


def test_failed_baseline_never_starts_training(tmp_path, monkeypatch):
    workflow = module();cfg = config(root=tmp_path)
    monkeypatch.setattr(workflow, 'check', lambda cfg: {'ready': True})
    observed=[]

    def child(cfg,args,python):
        observed.append(args)
        if '--track' in args and 'gt-box' in args:
            raise RuntimeError('baseline interrupted')

    monkeypatch.setattr(workflow, 'child', child)
    with pytest.raises(RuntimeError, match='interrupted'):
        workflow.run(cfg)
    assert not any(args[0] == 'train' for args in observed)
    assert read(tmp_path/'suite_status.json')['stage'] == 'failed'


def test_dry_run_has_no_side_effects_and_does_not_change_legacy_identity(tmp_path,monkeypatch):
    workflow=module()
    monkeypatch.setattr(workflow,'check',lambda cfg: pytest.fail('dry-run must not check PPU'))
    root=tmp_path/'absent'
    assert workflow.main(['--root',str(root),'--dry-run']) == 0
    assert not root.exists()
    assert not any(name.endswith('/workflow.py') for name in source_code())
