from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from robot_realism.common import atomic_json, digest, read_json, safe_child
from robot_realism.config import config_id
from robot_realism.controls import depth_control
from robot_realism.dataset import bundle
from robot_realism.planning import plan, load_plan, select
from robot_realism.pipeline import (commit_generated, initialise_output, make_profile, prepare,
                                     verify_result, worker_env, write_summary)
from robot_realism.reporting import report, validate


def test_plan_reproducible_preserves_split_and_camera_pairs(config):
    first = plan(config)
    assert plan(config) == first == load_plan(config)
    assert first['targets'] == {'train': 4, 'eval': 4}
    assert len(select(first, 2)) == 1
    assert len(select(first, 200)) == 4
    for limit in (0, -2, 3):
        with pytest.raises(ValueError, match='divisible'):
            select(first, limit)
    changed = deepcopy(config)
    changed['generation']['steps'] += 1
    with pytest.raises(ValueError, match='differs'):
        plan(changed)


def test_modified_plan_rejected(config):
    plan(config)
    path = Path(config['output']['run_dir']) / 'plan.json'
    value = read_json(path)
    value['records'][0]['split'] = 'eval'
    atomic_json(path, value)
    with pytest.raises(ValueError, match='modified'):
        load_plan(config)


def test_prepare_single_frame_uses_real_depth_keeps_source_untouched(config):
    result = prepare(config, 2)
    assert result['prepared_images'] == 2
    frozen = load_plan(config)
    job = read_json(Path(config['output']['run_dir']) / 'prepared.json')['jobs'][0]
    specs = [read_json(view['spec']) for view in job['views']]
    assert specs[0]['seed'] == specs[1]['seed']
    assert specs[0]['max_frames'] == specs[0]['num_video_frames_per_chunk'] == 1
    with Image.open(specs[0]['depth']['control_path']) as image:
        hint = np.array(image)
    assert hint.shape == (32, 32, 3) and hint.dtype == np.uint8
    assert hint[0, 0, 0] > hint[0, -1, 0]
    assert specs[0]['keep_input_resolution']
    assert not Path(config['output']['root']).exists()
    # Replacing a payload without updating COMMIT is detected before generation.
    (bundle(config['input']['root'], frozen['records'][0]) / 'agentview/depth.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        prepare(config, 2)


def test_depth_invalid_background_and_flat_surface():
    depth = np.array([[.3, 1, 0]], dtype=np.float32)
    hint = depth_control(depth, np.array([[True, True, False]]), .05, 5)
    assert hint[0, 0, 0] == 255 and hint[0, 1, 0] == hint[0, 2, 0] == 0
    assert (depth_control(np.ones((2, 2)), np.ones((2, 2), bool), .05, 5) == 255).all()
    with pytest.raises(ValueError):
        depth_control(depth, np.zeros((1, 3), bool), .05, 5)


def make_test_commit(config):
    """Unit test only: direct storage call, never a fake Cosmos CLI backend."""
    prepare(config, 2)
    frozen = load_plan(config)
    profile = make_profile(config, frozen)
    initialise_output(config, frozen, profile, False)
    job = read_json(Path(config['output']['run_dir']) / 'prepared.json')['jobs'][0]
    generated = {v['camera']: str(Path(v['source_directory']) / 'rgb.png') for v in job['views']}
    runtime = {'test_only': True}
    atomic_json(Path(config['output']['root']) / 'runtime_identity.json', runtime)
    commit_generated(config, frozen, digest(profile), job, generated, runtime)
    return frozen, profile, job


def test_commit_validate_report_and_resuming_identity(config):
    frozen, profile, job = make_test_commit(config)
    assert verify_result(config['output']['root'], job['record'], frozen['cameras'], digest(profile))
    result = validate(config, frozen, allow_partial=True)
    assert result['valid'] and not result['complete'] and not result['visual_gt_alignment_certified']
    assert not validate(config, frozen)['valid']
    assert report(config, frozen, 2)['previews'] == 2
    summary = write_summary(config, frozen, digest(profile))
    assert summary['accepted_images'] == 2
    with pytest.raises(ValueError, match='resume'):
        initialise_output(config, frozen, profile, False)
    initialise_output(config, frozen, profile, True)
    changed = {**profile, 'source_code_sha256': 'changed'}
    with pytest.raises(ValueError, match='differs'):
        initialise_output(config, frozen, changed, True)


def test_corrupt_committed_result_is_not_overwritten(config):
    frozen, profile, job = make_test_commit(config)
    folder = bundle(config['output']['root'], job['record'])
    (folder / 'agentview/rgb.png').write_bytes(b'damaged')
    with pytest.raises(ValueError, match='corrupt'):
        verify_result(config['output']['root'], job['record'], frozen['cameras'], digest(profile))
    assert not validate(config, frozen, True)['valid']


def test_reject_unowned_partial_and_modified_control(config):
    prepare(config, 2)
    frozen = load_plan(config)
    profile = make_profile(config, frozen)
    initialise_output(config, frozen, profile, False)
    job = read_json(Path(config['output']['run_dir']) / 'prepared.json')['jobs'][0]
    generated = {v['camera']: str(Path(v['source_directory']) / 'rgb.png') for v in job['views']}
    target = bundle(config['output']['root'], job['record'])
    target.mkdir(parents=True)
    (target / 'foreign.txt').write_text('keep')
    with pytest.raises(ValueError, match='unowned'):
        commit_generated(config, frozen, digest(profile), job, generated, {})
    assert (target / 'foreign.txt').read_text() == 'keep'
    view = job['views'][0]
    Path(read_json(view['spec'])['depth']['control_path']).write_bytes(b'modified')
    with pytest.raises(ValueError, match='control was modified'):
        commit_generated(config, frozen, digest(profile), job, generated, {})


def test_no_smoke_generation_or_unsafe_paths(config, monkeypatch):
    monkeypatch.setenv('COSMOS_SMOKE', '1')
    with pytest.raises(ValueError, match='forbidden'):
        worker_env(config)
    with pytest.raises(ValueError, match='escapes'):
        safe_child(config['input']['root'], '../outside')
    other = deepcopy(config)
    other['output']['run_dir'] = '/different/run'
    assert config_id(config) == config_id(other)


def test_interpreter_symlink_does_not_escape_venv(tmp_path):
    import sys
    import yaml
    from robot_realism.common import PROJECT
    from robot_realism.config import load_config
    value = yaml.safe_load((PROJECT / 'configs/realism/cosmos25_libero_v1.yaml').read_text())
    interpreter = tmp_path / 'venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    value['runtime']['python'] = str(interpreter)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(value))
    assert load_config(path)['runtime']['python'] == str(interpreter)


def test_failed_runtime_gate_does_not_create_dataset(config, monkeypatch):
    from robot_realism.pipeline import render
    plan(config)
    monkeypatch.setattr('robot_realism.doctor.doctor', lambda config: {'ready': False})
    with pytest.raises(ValueError, match='not ready'):
        render(config, 2)
    assert not Path(config['output']['root']).exists()


def test_modified_runtime_provenance_is_rejected(config):
    frozen, _, _ = make_test_commit(config)
    atomic_json(Path(config['output']['root']) / 'runtime_identity.json', {'test_only': True, 'changed_weights': True})
    result = validate(config, frozen, True)
    assert not result['valid']
    assert any('runtime/checkpoint' in e['error'] for e in result['errors'])
