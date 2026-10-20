from copy import deepcopy
from pathlib import Path

import pytest

from robot_synth.collection import collect, collection_goals
from robot_synth.common import local_lock, read_json
from robot_synth.planning.sampling import build_plan
from robot_synth.reporting.summary import report
from robot_synth.sources.libero_hdf5 import inspect_source
from robot_synth.storage.writer import DatasetWriter
from robot_synth.validation.dataset import validate_dataset

from conftest import FakeReplay, make_view

PROFILE = {'backend': 'unit_test', 'test_only': True}


def test_pilot_resume_full_and_report(config):
    inventory = inspect_source(config)
    plan = build_plan(config, inventory)
    first = collect(config, plan, inventory, limit=12, replay_factory=FakeReplay, profile=PROFILE)
    assert first['status'] == 'partial' and first['accepted_images'] == 12
    assert len(first['counts_by_scene']) == 6
    assert validate_dataset(config, plan, allow_partial=True)['valid']
    assert not validate_dataset(config, plan)['complete']
    second = collect(config, plan, inventory, resume=True, replay_factory=FakeReplay, profile=PROFILE)
    assert second['status'] == 'complete' and second['accepted_images'] == 48
    validation = validate_dataset(config, plan)
    assert validation['valid'] and validation['counts'] == {'train': 36, 'eval': 12}
    assert not validation['native_renderer_verified']
    result = report(config, plan, maximum_previews=12)
    assert len(result['previews']) == 12
    before = FakeReplay.loads
    collect(config, plan, inventory, resume=True, replay_factory=FakeReplay, profile=PROFILE)
    assert FakeReplay.loads == before


def test_no_resume_profile_change_and_corrupt_payload_rejected(config):
    inventory = inspect_source(config)
    plan = build_plan(config, inventory)
    collect(config, plan, inventory, limit=2, replay_factory=FakeReplay, profile=PROFILE)
    with pytest.raises(ValueError, match='resume'):
        collect(config, plan, inventory, limit=2, replay_factory=FakeReplay, profile=PROFILE)
    with pytest.raises(ValueError, match='identity differs'):
        collect(config, plan, inventory, limit=2, resume=True, replay_factory=FakeReplay, profile={'backend': 'changed'})
    rgb = next(Path(config['output']['root']).rglob('rgb.png'))
    rgb.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='missing/corrupt'):
        collect(config, plan, inventory, limit=2, resume=True, replay_factory=FakeReplay, profile=PROFILE)
    assert not validate_dataset(config, plan, allow_partial=True)['valid']


def test_pair_limits_and_runtime_gate(config):
    inventory = inspect_source(config)
    plan = build_plan(config, inventory)
    with pytest.raises(ValueError, match='divisible'):
        collection_goals(plan, 3)
    import importlib.util
    if importlib.util.find_spec('mujoco') is None:
        with pytest.raises(RuntimeError, match='not ready'):
            collect(config, plan, inventory, limit=2)
        assert not Path(config['output']['root']).exists()


def test_owned_interrupted_copy_can_retry(config, monkeypatch):
    inventory = inspect_source(config)
    plan = build_plan(config, inventory)
    writer = DatasetWriter(config, plan, PROFILE)
    record = plan['states'][0]
    from robot_synth.sources.libero_hdf5 import read_episode
    xml, states = read_episode(config, record['source'], record['episode'])
    state = states[record['state_index']]
    replay = FakeReplay(config, xml)
    views = replay.render(state)
    from robot_synth.validation.geometry import validate_views
    quality = validate_views(views, config)
    import robot_synth.storage.writer as module
    original = module.shutil.copyfileobj
    def fail(*args, **kwargs):
        raise OSError('simulated copy interruption')
    monkeypatch.setattr(module.shutil, 'copyfileobj', fail)
    with pytest.raises(OSError):
        writer.save(record, views, state, xml, quality)
    assert writer.existing(record) is None
    monkeypatch.setattr(module.shutil, 'copyfileobj', original)
    assert writer.save(record, views, state, xml, quality)
    assert writer.existing(record)


def test_lock_prevents_second_writer(tmp_path):
    with local_lock(tmp_path):
        with pytest.raises(RuntimeError, match='active writer'):
            with local_lock(tmp_path):
                pass


def test_resume_recovers_commit_when_progress_was_lost(config):
    inventory = inspect_source(config)
    plan = build_plan(config, inventory)
    first = collect(config, plan, inventory, limit=12, replay_factory=FakeReplay, profile=PROFILE)
    original_ids = set(first['accepted'])
    (Path(config['output']['run_dir']) / 'progress.json').unlink()
    (Path(config['output']['root']) / 'manifest.json').unlink()
    before = FakeReplay.loads
    recovered = collect(config, plan, inventory, limit=12, resume=True, replay_factory=FakeReplay, profile=PROFILE)
    assert set(recovered['accepted']) == original_ids
    assert FakeReplay.loads == before


def test_rejected_states_do_not_count_toward_image_quotas(config):
    inventory = inspect_source(config)
    plan = build_plan(config, inventory)
    class RejectOnce(FakeReplay):
        rejected = False
        def render(self, state):
            values = super().render(state)
            if not RejectOnce.rejected:
                RejectOnce.rejected = True
                for view in values.values():
                    view['rgb'][:] = 0
            return values
    result = collect(config, plan, inventory, limit=12, replay_factory=RejectOnce, profile=PROFILE)
    assert result['accepted_images'] == 12
    assert len(result['rejected']) == 1


def test_resumed_identical_images_rejected(config):
    inventory = inspect_source(config)
    plan = build_plan(config, inventory)
    class StaticReplay(FakeReplay):
        def render(self, state):
            return {camera: make_view(camera) for camera in self.config['simulation']['cameras']}
    # First state is legitimate; all later states have the identical image pair.
    with pytest.raises(RuntimeError, match='quality failures|Candidate pool exhausted'):
        collect(config, plan, inventory, replay_factory=StaticReplay, profile=PROFILE)
    progress = read_json(Path(config['output']['run_dir']) / 'progress.json')
    assert progress['accepted_images'] == 2
    assert any('Identical' in item['reason'] for item in progress['rejected'].values())
    with pytest.raises(RuntimeError, match='Candidate pool exhausted'):
        collect(config, plan, inventory, resume=True, replay_factory=StaticReplay, profile=PROFILE)
    assert read_json(Path(config['output']['run_dir']) / 'progress.json')['accepted_images'] == 2
