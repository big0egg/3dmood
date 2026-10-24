from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from robot_realism.common import atomic_json, digest, file_hash, read_json
from robot_realism.dataset import bundle, source_records
from robot_realism.planning import load_plan, plan
from robot_realism.pipeline import prepare
from robot_realism.prompting import (GENERIC_PROMPT, build_selection, build_specific_row,
    describe_frame, observed_color, record_identity, resolve_assignments, write_jsonl)
from robot_realism.prompt_suite import export_results, initialize, migrate_cache, paired_specs_match


def test_full_selection_exact_counts_balances_scenes_and_preserves_eval(monkeypatch):
    records = []
    for split, counts in [('train', {'KITCHEN': 2500, 'LIVING_ROOM': 1500, 'STUDY': 1000}),
                          ('eval', {'KITCHEN': 250, 'LIVING_ROOM': 125, 'STUDY': 125})]:
        for room, count in counts.items():
            for i in range(count):
                records.append({'state_id': f'{split}-{room}-{i}', 'split': split, 'room': room,
                    'scene_id': f'{room}_{split}_SCENE{i % 3}', 'source': f'{room}/task_{i % 5}',
                    'episode': f'demo_{i % 7}', 'state_index': i})
    info = {'plan_id': 'test-source', 'profile_id': 'test-profile'}
    monkeypatch.setattr('robot_realism.prompting.source_records', lambda config: (info, records))
    config = {'input': {'cameras': ['agentview']}, 'selection': {'seed': 20261022}}
    selection, chosen = build_selection(config)
    assert build_selection(config) == (selection, chosen)
    assert {k: len(v) for k, v in selection['groups'].items()} == {
        'train_general': 4800, 'train_specific': 200, 'eval_general': 500}
    assert Counter(r['room'] for r in chosen) == Counter(KITCHEN=100, LIVING_ROOM=60, STUDY=40)
    assert all(r['split'] == 'train' for r in chosen)
    assert len({r['scene_id'] for r in chosen}) == 9
    assert len({r['episode'] for r in chosen}) > 1
    assert set(selection['groups']['eval_general']) == {r['state_id'] for r in records if r['split'] == 'eval'}
    records.pop()
    with pytest.raises(ValueError, match='exactly 5000 train'):
        build_selection(config)


def object_record(instance_id=1, **extra):
    return dict({'instance_id': instance_id, 'category': 'bowl', 'raw_category': 'black_bowl',
        'kind': 'object', 'parent_instance_id': None, 'visible': True, 'visible_pixels': 256,
        'segmentation_ids': [instance_id], 'bbox_visible_xyxy': [8, 8, 24, 24],
        'size_xyz': [.1, .1, .1], 'center_camera': [0, 0, 1],
        'R_camera_from_box': np.eye(3).tolist()}, **extra)


def test_specific_description_uses_visible_roots_and_masked_rgb():
    rgb = np.zeros((32, 32, 3), np.uint8)
    mask = np.ones((32, 32), np.uint16)
    obj = object_record()
    annotation = {'objects': [obj, object_record(2, kind='part'), object_record(3, visible=False)]}
    text, evidence = describe_frame(rgb, mask, annotation)
    assert len(evidence) == 1
    assert 'black bowl' in text and 'middle central' in text
    assert 'grasping' not in text and 'picking' not in text
    # Name says black, but observed RGB is red: don't invent the named color.
    rgb[..., 0] = 255
    assert observed_color(rgb, mask, obj) == 'red'
    assert 'red bowl' in describe_frame(rgb, mask, annotation)[0]
    with pytest.raises(ValueError, match='No reliably visible'):
        describe_frame(rgb, mask, {'objects': [object_record(1, visible=False)]})


@pytest.fixture
def prompt_config(config, monkeypatch, tmp_path):
    config['input']['cameras'] = ['agentview']
    info, records = source_records(config)
    selected = records[0]
    folder = bundle(config['input']['root'], selected)
    annotation = read_json(folder/'agentview/annotation.json')
    annotation['objects'] = [object_record()]
    atomic_json(folder/'agentview/annotation.json', annotation)
    commit = read_json(folder/'COMMIT.json')
    commit['files_sha256']['agentview/annotation.json'] = file_hash(folder/'agentview/annotation.json')
    atomic_json(folder/'COMMIT.json', commit)
    config['generation']['prompt'] = GENERIC_PROMPT
    selection = {'camera': 'agentview', 'source_id': record_identity(info, records),
        'seed': config['selection']['seed'], 'groups': {
            'train_general': [records[1]['state_id']], 'train_specific': [selected['state_id']],
            'eval_general': [r['state_id'] for r in records if r['split'] == 'eval']}}
    selection['selection_id'] = digest(selection)
    selection_path = tmp_path/'selection.json'
    specific_path = tmp_path/'specific.jsonl'
    atomic_json(selection_path, selection)
    write_jsonl(specific_path, [build_specific_row(config, selected, GENERIC_PROMPT)])
    config['prompting'] = {'mode': 'mixed', 'selection_manifest': str(selection_path),
                           'specific_manifest': str(specific_path)}
    monkeypatch.setattr('robot_realism.prompting.GROUP_COUNTS',
                        {'train_general': 1, 'train_specific': 1, 'eval_general': 2})
    return config


def test_single_view_prompt_dispatch_and_paired_seeds(prompt_config):
    config = prompt_config
    mixed = plan(config)
    assert mixed['targets'] == {'train': 2, 'eval': 2}
    prepare(config)
    jobs = read_json(Path(config['output']['run_dir'])/'prepared.json')['jobs']
    specs = {j['record']['state_id']: read_json(j['views'][0]['spec']) for j in jobs}
    assert all(len(j['views']) == 1 and j['views'][0]['camera'] == 'agentview' for j in jobs)
    for state_id, spec in specs.items():
        group = mixed['prompt_assignments'][state_id]['group']
        assert (spec['prompt'] != GENERIC_PROMPT) == (group == 'train_specific')
    pair = deepcopy(config)
    pair['prompting']['mode'] = 'paired_general'
    pair['output'] = {k: v+'_paired' for k, v in config['output'].items()}
    frozen = plan(pair)
    assert frozen['targets'] == {'train': 1, 'eval': 0}
    prepare(pair)
    job = read_json(Path(pair['output']['run_dir'])/'prepared.json')['jobs'][0]
    baseline = read_json(job['views'][0]['spec'])
    specific = specs[job['record']['state_id']]
    assert baseline['prompt'] == GENERIC_PROMPT
    assert baseline['seed'] == specific['seed']
    assert paired_specs_match(baseline, specific)
    assert file_hash(baseline['depth']['control_path']) == file_hash(specific['depth']['control_path'])


def test_prompt_mutation_missing_entries_and_changed_source_are_rejected(prompt_config):
    config = prompt_config
    plan(config)
    path = Path(config['prompting']['specific_manifest'])
    content = path.read_text()
    row = json.loads(content)
    row['prompt'] += ' changed'
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match='Configuration differs'):
        load_plan(config)
    write_jsonl(path, [])
    info, records = source_records(config)
    with pytest.raises(ValueError, match='Exactly one'):
        resolve_assignments(config, info, records)
    path.write_text(content)
    row = json.loads(content)
    source = next(r for r in records if r['state_id'] == row['state_id'])
    (bundle(config['input']['root'], source)/'agentview/rgb.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Specific prompt source changed'):
        resolve_assignments(config, info, records)


def test_frozen_specific_prompt_rejects_recommitted_rgb(prompt_config):
    from robot_realism.pipeline import _prepare
    config = prompt_config
    frozen = plan(config)
    chosen = next(r for r in frozen['records']
                  if frozen['prompt_assignments'][r['state_id']]['group'] == 'train_specific')
    source = bundle(config['input']['root'], chosen)
    rgb = source/'agentview/rgb.png'
    rgb.write_bytes(rgb.read_bytes()+b'changed-source')
    commit = read_json(source/'COMMIT.json')
    commit['files_sha256']['agentview/rgb.png'] = file_hash(rgb)
    atomic_json(source/'COMMIT.json', commit)
    with pytest.raises(ValueError, match='Specific prompt source changed after planning'):
        _prepare(config, frozen, frozen['records'])


def test_cache_copy_preserves_relative_links_checks_existing_files_and_ignores_downloads(tmp_path):
    source, target = tmp_path/'old', tmp_path/'new'
    blobs = source/'huggingface/hub/model/blobs'
    snapshot = source/'huggingface/hub/model/snapshots/revision'
    blobs.mkdir(parents=True); snapshot.mkdir(parents=True)
    (blobs/'weight').write_bytes(b'UNIT TEST ONLY')
    (blobs/'partial.incomplete').write_bytes(b'incomplete')
    (snapshot/'model.pt').symlink_to('../../blobs/weight')
    migrate_cache(source, target)
    copied = target/'huggingface/hub/model/snapshots/revision/model.pt'
    assert copied.is_symlink() and copied.read_bytes() == b'UNIT TEST ONLY'
    assert not list(target.rglob('*.incomplete'))
    migrate_cache(source, target)
    (target/'huggingface/hub/model/blobs/weight').write_bytes(b'changed')
    with pytest.raises(ValueError, match='content differs'):
        migrate_cache(source, target)


def test_suite_setup_resume_export_and_pair_report(prompt_config, monkeypatch, tmp_path):
    # Unit-test fixtures only: no mock render backend is exposed by the CLI.
    from robot_realism.pipeline import commit_generated, initialise_output, make_profile
    config = prompt_config
    info, records = source_records(config)
    selection = read_json(config['prompting']['selection_manifest'])
    chosen = [r for r in records if r['state_id'] in selection['groups']['train_specific']]
    monkeypatch.setattr('robot_realism.prompt_suite.build_selection', lambda config: (selection, chosen))
    monkeypatch.setattr('robot_realism.prompt_suite.GROUP_COUNTS',
                        {'train_general': 1, 'train_specific': 1, 'eval_general': 2})
    root = tmp_path/'suite'
    configs = initialize(root, config['input']['root'], config['runtime']['python'], config['runtime']['cache'])
    initial_text = (root/'prompts/specific.jsonl').read_text()
    assert initialize(root, config['input']['root'], config['runtime']['python'], config['runtime']['cache']) == configs
    assert (root/'prompts/specific.jsonl').read_text() == initial_text
    from robot_realism.config import load_config
    for path in configs.values():
        current = load_config(path)
        frozen = plan(current)
        prepare(current)
        profile = make_profile(current, frozen)
        initialise_output(current, frozen, profile, False)
        runtime = {'test_only': True}
        atomic_json(Path(current['output']['root'])/'runtime_identity.json', runtime)
        jobs = read_json(Path(current['output']['run_dir'])/'prepared.json')['jobs']
        for job in jobs:
            generated = {v['camera']: str(Path(v['source_directory'])/'rgb.png') for v in job['views']}
            commit_generated(current, frozen, digest(profile), job, generated, runtime)
    summary = export_results(root, configs)
    assert summary['generated_images'] == 5 and summary['complete']
    for name in ('train_all_general', 'train_mixed'):
        lines = (root/'manifests'/f'{name}.jsonl').read_text().splitlines()
        assert len(lines) == 2
        assert len({json.loads(line)['state_id'] for line in lines}) == 2
    assert len(list((root/'previews/paired').glob('*.jpg'))) == 1
    assert len(list((root/'previews/eval').glob('*.jpg'))) == 2
    assert (root/'reports/paired_review.csv').is_file()
