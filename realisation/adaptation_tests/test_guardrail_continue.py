"""Fixture-only inference tests. No fake generation backend is exposed to users."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cosmos_transfer2.guardrail_errors import GuardrailBlockedError, rejection
from robot_realism.blocked import (ProgressTracker, blocked_path, read_blocked, record_blocked)
from robot_realism.common import PROJECT, atomic_json, digest, read_json
from robot_realism.dataset import bundle
from robot_realism.pipeline import initialise_output, make_profile, prepare
from robot_realism.planning import plan
from robot_realism.reporting import report, validate
from robot_realism.worker import process_jobs
from test_prompt_suite import prompt_config  # Shared source-backed prompt fixture.


@pytest.fixture
def v3_run(config):
    config['input']['cameras'] = ['agentview']
    config['execution'] = {'guardrail_policy': 'record_and_skip'}
    frozen = plan(config)
    prepare(config)
    profile = make_profile(config, frozen)
    initialise_output(config, frozen, profile, False)
    runtime = {'test_only': True}
    atomic_json(Path(config['output']['root'])/'runtime_identity.json', runtime)
    jobs = read_json(Path(config['output']['run_dir'])/'prepared.json')['jobs']
    return config, frozen, profile, jobs, runtime


class FixtureInference:
    def __init__(self, blocked=(), failure=None):
        self.blocked = set(blocked)
        self.failure = failure
        self.calls = []

    def generate(self, samples, output_dir):
        sample = samples[0]
        self.calls.append(sample.name)
        if self.failure:
            raise self.failure
        if sample.name in self.blocked:
            raise GuardrailBlockedError('output', 'unsafe frames detected', {
                'guardrail': 'VIDEOCONTENTSAFETYFILTER',
                'details': {'frame_scores': [{'frame_number': 0, 'class': 'Sexual_Content'}]}})
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir/(sample.name+'.png')
        shutil.copyfile(sample.video_path, target)
        return [target]


def job_name(job):
    return read_json(job['views'][0]['spec'])['name']


def run_fixture(v3_run, inference):
    config, frozen, profile, jobs, runtime = v3_run
    return process_jobs(config, frozen, profile, jobs, inference, runtime, SimpleNamespace)


def test_block_continues_and_resume_skips_terminal_states(v3_run):
    config, frozen, profile, jobs, _ = v3_run
    generator = FixtureInference(blocked=[job_name(jobs[0])])
    result = run_fixture(v3_run, generator)
    assert len(generator.calls) == 4
    assert result['status'] == 'finished_with_blocks'
    assert result['generated_images'] == 3 and result['blocked_images'] == 1
    assert result['pending_images'] == 0 and result['execution_complete']
    assert not result['complete'] and not result['dataset_complete']
    blocked = read_blocked(config, frozen, digest(profile), jobs[0]['record'], 'agentview')
    assert blocked['stage'] == 'output' and blocked['resolution_status'] == 'pending_manual_review'
    assert blocked['details']['details']['frame_scores'][0]['class'] == 'Sexual_Content'
    assert not (bundle(config['output']['root'], jobs[0]['record'])/'COMMIT.json').exists()
    assert len((Path(config['output']['run_dir'])/'blocked.jsonl').read_text().splitlines()) == 1
    progress = read_json(Path(config['output']['run_dir'])/'progress.json')
    assert progress['accepted_images'] == 3
    assert not validate(config, frozen)['valid']
    checked = validate(config, frozen, allow_blocked=True)
    assert checked['valid'] and checked['execution_complete'] and not checked['complete']
    resumed = FixtureInference()
    assert run_fixture(v3_run, resumed)['blocked_images'] == 1
    assert resumed.calls == []


@pytest.mark.parametrize('message', ['out of memory', 'Guardrail blocked video2world generation.'])
def test_ordinary_errors_are_fatal_even_if_message_mentions_guardrail(v3_run, message):
    generator = FixtureInference(failure=RuntimeError(message))
    with pytest.raises(RuntimeError, match=message.replace('.', r'\.')):
        run_fixture(v3_run, generator)
    assert len(generator.calls) == 1
    config = v3_run[0]
    assert not list((Path(config['output']['run_dir'])/'blocked').glob('*.json'))


def test_failed_block_record_write_stops_before_next_frame(v3_run, monkeypatch):
    generator = FixtureInference(blocked=[job_name(v3_run[3][0])])
    monkeypatch.setattr('robot_realism.blocked.atomic_json',
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        run_fixture(v3_run, generator)
    assert len(generator.calls) == 1


def test_resume_recovers_atomic_record_without_progress_file(v3_run):
    config, frozen, profile, jobs, runtime = v3_run
    error = GuardrailBlockedError('positive_prompt', 'classified unsafe')
    record_blocked(config, frozen, digest(profile), jobs[0], jobs[0]['views'][0], error, runtime)
    assert not (Path(config['output']['run_dir'])/'progress.json').exists()
    recovered = ProgressTracker(config, frozen, digest(profile))
    assert recovered.summary()['blocked_images'] == 1
    generator = FixtureInference()
    result = run_fixture(v3_run, generator)
    assert len(generator.calls) == 3 and result['execution_complete']


def test_corrupt_or_mismatched_record_is_never_silently_skipped(v3_run):
    config, frozen, profile, jobs, runtime = v3_run
    record = jobs[0]['record']
    record_blocked(config, frozen, digest(profile), jobs[0], jobs[0]['views'][0],
                   GuardrailBlockedError('negative_prompt', 'classified unsafe'), runtime)
    changed = deepcopy(config)
    changed['generation']['prompt'] += ' changed'
    with pytest.raises(ValueError, match='another input/configuration'):
        read_blocked(changed, frozen, digest(profile), record, 'agentview')
    path = blocked_path(config, record, 'agentview')
    value = read_json(path)
    value['reason'] = 'tampered'
    atomic_json(path, value)
    with pytest.raises(ValueError, match='corrupt'):
        ProgressTracker(config, frozen, digest(profile))


def test_all_blocked_is_finished_but_not_a_complete_dataset(v3_run):
    config, frozen, _, jobs, _ = v3_run
    generator = FixtureInference(blocked=[job_name(job) for job in jobs])
    result = run_fixture(v3_run, generator)
    assert result['generated_images'] == 0 and result['blocked_images'] == 4
    assert result['execution_complete'] and not result['dataset_complete']
    assert validate(config, frozen, allow_blocked=True)['valid']
    assert report(config, frozen)['previews'] == 0


def test_allow_blocked_does_not_accept_unaccounted_missing_frames(v3_run):
    config, frozen, _, _, _ = v3_run
    checked = validate(config, frozen, allow_blocked=True)
    assert not checked['valid'] and checked['pending_images'] == 4


def test_stop_policy_propagates_explicit_rejections(v3_run):
    config, frozen, _, _, runtime = v3_run
    config['execution']['guardrail_policy'] = 'stop'
    # This is a separate test execution identity, not a compatibility override.
    profile = make_profile(config, frozen)
    atomic_json(Path(config['output']['root'])/'dataset_info.json',
                {'profile_id': digest(profile), 'profile': profile})
    generator = FixtureInference(blocked=[job_name(v3_run[3][0])])
    with pytest.raises(GuardrailBlockedError):
        process_jobs(config, frozen, profile, v3_run[3], generator, runtime, SimpleNamespace)


def test_rejection_carries_stage_and_runner_decision():
    runner = SimpleNamespace(last_result={'message': 'unsafe frames', 'guardrail': 'FILTER',
                            'details': {'frame_scores': [{'class': 'Sexual_Content'}]}})
    error = rejection('output', runner, 'fallback')
    assert isinstance(error, GuardrailBlockedError)
    assert error.stage == 'output' and error.reason == 'unsafe frames'
    assert error.details['guardrail'] == 'FILTER'


@pytest.fixture
def guardrail_classes(monkeypatch):
    # Isolate constructor-only model imports. The actual checker methods run below;
    # no weights, Transformers model initialization or PPU SDK are needed by these unit tests.
    vision_name = 'cosmos_transfer2._src.imaginaire.auxiliary.guardrail.video_content_safety_filter.vision_encoder'
    vision = ModuleType(vision_name)
    vision.SigLIPEncoder = object
    monkeypatch.setitem(sys.modules, vision_name, vision)
    transformers = ModuleType('transformers')
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common.core import GuardrailRunner
    modules = []
    for name in ['video_content_safety_filter/video_content_safety_filter.py', 'qwen3guard/qwen3guard.py']:
        path = PROJECT/'cosmos_transfer2/_src/imaginaire/auxiliary/guardrail'/name
        spec = importlib.util.spec_from_file_location('test_v3_'+path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return GuardrailRunner, modules[0].VideoContentSafetyFilter, modules[1].Qwen3Guard


def test_safety_classifier_failure_is_not_a_content_rejection(guardrail_classes):
    _, VideoContentSafetyFilter, _ = guardrail_classes
    classifier = VideoContentSafetyFilter.__new__(VideoContentSafetyFilter)
    offloads = []
    classifier._to_cuda_if_offload = lambda: None
    classifier._to_cpu_if_offload = lambda: offloads.append(True)
    classifier._VideoContentSafetyFilter__infer = lambda image: (_ for _ in ()).throw(RuntimeError('classifier failed'))
    with pytest.raises(RuntimeError, match='classifier failed'):
        classifier.is_safe_frames([np.zeros((8, 8, 3), dtype=np.uint8)])
    assert offloads == [True]


def test_safety_rejection_has_class_details_and_same_threshold(guardrail_classes):
    GuardrailRunner, VideoContentSafetyFilter, _ = guardrail_classes
    classifier = VideoContentSafetyFilter.__new__(VideoContentSafetyFilter)
    classifier._to_cuda_if_offload = lambda: None
    classifier._to_cpu_if_offload = lambda: None
    classifier._VideoContentSafetyFilter__infer = lambda image: 1
    runner = GuardrailRunner(safety_models=[classifier])
    assert not runner.run_safety_check([np.zeros((8, 8, 3), dtype=np.uint8)])[0]
    assert runner.last_result['details']['frame_scores'][0]['class'] == 'Sexual_Content'
    classifier._VideoContentSafetyFilter__infer = lambda image: 0
    assert runner.run_safety_check([np.zeros((8, 8, 3), dtype=np.uint8)])[0]
    assert runner.last_result is None


def test_text_classifier_errors_propagate(guardrail_classes):
    _, _, Qwen3Guard = guardrail_classes
    classifier = Qwen3Guard.__new__(Qwen3Guard)
    classifier.extract_label_and_categories = lambda prompt: (_ for _ in ()).throw(RuntimeError('text classifier failed'))
    with pytest.raises(RuntimeError, match='text classifier failed'):
        classifier.is_safe('robot scene')


@pytest.mark.parametrize('block_main,block_pair', [('specific', False), (False, True), ('all', True)])
def test_suite_exports_partial_pairs_or_all_blocked_without_fabricating_results(
        prompt_config, monkeypatch, tmp_path, block_main, block_pair):
    from robot_realism.config import load_config
    from robot_realism.dataset import source_records
    from robot_realism.prompt_suite import export_results, initialize
    config = prompt_config
    _, records = source_records(config)
    selection = read_json(config['prompting']['selection_manifest'])
    chosen = [r for r in records if r['state_id'] in selection['groups']['train_specific']]
    counts = {'train_general': 1, 'train_specific': 1, 'eval_general': 2}
    monkeypatch.setattr('robot_realism.prompt_suite.build_selection', lambda config: (selection, chosen))
    monkeypatch.setattr('robot_realism.prompt_suite.GROUP_COUNTS', counts)
    root = tmp_path/'suite'
    configs = initialize(root, config['input']['root'], config['runtime']['python'], config['runtime']['cache'])
    for variant, path in configs.items():
        current = load_config(path)
        frozen = plan(current)
        prepare(current)
        profile = make_profile(current, frozen)
        initialise_output(current, frozen, profile, False)
        runtime = {'test_only': True}
        atomic_json(Path(current['output']['root'])/'runtime_identity.json', runtime)
        jobs = read_json(Path(current['output']['run_dir'])/'prepared.json')['jobs']
        rejected = []
        for job in jobs:
            if variant == 'main':
                should_block = block_main == 'all' or (block_main == 'specific' and
                    frozen['prompt_assignments'][job['record']['state_id']]['group'] == 'train_specific')
            else:
                should_block = bool(block_pair)
            if should_block:
                rejected.append(job_name(job))
        process_jobs(current, frozen, profile, jobs, FixtureInference(rejected), runtime, SimpleNamespace)
        assert validate(current, frozen, allow_blocked=True)['valid']
    summary = export_results(root, configs)
    assert summary['execution_complete'] and not summary['dataset_complete']
    assert summary['planned_images'] == 5
    assert summary['generated_images']+summary['blocked_images'] == 5
    assert summary['comparison_pairs'] == 0
    assert summary['comparison_training_samples'] == (0 if block_main == 'all' else 1)
    assert (root/'reports/blocked.csv').is_file()
    assert len((root/'reports/blocked.jsonl').read_text().splitlines()) == summary['blocked_images']
    first = [json.loads(line) for line in (root/'manifests/train_all_general.jsonl').read_text().splitlines()]
    second = [json.loads(line) for line in (root/'manifests/train_mixed.jsonl').read_text().splitlines()]
    assert {r['state_id'] for r in first} == {r['state_id'] for r in second}


def test_v3_configuration_keeps_guardrails_and_new_root():
    from robot_realism.config import load_config
    config = load_config(PROJECT/'configs/realism/cosmos25_libero_agentview_prompt_v3.yaml')
    assert config['execution']['guardrail_policy'] == 'record_and_skip'
    assert config['runtime']['enable_guardrails']
    assert config['input']['cameras'] == ['agentview']
    assert config['output']['root'].startswith('/root/robot-realism/agentview_prompt_v3/')


def test_reuse_prompts_only_does_not_import_old_generated_images(prompt_config, monkeypatch, tmp_path):
    from robot_realism.dataset import source_records
    from robot_realism.prompt_suite import initialize
    config = prompt_config
    _, records = source_records(config)
    selection = read_json(config['prompting']['selection_manifest'])
    chosen = [r for r in records if r['state_id'] in selection['groups']['train_specific']]
    monkeypatch.setattr('robot_realism.prompt_suite.build_selection', lambda config: (selection, chosen))
    old, new = tmp_path/'v2', tmp_path/'v3'
    initialize(old, config['input']['root'], config['runtime']['python'], config['runtime']['cache'])
    old_image = old/'datasets/main/old.png'
    old_image.parent.mkdir(parents=True)
    old_image.write_bytes(b'OLD FIXTURE RESULT; MUST NOT BE IMPORTED')
    initialize(new, config['input']['root'], config['runtime']['python'], config['runtime']['cache'], reuse_from=old)
    assert (new/'prompts/specific.jsonl').read_bytes() == (old/'prompts/specific.jsonl').read_bytes()
    assert (new/'manifests/selection.json').read_bytes() == (old/'manifests/selection.json').read_bytes()
    assert not (new/'datasets').exists()
    assert old_image.exists()
