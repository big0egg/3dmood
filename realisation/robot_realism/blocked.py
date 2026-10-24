"""Durable terminal states for rejected frames, separate from successful commits."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import atomic_json, digest, file_hash, read_json, safe_child
from .dataset import bundle, verify_source


def skip_enabled(config):
    return config.get('execution', {}).get('guardrail_policy', 'stop') == 'record_and_skip'


def blocked_path(config, record, camera):
    return safe_child(config['output']['run_dir'], f'blocked/{record["state_id"]}_{camera}.json')


def block_context(config, frozen, profile_id, record, camera):
    source = bundle(config['input']['root'], record)
    prompt = frozen.get('prompt_assignments', {}).get(record['state_id'], {}).get(
        'prompt', config['generation']['prompt'])
    return {'plan_id': frozen['plan_id'], 'config_id': frozen['config_id'], 'profile_id': profile_id,
            'record': record, 'camera': camera,
            'source_commit_sha256': file_hash(source/'COMMIT.json'),
            'prompt_sha256': digest({'positive': prompt, 'negative': config['generation']['negative_prompt']}),
            'seed': int(digest([config['selection']['seed'], record['state_id']])[:8], 16)}


def read_blocked(config, frozen, profile_id, record, camera):
    path = blocked_path(config, record, camera)
    if not path.exists():
        return None
    if not skip_enabled(config):
        raise ValueError('Blocked records require the record_and_skip execution policy')
    value = read_json(path)
    checksum = value.pop('record_sha256', None)
    if checksum != digest(value) or value.get('schema_version') != 'robot_realism.blocked.v1':
        raise ValueError(f'Blocked record is corrupt: {path}')
    value['record_sha256'] = checksum
    if value['context'] != block_context(config, frozen, profile_id, record, camera):
        raise ValueError(f'Blocked record belongs to another input/configuration: {path}')
    if (value.get('status') != 'blocked' or value.get('stage') not in
            ('positive_prompt', 'negative_prompt', 'output') or not value.get('reason')):
        raise ValueError(f'Invalid rejection metadata: {path}')
    runtime = Path(config['output']['root'])/'runtime_identity.json'
    if not runtime.is_file() or digest(read_json(runtime)) != value.get('runtime_id'):
        raise ValueError(f'Blocked runtime identity differs: {path}')
    verify_source(config['input']['root'], record, [camera])
    if (bundle(config['output']['root'], record)/'COMMIT.json').exists():
        raise ValueError(f'State has both a successful commit and a blocked record: {record["state_id"]}')
    return value


def record_blocked(config, frozen, profile_id, job, view, error, runtime):
    from cosmos_transfer2.guardrail_errors import GuardrailBlockedError
    if not skip_enabled(config) or not isinstance(error, GuardrailBlockedError):
        raise ValueError('Only explicit guardrail rejections may be recorded as blocked')
    if frozen['cameras'] != ['agentview'] or len(job['views']) != 1:
        raise ValueError('Blocked-frame handling requires one agentview camera')
    record, camera = job['record'], view['camera']
    existing = read_blocked(config, frozen, profile_id, record, camera)
    if existing:
        return existing
    source = bundle(config['input']['root'], record)/camera
    spec = read_json(view['spec'])
    if file_hash(view['spec']) != view['spec_sha256']:
        raise ValueError('Blocked inference spec changed')
    for key, checksum in view['controls_sha256'].items():
        if file_hash(spec[key]['control_path']) != checksum:
            raise ValueError('Blocked control file changed')
    if file_hash(bundle(config['input']['root'], record)/'COMMIT.json') != job['source_commit_sha256']:
        raise ValueError('Blocked source changed after preparation')
    runtime_path = Path(config['output']['root'])/'runtime_identity.json'
    if not runtime_path.is_file() or read_json(runtime_path) != runtime:
        raise ValueError('Missing or changed runtime identity before blocked record')
    value = {'schema_version': 'robot_realism.blocked.v1', 'status': 'blocked',
        'state_id': record['state_id'], 'camera': camera, 'split': record['split'],
        'scene_id': record['scene_id'], 'group': frozen.get('prompt_assignments', {}).get(
            record['state_id'], {}).get('group', record['split']),
        'context': block_context(config, frozen, profile_id, record, camera),
        'stage': error.stage, 'reason': error.reason, 'details': error.details,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'resolution_status': 'pending_manual_review',
        'inference': spec, 'control_files_sha256': view['controls_sha256'],
        'source_rgb_path': str(source/'rgb.png'), 'source_depth_path': str(source/'depth.npz'),
        'source_annotation_path': str(source/'annotation.json'),
        'runtime_id': digest(runtime)}
    value['record_sha256'] = digest(value)
    atomic_json(blocked_path(config, record, camera), value)
    return read_blocked(config, frozen, profile_id, record, camera)


class ProgressTracker:
    """Scan once on startup; update counters without rehashing every prior frame."""
    def __init__(self, config, frozen, profile_id):
        from .pipeline import verify_result
        self.config, self.frozen, self.profile_id = config, frozen, profile_id
        self.accepted, self.blocked = {}, {}
        expected = {blocked_path(config, r, c) for r in frozen['records'] for c in frozen['cameras']}
        if set((Path(config['output']['run_dir'])/'blocked').glob('*.json')) - expected:
            raise ValueError('Foreign blocked records found in run directory')
        for record in frozen['records']:
            commit = verify_result(config['output']['root'], record, frozen['cameras'], profile_id)
            blocks = [read_blocked(config, frozen, profile_id, record, c) for c in frozen['cameras']]
            blocks = [b for b in blocks if b]
            if commit:
                self.accepted[record['state_id']] = record
            elif blocks:
                self.blocked[record['state_id']] = blocks[0]

    def is_done(self, record):
        return record['state_id'] in self.accepted or record['state_id'] in self.blocked

    def successful(self, record):
        from .pipeline import verify_result
        if record['state_id'] in self.blocked:
            raise ValueError('Cannot mark a blocked state successful')
        if not verify_result(self.config['output']['root'], record, self.frozen['cameras'], self.profile_id):
            raise ValueError('Cannot count a state before its successful commit')
        self.accepted[record['state_id']] = record

    def rejected(self, record):
        value = read_blocked(self.config, self.frozen, self.profile_id, record, 'agentview')
        if value is None or record['state_id'] in self.accepted:
            raise ValueError('Cannot count an unrecorded rejection or a successful state')
        self.blocked[record['state_id']] = value
        from .prompting import write_jsonl
        write_jsonl(Path(self.config['output']['run_dir'])/'blocked.jsonl', self.blocked.values())

    def summary(self, error=None):
        cameras = len(self.frozen['cameras'])
        counts = Counter(r['split'] for r in self.accepted.values())
        counts = {k: v*cameras for k, v in counts.items()}
        blocked_counts = dict(Counter(v['split'] for v in self.blocked.values()))
        generated = len(self.accepted)*cameras
        blocked = len(self.blocked)
        planned = len(self.frozen['records'])*cameras
        pending = planned-generated-blocked
        status = ('failed' if error else 'complete' if generated == planned else
                  'finished_with_blocks' if pending == 0 else 'partial')
        value = {'plan_id': self.frozen['plan_id'], 'profile_id': self.profile_id,
            'status': status, 'complete': generated == planned, 'dataset_complete': generated == planned,
            'execution_complete': pending == 0 and not error, 'planned_images': planned,
            'accepted_images': generated, 'generated_images': generated,
            'blocked_images': blocked, 'pending_images': pending, 'counts': counts,
            'blocked_counts': blocked_counts, 'targets': self.frozen['targets'], 'last_error': error,
            'gt_status': 'inherited_simulation_gt_requires_visual_review'}
        if 'prompt_assignments' in self.frozen:
            assignments = self.frozen['prompt_assignments']
            value['prompt_groups'] = dict(Counter(assignments[s]['group'] for s in self.accepted))
            value['blocked_prompt_groups'] = dict(Counter(v['group'] for v in self.blocked.values()))
            value['target_prompt_groups'] = self.frozen['prompt_groups']
        return value

    def flush(self, error=None, full=False):
        value = self.summary(error)
        atomic_json(Path(self.config['output']['run_dir'])/'progress.json', value)
        if full:
            from .prompting import write_jsonl
            write_jsonl(Path(self.config['output']['run_dir'])/'blocked.jsonl', self.blocked.values())
            value = dict(value, accepted=self.accepted, blocked=self.blocked)
            atomic_json(Path(self.config['output']['root'])/'manifest.json', value)
        return value
