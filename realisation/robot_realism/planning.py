from collections import defaultdict, deque
from pathlib import Path
import random

from .common import atomic_json, digest, read_json
from .config import config_id
from .dataset import source_records


def plan(config):
    info, records = source_records(config)
    identity = config_id(config)
    source = digest({'plan_id': info['plan_id'], 'profile_id': info['profile_id'],
                     'accepted': sorted(records, key=lambda r: r['state_id'])})
    path = Path(config['output']['run_dir']) / 'plan.json'
    if path.exists():
        previous = read_json(path)
        if previous['config_id'] != identity or previous['source_id'] != source:
            raise ValueError('Existing realism plan differs; use original config or a new output and run_dir')
        return load_plan(config)
    groups = defaultdict(list)
    for record in records:
        groups[record['scene_id']].append(record)
    queues = {}
    for name, items in sorted(groups.items()):
        items.sort(key=lambda r: r['state_id'])
        random.Random(f"{config['selection']['seed']}:{name}").shuffle(items)
        queues[name] = deque(items)
    ordered = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                ordered.append(queue.popleft())
    value = {'schema_version': 1, 'config_id': identity, 'source_id': source,
             'source_plan_id': info['plan_id'], 'source_profile_id': info['profile_id'],
             'cameras': config['input']['cameras'], 'records': ordered,
             'targets': {split: sum(r['split'] == split for r in records) * len(config['input']['cameras'])
                         for split in ('train', 'eval')}}
    value['plan_id'] = digest(value)
    atomic_json(path, value)
    return value


def load_plan(config):
    value = read_json(Path(config['output']['run_dir']) / 'plan.json')
    identity = value.pop('plan_id')
    if digest(value) != identity:
        raise ValueError('Frozen realism plan was modified')
    value['plan_id'] = identity
    if value['config_id'] != config_id(config):
        raise ValueError('Configuration differs from realism plan; use original config or a new version')
    return value


def select(plan, limit):
    cameras = len(plan['cameras'])
    if limit is not None and (limit < 1 or limit % cameras):
        raise ValueError(f'--limit counts total camera images and must be positive/divisible by {cameras}')
    return plan['records'][:None if limit is None else limit // cameras]
