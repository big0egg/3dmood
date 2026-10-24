"""Frozen, source-grounded prompt assignments for the agentview experiment."""
from collections import Counter, defaultdict, deque
from pathlib import Path
import json
import random

import numpy as np
from PIL import Image

from .common import digest, file_hash, read_json
from .dataset import bundle, source_records, verify_source


GENERIC_PROMPT = (
    "A photorealistic RGB image of an indoor robot manipulation scene, recorded by a fixed external "
    "camera for a robotics dataset. The visible robot arm, gripper, objects, furniture, and background "
    "have physically plausible real-world materials and surface textures. The scene has natural indoor "
    "illumination, realistic contact shadows, and restrained reflections. The entire manipulation "
    "workspace is in focus. Follow the spatial structure of the provided depth control. Preserve the "
    "robot configuration, gripper opening, object shapes, object locations, relative scale, camera "
    "viewpoint, and occlusion order. Keep the existing scene layout and visible object arrangement. "
    "Render the current static state with realistic surface appearance."
)
NEGATIVE_PROMPT = (
    "cartoon, illustration, stylized rendering, obvious CGI, plastic-looking surfaces, exaggerated "
    "reflections, cinematic lighting, shallow depth of field, motion blur, distorted geometry, "
    "deformed robot arm, duplicated gripper, extra robot arms, added objects, missing objects, "
    "floating objects, changed camera viewpoint, watermark, overlaid text"
)
GROUP_COUNTS = {'train_general': 4800, 'train_specific': 200, 'eval_general': 500}
ROOM_QUOTAS = {'KITCHEN': 100, 'LIVING_ROOM': 60, 'STUDY': 40}
OBJECT_NAMES = {
    'chefmate_8_frypan': 'frying pan', 'flat_stove': 'stovetop', 'desk_caddy': 'desk organizer',
    'living_room_table': 'table', 'study_table': 'desk', 'white_cabinet': 'cabinet',
    'butter': 'butter package', 'cream_cheese': 'cream cheese package',
    'chocolate_pudding': 'pudding package', 'alphabet_soup': 'soup container',
    'tomato_sauce': 'tomato sauce container', 'ketchup': 'ketchup container',
    'orange_juice': 'orange juice container', 'milk': 'milk container',
    'salad_dressing': 'salad dressing container', 'new_salad_dressing': 'salad dressing container',
    'moka_pot': 'stovetop coffee maker',
}


def write_jsonl(path, rows):
    from .common import atomic_text
    atomic_text(path, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def room(record):
    value = record.get('room')
    if value in ROOM_QUOTAS:
        return value
    for value in ROOM_QUOTAS:
        if record['scene_id'].startswith(value + '_'):
            return value
    raise ValueError(f"Unknown room: {record['scene_id']}")


def record_identity(info, records):
    return digest({'plan_id': info['plan_id'], 'profile_id': info['profile_id'],
                   'records': sorted(records, key=lambda r: r['state_id'])})


def balanced_specific(records, quotas, seed):
    """Balance rooms/scenes and spread each scene's choices across episodes."""
    selected = []
    for name, quota in sorted(quotas.items()):
        scenes = defaultdict(lambda: defaultdict(list))
        for record in records:
            if record['split'] == 'train' and room(record) == name:
                scenes[record['scene_id']][(record['source'], record['episode'])].append(record)
        queues = {}
        for scene, episodes in sorted(scenes.items()):
            episode_queues = []
            for episode, items in sorted(episodes.items()):
                items = sorted(items, key=lambda r: r['state_id'])
                random.Random(f'{seed}:{scene}:{episode}').shuffle(items)
                episode_queues.append(deque(items))
            random.Random(f'{seed}:{scene}:episodes').shuffle(episode_queues)
            ordered = []
            while any(episode_queues):
                for queue in episode_queues:
                    if queue:
                        ordered.append(queue.popleft())
            queues[scene] = deque(ordered)
        if sum(map(len, queues.values())) < quota:
            raise ValueError(f'Insufficient {name} training states for {quota} specific prompts')
        for _ in range(quota):
            # A scene with fewer selections is visited first; insertion order is fixed.
            scene = min((s for s in queues if queues[s]),
                        key=lambda s: (sum(r['scene_id'] == s for r in selected), s))
            selected.append(queues[scene].popleft())
    return sorted(selected, key=lambda r: r['state_id'])


def build_selection(config):
    info, records = source_records(config)
    if config['input']['cameras'] != ['agentview']:
        raise ValueError('The prompt experiment requires only agentview')
    if len({r['state_id'] for r in records}) != len(records):
        raise ValueError('Duplicate source state IDs')
    if Counter(r['split'] for r in records) != Counter(train=5000, eval=500):
        raise ValueError('Full source required: exactly 5000 train and 500 eval states')
    if {r['scene_id'] for r in records if r['split'] == 'train'} & {
            r['scene_id'] for r in records if r['split'] == 'eval'}:
        raise ValueError('Source train/eval scenes overlap')
    for split, expected in [('train', {'KITCHEN': 2500, 'LIVING_ROOM': 1500, 'STUDY': 1000}),
                            ('eval', {'KITCHEN': 250, 'LIVING_ROOM': 125, 'STUDY': 125})]:
        if Counter(room(r) for r in records if r['split'] == split) != Counter(expected):
            raise ValueError(f'Unexpected source room quotas for {split}')
    chosen = balanced_specific(records, ROOM_QUOTAS, config['selection']['seed'])
    specific_ids = {r['state_id'] for r in chosen}
    groups = {name: [] for name in GROUP_COUNTS}
    for record in records:
        group = ('eval_general' if record['split'] == 'eval' else
                 'train_specific' if record['state_id'] in specific_ids else 'train_general')
        groups[group].append(record['state_id'])
    for values in groups.values():
        values.sort()
    value = {'schema_version': 1, 'source_id': record_identity(info, records),
             'seed': config['selection']['seed'], 'camera': 'agentview',
             'groups': groups, 'specific_room_quotas': ROOM_QUOTAS}
    value['selection_id'] = digest(value)
    return value, chosen


def read_selection(path):
    value = read_json(path)
    identity = value.pop('selection_id')
    if digest(value) != identity:
        raise ValueError('Selection manifest was modified')
    value['selection_id'] = identity
    return value


def visible_objects(annotation):
    # Child parts are never counted again as complete objects.
    return [o for o in annotation['objects'] if o.get('kind') == 'object'
            and o.get('parent_instance_id') is None and o.get('visible')
            and o.get('visible_pixels', 0) >= 32 and o.get('bbox_visible_xyxy')]


def observed_color(rgb, instance, obj):
    ids = obj.get('segmentation_ids', [obj['instance_id']])
    pixels = rgb[np.isin(instance, ids)].astype(float) / 255
    if len(pixels) < 32:
        return None
    maximum, minimum = pixels.max(1), pixels.min(1)
    delta = maximum - minimum
    colors = np.full(len(pixels), 'unknown', dtype=object)
    colors[maximum < .20] = 'black'
    neutral = delta < .12
    colors[neutral & (maximum >= .20)] = 'gray'
    colors[neutral & (minimum > .72)] = 'white'
    hue = np.zeros(len(pixels))
    valid = delta > .12
    for channel in range(3):
        mask = valid & (pixels.argmax(1) == channel)
        if channel == 0:
            hue[mask] = ((pixels[mask, 1] - pixels[mask, 2]) / delta[mask]) % 6
        elif channel == 1:
            hue[mask] = (pixels[mask, 2] - pixels[mask, 0]) / delta[mask] + 2
        else:
            hue[mask] = (pixels[mask, 0] - pixels[mask, 1]) / delta[mask] + 4
    hue *= 60
    chromatic = valid & (maximum >= .20)
    for name, mask in [('red', (hue < 20) | (hue >= 340)),
                       ('orange', (hue >= 20) & (hue < 45)),
                       ('yellow', (hue >= 45) & (hue < 75)),
                       ('green', (hue >= 75) & (hue < 165)),
                       ('blue', (hue >= 165) & (hue < 265)),
                       ('purple', (hue >= 265) & (hue < 340))]:
        colors[chromatic & mask] = name
    name, count = Counter(colors).most_common(1)[0]
    return name if name != 'unknown' and count / len(pixels) >= .65 else None


def describe_frame(rgb, instance, annotation):
    """Describe observed semantics, color and image positions; never infer an action."""
    h, w = rgb.shape[:2]
    descriptions, evidence = [], []
    objects = sorted(visible_objects(annotation),
                     key=lambda o: (o['bbox_visible_xyxy'][0], o['instance_id']))
    for obj in objects:
        raw = obj.get('raw_category', '')
        label = OBJECT_NAMES.get(obj['category'], obj['category'].replace('_', ' '))
        color = observed_color(rgb, instance, obj)
        # Named colors are only used if the actual masked RGB supports them.
        tokens = set(raw.lower().split('_'))
        appearance = [color] if color else []
        if 'wooden' in tokens:
            appearance.append('wooden')
        if 'porcelain' in tokens:
            appearance.append('porcelain')
        phrase = ' '.join([*appearance, label])
        x0, y0, x1, y1 = obj['bbox_visible_xyxy']
        x, y = (x0 + x1) / (2*w), (y0 + y1) / (2*h)
        horizontal = 'left' if x < 1/3 else 'right' if x > 2/3 else 'central'
        vertical = 'upper' if y < 1/3 else 'lower' if y > 2/3 else 'middle'
        descriptions.append(f'a visible {phrase} in the {vertical} {horizontal} area '
                            f'(center near {x*100:.1f}% of image width, {y*100:.1f}% of image height)')
        evidence.append({'instance_id': obj['instance_id'], 'category': obj['category'],
                         'raw_category': raw, 'observed_color': color,
                         'bbox_visible_xyxy': obj['bbox_visible_xyxy'],
                         'visible_pixels': obj['visible_pixels']})
    if not descriptions:
        raise ValueError('No reliably visible root objects for a specific prompt')
    robot_positions = []
    for obj in annotation['objects']:
        if obj.get('kind') == 'robot' and obj.get('visible'):
            ys, xs = np.nonzero(np.isin(instance, obj.get('segmentation_ids', [obj['instance_id']])))
            if len(xs) >= 32:
                robot_positions.append(f'{(xs.mean()+.5)/w*100:.1f}% of image width and '
                                       f'{(ys.mean()+.5)/h*100:.1f}% of image height')
    robot_text = (' The visible robot silhouette is centered near ' + ', '.join(robot_positions) + '.') if robot_positions else ''
    return ('The visible scene includes ' + '; '.join(descriptions) + '.' + robot_text + ' '
            'Keep each described instance in its observed image position and preserve the visible '
            'robot and gripper configuration. Use plausible real-world surface detail appropriate '
            'to these objects while following the depth-defined geometry.'), evidence


def build_specific_row(config, record, generic):
    verify_source(config['input']['root'], record, ['agentview'])
    source = bundle(config['input']['root'], record) / 'agentview'
    annotation = read_json(source / 'annotation.json')
    with Image.open(source / 'rgb.png') as image:
        rgb = np.asarray(image.convert('RGB'))
    with Image.open(source / 'instance.png') as image:
        instance = np.asarray(image)
    if instance.shape != rgb.shape[:2]:
        raise ValueError('RGB and instance mask shape mismatch')
    description, evidence = describe_frame(rgb, instance, annotation)
    return {'state_id': record['state_id'], 'camera': 'agentview',
            'source_rgb_sha256': file_hash(source / 'rgb.png'),
            'source_annotation_sha256': file_hash(source / 'annotation.json'),
            'source_instance_sha256': file_hash(source / 'instance.png'),
            'prompt': generic + '\n\n' + description, 'scene_description': description,
            'prompt_version': 'grounded-visible-instances-v1',
            'authoring_method': 'deterministic_RGB_mask_and_GT_description',
            'visual_review_status': 'pending', 'visible_evidence': evidence}


def prompt_inputs(config):
    policy = config.get('prompting')
    if not policy:
        return {}
    return {key: file_hash(policy[key]) for key in ('selection_manifest', 'specific_manifest')}


def resolve_assignments(config, info, records):
    policy = config.get('prompting')
    if not policy:
        return records, {}, {}
    selected = read_selection(policy['selection_manifest'])
    if selected['camera'] != 'agentview' or config['input']['cameras'] != ['agentview']:
        raise ValueError('Prompt manifests require agentview')
    if selected['seed'] != config['selection']['seed']:
        raise ValueError('Selection seed differs')
    if selected['source_id'] != record_identity(info, records):
        raise ValueError('Selection source differs from current source dataset')
    groups = selected['groups']
    if {k: len(v) for k, v in groups.items()} != GROUP_COUNTS:
        raise ValueError('Expected prompt groups 4800/200/500')
    ids = [s for group in groups.values() for s in group]
    lookup = {r['state_id']: r for r in records}
    if len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError('Prompt selection has duplicate, missing or foreign states')
    for name, values in groups.items():
        expected = 'eval' if name == 'eval_general' else 'train'
        if any(lookup[s]['split'] != expected for s in values):
            raise ValueError('Prompt groups must retain source train/eval split')
    rows = [json.loads(line) for line in Path(policy['specific_manifest']).read_text().splitlines() if line]
    specific = {r['state_id']: r for r in rows}
    if len(rows) != len(specific) or set(specific) != set(groups['train_specific']):
        raise ValueError('Exactly one specific prompt per selected training state is required')
    for state_id, row in specific.items():
        if row.get('camera') != 'agentview' or not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError(f'Invalid specific prompt: {state_id}')
        source = bundle(config['input']['root'], lookup[state_id]) / 'agentview'
        for filename, field in [('rgb.png', 'source_rgb_sha256'),
                                ('annotation.json', 'source_annotation_sha256'),
                                ('instance.png', 'source_instance_sha256')]:
            if file_hash(source / filename) != row.get(field):
                raise ValueError(f'Specific prompt source changed: {state_id}/{filename}')
    assignments = {}
    for group, state_ids in groups.items():
        for state_id in state_ids:
            assignments[state_id] = {'group': group,
                'prompt': specific[state_id]['prompt'] if group == 'train_specific' and policy['mode'] == 'mixed'
                          else config['generation']['prompt']}
            if state_id in specific:
                assignments[state_id]['source_files_sha256'] = {
                    'rgb.png': specific[state_id]['source_rgb_sha256'],
                    'annotation.json': specific[state_id]['source_annotation_sha256'],
                    'instance.png': specific[state_id]['source_instance_sha256']}
    if policy['mode'] == 'paired_general':
        records = [r for r in records if r['state_id'] in specific]
        assignments = {r['state_id']: {**assignments[r['state_id']], 'group': 'paired_general',
                                       'prompt': config['generation']['prompt']} for r in records}
    return records, assignments, prompt_inputs(config)
