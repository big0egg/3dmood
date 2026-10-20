"""Build exactly 15k train / 3k eval, without touching the old benchmark."""
import collections
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (REPO, atomic_bytes, bind, checksum, data_path, digest, jsonlines,
                     read, readlines, source_stamp, write)
from .raw import MASKS, RawSequence, eligible_times, triangles


def balanced_select(rows, count, seed):
    """First cover assets/classes/sequences, then round-robin shuffled sequences."""
    if count > len(rows) or count < 1:
        raise ValueError(f'有效候选 {len(rows)}，无法选取 {count}')
    if len({r['frame_id'] for r in rows}) != len(rows):
        raise ValueError('候选存在重复帧')
    rng = random.Random(seed)
    shuffled = sorted(rows, key=lambda r: r['frame_id'])
    rng.shuffle(shuffled)
    chosen, seen, assets, classes, sequences = [], set(), set(), set(), set()

    def add(row):
        chosen.append(row)
        seen.add(row['frame_id'])
        assets.update(row.get('asset_ids', []))
        classes.update(row.get('class_ids', []))
        sequences.add(row['sequence_id'])

    for row in shuffled:
        if (set(row.get('asset_ids', [])) - assets or set(row.get('class_ids', [])) - classes
                or row['sequence_id'] not in sequences):
            add(row)
            if len(chosen) == count:
                return sorted(chosen, key=lambda r: r['frame_id'])
    groups = collections.defaultdict(list)
    for row in shuffled:
        if row['frame_id'] not in seen:
            groups[row['sequence_id']].append(row)
    keys = sorted(groups)
    while len(chosen) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(chosen) < count:
                add(groups[key].pop())
                progressed = True
        if not progressed:
            raise ValueError('抽帧池不足')
    return sorted(chosen, key=lambda r: r['frame_id'])


def audit_sources(cfg):
    sequences = []
    for directory in sorted((Path(cfg['source']) / 'Aria').iterdir()):
        if directory.name.split('_')[0] not in cfg['train_participants']:
            continue
        archive = next(directory.glob('*ground_truth.zip'), None)
        vrs = next(directory.glob('*main_recording.vrs'), None)
        if archive is None or vrs is None:
            raise FileNotFoundError(f'原始录制缺少 VRS 或 GT：{directory}')
        with zipfile.ZipFile(archive) as z:
            missing = [name for name in MASKS if 'masks/' + name not in z.namelist()]
            if missing:
                raise ValueError(f'{directory} 缺少质量标记：{missing}')
            times = eligible_times(z, cfg['candidate_interval_ns'])
        sequences.append(dict(sequence_id=directory.name, candidates=len(times),
                              gt=source_stamp(archive), gt_sha256=checksum(archive), vrs=source_stamp(vrs)))
    return dict(sequences=sequences, count=len(sequences),
                candidates=sum(s['candidates'] for s in sequences),
                train_count=cfg['train_count'], eval_count=cfg['eval_count'],
                selection_uses_predictions=False)


def valid_ids(boxes):
    valid = [b for b in boxes if not b['ignore']]
    return dict(asset_ids=sorted({b['object_bop_id'] for b in valid}),
                class_ids=sorted({b['class_id'] for b in valid}))


def prepare(cfg):
    from ioutest.hot3d.common import validate_box
    old = Path(cfg['existing_data'])
    root = data_path(cfg)
    old_protocol = read(old / 'protocol.json')
    if cfg['camera'] != old_protocol['settings']['camera']:
        raise ValueError('新旧虚拟相机配置不同，不能复用评测帧')
    if set(cfg['train_participants']) & set(cfg['eval_quotas']):
        raise ValueError('训练/评测参与者重叠')
    if sum(cfg['eval_quotas'].values()) != cfg['eval_count']:
        raise ValueError('评测参与者配额与总数不符')
    # Fail before lengthy extraction when the native VRS decoder is missing.
    import importlib.util
    if importlib.util.find_spec('projectaria_tools') is None:
        raise RuntimeError('缺少 projectaria-tools；先运行 setup_data_env.sh，或使用配置中的 data_python')
    audit = audit_sources(cfg)
    registry = read(old / 'object_models/registry.json')
    classes = registry['classes']
    if classes != old_protocol['classes']:
        raise ValueError('物体资产注册表与旧评测协议的类别映射不一致')
    identity = dict(settings={k: cfg[k] for k in ('dataset_name', 'source', 'existing_data', 'seed',
                    'train_participants', 'eval_quotas', 'train_count', 'eval_count',
                    'candidate_interval_ns', 'camera', 'minimum_amodal_pixels',
                    'minimum_native_modeled_visibility')}, sources=audit,
                    old_protocol=digest(old_protocol), registry=digest(registry),
                    code={**{name: checksum(Path(__file__).with_name(name)) for name in ('dataset.py', 'raw.py', 'common.py')},
                          **{name: checksum(REPO / name) for name in ('ioutest/geometry.py', 'ioutest/hot3d/camera.py',
                                                                     'ioutest/hot3d/common.py')}})
    signature = bind(root / 'preparation.json', identity)
    meshes = {}
    for bop, model in registry['objects'].items():
        path = old / 'object_models' / (model['uid'] + '.glb')
        if checksum(path) != model['sha256']:
            raise ValueError(f'物体模型已变化：{path}')
        meshes[bop] = triangles(path.read_bytes())
    candidates = []
    for index, source in enumerate(audit['sequences'], 1):
        sequence = source['sequence_id']
        cached = root / 'candidate_cache' / (sequence + '.json')
        if cached.exists():
            entry = read(cached)
            if entry['signature'] != signature:
                raise ValueError('候选缓存来源发生变化')
            rows = entry['rows']
        else:
            raw = RawSequence(Path(cfg['source']) / 'Aria' / sequence, cfg, registry, meshes)
            rows = []
            for number, timestamp in enumerate(raw.times, 1):
                boxes = raw.labels(timestamp)
                if boxes and any(not b['ignore'] for b in boxes):
                    for box in boxes:
                        validate_box(box, classes)
                    rows.append(dict(frame_id=f'{sequence}:{timestamp}', sequence_id=sequence,
                                     participant_id=sequence.split('_')[0], timestamp_ns=timestamp,
                                     intrinsics=raw.maps[3].tolist(), image_hw=[cfg['camera']['height'], cfg['camera']['width']],
                                     labels=boxes, **valid_ids(boxes)))
                if number % 50 == 0:
                    print(f'GT 准备 {index}/{audit["count"]} {sequence} {number}/{len(raw.times)}', flush=True)
                    write(Path(cfg['root']) / 'status.json', dict(stage='prepare_gt', sequence=sequence,
                          sequences_completed=index-1, candidates_checked=number, candidates_total=len(raw.times)))
            write(cached, dict(signature=signature, rows=rows))
            del raw
        candidates.extend(rows)
    selected = balanced_select(candidates, cfg['train_count'], cfg['seed'])
    train = []
    grouped = collections.defaultdict(list)
    for row in selected:
        grouped[row['sequence_id']].append(row)
    for index, (sequence, rows) in enumerate(sorted(grouped.items()), 1):
        raw = None
        for row in rows:
            timestamp = row['timestamp_ns']
            image_path = root / 'rgb_pinhole' / sequence / f'{timestamp}.png'
            gt_path = root / 'groundtruth' / sequence / f'{timestamp}.json'
            calibration_path = root / 'calibration' / sequence / f'{timestamp}.json'
            completed = root / 'completed_frames' / sequence / f'{timestamp}.json'
            if completed.exists():
                frame = read(completed)
                if (frame['preparation_signature'] != signature or
                        checksum(frame['rgb_path']) != frame['rgb_sha256'] or
                        digest(read(frame['gt_path'])) != frame['gt_sha256'] or
                        checksum(calibration_path) != frame['calibration_sha256']):
                    raise ValueError(f'已完成数据校验失败：{completed}')
            else:
                if raw is None:
                    raw = RawSequence(Path(cfg['source']) / 'Aria' / sequence, cfg, registry, meshes)
                array = raw.image(timestamp)
                stream = io.BytesIO()
                Image.fromarray(array).save(stream, format='PNG')
                atomic_bytes(image_path, stream.getvalue())
                write(gt_path, row['labels'])
                write(calibration_path, dict(raw=raw.calibration, device_from_camera=raw.device_from_camera.tolist(),
                      virtual_intrinsics=raw.maps[3].tolist(), virtual_from_raw_rotation=raw.maps[4].tolist(),
                      timestamp_ns=timestamp, device_time_ns=raw.device_times[timestamp]))
                frame = {k: v for k, v in row.items() if k != 'labels'}
                frame.update(clip_id=sequence, frame_key=str(timestamp), rgb_path=str(image_path),
                             rgb_sha256=checksum(image_path), gt_path=str(gt_path), gt_sha256=digest(row['labels']),
                             calibration_sha256=checksum(calibration_path), preparation_signature=signature)
                write(completed, frame)
            train.append(frame)
        print(f'RGB 准备 {index}/{len(grouped)}，训练帧 {len(train)}/{cfg["train_count"]}', flush=True)
        write(Path(cfg['root'])/'status.json', dict(stage='prepare_rgb', completed_frames=len(train),
                                                   expected_frames=cfg['train_count'], sequence=sequence))
        del raw
    old_gt = {}
    evaluation = []
    old_frames = readlines(old / 'frames.jsonl')
    for participant, quota in sorted(cfg['eval_quotas'].items()):
        pool = []
        for frame in old_frames:
            if frame['participant_id'] != participant:
                continue
            clip = frame['clip_id']
            if clip not in old_gt:
                old_gt[clip] = read(old / 'groundtruth' / (clip + '.json'))
            boxes = old_gt[clip][frame['frame_id']]
            pool.append(dict(frame, **valid_ids(boxes)))
        for frame in balanced_select(pool, quota, cfg['seed']):
            boxes = old_gt[frame['clip_id']][frame['frame_id']]
            label_path = root / 'groundtruth' / frame['clip_id'] / (frame['frame_key'] + '.json')
            write(label_path, boxes)
            frame.update(rgb_path=str(old / frame['rgb_relative']), gt_path=str(label_path),
                         gt_sha256=digest(boxes), preparation_signature=signature)
            if checksum(frame['rgb_path']) != frame['rgb_sha256']:
                raise ValueError('原评测图片摘要不匹配')
            evaluation.append(frame)
    train.sort(key=lambda f: f['frame_id'])
    evaluation.sort(key=lambda f: f['frame_id'])
    jsonlines(root / 'train.jsonl', train)
    jsonlines(root / 'eval.jsonl', evaluation)
    protocol = dict(schema='hot3d.box3d.v1', classes=classes, preparation_signature=signature,
                    train_sha256=digest(train), eval_sha256=digest(evaluation),
                    training_validity='official QA/exposure/pose masks, native visibility, CPU mesh amodal rasterization',
                    evaluation_validity='unchanged original benchmark GT/ignore flags',
                    model_selection='fixed epoch 4; no evaluation-based checkpoint selection',
                    benchmark_id=digest([signature, digest(train), digest(evaluation)]))
    write(root / 'protocol.json', protocol)
    result = verify(cfg, images=True)
    write(root / 'dataset_audit.json', result)
    write(Path(cfg['root'])/'status.json', dict(stage='data_ready', train_frames=len(train),
                                               eval_frames=len(evaluation), benchmark_id=protocol['benchmark_id']))
    return result


def load(cfg):
    root = data_path(cfg)
    protocol = read(root / 'protocol.json')
    train, evaluation = readlines(root / 'train.jsonl'), readlines(root / 'eval.jsonl')
    if digest(train) != protocol['train_sha256'] or digest(evaluation) != protocol['eval_sha256']:
        raise ValueError('冻结数据清单发生变化')
    return protocol, train, evaluation


def verify(cfg, images=False):
    from ioutest.hot3d.common import validate_box
    protocol, train, evaluation = load(cfg)
    if len(train) != cfg['train_count'] or len(evaluation) != cfg['eval_count']:
        raise ValueError('训练/评测数量不符合配置')
    if set(f['participant_id'] for f in train) - set(cfg['train_participants']):
        raise ValueError('训练混入非训练参与者')
    if collections.Counter(f['participant_id'] for f in evaluation) != cfg['eval_quotas']:
        raise ValueError('评测参与者配额不符')
    for field in ('frame_id', 'sequence_id', 'participant_id', 'rgb_sha256'):
        if set(f[field] for f in train) & set(f[field] for f in evaluation):
            raise ValueError(f'训练/评测重叠：{field}')
    audit = dict(complete=True, benchmark_id=protocol['benchmark_id'])
    for name, rows in [('train', train), ('eval', evaluation)]:
        if len({f['frame_id'] for f in rows}) != len(rows):
            raise ValueError('存在重复时间戳帧')
        classes, objects, targets = set(), set(), 0
        for frame in rows:
            boxes = read(frame['gt_path'])
            if digest(boxes) != frame['gt_sha256']:
                raise ValueError(f'GT 已变化：{frame["gt_path"]}')
            for box in boxes:
                validate_box(box, protocol['classes'])
                if not box['ignore']:
                    classes.add(box['class_id']); objects.add(box['object_bop_id']); targets += 1
            if images:
                if checksum(frame['rgb_path']) != frame['rgb_sha256']:
                    raise ValueError(f'图像已变化：{frame["rgb_path"]}')
                with Image.open(frame['rgb_path']) as im:
                    if [im.height, im.width] != frame['image_hw']:
                        raise ValueError('图像尺寸错误')
                    im.verify()
        if classes != set(range(len(protocol['classes']))) or objects != set(range(1, 34)):
            raise ValueError(f'{name} 未覆盖全部25类/33个资产')
        audit[name] = dict(frames=len(rows), valid_targets=targets, classes=len(classes), objects=len(objects),
                           sequences=len({f['sequence_id'] for f in rows}),
                           participants=dict(collections.Counter(f['participant_id'] for f in rows)))
    return audit
