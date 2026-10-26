"""Read existing prepared artifacts only; never extract, sample or render data."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .common import digest, file_digest, read_json, read_jsonl
from .geometry import projected_box
from .hot3d.common import validate_box

def path_at(root, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(root) / p).resolve()

def class_table(values):
    if not values: raise ValueError('必须提供冻结类别表，不能从每帧 GT 构造提示')
    if any(isinstance(v, dict) and v.get('class_id', i) != i for i, v in enumerate(values)):
        raise ValueError('类别 ID 必须按冻结顺序连续编号')
    out = [dict(class_id=i, name=v if isinstance(v, str) else v['name'],
        prompt=v.replace('_', ' ') if isinstance(v, str) else v.get('prompt', v['name'].replace('_', ' ')))
        for i, v in enumerate(values)]
    if len({v['name'] for v in out}) != len(out): raise ValueError('类别表重复')
    return out

def normal_frame(row, root):
    f = {k: row[k] for k in ('frame_id', 'image_hw', 'intrinsics')}
    f['rgb_path'] = str(path_at(root, row.get('rgb_path', row.get('rgb_relative', ''))))
    f['rgb_sha256'] = row.get('rgb_sha256')
    for key in ('state_id', 'scene_id', 'sequence_id', 'participant_id', 'episode', 'source', 'split', 'timestamp_ns'):
        if key in row: f[key] = row[key]
    f['group_id'] = str(row.get('scene_id', row.get('sequence_id', row.get('visit_id', row.get('video_id', f['frame_id'])))))
    return f

def hot3d(spec):
    root = Path(spec['root']); protocol = read_json(spec['protocol'])
    manifest = Path(spec['manifest']); rows = read_jsonl(manifest)
    checksum = protocol.get('eval_sha256') if manifest.name == 'eval.jsonl' else protocol.get('frames_sha256')
    if checksum and file_digest(manifest) != checksum and digest(rows) != checksum:
        raise ValueError('HOT3D 冻结清单摘要变化')
    classes = class_table(protocol['classes']); frames = []; gt = {}; grouped = {}
    for row in rows:
        f = normal_frame(row, root)
        if 'gt_path' in row:
            p = path_at(root, row['gt_path'])
            boxes = read_json(p)
            if row.get('gt_sha256') and file_digest(p) != row['gt_sha256'] and digest(boxes) != row['gt_sha256']:
                raise ValueError('HOT3D GT 摘要变化：' + str(p))
        else:
            clip = row['clip_id']
            if clip not in grouped: grouped[clip] = read_json(root / 'groundtruth' / (clip + '.json'))
            boxes = grouped[clip][row['frame_id']]
        frames.append(f); gt[f['frame_id']] = boxes
    if 'groundtruth_sha256' in protocol and digest(gt) != protocol['groundtruth_sha256']:
        raise ValueError('HOT3D GT 内容摘要变化')
    return classes, frames, gt, dict(protocol=protocol.get('benchmark_id'),
        label_status=protocol.get('label_status', 'provided_groundtruth'))

def arkit(spec):
    root = Path(spec['root']); protocol = read_json(spec['protocol'])
    if not protocol.get('complete'): raise ValueError('已处理数据不完整；请在数据制作模块修复')
    rows = read_jsonl(spec['manifest'])
    if file_digest(spec['manifest']) != protocol['frames_sha256']: raise ValueError('ARKit 帧清单变化')
    audit_path = root / 'dataset_audit.json'
    if file_digest(audit_path) != protocol['audit_sha256']: raise ValueError('ARKit 审计文件变化')
    audit = read_json(audit_path); gt = {}
    for scan in audit['scans']:
        p = root / 'groundtruth' / (scan['video_id'] + '.json')
        if file_digest(p) != scan['groundtruth_file_sha256']: raise ValueError('ARKit GT 摘要变化')
        for item in read_json(p)['frames']: gt[item['frame']['frame_id']] = item['boxes']
    cache = Path(spec['image_cache']); summary = read_json(cache / 'cache_summary.json')
    if not summary.get('complete') or summary['benchmark_id'] != protocol['benchmark_id']:
        raise ValueError('必须提供完整的、属于当前数据集的本地 RGB 缓存')
    if file_digest(cache / 'images.jsonl') != summary['images_manifest_sha256']: raise ValueError('RGB 缓存清单变化')
    cached = read_jsonl(cache / 'images.jsonl'); images = {r['frame_id']: r for r in cached}
    if len(images) != len(cached) or set(images) != {r['frame_id'] for r in rows}: raise ValueError('RGB 清单不一致')
    from .schema import CLASSES
    frames = []
    for row in rows:
        f = normal_frame(dict(row, rgb_path=str(path_at(cache, images[row['frame_id']]['raw_image']))), root)
        f.update(rotation_k=row['rotation_k'], raw_image_hw=row['raw_image_hw'], raw_rgb_pixels_sha256=row['rgb_pixels_sha256'])
        if images[row['frame_id']].get('raw_rgb_pixels_sha256') != row['rgb_pixels_sha256']:
            raise ValueError('ARKit 缓存像素摘要不一致')
        frames.append(f)
    return class_table(CLASSES), frames, gt, dict(protocol=protocol['benchmark_id'], label_status='provided_groundtruth')

def simulation(spec):
    root = Path(spec['root']); categories = spec.get('classes')
    if not categories:
        categories = [v for v in read_json(spec['categories_file'])['categories'] if 'object' in v['kinds']]
    classes = class_table(categories); lookup = {v['name']: i for i, v in enumerate(classes)}
    if spec['format'] == 'robot_synth_prepared':
        accepted = read_json(spec['manifest'])['accepted']
        rows = list(accepted.values()) if isinstance(accepted, dict) else accepted
        camera = spec.get('camera', 'agentview')
        rows = [dict(r, camera=camera,
            annotation_path=str(root / r['split'] / r['scene_id'] / r['state_id'] / camera / 'annotation.json'),
            rgb_path=str(root / r['split'] / r['scene_id'] / r['state_id'] / camera / 'rgb.png'))
            for r in rows if r['split'] == spec.get('eval_split', 'eval')]
    else: rows = read_jsonl(spec['manifest'])
    frames = []; gt = {}; statuses = set(); checks = {}
    for row in rows:
        p = path_at(root, row['annotation_path']); a = read_json(p)
        if a.get('length_unit') not in ('meter', 'meters', 'm') or 'opencv' not in a.get('camera_coordinates', '').lower():
            raise ValueError('仿真 GT 必须为米制 OpenCV 相机坐标')
        checks[str(p)] = file_digest(p); provenance = a['provenance']; state = row.get('state_id', provenance['state_id'])
        if provenance['state_id'] != state or provenance['split'] != spec.get('eval_split', 'eval'):
            raise ValueError('仿真状态或测试划分不一致')
        camera = row.get('camera', spec.get('camera', 'agentview'))
        if a.get('camera') != camera: raise ValueError('仿真相机清单不一致')
        f = normal_frame(dict(provenance, frame_id=state + ':' + camera,
            image_hw=a['image_hw'], intrinsics=a['intrinsics'], rgb_path=row['rgb_path']), root)
        boxes = []
        for o in a['objects']:
            if o['kind'] not in spec.get('target_kinds', ['object']) or o.get('parent_instance_id') is not None: continue
            if o['category'] not in lookup: raise ValueError('仿真类别不在冻结类别表：' + o['category'])
            boxes.append(dict(uid=str(o['instance_id']), class_id=lookup[o['category']], center=o['center_camera'],
                size=o['size_xyz'], rotation=o['R_camera_from_box'], projected_bbox=o['bbox_projected_clipped_xyxy'],
                ignore=not o['visible'] or o['visible_pixels'] < spec.get('minimum_visible_pixels', 32)))
        frames.append(f); gt[f['frame_id']] = boxes; statuses.add(a.get('gt_status', 'simulation_groundtruth'))
    return classes, frames, gt, dict(label_status=sorted(statuses), annotations=checks)

def read_rgb(frame):
    p = Path(frame['rgb_path'])
    if frame.get('rgb_sha256') and file_digest(p) != frame['rgb_sha256']: raise ValueError('RGB 摘要变化：' + str(p))
    with Image.open(p) as im: rgb = np.asarray(im.convert('RGB')).copy()
    if frame.get('raw_rgb_pixels_sha256'):
        if list(rgb.shape[:2]) != frame['raw_image_hw'] or hashlib.sha256(rgb.tobytes()).hexdigest() != frame['raw_rgb_pixels_sha256']:
            raise ValueError('ARKit 原始像素变化')
        rgb = np.ascontiguousarray(np.rot90(rgb, frame['rotation_k']))
    if list(rgb.shape[:2]) != frame['image_hw']: raise ValueError('RGB 与冻结标定尺寸不同')
    return rgb

def load_dataset(spec):
    fmt = spec['format']
    reader = {'hot3d_prepared': hot3d, 'arkit_prepared': arkit,
              'robot_synth_prepared': simulation, 'robot_realism_prepared': simulation}.get(fmt)
    if reader is None: raise ValueError('未知的已处理数据格式：' + fmt)
    classes, frames, gt, metadata = reader(spec); ids = [f['frame_id'] for f in frames]
    if not frames or len(set(ids)) != len(ids) or set(ids) != set(gt): raise ValueError('测试清单为空、重复或与 GT 不一致')
    if spec.get('depth_manifest'):
        indexed = {f['frame_id']: f for f in frames}; seen = set()
        for r in read_jsonl(spec['depth_manifest']):
            if r['frame_id'] not in indexed or r['frame_id'] in seen: raise ValueError('深度清单有未知或重复帧')
            seen.add(r['frame_id']); p = path_at(spec['root'], r['depth_path'])
            indexed[r['frame_id']].update(depth_path=str(p), depth_sha256=file_digest(p),
                depth_kind=r['depth_kind'], depth_scale=r.get('depth_scale', 1.))
    symmetries = read_json(spec['symmetries']) if spec.get('symmetries') else {}
    for f in frames:
        k = np.asarray(f['intrinsics']); hw = f['image_hw']
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(np.linalg.det(k)) < 1e-12:
            raise ValueError('相机内参非法')
        if len(hw) != 2 or any(not isinstance(v, int) or v < 1 for v in hw): raise ValueError('图片尺寸非法')
        p = Path(f['rgb_path'])
        if not p.is_file(): raise FileNotFoundError('已处理 RGB 缺失；评测不会制作数据：' + str(p))
        checksum = file_digest(p)
        if f.get('rgb_sha256') and checksum != f['rgb_sha256']: raise ValueError('RGB 摘要变化：' + str(p))
        f['rgb_sha256'] = checksum
        with Image.open(p) as im:
            if [im.height, im.width] != f.get('raw_image_hw', hw): raise ValueError('图片尺寸与冻结标定不同')
            im.verify()
        if len({b['uid'] for b in gt[f['frame_id']]}) != len(gt[f['frame_id']]): raise ValueError('GT 实例 ID 重复')
        for b in gt[f['frame_id']]:
            validate_box(b, classes)
            if b.get('projected_bbox') is None: b['projected_bbox'] = projected_box(b, k, hw[1], hw[0])
            b['ignore'] = bool(b.get('ignore', False))
    identity = dict(format=fmt, classes=classes, frames=frames, gt=gt, metadata=metadata, symmetries=symmetries)
    return dict(classes=classes, frames=frames, gt=gt, metadata=metadata, symmetries=symmetries, dataset_id=digest(identity))

def input_frames(data, track, sensor_depth=False):
    """Only permitted observations cross the inference boundary; no scoring GT."""
    rows = []
    for f in data['frames']:
        row = {k: f[k] for k in ('frame_id', 'rgb_path', 'rgb_sha256', 'image_hw', 'intrinsics')}
        for k in ('raw_image_hw', 'raw_rgb_pixels_sha256', 'rotation_k'):
            if k in f: row[k] = f[k]
        if sensor_depth:
            if not f.get('depth_path') or f.get('depth_kind') != 'sensor':
                raise ValueError('RGB-D 需要已有实测深度；不能输入仿真 GT 深度')
            row.update(depth_path=f['depth_path'], depth_sha256=f['depth_sha256'], depth_scale=f.get('depth_scale', 1.), depth_kind='sensor')
        if track == 'gt-box':
            row['prompts'] = []
            for b in data['gt'][f['frame_id']]:
                if b['ignore']: continue
                rect = b['projected_bbox']
                if rect is None or len(rect) != 4 or not np.isfinite(rect).all() or rect[2] <= rect[0] or rect[3] <= rect[1]:
                    raise ValueError('GT 二维提示框无效：' + b['uid'])
                row['prompts'].append(dict(target_id=b['uid'], box_xyxy=rect))
        rows.append(row)
    return rows

def audit_training(variant, data):
    source = variant.get('training_manifest')
    if not source: return dict(status='unknown', reason='未提供完整预训练清单；不宣称无重叠')
    rows = read_jsonl(source); root = variant.get('training_root', str(Path(source).parent))
    sets = {k: set() for k in ('frame_id', 'state_id', 'sequence_id', 'scene_id', 'rgb_sha256')}
    for r in rows:
        for k in sets:
            if r.get(k) is not None: sets[k].add(str(r[k]))
        image = r.get('rgb_path', r.get('rgb_relative'))
        if image:
            p = path_at(root, image)
            if not p.is_file(): raise FileNotFoundError('训练来源审计缺少图片：' + str(p))
            checksum = file_digest(p)
            if r.get('rgb_sha256') and checksum != r['rgb_sha256']: raise ValueError('训练来源图片摘要变化')
            sets['rgb_sha256'].add(checksum)
    protocol_path = Path(root) / 'protocol.json'
    if Path(source).name == 'train.jsonl' and protocol_path.exists():
        expected = read_json(protocol_path).get('train_sha256')
        if expected and digest(rows) != expected and file_digest(source) != expected:
            raise ValueError('冻结训练来源清单摘要变化')
    overlap = {k: sorted(v & {str(f[k]) for f in data['frames'] if f.get(k) is not None}) for k, v in sets.items()}
    if any(overlap.values()): raise ValueError('训练／测试重叠：' + str({k: len(v) for k, v in overlap.items() if v}))
    return dict(status='disjoint_for_available_fields', training_manifest=str(source), training_manifest_sha256=file_digest(source),
                checked_fields=[k for k, v in sets.items() if v], missing_fields=[k for k, v in sets.items() if not v], overlap=overlap)
