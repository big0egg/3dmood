from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import atomic_json, file_hash, read_json
from .dataset import bundle, verify_source
from .pipeline import make_profile, verify_result
from .common import digest


def validate(config, frozen, allow_partial=False, allow_blocked=False):
    from .blocked import read_blocked, skip_enabled
    if allow_blocked and not skip_enabled(config):
        raise ValueError('allow_blocked requires record_and_skip execution policy')
    root = Path(config['output']['root'])
    info = read_json(root / 'dataset_info.json')
    if info['profile_id'] != digest(make_profile(config, frozen)):
        raise ValueError('Realism code/config differs from saved results')
    errors, counts, blocked = [], Counter(), []
    accepted = []
    for record in frozen['records']:
        try:
            commit = verify_result(root, record, frozen['cameras'], info['profile_id'])
            rejections = [read_blocked(config, frozen, info['profile_id'], record, camera)
                          for camera in frozen['cameras']]
            rejections = [value for value in rejections if value]
            if commit is None:
                blocked.extend(rejections)
                continue
            source_commit = verify_source(config['input']['root'], record, frozen['cameras'])
            if commit['source_commit_sha256'] != file_hash(bundle(config['input']['root'], record) / 'COMMIT.json'):
                raise ValueError('Source commit provenance differs')
            for cam in frozen['cameras']:
                folder = bundle(root, record) / cam
                source = bundle(config['input']['root'], record) / cam
                with Image.open(folder / 'rgb.png') as image:
                    image.verify()
                with Image.open(folder / 'rgb.png') as image, Image.open(source / 'rgb.png') as original:
                    if image.size != original.size or image.mode != 'RGB':
                        raise ValueError('Generated RGB shape/mode differs from original')
                for name in ('depth.npz', 'instance.png'):
                    if file_hash(folder / name) != source_commit['files_sha256'][f'{cam}/{name}']:
                        raise ValueError('Inherited metric depth or instance GT was changed')
                annotation = read_json(folder / 'annotation.json')
                original = read_json(source / 'annotation.json')
                for key in ('objects', 'intrinsics', 'T_world_from_camera', 'T_camera_from_world', 'provenance'):
                    if annotation.get(key) != original.get(key):
                        raise ValueError(f'Inherited annotation changed: {key}')
                meta = read_json(folder / 'render_meta.json')
                expected_prompt = frozen.get('prompt_assignments', {}).get(record['state_id'], {}).get(
                    'prompt', config['generation']['prompt'])
                if meta['inference']['prompt'] != expected_prompt:
                    raise ValueError('Saved prompt differs from frozen prompt assignment')
                if meta.get('prompt_inputs_sha256', {}) != frozen.get('prompt_inputs_sha256', {}):
                    raise ValueError('Saved prompt manifest identity differs')
                if meta['source_rgb_sha256'] != source_commit['files_sha256'][f'{cam}/rgb.png']:
                    raise ValueError('Source RGB provenance differs')
                if meta.get('runtime_id') != digest(read_json(root / 'runtime_identity.json')):
                    raise ValueError('Saved runtime/checkpoint provenance was modified')
                if file_hash(bundle(root, record) / 'state.npy') != source_commit['files_sha256']['state.npy']:
                    raise ValueError('Inherited source state was changed')
            accepted.append(record)
            counts[record['split']] += len(frozen['cameras'])
        except Exception as error:
            errors.append({'state_id': record['state_id'], 'error': str(error)})
    complete = all(counts[s] == n for s, n in frozen['targets'].items())
    planned = sum(frozen['targets'].values())
    accounted = sum(counts.values()) + len(blocked)
    execution_complete = accounted == planned
    if not accepted and not (allow_blocked and blocked):
        errors.append({'error': 'No generated committed images to validate'})
    if not complete and not allow_partial and not (allow_blocked and execution_complete):
        errors.append({'error': 'Full planned image counts not reached'})
    if allow_blocked and not execution_complete and not allow_partial:
        errors.append({'error': 'Some planned frames have neither a valid commit nor a blocked record'})
    result = {'valid': not errors, 'complete': complete, 'counts': dict(counts), 'targets': frozen['targets'],
              'dataset_complete': complete, 'execution_complete': execution_complete and not errors,
              'planned_images': planned, 'generated_images': sum(counts.values()),
              'blocked_images': len(blocked), 'pending_images': planned-accounted,
              'errors': errors, 'visual_gt_alignment_certified': False,
              'scope': 'files, provenance, calibration and inherited GT integrity; generated RGB still needs visual review'}
    atomic_json(Path(config['output']['run_dir']) / 'validation.json', result)
    return result


def report(config, frozen, max_previews=40):
    if max_previews < 1:
        raise ValueError('--max-previews must be positive')
    root = Path(config['output']['root'])
    info = read_json(root / 'dataset_info.json')
    directory = Path(config['output']['run_dir']) / 'quality_report'
    previews = directory / 'previews'
    previews.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in frozen['records']:
        if not verify_result(root, record, frozen['cameras'], info['profile_id']):
            continue
        for camera in frozen['cameras']:
            source = bundle(config['input']['root'], record) / camera
            result = bundle(root, record) / camera
            with Image.open(source / 'rgb.png') as image:
                original = image.convert('RGB')
            with Image.open(result / 'rgb.png') as image:
                generated = image.convert('RGB')
            with Image.open(source / 'instance.png') as image:
                mask = np.array(image)
            boundary = np.zeros(mask.shape, bool)
            boundary[1:] |= mask[1:] != mask[:-1]
            boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
            pixels = np.array(generated)
            pixels[boundary] = [0, 255, 0]
            overlay = Image.fromarray(pixels)
            # Draw only visible finite object boxes, avoiding robot/support
            # clutter. These are inherited GT projections, not new predictions.
            annotation = read_json(source / 'annotation.json')
            K = np.asarray(annotation['intrinsics'])
            draw = ImageDraw.Draw(overlay)
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            for obj in annotation.get('objects', []):
                if obj.get('kind') != 'object' or not obj.get('visible'):
                    continue
                corners = signs * np.asarray(obj['size_xyz']) / 2
                corners = corners @ np.asarray(obj['R_camera_from_box']).T + np.asarray(obj['center_camera'])
                for i in range(8):
                    for axis in (1, 2, 4):
                        j = i ^ axis
                        if i >= j or min(corners[i, 2], corners[j, 2]) <= 1e-4:
                            continue
                        points = corners[[i, j]] @ K.T
                        points = points[:, :2] / points[:, 2:]
                        if np.isfinite(points).all() and np.abs(points).max() < 1e6:
                            draw.line(tuple(points.reshape(-1)), fill=(255, 210, 0), width=1)
            w, h = original.size
            canvas = Image.new('RGB', (w * 3, h + 28), 'white')
            for i, (label, panel) in enumerate(zip(('SOURCE RGB', 'COSMOS RGB', 'SOURCE MASK EDGES ON COSMOS'), (original, generated, overlay))):
                canvas.paste(panel, (i * w, 28))
                ImageDraw.Draw(canvas).text((i * w + 4, 5), label, fill='black')
            name = f"{record['state_id']}_{camera}.jpg"
            canvas.save(previews / name, quality=93)
            rows.append(f"- {record['split']}/{record['scene_id']}/{camera}: [对照图](previews/{name})")
            if len(rows) >= max_previews:
                break
        if len(rows) >= max_previews:
            break
    preview_count = len(rows)
    if not rows:
        from .blocked import ProgressTracker, skip_enabled
        summary = ProgressTracker(config, frozen, info['profile_id']).summary()
        if not (skip_enabled(config) and summary['execution_complete'] and summary['blocked_images']):
            raise ValueError('No committed generated images; run real inference first')
        rows.append('所有计划样本均被内容检查拦截；没有可展示的成功图片。请查看 blocked 清单。')
    text = '# 写实化图片对照检查\n\n左：原图；中：Cosmos；右：绿色原实例边界和黄色原物体 3D 框叠加生成图。\n\n'
    text += '**逐图检查物体增删、轮廓漂移和遮挡。绿色边界仅是对照，不证明新 RGB 的 3D GT 正确。**\n\n'
    text += '\n'.join(rows) + '\n'
    (directory / 'report.md').write_text(text, encoding='utf-8')
    return {'previews': preview_count, 'report': str(directory / 'report.md'), 'visual_review_required': True}
