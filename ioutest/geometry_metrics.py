"""Fixed image-plane association and conditional metric geometry diagnostics."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import box_iou, projected_box


def _representations():
    result = []
    for axes in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            p = np.eye(3)[:, axes] * signs
            if np.linalg.det(p) > 0: result.append(p)
    return np.asarray(result)

EQUIVALENT_AXES = _representations()


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a): return dict(count=0, mean=None, median=None, p90=None)
    if not np.isfinite(a).all(): raise ValueError('指标含非有限数值')
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)))


def geometry_error(prediction, target, symmetry=None):
    rp = np.asarray(prediction['rotation']); rt = np.asarray(target['rotation'])
    sp = np.asarray(prediction['size']); st = np.asarray(target['size'])
    symmetry = symmetry or {}
    rotations = [np.eye(3), *[np.asarray(r, dtype=float) for r in symmetry.get('rotations', [])]]
    axis = symmetry.get('continuous_axis')
    if axis is not None and axis not in (0, 1, 2): raise ValueError('continuous_axis 必须为 0、1、2')
    if axis is not None and not np.isclose(st[(axis + 1) % 3], st[(axis + 2) % 3], rtol=.02):
        raise ValueError('连续轴对称标注与框的横截面尺寸不一致')
    for s in rotations:
        if s.shape != (3, 3) or not np.isfinite(s).all() or not np.allclose(s.T @ s, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(s), 1.):
            raise ValueError('对称变换必须为 SO(3) 矩阵')
    candidates = []
    # A single jointly chosen representation supplies BOTH size and angle errors.
    for s in rotations:
        reference = rt @ s
        for p in EQUIVALENT_AXES:
            r = rp @ p; size = np.abs(p).T @ sp
            if axis is None:
                angle = np.arccos(np.clip((np.trace(r.T @ reference) - 1) / 2, -1, 1))
                cost = np.sum((r * size - reference * st) ** 2)
            else:
                angle = np.arccos(np.clip(np.dot(r[:, axis], reference[:, axis]), -1, 1))
                cost = np.sum((size - st) ** 2) + float(st.mean() ** 2 * angle ** 2)
            candidates.append((float(cost), float(angle), size))
    _, angle, size = min(candidates, key=lambda v: (v[0], v[1]))
    delta = np.asarray(prediction['center']) - target['center']
    absolute = np.abs(size - st)
    return dict(center_error_m=float(np.linalg.norm(delta)), center_abs_xyz_m=np.abs(delta).tolist(),
        size_abs_xyz_m=absolute.tolist(), size_relative_xyz=(absolute / st).tolist(), rotation_error_deg=float(np.degrees(angle)))


def rectangle_iou(a, b):
    if a is None or b is None: return 0.
    a = np.asarray(a); b = np.asarray(b)
    inter = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0., a[2:] - a[:2]).prod() + np.maximum(0., b[2:] - b[:2]).prod() - inter
    return float(inter / union) if union > 0 else 0.


def associate(predictions, targets, frame, threshold, semantic):
    h, w = frame['image_hw']; k = frame['intrinsics']
    pred_rects = [projected_box(p, k, w, h) for p in predictions]
    gt_rects = [projected_box(b, k, w, h) for b in targets]
    available = set(range(len(predictions))); pairs = {}; ignored_predictions = set()
    for ignored in (False, True):
        target_ids = [i for i, b in enumerate(targets) if b.get('ignore', False) == ignored]
        pred_ids = sorted(available)
        if not target_ids or not pred_ids: continue
        scores = np.asarray([[rectangle_iou(pred_rects[p], gt_rects[g])
            if not semantic or predictions[p]['class_id'] == targets[g]['class_id'] else 0.
            for g in target_ids] for p in pred_ids])
        # Maximize valid-pair COUNT first, then total IoU; invalid pairs never consume a match.
        eligible = scores >= threshold
        benefit = eligible * (len(pred_ids) + 1 + scores)
        rr, cc = linear_sum_assignment(-benefit)
        for r, c in zip(rr, cc):
            if not eligible[r, c]: continue
            pi, gi = pred_ids[r], target_ids[c]; available.remove(pi)
            if ignored: ignored_predictions.add(pi)
            else: pairs[gi] = pi
    return pairs, available, ignored_predictions


def diagnostics(data, records, track, settings, semantic=False):
    rows = []; unmatched = 0; ignored_predictions = 0; original_predictions = 0
    for frame in data['frames']:
        fid = frame['frame_id']; targets = data['gt'][fid]; rec = records[fid]
        original_predictions += len(rec['detections'])
        if track == 'gt-box':
            items = rec['items']; actual = {i['target_id']: i for i in items}
            expected = {b['uid'] for b in targets if not b.get('ignore', False)}
            if len(actual) != len(items) or set(actual) != expected: raise ValueError('GT 提示目标缺失、重复或关联错误')
            for item in items:
                if item['status'] not in ('ok', 'failed') or (item['status'] == 'ok') != (item.get('detection') is not None):
                    raise ValueError('逐目标状态与三维预测不一致')
            predictions = [i['detection'] for i in items if i['status'] == 'ok']
            if predictions != rec['detections']: raise ValueError('目标记录与帧预测不一致')
            pairs = {i: actual[b['uid']]['detection'] for i, b in enumerate(targets) if not b.get('ignore', False)}
        else:
            predictions = [p for p in rec['detections'] if p['score'] >= settings['score_floor']]
            indices, left, ignored = associate(predictions, targets, frame, settings['association_iou'], semantic)
            unmatched += len(left); ignored_predictions += len(ignored)
            pairs = {i: predictions[p] for i, p in indices.items()}
        for i, b in enumerate(targets):
            if b.get('ignore', False): continue
            prediction = pairs.get(i)
            row = dict(frame_id=fid, group_id=frame['group_id'], target_id=b['uid'], class_id=b['class_id'],
                       matched=prediction is not None, failure=None, center_error_m=None,
                       center_abs_xyz_m=None, size_abs_xyz_m=None, size_relative_xyz=None, rotation_error_deg=None, iou=None)
            if prediction is not None:
                syms = data.get('symmetries', {})
                symmetry = syms.get('by_object', {}).get(str(b.get('object_bop_id', b['uid'])),
                    syms.get('by_class', {}).get(data['classes'][b['class_id']]['name']))
                row.update(geometry_error(prediction, b, symmetry)); row['iou'] = box_iou(prediction, b)
            else:
                row['failure'] = actual[b['uid']].get('failure', 'no_output') if track == 'gt-box' else 'unassociated'
            rows.append(row)
    matched = [r for r in rows if r['matched']]; n = len(rows)
    errors = {key: distribution([r[key] for r in matched]) for key in ('center_error_m', 'rotation_error_deg')}
    for name in ('center_abs_xyz_m', 'size_abs_xyz_m', 'size_relative_xyz'):
        errors[name] = {axis: distribution([r[name][i] for r in matched]) for i, axis in enumerate('xyz')}
    success = {}
    for kind, thresholds in (('center', settings['center_thresholds']), ('iou', settings['iou_thresholds'])):
        for t in thresholds:
            count = sum(r['matched'] and (r['center_error_m'] <= t if kind == 'center' else r['iou'] >= t) for r in rows)
            success[f'{kind}@{t}'] = dict(passed=count, targets=n, rate=count / n if n else None)
    return dict(association='input_target_id' if track == 'gt-box' else 'projected_3d_box_iou_one_to_one',
        association_iou=None if track == 'gt-box' else settings['association_iou'], semantic=semantic,
        score_floor=None if track == 'gt-box' else settings['score_floor'],
        targets=n, matched=len(matched), missed=n-len(matched), association_recall=len(matched)/n if n else None,
        missed_rate=(n-len(matched))/n if n else None, unmatched_predictions=unmatched,
        ignored_predictions=ignored_predictions, original_predictions=original_predictions,
        errors=errors, localization_success=success, units=dict(distance='meter', angle='degree', relative_size='fraction'),
        orientation='OBB geometry modulo equivalent axes; optional explicitly supplied symmetries'), rows
