"""Shared AP plus five diagnostics, using complete numerical predictions only."""
from pathlib import Path
from .common import digest, read_json, write_json
from .geometry_metrics import diagnostics
from .hot3d.common import validate_box
from .hot3d.metrics import evaluate, conditional, thresholds
from .records import read_record, record_path
from .timing import summarize
from .report_io import write_dict_csv

def score(task, data, settings):
    root = Path(task['run_root']); model = read_json(root / 'model.json')
    if model['provenance']['request_id'] != task['request_id'] or digest(model['provenance']) != model['signature']:
        raise ValueError('权重／输入／模型配置与已有预测不一致')
    records = {}; predictions = {}; hashes = {}
    for f in data['frames']:
        fid = f['frame_id']; rec = read_record(record_path(root, fid), fid, task['request_id'], model['signature'])
        for b in rec['detections']: validate_box(b, data['classes'], True)
        records[fid] = rec; predictions[fid] = rec['detections']; hashes[fid] = rec['checksum']
    limits = thresholds(dict(metrics=settings)); classes = data['classes']; gt = data['gt']
    if task['track'] == 'gt-box':
        ap_gt = {f: [b for b in boxes if not b['ignore']] for f, boxes in gt.items()}
    else: ap_gt = gt
    cache = {}
    geometry = evaluate(ap_gt, predictions, classes, limits, pairs=cache)
    semantic = None
    if task['track'] == 'text' and task['model']['family'] != 'cutr':
        semantic = evaluate(gt, predictions, classes, limits, semantic=True, pairs=cache)
    workpoint = evaluate(ap_gt, predictions, classes, limits, score_floor=settings['score_floor'])
    geometry_diagnostics, rows = diagnostics(data, records, task['track'], settings)
    semantic_diagnostics = None
    if semantic is not None: semantic_diagnostics, _ = diagnostics(data, records, 'text', settings, semantic=True)
    conditional_result = None
    if task['track'] == 'gt-box':
        # Library conditional metrics expect the optional HOT3D identity field.
        conditional_gt = {f: [dict(b, object_bop_id=b.get('object_bop_id', b['class_id'])) for b in boxes] for f, boxes in gt.items()}
        conditional_result, _ = conditional(conditional_gt, records, limits)
    if task['track'] == 'text':
        n = sum(not b['ignore'] for boxes in gt.values() for b in boxes)
        detection = dict(mode='autonomous_geometry_detection', score_floor=settings['score_floor'],
            criteria={k: dict(tp=v['tp'], fp=v['fp'], fn=n-v['tp'], precision=v['precision'], recall=v['recall'],
                              missed_rate=(n-v['tp'])/n if n else None)
                      for k,v in workpoint['per_class'][0]['values'].items()})
    else:
        detection = dict(mode='conditioned_on_supplied_2d_targets', targets=geometry_diagnostics['targets'],
            valid_outputs=geometry_diagnostics['matched'], missing_outputs=geometry_diagnostics['missed'],
            output_failure_rate=geometry_diagnostics['missed_rate'], criteria=geometry_diagnostics['localization_success'])
    timing_records = list(records.values()); timing_source = 'inference_records'
    latency_paths = [root / 'latency_records' / record_path(root, f['frame_id']).name for f in data['frames']]
    if any(p.exists() for p in latency_paths):
        if not all(p.exists() for p in latency_paths): raise ValueError('独立延迟测量不完整，不能混用部分测量')
        timing_records = [read_record(p, f['frame_id'], task['request_id'], model['signature']) for p, f in zip(latency_paths, data['frames'])]
        timing_source = 'latency_records'
    latency = summarize(timing_records); latency['source'] = timing_source
    if (root / 'latency.json').exists():
        previous = read_json(root / 'latency.json')
        if previous.get('request_id') == task['request_id'] and previous.get('model_id') == model['signature']:
            latency.update({k: previous[k] for k in ('initialization_seconds', 'warmup_seconds', 'warmup_passes', 'reused_records') if k in previous})
    write_json(root / 'latency.json', dict(latency, request_id=task['request_id'], model_id=model['signature']))
    output = dict(schema='ioutest.metrics.v2', complete=True, request_id=task['request_id'], dataset_id=data['dataset_id'],
        model_id=model['signature'], predictions_sha256=digest(hashes), metrics_settings=settings,
        frames=len(records), classes=classes, geometry=geometry, semantic=semantic, workpoint=workpoint,
        conditional=conditional_result, diagnostics=geometry_diagnostics, semantic_diagnostics=semantic_diagnostics, detection=detection,
        latency=latency, data_metadata=data['metadata'])
    output['checksum'] = digest(output)
    write_json(root / 'metrics.json', output)
    flattened = []
    for row in rows:
        r = dict(row)
        for key in ('center_abs_xyz_m', 'size_abs_xyz_m', 'size_relative_xyz'):
            values = r.pop(key)
            for i, axis in enumerate('xyz'): r[key + '_' + axis] = values[i] if values is not None else None
        flattened.append(r)
    write_dict_csv(root / 'targets.csv', flattened)
    per_class = []
    for mode, result in (('geometry', geometry), ('semantic', semantic)):
        if result:
            for entry in result['per_class']:
                per_class.append(dict(mode=mode, class_id=entry['class_id'], class_name=entry['class_name'], valid_gt=entry['valid_gt'],
                    **{name: v['ap'] for name, v in entry['values'].items()}))
    write_dict_csv(root / 'classes.csv', per_class)
    return output
