"""Conditional localization success and detector AP are explicitly separate."""

import copy
from collections import Counter
from pathlib import Path

import numpy as np

from .benchmark_report import write_csv
from .benchmark_worker import prediction_path, read_prediction
from .common import digest, file_digest, read_json, write_json
from .geometry import box_iou
from .metrics import evaluate_ap
from .schema import CLASSES
from .suite_io import frame_path, read_record


THRESHOLDS = [('iou', .25), ('iou', .5), ('center', .25), ('center', .5), ('center', 1.)]
NAMES = ['IoU>=0.25', 'IoU>=0.50', 'center<=0.25m', 'center<=0.50m', 'center<=1.00m']


def conditional_success(gt, records):
    rows, failures, counts = [], Counter(), Counter()
    for frame_id, boxes in gt.items():
        expected = {b['uid']: b for b in boxes if not b.get('ignore', False)}
        items = records[frame_id]['items']
        by_id = {i['target_id']: i for i in items}
        if len(by_id) != len(items) or set(by_id) != set(expected):
            raise ValueError('GT 提示目标记录重复或缺失；不能将缺失文件当作估计失败')
        for uid, box in expected.items():
            item = by_id[uid]
            detection = item.get('detection')
            failed = item['status'] != 'ok'
            if failed != (detection is None):
                raise ValueError('目标状态和预测框不一致')
            overlap = 0. if failed else box_iou(detection, box)
            distance = None if failed else float(np.linalg.norm(np.asarray(detection['center'])-box['center']))
            hits = [not failed and (overlap >= t if kind == 'iou' else distance <= t) for kind, t in THRESHOLDS]
            rows.append(dict(frame_id=frame_id, target_id=uid, class_id=box['class_id'], failed=failed,
                             failure=item.get('failure'), iou=overlap, center_distance_m=distance, hits=hits))
            counts.update(i for i, yes in enumerate(hits) if yes)
            if failed: failures.update([item.get('failure', 'unspecified')])
    n = len(rows)
    classes = []
    for cid, name in enumerate(CLASSES):
        group = [r for r in rows if r['class_id'] == cid]
        classes.append(dict(class_name=name, targets=len(group), failures=sum(r['failed'] for r in group),
                            success=[sum(r['hits'][i] for r in group)/len(group) if group else None for i in range(5)]))
    return dict(targets=n, successful_estimates=n-sum(failures.values()), failures=sum(failures.values()),
                failure_rate=sum(failures.values())/n if n else None,
                success=[counts[i]/n if n else None for i in range(5)],
                mean_iou_including_failures=float(np.mean([r['iou'] for r in rows])) if n else None,
                median_center_error_successful=float(np.median([r['center_distance_m'] for r in rows if not r['failed']]))
                    if any(not r['failed'] for r in rows) else None,
                failure_reasons=dict(failures), per_class=classes), rows


def workpoint(gt, predictions, score=.7):
    # Use a NEW cache: filtering reindexes per-frame predictions. Reusing the
    # unfiltered evaluator's (frame,index) cache would silently mix box pairs.
    filtered = {frame: [box for box in boxes if box['score'] >= score] for frame, boxes in predictions.items()}
    metrics = evaluate_ap(gt, filtered, class_agnostic=True)
    row = metrics['per_class'][0]
    n = row['gt_count']
    entries = []
    for kind, threshold in THRESHOLDS:
        entry = row[kind][str(float(threshold))]
        tp, fp = entry['tp'], entry['fp']
        p, r = tp/(tp+fp) if tp+fp else 0., tp/n if n else 0.
        entries.append(dict(kind=kind, threshold=threshold, tp=tp, fp=fp,
                            precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.))
    return dict(score_threshold=score, retained_predictions=sum(map(len, filtered.values())), values=entries)


def score_model(config, root, track, phase, model_id, frames, gt, protocol, audit):
    root = Path(root)
    selected_gt = {f['frame_id']: gt[f['frame_id']] for f in frames}
    directory = root/track/phase/'metrics'
    output = directory/(model_id+'.json')
    source = root/track/'models'/model_id
    model = read_json(source/'model.json')
    reused = model.get('reuse')
    record_identity = {}
    predictions, records, timings = {}, {}, []
    for frame in frames:
        if reused:
            origin = Path(reused['model_root'])
            old = read_prediction(prediction_path(origin, frame), frame, reused['signature'],
                                  protocol['benchmark_id'], True)
            predictions[frame['frame_id']] = old['detections']
            records[frame['frame_id']] = old
            timings.append(old['timing'])
        elif model_id in ('3dmood', 'wilddet3d'):
            old = read_prediction(prediction_path(source, frame), frame, model['signature'],
                                  protocol['benchmark_id'], True)
            predictions[frame['frame_id']] = old['detections']
            records[frame['frame_id']] = old
            timings.append(old['timing'])
        else:
            record = read_record(frame_path(source/'predictions', frame), frame['frame_id'], model['signature'])
            records[frame['frame_id']] = record
            predictions[frame['frame_id']] = [i['detection'] for i in record['items'] if i['status'] == 'ok']
            timings.append(record['timing'])
        record_identity[frame['frame_id']] = digest(records[frame['frame_id']])
        if len(record_identity) % 100 == 0:
            print(f"评分读取 {model_id} {len(record_identity)}/{len(frames)} 帧", flush=True)
    identity = dict(model_signature=model['signature'], benchmark_id=protocol['benchmark_id'],
                    frames=digest([f['frame_id'] for f in frames]), predictions=digest(record_identity),
                    metric_code=digest({n:file_digest(Path(__file__).with_name(n)) for n in
                                        ('suite_score.py','metrics.py','geometry.py','schema.py')}),
                    display_score=config['display_score'])
    if output.exists() and read_json(output).get('identity') == identity and read_json(output).get('complete'):
        return read_json(output)
    excluded_scans = set(audit['official_training_visit_overlap']['validation_scan_ids'])
    disjoint = {f['frame_id'] for f in frames if f['video_id'] not in excluded_scans}
    metrics, conditional = {}, None
    if track == 'gt-box':
        conditional, details = conditional_success(selected_gt, records)
        write_csv(directory/(model_id+'_targets.csv'), ['frame_id','target_id','class','failed','failure','iou','center_distance_m',*NAMES],
                  [[r['frame_id'],r['target_id'],CLASSES[r['class_id']],r['failed'],r['failure'],r['iou'],
                    r['center_distance_m'],*r['hits']] for r in details])
        # AP is supplementary: it matches within the frame, unlike the primary
        # fixed-input-target success metric. Only supplied valid targets count.
        ap_gt = {k:[b for b in boxes if not b.get('ignore', False)] for k,boxes in selected_gt.items()}
        metric = evaluate_ap(ap_gt, predictions, class_agnostic=True)
        subset = evaluate_ap({k:ap_gt[k] for k in disjoint}, {k:predictions[k] for k in disjoint}, class_agnostic=True)
        metrics['geometry'] = dict(full=metric, official_training_visit_disjoint=subset)
        wp = None
    else:
        for mode, agnostic in [('geometry', True),('semantic_17class',False)]:
            cached = None
            if reused:
                old_path = Path(reused['report_root'])/phase/'metrics'/(model_id+'.json')
                if old_path.exists():
                    old = read_json(old_path)
                    old_ids = read_json(Path(reused['report_root'])/phase/'selection.json')['frame_ids']
                    current_code = digest({n:file_digest(Path(__file__).with_name(n)) for n in ('metrics.py','geometry.py','schema.py')})
                    if (old['complete'] and old_ids == [f['frame_id'] for f in frames] and
                            old['identity']['predictions_sha256'] == digest(record_identity) and
                            old['identity']['metric_implementation_sha256'] == current_code):
                        cached = old['metrics'][mode]
            if cached:
                metrics[mode] = copy.deepcopy(cached)
                print(f'{model_id} {phase} {mode} 复用已验证 AP', flush=True)
            else:
                pairs = {}
                all_metric = evaluate_ap(selected_gt, predictions, class_agnostic=agnostic, _pair_cache=pairs)
                subset = evaluate_ap({k:selected_gt[k] for k in disjoint}, {k:predictions[k] for k in disjoint},
                                     class_agnostic=agnostic, _pair_cache=pairs)
                metrics[mode] = dict(full=all_metric, official_training_visit_disjoint=subset)
        wp = workpoint(selected_gt, predictions, config['display_score'])
    result = dict(complete=True, identity=identity, model_id=model_id, track=track, phase=phase,
                  frames=len(frames), metrics=metrics, conditional=conditional, workpoint=wp, model=model,
                  predictions=sum(map(len,predictions.values())),
                  excluded_frames=len(frames)-len(disjoint),
                  failures=0 if model_id in ('3dmood','wilddet3d') else
                      sum(i['status']!='ok' for r in records.values() for i in r['items']),
                  timing=dict(total_predict_seconds=sum(t['predict_seconds'] for t in timings),
                              total_frontend_seconds=sum(t.get('frontend_seconds',0.) for t in timings),
                              total_depth_seconds=sum(t.get('depth_seconds',0.) for t in timings),
                              historical_reuse=bool(reused)))
    write_json(output, result)
    for mode, entry in metrics.items():
        write_csv(directory/(model_id+'_'+mode+'_classes.csv'), ['class','valid_gt','ignored_gt','predictions',*NAMES],
                  [[r['class_name'],r['gt_count'],r['ignored_gt_count'],r['prediction_count'],
                    *[r[k][str(float(t))]['ap'] for k,t in THRESHOLDS]] for r in entry['full']['per_class']])
    return result
