"""101-recall-point AP for full 3D IoU and absolute center distance."""

import numpy as np

from .geometry import box_iou
from .schema import CLASSES, validate_box


def interpolated_ap(tp, fp, gt_count):
    if gt_count == 0:
        return None
    if not tp:
        return 0.
    true = np.cumsum(tp)
    false = np.cumsum(fp)
    recall = true / gt_count
    precision = true / np.maximum(true + false, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indexes = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    sampled = np.zeros(101)
    valid = indexes < len(envelope)
    sampled[valid] = envelope[indexes[valid]]
    return float(sampled.mean())


def evaluate_ap(groundtruth, predictions, iou_thresholds=(.25, .5),
                distance_thresholds=(.25, .5, 1.), _pair_cache=None, class_agnostic=False):
    if set(groundtruth) != set(predictions):
        missing = set(groundtruth) - set(predictions)
        extra = set(predictions) - set(groundtruth)
        raise ValueError(f"预测清单不完整或不一致：缺失 {len(missing)}，多余 {len(extra)}")
    if not iou_thresholds or any(not np.isfinite(x) or not 0 < x <= 1 for x in iou_thresholds):
        raise ValueError("IoU 阈值应在 (0,1] 范围")
    if not distance_thresholds or any(not np.isfinite(x) or x <= 0 for x in distance_thresholds):
        raise ValueError("中心距离阈值必须为正数，单位米")
    if len(set(iou_thresholds)) != len(iou_thresholds) or len(set(distance_thresholds)) != len(distance_thresholds):
        raise ValueError("评测阈值不能重复")
    for groups, pred in ((groundtruth, False), (predictions, True)):
        for boxes in groups.values():
            for box in boxes:
                validate_box(box, prediction=pred, allow_unlabeled=pred and class_agnostic)

    if class_agnostic:
        # Only matching ignores class; persisted boxes and semantic labels are untouched.
        groundtruth = {f: [dict(b, class_id=0) for b in bs] for f, bs in groundtruth.items()}
        predictions = {f: [dict(b, class_id=0) for b in bs] for f, bs in predictions.items()}

    results = []
    pairs = {} if _pair_cache is None else _pair_cache
    for cid, name in enumerate(("object",) if class_agnostic else CLASSES):
        gt = {frame: [b for b in boxes if b["class_id"] == cid]
              for frame, boxes in groundtruth.items()}
        n_gt = sum(not b.get("ignore", False) for boxes in gt.values() for b in boxes)
        ranked = [(float(p["score"]), frame, index, p)
                  for frame, boxes in predictions.items()
                  for index, p in enumerate(boxes) if p["class_id"] == cid]
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        # Compute each pair once and reuse across thresholds.
        for _, frame, index, pred in ranked:
            if (frame, index) not in pairs:
                boxes = gt[frame]
                distances = np.asarray([np.linalg.norm(np.asarray(pred["center"]) - b["center"])
                                        for b in boxes])
                overlaps = np.asarray([box_iou(pred, b) for b in boxes])
                pairs[frame, index] = (overlaps, distances)
        entry = {"class_id": cid, "class_name": name, "gt_count": int(n_gt),
                 "ignored_gt_count": sum(bool(b.get("ignore", False)) for bs in gt.values() for b in bs),
                 "prediction_count": len(ranked), "iou": {}, "center": {}}
        for kind, thresholds in (("iou", iou_thresholds), ("center", distance_thresholds)):
            for threshold in thresholds:
                used, used_ignore = {}, {}
                tp, fp, ignored_count = [], [], 0
                for _, frame, index, _ in ranked:
                    boxes = gt[frame]
                    values = pairs[frame, index][kind == "center"]
                    qualifies = values >= threshold if kind == "iou" else values <= threshold
                    matched = used.setdefault(frame, set())
                    ignored = used_ignore.setdefault(frame, set())
                    candidates = [j for j, b in enumerate(boxes)
                                  if not b.get("ignore", False) and j not in matched and qualifies[j]]
                    if candidates:
                        best = (max if kind == "iou" else min)(candidates, key=lambda j: values[j])
                        matched.add(best)
                        tp.append(1); fp.append(0)
                    else:
                        candidates = [j for j, b in enumerate(boxes)
                                      if b.get("ignore", False) and j not in ignored and qualifies[j]]
                        if candidates:
                            best = (max if kind == "iou" else min)(candidates, key=lambda j: values[j])
                            ignored.add(best)
                            ignored_count += 1
                        else:
                            tp.append(0); fp.append(1)
                entry[kind][str(float(threshold))] = {
                    "ap": interpolated_ap(tp, fp, n_gt), "tp": sum(tp), "fp": sum(fp),
                    "ignored_predictions": ignored_count,
                }
        results.append(entry)
    summaries = {}
    for kind, thresholds in (("iou", iou_thresholds), ("center", distance_thresholds)):
        per_threshold = {}
        for threshold in thresholds:
            values = [row[kind][str(float(threshold))]["ap"] for row in results
                      if row["gt_count"] > 0]
            per_threshold[str(float(threshold))] = float(np.mean(values)) if values else None
        values = [v for v in per_threshold.values() if v is not None]
        summaries[kind] = {"map": float(np.mean(values)) if values else None,
                           "map_per_threshold": per_threshold}
    return {"ap_interpolation": "101 recall positions, precision envelope",
            "matching": ("same frame, category-agnostic" if class_agnostic else "same frame/class") +
                        ", confidence descending, one-to-one (including ignore)",
            "class_agnostic": class_agnostic,
            "distance_unit": "meter", "frame_count": len(groundtruth),
            "valid_class_count": sum(row["gt_count"] > 0 for row in results),
            "summary": summaries, "per_class": results}
