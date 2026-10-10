"""Durable model runs, prediction import and reproducible offline reports."""

import csv
import io
import time
from collections import defaultdict
from pathlib import Path

from .adapters import create_adapter
from .common import (atomic_text, digest, file_digest, read_json, read_jsonl,
                     safe_name, utc_now, write_json)
from .dataset import ScanReader, load_benchmark, metadata
from .metrics import evaluate_ap
from .schema import SCHEMA_VERSION, validate_box


def model_dir(run_dir, model_id):
    return Path(run_dir) / "models" / safe_name(model_id)


def training_audit(root, annotation_paths):
    if not annotation_paths:
        return {"status": "not_supplied", "note": "未审计模型训练清单，不能宣称完全未见场景"}
    rows = metadata(root)
    identified, sources, unidentified = set(), [], 0
    for path in annotation_paths:
        path = Path(path).resolve()
        doc = read_json(path)
        if "images" not in doc:
            raise ValueError("training-annotations 应为包含 images 的 Omni3D/COCO JSON")
        sources.append({"path": str(path), "sha256": file_digest(path)})
        for image in doc["images"]:
            name = str(image.get("file_path", image.get("file_name", "")))
            parts = Path(name).parts
            ids = {p for p in parts if p in rows}
            if not ids and parts:
                prefix = Path(parts[-1]).stem.split("_")[0]
                if prefix in rows:
                    ids.add(prefix)
            identified.update(ids)
            if "arkitscenes" in name.lower() and not ids:
                unidentified += 1
    if unidentified:
        raise ValueError(f"训练标注中 {unidentified} 个 ARKitScenes 图片无法识别扫描 ID")
    if not identified:
        raise ValueError("训练标注未识别出任何 ARKitScenes 扫描，不能声称审计成功")
    visits = {rows[s]["visit_id"] for s in identified if rows[s]["visit_id"] not in ("NA", "")}
    excluded = {r["video_id"] for r in rows.values() if r["fold"] == "Validation"
                and (r["video_id"] in identified or r["visit_id"] in visits)}
    return {"status": "declared_annotations_audited", "sources": sources,
            "identified_training_scans": sorted(identified), "excluded_validation_scans": sorted(excluded),
            "note": "基于用户提供的训练标注；不推断标注之外的训练数据"}


def _check_record(record, allowed):
    if record.get("frame_id") not in allowed or record.get("status") != "ok":
        raise ValueError("预测 frame_id 不属于清单或 status 不是 ok")
    if not isinstance(record.get("detections"), list):
        raise ValueError("detections 必须为列表；空预测使用 []")
    for box in record["detections"]:
        validate_box(box, prediction=True)


def infer(run_dir, model_id="3dmood-swin-b", adapter_name="3dmood", options=None,
          training_annotations=None):
    protocol, audit, frames, _ = load_benchmark(run_dir)
    path = model_dir(run_dir, model_id)
    by_scan = defaultdict(list)
    for frame in frames:
        by_scan[frame["video_id"]].append(frame)
    allowed = {f["frame_id"] for f in frames}
    options = dict(options or {})
    adapter, factory = create_adapter(adapter_name, options)
    try:
        start = time.perf_counter()
        model_info = adapter.load()
        load_seconds = time.perf_counter() - start
        info = {"benchmark_id": protocol["benchmark_id"], "schema_version": SCHEMA_VERSION,
                "adapter": factory, "options": options, "model": model_info,
                "training_audit": training_audit(protocol["settings"]["dataset_root"], training_annotations)}
        info["signature"] = digest(info)
        info_path = path / "model.json"
        if info_path.exists() and read_json(info_path)["signature"] != info["signature"]:
            raise ValueError("model-id 已有不同模型、权重、协议或适配器版本，请使用新 model-id")
        if not info_path.exists():
            write_json(info_path, info)
        root = protocol["settings"]["dataset_root"]
        rows = metadata(root)
        complete = 0
        for scan, scan_frames in by_scan.items():
            shard = path / "predictions" / (scan + ".json")
            artifact = read_json(shard) if shard.exists() else {"model_signature": info["signature"], "records": []}
            if artifact["model_signature"] != info["signature"]:
                raise ValueError("预测分片模型签名不一致")
            records = {}
            for record in artifact["records"]:
                _check_record(record, allowed)
                if record["frame_id"] in records:
                    raise ValueError("预测缓存存在重复 frame_id")
                records[record["frame_id"]] = record
            if not set(records) <= {f["frame_id"] for f in scan_frames}:
                raise ValueError("预测分片混入其他扫描的帧")
            with ScanReader(root, rows[scan]) as reader:
                for frame in scan_frames:
                    if frame["frame_id"] not in records:
                        begin = time.perf_counter()
                        sample = reader.model_input(frame)
                        input_seconds = time.perf_counter() - begin
                        begin = time.perf_counter()
                        boxes = adapter.predict(sample)
                        detections = [b.to_dict() for b in boxes]
                        record = {"frame_id": frame["frame_id"], "status": "ok", "detections": detections,
                                  "input_seconds": input_seconds, "predict_seconds": time.perf_counter() - begin}
                        _check_record(record, allowed)
                        records[frame["frame_id"]] = record
                        artifact["records"] = list(records.values())
                        write_json(shard, artifact)
                    complete += 1
                    print(f"推理 {complete}/{len(frames)} {frame['frame_id']}: "
                          f"{len(records[frame['frame_id']]['detections'])} 个预测", flush=True)
        _, records, _ = load_predictions(run_dir, model_id)
        durations = [r.get("predict_seconds", 0.) for r in records.values()]
        steady = durations[1:] or durations
        import numpy as np
        timing = {"complete": True, "frame_count": len(records), "model_load_seconds_this_invocation": load_seconds,
                  "mean_predict_seconds_excluding_first": float(np.mean(steady)),
                  "estimated_10980_frame_predict_hours": float(np.mean(steady) * 10980 / 3600),
                  "note": "仅预测时间估计；不含准备、模型加载、写盘和指标计算", "utc": utc_now()}
        write_json(path / "timing.json", timing)
        return timing
    except Exception as exc:
        write_json(path / "error.json", {"error": str(exc), "utc": utc_now()})
        raise
    finally:
        adapter.close()


def load_predictions(run_dir, model_id):
    protocol, _, frames, _ = load_benchmark(run_dir)
    path = model_dir(run_dir, model_id)
    info = read_json(path / "model.json")
    if info["benchmark_id"] != protocol["benchmark_id"] or info["schema_version"] != SCHEMA_VERSION:
        raise ValueError("模型预测与当前 benchmark/schema 不一致")
    allowed = {f["frame_id"] for f in frames}
    records, hashes = {}, {}
    for shard in sorted((path / "predictions").glob("*.json")):
        artifact = read_json(shard)
        if artifact["model_signature"] != info["signature"]:
            raise ValueError("预测分片模型签名不一致")
        hashes[shard.name] = file_digest(shard)
        for record in artifact["records"]:
            _check_record(record, allowed)
            if record["frame_id"] in records:
                raise ValueError("预测有重复 frame_id")
            if record["frame_id"].split(":", 1)[0] != shard.stem:
                raise ValueError("预测 frame_id 与分片扫描不一致")
            records[record["frame_id"]] = record
    if set(records) != allowed:
        raise ValueError(f"预测尚未完成：{len(records)}/{len(allowed)} 帧，请先完成 infer 或完整导入")
    return info, records, digest(hashes)


def import_predictions(run_dir, model_id, prediction_path, benchmark_id):
    protocol, _, frames, _ = load_benchmark(run_dir)
    if benchmark_id != protocol["benchmark_id"]:
        raise ValueError("必须提供与 protocol.json 一致的 benchmark-id，确认预测属于该清单")
    allowed = {f["frame_id"] for f in frames}
    values = read_jsonl(prediction_path)
    by_scan, seen = defaultdict(list), set()
    for record in values:
        _check_record(record, allowed)
        if record["frame_id"] in seen:
            raise ValueError("导入文件有重复 frame_id")
        seen.add(record["frame_id"])
        by_scan[record["frame_id"].split(":", 1)[0]].append(record)
    if seen != allowed:
        raise ValueError(f"导入文件必须覆盖每一帧，当前 {len(seen)}/{len(allowed)}")
    path = model_dir(run_dir, model_id)
    info = {"benchmark_id": benchmark_id, "schema_version": SCHEMA_VERSION,
            "model": {"model": model_id, "source": "offline_prediction_import",
                      "prediction_file_sha256": file_digest(prediction_path)},
            "training_audit": {"status": "not_supplied"}}
    info["signature"] = digest(info)
    info_path = path / "model.json"
    if info_path.exists() and read_json(info_path)["signature"] != info["signature"]:
        raise ValueError("model-id 已占用，请用新的 model-id")
    write_json(info_path, info)
    for scan, records in by_scan.items():
        write_json(path / "predictions" / (scan + ".json"), {"model_signature": info["signature"], "records": records})
    load_predictions(run_dir, model_id)
    return {"model_id": model_id, "frames": len(seen)}


def evaluate(run_dir, model_id, iou_thresholds=(.25, .5), distance_thresholds=(.25, .5, 1.),
             report_name="metrics"):
    protocol, audit, frames, gt = load_benchmark(run_dir)
    info, records, prediction_hash = load_predictions(run_dir, model_id)
    predictions = {key: r["detections"] for key, r in records.items()}
    metrics_protocol = {"iou_thresholds": list(iou_thresholds), "center_thresholds_m": list(distance_thresholds),
                        "interpolation": "101_recall_precision_envelope", "iou_backend": "CPU oriented convex polyhedron",
                        "ignore_matching": "one-to-one, valid GT first"}
    pair_cache = {}
    full = evaluate_ap(gt, predictions, iou_thresholds, distance_thresholds, _pair_cache=pair_cache)
    excluded = set(audit["official_training_visit_overlap"]["validation_scan_ids"])
    subsets = {}
    scopes = {"official_training_visit_disjoint": excluded}
    train = info.get("training_audit", {})
    if train.get("status") == "declared_annotations_audited":
        scopes["declared_model_training_disjoint"] = excluded | set(train["excluded_validation_scans"])
    for name, excluded_scans in scopes.items():
        ids = {f["frame_id"] for f in frames if f["video_id"] not in excluded_scans}
        subset = evaluate_ap({f: gt[f] for f in ids}, {f: predictions[f] for f in ids},
                             iou_thresholds, distance_thresholds, _pair_cache=pair_cache)
        subset["excluded_scan_ids"] = sorted(excluded_scans & {f["video_id"] for f in frames})
        subset["scan_count"] = len({f["video_id"] for f in frames if f["frame_id"] in ids})
        subset["visit_count"] = len({f["visit_id"] for f in frames if f["frame_id"] in ids})
        subsets[name] = subset
    report = {"model_id": model_id, "benchmark_id": protocol["benchmark_id"],
              "model_signature": info["signature"], "predictions_sha256": prediction_hash,
              "metrics_protocol": metrics_protocol, "training_audit": train,
              "scope": "ARKitScenes Validation sampled-frame, fixed 17-class prompts",
              "scan_count": len({f["video_id"] for f in frames}),
              "visit_count": len({f["visit_id"] for f in frames}), "complete": True,
              "full": full, "subsets": subsets, "created_utc": utc_now()}
    path = model_dir(run_dir, model_id) / "reports"
    name = safe_name(report_name)
    write_json(path / (name + ".json"), report)
    csv_text = io.StringIO()
    writer = csv.writer(csv_text)
    writer.writerow(["class_id", "class_name", "gt_count", "ignored_gt_count", "prediction_count"] +
                    [f"AP3D@{x:g}" for x in iou_thresholds] + [f"AP_center@{x:g}m" for x in distance_thresholds])
    for row in full["per_class"]:
        writer.writerow([row[k] for k in ("class_id", "class_name", "gt_count", "ignored_gt_count", "prediction_count")] +
                        [row["iou"][str(float(x))]["ap"] for x in iou_thresholds] +
                        [row["center"][str(float(x))]["ap"] for x in distance_thresholds])
    atomic_text(path / (name + "_per_class.csv"), csv_text.getvalue())
    csv_text = io.StringIO(); writer = csv.writer(csv_text)
    writer.writerow(["video_id", "visit_id", "frames", "valid_gt", "ignored_gt", "predictions"])
    for scan in audit["scans"]:
        scan_ids = [f["frame_id"] for f in frames if f["video_id"] == scan["video_id"]]
        writer.writerow([scan["video_id"], scan["visit_id"], len(scan_ids),
                         sum(not b["ignore"] for f in scan_ids for b in gt[f]),
                         sum(b["ignore"] for f in scan_ids for b in gt[f]),
                         sum(len(predictions[f]) for f in scan_ids)])
    atomic_text(path / (name + "_per_scan.csv"), csv_text.getvalue())
    return report
