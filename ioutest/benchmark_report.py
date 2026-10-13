"""Complete-input scoring and separate RGB, RGB-D and semantic reports."""

import csv
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .benchmark_worker import prediction_path, read_prediction
from .common import atomic_text, digest, file_digest, read_json, utc_now, write_json
from .dataset import load_benchmark
from .image_cache import FrameInputReader
from .metrics import evaluate_ap
from .schema import CLASSES
from .report_document import report_context, render_report
from .visualize import _draw


METRIC_COLUMNS = ["AP3D@0.25", "AP3D@0.50", "APcenter@0.25m", "APcenter@0.50m", "APcenter@1.00m"]


def metric_values(metrics):
    s = metrics["summary"]
    return [s["iou"]["map_per_threshold"][k] for k in ("0.25", "0.5")] + [
        s["center"]["map_per_threshold"][k] for k in ("0.25", "0.5", "1.0")]


def write_csv(path, header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header); writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def evaluate_model(run_dir, report_root, phase, model_id, model, frames):
    protocol, audit, _, gt = load_benchmark(run_dir)
    root = Path(report_root)
    model_root = root / "models" / model_id
    info = read_json(model_root / "model.json")
    if info["provenance"]["benchmark_id"] != protocol["benchmark_id"]:
        raise ValueError("数值预测的 benchmark 不一致")
    predictions, timings = {}, []
    records_identity = {}
    for frame in frames:
        record = read_prediction(prediction_path(model_root, frame), frame, info["signature"],
                                 protocol["benchmark_id"], model["semantic_labels"])
        predictions[frame["frame_id"]] = record["detections"]
        timings.append(record["timing"])
        records_identity[frame["frame_id"]] = digest(record)
    output = root / phase / "metrics" / (model_id + ".json")
    selection_id = read_json(root / phase / "selection.json")["selection_id"]
    identity = {"benchmark_id": protocol["benchmark_id"], "selection_id": selection_id,
                "model_signature": info["signature"], "predictions_sha256": digest(records_identity),
                "metric_implementation_sha256": digest({name: file_digest(Path(__file__).parent / name)
                                                          for name in ("metrics.py", "geometry.py", "schema.py")})}
    if output.exists():
        old = read_json(output)
        if old.get("identity") == identity and old.get("complete"):
            print(f"[{phase}] {model_id} AP 复用完整结果", flush=True)
            return old
    gt = {f["frame_id"]: gt[f["frame_id"]] for f in frames}
    excluded = set(audit["official_training_visit_overlap"]["validation_scan_ids"])
    disjoint = [f["frame_id"] for f in frames if f["video_id"] not in excluded]
    metrics = {}
    for mode, agnostic in (("geometry", True), ("semantic_17class", False)):
        if not agnostic and not model["semantic_labels"]:
            continue
        print(f"[{phase}] {model_id} 计算 {mode} AP：{len(frames)} 帧", flush=True)
        pairs = {}
        full = evaluate_ap(gt, predictions, _pair_cache=pairs, class_agnostic=agnostic)
        subset = evaluate_ap({i: gt[i] for i in disjoint}, {i: predictions[i] for i in disjoint},
                             _pair_cache=pairs, class_agnostic=agnostic)
        metrics[mode] = {"full": full, "official_training_visit_disjoint": subset}
        del pairs
    steady = [t for t in timings if not t["warmup"]]
    samples = steady or timings
    seconds = np.array([t["predict_seconds"] for t in samples])
    input_seconds = np.array([t["input_seconds"] for t in samples])
    peak = [t["peak_memory_bytes"] for t in timings if t["peak_memory_bytes"] is not None]
    result = {"complete": True, "identity": identity, "model_id": model_id, "phase": phase,
              "frames": len(frames), "scans": len({f["video_id"] for f in frames}),
              "visits": len({f["visit_id"] for f in frames}), "metrics": metrics,
              "excluded_scan_ids": sorted(excluded & {f["video_id"] for f in frames}),
              "excluded_frame_count": len(frames) - len(disjoint), "model": info,
              "runtime": {"samples": len(samples), "warmup_excluded": len(timings) - len(samples),
                  "warmup_only_fallback": not bool(steady),
                  "mean_predict_seconds": float(seconds.mean()), "median_predict_seconds": float(np.median(seconds)),
                  "p95_predict_seconds": float(np.percentile(seconds, 95)),
                  "mean_input_seconds": float(input_seconds.mean()),
                  "fps_predict": 1 / float(seconds.mean()),
                  "fps_input_and_predict": 1 / float((seconds + input_seconds).mean()),
                  "peak_memory_bytes": max(peak) if peak else None,
                  "timing_scope": "decode separately; predict includes native preprocess, forward, CPU box export; CUDA synchronized; disk writes excluded"},
              "created_utc": utc_now()}
    write_json(output, result)
    for mode, scope in metrics.items():
        rows = scope["full"]["per_class"]
        write_csv(output.with_name(model_id + "_" + mode + "_classes.csv"),
                  ["class", "valid_gt", "ignored_gt", "predictions", *METRIC_COLUMNS],
                  [[r["class_name"], r["gt_count"], r["ignored_gt_count"], r["prediction_count"],
                    *[r["iou"][k]["ap"] for k in ("0.25", "0.5")],
                    *[r["center"][k]["ap"] for k in ("0.25", "0.5", "1.0")]] for r in rows])
    grouped = {}
    for f in frames:
        grouped.setdefault((f["video_id"], f["visit_id"]), []).append(f["frame_id"])
    write_csv(output.with_name(model_id + "_scan_counts.csv"),
              ["scan", "visit", "frames", "valid_gt", "ignored_gt", "predictions"],
              [[scan, visit, len(ids), sum(not b["ignore"] for i in ids for b in gt[i]),
                sum(b["ignore"] for i in ids for b in gt[i]), sum(len(predictions[i]) for i in ids)]
               for (scan, visit), ids in grouped.items()])
    return result


def _table(results, keys, mode="geometry", scope="full"):
    rows = []
    for key in keys:
        if key in results and mode in results[key]["metrics"]:
            values = metric_values(results[key]["metrics"][mode][scope])
            rows.append([key, *values])
    lines = ["| 模型 | IoU AP25 | IoU AP50 | 中心 AP0.25m | 中心 AP0.5m | 中心 AP1m |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([row[0], *["—" if v is None else f"{100 * v:.2f}" for v in row[1:]]]) + " |")
    return rows, "\n".join(lines)


def _visualizations(run_dir, root, phase, results, frames, limit=12):
    protocol, _, all_frames, gt = load_benchmark(run_dir)
    inputs = FrameInputReader(run_dir, protocol, all_frames)
    grouped = {}
    for f in frames:
        grouped.setdefault(f["video_id"], []).append(f)
    chosen = [fs[len(fs) // 2] for fs in grouped.values()][:limit]
    saved = []
    for frame in chosen:
        with inputs.scan(frame["video_id"]) as reader:
            sample = reader.model_input(frame)
        h, w = sample.rgb.shape[:2]
        for key, report in results.items():
            record = read_prediction(prediction_path(root / "models" / key, frame), frame,
                                     report["identity"]["model_signature"], protocol["benchmark_id"],
                                     key in ("3dmood", "wilddet3d"))
            image = Image.fromarray(sample.rgb).resize((w * 3, h * 3))
            draw = ImageDraw.Draw(image)
            for box in gt[frame["frame_id"]]:
                if not box["ignore"]:
                    _draw(draw, box, sample.intrinsics, w, h, "lime", 3, "GT " + CLASSES[box["class_id"]])
            for box in record["detections"]:
                if box["score"] >= .1:
                    name = "object" if box["class_id"] is None else CLASSES[box["class_id"]]
                    _draw(draw, box, sample.intrinsics, w, h, "red", 3, f"{name} {box['score']:.2f}")
            path = root / phase / "visualizations" / key / (Path(frame["rgb_entry"]).stem + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            saved.append(str(path.relative_to(root / phase)))
    return saved


def _plot(root, rows):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "cache/matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [r[0] for r in rows]
    for ax, indexes, title in ((axes[0], (1, 2), "3D IoU AP"), (axes[1], (3, 4, 5), "3D center-distance AP")):
        x = np.arange(len(rows))
        width = .8 / len(indexes)
        for offset, i in enumerate(indexes):
            ax.bar(x + (offset - (len(indexes) - 1) / 2) * width,
                   [np.nan if r[i] is None else 100 * r[i] for r in rows], width,
                   label=METRIC_COLUMNS[i - 1])
        ax.set_xticks(x, labels, rotation=15); ax.set_ylabel("AP (%)"); ax.set_ylim(0, 100)
        ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "geometry_ap.png", dpi=160)
    fig.savefig(directory / "geometry_ap.svg")
    plt.close(fig)


def assemble_report(run_dir, report_root, phase, models, frames):
    root = Path(report_root)
    phase_root = root / phase
    results = {k: read_json(phase_root / "metrics" / (k + ".json")) for k in models}
    selection = read_json(phase_root / "selection.json")
    for key, result in results.items():
        if not result.get("complete") or result["identity"]["selection_id"] != selection["selection_id"]:
            raise ValueError(f"{key} 未完成当前采样清单，不能生成对比排名")
    main_rows, main_table = _table(results, ("3dmood", "wilddet3d", "cutr-rgb"))
    depth_rows, depth_table = _table(results, ("cutr-rgb", "cutr-rgbd"))
    semantic_rows, semantic_table = _table(results, ("3dmood", "wilddet3d"), "semantic_17class")
    disjoint_rows, disjoint_table = _table(results, ("3dmood", "wilddet3d", "cutr-rgb"),
                                          scope="official_training_visit_disjoint")
    for name, rows in (("comparison_rgb", main_rows), ("comparison_rgbd", depth_rows),
                       ("comparison_17class", semantic_rows), ("comparison_visit_disjoint", disjoint_rows)):
        write_csv(phase_root / (name + ".csv"), ["model", *METRIC_COLUMNS], rows)
    write_csv(phase_root / "runtime.csv", ["model", "frames", "mean_seconds", "p95_seconds", "fps", "peak_memory_bytes"],
              [[key, r["frames"], r["runtime"]["mean_predict_seconds"], r["runtime"]["p95_predict_seconds"],
                r["runtime"]["fps_predict"], r["runtime"]["peak_memory_bytes"]] for key, r in results.items()])
    _plot(phase_root, main_rows)
    images = _visualizations(run_dir, root, phase, results, frames)
    if "cutr-rgb-gravity" in results:
        rows, _ = _table(results, ("cutr-rgb", "cutr-rgb-gravity"))
        write_csv(phase_root / "comparison_gravity.csv", ["model", *METRIC_COLUMNS], rows)
    protocol, audit, all_frames, gt = load_benchmark(run_dir)
    context = report_context(run_dir, root, phase, protocol, audit, all_frames,
                             gt, frames, results, images)
    text = render_report(context, results, selection)
    write_json(phase_root / "report_context.json", context)
    atomic_text(phase_root / "report.md", text)
    write_json(phase_root / "summary.json", {"complete": True, "phase": phase, "selection": selection,
               "models": list(models), "frames": len(frames), "reports": {k: f"metrics/{k}.json" for k in results},
               "created_utc": utc_now()})
    atomic_text(root / "README.md", f"# 评测输出\n\n最新完成阶段：[{phase}]({phase}/report.md)，{len(frames):,} 帧。\n\n运行状态见 `status.json`；试评测与全量报告分别保存，不互相覆盖。\n")


def refresh_report(report_root, phase="full", run_dir=None, output=None):
    """Rewrite documentation from completed metrics; no model or AP execution.

    An alternate output path permits reviewing the report before replacing it.
    Relative artifact links continue to target the original phase directory.
    """
    root = Path(report_root).expanduser().resolve()
    if run_dir is None:
        run_dir = read_json(root / "experiment.json")["settings"]["run_dir"]
    protocol, audit, all_frames, gt = load_benchmark(run_dir)
    phase_root = root / phase
    selection = read_json(phase_root / "selection.json")
    summary = read_json(phase_root / "summary.json")
    if (not summary.get("complete") or selection["benchmark_id"] != protocol["benchmark_id"] or
            summary["selection"] != selection):
        raise ValueError("当前阶段尚未完成或数据清单不一致，不能刷新报告")
    by_id = {f["frame_id"]: f for f in all_frames}
    ids = selection["frame_ids"]
    if len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
        raise ValueError("阶段帧清单有重复或包含未准备的帧")
    frames = [by_id[i] for i in ids]
    results = {key: read_json(phase_root / "metrics" / (key + ".json")) for key in summary["models"]}
    if not results or not frames:
        raise ValueError("不能从空阶段生成报告")
    for key, result in results.items():
        identity = result["identity"]
        if (not result.get("complete") or result["frames"] != len(frames) or
                identity["selection_id"] != selection["selection_id"] or
                identity["benchmark_id"] != protocol["benchmark_id"]):
            raise ValueError(f"{key} 指标尚未完成或对应其他阶段，不能刷新报告")
        saved = read_json(root / "models" / key / "model.json")
        # A later full-stage worker may rewrite created_utc for the same model.
        # Compare prediction identity and provenance, not the metadata write time.
        if (saved["signature"] != identity["model_signature"] or
                saved["provenance"] != result["model"]["provenance"]):
            raise ValueError(f"{key} 模型来源与已保存指标不一致")
    groups = {}
    for frame in frames:
        groups.setdefault(frame["video_id"], []).append(frame)
    chosen = [fs[len(fs) // 2] for fs in groups.values()][:12]
    images = []
    for frame in chosen:
        for key in results:
            path = phase_root / "visualizations" / key / (Path(frame["rgb_entry"]).stem + ".png")
            if path.is_file():
                images.append(str(path.relative_to(phase_root)))
    context = report_context(run_dir, root, phase, protocol, audit, all_frames,
                             gt, frames, results, images)
    text = render_report(context, results, selection)
    destination = Path(output).expanduser().resolve() if output else phase_root / "report.md"
    atomic_text(destination, text)
    context_path = destination.with_name("report_context.json")
    write_json(context_path, context)
    return {"report": str(destination), "context": str(context_path), "frames": len(frames),
            "models": list(results), "inference_rerun": False, "ap_recalculated": False}


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description="从已完成 benchmark 结果补全中文报告，不重新推理或计算 AP")
    parser.add_argument("--report-root", type=Path, default=Path("~/report/arkitscenes-three-models-v1"))
    parser.add_argument("--phase", choices=("pilot", "full", "cpu-smoke"), default="full")
    parser.add_argument("--run-dir", type=Path, help="冻结数据目录，默认读取 experiment.json 中的历史路径")
    parser.add_argument("--output", type=Path, help="可选的报告输出文件；省略则更新原阶段 report.md；同目录写 report_context.json")
    args = parser.parse_args(argv)
    print(json.dumps(refresh_report(args.report_root, args.phase, args.run_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
