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
    example = next(iter(results.values()))
    text = f"""# ARKitScenes 三项目评测：{'200 帧试评测' if phase == 'pilot' and len(frames) == 200 else phase}

本阶段全部指定模型已完成：{len(frames):,} 帧、{example['scans']} 个扫描、{example['visits']} 个 visit。数值为实际预测计算结果；AP 表格单位为百分数，CSV 使用 0～1。benchmark_id：`{selection['benchmark_id']}`。

## RGB 几何 AP：主要比较

三组均输入相同的转正 RGB 和相机内参。匹配忽略类别；3D-MOOD/WildDet3D 使用固定 17 类文本，CuTR 为类别无关检测。

{main_table}

## CuTR RGB-D：深度输入补充实验

RGB-D 使用同帧 ARKit lowres_depth 的米制深度，与 RGB 一起转正；不使用 FARO 深度或 GT 三维框。RGB-D 权重与 RGB 权重不同，因此差异同时包含模型训练与输入变化。

{depth_table}

## 17 类语义 AP

按类别分别计算 AP 后对有有效 GT 的类别取均值；CuTR 没有这 17 类语义标签，不纳入此表。

{semantic_table}

## 官方训练 visit 排除后的 RGB 几何 AP

本阶段排除 {example['excluded_frame_count']} 帧、扫描 {example['excluded_scan_ids']}。这只审计 ARKitScenes 官方训练/验证元数据重叠，不能证明各模型未训练过这些场景。

{disjoint_table}

## 评测约定与边界

- 3D IoU 使用完整旋转框体积交并比，阈值 0.25、0.50；中心距离单位米，阈值 0.25、0.50、1.00；101 点插值 AP。
- 每帧按分数降序一对一匹配，优先有效 GT，然后一对一匹配 ignore GT；空 GT 帧仍参与误检计数。
- GT 筛选沿用冻结协议：视野内且框内观测深度点严格大于 20。GT 只提供给评分器与可视化。
- AP 不使用 Demo 的显示分数阈值。3D-MOOD/WildDet3D 保留各自原生 NMS，最多 300 框；CuTR 在原生 top-100 之前只选择监督前景通道 0，避免另一通道占满候选；候选规则随模型来源一起记录。
- 当前图像为 ARKitScenes 冻结 lowres_wide 帧，原始 256×192。CuTR 长边缩放至 1024（RGB 双线性、深度最近邻、同步更新像素中心内参），再使用原生预处理；两输入组尺寸相同。不能直接与论文原始设置的 AP 对比。
- CuTR 的通用物体可能超出 ARKitScenes 17 类标注范围，这些未标注物体可能计为误检；因此本报告是限定数据标注范围的几何比较。
- 默认 CuTR RGB/RGB-D 不输入传感器重力，采用原生模型相机直立假设；已知重力可以另加独立实验，不混入 RGB 主表。
- GPU 时间通过同步测量，排除每次模型加载后的首帧，包含模型预处理、前向和数值导出；输入解码时间独立记录。显存为 PyTorch 最大已分配量，不等同于设备总占用。

## 输出

`metrics/` 保存完整指标、逐类 AP 和逐扫描计数；`runtime.csv` 保存性能；`logs/` 保存各模型日志；`../models/` 保存权重/环境/源码摘要和逐帧压缩数值预测，续跑可复用；`visualizations/` 的绿框为 GT，红框为预测，显示分数阈值 0.1 不影响 AP。共有 {len(images)} 张检查图。

![RGB 几何 AP](figures/geometry_ap.png)
"""
    if phase == "cpu-smoke":
        text = text.replace("# ARKitScenes 三项目评测：cpu-smoke", "# CPU 单帧链路验证")
        text = text.replace("本阶段全部指定模型已完成：", "本报告仅验证真实单帧 CPU 执行链路，不作为正式效果或设备性能比较。\n\n本次完成：")
    if "cutr-rgb-gravity" in results:
        rows, table = _table(results, ("cutr-rgb", "cutr-rgb-gravity"))
        write_csv(phase_root / "comparison_gravity.csv", ["model", *METRIC_COLUMNS], rows)
        text += "\n## CuTR RGB + 已知重力辅助实验\n\n重力只来自 ARKit 相机姿态，不读取 GT 框；这是额外传感器信息，单独比较。\n\n" + table + "\n"
    atomic_text(phase_root / "report.md", text)
    write_json(phase_root / "summary.json", {"complete": True, "phase": phase, "selection": selection,
               "models": list(models), "frames": len(frames), "reports": {k: f"metrics/{k}.json" for k in results},
               "created_utc": utc_now()})
    atomic_text(root / "README.md", f"# 评测输出\n\n最新完成阶段：[{phase}]({phase}/report.md)，{len(frames):,} 帧。\n\n运行状态见 `status.json`；试评测与全量报告分别保存，不互相覆盖。\n")
