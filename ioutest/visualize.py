"""GT/prediction projection checks without matplotlib or GPU dependencies."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import safe_name
from .dataset import ScanReader, load_benchmark, metadata
from .geometry import EDGES, corners, projected_box
from .schema import CLASSES


def _draw(draw, box, intrinsics, width, height, color, scale, text):
    bbox = projected_box(box, intrinsics, width, height)
    if bbox is None:
        return
    points = corners(box)
    near = 1e-4
    for i, j in EDGES:
        a, b = points[i].copy(), points[j].copy()
        if a[2] < near and b[2] < near:
            continue
        if a[2] < near:
            a += (near - a[2]) / (b[2] - a[2]) * (b - a)
        elif b[2] < near:
            b += (near - b[2]) / (a[2] - b[2]) * (a - b)
        projected = np.stack((a, b)) @ np.asarray(intrinsics).T
        uv = projected[:, :2] / projected[:, 2:]
        # Avoid passing astronomical near-plane coordinates to the drawing backend.
        uv = np.clip(uv, -2 * max(width, height), 3 * max(width, height))
        draw.line([tuple(p * scale) for p in uv], fill=color, width=2)
    draw.text((bbox[0] * scale, bbox[1] * scale), text, fill=color)


def visualize(run_dir, model_id=None, max_frames=12, score_threshold=.1, show_ignore=False):
    if max_frames < 1 or not np.isfinite(score_threshold):
        raise ValueError("可视化帧数须为正数，显示分数须为有限数值")
    protocol, _, frames, gt = load_benchmark(run_dir)
    predictions = {}
    if model_id:
        from .pipeline import load_predictions
        _, records, _ = load_predictions(run_dir, model_id)
        predictions = {f: r["detections"] for f, r in records.items()}
    rows = metadata(protocol["settings"]["dataset_root"])
    # Choose central frames round-robin across orientations and scans.
    by_scan = {}
    for frame in frames:
        by_scan.setdefault(frame["video_id"], []).append(frame)
    pools = {}
    for scan_frames in by_scan.values():
        frame = scan_frames[len(scan_frames) // 2]
        pools.setdefault(frame["sky_direction"], []).append(frame)
    chosen = []
    while pools and len(chosen) < max_frames:
        for direction in sorted(list(pools)):
            chosen.append(pools[direction].pop(0))
            if not pools[direction]:
                del pools[direction]
            if len(chosen) >= max_frames:
                break
    output = Path(run_dir) / "visualizations" / (safe_name(model_id) if model_id else "groundtruth")
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for frame in chosen:
        with ScanReader(protocol["settings"]["dataset_root"], rows[frame["video_id"]]) as reader:
            sample = reader.model_input(frame)
        h, w = sample.rgb.shape[:2]
        scale = 3
        image = Image.fromarray(sample.rgb).resize((w * scale, h * scale))
        canvas = Image.new("RGB", (w * scale, h * scale + 36), "black")
        canvas.paste(image, (0, 36))
        # Draw boxes on image first so image coordinates remain simple.
        draw = ImageDraw.Draw(image)
        for box in gt[frame["frame_id"]]:
            if box["ignore"] and not show_ignore:
                continue
            color = "gray" if box["ignore"] else "lime"
            _draw(draw, box, sample.intrinsics, w, h, color, scale,
                  f"GT {CLASSES[box['class_id']]} ({box['observed_depth_points']} pts)")
        for box in predictions.get(frame["frame_id"], []):
            if box["score"] >= score_threshold:
                _draw(draw, box, sample.intrinsics, w, h, "red", scale,
                      f"P {CLASSES[box['class_id']]} {box['score']:.2f}")
        canvas.paste(image, (0, 36))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 3), f"{frame['video_id']} t={frame['timestamp']:.3f} {frame['sky_direction']}", fill="white")
        draw.text((4, 18), "Green: GT   Red: prediction   Gray: ignored GT", fill="white")
        path = output / (frame["frame_id"].replace(":", "_") + ".png")
        canvas.save(path)
        saved.append(str(path))
    return saved
