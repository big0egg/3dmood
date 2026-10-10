"""Read-only ARKitScenes ZIP sampling and camera-frame ground truth."""

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .common import (digest, file_digest, implementation_digest, read_json,
                     read_jsonl, utc_now, write_json, write_jsonl)
from .geometry import inside, projected_box, transform_box, upright_parameters
from .schema import CLASSES, PROMPTS, SCHEMA_VERSION, class_id, validate_box


DEFAULT_ROOT = Path("/mnt/data/openBenchmark/ARKitScenes")


def metadata(root):
    with (Path(root) / "source/metadata.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("metadata.csv 有重复 video_id")
    return {r["video_id"]: r for r in rows}


def official_overlap(rows):
    train = {r["visit_id"] for r in rows.values()
             if r["fold"] == "Training" and r["visit_id"] not in ("NA", "")}
    val = {r["visit_id"] for r in rows.values()
           if r["fold"] == "Validation" and r["visit_id"] not in ("NA", "")}
    common = train & val
    return {"visit_ids": sorted(common), "validation_scan_ids": sorted(
        r["video_id"] for r in rows.values()
        if r["fold"] == "Validation" and r["visit_id"] in common)}


def _timestamp(name):
    return float(Path(name).stem.rsplit("_", 1)[1])


class ScanReader:
    def __init__(self, root, scene):
        self.root, self.scene = Path(root), scene
        self.video_id = scene["video_id"]
        self.path = self.root / "3dod" / scene["fold"] / (self.video_id + ".zip")

    def __enter__(self):
        self.archive = zipfile.ZipFile(self.path)
        self.prefix = f"{self.video_id}/{self.video_id}_frames/"
        return self

    def __exit__(self, *args):
        self.archive.close()

    def image(self, name, rgb=False):
        with Image.open(io.BytesIO(self.archive.read(name))) as image:
            return np.asarray(image.convert("RGB") if rgb else image).copy()

    def annotations(self):
        doc = json.loads(self.archive.read(f"{self.video_id}/{self.video_id}_3dod_annotation.json"))
        if doc.get("skipped", False):
            return True, []
        boxes = []
        for row in doc["data"]:
            obb = row["segments"]["obbAligned"]
            rotation = np.asarray(obb["normalizedAxes"], dtype=float).reshape(3, 3).T
            # Reversing a local axis preserves an OBB's geometry and fixes handedness.
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            box = {"uid": row["uid"], "class_id": class_id(row["label"]),
                   "center": obb["centroid"], "size": obb["axesLengths"],
                   "rotation": rotation.tolist()}
            validate_box(box)
            boxes.append(box)
        if len({b["uid"] for b in boxes}) != len(boxes):
            raise ValueError(f"扫描 {self.video_id} 的 GT uid 重复")
        return bool(doc.get("skipped", False)), boxes

    def candidates(self, minimum_interval):
        names = set(self.archive.namelist())
        trajectory = np.loadtxt(io.StringIO(self.archive.read(self.prefix + "lowres_wide.traj").decode()),
                                ndmin=2)
        if trajectory.shape[1] != 7 or not np.isfinite(trajectory).all():
            raise ValueError(f"扫描 {self.video_id} 的轨迹形状或数值非法")
        trajectory = trajectory[np.argsort(trajectory[:, 0])]
        intrinsics = {Path(n).stem: n for n in names
                      if n.startswith(self.prefix + "lowres_wide_intrinsics/") and n.endswith(".pincam")}
        rgb = sorted((n for n in names if n.startswith(self.prefix + "lowres_wide/")
                      and n.endswith(".png")), key=_timestamp)
        errors = Counter()
        frames, last = [], -float("inf")
        for name in rgb:
            stem = Path(name).stem
            timestamp = _timestamp(name)
            depth = self.prefix + "lowres_depth/" + stem + ".png"
            if depth not in names:
                errors["missing_depth"] += 1; continue
            key = stem
            if key not in intrinsics:
                options = [f"{self.video_id}_{timestamp + offset:.3f}" for offset in (-.001, .001)]
                key = next((k for k in options if k in intrinsics), None)
            if key is None:
                errors["missing_intrinsics"] += 1; continue
            index = int(np.searchsorted(trajectory[:, 0], timestamp))
            indexes = [j for j in (index - 1, index) if 0 <= j < len(trajectory)]
            line = trajectory[min(indexes, key=lambda j: abs(trajectory[j, 0] - timestamp))]
            delta = float(abs(line[0] - timestamp))
            if delta >= .005:
                errors["pose_timestamp_gap"] += 1; continue
            if timestamp - last < minimum_interval - 1e-9:
                continue
            last = timestamp
            frames.append({"frame_id": f"{self.video_id}:{stem}", "video_id": self.video_id,
                           "visit_id": self.scene["visit_id"], "timestamp": timestamp,
                           "rgb_entry": name, "depth_entry": depth,
                           "intrinsics_entry": intrinsics[key], "trajectory": line.tolist(),
                           "pose_delta_seconds": delta,
                           "intrinsics_delta_seconds": abs(_timestamp(intrinsics[key]) - timestamp),
                           "sky_direction": self.scene["sky_direction"]})
        return frames, len(rgb), dict(errors)

    def prepare_frame(self, frame, boxes, depth_stride, visible_points):
        rgb = self.image(frame["rgb_entry"], rgb=True)
        depth = self.image(frame["depth_entry"]).astype(float) / 1000.
        h, w = rgb.shape[:2]
        values = np.loadtxt(io.StringIO(self.archive.read(frame["intrinsics_entry"]).decode()))
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"{frame['frame_id']} 的内参非法")
        iw, ih, fx, fy, cx, cy = values
        if (int(iw), int(ih)) != (w, h) or depth.shape != (h, w) or fx <= 0 or fy <= 0:
            raise ValueError(f"{frame['frame_id']} 的 RGB、深度与内参尺寸不一致")
        intrinsic = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        k, q, upright_k = upright_parameters(intrinsic, w, h, frame["sky_direction"])
        upright_rgb = np.rot90(rgb, k)
        uh, uw = upright_rgb.shape[:2]
        line = np.asarray(frame["trajectory"])
        world_to_camera = Rotation.from_rotvec(line[1:4]).as_matrix()
        translation = line[4:7]
        yy, xx = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
        dd = depth[::depth_stride, ::depth_stride].reshape(-1)
        valid = np.isfinite(dd) & (dd > 0)
        pixels = np.stack((xx.reshape(-1), yy.reshape(-1), np.ones(dd.size)), axis=1)[valid]
        camera_points = (pixels @ np.linalg.inv(intrinsic).T) * dd[valid, None]
        world_points = (camera_points - translation) @ world_to_camera
        gt = []
        for box in boxes:
            count = int(inside(world_points, box).sum())
            transformed = transform_box(box, q @ world_to_camera, q @ translation)
            projected = projected_box(transformed, upright_k, uw, uh)
            reasons = []
            if count <= visible_points:
                reasons.append("insufficient_observed_depth_points")
            if projected is None:
                reasons.append("outside_camera_view")
            transformed.update(ignore=bool(reasons), ignore_reasons=reasons,
                               observed_depth_points=count, projected_bbox=projected)
            validate_box(transformed)
            gt.append(transformed)
        out = dict(frame, intrinsics=upright_k.tolist(), raw_intrinsics=intrinsic.tolist(),
                   rgb_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                   image_hw=[uh, uw], raw_image_hw=[h, w], camera_rotation=q.tolist(),
                   world_to_raw_camera_rotation=world_to_camera.tolist(),
                   world_to_raw_camera_translation=translation.tolist(), rotation_k=k)
        return out, gt

    def model_input(self, frame):
        from .schema import ModelInput
        rgb = self.image(frame["rgb_entry"], rgb=True)
        if list(rgb.shape[:2]) != frame["raw_image_hw"]:
            raise ValueError("原始图片尺寸发生变化，请重新准备数据")
        if hashlib.sha256(rgb.tobytes()).hexdigest() != frame["rgb_pixels_sha256"]:
            raise ValueError("原始图片内容发生变化，请重新准备数据")
        rgb = np.ascontiguousarray(np.rot90(rgb, frame["rotation_k"]))
        return ModelInput(frame["frame_id"], rgb, np.asarray(frame["intrinsics"]))


def _sample(frames, count):
    if len(frames) <= count:
        return frames
    times = np.asarray([f["timestamp"] for f in frames])
    indexes = {int(np.argmin(abs(times - target))) for target in np.linspace(times[0], times[-1], count)}
    return [frames[index] for index in sorted(indexes)]


def prepare(root, run_dir, scan_ids=None, max_scans=None, frames_per_scan=20,
            minimum_interval=.5, depth_stride=2, visible_points=20):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if root == run_dir or root in run_dir.parents:
        raise ValueError("评测输出必须放在原始数据目录之外")
    if frames_per_scan < 1 or depth_stride < 1 or visible_points < 0 or not np.isfinite(minimum_interval) or minimum_interval < 0:
        raise ValueError("帧数/深度步长须为正数，时间间隔和可见点数须非负")
    rows = metadata(root)
    validation = sorted(r["video_id"] for r in rows.values() if r["fold"] == "Validation")
    selected = sorted(set(scan_ids)) if scan_ids else validation
    if not set(selected) <= set(validation):
        raise ValueError("只能选择 metadata 中的 Validation 扫描")
    if max_scans is not None:
        if max_scans < 1:
            raise ValueError("max_scans 必须为正数")
        # Fixed permutation, one scan per visit first, then remaining scans.
        rng = np.random.default_rng(20261010)
        pools = {}
        for scan in rng.permutation(selected):
            pools.setdefault(rows[scan]["sky_direction"], []).append(scan)
        shuffled = []
        while pools:
            for direction in sorted(list(pools)):
                shuffled.append(pools[direction].pop())
                if not pools[direction]:
                    del pools[direction]
        visits, first, rest = set(), [], []
        for scan in shuffled:
            visit = rows[scan]["visit_id"]
            (rest if visit in visits else first).append(scan)
            visits.add(visit)
        selected = sorted((first + rest)[:max_scans])
    if not selected:
        raise ValueError("没有选中验证扫描")
    source = {"metadata_sha256": file_digest(root / "source/metadata.csv")}
    manifest_path = root / "verified_manifest.jsonl"
    manifest = {}
    if manifest_path.is_file():
        source["download_manifest_sha256"] = file_digest(manifest_path)
        manifest = {r["video_id"]: r for r in read_jsonl(manifest_path)}
    settings = {"dataset_root": str(root), "scan_ids": selected, "frames_per_scan": frames_per_scan,
                "minimum_interval_seconds": minimum_interval, "depth_stride": depth_stride,
                "visible_points_strictly_greater_than": visible_points,
                "schema_version": SCHEMA_VERSION, "classes": list(CLASSES), "prompts": list(PROMPTS),
                "implementation_sha256": implementation_digest(), "source": source}
    signature = digest(settings)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol["signature"] != signature:
            raise ValueError("输出目录已有不同协议或版本的数据，请使用新的 run-dir")
        if protocol.get("complete"):
            load_benchmark(run_dir)
            print(f"复用已验证的数据清单：{run_dir}", flush=True)
            return protocol
    write_json(protocol_path, {"signature": signature, "settings": settings, "complete": False})
    all_frames, scans, counts, excluded = [], [], Counter(), []
    for index, scan in enumerate(selected):
        path = run_dir / "groundtruth" / (scan + ".json")
        if path.exists():
            result = read_json(path)
            if result.get("signature") != signature:
                raise ValueError("扫描缓存协议不一致")
        else:
            scene = rows[scan]
            try:
                with ScanReader(root, scene) as reader:
                    skipped, boxes = reader.annotations()
                    result = {"signature": signature, "scene": scene, "frames": [],
                              "scene_class_counts": dict(Counter(CLASSES[b["class_id"]] for b in boxes)),
                              "archive_bytes": reader.path.stat().st_size,
                              "archive_sha256_from_download_manifest": manifest.get(scan, {}).get("sha256")}
                    if skipped:
                        result["excluded_reason"] = "official_skipped"
                    else:
                        candidates, total, errors = reader.candidates(minimum_interval)
                        result.update(rgb_frames=total, candidate_frames=len(candidates), pairing_errors=errors)
                        if not candidates:
                            raise ValueError("扫描没有可配对帧，不能静默排除")
                        for frame in _sample(candidates, frames_per_scan):
                            frame, gt = reader.prepare_frame(frame, boxes, depth_stride, visible_points)
                            result["frames"].append({"frame": frame, "boxes": gt})
                write_json(path, result)
            except Exception as exc:
                write_json(run_dir / "prepare_error.json", {"video_id": scan, "error": str(exc), "utc": utc_now()})
                raise RuntimeError(f"准备扫描 {scan} 失败：{exc}") from exc
        frames = [r["frame"] for r in result["frames"]]
        all_frames.extend(frames)
        counts.update(result["scene_class_counts"])
        scans.append({"video_id": scan, "visit_id": rows[scan]["visit_id"],
                      "sky_direction": rows[scan]["sky_direction"], "frame_count": len(frames),
                      "groundtruth_file_sha256": file_digest(path),
                      "pairing_errors": result.get("pairing_errors", {})})
        if "excluded_reason" in result:
            excluded.append({"video_id": scan, "reason": result["excluded_reason"]})
        print(f"准备 {index + 1}/{len(selected)} 扫描 {scan}: {len(frames)} 帧", flush=True)
    if not all_frames:
        raise ValueError("清单无可评测帧")
    if len({f["frame_id"] for f in all_frames}) != len(all_frames):
        raise ValueError("清单存在重复 frame_id")
    write_jsonl(run_dir / "frames.jsonl", all_frames)
    audit = {"source_validation_scan_count": len(validation), "selected_scan_count": len(selected),
             "selected_visit_count": len({rows[s]["visit_id"] for s in selected}),
             "frame_count": len(all_frames), "scene_gt_class_counts": dict(counts), "excluded_scans": excluded,
             "official_training_visit_overlap": official_overlap(rows), "scans": scans,
             "verification": "ZIP entries read with CRC; archive SHA recorded from download manifest, not rehashed"}
    write_json(run_dir / "dataset_audit.json", audit)
    protocol = {"signature": signature, "settings": settings, "complete": True, "created_utc": utc_now(),
                "frames_sha256": file_digest(run_dir / "frames.jsonl"), "audit_sha256": file_digest(run_dir / "dataset_audit.json")}
    protocol["benchmark_id"] = digest({"signature": signature, "frames": protocol["frames_sha256"],
                                        "audit": protocol["audit_sha256"]})
    write_json(protocol_path, protocol)
    return protocol


def load_benchmark(run_dir):
    run_dir = Path(run_dir)
    protocol = read_json(run_dir / "protocol.json")
    if not protocol.get("complete"):
        raise ValueError("数据准备尚未完成，请先执行 prepare")
    for name, key in (("frames.jsonl", "frames_sha256"), ("dataset_audit.json", "audit_sha256")):
        if file_digest(run_dir / name) != protocol[key]:
            raise ValueError(f"{name} 已变化，不能继续复用清单")
    audit = read_json(run_dir / "dataset_audit.json")
    frames = read_jsonl(run_dir / "frames.jsonl")
    gt = {}
    for scan in audit["scans"]:
        path = run_dir / "groundtruth" / (scan["video_id"] + ".json")
        if file_digest(path) != scan["groundtruth_file_sha256"]:
            raise ValueError(f"真值文件 {path.name} 已变化")
        for row in read_json(path)["frames"]:
            gt[row["frame"]["frame_id"]] = row["boxes"]
    if set(gt) != {f["frame_id"] for f in frames}:
        raise ValueError("GT 与帧清单不一致")
    return protocol, audit, frames, gt
