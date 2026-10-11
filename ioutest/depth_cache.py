"""Selected ARKit LiDAR depth, kept separate from GT and RGB provenance."""

import hashlib
import io
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .common import file_digest, read_json, read_jsonl, write_json, write_jsonl
from .dataset import load_benchmark
from .image_cache import _cache_path, _write_bytes, default_cache_dir
from .schema import ModelInput


@dataclass(frozen=True)
class SensorModelInput(ModelInput):
    depth: np.ndarray | None = None
    gravity: np.ndarray | None = None


def _decode(blob, frame):
    with Image.open(io.BytesIO(blob)) as image:
        depth = np.asarray(image).copy()
    if depth.shape != tuple(frame["raw_image_hw"]) or depth.dtype.kind not in "iu":
        raise ValueError(f"深度必须与原始 RGB 同尺寸、使用整型毫米：{frame['frame_id']}")
    if (depth < 0).any() or (depth > 65535).any():
        raise ValueError("ARKit 深度超出 uint16 范围")
    return depth.astype("<u2")


def cache_depth(run_dir, image_cache=None, workers=4):
    if workers < 1:
        raise ValueError("workers 必须为正整数")
    protocol, _, frames, _ = load_benchmark(run_dir)
    root = Path(image_cache or default_cache_dir(run_dir)).expanduser().resolve()
    source = Path(protocol["settings"]["dataset_root"])
    if root == source or root.is_relative_to(source):
        raise ValueError("深度缓存不能写入原始数据集")
    summary_path = root / "depth_cache_summary.json"
    old = read_json(summary_path) if summary_path.exists() else None
    if old and old.get("benchmark_id") != protocol["benchmark_id"]:
        raise ValueError("深度缓存属于不同 benchmark")
    # Existing completed manifests are authoritative on resume, including pixel digests.
    prior = {}
    if old and old.get("complete"):
        if file_digest(root / "depths.jsonl") != old["manifest_sha256"]:
            raise ValueError("深度缓存清单已变化")
        prior = {r["frame_id"]: r for r in read_jsonl(root / "depths.jsonl")}
    summary = {"benchmark_id": protocol["benchmark_id"], "complete": False,
               "source": "ARKit lowres_depth, uint16 millimeters; not FARO/GT depth"}
    write_json(summary_path, summary)
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame["video_id"]].append(frame)

    def extract(scan, selected):
        archive, result = None, []
        try:
            for frame in selected:
                path = root / "depth_images" / scan / Path(frame["depth_entry"]).name
                if path.exists():
                    blob = path.read_bytes()
                else:
                    if archive is None:
                        archive = zipfile.ZipFile(source / "3dod/Validation" / (scan + ".zip"))
                    blob = archive.read(frame["depth_entry"])
                depth = _decode(blob, frame)
                sha = hashlib.sha256(depth.tobytes()).hexdigest()
                if frame["frame_id"] in prior and prior[frame["frame_id"]]["pixels_sha256"] != sha:
                    raise ValueError(f"已有深度图片变化：{path}")
                if not path.exists():
                    _write_bytes(path, blob)
                result.append({"frame_id": frame["frame_id"], "video_id": scan,
                    "depth_image": path.relative_to(root).as_posix(), "pixels_sha256": sha,
                    "png_sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
        finally:
            if archive is not None:
                archive.close()
        return result

    start, records = time.perf_counter(), {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract, scan, selected) for scan, selected in grouped.items()]
        for i, future in enumerate(as_completed(futures), 1):
            records.update((r["frame_id"], r) for r in future.result())
            if i % 25 == 0 or i == len(futures):
                print(f"深度缓存 {i}/{len(futures)} 扫描，{len(records)}/{len(frames)} 帧", flush=True)
    ordered = [records[f["frame_id"]] for f in frames]
    write_jsonl(root / "depths.jsonl", ordered)
    summary.update(complete=True, frames=len(frames), scans=len(grouped),
                   total_bytes=sum(r["bytes"] for r in ordered),
                   manifest_sha256=file_digest(root / "depths.jsonl"),
                   elapsed_seconds=time.perf_counter() - start)
    write_json(summary_path, summary)
    return summary


class DepthInputReader:
    def __init__(self, root, protocol, frames):
        self.root = Path(root).resolve()
        summary = read_json(self.root / "depth_cache_summary.json")
        if not summary.get("complete") or summary["benchmark_id"] != protocol["benchmark_id"]:
            raise ValueError("深度缓存未完成或 benchmark 不一致；请先执行 cache-depth")
        if file_digest(self.root / "depths.jsonl") != summary["manifest_sha256"]:
            raise ValueError("深度缓存清单摘要不一致")
        records = read_jsonl(self.root / "depths.jsonl")
        self.records = {r["frame_id"]: r for r in records}
        if (len(records) != len(self.records) or len(frames) != summary["frames"] or
                set(self.records) != {f["frame_id"] for f in frames}):
            raise ValueError("深度缓存帧清单不一致")
        for record in records:
            if not _cache_path(self.root, record["depth_image"]).is_file():
                raise FileNotFoundError(record["depth_image"])
        self.source_info = summary

    def read(self, frame):
        record = self.records[frame["frame_id"]]
        depth = _decode(_cache_path(self.root, record["depth_image"]).read_bytes(), frame)
        if hashlib.sha256(depth.tobytes()).hexdigest() != record["pixels_sha256"]:
            raise ValueError(f"深度图片变化：{frame['frame_id']}")
        return np.ascontiguousarray(np.rot90(depth.astype(np.float32) / 1000., frame["rotation_k"]))


def camera_gravity(frame):
    """Optional sensor-only gravity basis: ARKit world Z up -> camera Y down."""
    r = np.asarray(frame["camera_rotation"]) @ np.asarray(frame["world_to_raw_camera_rotation"])
    y = r @ np.array([0., 0., -1.])
    y /= np.linalg.norm(y)
    z = np.array([0., 0., 1.]) - y * y[2]
    if np.linalg.norm(z) < 1e-6:
        z = np.array([1., 0., 0.]) - y * y[0]
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.column_stack((x, y, np.cross(x, y)))
