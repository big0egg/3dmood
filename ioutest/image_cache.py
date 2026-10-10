"""Local selected-frame PNG cache; frozen benchmark pixels remain authoritative."""

import hashlib
import io
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from .common import file_digest, read_json, read_jsonl, write_json, write_jsonl
from .dataset import ScanReader, load_benchmark
from .schema import ModelInput


def default_cache_dir(run_dir):
    return Path(__file__).resolve().parent / "cache" / Path(run_dir).resolve().name


def _checked_rgb(rgb, frame):
    if list(rgb.shape[:2]) != frame["raw_image_hw"]:
        raise ValueError(f"{frame['frame_id']} 的原始图片尺寸发生变化")
    if hashlib.sha256(rgb.tobytes()).hexdigest() != frame["rgb_pixels_sha256"]:
        raise ValueError(f"{frame['frame_id']} 的原始图片内容发生变化，不能继续评测")
    return np.ascontiguousarray(np.rot90(rgb, frame["rotation_k"]))


def _cache_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("缓存图片必须使用相对路径")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("缓存图片路径不能越出缓存目录")
    return path


class _CachedScanReader:
    def __init__(self, root, records):
        self.root, self.records = root, records

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def model_input(self, frame):
        record = self.records[frame["frame_id"]]
        with Image.open(_cache_path(self.root, record["raw_image"])) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        rgb = _checked_rgb(rgb, frame)
        return ModelInput(frame["frame_id"], rgb, np.asarray(frame["intrinsics"]))


class FrameInputReader:
    """None: auto-detect cache; False: use ZIP; path: require that cache."""

    def __init__(self, run_dir, protocol, frames, image_cache=None):
        self.dataset_root = Path(protocol["settings"]["dataset_root"])
        self.cache_root, self.records = None, None
        candidate = default_cache_dir(run_dir) if image_cache is None else image_cache
        if candidate is False or (image_cache is None and not candidate.exists()):
            self.source_info = {"kind": "zip", "root": str(self.dataset_root)}
            return
        candidate = Path(candidate).expanduser().resolve()
        summary_path = candidate / "cache_summary.json"
        if not summary_path.is_file():
            raise ValueError(f"图片缓存尚未完成：{candidate}；请先执行 cache-images")
        summary = read_json(summary_path)
        if not summary.get("complete") or summary.get("benchmark_id") != protocol["benchmark_id"]:
            raise ValueError("图片缓存未完成或 benchmark_id 不一致，不能混用")
        manifest_path = candidate / "images.jsonl"
        if file_digest(manifest_path) != summary.get("images_manifest_sha256"):
            raise ValueError("图片缓存清单摘要不一致")
        records = read_jsonl(manifest_path)
        self.records = {r["frame_id"]: r for r in records}
        expected = {f["frame_id"] for f in frames}
        if (len(self.records) != len(records) or set(self.records) != expected
                or summary.get("frames") != len(frames)):
            raise ValueError("图片缓存帧编号重复、缺失或与评测清单不一致")
        for frame in frames:
            record = self.records[frame["frame_id"]]
            if record.get("raw_rgb_pixels_sha256") != frame["rgb_pixels_sha256"]:
                raise ValueError("图片缓存原图摘要与冻结清单不一致")
            path = _cache_path(candidate, record["raw_image"])
            if not path.is_file():
                raise ValueError(f"图片缓存缺少原图：{path}")
        self.cache_root = candidate
        self.source_info = {"kind": "local_png", "root": str(candidate),
                            "manifest_sha256": summary["images_manifest_sha256"]}

    def scan(self, video_id):
        if self.cache_root is not None:
            return _CachedScanReader(self.cache_root, self.records)
        # Frozen benchmarks contain Validation frames; inference needs no remote CSV.
        return ScanReader(self.dataset_root, {"video_id": video_id, "fold": "Validation"})


def _write_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(value)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def cache_images(run_dir, image_cache=None, workers=4):
    """Materialize only frozen selected RGB frames, resumable per image."""
    if workers < 1:
        raise ValueError("workers 必须为正整数")
    protocol, _, frames, _ = load_benchmark(run_dir)
    root = Path(image_cache or default_cache_dir(run_dir)).expanduser().resolve()
    source = Path(protocol["settings"]["dataset_root"])
    if root == source or root.is_relative_to(source):
        raise ValueError("图片缓存不能放在原始数据集目录内")
    summary_path = root / "cache_summary.json"
    if summary_path.exists() and read_json(summary_path).get("benchmark_id") != protocol["benchmark_id"]:
        raise ValueError("缓存目录已有不同 benchmark，请选择新的 image-cache")
    summary = {"benchmark_id": protocol["benchmark_id"], "complete": False,
               "source_root": str(source), "cache_root": str(root), "expected_frames": len(frames)}
    write_json(summary_path, summary)
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame["video_id"]].append(frame)

    def extract(scan, selected):
        records = []
        # Open ZIP lazily so a complete local cache can be revalidated offline.
        archive = None
        try:
            for frame in selected:
                name = Path(frame["rgb_entry"]).name
                raw = root / "raw_images" / scan / name
                upright = root / "images" / scan / name
                if raw.is_file():
                    blob = raw.read_bytes()
                else:
                    if archive is None:
                        import zipfile
                        archive = zipfile.ZipFile(source / "3dod/Validation" / (scan + ".zip"))
                    blob = archive.read(frame["rgb_entry"])
                with Image.open(io.BytesIO(blob)) as image:
                    rgb = np.asarray(image.convert("RGB")).copy()
                oriented = _checked_rgb(rgb, frame)
                if not raw.is_file():
                    _write_bytes(raw, blob)
                if upright.is_file():
                    with Image.open(upright) as image:
                        if not np.array_equal(np.asarray(image.convert("RGB")), oriented):
                            raise ValueError(f"已有转正图片内容不一致：{upright}")
                else:
                    buffer = io.BytesIO()
                    Image.fromarray(oriented).save(buffer, format="PNG")
                    _write_bytes(upright, buffer.getvalue())
                records.append({"frame_id": frame["frame_id"], "video_id": scan,
                    "raw_image": raw.relative_to(root).as_posix(), "image": upright.relative_to(root).as_posix(),
                    "intrinsics": frame["intrinsics"], "image_hw": frame["image_hw"],
                    "rotation_k": frame["rotation_k"], "raw_rgb_pixels_sha256": frame["rgb_pixels_sha256"],
                    "rgb_pixels_sha256": hashlib.sha256(oriented.tobytes()).hexdigest(),
                    "raw_png_sha256": hashlib.sha256(blob).hexdigest(),
                    "raw_bytes": raw.stat().st_size, "upright_bytes": upright.stat().st_size})
        finally:
            if archive is not None:
                archive.close()
        return records

    start = time.perf_counter()
    records = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract, scan, selected) for scan, selected in grouped.items()]
        for index, future in enumerate(as_completed(futures), 1):
            records.update((r["frame_id"], r) for r in future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"缓存 {index}/{len(futures)} 扫描，{len(records)}/{len(frames)} 帧", flush=True)
    ordered = [records[f["frame_id"]] for f in frames]
    raw_bytes = sum(r.pop("raw_bytes") for r in ordered)
    upright_bytes = sum(r.pop("upright_bytes") for r in ordered)
    write_jsonl(root / "images.jsonl", ordered)
    summary.update(complete=True, frames=len(frames), scans=len(grouped), raw_bytes=raw_bytes,
                   upright_bytes=upright_bytes, total_bytes=raw_bytes + upright_bytes,
                   elapsed_seconds=round(time.perf_counter() - start, 3),
                   images_manifest_sha256=file_digest(root / "images.jsonl"),
                   note="推理读取 raw_images 并转正一次；images 为转正图片，配套内参在 images.jsonl 中")
    write_json(summary_path, summary)
    return summary
