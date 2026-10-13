"""Immutable suite records and prompt-only inputs; no detector imports."""

import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .benchmark_worker import write_prediction
from .common import digest, file_digest, read_json, write_json
from .image_cache import _write_bytes
from .schema import SCHEMA_VERSION, validate_box


GT_MODELS = ("wilddet3d-box", "unipose9d-box-rgb", "unipose9d-box-rgbd")
TEXT_MODELS = ("3dmood", "wilddet3d", "unipose9d-text-rgb")


@dataclass(frozen=True)
class ObjectInput:
    rgb: np.ndarray
    intrinsics: np.ndarray
    box_xyxy: tuple
    mask: np.ndarray | None = None
    depth: np.ndarray | None = None


def frame_path(root, frame, suffix=".json.gz"):
    scan, stem = frame["video_id"], Path(frame["rgb_entry"]).stem
    if Path(scan).name != scan or not stem:
        raise ValueError("非法帧路径")
    return Path(root) / scan / (stem + suffix)


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("缓存路径越界")
    return path


def read_record(path, frame_id, signature):
    with gzip.open(path, "rt") as f:
        record = json.load(f)
    if (record.get("frame_id") != frame_id or record.get("signature") != signature or
            record.get("status") != "ok" or record.get("schema_version") != SCHEMA_VERSION):
        raise ValueError(f"缓存标识或状态不一致：{path}")
    for item in record.get("items", []):
        if item.get("detection") is not None:
            validate_box(item["detection"], prediction=True, allow_unlabeled=True)
    return record


def commit_record(path, frame, signature, **values):
    record = dict(schema_version=SCHEMA_VERSION, status="ok", frame_id=frame["frame_id"],
                  signature=signature, **values)
    write_prediction(path, record)
    return record


def save_array(path, array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    _write_bytes(Path(path), stream.getvalue())


def load_array(root, record):
    path = safe_child(root, record["path"])
    if file_digest(path) != record["sha256"]:
        raise ValueError("深度缓存内容变化")
    array = np.load(path, allow_pickle=False)
    if array.shape != tuple(record["shape"]) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("深度缓存尺寸或数值不合法")
    return array


def object_seed(base, frame_id, target_id):
    return int(digest([int(base), frame_id, target_id])[:8], 16)


def bind_identity(path, identity):
    signature = digest(identity)
    value = {"signature": signature, "provenance": identity}
    if Path(path).exists() and read_json(path) != value:
        raise ValueError(f"配置、代码或输入变化，不能混用现有缓存：{path}")
    write_json(path, value)
    return signature


def project_source(repo, folder):
    repo = Path(repo)
    return digest({str(p.relative_to(repo)): file_digest(p) for p in sorted((repo / folder).rglob("*.py"))})


def mask_checksum(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def require_mask(root, item, shape):
    from PIL import Image
    with Image.open(safe_child(root, item["mask_path"])) as f:
        mask = np.asarray(f) > 0
    if mask.shape != tuple(shape) or mask_checksum(mask) != item["mask_sha256"]:
        raise ValueError("掩码缓存尺寸或像素校验不一致")
    return mask
