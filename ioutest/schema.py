"""The public model interface. Ground truth and depth are deliberately absent."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


CLASSES = ("cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer",
           "toilet", "bathtub", "oven", "dishwasher", "fireplace", "stool",
           "chair", "table", "tv_monitor", "sofa")
PROMPTS = tuple({"washer": "washing machine", "tv_monitor": "television"}.get(c, c)
                for c in CLASSES)
SCHEMA_VERSION = "ioutest.box3d.v1"


@dataclass(frozen=True)
class ModelInput:
    frame_id: str
    rgb: np.ndarray                   # Upright RGB uint8 H,W,3
    intrinsics: np.ndarray            # 3,3, for rgb; no padding or resizing yet
    class_names: tuple = CLASSES
    prompts: tuple = PROMPTS


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    center: np.ndarray                # Meters, upright OpenCV camera
    size: np.ndarray                  # Full lengths along LOCAL x,y,z
    rotation: np.ndarray              # LOCAL box -> camera, column vectors

    def to_dict(self):
        if isinstance(self.class_id, (bool, np.bool_)) or not isinstance(self.class_id, (int, np.integer)):
            raise ValueError("适配器必须返回整数 class_id，不能截断浮点数")
        value = {"class_id": int(self.class_id), "score": float(self.score),
                 "center": np.asarray(self.center).tolist(),
                 "size": np.asarray(self.size).tolist(),
                 "rotation": np.asarray(self.rotation).tolist()}
        validate_box(value, prediction=True)
        return value


class ModelAdapter(Protocol):
    def load(self) -> dict:
        """Load once; return model/weight/preprocessing provenance."""

    def predict(self, sample: ModelInput) -> list[Detection]:
        """Return detections in the canonical camera frame."""

    def close(self) -> None:
        """Release resources."""


def validate_box(box, prediction=False):
    cid = box.get("class_id")
    if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < len(CLASSES):
        raise ValueError(f"非法 class_id: {cid!r}")
    for name, shape in (("center", (3,)), ("size", (3,)), ("rotation", (3, 3))):
        value = np.asarray(box[name], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} 必须为有限数值，形状 {shape}")
    if np.any(np.asarray(box["size"]) <= 0):
        raise ValueError("三维框完整尺寸必须为正数")
    rotation = np.asarray(box["rotation"])
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4) or not np.isclose(
            np.linalg.det(rotation), 1, atol=2e-4):
        raise ValueError("rotation 必须为正交、右手旋转矩阵")
    if prediction and not np.isfinite(float(box["score"])):
        raise ValueError("预测 score 必须为有限数值")


def class_id(label):
    normalized = str(label).replace("-", "_").replace("/", "_").replace(" ", "_")
    if normalized not in CLASSES:
        raise ValueError(f"未识别的 ARKitScenes 类别: {label!r}")
    return CLASSES.index(normalized)
