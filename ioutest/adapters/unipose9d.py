"""Single-instance UniPose9D using RGB, predicted/sensor depth and a predicted mask."""

import sys
from pathlib import Path

import numpy as np

from ..common import file_digest
from ..schema import Detection
from ..suite_io import project_source


class ObjectFailure(Exception):
    """Expected, counted object-estimation failure; device/model errors remain fatal."""


class UniPoseAdapter:
    def __init__(self, options):
        self.options = dict(options)

    def load(self):
        import torch
        repo = Path(self.options["project_root"])
        sys.path.insert(0, str(repo / "infer"))
        from unipose9d_inference import estimate_pose, load_pose_model, set_seed
        self.torch, self.estimate_pose, self.set_seed = torch, estimate_pose, set_seed
        self.device = torch.device(self.options.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前进程不能访问 PPU，UniPose9D 不回退 CPU")
        checkpoint, cfg = Path(self.options["checkpoint"]), Path(self.options["config"])
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            self.model, config = load_pose_model(checkpoint, cfg, self.device)
        finally:
            torch.set_num_threads(threads)
        defaults = dict(num_points=1024, num_tuples=1000, num_steps=32, num_repeats=10,
                        ransac_hops=2, ransac_iterations=1000, scale_clustering=True)
        self.parameters = {k: self.options.get(k, v) for k, v in defaults.items()}
        if any(int(v) < 1 for k, v in self.parameters.items() if k != "scale_clustering"):
            raise ValueError("UniPose9D 采样与迭代预算必须为正数")
        self.parameters["image_size"] = int(config.get("image_size", 448))
        return {"model": "UniPose9D", "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_digest(checkpoint), "config_sha256": file_digest(cfg),
                "project_source_sha256": project_source(repo, "infer"),
                "third_party_source_sha256": project_source(repo, "third_party"),
                "parameters": self.parameters, "depth": "sensor" if self.options.get("depth_model") else "MoGe",
                "device": str(self.device), "torch": torch.__version__}

    def predict_object(self, sample, seed):
        if sample.mask is None or sample.depth is None:
            raise ValueError("UniPose9D 需要预测掩码和明确来源的深度")
        self.set_seed(seed)
        try:
            result = self.estimate_pose(sample.rgb, sample.depth, sample.mask, sample.intrinsics,
                                        self.model, self.device, **self.parameters)
        except ValueError as exc:
            if any(s in str(exc) for s in ("Mask is empty", "No valid masked depth", "Point cloud too small", "Not enough points")):
                raise ObjectFailure(str(exc)) from exc
            raise
        except RuntimeError as exc:
            if "RANSAC failed to produce an inlier mask" in str(exc):
                raise ObjectFailure(str(exc)) from exc
            raise
        quality = {k: result[k] for k in ("inlier_count", "inlier_ratio", "mse")}
        if not all(np.isfinite(v) for v in quality.values()):
            raise ObjectFailure("nonfinite_pose_quality")
        score = float(result["inlier_ratio"])
        try:
            box = Detection(None, score, result["translation"], result["predicted_scale"],
                            result["rotation"]).to_dict()
        except ValueError as exc:
            raise ObjectFailure("invalid_pose_geometry: " + str(exc)) from exc
        return box, quality

    def close(self):
        self.model = None
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
