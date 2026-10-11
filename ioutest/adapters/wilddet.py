"""WildDet3D numerical predictions; run in its own PPU environment."""

import os
import sys
from pathlib import Path

import numpy as np

from ..common import digest, file_digest
from ..schema import Detection


class WildDetAdapter:
    def __init__(self, options):
        self.options = dict(options)
        self.model = None

    def load(self):
        os.environ.update(SAM3_COMPILE="0", XFORMERS_DISABLED="1", HF_HUB_OFFLINE="1")
        repo = Path(self.options.get("project_root", "/mnt/workspace/wilddet3d")).resolve()
        sys.path.insert(0, str(repo))
        import torch
        from wilddet3d import build_model, preprocess
        from vis4d.op.geometry.rotation import quaternion_to_matrix
        self.torch = torch
        self.device = str(self.options.get("device", "cuda"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("WildDet3D 需要在能访问 PPU 的 DSW 原生 Terminal 运行")
        self.checkpoint = Path(self.options.get("checkpoint",
            "/mnt/workspace/TEMP-FILE-STATION/wilddet3d_alldata_all_prompt_v1.0.pt"))
        lingbot = Path(self.options.get("lingbot_checkpoint", "/mnt/workspace/TEMP-FILE-STATION/model.pt"))
        for path in (self.checkpoint, lingbot):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.max_detections = int(self.options.get("max_detections", 300))
        self.model = build_model(checkpoint=str(self.checkpoint), lingbot_pretrained_model=str(lingbot),
            skip_pretrained=True, canonical_rotation=True, score_threshold=0., score_3d_threshold=0.,
            device=self.device, use_depth_input_test=False, use_predicted_intrinsics=False)
        self.preprocess, self.quaternion_to_matrix = preprocess, quaternion_to_matrix
        return {"model": "WildDet3D stage 3", "checkpoint_sha256": file_digest(self.checkpoint),
            "lingbot_sha256": file_digest(lingbot), "device": self.device, "torch_version": torch.__version__,
            "project_source_sha256": digest({str(p.relative_to(repo)): file_digest(p)
                for p in sorted((repo / "wilddet3d").rglob("*.py"))}),
            "input_modalities": ["rgb", "intrinsics", "fixed_17_class_prompts"],
            "score": "native combined 2D/3D confidence", "score_threshold": 0., "score_3d_threshold": 0.,
            "max_candidates": self.max_detections, "nms": "native 2D, IoU 0.6", "compile": False}

    def predict(self, sample):
        if not self.device.startswith("cuda"):
            raise RuntimeError("WildDet3D 前向使用 CUDA Stream；CPU 仅用于权重构建检查")
        data = self.preprocess(sample.rgb, sample.intrinsics)
        with self.torch.inference_mode():
            output = self.model(images=data["images"].to(self.device),
                intrinsics=data["intrinsics"].to(self.device)[None], input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]], padding=[data["padding"]], input_texts=list(sample.prompts))
        boxes, scores, labels = output[1][0], output[2][0], output[5][0]
        order = scores.argsort(descending=True)[:self.max_detections]
        rotations = self.quaternion_to_matrix(boxes[order, 6:10]).detach().cpu().numpy()
        values = boxes[order].detach().cpu().numpy()
        return [Detection(int(label), float(score), box[:3], box[[4, 5, 3]], rotation)
            for box, rotation, score, label in zip(values, rotations,
                scores[order].detach().cpu().numpy(), labels[order].detach().cpu().numpy())]

    def close(self):
        self.model = None
