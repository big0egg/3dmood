"""The first model adapter; all 3D-MOOD dependencies are isolated here."""

import importlib.util
import os
from pathlib import Path

import numpy as np

from ..common import digest, file_digest
from ..schema import Detection


DEFAULT_CHECKPOINT = Path("/mnt/workspace/.cache/3d-mood/torch/hub/checkpoints/gdino3d_swin-b_120e_omni3d_834c97.pt")


class Mood3DAdapter:
    def __init__(self, options):
        self.options = dict(options)
        self.model = None

    def load(self):
        # Set offline mode before importing transformers through the model.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("MPLBACKEND", "Agg")
        import torch
        from vis4d.common.ckpt import load_model_checkpoint
        from vis4d.data.transforms.base import compose
        from vis4d.data.transforms.normalize import NormalizeImages
        from vis4d.data.transforms.resize import ResizeImages, ResizeIntrinsics
        from vis4d.data.transforms.to_tensor import ToTensor
        from vis4d.op.geometry.rotation import quaternion_to_matrix
        from opendet3d.data.transforms.pad import CenterPadImages, CenterPadIntrinsics
        from opendet3d.data.transforms.resize import GenResizeParameters

        self.torch = torch
        self.device = str(self.options.get("device", "cuda"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("当前进程无法访问 PPU，请在 DSW 原生 Terminal 的项目环境中执行 infer；不会自动改用 CPU")
        self.checkpoint = Path(self.options.get("checkpoint", DEFAULT_CHECKPOINT)).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"本地检测权重不存在：{self.checkpoint}")
        self.max_detections = int(self.options.get("max_detections", 300))
        self.score_threshold = float(self.options.get("score_threshold", 0.))
        if not 1 <= self.max_detections <= 900 or not 0 <= self.score_threshold < 1:
            raise ValueError("max_detections 应在 1～900，score_threshold 应在 [0,1)")
        repo = Path(self.options.get("project_root", Path(__file__).resolve().parents[2])).resolve()
        demo = repo / "scripts/demo.py"
        spec = importlib.util.spec_from_file_location("_ioutest_3dmood_demo", demo)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.get_3d_mood_swin_base(self.max_detections, self.score_threshold).to(self.device)
        load_model_checkpoint(self.model, weights=str(self.checkpoint),
                              rev_keys=[(r"^model\.", ""), (r"^module\.", "")])
        self.model.eval()
        self.preprocess = compose(transforms=[GenResizeParameters(shape=(800, 1333)), ResizeImages(),
            ResizeIntrinsics(), NormalizeImages(),
            CenterPadImages(stride=1, shape=(800, 1333), update_input_hw=True), CenterPadIntrinsics()])
        self.to_tensor = ToTensor()
        self.quaternion_to_matrix = quaternion_to_matrix
        return {"model": "3D-MOOD Swin-B", "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": file_digest(self.checkpoint), "demo_sha256": file_digest(demo),
                "project_source_sha256": digest({str(p.relative_to(repo)): file_digest(p)
                    for p in sorted((repo / "opendet3d").rglob("*.py"))}),
                "torch_version": torch.__version__, "device": self.device,
                "input_modalities": ["rgb", "intrinsics", "fixed_class_prompts"],
                "preprocessing": "upright frame -> existing Demo resize/normalize/center-pad (800,1333)",
                "max_candidates": self.max_detections, "score_threshold": self.score_threshold,
                "nms": "2D class-agnostic, IoU 0.5, same as Demo",
                "hf_home": os.environ.get("HF_HOME", "default Hugging Face cache")}

    def predict(self, sample):
        torch = self.torch
        image = sample.rgb.astype(np.float32)[None]
        hw = sample.rgb.shape[:2]
        values = {"images": image, "original_images": image, "input_hw": hw, "original_hw": hw,
                  "intrinsics": sample.intrinsics.astype(np.float32),
                  "original_intrinsics": sample.intrinsics.astype(np.float32)}
        data = self.to_tensor(self.preprocess([values]))[0]
        with torch.inference_mode():
            output = self.model(images=data["images"].to(self.device), input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]], intrinsics=data["intrinsics"].to(self.device)[None],
                padding=[data["padding"]], input_texts=[list(sample.prompts)])
        _, boxes3d, scores, labels, _, _ = output
        boxes = boxes3d[0].detach().to(dtype=torch.float64, device="cpu")
        rotations = self.quaternion_to_matrix(boxes[:, 6:10]).numpy()
        scores = scores[0].detach().cpu().numpy()
        labels = labels[0].detach().cpu().numpy()
        detections = []
        for box, rotation, score, label in zip(boxes.numpy(), rotations, scores, labels):
            # vis4d OPENCV corners use local x=l, y=h, z=w. Its stored size is w,l,h.
            detections.append(Detection(int(label), float(score), box[:3], box[[4, 5, 3]], rotation))
        return detections

    def close(self):
        self.model = None
        if hasattr(self, "torch") and self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
