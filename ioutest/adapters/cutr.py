"""CuTR RGB/RGB-D adapter; no GT instances or laser-scanner depth are fed."""

import sys
from pathlib import Path

import numpy as np

from ..common import digest, file_digest
from ..schema import Detection


class CuTRAdapter:
    def __init__(self, options):
        self.options = dict(options)
        self.model = None

    def load(self):
        repo = Path(self.options.get("project_root", "/mnt/workspace/CA1M")).resolve()
        sys.path.insert(0, str(repo))
        import torch
        from cubifyanything.cubify_transformer import make_cubify_transformer
        from cubifyanything.preprocessor import Augmentor, Preprocessor
        self.torch = torch
        self.device = str(self.options.get("device", "cuda"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CuTR 需要在能访问 PPU 的 DSW 原生 Terminal 运行")
        self.depth_model = bool(self.options.get("depth_model", False))
        self.use_gravity = bool(self.options.get("use_gravity", False))
        self.input_long_edge = int(self.options.get("input_long_edge", 1024))
        if not 32 <= self.input_long_edge <= 1024:
            raise ValueError("CuTR input_long_edge 必须在 32～1024")
        name = "cutr_rgbd.pth" if self.depth_model else "cutr_rgb.pth"
        checkpoint = Path(self.options.get("checkpoint", repo / "models" / name)).resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        depth_weights = any(k.startswith("backbone.0.patch_embed_depth.") for k in state)
        if depth_weights != self.depth_model:
            raise ValueError("CuTR 权重与 RGB/RGB-D 配置不一致")
        dimension = state["backbone.0.patch_embed.proj.weight"].shape[0]
        # The PPU image's CPU trunc_normal_/erfinv initialization can crash with
        # multiple host threads. All randomly initialized parameters are then
        # replaced by the strictly checked checkpoint; serialize only construction.
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            self.model = make_cubify_transformer(dimension=dimension, depth_model=self.depth_model)
        finally:
            torch.set_num_threads(previous_threads)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        # Native inference ranks both channels before top-k; channel 1 is not
        # supervised foreground and can otherwise consume all 100 slots.
        # Select foreground BEFORE native top-k without changing network weights.
        for prompter in self.model.prompting.prompters:
            if hasattr(prompter, "inference_single_image"):
                native = prompter.inference_single_image

                def foreground_output(output, image_size, topk, native=native):
                    output = output.clone()
                    output.pred_logits = output.pred_logits.clone()
                    output.pred_logits[..., 1:] = -torch.inf
                    return native(output, image_size, min(topk, len(output)))

                prompter.inference_single_image = foreground_output
        self.augmentor = Augmentor(("wide/image", "wide/depth") if self.depth_model else ("wide/image",))
        self.preprocessor = Preprocessor()
        return {"model": "CuTR RGB-D" if self.depth_model else "CuTR RGB", "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_digest(checkpoint), "device": self.device, "torch_version": torch.__version__,
            "project_source_sha256": digest({str(p.relative_to(repo)): file_digest(p)
                for p in sorted((repo / "cubifyanything").rglob("*.py"))}),
            "input_modalities": ["rgb", "intrinsics"] + (["ARKit_LiDAR_depth_meters"] if self.depth_model else []) +
                                (["trajectory_derived_gravity"] if self.use_gravity else []),
            "score": "native foreground objectness (channel 0)", "semantic_labels": False,
            "native_topk": self.model.topk_per_image, "nms": "none (native DETR)", "score_threshold": 0.,
            "candidate_selection": "channel 0 foreground only BEFORE native top-k; no GT-derived labels",
            "initialization_cpu_threads": 1,
            "preprocessing": {"long_edge": self.input_long_edge, "rgb_resize": "bilinear align_corners=False",
                              "depth_resize": "nearest", "intrinsics": "pixel-center scaling then native normalization/padding"},
            "gravity": "known sensor gravity" if self.use_gravity else "not supplied; camera-upright assumption"}

    def predict(self, sample):
        from cubifyanything.measurement import ImageMeasurementInfo, DepthMeasurementInfo
        from cubifyanything.sensor import PosedSensorInfo, SensorArrayInfo
        torch = self.torch
        h, w = sample.rgb.shape[:2]
        image = torch.from_numpy(sample.rgb.copy()).permute(2, 0, 1)[None].float()
        intrinsic = np.asarray(sample.intrinsics, dtype=np.float32).copy()
        target_h = max(1, round(h * self.input_long_edge / max(h, w)))
        target_w = max(1, round(w * self.input_long_edge / max(h, w)))
        scale_x, scale_y = target_w / w, target_h / h
        intrinsic[0] *= scale_x; intrinsic[1] *= scale_y
        intrinsic[0, 2] += (scale_x - 1) / 2
        intrinsic[1, 2] += (scale_y - 1) / 2
        image = torch.nn.functional.interpolate(image, (target_h, target_w), mode="bilinear", align_corners=False)
        camera = PosedSensorInfo()
        camera.RT = torch.eye(4)[None]
        camera.image = ImageMeasurementInfo((target_w, target_h), torch.tensor(intrinsic)[None])
        values = {"image": image}
        if self.depth_model:
            depth = getattr(sample, "depth", None)
            if depth is None or depth.shape != (h, w) or not np.isfinite(depth).all() or (depth < 0).any():
                raise ValueError("CuTR RGB-D 必须提供与 RGB 对齐的非负米制 ARKit 深度")
            camera.depth = DepthMeasurementInfo((target_w, target_h), camera.image.K.clone())
            values["depth"] = torch.nn.functional.interpolate(torch.tensor(depth, dtype=torch.float32)[None, None],
                                    (target_h, target_w), mode="nearest")[:, 0]
        if self.use_gravity:
            gravity = getattr(sample, "gravity", None)
            if gravity is None:
                raise ValueError("已知重力实验缺少传感器重力旋转")
            camera.T_gravity = torch.tensor(gravity, dtype=torch.float32)[None]
        sensors = SensorArrayInfo()
        sensors.wide = camera
        packaged = self.augmentor.package({"sensor_info": sensors, "wide": values})
        packaged = {s: {m: value.to(self.device) for m, value in measures.items()}
                    for s, measures in packaged.items()}
        with torch.inference_mode():
            result = self.model(self.preprocessor.preprocess([packaged]))[0]
        # Channel 1 is not a semantic category. Never infer labels from GT.
        result = result[result.pred_classes == 0]
        boxes = result.pred_boxes_3d
        centers, sizes, rotations = (boxes.gravity_center.detach().cpu().numpy(),
                                   boxes.dims.detach().cpu().numpy(), boxes.R.detach().cpu().numpy())
        return [Detection(None, float(score), center, size, rotation)
                for center, size, rotation, score in zip(centers, sizes, rotations,
                                                         result.scores.detach().cpu().numpy())]

    def close(self):
        self.model = None
