"""WildDet3D native one-object geometric prompting, without target class labels."""

import numpy as np

from .wilddet import WildDetAdapter
from .unipose9d import ObjectFailure
from ..schema import Detection


class WildBoxAdapter(WildDetAdapter):
    def load(self):
        info = super().load()
        info.update(input_modalities=["rgb", "intrinsics", "supplied_2d_box"],
                    prompting="geometric, one prediction per prompt; no semantic label")
        return info

    def predict_object(self, sample, seed):
        # One prompt per call avoids the native wrapper's per-image coordinate
        # conversion indexing multiple boxes as if they were different images.
        data = self.preprocess(sample.rgb, sample.intrinsics)
        with self.torch.inference_mode():
            out = self.model(images=data["images"].to(self.device),
                intrinsics=data["intrinsics"].to(self.device)[None], input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]], padding=[data["padding"]],
                input_boxes=[list(sample.box_xyxy)], prompt_text="geometric")
        if len(out[1][0]) != 1:
            raise ObjectFailure("native_geometric_prompt_has_no_single_prediction")
        raw = out[1][0][0].detach().cpu()
        rotation = self.quaternion_to_matrix(raw[None, 6:10]).numpy()[0]
        score = float(out[2][0][0].detach().cpu())
        box = raw.numpy()
        try:
            detection = Detection(None, score, box[:3], box[[4, 5, 3]], rotation).to_dict()
        except ValueError as exc:
            raise ObjectFailure("invalid_pose_geometry: " + str(exc)) from exc
        return detection, {"native_score": score}
