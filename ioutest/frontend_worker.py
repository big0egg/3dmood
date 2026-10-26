"""Private resident RGB frontend, called through local RPC; no dataset/GT access."""
import importlib.metadata
import sys
import traceback
from pathlib import Path
import numpy as np
from .common import digest, file_digest
from .hot3d.common import prompt_text
from .suite_frontend import category_tokens, grounding_probabilities
from .timing import timed
from .worker_rpc import PipeConnection, receive, send

class Frontend:
    def __init__(self, cfg, track, classes, execution):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch; self.cfg = cfg; self.track = track; self.device = execution['device']
        torch.set_num_threads(execution['threads']); torch.manual_seed(execution['seed'])
        if str(self.device).startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('前端不可见 PPU，不回退 CPU')
        self.predictor = SAM2ImagePredictor(build_sam2(cfg['sam2_config'], cfg['sam2_checkpoint'], device=self.device))
        sys.path.insert(0, str(Path(cfg['project_root']) / 'infer'))
        from moge_model import load_moge_model, predict_depth_intrinsics
        self.predict_depth = predict_depth_intrinsics
        self.moge = None if cfg.get('sensor_depth') else load_moge_model(cfg['moge_checkpoint'], torch.device(self.device))
        self.identity = dict(sam2_sha256=file_digest(cfg['sam2_checkpoint']),
            moge_sha256=None if self.moge is None else file_digest(cfg['moge_checkpoint']), torch=torch.__version__)
        sam_root = Path(sys.modules['sam2'].__file__).parent
        self.identity['sam2_source'] = digest({str(p.relative_to(sam_root)): file_digest(p) for p in sorted(sam_root.rglob('*.py'))})
        self.identity['versions'] = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy')}
        self.grounding = None
        if track == 'text':
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(cfg['grounding_model'], local_files_only=True)
            self.grounding = AutoModelForZeroShotObjectDetection.from_pretrained(cfg['grounding_model'],
                local_files_only=True, disable_custom_kernels=True).to(self.device).eval()
            self.text, spans = prompt_text(classes)
            self.encoded = self.processor.tokenizer(self.text, return_offsets_mapping=True)
            self.tokens = category_tokens(self.encoded['input_ids'], self.encoded['offset_mapping'], self.text, spans)
            self.identity['grounding_sha256'] = digest({str(p.relative_to(cfg['grounding_model'])): file_digest(p)
                for p in sorted(Path(cfg['grounding_model']).rglob('*')) if p.is_file() and '.cache' not in p.parts})

    def predict(self, rgb, intrinsics, prompts, depth=None):
        torch = self.torch; h, w = rgb.shape[:2]; depth_seconds = 0.
        if depth is None:
            result, depth_seconds = timed(torch, self.device, lambda: self.predict_depth(self.moge, rgb))
            depth = np.asarray(result[0], dtype=np.float32)
            depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
            if depth.shape != (h, w) or not (depth > 0).any():
                raise RuntimeError('MoGe 深度非法，不能当作无检出')
        def frontend():
            proposals = [dict(p) for p in prompts] if self.track == 'gt-box' else self.detect(rgb)
            if not proposals: return []
            self.predictor.set_image(rgb); items = []
            for proposal in proposals:
                rect = np.asarray(proposal['box_xyxy'], dtype=np.float32)
                if rect.shape != (4,) or not np.isfinite(rect).all() or rect[2] <= rect[0] or rect[3] <= rect[1]:
                    items.append(dict(proposal, status='failed', failure='invalid_2d_box')); continue
                masks, scores, _ = self.predictor.predict(box=rect[None], multimask_output=False)
                mask = np.asarray(masks).reshape(-1, h, w)[0].astype(bool)
                if not mask.any(): items.append(dict(proposal, status='failed', failure='empty_predicted_mask'))
                else: items.append(dict(proposal, status='ok', mask=mask))
            return items
        items, frontend_seconds = timed(torch, self.device, frontend)
        return dict(items=items, depth=depth, timing=dict(frontend_seconds=frontend_seconds, depth_seconds=depth_seconds))

    def detect(self, rgb):
        from PIL import Image
        torch = self.torch; cfg = self.cfg; h, w = rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(rgb), text=self.text, return_tensors='pt').to(self.device)
        if inputs['input_ids'][0].tolist() != self.encoded['input_ids']: raise ValueError('固定类别提示截断或改变')
        with torch.inference_mode(): output = self.grounding(**inputs)
        probabilities = grounding_probabilities(output, inputs['attention_mask'])[0]
        per_class = torch.stack([probabilities[:, ids].max(-1).values for ids in self.tokens], dim=-1)
        class_scores, classes = per_class.max(-1); scores = probabilities.max(-1).values
        selected = torch.where((scores > cfg['box_threshold']) & (class_scores > cfg['text_threshold']))[0]
        selected = selected[scores[selected].argsort(descending=True)[:cfg['max_detections']]]
        boxes = output.pred_boxes[0, selected].detach().cpu().numpy(); proposals = []
        for query, (cx, cy, bw, bh) in zip(selected.tolist(), boxes):
            rect = [max(0., float((cx-bw/2)*w)), max(0., float((cy-bh/2)*h)),
                    min(w-1., float((cx+bw/2)*w)), min(h-1., float((cy+bh/2)*h))]
            proposals.append(dict(target_id=f'q{query}', box_xyxy=rect, class_id=int(classes[query]), detection_score=float(scores[query])))
        return proposals

def serve(read_fd, write_fd):
    sock = PipeConnection(read_fd, write_fd)
    try:
        init = receive(sock); frontend = Frontend(init['config'], init['track'], init['classes'], init['execution'])
        send(sock, dict(ok=True, identity=frontend.identity))
        while True:
            request = receive(sock)
            if request.get('close'): return
            send(sock, dict(ok=True, result=frontend.predict(**request)))
    except BaseException as exc:
        traceback.print_exc()
        try: send(sock, dict(ok=False, error=f'{type(exc).__name__}: {exc}'))
        except (OSError, RuntimeError): pass
        raise
    finally: sock.close()

if __name__ == '__main__': serve(int(sys.argv[1]), int(sys.argv[2]))
