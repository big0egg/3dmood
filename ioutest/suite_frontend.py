"""SAM2/GroundingDINO inputs and shared MoGe cache, with no 3D GT access."""

import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .common import digest, file_digest, read_json, write_json
from .image_cache import _write_bytes
from .schema import PROMPTS
from .suite_frontend_identity import bind_frontend_identity
from .suite_io import (bind_identity, commit_record, frame_path, load_array,
                       mask_checksum, read_record, require_mask, save_array)


def category_tokens(token_ids, offsets, prompt, spans):
    """Map fixed prompt spans to tokenizer positions without looking at GT."""
    if len(token_ids) != len(offsets):
        raise ValueError("文本 token 与 offset 数量不同")
    groups = [[i for i, (start, end) in enumerate(offsets)
               if end > start and start < hi and end > lo] for lo, hi in spans]
    if any(not indices for indices in groups):
        raise ValueError("类别提示无法映射到 tokenizer")
    return groups


def fixed_prompt():
    text, spans = "", []
    for word in PROMPTS:
        start = len(text)
        text += word
        spans.append((start, len(text)))
        text += " . "
    return text.strip(), spans


def grounding_probabilities(output, attention_mask):
    """Allow native -inf text padding, but reject invalid real-token scores."""
    import torch
    logits = output.logits
    if (logits.ndim != 3 or attention_mask.ndim != 2 or
            logits.shape[0] != attention_mask.shape[0] or logits.shape[-1] < attention_mask.shape[-1]):
        raise RuntimeError('GroundingDINO 输出与文本 attention_mask 尺寸不一致')
    active = torch.zeros((logits.shape[0], logits.shape[-1]), dtype=torch.bool, device=logits.device)
    active[:, :attention_mask.shape[-1]] = attention_mask.to(device=logits.device, dtype=torch.bool)
    active = active[:, None, :].expand_as(logits)
    if (torch.isnan(logits).any() or torch.isposinf(logits).any() or
            not torch.isfinite(logits[active]).all() or not torch.isfinite(output.pred_boxes).all()):
        raise RuntimeError('GroundingDINO 有效文本分数或预测框输出非有限数值，不能当作无检出计分')
    # Native negative-infinity padding maps to probability zero under sigmoid.
    return logits.sigmoid()


def work(task):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4)
    cfg, track = task["config"]["frontend"], task["track"]
    device = torch.device(task["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("前处理进程不能访问 PPU，不回退 CPU")
    root = Path(task["report_root"])
    aux = root / "aux" / track
    frame_rows = task["frames"]  # Only RGB/K metadata and allowed 2D prompts.
    sys.path.insert(0, str(Path(cfg["project_root"]) / "infer"))
    from moge_model import load_moge_model, predict_depth_intrinsics
    predictor = SAM2ImagePredictor(build_sam2(cfg["sam2_config"], cfg["sam2_checkpoint"], device=str(device)))
    identity = {"benchmark_id": task["benchmark_id"], "track": track, "config": cfg,
                "input_manifest": task["input_manifest_sha256"],
                "sam2_sha256": file_digest(cfg["sam2_checkpoint"]),
                "worker_sha256": file_digest(__file__), "io_sha256": file_digest(Path(__file__).with_name('suite_io.py')),
                "device": str(device), "torch": torch.__version__,
                "sam2_source": digest({str(p.relative_to(Path(sys.modules['sam2'].__file__).parent)): file_digest(p)
                    for p in sorted(Path(sys.modules['sam2'].__file__).parent.rglob('*.py'))})}
    grounding = processor = token_groups = None
    if track == "text":
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        processor = AutoProcessor.from_pretrained(cfg["grounding_model"], local_files_only=True)
        grounding = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg["grounding_model"], local_files_only=True, disable_custom_kernels=True).to(device).eval()
        text, spans = fixed_prompt()
        encoded = processor.tokenizer(text, return_offsets_mapping=True)
        token_groups = category_tokens(encoded['input_ids'], encoded['offset_mapping'], text, spans)
        identity['grounding_sha256'] = digest({str(p.relative_to(cfg['grounding_model'])): file_digest(p)
            for p in sorted(Path(cfg['grounding_model']).rglob('*')) if p.is_file() and '.cache' not in p.parts})
        identity['prompt'] = text
        identity['grounding_disable_custom_kernels'] = True
        identity['grounding_logits_policy'] = 'finite-active-tokens; allow-native-negative-infinity-padding'
    signature = bind_frontend_identity(aux / "model.json", identity, track)
    depth_root = root / "aux" / "predicted_depth"
    depth_identity = {"benchmark_id": task["benchmark_id"], "moge_sha256": file_digest(cfg["moge_checkpoint"]),
                      "source_sha256": file_digest(Path(cfg["project_root"]) / "infer/moge_model.py"),
                      "worker_sha256": file_digest(__file__),
                      "io_sha256": file_digest(Path(__file__).with_name('suite_io.py')),
                      "device": str(device), "dtype_policy": "native MoGe fp16 on accelerator",
                      "rgb_manifest": task["rgb_manifest_sha256"]}
    depth_signature = bind_frontend_identity(depth_root / "model.json", depth_identity, 'depth')
    moge = None
    done = reused = 0
    started = time.perf_counter()
    for frame in frame_rows:
        path = frame_path(aux / "frames", frame)
        if path.exists():
            old = read_record(path, frame['frame_id'], signature)
            for item in old['items']:
                if item['status'] == 'ok':
                    require_mask(aux, item, frame['image_hw'])
            depth_record = read_record(frame_path(depth_root, frame), frame['frame_id'], depth_signature)
            load_array(depth_root, depth_record['array'])
            reused += 1
        else:
            with Image.open(frame['rgb_path']) as image:
                rgb = np.asarray(image.convert('RGB')).copy()
            if list(rgb.shape[:2]) != frame['image_hw'] or mask_checksum(rgb) != frame['upright_rgb_sha256']:
                raise ValueError("前处理 RGB 与冻结输入不一致")
            depth_record_path = frame_path(depth_root, frame)
            if depth_record_path.exists():
                depth_record = read_record(depth_record_path, frame['frame_id'], depth_signature)
                load_array(depth_root, depth_record['array'])
                depth_seconds = depth_record['seconds']
            else:
                if moge is None:
                    moge = load_moge_model(cfg['moge_checkpoint'], device)
                if device.type == 'cuda': torch.cuda.synchronize()
                start = time.perf_counter()
                depth, predicted_k = predict_depth_intrinsics(moge, rgb)
                if device.type == 'cuda': torch.cuda.synchronize()
                depth_seconds = time.perf_counter() - start
                depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
                if depth.shape != tuple(frame['image_hw']):
                    raise ValueError("MoGe 预测深度尺寸不一致")
                array_path = frame_path(depth_root / 'arrays', frame, '.npy')
                save_array(array_path, depth)
                commit_record(depth_record_path, frame, depth_signature, seconds=depth_seconds,
                              predicted_intrinsics=predicted_k.tolist(),
                              array=dict(path=str(array_path.relative_to(depth_root)),
                                         sha256=file_digest(array_path), shape=list(depth.shape)))
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            if track == 'gt-box':
                proposals = [dict(p) for p in frame['prompts']]
            else:
                inputs = processor(images=Image.fromarray(rgb), text=text, return_tensors='pt').to(device)
                if inputs['input_ids'][0].tolist() != encoded['input_ids']:
                    raise ValueError("GroundingDINO processor 改变了固定提示的 token 顺序")
                with torch.inference_mode(): output = grounding(**inputs)
                probabilities = grounding_probabilities(output, inputs['attention_mask'])[0]
                per_class = torch.stack([probabilities[:, indices].max(-1).values for indices in token_groups], dim=-1)
                class_scores, class_ids = per_class.max(-1)
                native_scores = probabilities.max(-1).values
                # Preserve native box/text floors, then retain every instance,
                # rather than the single highest-scoring instance in the Demo.
                selected = torch.where((native_scores > cfg['box_threshold']) &
                                       (class_scores > cfg['text_threshold']))[0]
                selected = selected[native_scores[selected].argsort(descending=True)[:cfg['max_detections']]]
                boxes = output.pred_boxes[0, selected].detach().cpu().numpy()
                h, w = rgb.shape[:2]
                proposals = []
                for query, box in zip(selected.tolist(), boxes):
                    cx, cy, bw, bh = box
                    xyxy = [max(0., (cx - bw/2)*w), max(0., (cy - bh/2)*h),
                            min(w-1., (cx+bw/2)*w), min(h-1., (cy+bh/2)*h)]
                    proposals.append(dict(target_id=f'q{query}', box_xyxy=list(map(float, xyxy)),
                        class_id=int(class_ids[query]), detection_score=float(native_scores[query])))
            detection_seconds = time.perf_counter() - start
            predictor.set_image(rgb)
            items = []
            mask_write_seconds = 0.
            for proposal in proposals:
                box = proposal['box_xyxy']
                if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                    items.append(dict(proposal, status='failed', failure='invalid_2d_box'))
                    continue
                masks, scores, _ = predictor.predict(box=np.asarray(box, dtype=np.float32)[None], multimask_output=False)
                mask = np.asarray(masks).reshape(-1, *rgb.shape[:2])[0].astype(bool)
                if not mask.any():
                    items.append(dict(proposal, status='failed', failure='empty_predicted_mask'))
                    continue
                mask_path = frame_path(aux / 'masks', frame, '_' + digest(proposal['target_id'])[:16] + '.png')
                write_started = time.perf_counter()
                buffer = io.BytesIO()
                Image.fromarray(mask.astype(np.uint8)*255).save(buffer, format='PNG')
                _write_bytes(mask_path, buffer.getvalue())
                mask_write_seconds += time.perf_counter() - write_started
                items.append(dict(proposal, status='ok', mask_path=str(mask_path.relative_to(aux)),
                                  mask_sha256=mask_checksum(mask), mask_score=float(np.asarray(scores).reshape(-1)[0])))
            if device.type == 'cuda': torch.cuda.synchronize()
            total = time.perf_counter() - start - mask_write_seconds
            commit_record(path, frame, signature, items=items, depth_signature=depth_signature,
                          timing=dict(frontend_seconds=total, detection_seconds=detection_seconds,
                                      depth_seconds=depth_seconds))
        done += 1
        if done % 5 == 0 or done == len(frame_rows):
            print(f"前处理 {track} {done}/{len(frame_rows)} 帧，复用 {reused}", flush=True)
            write_json(root / 'worker_status.json', dict(stage='frontend', track=track, completed=done,
                                                       expected=len(frame_rows), reused=reused))
    result = dict(complete=True, frames=done, reused=reused, wall_seconds=time.perf_counter()-started,
                  signature=signature, depth_signature=depth_signature)
    write_json(root / track / task['phase'] / 'frontend.json', result)
    return result
