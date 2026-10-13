"""Resumable per-object inference. The worker receives no 3D target geometry."""

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .adapters.unipose9d import ObjectFailure
from .common import file_digest, read_json, write_json
from .suite_io import (ObjectInput, bind_identity, commit_record, frame_path, load_array,
                       mask_checksum, object_seed, read_record, require_mask)


def read_sensor_depth(frame, cache_root):
    path = Path(cache_root) / frame['sensor_depth_path']
    if file_digest(path) != frame['sensor_depth_sha256']:
        raise ValueError("实测深度文件校验不一致")
    with Image.open(path) as f:
        raw = np.asarray(f).copy()
    if raw.shape != tuple(frame['raw_image_hw']):
        raise ValueError("实测深度尺寸不一致")
    return np.ascontiguousarray(np.rot90(raw.astype(np.float32) / 1000., frame['rotation_k']))


def work(task):
    import torch
    torch.set_num_threads(4)
    root, track = Path(task['report_root']), task['track']
    model_id, config = task['model_id'], task['config']
    model_cfg = config['models'][model_id]
    options = dict(model_cfg['options'], device=task['device'])
    module_name, name = model_cfg['adapter'].split(':')
    adapter = getattr(importlib.import_module(module_name), name)(options)
    start = time.perf_counter()
    info = adapter.load()
    init_seconds = time.perf_counter() - start
    model_root = root / track / 'models' / model_id
    aux = root / 'aux' / track
    is_wild = model_id == 'wilddet3d-box'
    aux_identity = None if is_wild else read_json(aux / 'model.json')
    identity = dict(benchmark_id=task['benchmark_id'], model_id=model_id, track=track,
        options=options, model=info, seed=config['seed'],
        input_manifest=task['input_manifest_sha256'], worker_sha256=file_digest(__file__),
        io_sha256=file_digest(Path(__file__).with_name('suite_io.py')),
        inherited_adapter_sha256=file_digest(Path(__file__).parent/'adapters/wilddet.py') if is_wild else None,
        adapter_sha256=file_digest(sys.modules[module_name].__file__),
        frontend=aux_identity,
        sensor_depth_manifest=task['sensor_depth_manifest_sha256'] if options.get('depth_model') else None,
        predicted_depth=None if options.get('depth_model') or is_wild else
            read_json(root / 'aux/predicted_depth/model.json'))
    signature = bind_identity(model_root / 'model.json', identity)
    device = torch.device(task['device'])
    warmup = True
    total_targets = failures = done = reused = 0
    started = time.perf_counter()
    try:
        for frame in task['frames']:
            path = frame_path(model_root / 'predictions', frame)
            if path.exists():
                record = read_record(path, frame['frame_id'], signature)
                reused += 1
            else:
                with Image.open(frame['rgb_path']) as f:
                    rgb = np.asarray(f.convert('RGB')).copy()
                if list(rgb.shape[:2]) != frame['image_hw'] or mask_checksum(rgb) != frame['upright_rgb_sha256']:
                    raise ValueError("推理 RGB 与冻结输入不一致")
                auxiliary = None if is_wild else read_record(frame_path(aux/'frames', frame),
                                                            frame['frame_id'], aux_identity['signature'])
                proposals = frame['prompts'] if is_wild else auxiliary['items']
                if not is_wild and options.get('depth_model'):
                    depth = read_sensor_depth(frame, config['image_cache'])
                elif not is_wild:
                    depth_root = root / 'aux/predicted_depth'
                    depth_record = read_record(frame_path(depth_root, frame), frame['frame_id'],
                                               identity['predicted_depth']['signature'])
                    depth = load_array(depth_root, depth_record['array'])
                else:
                    depth = None
                items = []
                for proposal in proposals:
                    write_json(root/'worker_status.json',dict(stage='inference',track=track,model=model_id,
                        frames_completed=done,expected_frames=len(task['frames']),frame_id=frame['frame_id'],
                        completed_targets=total_targets+len(items),target_index=len(items)+1,
                        targets_in_frame=len(proposals),target_id=proposal['target_id']))
                    seed = object_seed(config['seed'], frame['frame_id'], proposal['target_id'])
                    if proposal.get('status') == 'failed':
                        items.append(dict(target_id=proposal['target_id'], status='failed',
                                          failure=proposal['failure'], detection=None, seed=seed))
                        continue
                    mask = None if is_wild else require_mask(aux, proposal, frame['image_hw'])
                    sample = ObjectInput(rgb, np.asarray(frame['intrinsics']), tuple(proposal['box_xyxy']), mask, depth)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    try:
                        detection, quality = adapter.predict_object(sample, seed)
                        if track == 'text':
                            detection['score'] = proposal['detection_score']
                            detection['class_id'] = proposal['class_id']
                        item = dict(target_id=proposal['target_id'], status='ok', detection=detection, quality=quality, seed=seed)
                    except ObjectFailure as exc:
                        item = dict(target_id=proposal['target_id'], status='failed', failure=str(exc), detection=None, seed=seed)
                    if device.type == 'cuda': torch.cuda.synchronize()
                    item['predict_seconds'] = time.perf_counter()-start
                    item['warmup'] = warmup
                    item['peak_memory_bytes'] = int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else None
                    warmup = False
                    items.append(item)
                front = auxiliary['timing'] if auxiliary else {}
                record = commit_record(path, frame, signature, items=items,
                    timing=dict(frontend_seconds=front.get('frontend_seconds', 0.),
                                depth_seconds=0. if options.get('depth_model') or is_wild else front.get('depth_seconds', 0.),
                                predict_seconds=sum(i.get('predict_seconds', 0.) for i in items)))
            total_targets += len(record['items'])
            failures += sum(i['status'] != 'ok' for i in record['items'])
            done += 1
            if done % 5 == 0 or done == len(task['frames']):
                status = dict(stage='inference', track=track, model=model_id, frames=done,
                              expected_frames=len(task['frames']), targets=total_targets, failures=failures, reused=reused)
                write_json(root / 'worker_status.json', status)
                print(f"{model_id} {done}/{len(task['frames'])} 帧，{total_targets} 目标，失败 {failures}，复用 {reused}", flush=True)
        result = dict(complete=True, signature=signature, frames=done, targets=total_targets,
                      failures=failures, reused=reused, init_seconds=init_seconds,
                      wall_seconds=time.perf_counter()-started)
        write_json(root/track/task['phase']/(model_id+'_inference.json'), result)
        return result
    finally:
        adapter.close()


if __name__ == '__main__':
    import json
    task = read_json(sys.argv[1])
    if task.get('job') == 'frontend':
        from .suite_frontend import work as frontend
        result = frontend(task)
    else:
        result = work(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
