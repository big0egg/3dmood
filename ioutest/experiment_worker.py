"""Private model worker: allowed observations in, numeric predictions out."""
import importlib
import importlib.metadata
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from .checkpoints import load_head
from .common import digest, file_digest, read_json, write_json
from .experiment_data import read_rgb
from .hot3d.common import bind, export_detection
from .records import commit, read_record, record_path
from .schema import ModelInput
from .suite_io import ObjectInput, object_seed
from .adapters.unipose9d import ObjectFailure
from .timing import summarize, synchronize, timed
from .worker_rpc import FrontendClient

ADAPTERS = {'3dmood': 'mood3d:Mood3DAdapter', 'wilddet3d': 'wilddet:WildDetAdapter',
            'unipose9d': 'unipose9d:UniPoseAdapter', 'cutr': 'cutr:CuTRAdapter'}

def create_adapter(task):
    spec = task['model']; family = spec['family']; track = task['track']
    target = 'wildbox:WildBoxAdapter' if family == 'wilddet3d' and track == 'gt-box' else ADAPTERS[family]
    module, cls = target.split(':')
    return getattr(importlib.import_module('ioutest.adapters.' + module), cls)(dict(task['options'],
        device=task['execution']['device'], environment=task.get('environment', {})))

def sensor_depth(frame):
    path = Path(frame['depth_path'])
    if file_digest(path) != frame['depth_sha256']: raise ValueError('实测深度摘要变化')
    if path.suffix == '.npy': depth = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as im: depth = np.asarray(im).copy()
    depth = np.asarray(depth, dtype=np.float32) * frame['depth_scale']
    if frame.get('rotation_k'): depth = np.ascontiguousarray(np.rot90(depth, frame['rotation_k']))
    if list(depth.shape) != frame['image_hw'] or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError('实测深度必须与 RGB 对齐且为非负米制')
    return depth

def predict_frame(task, adapter, frontend, frame, rgb, depth, torch):
    family = task['model']['family']; track = task['track']; device = task['execution']['device']
    classes = task['classes']; k = np.asarray(frame['intrinsics']); items = []; front_timing = {}
    if family == 'unipose9d' or track == 'gt-box':
        if frontend is not None:
            front = frontend.predict(rgb, k, frame.get('prompts', []), depth)
            proposals = front['items']; depth = front['depth']; front_timing = front['timing']
        else: proposals = frame['prompts']
        def objects():
            for proposal in proposals:
                target = proposal['target_id']; seed = object_seed(task['execution']['seed'], frame['frame_id'], target)
                if proposal.get('status') == 'failed':
                    items.append(dict(target_id=target, status='failed', failure=proposal['failure'], detection=None)); continue
                sample = ObjectInput(rgb, k, tuple(proposal['box_xyxy']), proposal.get('mask'), depth)
                try:
                    box, quality = adapter.predict_object(sample, seed)
                    if track == 'text': box = dict(box, class_id=proposal['class_id'], score=proposal['detection_score'])
                    items.append(dict(target_id=target, status='ok', detection=export_detection(box, classes), quality=quality, seed=seed))
                except ObjectFailure as exc:
                    items.append(dict(target_id=target, status='failed', failure=str(exc), detection=None, seed=seed))
            return [i['detection'] for i in items if i['status'] == 'ok']
        detections, seconds = timed(torch, device, objects)
    else:
        names = tuple(c['name'] for c in classes); prompts = tuple(c['prompt'] for c in classes)
        sample = ModelInput(frame['frame_id'], rgb, k, names, prompts)
        if family == 'cutr':
            sample = SimpleNamespace(frame_id=frame['frame_id'], rgb=rgb, intrinsics=k, class_names=names,
                                     prompts=prompts, depth=depth, gravity=None)
        detections, seconds = timed(torch, device, lambda: [export_detection(b, classes) for b in adapter.predict(sample)])
    return detections, items, dict(predict_seconds=seconds, frontend_seconds=front_timing.get('frontend_seconds', 0.),
                                   depth_seconds=front_timing.get('depth_seconds', 0.))

def work(task):
    import torch
    torch.set_num_threads(task['execution']['threads']); torch.manual_seed(task['execution']['seed'])
    np.random.seed(task['execution']['seed']); device = task['execution']['device']; root = Path(task['run_root'])
    if str(device).startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('当前进程不可见 PPU，不回退 CPU')
    adapter = create_adapter(task); frontend = None
    try:
        started = time.perf_counter()
        info = adapter.load(); info = load_head(adapter, task['model']['family'], task['variant'], info)
        if task['model']['family'] == 'unipose9d':
            config = dict(task['frontend'], sensor_depth=task['options'].get('depth_model', False))
            frontend = FrontendClient(config['python'], config, task['track'], task['classes'], task['execution'])
            info['frontend'] = frontend.identity
        initialization_seconds = time.perf_counter() - started
        versions = {}
        for package in ('torch', 'torchvision', 'transformers', 'vis4d', 'numpy', 'scipy', 'timm', 'sam2', 'sam3'):
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package] = None
        info.update(device_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
                    torch=torch.__version__, versions=versions, python=sys.version, cuda_version=torch.version.cuda)
        identity = dict(request_id=task['request_id'], runtime=info)
        model_id = bind(root / 'model.json', identity)
        reused = 0; results = []; warmed = False; warmup_seconds = 0.
        for index, frame in enumerate(task['frames']):
            path = record_path(root, frame['frame_id'])
            if task['mode'] == 'infer' and path.exists():
                record = read_record(path, frame['frame_id'], task['request_id'], model_id); reused += 1
            else:
                rgb = read_rgb(frame); depth = sensor_depth(frame) if frame.get('depth_path') else None
                if not warmed:
                    warmup_started = time.perf_counter()
                    for _ in range(task['timing']['warmup_frames']):
                        predict_frame(task, adapter, frontend, frame, rgb, depth, torch)
                    warmup_seconds = time.perf_counter() - warmup_started
                    warmed = True
                synchronize(torch, device); begin = time.perf_counter()
                detections, items, timing = predict_frame(task, adapter, frontend, frame, rgb, depth, torch)
                synchronize(torch, device); timing['pipeline_wall_seconds'] = time.perf_counter() - begin
                timing['warmup'] = False
                record = dict(schema='ioutest.prediction.v2', status='ok', frame_id=frame['frame_id'],
                    request_id=task['request_id'], model_id=model_id, detections=detections, items=items, timing=timing)
                if task['mode'] == 'infer': record = commit(path, record)
                else: record = commit(root / 'latency_records' / path.name, record)
            results.append(record)
            write_json(root / 'progress.json', dict(stage=task['mode'], completed=index+1, total=len(task['frames']), reused=reused))
            if (index+1) % 25 == 0 or index+1 == len(task['frames']):
                print(f'{task["model_id"]} {task["mode"]} {index+1}/{len(task["frames"])}，复用 {reused}', flush=True)
        latency = summarize(results); latency.update(request_id=task['request_id'], model_id=model_id,
            warmup_passes=task['timing']['warmup_frames'] if warmed else 0, source=task['mode'], complete=True,
            initialization_seconds=initialization_seconds, warmup_seconds=warmup_seconds, reused_records=reused)
        write_json(root / 'latency.json', latency)
        return dict(complete=True, frames=len(results), reused=reused)
    finally:
        if frontend is not None: frontend.close()
        adapter.close()

if __name__ == '__main__':
    print(work(read_json(sys.argv[1])))
