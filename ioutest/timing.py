"""Observed per-frame latency, separate from historical sums of compute stages."""
import time
import numpy as np

def synchronize(torch, device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)

def timed(torch, device, function):
    synchronize(torch, device); begin = time.perf_counter()
    value = function()
    synchronize(torch, device)
    return value, time.perf_counter() - begin

def summarize(records):
    measured = []; warmups = []; stage_sum = []
    for record in records:
        t = record.get('timing', {})
        vals = [float(t.get(k, 0.)) for k in ('predict_seconds', 'frontend_seconds', 'depth_seconds')]
        if any(not np.isfinite(v) or v < 0 for v in vals): raise ValueError('阶段耗时非法')
        stage_sum.append(sum(vals))
        if t.get('pipeline_wall_seconds') is not None:
            value = float(t['pipeline_wall_seconds'])
            if not np.isfinite(value) or value < 0: raise ValueError('实际延迟非法')
            (warmups if t.get('warmup', False) else measured).append(value)
    total = sum(measured)
    return dict(status='measured' if measured else 'unavailable', measured_frames=len(measured),
        warmup_frames=len(warmups), mean_ms=1000*total/len(measured) if measured else None,
        p95_ms=float(np.percentile(measured, 95)*1000) if measured else None,
        fps=len(measured)/total if total else None, stage_sum_seconds=sum(stage_sum),
        stage_sum_mean_ms=1000*sum(stage_sum)/len(stage_sum) if stage_sum else None,
        warmup_seconds=sum(warmups), definition='decoded in-memory input to CPU numerical boxes; includes required IPC',
        excludes=['model_initialization', 'warmup_in_primary_summary', 'source_image_decode', 'artifact_writes', 'scoring'])

import math


def inference_speed(result):
    """Derive throughput from recorded compute time, without rerunning AP."""
    timing = result['timing']
    frames = result['frames']
    predict, frontend, depth = [float(timing[name]) for name in
        ('total_predict_seconds', 'total_frontend_seconds', 'total_depth_seconds')]
    if frames < 0 or any(not math.isfinite(t) or t < 0 for t in (predict, frontend, depth)):
        raise ValueError('速度统计要求非负帧数和有限的非负计时')
    pipeline = predict + frontend + depth
    targets = (result.get('conditional') or {}).get('targets')
    return dict(frames=frames, targets=targets, predict_seconds=predict,
                frontend_seconds=frontend, depth_seconds=depth, pipeline_seconds=pipeline,
                mean_predict_seconds_per_frame=predict/frames if frames and predict else None,
                fps_predict=frames/predict if frames and predict else None,
                mean_pipeline_seconds_per_frame=pipeline/frames if frames and pipeline else None,
                fps_pipeline=frames/pipeline if frames and pipeline else None,
                mean_predict_seconds_per_target=predict/targets if targets and predict else None,
                mean_pipeline_seconds_per_target=pipeline/targets if targets and pipeline else None,
                historical_reuse=timing['historical_reuse'], includes_warmup=True,
                excludes_io_initialization_and_scoring=True)
