"""One model, one environment, one load; durable per-frame numerical outputs."""

import fcntl
import gzip
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .adapters import create_adapter
from .common import digest, implementation_digest, read_json, utc_now, write_json
from .dataset import load_benchmark
from .depth_cache import DepthInputReader, SensorModelInput, camera_gravity
from .image_cache import FrameInputReader
from .schema import SCHEMA_VERSION, validate_box


def prediction_path(root, frame):
    # Source names are frozen but still prohibit directory traversal.
    scan = frame["video_id"]
    name = Path(frame["rgb_entry"]).stem
    if Path(scan).name != scan or scan in (".", "..") or not name:
        raise ValueError("非法扫描/帧文件名")
    return Path(root) / "predictions" / scan / (name + ".json.gz")


def write_prediction(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            value = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            stream.write(gzip.compress(value, compresslevel=3, mtime=0))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_prediction(path, frame, signature, benchmark_id, semantic_labels):
    with gzip.open(path, "rt") as stream:
        value = json.load(stream)
    if (value.get("frame_id") != frame["frame_id"] or value.get("signature") != signature or
            value.get("benchmark_id") != benchmark_id or value.get("schema_version") != SCHEMA_VERSION or
            value.get("status") != "ok"):
        raise ValueError(f"预测标识、版本或状态不一致：{path}")
    for box in value["detections"]:
        validate_box(box, prediction=True, allow_unlabeled=not semantic_labels)
    return value


def _versions():
    result = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "vis4d", "transformers", "timm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def work(task):
    import torch
    torch.set_num_threads(4)
    run_dir, report_root = Path(task["run_dir"]), Path(task["report_root"])
    protocol, _, all_frames, _ = load_benchmark(run_dir)
    selection = task["selection"]
    if selection["benchmark_id"] != protocol["benchmark_id"]:
        raise ValueError("子进程输入 benchmark 不一致")
    ids = selection["frame_ids"]
    available = {f["frame_id"]: f for f in all_frames}
    if len(set(ids)) != len(ids) or not set(ids).issubset(available):
        raise ValueError("子进程采样清单存在重复或未知帧")
    frames = [available[i] for i in ids]
    inputs = FrameInputReader(run_dir, protocol, all_frames, task["image_cache"])
    options = dict(task["model"]["options"], device=task["device"])
    depth = DepthInputReader(inputs.cache_root, protocol, all_frames) if options.get("depth_model") else None
    model_root = report_root / "models" / task["model_id"]
    model_root.mkdir(parents=True, exist_ok=True)
    with (model_root / ".worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        adapter, adapter_info = create_adapter(task["model"]["adapter"], options)
        start = time.perf_counter()
        info = adapter.load()
        init_seconds = time.perf_counter() - start
        is_cuda = task["device"].startswith("cuda")
        if is_cuda:
            chosen_device = torch.device(task["device"])
            torch.cuda.set_device(chosen_device.index if chosen_device.index is not None else torch.cuda.current_device())
            torch.cuda.synchronize()
        device_info = {"device": task["device"], "cuda_available": torch.cuda.is_available()}
        if is_cuda:
            prop = torch.cuda.get_device_properties(torch.device(task["device"]))
            device_info.update(name=prop.name, total_memory=int(prop.total_memory))
        provenance = {"benchmark_id": protocol["benchmark_id"], "schema_version": SCHEMA_VERSION,
                      "model_id": task["model_id"], "adapter": adapter_info, "options": options,
                      "model": info, "versions": _versions(), "device": device_info,
                      "input_source": inputs.source_info, "implementation_sha256": implementation_digest(),
                      "depth_source": depth.source_info if depth else None}
        # Preparation elapsed time is not prediction identity.
        if provenance["depth_source"]:
            provenance["depth_source"] = {k: v for k, v in provenance["depth_source"].items()
                                           if k != "elapsed_seconds"}
        signature = digest(provenance)
        info_path = model_root / "model.json"
        if info_path.exists() and read_json(info_path)["signature"] != signature:
            adapter.close()
            raise ValueError("模型、代码、输入或环境已变化，不能复用该目录预测；请用新 report-root")
        write_json(info_path, {"signature": signature, "provenance": provenance, "created_utc": utc_now()})
        grouped = defaultdict(list)
        for frame in frames:
            grouped[frame["video_id"]].append(frame)
        completed = reused = executed = 0
        started = time.perf_counter()
        try:
            for scan, scan_frames in grouped.items():
                with inputs.scan(scan) as reader:
                    for frame in scan_frames:
                        path = prediction_path(model_root, frame)
                        if path.exists():
                            read_prediction(path, frame, signature, protocol["benchmark_id"],
                                            task["model"]["semantic_labels"])
                            reused += 1
                        else:
                            start = time.perf_counter()
                            sample = reader.model_input(frame)
                            if depth or options.get("use_gravity"):
                                sample = SensorModelInput(sample.frame_id, sample.rgb, sample.intrinsics,
                                    depth=depth.read(frame) if depth else None,
                                    gravity=camera_gravity(frame) if options.get("use_gravity") else None)
                            input_seconds = time.perf_counter() - start
                            if is_cuda:
                                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                            start = time.perf_counter()
                            detections = [d.to_dict() for d in adapter.predict(sample)]
                            for box in detections:
                                validate_box(box, prediction=True,
                                             allow_unlabeled=not task["model"]["semantic_labels"])
                            if is_cuda:
                                torch.cuda.synchronize()
                            elapsed = time.perf_counter() - start
                            record = {"benchmark_id": protocol["benchmark_id"], "signature": signature,
                                "schema_version": SCHEMA_VERSION, "frame_id": frame["frame_id"],
                                "status": "ok", "detections": detections, "timing": {
                                    "input_seconds": input_seconds, "predict_seconds": elapsed,
                                    "warmup": executed == 0, "peak_memory_bytes":
                                        int(torch.cuda.max_memory_allocated()) if is_cuda else None}}
                            write_prediction(path, record)
                            executed += 1
                        completed += 1
                        if completed % 20 == 0 or completed == len(frames):
                            print(f"{task['model_id']} {completed}/{len(frames)} 帧，复用 {reused}", flush=True)
            result = {"complete": True, "model_id": task["model_id"], "frames": len(frames),
                      "selection_id": selection["selection_id"], "signature": signature,
                      "reused": reused, "executed": executed, "init_seconds": init_seconds,
                      "wall_seconds": time.perf_counter() - started, "created_utc": utc_now()}
            write_json(report_root / task["phase"] / (task["model_id"] + "_inference.json"), result)
            return result
        except BaseException as exc:
            write_json(report_root / task["phase"] / (task["model_id"] + "_failure.json"),
                       {"error": str(exc), "completed": completed, "complete": False, "created_utc": utc_now()})
            raise
        finally:
            adapter.close()


if __name__ == "__main__":
    print(json.dumps(work(read_json(sys.argv[1])), ensure_ascii=False, indent=2))
