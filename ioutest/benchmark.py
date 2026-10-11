"""One-instance, separate-venv benchmark launcher. No model code imports here."""

import argparse
import fcntl
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

from .common import digest, read_json, utc_now, write_json
from .dataset import load_benchmark
from .depth_cache import DepthInputReader, cache_depth
from .image_cache import FrameInputReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "ioutest/runs/arkit-validation-v1"
DEFAULT_CONFIG = ROOT / "ioutest/benchmark_config.json"


def select_frames(frames, phase, pilot_scans=10, seed=20261010):
    if phase == "full":
        return frames
    if phase != "pilot" or pilot_scans < 1:
        raise ValueError("pilot 扫描数必须为正整数")
    groups = defaultdict(list)
    for f in frames:
        groups[f["video_id"]].append(f)
    pools = defaultdict(list)
    for scan in sorted(groups):
        pools[groups[scan][0]["sky_direction"]].append(scan)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected, visits = set(), set()
    while pools and len(selected) < pilot_scans:
        for key in sorted(list(pools)):
            pool = pools[key]
            while pool:
                scan = pool.pop()
                visit = groups[scan][0]["visit_id"]
                if visit not in visits:
                    selected.add(scan); visits.add(visit)
                    break
            if not pool:
                del pools[key]
            if len(selected) >= pilot_scans:
                break
    if len(selected) != pilot_scans:
        raise ValueError("无法选出指定数量的不同 visit 扫描")
    return [f for f in frames if f["video_id"] in selected]


def child_environment(config):
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               SAM3_COMPILE="0", XFORMERS_DISABLED="1", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    env.setdefault("MPLCONFIGDIR", str(ROOT / "ioutest/cache/matplotlib"))
    env.update({str(k): str(v) for k, v in config.get("environment", {}).items()})
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def preflight(models, env, device):
    reports = {}
    for model_id, model in models.items():
        imports = ("import opendet3d,vis4d_cuda_ops; " if model["adapter"] == "3dmood" else
                   "import wilddet3d; " if "wilddet" in model["adapter"] else
                   "from cubifyanything.cubify_transformer import make_cubify_transformer; ")
        script = ("import json,torch; " + imports +
                  "print(json.dumps({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),"
                  "'devices':torch.cuda.device_count()}))")
        for path in (model["python"], model["options"]["project_root"], model["options"]["checkpoint"]):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        lingbot = model["options"].get("lingbot_checkpoint")
        if lingbot and not Path(lingbot).is_file():
            raise FileNotFoundError(lingbot)
        result = subprocess.run([model["python"], "-B", "-c", script], env=env,
                                cwd=model["options"]["project_root"], text=True, capture_output=True,
                                timeout=120, check=False)
        if result.returncode:
            raise RuntimeError(f"{model_id} 环境检查失败：{result.stderr[-2000:]}")
        info = json.loads(result.stdout.strip().splitlines()[-1])
        reports[model_id] = info
        if device.startswith("cuda") and (not info["cuda_available"] or info["devices"] < 1):
            raise RuntimeError(f"{model_id}：当前执行进程无法访问 PPU。请在 DSW 原生 Terminal 运行此命令。")
        if not device.startswith("cuda") and model_id == "wilddet3d":
            raise RuntimeError("WildDet3D 的前向需要 CUDA Stream，不能使用 CPU 评测")
    return reports


def run_benchmark(run_dir=DEFAULT_RUN, report_root=None, phase="all", config_path=DEFAULT_CONFIG,
                  model_ids=None, pilot_scans=10, with_gravity=False, device="cuda", score_only=False):
    run_dir = Path(run_dir).resolve()
    report_root = Path(report_root or "~/report/arkitscenes-three-models-v1").expanduser().resolve()
    config = read_json(config_path)
    requested = model_ids or config["default_models"]
    if with_gravity and "cutr-rgb-gravity" not in requested:
        requested = [*requested, "cutr-rgb-gravity"]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("模型列表不能为空或重复")
    models = {key: config["models"][key] for key in requested}
    protocol, audit, frames, _ = load_benchmark(run_dir)
    inputs = FrameInputReader(run_dir, protocol, frames)
    if inputs.cache_root is None:
        raise ValueError("统一评测要求本地 RGB 缓存，请先执行 cache-images")
    phases = ["pilot", "full"] if phase == "all" else [phase]
    selections = {p: select_frames(frames, p, pilot_scans) for p in phases}
    env = child_environment(config)
    # Validate native device access before long processing and before writing report_root.
    checks = {} if score_only else preflight(models, env, device)
    if any(m["options"].get("depth_model") for m in models.values()) and not score_only:
        summary = inputs.cache_root / "depth_cache_summary.json"
        if not summary.is_file() or not read_json(summary).get("complete"):
            cache_depth(run_dir, inputs.cache_root)
        DepthInputReader(inputs.cache_root, protocol, frames)
    report_root.mkdir(parents=True, exist_ok=True)
    with (report_root / ".benchmark.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = {"benchmark_id": protocol["benchmark_id"], "run_dir": str(run_dir),
                    "pilot_scans": pilot_scans, "seed": 20261010, "config": config,
                    "device": device, "image_cache": str(inputs.cache_root)}
        existing = report_root / "experiment.json"
        if existing.exists() and read_json(existing)["settings"] != settings:
            raise ValueError("报告目录已有不同实验配置；请使用新的 report-root")
        write_json(existing, {"settings": settings, "created_utc": utc_now(), "preflight": checks,
                               "training_overlap": audit["official_training_visit_overlap"]})
        status_path = report_root / "status.json"
        try:
            for p in phases:
                selected = selections[p]
                phase_dir = report_root / p
                selection = {"benchmark_id": protocol["benchmark_id"], "phase": p,
                             "frame_ids": [f["frame_id"] for f in selected]}
                selection["selection_id"] = digest(selection)
                write_json(phase_dir / "selection.json", selection)
                write_json(status_path, {"status": "running", "phase": p, "models": requested,
                                          "expected_frames": len(selected), "updated_utc": utc_now()})
                for key, model in models.items():
                    if not score_only:
                        task_path = phase_dir / (key + "_task.json")
                        task = {"run_dir": str(run_dir), "report_root": str(report_root),
                                "phase": p, "model_id": key, "model": model, "device": device,
                                "image_cache": str(inputs.cache_root), "selection": selection}
                        write_json(task_path, task)
                        command = [model["python"], "-B", "-m", "ioutest.benchmark_worker", str(task_path)]
                        log_path = phase_dir / "logs" / (key + ".log")
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        print(f"[{p}] {key}：{len(selected)} 帧；日志 {log_path}", flush=True)
                        with log_path.open("a") as log:
                            subprocess.run(command, env=env, cwd=model["options"]["project_root"],
                                           stdout=log, stderr=subprocess.STDOUT, check=True)
                    from .benchmark_report import evaluate_model
                    evaluate_model(run_dir, report_root, p, key, model, selected)
                from .benchmark_report import assemble_report
                assemble_report(run_dir, report_root, p, models, selected)
                write_json(status_path, {"status": "complete", "phase": p, "models": requested,
                                          "frames": len(selected), "updated_utc": utc_now()})
        except BaseException as exc:
            write_json(status_path, {"status": "failed", "phase": p, "error": str(exc),
                                      "updated_utc": utc_now(), "note": "不会将缺失预测计为无检出；可原命令续跑"})
            raise
    return {"report_root": str(report_root), "phases": phases, "models": requested, "complete": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description="三项目四输入组统一 ARKitScenes 评测")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report-root", type=Path, default=Path("~/report/arkitscenes-three-models-v1"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("pilot", "full", "all"), default="all")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--pilot-scans", type=int, default=10)
    parser.add_argument("--with-gravity", action="store_true", help="额外运行 CuTR RGB + 已知传感器重力")
    parser.add_argument("--device", default="cuda", help="正式评测 cuda；cpu 仅适配器验证")
    parser.add_argument("--score-only", action="store_true", help="仅从完整数值预测生成报告，不加载模型")
    args = parser.parse_args(argv)
    result = run_benchmark(args.run_dir, args.report_root, args.phase, args.config, args.models,
                           args.pilot_scans, args.with_gravity, args.device, args.score_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
