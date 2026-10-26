"""Explicit train/data/evaluation/suite entry points, with isolated model envs."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from .common import (PROJECT, REPO, bind, checksum, data_path, digest,
                     jsonlines, lock, outside_sources, read, source_code, write)
from .config import dataset_config, document, path_value, resolve
from ioutest.model_environment import bert_cache_files, child_environment


def parser():
    p = argparse.ArgumentParser(description="finefune：多数据集三维几何头微调与实验调度")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "verify", "train", "evaluate", "export-eval"):
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, help="实验配置或已保存的 resolved_config.yaml")
        s.add_argument("--model", choices=("3dmood", "wilddet3d"))
        s.add_argument("--model-config", type=Path)
        s.add_argument("--origin-ckpt", type=Path)
        s.add_argument("--dataset-config", type=Path)
        s.add_argument("--dataset-root", type=Path)
        s.add_argument("--train-config", type=Path)
        s.add_argument("--output-dir", type=Path)
        s.add_argument("--epochs", type=int)
        s.add_argument("--lr", type=float)
        s.add_argument("--grad-accum-steps", type=int)
        s.add_argument("--seed", type=int)
        s.add_argument("--device")
        s.add_argument("--resume", nargs="?", const="auto")
        s.add_argument("--dry-run", action="store_true", help="只读解析和数据审计，不启动训练或写产物")
        if name == "train":
            s.add_argument("--stop-after-updates", type=int, help="在参数更新边界保存后停止；用于真实小批量验证")
        if name == "verify":
            s.add_argument("--images", action="store_true")
        if name in ("evaluate", "export-eval"):
            s.add_argument("--variant", choices=("official", "finetuned"), default="finetuned")
            s.add_argument("--finetuned-ckpt", type=Path, help="覆盖本实验的最终三维头；核验基础权重和训练来源")
            s.add_argument("--tracks", nargs="+", choices=("text", "gt-box"), default=["text"])
            s.add_argument("--tests-config", type=Path, help="已有 ioutest 配置；不填则用本实验冻结 eval")
            s.add_argument("--datasets", nargs="+", help="从 tests-config 选择测试集 ID")
            s.add_argument("--stage", choices=("check", "infer", "score", "latency", "report", "all"), default="all")
    s = sub.add_parser("suite", help="按配置顺序执行基线、训练和微调后评测")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--models", nargs="+", choices=("3dmood", "wilddet3d"))
    s.add_argument("--datasets", nargs="+")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("report")
    s.add_argument("--suite-dir", type=Path, required=True)
    s = sub.add_parser("convert-config", help="显式转换旧 HOT3D 配置，不改旧权重或数据")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--model", choices=("3dmood", "wilddet3d"), required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    return p


def save_config(cfg):
    import yaml
    from .common import atomic_bytes
    atomic_bytes(Path(cfg["root"]) / "resolved_config.yaml",
                 yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode())


def model_check(cfg):
    m = cfg["model"]
    files = {"python": m["python"], **{k: v for k, v in m["options"].items()
             if k in ("checkpoint", "project_root", "lingbot_checkpoint")}}
    missing = [v for v in files.values() if not Path(v).exists()]
    if missing:
        return dict(ready=False, missing=missing)
    caches = {}
    if cfg['model_id'] == '3dmood':
        try: caches = bert_cache_files(m.get('environment', {}))
        except (ValueError, OSError) as exc: return dict(ready=False, error=str(exc))
    code = (
        "import torch,sys,json; "
        "ok=not sys.argv[1].startswith('cuda') or torch.cuda.is_available(); "
        "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,ready=ok))); "
        "sys.exit(0 if ok else 2)"
    )
    try:
        r = subprocess.run([m["python"], "-B", "-c", code, cfg["device"]],
                           env=child_environment(os.environ, m.get('environment', {})),
                           capture_output=True, text=True, timeout=120)
        return dict(ready=r.returncode == 0, python=m["python"], caches=caches, detail=(r.stdout + r.stderr)[-3000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(ready=False, python=m["python"], error=str(exc))


def dispatch(cfg, argv):
    expected = os.path.abspath(cfg["model"]["python"])
    if os.path.abspath(sys.executable) == expected:
        return None
    if os.environ.get("FINEFUNE_DISPATCHED"):
        raise RuntimeError("模型解释器路径不匹配：" + sys.executable + " / " + expected)
    env = dict(child_environment(os.environ, cfg['model'].get('environment', {})), FINEFUNE_DISPATCHED="1", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + str(REPO)
    print("切换到模型环境：" + expected, flush=True)
    return subprocess.run([expected, "-B", str(PROJECT / "run.py"), *argv], env=env).returncode


def prepare_inputs(cfg, data=None):
    with lock(Path(cfg["root"]) / ".prepare.lock"):
        return _prepare_inputs(cfg, data)


def _prepare_inputs(cfg, data=None):
    from .datasets import prepare
    root = Path(cfg["root"])
    marker = root / "config_identity.json"
    existing = [p for p in root.iterdir() if p.name != ".prepare.lock"] if root.exists() else []
    if existing and not marker.exists():
        raise ValueError("输出目录已有非本实验产物，请指定新的 --output-dir：" + str(root))
    identity = {k: cfg[k] for k in ("model", "dataset", "training", "seed", "device")}
    identity["weight_hashes"] = {
        k: checksum(v) for k, v in cfg["model"]["options"].items()
        if k in ("checkpoint", "lingbot_checkpoint")
    }
    identity["source_code"] = source_code()
    bind(marker, identity)
    protocol, train, evaluation = prepare(cfg, data)
    save_config(cfg)
    jsonlines(Path(cfg["root"]) / "train.jsonl", train)
    write(Path(cfg["root"]) / "dataset_reference.json", dict(
        dataset_id=protocol["dataset_id"], training_id=protocol["benchmark_id"],
        source=cfg["dataset"], normalized_root=str(data_path(cfg)),
        train_frames=len(train), eval_frames=len(evaluation)))
    return protocol, train, evaluation


def select_tests(args):
    if not args.tests_config:
        if args.datasets:
            raise ValueError("--datasets 需要 --tests-config")
        return None
    from ioutest.experiment_config import load_config
    config = load_config(args.tests_config)
    keys = args.datasets or list(config["datasets"])
    if len(keys) != len(set(keys)) or set(keys) - set(config["datasets"]):
        raise ValueError("未知或重复的测试集 ID")
    return {k: config["datasets"][k] for k in keys}


def select_evaluation_profile(args):
    from ioutest.experiment_config import DEFAULT_CONFIG, load_config
    from .evaluation import evaluation_settings
    profile = evaluation_settings(load_config(args.tests_config or DEFAULT_CONFIG))
    for key in ('device', 'seed'):
        if getattr(args, key, None) is not None:
            profile['execution'][key] = getattr(args, key)
    return profile


def execute(args, argv):
    if args.command == "suite":
        from .experiments.runner import run, plan
        return plan(args.config, args.models, args.datasets) if args.dry_run else run(args.config, args.models, args.datasets)
    if args.command == "report":
        from .experiments.runner import report
        return report(args.suite_dir)
    if args.command == "convert-config":
        from .compatibility import convert_config
        return convert_config(args.input, args.model, args.output, args.output_dir)
    if args.command in ("prepare", "verify") and not args.model and not args.config:
        if not args.output_dir:
            raise ValueError("必须提供 --output-dir")
        cfg = dict(root=str(args.output_dir.expanduser().resolve()))
        if args.command == "prepare":
            if not args.dataset_config:
                raise ValueError("必须提供 --dataset-config")
            cfg["dataset"] = dataset_config(args.dataset_config, args.dataset_root)
            outside_sources(cfg["root"], [cfg["dataset"]["root"]])
            from .datasets import load_source, prepare, summary
            data = load_source(cfg["dataset"])
            if not args.dry_run:
                prepare(cfg, data)
            return dict(dry_run=args.dry_run, **summary(data))
        from .datasets import verify
        return verify(cfg, args.images)
    cfg = resolve(args)
    if args.command == "train" and args.stop_after_updates is not None and args.stop_after_updates < 1:
        raise ValueError("--stop-after-updates 必须为正整数")
    if args.command == "train" and not args.dry_run:
        code = dispatch(cfg, argv)
        if code is not None:
            return code
        readiness = model_check(cfg)
        if not readiness["ready"]:
            raise RuntimeError("模型环境或 PPU 未就绪：" + json.dumps(readiness, ensure_ascii=False))
    if args.command in ("check", "prepare", "train"):
        from .datasets import load_source, summary
        data = load_source(cfg["dataset"])
        count = len(data["train"]["frames"])
        if args.dry_run:
            return dict(dry_run=True, resolved_config=cfg, data=summary(data),
                        optimizer_updates=cfg["training"]["epochs"] * math.ceil(count / cfg["training"]["accumulate"]))
        if args.command == "check":
            return dict(data=summary(data), **model_check(cfg))
        prepare_inputs(cfg, data)
        if args.command == "prepare":
            return dict(complete=True, **summary(data))
        from .training.engine import train
        previous = {}
        def terminate(signum, frame):
            raise KeyboardInterrupt("收到停止信号；保存最近完整参数更新")
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, terminate)
        try:
            result = train(cfg, stop_after_updates=args.stop_after_updates)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if result.get("complete"):
            write(Path(cfg["root"]) / "model_identity.json", read(Path(cfg["root"]) / "training.json"))
        return result
    if args.command == "verify":
        from .datasets import verify
        return verify(cfg, args.images)
    tests = select_tests(args)
    profile = select_evaluation_profile(args)
    from .evaluation import validate_selection
    validate_selection(cfg, args.tracks, args.variant, args.finetuned_ckpt)
    if args.dry_run:
        return dict(dry_run=True, resolved_config=cfg, tests=tests,
                    evaluation_profile=profile,
                    variant=args.variant, tracks=args.tracks, stage=args.stage,
                    finetuned_ckpt=str(args.finetuned_ckpt) if args.finetuned_ckpt else None)
    from .evaluation import evaluate, export_config
    if args.command == "export-eval":
        return dict(config=str(export_config(cfg, tests, args.tracks, args.variant,
                                            head_path=args.finetuned_ckpt, evaluation_profile=profile)))
    return evaluate(cfg, tests, args.tracks, args.variant, args.stage,
                    head_path=args.finetuned_ckpt, evaluation_profile=profile)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    try:
        result = execute(args, argv)
        if isinstance(result, int):
            return result
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get("ready") is False else 0
    except KeyboardInterrupt:
        print("任务已停止；使用同一配置和 --resume auto 恢复。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
