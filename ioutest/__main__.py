"""CLI for benchmark preparation, local images, inference and offline scoring."""

import argparse
import json
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = MODULE_ROOT / "runs" / "arkit-validation-v1"


def _prepare_arguments(parser):
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/data/openBenchmark/ARKitScenes"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--scans", nargs="+")
    parser.add_argument("--max-scans", type=int)
    parser.add_argument("--frames-per-scan", type=int, default=20)
    parser.add_argument("--minimum-interval", type=float, default=.5)
    parser.add_argument("--depth-stride", type=int, default=2)
    parser.add_argument("--visible-points", type=int, default=20,
                        help="GT 框内观测深度点数必须严格大于该值")


def _image_cache_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image-cache", type=Path,
                       help="指定本地图片缓存；默认自动检测 ioutest/cache/<run-dir 名称>")
    group.add_argument("--no-image-cache", dest="image_cache", action="store_const", const=False,
                       help="跳过本地图片缓存，直接读取原始 ZIP")


def _infer_arguments(parser, run_dir=True):
    if run_dir:
        parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model-id", default="3dmood-swin-b")
    parser.add_argument("--adapter", default="3dmood", help="3dmood 或可导入的 module:AdapterClass")
    parser.add_argument("--adapter-options", type=Path, help="传给适配器构造函数的 JSON 配置")
    parser.add_argument("--device", default=None, help="3D-MOOD 默认 cuda；CPU 必须显式指定")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-detections", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--training-annotations", type=Path, nargs="+",
                        help="可选的模型训练 Omni3D/COCO 标注，用于扫描/visit 重叠审计")
    _image_cache_arguments(parser)


def _metric_arguments(parser):
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[.25, .5])
    parser.add_argument("--distance-thresholds", type=float, nargs="+", default=[.25, .5, 1.])
    parser.add_argument("--report-name", default="metrics")


def _options(args):
    options = json.loads(args.adapter_options.read_text()) if args.adapter_options else {}
    if not isinstance(options, dict):
        raise ValueError("adapter-options 必须是 JSON 对象")
    for name in ("device", "checkpoint", "max_detections", "score_threshold"):
        value = getattr(args, name)
        if value is not None:
            options[name] = str(value) if isinstance(value, Path) else value
    return options


def doctor(device="cuda", checkpoint=None):
    from .adapters.mood3d import DEFAULT_CHECKPOINT
    from .geometry import box_iou
    import numpy as np
    import scipy
    import PIL
    box = {"center": [0., 0., 0.], "size": [1., 1., 1.], "rotation": np.eye(3).tolist()}
    assert abs(box_iou(box, box) - 1.) < 1e-8
    report = {"numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__,
              "cpu_oriented_iou_selfcheck": "passed", "errors": []}
    try:
        import torch
        import opendet3d
        report.update(torch=torch.__version__, project_import=opendet3d.__file__,
                      cuda_available=torch.cuda.is_available())
        if device.startswith("cuda") and not report["cuda_available"]:
            report["errors"].append("当前进程不能访问 PPU；请在 DSW 原生 Terminal 执行设备推理")
        # Import torch first; required by compiled libtorch extension libraries.
        import vis4d_cuda_ops
        report["vis4d_cuda_ops"] = vis4d_cuda_ops.__file__
    except Exception as exc:
        report["errors"].append(f"模型环境：{exc}")
    path = Path(checkpoint or DEFAULT_CHECKPOINT).expanduser().resolve()
    report["checkpoint"] = str(path)
    if not path.is_file() or path.stat().st_size == 0:
        report["errors"].append("本地检测权重缺失或为空")
    else:
        report["checkpoint_bytes"] = path.stat().st_size
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import HF_HUB_CACHE
        report["bert_cache_root"] = HF_HUB_CACHE
        cached = {name: isinstance(try_to_load_from_cache("bert-base-uncased", name), str)
                  for name in ("config.json", "vocab.txt", "tokenizer.json", "model.safetensors", "pytorch_model.bin")}
        report["bert_cached_files"] = cached
        if not (cached["config.json"] and (cached["vocab.txt"] or cached["tokenizer.json"])
                and (cached["model.safetensors"] or cached["pytorch_model.bin"])):
            report["errors"].append("当前 Hugging Face 缓存目录中的 BERT 文件不完整，请沿用 Demo 成功时的 HF_HOME")
    except Exception as exc:
        report["errors"].append(f"BERT 缓存检查：{exc}")
    report["ready"] = not report["errors"]
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "benchmark":
        from .benchmark import main as benchmark_main
        try:
            benchmark_main(argv[1:])
            return 0
        except Exception as exc:
            print(f"ioutest benchmark: {exc}", file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(description="ARKitScenes 单帧三维检测评测，数据/模型/指标独立")
    parser.add_argument("--debug", action="store_true", help="错误时显示完整 traceback")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="检查本地模型环境、离线缓存与 CPU IoU")
    p.add_argument("--device", default="cuda"); p.add_argument("--checkpoint", type=Path)
    _prepare_arguments(commands.add_parser("prepare", help="读取 ZIP，冻结采样清单和逐帧 GT，可断点续跑"))
    p = commands.add_parser("cache-images", help="将冻结清单的图片提取到本地，可断点续跑")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--image-cache", type=Path, help="默认 ioutest/cache/<run-dir 名称>")
    p.add_argument("--workers", type=int, default=4)
    commands.add_parser("benchmark", help="三项目四输入组，200 帧试评测后全量，详情 benchmark --help")
    p = commands.add_parser("cache-depth", help="缓存同一冻结帧清单的 ARKit 米制深度源图片")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--image-cache", type=Path)
    p.add_argument("--workers", type=int, default=4)
    _infer_arguments(commands.add_parser("infer", help="一次加载模型，逐帧推理，按帧断点续跑"))
    p = commands.add_parser("evaluate", help="从完整预测缓存计算两类 AP，无需 GPU 或模型")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--model-id", default="3dmood-swin-b"); _metric_arguments(p)
    p = commands.add_parser("run", help="prepare → infer → evaluate")
    _prepare_arguments(p); _infer_arguments(p, run_dir=False); _metric_arguments(p)
    p = commands.add_parser("visualize", help="输出转正后 GT/预测投影检查图片")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--model-id", help="不传则仅绘制 GT")
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--score-threshold", type=float, default=.1, help="仅影响可视化，不影响 AP")
    p.add_argument("--show-ignore", action="store_true")
    _image_cache_arguments(p)
    p = commands.add_parser("import-predictions", help="校验并导入通用 JSONL 预测，不加载模型")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--model-id", required=True); p.add_argument("--path", type=Path, required=True)
    p.add_argument("--benchmark-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.device, args.checkpoint)
        elif args.command == "visualize":
            from .visualize import visualize
            result = {"images": visualize(args.run_dir, args.model_id, args.max_frames,
                                          args.score_threshold, args.show_ignore, args.image_cache)}
        elif args.command == "cache-images":
            from .image_cache import cache_images
            result = cache_images(args.run_dir, args.image_cache, args.workers)
        elif args.command == "cache-depth":
            from .depth_cache import cache_depth
            result = cache_depth(args.run_dir, args.image_cache, args.workers)
        elif args.command == "import-predictions":
            from .pipeline import import_predictions
            result = import_predictions(args.run_dir, args.model_id, args.path, args.benchmark_id)
        else:
            if args.command == "run" and args.adapter == "3dmood":
                options = _options(args)
                preflight = doctor(options.get("device", "cuda"), options.get("checkpoint"))
                if not preflight["ready"]:
                    raise RuntimeError("；".join(preflight["errors"]))
            if args.command in ("prepare", "run"):
                from .dataset import prepare
                result = prepare(args.dataset_root, args.run_dir, args.scans, args.max_scans,
                                 args.frames_per_scan, args.minimum_interval, args.depth_stride, args.visible_points)
                if args.command == "prepare":
                    from .common import read_json
                    audit = read_json(args.run_dir / "dataset_audit.json")
                    result = {"run_dir": str(args.run_dir), "benchmark_id": result["benchmark_id"],
                              "scans": audit["selected_scan_count"], "visits": audit["selected_visit_count"],
                              "frames": audit["frame_count"], "excluded_scans": audit["excluded_scans"]}
            if args.command in ("infer", "run"):
                from .pipeline import infer
                result = infer(args.run_dir, args.model_id, args.adapter, _options(args),
                               args.training_annotations, args.image_cache)
            if args.command in ("evaluate", "run"):
                from .pipeline import evaluate
                report = evaluate(args.run_dir, args.model_id, args.iou_thresholds, args.distance_thresholds,
                                  args.report_name)
                result = {"model_id": report["model_id"], "benchmark_id": report["benchmark_id"],
                          "frames": report["full"]["frame_count"], "summary": report["full"]["summary"],
                          "training_audit": report["training_audit"],
                          "report": str(args.run_dir / "models" / args.model_id / "reports" / (args.report_name + ".json"))}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if args.command == "doctor" and not result["ready"] else 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"ioutest: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
