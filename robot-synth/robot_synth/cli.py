import argparse
import json
from pathlib import Path
import sys
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser(description="LIBERO ground-truth acquisition; no model training or pseudo-labels")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "doctor", "plan", "collect", "validate", "report"):
        child = commands.add_parser(command)
        child.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/libero_gt_v1.yaml")
        child.add_argument("--input", type=Path, help="Override original HDF5 root")
        child.add_argument("--output", type=Path, help="Override generated dataset root; must match existing plan")
        if command in ("inspect", "plan"):
            child.add_argument("--refresh", action="store_true", help="Refresh read-only source metadata inventory")
        if command == "collect":
            child.add_argument("--limit", type=int, help="Total desired camera images, including existing results; camera pairs stay together")
            child.add_argument("--resume", action="store_true")
        if command == "validate":
            child.add_argument("--allow-partial", action="store_true", help="Validate pilot samples without requiring all 11,000 images")
        if command == "report":
            child.add_argument("--max-previews", type=int, default=40)
    args = parser.parse_args(argv)
    try:
        from .config import load_config
        from .sources.libero_hdf5 import inspect_source
        config = load_config(args.config, input_root=args.input, output_root=args.output)
        if args.command in ("inspect", "plan", "doctor", "collect"):
            inventory = inspect_source(config, refresh=getattr(args, "refresh", False))
        if args.command == "inspect":
            from .planning.scenes import group_scenes
            result = {key: inventory[key] for key in ("task_count", "episode_count", "timesteps", "inventory_id")}
            result["scene_groups"] = [{"id": g["id"], "room": g["room"], "aliases": g["aliases"], "tasks": len(g["tasks"])}
                                      for g in group_scenes(inventory)]
        elif args.command == "doctor":
            from .simulation.doctor import diagnose
            result = diagnose(config, inventory)
        elif args.command == "plan":
            from .planning.sampling import build_plan
            plan = build_plan(config, inventory)
            result = {"plan_id": plan["plan_id"], "targets": plan["targets"],
                      "scenes": [{k: v for k, v in s.items()} for s in plan["scenes"]],
                      "candidate_states": len(plan["states"])}
        else:
            from .planning.sampling import load_plan
            plan = load_plan(config)
            if args.command == "collect":
                from .collection import collect
                progress = collect(config, plan, inventory, limit=args.limit, resume=args.resume)
                result = {k: v for k, v in progress.items() if k not in ("accepted", "rejected", "errors")}
            elif args.command == "validate":
                from .validation.dataset import validate_dataset
                result = validate_dataset(config, plan, allow_partial=args.allow_partial)
            else:
                if args.max_previews < 1:
                    raise ValueError("--max-previews must be positive")
                from .reporting.summary import report
                result = report(config, plan, maximum_previews=args.max_previews)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if result.get("ready") is False or result.get("valid") is False:
            return 2
        return 0
    except KeyboardInterrupt:
        print("采集已中断；完整提交的样本可通过 --resume 复用。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"失败：{error}", file=sys.stderr)
        if not isinstance(error, (ValueError, FileNotFoundError)):
            traceback.print_exc()
        return 1
