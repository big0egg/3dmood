#!/usr/bin/env python3
"""Plan, or explicitly run, one real optimizer update for each selected pair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT.parent)]
from finefune.cli import model_check
from finefune.common import read, write
from finefune.config import resolve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=("3dmood", "wilddet3d"), default=["3dmood", "wilddet3d"])
    p.add_argument("--datasets", nargs="+", choices=("hot3d", "libero_matched", "cosmos_generic", "cosmos_mixed"),
                   default=["hot3d", "libero_matched", "cosmos_generic", "cosmos_mixed"])
    p.add_argument("--output-root", type=Path, default=Path("/root/finefune-native-check"))
    p.add_argument("--run", action="store_true", help="实际执行；不填只打印计划，不写产物或访问 PPU")
    args = p.parse_args()
    if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
        p.error("模型和数据不能重复")
    configs = []
    for model in args.models:
        for dataset in args.datasets:
            cfg = resolve(dict(model=model, dataset_config=PROJECT / "configs/datasets" / (dataset + ".yaml"),
                 output_dir=args.output_root / (model + "--" + dataset), device="cuda:0", resume="auto"))
            cfg["wandb"]["enabled"] = False
            configs.append(cfg)
    if not args.run:
        print(json.dumps(dict(executed=False, updates_per_experiment=1,
              jobs=[dict(model=c["model_id"], dataset=c["dataset"]["id"], output=c["root"])
                    for c in configs]), ensure_ascii=False, indent=2))
        return 0
    readiness = {model: model_check(next(c for c in configs if c["model_id"] == model))
                 for model in args.models}
    if not all(r["ready"] for r in readiness.values()):
        print(json.dumps(dict(executed=False, ready=False, checks=readiness), ensure_ascii=False, indent=2))
        return 2
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    env.pop("FINEFUNE_DISPATCHED", None)
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + str(PROJECT.parent)
    from ioutest.model_environment import child_environment
    with tempfile.TemporaryDirectory(prefix="finefune-native-configs-") as temp:
        for index, cfg in enumerate(configs):
            path = Path(temp) / (str(index) + ".json")
            write(path, cfg)
            subprocess.run([cfg["model"]["python"], "-u", "-B", str(PROJECT / "run.py"), "train",
                            "--config", str(path), "--resume", "auto", "--stop-after-updates", "1"],
                           env=child_environment(env, cfg['model'].get('environment', {})), cwd=PROJECT.parent, check=True)
            result = read(Path(cfg["root"]) / "status.json")
            if (result.get("stage") != "stopped" or result.get("update") != 1 or
                    not (Path(cfg["root"]) / "checkpoints/last.pt").is_file()):
                raise RuntimeError("未能确认一次真实参数更新：" + cfg["root"])
    print("全部选中组合完成一次真实参数更新；这些目录不是完整微调结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
