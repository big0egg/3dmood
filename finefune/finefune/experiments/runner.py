"""Model x dataset suites; evaluation is delegated to ioutest."""
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import subprocess

from ..common import (PROJECT, REPO, atomic_bytes, bind, checksum, digest, lock,
                      read, write)
from ..config import document, path_value, resolve


def safe_name(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("非法实验 ID：" + str(value))
    return value


def plan(path, models=None, datasets=None):
    c, base = document(path)
    if c.get("schema_version") != "finefune.suite.v1":
        raise ValueError("suite schema_version 必须为 finefune.suite.v1")
    selected_models = models or list(c["models"])
    selected_data = datasets or list(c["datasets"])
    for keys, mapping in ((selected_models, c["models"]), (selected_data, c["datasets"])):
        if not keys or len(keys) != len(set(keys)) or set(keys) - set(mapping):
            raise ValueError("未知、空或重复的 suite 选择")
        for key in keys:
            safe_name(key)
    root = path_value(c["output_dir"], base)
    evaluation = deepcopy(c.get("evaluation", {}))
    if evaluation.get("config"):
        evaluation["config"] = path_value(evaluation["config"], base)
        from ioutest.experiment_config import load_config
        test_config = load_config(evaluation["config"])
        from ..evaluation import evaluation_settings
        profile = evaluation_settings(test_config)
        test_ids = evaluation.get("datasets") or list(test_config["datasets"])
        if not test_ids or len(set(test_ids)) != len(test_ids) or set(test_ids) - set(test_config["datasets"]):
            raise ValueError("suite 包含未知或重复的评测数据集")
        tests = {key: test_config["datasets"][key] for key in test_ids}
    else:
        raise ValueError("suite.evaluation.config 必须指定统一的 ioutest 测试集配置")
    jobs = []
    for model in selected_models:
        for dataset in selected_data:
            job_id = safe_name(model + "--" + dataset)
            cfg = resolve(dict(
                model=model, model_config=path_value(c["models"][model], base),
                dataset_config=path_value(c["datasets"][dataset], base),
                train_config=path_value(c["train_config"], base),
                output_dir=str(Path(root) / "experiments" / job_id),
                device=c.get("device", "cuda:0"), seed=c.get("seed", 20261020), resume="auto"))
            cfg["wandb"].update(c.get("wandb", {}))
            tracks = evaluation.get("tracks", {}).get(model, ["text"])
            if (not tracks or len(set(tracks)) != len(tracks) or
                    set(tracks) - ({"text"} if model == "3dmood" else {"text", "gt-box"})):
                raise ValueError("模型不支持指定评测输入条件：" + model)
            jobs.append(dict(id=job_id, model=model, dataset=dataset, config=cfg, tracks=tracks))
    return dict(schema_version="finefune.suite.v1", root=root, jobs=jobs,
                tests=tests, evaluation_profile=profile, baseline_models=selected_models,
                order="prepare -> audit -> official baselines -> train -> finetuned evaluation -> report")


def run_training(cfg):
    from ioutest.model_environment import child_environment
    env = dict(child_environment(os.environ, cfg['model'].get('environment', {})), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    env.pop("FINEFUNE_DISPATCHED", None)
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + str(REPO)
    command = [cfg["model"]["python"], "-u", "-B", str(PROJECT / "run.py"), "train",
               "--config", str(Path(cfg["root"]) / "resolved_config.yaml"), "--resume", "auto"]
    logs = Path(cfg["root"]) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "worker.log").open("a") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                       env=env, cwd=REPO, check=True)
    status = read(Path(cfg["root"]) / "status.json")
    if status.get("stage") != "complete":
        raise RuntimeError("训练没有完整结束：" + cfg["root"])
    return status


def run(path, models=None, datasets=None):
    from ..cli import model_check, prepare_inputs
    from ..datasets import load_source
    from ..evaluation import evaluate
    from ioutest.experiment_data import audit_training, load_dataset
    planned = plan(path, models, datasets)
    root = Path(planned["root"])
    with lock(root / ".suite.lock"):
        bind(root / "suite_identity.json", planned)
        write(root / "suite_plan.json", planned)
        def status(stage, **values):
            write(root / "suite_status.json", dict(complete=False, stage=stage, **values))
        try:
            for model in planned["baseline_models"]:
                cfg = next(j["config"] for j in planned["jobs"] if j["model"] == model)
                status("check", model=model)
                check = model_check(cfg)
                if not check["ready"]:
                    raise RuntimeError("模型运行环境未就绪：" + json.dumps(check, ensure_ascii=False))
            cache = {}
            for job in planned["jobs"]:
                cfg = job["config"]
                status("prepare", experiment=job["id"])
                key = digest(cfg["dataset"])
                if key not in cache:
                    cache[key] = load_source(cfg["dataset"])
                prepare_inputs(cfg, cache[key])
            # Check every training split against every selected independent test
            # before spending PPU time; shared original/generated states collide.
            status("audit")
            test_data = {key: load_dataset(spec) for key, spec in planned["tests"].items()}
            for job in planned["jobs"]:
                cfg = job["config"]
                variant = dict(training_manifest=str(Path(cfg["root"]) / "train.jsonl"),
                               training_root=str(Path(cfg["root"]) / "dataset"))
                audit = {key: audit_training(variant, data) for key, data in test_data.items()}
                write(Path(cfg["root"]) / "training_test_audit.json", audit)
            for model in planned["baseline_models"]:
                job = next(j for j in planned["jobs"] if j["model"] == model)
                cfg = deepcopy(job["config"])
                cfg["root"] = str(root / "baselines" / model)
                prepare_inputs(cfg, cache[digest(cfg["dataset"])])
                status("baseline", model=model)
                evaluate(cfg, planned["tests"], job["tracks"], "official", evaluation_profile=planned['evaluation_profile'])
            for job in planned["jobs"]:
                status("training", experiment=job["id"])
                run_training(job["config"])
            for job in planned["jobs"]:
                status("finetuned", experiment=job["id"])
                evaluate(job["config"], planned["tests"], job["tracks"], "finetuned", evaluation_profile=planned['evaluation_profile'])
            result = report(root)
            if not result["complete"]:
                raise RuntimeError("实验阶段返回成功，但最终产物或评测报告不完整")
            write(root / "suite_status.json", dict(stage="complete", **result))
            return result
        except BaseException as exc:
            write(root / "suite_status.json", dict(stage="failed", complete=False, error=str(exc),
                  resume="保持配置，重新执行同一 suite 命令；已完成结果会校验并复用"))
            raise


def validated_metrics(path, expected):
    """A CSV alone is not evidence of complete, current evaluation results."""
    if not path.is_file():
        raise ValueError("缺少评测 CSV")
    result = read(path.parent / "results.json")
    if result.get("complete") is not True:
        raise ValueError("评测汇总不完整")
    def selection(value):
        return tuple(value[k] for k in ("dataset", "model_name", "variant_name", "track"))
    refs = result.get("jobs", [])
    if len(refs) != len(expected) or {selection(r["job"]) for r in refs} != expected:
        raise ValueError("评测汇总与计划的测试集、模型、权重或输入条件不一致")
    for ref in refs:
        p = Path(ref["metrics"]).resolve()
        if not p.is_relative_to(path.parent.parent.resolve()):
            raise ValueError("指标文件不属于当前评测目录")
        metric = read(p)
        recorded = metric.pop("checksum", None)
        if (not metric.get("complete") or not metric.get("request_id") or
                recorded != ref["checksum"] or digest(metric) != recorded):
            raise ValueError("指标损坏、不完整或来源摘要变化")
    with path.open() as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    if len(rows) != len(expected) or {selection(r) for r in rows} != expected:
        raise ValueError("CSV 中的任务集合与实验计划不一致")
    # Check the exported values against the validated evaluator summary too.
    comparison = result.get("comparison", [])
    def csv_value(v):
        return "" if v is None else str(v)
    if rows != [{k: csv_value(row.get(k)) for k in fields} for row in comparison]:
        raise ValueError("评测 CSV 与汇总值不一致")
    return rows


def report(directory):
    root = Path(directory).resolve()
    planned = read(root / "suite_plan.json")
    records, jobs, pending = [], [], []
    sources = []
    for model in planned["baseline_models"]:
        tracks = next(j["tracks"] for j in planned["jobs"] if j["model"] == model)
        sources.append(("official--" + model, model, "pretrained", "official", tracks,
                        root / "baselines" / model / "evaluation/official/reports/comparison.csv"))
    for job in planned["jobs"]:
        status_path = Path(job["config"]["root"]) / "status.json"
        state = read(status_path) if status_path.exists() else {"stage": "pending"}
        jobs.append(dict(id=job["id"], model=job["model"], dataset=job["dataset"],
                         stage=state.get("stage"), output=job["config"]["root"]))
        final = Path(job["config"]["root"]) / "checkpoints/final_head.pt"
        if (state.get("stage") != "complete" or not final.is_file() or
                (state.get("final_head_sha256") and checksum(final) != state["final_head_sha256"])):
            pending.append(job["id"] + ":training")
        sources.append((job["id"], job["model"], job["dataset"], "finetuned", job["tracks"],
                        Path(job["config"]["root"]) / "evaluation/finetuned/reports/comparison.csv"))
    for experiment, model, dataset, variant, tracks, path in sources:
        expected = {(test, model, variant, track) for test in planned["tests"] for track in tracks}
        try:
            rows = validated_metrics(path, expected)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            pending.append(experiment + ":evaluation:" + str(exc))
            continue
        for row in rows:
            records.append(dict(row, experiment_id=experiment, training_dataset=dataset,
                                model=model, variant=variant, source_report=str(path)))
    fields = list(dict.fromkeys(k for row in records for k in row)) or ["experiment_id"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    output = root / "reports"
    atomic_bytes(output / "comparison.csv", buffer.getvalue().encode())
    result = dict(complete=not pending, experiments=jobs, metrics=records, pending=pending,
                  report=str(output / "report.md"))
    write(output / "results.json", result)
    lines = ["# finefune 微调实验对比", "", "完整：" + str(not pending), "",
             "| 实验 | 模型 | 训练数据 | 训练状态 |", "|---|---|---|---|"]
    lines += [f'| {j["id"]} | {j["model"]} | {j["dataset"]} | {j["stage"]} |' for j in jobs]
    lines += ["", "全部指标及来源：[comparison.csv](comparison.csv)。",
              "空缺阶段不补零，不把部分结果标为完整实验。"]
    if pending:
        lines += ["", "待完成：", "", *["- " + p for p in pending]]
    atomic_bytes(output / "report.md", ("\n".join(lines) + "\n").encode())
    return result
