"""Explicit v1 configuration conversion, without rewriting historical weights."""
from pathlib import Path

from .common import atomic_bytes, read
from .config import normalize_dataset, path_value, training_options, SCHEMA


def convert_config(source, key, output, output_dir):
    import yaml
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError("转换结果必须写入新的配置文件")
    old = read(source)
    if key not in old.get("models", {}):
        raise ValueError("旧配置没有模型：" + key)
    model = dict(old["models"][key])
    python = path_value(model.pop("python"), source.parent, interpreter=True)
    for name in ("project_root", "checkpoint", "lingbot_checkpoint"):
        if model.get(name):
            model[name] = path_value(model[name], source.parent)
    data = Path(old["root"]) / "data" / old["dataset_name"]
    dataset = normalize_dataset(dict(
        schema_version="finefune.dataset.v1", id="legacy_hot3d",
        format="hot3d_prepared", root=str(data), protocol="protocol.json",
        train=dict(manifest="train.jsonl"), eval=dict(manifest="eval.jsonl"),
        group_by="participant_id"), source.parent)
    training, _ = training_options(old["training"])
    cfg = dict(schema_version=SCHEMA, resolved=True, model_id=key,
               model=dict(schema_version="finefune.model.v1", family=key,
                          python=python, options=model),
               dataset=dataset, training=training, root=str(Path(output_dir).resolve()),
               seed=old["seed"], device=old["device"],
               wandb=dict({"enabled": True, "project": "finefune"}, **old.get("wandb", {})))
    atomic_bytes(output, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode())
    return dict(config=str(output), source=str(source), dataset_root=str(data),
                checkpoints_modified=False,
                note="复用旧数据，从原始 checkpoint 开始新实验；旧 last.pt 不改写身份或伪装恢复")
