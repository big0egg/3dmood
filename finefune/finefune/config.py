"""Strict model/dataset/recipe composition; explicit CLI overrides win."""
from copy import deepcopy
import math
import os
from pathlib import Path

from .common import PROJECT, outside_sources

SCHEMA = "finefune.experiment.v1"
DEFAULT_TRAINING = {
    "finetune_mode": "geometry_head", "precision": "fp32",
    "optimizer": "adamw", "scheduler": "cosine",
    "epochs": 4, "micro_batch": 1, "accumulate": 8, "lr": .0001,
    "weight_decay": .0001, "warmup_updates": 300, "clip_grad": .1,
    "checkpoint_updates": 250, "threads": 1,
}
ALIASES = {"micro_batch_size": "micro_batch", "grad_accum_steps": "accumulate",
           "max_grad_norm": "clip_grad", "checkpoint_every_updates": "checkpoint_updates"}
FORMATS = {"hot3d_prepared", "robot_synth_prepared", "robot_realism_prepared", "finefune_prepared"}


def document(path):
    import yaml
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"配置必须为对象：{path}")
    return value, path.parent


def path_value(value, base, interpreter=False):
    p = Path(os.path.expandvars(str(value))).expanduser()
    p = p if p.is_absolute() else Path(base) / p
    # A venv interpreter is often a symlink: never resolve it to system Python.
    return os.path.abspath(p) if interpreter else str(p.resolve())


def dataset_config(path, root_override=None):
    d, base = document(path)
    return normalize_dataset(d, base, root_override)


def normalize_dataset(d, base, root_override=None):
    d = deepcopy(d)
    if d.get("schema_version") != "finefune.dataset.v1":
        raise ValueError("数据配置 schema_version 必须为 finefune.dataset.v1")
    if d.get("format") not in FORMATS:
        raise ValueError("不支持的数据格式：" + str(d.get("format")))
    previous_root = Path(path_value(d["root"], base))
    d["root"] = path_value(root_override, Path.cwd()) if root_override else str(previous_root)
    def at_root(value):
        p = Path(value).expanduser()
        if root_override and p.is_absolute() and p.is_relative_to(previous_root):
            p = p.relative_to(previous_root)
        return path_value(p, d["root"])
    for name in ("protocol", "categories_file"):
        if d.get(name):
            d[name] = at_root(d[name])
    for split in ("train", "eval"):
        if not isinstance(d.get(split), dict) or not d[split].get("manifest"):
            raise ValueError(f"数据配置必须提供 {split}.manifest")
        d[split]["manifest"] = at_root(d[split]["manifest"])
        if d[split].get("align_to"):
            d[split]["align_to"] = path_value(d[split]["align_to"], d["root"])
    if not d.get("group_by"):
        d["group_by"] = "participant_id" if d["format"] == "hot3d_prepared" else "scene_id"
    if d["group_by"] not in ("participant_id", "sequence_id", "scene_id"):
        raise ValueError("group_by 必须为 participant_id、sequence_id 或 scene_id")
    if type(d.get("minimum_visible_pixels", 32)) is not int or d.get("minimum_visible_pixels", 32) < 0:
        raise ValueError("minimum_visible_pixels 必须为非负整数")
    if d.get("target_kinds", ["object"]) != ["object"]:
        raise ValueError("当前训练仅支持 target_kinds: [object]")
    if "training_epochs" in d and (type(d["training_epochs"]) is not int or d["training_epochs"] < 1):
        raise ValueError("数据配置 training_epochs 必须为正整数")
    return d


def training_options(value):
    value = deepcopy(value)
    seed = value.pop("seed", None)
    for public, internal in ALIASES.items():
        if public in value:
            if internal in value:
                raise ValueError(f"训练参数重复：{public} / {internal}")
            value[internal] = value.pop(public)
    unknown = set(value) - set(DEFAULT_TRAINING)
    if unknown:
        raise ValueError("未知训练参数：" + ", ".join(sorted(unknown)))
    t = dict(DEFAULT_TRAINING, **value)
    for key, supported in (("finetune_mode", "geometry_head"), ("precision", "fp32"),
                           ("optimizer", "adamw"), ("scheduler", "cosine")):
        if t[key] != supported:
            raise ValueError(f"当前仅支持 {key}={supported}")
    for name in ("epochs", "micro_batch", "accumulate", "checkpoint_updates", "threads"):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError(name + " 必须为正整数")
    if t["micro_batch"] != 1:
        raise ValueError("当前仅支持 micro_batch_size=1，请使用 grad_accum_steps")
    if type(t["warmup_updates"]) is not int or t["warmup_updates"] < 0:
        raise ValueError("warmup_updates 必须为非负整数")
    for name in ("lr", "weight_decay", "clip_grad"):
        v = t[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(name + " 必须为有限非负数")
    if t["lr"] <= 0 or t["clip_grad"] <= 0:
        raise ValueError("lr 和 max_grad_norm 必须大于0")
    return t, seed


def resolve(args):
    cli = vars(args) if not isinstance(args, dict) else args
    if cli.get("config"):
        c, base = document(cli["config"])
        if c.get("schema_version") != SCHEMA:
            raise ValueError("实验配置 schema_version 必须为 " + SCHEMA)
    else:
        c, base = {}, Path.cwd()
    inline_model = c.get("model") if isinstance(c.get("model"), dict) else None
    key = cli.get("model") or c.get("model_id") or (inline_model or {}).get("family") or c.get("model")
    if key not in ("3dmood", "wilddet3d"):
        raise ValueError("必须选择 --model 3dmood 或 wilddet3d")
    model_file = cli.get("model_config")
    if inline_model and not model_file:
        m, mbase = deepcopy(inline_model), base
    else:
        model_file = path_value(model_file, Path.cwd()) if model_file else path_value(
            c.get("model_config", PROJECT / "configs/models" / (key + ".yaml")), base)
        m, mbase = document(model_file)
    if m.get("schema_version") != "finefune.model.v1" or m.get("family") != key:
        raise ValueError("模型配置与 --model 不匹配")
    m["python"] = path_value(m["python"], mbase, interpreter=True)
    from ioutest.model_environment import normalize_environment
    m["environment"] = normalize_environment(m.get("environment", {}), mbase)
    options = dict(m.get("options", {}))
    for name in ("project_root", "checkpoint", "lingbot_checkpoint"):
        if options.get(name):
            options[name] = path_value(options[name], mbase)
    origin = cli.get("origin_ckpt") or c.get("origin_ckpt")
    if origin:
        options["checkpoint"] = path_value(origin, Path.cwd() if cli.get("origin_ckpt") else base)
    if not options.get("checkpoint"):
        raise ValueError("必须提供 --origin-ckpt 或模型配置中的 checkpoint")
    if key == "wilddet3d" and not options.get("lingbot_checkpoint"):
        raise ValueError("WildDet3D 模型配置必须提供 lingbot_checkpoint")
    dfile = cli.get("dataset_config") or c.get("dataset_config")
    if not dfile and not c.get("dataset"):
        raise ValueError("必须提供 --dataset-config")
    if dfile:
        dfile = path_value(dfile, Path.cwd() if cli.get("dataset_config") else base)
        d = dataset_config(dfile, cli.get("dataset_root"))
    else:
        d = normalize_dataset(c["dataset"], base, cli.get("dataset_root"))
    if c.get("resolved") and not cli.get("train_config"):
        t, recipe_seed = training_options(c["training"])
    else:
        tfile = cli.get("train_config") or c.get("train_config", PROJECT / "configs/training/geometry_head.yaml")
        tfile = path_value(tfile, Path.cwd() if cli.get("train_config") else base)
        t, recipe_seed = training_options(document(tfile)[0])
        # Dataset defaults apply to new recipes; saved experiment epochs stay frozen.
        t["epochs"] = d.get("training_epochs", t["epochs"])
    overrides = {} if cli.get("train_config") else dict(c.get("training", {}))
    for name in ("epochs", "lr", "grad_accum_steps"):
        if cli.get(name) is not None:
            overrides[name] = cli[name]
    # Normalize each overlay before merging aliases into the normalized recipe.
    for public, internal in ALIASES.items():
        if public in overrides:
            overrides[internal] = overrides.pop(public)
    t, _ = training_options(dict(t, **overrides))
    output = cli.get("output_dir") or c.get("output_dir") or c.get("root")
    if not output:
        raise ValueError("必须提供 --output-dir")
    output = path_value(output, Path.cwd() if cli.get("output_dir") else base)
    outside_sources(output, [d["root"]])
    seed = cli.get("seed") if cli.get("seed") is not None else c.get("seed", recipe_seed if recipe_seed is not None else 20261020)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed 必须为 0～2^32-1 的整数")
    device = cli.get("device") or c.get("device", "cuda:0")
    if not (device == "cpu" or device == "cuda" or (isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("device 必须为 cpu、cuda 或 cuda:N")
    resume = cli.get("resume") if cli.get("resume") is not None else c.get("resume")
    if resume and resume != "auto":
        resume = path_value(resume, Path.cwd() if cli.get("resume") else base)
    return dict(schema_version=SCHEMA, resolved=True, model_id=key, model=m | {"options": options},
                models={key: dict(options, python=m["python"])}, dataset=d,
                training=t, root=output, seed=seed, device=device, resume=resume,
                max_detections=options.get("max_detections", 300),
                wandb=dict({"enabled": True, "project": "finefune"}, **c.get("wandb", {})))


def public_config(cfg):
    return deepcopy(cfg)
