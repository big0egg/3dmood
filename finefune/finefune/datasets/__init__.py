"""Prepared-data adapters shared with the independent ioutest evaluator."""
from copy import deepcopy
from pathlib import Path
import math

from ..common import (bind, checksum, data_path, digest, jsonlines,
                      outside_sources, read, readlines, write)


def _split(spec, split):
    from ioutest.experiment_data import load_dataset
    options = deepcopy(spec)
    options.update(manifest=spec[split]["manifest"], eval_split=split)
    if spec["format"] in ("hot3d_prepared", "finefune_prepared"):
        options["format"] = "hot3d_prepared"
        protocol = read(spec["protocol"])
        expected = protocol.get(split + "_sha256")
        if expected:
            rows = readlines(options["manifest"])
            if expected not in (checksum(options["manifest"]), digest(rows)):
                raise ValueError(f"{split} 冻结清单摘要变化")
    data = load_dataset(options)
    frames = data["frames"]
    original_count = len(frames)
    if spec[split].get("align_to"):
        rows = readlines(spec[split]["align_to"])
        selected = [str(r["state_id"]) for r in rows]
        if len(selected) != len(set(selected)):
            raise ValueError("对齐清单包含重复 state_id")
        available = {str(f["state_id"]) for f in frames if f.get("state_id")}
        if not set(selected) <= available:
            raise ValueError("对齐清单包含源数据中不存在的状态")
        frames = [f for f in frames if str(f.get("state_id")) in set(selected)]
    empty = []
    if split == "train":
        kept = []
        for frame in frames:
            valid = [b for b in data["gt"][frame["frame_id"]] if not b["ignore"]]
            if not valid:
                empty.append(frame["frame_id"])
                continue
            for box in valid:
                r = box.get("projected_bbox")
                if (r is None or len(r) != 4 or
                        not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in r) or
                        r[2] <= r[0] or r[3] <= r[1]):
                    raise ValueError("有效训练目标缺少合法二维投影框：" + frame["frame_id"])
            kept.append(frame)
        frames = kept
    if not frames:
        raise ValueError(f"{split} 没有可用样本")
    gt = {f["frame_id"]: data["gt"][f["frame_id"]] for f in frames}
    identity = dict(classes=data["classes"], frames=frames, gt=gt,
                    label_status=data["metadata"].get("label_status"))
    return dict(classes=data["classes"], frames=frames, gt=gt,
                dataset_id=digest(identity), metadata=data["metadata"],
                audit=dict(source_frames=original_count, selected_frames=len(frames),
                           excluded_empty_training_frames=empty,
                           aligned_to=spec[split].get("align_to")))


def load_source(spec):
    train, evaluation = _split(spec, "train"), _split(spec, "eval")
    if train["classes"] != evaluation["classes"]:
        raise ValueError("训练和评测必须使用同一冻结类别表")
    overlaps = {}
    for field in dict.fromkeys(("frame_id", "state_id", "rgb_sha256", spec["group_by"])):
        a = {str(f[field]) for f in train["frames"] if f.get(field) is not None}
        b = {str(f[field]) for f in evaluation["frames"] if f.get(field) is not None}
        if field == spec["group_by"] and (
            any(f.get(field) is None for f in train["frames"] + evaluation["frames"])
        ):
            raise ValueError("数据缺少划分隔离字段：" + field)
        if a & b:
            overlaps[field] = sorted(a & b)[:10]
    if overlaps:
        raise ValueError("训练／评测来源重叠：" + str(overlaps))
    return dict(train=train, evaluation=evaluation, classes=train["classes"],
                group_by=spec["group_by"],
                dataset_id=digest([train["dataset_id"], evaluation["dataset_id"]]))


def summary(data):
    return dict(train_frames=len(data["train"]["frames"]),
                eval_frames=len(data["evaluation"]["frames"]),
                classes=len(data["classes"]), dataset_id=data["dataset_id"],
                group_by=data["group_by"],
                train=data["train"]["audit"], evaluation=data["evaluation"]["audit"])


def prepare(cfg, data=None):
    """Write only indexes/normalized GT; original RGB and GT stay untouched."""
    data = data or load_source(cfg["dataset"])
    target = data_path(cfg)
    outside_sources(cfg["root"], [cfg["dataset"]["root"]])
    if target.exists() and any(target.iterdir()) and not (target / "preparation.json").exists():
        raise ValueError("数据索引目录已有非本工具产物：" + str(target))
    bind(target / "preparation.json", dict(schema="finefune.dataset.v1",
         dataset_id=data["dataset_id"], specification=cfg["dataset"]))
    manifests = {}
    for split, source in (("train", data["train"]), ("eval", data["evaluation"])):
        rows = []
        for frame in source["frames"]:
            token = digest([split, frame["frame_id"]])[:32]
            boxes = source["gt"][frame["frame_id"]]
            gt_path = target / "groundtruth" / split / (token + ".json")
            if gt_path.exists() and read(gt_path) != boxes:
                raise ValueError("已准备 GT 内容发生变化：" + str(gt_path))
            write(gt_path, boxes)
            row = dict(frame, gt_path=str(gt_path), gt_sha256=digest(boxes),
                       split=split, clip_id=str(frame.get("scene_id", frame.get("sequence_id", split))),
                       frame_key=token)
            rows.append(row)
        manifests[split] = rows
        path = target / (split + ".jsonl")
        if path.exists() and readlines(path) != rows:
            raise ValueError("已准备清单发生变化：" + str(path))
        jsonlines(path, rows)
    protocol = dict(schema_version="finefune.dataset.v1", classes=data["classes"],
                    benchmark_id=data["train"]["dataset_id"],
                    dataset_id=data["dataset_id"],
                    train_dataset_id=data["train"]["dataset_id"],
                    eval_dataset_id=data["evaluation"]["dataset_id"],
                    train_sha256=digest(manifests["train"]),
                    eval_sha256=digest(manifests["eval"]),
                    label_status=data["evaluation"]["metadata"].get("label_status"))
    write(target / "protocol.json", protocol)
    write(target / "dataset_audit.json", summary(data))
    import yaml
    from ..common import atomic_bytes
    prepared_spec = dict(schema_version="finefune.dataset.v1", id=cfg["dataset"].get("id", "prepared"),
                         format="finefune_prepared", root=str(target), protocol="protocol.json",
                         train=dict(manifest="train.jsonl"), eval=dict(manifest="eval.jsonl"),
                         group_by=cfg["dataset"]["group_by"])
    if "training_epochs" in cfg["dataset"]:
        prepared_spec["training_epochs"] = cfg["dataset"]["training_epochs"]
    atomic_bytes(Path(cfg["root"]) / "dataset_config.yaml",
                 yaml.safe_dump(prepared_spec, allow_unicode=True, sort_keys=False).encode())
    return protocol, manifests["train"], manifests["eval"]


def load(cfg):
    root = data_path(cfg)
    protocol = read(root / "protocol.json")
    train, evaluation = readlines(root / "train.jsonl"), readlines(root / "eval.jsonl")
    if digest(train) != protocol["train_sha256"] or digest(evaluation) != protocol["eval_sha256"]:
        raise ValueError("规范化训练／评测清单变化")
    return protocol, train, evaluation


def verify(cfg, images=False):
    from ioutest.hot3d.common import validate_box
    from ioutest.experiment_data import read_rgb
    protocol, train, evaluation = load(cfg)
    for frame in train + evaluation:
        boxes = read(frame["gt_path"])
        if digest(boxes) != frame["gt_sha256"]:
            raise ValueError("GT 摘要变化：" + frame["frame_id"])
        for box in boxes:
            validate_box(box, protocol["classes"])
        if checksum(frame["rgb_path"]) != frame["rgb_sha256"]:
            raise ValueError("图片摘要变化：" + frame["frame_id"])
        if images:
            read_rgb(frame)
    return dict(complete=True, train_frames=len(train), eval_frames=len(evaluation),
                dataset_id=protocol["dataset_id"])
