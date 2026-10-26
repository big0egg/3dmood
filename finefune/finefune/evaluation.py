"""Export immutable evaluation requests to the independent ioutest module."""
from copy import deepcopy
from pathlib import Path
import subprocess

from .common import REPO, checksum, data_path, read, write


def evaluation_settings(config):
    """Freeze evaluator settings independently of training recipes."""
    fields = ('metrics', 'timing', 'execution', 'environment')
    return dict({key: deepcopy(config[key]) for key in fields if key in config},
                model_environments={key: deepcopy(value.get('environment', {}))
                                    for key, value in config.get('models', {}).items()})


def head_metadata(path, python):
    """The data/suite controller need not install torch or model dependencies."""
    import importlib.util
    keys = ("model_id", "base_sha256", "epochs_completed", "identity", "benchmark_id")
    if importlib.util.find_spec("torch") is not None:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
        return {key: state.get(key) for key in keys}
    import json
    code = ("import torch,json,sys; d=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
            "print(json.dumps({k:d.get(k) for k in " + repr(keys) + "}))")
    process = subprocess.run([python, "-B", "-c", code, str(path)],
                             capture_output=True, text=True, check=True)
    return json.loads(process.stdout.splitlines()[-1])


def validate_selection(cfg, tracks=None, variant="finetuned", head_path=None):
    tracks = tracks or ["text"]
    key = cfg["model_id"]
    if key == "3dmood" and tracks != ["text"]:
        raise ValueError("3D-MOOD 仅支持 text 评测")
    if not tracks or len(set(tracks)) != len(tracks) or set(tracks) - {"text", "gt-box"}:
        raise ValueError("不支持的评测输入条件")
    if variant not in ("official", "finetuned"):
        raise ValueError("variant 必须为 official 或 finetuned")
    if head_path and variant != "finetuned":
        raise ValueError("--finetuned-ckpt 仅用于 finetuned 版本")
    return tracks


def export_config(cfg, tests=None, tracks=None, variant="finetuned", output=None, head_path=None, evaluation_profile=None):
    from ioutest.experiment_config import DEFAULT_CONFIG, load_config
    profile = deepcopy(evaluation_profile) if evaluation_profile is not None else evaluation_settings(load_config(DEFAULT_CONFIG))
    tracks = validate_selection(cfg, tracks, variant, head_path)
    key = cfg["model_id"]
    root = Path(cfg["root"])
    # Explicit tests use ioutest's prepared-dataset configuration vocabulary.
    if tests:
        datasets = deepcopy(tests)
    else:
        droot = data_path(cfg)
        datasets = {"experiment_eval": dict(format="hot3d_prepared", root=str(droot),
                     manifest=str(droot / "eval.jsonl"), protocol=str(droot / "protocol.json"))}
    spec = deepcopy(cfg["model"])
    spec['environment'] = dict(spec.get('environment', {}), **profile.get('model_environments', {}).get(key, {}))
    spec.pop("schema_version", None)
    spec["variants"] = {"official": {}}
    if variant == "finetuned":
        head = Path(head_path).expanduser().resolve() if head_path else root / "checkpoints/final_head.pt"
        if not head.is_file():
            raise FileNotFoundError("训练尚未产生最终三维头：" + str(head))
        state = head_metadata(head, cfg["model"]["python"])
        if state.get("base_sha256") != checksum(spec["options"]["checkpoint"]):
            raise ValueError("最终三维头与指定原始权重不匹配")
        if state.get("model_id") != key:
            raise ValueError("最终三维头与模型不匹配")
        protocol = read(data_path(cfg) / "protocol.json")
        if (state.get("benchmark_id") != protocol["benchmark_id"] or
                not state.get("identity") or state.get("epochs_completed", 0) < 1):
            raise ValueError("三维头训练来源与本实验不一致或训练未完成")
        status = root / "status.json"
        if status.exists() and read(status).get("stage") != "complete":
            raise ValueError("本实验训练尚未完整结束")
        spec["variants"]["finetuned"] = dict(
            head_checkpoint=str(head), training_manifest=str(root / "train.jsonl"),
            training_root=str(data_path(cfg)), epochs_completed=state["epochs_completed"],
            training_identity=state["identity"], training_benchmark_id=state["benchmark_id"])
    elif (root / "train.jsonl").exists():
        # Audit the planned finetuning split for baseline comparisons too.
        spec["variants"]["official"] = dict(
            training_manifest=str(root / "train.jsonl"), training_root=str(data_path(cfg)))
    result = dict(schema_version="ioutest.experiment.v1",
                  output=dict(root=str(root / "evaluation" / variant)),
                  selection=dict(models=[key], variants=[variant], datasets=list(datasets), tracks=tracks),
                  execution=dict(device=cfg["device"], seed=cfg["seed"], threads=cfg["training"]["threads"]),
                  timing=dict(warmup_frames=1), datasets=datasets, models={key: spec})
    for field in ('metrics', 'timing', 'execution', 'environment'):
        if field in profile:
            result[field] = deepcopy(profile[field])
    path = Path(output) if output else root / "evaluation" / (variant + "_config.json")
    write(path, result)
    # Apply ioutest's own strict config validation before launching processes.
    load_config(path)
    return path


def evaluate(cfg, tests=None, tracks=None, variant="finetuned", stage="all", head_path=None, evaluation_profile=None):
    path = export_config(cfg, tests, tracks, variant, head_path=head_path, evaluation_profile=evaluation_profile)
    python = cfg["model"]["python"]
    command = [python, "-B", "-m", "ioutest", "--config", str(path), "--stage", stage]
    print("评测：" + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    return dict(complete=True, config=str(path), root=str(Path(cfg["root"]) / "evaluation" / variant))
