"""Contracts for prepared data, configuration precedence and durable training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from finefune.common import checksum, digest, jsonlines, read, write
from finefune.config import dataset_config, document, resolve
from finefune.datasets import load_source, prepare, verify


def image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), color, np.uint8)).save(path)


def box():
    return dict(uid="one", class_id=0, center=[0., 0., 2.], size=[.2, .3, .4],
                rotation=np.eye(3).tolist(), projected_bbox=[10., 10., 22., 22.], ignore=False)


def make_hot3d(root, train_count=5):
    rows = {"train": [], "eval": []}
    for split, count, person in (("train", train_count, "P1"), ("eval", 1, "P2")):
        for n in range(count):
            rgb = root / split / f"{n}.png"
            image(rgb, n + (10 if split == "train" else 100))
            gt = root / split / f"{n}.json"
            write(gt, [box()])
            rows[split].append(dict(frame_id=f"{split}{n}", participant_id=person,
                sequence_id=person + "_seq", rgb_path=str(rgb), rgb_sha256=checksum(rgb),
                image_hw=[32, 32], intrinsics=[[30., 0., 16.], [0., 30., 16.], [0., 0., 1.]],
                gt_path=str(gt), gt_sha256=digest([box()])))
        jsonlines(root / (split + ".jsonl"), rows[split])
    write(root / "protocol.json", dict(classes=[dict(class_id=0, name="cup", prompt="cup")],
          train_sha256=digest(rows["train"]), eval_sha256=digest(rows["eval"]), benchmark_id="original"))
    return dict(schema_version="finefune.dataset.v1", id="tiny_hot3d", format="hot3d_prepared",
                root=str(root), protocol="protocol.json", train=dict(manifest="train.jsonl"),
                eval=dict(manifest="eval.jsonl"), group_by="participant_id")


def make_simulation(root, realistic):
    accepted = {}
    for split, scene, color in (("train", "TRAIN_SCENE", 10), ("eval", "EVAL_SCENE", 100)):
        state = split + "_state"
        folder = root / split / scene / state / "agentview"
        image(folder / "rgb.png", color)
        provenance = dict(state_id=state, scene_id=scene, split=split)
        target = dict(instance_id=1, kind="object", parent_instance_id=None, category="cup",
                      center_camera=[0., 0., 2.], size_xyz=[.2, .3, .4], R_camera_from_box=np.eye(3).tolist(),
                      bbox_projected_clipped_xyxy=[10., 10., 22., 22.], visible=True, visible_pixels=50)
        child = dict(target, instance_id=2, parent_instance_id=1)
        write(folder / "annotation.json", dict(length_unit="meter", camera_coordinates="opencv",
              camera="agentview", image_hw=[32, 32], intrinsics=[[30., 0., 16.], [0., 30., 16.], [0., 0., 1.]],
              provenance=provenance, objects=[target, child], gt_status="inherited_requires_review"))
        row = dict(provenance, camera="agentview", rgb_path=str(folder / "rgb.png"),
                   annotation_path=str(folder / "annotation.json"))
        jsonlines(root / (split + ".jsonl"), [row])
        accepted[state] = provenance
    write(root / "manifest.json", dict(accepted=accepted))
    write(root / "categories.json", dict(categories=[dict(name="cup", kinds=["object"])]))
    return dict(schema_version="finefune.dataset.v1", id="simulation",
                format="robot_realism_prepared" if realistic else "robot_synth_prepared", root=str(root),
                categories_file="categories.json", camera="agentview", group_by="scene_id",
                train=dict(manifest="train.jsonl" if realistic else "manifest.json"),
                eval=dict(manifest="eval.jsonl" if realistic else "manifest.json"))


@pytest.fixture
def configured(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    dataset = make_hot3d(tmp_path / "source")
    dataset_file = directory / "dataset.yaml"
    dataset_file.write_text(yaml.safe_dump(dataset))
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"origin")
    model = dict(schema_version="finefune.model.v1", family="3dmood", python=sys.executable,
                 options=dict(project_root=str(tmp_path), checkpoint=str(checkpoint)))
    model_file = directory / "model.yaml"
    model_file.write_text(yaml.safe_dump(model))
    recipe_file = directory / "recipe.yaml"
    recipe_file.write_text(yaml.safe_dump(dict(epochs=2, grad_accum_steps=2, checkpoint_every_updates=2,
                                              warmup_updates=1, seed=123)))
    args = dict(model="3dmood", model_config=model_file, dataset_config=dataset_file,
                train_config=recipe_file, output_dir=tmp_path / "experiment", device="cpu")
    cfg = resolve(args)
    cfg["wandb"]["enabled"] = False
    return cfg, args, directory


@pytest.mark.parametrize("kind", ["hot3d", "libero", "cosmos"])
def test_three_adapters_are_read_only_and_do_not_require_33_objects(tmp_path, kind):
    source = tmp_path / "source"
    raw = make_hot3d(source) if kind == "hot3d" else make_simulation(source, kind == "cosmos")
    config = tmp_path / "dataset.yaml"
    config.write_text(yaml.safe_dump(raw))
    spec = dataset_config(config)
    before = {str(p): checksum(p) for p in source.rglob("*") if p.is_file()}
    data = load_source(spec)
    assert len(data["classes"]) == 1
    if kind != "hot3d":
        assert len(next(iter(data["train"]["gt"].values()))) == 1
        assert data["train"]["metadata"]["label_status"] == ["inherited_requires_review"]
    cfg = dict(root=str(tmp_path / "out"), dataset=spec)
    prepare(cfg, data)
    assert verify(cfg, images=True)["complete"]
    assert before == {str(p): checksum(p) for p in source.rglob("*") if p.is_file()}
    p = next((tmp_path / "out/dataset/groundtruth/train").glob("*.json"))
    write(p, [])
    with pytest.raises(ValueError, match="GT"):
        verify(cfg)


def test_cli_precedence_and_self_contained_snapshot(configured, tmp_path):
    cfg, args, directory = configured
    experiment = directory / "experiment.yaml"
    experiment.write_text(yaml.safe_dump(dict(schema_version="finefune.experiment.v1", model="3dmood",
        model_config="model.yaml", dataset_config="dataset.yaml", train_config="recipe.yaml",
        output_dir="../run", training=dict(epochs=3, lr=.001), wandb=dict(enabled=False))))
    result = resolve(dict(config=experiment, epochs=5, lr=.002))
    assert result["training"]["epochs"] == 5 and result["training"]["lr"] == .002
    assert result["wandb"]["enabled"] is False
    snapshot = tmp_path / "resolved.yaml"
    snapshot.write_text(yaml.safe_dump(result))
    assert resolve(dict(config=snapshot)) == result
    # A new recipe selected explicitly must override the snapshot recipe.
    new = directory / "new.yaml"
    new.write_text("epochs: 7\nlr: 0.005\n")
    updated = resolve(dict(config=snapshot, train_config=new))
    assert updated["training"]["epochs"] == 7 and updated["training"]["lr"] == .005


def test_python_symlink_is_not_resolved_into_base_environment(configured):
    cfg, args, directory = configured
    fake = directory / "venv/bin/python"
    fake.parent.mkdir(parents=True)
    fake.symlink_to(sys.executable)
    model = yaml.safe_load(Path(args["model_config"]).read_text())
    model["python"] = str(fake)
    Path(args["model_config"]).write_text(yaml.safe_dump(model))
    assert resolve(args)["model"]["python"] == str(fake)


def test_alignment_and_cross_appearance_leakage(tmp_path):
    raw = make_simulation(tmp_path / "source", True)
    source = Path(raw["root"])
    alignment = tmp_path / "aligned.jsonl"
    jsonlines(alignment, [dict(state_id="train_state")])
    raw["train"]["align_to"] = str(alignment)
    f = tmp_path / "d.yaml"
    f.write_text(yaml.safe_dump(raw))
    assert len(load_source(dataset_config(f))["train"]["frames"]) == 1
    # Different RGB, split, and scene, but the same original simulation state.
    row = json.loads((source / "eval.jsonl").read_text())
    ann = read(row["annotation_path"])
    ann["provenance"]["state_id"] = row["state_id"] = "train_state"
    write(row["annotation_path"], ann)
    jsonlines(source / "eval.jsonl", [row])
    with pytest.raises(ValueError, match="来源重叠"):
        load_source(dataset_config(f))


def test_invalid_recipe_and_unknown_options_are_rejected(configured):
    _, args, directory = configured
    f = directory / "bad.yaml"
    for text in ["precision: bf16", "micro_batch_size: 2", "epochs: 0", "lr: .nan", "unimplemented: true"]:
        f.write_text(text)
        with pytest.raises(ValueError):
            resolve(dict(args, train_config=f))


def test_wrong_resume_settings_do_not_overwrite_saved_configuration(configured):
    from finefune.cli import prepare_inputs
    cfg, _, _ = configured
    prepare_inputs(cfg)
    snapshot = (Path(cfg["root"]) / "resolved_config.yaml").read_bytes()
    other = deepcopy(cfg)
    other["training"]["lr"] *= 10
    with pytest.raises(ValueError, match="新实验目录"):
        prepare_inputs(other)
    assert snapshot == (Path(cfg["root"]) / "resolved_config.yaml").read_bytes()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.bbox3d_head = torch.nn.Linear(2, 1)


def test_new_engine_resume_replays_exact_optimizer_boundary(configured, tmp_path, monkeypatch):
    from finefune.training import engine
    from finefune.models.native import freeze
    from finefune.cli import prepare_inputs
    cfg, _, _ = configured
    source_data = load_source(cfg["dataset"])
    state = {"calls": 0, "interrupt_at": None}
    class Trainer:
        def __init__(self, adapter, model, classes):
            self.model = model
            freeze(model)
        def __call__(self, frame, image):
            state["calls"] += 1
            if state["calls"] == state["interrupt_at"]:
                torch.randn(20)
                raise RuntimeError("interrupted mid accumulation")
            value = torch.tensor([float(frame["frame_id"][-1]), 1.]) + torch.randn(2) * .01
            return {"loss": (self.model.bbox3d_head(self.model.backbone(value)) - .5).square().sum()}
    monkeypatch.setattr(engine, "MoodTrainer", Trainer)
    monkeypatch.setattr(engine, "training_image", lambda *a: None)
    monkeypatch.setattr(engine, "load_adapter", lambda c, k, d: (
        SimpleNamespace(close=lambda: None), Tiny(), dict(checkpoint_sha256=checksum(c["model"]["options"]["checkpoint"])) ))
    prepare_inputs(cfg, source_data)
    engine.train(cfg)
    reference = torch.load(Path(cfg["root"]) / "checkpoints/final_head.pt", weights_only=False)
    cfg["root"] = str(tmp_path / "interrupted")
    prepare_inputs(cfg, source_data)
    state.update(calls=0, interrupt_at=8)
    with pytest.raises(RuntimeError, match="mid accumulation"):
        engine.train(cfg)
    checkpoint = torch.load(Path(cfg["root"]) / "checkpoints/last.pt", weights_only=False)
    assert (checkpoint["epoch"], checkpoint["next_index"], checkpoint["update"]) == (1, 2, 4)
    state.update(calls=0, interrupt_at=None)
    cfg["resume"] = "auto"
    engine.train(cfg)
    resumed = torch.load(Path(cfg["root"]) / "checkpoints/final_head.pt", weights_only=False)
    assert all(torch.equal(v, resumed["head"][k]) for k, v in reference["head"].items())
    assert engine.train(cfg)["reused"]
    assert json.loads((Path(cfg["root"]) / "status.json").read_text())["complete"]


def test_dry_run_has_no_outputs_or_device_calls(configured, monkeypatch):
    from finefune.cli import main
    cfg, args, _ = configured
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("dry-run checked GPU"))
    argv = ["train", "--dry-run"]
    for k, v in args.items():
        argv.extend(["--" + k.replace("_", "-"), str(v)])
    assert main(argv) == 0
    assert not Path(cfg["root"]).exists()


def test_complete_head_exports_to_ioutest_and_wrong_base_is_rejected(configured):
    from finefune.cli import prepare_inputs
    from finefune.evaluation import export_config
    from ioutest.checkpoints import load_head
    from ioutest.experiment_config import load_config, select
    cfg, _, _ = configured
    model = Tiny()
    base = Path(cfg["model"]["options"]["checkpoint"])
    torch.save(model.state_dict(), base)
    prepare_inputs(cfg)
    final = Path(cfg["root"]) / "checkpoints/final_head.pt"
    final.parent.mkdir()
    torch.save(dict(model_id="3dmood", base_sha256=checksum(base), head=model.bbox3d_head.state_dict(),
                    identity="trained", benchmark_id=read(Path(cfg["root"])/"dataset/protocol.json")["benchmark_id"],
                    epochs_completed=2), final)
    path = export_config(cfg)
    exported = load_config(path)
    assert select(exported) == [dict(dataset="experiment_eval", model="3dmood", variant="finetuned", track="text")]
    variant = exported["models"]["3dmood"]["variants"]["finetuned"]
    info = load_head(SimpleNamespace(model=model, checkpoint=base), "3dmood", variant,
                     dict(checkpoint_sha256=checksum(base)))
    assert info["training_identity"] == "trained"
    base.write_bytes(b"wrong base")
    with pytest.raises(ValueError, match="原始权重"):
        export_config(cfg)


def suite_fixture(configured, tmp_path):
    cfg, args, directory = configured
    tests = dict(schema_version="ioutest.experiment.v1", output=dict(root=str(tmp_path / "unused_eval")),
                 datasets={"heldout": dict(format="hot3d_prepared", root=cfg["dataset"]["root"],
                     manifest=cfg["dataset"]["eval"]["manifest"], protocol=cfg["dataset"]["protocol"])},
                 models={"3dmood": dict(family="3dmood", python=sys.executable,
                     options=cfg["model"]["options"], variants={"official": {}})})
    tests_file = directory / "tests.json"
    write(tests_file, tests)
    suite = dict(schema_version="finefune.suite.v1", output_dir=str(tmp_path / "suite"),
                 models={"3dmood": str(args["model_config"])},
                 datasets={"first": str(args["dataset_config"]), "second": str(args["dataset_config"])},
                 train_config=str(args["train_config"]), device="cpu", wandb=dict(enabled=False),
                 evaluation=dict(config=str(tests_file), datasets=["heldout"], tracks={"3dmood": ["text"]}))
    path = directory / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    return path


def fake_evaluation(cfg, tests, tracks, variant):
    """A complete ioutest report contract, without pretending to run a model."""
    import csv
    root = Path(cfg["root"]) / "evaluation" / variant
    refs, comparison = [], []
    for key in tests:
        for track in tracks:
            job = dict(dataset=key, model_name=cfg["model_id"], variant_name=variant, track=track)
            metric = dict(complete=True, request_id=digest(job))
            metric["checksum"] = digest(metric)
            p = root / "runs" / key / track / "metrics.json"
            write(p, metric)
            refs.append(dict(job=job, metrics=str(p), checksum=metric["checksum"]))
            comparison.append(dict(job, **{"AP3D@0.25": .1}))
    write(root / "reports/results.json", dict(complete=True, jobs=refs, comparison=comparison))
    with (root / "reports/comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    return dict(complete=True)


def test_suite_orders_all_baselines_before_training_and_validates_reports(configured, tmp_path, monkeypatch):
    from finefune import cli, evaluation
    from finefune.experiments import runner
    path = suite_fixture(configured, tmp_path)
    events = []
    monkeypatch.setattr(cli, "model_check", lambda c: dict(ready=True))
    def evaluate(cfg, tests, tracks, variant, evaluation_profile=None):
        assert evaluation_profile is not None and 'metrics' in evaluation_profile
        events.append(variant)
        return fake_evaluation(cfg, tests, tracks, variant)
    def train(cfg):
        events.append("train")
        p = Path(cfg["root"]) / "checkpoints/final_head.pt"
        p.parent.mkdir(parents=True); p.write_bytes(b"test-artifact")
        write(Path(cfg["root"]) / "status.json", dict(stage="complete", final_head_sha256=checksum(p)))
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setattr(runner, "run_training", train)
    result = runner.run(path)
    assert events == ["official", "train", "train", "finetuned", "finetuned"]
    assert result["complete"] and len(result["metrics"]) == 3
    root = Path(tmp_path / "suite")
    p = root / "experiments/3dmood--first/evaluation/finetuned/reports/comparison.csv"
    p.write_text(p.read_text().replace("0.1", "0.9"))
    partial = runner.report(root)
    assert not partial["complete"]
    assert any("CSV" in value for value in partial["pending"])


def test_failed_baseline_never_starts_training(configured, tmp_path, monkeypatch):
    from finefune import cli, evaluation
    from finefune.experiments import runner
    path = suite_fixture(configured, tmp_path)
    monkeypatch.setattr(cli, "model_check", lambda c: dict(ready=True))
    monkeypatch.setattr(runner, "run_training", lambda c: pytest.fail("training after failed baseline"))
    def fail(*args, **kwargs):
        raise RuntimeError("baseline failed")
    monkeypatch.setattr(evaluation, "evaluate", fail)
    with pytest.raises(RuntimeError, match="baseline failed"):
        runner.run(path)
    assert read(tmp_path / "suite/suite_status.json")["stage"] == "failed"


def test_prepared_cosmos_keeps_annotation_status_and_can_be_reused(tmp_path):
    raw = make_simulation(tmp_path / "source", True)
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = dict(root=str(tmp_path / "indexes"), dataset=dataset_config(path))
    prepare(cfg)
    normalized = dataset_config(tmp_path / "indexes/dataset_config.yaml")
    assert normalized["format"] == "finefune_prepared"
    data = load_source(normalized)
    assert data["evaluation"]["metadata"]["label_status"] == ["inherited_requires_review"]


def test_convert_old_config_reuses_data_without_touching_original(configured, tmp_path):
    from finefune.compatibility import convert_config
    cfg, _, _ = configured
    old = dict(root="/existing/artifacts", dataset_name="prepared", models=cfg["models"],
               training=cfg["training"], seed=123, device="cpu")
    source, dest = tmp_path / "old.json", tmp_path / "converted.yaml"
    write(source, old)
    before = source.read_bytes()
    result = convert_config(source, "3dmood", dest, tmp_path / "new_experiment")
    assert not result["checkpoints_modified"]
    assert source.read_bytes() == before
    converted = resolve(dict(config=dest))
    assert converted["dataset"]["root"] == "/existing/artifacts/data/prepared"


def test_unsupported_gt_box_evaluation_is_rejected(configured):
    from finefune.evaluation import export_config
    cfg, _, _ = configured
    with pytest.raises(ValueError, match="仅支持 text"):
        export_config(cfg, tracks=["gt-box"], variant="official")


def test_evaluation_profile_preserves_changed_metrics_timing_and_cache(configured, tmp_path):
    from finefune.evaluation import evaluation_settings, export_config
    from ioutest.experiment_config import load_config
    cfg, _, _ = configured
    profile = dict(metrics=dict(iou_thresholds=[.3,.6],center_thresholds=[.01,.03],association_iou=.7,score_floor=.2),
                   timing=dict(warmup_frames=7), execution=dict(device='cpu',seed=987,threads=2),
                   environment=dict(TORCH_HOME=str(tmp_path/'torch-cache')),
                   models={'3dmood':dict(environment=dict(HF_HOME=str(tmp_path/'bert-cache')))})
    settings = evaluation_settings(profile)
    path = export_config(cfg, variant='official', evaluation_profile=settings)
    actual = load_config(path)
    assert actual['metrics'] == profile['metrics']
    assert actual['timing'] == profile['timing'] and actual['execution'] == profile['execution']
    assert actual['models']['3dmood']['environment']['HF_HOME'] == str(tmp_path/'bert-cache')
    assert actual['models']['3dmood']['options']['checkpoint'] == cfg['model']['options']['checkpoint']
    assert profile['timing']['warmup_frames'] == 7


def test_suite_carries_evaluator_profile(configured, tmp_path):
    from finefune.experiments.runner import plan
    path = suite_fixture(configured, tmp_path)
    suite = yaml.safe_load(path.read_text()); tests_path = Path(suite['evaluation']['config'])
    test_config = read(tests_path)
    test_config['metrics'] = dict(iou_thresholds=[.35,.65],center_thresholds=[.03],association_iou=.6,score_floor=.1)
    test_config['timing'] = dict(warmup_frames=3)
    write(tests_path, test_config)
    planned = plan(path)
    assert planned['evaluation_profile']['metrics'] == test_config['metrics']
    assert planned['evaluation_profile']['timing'] == test_config['timing']


def test_command_device_seed_override_evaluation_profile(configured, tmp_path):
    from finefune.cli import select_evaluation_profile
    path = suite_fixture(configured, tmp_path)
    suite = yaml.safe_load(path.read_text())
    args = SimpleNamespace(tests_config=Path(suite['evaluation']['config']), device='cpu', seed=345)
    settings = select_evaluation_profile(args)
    assert settings['execution']['device'] == 'cpu' and settings['execution']['seed'] == 345


def test_repository_suite_has_eight_independent_training_branches():
    from finefune.experiments.runner import plan
    root = Path(__file__).parents[1]
    result = plan(root/'configs/suites/model_data_compare.yaml')
    assert len(result['jobs']) == 8 and len(result['baseline_models']) == 2
    assert {j['dataset'] for j in result['jobs']} == {'hot3d','libero','cosmos_generic','cosmos_mixed'}
    for job in result['jobs']:
        m = job['config']['model']
        if job['model'] == 'wilddet3d':
            assert m['options']['checkpoint'].startswith('/root/wilddet3d-weights/')
            assert m['options']['lingbot_checkpoint'].startswith('/root/wilddet3d-weights/')
        else:
            assert m['environment']['HF_HOME'] == '/mnt/workspace/.cache/3d-mood/huggingface'


def test_unavailable_model_runtime_creates_no_experiment(configured, monkeypatch):
    from finefune import cli
    cfg, args, _ = configured
    monkeypatch.setattr(cli, "dispatch", lambda *a: None)
    monkeypatch.setattr(cli, "model_check", lambda *a: dict(ready=False, error="PPU missing"))
    argv = ["train"]
    for key, value in args.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    assert cli.main(argv) == 1
    assert not Path(cfg["root"]).exists()


def test_stopping_at_one_update_is_repeatable_without_extra_steps(configured, monkeypatch):
    from finefune.cli import prepare_inputs
    from finefune.training import engine
    from finefune.models.native import freeze
    cfg, _, _ = configured
    prepare_inputs(cfg)
    calls = []
    class Trainer:
        def __init__(self, adapter, model, classes):
            self.model = model
            freeze(model)
        def __call__(self, frame, image):
            calls.append(frame["frame_id"])
            return dict(loss=self.model.bbox3d_head(self.model.backbone(torch.ones(2))).square().sum())
    monkeypatch.setattr(engine, "MoodTrainer", Trainer)
    monkeypatch.setattr(engine, "training_image", lambda *a: None)
    monkeypatch.setattr(engine, "load_adapter", lambda c, k, d: (
        SimpleNamespace(close=lambda: None), Tiny(), dict(checkpoint_sha256=checksum(c["model"]["options"]["checkpoint"]))))
    assert engine.train(cfg, stop_after_updates=1)["stage"] == "stopped"
    assert len(calls) == cfg["training"]["accumulate"]
    assert not (Path(cfg["root"]) / "checkpoints/final_head.pt").exists()
    cfg["resume"] = "auto"
    result = engine.train(cfg, stop_after_updates=1)
    assert result["reused"] and len(calls) == cfg["training"]["accumulate"]


def test_evaluation_dry_run_rejects_invalid_model_capabilities(configured):
    from finefune.cli import main
    cfg, args, _ = configured
    argv = ["evaluate", "--dry-run", "--variant", "official", "--tracks", "gt-box"]
    for key, value in args.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    assert main(argv) == 1
    assert not Path(cfg["root"]).exists()


def test_origin_checkpoint_checks_nested_prefixes_and_missing_parameters(tmp_path):
    from finefune.models.native import checkpoint_coverage
    model = Tiny()
    path = tmp_path / "origin.pt"
    state = {"model.module." + k: v for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=state), path)
    assert checkpoint_coverage(model, path)["parameter_tensors"] == len(list(model.parameters()))
    state.pop(next(iter(state)))
    torch.save(dict(state_dict=state), path)
    with pytest.raises(ValueError, match="checkpoint 不完整"):
        checkpoint_coverage(model, path)
