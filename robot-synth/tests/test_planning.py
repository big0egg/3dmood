from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from robot_synth.common import read_json
from robot_synth.config import load_config
from robot_synth.planning.sampling import build_plan, load_plan, novel_indices, spread
from robot_synth.planning.scenes import group_scenes
from robot_synth.planning.splits import allocate_scenes
from robot_synth.sources.libero_hdf5 import inspect_source, state_layout

from conftest import scene_xml


def test_inventory_plan_and_scene_isolation(config):
    inventory = inspect_source(config)
    assert inventory["task_count"] == 6
    assert inventory["episode_count"] == 18
    assert inventory["timesteps"] == 360
    plan = build_plan(config, inventory)
    assert plan["targets"] == {"train": 36, "eval": 12}
    assert sum(s["target_states"] for s in plan["scenes"]) == 24
    train = {s["id"] for s in plan["scenes"] if s["split"] == "train"}
    evaluation = {s["id"] for s in plan["scenes"] if s["split"] == "eval"}
    assert len(train) == len(evaluation) == 3 and not train & evaluation
    assert len({s["state_id"] for s in plan["states"]}) == len(plan["states"])
    assert build_plan(config, inventory) == load_plan(config) == plan
    assert inspect_source(config) == inventory


def test_changed_source_cannot_silently_reuse_plan(config):
    inventory = inspect_source(config)
    build_plan(config, inventory)
    path = next(Path(config["source"]["root"]).rglob("*.hdf5"))
    with h5py.File(path, "r+") as f:
        f["data/demo_0/states"][0, 1] = 9
    updated = inspect_source(config)
    with pytest.raises(ValueError, match="differs"):
        build_plan(config, updated)


def test_modified_plan_and_config_are_rejected(config):
    plan = build_plan(config, inspect_source(config))
    changed = deepcopy(config)
    changed["simulation"]["width"] = 64
    with pytest.raises(ValueError, match="Configuration differs"):
        load_plan(changed)
    path = Path(config["output"]["run_dir"]) / "plan.json"
    from robot_synth.common import atomic_json
    plan["states"][0]["state_index"] += 1
    atomic_json(path, plan)
    with pytest.raises(ValueError, match="modified"):
        load_plan(config)


def test_xml_scene_duplicates_merge_across_suite_labels(config):
    inventory = inspect_source(config)
    original = inventory["tasks"][0]
    duplicate = deepcopy(original)
    duplicate["scene_id"] = "KITCHEN_SCENE3"
    duplicate["source"] = "libero_10/KITCHEN_SCENE3_other_demo.hdf5"
    inventory["tasks"].append(duplicate)
    groups = group_scenes(inventory)
    group = next(g for g in groups if original["scene_id"] in g["aliases"])
    assert "KITCHEN_SCENE3" in group["aliases"] and len(groups) == 6


def test_true_qpos_layout_and_static_filter():
    assert state_layout(scene_xml()) == {"nq": 7, "nv": 6, "width": 14, "format": "robosuite.time_qpos_qvel"}
    states = np.zeros((10, 14))
    states[:, 0] = np.arange(10)  # time alone must not make new examples
    states[:, 8:] = np.arange(10)[:, None]  # velocities also do not count
    assert novel_indices(states, .01, 7) == [0]
    states[5:, 1] = 1
    assert novel_indices(states, .01, 7) == [0, 5]
    assert spread(list(range(20)), 4) == [0, 6, 12, 19]


def test_insufficient_states_stop_before_freezing_plan(config):
    config["sampling"]["min_qpos_distance"] = 100
    with pytest.raises(ValueError, match="Not enough"):
        build_plan(config, inspect_source(config))
    assert not (Path(config["output"]["run_dir"]) / "plan.json").exists()


def test_default_exact_11000_quota_and_no_overlap():
    config = load_config(Path(__file__).parents[1] / "configs/libero_gt_v1.yaml")
    groups = [{"id": f"{room}_SCENE{i}", "room": room, "aliases": [f"{room}_SCENE{i}"], "tasks": []}
              for room, count in (("KITCHEN", 10), ("LIVING_ROOM", 6), ("STUDY", 4)) for i in range(1, count + 1)]
    result = allocate_scenes(config, groups)
    assert sum(s["target_images"] for s in result if s["split"] == "train") == 10000
    assert sum(s["target_images"] for s in result if s["split"] == "eval") == 1000
    assert all(s["target_images"] % 2 == 0 for s in result)
    assert sum(s["split"] == "eval" for s in result) == 4


def test_configuration_path_safety(tmp_path):
    import yaml
    base = load_config(Path(__file__).parents[1] / "configs/libero_gt_v1.yaml")
    base["output"]["root"] = base["source"]["root"] + "/bad-output"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="must not contain"):
        load_config(path)
