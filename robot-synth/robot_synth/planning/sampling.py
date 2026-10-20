"""Freeze finite source states before any graphics context is initialized."""

from collections import deque
import hashlib
import math
from pathlib import Path
import random

import numpy as np

from ..common import atomic_json, digest, read_json
from ..config import config_id
from ..sources.libero_hdf5 import read_episode, state_digest, state_layout
from .scenes import group_scenes
from .splits import allocate_scenes


def novel_indices(states, minimum_distance, nq):
    """Use qpos only: robosuite state is [time, qpos, qvel], with na == 0."""
    conservative_qpos = states[:, 1:1 + nq]
    keep = [0]
    for index in range(1, len(states)):
        if np.linalg.norm(conservative_qpos[index] - conservative_qpos[keep[-1]]) >= minimum_distance:
            keep.append(index)
    return keep


def spread(values, count):
    if count >= len(values):
        return list(values)
    return [values[i] for i in np.linspace(0, len(values) - 1, count, dtype=int)]


def build_plan(config, inventory):
    path = Path(config["output"]["run_dir"]) / "plan.json"
    identity = config_id(config)
    if path.exists():
        old = read_json(path)
        if old["config_id"] != identity or old["inventory_id"] != inventory["inventory_id"]:
            raise ValueError("Existing plan differs from configuration/source; choose a new run_dir and output root")
        print(f"复用冻结计划：{old['plan_id']}", flush=True)
        return old
    scenes = allocate_scenes(config, group_scenes(inventory))
    states_out = []
    scene_records = []
    for scene in scenes:
        rng = random.Random(f"{config['sampling']['seed']}:{scene['id']}")
        tasks = sorted(scene["tasks"], key=lambda t: t["source"])
        rng.shuffle(tasks)
        queues = []
        for task in tasks:
            episodes = list(task["episodes"])
            rng.shuffle(episodes)
            queues.append(deque((task, ep) for ep in episodes))
        selected = []
        goal = config["sampling"]["episodes_per_scene"][scene["split"]]
        # Round-robin across tasks before adding more episodes from any one task.
        while len(selected) < goal and any(queues):
            for queue in queues:
                if queue and len(selected) < goal:
                    selected.append(queue.popleft())
        per_episode = []
        for task, ep in selected:
            xml, states = read_episode(config, task["source"], ep["name"])
            layout = state_layout(xml)
            if states.shape[1] != layout["width"]:
                raise ValueError(f"State width differs from XML joint layout: {task['source']}/{ep['name']}")
            indices = novel_indices(states, config["sampling"]["min_qpos_distance"], layout["nq"])
            per_episode.append((task, ep, xml, states, indices))
        needed = scene["target_states"]
        reserve = math.ceil(needed * config["sampling"]["reserve_fraction"])
        available = sum(len(values[-1]) for values in per_episode)
        if available < needed:
            raise ValueError(f"Not enough non-static states in {scene['id']}: {available} < {needed}; increase episodes_per_scene")
        wanted = min(available, needed + reserve)
        allocations = [0] * len(per_episode)
        while sum(allocations) < wanted:
            for i, values in enumerate(per_episode):
                if sum(allocations) < wanted and allocations[i] < len(values[-1]):
                    allocations[i] += 1
        candidates = []
        for allocation, (task, ep, xml, states, indices) in zip(allocations, per_episode):
            xml_sha = hashlib.sha256(xml.encode()).hexdigest()
            for index in spread(indices, allocation):
                record = {"scene_id": scene["id"], "split": scene["split"], "room": scene["room"],
                          "source": task["source"], "episode": ep["name"], "state_index": index,
                          "state_sha256": state_digest(states[index]), "xml_sha256": xml_sha}
                record["state_id"] = digest(record)[:24]
                candidates.append(record)
        rng.shuffle(candidates)
        states_out.extend(candidates)
        scene_records.append({key: scene[key] for key in ("id", "room", "aliases", "split", "target_states", "target_images")})
        scene_records[-1].update(candidate_states=len(candidates), episodes=len(per_episode), tasks=len(tasks))
        print(f"计划 {scene['split']}/{scene['id']}：目标 {scene['target_images']} 张，候选 {len(candidates)} 个状态", flush=True)
    plan = {"schema_version": 1, "config_id": identity, "inventory_id": inventory["inventory_id"],
            "scenes": scene_records, "states": states_out,
            "cameras": config["simulation"]["cameras"],
            "image_count_definition": "one camera observation, not one simulation timestamp",
            "targets": {split: sum(s["target_images"] for s in scene_records if s["split"] == split)
                        for split in ("train", "eval")}}
    plan["plan_id"] = digest(plan)
    atomic_json(path, plan)
    atomic_json(path.parent / "scene_split.json", {"plan_id": plan["plan_id"], "scenes": scene_records})
    return plan


def load_plan(config):
    plan = read_json(Path(config["output"]["run_dir"]) / "plan.json")
    if plan["config_id"] != config_id(config):
        raise ValueError("Configuration differs from frozen plan; use its original configuration")
    identity = plan.pop("plan_id")
    if digest(plan) != identity:
        raise ValueError("Frozen plan content was modified")
    plan["plan_id"] = identity
    return plan
