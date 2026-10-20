"""Read-only access to the original LIBERO demonstrations."""

import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from ..common import atomic_json, digest, read_json


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def scene_name(path):
    match = re.match(r"(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+(?=_)", Path(path).name)
    if match is None:
        raise ValueError(f"No scene identifier in {path}; use libero_90/libero_10 for this acquisition")
    return match.group(), match.group(1)


def normalized_xml(xml):
    """Identity ignores machine asset prefixes, not mesh content or object names."""
    root = ET.fromstring(xml)
    for node in root.iter():
        for name, value in list(node.attrib.items()):
            if name in ("file", "meshdir", "texturedir"):
                marker = "/assets/"
                node.set(name, value.split(marker, 1)[-1] if marker in value else value)
    return root


def layout_signature(xml):
    root = normalized_xml(xml)
    world = root.find("worldbody")
    if world is not None:
        for body in list(world.findall("body")):
            if re.match(r"(?:robot|gripper|mount)\d*_", body.get("name", "")):
                world.remove(body)
            elif any(j.get("type") == "free" for j in body.findall("joint")) or body.find("freejoint") is not None:
                # Remove initialization pose only, retain object geometry and category.
                body.attrib.pop("pos", None)
                body.attrib.pop("quat", None)
    return hashlib.sha256(ET.tostring(root)).hexdigest()


def state_digest(state):
    return hashlib.sha256(np.asarray(state, dtype="<f8").tobytes()).hexdigest()


def state_layout(xml):
    world = ET.fromstring(xml).find("worldbody")
    nq = nv = 0
    sizes = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}
    for node in world.iter():
        if node.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
        if kind not in sizes:
            raise ValueError(f"Unsupported joint type: {kind}")
        q, v = sizes[kind]
        nq += q
        nv += v
    return {"nq": nq, "nv": nv, "width": 1 + nq + nv, "format": "robosuite.time_qpos_qvel"}


def source_path(config, relative):
    root = Path(config["source"]["root"])
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe HDF5 path: {relative}")
    return path


def source_stamp(path):
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def inspect_source(config, *, refresh=False):
    import h5py

    root = Path(config["source"]["root"])
    paths = []
    for suite in config["source"]["suites"]:
        if suite not in ("libero_90", "libero_10"):
            raise ValueError("Scene-group acquisition currently supports libero_90 and libero_10")
        paths.extend(sorted((root / suite).glob("*.hdf5")))
    if not paths:
        raise ValueError(f"No HDF5 files found in {root}")
    stamps = {p.relative_to(root).as_posix(): source_stamp(p) for p in paths}
    cache = Path(config["output"]["run_dir"]) / "inventory.json"
    if cache.exists() and not refresh:
        previous = read_json(cache)
        if previous.get("source_root") == str(root) and previous.get("source_stamps") == stamps:
            print(f"复用清点记录：{len(previous['tasks'])} 个任务", flush=True)
            return previous
    tasks = []
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(root).as_posix()
        scene, room = scene_name(path)
        with h5py.File(path, "r") as file:
            data = file["data"]
            episodes = []
            for name in sorted(data, key=lambda s: int(s.rsplit("_", 1)[-1])):
                episode = data[name]
                if "states" not in episode or "model_file" not in episode.attrs:
                    raise ValueError(f"Replay state/XML missing: {relative}/{name}")
                states = episode["states"]
                if len(states.shape) != 2 or states.shape[0] < 1:
                    raise ValueError(f"Invalid states shape: {relative}/{name}")
                episodes.append({"name": name, "length": states.shape[0], "state_width": states.shape[1]})
            if not episodes:
                raise ValueError(f"No episodes: {relative}")
            example = data[episodes[0]["name"]]
            xml = text(example.attrs["model_file"])
            env_args = json.loads(text(data.attrs["env_args"]))
            obs = example.get("obs")
            observations = {name: list(value.shape) for name, value in obs.items()} if obs is not None else {}
            tasks.append({"source": relative, "suite": path.parent.name, "scene_id": scene,
                          "room": room, "episodes": episodes, "layout_signature": layout_signature(xml),
                          "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                          "asset_files": sorted({n.get("file") for n in ET.fromstring(xml).iter() if n.get("file")}),
                          "tag": text(data.attrs.get("tag", "unknown")),
                          "image_convention": text(data.attrs.get("macros_image_convention", "unknown")),
                          "env_args": env_args, "example_observations": observations})
        if index == 1 or index % 10 == 0 or index == len(paths):
            print(f"清点 [{index}/{len(paths)}] {relative}", flush=True)
    inventory = {"schema_version": 1, "source_root": str(root), "source_stamps": stamps,
                 "tasks": tasks, "task_count": len(tasks),
                 "episode_count": sum(len(t["episodes"]) for t in tasks),
                 "timesteps": sum(e["length"] for t in tasks for e in t["episodes"])}
    inventory["inventory_id"] = digest({"stamps": stamps, "tasks": tasks})
    atomic_json(cache, inventory)
    return inventory


def read_episode(config, task, episode):
    import h5py

    with h5py.File(source_path(config, task), "r") as file:
        group = file["data"][episode]
        xml = text(group.attrs["model_file"])
        states = np.asarray(group["states"], dtype=np.float64)
    if not np.isfinite(states).all():
        raise ValueError(f"Non-finite simulation states: {task}/{episode}")
    return xml, states
