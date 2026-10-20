from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import atomic_json, read_json
from ..storage.manifest import bundle_relative, validate_commit
from .geometry import validate_views


def load_views(root, record, cameras):
    folder = Path(root) / bundle_relative(record)
    views = {}
    for name in cameras:
        directory = folder / name
        with Image.open(directory / "rgb.png") as image:
            rgb = np.array(image)
        with Image.open(directory / "instance.png") as image:
            instance = np.asarray(image).astype(np.uint16)
        with np.load(directory / "depth.npz", allow_pickle=False) as values:
            depth, valid = values["z_m"].copy(), values["valid"].copy()
        annotation = read_json(directory / "annotation.json")
        views[name] = {"rgb": rgb, "instance": instance, "depth": depth,
                       "depth_valid": valid, "annotation": annotation}
    return views


def validate_dataset(config, plan, *, allow_partial=False):
    root = Path(config["output"]["root"])
    info = read_json(root / "dataset_info.json")
    if info["plan_id"] != plan["plan_id"]:
        raise ValueError("Dataset plan identity differs")
    progress = read_json(root / "manifest.json")
    planned = {r["state_id"]: r for r in plan["states"]}
    counts, rooms, categories, by_scene = Counter(), Counter(), Counter(), Counter()
    scene_sets = {"train": set(), "eval": set()}
    errors = []
    accepted = progress["accepted"]
    image_pairs = {}
    for index, (state_id, record) in enumerate(accepted.items(), 1):
        try:
            if planned.get(state_id) != record:
                raise ValueError("Accepted state was not in frozen plan")
            commit = validate_commit(root, record, info["plan_id"], info["profile_id"])
            if commit is None or commit["cameras"] != plan["cameras"]:
                raise ValueError("Missing/incomplete camera-pair commit")
            with (root / bundle_relative(record) / "state.npy").open("rb") as stream:
                state = np.load(stream, allow_pickle=False)
            from ..sources.libero_hdf5 import state_digest
            if state_digest(state) != record["state_sha256"]:
                raise ValueError("Saved state differs from source state identity")
            views = load_views(root, record, plan["cameras"])
            if any(view["annotation"].get("provenance") != record for view in views.values()):
                raise ValueError("Image annotation provenance differs from committed source state")
            validate_views(views, config)
            pair = tuple(commit["rgb_sha256"][c] for c in plan["cameras"])
            if pair in image_pairs:
                raise ValueError(f"Duplicate two-camera images: {state_id} and {image_pairs[pair]}")
            image_pairs[pair] = state_id
            counts[record["split"]] += len(views)
            by_scene[record["scene_id"]] += len(views)
            rooms[f"{record['split']}/{record['room']}"] += len(views)
            scene_sets[record["split"]].add(record["scene_id"])
            for view in views.values():
                for obj in view["annotation"]["objects"]:
                    if obj["visible"]:
                        categories[obj["category"]] += 1
        except Exception as error:
            errors.append({"state_id": state_id, "error": str(error)})
        if index % 50 == 0:
            print(f"验证 [{index}/{len(accepted)}]", flush=True)
    overlap = scene_sets["train"] & scene_sets["eval"]
    if overlap:
        errors.append({"error": "Train/eval scene overlap", "scenes": sorted(overlap)})
    complete = all(counts[split] == target for split, target in plan["targets"].items())
    for scene in plan["scenes"]:
        count = by_scene[scene["id"]]
        if count > scene["target_images"] or (not allow_partial and count != scene["target_images"]):
            errors.append({"error": "Scene quota mismatch", "scene_id": scene["id"],
                           "images": count, "target": scene["target_images"]})
    if not complete and not allow_partial:
        errors.append({"error": "Dataset has not reached all frozen image quotas", "counts": dict(counts)})
    result = {"valid": not errors, "complete": complete, "plan_id": plan["plan_id"],
              "counts": dict(counts), "targets": plan["targets"], "by_room": dict(rooms),
              "scene_counts": {split: len(values) for split, values in scene_sets.items()},
              "visible_category_instances": dict(categories), "errors": errors,
              "saved_groundtruth_validated": bool(accepted) and not errors,
              "native_renderer_verified": bool(accepted) and not errors and info["profile"].get("backend") == "native_mujoco",
              "note": "Dataset checks do not constitute model evaluation or accuracy claims"}
    atomic_json(Path(config["output"]["run_dir"]) / "validation.json", result)
    return result
