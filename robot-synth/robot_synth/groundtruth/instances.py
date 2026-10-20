"""Explicit hierarchy; segmentation belongs to deepest annotated part."""

import re
import numpy as np


def descendants(model, root):
    values = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in values:
            values.add(body)
    return values


def registry(model, mujoco, config):
    groups = set(config["simulation"]["visual_geom_groups"])
    visual = [g for g in range(model.ngeom) if int(model.geom_group[g]) in groups]
    entries = []
    aliases = config["groundtruth"].get("category_aliases", {})

    def add(body, parent=None, kind="object"):
        bodies = descendants(model, body)
        geoms = [g for g in visual if int(model.geom_bodyid[g]) in bodies]
        if not geoms:
            return None
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body_{body}"
        raw = re.sub(r"_\d+$", "", re.sub(r"_main$", "", name))
        category = ("robot" if kind == "robot" else "table" if kind == "support"
                    else "articulated_part" if kind == "part" else aliases.get(raw, raw))
        instance_id = len(entries) + 1
        entries.append({"instance_id": instance_id, "body_id": body, "body_name": name,
                        "raw_category": raw, "category": category, "kind": kind,
                        "parent_instance_id": parent, "body_ids": sorted(bodies), "geom_ids": geoms})
        return instance_id

    for body in range(1, model.nbody):
        if model.body_parentid[body] != 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        robot = bool(re.match(r"(?:robot|gripper|mount)\d*_", name))
        support = name in ("table", "floor")
        if robot and not config["groundtruth"]["include_robot"]:
            continue
        if support and not config["groundtruth"]["include_support"]:
            continue
        # World-level scenery meshes are deliberately not promoted to semantic
        # objects: some LIBERO backgrounds are textured scene props.
        parent = add(body, kind="robot" if robot else "support" if support else "object")
        if parent is None or robot or support or not config["groundtruth"]["include_articulated_parts"]:
            continue
        child_ids = descendants(model, body) - {body}
        part_by_body = {body: parent}
        for child in sorted(child_ids):
            if int(model.body_jntnum[child]) > 0:
                ancestor = int(model.body_parentid[child])
                while ancestor not in part_by_body:
                    ancestor = int(model.body_parentid[ancestor])
                part = add(child, parent=part_by_body[ancestor], kind="part")
                if part is not None:
                    part_by_body[child] = part
    if not any(e["kind"] == "object" for e in entries):
        raise ValueError("No annotated objects found in scene model")
    if len(entries) > np.iinfo(np.uint16).max:
        raise ValueError("Instance ID space exceeds uint16; refusing to wrap IDs")
    owners = np.zeros(model.ngeom, dtype=np.uint16)
    for entry in entries:
        owners[entry["geom_ids"]] = entry["instance_id"]
    for entry in entries:
        entry["segmentation_ids"] = sorted(set(map(int, owners[entry["geom_ids"]])))
    return entries, owners
