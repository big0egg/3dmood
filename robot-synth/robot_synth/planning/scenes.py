from collections import defaultdict


def group_scenes(inventory):
    """Merge labels sharing an identical normalized example scene configuration.

    This conservative check catches repeated scene definitions; it is not a
    claim that other scene groups correspond to independent physical rooms.
    """
    parents = {t["scene_id"]: t["scene_id"] for t in inventory["tasks"]}

    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    seen = {}
    for task in inventory["tasks"]:
        name = task["scene_id"]
        signature = task["layout_signature"]
        if signature in seen:
            first, second = sorted((find(name), find(seen[signature])))
            parents[second] = first
        seen[signature] = name
    groups = {}
    for task in inventory["tasks"]:
        key = find(task["scene_id"])
        group = groups.setdefault(key, {"id": key, "room": task["room"], "aliases": [], "tasks": []})
        if group["room"] != task["room"]:
            raise ValueError(f"Identical scene definition spans room labels: {key}")
        group["aliases"].append(task["scene_id"])
        group["tasks"].append(task)
    for group in groups.values():
        group["aliases"] = sorted(set(group["aliases"]))
    return [groups[key] for key in sorted(groups)]
