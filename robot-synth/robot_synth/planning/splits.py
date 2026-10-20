from collections import defaultdict
import random


def allocate_scenes(config, groups):
    sampling = config["sampling"]
    cameras = len(config["simulation"]["cameras"])
    rooms = defaultdict(list)
    for group in groups:
        rooms[group["room"]].append(group)
    explicit = set(sampling.get("eval_scene_ids", []))
    known = {alias for group in groups for alias in group["aliases"]}
    if explicit - known:
        raise ValueError(f"Unknown eval scene IDs: {sorted(explicit - known)}")
    result = []
    for room in sorted(sampling["quotas"]["train"]):
        choices = sorted(rooms[room], key=lambda g: g["id"])
        random.Random(f"{sampling['seed']}:{room}").shuffle(choices)
        count = sampling["eval_scenes"][room]
        if len(choices) <= count:
            raise ValueError(f"Insufficient independent scene groups in {room}: {len(choices)}")
        evaluation = ([g for g in choices if explicit.intersection(g["aliases"])]
                      if explicit else choices[:count])
        if len(evaluation) != count:
            raise ValueError(f"Expected {count} eval scene groups in {room}, got {len(evaluation)}")
        evaluation_ids = {g["id"] for g in evaluation}
        for split, subset in (("train", [g for g in choices if g["id"] not in evaluation_ids]), ("eval", evaluation)):
            quota = sampling["quotas"][split][room] // cameras
            base, remainder = divmod(quota, len(subset))
            if base < 1:
                raise ValueError(f"Quota too small to cover all {room}/{split} scene groups")
            for index, group in enumerate(sorted(subset, key=lambda g: g["id"])):
                result.append({**group, "split": split, "target_states": base + (index < remainder),
                               "target_images": (base + (index < remainder)) * cameras})
    return sorted(result, key=lambda g: (g["split"], g["id"]))
