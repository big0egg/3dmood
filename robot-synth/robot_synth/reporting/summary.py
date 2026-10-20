from collections import Counter
from pathlib import Path

from ..common import atomic_json, read_json
from ..validation.dataset import load_views
from .visualize import preview


def report(config, plan, maximum_previews=20):
    root = Path(config["output"]["root"])
    directory = Path(config["output"]["run_dir"]) / "quality_report"
    progress = read_json(root / "manifest.json")
    records = list(progress["accepted"].values())
    scenes = {}
    for record in records:
        scenes.setdefault(record["scene_id"], []).append(record)
    selected = []
    # At least one example per scene before taking more from the same scene.
    for index in range(maximum_previews):
        for name in sorted(scenes):
            if index < len(scenes[name]) and len(selected) * len(plan["cameras"]) < maximum_previews:
                selected.append(scenes[name][index])
    categories, files = Counter(), []
    for record in selected:
        views = load_views(root, record, plan["cameras"])
        for camera, view in views.items():
            if len(files) >= maximum_previews:
                break
            path = directory / "previews" / f"{record['scene_id']}_{record['state_id']}_{camera}.png"
            preview(view, path)
            files.append(str(path))
            for obj in view["annotation"]["objects"]:
                categories[obj["category"]] += 1
    result = {"plan_id": plan["plan_id"], "status": progress["status"],
              "accepted_images": len(records) * len(plan["cameras"]), "targets": plan["targets"],
              "scene_states": {name: len(values) for name, values in scenes.items()},
              "rejected_states": len(progress["rejected"]), "previews": files,
              "preview_category_counts_only": dict(categories)}
    atomic_json(directory / "summary.json", result)
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# LIBERO ground-truth acquisition", "", f"Status: {result['status']}",
             f"Accepted camera images: {result['accepted_images']}",
             f"Targets: {plan['targets']}", "", "Preview boxes are simulation ground truth.",
             "Full visible-mesh boxes can be larger than visible masks. No model predictions are used.", ""]
    for path in files:
        lines.append(f"![GT preview](previews/{Path(path).name})")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
