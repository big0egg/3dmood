"""A single writer exports complete two-camera states up to frozen quotas."""

from collections import Counter
import hashlib
from pathlib import Path
import time
import traceback

from .common import atomic_json, digest, local_lock, read_json
from .simulation.doctor import diagnose, runtime_identity
from .simulation.libero_replay import LiberoReplay
from .sources.libero_hdf5 import read_episode, source_path, source_stamp, state_digest
from .storage.writer import DatasetWriter, rgb_png_hash
from .validation.geometry import validate_views


def collection_goals(plan, limit=None):
    """A small pilot covers every scene group instead of only the first scene."""
    cameras = len(plan["cameras"])
    full = {s["id"]: s["target_states"] for s in plan["scenes"]}
    if limit is None:
        return full
    if limit <= 0 or limit % cameras:
        raise ValueError(f"--limit counts images and must be positive/divisible by {cameras}")
    remaining = min(limit // cameras, sum(full.values()))
    goals = {name: 0 for name in full}
    while remaining:
        for name in sorted(full):
            if remaining and goals[name] < full[name]:
                goals[name] += 1
                remaining -= 1
    return goals


def collect(config, plan, inventory, *, limit=None, resume=False, replay_factory=None, profile=None):
    """Dependency injection is for unit tests only; CLI has no mock data mode."""
    goals = collection_goals(plan, limit)
    if replay_factory is None:
        report = diagnose(config, inventory)
        if not report["ready"]:
            problems = [c["component"] for c in report["checks"] if not c["ok"]]
            raise RuntimeError("Runtime is not ready: " + ", ".join(problems) + "; see doctor.json and README")
        replay_factory = LiberoReplay
        profile = runtime_identity(config, inventory)
    elif profile is None:
        raise ValueError("Injected replay backend requires an explicit test runtime profile")
    run_dir = Path(config["output"]["run_dir"])
    with local_lock(run_dir):
        writer = DatasetWriter(config, plan, profile, resume=resume)
        progress_path = run_dir / "progress.json"
        progress = {"plan_id": plan["plan_id"], "profile_id": writer.profile_id,
                    "accepted": {}, "rejected": {}, "errors": []}
        for snapshot in (writer.root / "manifest.json", progress_path):
            if snapshot.exists():
                previous = read_json(snapshot)
                if previous.get("plan_id") != plan["plan_id"] or previous.get("profile_id") != writer.profile_id:
                    raise ValueError("Progress identity differs from dataset")
                progress["accepted"].update(previous["accepted"])
                progress["rejected"].update(previous["rejected"])
                progress["errors"].extend(previous["errors"])
        accepted, rejected = progress["accepted"], progress["rejected"]
        planned = {r["state_id"]: r for r in plan["states"]}
        commits = {}
        if resume:
            # Recover final COMMITs even if the process died after writing a
            # bundle but before saving local progress/remote summaries.
            print("校验并恢复已提交的状态组…", flush=True)
            for scene in plan["scenes"]:
                directory = writer.root / scene["split"] / scene["id"]
                if not directory.exists():
                    continue
                for folder in sorted(directory.iterdir()):
                    if not folder.is_dir() or not (folder / "COMMIT.json").exists():
                        continue
                    if folder.name not in planned:
                        raise ValueError(f"Unplanned committed state in output: {folder}")
                    record = planned[folder.name]
                    commit = writer.existing(record)
                    commits[folder.name] = commit
                    accepted[folder.name] = record
                    rejected.pop(folder.name, None)
        counts = Counter()
        image_hashes = set()
        category_catalog = {}

        def remember_categories(commit):
            for value in commit["categories"]:
                item = category_catalog.setdefault(value["name"], {"name": value["name"], "raw_categories": set(), "kinds": set()})
                item["raw_categories"].update(value["raw_categories"])
                item["kinds"].update(value["kinds"])

        for state_id, record in accepted.items():
            if planned.get(state_id) != record:
                raise ValueError(f"Progress contains an unplanned state: {state_id}")
            commit = commits.get(state_id) or writer.existing(record)
            if commit is None:
                raise ValueError(f"Progress references a missing committed state: {state_id}")
            counts[record["scene_id"]] += 1
            pair = tuple(commit["rgb_sha256"][c] for c in plan["cameras"])
            if pair in image_hashes:
                raise ValueError("Committed data contains duplicate two-camera images")
            image_hashes.add(pair)
            remember_categories(commit)
        if any(counts[s["id"]] > s["target_states"] for s in plan["scenes"]):
            raise ValueError("Committed state count exceeds frozen scene quota")
        started = time.monotonic()
        backend = None
        episode_key = None
        states = xml = None
        fatal = None
        consecutive_rejections = 0
        verified_sources = set()
        candidate_by_scene = {scene["id"]: [] for scene in plan["scenes"]}
        for record in plan["states"]:
            candidate_by_scene[record["scene_id"]].append(record)
        try:
            for scene in plan["scenes"]:
                name = scene["id"]
                if counts[name] >= goals[name]:
                    continue
                candidates = [r for r in candidate_by_scene[name] if r["state_id"] not in accepted and r["state_id"] not in rejected]
                # Preserve the planned random priority, but group each next batch
                # by episode to avoid rebuilding a large model for every frame.
                while counts[name] < goals[name] and candidates:
                    amount = goals[name] - counts[name]
                    batch, candidates = candidates[:amount], candidates[amount:]
                    for record in sorted(batch, key=lambda r: (r["source"], r["episode"], r["state_index"])):
                        state_id = record["state_id"]
                        existing = writer.existing(record)
                        if existing is not None:
                            accepted[state_id] = record
                            counts[name] += 1
                            image_hashes.add(tuple(existing["rgb_sha256"][c] for c in plan["cameras"]))
                            remember_categories(existing)
                            atomic_json(progress_path, progress)
                            continue
                        if record["source"] not in verified_sources:
                            path = source_path(config, record["source"])
                            if source_stamp(path) != inventory["source_stamps"][record["source"]]:
                                raise ValueError(f"Source file changed since plan: {path}")
                            verified_sources.add(record["source"])
                        key = (record["source"], record["episode"])
                        if key != episode_key:
                            if backend is not None:
                                backend.close()
                                backend = None
                            xml, states = read_episode(config, *key)
                            if hashlib.sha256(xml.encode()).hexdigest() != record["xml_sha256"]:
                                raise ValueError("Scene XML differs from frozen plan")
                            backend = replay_factory(config, xml)
                            episode_key = key
                        state = states[record["state_index"]]
                        if state_digest(state) != record["state_sha256"]:
                            raise ValueError("Source state differs from frozen plan")
                        views = backend.render(state)
                        try:
                            quality = validate_views(views, config)
                            # Exactly repeated two-camera images do not add a new
                            # usable state, even when simulation timestamps differ.
                            pair = tuple(rgb_png_hash(views[c]["rgb"]) for c in plan["cameras"])
                            if pair in image_hashes:
                                raise ValueError("Identical two-camera RGB state already selected")
                        except ValueError as error:
                            rejected[state_id] = {"record": record, "reason": str(error)}
                            consecutive_rejections += 1
                            atomic_json(progress_path, progress)
                            print(f"拒绝 {state_id}: {error}", flush=True)
                            if consecutive_rejections >= config["validation"]["max_failures"]:
                                raise RuntimeError("Too many consecutive GT quality failures; inspect errors before resuming") from error
                            continue
                        commit = writer.save(record, views, state, xml, quality)
                        remember_categories(commit)
                        image_hashes.add(tuple(commit["rgb_sha256"][c] for c in plan["cameras"]))
                        accepted[state_id] = record
                        counts[name] += 1
                        consecutive_rejections = 0
                        atomic_json(progress_path, progress)
                        print(f"{record['split']}/{name}: {counts[name]}/{goals[name]} 个状态；累计 {len(accepted) * len(plan['cameras'])} 张", flush=True)
                if counts[name] < goals[name]:
                    raise RuntimeError(f"Candidate pool exhausted in {name}: {counts[name]}/{goals[name]} states")
        except BaseException as error:
            fatal = str(error) or type(error).__name__
            progress["errors"].append({"error": fatal, "traceback": traceback.format_exc()})
            raise
        finally:
            if backend is not None:
                backend.close()
            full_targets = {s["id"]: s["target_states"] for s in plan["scenes"]}
            complete = all(counts[k] == value for k, value in full_targets.items())
            progress["status"] = "failed" if fatal else "complete" if complete else "partial"
            progress["last_elapsed_seconds"] = time.monotonic() - started
            progress["accepted_images"] = len(accepted) * len(plan["cameras"])
            progress["counts_by_scene"] = dict(counts)
            atomic_json(progress_path, progress)
            # The immutable per-state COMMIT markers remain authoritative if an
            # OSS metadata summary is interrupted or stale.
            atomic_json(writer.root / "manifest.json", progress)
            atomic_json(writer.root / "categories.json", {
                "identity": "stable category names, no model-specific numeric IDs",
                "categories": [{"name": name, "raw_categories": sorted(item["raw_categories"]), "kinds": sorted(item["kinds"])}
                               for name, item in sorted(category_catalog.items())]})
        return progress
