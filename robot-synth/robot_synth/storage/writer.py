import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..common import atomic_json, digest, file_digest, read_json
from .manifest import bundle_relative, validate_commit


def rgb_png_hash(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return hashlib.sha256(stream.getvalue()).hexdigest()


def exclusive_bytes(path, content):
    """Final remote commit files are created after every payload is copied."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
    if path.read_bytes() != content:
        raise OSError(f"Read-after-write verification failed: {path}")


def exclusive_json(path, value):
    exclusive_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


class DatasetWriter:
    def __init__(self, config, plan, profile, *, resume=False):
        self.config, self.plan, self.profile = config, plan, profile
        self.profile_id = digest(profile)
        self.root = Path(config["output"]["root"])
        self.scratch = Path(config["output"]["scratch"])
        info = self.root / "dataset_info.json"
        expected = {"schema_version": "robot_synth.gt.v1", "plan_id": plan["plan_id"],
                    "profile_id": self.profile_id, "profile": profile, "config": config,
                    "targets": plan["targets"], "status_policy": "only validated COMMIT.json bundles are samples",
                    "collection_backend": "recorded XML/native MuJoCo; no policy rollout"}
        if info.exists():
            current = read_json(info)
            if not resume:
                raise ValueError("Dataset already exists; use --resume")
            for key in ("plan_id", "profile_id", "targets"):
                if current.get(key) != expected[key]:
                    raise ValueError(f"Dataset identity differs ({key}); choose a new output and run_dir")
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("Output directory is not empty and has no dataset ownership record")
            self.root.mkdir(parents=True, exist_ok=True)
            exclusive_json(info, expected)
        for name, content in (("scene_split.json", {"plan_id": plan["plan_id"], "scenes": plan["scenes"]}),
                              ("sampling_plan.json", plan)):
            path = self.root / name
            if path.exists():
                if read_json(path) != content:
                    raise ValueError(f"Frozen dataset metadata differs: {name}")
            else:
                exclusive_json(path, content)
        self.scratch.mkdir(parents=True, exist_ok=True)

    def existing(self, record):
        return validate_commit(self.root, record, self.plan["plan_id"], self.profile_id)

    def save(self, record, views, state, xml, quality):
        existing = self.existing(record)
        if existing is not None:
            return existing
        local = self.scratch / self.plan["plan_id"] / record["state_id"]
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        np.save(local / "state.npy", np.asarray(state, dtype=np.float64), allow_pickle=False)
        episode_id = digest({"source": record["source"], "episode": record["episode"], "xml": record["xml_sha256"]})[:24]
        xml_path = self.root / "provenance" / (episode_id + ".xml.gz")
        xml_bytes = gzip.compress(xml.encode(), mtime=0)
        if xml_path.exists():
            if gzip.decompress(xml_path.read_bytes()).decode() != xml:
                raise ValueError(f"Episode provenance differs: {xml_path}")
        else:
            exclusive_bytes(xml_path, xml_bytes)
        rgb_hashes = {}
        categories = {}
        for camera, view in views.items():
            folder = local / camera
            folder.mkdir()
            Image.fromarray(view["rgb"]).save(folder / "rgb.png")
            Image.fromarray(view["instance"]).save(folder / "instance.png")
            np.savez_compressed(folder / "depth.npz", z_m=view["depth"], valid=view["depth_valid"])
            annotation = {**view["annotation"], "provenance": record,
                          "source_xml": str(xml_path.relative_to(self.root)),
                          "files": {"rgb": "rgb.png", "depth": "depth.npz", "instances": "instance.png"},
                          "quality": quality[camera]}
            atomic_json(folder / "annotation.json", annotation)
            rgb_hashes[camera] = file_digest(folder / "rgb.png")
            for obj in annotation["objects"]:
                category = categories.setdefault(obj["category"], {"name": obj["category"], "raw_categories": set(), "kinds": set()})
                category["raw_categories"].add(obj["raw_category"])
                category["kinds"].add(obj["kind"])
        hashes = {p.relative_to(local).as_posix(): file_digest(p) for p in sorted(local.rglob("*")) if p.is_file()}
        commit = {"plan_id": self.plan["plan_id"], "profile_id": self.profile_id,
                  "state_record": record, "cameras": list(views), "files_sha256": hashes,
                  "rgb_sha256": rgb_hashes,
                  "categories": [{"name": k, "raw_categories": sorted(v["raw_categories"]), "kinds": sorted(v["kinds"])}
                                 for k, v in sorted(categories.items())],
                  "provenance_xml": {"path": xml_path.relative_to(self.root).as_posix(),
                                     "sha256": file_digest(xml_path)}}
        target = self.root / bundle_relative(record)
        ownership = {"plan_id": self.plan["plan_id"], "state_id": record["state_id"]}
        pending = target / "PENDING.json"
        if target.exists():
            if not pending.exists() or read_json(pending) != ownership:
                raise ValueError(f"Unowned incomplete bundle: {target}")
            # Delete only the explicitly owned, never-committed partial bundle.
            shutil.rmtree(target)
        target.mkdir(parents=True)
        exclusive_json(pending, ownership)
        for name, expected_hash in hashes.items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (local / name).open("rb") as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            if file_digest(destination) != expected_hash:
                raise OSError(f"Payload verification failed: {destination}")
        exclusive_json(target / "COMMIT.json", commit)
        # Keep PENDING as an ownership record; COMMIT is the final visibility gate.
        validate_commit(self.root, record, self.plan["plan_id"], self.profile_id)
        shutil.rmtree(local)
        return commit
