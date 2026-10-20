"""A bundle is usable only after its final COMMIT.json and hashes validate."""

from pathlib import Path

from ..common import file_digest, read_json


def bundle_relative(record):
    return Path(record["split"]) / record["scene_id"] / record["state_id"]


def validate_commit(root, record, plan_id, profile_id):
    folder = Path(root) / bundle_relative(record)
    marker = folder / "COMMIT.json"
    if not marker.exists():
        return None
    commit = read_json(marker)
    if (commit.get("plan_id") != plan_id or commit.get("profile_id") != profile_id
            or commit.get("state_record") != record):
        raise ValueError(f"Existing bundle belongs to a different run: {folder}")
    cameras = commit.get("cameras", [])
    expected = {"state.npy"} | {f"{camera}/{name}" for camera in cameras
                                   for name in ("rgb.png", "instance.png", "depth.npz", "annotation.json")}
    if not cameras or len(set(cameras)) != len(cameras) or set(commit.get("files_sha256", {})) != expected:
        raise ValueError(f"Incomplete payload manifest: {folder}")
    for name, expected in commit["files_sha256"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Committed bundle is missing/corrupt: {path}")
    provenance = commit.get("provenance_xml", {})
    path = (Path(root) / provenance.get("path", "")).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or file_digest(path) != provenance.get("sha256"):
        raise ValueError(f"Episode XML provenance is missing/corrupt: {path}")
    return commit
