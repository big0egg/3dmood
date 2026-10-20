"""Read-only runtime readiness. Never downloads or writes to the OSS mount."""

import ctypes.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

from ..common import atomic_json, file_digest, source_identity
from .asset_paths import resolve_asset


def diagnose(config, inventory=None):
    checks = []
    for name in ("numpy", "h5py", "Pillow", "PyYAML", "mujoco"):
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = None
        expected = config["simulation"]["mujoco_version"] if name == "mujoco" else None
        checks.append({"component": name, "ok": actual is not None and (expected is None or actual == expected),
                       "version": actual, "expected": expected})
    backend = config["simulation"]["gl"]
    library = {"osmesa": "OSMesa", "egl": "EGL", "glfw": "glfw"}[backend]
    found = ctypes.util.find_library(library)
    checks.append({"component": f"rendering:{library}", "ok": bool(found), "library": found})
    for name, path in config["simulation"]["asset_roots"].items():
        checks.append({"component": f"assets:{name}", "ok": Path(path).is_dir(), "path": path})
    missing = []
    if inventory is not None:
        for asset in sorted({p for task in inventory["tasks"] for p in task["asset_files"]}):
            try:
                resolve_asset(asset, config["simulation"]["asset_roots"])
            except (FileNotFoundError, ValueError) as error:
                missing.append(str(error))
        checks.append({"component": "recorded_scene_asset_files", "ok": not missing,
                       "missing_count": len(missing), "examples": missing[:10]})
    result = {"ready": all(item["ok"] for item in checks), "checks": checks,
              "scope": "dependency and asset presence only; collect must still test actual rendering",
              "backend": "mujoco_xml", "network_downloads": False}
    atomic_json(Path(config["output"]["run_dir"]) / "doctor.json", result)
    return result


def runtime_identity(config, inventory):
    packages = {name: version(name) for name in ("numpy", "h5py", "Pillow", "PyYAML", "mujoco")}
    assets = {}
    print("校验运行版本及完整资产内容…", flush=True)
    for task in inventory["tasks"]:
        for original in task["asset_files"]:
            path = resolve_asset(original, config["simulation"]["asset_roots"])
            key = str(path)
            if key not in assets:
                assets[key] = file_digest(path)
    return {"backend": "native_mujoco", "packages": packages, "assets_sha256": assets, "source_sha256": source_identity(),
            "platform": platform.machine(), "rendering_backend": config["simulation"]["gl"]}
