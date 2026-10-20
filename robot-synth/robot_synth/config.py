from copy import deepcopy
from pathlib import Path

from .common import digest


def load_config(path, *, input_root=None, output_root=None):
    import yaml

    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Configuration must be a schema_version: 1 mapping")
    for section in ("source", "output", "sampling", "simulation", "groundtruth", "validation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    for section, key in (("source", "root"), ("output", "root"),
                         ("output", "run_dir"), ("output", "scratch")):
        value = Path(config[section][key]).expanduser()
        config[section][key] = str((path.parent / value).resolve())
    for name, value in config["simulation"].get("asset_roots", {}).items():
        config["simulation"]["asset_roots"][name] = str((path.parent / Path(value).expanduser()).resolve())
    if input_root is not None:
        config["source"]["root"] = str(Path(input_root).expanduser().resolve())
    if output_root is not None:
        config["output"]["root"] = str(Path(output_root).expanduser().resolve())
    roots = [Path(config["source"]["root"])] + [Path(v) for v in config["output"].values()]
    if len(roots) != 4:
        raise ValueError("output accepts exactly root, run_dir and scratch")
    for i, first in enumerate(roots):
        for second in roots[i + 1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError(f"Source, output, run_dir and scratch must not contain each other: {first}, {second}")
    cameras = config["simulation"]["cameras"]
    suites = config["source"]["suites"]
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("Source suites must be nonempty and distinct")
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("Camera names must be nonempty and distinct")
    import re
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in cameras):
        raise ValueError("Unsafe camera name")
    for key in ("width", "height"):
        value = config["simulation"][key]
        if not isinstance(value, int) or not 32 <= value <= 4096:
            raise ValueError(f"Invalid image {key}: {value}")
    if config["simulation"]["gl"] not in ("osmesa", "egl", "glfw"):
        raise ValueError("simulation.gl must be osmesa, egl or glfw")
    groups = config["simulation"]["visual_geom_groups"]
    if not groups or any(not isinstance(g, int) or not 0 <= g < 6 for g in groups):
        raise ValueError("visual_geom_groups must contain MuJoCo group IDs 0..5")
    for split in ("train", "eval"):
        if config["sampling"]["episodes_per_scene"][split] < 1:
            raise ValueError("episodes_per_scene must be positive")
        for room, count in config["sampling"]["quotas"][split].items():
            if not isinstance(count, int) or count <= 0 or count % len(cameras):
                raise ValueError(f"{split}/{room} image quota must be positive and divisible by camera count")
    if not 0 <= config["sampling"]["reserve_fraction"] <= 3:
        raise ValueError("reserve_fraction must be between 0 and 3")
    if config["sampling"]["min_qpos_distance"] < 0:
        raise ValueError("min_qpos_distance cannot be negative")
    if config["validation"]["max_failures"] < 1:
        raise ValueError("max_failures must be positive")
    return config


def acquisition_config(config):
    """Run location may change; data generation parameters may not."""
    value = deepcopy(config)
    value["output"] = {"root": value["output"]["root"]}
    return value


def config_id(config):
    return digest(acquisition_config(config))
