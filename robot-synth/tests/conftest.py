from copy import deepcopy
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from robot_synth.config import load_config
from robot_synth.groundtruth.annotations import annotate_view


def scene_xml(size=0.1):
    return f'''<mujoco><worldbody><camera name="agentview"/><body name="box_1_main">
    <freejoint/><geom name="visual" group="1" type="box" size="{size} {size} {size}"/>
    </body></worldbody></mujoco>'''


@pytest.fixture
def config(tmp_path):
    value = load_config(Path(__file__).parents[1] / "configs/libero_gt_v1.yaml")
    value["source"]["root"] = str(tmp_path / "raw")
    value["source"]["suites"] = ["libero_90", "libero_10"]
    value["output"] = {"root": str(tmp_path / "data"), "run_dir": str(tmp_path / "run"), "scratch": str(tmp_path / "scratch")}
    value["simulation"].update(width=32, height=32)
    value["sampling"]["eval_scenes"] = {"KITCHEN": 1, "LIVING_ROOM": 1, "STUDY": 1}
    value["sampling"]["quotas"] = {"train": {"KITCHEN": 12, "LIVING_ROOM": 12, "STUDY": 12},
                                   "eval": {"KITCHEN": 4, "LIVING_ROOM": 4, "STUDY": 4}}
    value["sampling"]["episodes_per_scene"] = {"train": 3, "eval": 2}
    value["sampling"]["min_qpos_distance"] = .005
    value["validation"]["minimum_owned_pixels"] = 1
    for index, room in enumerate(("KITCHEN", "LIVING_ROOM", "STUDY")):
        for number in (1, 2):
            path = Path(value["source"]["root"]) / "libero_90" / f"{room}_SCENE{number}_put_box_demo.hdf5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as file:
                data = file.create_group("data")
                data.attrs.update(env_args='{"env_kwargs": {}}', num_demos=3, total=60, tag="unit-test")
                for episode in range(3):
                    group = data.create_group(f"demo_{episode}")
                    group.attrs["model_file"] = scene_xml(.1 + (index * 2 + number) / 100)
                    states = np.zeros((20, 14))
                    states[:, 0] = np.arange(20) * .05
                    states[:, 1] = np.arange(20) * .01 + episode * .001
                    states[:, 3] = .5
                    states[:, 4] = 1
                    group.create_dataset("states", data=states)
                    group.create_dataset("actions", data=np.zeros((20, 7)))
                    group.create_group("obs").create_dataset("agentview_rgb", data=np.zeros((20, 32, 32, 3), dtype=np.uint8))
    return value


def make_view(camera="agentview", salt=b"test"):
    rgb = np.tile(np.arange(32, dtype=np.uint8)[None, :, None], (32, 1, 3))
    # Embed identity in pixels, making fake test images exactly reproducible.
    rgb[0, :, 0] = np.frombuffer(hashlib.sha256(salt + camera.encode()).digest(), dtype=np.uint8)
    instance = np.zeros((32, 32), dtype=np.uint16)
    instance[8:24, 8:24] = 1
    depth = np.full((32, 32), 2, dtype=np.float32)
    valid = np.ones((32, 32), dtype=bool)
    K = np.array([[32, 0, 16], [0, 32, 16], [0, 0, 1]], dtype=float)
    entry = {"instance_id": 1, "body_id": 1, "body_name": "test_box", "raw_category": "box", "category": "box",
             "kind": "object", "parent_instance_id": None, "body_ids": [1], "geom_ids": [0], "segmentation_ids": [1]}
    box = {"center_world": [0, 0, 2], "center_body": [0, 0, 0], "size_xyz": [1.5, 1.5, 1], "R_world_from_box": np.eye(3).tolist()}
    annotation = annotate_view([entry], [box], instance, K, np.eye(4), np.eye(4), 32, 32)
    annotation["camera"] = camera
    return {"rgb": rgb, "instance": instance, "depth": depth, "depth_valid": valid, "annotation": annotation}


class FakeReplay:
    """Test fixture only; no CLI option can select this backend."""
    loads = 0
    closes = 0

    def __init__(self, config, xml):
        self.config, self.xml = config, xml
        type(self).loads += 1

    def render(self, state):
        salt = self.xml.encode() + state.tobytes()
        return {camera: make_view(camera, salt) for camera in self.config["simulation"]["cameras"]}

    def close(self):
        type(self).closes += 1
