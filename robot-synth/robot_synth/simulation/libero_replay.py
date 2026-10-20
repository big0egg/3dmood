"""Recorded XML + [time,qpos,qvel] replay; no policy or dynamics stepping."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import os

import numpy as np

from ..groundtruth.annotations import annotate_view
from ..groundtruth.geometry import GeometryExtractor
from ..groundtruth.instances import registry
from .asset_paths import resolve_xml
from .cameras import capture


class LiberoReplay:
    def __init__(self, config, xml):
        expected = config["simulation"]["mujoco_version"]
        try:
            actual = version("mujoco")
        except PackageNotFoundError as error:
            raise RuntimeError("MuJoCo is not installed. See README runtime setup; no fake GT will be generated") from error
        if actual != expected:
            raise RuntimeError(f"MuJoCo must be {expected}, found {actual}; revalidation is required for other versions")
        selected_gl = config["simulation"]["gl"]
        if "mujoco" in __import__("sys").modules and os.environ.get("MUJOCO_GL") != selected_gl:
            raise RuntimeError("MuJoCo was already imported with another rendering backend; start a fresh process")
        os.environ["MUJOCO_GL"] = selected_gl
        os.environ["PYOPENGL_PLATFORM"] = selected_gl if selected_gl in ("egl", "osmesa") else "glx"
        import mujoco

        self.mujoco = mujoco
        self.config = config
        self.original_xml_sha256 = hashlib.sha256(xml.encode()).hexdigest()
        self.resolved_xml, self.asset_files = resolve_xml(xml, config["simulation"]["asset_roots"])
        self.model = mujoco.MjModel.from_xml_string(self.resolved_xml)
        if self.model.na != 0:
            raise ValueError("Recorded robosuite state has no actuator activation state; model.na must be zero")
        self.data = mujoco.MjData(self.model)
        self.entries, self.owners = registry(self.model, mujoco, config)
        self.extractor = GeometryExtractor(self.model)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, config["simulation"]["width"])
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, config["simulation"]["height"])
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        for group in config["simulation"]["visual_geom_groups"]:
            self.option.geomgroup[group] = 1
        self.option.sitegroup[:] = 0
        self.renderer = mujoco.Renderer(self.model, height=config["simulation"]["height"],
                                        width=config["simulation"]["width"], max_geom=config["simulation"]["max_geom"])

    def restore(self, state):
        state = np.asarray(state, dtype=np.float64)
        model, data, mj = self.model, self.data, self.mujoco
        if state.shape != (1 + model.nq + model.nv,) or not np.isfinite(state).all():
            raise ValueError(f"State shape {state.shape} differs from compiled model nq={model.nq}, nv={model.nv}")
        mj.mj_resetData(model, data)
        data.time = state[0]
        data.qpos[:] = state[1:1 + model.nq]
        data.qvel[:] = state[1 + model.nq:]
        mj.mj_forward(model, data)

    def render(self, state):
        self.restore(state)
        boxes = [self.extractor.box(self.data, entry) for entry in self.entries]
        views = {}
        for camera in self.config["simulation"]["cameras"]:
            view = capture(self.mujoco, self.model, self.data, self.renderer, self.option, camera, self.owners)
            view["annotation"] = annotate_view(
                self.entries, boxes, view["instance"], view["intrinsics"], view["T_world_from_camera"],
                view["T_camera_from_world"], self.renderer.width, self.renderer.height)
            view["annotation"]["camera"] = camera
            views[camera] = view
        return views

    def close(self):
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None
