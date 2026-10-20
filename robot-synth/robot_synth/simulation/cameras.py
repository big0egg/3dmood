"""MuJoCo 3.2.7 Renderer outputs top-left-origin RGB and metric Z depth."""

import numpy as np

from ..groundtruth.coordinates import camera_parameters


def capture(mujoco, model, data, renderer, scene_option, camera_name, owners):
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera < 0:
        raise ValueError(f"Recorded model has no camera {camera_name}")
    if getattr(model, "cam_orthographic", np.zeros(model.ncam))[camera]:
        raise ValueError("Orthographic cameras are not supported by this pinhole GT schema")
    if hasattr(model, "cam_sensorsize") and np.any(model.cam_sensorsize[camera] > 0):
        raise ValueError("Physical-sensor camera intrinsics require a separate calibration adapter")
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name, scene_option=scene_option)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    # Native Renderer returns (object ID, object type), unlike robosuite's old
    # low-level read_pixels helper which returns (type, ID).
    ids, types = segmentation[..., 0], segmentation[..., 1]
    geoms = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0) & (ids < len(owners))
    instance = np.zeros(ids.shape, dtype=np.uint16)
    instance[geoms] = owners[ids[geoms]]
    valid = geoms & np.isfinite(depth) & (depth > 0)
    depth[~valid] = 0
    K, world_from_camera, camera_from_world = camera_parameters(
        data.cam_xpos[camera], data.cam_xmat[camera], model.cam_fovy[camera], renderer.width, renderer.height)
    return {"rgb": rgb, "depth": depth.astype(np.float32), "depth_valid": valid,
            "instance": instance, "intrinsics": K, "T_world_from_camera": world_from_camera,
            "T_camera_from_world": camera_from_world}
