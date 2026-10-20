import numpy as np

from ..groundtruth.coordinates import backproject, validate_rotation


def validate_view(view, config):
    annotation = view["annotation"]
    rgb, depth, valid, instance = (view[key] for key in ("rgb", "depth", "depth_valid", "instance"))
    height, width = annotation["image_hw"]
    if (rgb.shape != (height, width, 3) or rgb.dtype != np.uint8 or depth.shape != (height, width)
            or depth.dtype != np.float32 or valid.dtype != np.bool_
            or valid.shape != depth.shape or instance.shape != depth.shape or instance.dtype != np.uint16):
        raise ValueError("RGB/depth/mask shapes or dtypes disagree")
    if not np.isfinite(depth).all() or (depth[valid] <= 0).any() or (depth[~valid] != 0).any():
        raise ValueError("Depth must be finite; valid Z > 0 and invalid Z == 0")
    if np.std(rgb.astype(float)) < 0.1:
        raise ValueError("Blank RGB image")
    K = np.asarray(annotation["intrinsics"])
    world = np.asarray(annotation["T_world_from_camera"])
    camera = np.asarray(annotation["T_camera_from_world"])
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or not np.allclose(world @ camera, np.eye(4), atol=1e-6)):
        raise ValueError("Invalid calibration or inverse transforms")
    known = {obj["instance_id"] for obj in annotation["objects"]} | {0}
    if set(map(int, np.unique(instance))) - known:
        raise ValueError("Segmentation has unknown instance IDs")
    checks = []
    owned = 0
    tolerance = config["validation"]["geometry_tolerance_m"]
    for obj in annotation["objects"]:
        rotation = validate_rotation(obj["R_camera_from_box"])
        validate_rotation(obj["R_world_from_box"])
        center = np.asarray(obj["center_camera"])
        size = np.asarray(obj["size_xyz"])
        if size.shape != (3,) or (size <= 0).any() or not np.isfinite(size).all() or not np.isfinite(center).all():
            raise ValueError("Invalid 3D box")
        restored = world[:3, :3] @ center + world[:3, 3]
        if not np.allclose(restored, obj["center_world"], atol=1e-6):
            raise ValueError("Camera/world box centers do not agree")
        if not np.allclose(world[:3, :3] @ rotation, obj["R_world_from_box"], atol=1e-6):
            raise ValueError("Camera/world box rotations do not agree")
        mask = np.isin(instance, obj["segmentation_ids"])
        rows, columns = np.nonzero(mask)
        if len(rows) != obj["visible_pixels"]:
            raise ValueError("Visible pixel count differs from instance map")
        expected_bbox = [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1] if len(rows) else None
        if expected_bbox != obj["bbox_visible_xyxy"]:
            raise ValueError("Visible 2D box differs from instance mask")
        if obj["kind"] == "object":
            owned += len(rows)
        rows, columns = np.nonzero(mask & valid)
        if obj["visible_pixels"] and len(rows) < .99 * obj["visible_pixels"]:
            raise ValueError("Visible instance mask has missing metric depth")
        chosen = np.linspace(0, len(rows) - 1, min(1024, len(rows)), dtype=int) if len(rows) else []
        rows, columns = rows[chosen], columns[chosen]
        if len(rows):
            points = backproject(depth, K, rows, columns)
            local = (points - center) @ rotation
            violation = np.maximum(np.abs(local) - size / 2, 0).max(axis=1)
            outside = float(np.mean(violation > tolerance))
            if outside > 0.02:
                raise ValueError(f"Depth/mask points outside full geometry box: {obj['body_name']} ({outside:.1%})")
            checks.append({"instance_id": obj["instance_id"], "checked_points": len(rows),
                           "max_box_violation_m": float(violation.max()), "outside_fraction": outside})
    if owned < config["validation"]["minimum_owned_pixels"]:
        raise ValueError(f"Too few visible object pixels: {owned}")
    return {"valid": True, "visible_object_pixels": owned, "depth_box_checks": checks}


def validate_views(views, config):
    if set(views) != set(config["simulation"]["cameras"]):
        raise ValueError("State must contain every configured camera")
    result = {name: validate_view(view, config) for name, view in views.items()}
    first = next(iter(views.values()))["annotation"]["objects"]
    for view in views.values():
        objects = view["annotation"]["objects"]
        if len(objects) != len(first):
            raise ValueError("Cameras disagree on scene instance registry")
        for a, b in zip(first, objects):
            for key in ("center_world", "size_xyz", "R_world_from_box"):
                if not np.allclose(a[key], b[key], atol=1e-6):
                    raise ValueError(f"Multi-camera full object geometry differs: {key}")
    return result
