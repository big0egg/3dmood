import numpy as np

from .coordinates import project_box


def annotate_view(entries, boxes, instance_map, K, world_from_camera, camera_from_world, width, height):
    records = []
    for entry, box in zip(entries, boxes):
        center = camera_from_world[:3, :3] @ np.asarray(box["center_world"]) + camera_from_world[:3, 3]
        rotation = camera_from_world[:3, :3] @ np.asarray(box["R_world_from_box"])
        mask = np.isin(instance_map, entry["segmentation_ids"])
        rows, columns = np.nonzero(mask)
        visible = len(rows)
        bbox = [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1] if visible else None
        projected, clipped, truncated = project_box(center, box["size_xyz"], rotation, K, width, height)
        records.append({**entry, **box, "center_camera": center.tolist(), "R_camera_from_box": rotation.tolist(),
                        "visible_pixels": visible, "visible": bool(visible),
                        "bbox_visible_xyxy": bbox, "bbox_projected_xyxy": projected,
                        "bbox_projected_clipped_xyxy": clipped, "box_truncated": truncated,
                        "occlusion_fraction": None,
                        "box_definition": "full visual geometry extent in current root-body axes"})
    return {"schema_version": "robot_synth.gt.v1", "groundtruth_source": "simulation_state_and_visual_geometry",
            "image_hw": [height, width], "length_unit": "meter", "camera_coordinates": "opencv",
            "world_coordinates": "mujoco_z_up", "pixel_coordinates": "top_left_edge_coordinates_pixel_centers_at_half_integers",
            "bbox_convention": "xyxy_right_bottom_exclusive", "intrinsics": np.asarray(K).tolist(),
            "T_world_from_camera": np.asarray(world_from_camera).tolist(),
            "T_camera_from_world": np.asarray(camera_from_world).tolist(),
            "depth_convention": "camera_z_meters", "background_instance_id": 0,
            "background_policy": "unannotated_world_scenery_and_infinite_planes_are_0_not_semantic_object_groundtruth",
            "objects": records}
