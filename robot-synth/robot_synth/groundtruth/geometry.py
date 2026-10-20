"""Bounds of full rendered geometry, not visible depth points/collision boxes."""

import numpy as np


def primitive_extent(kind, size, rotation):
    """Exact support bounds along the requested box axes (MuJoCo type IDs)."""
    size = np.asarray(size)
    if kind == 2:  # sphere
        return np.full(3, size[0])
    if kind == 3:  # capsule: radius, half cylinder length, aligned with local z
        return size[0] + size[1] * np.abs(rotation[:, 2])
    if kind == 4:  # ellipsoid
        return np.sqrt(((rotation * size) ** 2).sum(axis=1))
    if kind == 5:  # cylinder
        return size[0] * np.linalg.norm(rotation[:, :2], axis=1) + size[1] * np.abs(rotation[:, 2])
    if kind == 6:  # box sizes are half extents
        return np.abs(rotation) @ size
    raise ValueError(f"Unsupported finite geom type {kind}; no approximate GT is emitted")


class GeometryExtractor:
    def __init__(self, model):
        self.model = model
        self.meshes = {}
        self.mesh_bounds_cache = {}

    def mesh_bounds(self, geom, rotation):
        mesh = int(self.model.geom_dataid[geom])
        if mesh not in self.meshes:
            start = int(self.model.mesh_vertadr[mesh])
            count = int(self.model.mesh_vertnum[mesh])
            self.meshes[mesh] = np.asarray(self.model.mesh_vert[start:start + count], dtype=np.float64)
        # Most body-relative mesh rotations are constant over a replay. Rounded
        # rotations bound numerical FK noise without changing metric geometry.
        key = (mesh, tuple(np.round(rotation, 10).flat))
        if key not in self.mesh_bounds_cache:
            points = self.meshes[mesh] @ rotation.T
            self.mesh_bounds_cache[key] = (points.min(0), points.max(0))
            if len(self.mesh_bounds_cache) > 20000:
                self.mesh_bounds_cache.clear()
                return points.min(0), points.max(0)
        return self.mesh_bounds_cache[key]

    def box(self, data, entry):
        body = entry["body_id"]
        world_rotation = np.asarray(data.xmat[body]).reshape(3, 3)
        world_origin = np.asarray(data.xpos[body])
        lows, highs = [], []
        for geom in entry["geom_ids"]:
            rotation = world_rotation.T @ np.asarray(data.geom_xmat[geom]).reshape(3, 3)
            offset = world_rotation.T @ (data.geom_xpos[geom] - world_origin)
            kind = int(self.model.geom_type[geom])
            if kind == 7:
                low, high = self.mesh_bounds(geom, rotation)
            else:
                extent = primitive_extent(kind, self.model.geom_size[geom], rotation)
                low, high = -extent, extent
            lows.append(low + offset)
            highs.append(high + offset)
        low, high = np.min(lows, 0), np.max(highs, 0)
        size = high - low
        if not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError(f"Degenerate full-geometry box: {entry['body_name']}")
        center_local = (low + high) / 2
        return {"center_world": (world_rotation @ center_local + world_origin).tolist(),
                "center_body": center_local.tolist(), "size_xyz": size.tolist(),
                "R_world_from_box": world_rotation.tolist()}
