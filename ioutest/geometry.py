"""Full oriented-box geometry on CPU, independent of torch/CUDA/vis4d."""

import itertools

import numpy as np
from scipy.spatial import ConvexHull, QhullError


SIGNS = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
EDGES = np.asarray([(i, j) for i in range(8) for j in range(i + 1, 8)
                    if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1])


def corners(box):
    return (SIGNS * np.asarray(box["size"]) / 2) @ np.asarray(box["rotation"]).T + box["center"]


def inside(points, box, eps=1e-7):
    local = (np.asarray(points) - box["center"]) @ np.asarray(box["rotation"])
    return np.all(np.abs(local) <= np.asarray(box["size"]) / 2 + eps, axis=-1)


def planes(box):
    rotation = np.asarray(box["rotation"])
    normals = np.concatenate((rotation.T, -rotation.T))
    offsets = normals @ np.asarray(box["center"]) + np.tile(np.asarray(box["size"]) / 2, 2)
    return normals, offsets


def _edge_plane_vertices(vertices, other, eps):
    normals, offsets = planes(other)
    starts, ends = vertices[EDGES[:, 0]], vertices[EDGES[:, 1]]
    directions = ends - starts
    denominator = directions @ normals.T
    valid = np.abs(denominator) > eps
    parameter = np.divide(offsets[None] - starts @ normals.T, denominator,
                          out=np.full_like(denominator, np.inf), where=valid)
    mask = valid & (parameter >= -eps) & (parameter <= 1 + eps)
    rows, columns = np.nonzero(mask)
    points = starts[rows] + directions[rows] * parameter[rows, columns, None]
    return points[inside(points, other, eps)]


def box_iou(first, second):
    """Exact convex intersection, including boxes tilted around all three axes."""
    # Translate before hull construction to avoid loss of precision far from origin.
    center = np.asarray(first["center"], dtype=float)
    a = dict(first, center=np.zeros(3))
    b = dict(second, center=np.asarray(second["center"], dtype=float) - center)
    aa, bb = corners(a), corners(b)
    if np.any(np.minimum(aa.max(0), bb.max(0)) <= np.maximum(aa.min(0), bb.min(0))):
        return 0.
    eps = 1e-8 * max(1., float(np.max(a["size"])), float(np.max(b["size"])))
    points = np.concatenate((aa[inside(aa, b, eps)], bb[inside(bb, a, eps)],
                             _edge_plane_vertices(aa, b, eps),
                             _edge_plane_vertices(bb, a, eps)))
    if len(points) < 4:
        return 0.
    points = np.unique(np.round(points, decimals=10), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points - points[0], tol=eps) < 3:
        return 0.
    try:
        intersection = float(ConvexHull(points).volume)
    except QhullError as exc:
        raise ValueError("有向框求交失败，请检查几何数据") from exc
    va, vb = float(np.prod(a["size"])), float(np.prod(b["size"]))
    intersection = min(intersection, va, vb)
    return float(np.clip(intersection / (va + vb - intersection), 0, 1))


def projected_box(box, intrinsics, width, height, near=1e-4):
    """Project an OBB after clipping its edges against the camera near plane."""
    points = corners(box)
    clipped = list(points[points[:, 2] >= near])
    for i, j in EDGES:
        a, b = points[i], points[j]
        if (a[2] < near) != (b[2] < near):
            t = (near - a[2]) / (b[2] - a[2])
            clipped.append(a + t * (b - a))
    if not clipped:
        return None
    points = np.asarray(clipped) @ np.asarray(intrinsics).T
    uv = points[:, :2] / points[:, 2:]
    lo, hi = uv.min(0), uv.max(0)
    if hi[0] < 0 or hi[1] < 0 or lo[0] > width - 1 or lo[1] > height - 1:
        return None
    return [max(0., lo[0]), max(0., lo[1]), min(width - 1., hi[0]), min(height - 1., hi[1])]


def upright_parameters(intrinsics, width, height, direction):
    """Return pixel rotation k for np.rot90, proper camera rotation Q, and K'."""
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    values = {
        "Up": (0, np.eye(3), fx, fy, cx, cy),
        "Left": (-1, np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]),
                 fy, fx, height - 1 - cy, cx),
        "Right": (1, np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]]),
                  fy, fx, cy, width - 1 - cx),
        "Down": (2, np.diag([-1., -1, 1]), fx, fy, width - 1 - cx, height - 1 - cy),
    }
    if direction not in values:
        raise ValueError(f"未知 sky_direction: {direction}")
    k, q, fx, fy, cx, cy = values[direction]
    return k, q, np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


def transform_box(box, rotation, translation):
    out = dict(box)
    out["center"] = (rotation @ np.asarray(box["center"]) + translation).tolist()
    out["rotation"] = (rotation @ np.asarray(box["rotation"])).tolist()
    return out
