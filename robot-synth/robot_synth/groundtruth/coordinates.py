import itertools

import numpy as np

SIGNS = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1]


def validate_rotation(rotation, tolerance=1e-5):
    value = np.asarray(rotation, dtype=np.float64)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=tolerance)
            or not np.isclose(np.linalg.det(value), 1, atol=tolerance)):
        raise ValueError("Invalid right-handed rotation matrix")
    return value


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = validate_rotation(rotation)
    result[:3, 3] = np.asarray(translation)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite transform")
    return result


def inverse_pose(pose):
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def camera_parameters(position, opengl_rotation, fovy, width, height):
    # Pixel coordinates are continuous edge coordinates: array pixel (v,u)
    # has center (u+0.5,v+0.5). No additional image flips are applied to the
    # modern mujoco.Renderer, which already returns top-left-origin images.
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    world_from_camera = transform(np.asarray(opengl_rotation).reshape(3, 3) @ np.diag([1, -1, -1]), position)
    return K, world_from_camera, inverse_pose(world_from_camera)


def corners(center, size, rotation):
    return (SIGNS * np.asarray(size) / 2) @ np.asarray(rotation).T + center


def project_box(center, size, rotation, K, width, height, near=1e-4):
    points = corners(center, size, rotation)
    clipped = list(points[points[:, 2] >= near])
    for first, second in EDGES:
        a, b = points[first], points[second]
        if (a[2] < near) != (b[2] < near):
            clipped.append(a + (near - a[2]) / (b[2] - a[2]) * (b - a))
    if not clipped:
        return None, None, True
    values = np.asarray(clipped) @ np.asarray(K).T
    uv = values[:, :2] / values[:, 2:]
    lower, upper = uv.min(0), uv.max(0)
    full = [*lower, *upper]
    visible = [max(0., lower[0]), max(0., lower[1]), min(float(width), upper[0]), min(float(height), upper[1])]
    if visible[2] <= visible[0] or visible[3] <= visible[1]:
        visible = None
    truncated = bool((points[:, 2] < near).any() or lower[0] < 0 or lower[1] < 0
                     or upper[0] > width or upper[1] > height)
    return full, visible, truncated


def backproject(depth, K, rows, columns):
    pixels = np.column_stack((columns + 0.5, rows + 0.5, np.ones(len(rows))))
    return (pixels @ np.linalg.inv(K).T) * depth[rows, columns, None]
