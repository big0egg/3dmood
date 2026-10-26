"""Projection drawing helper also used by the training reports."""
import numpy as np
from .geometry import corners, EDGES, projected_box

def _draw(draw, box, intrinsics, width, height, color, scale, text):
    bbox = projected_box(box, intrinsics, width, height)
    if bbox is None:
        return
    points = corners(box)
    near = 1e-4
    for i, j in EDGES:
        a, b = points[i].copy(), points[j].copy()
        if a[2] < near and b[2] < near:
            continue
        if a[2] < near:
            a += (near - a[2]) / (b[2] - a[2]) * (b - a)
        elif b[2] < near:
            b += (near - b[2]) / (a[2] - b[2]) * (a - b)
        projected = np.stack((a, b)) @ np.asarray(intrinsics).T
        uv = projected[:, :2] / projected[:, 2:]
        # Avoid passing astronomical near-plane coordinates to the drawing backend.
        uv = np.clip(uv, -2 * max(width, height), 3 * max(width, height))
        draw.line([tuple(p * scale) for p in uv], fill=color, width=2)
    draw.text((bbox[0] * scale, bbox[1] * scale), text, fill=color)
