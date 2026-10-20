import numpy as np
from PIL import Image, ImageDraw

from ..groundtruth.coordinates import corners, EDGES


def preview(view, path):
    image = Image.fromarray(view["rgb"])
    draw = ImageDraw.Draw(image)
    annotation = view["annotation"]
    K = np.asarray(annotation["intrinsics"])
    height, width = annotation["image_hw"]
    for obj in annotation["objects"]:
        if not obj["visible"] or obj["kind"] in ("robot", "support", "part"):
            continue
        points = corners(obj["center_camera"], obj["size_xyz"], obj["R_camera_from_box"])
        for first, second in EDGES:
            a, b = points[first].copy(), points[second].copy()
            near = 1e-3
            if a[2] < near and b[2] < near:
                continue
            if a[2] < near:
                a += (near - a[2]) / (b[2] - a[2]) * (b - a)
            elif b[2] < near:
                b += (near - b[2]) / (a[2] - b[2]) * (a - b)
            pixels = np.stack((a, b)) @ K.T
            uv = pixels[:, :2] / pixels[:, 2:]
            uv = np.clip(uv, -4 * max(width, height), 5 * max(width, height))
            draw.line([tuple(p) for p in uv], fill="lime", width=2)
        bbox = obj["bbox_visible_xyxy"]
        if bbox:
            draw.text((bbox[0], max(0, bbox[1] - 12)), obj["category"], fill="yellow")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
