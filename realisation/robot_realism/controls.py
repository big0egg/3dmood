import numpy as np
from PIL import Image


def depth_control(depth, valid, near, far):
    """Near-white inverse depth hint. Original metric GT is never overwritten."""
    if depth.ndim != 2 or valid.shape != depth.shape or not np.isfinite(depth).all():
        raise ValueError('Invalid source depth/valid arrays')
    if not valid.any() or (depth[valid] <= 0).any():
        raise ValueError('Depth hint requires positive valid metric depth')
    value = np.zeros(depth.shape, dtype=np.float32)
    clipped = np.clip(depth[valid], near, far)
    inverse = 1 / clipped
    low, high = float(inverse.min()), float(inverse.max())
    value[valid] = (inverse - low) / (high - low) if high - low > 1e-8 else 1.0
    gray = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def prepare_camera(source, target, config):
    target.mkdir(parents=True, exist_ok=True)
    with Image.open(source / 'rgb.png') as image:
        rgb = np.asarray(image.convert('RGB'))
    result = {}
    if 'depth' in config['generation']['controls']:
        with np.load(source / 'depth.npz', allow_pickle=False) as values:
            depth, valid = values['z_m'], values['valid']
        if depth.shape != rgb.shape[:2]:
            raise ValueError('Source RGB/depth sizes disagree')
        hint = depth_control(depth, valid, config['controls']['near_m'], config['controls']['far_m'])
        p = target / 'depth.png'
        Image.fromarray(hint).save(p)
        result['depth'] = str(p)
    if 'edge' in config['generation']['controls']:
        import cv2
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hint = cv2.Canny(gray, config['controls']['canny_low'], config['controls']['canny_high'])
        p = target / 'edge.png'
        Image.fromarray(np.repeat(hint[..., None], 3, axis=2)).save(p)
        result['edge'] = str(p)
    return result
