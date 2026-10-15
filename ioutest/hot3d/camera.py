"""FISHEYE624 forward projection and fixed pinhole resampling.

Model equations checked against Meta's hand_tracking_toolkit camera.py and
camera_distortion.py (OVR624Distortion / ArctanProjection):
https://github.com/facebookresearch/hand_tracking_toolkit
No inverse lens fitting or OpenCV fisheye approximation is used.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def pose(value):
    q = np.asarray(value['quaternion_wxyz'], dtype=float)
    t = np.asarray(value['translation_xyz'], dtype=float)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError('无效的 HOT3D 位姿')
    if not np.isclose(np.linalg.norm(q), 1, atol=2e-3):
        raise ValueError('HOT3D 四元数不是单位四元数')
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = t
    return out


def project_fisheye(points, calibration):
    """Return source pixels and calibrated angular validity, for Aria RGB."""
    if calibration['projection_model_type'] != 'CameraModelType.FISHEYE624':
        raise ValueError('仅支持已经核验的 Aria FISHEYE624')
    params = np.asarray(calibration['projection_params'], dtype=float)
    if params.shape != (15,) or not np.isfinite(params).all():
        raise ValueError('Aria RGB FISHEYE624 必须有 15 个参数')
    f, cx, cy = params[:3]
    k = params[3:9];p1,p2,s1,s2,s3,s4 = params[9:]
    xyz = np.asarray(points, dtype=float)
    radius = np.linalg.norm(xyz[..., :2], axis=-1)
    theta = np.arctan2(radius, xyz[..., 2])
    angular = xyz[..., :2] * np.divide(theta, radius, out=np.zeros_like(theta), where=radius>1e-15)[..., None]
    r2 = np.minimum(np.sum(angular**2, axis=-1), np.pi**2)
    radial = np.ones_like(r2)
    power = np.ones_like(r2)
    for coefficient in k:
        power = power*r2
        radial += coefficient*power
    x,y = np.moveaxis(angular*radial[..., None], -1, 0)
    a,b,ab = x*x,y*y,x*y
    rr = a+b
    u = x+2*p2*ab+p1*(rr+2*a)+s1*rr+s2*rr**2
    v = y+2*p1*ab+p2*(rr+2*b)+s3*rr+s4*rr**2
    pixels = np.stack((f*u+cx, f*v+cy), axis=-1)
    valid = (np.isfinite(pixels).all(axis=-1) & (xyz[..., 2]>0) &
             (theta<=float(calibration['max_solid_angle'])))
    return pixels, valid


def pinhole_maps(calibration, settings):
    """Directly form upright rays; rotate camera coordinates with the image."""
    width,height = settings['width'],settings['height']
    fov = float(settings['horizontal_fov_degrees'])
    turns = settings.get('clockwise_quarter_turns', 1)
    if width<16 or height<16 or not 10<fov<150 or turns not in (0,1,2,3):
        raise ValueError('无效的虚拟相机参数')
    f = width/(2*np.tan(np.deg2rad(fov)/2))
    intrinsics = np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])
    angle = turns*np.pi/2
    camera_from_raw = Rotation.from_rotvec([0,0,angle]).as_matrix()
    y,x = np.mgrid[:height,:width]
    rays = np.stack(((x-intrinsics[0,2])/f,(y-intrinsics[1,2])/f,np.ones_like(x)),axis=-1)
    raw_rays = rays@camera_from_raw
    pixels,valid = project_fisheye(raw_rays,calibration)
    sw,sh = calibration['image_width'],calibration['image_height']
    valid &= (pixels[...,0]>=0)&(pixels[...,0]<sw-1)&(pixels[...,1]>=0)&(pixels[...,1]<sh-1)
    maps = pixels.astype(np.float32)
    maps[~valid] = -1
    return maps[...,0].copy(),maps[...,1].copy(),valid,intrinsics,camera_from_raw


def remap(image, maps, mask=False):
    import cv2
    x,y,valid,*_ = maps
    result = cv2.remap(np.asarray(image),x,y,cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    result[~valid] = 0
    return result


def decode_mask(value):
    """HOT3D one-based start/length, row-major RLE; not COCO RLE."""
    height,width = int(value['height']),int(value['width'])
    runs = np.asarray(value['rle'],dtype=np.int64)
    if runs.ndim!=1 or len(runs)%2:
        raise ValueError('无效的 HOT3D mask RLE')
    starts,lengths = runs[::2]-1,runs[1::2]
    if (starts<0).any() or (lengths<0).any() or (starts+lengths>height*width).any():
        raise ValueError('HOT3D mask RLE 越界')
    flat = np.zeros(height*width,dtype=np.uint8)
    for start,length in zip(starts,lengths):flat[start:start+length] = 1
    return flat.reshape(height,width)
