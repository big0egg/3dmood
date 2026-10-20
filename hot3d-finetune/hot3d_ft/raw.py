"""Read original Aria CSV GT; decode VRS only for selected RGB timestamps.

All joins use exact HOT3D TIME_CODE timestamps. Device time is looked up in
the provided mapping, not inferred by rounding frame numbers.
"""
import csv
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np

from ioutest.geometry import projected_box, transform_box
from ioutest.hot3d.camera import pinhole_maps, pose, remap
from .common import digest

MASKS = ('mask_qa_pass.csv', 'mask_good_exposure.csv',
         'mask_headset_pose_available.csv', 'mask_object_pose_available.csv',
         'mask_object_visible.csv')


def csv_rows(archive, name):
    with archive.open(name) as stream:
        yield from csv.DictReader(io.TextIOWrapper(stream))


def eligible_times(archive, interval_ns):
    sets = [{int(row['timestamp[ns]']) for row in csv_rows(archive, 'masks/' + name)
             if row['stream_id'] == '214-1' and row['mask'].lower() == 'true'}
            for name in MASKS]
    selected = []
    for timestamp in sorted(set.intersection(*sets)):
        if not selected or timestamp - selected[-1] >= interval_ns:
            selected.append(timestamp)
    return selected


def row_pose(row):
    return pose(dict(translation_xyz=[float(row[f't_wo_{axis}[m]']) for axis in 'xyz'],
                     quaternion_wxyz=[float(row['q_wo_' + axis]) for axis in 'wxyz']))


def triangles(blob):
    """GLB triangle primitives with full scene-node transforms, in meters."""
    magic, version, total = struct.unpack_from('<4sII', blob)
    if magic != b'glTF' or version != 2 or total != len(blob):
        raise ValueError('Invalid GLB')
    cursor = 12
    doc = binary = None
    while cursor < len(blob):
        length, kind = struct.unpack_from('<II', blob, cursor)
        cursor += 8
        chunk = blob[cursor:cursor + length]
        cursor += length
        if kind == 0x4e4f534a:
            doc = json.loads(chunk)
        elif kind == 0x004e4942:
            binary = chunk
    if doc is None or binary is None:
        raise ValueError('GLB missing JSON/BIN')
    types = {5121: 'u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}

    def accessor(index):
        item = doc['accessors'][index]
        if 'sparse' in item:
            raise ValueError('Sparse GLB accessor unsupported')
        count = {'SCALAR': 1, 'VEC3': 3}[item['type']]
        dtype = np.dtype(types[item['componentType']])
        view = doc['bufferViews'][item['bufferView']]
        if view.get('buffer', 0) != 0:
            raise ValueError('External GLB buffer unsupported')
        return np.ndarray((item['count'], count), buffer=binary, dtype=dtype,
                          offset=view.get('byteOffset', 0) + item.get('byteOffset', 0),
                          strides=(view.get('byteStride', count * dtype.itemsize), dtype.itemsize)).copy()

    result = []

    def walk(index, parent, seen):
        from scipy.spatial.transform import Rotation
        if index in seen:
            raise ValueError('Cyclic GLB scene')
        node = doc['nodes'][index]
        if 'matrix' in node:
            local = np.asarray(node['matrix']).reshape(4, 4, order='F')
        else:
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(node.get('rotation', [0, 0, 0, 1])).as_matrix() @ np.diag(node.get('scale', [1, 1, 1]))
            local[:3, 3] = node.get('translation', [0, 0, 0])
        transform = parent @ local
        if 'mesh' in node:
            for primitive in doc['meshes'][node['mesh']]['primitives']:
                if primitive.get('mode', 4) != 4:
                    raise ValueError('Non-triangle mesh unsupported')
                vertices = accessor(primitive['attributes']['POSITION'])
                vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
                indices = accessor(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(len(vertices))
                result.append(vertices[indices.reshape(-1, 3)])
        for child in node.get('children', []):
            walk(child, transform, seen | {index})

    for node in doc['scenes'][doc.get('scene', 0)]['nodes']:
        walk(node, np.eye(4), set())
    if not result:
        raise ValueError('GLB has no triangles')
    return np.concatenate(result)


def amodal_mask(mesh, transform, intrinsics, hw, minimum_area=None):
    """Union of actual mesh triangles; handles clipping at the near plane.

    This CPU rasterization is used only for training-frame validity. Evaluation
    keeps its existing, frozen GT masks and ignore flags.
    """
    import cv2
    height, width = hw
    output = np.zeros(hw, np.uint8)
    points = mesh @ transform[:3, :3].T + transform[:3, 3]
    # The validity rule only asks whether area >=64. Once enough pixels are
    # covered the union can stop: later triangles cannot remove those pixels.
    # Vectorized offscreen rejection also avoids Python loops over most faces.
    front = (points[:, :, 2] >= .001).all(axis=1)
    projected = points @ intrinsics.T
    xy = projected[:, :, :2] / np.maximum(projected[:, :, 2:], .001)
    offscreen = ((xy[:, :, 0].max(axis=1) < 0) | (xy[:, :, 1].max(axis=1) < 0) |
                 (xy[:, :, 0].min(axis=1) >= width) | (xy[:, :, 1].min(axis=1) >= height))
    points = points[~(front & offscreen) & (points[:, :, 2] >= .001).any(axis=1)]
    for number, triangle in enumerate(points):
        polygon = list(triangle)
        clipped = []
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            a, b = start[2] >= .001, end[2] >= .001
            if a:
                clipped.append(start)
            if a != b:
                clipped.append(start + (.001 - start[2]) / (end[2] - start[2]) * (end - start))
        if len(clipped) < 3:
            continue
        p = np.asarray(clipped) @ intrinsics.T
        xy = p[:, :2] / p[:, 2:]
        if (xy[:, 0].max() < 0 or xy[:, 1].max() < 0 or
                xy[:, 0].min() >= width or xy[:, 1].min() >= height):
            continue
        # Fixed-point subpixel coordinates; clip huge offscreen coordinates
        # before conversion to avoid int32 overflow for near-plane triangles.
        xy = np.rint(np.clip(xy, -1e6, 1e6) * 16).astype(np.int32)
        cv2.fillConvexPoly(output, xy, 1, shift=4)
        if minimum_area is not None and (number < 4 or number % 32 == 0):
            if cv2.countNonZero(output) >= minimum_area:
                break
    return output


class RawSequence:
    def __init__(self, directory, cfg, registry, mesh_cache):
        self.directory = Path(directory)
        self.sequence = self.directory.name
        self.cfg = cfg
        self.registry = registry
        self.meshes = mesh_cache
        self.archive = next(self.directory.glob('*ground_truth.zip'))
        with zipfile.ZipFile(self.archive) as archive:
            meta = json.loads(archive.read('metadata.json'))
            if not meta['have_hand_object_pose_gt']:
                raise ValueError(f'{self.sequence} 没有公开物体 GT')
            camera = next(c for c in json.loads(archive.read('camera_models.json')) if c['stream_id'] == '214-1')
            self.calibration = dict(projection_model_type=camera['projectionModelType'],
                                    projection_params=camera['projectionParams'],
                                    image_width=camera['imageWidth'], image_height=camera['imageHeight'],
                                    max_solid_angle=camera['maxSolidAngle'])
            self.device_from_camera = pose(camera['T_Device_Camera'])
            self.times = eligible_times(archive, cfg['candidate_interval_ns'])
            wanted = set(self.times)
            self.headset = {int(r['timestamp[ns]']): row_pose(r)
                            for r in csv_rows(archive, 'headset_trajectory.csv')
                            if int(r['timestamp[ns]']) in wanted}
            self.objects = {}
            for row in csv_rows(archive, 'dynamic_objects.csv'):
                timestamp = int(row['timestamp[ns]'])
                if timestamp in wanted:
                    self.objects.setdefault(timestamp, {})[row['object_uid']] = row_pose(row)
            self.visibility = {}
            for row in csv_rows(archive, 'box2d_objects.csv'):
                timestamp = int(row['timestamp[ns]'])
                if row['stream_id'] == '214-1' and timestamp in wanted:
                    value = float(row['visibility_ratio[%]'])
                    if not 0 <= value <= 1.00001:
                        raise ValueError('HOT3D visibility is a fraction despite the CSV column name')
                    self.visibility[(timestamp, row['object_uid'])] = value
            self.device_times = {int(r['timecode_ns']): int(r['devicetime_ns'])
                                 for r in csv_rows(archive, 'timecode_devicetime_mapping.csv')
                                 if int(r['timecode_ns']) in wanted}
        self.maps = pinhole_maps(self.calibration, cfg['camera'])
        if self.maps[2].mean() < .95:
            raise ValueError('虚拟相机有效像素不足95%')
        self.by_uid = {v['uid']: (key, v) for key, v in registry['objects'].items()}
        self.provider = None

    def labels(self, timestamp):
        if timestamp not in self.headset or timestamp not in self.objects or timestamp not in self.device_times:
            return None
        virtual = np.eye(4)
        virtual[:3, :3] = self.maps[4]
        transform = virtual @ np.linalg.inv(self.headset[timestamp] @ self.device_from_camera)
        frame_id = f'{self.sequence}:{timestamp}'
        boxes = []
        height, width = self.cfg['camera']['height'], self.cfg['camera']['width']
        for uid, world_from_object in sorted(self.objects[timestamp].items()):
            if uid not in self.by_uid:
                continue
            bop, model = self.by_uid[uid]
            camera_from_object = transform @ world_from_object
            box = transform_box(model['local_box'], camera_from_object[:3, :3], camera_from_object[:3, 3])
            box = {k: box[k] for k in ('center', 'size', 'rotation')}
            rectangle = projected_box(box, self.maps[3], width, height)
            visibility = self.visibility.get((timestamp, uid), 0.)
            reasons = []
            area = 0
            if rectangle is None:
                reasons.append('outside_virtual_view')
            elif visibility >= self.cfg['minimum_native_modeled_visibility']:
                area = int(amodal_mask(self.meshes[bop], camera_from_object, self.maps[3], (height, width),
                                       minimum_area=self.cfg['minimum_amodal_pixels']).sum())
            if visibility < self.cfg['minimum_native_modeled_visibility']:
                reasons.append('low_native_modeled_visibility')
            if area < self.cfg['minimum_amodal_pixels']:
                reasons.append('insufficient_virtual_amodal_pixels')
            box.update(uid=digest([frame_id, uid])[:24], object_bop_id=int(bop), object_uid=uid,
                       object_name=model['instance_name'], class_id=model['class_id'],
                       projected_bbox=rectangle, ignore=bool(reasons), ignore_reasons=reasons,
                       virtual_amodal_pixels=area, virtual_amodal_pixels_is_lower_bound=True,
                       native_modeled_visibility=visibility)
            boxes.append(box)
        return boxes

    def image(self, timestamp):
        if self.provider is None:
            from projectaria_tools.core import data_provider
            self.provider = data_provider.create_vrs_data_provider(str(next(self.directory.glob('*main_recording.vrs'))))
            if self.provider is None:
                raise RuntimeError(f'不能打开 VRS：{self.directory}')
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        from projectaria_tools.core.stream_id import StreamId
        expected = self.device_times[timestamp]
        image, record = self.provider.get_image_data_by_time_ns(
            StreamId('214-1'), expected, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
        if abs(record.capture_timestamp_ns - expected) > 1_000_000:
            raise ValueError(f'RGB/GT 同步误差超过1ms：{self.sequence}:{timestamp}')
        array = np.asarray(image.to_numpy_array())
        if array.shape != (self.calibration['image_height'], self.calibration['image_width'], 3):
            raise ValueError(f'VRS RGB 形状异常：{array.shape}')
        return remap(array, self.maps)
