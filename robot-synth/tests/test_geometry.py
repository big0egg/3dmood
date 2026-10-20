from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from robot_synth.groundtruth.coordinates import backproject, camera_parameters, project_box, validate_rotation
from robot_synth.groundtruth.geometry import GeometryExtractor, primitive_extent
from robot_synth.simulation.asset_paths import resolve_asset, resolve_xml
from robot_synth.validation.geometry import validate_view, validate_views

from conftest import make_view


def test_camera_axes_pixel_centers_and_depth_roundtrip():
    K, world, camera = camera_parameters([1, 2, 3], np.eye(3), 60, 640, 480)
    assert np.allclose(world @ camera, np.eye(4))
    assert np.allclose(world[:3, :3], np.diag([1, -1, -1]))
    assert np.allclose(K[:2, 2], [320, 240])
    depth = np.full((480, 640), 2)
    points = backproject(depth, K, np.array([50, 240]), np.array([40, 320]))
    pixels = points @ K.T
    assert np.allclose(pixels[:, :2] / pixels[:, 2:], [[40.5, 50.5], [320.5, 240.5]])


def test_primitive_exact_bounds():
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    assert np.allclose(primitive_extent(6, [1, 2, 3], rotation), [3 / np.sqrt(2), 3 / np.sqrt(2), 3])
    assert np.allclose(primitive_extent(2, [2, 0, 0], rotation), [2, 2, 2])
    assert np.allclose(primitive_extent(3, [1, 2, 0], rotation), [1, 1, 3])
    assert np.allclose(primitive_extent(5, [1, 2, 0], rotation), [1, 1, 2])
    assert np.allclose(primitive_extent(4, [1, 2, 3], np.eye(3)), [1, 2, 3])
    with pytest.raises(ValueError, match="Unsupported"):
        primitive_extent(0, [1, 1, 1], rotation)


def test_full_mesh_box_center_offset_and_articulated_motion():
    vertices = np.array([[1, 0, 0], [3, 2, 1], [2, 1, 0]])
    model = SimpleNamespace(geom_type=np.array([7]), geom_dataid=np.array([0]), mesh_vertadr=np.array([0]),
                            mesh_vertnum=np.array([3]), mesh_vert=vertices)
    data = SimpleNamespace(xmat=np.eye(3)[None].reshape(1, 9), xpos=np.array([[10., 0, 0]]),
                           geom_xmat=np.eye(3)[None].reshape(1, 9), geom_xpos=np.array([[11., 0, 0]]))
    entry = {"body_id": 0, "geom_ids": [0], "body_name": "mesh"}
    extractor = GeometryExtractor(model)
    box = extractor.box(data, entry)
    assert np.allclose(box["center_world"], [13, 1, .5])
    assert np.allclose(box["size_xyz"], [2, 2, 1])
    data.geom_xpos[0, 0] += 1
    changed = extractor.box(data, entry)
    assert np.allclose(changed["center_world"], [14, 1, .5])


def test_near_plane_and_behind_camera():
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    full, visible, truncated = project_box([0, 0, -3], [1, 1, 1], np.eye(3), K, 100, 100)
    assert full is visible is None
    full, visible, truncated = project_box([0, 0, .2], [1, 1, 1], np.eye(3), K, 100, 100)
    assert np.isfinite(full).all() and truncated and visible == [0, 0, 100, 100]
    with pytest.raises(ValueError):
        validate_rotation(np.diag([1, 1, -1]))


def test_asset_remapping_and_missing_files(tmp_path):
    roots = {"libero": tmp_path / "libero", "robosuite": tmp_path / "robosuite"}
    mesh = roots["libero"] / "objects/a.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("mesh")
    texture = roots["robosuite"] / "textures/t.png"
    texture.parent.mkdir(parents=True)
    texture.write_text("texture")
    assert resolve_asset('/home/old/chiliocosm/assets/objects/a.obj', roots) == mesh
    assert resolve_asset('/old/robosuite/models/assets/textures/t.png', roots) == texture
    xml, mapping = resolve_xml('<mujoco><compiler meshdir="old"/><asset><mesh file="/old/assets/objects/a.obj"/></asset></mujoco>', roots)
    assert str(mesh) in xml and 'meshdir=' not in xml and len(mapping) == 1
    with pytest.raises(FileNotFoundError):
        resolve_asset('/old/assets/objects/missing.obj', roots)


@pytest.mark.parametrize('kind', ['depth', 'mask', 'rotation', 'box', 'bbox', 'world', 'blank'])
def test_geometry_audits_reject_wrong_groundtruth(config, kind):
    view = make_view()
    assert validate_view(view, config)["valid"]
    obj = view['annotation']['objects'][0]
    if kind == 'depth':
        view['depth'][8:24, 8:24] = 20
    elif kind == 'mask':
        view['instance'][0, 0] = 9
    elif kind == 'rotation':
        obj['R_camera_from_box'][0][0] = -1
    elif kind == 'box':
        obj['size_xyz'][0] = -1
    elif kind == 'bbox':
        obj['bbox_visible_xyxy'][0] = 0
    elif kind == 'world':
        obj['center_world'][0] = 10
    else:
        view['rgb'][:] = 0
    with pytest.raises(ValueError):
        validate_view(view, config)


def test_camera_pair_must_match(config):
    with pytest.raises(ValueError, match="every configured"):
        validate_views({'agentview': make_view()}, config)


def test_hierarchical_instance_masks(config):
    from robot_synth.groundtruth.instances import registry
    model = SimpleNamespace(nbody=3, ngeom=2, body_parentid=np.array([0, 0, 1]),
                            body_jntnum=np.array([0, 1, 1]), geom_group=np.array([1, 1]),
                            geom_bodyid=np.array([1, 2]))
    names = {0: 'world', 1: 'box_1_main', 2: 'box_1_drawer'}
    mj = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1), mj_id2name=lambda m, t, i: names[i])
    entries, owners = registry(model, mj, config)
    assert owners.tolist() == [1, 2]
    assert entries[0]['segmentation_ids'] == [1, 2]
    assert entries[1]['parent_instance_id'] == 1
    assert entries[1]['category'] == 'articulated_part'


def test_native_camera_contract_without_initializing_runtime():
    from robot_synth.simulation.cameras import capture
    class Renderer:
        width = height = 4
        def disable_depth_rendering(self): self.mode = 'rgb'
        def disable_segmentation_rendering(self): self.mode = 'rgb'
        def enable_depth_rendering(self): self.mode = 'depth'
        def enable_segmentation_rendering(self): self.mode = 'seg'
        def update_scene(self, *args, **kwargs): pass
        def render(self):
            if self.mode == 'rgb': return np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
            if self.mode == 'depth': return np.full((4, 4), 3, dtype=np.float32)
            value = np.full((4, 4, 2), -1, dtype=np.int32)
            value[1, 2] = [0, 6]  # native (geom ID, object type)
            return value
    mj = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_CAMERA=7, mjOBJ_GEOM=6), mj_name2id=lambda *args: 0)
    model = SimpleNamespace(ncam=1, cam_fovy=np.array([60]))
    data = SimpleNamespace(cam_xpos=np.array([[0., 0, 0]]), cam_xmat=np.eye(3)[None].reshape(1, 9))
    view = capture(mj, model, data, Renderer(), None, 'agentview', np.array([2], dtype=np.uint16))
    assert view['instance'][1, 2] == 2
    assert view['depth'][1, 2] == 3 and view['depth'][0, 0] == 0
    assert view['rgb'][0, 0, 0] == 0  # no duplicate flip of native top-down output
