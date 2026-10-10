import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ioutest.geometry import box_iou, corners, projected_box, transform_box, upright_parameters
from ioutest.metrics import evaluate_ap
from ioutest.schema import validate_box


def box(center=(0., 0., 2.), size=(2., 2., 2.), rotation=None, cid=13, score=None, ignore=False):
    out = {"center": list(center), "size": list(size), "rotation": (
        np.eye(3) if rotation is None else rotation).tolist(), "class_id": cid, "ignore": ignore}
    if score is not None:
        out["score"] = score
    return out


def test_iou_axis_aligned_analytic():
    a = box()
    assert box_iou(a, a) == pytest.approx(1.)
    assert box_iou(a, box(center=(1., 0., 2.))) == pytest.approx(1 / 3)
    assert box_iou(a, box(center=(2., 0., 2.))) == 0.
    assert box_iou(a, box(center=(4., 0., 2.))) == 0.
    assert box_iou(a, box(size=(1., 1., 1.))) == pytest.approx(1 / 8)
    tilted = box(rotation=Rotation.from_euler("z", 45, degrees=True).as_matrix())
    assert box_iou(a, tilted) == pytest.approx(2 ** -.5, abs=1e-8)


def test_iou_full_rotation_and_rigid_invariance():
    a, b = box(size=(2., 1., 3.)), box(center=(.4, .1, 2.2), size=(1., 2., 2.))
    angle = Rotation.from_euler("xyz", [31, 48, -27], degrees=True).as_matrix()
    offset = np.array([100., -40., 300.])
    moved_a, moved_b = transform_box(a, angle, offset), transform_box(b, angle, offset)
    assert box_iou(moved_a, moved_a) == pytest.approx(1., abs=1e-8)
    assert box_iou(moved_a, moved_b) == pytest.approx(box_iou(a, b), abs=1e-8)
    tilted = box(rotation=angle, size=(1., 2., 3.))
    assert box_iou(tilted, a) == pytest.approx(box_iou(a, tilted), abs=1e-8)
    assert 0 < box_iou(a, tilted) < 1


@pytest.mark.parametrize("direction,k_expected", [("Up", 0), ("Left", -1), ("Right", 1), ("Down", 2)])
def test_upright_projection_matches_pixel_rotation(direction, k_expected):
    w, h = 256, 192
    intrinsic = np.array([[210., 0., 121.], [0., 220., 93.], [0., 0., 1.]])
    pixels = np.array([[0., 0.], [100., 80.], [w - 1., h - 1.]])
    points = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(intrinsic).T * 2
    k, q, upright_k = upright_parameters(intrinsic, w, h, direction)
    transformed = points @ q.T @ upright_k.T
    actual = transformed[:, :2] / transformed[:, 2:]
    expected = {
        "Up": pixels,
        "Left": np.c_[h - 1 - pixels[:, 1], pixels[:, 0]],
        "Right": np.c_[pixels[:, 1], w - 1 - pixels[:, 0]],
        "Down": np.c_[w - 1 - pixels[:, 0], h - 1 - pixels[:, 1]],
    }[direction]
    assert k == k_expected
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert np.linalg.det(q) == pytest.approx(1)


def test_near_plane_clipping():
    intrinsic = np.array([[50., 0., 50.], [0., 50., 50.], [0., 0., 1.]])
    assert projected_box(box(center=(0, 0, -3)), intrinsic, 100, 100) is None
    assert projected_box(box(center=(0, 0, .3)), intrinsic, 100, 100) == [0., 0., 99., 99.]


def test_ap_perfect_duplicate_and_wrong_class():
    gt = {"frame": [box()]}
    pred = {"frame": [box(score=.9), box(score=.8)]}
    result = evaluate_ap(gt, pred)
    assert result["summary"]["iou"]["map"] == 1.
    assert result["per_class"][13]["iou"]["0.25"]["fp"] == 1
    wrong = evaluate_ap(gt, {"frame": [box(cid=0, score=.9)]})
    assert wrong["summary"]["center"]["map"] == 0
    assert wrong["per_class"][0]["center"]["0.5"]["ap"] is None
    assert wrong["per_class"][0]["center"]["0.5"]["fp"] == 1


def test_empty_predictions_and_negative_frames():
    gt = {"a": [box()], "negative": []}
    assert evaluate_ap(gt, {"a": [], "negative": []})["summary"]["iou"]["map"] == 0
    result = evaluate_ap(gt, {"a": [box(score=.8)], "negative": [box(score=.9)]})
    assert result["summary"]["iou"]["map"] == pytest.approx(.5)


def test_distance_is_absolute_meters_and_nearest_matching():
    gt = {"f": [box(center=(0, 0, 3), size=(20, 20, 20))]}
    pred = {"f": [box(center=(.3, 0, 3), score=.9)]}
    row = evaluate_ap(gt, pred)["per_class"][13]
    assert row["center"]["0.25"]["ap"] == 0
    assert row["center"]["0.5"]["ap"] == 1
    # First prediction must take the nearest GT, leaving the other for prediction two.
    gt = {"f": [box(center=(0, 0, 3)), box(center=(.4, 0, 3))]}
    pred = {"f": [box(center=(.1, 0, 3), score=.9), box(center=(.5, 0, 3), score=.8)]}
    assert evaluate_ap(gt, pred)["per_class"][13]["center"]["0.25"]["tp"] == 2


def test_ignore_is_one_to_one_and_valid_first():
    gt = {"f": [box(), box(center=(4, 0, 2), ignore=True)]}
    pred = {"f": [box(center=(4, 0, 2), score=.99), box(score=.9)]}
    row = evaluate_ap(gt, pred)["per_class"][13]["iou"]["0.25"]
    assert row["ap"] == 1 and row["ignored_predictions"] == 1
    gt = {"f": [box(), box(ignore=True)]}
    row = evaluate_ap(gt, {"f": [box(score=.9)]})["per_class"][13]["iou"]["0.25"]
    assert row["tp"] == 1 and row["ignored_predictions"] == 0


def test_equal_scores_stable_and_incomplete_rejected():
    gt = {"a": [], "b": [box()]}
    p1 = {"b": [box(score=.5)], "a": [box(score=.5)]}
    p2 = {"a": [box(score=.5)], "b": [box(score=.5)]}
    assert evaluate_ap(gt, p1) == evaluate_ap(gt, p2)
    with pytest.raises(ValueError, match="不完整"):
        evaluate_ap(gt, {"b": []})


def test_invalid_box_and_no_gt_summary():
    with pytest.raises(ValueError, match="正数"):
        validate_box(box(size=(0, 1, 1)))
    bad = box(); bad["rotation"][0][0] = -1
    with pytest.raises(ValueError, match="右手"):
        validate_box(bad)
    assert evaluate_ap({"f": []}, {"f": []})["summary"]["iou"]["map"] is None


def test_vis4d_size_conversion_matches_actual_corner_geometry():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vis4d")
    from vis4d.data.const import AxisMode
    from vis4d.op.box.box3d import boxes3d_to_corners
    from vis4d.op.geometry.rotation import quaternion_to_matrix
    data = torch.tensor([[.2, -.4, 3., .7, 1.2, 2.1, .9, .1, .2, .3]], dtype=torch.float64)
    data[:, 6:] /= torch.linalg.norm(data[:, 6:], dim=-1, keepdim=True)
    rotation = quaternion_to_matrix(data[:, 6:])[0].numpy()
    canonical = box(center=data[0, :3].numpy(), size=data[0, [4, 5, 3]].numpy(), rotation=rotation)
    expected = boxes3d_to_corners(data, AxisMode.OPENCV)[0].numpy()
    actual = corners(canonical)
    distances = np.linalg.norm(expected[:, None] - actual[None], axis=-1)
    assert distances.min(axis=1).max() < 1e-10


def test_independent_cpu_iou_against_existing_vis4d_operator():
    torch = pytest.importorskip("torch")
    ops = pytest.importorskip("vis4d_cuda_ops")
    rng = np.random.default_rng(42)
    order = [0, 4, 6, 2, 1, 5, 7, 3]
    for _ in range(8):
        a = box(center=rng.normal(size=3), size=rng.uniform(.5, 3, size=3),
                rotation=Rotation.random(random_state=rng).as_matrix())
        b = box(center=rng.normal(size=3), size=rng.uniform(.5, 3, size=3),
                rotation=Rotation.random(random_state=rng).as_matrix())
        tensors = [torch.tensor(corners(x)[order][None], dtype=torch.float32) for x in (a, b)]
        try:
            reference = ops.iou_box3d(*tensors)[1].item()
        except RuntimeError as exc:
            pytest.skip(f"当前环境的 vis4d IoU 无 CPU 支持: {exc}")
        assert box_iou(a, b) == pytest.approx(reference, abs=2e-5)
