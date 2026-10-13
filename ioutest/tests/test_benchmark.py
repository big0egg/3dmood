"""Protocol, RGB-D alignment, resume and report completeness tests."""

import copy
import hashlib

import numpy as np
import pytest
from PIL import Image

from ioutest.benchmark import select_frames
from ioutest.benchmark_report import assemble_report, evaluate_model, refresh_report
from ioutest.benchmark_worker import prediction_path, read_prediction, work, write_prediction
from ioutest.common import digest, read_json, write_json
from ioutest.dataset import load_benchmark
from ioutest.depth_cache import DepthInputReader, cache_depth, camera_gravity
from ioutest.image_cache import cache_images
from ioutest.metrics import evaluate_ap
from ioutest.report_document import report_context
from ioutest.schema import Detection
from ioutest.tests.test_geometry_metrics import box
from ioutest.tests.test_pipeline import prepared


def test_geometry_unknown_labels_and_semantic_separation():
    gt = {"a": [box(cid=13)]}
    prediction = box(cid=None, score=.9)
    before = copy.deepcopy(prediction)
    result = evaluate_ap(gt, {"a": [prediction]}, class_agnostic=True)
    assert result["summary"]["iou"]["map"] == pytest.approx(1)
    assert result["summary"]["center"]["map"] == pytest.approx(1)
    assert prediction == before
    with pytest.raises(ValueError, match="class_id"):
        evaluate_ap(gt, {"a": [prediction]})
    prediction["class_id"] = 14
    assert evaluate_ap(gt, {"a": [prediction]})["summary"]["iou"]["map"] == 0
    assert evaluate_ap(gt, {"a": [prediction]}, class_agnostic=True)["summary"]["iou"]["map"] == 1


def test_preflight_uses_project_specific_dependencies(tmp_path, monkeypatch):
    from ioutest.benchmark import preflight
    from types import SimpleNamespace
    marker = tmp_path / "exists"
    marker.touch()
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='{"torch":"test","cuda_available":true,"devices":1}', stderr="")
    monkeypatch.setattr("ioutest.benchmark.subprocess.run", fake_run)
    model = {"python": str(marker), "adapter": "ioutest.adapters.cutr:CuTRAdapter",
             "options": {"project_root": str(tmp_path), "checkpoint": str(marker)}}
    assert preflight({"cutr-rgbd": model}, {}, "cuda")["cutr-rgbd"]["devices"] == 1
    assert "vis4d_cuda_ops" not in calls[0][-1]
    assert "cubifyanything" in calls[0][-1]


def test_preflight_rejects_unavailable_device(tmp_path, monkeypatch):
    from ioutest.benchmark import preflight
    from types import SimpleNamespace
    marker = tmp_path / "exists"
    marker.touch()
    monkeypatch.setattr("ioutest.benchmark.subprocess.run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout='{"torch":"test","cuda_available":false,"devices":0}', stderr=""))
    model = {"python": str(marker), "adapter": "3dmood",
             "options": {"project_root": str(tmp_path), "checkpoint": str(marker)}}
    with pytest.raises(RuntimeError, match="DSW 原生 Terminal"):
        preflight({"3dmood": model}, {}, "cuda")


def test_geometry_ignore_and_empty_frame_fp():
    gt = {"a": [box(ignore=True)], "b": [box()], "c": []}
    predictions = {"a": [box(cid=None, score=.99), box(cid=None, score=.98)],
                   "b": [box(cid=None, score=.9)], "c": [box(cid=None, score=.95)]}
    row = evaluate_ap(gt, predictions, class_agnostic=True)["per_class"][0]["iou"]["0.5"]
    assert row["ignored_predictions"] == 1
    assert row["tp"] == 1 and row["fp"] == 2
    assert row["ap"] == pytest.approx(1 / 3)


def test_depth_cache_units_rotation_integrity_and_resume(prepared, tmp_path):
    source, run = prepared
    root = tmp_path / "cache"
    protocol, _, frames, _ = load_benchmark(run)
    first = cache_depth(run, root)
    reader = DepthInputReader(root, protocol, frames)
    for f in frames:
        depth = reader.read(f)
        assert depth.shape == tuple(f["image_hw"])
        assert depth.dtype == np.float32 and np.all(depth == 2.)
    # Full cache resumes without opening the remote source.
    source.rename(tmp_path / "unavailable")
    second = cache_depth(run, root)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    frame = frames[0]
    path = root / reader.records[frame["frame_id"]]["depth_image"]
    Image.fromarray(np.full(tuple(frame["raw_image_hw"]), 3000, np.uint16)).save(path)
    with pytest.raises(ValueError, match="深度图片变化"):
        reader.read(frame)


def test_depth_not_gt_and_gravity_basis(prepared):
    _, run = prepared
    _, _, frames, _ = load_benchmark(run)
    for frame in frames:
        gravity = camera_gravity(frame)
        np.testing.assert_allclose(gravity.T @ gravity, np.eye(3), atol=1e-9)
        assert np.linalg.det(gravity) == pytest.approx(1)
        r = np.asarray(frame["camera_rotation"]) @ np.asarray(frame["world_to_raw_camera_rotation"])
        np.testing.assert_allclose(gravity[:, 1], r @ [0., 0., -1.])


def test_pilot_fixed_visits_and_original_frame_ids(prepared):
    _, run = prepared
    _, _, frames, _ = load_benchmark(run)
    chosen = select_frames(frames, "pilot", pilot_scans=2)
    assert chosen == select_frames(frames, "pilot", pilot_scans=2)
    assert len({f["video_id"] for f in chosen}) == len({f["visit_id"] for f in chosen}) == 2
    assert len(chosen) == 6
    assert select_frames(frames, "full") == frames
    with pytest.raises(ValueError):
        select_frames(frames, "pilot", pilot_scans=3)


def test_prediction_atomic_resume_rejects_changed_identity(prepared, tmp_path):
    _, run = prepared
    p, _, frames, _ = load_benchmark(run)
    f = frames[0]
    path = prediction_path(tmp_path, f)
    d = Detection(None, .9, np.array([0., 0., 2.]), np.ones(3), np.eye(3)).to_dict()
    value = {"frame_id": f["frame_id"], "benchmark_id": p["benchmark_id"], "signature": "fixed",
             "schema_version": "ioutest.box3d.v1", "status": "ok", "detections": [d]}
    write_prediction(path, value)
    assert read_prediction(path, f, "fixed", p["benchmark_id"], False) == value
    with pytest.raises(ValueError):
        read_prediction(path, f, "changed", p["benchmark_id"], False)
    with pytest.raises(ValueError):
        read_prediction(path, f, "fixed", p["benchmark_id"], True)


def test_report_counts_phase_subset_and_overlapping_ignore_reasons(prepared, tmp_path):
    _, run = prepared
    protocol, audit, all_frames, gt = load_benchmark(run)
    frames = all_frames[:3]
    gt[frames[0]["frame_id"]][0].update(
        ignore=True, ignore_reasons=["insufficient_observed_depth_points", "outside_camera_view"])
    result = {"frames": 3, "metrics": {"geometry": {"full": {"per_class": [{"prediction_count": 4}]}}}}
    context = report_context(run, tmp_path, "pilot", protocol, audit, all_frames,
                             gt, frames, {"test": result}, [])
    assert context["frozen_frames"] == 9 and context["frames"] == 3
    assert context["frozen_scans"] == 3 and context["scans"] == 1
    assert context["valid_gt"] == 2 and context["ignored_gt"] == 1
    assert context["empty_valid_gt_frames"] == 1
    assert sum(context["ignore_reasons"].values()) == 2
    assert context["model_prediction_counts"] == {"test": 4}
    assert context["excluded_frames"] == 0


class FakeCuTR:
    calls = 0

    def load(self):
        return {"test_only": True}

    def predict(self, sample):
        assert not any("gt" in k or "trajectory" in k for k in vars(sample))
        assert not hasattr(sample, "depth")
        FakeCuTR.calls += 1
        return [Detection(None, .9, np.array([0., 0., 2.]), np.array([2., 2., 1.]), np.eye(3))]

    def close(self):
        pass


def test_worker_resume_and_complete_comparison_report(prepared, tmp_path, monkeypatch):
    _, run = prepared
    root = tmp_path / "report"
    cache = tmp_path / "images"
    cache_images(run, cache)
    p, _, frames, _ = load_benchmark(run)
    selection = {"benchmark_id": p["benchmark_id"], "phase": "pilot", "frame_ids": [f["frame_id"] for f in frames]}
    selection["selection_id"] = digest(selection)
    write_json(root / "pilot/selection.json", selection)
    fake = FakeCuTR()
    FakeCuTR.calls = 0
    monkeypatch.setattr("ioutest.benchmark_worker.create_adapter", lambda *args: (fake, {"test_only": True}))
    model = {"adapter": "test", "options": {}, "semantic_labels": False}
    task = {"run_dir": str(run), "report_root": str(root), "phase": "pilot", "model_id": "cutr-rgb",
            "model": model, "device": "cpu", "image_cache": str(cache), "selection": selection}
    assert work(task)["executed"] == 9
    assert work(task)["reused"] == 9 and FakeCuTR.calls == 9
    report = evaluate_model(run, root, "pilot", "cutr-rgb", model, frames)
    assert report["complete"] and report["frames"] == 9
    assert "semantic_17class" not in report["metrics"]
    assemble_report(run, root, "pilot", {"cutr-rgb": model}, frames)
    assert (root / "pilot/report.md").exists()
    assert (root / "pilot/figures/geometry_ap.svg").exists()
    assert read_json(root / "pilot/summary.json")["complete"]
    document = (root / "pilot/report.md").read_text()
    context = read_json(root / "pilot/report_context.json")
    assert context["frames"] == context["valid_gt"] == 9
    assert context["ignored_gt"] == context["empty_valid_gt_frames"] == 0
    assert context["model_frame_results"] == 9
    assert len(context["visualizations"]) == 3
    assert "公平性、适用范围" in document and "数学例子" in document
    assert "frames.jsonl" in document and "不是逐扫描 AP" in document
    assert "实际保存 3 张图片" in document
    assert "10,980" not in document and "256×192" not in document
    before = (root / "pilot/metrics/cutr-rgb.json").read_bytes()
    # Documentation refresh uses historical provenance, without model/AP execution.
    monkeypatch.setattr("ioutest.benchmark_report.evaluate_ap", lambda *a, **kw: pytest.fail("AP rerun"))
    model_path = root / "models/cutr-rgb/model.json"
    metadata = read_json(model_path)
    write_json(model_path, dict(metadata, created_utc="later-full-stage-write"))
    refreshed = refresh_report(root, "pilot", run, tmp_path / "review/report.md")
    assert refreshed["inference_rerun"] is False and refreshed["ap_recalculated"] is False
    assert (tmp_path / "review/report.md").read_text() == document
    assert (root / "pilot/metrics/cutr-rgb.json").read_bytes() == before
    write_json(root / "pilot/summary.json", dict(read_json(root / "pilot/summary.json"), complete=False))
    with pytest.raises(ValueError, match="尚未完成"):
        refresh_report(root, "pilot", run)
    # Deleted numerical predictions cannot be represented as zero detections.
    prediction_path(root / "models/cutr-rgb", frames[0]).unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_model(run, root, "pilot", "cutr-rgb", model, frames)
    task["model"] = dict(model, options={"changed": True})
    with pytest.raises(ValueError, match="已变化"):
        work(task)


def test_depth_alignment_has_nonuniform_pixel_rotation(prepared, tmp_path):
    root, run = prepared
    p, _, frames, _ = load_benchmark(run)
    # Manufacture sensor depth only to check all actual cached orientation cases.
    cache = tmp_path / "depth"
    cache_depth(run, cache)
    reader = DepthInputReader(cache, p, frames)
    for f in frames:
        depth = np.arange(np.prod(f["raw_image_hw"]), dtype=np.uint16).reshape(f["raw_image_hw"])
        record = reader.records[f["frame_id"]]
        Image.fromarray(depth).save(cache / record["depth_image"])
        record["pixels_sha256"] = hashlib.sha256(depth.astype("<u2").tobytes()).hexdigest()
        np.testing.assert_array_equal(reader.read(f), np.rot90(depth.astype(np.float32) / 1000., f["rotation_k"]))


def test_cutr_native_box_corners_match_canonical_contract(monkeypatch):
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from ioutest.geometry import corners
    torch = pytest.importorskip("torch")
    repo = Path("/mnt/workspace/CA1M")
    if not (repo / "cubifyanything/boxes.py").is_file():
        pytest.skip("CuTR native checkout unavailable")
    monkeypatch.syspath_prepend(str(repo))
    from cubifyanything.boxes import GeneralInstance3DBoxes
    rotation = Rotation.from_euler("xyz", [23, -47, 15], degrees=True).as_matrix()
    native = GeneralInstance3DBoxes(torch.tensor([[.2, -.4, 2.3, .7, 1.2, 1.8]]),
                                   torch.tensor(rotation[None], dtype=torch.float32))
    canonical = Detection(None, .8, native.gravity_center[0].numpy(), native.dims[0].numpy(),
                          native.R[0].numpy()).to_dict()
    actual = corners(canonical)
    expected = native.corners[0].numpy()
    # Corner order differs, but each physical corner must agree in camera space.
    for point in actual:
        assert np.min(np.linalg.norm(expected - point, axis=1)) < 1e-6
