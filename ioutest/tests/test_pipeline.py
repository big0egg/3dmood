import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ioutest.common import read_json, write_json, write_jsonl
from ioutest.dataset import ScanReader, load_benchmark, metadata, prepare
from ioutest.image_cache import FrameInputReader, cache_images
from ioutest.pipeline import evaluate, import_predictions, infer, load_predictions, training_audit
from ioutest.schema import Detection
from ioutest.visualize import visualize


class FakeAdapter:
    """Exercise the public interface without leaking GT or loading a real model."""
    calls = []
    fail_once = None

    def __init__(self, options):
        self.options = options

    def load(self):
        return {"model": "deterministic test model"}

    def predict(self, sample):
        assert set(vars(sample)) == {"frame_id", "rgb", "intrinsics", "class_names", "prompts"}
        self.calls.append(sample.frame_id)
        if sample.frame_id == self.fail_once:
            FakeAdapter.fail_once = None
            raise RuntimeError("temporary test failure")
        return [Detection(13, .9, np.array([0., 0., 2.]), np.array([2., 2., 1.]), np.eye(3))]

    def close(self):
        pass


def png(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "data"
    (root / "source").mkdir(parents=True)
    lines = ["video_id,visit_id,sky_direction,fold", "99999999,B,Up,Training"]
    scenes = [("11111111", "A", "Up"), ("22222222", "A", "Left"), ("33333333", "B", "Down")]
    for scan, visit, direction in scenes:
        lines.append(f"{scan},{visit},{direction},Validation")
        archive = root / "3dod/Validation" / (scan + ".zip")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(f"{scan}/{scan}_3dod_annotation.json", json.dumps({"skipped": False, "data": [{
                "uid": "chair-" + scan, "label": "chair", "segments": {"obbAligned": {
                    "centroid": [0., 0., 2.], "axesLengths": [2., 2., 1.],
                    "normalizedAxes": np.eye(3).reshape(-1).tolist()}}}]}))
            prefix = f"{scan}/{scan}_frames/"
            z.writestr(prefix + "lowres_wide.traj", "\n".join(f"{t}.000 0 0 0 0 0 0" for t in (1, 2, 3)))
            for t in (1, 2, 3):
                stem = f"{scan}_{t}.000"
                z.writestr(prefix + "lowres_wide/" + stem + ".png", png(np.full((12, 16, 3), t * 40, np.uint8)))
                z.writestr(prefix + "lowres_depth/" + stem + ".png", png(np.full((12, 16), 2000, np.uint16)))
                z.writestr(prefix + "lowres_wide_intrinsics/" + stem + ".pincam", "16 12 12 12 7.5 5.5")
    (root / "source/metadata.csv").write_text("\n".join(lines) + "\n")
    run = tmp_path / "run"
    prepare(root, run, frames_per_scan=3)
    return root, run


def test_prepare_reuse_integrity_and_upright_gt(prepared):
    root, run = prepared
    protocol, audit, frames, gt = load_benchmark(run)
    assert len(frames) == 9
    assert audit["official_training_visit_overlap"]["validation_scan_ids"] == ["33333333"]
    assert all(not boxes[0]["ignore"] for boxes in gt.values())
    assert {tuple(f["image_hw"]) for f in frames} == {(12, 16), (16, 12)}
    assert prepare(root, run, frames_per_scan=3)["benchmark_id"] == protocol["benchmark_id"]
    with pytest.raises(ValueError, match="不同协议"):
        prepare(root, run, frames_per_scan=2)
    path = run / "groundtruth/11111111.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="已变化"):
        load_benchmark(run)


def test_infer_resume_and_offline_reports(prepared):
    _, run = prepared
    FakeAdapter.calls = []
    FakeAdapter.fail_once = "11111111:11111111_2.000"
    with pytest.raises(RuntimeError, match="temporary"):
        infer(run, "fake", "ioutest.tests.test_pipeline:FakeAdapter")
    assert (run / "models/fake/predictions/11111111.json").exists()
    with pytest.raises(ValueError, match="尚未完成"):
        evaluate(run, "fake")
    infer(run, "fake", "ioutest.tests.test_pipeline:FakeAdapter")
    assert FakeAdapter.calls.count("11111111:11111111_1.000") == 1
    assert len(FakeAdapter.calls) == 10
    report = evaluate(run, "fake")
    assert report["full"]["summary"]["iou"]["map"] == pytest.approx(1)
    assert report["full"]["summary"]["center"]["map"] == pytest.approx(1)
    assert report["subsets"]["official_training_visit_disjoint"]["frame_count"] == 6
    assert (run / "models/fake/reports/metrics_per_class.csv").exists()
    assert (run / "models/fake/reports/metrics_per_scan.csv").exists()
    assert len(visualize(run, "fake", max_frames=3)) == 3
    with pytest.raises(ValueError, match="不同模型"):
        infer(run, "fake", "ioutest.tests.test_pipeline:FakeAdapter", {"new_option": 1})


def test_generic_import_empty_predictions_and_audit(prepared, tmp_path):
    root, run = prepared
    protocol, _, frames, _ = load_benchmark(run)
    source = tmp_path / "predictions.jsonl"
    write_jsonl(source, [{"frame_id": f["frame_id"], "status": "ok", "detections": []} for f in frames])
    with pytest.raises(ValueError, match="benchmark-id"):
        import_predictions(run, "offline", source, "wrong-id")
    import_predictions(run, "offline", source, protocol["benchmark_id"])
    assert evaluate(run, "offline")["full"]["summary"]["iou"]["map"] == 0
    shard = run / "models/offline/predictions/11111111.json"
    value = read_json(shard); value["records"].append(value["records"][0]); write_json(shard, value)
    with pytest.raises(ValueError, match="重复"):
        load_predictions(run, "offline")
    annotation = tmp_path / "train.json"
    write_json(annotation, {"images": [{"file_path": "ARKitScenes/train/99999999/1.000.png"}]})
    assert training_audit(root, [annotation])["excluded_validation_scans"] == ["33333333"]


def test_model_input_detects_changed_pixels(prepared):
    root, run = prepared
    _, _, frames, _ = load_benchmark(run)
    frame = frames[0]
    zip_path = root / "3dod/Validation" / (frame["video_id"] + ".zip")
    with zipfile.ZipFile(zip_path) as z:
        values = {name: z.read(name) for name in z.namelist()}
    image = np.full((12, 16, 3), 7, dtype=np.uint8)
    values[frame["rgb_entry"]] = png(image)
    with zipfile.ZipFile(zip_path, "w") as z:
        for name, value in values.items():
            z.writestr(name, value)
    with ScanReader(root, metadata(root)[frame["video_id"]]) as reader:
        with pytest.raises(ValueError, match="内容发生变化"):
            reader.model_input(frame)


def test_offline_metrics_have_no_model_imports():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; import ioutest.metrics; assert 'torch' not in sys.modules; "
        "assert 'opendet3d' not in sys.modules; assert 'vis4d' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_local_cache_infer_and_visualize_without_source(prepared, tmp_path, monkeypatch):
    root, run = prepared
    protocol, _, frames, _ = load_benchmark(run)
    expected = {}
    for frame in frames:
        with ScanReader(root, metadata(root)[frame["video_id"]]) as reader:
            expected[frame["frame_id"]] = reader.model_input(frame)
    cache = tmp_path / "image-cache"
    result = cache_images(run, cache, workers=2)
    assert result["complete"] and result["frames"] == len(frames)
    monkeypatch.setattr("ioutest.image_cache.default_cache_dir", lambda unused: cache)
    root.rename(tmp_path / "disconnected-source")
    # Failing every ZIP open proves local inference and visualization use no ZIP.
    def fail_zip(*args, **kwargs):
        raise AssertionError("unexpected ZIP access")
    monkeypatch.setattr(zipfile, "ZipFile", fail_zip)
    inputs = FrameInputReader(run, protocol, frames)
    assert inputs.source_info["kind"] == "local_png"
    for frame in frames:
        with inputs.scan(frame["video_id"]) as reader:
            actual = reader.model_input(frame)
        original = expected[frame["frame_id"]]
        np.testing.assert_array_equal(actual.rgb, original.rgb)
        np.testing.assert_array_equal(actual.intrinsics, original.intrinsics)
    FakeAdapter.calls = []; FakeAdapter.fail_once = None
    timing = infer(run, "cached", "ioutest.tests.test_pipeline:FakeAdapter")
    assert len(FakeAdapter.calls) == len(frames)
    assert timing["input_source"]["kind"] == "local_png"
    assert len(visualize(run, "cached", max_frames=3)) == 3
    assert evaluate(run, "cached")["full"]["summary"]["iou"]["map"] == pytest.approx(1)
    # Export is resumable and can revalidate already materialized files offline.
    assert cache_images(run, cache)["frames"] == len(frames)


def test_image_cache_selection(prepared, tmp_path, monkeypatch):
    root, run = prepared
    protocol, _, frames, _ = load_benchmark(run)
    cache = tmp_path / "image-cache"
    monkeypatch.setattr("ioutest.image_cache.default_cache_dir", lambda unused: cache)
    assert FrameInputReader(run, protocol, frames).source_info["kind"] == "zip"
    with pytest.raises(ValueError, match="尚未完成"):
        FrameInputReader(run, protocol, frames, image_cache=cache)
    cache_images(run, cache)
    assert FrameInputReader(run, protocol, frames).source_info["kind"] == "local_png"
    assert FrameInputReader(run, protocol, frames, image_cache=False).source_info["kind"] == "zip"


@pytest.mark.parametrize("damage", ["benchmark", "incomplete", "manifest", "duplicate", "missing_id",
                                    "outside_path", "frozen_hash", "missing_image", "pixels"])
def test_invalid_image_cache_is_rejected(prepared, tmp_path, damage):
    from ioutest.common import file_digest
    _, run = prepared
    protocol, _, frames, _ = load_benchmark(run)
    cache = tmp_path / "image-cache"
    cache_images(run, cache)
    summary_path = cache / "cache_summary.json"
    manifest = cache / "images.jsonl"
    summary = read_json(summary_path)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if damage == "benchmark":
        summary["benchmark_id"] = "other-benchmark"
    elif damage == "incomplete":
        summary["complete"] = False
    elif damage == "manifest":
        manifest.write_text(manifest.read_text() + " ")
    elif damage == "missing_image":
        (cache / records[0]["raw_image"]).unlink()
    elif damage == "pixels":
        (cache / records[0]["raw_image"]).write_bytes(png(np.zeros((12, 16, 3), dtype=np.uint8)))
    else:
        if damage == "duplicate":
            records.append(records[0])
        elif damage == "missing_id":
            records.pop()
        elif damage == "outside_path":
            records[0]["raw_image"] = "../outside.png"
        elif damage == "frozen_hash":
            records[0]["raw_rgb_pixels_sha256"] = "wrong"
        write_jsonl(manifest, records)
        summary["images_manifest_sha256"] = file_digest(manifest)
    write_json(summary_path, summary)
    with pytest.raises(ValueError):
        inputs = FrameInputReader(run, protocol, frames, image_cache=cache)
        with inputs.scan(frames[0]["video_id"]) as reader:
            reader.model_input(frames[0])


@pytest.mark.parametrize("rotation_k", [0, 1, 2, 3])
def test_cached_input_rotates_original_pixels_once(tmp_path, rotation_k):
    import hashlib
    from ioutest.common import file_digest
    cache = tmp_path / "cache"
    (cache / "raw_images").mkdir(parents=True)
    rgb = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
    (cache / "raw_images/frame.png").write_bytes(png(rgb))
    frame = {"frame_id": "scan:frame", "video_id": "scan", "raw_image_hw": [4, 7],
             "rgb_pixels_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
             "rotation_k": rotation_k, "intrinsics": [[10., 0., 2.], [0., 10., 3.], [0., 0., 1.]]}
    write_jsonl(cache / "images.jsonl", [{"frame_id": frame["frame_id"], "raw_image": "raw_images/frame.png",
                                        "raw_rgb_pixels_sha256": frame["rgb_pixels_sha256"]}])
    write_json(cache / "cache_summary.json", {"complete": True, "benchmark_id": "test", "frames": 1,
                                              "images_manifest_sha256": file_digest(cache / "images.jsonl")})
    inputs = FrameInputReader(tmp_path / "run", {"benchmark_id": "test", "settings": {"dataset_root": "unused"}},
                              [frame], image_cache=cache)
    with inputs.scan("scan") as reader:
        actual = reader.model_input(frame)
    np.testing.assert_array_equal(actual.rgb, np.rot90(rgb, rotation_k))
    np.testing.assert_array_equal(actual.intrinsics, frame["intrinsics"])
