"""Export installed core packages as wheels without changing the source Python.

Run with the PPU image's /usr/local/bin/python3. The exported packages still
require the matching system PPU SDK; they do not contain that SDK.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import platform
import runpy
import sys
import zipfile
from importlib import metadata
from pathlib import Path, PurePosixPath


def export_wheel(dist, output_dir):
    """Rebuild RECORD from installed files; pip recreates console scripts."""
    files = dist.files
    if not files:
        raise RuntimeError(f"{dist.name} has no installed RECORD")
    wheel_files = [f for f in files if str(f).endswith(".dist-info/WHEEL")]
    if len(wheel_files) != 1:
        raise RuntimeError(f"Cannot identify WHEEL metadata for {dist.name}")
    dist_info = PurePosixPath(str(wheel_files[0])).parent
    wheel_metadata = dist.read_text("WHEEL") or ""
    tags = [
        line[5:]
        for line in wheel_metadata.splitlines()
        if line.startswith("Tag: ")
    ]
    if len(tags) != 1:
        raise RuntimeError(f"Expected one platform tag for {dist.name}: {tags}")
    wheel_name = f"{dist_info.name[:-10]}-{tags[0]}.whl"
    destination = output_dir / wheel_name
    temporary = destination.with_suffix(".whl.tmp")
    if destination.exists():
        raise FileExistsError(f"Already exists: {destination}")
    record = []
    omitted_metadata = {
        str(dist_info / name)
        for name in [
            "RECORD", "RECORD.jws", "RECORD.p7s", "INSTALLER",
            "REQUESTED", "direct_url.json",
        ]
    }
    print(f"Exporting {dist.name} {dist.version} -> {destination}", flush=True)
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for file in files:
                name = str(file)
                path = PurePosixPath(name)
                if ".." in path.parts:
                    # Installed entry-point wrappers contain absolute shebangs.
                    # Keep entry_points.txt; pip regenerates the wrappers.
                    if (
                        path.parts[:3] == ("..", "..", "..")
                        and path.parts[3:4] == ("bin",)
                    ):
                        continue
                    raise RuntimeError(f"Unexpected external file: {name}")
                if path.is_absolute():
                    raise RuntimeError(f"Unexpected absolute path: {name}")
                if name in omitted_metadata or path.suffix == ".pyc":
                    continue
                source = Path(dist.locate_file(file))
                digest = hashlib.sha256()
                size = 0
                info = zipfile.ZipInfo.from_file(source, arcname=name)
                info.compress_type = zipfile.ZIP_DEFLATED
                with (
                    source.open("rb") as reader,
                    archive.open(info, "w") as writer,
                ):
                    for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        writer.write(chunk)
                value = (
                    base64.urlsafe_b64encode(digest.digest())
                    .rstrip(b"=").decode()
                )
                record.append((name, f"sha256={value}", size))
            record_name = str(dist_info / "RECORD")
            record.append((record_name, "", ""))
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerows(record)
            archive.writestr(record_name, buffer.getvalue())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with destination.open("rb") as wheel:
        checksum = hashlib.file_digest(wheel, "sha256").hexdigest()
    return {
        "name": dist.name,
        "version": dist.version,
        "source_metadata": str(dist._path),
        "wheel": destination.name,
        "sha256": checksum,
        "recorded_files": len(record),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cuda", default="13.0")
    args = parser.parse_args()
    distributions = [
        metadata.distribution(name)
        for name in ["torch", "torchvision", "triton"]
    ]
    torch_version = runpy.run_path(
        str(distributions[0].locate_file("torch/version.py"))
    )
    if torch_version["cuda"] != args.expected_cuda:
        parser.error(
            f"Expected CUDA {args.expected_cuda}, "
            f"found {torch_version['cuda']}; "
            f"check the Python executable: {sys.executable}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Already exists: {manifest_path}")
    packages = [export_wheel(dist, args.output_dir) for dist in distributions]
    sdk_release = Path("/usr/local/PPU_SDK/release.yaml")
    manifest = {
        "source_python": sys.executable,
        "python": platform.python_version(),
        "torch_build": torch_version["__version__"],
        "cuda": torch_version["cuda"],
        "ppu_sdk": sdk_release.read_text() if sdk_release.exists() else None,
        "packages": packages,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
