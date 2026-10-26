"""Atomic artifacts, stable content identities and cooperative process locks."""
import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write(path, value):
    atomic_bytes(path, (json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n").encode())


def jsonlines(path, rows):
    atomic_bytes(path, "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ).encode())


def readlines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bind(path, identity):
    value = {"signature": digest(identity), "identity": identity}
    if Path(path).exists() and read(path) != value:
        raise ValueError(f"配置、数据、权重或训练源码变化；请使用新实验目录：{path}")
    write(path, value)
    return value["signature"]


@contextlib.contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有任务运行：{path}") from exc
        yield


def data_path(cfg):
    return Path(cfg["root"]) / "dataset"


def source_code():
    """Logical package-relative keys survive relocation of the outer folder."""
    package = Path(__file__).parent
    sources = {"finefune/" + str(p.relative_to(package)): checksum(p)
               for p in sorted(package.rglob("*.py"))}
    for name in ("experiment_data.py", "schema.py", "geometry.py", "checkpoints.py", "model_environment.py"):
        sources["ioutest/" + name] = checksum(REPO / "ioutest" / name)
    for p in sorted((REPO / "ioutest/adapters").glob("*.py")):
        sources["ioutest/adapters/" + p.name] = checksum(p)
    return sources


def outside_sources(output, sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"输入与输出目录必须互不包含：{source} / {output}")
