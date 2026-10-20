"""Deterministic identities and finite, atomic local records."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"JSON contains non-finite number {value}: {path}")
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject)


def atomic_json(path, value):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_identity():
    root = Path(__file__).parent
    return digest({p.relative_to(root).as_posix(): file_digest(p)
                   for p in sorted(root.rglob("*.py"))})


@contextmanager
def local_lock(run_dir):
    """One writer on this host; an OSS mount is not a distributed lock."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with (Path(run_dir) / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This run directory already has an active writer") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
