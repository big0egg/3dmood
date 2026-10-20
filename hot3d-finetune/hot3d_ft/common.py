"""Configuration, fingerprints, atomic checkpoints and cooperative locks."""
import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / 'hot3d-finetune'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n').encode())


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def jsonlines(path, rows):
    atomic_bytes(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)
                               + '\n' for row in rows).encode())


def readlines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def config(path=None, root=None):
    result = read(path or PROJECT / 'configs/default.json')
    if root:
        result['root'] = str(Path(root).expanduser().resolve())
    return result


def data_path(cfg):
    return Path(cfg['root']) / 'data' / cfg['dataset_name']


def source_stamp(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def source_code():
    files = [*sorted((PROJECT / 'hot3d_ft').glob('*.py')),
             REPO / 'ioutest/geometry.py', REPO / 'ioutest/hot3d/camera.py',
             REPO / 'ioutest/hot3d/metrics.py']
    return {str(p.relative_to(REPO)): checksum(p) for p in files}


def bind(path, identity):
    signature = digest(identity)
    path = Path(path)
    if path.exists():
        if read(path) != dict(signature=signature, identity=identity):
            raise ValueError(f'配置或输入改变，拒绝复用旧目录：{path}；请使用新的 --root')
    else:
        write(path, dict(signature=signature, identity=identity))
    return signature


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'已有任务持有锁：{path}') from exc
        yield


def inputs_only(frame):
    keys = ('frame_id', 'clip_id', 'frame_key', 'image_hw', 'intrinsics',
            'rgb_sha256', 'rgb_path')
    return {key: frame[key] for key in keys}
