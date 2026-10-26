"""Atomic, checksummed per-frame records for resumable inference."""
import gzip
import json
import os
import tempfile
from pathlib import Path
from .common import digest

def record_path(root, frame_id):
    return Path(root) / 'predictions' / (digest(frame_id) + '.json.gz')

def write_prediction(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(gzip.compress(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def commit(path, value):
    value = dict(value); value['checksum'] = digest(value)
    write_prediction(path, value)
    return value

def read_record(path, frame_id, request_id, model_id):
    with gzip.open(path, 'rt') as f: value = json.load(f)
    checksum = value.pop('checksum', None)
    if checksum != digest(value): raise ValueError('预测文件校验失败：' + str(path))
    if (value.get('schema') != 'ioutest.prediction.v2' or value.get('status') != 'ok' or value.get('frame_id') != frame_id
            or value.get('request_id') != request_id or value.get('model_id') != model_id):
        raise ValueError('预测来源或状态不匹配：' + str(path))
    value['checksum'] = checksum
    return value
