"""Dataset-specific labels with the existing oriented-box geometry contract."""

import gzip
import io
import json
from pathlib import Path

import numpy as np

from ..common import digest,file_digest,read_json,write_json
from ..storage import _write_bytes
from ..schema import validate_box as validate_geometry

SCHEMA='hot3d.box3d.v1'
TRACKS={'gt-box':('wilddet3d-box','unipose9d-box-rgb'),
        'text':('3dmood','wilddet3d','unipose9d-text-rgb')}


def implementation(names):
    root=Path(__file__).parent
    return digest({name:file_digest(root/name) for name in names})


def validate_box(box,classes,prediction=False):
    cid=box.get('class_id')
    if cid is not None and (isinstance(cid,bool) or not isinstance(cid,int) or not 0<=cid<len(classes)):
        raise ValueError('HOT3D class_id 越界')
    if cid is None and not prediction:raise ValueError('GT 不能缺少类别')
    validate_geometry(dict(box,class_id=None),prediction=prediction,allow_unlabeled=True)


def export_detection(box,classes):
    if not isinstance(box,dict):
        box={key:getattr(box,key) for key in ('class_id','score','center','size','rotation')}
    if box['class_id'] is not None and (isinstance(box['class_id'],(bool,np.bool_)) or not isinstance(box['class_id'],(int,np.integer))):
        raise ValueError('预测类别必须为整数，不能截断浮点数')
    result={key:np.asarray(box[key],dtype=float).tolist() for key in ('center','size','rotation')}
    result.update(class_id=None if box['class_id'] is None else int(box['class_id']),score=float(box['score']))
    validate_box(result,classes,prediction=True)
    return result


def bind(path,identity):
    value=dict(signature=digest(identity),provenance=identity)
    if Path(path).exists() and read_json(path)!=value:
        raise ValueError(f'配置、数据或实现改变，不能复用缓存：{path}')
    write_json(path,value)
    return value['signature']


def record_path(root,frame,suffix='.json.gz'):
    for value in (frame['clip_id'],frame['frame_key']):
        if Path(value).name!=value:raise ValueError('非法帧标识')
    return Path(root)/frame['clip_id']/(frame['frame_key']+suffix)


def commit(path,frame,signature,**values):
    record=dict(schema=SCHEMA,status='ok',frame_id=frame['frame_id'],signature=signature,**values)
    _write_bytes(Path(path),gzip.compress(json.dumps(record,allow_nan=False).encode(),mtime=0))
    return record


def read_record(path,frame,signature):
    with gzip.open(path,'rt') as f:record=json.load(f)
    if (record.get('schema')!=SCHEMA or record.get('status')!='ok' or record.get('frame_id')!=frame['frame_id'] or record.get('signature')!=signature):
        raise ValueError(f'缓存不完整或来源不一致：{path}')
    return record


def save_png(path,array):
    from PIL import Image
    stream=io.BytesIO();Image.fromarray(array).save(stream,format='PNG')
    _write_bytes(Path(path),stream.getvalue())


def rgb(frame):
    from PIL import Image
    if file_digest(frame['rgb_path'])!=frame['rgb_sha256']:raise ValueError('本地 RGB 文件已变化')
    with Image.open(frame['rgb_path']) as im:out=np.asarray(im.convert('RGB')).copy()
    if list(out.shape[:2])!=frame['image_hw']:raise ValueError('RGB 尺寸不匹配')
    return out


def prompt_text(classes):
    text='';spans=[]
    for entry in classes:
        word=entry['prompt'];start=len(text);text+=word;spans.append((start,len(text)));text+=' . '
    return text.strip(),spans
