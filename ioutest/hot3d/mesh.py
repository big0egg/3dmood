"""Read GLB geometry including scene-node transforms; no renderer dependency."""

import json
import struct

import numpy as np
from scipy.spatial.transform import Rotation


def mesh_vertices(blob):
    magic,version,total = struct.unpack_from('<4sII',blob)
    if magic!=b'glTF' or version!=2 or total!=len(blob):raise ValueError('不是完整 GLB 2.0')
    cursor=12;doc=binary=None
    while cursor<len(blob):
        length,kind=struct.unpack_from('<II',blob,cursor);cursor+=8
        chunk=blob[cursor:cursor+length];cursor+=length
        if len(chunk)!=length:raise ValueError('GLB 截断')
        if kind==0x4E4F534A:doc=json.loads(chunk)
        elif kind==0x004E4942:binary=chunk
    if doc is None or binary is None:raise ValueError('GLB 缺少 JSON/BIN')
    types={5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}
    def position(index):
        a=doc['accessors'][index]
        if a['type']!='VEC3' or 'sparse' in a:raise ValueError('不支持的 GLB POSITION')
        view=doc['bufferViews'][a['bufferView']]
        if view.get('buffer',0)!=0:raise ValueError('GLB 引用了外部 buffer')
        dtype=np.dtype(types[a['componentType']]);stride=view.get('byteStride',dtype.itemsize*3)
        offset=view.get('byteOffset',0)+a.get('byteOffset',0)
        if offset+(a['count']-1)*stride+3*dtype.itemsize>len(binary):raise ValueError('GLB accessor 越界')
        v=np.ndarray((a['count'],3),dtype=dtype,buffer=binary,offset=offset,strides=(stride,dtype.itemsize)).astype(float)
        if a.get('normalized',False):
            info=np.iinfo(dtype);v=np.maximum(v/info.max,-1.)
        return v
    points=[]
    def walk(index,parent,ancestors):
        if index in ancestors:raise ValueError('GLB scene 存在循环')
        node=doc['nodes'][index]
        if 'matrix' in node:local=np.asarray(node['matrix'],dtype=float).reshape(4,4,order='F')
        else:
            local=np.eye(4)
            local[:3,:3]=Rotation.from_quat(node.get('rotation',[0,0,0,1])).as_matrix()@np.diag(node.get('scale',[1,1,1]))
            local[:3,3]=node.get('translation',[0,0,0])
        transform=parent@local
        if 'mesh' in node:
            for primitive in doc['meshes'][node['mesh']]['primitives']:
                v=position(primitive['attributes']['POSITION'])
                points.append(v@transform[:3,:3].T+transform[:3,3])
        for child in node.get('children',[]):walk(child,transform,ancestors|{index})
    for index in doc['scenes'][doc.get('scene',0)]['nodes']:walk(index,np.eye(4),set())
    if not points:raise ValueError('GLB 没有顶点')
    vertices=np.concatenate(points)
    if not np.isfinite(vertices).all():raise ValueError('GLB 顶点不是有限数值')
    return vertices


def mesh_bounds(blob):
    vertices=mesh_vertices(blob);lo,hi=vertices.min(0),vertices.max(0)
    size=hi-lo
    if (size<=0).any() or size.max()>5 or size.max()<.001:
        raise ValueError('GLB 包围盒尺寸异常，请检查节点变换和米制单位')
    return dict(center=((lo+hi)/2).tolist(),size=size.tolist(),rotation=np.eye(3).tolist(),
                bounds=[lo.tolist(),hi.tolist()],vertices=len(vertices))
