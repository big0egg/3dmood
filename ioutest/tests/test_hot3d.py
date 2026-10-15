"""Geometry, input isolation, >17 labels, metric matching and resumable reports."""

import copy
import io
import json
import struct
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ioutest.common import read_json,write_json
from ioutest.hot3d.camera import decode_mask,pinhole_maps,project_fisheye,remap
from ioutest.hot3d.common import TRACKS,bind,commit,export_detection,read_record,record_path
from ioutest.hot3d.dataset import allowed_inputs,load,paths,prepare
from ioutest.hot3d.mesh import mesh_bounds
from ioutest.hot3d.metrics import conditional,evaluate,thresholds
from ioutest.hot3d.report import assemble
from ioutest.hot3d.score import score
from ioutest.hot3d.worker import work
from ioutest.schema import Detection


def glb_cube():
    vertices=np.array([[x,y,z] for x in (-500,500) for y in (-500,500) for z in (-500,500)],dtype='<f4')
    document=dict(asset=dict(version='2.0'),scene=0,scenes=[dict(nodes=[0])],
        nodes=[dict(mesh=0,scale=[.001,.001,.001])],meshes=[dict(primitives=[dict(attributes=dict(POSITION=0))])],
        buffers=[dict(byteLength=vertices.nbytes)],bufferViews=[dict(buffer=0,byteLength=vertices.nbytes)],
        accessors=[dict(bufferView=0,componentType=5126,count=8,type='VEC3')])
    raw=json.dumps(document).encode();raw+=b' '*((-len(raw))%4);binary=vertices.tobytes()
    return struct.pack('<4sII',b'glTF',2,12+8+len(raw)+8+len(binary))+struct.pack('<II',len(raw),0x4E4F534A)+raw+struct.pack('<II',len(binary),0x004E4942)+binary


@pytest.fixture
def prepared_hot3d(tmp_path):
    config=read_json(Path(__file__).parents[1]/'hot3d_config.json')
    source=tmp_path/'source';config.update(root=str(tmp_path/'benchmark'),dataset_root=str(source),device='cpu')
    config['data'].update(clip_ids=[1849],frame_indices=[7,22],minimum_amodal_pixels=1)
    config['data']['camera'].update(width=32,height=32)
    asset=source/'Assets/assets/Hot3DAssets_assets_assets.zip';asset.parent.mkdir(parents=True)
    with zipfile.ZipFile(asset,'w') as z:
        index={str(i):dict(instance_bop_id=str(i),instance_id=str(i),instance_name='object_'+str(i),instance_type='object') for i in range(1,34)}
        index['headset']=dict(instance_bop_id='33',instance_type='headset')
        z.writestr('instance.json',json.dumps(index))
        for i in range(1,34):z.writestr(f'{i}.glb',glb_cube())
    write_json(source/'Clips/clip_splits.json',{'train':{'Aria':[1849]}})
    definition=dict(sequence_id='P0001_fixture',device='Aria',per_frame_timestamps_ns=[{'214-1':i+1} for i in range(150)])
    write_json(source/'Clips/clip_definitions.json',{'1849':definition})
    archive=source/'Clips/train_aria/clip-001849.tar';archive.parent.mkdir()
    image=io.BytesIO();Image.fromarray(np.full((32,32,3),120,np.uint8)).save(image,format='JPEG')
    pose=dict(quaternion_wxyz=[1,0,0,0],translation_xyz=[0,0,0])
    calibration=dict(projection_model_type='CameraModelType.FISHEYE624',projection_params=[16,15.5,15.5,*([0]*12)],
        image_width=32,image_height=32,max_solid_angle=1.4)
    mask=dict(height=32,width=32,rle=[v for row in range(13,19) for v in (row*32+14,6)])
    obj={'33':[dict(object_bop_id='33',object_uid='33',T_world_from_object=dict(pose,translation_xyz=[0,0,2]),
        masks_amodal={'214-1':mask},visibilities_modeled={'214-1':1.},visibilities_predicted={'214-1':0.})]}
    with tarfile.open(archive,'w') as t:
        for i in range(150):
            values={'image_214-1.jpg':image.getvalue(),'objects.json':json.dumps(obj).encode()}
            if i in (7,22):values.update({
                'cameras.json':json.dumps({'214-1':dict(T_world_from_camera=pose,calibration=calibration)}).encode(),
                'info.json':json.dumps(dict(sequence_id='P0001_fixture',participant_id='P0001',image_timestamps_ns={'214-1':i+1})).encode()})
            for suffix,blob in values.items():
                entry=tarfile.TarInfo(f'{i:06d}.'+suffix);entry.size=len(blob);t.addfile(entry,io.BytesIO(blob))
    result=prepare(config)
    return config,*result


def test_mesh_node_scale_and_malformed_glb():
    bounds=mesh_bounds(glb_cube())
    assert bounds['size']==pytest.approx([1,1,1])
    assert bounds['center']==pytest.approx([0,0,0])
    with pytest.raises(ValueError):mesh_bounds(glb_cube()[:-4])


def test_fisheye_zero_coefficients_and_upright_camera():
    cal=dict(projection_model_type='CameraModelType.FISHEYE624',projection_params=[100,200,200,*([0]*12)],
             image_width=400,image_height=400,max_solid_angle=1.4)
    pixels,valid=project_fisheye(np.array([[0,0,1],[1,0,1],[0,1,1]]),cal)
    assert pixels[0].tolist()==[200,200]
    assert pixels[1,0]==pytest.approx(200+100*np.pi/4)
    maps=pinhole_maps(cal,dict(width=33,height=33,horizontal_fov_degrees=90,clockwise_quarter_turns=1))
    assert maps[0][16,24]==pytest.approx(200)
    assert maps[1][16,24]<200  # right in upright view samples above in raw view.
    assert np.linalg.det(maps[4])==pytest.approx(1)


def test_preparation_resume_input_isolation_and_corruption(prepared_hot3d):
    config,protocol,audit,frames,gt=prepared_hot3d
    assert audit['valid_targets']==2 and len(protocol['classes'])==25
    assert audit['minimum_mask_containment']==1
    assert gt[frames[0]['frame_id']][0]['class_id']==24
    assert gt[frames[0]['frame_id']][0]['uid']!='33:0'
    inputs=allowed_inputs(config,frames,gt)
    for row in inputs['text']:
        assert not set(row)&{'prompts','class_id','object_bop_id','center','size','rotation','T_world_from_camera','depth'}
    assert set(inputs['gt-box'][0]['prompts'][0])=={'target_id','box_xyxy'}
    again=prepare(config)
    assert again[0]['benchmark_id']==protocol['benchmark_id']
    _,data,_=paths(config);(data/frames[0]['rgb_relative']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='损坏'):prepare(config)


def test_dynamic_labels_and_metric_score_floor_cache():
    classes=[dict(name=str(i),prompt=str(i)) for i in range(25)]
    target=dict(center=[0,0,2],size=[1,1,1],rotation=np.eye(3).tolist(),class_id=24,uid='x',object_bop_id=33)
    pred=dict(target,score=.9);wrong=dict(pred,center=[5,0,2],score=.1)
    actual=export_detection(Detection(24,.9,np.array([0,0,2]),np.ones(3),np.eye(3)),classes)
    assert actual['class_id']==24
    with pytest.raises(ValueError):export_detection(dict(pred,class_id=1.5),classes)
    cache={};limits=[('iou',.5),('center',.02)]
    evaluate({'f':[target]},{'f':[wrong,pred]},classes,limits,pairs=cache)
    workpoint=evaluate({'f':[target]},{'f':[wrong,pred]},classes,limits,score_floor=.7,pairs=cache)
    semantic=evaluate({'f':[target]},{'f':[wrong,pred]},classes,limits,semantic=True,pairs=cache)
    assert workpoint['summary']['AP3D@0.50']==1
    assert semantic['summary']['AP3D@0.50']==1
    missing={'f':dict(items=[])}
    with pytest.raises(ValueError,match='缺失'):conditional({'f':[target]},missing,limits)
    result,_=conditional({'f':[target]},{'f':dict(items=[dict(target_id='x',status='failed',failure='empty',detection=None)])},limits)
    assert result['targets']==1 and result['success']['success3D@0.50']==0


def test_five_config_scoring_and_reports(prepared_hot3d):
    config,protocol,audit,frames,gt=prepared_hot3d
    root,data,run=paths(config);classes=protocol['classes']
    for track,keys in TRACKS.items():
        for key in keys:
            model=run/track/'models'/key;signature=bind(model/'model.json',{'model':key})
            for f in frames:
                b=gt[f['frame_id']][0];det={k:b[k] for k in ('center','size','rotation','class_id')}
                det.update(score=.9,class_id=None if track=='gt-box' else det['class_id'])
                items=[] if key in ('3dmood','wilddet3d') else [dict(target_id=b['uid'],status='ok',detection=det)]
                commit(record_path(model/'predictions',f),f,signature,detections=[det],items=items,
                    timing=dict(predict_seconds=.01,frontend_seconds=.02,depth_seconds=.03))
            metric=score(config,run,track,key,frames,gt,protocol)
            assert metric['geometry']['summary']['AP3D@0.50']==pytest.approx(1)
            assert metric['speed']['fps_pipeline']==pytest.approx(1/.06)
        assemble(config,track,frames,gt,protocol,audit)
        document=(run/track/'full/report.md').read_text()
        assert 'FISHEYE624' in document and '完整流程FPS' in document and '输入' in document
        assert read_json(run/track/'full/summary.json')['complete']
    assert (root/'report.md').is_file()
    record_path(run/'text/models/unipose9d-text-rgb/predictions',frames[0]).unlink()
    with pytest.raises(FileNotFoundError):score(config,run,'text','unipose9d-text-rgb',frames,gt,protocol)


def test_worker_object_failure_resume_and_dynamic_text_labels(prepared_hot3d,tmp_path,monkeypatch):
    from ioutest.adapters.unipose9d import ObjectFailure
    config,protocol,audit,frames,gt=prepared_hot3d
    _,_,run=paths(config);inputs=allowed_inputs(config,frames,gt)
    module=types.ModuleType('hot3d_test_adapter');source=tmp_path/'fake.py';source.write_text('# controlled test model')
    module.__file__=str(source);calls=[]
    class Fake:
        def __init__(self,options):pass
        def load(self):return dict(model='test')
        def close(self):pass
        def predict_object(self,sample,seed):
            calls.append(seed)
            if len(calls)==2:raise ObjectFailure('empty')
            return dict(class_id=None,score=.9,center=[0,0,2],size=[1,1,1],rotation=np.eye(3).tolist()),{}
        def predict(self,sample):
            assert len(sample.prompts)==25
            return [Detection(24,.9,np.array([0,0,2]),np.ones(3),np.eye(3))]
    module.Fake=Fake;monkeypatch.setitem(sys.modules,'hot3d_test_adapter',module)
    for key in ('wilddet3d-box','3dmood'):config['models'][key]['adapter']='hot3d_test_adapter:Fake'
    task=dict(config=config,run_root=str(run),track='gt-box',model_id='wilddet3d-box',frames=inputs['gt-box'],
        benchmark_id=protocol['benchmark_id'],input_manifest='test-box',classes=protocol['classes'])
    result=work(task)
    assert result['failures']==1 and result['frames']==2
    resumed=work(task)
    assert resumed['reused']==2 and len(calls)==2
    task.update(track='text',model_id='3dmood',frames=inputs['text'],input_manifest='test-text')
    work(task)
    identity=read_json(run/'text/models/3dmood/model.json')
    record=read_record(record_path(run/'text/models/3dmood/predictions',frames[0]),frames[0],identity['signature'])
    assert record['detections'][0]['class_id']==24


def test_frontend_text_padding_multiple_instances_and_shared_depth(prepared_hot3d,tmp_path,monkeypatch):
    import torch
    from ioutest.hot3d.frontend import work as frontend
    from ioutest.hot3d.common import prompt_text
    config,protocol,_,frames,gt=prepared_hot3d
    _,_,run=paths(config);inputs=allowed_inputs(config,frames,gt)
    _,spans=prompt_text(protocol['classes']);ids=[101,*range(1,26),102]
    class Inputs(dict):
        def to(self,device):return self
    class Processor:
        def tokenizer(self,text,**kwargs):return dict(input_ids=ids,offset_mapping=[(0,0),*spans,(0,0)])
        def __call__(self,**kwargs):return Inputs(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids))))
    class Grounding:
        def to(self,device):return self
        def eval(self):return self
        def __call__(self,**kwargs):
            logits=torch.full((1,3,64),float('-inf'));logits[:,:,:len(ids)]=-10
            logits[0,0,1]=9;logits[0,1,25]=9;logits[0,2,25]=8
            return types.SimpleNamespace(logits=logits,pred_boxes=torch.tensor([[[.3,.3,.2,.2],[.6,.6,.3,.3],[.8,.7,.2,.2]]]))
    class Predictor:
        def __init__(self,model):pass
        def set_image(self,image):self.hw=image.shape[:2]
        def predict(self,**kwargs):return np.ones((1,*self.hw),bool),np.array([.9]),None
    package=tmp_path/'sam2';package.mkdir();(package/'__init__.py').write_text('# fixture')
    monkeypatch.setitem(sys.modules,'sam2',types.SimpleNamespace(__file__=str(package/'__init__.py')))
    monkeypatch.setitem(sys.modules,'sam2.build_sam',types.SimpleNamespace(build_sam2=lambda *a,**kw:None))
    monkeypatch.setitem(sys.modules,'sam2.sam2_image_predictor',types.SimpleNamespace(SAM2ImagePredictor=Predictor))
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(__version__='fixture',
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a,**kw:Processor()),
        AutoModelForZeroShotObjectDetection=types.SimpleNamespace(from_pretrained=lambda *a,**kw:Grounding())))
    calls=[]
    def depth(model,image):calls.append(1);return np.ones(image.shape[:2],np.float32),np.eye(3)
    monkeypatch.setitem(sys.modules,'moge_model',types.SimpleNamespace(load_moge_model=lambda *a:None,predict_depth_intrinsics=depth))
    for field in ('sam2_checkpoint','moge_checkpoint'):
        p=tmp_path/(field+'.pt');p.write_bytes(b'fixture');config['frontend'][field]=str(p)
    grounding=tmp_path/'grounding';grounding.mkdir();(grounding/'config.json').write_text('{}')
    config['frontend']['grounding_model']=str(grounding)
    task=dict(config=config,run_root=str(run),track='text',frames=inputs['text'],classes=protocol['classes'],
        benchmark_id=protocol['benchmark_id'],input_manifest='text',rgb_manifest='same-rgb')
    assert frontend(task)['frames']==2 and len(calls)==2
    identity=read_json(run/'aux/text/model.json')
    record=read_record(record_path(run/'aux/text/frames',frames[0]),frames[0],identity['signature'])
    assert [i['class_id'] for i in record['items']]==[0,24,24]
    assert frontend(task)['reused']==2 and len(calls)==2
    box_task=dict(task,track='gt-box',frames=inputs['gt-box'],input_manifest='box')
    frontend(box_task)
    assert len(calls)==2  # GT-box and text reuse the same predicted RGB depth.


def test_asset_recovery_paths_and_checksum(tmp_path,monkeypatch):
    import hashlib
    from ioutest.hot3d import assets
    c=read_json(Path(__file__).parents[1]/'hot3d_config.json');c['root']=str(tmp_path/'target')
    for key in c['models']:
        c['models'][key]['options']['checkpoint']=str(tmp_path/'missing'/key)
    c['frontend']['sam2_checkpoint']=str(tmp_path/'missing-sam')
    c['frontend']['moge_checkpoint']=str(tmp_path/'missing-moge')
    resolved=assets.localize(c)
    assert resolved['models']['unipose9d-box-rgb']['options']['checkpoint']==str(tmp_path/'target/assets/unipose9d/last.ckpt')
    assert c['models']['unipose9d-box-rgb']['options']['checkpoint']!=resolved['models']['unipose9d-box-rgb']['options']['checkpoint']
    source=tmp_path/'source.pt';source.write_bytes(b'weight'*(200000))
    monkeypatch.setitem(assets.EXPECTED,'wild',hashlib.sha256(source.read_bytes()).hexdigest())
    destination=tmp_path/'target.pt';assets.copy_weight(source,destination,'wild')
    assert destination.read_bytes()==source.read_bytes()
    destination.write_bytes(b'bad'*400000)
    with pytest.raises(ValueError,match='SHA256'):assets.copy_weight(source,destination,'wild')
