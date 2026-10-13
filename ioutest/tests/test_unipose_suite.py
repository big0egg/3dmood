"""Task isolation, conditional denominators, AP ranking and cache integrity."""

import copy
from pathlib import Path

import numpy as np
import pytest

from ioutest.common import read_json, write_json
from ioutest.dataset import load_benchmark
from ioutest.depth_cache import cache_depth
from ioutest.image_cache import cache_images
from ioutest.schema import Detection
from ioutest.suite_frontend import category_tokens, fixed_prompt
from ioutest.suite_io import ObjectInput, bind_identity, commit_record, frame_path, object_seed, read_record
from ioutest.suite_report import assemble, inference_speed
from ioutest.suite_score import conditional_success, score_model, workpoint
from ioutest.unipose_suite import prepare_inputs, smoke_frames
from ioutest.tests.test_geometry_metrics import box
from ioutest.tests.test_pipeline import prepared


def test_allowed_inputs_exclude_text_groundtruth(prepared, tmp_path):
    _, run=prepared
    cache=tmp_path/'cache'
    cache_images(run,cache);cache_depth(run,cache)
    config=dict(run_dir=str(run),image_cache=str(cache))
    protocol,audit,frames,gt,rows,info=prepare_inputs(config)
    assert len(rows['text'])==len(rows['gt-box'])==9
    for row in rows['text']:
        assert not set(row)&{'prompts','sensor_depth_path','class_id','center','size','rotation','trajectory'}
    for row in rows['gt-box']:
        assert set(row['prompts'][0])=={'target_id','box_xyxy'}
    assert len(smoke_frames(frames,gt))==9
    assert set(vars(ObjectInput(np.zeros((2,2,3)),np.eye(3),(0,0,1,1))))=={'rgb','intrinsics','box_xyxy','mask','depth'}


def test_conditional_failures_stay_in_denominator_and_no_target_swapping():
    a=dict(box(),uid='a');b=dict(box(center=[5,0,0]),uid='b')
    gt={'f':[a,b]}
    records={'f':{'items':[dict(target_id='a',status='ok',detection=box(center=[5,0,0],cid=None,score=.9)),
                             dict(target_id='b',status='failed',detection=None,failure='empty_mask')]}}
    result,rows=conditional_success(gt,records)
    assert result['targets']==2 and result['failure_rate']==.5
    assert result['success']==[0.]*5
    assert result['mean_iou_including_failures']==0
    assert result['failure_reasons']=={'empty_mask':1}
    missing=copy.deepcopy(records);missing['f']['items'].pop()
    with pytest.raises(ValueError,match='缺失'):conditional_success(gt,missing)


def test_success_does_not_depend_on_ranking_but_ap_does():
    from ioutest.metrics import evaluate_ap
    gt={'a':[box()], 'b':[box()]}
    prediction={'a':[box(cid=None,score=.9)],'b':[box(center=[10,0,0],cid=None,score=.1)]}
    good=evaluate_ap(gt,prediction,class_agnostic=True)['summary']['iou']['map']
    prediction['a'][0]['score']=.1;prediction['b'][0]['score']=.9
    bad=evaluate_ap(gt,prediction,class_agnostic=True)['summary']['iou']['map']
    assert good>bad


def test_workpoint_filter_does_not_reuse_reindexed_iou_cache():
    gt={'f':[box()]}
    predictions={'f':[box(center=[10,0,0],cid=None,score=.1),box(cid=None,score=.8)]}
    result=workpoint(gt,predictions,.7)
    assert result['retained_predictions']==1
    assert result['values'][1]['precision']==result['values'][1]['recall']==1


def test_seed_independent_of_phase_order_and_cache_identity(tmp_path):
    assert object_seed(0,'f','x')==object_seed(0,'f','x')
    assert object_seed(0,'f','x')!=object_seed(0,'f','y')
    identity={'weights':'one','input':'fixed'}
    sig=bind_identity(tmp_path/'model.json',identity)
    assert bind_identity(tmp_path/'model.json',identity)==sig
    with pytest.raises(ValueError,match='变化'):bind_identity(tmp_path/'model.json',dict(identity,weights='two'))
    frame={'frame_id':'f','video_id':'scan','rgb_entry':'scan_1.png'}
    path=frame_path(tmp_path,frame)
    commit_record(path,frame,sig,items=[])
    assert read_record(path,'f',sig)['items']==[]
    with pytest.raises(ValueError,match='不一致'):read_record(path,'f','other')


def test_fixed_text_token_spans_never_use_target_labels():
    text,spans=fixed_prompt()
    assert len(spans)==17 and 'washing machine' in text and 'television' in text
    tokens=[0,1,2,3]
    assert category_tokens(tokens,[(0,0),(0,5),(6,10),(0,0)],'chair . bed',[(0,5),(6,10)])==[[1],[2]]
    with pytest.raises(ValueError):category_tokens(tokens,[(0,0)]*4,'chair',[(0,5)])


def test_speed_counts_frontend_depth_and_failed_targets():
    result = dict(frames=100, conditional=dict(targets=200, failures=20), timing=dict(
        total_predict_seconds=40, total_frontend_seconds=10, total_depth_seconds=50, historical_reuse=False))
    rgb = inference_speed(result)
    assert rgb['fps_predict'] == 2.5
    assert rgb['fps_pipeline'] == rgb['mean_pipeline_seconds_per_frame'] == 1
    assert rgb['mean_predict_seconds_per_target'] == .2
    assert rgb['mean_pipeline_seconds_per_target'] == .5
    result['timing']['total_depth_seconds'] = 0
    rgbd = inference_speed(result)
    assert rgbd['fps_pipeline'] == 2 and rgbd['mean_pipeline_seconds_per_target'] == .25
    result['conditional'] = None
    result['timing'].update(total_predict_seconds=0, total_frontend_seconds=0, historical_reuse=True)
    empty = inference_speed(result)
    assert empty['fps_pipeline'] is None and empty['targets'] is None
    assert empty['mean_predict_seconds_per_target'] is None and empty['historical_reuse']
    result['timing']['total_predict_seconds'] = float('nan')
    with pytest.raises(ValueError, match='计时'):inference_speed(result)


def test_complete_gt_box_report_and_missing_predictions(prepared,tmp_path):
    _,run=prepared
    cache=tmp_path/'cache';cache_images(run,cache);cache_depth(run,cache)
    config=dict(run_dir=str(run),image_cache=str(cache),display_score=.7,
                frontend=dict(box_threshold=.4,text_threshold=.3,max_detections=300))
    protocol,audit,frames,gt,inputs,_=prepare_inputs(config)
    root=tmp_path/'report'
    write_json(root/'gt-box/input_manifest.json',inputs['gt-box'])
    for model_id in ('wilddet3d-box','unipose9d-box-rgb','unipose9d-box-rgbd'):
        model_root=root/'gt-box/models'/model_id
        sig=bind_identity(model_root/'model.json',{'model':model_id})
        for f in frames:
            target=gt[f['frame_id']][0]
            det=Detection(None,.9,np.array(target['center']),np.array(target['size']),np.array(target['rotation'])).to_dict()
            commit_record(frame_path(model_root/'predictions',f),f,sig,
                          items=[dict(target_id=target['uid'],status='ok',detection=det)],
                          timing=dict(predict_seconds=.01))
        result=score_model(config,root,'gt-box','pilot',model_id,frames,gt,protocol,audit)
        assert result['conditional']['success']==[1.]*5
        assert result['metrics']['geometry']['full']['summary']['iou']['map']==pytest.approx(1)
    assemble(config,root,'gt-box','pilot',frames,gt,protocol,audit)
    document=(root/'gt-box/pilot/report.md').read_text()
    assert '补充 AP' in document and '不给' not in document
    assert '真值二维位置' in document and '固定关联' in document
    assert '完整流程 FPS' in document and '平均秒/目标' in document
    speed = read_json(root/'gt-box/pilot/speed.json')['unipose9d-box-rgb']
    assert speed['fps_pipeline'] == pytest.approx(100)
    assert speed['mean_pipeline_seconds_per_target'] == pytest.approx(.01)
    assert read_json(root/'gt-box/pilot/summary.json')['complete']
    frame_path(root/'gt-box/models/unipose9d-box-rgb/predictions',frames[0]).unlink()
    with pytest.raises(FileNotFoundError):score_model(config,root,'gt-box','pilot','unipose9d-box-rgb',frames,gt,protocol,audit)


def test_text_frontend_keeps_multiple_instances_and_reuses_depth(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from PIL import Image
    import torch
    from ioutest.suite_frontend import work
    from ioutest.suite_io import mask_checksum
    text,spans=fixed_prompt()
    ids=[101,*range(1,18),102]
    offsets=[(0,0),*spans,(0,0)]
    class Inputs(dict):
        def to(self,device):return self
    class Processor:
        def tokenizer(self,text,**kwargs):return dict(input_ids=ids,offset_mapping=offsets)
        def __call__(self,**kwargs):return Inputs(input_ids=torch.tensor([ids]))
    class Grounding:
        def to(self,device):return self
        def eval(self):return self
        def __call__(self,**inputs):
            logits=torch.full((1,3,len(ids)),-10.)
            logits[0,0,1]=9.;logits[0,1,1]=8.;logits[0,2,14]=9.
            return SimpleNamespace(logits=logits,pred_boxes=torch.tensor([[[.3,.3,.3,.3],[.7,.7,.3,.3],[.5,.5,.2,.2]]]))
    class Predictor:
        def __init__(self,model):pass
        def set_image(self,rgb):self.shape=rgb.shape[:2]
        def predict(self,**kwargs):return np.ones((1,*self.shape),bool),np.array([.9]),None
    package=tmp_path/'sam2';package.mkdir();(package/'__init__.py').write_text('')
    monkeypatch.setitem(sys.modules,'sam2',SimpleNamespace(__file__=str(package/'__init__.py')))
    monkeypatch.setitem(sys.modules,'sam2.build_sam',SimpleNamespace(build_sam2=lambda *a,**kw:None))
    monkeypatch.setitem(sys.modules,'sam2.sam2_image_predictor',SimpleNamespace(SAM2ImagePredictor=Predictor))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**kw:Processor()),
        AutoModelForZeroShotObjectDetection=SimpleNamespace(from_pretrained=lambda *a,**kw:Grounding())))
    calls=[]
    def depth(model,rgb):
        calls.append(1);return np.ones(rgb.shape[:2],np.float32),np.eye(3)
    monkeypatch.setitem(sys.modules,'moge_model',SimpleNamespace(load_moge_model=lambda *a:None,predict_depth_intrinsics=depth))
    repo=tmp_path/'repo';(repo/'infer').mkdir(parents=True);(repo/'infer/moge_model.py').write_text('# fixture')
    sam=tmp_path/'sam.pt';sam.write_bytes(b'fixture')
    moge=tmp_path/'moge.pt';moge.write_bytes(b'fixture')
    grounding=tmp_path/'grounding';grounding.mkdir();(grounding/'config.json').write_text('{}')
    image=tmp_path/'rgb.png';rgb=np.ones((8,12,3),np.uint8);Image.fromarray(rgb).save(image)
    frame=dict(frame_id='f',video_id='s',rgb_entry='s_1.png',rgb_path=str(image),image_hw=[8,12],upright_rgb_sha256=mask_checksum(rgb))
    cfg=dict(project_root=str(repo),sam2_config='fixture',sam2_checkpoint=str(sam),grounding_model=str(grounding),
             moge_checkpoint=str(moge),box_threshold=.4,text_threshold=.3,max_detections=300)
    root=tmp_path/'report'
    task=dict(config={'frontend':cfg},track='text',device='cpu',report_root=str(root),frames=[frame],phase='pilot',
              benchmark_id='fixture',input_manifest_sha256='text',rgb_manifest_sha256='rgb')
    assert work(task)['frames']==1
    model=read_json(root/'aux/text/model.json')
    record=read_record(frame_path(root/'aux/text/frames',frame),'f',model['signature'])
    assert len(record['items'])==3
    assert [i['class_id'] for i in record['items']].count(0)==2
    assert work(task)['reused']==1 and len(calls)==1
