import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from hot3d_ft.boxeval import evaluate as boxes
from hot3d_ft.boxeval.report import MARKER, merge
from hot3d_ft.common import checksum, data_path, digest, jsonlines, read, source_code, write
from test_data import fixture_dataset


def dataset(tmp_path):
    cfg, frames, protocol = fixture_dataset(tmp_path)
    cfg['device'] = 'cpu'
    root = data_path(cfg)
    for index, f in enumerate(frames):
        labels = read(f['gt_path'])
        for b in labels:
            b['projected_bbox'] = [10., 20., 40., 60.]
        write(f['gt_path'], labels)
        p = root/f'image-{index}.png'
        Image.fromarray(np.full((768, 768, 3), index*100, dtype=np.uint8)).save(p)
        f.update(rgb_path=str(p), rgb_sha256=checksum(p), gt_sha256=digest(labels))
    protocol.update(train_sha256=digest(frames[:1]), eval_sha256=digest(frames[1:]))
    write(root/'protocol.json', protocol)
    jsonlines(root/'train.jsonl', frames[:1]);jsonlines(root/'eval.jsonl', frames[1:])
    report = Path(cfg['root'])/'report'
    write(report/'results.json', dict(models={'legacy-text': {'complete': True}}))
    (report/'report.md').write_text('# HOT3D：三维头微调前后对比\n\n原文本AP：12.34\n')
    return cfg, frames, protocol


def adapter(monkeypatch, cfg, *, interrupted=None, failed=None):
    from ioutest.adapters.unipose9d import ObjectFailure
    state = dict(calls=0, interrupted=interrupted, failed=failed, closed=0)

    class Adapter:
        def close(self):
            state['closed'] += 1

    def predict(sample, seed):
        state['calls'] += 1
        assert set(vars(sample)) == {'rgb','intrinsics','box_xyxy','mask','depth'}
        assert sample.mask is None and sample.depth is None
        assert sample.box_xyxy == (10., 20., 40., 60.)
        if state['calls'] == state['interrupted']:
            raise RuntimeError('injected fatal device error')
        if state['calls'] == state['failed']:
            raise ObjectFailure('counted_geometry_failure')
        return dict(center=[0.,0.,1.],size=[.1,.2,.3],rotation=np.eye(3).tolist(),
                    class_id=None,score=.01), {'native_score': .01}

    monkeypatch.setattr(boxes, 'load_box_adapter', lambda cfg, variant: (
        Adapter(), dict(model='test', variant=variant), predict))
    return state


def test_prompt_boundary_excludes_gt_pose_and_category(tmp_path):
    cfg, frames, _ = dataset(tmp_path)
    prompts = boxes.prompts_for_frame(frames[1])
    assert len(prompts) == 33
    assert set(prompts[0]) == {'target_id','box_xyxy'}
    manifest = boxes.prompt_manifest(frames[1:])[0]
    assert not {'gt_path','class_id','size','center','rotation'} & set(manifest)
    labels = read(frames[1]['gt_path']);labels[0]['ignore'] = True
    write(frames[1]['gt_path'], labels)
    assert len(boxes.prompts_for_frame(frames[1])) == 32


def test_invalid_projection_is_not_silently_skipped(tmp_path):
    cfg, frames, _ = dataset(tmp_path)
    labels = read(frames[1]['gt_path']);labels[0]['projected_bbox']=[-1,20,40,60]
    write(frames[1]['gt_path'],labels)
    with pytest.raises(ValueError, match='投影框'):
        boxes.prompts_for_frame(frames[1])


def test_target_resume_failures_scores_pairing_and_merged_report(tmp_path, monkeypatch):
    cfg, frames, protocol = dataset(tmp_path)
    state = adapter(monkeypatch,cfg,interrupted=5,failed=3)
    with pytest.raises(RuntimeError,match='fatal device'):
        boxes.infer(cfg,'baseline')
    root=Path(cfg['root']);directory=root/'evaluation/gt-box/models/wilddet3d-baseline'
    assert len(list((directory/'objects').rglob('*.gz'))) == 4
    assert not list((directory/'predictions').rglob('*.gz'))
    assert state['closed']==1
    state['interrupted']=None
    result=boxes.infer(cfg,'baseline')
    assert result['targets']==33 and result['reused_objects']==4 and result['failures']==1
    before=state['calls']
    assert boxes.infer(cfg,'baseline')['reused_frames']==1
    assert state['calls']==before
    scored=boxes.score(cfg,'baseline')
    assert scored['conditional']['targets']==33
    assert scored['conditional']['failures']==1
    assert scored['conditional']['success']['success3D@0.25']==pytest.approx(32/33)
    assert scored['semantic'] is None
    assert scored['predictions']==32  # Confidence .01 is retained, below display_score .7.
    partial=merge(cfg)
    assert not partial['box_evaluation_complete']
    assert '尚未完成完整评分' in Path(partial['report']).read_text()
    state=adapter(monkeypatch,cfg)
    boxes.infer(cfg,'finetuned');boxes.score(cfg,'finetuned')
    result=merge(cfg)
    assert result['box_evaluation_complete']
    assert read(root/'report/results.json')['models']=={'legacy-text':{'complete':True}}
    paired=read(root/'report/results.json')['gt_box']['paired']['success3D@0.25']
    assert paired['improved']==1 and paired['regressed']==0 and paired['both_success']==32
    text=Path(result['report']).read_text()
    assert '原文本AP：12.34' in text and '96.9697' in text and '100.0000' in text
    assert len(list((root/'report/visualizations/gt-box').rglob('*.png')))==2
    assert merge(cfg)['box_evaluation_complete']
    assert Path(result['report']).read_text().count(MARKER)==1
    target=root/'evaluation/gt-box/full/metrics/wilddet3d-baseline_targets.csv'
    target.write_text(target.read_text()+'corrupt\n')
    with pytest.raises(ValueError,match='审计'):
        merge(cfg)
    with pytest.raises(ValueError,match='损坏'):
        boxes.score(cfg,'baseline')


def test_pending_report_does_not_invent_box_scores(tmp_path):
    cfg, _, _=dataset(tmp_path)
    result=merge(cfg, tmp_path/'preview')
    assert not result['box_evaluation_complete']
    assert result['scored_variants']==[]
    text=Path(result['report']).read_text()
    assert '原文本AP：12.34' in text
    assert '暂无可报告分数' in text
    assert '### 主要结果' not in text
    assert MARKER not in (tmp_path/'report/report.md').read_text()


def test_conditional_binding_does_not_rematch_to_other_target():
    from ioutest.hot3d.metrics import conditional,evaluate
    first=dict(uid='a',class_id=0,object_bop_id=1,ignore=False,
               center=[0,0,1],size=[.1,.1,.1],rotation=np.eye(3).tolist())
    second=dict(first,uid='b',center=[1,0,1],object_bop_id=2)
    predictions=[dict(second,score=.9,class_id=None),dict(first,score=.8,class_id=None)]
    records={'f':dict(items=[dict(target_id=uid,status='ok',detection=p)
                             for uid,p in zip(('a','b'),predictions)])}
    gt={'f':[first,second]};limits=[('iou',.25)]
    conditional_result,_=conditional(gt,records,limits)
    ap=evaluate(gt,{'f':predictions},[dict(name='object')],limits)
    assert conditional_result['success']['success3D@0.25']==0
    assert ap['summary']['AP3D@0.25']==pytest.approx(1.)


def test_box_loader_uses_the_existing_strict_finetune_loader(tmp_path,monkeypatch):
    from hot3d_ft import models
    cfg,_,_=dataset(tmp_path)
    calls=[]
    class Model:
        def eval(self):pass
    monkeypatch.setattr(models,'load_adapter',lambda *args:(calls.append(args) or (object(),Model(),{})))
    boxes.load_box_adapter(cfg,'baseline');boxes.load_box_adapter(cfg,'finetuned')
    assert calls[0][3] is None
    assert calls[1][3]==tmp_path/'runs/wilddet3d/checkpoints/final_head.pt'


def test_box_extension_is_outside_legacy_training_fingerprint():
    files=source_code()
    assert not any('/boxeval/' in name or name.endswith('/run.py') for name in files)


@pytest.mark.parametrize('args,routes',[
    (['evaluate','--track','text','--model','3dmood','--variant','baseline'],['legacy']),
    (['evaluate','--track=gt-box','--variant','baseline'],['box']),
    (['report','--report-dir','/tmp/preview'],['box']),
    (['box-supplement','--root','/tmp/products'],['box']),
    (['run','--root','/tmp/products'],['legacy','box']),
])
def test_entrypoint_preserves_text_and_routes_supplement(args,routes,monkeypatch):
    import importlib.util
    from hot3d_ft.boxeval import cli
    path=Path(__file__).parents[1]/'run.py'
    spec=importlib.util.spec_from_file_location('hot3d_test_entrypoint',path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    seen=[]
    monkeypatch.setattr(entry,'legacy_main',lambda argv:(seen.append(('legacy',argv)) or 0))
    monkeypatch.setattr(cli,'main',lambda argv:(seen.append(('box',argv)) or 0))
    assert entry.main(args)==0
    assert [r[0] for r in seen]==routes
    if args[0]=='run':assert seen[-1][1][0]=='box-supplement'
    if '--track' in args and 'text' in args:assert '--track' not in seen[0][1]


def test_unavailable_ppu_fails_before_creating_products(tmp_path,monkeypatch):
    import torch
    from hot3d_ft.common import config
    cfg=config(root=tmp_path/'never-created')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='原生Terminal'):
        boxes.infer(cfg,'baseline')
    assert not Path(cfg['root']).exists()
