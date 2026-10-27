"""Prepared-data isolation, real scoring, source binding and resumable worker tests."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation
from ioutest.common import digest, file_digest, read_json, write_json, write_jsonl
from ioutest.experiment_config import load_config, select
from ioutest.experiment_data import load_dataset, input_frames, audit_training, read_rgb
from ioutest.experiment import execute, make_tasks
from ioutest.geometry_metrics import geometry_error, diagnostics, associate
from ioutest.records import commit, read_record, record_path
from ioutest.timing import summarize


def box(center=(0., 0., 2.), size=(.2, .4, .3), rotation=None, cid=0, uid='target', score=None):
    b = dict(center=list(center), size=list(size), rotation=(np.eye(3) if rotation is None else rotation).tolist(),
             class_id=cid, uid=uid, ignore=False, object_bop_id=1)
    if score is not None: b['score'] = score
    return b


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / 'prepared'; root.mkdir(); rows = []
    for i in range(2):
        rgb = root / f'{i}.png'; Image.fromarray(np.full((40, 40, 3), 30+i, np.uint8)).save(rgb)
        gt = root / f'{i}.json'; write_json(gt, [box()])
        rows.append(dict(frame_id=str(i), sequence_id='heldout', image_hw=[40,40], intrinsics=[[40,0,20],[0,40,20],[0,0,1]],
            rgb_path=str(rgb), rgb_sha256=file_digest(rgb), gt_path=str(gt), gt_sha256=file_digest(gt)))
    write_jsonl(root / 'eval.jsonl', rows)
    write_json(root / 'protocol.json', dict(classes=[dict(name='cup',prompt='cup')], eval_sha256=file_digest(root/'eval.jsonl'), benchmark_id='fixture'))
    repo = tmp_path/'repo'; (repo/'opendet3d').mkdir(parents=True); (repo/'opendet3d/model.py').write_text('fixture=1\n')
    weight = tmp_path/'base.pt'; weight.write_bytes(b'fixture')
    cache = tmp_path/'bert'; model = cache/'hub/models--bert-base-uncased'
    (model/'refs').mkdir(parents=True); (model/'refs/main').write_text('fixture')
    snapshot = model/'snapshots/fixture'; snapshot.mkdir(parents=True)
    for name in ('config.json','tokenizer_config.json','vocab.txt','model.safetensors'):
        (snapshot/name).write_bytes(b'fixture-cache')
    c = dict(schema_version='ioutest.experiment.v1', output=dict(root=str(tmp_path/'output')),
        execution=dict(device='cpu',seed=1,threads=1), timing=dict(warmup_frames=0),
        datasets=dict(test=dict(format='hot3d_prepared',root=str(root),manifest='eval.jsonl',protocol='protocol.json')),
        models={'3dmood':dict(family='3dmood',python=sys.executable,environment=dict(HF_HOME=str(cache)),options=dict(project_root=str(repo),checkpoint=str(weight)),
                              variants=dict(official={}))})
    path = tmp_path/'config.json'; write_json(path,c)
    return path, load_config(path)


def test_prepared_only_and_no_output_for_check(prepared,monkeypatch):
    from ioutest import experiment
    path,c=prepared; root=Path(c['datasets']['test']['root']); before={str(p):file_digest(p) for p in root.iterdir()}
    monkeypatch.setattr(experiment,'check_environments',lambda tasks:dict(ready=True,environments=[]))
    result=execute(c,select(c),'check')
    assert result['ready'] and result['datasets']['test']['frames']==2
    assert not Path(c['output']['root']).exists()
    assert before=={str(p):file_digest(p) for p in root.iterdir()}


def test_missing_prepared_file_fails_without_repair(prepared):
    _,c=prepared; root=Path(c['datasets']['test']['root']); (root/'0.png').unlink()
    with pytest.raises(FileNotFoundError,match='不会制作'):load_dataset(c['datasets']['test'])
    assert not (root/'0.png').exists() and not Path(c['output']['root']).exists()


def test_input_boundary_and_gt_identity(prepared):
    _,c=prepared; data=load_dataset(c['datasets']['test'])
    for row in input_frames(data,'text'):
        assert not set(row)&{'prompts','gt','gt_path','center','size','rotation','state_id','class_ids','depth_path'}
    row=input_frames(data,'gt-box')[0]
    assert set(row['prompts'][0])=={'target_id','box_xyxy'}
    with pytest.raises(ValueError,match='实测深度'):input_frames(data,'text',True)
    Path(data['frames'][0]['rgb_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='摘要'):read_rgb(data['frames'][0])


def test_training_overlap_exact_and_cross_view_state(prepared,tmp_path):
    _,c=prepared; data=load_dataset(c['datasets']['test']); manifest=tmp_path/'train.jsonl'
    write_jsonl(manifest,[dict(frame_id='0')])
    with pytest.raises(ValueError,match='重叠'):audit_training(dict(training_manifest=str(manifest)),data)
    data['frames'][0]['state_id']='same_state'
    write_jsonl(manifest,[dict(frame_id='other_camera',state_id='same_state')])
    with pytest.raises(ValueError,match='重叠'):audit_training(dict(training_manifest=str(manifest)),data)
    assert audit_training({},data)['status']=='unknown'


def test_selection_is_explicit_and_dependencies_only_selected(prepared):
    _,c=prepared
    c['models']['broken']=dict(family='unipose9d',python='/missing/python',options={},variants=dict(official={}))
    selections=select(c,models=['3dmood']);data={'test':load_dataset(c['datasets']['test'])}
    assert len(make_tasks(c,selections,data))==1
    with pytest.raises(ValueError,match='不支持'):select(c,models=['3dmood'],tracks=['gt-box'])
    with pytest.raises(ValueError,match='未配置'):select(c,models=['3dmood'],variants=['missing'])
    with pytest.raises(ValueError,match='未知'):select(c,models=['no'])


@pytest.mark.parametrize('sam3_present,device', [(True, 'cpu'), (False, 'cpu'), (True, 'cuda:0')])
def test_readiness_finds_vendored_sam3_without_shell_paths(tmp_path, monkeypatch, sam3_present, device):
    import venv
    from ioutest.experiment import check_environments

    # An isolated interpreter ensures an installed SAM3 cannot hide missing sources.
    env = tmp_path / 'env'
    venv.EnvBuilder(with_pip=False).create(env)
    site = next((env / 'lib').glob('python*/site-packages'))
    (site / 'torch.py').write_text(
        "from types import SimpleNamespace\n"
        "__version__ = 'test'\n"
        "def device(name):\n"
        "    kind, _, index = name.partition(':')\n"
        "    return SimpleNamespace(type=kind, index=int(index) if index else None)\n"
        "cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)\n")
    repo = tmp_path / 'wilddet-project'
    for module in ['wilddet3d', 'vis4d']:
        package = repo / module
        package.mkdir(parents=True)
        (package / '__init__.py').write_text("raise AssertionError('precheck must not import models')\n")
    if sam3_present:
        package = repo / 'third_party/sam3/sam3'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text("raise AssertionError('precheck must not import SAM3')\n")
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.delenv('PYTHONHOME', raising=False)
    monkeypatch.chdir(tmp_path)
    task = dict(model=dict(family='wilddet3d', python=str(env / 'bin/python')),
                options=dict(project_root=str(repo)), execution=dict(device=device))
    result = check_environments([task])
    assert result['ready'] == (sam3_present and device == 'cpu')
    detail = json.loads(result['environments'][0]['detail'])
    assert detail['missing_modules'] == ([] if sam3_present else ['sam3'])
    assert detail['device_ready'] == (device == 'cpu')


def test_geometry_errors_and_equivalent_axes():
    target=box();pred=box(center=(.03,.04,2),size=(.22,.4,.3))
    e=geometry_error(pred,target)
    assert e['center_error_m']==pytest.approx(.05)
    assert e['size_abs_xyz_m']==pytest.approx([.02,0,0])
    assert e['size_relative_xyz']==pytest.approx([.1,0,0])
    p=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    same=box(size=(.4,.2,.3),rotation=p)
    e=geometry_error(same,target)
    assert e['rotation_error_deg']==pytest.approx(0)
    assert e['size_abs_xyz_m']==pytest.approx([0,0,0])
    tilted=box(rotation=Rotation.from_euler('z',15,degrees=True).as_matrix())
    assert geometry_error(tilted,target)['rotation_error_deg']==pytest.approx(15)


def test_continuous_symmetry_requires_explicit_metadata():
    t=box(size=(.2,.2,.4));p=box(size=(.2,.2,.4),rotation=Rotation.from_euler('z',35,degrees=True).as_matrix())
    assert geometry_error(p,t)['rotation_error_deg']==pytest.approx(35)
    assert geometry_error(p,t,dict(continuous_axis=2))['rotation_error_deg']==pytest.approx(0)
    with pytest.raises(ValueError,match='尺寸'):geometry_error(box(),box(),dict(continuous_axis=2))


def test_missing_targets_are_not_zero_error(prepared):
    _,c=prepared;data=load_dataset(c['datasets']['test'])
    records={f['frame_id']:dict(detections=[],items=[]) for f in data['frames']}
    records['0']['detections']=[dict(box(),score=.9)]
    d,rows=diagnostics(data,records,'text',c['metrics'])
    assert d['matched']==1 and d['missed']==1 and d['association_recall']==.5
    assert rows[1]['center_error_m'] is None and d['errors']['center_error_m']['count']==1
    assert d['localization_success']['center@0.02']['rate']==.5


def test_target_binding_never_rematches(prepared):
    _,c=prepared;data=load_dataset(c['datasets']['test']);records={}
    for f in data['frames']:
        pred=dict(box(center=(4,0,2)),score=.9)
        records[f['frame_id']]=dict(detections=[pred],items=[dict(target_id='target',status='ok',detection=pred)])
    d,_=diagnostics(data,records,'gt-box',c['metrics'])
    assert d['matched']==2 and d['errors']['center_error_m']['median']==4
    assert d['localization_success']['center@0.02']['rate']==0
    records['0']['items'][0]['target_id']='other'
    with pytest.raises(ValueError,match='关联错误'):diagnostics(data,records,'gt-box',c['metrics'])


def test_ignored_targets_and_duplicates_are_one_to_one(prepared):
    _,c=prepared;data=load_dataset(c['datasets']['test']);f=data['frames'][0]
    target=box();ignored=dict(box(center=(.6,0,2),uid='ignored'),ignore=True)
    predictions=[dict(target,score=.9),dict(target,score=.8),dict(ignored,score=.7)]
    pairs,left,ignored_pred=associate(predictions,[target,ignored],f,.5,False)
    assert len(pairs)==1 and len(left)==1 and len(ignored_pred)==1


def test_latency_is_not_a_sum_disguised_as_wall_time():
    old=summarize([dict(timing=dict(predict_seconds=1,frontend_seconds=2,depth_seconds=3))])
    assert old['status']=='unavailable' and old['mean_ms'] is None and old['stage_sum_mean_ms']==6000
    result=summarize([dict(timing=dict(pipeline_wall_seconds=v,warmup=w)) for v,w in [(10,True),(.1,False),(.3,False)]])
    assert result['mean_ms']==pytest.approx(200) and result['p95_ms']==pytest.approx(290)
    assert result['measured_frames']==2 and result['warmup_frames']==1


class FixtureAdapter:
    def __init__(self):self.calls=0
    def load(self):return dict(checkpoint_sha256='fixture')
    def predict(self,sample):
        self.calls+=1
        assert not hasattr(sample,'gt') and not hasattr(sample,'prompts_for_objects')
        return [dict(box(),score=.9)]
    def close(self):pass


def test_worker_resume_real_scoring_and_metric_only_change(prepared,monkeypatch):
    from ioutest import experiment,experiment_worker
    _,c=prepared;adapter=FixtureAdapter()
    monkeypatch.setattr(experiment_worker,'create_adapter',lambda task:adapter)
    monkeypatch.setattr(experiment_worker,'load_head',lambda adapter,family,variant,info:info)
    monkeypatch.setattr(experiment,'check_environments',lambda tasks:dict(ready=True,environments=[]))
    monkeypatch.setattr(experiment,'run_worker',lambda task,mode:experiment_worker.work(dict(task,mode=mode)))
    result=execute(c,select(c),'all');assert result['complete'] and adapter.calls==2
    directory=Path(c['output']['root'])/'runs/test/3dmood/official/text'
    metrics=read_json(directory/'metrics.json')
    assert metrics['geometry']['summary']['AP3D@0.25']==1
    assert metrics['diagnostics']['errors']['center_error_m']['median']==0
    assert metrics['latency']['status']=='measured'
    execute(c,select(c),'all');assert adapter.calls==2
    c['metrics']['center_thresholds']=[.01,.02]
    execute(c,select(c),'score');assert adapter.calls==2
    assert 'APcenter@0.01m' in read_json(directory/'metrics.json')['geometry']['summary']
    c['execution']['seed']=2
    with pytest.raises(ValueError,match='不能复用'):execute(c,select(c),'score')


def test_partial_prediction_or_tampering_is_fatal(prepared,monkeypatch):
    from ioutest import experiment,experiment_worker
    _,c=prepared
    monkeypatch.setattr(experiment_worker,'create_adapter',lambda task:FixtureAdapter())
    monkeypatch.setattr(experiment_worker,'load_head',lambda adapter,family,variant,info:info)
    monkeypatch.setattr(experiment,'check_environments',lambda tasks:dict(ready=True,environments=[]))
    monkeypatch.setattr(experiment,'run_worker',lambda task,mode:experiment_worker.work(dict(task,mode=mode)))
    execute(c,select(c),'infer');directory=Path(c['output']['root'])/'runs/test/3dmood/official/text'
    record_path(directory,'1').unlink()
    with pytest.raises(FileNotFoundError):execute(c,select(c),'score')
    execute(c,select(c),'infer');model=read_json(directory/'model.json');p=record_path(directory,'0')
    import gzip
    value=json.loads(gzip.decompress(p.read_bytes()));value['detections']=[]
    p.write_bytes(gzip.compress(json.dumps(value).encode()))
    with pytest.raises(ValueError,match='校验'):read_record(p,'0',model['provenance']['request_id'],model['signature'])


def test_runtime_error_is_not_a_detection_failure(prepared,monkeypatch):
    from ioutest import experiment_worker
    _,c=prepared;data={'test':load_dataset(c['datasets']['test'])};task=make_tasks(c,select(c),data)[0]
    adapter=FixtureAdapter()
    def failed(sample):raise RuntimeError('out of memory')
    adapter.predict=failed
    monkeypatch.setattr(experiment_worker,'create_adapter',lambda task:adapter)
    monkeypatch.setattr(experiment_worker,'load_head',lambda adapter,family,variant,info:info)
    with pytest.raises(RuntimeError,match='memory'):experiment_worker.work(dict(task,mode='infer'))
    assert not record_path(task['run_root'],'0').exists()


def test_old_cli_is_removed():
    from ioutest.experiment import main
    with pytest.raises(SystemExit):main(['benchmark'])
    assert not (Path(__file__).parents[1]/'run_hot3d_benchmark.sh').exists()
