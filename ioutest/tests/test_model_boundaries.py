"""Checkpoint provenance, frontend numerics and real local-process communication."""
import threading
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from ioutest.checkpoints import load_head
from ioutest.common import file_digest, write_json
from ioutest.experiment_config import load_config
from ioutest.worker_rpc import pipe_pair, receive, send
from ioutest.suite_io import ObjectInput
from ioutest.suite_frontend import grounding_probabilities
from .test_experiment import box, prepared


def test_interpreter_symlink_preserves_virtual_environment(prepared,tmp_path):
    path,_=prepared;cfg=__import__('json').loads(path.read_text())
    python=tmp_path/'venv/bin/python';python.parent.mkdir(parents=True);python.symlink_to('/usr/bin/python3')
    cfg['models']['3dmood']['python']=str(python);write_json(path,cfg)
    assert load_config(path)['models']['3dmood']['python']==str(python)
    assert load_config(path)['models']['3dmood']['python']!=str(python.resolve())


def test_strict_head_loading_and_wrong_base_rejected(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bbox3d_head=torch.nn.Linear(2,2)
    model=Model();base=tmp_path/'base.pt';torch.save(dict(state_dict=model.state_dict()),base)
    adapter=SimpleNamespace(model=model,checkpoint=base);info=dict(checkpoint_sha256=file_digest(base))
    head=tmp_path/'head.pt';weights={k:torch.full_like(v,7) for k,v in model.bbox3d_head.state_dict().items()}
    state=dict(model_id='3dmood',base_sha256=info['checkpoint_sha256'],head=weights,identity='training',benchmark_id='data',epochs_completed=4)
    torch.save(state,head)
    result=load_head(adapter,'3dmood',dict(head_checkpoint=str(head),training_benchmark_id='data',epochs_completed=4),info)
    assert result['head_sha256']==file_digest(head)
    assert torch.equal(model.bbox3d_head.weight,weights['weight'])
    state['base_sha256']='wrong';torch.save(state,head)
    with pytest.raises(ValueError,match='不匹配'):load_head(adapter,'3dmood',dict(head_checkpoint=str(head)),info)
    torch.save(dict(state_dict={'unexpected':torch.ones(1)}),base)
    with pytest.raises(ValueError,match='参数缺失'):load_head(adapter,'3dmood',{},info)


def test_rpc_transfers_numpy_arrays_and_detects_closed_connection():
    left,right=pipe_pair();errors=[]
    def server():
        try:
            request=receive(right)
            send(right,dict(rgb=request['rgb']+1,mask=np.ones((3,4),bool)))
        except BaseException as exc:errors.append(exc)
        finally:right.close()
    thread=threading.Thread(target=server);thread.start()
    send(left,dict(rgb=np.zeros((3,4,3),np.uint8)))
    response=receive(left);thread.join(timeout=5)
    assert not errors and not thread.is_alive()
    assert np.all(response['rgb']==1) and response['mask'].dtype==bool
    with pytest.raises(RuntimeError,match='断开'):receive(left)
    left.close()


def test_resident_frontend_process_protocol_and_shutdown(monkeypatch):
    from ioutest.worker_rpc import FrontendClient
    native_popen = subprocess.Popen
    program = """import sys
from ioutest.worker_rpc import PipeConnection,receive,send
c=PipeConnection(int(sys.argv[1]),int(sys.argv[2]))
initial=receive(c)
send(c,dict(ok=True,identity=dict(track=initial['track'])))
while True:
    request=receive(c)
    if request.get('close'):break
    send(c,dict(ok=True,result=dict(rgb=request['rgb']+1,depth=request['depth'])))
c.close()
"""
    def start(args,**kwargs):
        return native_popen([sys.executable,'-B','-c',program,*args[-2:]],**kwargs)
    monkeypatch.setattr(subprocess,'Popen',start)
    client=FrontendClient(sys.executable,{},'text',[],{})
    try:
        assert client.identity==dict(track='text')
        response=client.predict(np.zeros((4,4,3),np.uint8),np.eye(3),[],np.ones((4,4)))
        assert np.all(response['rgb']==1) and np.all(response['depth']==1)
    finally:client.close()
    assert client.process.returncode==0


def test_native_negative_infinity_padding_and_real_token_errors():
    output=SimpleNamespace(logits=torch.tensor([[[1.,2.,float('-inf')]]]),pred_boxes=torch.ones((1,1,4)))
    p=grounding_probabilities(output,torch.tensor([[1,1]]))
    assert p[0,0,2]==0
    output.logits[0,0,0]=float('nan')
    with pytest.raises(RuntimeError,match='非有限'):grounding_probabilities(output,torch.tensor([[1,1]]))


def test_frontend_keeps_depth_failures_fatal_and_masks_explicit():
    from ioutest.frontend_worker import Frontend
    front=Frontend.__new__(Frontend);front.torch=torch;front.device='cpu';front.track='gt-box'
    front.moge=object();rgb=np.zeros((4,4,3),np.uint8)
    class Sam:
        def set_image(self,rgb):pass
        def predict(self,**kwargs):return np.zeros((1,4,4),bool),np.array([.5]),None
    front.predictor=Sam();front.predict_depth=lambda *a:(np.full((4,4),np.nan),None)
    with pytest.raises(RuntimeError,match='MoGe'):front.predict(rgb,np.eye(3),[])
    depth=np.ones((4,4),np.float32);depth[0,0]=np.nan
    front.predict_depth=lambda *a:(depth,None)
    result=front.predict(rgb,np.eye(3),[dict(target_id='id',box_xyxy=[0,0,3,3])])
    assert result['items'][0]['failure']=='empty_predicted_mask'
    assert result['depth'][0,0]==0


def test_gt_box_worker_path_has_no_class_or_gt_pose(prepared,monkeypatch):
    from ioutest import experiment_worker
    from ioutest.experiment import make_tasks
    from ioutest.experiment_config import select
    from ioutest.experiment_data import load_dataset
    _,c=prepared;c['models']['3dmood']['family']='wilddet3d'
    # Use a matching fixture source folder; production adapters remain unmodified.
    repo=Path(c['models']['3dmood']['options']['project_root']);(repo/'wilddet3d').mkdir();(repo/'wilddet3d/test.py').write_text('fixture=1')
    data={'test':load_dataset(c['datasets']['test'])};task=make_tasks(c,select(c,tracks=['gt-box']),data)[0]
    calls=[]
    class Adapter:
        def load(self):return dict(checkpoint_sha256='fixture')
        def predict_object(self,sample,seed):
            assert isinstance(sample,ObjectInput) and not hasattr(sample,'class_id') and not hasattr(sample,'center')
            calls.append(seed)
            if len(calls)==1:
                from ioutest.adapters.unipose9d import ObjectFailure
                raise ObjectFailure('no_single_prediction')
            return dict(box(cid=None),score=.5),{}
        def close(self):pass
    monkeypatch.setattr(experiment_worker,'create_adapter',lambda task:Adapter())
    monkeypatch.setattr(experiment_worker,'load_head',lambda adapter,family,variant,info:info)
    result=experiment_worker.work(dict(task,mode='infer'))
    assert result['complete'] and len(calls)==2
    from ioutest.experiment_score import score
    metrics=score(task,data['test'],c['metrics'])
    assert metrics['conditional']['failure_rate']==.5
    assert metrics['diagnostics']['matched']==1 and metrics['diagnostics']['missed']==1
