"""Per-environment inference worker; input manifests contain no scoring GT."""

import importlib
import importlib.metadata
import sys
import time
from pathlib import Path

import numpy as np

from ..common import digest,file_digest,read_json,write_json
from ..schema import ModelInput
from ..suite_io import ObjectInput,load_array,object_seed
from ..adapters.unipose9d import ObjectFailure
from .common import bind,commit,export_detection,implementation,read_record,record_path,rgb
from .frontend import read_mask


def work(task):
    import torch
    torch.set_num_threads(4)
    root=Path(task['run_root']);track=task['track'];key=task['model_id']
    config=task['config'];model_config=config['models'][key];device=torch.device(config['device'])
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('模型进程不可见PPU，不回退CPU')
    options=dict(model_config['options'],device=str(device))
    adapter_path='ioutest.adapters.mood3d:Mood3DAdapter' if model_config['adapter']=='3dmood' else model_config['adapter']
    module_name,class_name=adapter_path.split(':')
    module=importlib.import_module(module_name)
    adapter=getattr(module,class_name)(options)
    write_json(root/'worker_status.json',dict(stage='model_loading',track=track,model=key))
    init=time.perf_counter();info=adapter.load();init_seconds=time.perf_counter()-init
    versions={}
    for name in ('torch','torchvision','numpy','scipy','transformers','vis4d','timm','sam3'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    is_object=track=='gt-box' or key=='unipose9d-text-rgb'
    is_wild_box=key=='wilddet3d-box'
    if not is_object:info['input_modalities']=['rgb','intrinsics','fixed_dataset_class_prompts']
    aux=root/'aux'/track
    front_identity=None if is_wild_box or not is_object else read_json(aux/'model.json')
    depth_identity=read_json(root/'aux/predicted_depth/model.json') if is_object and not is_wild_box else None
    identity=dict(benchmark_id=task['benchmark_id'],inputs=task['input_manifest'],classes=task['classes'],
        options=options,model=info,adapter_sha256=file_digest(module.__file__),
        adapter_sources={p.name:file_digest(p) for p in (Path(__file__).parents[1]/'adapters').glob('*.py')},
        code=implementation(['worker.py','common.py','frontend.py']),frontend=front_identity,depth=depth_identity,
        versions=versions,device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
    directory=root/track/'models'/key;signature=bind(directory/'model.json',identity)
    total=failures=reused=0;warmup=True;started=time.perf_counter()
    try:
        for done,frame in enumerate(task['frames'],1):
            write_json(root/'worker_status.json',dict(stage='inference',track=track,model=key,frames_completed=done-1,
                expected_frames=len(task['frames']),targets=total,failures=failures,reused=reused,frame_id=frame['frame_id']))
            path=record_path(directory/'predictions',frame)
            if path.exists():record=read_record(path,frame,signature);reused+=1
            else:
                image=rgb(frame);front={}
                if is_object:
                    if is_wild_box:proposals=frame['prompts'];depth=None
                    else:
                        auxiliary=read_record(record_path(aux/'frames',frame),frame,front_identity['signature'])
                        proposals=auxiliary['items'];front=auxiliary['timing']
                        depth_root=root/'aux/predicted_depth'
                        depth_record=read_record(record_path(depth_root/'frames',frame),frame,depth_identity['signature'])
                        depth=load_array(depth_root,depth_record['array'])
                    items=[]
                    for index,proposal in enumerate(proposals):
                        write_json(root/'worker_status.json',dict(stage='inference',track=track,model=key,
                            frames_completed=done-1,expected_frames=len(task['frames']),targets=total+len(items),failures=failures,
                            frame_id=frame['frame_id'],target_index=index+1,targets_in_frame=len(proposals)))
                        seed=object_seed(config['seed'],frame['frame_id'],proposal['target_id'])
                        if proposal.get('status')=='failed':
                            items.append(dict(target_id=proposal['target_id'],status='failed',failure=proposal['failure'],detection=None));continue
                        mask=None if is_wild_box else read_mask(aux,proposal,frame)
                        sample=ObjectInput(image,np.asarray(frame['intrinsics']),tuple(proposal['box_xyxy']),mask,depth)
                        if device.type=='cuda':torch.cuda.synchronize()
                        start=time.perf_counter()
                        try:
                            box,quality=adapter.predict_object(sample,seed)
                            if track=='text':box=dict(box,class_id=proposal['class_id'],score=proposal['detection_score'])
                            detection=export_detection(box,task['classes'])
                            item=dict(target_id=proposal['target_id'],status='ok',detection=detection,quality=quality,seed=seed)
                        except ObjectFailure as exc:
                            item=dict(target_id=proposal['target_id'],status='failed',detection=None,failure=str(exc),seed=seed)
                        if device.type=='cuda':torch.cuda.synchronize()
                        item.update(predict_seconds=time.perf_counter()-start,warmup=warmup);warmup=False;items.append(item)
                    detections=[i['detection'] for i in items if i['status']=='ok']
                    seconds=sum(i.get('predict_seconds',0.) for i in items)
                else:
                    sample=ModelInput(frame['frame_id'],image,np.asarray(frame['intrinsics']),
                        class_names=tuple(c['name'] for c in task['classes']),prompts=tuple(c['prompt'] for c in task['classes']))
                    if device.type=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter()
                    detections=[export_detection(b,task['classes']) for b in adapter.predict(sample)]
                    if device.type=='cuda':torch.cuda.synchronize()
                    seconds=time.perf_counter()-start;items=[]
                record=commit(path,frame,signature,items=items,detections=detections,
                    timing=dict(predict_seconds=seconds,frontend_seconds=front.get('frontend_seconds',0.),
                        depth_seconds=front.get('depth_seconds',0.)))
            total+=len(record['items']) if is_object else len(record['detections'])
            failures+=sum(i['status']!='ok' for i in record['items'])
            if done%5==0 or done==len(task['frames']):
                print(f'{key} {done}/{len(task["frames"])} 帧，{total} 目标，失败 {failures}，复用 {reused}',flush=True)
        result=dict(complete=True,frames=len(task['frames']),targets=total,failures=failures,reused=reused,
            init_seconds=init_seconds,wall_seconds=time.perf_counter()-started)
        write_json(root/track/'full'/(key+'_inference.json'),result)
        return result
    finally:adapter.close()


if __name__=='__main__':
    task=read_json(sys.argv[1])
    if task.get('job')=='frontend':
        from .frontend import work
    print(__import__('json').dumps(work(task),ensure_ascii=False,indent=2))
