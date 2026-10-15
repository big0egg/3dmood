"""Audit all predictions before scoring; export AP, target outcomes and speed."""

from pathlib import Path

from ..benchmark_report import write_csv
from ..common import digest,file_digest,read_json,write_json
from ..suite_report import inference_speed
from .common import implementation,read_record,record_path,validate_box
from .metrics import conditional,evaluate,label,thresholds


def score(config,root,track,key,frames,gt,protocol):
    root=Path(root);directory=root/track/'full';source=root/track/'models'/key
    model=read_json(source/'model.json');records={};predictions={};checks={};timings=[]
    for done,frame in enumerate(frames,1):
        record=read_record(record_path(source/'predictions',frame),frame,model['signature'])
        for box in record['detections']:validate_box(box,protocol['classes'],True)
        if track=='gt-box' or key=='unipose9d-text-rgb':
            if [i['detection'] for i in record['items'] if i['status']=='ok']!=record['detections']:
                raise ValueError('逐目标记录与帧预测列表不一致')
        records[frame['frame_id']]=record;predictions[frame['frame_id']]=record['detections']
        checks[frame['frame_id']]=digest(record);timings.append(record['timing'])
        if done%100==0:print(f'HOT3D 评分核验 {key} {done}/{len(frames)}',flush=True)
    identity=dict(model=model['signature'],benchmark_id=protocol['benchmark_id'],predictions=digest(checks),
        metrics=config['metrics'],score_floor=config['display_score'],
        code=implementation(['score.py','metrics.py','common.py']),
        interpolation_sha256=file_digest(Path(__file__).parents[1]/'metrics.py'),
        geometry_sha256=file_digest(Path(__file__).parents[1]/'geometry.py'))
    output=directory/'metrics'/(key+'.json')
    if output.exists() and read_json(output).get('identity')==identity and read_json(output).get('complete'):
        return read_json(output)
    limits=thresholds(config);classes=protocol['classes'];conditional_result=None;workpoint=None
    def progress(step):
        write_json(root/'worker_status.json',dict(stage='scoring',track=track,model=key,step=step,frames_verified=len(frames)))
        print(f'HOT3D {key} 正在计算 {step}',flush=True)
    if track=='gt-box':
        progress('conditional_success')
        conditional_result,details=conditional(gt,records,limits)
        names=[label(k,t,False) for k,t in limits]
        write_csv(directory/'metrics'/(key+'_targets.csv'),
            ['frame_id','target_id','object_bop_id','class_id','failed','failure','iou','center_distance_m',*names],
            [[*[r[n] for n in ('frame_id','target_id','object_bop_id','class_id','failed','failure','iou','center_distance_m')],
              *[r['hits'][n] for n in names]] for r in details])
        score_gt={f:[b for b in boxes if not b.get('ignore',False)] for f,boxes in gt.items()}
        progress('supplementary_geometry_AP');geometry=evaluate(score_gt,predictions,classes,limits)
        semantic=None
    else:
        cache={};progress('geometry_AP');geometry=evaluate(gt,predictions,classes,limits,pairs=cache)
        progress('semantic_AP');semantic=evaluate(gt,predictions,classes,limits,semantic=True,pairs=cache)
        progress('confidence_workpoint');workpoint=evaluate(gt,predictions,classes,limits,score_floor=config['display_score'],pairs=cache)
    timing=dict(total_predict_seconds=sum(t['predict_seconds'] for t in timings),
        total_frontend_seconds=sum(t['frontend_seconds'] for t in timings),
        total_depth_seconds=sum(t['depth_seconds'] for t in timings),historical_reuse=False)
    result=dict(complete=True,identity=identity,model_id=key,model=model,track=track,frames=len(frames),
        predictions=sum(len(p) for p in predictions.values()),failures=sum(i['status']!='ok' for r in records.values() for i in r['items']),
        conditional=conditional_result,geometry=geometry,semantic=semantic,workpoint=workpoint,timing=timing)
    result['speed']=inference_speed(result)
    write_json(output,result)
    if semantic:
        names=list(semantic['summary'])
        write_csv(directory/'metrics'/(key+'_classes.csv'),['class','valid_gt','predictions',*names],
            [[r['class_name'],r['valid_gt'],r['predictions'],*[r['values'][n]['ap'] for n in names]] for r in semantic['per_class']])
        counts={str(i):sum(not b.get('ignore',False) and b['object_bop_id']==i for bs in gt.values() for b in bs) for i in range(1,34)}
        wp=workpoint['per_class'][0]['values'];names=list(wp)
        write_csv(directory/'metrics'/(key+'_object_recall_at_0.7.csv'),['object_bop_id','valid_gt',*names],
            [[i,n,*[wp[name]['tp_by_object'].get(i,0)/n if n else None for name in names]] for i,n in counts.items()])
    return result
