"""Dataset-configurable AP and strict input-target success; no model imports."""

import collections

import numpy as np

from ..geometry import box_iou
from ..metrics import interpolated_ap
from .common import validate_box


def thresholds(config):
    out=[('iou',float(t)) for t in config['metrics']['iou_thresholds']]
    out += [('center',float(t)) for t in config['metrics']['center_thresholds']]
    if not out or len(set(out))!=len(out) or any(not np.isfinite(t) or t<=0 or (k=='iou' and t>1) for k,t in out):
        raise ValueError('无效或重复的评测阈值')
    return out


def label(kind,value,ap=True):
    return ('AP3D' if ap else 'success3D')+f'@{value:.2f}' if kind=='iou' else (
        'APcenter' if ap else 'successCenter')+f'@{value:.2f}m'


def evaluate(gt,predictions,classes,limits,semantic=False,score_floor=None,pairs=None):
    if set(gt)!=set(predictions):raise ValueError('评测帧缺失或多余；不能将缺失预测当作无检出')
    pairs={} if pairs is None else pairs
    for frame,boxes in gt.items():
        for box in boxes:validate_box(box,classes)
        for box in predictions[frame]:
            validate_box(box,classes,True)
            if semantic and box['class_id'] is None:raise ValueError('语义AP需要预测类别')
    groups=list(enumerate(classes)) if semantic else [(None,dict(name='object'))]
    entries=[]
    for cid,category in groups:
        valid_count=sum(not b.get('ignore',False) and (cid is None or b['class_id']==cid) for bs in gt.values() for b in bs)
        ranked=[(float(b['score']),f,i,b) for f,bs in predictions.items() for i,b in enumerate(bs)
            if (cid is None or b['class_id']==cid) and (score_floor is None or b['score']>=score_floor)]
        ranked.sort(key=lambda r:(-r[0],r[1],r[2]))
        for _,frame,index,p in ranked:
            cache_key=(frame,index)
            if cache_key not in pairs:
                bs=gt[frame]
                pairs[cache_key]=(np.array([box_iou(p,b) for b in bs]),
                    np.array([np.linalg.norm(np.asarray(p['center'])-b['center']) for b in bs]))
        values={}
        for kind,limit in limits:
            used=collections.defaultdict(set);ignored_used=collections.defaultdict(set);tp=[];fp=[];ignored=0
            tp_by_object=collections.Counter()
            for _,frame,index,p in ranked:
                scores=pairs[(frame,index)][kind=='center'];boxes=gt[frame]
                candidates=[i for i,b in enumerate(boxes) if (cid is None or b['class_id']==cid) and
                    not b.get('ignore',False) and i not in used[frame] and (scores[i]>=limit if kind=='iou' else scores[i]<=limit)]
                if candidates:
                    best=(max if kind=='iou' else min)(candidates,key=lambda i:scores[i])
                    used[frame].add(best);tp.append(1);fp.append(0)
                    tp_by_object[str(boxes[best].get('object_bop_id','unknown'))]+=1
                    continue
                candidates=[i for i,b in enumerate(boxes) if (cid is None or b['class_id']==cid) and
                    b.get('ignore',False) and i not in ignored_used[frame] and (scores[i]>=limit if kind=='iou' else scores[i]<=limit)]
                if candidates:
                    best=(max if kind=='iou' else min)(candidates,key=lambda i:scores[i]);ignored_used[frame].add(best);ignored+=1
                else:tp.append(0);fp.append(1)
            true,false=sum(tp),sum(fp);precision=true/(true+false) if true+false else 0.
            recall=true/valid_count if valid_count else None
            values[label(kind,limit)]=dict(ap=interpolated_ap(tp,fp,valid_count),tp=true,fp=false,
                ignored_predictions=ignored,precision=precision,recall=recall,
                f1=2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.,
                tp_by_object=dict(tp_by_object))
        entries.append(dict(class_id=cid,class_name=category['name'],valid_gt=valid_count,predictions=len(ranked),values=values))
    summary={}
    for kind,limit in limits:
        name=label(kind,limit);vs=[r['values'][name]['ap'] for r in entries if r['valid_gt']]
        summary[name]=float(np.mean(vs)) if vs else None
    return dict(summary=summary,per_class=entries,semantic=semantic,score_floor=score_floor,
        interpolation='101 recall points; precision envelope',matching='confidence descending; same frame; one-to-one; valid GT before ignore')


def conditional(gt,records,limits):
    details=[];reasons=collections.Counter()
    for frame,boxes in gt.items():
        expected={b['uid']:b for b in boxes if not b.get('ignore',False)}
        items=records[frame]['items'];actual={i['target_id']:i for i in items}
        if len(actual)!=len(items) or set(actual)!=set(expected):raise ValueError('逐目标预测缺失/重复或关联错误')
        for uid,b in expected.items():
            item=actual[uid];p=item.get('detection');failed=item['status']!='ok'
            if failed!=(p is None):raise ValueError('失败状态与三维框不一致')
            overlap=0. if failed else box_iou(p,b)
            distance=None if failed else float(np.linalg.norm(np.asarray(p['center'])-b['center']))
            hits={label(k,t,False):not failed and (overlap>=t if k=='iou' else distance<=t) for k,t in limits}
            details.append(dict(frame_id=frame,target_id=uid,object_bop_id=b['object_bop_id'],class_id=b['class_id'],
                failed=failed,failure=item.get('failure'),iou=overlap,center_distance_m=distance,hits=hits))
            if failed:reasons[item.get('failure','unspecified')]+=1
    count=len(details)
    def aggregate(rows):
        return dict(targets=len(rows),failures=sum(r['failed'] for r in rows),
            success={label(k,t,False):sum(r['hits'][label(k,t,False)] for r in rows)/len(rows) if rows else None for k,t in limits})
    result=aggregate(details)
    result.update(failure_rate=result['failures']/count if count else None,failure_reasons=dict(reasons),
        mean_iou_including_failures=float(np.mean([r['iou'] for r in details])) if details else None,
        median_center_error_m=float(np.median([r['center_distance_m'] for r in details if not r['failed']])) if any(not r['failed'] for r in details) else None,
        per_object={str(i):aggregate([r for r in details if r['object_bop_id']==i]) for i in sorted({r['object_bop_id'] for r in details})})
    return result,details
