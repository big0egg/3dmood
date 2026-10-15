"""Resume clip extraction into a frozen, audited local pinhole dataset."""

import collections
import hashlib
import io
import json
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import digest,file_digest,read_json,read_jsonl,write_json,write_jsonl
from ..geometry import projected_box,transform_box
from ..image_cache import _write_bytes
from ..visualize import _draw
from .camera import decode_mask,pinhole_maps,pose,remap
from .common import bind,implementation,save_png,validate_box
from .mesh import mesh_bounds


def paths(config):
    root=Path(config['root']).expanduser().resolve()
    return root,root/'data'/config['data']['name'],root/'runs'/config['data']['name']


def registry(config,asset_index):
    objects={str(int(v['instance_bop_id'])):dict(v,uid=k) for k,v in asset_index.items() if v['instance_type']=='object'}
    if set(objects)!=set(config['object_semantics']):raise ValueError('HOT3D 物体ID与语义映射不完整或有多余项')
    labels=list(dict.fromkeys(config['object_semantics'][str(i)] for i in sorted(map(int,objects))))
    classes=[dict(class_id=i,name=name.replace(' ','_'),prompt=name) for i,name in enumerate(labels)]
    for key,value in objects.items():
        value['class_id']=labels.index(config['object_semantics'][key])
        value['prompt']=labels[value['class_id']]
    return classes,objects


def extract_models(config,destination):
    source=Path(config['dataset_root'])/'Assets/assets/Hot3DAssets_assets_assets.zip'
    with zipfile.ZipFile(source) as z:
        index=json.loads(z.read('instance.json'));classes,objects=registry(config,index)
        for key,entry in objects.items():
            blob=z.read(entry['uid']+'.glb')
            entry.update(local_box=mesh_bounds(blob),sha256=hashlib.sha256(blob).hexdigest())
            target=destination/'object_models'/(entry['uid']+'.glb')
            if not target.exists() or file_digest(target)!=entry['sha256']:_write_bytes(target,blob)
    result=dict(classes=classes,objects=objects,source_zip_sha256=file_digest(source))
    write_json(destination/'object_models/registry.json',result)
    return result


def annotations(raw,camera,model_registry,maps,settings,frame_id):
    k,q=maps[3:];twc=pose(camera['T_world_from_camera'])
    virtual=np.eye(4);virtual[:3,:3]=q
    world_to_virtual=virtual@np.linalg.inv(twc)
    boxes=[]
    for key,instances in raw.items():
        for instance_index,item in enumerate(instances):
            bop=str(int(item['object_bop_id']))
            if bop not in model_registry['objects']:raise ValueError('标注含未知物体ID')
            model=model_registry['objects'][bop]
            if item['object_uid']!=model['uid']:raise ValueError('物体 UID/BOP 映射不一致')
            transform=world_to_virtual@pose(item['T_world_from_object'])
            box=transform_box(model['local_box'],transform[:3,:3],transform[:3,3])
            # Remove mesh-only metadata, retaining just canonical box geometry.
            box={name:box[name] for name in ('center','size','rotation')}
            rectangle=projected_box(box,k,settings['camera']['width'],settings['camera']['height'])
            mask_data=item.get('masks_amodal',{}).get('214-1')
            if mask_data:
                if [mask_data['height'],mask_data['width']]!=[camera['calibration']['image_height'],camera['calibration']['image_width']]:
                    raise ValueError('GT amodal mask 与鱼眼图片尺寸不一致')
                mask=remap(decode_mask(mask_data),maps,mask=True)
            else:mask=np.zeros(maps[2].shape,np.uint8)
            ys,xs=np.nonzero(mask);area=len(xs)
            containment=None
            if area>=settings['minimum_amodal_pixels']:
                if rectangle is None:raise ValueError('GT mask 有面积，但三维框投影在视野外')
                x0,y0,x1,y1=rectangle
                containment=float(np.mean((xs>=x0-3)&(xs<=x1+3)&(ys>=y0-3)&(ys<=y1+3)))
                if containment<.95:raise ValueError('GT 三维框与重投影轮廓不对齐，请检查坐标/模型单位')
            native_visibility=float(item.get('visibilities_modeled',{}).get('214-1',0))
            reasons=[]
            if rectangle is None:reasons.append('outside_virtual_view')
            if area<settings['minimum_amodal_pixels']:reasons.append('insufficient_virtual_amodal_pixels')
            if native_visibility<settings['minimum_native_modeled_visibility']:reasons.append('low_native_modeled_visibility')
            box.update(uid=digest([frame_id,model['uid'],instance_index])[:24],object_bop_id=int(bop),object_uid=model['uid'],
                object_name=model['instance_name'],class_id=model['class_id'],projected_bbox=rectangle,
                ignore=bool(reasons),ignore_reasons=reasons,virtual_amodal_pixels=area,
                native_modeled_visibility=native_visibility,
                native_sam2_predicted_visibility=item.get('visibilities_predicted',{}).get('214-1'),
                mask_in_projected_box_fraction=containment)
            validate_box(box,model_registry['classes']);boxes.append(box)
    return boxes


def prepare(config):
    root,destination,run=paths(config);source=Path(config['dataset_root']);settings=config['data']
    if settings['split']!='train_aria' or settings['stream_id']!='214-1':raise ValueError('本协议仅支持 train_aria RGB')
    indices=settings['frame_indices']
    if not indices or len(set(indices))!=len(indices) or any(type(i)!=int or not 0<=i<150 for i in indices):
        raise ValueError('抽帧编号必须是不重复的 0～149 整数')
    split_path=source/'Clips/clip_splits.json';definition_path=source/'Clips/clip_definitions.json'
    splits=read_json(split_path);definitions=read_json(definition_path)
    expected=splits['train']['Aria'];selected=expected if settings['clip_ids'] is None else settings['clip_ids']
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(expected):raise ValueError('片段不属于官方 train Aria 划分')
    selected=sorted(selected)
    models=extract_models(config,destination)
    identity=dict(dataset_root=str(source.resolve()),settings=settings,classes=models['classes'],
        objects_sha256=digest(models),split_sha256=file_digest(split_path),definitions_sha256=file_digest(definition_path),
        geometry_sha256=file_digest(Path(__file__).parents[1]/'geometry.py'),
        preparation_code=implementation(['dataset.py','camera.py','mesh.py','common.py']))
    signature=bind(destination/'preparation.json',identity)
    write_json(root/'config/hot3d_aria.json',config)
    write_json(root/'config/categories.json',dict(classes=models['classes'],objects={k:{n:v[n] for n in
        ('uid','instance_name','class_id','prompt')} for k,v in models['objects'].items()}))
    frames=[];seen=set();gt={};all_checks=[];started=time.perf_counter();reused=0
    for done,clipid in enumerate(selected,1):
        clip=f'clip-{clipid:06d}';archive=source/'Clips/train_aria'/(clip+'.tar')
        stat=archive.stat();source_stat=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        completed=destination/'completed_clips'/(clip+'.json')
        if completed.exists():
            record=read_json(completed)
            if record['preparation_signature']!=signature or record['source_stat']!=source_stat:
                raise ValueError('原始 tar 或准备配置已变化')
            for f in record['frames']:
                for path,checksum in f['prepared_files'].items():
                    if file_digest(destination/path)!=checksum:raise ValueError('准备后的本地数据已损坏')
            if digest(read_json(destination/record['groundtruth_file']))!=record['groundtruth_digest']:
                raise ValueError('准备后的 GT 已变化')
            reused+=1
        else:
            rows=[];groundtruth={};checks=[];last_calibration=None;maps=None
            with tarfile.open(archive,'r:') as tar:
                names={m.name for m in tar.getmembers()}
                if sum(n.endswith('.image_214-1.jpg') for n in names)!=150 or sum(n.endswith('.objects.json') for n in names)!=150:
                    raise ValueError(f'{clip} 缺少完整的150帧RGB/物体GT')
                for number in indices:
                    key=f'{number:06d}'
                    def member(suffix):return tar.extractfile(key+'.'+suffix).read()
                    camera_raw=member('cameras.json');object_raw=member('objects.json');info_raw=member('info.json')
                    camera=json.loads(camera_raw)['214-1'];info=json.loads(info_raw)
                    definition=definitions[str(clipid)]
                    timestamp=info['image_timestamps_ns']['214-1']
                    if (info['sequence_id']!=definition['sequence_id'] or
                            timestamp!=definition['per_frame_timestamps_ns'][number]['214-1']):
                        raise ValueError('帧时间戳/原始序列与官方定义不一致')
                    calibration=camera['calibration'];calibration_key=digest(calibration)
                    if calibration_key!=last_calibration:
                        maps=pinhole_maps(calibration,settings['camera']);last_calibration=calibration_key
                    if float(maps[2].mean())<.95:raise ValueError('虚拟针孔视图有效源像素不足95%，请调整相机参数')
                    jpg=member('image_214-1.jpg')
                    with Image.open(io.BytesIO(jpg)) as im:original=np.asarray(im.convert('RGB')).copy()
                    if list(original.shape[:2])!=[calibration['image_height'],calibration['image_width']]:
                        raise ValueError('RGB与相机标定尺寸不一致')
                    upright=remap(original,maps)
                    image_relative=f'rgb_pinhole/{clip}/{key}.png';raw_relative=f'rgb_fisheye/{clip}/{key}.jpg'
                    calibration_relative=f'calibration/{clip}/{key}.json'
                    _write_bytes(destination/raw_relative,jpg);save_png(destination/image_relative,upright)
                    write_json(destination/calibration_relative,dict(raw=camera,virtual_intrinsics=maps[3].tolist(),
                        virtual_from_raw_rotation=maps[4].tolist(),valid_source_fraction=float(maps[2].mean())))
                    frame_id=info['sequence_id']+':'+str(timestamp)
                    boxes=annotations(json.loads(object_raw),camera,models,maps,settings,frame_id)
                    groundtruth[frame_id]=boxes
                    prepared={name:file_digest(destination/name) for name in (image_relative,raw_relative,calibration_relative)}
                    row=dict(frame_id=frame_id,clip_id=clip,frame_key=key,sequence_id=info['sequence_id'],
                        participant_id=info['participant_id'],timestamp_ns=timestamp,image_hw=list(upright.shape[:2]),
                        rgb_relative=image_relative,rgb_sha256=prepared[image_relative],intrinsics=maps[3].tolist(),
                        prepared_files=prepared,source_selected_content_sha256=digest([hashlib.sha256(b).hexdigest()
                            for b in (jpg,camera_raw,object_raw,info_raw)]))
                    rows.append(row)
                    if number==indices[0]:
                        from PIL import ImageDraw
                        preview=Image.fromarray(upright);draw=ImageDraw.Draw(preview)
                        for b in boxes:
                            _draw(draw,b,maps[3],upright.shape[1],upright.shape[0],
                                  'orange' if b['ignore'] else 'lime',1,b['object_name'])
                        save_png(destination/'geometry_checks'/(clip+'.png'),np.asarray(preview))
                    checks.extend([b['mask_in_projected_box_fraction'] for b in boxes if b['mask_in_projected_box_fraction'] is not None])
            gt_relative=f'groundtruth/{clip}.json';write_json(destination/gt_relative,groundtruth)
            record=dict(preparation_signature=signature,source_stat=source_stat,frames=rows,
                groundtruth_file=gt_relative,groundtruth_digest=digest(groundtruth),containment_checks=checks)
            write_json(completed,record)
        for frame in record['frames']:
            if frame['frame_id'] in seen:raise ValueError('抽帧存在重复原始RGB时间戳')
            seen.add(frame['frame_id']);frames.append(frame)
        gt.update(read_json(destination/record['groundtruth_file']));all_checks.extend(record['containment_checks'])
        write_json(root/'prepare_status.json',dict(stage='data_preparation',clips_completed=done,clips_total=len(selected),
            frames=len(frames),reused_clips=reused,elapsed_seconds=time.perf_counter()-started))
        print(f'HOT3D 准备 {done}/{len(selected)} 片段，{len(frames)} 帧，复用 {reused} 片段',flush=True)
    manifest_hash=digest(frames);gt_hash=digest(gt)
    protocol=dict(schema='hot3d.box3d.v1',benchmark_id=digest([signature,manifest_hash,gt_hash]),
        preparation_signature=signature,settings=settings,classes=models['classes'],frame_count=len(frames),
        frames_sha256=manifest_hash,groundtruth_sha256=gt_hash,source_identity=identity)
    ignored=collections.Counter(reason for boxes in gt.values() for b in boxes for reason in b['ignore_reasons'])
    audit=dict(clips=len(selected),frames=len(frames),source_sequences=len({f['sequence_id'] for f in frames}),
        participants=len({f['participant_id'] for f in frames}),valid_targets=sum(not b['ignore'] for bs in gt.values() for b in bs),
        frames_per_sequence=dict(collections.Counter(f['sequence_id'] for f in frames)),
        frames_per_participant=dict(collections.Counter(f['participant_id'] for f in frames)),
        valid_targets_per_sequence=dict(collections.Counter({seq:sum(not b['ignore'] for f in frames if f['sequence_id']==seq for b in gt[f['frame_id']])
            for seq in sorted({f['sequence_id'] for f in frames})})),
        ignored_targets=sum(b['ignore'] for bs in gt.values() for b in bs),ignore_reasons=dict(ignored),
        empty_valid_gt_frames=sum(not any(not b['ignore'] for b in gt[f['frame_id']]) for f in frames),
        geometry_checks=len(all_checks),minimum_mask_containment=min(all_checks) if all_checks else None,
        visibility_policy='native fisheye modeled visibility + virtual amodal pixels; SAM2 visibility is diagnostic only',
        source_integrity='official split IDs, tar structure, selected frame content hashes; no full tar SHA audit',
        evaluation_split='official training split used as external evaluation; training-overlap not ruled out')
    write_jsonl(destination/'frames.jsonl',frames);write_json(destination/'dataset_audit.json',audit)
    write_json(destination/'protocol.json',protocol)
    return protocol,audit,frames,gt


def load(config):
    _,destination,_=paths(config)
    protocol=read_json(destination/'protocol.json');audit=read_json(destination/'dataset_audit.json')
    frames=read_jsonl(destination/'frames.jsonl');gt={}
    for clip in sorted({f['clip_id'] for f in frames}):gt.update(read_json(destination/'groundtruth'/(clip+'.json')))
    if digest(frames)!=protocol['frames_sha256'] or digest(gt)!=protocol['groundtruth_sha256']:
        raise ValueError('冻结的帧清单/GT已改变')
    if protocol['settings']!=config['data']:raise ValueError('数据配置与冻结协议不同')
    return protocol,audit,frames,gt


def allowed_inputs(config,frames,gt):
    _,destination,_=paths(config);rows={'gt-box':[],'text':[]}
    for f in frames:
        # No world poses, GT class labels, target count, or depth in text input.
        row={key:f[key] for key in ('frame_id','clip_id','frame_key','image_hw','intrinsics','rgb_sha256')}
        row['rgb_path']=str(destination/f['rgb_relative'])
        rows['text'].append(dict(row))
        rows['gt-box'].append(dict(row,prompts=[dict(target_id=b['uid'],box_xyxy=b['projected_bbox'])
            for b in gt[f['frame_id']] if not b['ignore']]))
    return rows
