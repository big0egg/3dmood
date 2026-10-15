"""Fixed-vocabulary GroundingDINO -> SAM2; shared RGB-only MoGe depth."""

import sys
import time
from pathlib import Path

import numpy as np

from ..common import digest,file_digest,read_json,write_json
from ..suite_frontend import category_tokens,grounding_probabilities
from ..suite_io import load_array,mask_checksum,save_array
from .common import bind,commit,implementation,prompt_text,read_record,record_path,rgb,save_png


def work(task):
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4);torch.manual_seed(task['config']['seed'])
    device=torch.device(task['config']['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('前处理不可见PPU，不回退CPU')
    root=Path(task['run_root']);cfg=task['config']['frontend'];track=task['track']
    aux=root/'aux'/track;depth_root=root/'aux/predicted_depth'
    sys.path.insert(0,str(Path(cfg['project_root'])/'infer'))
    from moge_model import load_moge_model,predict_depth_intrinsics
    write_json(root/'worker_status.json',dict(stage='frontend_model_loading',track=track))
    predictor=SAM2ImagePredictor(build_sam2(cfg['sam2_config'],cfg['sam2_checkpoint'],device=str(device)))
    shared=dict(benchmark_id=task['benchmark_id'],input_manifest=task['input_manifest'],device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',torch=torch.__version__,
        code=implementation(['frontend.py','common.py']),numerics_guard=file_digest(Path(__file__).parents[1]/'suite_frontend.py'))
    identity=dict(shared,track=track,config=cfg,sam2_sha256=file_digest(cfg['sam2_checkpoint']),
        sam2_source=digest({str(p.relative_to(Path(sys.modules['sam2'].__file__).parent)):file_digest(p)
            for p in sorted(Path(sys.modules['sam2'].__file__).parent.rglob('*.py'))}))
    processor=grounding=None
    if track=='text':
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        processor=AutoProcessor.from_pretrained(cfg['grounding_model'],local_files_only=True)
        grounding=AutoModelForZeroShotObjectDetection.from_pretrained(cfg['grounding_model'],
            local_files_only=True,disable_custom_kernels=True).to(device).eval()
        text,spans=prompt_text(task['classes']);encoded=processor.tokenizer(text,return_offsets_mapping=True)
        groups=category_tokens(encoded['input_ids'],encoded['offset_mapping'],text,spans)
        identity.update(prompt=text,grounding_disable_custom_kernels=True,
            transformers=__import__('transformers').__version__,grounding_sha256=digest({str(p.relative_to(cfg['grounding_model'])):file_digest(p)
            for p in sorted(Path(cfg['grounding_model']).rglob('*')) if p.is_file() and '.cache' not in p.parts}))
    signature=bind(aux/'model.json',identity)
    # Both tracks read the same RGB dataset; the depth identity has no track or
    # prompt dependency, so measured generation cost is shared, not discarded.
    depth_identity=dict(benchmark_id=task['benchmark_id'],device=str(device),torch=torch.__version__,
        code=shared['code'],moge_sha256=file_digest(cfg['moge_checkpoint']),
        moge_source=file_digest(Path(cfg['project_root'])/'infer/moge_model.py'),
        dataset_rgb_manifest=task['rgb_manifest'],dtype='native MoGe accelerator fp16')
    depth_signature=bind(depth_root/'model.json',depth_identity)
    moge=None;reused=0;started=time.perf_counter()
    for done,frame in enumerate(task['frames'],1):
        write_json(root/'worker_status.json',dict(stage='frontend',track=track,frames_completed=done-1,
            expected_frames=len(task['frames']),frame_id=frame['frame_id'],reused=reused))
        path=record_path(aux/'frames',frame)
        depth_path=record_path(depth_root/'frames',frame)
        if path.exists():
            old=read_record(path,frame,signature)
            for item in old['items']:
                if item['status']=='ok':read_mask(aux,item,frame)
            depth_record=read_record(depth_path,frame,depth_signature);load_array(depth_root,depth_record['array'])
            reused+=1
        else:
            image=rgb(frame)
            if depth_path.exists():
                depth_record=read_record(depth_path,frame,depth_signature);load_array(depth_root,depth_record['array'])
                depth_seconds=depth_record['seconds']
            else:
                if moge is None:moge=load_moge_model(cfg['moge_checkpoint'],device)
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();depth,predicted_k=predict_depth_intrinsics(moge,image)
                if device.type=='cuda':torch.cuda.synchronize()
                depth_seconds=time.perf_counter()-start
                depth=np.where(np.isfinite(depth)&(depth>0),depth,0).astype(np.float32)
                if list(depth.shape)!=frame['image_hw']:raise ValueError('MoGe深度尺寸不一致')
                if not np.any(depth>0):raise RuntimeError('MoGe没有输出任何有效深度，不能将整帧误记为目标失败')
                array=record_path(depth_root/'arrays',frame,'.npy');save_array(array,depth)
                commit(depth_path,frame,depth_signature,seconds=depth_seconds,predicted_intrinsics=predicted_k.tolist(),
                    array=dict(path=str(array.relative_to(depth_root)),sha256=file_digest(array),shape=list(depth.shape)))
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            if track=='gt-box':proposals=[dict(p) for p in frame['prompts']]
            else:
                inputs=processor(images=Image.fromarray(image),text=text,return_tensors='pt').to(device)
                if inputs['input_ids'][0].tolist()!=encoded['input_ids']:raise ValueError('固定类别提示被意外截断/改变')
                with torch.inference_mode():output=grounding(**inputs)
                probabilities=grounding_probabilities(output,inputs['attention_mask'])[0]
                per_class=torch.stack([probabilities[:,ids].max(-1).values for ids in groups],dim=-1)
                class_scores,class_ids=per_class.max(-1);native_scores=probabilities.max(-1).values
                selected=torch.where((native_scores>cfg['box_threshold'])&(class_scores>cfg['text_threshold']))[0]
                selected=selected[native_scores[selected].argsort(descending=True)[:cfg['max_detections']]]
                coordinates=output.pred_boxes[0,selected].detach().cpu().numpy()
                height,width=frame['image_hw'];proposals=[]
                for query,(cx,cy,w,h) in zip(selected.tolist(),coordinates):
                    box=[max(0.,(cx-w/2)*width),max(0.,(cy-h/2)*height),min(width-1.,(cx+w/2)*width),min(height-1.,(cy+h/2)*height)]
                    proposals.append(dict(target_id=f'q{query}',box_xyxy=list(map(float,box)),
                        class_id=int(class_ids[query]),detection_score=float(native_scores[query])))
            if device.type=='cuda':torch.cuda.synchronize()
            detection_seconds=time.perf_counter()-start
            predictor.set_image(image);items=[];write_seconds=0.
            for proposal in proposals:
                box=proposal['box_xyxy']
                if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
                    items.append(dict(proposal,status='failed',failure='invalid_2d_box'));continue
                masks,scores,_=predictor.predict(box=np.asarray(box,dtype=np.float32)[None],multimask_output=False)
                mask=np.asarray(masks).reshape(-1,*image.shape[:2])[0].astype(bool)
                if not mask.any():items.append(dict(proposal,status='failed',failure='empty_predicted_mask'));continue
                mask_path=record_path(aux/'masks',frame,'_'+digest(proposal['target_id'])[:16]+'.png')
                t=time.perf_counter();save_png(mask_path,mask.astype(np.uint8)*255);write_seconds+=time.perf_counter()-t
                items.append(dict(proposal,status='ok',mask_path=str(mask_path.relative_to(aux)),
                    mask_sha256=mask_checksum(mask),mask_score=float(np.asarray(scores).reshape(-1)[0])))
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-start-write_seconds
            commit(path,frame,signature,items=items,timing=dict(frontend_seconds=elapsed,
                detection_seconds=detection_seconds,depth_seconds=depth_seconds))
        if done%5==0 or done==len(task['frames']):print(f'HOT3D 前处理 {track} {done}/{len(task["frames"])} 帧，复用 {reused}',flush=True)
    result=dict(complete=True,frames=len(task['frames']),reused=reused,wall_seconds=time.perf_counter()-started)
    write_json(root/track/'full/frontend.json',result)
    return result


def read_mask(aux,item,frame):
    from PIL import Image
    relative=Path(item['mask_path']);path=(Path(aux)/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(Path(aux).resolve()):raise ValueError('mask路径越界')
    with Image.open(path) as im:mask=np.asarray(im)>0
    if list(mask.shape)!=frame['image_hw'] or mask_checksum(mask)!=item['mask_sha256']:raise ValueError('mask缓存已变化')
    return mask
