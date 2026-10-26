"""Dataset-format adapters read artifacts and preserve camera/GT conventions."""
import copy
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from ioutest.common import digest,file_digest,write_json,write_jsonl
from ioutest.experiment_data import load_dataset,input_frames,read_rgb
from .test_experiment import box,prepared


@pytest.mark.parametrize('realistic',[False,True])
def test_simulation_is_read_only_and_filters_children(tmp_path,realistic):
    root=tmp_path/'source';folder=root/'eval/SCENE/state/agentview';folder.mkdir(parents=True)
    rgb=folder/'rgb.png';Image.fromarray(np.full((40,40,3),80,np.uint8)).save(rgb)
    target=dict(instance_id=1,kind='object',parent_instance_id=None,category='cup',center_camera=[0,0,2],size_xyz=[.2,.4,.3],
        R_camera_from_box=np.eye(3).tolist(),bbox_projected_clipped_xyxy=[18,16,22,24],visible=True,visible_pixels=50)
    child=dict(target,instance_id=2,parent_instance_id=1)
    robot=dict(target,instance_id=3,kind='robot',category='robot')
    annotation=dict(length_unit='meter',camera_coordinates='opencv',camera='agentview',image_hw=[40,40],
        intrinsics=[[40,0,20],[0,40,20],[0,0,1]],provenance=dict(state_id='state',scene_id='SCENE',split='eval'),
        objects=[target,child,robot],gt_status='inherited_simulation_gt_requires_visual_review')
    write_json(folder/'annotation.json',annotation)
    write_json(root/'categories.json',dict(categories=[dict(name='cup',kinds=['object']),dict(name='robot',kinds=['robot'])]))
    row=dict(state_id='state',scene_id='SCENE',split='eval')
    if realistic:
        row.update(rgb_path=str(rgb),annotation_path=str(folder/'annotation.json'),camera='agentview')
        write_jsonl(root/'eval.jsonl',[row]);manifest='eval.jsonl'
    else:write_json(root/'manifest.json',dict(accepted={'state':row}));manifest='manifest.json'
    spec=dict(root=str(root),format='robot_realism_prepared' if realistic else 'robot_synth_prepared',
        manifest=str(root/manifest),categories_file=str(root/'categories.json'),eval_split='eval',camera='agentview')
    before={str(p):file_digest(p) for p in root.rglob('*') if p.is_file()}
    data=load_dataset(spec)
    assert len(data['frames'])==1 and len(data['gt']['state:agentview'])==1
    assert data['metadata']['label_status']==['inherited_simulation_gt_requires_visual_review']
    assert before=={str(p):file_digest(p) for p in root.rglob('*') if p.is_file()}
    assert input_frames(data,'text')[0]['frame_id']=='state:agentview'


def test_prepared_arkit_reads_only_local_cache(tmp_path):
    root=tmp_path/'arkit';root.mkdir();cache=tmp_path/'cache';cache.mkdir()
    rgb=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3);Image.fromarray(rgb).save(cache/'rgb.png')
    import hashlib
    pixels=hashlib.sha256(rgb.tobytes()).hexdigest()
    row=dict(frame_id='frame',video_id='scan',visit_id='visit',image_hw=[6,4],raw_image_hw=[4,6],rotation_k=1,
        intrinsics=[[4,0,2],[0,4,3],[0,0,1]],rgb_pixels_sha256=pixels)
    write_jsonl(root/'frames.jsonl',[row]);p=root/'groundtruth/scan.json'
    write_json(p,dict(frames=[dict(frame=row,boxes=[box(cid=13)])]))
    audit=dict(scans=[dict(video_id='scan',groundtruth_file_sha256=file_digest(p))]);write_json(root/'dataset_audit.json',audit)
    write_json(root/'protocol.json',dict(complete=True,benchmark_id='arkit',frames_sha256=file_digest(root/'frames.jsonl'),
        audit_sha256=file_digest(root/'dataset_audit.json')))
    write_jsonl(cache/'images.jsonl',[dict(frame_id='frame',raw_image='rgb.png',raw_rgb_pixels_sha256=pixels)])
    write_json(cache/'cache_summary.json',dict(complete=True,benchmark_id='arkit',images_manifest_sha256=file_digest(cache/'images.jsonl')))
    spec=dict(format='arkit_prepared',root=str(root),manifest=str(root/'frames.jsonl'),protocol=str(root/'protocol.json'),image_cache=str(cache))
    data=load_dataset(spec);np.testing.assert_array_equal(read_rgb(data['frames'][0]),np.rot90(rgb))
    assert len(data['classes'])==17
    (cache/'rgb.png').unlink()
    with pytest.raises(FileNotFoundError,match='不会制作'):load_dataset(spec)


def test_sensor_depth_metadata_is_required_and_bound(prepared,tmp_path):
    _,c=prepared;spec=c['datasets']['test'];data=load_dataset(spec)
    manifest=tmp_path/'depths.jsonl';depth=tmp_path/'depth.npy';np.save(depth,np.ones((40,40),np.float32))
    rows=[dict(frame_id=f['frame_id'],depth_path=str(depth),depth_kind='sensor',depth_scale=1) for f in data['frames']]
    write_jsonl(manifest,rows);spec['depth_manifest']=str(manifest)
    data=load_dataset(spec);inputs=input_frames(data,'text',True)
    assert inputs[0]['depth_sha256']==file_digest(depth)
    rows[0]['depth_kind']='simulation';write_jsonl(manifest,rows)
    with pytest.raises(ValueError,match='仿真 GT'):input_frames(load_dataset(spec),'text',True)


def test_hot3d_accepts_content_digests_from_existing_finetune(prepared):
    _,c=prepared;spec=c['datasets']['test'];root=Path(spec['root'])
    from ioutest.common import read_json,read_jsonl
    rows=read_jsonl(spec['manifest'])
    for row in rows:row['gt_sha256']=digest(read_json(row['gt_path']))
    write_jsonl(spec['manifest'],rows)
    protocol=read_json(spec['protocol']);protocol['eval_sha256']=digest(rows);write_json(spec['protocol'],protocol)
    assert len(load_dataset(spec)['frames'])==2
