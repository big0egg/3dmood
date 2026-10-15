"""Recover deleted helper assets on the large destination disk, before inference."""

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..common import file_digest,read_json,write_json

EXPECTED={
 'pose':'02c91a3df7d37740e552c421c9bd697189155c7e3936fdb1e7798a5e16dd3e73',
 'sam':'2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318',
 'moge':'280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01',
 'wild':'f8b6a9e548f733ba62625a0d2adc4b0f4fdb6007ee11d9927f9c1027010fee57',
 'lingbot':'b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23',
 'grounding':'1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'}


def real_weight(path):
    p=Path(path)
    return p.is_file() and p.stat().st_size>1024*1024


def localize(config):
    """Keep explicit existing paths; redirect only missing defaults to root/assets."""
    c=copy.deepcopy(config);assets=Path(c['root']).expanduser().resolve()/'assets'
    for key,model in c['models'].items():
        o=model['options']
        if key.startswith('unipose9d') and not real_weight(o['checkpoint']):o['checkpoint']=str(assets/'unipose9d/last.ckpt')
        if key.startswith('wilddet3d'):
            for field,name in [('checkpoint','wilddet3d_alldata_all_prompt_v1.0.pt'),('lingbot_checkpoint','model.pt')]:
                if not real_weight(o[field]):o[field]=str(assets/'wilddet3d'/name)
    f=c['frontend']
    for field,name in [('sam2_checkpoint','sam2/sam2.1_hiera_large.pt'),('moge_checkpoint','moge2/model.pt')]:
        if not real_weight(f[field]):f[field]=str(assets/'unipose9d'/name)
    g=Path(f['grounding_model'])
    if not (g/'config.json').exists() or not real_weight(g/'model.safetensors'):
        f['grounding_model']=str(assets/'unipose9d/grounding-dino-tiny')
    return c


def verify(path,key):
    actual=file_digest(path)
    if actual!=EXPECTED[key]:raise ValueError(f'{key} 权重SHA256不符；不会继续推理：{path}')
    return actual


def copy_weight(source,target,key):
    target=Path(target);source=Path(source)
    if real_weight(target):return verify(target,key)
    if not real_weight(source):raise FileNotFoundError(f'找不到可恢复的权重：{source}')
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.part')
    print(f'恢复本地权重：{source} -> {target}',flush=True)
    shutil.copyfile(source,temporary);checksum=verify(temporary,key);os.replace(temporary,target)
    return checksum


def download_helpers(config):
    """Runs in the existing frontend env, with online Hub explicitly enabled."""
    from huggingface_hub import hf_hub_download,snapshot_download
    front=config['frontend']
    targets=[('sam2_checkpoint','sam','facebook/sam2.1-hiera-large','sam2.1_hiera_large.pt','665f8e2ad61cf5f53d65644ff27c8ee525124610'),
             ('moge_checkpoint','moge','Ruicheng/moge-2-vitl-normal','model.pt','main')]
    for field,key,repo,filename,revision in targets:
        target=Path(front[field])
        if not real_weight(target):
            target.parent.mkdir(parents=True,exist_ok=True)
            print(f'下载官方辅助权重：{repo}/{filename}',flush=True)
            downloaded=Path(hf_hub_download(repo,filename,revision=revision,local_dir=str(target.parent)))
            if downloaded!=target:raise ValueError('Hub返回了非预期路径')
        verify(target,key)
    target=Path(front['grounding_model'])
    if not (target/'config.json').is_file() or not real_weight(target/'model.safetensors'):
        snapshot_download('IDEA-Research/grounding-dino-tiny',revision='a2bb814dd30d776dcf7e30523b00659f4f141c71',
            local_dir=str(target),allow_patterns=['*.json','*.txt','model.safetensors'])
    verify(target/'model.safetensors','grounding')


def prepare(config):
    c=localize(config);root=Path(c['root']).expanduser().resolve();assets=root/'assets'
    for key in ('wilddet3d-box','wilddet3d'):
        options=c['models'][key]['options']
        for field,name,kind in [('checkpoint','wilddet3d_alldata_all_prompt_v1.0.pt','wild'),('lingbot_checkpoint','model.pt','lingbot')]:
            if not real_weight(options[field]):copy_weight(Path('/mnt/lvyifan')/name,options[field],kind)
    pose=Path(c['models']['unipose9d-box-rgb']['options']['checkpoint'])
    if not real_weight(pose):
        pose.parent.mkdir(parents=True,exist_ok=True);partial=pose.with_name(pose.name+'.part')
        url='https://media.githubusercontent.com/media/qq456cvb/UniPose9D/a2b95c338b26be82acfdb188e9f0a99acc9c6db8/checkpoints/last.ckpt'
        print('下载官方UniPose9D权重（Git LFS内容）',flush=True)
        subprocess.run(['curl','-fL','--retry','3','-C','-','--output',str(partial),url],check=True)
        verify(partial,'pose');os.replace(partial,pose)
    # Installed model packages remain untouched; downloads are a separate process
    # so any caller's offline HuggingFace globals cannot accidentally block them.
    task=assets/'restore_config.json';write_json(task,c)
    env=dict(os.environ,HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0')
    subprocess.run([c['frontend']['python'],'-B','-m','ioutest.hot3d.assets',str(task)],env=env,check=True)
    inventory={}
    for key,model in c['models'].items():
        for name in ('checkpoint','lingbot_checkpoint'):
            p=model['options'].get(name)
            if p:inventory[p]=file_digest(p)
    for field in ('sam2_checkpoint','moge_checkpoint'):
        p=c['frontend'][field];inventory[p]=file_digest(p)
    p=Path(c['frontend']['grounding_model'])/'model.safetensors';inventory[str(p)]=file_digest(p)
    write_json(assets/'manifest.json',dict(sha256=inventory,source='local WildDet3D recovery + official GitHub/Hugging Face; auxiliary weights verified against previous run'))
    return c


if __name__=='__main__':download_helpers(read_json(sys.argv[1]))
