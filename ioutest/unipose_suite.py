"""Six configurations, two explicitly different benchmark tasks."""

import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .benchmark import child_environment, select_frames
from .common import digest, file_digest, read_json, read_jsonl, utc_now, write_json
from .dataset import load_benchmark
from .depth_cache import DepthInputReader
from .image_cache import FrameInputReader
from .suite_io import GT_MODELS, TEXT_MODELS, project_source, safe_child


DEFAULT_CONFIG = Path(__file__).with_name('unipose_suite_config.json')


def prepare_inputs(config):
    protocol, audit, frames, gt = load_benchmark(config['run_dir'])
    images = FrameInputReader(config['run_dir'], protocol, frames, config['image_cache'])
    depths = DepthInputReader(config['image_cache'], protocol, frames)
    rows = {'gt-box': [], 'text': []}
    for frame in frames:
        image = images.records[frame['frame_id']]
        depth = depths.records[frame['frame_id']]
        row = {k: frame[k] for k in ('frame_id','video_id','visit_id','rgb_entry','image_hw',
                                    'raw_image_hw','rotation_k','intrinsics','sky_direction')}
        row.update(rgb_path=str(safe_child(images.cache_root, image['image'])),
                   upright_rgb_sha256=image['rgb_pixels_sha256'])
        # The text manifest never contains GT locations, counts, categories,
        # measured depth paths or pose matrices.
        rows['text'].append(copy.deepcopy(row))
        row.update(sensor_depth_path=depth['depth_image'], sensor_depth_sha256=depth['png_sha256'],
                   prompts=[dict(target_id=b['uid'], box_xyxy=b['projected_bbox'])
                            for b in gt[frame['frame_id']] if not b.get('ignore', False)])
        for prompt in row['prompts']:
            if prompt['box_xyxy'] is None:
                raise ValueError('有效 GT 缺少投影框')
        rows['gt-box'].append(row)
    metadata = dict(benchmark_id=protocol['benchmark_id'], rgb_manifest_sha256=images.source_info['manifest_sha256'],
                    sensor_depth_manifest_sha256=depths.source_info['manifest_sha256'])
    return protocol, audit, frames, gt, rows, metadata


def smoke_frames(frames, gt):
    chosen = {}
    for direction in sorted({f['sky_direction'] for f in frames}):
        group = [f for f in frames if f['sky_direction'] == direction]
        for has_gt in (True, False):
            options = [f for f in group if any(not b.get('ignore', False) for b in gt[f['frame_id']]) == has_gt]
            for f in options[:2]: chosen[f['frame_id']] = f
    for f in frames:
        if len(chosen) >= 15: break
        chosen.setdefault(f['frame_id'], f)
    return [f for f in frames if f['frame_id'] in chosen]


def historical_identity(config, model_id, protocol, input_info):
    old_root = Path(config['historical_report'])
    path = old_root/'models'/model_id/'model.json'
    if not path.is_file(): return None
    old = read_json(path)
    p = old['provenance']
    model = config['models'][model_id]
    options = model['options']
    old_options = dict(p['options']); new_options = dict(options, device=config['device'])
    for key in ('checkpoint','lingbot_checkpoint'):
        old_options.pop(key, None); new_options.pop(key, None)
    if (p['benchmark_id'] != protocol['benchmark_id'] or old_options != new_options or
            p['input_source']['manifest_sha256'] != input_info['rgb_manifest_sha256']): return None
    folder = 'opendet3d' if model_id == '3dmood' else 'wilddet3d'
    if project_source(options['project_root'], folder) != p['model']['project_source_sha256']: return None
    if model_id == '3dmood' and file_digest(Path(options['project_root'])/'scripts/demo.py') != p['model']['demo_sha256']:
        return None
    adapter = Path(__file__).parent/'adapters'/('mood3d.py' if model_id=='3dmood' else 'wilddet.py')
    if file_digest(adapter) != p['adapter']['adapter_file_sha256']: return None
    if file_digest(options['checkpoint']) != p['model']['checkpoint_sha256']: return None
    if model_id == 'wilddet3d' and file_digest(options['lingbot_checkpoint']) != p['model']['lingbot_sha256']: return None
    # Verify the exact inference package versions in its own environment.
    script = "import json,importlib.metadata as m; names=json.loads(__import__('sys').argv[1]); print(json.dumps({n:m.version(n) if v is not None else None for n,v in names.items()}))"
    versions = subprocess.run([model['python'],'-B','-c',script,json.dumps(p['versions'])],
                              capture_output=True,text=True,check=True)
    if json.loads(versions.stdout.strip().splitlines()[-1]) != p['versions']: return None
    return dict(**old, reuse=dict(model_root=str(path.parent), report_root=str(old_root),
                                 signature=old['signature'], verified_utc=utc_now()))


def readiness(config, offline=False):
    errors, environments = [], {}
    front = config['frontend']
    for path in (front['sam2_checkpoint'], str(Path(front['grounding_model'])/'config.json'),
                 front['moge_checkpoint'], *[m['options']['checkpoint'] for m in config['models'].values()]):
        if not Path(path).is_file(): errors.append('缺少权重/配置：'+path)
    for label, python in [('frontend',front['python']),('pose',config['models']['unipose9d-box-rgb']['python']),
                          ('wilddet3d',config['models']['wilddet3d']['python']),('3dmood',config['models']['3dmood']['python'])]:
        if not Path(python).is_file():
            errors.append('缺少解释器：'+python); continue
        script = "import json,torch,importlib.util as u; ok=torch.cuda.is_available(); print(json.dumps({'torch':torch.__version__,'cuda':ok,'device_name':torch.cuda.get_device_name(0) if ok else None,'sam2':u.find_spec('sam2') is not None,'transformers':u.find_spec('transformers') is not None}))"
        result = subprocess.run([python,'-B','-c',script],capture_output=True,text=True)
        if result.returncode:
            errors.append(label+' 环境导入失败：'+result.stderr[-400:]);continue
        info = json.loads(result.stdout.strip().splitlines()[-1]); environments[label]=info
        if not offline and config['device'].startswith('cuda') and not info['cuda']:
            errors.append(label+'：当前进程不可见 PPU，需 DSW 原生 Terminal')
        if label=='frontend' and not (info['sam2'] and info['transformers']):
            errors.append('前处理环境缺少 SAM2/Transformers')
    return dict(ready=not errors, errors=errors, environments=environments)


def run_child(module, task, python, env, path):
    write_json(path, task)
    log = path.with_suffix('.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"启动 {module}；日志 {log}",flush=True)
    with log.open('a') as stream:
        subprocess.run([python,'-B','-m',module,str(path)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def run_suite(config_path=DEFAULT_CONFIG, report_root='~/report/arkitscenes-unipose9d-v2',
              phase='all', tracks=('gt-box','text'), prepare=False, baselines_only=False):
    config = read_json(config_path)
    if prepare:
        from .suite_prepare import prepare as prepare_frontend
        prepare_frontend(config)
    protocol, audit, frames, gt, inputs, input_info = prepare_inputs(config)
    root = Path(report_root).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.suite.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        settings = dict(config=config, benchmark_id=protocol['benchmark_id'], inputs=input_info)
        if (root/'experiment.json').exists() and read_json(root/'experiment.json')['settings'] != settings:
            raise ValueError('实验配置已变化，请使用新的 report-root')
        write_json(root/'experiment.json',dict(settings=settings,created_utc=utc_now()))
        for track in tracks:
            write_json(root/track/'input_manifest.json', inputs[track])
        phases = ['smoke','pilot','full'] if phase=='all' else [phase]
        env = child_environment({})
        # Downloading is a separate preparation step; all inference runs offline.
        env.update(HF_HOME='/root/.cache/huggingface', XFORMERS_DISABLED='1')
        historical = {}
        if 'text' in tracks:
            for key in ('3dmood','wilddet3d'):
                history = historical_identity(config,key,protocol,input_info)
                if history:
                    historical[key]=history
                    destination=root/'text/models'/key/'model.json'
                    if destination.exists():
                        prior=read_json(destination)
                        history['reuse']['verified_utc']=prior.get('reuse',{}).get('verified_utc',history['reuse']['verified_utc'])
                        if prior!=history:raise ValueError('历史预测来源已变化')
                    write_json(destination,history)
        if not baselines_only:
            check=readiness(config)
            write_json(root/'readiness.json',check)
            if not check['ready']:
                write_json(root/'status.json',dict(status='not_started',errors=check['errors'],updated_utc=utc_now()))
                raise RuntimeError('；'.join(check['errors']))
        try:
            for p in phases:
                selected = smoke_frames(frames,gt) if p=='smoke' else select_frames(frames,p)
                chosen = {f['frame_id'] for f in selected}
                for track in tracks:
                    keys = GT_MODELS if track=='gt-box' else TEXT_MODELS
                    if baselines_only:
                        if track!='text':continue
                        keys=tuple(k for k in ('3dmood','wilddet3d') if k in historical)
                    directory=root/track/p
                    selection=dict(benchmark_id=protocol['benchmark_id'],track=track,phase=p,
                                   frame_ids=[f['frame_id'] for f in selected])
                    selection['selection_id']=digest(selection)
                    write_json(directory/'selection.json',selection)
                    task=dict(config=config,report_root=str(root),track=track,phase=p,device=config['device'],
                              frames=[r for r in inputs[track] if r['frame_id'] in chosen],
                              input_manifest_sha256=digest(inputs[track]),**input_info)
                    if not baselines_only:
                        write_json(root/'status.json',dict(status='running',phase=p,track=track,stage='frontend',updated_utc=utc_now()))
                        run_child('ioutest.suite_worker',dict(task,job='frontend'),config['frontend']['python'],env,directory/'logs/frontend_task.json')
                    for key in keys:
                        write_json(root/'status.json',dict(status='running',phase=p,track=track,model=key,
                            stage='historical_score' if key in historical else 'inference_and_score',updated_utc=utc_now()))
                        if key in ('3dmood','wilddet3d') and key not in historical:
                            legacy=dict(run_dir=config['run_dir'],report_root=str(root/'text'),phase=p,model_id=key,
                                        model=config['models'][key],device=config['device'],image_cache=config['image_cache'],selection=selection)
                            run_child('ioutest.benchmark_worker',legacy,config['models'][key]['python'],env,directory/'logs'/(key+'_task.json'))
                        elif key not in historical:
                            run_child('ioutest.suite_worker',dict(task,model_id=key),config['models'][key]['python'],env,directory/'logs'/(key+'_task.json'))
                        from .suite_score import score_model
                        if key in historical:
                            precomputed = Path(__file__).parent/'runs/arkitscenes-unipose9d-v2/text'/p/'metrics'
                            destination = directory/'metrics'
                            if precomputed.resolve() != destination.resolve() and (precomputed/(key+'.json')).is_file():
                                destination.mkdir(parents=True,exist_ok=True)
                                for saved in precomputed.glob(key+'*'):
                                    if saved.is_file() and not (destination/saved.name).exists():
                                        shutil.copyfile(saved,destination/saved.name)
                                # score_model still verifies the actual frame
                                # records and metric identity before accepting it.
                        score_model(config,root,track,p,key,selected,gt,protocol,audit)
                    if not baselines_only:
                        from .suite_report import assemble
                        assemble(config,root,track,p,selected,gt,protocol,audit)
                    else:
                        write_json(directory/'baseline_reuse.json',dict(complete=True,models=list(keys),frames=len(selected),
                                                                      pending=['unipose9d-text-rgb']))
            write_json(root/'status.json',dict(status='baselines_ready' if baselines_only else 'complete',
                                               phases=phases,tracks=list(tracks),updated_utc=utc_now()))
        except BaseException as exc:
            write_json(root/'status.json',dict(status='failed',error=str(exc),updated_utc=utc_now()))
            raise
    return dict(report_root=str(root),complete=not baselines_only,baseline_prepared=baselines_only)


def main(argv=None):
    parser=argparse.ArgumentParser(description='UniPose9D 六配置、框提示与文本两类 benchmark')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--report-root',default='~/report/arkitscenes-unipose9d-v2')
    parser.add_argument('--phase',choices=['all','smoke','pilot','full'],default='all')
    parser.add_argument('--tracks',nargs='+',choices=['gt-box','text'],default=['gt-box','text'])
    parser.add_argument('--prepare',action='store_true',help='先准备独立前处理依赖和权重，再运行')
    parser.add_argument('--check',action='store_true',help='只检查，不运行模型')
    parser.add_argument('--reuse-baselines-only',action='store_true',help='核验并评分可复用的历史文本组，暂不运行 PPU')
    args=parser.parse_args(argv)
    if args.check:
        result=readiness(read_json(args.config))
    else:
        result=run_suite(args.config,args.report_root,args.phase,args.tracks,args.prepare,args.reuse_baselines_only)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if args.check and not result['ready'] else 0


if __name__=='__main__':raise SystemExit(main())
