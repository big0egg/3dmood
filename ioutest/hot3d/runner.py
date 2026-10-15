"""HOT3D preparation/full/resume entry point, using each native model env."""

import argparse
import fcntl
import json
import shutil
import sys
from pathlib import Path

from ..benchmark import child_environment
from ..common import atomic_text,digest,read_json,utc_now,write_json
from ..unipose_suite import readiness,run_child
from .common import TRACKS,bind
from .dataset import allowed_inputs,load,paths,prepare
from .assets import localize

DEFAULT_CONFIG=Path(__file__).parents[1]/'hot3d_config.json'


def run(config,tracks=('gt-box','text'),prepare_only=False,score_only=False):
    root,data,destination=paths(config)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.hot3d.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=root/'status.json'
        try:
            atomic_text(root/'README.md', '# HOT3D benchmarkv2\n\n'+
                f'完整操作说明：[HOT3D.md]({Path(__file__).parents[1]/"HOT3D.md"})。\n\n'+
                '运行状态见status.json，数据准备进度见prepare_status.json，完成报告见[report.md](report.md)。\n\n'+
                '数据位于data/，缺失权重恢复到assets/，预测和共享模型缓存位于runs/；源码和模型环境在/mnt/workspace。\n')
            config=localize(config)
            if not prepare_only and not score_only:
                from .assets import prepare as prepare_assets
                write_json(status,dict(status='running',stage='restore_assets',updated_utc=utc_now()))
                config=prepare_assets(config)
                check=readiness(config);write_json(root/'readiness.json',check)
                if not check['ready']:raise RuntimeError('；'.join(check['errors']))
            if not (data/'protocol.json').exists() and shutil.disk_usage(root).free<60*1024**3 and len(config['data']['clip_ids'] or range(1516))>100:
                raise RuntimeError('目标磁盘可用空间不足60GiB；请将root设在更大的本地盘')
            write_json(status,dict(status='running',stage='load_dataset' if score_only else 'data_preparation',updated_utc=utc_now()))
            protocol,audit,frames,gt=load(config) if score_only else prepare(config)
            if prepare_only:
                write_json(status,dict(status='data_ready',frames=len(frames),benchmark_id=protocol['benchmark_id'],updated_utc=utc_now()))
                return dict(data_ready=True,frames=len(frames),root=str(root))
            bind(destination/'experiment.json',dict(config=config,benchmark_id=protocol['benchmark_id']))
            inputs=allowed_inputs(config,frames,gt);env=child_environment({})
            env.update(HF_HOME='/root/.cache/huggingface',XFORMERS_DISABLED='1')
            rgb_manifest=digest([{k:f[k] for k in ('frame_id','rgb_sha256','intrinsics')} for f in frames])
            for track in tracks:
                directory=destination/track/'full'
                write_json(destination/track/'input_manifest.json',inputs[track])
                task=dict(config=config,run_root=str(destination),track=track,frames=inputs[track],
                    benchmark_id=protocol['benchmark_id'],classes=protocol['classes'],
                    input_manifest=digest(inputs[track]),rgb_manifest=rgb_manifest)
                write_json(directory/'selection.json',dict(phase='full',frame_ids=[f['frame_id'] for f in frames],benchmark_id=protocol['benchmark_id']))
                if not score_only:
                    write_json(status,dict(status='running',stage='frontend',track=track,updated_utc=utc_now()))
                    run_child('ioutest.hot3d.worker',dict(task,job='frontend'),config['frontend']['python'],env,directory/'logs/frontend_task.json')
                for key in TRACKS[track]:
                    write_json(status,dict(status='running',stage='score' if score_only else 'inference',track=track,model=key,updated_utc=utc_now()))
                    if not score_only:run_child('ioutest.hot3d.worker',dict(task,model_id=key),config['models'][key]['python'],env,directory/'logs'/(key+'_task.json'))
                    from .score import score
                    write_json(status,dict(status='running',stage='score',track=track,model=key,updated_utc=utc_now()))
                    score(config,destination,track,key,frames,gt,protocol)
                from .report import assemble
                assemble(config,track,frames,gt,protocol,audit)
            write_json(status,dict(status='complete',tracks=list(tracks),frames=len(frames),benchmark_id=protocol['benchmark_id'],updated_utc=utc_now()))
            return dict(complete=True,frames=len(frames),root=str(root))
        except BaseException as exc:
            write_json(status,dict(status='failed',error=str(exc),updated_utc=utc_now()))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description='HOT3D Aria：固定针孔视图、五配置、full和断点续跑')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--root',help='本地数据、缓存与报告的根目录；默认/root/benchmarkv2')
    parser.add_argument('--phase',choices=['full'],default='full',help='直接全量，不运行smoke/pilot')
    parser.add_argument('--tracks',nargs='+',choices=list(TRACKS),default=list(TRACKS))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true',help='只准备数据，不需要PPU，不运行模型')
    mode.add_argument('--score-only',action='store_true',help='只核验已有预测、评分并生成报告')
    mode.add_argument('--check',action='store_true',help='只读检查环境/权重/PPU，不生成数据')
    args=parser.parse_args(argv);config=read_json(args.config)
    if args.root:config['root']=str(Path(args.root).expanduser().resolve())
    result=readiness(localize(config)) if args.check else run(config,args.tracks,args.prepare_only,args.score_only)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return int(args.check and not result['ready'])


if __name__=='__main__':raise SystemExit(main())
