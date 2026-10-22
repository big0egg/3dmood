"""Supplement an existing text benchmark without reloading its training jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from ..common import PROJECT, REPO, config, lock, write


def supplement(cfg, report_dir=None, score_only=False):
    directory = Path(cfg['root'])/'evaluation/gt-box'
    with lock(directory/'.suite.lock'):
        snapshot = directory/'resolved_config.json'
        write(snapshot, cfg)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SAM3_COMPILE='0',
                           XFORMERS_DISABLED='1', MPLBACKEND='Agg',
                           OMP_NUM_THREADS=str(cfg['training']['threads']),
                           OMP_THREAD_LIMIT=str(cfg['training']['threads']), PYTHONFAULTHANDLER='1')
        environment['PYTHONPATH'] = str(PROJECT)+os.pathsep+str(REPO)
        try:
            for variant in ('baseline', 'finetuned'):
                args = ['evaluate', '--track', 'gt-box', '--model', 'wilddet3d', '--variant', variant]
                python = cfg['models']['wilddet3d']['python']
                if score_only:
                    args.append('--score-only')
                    python = cfg['data_python']
                write(directory/'suite_status.json', dict(stage=variant, complete=False))
                subprocess.run([python, '-B', str(PROJECT/'run.py'), *args, '--config', str(snapshot)],
                               env=environment, cwd=REPO, check=True)
            from .report import merge
            result = merge(cfg, report_dir)
            write(directory/'suite_status.json', dict(stage='complete', complete=True, **result))
            return result
        except BaseException as exc:
            write(directory/'suite_status.json', dict(stage='failed', complete=False, error=str(exc),
                  resume='重新执行box-supplement；保留已完成逐目标/逐帧记录'))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='HOT3D框提示补充评测与合并报告；无需重新训练')
    parser.add_argument('command', choices=['evaluate', 'report', 'box-supplement'])
    parser.add_argument('--track', choices=['gt-box'], default='gt-box', help='GT投影二维框提示')
    parser.add_argument('--model', choices=['wilddet3d'], default='wilddet3d', help='仅WildDet3D支持原生框提示')
    parser.add_argument('--variant', choices=['baseline', 'finetuned'], help='evaluate必填')
    parser.add_argument('--config', type=Path, help='默认configs/default.json；可使用原resolved_config.json')
    parser.add_argument('--root', type=Path, help='沿用原产物目录，覆盖配置root')
    parser.add_argument('--score-only', action='store_true', help='只给完整预测评分，不加载模型，不需要PPU')
    parser.add_argument('--report-dir', type=Path, help='report/box-supplement的报告输出目录；默认ROOT/report')
    parser.add_argument('--output', type=Path, help='可选保存命令返回的状态JSON')
    args = parser.parse_args(argv)
    cfg = config(args.config, args.root)
    if args.command == 'evaluate':
        if not args.variant:
            parser.error('evaluate --track gt-box必须指定--variant')
        from .evaluate import infer, score
        if not args.score_only:
            infer(cfg, args.variant)
        result = score(cfg, args.variant)
    elif args.command == 'report':
        if args.score_only:
            parser.error('--score-only用于evaluate/box-supplement')
        from .report import merge
        result = merge(cfg, args.report_dir)
    else:
        result = supplement(cfg, args.report_dir, args.score_only)
    if args.output:
        write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
