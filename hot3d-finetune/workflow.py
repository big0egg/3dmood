#!/usr/bin/env python3
"""Standard experiment order, without changing existing prediction identities."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hot3d_ft.cli import check, child
from hot3d_ft.common import PROJECT, checksum, config, lock, write


def steps(cfg):
    data_python = cfg['data_python']
    jobs = [dict(stage='prepare', args=['prepare'], python=data_python),
            dict(stage='verify', args=['verify', '--images'], python=data_python)]
    for phase in ('baseline', 'training', 'finetuned'):
        if phase == 'training':
            for model in ('3dmood', 'wilddet3d'):
                jobs.append(dict(stage=phase, model=model, args=['train', '--model', model],
                                 python=cfg['models'][model]['python']))
            continue
        for track, model in (('text', '3dmood'), ('text', 'wilddet3d'), ('gt-box', 'wilddet3d')):
            jobs.append(dict(stage=phase, model=model, track=track,
                             args=['evaluate', '--track', track, '--model', model, '--variant', phase],
                             python=cfg['models'][model]['python']))
    jobs.append(dict(stage='report', args=['report'], python=data_python))
    return jobs


def run(cfg):
    readiness = check(cfg)
    if not readiness['ready']:
        raise RuntimeError('环境检查未通过：' + json.dumps(readiness, ensure_ascii=False))
    root = Path(cfg['root'])
    jobs = steps(cfg)
    with lock(root/'.suite.lock'):
        try:
            write(root/'workflow.json', dict(
                source_sha256=checksum(Path(__file__)),
                order='prepare -> baseline text+gt-box -> train -> finetuned text+gt-box -> report',
                jobs=jobs))
            for number, job in enumerate(jobs, 1):
                status = {k: v for k, v in job.items() if k not in ('args', 'python')}
                write(root/'suite_status.json', dict(complete=False, step=number, total_steps=len(jobs), **status))
                print(f'全流程 {number}/{len(jobs)}：' + ' '.join(job['args']), flush=True)
                child(cfg, job['args'], job['python'])
            result = dict(stage='complete', complete=True, total_steps=len(jobs),
                          report=str(root/'report/report.md'), tracks=['text', 'gt-box'])
            write(root/'suite_status.json', result)
            return result
        except BaseException as exc:
            write(root/'suite_status.json', dict(stage='failed', complete=False, error=str(exc),
                  resume='再次执行相同run_full.sh命令，保持配置与源码，校验并复用已完成结果'))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='准备数据→两类基线→微调→两类微调后评测→最终报告')
    parser.add_argument('--config', type=Path, help='默认configs/default.json')
    parser.add_argument('--root', type=Path, help='覆盖产物根目录')
    parser.add_argument('--dry-run', action='store_true', help='只展示顺序，不检查PPU、不写文件、不启动任务')
    args = parser.parse_args(argv)
    cfg = config(args.config, args.root)
    result = steps(cfg) if args.dry_run else run(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
