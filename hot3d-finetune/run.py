#!/usr/bin/env python3
"""Keep the hyphenated project directory usable without installing a package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hot3d_ft.cli import main as legacy_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ['--help']:
        print('新增入口：box-supplement（补跑两组框提示并合并报告）；'
              'evaluate --track gt-box；report --report-dir。原训练与文本命令保持兼容。\n')
    track = None
    if '--track' in args:
        index = args.index('--track')
        if index+1 < len(args):
            track = args[index+1]
    for arg in args:
        if arg.startswith('--track='):
            track = arg.split('=', 1)[1]
    if args and (args[0] in ('report', 'box-supplement') or track == 'gt-box'):
        from hot3d_ft.boxeval.cli import main as box_main
        return box_main(args)
    if track == 'text':
        if '--track' in args:
            index = args.index('--track'); del args[index:index+2]
        args = [a for a in args if a != '--track=text']
    result = legacy_main(args)
    if args and args[0] == 'run' and result == 0:
        from hot3d_ft.boxeval.cli import main as box_main
        return box_main(['box-supplement', *args[1:]])
    return result

if __name__ == '__main__':
    raise SystemExit(main())
