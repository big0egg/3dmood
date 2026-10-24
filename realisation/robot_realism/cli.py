import argparse
import json
from pathlib import Path
import sys

from .common import PROJECT, read_json


def main(argv=None):
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == 'prompt-suite':
        from .prompt_suite import main as suite_main
        return suite_main(args_list[1:])
    parser = argparse.ArgumentParser(description='LIBERO RGB realism with Cosmos-Transfer2.5; original GT stays read-only. '
                                    '全量单视角提示词实验：python -m robot_realism prompt-suite --help')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('doctor', 'plan', 'prepare', 'render', 'validate', 'report', 'status'):
        child = commands.add_parser(command)
        child.add_argument('--config', type=Path, default=PROJECT / 'configs/realism/cosmos25_libero_v1.yaml')
        child.add_argument('--input', type=Path, help='Override committed simulation GT root')
        child.add_argument('--output', type=Path, help='Override new generated dataset root')
        if command in ('prepare', 'render'):
            child.add_argument('--limit', type=int, help='Total camera images including existing results; divisible by camera count')
        if command == 'render':
            child.add_argument('--resume', action='store_true')
        if command == 'validate':
            child.add_argument('--allow-partial', action='store_true')
            child.add_argument('--allow-blocked', action='store_true',
                               help='允许明确登记的拦截项；未登记缺帧或文件损坏仍视为错误')
        if command == 'report':
            child.add_argument('--max-previews', type=int, default=40)
    args = parser.parse_args(args_list)
    try:
        from .config import load_config
        config = load_config(args.config, input_root=args.input, output_root=args.output)
        if args.command == 'doctor':
            from .doctor import doctor
            result = doctor(config)
        elif args.command == 'plan':
            from .planning import plan
            value = plan(config)
            result = {'plan_id': value['plan_id'], 'states': len(value['records']), 'targets': value['targets']}
        elif args.command in ('prepare', 'render'):
            from .pipeline import prepare, render
            result = prepare(config, args.limit) if args.command == 'prepare' else render(config, args.limit, args.resume)
        elif args.command == 'status':
            path = Path(config['output']['run_dir']) / 'progress.json'
            if path.exists():
                result = {k: v for k, v in read_json(path).items() if k not in ('accepted', 'blocked')}
            else:
                result = {'status': 'not_started', 'accepted_images': 0, 'run_dir': config['output']['run_dir']}
        else:
            from .planning import load_plan
            from .reporting import validate, report
            frozen = load_plan(config)
            result = validate(config, frozen, args.allow_partial, args.allow_blocked) if args.command == 'validate' else report(config, frozen, args.max_previews)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get('ready') is False or result.get('valid') is False else 0
    except KeyboardInterrupt:
        print('已中断；同一版本可用 render --resume 恢复已提交状态。', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'失败：{error}', file=sys.stderr)
        return 1
