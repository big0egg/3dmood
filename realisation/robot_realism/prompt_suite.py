"""Full agentview experiment: 4800 generic + 200 specific + 500 eval + 200 paired."""
import argparse
from copy import deepcopy
import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from .common import (PROJECT, atomic_json, atomic_text, file_hash, read_json,
                     writer_lock)
from .config import load_config
from .dataset import bundle
from .planning import load_plan
from .prompting import (GROUP_COUNTS, build_selection,
                        build_specific_row, write_jsonl)

DEFAULT_ROOT = Path('/root/robot-realism/agentview_prompt_v2')
DEFAULT_CACHE = Path('/root/robot-realism/cosmos25/cache')
OLD_CACHE = Path('/tmp/robot-realism/cosmos25/cache')


def freeze_json(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen experiment differs: {path}; use a new experiment root')
    else:
        atomic_json(path, value)


def freeze_text(path, value):
    if Path(path).exists():
        if Path(path).read_text() != value:
            raise ValueError(f'Frozen text differs: {path}; use a new experiment root')
    else:
        atomic_text(path, value)


def migrate_cache(source, target):
    """Copy missing cache files and preserve internal HF symlinks; never move a live cache."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or not source.is_dir():
        return
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Cache source and destination must not contain each other')
    for directory, directories, filenames in os.walk(source, followlinks=False):
        for name in directories[:]:
            if name == '.locks':
                directories.remove(name)
        relative = Path(directory).relative_to(source)
        for name in filenames:
            src = Path(directory) / name
            # Incomplete downloads stay in the old cache; Hub can resume missing files later.
            if name.endswith(('.incomplete', '.lock')):
                continue
            dst = target / relative / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_symlink():
                if not src.resolve().is_relative_to(source):
                    raise ValueError(f'Cache symlink leaves source: {src}')
                link = os.readlink(src)
                if os.path.isabs(link):
                    link = os.path.relpath(target / src.resolve().relative_to(source), dst.parent)
                if dst.is_symlink():
                    if os.readlink(dst) != link:
                        raise ValueError(f'Cache symlink differs: {dst}')
                elif dst.exists():
                    if file_hash(src) != file_hash(dst):
                        raise ValueError(f'Cache content differs: {dst}')
                else:
                    dst.symlink_to(link)
            elif dst.exists():
                if file_hash(src) != file_hash(dst):
                    raise ValueError(f'Cache content differs: {dst}')
            else:
                temporary = dst.with_name(dst.name + '.copying')
                shutil.copy2(src, temporary)
                if file_hash(src) != file_hash(temporary):
                    raise OSError(f'Cache copy checksum failed: {dst}')
                os.replace(temporary, dst)


def initialize(root, source, python, cache, template=None):
    import yaml
    root = Path(root)
    base = load_config(template or PROJECT / 'configs/realism/cosmos25_libero_agentview_prompt_v2.yaml',
                       input_root=source)
    base['input']['cameras'] = ['agentview']
    base['runtime']['python'] = os.path.abspath(python)
    base['runtime']['cache'] = str(Path(cache).resolve())
    generic = base['generation']['prompt']
    negative = base['generation']['negative_prompt']
    selection, chosen = build_selection(base)
    source_code = {name: file_hash(Path(__file__).with_name(name))
                   for name in ('prompting.py', 'prompt_suite.py')}
    identity = {'selection_id': selection['selection_id'], 'input': base['input'],
                'runtime': base['runtime'], 'generation': base['generation'],
                'controls': base['controls'], 'source_code_sha256': source_code}
    freeze_json(root / 'experiment.json', identity)
    selection_path = root / 'manifests/selection.json'
    freeze_json(selection_path, selection)
    freeze_text(root / 'prompts/generic.txt', generic + '\n')
    freeze_text(root / 'prompts/negative.txt', negative + '\n')
    specific_path = root / 'prompts/specific.jsonl'
    rows = []
    existing = {}
    if specific_path.exists():
        import json
        previous = [json.loads(line) for line in specific_path.read_text().splitlines() if line]
        existing = {r['state_id']: r for r in previous}
        if len(existing) != len(previous) or set(existing) - {r['state_id'] for r in chosen}:
            raise ValueError('Specific prompt cache has duplicate or foreign IDs')
    for number, record in enumerate(chosen, 1):
        row = build_specific_row(base, record, generic)
        if record['state_id'] in existing and existing[record['state_id']] != row:
            raise ValueError('Specific prompt cache/source changed; use a new experiment root')
        rows.append(row)
        if record['state_id'] not in existing:
            write_jsonl(specific_path, rows + [existing[r['state_id']] for r in chosen[number:]
                                              if r['state_id'] in existing])
        rgb = bundle(source, record) / 'agentview/rgb.png'
        target = root / 'previews/source' / (record['state_id'] + '.png')
        if target.exists():
            if file_hash(target) != row['source_rgb_sha256']:
                raise ValueError(f'Source preview changed: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rgb, target)
        if number % 20 == 0:
            print(f'逐帧提示词已准备 {number}/200', flush=True)
    # Order is deterministic, including after an interrupted partial preparation.
    write_jsonl(specific_path, rows)
    configs = {}
    for name, mode in [('main', 'mixed'), ('paired_general', 'paired_general')]:
        config = deepcopy(base)
        config['name'] = 'agentview_prompt_v2_' + name
        config['output'] = {'root': str(root / 'datasets' / name),
                            'run_dir': str(root / 'runs' / name),
                            'scratch': str(root / 'scratch' / name)}
        config['prompting'] = {'mode': mode, 'selection_manifest': str(selection_path),
                               'specific_manifest': str(specific_path)}
        path = root / 'configs' / (name + '.yaml')
        freeze_text(path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        configs[name] = path
    return configs


def paired_specs_match(first, second):
    a, b = deepcopy(first), deepcopy(second)
    a.pop('prompt'); b.pop('prompt')
    for control in ('depth', 'edge'):
        if control in a:
            a[control].pop('control_path', None)
            b.get(control, {}).pop('control_path', None)
    return a == b


def validate_prompt_lengths(configs):
    """The native text encoder truncates at 512 tokens, including its chat wrapper."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    config = load_config(configs['main'])
    snapshot = snapshot_download(repo_id='nvidia/Cosmos-Reason1-7B',
        revision='3210bec0495fdc7a8d3dbb8d58da5711eab4b423',
        cache_dir=str(Path(config['runtime']['cache'])/'huggingface/hub'), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    frozen = load_plan(config)
    prompts = {a['prompt'] for a in frozen['prompt_assignments'].values()}
    prompts.add(config['generation']['negative_prompt'])
    lengths = []
    for prompt in prompts:
        messages = [{'role': 'system', 'content':
                     'You are a helpful assistant who will provide prompts to an image generator.'},
                    {'role': 'user', 'content': prompt}]
        length = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        if length > 512:
            raise ValueError(f'Prompt has {length} tokens including chat wrapper; native limit is 512. '
                             'Use a shorter prompt in a new experiment version')
        lengths.append(length)
    return {'checked_prompts': len(prompts), 'max_tokens': max(lengths), 'native_limit': 512}


def comparison_panel(source, generic, specific, mask_path, target):
    with Image.open(source) as im:
        original = im.convert('RGB')
    with Image.open(generic) as im:
        general = im.convert('RGB')
    if specific is None:
        detail = general.copy()
    else:
        with Image.open(specific) as im:
            detail = im.convert('RGB')
    if original.size != general.size or original.size != detail.size:
        raise ValueError('Comparison image sizes disagree')
    with Image.open(mask_path) as im:
        mask = np.array(im)
    edges = np.zeros(mask.shape, bool)
    edges[1:] |= mask[1:] != mask[:-1]
    edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    overlay = np.array(detail)
    overlay[edges] = [0, 255, 0]
    w, h = original.size
    if specific is None:
        labels = ('SOURCE', 'GENERIC', 'SOURCE MASK ON GENERIC')
        panels = (original, general, Image.fromarray(overlay))
    else:
        labels = ('SOURCE', 'GENERIC', 'SPECIFIC', 'SOURCE MASK ON SPECIFIC')
        panels = (original, general, detail, Image.fromarray(overlay))
    canvas = Image.new('RGB', (len(panels)*w, h+28), 'white')
    for i, (label, panel) in enumerate(zip(labels, panels)):
        canvas.paste(panel, (i*w, 28))
        ImageDraw.Draw(canvas).text((i*w+4, 5), label, fill='black')
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=93)


def export_results(root, configs):
    from .pipeline import verify_result
    root = Path(root)
    rows = {name: [] for name in [*GROUP_COUNTS, 'paired_general']}
    metas = {}
    for name, path in configs.items():
        config = load_config(path)
        frozen = load_plan(config)
        info = read_json(Path(config['output']['root']) / 'dataset_info.json')
        runtime = read_json(Path(config['output']['root']) / 'runtime_identity.json')
        metas[name] = {'runtime': runtime, 'frames': {}}
        for record in frozen['records']:
            if not verify_result(config['output']['root'], record, ['agentview'], info['profile_id']):
                raise ValueError('Cannot export an incomplete experiment')
            folder = bundle(config['output']['root'], record) / 'agentview'
            meta = read_json(folder / 'render_meta.json')
            state_id = record['state_id']
            assignment = frozen['prompt_assignments'][state_id]
            if meta['inference']['prompt'] != assignment['prompt']:
                raise ValueError('Generated prompt differs from frozen plan')
            metas[name]['frames'][state_id] = (record, folder, meta)
            rows[assignment['group']].append({'state_id': state_id, 'camera': 'agentview',
                'split': record['split'], 'scene_id': record['scene_id'], 'group': assignment['group'],
                'rgb_path': str(folder / 'rgb.png'), 'annotation_path': str(folder / 'annotation.json'),
                'depth_path': str(folder / 'depth.npz'), 'instance_path': str(folder / 'instance.png'),
                'render_meta_path': str(folder / 'render_meta.json'), 'seed': meta['inference']['seed'],
                'prompt': assignment['prompt'], 'source_rgb_sha256': meta['source_rgb_sha256'],
                'gt_status': 'inherited_simulation_gt_requires_visual_review'})
    expected = {**GROUP_COUNTS, 'paired_general': GROUP_COUNTS['train_specific']}
    if {k: len(v) for k, v in rows.items()} != expected:
        raise ValueError('Final groups do not equal 4800/200/500/200')
    if metas['main']['runtime'] != metas['paired_general']['runtime']:
        raise ValueError('Paired generations used different runtime/checkpoint identities')
    report_lines = ['# Agentview 提示词对照报告', '',
        '主数据 5,500 张，对照 200 张。所有逐帧描述来自源 RGB、实例掩码和可见标注；未人工审核。', '',
        '原始仿真图 / 通用提示词 / 逐帧提示词 / 原实例边界叠加逐帧结果。',
        '边界叠加仅用于检查，不证明生成 RGB 与继承 GT 对齐。', '']
    review = io.StringIO()
    writer = csv.writer(review)
    writer.writerow(['state_id', 'scene_id', 'realism_preference', 'generic_semantics_ok',
                     'specific_semantics_ok', 'generic_geometry_ok', 'specific_geometry_ok', 'notes'])
    for row in sorted(rows['train_specific'], key=lambda r: r['state_id']):
        state_id = row['state_id']
        record, folder, specific = metas['main']['frames'][state_id]
        _, general_folder, general = metas['paired_general']['frames'][state_id]
        if (specific['source_rgb_sha256'] != general['source_rgb_sha256']
                or specific['control_files_sha256'] != general['control_files_sha256']
                or not paired_specs_match(specific['inference'], general['inference'])):
            raise ValueError(f'Paired controls/seed/parameters differ: {state_id}')
        source = bundle(load_config(configs['main'])['input']['root'], record) / 'agentview'
        image = root / 'previews/paired' / (state_id + '.jpg')
        comparison_panel(source/'rgb.png', general_folder/'rgb.png', folder/'rgb.png',
                         source/'instance.png', image)
        atomic_text(root/'reports/prompts'/f'{state_id}.txt',
                    'GENERIC\n\n'+general['inference']['prompt']+'\n\nSPECIFIC\n\n'+specific['inference']['prompt']+'\n')
        report_lines.append(f'- {record["scene_id"]}/{state_id}: [对照图](../previews/paired/{image.name}) '
                            f'[提示词](prompts/{state_id}.txt)')
        writer.writerow([state_id, record['scene_id'], '', '', '', '', '', ''])
    from .reporting import report
    # Export every eval frame for review without using eval results to tune prompts.
    main_config = load_config(configs['main'])
    frozen = load_plan(main_config)
    report(main_config, frozen, max_previews=40)
    for row in rows['eval_general']:
        state_id = row['state_id']
        record, folder, _ = metas['main']['frames'][state_id]
        source = bundle(main_config['input']['root'], record) / 'agentview'
        image = root / 'previews/eval' / (state_id + '.jpg')
        comparison_panel(source/'rgb.png', folder/'rgb.png', None, source/'instance.png', image)
    for name, values in rows.items():
        write_jsonl(root / 'manifests' / (name + '.jsonl'), sorted(values, key=lambda r: r['state_id']))
    write_jsonl(root/'manifests/train_all_general.jsonl', rows['train_general']+rows['paired_general'])
    write_jsonl(root/'manifests/train_mixed.jsonl', rows['train_general']+rows['train_specific'])
    atomic_text(root/'reports/paired_review.csv', review.getvalue())
    report_lines += ['', '500 张评测图预览位于 `../previews/eval/`。',
                     '人工审核表为 `paired_review.csv`；空白表示尚未审核。']
    atomic_text(root/'reports/comparison.md', '\n'.join(report_lines)+'\n')
    summary = {'complete': True, 'generated_images': sum(expected.values()), 'counts': expected,
               'visual_gt_alignment_certified': False, 'visual_review_status': 'pending',
               'prompt_authoring': 'per-frame RGB/mask/GT grounded descriptions, not VLM or manual captions',
               'report': str(root/'reports/comparison.md')}
    atomic_json(root/'reports/summary.json', summary)
    return summary


def run_stage(root, config_path, command, extra=()):
    from .pipeline import worker_env
    config = load_config(config_path)
    label = config['name'] + '_' + command
    atomic_json(Path(root)/'runs/suite_status.json', {'stage': label, 'complete': False})
    log = Path(root)/'runs'/config_path.stem/(command+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'执行 {label}，日志：{log}', flush=True)
    with log.open('a') as stream:
        subprocess.run([config['runtime']['python'], '-u', '-B', '-m', 'robot_realism', command,
                        '--config', str(config_path), *extra], cwd=PROJECT, env=worker_env(config),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='直接全量：agentview 4800+200+500，额外同帧对照200')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--input', type=Path, default=Path('/root/opendatasets/LIBERO_GT_v1'))
    parser.add_argument('--python', default='/tmp/robot-realism/cosmos25/venv/bin/python')
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--cache-source', type=Path, default=OLD_CACHE)
    parser.add_argument('--template', type=Path, help='开始新实验时使用的配置模板')
    parser.add_argument('--setup-only', action='store_true', help='只准备清单、逐帧提示词和配置')
    args = parser.parse_args(argv)
    root, cache = args.root.expanduser().resolve(), args.cache.expanduser().resolve()
    if not root.is_relative_to('/root') or not cache.is_relative_to('/root'):
        parser.error('实验产物和权重缓存必须位于 /root 下')
    if root == cache or root.is_relative_to(cache) or cache.is_relative_to(root):
        parser.error('实验根目录与权重缓存必须互不包含')
    source = args.input.expanduser().resolve()
    if source == root or source.is_relative_to(root) or root.is_relative_to(source):
        parser.error('输入数据与实验根目录必须互不包含')
    acquired = False
    try:
        with writer_lock(root/'runs'):
            acquired = True
            configs = initialize(root, source, args.python, cache, args.template)
            # Validate exact prompt membership and bind prompt content before any model loads.
            from .planning import plan
            for config in configs.values():
                plan(load_config(config))
            if args.setup_only:
                result = {'setup_complete': True, 'root': str(root), 'render_started': False}
            else:
                atomic_json(root/'runs/suite_status.json', {'stage': 'copy_weight_cache', 'complete': False})
                migrate_cache(args.cache_source, cache)
                atomic_json(root/'runs/prompt_token_check.json', validate_prompt_lengths(configs))
                for config in configs.values():
                    run_stage(root, config, 'doctor')
                    run_stage(root, config, 'render', ['--resume'])
                    run_stage(root, config, 'validate')
                atomic_json(root/'runs/suite_status.json', {'stage': 'export_reports', 'complete': False})
                result = export_results(root, configs)
                atomic_json(root/'runs/suite_status.json', dict(result, stage='complete'))
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if acquired:
            try:
                atomic_json(root/'runs/suite_status.json', {'stage': 'failed', 'complete': False,
                            'error': str(error) or type(error).__name__,
                            'resume': '重新执行相同命令；不需要试生成阶段'})
            except OSError:
                pass
        print(f'全量任务停止：{error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
