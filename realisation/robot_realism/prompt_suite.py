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

DEFAULT_ROOT = Path('/root/robot-realism/agentview_prompt_v3')
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


def initialize(root, source, python, cache, template=None, reuse_from=None):
    import yaml
    root = Path(root)
    base = load_config(template or PROJECT / 'configs/realism/cosmos25_libero_agentview_prompt_v3.yaml',
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
                'controls': base['controls'], 'execution': base.get('execution', {}),
                'source_code_sha256': source_code}
    freeze_json(root / 'experiment.json', identity)
    selection_path = root / 'manifests/selection.json'
    freeze_json(selection_path, selection)
    freeze_text(root / 'prompts/generic.txt', generic + '\n')
    freeze_text(root / 'prompts/negative.txt', negative + '\n')
    specific_path = root / 'prompts/specific.jsonl'
    rows = []
    existing = {}
    if not specific_path.exists() and reuse_from and Path(reuse_from).is_dir():
        from .prompting import read_selection
        previous_root = Path(reuse_from)
        if read_selection(previous_root/'manifests/selection.json') != selection:
            raise ValueError('Previous selection differs; refusing to change the selected 200 frames')
        for name, expected in [('generic', generic), ('negative', negative)]:
            if (previous_root/'prompts'/f'{name}.txt').read_text().strip() != expected.strip():
                raise ValueError(f'Previous {name} prompt differs')
        import json
        previous_rows = [json.loads(line) for line in (previous_root/'prompts/specific.jsonl').read_text().splitlines() if line]
        existing = {r['state_id']: r for r in previous_rows}
        if (len(existing) != len(chosen) or len(previous_rows) != len(existing)
                or set(existing) != {r['state_id'] for r in chosen}):
            raise ValueError('Previous specific prompt manifest must contain exactly 200 unique entries')
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
        config['name'] = base['name'].removesuffix('_main') + '_' + name
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
    from .blocked import ProgressTracker, blocked_path
    root = Path(root)
    rows = {name: [] for name in [*GROUP_COUNTS, 'paired_general']}
    blocked_counts = {name: 0 for name in rows}
    planned_counts = {name: 0 for name in rows}
    blocked_rows, metas = [], {}
    for name, path in configs.items():
        config = load_config(path)
        frozen = load_plan(config)
        info = read_json(Path(config['output']['root']) / 'dataset_info.json')
        tracker = ProgressTracker(config, frozen, info['profile_id'])
        if not tracker.summary()['execution_complete']:
            raise ValueError('Cannot export: some frames have neither a valid commit nor a blocked record')
        runtime = read_json(Path(config['output']['root']) / 'runtime_identity.json')
        metas[name] = {'runtime': runtime, 'frames': {}, 'blocked': tracker.blocked}
        run_blocked = []
        for record in frozen['records']:
            state_id = record['state_id']
            assignment = frozen['prompt_assignments'][state_id]
            group = assignment['group']
            planned_counts[group] += 1
            if state_id in tracker.blocked:
                rejection = tracker.blocked[state_id]
                blocked_counts[group] += 1
                run_blocked.append(rejection)
                blocked_rows.append(dict(rejection, variant=name,
                    blocked_record_path=str(blocked_path(config, record, 'agentview'))))
                continue
            folder = bundle(config['output']['root'], record) / 'agentview'
            meta = read_json(folder / 'render_meta.json')
            if meta['inference']['prompt'] != assignment['prompt']:
                raise ValueError('Generated prompt differs from frozen plan')
            metas[name]['frames'][state_id] = (record, folder, meta)
            rows[group].append({'state_id': state_id, 'camera': 'agentview',
                'split': record['split'], 'scene_id': record['scene_id'], 'group': group,
                'rgb_path': str(folder / 'rgb.png'), 'annotation_path': str(folder / 'annotation.json'),
                'depth_path': str(folder / 'depth.npz'), 'instance_path': str(folder / 'instance.png'),
                'render_meta_path': str(folder / 'render_meta.json'), 'seed': meta['inference']['seed'],
                'prompt': assignment['prompt'], 'source_rgb_sha256': meta['source_rgb_sha256'],
                'gt_status': 'inherited_simulation_gt_requires_visual_review'})
        write_jsonl(Path(config['output']['run_dir'])/'blocked.jsonl', run_blocked)
    expected = {**GROUP_COUNTS, 'paired_general': GROUP_COUNTS['train_specific']}
    if planned_counts != expected or any(len(rows[k])+blocked_counts[k] != n for k, n in expected.items()):
        raise ValueError('Group coverage must account for every planned frame: success or explicit block')
    if metas['main']['runtime'] != metas['paired_general']['runtime']:
        raise ValueError('Paired generations used different runtime/checkpoint identities')
    generated = sum(map(len, rows.values()))
    planned = sum(expected.values())
    report_lines = ['# Agentview 提示词对照报告', '',
        f'计划 {planned} 张；实际成功 {generated} 张；内容拦截 {len(blocked_rows)} 张；未处理 0 张。',
        '逐帧描述来自源 RGB、实例掩码和可见标注，尚未人工审核。', '',
        '成对图片：原始仿真图 / 通用提示词 / 逐帧提示词 / 原实例边界叠加逐帧结果。',
        '边界叠加仅用于检查，不证明生成 RGB 与继承 GT 对齐。', '']
    review = io.StringIO()
    writer = csv.writer(review)
    writer.writerow(['state_id', 'scene_id', 'realism_preference', 'generic_semantics_ok',
                     'specific_semantics_ok', 'generic_geometry_ok', 'specific_geometry_ok', 'notes'])
    main_config = load_config(configs['main'])
    selection = read_json(main_config['prompting']['selection_manifest'])
    paired_ids, missing_pairs = set(), []
    for state_id in sorted(selection['groups']['train_specific']):
        if state_id not in metas['main']['frames'] or state_id not in metas['paired_general']['frames']:
            missing_pairs.append({'state_id': state_id,
                'specific_status': 'generated' if state_id in metas['main']['frames'] else 'blocked',
                'generic_status': 'generated' if state_id in metas['paired_general']['frames'] else 'blocked'})
            continue
        record, folder, specific = metas['main']['frames'][state_id]
        _, general_folder, general = metas['paired_general']['frames'][state_id]
        if (specific['source_rgb_sha256'] != general['source_rgb_sha256']
                or specific['control_files_sha256'] != general['control_files_sha256']
                or not paired_specs_match(specific['inference'], general['inference'])):
            raise ValueError(f'Paired controls/seed/parameters differ: {state_id}')
        paired_ids.add(state_id)
        source = bundle(main_config['input']['root'], record) / 'agentview'
        image = root / 'previews/paired' / (state_id + '.jpg')
        comparison_panel(source/'rgb.png', general_folder/'rgb.png', folder/'rgb.png',
                         source/'instance.png', image)
        atomic_text(root/'reports/prompts'/f'{state_id}.txt',
                    'GENERIC\n\n'+general['inference']['prompt']+'\n\nSPECIFIC\n\n'+specific['inference']['prompt']+'\n')
        report_lines.append(f'- {record["scene_id"]}/{state_id}: [对照图](../previews/paired/{image.name}) '
                            f'[提示词](prompts/{state_id}.txt)')
        writer.writerow([state_id, record['scene_id'], '', '', '', '', '', ''])
    from .reporting import report
    report(main_config, load_plan(main_config), max_previews=40)
    for row in rows['eval_general']:
        state_id = row['state_id']
        record, folder, _ = metas['main']['frames'][state_id]
        source = bundle(main_config['input']['root'], record) / 'agentview'
        comparison_panel(source/'rgb.png', folder/'rgb.png', None, source/'instance.png',
                         root/'previews/eval'/f'{state_id}.jpg')
    for name, values in rows.items():
        write_jsonl(root/'manifests'/f'{name}.jsonl', sorted(values, key=lambda r: r['state_id']))
    # Both training experiments use exactly the same available source states.
    general_train = rows['train_general'] + [r for r in rows['paired_general'] if r['state_id'] in paired_ids]
    mixed_train = rows['train_general'] + [r for r in rows['train_specific'] if r['state_id'] in paired_ids]
    write_jsonl(root/'manifests/train_all_general.jsonl', general_train)
    write_jsonl(root/'manifests/train_mixed.jsonl', mixed_train)
    write_jsonl(root/'manifests/paired_missing.jsonl', missing_pairs)
    write_jsonl(root/'reports/blocked.jsonl', blocked_rows)
    blocked_csv = io.StringIO()
    bw = csv.writer(blocked_csv)
    bw.writerow(['variant', 'state_id', 'camera', 'split', 'group', 'stage', 'reason', 'created_utc',
                 'seed', 'source_rgb_path', 'blocked_record_path', 'resolution_status'])
    for row in blocked_rows:
        bw.writerow([row.get(k, '') for k in ['variant', 'state_id', 'camera', 'split', 'group', 'stage',
                     'reason', 'created_utc']] + [row['context']['seed']] +
                    [row.get(k, '') for k in ['source_rgb_path', 'blocked_record_path', 'resolution_status']])
    atomic_text(root/'reports/blocked.csv', blocked_csv.getvalue())
    atomic_text(root/'reports/paired_review.csv', review.getvalue())
    report_lines += ['', f'完整成对对照 {len(paired_ids)} 组；缺少至少一侧的状态 {len(missing_pairs)} 个。',
        f'两套对照训练入口各有 {len(mixed_train)} 个相同源状态。',
        f'实际可用评测图片 {len(rows["eval_general"])} 张，预览位于 `../previews/eval/`。',
        '拦截清单：[blocked.csv](blocked.csv)；每条记录保留提示词、种子和源图位置。',
        '人工审核表为 `paired_review.csv`；空白表示尚未审核。']
    atomic_text(root/'reports/comparison.md', '\n'.join(report_lines)+'\n')
    complete = len(blocked_rows) == 0
    summary = {'complete': complete, 'dataset_complete': complete, 'execution_complete': True,
        'status': 'complete' if complete else 'finished_with_blocks',
        'planned_images': planned, 'generated_images': generated, 'blocked_images': len(blocked_rows),
        'pending_images': 0, 'counts': {k: len(v) for k, v in rows.items()},
        'blocked_counts': blocked_counts, 'planned_counts': expected,
        'comparison_pairs': len(paired_ids), 'comparison_training_samples': len(mixed_train),
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
    parser.add_argument('--reuse-prompts-from', type=Path,
                        default=Path('/root/robot-realism/agentview_prompt_v2'),
                        help='核验并复用旧分组和提示词，不导入旧生成图片')
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
            configs = initialize(root, source, args.python, cache, args.template, args.reuse_prompts_from)
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
                    from .blocked import skip_enabled
                    run_stage(root, config, 'validate', ['--allow-blocked'] if skip_enabled(load_config(config)) else [])
                atomic_json(root/'runs/suite_status.json', {'stage': 'export_reports', 'complete': False})
                result = export_results(root, configs)
                atomic_json(root/'runs/suite_status.json', dict(result, stage=result['status']))
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
