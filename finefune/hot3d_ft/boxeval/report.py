"""Append a separately labelled conditional experiment to the existing report."""
import csv
import io
import re
from pathlib import Path

import numpy as np

from ..common import atomic_bytes, checksum, data_path, digest, read, write
from ..dataset import load
from .evaluate import TRACK, VARIANTS, model_id, prompt_manifest

MARKER = '<!-- HOT3D_GT_BOX_SUPPLEMENT_BEGIN -->'
END_MARKER = '<!-- HOT3D_GT_BOX_SUPPLEMENT_END -->'


def write_csv(path, headings, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(headings)
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def completed_results(cfg, protocol, frames):
    base = Path(cfg['root']) / 'evaluation' / TRACK
    results = {}
    expected_inputs = digest(prompt_manifest(frames))
    count = sum(len(f['prompts']) for f in prompt_manifest(frames))
    for variant in VARIANTS:
        key = model_id(variant)
        metrics = base / 'full/metrics' / (key + '.json')
        if not metrics.exists():
            continue
        value = read(metrics)
        if not value.get('complete') or value['identity']['benchmark_id'] != protocol['benchmark_id']:
            raise ValueError('框提示指标不完整或数据协议不一致：' + key)
        audit = read(metrics.with_name(key + '_audit.json'))
        binding = read(base / 'models' / key / 'model.json')
        target_csv = metrics.with_name(key + '_targets.csv')
        if (audit['benchmark_id'] != protocol['benchmark_id']
                or audit['input_manifest_sha256'] != expected_inputs
                or binding['provenance']['input_manifest_sha256'] != expected_inputs
                or audit['signature'] != value['identity']['model']
                or audit['signature'] != binding['signature']
                or checksum(metrics) != audit['metrics_sha256']
                or checksum(target_csv) != audit['targets_sha256']
                or value['track'] != TRACK or value['frames'] != len(frames)
                or value['conditional']['targets'] != count):
            raise ValueError('框提示结果/逐目标表与已审计数据不一致：' + key)
        results[key] = value
    return results


def paired_results(cfg, results, frames):
    if len(results) != 2:
        return None
    rows = []
    for variant in VARIANTS:
        path = Path(cfg['root']) / 'evaluation' / TRACK / 'full/metrics' / (model_id(variant) + '_targets.csv')
        with path.open() as stream:
            items = list(csv.DictReader(stream))
        indexed = {(i['frame_id'], i['target_id']): i for i in items}
        expected = {(f['frame_id'], b['target_id']) for f in prompt_manifest(frames) for b in f['prompts']}
        if len(items) != len(indexed) or set(indexed) != expected:
            raise ValueError('配对评测目标清单缺失或重复')
        rows.append(indexed)
    a, b = rows
    names = list(results[model_id('baseline')]['conditional']['success'])
    summary = {}
    for name in names:
        counts = dict(both_success=0, improved=0, regressed=0, both_failed=0)
        for key in a:
            first, second = a[key][name], b[key][name]
            if first not in ('True', 'False') or second not in ('True', 'False'):
                raise ValueError('逐目标成功标记必须为True/False')
            category = ('both_success' if second == 'True' else 'regressed') if first == 'True' else (
                'improved' if second == 'True' else 'both_failed')
            counts[category] += 1
        summary[name] = dict(targets=len(a), **counts,
                             delta_percentage_points=100*(counts['improved']-counts['regressed'])/len(a) if a else None)
    return summary


def box_visualizations(cfg, frames, results, directory):
    from PIL import Image, ImageDraw
    from ioutest.hot3d.common import read_record, record_path, rgb
    from ioutest.visualize import _draw
    lines = ['### GT二维框提示固定帧对比图', '',
             '沿用评测清单均匀选取的12帧；青色矩形为输入GT二维框，绿色为GT三维框，'
             '红色为对应目标的预测三维框。此处展示全部有效返回，不用0.7筛选；'
             '逐目标表中保留失败及置信度。样例不是随机代表性抽样。', '']
    for index in np.linspace(0, len(frames)-1, min(12, len(frames)), dtype=int):
        frame = frames[int(index)]
        targets = {b['uid']: b for b in read(frame['gt_path']) if not b.get('ignore', False)}
        for key in results:
            base = Path(cfg['root']) / 'evaluation' / TRACK / 'models' / key
            binding = read(base / 'model.json')
            record = read_record(record_path(base/'predictions', frame), frame, binding['signature'])
            picture = Image.fromarray(rgb(frame)); draw = ImageDraw.Draw(picture)
            k = np.asarray(frame['intrinsics'])
            for i, item in enumerate(record['items'], 1):
                box = targets[item['target_id']]
                draw.rectangle(box['projected_bbox'], outline='cyan', width=2)
                draw.text(tuple(box['projected_bbox'][:2]), str(i), fill='cyan')
                _draw(draw, box, k, picture.width, picture.height, 'lime', 2, 'GT'+str(i))
                if item['status'] == 'ok':
                    _draw(draw, item['detection'], k, picture.width, picture.height, 'red', 2,
                          str(i)+f' {item["detection"]["score"]:.2f}')
                else:
                    draw.text((4, 4+12*i), f'target {i}: FAILED', fill='red')
            relative = Path('visualizations/gt-box') / key / f'{index:04d}.png'
            content = io.BytesIO(); picture.save(content, format='PNG')
            atomic_bytes(directory/relative, content.getvalue())
            lines.append(f'- {key}，{frame["frame_id"]}：[查看]({relative.as_posix()})')
    return lines


def merge(cfg, output=None):
    protocol, train, frames = load(cfg)
    root = Path(cfg['root'])
    source = root / 'report/report.md'
    directory = Path(output) if output else source.parent
    directory.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        from ..evaluation import report
        report(cfg)
    base = source.read_text().split(MARKER)[0].rstrip()
    base = base.replace('# HOT3D：三维头微调前后对比', '# HOT3D：文本检测与GT二维框提示下的微调前后对比', 1)
    note = '<!-- HOT3D_TEXT_SCOPE -->'
    if note not in base:
        heading, rest = base.split('\n', 1)
        base = (heading + '\n\n' + note + '\n'
                '> 前文指标与训练过程对应RGB＋固定类别文本＋相机内参；'
                '新增RGB＋GT二维框＋相机内参实验在文末独立展示，两种输入条件不混排。\n' + rest)
    if directory.resolve() != source.parent.resolve():
        # A preview in the workspace still links to the existing text figures.
        base = re.sub(r'\]\(([^)]+)\)', lambda m: m.group(0) if (
            '://' in m.group(1) or m.group(1).startswith(('/', '#'))
        ) else '](' + str(source.parent/m.group(1)) + ')', base)
    results = completed_results(cfg, protocol, frames)
    paired = paired_results(cfg, results, frames)
    manifest = prompt_manifest(frames)
    count = sum(len(f['prompts']) for f in manifest)
    lines = [MARKER, '## 补充实验：RGB＋GT二维框提示，WildDet3D微调前后对比', '',
             f'评测使用同一批 **{len(frames):,}张图片、{count:,}个有效目标**；'
             f'benchmark_id：`{protocol["benchmark_id"]}`。', '',
             '### 输入、权重与任务边界', '',
             '- 输入仅为RGB、相机内参和当前目标的GT三维框投影二维框；沿用GT中的裁剪后像素xyxy，'
             '每次调用只传一个框，使用原生 `geometric` 模式。`geometric`是固定模式标记，不含目标类别。',
             '- 官方组加载原checkpoint；微调组在相同基础权重上严格加载本次 `final_head.pt`。'
             '没有重新训练。这测量的是文本流程下的几何微调向框提示流程迁移的效果。',
             '- 不向模型提供GT类别、三维中心、尺寸、旋转、mask或实测深度。'
             '模型仍需自行完成三维估计；GT目标uid只用于模型外的关联、恢复和评分。',
             '- 3D-MOOD当前原生推理入口没有框提示路径，不参与本补充实验。',
             '- GT二维框提供了理想二维定位和目标数量信息，这是已知定位条件下的几何评测；'
             '不等同于RGB＋文本的完整检测任务，不能将两组分数直接当作同输入模型优劣。', '',
             '### 指标与失败处理', '',
             '主要指标为逐目标成功率：预测直接与输入框绑定的目标比较，分母包括全部有效GT目标；'
             '无预测或无效几何返回计失败，设备/OOM等意外错误则中止并保留断点。'
             'success3D判断完整旋转三维IoU，successCenter判断中心欧氏距离。',
             '辅助指标为geometry AP3D与APcenter，沿用原生置信度排序和帧内一对一匹配；'
             'AP可能重新匹配到同帧其他目标，因此与严格绑定目标的成功率分开解释。'
             '没有类别输入，不计算semantic AP。0.7不会过滤本实验的有效预测。',
             '成功率和AP在下表均按0～100展示；平均IoU为0～1，失败按IoU=0计；'
             '中心误差中位数仅统计成功返回三维框的目标，并不代表这些目标通过某个阈值。', '',
             '### 完成状态', '',
             '| 权重 | 状态 |', '| --- | --- |']
    for variant in VARIANTS:
        key = model_id(variant)
        lines.append(f'| {key} | '+('完整评分已完成' if key in results else '尚未完成完整评分，暂无可报告分数')+' |')
    success_names = list(next(iter(results.values()))['conditional']['success']) if results else []
    ap_names = list(next(iter(results.values()))['geometry']['summary']) if results else []
    success_rows, ap_rows = [], []
    if results:
        lines += ['', '### 主要结果：逐目标成功率', '',
                  '| 权重 | success3D@0.25 | success3D@0.50 | successCenter@0.02m | successCenter@0.05m | successCenter@0.10m |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        selected = ('success3D@0.25', 'success3D@0.50', 'successCenter@0.02m', 'successCenter@0.05m', 'successCenter@0.10m')
        for key, result in results.items():
            c = result['conditional'];speed = result['speed']
            success_rows.append([key, c['targets'], c['failures'], c['failure_rate']*100,
                                 *[c['success'][n]*100 for n in success_names], c['mean_iou_including_failures'],
                                 c['median_center_error_m'], speed['mean_predict_seconds_per_target'], speed['fps_predict']])
            ap_rows.append([key, *[result['geometry']['summary'][n]*100 for n in ap_names]])
            cells = [f'{c["success"][n]*100:.4f}' if n in c['success'] else '未配置' for n in selected]
            lines.append('| '+key+' | '+' | '.join(cells)+' |')
        lines += ['', '### 辅助结果：geometry AP', '',
                  '| 权重 | AP3D@0.25 | AP3D@0.50 | APcenter@0.02m | APcenter@0.05m | APcenter@0.10m |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for key, r in results.items():
            summary = r['geometry']['summary']
            cells = [f'{summary[n]*100:.4f}' if n in summary else '未配置' for n in
                     ('AP3D@0.25', 'AP3D@0.50', 'APcenter@0.02m', 'APcenter@0.05m', 'APcenter@0.10m')]
            lines.append('| '+key+' | '+' | '.join(cells)+' |')
        lines += ['', '### 几何误差、失败与速度', '',
                  '| 权重 | 失败/全部目标 | 平均IoU（含失败） | 中心误差中位数(m，返回框) | 秒/目标 | FPS |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        fmt = lambda value: '—' if value is None else f'{value:.4f}'
        for key, r in results.items():
            c=r['conditional'];s=r['speed']
            lines.append(f'| {key} | {c["failures"]}/{c["targets"]} | '+
                         ' | '.join(fmt(v) for v in (c['mean_iou_including_failures'], c['median_center_error_m'],
                                                      s['mean_predict_seconds_per_target'], s['fps_predict']))+' |')
        lines += ['', '计时包含每目标调用的预处理、设备传输、模型推理和导出，包含失败尝试与首次调用开销；'
                  '不含图片读取、模型初始化、恢复校验和CPU评分。多目标图片会逐目标重复调用，'
                  '因此同时报告秒/目标与按全部图片计算的FPS，不能当作单次整图前向速度。', '']
        for key, r in results.items():
            lines.append(f'- {key}失败原因：`{r["conditional"]["failure_reasons"]}`。')
    if paired:
        lines += ['', '### 同目标配对比较', '',
                  '所有目标均按frame_id＋target_id严格配对。提高量为成功率的百分点差；'
                  '“改善”指原来未通过该阈值、微调后通过，“退步”相反。', '',
                  '| 阈值 | 两者通过 | 改善 | 退步 | 两者未通过 | 成功率变化(百分点) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, r in paired.items():
            lines.append(f'| {name} | {r["both_success"]} | {r["improved"]} | {r["regressed"]} | '
                         f'{r["both_failed"]} | {r["delta_percentage_points"]:+.4f} |')
    lines += ['', '### 文件与恢复', '',
              f'- 输入框清单：`{root}/evaluation/gt-box/input_manifest.json`。',
              f'- 逐目标断点：`{root}/evaluation/gt-box/models/<模型-权重>/objects/`；'
              '逐帧完整记录在同级 `predictions/`，包含失败和耗时。',
              f'- 指标与逐目标CSV：`{root}/evaluation/gt-box/full/metrics/`；主成功率表为本报告旁的 `box_comparison.csv`，'
              '辅助AP为 `box_ap_comparison.csv`，配对表为 `box_paired_comparison.csv`。',
              '- 中断后重跑相同框提示命令，校验并复用完整目标；模型、数据或提示变化时拒绝混用缓存。', '']
    if results:
        lines += box_visualizations(cfg, frames, results, directory)
    else:
        lines += ['尚未产生完整框提示评分。当前只合并了实验协议和待完成状态，不把已有文本AP冒充框提示结果。', '']
    lines += [END_MARKER]
    write_csv(directory/'box_comparison.csv', ['model','targets','failures','failure_rate_percent',
              *[n+'_percent' for n in success_names], 'mean_iou_including_failures','median_center_error_m_returned',
              'seconds_per_target','fps'], success_rows)
    write_csv(directory/'box_ap_comparison.csv', ['model', *ap_names], ap_rows)
    write_csv(directory/'box_paired_comparison.csv',
              ['threshold','targets','both_success','improved','regressed','both_failed','delta_percentage_points'],
              [[n, *[v[k] for k in ('targets','both_success','improved','regressed','both_failed','delta_percentage_points')]]
               for n,v in (paired or {}).items()])
    original_results = root/'report/results.json'
    combined = read(original_results) if original_results.exists() else dict(protocol=protocol)
    combined['gt_box'] = dict(complete=len(results)==2, models=results, paired=paired, targets=count,
                              input_manifest_sha256=digest(manifest))
    write(directory/'results.json', combined)
    atomic_bytes(directory/'report.md', (base+'\n\n'+ '\n'.join(lines)+'\n').encode())
    return dict(report=str(directory/'report.md'), text_report_preserved=True,
                box_evaluation_complete=len(results)==2, scored_variants=list(results), targets=count)
