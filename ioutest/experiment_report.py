"""Reports include only explicitly selected jobs, with complete source identities."""
from pathlib import Path
import os
from .common import atomic_text, digest, read_json, write_json
from .report_io import write_dict_csv

def assemble(root, jobs, data, settings):
    root = Path(root); rows = []; class_rows = []; references = []; details = []
    for task in jobs:
        p = Path(task['run_root']) / 'metrics.json'; result = read_json(p)
        checksum = result.pop('checksum', None)
        if checksum != digest(result) or not result['complete'] or result['request_id'] != task['request_id'] or result['metrics_settings'] != settings:
            raise ValueError('指标过期、不完整或来源不一致：' + str(p))
        result['checksum'] = checksum
        details.append((task, result))
        d = result['diagnostics']; latency = result['latency']
        row = {k: task[k] for k in ('dataset', 'model_name', 'variant_name', 'track')}
        row.update(frames=result['frames'], targets=d['targets'], associated_targets=d['matched'],
            association_recall=d['association_recall'], missed_rate=d['missed_rate'],
            center_median_cm=_scale(d['errors']['center_error_m']['median'], 100),
            center_p90_cm=_scale(d['errors']['center_error_m']['p90'], 100),
            rotation_median_deg=d['errors']['rotation_error_deg']['median'],
            latency_mean_ms=latency['mean_ms'], latency_p95_ms=latency['p95_ms'], fps=latency['fps'])
        for axis in 'xyz':
            row['size_' + axis + '_median_cm'] = _scale(d['errors']['size_abs_xyz_m'][axis]['median'], 100)
            row['size_' + axis + '_median_percent'] = _scale(d['errors']['size_relative_xyz'][axis]['median'], 100)
        row.update(result['geometry']['summary'])
        if result['semantic']: row.update({'semantic_' + k: v for k, v in result['semantic']['summary'].items()})
        if result['conditional']:
            row.update(result['conditional']['success']); row['output_failure_rate'] = result['conditional']['failure_rate']
        for name, v in result['workpoint']['per_class'][0]['values'].items():
            for field in ('precision', 'recall', 'tp', 'fp'): row['workpoint_' + name + '_' + field] = v[field]
        if result['detection']['mode'] == 'autonomous_geometry_detection':
            for name, values in result['detection']['criteria'].items():
                row['workpoint_' + name + '_fn'] = values['fn']
                row['workpoint_' + name + '_missed_rate'] = values['missed_rate']
        rows.append(row)
        for mode in ('geometry', 'semantic'):
            if result[mode]:
                for c in result[mode]['per_class']:
                    class_rows.append(dict(**{k: task[k] for k in ('dataset', 'model_name', 'variant_name', 'track')},
                        mode=mode, class_name=c['class_name'], targets=c['valid_gt'], **{k: v['ap'] for k, v in c['values'].items()}))
        references.append(dict(job={k: task[k] for k in ('dataset', 'model_name', 'variant_name', 'track')}, metrics=str(p), checksum=checksum))
    directory = root / 'reports'
    write_dict_csv(directory / 'comparison.csv', rows); write_dict_csv(directory / 'per_class.csv', class_rows)
    write_json(directory / 'results.json', dict(complete=True, jobs=references, comparison=rows))
    headings = ['数据集', '模型', '权重', '输入', 'AP3D@0.25', 'AP3D@0.50', '中心中位数 cm', '关联／输出覆盖率', '延迟 ms', 'P95 ms']
    lines = ['# 三维框评测', '', '| ' + ' | '.join(headings) + ' |', '| ' + ' | '.join(['---'] * len(headings)) + ' |']
    for r in rows:
        values = [r['dataset'], r['model_name'], r['variant_name'], r['track'], r.get('AP3D@0.25'), r.get('AP3D@0.50'),
            r['center_median_cm'], r['association_recall'], r['latency_mean_ms'], r['latency_p95_ms']]
        lines.append('| ' + ' | '.join(_fmt(v) for v in values) + ' |')
    lines += ['', 'AP 和概览覆盖率为 0～1；详细表中标注 % 的列按百分比显示。几何误差只统计关联成功目标，必须结合覆盖率和全目标达标率阅读。',
              'TEXT 诊断按三维框投影的二维 IoU 关联；GT 二维框组严格绑定输入目标，不能混合排名。',
              '方向评价框几何方向，并非物体语义朝向。实测延迟缺失显示 —，阶段耗时之和不代替实测值。', '']
    for task, result in details:
        lines.extend(_detail(task, result, directory))
    lines += ['## 数据来源', '']
    for name in dict.fromkeys(t['dataset'] for t in jobs):
        lines.append(f"数据 {name}：{data[name]['dataset_id']}；标签状态：{data[name]['metadata'].get('label_status')}。")
    atomic_text(directory / 'report.md', '\n'.join(lines) + '\n')
    return dict(complete=True, jobs=len(jobs), report=str(directory / 'report.md'))

def _scale(value, factor):
    return None if value is None else value * factor


def _fmt(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _table(headings, rows):
    return ['', '| ' + ' | '.join(headings) + ' |',
            '| ' + ' | '.join(['---'] * len(headings)) + ' |',
            *['| ' + ' | '.join(_fmt(v) for v in row) + ' |' for row in rows], '']


def _stats(named, unit, factor=1):
    rows = [[name, values['count'], *[_scale(values[k], factor) for k in ('mean', 'median', 'p90')]]
            for name, values in named]
    return _table(['指标', '有效目标数', f'均值 {unit}', f'中位数 {unit}', f'P90 {unit}'], rows)


def _detail(task, result, report_dir):
    d = result['diagnostics']; errors = d['errors']; latency = result['latency']
    title = ' / '.join(_fmt(task[k]) for k in ('dataset', 'model_name', 'variant_name', 'track'))
    lines = [f'## {title}', '', f"评测帧数：{result['frames']}；有效 GT 目标数：{d['targets']}；几何误差统计目标数：{d['matched']}。", '',
             '### AP', '']
    geometry = result['geometry']['summary']
    semantic = result['semantic']['summary'] if result['semantic'] else {}
    lines += _table(['指标', '几何 AP', '类别与几何联合 AP'],
                    [[name, value, semantic.get(name)] for name, value in geometry.items()])
    if task['track'] == 'gt-box':
        lines += ['GT-BOX 的 AP 是给定二维目标条件下的辅助指标；主要结合下方目标绑定的定位达标率阅读，不能与 TEXT 自主检测混合排名。', '']
    lines += ['### 中心误差', '']
    lines += _stats([('三维中心距离', errors['center_error_m']),
                     *[(f'相机 {axis.upper()} 轴绝对误差', errors['center_abs_xyz_m'][axis]) for axis in 'xyz']], 'cm', 100)
    lines += ['### 尺寸误差', '', '尺寸轴使用评分算法选择的等价框表示，与方向误差采用同一轴对齐。', '', '绝对尺寸误差：']
    lines += _stats([(axis.upper(), errors['size_abs_xyz_m'][axis]) for axis in 'xyz'], 'cm', 100)
    lines += ['相对尺寸误差：']
    lines += _stats([(axis.upper(), errors['size_relative_xyz'][axis]) for axis in 'xyz'], '%', 100)
    lines += ['### 方向误差', '']
    lines += _stats([('框几何旋转误差', errors['rotation_error_deg'])], '°')
    if task['track'] == 'gt-box':
        lines += ['### 条件定位与输出失败', '',
                  '输入已给定 GT 二维框。输出覆盖率表示是否返回有效框，不代表自主检测召回率，也不代表三维定位准确率。']
        lines += _table(['指标', '值'], [
            ['给定目标数', d['targets']], ['有效输出数', d['matched']], ['缺失输出数', d['missed']],
            ['输出覆盖率 %', _scale(d['association_recall'], 100)], ['输出失败率 %', _scale(d['missed_rate'], 100)]])
        conditional = result.get('conditional')
        if conditional:
            lines += _table(['补充统计', '值'], [['平均 IoU（包含失败项）', conditional['mean_iou_including_failures']]])
            if conditional.get('failure_reasons'):
                lines += _table(['失败原因', '数量'], list(conditional['failure_reasons'].items()))
    else:
        lines += ['### 检出、漏检与关联覆盖', '',
                  f"诊断工作点 score_floor={d['score_floor']}；投影二维 IoU 关联阈值={d['association_iou']}。关联覆盖与下方按三维标准计算的检出率分别统计。"]
        lines += _table(['诊断统计', '值'], [
            ['有效 GT 目标数', d['targets']], ['关联目标数', d['matched']], ['未关联目标数', d['missed']],
            ['关联覆盖率 %', _scale(d['association_recall'], 100)], ['未关联目标比例 %', _scale(d['missed_rate'], 100)],
            ['未关联预测数', d['unmatched_predictions']], ['忽略区域预测数', d['ignored_predictions']]])
        criteria = result['detection']['criteria']
        lines += _table(['检出标准', 'TP', 'FP', 'FN', 'Precision %', 'Recall %', '漏检率 %'], [
            [name, values['tp'], values['fp'], values['fn'],
             _scale(values['precision'], 100), _scale(values['recall'], 100), _scale(values['missed_rate'], 100)]
            for name, values in criteria.items()])
        sd = result.get('semantic_diagnostics')
        if sd:
            lines += _table(['类别与几何联合诊断', '值'], [
                ['关联目标数', sd['matched']], ['关联覆盖率 %', _scale(sd['association_recall'], 100)],
                ['未关联目标比例 %', _scale(sd['missed_rate'], 100)]])
    lines += ['全目标定位达标率（分母包含缺失／未关联目标）：']
    lines += _table(['定位标准', '达标目标数', '总目标数', '达标率 %'], [
        [name, values['passed'], values['targets'], _scale(values['rate'], 100)]
        for name, values in d['localization_success'].items()])
    lines += ['### 感知流程延迟', '']
    lines += _table(['指标', '值'], [
        ['测量状态', latency['status']], ['实测帧数', latency['measured_frames']],
        ['平均延迟 ms', latency['mean_ms']], ['P95 延迟 ms', latency['p95_ms']], ['FPS', latency['fps']],
        ['初始化耗时 s', latency.get('initialization_seconds')], ['预热耗时 s', latency.get('warmup_seconds')],
        ['预热帧数（pass）', latency.get('warmup_passes')], ['复用预测帧数', latency.get('reused_records')],
        ['计时来源', latency.get('source')], ['阶段耗时之和的均值 ms（参考）', latency.get('stage_sum_mean_ms')]])
    lines += ['计时口径：已解码的内存图片到 CPU 数值框，包含必要进程通信；不含图片读取／解码、模型初始化、预热、结果写盘和评分。',
              '复用预测时保留原测量；阶段耗时之和不代替实测延迟。']
    if task['track'] == 'gt-box':
        lines += ['GT-BOX 延迟不包含外部检测器获取二维框的耗时。']
    def link(name):
        path = Path(task['run_root']) / name
        return f'[{name}]({Path(os.path.relpath(path, report_dir)).as_posix()})'
    lines += ['', '逐类 AP、逐目标误差及原始统计：' + '；'.join(link(name) for name in ('classes.csv', 'targets.csv', 'metrics.json')) + '。', '']
    return lines
