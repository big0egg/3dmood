"""Reports include only explicitly selected jobs, with complete source identities."""
from pathlib import Path
from .common import atomic_text, digest, read_json, write_json
from .report_io import write_dict_csv

def assemble(root, jobs, data, settings):
    root = Path(root); rows = []; class_rows = []; references = []
    for task in jobs:
        p = Path(task['run_root']) / 'metrics.json'; result = read_json(p)
        checksum = result.pop('checksum', None)
        if checksum != digest(result) or not result['complete'] or result['request_id'] != task['request_id'] or result['metrics_settings'] != settings:
            raise ValueError('指标过期、不完整或来源不一致：' + str(p))
        result['checksum'] = checksum
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
    headings = ['数据集', '模型', '权重', '输入', 'AP3D@0.25', '中心中位数 cm', '关联覆盖率', '延迟 ms', 'P95 ms']
    lines = ['# 三维框评测', '', '| ' + ' | '.join(headings) + ' |', '| ' + ' | '.join(['---'] * len(headings)) + ' |']
    def fmt(v): return '—' if v is None else f'{v:.4f}' if isinstance(v, float) else str(v)
    for r in rows:
        values = [r['dataset'], r['model_name'], r['variant_name'], r['track'], r.get('AP3D@0.25'),
            r['center_median_cm'], r['association_recall'], r['latency_mean_ms'], r['latency_p95_ms']]
        lines.append('| ' + ' | '.join(fmt(v) for v in values) + ' |')
    lines += ['', 'AP／比例为 0～1。几何误差只统计关联成功目标，覆盖率和全目标达标率同时保留。',
              'TEXT 诊断按三维框投影的二维 IoU 关联；GT 二维框组严格绑定输入目标，不能混合排名。',
              '方向评价框几何方向，并非物体语义朝向。实测延迟缺失显示 —，阶段耗时之和不代替实测值。', '']
    for name in dict.fromkeys(t['dataset'] for t in jobs):
        lines.append(f"数据 {name}：{data[name]['dataset_id']}；标签状态：{data[name]['metadata'].get('label_status')}。")
    atomic_text(directory / 'report.md', '\n'.join(lines) + '\n')
    return dict(complete=True, jobs=len(jobs), report=str(directory / 'report.md'))

def _scale(value, factor):
    return None if value is None else value * factor
