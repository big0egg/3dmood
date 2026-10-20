"""Text-only predictions, existing AP implementation, and complete reports."""
import csv
import time
from pathlib import Path

import numpy as np

from .common import (checksum, data_path, digest, inputs_only, lock, read, readlines,
                     source_code, write, atomic_bytes)
from .dataset import load, verify
from .models import load_adapter


def evaluate(cfg, key, variant):
    import torch
    from ioutest.hot3d.common import bind, commit, export_detection, read_record, record_path, rgb
    from ioutest.hot3d.score import score
    from ioutest.schema import ModelInput
    verify(cfg)
    protocol, _, frames = load(cfg)
    root = Path(cfg['root']) / 'evaluation'
    model_id = key + '-' + variant
    directory = root / 'text/models' / model_id
    delta = Path(cfg['root']) / 'runs' / key / 'checkpoints/final_head.pt' if variant == 'finetuned' else None
    with lock(root / '.evaluation.lock'), lock('/mnt/workspace/.hot3d-ppu.lock'):
        adapter, model, info = load_adapter(cfg, key, cfg['device'], delta)
        model.eval()
        identity = dict(model=info, model_id=model_id, benchmark_id=protocol['benchmark_id'],
                        inputs=digest([inputs_only(f) for f in frames]), classes=protocol['classes'],
                        code=source_code(), device=torch.cuda.get_device_name(), config=cfg)
        signature = bind(directory / 'model.json', identity)
        try:
            for number, frame in enumerate(frames, 1):
                path = record_path(directory / 'predictions', frame)
                if path.exists():
                    read_record(path, frame, signature)
                else:
                    # The adapter receives no target count, class truth, mask,
                    # GT positions, CAD model, world pose or measured depth.
                    sample = ModelInput(frame['frame_id'], rgb(frame), np.asarray(frame['intrinsics']),
                                        class_names=tuple(c['name'] for c in protocol['classes']),
                                        prompts=tuple(c['prompt'] for c in protocol['classes']))
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    detections = [export_detection(b, protocol['classes']) for b in adapter.predict(sample)]
                    torch.cuda.synchronize()
                    commit(path, frame, signature, items=[], detections=detections,
                           timing=dict(predict_seconds=time.perf_counter()-tick, frontend_seconds=0., depth_seconds=0.))
                if number % 25 == 0 or number == len(frames):
                    print(f'{model_id} {number}/{len(frames)} 帧', flush=True)
                    write(root / 'status.json', dict(stage='inference', model=model_id, completed=number, total=len(frames)))
        finally:
            adapter.close()
    gt = {frame['frame_id']: read(frame['gt_path']) for frame in frames}
    write(root / 'text/full/selection.json', dict(frame_ids=[f['frame_id'] for f in frames],
                                                benchmark_id=protocol['benchmark_id']))
    return score(cfg, root, 'text', model_id, frames, gt, protocol)


def report(cfg):
    protocol, train, frames = load(cfg)
    root = Path(cfg['root'])
    directory = root / 'report'
    directory.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for key in cfg['models']:
        for variant in ('baseline', 'finetuned'):
            model_id = key + '-' + variant
            path = root / 'evaluation/text/full/metrics' / (model_id + '.json')
            if not path.exists():
                raise FileNotFoundError(f'尚未完成评分：{path}')
            result = read(path)
            if not result['complete'] or result['identity']['benchmark_id'] != protocol['benchmark_id']:
                raise ValueError('评分不完整或不是当前数据集')
            metrics[model_id] = result
    headings = ['model', 'mode', 'AP3D@0.25', 'AP3D@0.50', 'APcenter@0.02m',
                'APcenter@0.05m', 'APcenter@0.10m', 'APcenter@0.25m', 'APcenter@0.50m',
                'APcenter@1.00m', 'fps']
    records = []
    for model_id, result in metrics.items():
        for mode in ('geometry', 'semantic'):
            records.append([model_id, mode, *[result[mode]['summary'][name]*100 for name in headings[2:-1]],
                            result['speed']['fps_pipeline']])
    with (directory / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(headings); writer.writerows(records)
    write(directory / 'results.json', dict(protocol=protocol, models=metrics, audit=read(data_path(cfg)/'dataset_audit.json')))
    lines = ['# HOT3D：三维头微调前后对比', '',
             f'训练 {len(train):,} 帧，评测 {len(frames):,} 帧。benchmark_id：`{protocol["benchmark_id"]}`。', '',
             '## 数据来源和划分', '',
             '训练：'+'、'.join(cfg['train_participants'])+' 的原始 Aria RGB；'
             '按人工 QA、曝光、相机/物体位姿、物体可见性筛选，固定虚拟针孔相机 768×768、90°视场。',
             '评测：'+'、'.join(f'{p}（{n:,}帧）' for p,n in cfg['eval_quotas'].items())+'，复用原 benchmark 图片、GT 和 ignore 标记。'
             'P0012 不参加本次实验。训练和评测按参与者隔离，图片摘要无交集。', '',
             '两组均为同批33个物体、25个语义类别。该实验衡量跨参与者适应效果，'
             '不是官方隐藏测试集或全新物体泛化测试；此前已进行过全量探索性评测。', '',
             '## 文件及用途', '',
             f'- 数据协议与审计：`{data_path(cfg)}/protocol.json`、`dataset_audit.json`。',
             '- `train.jsonl` / `eval.jsonl`：图片、内参、时间戳、参与者、GT 文件与摘要。',
             '- `groundtruth/`：米制相机坐标中心、尺寸、完整旋转、类别和忽略原因。',
             '- `calibration/`：新增训练帧的原始和虚拟相机变换；原评测标定位于旧数据目录。',
             f'- 逐帧预测：`{root}/evaluation/text/models/<model>/predictions/`。',
             f'- 训练日志与恢复点：`{root}/runs/<model>/`。', '',
             '## 训练与公平性', '',
             '从各自原权重开始，只更新三维头；使用模型原生匹配与最终层正样本三维回归损失。'
             '关闭分类、presence、三维置信度负样本损失及实测深度图辅助损失。'
             f'预设训练{cfg["training"]["epochs"]}轮，以最后一轮作为正式权重，不使用评测集选轮次或阈值。',
             '每张评测图片提供相同完整25类文本词表、RGB和内参；不提供GT目标数、GT位置、'
             '物体模型、GT掩码或深度。模型结构、预训练数据和原生NMS不同，不声称这些因素相同。',
             'GT仅覆盖建模物体，背景同类物体可能被计为FP；完整三维框对细小物体的几厘米误差敏感。', '',
             '## 指标', '',
             'AP按0～100展示，CSV同样按0～100。AP3D使用完整旋转框IoU匹配；'
             'APcenter使用三维中心欧氏距离匹配。geometry忽略类别，semantic要求类别一致并按类别平均。'
             '所有AP使用保留预测的完整置信度排序，0.7仅作为可视化工作点。', '',
             '| 模型 | 统计 | AP3D@0.25 | AP3D@0.50 | APcenter@0.02m | APcenter@0.05m | APcenter@0.10m | FPS |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        lines.append('| ' + ' | '.join([row[0], row[1], *[f'{v:.4f}' for v in row[2:7]], f'{row[-1]:.3f}']) + ' |')
    lines += ['', '## 速度、训练过程与失败', '',
              '推理FPS包含预处理、PPU传输和数值导出，设备同步计时；不含模型初始化、图片读写和CPU评分，'
              '保留首次推理开销。两个模型顺序运行。',
              '训练日志 train.jsonl 含四类损失、学习率、梯度范数、每次更新时间和显存峰值。'
              '异常会中止并保留最近完成的参数更新；不会静默跳过失败训练帧。', '',
              '## 可视化', '',
              '固定按评测清单均匀选择12帧：绿框GT、红框预测；预测展示阈值0.7。'
              '这是固定示例，不作为代表性随机抽样或召回率证据。', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines += ['## 训练曲线', '']
    for key in cfg['models']:
        journal = root / 'runs' / key / 'train.jsonl'
        if not journal.exists():
            raise FileNotFoundError(f'正式报告缺少训练记录：{journal}')
        history = readlines(journal)
        updates = [row['update'] for row in history]
        expected = cfg['training']['epochs'] * int(np.ceil(len(train)/cfg['training']['accumulate']))
        if updates != list(range(1, expected+1)):
            raise ValueError(f'{key} 训练日志不完整或存在重复更新')
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        window = min(50, len(history))
        for name in ('loss_delta_2d', 'loss_depth', 'loss_dim', 'loss_rot'):
            values = np.array([row[name] for row in history])
            mean = np.convolve(values, np.ones(window)/window, mode='valid')
            axes[0].plot(updates[window-1:], mean, label=name)
        axes[0].set_ylabel('Training loss (rolling mean)')
        axes[0].legend()
        axes[1].plot(updates, [row['lr'] for row in history])
        axes[1].set_ylabel('Learning rate'); axes[1].set_xlabel('Optimizer updates')
        figure.suptitle(key + f': {cfg["training"]["epochs"]}-epoch training')
        figure.tight_layout()
        relative = key + '-training.png'
        figure.savefig(directory/relative, dpi=140)
        plt.close(figure)
        lines += [f'![{key}训练损失与学习率]({relative})', '',
                  f'{key}：{len(history):,}次更新，记录的更新总时间 '
                  f'{sum(row["seconds"] for row in history)/3600:.2f}小时；'
                  f'峰值张量显存 {max(row["peak_allocated_bytes"] for row in history)/1024**3:.2f}GiB。'
                  '该时间不含模型初始化、恢复点保存和已丢弃重算的工作。', '']
        from .wandb_log import statuses
        attempts = statuses(cfg, key)[key]
        if attempts:
            latest = attempts[-1]
            lines += [f'W&B：模式 `{latest.get("mode", "unknown")}`，状态 `{latest["stage"]}`；'
                      f'状态文件 `{latest["path"]}`。', '']
            if latest.get('url'):
                lines += [f'[打开{key}的W&B训练曲线]({latest["url"]})', '']
    lines += ['## 固定帧对比图', '']
    from ioutest.hot3d.common import read_record, record_path, rgb
    from ioutest.visualize import _draw
    from PIL import Image, ImageDraw
    import io
    for index in np.linspace(0, len(frames)-1, min(12, len(frames)), dtype=int):
        frame = frames[int(index)]
        gt = read(frame['gt_path'])
        for model_id in metrics:
            source = root / 'evaluation/text/models' / model_id
            signature = read(source/'model.json')['signature']
            predictions = read_record(record_path(source/'predictions', frame), frame, signature)['detections']
            picture = Image.fromarray(rgb(frame)); draw = ImageDraw.Draw(picture)
            for box in gt:
                if not box['ignore']:
                    _draw(draw, box, np.asarray(frame['intrinsics']), picture.width, picture.height, 'lime', 2, 'GT')
            for box in predictions:
                if box['score'] >= cfg['display_score']:
                    _draw(draw, box, np.asarray(frame['intrinsics']), picture.width, picture.height, 'red', 2, f'{box["score"]:.2f}')
            relative = Path('visualizations') / model_id / f'{index:04d}.png'
            stream = io.BytesIO(); picture.save(stream, format='PNG')
            atomic_bytes(directory/relative, stream.getvalue())
            lines.append(f'- {model_id}，{frame["frame_id"]}：[查看]({relative})')
    atomic_bytes(directory/'report.md', ('\n'.join(lines)+'\n').encode())
    return dict(complete=True, report=str(directory/'report.md'))
