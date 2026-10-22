"""Native geometric prompts with paired target records and resumable inference."""
import functools
import time
from pathlib import Path

import numpy as np

from ..common import checksum, digest, inputs_only, lock, read, write
from ..dataset import load, verify

TRACK = 'gt-box'
VARIANTS = ('baseline', 'finetuned')


def model_id(variant):
    if variant not in VARIANTS:
        raise ValueError('variant 必须为 baseline 或 finetuned')
    return 'wilddet3d-' + variant


def prompts_for_frame(frame):
    """Only image-plane boxes cross the inference boundary; labels stay outside."""
    targets = [b for b in read(frame['gt_path']) if not b.get('ignore', False)]
    if len({b['uid'] for b in targets}) != len(targets):
        raise ValueError('GT target uid 重复：' + frame['frame_id'])
    height, width = frame['image_hw']
    prompts = []
    for box in targets:
        rectangle = np.asarray(box['projected_bbox'], dtype=float)
        if (rectangle.shape != (4,) or not np.isfinite(rectangle).all()
                or not 0 <= rectangle[0] < rectangle[2] <= width
                or not 0 <= rectangle[1] < rectangle[3] <= height):
            raise ValueError('无效或未裁剪的GT投影框：' + frame['frame_id'] + ':' + box['uid'])
        prompts.append(dict(target_id=box['uid'], box_xyxy=rectangle.tolist()))
    return prompts


def prompt_manifest(frames):
    return [dict(**inputs_only(f), prompts=prompts_for_frame(f)) for f in frames]


def object_frame(frame, target_id):
    # Hash the uid in filesystem names, without passing it to the predictor.
    return dict(frame, frame_id=frame['frame_id'] + ':' + target_id,
                frame_key=frame['frame_key'] + '-' + digest(target_id)[:24])


def load_box_adapter(cfg, variant):
    import torch
    from ioutest.adapters.wildbox import WildBoxAdapter
    from ..models import load_adapter
    delta = Path(cfg['root']) / 'runs/wilddet3d/checkpoints/final_head.pt' if variant == 'finetuned' else None
    adapter, model, info = load_adapter(cfg, 'wilddet3d', cfg['device'], delta)
    model.eval()
    info = dict(info, input_modalities=['rgb', 'intrinsics', 'supplied_2d_box'],
                prompting='native geometric; one prediction per target; no semantic labels',
                torch=torch.__version__)
    # The tested native box path uses exactly the same loaded wrapper/weights.
    return adapter, info, functools.partial(WildBoxAdapter.predict_object, adapter)


def validate_record(record, frame, classes):
    from ioutest.hot3d.common import export_detection
    expected = [p['target_id'] for p in prompts_for_frame(frame)]
    items = record['items']
    if [i['target_id'] for i in items] != expected:
        raise ValueError('框提示记录缺失、重复或顺序变化：' + frame['frame_id'])
    for item in items:
        if item['status'] not in ('ok', 'failed') or (item['status'] == 'failed') != (item['detection'] is None):
            raise ValueError('框提示记录状态不一致')
        if item['status'] == 'ok':
            export_detection(item['detection'], classes)
            if item['detection']['class_id'] is not None:
                raise ValueError('无类别框提示预测不能包含GT类别')
        if not np.isfinite(item['predict_seconds']) or item['predict_seconds'] < 0:
            raise ValueError('逐目标耗时无效')
    if record['detections'] != [i['detection'] for i in items if i['status'] == 'ok']:
        raise ValueError('帧预测与逐目标记录不一致')


def infer(cfg, variant):
    import torch
    from ioutest.adapters.unipose9d import ObjectFailure
    from ioutest.hot3d.common import bind, commit, export_detection, read_record, record_path, rgb
    from ioutest.suite_io import ObjectInput
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('当前进程不可见PPU；请在DSW原生Terminal运行框提示评测，不回退CPU')
    verify(cfg)
    protocol, _, frames = load(cfg)
    prompts = prompt_manifest(frames)
    root = Path(cfg['root']) / 'evaluation'
    key = model_id(variant)
    directory = root / TRACK / 'models' / key
    with lock(root / '.evaluation.lock'), lock('/mnt/workspace/.hot3d-ppu.lock'):
        adapter, info, predict = load_box_adapter(cfg, variant)
        try:
            sources = [*sorted(Path(__file__).parent.glob('*.py')),
                       Path(__file__).parents[2] / 'run.py']
            # parents[2] is hot3d-finetune; legacy source_code() is nonrecursive.
            from ..common import REPO
            sources += [REPO / 'ioutest/adapters/wildbox.py', REPO / 'ioutest/adapters/wilddet.py',
                        REPO / 'ioutest/suite_io.py', REPO / 'ioutest/hot3d/common.py']
            identity = dict(model=info, model_id=key, track=TRACK, benchmark_id=protocol['benchmark_id'],
                            input_manifest_sha256=digest(prompts), eval_sha256=protocol['eval_sha256'],
                            classes=protocol['classes'], seed=cfg['seed'],
                            cpu_threads=cfg['training']['threads'],
                            device=torch.cuda.get_device_name() if cfg['device'].startswith('cuda') else 'CPU',
                            code={str(p): checksum(p) for p in sources},
                            box_source='valid GT projected_bbox; original virtual pinhole pixel xyxy',
                            no_gt_semantics=True, per_target_prediction=True)
            signature = bind(directory / 'model.json', identity)
            write(root / TRACK / 'input_manifest.json', dict(
                benchmark_id=protocol['benchmark_id'], sha256=digest(prompts), frames=prompts,
                targets=sum(len(f['prompts']) for f in prompts)))
            failures = targets = reused_frames = reused_objects = 0
            for number, (frame, manifest) in enumerate(zip(frames, prompts), 1):
                path = record_path(directory / 'predictions', frame)
                if path.exists():
                    record = read_record(path, frame, signature)
                    validate_record(record, frame, protocol['classes'])
                    reused_frames += 1
                else:
                    image = None
                    items = []
                    for index, proposal in enumerate(manifest['prompts'], 1):
                        target_id = proposal['target_id']
                        object_input = object_frame(frame, target_id)
                        object_path = record_path(directory / 'objects', object_input)
                        write(root / TRACK / 'status.json', dict(stage='inference', model=key,
                              completed_frames=number-1, total_frames=len(frames), frame_id=frame['frame_id'],
                              target_index=index, targets_in_frame=len(manifest['prompts']),
                              targets=targets+len(items), failures=failures))
                        if object_path.exists():
                            item = read_record(object_path, object_input, signature)['item']
                            if item['target_id'] != target_id:
                                raise ValueError('逐目标断点的uid不一致')
                            reused_objects += 1
                        else:
                            if image is None:
                                image = rgb(frame)
                            sample = ObjectInput(image, np.asarray(frame['intrinsics']), tuple(proposal['box_xyxy']))
                            seed = int(digest([cfg['seed'], frame['frame_id'], target_id])[:8], 16)
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            if cfg['device'].startswith('cuda'):
                                torch.cuda.synchronize()
                            tick = time.perf_counter()
                            try:
                                detection, quality = predict(sample, seed)
                                detection = export_detection(detection, protocol['classes'])
                                if detection['class_id'] is not None:
                                    raise ValueError('框提示适配器意外输出语义类别')
                                item = dict(target_id=target_id, status='ok', detection=detection, quality=quality)
                            except ObjectFailure as exc:
                                item = dict(target_id=target_id, status='failed', failure=str(exc), detection=None)
                            if cfg['device'].startswith('cuda'):
                                torch.cuda.synchronize()
                            item.update(seed=seed, predict_seconds=time.perf_counter()-tick)
                            # Device/OOM/unexpected exceptions propagate; never silently become missing predictions.
                            commit(object_path, object_input, signature, item=item)
                        items.append(item)
                    record = dict(items=items, detections=[i['detection'] for i in items if i['status'] == 'ok'])
                    validate_record(record, frame, protocol['classes'])
                    record = commit(path, frame, signature, **record, timing=dict(
                        predict_seconds=sum(i['predict_seconds'] for i in items), frontend_seconds=0., depth_seconds=0.))
                targets += len(record['items'])
                failures += sum(i['status'] == 'failed' for i in record['items'])
                if number % 5 == 0 or number == len(frames):
                    print(f'{key} gt-box {number}/{len(frames)} 帧，{targets}目标，失败{failures}，'
                          f'复用帧{reused_frames}/目标{reused_objects}', flush=True)
            result = dict(stage='inference_complete', complete=True, model=key, track=TRACK,
                          frames=len(frames), targets=targets, failures=failures,
                          reused_frames=reused_frames, reused_objects=reused_objects)
            write(root / TRACK / 'status.json', result)
            return result
        finally:
            adapter.close()


def score(cfg, variant):
    from ioutest.hot3d.common import read_record, record_path
    from ioutest.hot3d.score import score as score_records
    verify(cfg)
    protocol, _, frames = load(cfg)
    root = Path(cfg['root']) / 'evaluation'
    key = model_id(variant)
    directory = root / TRACK / 'models' / key
    with lock(root / '.box-score.lock'):
        binding = read(directory / 'model.json')
        if binding['provenance']['benchmark_id'] != protocol['benchmark_id']:
            raise ValueError('框提示预测来自其他数据协议')
        if binding['provenance']['input_manifest_sha256'] != digest(prompt_manifest(frames)):
            raise ValueError('GT框提示清单变化，拒绝给旧预测评分')
        for frame in frames:
            validate_record(read_record(record_path(directory / 'predictions', frame), frame, binding['signature']),
                            frame, protocol['classes'])
        gt = {f['frame_id']: read(f['gt_path']) for f in frames}
        metrics = root / TRACK / 'full/metrics'
        previous_audit = metrics / (key + '_audit.json')
        if previous_audit.exists():
            audit = read(previous_audit)
            metric_path, target_path = metrics / (key + '.json'), metrics / (key + '_targets.csv')
            if (not metric_path.is_file() or not target_path.is_file()
                    or checksum(metric_path) != audit['metrics_sha256']
                    or checksum(target_path) != audit['targets_sha256']):
                raise ValueError('已审计框提示评分文件缺失或损坏，拒绝将其重新标记为有效')
        result = score_records(cfg, root, TRACK, key, frames, gt, protocol)
        write(metrics / (key + '_audit.json'), dict(benchmark_id=protocol['benchmark_id'],
              signature=binding['signature'], input_manifest_sha256=binding['provenance']['input_manifest_sha256'],
              metrics_sha256=checksum(metrics / (key + '.json')),
              targets_sha256=checksum(metrics / (key + '_targets.csv'))))
        write(root / TRACK / 'status.json', dict(stage='scored', model=key, complete=True,
              frames=result['frames'], targets=result['conditional']['targets'], failures=result['failures']))
        return result
