"""Single-device FP32 training with resumes at committed optimizer boundaries."""
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from .common import atomic_bytes, bind, checksum, data_path, digest, lock, source_code, write
from .dataset import load, verify
from .models import (MoodTrainer, WildTrainer, freeze, head_state, load_adapter,
                     training_image)
from .wandb_log import Tracker, start_tracking


def rng_state():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def lr_factor(update, total, warmup):
    if update < warmup:
        return .1 + .9 * update / max(1, warmup)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (update - warmup) / max(1, total - warmup)))


def train(cfg, key):
    import torch
    options = cfg['training']
    if options['micro_batch'] != 1:
        raise ValueError('首版支持 micro_batch=1；使用 accumulate 调整有效批量')
    for name in ('epochs', 'accumulate', 'checkpoint_updates', 'threads'):
        if type(options[name]) != int or options[name] < 1:
            raise ValueError(f'{name} 必须为正整数')
    verify(cfg)
    protocol, frames, _ = load(cfg)
    directory = Path(cfg['root']) / 'runs' / key
    with lock(directory / '.training.lock'), lock('/mnt/workspace/.hot3d-ppu.lock'):
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        adapter, model, info = load_adapter(cfg, key, cfg['device'])
        identity = dict(model_id=key, model=info, seed=cfg['seed'], training=options,
                        train_sha256=protocol['train_sha256'], benchmark_id=protocol['benchmark_id'],
                        code=source_code(), loss='native text matching + final-layer positive 3D regression only',
                        device=torch.cuda.get_device_name(), torch=torch.__version__,
                        train_uses_eval=False, depth_gt=False, geometry_prompts=False)
        signature = bind(directory / 'training.json', identity)
        trainer = (MoodTrainer if key == '3dmood' else WildTrainer)(adapter, model, protocol['classes'])
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError('没有可训练参数')
        write(directory / 'parameters.json', dict(
            total=sum(p.numel() for p in model.parameters()), trainable=sum(p.numel() for p in parameters),
            names=[n for n, p in model.named_parameters() if p.requires_grad]))
        optimizer = torch.optim.AdamW(parameters, lr=options['lr'], weight_decay=options['weight_decay'])
        total_updates = options['epochs'] * math.ceil(len(frames) / options['accumulate'])
        epoch = next_index = update = 0
        last = directory / 'checkpoints/last.pt'
        if last.exists():
            state = torch.load(last, map_location='cpu', weights_only=False)
            if state['signature'] != signature:
                raise ValueError('恢复点与数据、源码、模型或训练参数不一致')
            model.bbox3d_head.load_state_dict(state['head'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            # Optimizer buffers must follow their parameters to the PPU.
            for opt_state in optimizer.state.values():
                for name, value in opt_state.items():
                    if isinstance(value, torch.Tensor) and name != 'step':
                        opt_state[name] = value.to(cfg['device'])
            epoch, next_index, update = state['epoch'], state['next_index'], state['update']
            restore_rng(state['rng'])
            journal = directory / 'train.jsonl'
            if journal.exists():
                # An abrupt kill can leave logs newer than the last durable
                # checkpoint. Replay those samples and keep one row per update.
                committed = {}
                for line in journal.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row['update'] <= update:
                        committed[row['update']] = row
                atomic_bytes(journal, ''.join(json.dumps(committed[u])+'\n' for u in sorted(committed)).encode())
            print(f'{key} 恢复：已完成 {epoch} 轮，下一帧位置 {next_index}，更新 {update}/{total_updates}', flush=True)
            final = directory / 'checkpoints/final_head.pt'
            if epoch == options['epochs'] and final.exists():
                result = torch.load(final, map_location='cpu', weights_only=False)
                if result['identity'] != signature or any(
                    not torch.equal(value, result['head'][name]) for name, value in head_state(model).items()
                ):
                    raise ValueError('最终权重与完整恢复点不一致')
                adapter.close()
                result = dict(stage='complete', complete=True, reused=True, model=key, final_head=str(final), updates=update)
                write(directory / 'status.json', result)
                return result
        boundary_rng = rng_state()

        def checkpoint():
            save_torch(last, dict(signature=signature, head=head_state(model), optimizer=optimizer.state_dict(),
                                 epoch=epoch, next_index=next_index, update=update, rng=boundary_rng,
                                 scheduler=dict(total=total_updates, warmup=options['warmup_updates'], next_update=update)))

        started = time.perf_counter()
        checked_gradients = False
        optimizer_transaction = False
        tracker, outcome = Tracker(), 'failed'
        try:
            tracker = start_tracking(cfg, key, directory, signature, len(frames))
            while epoch < options['epochs']:
                order = list(range(len(frames)))
                random.Random(cfg['seed'] + epoch).shuffle(order)
                freeze(model, wild=key == 'wilddet3d')
                while next_index < len(order):
                    group = order[next_index:next_index + options['accumulate']]
                    optimizer.zero_grad(set_to_none=True)
                    losses_sum = {}
                    factor = lr_factor(update, total_updates, options['warmup_updates'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = options['lr'] * factor
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    for position in group:
                        frame = frames[position]
                        write(directory / 'status.json', dict(stage='training', epoch=epoch + 1,
                              update=update, total_updates=total_updates, next_index=next_index,
                              frame_id=frame['frame_id']))
                        image = training_image(frame, epoch, cfg['seed'])
                        losses = trainer(frame, image)
                        loss = sum(losses.values()) / len(group)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'非有限损失：{frame["frame_id"]}')
                        loss.backward()
                        for name, value in losses.items():
                            losses_sum[name] = losses_sum.get(name, 0.) + float(value.detach()) / len(group)
                    if not checked_gradients:
                        if not any(p.grad is not None and p.grad.abs().max().item() > 0 for p in parameters):
                            raise RuntimeError('三维头没有非零梯度')
                        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                            raise RuntimeError('冻结参数意外产生梯度')
                        checked_gradients = True
                    norm = torch.nn.utils.clip_grad_norm_(parameters, options['clip_grad'], error_if_nonfinite=True)
                    optimizer_transaction = True
                    optimizer.step()
                    next_index += len(group)
                    update += 1
                    boundary_rng = rng_state()
                    optimizer_transaction = False
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - tick
                    row = dict(update=update, epoch=epoch + 1, images_completed=next_index,
                               lr=optimizer.param_groups[0]['lr'], grad_norm=float(norm),
                               seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(), **losses_sum)
                    with (directory / 'train.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                    if update % 10 == 0:
                        print(f'{key} 第 {epoch+1}/{options["epochs"]} 轮 {next_index}/{len(frames)} 帧，更新 {update}/{total_updates}，'
                              f'loss={sum(losses_sum.values()):.4f}，{seconds:.2f}秒/更新', flush=True)
                    if update % options['checkpoint_updates'] == 0:
                        checkpoint()
                epoch += 1
                next_index = 0
                boundary_rng = rng_state()
                checkpoint()
            final = directory / 'checkpoints/final_head.pt'
            save_torch(final, dict(model_id=key, base_sha256=info['checkpoint_sha256'], head=head_state(model),
                                   identity=signature, benchmark_id=protocol['benchmark_id'], epochs_completed=epoch))
            result = dict(stage='complete', model=key, epochs=epoch, updates=update,
                          final_head=str(final), final_head_sha256=checksum(final),
                          invocation_seconds=time.perf_counter()-started,
                          checkpoint_selection=f'fixed final epoch {epoch}; evaluation not consulted')
            write(directory / 'status.json', result)
            outcome = 'complete'
            return result
        except BaseException as exc:
            # No optimizer update occurs mid accumulation; discard partial
            # gradients on restart and restore RNG at the preceding boundary.
            checkpoint_error = None
            try:
                if not optimizer_transaction:
                    checkpoint()
                else:
                    checkpoint_error = '参数更新被打断，保留此前持久恢复点，拒绝保存可能只更新一部分参数的状态'
            except BaseException as save_error:
                checkpoint_error = str(save_error)
            write(directory / 'status.json', dict(stage='failed', error=str(exc), epoch=epoch,
                  next_index=next_index, update=update, resumable=last.exists(), checkpoint_error=checkpoint_error))
            raise
        finally:
            tracker.finish(outcome)
            adapter.close()
