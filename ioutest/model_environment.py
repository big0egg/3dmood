"""Explicit local model caches, shared by training and evaluation."""
import os
from pathlib import Path

PATH_KEYS = {'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'TORCH_HOME', 'MPLCONFIGDIR'}


def normalize_environment(value, base):
    if not isinstance(value, dict) or set(value) - PATH_KEYS:
        raise ValueError('environment 只支持缓存目录：' + ', '.join(sorted(PATH_KEYS)))
    result = {}
    for key, path in value.items():
        if not isinstance(path, str) or not path:
            raise ValueError('缓存目录必须为非空路径：' + key)
        p = Path(path).expanduser()
        result[key] = str((p if p.is_absolute() else Path(base) / p).resolve())
    return result


def child_environment(base, settings):
    result = dict(base)
    # An inherited old cache override must not defeat the explicitly selected HF_HOME.
    if 'HF_HOME' in settings:
        result.pop('TRANSFORMERS_CACHE', None)
        result['HF_HUB_CACHE'] = str(Path(settings['HF_HOME']) / 'hub')
    result.update(settings)
    return result


def apply_environment(settings):
    updated = child_environment(os.environ, settings)
    if 'HF_HOME' in settings and 'TRANSFORMERS_CACHE' not in settings:
        os.environ.pop('TRANSFORMERS_CACHE', None)
    for key in PATH_KEYS:
        if key in updated:
            os.environ[key] = updated[key]


def bert_cache_files(settings, base=None):
    """Verify exactly the cached main revision used by native from_pretrained calls."""
    env = child_environment(os.environ if base is None else base, settings)
    cache = env.get('TRANSFORMERS_CACHE') or env.get('HF_HUB_CACHE') or str(Path(env.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub')
    model = Path(cache) / 'models--bert-base-uncased'
    ref = model / 'refs/main'
    if not ref.is_file():
        raise FileNotFoundError('BERT 本地缓存缺少 refs/main：' + str(model))
    revision = ref.read_text().strip()
    if not revision or Path(revision).name != revision:
        raise ValueError('BERT 缓存 revision 非法')
    snapshot = model / 'snapshots' / revision
    files = {}
    for name in ('config.json', 'tokenizer_config.json'):
        p = snapshot / name
        if not p.is_file() or not p.stat().st_size:
            raise FileNotFoundError('BERT 缓存文件缺失：' + str(p))
        files[name] = str(p)
    for names in (('tokenizer.json', 'vocab.txt'), ('model.safetensors', 'pytorch_model.bin')):
        found = [snapshot / n for n in names if (snapshot / n).is_file() and (snapshot / n).stat().st_size]
        if not found:
            raise FileNotFoundError('BERT 缓存不完整：' + str(snapshot) + '，需要 ' + ' 或 '.join(names))
        for p in found:
            files[p.name] = str(p)
    return files
