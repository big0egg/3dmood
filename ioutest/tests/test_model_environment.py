"""Explicit cache paths override inherited shell settings and bind inference identity."""
from pathlib import Path
import pytest
from ioutest.model_environment import bert_cache_files, child_environment, normalize_environment
from ioutest.experiment import make_tasks
from ioutest.experiment_config import select
from ioutest.experiment_data import load_dataset
from .test_experiment import prepared


def cache_at(root):
    model = root / 'hub/models--bert-base-uncased'
    (model / 'refs').mkdir(parents=True)
    (model / 'refs/main').write_text('fixture-revision')
    snapshot = model / 'snapshots/fixture-revision'
    snapshot.mkdir(parents=True)
    for name in ('config.json', 'tokenizer_config.json', 'vocab.txt', 'model.safetensors'):
        (snapshot / name).write_bytes(b'fixture-cache-file')
    return snapshot


def test_explicit_hf_home_replaces_old_shell_cache(tmp_path):
    settings = normalize_environment({'HF_HOME': 'cache'}, tmp_path)
    inherited = {'HF_HOME': '/old', 'HF_HUB_CACHE': '/old/hub', 'TRANSFORMERS_CACHE': '/old/tokenizer'}
    env = child_environment(inherited, settings)
    assert env['HF_HOME'] == str(tmp_path / 'cache')
    assert env['HF_HUB_CACHE'] == str(tmp_path / 'cache/hub')
    assert 'TRANSFORMERS_CACHE' not in env and inherited['HF_HOME'] == '/old'


def test_cache_validation_is_local_and_requires_weights(tmp_path):
    home = tmp_path / 'cache'; snapshot = cache_at(home)
    files = bert_cache_files({'HF_HOME': str(home)}, base={})
    assert 'model.safetensors' in files and 'vocab.txt' in files
    (snapshot / 'model.safetensors').unlink()
    with pytest.raises(FileNotFoundError, match='不完整'):
        bert_cache_files({'HF_HOME': str(home)}, base={})
    assert not (snapshot / 'model.safetensors').exists()


def test_cache_content_and_environment_bind_request(prepared, tmp_path):
    _, config = prepared; home = tmp_path / 'cache'; snapshot = cache_at(home)
    config['models']['3dmood']['environment'] = {'HF_HOME': str(home)}
    data = {'test': load_dataset(config['datasets']['test'])}
    first = make_tasks(config, select(config), data)[0]
    assert first['environment']['HF_HOME'] == str(home)
    (snapshot / 'model.safetensors').write_bytes(b'changed-cache-weights')
    second = make_tasks(config, select(config), data)[0]
    assert first['request_id'] != second['request_id']
