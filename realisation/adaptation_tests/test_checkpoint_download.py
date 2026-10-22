"""Exercise the patched upstream download classes without network/CUDA imports."""
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

from robot_realism.common import PROJECT


def test_hub_api_preserves_revision_patterns_and_subdirectory(tmp_path, monkeypatch):
    flags = ModuleType('cosmos_transfer2._src.imaginaire.flags')
    flags.INTERNAL = flags.EXPERIMENTAL_CHECKPOINTS = False
    utils = ModuleType('cosmos_transfer2._src.imaginaire.utils')
    utils.log = SimpleNamespace(info=lambda *args: None)
    monkeypatch.setitem(sys.modules, flags.__name__, flags)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    path = PROJECT / 'cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py'
    spec = importlib.util.spec_from_file_location('realisation_test_checkpoint_db', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import huggingface_hub
    calls = []
    weight = tmp_path / 'test_weight.pt'
    weight.write_bytes(b'UNIT TEST ONLY, NOT MODEL WEIGHTS')
    subdir = tmp_path / 'encoder'
    subdir.mkdir()
    def file_download(**kwargs):
        calls.append(kwargs)
        return str(weight)
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', file_download)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot_download)
    file = module.CheckpointFileHf(repository='test/model', revision='fixed-commit', filename='weights.pt')
    assert file.download() == str(weight)
    assert file.download() == str(weight)  # Cached; one API call.
    assert calls == [{'repo_id': 'test/model', 'repo_type': 'model', 'revision': 'fixed-commit', 'filename': 'weights.pt'}]
    folder = module.CheckpointDirHf(repository='test/model', revision='fixed-commit',
                                    subdirectory='encoder', include=('*.json', '*.bin'), exclude=('large*',))
    assert folder.download() == str(subdir)
    assert calls[-1] == {'repo_id': 'test/model', 'repo_type': 'model', 'revision': 'fixed-commit',
                         'allow_patterns': ['encoder/*.json', 'encoder/*.bin'], 'ignore_patterns': ['encoder/large*']}
