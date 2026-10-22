"""The upstream NumPy exception must never hide other dependency failures."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def bootstrap(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/bootstrap_cosmos25_ppu.py'
    spec = importlib.util.spec_from_file_location('test_cosmos_bootstrap', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    versions = {'megatron-core': '0.14.0', 'numpy': '2.2.6', 'torch': '2.9.0', 'packaging': '24.2'}
    requirements = ['torch', 'numpy<2.0.0', 'packaging', 'missing-extra; extra == "dev"']
    monkeypatch.setattr(module.metadata, 'version', versions.__getitem__)
    monkeypatch.setattr(module.metadata, 'requires', lambda name: requirements)
    return module, versions, requirements


def test_accepts_only_the_upstream_numpy_exception(bootstrap):
    module, _, _ = bootstrap
    module.verify_megatron_dependencies()


@pytest.mark.parametrize('name,version', [('numpy', '1.26.4'), ('numpy', '2.3.4'),
                                        ('megatron-core', '0.15.0')])
def test_rejects_changed_version_combination(bootstrap, name, version):
    module, versions, _ = bootstrap
    versions[name] = version
    with pytest.raises(RuntimeError):
        module.verify_megatron_dependencies()


def test_numpy_exception_does_not_skip_other_requirements(bootstrap):
    module, _, requirements = bootstrap
    requirements.append('packaging>=99')
    with pytest.raises(RuntimeError, match='Unsatisfied Megatron dependency'):
        module.verify_megatron_dependencies()


def test_missing_dependency_metadata_is_rejected(bootstrap):
    module, _, requirements = bootstrap
    requirements.clear()
    with pytest.raises(RuntimeError, match='metadata is missing'):
        module.verify_megatron_dependencies()
