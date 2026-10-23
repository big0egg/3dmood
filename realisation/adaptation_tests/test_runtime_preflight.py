"""Missing decoder/configuration dependencies must fail before weight loading."""
import importlib.metadata as metadata

import pytest

from robot_realism import dependencies, probe


@pytest.mark.parametrize('omitted', ['decord', 'shapely'])
def test_upstream_requirement_omissions_are_detected(monkeypatch, omitted):
    requirements = dependencies.runtime_requirements()
    monkeypatch.setattr(dependencies, 'runtime_requirements',
                        lambda: [r for r in requirements if r.name.lower() != omitted])
    result = dependencies.coverage_check()
    assert not result['ok']
    assert omitted in result['error']


def test_dependency_audit_reports_all_missing_and_wrong_versions(monkeypatch):
    def version(name):
        if name in ('decord', 'shapely'):
            raise metadata.PackageNotFoundError(name)
        if name == 'numpy':
            return '1.26.4'
        return '999.0'
    monkeypatch.setattr(dependencies.metadata, 'version', version)
    failures = {c['component']: c for c in dependencies.dependency_checks() if not c['ok']}
    assert 'Not installed' in failures['dependency:decord']['error']
    assert 'Not installed' in failures['dependency:shapely']['error']
    assert 'installed 1.26.4' in failures['dependency:numpy']['error']


def test_lazy_targets_are_imported_but_never_constructed():
    import attrs
    from omegaconf import OmegaConf

    class NeverConstruct:
        def __init__(self):
            raise AssertionError('Preflight must not construct models')

    @attrs.define
    class Config:
        nested: object

    config = Config([OmegaConf.create({'_target_': 'builtins.dict'}),
                     {'_target_': NeverConstruct}])
    assert probe.resolve_model_targets(config) == 2
    config.nested.append({'_target_': 'missing_cosmos_test_dependency.Model'})
    with pytest.raises(ImportError):
        probe.resolve_model_targets(config)


@pytest.mark.parametrize('bad_config', [True, False])
def test_ready_requires_actual_model_config_and_every_module(monkeypatch, bad_config):
    import torch

    monkeypatch.setattr(probe, 'dependency_checks', lambda: [])
    monkeypatch.setattr(probe.metadata, 'version', lambda name: '1.0')

    def import_module(name):
        if not bad_config and name in ('decord', 'shapely.geometry'):
            raise ModuleNotFoundError(name)

    def config_check(controls):
        if bad_config:
            raise ModuleNotFoundError('dependency_hidden_inside_model_config')
        return {'experiment': 'unit_test', 'resolved_targets': 1}

    def unexpected_cuda_call():
        raise AssertionError('imports-only must not execute CUDA tests')

    monkeypatch.setattr(probe.importlib, 'import_module', import_module)
    monkeypatch.setattr(probe, 'model_config_check', config_check)
    monkeypatch.setattr(torch.cuda, 'is_available', unexpected_cuda_call)
    result = probe.probe(imports_only=True)
    assert not result['ready']
    failures = {c['component'] for c in result['checks'] if not c['ok']}
    assert failures == ({'cosmos_model_config'} if bad_config
                        else {'import:decord', 'import:shapely.geometry'})
