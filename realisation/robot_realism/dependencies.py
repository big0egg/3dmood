"""Audit the adapter's dependency list without importing CUDA or loading weights."""
import importlib.metadata as metadata
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .common import PROJECT


# The adapter always supplies simulation depth through control_path. It never
# runs the upstream automatic depth estimator or sparse-attention variants.
FEATURE_EXCLUSIONS = {
    'video-depth-anything': 'Uses precomputed simulation depth, not estimated depth',
}
LOCAL_PACKAGES = {'cosmos-transfer2', 'cosmos-oss', 'cosmos-cuda'}


def runtime_requirements():
    requirements = []
    for name in ('requirements-ppu.txt', 'requirements-ppu-overrides.txt'):
        for line in (PROJECT / 'runtime' / name).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                requirement = Requirement(line)
                if requirement.marker is None or requirement.marker.evaluate():
                    requirements.append(requirement)
    return requirements


def coverage_check():
    """Fail if an upstream ordinary dependency was omitted from the PPU list."""
    covered = {canonicalize_name(r.name) for r in runtime_requirements()} | LOCAL_PACKAGES
    missing, excluded = [], {}
    for name in ('pyproject.toml', 'packages/cosmos-oss/pyproject.toml',
                 'packages/cosmos-cuda/pyproject.toml'):
        project = tomllib.loads((PROJECT / name).read_text())['project']
        declared = list(project.get('dependencies', []))
        if project['name'] == 'cosmos-oss':
            # decord is grouped with CUDA extras upstream, but is an ordinary
            # CPU decoder on this Python 3.12 / x86_64 instance.
            declared += [r for r in project['optional-dependencies']['cu130_torch29']
                         if canonicalize_name(Requirement(r).name) == 'decord']
        for raw in declared:
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            key = canonicalize_name(requirement.name)
            if key in FEATURE_EXCLUSIONS:
                excluded[key] = FEATURE_EXCLUSIONS[key]
            elif key not in covered:
                missing.append(f'{name}: {raw}')
    result = {'component': 'upstream_dependency_coverage', 'ok': not missing,
              'excluded_features': excluded}
    if missing:
        result['error'] = 'Missing PPU requirements: ' + '; '.join(missing)
    return result


def dependency_checks():
    """Report every missing/version-mismatched direct dependency in one pass."""
    checks = [coverage_check()]
    for requirement in runtime_requirements():
        result = {'component': 'dependency:' + canonicalize_name(requirement.name),
                  'requirement': str(requirement), 'ok': False}
        try:
            version = metadata.version(requirement.name)
            result['version'] = version
            result['ok'] = requirement.specifier.contains(version, prereleases=True)
            if not result['ok']:
                result['error'] = f'Expected {requirement}; installed {version}'
        except metadata.PackageNotFoundError:
            result['error'] = f'Not installed: {requirement}'
        checks.append(result)
    return checks
