#!/usr/bin/env python3
"""Create a private venv inheriting the PPU SDK; never install into the base image."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ('torch', 'torchvision', 'transformer-engine', 'flash-attn', 'triton')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Print base SDK versions only; no writes/downloads')
    parser.add_argument('--venv', type=Path, default=Path('/tmp/robot-realism/cosmos25/venv'))
    parser.add_argument('--cache', type=Path, default=Path('/tmp/robot-realism/cosmos25/cache'))
    parser.add_argument('--index-url', default='https://pypi.org/simple', help='General Python package index; SDK packages stay pinned')
    args = parser.parse_args()
    if sys.prefix != sys.base_prefix:
        raise SystemExit('Use the image Python: /usr/local/bin/python scripts/bootstrap_cosmos25_ppu.py')
    sdk = {}
    for name in PROTECTED:
        try:
            sdk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise SystemExit(f'PPU base image lacks {name}; obtain a matching SDK image before installing')
    # Preserve installed CUDA runtime libraries as well as the Python kernels.
    # nvidia-ml-py is a Python wrapper, not a binary CUDA runtime library.
    for dist in metadata.distributions():
        name = dist.metadata.get('Name', '').lower().replace('_', '-')
        if name.startswith('nvidia-') and name != 'nvidia-ml-py':
            sdk[name] = dist.version
    print(json.dumps({'base_python': sys.executable, 'sdk': sdk}, indent=2), flush=True)
    if args.check:
        return
    target, cache = args.venv.expanduser().resolve(), args.cache.expanduser().resolve()
    if target == ROOT or ROOT.is_relative_to(target) or target.is_relative_to(Path(sys.base_prefix).resolve()):
        raise SystemExit('Unsafe venv directory; choose an independent local directory')
    marker = target / 'realisation-owner.json'
    expected = {'project': str(ROOT), 'base_python': sys.executable, 'sdk': sdk}
    if target.exists() and any(target.iterdir()):
        previous = json.loads(marker.read_text()) if marker.exists() else {}
        if (previous.get('project') != str(ROOT) or previous.get('base_python') != sys.executable
                or any(sdk.get(k) != v for k, v in previous.get('sdk', {}).items())):
            raise SystemExit('Existing venv is not owned by this SDK/project; choose a new --venv')
    else:
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(target)
        marker.write_text(json.dumps(expected, indent=2) + '\n')
    marker.write_text(json.dumps(expected, indent=2) + '\n')
    cache.mkdir(parents=True, exist_ok=True)
    constraints = target / 'sdk-constraints.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name, version in sdk.items()))
    python = target / 'bin/python'
    env = os.environ.copy()
    env.update(PIP_REQUIRE_VIRTUALENV='true', PIP_DISABLE_PIP_VERSION_CHECK='1', SAM2_BUILD_CUDA='0',
               HF_HOME=str(cache / 'huggingface'), PIP_CACHE_DIR=str(cache / 'pip'))
    pip = [str(python), '-m', 'pip', '--disable-pip-version-check']
    print('Installing Python dependencies into', target, flush=True)
    subprocess.run(pip + ['install', '--index-url', args.index_url, '--no-build-isolation', '--prefer-binary', '--timeout', '15', '--retries', '0',
                         '-c', str(constraints), '-r', str(ROOT / 'runtime/requirements-ppu.txt')], env=env, check=True)
    # Local packages are installed after their dependencies. No CUDA extras, no
    # isolated build that could download an unrelated official PyTorch wheel.
    subprocess.run(pip + ['install', '--no-deps', '--no-build-isolation', '-e', str(ROOT / 'packages/cosmos-cuda'),
                         '-e', str(ROOT / 'packages/cosmos-oss'), '-e', str(ROOT)], env=env, check=True)
    output = subprocess.check_output([str(python), '-c',
        'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in '+repr(tuple(sdk))+'}))'],
        env=env, text=True)
    if json.loads(output) != sdk:
        raise SystemExit('SDK versions changed unexpectedly; do not render with this environment')
    freeze = subprocess.check_output(pip + ['freeze', '--local'], env=env, text=True)
    (target / 'requirements-installed.txt').write_text(freeze)
    print('Installation complete. Activate:', target / 'bin/activate')
    print('Run: python -m robot_realism doctor --config configs/realism/cosmos25_libero_v1.yaml')


if __name__ == '__main__':
    main()
