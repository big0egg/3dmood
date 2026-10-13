"""Isolated frontend setup; protects the already validated pose/framework env."""

import argparse
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .common import file_digest, read_json, write_json


def run(command, **kwargs):
    print('执行：' + ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def clone_frontend(config):
    target = Path(config['frontend']['python']).parent.parent
    core = Path(config['models']['unipose9d-box-rgb']['python']).parent.parent
    if not target.exists():
        run([sys.executable, '-m', 'venv', target])
    cfg = (target/'pyvenv.cfg').read_text()
    if 'include-system-site-packages = false' not in cfg:
        raise ValueError('前处理环境必须隔离系统 site-packages')
    destination = target/'lib/python3.12/site-packages'
    borrowed = {}
    # Borrow the selected working core packages, not the entire global environment.
    for path in (core/'lib/python3.12/site-packages').iterdir():
        name = path.name
        if name in ('pip', '_distutils_hack', 'setuptools', '__pycache__') or name.startswith(('pip-', 'setuptools-')) or name.endswith('.pth'):
            continue
        source = path.resolve()
        out = destination/name
        if not out.exists() and not out.is_symlink():
            out.symlink_to(source, target_is_directory=source.is_dir())
        borrowed[name] = str(source)
    # These already installed distributions can be reused offline. Versions are
    # constrained by the compatible core env; missing ones are installed later.
    pending = ['transformers', 'hydra-core', 'iopath', 'setuptools']
    seen = set()
    while pending:
        name = pending.pop()
        key = canonicalize_name(name)
        if key in seen: continue
        seen.add(key)
        # Resolve core-provided requirements using the new isolated interpreter.
        query = subprocess.run([str(target/'bin/python'), '-B', '-c',
            'import importlib.metadata as m,sys; print(m.version(sys.argv[1]))', name],
            capture_output=True, text=True)
        if query.returncode == 0:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if not requirement.marker or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
        entries = {Path(str(f)).parts[0] for f in dist.files or [] if Path(str(f)).parts}
        for entry in entries - {'.', '..', '__pycache__'}:
            source = Path(dist.locate_file(entry)).resolve()
            out = destination/entry
            if source.exists() and not out.exists() and not out.is_symlink():
                out.symlink_to(source, target_is_directory=source.is_dir())
                borrowed[entry] = str(source)
    write_json(target/'borrowed-packages.json', borrowed)
    # This file prevents pip from replacing the factory framework or numerics.
    versions = run([str(target/'bin/python'), '-B', '-c',
        "import importlib.metadata as m; print(chr(10).join(n+'=='+m.version(n) for n in ['torch','torchvision','numpy','scipy','Pillow','huggingface-hub','requests']))"],
        capture_output=True, text=True).stdout
    (target/'benchmark-constraints.txt').write_text(versions)
    return target


def prepare(config, offline=False):
    target = clone_frontend(config)
    python = target/'bin/python'
    front = config['frontend']
    assets = Path(front['sam2_checkpoint']).parents[1]
    assets.mkdir(parents=True, exist_ok=True)
    if offline:
        return dict(frontend_env=str(target), network_used=False,
                    missing_assets=[str(p) for p in (Path(front['sam2_checkpoint']), Path(front['grounding_model'])/'config.json') if not p.exists()])
    # Install only into the auxiliary env. SAM2's optional NVIDIA extension is
    # disabled; benchmark mask postprocessing leaves hole/sprinkle removal off.
    env = os.environ.copy()
    # The driver needs the benchmark source path; the setup subprocesses do not.
    # Inheriting it exposes opendet3d.egg-info to pip in this separate env.
    env.pop('PYTHONPATH', None)
    env.update(SAM2_BUILD_CUDA='0', SAM2_BUILD_ALLOW_ERRORS='0', XFORMERS_DISABLED='1',
               HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0')
    run([python, '-m', 'pip', 'install', '-c', target/'benchmark-constraints.txt',
         'transformers==4.57.0', 'hydra-core==1.3.2', 'iopath==0.1.10', 'wheel'], env=env)
    source = assets/'sam2-source'
    if not (source/'.git').exists():
        run(['git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/sam2.git', source], env=dict(env, GIT_LFS_SKIP_SMUDGE='1'))
    revision = run(['git', '-C', source, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    run([python, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', source], env=env)
    # Run Hub downloads in the auxiliary env. Paths and all file hashes are
    # captured in the subsequent inference provenance; repeated setup reuses them.
    script = """import json,sys
from pathlib import Path
from huggingface_hub import hf_hub_download,snapshot_download
c=json.loads(sys.argv[1]); sam=Path(c['sam2_checkpoint']);sam.parent.mkdir(parents=True,exist_ok=True)
if not sam.is_file():
 hf_hub_download('facebook/sam2.1-hiera-large','sam2.1_hiera_large.pt',local_dir=str(sam.parent))
g=Path(c['grounding_model'])
if not (g/'config.json').is_file() or not any((g/n).is_file() for n in ['model.safetensors','pytorch_model.bin']):
 snapshot_download('IDEA-Research/grounding-dino-tiny',local_dir=str(g),allow_patterns=['*.json','*.txt','*.safetensors','pytorch_model.bin'])
"""
    run([python, '-B', '-c', script, json.dumps(front)], env=env)
    run([python, '-m', 'pip', 'check'], env=env)
    summary = dict(frontend_env=str(target), sam2_git_revision=revision,
                   sam2_checkpoint_sha256=file_digest(front['sam2_checkpoint']),
                   grounding_model=front['grounding_model'], sam2_cuda_extension=False)
    write_json(assets/'frontend_preparation.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='准备隔离的 SAM2/文本前处理环境与权重')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('unipose_suite_config.json'))
    parser.add_argument('--offline', action='store_true', help='只复用本机依赖创建环境，不联网下载')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(read_json(args.config), args.offline), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
