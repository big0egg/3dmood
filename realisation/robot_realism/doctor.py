from pathlib import Path
import json
import subprocess

from .common import PROJECT, atomic_json
from .dataset import source_records


def doctor(config):
    from .pipeline import worker_env
    checks = []
    runtime = None
    try:
        _, records = source_records(config)
        checks.append({'component': 'input_dataset', 'ok': True, 'states': len(records)})
    except Exception as error:
        checks.append({'component': 'input_dataset', 'ok': False, 'error': str(error)})
    python = Path(config['runtime']['python'])
    checks.append({'component': 'runtime_python', 'ok': python.is_file(), 'path': str(python)})
    if python.is_file():
        result = subprocess.run([str(python), '-B', '-m', 'robot_realism.probe',
                                 '--controls', *config['generation']['controls']], cwd=PROJECT,
                                env=worker_env(config), text=True, capture_output=True, timeout=180)
        lines = [line for line in result.stdout.splitlines() if line.startswith('REALISM_PROBE_JSON=')]
        if lines:
            runtime = json.loads(lines[-1].split('=', 1)[1])
            checks.append({'component': 'runtime_probe', 'ok': result.returncode == 0 and runtime['ready']})
        else:
            checks.append({'component': 'runtime_probe', 'ok': False,
                           'error': (result.stderr + result.stdout)[-3000:]})
    lfs = []
    for name in ('assets/image_example/image2image.json', 'assets/robot_example/robot_input.mp4'):
        path = PROJECT / name
        if path.exists() and path.open('rb').read(80).startswith(b'version https://git-lfs.github.com/spec/'):
            lfs.append(name)
    result = {'ready': all(item['ok'] for item in checks), 'checks': checks, 'runtime': runtime,
              'lfs_placeholders': lfs,
              'lfs_note': 'Upstream demo assets need LFS; this adapter uses your GT files and does not require those demos',
              'checkpoints_verified': False, 'native_image_generated': False}
    atomic_json(Path(config['output']['run_dir']) / 'doctor.json', result)
    return result
