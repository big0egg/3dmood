from pathlib import Path
import gzip

import numpy as np
from PIL import Image
import pytest

from robot_realism.common import PROJECT, atomic_json, file_hash
from robot_realism.config import load_config


@pytest.fixture
def config(tmp_path):
    value = load_config(PROJECT / 'configs/realism/cosmos25_libero_v1.yaml')
    value['input']['root'] = str(tmp_path / 'source')
    value['output'] = {'root': str(tmp_path / 'result'), 'run_dir': str(tmp_path / 'run'), 'scratch': str(tmp_path / 'scratch')}
    value['runtime']['python'] = str(tmp_path / 'runtime/bin/python')
    value['runtime']['cache'] = str(tmp_path / 'cache')
    root = Path(value['input']['root'])
    accepted = {}
    for i in range(4):
        split = 'train' if i < 2 else 'eval'
        record = {'state_id': f'test_state_{i}', 'scene_id': f'TEST_SCENE{i}', 'split': split,
                  'source': 'TEST_FIXTURE_NOT_LIBERO', 'episode': 'test', 'state_index': i}
        folder = root / split / record['scene_id'] / record['state_id']
        folder.mkdir(parents=True)
        np.save(folder / 'state.npy', np.array([0., 1., 2.]))
        provenance = root / 'provenance/test.xml.gz'
        provenance.parent.mkdir(exist_ok=True)
        provenance.write_bytes(gzip.compress(b'<mujoco/>', mtime=0))
        for camera in value['input']['cameras']:
            target = folder / camera
            target.mkdir()
            rgb = np.tile(np.arange(32, dtype=np.uint8)[None, :, None], (32, 1, 3))
            Image.fromarray(rgb).save(target / 'rgb.png')
            mask = np.zeros((32, 32), dtype=np.uint16)
            mask[8:24, 8:24] = 1
            Image.fromarray(mask).save(target / 'instance.png')
            depth = np.tile(np.linspace(.3, 2, 32, dtype=np.float32), (32, 1))
            np.savez_compressed(target / 'depth.npz', z_m=depth, valid=np.ones((32, 32), dtype=bool))
            atomic_json(target / 'annotation.json', {'objects': [], 'intrinsics': [[32, 0, 16], [0, 32, 16], [0, 0, 1]],
                        'T_world_from_camera': np.eye(4).tolist(), 'T_camera_from_world': np.eye(4).tolist(),
                        'provenance': record, 'source_xml': 'provenance/test.xml.gz'})
        hashes = {str(p.relative_to(folder)): file_hash(p) for p in folder.rglob('*') if p.is_file()}
        atomic_json(folder / 'COMMIT.json', {'state_record': record, 'plan_id': 'test_plan', 'profile_id': 'test_profile',
                    'cameras': value['input']['cameras'], 'files_sha256': hashes,
                    'provenance_xml': {'path': 'provenance/test.xml.gz', 'sha256': file_hash(provenance)}})
        accepted[record['state_id']] = record
    atomic_json(root / 'dataset_info.json', {'plan_id': 'test_plan', 'profile_id': 'test_profile',
                'test_only': True, 'config': {'simulation': {'cameras': value['input']['cameras']}}})
    atomic_json(root / 'manifest.json', {'plan_id': 'test_plan', 'profile_id': 'test_profile', 'accepted': accepted})
    return value
