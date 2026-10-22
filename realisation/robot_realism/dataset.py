from pathlib import Path

from .common import file_hash, read_json, safe_child


def bundle(root, record):
    return safe_child(root, f"{record['split']}/{record['scene_id']}/{record['state_id']}")


def verify_source(root, record, cameras):
    folder = bundle(root, record)
    marker = folder / 'COMMIT.json'
    commit = read_json(marker)
    if commit.get('state_record') != record or not set(cameras).issubset(commit.get('cameras', [])):
        raise ValueError(f"Source state/cameras differ from COMMIT: {folder}")
    hashes = commit.get('files_sha256', {})
    required = {'state.npy'} | {f'{camera}/{name}' for camera in cameras
                               for name in ('rgb.png', 'depth.npz', 'instance.png', 'annotation.json')}
    if not required.issubset(hashes):
        raise ValueError(f"Source COMMIT lacks payload hashes: {folder}")
    for name, expected in hashes.items():
        p = safe_child(folder, name)
        if not p.is_file() or file_hash(p) != expected:
            raise ValueError(f"Source file missing/corrupt: {p}")
    provenance = commit.get('provenance_xml', {})
    p = safe_child(root, provenance.get('path', ''))
    if not p.is_file() or file_hash(p) != provenance.get('sha256'):
        raise ValueError(f'Source scene XML missing/corrupt: {p}')
    return commit


def source_records(config):
    root = Path(config['input']['root'])
    info = read_json(root / 'dataset_info.json')
    manifest = read_json(root / 'manifest.json')
    if manifest.get('plan_id') != info.get('plan_id') or manifest.get('profile_id') != info.get('profile_id'):
        raise ValueError('Source dataset/manifest identities differ')
    records = list(manifest['accepted'].values())
    cameras = config['input']['cameras']
    if not set(cameras).issubset(info['config']['simulation']['cameras']):
        raise ValueError('Selected cameras are absent from source dataset')
    if not records:
        raise ValueError('No committed source samples')
    for record in records:
        if record['split'] not in ('train', 'eval'):
            raise ValueError('Source must retain train/eval split')
        bundle(root, record)  # Reject unsafe paths before any writes.
    return info, records
