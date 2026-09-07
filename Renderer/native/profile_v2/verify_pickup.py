"""Verify the frozen archive/art/evidence while active Lab sources evolve."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'Renderer/handoffs/candidates/lab_v2_terrain_lighting_r1'


def verify():
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    def digest(data):
        return hashlib.sha256(data).hexdigest()
    def check(row):
        path = ROOT / row['path']
        path.resolve().relative_to(ROOT)
        if digest(path.read_bytes()) != row['sha256']:
            raise ValueError('Frozen dependency changed: ' + row['path'])
    check(manifest['source_archive'])
    expected = {r['path']: r['sha256'] for r in manifest['source_files']}
    seen = set()
    with tarfile.open(ROOT / manifest['source_archive']['path']) as archive:
        for entry in archive:
            if not entry.isfile() or entry.name not in expected or entry.name in seen:
                raise ValueError('Unexpected pinned archive entry')
            if digest(archive.extractfile(entry).read()) != expected[entry.name]:
                raise ValueError('Pinned archive member changed: ' + entry.name)
            seen.add(entry.name)
    if seen != set(expected):
        raise ValueError('Incomplete pinned archive')
    for row in manifest['local_assets'] + manifest['baseline_handoffs']:
        check(row)
    frames = 0
    for reference in manifest['references']:
        check({'path': reference['report'], 'sha256': reference['report_sha256']})
        for frame in reference['frames']:
            check({'path': frame['image'], 'sha256': frame['sha256']})
            frames += 1
    print(f'PASS pinned pickup: {len(seen)} archived sources, '
          f'{len(manifest["local_assets"])} local assets, {frames} retained frames; '
          'active Lab sources are outside the frozen port.')


if __name__ == '__main__':
    verify()
