"""Promote hash-verified local pickup art into a source-independent runtime pack.

No art is redistributed; this reproducible offline adapter rewrites only the
bundle's path table. Vertex/index bytes and all texture payloads stay exact.
"""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'Renderer/handoffs/candidates/lab_v2_terrain_lighting_r1'
OUTPUT = ROOT / 'Renderer/packs/TerrainProfileR1'


def prepare():
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    inventory = {r['path']: r for r in manifest['local_assets']}
    records = []
    def copy(source, destination, view_format=None):
        path = ROOT / source
        path.resolve().relative_to(ROOT)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != inventory[source]['sha256']:
            raise ValueError('Pinned asset changed: ' + source)
        target = OUTPUT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        source_digest = digest
        if view_format is not None:
            # The Lab requests a linear view even when the source wrapper says
            # sRGB. Normalize the DDS header, preserving the BC payload exactly.
            changed = bytearray(data)
            struct.pack_into('<I', changed, 128, view_format)
            data = bytes(changed)
            digest = hashlib.sha256(data).hexdigest()
        target.write_bytes(data)
        records.append({'path': destination, 'sha256': digest, 'source': source,
                        'source_sha256': source_digest})
        return data
    module = manifest['references'][0]['module']
    copy(module['hill_source']['path'], 'height.dds')
    source = module['coastal_rocks']['path']
    data = copy(source, 'cliffs.bin')
    if data[:8] != b'C3XVEG1\0' or struct.unpack_from('<4I', data, 8) != (1, 24, 6, 1):
        raise ValueError('Unexpected pinned cliff bundle')
    cursor = 24
    rewritten = bytearray(data[:24])
    for index in range(24):
        length, = struct.unpack_from('<I', data, cursor)
        cursor += 4
        path = data[cursor:cursor+length].decode()
        cursor += length
        destination = f'textures/cliff_{index//4}_{index%4}.dds'
        copy(path, destination, 71 if index % 4 == 3 else None)
        name = destination.encode()
        rewritten += struct.pack('<I', len(name)) + name
    rewritten += data[cursor:]
    (OUTPUT / 'cliffs.bin').write_bytes(rewritten)
    records[1]['source_sha256'] = records[1]['sha256']
    records[1]['sha256'] = hashlib.sha256(rewritten).hexdigest()
    record = {'schema': 'c3x.terrain_profile.v1', 'revision': 1,
              'candidate': manifest['id'], 'files': records,
              'source_mesh_bytes_preserved': True}
    (OUTPUT / 'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'Prepared {len(records)} pinned assets in {OUTPUT.relative_to(ROOT)}')


if __name__ == '__main__':
    prepare()
