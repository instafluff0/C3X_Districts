"""Trace installed ground selectors and normalized material channels to a render.

Metadata and hashes only. Presence in a packet is not proof of correct shader
usage or source-equivalent composition. Keep unresolved layer semantics explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / 'Renderer/terrain_lab/v2'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def artdef_fields(path):
    result = []
    def walk(node, context):
        for tag in ('m_CollectionName', 'm_Name'):
            label = node.find(tag)
            if label is not None and label.get('text'):
                context = context + [label.get('text')]
        for field in node.findall('./m_Fields/m_Values/Element'):
            result.append({'context': context, 'class': field.get('class'),
                           'fields': {child.tag: child.get('text', child.text)
                                      for child in field}})
        for child in node:
            if child.tag not in ('m_Fields', 'm_Values'):
                walk(child, context)
    walk(ET.parse(path).getroot(), [])
    return result


def packet_textures(path):
    with path.open('rb') as f:
        def read(n):
            b = f.read(n)
            if len(b) != n: raise ValueError('truncated packet')
            return b
        def u32(): return struct.unpack('<I', read(4))[0]
        def blob(skip=False):
            size = u32(); external = bool(size & 0x80000000); size &= 0x7fffffff
            if external:
                key = read(64).decode()
                if skip: return None
                b = (Path(str(path)+'.blobs') / key).read_bytes()
                if len(b) != size or digest(b) != key: raise ValueError('corrupt blob')
                return b
            if skip: f.seek(size, 1); return None
            return read(size)
        magic, version, w, h, down = [u32() for _ in range(5)]
        if magic != 0x32514c43 or version not in range(1, 7): raise ValueError('packet version')
        if version >= 3: read(28 if version >= 4 else 24)
        textures = []
        for index in range(u32()):
            tw, th, fmt, mips = [u32() for _ in range(4)]
            sha = hashlib.sha256()
            for _ in range(mips): u32(); sha.update(blob())
            textures.append({'index': index+1, 'dimensions': [tw, th], 'format': fmt,
                             'mips': mips, 'payload_sha256': sha.hexdigest(), 'bindings': []})
        for _ in range(u32()): blob(skip=True)
        for draw in range(u32()):
            values = [u32() for _ in range(7)]
            if version >= 3: read(8)
            if version >= 4: read(4)
            if version >= 5: read(4)
            if version >= 6: read(24)
            read(u32()*8)
            for slot in range(128):
                index = u32()
                if index:
                    textures[index-1]['bindings'].append({'draw': draw, 'slot': slot,
                                                         'feature_shader': bool(values[4])})
        if f.read(1): raise ValueError('trailing packet data')
    return {'version': version, 'viewport': [w, h], 'downsample': down,
            'sha256': digest(path.read_bytes()), 'textures': textures}


def main():
    steam = Path.home() / 'Library/Application Support/Steam/steamapps'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-root', type=Path, default=steam/'common/Sid Meier\'s Civilization VI/Civ6.app/Contents/Assets')
    p.add_argument('--overlay-root', type=Path, default=steam/'workshop/content/289070/1702339134')
    p.add_argument('--report', type=Path, default=V2/'audits/beauty/out/river-corridor-r3/inland/report.json')
    a = p.parse_args()
    run = json.loads(a.report.read_text())
    packet = packet_textures(ROOT/run['outputs'][0]['packet'])
    by_hash = {}
    for t in packet['textures']: by_hash.setdefault(t['payload_sha256'], []).append(t)
    pack = ROOT/run['effective']['fixture']['packs']['terrain']
    channels = []
    def channel_walk(value, context, descriptor):
        if isinstance(value, dict):
            if 'texture' in value:
                path = pack/value['texture']; data = path.read_bytes()
                if data[:4] != b'DDS ' or data[84:88] != b'DX10': raise ValueError('DDS contract')
                matches = by_hash.get(digest(data[148:]), [])
                channels.append({'descriptor': descriptor, 'role': context,
                                 'texture': path.relative_to(ROOT).as_posix(),
                                 'sha256': digest(data), 'packet_matches': matches,
                                 'status': 'uploaded_shader_use_unproven' if matches else 'not_uploaded',
                                 'declared_color_space': value.get('color_space')})
            for k, v in value.items(): channel_walk(v, context+[k], descriptor)
        elif isinstance(value, list):
            for i, v in enumerate(value): channel_walk(v, context+[str(i)], descriptor)
    for path in sorted((pack/'materials').rglob('*.json')) + [pack/'water/catalog.json']:
        channel_walk(json.loads(path.read_text()), [], path.relative_to(ROOT).as_posix())
    definitions = []
    names = {'TerrainStyle.artdef', 'Terrains.artdef', 'Clutter.artdef', 'Features.artdef',
             'Water.artdef', 'WaterMaterials.artdef', 'Wave.artdef'}
    for label, root in [('installed_assets', a.assets_root), ('selected_overlay', a.overlay_root)]:
        for path in sorted(root.rglob('*.artdef')):
            if path.name not in names: continue
            definitions.append({'source_tree': label, 'path': path.relative_to(root).as_posix(),
                                'sha256': digest(path.read_bytes()), 'parameters': artdef_fields(path)})
    output = {'schema': 'c3x.lab_ground_layer_audit.v1',
              'status': 'incomplete_layering_audit',
              'source_report': a.report.relative_to(ROOT).as_posix(),
              'packet': packet, 'normalized_material_channels': channels,
              'installed_artdef_parameters': definitions,
              'limits': ['ArtDef field order is not engine rendering order.',
                         'Base/DLC/overlay records are preserved separately; merge precedence is not inferred.',
                         'Uploaded texture or filename match does not prove shader use or correct UV/masks.',
                         'Material descriptor coverage does not yet cover all clutter/water/relief package channels.',
                         'Absent channels may be intentional for unsupported biomes or retained Civ III FOW; classify each.',
                         'Natural-wonder records are inventoried as source context; implementation remains deferred.']}
    # Preserve the complete source inventory locally without adding megabytes of
    # unrelated resource/wonder ArtDef parameters to the reviewable source tree.
    inventory = V2/'audits/beauty/out/ground-layer-audit/source_inventory.json'
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps(definitions, indent=2)+'\n')
    output['installed_artdef_parameters'] = {
        'path': inventory.relative_to(ROOT).as_posix(),
        'sha256': digest(inventory.read_bytes()),
        'files': [{k: v for k, v in d.items() if k != 'parameters'} for d in definitions]}
    target = V2/'audits/beauty/GROUND_LAYER_AUDIT.json'
    target.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'artdef_files': len(definitions), 'material_channels': len(channels),
                      'not_uploaded': sum(c['status']=='not_uploaded' for c in channels),
                      'output': target.relative_to(ROOT).as_posix()}))


if __name__ == '__main__': main()
