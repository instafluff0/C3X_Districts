"""Recover source decal triangle/UV data omitted by the old bounds-only import.

Produces local, ignored source-derived data. No source-specific runtime loader.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/'Renderer/tools/asset_compiler'))
from generic_decal_compiler import StaticPackage, landmark_base_model, read_artdef_group
from clutter_blp_extractor import decode_buffer_entry, decode_texture_entry, extract_civbig_texture, TYPE_VERTEX_BUFFER, TYPE_TEXTURE


def decode(package, name, descriptor_index=None):
    package.select_direct_string(name)
    _, owner, _ = landmark_base_model(package)
    variants = [('LandmarkPackageEntry::DecalDescVectorEntry', 'DecalDesc', 92, 0x2c),
                ('LandmarkPackageEntry::DecalDesc2VectorEntry', 'DecalDesc2', 108, 0x3c)]
    found = [(kind, size, offset, fields[0][1]) for vector, kind, size, offset in variants
             if (fields := package.pointer_fields(owner, vector))]
    if len(found) != 1: raise ValueError('ambiguous decal descriptor version')
    kind, size, offset, vector = found[0]
    pointers = package.pointer_fields(vector, kind)
    if len(pointers) != 1: raise ValueError('ambiguous decal array')
    pointer = pointers[0][1]
    parts = package.allocations[pointer-1]['element_count']
    if descriptor_index is None:
        if parts != 1: raise ValueError('compound decal requires explicit parts')
        descriptor_index = 0
    if not 0 <= descriptor_index < parts: raise ValueError('decal part outside descriptor array')
    raw = package.array_element(pointer, descriptor_index)
    if len(raw) != size or struct.unpack_from('<I', raw, 8)[0] != 4: raise ValueError('unsupported decal primitive')
    vb, ib, start, index_start, count = struct.unpack_from('<5I', raw, offset)
    # The observed ground profile has direct triangle triples in DecalVB.
    # Do not apply conventional mesh index-buffer rules to these descriptors.
    if ib or index_start or not count or count % 3: raise ValueError('unsupported indexed decal')
    entry = decode_buffer_entry(package, package.unique_allocation(TYPE_VERTEX_BUFFER), vb, True)
    if entry['name'] != 'DecalVB' or entry['stride'] != 8 or entry['format'] != 3864023767:
        raise ValueError('unknown decal vertex format')
    if start+count > entry['count']: raise ValueError('decal range exceeds vertex buffer')
    payload = package.big_data(entry['offset'], entry['bytes'])
    vertices = [list(struct.unpack_from('<4e', payload, (start+i)*8)) for i in range(count)]
    if any(not math.isfinite(v) for row in vertices for v in row): raise ValueError('nonfinite decal vertex')
    if any(not -.02 <= v <= 1.02 for row in vertices for v in row): raise ValueError('unexpected local/atlas coordinate')
    bounds = list(struct.unpack_from('<4f', raw, 0x14))
    if not bounds[0] < bounds[2] or not bounds[1] < bounds[3]: raise ValueError('bad footprint')
    for i in range(0, count, 3):
        a,b,c = vertices[i:i+3]
        area = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        if abs(area) < 1e-6: raise ValueError('degenerate source decal triangle')
    content_error = None
    if size == 108:
        actual = [min(v[j] for v in vertices) for j in (0,1)] + [max(v[j] for v in vertices) for j in (0,1)]
        physical = [bounds[j%2]+actual[j]*(bounds[j%2+2]-bounds[j%2]) for j in range(4)]
        expected = struct.unpack_from('<4f', raw, 0x24)
        content_error = max(abs(a-b) for a,b in zip(physical, expected))
        if content_error > .015: raise ValueError('packed XY disagrees with exact source bounds')
    textures = [decode_texture_entry(package, package.unique_allocation(TYPE_TEXTURE),
                                    struct.unpack_from('<I', raw, offset+20+i*4)[0]) for i in range(2)]
    if [t['class'] for t in textures] != ['Decal_BaseColor','Decal_Heightmap']: raise ValueError('wrong source channels')
    return {'vertices': vertices, 'bounds': bounds, 'source_descriptor_type': kind,
            'source_descriptor_sha256': hashlib.sha256(raw).hexdigest(),
            'vertex_slice_sha256': hashlib.sha256(payload[start*8:(start+count)*8]).hexdigest(),
            'exact_bounds_max_error': content_error, 'textures': textures}


def main():
    steam = Path.home()/'Library/Application Support/Steam/steamapps'
    base = steam/'common/Sid Meier\'s Civilization VI/Civ6.app/Contents/Assets/Base'
    overlay = steam/'workshop/content/289070/1702339134'
    relative = 'Platforms/Windows/BLPs/environment/clutter.blp'
    source = StaticPackage(overlay/relative, 'TER_Grass_Decal02')
    baseline = StaticPackage(base/relative, 'TER_Grass_Decal02')
    mapping = json.loads((ROOT/'Renderer/tools/asset_compiler/decal_sets.json').read_text())
    records = []
    for family in ('grassland','plains'):
        group = next(g for g in mapping['groups'] if g['group_id']==f'terrain/{family}/surface')
        names = {a['source_asset']: a['asset_id'] for a in group['assets']}
        placements,_ = read_artdef_group(overlay/'ArtDefs/Clutter.artdef', group['artdef_set'], 'Plants', names)
        for name, asset in names.items():
            placement = next(p for p in placements if p['asset']==asset)
            data = decode(source, name); old = decode(baseline, name)
            records.append({'id': asset, 'family': family, 'placement': placement, **data,
                            'baseline_triangle_uv_identical': data['vertices']==old['vertices'],
                            'baseline_exact_bounds_max_error': old['exact_bounds_max_error']})
    out = ROOT/'Renderer/terrain_lab/v2/fixtures/beauty/source-ground-decals-r1'
    out.mkdir(parents=True, exist_ok=True)
    (out/'.gitignore').write_text('geometry.json\ngeometry.hlsl\n*.dds\n')
    (out/'geometry.json').write_text(json.dumps(records, indent=2)+'\n')
    vertices=[]; lines=['// Local source-derived data; regenerate with prepare_ground_decals.py.']
    parameters=[]; placement_data=[]
    for r in records:
        parameters.append([len(vertices), len(r['vertices']), r['placement']['count'], 0 if r['family']=='grassland' else 1])
        vertices.extend(r['vertices'])
        b=r['bounds'];placement_data.append([(b[2]-b[0])/12, (b[3]-b[1])/12,r['placement']['scale'],r['placement']['scale_variation']])
    for name,kind,rows in [('ground_decal_vertices','float4',vertices),('ground_decal_ranges','int4',parameters),('ground_decal_placement','float4',placement_data)]:
        lines.append(f'static const {kind} {name}[{len(rows)}] = {{')
        lines.extend(' '+kind+'('+','.join(format(v,'.9g') for v in row)+'),' for row in rows)
        lines.append('};')
    lines.append(f'#define GROUND_DECAL_COUNT {len(records)}')
    (out/'geometry.hlsl').write_text('\n'.join(lines)+'\n')
    texture_evidence=[]
    for name in ('Grass_Decal_B','Grass_Decal_H','Plains_Decal_B','Plains_Decal_H'):
        source_texture=overlay/'Platforms/Windows/BLPs/SHARED_DATA'/('TEXTURE_TER_'+name)
        info=extract_civbig_texture(source_texture,out/(name+'.dds'))
        texture_evidence.append({'name':name,'source_sha256':info['source_sha256'],
                                 'dds_sha256':info['dds_sha256']})
    evidence={'schema':'c3x.ground_decal_geometry.v1','classification':'source_geometry_adaptation',
              'source_package_relative':relative,'selected_overlay_package_sha256':hashlib.sha256(source.data).hexdigest(),
              'baseline_package_sha256':hashlib.sha256(baseline.data).hexdigest(),
              'artdef_sha256':hashlib.sha256((overlay/'ArtDefs/Clutter.artdef').read_bytes()).hexdigest(),
              'records':len(records),'vertices':len(vertices),
              'geometry_sha256':hashlib.sha256((out/'geometry.json').read_bytes()).hexdigest(),
              'shader_data_sha256':hashlib.sha256((out/'geometry.hlsl').read_bytes()).hexdigest(),
              'all_base_overlay_triangle_uv_identical':all(r['baseline_triangle_uv_identical'] for r in records),
              'max_exact_bounds_error':max(r['baseline_exact_bounds_max_error'] for r in records),
              'selected_overlay_textures':texture_evidence,
              'remaining':'Source scale/count evaluation and render composition are unproven; preserve triangles rather than full-sheet rectangles.'}
    (out/'provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence))


if __name__ == '__main__': main()
