#!/usr/bin/env python3
"""Audit shore/river channels against an installed source without editing a pack."""
from pathlib import Path
import argparse
import io
import json
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import civblp_material_resolver as resolver
from Renderer.tools.asset_compiler import grassland_pack_builder as builder


def probe(package, pack, output):
    from PIL import Image, ImageStat
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for material in ('beach', 'river'):
        binding = resolver.resolve_file(package,
            'ART_DEF_TERRAIN_MATERIAL_' + material.upper(),
            0 if material == 'beach' else None)
        for role in ('base_color', 'height', 'specular'):
            name = f'{material}_{role}.dds'
            extracted = output / name
            info = builder.extract_embedded_texture_role(package, binding, role, extracted)
            active = pack / 'textures' / name
            raw = bytearray(active.read_bytes())
            # Pillow decodes the identical BC blocks through their UNORM enum;
            # leave the actual DDS and its source sRGB declaration untouched.
            fmt, = struct.unpack_from('<I', raw, 128)
            if fmt in (72, 75, 78, 99):
                struct.pack_into('<I', raw, 128, fmt - 1)
            image = Image.open(io.BytesIO(raw)).convert('RGBA')
            channel = image.getchannel('A' if role == 'base_color' else 'R')
            stats = ImageStat.Stat(channel)
            records.append({
                'material': material, 'role': role, **info,
                'runtime_path': 'textures/' + name,
                'exact_source_match': active.read_bytes() == extracted.read_bytes(),
                'inspected_channel': 'alpha' if role == 'base_color' else 'red',
                'channel_range': channel.getextrema(),
                'channel_stddev': stats.stddev[0],
            })
    report = {
        'source_package_name': package.name,
        'materials': records,
        'limitation': 'Channel contents and typed bindings are confirmed; source-engine shader semantics are not recovered.',
    }
    (output / 'channels.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pack = args.pack.resolve()
    output = args.output.resolve()
    if output == pack or pack in output.parents:
        parser.error('Audit output must be outside the runtime pack')
    report = probe(args.package, pack, output)
    matches = sum(record['exact_source_match'] for record in report['materials'])
    print(f'{matches}/{len(report["materials"])} channels match the installed source exactly')
    return 0 if matches == len(report['materials']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
