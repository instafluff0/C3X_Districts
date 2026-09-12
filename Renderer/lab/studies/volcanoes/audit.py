#!/usr/bin/env python3
"""Inspect local volcano source channels without modifying source/runtime art."""
from pathlib import Path
import argparse
import io
import json
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.tools.asset_compiler import water_pack_builder as water
from Renderer.tools.asset_compiler import c3x_asset_compiler as compiler
from Renderer.tools.asset_compiler import terrain_relief_builder as relief


def dds_image(path):
    from PIL import Image
    data = bytearray(path.read_bytes())
    height, width = struct.unpack_from('<II', data, 12)
    fmt, = struct.unpack_from('<I', data, 128)
    if fmt in (61, 62):
        return Image.frombytes('L', (width, height), bytes(data[148:148+width*height]))
    # Pillow lacks BC3 sRGB. Only the in-memory decoder view changes; compressed
    # bytes and encoded colors are intact, with no source file modification.
    if fmt == 78:
        struct.pack_into('<I', data, 128, 77)
    return Image.open(io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=water.DEFAULT_ASSETS)
    args = parser.parse_args()
    from PIL import Image, ImageDraw
    import numpy as np
    out = ROOT / 'Renderer/lab/out/volcanoes'
    out.mkdir(parents=True, exist_ok=True)
    records, panels = {}, []
    for name in ('base', 'height', 'active_base', 'active_specular'):
        path = ROOT / 'Renderer/packs/Civ5EnvironmentSkin/textures/water/volcano' / (name+'.dds')
        source_name = water.OPTIONAL_TEXTURES['volcano/'+name]
        source = args.assets / source_name
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            decoded = Path(tmp) / 'channel.dds'
            info = compiler.extract_civbig_to_dds(source, decoded)
            exact = decoded.read_bytes() == path.read_bytes()
        im = dds_image(path)
        records[name] = dict(source=source_name, runtime=renderer.relative(path),
            source_sha256=renderer.checksum(source), runtime_sha256=renderer.checksum(path),
            exact_extracted_dds=exact, size=list(im.size), extrema=im.getextrema(),
            format=info['dxgi_format'], mip_count=info['mip_count'])
        panels.append((name+' RGB', im.convert('RGB')))
        if name in ('base', 'active_base'):
            panels.append((name+' alpha', im.getchannel('A').convert('RGB')))
        if name == 'height':
            for channel in ('R', 'G'):
                panels.append(('BC5 '+channel, im.getchannel(channel).convert('RGB')))
            pixels = np.asarray(im, dtype=float)/255
            records[name]['old_normal_xy_mean'] = ((pixels[:,:,:2]*2-1)*.34).mean(axis=(0,1)).tolist()
            records[name]['green_below_neutral_fraction'] = float((pixels[:,:,1]<.5).mean())
    package_name = 'DLC/Expansion2/Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp'
    package_path = args.assets/package_name
    _, elements, package_report = relief.inspect_terrain_element_package(package_path)
    element = elements['ART_DEF_TERRAIN_ELEMENT_FEATURE_VOLCANO_01']
    records['terrain_element'] = dict(source=package_name, parameters=element['parameters'],
                                     grid_dimensions=element['grid_dimensions'])
    package = package_path.read_bytes()
    for name in ('height','blend','region_ids'):
        path = ROOT / 'Renderer/packs/TerrainElementsNormalized/textures/terrain_elements/terrain_feature_volcano' / (name+'_lod0.dds')
        im = dds_image(path)
        records['macro_'+name] = dict(runtime=renderer.relative(path), sha256=renderer.checksum(path),
            size=list(im.size), extrema=im.getextrema())
        lod = next(item for item in element['channels'][name] if item['level']==0)
        offset = package_report['big_data_offset']+lod['relative_offset']
        extracted = relief.make_r8_dds(lod['width'],lod['height'],
            package[offset:offset+lod['bytes']],62 if name=='region_ids' else 61)
        records['macro_'+name].update(source_resource=lod['name'], exact_extracted_dds=extracted==path.read_bytes())
        panels.append(('macro '+name, im.convert('RGB')))
    canvas = Image.new('RGB', (1120, 930), (30,33,38))
    draw = ImageDraw.Draw(canvas)
    for i, (name, im) in enumerate(panels):
        x, y = i%4*280, i//4*310
        draw.text((x+8,y+8),name,fill='white')
        im.thumbnail((270,270)); canvas.paste(im,(x+5,y+32))
    canvas.save(out/'source-channels.png')
    renderer.write(out/'source-audit.json', records)
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    main()
