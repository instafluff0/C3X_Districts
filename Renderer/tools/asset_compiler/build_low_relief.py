"""Compile local authored ground relief to optional source-neutral C3X height fields."""
from pathlib import Path
import hashlib
import json
import struct

ROOT = Path(__file__).resolve().parents[3]
INPUTS = [f'Renderer/packs/Civ5EnvironmentSkin/textures/relief/hills/{family}/height_lod0.dds'
          for family in ('continental', 'continental_plains')]


def input_paths(terrain_pack=None):
    if terrain_pack is None:
        return INPUTS
    root=Path(terrain_pack).resolve()
    root.relative_to(ROOT)
    return [(root/f'textures/relief/hills/{family}/height_lod0.dds').relative_to(ROOT).as_posix()
            for family in ('continental', 'continental_plains')]


def sources(terrain_pack=None):
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in input_paths(terrain_pack)}


def broad_field(source, width, height):
    """Move authored microheight into a few-tile landform band, preserving wrap."""
    target = 256
    if width != 2048 or height != 2048:
        raise ValueError('The calibrated broad relief needs 2048-square source fields')
    # Pool 8x8 source texels, then apply a periodic, separable 9x9 low pass.
    # The kernel spans about three native tiles at the configured repeat span.
    pooled = [0.0] * (target * target)
    for y in range(target):
        for x in range(target):
            pooled[y * target + x] = sum(source[(y * 8 + dy) * width + x * 8 + dx]
                                         for dy in range(8) for dx in range(8)) / 64.0
    rows = [0.0] * len(pooled)
    for y in range(target):
        base = y * target
        for x in range(target):
            rows[base + x] = sum(pooled[base + (x + dx) % target]
                                 for dx in range(-4, 5)) / 9.0
    mean = sum(rows) / len(rows)
    result = bytearray(target * target)
    for y in range(target):
        for x in range(target):
            smooth = sum(rows[((y + dy) % target) * target + x]
                         for dy in range(-4, 5)) / 9.0
            result[y * target + x] = round(max(0, min(255, mean + (smooth - mean) * 1.55)))
    return bytes(result), target, target


def build(output, variant='broad', terrain_pack=None):
    if variant not in ('broad', 'source'):
        raise ValueError('Unknown low-relief variant')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = bytearray(b'C3XLOW1\0')
    for name in input_paths(terrain_pack):
        raw = (ROOT/name).read_bytes()
        height, width = struct.unpack_from('<II', raw, 12)
        if raw[:4] != b'DDS ' or raw[84:88] != b'DX10' or struct.unpack_from('<I', raw, 128)[0] != 61:
            raise ValueError('Ground relief requires an R8_UNORM source')
        if width > 2048 or height > 2048 or len(raw) != 148 + width*height:
            raise ValueError('Unexpected authored field extent')
        field = raw[148:]
        if variant == 'broad':
            field, width, height = broad_field(field, width, height)
        # Height and field span are C3X visual calibrations, not inferred
        # proprietary engine settings. The source variant is an A/B control.
        data += struct.pack('<IIff', width, height, 64., 96.) + field
    (output/'low-relief.bin').write_bytes(data)
    inputs = sources(terrain_pack)
    (output/'low-relief.json').write_text(json.dumps({
        'schema': 'c3x.low_relief.v1', 'source_sha256': inputs,
        'amplitude_pixels': 64, 'repeat_native_tiles': 96, 'variant': variant,
        'calibration': 'C3X visual study; engine placement/amplitude unconfirmed',
        'sha256': hashlib.sha256(data).hexdigest()}, indent=2)+'\n')
    return inputs


if __name__ == '__main__':
    build(ROOT/'Renderer/packs/NaturalFidelityRuntime')
