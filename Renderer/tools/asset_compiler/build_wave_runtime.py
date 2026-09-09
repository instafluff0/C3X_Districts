"""Compile generic normalized coastal art; source-game extraction is separate."""
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'Renderer/packs/CoastalWavesNormalized'
FILES = ('wave.json', 'crest.dds', 'auxiliary.dds', 'crest-delays.f32')


def sources():
    return [(SOURCE / name).relative_to(ROOT).as_posix() for name in FILES]


def build(output, source=SOURCE):
    data = {name: (source / name).read_bytes() for name in FILES}
    config = json.loads(data['wave.json'])
    atlas = config['atlas']
    if config['schema'] != 'c3x.coastal_wave_assets.v1' or (atlas['columns'], atlas['rows'], atlas['variants']) != (8, 2, 16):
        raise ValueError('Unsupported coastal atlas layout')
    for name, size in [('crest.dds', (1024, 1024)), ('auxiliary.dds', (512, 256))]:
        dds = data[name]
        if len(dds) < 148 or dds[:4] != b'DDS ' or struct.unpack_from('<2I', dds, 12) != size[::-1] or struct.unpack_from('<I', dds, 128)[0] != 28:
            raise ValueError('Expected complete linear RGBA8 coastal DDS: ' + name)
        width, height = size
        count = struct.unpack_from('<I', dds, 28)[0]
        expected = sum(max(1, width >> mip)*max(1, height >> mip)*4 for mip in range(count))
        if not 1 <= count <= 11 or len(dds) != 148 + expected:
            raise ValueError('Truncated coastal mip chain')
    values = struct.unpack('<8192f', data['crest-delays.f32'])
    if any(not math.isfinite(v) or not (0 <= v <= 1 or v > 1e30) for v in values):
        raise ValueError('Invalid crest delay')
    # Preserve the exact float values, including inactive sentinels. R32_FLOAT
    # is an ordinary runtime texture, addressed by variant and along-crest row.
    header = bytearray(data['crest.dds'][:148])
    for offset, value in [(12, 16), (16, 512), (20, 2048), (28, 1), (128, 41)]:
        struct.pack_into('<I', header, offset, value)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('crest.dds', 'auxiliary.dds'):
        (output / name).write_bytes(data[name])
    (output / 'delays.dds').write_bytes(header + data['crest-delays.f32'])
    (output / 'waves.bin').write_bytes(struct.pack('<4sIII', b'CWV1', 1, int(config['enabled']), 0))
    consumed = {(source / name).relative_to(ROOT).as_posix(): hashlib.sha256(value).hexdigest() for name, value in data.items()}
    (output / 'manifest.json').write_text(json.dumps(dict(schema='c3x.coastal_waves.v1', source_sha256=consumed), indent=2)+'\n')
    return consumed


if __name__ == '__main__':
    build(ROOT / 'Renderer/packs/CoastalWavesRuntime')
