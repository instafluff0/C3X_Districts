#!/usr/bin/env python3
"""Compare current production rendering with the accepted mountain body study.

No source mutations, fixed-reference replacement, staging or game launch.
Uses the renderer.py native dispatcher and requires its matching candidate.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer


def main():
    from PIL import Image, ImageChops, ImageDraw, ImageFont
    renderer.require_prepared(['mountains', 'volcanoes'])
    renderer.require_current_candidate()
    identity = renderer.implementation_identity()
    out = ROOT / 'Renderer/lab/out/mountains/body-promotion'
    accepted = ROOT / 'Renderer/lab/out/mountains/body-study/collar-triplanar'
    cases = [('mountains', 'detail', 12, 224), ('mountains', 'coastal', 12, 224),
             ('volcanoes', 'gameplay', 8, 224), ('volcanoes', 'gameplay', 12, 128)]
    records = []
    for category, case, hour, zoom in cases:
        key = f'{category}-{case}-h{hour:02}-z{zoom}'
        record = renderer.native_render(category, case, hour, zoom, out / key)
        actual = Image.open(ROOT / record['image']).convert('RGB')
        reference = Image.open(accepted / key / 'preview.png').convert('RGB')
        difference = ImageChops.difference(actual, reference)
        delta = difference.tobytes()
        pixels = sum(any(delta[i:i+3]) for i in range(0, len(delta), 3))
        record.update(category=category, changed_pixels_from_accepted=pixels,
                      maximum_channel_difference=max(v[1] for v in difference.getextrema()))
        actual.save(out / key / 'preview.png')
        sheet = Image.new('RGB', (actual.width * 2, actual.height + 40), '#18212b')
        draw = ImageDraw.Draw(sheet)
        for i, (label, frame) in enumerate([('Accepted Lab', reference), ('Current production', actual)]):
            sheet.paste(frame, (i * actual.width, 40))
            draw.text((i * actual.width + 12, 10), label, fill='white', font=ImageFont.load_default(size=18))
        sheet.save(out / key / 'comparison.png')
        print(f'{key}: {pixels} changed pixels, max channel delta {record["maximum_channel_difference"]}', flush=True)
        records.append(record)
    if renderer.implementation_identity() != identity:
        raise ValueError('Production inputs changed during visual verification')
    renderer.write(out / 'visual-verification.json', dict(outputs=records,
        implementation_identity=identity, dll_sha256=renderer.checksum(
            ROOT / 'Renderer/native/build/candidate/C3XRenderer.dll')))


if __name__ == '__main__':
    main()
