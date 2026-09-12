#!/usr/bin/env python3
"""Arrange unfiltered, native-resolution D3D frames for mountain body review."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[4] / 'Renderer/lab/out/mountains/body-study'
FONT = ImageFont.load_default(size=20)
SMALL = ImageFont.load_default(size=14)


def frame(variant, case='gameplay', zoom=224, hour=12, category='volcanoes'):
    return Image.open(OUT / variant / f'{category}-{case}-h{hour:02}-z{zoom}' / 'preview.png').convert('RGB')


def comparison(case, zoom, hour=12, category='volcanoes', control='current'):
    images = [frame(v, case, zoom, hour, category) for v in (control, 'slope-rock')]
    width, height = images[0].size
    sheet = Image.new('RGB', (width * 2, height + 80), '#18212b')
    draw = ImageDraw.Draw(sheet)
    labels = ('Current mountains' if control == 'current' else 'First balanced test',
              'Revised Lab - rocky base, cleaner upper slopes')
    for i, (label, im) in enumerate(zip(labels, images)):
        draw.text((i * width + 16, 12), label, font=FONT, fill='white')
        sheet.paste(im, (i * width, 42))
    draw.text((16, height + 54), f'{case.capitalize()} | {hour:02}:00 | tile width {zoom} | Same geometry, textures, snow masks, ground blend and lighting | Lab only', font=SMALL, fill='#cbd5df')
    sheet.save(OUT / f'comparison-{category}-{case}-h{hour:02}-z{zoom}-{control}.png')


def main():
    for case, zoom, hour in [('gameplay', 224, 12), ('gameplay', 128, 12), ('gameplay', 224, 8)]:
        if (OUT / 'slope-rock' / f'volcanoes-{case}-h{hour:02}-z{zoom}' / 'preview.png').exists():
            comparison(case, zoom, hour)
    for case in ('coastal', 'detail'):
        for control in ('current', 'balanced-rock'):
            comparison(case, 224, category='mountains', control=control)
    sheet = Image.new('RGB', (1020, 345), '#18212b')
    draw = ImageDraw.Draw(sheet)
    for i, (variant, label) in enumerate([('balanced-rock', 'First balanced test'),
                                        ('slope-rock', 'Revised - rocky base, cleaner upper slopes')]):
        sheet.paste(frame(variant, 'coastal', category='mountains').crop((130, 100, 640, 400)), (i * 510, 40))
        draw.text((i * 510 + 12, 10), label, font=FONT, fill='white')
    sheet.save(OUT / 'coastal-focused.png')
    variants = [('current', 'Current'), ('balanced-rock', 'Proposed balance'),
                ('slope-rock', 'Revised slopes')]
    sheet = Image.new('RGB', (840, 300), '#18212b')
    draw = ImageDraw.Draw(sheet)
    for i, (variant, label) in enumerate(variants):
        sheet.paste(frame(variant).crop((0, 0, 280, 240)), (i * 280, 38))
        draw.text((i * 280 + 10, 10), label, font=FONT, fill='white')
    draw.text((10, 282), 'Native-resolution crops; no image sharpening, filtering or retouching.', font=SMALL, fill='#cbd5df')
    sheet.save(OUT / 'body-options.png')
    for category, case, hour, zoom in [('volcanoes', 'gameplay', 8, 224),
                                      ('volcanoes', 'gameplay', 12, 224),
                                      ('volcanoes', 'gameplay', 12, 128),
                                      ('mountains', 'coastal', 12, 224),
                                      ('mountains', 'detail', 12, 224)]:
        key=f'{category}-{case}-h{hour:02}-z{zoom}'
        if not (OUT / 'collar-triplanar' / key / 'preview.png').exists():
            continue
        sheet=Image.new('RGB',(1280,550),'#18212b')
        draw=ImageDraw.Draw(sheet)
        for i,(variant,label) in enumerate([('slope-rock','Before - stretched ground projection'),
                                            ('collar-triplanar','Lab fix - ground texture follows the slope')]):
            sheet.paste(frame(variant,case,zoom,hour,category),(i*640,42))
            draw.text((i*640+12,12),label,font=FONT,fill='white')
        draw.text((12,529),'Same upper rock, snow, terrain blend and geometry | Native D3D frames | Lab only',font=SMALL,fill='#cbd5df')
        sheet.save(OUT/f'base-fix-{key}.png')


if __name__ == '__main__':
    main()
