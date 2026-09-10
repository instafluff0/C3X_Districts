#!/usr/bin/env python3
"""Arrange actual D3D Lab frames; no color edits, filtering or AI retouching."""
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'Renderer/lab/out/mountains/zebra-study'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hour', type=int, default=12)
    parser.add_argument('--zoom', type=int, default=128)
    parser.add_argument('--cases', nargs='+', default=['detail', 'gameplay', 'coastal'])
    parser.add_argument('--proposal', choices=['summit', 'snowcaps', 'micro-relief', 'micro-relief-balanced'], default='summit')
    parser.add_argument('--control', choices=['current', 'snowcaps'], default='current')
    args = parser.parse_args()
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=13)
    for case in args.cases:
        key = f'{case}-h{args.hour:02}-z{args.zoom}'
        labels = {'summit': 'Same skin - summit blend', 'snowcaps': 'Same skin - fuller white caps',
                  'micro-relief': 'New Lab - fine rock relief',
                  'micro-relief-balanced': 'New Lab - balanced rock relief'}
        variants = [(args.control, 'Previous Lab - snow caps' if args.control == 'snowcaps' else 'Current Civ 5 skin'),
                    (args.proposal, labels[args.proposal])]
        frames = [Image.open(OUT / v / key / 'preview.png').convert('RGB') for v, _ in variants]
        width, height = frames[0].size
        sheet = Image.new('RGB', (width * 2, height + 78), '#17202a')
        draw = ImageDraw.Draw(sheet)
        for index, ((_, label), frame) in enumerate(zip(variants, frames)):
            draw.text((index * width + 16, 13), label, font=font, fill='white')
            sheet.paste(frame, (index * width, 44))
        draw.text((16, height + 54), f'{case.capitalize()} | {args.hour:02}:00 | zoom {args.zoom} | Same skin, geometry, lighting and ground blend | Lab only',
                  font=small, fill='#c7d4df')
        sheet.save(OUT / f'compare-{args.proposal}-{key}.png')
    # One-variable diagnostic: the identical height/specular pair makes this
    # an albedo-layer test, with all crevice and grain enhancements retained.
    key = f'detail-h{args.hour:02}-z{args.zoom}'
    if (OUT / 'no-top' / key / 'preview.png').is_file():
        sheet = Image.new('RGB', (1280, 524), '#17202a')
        draw = ImageDraw.Draw(sheet)
        for index, (variant, label) in enumerate([
                ('current', 'Current'), ('no-top', 'Only disable patchy-white top layer')]):
            sheet.paste(Image.open(OUT / variant / key / 'preview.png'), (index * 640, 44))
            draw.text((index * 640 + 16, 13), label, font=font, fill='white')
        sheet.save(OUT / f'diagnosis-{key}.png')


if __name__ == '__main__':
    main()
