#!/usr/bin/env python3
"""Build matched on/off and amplified cast-shadow evidence for the pickup."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont


ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / "Renderer/terrain_lab/v2"


def pin(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control", type=Path,
        default=V2 / "audits/beauty/out/source-fidelity-r13/shadow-control/h12-z1-pan00.png",
    )
    parser.add_argument(
        "--candidate", type=Path,
        default=V2 / "audits/beauty/out/source-fidelity-r13/inland/h12-z1-pan00.png",
    )
    parser.add_argument(
        "--output", type=Path,
        default=V2 / "audits/beauty/out/source-fidelity-r13/inland/shadow-evidence.png",
    )
    args = parser.parse_args()
    for path in (args.control, args.candidate, args.output.parent):
        path.resolve().relative_to(ROOT / "Renderer")

    control = Image.open(args.control).convert("RGB")
    candidate = Image.open(args.candidate).convert("RGB")
    if control.size != candidate.size:
        raise ValueError("matched shadow evidence must preserve the same viewport")

    # Only positive darkening is a cast/self-shadow contribution. Amplify it
    # for diagnosis without changing the accepted candidate pixels.
    darkness = ImageChops.subtract(control, candidate)
    amplified = ImageEnhance.Brightness(darkness).enhance(4.0)
    luminance = darkness.convert("L")
    histogram = luminance.histogram()
    changed_pixels = sum(histogram[7:])
    max_delta = max(index for index, count in enumerate(histogram) if count)

    header = 48
    canvas = Image.new("RGB", (control.width * 3, control.height + header), "#151515")
    canvas.paste(control, (0, header))
    canvas.paste(candidate, (control.width, header))
    canvas.paste(amplified, (control.width * 2, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 10), "matched control: cast/self shadow lookup disabled", fill="white", font=font)
    draw.text((control.width + 8, 10), "candidate: one Q6 field for terrain + mountains + trees", fill="white", font=font)
    draw.text((control.width * 2 + 8, 10), "4x positive darkening: actual shared-field contribution", fill="white", font=font)
    # The checked Q6/native coordinate probe defines noon's canonical ground
    # delta as positive screen-X and negative screen-Y: one up-right cast.
    arrow_start = (control.width + 1040, 37)
    arrow_end = (arrow_start[0] + 58, arrow_start[1] - 22)
    draw.line((arrow_start, arrow_end), fill="#55e8ff", width=4)
    draw.polygon((arrow_end, (arrow_end[0] - 13, arrow_end[1] - 1),
                  (arrow_end[0] - 8, arrow_end[1] - 11)), fill="#55e8ff")
    draw.text((arrow_start[0] - 2, 6), "common noon cast", fill="#55e8ff", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    crop_box = (45, 175, 930, 650)
    crop_control = control.crop(crop_box).resize((1770, 950), Image.Resampling.NEAREST)
    crop_candidate = candidate.crop(crop_box).resize((1770, 950), Image.Resampling.NEAREST)
    crop_diff = amplified.crop(crop_box).resize((1770, 950), Image.Resampling.NEAREST)
    detail_path = args.output.with_name("shadow-detail.png")
    detail = Image.new("RGB", (1770 * 3, 998), "#151515")
    detail.paste(crop_control, (0, 48))
    detail.paste(crop_candidate, (1770, 48))
    detail.paste(crop_diff, (3540, 48))
    detail_draw = ImageDraw.Draw(detail)
    detail_draw.text((8, 12), "2x nearest: shadow lookup off", fill="white", font=font)
    detail_draw.text((1778, 12), "2x nearest: shared shadows on", fill="white", font=font)
    detail_draw.text((3548, 12), "2x nearest: shadow contribution amplified", fill="white", font=font)
    detail.save(detail_path)
    evidence = {
        "schema": "c3x.source_fidelity_shadow_evidence.v1",
        "native_viewport": list(control.size),
        "matched_geometry_and_materials": True,
        "control_difference": "Q6 shadow receive flag only",
        "shared_light_contract": "ShadowL drives direct face lighting and shadow projection",
        "noon_screen_cast_direction": "up-right",
        "changed_pixels_delta_gt_6": changed_pixels,
        "changed_fraction": changed_pixels / (control.width * control.height),
        "maximum_luma_darkening": max_delta,
        "control": pin(args.control),
        "candidate": pin(args.candidate),
        "comparison": pin(args.output),
        "detail": pin(detail_path),
        "cities_rendered": False,
        "cities_changed": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(args.output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
