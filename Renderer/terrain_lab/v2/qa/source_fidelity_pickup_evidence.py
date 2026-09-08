#!/usr/bin/env python3
"""Build the lossless before/after sheet for the consolidated source-fidelity pickup."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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
        "--baseline",
        type=Path,
        default=V2 / "audits/beauty/out/source-fidelity-r4/inland/h12-z1-pan00.png",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=V2 / "audits/beauty/out/source-fidelity-r11/inland/h12-z1-pan00.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=V2 / "audits/beauty/out/source-fidelity-r11/inland/comparison.png",
    )
    args = parser.parse_args()
    for path in (args.baseline, args.candidate, args.output.parent):
        path.resolve().relative_to(ROOT / "Renderer")
    before = Image.open(args.baseline).convert("RGB")
    after = Image.open(args.candidate).convert("RGB")
    if before.size != after.size:
        raise ValueError("before and after must preserve the same native viewport")
    header = 36
    canvas = Image.new("RGB", (before.width * 2, before.height + header), "#151515")
    canvas.paste(before, (0, header))
    canvas.paste(after, (before.width, header))
    labels = (
        "previous composed Lab scene",
        "consolidated source-fidelity pickup | cities unchanged",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 12), labels[0], fill="white", font=font)
    draw.text((before.width + 8, 12), labels[1], fill="white", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    evidence = {
        "schema": "c3x.source_fidelity_pickup_evidence.v1",
        "native_viewport": list(before.size),
        "resampling": False,
        "baseline": pin(args.baseline),
        "candidate": pin(args.candidate),
        "comparison": pin(args.output),
        "cities_changed": False,
    }
    evidence_path = args.output.with_name("comparison.json")
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(args.output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
