#!/usr/bin/env python3
"""Adapt terrain RGB for an opaque city-ground swatch without altering color blocks."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HEADER = 148
OPAQUE_BC3_ALPHA = b"\xff\xff\x00\x00\x00\x00\x00\x00"


def opaque_color(source: Path, target: Path) -> dict:
    source = source.resolve()
    target = target.resolve()
    source.relative_to(ROOT / "Renderer/packs")
    target.relative_to(ROOT / "Renderer/lab/out/cities")
    original = source.read_bytes()
    if original[:4] != b"DDS " or original[84:88] != b"DX10":
        raise ValueError("expected DX10 DDS terrain color")
    height, width = struct.unpack_from("<II", original, 12)
    mip_count = struct.unpack_from("<I", original, 28)[0]
    if struct.unpack_from("<I", original, 128)[0] != 78 or not width or not height or not mip_count:
        raise ValueError("expected nonempty BC3 sRGB terrain color")
    converted = bytearray(original)
    cursor = HEADER
    blocks = 0
    for mip in range(mip_count):
        mw, mh = max(1, width >> mip), max(1, height >> mip)
        count = ((mw + 3) // 4) * ((mh + 3) // 4)
        end = cursor + count * 16
        if end > len(converted):
            raise ValueError("DDS mip data is truncated")
        for block in range(count):
            at = cursor + block * 16
            converted[at:at + 8] = OPAQUE_BC3_ALPHA
        blocks += count
        cursor = end
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(converted)
    return {"source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(original).hexdigest(),
            "output": str(target.relative_to(ROOT)), "output_sha256": hashlib.sha256(converted).hexdigest(),
            "color_blocks_unchanged": True, "opaque_alpha_blocks": blocks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = opaque_color(args.source, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
