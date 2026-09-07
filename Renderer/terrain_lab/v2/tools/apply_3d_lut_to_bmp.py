#!/usr/bin/env python3
"""Apply an uncompressed legacy DDS RGBA8 3D LUT to a 32-bit BMP.

This is an offline diagnostic for source-authored color transforms. It keeps
the renderer and runtime source-agnostic: callers provide both paths and the
output remains an ordinary BMP.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def read_lut(path: Path) -> tuple[int, bytes]:
    data = path.read_bytes()
    if len(data) < 128 or data[:4] != b"DDS " or struct.unpack_from("<I", data, 4)[0] != 124:
        raise ValueError("expected a legacy DDS header")
    height, width = struct.unpack_from("<II", data, 12)
    depth = struct.unpack_from("<I", data, 24)[0]
    pf_size, pf_flags, fourcc, bits = struct.unpack_from("<IIII", data, 76)
    masks = struct.unpack_from("<IIII", data, 92)
    if (
        width != height or depth != width or width < 2 or pf_size != 32
        or pf_flags & 0x40 == 0 or fourcc != 0 or bits != 32
        or masks != (0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000)
    ):
        raise ValueError("expected a cubic uncompressed RGBA8 DDS volume")
    payload = data[128:]
    if len(payload) != width * width * width * 4:
        raise ValueError("DDS LUT payload size mismatch")
    return width, payload


def sample_lut(size: int, lut: bytes, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    coordinates = [value / 255.0 * (size - 1) for value in rgb]
    low = [math.floor(value) for value in coordinates]
    high = [min(size - 1, value + 1) for value in low]
    fractions = [coordinates[axis] - low[axis] for axis in range(3)]
    result = [0.0, 0.0, 0.0]
    for bz in range(2):
        z = high[2] if bz else low[2]
        wz = fractions[2] if bz else 1.0 - fractions[2]
        for by in range(2):
            y = high[1] if by else low[1]
            wy = fractions[1] if by else 1.0 - fractions[1]
            for bx in range(2):
                x = high[0] if bx else low[0]
                wx = fractions[0] if bx else 1.0 - fractions[0]
                offset = ((z * size + y) * size + x) * 4
                weight = wx * wy * wz
                for channel in range(3):
                    result[channel] += lut[offset + channel] * weight
    return tuple(max(0, min(255, round(value))) for value in result)


def apply(input_path: Path, lut_path: Path, output_path: Path) -> None:
    image = bytearray(input_path.read_bytes())
    if len(image) < 54 or image[:2] != b"BM":
        raise ValueError("expected a BMP input")
    pixel_offset = struct.unpack_from("<I", image, 10)[0]
    dib_size = struct.unpack_from("<I", image, 14)[0]
    width, height = struct.unpack_from("<ii", image, 18)
    planes, bits, compression = struct.unpack_from("<HHI", image, 26)
    if dib_size < 40 or width <= 0 or height == 0 or planes != 1 or bits != 32 or compression != 0:
        raise ValueError("expected an uncompressed 32-bit BMP")
    pixel_bytes = width * abs(height) * 4
    if pixel_offset + pixel_bytes != len(image):
        raise ValueError("BMP payload size mismatch")
    size, lut = read_lut(lut_path)
    for offset in range(pixel_offset, len(image), 4):
        blue, green, red = image[offset : offset + 3]
        mapped = sample_lut(size, lut, (red, green, blue))
        image[offset : offset + 3] = bytes((mapped[2], mapped[1], mapped[0]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply(args.input, args.lut, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
