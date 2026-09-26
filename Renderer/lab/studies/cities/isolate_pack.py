#!/usr/bin/env python3
"""Make an isolated one-building city pack for staged D3D fidelity probes.

Only the selected Lab composition's instance list changes. All original mesh,
UV, material, texture and shader data remain byte-for-byte in the candidate.
This is diagnostic output for headless previews, never a runtime pack recipe.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def take(self, count: int) -> bytes:
        end = self.at + count
        if count < 0 or end > len(self.data):
            raise ValueError("truncated city pack")
        result = self.data[self.at:end]
        self.at = end
        return result

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def string(self) -> str:
        return self.take(self.u32()).decode("utf-8")


def isolate(data: bytes, culture: int, era: int, size: int,
            capital: bool, scale: float = 1.0) -> bytes:
    reader = Reader(data)
    magic = reader.take(8)
    if magic not in (b"C3XCITY2", b"C3XCITY3"):
        raise ValueError("unsupported city pack")
    material_count, model_count, template_count = (reader.u32() for _ in range(3))
    if not (material_count and model_count and template_count):
        raise ValueError("empty city pack")
    for _ in range(material_count):
        reader.take(12)
        for _ in range(7):
            reader.string()
    for _ in range(model_count):
        parts = reader.u32()
        reader.take(24)
        reader.take(reader.u32() * 8)
        for _ in range(parts):
            reader.u32()
            vertices, indices = reader.u32(), reader.u32()
            reader.take(vertices * 72 + indices * 4)
    chunks = [data[:reader.at]]
    changed = 0
    for _ in range(template_count):
        start = reader.at
        key = tuple(reader.u32() for _ in range(5))
        authority = reader.string()
        reader.take(16)
        count_offset = reader.at
        instance_count = reader.u32()
        instances = []
        for _ in range(instance_count):
            first = reader.at
            reader.u32()
            is_palace = bool(reader.u32())
            reader.take(32)
            reader.take(reader.u32() * 48)
            instances.append((is_palace, data[first:reader.at]))
        instance_end = reader.at
        if reader.u32():
            reader.u32()
            reader.take(24)
            vertices, indices = reader.u32(), reader.u32()
            reader.take(vertices * 12 + indices * 4)
        if magic == b"C3XCITY3" and reader.u32():
            reader.u32()
            reader.take(24)
        end = reader.at
        if (key[:4] == (culture, era, size, int(capital)) and
                authority.startswith("lab-fixed-")):
            choices = [raw for is_palace, raw in instances if is_palace == capital]
            if not choices:
                raise ValueError("selected composition has no requested building")
            selected = bytearray(choices[0])
            original_scale = struct.unpack_from("<f", selected, 8)[0]
            struct.pack_into("<f", selected, 8, original_scale * scale)
            chunks.append(data[start:count_offset] + struct.pack("<I", 1) +
                          selected + data[instance_end:end])
            changed += 1
        else:
            chunks.append(data[start:end])
    if reader.at != len(data) or changed != 1:
        raise ValueError(f"city pack parse or composition selection failed: {changed}")
    return b"".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--culture", type=int, required=True)
    parser.add_argument("--era", type=int, required=True)
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--capital", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.scale <= 0 or args.scale > 4:
        raise ValueError("diagnostic scale must be within (0, 4]")
    candidate = isolate(args.source.read_bytes(), args.culture, args.era,
                        args.size, args.capital, args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(candidate)
    print("wrote isolated city candidate", args.output, len(candidate))


if __name__ == "__main__":
    main()
