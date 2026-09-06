#!/usr/bin/env python3
"""Strict reader/writer for source-independent model-aware baked pose caches."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MAGIC = b"C3XPOSE\0"
VERSION = 1
HEADER_BYTES = 36


@dataclass(frozen=True)
class PoseCache:
    duration: float
    sample_rate: float
    frame_count: int
    bone_names: tuple[str, ...]
    matrices: tuple[float, ...]

    def sample(self, time: float, loop: bool = True) -> tuple[tuple[float, ...], ...]:
        if not math.isfinite(time):
            raise ValueError("pose-cache sample time must be finite")
        if loop:
            time %= self.duration
        else:
            time = min(self.duration, max(0.0, time))
        frame = min(self.frame_count - 1, int(round(time * self.sample_rate)))
        stride = len(self.bone_names) * 16
        start = frame * stride
        return tuple(
            tuple(self.matrices[start + bone * 16 : start + (bone + 1) * 16])
            for bone in range(len(self.bone_names))
        )


def write_pose_cache(
    path: Path,
    duration: float,
    sample_rate: float,
    bone_names: Sequence[str],
    frames: Sequence[Sequence[Sequence[float]]],
) -> PoseCache:
    """Write canonical tightly-packed world matrices and read them back strictly."""
    if not math.isfinite(duration) or not math.isfinite(sample_rate) or duration <= 0 or sample_rate <= 0:
        raise ValueError("pose cache has invalid timing")
    names = tuple(bone_names)
    if not names or len(set(names)) != len(names) or any(not name or "\0" in name for name in names):
        raise ValueError("pose cache contains an empty or duplicate bone name")
    encoded = tuple(name.encode("utf-8") for name in names)
    frame_count = len(frames)
    if frame_count < 2 or abs((frame_count - 1) / sample_rate - duration) > 0.001:
        raise ValueError("pose-cache timing is internally inconsistent")

    flat: list[float] = []
    for frame in frames:
        if len(frame) != len(names):
            raise ValueError("pose-cache frame does not contain one matrix per bone")
        for matrix in frame:
            if len(matrix) != 16 or not all(math.isfinite(value) for value in matrix):
                raise ValueError("pose cache contains an invalid matrix")
            if (
                abs(matrix[3]) > 1e-4
                or abs(matrix[7]) > 1e-4
                or abs(matrix[11]) > 1e-4
                or abs(matrix[15] - 1.0) > 1e-4
            ):
                raise ValueError("pose cache contains a non-affine row-vector matrix")
            flat.extend(matrix)

    records = HEADER_BYTES
    names_start = records + len(names) * 8
    samples = (names_start + sum(len(value) for value in encoded) + 3) & ~3
    output = bytearray(
        struct.pack(
            "<8sIffIIII",
            MAGIC,
            VERSION,
            duration,
            sample_rate,
            frame_count,
            len(names),
            records,
            samples,
        )
    )
    offset = names_start
    for value in encoded:
        output.extend(struct.pack("<II", offset, len(value)))
        offset += len(value)
    for value in encoded:
        output.extend(value)
    output.extend(bytes(samples - len(output)))
    output.extend(struct.pack(f"<{len(flat)}f", *flat))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)
    return load_pose_cache(path)


def load_pose_cache(path: Path) -> PoseCache:
    data = path.read_bytes()
    if len(data) < HEADER_BYTES:
        raise ValueError("pose cache is truncated")
    magic, version, duration, sample_rate, frames, bones, records, samples = struct.unpack_from(
        "<8sIffIIII", data, 0
    )
    if magic != MAGIC or version != VERSION or records != HEADER_BYTES:
        raise ValueError("unsupported pose-cache header")
    if not math.isfinite(duration) or not math.isfinite(sample_rate) or duration <= 0 or sample_rate <= 0:
        raise ValueError("pose cache has invalid timing")
    if frames < 2 or bones < 1 or frames > 1_000_000 or bones > 100_000:
        raise ValueError("pose cache has invalid dimensions")
    if abs((frames - 1) / sample_rate - duration) > 0.001:
        raise ValueError("pose-cache timing is internally inconsistent")
    records_end = records + bones * 8
    expected_end = samples + frames * bones * 16 * 4
    if records_end > samples or samples % 4 or expected_end != len(data):
        raise ValueError("pose-cache sections are invalid")
    names = []
    for index in range(bones):
        offset, size = struct.unpack_from("<II", data, records + index * 8)
        if size < 1 or offset < records_end or offset + size > samples:
            raise ValueError(f"pose-cache bone {index} has an invalid name range")
        try:
            name = data[offset : offset + size].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"pose-cache bone {index} name is not UTF-8") from error
        if not name or "\0" in name or name in names:
            raise ValueError("pose cache contains an empty or duplicate bone name")
        names.append(name)
    matrices = struct.unpack_from(f"<{frames * bones * 16}f", data, samples)
    if not all(math.isfinite(value) for value in matrices):
        raise ValueError("pose cache contains a non-finite matrix")
    for start in range(0, len(matrices), 16):
        if (
            abs(matrices[start + 3]) > 1e-4
            or abs(matrices[start + 7]) > 1e-4
            or abs(matrices[start + 11]) > 1e-4
            or abs(matrices[start + 15] - 1.0) > 1e-4
        ):
            raise ValueError("pose cache contains a non-affine row-vector matrix")
    return PoseCache(duration, sample_rate, frames, tuple(names), tuple(matrices))


def validate_skeleton_binding(cache: PoseCache, skeleton: dict) -> dict[str, float | int]:
    names = tuple(bone["name"] for bone in skeleton["bones"])
    if cache.bone_names != names:
        raise ValueError("pose-cache bones do not exactly match the normalized skeleton")
    from Renderer.tools.asset_compiler import normalized_skin

    rest_world = normalized_skin.world_matrices(
        skeleton, [bone["local"] for bone in skeleton["bones"]]
    )
    first = cache.sample(0.0, False)
    maximum = max(abs(a - b) for expected, actual in zip(rest_world, first) for a, b in zip(expected, actual))
    # An authored clip need not begin in the bind pose. Exact ordered name
    # equality is the binding proof; this delta is useful diagnostic evidence.
    return {
        "bones": len(names),
        "frames": cache.frame_count,
        "maximum_first_frame_rest_matrix_delta": maximum,
    }
