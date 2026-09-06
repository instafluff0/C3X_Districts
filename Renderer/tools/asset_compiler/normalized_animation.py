#!/usr/bin/env python3
"""Strict reader and deterministic sampler for source-independent C3X clips."""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MAGIC = b"C3XANIM\0"
VERSION = 1
HEADER = struct.Struct("<8sIIffIIIIIIII")
GROUP_RECORD = struct.Struct("<IIII")
TRACK_RECORD = struct.Struct("<IIIIIIII")
IDENTITY = 0
CONSTANT = 1
SAMPLED = 2
POSITION_DIMENSION = 3
ORIENTATION_DIMENSION = 4
SCALE_SHEAR_DIMENSION = 9


@dataclass(frozen=True)
class Channel:
    mode: int
    dimension: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class TransformTrack:
    name: str
    flags: int
    position: Channel
    orientation: Channel
    scale_shear: Channel


@dataclass(frozen=True)
class TrackGroup:
    name: str
    tracks: tuple[TransformTrack, ...]


@dataclass(frozen=True)
class Transform:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    scale_shear: tuple[float, float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class AnimationClip:
    duration: float
    sample_rate: float
    frame_count: int
    groups: tuple[TrackGroup, ...]

    def track(self, group_index: int, track_name: str) -> TransformTrack:
        if not 0 <= group_index < len(self.groups):
            raise IndexError(f"group index {group_index} is out of range")
        matches = [track for track in self.groups[group_index].tracks if track.name == track_name]
        if not matches:
            raise KeyError(track_name)
        return matches[0]

    def sample(self, group_index: int, track_name: str, time: float, *, loop: bool) -> Transform:
        if not math.isfinite(time):
            raise ValueError("sample time must be finite")
        if loop:
            local_time = time % self.duration
        else:
            local_time = min(self.duration, max(0.0, time))
        frame_position = local_time * (self.frame_count - 1) / self.duration
        first = min(self.frame_count - 1, int(math.floor(frame_position)))
        second = min(self.frame_count - 1, first + 1)
        alpha = frame_position - first
        track = self.track(group_index, track_name)
        position = _sample_channel(track.position, first, second, alpha, (0.0, 0.0, 0.0))
        orientation = _sample_orientation(track.orientation, first, second, alpha)
        scale_shear = _sample_channel(
            track.scale_shear,
            first,
            second,
            alpha,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        return Transform(position, orientation, scale_shear)


def _sample_channel(
    channel: Channel,
    first: int,
    second: int,
    alpha: float,
    identity: Sequence[float],
) -> tuple[float, ...]:
    if channel.mode == IDENTITY:
        return tuple(identity)
    if channel.mode == CONSTANT:
        return channel.values
    offset_a = first * channel.dimension
    offset_b = second * channel.dimension
    return tuple(
        channel.values[offset_a + component] * (1.0 - alpha)
        + channel.values[offset_b + component] * alpha
        for component in range(channel.dimension)
    )


def _normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    length_squared = sum(value * value for value in values)
    if length_squared <= 1.0e-20:
        raise ValueError("animation contains a zero-length orientation")
    inverse_length = 1.0 / math.sqrt(length_squared)
    return tuple(value * inverse_length for value in values)  # type: ignore[return-value]


def _sample_orientation(
    channel: Channel, first: int, second: int, alpha: float
) -> tuple[float, float, float, float]:
    if channel.mode == IDENTITY:
        return (0.0, 0.0, 0.0, 1.0)
    if channel.mode == CONSTANT:
        return _normalize_quaternion(channel.values)
    offset_a = first * ORIENTATION_DIMENSION
    offset_b = second * ORIENTATION_DIMENSION
    a = channel.values[offset_a : offset_a + ORIENTATION_DIMENSION]
    b = channel.values[offset_b : offset_b + ORIENTATION_DIMENSION]
    if sum(x * y for x, y in zip(a, b)) < 0.0:
        b = tuple(-value for value in b)
    return _normalize_quaternion(
        tuple(x * (1.0 - alpha) + y * alpha for x, y in zip(a, b))
    )


def _read_name(data: bytes, offset: int, byte_count: int, string_end: int) -> str:
    if byte_count <= 0 or offset < HEADER.size or offset + byte_count > string_end:
        raise ValueError("animation name range is outside the string table")
    raw = data[offset : offset + byte_count]
    if b"\0" in raw:
        raise ValueError("animation names must not contain NUL bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("animation name is not valid UTF-8") from error


def _read_channel(
    data: bytes,
    mode: int,
    dimension: int,
    frame_count: int,
    offset: int,
    expected_offset: int,
    data_end: int,
) -> tuple[Channel, int]:
    if mode == IDENTITY:
        if offset != 0:
            raise ValueError("identity channel must have a zero data offset")
        return Channel(mode, dimension, ()), expected_offset
    if mode not in (CONSTANT, SAMPLED):
        raise ValueError(f"unsupported animation channel mode {mode}")
    value_count = dimension if mode == CONSTANT else dimension * frame_count
    byte_count = value_count * 4
    if offset != expected_offset or offset + byte_count > data_end:
        raise ValueError("animation channel data is not tightly packed in declared order")
    values = struct.unpack_from(f"<{value_count}f", data, offset)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("animation channel contains a non-finite value")
    return Channel(mode, dimension, values), expected_offset + byte_count


def load_clip(path: Path | str) -> AnimationClip:
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("animation is shorter than the fixed header")
    (
        magic,
        version,
        flags,
        duration,
        sample_rate,
        frame_count,
        group_count,
        track_count,
        string_bytes,
        group_table_offset,
        track_table_offset,
        data_offset,
        data_bytes,
    ) = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("animation magic is not C3XANIM")
    if version != VERSION:
        raise ValueError(f"unsupported animation version {version}")
    if flags != 0:
        raise ValueError(f"unsupported animation flags 0x{flags:x}")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("animation duration must be positive and finite")
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("animation sample rate must be positive and finite")
    if frame_count < 2 or group_count < 1 or track_count < 1:
        raise ValueError("animation must contain at least two frames, one group, and one track")
    string_end = HEADER.size + string_bytes
    if group_table_offset != (string_end + 3) & ~3:
        raise ValueError("animation group table is not at the canonical aligned offset")
    if any(data[string_end:group_table_offset]):
        raise ValueError("animation string-table padding must be zero")
    if track_table_offset != group_table_offset + group_count * GROUP_RECORD.size:
        raise ValueError("animation track table offset is inconsistent")
    if data_offset != track_table_offset + track_count * TRACK_RECORD.size:
        raise ValueError("animation data offset is inconsistent")
    data_end = data_offset + data_bytes
    if data_end != len(data):
        raise ValueError("animation data length does not match the file size")

    group_records: list[tuple[str, int, int]] = []
    expected_first_track = 0
    for index in range(group_count):
        name_offset, name_bytes, first_track, group_tracks = GROUP_RECORD.unpack_from(
            data, group_table_offset + index * GROUP_RECORD.size
        )
        name = _read_name(data, name_offset, name_bytes, string_end)
        if first_track != expected_first_track or group_tracks < 1:
            raise ValueError("animation groups must cover non-empty contiguous track ranges")
        expected_first_track += group_tracks
        if expected_first_track > track_count:
            raise ValueError("animation group track range exceeds the track table")
        group_records.append((name, first_track, group_tracks))
    if expected_first_track != track_count:
        raise ValueError("animation groups do not cover every track")

    tracks: list[TransformTrack] = []
    expected_channel_offset = data_offset
    for index in range(track_count):
        (
            name_offset,
            name_bytes,
            track_flags,
            modes,
            position_offset,
            orientation_offset,
            scale_shear_offset,
            reserved,
        ) = TRACK_RECORD.unpack_from(data, track_table_offset + index * TRACK_RECORD.size)
        if track_flags != 0 or reserved != 0 or modes & ~0x3F:
            raise ValueError("animation track has non-zero reserved fields")
        name = _read_name(data, name_offset, name_bytes, string_end)
        position, expected_channel_offset = _read_channel(
            data, modes & 3, POSITION_DIMENSION, frame_count, position_offset, expected_channel_offset, data_end
        )
        orientation, expected_channel_offset = _read_channel(
            data, (modes >> 2) & 3, ORIENTATION_DIMENSION, frame_count, orientation_offset, expected_channel_offset, data_end
        )
        scale_shear, expected_channel_offset = _read_channel(
            data, (modes >> 4) & 3, SCALE_SHEAR_DIMENSION, frame_count, scale_shear_offset, expected_channel_offset, data_end
        )
        tracks.append(TransformTrack(name, track_flags, position, orientation, scale_shear))
    if expected_channel_offset != data_end:
        raise ValueError("animation has unreferenced channel bytes")

    groups: list[TrackGroup] = []
    for group_name, first_track, group_tracks in group_records:
        selected = tuple(tracks[first_track : first_track + group_tracks])
        names = [track.name for track in selected]
        if len(names) != len(set(names)):
            raise ValueError("animation contains duplicate track names within a group")
        groups.append(TrackGroup(group_name, selected))
    return AnimationClip(duration, sample_rate, frame_count, tuple(groups))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--time", type=float)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--track")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    clip = load_clip(args.clip)
    result: dict[str, object] = {
        "duration": clip.duration,
        "sample_rate": clip.sample_rate,
        "frame_count": clip.frame_count,
        "group_count": len(clip.groups),
        "track_count": sum(len(group.tracks) for group in clip.groups),
        "groups": [
            {"index": index, "name": group.name, "tracks": [track.name for track in group.tracks]}
            for index, group in enumerate(clip.groups)
        ],
    }
    if args.time is not None or args.track is not None:
        if args.time is None or args.track is None:
            parser.error("--time and --track must be supplied together")
        transform = clip.sample(args.group, args.track, args.time, loop=args.loop)
        result["sample"] = {
            "group": args.group,
            "track": args.track,
            "time": args.time,
            "loop": args.loop,
            "position": transform.position,
            "orientation": transform.orientation,
            "scale_shear": transform.scale_shear,
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
