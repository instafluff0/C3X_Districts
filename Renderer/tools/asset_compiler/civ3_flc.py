"""Read Civ III unit FLC animations (offline asset tooling only).

Ported from the C3X Editor preview decoder (`../C3X_Editor/src/artPreview.js`,
documented in `../C3X_Editor/docs/FlcReference.md`). Civ III stores a cropped
image at (x_offset, y_offset) inside an xs_orig x ys_orig frame (240x240 for
units). Frames are grouped by direction (SW, S, SE, E, NE, N, NW, W), each block
holding anim_length frames plus one ring frame.

Palette conventions: 0..63 civ colour (replaced from ntpNN.pcx), 224..239
smoke ramp, 240..254 shadow ramp, 255 transparent.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAME, COLOR_256, DELTA_FLC, DELTA_FLI, BLACK, BYTE_RUN, FLI_COPY = 0xF1FA, 4, 7, 12, 13, 15, 16
DIRECTIONS = ("SW", "S", "SE", "E", "NE", "N", "NW", "W")


@dataclass
class Flc:
    width: int
    height: int
    x_offset: int
    y_offset: int
    full_width: int
    full_height: int
    directions: int
    anim_length: int
    palette: np.ndarray  # 256x3 uint8
    frames: list  # list of HxW uint8 index arrays (cropped)

    def direction_frame(self, direction: int, frame: int = 0) -> np.ndarray:
        if self.directions > 1 and self.anim_length:
            index = (self.anim_length + 1) * direction + frame
        else:
            index = frame
        return self.frames[min(index, len(self.frames) - 1)]


def _s8(value: int) -> int:
    return value - 256 if value > 127 else value


def _color256(payload: bytes, palette: np.ndarray) -> None:
    packets, = struct.unpack_from("<H", payload, 0)
    p, index = 2, 0
    for _ in range(packets):
        if p + 2 > len(payload):
            break
        index += payload[p]
        count = payload[p + 1] or 256
        p += 2
        for _ in range(count):
            if p + 3 > len(payload) or index >= 256:
                break
            palette[index] = tuple(payload[p:p + 3])
            p += 3
            index += 1


def _byte_run(payload: bytes, w: int, h: int) -> bytearray:
    out = bytearray(w * h)
    p = 0
    for y in range(h):
        if p >= len(payload):
            break
        p += 1
        x, row = 0, y * w
        while x < w and p < len(payload):
            n = _s8(payload[p]); p += 1
            if n >= 0:
                run = min(n, w - x)
                out[row + x:row + x + run] = bytes([payload[p]]) * run
                p += 1
            else:
                run = min(-n, w - x, len(payload) - p)
                out[row + x:row + x + run] = payload[p:p + run]
                p += run
            x += run
    return out


def _delta_fli(payload: bytes, frame: bytearray, w: int, h: int) -> None:
    y, lines = struct.unpack_from("<HH", payload, 0)
    p = 4
    while lines > 0 and y < h and p < len(payload):
        packets = payload[p]; p += 1
        x, row = 0, y * w
        for _ in range(packets):
            if p + 2 > len(payload):
                break
            x += payload[p]; n = _s8(payload[p + 1]); p += 2
            if n >= 0:
                write = max(0, min(n, len(payload) - p, w - x))
                frame[row + x:row + x + write] = payload[p:p + write]
                p += n; x += n
            else:
                run = min(-n, w - x)
                frame[row + x:row + x + run] = bytes([payload[p]]) * run
                p += 1; x += run
        y += 1; lines -= 1


def _delta_flc(payload: bytes, frame: bytearray, w: int, h: int) -> None:
    lines, = struct.unpack_from("<H", payload, 0)
    p, y = 2, 0
    while lines > 0 and y < h and p + 2 <= len(payload):
        op, = struct.unpack_from("<h", payload, p); p += 2
        if op < 0:
            if (op & 0xC000) == 0xC000:
                y += -op
            elif (op & 0xC000) == 0x8000:
                frame[y * w + w - 1] = op & 0xFF
            continue
        x, row = 0, y * w
        for _ in range(op):
            if p + 2 > len(payload):
                break
            x += payload[p]; n = _s8(payload[p + 1]); p += 2
            if n >= 0:
                count = n * 2
                write = max(0, min(count, len(payload) - p, w - x))
                frame[row + x:row + x + write] = payload[p:p + write]
                p += count; x += count
            else:
                pair = payload[p:p + 2]; p += 2
                for _ in range(-n):
                    if x + 1 >= w:
                        break
                    frame[row + x:row + x + 2] = pair
                    x += 2
        y += 1; lines -= 1


def read(path: Path, limit: int | None = None) -> Flc:
    data = Path(path).read_bytes()
    if len(data) < 128 or struct.unpack_from("<H", data, 4)[0] not in (0xAF12, 0xAF11):
        raise ValueError(f"not an FLC: {path}")
    w, h = struct.unpack_from("<HH", data, 8)
    directions, anim_length, x_offset, y_offset, full_w, full_h = struct.unpack_from("<6H", data, 96)
    palette = np.zeros((256, 3), np.uint8)
    frames, frame, offset = [], bytearray(w * h), 128
    while offset + 6 <= len(data):
        size, kind = struct.unpack_from("<IH", data, offset)
        if size < 6 or offset + size > len(data):
            break
        if kind == FRAME:
            count, = struct.unpack_from("<H", data, offset + 6)
            sub, touched, end = offset + 16, False, offset + size
            for _ in range(count):
                if sub + 6 > end:
                    break
                sub_size, sub_kind = struct.unpack_from("<IH", data, sub)
                if sub_kind == COLOR_256 and (sub_size < 6 or sub + sub_size > end):
                    sub_size = min(778, end - sub)  # Civ III malformed colour chunk sizes
                if sub_size < 6 or sub + sub_size > end:
                    break
                payload = data[sub + 6:sub + sub_size]
                if sub_kind == COLOR_256:
                    _color256(payload, palette)
                elif sub_kind == BYTE_RUN:
                    frame = _byte_run(payload, w, h); touched = True
                elif sub_kind == FLI_COPY and len(payload) >= w * h:
                    frame = bytearray(payload[:w * h]); touched = True
                elif sub_kind == BLACK:
                    frame = bytearray(w * h); touched = True
                elif sub_kind == DELTA_FLI:
                    _delta_fli(payload, frame, w, h); touched = True
                elif sub_kind == DELTA_FLC:
                    _delta_flc(payload, frame, w, h); touched = True
                sub += sub_size
            if touched:
                frames.append(np.frombuffer(bytes(frame), np.uint8).reshape(h, w).copy())
                if limit and len(frames) >= limit:
                    break
        offset += size
    if not frames:
        raise ValueError(f"no decodable frames: {path}")
    return Flc(w, h, x_offset, y_offset, full_w or w, full_h or h, directions or 1,
               anim_length or len(frames), palette, frames)


def read_pcx_palette(path: Path) -> np.ndarray:
    data = Path(path).read_bytes()
    if len(data) < 769 or data[-769] != 12:
        raise ValueError(f"PCX has no 256-colour palette: {path}")
    return np.frombuffer(data[-768:], np.uint8).reshape(256, 3).copy()


def rgba(indices: np.ndarray, palette: np.ndarray, civ: np.ndarray | None = None) -> np.ndarray:
    """Composite-ready RGBA using Civ III's smoke/shadow/transparent ramps."""
    table = palette.copy()
    if civ is not None:
        table[:64] = civ[:64]
    out = np.zeros(indices.shape + (4,), np.uint8)
    out[..., :3] = table[indices]
    out[..., 3] = 255
    shadow = (indices >= 240) & (indices <= 254)
    out[shadow, :3] = 0
    out[shadow, 3] = np.minimum(255, (255 - indices[shadow].astype(int)) * 16)
    smoke = (indices >= 224) & (indices <= 239)
    out[smoke, :3] = 255
    out[smoke, 3] = np.minimum(255, (indices[smoke].astype(int) - 224) * 16)
    out[indices == 255] = 0
    return out
