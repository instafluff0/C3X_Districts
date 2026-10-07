"""Minimal BMP read / PNG write so the study needs only NumPy."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def read_bmp(path: Path) -> np.ndarray:
    """RGB uint8 array from an uncompressed 24/32-bit BMP."""
    data = Path(path).read_bytes()
    offset, = struct.unpack_from("<I", data, 10)
    width, height, _planes, bits = struct.unpack_from("<iiHH", data, 18)
    if bits not in (24, 32):
        raise ValueError(f"unsupported BMP depth {bits}: {path}")
    stride = (width * bits // 8 + 3) & ~3
    rows = np.frombuffer(data, np.uint8, stride * abs(height), offset).reshape(abs(height), stride)
    pixels = rows[:, :width * bits // 8].reshape(abs(height), width, bits // 8)[..., 2::-1]
    return np.ascontiguousarray(pixels[::-1] if height > 0 else pixels)


def write_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    channels = 4 if rgb.ndim == 3 and rgb.shape[2] == 4 else 3
    raw = b"".join(b"\0" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, 6 if channels == 4 else 2, 0, 0, 0)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
                           chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def over(base: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite an RGBA sprite onto an RGB canvas in display space."""
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(base.shape[1], x + w), min(base.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(float)
    alpha = src[..., 3:4] / 255
    region = base[y0:y1, x0:x1].astype(float)
    base[y0:y1, x0:x1] = np.clip(region * (1 - alpha) + src[..., :3] * alpha, 0, 255).astype(np.uint8)


def scale_nearest(image: np.ndarray, factor: int) -> np.ndarray:
    return image.repeat(factor, axis=0).repeat(factor, axis=1)
