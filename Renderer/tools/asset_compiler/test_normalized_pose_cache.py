from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import normalized_pose_cache
from Renderer.tools.asset_compiler.test_normalized_skin import fixture_skeleton


def write_cache(path: Path, names: tuple[str, ...], frames: tuple[tuple[float, ...], ...]) -> None:
    encoded = [name.encode() for name in names]
    strings = 36 + len(names) * 8
    samples = (strings + sum(map(len, encoded)) + 3) & ~3
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sIffIIII", b"C3XPOSE\0", 1, 1.0, 1.0, 2, len(names), 36, samples))
        offset = strings
        for value in encoded:
            stream.write(struct.pack("<II", offset, len(value)))
            offset += len(value)
        for value in encoded:
            stream.write(value)
        stream.write(bytes(samples - stream.tell()))
        for frame in frames:
            stream.write(struct.pack(f"<{len(frame)}f", *frame))


class PoseCacheTests(unittest.TestCase):
    def test_canonical_writer_is_deterministic(self) -> None:
        identity = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        moved = identity[:12] + (0.25, -0.5, 1.0, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.c3pose"
            second = Path(directory) / "second.c3pose"
            frames = ((identity,), (moved,))
            cache = normalized_pose_cache.write_pose_cache(first, 1.0, 1.0, ("Root",), frames)
            normalized_pose_cache.write_pose_cache(second, 1.0, 1.0, ("Root",), frames)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(0.25, cache.sample(1.0, False)[0][12])

    def test_writer_rejects_non_affine_matrix(self) -> None:
        bad = (1.0, 0.0, 0.0, 1.0) * 4
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "non-affine"):
                normalized_pose_cache.write_pose_cache(
                    Path(directory) / "bad.c3pose", 1.0, 1.0, ("Root",), ((bad,), (bad,))
                )

    def test_load_sample_and_validate_rest_binding(self) -> None:
        skeleton = fixture_skeleton()
        root = tuple(skeleton["bones"][0]["inverse_bind_matrix"])
        child = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        moved = child[:12] + (2.0, 0.0, 0.0, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.c3pose"
            write_cache(path, ("Root", "Child"), (root + child, root + moved))
            cache = normalized_pose_cache.load_pose_cache(path)
        self.assertEqual(cache.sample(1.0, False)[1][12], 2.0)
        self.assertEqual(normalized_pose_cache.validate_skeleton_binding(cache, skeleton)["bones"], 2)

    def test_rejects_trailing_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.c3pose"
            identity = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            write_cache(path, ("Root",), (identity, identity))
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "sections"):
                normalized_pose_cache.load_pose_cache(path)


if __name__ == "__main__":
    unittest.main()
