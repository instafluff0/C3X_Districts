import math
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import normalized_animation as animation


def build_clip_bytes() -> bytes:
    duration = 2.0
    sample_rate = 1.0
    frame_count = 3
    groups = [
        ("Actor", [("Bone", 0, animation.SAMPLED, animation.SAMPLED)]),
        ("Actor", [("Bone", 0, animation.CONSTANT, animation.IDENTITY)]),
    ]
    strings = bytearray()
    group_names = []
    track_names = []
    for group_name, _tracks in groups:
        encoded = group_name.encode("utf-8")
        group_names.append((animation.HEADER.size + len(strings), len(encoded)))
        strings.extend(encoded)
    for _group_name, tracks in groups:
        for track_name, _flags, _position_mode, _orientation_mode in tracks:
            encoded = track_name.encode("utf-8")
            track_names.append((animation.HEADER.size + len(strings), len(encoded)))
            strings.extend(encoded)

    group_table_offset = (animation.HEADER.size + len(strings) + 3) & ~3
    track_count = sum(len(tracks) for _name, tracks in groups)
    track_table_offset = group_table_offset + len(groups) * animation.GROUP_RECORD.size
    data_offset = track_table_offset + track_count * animation.TRACK_RECORD.size
    data = bytearray()
    track_records = []
    track_name_index = 0

    def channel(mode: int, values: tuple[float, ...]) -> int:
        if mode == animation.IDENTITY:
            return 0
        offset = data_offset + len(data)
        data.extend(struct.pack(f"<{len(values)}f", *values))
        return offset

    for group_index, (_group_name, tracks) in enumerate(groups):
        for _track_name, flags, position_mode, orientation_mode in tracks:
            if group_index == 0:
                position_values = (0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0)
                orientation_values = (
                    0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, -1.0,
                    0.0, 0.0, 1.0, 0.0,
                )
            else:
                position_values = (100.0, 0.0, 0.0)
                orientation_values = ()
            position_offset = channel(position_mode, position_values)
            orientation_offset = channel(orientation_mode, orientation_values)
            modes = position_mode | (orientation_mode << 2) | (animation.IDENTITY << 4)
            name_offset, name_bytes = track_names[track_name_index]
            track_name_index += 1
            track_records.append(
                (name_offset, name_bytes, flags, modes, position_offset, orientation_offset, 0, 0)
            )

    output = bytearray(
        animation.HEADER.pack(
            animation.MAGIC,
            animation.VERSION,
            0,
            duration,
            sample_rate,
            frame_count,
            len(groups),
            track_count,
            len(strings),
            group_table_offset,
            track_table_offset,
            data_offset,
            len(data),
        )
    )
    output.extend(strings)
    output.extend(b"\0" * (group_table_offset - len(output)))
    first_track = 0
    for index, (_group_name, tracks) in enumerate(groups):
        output.extend(animation.GROUP_RECORD.pack(*group_names[index], first_track, len(tracks)))
        first_track += len(tracks)
    for record in track_records:
        output.extend(animation.TRACK_RECORD.pack(*record))
    output.extend(data)
    return bytes(output)


class NormalizedAnimationTests(unittest.TestCase):
    def load(self, data: bytes) -> animation.AnimationClip:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.c3anim"
            path.write_bytes(data)
            return animation.load_clip(path)

    def test_loads_multiple_groups_with_duplicate_names_across_groups(self) -> None:
        clip = self.load(build_clip_bytes())
        self.assertEqual(len(clip.groups), 2)
        self.assertEqual([group.name for group in clip.groups], ["Actor", "Actor"])
        self.assertEqual(clip.groups[0].tracks[0].name, "Bone")
        self.assertEqual(clip.groups[1].tracks[0].name, "Bone")
        self.assertEqual(clip.groups[0].tracks[0].flags, 0)
        self.assertEqual(clip.groups[1].tracks[0].flags, 0)

    def test_samples_channels_at_fixed_timestamp(self) -> None:
        clip = self.load(build_clip_bytes())
        value = clip.sample(0, "Bone", 1.5, loop=False)
        self.assertEqual(value.position, (15.0, 0.0, 0.0))
        self.assertAlmostEqual(sum(v * v for v in value.orientation), 1.0, places=6)
        self.assertEqual(value.scale_shear, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    def test_quaternion_interpolation_uses_shortest_arc(self) -> None:
        clip = self.load(build_clip_bytes())
        value = clip.sample(0, "Bone", 0.5, loop=False)
        self.assertEqual(value.orientation, (0.0, 0.0, 0.0, 1.0))

    def test_clamp_and_loop_are_explicit_and_deterministic(self) -> None:
        clip = self.load(build_clip_bytes())
        self.assertEqual(clip.sample(0, "Bone", 2.5, loop=False).position, (20.0, 0.0, 0.0))
        self.assertEqual(clip.sample(0, "Bone", 2.5, loop=True).position, (5.0, 0.0, 0.0))
        self.assertEqual(clip.sample(1, "Bone", -1.0, loop=False).position, (100.0, 0.0, 0.0))

    def test_rejects_bad_magic_and_trailing_data(self) -> None:
        original = build_clip_bytes()
        with self.assertRaisesRegex(ValueError, "magic"):
            self.load(b"BROKEN!!" + original[8:])
        with self.assertRaisesRegex(ValueError, "file size"):
            self.load(original + b"x")

    def test_rejects_non_finite_time(self) -> None:
        clip = self.load(build_clip_bytes())
        with self.assertRaisesRegex(ValueError, "finite"):
            clip.sample(0, "Bone", math.nan, loop=False)


if __name__ == "__main__":
    unittest.main()
