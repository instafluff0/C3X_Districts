from __future__ import annotations

import struct
import json
import tempfile
import unittest
from pathlib import Path

from Renderer.preview import render_skinned_resource
from Renderer.tools.asset_compiler import normalized_animation as animation
from Renderer.tools.asset_compiler.test_normalized_skin import fixture_mesh, fixture_skeleton


class SkinnedResourcePreviewTests(unittest.TestCase):
    def test_deforms_mesh_at_requested_absolute_time(self) -> None:
        identity_position = animation.Channel(animation.IDENTITY, 3, ())
        identity_orientation = animation.Channel(animation.IDENTITY, 4, ())
        identity_scale = animation.Channel(animation.IDENTITY, 9, ())
        clip = animation.AnimationClip(
            1.0,
            1.0,
            2,
            (
                animation.TrackGroup(
                    "Actor",
                    (
                        animation.TransformTrack("Root", 0, identity_position, identity_orientation, identity_scale),
                        animation.TransformTrack(
                            "Child",
                            0,
                            animation.Channel(animation.CONSTANT, 3, (2.0, 0.0, 0.0)),
                            identity_orientation,
                            identity_scale,
                        ),
                    ),
                ),
            ),
        )
        result = render_skinned_resource.deformed_mesh(
            fixture_mesh(), fixture_skeleton(), clip, 0, 0.5
        )
        self.assertEqual(result["vertices"][0]["position"], [2.0, 0.0, 0.0])
        self.assertEqual(result["vertices"][1]["position"], [3.0, 0.0, 0.0])

    def test_resource_texture_wraps_uvs(self) -> None:
        dds = bytearray(148)
        dds[:4] = b"DDS "
        dds[84:88] = b"DX10"
        struct.pack_into("<I", dds, 12, 4)
        struct.pack_into("<I", dds, 16, 4)
        struct.pack_into("<I", dds, 28, 1)
        struct.pack_into("<I", dds, 128, 71)
        dds.extend(struct.pack("<HHI", 0xF800, 0x001F, 0))
        texture = render_skinned_resource.WrappingBc1Texture(bytes(dds))
        self.assertEqual(texture.sample(1.25, -0.75), texture.sample(0.25, 0.25))

    def test_deforms_mesh_from_baked_world_matrices(self) -> None:
        identity = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        child = identity[:12] + (2.0, 0.0, 0.0, 1.0)
        result = render_skinned_resource.deformed_mesh_from_world_matrices(
            fixture_mesh(), fixture_skeleton(), (identity, child)
        )
        self.assertEqual(result["vertices"][0]["position"], [2.0, 0.0, 0.0])
        self.assertEqual(result["vertices"][1]["position"], [3.0, 0.0, 0.0])

    def test_unvalidated_pose_is_rejected_before_asset_loading(self) -> None:
        manifest = {
            "schema": "c3x.resource_pack.v0",
            "assets": {},
            "resources": {"resource/fish": {"landmark_asset": "resource/fish/landmark"}},
            "animations": {
                "resource/fish": {"pose_status": "model_aware_sampling_required"}
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not validated"):
                render_skinned_resource.render_contact_sheet(
                    path, "resource/fish", 512, 256, 2
                )


if __name__ == "__main__":
    unittest.main()
