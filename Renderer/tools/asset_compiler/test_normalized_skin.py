from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import normalized_animation as animation
from Renderer.tools.asset_compiler import normalized_skin as skin


def fixture_skeleton() -> dict:
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return {
        "schema": skin.SKELETON_SCHEMA,
        "asset_id": "test.skeleton",
        "matrix_convention": "row_major_row_vector",
        "position_unit": "tile",
        "bones": [
            {
                "name": "Root",
                "parent": -1,
                "local": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0], "scale_shear": identity},
                "inverse_bind_matrix": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            },
            {
                "name": "Child",
                "parent": 0,
                "local": {"position": [1.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0], "scale_shear": identity},
                "inverse_bind_matrix": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0],
            },
        ],
    }


def fixture_mesh() -> dict:
    vertices = []
    for position, uv in (((1.0, 0.0, 0.0), (0.0, 0.0)), ((2.0, 0.0, 0.0), (1.0, 0.0)), ((1.0, 1.0, 0.0), (0.0, 1.0))):
        vertices.append(
            {"position": list(position), "normal": [0.0, 0.0, 1.0], "uv0": list(uv), "joints": [1, 1, 1, 1], "weights": [1.0, 0.0, 0.0, 0.0]}
        )
    return {
        "schema": skin.MESH_SCHEMA,
        "asset_id": "test.mesh",
        "skeleton": "test.skeleton",
        "topology": {"primitive": "triangles", "indices": [0, 1, 2]},
        "vertices": vertices,
    }


class NormalizedSkinTests(unittest.TestCase):
    def load_fixtures(self, skeleton: dict | None = None, mesh: dict | None = None) -> tuple[dict, dict]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skeleton_path = root / "skeleton.json"
            mesh_path = root / "mesh.json"
            skeleton_path.write_text(json.dumps(skeleton or fixture_skeleton()), encoding="utf-8")
            mesh_path.write_text(json.dumps(mesh or fixture_mesh()), encoding="utf-8")
            loaded_skeleton = skin.load_skeleton(skeleton_path)
            loaded_mesh = skin.load_mesh(mesh_path, len(loaded_skeleton["bones"]))
        return loaded_skeleton, loaded_mesh

    def test_rest_pose_reconstructs_source_vertices(self) -> None:
        skeleton, mesh = self.load_fixtures()
        evidence = skin.validate_rest_pose(mesh, skeleton)
        self.assertLess(evidence["maximum_matrix_error"], 1.0e-9)
        self.assertLess(evidence["maximum_vertex_error"], 1.0e-9)

    def test_binds_by_bone_name_and_cpu_skins_sampled_pose(self) -> None:
        skeleton, mesh = self.load_fixtures()
        identity = animation.Channel(animation.IDENTITY, 3, ())
        identity_q = animation.Channel(animation.IDENTITY, 4, ())
        identity_s = animation.Channel(animation.IDENTITY, 9, ())
        clip = animation.AnimationClip(
            1.0,
            1.0,
            2,
            (
                animation.TrackGroup(
                    "Actor",
                    (
                        animation.TransformTrack("Root", 0, identity, identity_q, identity_s),
                        animation.TransformTrack(
                            "Child",
                            0,
                            animation.Channel(animation.CONSTANT, 3, (2.0, 0.0, 0.0)),
                            identity_q,
                            identity_s,
                        ),
                    ),
                ),
            ),
        )
        binding = skin.bind_clip(mesh, skeleton, clip, 0)
        self.assertEqual(binding["missing_weighted_bones"], [])
        pose = skin.sample_pose(skeleton, clip, 0, 0.5, False)
        positions = skin.skin_positions(mesh, skeleton, skin.world_matrices(skeleton, pose))
        self.assertEqual(positions[0], (2.0, 0.0, 0.0))
        self.assertEqual(positions[1], (3.0, 0.0, 0.0))

    def test_rejects_invalid_weight_sum(self) -> None:
        mesh = fixture_mesh()
        mesh["vertices"][0]["weights"] = [0.5, 0.0, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.json"
            path.write_text(json.dumps(mesh), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sum to one"):
                skin.load_mesh(path, 2)

    def test_all_zero_source_orientation_uses_skeleton_rest_orientation(self) -> None:
        skeleton = fixture_skeleton()
        skeleton["bones"][1]["local"]["orientation"] = [0.0, 0.0, 0.70710678, 0.70710678]
        zero_orientation = animation.Channel(
            animation.SAMPLED, 4, (0.0, 0.0, 0.0, 0.0) * 2
        )
        clip = animation.AnimationClip(
            1.0,
            1.0,
            2,
            (
                animation.TrackGroup(
                    "Actor",
                    (
                        animation.TransformTrack(
                            "Child",
                            0,
                            animation.Channel(animation.CONSTANT, 3, (1.0, 0.0, 0.0)),
                            zero_orientation,
                            animation.Channel(animation.IDENTITY, 9, ()),
                        ),
                    ),
                ),
            ),
        )
        pose = skin.sample_pose(skeleton, clip, 0, 0.5, False)
        self.assertEqual(pose[1].orientation, (0.0, 0.0, 0.70710678, 0.70710678))

    def test_zero_sample_inside_otherwise_valid_orientation_uses_rest_orientation(self) -> None:
        skeleton = fixture_skeleton()
        skeleton["bones"][1]["local"]["orientation"] = [0.0, 0.0, 0.70710678, 0.70710678]
        mixed_orientation = animation.Channel(
            animation.SAMPLED,
            4,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        clip = animation.AnimationClip(
            1.0,
            1.0,
            2,
            (
                animation.TrackGroup(
                    "Actor",
                    (
                        animation.TransformTrack(
                            "Child",
                            0,
                            animation.Channel(animation.IDENTITY, 3, ()),
                            mixed_orientation,
                            animation.Channel(animation.IDENTITY, 9, ()),
                        ),
                    ),
                ),
            ),
        )
        pose = skin.sample_pose(skeleton, clip, 0, 0.0, False)
        self.assertEqual(pose[1].orientation, (0.0, 0.0, 0.70710678, 0.70710678))


if __name__ == "__main__":
    unittest.main()
