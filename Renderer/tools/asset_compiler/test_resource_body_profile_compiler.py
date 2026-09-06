import unittest

from Renderer.tools.asset_compiler import normalized_animation
from Renderer.tools.asset_compiler.resource_body_profile_compiler import _pose_frames, _resource_slug


class ResourceBodyProfileCompilerTests(unittest.TestCase):
    def test_generic_resource_ids_are_strict(self) -> None:
        self.assertEqual("horses", _resource_slug("resource/horses"))
        with self.assertRaises(ValueError):
            _resource_slug("UNIT_HORSE")

    def test_pose_bake_replaces_impossible_shipped_channel_sentinel(self) -> None:
        skeleton = {
            "bones": [{
                "name": "Root",
                "parent": -1,
                "local": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1], "scale_shear": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
                "inverse_bind_matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            }]
        }
        channel = normalized_animation.Channel
        track = normalized_animation.TransformTrack(
            "Root", 0,
            channel(normalized_animation.SAMPLED, 3, (0, 0, 0, 1.0e18, 0, 0)),
            channel(normalized_animation.IDENTITY, 4, ()),
            channel(normalized_animation.IDENTITY, 9, ()),
        )
        clip = normalized_animation.AnimationClip(
            1.0, 1.0, 2, (normalized_animation.TrackGroup("body", (track,)),)
        )
        frames, repairs = _pose_frames(skeleton, clip, 0)
        self.assertEqual(1, repairs)
        self.assertEqual(0.0, frames[1][0][12])


if __name__ == "__main__":
    unittest.main()
