import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import normalized_animation as animation
from Renderer.tools.asset_compiler.build_resource_animation_runtime import calibrated_resource_poses


class ResourceAnimationCalibrationTests(unittest.TestCase):
    def test_translation_units_root_placement_and_missing_tracks(self):
        def bone(name, parent, position):
            return {"name": name, "parent": parent, "local": {
                "position": position, "orientation": [0, 0, 0, 1],
                "scale_shear": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}
        skeleton = {"bones": [bone("root", -1, [.5, 0, 0]),
                              bone("animated_child", 0, [0, .2, 0]),
                              bone("untracked_child", 0, [0, 0, .3])]}
        def track(name, positions):
            return animation.TransformTrack(name, 0,
                animation.Channel(animation.SAMPLED, 3, positions),
                animation.Channel(animation.IDENTITY, 4, ()),
                animation.Channel(animation.IDENTITY, 9, ()))
        clip = animation.AnimationClip(1, 1, 2, (animation.TrackGroup("body", (
            track("root", (30, 0, 0, 40, 0, 0)),
            track("animated_child", (0, 2, 0, 0, 2, 0)))),))
        with tempfile.TemporaryDirectory() as directory:
            cache = calibrated_resource_poses(skeleton, clip, 0, .1, Path(directory)/"pose.bin")
        start, end = cache.sample(0, False), cache.sample(1, False)
        self.assertAlmostEqual(start[0][12], .5)  # Preserve normalized anchor, not source-scene x=30.
        self.assertAlmostEqual(end[0][12], 1.5)  # Preserve authored root motion after unit conversion.
        for pose in (start, end):
            self.assertAlmostEqual(pose[1][13], .2)  # Authored child uses converted units.
            self.assertAlmostEqual(pose[2][14], .3)  # Missing track already uses normalized rest units.


if __name__ == "__main__":
    unittest.main()
