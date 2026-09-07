"""Execute the actual DLL animation evaluator, including licensed pose parity."""
from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import normalized_pose_cache, normalized_skin

ROOT = Path(__file__).resolve().parents[2]


class AnimationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="c3x-animation-")
        cls.executable = Path(cls.scratch.name) / "animation_test"
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("portable C++ compiler unavailable")
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "Renderer/native/test_animation_runtime.cpp"),
                        "-o", str(cls.executable)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_timing_skinning_and_rejection(self):
        subprocess.run([str(self.executable)], check=True, capture_output=True, text=True)

    def test_installed_resource_pose_parity(self):
        runtime = ROOT / "Renderer/packs/ResourceAnimationRuntime"
        animated = ROOT / "Renderer/packs/ResourceAnimatedLab"
        landmarks = ROOT / "Renderer/packs/ResourceNormalized"
        if not all((p / "manifest.json").exists() for p in (runtime, animated, landmarks)):
            self.skipTest("optional local resource packs unavailable; run resource animation compiler")
        compiled = json.loads((runtime / "manifest.json").read_text())
        source = json.loads((animated / "manifest.json").read_text())
        cases = []
        for resource, record in source["resources"].items():
            for index, candidate in enumerate(record["subject_candidates"]):
                component = json.loads((animated / source["assets"][candidate["asset"]]["component"]).read_text())
                animation = source["animations"][candidate["animation"]]
                cases.append((animated, component, animation, compiled["resources"][resource][index]))
        source = json.loads((landmarks / "manifest.json").read_text())
        for resource in ("resource/fish", "resource/whales"):
            asset = source["assets"][source["resources"][resource]["landmark_asset"]]
            component = {**asset, "meshes": [asset["mesh"]], "draw_bindings": [{"mesh": 0}]}
            cases.append((landmarks, component, source["animations"][resource], compiled["resources"][resource][0]))
        samples = 0
        max_error = 0.0
        for pack, component, animation, subject in cases:
            skeleton = normalized_skin.load_skeleton(pack / component["skeleton"])
            calibrated = subject.get("calibration")
            cache = normalized_pose_cache.load_pose_cache(runtime / calibrated["pose_cache"] if calibrated else pack / animation["pose_cache"])
            self.assertEqual(len(component["draw_bindings"]), len(subject["parts"]))
            for binding, part in zip(component["draw_bindings"], subject["parts"]):
                mesh = normalized_skin.load_mesh(pack / component["meshes"][binding["mesh"]], len(skeleton["bones"]))
                first_pose = None
                maximum_motion = 0.0
                # Endpoints and two interior authored frames, with no clip-loop restart.
                for frame in (0, cache.frame_count // 3, cache.frame_count // 2, cache.frame_count - 1):
                    time = cache.duration * frame / (cache.frame_count - 1)
                    path = Path(self.scratch.name) / "sample.bin"
                    subprocess.run([str(self.executable), str(runtime / part["mesh"]), repr(time), "0", str(path)],
                                   check=True, capture_output=True, text=True)
                    actual = list(struct.iter_unpack("<8f", path.read_bytes()))
                    stride = len(skeleton["bones"]) * 16
                    start = frame * stride
                    worlds = [cache.matrices[start+i*16:start+(i+1)*16] for i in range(len(skeleton["bones"]))]
                    expected = normalized_skin.skin_positions(mesh, skeleton, worlds)
                    self.assertEqual(len(actual), len(expected))
                    for vertex, position, source_vertex in zip(actual, expected, mesh["vertices"]):
                        error = max(abs(vertex[a] - position[a]) for a in range(3))
                        max_error = max(max_error, error)
                        self.assertLess(error, 2e-5, (part["mesh"], frame, error))
                        self.assertTrue(all(math.isfinite(value) for value in vertex))
                        self.assertAlmostEqual(sum(value*value for value in vertex[3:6]), 1.0, places=4)
                        for a in range(2):
                            self.assertAlmostEqual(vertex[6+a], source_vertex["uv0"][a], places=5)
                    if first_pose is None:
                        first_pose = actual
                    else:
                        maximum_motion = max(maximum_motion, max(abs(v[a]-first_pose[i][a])
                            for i,v in enumerate(actual) for a in range(3)))
                    samples += 1
                self.assertGreater(maximum_motion, 1e-6, (part["mesh"], "clip does not move its body"))
        print(json.dumps({"resource_subjects": len(cases), "pose_samples": samples,
                          "maximum_position_error_tiles": max_error, "payload_bytes": compiled["payload_bytes"]}))


if __name__ == "__main__":
    unittest.main()
