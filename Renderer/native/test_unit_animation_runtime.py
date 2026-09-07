"""Compare the DLL evaluator with independently sampled complete source unit kits."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from Renderer.preview.render_unit_turntable import _rigid_mesh, _skinned_mesh
from Renderer.tools.asset_compiler import normalized_animation, normalized_skin
from Renderer.tools.asset_compiler.unit_family_action_validator import SOCKET_PROFILE, _best_group

ROOT = Path(__file__).resolve().parents[2]


class UnitAnimationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="c3x-unit-animation-")
        external = os.environ.get("C3X_ANIMATION_TEST_EXE")
        cls.backend = "portable_cpp"
        if external:
            cls.executable = Path(external).resolve()
            data = cls.executable.read_bytes()
            pe = struct.unpack_from("<I", data, 0x3c)[0]
            if data[:2] != b"MZ" or data[pe:pe+4] != b"PE\0\0" or struct.unpack_from("<H", data, pe+4)[0] != 0x14c:
                raise ValueError("explicit Windows animation evaluator must be x86")
            cls.backend = "windows_x86"
            return
        cls.executable = Path(cls.scratch.name)/"animation_test"
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("portable C++ compiler unavailable")
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(ROOT/"Renderer/native/test_animation_runtime.cpp"),
                        "-o", str(cls.executable)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_complete_source_kits_at_native_action_phases(self):
        runtime = ROOT/"Renderer/packs/UnitAnimationRuntime"
        if not (runtime/"manifest.json").exists():
            self.skipTest("local unit pack unavailable; run build_unit_animation_runtime.py")
        exported = json.loads((runtime/"manifest.json").read_text())
        cases = 0
        maximum_error = 0.0
        rigid_cases = 0
        for unit_id, unit in exported["units"].items():
            pack = ROOT/"Renderer/packs"/unit["source_pack"]
            source = json.loads((pack/"manifest.json").read_text())
            recipe = json.loads((pack/unit["source_recipe"]).read_text())
            self.assertEqual(set(unit["actions"]), set(recipe["actions"]))
            components = {c["asset"]: json.loads((pack/source["assets"][c["asset"]]["component"]).read_text())
                          for c in recipe["components"]}
            skeletons = {asset: normalized_skin.load_skeleton(pack/c["skeleton"])
                         for asset, c in components.items() if c["binding_mode"] == "vertex_skin"}
            driver = recipe.get("animation_driver", "unit/warrior/body")
            sockets = source.get("unit_binding", {}).get("sockets", SOCKET_PROFILE)
            for action, compiled in unit["actions"].items():
                animation = source["animations"][recipe["actions"][action]]
                clip = normalized_animation.load_clip(pack/animation["clip"])
                self.assertEqual(compiled["frames"], clip.frame_count)
                self.assertEqual(compiled["loop"], animation["loop"])
                self.assertEqual({p["asset"] for p in compiled["parts"]}, set(components))
                expected_count = sum(len(c.get("draw_bindings", [None])) for c in components.values())
                self.assertEqual(len(compiled["parts"]), expected_count)
                # Native initial/ongoing/final phases, including exact one-shot endpoints.
                for frame in (0, clip.frame_count//2, clip.frame_count-1):
                    time = clip.duration*frame/(clip.frame_count-1)
                    worlds = {}
                    for asset, skeleton in skeletons.items():
                        group, _ = _best_group(clip, {b["name"] for b in skeleton["bones"]})
                        pose = normalized_skin.sample_pose(skeleton, clip, group, time, False)
                        worlds[asset] = normalized_skin.world_matrices(skeleton, pose)
                    root = next(i for i, b in enumerate(skeletons[driver]["bones"]) if b["parent"] == -1)
                    rest_root = normalized_skin.world_matrices(skeletons[driver])[root]
                    root_delta = [worlds[driver][root][12+a]-rest_root[12+a] for a in (0, 1)]
                    for part in compiled["parts"]:
                        component = components[part["asset"]]
                        if component["binding_mode"] == "vertex_skin":
                            skeleton = skeletons[part["asset"]]
                            mesh = normalized_skin.load_mesh(pack/part["source_mesh"], len(skeleton["bones"]))
                            expected = _skinned_mesh(mesh, skeleton, worlds[part["asset"]])
                        else:
                            bone = sockets[component["attachment_point"]]["bone"]
                            index = [b["name"] for b in skeletons[driver]["bones"]].index(bone)
                            mesh = json.loads((pack/part["source_mesh"]).read_text())
                            expected = _rigid_mesh(mesh, worlds[driver][index], component["model_scale"])
                            rigid_cases += 1
                        payload = runtime/part["mesh"]
                        self.assertEqual(payload.stem, hashlib.sha256(payload.read_bytes()).hexdigest())
                        target = Path(self.scratch.name)/"sample.bin"
                        subprocess.run([str(self.executable), str(payload), repr(time), "0", str(target)],
                                       check=True, capture_output=True, text=True)
                        vertices = list(struct.iter_unpack("<8f", target.read_bytes()))
                        self.assertEqual(len(vertices), len(expected["vertices"]))
                        for actual, wanted in zip(vertices, expected["vertices"]):
                            expected_position = [wanted["position"][a]-(root_delta[a] if a<2 else 0) for a in range(3)]
                            error = max(abs(actual[a]-expected_position[a]) for a in range(3))
                            maximum_error = max(maximum_error, error)
                            self.assertLess(error, 2e-5, (unit_id, action, part["asset"], frame, error))
                            self.assertTrue(all(math.isfinite(v) for v in actual))
                            self.assertAlmostEqual(sum(v*v for v in actual[3:6]), 1.0, places=4)
                            for a in range(2):
                                self.assertAlmostEqual(actual[6+a], wanted["uv0"][a], places=5)
                        cases += 1
                        # Every source channel survives; source tint cannot be discarded.
                        material = json.loads((pack/part["source_material"]).read_text())
                        self.assertEqual(set(part["material"]["channels"]), set(material["channels"]))
                        self.assertEqual(part["material"]["source_tint"], component.get("tint"))
                        for channel, info in material["channels"].items():
                            self.assertEqual((pack/info["texture"]).read_bytes(),
                                             (runtime/part["material"]["channels"][channel]["texture"]).read_bytes())
        report = {"schema": "c3x.unit_animation_payload_proof.v1", "status": "pass",
            "backend": self.backend,
            "units": len(exported["units"]), "actions": sum(len(u["actions"]) for u in exported["units"].values()),
            "part_pose_samples": cases, "socket_pose_samples": rigid_cases,
            "maximum_position_error_tiles": maximum_error, "payload_bytes": exported["payload_bytes"],
            "scope": "actual portable DLL evaluator versus raw authored clips; native body bridge/rendering not yet enabled"}
        (ROOT/f"Renderer/verification/animation/unit-payloads-{self.backend}.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report))


if __name__ == "__main__":
    unittest.main()
