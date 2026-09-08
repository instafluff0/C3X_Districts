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

    def test_static_fortify_blends_local_joints_without_shrinking(self):
        from Renderer.tools.asset_compiler.build_unit_animation_runtime import fortify_transition
        from Renderer.tools.asset_compiler import normalized_animation as a
        identity = (1.,0.,0.,0.,1.,0.,0.,0.,1.)
        skeleton = {"bones": [{"name":"Root", "parent":-1,
            "local":{"position":(0.,0.,0.),"orientation":(0.,0.,0.,1.),"scale_shear":identity}}]}
        def clip(q,x):
            track=a.TransformTrack("Root",0,a.Channel(a.CONSTANT,3,(x,0.,0.)),
                a.Channel(a.CONSTANT,4,q),a.Channel(a.CONSTANT,9,identity))
            return a.AnimationClip(1/30,30.,2,(a.TrackGroup("Root",(track,)),))
        cache=fortify_transition(skeleton,clip((0.,0.,0.,1.),0.),clip((0.,0.,1.,0.),2.))
        self.assertEqual(cache.frame_count,16)
        self.assertAlmostEqual(cache.matrices[12],0.)
        self.assertAlmostEqual(cache.matrices[-4],2.)
        for frame in range(16):
            m=cache.matrices[frame*16:(frame+1)*16]
            for row in range(3):self.assertAlmostEqual(sum(m[row*4+c]**2 for c in range(3)),1.)
        # Equivalent opposite quaternion signs take the same shortest path.
        same=fortify_transition(skeleton,clip((0.,0.,0.,1.),0.),clip((0.,0.,0.,-1.),0.))
        for frame in range(16):self.assertAlmostEqual(same.matrices[frame*16],1.)

    def test_complete_source_kits_at_native_action_phases(self):
        runtime = ROOT/"Renderer/packs"/os.environ.get("C3X_UNIT_TEST_PACK", "UnitAnimationRuntime")
        if not (runtime/"manifest.json").exists():
            self.skipTest("local unit pack unavailable; run build_unit_animation_runtime.py")
        exported = json.loads((runtime/"manifest.json").read_text())
        cases = 0
        maximum_error = 0.0
        rigid_cases = 0
        reference_units = 0
        reference_actions = 0
        composed_units = []
        for unit_id, unit in exported["units"].items():
            if unit["source_pack"] == "C3XGenericMissiles":
                composed_units.append(unit_id)
                continue
            pack = ROOT/"Renderer/packs"/unit["source_pack"]
            source = json.loads((pack/"manifest.json").read_text())
            recipe = json.loads((pack/unit["source_recipe"]).read_text())
            if recipe.get("schema") == "c3x.unit_composition.v0":
                composed_units.append(unit_id)
                continue
            reference_units += 1
            self.assertEqual(set(unit["actions"]), set(recipe["actions"]))
            components = {c["asset"]: json.loads((pack/source["assets"][c["asset"]]["component"]).read_text())
                          for c in recipe["components"]}
            skeletons = {asset: normalized_skin.load_skeleton(pack/c["skeleton"])
                         for asset, c in components.items() if c["binding_mode"] in {"vertex_skin", "mixed"}}
            driver = recipe.get("animation_driver", "unit/warrior/body")
            sockets = source.get("unit_binding", {}).get("sockets", SOCKET_PROFILE)
            for action, compiled in unit["actions"].items():
                reference_actions += 1
                animation = source["animations"][recipe["actions"][action]]
                clip = normalized_animation.load_clip(pack/animation["clip"])
                transition = compiled.get("presentation") == "idle_to_static_fortify"
                self.assertEqual(compiled["frames"], 16 if transition else clip.frame_count)
                self.assertEqual(compiled["loop"], animation["loop"])
                selected = set(recipe.get("action_components", {}).get(action, components))
                self.assertEqual({p["asset"] for p in compiled["parts"]}, selected)
                expected_count = sum(len(c.get("draw_bindings", [None])) for a, c in components.items() if a in selected)
                self.assertEqual(len(compiled["parts"]), expected_count)
                # Native initial/ongoing/final phases, including exact one-shot endpoints.
                # Baked transitions retain raw source endpoints. Their interior
                # rotation/translation contract is checked independently below.
                samples = (0,15) if transition else (0,clip.frame_count//2,clip.frame_count-1)
                for frame in samples:
                    time = compiled["duration"]*frame/(compiled["frames"]-1)
                    source_clip = clip
                    source_time = time
                    if transition:
                        if frame == 0:
                            source_clip = normalized_animation.load_clip(pack/source["animations"][recipe["actions"]["idle"]]["clip"])
                            source_time = 0.
                        else:
                            source_time = clip.duration
                    worlds = {}
                    for asset, skeleton in skeletons.items():
                        group, _ = _best_group(source_clip, {b["name"] for b in skeleton["bones"]})
                        pose = normalized_skin.sample_pose(skeleton, source_clip, group, source_time, False)
                        worlds[asset] = normalized_skin.world_matrices(skeleton, pose)
                    root = next(i for i, b in enumerate(skeletons[driver]["bones"]) if b["parent"] == -1)
                    rest_root = normalized_skin.world_matrices(skeletons[driver])[root]
                    root_delta = [worlds[driver][root][12+a]-rest_root[12+a] for a in (0, 1)]
                    if action == "move" and recipe.get("move_cycle_translation_bone"):
                        skeleton = skeletons[driver]
                        index = [b["name"] for b in skeleton["bones"]].index(recipe["move_cycle_translation_bone"])
                        group = _best_group(clip, {b["name"] for b in skeleton["bones"]})[0]
                        endpoints = [normalized_skin.world_matrices(skeleton, normalized_skin.sample_pose(skeleton, clip, group, t, False))
                                     for t in (0., clip.duration)]
                        for a in (0, 1):
                            drift = (endpoints[1][index][12+a] - endpoints[1][root][12+a]) - (endpoints[0][index][12+a] - endpoints[0][root][12+a])
                            root_delta[a] += drift * frame / (compiled["frames"] - 1)
                    for part in compiled["parts"]:
                        component = components[part["asset"]]
                        mesh_document = json.loads((pack/part["source_mesh"]).read_text())
                        if mesh_document["schema"] == normalized_skin.MESH_SCHEMA:
                            skeleton = skeletons[part["asset"]]
                            mesh = normalized_skin.load_mesh(pack/part["source_mesh"], len(skeleton["bones"]))
                            expected = _skinned_mesh(mesh, skeleton, worlds[part["asset"]])
                        else:
                            local_driver = part["asset"] if part["asset"] in worlds and component.get("rigid_driver_bone") else driver
                            bone = component.get("rigid_driver_bone") or sockets[component["attachment_point"]]["bone"]
                            index = [b["name"] for b in skeletons[local_driver]["bones"]].index(bone)
                            mesh = json.loads((pack/part["source_mesh"]).read_text())
                            expected = _rigid_mesh(mesh, worlds[local_driver][index], component["model_scale"])
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
            "source_reference_units": reference_units,
            "composed_or_original_units_requiring_separate_verification": composed_units,
            "units": reference_units, "actions": reference_actions,
            "runtime_catalog_units": len(exported["units"]),
            "part_pose_samples": cases, "socket_pose_samples": rigid_cases,
            "maximum_position_error_tiles": maximum_error, "payload_bytes": exported["payload_bytes"],
            "scope": "actual portable DLL evaluator versus raw authored clips; source-payload proof; live gameplay behavior is a separate checkpoint"}
        prefix = "roster/" if os.environ.get("C3X_UNIT_TEST_PACK") else ""
        output = ROOT/f"Renderer/lab/out/verification/animation/{prefix}unit-payloads-{self.backend}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report))

    def test_parent_scene_horse_clips_apply_cart_translation_once(self):
        runtime = ROOT/"Renderer/packs"/os.environ.get("C3X_UNIT_TEST_PACK", "UnitAnimationRuntime")
        if not (runtime/"manifest.json").exists():self.skipTest("local unit pack unavailable")
        exported=json.loads((runtime/"manifest.json").read_text())
        unit=exported['units'].get('unit/heavy_chariot')
        if not unit:self.skipTest("expanded roster not selected")
        pack=ROOT/'Renderer/packs'/unit['source_pack']
        manifest=json.loads((pack/'manifest.json').read_text())
        recipe=json.loads((pack/unit['source_recipe']).read_text())
        parent=recipe['nodes'][recipe['root_node']]
        parent_skeleton=normalized_skin.load_skeleton(pack/parent['skeleton'])
        root=next(i for i,b in enumerate(parent_skeleton['bones']) if b['parent']==-1)
        rest=normalized_skin.world_matrices(parent_skeleton)[root]
        samples=0
        for node_id,node in recipe['nodes'].items():
            for action in node.get('parent_space_actions',[]):
                compiled=unit['actions'][action]
                part=next(p for p in compiled['parts'] if p['asset']==node['animation_driver'])
                component=json.loads((pack/manifest['assets'][part['asset']]['component']).read_text())
                skeleton=normalized_skin.load_skeleton(pack/component['skeleton'])
                mesh=normalized_skin.load_mesh(pack/part['source_mesh'],len(skeleton['bones']))
                clips={n:normalized_animation.load_clip(pack/manifest['animations'][clip]['clip'])
                       for n,clip in recipe['actions'][action]['node_clips'].items()}
                for frame in (0,compiled['frames']//2,compiled['frames']-1):
                    phase=frame/(compiled['frames']-1)
                    def worlds(sk,clip):
                        group=_best_group(clip,{b['name'] for b in sk['bones']})[0]
                        return normalized_skin.world_matrices(sk,normalized_skin.sample_pose(sk,clip,group,clip.duration*phase,False))
                    expected=_skinned_mesh(mesh,skeleton,worlds(skeleton,clips[node_id]))
                    parent_root=worlds(parent_skeleton,clips[recipe['root_node']])[root]
                    target=Path(self.scratch.name)/'parent-scene.bin'
                    subprocess.run([str(self.executable),str(runtime/part['mesh']),str(compiled['duration']*phase),'0',str(target)],check=True,capture_output=True)
                    actual=list(struct.iter_unpack('<8f',target.read_bytes()))
                    self.assertEqual(len(actual),len(expected['vertices']))
                    for vertex,wanted in zip(actual,expected['vertices']):
                        for axis in range(3):
                            position=wanted['position'][axis]*node['variation_scale']*parent['variation_scale']
                            if axis<2:position-=(parent_root[12+axis]-rest[12+axis])*parent['variation_scale']
                            self.assertAlmostEqual(vertex[axis],position,places=5)
                    samples+=1
        self.assertGreater(samples,0)
        output = ROOT/'Renderer/lab/out/verification/animation/roster/parent-scene-reference.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(
            {'status':'pass','part_pose_samples':samples,'contract':'paired horse clips contain one copy of parent scene translation'},indent=2)+'\n')
        print(f"parent-scene source reference: {samples} part poses passed")


if __name__ == "__main__":
    unittest.main()
