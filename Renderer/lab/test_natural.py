"""Shared production natural inputs, executable without D3D or licensed assets."""
from pathlib import Path
import subprocess
import json
import struct
import hashlib
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NaturalInputs(unittest.TestCase):
    def test_production_placement_and_dune_patterns(self):
        self.shared_probe("patterns", "200000 hashes/random values, 132612 exact dune samples")

    def test_mesh_emission_and_exclusion_queries(self):
        self.shared_probe("mesh", "248 scopes")

    def test_canopy_clears_routes_resources_and_selects_varieties(self):
        self.shared_probe("canopy", "uncleared crowns hiding a road")

    def test_source_surface_composition(self):
        self.shared_probe("surface", "grass omitted, 3 active biomes")

    def test_relief_inputs_and_query_policy(self):
        self.shared_probe("relief", "1440 exact field samples, 16 scopes")

    def test_river_world_queries_and_page_lifetime(self):
        self.shared_probe("world", "312 exact samples; wrapped topology, revision invalidation and 16-page LRU")

    def shared_probe(self, name, expected):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / name
            result = subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                str(ROOT / ("Renderer/lab/shared/natural/test_" + name + ".cpp")), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn(expected, result.stdout)

    def test_surface_queries_preserve_values_and_invalidation_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "queries"
            result = subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                str(ROOT / "Renderer/lab/shared/natural/test_queries.cpp"), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("18 scopes", result.stdout)
            self.assertIn("world/coast observations and cache statistics", result.stdout)

    def test_shared_production_ground_compiles_without_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "ground"
            subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "Renderer/lab/shared/natural/test_ground.cpp"),
                            "-o", str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("324 exact vertices, 289 corners, 512 triangles", result.stdout)

    def test_current_natural_rebuild_preserves_independent_payloads(self):
        from Renderer.native.source_fidelity import prepare as natural
        from Renderer.lab.preparation import digest
        if not (ROOT / "Renderer/packs/BeautyStudies/beauty_objects.bin").exists():
            self.skipTest("Local natural source inputs are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = natural.build_pack(output)
            current = json.loads((natural.HERE / "provenance.json").read_text())
            self.assertEqual(record, current)
            # natural.bin and forest-varieties.bin beside every distinct texture.
            self.assertEqual(len(list(output.iterdir())),
                             record["texture_count"] + 2 + record["forest_varieties"]["new_textures"])
            for path in output.iterdir():
                self.assertEqual(digest(path), digest(natural.PACK / path.name), path.name)
                self.assertEqual(path.stat().st_nlink, 1)

    def test_cliff_path_table_and_source_protection(self):
        from Renderer.native.render_core import prepare_assets as assets
        header = b"C3XVEG1\0" + struct.pack("<4I", 1, 24, 6, 1)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            assets.bundle_paths(header)
        with self.assertRaisesRegex(ValueError, "length"):
            assets.bundle_paths(header + struct.pack("<I", 4097))
        with self.assertRaisesRegex(ValueError, "overlap"):
            assets.prepare(source=assets.SOURCE, output=assets.SOURCE)
        with self.assertRaises(ValueError):
            assets.local(assets.SOURCE / "../outside.dds", assets.SOURCE)

    def test_current_cliff_rebuild_preserves_payloads(self):
        from Renderer.native.render_core import prepare_assets as assets
        if not (assets.SOURCE / "cliffs.bin").exists():
            self.skipTest("Preserved local hill/cliff inputs are unavailable")
        with tempfile.TemporaryDirectory(dir=ROOT / "Renderer/lab/.cache") as directory:
            output = Path(directory)
            assets.prepare(output=output)
            record = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(record["files"]), 26)
            for row in record["files"]:
                self.assertEqual((output / row["path"]).read_bytes(),
                                 (assets.OUTPUT / row["path"]).read_bytes(), row["path"])
                self.assertTrue(row["source"].startswith("Renderer/packs/TerrainProfileSources/current/"))
            preserved = (assets.SOURCE / "textures/cliff_0_3.dds").read_bytes()
            compiled = (output / "textures/cliff_0_3.dds").read_bytes()
            self.assertEqual(struct.unpack_from("<I", compiled, 128)[0], 71)
            self.assertEqual(preserved[:128], compiled[:128])
            self.assertEqual(preserved[132:], compiled[132:])

    def test_decode_height_and_lighting(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "natural-data"
            subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "Renderer/lab/shared/natural/test_data.cpp"),
                            str(ROOT / "Renderer/native/environment_runtime.cpp"),
                            "-o", str(binary)], check=True, capture_output=True)
            args = [str(binary)]
            local_pack = ROOT / "Renderer/packs/NaturalFidelityRuntime/natural.bin"
            if local_pack.exists():
                args.append(str(ROOT))
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            self.assertIn("23 invalid inputs, height sampling and 24 lighting phases", result.stdout)
            if local_pack.exists():
                self.assertIn("PASS production natural payload:", result.stdout)
            for pack in ("TerrainNormalized", "Civ5EnvironmentSkin"):
                relative="Renderer/packs/"+pack+"/natural_runtime"
                if (ROOT / relative / "natural.bin").exists():
                    result=subprocess.run([str(binary),str(ROOT),relative],check=True,capture_output=True,text=True)
                    self.assertIn("PASS production natural payload:",result.stdout)

    def test_terrain_pack_materials_and_height_fields_are_selected_independently(self):
        signatures=[]
        for name in ("TerrainNormalized", "Civ5EnvironmentSkin"):
            pack=ROOT / "Renderer/packs" / name
            runtime=pack / "natural_runtime"
            if not (runtime / "natural.bin").exists():
                self.skipTest("Prepared local terrain packs are unavailable")
            data=(runtime / "natural.bin").read_bytes()
            count=struct.unpack_from('<6I',data,8)[0]
            paths=[];cursor=32
            for _ in range(count):
                length=struct.unpack_from('<I',data,cursor)[0];cursor+=4
                paths.append(data[cursor:cursor+length].decode());cursor+=length
            bindings=struct.unpack_from('<57I',data,cursor)
            # Grass, hill tops, plains, tundra, desert, mountain material and
            # all five macro fields must come from this pack, even where one
            # pack deduplicates channels that the other keeps distinct.
            channels={14:'relief/hills/standard/height_lod0',30:'relief_surface_detail'}
            for first,family in ((0,'grassland'),(3,'grasshill_top'),(6,'plains'),
                    (9,'plainshill_top'),(15,'tundra_blend'),(19,'desert'),
                    (37,'mtn_base'),(40,'mtn_top'),(43,'mtn_snow')):
                for offset,role in enumerate(('base_color','height','specular')):
                    channels[first+offset]=family+'_'+role
            for variant in range(5):
                for offset,role in enumerate(('height','blend')):
                    channels[47+2*variant+offset]=f'relief/mountains/standard/variant_{variant+1:02d}/{role}_lod0'
            for slot,source in channels.items():
                expected=(pack / ('textures/'+source+'.dds')).read_bytes()
                actual=(runtime / paths[bindings[slot]]).read_bytes()
                self.assertEqual(actual,expected,name+': '+source)
            signatures.append(hashlib.sha256((runtime / paths[bindings[0]]).read_bytes()).hexdigest())
        self.assertNotEqual(*signatures)


if __name__ == "__main__":
    unittest.main()
