"""Unit packaging preserves native keys, source vertices and action palettes."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from Renderer.native.environment_refresh import prepare_units as units


class UnitPreparation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.packs = self.root / "Renderer/packs"
        for name, value in (("ROOT", self.root), ("PACKS", self.packs)):
            replacement = patch.object(units, name, value)
            replacement.start(); self.addCleanup(replacement.stop)
        self.write(self.root / "Renderer/native/environment_refresh/unit_quality.json", {"frame_pack":"UnitFrameFidelity", "units":{}})
        self.source = self.packs / "UnitAnimationRuntime"
        self.output = self.root / "Renderer/candidate"
        self.mesh = {"vertices": [{"position": [x, y, 0], "normal": [0, 0, 1], "uv0": [x, y]}
                                  for x, y in ((0, 0), (1, 0), (0, 1))], "topology": {"indices": [0, 1, 2]}}
        self.normal_mesh = self.packs / "Imported/mesh.json"
        self.write(self.normal_mesh, self.mesh)
        self.write(self.packs / "Imported/manifest.json", {})
        self.normal = self.packs / "UnitNormalFidelity/normal.json"
        self.write(self.normal, {"normal_source": "authored_octahedral_snorm8", "address_mode": "clamp",
            "normalized_mesh_sha256": units.digest(self.normal_mesh), "normals": [[0, 1, 0]] * 3})
        self.write(self.normal.parent / "manifest.json", {"meshes": {
            "Imported/mesh.json": self.normal.relative_to(self.root).as_posix()}})
        self.blob = b"C3XANM1\0" + struct.pack("<6I", 1, 3, 3, 1, 2, 0)
        for vertex in self.mesh["vertices"]:
            self.blob += struct.pack("<8f4I4f", *vertex["position"], *vertex["normal"], *vertex["uv0"],
                                     0, 0, 0, 0, 1, 0, 0, 0)
        self.blob += struct.pack("<3I", 0, 1, 2) + b"unchanged action palette and attachment bytes"
        self.payload = self.source / "clips/source.bin"
        self.payload.parent.mkdir(parents=True)
        self.payload.write_bytes(self.blob)
        self.texture = self.source / "textures/base.dds"
        self.texture.parent.mkdir()
        self.texture.write_bytes(b"unchanged texture mip chain")
        part = {"mesh": "clips/source.bin", "source_mesh": "mesh.json", "material": {"channels": {
            "base_color": {"texture": "textures/base.dds", "address_u": "repeat", "address_v": "repeat"}}}}
        self.manifest = {"units": {name: {"source_pack": pack, "civ3_ids": [key],
            "actions": {"default": {"parts": [part]}}}
            for name, pack, key in (("unit/imported", "Imported", 1), ("unit/original", "Original", 2))}}
        self.write(self.source / "manifest.json", self.manifest)
        self.write(self.source / "bindings.json", {"unit" + str(key): {"key_count": 1, "key0": key,
            "default": {"part0": {"mesh": "clips/source.bin", "base_texture": "textures/base.dds"}}}
            for key in (1, 2)})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_only_recovered_normal_fields_change(self):
        evidence, consumed = units.build_pack(self.output)
        manifest = json.loads((self.output / "manifest.json").read_text())
        part = manifest["units"]["unit/imported"]["actions"]["default"]["parts"][0]
        expected = bytearray(self.blob)
        for index in range(3):
            struct.pack_into("<3f", expected, 32 + index * 64 + 12, 0, 1, 0)
        self.assertEqual((self.output / part["mesh"]).read_bytes(), bytes(expected))
        self.assertEqual((self.output / "clips/source.bin").read_bytes(), self.blob)
        self.assertEqual(self.payload.read_bytes(), self.blob)
        self.assertEqual(evidence["unit_count"], 2)
        self.assertEqual(evidence["native_keys"], 2)
        self.assertEqual(evidence["unchanged_palette_frames"], 2)
        self.assertIsNone(consumed["Renderer/packs/Original/manifest.json"])
        self.assertIn("Renderer/packs/Imported/mesh.json", consumed)
        self.assertEqual((self.output / "textures/base.dds").read_bytes(), self.texture.read_bytes())
        self.assertFalse((self.output / "textures/base.dds").samefile(self.texture))
        bindings = json.loads((self.output / "bindings.json").read_text())
        self.assertEqual(bindings["unit1"]["default"]["part0"]["address_mode"], 3)
        self.assertEqual(bindings["unit2"]["default"]["part0"]["address_mode"], 0)

    def test_normal_source_fingerprint_must_match(self):
        self.normal_mesh.write_text("{}")
        with self.assertRaisesRegex(ValueError, "normalized mesh changed"):
            units.build_pack(self.output)
        self.assertFalse((self.output / "manifest.json").exists())

    def test_existing_modified_normal_payload_is_preserved(self):
        units.build_pack(self.output)
        manifest = json.loads((self.output / "manifest.json").read_text())
        part = manifest["units"]["unit/imported"]["actions"]["default"]["parts"][0]
        target = self.output / part["mesh"]
        target.write_bytes(b"local edit")
        with self.assertRaisesRegex(ValueError, "normal payload collision"):
            units.build_pack(self.output)
        self.assertEqual(target.read_bytes(), b"local edit")

    def test_imported_component_cannot_fall_back_to_procedural_normals(self):
        self.write(self.normal.parent / "manifest.json", {"meshes": {}})
        with self.assertRaisesRegex(ValueError, "missing component normal authority"):
            units.build_pack(self.output)

    def test_source_directory_cannot_be_used_as_output(self):
        for output in (self.packs, self.source, self.source / "candidate", self.packs / "Imported"):
            with self.assertRaisesRegex(ValueError, "overlap"):
                units.build_pack(output)

    def test_payload_paths_cannot_escape_the_source_pack(self):
        self.manifest["units"]["unit/imported"]["actions"]["default"]["parts"][0]["mesh"] = "../outside.bin"
        self.write(self.source / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "escapes"):
            units.build_pack(self.output)

    def test_changed_topology_and_uv_are_rejected(self):
        for offset, value, message in ((32 + 3 * 64, struct.pack("<I", 2), "source index order"),
                                       (32 + 6 * 4, struct.pack("<f", .5), "UV0 changed")):
            changed = bytearray(self.blob)
            changed[offset:offset + len(value)] = value
            self.payload.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, message):
                units.build_pack(self.output)

    def test_look_metadata_is_published_and_bounded(self):
        quality = self.root / "Renderer/native/environment_refresh/unit_quality.json"
        self.write(quality, {"frame_pack": "UnitFrameFidelity", "units": {},
                             "look": {"gain": 0.5, "saturation": 0.25, "owner": 1.0}})
        units.build_pack(self.output)
        bindings = json.loads((self.output / "bindings.json").read_text())
        self.assertEqual((bindings["look_gain"], bindings["look_saturation"], bindings["look_owner"]), (0.5, 0.25, 1.0))
        self.write(quality, {"frame_pack": "UnitFrameFidelity", "units": {}, "look": {"gain": 3}})
        with self.assertRaises(ValueError):
            units.build_pack(self.root / "Renderer/candidate2")

    def test_native_sprite_area_sizing_and_shape_neighbours(self):
        # A standing quad 0.2 wide and 0.5 tall; identity skin palette.
        quad = [(-.1, 0, 0), (.1, 0, 0), (.1, 0, .5), (-.1, 0, .5)]
        blob = b"C3XANM1\0" + struct.pack("<6I", 1, 4, 6, 1, 1, 0)
        for x, y, z in quad:
            blob += struct.pack("<8f4I4f", x, y, z, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
        blob += struct.pack("<6I", 0, 1, 2, 0, 2, 3) + struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        target = self.root / "Renderer/sized"
        (target / "clips").mkdir(parents=True)
        (target / "clips/quad.bin").write_bytes(blob)
        idle = {"part_count": 1, "part0": {"mesh": "clips/quad.bin"}}
        bindings = {"unit0": {"key_count": 1, "key0": "PRTO_A", "scale": 1.0, "offset_z": 0.0, "idle": idle},
                    "unit1": {"key_count": 1, "key0": "PRTO_Custom", "scale": 1.0, "offset_z": 0.0, "idle": idle},
                    "unit_count": 2}
        ours = units.silhouette(*units.idle_geometry(blob), 1.0, 0.0, 225.0)
        sprites = {"PRTO_A": {"area": ours["area"] * 4}}
        self.assertEqual(units.fit_sizes(bindings, target, sprites, 1.0), (1, 1))
        # Area grows with the square of scale: four times the area doubles it.
        self.assertAlmostEqual(bindings["unit0"]["scale"], 2.0, places=6)
        self.assertEqual(bindings["unit0"]["fit_policy"], "native_sprite_area")
        # A unit without a native sprite takes the change of similar shapes.
        self.assertAlmostEqual(bindings["unit1"]["scale"], 2.0, places=6)
        self.assertEqual(bindings["unit1"]["fit_policy"], "native_sprite_area_shape_neighbors")

    def test_stray_rigid_primitive_does_not_lift_a_skinned_body(self):
        def payload(name, points, bones):
            # Vertices skin to joint 0; a skinned payload carries `bones` identity joints.
            blob = b"C3XANM1\0" + struct.pack("<6I", 1, len(points), 3 * (len(points) - 2), bones, 1, 0)
            for x, y, z in points:
                blob += struct.pack("<8f4I4f", x, y, z, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
            blob += struct.pack(f"<{3 * (len(points) - 2)}I", *[i for k in range(len(points) - 2) for i in (0, k + 1, k + 2)])
            blob += struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1) * bones
            (target / "clips" / name).write_bytes(blob)
        target = self.root / "Renderer/grounded"
        (target / "clips").mkdir(parents=True)
        body = [(x, y, z) for z in (.3, .6, .9, 1.2) for x, y in ((-.1, 0), (.1, 0), (0, .1))] * 20
        payload("body.bin", body, 2)            # a skinned figure whose feet are at 0.3
        payload("stick.bin", [(0, 0, 0), (.01, 0, 0), (0, 0, .5)], 1)   # stray rigid pole from the origin
        payload("hull.bin", [(x, y, 0) for x in (-1, 1) for y in (-1, 1)] * 20, 1)  # a large rigid hull at 0
        def binding(parts):
            idle = {"part_count": len(parts), **{f"part{i}": {"mesh": "clips/" + m} for i, m in enumerate(parts)}}
            return {"key_count": 1, "key0": "PRTO_" + parts[-1], "scale": 1.0, "offset_z": 0.0, "idle": idle}
        bindings = {"unit0": binding(["body.bin", "stick.bin"]), "unit1": binding(["body.bin", "hull.bin"]), "unit_count": 2}
        self.assertEqual(units.ground_skinned_bodies(bindings, target), ["PRTO_stick.bin"])
        self.assertAlmostEqual(bindings["unit0"]["offset_z"], -.3, places=5)
        self.assertEqual(bindings["unit1"]["offset_z"], 0.0)  # the hull really is the ground

    def column(self, target, name, low, high):
        # A vertical identity-skinned strip from z=low to z=high.
        points = [(-.1, 0, low), (.1, 0, low), (.1, 0, high), (-.1, 0, high)]
        blob = b"C3XANM1\0" + struct.pack("<6I", 1, 4, 6, 1, 1, 0)
        for x, y, z in points:
            blob += struct.pack("<8f4I4f", x, y, z, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
        blob += struct.pack("<6I", 0, 1, 2, 0, 2, 3) + struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        (target / "clips").mkdir(parents=True, exist_ok=True)
        (target / "clips" / name).write_bytes(blob)
        return {"key_count": 1, "key0": "PRTO_" + name, "scale": 2.0, "offset_z": -low,
                "idle": {"part_count": 1, "part0": {"mesh": "clips/" + name}}}

    def test_sea_units_float_at_their_authored_waterline(self):
        target = self.root / "Renderer/afloat"
        bindings = {"unit0": self.column(target, "hull", -.1, .4),   # authored below its origin
                    "unit1": self.column(target, "raft", .2, 1.2),   # nothing below its origin
                    "unit2": self.column(target, "cart", -.1, .4), "unit_count": 3}
        domains = {"PRTO_hull": "sea", "PRTO_raft": "sea", "PRTO_cart": "land"}
        self.assertEqual(units.float_at_waterline(bindings, target, domains, {"sea"}, .15), ["PRTO_hull", "PRTO_raft"])
        self.assertEqual((bindings["unit0"]["offset_z"], bindings["unit0"]["ground_policy"]), (0.0, "authored_waterline"))
        # Without its own waterline a hull takes its authored peers' (20% of height).
        self.assertAlmostEqual(bindings["unit1"]["offset_z"], -(.2 + .2 * 1.0))
        self.assertEqual(bindings["unit1"]["ground_policy"], "shape_neighbor_waterline")
        self.assertEqual(bindings["unit2"]["offset_z"], .1)  # land keeps its ground contact
        # With no authored peers the configured draft applies.
        alone = {"unit0": self.column(target, "skiff", 0, .5), "unit_count": 1}
        units.float_at_waterline(alone, target, {"PRTO_skiff": "sea"}, {"sea"}, .15)
        self.assertAlmostEqual(alone["unit0"]["offset_z"], -.075)
        # Sizing then measures only the part above the water plane.
        above = units.silhouette(*units.idle_geometry((target / "clips/hull").read_bytes()), 2.0, 0.0, 225.0)
        whole = units.silhouette(*units.idle_geometry((target / "clips/hull").read_bytes()), 2.0, .1, 225.0)
        self.assertAlmostEqual(above["height"] / whole["height"], .8, delta=.03)

    def test_flying_units_hover_at_their_native_sprite_lift(self):
        target = self.root / "Renderer/hover"
        bindings = {"unit0": self.column(target, "plane", .05, .3), "unit1": self.column(target, "walker", 0, .5),
                    "unit_count": 2}
        sprites = {"PRTO_plane": {"lift": 30.0}, "PRTO_walker": {"lift": -4.0}}
        self.assertEqual(units.hover_flying_units(bindings, target, sprites, .5, 8), ["PRTO_plane"])
        plane = bindings["unit0"]
        # The lowest idle point is factor * lift pixels above the ground.
        self.assertAlmostEqual((.05 + plane["offset_z"]) * plane["scale"] * units.Z_PIXELS, 15.0, places=4)
        self.assertEqual(plane["hover_policy"], "native_sprite_lift")
        self.assertEqual(bindings["unit1"]["offset_z"], 0.0)
        self.assertNotIn("flight_lift", plane)

    def test_flying_units_climb_to_their_flight_lift(self):
        target = self.root / "Renderer/flight"
        bindings = {"unit0": self.column(target, "plane", .05, .3), "unit1": self.column(target, "walker", 0, .5),
                    "unit_count": 2}
        sprites = {"PRTO_plane": {"lift": 30.0}, "PRTO_walker": {"lift": -4.0}}
        units.hover_flying_units(bindings, target, sprites, .5, 8, flight=2.0)
        plane = bindings["unit0"]
        # In flight the lowest point is 2 sprite lifts up: 1.5 lifts above the hover.
        self.assertAlmostEqual(plane["flight_lift"] * plane["scale"] * units.Z_PIXELS, 45.0, places=3)
        self.assertNotIn("flight_lift", bindings["unit1"])


if __name__ == "__main__":
    unittest.main()
