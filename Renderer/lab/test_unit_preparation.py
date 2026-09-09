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


if __name__ == "__main__":
    unittest.main()
