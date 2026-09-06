#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.definitions import definition_parser as parser


class DefinitionParserTests(unittest.TestCase):
    def parse(self, text: str, root: Path, layer: str = "default", name: str | None = None):
        source = root / (name or f"{layer}.custom_rendering.txt")
        return parser.parse_definitions(text, source, layer, root, root / "scenario")

    def test_starter_fixture_parses_typed_values_and_references(self) -> None:
        fixture = Path("Renderer/samples/config/default.custom_rendering.txt")
        definitions = parser.parse_definition_file(fixture, "default", Path.cwd(), Path.cwd())
        catalog = parser.merge_layers([("default", definitions)])

        self.assertEqual(catalog["schema"], "c3x.renderer_definition_catalog.v0")
        self.assertEqual(len(catalog["rules"]), 2)
        self.assertEqual(catalog["rules"][0]["values"]["priority"], 100)
        self.assertEqual(catalog["assets"][0]["values"]["anchor_x"], 0.5)
        self.assertEqual(catalog["environments"][0]["values"]["noon_sun_color"], [255, 244, 220])
        self.assertEqual(
            catalog["packs"][0]["values"]["path"],
            {
                "raw": r"mod:Renderer\packs\GrasslandNormalized",
                "root": "mod",
                "path": "Renderer/packs/GrasslandNormalized",
            },
        )

    def test_later_layer_replaces_complete_section_and_can_disable_rule(self) -> None:
        default_text = """#Pack
id = world
path = mod:packs/base
#Asset
id = grass
pack = world
asset = terrain/grass
scale = 2.0
#Rule
id = grass.rule
category = terrain
asset = grass
priority = 10
"""
        scenario_text = """#Asset
id = grass
pack = world
asset = terrain/scenario-grass
"""
        custom_text = """#Rule
id = grass.rule
disabled = true
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = [
                ("default", self.parse(default_text, root, "default")),
                ("scenario", self.parse(scenario_text, root, "scenario")),
                ("custom", self.parse(custom_text, root, "custom")),
            ]
            catalog = parser.merge_layers(layers)

        self.assertEqual(catalog["assets"][0]["values"]["asset"], "terrain/scenario-grass")
        self.assertNotIn("scale", catalog["assets"][0]["values"])
        self.assertEqual(catalog["assets"][0]["source"]["layer"], "scenario")
        self.assertEqual(catalog["rules"], [])
        self.assertEqual(catalog["disabled"][0]["id"], "grass.rule")

    def test_unknown_key_and_invalid_value_report_file_line_section_and_key(self) -> None:
        text = """#Rule
id = broken.rule
category = terrain
priority = fast
unknown_selector = yes
asset = grass
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(parser.DefinitionError) as raised:
                self.parse(text, root)

        diagnostics = raised.exception.diagnostics
        self.assertEqual([(item.line, item.key) for item in diagnostics], [(4, "priority"), (5, "unknown_selector")])
        self.assertTrue(all(item.file.endswith("default.custom_rendering.txt") for item in diagnostics))
        self.assertTrue(all(item.section_type == "Rule" and item.section_id == "broken.rule" for item in diagnostics))

    def test_missing_required_profile_fallback_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(parser.DefinitionError) as raised:
                self.parse("#Profile\nid = default\n", Path(tmp))

        diagnostic = raised.exception.diagnostics[0]
        self.assertEqual((diagnostic.line, diagnostic.section_type, diagnostic.section_id, diagnostic.key), (1, "Profile", "default", "missing_asset"))

    def test_path_prefixes_and_root_escape_are_validated(self) -> None:
        cases = (
            ("mod:packs/world", False, "mod", "packs/world"),
            (r"scenario:Art\Packs\World", False, "scenario", "Art/Packs/World"),
            ("file:dev/world", True, "file", None),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = root / "scenario"
            scenario.mkdir()
            source = root / "config" / "custom.custom_rendering.txt"
            source.parent.mkdir()
            for raw, allow_file, root_kind, normalized in cases:
                with self.subTest(raw=raw):
                    result = parser.resolve_pack_path(raw, source, root, scenario, allow_file)
                    self.assertEqual(result["root"], root_kind)
                    if normalized is not None:
                        self.assertEqual(result["path"], normalized)
            with self.assertRaisesRegex(ValueError, "escapes"):
                parser.resolve_pack_path("mod:../outside", source, root, scenario)
            with self.assertRaisesRegex(ValueError, "disabled"):
                parser.resolve_pack_path("file:dev/world", source, root, scenario)
            with self.assertRaisesRegex(ValueError, "escapes"):
                parser.resolve_pack_asset_path(root / "pack", "../outside.glb")

    def test_hours_seasons_and_boolean_selectors_are_typed(self) -> None:
        text = """#Rule
id = seasonal.rule
category = terrain
asset = grass
show_in_day_night_hours = 18-5, 12
show_in_seasons = spring, autumn
has_forest = false
"""
        with tempfile.TemporaryDirectory() as tmp:
            definition = self.parse(text, Path(tmp))[0]

        self.assertEqual(definition.values["show_in_day_night_hours"], [{"start": 18, "end": 5}, 12])
        self.assertEqual(definition.values["show_in_seasons"], ["spring", "fall"])
        self.assertIs(definition.values["has_forest"], False)

    def test_merge_rejects_dangling_asset_and_pack_references(self) -> None:
        text = """#Asset
id = grass
pack = missing.pack
asset = terrain/grass
#Rule
id = grass.rule
category = terrain
asset = missing.asset
"""
        with tempfile.TemporaryDirectory() as tmp:
            definitions = self.parse(text, Path(tmp))
            with self.assertRaises(parser.DefinitionError) as raised:
                parser.merge_layers([("default", definitions)])

        self.assertEqual({item.key for item in raised.exception.diagnostics}, {"pack", "asset"})

    def test_duplicate_ids_and_layer_order_are_rejected(self) -> None:
        duplicate = """#Pack
id = world
path = mod:a
#Pack
id = world
path = mod:b
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(parser.DefinitionError, "duplicate section type and ID"):
                self.parse(duplicate, root)
            default = self.parse("#Pack\nid = a\npath = mod:a\n", root, "default")
            custom = self.parse("#Pack\nid = b\npath = mod:b\n", root, "custom")
            with self.assertRaisesRegex(ValueError, "ordered"):
                parser.merge_layers([("custom", custom), ("default", default)])


if __name__ == "__main__":
    unittest.main()
