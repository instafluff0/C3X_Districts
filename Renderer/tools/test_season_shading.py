"""Season overlay: marked, season-guarded insertions that leave Summer and the runtime patches intact."""
import re
import unittest
from pathlib import Path

from Renderer.lab.shared.seasons import seasons

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "Renderer/packs/Renderer64ResidentRuntime"
# Unguarded insertions: file-scope libraries, the season index, the natural
# Frame members, the shared tree varyings and neutral declarations.
UNGUARDED = re.compile(r"^(core/|[a-z]+/library$|spring/locals$)")


def trees():
    yield "checkout", ROOT
    if all((PACK / relative).is_file() for relative in seasons.TARGETS):
        yield "pack", PACK


class SeasonShading(unittest.TestCase):
    def test_applies_strips_and_reapplies_exactly(self):
        for name, root in trees():
            for relative in seasons.TARGETS:
                with self.subTest(tree=name, file=relative):
                    text = seasons.strip((root / relative).read_text())
                    applied = seasons.apply(relative, text)
                    self.assertNotEqual(applied, text)
                    self.assertEqual(seasons.strip(applied), text)
                    self.assertEqual(seasons.apply(relative, applied), applied)

    def test_missing_anchor_fails_loudly(self):
        text = (ROOT / seasons.TERRAIN).read_text().replace(seasons.GRAIN, "")
        with self.assertRaisesRegex(ValueError, "anchor occurs 0 times"):
            seasons.apply(seasons.TERRAIN, text)

    def test_every_season_edit_is_guarded(self):
        for edit in seasons.EDITS:
            if UNGUARDED.match(edit.tag) or edit.tag.startswith("water/"):  # water: one guard per season
                continue
            season = {"autumn": 1, "winter": 2, "spring": 3}[edit.tag.split("/")[0]]
            with self.subTest(edit=edit.tag):
                guard = f"[branch] if (C3X_SEASON_INDEX == {season}) {{"
                self.assertIn(guard, edit.code)
                # Outside the guard only declarations with neutral values remain.
                outside = edit.code[:edit.code.index(guard)].rsplit("\n", 1)[0]
                for statement in filter(None, (part.strip() for part in outside.split(";"))):
                    self.assertRegex(statement, r"^(float[234]?|bool) [a-z_]+ ?= ?(0|false|1|n|lerp\(1,ao,WINTER_ROOF_AO\))$")

    def test_no_insertion_becomes_a_braceless_body(self):
        # Code inserted right after `if (...)` or `else` would become that branch's
        # whole body and push the original statement out of it (the water dispatch
        # once sent every land pixel down the water path).
        for name, root in trees():
            for relative in seasons.TARGETS:
                base = seasons.strip((root / relative).read_text())
                first = {}
                for edit in seasons.EDITS:
                    if relative in edit.files:
                        first.setdefault((edit.head, edit.tail), edit)
                for (head, tail), edit in first.items():
                    at = base.index(head + tail) + len(head)
                    lines = [line.split("//")[0].strip() for line in base[:at].split("\n")]
                    previous = next((line for line in reversed(lines) if line and not line.startswith("#")), "")
                    if re.search(r"(\)|\belse)$", previous) and not previous.startswith(("float", "return")):
                        with self.subTest(tree=name, file=relative, edit=edit.tag):
                            self.assertTrue(edit.code.lstrip().startswith("{"), previous)

    def test_runtime_rewritten_ranges_carry_no_season_code(self):
        # sandbox_terrain_material_fast replaces these ranges in Renderer64.
        for name, root in trees():
            text = seasons.apply(seasons.TERRAIN, (root / seasons.TERRAIN).read_text())
            for begin, end in (("        float3 grass_fine = GrassColor.Sample(Wrap, uv0).rgb;", "        float hill_ratio ="),
                               ("        float3 weights=pow(abs(face),4);",
                                "        geometric=normalize(lerp(geometric,face,exposure));")):
                with self.subTest(tree=name, range=begin.strip()):
                    start = text.index(begin)
                    self.assertNotIn(seasons.BEGIN, text[start:text.index(end, start)])

    def test_natural_frame_season_is_the_seventh_float4(self):
        for name, root in trees():
            for relative in seasons.NATURAL:
                with self.subTest(tree=name, file=relative):
                    text = seasons.apply(relative, (root / relative).read_text())
                    block = re.search(r"cbuffer Frame[^{]*\{(.*?)\};", text, re.S).group(1)
                    members = re.findall(r"^\s*float4 (\w+);", re.sub(r"/\*.*?\*/", "", block, flags=re.S), re.M)
                    self.assertEqual(members.index("C3XSeason"), 6)

    def test_season_override_reaches_every_environment_upload(self):
        # The bridge re-uploads TerrainSettings and the natural Frame every frame;
        # with the frame's own season it replaced the fresh pipeline's pinned one.
        bridge = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        upload = bridge.index("context->UpdateSubresource(terrain_settings_buffer, 0, nullptr, &frame_settings, 0, 0);")
        evaluate = bridge.rindex("evaluate_environment(", 0, upload)
        self.assertIn("frame_season(frame)", bridge[evaluate:bridge.index(";", evaluate)])
        fresh = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        update = fresh[fresh.index("void update_environment("):]
        self.assertIn("renderer.frame_season(frame)", update[:update.index("evaluate_environment(")])

    def test_coast_and_beds_keep_summer_water(self):
        # User, 2026-10-22: the translucent Summer coast in every season. Every
        # seasonal water grade is weighted by the open-sea fraction (0 on Coast,
        # 0 on shallow beds); Winter's bed ice is gone.
        game = seasons.apply(seasons.WATER_SURFACE, (ROOT / seasons.WATER_SURFACE).read_text())
        self.assertEqual(game.count("smoothstep(.34, .63, saturate(input.surface_coordinate))"), 3)
        lab = seasons.apply(seasons.HYDROLOGY, (ROOT / seasons.HYDROLOGY).read_text())
        self.assertIn("input.surface_kind < 4.5 ? 0 : smoothstep(.34, .63, saturate(input.surface_coordinate))", lab)
        for name in ("autumn_strength", "spring_strength"):
            line = lab[lab.index(f"float {name} ="):]
            self.assertIn("* c3x_open_sea;", line[:line.index(";") + 1])
        winter = lab[lab.index("c3x-season:winter/water"):]
        self.assertIn("c3x_open_sea", winter[:winter.index("c3x-season-end")])
        self.assertNotIn("WINTER_ICE_BED", winter[:winter.index("c3x-season-end")])

    def test_spring_flower_ground_keeps_the_turf(self):
        # User, 2026-10-23: the leaf shade made ground under the flower drifts
        # look gritty. Only the blooms themselves change the ground's colour.
        relief = seasons.apply(seasons.TERRAIN, (ROOT / seasons.TERRAIN).read_text())
        spring = relief[relief.index("c3x-season:spring/ground"):]
        spring = spring[:spring.index("c3x-season-end")]
        self.assertNotIn("SPRING_LEAF_SHADE", spring)
        self.assertIn("albedo = albedo * (1 - spring_bloom.a) + spring_bloom.rgb;", spring)

    def test_game_water_recolour_is_relative(self):
        # Coast, Sea and Ocean keep their family tint: a hue offset, never a pull to one hue.
        text = seasons.apply(seasons.WATER_SURFACE, (ROOT / seasons.WATER_SURFACE).read_text())
        insertion = text[text.index(seasons.BEGIN):text.rindex(seasons.END)]
        for season in (1, 2, 3):
            self.assertIn(f"C3X_SEASON_INDEX == {season}", insertion)
        self.assertNotRegex(insertion, r"lch\.z = lerp\(")
        self.assertEqual(insertion.count("lch.z = hue + "), 3)


if __name__ == "__main__":
    unittest.main()
