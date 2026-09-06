import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CPP = (ROOT / "Renderer" / "terrain_lab" / "terrain_lab.cpp").read_text(encoding="utf-8")
HLSL = (ROOT / "Renderer" / "terrain_lab" / "terrain_lab.hlsl").read_text(encoding="utf-8")
EXPORTER = (ROOT / "Renderer" / "tools" / "export_biq_terrain_scene.js").read_text(
    encoding="utf-8"
)
SHORE_BUILDER = (
    ROOT / "Renderer" / "tools" / "asset_compiler" / "build_shore_runtime.py"
).read_text(encoding="utf-8")
L13_PREPARE = (ROOT / "Renderer" / "terrain_lab" / "PREPARE_L13_BIQ_VIEWPORT.bat").read_text(
    encoding="utf-8"
)
L13_RUN = (ROOT / "Renderer" / "terrain_lab" / "RUN_L13.bat").read_text(encoding="utf-8")
L13_TOPOLOGY_PREPARE = (
    ROOT / "Renderer" / "terrain_lab" / "PREPARE_L13_RIVER_TOPOLOGY_VIEWPORT.bat"
).read_text(encoding="utf-8")
L13A_RUN = (ROOT / "Renderer" / "terrain_lab" / "RUN_L13A.bat").read_text(
    encoding="utf-8"
)
LAB_BUILD = (ROOT / "Renderer" / "terrain_lab" / "BUILD.bat").read_text(
    encoding="utf-8"
)


class ContinuousSurfaceContractTests(unittest.TestCase):
    def test_biq_surface_is_pass_major_and_contour_clipped(self):
        match = re.search(
            r"void add_biq_patch\([^\{]+\) \{(?P<body>.*?)\n\}", CPP, re.DOTALL
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        ground = body.index("uv_scale, 0.5f")
        relief = body.index("uv_scale, 1.0f")
        bed = body.index("uv_scale, 4.0f")
        water = body.index("uv_scale, 5.0f")
        self.assertLess(ground, relief)
        self.assertLess(relief, bed)
        self.assertLess(bed, water)
        self.assertGreaterEqual(HLSL.count("clip(input.shore_distance - 0.001)"), 2)

    def test_material_and_depth_fields_are_world_continuous(self):
        self.assertIn("void biq_center_material_weights", CPP)
        self.assertIn("float warp_x = std::sin(source_x", CPP)
        self.assertIn("float warp_y = std::sin(source_x", CPP)
        self.assertIn("if (signed_shore_distance <= 0.0f)", CPP)
        self.assertNotIn(
            "if (!is_water_terrain(tile.base))\n        return signed_shore_distance;",
            CPP,
        )

    def test_l12_uses_authored_volcano_channels_and_192_tile_fixture(self):
        self.assertIn("void biq_volcano_sample", CPP)
        self.assertIn("promotion_volcano_height_field", CPP)
        self.assertIn("promotion_volcano_blend_field", CPP)
        self.assertIn("beauty_volcano_thumbnail", CPP)
        self.assertIn("biq_window.columns != 16 || biq_window.rows != 12", CPP)
        self.assertIn("volcano_base_texture", HLSL)
        self.assertIn("volcano_active_base_texture", HLSL)
        self.assertIn("volcano_active_specular_texture", HLSL)
        self.assertIn("input.real_terrain - 10.0", HLSL)
        self.assertIn("void biq_chain_relief_sample", CPP)
        self.assertIn("chain_displacement * coastal_envelope", CPP)
        self.assertIn("authored_relief_blend *=", CPP)
        self.assertIn("biq_relief_envelope(tile, u, v)", CPP)
        self.assertIn("float biq_coastal_relief_envelope", CPP)
        self.assertIn("coastal_envelope", CPP)
        self.assertIn("float biq_mountain_hill_transition_envelope", CPP)
        self.assertIn("float biq_mountain_material_envelope", CPP)
        self.assertIn("biq_mountain_hill_transition_envelope(tile, u, v) * 0.35f", CPP)
        self.assertIn("smooth_relief_max(hill_displacement, authored_displacement)", CPP)
        self.assertIn("tile.real != 5", CPP)
        self.assertIn("73856093u", CPP)
        self.assertIn("active * active_surface.a", HLSL)
        self.assertNotIn("add_volcano_smoke", CPP)
        self.assertNotIn("FireFX_Volcano", CPP + HLSL)
        self.assertNotIn("Volcano_Idle_Smoke", CPP + HLSL)

    def test_biq_vegetation_forms_dense_authored_canopies(self):
        self.assertIn("constexpr unsigned biq_forest_density = 36", CPP)
        self.assertIn("constexpr unsigned biq_jungle_density = 49", CPP)
        self.assertIn("bool dense_biq_canopy", CPP)
        self.assertIn("unsigned grid_side = group.name == \"forest\" ? 6u : 7u", CPP)
        self.assertIn("group.name == \"forest\" ? 0.42f : 0.40f", CPP)
        self.assertNotIn("compact_biq_jungle", CPP)

    def test_l12_uses_source_backed_land_and_underwater_clutter(self):
        for texture in (
            "water_decal_base_texture",
            "water_decal_height_texture",
            "grassland_decal_base_texture",
            "grassland_decal_height_texture",
            "plains_decal_base_texture",
            "plains_decal_height_texture",
        ):
            self.assertIn(texture, HLSL)
        self.assertIn("float4 sample_water_clutter", HLSL)
        self.assertIn("float4 sample_land_clutter", HLSL)
        self.assertIn("water_clutter.a", HLSL)
        self.assertIn("input.material_weights", HLSL)
        self.assertIn("clip(input.shore_distance - 0.001)", HLSL)
        self.assertNotIn("add_water_rock", CPP)

    def test_l13_uses_shared_edge_rivers_and_source_backed_detail(self):
        self.assertIn("C3X_BIQ_TERRAIN_WINDOW_V2", CPP)
        self.assertIn("tile.river_mask", CPP)
        self.assertIn("void build_river_graph()", CPP)
        self.assertIn("add_river_graph_edge", CPP)
        self.assertIn("direction_bit == 128u", CPP)
        self.assertIn("tile.column + tile.row", CPP)
        self.assertIn("tile.column - tile.row", CPP)
        self.assertNotIn("tile.source_x - first.source_x", CPP)
        self.assertNotIn("biq_tile->source_x - first.source_x", CPP)
        self.assertIn("beauty_rivers", CPP)
        self.assertIn("beauty_rivers_no_rivers", CPP)
        self.assertIn("beauty_rivers_only", CPP)
        self.assertIn("beauty_rivers_thumbnail", CPP)
        self.assertIn("river_bank_noise_texture", HLSL)
        self.assertIn("river_source_base_texture", HLSL)
        self.assertIn("river_clutter_base_texture", HLSL)
        self.assertIn("river_lean0_texture", HLSL)
        self.assertIn("river_lean1_texture", HLSL)
        self.assertIn("river_rock_base_texture_4", HLSL)
        self.assertIn("feature_base_texture_7", HLSL)
        self.assertIn("asset.texture_index += 8u", CPP)
        self.assertIn("add_river_rock_scene", CPP)
        self.assertNotIn("waterfall", CPP.lower())

    def test_l13_fixture_is_192_tiles_deterministic_and_wrap_qualified(self):
        self.assertIn("riverTopologyMetrics", EXPORTER)
        self.assertIn("riverMask", EXPORTER)
        self.assertIn("--require-all-preferred", EXPORTER)
        self.assertIn("--require-wrap", EXPORTER)
        self.assertIn("--window-columns 16 --window-rows 12", L13_PREPARE)
        self.assertIn("--prefer-real marsh", L13_PREPARE)
        self.assertIn("--prefer-real jungle", L13_PREPARE)
        self.assertIn("--prefer-real volcano", L13_PREPARE)
        self.assertIn("--prefer-real ocean", L13_PREPARE)
        self.assertIn("--require-river", L13_PREPARE)
        self.assertIn("--require-wrap", L13_PREPARE)
        self.assertIn("test_biq_l13_rivers_192.csv", L13_RUN)
        self.assertIn("test_biq_l13_river_topology_192.csv", L13_RUN)
        self.assertIn("--prefer-river", L13_TOPOLOGY_PREPARE)
        self.assertIn("--require-wrap", L13_TOPOLOGY_PREPARE)
        self.assertIn("Civ5EnvironmentSkin", L13_RUN)
        self.assertIn("Civ5EnvironmentVegetation", L13_RUN)

    def test_l13_river_rocks_use_all_verified_normalized_variants(self):
        self.assertIn('manifest["feature_sets"]["river_rock"]["variants"]', SHORE_BUILDER)
        self.assertIn("for texture_index, asset_id in enumerate(river_ids)", SHORE_BUILDER)
        self.assertIn("shore_runtime.bin", SHORE_BUILDER)
        self.assertIn("river_feature_bundle.texture_paths.size() == 5u", CPP)
        self.assertIn("shore_runtime.bin", L13_RUN)

    def test_l13a_uses_shared_environment_and_eight_locked_fixtures(self):
        self.assertIn("environment_runtime.cpp", LAB_BUILD)
        self.assertIn("c3x_renderer::evaluate_environment", CPP)
        self.assertIn("frame_environment.sun_direction", CPP)
        self.assertIn("biq_cast_shadow_visibility", CPP)
        self.assertIn("add_biq_shadow_surface", CPP)
        self.assertIn("for (int lane = -1; lane <= 1; ++lane)", CPP)
        self.assertIn("height_shadow_length", CPP)
        self.assertIn("projected_height_ratio", CPP)
        self.assertIn("minimum_shadow_length", CPP)
        self.assertIn("float ray_slope = 96.0f", CPP)
        self.assertIn("float direction_y = -active_light_direction[1] / horizontal", CPP)
        self.assertIn("float cast_world_y = horizontal > 0.001f", CPP)
        self.assertIn("? -active_light_direction[1] / horizontal : 1.0f", CPP)
        self.assertIn("float feature_form", HLSL)
        self.assertIn("float relief_form", HLSL)
        self.assertIn("float raised_form_response", HLSL)
        self.assertIn("float3 vegetation_normal", HLSL)
        self.assertIn("normal.xy * 2.40", HLSL)
        self.assertIn("feature_form = raised_form_response(signed_diffuse)", HLSL)
        self.assertIn("relief_form = raised_form_response(diffuse)", HLSL)
        self.assertIn("shadow_depth_at", CPP)
        self.assertIn("projected_base_y", CPP)
        self.assertIn("feature_height * feature_height_pixels_per_tile", CPP)
        self.assertIn("l13a_scene_enabled", CPP)
        self.assertIn("environment_sun_direction", HLSL)
        self.assertIn("environment_moon_direction", HLSL)
        self.assertIn("environment_shadow_strength", HLSL)
        self.assertIn("environment_night_activation", HLSL)
        self.assertIn("environment_water_specular", HLSL)
        self.assertIn("frame_illumination", HLSL)
        self.assertIn("frame_cast_shadow_strength", HLSL)
        self.assertIn("Civ5EnvironmentSkin", L13A_RUN)
        self.assertIn("Civ5EnvironmentVegetation", L13A_RUN)
        self.assertIn("for %%H in (noon sunset midnight sunrise)", L13A_RUN)
        self.assertIn("beauty_lighting_%%H_zoom2", L13A_RUN)
        self.assertIn("c3x_renderer::evaluate_emissive", CPP)
        self.assertIn("static_redraw=idle", CPP)
        self.assertNotIn("presentation_time_ticks", CPP)
        self.assertNotIn("RUN_L13.bat", L13A_RUN)

if __name__ == "__main__":
    unittest.main()
