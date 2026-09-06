#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


RENDERER_ROOT = Path(__file__).resolve().parents[1]
C3X_ROOT = RENDERER_ROOT.parent


class NativeBridgeContractTests(unittest.TestCase):
    def test_custom_rendering_is_default_off_and_hard_gated(self) -> None:
        config = (C3X_ROOT / "default.c3x_config.ini").read_text(encoding="utf-8")
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        self.assertRegex(config, r"(?m)^enable_custom_rendering\s*=\s*false\s*$")
        off_branch = re.search(
            r"if \(! is->current_config\.enable_custom_rendering\).*?"
            r"Map_Renderer_m71_Draw_Tiles \(this, __, param_1, param_2, param_3\);\s*return;",
            injected,
            re.DOTALL,
        )
        self.assertIsNotNone(off_branch)
        self.assertNotIn("ensure_custom_renderer_loaded", off_branch.group(0))

    def test_custom_rendering_suppresses_legacy_tile_flc_animations(self) -> None:
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        normalized = " ".join(injected.split())
        positive_guard = (
            "is->current_config.enable_custom_animations && "
            "! is->current_config.enable_custom_rendering"
        )
        negative_guard = (
            "! is->current_config.enable_custom_animations || "
            "is->current_config.enable_custom_rendering"
        )

        self.assertGreaterEqual(normalized.count(positive_guard), 10)
        self.assertGreaterEqual(normalized.count(negative_guard), 4)
        self.assertRegex(
            injected,
            r"load_tile_animation_configs \(\)\s*\{\s*"
            r"if \(! is->current_config\.enable_custom_animations \|\| "
            r"is->current_config\.enable_custom_rendering\) \{\s*"
            r"clear_tile_animation_configs \(\);\s*return;",
        )
        self.assertRegex(
            injected,
            r"tile_animation_scheduler_tick \(\)\s*\{\s*"
            r"if \(! is->current_config\.enable_custom_animations \|\| "
            r"is->current_config\.enable_custom_rendering\)\s*return;",
        )

    def test_bridge_uses_proven_pass_boundary_and_bounded_capture(self) -> None:
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(encoding="utf-8")
        self.assertIn("param_8 == 9", injected)
        self.assertIn("param_8 == 0x1FEF0", injected)
        self.assertLess(injected.index("composite_custom_renderer_frame ()"), injected.index("Map_Renderer_m19_Draw_Tile_by_XY_and_Flags (this"))
        self.assertIn("int const max_tiles = 8192", injected)
        self.assertNotIn("restore_civ3_terrain_for_custom_renderer_frame", injected)
        self.assertIn("C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED = 4u", api)
        ownership_mark = injected.index(
            "is->custom_renderer_tiles[n].tile_flags |= output.replacement_tile_flags[n]"
        )
        successful_blit = injected.rindex("if (result == C3X_RENDERER_RESULT_OK) {", 0, ownership_mark)
        self.assertLess(successful_blit, ownership_mark)
        self.assertIn("validate_custom_renderer_replacement_ownership (&output)", injected)
        self.assertIn("output->fallback_tile_count != 0", injected[:ownership_mark])
        m19 = injected[injected.index("patch_Map_Renderer_m19_Draw_Tile_by_XY_and_Flags") :]
        m19 = m19[:m19.index("void __fastcall\npatch_Map_Renderer_m08_Draw_Tile_Forests_Jungle_Swamp")]
        self.assertIn("if (! is->custom_renderer_frame_active)", m19)
        self.assertEqual(1, m19.count("Map_Renderer_m19_Draw_Tile_by_XY_and_Flags (this"))
        self.assertNotIn("draw_flags &=", m19)

    def test_i12_consumes_integrated_generic_assets_without_runtime_handoff_files(self) -> None:
        native = (Path(__file__).parent / "terrain_scene_runtime.cpp").read_text(
            encoding="utf-8"
        )
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        definitions = (Path(__file__).parent / "terrain_definition_runtime.cpp").read_text(
            encoding="utf-8"
        )
        default_definitions = (RENDERER_ROOT / "default.custom_rendering.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("configure_integrated_assets", renderer)
        for pack_id in (
            "vegetation_normalized", "decals_normalized", "terrain_elements_normalized"
        ):
            self.assertIn(pack_id, definitions)
            self.assertIn(f"id = {pack_id}", default_definitions)
        self.assertIn("companion_packs.vegetation", renderer)
        self.assertIn("companion_packs.decals", renderer)
        self.assertIn("companion_packs.terrain_elements", renderer)
        self.assertIn("vegetation_runtime.bin", renderer)
        self.assertIn("terrain_desert_dune_decal_01.json", renderer)
        self.assertIn("terrain_marsh_decal_01.json", renderer)
        self.assertIn("terrain_feature_volcano\\\\height_lod0.dds", renderer)
        self.assertIn("terrain_water_ocean_decal_01.json", renderer)
        self.assertNotIn("load_approved_terrain_handoffs", native + renderer)
        self.assertNotIn("L9_terrain.json", native + renderer)
        self.assertNotIn("L10_dunes.json", native + renderer)
        self.assertNotIn("L11_marsh.json", native + renderer)
        self.assertNotIn("L12_volcano.json", native + renderer)
        for obsolete_runtime_gate in (
            "approved_mode", "approved_ground", "approved_relief", "approved_feature",
            "native-failure=unapproved-terrain",
        ):
            self.assertNotIn(obsolete_runtime_gate, renderer)
        self.assertIn("bool draw_feature = feature_assets_ready", renderer)
        self.assertIn("bool draw_marsh = marsh_assets_ready", renderer)
        self.assertIn("bool draw_dunes = dune_assets_ready", renderer)
        self.assertIn("bool draw_volcano = volcano_assets_ready", renderer)
        relief_type = renderer[renderer.index("static int relief_type(") :]
        relief_type = relief_type[:relief_type.index("static float sample_height_field")]
        self.assertIn("tile.real_terrain_type == 10", relief_type)
        self.assertIn("profile == 5", renderer)
        self.assertIn("volcano_base_texture : register(t69)", shader)
        self.assertIn("volcano_active_base_texture : register(t71)", shader)
        self.assertIn("water_decal_base_texture : register(t73)", shader)
        self.assertIn("views[69] = volcano_base_view", renderer)
        self.assertIn("views[73] = water_clutter_base_view", renderer)
        self.assertIn("instance_count = tile.real_terrain_type == 7 ? 36u : 49u", renderer)
        self.assertIn('"feature/forest/leafy"', renderer)
        self.assertIn("texture_count > 8", native)
        self.assertIn("feature_base_texture_7 : register(t97)", shader)
        self.assertIn("PSSetShaderResources(94, 4, feature_texture_views.data() + 4)", renderer)
        self.assertIn("tile.real_terrain_type == 7 ? 0.42f : 0.40f", renderer)

    def test_m6_7_cache_key_is_terrain_specific_and_revision_aware(self) -> None:
        runtime = (Path(__file__).parent / "terrain_scene_runtime.cpp").read_text(
            encoding="utf-8"
        )
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(encoding="utf-8")
        signature = runtime[runtime.index("terrain_frame_signature(") :]
        for required in (
            "anchor_x", "anchor_y", "tile_width", "tile_height", "world_wrap_x",
            "world_wrap_y", "hour", "season", "content_revision", "device_generation",
            "feature_flags",
            "has_effect", "river_code", "road_mask", "railroad_mask", "route_style",
            "resource_id", "resource_class", "resource_name", "city_id",
            "city_owner_id", "city_size", "city_culture_group",
            "city_era", "city_flags",
        ):
            self.assertIn(required, signature)
        for retained_only in (
            "unit_type_id", "unit_state", "unit_direction", "square_parts",
            "terrain_overlays", "visibility_mask", "city_population",
        ):
            self.assertNotIn(retained_only, signature)
        self.assertIn("viewport_cache_capacity = 8u", renderer)
        self.assertIn("std::vector<CachedViewport> viewport_cache", renderer)
        self.assertIn("output.visible_animation_count = frame.visible_animation_count", renderer)
        self.assertNotIn("frame.visible_animation_count == 0 &&", renderer)
        self.assertIn("cache_stale_rejections", renderer)
        self.assertIn("C3X_RENDERER_INVALIDATE_PACK_DEFINITION", renderer)
        self.assertIn("frame_invalidation_flags", api)
        self.assertIn("result.geometry = fnv_offset", signature)
        self.assertIn("reuse_geometry_for_translation", renderer)
        self.assertIn("current.anchor_x - cached.anchor_x != translation_x", renderer)
        self.assertIn("signature.geometry != geometry_cache.signature.geometry", renderer)
        self.assertIn("c3x_viewport_translation", (
            Path(__file__).parent / "integrated_terrain.hlsl"
        ).read_text(encoding="utf-8"))
        self.assertIn("VSSetConstantBuffers(1, 1, &viewport_settings_buffer)", renderer)

    def test_m6_7_feature_ownership_is_exact_and_post_composite(self) -> None:
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        feature_hook = injected[injected.index(
            "patch_Map_Renderer_m08_Draw_Tile_Forests_Jungle_Swamp"
        ) :]
        feature_hook = feature_hook[:feature_hook.index("patch_Map_Renderer_m52_Draw_Roads")]
        self.assertIn("custom_renderer_composited", feature_hook)
        self.assertIn("captured->tile_x == tile_x", feature_hook)
        self.assertIn("captured->tile_y == tile_y", feature_hook)
        self.assertIn("captured->anchor_x == pixel_x", feature_hook)
        self.assertIn("captured->anchor_y == pixel_y", feature_hook)
        self.assertIn("C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED", feature_hook)
        self.assertIn("replacement_tile_flags[index]", renderer)
        self.assertIn("feature_assets_ready", renderer)
        self.assertIn("tile.real_terrain_type == 7 || tile.real_terrain_type == 8", renderer)
        self.assertIn("tile.real_terrain_type == 9", renderer)
        self.assertIn("marsh_assets_ready", renderer)
        self.assertIn("tile.real_terrain_type == 10", renderer)
        self.assertIn("volcano_assets_ready", renderer)
        ownership = injected[injected.index("validate_custom_renderer_replacement_ownership") :]
        ownership = ownership[:ownership.index("composite_custom_renderer_frame")]
        self.assertIn("captured->real_terrain_type != SQ_Volcano", ownership)

    def test_m6_7_ports_approved_dunes_and_feature_bodies(self) -> None:
        runtime = (Path(__file__).parent / "terrain_scene_runtime.cpp").read_text(
            encoding="utf-8"
        )
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        for literal in ("0.300001f", "4.0f", "0.6f", "3.65f", "17.0f"):
            self.assertIn(literal, runtime)
        self.assertIn("c3x_renderer::dune_height", renderer)
        self.assertIn("dune_decal_base", renderer)
        self.assertIn("dune_decal_height", renderer)
        self.assertIn("vegetation_runtime.bin", renderer)
        self.assertIn("DXGI_FORMAT_BC1_UNORM", renderer)
        self.assertIn("find_feature_placement_by_suffix", renderer)
        self.assertIn("instance_count = tile.real_terrain_type == 7 ? 36u : 49u", renderer)
        self.assertIn("scene_feature_scale = tile.real_terrain_type == 7 ? 0.42f : 0.40f", renderer)
        self.assertIn("feature_vertices", renderer)
        self.assertIn("terrain_marsh_decal_01.json", renderer)
        for literal in ("0.48, 0.52", "1.42", "0.24, 0.70", "0.86", "1.91", "0.76, 0.28", "0.72", "1.37", "0.88"):
            self.assertIn(literal, renderer + shader)

    def test_production_uses_a_frozen_copy_of_the_approved_lab_shader(self) -> None:
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(
            encoding="utf-8"
        )
        adapter = (Path(__file__).parent / "integrated_terrain.hlsl").read_text(
            encoding="utf-8"
        )
        production_shader_path = Path(__file__).parent / "terrain_rendering.hlsl"
        production_shader = production_shader_path.read_text(encoding="utf-8")
        self.assertIn('#include "terrain_rendering.hlsl"', adapter)
        self.assertNotIn("terrain_lab", adapter)
        self.assertEqual(
            "e1216eb007348fee650c8583e4d90dcb800cdfce0850aa76e34a2fccd8ec5dec",
            hashlib.sha256(production_shader_path.read_bytes()).hexdigest(),
        )
        self.assertIn("return PSMain(input);", adapter)
        self.assertIn("return PSFeature(input);", adapter)
        self.assertIn('compile_terrain_shader("PSIntegrated"', renderer)
        self.assertIn('compile_terrain_shader("PSIntegratedFeature"', renderer)
        self.assertIn("D3DCompileFromFile", renderer)
        self.assertIn("std::array<ID3D11ShaderResourceView *, 128> views", renderer)
        self.assertIn("context->PSSetConstantBuffers(0, 1, &terrain_settings_buffer)", renderer)
        self.assertIn("#define C3X_GAME_RENDERER 1", adapter)
        self.assertIn("#define rivers_enabled 1.0", production_shader)
        self.assertIn("#define l13_layout 1.0", production_shader)
        self.assertIn("#define l13a_layout 1.0", production_shader)
        self.assertNotIn("l13_layout", renderer.casefold())
        self.assertIn("input.active_effect >= 0.0", production_shader)
        self.assertIn("output.active_effect = input.active_effect", adapter)
        self.assertIn("output.authored_relief = input.authored_relief", adapter)
        self.assertNotIn("surface_kind = 9.0", adapter)

    def test_i12_preview_accepts_the_exact_lab_biq_window_contract(self) -> None:
        preview = (Path(__file__).parent / "biq_preview.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("C3X_BIQ_TERRAIN_WINDOW_V1", preview)
        self.assertIn("C3X_BIQ_TERRAIN_WINDOW_V2", preview)
        self.assertIn("tile.x = source_x", preview)
        self.assertIn("tile.y = source_y", preview)
        self.assertIn("count + halo_count", preview)
        self.assertIn("tile.river_code = source.river", preview)
        self.assertIn("C3X_RENDERER_TILE_TOPOLOGY_HALO", preview)
        self.assertIn("argc == 12 ? std::atoi(argv[11]) : 12", preview)

    def test_i13_consumes_the_approved_river_contract_atomically(self) -> None:
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(
            encoding="utf-8"
        )
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        runtime = (Path(__file__).parent / "terrain_scene_runtime.cpp").read_text(
            encoding="utf-8"
        )
        definitions = (Path(__file__).parent / "terrain_definition_runtime.cpp").read_text(
            encoding="utf-8"
        )
        default_definition = (C3X_ROOT / "Renderer/default.custom_rendering.txt").read_text(
            encoding="utf-8"
        )
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")

        self.assertIn('companion("shore_normalized")', definitions)
        self.assertIn("id = shore_normalized", default_definition)
        self.assertIn("rivers = replace", default_definition)
        self.assertIn("features = replace", default_definition)
        self.assertIn("roads = replace", default_definition)
        self.assertIn("resources = replace", default_definition)
        self.assertIn("cities = replace", default_definition)
        for asset in (
            "river_base_color.dds", "river_height.dds", "river_specular.dds",
            "river_lean0.dds", "river_lean1.dds", "river_bank_noise",
            "shore_runtime.bin", 'find_feature_group(river_rock_bundle, "river_rock")',
        ):
            self.assertIn(asset, renderer)
        self.assertIn("tile.river_code", runtime)
        self.assertIn("tile.river_code & 170u", renderer)
        self.assertIn("add_river_edge(tile.tile_x, tile.tile_y - 1", renderer)
        self.assertIn("add_river_edge(tile.tile_x + 1, tile.tile_y", renderer)
        self.assertIn("canonical_river_x = canonical_component", renderer)
        self.assertIn("(distance - 4.0f) / 16.0f", renderer)
        self.assertIn("valley * 0.92f", renderer)
        self.assertIn("append_ground_layer(river_vertices, 9.0f", renderer)
        self.assertIn("views[79 + index] = river_surface_views[index]", renderer)
        self.assertIn("views[89 + index] = river_rock_texture_views[index]", renderer)
        self.assertIn("C3X_RENDERER_TILE_CUSTOM_RIVER_REPLACED", renderer + injected)
        self.assertIn("river_assets_ready && (tile.river_code & 170u)", renderer)
        self.assertIn("river_assets_ready = river_surface_ready && river_rocks_ready", renderer)
        reset = renderer[renderer.index("void reset()") : renderer.index("bool initialize()")]
        self.assertIn("river_surface_views", reset)
        self.assertIn("river_rock_texture_views", reset)
        self.assertIn("float distance_pixels = input.river_data.x", shader)
        self.assertIn("river_lean0_texture.Sample", shader)

    def test_i13a_consumes_the_shared_environment_and_shadow_contract(self) -> None:
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(
            encoding="utf-8"
        )
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        runtime = (Path(__file__).parent / "terrain_scene_runtime.cpp").read_text(
            encoding="utf-8"
        )
        for field in (
            "sun_direction", "sun_intensity", "sun_color", "shadow_strength",
            "moon_direction", "moon_intensity", "moon_color", "night_activation",
            "ambient_color", "environment_exposure", "water_fresnel",
            "water_specular", "emissive_scale", "hour",
        ):
            self.assertIn(field, renderer)
        self.assertIn("evaluate_environment(", renderer)
        self.assertIn("frame.hour", runtime)
        self.assertIn("frame.season", runtime)
        self.assertIn("append_ground_layer(shadow_vertices, 10.0f", renderer)
        self.assertIn("cast_shadow_visibility", renderer)
        self.assertIn("append_object_shadow", renderer)
        self.assertIn("#define l13a_layout 1.0", shader)
        self.assertIn("frame_illumination", shader)
        self.assertIn("frame_output_exposure", shader)
        self.assertIn("environment_water_fresnel", shader)
        self.assertIn("environment_water_specular", shader)
        self.assertIn("road_base_texture_0", shader)
        self.assertIn("railroad_base_texture", shader)
        self.assertIn("resource_base_texture_0", shader)
        self.assertIn("city_base_texture_0", shader)

    def test_i14_i17_consume_approved_routes_resources_and_cities(self) -> None:
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(
            encoding="utf-8"
        )
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(
            encoding="utf-8"
        )
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        default_definition = (C3X_ROOT / "Renderer/default.custom_rendering.txt").read_text(
            encoding="utf-8"
        )

        for runtime_pack in (
            "bridge_runtime.bin", "resource_runtime.bin", "city_runtime.bin",
            "wall_runtime.bin",
        ):
            self.assertIn(runtime_pack, renderer)
        for pack_id in (
            "route_styles_normalized", "route_doodads_normalized",
            "resource_normalized", "city_components_normalized",
            "city_adjuncts_normalized",
        ):
            self.assertIn(f"id = {pack_id}", default_definition)
        for register in (
            "road_base_texture_0 : register(t98)",
            "railroad_base_texture_1 : register(t107)",
            "road_bridge_base_texture_0 : register(t108)",
            "resource_base_texture_0 : register(t116)",
            "city_base_texture_3 : register(t127)",
        ):
            self.assertIn(register, shader)
        for implementation in (
            "append_route_segment", "bridge_", "resource_name.find(candidate)",
            "constexpr unsigned counts[] = {4u, 7u, 11u}",
            "C3X_RENDERER_CITY_WALLED", "city_emissive_views",
        ):
            self.assertIn(implementation, renderer)
        for ownership in (
            "C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED",
            "C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED",
            "C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED",
            "C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED",
        ):
            self.assertIn(ownership, api)
            self.assertIn(ownership, renderer)
            self.assertIn(ownership, injected)
        self.assertIn("record->route_style", injected)
        self.assertIn("record->resource_name", injected)
        self.assertIn("record->city_population", injected)
        self.assertIn("has_active_building (city, improvement_id)", injected)

    def test_i19_consumes_approved_mines_farms_and_tundra(self) -> None:
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(
            encoding="utf-8"
        ).casefold()
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        ).casefold()
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(
            encoding="utf-8"
        )
        default_definition = (C3X_ROOT / "Renderer/default.custom_rendering.txt").read_text(
            encoding="utf-8"
        ).casefold()
        self.assertIn("mine_runtime.bin", renderer)
        self.assertIn("mine_vertices", renderer)
        self.assertIn('"mine_" + std::to_string', renderer)
        self.assertIn("sample_reused_resource_slot", shader)
        self.assertIn("mine_emissive_code", shader)
        self.assertIn("C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED", api)
        self.assertIn("improvements = replace", default_definition)
        self.assertIn("id = improvements_normalized", default_definition)
        self.assertIn("farm_runtime.bin", renderer)
        self.assertIn("farm_vertices", renderer)
        self.assertIn('"farm_" + std::to_string', renderer)
        self.assertIn("C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED", api)
        self.assertIn("irrigation_mask", api)
        self.assertIn("material_tundra", renderer)
        self.assertIn("feature_base_texture_4", shader)

    def test_live_terrain_mesh_reuses_shared_corners_and_bounds_shadow_density(self) -> None:
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("std::vector<Vertex> grid_vertices", renderer)
        self.assertIn("ground_point_cache.reserve(2048)", renderer)
        self.assertIn("GroundPoint const & point = ground_point_at(u, v)", renderer)
        self.assertIn("shadow_field_signature", renderer)
        self.assertIn("reuse_shadow_field", renderer)
        self.assertIn("shadow_visibility_cache[shadow_visibility_index++]", renderer)
        self.assertIn("hash_shadow_value(frame.tile_width)", renderer)
        self.assertIn("hash_shadow_value(frame.tile_height)", renderer)
        self.assertNotIn("tile.anchor_x", renderer[
            renderer.index("current_shadow_field_signature"):
            renderer.index("c3x_renderer::EnvironmentState environment")
        ])
        self.assertIn("grid_v <= subdivisions", renderer)
        self.assertIn("grid_u <= subdivisions", renderer)
        self.assertIn("bool river_surface = layer > 8.5f", renderer)
        self.assertIn("river_surface ? river_node_distance", renderer)
        self.assertIn("frame.tile_count <= 512", renderer)
        self.assertIn("? 16 : 8", renderer)

    def test_static_terrain_cache_survives_partial_unit_redraw_traversals(self) -> None:
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(
            encoding="utf-8"
        )
        runtime = (Path(__file__).parent / "terrain_scene_runtime.cpp").read_text(
            encoding="utf-8"
        )
        camera_signature = runtime[
            runtime.index("result.camera = fnv_offset"):
            runtime.index("result.scene = fnv_offset")
        ]
        self.assertNotIn("frame.clip_left", camera_signature)
        self.assertNotIn("frame.clip_right", camera_signature)
        self.assertIn("reuse_cached_subset", renderer)
        self.assertIn("same_terrain_record", renderer)
        self.assertIn("cached_tiles", renderer)
        self.assertIn("cached_replacement_tile_flags", renderer)
        self.assertIn("D3D11_RECT scissor = {0, 0, width, height}", renderer)
        self.assertIn("fill_output(frame, output, 0, 0)", renderer)

        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        custom_draw = injected[
            injected.index("is->custom_renderer_draw_in_progress = true;"):
            injected.index("is->custom_renderer_frame_active = false;", injected.index("is->custom_renderer_draw_in_progress = true;"))
        ]
        self.assertIn(
            "Map_Renderer_m71_Draw_Tiles (this, __, param_1, param_2, 0);",
            custom_draw,
        )
        self.assertNotIn(
            "Map_Renderer_m71_Draw_Tiles (this, __, param_1, param_2, param_3);",
            custom_draw,
        )

    def test_feature_depth_uses_the_approved_lab_ground_plane_rule(self) -> None:
        renderer = (Path(__file__).parent / "c3x_renderer.cpp").read_text(
            encoding="utf-8"
        )
        self.assertGreaterEqual(renderer.count("float base_ground_y = center_y"), 2)
        self.assertGreaterEqual(renderer.count("float feature_height_tiles = local_z"), 2)
        self.assertGreaterEqual(
            renderer.count("feature_height_tiles * 0.0012f"), 2
        )
        self.assertNotIn(
            "1.0f - screen_y / static_cast<float>(frame.target_height) -\n"
            "                            local_z * 0.035f",
            renderer,
        )

    def test_bridge_never_falls_back_to_native_when_custom_rendering_is_on(self) -> None:
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        state = (C3X_ROOT / "C3X.h").read_text(encoding="utf-8")

        reentrant = re.search(
            r"if \(is->custom_renderer_draw_in_progress\) \{(.*?)\n\t\}",
            injected,
            re.DOTALL,
        )
        self.assertIsNotNone(reentrant)
        self.assertNotIn("Map_Renderer_m71_Draw_Tiles", reentrant.group(1))
        self.assertNotIn("custom_renderer_vanilla_base_restored", state)
        self.assertNotIn("restore_civ3_fallback_terrain", injected)
        enabled = injected[injected.index("if (! is->current_config.enable_custom_rendering)") :]
        unavailable = enabled[enabled.index("if (! ensure_custom_renderer_loaded ()") :]
        unavailable = unavailable[:unavailable.index("is->custom_renderer_draw_in_progress = true;")]
        self.assertNotIn("Map_Renderer_m71_Draw_Tiles", unavailable)
        self.assertIn('log_custom_renderer_event ("performance-counter"', unavailable)
        self.assertIn("render_result == C3X_RENDERER_RESULT_DEVICE_ERROR", injected)
        self.assertIn("is->custom_renderer_reset ()", injected)
        scenario_load = injected[injected.index("patch_load_scenario (") :]
        scenario_load = scenario_load[:scenario_load.index("record_unit_type_alt_strategy")]
        ignored_call = scenario_load.index("ADDR_LOAD_SCENARIO_RESUME_SAVE_2_RETURN")
        unload = scenario_load.index("unload_custom_renderer ();", ignored_call)
        config = scenario_load.index("reset_to_base_config ();", unload)
        self.assertLess(ignored_call, unload)
        self.assertLess(unload, config)

    def test_live_failure_diagnostics_emit_each_bridge_stage(self) -> None:
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        self.assertIn("(*p_OutputDebugStringA) (message)", injected)
        self.assertNotIn("custom_renderer.log", injected)
        for stage in (
            "definition-load", "dll-api", "dll-load", "capture-target", "target-image",
            "target-bounds", "render", "ownership-validation", "target-dc", "blit",
            "reentrant-map-draw", "performance-counter", "composite-boundary-not-reached",
        ):
            self.assertIn(f'"{stage}"', injected)

    def test_native_renderer_is_offscreen_and_source_independent(self) -> None:
        native = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8").casefold()
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(encoding="utf-8").casefold()
        self.assertIn("d3d11_bind_render_target", native)
        self.assertIn("d3d11_usage_staging", native)
        self.assertIn("copyresource", native)
        self.assertIn("runtime_width > 2048", native)
        for forbidden in ("createswapchain", "present(", "civ6", ".blp", ".fgx", "steamapps"):
            self.assertNotIn(forbidden, native + api)

    def test_native_renderer_loads_definition_driven_normalized_terrain(self) -> None:
        native = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        definitions = (Path(__file__).parent / "terrain_definition_runtime.cpp").read_text(encoding="utf-8")
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(encoding="utf-8")
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        build = (Path(__file__).parent / "BUILD.bat").read_text(encoding="utf-8")
        adapter = (Path(__file__).parent / "integrated_terrain.hlsl").read_text(
            encoding="utf-8"
        )
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        self.assertIn("c3x_renderer_set_pack_path", api)
        self.assertIn("c3x_renderer_set_definition_paths", api)
        self.assertIn("terrain/grassland/base", native)
        self.assertIn("c3x.normalized_mesh.v0", native)
        self.assertIn("c3x.material.v0", native)
        self.assertIn("DXGI_FORMAT_BC3_UNORM_SRGB", native)
        self.assertIn("CreateShaderResourceView", native)
        self.assertIn('#include "terrain_rendering.hlsl"', adapter)
        self.assertIn('read_file(integrated_shader_path.c_str(), shader_source)', native)
        self.assertIn('"Renderer\\\\native\\\\terrain_rendering.hlsl"', native)
        self.assertIn("mix_content_revision(shader_source)", native)
        self.assertIn("base_color_texture.Sample", shader)
        self.assertIn("1.0 / 2.2", shader)
        self.assertIn("load_terrain_definition_layers", native)
        if "load_approved_terrain_handoffs" in native:
            self.assertIn("terrain_scene_runtime.cpp", build)
        self.assertIn("scenario", definitions)
        self.assertIn("merged.erase(key)", definitions)
        self.assertIn("cached_textured_tile_count = textured_tile_count", native)
        self.assertIn("output.textured_tile_count = cached_textured_tile_count", native)
        self.assertIn("output.fallback_tile_indices =", native)

        for field in (
            "cache_hits",
            "cache_misses",
            "cache_evictions",
            "cache_stale_rejections",
            "frame_invalidation_flags",
            "device_generation",
            "device_recoveries",
        ):
            self.assertIn(f"output.{field}", injected)
        self.assertIn(r"Renderer\\default.custom_rendering.txt", injected)
        self.assertIn("custom_renderer_set_definition_paths", injected)
        self.assertNotIn("restore_civ3_fallback_terrain_for_custom_renderer_frame", injected)
        self.assertIn("output->fallback_tile_count != 0", injected)

    def test_connected_terrain_blends_without_tile_checkerboarding(self) -> None:
        native = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        adapter = (Path(__file__).parent / "integrated_terrain.hlsl").read_text(
            encoding="utf-8"
        )
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        self.assertIn("ground_by_coordinate", native)
        self.assertIn("neighbor_coordinates[4][2]", native)
        self.assertIn("base_ground_grid = frame.tile_width >= 96", native)
        self.assertIn("terrain_at_lattice", native)
        self.assertIn("center_material_weights", native)
        self.assertIn("material_weights_for", native)
        self.assertNotIn("contour_warp", native)
        self.assertIn("frame.world_wrap_x", native)
        self.assertIn("D3D11_FILTER_ANISOTROPIC", native)
        self.assertIn("sampler.MaxAnisotropy = 8", native)
        self.assertIn("float const uv_scale = 0.26f", native)
        self.assertNotIn("SampleBias", native)
        self.assertIn("output.material_weights = input.material_weights", adapter)
        self.assertIn("float3 material_normal", shader)
        self.assertIn("float3 light_direction", shader)
        self.assertIn("material_grass", native)
        self.assertIn("albedo = grass * weights.x + plains * weights.y", shader)
        self.assertIn("clip(input.shore_distance - 0.001)", shader)
        self.assertIn("frame_settings.light_direction", native)
        self.assertNotIn("0.88f + 0.04f", native)

    def test_m6_6_uses_real_terrain_relief_and_a_depth_target(self) -> None:
        native = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(encoding="utf-8")
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        self.assertIn("DXGI_FORMAT_R8_UNORM", native)
        self.assertIn("mountain_atlas", native)
        self.assertIn("surface_by_coordinate", native)
        self.assertIn("relief_sample", native)
        self.assertIn("hill_compatibility", native)
        self.assertIn("hill_support", native)
        self.assertIn("CreateDepthStencilView", native)
        self.assertIn("ClearDepthStencilView", native)
        self.assertIn("tile.real_terrain_type == 6", native)
        self.assertIn("static_cast<float>(frame.tile_width) / 224.0f * 0.82f", native)
        self.assertIn("feature_projection_scale =", native)
        self.assertIn("static_cast<float>(frame.tile_width) / 224.0f", native)
        self.assertNotIn("static_cast<float>(frame.tile_width) / 128.0f", native)
        self.assertIn("int x = lattice_u + lattice_v", native)
        self.assertIn("int y = lattice_u - lattice_v", native)
        self.assertIn("map_x = world_u + world_v - 1.0f", native)
        self.assertIn("map_y = world_u - world_v", native)
        self.assertIn("static_cast<float>(tile.tile_x - tile.tile_y) * 0.5f", native)
        self.assertNotIn("static_cast<float>(tile.tile_y - tile.tile_x) * 0.5f", native)
        self.assertIn("water_vertices", native)
        self.assertIn("material_height_view", native)
        self.assertIn("specular_view", native)
        self.assertIn("elevated_height_view", native)
        self.assertIn("elevated_specular_view", native)
        self.assertIn("mountain_base_texture : register(t6)", shader)
        self.assertIn("mountain_snow_texture : register(t8)", shader)
        self.assertIn("desert_mountain_base_texture : register(t57)", shader)
        self.assertIn("beach_base_texture : register(t11)", shader)
        self.assertIn("cliff_base_texture : register(t14)", shader)
        self.assertIn("mountain_height_texture.Sample", shader)
        self.assertIn("mountain_mask", shader)
        self.assertIn("relief_height_variants", native)
        self.assertIn("relief_blend_variants", native)
        self.assertIn('"textures\\\\relief\\\\hills\\\\standard\\\\height_lod0.dds"', native)
        self.assertIn("hills.height_scale_px = 52.0f", native)
        self.assertIn("mountains.height_scale_px = 104.0f", native)
        self.assertIn('default_filename == "default.custom_rendering.txt"', native)
        self.assertNotIn('default_name.find("Renderer\\\\default.custom_rendering.txt")', native)
        self.assertIn("smoothstep01((authored_macro - 0.22f) / 0.38f)", native)
        self.assertIn("measure_field_limits(mountains.relief_height_variants[variant]", native)
        self.assertIn("sample_normalized_field(", native)
        self.assertIn("float3 micro_normal", shader)
        self.assertIn("float3 geometry_normal", shader)
        self.assertIn("bool relief_neighborhood =", native)
        self.assertIn("frame.tile_count <= 512 ? 24", native)
        self.assertIn("frame.tile_count <= 768 ? 16 : 12", native)
        self.assertIn("frame.tile_count <= 2048 ? 12 : 8", native)
        self.assertIn("chunk_capacity = 262143u", native)
        self.assertNotIn("shore_vertices", native)
        self.assertNotIn("bool rugged_shore", native)
        self.assertIn("draw_streaming_batches(geometry_cache.water)", native)
        self.assertIn("D3D11_USAGE_DYNAMIC", native)
        self.assertIn("D3D11_MAP_WRITE_DISCARD", native)
        self.assertIn("(seed >> 3) % 5u", native)
        self.assertIn("has_relief_neighbor ? 0.50f : 0.68f", native)
        self.assertIn("constexpr int candidates[5][2]", native)
        self.assertIn("1.0f - (world_v - static_cast<float>(candidate_v))", native)
        self.assertIn("candidate_relief == 6 ? 104.0f", native)
        self.assertIn("mountain_displacement = chain_displacement", native)
        self.assertIn("canonical_feature_x * 0x193", native)
        self.assertIn("canonical_feature_y * 0x217", native)
        self.assertIn("instance * 103u + 59u", native)
        self.assertIn("instance * 107u + 61u", native)
        self.assertIn("0.07f + 0.86f * u_t", native)
        self.assertIn("0.07f + 0.86f * v_t", native)
        self.assertNotIn("mountains\\\\desert", native)
        self.assertNotIn("float radial =", native)
        self.assertNotIn("float secondary =", native)
        self.assertNotIn("float tertiary =", native)

    def test_blit_replaces_the_map_surface_without_frame_accumulation(self) -> None:
        native = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        adapter = (Path(__file__).parent / "integrated_terrain.hlsl").read_text(
            encoding="utf-8"
        )
        shader = (Path(__file__).parent / "terrain_rendering.hlsl").read_text(
            encoding="utf-8"
        )
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(encoding="utf-8")
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        blit = native[native.index('extern "C" __declspec(dllexport) int c3x_renderer_blit') :]
        blit = blit[:blit.index('extern "C" __declspec(dllexport) void c3x_renderer_reset')]
        self.assertIn("renderer.blit(*output", blit)
        self.assertIn("BitBlt", native)
        self.assertIn("SRCCOPY", native)
        self.assertIn("output.clip_right - output.clip_left", native)
        self.assertIn("blit_width != output.width", native)
        self.assertNotIn("AlphaBlend", blit)
        self.assertNotIn("AC_SRC_ALPHA", blit)
        self.assertIn("output.surface_kind = input.surface_kind", adapter)
        self.assertIn("canonical_component", native)
        self.assertIn("periodic_surface_uv", native)
        self.assertIn("canonical_feature_x", native)
        self.assertIn("signed_shore_distance", native)
        self.assertIn("water_family_depth", native)
        self.assertIn("append_ground_layer(underlay_vertices, 0.5f,", native)
        self.assertIn("append_ground_layer(land_vertices, 1.0f,", native)
        self.assertIn("append_ground_layer(bed_vertices, 4.0f,", native)

        self.assertIn("append_ground_layer(water_vertices, 5.0f,", native)
        self.assertIn("draw_streaming_batches(geometry_cache.underlay)", native)
        self.assertIn("draw_streaming_batches(geometry_cache.land)", native)
        self.assertIn("draw_streaming_batches(geometry_cache.bed)", native)
        self.assertIn("draw_streaming_batches(geometry_cache.water)", native)
        self.assertIn("beach_base_texture.Sample", shader)
        self.assertIn("float3 water_normal = normalize", shader)
        self.assertIn("float sun_glint =", shader)
        self.assertIn("water_large_lean0_texture : register(t20)", shader)
        self.assertIn("water_small_lean1_texture : register(t23)", shader)
        self.assertIn("water_foam_texture : register(t24)", shader)
        self.assertIn("DXGI_FORMAT_R16G16B16A16_UNORM", native)
        self.assertIn("c3x_renderer_i32 clip_left", api)
        self.assertIn("#define C3X_RENDERER_API_VERSION 10u", api)
        self.assertIn("DXGI_FORMAT_R16G16_UNORM", native)
        self.assertIn("float2 combined_lean =", shader)
        self.assertIn("water_foam_texture.Sample", shader)
        self.assertIn("frame.world_width_tiles", native)
        self.assertIn("world_wrap_x", api)
        self.assertIn("frame.world_wrap_x = (p_bic_data->Map.Flags & 1) != 0", injected)
        self.assertIn(
            "record->terrain_type = tile->vtable->m49_Get_Square_RealType (tile)",
            injected,
        )
        self.assertIn(
            "record->real_terrain_type = tile->vtable->m50_Get_Square_BaseType (tile)",
            injected,
        )
        self.assertNotIn(
            "record->terrain_type = tile->vtable->m50_Get_Square_BaseType (tile)",
            injected,
        )
        self.assertIn("record->real_terrain_type == SQ_Forest", injected)
        self.assertNotIn("record->terrain_type == SQ_Forest", injected)

    def test_approved_scene_failure_cannot_publish_candidate_dll(self) -> None:
        build = (Path(__file__).parent / "BUILD.bat").read_text(encoding="utf-8")
        smoke = build.index(
            'build\\native_smoke.exe "build\\candidate\\C3XRenderer.dll" '
            '--definitions ..\\.. ..\\..\\Renderer\\default.custom_rendering.txt'
        )
        gate = build.index("if errorlevel 1", smoke)
        publish = build.index(
            'copy /y "build\\candidate\\C3XRenderer.dll" "..\\bin\\C3XRenderer.dll"'
        )
        self.assertLess(smoke, gate)
        self.assertLess(gate, publish)

    def test_production_vertex_contract_preserves_exact_terrain_and_relief_inputs(self) -> None:
        native = (Path(__file__).parent / "c3x_renderer.cpp").read_text(encoding="utf-8")
        adapter = (Path(__file__).parent / "integrated_terrain.hlsl").read_text(
            encoding="utf-8"
        )
        self.assertIn("static_cast<float>(tile.terrain_type)", native)
        self.assertIn("static_cast<float>(tile.real_terrain_type)", native)
        self.assertIn("normal_x, normal_y, normal_z", native)
        self.assertIn("relief_sample[1], relief_sample[2], signed_shore", native)
        self.assertIn("float authored_height = 0.0f", native)
        self.assertIn("float authored_blend = 0.0f", native)
        self.assertIn("float3 geometry_normal : NORMAL0", adapter)
        self.assertIn("float2 authored_relief : TEXCOORD9", adapter)
        self.assertIn("float shore_distance : TEXCOORD10", adapter)
        self.assertNotIn("dot(input.base_slots", adapter)
        self.assertNotIn("dot(surface_slots", adapter)

    def test_live_scene_export_is_native_bounded_and_automatic(self) -> None:
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        exporter = (Path(__file__).parent / "scene_export.cpp").read_text(encoding="utf-8").casefold()
        self.assertIn("c3x_renderer_export_scene", injected)
        self.assertIn("custom_renderer_export_requested = true", injected)
        self.assertIn("VK_CONTROL", injected)
        self.assertIn("VK_SHIFT", injected)
        self.assertIn("VK_F12", injected)
        self.assertIn("civ3-live.scene.json", injected)
        self.assertIn("c3x.visible_scene.v0", exporter)
        self.assertIn("movefileexa", exporter)
        self.assertNotIn("civ6", exporter)

    def test_frame_scheduler_is_dirty_driven_and_timer_safe(self) -> None:
        injected = (C3X_ROOT / "injected_code.c").read_text(encoding="utf-8")
        scheduler = (Path(__file__).parent / "frame_scheduler.cpp").read_text(encoding="utf-8")
        api = (Path(__file__).parent / "c3x_renderer_api.h").read_text(encoding="utf-8")
        tick = injected[injected.index("custom_renderer_scheduler_tick ()"):]
        tick = tick[:tick.index("void __stdcall\npatch_on_timer_0x9F6500")]

        self.assertIn("QueryPerformanceCounter", tick)
        self.assertIn("custom_renderer_schedule (&input, &decision)", tick)
        self.assertIn("animator.field_18E4 + 10", tick)
        self.assertNotIn("custom_renderer_render", tick)
        self.assertNotIn("custom_renderer_blit", tick)
        self.assertNotIn("composite_custom_renderer_frame", tick)
        self.assertNotIn("m73_call_m22_Draw", tick)
        self.assertNotIn("Sleep", tick)

        self.assertIn("custom_renderer_scheduler_tick ();\n\ton_timer_0x9F6500 ();", injected)
        self.assertIn("custom_renderer_max_capture_ticks", injected)
        self.assertIn("custom_renderer_max_render_ticks", injected)
        self.assertIn("custom_renderer_max_blit_ticks", injected)
        self.assertIn("custom_renderer_max_map_pass_ticks", injected)
        self.assertIn("custom_renderer_redraw_pending = true", injected)
        self.assertIn("custom_renderer_requested_frames != 0xFFFFFFFFu", injected)
        self.assertIn("presentation_time_ticks", api)
        self.assertIn("visible_animation_count", api)
        self.assertIn("C3X_RENDERER_SCHEDULER_REDRAW_PENDING", api)
        self.assertIn("input->visible_animation_count == 0", scheduler)
        self.assertIn("input.now_ticks - input.event_start_ticks", scheduler)
        self.assertIn("output->request_redraw = 1", scheduler)
        for forbidden in ("Sleep", "while (", "CreateThread", "CreateSwapChain", "Present("):
            self.assertNotIn(forbidden, scheduler)


if __name__ == "__main__":
    unittest.main()
