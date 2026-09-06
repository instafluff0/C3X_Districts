#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>

#include "c3x_renderer_api.h"
#include "environment_runtime.h"

int fail(char const * message) {
    std::fprintf(stderr, "FAIL native_renderer_smoke: %s\n", message);
    return 1;
}

bool write_test_file(char const * path, void const * data, DWORD bytes) {
    HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return false;
    DWORD written = 0;
    bool ok = WriteFile(file, data, bytes, &written, nullptr) != 0 && written == bytes;
    CloseHandle(file);
    return ok;
}

void write_u32(std::vector<std::uint8_t> & data, std::size_t offset, std::uint32_t value) {
    data[offset] = static_cast<std::uint8_t>(value);
    data[offset + 1] = static_cast<std::uint8_t>(value >> 8);
    data[offset + 2] = static_cast<std::uint8_t>(value >> 16);
    data[offset + 3] = static_cast<std::uint8_t>(value >> 24);
}

std::vector<std::uint8_t> make_test_dds(std::uint16_t color) {
    std::vector<std::uint8_t> dds(164, 0);
    std::memcpy(dds.data(), "DDS ", 4);
    write_u32(dds, 4, 124);
    write_u32(dds, 12, 4);
    write_u32(dds, 16, 4);
    write_u32(dds, 28, 1);
    std::memcpy(dds.data() + 84, "DX10", 4);
    write_u32(dds, 128, 78);
    write_u32(dds, 132, 3);
    write_u32(dds, 140, 1);
    dds[148] = 255;
    dds[149] = 255;
    dds[156] = static_cast<std::uint8_t>(color);
    dds[157] = static_cast<std::uint8_t>(color >> 8);
    dds[158] = dds[156];
    dds[159] = dds[157];
    return dds;
}

bool make_synthetic_pack(char const * root) {
    CreateDirectoryA(root, nullptr);
    CreateDirectoryA("build\\synthetic_pack\\meshes", nullptr);
    CreateDirectoryA("build\\synthetic_pack\\materials", nullptr);
    CreateDirectoryA("build\\synthetic_pack\\textures", nullptr);
    char const manifest[] =
        "{\"schema\":\"c3x.asset_pack.v0\",\"assets\":{\"terrain/grassland/base\":{"
        "\"mesh\":\"meshes/flat_terrain_patch.json\",\"material\":\"materials/grassland.json\"},"
        "\"terrain/desert/base\":{\"mesh\":\"meshes/flat_terrain_patch.json\","
        "\"material\":\"materials/desert.json\"},\"terrain/plains/base\":{"
        "\"mesh\":\"meshes/flat_terrain_patch.json\",\"material\":\"materials/plains.json\"},"
        "\"terrain/coast/base\":{\"mesh\":\"meshes/flat_terrain_patch.json\","
        "\"material\":\"materials/coast.json\"},"
        "\"terrain/tundra/base\":{\"mesh\":\"meshes/flat_terrain_patch.json\","
        "\"material\":\"materials/missing.json\"}}}";
    char const mesh[] =
        "{\"schema\":\"c3x.normalized_mesh.v0\",\"topology\":{\"primitive\": \"triangles\","
        "\"indices\":[0,1,2,0,2,3]},\"vertices\":[]}";
    char const grassland_material[] =
        "{\"schema\":\"c3x.material.v0\",\"base_color\":{"
        "\"texture\":\"textures/grassland_base_color.dds\"}}";
    char const desert_material[] =
        "{\"schema\":\"c3x.material.v0\",\"base_color\":{"
        "\"texture\":\"textures/desert_base_color.dds\"}}";
    char const plains_material[] =
        "{\"schema\":\"c3x.material.v0\",\"base_color\":{"
        "\"texture\":\"textures/plains_base_color.dds\"}}";
    char const coast_material[] =
        "{\"schema\":\"c3x.material.v0\",\"base_color\":{"
        "\"texture\":\"textures/coast_base_color.dds\"}}";
    std::vector<std::uint8_t> grassland_dds = make_test_dds(0x07e0);
    std::vector<std::uint8_t> desert_dds = make_test_dds(0xfd20);
    std::vector<std::uint8_t> plains_dds = make_test_dds(0xafe5);
    std::vector<std::uint8_t> coast_dds = make_test_dds(0x4e99);
    return write_test_file("build\\synthetic_pack\\manifest.json", manifest, sizeof(manifest) - 1) &&
           write_test_file("build\\synthetic_pack\\meshes\\flat_terrain_patch.json", mesh, sizeof(mesh) - 1) &&
           write_test_file("build\\synthetic_pack\\materials\\grassland.json", grassland_material, sizeof(grassland_material) - 1) &&
           write_test_file("build\\synthetic_pack\\materials\\desert.json", desert_material, sizeof(desert_material) - 1) &&
           write_test_file("build\\synthetic_pack\\materials\\plains.json", plains_material, sizeof(plains_material) - 1) &&
           write_test_file("build\\synthetic_pack\\materials\\coast.json", coast_material, sizeof(coast_material) - 1) &&
           write_test_file("build\\synthetic_pack\\textures\\grassland_base_color.dds", grassland_dds.data(), static_cast<DWORD>(grassland_dds.size())) &&
           write_test_file("build\\synthetic_pack\\textures\\desert_base_color.dds", desert_dds.data(), static_cast<DWORD>(desert_dds.size())) &&
           write_test_file("build\\synthetic_pack\\textures\\plains_base_color.dds", plains_dds.data(), static_cast<DWORD>(plains_dds.size())) &&
           write_test_file("build\\synthetic_pack\\textures\\coast_base_color.dds", coast_dds.data(), static_cast<DWORD>(coast_dds.size()));
}

bool make_synthetic_definitions() {
    char const defaults[] =
        "#Pack\nid = terrain\npath = mod:build\\synthetic_pack\n"
        "#Asset\nid = grass\npack = terrain\nasset = terrain/grassland/base\n"
        "#Asset\nid = desert\npack = terrain\nasset = terrain/desert/base\n"
        "#Asset\nid = tundra\npack = terrain\nasset = terrain/tundra/base\n"
        "#Asset\nid = coast\npack = terrain\nasset = terrain/coast/base\n"
        "#Rule\nid = terrain.grass\ncategory = terrain\nterrain_type = grassland\nasset = grass\nreplacement = replace\n"
        "#Rule\nid = terrain.desert\ncategory = terrain\nterrain_type = desert\nasset = desert\nreplacement = replace\n"
        "#Rule\nid = terrain.tundra\ncategory = terrain\nterrain_type = tundra\nasset = tundra\nreplacement = replace\n"
        "#Rule\nid = terrain.coast\ncategory = terrain\nterrain_type = coast\nasset = coast\nreplacement = replace\n";
    char const scenario[] =
        "#Pack\nid = terrain\npath = mod:build\\synthetic_pack\n"
        "#Asset\nid = plains\npack = terrain\nasset = terrain/plains/base\n"
        "#Rule\nid = terrain.plains\ncategory = terrain\nterrain_type = plains\nasset = plains\nreplacement = replace\n";
    char const custom[] =
        "#Rule\nid = terrain.desert\ncategory = terrain\nterrain_type = desert\nasset = grass\npriority = 200\nreplacement = replace\n";
    return write_test_file("build\\synthetic_default.txt", defaults, sizeof(defaults) - 1) &&
           write_test_file("build\\synthetic_scenario.txt", scenario, sizeof(scenario) - 1) &&
           write_test_file("build\\synthetic_custom.txt", custom, sizeof(custom) - 1);
}

std::uint64_t hash_pixels(void const * data, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ull;
    auto const * bytes = static_cast<std::uint8_t const *>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

c3x_renderer_tile_v1 make_tile(int x, int y, int anchor_x, int anchor_y, int terrain, c3x_renderer_u32 seed) {
    c3x_renderer_tile_v1 tile = {};
    tile.tile_x = x;
    tile.tile_y = y;
    tile.anchor_x = anchor_x;
    tile.anchor_y = anchor_y;
    tile.terrain_type = terrain;
    tile.real_terrain_type = terrain;
    tile.visibility_mask = 1;
    tile.tile_visibility = 1;
    tile.variant_seed = seed;
    tile.tile_flags = C3X_RENDERER_TILE_RENDER;
    tile.resource_id = -1;
    tile.resource_class = -1;
    tile.tile_building_id = -1;
    tile.city_id = -1;
    tile.city_owner_id = -1;
    tile.city_size = -1;
    tile.city_culture_group = -1;
    tile.city_era = -1;
    tile.unit_type_id = -1;
    tile.unit_owner_id = -1;
    tile.unit_class = -1;
    tile.unit_state = -1;
    tile.unit_damage = -1;
    tile.unit_direction = -1;
    tile.territory_owner_id = -1;
    tile.fog_status = 0;
    return tile;
}

int main(int argc, char ** argv) {
    SetEnvironmentVariableA("C3X_RENDERER_TRACE", "2");
    SetEnvironmentVariableA("C3X_RENDERER_TRACE_FILE", "build\\cache-trace.log");
    char const * path = argc > 1 ? argv[1] : "..\\bin\\C3XRenderer.dll";
    char const * pack_path = argc > 2 ? argv[2] : "build\\synthetic_pack";
    bool synthetic_definition_mode = argc <= 2;
    bool external_definition_mode = argc >= 5 && std::strcmp(argv[2], "--definitions") == 0;
    if (synthetic_definition_mode && (!make_synthetic_pack(pack_path) || !make_synthetic_definitions()))
        return fail("could not create synthetic normalized pack and definition layers");
    HMODULE module = LoadLibraryA(path);
    if (module == nullptr)
        return fail("could not load 32-bit C3XRenderer.dll");

    auto get_version = reinterpret_cast<c3x_renderer_get_api_version_fn>(GetProcAddress(module, "c3x_renderer_get_api_version"));
    auto set_pack_path = reinterpret_cast<c3x_renderer_set_pack_path_fn>(GetProcAddress(module, "c3x_renderer_set_pack_path"));
    auto set_definition_paths = reinterpret_cast<c3x_renderer_set_definition_paths_fn>(GetProcAddress(module, "c3x_renderer_set_definition_paths"));
    auto render = reinterpret_cast<c3x_renderer_render_fn>(GetProcAddress(module, "c3x_renderer_render"));
    auto blit = reinterpret_cast<c3x_renderer_blit_fn>(GetProcAddress(module, "c3x_renderer_blit"));
    auto export_scene = reinterpret_cast<c3x_renderer_export_scene_fn>(GetProcAddress(module, "c3x_renderer_export_scene"));
    auto schedule = reinterpret_cast<c3x_renderer_schedule_fn>(GetProcAddress(module, "c3x_renderer_schedule"));
    auto reset = reinterpret_cast<c3x_renderer_reset_fn>(GetProcAddress(module, "c3x_renderer_reset"));
    if (get_version == nullptr || set_pack_path == nullptr || set_definition_paths == nullptr || render == nullptr || blit == nullptr || export_scene == nullptr ||
        schedule == nullptr || reset == nullptr)
        return fail("required undecorated C ABI exports are missing");
    if (get_version() != C3X_RENDERER_API_VERSION)
        return fail("API version mismatch");

    c3x_renderer::EnvironmentState noon = c3x_renderer::evaluate_environment(12.0f, 0);
    c3x_renderer::EnvironmentState sunset = c3x_renderer::evaluate_environment(18.5f, 0);
    c3x_renderer::EnvironmentState midnight = c3x_renderer::evaluate_environment(0.0f, 0);
    c3x_renderer::EnvironmentState sunrise = c3x_renderer::evaluate_environment(6.0f, 0);
    if (noon.sun_intensity <= sunset.sun_intensity || noon.sun_intensity <= sunrise.sun_intensity ||
        midnight.sun_intensity != 0.0f || midnight.moon_intensity <= noon.moon_intensity ||
        midnight.night_activation < 0.99f || noon.night_activation > 0.01f ||
        !std::isfinite(sunset.exposure) || !std::isfinite(sunrise.water_fresnel))
        return fail("continuous noon/sunset/midnight/sunrise environment profile is invalid");
    float water_a[3] = {}, water_b[3] = {}, land[3] = {};
    c3x_renderer::shade_terrain(midnight, 11, 10, 20, 1.0f, water_a);
    c3x_renderer::shade_terrain(midnight, 11, 13, 21, 1.0f, water_b);
    c3x_renderer::shade_terrain(midnight, 2, 10, 20, 1.0f, land);
    float water_difference = std::abs(water_a[0] - water_b[0]) + std::abs(water_a[1] - water_b[1]) +
        std::abs(water_a[2] - water_b[2]);
    if (water_difference <= 0.0001f || water_difference > 0.45f || water_a[2] <= land[2])
        return fail("directional moonlit-water response is absent or unbounded");

    c3x_renderer::EmissiveChannel window_emissive = {};
    window_emissive.color[0] = 1.0f;
    window_emissive.color[1] = 0.42f;
    window_emissive.color[2] = 0.12f;
    window_emissive.intensity = 2.0f;
    window_emissive.activation_policy = c3x_renderer::ActivationPolicy::night;
    if (c3x_renderer::evaluate_emissive(window_emissive, midnight, 0.0f) <= 2.0f ||
        c3x_renderer::evaluate_emissive(window_emissive, noon, 12.0f) != 0.0f)
        return fail("generic emissive channel did not activate only at night");
    c3x_renderer::AnalyticLight fire_light = {};
    fire_light.stable_id = 0x5511u;
    fire_light.type = c3x_renderer::AnalyticLightType::point;
    fire_light.bounds.radius = 2.25f;
    fire_light.intensity = 3.0f;
    fire_light.activation_policy = c3x_renderer::ActivationPolicy::night;
    fire_light.required_state_mask = 4u;
    if (c3x_renderer::evaluate_analytic_light(fire_light, midnight, 0.0f, 4u) <= 0.0f ||
        c3x_renderer::evaluate_analytic_light(fire_light, midnight, 0.0f, 2u) != 0.0f)
        return fail("generic analytic light did not honor activation and state requirements");

    c3x_renderer::AmbientAttachment flame = {};
    flame.stable_id = 0x43a8u;
    flame.analytic_light_id = fire_light.stable_id;
    flame.bounds.radius = 2.0f;
    flame.activation_policy = c3x_renderer::ActivationPolicy::night;
    flame.required_state_mask = 4u;
    flame.stable_phase_seed = 1777u;
    flame.period_ticks = 1000000;
    flame.animated = true;
    c3x_renderer::AttachmentInput attachment_input = {2468000, 4u, true, true, true};
    c3x_renderer::AttachmentState attached = c3x_renderer::evaluate_attachment(
        flame, attachment_input, midnight, 0.0f);
    c3x_renderer::AttachmentState repeated_attachment = c3x_renderer::evaluate_attachment(
        flame, attachment_input, midnight, 0.0f);
    if (!attached.active || attached.visible_animation_count != 1 ||
        attached.phase_millionths != repeated_attachment.phase_millionths)
        return fail("attached flame/light phase is not deterministic from absolute time and stable seed");
    attachment_input.visible = false;
    if (c3x_renderer::evaluate_attachment(flame, attachment_input, midnight, 0.0f).visible_animation_count != 0)
        return fail("hidden ambient attachment requested animation frames");
    attachment_input.visible = true;
    attachment_input.resources_available = false;
    c3x_renderer::AttachmentState missing_attachment = c3x_renderer::evaluate_attachment(
        flame, attachment_input, midnight, 0.0f);
    if (!missing_attachment.fallback || missing_attachment.active || missing_attachment.visible_animation_count != 0)
        return fail("missing attachment resources did not degrade to idle owner fallback");
    attachment_input.resources_available = true;
    flame.animated = false;
    c3x_renderer::AttachmentState static_emissive = c3x_renderer::evaluate_attachment(
        flame, attachment_input, midnight, 0.0f);
    if (!static_emissive.active || static_emissive.visible_animation_count != 0)
        return fail("static night emissive requested continuous redraw");

    if (set_pack_path("build\\missing_pack") != C3X_RENDERER_RESULT_ERROR)
        return fail("missing normalized pack was not rejected");
    if (synthetic_definition_mode) {
        if (set_definition_paths(".", "build\\synthetic_default.txt", "build\\synthetic_scenario.txt", nullptr) != C3X_RENDERER_RESULT_OK)
            return fail("layered terrain definitions were not accepted");
    } else if (external_definition_mode) {
        if (set_definition_paths(argv[3], argv[4], nullptr, nullptr) != C3X_RENDERER_RESULT_OK) {
            std::fprintf(stderr, "external definition diagnostic=%lu\n", GetLastError());
            return fail("external terrain definitions were not accepted");
        }
    } else if (set_pack_path(pack_path) != C3X_RENDERER_RESULT_OK) {
        return fail("normalized grassland pack was not accepted");
    }

    c3x_renderer_tile_v1 tiles[] = {
        make_tile(10, 20, 160, 32, 2, 1001),
        make_tile(11, 20, 224, 64, 0, 1002),
        make_tile(10, 21, 96, 64, 3, 1003),
        make_tile(11, 21, 160, 96, 1, 1004),
        make_tile(12, 21, 288, 96, 11, 1005)
    };
    tiles[0].feature_flags = C3X_RENDERER_FEATURE_FOREST;
    tiles[0].road_mask = 1;
    tiles[0].railroad_mask = 1;
    tiles[0].river_code = 3;
    tiles[0].improvement_flags = C3X_RENDERER_IMPROVEMENT_IRRIGATION | C3X_RENDERER_IMPROVEMENT_TILE_BUILDING;
    tiles[0].tile_building_id = 7;
    tiles[0].resource_id = 5;
    tiles[0].resource_class = 2;
    strcpy_s(tiles[0].resource_name, "Horses");
    tiles[0].city_id = 12;
    tiles[0].city_owner_id = 1;
    tiles[0].city_population = 5;
    tiles[0].city_size = 0;
    tiles[0].city_culture_group = 2;
    tiles[0].city_era = 0;
    tiles[0].city_flags = C3X_RENDERER_CITY_CAPITAL;
    strcpy_s(tiles[0].city_owner, "Romans");
    strcpy_s(tiles[0].city_civilization, "Roman");
    strcpy_s(tiles[0].city_era_name, "Ancient Times");
    tiles[0].unit_type_id = 2;
    tiles[0].unit_owner_id = 1;
    tiles[0].unit_class = 0;
    tiles[0].unit_state = 1;
    tiles[0].unit_damage = 0;
    tiles[0].unit_direction = 3;
    strcpy_s(tiles[0].unit_owner, "Romans");
    strcpy_s(tiles[0].unit_civilization, "Roman");
    strcpy_s(tiles[0].unit_era_name, "Ancient Times");
    strcpy_s(tiles[0].unit_type_name, "Warrior");
    tiles[0].has_effect = 1;
    c3x_renderer_frame_v1 frame = {};
    frame.api_version = C3X_RENDERER_API_VERSION;
    frame.struct_size = sizeof(frame);
    frame.target_width = 320;
    frame.target_height = 200;
    frame.clip_right = 320;
    frame.clip_bottom = 200;
    frame.tile_width = 128;
    frame.tile_height = 64;
    frame.hour = 12;
    frame.tile_count = static_cast<c3x_renderer_u32>(std::size(tiles));
    frame.tiles = tiles;
    frame.presentation_time_ticks = 1000000;
    frame.presentation_frequency = 1000000;
    frame.dirty_flags = C3X_RENDERER_DIRTY_ALL;
    frame.visible_animation_count = 1;
    c3x_renderer_output_v1 output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
        return fail("off-screen render failed");
    c3x_renderer_u32 expected_textured = synthetic_definition_mode ? 4u : (external_definition_mode ? 5u : 1u);
    if (output.width != 320 || output.height != 200 ||
        output.clip_left != 0 || output.clip_top != 0 ||
        output.clip_right != 320 || output.clip_bottom != 200 ||
        output.rendered_tile_count != expected_textured ||
        output.fallback_tile_count != 0 || output.bgra_pixels == nullptr ||
        output.visible_animation_count != 1 || output.request_continuous_redraw != 1 ||
        output.renderer_cpu_ticks < 0 || output.textured_tile_count != expected_textured)
        return fail("off-screen output metadata is invalid");
    if (output.replacement_tile_count != static_cast<c3x_renderer_u32>(std::size(tiles)) ||
        output.replacement_tile_flags == nullptr || output.cache_capacity != 32 ||
        output.cache_entries != 1 || output.content_revision == 0)
        return fail("replacement ownership or bounded cache metadata is invalid");
    for (std::size_t index = 0; index < std::size(tiles); ++index) {
        bool expected_replacement = external_definition_mode ||
            (index != 2 && (synthetic_definition_mode || index == 0));
        bool is_replacement =
            (output.replacement_tile_flags[index] & C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED) != 0;
        if (is_replacement != expected_replacement)
            return fail("per-tile terrain ownership did not match successful rendering");
    }
    if (external_definition_mode &&
        (output.replacement_tile_flags[1] & C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED) == 0)
        return fail("approved L10 flat-desert tile did not acquire dune ownership");

    std::size_t byte_count = static_cast<std::size_t>(output.stride_bytes) * output.height;
    std::vector<std::uint8_t> first(byte_count);
    std::memcpy(first.data(), output.bgra_pixels, byte_count);
    std::size_t opaque = 0;
    for (std::size_t index = 3; index < byte_count; index += 4)
        opaque += first[index] != 0;
    if (opaque < 1000 || first[3] != 0)
        return fail("render is blank or overwrote transparent bounds");
    std::uint64_t first_hash = hash_pixels(first.data(), first.size());

    if (synthetic_definition_mode) {
        c3x_renderer_i64 base_revision = output.content_revision;
        if (set_definition_paths(".", "build\\synthetic_default.txt", "build\\synthetic_scenario.txt", "build\\synthetic_custom.txt") != C3X_RENDERER_RESULT_OK)
            return fail("custom terrain-definition layer was not accepted");
        output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
            return fail("custom-layer render failed");
        std::uint64_t layered_hash = hash_pixels(output.bgra_pixels, byte_count);
        if (layered_hash == first_hash || output.textured_tile_count != 4 || output.fallback_tile_count != 0 ||
            output.content_revision == base_revision ||
            (output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_PACK_DEFINITION) == 0)
            return fail("custom definition did not atomically override one terrain rule");
        std::memcpy(first.data(), output.bgra_pixels, byte_count);
        first_hash = layered_hash;
    }

    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        hash_pixels(output.bgra_pixels, byte_count) != first_hash)
        return fail("same-frame render is not deterministic");

    // Static-map reuse must be decided by captured content, never by dirty hints.
    // Exercise an exact hit plus the scroll/clip/wrap/environment dimensions that
    // would otherwise leave stale strips in Civ III's retained map surface.
    frame.visible_animation_count = 0;
    frame.world_width_tiles = 100;
    frame.world_height_tiles = 100;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
        return fail("static cache priming render failed");
    // Subsequent cache tests compare the exact primed world snapshot. A
    // separate reset/translation witness below checks independently rasterized
    // views with its explicit edge tolerance.
    std::memcpy(first.data(), output.bgra_pixels, byte_count);
    first_hash = hash_pixels(first.data(), first.size());
    if (output.cache_entries != 2 || output.cache_capacity != 32)
        return fail("static terrain cache is not populated within its fixed bound");
    c3x_renderer_u32 cache_hits_before = output.cache_hits;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        output.cache_hits != cache_hits_before + 1 || output.renderer_cpu_ticks != 0 ||
        output.frame_invalidation_flags != 0)
        return fail("identical static frame did not produce an exact cache hit");

    if (!external_definition_mode) {
        // Keep enough exact viewports for normal unit-cycling camera jumps while
        // retaining a hard bound suitable for Civ III's 32-bit process. The
        // portable 320x200 fixture fills and evicts the cache without extending
        // the licensed full-scene VM replay past the remote-command time bound.
        for (int offset = 1; offset <= 30; ++offset) {
            frame.world_width_tiles = 100 + offset;
            output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
                return fail("multi-viewport cache priming render failed");
        }
        frame.world_width_tiles = 100;
        c3x_renderer_u32 multi_view_hits = output.cache_hits;
        output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
            output.cache_hits != multi_view_hits + 1 || output.renderer_cpu_ticks != 0 ||
            output.cache_entries != 32 || output.cache_capacity != 32)
            return fail("recent unit-jump viewport was not retained in the bounded LRU");
        c3x_renderer_u32 evictions_before = output.cache_evictions;
        for (int offset = 31; offset <= 33; ++offset) {
            frame.world_width_tiles = 100 + offset;
            output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
                return fail("bounded viewport eviction render failed");
        }
        if (output.cache_entries != 32 || output.cache_evictions <= evictions_before)
            return fail("multi-viewport cache exceeded its bound or failed to evict LRU views");
        frame.world_width_tiles = 100;
        output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
            return fail("could not restore static viewport after bounded LRU exercise");
    }

    // Units and their UI selectors remain in Civ III's retained overlay planes.
    // Visible animation must keep driving redraw without evicting static map art.
    c3x_renderer_u32 retained_hits_before = output.cache_hits;
    std::uint64_t retained_hash_before = hash_pixels(output.bgra_pixels, byte_count);
    tiles[0].unit_direction = (tiles[0].unit_direction + 1) & 7;
    tiles[0].square_parts ^= 0x20u;
    tiles[0].terrain_overlays ^= 0x40u;
    tiles[0].visibility_mask ^= 0x80u;
    tiles[0].city_population += 1;
    frame.visible_animation_count = 1;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        output.cache_hits != retained_hits_before + 1 || output.renderer_cpu_ticks != 0 ||
        output.frame_invalidation_flags != 0 ||
        output.geometry_tiles_built != 0 || output.geometry_upload_bytes != 0 ||
        output.raster_draw_pixels != 0 ||
        output.request_continuous_redraw != 1 ||
        hash_pixels(output.bgra_pixels, byte_count) != retained_hash_before)
    {
        FILE * diagnostic = nullptr;
        fopen_s(&diagnostic, "build\\cache-first.bgra", "wb");
        if (diagnostic) { std::fwrite(first.data(), 1, byte_count, diagnostic); std::fclose(diagnostic); }
        fopen_s(&diagnostic, "build\\cache-retained.bgra", "wb");
        if (diagnostic) { std::fwrite(output.bgra_pixels, 1, byte_count, diagnostic); std::fclose(diagnostic); }
        std::fprintf(stderr, "retained hits=%u expected=%u ticks=%lld invalidations=%u animation=%u hash=%llu expected_hash=%llu\n",
            output.cache_hits, retained_hits_before + 1, output.renderer_cpu_ticks,
            output.frame_invalidation_flags, output.request_continuous_redraw,
            hash_pixels(output.bgra_pixels, byte_count), retained_hash_before);
        return fail("animated Civ III overlay state invalidated static map cache work");
    }
    tiles[0].unit_direction = (tiles[0].unit_direction + 7) & 7;
    tiles[0].square_parts ^= 0x20u;
    tiles[0].terrain_overlays ^= 0x40u;
    tiles[0].visibility_mask ^= 0x80u;
    tiles[0].city_population -= 1;
    frame.visible_animation_count = 0;

    // Renderer-owned routes, resources, and cities are authoritative static
    // scene inputs and must invalidate when their captured state changes.
    tiles[0].road_mask ^= 4u;
    tiles[0].resource_id += 1;
    tiles[0].city_size += 1;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        (output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0 ||
        output.renderer_cpu_ticks == 0)
        return fail("renderer-owned route/resource/city change reused stale terrain");
    tiles[0].road_mask ^= 4u;
    tiles[0].resource_id -= 1;
    tiles[0].city_size -= 1;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        hash_pixels(output.bgra_pixels, byte_count) != first_hash)
        return fail("renderer-owned route/resource/city restore was not deterministic");

    // Rivers are renderer-owned geometry after I13, so authoritative mask
    // changes must invalidate the retained terrain result.
    tiles[0].river_code ^= 8u;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        (output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0 ||
        output.renderer_cpu_ticks == 0)
        return fail("river topology change reused stale terrain");
    tiles[0].river_code ^= 8u;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
        return fail("river topology restore failed");

    tiles[0].anchor_x += 1;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        (output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0)
        return fail("pixel-scroll occurrence change reused stale terrain");
    tiles[0].anchor_x -= 1;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        hash_pixels(output.bgra_pixels, byte_count) != first_hash)
        return fail("pixel-scroll occurrence restore did not recover the retained terrain");

    c3x_renderer_u32 clip_hits_before = output.cache_hits;
    frame.clip_left = 1;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        output.clip_left != 1 || output.cache_hits != clip_hits_before + 1 ||
        output.renderer_cpu_ticks != 0 || output.frame_invalidation_flags != 0 ||
        hash_pixels(output.bgra_pixels, byte_count) != first_hash)
        return fail("partial-clip redraw did not reuse the complete retained terrain surface");
    frame.clip_left = 0;

    c3x_renderer_frame_v1 subset_frame = frame;
    subset_frame.clip_left = 37;
    subset_frame.clip_top = 19;
    subset_frame.clip_right = 249;
    subset_frame.clip_bottom = 151;
    subset_frame.tile_count = 2;
    subset_frame.tiles = tiles + 1;
    c3x_renderer_u32 subset_hits_before = output.cache_hits;
    int prior_subset_unit_direction = tiles[1].unit_direction;
    tiles[1].unit_direction = (tiles[1].unit_direction + 1) & 7;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&subset_frame, &output) != C3X_RENDERER_RESULT_OK ||
        output.cache_hits != subset_hits_before + 1 || output.renderer_cpu_ticks != 0 ||
        output.frame_invalidation_flags != 0 || output.replacement_tile_count != 2 ||
        output.clip_left != 37 || output.clip_top != 19 ||
        output.clip_right != 249 || output.clip_bottom != 151 ||
        hash_pixels(output.bgra_pixels, byte_count) != first_hash)
        return fail("unit-only partial traversal did not reuse matching static terrain records");
    tiles[1].unit_direction = prior_subset_unit_direction;

    frame.world_wrap_x = 1;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        (output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_WRAP) == 0)
        return fail("wrap-state change reused stale terrain");
    frame.world_wrap_x = 0;

    frame.hour = 13;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        (output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_ENVIRONMENT) == 0)
        return fail("environment change reused stale terrain");
    frame.hour = 12;

    // Restore the baseline and prove the invalidation sequence converges to the
    // same complete viewport rather than retaining a partial-clip artifact.
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK ||
        hash_pixels(output.bgra_pixels, byte_count) != first_hash ||
        output.cache_stale_rejections < 4)
        return fail("cache invalidation sequence did not restore the baseline viewport");

    int environment_hours[] = {12, 18, 0, 6};
    std::uint64_t small_environment_hashes[4] = {};
    for (std::size_t index = 0; index < std::size(environment_hours); ++index) {
        frame.hour = environment_hours[index];
        output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
            return fail("small environment fixture render failed");
        small_environment_hashes[index] = hash_pixels(output.bgra_pixels, byte_count);
    }
    if (small_environment_hashes[0] == small_environment_hashes[1] ||
        small_environment_hashes[0] == small_environment_hashes[2] ||
        small_environment_hashes[2] == small_environment_hashes[3])
        return fail("small noon/sunset/midnight/sunrise fixtures are not visually distinct");
    frame.hour = 12;
    output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
    if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
        return fail("could not restore noon fixture after environment matrix");

    if (external_definition_mode) {
        c3x_renderer_tile_v1 approved_tiles[] = {
            make_tile(0, 0, 16, 32, 0, 0x1001u),
            make_tile(1, 1, 80, 64, 2, 0x1002u),
            make_tile(2, 2, 144, 32, 1, 0x1003u),
            make_tile(3, 3, 208, 64, 2, 0x1004u),
            make_tile(4, 4, 272, 32, 0, 0x1005u),
            make_tile(5, 5, 336, 64, 2, 0x1006u),
            make_tile(6, 6, 400, 32, 1, 0x1007u),
            make_tile(7, 7, 464, 64, 11, 0x1008u),
            make_tile(8, 8, 528, 32, 12, 0x1009u),
            make_tile(9, 9, 592, 64, 13, 0x100au),
            make_tile(10, 10, 656, 32, 3, 0x100bu),
            make_tile(11, 11, 720, 64, 2, 0x100cu),
            make_tile(12, 12, 784, 32, 0, 0x100du),
            make_tile(100, 0, 848, 64, 2, 0x100eu)
        };
        approved_tiles[3].real_terrain_type = 5;
        approved_tiles[4].real_terrain_type = 6;
        approved_tiles[5].real_terrain_type = 7;
        approved_tiles[5].feature_flags = C3X_RENDERER_FEATURE_FOREST;
        approved_tiles[6].real_terrain_type = 8;
        approved_tiles[6].feature_flags = C3X_RENDERER_FEATURE_JUNGLE;
        approved_tiles[11].real_terrain_type = 9;
        approved_tiles[11].feature_flags = C3X_RENDERER_FEATURE_MARSH;
        approved_tiles[10].real_terrain_type = 10;
        approved_tiles[10].feature_flags = C3X_RENDERER_FEATURE_VOLCANO;
        approved_tiles[1].river_code = 2u;
        approved_tiles[2].river_code = 8u;
        approved_tiles[1].road_mask = 1u;
        approved_tiles[1].route_style = 2;
        approved_tiles[2].road_mask = 1u;
        approved_tiles[2].route_style = 2;
        approved_tiles[2].improvement_flags |= C3X_RENDERER_IMPROVEMENT_IRRIGATION;
        approved_tiles[2].irrigation_mask = 15u;
        approved_tiles[3].railroad_mask = 1u;
        approved_tiles[4].railroad_mask = 1u;
        approved_tiles[5].resource_id = 4;
        strcpy_s(approved_tiles[5].resource_name, "Horses");
        approved_tiles[6].improvement_flags |= C3X_RENDERER_IMPROVEMENT_MINE;
        approved_tiles[6].route_style = 3;
        approved_tiles[3].city_id = 7;
        approved_tiles[3].city_owner_id = 2;
        approved_tiles[3].city_population = 4;
        approved_tiles[3].city_size = 0;
        approved_tiles[3].city_culture_group = 1;
        approved_tiles[3].city_era = 1;
        approved_tiles[3].city_flags = C3X_RENDERER_CITY_WALLED;

        c3x_renderer_frame_v1 approved_frame = frame;
        approved_frame.target_width = 1024;
        approved_frame.target_height = 384;
        approved_frame.clip_left = 0;
        approved_frame.clip_top = 0;
        approved_frame.clip_right = approved_frame.target_width;
        approved_frame.clip_bottom = approved_frame.target_height;
        approved_frame.tile_count = static_cast<c3x_renderer_u32>(std::size(approved_tiles));
        approved_frame.tiles = approved_tiles;
        approved_frame.world_width_tiles = 100;
        approved_frame.world_height_tiles = 80;
        approved_frame.world_wrap_x = 0;
        approved_frame.visible_animation_count = 0;
        std::uint64_t zoom_hashes[2] = {};
        int zoom_widths[2] = {128, 64};
        int zoom_heights[2] = {64, 32};
        c3x_renderer_output_v1 approved_output = {
            C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        for (int zoom = 0; zoom < 2; ++zoom) {
            approved_frame.tile_width = zoom_widths[zoom];
            approved_frame.tile_height = zoom_heights[zoom];
            for (std::size_t index = 0; index < std::size(approved_tiles); ++index) {
                approved_tiles[index].anchor_x = 16 + static_cast<int>(index) * (zoom == 0 ? 64 : 44);
                approved_tiles[index].anchor_y = 24 + static_cast<int>(index & 1u) * zoom_heights[zoom];
            }
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
                approved_output.rendered_tile_count != 14 || approved_output.fallback_tile_count != 0 ||
                approved_output.replacement_tile_count != static_cast<c3x_renderer_u32>(std::size(approved_tiles)) ||
                approved_output.replacement_tile_flags == nullptr) {
                std::fprintf(stderr, "approved fixture zoom=%d rendered=%u fallback=%u ownership=%u\n",
                    zoom, approved_output.rendered_tile_count, approved_output.fallback_tile_count,
                    approved_output.replacement_tile_count);
                for (c3x_renderer_u32 index = 0; index < approved_output.fallback_tile_count; ++index)
                    std::fprintf(stderr, " fallback_index=%u\n", approved_output.fallback_tile_indices[index]);
                return fail("approved L9-L19 category fixture did not render with exclusive ownership");
            }
            c3x_renderer_u32 const * flags = approved_output.replacement_tile_flags;
            if ((flags[0] & (C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED |
                             C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED)) !=
                    (C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED |
                     C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED) ||
                (flags[5] & C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED) == 0 ||
                (flags[6] & C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED) == 0 ||
                (flags[1] & C3X_RENDERER_TILE_CUSTOM_RIVER_REPLACED) == 0 ||
                (flags[2] & C3X_RENDERER_TILE_CUSTOM_RIVER_REPLACED) == 0 ||
                (flags[1] & C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED) == 0 ||
                (flags[2] & C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED) == 0 ||
                (flags[3] & C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED) == 0 ||
                (flags[4] & C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED) == 0 ||
                (flags[5] & C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED) == 0 ||
                (flags[3] & C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED) == 0 ||
                (flags[6] & C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED) == 0 ||
                (flags[2] & C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED) == 0 ||
                flags[10] != (C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED |
                              C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED) ||
                flags[11] != (C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED |
                              C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED) ||
                (flags[12] & (C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED |
                              C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED)) !=
                    (C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED |
                     C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED))
                return fail("approved dune/vegetation/marsh/volcano ownership is invalid");
            std::size_t approved_bytes = static_cast<std::size_t>(approved_output.stride_bytes) *
                approved_output.height;
            zoom_hashes[zoom] = hash_pixels(approved_output.bgra_pixels, approved_bytes);
            if (zoom_hashes[zoom] == 0)
                return fail("approved L9-L19 zoom fixture was blank");

            approved_tiles[10].has_effect = 1;
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
                (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0 ||
                (approved_output.replacement_tile_flags[10] &
                 C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED) == 0 ||
                hash_pixels(approved_output.bgra_pixels, approved_bytes) == zoom_hashes[zoom])
                return fail("approved active volcano did not invalidate and change authored surface");
            approved_tiles[10].has_effect = 0;
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            int dormant_result = render(&approved_frame, &approved_output);
            std::uint64_t dormant_hash = hash_pixels(
                approved_output.bgra_pixels, approved_bytes);
            if (dormant_result != C3X_RENDERER_RESULT_OK ||
                dormant_hash != zoom_hashes[zoom]) {
                std::fprintf(stderr,
                    "dormant volcano zoom=%d expected=%llu actual=%llu result=%d\n",
                    zoom, static_cast<unsigned long long>(zoom_hashes[zoom]),
                    static_cast<unsigned long long>(dormant_hash), dormant_result);
                return fail("approved dormant volcano did not restore deterministically");
            }

            approved_frame.clip_left = 13;
            approved_frame.clip_top = 9;
            approved_frame.clip_right = approved_frame.target_width - 11;
            approved_frame.clip_bottom = approved_frame.target_height - 7;
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
                approved_output.clip_left != 13 || approved_output.clip_top != 9 ||
                approved_output.clip_right != approved_frame.target_width - 11 ||
                approved_output.clip_bottom != approved_frame.target_height - 7 ||
                approved_output.renderer_cpu_ticks != 0 ||
                approved_output.frame_invalidation_flags != 0)
                return fail("approved L9-L18 partial redraw did not retain complete terrain");
            approved_frame.clip_left = 0;
            approved_frame.clip_top = 0;
            approved_frame.clip_right = approved_frame.target_width;
            approved_frame.clip_bottom = approved_frame.target_height;

            approved_tiles[5].anchor_x += 3;
            approved_tiles[5].anchor_y -= 2;
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
                (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0)
                return fail("approved L9-L18 pixel scroll did not invalidate exact anchors");
            approved_tiles[5].anchor_x -= 3;
            approved_tiles[5].anchor_y += 2;

            approved_frame.world_wrap_x = 1;
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
                (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_WRAP) == 0 ||
                approved_output.replacement_tile_flags[13] == 0)
                return fail("approved L9-L18 horizontal wrap occurrence was not rendered independently");
            approved_frame.world_wrap_x = 0;
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            int restored_result = render(&approved_frame, &approved_output);
            std::uint64_t restored_zoom_hash = hash_pixels(
                approved_output.bgra_pixels, approved_bytes);
            if (restored_result != C3X_RENDERER_RESULT_OK ||
                restored_zoom_hash != zoom_hashes[zoom]) {
                std::fprintf(stderr, "zoom restore zoom=%d expected=%llu actual=%llu result=%d\n",
                    zoom, static_cast<unsigned long long>(zoom_hashes[zoom]),
                    static_cast<unsigned long long>(restored_zoom_hash), restored_result);
                return fail("approved L9-L18 zoom fixture retained stale clip, scroll, or wrap pixels");
            }
            c3x_renderer_u32 zoom_hits = approved_output.cache_hits;
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
                hash_pixels(approved_output.bgra_pixels, approved_bytes) != zoom_hashes[zoom] ||
                approved_output.cache_hits != zoom_hits + 1 ||
                approved_output.frame_invalidation_flags != 0)
                return fail("approved L9-L18 static zoom fixture was not exact and idle");
        }
        if (zoom_hashes[0] == zoom_hashes[1])
            return fail("approved L9-L18 zoom fixtures were not distinct");

        // A terrain owner change is cache-relevant, while unit overlays are
        // not.  Reset must rebuild D3D resources from the same
        // approved CPU-side handoff without changing the resulting pixels.
        approved_frame.tile_width = 128;
        approved_frame.tile_height = 64;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK)
            return fail("approved ownership fixture could not be restored");
        std::size_t ownership_bytes = static_cast<std::size_t>(approved_output.stride_bytes) *
            approved_output.height;
        std::uint64_t owned_hash = hash_pixels(approved_output.bgra_pixels, ownership_bytes);
        c3x_renderer_u32 ownership_hits = approved_output.cache_hits;
        approved_tiles[5].unit_type_id = 9;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            approved_output.cache_hits != ownership_hits + 1 ||
            approved_output.frame_invalidation_flags != 0)
            return fail("approved retained unit selector invalidated terrain ownership");
        approved_tiles[5].real_terrain_type = 2;
        approved_tiles[5].feature_flags = 0;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_OWNERSHIP) == 0 ||
            (approved_output.replacement_tile_flags[5] & C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED) != 0 ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) == owned_hash)
            return fail("approved feature ownership change reused stale terrain");
        approved_tiles[5].real_terrain_type = 7;
        approved_tiles[5].feature_flags = C3X_RENDERER_FEATURE_FOREST;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        int vegetation_restore_result = render(&approved_frame, &approved_output);
        std::uint64_t vegetation_restore_hash = hash_pixels(
            approved_output.bgra_pixels, ownership_bytes);
        if (vegetation_restore_result != C3X_RENDERER_RESULT_OK ||
            vegetation_restore_hash != owned_hash) {
            std::fprintf(stderr, "vegetation restore expected=%llu actual=%llu result=%d\n",
                static_cast<unsigned long long>(owned_hash),
                static_cast<unsigned long long>(vegetation_restore_hash),
                vegetation_restore_result);
            return fail("approved vegetation fixture did not restore deterministically");
        }
        approved_tiles[6].improvement_flags &= ~C3X_RENDERER_IMPROVEMENT_MINE;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0 ||
            (approved_output.replacement_tile_flags[6] &
             C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED) != 0 ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) == owned_hash)
            return fail("approved mine body was absent or retained after ownership changed");
        approved_tiles[6].improvement_flags |= C3X_RENDERER_IMPROVEMENT_MINE;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) != owned_hash)
            return fail("approved mine fixture did not restore deterministically");
        approved_tiles[2].improvement_flags &= ~C3X_RENDERER_IMPROVEMENT_IRRIGATION;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0 ||
            (approved_output.replacement_tile_flags[2] &
             C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED) != 0 ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) == owned_hash)
            return fail("approved farm body was absent or retained after ownership changed");
        approved_tiles[2].improvement_flags |= C3X_RENDERER_IMPROVEMENT_IRRIGATION;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        int farm_restore_result = render(&approved_frame, &approved_output);
        if (farm_restore_result != C3X_RENDERER_RESULT_OK)
            return fail("approved farm fixture restore render failed");
        std::uint64_t farm_restore_hash = hash_pixels(
            approved_output.bgra_pixels, ownership_bytes);
        if (farm_restore_hash != owned_hash)
            return fail("approved farm fixture did not restore deterministically");
        approved_tiles[2].irrigation_mask = 7u;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        int farm_topology_result = render(&approved_frame, &approved_output);
        if (farm_topology_result != C3X_RENDERER_RESULT_OK)
            return fail("farm topology render failed");
        std::uint64_t farm_topology_hash = hash_pixels(
            approved_output.bgra_pixels, ownership_bytes);
        if ((approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0 ||
            farm_topology_hash == owned_hash)
            return fail("farm topology did not invalidate static terrain");
        approved_tiles[2].irrigation_mask = 15u;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) != owned_hash)
            return fail("farm topology fixture did not restore deterministically");
        approved_tiles[11].real_terrain_type = 2;
        approved_tiles[11].feature_flags = 0;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_OWNERSHIP) == 0 ||
            (approved_output.replacement_tile_flags[11] & C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED) != 0 ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) == owned_hash)
            return fail("approved marsh body was absent or retained after ownership changed");
        approved_tiles[11].real_terrain_type = 9;
        approved_tiles[11].feature_flags = C3X_RENDERER_FEATURE_MARSH;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) != owned_hash)
            return fail("approved marsh fixture did not restore deterministically");
        approved_tiles[0].real_terrain_type = 4;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            (approved_output.replacement_tile_flags[0] & C3X_RENDERER_TILE_CUSTOM_DUNES_REPLACED) != 0 ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) == owned_hash)
            return fail("approved dune body was absent or retained after ownership changed");
        approved_tiles[0].real_terrain_type = 0;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) != owned_hash)
            return fail("approved dune fixture did not restore deterministically");
        approved_tiles[3].real_terrain_type = 2;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) == owned_hash)
            return fail("approved authored hill relief was absent after exact terrain identity transfer");
        approved_tiles[3].real_terrain_type = 5;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) != owned_hash)
            return fail("approved authored hill relief did not restore deterministically");
        approved_tiles[4].real_terrain_type = 2;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) == owned_hash)
            return fail("approved authored mountain relief was absent after exact terrain identity transfer");
        approved_tiles[4].real_terrain_type = 6;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            hash_pixels(approved_output.bgra_pixels, ownership_bytes) != owned_hash)
            return fail("approved authored mountain relief did not restore deterministically");
        // Establish a fresh geometry baseline. Exact viewport-cache hits above
        // intentionally do not replace the reusable geometry owner.
        reset();
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK)
            return fail("approved geometry baseline did not rebuild after reset");
        c3x_renderer_u32 generation_before = approved_output.device_generation;
        c3x_renderer_u32 translation_hits_before = approved_output.cache_hits;
        for (c3x_renderer_tile_v1 & tile : approved_tiles) {
            tile.anchor_x += 3;
            tile.anchor_y -= 2;
        }
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            approved_output.cache_hits != translation_hits_before + 1)
            return fail("approved translated viewport did not reuse world geometry");
        std::size_t reset_bytes = static_cast<std::size_t>(approved_output.stride_bytes) *
            approved_output.height;
        std::vector<std::uint8_t> translated_pixels(
            static_cast<std::uint8_t const *>(approved_output.bgra_pixels),
            static_cast<std::uint8_t const *>(approved_output.bgra_pixels) + reset_bytes);
        std::uint64_t reset_hash = hash_pixels(translated_pixels.data(), reset_bytes);
        reset();
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        int reset_result = render(&approved_frame, &approved_output);
        std::uint64_t reset_after_hash = hash_pixels(
            approved_output.bgra_pixels, reset_bytes);
        std::size_t differing_pixels = 0;
        std::size_t differing_bytes = 0, material_differences = 0;
        std::uint64_t total_channel_delta = 0;
        unsigned int maximum_channel_delta = 0;
        auto const * cold_pixels =
            static_cast<std::uint8_t const *>(approved_output.bgra_pixels);
        if (cold_pixels != nullptr) {
            for (std::size_t offset = 0; offset < reset_bytes; offset += 4) {
                bool pixel_differs = false, material_differs = false;
                for (std::size_t channel = 0; channel < 4; ++channel) {
                    unsigned int translated_value = translated_pixels[offset + channel];
                    unsigned int cold_value = cold_pixels[offset + channel];
                    unsigned int delta = translated_value > cold_value
                        ? translated_value - cold_value : cold_value - translated_value;
                    total_channel_delta += delta;
                    material_differs = material_differs || delta > 1u;
                    if (delta > maximum_channel_delta)
                        maximum_channel_delta = delta;
                    if (delta != 0) {
                        ++differing_bytes;
                        pixel_differs = true;
                    }
                }
                if (pixel_differs)
                    ++differing_pixels;
                if (material_differs) ++material_differences;
            }
        }
        // Applying an integer camera offset in the vertex shader is allowed to
        // choose the neighboring raster edge when the equivalent CPU rebuild
        // rounds NDC in a different order. Keep that bounded to sub-per-mille
        // edge coverage with no large color excursion.
        // Reusing pixels preserves their shading; a fresh raster can round a
        // channel differently when an odd-pixel shift changes GPU derivative
        // quads. Bound that separately from actual edge/material changes.
        bool translation_equivalent = material_differences <= reset_bytes / 4000 &&
            total_channel_delta <= reset_bytes / 100 && maximum_channel_delta <= 128;
        if (reset_result != C3X_RENDERER_RESULT_OK ||
            approved_output.device_generation <= generation_before ||
            (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_DEVICE) == 0 ||
            !translation_equivalent) {
            std::fprintf(stderr,
                "device reset expected=%llu actual=%llu result=%d generation=%u->%u invalidations=%u differing_pixels=%zu/%zu differing_bytes=%zu max_channel_delta=%u\n",
                static_cast<unsigned long long>(reset_hash),
                static_cast<unsigned long long>(reset_after_hash), reset_result,
                generation_before, approved_output.device_generation,
                approved_output.frame_invalidation_flags, differing_pixels,
                reset_bytes / 4, differing_bytes, maximum_channel_delta);
            std::fprintf(stderr, "material_differences=%zu channel_delta=%llu\n", material_differences, total_channel_delta);
            FILE * diagnostic = nullptr;
            fopen_s(&diagnostic, "build\\translation-warm.bgra", "wb");
            if (diagnostic) { std::fwrite(translated_pixels.data(), 1, reset_bytes, diagnostic); std::fclose(diagnostic); }
            fopen_s(&diagnostic, "build\\translation-cold.bgra", "wb");
            if (diagnostic) { std::fwrite(cold_pixels, 1, reset_bytes, diagnostic); std::fclose(diagnostic); }
            return fail("translated viewport did not match a cold device-reset rebuild");
        }

        // A small alternate region may temporarily replace the active compiled
        // geometry. Revisit the original region at a new camera occurrence so
        // the exact pixel LRU cannot satisfy it and the bounded GPU cache must.
        int prior_region_terrain = approved_tiles[0].terrain_type;
        int prior_region_real_terrain = approved_tiles[0].real_terrain_type;
        approved_tiles[0].terrain_type = 2;
        approved_tiles[0].real_terrain_type = 2;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            approved_output.renderer_cpu_ticks <= 0)
            return fail("alternate bounded geometry region did not build");
        approved_tiles[0].terrain_type = prior_region_terrain;
        approved_tiles[0].real_terrain_type = prior_region_real_terrain;
        for (c3x_renderer_tile_v1 & tile : approved_tiles) {
            tile.anchor_x += 2;
            tile.anchor_y += 1;
        }
        c3x_renderer_u32 region_hits_before = approved_output.cache_hits;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&approved_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            approved_output.cache_hits != region_hits_before + 1 ||
            approved_output.renderer_cpu_ticks <= 0)
            return fail("bounded GPU region cache did not restore a revisited world region");

        // Match the full-screen capture volume observed at the live m19 boundary:
        // 400 logical tiles plus 400 odd-parity companion records.
        std::vector<c3x_renderer_tile_v1> live_scale_tiles;
        live_scale_tiles.reserve(800);
        int live_types[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
        for (int index = 0; index < 400; ++index) {
            int column = index % 20;
            int row = index / 20;
            int real_type = live_types[index % static_cast<int>(std::size(live_types))];
            int base_type = (real_type >= 5 && real_type <= 10) ? 2 : real_type;
            c3x_renderer_tile_v1 tile = make_tile(
                column * 2, row * 2, 512 + (column - row) * 64,
                32 + (column + row) * 32, base_type,
                static_cast<c3x_renderer_u32>(0x2000 + index));
            tile.real_terrain_type = real_type;
            if (real_type == 7) tile.feature_flags = C3X_RENDERER_FEATURE_FOREST;
            if (real_type == 8) tile.feature_flags = C3X_RENDERER_FEATURE_JUNGLE;
            if (real_type == 9) tile.feature_flags = C3X_RENDERER_FEATURE_MARSH;
            if (real_type == 10) tile.feature_flags = C3X_RENDERER_FEATURE_VOLCANO;
            live_scale_tiles.push_back(tile);
            c3x_renderer_tile_v1 companion = make_tile(
                column * 2 + 1, row * 2, tile.anchor_x + 64, tile.anchor_y,
                -1, static_cast<c3x_renderer_u32>(0x4000 + index));
            companion.tile_flags = C3X_RENDERER_TILE_VANILLA_BASE_CALL;
            companion.real_terrain_type = -1;
            live_scale_tiles.push_back(companion);
        }
        c3x_renderer_frame_v1 live_scale_frame = approved_frame;
        live_scale_frame.target_width = 1121;
        live_scale_frame.target_height = 1192;
        live_scale_frame.clip_left = 0;
        live_scale_frame.clip_top = 0;
        live_scale_frame.clip_right = live_scale_frame.target_width;
        live_scale_frame.clip_bottom = live_scale_frame.target_height;
        live_scale_frame.tile_width = 128;
        live_scale_frame.tile_height = 64;
        live_scale_frame.tile_count = static_cast<c3x_renderer_u32>(live_scale_tiles.size());
        live_scale_frame.tiles = live_scale_tiles.data();
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&live_scale_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            approved_output.rendered_tile_count != 400 ||
            approved_output.fallback_tile_count != 0 ||
            approved_output.replacement_tile_count != 800 ||
            hash_pixels(approved_output.bgra_pixels,
                static_cast<std::size_t>(approved_output.stride_bytes) * approved_output.height) == 0)
            return fail("live-scale 800-record capture did not render atomically");
        LARGE_INTEGER performance_frequency = {};
        QueryPerformanceFrequency(&performance_frequency);
        double live_scale_milliseconds = performance_frequency.QuadPart > 0
            ? static_cast<double>(approved_output.renderer_cpu_ticks) * 1000.0 /
                static_cast<double>(performance_frequency.QuadPart)
            : -1.0;
        std::printf("PERF approved_live_scale_cold: tiles=400 records=800 milliseconds=%.3f ticks=%lld.\n",
                    live_scale_milliseconds,
                    static_cast<long long>(approved_output.renderer_cpu_ticks));
        c3x_renderer_u32 cold_upload_bytes = approved_output.geometry_upload_bytes;
        if (approved_output.geometry_tiles_built != 400 || cold_upload_bytes == 0 ||
            approved_output.geometry_cache_bytes > 192u * 1024u * 1024u)
            return fail("live-scale mesh compilation exceeded its memory contract");
        c3x_renderer_i64 live_scale_cold_ticks = approved_output.renderer_cpu_ticks;
        c3x_renderer_u32 live_scale_hits = approved_output.cache_hits;
        for (c3x_renderer_tile_v1 & tile : live_scale_tiles) {
            tile.anchor_x += 4;
            tile.anchor_y -= 2;
        }
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&live_scale_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            (approved_output.frame_invalidation_flags & C3X_RENDERER_INVALIDATE_SCENE) == 0 ||
            approved_output.renderer_cpu_ticks <= 0 ||
            approved_output.renderer_cpu_ticks * 4 >= live_scale_cold_ticks ||
            approved_output.cache_hits != live_scale_hits + 1 ||
            approved_output.geometry_tiles_built != 0 || approved_output.geometry_upload_bytes != 0 ||
            approved_output.raster_reused_pixels < 99u *
                static_cast<c3x_renderer_u32>(live_scale_frame.target_width * live_scale_frame.target_height) / 100u)
            return fail("live-scale pixel scroll did not reuse world geometry");
        double live_scale_scroll_milliseconds = performance_frequency.QuadPart > 0
            ? static_cast<double>(approved_output.renderer_cpu_ticks) * 1000.0 /
                static_cast<double>(performance_frequency.QuadPart)
            : -1.0;
        std::printf("PERF approved_live_scale_scroll: tiles=400 records=800 milliseconds=%.3f ticks=%lld.\n",
                    live_scale_scroll_milliseconds,
                    static_cast<long long>(approved_output.renderer_cpu_ticks));

        // Cross a tile boundary while retaining nearly the entire captured
        // world region. This defeats both the exact pixel cache and the whole-
        // viewport geometry key, while overlapping tiles can reuse their
        // anchor-independent semantic shadow fields.
        live_scale_tiles[0] = make_tile(40, 0, 1795, 670, 0, 0x6200u);
        live_scale_tiles[0].real_terrain_type = 0;
        live_scale_tiles[1] = make_tile(41, 0, 1859, 670, -1, 0x6201u);
        live_scale_tiles[1].tile_flags = C3X_RENDERER_TILE_VANILLA_BASE_CALL;
        live_scale_tiles[1].real_terrain_type = -1;
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&live_scale_frame, &approved_output) != C3X_RENDERER_RESULT_OK ||
            approved_output.renderer_cpu_ticks <= 0 ||
            approved_output.renderer_cpu_ticks * 4 >= live_scale_cold_ticks ||
            approved_output.fallback_tile_count != 0 ||
            approved_output.geometry_tiles_built > 40 ||
            approved_output.geometry_tiles_reused < 360 ||
            approved_output.geometry_tiles_built + approved_output.geometry_tiles_reused != 400 ||
            approved_output.geometry_upload_bytes >= cold_upload_bytes / 5 ||
            approved_output.geometry_cache_bytes > 192u * 1024u * 1024u)
            return fail("tile-boundary scroll did not reuse world semantic fields");
        double live_scale_boundary_milliseconds = performance_frequency.QuadPart > 0
            ? static_cast<double>(approved_output.renderer_cpu_ticks) * 1000.0 /
                static_cast<double>(performance_frequency.QuadPart)
            : -1.0;
        std::printf("PERF approved_live_scale_tile_boundary: tiles=400 records=800 milliseconds=%.3f ticks=%lld.\n",
                    live_scale_boundary_milliseconds,
                    static_cast<long long>(approved_output.renderer_cpu_ticks));

        std::printf("CACHE boundary: built=%u reused=%u upload_bytes=%u cache_bytes=%u.\n",
            approved_output.geometry_tiles_built, approved_output.geometry_tiles_reused,
            approved_output.geometry_upload_bytes, approved_output.geometry_cache_bytes);
        std::size_t live_bytes = static_cast<std::size_t>(approved_output.stride_bytes) * approved_output.height;
        std::vector<std::uint8_t> boundary_pixels(
            static_cast<std::uint8_t const *>(approved_output.bgra_pixels),
            static_cast<std::uint8_t const *>(approved_output.bgra_pixels) + live_bytes);
        reset();
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&live_scale_frame, &approved_output) != C3X_RENDERER_RESULT_OK)
            return fail("cold boundary witness failed to render");
        auto const * boundary_cold = static_cast<std::uint8_t const *>(approved_output.bgra_pixels);
        std::uint64_t boundary_error = 0;
        std::size_t boundary_changed = 0;
        for (std::size_t i = 0; i < live_bytes; i += 4) {
            bool changed = false;
            for (std::size_t c = 0; c < 4; ++c) {
                unsigned delta = static_cast<unsigned>(std::abs(
                    static_cast<int>(boundary_pixels[i+c]) - boundary_cold[i+c]));
                boundary_error += delta;
                changed = changed || delta > 2u;
            }
            if (changed) ++boundary_changed;
        }
        // Classify <=2/255 as quantization noise, retaining a strict total
        // error cap and sub-per-mille coverage budget for larger differences.
        // Exact cache returns below still require byte-identical images.
        if (boundary_changed > live_bytes / 4000 || boundary_error > live_bytes / 100) {
            std::printf("DIFF boundary: changed=%zu error=%llu bytes=%zu.\n", boundary_changed,
                static_cast<unsigned long long>(boundary_error), live_bytes);
            FILE * diagnostic = nullptr;
            if (fopen_s(&diagnostic, "build/boundary-warm.bgra", "wb") == 0 && diagnostic) {
                fwrite(boundary_pixels.data(), 1, live_bytes, diagnostic); fclose(diagnostic);
            }
            if (fopen_s(&diagnostic, "build/boundary-cold.bgra", "wb") == 0 && diagnostic) {
                fwrite(boundary_cold, 1, live_bytes, diagnostic); fclose(diagnostic);
            }
            return fail("incremental boundary cache differs from authoritative cold rebuild");
        }

        std::vector<double> unit_times, scroll_times;
        std::uint64_t static_hash = hash_pixels(approved_output.bgra_pixels, live_bytes);
        for (int i = 0; i < 64; ++i) {
            live_scale_tiles[0].unit_direction = i & 7;
            live_scale_tiles[0].unit_type_id = i;
            live_scale_frame.visible_animation_count = 1;
            LARGE_INTEGER begin = {}, end = {};
            QueryPerformanceCounter(&begin);
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            int result = render(&live_scale_frame, &approved_output);
            QueryPerformanceCounter(&end);
            unit_times.push_back(1000.0 * (end.QuadPart - begin.QuadPart) / performance_frequency.QuadPart);
            if (result != C3X_RENDERER_RESULT_OK || approved_output.renderer_cpu_ticks != 0 ||
                approved_output.geometry_tiles_built != 0 || approved_output.geometry_upload_bytes != 0 ||
                approved_output.raster_draw_pixels != 0 || approved_output.request_continuous_redraw != 1 ||
                hash_pixels(approved_output.bgra_pixels, live_bytes) != static_hash)
                return fail("repeated unit selection performed terrain work or changed the cached image");
        }
        live_scale_frame.visible_animation_count = 0;
        for (int i = 0; i < 32; ++i) {
            int dx = i < 16 ? 4 : -4;
            int dy = i < 8 || i >= 24 ? 2 : -2;
            for (auto & tile : live_scale_tiles) { tile.anchor_x += dx; tile.anchor_y += dy; }
            LARGE_INTEGER begin = {}, end = {};
            QueryPerformanceCounter(&begin);
            approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
            int result = render(&live_scale_frame, &approved_output);
            QueryPerformanceCounter(&end);
            scroll_times.push_back(1000.0 * (end.QuadPart - begin.QuadPart) / performance_frequency.QuadPart);
            if (result != C3X_RENDERER_RESULT_OK || approved_output.geometry_tiles_built != 0 ||
                approved_output.geometry_upload_bytes != 0 || approved_output.fallback_tile_count != 0 ||
                approved_output.raster_draw_pixels > 20000)
                return fail("four-direction warm scroll rebuilt geometry or rerasterized the viewport");
        }
        std::sort(unit_times.begin(), unit_times.end());
        std::sort(scroll_times.begin(), scroll_times.end());
        std::printf("PERF repeated_unit_selection: samples=64 p50_ms=%.3f p95_ms=%.3f max_ms=%.3f.\n",
            unit_times[32], unit_times[60], unit_times.back());
        std::printf("PERF repeated_pixel_scroll: samples=32 p50_ms=%.3f p95_ms=%.3f max_ms=%.3f.\n",
            scroll_times[16], scroll_times[30], scroll_times.back());
        if (unit_times[60] > 16.7 || scroll_times[30] > 33.4)
            return fail("warm interaction latency exceeded the integration budget");

        // Primary scroll workload: Civ III's staggered map coordinates and
        // discrete several-tile jumps, with whole rows/columns entering/leaving.
        // Retain the tiny-pixel test above only as a strip-copy regression.
        auto jump_tiles = [&](int camera_x, int camera_y) {
            std::vector<c3x_renderer_tile_v1> result;
            for (int y = camera_y-8; y < camera_y + 48; ++y)
                for (int x = camera_x-8; x < camera_x + 28; ++x) {
                    bool visible = y >= camera_y && y < camera_y+40 && x >= camera_x && x < camera_x+20;
                    bool logical = ((x+y) & 1) == 0;
                    if (!visible && !logical) continue;
                    unsigned seed = static_cast<unsigned>(x) * 73856093u ^
                        static_cast<unsigned>(y) * 19349663u;
                    int real_type = static_cast<int>((seed >> 8) % 14u);
                    int ground = real_type >= 5 && real_type <= 10 ? 2 : real_type;
                    auto tile = make_tile(x, y, (x-camera_x-1)*64, (y-camera_y-1)*32,
                        logical ? ground : -1, seed);
                    tile.real_terrain_type = logical ? real_type : -1;
                    if (!logical) tile.tile_flags = C3X_RENDERER_TILE_VANILLA_BASE_CALL;
                    else if (!visible) tile.tile_flags = C3X_RENDERER_TILE_TOPOLOGY_HALO;
                    result.push_back(tile);
                }
            return result;
        };
        c3x_renderer_frame_v1 jump_frame = live_scale_frame;
        jump_frame.visible_animation_count = 0;
        jump_frame.world_width_tiles = 160; jump_frame.world_height_tiles = 160;
        jump_frame.world_wrap_x = jump_frame.world_wrap_y = 0;
        std::vector<double> jump_first, jump_return;
        constexpr int jumps[][2] = {{40,40},{44,40},{48,40},{48,44},{44,44},{40,44}};
        std::vector<std::uint64_t> jump_hashes;
        std::vector<unsigned char> jump_witness;
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < 6; ++j) {
                int view = pass == 0 ? j : 5-j;
                auto jump_records = jump_tiles(jumps[view][0], jumps[view][1]);
                jump_frame.tiles = jump_records.data(); jump_frame.tile_count = static_cast<unsigned>(jump_records.size());
                LARGE_INTEGER begin = {}, end = {};
                QueryPerformanceCounter(&begin);
                approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
                int result = render(&jump_frame, &approved_output);
                QueryPerformanceCounter(&end);
                double milliseconds = 1000.0*(end.QuadPart-begin.QuadPart)/performance_frequency.QuadPart;
                if (result != C3X_RENDERER_RESULT_OK || approved_output.rendered_tile_count != 400 ||
                    approved_output.fallback_tile_count != 0 ||
                    approved_output.geometry_cache_bytes > 192u*1024u*1024u)
                    return fail("several-tile jump failed its rendering or memory contract");
                if (pass == 0 && j != 0 && (approved_output.geometry_tiles_built > 80 ||
                    approved_output.geometry_tiles_reused < 320 ||
                    approved_output.raster_reused_pixels < static_cast<unsigned>(jump_frame.target_width*jump_frame.target_height)/2))
                    return fail("several-tile jump discarded unchanged topology or bitmap overlap");
                if (pass != 0 && milliseconds > 16.7)
                    return fail("cached several-tile return exceeded the interaction budget");
                auto hash = hash_pixels(approved_output.bgra_pixels, live_bytes);
                if (pass == 0 && j == 5) jump_witness.assign(
                    static_cast<unsigned char const *>(approved_output.bgra_pixels),
                    static_cast<unsigned char const *>(approved_output.bgra_pixels)+live_bytes);
                if (pass == 0) jump_hashes.push_back(hash);
                else if (hash != jump_hashes[view] || approved_output.geometry_tiles_built != 0 ||
                    approved_output.geometry_upload_bytes != 0 || approved_output.raster_draw_pixels != 0)
                    return fail("several-tile return trip rebuilt or changed a retained view");
                std::printf("PERF tile_jump: pass=%s view=%d camera_x=%d camera_y=%d milliseconds=%.3f built=%u reused=%u reused_pixels=%u draw_pixels=%u.\n",
                    pass == 0 ? "first" : "return", view, jumps[view][0], jumps[view][1], milliseconds,
                    approved_output.geometry_tiles_built, approved_output.geometry_tiles_reused,
                    approved_output.raster_reused_pixels, approved_output.raster_draw_pixels);
                if (pass != 0) jump_return.push_back(milliseconds);
                else if (j != 0) jump_first.push_back(milliseconds);

            }
        }
        auto witness_records = jump_tiles(jumps[5][0], jumps[5][1]);
        jump_frame.tiles = witness_records.data(); jump_frame.tile_count = static_cast<unsigned>(witness_records.size());
        reset();
        approved_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&jump_frame, &approved_output) != C3X_RENDERER_RESULT_OK)
            return fail("cold several-tile jump witness failed");
        auto witness_pixels = static_cast<unsigned char const *>(approved_output.bgra_pixels);
        std::size_t jump_changed = 0; std::uint64_t jump_error = 0;
        for (std::size_t i = 0; i < live_bytes; i += 4) {
            bool changed = false;
            for (std::size_t c = 0; c < 4; ++c) {
                unsigned delta = static_cast<unsigned>(std::abs(static_cast<int>(jump_witness[i+c])-witness_pixels[i+c]));
                jump_error += delta; changed = changed || delta > 2;
            }
            if (changed) ++jump_changed;
        }
        std::printf("DIFF tile_jump: changed=%zu error=%llu bytes=%zu.\n", jump_changed,
            static_cast<unsigned long long>(jump_error), live_bytes);
        if (jump_changed > live_bytes / 4000 || jump_error > live_bytes / 100)
            return fail("several-tile jump overlap differs from its cold reconstruction");
        std::sort(jump_first.begin(),jump_first.end()); std::sort(jump_return.begin(),jump_return.end());
        std::printf("PERF repeated_tile_jump: samples=5 first_p50_ms=%.3f first_p95_ms=%.3f return_p95_ms=%.3f.\n",
            jump_first[2],jump_first.back(),jump_return.back());

        output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&frame, &output) != C3X_RENDERER_RESULT_OK)
            return fail("could not restore primary output after approved handoff matrix");
    }

    if (synthetic_definition_mode) {
        c3x_renderer_tile_v1 blend_tiles[] = {
            make_tile(0, 0, 32, 32, 2, 2001),
            make_tile(1, 1, 96, 64, 1, 2002)
        };
        c3x_renderer_frame_v1 blend_frame = frame;
        blend_frame.target_width = 256;
        blend_frame.target_height = 160;
        blend_frame.clip_left = 0;
        blend_frame.clip_top = 0;
        blend_frame.clip_right = 256;
        blend_frame.clip_bottom = 160;
        blend_frame.tile_count = static_cast<c3x_renderer_u32>(std::size(blend_tiles));
        blend_frame.tiles = blend_tiles;
        c3x_renderer_output_v1 blend_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&blend_frame, &blend_output) != C3X_RENDERER_RESULT_OK)
            return fail("connected material-boundary fixture render failed");
        auto const * blend_pixels = static_cast<std::uint32_t const *>(blend_output.bgra_pixels);
        std::uint32_t grass_interior = blend_pixels[64 * blend_output.width + 96] & 0x00ffffffu;
        std::uint32_t plains_interior = blend_pixels[96 * blend_output.width + 160] & 0x00ffffffu;
        std::uint32_t shared_edge = blend_pixels[80 * blend_output.width + 128] & 0x00ffffffu;
        if (shared_edge == grass_interior || shared_edge == plains_interior ||
            ((blend_pixels[80 * blend_output.width + 128] >> 24) < 240u))
            return fail("adjacent mapped terrain materials retained a hard diamond edge");

        blend_tiles[1].terrain_type = 3;
        blend_tiles[1].real_terrain_type = 3;
        blend_output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&blend_frame, &blend_output) != C3X_RENDERER_RESULT_OK ||
            blend_output.fallback_tile_count != 0 || blend_output.replacement_tile_flags[1] != 0)
            return fail("unconfigured terrain was not simply omitted");
        blend_pixels = static_cast<std::uint32_t const *>(blend_output.bgra_pixels);
        if ((blend_pixels[96 * blend_output.width + 200] >> 24) != 0)
            return fail("omitted terrain unexpectedly drew pixels");
    }

    BITMAPINFO info = {};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = output.width;
    info.bmiHeader.biHeight = -output.height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    HDC dc = CreateCompatibleDC(nullptr);
    void * dest_bits = nullptr;
    HBITMAP dest_bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &dest_bits, nullptr, 0);
    if (dc == nullptr || dest_bitmap == nullptr || dest_bits == nullptr)
        return fail("could not create smoke-test destination surface");
    HGDIOBJ previous = SelectObject(dc, dest_bitmap);
    std::memset(dest_bits, 0x20, byte_count);
    if (blit(&output, dc) != C3X_RENDERER_RESULT_OK)
        return fail("bounded replacement blit failed");
    auto const * destination = static_cast<std::uint8_t const *>(dest_bits);
    auto const * source = static_cast<std::uint8_t const *>(output.bgra_pixels);
    if (std::memcmp(destination, source, byte_count) != 0)
        return fail("replacement blit did not overwrite the previous map surface");
    std::memset(dest_bits, 0x20, byte_count);
    output.clip_left = 17;
    output.clip_top = 13;
    output.clip_right = output.width - 19;
    output.clip_bottom = output.height - 11;
    if (blit(&output, dc) != C3X_RENDERER_RESULT_OK)
        return fail("partial replacement blit failed");
    auto const * destination_pixels = static_cast<std::uint32_t const *>(dest_bits);
    auto const * source_pixels = static_cast<std::uint32_t const *>(output.bgra_pixels);
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            std::size_t pixel_index = static_cast<std::size_t>(y) * output.width + x;
            bool inside_clip = x >= output.clip_left && x < output.clip_right &&
                y >= output.clip_top && y < output.clip_bottom;
            std::uint32_t expected = inside_clip ? source_pixels[pixel_index] : 0x20202020u;
            if (destination_pixels[pixel_index] != expected)
                return fail("partial replacement blit did not preserve its exact dirty rectangle");
        }
    }
    SelectObject(dc, previous);
    DeleteObject(dest_bitmap);
    DeleteDC(dc);

    c3x_renderer_scene_export_v1 request = {};
    request.api_version = C3X_RENDERER_API_VERSION;
    request.struct_size = sizeof(request);
    request.output_path = "build\\native-smoke.scene.json";
    request.fixture_id = "native-smoke";
    request.profile_id = "default";
    request.world_seed = 424242;
    request.world_width_tiles = 100;
    request.world_height_tiles = 80;
    request.world_wrap_x = 1;
    if (export_scene(&frame, &request) != C3X_RENDERER_RESULT_OK)
        return fail("versioned visible-scene export failed");

    c3x_renderer_schedule_v1 schedule_input = {};
    schedule_input.api_version = C3X_RENDERER_API_VERSION;
    schedule_input.struct_size = sizeof(schedule_input);
    schedule_input.now_ticks = 1132000;
    schedule_input.last_presented_ticks = 1000000;
    schedule_input.frequency = 1000000;
    schedule_input.event_start_ticks = 1000000;
    schedule_input.event_duration_ticks = 1000000;
    schedule_input.visible_animation_count = 1;
    schedule_input.state_flags = C3X_RENDERER_SCHEDULER_MAP_VISIBLE | C3X_RENDERER_SCHEDULER_FOCUSED;
    schedule_input.cadence_ms = 66;
    c3x_renderer_schedule_result_v1 decision = {C3X_RENDERER_API_VERSION, sizeof(decision)};
    if (schedule(&schedule_input, &decision) != C3X_RENDERER_RESULT_OK || decision.request_redraw != 1 ||
        decision.skipped_frame_count != 1 || decision.phase_millionths != 132000 ||
        decision.dirty_flags != (C3X_RENDERER_DIRTY_DYNAMIC | C3X_RENDERER_DIRTY_COMPOSITE))
        return fail("absolute-time scheduler did not skip to one bounded redraw");

    c3x_renderer_schedule_result_v1 repeated = {C3X_RENDERER_API_VERSION, sizeof(repeated)};
    if (schedule(&schedule_input, &repeated) != C3X_RENDERER_RESULT_OK ||
        std::memcmp(&decision, &repeated, sizeof(decision)) != 0)
        return fail("same timestamped event did not produce the same scheduler decision");

    schedule_input.last_presented_ticks = 1066000;
    repeated = {C3X_RENDERER_API_VERSION, sizeof(repeated)};
    if (schedule(&schedule_input, &repeated) != C3X_RENDERER_RESULT_OK ||
        repeated.phase_millionths != decision.phase_millionths)
        return fail("animation pose depends on rendered-frame count");

    schedule_input.visible_animation_count = 0;
    repeated = {C3X_RENDERER_API_VERSION, sizeof(repeated)};
    if (schedule(&schedule_input, &repeated) != C3X_RENDERER_RESULT_OK || repeated.request_redraw != 0)
        return fail("static scene requested a continuous redraw");

    schedule_input.visible_animation_count = 1;
    c3x_renderer_u32 blocked_states[] = {
        C3X_RENDERER_SCHEDULER_FOCUSED,
        C3X_RENDERER_SCHEDULER_MAP_VISIBLE,
        C3X_RENDERER_SCHEDULER_MAP_VISIBLE | C3X_RENDERER_SCHEDULER_FOCUSED | C3X_RENDERER_SCHEDULER_MODAL,
        C3X_RENDERER_SCHEDULER_MAP_VISIBLE | C3X_RENDERER_SCHEDULER_FOCUSED | C3X_RENDERER_SCHEDULER_DRAWING,
        C3X_RENDERER_SCHEDULER_MAP_VISIBLE | C3X_RENDERER_SCHEDULER_FOCUSED | C3X_RENDERER_SCHEDULER_REDRAW_PENDING
    };
    for (std::size_t blocked_index = 0; blocked_index < std::size(blocked_states); ++blocked_index) {
        c3x_renderer_u32 state = blocked_states[blocked_index];
        schedule_input.state_flags = state;
        repeated = {C3X_RENDERER_API_VERSION, sizeof(repeated)};
        c3x_renderer_u32 expected_rebase = blocked_index < 3 ? 1u : 0u;
        if (schedule(&schedule_input, &repeated) != C3X_RENDERER_RESULT_OK ||
            repeated.request_redraw != 0 || repeated.rebase_clock != expected_rebase)
            return fail("hidden, unfocused, modal, drawing, or pending state requested a redraw");
    }

    schedule_input.state_flags = C3X_RENDERER_SCHEDULER_MAP_VISIBLE | C3X_RENDERER_SCHEDULER_FOCUSED;
    schedule_input.last_presented_ticks = 1;
    schedule_input.now_ticks = 3000002;
    repeated = {C3X_RENDERER_API_VERSION, sizeof(repeated)};
    if (schedule(&schedule_input, &repeated) != C3X_RENDERER_RESULT_OK ||
        repeated.request_redraw != 0 || repeated.rebase_clock != 1)
        return fail("large pause was not rebased without a catch-up redraw");

    frame.target_width = 640;
    frame.target_height = 480;
    frame.clip_right = 640;
    frame.clip_bottom = 480;
    std::size_t large_byte_count = static_cast<std::size_t>(640 * 480 * 4);
    std::uint64_t large_environment_hashes[4] = {};
    for (std::size_t index = 0; index < std::size(environment_hours); ++index) {
        frame.hour = environment_hours[index];
        output = {C3X_RENDERER_API_VERSION, sizeof(c3x_renderer_output_v1)};
        if (render(&frame, &output) != C3X_RENDERER_RESULT_OK || output.width != 640 || output.height != 480)
            return fail("large environment fixture render failed");
        large_environment_hashes[index] = hash_pixels(output.bgra_pixels, large_byte_count);
    }
    if (large_environment_hashes[0] == large_environment_hashes[1] ||
        large_environment_hashes[0] == large_environment_hashes[2] ||
        large_environment_hashes[2] == large_environment_hashes[3])
        return fail("large noon/sunset/midnight/sunrise fixtures are not visually distinct");

    frame.api_version = 999;
    if (render(&frame, &output) != C3X_RENDERER_RESULT_BAD_ARGUMENT)
        return fail("invalid ABI input was not rejected");

    reset();
    FreeLibrary(module);
    if (external_definition_mode) {
        std::printf("PASS approved_terrain_integration: frozen approved L9-L19 terrain, dune, vegetation, marsh, volcano, river, lighting, road, railroad, resource, city, mine, farm, and tundra handoffs; authoritative invalidation; exclusive ownership; retained overlays; bounded viewport/indexed tile caches, incremental/cold parity, four-direction bitmap-strip reuse, and repeated interaction budgets; both zooms; clipping; scrolling; horizontal wrapping; deterministic reset; and hard-failure behavior passed; pixel_hash=%llu.\n",
                    static_cast<unsigned long long>(first_hash));
        return 0;
    }
    std::printf("PASS native_renderer_smoke: layered terrain definitions, multi-material DDS sampling, connected material blending, native-underlay feathering, shared environment, moonlit water, emissive/attachment primitives, atomic fallback, off-screen render, export, absolute-time scheduling, idle/pause guards, frame skipping, ABI, and bounded blit passed; pixel_hash=%llu.\n",
                static_cast<unsigned long long>(first_hash));
    return 0;
}
