#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include "c3x_renderer_api.h"

struct CsvTile {
    int x, y, base, real;
    unsigned bonus, overlays, river;
    bool topology_halo;
};

std::uint32_t preview_seed(int x, int y) {
    std::uint32_t value = 2166136261u;
    value = (value ^ static_cast<std::uint32_t>(x)) * 16777619u;
    return (value ^ static_cast<std::uint32_t>(y)) * 16777619u;
}

bool read_scene(char const * path, int & map_width, int & map_height, std::vector<CsvTile> & tiles) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || file == nullptr)
        return false;
    char header[256] = {};
    char magic[40] = {};
    unsigned count = 0;
    unsigned halo_count = 0;
    int columns = 0, rows = 0, origin_column = 0, origin_row = 0;
    bool ok = std::fgets(header, sizeof(header), file) != nullptr;
    bool viewport = ok &&
        sscanf_s(header, "%39[^,],%d,%d,%u,%d,%d,%d,%d,%u", magic,
                 static_cast<unsigned>(sizeof(magic)), &columns, &rows, &count,
                 &origin_column, &origin_row, &map_width, &map_height,
                 &halo_count) == 9 &&
        (std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V1") == 0 ||
         std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V2") == 0);
    if (!viewport) {
        ok = ok && sscanf_s(header, "%39[^,],%d,%d,%u", magic,
                            static_cast<unsigned>(sizeof(magic)),
                            &map_width, &map_height, &count) == 4 &&
             std::strcmp(magic, "C3X_BIQ_TERRAIN_V0") == 0;
        halo_count = 0;
    }
    ok = ok && count <= 1000000u && halo_count <= 1000000u;
    if (ok) {
        bool has_river_topology = viewport &&
            std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V2") == 0;
        tiles.reserve(count + halo_count);
        for (unsigned index = 0; index < count + halo_count; ++index) {
            CsvTile tile = {};
            if (viewport) {
                int column = 0, row = 0, source_x = 0, source_y = 0;
                int parsed = has_river_topology
                    ? fscanf_s(file, "%d,%d,%d,%d,%d,%d,%u,%u,%u\n",
                               &column, &row, &source_x, &source_y,
                               &tile.base, &tile.real, &tile.bonus,
                               &tile.overlays, &tile.river)
                    : fscanf_s(file, "%d,%d,%d,%d,%d,%d,%u,%u\n",
                               &column, &row, &source_x, &source_y,
                               &tile.base, &tile.real, &tile.bonus,
                               &tile.overlays);
                if (parsed != (has_river_topology ? 9 : 8)) {
                    ok = false;
                    break;
                }
                tile.x = source_x;
                tile.y = source_y;
            } else if (fscanf_s(file, "%d,%d,%d,%d,%u,%u\n",
                                &tile.x, &tile.y, &tile.base, &tile.real,
                                &tile.bonus, &tile.overlays) != 6) {
                ok = false;
                break;
            }
            tile.topology_halo = index >= count;
            tiles.push_back(tile);
        }
        ok = ok && tiles.size() == count + halo_count;
    }
    fclose(file);
    return ok;
}

bool write_bmp(char const * path, c3x_renderer_output_v1 const & output) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "wb") != 0 || file == nullptr)
        return false;
    BITMAPFILEHEADER file_header = {};
    BITMAPINFOHEADER info = {};
    file_header.bfType = 0x4d42;
    file_header.bfOffBits = sizeof(file_header) + sizeof(info);
    file_header.bfSize = file_header.bfOffBits + output.stride_bytes * output.height;
    info.biSize = sizeof(info);
    info.biWidth = output.width;
    info.biHeight = -output.height;
    info.biPlanes = 1;
    info.biBitCount = 32;
    info.biCompression = BI_RGB;
    bool ok = fwrite(&file_header, sizeof(file_header), 1, file) == 1 &&
              fwrite(&info, sizeof(info), 1, file) == 1 &&
              fwrite(output.bgra_pixels, output.stride_bytes * output.height, 1, file) == 1;
    fclose(file);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 11 && argc != 12) {
        std::fprintf(stderr, "usage: biq_preview <dll> <mod-root> <definitions> <scene.csv> <out.bmp> <width> <height> <center-x> <center-y> <tile-width> [hour]\n");
        return 2;
    }
    int target_width = std::atoi(argv[6]);
    int target_height = std::atoi(argv[7]);
    int center_x = std::atoi(argv[8]);
    int center_y = std::atoi(argv[9]);
    int tile_width = std::atoi(argv[10]);
    int tile_height = tile_width / 2;
    if (target_width < 320 || target_height < 200 || tile_width < 32 || tile_width > 256)
        return 2;

    int map_width = 0, map_height = 0;
    std::vector<CsvTile> source_tiles;
    if (!read_scene(argv[4], map_width, map_height, source_tiles)) {
        std::fprintf(stderr, "error: could not read BIQ terrain scene\n");
        return 1;
    }
    HMODULE module = LoadLibraryA(argv[1]);
    if (module == nullptr)
        return 1;
    auto set_definitions = reinterpret_cast<c3x_renderer_set_definition_paths_fn>(
        GetProcAddress(module, "c3x_renderer_set_definition_paths"));
    auto render = reinterpret_cast<c3x_renderer_render_fn>(GetProcAddress(module, "c3x_renderer_render"));
    auto reset = reinterpret_cast<c3x_renderer_reset_fn>(GetProcAddress(module, "c3x_renderer_reset"));
    if (set_definitions == nullptr || render == nullptr || reset == nullptr ||
        set_definitions(argv[2], argv[3], nullptr, nullptr) != C3X_RENDERER_RESULT_OK)
        return 1;

    int center_raw_x = center_x * tile_width / 2;
    int center_raw_y = center_y * tile_height / 2;
    int shift_x = target_width / 2 - tile_width / 2 - center_raw_x;
    int shift_y = target_height / 2 - tile_height / 2 - center_raw_y;
    std::vector<c3x_renderer_tile_v1> tiles;
    for (CsvTile const & source : source_tiles) {
      for (int wrap_copy = -1; wrap_copy <= 1; ++wrap_copy) {
        int render_x = source.x + wrap_copy * map_width;
        int anchor_x = render_x * tile_width / 2 + shift_x;
        int anchor_y = source.y * tile_height / 2 + shift_y;
        if (anchor_x + tile_width < -96 || anchor_x > target_width + 96 ||
            anchor_y + tile_height < -128 || anchor_y > target_height + 128)
            continue;
        c3x_renderer_tile_v1 tile = {};
        tile.tile_x = render_x;
        tile.tile_y = source.y;
        tile.anchor_x = anchor_x;
        tile.anchor_y = anchor_y;
        tile.terrain_type = source.base;
        tile.real_terrain_type = source.real;
        tile.square_parts = source.bonus;
        tile.terrain_overlays = source.overlays;
        tile.river_code = source.river;
        tile.visibility_mask = 1;
        tile.tile_visibility = 1;
        tile.variant_seed = preview_seed(source.x, source.y);
        tile.tile_flags = source.topology_halo
            ? C3X_RENDERER_TILE_TOPOLOGY_HALO : C3X_RENDERER_TILE_RENDER;
        tile.resource_id = tile.resource_class = tile.tile_building_id = -1;
        tile.city_id = tile.city_owner_id = tile.city_size = tile.city_culture_group = tile.city_era = -1;
        tile.unit_type_id = tile.unit_owner_id = tile.unit_class = tile.unit_state = tile.unit_damage = tile.unit_direction = -1;
        tile.territory_owner_id = -1;
        if (source.real == 7) tile.feature_flags = C3X_RENDERER_FEATURE_FOREST;
        if (source.real == 8) tile.feature_flags = C3X_RENDERER_FEATURE_JUNGLE;
        if (source.real == 9) tile.feature_flags = C3X_RENDERER_FEATURE_MARSH;
        if (source.real == 10) tile.feature_flags = C3X_RENDERER_FEATURE_VOLCANO;
        tiles.push_back(tile);
      }
    }
    c3x_renderer_frame_v1 frame = {};
    frame.api_version = C3X_RENDERER_API_VERSION;
    frame.struct_size = sizeof(frame);
    frame.target_width = target_width;
    frame.target_height = target_height;
    frame.clip_right = target_width;
    frame.clip_bottom = target_height;
    frame.tile_width = tile_width;
    frame.tile_height = tile_height;
    frame.hour = argc == 12 ? std::atoi(argv[11]) : 12;
    frame.tile_count = static_cast<c3x_renderer_u32>(tiles.size());
    frame.tiles = tiles.data();
    frame.presentation_time_ticks = 1000000;
    frame.presentation_frequency = 1000000;
    frame.dirty_flags = C3X_RENDERER_DIRTY_ALL;
    frame.world_width_tiles = map_width;
    frame.world_height_tiles = map_height;
    frame.world_wrap_x = 1;
    c3x_renderer_output_v1 output = {C3X_RENDERER_API_VERSION, sizeof(output)};
    int result = render(&frame, &output);
    std::size_t expected_rendered = 0;
    for (c3x_renderer_tile_v1 const & tile : tiles)
        if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) != 0)
            ++expected_rendered;
    bool ok = result == C3X_RENDERER_RESULT_OK &&
              output.rendered_tile_count == expected_rendered &&
              output.fallback_tile_count == 0 && write_bmp(argv[5], output);
    std::printf("BIQ %dx%d viewport: %u visible tiles, %u fallback, output=%s\n",
                map_width, map_height, output.rendered_tile_count, output.fallback_tile_count, argv[5]);
    reset();
    FreeLibrary(module);
    return ok ? 0 : 1;
}
