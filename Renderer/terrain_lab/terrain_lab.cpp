#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgiformat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../native/environment_runtime.h"

namespace {

unsigned output_width = 1024;
unsigned output_height = 512;
float scene_world_width = 2.0f;
float scene_world_height = 2.0f;
bool promotion_scene_enabled = false;
bool l10_scene_enabled = false;
bool dune_scene_enabled = false;
bool l11_scene_enabled = false;
bool l12_scene_enabled = false;
bool l13_scene_enabled = false;
bool l13a_scene_enabled = false;
bool l14_scene_enabled = false;
bool l15_scene_enabled = false;
bool l16_scene_enabled = false;
bool l17_scene_enabled = false;
bool biq_scene_enabled = false;
bool volcano_geometry_enabled = false;
bool river_geometry_enabled = false;
bool road_geometry_enabled = false;
bool railroad_geometry_enabled = false;
bool resource_geometry_enabled = false;
bool city_geometry_enabled = false;
c3x_renderer::EnvironmentState frame_environment = {};
float active_light_direction[3] = {-0.808f, -0.514f, 0.323f};
float frame_hour = 12.0f;

template <typename T>
void release(T *& value) {
    if (value != nullptr) {
        value->Release();
        value = nullptr;
    }
}

struct Vertex {
    float x, y;
    float u, v;
    float panel;
    float normal_x, normal_y, normal_z;
    float shadow_visibility, ambient_visibility;
    float macro_u, macro_v;
    float surface_kind;
    float surface_coordinate;
    float base_terrain = 2.0f;
    float real_terrain = 2.0f;
    float material_grass = 0.0f;
    float material_plains = 0.0f;
    float material_desert = 0.0f;
    float material_marsh = 0.0f;
    float terrain_depth = 0.5f;
    float authored_relief_height = 0.0f;
    float authored_relief_blend = 0.0f;
    float shore_distance = 1.0f;
    float river_distance = 1000.0f;
    float river_branch_count = 0.0f;
    float river_mouth_distance = 1000.0f;
    float river_padding = 0.0f;
};

struct BiqWindowTile {
    int column = 0;
    int row = 0;
    int source_x = 0;
    int source_y = 0;
    int base = 2;
    int real = 2;
    unsigned bonus = 0;
    unsigned overlays = 0;
    unsigned river_mask = 0;
};

struct BiqWindow {
    int columns = 0;
    int rows = 0;
    int origin_column = 0;
    int origin_row = 0;
    int map_width = 0;
    int map_height = 0;
    std::vector<BiqWindowTile> tiles;
    std::vector<BiqWindowTile> halo_tiles;
};

BiqWindow biq_window;

struct RoadEdge {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    unsigned wraps = 0;
    unsigned style = 0;
    unsigned pillaged = 0;
    unsigned bridge = 0;
};

struct RoadScenario {
    int columns = 0;
    int rows = 0;
    std::vector<RoadEdge> edges;
};

RoadScenario road_scenario;
RoadScenario railroad_scenario;

struct ResourceInstance {
    int column = 0;
    int row = 0;
    unsigned resource = 0;
    unsigned visible = 0;
    unsigned variant = 0;
};

struct ResourceScenario {
    int columns = 0;
    int rows = 0;
    std::vector<ResourceInstance> instances;
};

ResourceScenario resource_scenario;

struct CityInstance {
    int column = 0;
    int row = 0;
    unsigned era = 0;
    unsigned size = 0;
    unsigned culture = 0;
    unsigned owner = 0;
    unsigned walls = 0;
    unsigned capital = 0;
    unsigned visible = 0;
    unsigned variant = 0;
};

struct CityScenario {
    int columns = 0;
    int rows = 0;
    std::vector<CityInstance> instances;
};

CityScenario city_scenario;

struct MineInstance {
    int column = 0;
    int row = 0;
    unsigned era = 0;
    unsigned variant = 0;
    unsigned visible = 0;
    unsigned resource_owned = 0;
};

struct MineScenario {
    int columns = 0;
    int rows = 0;
    std::vector<MineInstance> instances;
};

MineScenario mine_scenario;

struct RiverNode {
    int lattice_x = 0;
    int lattice_y = 0;
    unsigned degree = 0;
    bool touches_water = false;
};

std::vector<RiverNode> river_nodes;

struct FeatureSourceVertex {
    float position[3];
    float normal[3];
    float uv[2];
};

struct FeatureAsset {
    std::string id;
    unsigned texture_index = 0;
    std::vector<FeatureSourceVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct FeaturePlacement {
    unsigned asset_index = 0;
    float scale = 1.0f;
    float scale_variation = 0.0f;
    unsigned count = 0;
    unsigned min_count = 0;
    unsigned priority = 0;
    unsigned flags = 0;
    float width = 0.0f;
    float low_end_reduction = 0.0f;
};

struct FeatureGroup {
    std::string name;
    std::vector<FeaturePlacement> placements;
};

struct FeatureBundle {
    std::vector<std::string> texture_paths;
    std::vector<FeatureAsset> assets;
    std::vector<FeatureGroup> groups;
};

struct FeatureVertex {
    float x, y, depth;
    float u, v;
    float normal_x, normal_y, normal_z;
    float material_index;
};

struct LabSettings {
    float height_texel[2];
    float normal_strength;
    float exposure;
    float lab_mode;
    float beauty_relief_enabled;
    float beauty_water_enabled;
    float shoreline_integrated;
    float promotion_tile_layout;
    float scene_width;
    float scene_height;
    float dune_enabled;
    float dune_only;
    float l10_layout;
    float biq_layout;
    float marsh_enabled;
    float marsh_only;
    float volcano_enabled;
    float volcano_only;
    float l12_layout;
    float rivers_enabled;
    float rivers_only;
    float l13_layout;
    float l13a_layout;
    float sun_direction[3];
    float sun_intensity;
    float sun_color[3];
    float shadow_strength;
    float moon_direction[3];
    float moon_intensity;
    float moon_color[3];
    float night_activation;
    float ambient_color[3];
    float environment_exposure;
    float water_fresnel;
    float water_specular;
    float emissive_scale;
    float hour;
    float roads_enabled;
    float roads_only;
    float l14_layout;
    float road_style_override;
    float railroads_enabled;
    float railroads_only;
    float l15_layout;
    float railroad_padding;
    float resources_enabled;
    float resources_only;
    float l16_layout;
    float resource_padding;
    float cities_enabled;
    float cities_only;
    float l17_layout;
    float city_padding;
};

struct HeightField {
    unsigned width = 0;
    unsigned height = 0;
    float minimum = 0.0f;
    float maximum = 1.0f;
    float world_uv_scale = 0.085f;
    std::vector<std::uint8_t> pixels;

    float sample(float u, float v) const {
        if (pixels.empty() || width == 0 || height == 0)
            return 0.0f;
        u -= std::floor(u);
        v -= std::floor(v);
        float px = u * static_cast<float>(width);
        float py = v * static_cast<float>(height);
        unsigned x0 = static_cast<unsigned>(std::floor(px)) % width;
        unsigned y0 = static_cast<unsigned>(std::floor(py)) % height;
        unsigned x1 = (x0 + 1) % width;
        unsigned y1 = (y0 + 1) % height;
        float tx = px - std::floor(px);
        float ty = py - std::floor(py);
        auto value = [this](unsigned x, unsigned y) {
            float raw = static_cast<float>(pixels[static_cast<std::size_t>(y) * width + x]) / 255.0f;
            return (raw - minimum) / std::max(0.0001f, maximum - minimum);
        };
        float top = value(x0, y0) * (1.0f - tx) + value(x1, y0) * tx;
        float bottom = value(x0, y1) * (1.0f - tx) + value(x1, y1) * tx;
        return top * (1.0f - ty) + bottom * ty;
    }
};

HeightField const * promotion_hill_height_fields[4] = {};
HeightField const * promotion_mountain_height_fields[5] = {};
HeightField const * promotion_mountain_blend_fields[5] = {};
HeightField const * promotion_volcano_height_field = nullptr;
HeightField const * promotion_volcano_blend_field = nullptr;

struct CoastProjection {
    float origin_x;
    float origin_y;
    float half_width;
    float half_height;
    float vertical_scale;
};

CoastProjection coast_projection = {752.0f, 140.0f, 108.0f, 54.0f, 1.0f};

std::wstring widen(char const * text) {
    int count = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (count <= 0)
        return {};
    std::wstring result(static_cast<std::size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, result.data(), count);
    result.pop_back();
    return result;
}

bool read_file(std::string const & path, std::vector<std::uint8_t> & output) {
    FILE * file = nullptr;
    if (fopen_s(&file, path.c_str(), "rb") != 0 || file == nullptr)
        return false;
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return false;
    }
    long size = std::ftell(file);
    if (size <= 0 || size > 256L * 1024L * 1024L || std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        return false;
    }
    output.resize(static_cast<std::size_t>(size));
    bool ok = std::fread(output.data(), output.size(), 1, file) == 1;
    std::fclose(file);
    if (!ok)
        output.clear();
    return ok;
}

std::uint32_t read_u32(std::vector<std::uint8_t> const & data, std::size_t offset) {
    return static_cast<std::uint32_t>(data[offset]) |
           (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(data[offset + 3]) << 24);
}

bool load_biq_window(char const * path, BiqWindow & output) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || file == nullptr)
        return false;
    char magic[40] = {};
    unsigned count = 0;
    unsigned halo_count = 0;
    bool ok = fscanf_s(file, "%39[^,],%d,%d,%u,%d,%d,%d,%d,%u\n", magic,
                       static_cast<unsigned>(sizeof(magic)), &output.columns, &output.rows,
                       &count, &output.origin_column, &output.origin_row,
                       &output.map_width, &output.map_height, &halo_count) == 9 &&
              (std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V1") == 0 ||
               std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V2") == 0) &&
              output.columns > 0 && output.rows > 0 &&
              output.columns * output.rows > 0 &&
              output.columns * output.rows <= 4096 &&
              count == static_cast<unsigned>(output.columns * output.rows) &&
              halo_count <= static_cast<unsigned>(
                  4 * (output.columns + output.rows) + 16) &&
              output.map_width >= output.columns * 2 && output.map_height >= output.rows;
    if (ok) {
        bool has_river_topology =
            std::strcmp(magic, "C3X_BIQ_TERRAIN_WINDOW_V2") == 0;
        output.tiles.assign(count, BiqWindowTile{});
        for (BiqWindowTile & tile : output.tiles)
            tile.base = -1;
        for (unsigned index = 0; index < count; ++index) {
            BiqWindowTile tile;
            int parsed = has_river_topology
                ? fscanf_s(file, "%d,%d,%d,%d,%d,%d,%u,%u,%u\n",
                           &tile.column, &tile.row, &tile.source_x, &tile.source_y,
                           &tile.base, &tile.real, &tile.bonus, &tile.overlays,
                           &tile.river_mask)
                : fscanf_s(file, "%d,%d,%d,%d,%d,%d,%u,%u\n",
                           &tile.column, &tile.row, &tile.source_x, &tile.source_y,
                           &tile.base, &tile.real, &tile.bonus, &tile.overlays);
            if (parsed != (has_river_topology ? 9 : 8) ||
                tile.column < 0 || tile.column >= output.columns ||
                tile.row < 0 || tile.row >= output.rows ||
                tile.base < 0 || tile.base > 13 || tile.real < 0 || tile.real > 13) {
                ok = false;
                break;
            }
            std::size_t slot = static_cast<std::size_t>(tile.row * output.columns + tile.column);
            if (output.tiles[slot].base >= 0) {
                ok = false;
                break;
            }
            output.tiles[slot] = tile;
        }
        ok = ok && std::all_of(output.tiles.begin(), output.tiles.end(),
                               [](BiqWindowTile const & tile) { return tile.base >= 0; });
        for (unsigned index = 0; ok && index < halo_count; ++index) {
            BiqWindowTile tile;
            int parsed = has_river_topology
                ? fscanf_s(file, "%d,%d,%d,%d,%d,%d,%u,%u,%u\n",
                           &tile.column, &tile.row, &tile.source_x, &tile.source_y,
                           &tile.base, &tile.real, &tile.bonus, &tile.overlays,
                           &tile.river_mask)
                : fscanf_s(file, "%d,%d,%d,%d,%d,%d,%u,%u\n",
                           &tile.column, &tile.row, &tile.source_x, &tile.source_y,
                           &tile.base, &tile.real, &tile.bonus, &tile.overlays);
            if (parsed != (has_river_topology ? 9 : 8) ||
                tile.column < -2 || tile.column >= output.columns + 2 ||
                tile.row < -2 || tile.row >= output.rows + 2 ||
                (tile.column >= 0 && tile.column < output.columns &&
                 tile.row >= 0 && tile.row < output.rows) ||
                tile.base < 0 || tile.base > 13 || tile.real < 0 || tile.real > 13) {
                ok = false;
                break;
            }
            output.halo_tiles.push_back(tile);
        }
    }
    std::fclose(file);
    if (!ok) {
        output = {};
        std::fprintf(stderr, "terrain_lab: invalid BIQ viewport: %s\n", path);
    }
    return ok;
}

bool load_route_scenario(char const * path, char const * expected_magic,
                         unsigned maximum_style, RoadScenario & output) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || file == nullptr)
        return false;
    char magic[40] = {};
    char source_hash[80] = {};
    char provenance[40] = {};
    unsigned count = 0;
    bool ok = fscanf_s(file, "%39[^,],%d,%d,%u,%79[^,],%39[^\n]\n", magic,
                       static_cast<unsigned>(sizeof(magic)), &output.columns,
                       &output.rows, &count, source_hash,
                       static_cast<unsigned>(sizeof(source_hash)), provenance,
                       static_cast<unsigned>(sizeof(provenance))) == 6 &&
              std::strcmp(magic, expected_magic) == 0 &&
              std::strcmp(provenance, "lab_augmentation") == 0 &&
              output.columns > 0 && output.rows > 0 && count > 0 && count <= 4096;
    output.edges.reserve(count);
    for (unsigned index = 0; ok && index < count; ++index) {
        RoadEdge edge;
        ok = fscanf_s(file, "%d,%d,%d,%d,%u,%u,%u,%u\n",
                      &edge.x0, &edge.y0, &edge.x1, &edge.y1,
                      &edge.wraps, &edge.style, &edge.pillaged, &edge.bridge) == 8 &&
             edge.x0 >= 0 && edge.x0 < output.columns &&
             edge.x1 >= 0 && edge.x1 < output.columns &&
             edge.y0 >= 0 && edge.y0 < output.rows &&
             edge.y1 >= 0 && edge.y1 < output.rows &&
             edge.wraps <= 1 && edge.style <= maximum_style && edge.pillaged <= 1 &&
             edge.bridge <= 1 &&
             ((edge.wraps && edge.y0 == edge.y1 &&
               edge.x0 == output.columns - 1 && edge.x1 == 0) ||
              (!edge.wraps && std::abs(edge.x0 - edge.x1) +
               std::abs(edge.y0 - edge.y1) == 1));
        if (ok)
            output.edges.push_back(edge);
    }
    std::fclose(file);
    ok = ok && output.edges.size() == count;
    if (!ok) {
        output = {};
        std::fprintf(stderr, "terrain_lab: invalid Lab-only route scenario: %s\n", path);
    }
    return ok;
}

bool load_road_scenario(char const * path, RoadScenario & output) {
    return load_route_scenario(path, "C3X_LAB_ROAD_SCENARIO_V0", 3u, output);
}

bool load_railroad_scenario(char const * path, RoadScenario & output) {
    return load_route_scenario(path, "C3X_LAB_RAILROAD_SCENARIO_V0", 4u, output);
}

bool load_resource_scenario(char const * path, ResourceScenario & output) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || file == nullptr)
        return false;
    char magic[48] = {};
    char source_hash[80] = {};
    char provenance[40] = {};
    unsigned count = 0;
    bool ok = fscanf_s(file, "%47[^,],%d,%d,%u,%79[^,],%39[^\n]\n", magic,
                       static_cast<unsigned>(sizeof(magic)), &output.columns,
                       &output.rows, &count, source_hash,
                       static_cast<unsigned>(sizeof(source_hash)), provenance,
                       static_cast<unsigned>(sizeof(provenance))) == 6 &&
              std::strcmp(magic, "C3X_LAB_RESOURCE_SCENARIO_V0") == 0 &&
              std::strcmp(provenance, "lab_augmentation") == 0 &&
              output.columns > 0 && output.rows > 0 && count > 0 && count <= 4096;
    output.instances.reserve(count);
    for (unsigned index = 0; ok && index < count; ++index) {
        ResourceInstance instance;
        ok = fscanf_s(file, "%d,%d,%u,%u,%u\n", &instance.column, &instance.row,
                      &instance.resource, &instance.visible, &instance.variant) == 5 &&
             instance.column >= 0 && instance.column < output.columns &&
             instance.row >= 0 && instance.row < output.rows &&
             instance.resource < 8u && instance.visible <= 1u;
        if (ok)
            output.instances.push_back(instance);
    }
    std::fclose(file);
    ok = ok && output.instances.size() == count;
    if (!ok) {
        output = {};
        std::fprintf(stderr, "terrain_lab: invalid Lab-only resource scenario: %s\n", path);
    }
    return ok;
}

bool load_city_scenario(char const * path, CityScenario & output) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || file == nullptr)
        return false;
    char magic[48] = {};
    char source_hash[80] = {};
    char provenance[40] = {};
    unsigned count = 0;
    bool ok = fscanf_s(file, "%47[^,],%d,%d,%u,%79[^,],%39[^\n]\n", magic,
                       static_cast<unsigned>(sizeof(magic)), &output.columns,
                       &output.rows, &count, source_hash,
                       static_cast<unsigned>(sizeof(source_hash)), provenance,
                       static_cast<unsigned>(sizeof(provenance))) == 6 &&
              std::strcmp(magic, "C3X_LAB_CITY_SCENARIO_V0") == 0 &&
              std::strcmp(provenance, "lab_augmentation") == 0 &&
              output.columns > 0 && output.rows > 0 && count > 0 && count <= 128;
    output.instances.reserve(count);
    for (unsigned index = 0; ok && index < count; ++index) {
        CityInstance instance;
        ok = fscanf_s(file, "%d,%d,%u,%u,%u,%u,%u,%u,%u,%u\n",
                      &instance.column, &instance.row, &instance.era, &instance.size,
                      &instance.culture, &instance.owner, &instance.walls,
                      &instance.capital, &instance.visible, &instance.variant) == 10 &&
             instance.column >= 0 && instance.column < output.columns &&
             instance.row >= 0 && instance.row < output.rows &&
             instance.era < 4u && instance.size < 3u && instance.culture < 5u &&
             instance.owner < 4u && instance.walls <= 1u && instance.capital <= 1u &&
             instance.visible <= 1u;
        if (ok)
            output.instances.push_back(instance);
    }
    std::fclose(file);
    ok = ok && output.instances.size() == count;
    if (!ok) {
        output = {};
        std::fprintf(stderr, "terrain_lab: invalid Lab-only city scenario: %s\n", path);
    }
    return ok;
}

bool load_mine_scenario(char const * path, MineScenario & output) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || file == nullptr)
        return false;
    char magic[48] = {};
    char source_hash[80] = {};
    char provenance[40] = {};
    unsigned count = 0;
    bool ok = fscanf_s(file, "%47[^,],%d,%d,%u,%79[^,],%39[^\n]\n", magic,
                       static_cast<unsigned>(sizeof(magic)), &output.columns,
                       &output.rows, &count, source_hash,
                       static_cast<unsigned>(sizeof(source_hash)), provenance,
                       static_cast<unsigned>(sizeof(provenance))) == 6 &&
              std::strcmp(magic, "C3X_LAB_MINE_SCENARIO_V0") == 0 &&
              std::strcmp(provenance, "lab_augmentation") == 0 &&
              output.columns > 0 && output.rows > 0 && count > 0 && count <= 128;
    output.instances.reserve(count);
    for (unsigned index = 0; ok && index < count; ++index) {
        MineInstance instance;
        ok = fscanf_s(file, "%d,%d,%u,%u,%u,%u\n", &instance.column, &instance.row,
                      &instance.era, &instance.variant, &instance.visible,
                      &instance.resource_owned) == 6 &&
             instance.column >= 0 && instance.column < output.columns &&
             instance.row >= 0 && instance.row < output.rows &&
             instance.era < 4u && instance.variant < 3u &&
             instance.visible <= 1u && instance.resource_owned <= 1u;
        if (ok)
            output.instances.push_back(instance);
    }
    std::fclose(file);
    ok = ok && output.instances.size() == count;
    if (!ok) {
        output = {};
        std::fprintf(stderr, "terrain_lab: invalid Lab-only mine scenario: %s\n", path);
    }
    return ok;
}

BiqWindowTile const * biq_tile_at(int column, int row) {
    if (!biq_scene_enabled)
        return nullptr;
    if (column >= 0 && row >= 0 &&
        column < biq_window.columns && row < biq_window.rows)
        return &biq_window.tiles[static_cast<std::size_t>(row * biq_window.columns + column)];
    for (BiqWindowTile const & tile : biq_window.halo_tiles)
        if (tile.column == column && tile.row == row)
            return &tile;
    return nullptr;
}

BiqWindowTile const * biq_tile_at(float world_x, float world_y) {
    return biq_tile_at(static_cast<int>(std::floor(world_x)),
                       static_cast<int>(std::floor(world_y)));
}

RiverNode & river_node_at(int lattice_x, int lattice_y) {
    for (RiverNode & node : river_nodes)
        if (node.lattice_x == lattice_x && node.lattice_y == lattice_y)
            return node;
    river_nodes.push_back(RiverNode{lattice_x, lattice_y, 0u, false});
    return river_nodes.back();
}

void add_river_graph_edge(int start_x, int start_y, int endpoint_x, int endpoint_y) {
    river_node_at(start_x, start_y).degree += 1;
    river_node_at(endpoint_x, endpoint_y).degree += 1;
}

void build_river_graph() {
    river_nodes.clear();
    auto add_tile_edges = [](BiqWindowTile const & tile) {
        int center_x = tile.column + tile.row;
        int center_y = tile.column - tile.row;
        // Enumerate only the northeast and southeast owners. Their southwest
        // and northwest reciprocal flags describe these same physical edges.
        if ((tile.river_mask & 2u) != 0)
            add_river_graph_edge(center_x, center_y - 1,
                                 center_x + 1, center_y);
        if ((tile.river_mask & 8u) != 0)
            add_river_graph_edge(center_x + 1, center_y,
                                 center_x, center_y + 1);
    };
    for (BiqWindowTile const & tile : biq_window.tiles)
        add_tile_edges(tile);
    for (BiqWindowTile const & tile : biq_window.halo_tiles)
        add_tile_edges(tile);

    auto mark_water_corners = [](BiqWindowTile const & tile) {
        if (tile.base < 11)
            return;
        int center_x = tile.column + tile.row;
        int center_y = tile.column - tile.row;
        constexpr int corner_offsets[4][2] = {
            {0, -1}, {1, 0}, {0, 1}, {-1, 0}
        };
        for (auto const & offset : corner_offsets)
            for (RiverNode & node : river_nodes)
                if (node.lattice_x == center_x + offset[0] &&
                    node.lattice_y == center_y + offset[1])
                    node.touches_water = true;
    };
    for (BiqWindowTile const & tile : biq_window.tiles)
        mark_water_corners(tile);
    for (BiqWindowTile const & tile : biq_window.halo_tiles)
        mark_water_corners(tile);
}

float biq_river_node_distance(BiqWindowTile const & tile, float u, float v,
                              unsigned node_kind) {
    float point_x = static_cast<float>(tile.column + tile.row) + u - v;
    float point_y = static_cast<float>(tile.column - tile.row) + u + v - 1.0f;
    float distance = 1000.0f;
    for (RiverNode const & node : river_nodes) {
        bool selected = node_kind == 0u
            ? node.degree == 1u && !node.touches_water
            : (node_kind == 1u ? node.degree >= 3u
                               : node.degree == 1u && node.touches_water);
        if (!selected)
            continue;
        float delta_x = (point_x - static_cast<float>(node.lattice_x)) *
                        coast_projection.half_width;
        float delta_y = (point_y - static_cast<float>(node.lattice_y)) *
                        coast_projection.half_height;
        distance = std::min(distance, std::sqrt(delta_x * delta_x +
                                                delta_y * delta_y));
    }
    return distance;
}

bool consume_u32(std::vector<std::uint8_t> const & data, std::size_t & cursor,
                 std::uint32_t & output) {
    if (cursor + 4 > data.size())
        return false;
    output = read_u32(data, cursor);
    cursor += 4;
    return true;
}

bool consume_float(std::vector<std::uint8_t> const & data, std::size_t & cursor, float & output) {
    std::uint32_t bits = 0;
    if (!consume_u32(data, cursor, bits))
        return false;
    std::memcpy(&output, &bits, sizeof(output));
    return std::isfinite(output);
}

bool consume_string(std::vector<std::uint8_t> const & data, std::size_t & cursor,
                    std::string & output) {
    std::uint32_t bytes = 0;
    if (!consume_u32(data, cursor, bytes) || bytes == 0 || bytes > 4096 || cursor + bytes > data.size())
        return false;
    output.assign(reinterpret_cast<char const *>(data.data() + cursor), bytes);
    cursor += bytes;
    return true;
}

bool load_feature_bundle(std::string const & path, FeatureBundle & output) {
    std::vector<std::uint8_t> data;
    if (!read_file(path, data) || data.size() < 24 ||
        std::memcmp(data.data(), "C3XVEG1\0", 8) != 0) {
        std::fprintf(stderr, "terrain_lab: invalid vegetation runtime bundle: %s\n", path.c_str());
        return false;
    }
    std::size_t cursor = 8;
    std::uint32_t version = 0, texture_count = 0, asset_count = 0, group_count = 0;
    if (!consume_u32(data, cursor, version) || !consume_u32(data, cursor, texture_count) ||
        !consume_u32(data, cursor, asset_count) || !consume_u32(data, cursor, group_count) ||
        version != 1 || texture_count == 0 || texture_count > 8 || asset_count == 0 ||
        asset_count > 256 || group_count == 0 || group_count > 16) {
        std::fprintf(stderr, "terrain_lab: unsupported vegetation runtime bundle header\n");
        return false;
    }
    output.texture_paths.resize(texture_count);
    for (std::string & texture_path : output.texture_paths)
        if (!consume_string(data, cursor, texture_path))
            return false;
    output.assets.resize(asset_count);
    for (FeatureAsset & asset : output.assets) {
        std::uint32_t vertex_count = 0, index_count = 0;
        if (!consume_string(data, cursor, asset.id) ||
            !consume_u32(data, cursor, asset.texture_index) ||
            !consume_u32(data, cursor, vertex_count) ||
            !consume_u32(data, cursor, index_count) ||
            asset.texture_index >= texture_count || vertex_count < 3 || vertex_count > 100000 ||
            index_count < 3 || index_count > 300000 || index_count % 3 != 0)
            return false;
        asset.vertices.resize(vertex_count);
        for (FeatureSourceVertex & vertex : asset.vertices) {
            for (float & component : vertex.position)
                if (!consume_float(data, cursor, component)) return false;
            for (float & component : vertex.normal)
                if (!consume_float(data, cursor, component)) return false;
            for (float & component : vertex.uv)
                if (!consume_float(data, cursor, component)) return false;
        }
        asset.indices.resize(index_count);
        for (std::uint32_t & index : asset.indices)
            if (!consume_u32(data, cursor, index) || index >= vertex_count)
                return false;
    }
    output.groups.resize(group_count);
    for (FeatureGroup & group : output.groups) {
        std::uint32_t placement_count = 0;
        if (!consume_string(data, cursor, group.name) ||
            !consume_u32(data, cursor, placement_count) || placement_count > 1024)
            return false;
        group.placements.resize(placement_count);
        for (FeaturePlacement & placement : group.placements) {
            if (!consume_u32(data, cursor, placement.asset_index) ||
                !consume_float(data, cursor, placement.scale) ||
                !consume_float(data, cursor, placement.scale_variation) ||
                !consume_u32(data, cursor, placement.count) ||
                !consume_u32(data, cursor, placement.min_count) ||
                !consume_u32(data, cursor, placement.priority) ||
                !consume_u32(data, cursor, placement.flags) ||
                !consume_float(data, cursor, placement.width) ||
                !consume_float(data, cursor, placement.low_end_reduction) ||
                placement.asset_index >= asset_count || placement.scale < 0.0f ||
                placement.scale_variation < 0.0f || placement.scale_variation > 2.0f)
                return false;
        }
    }
    if (cursor != data.size()) {
        std::fprintf(stderr, "terrain_lab: vegetation runtime bundle has trailing data\n");
        return false;
    }
    return true;
}

bool load_dds(ID3D11Device * device, std::string const & path, DXGI_FORMAT expected,
              ID3D11ShaderResourceView ** output, unsigned & width, unsigned & height) {
    std::vector<std::uint8_t> dds;
    if (!read_file(path, dds) || dds.size() < 156 || std::memcmp(dds.data(), "DDS ", 4) != 0 ||
        std::memcmp(dds.data() + 84, "DX10", 4) != 0) {
        std::fprintf(stderr, "terrain_lab: invalid DX10 DDS: %s\n", path.c_str());
        return false;
    }
    width = read_u32(dds, 16);
    height = read_u32(dds, 12);
    unsigned mip_count = std::max(1u, read_u32(dds, 28));
    DXGI_FORMAT format = static_cast<DXGI_FORMAT>(read_u32(dds, 128));
    bool format_matches = format == expected ||
        (expected == DXGI_FORMAT_BC3_UNORM_SRGB && format == DXGI_FORMAT_BC3_UNORM) ||
        (expected == DXGI_FORMAT_BC1_UNORM_SRGB && format == DXGI_FORMAT_BC1_UNORM);
    if (!format_matches || width == 0 || height == 0 || width > 16384 || height > 16384 || mip_count > 15) {
        std::fprintf(stderr, "terrain_lab: unsupported DDS dimensions or format: %s\n", path.c_str());
        return false;
    }

    unsigned block_bytes = 0;
    unsigned bytes_per_pixel = 0;
    if (format == DXGI_FORMAT_BC1_UNORM || format == DXGI_FORMAT_BC1_UNORM_SRGB ||
        format == DXGI_FORMAT_BC4_UNORM)
        block_bytes = 8u;
    else if (format == DXGI_FORMAT_BC3_UNORM || format == DXGI_FORMAT_BC3_UNORM_SRGB ||
             format == DXGI_FORMAT_BC5_UNORM)
        block_bytes = 16u;
    else if (format == DXGI_FORMAT_R16G16B16A16_UNORM ||
             format == DXGI_FORMAT_R16G16B16A16_FLOAT)
        bytes_per_pixel = 8u;
    else if (format == DXGI_FORMAT_R16G16_UNORM)
        bytes_per_pixel = 4u;
    else {
        std::fprintf(stderr, "terrain_lab: unsupported DDS layout: %s\n", path.c_str());
        return false;
    }
    std::vector<D3D11_SUBRESOURCE_DATA> subresources(mip_count);
    std::size_t offset = 148;
    unsigned mip_width = width;
    unsigned mip_height = height;
    for (unsigned mip = 0; mip < mip_count; ++mip) {
        unsigned row_pitch = bytes_per_pixel != 0
            ? mip_width * bytes_per_pixel
            : std::max(1u, (mip_width + 3) / 4) * block_bytes;
        unsigned rows = bytes_per_pixel != 0 ? mip_height : std::max(1u, (mip_height + 3) / 4);
        std::size_t bytes = static_cast<std::size_t>(row_pitch) * rows;
        if (offset + bytes > dds.size()) {
            std::fprintf(stderr, "terrain_lab: truncated DDS mip chain: %s\n", path.c_str());
            return false;
        }
        subresources[mip].pSysMem = dds.data() + offset;
        subresources[mip].SysMemPitch = row_pitch;
        subresources[mip].SysMemSlicePitch = static_cast<UINT>(bytes);
        offset += bytes;
        mip_width = std::max(1u, mip_width / 2);
        mip_height = std::max(1u, mip_height / 2);
    }

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = mip_count;
    desc.ArraySize = 1;
    // Always use the sRGB view for base color.  Material channels remain linear.
    desc.Format = expected;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    ID3D11Texture2D * texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&desc, subresources.data(), &texture);
    if (SUCCEEDED(hr))
        hr = device->CreateShaderResourceView(texture, nullptr, output);
    release(texture);
    if (FAILED(hr))
        std::fprintf(stderr, "terrain_lab: Direct3D could not create %s (0x%08lx)\n", path.c_str(), hr);
    return SUCCEEDED(hr);
}

DXGI_FORMAT source_color_dds_format(std::string const & path) {
    std::vector<std::uint8_t> dds;
    if (!read_file(path, dds) || dds.size() < 148 ||
        std::memcmp(dds.data(), "DDS ", 4) != 0 ||
        std::memcmp(dds.data() + 84, "DX10", 4) != 0)
        return DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT format = static_cast<DXGI_FORMAT>(read_u32(dds, 128));
    if (format == DXGI_FORMAT_BC1_UNORM || format == DXGI_FORMAT_BC1_UNORM_SRGB)
        return DXGI_FORMAT_BC1_UNORM_SRGB;
    if (format == DXGI_FORMAT_BC3_UNORM || format == DXGI_FORMAT_BC3_UNORM_SRGB)
        return DXGI_FORMAT_BC3_UNORM_SRGB;
    return DXGI_FORMAT_UNKNOWN;
}

bool load_r8_height(ID3D11Device * device, std::string const & path, DXGI_FORMAT source_format,
                    HeightField & field,
                    ID3D11ShaderResourceView ** output) {
    std::vector<std::uint8_t> dds;
    if (!read_file(path, dds) || dds.size() < 149 || std::memcmp(dds.data(), "DDS ", 4) != 0 ||
        std::memcmp(dds.data() + 84, "DX10", 4) != 0 ||
        static_cast<DXGI_FORMAT>(read_u32(dds, 128)) != source_format) {
        std::fprintf(stderr, "terrain_lab: invalid R8 height DDS: %s\n", path.c_str());
        return false;
    }
    field.width = read_u32(dds, 16);
    field.height = read_u32(dds, 12);
    std::size_t count = static_cast<std::size_t>(field.width) * field.height;
    if (field.width == 0 || field.height == 0 || field.width > 4096 || field.height > 4096 ||
        count > dds.size() - 148) {
        std::fprintf(stderr, "terrain_lab: invalid R8 height dimensions: %s\n", path.c_str());
        return false;
    }
    field.pixels.assign(dds.begin() + 148, dds.begin() + 148 + static_cast<std::ptrdiff_t>(count));
    auto limits = std::minmax_element(field.pixels.begin(), field.pixels.end());
    field.minimum = static_cast<float>(*limits.first) / 255.0f;
    field.maximum = static_cast<float>(*limits.second) / 255.0f;

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = field.width;
    desc.Height = field.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial = {field.pixels.data(), field.width, static_cast<UINT>(count)};
    ID3D11Texture2D * texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&desc, &initial, &texture);
    if (SUCCEEDED(hr))
        hr = device->CreateShaderResourceView(texture, nullptr, output);
    release(texture);
    if (FAILED(hr))
        std::fprintf(stderr, "terrain_lab: Direct3D could not create %s (0x%08lx)\n", path.c_str(), hr);
    return SUCCEEDED(hr);
}

float ndc_x(float pixels) {
    return pixels / static_cast<float>(output_width) * 2.0f - 1.0f;
}

float ndc_y(float pixels) {
    return 1.0f - pixels / static_cast<float>(output_height) * 2.0f;
}

void add_triangle(std::vector<Vertex> & vertices, Vertex const & a, Vertex const & b, Vertex const & c) {
    vertices.push_back(a);
    vertices.push_back(b);
    vertices.push_back(c);
}

void add_source_quad(std::vector<Vertex> & vertices, float left, float top, float right, float bottom,
                     float panel) {
    Vertex top_left = {ndc_x(left), ndc_y(top), 0.0f, 0.0f, panel, 0.0f, 0.0f, 1.0f,
                       1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Vertex top_right = {ndc_x(right), ndc_y(top), 1.0f, 0.0f, panel, 0.0f, 0.0f, 1.0f,
                        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    Vertex bottom_right = {ndc_x(right), ndc_y(bottom), 1.0f, 1.0f, panel, 0.0f, 0.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    Vertex bottom_left = {ndc_x(left), ndc_y(bottom), 0.0f, 1.0f, panel, 0.0f, 0.0f, 1.0f,
                          1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    add_triangle(vertices, top_left, top_right, bottom_right);
    add_triangle(vertices, top_left, bottom_right, bottom_left);
}

void add_source_panel(std::vector<Vertex> & vertices, bool mountain_mode, bool coast_mode) {
    if (coast_mode) {
        add_source_quad(vertices, 32.0f, 32.0f, 480.0f, 248.0f, 0.0f);
        add_source_quad(vertices, 32.0f, 264.0f, 480.0f, 480.0f, -1.0f);
        return;
    }
    if (!mountain_mode) {
        add_source_quad(vertices, 32.0f, 32.0f, 480.0f, 480.0f, 0.0f);
        return;
    }
    add_source_quad(vertices, 32.0f, 32.0f, 248.0f, 248.0f, 0.0f);
    add_source_quad(vertices, 264.0f, 32.0f, 480.0f, 248.0f, -1.0f);
    add_source_quad(vertices, 32.0f, 264.0f, 248.0f, 480.0f, -2.0f);
}

float broad_height(float world_x, float world_y) {
    auto mound = [](float x, float y, float center_x, float center_y, float radius_x, float radius_y) {
        float dx = (x - center_x) / radius_x;
        float dy = (y - center_y) / radius_y;
        float distance = dx * dx + dy * dy;
        if (distance >= 1.0f)
            return 0.0f;
        float falloff = 1.0f - distance;
        return falloff * falloff * (3.0f - 2.0f * falloff);
    };
    float primary = mound(world_x, world_y, 0.82f, 0.86f, 0.88f, 0.66f);
    float secondary = mound(world_x, world_y, 1.58f, 1.26f, 0.62f, 0.48f);
    float shoulder = mound(world_x, world_y, 1.42f, 0.38f, 0.58f, 0.42f);
    return std::clamp(primary * 0.58f + secondary * 0.36f + shoulder * 0.22f, 0.0f, 1.0f);
}

float smoothstep01(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

float sample_hill_macro(HeightField const & field, float u, float v) {
    // The authored hill maps contain both broad landform and fine terrain
    // breakup. A small source-space low-pass keeps the macro elevation while
    // preventing that fine breakup from becoming a cluster of miniature
    // mountains at Civ III's much smaller tile scale.
    constexpr float radius = 0.018f;
    float center = field.sample(u, v) * 4.0f;
    float cardinal = field.sample(u - radius, v) + field.sample(u + radius, v) +
                     field.sample(u, v - radius) + field.sample(u, v + radius);
    float diagonal = field.sample(u - radius, v - radius) +
                     field.sample(u + radius, v - radius) +
                     field.sample(u - radius, v + radius) +
                     field.sample(u + radius, v + radius);
    return (center + cardinal * 2.0f + diagonal) / 16.0f;
}

void biq_mountain_sample(BiqWindowTile const & tile, float local_x, float local_y,
                         float & height, float & blend) {
    height = 0.0f;
    blend = 0.0f;
    if (tile.real != 6)
        return;
    unsigned seed = static_cast<unsigned>(tile.source_x * 73 + tile.source_y * 151);
    unsigned variant = (seed >> 3) % 5u;
    HeightField const * height_field = promotion_mountain_height_fields[variant];
    HeightField const * blend_field = promotion_mountain_blend_fields[variant];
    if (height_field == nullptr || blend_field == nullptr)
        return;

    // Preserve the five authored silhouettes while deterministically changing
    // their orientation. Sampling the central 68% expands the authored
    // shoulders and skirt to Civ III's much wider 2:1 diamond instead of
    // turning the source massif into a narrow spike.
    unsigned transform = seed & 7u;
    if ((transform & 1u) != 0)
        std::swap(local_x, local_y);
    if ((transform & 2u) != 0)
        local_x = 1.0f - local_x;
    if ((transform & 4u) != 0)
        local_y = 1.0f - local_y;
    constexpr int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    bool has_relief_neighbor = false;
    for (auto const & offset : offsets) {
        BiqWindowTile const * neighbor = biq_tile_at(
            tile.column + offset[0], tile.row + offset[1]);
        has_relief_neighbor = has_relief_neighbor ||
            (neighbor != nullptr && (neighbor->real == 6 || neighbor->real == 10));
    }
    // Connected cells enlarge the same authored footprint so neighboring
    // bodies overlap as a massif. Isolated cells retain the accepted L11 fit.
    float footprint_scale = has_relief_neighbor ? 0.50f : 0.68f;
    float source_u = 0.5f + (local_x - 0.5f) * footprint_scale;
    float source_v = 0.5f + (local_y - 0.5f) * footprint_scale;
    if (source_u < 0.0f || source_u > 1.0f ||
        source_v < 0.0f || source_v > 1.0f)
        return;
    float source_edge = std::min(std::min(source_u, 1.0f - source_u),
                                 std::min(source_v, 1.0f - source_v));
    float edge = smoothstep01(source_edge / 0.055f);
    height = height_field->sample(source_u, source_v);
    blend = blend_field->sample(source_u, source_v) * edge;
}

void biq_volcano_sample(BiqWindowTile const & tile, float local_x, float local_y,
                        float & height, float & blend) {
    height = 0.0f;
    blend = 0.0f;
    if (!volcano_geometry_enabled || tile.real != 10 ||
        promotion_volcano_height_field == nullptr ||
        promotion_volcano_blend_field == nullptr)
        return;

    unsigned seed = static_cast<unsigned>(tile.source_x) * 73856093u ^
                    static_cast<unsigned>(tile.source_y) * 19349663u;
    seed ^= seed >> 13;
    if ((seed & 1u) != 0)
        std::swap(local_x, local_y);
    if ((seed & 2u) != 0)
        local_x = 1.0f - local_x;
    // Preserve the complete normalized terrain element. Only a rigid
    // orientation and the Civ III projection/height calibration are applied;
    // no synthetic cone, crater, smoke, or effect geometry is introduced.
    constexpr int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    bool has_relief_neighbor = false;
    for (auto const & offset : offsets) {
        BiqWindowTile const * neighbor = biq_tile_at(
            tile.column + offset[0], tile.row + offset[1]);
        has_relief_neighbor = has_relief_neighbor ||
            (neighbor != nullptr && (neighbor->real == 6 || neighbor->real == 10));
    }
    float footprint_scale = has_relief_neighbor ? 0.60f : 1.0f;
    float aspect = (seed & 4u) != 0 ? 0.88f : 1.12f;
    float source_u = 0.5f + (local_x - 0.5f) * footprint_scale * aspect;
    float source_v = 0.5f + (local_y - 0.5f) * footprint_scale / aspect;
    if (source_u < 0.0f || source_u > 1.0f ||
        source_v < 0.0f || source_v > 1.0f)
        return;
    float source_edge = std::min(std::min(source_u, 1.0f - source_u),
                                 std::min(source_v, 1.0f - source_v));
    float edge = smoothstep01(source_edge / 0.055f);
    height = promotion_volcano_height_field->sample(source_u, source_v);
    blend = promotion_volcano_blend_field->sample(source_u, source_v) * edge;
}

void biq_chain_relief_sample(float world_x, float world_y, float & height,
                             float & blend, float & displacement,
                             bool include_volcano = true) {
    height = 0.0f;
    blend = 0.0f;
    displacement = 0.0f;
    int center_x = static_cast<int>(std::floor(world_x));
    int center_y = static_cast<int>(std::floor(world_y));
    constexpr int candidates[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (auto const & offset : candidates) {
        BiqWindowTile const * candidate = biq_tile_at(
            center_x + offset[0], center_y + offset[1]);
        if (candidate == nullptr || (candidate->real != 6 && candidate->real != 10) ||
            (!include_volcano && candidate->real == 10))
            continue;
        float local_x = world_x - static_cast<float>(candidate->column);
        float local_y = 1.0f - (world_y - static_cast<float>(candidate->row));
        float candidate_height = 0.0f;
        float candidate_blend = 0.0f;
        if (candidate->real == 6)
            biq_mountain_sample(*candidate, local_x, local_y,
                                candidate_height, candidate_blend);
        else
            biq_volcano_sample(*candidate, local_x, local_y,
                               candidate_height, candidate_blend);
        float candidate_displacement = candidate_height *
            smoothstep01(candidate_blend / 0.34f) *
            (candidate->real == 6 ? 104.0f :
             ((((static_cast<unsigned>(candidate->source_x) * 73856093u ^
                 static_cast<unsigned>(candidate->source_y) * 19349663u) >> 3) & 1u)
                  ? 104.0f : 88.0f));
        if (candidate_displacement > displacement) {
            height = candidate_height;
            blend = candidate_blend;
            displacement = candidate_displacement;
        }
    }
}

float promotion_hill_value(float world_x, float world_y) {
    if (!promotion_scene_enabled || promotion_hill_height_fields[0] == nullptr)
        return 0.0f;
    int tile_x = static_cast<int>(std::floor(world_x));
    int tile_y = static_cast<int>(std::floor(world_y));
    if (biq_scene_enabled) {
        BiqWindowTile const * tile = biq_tile_at(tile_x, tile_y);
        if (tile == nullptr)
            return 0.0f;
        // These BIQ hills all sit on grassland, so the confirmed standard hill
        // field is authoritative. The continental/plain/snow fields remain
        // loaded for their future biome fixtures; they are not interchangeable
        // shape variants when their source selection rule is unresolved. Keep
        // this field in world space so neighboring hill cells share one broad
        // rolling surface. A separate continuous support field owns the BIQ
        // topology transition and lets an authored skirt cross onto adjacent
        // land instead of stopping at the diamond edge.
        HeightField const * field = promotion_hill_height_fields[0];
        float authored_macro = sample_hill_macro(
            *field, 0.11f + world_x * 0.035f, 0.17f + world_y * 0.035f);
        // Let the authored scalar's broad irregular contours define the hill
        // body. The previous unremapped positive field elevated nearly the
        // entire ownership diamond and left the BIQ mask as its silhouette.
        return smoothstep01((authored_macro - 0.22f) / 0.38f);
    }
    bool hill_cell = l10_scene_enabled
        ? ((tile_x == 2 && tile_y == 1) ||
           (tile_x == 2 && tile_y == 2) ||
           (tile_x == 3 && tile_y == 2) ||
           (tile_x == 4 && tile_y == 2) ||
           (tile_x == 5 && tile_y == 1) ||
           (tile_x == 5 && tile_y == 2) ||
           (tile_x == 6 && tile_y == 1) ||
           (tile_x == 6 && tile_y == 2) ||
           (tile_x == 7 && tile_y == 3) ||
           (tile_x == 8 && tile_y == 3) ||
           (tile_x == 8 && tile_y == 4) ||
           (tile_x == 9 && tile_y == 5))
        : ((tile_x == 3 && tile_y == 2) ||
           (tile_x == 4 && tile_y == 2) ||
           (tile_x == 5 && tile_y == 1) ||
           (tile_x == 5 && tile_y == 2) ||
           (tile_x == 2 && tile_y == 3) ||
           (tile_x == 3 && tile_y == 4) ||
           (tile_x == 4 && tile_y == 3) ||
           (tile_x == 5 && tile_y == 4));
    if (!hill_cell)
        return 0.0f;
    float local_x = world_x - static_cast<float>(tile_x);
    float local_y = world_y - static_cast<float>(tile_y);
    if (((tile_x + tile_y) & 1) != 0)
        local_x = 1.0f - local_x;
    float edge_distance = std::min(std::min(local_x, 1.0f - local_x),
                                   std::min(local_y, 1.0f - local_y));
    float edge = std::clamp(edge_distance / 0.14f, 0.0f, 1.0f);
    edge = edge * edge * (3.0f - 2.0f * edge);
    return promotion_hill_height_fields[0]->sample(0.11f + local_x * 0.17f,
                                                  0.17f + local_y * 0.17f) * edge;
}

float dune_region_weight(float world_x, float world_y) {
    if (!dune_scene_enabled)
        return 0.0f;
    if (biq_scene_enabled) {
        BiqWindowTile const * tile = biq_tile_at(world_x, world_y);
        return tile != nullptr && tile->base == 0 && tile->real == 0 ? 1.0f : 0.0f;
    }
    // The L10 dune field occupies one connected 4x4 desert block. Its outer
    // envelope is softened, but the directional height function below stays
    // in world space and never restarts at a Civ III cell boundary.
    float left = smoothstep01((world_x - 2.0f) / 0.12f);
    float right = smoothstep01((6.0f - world_x) / 0.12f);
    float top = smoothstep01((world_y - 4.0f) / 0.12f);
    float bottom = smoothstep01((8.0f - world_y) / 0.12f);
    return left * right * top * bottom;
}

float dune_height_value(float world_x, float world_y) {
    float region = dune_region_weight(world_x, world_y);
    if (region <= 0.0f)
        return 0.0f;
    // Civ VI's source controls are DuneHeight=4, DuneWidth=4,
    // DuneNoise=0.6, and DuneAngle=0.300001. Their exact engine formula is not
    // exposed, so retain those authored relationships while calibrating the
    // final amplitude to the much smaller Civ III projection.
    constexpr float angle = 0.300001f;
    constexpr float dune_width = 4.0f;
    constexpr float dune_noise = 0.6f;
    float along = world_x * std::cos(angle) + world_y * std::sin(angle);
    float across = -world_x * std::sin(angle) + world_y * std::cos(angle);
    float broad_bend = std::sin(across * 1.05f + 0.8f) * dune_noise * 3.65f +
                       std::sin(across * 2.35f - 0.6f) * dune_noise * 0.90f;
    float phase_noise = broad_bend +
                        std::sin(across * 6.7f + along * 0.43f) * dune_noise * 0.32f +
                        std::sin(across * 11.9f - along * 0.31f + 1.7f) * dune_noise * 0.13f;
    float wave = 0.5f + 0.5f * std::sin(along * 6.28318530718f /
                                        (dune_width * 0.24f) + phase_noise);
    float windward = smoothstep01(wave);
    float crest = windward * windward * (1.18f - 0.18f * windward);
    float fine_wave = 0.5f + 0.5f * std::sin(along * 13.1f + across * 1.4f + 0.9f);
    return region * (crest * 17.0f + fine_wave * 1.6f);
}

float relief_value(float world_x, float world_y, HeightField const * authored_height,
                   HeightField const * authored_blend) {
    if (authored_height == nullptr)
        return broad_height(world_x, world_y);
    if (authored_blend != nullptr) {
        if (promotion_scene_enabled) {
            if (biq_scene_enabled) {
                int tile_x = static_cast<int>(std::floor(world_x));
                int tile_y = static_cast<int>(std::floor(world_y));
                BiqWindowTile const * tile = biq_tile_at(tile_x, tile_y);
                if (tile == nullptr || tile->real != 6)
                    return 0.0f;
                float local_x = world_x - static_cast<float>(tile_x);
                float local_y = world_y - static_cast<float>(tile_y);
                float height = 0.0f;
                float blend = 0.0f;
                biq_mountain_sample(*tile, local_x, local_y, height, blend);
                return height * blend;
            }
            // One authored mountain occupies exactly the upper-middle Civ III
            // cell. The source blend already tapers its footprint; the extra
            // edge envelope guarantees no relief leaks into adjacent biomes.
            bool lower_mountain = world_x >= 3.0f && world_x <= 4.0f &&
                                  world_y >= 1.0f && world_y <= 2.0f;
            bool upper_mountain = world_x >= 4.0f && world_x <= 5.0f &&
                                  world_y >= 0.0f && world_y <= 1.0f;
            bool desert_mountain = l10_scene_enabled &&
                                   world_x >= 8.0f && world_x <= 9.0f &&
                                   world_y >= 6.0f && world_y <= 7.0f;
            if (!lower_mountain && !upper_mountain && !desert_mountain)
                return 0.0f;
            float local_x = lower_mountain ? world_x - 3.0f :
                            (upper_mountain ? world_x - 4.0f : world_x - 8.0f);
            float local_y = lower_mountain ? world_y - 1.0f :
                            (upper_mountain ? world_y : world_y - 6.0f);
            if (upper_mountain)
                local_x = 1.0f - local_x;
            if (desert_mountain)
                local_y = 1.0f - local_y;
            if (local_x < 0.0f || local_x > 1.0f || local_y < 0.0f || local_y > 1.0f)
                return 0.0f;
            float edge_distance = std::min(std::min(local_x, 1.0f - local_x),
                                           std::min(local_y, 1.0f - local_y));
            float edge = std::clamp(edge_distance / 0.10f, 0.0f, 1.0f);
            edge = edge * edge * (3.0f - 2.0f * edge);
            float u = 0.25f + local_x * 0.50f;
            float v = 0.25f + local_y * 0.50f;
            return authored_height->sample(u, v) * authored_blend->sample(u, v) * edge;
        }
        float u = world_x * 0.5f;
        float v = world_y * 0.5f;
        return authored_height->sample(u, v) * authored_blend->sample(u, v);
    }
    // Use one continuous authored crop across the complete 2x2 patch. This is
    // intentionally world-space sampling, never a per-tile restart.
    return authored_height->sample(0.11f + world_x * authored_height->world_uv_scale,
                                   0.17f + world_y * authored_height->world_uv_scale);
}

float macro_height(float world_x, float world_y, float relief_height,
                   HeightField const * authored_height, HeightField const * authored_blend) {
    constexpr float world_scale_pixels = 216.0f;
    return relief_value(world_x, world_y, authored_height, authored_blend) * relief_height / world_scale_pixels;
}

void shape_visibility(float world_x, float world_y, float relief_height,
                      HeightField const * authored_height, HeightField const * authored_blend,
                      float & shadow_visibility, float & ambient_visibility) {
    if (relief_height <= 0.0f) {
        shadow_visibility = 1.0f;
        ambient_visibility = 1.0f;
        return;
    }

    constexpr float sun_x = -0.55f;
    constexpr float sun_y = -0.35f;
    constexpr float sun_z = 0.22f;
    float horizontal_length = std::sqrt(sun_x * sun_x + sun_y * sun_y);
    float direction_x = sun_x / horizontal_length;
    float direction_y = sun_y / horizontal_length;
    float ray_slope = sun_z / horizontal_length;
    float origin_height = macro_height(world_x, world_y, relief_height, authored_height, authored_blend);
    float greatest_obstruction = 0.0f;
    for (int step = 1; step <= 48; ++step) {
        float distance = static_cast<float>(step) * 0.035f;
        float sample_x = world_x + direction_x * distance;
        float sample_y = world_y + direction_y * distance;
        if (sample_x < 0.0f || sample_y < 0.0f ||
            sample_x > scene_world_width || sample_y > scene_world_height)
            break;
        float ray_height = origin_height + ray_slope * distance + 0.006f;
        greatest_obstruction = std::max(greatest_obstruction,
            macro_height(sample_x, sample_y, relief_height, authored_height, authored_blend) - ray_height);
    }
    shadow_visibility = 1.0f - std::clamp(greatest_obstruction * 16.0f, 0.0f, 0.55f);

    constexpr float directions[][2] = {
        {1.0f, 0.0f}, {0.7071f, 0.7071f}, {0.0f, 1.0f}, {-0.7071f, 0.7071f},
        {-1.0f, 0.0f}, {-0.7071f, -0.7071f}, {0.0f, -1.0f}, {0.7071f, -0.7071f},
    };
    constexpr float radii[] = {0.08f, 0.18f, 0.34f};
    float horizon_sum = 0.0f;
    int horizon_samples = 0;
    for (auto const & direction : directions) {
        float direction_horizon = 0.0f;
        for (float radius : radii) {
            float sample_x = world_x + direction[0] * radius;
            float sample_y = world_y + direction[1] * radius;
            if (sample_x < 0.0f || sample_y < 0.0f ||
                sample_x > scene_world_width || sample_y > scene_world_height)
                continue;
            float rise = macro_height(sample_x, sample_y, relief_height, authored_height, authored_blend) - origin_height;
            direction_horizon = std::max(direction_horizon, rise / radius);
        }
        horizon_sum += std::clamp(direction_horizon, 0.0f, 0.7f);
        ++horizon_samples;
    }
    float horizon = horizon_samples > 0 ? horizon_sum / static_cast<float>(horizon_samples) : 0.0f;
    ambient_visibility = 1.0f - std::clamp(horizon * 0.52f, 0.0f, 0.30f);
}

Vertex make_patch_vertex(float world_x, float world_y, float uv_scale, float relief_height,
                         bool shadows_enabled, HeightField const * authored_height,
                         HeightField const * authored_blend) {
    constexpr float origin_x = 752.0f;
    constexpr float origin_y = 140.0f;
    constexpr float half_width = 108.0f;
    constexpr float half_height = 54.0f;
    float height = relief_height > 0.0f ? relief_value(world_x, world_y, authored_height, authored_blend) : 0.0f;
    float screen_x = origin_x + (world_x - world_y) * half_width;
    float screen_y = origin_y + (world_x + world_y) * half_height - height * relief_height;

    constexpr float normal_step = 0.005f;
    float slope_x = relief_height > 0.0f
        ? (relief_value(world_x + normal_step, world_y, authored_height, authored_blend) -
           relief_value(world_x - normal_step, world_y, authored_height, authored_blend)) /
          (2.0f * normal_step) * relief_height / (half_width * 2.0f)
        : 0.0f;
    float slope_y = relief_height > 0.0f
        ? (relief_value(world_x, world_y + normal_step, authored_height, authored_blend) -
           relief_value(world_x, world_y - normal_step, authored_height, authored_blend)) /
          (2.0f * normal_step) * relief_height / (half_width * 2.0f)
        : 0.0f;
    float normal_length = std::sqrt(slope_x * slope_x + slope_y * slope_y + 1.0f);
    float shadow = 1.0f;
    float ambient = 1.0f;
    if (shadows_enabled)
        shape_visibility(world_x, world_y, relief_height, authored_height, authored_blend, shadow, ambient);
    return Vertex{ndc_x(screen_x), ndc_y(screen_y), world_x * uv_scale, world_y * uv_scale, 1.0f,
                  -slope_x / normal_length, -slope_y / normal_length, 1.0f / normal_length,
                  shadow, ambient, world_x * 0.5f, world_y * 0.5f, 1.0f, 0.0f};
}

void add_patch_grid(std::vector<Vertex> & vertices, float uv_scale, float relief_height,
                    bool shadows_enabled, HeightField const * authored_height,
                    HeightField const * authored_blend) {
    constexpr int subdivisions_per_tile = 32;
    constexpr int total_subdivisions = subdivisions_per_tile * 2;
    for (int y = 0; y < total_subdivisions; ++y) {
        for (int x = 0; x < total_subdivisions; ++x) {
            float x0 = static_cast<float>(x) / subdivisions_per_tile;
            float y0 = static_cast<float>(y) / subdivisions_per_tile;
            float x1 = static_cast<float>(x + 1) / subdivisions_per_tile;
            float y1 = static_cast<float>(y + 1) / subdivisions_per_tile;
            Vertex top = make_patch_vertex(x0, y0, uv_scale, relief_height, shadows_enabled, authored_height, authored_blend);
            Vertex right = make_patch_vertex(x1, y0, uv_scale, relief_height, shadows_enabled, authored_height, authored_blend);
            Vertex bottom = make_patch_vertex(x1, y1, uv_scale, relief_height, shadows_enabled, authored_height, authored_blend);
            Vertex left = make_patch_vertex(x0, y1, uv_scale, relief_height, shadows_enabled, authored_height, authored_blend);
            add_triangle(vertices, top, right, bottom);
            add_triangle(vertices, top, bottom, left);
        }
    }
}

float coast_position(float world_y, bool corner, bool integrated_shore) {
    if (!corner) {
        if (integrated_shore) {
            float centered = world_y - 1.0f;
            float cove = 0.28f * std::exp(-(centered * centered) / 0.24f);
            float upper = world_y - 0.16f;
            float upper_headland = 0.11f * std::exp(-(upper * upper) / 0.055f);
            float lower = world_y - 1.78f;
            float lower_point = 0.07f * std::exp(-(lower * lower) / 0.08f);
            float edge_roughness = std::sin(world_y * 15.7f + 0.4f) * 0.017f +
                                   std::sin(world_y * 37.1f + 1.8f) * 0.008f +
                                   std::sin(world_y * 83.3f + 0.9f) * 0.003f;
            float coast = 1.14f - cove + upper_headland + lower_point + centered * 0.025f +
                          edge_roughness;
            if (promotion_scene_enabled) {
                // Promotion shots use the same shoreline system over their
                // enlarged Civ III grids, with several bays and asymmetric
                // headlands across the complete coastal column.
                float map_centered = world_y - scene_world_height * 0.5f;
                float bay0 = world_y - 0.70f;
                float bay1 = world_y - 2.05f;
                float bay2 = world_y - 3.35f;
                float bay3 = world_y - 4.72f;
                float bay4 = world_y - 5.62f;
                float bay5 = world_y - 6.72f;
                float bay6 = world_y - 7.58f;
                float point0 = world_y - 0.12f;
                float point1 = world_y - 2.78f;
                float point2 = world_y - 5.25f;
                float point3 = world_y - 6.28f;
                float point4 = world_y - 7.22f;
                if (l10_scene_enabled) {
                    // Keep the organic bays and points, but center their
                    // complete excursion on the x=10 ownership line in the
                    // 12x8 fixture. The ten western columns read as land and
                    // the two eastern columns read as naval water.
                    float medium_roughness =
                        std::sin(world_y * 2.60f + 0.55f) * 0.045f +
                        std::sin(world_y * 5.70f + 1.90f) * 0.025f;
                    float coast_center = scene_world_width - 1.82f;
                    coast = coast_center -
                            0.15f * std::exp(-(bay0 * bay0) / 0.16f) -
                            0.20f * std::exp(-(bay1 * bay1) / 0.24f) -
                            0.13f * std::exp(-(bay2 * bay2) / 0.15f) -
                            0.17f * std::exp(-(bay3 * bay3) / 0.20f) -
                            0.11f * std::exp(-(bay4 * bay4) / 0.12f) -
                            0.16f * std::exp(-(bay5 * bay5) / 0.18f) -
                            0.10f * std::exp(-(bay6 * bay6) / 0.11f) +
                            0.09f * std::exp(-(point0 * point0) / 0.055f) +
                            0.11f * std::exp(-(point1 * point1) / 0.075f) +
                            0.08f * std::exp(-(point2 * point2) / 0.060f) +
                            0.10f * std::exp(-(point3 * point3) / 0.070f) +
                            0.07f * std::exp(-(point4 * point4) / 0.055f) +
                            map_centered * 0.014f + medium_roughness + edge_roughness;
                } else {
                    coast = 6.62f -
                            0.20f * std::exp(-(bay0 * bay0) / 0.16f) -
                            0.28f * std::exp(-(bay1 * bay1) / 0.24f) -
                            0.18f * std::exp(-(bay2 * bay2) / 0.15f) -
                            0.24f * std::exp(-(bay3 * bay3) / 0.20f) -
                            0.15f * std::exp(-(bay4 * bay4) / 0.12f) +
                            0.12f * std::exp(-(point0 * point0) / 0.055f) +
                            0.15f * std::exp(-(point1 * point1) / 0.075f) +
                            0.11f * std::exp(-(point2 * point2) / 0.060f) +
                            map_centered * 0.022f + edge_roughness;
                }
            }
            return coast;
        }
        float centered = world_y - 1.0f;
        return 1.0f + centered * 0.11f + centered * centered * centered * 0.035f;
    }
    float t = std::clamp((world_y - 0.55f) / 0.90f, 0.0f, 1.0f);
    float smooth = t * t * (3.0f - 2.0f * t);
    return 0.62f + smooth * 0.94f;
}

float beach_width_at(float world_y, bool integrated_shore) {
    if (!integrated_shore)
        return 0.20f;
    float centered = (world_y - 1.08f) / 0.58f;
    float variable_width = 0.010f * std::sin(world_y * 11.9f + 0.7f) +
                           0.006f * std::sin(world_y * 29.3f + 2.1f);
    return std::clamp(0.072f + 0.038f * std::exp(-(centered * centered)) +
                      variable_width, 0.050f, 0.128f);
}

Vertex make_coast_vertex(float world_x, float world_y, float height_pixels, float uv_scale,
                         float surface_kind, float normal_x, float normal_y, float normal_z) {
    float screen_x = coast_projection.origin_x +
                     (world_x - world_y) * coast_projection.half_width;
    float screen_y = coast_projection.origin_y +
                     (world_x + world_y) * coast_projection.half_height -
                     height_pixels * coast_projection.vertical_scale;
    float length = std::sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
    return Vertex{ndc_x(screen_x), ndc_y(screen_y), world_x * uv_scale, world_y * uv_scale, 1.0f,
                  normal_x / length, normal_y / length, normal_z / length, 1.0f, 1.0f,
                  world_x * 0.5f, world_y * 0.5f, surface_kind, 0.0f};
}

void add_coast_strip(std::vector<Vertex> & vertices, float y0, float y1,
                     float left0, float left1, float right0, float right1,
                     float height_left0, float height_left1, float height_right0, float height_right1,
                     float uv_scale, float surface_kind, float normal_x, float normal_y, float normal_z,
                     int subdivisions) {
    for (int index = 0; index < subdivisions; ++index) {
        float t0 = static_cast<float>(index) / subdivisions;
        float t1 = static_cast<float>(index + 1) / subdivisions;
        float x00 = left0 + (right0 - left0) * t0;
        float x01 = left0 + (right0 - left0) * t1;
        float x10 = left1 + (right1 - left1) * t0;
        float x11 = left1 + (right1 - left1) * t1;
        float h00 = height_left0 + (height_right0 - height_left0) * t0;
        float h01 = height_left0 + (height_right0 - height_left0) * t1;
        float h10 = height_left1 + (height_right1 - height_left1) * t0;
        float h11 = height_left1 + (height_right1 - height_left1) * t1;
        Vertex top = make_coast_vertex(x00, y0, h00, uv_scale, surface_kind, normal_x, normal_y, normal_z);
        Vertex right = make_coast_vertex(x01, y0, h01, uv_scale, surface_kind, normal_x, normal_y, normal_z);
        Vertex bottom = make_coast_vertex(x11, y1, h11, uv_scale, surface_kind, normal_x, normal_y, normal_z);
        Vertex left = make_coast_vertex(x10, y1, h10, uv_scale, surface_kind, normal_x, normal_y, normal_z);
        top.surface_coordinate = t0;
        right.surface_coordinate = t1;
        bottom.surface_coordinate = t1;
        left.surface_coordinate = t0;
        add_triangle(vertices, top, right, bottom);
        add_triangle(vertices, top, bottom, left);
    }
}

float beauty_land_height(float world_x, float world_y, float land_edge, float base_height,
                         HeightField const * authored_height,
                         HeightField const * authored_blend) {
    if (authored_height == nullptr || authored_blend == nullptr)
        return base_height;
    float shore_taper = biq_scene_enabled ? 1.0f :
        smoothstep01((land_edge - world_x) / 0.34f);
    float rolling_ground = promotion_scene_enabled ? 0.0f : broad_height(world_x, world_y) * 24.0f;
    float mountain_height_pixels = promotion_scene_enabled ? 128.0f : 92.0f;
    float authored_peak = relief_value(world_x, world_y, authored_height, authored_blend) *
                          mountain_height_pixels;
    float authored_hills = promotion_hill_value(world_x, world_y) * 42.0f;
    float dune_height = dune_height_value(world_x, world_y);
    return base_height + (rolling_ground + authored_peak + authored_hills + dune_height) * shore_taper;
}

Vertex make_beauty_land_vertex(float world_x, float world_y, float land_edge, float base_height,
                               float uv_scale,
                               HeightField const * authored_height,
                               HeightField const * authored_blend, bool integrated_shore) {
    float height = beauty_land_height(world_x, world_y, land_edge, base_height,
                                      authored_height, authored_blend);
    constexpr float step = 0.006f;
    float edge_minus = coast_position(world_y - step, false, integrated_shore) -
                       beach_width_at(world_y - step, integrated_shore);
    float edge_plus = coast_position(world_y + step, false, integrated_shore) -
                      beach_width_at(world_y + step, integrated_shore);
    float slope_x = (beauty_land_height(world_x + step, world_y, land_edge, base_height,
                                        authored_height, authored_blend) -
                     beauty_land_height(world_x - step, world_y, land_edge, base_height,
                                        authored_height, authored_blend)) /
                    (2.0f * step * coast_projection.half_width * 2.0f);
    float slope_y = (beauty_land_height(world_x, world_y + step, edge_plus, base_height,
                                        authored_height, authored_blend) -
                     beauty_land_height(world_x, world_y - step, edge_minus, base_height,
                                        authored_height, authored_blend)) /
                    (2.0f * step * coast_projection.half_width * 2.0f);
    Vertex vertex = make_coast_vertex(world_x, world_y, height, uv_scale, 1.0f,
                                      -slope_x, -slope_y, 1.0f);
    shape_visibility(world_x, world_y, 92.0f, authored_height, authored_blend,
                     vertex.shadow_visibility, vertex.ambient_visibility);
    vertex.surface_coordinate = smoothstep01((land_edge - world_x) / 0.34f);
    return vertex;
}

Vertex make_promotion_grid_vertex(float world_x, float world_y,
                                  HeightField const * authored_height,
                                  HeightField const * authored_blend) {
    float land_edge = coast_position(world_y, false, true) - beach_width_at(world_y, true);
    float height = world_x < land_edge
        ? beauty_land_height(world_x, world_y, land_edge, 2.5f,
                             authored_height, authored_blend)
        : 0.0f;
    return make_coast_vertex(world_x, world_y, height + 0.8f, 1.0f, 8.0f,
                             0.0f, 0.0f, 1.0f);
}

void add_promotion_grid(std::vector<Vertex> & vertices,
                        HeightField const * authored_height,
                        HeightField const * authored_blend) {
    constexpr float half_thickness = 0.006f;
    constexpr int segments_per_tile = 24;
    int y_segments = static_cast<int>(scene_world_height * segments_per_tile);
    for (int line = 0; line <= static_cast<int>(scene_world_width); ++line) {
        float world_x = static_cast<float>(line);
        for (int segment = 0; segment < y_segments; ++segment) {
            float y0 = static_cast<float>(segment) / segments_per_tile;
            float y1 = static_cast<float>(segment + 1) / segments_per_tile;
            Vertex top_left = make_promotion_grid_vertex(
                world_x - half_thickness, y0, authored_height, authored_blend);
            Vertex top_right = make_promotion_grid_vertex(
                world_x + half_thickness, y0, authored_height, authored_blend);
            Vertex bottom_right = make_promotion_grid_vertex(
                world_x + half_thickness, y1, authored_height, authored_blend);
            Vertex bottom_left = make_promotion_grid_vertex(
                world_x - half_thickness, y1, authored_height, authored_blend);
            add_triangle(vertices, top_left, top_right, bottom_right);
            add_triangle(vertices, top_left, bottom_right, bottom_left);
        }
    }
    int x_segments = static_cast<int>(scene_world_width * segments_per_tile);
    for (int line = 0; line <= static_cast<int>(scene_world_height); ++line) {
        float world_y = static_cast<float>(line);
        for (int segment = 0; segment < x_segments; ++segment) {
            float x0 = static_cast<float>(segment) / segments_per_tile;
            float x1 = static_cast<float>(segment + 1) / segments_per_tile;
            Vertex top_left = make_promotion_grid_vertex(
                x0, world_y - half_thickness, authored_height, authored_blend);
            Vertex top_right = make_promotion_grid_vertex(
                x1, world_y - half_thickness, authored_height, authored_blend);
            Vertex bottom_right = make_promotion_grid_vertex(
                x1, world_y + half_thickness, authored_height, authored_blend);
            Vertex bottom_left = make_promotion_grid_vertex(
                x0, world_y + half_thickness, authored_height, authored_blend);
            add_triangle(vertices, top_left, top_right, bottom_right);
            add_triangle(vertices, top_left, bottom_right, bottom_left);
        }
    }
}

bool is_water_terrain(int terrain) {
    return terrain >= 11;
}

BiqWindowTile const * find_biq_source_tile(int source_x, int source_y) {
    for (BiqWindowTile const & tile : biq_window.tiles)
        if (tile.source_x == source_x && tile.source_y == source_y)
            return &tile;
    return nullptr;
}

int biq_material_class(BiqWindowTile const & tile) {
    if (tile.real == 9)
        return 3;
    if (tile.base == 0)
        return 2;
    if (tile.base == 1)
        return 1;
    return 0;
}

void biq_center_material_weights(int column, int row, float (&weights)[4]) {
    for (float & weight : weights)
        weight = 0.0f;
    BiqWindowTile const * center = biq_tile_at(column, row);
    if (center != nullptr && !is_water_terrain(center->base)) {
        weights[biq_material_class(*center)] = 1.0f;
        return;
    }

    // A water center still needs a deterministic land material beneath the
    // beach and translucent shallows. Derive it from nearby authoritative
    // land centers instead of defaulting each water diamond independently.
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            BiqWindowTile const * sample = biq_tile_at(column + x, row + y);
            if (sample == nullptr || is_water_terrain(sample->base))
                continue;
            float influence = (x == 0 || y == 0) ? 1.0f : 0.70f;
            weights[biq_material_class(*sample)] += influence;
        }
    }
    float total = weights[0] + weights[1] + weights[2] + weights[3];
    if (total <= 0.0f)
        weights[0] = 1.0f;
    else
        for (float & weight : weights)
            weight /= total;
}

void biq_material_weights(BiqWindowTile const & tile, float u, float v,
                          float (&weights)[4]) {
    float world_x = static_cast<float>(tile.column) + u;
    float world_y = static_cast<float>(tile.row) + (1.0f - v);
    BiqWindowTile const & first = biq_window.tiles.front();
    float source_x = static_cast<float>(first.source_x) + world_x + world_y - 1.0f;
    float source_y = static_cast<float>(first.source_y) + world_x - world_y;
    // Distort the interpolation domain, not either tile's geometry. This keeps
    // both incident sides bit-identical while preventing a grass/plains/desert
    // transition from tracing the hidden diamond edge as a straight line.
    float warp_x = std::sin(source_x * 0.83f + source_y * 1.19f) * 0.10f +
                   std::sin(source_x * 2.31f - source_y * 0.67f) * 0.035f;
    float warp_y = std::sin(source_x * 1.07f - source_y * 0.91f) * 0.10f +
                   std::sin(source_x * 0.59f + source_y * 2.03f) * 0.035f;
    float grid_x = world_x - 0.5f + warp_x;
    float grid_y = world_y - 0.5f + warp_y;
    int x0 = static_cast<int>(std::floor(grid_x));
    int y0 = static_cast<int>(std::floor(grid_y));
    float tx = smoothstep01((grid_x - static_cast<float>(x0) - 0.20f) / 0.60f);
    float ty = smoothstep01((grid_y - static_cast<float>(y0) - 0.20f) / 0.60f);
    float centers[4][4] = {};
    biq_center_material_weights(x0, y0, centers[0]);
    biq_center_material_weights(x0 + 1, y0, centers[1]);
    biq_center_material_weights(x0, y0 + 1, centers[2]);
    biq_center_material_weights(x0 + 1, y0 + 1, centers[3]);
    for (unsigned material = 0; material < 4; ++material) {
        float top = centers[0][material] * (1.0f - tx) + centers[1][material] * tx;
        float bottom = centers[2][material] * (1.0f - tx) + centers[3][material] * tx;
        weights[material] = top * (1.0f - ty) + bottom * ty;
    }
}

float biq_water_family_depth(BiqWindowTile const & tile, float u, float v) {
    // Reconstruct optical depth as one smooth field sampled at BIQ tile
    // centers. Edge-only averaging made every coast/sea cell retain a broad
    // flat interior, so the water still read as translucent diamonds even
    // though values happened to match at their shared boundaries.
    float world_x = static_cast<float>(tile.column) + u;
    // BIQ row adjacency joins this tile's v=0 edge to the next row's v=1
    // edge. Reverse the local v contribution so both incident vertices map
    // to the same continuous surface coordinate.
    float world_y = static_cast<float>(tile.row) + (1.0f - v);
    float grid_x = world_x - 0.5f;
    float grid_y = world_y - 0.5f;
    int x0 = static_cast<int>(std::floor(grid_x));
    int y0 = static_cast<int>(std::floor(grid_y));
    float tx = smoothstep01(grid_x - static_cast<float>(x0));
    float ty = smoothstep01(grid_y - static_cast<float>(y0));
    auto center_depth = [](int column, int row) {
        BiqWindowTile const * sample = biq_tile_at(column, row);
        if (sample == nullptr || !is_water_terrain(sample->base))
            // Shore distance already brings optical depth to zero at the
            // actual contour. Treat neighboring land centers as the coast
            // family here so shallow water is not faded twice into brown bed.
            return 0.34f;
        return std::clamp((sample->base - 10) * 0.34f, 0.18f, 1.0f);
    };
    float top = center_depth(x0, y0) * (1.0f - tx) +
                center_depth(x0 + 1, y0) * tx;
    float bottom = center_depth(x0, y0 + 1) * (1.0f - tx) +
                   center_depth(x0 + 1, y0 + 1) * tx;
    return top * (1.0f - ty) + bottom * ty;
}

float biq_signed_shore_distance(BiqWindowTile const & tile, float u, float v) {
    float world_x = static_cast<float>(tile.column) + u;
    float world_y = static_cast<float>(tile.row) + (1.0f - v);
    float grid_x = world_x - 0.5f;
    float grid_y = world_y - 0.5f;
    int x0 = static_cast<int>(std::floor(grid_x));
    int y0 = static_cast<int>(std::floor(grid_y));
    float tx = smoothstep01(grid_x - static_cast<float>(x0));
    float ty = smoothstep01(grid_y - static_cast<float>(y0));
    auto raw_sign = [](int column, int row) {
        BiqWindowTile const * sample = biq_tile_at(column, row);
        return sample != nullptr && is_water_terrain(sample->base) ? 1.0f : -1.0f;
    };
    auto center_sign = [&raw_sign](int column, int row) {
        return raw_sign(column, row) * 0.62f +
               (raw_sign(column - 1, row) + raw_sign(column + 1, row) +
                raw_sign(column, row - 1) + raw_sign(column, row + 1)) * 0.095f;
    };
    float top = center_sign(x0, y0) * (1.0f - tx) +
                center_sign(x0 + 1, y0) * tx;
    float bottom = center_sign(x0, y0 + 1) * (1.0f - tx) +
                   center_sign(x0 + 1, y0 + 1) * tx;
    float field = top * (1.0f - ty) + bottom * ty;

    // Keep the center of every BIQ cell on its authoritative side of the
    // contour. This preserves clear land/naval ownership even for a lone
    // coast cell surrounded by land, while leaving the shared-edge region
    // free to round and wander.
    float center_dx = u - 0.5f;
    float center_dy = v - 0.5f;
    float center_anchor = 1.0f - smoothstep01(
        std::sqrt(center_dx * center_dx + center_dy * center_dy) / 0.34f);
    float own_sign = is_water_terrain(tile.base) ? 1.0f : -1.0f;
    field = field * (1.0f - center_anchor * 0.88f) +
            own_sign * center_anchor * 0.88f;

    // Perturb the single two-dimensional zero contour in authoritative source
    // coordinates. Unlike four independent edge curves, these frequencies are
    // continuous through corners and therefore form natural coves and points.
    BiqWindowTile const & first = biq_window.tiles.front();
    float source_x = static_cast<float>(first.source_x) + world_x + world_y - 1.0f;
    float source_y = static_cast<float>(first.source_y) + world_x - world_y;
    float contour_noise = std::sin(source_x * 1.37f + source_y * 0.71f) * 0.38f +
                          std::sin(source_x * 0.79f - source_y * 1.11f) * 0.20f +
                          std::sin(source_x * 3.83f - source_y * 2.17f) * 0.13f +
                          std::sin(source_x * 0.53f + source_y * 2.91f) * 0.07f;
    float boundary_weight = 1.0f - smoothstep01(std::abs(field) / 0.92f);
    contour_noise *= 1.0f - center_anchor * 0.90f;
    float result = field + contour_noise * boundary_weight;

    // A lone authoritative coast cell must remain visibly navigable. Add a
    // global, edge-zero round basin for isolated water centers; evaluating it
    // from world coordinates keeps both incident tiles identical at edges.
    int nearby_x = static_cast<int>(std::floor(world_x));
    int nearby_y = static_cast<int>(std::floor(world_y));
    constexpr int cardinal[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int row = nearby_y - 1; row <= nearby_y + 1; ++row) {
        for (int column = nearby_x - 1; column <= nearby_x + 1; ++column) {
            BiqWindowTile const * candidate = biq_tile_at(column, row);
            if (candidate == nullptr || !is_water_terrain(candidate->base))
                continue;
            bool connected_water = false;
            for (auto const & offset : cardinal) {
                BiqWindowTile const * neighbor = biq_tile_at(
                    column + offset[0], row + offset[1]);
                connected_water = connected_water ||
                    (neighbor != nullptr && is_water_terrain(neighbor->base));
            }
            if (connected_water)
                continue;
            float dx = world_x - (static_cast<float>(column) + 0.5f);
            float dy = world_y - (static_cast<float>(row) + 0.5f);
            float basin = 1.0f - smoothstep01(std::sqrt(dx * dx + dy * dy) / 0.46f);
            result = std::max(result, basin);
        }
    }
    return std::clamp(result, -1.0f, 1.0f);
}

float biq_surface_coordinate(BiqWindowTile const & tile, float u, float v,
                             float signed_shore_distance) {
    if (signed_shore_distance <= 0.0f)
        return signed_shore_distance;
    float shore_depth = std::sqrt(smoothstep01(
        signed_shore_distance));
    return shore_depth * biq_water_family_depth(tile, u, v);
}

float biq_coastal_relief_envelope(BiqWindowTile const & tile, float u, float v) {
    if (is_water_terrain(tile.base))
        return 0.0f;
    // The shoreline field is continuous across BIQ cells. Flatten authored
    // relief before its material reaches the beach so hills, mountains, and
    // volcanoes meet the same low coastal shelf instead of ending in an
    // elevated tile-shaped lip.
    float signed_shore = biq_signed_shore_distance(tile, u, v);
    return smoothstep01((-signed_shore - 0.02f) / 0.42f);
}

float biq_relief_envelope(BiqWindowTile const & tile, float u, float v) {
    constexpr int offsets[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    float distances[4] = {u, v, 1.0f - u, 1.0f - v};
    float envelope = 1.0f;
    bool chain_relief = tile.real == 6 || tile.real == 10;
    float fade_width = chain_relief ? 0.28f : 0.16f;
    for (unsigned edge = 0; edge < 4; ++edge) {
        BiqWindowTile const * neighbor = find_biq_source_tile(
            tile.source_x + offsets[edge][0], tile.source_y + offsets[edge][1]);
        bool continues = neighbor != nullptr &&
            (chain_relief
                ? (neighbor->real == 5 || neighbor->real == 6 || neighbor->real == 10)
                : (neighbor->real == tile.real &&
                   (tile.real != 0 || neighbor->base == tile.base)));
        if (!continues)
            envelope *= smoothstep01(distances[edge] / fade_width);
    }
    return envelope;
}

float biq_mountain_hill_transition_envelope(BiqWindowTile const & tile,
                                            float u, float v) {
    if (tile.real != 5)
        return 1.0f;
    constexpr int offsets[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    float distances[4] = {u, v, 1.0f - u, 1.0f - v};
    float envelope = 0.0f;
    for (unsigned edge = 0; edge < 4; ++edge) {
        BiqWindowTile const * neighbor = find_biq_source_tile(
            tile.source_x + offsets[edge][0], tile.source_y + offsets[edge][1]);
        if (neighbor != nullptr && neighbor->real == 6)
            envelope = std::max(envelope,
                1.0f - smoothstep01(distances[edge] / 0.24f));
    }
    return envelope;
}

float biq_mountain_material_envelope(BiqWindowTile const & tile,
                                     float u, float v) {
    if (tile.real != 6 && tile.real != 10)
        return 0.0f;
    constexpr int offsets[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    float distances[4] = {u, v, 1.0f - u, 1.0f - v};
    float envelope = 1.0f;
    for (unsigned edge = 0; edge < 4; ++edge) {
        BiqWindowTile const * neighbor = find_biq_source_tile(
            tile.source_x + offsets[edge][0], tile.source_y + offsets[edge][1]);
        bool rock_continues = neighbor != nullptr &&
            (neighbor->real == 6 || neighbor->real == 10);
        if (!rock_continues)
            envelope *= smoothstep01(distances[edge] / 0.28f);
    }
    return envelope;
}

float smooth_relief_max(float hill, float mountain) {
    if (hill <= 0.001f || mountain <= 0.001f)
        return std::max(hill, mountain);
    constexpr float blend_width = 12.0f;
    float weight = std::clamp(
        0.5f + 0.5f * (mountain - hill) / blend_width, 0.0f, 1.0f);
    return hill * (1.0f - weight) + mountain * weight +
           blend_width * weight * (1.0f - weight);
}

float biq_hill_support(float world_x, float world_y) {
    int center_x = static_cast<int>(std::floor(world_x));
    int center_y = static_cast<int>(std::floor(world_y));
    float support = 0.0f;
    for (int row = center_y - 1; row <= center_y + 1; ++row) {
        for (int column = center_x - 1; column <= center_x + 1; ++column) {
            BiqWindowTile const * candidate = biq_tile_at(column, row);
            if (candidate == nullptr || candidate->real != 5)
                continue;
            // Hill ownership chooses the source landform, but does not clip
            // its skirt. This rounded support reaches into neighboring land;
            // the world-space Civ VI heightfield still supplies the visible
            // irregular relief inside it.
            float dx = (world_x - (static_cast<float>(column) + 0.5f)) / 0.92f;
            float dy = (world_y - (static_cast<float>(row) + 0.5f)) / 0.78f;
            float distance = std::sqrt(dx * dx + dy * dy);
            support = std::max(support, smoothstep01((1.0f - distance) / 0.42f));
        }
    }
    return support;
}

float biq_hill_compatibility_envelope(BiqWindowTile const & tile, float u, float v) {
    constexpr int offsets[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    float distances[4] = {u, v, 1.0f - u, 1.0f - v};
    float envelope = 1.0f;
    for (unsigned edge = 0; edge < 4; ++edge) {
        BiqWindowTile const * neighbor = find_biq_source_tile(
            tile.source_x + offsets[edge][0], tile.source_y + offsets[edge][1]);
        bool compatible = neighbor != nullptr && !is_water_terrain(neighbor->base);
        if (!compatible)
            envelope *= smoothstep01(distances[edge] / 0.22f);
    }
    return envelope;
}

float biq_river_distance(BiqWindowTile const & tile, float u, float v);

float biq_tile_height(BiqWindowTile const & tile, float u, float v,
                      HeightField const * authored_height,
                      HeightField const * authored_blend,
                      bool flat_underlay = false) {
    if (is_water_terrain(tile.base))
        // Land, submerged bed, and transparent water must meet on one geometric
        // datum at the shared tile edge. Optical depth remains a material
        // property; lowering the entire water diamond exposed the black clear
        // color as a narrow vertical slit between otherwise matching cells.
        return 2.5f;
    if (flat_underlay)
        return 2.5f;
    float world_x = static_cast<float>(tile.column) + u;
    float world_y = static_cast<float>(tile.row) + (1.0f - v);
    float relief_envelope = biq_relief_envelope(tile, u, v);
    float coastal_envelope = biq_coastal_relief_envelope(tile, u, v);
    float height = 2.5f;
    float hill_support = biq_hill_support(world_x, world_y) *
        biq_hill_compatibility_envelope(tile, u, v) * coastal_envelope;
    float hill_displacement = promotion_hill_value(world_x, world_y) *
                              52.0f * hill_support;
    float authored_displacement = 0.0f;
    if ((tile.real == 5 || tile.real == 6 || tile.real == 10) &&
        (authored_height != nullptr || tile.real != 6) &&
        (authored_blend != nullptr || tile.real != 6) &&
        (tile.real != 10 || volcano_geometry_enabled)) {
        float chain_height = 0.0f;
        float chain_blend = 0.0f;
        float chain_displacement = 0.0f;
        biq_chain_relief_sample(world_x, world_y, chain_height, chain_blend,
                                chain_displacement, tile.real != 5);
        // Compose overlapping authored footprints in world space. Adjacent
        // mountain/volcano cells therefore share shoulders and saddles without
        // any generated connector mesh. Mountain shoulders may also continue
        // into an adjacent authored hill instead of being cut flat there.
        authored_displacement = chain_displacement * coastal_envelope *
            biq_mountain_hill_transition_envelope(tile, u, v) *
            ((tile.real == 6 || tile.real == 10) ? relief_envelope : 1.0f);
    }
    height += smooth_relief_max(hill_displacement, authored_displacement);
    if (tile.base == 0 && tile.real == 0)
        height += dune_height_value(world_x, world_y) * relief_envelope;
    if (l13_scene_enabled && tile.river_mask != 0) {
        float river_distance = biq_river_distance(tile, u, v);
        float valley = 1.0f - smoothstep01((river_distance - 4.0f) / 16.0f);
        float valley_floor = 2.5f + (height - 2.5f) * 0.10f;
        height = height * (1.0f - valley * 0.92f) +
                 valley_floor * valley * 0.92f;
    }
    return height;
}

unsigned biq_river_branch_count(unsigned mask) {
    unsigned count = 0;
    for (unsigned bit : {2u, 8u, 32u, 128u})
        if ((mask & bit) != 0)
            ++count;
    return count;
}

float biq_river_screen_segment_distance(float point_x, float point_y,
                                        float start_x, float start_y,
                                        float endpoint_x, float endpoint_y) {
    float segment_x = endpoint_x - start_x;
    float segment_y = endpoint_y - start_y;
    float point_offset_x = point_x - start_x;
    float point_offset_y = point_y - start_y;
    float denominator = segment_x * segment_x + segment_y * segment_y;
    float t = denominator > 0.0f
        ? std::clamp((point_offset_x * segment_x + point_offset_y * segment_y) /
                         denominator,
                     0.0f, 1.0f)
        : 0.0f;
    float delta_x = point_offset_x - segment_x * t;
    float delta_y = point_offset_y - segment_y * t;
    return std::sqrt(delta_x * delta_x + delta_y * delta_y);
}

float biq_river_edge_distance(BiqWindowTile const & tile, float u, float v,
                             float start_u, float start_v,
                             float endpoint_u, float endpoint_v,
                             unsigned direction_bit) {
    // Lay out the exported diamond in its continuous local lattice. Raw BIQ X
    // wraps at the map seam, so subtracting raw source coordinates can throw a
    // neighboring tile across the viewport even though its local column/row is
    // adjacent.
    float center_x = static_cast<float>(tile.column + tile.row) *
                     coast_projection.half_width;
    float center_y = static_cast<float>(tile.column - tile.row) *
                     coast_projection.half_height;
    auto screen_point = [&](float local_u, float local_v, float & out_x, float & out_y) {
        out_x = center_x + (local_u - local_v) * coast_projection.half_width;
        out_y = center_y + (local_u + local_v - 1.0f) *
                coast_projection.half_height;
    };
    float point_x = 0.0f;
    float point_y = 0.0f;
    float start_x = 0.0f;
    float start_y = 0.0f;
    float endpoint_x = 0.0f;
    float endpoint_y = 0.0f;
    screen_point(u, v, point_x, point_y);
    screen_point(start_u, start_v, start_x, start_y);
    screen_point(endpoint_u, endpoint_v, endpoint_x, endpoint_y);

    // River flags name shared tile edges. Canonicalize each reciprocal pair
    // to the same local grid coordinate so both owner diamonds evaluate the
    // exact same curve, including at horizontal map wrap.
    int canonical_column = tile.column;
    int canonical_row = tile.row;
    unsigned edge_family = 0u;
    if (direction_bit == 32u)
        canonical_row -= 1;
    else if (direction_bit == 128u) {
        canonical_column -= 1;
        edge_family = 1u;
    } else if (direction_bit == 8u)
        edge_family = 1u;
    unsigned seed = static_cast<unsigned>(canonical_column + 4096) * 73856093u ^
                    static_cast<unsigned>(canonical_row + 4096) * 19349663u ^
                    edge_family * 83492791u;
    seed ^= seed >> 13;
    float direction_x = endpoint_x - start_x;
    float direction_y = endpoint_y - start_y;
    float direction_length = std::sqrt(direction_x * direction_x +
                                       direction_y * direction_y);
    float normal_x = -direction_y / std::max(direction_length, 0.001f);
    float normal_y = direction_x / std::max(direction_length, 0.001f);
    float primary_bend = 5.0f +
        static_cast<float>((seed >> 5) & 15u) / 15.0f * 8.0f;
    float secondary_bend = 1.5f +
        static_cast<float>((seed >> 11) & 7u) / 7.0f * 4.0f;
    if ((seed & 1u) != 0)
        primary_bend = -primary_bend;
    if ((seed & 2u) != 0)
        secondary_bend = -secondary_bend;
    float distance = 1000.0f;
    float previous_x = start_x;
    float previous_y = start_y;
    for (int index = 1; index <= 16; ++index) {
        float t = static_cast<float>(index) / 16.0f;
        float offset = std::sin(t * 3.14159265f) * primary_bend +
                       std::sin(t * 6.28318531f) * secondary_bend;
        float curve_x = start_x + direction_x * t + normal_x * offset;
        float curve_y = start_y + direction_y * t + normal_y * offset;
        distance = std::min(distance, biq_river_screen_segment_distance(
            point_x, point_y, previous_x, previous_y, curve_x, curve_y));
        previous_x = curve_x;
        previous_y = curve_y;
    }
    return distance;
}

float biq_river_distance(BiqWindowTile const & tile, float u, float v) {
    float distance = 1000.0f;
    if ((tile.river_mask & 2u) != 0)
        distance = std::min(distance, biq_river_edge_distance(
            tile, u, v, 0.0f, 0.0f, 1.0f, 0.0f, 2u));
    if ((tile.river_mask & 8u) != 0)
        distance = std::min(distance, biq_river_edge_distance(
            tile, u, v, 1.0f, 0.0f, 1.0f, 1.0f, 8u));
    if ((tile.river_mask & 32u) != 0)
        distance = std::min(distance, biq_river_edge_distance(
            tile, u, v, 0.0f, 1.0f, 1.0f, 1.0f, 32u));
    if ((tile.river_mask & 128u) != 0)
        distance = std::min(distance, biq_river_edge_distance(
            tile, u, v, 0.0f, 0.0f, 0.0f, 1.0f, 128u));
    return distance;
}

float biq_river_point_distance(float u, float v, float point_u, float point_v) {
    float delta_u = u - point_u;
    float delta_v = v - point_v;
    float screen_x = (delta_u - delta_v) * coast_projection.half_width;
    float screen_y = (delta_u + delta_v) * coast_projection.half_height;
    return std::sqrt(screen_x * screen_x + screen_y * screen_y);
}

float biq_river_mouth_distance(BiqWindowTile const & tile, float u, float v) {
    return biq_river_node_distance(tile, u, v, 2u);
}

float biq_world_height(float world_x, float world_y,
                       HeightField const * authored_height,
                       HeightField const * authored_blend) {
    int tile_x = static_cast<int>(std::floor(world_x));
    int tile_y = static_cast<int>(std::floor(world_y));
    BiqWindowTile const * tile = biq_tile_at(tile_x, tile_y);
    if (tile == nullptr)
        return 2.5f;
    float u = world_x - static_cast<float>(tile_x);
    float v = 1.0f - (world_y - static_cast<float>(tile_y));
    return biq_tile_height(*tile, u, v, authored_height, authored_blend, false);
}

float biq_cast_shadow_visibility(BiqWindowTile const & tile, float u, float v,
                                 HeightField const * authored_height,
                                 HeightField const * authored_blend) {
    if (!l13a_scene_enabled)
        return 1.0f;
    float world_x = static_cast<float>(tile.column) + u;
    float world_y = static_cast<float>(tile.row) + (1.0f - v);
    float origin_height = biq_tile_height(
        tile, u, v, authored_height, authored_blend, false);
    float horizontal = std::sqrt(active_light_direction[0] * active_light_direction[0] +
                                 active_light_direction[1] * active_light_direction[1]);
    if (horizontal < 0.001f)
        return 1.0f;
    float direction_x = active_light_direction[0] / horizontal;
    // biq_world_height uses row + (1 - local_v), so its continuous Y axis is
    // inverted relative to feature placement/local-v space. Convert the
    // shared light vector here; otherwise relief shadows mirror vegetation and
    // feature shadows across the isometric X axis.
    float direction_y = -active_light_direction[1] / horizontal;
    float perpendicular_x = -direction_y;
    float perpendicular_y = direction_x;
    // The canonical day/night presentation rotates a stylized key light but
    // keeps cast-shadow length in a tight band.  Use one projection slope for
    // relief at every phase; object height, not clock elevation, determines
    // how far hills, mountains, and volcanoes cast.
    float ray_slope = 96.0f;
    float occlusion = 0.0f;
    for (int lane = -1; lane <= 1; ++lane) {
        float greatest_obstruction = 0.0f;
        float lane_offset = static_cast<float>(lane) * 0.075f;
        for (int step = 1; step <= 48; ++step) {
            float distance = static_cast<float>(step) * 0.12f;
            float sample_x = world_x + direction_x * distance +
                             perpendicular_x * lane_offset;
            float sample_y = world_y + direction_y * distance +
                             perpendicular_y * lane_offset;
            float sample_height = biq_world_height(
                sample_x, sample_y, authored_height, authored_blend);
            float ray_height = origin_height + ray_slope * distance + 0.8f;
            greatest_obstruction = std::max(
                greatest_obstruction, sample_height - ray_height);
        }
        occlusion += std::clamp(
            (greatest_obstruction - 0.5f) / 12.0f, 0.0f, 0.78f);
    }
    return 1.0f - occlusion / 3.0f;
}

Vertex make_biq_vertex(BiqWindowTile const & tile, float u, float v, float uv_scale,
                       float surface_kind, HeightField const * authored_height,
                       HeightField const * authored_blend) {
    float center_x = coast_projection.origin_x +
        static_cast<float>(tile.column + tile.row) * coast_projection.half_width;
    float center_y = coast_projection.origin_y +
        static_cast<float>(tile.column - tile.row) * coast_projection.half_height;
    bool flat_underlay = (surface_kind > 0.25f && surface_kind < 0.75f) ||
                         (surface_kind > 3.5f && surface_kind < 5.5f);
    float height = biq_tile_height(tile, u, v, authored_height, authored_blend,
                                   flat_underlay);
    constexpr float step = 0.006f;
    float left = biq_tile_height(tile, std::max(0.0f, u - step), v,
                                 authored_height, authored_blend, flat_underlay);
    float right = biq_tile_height(tile, std::min(1.0f, u + step), v,
                                  authored_height, authored_blend, flat_underlay);
    float down = biq_tile_height(tile, u, std::max(0.0f, v - step),
                                 authored_height, authored_blend, flat_underlay);
    float up = biq_tile_height(tile, u, std::min(1.0f, v + step),
                               authored_height, authored_blend, flat_underlay);
    float slope_u = (right - left) / (2.0f * step * coast_projection.half_width * 2.0f);
    float slope_v = (up - down) / (2.0f * step * coast_projection.half_width * 2.0f);
    float screen_x = center_x + (u - v) * coast_projection.half_width;
    float base_screen_y = center_y + (u + v - 1.0f) * coast_projection.half_height;
    float screen_y = base_screen_y - height * coast_projection.vertical_scale;
    float terrain_depth = std::clamp(
        0.94f - base_screen_y / static_cast<float>(output_height) * 0.75f -
        height * 0.0012f, 0.01f, 0.99f);
    if (surface_kind > 8.5f && surface_kind < 9.5f)
        terrain_depth = std::max(0.005f, terrain_depth - 0.025f);
    float length = std::sqrt(slope_u * slope_u + slope_v * slope_v + 1.0f);
    float shore_distance = biq_signed_shore_distance(tile, u, v);
    float coordinate = biq_surface_coordinate(tile, u, v, shore_distance);
    float authored_relief_height = 0.0f;
    float authored_relief_blend = 0.0f;
    if (tile.real == 5 || tile.real == 6 || tile.real == 10) {
        float chain_displacement = 0.0f;
        biq_chain_relief_sample(static_cast<float>(tile.column) + u,
                                static_cast<float>(tile.row) + (1.0f - v),
                                authored_relief_height, authored_relief_blend,
                                chain_displacement, tile.real != 5);
        // Geometry and material must share the same outer range envelope.
        // Without this factor the surface flattened toward grass while the
        // mountain albedo stayed opaque until the hidden diamond boundary.
        // The shape may continue into a neighboring hill, but rock material
        // remains owned by mountain/volcano cells and fades before that edge.
        // The hill-side shoulder therefore stays grass-covered instead of
        // becoming a vertical gray material wall.
        float material_envelope = tile.real == 5
            ? biq_mountain_hill_transition_envelope(tile, u, v) * 0.35f
            : biq_mountain_material_envelope(tile, u, v);
        authored_relief_blend *= material_envelope *
                                 biq_coastal_relief_envelope(tile, u, v);
    }
    float weights[4] = {};
    biq_material_weights(tile, u, v, weights);
    return Vertex{ndc_x(screen_x), ndc_y(screen_y),
                  (static_cast<float>(tile.column) + u) * uv_scale,
                  (static_cast<float>(tile.row) + (1.0f - v)) * uv_scale,
                  1.0f, -slope_u / length, -slope_v / length, 1.0f / length,
                  1.0f, 1.0f,
                  (static_cast<float>(tile.column) + u) * 0.5f,
                  (static_cast<float>(tile.row) + (1.0f - v)) * 0.5f,
                  surface_kind, coordinate,
                  static_cast<float>(tile.base), static_cast<float>(tile.real),
                  weights[0], weights[1], weights[2], weights[3], terrain_depth,
                  authored_relief_height, authored_relief_blend, shore_distance,
                  biq_river_distance(tile, u, v),
                  biq_river_node_distance(tile, u, v, 1u),
                  biq_river_mouth_distance(tile, u, v),
                  biq_river_node_distance(tile, u, v, 0u)};
}

void add_biq_tile_surface(std::vector<Vertex> & vertices, BiqWindowTile const & tile,
                          float uv_scale, float surface_kind,
                          HeightField const * authored_height,
                          HeightField const * authored_blend,
                          int subdivisions = 16) {
    for (int y = 0; y < subdivisions; ++y) {
        for (int x = 0; x < subdivisions; ++x) {
            float u0 = static_cast<float>(x) / subdivisions;
            float v0 = static_cast<float>(y) / subdivisions;
            float u1 = static_cast<float>(x + 1) / subdivisions;
            float v1 = static_cast<float>(y + 1) / subdivisions;
            Vertex top = make_biq_vertex(tile, u0, v0, uv_scale, surface_kind,
                                         authored_height, authored_blend);
            Vertex right = make_biq_vertex(tile, u1, v0, uv_scale, surface_kind,
                                           authored_height, authored_blend);
            Vertex bottom = make_biq_vertex(tile, u1, v1, uv_scale, surface_kind,
                                            authored_height, authored_blend);
            Vertex left = make_biq_vertex(tile, u0, v1, uv_scale, surface_kind,
                                          authored_height, authored_blend);
            add_triangle(vertices, top, right, bottom);
            add_triangle(vertices, top, bottom, left);
        }
    }
}

void add_biq_shadow_surface(std::vector<Vertex> & vertices,
                            BiqWindowTile const & tile,
                            float uv_scale,
                            HeightField const * authored_height,
                            HeightField const * authored_blend) {
    constexpr int subdivisions = 16;
    auto shadow_vertex = [&](float u, float v) {
        Vertex result = make_biq_vertex(tile, u, v, uv_scale, 10.0f,
                                        authored_height, authored_blend);
        result.shadow_visibility = biq_cast_shadow_visibility(
            tile, u, v, authored_height, authored_blend);
        result.terrain_depth = std::max(0.005f, result.terrain_depth - 0.003f);
        return result;
    };
    for (int y = 0; y < subdivisions; ++y) {
        for (int x = 0; x < subdivisions; ++x) {
            float u0 = static_cast<float>(x) / subdivisions;
            float v0 = static_cast<float>(y) / subdivisions;
            float u1 = static_cast<float>(x + 1) / subdivisions;
            float v1 = static_cast<float>(y + 1) / subdivisions;
            Vertex top = shadow_vertex(u0, v0);
            Vertex right = shadow_vertex(u1, v0);
            Vertex bottom = shadow_vertex(u1, v1);
            Vertex left = shadow_vertex(u0, v1);
            add_triangle(vertices, top, right, bottom);
            add_triangle(vertices, top, bottom, left);
        }
    }
}

void add_biq_patch(std::vector<Vertex> & vertices, float uv_scale,
                   HeightField const * authored_height,
                   HeightField const * authored_blend, bool water_enabled) {
    // Pass-major ordering is essential here. Every tile contributes to the
    // same ground, bed, and water surfaces, and the pixel shader clips the two
    // water passes against one viewport-wide signed shoreline. Emitting
    // surface types per ownership tile made later land diamonds overwrite
    // water from an earlier neighbor and forced the visible coast back onto
    // Civ III's tile edges even though the scalar field itself was continuous.
    for (BiqWindowTile const & tile : biq_window.tiles)
        add_biq_tile_surface(vertices, tile, uv_scale, 0.5f,
                             authored_height, authored_blend);

    for (BiqWindowTile const & tile : biq_window.tiles) {
        if (!is_water_terrain(tile.base)) {
            add_biq_tile_surface(vertices, tile, uv_scale, 1.0f,
                                 authored_height, authored_blend);
        }
    }
    for (BiqWindowTile const & tile : biq_window.tiles)
        add_biq_tile_surface(vertices, tile, uv_scale, 4.0f,
                             authored_height, authored_blend);
    if (water_enabled)
        for (BiqWindowTile const & tile : biq_window.tiles)
            add_biq_tile_surface(vertices, tile, uv_scale, 5.0f,
                                 authored_height, authored_blend);
    if (l13_scene_enabled && river_geometry_enabled)
        for (BiqWindowTile const & tile : biq_window.tiles)
            if (tile.river_mask != 0)
                add_biq_tile_surface(vertices, tile, uv_scale, 9.0f,
                                     authored_height, authored_blend, 32);
    if (l13a_scene_enabled)
        for (BiqWindowTile const & tile : biq_window.tiles)
            if (!is_water_terrain(tile.base))
                add_biq_shadow_surface(vertices, tile, uv_scale,
                                       authored_height, authored_blend);
}

Vertex make_biq_road_vertex(float world_x, float world_y, float atlas_u,
                            float atlas_v, unsigned style, unsigned pillaged,
                            unsigned bridge, HeightField const * authored_height,
                            HeightField const * authored_blend) {
    int logical_column = static_cast<int>(std::floor(world_x));
    int row = static_cast<int>(std::floor(world_y));
    int wrapped_column = logical_column;
    while (wrapped_column < 0)
        wrapped_column += biq_window.columns;
    while (wrapped_column >= biq_window.columns)
        wrapped_column -= biq_window.columns;
    BiqWindowTile const * tile = biq_tile_at(wrapped_column, row);
    if (tile == nullptr)
        tile = &biq_window.tiles.front();
    float u = world_x - std::floor(world_x);
    float v = 1.0f - (world_y - std::floor(world_y));
    Vertex result = make_biq_vertex(*tile, u, v, 1.0f, 11.0f,
                                    authored_height, authored_blend);
    int column_delta = logical_column - wrapped_column;
    result.x += static_cast<float>(column_delta) * coast_projection.half_width *
                2.0f / static_cast<float>(output_width);
    result.y -= static_cast<float>(column_delta) * coast_projection.half_height *
                2.0f / static_cast<float>(output_height);
    result.u = atlas_u;
    result.v = atlas_v;
    result.base_terrain = static_cast<float>(style);
    result.real_terrain = static_cast<float>(pillaged);
    result.material_grass = world_x;
    result.material_plains = world_y;
    result.authored_relief_height = static_cast<float>(bridge);
    result.terrain_depth = std::max(0.003f, result.terrain_depth - 0.010f);
    return result;
}

void add_road_segment(std::vector<Vertex> & vertices, float x0, float y0,
                      float x1, float y1, unsigned style, unsigned pillaged,
                      unsigned bridge, HeightField const * authored_height,
                      HeightField const * authored_blend) {
    constexpr int subdivisions = 16;
    bool railroad = style >= 4u;
    float half_width = railroad ? 0.076f : 0.105f;
    float atlas_half_width = railroad ? 0.058f : 0.075f;
    float dx = x1 - x0;
    float dy = y1 - y0;
    float length = std::sqrt(dx * dx + dy * dy);
    if (length < 0.001f)
        return;
    float original_x0 = x0;
    float original_y0 = y0;
    float original_x1 = x1;
    float original_y1 = y1;
    float original_length = length;
    float direction_x = dx / length;
    float direction_y = dy / length;
    // Tiled source pieces are authored to meet continuously. Extend them a
    // little through graph nodes so anisotropic filtering and alpha coverage
    // cannot expose pinholes at curves, crossings, or multi-way junctions.
    x0 -= direction_x * 0.14f;
    y0 -= direction_y * 0.14f;
    x1 += direction_x * 0.14f;
    y1 += direction_y * 0.14f;
    dx = x1 - x0;
    dy = y1 - y0;
    length = std::sqrt(dx * dx + dy * dy);
    float perpendicular_x = -dy / length;
    float perpendicular_y = dx / length;
    float atlas_x0 = 0.0f;
    float atlas_y0 = 0.90606654f;
    float atlas_x1 = 1.0f;
    float atlas_y1 = 0.99021526f;
    float atlas_dx = atlas_x1 - atlas_x0;
    float atlas_dy = atlas_y1 - atlas_y0;
    float atlas_length = std::sqrt(atlas_dx * atlas_dx + atlas_dy * atlas_dy);
    float atlas_perpendicular_x = -atlas_dy / atlas_length;
    float atlas_perpendicular_y = atlas_dx / atlas_length;
    float wave_seed = std::fmod(std::fabs(
        original_x0 * 17.0f + original_y0 * 31.0f +
        original_x1 * 47.0f + original_y1 * 61.0f), 19.0f) / 19.0f;
    float wave_phase = wave_seed * 6.28318530718f;
    auto road_vertex = [&](float along, float across) {
        float source_along = (along * length - 0.14f) / original_length;
        float curve_t = std::clamp(source_along, 0.0f, 1.0f);
        float curve_envelope = std::sin(curve_t * 3.14159265359f);
        float curve_amplitude = bridge ? 0.014f : (railroad ? 0.028f : 0.042f);
        float road_wave = curve_envelope * curve_amplitude *
            (0.62f * std::sin(wave_phase) +
             0.38f * std::sin(curve_t * 6.28318530718f + wave_phase));
        float world_x = x0 + dx * along + perpendicular_x *
            (half_width * across + road_wave);
        float world_y = y0 + dy * along + perpendicular_y *
            (half_width * across + road_wave);
        float atlas_u = atlas_x0 + atlas_dx * source_along +
                        atlas_perpendicular_x * atlas_half_width * across;
        float atlas_v = atlas_y0 + atlas_dy * source_along +
                        atlas_perpendicular_y * atlas_half_width * across;
        Vertex result = make_biq_road_vertex(world_x, world_y, atlas_u, atlas_v,
                                             style, pillaged, bridge,
                                             authored_height, authored_blend);
        result.macro_u = atlas_x0 + atlas_dx * source_along;
        result.macro_v = atlas_y0 + atlas_dy * source_along;
        result.shadow_visibility = across;
        result.ambient_visibility = curve_t;
        return result;
    };
    for (int index = 0; index < subdivisions; ++index) {
        float a0 = static_cast<float>(index) / subdivisions;
        float a1 = static_cast<float>(index + 1) / subdivisions;
        Vertex left0 = road_vertex(a0, -1.0f);
        Vertex right0 = road_vertex(a0, 1.0f);
        Vertex right1 = road_vertex(a1, 1.0f);
        Vertex left1 = road_vertex(a1, -1.0f);
        add_triangle(vertices, left0, right0, right1);
        add_triangle(vertices, left0, right1, left1);
    }
}

void add_road_scene(std::vector<Vertex> & vertices,
                    HeightField const * authored_height,
                    HeightField const * authored_blend,
                    int style_override) {
    for (RoadEdge const & edge : road_scenario.edges) {
        unsigned style = style_override >= 0
            ? static_cast<unsigned>(style_override) : edge.style;
        float x0 = static_cast<float>(edge.x0) + 0.5f;
        float y0 = static_cast<float>(edge.y0) + 0.5f;
        float x1 = static_cast<float>(edge.x1) + 0.5f;
        float y1 = static_cast<float>(edge.y1) + 0.5f;
        if (edge.wraps) {
            add_road_segment(vertices, x0, y0,
                             static_cast<float>(biq_window.columns), y1,
                             style, edge.pillaged, edge.bridge,
                             authored_height, authored_blend);
            add_road_segment(vertices, 0.0f, y0, x1, y1,
                             style, edge.pillaged, edge.bridge,
                             authored_height, authored_blend);
        } else {
            add_road_segment(vertices, x0, y0, x1, y1,
                             style, edge.pillaged, edge.bridge,
                             authored_height, authored_blend);
        }
    }
}

void add_railroad_scene(std::vector<Vertex> & vertices,
                        HeightField const * authored_height,
                        HeightField const * authored_blend) {
    for (RoadEdge const & edge : railroad_scenario.edges) {
        float x0 = static_cast<float>(edge.x0) + 0.5f;
        float y0 = static_cast<float>(edge.y0) + 0.5f;
        float x1 = static_cast<float>(edge.x1) + 0.5f;
        float y1 = static_cast<float>(edge.y1) + 0.5f;
        if (edge.wraps) {
            add_road_segment(vertices, x0, y0,
                             static_cast<float>(biq_window.columns), y1,
                             4u, edge.pillaged, edge.bridge,
                             authored_height, authored_blend);
            add_road_segment(vertices, 0.0f, y0, x1, y1,
                             4u, edge.pillaged, edge.bridge,
                             authored_height, authored_blend);
        } else {
            add_road_segment(vertices, x0, y0, x1, y1,
                             4u, edge.pillaged, edge.bridge,
                             authored_height, authored_blend);
        }
    }
}

void add_biq_grid(std::vector<Vertex> & vertices, HeightField const * authored_height,
                  HeightField const * authored_blend) {
    // The promotion grid was useful while validating the exact 96-cell BIQ
    // selection, but depth testing hid arbitrary portions of it behind raised
    // terrain and made the surviving segments look like black cracks. Keep the
    // diagnostic path callable without rasterizing it in the beauty candidate.
    constexpr float thickness = 0.0f;
    for (BiqWindowTile const & tile : biq_window.tiles) {
        auto vertex = [&](float u, float v) {
            return make_biq_vertex(tile, u, v, 1.0f, 8.0f,
                                   authored_height, authored_blend);
        };
        Vertex top0 = vertex(0.0f, 0.0f);
        Vertex top1 = vertex(1.0f, 0.0f);
        Vertex top2 = vertex(1.0f, thickness);
        Vertex top3 = vertex(0.0f, thickness);
        add_triangle(vertices, top0, top1, top2);
        add_triangle(vertices, top0, top2, top3);
        Vertex right0 = vertex(1.0f - thickness, 0.0f);
        Vertex right1 = vertex(1.0f, 0.0f);
        Vertex right2 = vertex(1.0f, 1.0f);
        Vertex right3 = vertex(1.0f - thickness, 1.0f);
        add_triangle(vertices, right0, right1, right2);
        add_triangle(vertices, right0, right2, right3);
        Vertex bottom0 = vertex(0.0f, 1.0f - thickness);
        Vertex bottom1 = vertex(1.0f, 1.0f - thickness);
        Vertex bottom2 = vertex(1.0f, 1.0f);
        Vertex bottom3 = vertex(0.0f, 1.0f);
        add_triangle(vertices, bottom0, bottom1, bottom2);
        add_triangle(vertices, bottom0, bottom2, bottom3);
        Vertex left0 = vertex(0.0f, 0.0f);
        Vertex left1 = vertex(thickness, 0.0f);
        Vertex left2 = vertex(thickness, 1.0f);
        Vertex left3 = vertex(0.0f, 1.0f);
        add_triangle(vertices, left0, left1, left2);
        add_triangle(vertices, left0, left2, left3);
    }
}

bool add_coast_patch(std::vector<Vertex> & vertices, float uv_scale, bool cliff_corner,
                     bool beauty_scene, bool water_enabled, bool surf_enabled,
                     bool integrated_shore,
                     HeightField const * authored_height,
                     HeightField const * authored_blend) {
    constexpr int rows_per_tile = 32;
    int rows = static_cast<int>(scene_world_height * rows_per_tile);
    float beach_inland_height = integrated_shore ? 2.5f : 12.0f;
    for (int row = 0; row < rows; ++row) {
        float y0 = static_cast<float>(row) * scene_world_height / rows;
        float y1 = static_cast<float>(row + 1) * scene_world_height / rows;
        float coast0 = coast_position(y0, cliff_corner, integrated_shore);
        float coast1 = coast_position(y1, cliff_corner, integrated_shore);
        float beach_width0 = beach_width_at(y0, integrated_shore);
        float beach_width1 = beach_width_at(y1, integrated_shore);
        bool cliff_section = cliff_corner;
        float cliff_height = integrated_shore ? beach_inland_height : 34.0f;
        if (!cliff_section) {
            if (!beauty_scene) {
                add_coast_strip(vertices, y0, y1, 0.0f, 0.0f,
                                coast0 - beach_width0, coast1 - beach_width1,
                                beach_inland_height, beach_inland_height,
                                beach_inland_height, beach_inland_height,
                                uv_scale, 1.0f, 0.0f, 0.0f, 1.0f, 20);
            } else {
                int land_columns = promotion_scene_enabled
                    ? static_cast<int>(scene_world_width * 32.0f) : 32;
                float edge0 = coast0 - beach_width0;
                float edge1 = coast1 - beach_width1;
                for (int column = 0; column < land_columns; ++column) {
                    float t0 = static_cast<float>(column) / land_columns;
                    float t1 = static_cast<float>(column + 1) / land_columns;
                    Vertex top = make_beauty_land_vertex(edge0 * t0, y0, edge0,
                                                         beach_inland_height, uv_scale,
                                                         authored_height, authored_blend,
                                                         integrated_shore);
                    Vertex right = make_beauty_land_vertex(edge0 * t1, y0, edge0,
                                                           beach_inland_height, uv_scale,
                                                           authored_height, authored_blend,
                                                           integrated_shore);
                    Vertex bottom = make_beauty_land_vertex(edge1 * t1, y1, edge1,
                                                            beach_inland_height, uv_scale,
                                                            authored_height, authored_blend,
                                                            integrated_shore);
                    Vertex left = make_beauty_land_vertex(edge1 * t0, y1, edge1,
                                                          beach_inland_height, uv_scale,
                                                          authored_height, authored_blend,
                                                          integrated_shore);
                    add_triangle(vertices, top, right, bottom);
                    add_triangle(vertices, top, bottom, left);
                }
            }
            add_coast_strip(vertices, y0, y1, coast0 - beach_width0, coast1 - beach_width1,
                            coast0, coast1,
                            beach_inland_height, beach_inland_height, 0.0f, 0.0f,
                            uv_scale, 2.0f, integrated_shore ? -0.06f : -0.28f,
                            0.0f, 1.0f, integrated_shore ? 8 : 5);
        } else {
            if (!beauty_scene) {
                add_coast_strip(vertices, y0, y1, 0.0f, 0.0f, coast0, coast1,
                                cliff_height, cliff_height, cliff_height, cliff_height,
                                uv_scale, 1.0f, 0.0f, 0.0f, 1.0f, 24);
            } else {
                int land_columns = promotion_scene_enabled
                    ? static_cast<int>(scene_world_width * 32.0f) : 32;
                for (int column = 0; column < land_columns; ++column) {
                    float t0 = static_cast<float>(column) / land_columns;
                    float t1 = static_cast<float>(column + 1) / land_columns;
                    Vertex top = make_beauty_land_vertex(coast0 * t0, y0, coast0, cliff_height, uv_scale,
                                                         authored_height, authored_blend,
                                                         integrated_shore);
                    Vertex right = make_beauty_land_vertex(coast0 * t1, y0, coast0, cliff_height, uv_scale,
                                                           authored_height, authored_blend,
                                                           integrated_shore);
                    Vertex bottom = make_beauty_land_vertex(coast1 * t1, y1, coast1, cliff_height, uv_scale,
                                                            authored_height, authored_blend,
                                                            integrated_shore);
                    Vertex left = make_beauty_land_vertex(coast1 * t0, y1, coast1, cliff_height, uv_scale,
                                                          authored_height, authored_blend,
                                                          integrated_shore);
                    add_triangle(vertices, top, right, bottom);
                    add_triangle(vertices, top, bottom, left);
                }
            }
            float tangent = (coast1 - coast0) / std::max(0.0001f, y1 - y0);
            Vertex top0 = make_coast_vertex(coast0, y0, cliff_height, uv_scale, 3.0f, 1.0f, -tangent, 0.12f);
            Vertex top1 = make_coast_vertex(coast1, y1, cliff_height, uv_scale, 3.0f, 1.0f, -tangent, 0.12f);
            Vertex bottom1 = make_coast_vertex(coast1, y1, 0.0f, uv_scale, 3.0f, 1.0f, -tangent, 0.12f);
            Vertex bottom0 = make_coast_vertex(coast0, y0, 0.0f, uv_scale, 3.0f, 1.0f, -tangent, 0.12f);
            // Cliff material coordinates follow the wall itself: U advances
            // along the shared contour and V descends from rim to waterline.
            top0.u = y0 * 0.55f; top0.v = 0.0f;
            top1.u = y1 * 0.55f; top1.v = 0.0f;
            bottom1.u = y1 * 0.55f; bottom1.v = 0.22f;
            bottom0.u = y0 * 0.55f; bottom0.v = 0.22f;
            add_triangle(vertices, top0, top1, bottom1);
            add_triangle(vertices, top0, bottom1, bottom0);
        }
        // Seafloor, water surface, and foam are distinct meshes. All begin at
        // the same coastline coordinates used by the beach or cliff above.
        float shallow_high = integrated_shore ? 0.0f : -8.0f;
        float shallow_low = integrated_shore ? -10.0f : -8.0f;
        add_coast_strip(vertices, y0, y1, coast0, coast1,
                        scene_world_width, scene_world_width,
                        shallow_high, shallow_high, shallow_low, shallow_low,
                        uv_scale, 4.0f, 0.0f, 0.0f, 1.0f, 20);
        if (water_enabled) {
            add_coast_strip(vertices, y0, y1, coast0, coast1,
                            scene_world_width, scene_world_width,
                            0.0f, 0.0f, 0.0f, 0.0f, uv_scale, 5.0f, 0.0f, 0.0f, 1.0f, 20);
        }
        if (surf_enabled) {
            float foam_variation0 = integrated_shore
                ? 0.018f * std::sin(y0 * 12.7f + 1.3f) +
                  0.007f * std::sin(y0 * 33.1f + 0.2f)
                : 0.0f;
            float foam_variation1 = integrated_shore
                ? 0.018f * std::sin(y1 * 12.7f + 1.3f) +
                  0.007f * std::sin(y1 * 33.1f + 0.2f)
                : 0.0f;
            float foam_width0 = integrated_shore ? 0.14f + foam_variation0 : 0.075f;
            float foam_width1 = integrated_shore ? 0.14f + foam_variation1 : 0.075f;
            float foam_height = integrated_shore ? 0.25f : 0.8f;
            float foam_inset0 = integrated_shore
                ? 0.010f + 0.006f * std::sin(y0 * 19.3f + 0.5f)
                : 0.0f;
            float foam_inset1 = integrated_shore
                ? 0.010f + 0.006f * std::sin(y1 * 19.3f + 0.5f)
                : 0.0f;
            float foam0 = std::min(scene_world_width, coast0 + foam_width0);
            float foam1 = std::min(scene_world_width, coast1 + foam_width1);
            add_coast_strip(vertices, y0, y1, coast0 - foam_inset0, coast1 - foam_inset1,
                            foam0, foam1,
                            foam_height, foam_height, foam_height, foam_height,
                            uv_scale, 6.0f, 0.0f, 0.0f, 1.0f,
                            integrated_shore ? 6 : 3);
        }
    }
    return true;
}

std::uint32_t feature_hash(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float feature_random(std::uint32_t value) {
    return static_cast<float>(feature_hash(value) & 0x00ffffffu) / 16777215.0f;
}

FeatureGroup const * find_feature_group(FeatureBundle const & bundle, char const * name) {
    for (FeatureGroup const & group : bundle.groups)
        if (group.name == name)
            return &group;
    return nullptr;
}

FeaturePlacement const * weighted_feature_placement(FeatureGroup const & group,
                                                     std::uint32_t seed) {
    unsigned total = 0;
    for (FeaturePlacement const & placement : group.placements)
        total += placement.count;
    if (total == 0)
        return nullptr;
    unsigned selected = feature_hash(seed) % total;
    for (FeaturePlacement const & placement : group.placements) {
        if (selected < placement.count)
            return &placement;
        selected -= placement.count;
    }
    return nullptr;
}

FeaturePlacement const * named_feature_placement(FeatureBundle const & bundle,
                                                  FeatureGroup const & group,
                                                  char const * suffix) {
    std::string expected = suffix;
    for (FeaturePlacement const & placement : group.placements) {
        std::string const & id = bundle.assets[placement.asset_index].id;
        if (placement.scale > 0.0f && id.size() >= expected.size() &&
            id.compare(id.size() - expected.size(), expected.size(), expected) == 0)
            return &placement;
    }
    return nullptr;
}

Vertex make_feature_shadow_vertex(float screen_x, float screen_y, float depth,
                                  float across, float along) {
    return Vertex{ndc_x(screen_x), ndc_y(screen_y), across, along, 1.0f,
                  0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 7.0f, 0.0f,
                  2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, depth};
}

void add_feature_instance(FeatureBundle const & bundle, FeaturePlacement const & placement,
                          float world_x, float world_y, float rotation, float scale,
                          HeightField const * authored_height, HeightField const * authored_blend,
                          bool integrated_shore,
                          std::vector<Vertex> & shadows,
                          std::vector<FeatureVertex> & output) {
    FeatureAsset const & asset = bundle.assets[placement.asset_index];
    float land_edge = coast_position(world_y, false, integrated_shore) -
                      beach_width_at(world_y, integrated_shore);
    BiqWindowTile const * biq_tile = biq_tile_at(world_x, world_y);
    float ground_height = biq_tile != nullptr
        ? biq_tile_height(*biq_tile, world_x - std::floor(world_x),
                          world_y - std::floor(world_y), authored_height, authored_blend)
        : beauty_land_height(world_x, world_y, land_edge,
                             integrated_shore ? 2.5f : 12.0f,
                             authored_height, authored_blend);
    float center_x = 0.0f;
    float center_y = 0.0f;
    if (biq_tile != nullptr) {
        float tile_center_x = coast_projection.origin_x +
            static_cast<float>(biq_tile->column + biq_tile->row) *
                coast_projection.half_width;
        float tile_center_y = coast_projection.origin_y +
            static_cast<float>(biq_tile->column - biq_tile->row) *
                coast_projection.half_height;
        float u = world_x - static_cast<float>(biq_tile->column);
        float v = world_y - static_cast<float>(biq_tile->row);
        center_x = tile_center_x + (u - v) * coast_projection.half_width;
        center_y = tile_center_y + (u + v - 1.0f) * coast_projection.half_height -
                   ground_height * coast_projection.vertical_scale;
    } else {
        center_x = coast_projection.origin_x +
                   (world_x - world_y) * coast_projection.half_width;
        center_y = coast_projection.origin_y +
                   (world_x + world_y) * coast_projection.half_height -
                   ground_height * coast_projection.vertical_scale;
    }
    float feature_height_pixels_per_tile = promotion_scene_enabled ? 150.0f : 216.0f;
    float radius = 0.0f;
    float feature_height = 0.0f;
    for (FeatureSourceVertex const & vertex : asset.vertices) {
        radius = std::max(radius, std::sqrt(vertex.position[0] * vertex.position[0] +
                                            vertex.position[1] * vertex.position[1]) * scale);
        feature_height = std::max(feature_height, vertex.position[2] * scale);
    }
    float shadow_width = std::max(4.0f, radius * coast_projection.half_width * 0.65f);
    float horizontal = std::sqrt(active_light_direction[0] * active_light_direction[0] +
                                 active_light_direction[1] * active_light_direction[1]);
    float cast_world_x = horizontal > 0.001f
        ? -active_light_direction[0] / horizontal : 0.0f;
    float cast_world_y = horizontal > 0.001f
        ? -active_light_direction[1] / horizontal : 1.0f;
    float cast_screen_x = cast_world_x - cast_world_y;
    float cast_screen_y = (cast_world_x + cast_world_y) *
                          coast_projection.half_height / coast_projection.half_width;
    float cast_length = std::sqrt(cast_screen_x * cast_screen_x +
                                  cast_screen_y * cast_screen_y);
    if (cast_length > 0.001f) {
        cast_screen_x /= cast_length;
        cast_screen_y /= cast_length;
    }
    float radial_shadow_length = shadow_width * 2.40f;
    float projected_height_ratio = 0.72f;
    float height_shadow_length = feature_height * feature_height_pixels_per_tile *
        projected_height_ratio;
    // Direction rotates with the shared light, but the canonical stylization
    // does not dramatically stretch the footprint at dawn or dusk.
    float minimum_shadow_length = shadow_width * 2.55f;
    float shadow_length = std::clamp(
        std::max(radial_shadow_length, height_shadow_length),
        minimum_shadow_length, std::min(180.0f, shadow_width * 10.0f));
    float perpendicular_x = -cast_screen_y;
    float perpendicular_y = cast_screen_x;
    float ground_base_screen_y = center_y + ground_height * coast_projection.vertical_scale;
    // Screen-space feature shadows must follow the receiver's depth gradient.
    // A single depth sampled at the feature origin is hidden by terrain when
    // the quad extends down-screen (the canonical 18:00 direction).
    auto shadow_depth_at = [&](float screen_y) {
        float projected_base_y = ground_base_screen_y + (screen_y - center_y);
        return std::max(0.005f, std::clamp(
            0.94f - projected_base_y / static_cast<float>(output_height) * 0.75f -
            ground_height * 0.0012f, 0.01f, 0.99f) - 0.004f);
    };
    float near_left_x = center_x - perpendicular_x * shadow_width * 0.42f;
    float near_left_y = center_y - perpendicular_y * shadow_width * 0.42f;
    float near_right_x = center_x + perpendicular_x * shadow_width * 0.42f;
    float near_right_y = center_y + perpendicular_y * shadow_width * 0.42f;
    float far_right_x = center_x + cast_screen_x * shadow_length +
                        perpendicular_x * shadow_width * 0.72f;
    float far_right_y = center_y + cast_screen_y * shadow_length +
                        perpendicular_y * shadow_width * 0.72f;
    float far_left_x = center_x + cast_screen_x * shadow_length -
                       perpendicular_x * shadow_width * 0.72f;
    float far_left_y = center_y + cast_screen_y * shadow_length -
                       perpendicular_y * shadow_width * 0.72f;
    Vertex shadow_near_left = make_feature_shadow_vertex(
        near_left_x, near_left_y, shadow_depth_at(near_left_y),
        0.0f, 0.0f);
    Vertex shadow_near_right = make_feature_shadow_vertex(
        near_right_x, near_right_y, shadow_depth_at(near_right_y),
        1.0f, 0.0f);
    Vertex shadow_far_right = make_feature_shadow_vertex(
        far_right_x, far_right_y, shadow_depth_at(far_right_y), 1.0f, 1.0f);
    Vertex shadow_far_left = make_feature_shadow_vertex(
        far_left_x, far_left_y, shadow_depth_at(far_left_y), 0.0f, 1.0f);
    if (!biq_scene_enabled || l13a_scene_enabled) {
        add_triangle(shadows, shadow_near_left, shadow_near_right, shadow_far_right);
        add_triangle(shadows, shadow_near_left, shadow_far_right, shadow_far_left);
    }

    float cosine = std::cos(rotation);
    float sine = std::sin(rotation);
    std::vector<FeatureVertex> transformed(asset.vertices.size());
    for (std::size_t index = 0; index < asset.vertices.size(); ++index) {
        FeatureSourceVertex const & source = asset.vertices[index];
        float local_x = (source.position[0] * cosine - source.position[1] * sine) * scale;
        float local_y = (source.position[0] * sine + source.position[1] * cosine) * scale;
        float local_z = source.position[2] * scale;
        float feature_x = world_x + local_x;
        float feature_y = world_y + local_y;
        float screen_x = biq_tile != nullptr
            ? center_x + (local_x - local_y) * coast_projection.half_width
            : coast_projection.origin_x +
              (feature_x - feature_y) * coast_projection.half_width;
        float screen_y = biq_tile != nullptr
            ? center_y + (local_x + local_y) * coast_projection.half_height -
              local_z * feature_height_pixels_per_tile
            : coast_projection.origin_y +
              (feature_x + feature_y) * coast_projection.half_height -
              ground_height * coast_projection.vertical_scale -
              local_z * feature_height_pixels_per_tile;
        float normal_x = source.normal[0] * cosine - source.normal[1] * sine;
        float normal_y = source.normal[0] * sine + source.normal[1] * cosine;
        float depth = 0.0f;
        if (biq_tile != nullptr) {
            float base_screen_y = center_y + ground_height * coast_projection.vertical_scale +
                                  (local_x + local_y) * coast_projection.half_height;
            float feature_height_tiles = local_z * feature_height_pixels_per_tile /
                                         coast_projection.vertical_scale;
            depth = std::clamp(
                0.94f - base_screen_y / static_cast<float>(output_height) * 0.75f -
                (ground_height + feature_height_tiles) * 0.0012f, 0.01f, 0.99f);
        } else {
            depth = std::clamp(0.88f - (feature_x + feature_y) * 0.18f - local_z * 0.04f,
                               0.02f, 0.98f);
        }
        transformed[index] = FeatureVertex{
            ndc_x(screen_x), ndc_y(screen_y), depth, source.uv[0], source.uv[1],
            normal_x, normal_y, source.normal[2], static_cast<float>(asset.texture_index)};
    }
    for (std::uint32_t index : asset.indices)
        output.push_back(transformed[index]);
}

bool add_feature_group(FeatureBundle const & bundle, FeatureGroup const & group,
                       unsigned instance_count, float x_min, float x_max,
                       float y_min, float y_max,
                       char const * const * anchors, unsigned anchor_count,
                       std::uint32_t seed, HeightField const * authored_height,
                       HeightField const * authored_blend, bool integrated_shore,
                       std::vector<Vertex> & shadows,
                       std::vector<FeatureVertex> & output) {
    if (group.placements.empty())
        return false;
    constexpr float two_pi = 6.28318530718f;
    float scene_feature_scale = promotion_scene_enabled ? 0.55f : 0.82f;
    bool dense_biq_canopy = biq_scene_enabled && promotion_scene_enabled &&
        (group.name == "forest" || group.name == "jungle");
    if (dense_biq_canopy)
        scene_feature_scale = group.name == "forest" ? 0.42f : 0.40f;
    for (unsigned instance = 0; instance < instance_count; ++instance) {
        FeaturePlacement const * placement = instance < anchor_count
            ? named_feature_placement(bundle, group, anchors[instance])
            : weighted_feature_placement(group, seed + instance * 31u);
        if (placement == nullptr || placement->scale <= 0.0f)
            continue;
        float x_t = feature_random(seed + instance * 43u + 11u);
        float y_t = std::fmod(0.17f + static_cast<float>(instance) * 0.61803398875f +
                              feature_random(seed + instance * 17u) * 0.19f, 1.0f);
        if (dense_biq_canopy) {
            // Civ VI's placement records describe overlapping stands, not a
            // handful of isolated specimens. More small bodies on a jittered
            // lattice preserve the authored tree-to-mountain scale and an
            // irregular silhouette while preventing holes in the canopy core.
            // Counts are perfect squares so neither axis gets a partial row.
            unsigned grid_side = group.name == "forest" ? 6u : 7u;
            float jitter_x = feature_random(seed + instance * 103u + 59u) - 0.5f;
            float jitter_y = feature_random(seed + instance * 107u + 61u) - 0.5f;
            x_t = (static_cast<float>(instance % grid_side) + 0.5f + jitter_x * 0.68f) /
                  static_cast<float>(grid_side);
            y_t = (static_cast<float>(instance / grid_side) + 0.5f + jitter_y * 0.68f) /
                  static_cast<float>(grid_side);
        }
        float world_y = y_min + (y_max - y_min) * y_t;
        float land_edge = coast_position(world_y, false, integrated_shore) -
                          beach_width_at(world_y, integrated_shore);
        float x_floor = std::max(0.07f, x_min);
        float x_limit = biq_scene_enabled ? x_max : std::min(x_max, land_edge - 0.11f);
        if (x_limit <= x_floor)
            continue;
        float world_x = x_floor;
        unsigned attempts = promotion_scene_enabled ? 16u : 6u;
        for (unsigned attempt = 0; attempt < attempts; ++attempt) {
            float attempt_x = attempt == 0u ? x_t :
                feature_random(seed + instance * 43u + attempt * 101u + 11u);
            world_x = x_floor + (x_limit - x_floor) * attempt_x;
            BiqWindowTile const * tile = biq_tile_at(world_x, world_y);
            float ground_height = tile != nullptr
                ? biq_tile_height(*tile, world_x - std::floor(world_x),
                                  world_y - std::floor(world_y),
                                  authored_height, authored_blend)
                : beauty_land_height(world_x, world_y, land_edge,
                                     integrated_shore ? 2.5f : 12.0f,
                                     authored_height, authored_blend);
            if (ground_height < (promotion_scene_enabled ? 30.0f : 40.0f))
                break;
        }
        float variation = (feature_random(seed + instance * 71u + 23u) * 2.0f - 1.0f) *
                          placement->scale_variation;
        float scale = placement->scale * (1.0f + variation) * scene_feature_scale;
        float rotation = feature_random(seed + instance * 97u + 47u) * two_pi;
        add_feature_instance(bundle, *placement, world_x, world_y, rotation, scale,
                             authored_height, authored_blend, integrated_shore,
                             shadows, output);
    }
    return true;
}

bool add_feature_scene(FeatureBundle const & bundle, HeightField const * authored_height,
                       HeightField const * authored_blend, bool integrated_shore,
                       std::vector<Vertex> & shadows,
                       std::vector<FeatureVertex> & output) {
    FeatureGroup const * forest = find_feature_group(bundle, "forest");
    FeatureGroup const * jungle = find_feature_group(bundle, "jungle");
    if (forest == nullptr || jungle == nullptr)
        return false;
    // Alternate packs may expose warm broadleaf bodies alongside retained
    // pine variants. Prefer that source-authored subset for ordinary Civ III
    // forest cells; cold-biome selection can be added once the scene contract
    // carries climate state rather than guessing it from screen position.
    FeatureGroup broadleaf_forest;
    broadleaf_forest.name = "forest";
    for (FeaturePlacement const & placement : forest->placements)
        if (bundle.assets[placement.asset_index].id.find("feature/forest/leafy") !=
            std::string::npos)
            broadleaf_forest.placements.push_back(placement);
    if (!broadleaf_forest.placements.empty())
        forest = &broadleaf_forest;
    char const * forest_anchors[] = {"pine_01", "pine_clump_01", "shrub_01"};
    char const * jungle_anchors[] = {
        "grass_04", "palm_01", "palm_02", "plant_01", "plant_02", "plant_03"};
    if (promotion_scene_enabled) {
        bool ok = true;
        constexpr float inset = 0.07f;
        constexpr unsigned forest_density = 11;
        constexpr unsigned jungle_density = 20;
        constexpr unsigned biq_forest_density = 36;
        constexpr unsigned biq_jungle_density = 49;
        if (biq_scene_enabled) {
            for (BiqWindowTile const & tile : biq_window.tiles) {
                FeatureGroup const * group = tile.real == 7 ? forest :
                    (tile.real == 8 ? jungle : nullptr);
                if (group == nullptr)
                    continue;
                char const * const * anchors = tile.real == 7 ? forest_anchors : jungle_anchors;
                unsigned anchor_count = tile.real == 7 ? 3u : 6u;
                unsigned density = tile.real == 7 ? biq_forest_density : biq_jungle_density;
                std::uint32_t seed = static_cast<std::uint32_t>(tile.source_x * 0x193u) ^
                                     static_cast<std::uint32_t>(tile.source_y * 0x217u);
                ok = add_feature_group(bundle, *group, density,
                                       static_cast<float>(tile.column) + inset,
                                       static_cast<float>(tile.column + 1) - inset,
                                       static_cast<float>(tile.row) + inset,
                                       static_cast<float>(tile.row + 1) - inset,
                                       anchors, anchor_count, seed,
                                       authored_height, authored_blend, integrated_shore,
                                       shadows, output);
                if (!ok)
                    return false;
            }
            return ok;
        }
        // L9 uses six forest and six jungle cells. The 96-tile L10 fixture
        // extends each block to eight cells, preserving a full empty column
        // before the central relief so canopies cannot cross mountains.
        unsigned cells_per_biome = l10_scene_enabled ? 8u : 6u;
        unsigned jungle_first_row = l10_scene_enabled ? 4u : 3u;
        for (unsigned index = 0; ok && index < cells_per_biome; ++index) {
            float x = static_cast<float>(index % 2u);
            float y = static_cast<float>(index / 2u);
            ok = add_feature_group(bundle, *forest, forest_density,
                                   x + inset, x + 1.0f - inset,
                                   y + inset, y + 1.0f - inset,
                                   forest_anchors, 3, 0x51a7u + index * 0x193u,
                                   authored_height, authored_blend, integrated_shore,
                                   shadows, output);
        }
        for (unsigned index = 0; ok && index < cells_per_biome; ++index) {
            float x = static_cast<float>(index % 2u);
            float y = static_cast<float>(jungle_first_row + index / 2u);
            ok = add_feature_group(bundle, *jungle, jungle_density,
                                   x + inset, x + 1.0f - inset,
                                   y + inset, y + 1.0f - inset,
                                   jungle_anchors, 6, 0x9e37u + index * 0x217u,
                                   authored_height, authored_blend, integrated_shore,
                                   shadows, output);
        }
        return ok;
    }
    return add_feature_group(bundle, *forest, 4, 0.07f, scene_world_width,
                             0.45f, 1.05f,
                             forest_anchors, 3, 0x51a7u,
                             authored_height, authored_blend, integrated_shore,
                             shadows, output) &&
           add_feature_group(bundle, *jungle, 8, 0.07f, scene_world_width,
                             1.05f, 1.92f,
                             jungle_anchors, 6, 0x9e37u,
                             authored_height, authored_blend, integrated_shore,
                             shadows, output);
}

bool add_river_rock_scene(FeatureBundle const & bundle,
                          HeightField const * authored_height,
                          HeightField const * authored_blend,
                          std::vector<Vertex> & shadows,
                          std::vector<FeatureVertex> & output) {
    FeatureGroup const * group = find_feature_group(bundle, "river_rock");
    if (group == nullptr || group->placements.empty())
        return false;
    constexpr float two_pi = 6.28318530718f;
    for (BiqWindowTile const & tile : biq_window.tiles) {
        struct EdgePlacement {
            unsigned bit;
            int neighbor_column;
            int neighbor_row;
            bool north_edge;
        };
        EdgePlacement edges[] = {
            {2u, tile.column, tile.row + 1, true},
            {8u, tile.column + 1, tile.row, false},
        };
        for (EdgePlacement const & edge : edges) {
            if ((tile.river_mask & edge.bit) == 0)
                continue;
            std::uint32_t seed = feature_hash(
                static_cast<std::uint32_t>(tile.source_x + 4096) * 0x193u ^
                static_cast<std::uint32_t>(tile.source_y + 4096) * 0x217u ^ edge.bit);
            // These are authored clutter bodies, not an outline. Sparse,
            // stable placement keeps the watercourse readable.
            if ((seed % 3u) != 0u)
                continue;
            FeaturePlacement const & placement =
                group->placements[(seed >> 5) % group->placements.size()];
            float along = 0.28f + feature_random(seed ^ 0x73a52u) * 0.44f;
            BiqWindowTile const * owner = &tile;
            float local_u = edge.north_edge ? along : 0.975f;
            float local_v = edge.north_edge ? 0.025f : along;
            BiqWindowTile const * neighbor = biq_tile_at(
                edge.neighbor_column, edge.neighbor_row);
            if (is_water_terrain(owner->base) && neighbor != nullptr &&
                !is_water_terrain(neighbor->base)) {
                owner = neighbor;
                local_u = edge.north_edge ? along : 0.025f;
                local_v = edge.north_edge ? 0.975f : along;
            }
            if (is_water_terrain(owner->base))
                continue;
            float scale = 0.155f + feature_random(seed ^ 0x91c37u) * 0.070f;
            float rotation = feature_random(seed ^ 0x4ad91u) * two_pi;
            add_feature_instance(bundle, placement,
                                 static_cast<float>(owner->column) + local_u,
                                 static_cast<float>(owner->row) + local_v,
                                 rotation, scale, authored_height, authored_blend,
                                 true, shadows, output);
        }
    }
    return true;
}

bool add_route_bridge_scene(FeatureBundle const & bundle,
                            RoadScenario const & scenario,
                            bool railroad,
                            HeightField const * authored_height,
                            HeightField const * authored_blend,
                            int style_override,
                            std::vector<Vertex> & shadows,
                            std::vector<FeatureVertex> & output) {
    for (RoadEdge const & edge : scenario.edges) {
        if (!edge.bridge || edge.wraps)
            continue;
        unsigned style = style_override >= 0
            ? static_cast<unsigned>(style_override) : edge.style;
        char const * bridge_style = railroad ? "railroad" :
                                    (style >= 3 ? "modern" :
                                    (style >= 2 ? "industrial" : "medieval"));
        char group_name[64] = {};
        sprintf_s(group_name, "bridge_%s_%s", bridge_style,
                  edge.pillaged ? "pillaged" : "normal");
        FeatureGroup const * group = find_feature_group(bundle, group_name);
        if (group == nullptr || group->placements.size() != 1)
            return false;
        FeaturePlacement const & placement = group->placements.front();
        float x0 = static_cast<float>(edge.x0) + 0.5f;
        float y0 = static_cast<float>(edge.y0) + 0.5f;
        float x1 = static_cast<float>(edge.x1) + 0.5f;
        float y1 = static_cast<float>(edge.y1) + 0.5f;
        float rotation = std::atan2(y1 - y0, x1 - x0);
        add_feature_instance(bundle, placement,
                             (x0 + x1) * 0.5f, (y0 + y1) * 0.5f,
                             rotation, placement.scale,
                             authored_height, authored_blend, true,
                             shadows, output);
    }
    return true;
}

bool add_road_bridge_scene(FeatureBundle const & bundle,
                           HeightField const * authored_height,
                           HeightField const * authored_blend,
                           int style_override,
                           std::vector<Vertex> & shadows,
                           std::vector<FeatureVertex> & output) {
    return add_route_bridge_scene(bundle, road_scenario, false,
                                  authored_height, authored_blend, style_override,
                                  shadows, output);
}

bool add_railroad_bridge_scene(FeatureBundle const & bundle,
                               HeightField const * authored_height,
                               HeightField const * authored_blend,
                               std::vector<Vertex> & shadows,
                               std::vector<FeatureVertex> & output) {
    return add_route_bridge_scene(bundle, railroad_scenario, true,
                                  authored_height, authored_blend, 4,
                                  shadows, output);
}

bool add_resource_scene(FeatureBundle const & bundle,
                        HeightField const * authored_height,
                        HeightField const * authored_blend,
                        std::vector<Vertex> & shadows,
                        std::vector<FeatureVertex> & output) {
    constexpr char const * names[] = {
        "horses", "iron", "uranium", "gold", "dye", "wheat", "cattle", "fish"
    };
    constexpr float two_pi = 6.28318530718f;
    std::vector<Vertex> submerged_shadows;
    for (ResourceInstance const & instance : resource_scenario.instances) {
        if (instance.visible == 0u)
            continue;
        FeatureGroup const * group = find_feature_group(bundle, names[instance.resource]);
        if (group == nullptr || group->placements.empty())
            return false;
        FeaturePlacement const & placement = group->placements.front();
        BiqWindowTile const * tile = biq_tile_at(instance.column, instance.row);
        if (tile == nullptr || ((instance.resource == 7u) != (tile->base >= 11)))
            return false;
        unsigned count = std::max(1u, placement.count);
        for (unsigned body = 0; body < count; ++body) {
            float angle = two_pi * (static_cast<float>(body) / static_cast<float>(count) +
                feature_random(instance.variant * 101u + body * 37u) * 0.11f);
            float ring = count == 1u ? 0.0f :
                (body == 0u ? 0.045f : 0.10f + 0.055f * static_cast<float>((body - 1u) % 3u));
            float world_x = static_cast<float>(instance.column) + 0.50f +
                std::cos(angle) * ring;
            float world_y = static_cast<float>(instance.row) + 0.50f +
                std::sin(angle) * ring * 0.78f;
            float variation = (feature_random(instance.variant * 59u + body * 71u + 13u) *
                2.0f - 1.0f) * placement.scale_variation;
            float scale = placement.scale * (1.0f + variation) * 0.78f;
            float rotation = feature_random(instance.variant * 83u + body * 97u + 29u) * two_pi;
            add_feature_instance(bundle, placement, world_x, world_y, rotation, scale,
                                 authored_height, authored_blend, true,
                                 instance.resource == 7u ? submerged_shadows : shadows,
                                 output);
        }
    }
    return true;
}

bool add_city_scene(FeatureBundle const & city_bundle,
                    FeatureBundle const & wall_bundle,
                    HeightField const * authored_height,
                    HeightField const * authored_blend,
                    std::vector<Vertex> & shadows,
                    std::vector<FeatureVertex> & city_output,
                    std::vector<FeatureVertex> & wall_output) {
    constexpr char const * era_names[] = {"ancient", "medieval", "industrial", "modern"};
    constexpr char const * wall_names[] = {"wall_ancient", "wall_medieval", "wall_industrial"};
    constexpr unsigned counts[] = {4u, 7u, 11u};
    constexpr float radii[] = {0.25f, 0.33f, 0.41f};
    constexpr float size_scales[] = {0.92f, 1.00f, 1.08f};
    constexpr float golden_angle = 2.39996322973f;
    for (CityInstance const & instance : city_scenario.instances) {
        if (instance.visible == 0u)
            continue;
        FeatureGroup const * group = find_feature_group(city_bundle, era_names[instance.era]);
        BiqWindowTile const * tile = biq_tile_at(instance.column, instance.row);
        if (group == nullptr || group->placements.empty() || tile == nullptr || tile->base >= 11)
            return false;
        unsigned component_count = counts[instance.size];
        for (unsigned slot = 0; slot < component_count; ++slot) {
            FeaturePlacement const & placement = group->placements[
                (instance.culture + instance.variant + slot) % group->placements.size()];
            float angle = static_cast<float>(slot) * golden_angle +
                feature_random(instance.variant * 53u + instance.culture * 19u) * 0.72f;
            float radius = slot == 0u ? 0.0f :
                radii[instance.size] * std::sqrt(static_cast<float>(slot) /
                                                 static_cast<float>(component_count - 1u));
            float world_x = static_cast<float>(instance.column) + 0.50f + std::cos(angle) * radius;
            float world_y = static_cast<float>(instance.row) + 0.50f + std::sin(angle) * radius * 0.78f;
            float scale = placement.scale * size_scales[instance.size] *
                (slot == 0u && instance.capital ? 1.30f : 1.0f);
            float rotation = angle + 0.55f;
            std::size_t first = city_output.size();
            add_feature_instance(city_bundle, placement, world_x, world_y, rotation, scale,
                                 authored_height, authored_blend, true, shadows, city_output);
            float owner_code = 0.08f * static_cast<float>(instance.owner + 1u);
            for (std::size_t index = first; index < city_output.size(); ++index)
                city_output[index].material_index += owner_code;
        }
        if (instance.walls != 0u) {
            FeatureGroup const * walls = find_feature_group(
                wall_bundle, wall_names[std::min(instance.era, 2u)]);
            if (walls == nullptr || walls->placements.empty())
                return false;
            FeaturePlacement const & wall = walls->placements.front();
            constexpr float offsets[4][3] = {
                {-0.29f, 0.00f, 0.785398163f},
                {0.29f, 0.00f, 0.785398163f},
                {0.00f, -0.23f, -0.785398163f},
                {0.00f, 0.23f, -0.785398163f},
            };
            for (auto const & offset : offsets) {
                std::size_t first = wall_output.size();
                add_feature_instance(
                    wall_bundle, wall,
                    static_cast<float>(instance.column) + 0.50f + offset[0],
                    static_cast<float>(instance.row) + 0.50f + offset[1],
                    offset[2], wall.scale * (instance.size == 0u ? 0.82f : 1.0f),
                    authored_height, authored_blend, true, shadows, wall_output);
                float owner_code = 0.08f * static_cast<float>(instance.owner + 1u);
                for (std::size_t index = first; index < wall_output.size(); ++index)
                    wall_output[index].material_index += owner_code;
            }
        }
    }
    return true;
}

bool add_mine_scene(FeatureBundle const & bundle,
                    HeightField const * authored_height,
                    HeightField const * authored_blend,
                    std::vector<Vertex> & shadows,
                    std::vector<FeatureVertex> & output) {
    std::vector<Vertex> discarded_child_shadows;
    for (MineInstance const & instance : mine_scenario.instances) {
        if (instance.visible == 0u)
            continue;
        unsigned family = instance.era < 2u ? 0u : 1u;
        unsigned group_index = family * 3u + instance.variant;
        std::string group_name = "mine_" + std::to_string(group_index);
        FeatureGroup const * group = find_feature_group(bundle, group_name.c_str());
        BiqWindowTile const * tile = biq_tile_at(instance.column, instance.row);
        if (group == nullptr || group->placements.empty() || tile == nullptr || tile->base >= 11)
            return false;
        float rotation = feature_random(instance.column * 71u + instance.row * 113u +
                                        instance.era * 29u) * 0.48f - 0.24f;
        for (std::size_t part = 0; part < group->placements.size(); ++part) {
            FeaturePlacement const & placement = group->placements[part];
            std::size_t first = output.size();
            add_feature_instance(bundle, placement,
                                 static_cast<float>(instance.column) + 0.50f,
                                 static_cast<float>(instance.row) + 0.50f,
                                 rotation, placement.scale,
                                 authored_height, authored_blend, true,
                                 part == 0 ? shadows : discarded_child_shadows, output);
            FeatureAsset const & asset = bundle.assets[placement.asset_index];
            unsigned emissive_code = 0u;
            std::size_t marker = asset.id.rfind(":e");
            if (marker != std::string::npos)
                emissive_code = static_cast<unsigned>(std::strtoul(
                    asset.id.c_str() + marker + 2u, nullptr, 10));
            float material_marker = 0.01f * static_cast<float>(emissive_code + 1u);
            for (std::size_t index = first; index < output.size(); ++index)
                output[index].material_index += material_marker;
        }
    }
    return true;
}

bool write_bmp(char const * path, D3D11_MAPPED_SUBRESOURCE const & mapped, unsigned downsample) {
    FILE * file = nullptr;
    if (fopen_s(&file, path, "wb") != 0 || file == nullptr)
        return false;
    unsigned target_width = output_width / downsample;
    unsigned target_height = output_height / downsample;
    BITMAPFILEHEADER file_header = {};
    BITMAPINFOHEADER info = {};
    unsigned row_bytes = target_width * 4;
    file_header.bfType = 0x4d42;
    file_header.bfOffBits = sizeof(file_header) + sizeof(info);
    file_header.bfSize = file_header.bfOffBits + row_bytes * target_height;
    info.biSize = sizeof(info);
    info.biWidth = target_width;
    info.biHeight = -static_cast<LONG>(target_height);
    info.biPlanes = 1;
    info.biBitCount = 32;
    info.biCompression = BI_RGB;
    bool ok = std::fwrite(&file_header, sizeof(file_header), 1, file) == 1 &&
              std::fwrite(&info, sizeof(info), 1, file) == 1;
    auto const * source = static_cast<std::uint8_t const *>(mapped.pData);
    std::vector<std::uint8_t> reduced_row(row_bytes);
    for (unsigned row = 0; ok && row < target_height; ++row) {
        if (downsample == 1) {
            ok = std::fwrite(source + static_cast<std::size_t>(row) * mapped.RowPitch,
                             row_bytes, 1, file) == 1;
            continue;
        }
        for (unsigned column = 0; column < target_width; ++column) {
            for (unsigned channel = 0; channel < 4; ++channel) {
                unsigned sum = 0;
                for (unsigned sample_y = 0; sample_y < downsample; ++sample_y) {
                    auto const * source_row = source +
                        static_cast<std::size_t>(row * downsample + sample_y) * mapped.RowPitch;
                    for (unsigned sample_x = 0; sample_x < downsample; ++sample_x)
                        sum += source_row[(column * downsample + sample_x) * 4 + channel];
                }
                reduced_row[column * 4 + channel] = static_cast<std::uint8_t>(
                    sum / (downsample * downsample));
            }
        }
        ok = std::fwrite(reduced_row.data(), row_bytes, 1, file) == 1;
    }
    std::fclose(file);
    return ok;
}

bool compile_shader(char const * path, char const * entry, char const * target, ID3DBlob ** output) {
    ID3DBlob * errors = nullptr;
    std::wstring wide_path = widen(path);
    HRESULT hr = D3DCompileFromFile(wide_path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                    entry, target, D3DCOMPILE_ENABLE_STRICTNESS, 0, output, &errors);
    if (errors != nullptr) {
        std::fwrite(errors->GetBufferPointer(), errors->GetBufferSize(), 1, stderr);
        release(errors);
    }
    return SUCCEEDED(hr);
}

std::string join(std::string const & root, char const * relative) {
    return root + "\\" + relative;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 26) {
        std::fprintf(stderr,
            "usage: terrain_lab <pack-root> <shader.hlsl> <output.bmp> "
            "<albedo|material|relief|shadow|hill|mountain|coast_beach|coast_cliff|beauty|"
            "beauty_no_relief|beauty_no_water|beauty_thumbnail|beauty_vegetation|"
            "beauty_vegetation_only|beauty_no_vegetation|beauty_vegetation_thumbnail|"
            "beauty_shore|beauty_shore_no_vegetation|beauty_shore_no_water|"
            "beauty_shore_no_surf|beauty_shore_thumbnail|beauty_promotion|"
            "beauty_promotion_no_vegetation|beauty_promotion_no_water|"
            "beauty_promotion_no_surf|beauty_promotion_thumbnail|beauty_dunes|"
            "beauty_dunes_no_dunes|beauty_dunes_only|beauty_dunes_thumbnail|beauty_marsh|"
            "beauty_marsh_no_marsh|beauty_marsh_only|beauty_marsh_thumbnail|beauty_volcano|"
            "beauty_volcano_no_volcano|beauty_volcano_only|beauty_volcano_thumbnail|beauty_rivers|"
            "beauty_rivers_no_rivers|beauty_rivers_only|beauty_rivers_thumbnail|"
            "beauty_lighting_noon|beauty_lighting_sunset|beauty_lighting_midnight|"
            "beauty_lighting_sunrise|beauty_lighting_noon_zoom2|"
            "beauty_lighting_sunset_zoom2|beauty_lighting_midnight_zoom2|"
            "beauty_lighting_sunrise_zoom2|beauty_roads|beauty_roads_zoom2|"
            "beauty_roads_no_roads|beauty_roads_only|beauty_roads_styles|"
            "beauty_railroads|beauty_railroads_zoom2|beauty_railroads_no_railroads|"
            "beauty_railroads_only|beauty_railroads_crossings|"
            "beauty_resources|beauty_resources_zoom2|beauty_resources_no_resources|"
            "beauty_resources_only|beauty_resources_hidden|"
            "beauty_cities_noon|beauty_cities_night|beauty_cities_zoom2|"
            "beauty_cities_no_cities|beauty_cities_only|beauty_mines_noon|"
            "beauty_mines_night|beauty_mines_zoom2|beauty_mines_no_mines|"
            "beauty_mines_only> "
            "[uv-scale=0.26] [normal-strength=4.0] [exposure=1.0] [relief-height=72] "
            "[hill-uv-scale=0.085] [vegetation-pack-root] [decal-pack-root] [biq-window.csv] "
            "[terrain-elements-pack-root] [shore-feature-pack-root] "
            "[route-style-pack-root] [road-scenario.csv] [route-doodad-pack-root] "
            "[railroad-scenario.csv]\n");
        return 2;
    }
    bool albedo_mode = std::strcmp(argv[4], "albedo") == 0;
    bool material_mode = std::strcmp(argv[4], "material") == 0;
    bool relief_mode = std::strcmp(argv[4], "relief") == 0;
    bool shadow_mode = std::strcmp(argv[4], "shadow") == 0;
    bool hill_mode = std::strcmp(argv[4], "hill") == 0;
    bool mountain_mode = std::strcmp(argv[4], "mountain") == 0;
    bool coast_beach_mode = std::strcmp(argv[4], "coast_beach") == 0;
    bool coast_cliff_mode = std::strcmp(argv[4], "coast_cliff") == 0;
    bool beauty_no_relief_mode = std::strcmp(argv[4], "beauty_no_relief") == 0;
    bool beauty_no_water_mode = std::strcmp(argv[4], "beauty_no_water") == 0;
    bool beauty_thumbnail_mode = std::strcmp(argv[4], "beauty_thumbnail") == 0;
    bool beauty_vegetation_mode = std::strcmp(argv[4], "beauty_vegetation") == 0;
    bool beauty_vegetation_only_mode = std::strcmp(argv[4], "beauty_vegetation_only") == 0;
    bool beauty_no_vegetation_mode = std::strcmp(argv[4], "beauty_no_vegetation") == 0;
    bool beauty_vegetation_thumbnail_mode =
        std::strcmp(argv[4], "beauty_vegetation_thumbnail") == 0;
    bool beauty_shore_mode = std::strcmp(argv[4], "beauty_shore") == 0;
    bool beauty_shore_no_vegetation_mode =
        std::strcmp(argv[4], "beauty_shore_no_vegetation") == 0;
    bool beauty_shore_no_water_mode = std::strcmp(argv[4], "beauty_shore_no_water") == 0;
    bool beauty_shore_no_surf_mode = std::strcmp(argv[4], "beauty_shore_no_surf") == 0;
    bool beauty_shore_thumbnail_mode = std::strcmp(argv[4], "beauty_shore_thumbnail") == 0;
    bool beauty_promotion_mode = std::strcmp(argv[4], "beauty_promotion") == 0;
    bool beauty_promotion_no_vegetation_mode =
        std::strcmp(argv[4], "beauty_promotion_no_vegetation") == 0;
    bool beauty_promotion_no_water_mode =
        std::strcmp(argv[4], "beauty_promotion_no_water") == 0;
    bool beauty_promotion_no_surf_mode =
        std::strcmp(argv[4], "beauty_promotion_no_surf") == 0;
    bool beauty_promotion_thumbnail_mode =
        std::strcmp(argv[4], "beauty_promotion_thumbnail") == 0;
    bool beauty_dunes_mode = std::strcmp(argv[4], "beauty_dunes") == 0;
    bool beauty_dunes_no_dunes_mode =
        std::strcmp(argv[4], "beauty_dunes_no_dunes") == 0;
    bool beauty_dunes_only_mode = std::strcmp(argv[4], "beauty_dunes_only") == 0;
    bool beauty_dunes_thumbnail_mode =
        std::strcmp(argv[4], "beauty_dunes_thumbnail") == 0;
    bool beauty_marsh_mode = std::strcmp(argv[4], "beauty_marsh") == 0;
    bool beauty_marsh_no_marsh_mode =
        std::strcmp(argv[4], "beauty_marsh_no_marsh") == 0;
    bool beauty_marsh_only_mode = std::strcmp(argv[4], "beauty_marsh_only") == 0;
    bool beauty_marsh_thumbnail_mode =
        std::strcmp(argv[4], "beauty_marsh_thumbnail") == 0;
    bool beauty_volcano_mode = std::strcmp(argv[4], "beauty_volcano") == 0;
    bool beauty_volcano_no_volcano_mode =
        std::strcmp(argv[4], "beauty_volcano_no_volcano") == 0;
    bool beauty_volcano_only_mode = std::strcmp(argv[4], "beauty_volcano_only") == 0;
    bool beauty_volcano_thumbnail_mode =
        std::strcmp(argv[4], "beauty_volcano_thumbnail") == 0;
    bool beauty_rivers_mode = std::strcmp(argv[4], "beauty_rivers") == 0;
    bool beauty_rivers_no_rivers_mode =
        std::strcmp(argv[4], "beauty_rivers_no_rivers") == 0;
    bool beauty_rivers_only_mode = std::strcmp(argv[4], "beauty_rivers_only") == 0;
    bool beauty_rivers_thumbnail_mode =
        std::strcmp(argv[4], "beauty_rivers_thumbnail") == 0;
    bool beauty_lighting_noon_mode =
        std::strcmp(argv[4], "beauty_lighting_noon") == 0;
    bool beauty_lighting_sunset_mode =
        std::strcmp(argv[4], "beauty_lighting_sunset") == 0;
    bool beauty_lighting_midnight_mode =
        std::strcmp(argv[4], "beauty_lighting_midnight") == 0;
    bool beauty_lighting_sunrise_mode =
        std::strcmp(argv[4], "beauty_lighting_sunrise") == 0;
    bool beauty_lighting_noon_zoom2_mode =
        std::strcmp(argv[4], "beauty_lighting_noon_zoom2") == 0;
    bool beauty_lighting_sunset_zoom2_mode =
        std::strcmp(argv[4], "beauty_lighting_sunset_zoom2") == 0;
    bool beauty_lighting_midnight_zoom2_mode =
        std::strcmp(argv[4], "beauty_lighting_midnight_zoom2") == 0;
    bool beauty_lighting_sunrise_zoom2_mode =
        std::strcmp(argv[4], "beauty_lighting_sunrise_zoom2") == 0;
    bool beauty_roads_mode = std::strcmp(argv[4], "beauty_roads") == 0;
    bool beauty_roads_zoom2_mode = std::strcmp(argv[4], "beauty_roads_zoom2") == 0;
    bool beauty_roads_no_roads_mode =
        std::strcmp(argv[4], "beauty_roads_no_roads") == 0;
    bool beauty_roads_only_mode = std::strcmp(argv[4], "beauty_roads_only") == 0;
    bool beauty_roads_styles_mode = std::strcmp(argv[4], "beauty_roads_styles") == 0;
    bool beauty_railroads_mode = std::strcmp(argv[4], "beauty_railroads") == 0;
    bool beauty_railroads_zoom2_mode =
        std::strcmp(argv[4], "beauty_railroads_zoom2") == 0;
    bool beauty_railroads_no_railroads_mode =
        std::strcmp(argv[4], "beauty_railroads_no_railroads") == 0;
    bool beauty_railroads_only_mode =
        std::strcmp(argv[4], "beauty_railroads_only") == 0;
    bool beauty_railroads_crossings_mode =
        std::strcmp(argv[4], "beauty_railroads_crossings") == 0;
    bool beauty_resources_mode = std::strcmp(argv[4], "beauty_resources") == 0;
    bool beauty_resources_zoom2_mode =
        std::strcmp(argv[4], "beauty_resources_zoom2") == 0;
    bool beauty_resources_no_resources_mode =
        std::strcmp(argv[4], "beauty_resources_no_resources") == 0;
    bool beauty_resources_only_mode =
        std::strcmp(argv[4], "beauty_resources_only") == 0;
    bool beauty_resources_hidden_mode =
        std::strcmp(argv[4], "beauty_resources_hidden") == 0;
    bool beauty_cities_noon_mode = std::strcmp(argv[4], "beauty_cities_noon") == 0;
    bool beauty_cities_night_mode = std::strcmp(argv[4], "beauty_cities_night") == 0;
    bool beauty_cities_zoom2_mode = std::strcmp(argv[4], "beauty_cities_zoom2") == 0;
    bool beauty_cities_no_cities_mode =
        std::strcmp(argv[4], "beauty_cities_no_cities") == 0;
    bool beauty_cities_only_mode = std::strcmp(argv[4], "beauty_cities_only") == 0;
    bool beauty_mines_noon_mode = std::strcmp(argv[4], "beauty_mines_noon") == 0;
    bool beauty_mines_night_mode = std::strcmp(argv[4], "beauty_mines_night") == 0;
    bool beauty_mines_zoom2_mode = std::strcmp(argv[4], "beauty_mines_zoom2") == 0;
    bool beauty_mines_no_mines_mode =
        std::strcmp(argv[4], "beauty_mines_no_mines") == 0;
    bool beauty_mines_only_mode = std::strcmp(argv[4], "beauty_mines_only") == 0;
    bool l18_mode = beauty_mines_noon_mode || beauty_mines_night_mode ||
        beauty_mines_zoom2_mode || beauty_mines_no_mines_mode || beauty_mines_only_mode;
    bool l17_mode = beauty_cities_noon_mode || beauty_cities_night_mode ||
        beauty_cities_zoom2_mode || beauty_cities_no_cities_mode ||
        beauty_cities_only_mode || l18_mode;
    bool l16_mode = beauty_resources_mode || beauty_resources_zoom2_mode ||
        beauty_resources_no_resources_mode || beauty_resources_only_mode ||
        beauty_resources_hidden_mode || l17_mode;
    bool l15_mode = beauty_railroads_mode || beauty_railroads_zoom2_mode ||
        beauty_railroads_no_railroads_mode || beauty_railroads_only_mode ||
        beauty_railroads_crossings_mode || l16_mode;
    bool l14_mode = beauty_roads_mode || beauty_roads_zoom2_mode ||
        beauty_roads_no_roads_mode || beauty_roads_only_mode ||
        beauty_roads_styles_mode || l15_mode;
    bool l13a_zoom2_mode = beauty_lighting_noon_zoom2_mode ||
        beauty_lighting_sunset_zoom2_mode || beauty_lighting_midnight_zoom2_mode ||
        beauty_lighting_sunrise_zoom2_mode;
    bool l13a_mode = beauty_lighting_noon_mode || beauty_lighting_sunset_mode ||
        beauty_lighting_midnight_mode || beauty_lighting_sunrise_mode ||
        l13a_zoom2_mode;
    bool l94_mode = beauty_vegetation_mode || beauty_vegetation_only_mode ||
        beauty_no_vegetation_mode || beauty_vegetation_thumbnail_mode;
    bool l95_mode = beauty_shore_mode || beauty_shore_no_vegetation_mode ||
        beauty_shore_no_water_mode || beauty_shore_no_surf_mode ||
        beauty_shore_thumbnail_mode;
    bool promotion_mode = beauty_promotion_mode || beauty_promotion_no_vegetation_mode ||
        beauty_promotion_no_water_mode || beauty_promotion_no_surf_mode ||
        beauty_promotion_thumbnail_mode || beauty_dunes_mode ||
        beauty_dunes_no_dunes_mode || beauty_dunes_only_mode ||
        beauty_dunes_thumbnail_mode || beauty_marsh_mode ||
        beauty_marsh_no_marsh_mode || beauty_marsh_only_mode ||
        beauty_marsh_thumbnail_mode || beauty_volcano_mode ||
        beauty_volcano_no_volcano_mode || beauty_volcano_only_mode ||
        beauty_volcano_thumbnail_mode || beauty_rivers_mode ||
        beauty_rivers_no_rivers_mode || beauty_rivers_only_mode ||
        beauty_rivers_thumbnail_mode || l13a_mode || l14_mode || l15_mode;
    bool l10_mode = beauty_dunes_mode || beauty_dunes_no_dunes_mode ||
        beauty_dunes_only_mode || beauty_dunes_thumbnail_mode;
    bool l11_mode = beauty_marsh_mode || beauty_marsh_no_marsh_mode ||
        beauty_marsh_only_mode || beauty_marsh_thumbnail_mode ||
        beauty_volcano_mode || beauty_volcano_no_volcano_mode ||
        beauty_volcano_only_mode || beauty_volcano_thumbnail_mode ||
        beauty_rivers_mode || beauty_rivers_no_rivers_mode ||
        beauty_rivers_only_mode || beauty_rivers_thumbnail_mode || l13a_mode || l14_mode || l15_mode;
    bool l12_mode = beauty_volcano_mode || beauty_volcano_no_volcano_mode ||
        beauty_volcano_only_mode || beauty_volcano_thumbnail_mode;
    bool l13_mode = beauty_rivers_mode || beauty_rivers_no_rivers_mode ||
        beauty_rivers_only_mode || beauty_rivers_thumbnail_mode || l13a_mode || l14_mode;
    bool beauty_mode = std::strcmp(argv[4], "beauty") == 0 || beauty_no_relief_mode ||
        beauty_no_water_mode || beauty_thumbnail_mode || l94_mode || l95_mode || promotion_mode;
    bool beauty_relief_enabled = beauty_mode && !beauty_no_relief_mode;
    bool beauty_water_enabled = beauty_mode && !beauty_no_water_mode &&
        !beauty_shore_no_water_mode && !beauty_promotion_no_water_mode &&
        !beauty_dunes_only_mode && !beauty_marsh_only_mode && !beauty_volcano_only_mode &&
        !beauty_rivers_only_mode && !beauty_roads_only_mode && !beauty_roads_styles_mode &&
        !beauty_railroads_only_mode && !beauty_railroads_crossings_mode &&
        !beauty_resources_only_mode && !beauty_cities_only_mode && !beauty_mines_only_mode;
    bool beauty_surf_enabled = beauty_water_enabled && !beauty_shore_no_surf_mode &&
        !beauty_promotion_no_surf_mode;
    bool beauty_vegetation_enabled =
        (l94_mode && !beauty_no_vegetation_mode) ||
        (l95_mode && !beauty_shore_no_vegetation_mode && !beauty_shore_no_surf_mode) ||
        (promotion_mode && !beauty_promotion_no_vegetation_mode &&
         !beauty_promotion_no_surf_mode && !beauty_dunes_only_mode &&
         !beauty_marsh_only_mode && !beauty_volcano_only_mode &&
         !beauty_rivers_only_mode && !beauty_roads_only_mode &&
         !beauty_roads_styles_mode && !beauty_railroads_only_mode &&
         !beauty_railroads_crossings_mode && !beauty_resources_only_mode &&
         !beauty_cities_only_mode && !beauty_mines_only_mode);
    bool beauty_shore_enabled = l95_mode || promotion_mode;
    bool beauty_terrain_enabled = !beauty_vegetation_only_mode &&
        !beauty_resources_only_mode && !beauty_cities_only_mode && !beauty_mines_only_mode;
    bool coast_mode = coast_beach_mode || coast_cliff_mode || beauty_mode;
    promotion_scene_enabled = promotion_mode;
    l10_scene_enabled = l10_mode || l11_mode;
    l11_scene_enabled = l11_mode;
    l12_scene_enabled = l12_mode || l13_mode;
    l13_scene_enabled = l13_mode;
    l13a_scene_enabled = l13a_mode || l14_mode || l15_mode || l16_mode;
    l14_scene_enabled = l14_mode || l15_mode || l16_mode;
    l15_scene_enabled = l15_mode || l16_mode;
    l16_scene_enabled = l16_mode || l17_mode;
    l17_scene_enabled = l17_mode;
    biq_scene_enabled = l11_mode;
    dune_scene_enabled = (l10_mode || l11_mode) && !beauty_dunes_no_dunes_mode;
    volcano_geometry_enabled = l12_scene_enabled && !beauty_volcano_no_volcano_mode;
    river_geometry_enabled = l13_mode && !beauty_rivers_no_rivers_mode &&
        !beauty_roads_only_mode && !beauty_roads_styles_mode &&
        !beauty_resources_only_mode && !beauty_cities_only_mode && !beauty_mines_only_mode;
    road_geometry_enabled = l14_scene_enabled && !beauty_roads_no_roads_mode &&
        !beauty_railroads_only_mode && !beauty_resources_only_mode &&
        !beauty_cities_only_mode && !beauty_mines_only_mode;
    railroad_geometry_enabled = l15_mode && !beauty_railroads_no_railroads_mode &&
        !beauty_resources_only_mode && !beauty_cities_only_mode && !beauty_mines_only_mode;
    resource_geometry_enabled = l16_mode && !beauty_resources_no_resources_mode &&
        !beauty_resources_hidden_mode && !beauty_cities_only_mode && !beauty_mines_only_mode;
    city_geometry_enabled = l17_mode && !beauty_cities_no_cities_mode && !beauty_mines_only_mode;
    bool mine_geometry_enabled = l18_mode && !beauty_mines_no_mines_mode;
    bool feature_geometry_enabled = beauty_vegetation_enabled || river_geometry_enabled ||
        road_geometry_enabled || railroad_geometry_enabled || resource_geometry_enabled;
    if (promotion_scene_enabled) {
        output_width = l12_scene_enabled ? 3200u : (l10_scene_enabled ? 2560u : 2048u);
        output_height = l12_scene_enabled ? 1800u : (l10_scene_enabled ? 1440u : 1280u);
        scene_world_width = l12_scene_enabled ? 16.0f : (l10_scene_enabled ? 12.0f : 8.0f);
        scene_world_height = l12_scene_enabled ? 12.0f : (l10_scene_enabled ? 8.0f : 6.0f);
    }
    if (!albedo_mode && !material_mode && !relief_mode && !shadow_mode && !hill_mode &&
        !mountain_mode && !coast_mode) {
        std::fprintf(stderr, "terrain_lab: unknown mode\n");
        return 2;
    }
    if (beauty_vegetation_enabled && argc < 11) {
        std::fprintf(stderr, "terrain_lab: vegetation modes require a vegetation pack root\n");
        return 2;
    }
    if (dune_scene_enabled && argc < 12) {
        std::fprintf(stderr, "terrain_lab: dune modes require a normalized decal pack root\n");
        return 2;
    }
    if (l11_scene_enabled && argc < 13) {
        std::fprintf(stderr, "terrain_lab: BIQ promotion modes require a decoded viewport\n");
        return 2;
    }
    if (l12_scene_enabled && argc < 14) {
        std::fprintf(stderr, "terrain_lab: volcano modes require a normalized terrain-elements pack root\n");
        return 2;
    }
    if (l13_scene_enabled && argc < 15) {
        std::fprintf(stderr, "terrain_lab: river modes require a normalized shore-feature pack root\n");
        return 2;
    }
    if (l14_scene_enabled && argc < 18) {
        std::fprintf(stderr, "terrain_lab: road modes require normalized route/bridge packs and a Lab road scenario\n");
        return 2;
    }
    if (l15_scene_enabled && argc < 19) {
        std::fprintf(stderr, "terrain_lab: railroad modes require a Lab railroad scenario\n");
        return 2;
    }
    if (l16_scene_enabled && argc < 21) {
        std::fprintf(stderr, "terrain_lab: resource modes require a normalized resource pack and Lab resource scenario\n");
        return 2;
    }
    if (l17_scene_enabled && argc < 24) {
        std::fprintf(stderr, "terrain_lab: city modes require normalized city/wall packs and a Lab city scenario\n");
        return 2;
    }
    if (l18_mode && argc < 26) {
        std::fprintf(stderr, "terrain_lab: mine modes require a normalized mine pack and Lab mine scenario\n");
        return 2;
    }
    float uv_scale = argc > 5 ? std::strtof(argv[5], nullptr) : 0.26f;
    float normal_strength = argc > 6 ? std::strtof(argv[6], nullptr) : 4.0f;
    float exposure = argc > 7 ? std::strtof(argv[7], nullptr) : 1.0f;
    float relief_height = argc > 8 ? std::strtof(argv[8], nullptr) : 72.0f;
    float hill_uv_scale = argc > 9 ? std::strtof(argv[9], nullptr) : 0.085f;
    if (!(uv_scale > 0.01f && uv_scale <= 8.0f && normal_strength >= 0.0f && normal_strength <= 32.0f &&
          exposure > 0.05f && exposure <= 8.0f && relief_height >= 0.0f && relief_height <= 180.0f &&
          hill_uv_scale >= 0.005f && hill_uv_scale <= 1.0f)) {
        std::fprintf(stderr, "terrain_lab: parameter outside safe range\n");
        return 2;
    }
    if (l13a_scene_enabled) {
        frame_hour = (beauty_lighting_sunset_mode || beauty_lighting_sunset_zoom2_mode)
            ? 18.0f
            : ((beauty_lighting_midnight_mode || beauty_lighting_midnight_zoom2_mode ||
                beauty_cities_night_mode || beauty_mines_night_mode)
                ? 0.0f
                : ((beauty_lighting_sunrise_mode || beauty_lighting_sunrise_zoom2_mode)
                    ? 6.0f : 12.0f));
        frame_environment = c3x_renderer::evaluate_environment(frame_hour, 0);
        float const * selected_direction =
            frame_environment.sun_intensity >= frame_environment.moon_intensity
                ? frame_environment.sun_direction : frame_environment.moon_direction;
        active_light_direction[0] = selected_direction[0];
        active_light_direction[1] = selected_direction[1];
        active_light_direction[2] = selected_direction[2];
        c3x_renderer::EmissiveChannel diagnostic_emissive = {};
        diagnostic_emissive.color[0] = 1.0f;
        diagnostic_emissive.color[1] = 1.0f;
        diagnostic_emissive.color[2] = 1.0f;
        diagnostic_emissive.intensity = 1.0f;
        diagnostic_emissive.activation_policy = c3x_renderer::ActivationPolicy::night;
        float emissive_activation = c3x_renderer::evaluate_emissive(
            diagnostic_emissive, frame_environment, frame_hour);
        std::printf(
            "terrain_lab: environment hour=%.1f sun=%.4f moon=%.4f shadow=%.4f "
            "night=%.4f emissive=%.4f water_fresnel=%.4f water_specular=%.4f "
            "static_redraw=idle\n",
            frame_hour, frame_environment.sun_intensity,
            frame_environment.moon_intensity, frame_environment.shadow_strength,
            frame_environment.night_activation, emissive_activation,
            frame_environment.water_fresnel, frame_environment.water_specular);
    }
    if (l11_scene_enabled && !load_biq_window(argv[12], biq_window))
        return 1;
    if (l11_scene_enabled &&
        ((l12_scene_enabled && (biq_window.columns != 16 || biq_window.rows != 12)) ||
         (!l12_scene_enabled && (biq_window.columns != 12 || biq_window.rows != 8)))) {
        std::fprintf(stderr, "terrain_lab: BIQ viewport dimensions do not match the selected lab gate\n");
        return 1;
    }
    if (l11_scene_enabled) {
        scene_world_width = static_cast<float>(biq_window.columns);
        scene_world_height = static_cast<float>(biq_window.rows);
    }
    if (l13_scene_enabled)
        build_river_graph();
    if (l14_scene_enabled && !load_road_scenario(argv[16], road_scenario))
        return 1;
    if (l14_scene_enabled &&
        (road_scenario.columns != biq_window.columns ||
         road_scenario.rows != biq_window.rows)) {
        std::fprintf(stderr, "terrain_lab: Lab road scenario does not match BIQ viewport\n");
        return 1;
    }
    if (l15_scene_enabled && !load_railroad_scenario(argv[18], railroad_scenario))
        return 1;
    if (l15_scene_enabled &&
        (railroad_scenario.columns != biq_window.columns ||
         railroad_scenario.rows != biq_window.rows)) {
        std::fprintf(stderr, "terrain_lab: Lab railroad scenario does not match BIQ viewport\n");
        return 1;
    }
    if (l16_scene_enabled && !load_resource_scenario(argv[20], resource_scenario))
        return 1;
    if (l16_scene_enabled &&
        (resource_scenario.columns != biq_window.columns ||
         resource_scenario.rows != biq_window.rows)) {
        std::fprintf(stderr, "terrain_lab: Lab resource scenario does not match BIQ viewport\n");
        return 1;
    }
    if (l17_scene_enabled && !load_city_scenario(argv[23], city_scenario))
        return 1;
    if (l17_scene_enabled &&
        (city_scenario.columns != biq_window.columns ||
         city_scenario.rows != biq_window.rows)) {
        std::fprintf(stderr, "terrain_lab: Lab city scenario does not match BIQ viewport\n");
        return 1;
    }
    if (l18_mode && !load_mine_scenario(argv[25], mine_scenario))
        return 1;
    if (l18_mode &&
        (mine_scenario.columns != biq_window.columns ||
         mine_scenario.rows != biq_window.rows)) {
        std::fprintf(stderr, "terrain_lab: Lab mine scenario does not match BIQ viewport\n");
        return 1;
    }

    ID3D11Device * device = nullptr;
    ID3D11DeviceContext * context = nullptr;
    D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &feature_level, 1,
                                   D3D11_SDK_VERSION, &device, nullptr, &context);
    if (FAILED(hr))
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, &feature_level, 1,
                               D3D11_SDK_VERSION, &device, nullptr, &context);
    if (FAILED(hr)) {
        std::fprintf(stderr, "terrain_lab: Direct3D 11 device creation failed (0x%08lx)\n", hr);
        return 1;
    }

    ID3D11ShaderResourceView * base_view = nullptr;
    ID3D11ShaderResourceView * height_view = nullptr;
    ID3D11ShaderResourceView * specular_view = nullptr;
    ID3D11ShaderResourceView * authored_height_view = nullptr;
    ID3D11ShaderResourceView * authored_blend_view = nullptr;
    ID3D11ShaderResourceView * authored_region_view = nullptr;
    ID3D11ShaderResourceView * authored_hill_view = nullptr;
    ID3D11ShaderResourceView * authored_hill_variant_views[3] = {};
    ID3D11ShaderResourceView * authored_mountain_height_variant_views[4] = {};
    ID3D11ShaderResourceView * authored_mountain_blend_variant_views[4] = {};
    ID3D11ShaderResourceView * mountain_base_view = nullptr;
    ID3D11ShaderResourceView * mountain_top_view = nullptr;
    ID3D11ShaderResourceView * mountain_snow_view = nullptr;
    ID3D11ShaderResourceView * mountain_height_view = nullptr;
    ID3D11ShaderResourceView * mountain_specular_view = nullptr;
    ID3D11ShaderResourceView * beach_base_view = nullptr;
    ID3D11ShaderResourceView * beach_height_view = nullptr;
    ID3D11ShaderResourceView * beach_specular_view = nullptr;
    ID3D11ShaderResourceView * cliff_base_view = nullptr;
    ID3D11ShaderResourceView * cliff_height_view = nullptr;
    ID3D11ShaderResourceView * cliff_specular_view = nullptr;
    ID3D11ShaderResourceView * shallow_bed_view = nullptr;
    ID3D11ShaderResourceView * ocean_bed_view = nullptr;
    ID3D11ShaderResourceView * water_height_view = nullptr;
    ID3D11ShaderResourceView * water_large_lean0_view = nullptr;
    ID3D11ShaderResourceView * water_large_lean1_view = nullptr;
    ID3D11ShaderResourceView * water_small_lean0_view = nullptr;
    ID3D11ShaderResourceView * water_small_lean1_view = nullptr;
    ID3D11ShaderResourceView * water_foam_view = nullptr;
    ID3D11ShaderResourceView * shallows_specular_view = nullptr;
    ID3D11ShaderResourceView * ocean_height_view = nullptr;
    ID3D11ShaderResourceView * ocean_specular_view = nullptr;
    ID3D11ShaderResourceView * water_gloss_view = nullptr;
    ID3D11ShaderResourceView * water_tiling_mask_view = nullptr;
    ID3D11ShaderResourceView * water_non_tiling_mask_view = nullptr;
    ID3D11ShaderResourceView * water_small_secondary_lean0_view = nullptr;
    ID3D11ShaderResourceView * water_small_secondary_lean1_view = nullptr;
    ID3D11ShaderResourceView * water_ripples_view = nullptr;
    ID3D11ShaderResourceView * water_turbulence_view = nullptr;
    ID3D11ShaderResourceView * coast_dark_profile_view = nullptr;
    ID3D11ShaderResourceView * coast_scatter_profile_view = nullptr;
    ID3D11ShaderResourceView * water_tiling_normal0_view = nullptr;
    ID3D11ShaderResourceView * water_tiling_normal1_view = nullptr;
    ID3D11ShaderResourceView * water_non_tiling_normal0_view = nullptr;
    ID3D11ShaderResourceView * water_non_tiling_normal1_view = nullptr;
    ID3D11ShaderResourceView * plains_base_view = nullptr;
    ID3D11ShaderResourceView * plains_height_view = nullptr;
    ID3D11ShaderResourceView * plains_specular_view = nullptr;
    ID3D11ShaderResourceView * desert_base_view = nullptr;
    ID3D11ShaderResourceView * desert_height_view = nullptr;
    ID3D11ShaderResourceView * desert_specular_view = nullptr;
    ID3D11ShaderResourceView * desert_hills_base_view = nullptr;
    ID3D11ShaderResourceView * desert_hills_height_view = nullptr;
    ID3D11ShaderResourceView * desert_hills_specular_view = nullptr;
    ID3D11ShaderResourceView * dune_decal_base_view = nullptr;
    ID3D11ShaderResourceView * dune_decal_height_view = nullptr;
    ID3D11ShaderResourceView * marsh_base_view = nullptr;
    ID3D11ShaderResourceView * marsh_height_view = nullptr;
    ID3D11ShaderResourceView * marsh_specular_view = nullptr;
    ID3D11ShaderResourceView * marsh_decal_base_view = nullptr;
    ID3D11ShaderResourceView * marsh_decal_height_view = nullptr;
    ID3D11ShaderResourceView * marsh_decal_specular_view = nullptr;
    ID3D11ShaderResourceView * volcano_base_view = nullptr;
    ID3D11ShaderResourceView * volcano_height_view = nullptr;
    ID3D11ShaderResourceView * volcano_active_base_view = nullptr;
    ID3D11ShaderResourceView * volcano_active_specular_view = nullptr;
    ID3D11ShaderResourceView * water_decal_base_view = nullptr;
    ID3D11ShaderResourceView * water_decal_height_view = nullptr;
    ID3D11ShaderResourceView * grassland_decal_base_view = nullptr;
    ID3D11ShaderResourceView * grassland_decal_height_view = nullptr;
    ID3D11ShaderResourceView * plains_decal_base_view = nullptr;
    ID3D11ShaderResourceView * plains_decal_height_view = nullptr;
    ID3D11ShaderResourceView * river_base_view = nullptr;
    ID3D11ShaderResourceView * river_height_view = nullptr;
    ID3D11ShaderResourceView * river_specular_view = nullptr;
    ID3D11ShaderResourceView * river_lean0_view = nullptr;
    ID3D11ShaderResourceView * river_lean1_view = nullptr;
    ID3D11ShaderResourceView * river_source_base_view = nullptr;
    ID3D11ShaderResourceView * river_source_height_view = nullptr;
    ID3D11ShaderResourceView * river_clutter_base_view = nullptr;
    ID3D11ShaderResourceView * river_clutter_height_view = nullptr;
    ID3D11ShaderResourceView * river_bank_noise_view = nullptr;
    ID3D11ShaderResourceView * volcano_element_height_view = nullptr;
    ID3D11ShaderResourceView * volcano_element_blend_view = nullptr;
    ID3D11ShaderResourceView * desert_mountain_base_view = nullptr;
    ID3D11ShaderResourceView * desert_mountain_stripe1_view = nullptr;
    ID3D11ShaderResourceView * desert_mountain_stripe2_view = nullptr;
    ID3D11ShaderResourceView * desert_mountain_stripe3_view = nullptr;
    ID3D11ShaderResourceView * desert_mountain_height_view = nullptr;
    ID3D11ShaderResourceView * desert_mountain_specular_view = nullptr;
    ID3D11ShaderResourceView * feature_base_views[8] = {};
    ID3D11ShaderResourceView * river_feature_base_views[5] = {};
    ID3D11ShaderResourceView * road_base_views[10] = {};
    ID3D11ShaderResourceView * road_bridge_base_views[8] = {};
    ID3D11ShaderResourceView * resource_base_views[8] = {};
    ID3D11ShaderResourceView * city_base_views[4] = {};
    ID3D11ShaderResourceView * city_emissive_views[4] = {};
    ID3D11ShaderResourceView * wall_base_views[1] = {};
    ID3D11ShaderResourceView * mine_base_views[6] = {};
    ID3D11ShaderResourceView * mine_emissive_views[2] = {};
    FeatureBundle feature_bundle;
    FeatureBundle river_feature_bundle;
    FeatureBundle road_bridge_bundle;
    FeatureBundle resource_bundle;
    FeatureBundle city_bundle;
    FeatureBundle wall_bundle;
    FeatureBundle mine_bundle;
    HeightField authored_height;
    HeightField authored_blend;
    HeightField authored_region;
    HeightField authored_hills[4];
    HeightField authored_mountain_heights[4];
    HeightField authored_mountain_blends[4];
    HeightField authored_volcano_height;
    HeightField authored_volcano_blend;
    HeightField authored_river_bank_noise;
    authored_height.world_uv_scale = hill_uv_scale;
    unsigned base_width = 0, base_height = 0, height_width = 0, height_height = 0;
    unsigned unused_width = 0, unused_height = 0;
    std::string pack = argv[1];
    bool ok = load_dds(device, join(pack, "textures\\grassland_base_color.dds"),
                       DXGI_FORMAT_BC3_UNORM_SRGB, &base_view, base_width, base_height) &&
              load_dds(device, join(pack, "textures\\grassland_height.dds"),
                       DXGI_FORMAT_BC4_UNORM, &height_view, height_width, height_height) &&
              load_dds(device, join(pack, "textures\\grassland_specular.dds"),
                       DXGI_FORMAT_BC4_UNORM, &specular_view, unused_width, unused_height);
    if (ok && hill_mode)
        ok = load_r8_height(device, join(pack, "textures\\relief\\hills\\standard\\height_lod0.dds"),
                            DXGI_FORMAT_R8_UNORM, authored_height, &authored_height_view);
    if (ok && (mountain_mode || beauty_relief_enabled)) {
        // Keep variant 02 in the original shader slots for the standalone
        // channel inspection. BIQ promotion scenes load and select all five
        // authored standard variants below.
        std::string mountain_root = "textures\\relief\\mountains\\standard\\variant_02\\";
        ok = load_r8_height(device, join(pack, (mountain_root + "height_lod0.dds").c_str()),
                            DXGI_FORMAT_R8_UNORM, authored_height, &authored_height_view) &&
             load_r8_height(device, join(pack, (mountain_root + "blend_lod0.dds").c_str()),
                            DXGI_FORMAT_R8_UNORM, authored_blend, &authored_blend_view) &&
             load_r8_height(device, join(pack, (mountain_root + "region_ids_lod0.dds").c_str()),
                            DXGI_FORMAT_R8_UINT, authored_region, &authored_region_view);
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        if (ok)
            ok = load_dds(device, join(pack, "textures\\mtn_base_base_color.dds"),
                          DXGI_FORMAT_BC3_UNORM_SRGB, &mountain_base_view, texture_width, texture_height) &&
                 load_dds(device, join(pack, "textures\\mtn_top_base_color.dds"),
                          DXGI_FORMAT_BC3_UNORM_SRGB, &mountain_top_view, texture_width, texture_height) &&
                 load_dds(device, join(pack, "textures\\mtn_snow_base_color.dds"),
                          DXGI_FORMAT_BC3_UNORM_SRGB, &mountain_snow_view, texture_width, texture_height) &&
                 load_dds(device, join(pack, "textures\\mtn_base_height.dds"),
                          DXGI_FORMAT_BC4_UNORM, &mountain_height_view, texture_width, texture_height) &&
                 load_dds(device, join(pack, "textures\\mtn_base_specular.dds"),
                          DXGI_FORMAT_BC4_UNORM, &mountain_specular_view, texture_width, texture_height);
        unsigned source_variants[4] = {1u, 3u, 4u, 5u};
        for (unsigned index = 0; ok && index < 4; ++index) {
            char root[96] = {};
            sprintf_s(root, "textures\\relief\\mountains\\standard\\variant_%02u\\",
                      source_variants[index]);
            ok = load_r8_height(device, join(pack, (std::string(root) + "height_lod0.dds").c_str()),
                                DXGI_FORMAT_R8_UNORM, authored_mountain_heights[index],
                                &authored_mountain_height_variant_views[index]) &&
                 load_r8_height(device, join(pack, (std::string(root) + "blend_lod0.dds").c_str()),
                                DXGI_FORMAT_R8_UNORM, authored_mountain_blends[index],
                                &authored_mountain_blend_variant_views[index]);
        }
        if (ok) {
            promotion_mountain_height_fields[0] = &authored_mountain_heights[0];
            promotion_mountain_height_fields[1] = &authored_height;
            promotion_mountain_height_fields[2] = &authored_mountain_heights[1];
            promotion_mountain_height_fields[3] = &authored_mountain_heights[2];
            promotion_mountain_height_fields[4] = &authored_mountain_heights[3];
            promotion_mountain_blend_fields[0] = &authored_mountain_blends[0];
            promotion_mountain_blend_fields[1] = &authored_blend;
            promotion_mountain_blend_fields[2] = &authored_mountain_blends[1];
            promotion_mountain_blend_fields[3] = &authored_mountain_blends[2];
            promotion_mountain_blend_fields[4] = &authored_mountain_blends[3];
        }
    }
    if (ok && promotion_mode) {
        for (HeightField & hill : authored_hills)
            hill.world_uv_scale = hill_uv_scale;
        ok = load_r8_height(device,
                            join(pack, "textures\\relief\\hills\\standard\\height_lod0.dds"),
                            DXGI_FORMAT_R8_UNORM, authored_hills[0], &authored_hill_view) &&
             load_r8_height(device,
                            join(pack, "textures\\relief\\hills\\continental\\height_lod0.dds"),
                            DXGI_FORMAT_R8_UNORM, authored_hills[1],
                            &authored_hill_variant_views[0]) &&
             load_r8_height(device,
                            join(pack, "textures\\relief\\hills\\continental_plains\\height_lod0.dds"),
                            DXGI_FORMAT_R8_UNORM, authored_hills[2],
                            &authored_hill_variant_views[1]) &&
             load_r8_height(device,
                            join(pack, "textures\\relief\\hills\\continental_snow\\height_lod0.dds"),
                            DXGI_FORMAT_R8_UNORM, authored_hills[3],
                            &authored_hill_variant_views[2]);
        if (ok)
            for (unsigned index = 0; index < 4; ++index)
                promotion_hill_height_fields[index] = &authored_hills[index];
    }
    if (ok && coast_mode) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        ok = load_dds(device, join(pack, "textures\\beach_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &beach_base_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\beach_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &beach_height_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\beach_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &beach_specular_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\cliff_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &cliff_base_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\cliff_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &cliff_height_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\cliff_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &cliff_specular_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\shallows_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &shallow_bed_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\ocean_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &ocean_bed_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\shallows_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &water_height_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\shallows_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &shallows_specular_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\ocean_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &ocean_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\ocean_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &ocean_specular_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\large_lean0.dds"),
                      DXGI_FORMAT_R16G16B16A16_UNORM, &water_large_lean0_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\large_lean1.dds"),
                      DXGI_FORMAT_R16G16_UNORM, &water_large_lean1_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\small_lean0.dds"),
                      DXGI_FORMAT_R16G16B16A16_UNORM, &water_small_lean0_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\small_lean1.dds"),
                      DXGI_FORMAT_R16G16_UNORM, &water_small_lean1_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\effects\\crash_foam.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &water_foam_view, texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\gloss.dds"),
                      DXGI_FORMAT_BC1_UNORM_SRGB, &water_gloss_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\tiling_mask.dds"),
                      DXGI_FORMAT_BC4_UNORM, &water_tiling_mask_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\non_tiling_mask.dds"),
                      DXGI_FORMAT_BC4_UNORM, &water_non_tiling_mask_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\small_secondary_lean0.dds"),
                      DXGI_FORMAT_R16G16B16A16_UNORM, &water_small_secondary_lean0_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\small_secondary_lean1.dds"),
                      DXGI_FORMAT_R16G16_UNORM, &water_small_secondary_lean1_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\effects\\ripples_primary.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &water_ripples_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\effects\\turbulence.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &water_turbulence_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\profiles\\coast\\dark.dds"),
                      DXGI_FORMAT_R16G16B16A16_FLOAT, &coast_dark_profile_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\profiles\\coast\\scatter.dds"),
                      DXGI_FORMAT_R16G16B16A16_FLOAT, &coast_scatter_profile_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\tiling_normal0.dds"),
                      DXGI_FORMAT_BC5_UNORM, &water_tiling_normal0_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\tiling_normal1.dds"),
                      DXGI_FORMAT_BC4_UNORM, &water_tiling_normal1_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\non_tiling_normal0.dds"),
                      DXGI_FORMAT_BC5_UNORM, &water_non_tiling_normal0_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\non_tiling_normal1.dds"),
                      DXGI_FORMAT_BC4_UNORM, &water_non_tiling_normal1_view,
                      texture_width, texture_height);
    }
    if (ok && promotion_mode) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        ok = load_dds(device, join(pack, "textures\\plains_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &plains_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\plains_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &plains_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\plains_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &plains_specular_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\desert_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &desert_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\desert_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &desert_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\desert_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &desert_specular_view,
                      texture_width, texture_height);
    }
    if (ok && l10_mode) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        ok = load_dds(device, join(pack, "textures\\desert_hills_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &desert_hills_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\desert_hills_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &desert_hills_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\desert_hills_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &desert_hills_specular_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\mtn_desert_base_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &desert_mountain_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\mtn_desert_stripe01_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &desert_mountain_stripe1_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\mtn_desert_stripe02_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &desert_mountain_stripe2_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\mtn_desert_stripe03_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &desert_mountain_stripe3_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\mtn_desert_base_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &desert_mountain_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\mtn_desert_base_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &desert_mountain_specular_view,
                      texture_width, texture_height);
    }
    if (ok && dune_scene_enabled) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        std::string decal_pack = argv[11];
        ok = load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\base_color_8f27cbd468d0e4fd.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &dune_decal_base_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\height_f0302076c5c278f5.dds"),
                      DXGI_FORMAT_BC5_UNORM, &dune_decal_height_view,
                      texture_width, texture_height);
    }
    if (ok && l11_scene_enabled) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        std::string decal_pack = argv[11];
        ok = load_dds(device, join(pack, "textures\\grassmarsh_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &marsh_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\grassmarsh_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &marsh_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\grassmarsh_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &marsh_specular_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\base_color_e48cf9469a284218.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &marsh_decal_base_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\height_b37e632844a0e3f2.dds"),
                      DXGI_FORMAT_BC5_UNORM, &marsh_decal_height_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\specular_2af348f2e159d8f7.dds"),
                      DXGI_FORMAT_BC4_UNORM, &marsh_decal_specular_view,
                      texture_width, texture_height);
    }
    if (ok && l12_scene_enabled) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        std::string terrain_elements_pack = argv[13];
        std::string decal_pack = argv[11];
        ok = load_r8_height(
                 device,
                 join(terrain_elements_pack,
                      "textures\\terrain_elements\\terrain_feature_volcano\\height_lod0.dds"),
                 DXGI_FORMAT_R8_UNORM, authored_volcano_height,
                 &volcano_element_height_view) &&
             load_r8_height(
                 device,
                 join(terrain_elements_pack,
                      "textures\\terrain_elements\\terrain_feature_volcano\\blend_lod0.dds"),
                 DXGI_FORMAT_R8_UNORM, authored_volcano_blend,
                 &volcano_element_blend_view) &&
             load_dds(device, join(pack, "textures\\water\\volcano\\base.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &volcano_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\volcano\\height.dds"),
                      DXGI_FORMAT_BC5_UNORM, &volcano_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\volcano\\active_base.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &volcano_active_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\volcano\\active_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &volcano_active_specular_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\base_color_bbba55639c23d574.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &water_decal_base_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\height_6964171784984fe8.dds"),
                      DXGI_FORMAT_BC5_UNORM, &water_decal_height_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\base_color_c996c6a9d015eebe.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &grassland_decal_base_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\height_31eb0f0117ea3beb.dds"),
                      DXGI_FORMAT_BC5_UNORM, &grassland_decal_height_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\base_color_211cf603f50c6f54.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &plains_decal_base_view,
                      texture_width, texture_height) &&
             load_dds(device,
                      join(decal_pack,
                           "textures\\decals\\height_3ba76b3b97d571a8.dds"),
                      DXGI_FORMAT_BC5_UNORM, &plains_decal_height_view,
                      texture_width, texture_height);
        if (ok) {
            promotion_volcano_height_field = &authored_volcano_height;
            promotion_volcano_blend_field = &authored_volcano_blend;
        }
    }
    if (ok && l13_scene_enabled) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        ok = load_dds(device, join(pack, "textures\\river_base_color.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &river_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\river_height.dds"),
                      DXGI_FORMAT_BC4_UNORM, &river_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\river_specular.dds"),
                      DXGI_FORMAT_BC4_UNORM, &river_specular_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\river_lean0.dds"),
                      DXGI_FORMAT_R16G16B16A16_UNORM, &river_lean0_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\surface\\river_lean1.dds"),
                      DXGI_FORMAT_R16G16_UNORM, &river_lean1_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\river\\source_decal_base.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &river_source_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\river\\source_decal_height.dds"),
                      DXGI_FORMAT_BC5_UNORM, &river_source_height_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\river\\clutter_decal_base.dds"),
                      DXGI_FORMAT_BC3_UNORM_SRGB, &river_clutter_base_view,
                      texture_width, texture_height) &&
             load_dds(device, join(pack, "textures\\water\\river\\clutter_decal_height.dds"),
                      DXGI_FORMAT_BC5_UNORM, &river_clutter_height_view,
                      texture_width, texture_height) &&
             load_r8_height(
                 device,
                 join(pack,
                      "textures\\water\\relief\\river_bank_noise\\height_lod0.dds"),
                 DXGI_FORMAT_R8_UNORM, authored_river_bank_noise,
                 &river_bank_noise_view);
        if (ok) {
            std::string shore_pack = argv[14];
            ok = load_feature_bundle(join(shore_pack, "shore_runtime.bin"),
                                     river_feature_bundle) &&
                 river_feature_bundle.texture_paths.size() == 5u;
            for (FeatureAsset & asset : river_feature_bundle.assets)
                asset.texture_index += 8u;
            for (unsigned index = 0; ok && index < 5u; ++index)
                ok = load_dds(
                    device,
                    join(shore_pack,
                         river_feature_bundle.texture_paths[index].c_str()),
                    DXGI_FORMAT_BC1_UNORM_SRGB,
                    &river_feature_base_views[index], texture_width, texture_height);
        }
    }
    if (ok && l14_scene_enabled) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        std::string route_pack = argv[15];
        char const * route_textures[] = {
            "textures\\routes\\base_color_f5c58170975c3c8b.dds",
            "textures\\routes\\base_color_6c2f0cc0e38fff7b.dds",
            "textures\\routes\\base_color_5ae696c133345a83.dds",
            "textures\\routes\\base_color_ba7297551e6ece35.dds",
            "textures\\routes\\base_color_3631908100843d0d.dds",
            "textures\\routes\\base_color_678d822a9338aab5.dds",
            "textures\\routes\\base_color_fc46370d90163f16.dds",
            "textures\\routes\\base_color_a832a6518fdab3b1.dds",
            "textures\\routes\\base_color_c3ae5ee0b5879164.dds",
            "textures\\routes\\base_color_39a5855bea35f6d1.dds",
        };
        for (unsigned index = 0; ok && index < 10; ++index)
            ok = load_dds(device, join(route_pack, route_textures[index]),
                          DXGI_FORMAT_BC3_UNORM_SRGB, &road_base_views[index],
                          texture_width, texture_height);
        std::string bridge_pack = argv[17];
        if (ok)
            ok = load_feature_bundle(join(bridge_pack, "bridge_runtime.bin"),
                                     road_bridge_bundle) &&
                 road_bridge_bundle.texture_paths.size() == 8u;
        for (FeatureAsset & asset : road_bridge_bundle.assets)
            asset.texture_index += 13u;
        for (unsigned index = 0; ok && index < 8; ++index)
            ok = load_dds(device,
                          join(bridge_pack,
                               road_bridge_bundle.texture_paths[index].c_str()),
                          DXGI_FORMAT_BC1_UNORM_SRGB,
                          &road_bridge_base_views[index],
                          texture_width, texture_height);
    }
    if (ok && l16_scene_enabled) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        std::string resource_pack = argv[19];
        ok = load_feature_bundle(join(resource_pack, "resource_runtime.bin"),
                                 resource_bundle) &&
             resource_bundle.texture_paths.size() == 8u;
        for (FeatureAsset & asset : resource_bundle.assets)
            asset.texture_index += 21u;
        for (unsigned index = 0; ok && index < 8u; ++index)
            ok = load_dds(device,
                          join(resource_pack,
                               resource_bundle.texture_paths[index].c_str()),
                          DXGI_FORMAT_BC1_UNORM_SRGB,
                          &resource_base_views[index], texture_width, texture_height);
    }
    if (ok && l17_scene_enabled) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        std::string city_pack = argv[21];
        std::string wall_pack = argv[22];
        ok = load_feature_bundle(join(city_pack, "city_runtime.bin"), city_bundle) &&
             city_bundle.texture_paths.size() == 8u &&
             load_feature_bundle(join(wall_pack, "wall_runtime.bin"), wall_bundle) &&
             wall_bundle.texture_paths.size() == 1u;
        for (FeatureAsset & asset : city_bundle.assets)
            asset.texture_index += 29u;
        for (FeatureAsset & asset : wall_bundle.assets)
            asset.texture_index += 29u;
        for (unsigned index = 0; ok && index < 4u; ++index) {
            ok = load_dds(device, join(city_pack, city_bundle.texture_paths[index].c_str()),
                          DXGI_FORMAT_BC1_UNORM_SRGB, &city_base_views[index],
                          texture_width, texture_height) &&
                 load_dds(device, join(city_pack, city_bundle.texture_paths[index + 4u].c_str()),
                          DXGI_FORMAT_BC1_UNORM_SRGB, &city_emissive_views[index],
                          texture_width, texture_height);
        }
        if (ok)
            ok = load_dds(device, join(wall_pack, wall_bundle.texture_paths[0].c_str()),
                          DXGI_FORMAT_BC1_UNORM_SRGB, &wall_base_views[0],
                          texture_width, texture_height);
    }
    if (ok && l18_mode) {
        unsigned texture_width = 0;
        unsigned texture_height = 0;
        std::string mine_pack = argv[24];
        ok = load_feature_bundle(join(mine_pack, "mine_runtime.bin"), mine_bundle) &&
             mine_bundle.texture_paths.size() == 8u;
        for (FeatureAsset & asset : mine_bundle.assets)
            asset.texture_index += 21u;
        for (unsigned index = 0; ok && index < 6u; ++index) {
            std::string path = join(mine_pack, mine_bundle.texture_paths[index].c_str());
            DXGI_FORMAT format = source_color_dds_format(path);
            ok = format != DXGI_FORMAT_UNKNOWN &&
                 load_dds(device, path, format, &mine_base_views[index],
                          texture_width, texture_height);
        }
        for (unsigned index = 0; ok && index < 2u; ++index) {
            std::string path = join(mine_pack, mine_bundle.texture_paths[index + 6u].c_str());
            DXGI_FORMAT format = source_color_dds_format(path);
            ok = format != DXGI_FORMAT_UNKNOWN &&
                 load_dds(device, path, format, &mine_emissive_views[index],
                          texture_width, texture_height);
        }
    }
    if (ok && feature_geometry_enabled) {
        std::string vegetation_pack = argv[10];
        ok = load_feature_bundle(join(vegetation_pack, "vegetation_runtime.bin"), feature_bundle);
        for (unsigned index = 0; ok && index < feature_bundle.texture_paths.size(); ++index) {
            unsigned texture_width = 0;
            unsigned texture_height = 0;
            ok = load_dds(device, join(vegetation_pack, feature_bundle.texture_paths[index].c_str()),
                          DXGI_FORMAT_BC1_UNORM_SRGB, &feature_base_views[index],
                          texture_width, texture_height);
        }
    }
    ID3DBlob * vertex_blob = nullptr;
    ID3DBlob * pixel_blob = nullptr;
    ID3D11VertexShader * vertex_shader = nullptr;
    ID3D11PixelShader * pixel_shader = nullptr;
    ID3D11InputLayout * input_layout = nullptr;
    ID3D11VertexShader * feature_vertex_shader = nullptr;
    ID3D11PixelShader * feature_pixel_shader = nullptr;
    ID3D11InputLayout * feature_input_layout = nullptr;
    if (ok)
        ok = compile_shader(argv[2], "VSMain", "vs_4_0", &vertex_blob) &&
             compile_shader(argv[2], "PSMain", "ps_4_0", &pixel_blob);
    if (ok)
        hr = device->CreateVertexShader(vertex_blob->GetBufferPointer(), vertex_blob->GetBufferSize(),
                                        nullptr, &vertex_shader);
    if (ok && SUCCEEDED(hr))
        hr = device->CreatePixelShader(pixel_blob->GetBufferPointer(), pixel_blob->GetBufferSize(),
                                       nullptr, &pixel_shader);
    D3D11_INPUT_ELEMENT_DESC elements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 1, DXGI_FORMAT_R32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 2, DXGI_FORMAT_R32G32_FLOAT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 3, DXGI_FORMAT_R32G32_FLOAT, 0, 40, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 4, DXGI_FORMAT_R32_FLOAT, 0, 48, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 5, DXGI_FORMAT_R32_FLOAT, 0, 52, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 6, DXGI_FORMAT_R32_FLOAT, 0, 56, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 7, DXGI_FORMAT_R32_FLOAT, 0, 60, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 8, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 64, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 9, DXGI_FORMAT_R32_FLOAT, 0, 80, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 10, DXGI_FORMAT_R32G32_FLOAT, 0, 84, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 11, DXGI_FORMAT_R32_FLOAT, 0, 92, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 12, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 96, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    if (ok && SUCCEEDED(hr))
        hr = device->CreateInputLayout(elements, 15, vertex_blob->GetBufferPointer(),
                                       vertex_blob->GetBufferSize(), &input_layout);
    ok = ok && SUCCEEDED(hr);
    release(vertex_blob);
    release(pixel_blob);

    if (ok && feature_geometry_enabled) {
        ID3DBlob * feature_vertex_blob = nullptr;
        ID3DBlob * feature_pixel_blob = nullptr;
        ok = compile_shader(argv[2], "VSFeature", "vs_4_0", &feature_vertex_blob) &&
             compile_shader(argv[2], "PSFeature", "ps_4_0", &feature_pixel_blob);
        if (ok)
            hr = device->CreateVertexShader(feature_vertex_blob->GetBufferPointer(),
                                            feature_vertex_blob->GetBufferSize(), nullptr,
                                            &feature_vertex_shader);
        if (ok && SUCCEEDED(hr))
            hr = device->CreatePixelShader(feature_pixel_blob->GetBufferPointer(),
                                           feature_pixel_blob->GetBufferSize(), nullptr,
                                           &feature_pixel_shader);
        D3D11_INPUT_ELEMENT_DESC feature_elements[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 1, DXGI_FORMAT_R32_FLOAT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        if (ok && SUCCEEDED(hr))
            hr = device->CreateInputLayout(feature_elements, 4,
                                           feature_vertex_blob->GetBufferPointer(),
                                           feature_vertex_blob->GetBufferSize(),
                                           &feature_input_layout);
        ok = ok && SUCCEEDED(hr);
        release(feature_vertex_blob);
        release(feature_pixel_blob);
    }

    std::vector<Vertex> vertices;
    std::vector<FeatureVertex> feature_vertices;
    std::vector<FeatureVertex> city_vertices;
    std::vector<FeatureVertex> wall_vertices;
    std::vector<FeatureVertex> mine_vertices;
    if (!beauty_mode)
        add_source_panel(vertices, mountain_mode, coast_mode);
    if (coast_mode) {
        if (promotion_scene_enabled)
            coast_projection = biq_scene_enabled
                ? (l12_scene_enabled
                  ? CoastProjection{100.0f, 770.0f, 112.0f, 56.0f, 0.82f}
                  : CoastProjection{140.0f, 600.0f, 120.0f, 60.0f, 0.82f})
                : (l10_scene_enabled
                ? CoastProjection{1000.0f, 110.0f, 125.0f, 62.5f, 0.82f}
                : CoastProjection{840.0f, 190.0f, 125.0f, 62.5f, 0.82f});
        else if (beauty_mode)
            coast_projection = {512.0f, 34.0f, 218.0f, 105.0f, 1.65f};
        if (!beauty_mode || beauty_terrain_enabled) {
            if (biq_scene_enabled)
                add_biq_patch(vertices, uv_scale,
                              beauty_relief_enabled ? &authored_height : nullptr,
                              beauty_relief_enabled ? &authored_blend : nullptr,
                              beauty_water_enabled);
            else
                ok = ok && add_coast_patch(vertices, uv_scale, coast_cliff_mode, beauty_mode,
                                           !beauty_mode || beauty_water_enabled,
                                           !beauty_mode || beauty_surf_enabled,
                                           beauty_shore_enabled,
                                           beauty_relief_enabled ? &authored_height : nullptr,
                                           beauty_relief_enabled ? &authored_blend : nullptr);
        }
        if (ok && promotion_scene_enabled) {
            if (biq_scene_enabled)
                add_biq_grid(vertices,
                             beauty_relief_enabled ? &authored_height : nullptr,
                             beauty_relief_enabled ? &authored_blend : nullptr);
            else
                add_promotion_grid(vertices,
                                   beauty_relief_enabled ? &authored_height : nullptr,
                                   beauty_relief_enabled ? &authored_blend : nullptr);
        }
        if (ok && road_geometry_enabled)
            add_road_scene(vertices,
                           beauty_relief_enabled ? &authored_height : nullptr,
                           beauty_relief_enabled ? &authored_blend : nullptr,
                           beauty_roads_styles_mode ? -1 : 1);
        if (ok && road_geometry_enabled)
            ok = add_road_bridge_scene(
                road_bridge_bundle,
                beauty_relief_enabled ? &authored_height : nullptr,
                beauty_relief_enabled ? &authored_blend : nullptr,
                beauty_roads_styles_mode ? -1 : 1,
                vertices, feature_vertices);
        if (ok && railroad_geometry_enabled)
            add_railroad_scene(vertices,
                               beauty_relief_enabled ? &authored_height : nullptr,
                               beauty_relief_enabled ? &authored_blend : nullptr);
        if (ok && railroad_geometry_enabled)
            ok = add_railroad_bridge_scene(
                road_bridge_bundle,
                beauty_relief_enabled ? &authored_height : nullptr,
                beauty_relief_enabled ? &authored_blend : nullptr,
                vertices, feature_vertices);
        if (ok && beauty_vegetation_enabled)
            ok = add_feature_scene(feature_bundle,
                                   beauty_relief_enabled ? &authored_height : nullptr,
                                   beauty_relief_enabled ? &authored_blend : nullptr,
                                   beauty_shore_enabled,
                                   vertices, feature_vertices);
        if (ok && river_geometry_enabled)
            ok = add_river_rock_scene(
                river_feature_bundle,
                beauty_relief_enabled ? &authored_height : nullptr,
                beauty_relief_enabled ? &authored_blend : nullptr,
                vertices, feature_vertices);
        if (ok && resource_geometry_enabled)
            ok = add_resource_scene(
                resource_bundle,
                beauty_relief_enabled ? &authored_height : nullptr,
                beauty_relief_enabled ? &authored_blend : nullptr,
                vertices, feature_vertices);
        if (ok && city_geometry_enabled)
            ok = add_city_scene(
                city_bundle, wall_bundle,
                beauty_relief_enabled ? &authored_height : nullptr,
                beauty_relief_enabled ? &authored_blend : nullptr,
                vertices, city_vertices, wall_vertices);
        if (ok && mine_geometry_enabled)
            ok = add_mine_scene(
                mine_bundle,
                beauty_relief_enabled ? &authored_height : nullptr,
                beauty_relief_enabled ? &authored_blend : nullptr,
                vertices, mine_vertices);
    } else {
        add_patch_grid(vertices, uv_scale,
                       (relief_mode || shadow_mode || hill_mode || mountain_mode) ? relief_height : 0.0f,
                       shadow_mode || hill_mode || mountain_mode,
                       (hill_mode || mountain_mode) ? &authored_height : nullptr,
                       mountain_mode ? &authored_blend : nullptr);
    }

    ID3D11Buffer * vertex_buffer = nullptr;
    ID3D11Buffer * feature_vertex_buffer = nullptr;
    ID3D11Buffer * city_vertex_buffer = nullptr;
    ID3D11Buffer * wall_vertex_buffer = nullptr;
    ID3D11Buffer * mine_vertex_buffer = nullptr;
    ID3D11Buffer * settings_buffer = nullptr;
    ID3D11SamplerState * sampler = nullptr;
    ID3D11SamplerState * decal_sampler = nullptr;
    ID3D11RasterizerState * rasterizer = nullptr;
    ID3D11BlendState * blend_state = nullptr;
    ID3D11Texture2D * render_texture = nullptr;
    ID3D11RenderTargetView * render_target = nullptr;
    ID3D11Texture2D * readback_texture = nullptr;
    ID3D11Texture2D * feature_depth_texture = nullptr;
    ID3D11DepthStencilView * feature_depth_view = nullptr;
    ID3D11DepthStencilState * feature_depth_state = nullptr;
    if (ok) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = static_cast<UINT>(vertices.size() * sizeof(Vertex));
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA initial = {vertices.data(), 0, 0};
        hr = device->CreateBuffer(&desc, &initial, &vertex_buffer);
    }
    if (ok && feature_geometry_enabled) {
        if (feature_vertices.empty()) {
            ok = false;
        } else {
            D3D11_BUFFER_DESC desc = {};
            desc.ByteWidth = static_cast<UINT>(feature_vertices.size() * sizeof(FeatureVertex));
            desc.Usage = D3D11_USAGE_IMMUTABLE;
            desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA initial = {feature_vertices.data(), 0, 0};
            hr = device->CreateBuffer(&desc, &initial, &feature_vertex_buffer);
        }
    }
    if (ok && city_geometry_enabled) {
        if (city_vertices.empty()) {
            ok = false;
        } else {
            D3D11_BUFFER_DESC desc = {};
            desc.ByteWidth = static_cast<UINT>(city_vertices.size() * sizeof(FeatureVertex));
            desc.Usage = D3D11_USAGE_IMMUTABLE;
            desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA initial = {city_vertices.data(), 0, 0};
            hr = device->CreateBuffer(&desc, &initial, &city_vertex_buffer);
        }
        if (ok && SUCCEEDED(hr) && !wall_vertices.empty()) {
            D3D11_BUFFER_DESC desc = {};
            desc.ByteWidth = static_cast<UINT>(wall_vertices.size() * sizeof(FeatureVertex));
            desc.Usage = D3D11_USAGE_IMMUTABLE;
            desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA initial = {wall_vertices.data(), 0, 0};
            hr = device->CreateBuffer(&desc, &initial, &wall_vertex_buffer);
        }
    }
    if (ok && mine_geometry_enabled) {
        if (mine_vertices.empty()) {
            ok = false;
        } else {
            D3D11_BUFFER_DESC desc = {};
            desc.ByteWidth = static_cast<UINT>(mine_vertices.size() * sizeof(FeatureVertex));
            desc.Usage = D3D11_USAGE_IMMUTABLE;
            desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA initial = {mine_vertices.data(), 0, 0};
            hr = device->CreateBuffer(&desc, &initial, &mine_vertex_buffer);
        }
    }
    float lab_mode_value = 0.0f;
    if (material_mode) lab_mode_value = 1.0f;
    if (relief_mode) lab_mode_value = 2.0f;
    if (shadow_mode) lab_mode_value = 3.0f;
    if (hill_mode) lab_mode_value = 4.0f;
    if (mountain_mode) lab_mode_value = 5.0f;
    if (coast_beach_mode) lab_mode_value = 6.0f;
    if (coast_cliff_mode) lab_mode_value = 7.0f;
    if (beauty_mode) lab_mode_value = 8.0f;
    LabSettings settings = {{1.0f / static_cast<float>(height_width),
                             1.0f / static_cast<float>(height_height)},
                            normal_strength, exposure, lab_mode_value,
                            beauty_relief_enabled ? 1.0f : 0.0f,
                            beauty_water_enabled ? 1.0f : 0.0f,
                            beauty_shore_enabled ? 1.0f : 0.0f,
                            promotion_scene_enabled ? 1.0f : 0.0f,
                            scene_world_width, scene_world_height,
                            dune_scene_enabled ? 1.0f : 0.0f,
                            beauty_dunes_only_mode ? 1.0f : 0.0f,
                            l10_mode ? 1.0f : 0.0f,
                            biq_scene_enabled ? 1.0f : 0.0f,
                            l11_scene_enabled && !beauty_marsh_no_marsh_mode ? 1.0f : 0.0f,
                            beauty_marsh_only_mode ? 1.0f : 0.0f,
                            volcano_geometry_enabled ? 1.0f : 0.0f,
                            beauty_volcano_only_mode ? 1.0f : 0.0f,
                            l12_scene_enabled ? 1.0f : 0.0f,
                            river_geometry_enabled ? 1.0f : 0.0f,
                            beauty_rivers_only_mode ? 1.0f : 0.0f,
                            l13_scene_enabled ? 1.0f : 0.0f,
                            l13a_scene_enabled ? 1.0f : 0.0f,
                            {frame_environment.sun_direction[0],
                             frame_environment.sun_direction[1],
                             frame_environment.sun_direction[2]},
                            frame_environment.sun_intensity,
                            {frame_environment.sun_color[0],
                             frame_environment.sun_color[1],
                             frame_environment.sun_color[2]},
                            frame_environment.shadow_strength,
                            {frame_environment.moon_direction[0],
                             frame_environment.moon_direction[1],
                             frame_environment.moon_direction[2]},
                            frame_environment.moon_intensity,
                            {frame_environment.moon_color[0],
                             frame_environment.moon_color[1],
                             frame_environment.moon_color[2]},
                            frame_environment.night_activation,
                            {frame_environment.ambient_color[0],
                             frame_environment.ambient_color[1],
                             frame_environment.ambient_color[2]},
                            frame_environment.exposure,
                            frame_environment.water_fresnel,
                            frame_environment.water_specular,
                            frame_environment.emissive_scale,
                            frame_hour,
                            road_geometry_enabled ? 1.0f : 0.0f,
                            (beauty_roads_only_mode || beauty_roads_styles_mode) ? 1.0f : 0.0f,
                            l14_scene_enabled ? 1.0f : 0.0f,
                            beauty_roads_styles_mode ? -1.0f : 1.0f,
                            railroad_geometry_enabled ? 1.0f : 0.0f,
                            (beauty_railroads_only_mode ||
                             beauty_railroads_crossings_mode) ? 1.0f : 0.0f,
                            l15_scene_enabled ? 1.0f : 0.0f,
                            0.0f,
                            resource_geometry_enabled ? 1.0f : 0.0f,
                            beauty_resources_only_mode ? 1.0f : 0.0f,
                            l16_scene_enabled ? 1.0f : 0.0f,
                            0.0f,
                            city_geometry_enabled ? 1.0f : 0.0f,
                            beauty_cities_only_mode ? 1.0f : 0.0f,
                            l17_scene_enabled ? 1.0f : 0.0f,
                            0.0f};
    if (ok && SUCCEEDED(hr)) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = sizeof(settings);
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial = {&settings, 0, 0};
        hr = device->CreateBuffer(&desc, &initial, &settings_buffer);
    }
    if (ok && SUCCEEDED(hr)) {
        D3D11_SAMPLER_DESC desc = {};
        desc.Filter = D3D11_FILTER_ANISOTROPIC;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.MaxAnisotropy = 8;
        desc.MaxLOD = D3D11_FLOAT32_MAX;
        hr = device->CreateSamplerState(&desc, &sampler);
    }
    if (ok && SUCCEEDED(hr)) {
        D3D11_SAMPLER_DESC desc = {};
        desc.Filter = D3D11_FILTER_ANISOTROPIC;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.MaxAnisotropy = 8;
        desc.MaxLOD = D3D11_FLOAT32_MAX;
        hr = device->CreateSamplerState(&desc, &decal_sampler);
    }
    if (ok && SUCCEEDED(hr)) {
        D3D11_RASTERIZER_DESC desc = {};
        desc.FillMode = D3D11_FILL_SOLID;
        desc.CullMode = D3D11_CULL_NONE;
        desc.DepthClipEnable = TRUE;
        hr = device->CreateRasterizerState(&desc, &rasterizer);
    }
    if (ok && SUCCEEDED(hr)) {
        D3D11_BLEND_DESC desc = {};
        desc.RenderTarget[0].BlendEnable = TRUE;
        desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
        desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        hr = device->CreateBlendState(&desc, &blend_state);
    }
    D3D11_TEXTURE2D_DESC texture_desc = {};
    texture_desc.Width = output_width;
    texture_desc.Height = output_height;
    texture_desc.MipLevels = 1;
    texture_desc.ArraySize = 1;
    texture_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.Usage = D3D11_USAGE_DEFAULT;
    texture_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (ok && SUCCEEDED(hr))
        hr = device->CreateTexture2D(&texture_desc, nullptr, &render_texture);
    if (ok && SUCCEEDED(hr))
        hr = device->CreateRenderTargetView(render_texture, nullptr, &render_target);
    texture_desc.Usage = D3D11_USAGE_STAGING;
    texture_desc.BindFlags = 0;
    texture_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (ok && SUCCEEDED(hr))
        hr = device->CreateTexture2D(&texture_desc, nullptr, &readback_texture);
    if (ok && SUCCEEDED(hr) && (beauty_vegetation_enabled || biq_scene_enabled)) {
        D3D11_TEXTURE2D_DESC depth_desc = {};
        depth_desc.Width = output_width;
        depth_desc.Height = output_height;
        depth_desc.MipLevels = 1;
        depth_desc.ArraySize = 1;
        depth_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        depth_desc.SampleDesc.Count = 1;
        depth_desc.Usage = D3D11_USAGE_DEFAULT;
        depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        hr = device->CreateTexture2D(&depth_desc, nullptr, &feature_depth_texture);
        if (SUCCEEDED(hr))
            hr = device->CreateDepthStencilView(feature_depth_texture, nullptr, &feature_depth_view);
        if (SUCCEEDED(hr)) {
            D3D11_DEPTH_STENCIL_DESC state_desc = {};
            state_desc.DepthEnable = TRUE;
            state_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
            state_desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
            hr = device->CreateDepthStencilState(&state_desc, &feature_depth_state);
        }
    }
    ok = ok && SUCCEEDED(hr);

    if (ok) {
        float clear[] = {0.035f, 0.035f, 0.035f, 1.0f};
        D3D11_VIEWPORT viewport = {0.0f, 0.0f, static_cast<float>(output_width),
                                   static_cast<float>(output_height), 0.0f, 1.0f};
        context->ClearRenderTargetView(render_target, clear);
        if (biq_scene_enabled) {
            context->ClearDepthStencilView(feature_depth_view, D3D11_CLEAR_DEPTH, 1.0f, 0);
            context->OMSetRenderTargets(1, &render_target, feature_depth_view);
            context->OMSetDepthStencilState(feature_depth_state, 0);
        } else {
            context->OMSetRenderTargets(1, &render_target, nullptr);
        }
        context->OMSetBlendState(blend_state, nullptr, 0xffffffffu);
        context->RSSetViewports(1, &viewport);
        context->RSSetState(rasterizer);
        UINT stride = sizeof(Vertex);
        UINT offset = 0;
        context->IASetVertexBuffers(0, 1, &vertex_buffer, &stride, &offset);
        context->IASetInputLayout(input_layout);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex_shader, nullptr, 0);
        context->PSSetShader(pixel_shader, nullptr, 0);
        ID3D11ShaderResourceView * views[] = {base_view, height_view, specular_view, authored_height_view,
                                             authored_blend_view, authored_region_view, mountain_base_view,
                                             mountain_top_view, mountain_snow_view, mountain_height_view,
                                             mountain_specular_view, beach_base_view, beach_height_view,
                                             beach_specular_view, cliff_base_view, cliff_height_view,
                                             cliff_specular_view, shallow_bed_view, ocean_bed_view,
                                             water_height_view, water_large_lean0_view,
                                             water_large_lean1_view, water_small_lean0_view,
                                             water_small_lean1_view, water_foam_view,
                                             feature_base_views[0], feature_base_views[1],
                                             feature_base_views[2], feature_base_views[3],
                                             shallows_specular_view, ocean_height_view,
                                             ocean_specular_view, water_gloss_view, water_tiling_mask_view,
                                             water_non_tiling_mask_view,
                                             water_small_secondary_lean0_view,
                                             water_small_secondary_lean1_view, water_ripples_view,
                                             water_turbulence_view, coast_dark_profile_view,
                                             coast_scatter_profile_view,
                                             water_tiling_normal0_view,
                                             water_tiling_normal1_view,
                                             water_non_tiling_normal0_view,
                                             water_non_tiling_normal1_view,
                                             plains_base_view, plains_height_view,
                                             plains_specular_view, desert_base_view,
                                             desert_height_view, desert_specular_view,
                                             authored_hill_view, desert_hills_base_view,
                                             desert_hills_height_view, desert_hills_specular_view,
                                             dune_decal_base_view, dune_decal_height_view,
                                             desert_mountain_base_view,
                                             desert_mountain_stripe1_view,
                                             desert_mountain_stripe2_view,
                                             desert_mountain_stripe3_view,
                                             desert_mountain_height_view,
                                             desert_mountain_specular_view,
                                             marsh_base_view, marsh_height_view,
                                             marsh_specular_view, marsh_decal_base_view,
                                             marsh_decal_height_view,
                                             marsh_decal_specular_view,
                                             volcano_base_view, volcano_height_view,
                                             volcano_active_base_view,
                                             volcano_active_specular_view,
                                             water_decal_base_view,
                                             water_decal_height_view,
                                             grassland_decal_base_view,
                                             grassland_decal_height_view,
                                             plains_decal_base_view,
                                             plains_decal_height_view,
                                             river_base_view,
                                             river_height_view,
                                             river_specular_view,
                                             river_lean0_view,
                                             river_lean1_view,
                                             river_source_base_view,
                                             river_source_height_view,
                                             river_clutter_base_view,
                                             river_clutter_height_view,
                                             river_bank_noise_view};
        context->PSSetShaderResources(0, 89, views);
        if (l14_scene_enabled)
            context->PSSetShaderResources(98, 10, road_base_views);
        ID3D11SamplerState * samplers[] = {sampler, decal_sampler};
        context->PSSetSamplers(0, 2, samplers);
        context->PSSetConstantBuffers(0, 1, &settings_buffer);
        context->Draw(static_cast<UINT>(vertices.size()), 0);
        if (feature_geometry_enabled) {
            if (!biq_scene_enabled)
                context->ClearDepthStencilView(feature_depth_view, D3D11_CLEAR_DEPTH, 1.0f, 0);
            context->OMSetRenderTargets(1, &render_target, feature_depth_view);
            context->OMSetDepthStencilState(feature_depth_state, 0);
            UINT feature_stride = sizeof(FeatureVertex);
            UINT feature_offset = 0;
            context->IASetInputLayout(feature_input_layout);
            context->VSSetShader(feature_vertex_shader, nullptr, 0);
            context->PSSetShader(feature_pixel_shader, nullptr, 0);
            context->IASetVertexBuffers(0, 1, &feature_vertex_buffer,
                                        &feature_stride, &feature_offset);
            context->PSSetShaderResources(25, 4, feature_base_views);
            context->PSSetShaderResources(94, 4, feature_base_views + 4);
            context->PSSetShaderResources(89, 5, river_feature_base_views);
            context->PSSetShaderResources(108, 8, road_bridge_base_views);
            context->PSSetShaderResources(116, 8, resource_base_views);
            context->Draw(static_cast<UINT>(feature_vertices.size()), 0);
            context->OMSetDepthStencilState(nullptr, 0);
        }
        if (mine_geometry_enabled) {
            if (beauty_mines_only_mode)
                context->ClearDepthStencilView(feature_depth_view, D3D11_CLEAR_DEPTH, 1.0f, 0);
            context->OMSetRenderTargets(1, &render_target, feature_depth_view);
            context->OMSetDepthStencilState(feature_depth_state, 0);
            UINT feature_stride = sizeof(FeatureVertex);
            UINT feature_offset = 0;
            context->IASetInputLayout(feature_input_layout);
            context->VSSetShader(feature_vertex_shader, nullptr, 0);
            context->PSSetShader(feature_pixel_shader, nullptr, 0);
            context->IASetVertexBuffers(0, 1, &mine_vertex_buffer,
                                        &feature_stride, &feature_offset);
            context->PSSetShaderResources(116, 6, mine_base_views);
            context->PSSetShaderResources(124, 2, mine_emissive_views);
            context->Draw(static_cast<UINT>(mine_vertices.size()), 0);
            context->OMSetDepthStencilState(nullptr, 0);
        }
        if (city_geometry_enabled) {
            context->OMSetRenderTargets(1, &render_target, feature_depth_view);
            context->OMSetDepthStencilState(feature_depth_state, 0);
            UINT feature_stride = sizeof(FeatureVertex);
            UINT feature_offset = 0;
            context->IASetInputLayout(feature_input_layout);
            context->VSSetShader(feature_vertex_shader, nullptr, 0);
            context->PSSetShader(feature_pixel_shader, nullptr, 0);
            context->IASetVertexBuffers(0, 1, &city_vertex_buffer,
                                        &feature_stride, &feature_offset);
            context->PSSetShaderResources(116, 4, city_emissive_views);
            context->PSSetShaderResources(124, 4, city_base_views);
            context->Draw(static_cast<UINT>(city_vertices.size()), 0);
            if (wall_vertex_buffer != nullptr) {
                ID3D11ShaderResourceView * no_emissive[4] = {};
                context->PSSetShaderResources(116, 4, no_emissive);
                context->PSSetShaderResources(124, 1, wall_base_views);
                context->IASetVertexBuffers(0, 1, &wall_vertex_buffer,
                                            &feature_stride, &feature_offset);
                context->Draw(static_cast<UINT>(wall_vertices.size()), 0);
            }
            context->OMSetDepthStencilState(nullptr, 0);
        }
        context->CopyResource(readback_texture, render_texture);
        D3D11_MAPPED_SUBRESOURCE mapped = {};
        hr = context->Map(readback_texture, 0, D3D11_MAP_READ, 0, &mapped);
        if (SUCCEEDED(hr)) {
            ok = write_bmp(argv[3], mapped,
                           (beauty_thumbnail_mode || beauty_vegetation_thumbnail_mode ||
                            beauty_shore_thumbnail_mode ||
                            beauty_promotion_thumbnail_mode ||
                            beauty_dunes_thumbnail_mode ||
                            beauty_marsh_thumbnail_mode ||
                            beauty_volcano_thumbnail_mode ||
                            beauty_rivers_thumbnail_mode) ? 4u :
                           ((l13a_zoom2_mode || beauty_roads_zoom2_mode ||
                             beauty_railroads_zoom2_mode ||
                             beauty_resources_zoom2_mode ||
                             beauty_cities_zoom2_mode ||
                             beauty_mines_zoom2_mode) ? 2u : 1u));
            context->Unmap(readback_texture, 0);
        } else {
            ok = false;
        }
    }

    release(readback_texture);
    release(feature_depth_state);
    release(feature_depth_view);
    release(feature_depth_texture);
    release(render_target);
    release(render_texture);
    release(blend_state);
    release(rasterizer);
    release(decal_sampler);
    release(sampler);
    release(settings_buffer);
    release(feature_vertex_buffer);
    release(city_vertex_buffer);
    release(wall_vertex_buffer);
    release(mine_vertex_buffer);
    release(vertex_buffer);
    release(feature_input_layout);
    release(feature_pixel_shader);
    release(feature_vertex_shader);
    release(input_layout);
    release(pixel_shader);
    release(vertex_shader);
    for (ID3D11ShaderResourceView *& view : river_feature_base_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : feature_base_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : road_base_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : road_bridge_base_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : resource_base_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : city_base_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : city_emissive_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : wall_base_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : mine_base_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : mine_emissive_views)
        release(view);
    release(desert_mountain_specular_view);
    release(desert_mountain_height_view);
    release(desert_mountain_stripe3_view);
    release(desert_mountain_stripe2_view);
    release(desert_mountain_stripe1_view);
    release(desert_mountain_base_view);
    release(dune_decal_height_view);
    release(dune_decal_base_view);
    release(marsh_decal_specular_view);
    release(marsh_decal_height_view);
    release(marsh_decal_base_view);
    release(marsh_specular_view);
    release(marsh_height_view);
    release(marsh_base_view);
    release(volcano_element_blend_view);
    release(volcano_element_height_view);
    release(volcano_active_specular_view);
    release(river_bank_noise_view);
    release(river_clutter_height_view);
    release(river_clutter_base_view);
    release(river_source_height_view);
    release(river_source_base_view);
    release(river_lean1_view);
    release(river_lean0_view);
    release(river_specular_view);
    release(river_height_view);
    release(river_base_view);
    release(plains_decal_height_view);
    release(plains_decal_base_view);
    release(grassland_decal_height_view);
    release(grassland_decal_base_view);
    release(water_decal_height_view);
    release(water_decal_base_view);
    release(volcano_active_base_view);
    release(volcano_height_view);
    release(volcano_base_view);
    release(desert_hills_specular_view);
    release(desert_hills_height_view);
    release(desert_hills_base_view);
    release(desert_specular_view);
    release(desert_height_view);
    release(desert_base_view);
    release(plains_specular_view);
    release(plains_height_view);
    release(plains_base_view);
    release(water_non_tiling_normal1_view);
    release(water_non_tiling_normal0_view);
    release(water_tiling_normal1_view);
    release(water_tiling_normal0_view);
    release(coast_scatter_profile_view);
    release(coast_dark_profile_view);
    release(water_turbulence_view);
    release(water_ripples_view);
    release(water_small_secondary_lean1_view);
    release(water_small_secondary_lean0_view);
    release(water_non_tiling_mask_view);
    release(water_tiling_mask_view);
    release(water_gloss_view);
    release(ocean_specular_view);
    release(ocean_height_view);
    release(shallows_specular_view);
    release(water_foam_view);
    release(water_small_lean1_view);
    release(water_small_lean0_view);
    release(water_large_lean1_view);
    release(water_large_lean0_view);
    release(water_height_view);
    release(ocean_bed_view);
    release(shallow_bed_view);
    release(cliff_specular_view);
    release(cliff_height_view);
    release(cliff_base_view);
    release(beach_specular_view);
    release(beach_height_view);
    release(beach_base_view);
    release(mountain_specular_view);
    release(mountain_height_view);
    release(mountain_snow_view);
    release(mountain_top_view);
    release(mountain_base_view);
    release(authored_region_view);
    release(authored_blend_view);
    release(authored_height_view);
    release(authored_hill_view);
    for (ID3D11ShaderResourceView *& view : authored_hill_variant_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : authored_mountain_blend_variant_views)
        release(view);
    for (ID3D11ShaderResourceView *& view : authored_mountain_height_variant_views)
        release(view);
    release(specular_view);
    release(height_view);
    release(base_view);
    release(context);
    release(device);

    if (!ok) {
        std::fprintf(stderr, "terrain_lab: render failed\n");
        return 1;
    }
    std::printf("terrain_lab: %s pass -> %s (uv=%.3f normal=%.3f exposure=%.3f relief=%.1fpx hill_uv=%.3f)\n",
                argv[4], argv[3], uv_scale, normal_strength, exposure,
                (relief_mode || shadow_mode || hill_mode || mountain_mode || beauty_relief_enabled)
                    ? relief_height : 0.0f,
                hill_uv_scale);
    return 0;
}
