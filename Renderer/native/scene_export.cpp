#include "c3x_renderer_api.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::string bounded_text(char const * value, std::size_t capacity) {
    std::size_t length = 0;
    while (length < capacity && value[length] != '\0')
        ++length;
    return std::string(value, length);
}

std::string json_string(std::string const & value) {
    static char const hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char byte : value) {
        switch (byte) {
        case '\"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (byte < 0x20u || byte >= 0x80u) {
                result += "\\u00";
                result += hex[(byte >> 4u) & 0x0fu];
                result += hex[byte & 0x0fu];
            } else {
                result += static_cast<char>(byte);
            }
            break;
        }
    }
    result += '\"';
    return result;
}

bool valid_identifier(char const * value) {
    if (value == nullptr || *value == '\0')
        return false;
    for (char const * cursor = value; *cursor != '\0'; ++cursor) {
        char ch = *cursor;
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
              (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.'))
            return false;
    }
    return true;
}

bool make_parent_directories(std::string const & path) {
    for (std::size_t index = 0; index < path.size(); ++index) {
        if (path[index] != '\\' && path[index] != '/')
            continue;
        if (index == 0 || (index == 2 && path[1] == ':'))
            continue;
        std::string directory = path.substr(0, index);
        if (!CreateDirectoryA(directory.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
            return false;
    }
    return true;
}

bool sha256(std::string const & material, unsigned char digest[32]) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    bool ok = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0;
    if (ok)
        ok = BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) >= 0;
    if (ok)
        ok = BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char *>(material.data())),
                            static_cast<ULONG>(material.size()), 0) >= 0;
    if (ok)
        ok = BCryptFinishHash(hash, digest, 32, 0) >= 0;
    if (hash != nullptr)
        BCryptDestroyHash(hash);
    if (algorithm != nullptr)
        BCryptCloseAlgorithmProvider(algorithm, 0);
    return ok;
}

std::uint64_t variant_seed(c3x_renderer_i32 world_seed, std::string const & identifier,
                           c3x_renderer_i32 map_x, c3x_renderer_i32 map_y) {
    std::string material = std::to_string(world_seed);
    material.push_back('\0');
    material += identifier;
    material.push_back('\0');
    material += std::to_string(map_x);
    material.push_back('\0');
    material += std::to_string(map_y);
    unsigned char digest[32] = {};
    if (!sha256(material, digest))
        return 0;
    std::uint64_t result = 0;
    for (int index = 0; index < 8; ++index)
        result = (result << 8u) | digest[index];
    return result;
}

char const * terrain_name(int type) {
    static char const * names[] = {
        "desert", "plains", "grassland", "tundra", "flood-plain", "hills", "mountains",
        "forest", "jungle", "marsh", "volcano", "coast", "sea", "ocean"
    };
    return (type >= 0 && type < static_cast<int>(sizeof(names) / sizeof(names[0]))) ? names[type] : "unknown";
}

char const * resource_class_name(int value) {
    static char const * names[] = {"bonus", "luxury", "strategic"};
    return (value >= 0 && value < 3) ? names[value] : "unknown";
}

char const * unit_class_name(int value) {
    static char const * names[] = {"land", "sea", "air"};
    return (value >= 0 && value < 3) ? names[value] : "unknown";
}

char const * direction_name(int value) {
    static char const * names[] = {"southwest", "northeast", "east", "southeast", "south", "southwest", "west", "northwest", "north"};
    return (value >= 0 && value < 9) ? names[value] : "unknown";
}

char const * unit_action_name(int value) {
    if (value == 1)
        return "fortify";
    if (value >= 2 && value <= 14)
        return "worker-job";
    if (value == 15)
        return "intercept";
    if (value == 16)
        return "go-to";
    if (value >= 26 && value <= 33)
        return "automated";
    return "idle";
}

char const * city_size_name(int value) {
    static char const * names[] = {"town", "city", "metropolis"};
    return (value >= 0 && value < 3) ? names[value] : "town";
}

char const * season_name(int value) {
    static char const * names[] = {"summer", "fall", "winter", "spring"};
    return (value >= 0 && value < 4) ? names[value] : "summer";
}

std::string make_instance(char const * category, c3x_renderer_tile_v1 const & tile, int ordinal,
                          c3x_renderer_i32 world_seed, std::string const & fields) {
    std::string identifier = std::string(category) + ":" + std::to_string(tile.tile_x) + ":" +
                             std::to_string(tile.tile_y) + ":" + std::to_string(ordinal);
    std::ostringstream metadata;
    metadata << "{\"category\":" << json_string(category)
             << ",\"map_x\":" << tile.tile_x << ",\"map_y\":" << tile.tile_y << fields << '}';
    std::ostringstream out;
    out << "{\"id\":" << json_string(identifier)
        << ",\"category\":" << json_string(category)
        << ",\"tile_id\":\"tile:" << tile.tile_x << ':' << tile.tile_y << "\""
        << ",\"ordinal\":" << ordinal
        << ",\"anchor_px\":{\"x\":" << tile.anchor_x << ",\"y\":" << tile.anchor_y << "}"
        << ",\"variant_seed\":" << variant_seed(world_seed, identifier, tile.tile_x, tile.tile_y)
        << ",\"resolver_input\":" << metadata.str() << '}';
    return out.str();
}

bool export_scene(c3x_renderer_frame_v1 const & frame, c3x_renderer_scene_export_v1 const & request) {
    std::vector<c3x_renderer_tile_v1 const *> tiles;
    c3x_renderer_tile_v1 const * origin = nullptr;
    for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
        c3x_renderer_tile_v1 const & tile = frame.tiles[index];
        if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) != 0) {
            origin = &tile;
            break;
        }
    }
    if (origin == nullptr)
        return false;

    int basis_x_x = frame.tile_width / 2;
    int basis_x_y = frame.tile_height / 2;
    int basis_y_x = -frame.tile_width / 2;
    int basis_y_y = frame.tile_height / 2;
    std::unordered_set<std::string> coordinates;
    for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
        c3x_renderer_tile_v1 const & tile = frame.tiles[index];
        if ((tile.tile_flags & C3X_RENDERER_TILE_RENDER) == 0)
            continue;
        int delta_x = tile.tile_x - origin->tile_x;
        int delta_y = tile.tile_y - origin->tile_y;
        if (tile.anchor_x != origin->anchor_x + delta_x * basis_x_x + delta_y * basis_y_x ||
            tile.anchor_y != origin->anchor_y + delta_x * basis_x_y + delta_y * basis_y_y)
            continue;
        std::string coordinate = std::to_string(tile.tile_x) + ":" + std::to_string(tile.tile_y);
        if (coordinates.insert(coordinate).second)
            tiles.push_back(&tile);
    }
    if (tiles.empty())
        return false;

    int scroll_x = origin->anchor_x - origin->tile_x * basis_x_x - origin->tile_y * basis_y_x;
    int scroll_y = origin->anchor_y - origin->tile_x * basis_x_y - origin->tile_y * basis_y_y;
    std::string scene_id = "scene:" + std::to_string(request.world_seed) + ":" +
        std::to_string(scroll_x) + ":" + std::to_string(scroll_y) + ":" +
        std::to_string(frame.target_width) + "x" + std::to_string(frame.target_height) + ":" +
        std::to_string(frame.hour) + ":" + season_name(frame.season);

    std::vector<std::string> tile_records;
    std::vector<std::string> instances;
    for (c3x_renderer_tile_v1 const * tile : tiles) {
        std::string terrain_id = "terrain:" + std::to_string(tile->tile_x) + ":" + std::to_string(tile->tile_y) + ":0";
        std::ostringstream record;
        record << "{\"id\":\"tile:" << tile->tile_x << ':' << tile->tile_y << "\""
               << ",\"map_x\":" << tile->tile_x << ",\"map_y\":" << tile->tile_y
               << ",\"anchor_px\":{\"x\":" << tile->anchor_x << ",\"y\":" << tile->anchor_y << "}"
               << ",\"terrain\":{\"id\":" << json_string(terrain_id)
               << ",\"variant_seed\":" << variant_seed(request.world_seed, terrain_id, tile->tile_x, tile->tile_y)
               << ",\"resolver_input\":{\"category\":\"terrain\",\"map_x\":" << tile->tile_x
               << ",\"map_y\":" << tile->tile_y
               << ",\"terrain_type\":" << json_string(terrain_name(tile->terrain_type))
               << ",\"real_terrain_type\":" << json_string(terrain_name(tile->real_terrain_type))
               << ",\"sprite_index\":" << tile->square_parts
               << ",\"visibility_mask\":" << tile->tile_visibility
               << ",\"fog_status\":" << tile->fog_status
               << ",\"territory_owner_id\":" << tile->territory_owner_id << "}}}";
        tile_records.push_back(record.str());

        if (tile->feature_flags != 0) {
            std::ostringstream fields;
            fields << ",\"has_forest\":" << ((tile->feature_flags & C3X_RENDERER_FEATURE_FOREST) ? "true" : "false")
                   << ",\"has_jungle\":" << ((tile->feature_flags & C3X_RENDERER_FEATURE_JUNGLE) ? "true" : "false")
                   << ",\"has_marsh\":" << ((tile->feature_flags & C3X_RENDERER_FEATURE_MARSH) ? "true" : "false")
                   << ",\"sprite_index\":" << tile->square_parts;
            instances.push_back(make_instance("feature", *tile, 0, request.world_seed, fields.str()));
        }
        if (tile->road_mask != 0)
            instances.push_back(make_instance("road", *tile, 0, request.world_seed, ",\"road_mask\":" + std::to_string(tile->road_mask) + ",\"railroad_mask\":" + std::to_string(tile->railroad_mask)));
        if (tile->river_code != 0)
            instances.push_back(make_instance("river", *tile, 0, request.world_seed, ",\"river_mask\":" + std::to_string(tile->river_code)));

        int improvement_ordinal = 0;
        struct Improvement { c3x_renderer_u32 flag; char const * name; } improvements[] = {
            {C3X_RENDERER_IMPROVEMENT_IRRIGATION, "irrigation"},
            {C3X_RENDERER_IMPROVEMENT_MINE, "mine"},
            {C3X_RENDERER_IMPROVEMENT_TILE_BUILDING, "tile-building"},
            {C3X_RENDERER_IMPROVEMENT_POLLUTION, "pollution"},
            {C3X_RENDERER_IMPROVEMENT_CRATER, "crater"}
        };
        for (Improvement const & improvement : improvements) {
            if ((tile->improvement_flags & improvement.flag) == 0)
                continue;
            std::string fields = ",\"improvement\":" + json_string(improvement.name);
            if (improvement.flag == C3X_RENDERER_IMPROVEMENT_TILE_BUILDING)
                fields += ",\"tile_building_id\":" + std::to_string(tile->tile_building_id);
            instances.push_back(make_instance("improvement", *tile, improvement_ordinal++, request.world_seed, fields));
        }
        if (tile->resource_id >= 0) {
            std::string fields = ",\"resource_id\":" + std::to_string(tile->resource_id) +
                ",\"resource_name\":" + json_string(bounded_text(tile->resource_name, sizeof(tile->resource_name))) +
                ",\"resource_class\":" + json_string(resource_class_name(tile->resource_class)) +
                ",\"pcx_index\":" + std::to_string(tile->resource_id);
            instances.push_back(make_instance("resource", *tile, 0, request.world_seed, fields));
        }
        if (tile->city_id >= 0) {
            std::string fields = ",\"city_id\":" + std::to_string(tile->city_id) +
                ",\"city_owner_id\":" + std::to_string(tile->city_owner_id) +
                ",\"city_population\":" + std::to_string(tile->city_population) +
                ",\"owner\":" + json_string(bounded_text(tile->city_owner, sizeof(tile->city_owner))) +
                ",\"civilization\":" + json_string(bounded_text(tile->city_civilization, sizeof(tile->city_civilization))) +
                ",\"culture_group\":\"culture-group-" + std::to_string(tile->city_culture_group) + "\"" +
                ",\"era\":" + json_string(bounded_text(tile->city_era_name, sizeof(tile->city_era_name))) +
                ",\"city_style_index\":" + std::to_string(tile->city_culture_group) +
                ",\"city_size\":" + json_string(city_size_name(tile->city_size)) +
                ",\"is_capital\":" + ((tile->city_flags & C3X_RENDERER_CITY_CAPITAL) ? "true" : "false");
            instances.push_back(make_instance("city", *tile, 0, request.world_seed, fields));
        }
        if (tile->unit_type_id >= 0) {
            char const * hit_points = tile->unit_damage <= 0 ? "healthy" : (tile->unit_damage < 5 ? "damaged" : "critical");
            std::string fields = ",\"unit_id\":" + std::to_string(tile->unit_type_id) +
                ",\"unit_owner_id\":" + std::to_string(tile->unit_owner_id) +
                ",\"unit_state\":" + std::to_string(tile->unit_state) +
                ",\"unit_damage\":" + std::to_string(tile->unit_damage) +
                ",\"unit_direction\":" + std::to_string(tile->unit_direction) +
                ",\"unit_type\":" + json_string(bounded_text(tile->unit_type_name, sizeof(tile->unit_type_name))) +
                ",\"unit_class\":" + json_string(unit_class_name(tile->unit_class)) +
                ",\"owner\":" + json_string(bounded_text(tile->unit_owner, sizeof(tile->unit_owner))) +
                ",\"civilization\":" + json_string(bounded_text(tile->unit_civilization, sizeof(tile->unit_civilization))) +
                ",\"era\":" + json_string(bounded_text(tile->unit_era_name, sizeof(tile->unit_era_name))) +
                ",\"direction\":" + json_string(direction_name(tile->unit_direction)) +
                ",\"action\":" + json_string(unit_action_name(tile->unit_state)) +
                ",\"fortified\":" + (tile->unit_state == 1 ? "true" : "false") +
                ",\"hit_point_band\":" + json_string(hit_points);
            instances.push_back(make_instance("unit", *tile, 0, request.world_seed, fields));
        }
        if (tile->has_effect != 0)
            instances.push_back(make_instance("effect", *tile, 0, request.world_seed, ",\"sprite_index\":0"));
    }

    std::ostringstream out;
    out << "{\n  \"schema\": \"c3x.visible_scene.v0\",\n  \"scene_id\": " << json_string(scene_id)
        << ",\n  \"profile_id\": " << json_string(request.profile_id)
        << ",\n  \"world\": {\"seed\": " << request.world_seed
        << ", \"width_tiles\": " << request.world_width_tiles
        << ", \"height_tiles\": " << request.world_height_tiles
        << ", \"wrap_x\": " << (request.world_wrap_x ? "true" : "false")
        << ", \"wrap_y\": " << (request.world_wrap_y ? "true" : "false") << "},\n"
        << "  \"viewport\": {\"width_px\": " << frame.target_width << ", \"height_px\": " << frame.target_height
        << ", \"map_rect_px\": {\"x\": " << frame.clip_left << ", \"y\": " << frame.clip_top
        << ", \"width\": " << frame.clip_right - frame.clip_left << ", \"height\": " << frame.clip_bottom - frame.clip_top
        << "}, \"scroll_px\": {\"x\": " << scroll_x << ", \"y\": " << scroll_y << "}},\n"
        << "  \"projection\": {\"type\": \"civ3-isometric-pixel\", \"origin_tile\": {\"x\": " << origin->tile_x
        << ", \"y\": " << origin->tile_y << "}, \"origin_px\": {\"x\": " << origin->anchor_x << ", \"y\": " << origin->anchor_y
        << "}, \"tile_x_basis_px\": {\"x\": " << basis_x_x << ", \"y\": " << basis_x_y
        << "}, \"tile_y_basis_px\": {\"x\": " << basis_y_x << ", \"y\": " << basis_y_y
        << "}, \"elevation_basis_px\": {\"x\": 0, \"y\": " << -frame.tile_height / 2 << "}},\n"
        << "  \"environment\": {\"id\": \"earthlike\", \"hour\": " << frame.hour
        << ", \"season\": " << json_string(season_name(frame.season)) << "},\n  \"tiles\": [";
    for (std::size_t index = 0; index < tile_records.size(); ++index)
        out << (index == 0 ? "\n    " : ",\n    ") << tile_records[index];
    out << "\n  ],\n  \"instances\": [";
    for (std::size_t index = 0; index < instances.size(); ++index)
        out << (index == 0 ? "\n    " : ",\n    ") << instances[index];
    out << "\n  ]\n}\n";

    std::string path = request.output_path;
    if (!make_parent_directories(path))
        return false;
    std::string temporary = path + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file)
            return false;
        std::string text = out.str();
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!file)
            return false;
    }
    return MoveFileExA(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
}

} // namespace

extern "C" __declspec(dllexport) int c3x_renderer_export_scene(
    c3x_renderer_frame_v1 const * frame, c3x_renderer_scene_export_v1 const * request) {
    if (frame == nullptr || request == nullptr ||
        frame->api_version != C3X_RENDERER_API_VERSION || frame->struct_size != sizeof(*frame) ||
        request->api_version != C3X_RENDERER_API_VERSION || request->struct_size != sizeof(*request) ||
        request->output_path == nullptr || !valid_identifier(request->fixture_id) ||
        !valid_identifier(request->profile_id) || request->world_width_tiles <= 0 ||
        request->world_height_tiles <= 0 || frame->tile_count == 0 || frame->tiles == nullptr)
        return C3X_RENDERER_RESULT_BAD_ARGUMENT;
    return export_scene(*frame, *request) ? C3X_RENDERER_RESULT_OK : C3X_RENDERER_RESULT_ERROR;
}
