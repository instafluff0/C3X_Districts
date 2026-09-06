#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "terrain_scene_runtime.h"

namespace c3x_renderer {
namespace {

constexpr std::uint64_t fnv_offset = 1469598103934665603ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;
void hash_bytes(std::uint64_t & hash, void const * data, std::size_t size) {
    auto const * bytes = static_cast<std::uint8_t const *>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= fnv_prime;
    }
}

template <typename T>
void hash_value(std::uint64_t & hash, T const & value) {
    hash_bytes(hash, &value, sizeof(value));
}

bool read_file(std::string const & path, std::vector<std::uint8_t> & output,
               std::size_t limit = 64u * 1024u * 1024u) {
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return false;
    LARGE_INTEGER size = {};
    bool ok = GetFileSizeEx(file, &size) != 0 && size.QuadPart > 0 &&
        static_cast<unsigned long long>(size.QuadPart) <= limit;
    if (ok) {
        output.resize(static_cast<std::size_t>(size.QuadPart));
        DWORD read = 0;
        ok = ReadFile(file, output.data(), static_cast<DWORD>(output.size()), &read, nullptr) != 0 &&
            read == static_cast<DWORD>(output.size());
    }
    CloseHandle(file);
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

bool consume_u32(std::vector<std::uint8_t> const & data, std::size_t & cursor,
                 std::uint32_t & output) {
    if (cursor + 4 > data.size())
        return false;
    output = read_u32(data, cursor);
    cursor += 4;
    return true;
}

bool consume_float(std::vector<std::uint8_t> const & data, std::size_t & cursor,
                   float & output) {
    std::uint32_t bits = 0;
    if (!consume_u32(data, cursor, bits))
        return false;
    std::memcpy(&output, &bits, sizeof(output));
    return std::isfinite(output);
}

bool consume_string(std::vector<std::uint8_t> const & data, std::size_t & cursor,
                    std::string & output) {
    std::uint32_t bytes = 0;
    if (!consume_u32(data, cursor, bytes) || bytes == 0 || bytes > 4096 ||
        cursor + bytes > data.size())
        return false;
    output.assign(reinterpret_cast<char const *>(data.data() + cursor), bytes);
    cursor += bytes;
    return true;
}

float smoothstep(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

std::uint32_t feature_hash(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

} // namespace

bool load_feature_bundle(std::string const & path, FeatureBundle & output) {
    output = {};
    std::vector<std::uint8_t> data;
    if (!read_file(path, data) || data.size() < 24 ||
        std::memcmp(data.data(), "C3XVEG1\0", 8) != 0)
        return false;
    std::size_t cursor = 8;
    std::uint32_t version = 0, texture_count = 0, asset_count = 0, group_count = 0;
    if (!consume_u32(data, cursor, version) || !consume_u32(data, cursor, texture_count) ||
        !consume_u32(data, cursor, asset_count) || !consume_u32(data, cursor, group_count) ||
        version != 1 || texture_count == 0 || texture_count > 8 || asset_count == 0 ||
        asset_count > 256 || group_count == 0 || group_count > 16)
        return false;
    output.texture_paths.resize(texture_count);
    for (std::string & texture : output.texture_paths)
        if (!consume_string(data, cursor, texture) ||
            texture.find("..") != std::string::npos ||
            texture.find(':') != std::string::npos ||
            texture.front() == '/' || texture.front() == '\\')
            return false;
    output.assets.resize(asset_count);
    for (FeatureAsset & asset : output.assets) {
        std::uint32_t vertex_count = 0, index_count = 0;
        if (!consume_string(data, cursor, asset.id) ||
            !consume_u32(data, cursor, asset.texture_index) ||
            !consume_u32(data, cursor, vertex_count) || !consume_u32(data, cursor, index_count) ||
            asset.texture_index >= texture_count || vertex_count < 3 || vertex_count > 100000 ||
            index_count < 3 || index_count > 300000 || index_count % 3 != 0)
            return false;
        asset.vertices.resize(vertex_count);
        for (FeatureSourceVertex & vertex : asset.vertices) {
            for (float & value : vertex.position) if (!consume_float(data, cursor, value)) return false;
            for (float & value : vertex.normal) if (!consume_float(data, cursor, value)) return false;
            for (float & value : vertex.uv) if (!consume_float(data, cursor, value)) return false;
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
    return cursor == data.size();
}

FeatureGroup const * find_feature_group(FeatureBundle const & bundle, char const * name) {
    for (FeatureGroup const & group : bundle.groups)
        if (group.name == name)
            return &group;
    return nullptr;
}

FeaturePlacement const * select_feature_placement(FeatureGroup const & group,
                                                  std::uint32_t seed) {
    std::uint32_t total = 0;
    for (FeaturePlacement const & placement : group.placements)
        total += placement.count;
    if (total == 0)
        return nullptr;
    std::uint32_t selected = feature_hash(seed) % total;
    for (FeaturePlacement const & placement : group.placements) {
        if (selected < placement.count)
            return &placement;
        selected -= placement.count;
    }
    return nullptr;
}

FeaturePlacement const * find_feature_placement_by_suffix(FeatureBundle const & bundle,
                                                          FeatureGroup const & group,
                                                          char const * suffix) {
    if (suffix == nullptr)
        return nullptr;
    std::string expected = suffix;
    for (FeaturePlacement const & placement : group.placements) {
        if (placement.asset_index >= bundle.assets.size() || placement.scale <= 0.0f)
            continue;
        std::string const & id = bundle.assets[placement.asset_index].id;
        if (id.size() >= expected.size() &&
            id.compare(id.size() - expected.size(), expected.size(), expected) == 0)
            return &placement;
    }
    return nullptr;
}

float stable_random(std::uint32_t value) {
    return static_cast<float>(feature_hash(value) & 0x00ffffffu) / 16777215.0f;
}

float dune_height(float world_x, float world_y, float desert_weight) {
    if (desert_weight <= 0.0f)
        return 0.0f;
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
    float windward = smoothstep(wave);
    float crest = windward * windward * (1.18f - 0.18f * windward);
    float fine_wave = 0.5f + 0.5f * std::sin(along * 13.1f + across * 1.4f + 0.9f);
    return desert_weight * (crest * 17.0f + fine_wave * 1.6f);
}

TerrainFrameSignature terrain_frame_signature(c3x_renderer_frame_v1 const & frame,
                                               std::uint64_t content_revision,
                                               std::uint32_t device_generation) {
    TerrainFrameSignature result = {};
    result.camera = fnv_offset;
    // The clip controls which portion Civ III asks us to composite, not the
    // identity of the retained terrain viewport. Render/cache the complete
    // viewport so unit- or UI-only dirty rectangles cannot evict static map art.
    for (auto value : {frame.target_width, frame.target_height,
                       frame.tile_width, frame.tile_height})
        hash_value(result.camera, value);
    result.scene = fnv_offset;
    result.ownership = fnv_offset;
    for (c3x_renderer_u32 index = 0; index < frame.tile_count; ++index) {
        c3x_renderer_tile_v1 const & tile = frame.tiles[index];
        for (auto value : {tile.tile_x, tile.tile_y, tile.anchor_x, tile.anchor_y,
                           tile.terrain_type, tile.real_terrain_type})
            hash_value(result.scene, value);
        // Cache only state that changes custom-renderer pixels. Civ III draw
        // selectors, native overlay bits, fog traversal, and exact population
        // remain authoritative capture data but do not belong to this static
        // terrain plane.
        for (auto value : {tile.variant_seed, tile.tile_flags, tile.feature_flags,
                           tile.improvement_flags,
                           tile.has_effect, tile.river_code, tile.road_mask,
                           tile.railroad_mask, static_cast<c3x_renderer_u32>(tile.route_style),
                           static_cast<c3x_renderer_u32>(tile.resource_id),
                           static_cast<c3x_renderer_u32>(tile.resource_class),
                           static_cast<c3x_renderer_u32>(tile.city_id),
                           static_cast<c3x_renderer_u32>(tile.city_owner_id),
                           static_cast<c3x_renderer_u32>(tile.city_size),
                           static_cast<c3x_renderer_u32>(tile.city_culture_group),
                           static_cast<c3x_renderer_u32>(tile.city_era), tile.city_flags})
            hash_value(result.scene, value);
        hash_bytes(result.scene, tile.resource_name, sizeof(tile.resource_name));
        hash_value(result.ownership, tile.tile_flags);
        hash_value(result.ownership, tile.feature_flags);
    }
    result.environment = fnv_offset;
    hash_value(result.environment, frame.hour);
    hash_value(result.environment, frame.season);
    result.wrap = fnv_offset;
    hash_value(result.wrap, frame.world_width_tiles);
    hash_value(result.wrap, frame.world_height_tiles);
    hash_value(result.wrap, frame.world_wrap_x);
    hash_value(result.wrap, frame.world_wrap_y);
    result.complete = fnv_offset;
    hash_value(result.complete, result.camera);
    hash_value(result.complete, result.scene);
    hash_value(result.complete, result.environment);
    hash_value(result.complete, result.wrap);
    hash_value(result.complete, result.ownership);
    hash_value(result.complete, content_revision);
    hash_value(result.complete, device_generation);
    return result;
}

} // namespace c3x_renderer
