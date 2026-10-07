#ifndef C3X_TERRAIN_SCENE_RUNTIME_H
#define C3X_TERRAIN_SCENE_RUNTIME_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "c3x_renderer_api.h"

namespace c3x_renderer {

struct FeatureSourceVertex {
    float position[3];
    float normal[3];
    float uv[2];
};

struct FeatureAsset {
    std::string id;
    std::uint32_t texture_index = 0;
    std::vector<FeatureSourceVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct FeaturePlacement {
    std::uint32_t asset_index = 0;
    float scale = 1.0f;
    float scale_variation = 0.0f;
    std::uint32_t count = 0;
    std::uint32_t min_count = 0;
    std::uint32_t priority = 0;
    std::uint32_t flags = 0;
    float width = 0.0f;
    float low_end_reduction = 0.0f;
};

struct FeatureGroup {
    std::string name;
    std::vector<FeaturePlacement> placements;
};

// Offline-baked tile composition: every model instance, including flat
// "decal/" ground meshes, is already positioned (tile UV), rotated, scaled and
// sunk by the asset pipeline. Instances draw in order (decals first). A
// variant applies to the Civ III real terrain types in its mask; ground_fit
// settles an instance on the lowest natural ground within that tile radius.
struct FeatureComposition {
    struct Instance {std::uint32_t asset = 0; float u = .5f, v = .5f, rotation = 0, scale = 1, lift = 0, ground_fit = 0;};
    struct Variant {std::uint32_t terrain_mask = ~0u; std::vector<Instance> instances;};
    std::string name;
    std::vector<Variant> variants;
    bool animated = false;   // some instance places an "animated/<binding>" subject
};

struct FeatureBundle {
    std::vector<std::string> texture_paths;
    std::vector<FeatureAsset> assets;
    std::vector<FeatureGroup> groups;
    std::vector<FeatureComposition> compositions;
    std::vector<std::pair<std::string, std::uint32_t>> composition_aliases;
};

struct TerrainFrameSignature {
    std::uint64_t complete = 0;
    std::uint64_t camera = 0;
    std::uint64_t scene = 0;
    std::uint64_t geometry = 0;
    std::uint64_t environment = 0;
    std::uint64_t wrap = 0;
    std::uint64_t ownership = 0;
};

bool load_feature_bundle(std::string const & path, FeatureBundle & output);
FeatureGroup const * find_feature_group(FeatureBundle const & bundle, char const * name);
// Exact, case-insensitive match of a map resource name against baked aliases.
FeatureComposition const * find_feature_composition(FeatureBundle const & bundle, char const * name);
// Stable variant for a tile, preferring variants authored for its terrain.
FeatureComposition::Variant const & select_composition_variant(FeatureComposition const & composition,
                                                               int real_terrain_type, std::uint32_t seed);
FeaturePlacement const * select_feature_placement(FeatureGroup const & group,
                                                  std::uint32_t seed);
FeaturePlacement const * find_feature_placement_by_suffix(FeatureBundle const & bundle,
                                                          FeatureGroup const & group,
                                                          char const * suffix);
float stable_random(std::uint32_t value);
std::uint32_t stable_hash(std::uint32_t value);
float dune_height(float world_x, float world_y, float desert_weight);
TerrainFrameSignature terrain_frame_signature(c3x_renderer_frame_v1 const & frame,
                                               std::uint64_t content_revision,
                                               std::uint32_t device_generation);

} // namespace c3x_renderer

#endif
