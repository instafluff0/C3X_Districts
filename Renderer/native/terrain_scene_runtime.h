#ifndef C3X_TERRAIN_SCENE_RUNTIME_H
#define C3X_TERRAIN_SCENE_RUNTIME_H

#include <cstdint>
#include <string>
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

struct FeatureBundle {
    std::vector<std::string> texture_paths;
    std::vector<FeatureAsset> assets;
    std::vector<FeatureGroup> groups;
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
