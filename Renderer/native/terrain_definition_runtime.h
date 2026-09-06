#ifndef C3X_TERRAIN_DEFINITION_RUNTIME_H
#define C3X_TERRAIN_DEFINITION_RUNTIME_H

#include <array>
#include <string>

namespace c3x_renderer {

constexpr int terrain_type_count = 14;

struct TerrainAssetBinding {
    bool configured = false;
    std::string pack_root;
    std::string logical_asset_id;
};

struct RendererPackRoots {
    std::string vegetation;
    std::string decals;
    std::string terrain_elements;
    std::string shore;
};

bool load_terrain_definition_layers(
    char const * mod_root,
    char const * default_path,
    char const * scenario_path,
    char const * custom_path,
    std::array<TerrainAssetBinding, terrain_type_count> & bindings,
    RendererPackRoots & companion_packs,
    std::string & diagnostic);

} // namespace c3x_renderer

#endif
