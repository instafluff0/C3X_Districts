#pragma once
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// A shared world-row origin bounds depth around the active view. Raster target
// placement never participates. A changed origin retires only incompatible
// retained depth; it does not invalidate resident world meshes.
struct SceneDepthBasis {
    std::int64_t world_origin=0;
    float translation=0;
};
inline SceneDepthBasis scene_depth_basis(int anchor_y,int tile_y,int tile_height,
        int view_height,int geometry_translation_y){
    auto screen_origin=std::int64_t(anchor_y)-std::int64_t(tile_y)*tile_height/2;
    auto center=std::int64_t(view_height)/2-screen_origin;
    // Center the active window instead of placing it at one edge of the range.
    auto rounded=center+2048;
    auto remainder=rounded%4096;if(remainder<0)remainder+=4096;
    auto world_origin=rounded-remainder;
    return {world_origin,float(std::int64_t(geometry_translation_y)-screen_origin-world_origin)};
}
} }
