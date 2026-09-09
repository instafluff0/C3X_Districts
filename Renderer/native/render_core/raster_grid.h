#pragma once
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// A camera translates both the world anchor and the render-region grid. This
// keeps a world's pixel at the same local raster coordinate after translation.
// Extents are the bounded, positive region sizes chosen by the renderer.
inline int raster_phase(std::int64_t value,int extent) {
    int phase=int(value%extent);return phase<0?phase+extent:phase;
}
inline int raster_region_floor(int value,int phase,int extent) {
    return value-raster_phase(std::int64_t(value)-phase,extent);
}
inline int raster_anchor_phase(int anchor,int coordinate,int tile_extent,int region_extent) {
    return raster_phase(std::int64_t(anchor)-std::int64_t(coordinate)*tile_extent/2,region_extent);
}
inline bool raster_translation_matches(int previous,int current,int delta,int extent) {
    return raster_phase(std::int64_t(previous)+delta,extent)==current;
}
// A bounded region grid may start before the viewport. Keep the full region
// origin for world identity and rendering; clip only the final output copy.
struct RasterRegionAxis {
    int origin, count, extent;
    RasterRegionAxis(int pixels,int phase,int side=128)
        : origin(raster_region_floor(0,phase,side)),count((pixels-origin+side-1)/side),extent(side) {}
    int at(int pixel) const {return (pixel-origin)/extent;}
    int start(int index) const {return origin+index*extent;}
};
inline std::uint64_t static_region_identity(std::uint64_t geometry,std::uint64_t topology_revision) {
    // Retained topology can change even when this frame carries no topology
    // payload. Never infer unchanged shadow/coast data from a zero count.
    return (geometry^topology_revision)*1099511628211ull;
}
} }
