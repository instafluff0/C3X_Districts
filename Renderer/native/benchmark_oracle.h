#ifndef C3X_RENDERER_BENCHMARK_ORACLE_H
#define C3X_RENDERER_BENCHMARK_ORACLE_H

#include <cstdint>

// Standalone benchmark contract only. This is deliberately separate from the
// production renderer ABI consumed by injected_code.c.
#define C3X_RENDERER_BENCHMARK_ORACLE_VERSION 1u

#pragma pack(push, 8)
struct c3x_renderer_benchmark_oracle_trim_v1 {
    std::uint32_t version;
    std::uint32_t struct_size;
    std::uint64_t cleared_viewport_bytes;
    std::uint64_t cleared_region_bytes;
    std::uint64_t cleared_pixel_block_bytes;
    std::uint64_t cleared_backdrop_bytes;
    std::uint64_t cleared_publication_bytes;
    std::uint64_t retained_geometry_bytes;
    std::uint64_t retained_natural_bytes;
    std::uint64_t retained_ground_bytes;
    std::uint64_t retained_wave_bytes;
    std::uint64_t retained_unit_pose_bytes;
    std::uint64_t retained_unit_payload_bytes;
    std::uint64_t retained_shadow_bytes;
    std::uint64_t retained_other_bytes;
    std::uint32_t retained_geometry_entries;
    std::uint32_t retained_unit_pose_entries;
    std::uint32_t retained_wave_entries;
    std::uint32_t capacity_geometry_evictions;
    std::uint32_t capacity_pose_evictions;
    std::uint32_t reserved[3];
};
#pragma pack(pop)

using c3x_renderer_benchmark_trim_to_prepared_v1_fn = int (*) (
    c3x_renderer_benchmark_oracle_trim_v1 *);

#endif
