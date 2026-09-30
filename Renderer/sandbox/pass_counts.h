#pragma once
#include <array>
#include <cstdint>
#include <cstdio>

// Opt-in CPU submission accounting. These are submitted index references and
// target footprints, not shader invocations, GPU time or scanout measurements.
struct SandboxPassCounts {
    enum Pass { selection, shadow, main_scene, reflection_scene, water,
        main_material, reflection_material, relight, units, reflected_units, unit_shadow,
        reconstruction, publication, pass_count };
    static constexpr unsigned layers=64, screen=layers-1;
    struct Counts {
        std::uint64_t tested_records=0,accepted_records=0;
        std::uint64_t tested_instances=0,accepted_instances=0;
        std::uint64_t submitted_instances=0,index_vertices=0,triangles=0;
        std::uint64_t draws=0,upload_bytes=0,target_pixels=0,copy_pixels=0;
        std::uint64_t rebuilds=0,reuses=0;
    };
    std::array<std::array<Counts,layers>,pass_count> counts{};
};

inline void sandbox_report_pass_counts(SandboxPassCounts const& work,std::size_t frame,char const* view="zoom") {
    for(unsigned pass=0;pass<SandboxPassCounts::pass_count;++pass)
        for(unsigned layer=0;layer<SandboxPassCounts::layers;++layer){
            auto const& c=work.counts[pass][layer];
            if(!c.tested_records&&!c.tested_instances&&!c.draws&&!c.upload_bytes&&!c.target_pixels&&!c.copy_pixels&&!c.rebuilds&&!c.reuses)continue;
            std::printf("CLIENT_PASS view=%s frame=%zu pass=%u layer=%u tested=%llu accepted=%llu tested_instances=%llu accepted_instances=%llu submitted_instances=%llu index_vertices=%llu triangles=%llu draws=%llu upload_bytes=%llu target_pixels=%llu copy_pixels=%llu rebuilds=%llu reuses=%llu\n",
                view,frame,pass,layer,c.tested_records,c.accepted_records,c.tested_instances,c.accepted_instances,c.submitted_instances,c.index_vertices,c.triangles,c.draws,c.upload_bytes,c.target_pixels,c.copy_pixels,c.rebuilds,c.reuses);
        }
}
