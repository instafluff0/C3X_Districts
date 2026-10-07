#pragma once
#include <algorithm>
#include <cstddef>
// Logical D3D allocation bytes, not physical VRAM or process virtual address
// usage. Native composition and simultaneous publication occupancy reduce the
// optional-cache allowance. Content growth uses measured process headroom, which
// already includes assets, driver mappings, native composition and Civ III.
namespace c3x_renderer { namespace render_core {
struct FrameWorkingSet {
    static constexpr std::size_t mib=1024u*1024u,limit=1024u*mib;
#if defined(_WIN64)
    // Renderer64 owns the large scene separately from Civ III. The old 1 GiB
    // combined envelope left no room for reusable unit content on a normal
    // full-screen view, despite several GiB of free physical memory.
    static constexpr std::size_t cache_limit=1792u*mib;
#else
    static constexpr std::size_t cache_limit=limit;
#endif
    static constexpr std::size_t scene_limit=672u*mib,unit_limit=96u*mib;
    struct Extent {unsigned width,height;};
    static Extent unit_scratch(unsigned w,unsigned h,unsigned previous_w,unsigned previous_h){
        // Different unit packs use different sprite/sample extents. Reuse the
        // larger allocation without changing the native raster viewport. Do not
        // combine a wide and a tall request into an over-budget square.
        auto width=std::max(w,previous_w),height=std::max(h,previous_h);
        if(std::size_t(width)*height*48u>unit_limit)return {w,h};
        return {width,height};
    }
    static std::size_t scene(unsigned w,unsigned h){return std::size_t(w+8)*(h+8)*240u;}
    static std::size_t mirror(unsigned w,unsigned h){return std::size_t(w+16)*(h+16)*32u;}
    // VA and logical GPU bytes are different measurements. Subtract a future
    // compiler/scratch reserve from actual free VA, not from a guessed VRAM sum.
    struct Content {std::size_t geometry,preparation;};
    struct Residency {std::size_t geometry,prepared;};
    // Current physical availability already includes retained CPU data, driver
    // mappings and all other owners. Reserve future compiler/publication work,
    // then divide growth between compact recipes and conservatively mirrored
    // GPU geometry. Adapter headroom independently bounds GPU growth.
    static Residency residency(std::size_t available,std::size_t physical,
            std::size_t geometry,std::size_t prepared,std::size_t gpu_headroom,
            std::size_t ceiling,unsigned workers){
        auto lanes=std::min(workers,6u);
        auto reserve=std::max(2048u*mib,physical/6)+512u*mib+
            std::max(2u,lanes+1u)*16u*mib+lanes*48u*mib;
        auto usable=available>reserve?available-reserve:0;
        auto compact_growth=usable/4;
        auto geometry_growth=std::min((usable-compact_growth)/2,gpu_headroom-gpu_headroom/5);
        auto shortfall=available<reserve?reserve-available:0;
        auto retained=geometry>shortfall/2?geometry-shortfall/2:0;
        return {std::min(ceiling,retained+geometry_growth),prepared+compact_growth};
    }
    // After the required compact recipes exist, do not reserve another quarter
    // of growth for duplicate recipes. Measurements already include live
    // sources/recipes/targets. Only missing future attachments and compiler /
    // publication overlap are reserved; each new GPU byte can need a driver
    // backing byte too. A missing adapter measurement permits no new uploads.
    static std::size_t world_geometry(std::size_t available,std::size_t physical,
            std::size_t owned,std::size_t gpu_headroom,std::size_t future,
            std::size_t ceiling,unsigned workers,bool loading){
        auto lanes=std::min(workers,6u);
        auto reserve=std::max(2048u*mib,physical/6)+512u*mib+
            std::max(2u,lanes+1u)*16u*mib+lanes*48u*mib;
        auto usable=available>reserve?available-reserve:0;
        usable=usable>future?usable-future:0;
        auto gpu=gpu_headroom>future?gpu_headroom-future:0;
        auto growth=std::min(usable/2,gpu-gpu/5);
        // Loading never evicts surviving content. Foreground still reduces
        // its allowance under physical pressure, as the existing residency
        // policy does; selected-content admission owns that fallback.
        auto shortfall=available<reserve?reserve-available:0;
        auto retained=loading?owned:owned>shortfall/2?owned-shortfall/2:0;
        if(loading && retained>=ceiling)return retained;
        return std::min(ceiling,retained+growth);
    }
    // Geometry the current camera job itself requires, once nothing older can
    // be evicted. Refusing it frees nothing; it fails the job and leaves the
    // map stale or black. Keep the system floor and active compile lanes, but
    // not the optional future reserve, and never shrink below what is owned:
    // the shortfall shrink above, reapplied each frame, otherwise falls below
    // a view that is already resident.
    static std::size_t required_geometry(std::size_t available,std::size_t physical,
            std::size_t owned,std::size_t gpu_headroom,std::size_t ceiling,unsigned workers){
        auto lanes=std::min(workers,6u);
        auto floor=std::max(2048u*mib,physical/6)+std::max(2u,lanes+1u)*16u*mib+lanes*48u*mib;
        auto usable=available>floor?available-floor:0;
        return std::min(ceiling,owned+std::min(usable/2,gpu_headroom-gpu_headroom/5));
    }
    static Content content(std::size_t available,std::size_t resident,std::size_t ceiling,unsigned workers){
        // Keep one ready slot beside every configured active lane. Reducing
        // this allowance under pressure serializes demanded compilation as soon
        // as even a small ready result occupies the pool; cap growth instead.
        auto preparation=std::max(2u,std::min(workers,6u)+1u)*16u*mib;
        // A backing lane can simultaneously hold bounded packed bytes, decoded
        // raw bytes and compiler/output scratch (three 16 MiB allocations).
        auto reserve=512u*mib+preparation+std::min(workers,6u)*48u*mib;
        auto growth=available>reserve?available-reserve:0;
        auto shortfall=available<reserve?reserve-available:0;
        auto capacity=resident>shortfall?resident-shortfall:0;
        // Protect the established supported geometry working set. Required
        // selected content still uses the existing admission/fallback contract.
        capacity=std::max(384u*mib,capacity+std::min(ceiling,growth));
        return {std::min(ceiling,capacity),preparation};
    }
    struct Caches {std::size_t regions,units;};
    static Caches caches(std::size_t attachments,bool pressure,bool shared_scene,std::size_t allowance=cache_limit){
        auto remaining=attachments<allowance?allowance-attachments:0;
        auto units=std::min(remaining,(pressure?48u:192u)*mib);
        auto regions=shared_scene?0:std::min(remaining-units,(pressure?64u:256u)*mib);
        return {regions,units};
    }
};
}}
