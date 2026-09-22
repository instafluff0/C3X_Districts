#pragma once
#include <algorithm>
#include <cstddef>
// Logical D3D allocation bytes, not physical VRAM or process virtual address
// usage. Native composition and simultaneous publication occupancy reduce the
// optional-cache allowance. Geometry/assets keep their existing content budgets;
// whole-process VA pressure includes their allocations and Civ III itself.
namespace c3x_renderer { namespace render_core {
struct FrameWorkingSet {
    static constexpr std::size_t mib=1024u*1024u,limit=1024u*mib;
    static constexpr std::size_t scene_limit=672u*mib,unit_limit=96u*mib;
    static std::size_t scene(unsigned w,unsigned h){return std::size_t(w+8)*(h+8)*240u;}
    static std::size_t mirror(unsigned w,unsigned h){return std::size_t(w+16)*(h+16)*32u;}
    struct Caches {std::size_t regions,units;};
    static Caches caches(std::size_t attachments,bool pressure,bool shared_scene){
        auto remaining=attachments<limit?limit-attachments:0;
        auto units=std::min(remaining,(pressure?48u:192u)*mib);
        auto regions=shared_scene?0:std::min(remaining-units,(pressure?64u:256u)*mib);
        return {regions,units};
    }
};
}}
