#pragma once
#include <array>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// Static (camera-independent) scene pixels are retained in world-anchored
// regions. Each zoom lane (exact 1x canonical, and any other display zoom)
// owns a front slot that is displayed and a back slot that is refined over
// several frames. A lighting-stale front stays displayable as a preview while its
// replacement is prepared; scene edits require current pixels; see docs/performance_overhaul_20261003.md.
enum StaticRasterReason : unsigned {
    raster_fractional_phase,raster_guard_bounds,raster_scene,raster_environment,
    raster_lights,raster_shadow,raster_depth_origin,raster_anchors,raster_strip_fills,
    raster_explicit_reset,raster_projection,raster_layout,raster_error,
    raster_classification,static_raster_reason_count
};
struct StaticRasterMetrics {
    unsigned version=1,slot=0,valid=0;
    float projection=0;
    unsigned full_draws=0,restores=0,reuses=0,strip_fills=0;
    std::array<unsigned,static_raster_reason_count> reasons{};
    std::uint64_t region_revision=0,region_bytes=0,total_region_bytes=0,
        shared_viewport_bytes=0,total_gpu_bytes=0;
    unsigned sample_count=0;
};

// One shared reflection/material target may be reused only by its exact last
// writer. Capture geometry, lighting, projection, depth, extent and receiver ROI.
struct RasterScratchIdentity {
    std::array<std::uint64_t,10> scene{};
    std::array<float,10> view{};
    std::array<int,4> area{};
    bool operator==(RasterScratchIdentity const& other)const{
        return scene==other.scene && view==other.view && area==other.area;
    }
    bool operator!=(RasterScratchIdentity const& other)const{return !(*this==other);}
};

// Content identity of retained static pixels: everything except the camera
// anchor and projection, which are tracked separately so a slot can be shifted
// or resampled for display.
using StaticRasterKey=std::array<std::uint64_t,6>;

template<class Target> struct StaticRasterState {
    Target region;
    struct Rect {
        int left=0,top=0,right=0,bottom=0;
        bool empty()const{return left>=right || top>=bottom;}
        bool contains(Rect const& r)const{return r.empty() ||
            (left<=r.left && top<=r.top && right>=r.right && bottom>=r.bottom);}
        long long area()const{return empty()?0:(long long)(right-left)*(bottom-top);}
        Rect clipped(Rect const& r)const{return {left>r.left?left:r.left,top>r.top?top:r.top,
            right<r.right?right:r.right,bottom<r.bottom?bottom:r.bottom};}
        Rect joined(Rect const& r)const{
            if(r.empty())return *this;
            if(empty())return r;
            return {left<r.left?left:r.left,top<r.top?top:r.top,right>r.right?right:r.right,bottom>r.bottom?bottom:r.bottom};
        }
        // Bounding box of this rectangle minus `r`.
        Rect outside(Rect const& r)const{
            if(empty() || r.contains(*this))return {};
            Rect i=clipped(r);
            if(i.empty())return *this;
            Rect const pieces[4]={{left,top,i.left,bottom},{i.right,top,right,bottom},
                                  {i.left,top,i.right,i.top},{i.left,i.bottom,i.right,bottom}};
            Rect result{};
            for(Rect const& piece:pieces)result=result.joined(piece);
            return result;
        }
    } covered;
    // Bounds of covered pixels drawn outside the shadow receiver field of
    // their frame (they sampled no shadow page).
    Rect unshadowed;
    float projection=0,depth_translation=0,environment_hour=-1;
    float raster_scale=1;
    int environment_season=-1,camera_x=0,camera_y=0;
    std::array<float,2> translation{};
    std::int64_t depth_origin=0;
    std::uint64_t signature=0,geometry_epoch=0,lighting_revision=0,revision=0;
    unsigned shadow_builds=0;
    // Shadow caster changes up to this serial are reflected in the pixels.
    std::uint64_t shadow_serial=0;
    StaticRasterKey key{};
    // valid: covered pixels are displayable. stale: they no longer match the
    // current content key (or failed dependency validation) and are only a
    // preview. refining: this slot is a back buffer under construction.
    bool valid=false,stale=false,refining=false;
    StaticRasterMetrics metrics;
    StaticRasterState()=default;
    StaticRasterState(StaticRasterState const&)=delete;
    StaticRasterState& operator=(StaticRasterState const&)=delete;
    bool fresh(StaticRasterKey const& current)const{return valid && !stale && !refining && key==current;}
};

template<class Target> struct StaticRasterStates {
    static constexpr unsigned lanes=2,slots_per_lane=2,slot_count=lanes*slots_per_lane;
    std::array<StaticRasterState<Target>,slot_count> states;
    std::array<unsigned,lanes> front_slot{{0,2}};
    unsigned lane=0,selected=0;
    std::array<unsigned,3> layout{};
    StaticRasterStates(){for(auto& state:states)state.projection=1.f;}
    StaticRasterStates(StaticRasterStates const&)=delete;
    StaticRasterStates& operator=(StaticRasterStates const&)=delete;
    static unsigned lane_of(float projection){return projection==1.f?0u:1u;}
    StaticRasterState<Target>& current(){return states[selected];}
    StaticRasterState<Target>& front(unsigned l){return states[front_slot[l]];}
    unsigned back_index(unsigned l)const{return front_slot[l]^1u;}
    StaticRasterState<Target>& back(unsigned l){return states[back_index(l)];}
    StaticRasterState<Target>& select_lane(unsigned l){lane=l;selected=front_slot[l];return current();}
    // The back slot replaces the front only after its required coverage is complete.
    void promote(unsigned l){
        front_slot[l]=back_index(l);selected=lane==l?front_slot[l]:selected;
        auto& old=states[back_index(l)];old.refining=false;old.stale=true;
    }
    void count(unsigned slot,StaticRasterReason reason){
        auto& state=states[slot];if(state.metrics.full_draws)++state.metrics.reasons[reason];
    }
    // Lighting may refine behind a preview. Changed geometry/visibility cannot
    // be combined with current dynamic water, borders and objects.
    void invalidate(unsigned slot,StaticRasterReason reason){
        auto& state=states[slot];state.stale=true;++state.revision;count(slot,reason);
        if(reason==raster_scene)state.valid=false;
    }
    void invalidate_all(StaticRasterReason reason){for(unsigned i=0;i<slot_count;++i)invalidate(i,reason);}
    // Pixels are unusable even as a preview (device error, resize, explicit reset).
    void discard(unsigned slot,StaticRasterReason reason){
        auto& state=states[slot];count(slot,reason);
        state.valid=false;state.stale=false;state.refining=false;state.covered={};state.unshadowed={};++state.revision;
    }
    void discard_all(StaticRasterReason reason){for(unsigned i=0;i<slot_count;++i)discard(i,reason);}
    void set_layout(unsigned width,unsigned height,unsigned samples){
        std::array<unsigned,3> next={width,height,samples};
        if(next==layout)return;
        discard_all(raster_layout);
        for(auto& state:states)state.region.reset();
        layout=next;
    }
    bool refining()const{for(auto const& state:states)if(state.refining)return true;return false;}
    bool lane_refining(unsigned l)const{return states[2*l].refining || states[2*l+1].refining;}
    std::size_t bytes()const{std::size_t result=0;for(auto const& state:states)result+=state.region.bytes();return result;}
};
}}
