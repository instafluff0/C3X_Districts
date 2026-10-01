#pragma once
#include <array>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// One canonical raster and one current display raster. World/asset owners and
// live frame scratch remain outside these slots. Target must own/reset itself.
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

template<class Target> struct StaticRasterState {
    Target region;
    struct Rect {int left=0,top=0,right=0,bottom=0;} covered;
    float projection=0,depth_translation=0,environment_hour=-1;
    int environment_season=-1,camera_x=0,camera_y=0;
    std::array<float,2> translation{};
    std::int64_t depth_origin=0;
    std::uint64_t signature=0,geometry_epoch=0,lighting_revision=0,revision=0;
    unsigned shadow_builds=0;
    bool valid=false;
    StaticRasterMetrics metrics;
    StaticRasterState()=default;
    StaticRasterState(StaticRasterState const&)=delete;
    StaticRasterState& operator=(StaticRasterState const&)=delete;
};

template<class Target> struct StaticRasterStates {
    std::array<StaticRasterState<Target>,2> states;
    unsigned selected=0;
    struct Writer {
        unsigned slot=2;
        std::uint64_t revision=0;
        int camera_x=0,camera_y=0;
        float depth_translation=0;
    } writer;
    std::array<unsigned,3> layout{};
    StaticRasterStates(){states[0].projection=1.f;}
    StaticRasterStates(StaticRasterStates const&)=delete;
    StaticRasterStates& operator=(StaticRasterStates const&)=delete;
    StaticRasterState<Target>& current(){return states[selected];}
    void invalidate(unsigned slot,StaticRasterReason reason){
        auto& state=states[slot];state.valid=false;++state.revision;
        if(state.metrics.full_draws)++state.metrics.reasons[reason];
        if(writer.slot==slot)writer.slot=2;
    }
    void invalidate_all(StaticRasterReason reason){
        invalidate(0,reason);invalidate(1,reason);writer.slot=2;
    }
    StaticRasterState<Target>& select(float projection){
        selected=projection==1.f?0u:1u;
        auto& state=current();
        if(state.projection!=projection){
            invalidate(selected,raster_projection);state.projection=projection;
        }
        return state;
    }
    // Layout belongs to both owners even if only the other slot has run since
    // resize/sample changes. Release old storage; display allocation stays lazy.
    void set_layout(unsigned width,unsigned height,unsigned samples){
        std::array<unsigned,3> next={width,height,samples};
        if(next==layout)return;
        invalidate_all(raster_layout);
        for(auto& state:states)state.region.reset();
        layout=next;
    }
    // Call before any full/strip mutation. A partial failed blend cannot be
    // described as a valid raster, and an old viewport writer cannot alias it.
    void begin_write(){
        auto& state=current();state.valid=false;++state.revision;
        if(writer.slot==selected)writer.slot=2;
    }
    bool needs_restore(bool cache_ready,int x,int y,float depth)const{
        auto const& state=states[selected];
        return !cache_ready || writer.slot!=selected || writer.revision!=state.revision ||
            writer.camera_x!=x || writer.camera_y!=y || writer.depth_translation!=depth;
    }
    void begin_restore(){writer.slot=2;}
    // Only after restore AND resolve succeed. Selection itself never claims a
    // valid shared viewport image, even if camera/depth values happen to match.
    void restored(int x,int y,float depth){
        writer={selected,current().revision,x,y,depth};++current().metrics.restores;
    }
    std::size_t bytes()const{return states[0].region.bytes()+states[1].region.bytes();}
};
}}
