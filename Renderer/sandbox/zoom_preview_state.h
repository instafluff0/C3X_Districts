#pragma once
#include <cmath>
#include <cstdint>

namespace c3x_renderer { namespace sandbox {
// Pure scheduling/authority policy. The caller serializes access and owns GPU
// resources; these descriptors never own pixels or submit rendering work.
struct ZoomPreviewIdentity {
    std::int64_t map_epoch=0,scene_epoch=0,viewer_epoch=0,visibility_epoch=0;
    int camera_x=0,camera_y=0,native_width=128,target_width=0,target_height=0;
    bool operator==(ZoomPreviewIdentity const& b)const {
        return map_epoch==b.map_epoch&&scene_epoch==b.scene_epoch&&
            viewer_epoch==b.viewer_epoch&&visibility_epoch==b.visibility_epoch&&
            camera_x==b.camera_x&&camera_y==b.camera_y&&native_width==b.native_width&&
            target_width==b.target_width&&target_height==b.target_height;
    }
    bool operator!=(ZoomPreviewIdentity const& b)const{return !(*this==b);}
};
struct ZoomPreviewSource {
    // id names an immutable FULL-QUALITY output, never a prior preview. Coverage
    // is the smallest absolute zoom whose complete target rectangle is valid.
    // The caller must establish that coverage from the actual retained pixels.
    std::uint64_t id=0,request_serial=0,source_time_ms=0,completed_time_ms=0;
    ZoomPreviewIdentity identity{};
    double zoom=1.,coverage_min_zoom=1.;
};
struct ZoomPreviewRefinement {
    bool valid=false,refresh_wide=false;
    std::uint64_t id=0,request_serial=0,source_time_ms=0;
    ZoomPreviewIdentity identity{};
    double zoom=1.;
};
struct ZoomPreviewFrame {
    bool valid=false;
    ZoomPreviewSource source{};
    std::uint64_t request_serial=0;
    double absolute_zoom=1.,relative_zoom=1.;
    std::uint64_t source_age_ms=0;
};
class ZoomPreviewState {
    ZoomPreviewIdentity wanted_identity{},displayed_identity{};
    ZoomPreviewSource wide{},latest{};
    ZoomPreviewRefinement active{};
    std::uint64_t serial=0,next_job=0,last_input=0;
    double wanted=1.,displayed=1.;
    bool requested=false,presented=false;
    static bool absolute(double zoom){return std::isfinite(zoom)&&zoom>=1.&&zoom<=3.;}
    static std::uint64_t age(std::uint64_t now,std::uint64_t then){return now>then?now-then:0;}
    static bool valid_source(ZoomPreviewSource const& source){
        return source.id&&absolute(source.zoom)&&absolute(source.coverage_min_zoom)&&
            source.coverage_min_zoom<=source.zoom&&source.identity.native_width>0&&
            source.identity.target_width>0&&source.identity.target_height>0&&
            source.completed_time_ms>=source.source_time_ms;
    }
    bool coherent(ZoomPreviewSource const& source)const {
        return requested&&valid_source(source)&&source.identity==wanted_identity;
    }
    ZoomPreviewSource const* current_quality_source()const {
        ZoomPreviewSource const* source=nullptr;
        if(coherent(wide)&&wide.zoom==wanted&&wide.source_time_ms>=last_input)source=&wide;
        if(coherent(latest)&&latest.zoom==wanted&&latest.source_time_ms>=last_input&&
            (!source||latest.source_time_ms>=source->source_time_ms))source=&latest;
        return source;
    }
public:
    static constexpr std::uint64_t idle_debounce_ms=60,refresh_age_ms=250;
    // Every accepted input has its own serial/time. Neither smoothing samples
    // nor successful presentations should call request() as an input event.
    bool request(double zoom,ZoomPreviewIdentity const& identity,std::uint64_t now_ms){
        if(!absolute(zoom)||identity.native_width<=0||identity.target_width<=0||identity.target_height<=0)return false;
        wanted=zoom;wanted_identity=identity;last_input=now_ms;++serial;requested=true;
        return true;
    }
    bool seed(ZoomPreviewSource const& source){
        if(!valid_source(source))return false;
        if(source.zoom==1.)wide=source;else latest=source;
        return true;
    }
    // Call before overwriting a retained GPU slot. The in-flight render is not
    // a readable completed source until the caller proves GPU completion and
    // calls complete(). Already presented pixels/picking remain authoritative.
    bool discard_source(std::uint64_t id){
        if(!id)return false;
        bool discarded=false;
        if(wide.id==id){wide={};discarded=true;}
        if(latest.id==id){latest={};discarded=true;}
        return discarded;
    }
    ZoomPreviewFrame preview(std::uint64_t now_ms)const {
        ZoomPreviewSource const* source=nullptr;
        if(coherent(wide)&&wanted>=wide.coverage_min_zoom)source=&wide;
        if(coherent(latest)&&wanted>=latest.coverage_min_zoom&&
            (!source||latest.source_time_ms>=source->source_time_ms))source=&latest;
        if(!source)return {};
        double relative=wanted/source->zoom;
        if(!std::isfinite(relative)||relative<=0)return {};
        return {true,*source,serial,wanted,relative,age(now_ms,source->source_time_ms)};
    }
    bool full_quality_current()const {
        return current_quality_source()!=nullptr;
    }
    // There is exactly one desired refinement, recomputed from newest input.
    // A 250 ms source age makes a refresh DUE; submitted GPU work can still
    // overrun this age. The caller measures actual displayed age/completion.
    ZoomPreviewRefinement desired(std::uint64_t now_ms)const {
        if(!requested)return {};
        ZoomPreviewRefinement job{};
        job.valid=true;job.request_serial=serial;job.source_time_ms=now_ms;job.identity=wanted_identity;
        if(age(now_ms,last_input)>=idle_debounce_ms){
            auto source=current_quality_source();
            if(source&&age(now_ms,source->source_time_ms)<refresh_age_ms)return {};
            job.zoom=wanted;return job;
        }
        if(coherent(wide)&&age(now_ms,wide.source_time_ms)<refresh_age_ms)return {};
        job.zoom=1.;job.refresh_wide=true;return job;
    }
    bool begin(std::uint64_t now_ms,ZoomPreviewRefinement& job){
        job={};if(active.valid)return false;
        job=desired(now_ms);if(!job.valid)return false;
        job.id=++next_job;active=job;return true;
    }
    // A superseded same-scene result may remain a preview source. It cannot
    // mark a different requested zoom/identity as current full quality.
    bool complete(ZoomPreviewRefinement const& job,ZoomPreviewSource source){
        if(!active.valid||job.id!=active.id)return false;
        auto completed=active;active={};
        if(!valid_source(source)||source.identity!=completed.identity||
            source.zoom!=completed.zoom||source.source_time_ms<completed.source_time_ms||
            !requested||source.identity!=wanted_identity)return false;
        source.request_serial=completed.request_serial;return seed(source);
    }
    bool fail(ZoomPreviewRefinement const& job){
        if(!active.valid||job.id!=active.id)return false;
        active={};return true;
    }
    bool present_success(ZoomPreviewFrame const& frame){
        if(!frame.valid||!requested||!valid_source(frame.source)||
            frame.source.identity!=wanted_identity||!absolute(frame.absolute_zoom)||
            frame.absolute_zoom<frame.source.coverage_min_zoom||
            !std::isfinite(frame.relative_zoom)||frame.relative_zoom<=0||
            frame.relative_zoom!=frame.absolute_zoom/frame.source.zoom)return false;
        displayed=frame.absolute_zoom;displayed_identity=frame.source.identity;presented=true;return true;
    }
    void present_failed(){} // Keep the last successfully displayed/picked transform.
    double requested_zoom()const{return wanted;}
    double displayed_zoom()const{return displayed;}
    ZoomPreviewIdentity const& requested_identity()const{return wanted_identity;}
    ZoomPreviewIdentity const& presented_identity()const{return displayed_identity;}
    bool has_presented()const{return presented;}
    std::uint64_t request_serial()const{return serial;}
    std::uint64_t last_input_ms()const{return last_input;}
    bool refining()const{return active.valid;}
    ZoomPreviewRefinement const& in_flight()const{return active;}
    ZoomPreviewSource const& wide_source()const{return wide;}
    ZoomPreviewSource const& latest_source()const{return latest;}
    unsigned source_count()const{return unsigned(wide.id!=0)+unsigned(latest.id!=0);}
};
}}
