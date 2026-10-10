#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include "scene_projection.h"

namespace c3x_renderer {
// Latest requested map zoom. The retained static layer prepares its next
// full-quality raster at this destination while the displayed zoom animates.
inline std::atomic<float>& zoom_destination_hint(){static std::atomic<float> value{1.f};return value;}
// A wheel request reaches the renderer twice: at once through the shared zoom
// hint, and later in Civ III's ordered native stream, which waited 150-250 ms
// behind queued interface work on busy maps (review, section 42). Requests
// carry the bridge's sequence; only a newer one retargets, so a late ordered
// copy never undoes a newer hint. Sequence zero (recorded inputs) always applies.
struct ZoomRequests {
    unsigned applied=0;
    bool accept(unsigned sequence){
        if(!sequence)return true;
        if(sequence<=applied)return false;
        applied=sequence;return true;
    }
};
// Renderer-thread view state. Retargeting carries both position and velocity;
// sampling depends on elapsed time, never on the number of frames submitted.
// Native image versions and the gameplay camera are deliberately absent here.
class ZoomTransition {
    double position=1.,velocity=0.,destination=1.,time=0.;
    bool initialized=false;
    double presented=1.;
public:
    static constexpr double minimum=SceneProjection::minimum,maximum=SceneProjection::maximum;
    double sample(long long ticks,long long frequency){
        if(frequency<=0)throw std::invalid_argument("zoom clock frequency");
        double now=double(ticks)/double(frequency);
        if(!initialized){time=now;initialized=true;return position;}
        if(now<=time)return position;
        double elapsed=now-time;time=now;
        // The analytical critically damped solution stays stable across a
        // delayed frame. It reaches 99% of a normal step in about 166 ms.
        constexpr double rate=40.;
        double offset=position-destination,coefficient=velocity+rate*offset;
        double decay=std::exp(-rate*elapsed);
        position=destination+(offset+coefficient*elapsed)*decay;
        velocity=(velocity-rate*coefficient*elapsed)*decay;
        if(position<minimum||position>maximum){
            position=std::clamp(position,minimum,maximum);velocity=0.;
        }
        if(std::abs(position-destination)<1.e-7&&std::abs(velocity)<1.e-6){position=destination;velocity=0.;}
        return position;
    }
    void target(double scale,long long ticks,long long frequency){
        if(!std::isfinite(scale)||scale<minimum||scale>maximum)
            throw std::invalid_argument("zoom target outside supported view");
        sample(ticks,frequency);destination=scale;
        zoom_destination_hint().store(float(scale),std::memory_order_relaxed);
    }
    // A request that arrived since the last sample: the next frame's sample
    // integrates the whole interval toward it, so that frame already shows
    // motion. target() samples first, which left the first frame unmoved
    // (review, section 42).
    void retarget(double scale){
        if(!std::isfinite(scale)||scale<minimum||scale>maximum)
            throw std::invalid_argument("zoom target outside supported view");
        destination=scale;zoom_destination_hint().store(float(scale),std::memory_order_relaxed);
    }
    void reset(double scale=1.){
        if(!std::isfinite(scale)||scale<minimum||scale>maximum)
            throw std::invalid_argument("zoom reset outside supported view");
        position=destination=presented=scale;velocity=time=0.;initialized=false;
        zoom_destination_hint().store(float(scale),std::memory_order_relaxed);
    }
    double current()const{return position;}
    double target()const{return destination;}
    bool moving()const{return position!=destination||velocity!=0.;}
    // Call only after successful presentation. Picking must never use an
    // unpresented sample or the requested endpoint of an active transition.
    void did_present(double scale){presented=scale;}
    double last_presented()const{return presented;}
    double project(double value,double center)const{return center+(value-center)*presented;}
    double unproject(double value,double center)const{return center+(value-center)/presented;}
};
}
