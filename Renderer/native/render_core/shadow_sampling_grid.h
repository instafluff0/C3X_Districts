#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace c3x_renderer { namespace render_core {
// Sampling identity and retained coverage are independent.
// A page is always rasterized with its own canonical integer page coordinate;
// its physical array slice never enters the light projection.
struct ShadowSamplingGrid {
    static constexpr unsigned page_texels=1024,quality_texels=4096,max_pages=25;
    static constexpr float guard=4.f;
    static constexpr std::size_t texture_bytes=std::size_t(max_pages)*page_texels*page_texels*4;
    std::array<float,2> quality_span{};
    std::array<int,2> low{};
    std::array<unsigned,2> count{};
    bool valid=false;

    float page_span(unsigned axis)const{return quality_span[axis]/4.f;}
    float pitch(unsigned axis)const{return quality_span[axis]/float(quality_texels);}
    float inverse_pitch(unsigned axis)const{return float(quality_texels)/quality_span[axis];}
    unsigned pages()const{return count[0]*count[1];}
    bool same_sampling(ShadowSamplingGrid const& other)const {
        return valid && other.valid && quality_span==other.quality_span;
    }
    bool covers(ShadowSamplingGrid const& query)const {
        if(!same_sampling(query))return false;
        for(unsigned axis=0;axis<2;++axis)
            if(query.low[axis]<low[axis] ||
               query.low[axis]+int(query.count[axis])>low[axis]+int(count[axis]))return false;
        return true;
    }
    std::array<int,2> page(unsigned slot)const {
        return {low[0]+int(slot%count[0]),low[1]+int(slot/count[0])};
    }
    int slot(int x,int y)const {
        auto dx=x-low[0],dy=y-low[1];
        return valid && dx>=0 && dy>=0 && dx<int(count[0]) && dy<int(count[1])?
            dy*int(count[0])+dx:-1;
    }
    std::array<float,4> coverage()const {
        return {float(low[0])*page_span(0),float(low[1])*page_span(1),
            float(count[0])*page_span(0),float(count[1])*page_span(1)};
    }
    std::array<float,4> page_box(unsigned slot)const {
        auto p=page(slot);
        return {float(p[0])*page_span(0),float(p[1])*page_span(1),page_span(0),page_span(1)};
    }
    bool configure(float const* bounds,std::array<float,12> const& basis) {
        valid=false;
        for(unsigned axis=0;axis<2;++axis){
            double minimum=bounds[axis],maximum=bounds[axis+2];
            if(!std::isfinite(minimum) || !std::isfinite(maximum) || maximum<minimum)return false;
            // Mathematically the old cold span adds an origin-snap remainder
            // in [0,2). Float rounding can erase a boundary remainder; cap by
            // the actual old expression as well, never by previous coverage.
            double extent=std::ceil((maximum-minimum+2*guard)/2)*2;
            if(!std::isfinite(extent) || extent<2*guard || extent>16384)return false;
            float old_origin=std::floor((bounds[axis]-guard)/2.f)*2.f;
            float old_span=std::ceil((bounds[axis+2]+guard-old_origin)/2.f)*2.f;
            quality_span[axis]=std::min(float(extent),old_span);
            if(!(quality_span[axis]>=2*guard))return false;
            // Global half-cell centers must remain representable in shader
            // float arithmetic, including the three-tap PCF halo.
            if(std::max(std::abs(minimum-guard),std::abs(maximum+guard))*inverse_pitch(axis)>4194300.)return false;
        }
        // A normalized normal can move each light-plane axis by at most this
        // amount. Keep the water bias and full three-by-three PCF unchanged.
        double normal_bias=std::max(6./1024.,1.5*double(std::max(pitch(0),pitch(1))));
        for(unsigned axis=0;axis<2;++axis){
            double length=0;
            for(unsigned component=0;component<3;++component){
                double value=basis[axis*4+component];
                if(!std::isfinite(value))return false;
                length+=value*value;
            }
            double halo=std::sqrt(length)*normal_bias+1.5*double(pitch(axis));
            // The cap above can remove a few float ULPs of unused padding.
            // Its half-extent stays independent of the physical page origin.
            double minimum=bounds[axis],maximum=bounds[axis+2];
            double padding=std::min(double(guard),(double(quality_span[axis])-(maximum-minimum))*.5);
            if(!std::isfinite(halo) || !(halo<padding))return false;
            double span=page_span(axis);
            double first=std::floor((minimum-padding)/span);
            double last=std::ceil((maximum+padding)/span)-1;
            if(last<first || first<std::numeric_limits<int>::min() || last>std::numeric_limits<int>::max())return false;
            auto size=last-first+1;
            if(size<1 || size>5)return false;
            low[axis]=int(first);count[axis]=unsigned(size);
        }
        valid=pages()>0 && pages()<=max_pages;
        return valid;
    }
};
} }
