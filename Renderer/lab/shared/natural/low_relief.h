#pragma once
// Optional generic, authored height fields for gently rolling base terrain.
// Source selection belongs to the offline compiler. Runtime has no asset-family
// names, random state, camera dependence or saved-game data.
#include "../../../native/render_core/terrain_query.h"
#include <cstring>
namespace c3x_renderer { namespace fidelity {
struct LowRelief {
    struct Field {unsigned width=0,height=0;float amplitude=0,span=0;std::vector<unsigned char> pixels;};
    std::array<Field,2> fields; // grassland, plains; optional together
    bool load(std::vector<unsigned char> const& bytes){
        fields={};if(bytes.empty())return true;
        if(bytes.size()<8||std::memcmp(bytes.data(),"C3XLOW1\0",8))return false;
        std::size_t pos=8;
        for(auto& f:fields){
            if(bytes.size()-pos<16)return false;
            std::memcpy(&f.width,bytes.data()+pos,4);std::memcpy(&f.height,bytes.data()+pos+4,4);
            std::memcpy(&f.amplitude,bytes.data()+pos+8,4);std::memcpy(&f.span,bytes.data()+pos+12,4);pos+=16;
            if(f.width<2||f.height<2||f.width>2048||f.height>2048||
               !std::isfinite(f.amplitude)||f.amplitude<0||f.amplitude>96||
               !std::isfinite(f.span)||f.span<8||f.span>512)return false;
            std::size_t count=std::size_t(f.width)*f.height;if(count>bytes.size()-pos)return false;
            f.pixels.assign(bytes.begin()+pos,bytes.begin()+pos+count);pos+=count;
        }
        return pos==bytes.size();
    }
    float sample(unsigned biome,float x,float y,render_core::World world) const {
        auto const& f=fields[biome];if(f.pixels.empty())return 0;
        // Native map X/Y axes give integral repeat counts across either wrap.
        // A small map still uses one full field; larger maps repeat continuously.
        float fx=world.wrap_x?std::max(1.f,std::round(world.width/f.span))/std::max(1,world.width):1/f.span;
        float fy=world.wrap_y?std::max(1.f,std::round(world.height/f.span))/std::max(1,world.height):1/f.span;
        float u=(x+y-1)*fx,v=(x-y)*fy;u-=std::floor(u);v-=std::floor(v);
        float px=u*f.width,py=v*f.height;unsigned ix=unsigned(px),iy=unsigned(py);
        float tx=px-ix,ty=py-iy;
        auto tap=[&](unsigned a,unsigned b){return f.pixels[(b%f.height)*f.width+a%f.width]/255.f;};
        return f.amplitude*((tap(ix,iy)*(1-tx)+tap(ix+1,iy)*tx)*(1-ty)+
            (tap(ix,iy+1)*(1-tx)+tap(ix+1,iy+1)*tx)*ty);
    }
};
}}
