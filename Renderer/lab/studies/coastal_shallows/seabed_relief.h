#pragma once
// Isolated Lab proposal for broad, world-continuous coast-bed geometry.
// The existing source shallows height channel is nearly flat; these shapes are
// inferred C3X terrain art, not recovered source-game bathymetry.
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace c3x_renderer { namespace coastal_lab {
inline float smooth(float x) {
    x=std::clamp(x,0.f,1.f);
    return x*x*(3.f-2.f*x);
}
inline std::uint32_t hash(std::uint32_t x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15;
    x*=0x846ca68bu; return x^(x>>16);
}
inline int mod(int x,int size) { return (x%size+size)%size; }
inline float noise(float x,float y,float frequency,int width,int height,
                   bool wrap_x,bool wrap_y,std::uint32_t salt) {
    int period=wrap_x?width:wrap_y?height:0;
    if(wrap_x && wrap_y){int a=width,b=height;while(b){int r=a%b;a=b;b=r;}period=a;}
    if(period>0)frequency=std::max(1.f,std::round(frequency*period*.5f))/(period*.5f);
    float gx=x*frequency,gy=y*frequency;
    int ix=int(std::floor(gx)),iy=int(std::floor(gy));
    float u=smooth(gx-ix),v=smooth(gy-iy);
    auto value=[&](int i,int j) {
        int rx=i+j,ry=i-j;
        if(wrap_x)rx=mod(rx,int(std::round(width*frequency)));
        if(wrap_y)ry=mod(ry,int(std::round(height*frequency)));
        return float(hash(std::uint32_t(rx)*73856093u^
                          std::uint32_t(ry)*19349663u^salt))/4294967295.f;
    };
    return (1-v)*((1-u)*value(ix,iy)+u*value(ix+1,iy))+
           v*((1-u)*value(ix,iy+1)+u*value(ix+1,iy+1));
}
inline float base_height(float shore_distance,float depth,float water_family) {
    if(shore_distance>=0 || depth<=0)return 0;
    float water=smooth(-shore_distance/.14f);
    float coast=1-smooth((water_family-.34f)/.29f);
    return std::min(-2.5f,-.75f-depth*23.f)*water*coast;
}
inline float height(float x,float y,float shore_distance,float depth,
                    float water_family,int width,int rows,bool wrap_x,bool wrap_y) {
    if(shore_distance>=0 || depth<=0)return 0;
    float water=smooth(-shore_distance/.14f);
    float coast=1-smooth((water_family-.34f)/.29f);
    float shelf=smooth((depth-.035f)/.11f)*(1-smooth((depth-.37f)/.11f));
    float warp_x=(noise(x,y,.38f,width,rows,wrap_x,wrap_y,2819)-.5f)*.44f;
    float warp_y=(noise(x,y,.38f,width,rows,wrap_x,wrap_y,9109)-.5f)*.44f;
    float broad=noise(x+warp_x,y+warp_y,.65f,width,rows,wrap_x,wrap_y,1597)-.5f;
    float middle=noise(x,y,1.45f,width,rows,wrap_x,wrap_y,6547)-.5f;
    // Broad basins and lips must survive the 128-unit shared-world projection
    // and the translucent water pass at the 256-pixel review zoom. The source
    // height map cannot supply this form; these amplitudes are Lab inference.
    float bed=std::min(-2.5f,-.75f-depth*23.f+shelf*(broad*38.f+middle*7.f));
    return bed*water*coast;
}
}}
