#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace c3x_renderer::tactical {
// Copied draw semantics only. No unit, route finder, native surface or game
// pointer survives capture. Coordinates are authoritative projected pixels.
struct Primitive {
    std::array<float,4> bounds{},shape{},color{},style{};
};
struct Input {
    std::vector<Primitive> primitives;
    bool animated=false;
    void append(Primitive p){
        if(primitives.size()>=16384)throw std::runtime_error("tactical primitive budget");
        for(auto a:{p.bounds,p.shape,p.color,p.style})for(float x:a)
            if(!std::isfinite(x)||std::abs(x)>131072)throw std::runtime_error("invalid tactical input");
        primitives.push_back(p);
    }
    void line(float x0,float y0,float x1,float y1,float width=2.2f,bool grid=false){
        if(!std::isfinite(width)||width<=0||width>16)throw std::runtime_error("tactical line width");
        float pad=grid?2.f:5.f;
        append({{std::min(x0,x1)-pad,std::min(y0,y1)-pad,std::max(x0,x1)+pad,std::max(y0,y1)+pad},
            {x0,y0,x1,y1},grid?std::array<float,4>{.64f,.66f,.68f,.52f}:std::array<float,4>{.94f,.17f,.18f,.97f},
            {0,width,grid?0.f:1.f,0}});
    }
    void ring(float x,float y,float tile_width,bool moving){
        if(!std::isfinite(tile_width)||tile_width<64||tile_width>192)throw std::runtime_error("tactical projection");
        float rx=tile_width*.47f,ry=tile_width*.235f;
        append({{x-rx-5,y-ry-5,x+rx+5,y+ry+5},{x,y,rx,ry},{.97f,.98f,1.f,.94f},{1,1.8f,moving?1.f:0.f,0}});
        animated|=moving;
    }
    void native_line(float x0,float y0,float x1,float y1,int width,int dash,unsigned argb){
        // Native integer anchors, flat caps and caller-supplied opacity. Dash 1
        // is GL's factor-five 0xAAAA stipple; 2 is GDI+'s width-scaled Dash.
        if(width<1||width>128||dash<0||dash>2)throw std::runtime_error("native line style");
        if((x0==x1&&y0==y1)||!(argb>>24))return;
        x0+=.5f;y0+=.5f;x1+=.5f;y1+=.5f;float pad=float(width)*.5f+1;
        append({{std::min(x0,x1)-pad,std::min(y0,y1)-pad,std::max(x0,x1)+pad,std::max(y0,y1)+pad},
            {x0,y0,x1,y1},{float((argb>>16)&255)/255.f,float((argb>>8)&255)/255.f,float(argb&255)/255.f,float(argb>>24)/255.f},
            {3,float(width),float(dash),0}});
    }
    void label(float x,float y,std::string const& text,float height){
        // Native supplies the complete turn string, including action suffixes.
        if(!std::isfinite(height)||height<8||height>128)throw std::runtime_error("tactical label size");
        if(text.size()>32)throw std::runtime_error("tactical label budget");
        float width=height*.95f,left=x-width*float(text.size())*.5f;
        for(unsigned i=0;i<text.size();++i){unsigned glyph=static_cast<unsigned char>(text[i]);
            if(glyph<32||glyph>126)throw std::runtime_error("unsupported tactical label glyph");
            float a=left+float(i)*width;
            append({{a-2,y-height*.5f-2,a+width+2,y+height*.5f+2},{a,y-height*.5f,width,height},
                {1.f,.96f,.92f,1.f},{2,float(glyph-32),0,0}});
        }
    }
    std::array<int,4> extent(std::array<int,4> clip)const{
        float l=float(clip[2]),t=float(clip[3]),r=float(clip[0]),b=float(clip[1]);
        for(auto const& p:primitives){l=std::min(l,p.bounds[0]);t=std::min(t,p.bounds[1]);r=std::max(r,p.bounds[2]);b=std::max(b,p.bounds[3]);}
        return {std::max(clip[0],int(std::floor(l))),std::max(clip[1],int(std::floor(t))),
            std::min(clip[2],int(std::ceil(r))),std::min(clip[3],int(std::ceil(b)))};
    }
};
}
