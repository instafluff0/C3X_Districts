#pragma once
// Stateless replay of the production six-tile source-depth page selection.
// Native retains its incremental GPU atlas; replay owns only this frame's plan.
#include "../contracts/packet_v1.h"
#include <algorithm>
#include <cstring>
#include <set>
namespace labv2 {
struct ShadowBounds {std::array<float,3> low{{1e9f,1e9f,1e9f}},high{{-1e9f,-1e9f,-1e9f}};};
struct ShadowCaster {unsigned draw;std::array<float,3> offset;std::array<float,4> projected;};
struct ShadowPage {int x,y;unsigned slot;std::vector<unsigned> casters;};
struct ShadowPlan {
    std::array<float,12> basis{};
    std::vector<ShadowCaster> casters;
    std::vector<ShadowPage> pages;
    std::array<std::array<float,4>,64> table{};
    std::array<float,4> project(ShadowBounds const&bounds,std::array<float,3> offset={})const {
        std::array<float,4> result{{1e9f,1e9f,-1e9f,-1e9f}};
        for(unsigned mask=0;mask<8;mask++){
            float u=0,v=0;
            for(unsigned i=0;i<3;i++){
                float value=(((mask>>i)&1)?bounds.high[i]:bounds.low[i])+offset[i];
                u+=value*basis[i];v+=value*basis[4+i];
            }
            result[0]=std::min(result[0],u);result[1]=std::min(result[1],v);
            result[2]=std::max(result[2],u);result[3]=std::max(result[3],v);
        }
        return result;
    }
    explicit ShadowPlan(Packet&packet){
        if(!packet.shadow.enabled())return;
        auto const&w=packet.shadow;
        std::memcpy(basis.data(),packet.buffers.at(w.frame_buffer).data(),sizeof(basis));
        for(float value:basis)if(!std::isfinite(value))throw std::runtime_error("nonfinite shadow basis");
        for(unsigned row=0;row<3;row++){
            float length=0;for(unsigned c=0;c<3;c++)length+=basis[row*4+c]*basis[row*4+c];
            if(std::abs(length-1)>.001f)throw std::runtime_error("shadow basis must be normalized");
            for(unsigned prior=0;prior<row;prior++){
                float dot=0;for(unsigned c=0;c<3;c++)dot+=basis[row*4+c]*basis[prior*4+c];
                if(std::abs(dot)>.001f)throw std::runtime_error("shadow basis must be orthogonal");
            }
        }
        std::set<std::pair<int,int>> needed;
        for(unsigned index=0;index<packet.draws.size();index++){
            auto const&draw=packet.draws[index];if(!(draw.geometry_flags&3) || !draw.count)continue;
            auto const&data=packet.buffers.at(draw.vertex_buffer);
            auto component=[&](unsigned vertex,unsigned offset){
                float value;std::memcpy(&value,data.data()+std::size_t(vertex)*draw.stride+offset,4);
                if(!std::isfinite(value) || std::abs(value)>1e6f)throw std::runtime_error("invalid shadow geometry coordinate");
                return value;
            };
            ShadowBounds bounds;std::array<float,4> screen{{1e9f,1e9f,-1e9f,-1e9f}};
            unsigned world=draw.attributes.at(draw.world_attribute).offset;
            unsigned position=draw.attributes.at(0).offset;
            for(unsigned v=0;v<draw.count;v++){
                for(unsigned i=0;i<3;i++){
                    float value=component(v,world+4*i);
                    bounds.low[i]=std::min(bounds.low[i],value);bounds.high[i]=std::max(bounds.high[i],value);
                }
                float x=component(v,position),y=component(v,position+4);
                screen[0]=std::min(screen[0],std::floor(x)-2);screen[1]=std::min(screen[1],std::floor(y)-2);
                screen[2]=std::max(screen[2],std::ceil(x)+2);screen[3]=std::max(screen[3],std::ceil(y)+2);
            }
            if(draw.geometry_flags&2){
                if(draw.frame_buffer==UINT32_MAX || packet.buffers.at(draw.frame_buffer).size()<8)
                    throw std::runtime_error("shadow receiver needs the native screen translation");
                float translation[2];std::memcpy(translation,packet.buffers.at(draw.frame_buffer).data(),8);
                for(float value:translation)if(!std::isfinite(value))throw std::runtime_error("nonfinite receiver translation");
                bool visible=!(screen[2]+translation[0]<=packet.valid_rect[0] || screen[0]+translation[0]>=packet.valid_rect[2] ||
                    screen[3]+translation[1]<=packet.valid_rect[1] || screen[1]+translation[1]>=packet.valid_rect[3]);
                if(visible){auto p=project(bounds);
                    int x0=int(std::floor((p[0]-.018f)/6)),x1=int(std::floor((p[2]+.018f)/6));
                    int y0=int(std::floor((p[1]-.018f)/6)),y1=int(std::floor((p[3]+.018f)/6));
                    if(x1-x0>=32 || y1-y0>=32)throw std::runtime_error("shadow page budget exceeded");
                    for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++){
                        needed.emplace(x,y);if(needed.size()>32)throw std::runtime_error("shadow page budget exceeded");
                    }
                }
            }
            if(draw.geometry_flags&1){
                for(int wy=w.wrap_y?-1:0;wy<=(w.wrap_y?1:0);wy++)for(int wx=w.wrap_x?-1:0;wx<=(w.wrap_x?1:0);wx++){
                    std::array<float,3> offset{{float(wx*int(w.world_width)+wy*int(w.world_height))*.5f,
                                              float(wx*int(w.world_width)-wy*int(w.world_height))*.5f,0}};
                    casters.push_back({index,offset,project(bounds,offset)});
                }
            }
        }
        for(auto key:needed){
            ShadowPage page{key.first,key.second,unsigned(pages.size()),{}};
            for(unsigned i=0;i<casters.size();i++){
                auto p=casters[i].projected;
                if(p[2]<key.first*6 || p[0]>(key.first+1)*6 || p[3]<key.second*6 || p[1]>(key.second+1)*6)continue;
                page.casters.push_back(i);
            }
            unsigned slot=(unsigned(key.first)*73856093u^unsigned(key.second)*19349663u)&63u;
            while(table[slot][3]>.5f)slot=(slot+1)&63u;
            table[slot]={float(key.first),float(key.second),float(page.slot),1};pages.push_back(page);
        }
        std::memcpy(packet.buffers.at(w.table_buffer).data(),table.data(),sizeof(table));
    }
};
}
