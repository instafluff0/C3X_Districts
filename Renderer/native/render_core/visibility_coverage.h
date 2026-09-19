#pragma once
#include "../c3x_renderer_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Final-output visibility, separate from retained geometry. Screen anchors are
// native; canonical coordinates only look up copied neighboring visibility.
struct VisibilityCoverage {
    struct Tile {float x,y;unsigned cells,reserved;}; // Nine packed two-bit states.
    std::vector<Tile> tiles;
    int width=0,height=0,tile_width=0,tile_height=0;
    std::map<std::pair<std::int64_t,std::int64_t>,unsigned> states;
    int world_width=0,world_height=0;bool wrap_x=false,wrap_y=false;
    auto key(std::int64_t x,std::int64_t y)const {
        auto canonical=[](std::int64_t value,int extent,bool wraps){auto r=wraps&&extent>0?value%extent:value;return wraps&&r<0?r+extent:r;};
        return std::make_pair(canonical(x,world_width,wrap_x),canonical(y,world_height,wrap_y));
    }
    unsigned state(std::int64_t x,std::int64_t y)const{auto found=states.find(key(x,y));return found==states.end()?0:found->second;}
    static constexpr float feather=.18f,gray=.18f,fog_alpha=.5f;
    bool capture(c3x_renderer_frame_v1 const& frame){
        tiles.clear();states.clear();
        if(frame.tile_count>8192 || (frame.tile_count&&!frame.tiles) ||
           frame.target_width<=0 || frame.target_height<=0 || frame.target_width>8192 || frame.target_height>8192 ||
           frame.tile_width<=0 || frame.tile_height<=0 || frame.tile_width>4096 || frame.tile_height>4096)return false;
        width=frame.target_width;height=frame.target_height;tile_width=frame.tile_width;tile_height=frame.tile_height;
        world_width=frame.world_width_tiles;world_height=frame.world_height_tiles;
        wrap_x=frame.world_wrap_x!=0;wrap_y=frame.world_wrap_y!=0;
        std::map<std::pair<std::int64_t,std::int64_t>,unsigned> anchors;
        for(unsigned i=0;i<frame.tile_count;++i){auto const& t=frame.tiles[i];
            auto flags=t.tile_flags;
            if(!(flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_TOPOLOGY_HALO)))continue;
            if(!(flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN)){
                if(flags&C3X_RENDERER_TILE_RENDER)return false;
                continue; // Unknown neighboring coverage stays black.
            }
            if((flags&C3X_RENDERER_TILE_VISIBLE)&&!(flags&C3X_RENDERER_TILE_EXPLORED))return false;
            unsigned state=(flags&C3X_RENDERER_TILE_VISIBLE)?2:(flags&C3X_RENDERER_TILE_EXPLORED)?1:0;
            auto prior=states.emplace(key(t.tile_x,t.tile_y),state);
            if(!prior.second&&prior.first->second!=state)return false;
        }
        for(unsigned i=0;i<frame.tile_count;++i){auto const& t=frame.tiles[i];
            if(!(t.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            if(std::int64_t(t.anchor_x)+tile_width<=0 || t.anchor_x>=width ||
               std::int64_t(t.anchor_y)+tile_height<=0 || t.anchor_y>=height)continue;
            unsigned cells=0;
            for(int v=-1;v<=1;++v)for(int u=-1;u<=1;++u){
                auto found=states.find(key(std::int64_t(t.tile_x)+u-v,std::int64_t(t.tile_y)+u+v));
                unsigned state=found==states.end()?0:found->second;
                cells|=state<<(2*((v+1)*3+u+1));
            }
            auto prior=anchors.emplace(std::make_pair(t.anchor_x,t.anchor_y),cells);
            if(!prior.second){if(prior.first->second!=cells)return false;continue;}
            // All nine cells currently visible: exact no-op, no GPU allocation.
            if(cells==0x2aaaau)continue;
            tiles.push_back({float(t.anchor_x),float(t.anchor_y),cells,0});
        }
        return true;
    }
    static float edge(float coordinate){
        float d=std::min(coordinate,1-coordinate),t=std::clamp(d/feather,0.f,1.f);
        return .5f*(1-t*t*(3-2*t));
    }
    static float coverage(unsigned cells,float u,float v,unsigned threshold){
        int x=u<.5f?-1:1,y=v<.5f?-1:1;
        auto value=[&](int dx,int dy){return float(((cells>>(2*((dy+1)*3+dx+1)))&3u)>=threshold);};
        float wx=edge(u),wy=edge(v);
        return (value(0,0)*(1-wx)+value(x,0)*wx)*(1-wy)+(value(0,y)*(1-wx)+value(x,y)*wx)*wy;
    }
    // Explicit CPU-output compatibility. Never darken retained source pixels.
    void apply(std::uint32_t const* source,std::vector<std::uint32_t>& output)const{
        output.assign(source,source+std::size_t(width)*height);
        for(auto tile:tiles){
            int left=std::max(0,int(tile.x)),top=std::max(0,int(tile.y));
            int right=std::min(width,int(tile.x)+tile_width),bottom=std::min(height,int(tile.y)+tile_height);
            for(int y=top;y<bottom;++y)for(int x=left;x<right;++x){
                float dx=(x+.5f-tile.x)/tile_width*2-1,dy=(y+.5f-tile.y)/tile_height*2;
                float u=(dx+dy)*.5f,v=(dy-dx)*.5f;
                if(u<0 || v<0 || u>=1 || v>=1)continue;
                float explored=coverage(tile.cells,u,v,1),visible=coverage(tile.cells,u,v,2);
                float fog=(explored-visible)*fog_alpha,scale=explored-fog;
                auto index=std::size_t(y)*width+x;unsigned original=source[index],pixel=original&0xff000000u;
                for(unsigned shift:{0u,8u,16u})pixel|=unsigned(std::clamp(std::nearbyint(((original>>shift)&255u)*scale+gray*255*fog),0.f,255.f))<<shift;
                output[index]=pixel;
            }
        }
    }
};
}}
