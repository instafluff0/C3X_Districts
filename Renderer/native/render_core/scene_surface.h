#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace c3x_renderer { namespace render_core {
inline bool scene_surface_extent(int width,int height) {
    return width>=8 && height>=8 && width<=2240 && height<=1192;
}

template<class Rect> struct SceneSpan {Rect rect;int x,y;};

// A captured screen coordinate plus span translation addresses a fixed world
// sample. Scrolling changes scissors/output mapping, never moves resident pixels.
template<class Rect> std::vector<SceneSpan<Rect>> scene_spans(
        int width,int height,std::int64_t world_x,std::int64_t world_y) {
    if(width<=0 || height<=0)return {};
    auto phase=[](std::int64_t value,int extent){int r=int(value%extent);return r>0?extent-r:-r;};
    int px=phase(world_x,width),py=phase(world_y,height);
    std::vector<SceneSpan<Rect>> result;
    for(int y:{py,py-height})for(int x:{px,px-width}){
        Rect r={std::max<int>(0,x),std::max<int>(0,y),std::min<int>(width,x+width),std::min<int>(height,y+height)};
        if(r.left<r.right && r.top<r.bottom)result.push_back({r,x,y});
    }
    return result;
}

// Bounded union aligned with finishing workgroups. It owns scissors only, not
// render targets or scene construction. At the 2248x1200 limit the grid is 42 KiB
// and even a checkerboard requires less than 340 KiB of output rectangles.
template<class Rect> std::vector<Rect> scene_damage_union(
        int width,int height,std::vector<Rect> const& inputs) {
    if(width<=0 || height<=0 || width>2248 || height>1200)return {};
    constexpr int cell=8;
    int columns=(width+cell-1)/cell,rows=(height+cell-1)/cell;
    std::vector<unsigned char> occupied(std::size_t(columns)*rows);
    for(auto r:inputs){
        int l=std::clamp<int>(r.left,0,width),t=std::clamp<int>(r.top,0,height);
        int right=std::clamp<int>(r.right,l,width),bottom=std::clamp<int>(r.bottom,t,height);
        if(l==right || t==bottom)continue;
        for(int y=t/cell;y<=(bottom-1)/cell;++y)
            std::fill(occupied.begin()+y*columns+l/cell,occupied.begin()+y*columns+(right-1)/cell+1,static_cast<unsigned char>(1));
    }
    std::vector<Rect> result;
    std::vector<std::size_t> previous(columns,std::size_t(-1)),current(columns,std::size_t(-1));
    for(int y=0;y<rows;++y){
        std::fill(current.begin(),current.end(),std::size_t(-1));
        for(int x=0;x<columns;){
            if(!occupied[y*columns+x]){++x;continue;}
            int left=x++;while(x<columns && occupied[y*columns+x])++x;
            int right=std::min<int>(width,x*cell),bottom=std::min<int>(height,(y+1)*cell);
            auto old=previous[left];
            if(old!=std::size_t(-1) && result[old].right==right){result[old].bottom=bottom;current[left]=old;}
            else{current[left]=result.size();result.push_back({left*cell,y*cell,right,bottom});}
        }
        previous.swap(current);
    }
    return result;
}

// The HDR lens has radius eight at 2x resolution: four native pixels.
// Changed samples invalidate neighboring outputs, including across the physical
// seam. Retained output uses the same workgroup origin/arithmetic as a full pass.
template<class Rect> std::vector<Rect> scene_filter_damage(
        int width,int height,std::vector<Rect> const& inputs,int radius) {
    if(width<=0 || height<=0 || width>2248 || height>1200 || radius<0 || radius>8)return {};
    std::vector<Rect> support;
    for(auto r:inputs){
        if(r.left>=r.right || r.top>=r.bottom)continue;
        for(int dy:{-height,0,height})for(int dx:{-width,0,width}){
            Rect s={std::max<int>(0,int(r.left)-radius+dx),std::max<int>(0,int(r.top)-radius+dy),
                    std::min<int>(width,int(r.right)+radius+dx),std::min<int>(height,int(r.bottom)+radius+dy)};
            if(s.left<s.right && s.top<s.bottom)support.push_back(s);
        }
    }
    return scene_damage_union(width,height,support);
}
} }
