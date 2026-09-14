#pragma once
#include "scene_surface.h"
#include <limits>

namespace c3x_renderer { namespace render_core {
// Coverage belongs to a persistent static color/depth surface. Cells bound
// bookkeeping and scissors only; they never own scenes or compiled geometry.
template<class Rect> struct SceneGuard {
    int width=0,height=0,columns=0,rows=0;
    std::vector<unsigned char> dirty;
    std::size_t pending=0;
    void reset(){width=height=columns=rows=0;dirty.clear();pending=0;}
    bool configure(int w,int h){
        if(w<8 || h<8 || w>4096 || h>2048)return false;
        if(width==w && height==h)return true;
        width=w;height=h;columns=(w+7)/8;rows=(h+7)/8;
        dirty.assign(std::size_t(columns)*rows,1);pending=dirty.size();return true;
    }
    void invalidate(std::vector<Rect> const& rectangles){
        for(auto r:rectangles){
            int l=std::clamp<int>(r.left,0,width),t=std::clamp<int>(r.top,0,height);
            int right=std::clamp<int>(r.right,l,width),bottom=std::clamp<int>(r.bottom,t,height);
            if(l==right || t==bottom)continue;
            for(int y=t/8;y<=(bottom-1)/8;++y)for(int x=l/8;x<=(right-1)/8;++x)
                if(!dirty[y*columns+x]){dirty[y*columns+x]=1;++pending;}
        }
    }
    void invalidate_all(){std::fill(dirty.begin(),dirty.end(),static_cast<unsigned char>(1));pending=dirty.size();}
    std::vector<Rect> select(std::vector<Rect> const& regions,
            std::size_t pixel_limit=std::numeric_limits<std::size_t>::max()) const {
        std::vector<Rect> result;std::size_t pixels=0;bool full=false;
        for(int y=0;y<rows && !full;++y)for(int x=0;x<columns;++x){
            if(!dirty[y*columns+x])continue;
            Rect r={x*8,y*8,std::min(width,x*8+8),std::min(height,y*8+8)};
            bool selected=regions.empty();for(auto q:regions)
                selected=selected || (r.left<q.right && r.right>q.left && r.top<q.bottom && r.bottom>q.top);
            if(!selected)continue;
            auto area=std::size_t(r.right-r.left)*(r.bottom-r.top);
            if(pixels+area>pixel_limit){full=true;break;}
            pixels+=area;
            if(!result.empty() && result.back().top==r.top && result.back().right==r.left)result.back().right=r.right;
            else result.push_back(r);
        }
        // Join vertical runs without overpainting clean cells.
        std::vector<Rect> joined;
        for(auto r:result){bool found=false;for(auto it=joined.rbegin();it!=joined.rend();++it){
            if(it->bottom==r.top && it->left==r.left && it->right==r.right){it->bottom=r.bottom;found=true;break;}
        }if(!found)joined.push_back(r);}
        return joined;
    }
    // Commit only after successful submission. A failed producer invalidates
    // the whole target; cancellation leaves unsubmitted cells dirty.
    void commit(std::vector<Rect> const& rectangles){
        for(auto r:rectangles)for(int y=r.top/8;y<(r.bottom+7)/8;++y)for(int x=r.left/8;x<(r.right+7)/8;++x)
            if(dirty[y*columns+x]){dirty[y*columns+x]=0;--pending;}
    }
};

template<class Rect> std::vector<Rect> scene_physical(int w,int h,std::int64_t x,std::int64_t y,
        std::vector<Rect> const& logical){
    std::vector<Rect> result;
    for(auto span:scene_spans<Rect>(w,h,x,y))for(auto r:logical){
        r={std::max(span.rect.left,r.left+span.x),std::max(span.rect.top,r.top+span.y),
           std::min(span.rect.right,r.right+span.x),std::min(span.rect.bottom,r.bottom+span.y)};
        if(r.left<r.right && r.top<r.bottom)result.push_back(r);
    }return result;
}

// Copy identical MSAA samples between two circular surfaces of different size.
// Both retain the same world-relative pixel lattice; padding adds no camera.
template<class Rect> std::vector<SceneSpan<Rect>> scene_guard_transfers(int w,int h,int pad,
        std::int64_t x,std::int64_t y,std::vector<Rect> const& damage){
    std::vector<SceneSpan<Rect>> result;
    for(auto source:scene_spans<Rect>(w+pad*2,h+pad*2,x,y))
    for(auto target:scene_spans<Rect>(w,h,x,y)){
        int dx=target.x-source.x-pad,dy=target.y-source.y-pad;
        for(auto r:damage){
            r={std::max({r.left,target.rect.left,source.rect.left+dx}),std::max({r.top,target.rect.top,source.rect.top+dy}),
               std::min({r.right,target.rect.right,source.rect.right+dx}),std::min({r.bottom,target.rect.bottom,source.rect.bottom+dy})};
            if(r.left<r.right && r.top<r.bottom)result.push_back({r,dx,dy});
        }
    }return result;
}
}}
