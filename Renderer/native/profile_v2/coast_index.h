#pragma once
#include "terrain_query.h"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace c3x_renderer { namespace profile_v2 {
// Sparse fixed-domain quadtree. The domain depends on world dimensions, never
// the viewport. Empty-subtree certificates detect a newly created closer coast;
// observing a root/global revision would invalidate every shaded tile instead.
class CoastIndex {
    struct Node {
        std::uint64_t revision=0;
        std::vector<CoastSegment> segments;
    };
    std::map<std::uint64_t,Node> nodes;
    int origin_x,origin_y,side;
    std::uint64_t fold(std::uint64_t h,std::uint64_t value) const {
        for(int i=0;i<8;i++) { h=(h^(value&255))*1099511628211ull; value>>=8; }
        return h;
    }
    std::uint64_t segment_revision(std::vector<CoastSegment> const& segments) const {
        if(segments.empty()) return 0;
        std::uint64_t h=14695981039346656037ull;
        for(auto const& e:segments) for(double value:{e.a.x,e.a.y,e.b.x,e.b.y,e.rocky}) {
            std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits)); h=fold(h,bits);
        }
        return h ? h : 1;
    }
    void set(std::uint64_t id,int x,int y,int size,int c,int r,std::vector<CoastSegment> const& segments) {
        if(size==1) {
            auto h=segment_revision(segments);
            if(h) nodes[id]={h,segments}; else nodes.erase(id);
            return;
        }
        int half=size/2;
        int child=(c>=x+half ? 1 : 0)+(r>=y+half ? 2 : 0);
        set(id*4+child,x+(child&1)*half,y+(child>>1)*half,half,c,r,segments);
        std::uint64_t h=14695981039346656037ull; bool any=false;
        for(int i=0;i<4;i++) { auto value=revision(id*4+i); any=any || value!=0; h=fold(h,value); }
        if(any) nodes[id].revision=h ? h : 1; else nodes.erase(id);
    }
    double box_distance_squared(Point p,int x,int y,int size) const {
        double dx=std::max({double(x)-p.x,0.,p.x-(x+size)});
        double dy=std::max({double(y)-p.y,0.,p.y-(y+size)});
        return dx*dx+dy*dy;
    }
public:
    struct Nearest {
        double squared=std::numeric_limits<double>::infinity();
        Point foot;
        std::uint64_t cell=0;
    };
    CoastIndex(int x,int y,int size):origin_x(x),origin_y(y),side(size) {
        if(size<=0 || (size&(size-1)) || size>65536)
            throw std::invalid_argument("coast index requires a bounded power-of-two domain");
    }
    void clear() { nodes.clear(); }
    std::uint64_t revision(std::uint64_t id) const {
        auto found=nodes.find(id); return found==nodes.end() ? 0 : found->second.revision;
    }
    void set_cell(int c,int r,std::vector<CoastSegment> const& segments) {
        if(c<origin_x || r<origin_y || c>=origin_x+side || r>=origin_y+side)
            throw std::out_of_range("coast cell outside authoritative world domain");
        for(auto const& e:segments) for(Point p:{e.a,e.b})
            if(!std::isfinite(p.x) || !std::isfinite(p.y) || p.x<c || p.y<r || p.x>c+1 || p.y>r+1)
                throw std::invalid_argument("coast segment leaves its contour cell");
        set(1,origin_x,origin_y,side,c,r,segments);
    }
    template<class Observe>
    Nearest nearest(Point p,Observe observe,double limit=std::numeric_limits<double>::infinity()) const {
        Nearest result; result.foot=p; result.squared=limit;
        auto visit=[&](auto&& self,std::uint64_t id,int x,int y,int size)->void {
            if(box_distance_squared(p,x,y,size)>result.squared) return;
            auto found=nodes.find(id);
            if(found==nodes.end()) { observe(id,0); return; }
            if(size==1) {
                observe(id,found->second.revision);
                for(auto const& segment:found->second.segments) {
                    Point ab=segment.b-segment.a;
                    double t=std::clamp(dot(p-segment.a,ab)/std::max(1e-15,dot(ab,ab)),0.,1.);
                    Point q=segment.a+ab*t; double d=dot(p-q,p-q);
                    if(d<result.squared || (d==result.squared && (result.cell==0 || id<result.cell)))
                        result={d,q,id};
                }
                return;
            }
            int half=size/2;
            std::array<std::pair<double,int>,4> order;
            for(int child=0;child<4;child++) order[child]={box_distance_squared(
                p,x+(child&1)*half,y+(child>>1)*half,half),child};
            std::sort(order.begin(),order.end());
            for(auto const& item:order) {
                int child=item.second;
                self(self,id*4+child,x+(child&1)*half,y+(child>>1)*half,half);
            }
        };
        visit(visit,1,origin_x,origin_y,side);
        return result;
    }
    std::size_t estimated_bytes() const {
        std::size_t bytes=nodes.size()*(sizeof(Node)+sizeof(std::uint64_t)+4*sizeof(void*));
        for(auto const& pair:nodes) bytes+=pair.second.segments.capacity()*sizeof(CoastSegment);
        return bytes;
    }
};
} }
