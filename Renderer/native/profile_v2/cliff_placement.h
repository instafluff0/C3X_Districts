#pragma once
#include "terrain_query.h"
#include <functional>
#include <tuple>
#include <stdexcept>

namespace c3x_renderer { namespace profile_v2 {
struct CliffPlacement {
    Point position;
    double z=0,scale=0,yaw=0;
    unsigned asset=0;
    bool eligible=false;
};

// Retained joined-source placement. The greedy spacing order is canonical
// world-cell order instead of fixture-crop order. Recursive earlier-neighbor
// evaluation gives the same result regardless of which tile is requested first.
// Only the connected .18-tile exclusion neighborhood is consulted.
template<class Lookup,class Index,class Height,class Shore,class Maximum,class Contour>
auto cliff_placements(World world,int owner_c,int owner_r,Lookup lookup,
                      Index index,Height height,Shore shore,Maximum maximum,Contour contour,
                      std::function<bool()> cancelled={}) {
    struct Candidate { int c,r,n; std::uint64_t rank; CliffPlacement placement; };
    std::map<std::tuple<int,int,int>,Candidate> candidates;
    std::map<std::uint64_t,bool> decisions;
    auto candidate=[&](int c,int r,int n)->Candidate {
        auto key=std::make_tuple(c,r,n);
        auto found=candidates.find(key); if(found!=candidates.end())return found->second;
        Candidate out{c,r,n,0,{}};
        if(cancelled && cancelled())throw std::runtime_error("cliff preparation cancelled");
        auto segment=contour(c,r).at(n);
        auto ci=index(c,r);
        if(ci==std::size_t(-1))return out;
        out.rank=std::uint64_t(ci)*1024+unsigned(n);
        auto p=(segment.a+segment.b)*.5+Point{.5,.5};
        out.placement.position=p;
        if(segment.rocky>=.62) {
            double nx=shore(p.x+.01,p.y)-shore(p.x-.01,p.y);
            double ny=shore(p.x,p.y+.01)-shore(p.x,p.y-.01);
            double length=std::hypot(nx,ny);
            if(length>1e-6) {
                nx/=length;ny/=length;
                auto inland=lookup(int(std::floor(p.x+nx*.42)),int(std::floor(p.y+ny*.42)));
                double top=(height(p.x+nx*.26,p.y+ny*.26)+2.5)/112.;
                if(inland.present && inland.real==5 && top>=.05) {
                    int tc=int(std::floor(p.x+nx*.15)),tr=int(std::floor(p.y+ny*.15));
                    int x=tc+tr,y=tc-tr;
                    if(world.wrap_x)x=mod(x,world.width);
                    if(world.wrap_y)y=mod(y,world.height);
                    // The pinned fixture uses center coordinates for its seed.
                    Point seed_point=p-Point{.5,.5};
                    unsigned seed=hash(unsigned(x)*73856093u^unsigned(y)*19349663u^
                        unsigned(std::floor((seed_point.x-std::floor(seed_point.x))*8))*139u^
                        unsigned(std::floor((seed_point.y-std::floor(seed_point.y))*8))*367u);
                    auto& a=out.placement;a.asset=seed%4;
                    double high=maximum(a.asset);
                    if(high>0) {
                        a.scale=std::clamp((top+.24)/high,.48,.70);
                        a.yaw=std::atan2(ny,nx)+(hash(seed^9347u)&0xffffffu)/16777215.*2.4;
                        a.z=top-high*a.scale-.015;a.eligible=true;
                    }
                }
            }
        }
        candidates.emplace(key,out);return out;
    };
    std::function<bool(Candidate const&,unsigned)> accepted;
    accepted=[&](Candidate const& a,unsigned depth) {
        if(!a.placement.eligible)return false;
        auto found=decisions.find(a.rank);if(found!=decisions.end())return found->second;
        if(depth>4096)throw std::runtime_error("cliff spacing dependency exceeds bound");
        bool result=true;
        for(int r=a.r-1;r<=a.r+1 && result;r++)for(int c=a.c-1;c<=a.c+1 && result;c++) {
            auto count=contour(c,r).size();
            for(int n=0;n<int(count) && result;n++) {
                Candidate b=candidate(c,r,n);
                Point d=a.placement.position-b.placement.position;
                if(b.rank<a.rank && dot(d,d)<.18*.18 && accepted(b,depth+1))result=false;
            }
        }
        decisions[a.rank]=result;return result;
    };
    std::vector<CliffPlacement> result;
    for(int r=owner_r-1;r<=owner_r;r++)for(int c=owner_c-1;c<=owner_c;c++) {
        auto count=contour(c,r).size();
        for(int n=0;n<int(count);n++) {
            auto a=candidate(c,r,n);auto p=a.placement.position;
            if(int(std::floor(p.x))!=owner_c || int(std::floor(p.y))!=owner_r || !accepted(a,0))continue;
            double nx=shore(p.x+.01,p.y)-shore(p.x-.01,p.y);
            double ny=shore(p.x,p.y+.01)-shore(p.x,p.y-.01),length=std::hypot(nx,ny);
            a.placement.position=a.placement.position+Point{nx/length,ny/length}*.02;
            result.push_back(a.placement);
        }
    }
    return result;
}
} }
