#pragma once
#include "world_coast.h"

namespace c3x_renderer { namespace render_core {
// Generic coastal ribbon. Placement follows the authoritative contour; atlas
// coordinates remain continuous along a ribbon, independently of camera/zoom.
struct WavePoint { Point position; float distance,along,coverage; };
inline double beach_wave_coverage(WorldTopology const& world,Point foot,double rocky) {
    if(rocky>.01)return 0;
    bool beach=false;
    int c=int(std::floor(foot.x)),r=int(std::floor(foot.y));
    for(int y=r-2;y<=r+2;++y)for(int x=c-2;x<=c+2;++x){
        auto t=world.tile(x,y);if(!t.present || water(t))continue;
        double dx=std::max({double(x)-foot.x,0.,foot.x-double(x+1)});
        double dy=std::max({double(y)-foot.y,0.,foot.y-double(y+1)});
        // Include the rocky shoulder outside a hill's nominal diamond. This
        // intentionally leaves a quiet join before the first regular beach.
        if((t.real==5 || t.real==6 || t.real==14) && dx*dx+dy*dy<.64)return 0;
        if(dx*dx+dy*dy<.36)beach=true;
    }
    return beach?1.:0.;
}

inline bool coastal_wave_site(WorldCoast const& coast,int c,int r,CoastSegment& chosen) {
    auto observe=[](auto...){ };
    if(coast.world().index(c,r)==std::size_t(-1))return false;
    auto own=coast.cell(c,r,observe);if(own.empty())return false;
    Point center{double(c)+.5,double(r)+.5};
    chosen=*std::min_element(own.begin(),own.end(),[&](auto const&a,auto const&b){
        return length((a.a+a.b)*.5-center)<length((b.a+b.b)*.5-center);
    });
    Point foot=(chosen.a+chosen.b)*.5+Point{.5,.5};
    return beach_wave_coverage(coast.world(),foot,coast.sample(foot,observe,observe).rocky)>0;
}

inline bool coastal_wave_spaced(WorldCoast const& coast,int c,int r,CoastSegment const& chosen) {
    auto identity=coast.world().index(c,r);
    auto priority=hash(std::uint32_t(identity)^0x77617665u);
    Point center=(chosen.a+chosen.b)*.5;
    // Stable local priority prevents clustered origins without shifting the
    // surviving waves' art or clocks. Consult world topology, not the viewport.
    for(int y=r-2;y<=r+2;++y)for(int x=c-2;x<=c+2;++x){
        auto other=coast.world().index(x,y);
        if(other==identity || other==std::size_t(-1))continue;
        auto rank=hash(std::uint32_t(other)^0x77617665u);
        if(rank>priority || (rank==priority && other>identity))continue;
        CoastSegment neighbor;
        if(!coastal_wave_site(coast,x,y,neighbor))continue;
        if(length((neighbor.a+neighbor.b)*.5-center)<.85)return false;
    }
    return true;
}

inline std::vector<WavePoint> coastal_wave_ribbon(WorldCoast const& coast,int c,int r,float scale) {
    CoastSegment chosen;
    if(!coastal_wave_site(coast,c,r,chosen) || !coastal_wave_spaced(coast,c,r,chosen))return {};
    auto observe=[](auto...){ };
    std::vector<CoastSegment> edges;
    for(int y=r-2;y<=r+2;++y)for(int x=c-2;x<=c+2;++x){
        auto part=coast.cell(x,y,observe);edges.insert(edges.end(),part.begin(),part.end());
    }
    Point middle=(chosen.a+chosen.b)*.5;
    auto walk=[&](double distance){
        Point previous=middle,at=distance<0?chosen.a:chosen.b;
        double remaining=std::abs(distance);
        for(unsigned step=0;step<256;++step){
            double span=length(at-previous);
            if(span>remaining)return previous+(at-previous)*(remaining/span);
            remaining-=span;
            bool next=false;
            for(auto const&e:edges){
                Point other;
                if(length(e.a-at)<1e-6)other=e.b;
                else if(length(e.b-at)<1e-6)other=e.a;
                else continue;
                if(length(other-previous)<1e-6)continue;
                // Do not walk back through the starting segment.
                if(step==0 && length(other-(distance<0?chosen.b:chosen.a))<1e-6)continue;
                previous=at;at=other;next=true;break;
            }
            if(!next)return at;
        }
        return at;
    };
    constexpr unsigned rows=32,columns=6;
    std::vector<WavePoint> grid;grid.reserve((rows+1)*(columns+1));
    unsigned active=0;
    auto patch=coast.prepare(middle+Point{.5,.5},2.2,0,observe);
    for(unsigned row=0;row<=rows;++row){
        double along=double(row)/rows,arc=(along-.5)*2*scale;
        Point foot=walk(arc)+Point{.5,.5};
        Point tangent=walk(arc+.14)-walk(arc-.14);
        double len=length(tangent);if(len<1e-8)tangent=chosen.b-chosen.a;
        len=std::max(length(tangent),1e-8);
        Point normal{-tangent.y/len,tangent.x/len};
        if(coast.sample(foot+normal*.08,observe,observe,&patch).distance>0)normal=normal*-1;
        auto shore=coast.sample(foot,observe,observe,&patch);
        float coverage=float(beach_wave_coverage(coast.world(),foot,shore.rocky));
        active+=coverage>0;
        for(unsigned column=0;column<=columns;++column){
            float d=.002f+.90f*column/columns;
            Point p=foot+normal*d;
            double actual=-coast.sample(p,observe,observe,&patch).distance;
            // Reject ribbons folding into land or crossing a second shore.
            float mask=actual>0 && std::abs(actual-d)<.10?coverage:0;
            grid.push_back({p,d,float(along),mask});
        }
    }
    if(!active)return {};
    std::vector<WavePoint> result;result.reserve(rows*columns*6);
    for(unsigned y=0;y<rows;++y)for(unsigned x=0;x<columns;++x){
        unsigned a=y*(columns+1)+x,b=a+1,d=a+columns+1,e=d+1;
        for(auto i:{a,b,e,a,e,d})result.push_back(grid[i]);
    }
    return result;
}
} }
