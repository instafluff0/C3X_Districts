#pragma once
#include "terrain_query.h"
#include <functional>
#include <tuple>
#include <stdexcept>

namespace c3x_renderer { namespace render_core {
struct CliffPlacement {
    Point position;
    double z=0,scale=0,yaw=0;
    unsigned asset=0;
    bool eligible=false;
};
struct CliffRecipe { unsigned asset=0; double scale=1,variation=.1; };

// Source position and normal travel through the same world basis. The vertical
// metric matches imported features; inverse transpose preserves face normals.
struct CliffTransform {
    CliffPlacement const& instance;
    float vertical_basis;
    double cosine, sine;
    CliffTransform(CliffPlacement const& p,float vertical)
        :instance(p),vertical_basis(vertical),cosine(std::cos(p.yaw)),sine(std::sin(p.yaw)) {}
    template<class Vector> std::array<float,3> position(Vector const& p) const {
        return {float(instance.position.x+(p[0]*cosine-p[1]*sine)*instance.scale),
            float(instance.position.y+(p[0]*sine+p[1]*cosine)*instance.scale),
            float(instance.z+p[2]*instance.scale*vertical_basis)};
    }
    template<class Vector> std::array<float,3> normal(Vector const& n) const {
        return {float(n[0]*cosine-n[1]*sine),float(n[0]*sine+n[1]*cosine),
            float(n[2]/vertical_basis)};
    }
};


// Retained joined-source placement. The greedy spacing order is canonical
// world-cell order instead of fixture-crop order. Recursive earlier-neighbor
// evaluation gives the same result regardless of which tile is requested first.
// Only the connected .20-tile exclusion neighborhood is consulted. The four
// large source bodies establish the cliff line; the four authored small bodies
// dress its foot and top instead of synthesizing replacement rock geometry.
template<class Lookup,class Index,class Height,class Shore,class Maximum,class Contour>
auto cliff_placements(World world,int owner_c,int owner_r,Lookup lookup,
                      Index index,Height height,Shore shore,Maximum maximum,Contour contour,
                      std::function<CliffRecipe(bool,unsigned)> recipe={},
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
                    auto selected=recipe?recipe(false,seed):CliffRecipe{seed%4,1,.1};
                    auto& a=out.placement;a.asset=selected.asset;
                    double high=maximum(a.asset);
                    if(high>0 && selected.scale>0) {
                        double variation=(hash(seed^2179u)&0xffffffu)/16777215.;
                        // Fit the above-anchor body to the local cliff height
                        // with one uniform scale; retain authored proportions.
                        a.scale=std::clamp((top-2.5/112.)/high,.42,.52)*selected.scale*
                            (1+(variation*2-1)*selected.variation);
                        a.yaw=std::atan2(ny,nx)+
                            (hash(seed^9347u)&0xffffffu)/16777215.*6.283185307179586;
                        // Embed the authored body into the raised terrain rim.
                        // Source origin is preserved; the rim fit is C3X placement.
                        a.z=std::max(2.5/112.,top-high*a.scale*.70);a.eligible=true;
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
                if(b.rank<a.rank && dot(d,d)<.20*.20 && accepted(b,depth+1))result=false;
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
            nx/=length;ny/=length;
            Point tangent{-ny,nx};
            unsigned seed=hash(unsigned(a.rank)^0x7f4a7c15u);
            a.placement.position=a.placement.position+Point{nx,ny}*.08;
            result.push_back(a.placement);

            auto append_detail=[&](unsigned salt,bool upper) {
                CliffPlacement detail;
                unsigned detail_seed=hash(seed^salt);
                auto selected=recipe?recipe(true,detail_seed):CliffRecipe{4+detail_seed%4,1,.15};
                detail.asset=selected.asset;
                double high=maximum(detail.asset);
                if(high<=0 || selected.scale<=0)return;
                detail.scale=(upper?.27:.40)*selected.scale*(1+selected.variation*
                    ((hash(detail_seed^0x27d4eb2du)&0xffffffu)/16777215.*2-1));
                detail.yaw=std::atan2(ny,nx)+
                    (hash(detail_seed^0x165667b1u)&0xffffffu)/16777215.*6.283185307179586;
                double along=((hash(detail_seed^0xd3a2646cu)&0xffffffu)/16777215.-.5)*.28;
                double spread=(hash(detail_seed^0xb5297a4du)&0xffffffu)/16777215.;
                double across=upper?.18+.28*spread:-(.19+.25*spread);
                detail.position=p+tangent*along+Point{nx,ny}*across;
                detail.z=upper?(height(detail.position.x,detail.position.y)+2.5)/112.:2.5/112.+.02;
                detail.eligible=true;result.push_back(detail);
            };
            append_detail(0x9e3779b9u,false);
            if((hash(seed^0x85ebca6bu)%3u)==0)append_detail(0xc2b2ae35u,true);
        }
    }
    return result;
}
} }
