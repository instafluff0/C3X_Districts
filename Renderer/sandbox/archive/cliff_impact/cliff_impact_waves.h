#pragma once

// Rocky breakers use the same authoritative coast and generic crest pack as
// the beach ribbons. Only the site selection and impact profile are sandbox art.
namespace sandbox_cliff_waves {
using namespace c3x_renderer::render_core;

struct Site { int c=0,r=0; CoastSegment segment{}; std::uint32_t rank=0; };

inline bool rocky_foot(WorldCoast const& coast,Point foot) {
    auto observe=[](auto...){ };
    auto sample=coast.sample(foot,observe,observe);
    if(sample.rocky<.62)return false;
    double nx=coast.sample(foot+Point{.02,0},observe,observe).distance-
              coast.sample(foot-Point{.02,0},observe,observe).distance;
    double ny=coast.sample(foot+Point{0,.02},observe,observe).distance-
              coast.sample(foot-Point{0,.02},observe,observe).distance;
    double magnitude=std::hypot(nx,ny);
    if(magnitude<1e-6)return false;
    auto inland=coast.world().tile(int(std::floor(foot.x+nx/magnitude*.42)),
                                   int(std::floor(foot.y+ny/magnitude*.42)));
    return inland.present && inland.real==5;
}

inline bool site(WorldCoast const& coast,int c,int r,Site& result) {
    auto index=coast.world().index(c,r);
    if(index==std::size_t(-1))return false;
    auto observe=[](auto...){ };
    auto segments=coast.cell(c,r,observe);
    Point center{double(c)+.5,double(r)+.5};
    double best=1e9;
    for(auto const& segment:segments) {
        if(segment.rocky<.62)continue;
        Point foot=(segment.a+segment.b)*.5+Point{.5,.5};
        if(!rocky_foot(coast,foot))continue;
        double distance=length(foot-center);
        if(distance<best){best=distance;result={c,r,segment,
            hash(std::uint32_t(index)^0x636c6966u)};}
    }
    return best<1e9;
}

inline std::vector<Site> selected_sites(WorldCoast const& coast) {
    auto world=coast.world().dimensions();
    std::map<std::pair<int,int>,bool> cells;
    for(int y=0;y<world.height;++y)for(int x=y&1;x<world.width;x+=2){
        int c=(x+y)/2,r=(x-y)/2;
        auto tile=coast.world().tile(c,r);
        if(!tile.present || tile.real!=5)continue;
        for(int dr=-2;dr<=2;++dr)for(int dc=-2;dc<=2;++dc)
            cells[{c+dc,r+dr}]=true;
    }
    std::vector<Site> candidates;
    for(auto const& cell:cells){Site value;
        if(site(coast,cell.first.first,cell.first.second,value))
            candidates.push_back(value);
    }
    std::sort(candidates.begin(),candidates.end(),[](Site const& a,Site const& b){
        return a.rank>b.rank || (a.rank==b.rank &&
            std::make_pair(a.c,a.r)<std::make_pair(b.c,b.r));
    });
    std::vector<Site> selected;
    for(auto const& candidate:candidates){
        Point at=(candidate.segment.a+candidate.segment.b)*.5;
        bool nearby=false;
        for(auto const& accepted:selected)
            if(length(at-(accepted.segment.a+accepted.segment.b)*.5)<.86){
                nearby=true;break;
            }
        if(!nearby)selected.push_back(candidate);
    }
    return selected;
}

inline std::vector<WavePoint> ribbon(WorldCoast const& coast,Site const& site,float scale) {
    auto observe=[](auto...){ };
    std::vector<CoastSegment> edges;
    for(int y=site.r-2;y<=site.r+2;++y)for(int x=site.c-2;x<=site.c+2;++x){
        auto part=coast.cell(x,y,observe);edges.insert(edges.end(),part.begin(),part.end());
    }
    auto const& chosen=site.segment;
    Point middle=(chosen.a+chosen.b)*.5;
    auto walk=[&](double distance){
        Point previous=middle,at=distance<0?chosen.a:chosen.b;
        double remaining=std::abs(distance);
        for(unsigned step=0;step<256;++step){
            double span=length(at-previous);
            if(span>remaining)return previous+(at-previous)*(remaining/span);
            remaining-=span;
            bool next=false;
            for(auto const& edge:edges){
                Point other;
                if(length(edge.a-at)<1e-6)other=edge.b;
                else if(length(edge.b-at)<1e-6)other=edge.a;
                else continue;
                if(length(other-previous)<1e-6)continue;
                if(step==0 && length(other-(distance<0?chosen.b:chosen.a))<1e-6)
                    continue;
                previous=at;at=other;next=true;break;
            }
            if(!next)return at;
        }
        return at;
    };
    constexpr unsigned rows=40,columns=8;
    std::vector<WavePoint> grid;grid.reserve((rows+1)*(columns+1));
    std::array<float,rows+1> rocky_coverage{};
    auto patch=coast.prepare(middle+Point{.5,.5},2.6,0,observe);
    unsigned active=0;
    for(unsigned row=0;row<=rows;++row){
        double along=double(row)/rows,arc=(along-.5)*2*scale;
        Point foot=walk(arc)+Point{.5,.5};
        Point tangent=walk(arc+.14)-walk(arc-.14);
        double span=length(tangent);
        if(span<1e-8)tangent=chosen.b-chosen.a;
        span=std::max(length(tangent),1e-8);
        Point normal{-tangent.y/span,tangent.x/span};
        if(coast.sample(foot+normal*.08,observe,observe,&patch).distance>0)
            normal=normal*-1;
        auto shore=coast.sample(foot,observe,observe,&patch);
        float coverage=float(std::clamp((shore.rocky-.48)/.30,0.,1.));
        coverage*=float(std::clamp(along/.22,0.,1.)*
            std::clamp((1.-along)/.22,0.,1.));
        rocky_coverage[row]=coverage;
        active+=coverage>.1f;
        for(unsigned column=0;column<=columns;++column){
            float d=.002f+1.08f*column/columns;
            Point p=foot+normal*d;
            double actual=-coast.sample(p,observe,observe,&patch).distance;
            float mask=actual>0 && std::abs(actual-d)<.30?1.f:0.f;
            grid.push_back({p,d,float(along),mask});
        }
    }
    if(active<8)return {};
    for(unsigned row=0;row<=rows;++row){
        float sum=0,weight=0;
        for(int delta=-5;delta<=5;++delta){
            int neighbor=int(row)+delta;
            if(neighbor<0 || neighbor>int(rows))continue;
            float w=float(6-std::abs(delta));
            sum+=rocky_coverage[unsigned(neighbor)]*w;weight+=w;
        }
        for(unsigned column=0;column<=columns;++column)
            grid[row*(columns+1)+column].coverage*=sum/std::max(weight,1.f);
    }
    std::vector<WavePoint> output;output.reserve(rows*columns*6);
    for(unsigned y=0;y<rows;++y)for(unsigned x=0;x<columns;++x){
        unsigned a=y*(columns+1)+x,b=a+1,d=a+columns+1,e=d+1;
        for(auto i:{a,b,e,a,e,d})output.push_back(grid[i]);
    }
    return output;
}
}
