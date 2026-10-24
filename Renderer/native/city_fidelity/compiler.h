#pragma once
#include "runtime.h"
#include "wonders.h"
#include "../c3x_renderer_api.h"
#include "../../lab/shared/natural/vertex.h"
#include <array>
#include <memory>
namespace c3x_renderer { namespace city_fidelity {
// Stable composition selection and CPU output, without a renderer/GPU owner.
// Site queries remain dependency-recording callbacks under the caller's lease.
struct Chunk {
    std::vector<fidelity::MapVertex> vertices;
    std::vector<unsigned> indices;
    unsigned material=0;bool environment=false;float atlas[4]={};
    std::shared_ptr<Lighting> lighting;
    // Rigid parts may share this source mesh plus placement. Ground and paving
    // require terrain-dependent vertices. IDs belong to the current Library.
    unsigned source_model=~0u,source_part=~0u;
    WorldInstance placement;
    bool terrain_conforming=false;
    // Attached effect quads follow the visual clock and redraw per frame.
    bool effect=false;
    // Seasons it draws in (render_core/season_stand.h), 0 for every season, and what
    // keeps its source resident (a constructed wonder's group, city_fidelity/wonders.h).
    unsigned char season_stand=0;
    std::shared_ptr<void const> lease;
};
struct Surfaces {std::vector<Chunk> chunks;};
template<class World,class Shore,class River,class Height>
Composition const* select(Library const& library,c3x_renderer_tile_v1 const& record,int column,int row,
        World world_lookup,Shore shore_sample_at,River river_distance,Height height_natural){
    if(record.city_id<0)return nullptr;
    unsigned culture=unsigned(std::clamp(record.city_culture_group,0,4));
    unsigned era=unsigned(std::clamp(record.city_era,0,3)),size=unsigned(std::clamp(record.city_size,0,2));
    bool capital=(record.city_flags&C3X_RENDERER_CITY_CAPITAL)!=0;
    bool walled=size==0 && (record.city_flags&C3X_RENDERER_CITY_WALLED)!=0;
    unsigned variants=1;
    for(auto const&t:library.compositions)
        if(t.culture==culture && t.era==era && t.size==size && t.capital==unsigned(capital) &&
           (!t.owns_walls || t.walled==unsigned(walled)))variants=std::max(variants,t.variant+1);
    // The capture seed is map seed + canonical tile coordinates. Ownership,
    // population and capture order never reshuffle a city's variant.
    unsigned chosen=record.variant_seed%variants;
    // Version-four recipes accept native city anchors directly. The remaining
    // clearance search supports older standalone study libraries only.
    for(unsigned pass=0;pass<2;pass++)for(unsigned alternative=0;alternative<variants;++alternative)
      for(auto const&t:library.compositions){
        if(t.culture!=culture || t.era!=era || t.size!=size ||
            t.variant!=(chosen+alternative)%variants || (t.owns_walls && t.walled!=unsigned(walled)) ||
            (pass==0?t.capital!=unsigned(capital):t.capital!=0 || !capital))continue;
        // Authored layouts use Civ III's legal city anchor. Their trees and
        // outskirts may cross neighboring tile edges; reject neither the whole
        // city nor its capital because a neighboring tile is coast or relief.
        // Individual rigid buildings still use the terrain at their own anchor.
        if(t.anchor_layout)return &t;
        bool legal=true;unsigned query_budget=4096;
        auto clear_box=[&](auto&&self,float x,float y,float dx,float dy,unsigned depth)->bool{
            if(!query_budget)return false;
            --query_budget;
            double shore=shore_sample_at(x,y).distance;
            double river=t.clearance[3]>0?river_distance(x,y):1e9;
            if(shore<t.clearance[0] || river<t.clearance[3])return false;
            double radius=std::hypot(dx,dy),screen_radius=std::hypot((dx+dy)*64,(dx+dy)*32);
            if(shore>=t.clearance[0]+radius && river>=t.clearance[3]+screen_radius)return true;
            // Certify smaller rectangles near a boundary rather than shrinking
            // the source body to satisfy its former containing-circle test.
            if(depth==6)return false;
            for(int sy:{-1,1})for(int sx:{-1,1})if(!self(self,x+float(sx)*dx*.5f,y+float(sy)*dy*.5f,dx*.5f,dy*.5f,depth+1))return false;
            return true;
        };
        for(auto const&i:t.instances){
            float x=float(column)+.5f+i.offset[0]+(i.bounds[0]+i.bounds[2])*.5f;
            float y=float(row)+.5f-i.offset[1]-(i.bounds[1]+i.bounds[3])*.5f;
            float dx=(i.bounds[2]-i.bounds[0])*.5f,dy=(i.bounds[3]-i.bounds[1])*.5f;
            float margin=t.clearance[2],paving_margin=t.paving.vertices.empty()?0.f:.1f;
            for(int ry=int(std::floor(y-dy-margin));ry<=int(std::floor(y+dy+margin));ry++)
                for(int cx=int(std::floor(x-dx-margin));cx<=int(std::floor(x+dx+margin));cx++){
                    auto land=world_lookup(cx,ry);
                    if(land.base<0 || land.base>=11 || land.real==6 || land.real==10 ||
                        (margin>0 && (land.real==7 || land.real==8)))legal=false;
                }
            if(!legal || !clear_box(clear_box,x,y,dx+paving_margin,dy+paving_margin,0)){legal=false;break;}
            float low=height_natural(x,y),high=low;
            for(int sy:{-1,1})for(int sx:{-1,1}){float h=height_natural(x+float(sx)*dx,y+float(sy)*dy);low=std::min(low,h);high=std::max(high,h);}
            if(high-low>t.clearance[1]){legal=false;break;}
        }
        if(legal)return &t;
    }
    return nullptr;
}
// A site-optional body (version-five packs) yields when its footprint reaches
// water or a mountain tile, comes within clearance[3] source pixels of a river
// centreline, within clearance[0] tiles of the shore, or spans more than
// clearance[1] relief units. Bridges stand over river channels, so the river
// clearance also keeps them open. Unflagged bodies are always kept.
// `offset` tests the body at another offset than its authored one.
template<class World,class Shore,class River,class Height>
bool site_keeps(Composition const& composition,Instance const& i,int nc,int nr,
        World world_lookup,Shore shore_sample_at,River river_distance,Height height_natural,float const* offset=nullptr){
    if(!(i.flags&instance_site_optional))return true;
    if(!offset)offset=i.offset;
    float cx=float(nc)+.5f+offset[0],cy=float(nr)+.5f-offset[1];
    float x0=cx+i.bounds[0],x1=cx+i.bounds[2],y0=cy-i.bounds[3],y1=cy-i.bounds[1];
    for(int row=int(std::floor(y0));row<=int(std::floor(y1));++row)
        for(int column=int(std::floor(x0));column<=int(std::floor(x1));++column){
            auto land=world_lookup(column,row);
            if(land.base<0 || land.base>=11 || land.real==6 || land.real==10)return false;
        }
    unsigned across=std::clamp(unsigned(std::ceil((x1-x0)/.04f)),1u,12u);
    unsigned deep=std::clamp(unsigned(std::ceil((y1-y0)/.04f)),1u,12u);
    float low=1e9f,high=-1e9f;
    for(unsigned b=0;b<=deep;++b)for(unsigned a=0;a<=across;++a){
        float x=x0+(x1-x0)*float(a)/float(across),y=y0+(y1-y0)*float(b)/float(deep);
        if(shore_sample_at(x,y).distance<composition.clearance[0])return false;
        if(composition.clearance[3]>0 && river_distance(x,y)<composition.clearance[3])return false;
        float h=height_natural(x,y);low=std::min(low,h);high=std::max(high,h);
    }
    return high-low<=composition.clearance[1];
}
// The ground plate of a site-aware composition fades out over water and
// mountain tiles and toward the same shore and river clearances as its bodies.
// From version six no plate lies nearer a river than a body may stand.
template<class World,class Shore,class River>
float site_ground(Composition const& composition,float x,float y,World world_lookup,Shore shore_sample_at,River river_distance){
    auto land=world_lookup(int(std::floor(x)),int(std::floor(y)));
    if(land.base<0 || land.base>=11 || land.real==6 || land.real==10)return 0.f;
    auto ramp=[](float value,float low,float high){
        float t=std::clamp((value-low)/(high-low),0.f,1.f);return t*t*(3.f-2.f*t);};
    float shore=float(shore_sample_at(x,y).distance);
    float coverage=ramp(shore,composition.clearance[0],composition.clearance[0]+.06f);
    float river=composition.clearance[3]-(composition.site_moves?0.f:2.f);
    if(composition.clearance[3]>0)
        coverage=std::min(coverage,ramp(float(river_distance(x,y)),river,river+6.f));
    return coverage;
}
struct ContinueCompilation {bool operator()()const{return false;}};
// A body where its site puts it: the instance and its offset in tiles. A
// moved body left its authored offset for a legal one.
struct SiteBody {unsigned index;float offset[2];bool moved;};
// Packs before version five carry no site flags: every body and the complete
// ground plate are kept.
struct EverySite {
    std::vector<SiteBody> layout(Composition const& composition)const{
        std::vector<SiteBody> bodies;
        for(unsigned index=0;index<composition.instances.size();++index)
            bodies.push_back({index,{composition.instances[index].offset[0],composition.instances[index].offset[1]},false});
        return bodies;
    }
    float ground(float,float)const{return 1.f;}
};
// Offset-space boxes: x is the column offset, y the authored (row-up) offset.
inline float site_box_distance(std::array<float,4> const& b,float x,float y){
    return std::hypot(std::max({b[0]-x,0.f,x-b[2]}),std::max({b[1]-y,0.f,y-b[3]}));
}
template<class World,class Shore,class River,class Height>
struct SiteFilter {
    Composition const& composition;int nc,nr;World world;Shore shore;River river;Height height;
    // Most sites have no water or mountain within reach of the plate: one
    // centre/neighbour test then spares a query per plate vertex.
    mutable int clear=-1;
    mutable bool laid=false;
    mutable std::vector<SiteBody> bodies;
    // Plate owners: bodies standing (kept or moved) and spots left empty.
    mutable std::vector<std::array<float,4>> standing,vacated;
    // Version six: a building or accent that yields moves to the nearest
    // free spot that passes the same site test, within 0.2 tile beyond the
    // authored reach, so a city by a river or a coast keeps its count of
    // buildings. A spot on the city's own bank wins over a nearer one across
    // the river. Trees under a moved body give way. A body with no such spot
    // stays out, as every yielded body does in earlier packs.
    std::vector<SiteBody> const& layout(Composition const&)const{
        if(laid)return bodies;
        laid=true;
        auto const& c=composition;
        auto box=[](Instance const& i,float x,float y){
            return std::array<float,4>{x+i.bounds[0],y+i.bounds[1],x+i.bounds[2],y+i.bounds[3]};};
        auto reach=[](std::array<float,4> const& b){
            float r=0;
            for(float x:{b[0],b[2]})for(float y:{b[1],b[3]})
                r=std::max(r,std::pow(std::pow(std::abs(x),6.f)+std::pow(std::abs(y),6.f),1.f/6.f));
            return r;};
        std::vector<char> kept(c.instances.size(),1);
        std::vector<unsigned> yielded;
        float limit=0;
        for(unsigned index=0;index<c.instances.size();++index){
            auto const& i=c.instances[index];
            bool tree=(i.flags&instance_tree)!=0;
            auto b=box(i,i.offset[0],i.offset[1]);
            if(!tree)limit=std::max(limit,reach(b));
            if(!c.site_aware || site_keeps(c,i,nc,nr,world,shore,river,height)){
                if(!tree)standing.push_back(b);
                continue;
            }
            kept[index]=0;
            if(!tree){vacated.push_back(b);yielded.push_back(index);}
        }
        limit+=.2f;
        constexpr float step=.03f,radius=.8f,gap=.006f;
        constexpr int span=int(radius/step);
        std::vector<std::pair<float,int>> order;
        std::vector<SiteBody> moved;
        unsigned tests=0;
        if(!c.site_moves)yielded.clear();
        for(unsigned index:yielded){
            auto const& i=c.instances[index];
            order.clear();
            for(int gy=-span;gy<=span;++gy)for(int gx=-span;gx<=span;++gx){
                float dx=float(gx)*step,dy=float(gy)*step,d=std::hypot(dx,dy);
                // Nearest to the authored spot, then nearer the city centre.
                if(d>0 && d<=radius)order.push_back({d+.35f*std::hypot(i.offset[0]+dx,i.offset[1]+dy),
                    (gy+span)*(2*span+1)+gx+span});
            }
            std::sort(order.begin(),order.end());
            // found: 0 no spot yet, 1 one across a river, 2 one on the own bank.
            unsigned own_tests=0,skipped=0,found=0;float spot[2]={};
            for(auto const& candidate:order){
                float x=i.offset[0]+float(candidate.second%(2*span+1)-span)*step;
                float y=i.offset[1]+float(candidate.second/(2*span+1)-span)*step;
                auto b=box(i,x,y);
                if(reach(b)>limit)continue;
                bool free=true;
                for(auto const& other:standing)
                    if(b[2]+gap>other[0] && other[2]+gap>b[0] && b[3]+gap>other[1] && other[3]+gap>b[1]){free=false;break;}
                // The site test always samples the corners: a cheap first check.
                for(float px:{b[0],b[2]})for(float py:{b[1],b[3]}){
                    float wx=float(nc)+.5f+px,wy=float(nr)+.5f-py;
                    free=free && shore(wx,wy).distance>=c.clearance[0] &&
                        (c.clearance[3]<=0 || river(wx,wy)>=c.clearance[3]);
                }
                if(!free)continue;
                // On the city's own bank the way from the centre never
                // crosses river water: samples under 8 pixels apart always
                // land in the 12-pixel band nearer than 6 to its centreline.
                float cx=(b[0]+b[2])*.5f,cy=(b[1]+b[3])*.5f;
                unsigned samples=unsigned(std::ceil(std::hypot(cx,cy)/.06f));
                bool crossing=false;
                for(unsigned s=1;s<=samples && !crossing;++s){
                    float t=float(s)/float(samples);
                    crossing=river(float(nc)+.5f+cx*t,float(nr)+.5f-cy*t)<6.;
                }
                // With a spot across the river in hand, only the own bank is sought.
                if(crossing && found){if(++skipped>48)break;continue;}
                if(++own_tests>24 || ++tests>192)break;
                float offset[2]={x,y};
                if(!site_keeps(c,i,nc,nr,world,shore,river,height,offset))continue;
                if(!found || !crossing){spot[0]=x;spot[1]=y;found=crossing?1:2;}
                if(found==2)break;
            }
            if(found){standing.push_back(box(i,spot[0],spot[1]));moved.push_back({index,{spot[0],spot[1]},true});}
        }
        for(unsigned index=0;index<c.instances.size();++index){
            auto const& i=c.instances[index];
            if(!kept[index])continue;
            if(i.flags&instance_tree){
                auto b=box(i,i.offset[0],i.offset[1]);bool under=false;
                for(auto const& m:moved){
                    auto o=box(c.instances[m.index],m.offset[0],m.offset[1]);
                    under=under || (b[2]>o[0] && o[2]>b[0] && b[3]>o[1] && o[3]>b[1]);
                }
                if(under)continue;
            }
            bodies.push_back({index,{i.offset[0],i.offset[1]},false});
        }
        bodies.insert(bodies.end(),moved.begin(),moved.end());
        return bodies;
    }
    float ground(float x,float y)const{
        if(!composition.site_aware)return 1.f;
        if(clear<0){
            float cx=float(nc)+.5f,cy=float(nr)+.5f,reach=1.3f; // plate stays within ~0.9 tile per axis
            clear=float(shore(cx,cy).distance)>composition.clearance[0]+reach+.06f &&
                (composition.clearance[3]<=0 || float(river(cx,cy))>composition.clearance[3]+6.f+reach*64.f);
            for(int r=nr-1;r<=nr+1 && clear;++r)for(int c=nc-1;c<=nc+1 && clear;++c){
                auto land=world(c,r);
                if(land.base<0 || land.base>=11 || land.real==6 || land.real==10)clear=0;
            }
        }
        float coverage=clear?1.f:site_ground(composition,x,y,world,shore,river);
        layout(composition);
        if(!composition.site_moves || vacated.empty() || coverage<=0)return coverage;
        // Plate laid for a body that left goes with it; plate nearer a
        // standing body (or the central plaza) stays.
        float ox=x-float(nc)-.5f,oy=float(nr)+.5f-y;
        float near_standing=std::max(0.f,std::hypot(ox,oy)-.12f),near_vacated=1e9f;
        for(auto const& b:standing)near_standing=std::min(near_standing,site_box_distance(b,ox,oy));
        for(auto const& b:vacated)near_vacated=std::min(near_vacated,site_box_distance(b,ox,oy));
        auto ramp=[](float value,float low,float high){
            float t=std::clamp((value-low)/(high-low),0.f,1.f);return t*t*(3.f-2.f*t);};
        return coverage*(1.f-ramp(near_standing-near_vacated,0.f,.03f)*ramp(near_standing,.05f,.09f));
    }
};
template<class World,class Shore,class River,class Height>
SiteFilter<World,Shore,River,Height> site_filter(Composition const& composition,int nc,int nr,
        World world,Shore shore,River river,Height height){
    return {composition,nc,nr,world,shore,river,height};
}
// An attached effect at a world anchor: column, row, height in tile units.
struct EffectAnchor {float world[3];Effect effect;};
// Attached effects become one chunk of camera-facing quads drawn by the city
// effect layer (q8_city_effect, city_scene_material.hlsl).
template<class Project>
void append_effects(Library const& library,std::vector<EffectAnchor> const& anchors,
        std::shared_ptr<Lighting> const& lighting,Project project_natural,bool indexed,Surfaces& output){
    if(anchors.empty() || library.effect_material>=library.materials.size() ||
       !library.materials[library.effect_material].ground)return;
    // Screen-aligned quads in world space: (+1,+1) in column/row moves
    // only across the screen and height only up it, so every projection
    // path places these exactly as it places the bodies. Read-only depth
    // and the ground material keep them out of shadows and depth writes.
    Chunk chunk;chunk.material=library.effect_material;chunk.lighting=lighting;chunk.terrain_conforming=true;
    chunk.effect=true;
    std::vector<fidelity::MapVertex> corners;
    for(auto const& a:anchors){
        auto const& e=a.effect;
        // Smoke rises from its anchor; flames and lights centre on theirs.
        // Natural-wonder geysers and spray rise from their foot too.
        bool rises=e.kind==float(effect_smoke) || e.kind==float(effect_volcano_plume) ||
            e.kind==float(effect_geyser) || e.kind==float(effect_spray);
        float half=e.width*.5f,low=rises?0.f:-.5f*e.height,high=low+e.height;
        float uv[4][2]={{-1,0},{1,0},{1,1},{-1,1}};
        unsigned base=unsigned(corners.size());
        for(auto const& c:uv){
            float x=a.world[0]+c[0]*half,y=a.world[1]+c[0]*half,z=a.world[2]+(c[1]?high:low);
            auto out=project_natural(x,y,z*112);
            out.u=c[0];out.v=c[1];out.normal_x=out.normal_y=0;out.normal_z=1;
            out.macro_u=e.seed;out.macro_v=e.intensity;
            out.material_grass=1;out.material_plains=out.material_desert=out.material_marsh=0;
            out.authored_relief_height=out.authored_relief_blend=0;
            out.relief_owner_u=e.width;out.relief_owner_v=e.height;
            out.base_terrain=90.f+e.kind;corners.push_back(out);
        }
        unsigned quad[]={base,base+1,base+2,base,base+2,base+3};
        chunk.indices.insert(chunk.indices.end(),std::begin(quad),std::end(quad));
    }
    if(indexed)chunk.vertices=std::move(corners);
    else {for(unsigned index:chunk.indices)chunk.vertices.push_back(corners[index]);chunk.indices.clear();}
    output.chunks.push_back(std::move(chunk));
}
// An active Civ III volcano's plume rising from its crater: `crater` is the
// rendered surface height at the tile centre (mountain units); an eruption
// thickens it and lights its base. Needs the city pack's effect material.
template<class Project>
bool volcano_plume(Library const& library,int nc,int nr,float crater,bool erupting,
        Project project_natural,bool indexed,Surfaces& output){
    // A narrow, tall quad (like the chimney smoke's) leaves the puffs room to
    // billow and rise; an eruption's column is larger and denser.
    EffectAnchor plume{{float(nc)+.5f,float(nr)+.5f,(crater+3.f)/112},
        Effect{{0,0,0},float(effect_volcano_plume),erupting?.55f:.45f,erupting?1.8f:1.5f,
            float((unsigned(nc)*73u^unsigned(nr)*151u)%64u),erupting?1.6f:1.f}};
    auto before=output.chunks.size();
    append_effects(library,{plume},std::make_shared<Lighting>(),project_natural,indexed,output);
    return output.chunks.size()>before;
}
// A natural wonder's attached effects (docs/natural_wonder_rendering.md). A record is
// kind, foot and top offsets from the tile centre, lift (mountain units), width, height
// (tile units; 0 reaches the lifted top on screen), strength and seed. Waterfalls (5)
// are a ribbon draped from the top point down the face to the foot, standing off it by
// the lift; the rest are camera-facing quads rising from the foot.
template<class Surface,class Project>
bool wonder_effects(Library const& library,int nc,int nr,std::vector<float> const& records,
        Surface surface,Project project_natural,bool indexed,Surfaces& output){
    std::vector<EffectAnchor> anchors;
    Chunk ribbon;ribbon.material=library.effect_material;ribbon.lighting=std::make_shared<Lighting>();
    ribbon.terrain_conforming=true;ribbon.effect=true;
    std::vector<fidelity::MapVertex> strip;
    for(std::size_t i=0;i+10<=records.size();i+=10){
        float const* e=records.data()+i;
        float fx=float(nc)+.5f+e[1],fy=float(nr)+.5f+e[2],tx=float(nc)+.5f+e[3],ty=float(nr)+.5f+e[4];
        if(e[0]>float(effect_lake)-.5f && e[0]<float(effect_channel)+.5f)continue;  // water layers (c3x_renderer.cpp)
        if(e[0]==float(effect_waterfall)){
            float dx=tx-fx,dy=ty-fy,length=std::max(std::sqrt(dx*dx+dy*dy),1e-4f);
            float ax=-dy/length*e[6]*.5f,ay=dx/length*e[6]*.5f;
            unsigned base=unsigned(strip.size());constexpr int steps=32;
            // Free fall: from the lip the water drops on a smooth arc to its foot instead of
            // following each ledge of the face; it stays clear of the rock, smoothed so a
            // ledge bends it gently rather than in a step.
            float top=surface(tx,ty)+e[5],bottom=surface(fx,fy)+e[5]*.5f;
            std::array<float,steps+1> clear{},fall{};
            for(int s=0;s<=steps;++s){
                float v=float(s)/steps,x=fx+dx*v,y=fy+dy*v,p=1-v;
                clear[s]=std::max({surface(x,y),surface(x+ax,y+ay),surface(x-ax,y-ay)})+e[5]*.6f;
                fall[s]=std::max(top-(top-bottom)*p*p,clear[s]);
            }
            for(int pass=0;pass<4;++pass)for(int s=1;s<steps;++s)
                fall[s]=std::max(clear[s],(fall[s-1]+2*fall[s]+fall[s+1])*.25f);
            for(int s=0;s<=steps;++s){
                float v=float(s)/steps,x=fx+dx*v,y=fy+dy*v,z=fall[s];
                for(int side=-1;side<=1;side+=2){
                    auto out=project_natural(x+ax*side,y+ay*side,z);
                    out.u=float(side);out.v=v;out.normal_x=out.normal_y=0;out.normal_z=1;
                    out.macro_u=e[9];out.macro_v=e[8];
                    out.material_grass=1;out.material_plains=out.material_desert=out.material_marsh=0;
                    out.authored_relief_height=out.authored_relief_blend=0;
                    out.relief_owner_u=e[6];out.relief_owner_v=length;
                    out.base_terrain=90.f+e[0];strip.push_back(out);
                }
            }
            for(unsigned s=0;s<unsigned(steps);++s){
                unsigned a=base+2*s,quad[]={a,a+1,a+3,a,a+3,a+2};
                ribbon.indices.insert(ribbon.indices.end(),std::begin(quad),std::end(quad));
            }
            continue;
        }
        float foot=surface(fx,fy),height=e[7];
        if(!(height>0)){
            auto a=project_natural(fx,fy,foot),b=project_natural(tx,ty,surface(tx,ty)+e[5]),
                 c=project_natural(fx,fy,foot+112.f);
            height=(a.y-b.y)/std::max(1e-3f,a.y-c.y);
        }
        anchors.push_back({{fx,fy,foot/112},Effect{{0,0,0},e[0],e[6],std::max(height,.05f),e[9],e[8]}});
    }
    auto before=output.chunks.size();
    append_effects(library,anchors,std::make_shared<Lighting>(),project_natural,indexed,output);
    if(!strip.empty() && library.effect_material<library.materials.size()){
        if(indexed)ribbon.vertices=std::move(strip);
        else {for(unsigned index:ribbon.indices)ribbon.vertices.push_back(strip[index]);ribbon.indices.clear();}
        output.chunks.push_back(std::move(ribbon));
    }
    return output.chunks.size()>before;
}
template<class Height,class Project,class Stop=ContinueCompilation,class Site=EverySite>
bool compile(Library const& library,Composition const& selected,int nc,int nr,
        Height height_natural,Project project_natural,Surfaces& output,Stop stop={},bool indexed=false,Site site={}){
    auto composition=&selected;
    auto lighting=std::make_shared<Lighting>();
    // Attached effects, in world space, become camera-facing quads after the
    // bodies they belong to.
    std::vector<EffectAnchor> anchors;
    lighting->blockers.reserve(composition->instances.size());
    auto const& bodies=site.layout(*composition);
    for(auto const& body:bodies){
        if(stop())return false;
        auto const&i=composition->instances[body.index];auto const&m=library.models[i.model];
        // Lights name their own body's blocker, so skipped bodies leave no gap.
        unsigned const owner=unsigned(lighting->blockers.size());
        float center_x=float(nc)+.5f+body.offset[0],center_y=float(nr)+.5f-body.offset[1];
        float ground_center=height_natural(center_x,center_y);
        float x_low=center_x+i.bounds[0],x_high=center_x+i.bounds[2];
        float y_low=center_y-i.bounds[3],y_high=center_y-i.bounds[1];
        float corners[4][2]={{x_low,y_low},{x_high,y_low},
                              {x_high,y_high},{x_low,y_high}};
        float ground_low=ground_center,ground_high=ground_center;
        // Hills can crest in the middle of a plot, not just at its corners.
        // Sample the complete building footprint before choosing its level.
        for(unsigned row=0;row<=8;row++)for(unsigned column=0;column<=8;column++){
            float x=x_low+(x_high-x_low)*float(column)/8.f;
            float y=y_low+(y_high-y_low)*float(row)/8.f;
            float ground=height_natural(x,y);
            ground_low=std::min(ground_low,ground);
            ground_high=std::max(ground_high,ground);
        }
        bool terrace=composition->foundation_material!=~0u &&
            ground_high-ground_low>.75f;
        float lowest_source=std::min(0.f,m.low[2]*i.scale/source_z_metric);
        auto placement=place(i,center_x-i.offset[0],center_y+i.offset[1],
                             terrace?ground_high-lowest_source+.02f:ground_center);
        for(auto const&l:i.lights)lighting->lights.push_back(placement.light(l,owner));
        for(auto const&e:i.effects){EffectAnchor a;placement.position(e.position,a.world);a.effect=e;anchors.push_back(a);}
        Lighting::Box b={{placement.x+i.bounds[0],-placement.y+i.bounds[1],
            placement.z*source_z_metric+std::max(0.f,m.low[2])*i.scale,0},
            {placement.x+i.bounds[2],-placement.y+i.bounds[3],placement.z*source_z_metric+m.high[2]*i.scale,0}};
        lighting->blockers.push_back(b);
        if(terrace){
            // A rigid building stays upright on a level plot. Its compact
            // retaining skirt follows the sampled downhill terrain while the
            // original mesh, normals and proportions remain untouched.
            unsigned foundation_material=~0u;
            float foundation_u=0.f,foundation_v=0.f,lowest=1e9f;
            for(auto const&part:m.parts)if(!library.materials[part.material].ground)
                for(auto const&vertex:part.vertices)if(vertex.position[2]<lowest){
                    lowest=vertex.position[2];foundation_material=part.material;
                    foundation_u=vertex.uv0[0];foundation_v=vertex.uv0[1];
                }
            if(foundation_material!=~0u){
                if(composition->foundation_material!=~0u)
                    foundation_material=composition->foundation_material;
                float u0=composition->foundation_material!=~0u?composition->foundation_uv[0]:foundation_u;
                float v0=composition->foundation_material!=~0u?composition->foundation_uv[1]:foundation_v;
                float u1=composition->foundation_material!=~0u?composition->foundation_uv[2]:foundation_u;
                float v1=composition->foundation_material!=~0u?composition->foundation_uv[3]:foundation_v;
                Chunk support;support.material=foundation_material;
                support.environment=composition->environment!=0;
                support.lighting=lighting;
                float top=placement.z*112.f+m.low[2]*i.scale/source_z_metric+.006f;
                auto vertex=[&](float x,float y,float z,float nx,float ny,float nz,float u,float v){
                    auto out=project_natural(x,y,z);
                    out.u=u;out.v=v;
                    out.normal_x=nx;out.normal_y=ny;out.normal_z=nz;
                    out.material_grass=1.f;out.material_plains=0.f;
                    out.material_desert=0.f;out.material_marsh=0.f;
                    out.base_terrain=100.f+float(library.materials[foundation_material].channels);
                    return out;
                };
                for(unsigned side=0;side<4;side++){
                    unsigned next=(side+1u)%4u;
                    float dx=corners[next][0]-corners[side][0];
                    float dy=corners[next][1]-corners[side][1];
                    float length=std::hypot(dx,dy);
                    float nx=length>0?dy/length:0.f,ny=length>0?-dx/length:0.f;
                    float module_x=composition->foundation_step[0]>0?
                        composition->foundation_step[0]:length/8.f;
                    float module_z=composition->foundation_step[1]>0?
                        composition->foundation_step[1]:std::max(1.f,top-ground_low);
                    unsigned sections=std::clamp(unsigned(std::ceil(length/module_x)),1u,64u);
                    for(unsigned section=0;section<sections;section++){
                        float start=float(section)/float(sections),end=float(section+1u)/float(sections);
                        float x0=corners[side][0]+dx*start,y0=corners[side][1]+dy*start;
                        float x1=corners[side][0]+dx*end,y1=corners[side][1]+dy*end;
                        float g0=height_natural(x0,y0)-.02f,g1=height_natural(x1,y1)-.02f;
                        float us=u0,ue=u0+(u1-u0)*(length*(end-start)/module_x);
                        unsigned courses=std::clamp(unsigned(std::ceil((top-std::min(g0,g1))/module_z)),1u,64u);
                        for(unsigned course=0;course<courses;course++){
                            float level=top-float(course)*module_z;
                            float upper0=std::max(level,g0),upper1=std::max(level,g1);
                            float lower0=std::max(level-module_z,g0),lower1=std::max(level-module_z,g1);
                            if(upper0<=lower0 && upper1<=lower1)continue;
                            auto uv_v=[&](float z){return v0+(v1-v0)*
                                std::clamp((level-z)/module_z,0.f,1.f);};
                            unsigned base=unsigned(support.vertices.size());
                            support.vertices.push_back(vertex(x0,y0,upper0,nx,ny,0.f,us,uv_v(upper0)));
                            support.vertices.push_back(vertex(x1,y1,upper1,nx,ny,0.f,ue,uv_v(upper1)));
                            support.vertices.push_back(vertex(x1,y1,lower1,nx,ny,0.f,ue,uv_v(lower1)));
                            support.vertices.push_back(vertex(x0,y0,lower0,nx,ny,0.f,us,uv_v(lower0)));
                            unsigned triangles[]={base,base+1u,base+2u,base,base+2u,base+3u};
                            support.indices.insert(support.indices.end(),std::begin(triangles),std::end(triangles));
                        }
                    }
                }
                float module_x=composition->foundation_step[0]>0?
                    composition->foundation_step[0]:std::max(x_high-x_low,y_high-y_low);
                unsigned across=std::clamp(unsigned(std::ceil((x_high-x_low)/module_x)),1u,64u);
                unsigned deep=std::clamp(unsigned(std::ceil((y_high-y_low)/module_x)),1u,64u);
                for(unsigned row=0;row<deep;row++)for(unsigned column=0;column<across;column++){
                    float xa=x_low+float(column)*module_x,xb=std::min(xa+module_x,x_high);
                    float ya=y_low+float(row)*module_x,yb=std::min(ya+module_x,y_high);
                    float ue=u0+(u1-u0)*(xb-xa)/module_x;
                    float ve=v0+(v1-v0)*(yb-ya)/module_x;
                    unsigned base=unsigned(support.vertices.size());
                    support.vertices.push_back(vertex(xa,ya,top,0.f,0.f,1.f,u0,v0));
                    support.vertices.push_back(vertex(xb,ya,top,0.f,0.f,1.f,ue,v0));
                    support.vertices.push_back(vertex(xb,yb,top,0.f,0.f,1.f,ue,ve));
                    support.vertices.push_back(vertex(xa,yb,top,0.f,0.f,1.f,u0,ve));
                    unsigned triangles[]={base,base+1u,base+2u,base,base+2u,base+3u};
                    support.indices.insert(support.indices.end(),std::begin(triangles),std::end(triangles));
                }
                if(!indexed){
                    std::vector<fidelity::MapVertex> expanded;
                    expanded.reserve(support.indices.size());
                    for(unsigned index:support.indices)expanded.push_back(support.vertices[index]);
                    support.vertices=std::move(expanded);
                    support.indices.clear();
                }
                output.chunks.push_back(std::move(support));
            }
        }
        for(auto const&p:m.parts){
            auto const&material=library.materials[p.material];Chunk chunk;
            chunk.source_model=i.model;chunk.source_part=unsigned(&p-m.parts.data());chunk.placement=placement;
            chunk.terrain_conforming=material.ground!=0;
            chunk.material=p.material;chunk.environment=composition->environment!=0;chunk.lighting=lighting;
            std::vector<fidelity::MapVertex> transformed; // native cache vertex, all auxiliary channels retained
            transformed.reserve(p.vertices.size());
            for(auto const&v:p.vertices){
                if((unsigned(&v-p.vertices.data())&255u)==0 && stop())return false;
                float world[3],n[3],tangent[3],bitangent[3];placement.position(v.position,world);
                if(material.ground)world[2]=(height_natural(world[0],world[1])+.005f)/112;
                auto out=project_natural(world[0],world[1],world[2]*112);
                placement.source_direction(v.normal,n);placement.source_direction(v.tangent,tangent);placement.source_direction(v.bitangent,bitangent);
                out.u=v.uv0[0];out.v=v.uv0[1];out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];
                out.macro_u=v.uv1[0];out.macro_v=v.uv1[1];out.relief_owner_u=v.uv2[0];out.relief_owner_v=v.uv2[1];
                out.material_grass=tangent[0];out.material_plains=tangent[1];out.material_desert=tangent[2];
                out.material_marsh=bitangent[0];out.authored_relief_height=bitangent[1];out.authored_relief_blend=bitangent[2];
                out.base_terrain=material.ground?60.f:100.f+float(material.channels);transformed.push_back(out);
            }
            if(indexed){chunk.vertices=std::move(transformed);chunk.indices.assign(p.indices.begin(),p.indices.end());}
            else {chunk.vertices.reserve(p.indices.size());for(unsigned vertex_index:p.indices)chunk.vertices.push_back(transformed[vertex_index]);}
            output.chunks.push_back(std::move(chunk));
        }
    }
    if(stop())return false;
    auto const&p=composition->paving;
    if(!p.vertices.empty()){
        Chunk chunk;chunk.material=p.material;chunk.lighting=lighting;chunk.terrain_conforming=true;
        std::copy(p.atlas,p.atlas+4,chunk.atlas);
        std::vector<fidelity::MapVertex> transformed;transformed.reserve(p.vertices.size());
        // A paving material with source normal detail uses a distinct
        // shader tag. The coverage still controls its feathered edge.
        float const tag=(library.materials[p.material].channels&2u)?64.f:62.f;
        auto pave=[&](float x,float y,float coverage){
            auto out=project_natural(x,y,height_natural(x,y)+.005f);
            out.u=x/p.period[0];out.v=y/p.period[1];
            out.base_terrain=tag+coverage*site.ground(x,y);
            transformed.push_back(out);
        };
        for(auto const&v:p.vertices)pave(float(nc)+.5f+v.x,float(nr)+.5f+v.y,v.coverage);
        std::vector<unsigned> indices(p.indices.begin(),p.indices.end());
        // A moved body brings its own plate, edged like the authored one
        // (margin .04, feather .016 and jitter .012 tiles on a .02 grid).
        for(auto const& body:bodies)if(body.moved){
            constexpr float margin=.04f,feather=.016f,jitter=.012f,cell=.02f,reach=margin+feather+jitter;
            auto const& i=composition->instances[body.index];
            std::array<float,4> b{body.offset[0]+i.bounds[0],body.offset[1]+i.bounds[1],
                                  body.offset[0]+i.bounds[2],body.offset[1]+i.bounds[3]};
            unsigned across=unsigned(std::ceil((b[2]-b[0]+2*reach)/cell))+1,deep=unsigned(std::ceil((b[3]-b[1]+2*reach)/cell))+1;
            unsigned first=unsigned(transformed.size());
            std::vector<float> alpha(across*deep);
            for(unsigned gy=0;gy<deep;++gy)for(unsigned gx=0;gx<across;++gx){
                float ox=b[0]-reach+float(gx)*cell,oy=b[1]-reach+float(gy)*cell;
                unsigned h=(gx/3u*73856093u)^(gy/3u*19349663u)^(body.index*83492791u);
                h=(h^(h>>13))*0x5bd1e995u;h^=h>>15;
                float t=std::clamp((margin-site_box_distance(b,ox,oy)-(float(h&1023u)/1023.f-.5f)*2.f*jitter)/feather,0.f,1.f);
                alpha[gy*across+gx]=t*t*(3.f-2.f*t);
                pave(float(nc)+.5f+ox,float(nr)+.5f-oy,alpha[gy*across+gx]);
            }
            for(unsigned gy=0;gy+1<deep;++gy)for(unsigned gx=0;gx+1<across;++gx){
                unsigned k=gy*across+gx;
                if(std::max({alpha[k],alpha[k+1],alpha[k+across],alpha[k+across+1]})<=0)continue;
                unsigned quad[]={k,k+1,k+across,k+1,k+across+1,k+across};
                for(unsigned q:quad)indices.push_back(first+q);
            }
        }
        if(indexed){chunk.vertices=std::move(transformed);chunk.indices=std::move(indices);}
        else for(unsigned vertex_index:indices)chunk.vertices.push_back(transformed[vertex_index]);
        // Paving precedes the source ground and bodies; depth is read-only.
        output.chunks.insert(output.chunks.begin(),std::move(chunk));
    }
    append_effects(library,anchors,lighting,project_natural,indexed,output);
    return !stop();
}
}}
