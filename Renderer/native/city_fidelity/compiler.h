#pragma once
#include "runtime.h"
#include "../c3x_renderer_api.h"
#include "../../lab/shared/natural/vertex.h"
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
};
struct Surfaces {std::vector<Chunk> chunks;};
template<class World,class Shore,class River,class Height>
Composition const* select(Library const& library,c3x_renderer_tile_v1 const& record,int column,int row,
        World world_lookup,Shore shore_sample_at,River river_distance,Height height_natural){
    if(record.city_id<0)return nullptr;
    unsigned culture=unsigned(std::clamp(record.city_culture_group,0,4));
    unsigned era=unsigned(std::clamp(record.city_era,0,3)),size=unsigned(std::clamp(record.city_size,0,2));
    bool capital=(record.city_flags&C3X_RENDERER_CITY_CAPITAL)!=0;
    // Selected compositions precede generic growth. Do not fit a coastal site
    // by moving vegetation, squashing a body, or searching during a draw.
    for(unsigned pass=0;pass<2;pass++)for(auto const&t:library.compositions){
        if(t.culture!=culture || t.era!=era || t.size!=size ||
            (pass==0?t.capital!=unsigned(capital):t.capital!=0 || !capital))continue;
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
struct ContinueCompilation {bool operator()()const{return false;}};
template<class Height,class Project,class Stop=ContinueCompilation>
bool compile(Library const& library,Composition const& selected,int nc,int nr,
        Height height_natural,Project project_natural,Surfaces& output,Stop stop={},bool indexed=false){
    auto composition=&selected;
    auto lighting=std::make_shared<Lighting>();
    lighting->blockers.reserve(composition->instances.size());
    for(unsigned owner_index=0;owner_index<composition->instances.size();owner_index++){
        if(stop())return false;
        auto const&i=composition->instances[owner_index];auto const&m=library.models[i.model];
        auto placement=place(i,float(nc)+.5f,float(nr)+.5f,
            height_natural(float(nc)+.5f+i.offset[0],float(nr)+.5f-i.offset[1]));
        for(auto const&l:i.lights)lighting->lights.push_back(placement.light(l,owner_index));
        Lighting::Box b={{placement.x+i.bounds[0],-placement.y+i.bounds[1],
            placement.z*source_z_metric+std::max(0.f,m.low[2])*i.scale,0},
            {placement.x+i.bounds[2],-placement.y+i.bounds[3],placement.z*source_z_metric+m.high[2]*i.scale,0}};
        lighting->blockers.push_back(b);
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
        for(auto const&v:p.vertices){
            float x=float(nc)+.5f+v.x,y=float(nr)+.5f+v.y;
            auto out=project_natural(x,y,height_natural(x,y)+.005f);
            out.u=x/p.period[0];out.v=y/p.period[1];out.base_terrain=62+v.coverage;
            transformed.push_back(out);
        }
        if(indexed){chunk.vertices=std::move(transformed);chunk.indices.assign(p.indices.begin(),p.indices.end());}
        else for(unsigned vertex_index:p.indices)chunk.vertices.push_back(transformed[vertex_index]);
        // Paving precedes the source ground and bodies; depth is read-only.
        output.chunks.insert(output.chunks.begin(),std::move(chunk));
    }
    return !stop();
}
}}
