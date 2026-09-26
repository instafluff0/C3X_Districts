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
        float center_x=float(nc)+.5f+i.offset[0],center_y=float(nr)+.5f-i.offset[1];
        float ground_center=height_natural(center_x,center_y);
        float x_low=center_x+i.bounds[0],x_high=center_x+i.bounds[2];
        float y_low=center_y-i.bounds[3],y_high=center_y-i.bounds[1];
        float corners[4][2]={{x_low,y_low},{x_high,y_low},
                              {x_high,y_high},{x_low,y_high}};
        float corner_ground[4]={};
        float ground_low=ground_center,ground_high=ground_center;
        for(unsigned corner=0;corner<4;corner++){
            corner_ground[corner]=height_natural(corners[corner][0],corners[corner][1]);
            ground_low=std::min(ground_low,corner_ground[corner]);
            ground_high=std::max(ground_high,corner_ground[corner]);
        }
        bool terrace=composition->authority.rfind("lab-fixed-",0)==0 &&
            ground_high-ground_low>.75f;
        auto placement=place(i,float(nc)+.5f,float(nr)+.5f,
                             terrace?ground_high+.02f:ground_center);
        for(auto const&l:i.lights)lighting->lights.push_back(placement.light(l,owner_index));
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
                Chunk support;support.material=foundation_material;
                support.environment=composition->environment!=0;
                support.lighting=lighting;
                float top=placement.z*112.f+.006f;
                auto vertex=[&](float x,float y,float z,float nx,float ny,float nz){
                    auto out=project_natural(x,y,z);
                    out.u=foundation_u;out.v=foundation_v;
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
                    unsigned base=unsigned(support.vertices.size());
                    support.vertices.push_back(vertex(corners[side][0],corners[side][1],top,nx,ny,0.f));
                    support.vertices.push_back(vertex(corners[next][0],corners[next][1],top,nx,ny,0.f));
                    support.vertices.push_back(vertex(corners[next][0],corners[next][1],corner_ground[next]-.02f,nx,ny,0.f));
                    support.vertices.push_back(vertex(corners[side][0],corners[side][1],corner_ground[side]-.02f,nx,ny,0.f));
                    unsigned triangles[]={base,base+1u,base+2u,base,base+2u,base+3u};
                    support.indices.insert(support.indices.end(),std::begin(triangles),std::end(triangles));
                }
                unsigned base=unsigned(support.vertices.size());
                for(auto const&corner:corners)
                    support.vertices.push_back(vertex(corner[0],corner[1],top,0.f,0.f,1.f));
                unsigned top_triangles[]={base,base+1u,base+2u,base,base+2u,base+3u};
                support.indices.insert(support.indices.end(),std::begin(top_triangles),std::end(top_triangles));
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
