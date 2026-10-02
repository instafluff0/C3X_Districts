#pragma once
#include "data.h"
#include "ground.h"
#include "../../../native/render_core/prepared_mesh.h"
namespace c3x_renderer { namespace fidelity {
// Draw the authored rock patches on the exact receiver triangles. Independent
// rotated grids intersect steep hills and give the patches unrelated normals.
// UV clipping in the material keeps the footprint without cutting the mesh.
template<class Emit>
inline bool emit_hill_decal_triangles(Tile owner,int column,int row,
        std::vector<MapVertex> const& surface,std::vector<unsigned> const* indices,
        Emit emit) {
    Hill hill=composed_hill(owner);std::uint32_t state=hill.seed;
    constexpr unsigned cells[]={0,0,0,0,1,1,1,2,2,2};
    for(unsigned ordinal=0;ordinal<10;++ordinal){
        float keep=random01(state),angle=random01(state)*6.283185307f;
        float radius=std::sqrt(random01(state))*.22f,phase=random01(state)*6.283185307f;
        float scale=.90f+.20f*random01(state);
        if(keep>hill.rockiness)continue;
        float cu=.5f+std::cos(phase)*radius,cv=.5f+std::sin(phase)*radius;
        float co=std::cos(angle),si=std::sin(angle);
        auto project=[&](MapVertex v){
            float u=v.world_x-column-cu,w=float(row)+1-v.world_y-cv;
            v.u=(co*u+si*w)/(.42f*scale)+.5f;
            v.v=(-si*u+co*w)/(.37f*scale)+.5f;
            v.material_grass=std::max(0.f,(v.world_z*112-2.5f)/112);
            v.material_plains=2;v.material_desert=float(cells[ordinal]);
            if(v.base_terrain>=42)v.base_terrain=-10+std::clamp(v.base_terrain-42,0.f,1.f);
            return v;
        };
        auto count=indices?indices->size():surface.size();
        for(std::size_t i=0;i+2<count;i+=3){
            MapVertex v[3];
            for(unsigned j=0;j<3;++j)v[j]=project(surface[indices?(*indices)[i+j]:i+j]);
            if(std::max({v[0].u,v[1].u,v[2].u})<0 || std::min({v[0].u,v[1].u,v[2].u})>1 ||
               std::max({v[0].v,v[1].v,v[2].v})<0 || std::min({v[0].v,v[1].v,v[2].v})>1)continue;
            if(!emit(v))return false;
        }
    }
    return true;
}

// The actual full triangle corner identity and first-reference ordering match
// PreparedMesh's original expanded-input hash/equality contract. The flat
// table, outputs and caller's other live arrays share the existing8MiB gate.
class HillDecalOutput {
    std::vector<MapVertex>& vertices;
    std::vector<unsigned>& elements;
    std::vector<unsigned> slots;
    std::size_t maximum=0;
    bool failed=false;
    bool reject(){failed=true;return false;}
public:
    HillDecalOutput(std::vector<MapVertex>& v,std::vector<unsigned>& e):vertices(v),elements(e){}
    std::size_t scratch_bytes()const{return slots.capacity()*sizeof(unsigned);}
    bool rejected()const{return failed;}
    void release_scratch(){std::vector<unsigned>().swap(slots);}
    template<class Admit> bool initialize(std::size_t corners,Admit admit){
        constexpr std::size_t limit=8u*1024u*1024u;
        if(!slots.empty() || !vertices.empty() || !elements.empty() || corners>limit/8)return reject();
        std::size_t capacity=1;while(capacity<corners*2)capacity*=2;
        if(!admit(capacity*sizeof(unsigned)))return reject();
        slots.assign(capacity,~0u);maximum=corners;return admit(0) || reject();
    }
    template<class Admit> bool append(MapVertex const* triangle,Admit admit){
        constexpr std::size_t limit=8u*1024u*1024u;
        if(failed || slots.empty() || maximum<3 || elements.size()>maximum-3 || !admit(0))return reject();
        render_core::VertexHash hash{sizeof(MapVertex),false};
        render_core::VertexEqual equal{sizeof(MapVertex),false};
        for(unsigned i=0;i<3;++i){
            auto slot=hash(triangle[i])&(slots.size()-1);
            while(slots[slot]!=~0u && !equal(vertices[slots[slot]],triangle[i]))slot=(slot+1)&(slots.size()-1);
            bool added=slots[slot]==~0u;
            auto v=vertices.capacity(),e=elements.capacity();
            if(added && vertices.size()==v)v=v?v*2:1;
            if(elements.size()==e)e=e?e*2:3;
            if(v>limit/sizeof(MapVertex) || e>limit/sizeof(unsigned))return reject();
            // reserve temporarily owns both old and new arrays. Charge the
            // entire replacement allocation; the prior allocation retires
            // before the second array grows.
            if(v>vertices.capacity()){
                if(!admit(v*sizeof(MapVertex)))return reject();
                vertices.reserve(v);
                if(!admit(0))return reject();
            }
            if(e>elements.capacity()){
                if(!admit(e*sizeof(unsigned)))return reject();
                elements.reserve(e);
            }
            if(!admit(0))return reject();
            if(added){slots[slot]=unsigned(vertices.size());vertices.push_back(triangle[i]);}
            elements.push_back(slots[slot]);
        }
        return true;
    }
};


inline void emit_hill_decals(Tile owner,int column,int row,
        std::vector<MapVertex> const& surface,std::vector<unsigned> const* indices,
        std::vector<MapVertex>& decals){
    emit_hill_decal_triangles(owner,column,row,surface,indices,[&](MapVertex const* v){
        decals.insert(decals.end(),v,v+3);return true;
    });
}
}}
