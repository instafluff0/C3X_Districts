#pragma once
#include "../../lab/shared/natural/vertex.h"
#include "projected_mesh_bounds.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <limits>
#include <algorithm>

namespace c3x_renderer { namespace render_core {
using Vertex=c3x_renderer::fidelity::MapVertex;
struct VertexHash {
    std::size_t stride = sizeof(Vertex);
    bool feature = false;
    std::size_t operator()(Vertex const & vertex) const {
        auto bytes = reinterpret_cast<unsigned char const *>(&vertex);
        float fields[12];
        if(feature){float values[]={vertex.x,vertex.y,vertex.z,vertex.u,vertex.v,
            vertex.normal_x,vertex.normal_y,vertex.normal_z,vertex.base_terrain,
            vertex.world_x,vertex.world_y,vertex.world_z};
            std::memcpy(fields,values,sizeof(fields));bytes=reinterpret_cast<unsigned char const*>(fields);}
        std::uint64_t result = 1469598103934665603ull;
        for (std::size_t i = 0; i < (feature?sizeof(fields):stride); i += sizeof(std::uint32_t)) {
            std::uint32_t word;
            std::memcpy(&word, bytes + i, sizeof(word));
            result = (result ^ word) * 1099511628211ull;
        }
        // Float grids often have identical low mantissa bits. The flat table
        // masks low hash bits, unlike the old prime-bucket unordered_map.
        // Avalanche high coordinate bits before masking to avoid quadratic
        // probing on otherwise ordinary flat terrain. Equality stays exact.
        result ^= result >> 33; result *= 0xff51afd7ed558ccdull;
        result ^= result >> 33; result *= 0xc4ceb9fe1a85ec53ull;
        result ^= result >> 33;
        return static_cast<std::size_t>(result);
    }
};
struct VertexEqual {
    std::size_t stride = sizeof(Vertex);
    bool feature = false;
    bool operator()(Vertex const & a, Vertex const & b) const {
        if(feature)return std::memcmp(&a,&b,20)==0 && std::memcmp(&a.normal_x,&b.normal_x,12)==0 &&
            std::memcmp(&a.base_terrain,&b.base_terrain,4)==0 && std::memcmp(&a.world_x,&b.world_x,12)==0;
        return std::memcmp(&a, &b, stride) == 0;
    }
};

// CPU-owned immutable draw payload. No device, camera ownership, asset pointers
// or game pointers. Geometry generation and GPU adoption meet at this boundary.
struct PreparedMesh {
    std::vector<std::uint8_t> vertices,indices;
    unsigned vertex_stride=0,index_stride=0,index_count=0,shared_grid=0;
    std::array<std::int32_t,4> bounds{};
    std::array<float,3> world_low{},world_high{};
    ProjectedMeshBounds projected_bounds;
    bool empty()const{return vertices.empty();}
    std::size_t bytes()const{return vertices.capacity()+indices.capacity();}
};
struct MeshFormat {
    bool pickup=true,feature=false,natural=false;
    unsigned projection_kind=0,source_tile_width=128,shared_grid=0;
    bool city=false;
};
template<class Layouts>
unsigned shared_mesh_grid(std::size_t vertices,std::vector<unsigned> const* indices,Layouts& layouts){
    if(!indices || vertices>65535u)return 0;
    unsigned side=unsigned(std::sqrt(double(vertices)));
    return side>1 && side<=65 && side*side==vertices && layouts.get(side-1).indices==*indices?side-1:0;
}
template<class Cancelled>
bool prepare_mesh(std::vector<Vertex> const& source,std::vector<unsigned> const* topology,
                  MeshFormat format,PreparedMesh& destination,Cancelled stop){
    PreparedMesh result;
    if(stop())return false;
    if(source.empty() || (topology && topology->empty())){destination=std::move(result);return true;}
    std::vector<unsigned> unique,indices;
    if(!topology){
        std::size_t capacity=1;while(capacity<source.size()*2u)capacity*=2u;
        std::vector<unsigned> slots(capacity,std::numeric_limits<unsigned>::max());
        VertexHash hash{format.pickup?sizeof(Vertex):120u,format.pickup && format.feature};
        VertexEqual equal{format.pickup?sizeof(Vertex):120u,format.pickup && format.feature};
        unique.reserve(source.size()/3u);indices.reserve(source.size());
        for(std::size_t i=0;i<source.size();++i){
            if((i&255u)==0 && stop())return false;
            auto slot=hash(source[i])&(capacity-1);
            while(slots[slot]!=std::numeric_limits<unsigned>::max() && !equal(source[unique[slots[slot]]],source[i]))slot=(slot+1)&(capacity-1);
            if(slots[slot]==std::numeric_limits<unsigned>::max()){slots[slot]=unsigned(unique.size());unique.push_back(unsigned(i));}
            indices.push_back(slots[slot]);
        }
    }
    auto const& elements=topology?*topology:indices;
    auto count=topology?source.size():unique.size();
    result.vertex_stride=format.city?88u:format.pickup?(format.natural?92u:format.feature?48u:sizeof(Vertex)):120u;
    result.index_stride=count<=65535u?2:4;result.index_count=unsigned(elements.size());
    result.shared_grid=format.shared_grid;
    result.vertices.resize(count*result.vertex_stride);result.indices.resize(elements.size()*result.index_stride);
    result.bounds={std::numeric_limits<std::int32_t>::max(),std::numeric_limits<std::int32_t>::max(),
        std::numeric_limits<std::int32_t>::min(),std::numeric_limits<std::int32_t>::min()};
    result.world_low.fill(1e9f);result.world_high.fill(-1e9f);
    for(std::size_t i=0;i<count;++i){
        if((i&255u)==0 && stop())return false;
        auto const& v=source[topology?i:unique[i]];
        if(format.natural)result.projected_bounds.include(v.world_x,v.world_y,v.world_z);
        float world[]={v.world_x,v.world_y,v.world_z};
        if(format.pickup)for(unsigned axis=0;axis<3;++axis){
            result.world_low[axis]=std::min(result.world_low[axis],world[axis]);
            result.world_high[axis]=std::max(result.world_high[axis],world[axis]);
        }
        result.bounds[0]=std::min(result.bounds[0],std::int32_t(std::floor(v.x))-2);
        result.bounds[1]=std::min(result.bounds[1],std::int32_t(std::floor(v.y))-2);
        result.bounds[2]=std::max(result.bounds[2],std::int32_t(std::ceil(v.x))+2);
        result.bounds[3]=std::max(result.bounds[3],std::int32_t(std::ceil(v.y))+2);
        float x=v.x,y=v.y,z=v.z;
        if(format.projection_kind==2 || format.projection_kind==3){
            x/=float(format.source_tile_width);y/=float(format.source_tile_width);
            if(format.projection_kind==3)z/=128.f;
        }
        void* output=result.vertices.data()+i*result.vertex_stride;
        if(format.city){
            // Exact channels consumed by the city color, reflection and caster
            // layouts. No quantization or source detail is lost; terrain-only
            // fields no longer occupy every city vertex.
            float data[]={x,y,z,v.u,v.v,v.normal_x,v.normal_y,v.normal_z,
                v.macro_u,v.macro_v,v.base_terrain,
                v.material_grass,v.material_plains,v.material_desert,
                v.material_marsh,v.authored_relief_height,v.authored_relief_blend,
                v.world_x,v.world_y,v.world_z,v.relief_owner_u,v.relief_owner_v};
            std::memcpy(output,data,sizeof(data));
        }else if(format.natural){
            float data[]={x,y,z,v.world_x,v.world_y,v.world_z,v.world_valid,
                v.normal_x,v.normal_y,v.normal_z,v.u,v.v,
                v.material_grass,v.material_plains,v.material_desert,v.material_marsh,
                v.authored_relief_height,v.authored_relief_blend,v.base_terrain,
                v.relief_owner_u,v.relief_owner_v,v.relief_owner_coverage,v.relief_owner_state};
            std::memcpy(output,data,sizeof(data));
        }else if(format.pickup && format.feature){
            float data[]={x,y,z,v.u,v.v,v.normal_x,v.normal_y,v.normal_z,v.base_terrain,v.world_x,v.world_y,v.world_z};
            std::memcpy(output,data,sizeof(data));
        }else {
            std::memcpy(output,&v,result.vertex_stride);
            if(format.projection_kind==2 || format.projection_kind==3){
                float position[]={x,y,z};std::memcpy(output,position,sizeof(position));
                if(format.projection_kind==3){
                    float branches=v.river_branch_count/128.f;
                    auto offset=reinterpret_cast<char const*>(&v.river_branch_count)-reinterpret_cast<char const*>(&v);
                    std::memcpy(static_cast<char*>(output)+offset,&branches,sizeof(branches));
                }
            }
        }
    }
    for(std::size_t i=0;i<elements.size();++i){
        if((i&1023u)==0 && stop())return false;
        if(elements[i]>=count)return false;
        if(result.index_stride==2){std::uint16_t value=std::uint16_t(elements[i]);std::memcpy(result.indices.data()+i*2,&value,2);}
        else std::memcpy(result.indices.data()+i*4,&elements[i],4);
    }
    destination=std::move(result);return true;
}
}}
