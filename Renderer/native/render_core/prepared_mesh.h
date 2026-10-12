#pragma once
#include "../../lab/shared/natural/vertex.h"
#include "projected_mesh_bounds.h"
#include "integrated_vertex.h"
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
    // Integrated (168-byte) elements elided from every vertex, and the full
    // constant row they are read from (render_core/integrated_vertex.h).
    // template_offset is that row's GPU placement, set by whoever uploads it.
    std::uint32_t elided=0;
    std::array<std::uint8_t,integrated_vertex_bytes> constants{};
    unsigned template_offset=0;
    bool empty()const{return vertices.empty();}
    std::size_t bytes()const{return vertices.capacity()+indices.capacity();}
};
struct MeshFormat {
    bool pickup=true,feature=false,natural=false;
    unsigned projection_kind=0,source_tile_width=128,shared_grid=0;
    bool city=false;
    // Natural vertices drawn only through world reprojection
    // (natural_projection.z>0) may drop their position, uv and owner.
    bool world_projected=false;
    // Integrated vertices drop elements constant across the mesh (exact).
    bool elide_constants=false;
    // Stage 5 step 2: world-projected natural vertices store 16-bit
    // normals, materials, relief and owner; integrated vertices that elide
    // constants store half vector elements (integrated_quantized).
    bool quantized=false;
};
// Exact compact natural vertex: world xyz+valid, normal, four material
// weights, relief height/blend, base terrain. Valid only for a world-projected
// mesh whose uv equals world xy and whose relief owner is zero, bit for bit;
// the input layouts alias position and uv to world and read the owner from
// a stride-0 zero stream. Any other natural mesh keeps all 92 bytes.
constexpr unsigned natural_vertex_stride=92u,natural_compact_stride=56u;
inline bool natural_vertex_compact(Vertex const& v){
    float zero[4]={},owner[]={v.relief_owner_u,v.relief_owner_v,v.relief_owner_coverage,v.relief_owner_state};
    return std::memcmp(&v.u,&v.world_x,4)==0 && std::memcmp(&v.v,&v.world_y,4)==0 && std::memcmp(owner,zero,16)==0;
}
// Quantized natural vertices (stage 5 step 2, approved October 11): the
// normal is SNORM16x4, the four material channels, relief height/blend and
// the owner are half floats, all expanded by the input assembler, so no
// shader changes. World xyz+valid, uv, position and base terrain stay 32-bit:
// base terrain is a class code plus coverage (42 + coverage needs a float).
// Byte offsets per natural stride; compact formats alias position and uv
// to world and read a zero owner from slot 1.
constexpr unsigned natural_quantized_stride=68u,natural_quantized_compact_stride=40u;
struct NaturalOffsets {unsigned position,world,normal,uv,material,relief,base,owner;bool zero_owner,quantized;};
inline bool natural_stride(unsigned stride){
    return stride==natural_vertex_stride || stride==natural_compact_stride ||
        stride==natural_quantized_stride || stride==natural_quantized_compact_stride;
}
// 0: 92 bytes, 1: 56, 2: 68, 3: 40.
inline unsigned natural_format_index(unsigned stride){
    return stride==natural_compact_stride?1u:stride==natural_quantized_stride?2u:stride==natural_quantized_compact_stride?3u:0u;
}
inline NaturalOffsets natural_offsets(unsigned stride){
    if(stride==natural_compact_stride)return {0,0,16,0,28,44,52,0,true,false};
    if(stride==natural_quantized_stride)return {0,12,28,36,44,52,56,60,false,true};
    if(stride==natural_quantized_compact_stride)return {0,0,16,0,24,32,36,0,true,true};
    return {0,12,28,40,48,64,72,76,false,false};
}
// IEEE half, round to nearest even (the input assembler's R16_FLOAT).
inline std::uint16_t half_bits(float value){
    std::uint32_t x;std::memcpy(&x,&value,4);
    std::uint32_t sign=(x>>16)&0x8000u;x&=0x7fffffffu;
    if(x>=0x47800000u)return std::uint16_t(sign|(x>0x7f800000u?0x7e00u:0x7c00u));
    if(x<0x38800000u){float f;std::memcpy(&f,&x,4);f+=.5f;std::uint32_t r;std::memcpy(&r,&f,4);return std::uint16_t(sign|(r-0x3f000000u));}
    x+=0xfffu+((x>>13)&1u)-(112u<<23);
    return std::uint16_t(sign|(x>>13));
}
inline std::int16_t snorm16_bits(float value){
    return std::int16_t(std::lround(double(std::clamp(value,-1.f,1.f))*32767.));
}
// A vertex quantizes when every 16-bit value is finite and in range.
inline bool natural_vertex_quantizable(Vertex const& v){
    float unit[]={v.normal_x,v.normal_y,v.normal_z};
    for(float n:unit)if(!(n>=-1.f && n<=1.f))return false;
    float halves[]={v.material_grass,v.material_plains,v.material_desert,v.material_marsh,
        v.authored_relief_height,v.authored_relief_blend,v.relief_owner_u,v.relief_owner_v,
        v.relief_owner_coverage,v.relief_owner_state};
    for(float h:halves)if(!(std::fabs(h)<=65504.f))return false;
    return true;
}
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
    bool compact=format.pickup && !format.city && format.natural && format.world_projected;
    for(std::size_t i=0;compact && i<count;++i)compact=natural_vertex_compact(source[topology?i:unique[i]]);
    bool quantize=format.pickup && !format.city && format.natural && format.world_projected && format.quantized;
    for(std::size_t i=0;quantize && i<count;++i)quantize=natural_vertex_quantizable(source[topology?i:unique[i]]);
    result.vertex_stride=format.city?88u:format.pickup?(format.natural?(quantize?(compact?natural_quantized_compact_stride:natural_quantized_stride):
        compact?natural_compact_stride:natural_vertex_stride):format.feature?48u:sizeof(Vertex)):120u;
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
        }else if(quantize){
            auto* out=static_cast<unsigned char*>(output);unsigned at=0;
            auto put=[&](void const* data,unsigned bytes){std::memcpy(out+at,data,bytes);at+=bytes;};
            if(!compact){float position[]={x,y,z};put(position,12);}
            float world_valid[]={v.world_x,v.world_y,v.world_z,v.world_valid};put(world_valid,16);
            std::int16_t normal[]={snorm16_bits(v.normal_x),snorm16_bits(v.normal_y),snorm16_bits(v.normal_z),0};put(normal,8);
            if(!compact){float uv[]={v.u,v.v};put(uv,8);}
            std::uint16_t material[]={half_bits(v.material_grass),half_bits(v.material_plains),half_bits(v.material_desert),
                half_bits(v.material_marsh),half_bits(v.authored_relief_height),half_bits(v.authored_relief_blend)};put(material,12);
            put(&v.base_terrain,4);
            if(!compact){std::uint16_t owner[]={half_bits(v.relief_owner_u),half_bits(v.relief_owner_v),
                half_bits(v.relief_owner_coverage),half_bits(v.relief_owner_state)};put(owner,8);}
        }else if(compact){
            float data[]={v.world_x,v.world_y,v.world_z,v.world_valid,v.normal_x,v.normal_y,v.normal_z,
                v.material_grass,v.material_plains,v.material_desert,v.material_marsh,
                v.authored_relief_height,v.authored_relief_blend,v.base_terrain};
            static_assert(sizeof(data)==natural_compact_stride,"compact natural vertex");
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
    if(format.elide_constants && format.pickup && !format.city && !format.natural && !format.feature &&
       result.vertex_stride==integrated_vertex_bytes && count){
        auto const* rows=result.vertices.data();
        std::uint32_t elided=integrated_elidable;
        for(std::size_t i=1;i<count && elided;++i)for(unsigned e=1;e<integrated_element_count;++e){
            auto const& d=integrated_elements[e];
            if(((elided>>e)&1u) && std::memcmp(rows+d.offset,rows+i*integrated_vertex_bytes+d.offset,d.bytes)!=0)elided&=~(1u<<e);
        }
        // Stored vector elements become half floats when every value fits.
        bool halves=format.quantized;
        for(std::size_t i=0;i<count && halves;++i)for(unsigned e=1;e<integrated_element_count && halves;++e){
            if(integrated_elided(elided,e) || integrated_half_bytes[e]==integrated_elements[e].bytes)continue;
            auto const& d=integrated_elements[e];
            for(unsigned c=0;c<d.bytes/4 && halves;++c){float value;std::memcpy(&value,rows+i*integrated_vertex_bytes+d.offset+c*4,4);
                halves=std::fabs(value)<=65504.f;}
        }
        if(halves)elided|=integrated_quantized;
        if(elided){
            // Rows compact in place: every destination precedes its source.
            std::memcpy(result.constants.data(),rows,integrated_vertex_bytes);
            unsigned stride=integrated_stride(elided);auto* bytes=result.vertices.data();
            for(std::size_t i=0;i<count;++i){
                if((i&1023u)==0 && stop())return false;
                unsigned packed=0;
                for(unsigned e=0;e<integrated_element_count;++e){if(integrated_elided(elided,e))continue;
                    integrated_store(elided,e,bytes+i*integrated_vertex_bytes,bytes+i*stride+packed,half_bits);
                    packed+=integrated_stored_bytes(elided,e);}
            }
            result.vertices.resize(count*stride);result.vertices.shrink_to_fit();
            result.vertex_stride=stride;result.elided=elided;
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
