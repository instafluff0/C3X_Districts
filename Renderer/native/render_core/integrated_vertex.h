#pragma once
#include <cstdint>
#include <cstring>
#include <utility>

namespace c3x_renderer { namespace render_core {
// The 168-byte integrated ground vertex (MapVertex) as its 19 shader input
// elements. A prepared mesh may elide elements whose bytes are identical on
// every vertex (stage 5, exact): the vertex keeps the others in order, and the
// chunk keeps one full 168-byte constant row beside its vertices. Input
// layouts read an elided element from that row through a stride-0 slot 1;
// the pulled path reads it from the row's copy in the page. Shader inputs are
// bit-identical either way.
struct IntegratedElement {unsigned offset,bytes;};
constexpr unsigned integrated_vertex_bytes=168u,integrated_element_count=19u;
constexpr IntegratedElement integrated_elements[integrated_element_count]={
    {0,12},{12,8},{20,4},{24,12},{36,8},{44,8},{52,4},{56,4},{60,4},{64,4},
    {68,16},{84,8},{92,4},{96,16},{112,4},{116,4},{120,16},{136,16},{152,16}};
// Position (element 0) is always stored, so a stride is never zero.
constexpr std::uint32_t integrated_elidable=((1u<<integrated_element_count)-1u)&~1u;
// The format mask is `elided` plus bit 0 (position is never elided): set
// when the stored vector elements are half floats (stage 5 step 2, approved
// October 11): normal, visibility, material weights, relief, river,
// hydrology and owner/flow. Position, uv, macro uv, world and the one-float
// elements stay 32-bit; elided elements still read the 32-bit constant row.
constexpr std::uint32_t integrated_quantized=1u;
constexpr unsigned integrated_half_bytes[integrated_element_count]={
    12,8,4,8,4,8,4,4,4,4,8,4,4,8,4,4,16,8,8};
inline bool integrated_elided(std::uint32_t mask,unsigned e){return e!=0 && ((mask>>e)&1u)!=0;}
inline bool integrated_half(std::uint32_t mask,unsigned e){
    return (mask&integrated_quantized)!=0 && !integrated_elided(mask,e) && integrated_half_bytes[e]<integrated_elements[e].bytes;
}
inline unsigned integrated_stored_bytes(std::uint32_t mask,unsigned e){
    return integrated_elided(mask,e)?0u:(mask&integrated_quantized)?integrated_half_bytes[e]:integrated_elements[e].bytes;
}
inline unsigned integrated_stride(std::uint32_t mask){
    unsigned stride=0;
    for(unsigned e=0;e<integrated_element_count;++e)stride+=integrated_stored_bytes(mask,e);
    return stride;
}
// Where `bytes` at byte `at` of a full vertex live: {slot, offset}. Slot 0 is
// the compact vertex, slot 1 the constant row. {0,~0u} when the range is not
// inside one element, or is part of a half element (read whole only).
inline std::pair<unsigned,unsigned> integrated_location(std::uint32_t mask,unsigned at,unsigned bytes=4){
    unsigned packed=0;
    for(unsigned e=0;e<integrated_element_count;++e){
        auto const& d=integrated_elements[e];
        if(at>=d.offset && at<d.offset+d.bytes){
            if(at+bytes>d.offset+d.bytes)break;
            if(integrated_elided(mask,e))return {1u,at};
            if(integrated_half(mask,e) && (at!=d.offset || bytes!=d.bytes))break;
            return {0u,packed+(at-d.offset)};
        }
        packed+=integrated_stored_bytes(mask,e);
    }
    return {0u,~0u};
}
// Rewrite a full-vertex input layout description for the format `mask`. Desc
// is D3D11_INPUT_ELEMENT_DESC; `bytes` gives each element's width and `half`
// switches a stored half element's description to its 16-bit format.
template<class Desc,class Bytes,class Half>
bool remap_integrated_layout(Desc* elements,unsigned count,std::uint32_t mask,Bytes bytes,Half half){
    for(unsigned i=0;i<count;++i){
        unsigned at=elements[i].AlignedByteOffset;
        auto location=integrated_location(mask,at,bytes(elements[i]));
        if(location.second==~0u)return false;
        for(unsigned e=0;e<integrated_element_count;++e)
            if(integrated_elements[e].offset==at && location.first==0 && integrated_half(mask,e) && !half(elements[i]))return false;
        elements[i].InputSlot=location.first;elements[i].AlignedByteOffset=location.second;
    }
    return true;
}
template<class Desc,class Bytes>
bool remap_integrated_layout(Desc* elements,unsigned count,std::uint32_t mask,Bytes bytes){
    return remap_integrated_layout(elements,count,mask,bytes,[](Desc&){return false;});
}
// One 32-bit vertex row's element in its stored form (half floats for a
// half element, padded to its stored size).
template<class Half>
void integrated_store(std::uint32_t mask,unsigned e,unsigned char const* full_row,unsigned char* out,Half half_bits){
    auto const& d=integrated_elements[e];
    if(!integrated_half(mask,e)){std::memmove(out,full_row+d.offset,d.bytes);return;}
    std::uint16_t halves[8]={};float values[4]={};std::memcpy(values,full_row+d.offset,d.bytes);
    for(unsigned c=0;c<d.bytes/4;++c)halves[c]=half_bits(values[c]);
    std::memcpy(out,halves,integrated_half_bytes[e]);
}
}}
