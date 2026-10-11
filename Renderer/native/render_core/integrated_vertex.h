#pragma once
#include <cstdint>
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
inline unsigned integrated_stride(std::uint32_t elided){
    unsigned stride=0;
    for(unsigned e=0;e<integrated_element_count;++e)if(!((elided>>e)&1u))stride+=integrated_elements[e].bytes;
    return stride;
}
// Where `bytes` at byte `at` of a full vertex live: {slot, offset}. Slot 0 is
// the compact vertex, slot 1 the constant row. {0,~0u} when the range is not
// inside one element.
inline std::pair<unsigned,unsigned> integrated_location(std::uint32_t elided,unsigned at,unsigned bytes=4){
    unsigned packed=0;
    for(unsigned e=0;e<integrated_element_count;++e){
        auto const& d=integrated_elements[e];bool gone=((elided>>e)&1u)!=0;
        if(at>=d.offset && at<d.offset+d.bytes){
            if(at+bytes>d.offset+d.bytes)break;
            return gone?std::make_pair(1u,at):std::make_pair(0u,packed+(at-d.offset));
        }
        if(!gone)packed+=d.bytes;
    }
    return {0u,~0u};
}
// Rewrite a full-vertex input layout description for `elided`. Desc is
// D3D11_INPUT_ELEMENT_DESC; `bytes` gives each element's width.
template<class Desc,class Bytes>
bool remap_integrated_layout(Desc* elements,unsigned count,std::uint32_t elided,Bytes bytes){
    for(unsigned i=0;i<count;++i){
        auto at=integrated_location(elided,elements[i].AlignedByteOffset,bytes(elements[i]));
        if(at.second==~0u)return false;
        elements[i].InputSlot=at.first;elements[i].AlignedByteOffset=at.second;
    }
    return true;
}
}}
