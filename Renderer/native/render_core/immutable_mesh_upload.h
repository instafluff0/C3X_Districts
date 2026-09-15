#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace c3x_renderer { namespace render_core {
// One content record owns one immutable allocation. Draws retain separate
// vertex/index ranges and material ordering; no transient GPU arena or fencing.
class ImmutableMeshUpload {
    std::vector<std::uint8_t> data;
public:
    static constexpr std::size_t limit=64u*1024u*1024u;
    std::size_t size()const{return data.size();}
    unsigned append(void const* source,std::size_t bytes){
        auto offset=(data.size()+3u)&~std::size_t(3u);
        if(!source || !bytes || offset>limit || bytes>limit-offset)throw std::length_error("content mesh upload");
        data.resize(offset+bytes);std::memcpy(data.data()+offset,source,bytes);return unsigned(offset);
    }
    template<class Device,class Buffer> bool create(Device* device,Buffer** result)const{
        if(data.empty())return true;
        D3D11_BUFFER_DESC desc={};desc.ByteWidth=unsigned(data.size());
        desc.Usage=D3D11_USAGE_IMMUTABLE;
        desc.BindFlags=D3D11_BIND_VERTEX_BUFFER|D3D11_BIND_INDEX_BUFFER;
        D3D11_SUBRESOURCE_DATA initial={};initial.pSysMem=data.data();
        return SUCCEEDED(device->CreateBuffer(&desc,&initial,result));
    }
};
}}
