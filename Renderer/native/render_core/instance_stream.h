#pragma once
#include "../../lab/shared/natural/instance.h"
namespace c3x_renderer { namespace render_core {
struct InstanceStream {
    using Instance=fidelity::MeshInstance;
    static constexpr unsigned limit=16384;
    ID3D11Buffer* buffer=nullptr;
    std::size_t bytes=0;unsigned uploads=0,discards=0,cursor=0,offset=0;
    InstanceStream()=default;InstanceStream(InstanceStream const&)=delete;
    ~InstanceStream(){clear();}
    void clear(){if(buffer)buffer->Release();buffer=nullptr;bytes=uploads=discards=cursor=offset=0;}
    bool upload(ID3D11Device*device,ID3D11DeviceContext*context,
            Instance const* data,std::size_t count){
        if(!data || !count || count>limit)return false;
        if(!buffer){D3D11_BUFFER_DESC d={};d.ByteWidth=limit*sizeof(Instance);
            d.Usage=D3D11_USAGE_DYNAMIC;d.BindFlags=D3D11_BIND_VERTEX_BUFFER;d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            bool borrowed=device!=nullptr;if(!borrowed)context->GetDevice(&device);
            HRESULT hr=device->CreateBuffer(&d,nullptr,&buffer);if(!borrowed)device->Release();
            if(FAILED(hr))return false;}
        // Append only to untouched bytes. On wrap, DISCARD gives the driver a
        // fresh allocation while earlier draws retain their submitted contents.
        // The cursor survives frames; resetting counters must not reuse live data.
        unsigned begin=cursor+count>limit?0:cursor;
        D3D11_MAPPED_SUBRESOURCE mapped={};
        if(FAILED(context->Map(buffer,0,begin?D3D11_MAP_WRITE_NO_OVERWRITE:D3D11_MAP_WRITE_DISCARD,0,&mapped)))return false;
        offset=begin*sizeof(Instance);
        std::memcpy(static_cast<char*>(mapped.pData)+offset,data,count*sizeof(Instance));context->Unmap(buffer,0);
        cursor=begin+unsigned(count);if(!begin)++discards;
        bytes+=count*sizeof(Instance);++uploads;return true;
    }
    bool upload(ID3D11Device*device,ID3D11DeviceContext*context,std::vector<Instance> const& data){
        return upload(device,context,data.data(),data.size());
    }
};
inline HRESULT create_instance_layout(ID3D11Device*device,ID3DBlob*code,ID3D11InputLayout**layout){
    D3D11_INPUT_ELEMENT_DESC e[]={
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",1,DXGI_FORMAT_R32G32B32A32_FLOAT,1,0,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"TEXCOORD",2,DXGI_FORMAT_R32G32B32A32_FLOAT,1,16,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"TEXCOORD",3,DXGI_FORMAT_R32G32B32A32_FLOAT,1,32,D3D11_INPUT_PER_INSTANCE_DATA,1},
        {"TEXCOORD",4,DXGI_FORMAT_R32G32B32A32_FLOAT,1,48,D3D11_INPUT_PER_INSTANCE_DATA,1}};
    return device->CreateInputLayout(e,7,code->GetBufferPointer(),code->GetBufferSize(),layout);
}
} }
