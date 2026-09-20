#pragma once
#include <d3d11_1.h>
#include <cstring>
namespace c3x_renderer { namespace render_core {
// Selected occurrence constants, uploaded together without changing shader math.
// One 64 KiB allocation; append into unused ranges when supported, DISCARD
// only on wrap (or on older drivers). Earlier submitted ranges remain intact. Offsets
// and sizes obey D3D11.1's 16-constant (256-byte) range alignment.
struct DrawParameterStream {
    static constexpr unsigned stride=256,limit=256;
    ID3D11DeviceContext1* context=nullptr;
    ID3D11Buffer* buffer=nullptr;
    bool checked=false,no_overwrite=false;
    unsigned cursor=limit,base=0,discards=0;
    unsigned uploads=0,records=0;
    DrawParameterStream()=default;DrawParameterStream(DrawParameterStream const&)=delete;
    ~DrawParameterStream(){clear();}
    void clear(){if(buffer)buffer->Release();if(context)context->Release();buffer=nullptr;context=nullptr;checked=no_overwrite=false;uploads=records=discards=base=0;cursor=limit;}
    bool available(ID3D11Device* device,ID3D11DeviceContext* immediate){
        if(checked)return buffer!=nullptr;
        checked=true;D3D11_FEATURE_DATA_D3D11_OPTIONS options={};
        if(FAILED(device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS,&options,sizeof(options))) || !options.ConstantBufferOffsetting)return false;
        if(FAILED(immediate->QueryInterface(__uuidof(ID3D11DeviceContext1),reinterpret_cast<void**>(&context))))return false;
        no_overwrite=options.MapNoOverwriteOnDynamicConstantBuffer!=FALSE;
        D3D11_BUFFER_DESC desc={};desc.ByteWidth=stride*limit;desc.Usage=D3D11_USAGE_DYNAMIC;
        desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;desc.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
        return SUCCEEDED(device->CreateBuffer(&desc,nullptr,&buffer));
    }
    template<class Parameters> bool upload(Parameters const* values,unsigned count){
        static_assert(sizeof(Parameters)<=stride,"draw constant range");
        if(!buffer || !count || count>limit)return false;
        D3D11_MAPPED_SUBRESOURCE mapped={};
        bool discard=!no_overwrite || count>limit-cursor;
        unsigned offset=discard?0:cursor;
        if(FAILED(context->Map(buffer,0,discard?D3D11_MAP_WRITE_DISCARD:D3D11_MAP_WRITE_NO_OVERWRITE,0,&mapped)))return false;
        char* destination=static_cast<char*>(mapped.pData)+offset*stride;
        // Clear range padding as well: shaders never inherit adjacent records.
        std::memset(destination,0,count*stride);
        for(unsigned i=0;i<count;++i)std::memcpy(destination+i*stride,values+i,sizeof(Parameters));
        context->Unmap(buffer,0);base=offset;cursor=offset+count;
        ++uploads;records+=count;discards+=unsigned(discard);return true;
    }
    void bind(unsigned slot,unsigned record){
        UINT first=(base+record)*(stride/16),count=stride/16;
        context->VSSetConstantBuffers1(slot,1,&buffer,&first,&count);
    }
};
}}
