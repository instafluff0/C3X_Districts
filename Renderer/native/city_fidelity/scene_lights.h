#pragma once
#include "light_spatial_index.h"
#include <wrl/client.h>
#include <cstdio>
namespace c3x_renderer { namespace city_fidelity {
// Complete serialized content is the cache identity, including order and global
// owner indices. No raw pointers survive upload or participate in cache reuse.
struct SceneLights {
    Microsoft::WRL::ComPtr<ID3D11Buffer> frame,data;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> view;
    unsigned capacity=0,cached_lights=0,cached_blockers=0;
    std::vector<LightSpatialIndex::Record> cached;
    LightSpatialIndex spatial;
    float envelope[8]={};
    bool options_read=false,full_scan=false,diagnostics=false,indexed=false;
    unsigned builds=0,uploads=0,reuses=0;
    void bind(ID3D11DeviceContext* context){
        auto* cb=frame.Get();auto* srv=view.Get();
        context->PSSetConstantBuffers(6,1,&cb);context->PSSetShaderResources(127,1,&srv);
    }
    bool ensure(ID3D11Device* device,unsigned count){
        if(count<=capacity)return true;
        unsigned next=256;while(next<count)next*=2;
        D3D11_BUFFER_DESC d={};d.ByteWidth=next*16;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        d.Usage=D3D11_USAGE_DYNAMIC;d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
        d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride=16;
        Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
        if(FAILED(device->CreateBuffer(&d,nullptr,&buffer)))return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC s={};s.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;s.Buffer.NumElements=next;
        if(FAILED(device->CreateShaderResourceView(buffer.Get(),&s,&srv)))return false;
        data=buffer;view=srv;capacity=next;return true;
    }
    bool upload(ID3D11DeviceContext* context,std::vector<Lighting const*>const& cities,float night,float emission){
        if(!options_read){
            char value[8]={};
            full_scan=GetEnvironmentVariableA("C3X_RENDERER_CITY_LIGHT_FULL_SCAN",value,sizeof(value)) && value[0]=='1';
            diagnostics=GetEnvironmentVariableA("C3X_RENDERER_CITY_LIGHT_DIAGNOSTICS",value,sizeof(value)) && value[0]=='1';
            options_read=true;
        }
        std::size_t nl=0,nb=0;
        if(night>0)for(auto city:cities){nl+=city->lights.size();nb+=city->blockers.size();}
        if(nl*3+nb*2>LightSpatialIndex::record_limit)return false;
        Microsoft::WRL::ComPtr<ID3D11Device> device;context->GetDevice(&device);
        if(!frame){D3D11_BUFFER_DESC d={};d.ByteWidth=80;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(device->CreateBuffer(&d,nullptr,&frame)))return false;}
        float constants[20]={float(nl),float(nb),night,emission,1e9f,1e9f,1e9f,0,-1e9f,-1e9f,-1e9f,0};
        if(nl || nb){
            std::vector<LightSpatialIndex::Record> field;
            try{field.resize(nl*3+nb*2);}catch(std::bad_alloc const&){return false;}
            unsigned li=0,bi=0;
            for(auto city:cities){
                for(auto const& l:city->lights){
                    std::memcpy(field.data()+li*3,&l,sizeof(l));
                    field[li*3+2][3]=float(l.owner+bi);++li;
                }
                for(auto const& b:city->blockers){
                    std::memcpy(field.data()+nl*3+bi*2,&b,sizeof(b));++bi;
                }
            }
            if(nl!=cached_lights || nb!=cached_blockers || field.size()!=cached.size() || std::memcmp(field.data(),cached.data(),field.size()*16)!=0){
                LARGE_INTEGER begin={},built={},end={},frequency={};
                if(diagnostics){QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&begin);}
                cached.clear();spatial={};indexed=false;
                if(!full_scan)try{indexed=spatial.build(field,unsigned(nl),unsigned(nb));}
                    catch(std::bad_alloc const&){spatial={};}
                if(!indexed)spatial={};
                if(diagnostics)QueryPerformanceCounter(&built);
                unsigned count=unsigned(field.size()+spatial.records.size());
                if(!ensure(device.Get(),count)){
                    spatial={};indexed=false;count=unsigned(field.size());
                    if(!ensure(device.Get(),count))return false;
                }
                D3D11_MAPPED_SUBRESOURCE mapped={};
                if(FAILED(context->Map(data.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped)))return false;
                std::memcpy(mapped.pData,field.data(),field.size()*16);
                if(indexed)std::memcpy(static_cast<char*>(mapped.pData)+field.size()*16,spatial.records.data(),spatial.records.size()*16);
                context->Unmap(data.Get(),0);
                for(unsigned a=0;a<3;++a){envelope[a]=1e9f;envelope[4+a]=-1e9f;}
                for(unsigned i=0;i<nl;++i)for(unsigned a=0;a<3;++a){
                    envelope[a]=std::min(envelope[a],field[i*3][a]-field[i*3][3]);
                    envelope[4+a]=std::max(envelope[4+a],field[i*3][a]+field[i*3][3]);
                }
                cached=std::move(field);cached_lights=unsigned(nl);cached_blockers=unsigned(nb);
                if(diagnostics){
                    ++builds;++uploads;QueryPerformanceCounter(&end);
                    std::printf("CITY_LIGHT_INDEX lights=%zu blockers=%zu indexed=%u cells=%u light_entries=%u blocker_entries=%u max_lights=%u max_blockers=%u field_bytes=%zu index_bytes=%zu allocation_bytes=%u build_ms=%.3f upload_ms=%.3f builds=%u uploads=%u reuses=%u\n",
                        nl,nb,unsigned(indexed),spatial.cells,spatial.light_entries,spatial.blocker_entries,spatial.max_lights,spatial.max_blockers,
                        cached.size()*16,spatial.records.size()*16,capacity*16,
                        1000.*double(built.QuadPart-begin.QuadPart)/double(frequency.QuadPart),
                        1000.*double(end.QuadPart-built.QuadPart)/double(frequency.QuadPart),builds,uploads,reuses);
                }
            }else if(diagnostics){
                ++reuses;
                if(reuses==1 || reuses%128==0)std::printf("CITY_LIGHT_REUSE reuses=%u uploads=%u constants_bytes=80\n",reuses,uploads);
            }
            std::copy(envelope,envelope+8,constants+4);
            if(indexed){std::copy(spatial.grid,spatial.grid+4,constants+12);std::copy(spatial.info,spatial.info+4,constants+16);}
        }
        context->UpdateSubresource(frame.Get(),0,nullptr,constants,0,0);bind(context);return true;
    }
};
} }
