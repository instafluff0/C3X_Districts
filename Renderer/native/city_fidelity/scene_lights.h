#pragma once
#include "runtime.h"
#include <wrl/client.h>
namespace c3x_renderer { namespace city_fidelity {
// A scene may contain many more city objects than a regional constant buffer.
// Keep all facade lights and blockers in one growable GPU buffer. Daylight
// submits an empty field, since these lights contribute only at night.
struct SceneLights {
    Microsoft::WRL::ComPtr<ID3D11Buffer> frame,data;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> view;
    unsigned capacity=0;
    void bind(ID3D11DeviceContext* context){
        auto* cb=frame.Get();auto* srv=view.Get();
        context->PSSetConstantBuffers(6,1,&cb);context->PSSetShaderResources(127,1,&srv);
    }
    bool upload(ID3D11DeviceContext* context,std::vector<Lighting const*>const& cities,float night,float emission){
        std::size_t nl=0,nb=0;
        if(night>0)for(auto city:cities){nl+=city->lights.size();nb+=city->blockers.size();}
        // Resource bound, not a per-city limit. No light or blocker truncation.
        if(nl*3+nb*2>4u*1024u*1024u)return false;
        unsigned count=unsigned(nl*3+nb*2);
        Microsoft::WRL::ComPtr<ID3D11Device> device;context->GetDevice(&device);
        if(!frame){D3D11_BUFFER_DESC d={};d.ByteWidth=48;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(device->CreateBuffer(&d,nullptr,&frame)))return false;}
        if(count>capacity){
            unsigned next=256;while(next<count)next*=2;
            D3D11_BUFFER_DESC d={};d.ByteWidth=next*16;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            d.Usage=D3D11_USAGE_DYNAMIC;d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride=16;
            Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
            Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
            if(FAILED(device->CreateBuffer(&d,nullptr,&buffer)))return false;
            D3D11_SHADER_RESOURCE_VIEW_DESC s={};s.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;s.Buffer.NumElements=next;
            if(FAILED(device->CreateShaderResourceView(buffer.Get(),&s,&srv)))return false;
            data=buffer;view=srv;capacity=next;
        }
        float constants[12]={float(nl),float(nb),night,emission,1e9f,1e9f,1e9f,0,-1e9f,-1e9f,-1e9f,0};
        if(count){
            D3D11_MAPPED_SUBRESOURCE mapped={};
            if(FAILED(context->Map(data.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped)))return false;
            auto* out=static_cast<float*>(mapped.pData);unsigned li=0,bi=0;
            for(auto city:cities){
                for(auto const& l:city->lights){
                    float* p=out+li*12,*c=p+4,*o=p+8;
                    for(unsigned j=0;j<3;++j){p[j]=l.position[j];c[j]=l.color[j];o[j]=l.direction[j];
                        constants[4+j]=std::min(constants[4+j],p[j]-l.range);
                        constants[8+j]=std::max(constants[8+j],p[j]+l.range);}
                    p[3]=l.range;c[3]=l.intensity;o[3]=float(l.owner+bi);++li;
                }
                for(auto const& b:city->blockers){
                    std::memcpy(out+(nl*3+bi*2)*4,b.low,16);
                    std::memcpy(out+(nl*3+bi*2+1)*4,b.high,16);++bi;
                }
            }
            context->Unmap(data.Get(),0);
        }
        context->UpdateSubresource(frame.Get(),0,nullptr,constants,0,0);bind(context);return true;
    }
};
} }
