#pragma once
#include "runtime.h"
#include <memory>
namespace c3x_renderer { namespace city_fidelity {
struct Lighting {
    std::vector<Light> lights;
    struct Box {float low[4],high[4];};
    std::vector<Box> blockers;
};
struct Gpu {
    Library library;
    std::vector<std::array<ID3D11ShaderResourceView*,7>> materials;
    std::unordered_map<std::string,ID3D11ShaderResourceView*> textures;
    ID3D11VertexShader*vs[2]={};ID3D11PixelShader*ps[4]={};
    ID3D11InputLayout*layout=nullptr;ID3D11Buffer*material_frame=nullptr,*light_frame=nullptr;
    ID3D11BlendState*emission=nullptr;ID3D11DepthStencilState*readonly_depth=nullptr;
    std::size_t texture_bytes=0;bool ready=false;
    float night=0,emissive_scale=1;
    template<class T>void drop(T*&p){if(p)p->Release();p=nullptr;}
    void reset(){for(auto&p:vs)drop(p);for(auto&p:ps)drop(p);drop(layout);drop(material_frame);drop(light_frame);
        drop(emission);drop(readonly_depth);for(auto&p:textures)drop(p.second);textures.clear();materials.clear();
        library={};texture_bytes=0;ready=false;}
    ~Gpu(){reset();}
    template<class Read,class Upload>bool load(ID3D11Device*device,std::string const&root,Read read,Upload upload){
        if(ready)return true;reset();
        std::vector<std::uint8_t>bytes;if(!read("Renderer/packs/CityCompositionRuntime/city.bin",bytes) || !library.decode(bytes))return false;
        std::wstring path(root.begin(),root.end());path+=L"/Renderer/native/city_fidelity/city.hlsl";
        char const*entries[]={"VSNativeCity","VSNativeCityReflection","PSNativeCity","PSNativeCityEmission","PSNativeCityReflection","PSNativeCityReflectionEmission"};
        for(unsigned i=0;i<6;i++){
            ID3DBlob*blob=nullptr,*errors=nullptr;
            HRESULT hr=profile_v2::compile_cached(path.c_str(),entries[i],i<2?"vs_5_0":"ps_5_0",&blob,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
            if(SUCCEEDED(hr))hr=i<2?device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vs[i]):
                device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&ps[i-2]);
            if(SUCCEEDED(hr) && i==0){
                D3D11_INPUT_ELEMENT_DESC e[]={
                    {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                    {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
                    {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
                    {"TEXCOORD",1,DXGI_FORMAT_R32G32_FLOAT,0,44,D3D11_INPUT_PER_VERTEX_DATA,0},
                    {"TEXCOORD",2,DXGI_FORMAT_R32_FLOAT,0,60,D3D11_INPUT_PER_VERTEX_DATA,0},
                    {"TEXCOORD",3,DXGI_FORMAT_R32G32B32_FLOAT,0,68,D3D11_INPUT_PER_VERTEX_DATA,0},
                    {"TEXCOORD",4,DXGI_FORMAT_R32G32B32_FLOAT,0,80,D3D11_INPUT_PER_VERTEX_DATA,0},
                    {"TEXCOORD",5,DXGI_FORMAT_R32G32B32_FLOAT,0,120,D3D11_INPUT_PER_VERTEX_DATA,0},
                    {"TEXCOORD",6,DXGI_FORMAT_R32G32_FLOAT,0,152,D3D11_INPUT_PER_VERTEX_DATA,0}};
                hr=device->CreateInputLayout(e,9,blob->GetBufferPointer(),blob->GetBufferSize(),&layout);
            }
            drop(blob);if(FAILED(hr)){reset();return false;}
        }
        materials.resize(library.materials.size());
        for(std::size_t m=0;m<materials.size();m++)for(unsigned c=0;c<7;c++){
            auto const&name=library.materials[m].textures[c];if(name.empty())continue;
            auto found=textures.find(name);
            if(found==textures.end()){
                if(!read(name,bytes) || texture_bytes+bytes.size()>64u*1024u*1024u){reset();return false;}
                auto inserted=textures.emplace(name,nullptr); // ownership before upload
                if(!upload(bytes,inserted.first->second)){reset();return false;}
                texture_bytes+=bytes.size();found=inserted.first;
            }
            materials[m][c]=found->second;
        }
        D3D11_BUFFER_DESC d={};d.ByteWidth=32;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        HRESULT hr=device->CreateBuffer(&d,nullptr,&material_frame);
        d.ByteWidth=(3+1024*3+256*2)*16;
        if(SUCCEEDED(hr))hr=device->CreateBuffer(&d,nullptr,&light_frame);
        D3D11_BLEND_DESC b={};auto&r=b.RenderTarget[0];r.BlendEnable=TRUE;
        r.SrcBlend=r.DestBlend=D3D11_BLEND_ONE;r.BlendOp=r.BlendOpAlpha=D3D11_BLEND_OP_ADD;
        r.SrcBlendAlpha=D3D11_BLEND_ZERO;r.DestBlendAlpha=D3D11_BLEND_ONE;r.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
        if(SUCCEEDED(hr))hr=device->CreateBlendState(&b,&emission);
        D3D11_DEPTH_STENCIL_DESC z={};z.DepthEnable=TRUE;z.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;z.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
        if(SUCCEEDED(hr))hr=device->CreateDepthStencilState(&z,&readonly_depth);
        if(FAILED(hr)){reset();return false;}ready=true;return true;
    }
    void bind(ID3D11DeviceContext*context,unsigned material,bool environment,float const*atlas,bool reflect,bool emit){
        auto const&m=materials[material];
        context->IASetInputLayout(layout);context->VSSetShader(vs[reflect?1:0],nullptr,0);
        context->PSSetShader(ps[(reflect?2:0)+(emit?1:0)],nullptr,0);
        float values[]={environment?1.f:0.f,0,0,0,atlas[0],atlas[1],atlas[2],atlas[3]};
        context->UpdateSubresource(material_frame,0,nullptr,values,0,0);context->PSSetConstantBuffers(7,1,&material_frame);
        context->PSSetShaderResources(124,1,m.data());
        ID3D11ShaderResourceView*extra[]={m[1],m[4],m[2],m[3],m[5],m[6]};context->PSSetShaderResources(116,6,extra);
        if(emit)context->OMSetBlendState(emission,nullptr,0xffffffffu);
        if(emit || library.materials[material].ground)context->OMSetDepthStencilState(readonly_depth,0);
    }
    bool lights(ID3D11DeviceContext*context,std::vector<Lighting const*>const&cities){
        std::vector<float> values((3+1024*3+256*2)*4,0);
        unsigned nl=0,nb=0;float*low=values.data()+4,*high=values.data()+8;
        for(unsigned j=0;j<3;j++){low[j]=1e9f;high[j]=-1e9f;}
        for(auto city:cities){
            if(nl+city->lights.size()>1024 || nb+city->blockers.size()>256)return false;
            for(auto const&l:city->lights){
                float*p=values.data()+(3+nl)*4,*c=values.data()+(3+1024+nl)*4,*o=values.data()+(3+2048+nl)*4;
                for(unsigned j=0;j<3;j++){p[j]=l.position[j];c[j]=l.color[j];o[j]=l.direction[j];
                    low[j]=std::min(low[j],p[j]-l.range);high[j]=std::max(high[j],p[j]+l.range);}
                p[3]=l.range;c[3]=l.intensity;o[3]=l.owner+float(nb);nl++;
            }
            for(auto const&b:city->blockers){
                std::memcpy(values.data()+(3+3072+nb)*4,b.low,16);
                std::memcpy(values.data()+(3+3072+256+nb)*4,b.high,16);nb++;
            }
        }
        values[0]=float(nl);values[1]=float(nb);values[2]=night;values[3]=emissive_scale;
        context->UpdateSubresource(light_frame,0,nullptr,values.data(),0,0);context->PSSetConstantBuffers(6,1,&light_frame);return true;
    }
};
} }
