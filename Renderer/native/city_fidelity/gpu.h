#pragma once
#include "runtime.h"
#include "scene_lights.h"
#include <memory>
namespace c3x_renderer { namespace city_fidelity {
struct Gpu {
    Library library;
    std::vector<std::array<ID3D11ShaderResourceView*,7>> materials;
    std::unordered_map<std::string,ID3D11ShaderResourceView*> textures;
    ID3D11VertexShader*vs[2]={};ID3D11PixelShader*ps[4]={};
    ID3D11InputLayout*layout=nullptr,*compact_layout=nullptr;ID3D11Buffer*material_frame=nullptr;
    ID3D11BlendState*emission=nullptr;ID3D11DepthStencilState*readonly_depth=nullptr;
    std::size_t texture_bytes=0;bool ready=false;
    float night=0,emissive_scale=1;SceneLights scene_lights;
    template<class T>void drop(T*&p){if(p)p->Release();p=nullptr;}
    void reset(){for(auto&p:vs)drop(p);for(auto&p:ps)drop(p);drop(layout);drop(compact_layout);drop(material_frame);scene_lights={};
        drop(emission);drop(readonly_depth);for(auto&p:textures)drop(p.second);textures.clear();materials.clear();
        library={};texture_bytes=0;ready=false;}
    ~Gpu(){reset();}
    template<class Read,class Upload>bool load(ID3D11Device*device,std::string const&root,Read read,Upload upload,
            std::string const&pack="Renderer/packs/CityCompositionRuntime"){
        if(ready)return true;reset();
        auto packed_read=[&](std::string const& path,std::vector<std::uint8_t>& bytes){
            std::string const prefix="Renderer/packs/CityCompositionRuntime/";
            return read(path.compare(0,prefix.size(),prefix)==0?pack+"/"+path.substr(prefix.size()):path,bytes);
        };
        std::vector<std::uint8_t>bytes;if(!packed_read("Renderer/packs/CityCompositionRuntime/city.bin",bytes) ||
            !library.decode(bytes) || !library.complete_city_set())return false;
        std::wstring path(root.begin(),root.end());path+=L"/Renderer/native/city_fidelity/city.hlsl";
        char const*entries[]={"VSNativeCity","VSNativeCityReflection","PSNativeCity","PSNativeCityEmission","PSNativeCityReflection","PSNativeCityReflectionEmission"};
        for(unsigned i=0;i<6;i++){
            ID3DBlob*blob=nullptr,*errors=nullptr;
            HRESULT hr=render_core::compile_cached(path.c_str(),entries[i],i<2?"vs_5_0":"ps_5_0",&blob,&errors);
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
                unsigned const offsets[]={0,12,20,32,40,44,56,68,80};
                for(unsigned field=0;field<9;++field)e[field].AlignedByteOffset=offsets[field];
                if(SUCCEEDED(hr))hr=device->CreateInputLayout(e,9,blob->GetBufferPointer(),blob->GetBufferSize(),&compact_layout);
            }
            drop(blob);if(FAILED(hr)){reset();return false;}
        }
        materials.resize(library.materials.size());
        for(std::size_t m=0;m<materials.size();m++)for(unsigned c=0;c<7;c++){
            auto const&name=library.materials[m].textures[c];if(name.empty())continue;
            auto found=textures.find(name);
            if(found==textures.end()){
                if(!packed_read(name,bytes) || texture_bytes+bytes.size()>128u*1024u*1024u){reset();return false;}
                auto inserted=textures.emplace(name,nullptr); // ownership before upload
                if(!upload(bytes,inserted.first->second)){reset();return false;}
                texture_bytes+=bytes.size();found=inserted.first;
            }
            materials[m][c]=found->second;
        }
        D3D11_BUFFER_DESC d={};d.ByteWidth=32;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        HRESULT hr=device->CreateBuffer(&d,nullptr,&material_frame);
        D3D11_BLEND_DESC b={};auto&r=b.RenderTarget[0];r.BlendEnable=TRUE;
        r.SrcBlend=r.DestBlend=D3D11_BLEND_ONE;r.BlendOp=r.BlendOpAlpha=D3D11_BLEND_OP_ADD;
        r.SrcBlendAlpha=D3D11_BLEND_ZERO;r.DestBlendAlpha=D3D11_BLEND_ONE;r.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
        if(SUCCEEDED(hr))hr=device->CreateBlendState(&b,&emission);
        D3D11_DEPTH_STENCIL_DESC z={};z.DepthEnable=TRUE;z.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;z.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
        if(SUCCEEDED(hr))hr=device->CreateDepthStencilState(&z,&readonly_depth);
        if(FAILED(hr)){reset();return false;}ready=true;return true;
    }
    void bind(ID3D11DeviceContext*context,unsigned material,bool environment,float const*atlas,bool reflect,bool emit,bool compact=false){
        auto const&m=materials[material];scene_lights.bind(context);
        context->IASetInputLayout(compact?compact_layout:layout);context->VSSetShader(vs[reflect?1:0],nullptr,0);
        context->PSSetShader(ps[(reflect?2:0)+(emit?1:0)],nullptr,0);
        float values[]={environment?1.f:0.f,0,0,0,atlas[0],atlas[1],atlas[2],atlas[3]};
        context->UpdateSubresource(material_frame,0,nullptr,values,0,0);context->PSSetConstantBuffers(7,1,&material_frame);
        context->PSSetShaderResources(124,1,m.data());
        ID3D11ShaderResourceView*extra[]={m[1],m[4],m[2],m[3],m[5],m[6]};context->PSSetShaderResources(116,6,extra);
        if(emit)context->OMSetBlendState(emission,nullptr,0xffffffffu);
        if(emit || library.materials[material].ground)context->OMSetDepthStencilState(readonly_depth,0);
    }
    // Body and emission consume the same geometry/material bindings. The
    // emission shader adds RGB only, preserves alpha, and writes no depth.
    bool emits(unsigned material)const {
        return material<materials.size() && materials[material][1]!=nullptr &&
            night!=0.f && emissive_scale!=0.f;
    }
    void bind_emission(ID3D11DeviceContext*context,bool reflect){
        context->PSSetShader(ps[reflect?3:1],nullptr,0);
        context->OMSetBlendState(emission,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(readonly_depth,0);
    }
    bool lights(ID3D11DeviceContext*context,std::vector<Lighting const*>const&cities){
        return scene_lights.upload(context,cities,night,emissive_scale);
    }
};
} }
