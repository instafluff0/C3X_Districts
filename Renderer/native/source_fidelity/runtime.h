#pragma once
#include "../../lab/shared/natural/world.h"
#include "coast_join.h"
// Generic natural payload. Source-specific names and recipes are compiled offline.
namespace c3x_renderer { namespace fidelity {
struct Natural : NaturalWorld {
    std::vector<ID3D11ShaderResourceView*> textures;
    ID3D11VertexShader* vs[3]={};ID3D11PixelShader* ps[3]={};
    ID3D11InputLayout* layout[3]={};ID3D11Buffer* frames[3]={};
    ID3D11DepthStencilState*decal_depth=nullptr;
    bool ready=false;
    template<class T>void drop(T*&p){if(p)p->Release();p=nullptr;}
    void reset(){drop(decal_depth);reset_world();for(auto&p:textures)drop(p);textures.clear();fields.clear();materials.clear();bodies.clear();recipes.clear();
        for(int i=0;i<3;i++){drop(vs[i]);drop(ps[i]);drop(layout[i]);drop(frames[i]);}ready=false;}
    ~Natural(){reset();}
    template<class Read,class Upload>
    bool load(ID3D11Device*device,std::string const&root,Read read,Upload upload,
              char const*shader_directory="source_fidelity"){
        if(ready)return true;reset();
        if(!load_data(textures,read,upload))return false;
        char const*names[]={"terrain","mountain","objects"};
        for(unsigned i=0;i<3;i++){
            std::string path=root+"/Renderer/native/"+shader_directory+"/"+names[i]+".hlsl";
            failure="shader "+std::string(names[i]);
            std::wstring wide(path.begin(),path.end());ID3DBlob *v=nullptr,*p=nullptr,*error=nullptr;
            HRESULT hr=profile_v2::compile_cached(wide.c_str(),"VSNative","vs_5_0",&v,&error);
            if(error){failure+=static_cast<char const*>(error->GetBufferPointer());OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));drop(error);}
            if(SUCCEEDED(hr))hr=profile_v2::compile_cached(wide.c_str(),"PSFeature","ps_5_0",&p,&error);
            if(error){failure+=static_cast<char const*>(error->GetBufferPointer());OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));drop(error);}
            if(SUCCEEDED(hr))hr=device->CreateVertexShader(v->GetBufferPointer(),v->GetBufferSize(),nullptr,&vs[i]);
            if(SUCCEEDED(hr))hr=device->CreatePixelShader(p->GetBufferPointer(),p->GetBufferSize(),nullptr,&ps[i]);
            D3D11_INPUT_ELEMENT_DESC e[]={
                {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,28,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",1,DXGI_FORMAT_R32G32_FLOAT,0,40,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",2,DXGI_FORMAT_R32G32B32A32_FLOAT,0,48,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",3,DXGI_FORMAT_R32G32_FLOAT,0,64,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",4,DXGI_FORMAT_R32_FLOAT,0,72,D3D11_INPUT_PER_VERTEX_DATA,0}};
            if(SUCCEEDED(hr))hr=device->CreateInputLayout(e,i==0?7:i==1?5:6,v->GetBufferPointer(),v->GetBufferSize(),&layout[i]);
            drop(v);drop(p);D3D11_BUFFER_DESC desc={};desc.ByteWidth=96;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(SUCCEEDED(hr))hr=device->CreateBuffer(&desc,nullptr,&frames[i]);
            if(FAILED(hr))return false;
        }
        D3D11_DEPTH_STENCIL_DESC depth={};depth.DepthEnable=TRUE;depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
        if(FAILED(device->CreateDepthStencilState(&depth,&decal_depth)))return false;
        ready=true;return true;
    }
    void update(ID3D11DeviceContext*c,EnvironmentState const&e,float const*light){
        auto values=frame_settings(e,light);
        for(unsigned i=0;i<3;i++)c->UpdateSubresource(frames[i],0,nullptr,&values[i],0,0);
    }
    void bind(ID3D11DeviceContext*c,unsigned provider,unsigned body=0){
        c->VSSetShader(vs[provider],nullptr,0);c->PSSetShader(ps[provider],nullptr,0);
        c->IASetInputLayout(layout[provider]);c->PSSetConstantBuffers(0,1,&frames[provider]);
        ID3D11ShaderResourceView*views[22]={};
        if(provider==0)for(unsigned i=0;i<22;i++)views[i]=textures[terrain[i]];
        if(provider==1)for(unsigned i=0;i<13;i++)views[i]=textures[mountain[i]];
        if(provider==2){auto const&m=materials[bodies[body].material];for(unsigned i=0;i<7;i++)if(m.channels[i]!=0xffffffffu)views[i+3]=textures[m.channels[i]];}
        // t17 is always the shared atlas, never the source specular channel.
        c->PSSetShaderResources(0,17,views);c->PSSetShaderResources(18,4,views+18);
    }
};
} }
