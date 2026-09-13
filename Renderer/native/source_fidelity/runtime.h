#pragma once
#include "../../lab/shared/natural/world.h"
#include "coast_join.h"
#include "../render_core/instance_stream.h"
// Generic natural payload. Source-specific names and recipes are compiled offline.
namespace c3x_renderer { namespace fidelity {
struct Natural : NaturalWorld {
    std::vector<ID3D11ShaderResourceView*> textures;
    ID3D11VertexShader* vs[3]={};ID3D11PixelShader* ps[3]={};
    ID3D11InputLayout* layout[3]={};ID3D11Buffer* frames[3]={};
    ID3D11DepthStencilState*decal_depth=nullptr;
    struct InstanceMesh {ID3D11Buffer*vertices=nullptr,*indices=nullptr;unsigned count=0;};
    std::vector<InstanceMesh> instance_meshes;
    std::size_t instance_mesh_bytes=0;
    ID3D11VertexShader*instance_vs=nullptr;ID3D11InputLayout*instance_layout=nullptr;
    ID3D11Buffer*instance_material=nullptr;
    render_core::InstanceStream instance_stream;
    std::wstring instance_path;
    bool ready=false;
    template<class T>void drop(T*&p){if(p)p->Release();p=nullptr;}
    void reset(){
        instance_stream.clear();drop(instance_vs);drop(instance_layout);drop(instance_material);
        for(auto&m:instance_meshes){drop(m.vertices);drop(m.indices);}instance_meshes.clear();instance_mesh_bytes=0;
        drop(decal_depth);reset_world();for(auto&p:textures)drop(p);textures.clear();fields.clear();materials.clear();bodies.clear();recipes.clear();surface_recipes.clear();surface_vertices.clear();
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
            if(i==2)instance_path=std::wstring(path.begin(),path.end());
            failure="shader "+std::string(names[i]);
            std::wstring wide(path.begin(),path.end());ID3DBlob *v=nullptr,*p=nullptr,*error=nullptr;
            HRESULT hr=render_core::compile_cached(wide.c_str(),"VSNative","vs_5_0",&v,&error);
            if(error){failure+=static_cast<char const*>(error->GetBufferPointer());OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));drop(error);}
            if(SUCCEEDED(hr))hr=render_core::compile_cached(wide.c_str(),"PSFeature","ps_5_0",&p,&error);
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
                {"TEXCOORD",4,DXGI_FORMAT_R32_FLOAT,0,72,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",6,DXGI_FORMAT_R32G32B32A32_FLOAT,0,76,D3D11_INPUT_PER_VERTEX_DATA,0}};
            if(SUCCEEDED(hr))hr=device->CreateInputLayout(e,i<2?8:6,v->GetBufferPointer(),v->GetBufferSize(),&layout[i]);
            drop(v);drop(p);D3D11_BUFFER_DESC desc={};desc.ByteWidth=96;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(SUCCEEDED(hr))hr=device->CreateBuffer(&desc,nullptr,&frames[i]);
            if(FAILED(hr))return false;
        }
        D3D11_DEPTH_STENCIL_DESC depth={};depth.DepthEnable=TRUE;depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
        if(FAILED(device->CreateDepthStencilState(&depth,&decal_depth)))return false;
        ready=true;return true;
    }
    bool ensure_instance_mesh(ID3D11Device*device,unsigned body){
        if(body>=bodies.size())return false;
        if(!instance_vs){
            ID3DBlob*code=nullptr,*errors=nullptr;
            HRESULT hr=render_core::compile_cached(instance_path.c_str(),"VSInstance","vs_5_0",&code,&errors);
            if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
            if(SUCCEEDED(hr))hr=device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&instance_vs);
            if(SUCCEEDED(hr))hr=render_core::create_instance_layout(device,code,&instance_layout);
            drop(code);D3D11_BUFFER_DESC d={};d.ByteWidth=32;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(SUCCEEDED(hr))hr=device->CreateBuffer(&d,nullptr,&instance_material);
            if(FAILED(hr)){drop(instance_vs);drop(instance_layout);drop(instance_material);return false;}
        }
        if(instance_meshes.empty())instance_meshes.resize(bodies.size());
        auto&m=instance_meshes[body];if(m.vertices)return true;
        std::map<std::array<unsigned,8>,unsigned> lookup;std::vector<BodyVertex> vertices;std::vector<unsigned> indices;
        for(auto const&v:bodies[body].vertices){
            std::array<unsigned,8> key;std::memcpy(key.data(),&v,sizeof(v));auto found=lookup.find(key);
            if(found==lookup.end()){auto n=unsigned(vertices.size());vertices.push_back(v);found=lookup.emplace(key,n).first;}
            indices.push_back(found->second);
        }
        std::size_t bytes=vertices.size()*sizeof(BodyVertex)+indices.size()*sizeof(unsigned);
        if(instance_mesh_bytes+bytes>32u*1024u*1024u)return false;
        D3D11_BUFFER_DESC d={};d.ByteWidth=UINT(vertices.size()*sizeof(BodyVertex));d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA input={};input.pSysMem=vertices.data();
        if(FAILED(device->CreateBuffer(&d,&input,&m.vertices)))return false;
        d.ByteWidth=UINT(indices.size()*sizeof(unsigned));d.BindFlags=D3D11_BIND_INDEX_BUFFER;input.pSysMem=indices.data();
        if(FAILED(device->CreateBuffer(&d,&input,&m.indices))){drop(m.vertices);return false;}
        m.count=unsigned(indices.size());instance_mesh_bytes+=bytes;return true;
    }
    void bind_instances(ID3D11DeviceContext*c,unsigned body){
        auto const&m=materials[bodies[body].material];float values[]={1,m.repeat?2.f:0.f,m.channels[3]!=0xffffffffu?1.f:0.f,
            m.channels[4]!=0xffffffffu?1.f:0.f,float(m.tint),m.channels[6]!=0xffffffffu?1.f:0.f,0,0};
        c->UpdateSubresource(instance_material,0,nullptr,values,0,0);
        c->VSSetConstantBuffers(9,1,&instance_material);c->VSSetShader(instance_vs,nullptr,0);c->IASetInputLayout(instance_layout);
    }
    void update(ID3D11DeviceContext*c,EnvironmentState const&e,float const*light){
        auto values=frame_settings(e,light);
        for(unsigned i=0;i<3;i++)c->UpdateSubresource(frames[i],0,nullptr,&values[i],0,0);
    }
    void bind(ID3D11DeviceContext*c,unsigned provider,unsigned body=0){
        c->VSSetShader(vs[provider],nullptr,0);c->PSSetShader(ps[provider],nullptr,0);
        c->IASetInputLayout(layout[provider]);c->PSSetConstantBuffers(0,1,&frames[provider]);
        ID3D11ShaderResourceView*views[31]={};
        if(provider==0)for(unsigned i=0;i<31;i++)views[i]=textures[terrain[i]];
        if(provider==1){
            for(unsigned i=0;i<13;i++)views[i]=textures[mountain[i]];
            // Reuse the generic terrain-pack families for the collar. Runtime
            // remains independent of any particular source game's file tree.
            views[13]=textures[terrain[6]];views[14]=textures[terrain[7]];
            views[15]=textures[terrain[8]];views[16]=textures[terrain[15]];
            views[18]=textures[terrain[16]];views[19]=textures[terrain[18]];
            views[20]=textures[terrain[19]];views[21]=textures[terrain[20]];
            views[22]=textures[terrain[21]];views[23]=textures[terrain[14]];
            views[24]=textures[terrain[3]];views[25]=textures[terrain[4]];
            views[26]=textures[terrain[5]];views[27]=textures[terrain[9]];
            views[28]=textures[terrain[10]];views[29]=textures[terrain[11]];
            views[30]=textures[terrain[30]];
        }
        if(provider==2){auto const&m=materials[bodies[body].material];for(unsigned i=0;i<7;i++)if(m.channels[i]!=0xffffffffu)views[i+3]=textures[m.channels[i]];}
        // t17 is always the shared atlas, never the source specular channel.
        c->PSSetShaderResources(0,17,views);c->PSSetShaderResources(18,13,views+18);
    }
};
} }
