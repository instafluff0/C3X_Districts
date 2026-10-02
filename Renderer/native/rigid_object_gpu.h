#pragma once
#include "rigid_object_instance.h"
#include "render_core/instance_stream.h"
#include "render_core/shared_instance_submission.h"

namespace c3x_renderer { namespace objects {
// Source assets are uploaded once per pack/device lifetime. Tile residency
// retains COM references plus small instances, never transformed source copies.
class RigidSourceGpu {
public:
    struct Mesh {ID3D11Buffer* buffer=nullptr;unsigned count=0,index_offset=0;};
    std::array<std::vector<Mesh>,family_count> meshes;
    ID3D11VertexShader* vertex[2]={},*resident_vertex[2]={};ID3D11InputLayout* layout=nullptr,*resident_layout=nullptr;
    render_core::InstanceStream stream;
    render_core::SharedSourceResidency residency;
    std::size_t bytes=0;unsigned allocations=0;
    std::shared_ptr<void const> pin()const{return residency.pin();}
    std::size_t resident_bytes()const{return residency.bytes();}
    std::size_t resident_gpu_bytes()const{return residency.gpu_bytes();}
    template<class T> void drop(T*& value){if(value)value->Release();value=nullptr;}
    ~RigidSourceGpu(){clear();}
    void clear(){stream.clear();for(auto& family:meshes){for(auto& mesh:family)drop(mesh.buffer);family.clear();}
        for(auto& shader:vertex)drop(shader);for(auto& shader:resident_vertex)drop(shader);drop(layout);drop(resident_layout);bytes=allocations=0;residency.clear();}
    bool ensure(ID3D11Device* device,std::string const& root,Assets const& assets){
        if(vertex[1])return true;clear();
        auto fail=[&]{clear();return false;};
        try{
            std::wstring path(root.begin(),root.end());path+=L"/Renderer/native/city_fidelity/rigid_feature.hlsl";
            for(unsigned pass=0;pass<2;++pass){ID3DBlob* code=nullptr,*errors=nullptr;
                auto hr=render_core::compile_cached(path.c_str(),pass?"VSSharedFeatureReflection":"VSSharedFeature","vs_5_0",&code,&errors);
                if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
                if(SUCCEEDED(hr))hr=device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vertex[pass]);
                if(!pass && SUCCEEDED(hr))hr=render_core::create_instance_layout(device,code,&layout);
                drop(code);if(FAILED(hr))return fail();
            }
            for(unsigned pass=0;pass<2;++pass){ID3DBlob* code=nullptr,*errors=nullptr;
                auto hr=render_core::compile_cached(path.c_str(),pass?"VSResidentSharedFeatureReflection":"VSResidentSharedFeature","vs_5_0",&code,&errors);
                if(errors){OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
                if(SUCCEEDED(hr))hr=device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&resident_vertex[pass]);
                if(!pass && SUCCEEDED(hr))hr=render_core::create_resident_instance_layout(device,code,&resident_layout);
                drop(code);if(FAILED(hr))return fail();
            }
            for(unsigned family=0;family<family_count;++family){auto const& bundle=assets[Family(family)];
                meshes[family].resize(bundle.assets.size());
                for(unsigned index=0;index<bundle.assets.size();++index){auto const& asset=bundle.assets[index];auto& mesh=meshes[family][index];
                    if(!shared_rigid_mesh(asset))continue;
                    auto vertices=asset.vertices.size()*sizeof(FeatureSourceVertex),indices=asset.indices.size()*sizeof(unsigned);
                    auto size=vertices+indices;if(size>32u*1024u*1024u-bytes || !residency.reserve(size,bytes+size))return fail();
                    std::vector<unsigned char> data(size);
                    std::memcpy(data.data(),asset.vertices.data(),vertices);
                    std::memcpy(data.data()+vertices,asset.indices.data(),indices);
                    D3D11_BUFFER_DESC desc{};desc.ByteWidth=unsigned(size);desc.Usage=D3D11_USAGE_IMMUTABLE;
                    desc.BindFlags=D3D11_BIND_VERTEX_BUFFER|D3D11_BIND_INDEX_BUFFER;
                    D3D11_SUBRESOURCE_DATA initial{};initial.pSysMem=data.data();
                    if(FAILED(device->CreateBuffer(&desc,&initial,&mesh.buffer)))return fail();
                    mesh.index_offset=unsigned(vertices);mesh.count=unsigned(asset.indices.size());bytes+=size;++allocations;
                    residency.reserve(0,bytes);
                }
            }
        }catch(...){return fail();}
        return true;
    }
};
}}
