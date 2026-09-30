#pragma once
#include "pass_counts.h"
struct SandboxPassWorkload : SandboxPassCounts {
    bool enabled=false;
    unsigned pass=main_scene;
    void begin() {
        char option[8]={};
        enabled=GetEnvironmentVariableA("C3X_SANDBOX_PASS_COUNTS",option,sizeof(option)) && option[0]=='1';
        if(enabled)counts={};
        pass=main_scene;
    }
    Counts& row(unsigned layer=screen){return counts[pass][layer];}
    void draw(std::uint64_t indices,std::uint64_t instances=1,unsigned layer=screen) {
        if(!enabled)return;
        auto& c=row(layer);++c.draws;c.submitted_instances+=instances;
        c.index_vertices+=indices*instances;c.triangles+=(indices/3)*instances;
    }
    void upload(std::size_t bytes,unsigned layer=screen){if(enabled)row(layer).upload_bytes+=bytes;}
    void upload_buffer(ID3D11Buffer* buffer) {
        if(!enabled || !buffer)return;D3D11_BUFFER_DESC d={};buffer->GetDesc(&d);upload(d.ByteWidth);
    }
    void copy(ID3D11Resource* resource,bool copied) {
        if(!enabled || !resource)return;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        if(FAILED(resource->QueryInterface(IID_PPV_ARGS(&texture))))return;
        D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
        auto pixels=std::uint64_t(d.Width)*d.Height*d.SampleDesc.Count;
        if(copied)row().copy_pixels+=pixels;else row().target_pixels+=pixels;
    }
    void clear(ID3D11View* view) {
        if(!enabled || !view)return;
        Microsoft::WRL::ComPtr<ID3D11Resource> resource;view->GetResource(&resource);
        copy(resource.Get(),false);
    }
    struct Scope {
        SandboxPassWorkload& work;unsigned previous;
        Scope(SandboxPassWorkload& w,unsigned p):work(w),previous(w.pass){work.pass=p;}
        ~Scope(){work.pass=previous;}
    };
};
