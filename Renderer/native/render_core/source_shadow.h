#pragma once
#include <set>
#include <map>
#include "shader_cache.h"
#include <array>
#include <atomic>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace c3x_renderer { namespace render_core {
// One bounded worker-owned atlas: 32 * 1024^2 * R32 = 128 MiB. Immutable
// geometry is drawn directly; there is no CPU triangle copy or readback.
class SourceShadow {
    struct Page { int x=0,y=0;std::uint64_t hash=0,used=0; };
    std::array<Page,32> pages{};
    std::array<float,12> basis{};
    std::uint64_t epoch=0;
    ID3D11Texture2D* texture=nullptr;
    std::array<ID3D11RenderTargetView*,32> targets{};
    ID3D11VertexShader* vertex=nullptr;
    ID3D11PixelShader *opaque=nullptr,*cutout=nullptr;
    ID3D11InputLayout *layout=nullptr,*feature_layout=nullptr,*natural_layout=nullptr;
    ID3D11Buffer* caster_settings=nullptr;
    ID3D11RasterizerState* raster=nullptr;
    ID3D11BlendState* maximum=nullptr;
    template<class T> void drop(T*& p){if(p)p->Release();p=nullptr;}
public:
    ID3D11ShaderResourceView* view=nullptr;
    ID3D11Buffer* table=nullptr;
    unsigned hits=0,rebuilt=0,draws=0;
    struct Bounds { float low[3]={},high[3]={}; };
    struct Caster {
        DXGI_FORMAT index_format=DXGI_FORMAT_R32_UINT;
        ID3D11Buffer *vertices=nullptr,*indices=nullptr;
        unsigned count=0,stride=0,layer=0;
        unsigned binding=0xffffffffu;
        std::uint64_t version=0;
        Bounds bounds;
        float offset[3]={};
    };
    SourceShadow()=default;
    SourceShadow(SourceShadow const&)=delete;
    ~SourceShadow(){clear();}
    void clear_cached_pages(){pages={};basis={};epoch=0;}
    void clear(){
        drop(view);drop(texture);for(auto& t:targets)drop(t);
        drop(vertex);drop(opaque);drop(cutout);drop(layout);drop(feature_layout);drop(natural_layout);drop(caster_settings);
        drop(table);drop(raster);drop(maximum);clear_cached_pages();
    }
    bool ensure(ID3D11Device* device,wchar_t const* path) {
        if(view)return true;
        clear();ID3DBlob *code=nullptr,*errors=nullptr;
        auto compile=[&](char const* entry,char const* target){
            drop(code);drop(errors);
            HRESULT hr=compile_cached(path,entry,target,&code,&errors);
            if(errors)OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));
            return SUCCEEDED(hr);
        };
        HRESULT hr=E_FAIL;
        if(compile("VS","vs_5_0")) {
            hr=device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vertex);
            D3D11_INPUT_ELEMENT_DESC elements[]={
                {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,12, D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",1,DXGI_FORMAT_R32_FLOAT,0,60,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",2,DXGI_FORMAT_R32G32B32A32_FLOAT,0,120,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",3,DXGI_FORMAT_R32_FLOAT,0,76,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",4,DXGI_FORMAT_R32G32B32A32_FLOAT,0,152,D3D11_INPUT_PER_VERTEX_DATA,0}};
            if(SUCCEEDED(hr))hr=device->CreateInputLayout(elements,5,code->GetBufferPointer(),code->GetBufferSize(),&layout);
            elements[4].AlignedByteOffset=0;
            elements[3].AlignedByteOffset=32;elements[1].AlignedByteOffset=32;elements[2].AlignedByteOffset=36;
            elements[2].Format=DXGI_FORMAT_R32G32B32_FLOAT;
            if(SUCCEEDED(hr))hr=device->CreateInputLayout(elements,5,code->GetBufferPointer(),code->GetBufferSize(),&feature_layout);
            elements[4].AlignedByteOffset=76;
            elements[3].AlignedByteOffset=56;elements[0].AlignedByteOffset=40;elements[1].AlignedByteOffset=72;
            elements[2].AlignedByteOffset=12;elements[2].Format=DXGI_FORMAT_R32G32B32A32_FLOAT;
            if(SUCCEEDED(hr))hr=device->CreateInputLayout(elements,5,code->GetBufferPointer(),code->GetBufferSize(),&natural_layout);
        }
        if(SUCCEEDED(hr) && compile("PSOpaque","ps_5_0"))
            hr=device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&opaque);
        else hr=E_FAIL;
        if(SUCCEEDED(hr) && compile("PSCutout","ps_5_0"))
            hr=device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&cutout);
        else hr=E_FAIL;
        drop(code);drop(errors);
        D3D11_TEXTURE2D_DESC d={};d.Width=d.Height=1024;d.ArraySize=32;d.MipLevels=1;
        d.Format=DXGI_FORMAT_R32_FLOAT;d.SampleDesc.Count=1;
        d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        if(SUCCEEDED(hr))hr=device->CreateTexture2D(&d,nullptr,&texture);
        if(SUCCEEDED(hr))hr=device->CreateShaderResourceView(texture,nullptr,&view);
        for(unsigned i=0;i<32 && SUCCEEDED(hr);++i){
            D3D11_RENDER_TARGET_VIEW_DESC r={};r.Format=d.Format;
            r.ViewDimension=D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
            r.Texture2DArray.FirstArraySlice=i;r.Texture2DArray.ArraySize=1;
            hr=device->CreateRenderTargetView(texture,&r,&targets[i]);
        }
        D3D11_BUFFER_DESC b={};b.Usage=D3D11_USAGE_DEFAULT;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;b.ByteWidth=80;
        if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&caster_settings);
        b.ByteWidth=64*16;if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&table);
        D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=FALSE;
        if(SUCCEEDED(hr))hr=device->CreateRasterizerState(&r,&raster);
        D3D11_BLEND_DESC blend={};auto& target=blend.RenderTarget[0];target.BlendEnable=TRUE;
        target.SrcBlend=target.DestBlend=target.SrcBlendAlpha=target.DestBlendAlpha=D3D11_BLEND_ONE;
        target.BlendOp=target.BlendOpAlpha=D3D11_BLEND_OP_MAX;target.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
        if(SUCCEEDED(hr))hr=device->CreateBlendState(&blend,&maximum);
        if(FAILED(hr)){clear();return false;}return true;
    }
    static std::array<float,4> project(Bounds const& b,float const* offset,std::array<float,12> const& projection) {
        std::array<float,4> out={1e9f,1e9f,-1e9f,-1e9f};
        for(unsigned mask=0;mask<8;++mask){float u=0,v=0;
            for(unsigned i=0;i<3;++i){float x=((mask>>i)&1?b.high[i]:b.low[i])+offset[i];u+=x*projection[i];v+=x*projection[4+i];}
            out[0]=std::min(out[0],u);out[1]=std::min(out[1],v);out[2]=std::max(out[2],u);out[3]=std::max(out[3],v);
        }return out;
    }
    std::array<float,4> projected(Bounds const& b,float const* offset)const {return project(b,offset,basis);}
    static std::set<std::pair<int,int>> required_pages(std::vector<Bounds> const& receivers,std::array<float,12> const& projection){
        std::set<std::pair<int,int>> needed;float zero[3]={};
        for(auto const& b:receivers){auto p=project(b,zero,projection);
            for(int y=int(std::floor((p[1]-.018f)/6));y<=int(std::floor((p[3]+.018f)/6));++y)
                for(int x=int(std::floor((p[0]-.018f)/6));x<=int(std::floor((p[2]+.018f)/6));++x)needed.emplace(x,y);
        }
        return needed;
    }
    // Borrowed by one outer submission's blocks/reflections only. Geometry,
    // material versions and light basis cannot change within that call. Optional
    // admission has a strict cap and failure falls back to ordinary projection.
    struct PreparedCasters {
        struct Selection {
            std::vector<std::size_t> indices;
            std::uint64_t hash=1469598103934665603ull;
        };
        struct Entry {std::pair<int,int> key{};Selection selection;bool valid=false;};
        std::vector<std::array<float,4>> bounds;
        std::array<float,12> basis{};
        Caster const* owner=nullptr;
        std::array<Entry,128> selections{};
        std::size_t selection_bytes=0;
        using ReceiverKey=std::vector<std::uint32_t>;
        using ReceiverProof=std::vector<std::uint64_t>;
        std::vector<std::uint64_t> retained_inputs;
        std::map<ReceiverKey,ReceiverProof> receiver_proofs;
        std::size_t receiver_bytes=0;
        std::uint64_t receiver_hits=0,receiver_misses=0;
        bool retained=false;
        static constexpr std::size_t receiver_budget=32u*1024u*1024u;
        static constexpr std::size_t receiver_limit=1024;
        // Exact value comparison, once per outer submission. No captured pointer
        // or hash alone authorizes reuse across requests. Pointer values below
        // describe GPU allocations and are never dereferenced by the proof cache.
        bool build(std::vector<Caster> const& casters,std::array<float,12> const& projection,bool retain=false) {
            std::vector<std::uint64_t> inputs;
            constexpr std::size_t input_cap=8u*1024u*1024u;
            if(retain && casters.size()<=input_cap/(18*sizeof(std::uint64_t)))try {
                inputs.reserve(casters.size()*18);
                for(auto const& c:casters){
                    for(auto value:{c.version,std::uint64_t(c.layer),std::uint64_t(c.index_format),std::uint64_t(c.binding),
                        std::uint64_t(c.count),std::uint64_t(c.stride),std::uint64_t(reinterpret_cast<std::uintptr_t>(c.vertices)),
                        std::uint64_t(reinterpret_cast<std::uintptr_t>(c.indices))})inputs.push_back(value);
                    for(auto values:{c.bounds.low,c.bounds.high,c.offset})for(unsigned j=0;j<3;++j){
                        std::uint32_t bits;std::memcpy(&bits,values+j,4);inputs.push_back(bits);
                    }
                }
                if(inputs.capacity()*sizeof(inputs[0])>input_cap)retain=false;
            }catch(...){retain=false;}
            else retain=false;
            receiver_hits=receiver_misses=0;
            if(retain && retained && inputs==retained_inputs && basis==projection && bounds.size()==casters.size()){
                owner=casters.data();return true;
            }
            retained=false;retained_inputs.clear();receiver_proofs.clear();receiver_bytes=0;
            owner=nullptr;bounds.clear();selections={};selection_bytes=0;
            constexpr std::size_t cap=8u*1024u*1024u;
            if(casters.size()>cap/sizeof(bounds[0]))return false;
            try {
                std::vector<std::array<float,4>> next;next.reserve(casters.size());
                if(next.capacity()>cap/sizeof(next[0]))return false;
                for(auto const& c:casters)next.push_back(project(c.bounds,c.offset,projection));
                bounds.swap(next);basis=projection;owner=casters.data();
                if(retain){retained_inputs=std::move(inputs);retained=true;}return true;
            }catch(...){return false;}
        }
        ReceiverKey receiver_key(std::vector<Bounds> const& receivers,bool narrowed)const {
            ReceiverKey key;
            // Oversized requests use ordinary validation without admission.
            if(!retained || receivers.size()>16384/6)return key;
            key.reserve(1+receivers.size()*6);key.push_back(narrowed?1u:0u);
            for(auto const& receiver:receivers)for(auto values:{receiver.low,receiver.high})
                for(unsigned j=0;j<3;++j){std::uint32_t bits;std::memcpy(&bits,values+j,4);key.push_back(bits);}
            return key;
        }
        ReceiverProof const* find_receiver(ReceiverKey const& key) {
            if(key.empty())return nullptr;
            auto found=receiver_proofs.find(key);
            if(found==receiver_proofs.end()){++receiver_misses;return nullptr;}
            ++receiver_hits;return &found->second;
        }
        void admit_receiver(ReceiverKey key,ReceiverProof proof) {
            if(key.empty() || proof.empty())return;
            auto bytes=key.capacity()*sizeof(key[0])+proof.capacity()*sizeof(proof[0])+sizeof(key)+sizeof(proof)+96;
            if(bytes>receiver_budget)return;
            // Optional CPU proofs may be discarded wholesale under pressure;
            // rendering and completed image ownership remain independent.
            if(receiver_proofs.size()>=receiver_limit || receiver_bytes>receiver_budget-bytes){receiver_proofs.clear();receiver_bytes=0;}
            if(receiver_proofs.emplace(std::move(key),std::move(proof)).second)receiver_bytes+=bytes;
        }
        bool matches(std::vector<Caster> const& casters,std::array<float,12> const& projection)const {
            return owner==casters.data() && bounds.size()==casters.size() && basis==projection;
        }
        Selection const* find(std::pair<int,int> key)const {
            for(auto const& entry:selections)if(entry.valid && entry.key==key)return &entry.selection;
            return nullptr;
        }
        Selection const* admit(std::pair<int,int> key,Selection&& selection) {
            auto bytes=selection.indices.capacity()*sizeof(std::size_t);
            constexpr std::size_t cap=8u*1024u*1024u;
            if(bytes>cap || selection_bytes>cap-bytes)return nullptr;
            for(auto& entry:selections)if(!entry.valid){
                entry.key=key;entry.selection=std::move(selection);entry.valid=true;
                selection_bytes+=bytes;return &entry.selection;
            }
            return nullptr;
        }
        static Selection select(std::vector<Caster> const& casters,std::vector<std::array<float,4>> const& projected_bounds,std::pair<int,int> key) {
            Selection result;
            auto mix=[&](std::uint64_t x){result.hash=(result.hash^x)*1099511628211ull;};
            for(std::size_t i=0;i<casters.size();++i){auto const& p=projected_bounds[i];
                if(p[2]<key.first*6 || p[0]>(key.first+1)*6 || p[3]<key.second*6 || p[1]>(key.second+1)*6)continue;
                result.indices.push_back(i);mix(casters[i].version);mix(casters[i].layer);mix(casters[i].index_format);
                if(casters[i].binding!=0xffffffffu)mix(casters[i].binding);
                for(float f:casters[i].offset){std::uint32_t bits;std::memcpy(&bits,&f,4);mix(bits);}
            }
            return result;
        }
        static Selection receiver_selection(Selection const& page,std::vector<Caster> const& casters,
                std::vector<std::array<float,4>> const& projected_bounds,
                std::vector<Bounds> const& receivers,std::array<float,12> const& projection) {
            // paged_shadow_v1: unit-normal offset, floor, then +/-1 texel taps.
            // Four texels conservatively cover that reach plus raster precision.
            constexpr float margin=4.f*6.f/1024.f;
            std::vector<std::array<float,4>> coverage;coverage.reserve(receivers.size());
            float zero[3]={};for(auto const& receiver:receivers)coverage.push_back(project(receiver,zero,projection));
            Selection result;auto mix=[&](std::uint64_t x){result.hash=(result.hash^x)*1099511628211ull;};
            for(auto i:page.indices){auto const& p=projected_bounds[i];bool reaches=false;
                for(auto const& r:coverage)if(!(p[2]<r[0]-margin || p[0]>r[2]+margin || p[3]<r[1]-margin || p[1]>r[3]+margin)){reaches=true;break;}
                if(!reaches)continue;
                result.indices.push_back(i);mix(casters[i].version);mix(casters[i].layer);mix(casters[i].index_format);
                if(casters[i].binding!=0xffffffffu)mix(casters[i].binding);
                for(float f:casters[i].offset){std::uint32_t bits;std::memcpy(&bits,&f,4);mix(bits);}
            }
            return result;
        }
    };
    template<class Bind>
    bool prepare(ID3D11DeviceContext* context,std::array<float,12> const& next_basis,
                 std::vector<Bounds> const& receivers,std::vector<Caster> const& casters,
                 Bind bind,std::atomic<bool> const* cancellation,PreparedCasters* prepared=nullptr,
                 std::set<std::pair<int,int>> const* selected_pages=nullptr) {
        hits=rebuilt=draws=0;++epoch;
        if(basis!=next_basis){basis=next_basis;pages={};}
        // A prepared pass may share the exact receiver-page union across regions.
        // The ordinary receiver query and the same 32-page residency cap remain.
        auto local_pages=selected_pages?std::set<std::pair<int,int>>{}:required_pages(receivers,basis);
        auto const& needed=selected_pages?*selected_pages:local_pages;
        if(needed.size()>32)return false;
        for(auto& p:pages)if(p.hash && needed.count({p.x,p.y}))p.used=epoch;
        std::vector<std::array<float,4>> local_bounds;
        if(!prepared || !prepared->matches(casters,basis)){
            local_bounds.reserve(casters.size());
            for(auto const& c:casters)local_bounds.push_back(projected(c.bounds,c.offset));
        }
        auto const& caster_bounds=prepared && prepared->matches(casters,basis)?prepared->bounds:local_bounds;
        bool pipeline_ready=false;
        std::array<std::array<float,4>,64> lookup{};
        for(auto const& key:needed){
            if(cancellation && cancellation->load(std::memory_order_relaxed))return false;
            bool can_reuse=prepared && prepared->matches(casters,basis);
            auto selection=can_reuse?prepared->find(key):nullptr;
            PreparedCasters::Selection local;
            if(!selection){
                local=PreparedCasters::select(casters,caster_bounds,key);
                if(can_reuse)selection=prepared->admit(key,std::move(local));
                if(!selection)selection=&local;
            }
            auto const& selected=selection->indices;auto hash=selection->hash;
            int slot=-1;for(int i=0;i<32;++i)if(pages[i].hash && pages[i].x==key.first && pages[i].y==key.second){slot=i;break;}
            if(slot<0){for(int i=0;i<32;++i)if(pages[i].used!=epoch && (slot<0 || pages[i].used<pages[slot].used))slot=i;}
            if(slot<0)return false;
            auto& page=pages[slot];
            unsigned hash_slot=(unsigned(key.first)*73856093u ^ unsigned(key.second)*19349663u)&63u;
            while(lookup[hash_slot][3]>.5f)hash_slot=(hash_slot+1)&63u;
            lookup[hash_slot]={float(key.first),float(key.second),float(slot),1};
            if(page.hash==hash && page.x==key.first && page.y==key.second){++hits;page.used=epoch;continue;}
            if(!pipeline_ready){
                std::array<ID3D11ShaderResourceView*,128> empty{};
                context->PSSetShaderResources(0,128,empty.data());
                context->IASetInputLayout(layout);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                context->VSSetShader(vertex,nullptr,0);context->VSSetConstantBuffers(0,1,&caster_settings);
                context->RSSetState(raster);D3D11_VIEWPORT viewport={0,0,1024,1024,0,1};context->RSSetViewports(1,&viewport);
                context->OMSetDepthStencilState(nullptr,0);context->OMSetBlendState(maximum,nullptr,0xffffffff);
                pipeline_ready=true;
            }
            // Publish identity only after the entire source field completes.
            page.hash=0;float clear[4]={-1e6f,-1e6f,-1e6f,-1e6f};
            context->ClearRenderTargetView(targets[slot],clear);context->OMSetRenderTargets(1,&targets[slot],nullptr);
            for(auto i:selected){
                if(cancellation && cancellation->load(std::memory_order_relaxed))return false;
                auto const& c=casters[i];float settings[20]={};std::copy(basis.begin(),basis.end(),settings);
                settings[12]=float(key.first);settings[13]=float(key.second);
                std::copy(c.offset,c.offset+3,settings+16);
                context->UpdateSubresource(caster_settings,0,nullptr,settings,0,0);
                bool alpha=bind(c.binding==0xffffffffu?c.layer:c.binding);context->PSSetShader(alpha?cutout:opaque,nullptr,0);
                context->IASetInputLayout(c.stride==92?natural_layout:c.stride==48?feature_layout:layout);
                UINT stride=c.stride,offset=0;context->IASetVertexBuffers(0,1,&c.vertices,&stride,&offset);
                context->IASetIndexBuffer(c.indices,c.index_format,0);context->DrawIndexed(c.count,0,0);++draws;
            }
            page={key.first,key.second,hash,epoch};++rebuilt;
        }
        if(pipeline_ready)context->OMSetRenderTargets(0,nullptr,nullptr);
        context->UpdateSubresource(table,0,nullptr,lookup.data(),0,0);
        return true;
    }
};
} }
