#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <vector>
#include <limits>
#include <d3d11.h>

namespace c3x_renderer { namespace render_core {
// Ordered source-mesh packets use the existing resident rigid shader. Each
// vertex supplies its placement selection rather than an instance-step input.
// Packets own compact placement snapshots, never an old scene/front lease.
template<class Key> class OrderedRigidSubmission {
public:
    static constexpr unsigned record_limit=256,entry_limit=16384;
    static constexpr std::size_t budget=64u*1024u*1024u,page_bytes_limit=8u*1024u*1024u;
    struct Input {
        Key key{};
        void const* vertices=nullptr;
        unsigned vertex_count=0;
        std::uint32_t const* indices=nullptr;
        unsigned index_count=0,placement=0;
    };
private:
    struct Ledger {
        std::atomic<std::size_t> bytes{0},cpu{0},gpu{0},peak{0};
        std::size_t limit=0;
    };
    struct Charge {
        std::shared_ptr<Ledger> ledger;
        std::size_t cpu=0,gpu=0;
        explicit Charge(std::shared_ptr<Ledger> value):ledger(std::move(value)){}
        bool resize(std::size_t c,std::size_t g){
            auto before=cpu+gpu,after=c+g;
            if(after>before){
                auto extra=after-before,live=ledger->bytes.load();
                do{if(extra>ledger->limit || live>ledger->limit-extra)return false;}
                while(!ledger->bytes.compare_exchange_weak(live,live+extra));
                auto high=ledger->peak.load();
                while(high<live+extra && !ledger->peak.compare_exchange_weak(high,live+extra)){}
            }else ledger->bytes.fetch_sub(before-after);
            if(c>cpu)ledger->cpu.fetch_add(c-cpu);else ledger->cpu.fetch_sub(cpu-c);
            if(g>gpu)ledger->gpu.fetch_add(g-gpu);else ledger->gpu.fetch_sub(gpu-g);
            cpu=c;gpu=g;return true;
        }
        ~Charge(){ledger->bytes.fetch_sub(cpu+gpu);ledger->cpu.fetch_sub(cpu);ledger->gpu.fetch_sub(gpu);}
    };
public:
    struct Item {Key key{};unsigned first=0,count=0;};
    struct Page {
        ID3D11Buffer *geometry=nullptr,*placements=nullptr;
        ID3D11ShaderResourceView* placement_view=nullptr;
        unsigned index_offset=0,selection_offset=0;
        std::vector<Item> items;
        std::uint64_t used=0;
    private:
        friend class OrderedRigidSubmission;
        Charge charge;
        explicit Page(std::shared_ptr<Ledger> ledger):charge(std::move(ledger)){}
    public:
        ~Page(){if(placement_view)placement_view->Release();if(placements)placements->Release();if(geometry)geometry->Release();
            std::vector<Item>().swap(items);}
    };
    using Lease=std::shared_ptr<Page>;
    struct Range {
        Lease page;
        unsigned first=0,count=0;
        explicit operator bool()const{return page && count;}
        bool contiguous(Range const& next)const{return page==next.page && first+count==next.first;}
    };
private:
    // Exact-key entries own the packet. A packet contains only value keys and
    // copied resources, so this ownership cannot retain a scene/front or cycle.
    struct Location {Lease page;unsigned ordinal=0;};
    std::shared_ptr<Ledger> ledger=std::make_shared<Ledger>();
    std::map<Key,Location> entries;
    std::array<std::uint64_t,3> scope{};
    std::uint64_t serial=0;
    unsigned pages=0;
    // Conservative map-node/control overhead is charged beside exact vector
    // capacities. Retired pages keep this charge until their last draw lease.
    static constexpr std::size_t node_bytes=sizeof(Key)+sizeof(Location)+96;
    std::size_t owner_bytes()const{return sizeof(*this)+sizeof(Ledger)+64;}
    bool room(std::size_t bytes,unsigned count)const{
        if(bytes>ledger->limit || count>entry_limit)return false;
        // Capacity refusal preserves already prepared subranges. Cycling an
        // oversized visible set through an LRU would repack it every request.
        // Scope retirement is the only replacement boundary.
        return ledger->bytes.load()<=ledger->limit-bytes && count<=entry_limit-entries.size();
    }
public:
    std::uint64_t builds=0,reuses=0,refusals=0,draws=0,drawn_records=0,uploaded_bytes=0,placement_copies=0;
    OrderedRigidSubmission(){ledger->limit=budget-owner_bytes();}
    OrderedRigidSubmission(OrderedRigidSubmission const&)=delete;
    OrderedRigidSubmission& operator=(OrderedRigidSubmission const&)=delete;
    void select_scope(std::array<std::uint64_t,3> const& value){if(scope!=value){clear();scope=value;}}
    void clear(){entries.clear();pages=0;scope={};}
    std::size_t bytes()const{return owner_bytes()+ledger->bytes.load();}
    std::size_t gpu_bytes()const{return ledger->gpu.load();}
    std::size_t metadata_bytes()const{return owner_bytes()+ledger->cpu.load();}
    std::size_t peak_bytes()const{return owner_bytes()+ledger->peak.load();}
    unsigned page_count()const{return pages;}
    bool can_append(std::size_t geometry_bytes,unsigned count)const{
        if(!count || count>record_limit || geometry_bytes>page_bytes_limit)return false;
        auto metadata=sizeof(Page)+64+count*(sizeof(Item)+node_bytes);
        return room(metadata+geometry_bytes*2+std::size_t(count)*64,count);
    }
    Range find(Key const& key){
        auto found=entries.find(key);if(found==entries.end())return {};
        auto page=found->second.page;
        auto const& item=page->items[found->second.ordinal];page->used=++serial;++reuses;
        return {std::move(page),item.first,item.count};
    }
    // Admission publishes all exact keys together. Failure leaves caller-owned
    // streaming draws valid; allocations/copy destinations retire through RAII.
    Lease append(ID3D11Device* device,ID3D11DeviceContext* context,
            ID3D11Buffer* source_placements,unsigned source_records,Input const* inputs,unsigned count){
        auto refuse=[&]()->Lease{++refusals;return {};};
        if(!device || !context || !source_placements || !inputs || !count || count>record_limit)return refuse();
        std::size_t vertices=0,indices=0;
        for(unsigned n=0;n<count;++n){auto const& input=inputs[n];
            if(!input.vertices || !input.indices || !input.vertex_count || !input.index_count ||
                    input.index_count%3 || input.placement>=source_records)return refuse();
            vertices+=input.vertex_count;indices+=input.index_count;
            if(vertices>page_bytes_limit/36 || indices>page_bytes_limit/4)return refuse();
        }
        auto geometry_bytes=vertices*36+indices*4,placement_bytes=std::size_t(count)*64;
        auto metadata=sizeof(Page)+64+count*(sizeof(Item)+node_bytes);
        if(!can_append(geometry_bytes,count))return refuse();
        for(unsigned n=0;n<count;++n)for(unsigned i=0;i<inputs[n].index_count;++i)
            if(inputs[n].indices[i]>=inputs[n].vertex_count)return refuse();
        try{
            Lease page(new Page(ledger));
            if(!page->charge.resize(metadata,geometry_bytes+placement_bytes))return refuse();
            Charge staging(ledger);if(!staging.resize(geometry_bytes,0))return refuse();
            std::vector<unsigned char> bytes(geometry_bytes);
            if(!staging.resize(bytes.capacity(),0))return refuse();
            page->items.reserve(count);
            metadata=sizeof(Page)+64+page->items.capacity()*sizeof(Item)+count*node_bytes;
            if(!page->charge.resize(metadata,geometry_bytes+placement_bytes))return refuse();
            page->index_offset=unsigned(vertices*32);page->selection_offset=unsigned(vertices*32+indices*4);
            unsigned vertex=0,index=0;
            for(unsigned n=0;n<count;++n){auto const& input=inputs[n];
                std::memcpy(bytes.data()+std::size_t(vertex)*32,input.vertices,std::size_t(input.vertex_count)*32);
                for(unsigned i=0;i<input.index_count;++i){auto value=input.indices[i]+vertex;
                    std::memcpy(bytes.data()+page->index_offset+std::size_t(index+i)*4,&value,4);}
                for(unsigned i=0;i<input.vertex_count;++i)std::memcpy(bytes.data()+page->selection_offset+std::size_t(vertex+i)*4,&n,4);
                page->items.push_back({input.key,index,input.index_count});vertex+=input.vertex_count;index+=input.index_count;
            }
            D3D11_BUFFER_DESC desc{};desc.ByteWidth=unsigned(geometry_bytes);desc.Usage=D3D11_USAGE_IMMUTABLE;
            desc.BindFlags=D3D11_BIND_VERTEX_BUFFER|D3D11_BIND_INDEX_BUFFER;
            D3D11_SUBRESOURCE_DATA data{};data.pSysMem=bytes.data();
            if(FAILED(device->CreateBuffer(&desc,&data,&page->geometry)))return refuse();
            desc={};desc.ByteWidth=unsigned(placement_bytes);desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;desc.StructureByteStride=64;
            if(FAILED(device->CreateBuffer(&desc,nullptr,&page->placements)) ||
                    FAILED(device->CreateShaderResourceView(page->placements,nullptr,&page->placement_view)))return refuse();
            for(unsigned n=0;n<count;++n){D3D11_BOX box={inputs[n].placement*64,0,0,(inputs[n].placement+1)*64,1,1};
                context->CopySubresourceRegion(page->placements,0,n*64,0,0,source_placements,0,&box);}
            // A duplicate key deliberately resolves its first exact occurrence.
            // Drawing that range again still preserves duplicate primitives.
            unsigned inserted=0;
            try{for(unsigned n=0;n<count;++n)inserted+=entries.emplace(inputs[n].key,Location{page,n}).second;}
            catch(...){for(auto const& item:page->items){auto found=entries.find(item.key);
                if(found!=entries.end() && found->second.page==page)entries.erase(found);}throw;}
            if(!inserted)return refuse();
            page->used=++serial;++pages;++builds;uploaded_bytes+=geometry_bytes;placement_copies+=count;
            return page;
        }catch(...){return refuse();}
    }
    template<class Callback> unsigned issue(ID3D11DeviceContext* context,ID3D11InputLayout* layout,
            ID3D11VertexShader* shader,Range const* ranges,unsigned count,Callback callback){
        unsigned n=0;
        while(n<count){if(!ranges[n]){++n;continue;}
            auto const& first=ranges[n];unsigned end=n+1,indices=first.count;
            for(;end<count && ranges[end-1].contiguous(ranges[end]);++end)indices+=ranges[end].count;
            auto const& page=*first.page;ID3D11Buffer* buffers[]={page.geometry,page.geometry};
            UINT strides[]={32,4},offsets[]={0,page.selection_offset};
            context->IASetInputLayout(layout);context->VSSetShader(shader,nullptr,0);
            context->IASetVertexBuffers(0,2,buffers,strides,offsets);
            context->IASetIndexBuffer(page.geometry,DXGI_FORMAT_R32_UINT,page.index_offset);
            context->VSSetShaderResources(15,1,&page.placement_view);
            context->DrawIndexed(indices,first.first,0);
            ++draws;drawn_records+=end-n;callback(indices,end-n);n=end;
        }
        // The cache lease accounts residency; context bindings must not keep
        // a retired page alive after that lease is released at a scope change.
        ID3D11Buffer* empty_buffers[]={nullptr,nullptr};UINT zeros[]={0,0};
        context->IASetVertexBuffers(0,2,empty_buffers,zeros,zeros);
        context->IASetIndexBuffer(nullptr,DXGI_FORMAT_R32_UINT,0);
        ID3D11ShaderResourceView* empty_view=nullptr;context->VSSetShaderResources(15,1,&empty_view);
        return n;
    }
};
}}
