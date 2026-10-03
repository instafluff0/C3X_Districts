#pragma once
#include <array>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <utility>
#include <vector>
#include <d3d11_1.h>

namespace c3x_renderer { namespace render_core {
// One prepared submission per water-dependent layer. Its constants have their
// own immutable pages: offsets into DrawParameterStream's dynamic ring cannot
// survive another pass. Indices borrow the current visibility generation only.
// At most 64 batches exist across all layers: <=4 MiB GPU constants, plus
// bounded CPU parameter/order arrays. No mesh/material resources are retained.
template<class Parameters,unsigned Layers> struct PreparedDrawParameters {
    static constexpr unsigned stride=256,limit=256,max_batches=64;
    static_assert(sizeof(Parameters)<=stride,"draw constant range");
    struct Key {
        std::array<std::uint64_t,4> scope{}; // view, visibility, device, content
        std::array<unsigned,4> scene{}; // record count, width, height, profiles
        Parameters viewport{};
        std::array<int,4> rect{};
        std::array<float,2> view{}; // zoom and reflected ordering extent
        bool operator==(Key const& b)const{
            return scope==b.scope && scene==b.scene && rect==b.rect &&
                !std::memcmp(&viewport,&b.viewport,sizeof(viewport)) &&
                !std::memcmp(view.data(),b.view.data(),sizeof(view));
        }
    };
    struct Batch {
        std::array<unsigned,limit> order{},parameter_index{};
        std::array<Parameters,limit> values{};
        unsigned count=0,parameter_count=0;
        ID3D11Buffer* buffer=nullptr;
        Batch()=default;Batch(Batch const&)=delete;Batch& operator=(Batch const&)=delete;
        Batch(Batch&& other)noexcept:order(other.order),parameter_index(other.parameter_index),
            values(other.values),count(other.count),parameter_count(other.parameter_count),buffer(other.buffer){other.buffer=nullptr;}
        ~Batch(){if(buffer)buffer->Release();}
        void bind(ID3D11DeviceContext1* context,unsigned slot,unsigned record)const{
            UINT first=parameter_index[record]*(stride/16),size=stride/16;
            context->VSSetConstantBuffers1(slot,1,&buffer,&first,&size);
        }
    };
    struct Entry {
        Key key{};
        bool valid=false,complete=false,blocked=false;
        std::vector<Batch> batches;
    };
    std::array<Entry,Layers> layers;
    std::array<std::uint64_t,4> scope{};
    unsigned batch_count=0;
    std::uint64_t misses=0,builds=0,reuses=0,fallbacks=0,prepared_records=0,reused_records=0;
    std::uint64_t uploads=0,uploaded_bytes=0;
    double cold_upload_ms=0;
    PreparedDrawParameters()=default;PreparedDrawParameters(PreparedDrawParameters const&)=delete;
    void discard(Entry& entry){
        batch_count-=unsigned(entry.batches.size());
        std::vector<Batch>().swap(entry.batches);entry.valid=entry.complete=entry.blocked=false;
    }
    void clear(){for(auto& entry:layers)discard(entry);scope={};}
    std::size_t gpu_bytes()const{
        std::size_t result=0;
        for(auto const& entry:layers)for(auto const& batch:entry.batches)
            if(batch.buffer)result+=batch.parameter_count*stride;
        return result;
    }
    std::size_t metadata_bytes()const{
        std::size_t result=sizeof(*this);
        for(auto const& entry:layers)result+=entry.batches.capacity()*sizeof(Batch);
        return result;
    }
    Entry* begin(unsigned layer,Key const& key){
        if(scope!=key.scope){clear();scope=key.scope;}
        auto& entry=layers[layer];
        if(entry.valid && entry.key==key){
            if(entry.blocked)return nullptr;
            if(entry.complete){++reuses;return &entry;}
            discard(entry);entry.key=key;entry.valid=true;++builds;return &entry;
        }
        // A changing camera keeps the existing streaming cost. Only a second
        // identical request pays for immutable pages; partial failures retry
        // from a clean entry and blocked admission retries only on a new key.
        discard(entry);entry.key=key;entry.valid=true;++misses;return nullptr;
    }
    Batch const* append(Entry& entry,ID3D11Device* device,
            std::array<unsigned,limit> const& order,unsigned count,
            std::array<Parameters,limit> const& values,
            std::array<unsigned,limit> const& parameter_index,
            Parameters const* parameters,unsigned parameter_count){
        auto fail=[&]()->Batch const*{discard(entry);entry.valid=entry.blocked=true;++fallbacks;return nullptr;};
        if(batch_count==max_batches || !count || count>limit || parameter_count>count)return fail();
        Batch batch;batch.order=order;batch.count=count;batch.values=values;
        batch.parameter_index=parameter_index;batch.parameter_count=parameter_count;
        if(parameter_count){
            auto started=std::chrono::steady_clock::now();
            struct Charge {double& total;std::chrono::steady_clock::time_point started;
                ~Charge(){total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();}
            } charge{cold_upload_ms,started};
            // Each selected nonrigid draw has its exact existing 256-byte range.
            std::array<unsigned char,stride*limit> bytes{};
            for(unsigned i=0;i<parameter_count;++i)std::memcpy(bytes.data()+i*stride,parameters+i,sizeof(Parameters));
            D3D11_BUFFER_DESC desc={};desc.ByteWidth=parameter_count*stride;
            desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            D3D11_SUBRESOURCE_DATA data={};data.pSysMem=bytes.data();
            if(FAILED(device->CreateBuffer(&desc,&data,&batch.buffer)))return fail();
            ++uploads;uploaded_bytes+=desc.ByteWidth;
        }
        try{entry.batches.push_back(std::move(batch));}
        catch(...){return fail();}
        ++batch_count;prepared_records+=count;
        return &entry.batches.back();
    }
};
}}
