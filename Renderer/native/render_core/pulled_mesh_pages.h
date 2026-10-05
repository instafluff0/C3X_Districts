#pragma once
#include <d3d11.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Ordered submission of many small non-rigid meshes (water, river, bed and
// terrain chunks are one immutable buffer per tile) with one draw per
// contiguous run. D3D11 has no multi-draw, and each per-tile DrawIndexed cost
// about 3 us of driver/translation time, so draw count, not shading, set the
// water pass cost (2.6k draws at 1x, 6.3k at 0.5x).
//
// Pages hold GPU-side copies of the original vertex bytes and index words in
// raw buffers plus one record per occurrence (translation, natural projection,
// projection kind). A generated vertex shader pulls each vertex through its
// record, so primitive order and per-occurrence constants are unchanged.
// Entries are keyed by content identity and placement, which are camera
// independent; scrolling reuses existing pages and only appends new edges.
class PulledMeshPages {
public:
    // HLSL-visible record (C3XPulledRecord); 64 bytes.
    struct Record {
        std::uint32_t first=0,count=0,vertex_byte=0,index_byte=0;
        std::uint32_t index16=0,stride=0,vertices=0,reserved1=0;
        float translation[2]={},kind=0,reserved2=0;
        float projection[4]={};
    };
    static_assert(sizeof(Record)==64,"pulled record ABI");
    struct Source {
        ID3D11Buffer* buffer=nullptr;ID3D11Buffer* indices=nullptr;
        std::uint64_t version=0;
        unsigned vertex_offset=0,index_offset=0,index_count=0,vertex_count=0,stride=0;
        bool index16=false;
        int tx=0,ty=0;float projection[4]={};unsigned kind=0;
    };
    struct Page {
        Microsoft::WRL::ComPtr<ID3D11Buffer> vertices,indices,records;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> vertex_view,index_view,record_view;
        std::vector<Record> table;std::uint64_t used=0;std::size_t bytes=0;unsigned mapped=0;
    };
    struct Range {Page* page=nullptr;unsigned first_record=0,records=0,first_vertex=0,vertices=0;};
    static constexpr unsigned page_records=256;
    static constexpr std::size_t page_bytes=16u*1024u*1024u,budget=512u*1024u*1024u;
    std::uint64_t builds=0,reuses=0,refusals=0,draws=0,drawn_records=0,copied_bytes=0,retired=0,rebuilds=0;
private:
    using Key=std::tuple<ID3D11Buffer*,ID3D11Buffer*,std::uint64_t,unsigned,unsigned,unsigned,unsigned,unsigned,bool,
        int,int,std::uint32_t,std::uint32_t,std::uint32_t,std::uint32_t,unsigned>;
    struct Location {std::shared_ptr<Page> page;unsigned slot=0;};
    std::map<Key,Location> entries;
    std::uint64_t frame=0;std::size_t resident=0;
    static std::uint32_t bits(float value){std::uint32_t v=0;std::memcpy(&v,&value,4);return v;}
    static Key key(Source const& s){
        return Key{s.buffer,s.indices,s.version,s.vertex_offset,s.index_offset,s.index_count,s.vertex_count,s.stride,s.index16,
            s.tx,s.ty,bits(s.projection[0]),bits(s.projection[1]),bits(s.projection[2]),bits(s.projection[3]),s.kind};
    }
    static std::size_t align4(std::size_t n){return (n+3)&~std::size_t(3);}
    static bool raw_buffer(ID3D11Device* device,std::size_t bytes,Microsoft::WRL::ComPtr<ID3D11Buffer>& buffer,
            Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>& view){
        D3D11_BUFFER_DESC desc{};desc.ByteWidth=UINT(std::max<std::size_t>(4,align4(bytes)));desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        if(FAILED(device->CreateBuffer(&desc,nullptr,&buffer)))return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC v{};v.Format=DXGI_FORMAT_R32_TYPELESS;v.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;
        v.BufferEx.NumElements=desc.ByteWidth/4;v.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
        return SUCCEEDED(device->CreateShaderResourceView(buffer.Get(),&v,&view));
    }
    void release(Location& location){
        if(location.page&&!--location.page->mapped)resident-=location.page->bytes;
        location.page.reset();
    }
    std::shared_ptr<Page> build(ID3D11Device* device,ID3D11DeviceContext* context,Source const* const* sources,unsigned count){
        std::size_t vertex_bytes=0,index_bytes=0;
        for(unsigned n=0;n<count;++n){auto const& s=*sources[n];
            vertex_bytes+=std::size_t(s.vertex_count)*s.stride;index_bytes+=align4(std::size_t(s.index_count)*(s.index16?2:4));}
        auto page=std::make_shared<Page>();page->table.resize(count);
        D3D11_BUFFER_DESC desc{};desc.ByteWidth=UINT(count*sizeof(Record));desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;desc.StructureByteStride=sizeof(Record);
        std::uint32_t first=0,vertex_at=0,index_at=0;
        for(unsigned n=0;n<count;++n){auto const& s=*sources[n];auto& r=page->table[n];
            r.first=first;r.count=s.index_count;r.vertex_byte=vertex_at;r.index_byte=index_at;
            r.index16=s.index16?1u:0u;r.stride=s.stride;r.vertices=s.vertex_count;r.translation[0]=float(s.tx);r.translation[1]=float(s.ty);
            r.kind=float(s.kind);std::copy(std::begin(s.projection),std::end(s.projection),r.projection);
            first+=s.index_count;vertex_at+=std::uint32_t(std::size_t(s.vertex_count)*s.stride);
            index_at+=std::uint32_t(align4(std::size_t(s.index_count)*(s.index16?2:4)));}
        D3D11_SUBRESOURCE_DATA data{};data.pSysMem=page->table.data();
        if(!raw_buffer(device,vertex_bytes,page->vertices,page->vertex_view)||
           !raw_buffer(device,index_bytes,page->indices,page->index_view)||
           FAILED(device->CreateBuffer(&desc,&data,&page->records)))return {};
        D3D11_SHADER_RESOURCE_VIEW_DESC v{};v.Format=DXGI_FORMAT_UNKNOWN;v.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;
        v.Buffer.NumElements=count;
        if(FAILED(device->CreateShaderResourceView(page->records.Get(),&v,&page->record_view)))return {};
        for(unsigned n=0;n<count;++n){auto const& s=*sources[n];auto const& r=page->table[n];
            D3D11_BOX vertices{s.vertex_offset,0,0,s.vertex_offset+UINT(std::size_t(s.vertex_count)*s.stride),1,1};
            context->CopySubresourceRegion(page->vertices.Get(),0,r.vertex_byte,0,0,s.buffer,0,&vertices);
            D3D11_BOX indices{s.index_offset,0,0,s.index_offset+UINT(std::size_t(s.index_count)*(s.index16?2:4)),1,1};
            context->CopySubresourceRegion(page->indices.Get(),0,r.index_byte,0,0,s.indices,0,&indices);
        }
        page->bytes=align4(vertex_bytes)+align4(index_bytes)+count*sizeof(Record);
        copied_bytes+=vertex_bytes+index_bytes;++builds;return page;
    }
    void trim(){
        if(resident<=budget)return;
        // Retire whole pages that this frame does not use, oldest first.
        std::vector<std::pair<std::uint64_t,Page*>> candidates;
        for(auto const& e:entries)if(e.second.page->used!=frame)candidates.push_back({e.second.page->used,e.second.page.get()});
        std::sort(candidates.begin(),candidates.end());
        for(auto const& c:candidates){
            if(resident<=budget*3/4)break;
            for(auto i=entries.begin();i!=entries.end();){
                if(i->second.page.get()==c.second){release(i->second);i=entries.erase(i);++retired;}else ++i;}
        }
    }
public:
    void begin_frame(){++frame;}
    void clear(){entries.clear();resident=0;}
    std::size_t bytes()const{return resident;}
    std::size_t entry_count()const{return entries.size();}
    // Resolve ordered sources into contiguous page ranges, building pages for
    // sources not yet resident. Returns false on refusal; the caller keeps its
    // ordinary per-record path for this run.
    bool resolve(ID3D11Device* device,ID3D11DeviceContext* context,Source const* sources,unsigned count,
            std::vector<Range>& ranges,unsigned fragment_limit=48){
        ranges.clear();if(!count)return true;
        for(unsigned n=0;n<count;++n){auto const& s=sources[n];
            if(!s.buffer||!s.indices||!s.vertex_count||!s.index_count||s.index_count%3||!s.stride||s.stride%4||
               std::size_t(s.vertex_count)*s.stride>page_bytes){++refusals;return false;}}
        std::vector<Location*> located(count,nullptr);
        auto lookup=[&]{
            for(unsigned n=0;n<count;++n){auto found=entries.find(key(sources[n]));located[n]=found==entries.end()?nullptr:&found->second;}
        };
        auto fragments=[&]{unsigned runs=0;for(unsigned n=0;n<count;++n)
            if(!n||!located[n]||!located[n-1]||located[n]->page!=located[n-1]->page||located[n]->slot!=located[n-1]->slot+1)++runs;return runs;};
        lookup();
        // Scrolling shifts retained rows and adds edge records; those stay
        // drawable as several ranges. Only heavy fragmentation repacks the
        // run in its current order (a bounded GPU-side copy), so continuous
        // scrolling does not recopy geometry every step.
        bool repack=fragments()>fragment_limit;
        if(repack)++rebuilds;
        std::vector<Source const*> missing;
        for(unsigned n=0;n<count;++n)if(repack||!located[n])missing.push_back(sources+n);
        std::size_t bytes=0;std::vector<Source const*> group;
        auto flush=[&]{
            if(group.empty())return true;
            auto page=build(device,context,group.data(),unsigned(group.size()));
            if(!page){++refusals;return false;}
            for(unsigned slot=0;slot<group.size();++slot){
                auto& location=entries[key(*group[slot])];release(location);
                location.page=page;location.slot=slot;++page->mapped;
            }
            resident+=page->bytes;group.clear();bytes=0;return true;
        };
        for(auto* s:missing){
            auto need=std::size_t(s->vertex_count)*s->stride+align4(std::size_t(s->index_count)*(s->index16?2:4));
            if(group.size()==page_records||bytes+need>page_bytes){if(!flush())return false;}
            group.push_back(s);bytes+=need;
        }
        if(!flush())return false;
        if(!missing.empty())lookup();
        reuses+=count-unsigned(missing.size());
        for(unsigned n=0;n<count;++n){
            if(!located[n]){++refusals;ranges.clear();return false;}
            auto* page=located[n]->page.get();auto slot=located[n]->slot;page->used=frame;
            auto const& record=page->table[slot];
            if(!ranges.empty()&&ranges.back().page==page&&ranges.back().first_record+ranges.back().records==slot){
                ranges.back().records++;ranges.back().vertices+=record.count;
            }else ranges.push_back({page,slot,1,record.first,record.count});
        }
        trim();
        return true;
    }
};
}}
