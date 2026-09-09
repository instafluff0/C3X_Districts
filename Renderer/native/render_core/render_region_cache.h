#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace c3x_renderer { namespace render_core {
using RenderRegionKey=std::vector<std::uint64_t>;

// Serialized worker ownership. Entries own only completed render resources;
// keys contain values, never borrowed scene pointers. D3D retains references
// for already queued copies when a cache owner is evicted.
template<class Resource> class RenderRegionCache {
public:
    static constexpr std::size_t gpu_budget=256u*1024u*1024u;
    static constexpr std::size_t metadata_budget=96u*1024u*1024u;
    static constexpr std::size_t key_words_limit=16384;
    static constexpr std::size_t entry_limit=4096;
    struct Entry {
        Resource* image=nullptr;
        std::size_t gpu_bytes=0,metadata_bytes=0;
        std::uint64_t age=0;
        Entry()=default;
        Entry(Entry const&)=delete;
        Entry& operator=(Entry const&)=delete;
        ~Entry(){if(image)image->Release();}
    };
    std::map<RenderRegionKey,Entry> entries;
    std::size_t gpu_bytes=0,metadata_bytes=0,metadata_limit=metadata_budget;
    std::uint64_t age=0,evictions=0;
    Resource* find(RenderRegionKey const& key){
        auto found=entries.find(key);
        if(found==entries.end())return nullptr;
        found->second.age=++age;return found->second.image;
    }
    static std::size_t metadata_size(RenderRegionKey const& key){
        // Include retained capacity and a conservative map-node allowance.
        return key.capacity()*sizeof(key[0])+sizeof(Entry)+sizeof(RenderRegionKey)+64;
    }
    bool make_room(RenderRegionKey const& key,std::size_t bytes){
        auto metadata=metadata_size(key);
        if(key.empty() || key.size()>key_words_limit || bytes>gpu_budget || metadata>metadata_limit)return false;
        while(!entries.empty() && (gpu_bytes+bytes>gpu_budget ||
              metadata_bytes+metadata>metadata_limit || entries.size()>=entry_limit)){
            auto oldest=std::min_element(entries.begin(),entries.end(),[](auto const& a,auto const& b){return a.second.age<b.second.age;});
            gpu_bytes-=oldest->second.gpu_bytes;metadata_bytes-=oldest->second.metadata_bytes;
            entries.erase(oldest);++evictions;
        }
        return gpu_bytes+bytes<=gpu_budget && metadata_bytes+metadata<=metadata_limit;
    }
    // Takes ownership only on success. The caller releases an unadmitted image.
    bool insert(RenderRegionKey key,Resource* image,std::size_t bytes){
        if(!image || entries.find(key)!=entries.end() || !make_room(key,bytes))return false;
        auto metadata=metadata_size(key);
        auto result=entries.try_emplace(std::move(key));
        if(!result.second)return false;
        auto& entry=result.first->second;entry.image=image;entry.gpu_bytes=bytes;
        entry.metadata_bytes=metadata;entry.age=++age;
        gpu_bytes+=bytes;metadata_bytes+=metadata;return true;
    }
    void clear(){entries.clear();gpu_bytes=metadata_bytes=0;age=evictions=0;}
};
} }
