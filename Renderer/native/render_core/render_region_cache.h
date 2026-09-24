#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace c3x_renderer { namespace render_core {
using RenderRegionKey=std::vector<std::uint64_t>;
struct RenderRegionKeyHash {
    std::size_t operator()(RenderRegionKey const& key) const noexcept {
        std::uint64_t hash=14695981039346656037ull;
        for(auto word:key){hash^=word;hash*=1099511628211ull;}
        return std::size_t(hash);
    }
};

// Serialized worker ownership. Entries own only completed render resources;
// keys contain values, never borrowed scene pointers. D3D retains references
// for already queued copies when a cache owner is evicted.
template<class Resource,std::size_t maximum_gpu_bytes=256u*1024u*1024u> class RenderRegionCache {
public:
    static constexpr std::size_t gpu_budget=maximum_gpu_bytes;
    static constexpr std::size_t metadata_budget=96u*1024u*1024u;
    static constexpr std::size_t key_words_limit=16384;
    static constexpr std::size_t entry_limit=4096;
    struct Entry {
        Resource* image=nullptr;
        std::size_t gpu_bytes=0,metadata_bytes=0;
        std::uint64_t age=0;
        std::list<RenderRegionKey const*>::iterator recent;
        Entry()=default;
        Entry(Entry const&)=delete;
        Entry& operator=(Entry const&)=delete;
        ~Entry(){if(image)image->Release();}
    };
    // Hash once for lookup. A full scene-value key still proves equality;
    // retaining pointers to node keys keeps LRU updates and eviction constant
    // time without duplicating their often-large dependency vectors.
    std::unordered_map<RenderRegionKey,Entry,RenderRegionKeyHash> entries;
    std::list<RenderRegionKey const*> recent;
    std::size_t gpu_bytes=0,metadata_bytes=0,metadata_limit=metadata_budget,gpu_limit=gpu_budget;
    std::uint64_t age=0,evictions=0;
    void evict_oldest(Resource** recycled=nullptr,std::size_t recycled_bytes=0){
        auto oldest=entries.find(*recent.front());
        gpu_bytes-=oldest->second.gpu_bytes;metadata_bytes-=oldest->second.metadata_bytes;
        if(recycled && !*recycled && oldest->second.gpu_bytes==recycled_bytes){
            *recycled=oldest->second.image;oldest->second.image=nullptr;
        }
        recent.pop_front();entries.erase(oldest);++evictions;
    }
    void set_gpu_limit(std::size_t limit){
        gpu_limit=std::min(limit,gpu_budget);
        while(!entries.empty() && gpu_bytes>gpu_limit)evict_oldest();
    }
    Resource* find(RenderRegionKey const& key){
        auto found=entries.find(key);
        if(found==entries.end())return nullptr;
        found->second.age=++age;recent.splice(recent.end(),recent,found->second.recent);
        return found->second.image;
    }
    static std::size_t metadata_size(RenderRegionKey const& key){
        // Include retained capacity and a conservative map-node allowance.
        return key.capacity()*sizeof(key[0])+sizeof(Entry)+sizeof(RenderRegionKey)+64;
    }
    // May transfer one evicted allocation to the caller for ordered reuse.
    // Matching byte size is not format compatibility; the caller checks the
    // resource description and releases any allocation it cannot reuse.
    bool make_room(RenderRegionKey const& key,std::size_t bytes,Resource** recycled=nullptr){
        auto metadata=metadata_size(key);
        if(key.empty() || key.size()>key_words_limit || bytes>gpu_limit || metadata>metadata_limit)return false;
        while(!entries.empty() && (gpu_bytes+bytes>gpu_limit ||
              metadata_bytes+metadata>metadata_limit || entries.size()>=entry_limit))
            evict_oldest(recycled,bytes);
        return gpu_bytes+bytes<=gpu_limit && metadata_bytes+metadata<=metadata_limit;
    }
    // Takes ownership only on success. The caller releases an unadmitted image.
    bool insert(RenderRegionKey key,Resource* image,std::size_t bytes){
        if(!image || entries.find(key)!=entries.end() || !make_room(key,bytes))return false;
        auto metadata=metadata_size(key);
        auto result=entries.try_emplace(std::move(key));
        if(!result.second)return false;
        auto& entry=result.first->second;entry.image=image;entry.gpu_bytes=bytes;
        recent.push_back(&result.first->first);entry.recent=std::prev(recent.end());
        entry.metadata_bytes=metadata;entry.age=++age;
        gpu_bytes+=bytes;metadata_bytes+=metadata;return true;
    }
    void clear(){recent.clear();entries.clear();gpu_bytes=metadata_bytes=0;age=evictions=0;}
};
} }
