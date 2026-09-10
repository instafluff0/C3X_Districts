#pragma once
#include <algorithm>
#include <map>
#include <vector>
#include "world_coast.h"

namespace c3x_renderer { namespace render_core {
// Retain the exact tile-center query and replay its observations into each new
// compiler scope. Topology supports geometry; this cache grants no draw eligibility.
struct CenterShoreCache {
    struct Entry {
        ShoreSample sample{};
        std::vector<std::pair<std::uint64_t,std::uint64_t>> coast;
        std::vector<std::pair<std::size_t,std::uint32_t>> world;
        std::size_t bytes=0;
    };
    static constexpr std::size_t budget=16u*1024u*1024u,entry_limit=8192;
    std::map<std::pair<int,int>,Entry> entries;
    World dimensions{};
    std::size_t bytes=0,hits=0,misses=0;
    void clear(){entries.clear();bytes=hits=misses=0;}
    template<class ObserveWorld,class ObserveCoast>
    ShoreSample get(WorldCoast const& source,int x,int y,ObserveWorld world,ObserveCoast coast){
        auto d=source.world().dimensions();
        if(d.width!=dimensions.width || d.height!=dimensions.height || d.wrap_x!=dimensions.wrap_x || d.wrap_y!=dimensions.wrap_y){clear();dimensions=d;}
        auto found=entries.find({x,y});
        if(found!=entries.end()){
            auto const& entry=found->second;bool valid=true;
            for(auto const& item:entry.world)if(source.world().at(item.first)!=item.second){valid=false;break;}
            if(valid)for(auto const& item:entry.coast)if(source.node_revision(item.first)!=item.second){valid=false;break;}
            if(valid){
                ++hits;for(auto const& item:entry.world)world(item.first,item.second);
                for(auto const& item:entry.coast)coast(item.first,item.second);
                return entry.sample;
            }
            bytes-=entry.bytes;entries.erase(found);
        }
        ++misses;Entry entry;bool cacheable=true;
        auto observe_world=[&](std::size_t index,std::uint32_t value){
            world(index,value);if(entry.world.size()<65536)entry.world.emplace_back(index,value);else cacheable=false;
        };
        auto observe_coast=[&](std::uint64_t index,std::uint64_t value){
            coast(index,value);if(entry.coast.size()<65536)entry.coast.emplace_back(index,value);else cacheable=false;
        };
        entry.sample=source.sample({float(x+y)*.5f+.5f,float(x-y)*.5f+.5f},observe_coast,observe_world);
        auto sample=entry.sample;
        if(cacheable)try {
            std::sort(entry.world.begin(),entry.world.end());entry.world.erase(std::unique(entry.world.begin(),entry.world.end()),entry.world.end());
            std::sort(entry.coast.begin(),entry.coast.end());entry.coast.erase(std::unique(entry.coast.begin(),entry.coast.end()),entry.coast.end());
            entry.bytes=sizeof(Entry)+96+entry.world.capacity()*sizeof(entry.world[0])+entry.coast.capacity()*sizeof(entry.coast[0]);
            if(entry.bytes<=budget){
                if(entries.size()>=entry_limit || bytes>budget-entry.bytes){entries.clear();bytes=0;}
                auto added=entry.bytes;
                if(entries.emplace(std::make_pair(x,y),std::move(entry)).second)bytes+=added;
            }
        }catch(...){} // Admission is optional; the exact computed result survives.
        return sample;
    }
};
} }
