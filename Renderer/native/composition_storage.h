#pragma once
#include <d3d11.h>
#include <map>
#include <algorithm>
#include <memory>
#include <cstdint>
namespace c3x_gpu_images {
// Track unique reachable texture allocations, including replacement overlap.
// Aliases share one lease. This excludes resources outside composition owners,
// driver padding, shader resources and explicit diagnostic readback staging.
class CompositionStorage {
    struct State;
    struct Entry {std::uint64_t bytes;};
    struct State {std::map<ID3D11Texture2D*,std::weak_ptr<Entry>> entries;
        std::uint64_t current=0,peak=0,allocations=0;};
    std::shared_ptr<State> state=std::make_shared<State>();
public:
    using Lease=std::shared_ptr<Entry>;
    Lease retain(ID3D11Texture2D* texture){
        if(!texture)return {};
        auto found=state->entries.find(texture);
        if(found!=state->entries.end())if(auto lease=found->second.lock())return lease;
        D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
        auto bytes=std::uint64_t(d.Width)*d.Height*4;
        auto owner=state;
        Lease lease(new Entry{bytes},[owner,texture](Entry* entry){
            owner->current-=entry->bytes;owner->entries.erase(texture);delete entry;});
        state->entries[texture]=lease;state->current+=bytes;
        state->peak=std::max(state->peak,state->current);++state->allocations;return lease;
    }
    bool contains(ID3D11Texture2D* texture)const{auto found=state->entries.find(texture);
        return found!=state->entries.end()&&!found->second.expired();}
    std::uint64_t bytes()const{return state->current;}
    std::uint64_t peak()const{return state->peak;}
    std::uint64_t allocations()const{return state->allocations;}
};
}
