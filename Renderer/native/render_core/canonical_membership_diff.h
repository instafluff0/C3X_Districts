#pragma once
#include "foreground_selection.h"
#include "captured_scene.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace c3x_renderer::render_core {
// An exact diff of the single retained occurrence membership, never another
// camera cache. Coordinates remain unwrapped: seam copies are distinct draws.
// Native dynamic/visibility capture continues independently of this static diff.
struct CanonicalMembershipDiff {
    static constexpr unsigned absent=~0u;
    std::vector<unsigned> previous;
    std::vector<bool> keep;
    int translation_x=0,translation_y=0;
    unsigned required=0,entering=0,leaving=0;
    bool ordered=true;
    static std::uint64_t occurrence(int x,int y){
        return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);
    }
    template<class Same,class Admitted>
    bool build(std::vector<c3x_renderer_tile_v1> const& cached,
            c3x_renderer_frame_v1 const& frame,ForegroundSelection const& selection,Same same,Admitted admitted){
        previous.assign(frame.tile_count,absent);keep.assign(cached.size(),false);
        required=entering=leaving=0;translation_x=translation_y=0;ordered=true;
        if(cached.empty() || !frame.tile_count || frame.tile_count>CapturedScene::occurrence_limit)return false;
        std::unordered_map<std::uint64_t,unsigned> index;index.reserve(cached.size());
        unsigned selected=0;
        for(unsigned i=0;i<cached.size();++i)if(admitted(i)){
            if(!index.emplace(occurrence(cached[i].tile_x,cached[i].tile_y),i).second)return false;
            ++selected;
        }
        bool affine=false;unsigned prior_index=absent;
        for(unsigned i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
            if(!selection.selects(tile))continue;++required;
            auto found=index.find(occurrence(tile.tile_x,tile.tile_y));
            if(found==index.end()){++entering;continue;}
            auto const& prior=cached[found->second];
            if(!same(prior,tile)){++entering;continue;}
            auto dx=std::int64_t(tile.anchor_x)-prior.anchor_x,dy=std::int64_t(tile.anchor_y)-prior.anchor_y;
            if(dx<INT32_MIN || dx>INT32_MAX || dy<INT32_MIN || dy>INT32_MAX)return false;
            if(!affine){translation_x=int(dx);translation_y=int(dy);affine=true;}
            else if(dx!=translation_x || dy!=translation_y)return false;
            if(keep[found->second])return false; // No duplicate selected occurrence.
            if(prior_index!=absent && found->second<prior_index)ordered=false;
            prior_index=found->second;
            keep[found->second]=true;previous[i]=found->second;
        }
        leaving=selected-(required-entering);
        return affine && required;
    }
    // Alpha draws keep native occurrence order even when every owner matches.
    bool covered()const{return required && !entering && ordered;}
    void reject(unsigned current){
        auto old=previous[current];if(old==absent)return;
        previous[current]=absent;keep[old]=false;++entering;++leaving;
    }
};
}
