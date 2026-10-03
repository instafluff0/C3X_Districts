#pragma once
#include "shadow_sampling_grid.h"
#include <vector>
#include <array>
#include <algorithm>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// Exact current caster proofs own no geometry or former camera generations.
// The fixed atlas slices survive logical-grid movement; only completed pages
// can be selected again. Sampling density/light/wrap/scope are explicit facts.
template<class Key> struct ShadowPageContents {
    using Grid=ShadowSamplingGrid;
    using Context=std::array<std::uint64_t,32>;
    using Inputs=std::array<std::vector<Key>,Grid::max_pages>;
    struct Page {std::array<int,2> coordinate{};Context context{};std::vector<Key> contributors;bool valid=false;};
    std::array<Page,Grid::max_pages> pages{};
    std::array<unsigned,Grid::max_pages> slots{};
    std::array<bool,Grid::max_pages> reused{};
    std::uint64_t hits=0,rebuilt=0,refused=0;
    std::size_t bytes()const{std::size_t result=sizeof(*this);for(auto const& page:pages)result+=page.contributors.capacity()*sizeof(Key);return result;}
    void clear(){for(auto& page:pages){page.valid=false;std::vector<Key>().swap(page.contributors);}slots={};reused={};}
    void select(Grid const& grid,Context const& context,Inputs const& inputs,bool proved){
        reused={};std::array<bool,Grid::max_pages> used{};
        // Claim every exact old page first, so recycling a leaving slice cannot
        // overwrite a later requested page during a shift in either axis.
        for(unsigned logical=0;logical<grid.pages();++logical){
            if(!proved)continue;
            for(unsigned physical=0;physical<Grid::max_pages;++physical){auto const& page=pages[physical];
                if(!used[physical]&&page.valid&&page.coordinate==grid.page(logical)&&page.context==context&&page.contributors==inputs[logical]){
                    slots[logical]=physical;used[physical]=reused[logical]=true;++hits;break;
                }
            }
        }
        for(unsigned logical=0;logical<grid.pages();++logical)if(!reused[logical]){
            unsigned physical=0;while(used[physical])++physical;
            slots[logical]=physical;used[physical]=true;pages[physical].valid=false;
        }
    }
    bool complete(unsigned logical,Grid const& grid,Context const& context,std::vector<Key> const& inputs,bool proved=true){
        auto& page=pages[slots[logical]];page.valid=false;
        try{page.contributors=inputs;page.coordinate=grid.page(logical);page.context=context;page.valid=proved;++rebuilt;return true;}
        catch(...){++refused;return false;}
    }
};
} }
