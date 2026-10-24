#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace c3x_renderer { namespace render_core {
// NaturalWonderRuntime/wonders.bin (Renderer/tools/asset_compiler/build_natural_wonder_pack.py;
// docs/natural_wonder_rendering.md). Generic kits keyed by name: C3X config names reach
// them only through the aliases, and topology bits 28-31 name a relief kit by its slot.
struct NaturalWonderPack {
    static constexpr unsigned flag_stamp=1,flag_snow=2,relief_slot_count=15,texture_slots=8,seasons=4;
    // A constructed wonder's stamp that follows its tile's river (WonderRuntime/stamps.bin).
    static constexpr unsigned flag_follow_river=4;
    // Effect record kinds (ten floats: kind, foot x/y, top x/y, lift, width, height,
    // strength, seed). Lake and channel records are water; the rest are effect quads.
    static constexpr unsigned effect_geyser=4,effect_waterfall=5,effect_lake=7,effect_channel=8,effect_spray=9;
    struct Kit {
        std::string key;
        unsigned flags=0,region_kind=0,field_size=0,relief_slot=0;
        float span=0,units=0,turn=0,snow_floor=0;
        std::vector<std::uint8_t> height,ground,special;   // field_size squared each
        std::vector<float> effects,waves;                  // ten and five floats per entry
        std::vector<std::pair<std::string,unsigned>> bodies; // composition, season mask
        bool stamp()const{return (flags&flag_stamp)!=0;}
    };
    std::vector<Kit> kits;
    std::vector<std::pair<std::string,unsigned>> aliases;   // config name, kit index
    std::array<std::array<unsigned,texture_slots>,seasons> season_slots{};
    std::array<unsigned,relief_slot_count> relief_kits{};   // slot - 1 -> kit index + 1

    bool parse(std::uint8_t const* data,std::size_t size){
        *this={};
        std::size_t at=8;
        auto take=[&](void* out,std::size_t n){
            if(n>size-at)return false;std::memcpy(out,data+at,n);at+=n;return true;};
        auto u32=[&](unsigned& out){std::uint32_t v=0;if(!take(&v,4))return false;out=v;return true;};
        auto text=[&](std::string& out){unsigned n=0;if(!u32(n) || n==0 || n>256 || n>size-at)return false;
            out.assign(reinterpret_cast<char const*>(data+at),n);at+=n;
            return out.find('\0')==std::string::npos;};
        auto floats=[&](std::vector<float>& out,unsigned n){out.resize(n);
            if(!take(out.data(),std::size_t(n)*4))return false;
            for(float v:out)if(!std::isfinite(v))return false;return true;};
        unsigned version=0,kit_count=0,alias_count=0,slots=0;
        if(!data || size<24 || std::memcmp(data,"C3XNWK1\0",8) || !u32(version) || !u32(kit_count) ||
           !u32(alias_count) || !u32(slots) || version!=1 || kit_count>64 || alias_count>256 || slots!=texture_slots)
            return fail();
        for(auto& season:season_slots)for(auto& slot:season)if(!u32(slot) || slot>=32)return fail();
        kits.resize(kit_count);
        unsigned relief=0;
        for(unsigned k=0;k<kit_count;++k){
            auto& kit=kits[k];unsigned count=0;
            float params[4]={};
            if(!text(kit.key) || !u32(kit.flags) || !u32(kit.region_kind) || !take(params,16) ||
               (kit.flags&~7u) || kit.region_kind>4 || !u32(kit.field_size))return fail();
            kit.span=params[0];kit.units=params[1];kit.turn=params[2];kit.snow_floor=params[3];
            for(float v:params)if(!std::isfinite(v))return fail();
            if(kit.stamp()!=(kit.field_size!=0) || kit.field_size>1024 ||
               (kit.stamp() && (!(kit.span>0) || !(kit.units>0) || !kit.region_kind)))return fail();
            std::size_t cells=std::size_t(kit.field_size)*kit.field_size;
            for(auto* field:{&kit.height,&kit.ground,&kit.special}){
                field->resize(cells);if(cells && !take(field->data(),cells))return fail();}
            if(!u32(count) || count>64 || !floats(kit.effects,count*10))return fail();
            for(unsigned e=0;e<count;++e){float kind=kit.effects[e*10];
                if(kind!=float(effect_geyser) && kind!=float(effect_waterfall) && kind!=float(effect_lake) &&
                   kind!=float(effect_channel) && kind!=float(effect_spray))return fail();}
            if(!u32(count) || count>100000 || !floats(kit.waves,count*5))return fail();
            if(!u32(count) || count>16)return fail();
            kit.bodies.resize(count);
            for(auto& body:kit.bodies)if(!text(body.first) || !u32(body.second) || body.second>15)return fail();
            if(kit.stamp() || !kit.waves.empty()){
                if(relief==relief_slot_count)return fail();
                kit.relief_slot=++relief;relief_kits[relief-1]=k+1;
            }
        }
        aliases.resize(alias_count);
        for(auto& alias:aliases)if(!text(alias.first) || !u32(alias.second) || alias.second>=kit_count)return fail();
        return at==size || fail();
    }
    // The kit a C3X config name resolves to (ASCII case-insensitive), or nullptr.
    Kit const* find(char const* name)const{
        if(!name || !*name)return nullptr;
        for(auto const& alias:aliases){
            auto const& a=alias.first;std::size_t i=0;
            for(;i<a.size() && name[i];++i){
                char x=a[i],y=name[i];if(x>='A'&&x<='Z')x=char(x-'A'+'a');if(y>='A'&&y<='Z')y=char(y-'A'+'a');
                if(x!=y)break;
            }
            if(i==a.size() && !name[i])return &kits[alias.second];
        }
        return nullptr;
    }
    // Appends another pack's kits and aliases (constructed wonders' stamps after the
    // natural kits): relief slots continue the numbering; a kit that would need a slot
    // past the last is dropped with its aliases. Returns the number dropped.
    unsigned append(NaturalWonderPack other){
        unsigned relief=0,dropped=0;
        for(auto const& kit:kits)relief=kit.relief_slot>relief?kit.relief_slot:relief;
        std::vector<unsigned> placed(other.kits.size(),~0u);
        for(std::size_t k=0;k<other.kits.size();++k){
            Kit kit=std::move(other.kits[k]);kit.relief_slot=0;
            if(kit.stamp() || !kit.waves.empty()){
                if(relief==relief_slot_count){++dropped;continue;}
                kit.relief_slot=++relief;relief_kits[relief-1]=unsigned(kits.size())+1;
            }
            placed[k]=unsigned(kits.size());kits.push_back(std::move(kit));
        }
        for(auto const& alias:other.aliases)
            if(placed[alias.second]!=~0u)aliases.emplace_back(alias.first,placed[alias.second]);
        return dropped;
    }
    // A follow-river kit's frame on its tile, as C3X places a river wonder
    // (get_primary_river_direction): the first river bit in C3X's priority E, W, SE, NE,
    // SW, NW, S, N (direction_to_neighbor_bit). (ox, oy) runs from the tile centre to that
    // edge's middle or corner in world tiles; (ax, ay) is the river line there.
    static bool river_frame(unsigned river,float& ox,float& oy,float& ax,float& ay){
        struct Direction {unsigned bit;int column,row;};
        static constexpr Direction order[8]={{2,1,1},{6,-1,-1},{3,1,0},{1,0,1},{5,0,-1},{7,-1,0},{4,1,-1},{0,-1,1}};
        for(auto const& d:order)if(river&(1u<<d.bit)){
            float const length=std::sqrt(float(d.column*d.column+d.row*d.row));
            ox=.5f*float(d.column);oy=.5f*float(d.row);ax=-float(d.row)/length;ay=float(d.column)/length;
            return true;
        }
        return false;
    }
    // The kit a topology relief slot (bits 28-31) names, or nullptr.
    Kit const* relief(unsigned slot)const{
        return slot && slot<=relief_slot_count && relief_kits[slot-1]?&kits[relief_kits[slot-1]-1]:nullptr;
    }
private:
    bool fail(){*this={};return false;}
};
}}
