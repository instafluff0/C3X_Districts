#pragma once
#include "../c3x_renderer_api.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Opt-in measurement only. Keys are compared exactly; none is adopted here.
struct WorldKeyDiagnostic {
    static constexpr unsigned witness_limit=16;
    struct Match {
        bool available=false,related=false,revision_only=false,nearest_recipe_empty=false;
        std::uint64_t mask=0,all_masks=0;
        std::array<std::uint64_t,25> nearest{};
        unsigned witness=witness_limit;
    };
    std::array<std::atomic<std::uint64_t>,5> counts{};
    std::atomic<std::uint64_t> identity_mask{0};
    std::atomic<unsigned> witnesses{0};
    unsigned fact_witnesses=0; // Only the serialized publication owner writes.
    void reset(){for(auto& value:counts)value.store(0);identity_mask.store(0);witnesses.store(0);fact_witnesses=0;}
    static bool foreground(bool enabled,unsigned component,bool backing_only,bool retain_prepared){
        return enabled && component==1 && !backing_only && !retain_prepared;
    }
    static unsigned bits(std::uint64_t value){
        unsigned count=0;for(;value;value&=value-1)++count;return count;
    }
    template<class Key,class Enumerate> Match inspect(Key const& current,Enumerate enumerate){
        Match result;++counts[0];
        try {
            enumerate([&](Key const& retained){
                if(retained.kind()!=current.kind() || retained.identity[0]!=current.identity[0] ||
                   retained.identity[1]!=current.identity[1])return;
                std::uint64_t mask=0;
                for(unsigned field=0;field<current.identity.size();++field)
                    if(retained.identity[field]!=current.identity[field])mask|=std::uint64_t(1)<<field;
                if(retained.recipe!=current.recipe)mask|=std::uint64_t(1)<<25;
                result.all_masks|=mask;
                result.revision_only=result.revision_only || mask==(std::uint64_t(1)<<17);
                if(!result.related || bits(mask)<bits(result.mask) ||
                   (bits(mask)==bits(result.mask) && (mask<result.mask ||
                    (mask==result.mask && retained.identity<result.nearest)))){
                    result.mask=mask;result.nearest=retained.identity;result.nearest_recipe_empty=retained.recipe.words.empty();
                }
                result.related=true;
            });
            result.available=true;
            ++counts[result.related?(result.revision_only?1:2):3];
            identity_mask.fetch_or(result.all_masks,std::memory_order_relaxed);
        }catch(...){++counts[4];} // Failed observation is explicitly unavailable.
        auto slot=witnesses.load(std::memory_order_relaxed);
        while(slot<witness_limit && !witnesses.compare_exchange_weak(slot,slot+1,std::memory_order_relaxed)){}
        if(slot<witness_limit)result.witness=slot;
        return result;
    }
    static std::uint64_t digest(void const* data,std::size_t size){
        auto bytes=static_cast<unsigned char const*>(data);std::uint64_t value=1469598103934665603ull;
        for(std::size_t i=0;i<size;++i)value=(value^bytes[i])*1099511628211ull;
        return value;
    }
    static std::string words(std::array<std::uint64_t,25> const& key){
        std::string out;out.reserve(425);
        for(auto word:key){if(!out.empty())out+=':';
            for(int nibble=15;nibble>=0;--nibble)out+="0123456789abcdef"[(word>>(nibble*4))&15u];}
        return out;
    }
    static std::vector<std::string> fact_parts(std::string const& fields){
        std::vector<std::string> parts;
        for(std::size_t offset=0;offset<fields.size();offset+=480)parts.push_back(fields.substr(offset,480));
        return parts;
    }
    // The caller normalizes through CapturedScene::content before comparing.
    // Source labels are hashed, never copied into a diagnostic log.
    static std::string facts(c3x_renderer_tile_v1 const& before,c3x_renderer_tile_v1 const& after){
        struct Field {char const* name;std::size_t offset,size;};
        Field const fields[]={
            {"terrain",offsetof(c3x_renderer_tile_v1,terrain_type),4},
            {"real",offsetof(c3x_renderer_tile_v1,real_terrain_type),4},
            {"seed",offsetof(c3x_renderer_tile_v1,variant_seed),4},
            {"resource_id",offsetof(c3x_renderer_tile_v1,resource_id),4},
            {"resource_class",offsetof(c3x_renderer_tile_v1,resource_class),4},
            {"resource_name_digest",offsetof(c3x_renderer_tile_v1,resource_name),sizeof(before.resource_name)},
            {"city_id",offsetof(c3x_renderer_tile_v1,city_id),4},
            {"city_owner",offsetof(c3x_renderer_tile_v1,city_owner_id),4},
            {"city_size",offsetof(c3x_renderer_tile_v1,city_size),4},
            {"city_culture",offsetof(c3x_renderer_tile_v1,city_culture_group),4},
            {"city_era",offsetof(c3x_renderer_tile_v1,city_era),4},
            {"city_flags",offsetof(c3x_renderer_tile_v1,city_flags),4},
            {"river",offsetof(c3x_renderer_tile_v1,river_code),4},
            {"road",offsetof(c3x_renderer_tile_v1,road_mask),4},
            {"railroad",offsetof(c3x_renderer_tile_v1,railroad_mask),4},
            {"route_style",offsetof(c3x_renderer_tile_v1,route_style),4},
            {"features",offsetof(c3x_renderer_tile_v1,feature_flags),4},
            {"improvements",offsetof(c3x_renderer_tile_v1,improvement_flags),4},
            {"irrigation",offsetof(c3x_renderer_tile_v1,irrigation_mask),4},
            {"effect",offsetof(c3x_renderer_tile_v1,has_effect),4},
            {"territory_edges",offsetof(c3x_renderer_tile_v1,territory_edge_mask),4},
            {"territory_rgb",offsetof(c3x_renderer_tile_v1,territory_color_rgb),4},
            {"barbarian",offsetof(c3x_renderer_tile_v1,barbarian_tribe_id),4}};
        std::array<bool,sizeof(before)> covered{};
        std::string out;out.reserve(1400);
        auto a=reinterpret_cast<unsigned char const*>(&before),b=reinterpret_cast<unsigned char const*>(&after);
        auto hex=[&](std::uint64_t value){for(int n=15;n>=0;--n)out+="0123456789abcdef"[(value>>(n*4))&15u];};
        for(auto const& field:fields){
            for(std::size_t i=field.offset;i<field.offset+field.size;++i)covered[i]=true;
            if(!std::memcmp(a+field.offset,b+field.offset,field.size))continue;
            if(!out.empty())out+=';';out+=field.name;out+=':';
            std::uint64_t prior=0,next=0;
            if(field.size==4){std::uint32_t p=0,q=0;std::memcpy(&p,a+field.offset,4);std::memcpy(&q,b+field.offset,4);prior=p;next=q;}
            else {prior=digest(a+field.offset,field.size);next=digest(b+field.offset,field.size);}
            hex(prior);out+='>';hex(next);
        }
        unsigned other=0;for(std::size_t i=0;i<sizeof(before);++i)if(!covered[i] && a[i]!=b[i])++other;
        if(other){if(!out.empty())out+=';';out+="other_bytes:";hex(other);}
        return out;
    }
};
}}
