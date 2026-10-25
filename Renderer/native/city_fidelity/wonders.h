#pragma once
#include "runtime.h"
#include "../c3x_renderer_api.h"
#include "../render_core/natural_wonder_pack.h"
#ifdef _WIN32
#include <d3d11.h>
inline void wonder_release_view(ID3D11ShaderResourceView* view){view->Release();}
#else
// Portable tests compile the store without D3D; they never upload a texture.
struct ID3D11ShaderResourceView;
inline void wonder_release_view(ID3D11ShaderResourceView*){}
#endif
#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>
namespace c3x_renderer { namespace city_fidelity {
// Constructed wonders (docs/constructed_wonder_rendering.md). The WonderRuntime
// pack's index names one city library per wonder variant and facing (or era
// district); each is decoded into its own Library and drawn as city bodies. A
// composition's authority is "<kind>|<alias>|<variant>|<season mask>|<flags>"; aliases
// are C3X config names, lowercased (pack metadata, never names in code).
enum WonderState:unsigned {wonder_none=0,wonder_unused=1,wonder_building=2,wonder_complete=3};
struct WonderFact {
    std::string alias;   // the C3X wonder config name, lowercased
    unsigned state=wonder_none,progress=0,alternate=0,era=0;  // progress 0-255
    bool on_river=false;  // its C3X config requires a river: it stands across the tile's river
};
struct WonderSlots;
// One uploaded wonder texture, shared by every group that names it; released, with its
// share of the resident budget, when the last such group goes.
struct WonderTexture {
    ID3D11ShaderResourceView* view=nullptr;std::size_t bytes=0;WonderSlots* slots=nullptr;
    ~WonderTexture();
};
// Material slots reserved for wonder groups at load (a few hundred groups' worth).
constexpr unsigned wonder_material_slots=2048;
// Resident wonder textures, at most the pack's hard cap (build_wonder_pack.BUDGET).
constexpr std::size_t wonder_texture_budget=64u*1024u*1024u;
// A loaded group: its own library, its range of the shared material table and its
// textures. Chunks compiled from it hold it (CachedVertexChunk::shared_source_lease),
// so it stays resident while a tile in the world uses it and frees its slots and
// textures after the last such chunk retires.
struct WonderGroup {
    Library library;
    unsigned material_base=~0u,material_count=0;
    std::vector<std::shared_ptr<WonderTexture>> textures;
    WonderSlots* slots=nullptr;
    ~WonderGroup();
};
// The shared material table's reserved tail (Gpu::load): fixed size, so no reader
// ever sees it move; groups take contiguous ranges.
struct WonderSlots {
    std::mutex mutex;
    std::vector<Material>* materials=nullptr;
    std::vector<std::array<ID3D11ShaderResourceView*,7>>* views=nullptr;
    unsigned first=0;std::vector<bool> used;
    std::size_t texture_bytes=0;  // resident wonder textures (wonder_texture_budget)
    unsigned take(unsigned count){
        std::lock_guard<std::mutex> lock(mutex);
        for(unsigned start=0;start+count<=used.size();++start){
            unsigned run=0;while(run<count && !used[start+run])++run;
            if(run==count){for(unsigned i=0;i<count;++i)used[start+i]=true;return first+start;}
            start+=run;
        }
        return ~0u;
    }
    void give(unsigned base,unsigned count){
        std::lock_guard<std::mutex> lock(mutex);
        // A group outliving a renderer reset finds a smaller table: leave it alone.
        for(unsigned i=0;i<count;++i){
            if(base+i<first || base-first+i>=used.size() || base+i>=materials->size() || base+i>=views->size())continue;
            used[base-first+i]=false;(*materials)[base+i]=Material{};(*views)[base+i]={};
        }
    }
};
inline WonderGroup::~WonderGroup(){
    if(slots && material_base!=~0u)slots->give(material_base,material_count);
}
inline WonderTexture::~WonderTexture(){
    if(view)wonder_release_view(view);
    if(slots){std::lock_guard<std::mutex> lock(slots->mutex);slots->texture_bytes-=std::min(slots->texture_bytes,bytes);}
}
struct WonderStore {
    struct Entry {std::string key,path;std::vector<std::string> textures;};
    std::vector<Entry> index;
    std::unordered_map<std::string,unsigned> by_key;
    mutable std::mutex mutex;
    // Held weakly: the chunks that use a group keep it alive (with the next build
    // stage a tile prefetches). A load in flight is marked so a second job needing the
    // same group waits for it; I/O runs outside the lock.
    struct Loaded {std::weak_ptr<WonderGroup const> group;bool loading=false;};
    std::unordered_map<std::string,Loaded> loaded;
    std::condition_variable loaded_changed;
    std::unordered_map<std::string,std::weak_ptr<WonderTexture>> textures;  // by name, shared by groups
    std::function<bool(std::string const&,std::vector<std::uint8_t>&)> read;
    std::function<bool(std::vector<std::uint8_t> const&,ID3D11ShaderResourceView*&)> upload;
    WonderSlots slots;
    bool ready=false;
    // Prefetch: the groups the world's Wonder Districts show (with their era district
    // and the next build stage) load on a preparation worker as soon as the district
    // appears in world state, and stay pinned while the world holds it, so a camera's
    // first visit finds them resident instead of loading them on the tile's preparation.
    std::mutex pin_mutex;
    std::vector<std::string> wanted,refused;  // sorted; refused: not loadable (budget, no kit)
    std::unordered_map<std::string,std::shared_ptr<WonderGroup const>> pinned;
    std::atomic<bool> prefetch_due{false};
    void clear(){
        {
            std::lock_guard<std::mutex> lock(pin_mutex);
            wanted.clear();refused.clear();pinned.clear();prefetch_due=false;
        }
        std::lock_guard<std::mutex> lock(mutex);
        index.clear();by_key.clear();loaded.clear();textures.clear();ready=false;
    }
    // The keys a Wonder District needs: the group it shows, its era district and the
    // next build stage.
    void needs(WonderFact const& fact,std::vector<std::string>& keys)const{
        for(auto const& key:{this->key(fact),district(fact),next(fact)})if(!key.empty())keys.push_back(key);
    }
    // The world's keys (any order, repeats allowed): unpins groups it no longer shows.
    void want(std::vector<std::string> keys){
        std::sort(keys.begin(),keys.end());keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
        std::lock_guard<std::mutex> lock(pin_mutex);
        if(keys==wanted)return;
        for(auto at=pinned.begin();at!=pinned.end();)
            at=std::binary_search(keys.begin(),keys.end(),at->first)?std::next(at):pinned.erase(at);
        wanted=std::move(keys);refused.clear();prefetch_due=true;
    }
    // Loads the wanted keys not yet pinned (a preparation worker); an interrupted pass
    // is due again.
    void prefetch(std::atomic<bool> const& stop){
        std::vector<std::string> keys;
        {
            std::lock_guard<std::mutex> lock(pin_mutex);
            for(auto const& key:wanted)
                if(!pinned.count(key) && !std::binary_search(refused.begin(),refused.end(),key))keys.push_back(key);
        }
        for(auto const& key:keys){
            if(stop.load(std::memory_order_relaxed)){prefetch_due=true;return;}
            auto group=acquire(key);
            std::lock_guard<std::mutex> lock(pin_mutex);
            if(!std::binary_search(wanted.begin(),wanted.end(),key))continue;
            if(group)pinned.emplace(key,std::move(group));
            else refused.insert(std::upper_bound(refused.begin(),refused.end(),key),key);
        }
    }

    bool parse(std::vector<std::uint8_t> const& bytes){
        std::size_t at=8;
        auto u32=[&](std::uint32_t& out){
            if(at+4>bytes.size())return false;std::memcpy(&out,bytes.data()+at,4);at+=4;return true;};
        auto text=[&](std::string& out){
            std::uint32_t n=0;if(!u32(n) || n>4096 || at+n>bytes.size())return false;
            out.assign(reinterpret_cast<char const*>(bytes.data()+at),n);at+=n;return true;};
        if(bytes.size()<12 || std::memcmp(bytes.data(),"C3XWND1\0",8))return false;
        std::uint32_t count=0;if(!u32(count) || count>4096)return false;
        index.resize(count);
        for(auto& entry:index){
            std::uint32_t size=0,names=0;
            if(!text(entry.key) || !text(entry.path) || !u32(size) || !u32(names) || names>256)return false;
            entry.textures.resize(names);
            for(auto& name:entry.textures)if(!text(name))return false;
        }
        if(at!=bytes.size())return false;
        for(unsigned i=0;i<index.size();++i)by_key.emplace(index[i].key,i);
        return true;
    }
    // The group a fact shows: a completed wonder, the build stage nearest its
    // progress (never above it, at least the first), or its era's district (an
    // unused district, and any wonder whose alias has no kit).
    std::string key(WonderFact const& fact)const{
        std::string const suffix=fact.alternate?"+alt":"";
        if(fact.state==wonder_complete || fact.state==wonder_building){
            std::string const base="wonder|"+fact.alias+"|";
            if(fact.state==wonder_complete){
                if(by_key.count(base+"complete"+suffix))return base+"complete"+suffix;
                if(by_key.count(base+"complete"))return base+"complete";
            } else {
                float const built=float(fact.progress)/255.f;
                std::string best;float best_value=-1.f,first_value=2.f;std::string first;
                // A kit without turned art for the alternate direction keeps its own facing.
                bool turned=false;
                if(fact.alternate)for(auto const& entry:index)
                    if(!entry.key.compare(0,base.size(),base) && entry.key.size()>4 &&
                       !entry.key.compare(entry.key.size()-4,4,"+alt"))turned=true;
                for(auto const& entry:index){
                    if(entry.key.compare(0,base.size(),base) || entry.key.compare(base.size(),6,"build:"))continue;
                    bool const alternate=entry.key.size()>4 && !entry.key.compare(entry.key.size()-4,4,"+alt");
                    if(alternate!=turned)continue;
                    float const stage=float(std::atof(entry.key.c_str()+base.size()+6));
                    // Half a progress step of slack: 191/255 is the 0.75 stage.
                    if(stage<=built+.5f/255.f && stage>best_value){best_value=stage;best=entry.key;}
                    if(stage<first_value){first_value=stage;first=entry.key;}
                }
                if(!best.empty())return best;
                if(!first.empty())return first;
            }
        }
        if(fact.state==wonder_none)return {};
        std::string const district="district|"+std::to_string(fact.era>3?3u:fact.era)+"|complete";
        return by_key.count(district)?district:std::string();
    }
    // The group for a key if it is resident now; never loads (the frame's paths).
    std::shared_ptr<WonderGroup const> resident(std::string const& key)const{
        if(key.empty())return nullptr;
        std::lock_guard<std::mutex> lock(mutex);
        auto found=loaded.find(key);return found==loaded.end()?nullptr:found->second.group.lock();
    }
    // The group for a key, loaded when absent: on the preparation job that needs it,
    // never on the frame. A load in flight for the same key is waited for.
    std::shared_ptr<WonderGroup const> acquire(std::string const& key){
        if(key.empty() || !ready)return nullptr;
        unsigned entry=0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            auto found=by_key.find(key);if(found==by_key.end())return nullptr;entry=found->second;
            for(;;){
                auto& state=loaded[key];
                if(auto group=state.group.lock())return group;
                if(!state.loading){state.loading=true;break;}
                loaded_changed.wait(lock);
            }
        }
        auto group=load(index[entry]);
        std::lock_guard<std::mutex> lock(mutex);
        auto& state=loaded[key];state.loading=false;
        if(group)state.group=group;
        loaded_changed.notify_all();
        return group;
    }
    // A group read, decoded and uploaded: its textures through the device
    // (free-threaded), its materials into a reserved range of the shared table.
    std::shared_ptr<WonderGroup const> load(Entry const& entry){
        auto group=std::make_shared<WonderGroup>();std::vector<std::uint8_t> bytes;
        if(!read || !upload || !read("Renderer/packs/WonderRuntime/"+entry.path,bytes) || !group->library.decode(bytes))
            return nullptr;
        unsigned const count=unsigned(group->library.materials.size());
        unsigned const base=slots.take(count);if(base==~0u)return nullptr;
        group->slots=&slots;group->material_base=base;group->material_count=count;
        for(unsigned m=0;m<count;++m){
            auto const& material=group->library.materials[m];std::array<ID3D11ShaderResourceView*,7> row{};
            for(unsigned c=0;c<7;++c){
                auto const& name=material.textures[c];if(name.empty())continue;
                auto texture=shared_texture(name);
                if(!texture)return nullptr;  // the group frees its slots and textures
                group->textures.push_back(texture);row[c]=texture->view;
            }
            std::lock_guard<std::mutex> lock(slots.mutex);
            (*slots.materials)[base+m]=material;(*slots.views)[base+m]=row;
        }
        return group;
    }
    // A texture by name: the resident upload another group shares, or a new one within
    // the resident budget (over it, the group is not loaded and its era district stands in).
    std::shared_ptr<WonderTexture> shared_texture(std::string const& name){
        {
            std::lock_guard<std::mutex> lock(mutex);
            if(auto texture=textures[name].lock())return texture;
        }
        std::vector<std::uint8_t> bytes;
        if(!read(name,bytes))return nullptr;
        auto texture=std::make_shared<WonderTexture>();
        {
            std::lock_guard<std::mutex> budget(slots.mutex);
            if(slots.texture_bytes+bytes.size()>wonder_texture_budget)return nullptr;
            slots.texture_bytes+=bytes.size();texture->bytes=bytes.size();texture->slots=&slots;
        }
        if(!upload(bytes,texture->view))return nullptr;
        std::lock_guard<std::mutex> lock(mutex);
        // Two loads may upload one name at once: keep the first, drop the other.
        if(auto other=textures[name].lock())return other;
        textures[name]=texture;
        return texture;
    }
    // The era's Wonder District for a fact: what stands in while its own group is not
    // resident, was refused or has no kit.
    std::string district(WonderFact const& fact)const{
        if(fact.state==wonder_none)return {};
        std::string const key="district|"+std::to_string(fact.era>3?3u:fact.era)+"|complete";
        return by_key.count(key)?key:std::string();
    }
    // The build stage after the one a fact shows, when its progress is within
    // `margin` of reaching it; empty otherwise.
    std::string next(WonderFact const& fact,unsigned margin=8)const{
        if(fact.state!=wonder_building)return {};
        WonderFact ahead=fact;ahead.progress=fact.progress+margin>255?255:fact.progress+margin;
        std::string const now=key(fact),later=key(ahead);
        return later!=now?later:std::string();
    }
};
inline WonderStore& wonder_store(){static WonderStore store;return store;}
// The tile's constructed-wonder facts (frame ABI 24): a Wonder District's state, its
// wonder (C3X config name, lowercased into the pack alias), progress, facing and era.
inline bool wonder_fact(c3x_renderer_tile_v1 const& tile,WonderFact& out){
    if(tile.wonder_state<wonder_unused || tile.wonder_state>wonder_complete)return false;
    out.alias.assign(tile.wonder,strnlen(tile.wonder,sizeof(tile.wonder)));
    for(auto& c:out.alias)c=char(std::tolower(static_cast<unsigned char>(c)));
    out.state=tile.wonder_state;out.progress=std::min(tile.wonder_progress,255u);
    out.alternate=(tile.wonder_flags&C3X_RENDERER_WONDER_ALTERNATE)?1u:0u;
    out.on_river=(tile.wonder_flags&C3X_RENDERER_WONDER_ON_RIVER)!=0;
    out.era=tile.wonder_era<0?0u:unsigned(tile.wonder_era);
    return true;
}
// The season mask an authority carries (field 4), 0 for every season.
inline unsigned wonder_season_mask(std::string const& authority){
    std::size_t at=0;
    for(unsigned field=0;field<3 && at!=std::string::npos;++field)at=authority.find('|',at)+1;
    return at==0 || at>=authority.size()?0u:unsigned(std::atoi(authority.c_str()+at))&15u;
}
}}
namespace c3x_renderer { namespace city_fidelity {
// A river wonder's composition (WonderFact::on_river) placed as C3X places one: centred on the
// tile's river edge (render_core::NaturalWonderPack::river_frame) with the long side of
// its bodies across the river. A tile without river bits keeps it at the centre.
inline Composition across_river(Composition placed,unsigned river){
    float ox=0,oy=0,ax=1,ay=0;
    if(!::c3x_renderer::render_core::NaturalWonderPack::river_frame(river,ox,oy,ax,ay))return placed;
    float low[2]={1e9f,1e9f},high[2]={-1e9f,-1e9f};
    for(auto const& i:placed.instances){
        low[0]=std::min(low[0],i.bounds[0]);low[1]=std::min(low[1],i.bounds[1]);
        high[0]=std::max(high[0],i.bounds[2]);high[1]=std::max(high[1],i.bounds[3]);
    }
    // Across the river: (-ay, ax). A model x unit lands on world (cos, -sin) after a
    // yaw; a model y unit on (-sin, -cos) (WorldInstance::position).
    float const cx=-ay,cy=ax;
    float const yaw=high[0]-low[0]>=high[1]-low[1]?std::atan2(-cy,cx):std::atan2(-cx,-cy);
    for(auto& i:placed.instances){i.yaw+=yaw;i.offset[0]+=ox;i.offset[1]-=oy;}
    return placed;
}
}}
