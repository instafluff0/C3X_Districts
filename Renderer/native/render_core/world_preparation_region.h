#pragma once
#include "captured_scene.h"
#include <array>
#include <vector>
#include <unordered_set>

namespace c3x_renderer { namespace render_core {
// An immutable compiler lease assembled from the existing world. Its halo is
// input only; this does not produce a viewport or grant display eligibility.
struct WorldPreparationRegion {
    static constexpr int extent=8,halo=12;
    c3x_renderer_frame_v1 frame{};
    std::vector<c3x_renderer_tile_v1> tiles;
    std::vector<unsigned> selected;
    static int bands(int cells,int pixels,int tile,bool wrap){
        if(!wrap || tile<=0)return 0;
        return std::min((cells+extent-1)/extent,((pixels+tile-1)/tile+halo+extent-1)/extent);
    }
    static unsigned count(c3x_renderer_frame_v1 const& f){
        return unsigned((f.world_width_tiles+extent-1)/extent+2*bands(f.world_width_tiles,f.target_width,f.tile_width,f.world_wrap_x!=0))*
            unsigned((f.world_height_tiles+extent-1)/extent+2*bands(f.world_height_tiles,f.target_height,f.tile_height,f.world_wrap_y!=0));
    }
    static std::array<int,2> core(unsigned index,int cells,int band){
        unsigned columns=unsigned((cells+extent-1)/extent);
        if(index<columns)return {int(index)*extent,0};
        index-=columns;
        if(index<unsigned(band))return {int(index)*extent,cells};
        return {int(columns-unsigned(band)+index-unsigned(band))*extent,-cells};
    }
    bool build(CapturedScene const& scene,c3x_renderer_frame_v1 const& source,unsigned region){
        tiles.clear();selected.clear();frame=source;
        if(!scene.matches_world(source) || source.world_width_tiles<=0 || source.world_height_tiles<=0 ||
           source.world_width_tiles>2048 || source.world_height_tiles>2048 || region>=count(source))return false;
        int horizontal=bands(source.world_width_tiles,source.target_width,source.tile_width,source.world_wrap_x!=0);
        int vertical=bands(source.world_height_tiles,source.target_height,source.tile_height,source.world_wrap_y!=0);
        unsigned columns=unsigned((source.world_width_tiles+extent-1)/extent+2*horizontal);
        auto x_core=core(region%columns,source.world_width_tiles,horizontal);
        auto y_core=core(region/columns,source.world_height_tiles,vertical);
        int left=x_core[0],top=y_core[0];
        std::unordered_set<std::uint64_t> seen;
        // Canonical cores plus bounded edge occurrences cover native wrapping
        // independently of navigation history. Halo coordinates stay unwrapped,
        // matching the authoritative occurrence lattice around each core.
        // Ground and rigid geometry use world units; anchors only supply relative
        // neighbor placement, with the native viewport's projection/detail inputs.
        for(int y=top-halo;y<top+extent+halo;++y)for(int x=left-halo;x<left+extent+halo;++x){
            if((x+y)&1)continue;
            if((!source.world_wrap_x && (x<0 || x>=source.world_width_tiles)) ||
               (!source.world_wrap_y && (y<0 || y>=source.world_height_tiles)))continue;
            auto key=scene.key(x,y);
            // Tiny wrapped worlds use the occurrence closest to the core, not
            // an arbitrary duplicate. They are handled by the demand path until
            // the whole-world lease can prove that projection unambiguously.
            if(!seen.insert(key).second)return false;
            auto record=scene.retained(key);
            if(!record || !(record->visibility_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN))return false;
            bool explored=(record->visibility_flags&C3X_RENDERER_TILE_EXPLORED)!=0;
            if(explored && !record->authoritative)return false;
            c3x_renderer_tile_v1 tile{};
            if(explored)tile=record->appearance;
            else {
                // Unseen halo contributes topology to neighboring explored art,
                // without importing objects or becoming a compiler core.
                auto canonical_x=x%source.world_width_tiles,canonical_y=y%source.world_height_tiles;
                if(canonical_x<0)canonical_x+=source.world_width_tiles;
                if(canonical_y<0)canonical_y+=source.world_height_tiles;
                auto index=(canonical_y*source.world_width_tiles+canonical_x)/2;
                if(!source.world_topology || index<0 || unsigned(index)>=source.world_topology_count)return false;
                auto topology=source.world_topology[index];
                tile.terrain_type=int(topology&255u);tile.real_terrain_type=int((topology>>8)&255u);
                tile.river_code=(topology>>16)&255u;
                tile.resource_id=tile.resource_class=tile.tile_building_id=-1;
                tile.city_id=tile.city_owner_id=tile.unit_type_id=tile.unit_owner_id=-1;
                tile.barbarian_tribe_id=-1;
            }
            tile.tile_x=x+x_core[1];tile.tile_y=y+y_core[1];
            tile.anchor_x=(x-left)*source.tile_width/2;
            tile.anchor_y=(y-top)*source.tile_height/2;
            tile.visibility_mask=record->visibility_mask;tile.tile_visibility=record->tile_visibility;
            tile.fog_status=record->fog_status;
            tile.tile_flags=record->visibility_flags|C3X_RENDERER_TILE_TOPOLOGY_HALO|
                (explored?C3X_RENDERER_TILE_PREFETCH:0u);
            if(x>=left && x<left+extent && y>=top && y<top+extent &&
               x<source.world_width_tiles && y<source.world_height_tiles && explored)
                selected.push_back(unsigned(tiles.size()));
            tiles.push_back(tile);
        }
        frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        return tiles.size()<=CapturedScene::occurrence_limit;
    }
};

// One background demand order for the existing world compiler. Camera motion
// reprioritizes unfinished regions; it neither resets completed work nor keeps
// another copy of tile/topology inputs. Missing authority is retried on the next
// appearance revision. A cancelled lease leaves its region pending.
class WorldPreparationSchedule {
    std::array<std::uint64_t,9> scope{};
    std::vector<unsigned> pending;
    std::vector<unsigned char> state;
    int center_x=0,center_y=0;
    bool reorder=true;
public:
    unsigned completed=0,unavailable=0;
    void clear(){pending.clear();state.clear();scope={};completed=unavailable=0;reorder=true;}
    void prioritize(c3x_renderer_frame_v1 const& f){
        std::int64_t distance=INT64_MAX;int next_x=center_x,next_y=center_y;
        for(unsigned i=0;i<f.tile_count;++i){auto const& tile=f.tiles[i];
            if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            auto dx=std::int64_t(tile.anchor_x)*2+f.tile_width-f.target_width;
            auto dy=std::int64_t(tile.anchor_y)*2+f.tile_height-f.target_height;
            auto d=(dx<0?-dx:dx)+2*(dy<0?-dy:dy);
            if(d<distance){distance=d;next_x=tile.tile_x;next_y=tile.tile_y;}
        }
        reorder|=center_x!=next_x || center_y!=next_y;center_x=next_x;center_y=next_y;
    }
    void configure(c3x_renderer_frame_v1 const& f,std::uint64_t lifetime,
                   std::uint64_t assets,unsigned device){
        std::array<std::uint64_t,9> next={lifetime,assets,device,
            std::uint64_t(f.world_width_tiles),std::uint64_t(f.world_height_tiles),
            std::uint64_t(f.tile_width),std::uint64_t(f.tile_height),
            std::uint64_t(f.target_width),std::uint64_t(f.target_height)};
        if(scope!=next){
            clear();scope=next;pending.resize(WorldPreparationRegion::count(f));
            state.resize(pending.size());
            for(unsigned i=0;i<pending.size();++i)pending[i]=i;
        }
        if(!reorder)return;
        reorder=false;
        int horizontal=WorldPreparationRegion::bands(f.world_width_tiles,f.target_width,f.tile_width,f.world_wrap_x!=0);
        int vertical=WorldPreparationRegion::bands(f.world_height_tiles,f.target_height,f.tile_height,f.world_wrap_y!=0);
        constexpr int extent=WorldPreparationRegion::extent;
        unsigned columns=unsigned((f.world_width_tiles+extent-1)/extent+2*horizontal);
        auto distance=[&](unsigned region){
            auto x=WorldPreparationRegion::core(region%columns,f.world_width_tiles,horizontal);
            auto y=WorldPreparationRegion::core(region/columns,f.world_height_tiles,vertical);
            auto dx=std::int64_t(x[0])+x[1]+extent/2-center_x,dy=std::int64_t(y[0])+y[1]+extent/2-center_y;
            return (dx<0?-dx:dx)+(dy<0?-dy:dy);
        };
        // Pop the nearest first. Stable ties keep preparation deterministic.
        std::stable_sort(pending.begin(),pending.end(),[&](unsigned a,unsigned b){return distance(a)>distance(b);});
    }
    void invalidate(c3x_renderer_frame_v1 const& f,int tile_x,int tile_y){
        if(state.size()!=WorldPreparationRegion::count(f))return;
        auto contains=[](int coordinate,int left,int world,bool wrap){
            for(int offset=wrap?-2:0;offset<=(wrap?2:0);++offset){
                int occurrence=coordinate+offset*world;
                if(occurrence>=left-WorldPreparationRegion::halo &&
                   occurrence<left+WorldPreparationRegion::extent+WorldPreparationRegion::halo)return true;
            }
            return false;
        };
        int horizontal=WorldPreparationRegion::bands(f.world_width_tiles,f.target_width,f.tile_width,f.world_wrap_x!=0);
        int vertical=WorldPreparationRegion::bands(f.world_height_tiles,f.target_height,f.tile_height,f.world_wrap_y!=0);
        unsigned columns=unsigned((f.world_width_tiles+WorldPreparationRegion::extent-1)/WorldPreparationRegion::extent+2*horizontal);
        for(unsigned region=0;region<state.size();++region){
            auto x=WorldPreparationRegion::core(region%columns,f.world_width_tiles,horizontal);
            auto y=WorldPreparationRegion::core(region/columns,f.world_height_tiles,vertical);
            if(!contains(tile_x,x[0],f.world_width_tiles,f.world_wrap_x!=0) ||
               !contains(tile_y,y[0],f.world_height_tiles,f.world_wrap_y!=0) || !state[region])continue;
            if(state[region]==2)--unavailable;
            --completed;state[region]=0;pending.push_back(region);reorder=true;
        }
    }
    bool empty()const{return pending.empty();}
    unsigned next()const{return pending.back();}
    void finish(bool success){state[pending.back()]=success?1:2;pending.pop_back();++completed;if(!success)++unavailable;}
};
}}
