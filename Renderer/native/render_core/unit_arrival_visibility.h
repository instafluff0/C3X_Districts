#pragma once
#include "../c3x_renderer_api.h"
#include "captured_scene.h"
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Native visibility remains authoritative. Only increases wait for the
// accepted moves represented by this capture; loss of sight is immediate.
// Terrain can be prepared while the body travels, with the final fog pass
// releasing the copied reveal on the very same frame as visual arrival.
class UnitArrivalVisibility {
public:
    using Arrival=std::pair<int,long long>;
    std::vector<Arrival> pending;
    // Diagnostic: cells the last sample newly displayed as visible/explored,
    // and the native capture (QPC time) whose visibility they show.
    unsigned revealed=0;long long revealed_capture=-1,revealed_frequency=0;
    // Diagnostic: cells natively explored/visible that the last sample kept
    // dark, waiting for a unit's arrival or for a view drawing them.
    unsigned held_waiting=0,held_undrawn=0;
private:
    // reveal_content: what the tile showed when Civ III revealed it
    // (content_hash), compared with what the drawn view drew there;
    // reveal_unit: a unit stood on it (bodies pose only on seen tiles).
    struct Cell {unsigned native=0,shown=0,displayed=0;std::vector<Arrival> waits;unsigned long long seen=0,reveal_content=0;bool sampled=false,reveal_unit=false;};
    std::map<std::pair<int,int>,Cell> cells;
    std::vector<c3x_renderer_tile_v1> tiles;
    std::vector<Cell*> sampled_cells;
    unsigned long long scope=0,sample_id=0;
    long long capture_ticks=-1,capture_frequency=0;
    bool admitted=false;
    static constexpr unsigned bits=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_EXPLORED;
    // A tile's drawn content (CapturedScene::same_compiled), hashed.
    static unsigned long long content_hash(c3x_renderer_tile_v1 tile){
        tile=CapturedScene::content(tile);tile.territory_edge_mask=tile.territory_color_rgb=0;
        unsigned long long hash=1469598103934665603ull;auto bytes=reinterpret_cast<unsigned char const*>(&tile);
        for(std::size_t i=0;i<sizeof(tile);++i)hash=(hash^bytes[i])*1099511628211ull;
        return hash|1;
    }
public:
    void reset(){cells.clear();tiles.clear();pending.clear();scope=0;capture_ticks=-1;capture_frequency=0;admitted=false;}
    // Admit native visibility once, before terrain work. Presentation may later
    // borrow an older completed view; that is not another native observation.
    void capture(c3x_renderer_frame_v1 const& frame,unsigned long long current_scope){
        if(scope!=current_scope){cells.clear();scope=current_scope;capture_ticks=-1;capture_frequency=0;}
        if(!frame.tiles||frame.tile_count>8192)return;
        if(admitted && frame.presentation_frequency>0 && capture_frequency==frame.presentation_frequency &&
            frame.presentation_time_ticks<capture_ticks)return;
        admitted=true;capture_ticks=frame.presentation_time_ticks;capture_frequency=frame.presentation_frequency;
        ++sample_id;
        std::vector<Arrival> represented;
        for(auto const& arrival:pending)
            if(capture_frequency<=0 || arrival.second<=capture_ticks)represented.push_back(arrival);
        for(unsigned i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
            auto key=std::make_pair(tile.tile_x,tile.tile_y);
            auto native=tile.tile_flags&bits;
            auto found=cells.find(key);
            if(found==cells.end())found=cells.emplace(key,Cell{native,native,{}}).first;
            auto& cell=found->second;cell.seen=sample_id;
            if((native&C3X_RENDERER_TILE_EXPLORED) && !(cell.native&C3X_RENDERER_TILE_EXPLORED)){
                cell.reveal_content=content_hash(tile);cell.reveal_unit=tile.unit_type_id>=0;}
            if((native&cell.native)!=cell.native){cell.shown&=native;cell.waits.clear();}
            if(native!=cell.native){
                if((native&~cell.shown)&&!represented.empty())cell.waits=represented;
                else {cell.shown=native;cell.waits.clear();}
                cell.native=native;
            }
        }
        if(cells.size()>8192){
            for(auto it=cells.begin();it!=cells.end();){
                if(it->second.seen!=sample_id)it=cells.erase(it);else ++it;
            }
        }
    }
    c3x_renderer_frame_v1 sample(c3x_renderer_frame_v1 const& frame,unsigned long long current_scope){
        // Standalone consumers have no ordered unit/camera admission owner.
        if(!admitted || scope!=current_scope)capture(frame,current_scope);
        auto result=frame;revealed=0;held_waiting=held_undrawn=0;
        if(!frame.tiles||frame.tile_count>8192)return result;
        tiles.assign(frame.tiles,frame.tiles+frame.tile_count);
        // The view draws explored ground, and unexplored tiles near it as
        // terrain with their contents (the hidden ring, CapturedScene), so
        // fog of war lifts on it as soon as native sight arrives. A reveal
        // lifts on it only when it draws every newly explored tile whole:
        // - contents as revealed: the ring draws Civ III's capture of the
        //   tile (ring_content) but withholds cities and live effects, and
        //   unit bodies pose only on seen tiles;
        // - ring terrain all around, so explored land within one tile: a
        //   tile two away meets undrawn ground, a sharp edge in the fade.
        // Otherwise all of them wait for the view built with the reveal, so
        // tiles, contents and soft edges appear at once (the user, October 27).
        std::set<std::pair<int,int>> explored;
        auto wrap=[](int value,int extent,bool wraps){return wraps && extent>0?((value%extent)+extent)%extent:value;};
        auto whole=[&](c3x_renderer_tile_v1 const& tile,Cell const& cell){
            if(cell.reveal_unit || (cell.reveal_content && content_hash(CapturedScene::ring_content(tile))!=cell.reveal_content))return false;
            if(explored.empty())for(unsigned i=0;i<frame.tile_count;++i)
                if(frame.tiles[i].tile_flags&C3X_RENDERER_TILE_EXPLORED)explored.emplace(frame.tiles[i].tile_x,frame.tiles[i].tile_y);
            return CapturedScene::each_ring_offset([&](int dx,int dy){
                return explored.count({wrap(tile.tile_x+dx,frame.world_width_tiles,frame.world_wrap_x!=0),
                                       wrap(tile.tile_y+dy,frame.world_height_tiles,frame.world_wrap_y!=0)})!=0;
            },CapturedScene::ring_reach-1);
        };
        bool reveal=true;sampled_cells.assign(tiles.size(),nullptr);
        for(std::size_t i=0;i<tiles.size();++i){auto const& tile=tiles[i];
            auto found=cells.find({tile.tile_x,tile.tile_y});
            if(found==cells.end())continue;
            auto& cell=found->second;sampled_cells[i]=&cell;
            cell.waits.erase(std::remove_if(cell.waits.begin(),cell.waits.end(),[&](auto const& arrival){
                return std::find(pending.begin(),pending.end(),arrival)==pending.end();
            }),cell.waits.end());
            if(cell.waits.empty())cell.shown=cell.native;
            if((cell.shown&C3X_RENDERER_TILE_EXPLORED) && !(tile.tile_flags&C3X_RENDERER_TILE_EXPLORED))
                reveal=reveal && whole(tile,cell);
        }
        for(std::size_t i=0;i<tiles.size();++i){auto& tile=tiles[i];
            if(!sampled_cells[i])continue;
            auto& cell=*sampled_cells[i];
            // An older view cannot expose new geometry, or undo a newer loss.
            auto allowed=tile.tile_flags&bits;
            if((cell.shown&~allowed) && ((tile.tile_flags&C3X_RENDERER_TILE_EXPLORED) || reveal))allowed=bits;
            held_waiting+=(cell.native&~cell.shown)!=0;held_undrawn+=(cell.shown&~allowed)!=0;
            tile.tile_flags=(tile.tile_flags&~bits)|(cell.shown&allowed);
            auto displayed=tile.tile_flags&bits;
            if(cell.sampled&&(displayed&~cell.displayed))++revealed;
            cell.displayed=displayed;cell.sampled=true;
        }
        if(revealed){revealed_capture=capture_ticks;revealed_frequency=capture_frequency;}
        result.tiles=tiles.data();return result;
    }
};
}}
