#pragma once
#include "../c3x_renderer_api.h"
#include <algorithm>
#include <map>
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
private:
    struct Cell {unsigned native=0,shown=0;std::vector<Arrival> waits;unsigned long long seen=0;};
    std::map<std::pair<int,int>,Cell> cells;
    std::vector<c3x_renderer_tile_v1> tiles;
    unsigned long long scope=0,sample_id=0;
    long long capture_ticks=-1,capture_frequency=0;
    bool admitted=false;
    static constexpr unsigned bits=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_EXPLORED;
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
        auto result=frame;
        if(!frame.tiles||frame.tile_count>8192)return result;
        tiles.assign(frame.tiles,frame.tiles+frame.tile_count);
        for(auto& tile:tiles){
            auto found=cells.find({tile.tile_x,tile.tile_y});
            if(found==cells.end())continue;
            auto& cell=found->second;
            cell.waits.erase(std::remove_if(cell.waits.begin(),cell.waits.end(),[&](auto const& arrival){
                return std::find(pending.begin(),pending.end(),arrival)==pending.end();
            }),cell.waits.end());
            if(cell.waits.empty())cell.shown=cell.native;
            // An older view cannot expose new geometry, or undo a newer loss.
            tile.tile_flags=(tile.tile_flags&~bits)|(cell.shown&tile.tile_flags&bits);
        }
        result.tiles=tiles.data();return result;
    }
};
}}
