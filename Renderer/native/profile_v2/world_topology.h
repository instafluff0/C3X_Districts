#pragma once
#include "terrain_query.h"
#include <stdexcept>

namespace c3x_renderer { namespace profile_v2 {
// A compact copy owned by the rendering worker. The game thread never lends
// mutable Tile pointers to asynchronous geometry, coast or shadow preparation.
class WorldTopology {
    World world;
    std::vector<std::uint32_t> values;
public:
    struct Change { int column,row; std::uint32_t before,after; };
    World dimensions() const { return world; }
    void clear() { values.clear(); }
    bool empty() const { return values.empty(); }
    std::vector<Change> update(World next,std::uint32_t const* data,std::size_t count) {
        if(next.width<=0 || next.height<=0 || (next.width&1) || (next.wrap_y && (next.height&1)) ||
            next.width>2048 || next.height>2048 || !data ||
            count!=std::size_t(next.width)*next.height/2)
            throw std::invalid_argument("incomplete authoritative world topology");
        for(std::size_t i=0;i<count;i++)
            if((data[i]&255)>13 || ((data[i]>>8)&255)>13)
                throw std::invalid_argument("invalid authoritative terrain category");
        bool reset=values.size()!=count || world.width!=next.width || world.height!=next.height ||
            world.wrap_x!=next.wrap_x || world.wrap_y!=next.wrap_y;
        if(reset) values.assign(count,0xffffffffu);
        world=next;
        std::vector<Change> changes;
        for(std::size_t i=0;i<count;i++) {
            if(values[i]==data[i]) continue;
            int y=int(i/(world.width/2));
            int x=int(i%std::size_t(world.width/2))*2+(y&1);
            changes.push_back({(x+y)/2,(x-y)/2,values[i],data[i]});
            values[i]=data[i];
        }
        return changes;
    }
    // Raw parity-lattice index, independent of wrapped occurrence/anchor.
    std::size_t index(int column,int row) const {
        int x=column+row,y=column-row;
        if(world.wrap_x) x=mod(x,world.width);
        if(world.wrap_y) y=mod(y,world.height);
        if(values.empty() || x<0 || y<0 || x>=world.width || y>=world.height || ((x+y)&1))
            return std::size_t(-1);
        return (std::size_t(y)*world.width+x)/2;
    }
    std::uint32_t at(std::size_t i) const { return i<values.size() ? values[i] : 0xffffffffu; }
    Tile tile(int column,int row) const {
        auto value=at(index(column,row));
        return value==0xffffffffu ? Tile{} : Tile{int(value&255),int((value>>8)&255),true};
    }
};
} }
