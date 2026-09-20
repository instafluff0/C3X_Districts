#pragma once
#include "terrain_query.h"
#include <stdexcept>

namespace c3x_renderer { namespace render_core {
// A compact copy owned by the rendering worker. The game thread never lends
// mutable Tile pointers to asynchronous geometry, coast or shadow preparation.
class WorldTopology {
    World world;
    std::vector<std::uint32_t> values;
    std::vector<unsigned char> flow; // Two bits per native edge: still, forward, reverse.
public:
    struct Change { int column,row; std::uint32_t before,after; };
    World dimensions() const { return world; }
    void clear() { values.clear();flow.clear(); }
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
        bool flow_changed=reset;
        for(auto const& change:changes)flow_changed=flow_changed ||
            ((change.before^change.after)&0x00aa00ffu)!=0;
        if(flow_changed)rebuild_flow();
        return changes;
    }
    // Visual drainage only. Civ III supplies connectivity, not flow direction.
    // Use shortest connected distance to water; closed components have a stable
    // canonical sink. No height/gameplay mutation and no per-frame traversal.
    void rebuild_flow() {
        flow.assign(values.size(),0);
        struct Node {int x,y,head=-1,distance=-1;};
        struct Arc {unsigned to;int next;};
        struct Edge {unsigned a,b;std::size_t tile;unsigned slot;};
        std::map<std::pair<int,int>,unsigned> lookup;
        std::vector<Node> nodes;std::vector<Arc> arcs;std::vector<Edge> edges;
        auto node=[&](int c,int r){int x=c+r,y=c-r;
            if(world.wrap_x)x=mod(x,world.width);if(world.wrap_y)y=mod(y,world.height);
            auto key=std::make_pair(x,y);auto it=lookup.find(key);
            if(it!=lookup.end())return it->second;
            auto id=unsigned(nodes.size());lookup.emplace(key,id);nodes.push_back({x,y});return id;
        };
        for(std::size_t i=0;i<values.size();++i){unsigned bits=(values[i]>>16)&170u;if(!bits)continue;
            int y=int(i/(world.width/2)),x=int(i%std::size_t(world.width/2))*2+(y&1);
            int c=(x+y)/2,r=(x-y)/2;
            for(unsigned slot=0;slot<4;++slot)if(bits&(2u<<(slot*2))){
                int ac=c+(slot==1),ar=r+(slot==0),bc=c+(slot!=3),br=r+(slot!=2);
                unsigned a=node(ac,ar),b=node(bc,br);
                // Bounded transient graph (well below 16 MiB). Oversized worlds
                // retain still river normals rather than partial direction data.
                if(nodes.size()>65536 || edges.size()>=131072)return;
                edges.push_back({a,b,i,slot});
                arcs.push_back({b,nodes[a].head});nodes[a].head=int(arcs.size()-1);
                arcs.push_back({a,nodes[b].head});nodes[b].head=int(arcs.size()-1);
            }
        }
        std::vector<unsigned> queue;queue.reserve(nodes.size());
        for(unsigned i=0;i<nodes.size();++i){auto& n=nodes[i];int c=(n.x+n.y)/2,r=(n.x-n.y)/2;bool mouth=false;
            for(int v=r-1;v<=r;++v)for(int u=c-1;u<=c;++u){auto value=at(index(u,v));
                mouth=mouth || (value!=0xffffffffu && (value&255)>=11);}
            if(mouth){n.distance=0;queue.push_back(i);}
        }
        auto flood=[&](){for(std::size_t j=0;j<queue.size();++j){auto a=queue[j];
            for(int arc=nodes[a].head;arc>=0;arc=arcs[arc].next){auto b=arcs[arc].to;
                if(nodes[b].distance<0){nodes[b].distance=nodes[a].distance+1;queue.push_back(b);}}
        }queue.clear();};
        flood();
        for(auto const& item:lookup)if(nodes[item.second].distance<0){nodes[item.second].distance=0;queue.push_back(item.second);flood();}
        for(auto const& edge:edges){auto const&a=nodes[edge.a];auto const&b=nodes[edge.b];
            bool forward=b.distance<a.distance || (b.distance==a.distance && std::make_pair(b.x,b.y)<std::make_pair(a.x,a.y));
            flow[edge.tile]|=(forward?1u:2u)<<(edge.slot*2);
        }
    }
    unsigned river_flow(std::size_t i) const {return i<flow.size()?flow[i]:0;}
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
