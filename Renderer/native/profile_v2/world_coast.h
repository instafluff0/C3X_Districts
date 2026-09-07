#pragma once
#include "coast_index.h"
#include "world_topology.h"

namespace c3x_renderer { namespace profile_v2 {
class WorldCoast {
    WorldTopology topology;
    CoastIndex coast{-4096,-4096,8192};
    std::int64_t source_revision=-1;
    bool ready=false;
public:
    struct Update { std::size_t topology_changes=0,cells_built=0,bytes=0; };
    void clear() { topology.clear(); coast.clear(); source_revision=-1; ready=false; }
    WorldTopology const& world() const { return topology; }
    std::uint64_t node_revision(std::uint64_t id) const { return coast.revision(id); }
    Update update(World dimensions,std::uint32_t const* values,std::size_t count,std::int64_t revision) {
        World previous=topology.dimensions();
        bool same=previous.width==dimensions.width && previous.height==dimensions.height &&
            previous.wrap_x==dimensions.wrap_x && previous.wrap_y==dimensions.wrap_y;
        if(ready && same && revision==source_revision) return {};
        if(!ready || !same) clear();
        ready=false;
        auto changes=topology.update(dimensions,values,count);
        std::map<std::pair<int,int>,bool> dirty;
        for(auto const& change:changes) {
            if((change.before&65535)==(change.after&65535)) continue;
            // Profile-2 displaced B-spline support is contained in this
            // neighborhood. Recompute only contour cells affected by a type edit.
            int nx=dimensions.wrap_x ? 1 : 0,ny=dimensions.wrap_y ? 1 : 0;
            for(int wy=-ny;wy<=ny;wy++) for(int wx=-nx;wx<=nx;wx++)
                for(int y=-3;y<=3;y++) for(int x=-3;x<=3;x++) {
                    int c=change.column+x+(wx*dimensions.width+wy*dimensions.height)/2;
                    int r=change.row+y+(wx*dimensions.width-wy*dimensions.height)/2;
                    int raw_x=c+r,raw_y=c-r;
                    if(raw_x>=-6 && raw_y>=-6 && raw_x<dimensions.width+6 && raw_y<dimensions.height+6)
                        dirty[{c,r}]=true;
                }
        }
        auto lookup=[&](int x,int y){return topology.tile(x,y);};
        ShoreField field(dimensions,lookup);
        Update result; result.topology_changes=changes.size();
        for(auto const& item:dirty) {
            int c=item.first.first,r=item.first.second;
            bool land=false,water_present=false;
            for(int y=-2;y<=3;y++) for(int x=-2;x<=3;x++) {
                Tile t=lookup(c+x,r+y); if(!t.present) continue;
                if(water(t)) water_present=true; else land=true;
            }
            if(land && water_present) coast.set_cell(c,r,field.cell(c,r));
            else coast.set_cell(c,r,{});
            ++result.cells_built;
            // Bound compile scratch independently of the persistent index.
            if((result.cells_built&63)==0) field.clear_scratch();
            if((result.cells_built&255)==0 && coast.estimated_bytes()>32u*1024u*1024u)
                throw std::runtime_error("world coast index exceeds 32 MiB budget");
        }
        result.bytes=coast.estimated_bytes();
        if(result.bytes>32u*1024u*1024u)
            throw std::runtime_error("world coast index exceeds 32 MiB budget");
        ready=true; source_revision=revision; return result;
    }
    template<class ObserveNode,class ObserveTile>
    ShoreSample sample(Point corner,ObserveNode observe_node,ObserveTile observe_tile) const {
        if(!ready) throw std::runtime_error("world coast field is not ready");
        World dimensions=topology.dimensions(); Point p=corner-Point{.5,.5};
        auto nearest=coast.nearest(p,[](auto,auto){});
        Point foot=nearest.foot;
        int nx=dimensions.wrap_x ? 1 : 0,ny=dimensions.wrap_y ? 1 : 0;
        // Compare periodic images before collecting dependency certificates.
        // This avoids depending on a distant unwrapped coast at a wrap seam.
        for(int y=-ny;y<=ny;y++) for(int x=-nx;x<=nx;x++) {
            Point offset{(x*dimensions.width+y*dimensions.height)*.5,
                         (x*dimensions.width-y*dimensions.height)*.5};
            auto candidate=coast.nearest(p+offset,[](auto,auto){},nearest.squared);
            if(candidate.cell && candidate.squared<nearest.squared) {
                nearest=candidate; foot=candidate.foot-offset;
            }
        }
        for(int y=-ny;y<=ny;y++) for(int x=-nx;x<=nx;x++) {
            Point offset{(x*dimensions.width+y*dimensions.height)*.5,
                         (x*dimensions.width-y*dimensions.height)*.5};
            coast.nearest(p+offset,observe_node,nearest.squared);
        }
        auto lookup=[&](int c,int r) {
            auto index=topology.index(c,r);
            if(index!=std::size_t(-1)) observe_tile(index,topology.at(index));
            return topology.tile(c,r);
        };
        ShoreField field(dimensions,lookup);
        double rocky=field.rockiness(foot);
        double distance=std::isfinite(nearest.squared) ? std::sqrt(nearest.squared) : 1e6;
        distance*=field.coverage(p)>=0 ? 1 : -1;
        return {distance,(.065+.18*field.world_noise(foot,.66,628))*(1-rocky),rocky,
            .46*(1-std::exp(-std::max(0.,-distance)/.85))};
    }
};
} }
