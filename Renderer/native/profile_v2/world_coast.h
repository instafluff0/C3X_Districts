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
    struct Patch {
        struct Edge {CoastSegment segment;Point offset;std::uint64_t cell;unsigned copy;};
        struct Node {Point low,high;std::size_t begin=0,end=0,left=0,right=0;};
        Point center;
        double radius=0;
        bool ready=false;
        std::vector<Edge> edges;
        std::vector<std::size_t> order;
        std::vector<Node> nodes;
        void build() {
            order.resize(edges.size());for(std::size_t i=0;i<order.size();++i)order[i]=i;
            nodes.reserve(edges.size()*2);
            auto bounds=[&](std::size_t index) {
                auto const& e=edges[index];
                return std::array<Point,2>{{{std::min(e.segment.a.x,e.segment.b.x)-e.offset.x,
                                            std::min(e.segment.a.y,e.segment.b.y)-e.offset.y},
                                           {std::max(e.segment.a.x,e.segment.b.x)-e.offset.x,
                                            std::max(e.segment.a.y,e.segment.b.y)-e.offset.y}}};
            };
            auto split=[&](auto&& self,std::size_t begin,std::size_t end)->std::size_t {
                Node node;node.begin=begin;node.end=end;
                auto first=bounds(order[begin]);node.low=first[0];node.high=first[1];
                for(auto i=begin+1;i<end;++i) {
                    auto box=bounds(order[i]);node.low.x=std::min(node.low.x,box[0].x);
                    node.low.y=std::min(node.low.y,box[0].y);node.high.x=std::max(node.high.x,box[1].x);
                    node.high.y=std::max(node.high.y,box[1].y);
                }
                auto index=nodes.size();nodes.push_back(node);
                if(end-begin>6) {
                    bool x_axis=node.high.x-node.low.x>=node.high.y-node.low.y;
                    auto middle=(begin+end)/2;
                    std::nth_element(order.begin()+begin,order.begin()+middle,order.begin()+end,[&](auto a,auto b) {
                        auto aa=bounds(a),bb=bounds(b);
                        double ac=x_axis?aa[0].x+aa[1].x:aa[0].y+aa[1].y;
                        double bc=x_axis?bb[0].x+bb[1].x:bb[0].y+bb[1].y;
                        return ac<bc || (ac==bc && a<b);
                    });
                    nodes[index].left=self(self,begin,middle);nodes[index].right=self(self,middle,end);
                }
                return index;
            };
            split(split,0,order.size());
        }
        bool contains(Point p) const {return ready && dot(p-center,p-center)<=radius*radius;}
    };
    // A query at most r from center has a nearest segment at most d(center)+2r
    // from center. Gathering that disk is exact, not a sampled/interpolated field.
    // Excessively distant/dense coasts use the ordinary index instead; scratch
    // storage stays bounded at 2048 edges per tile, never a second world cache.
    template<class Observe>
    Patch prepare(Point corner,double radius,double center_distance,Observe observe) const {
        Patch result;result.center=corner;result.radius=radius;
        if(!ready || !std::isfinite(center_distance) || center_distance>8)return result;
        Point p=corner-Point{.5,.5};
        auto dimensions=topology.dimensions();unsigned copy=0;
        result.edges.reserve(256);
        auto gather=[&](Point offset) {
            unsigned ordinal=copy++;
            return coast.gather(p+offset,center_distance+2*radius+1e-9,observe,
                [&](auto id,auto const& segment) {
                    if(result.edges.size()==2048)return false;
                    result.edges.push_back({segment,offset,id,ordinal});return true;
                });
        };
        bool complete=gather({});
        int nx=dimensions.wrap_x?1:0,ny=dimensions.wrap_y?1:0;
        for(int y=-ny;complete && y<=ny;y++)for(int x=-nx;complete && x<=nx;x++) {
            if(x==0 && y==0)continue;
            complete=gather({(x*dimensions.width+y*dimensions.height)*.5,
                             (x*dimensions.width-y*dimensions.height)*.5});
        }
        result.ready=complete && !result.edges.empty();
        if(result.ready)result.build();else result.edges.clear();
        return result;
    }
    struct Update { std::size_t topology_changes=0,cells_built=0,bytes=0; };
    void clear() { topology.clear(); coast.clear(); source_revision=-1; ready=false; }
    WorldTopology const& world() const { return topology; }
    std::uint64_t node_revision(std::uint64_t id) const { return coast.revision(id); }
    template<class Observe>
    std::vector<CoastSegment> cell(int c,int r,Observe observe) const {
        auto dimensions=topology.dimensions();int x=c+r,y=c-r;
        if(dimensions.wrap_x)x=mod(x,dimensions.width);
        if(dimensions.wrap_y)y=mod(y,dimensions.height);
        int cc=(x+y)/2,rr=(x-y)/2;
        auto segments=coast.cell(cc,rr,observe);Point offset{double(c-cc),double(r-rr)};
        for(auto& segment:segments){segment.a=segment.a+offset;segment.b=segment.b+offset;}
        return segments;
    }
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
    ShoreSample sample(Point corner,ObserveNode observe_node,ObserveTile observe_tile, Patch const* patch=nullptr) const {
        auto lookup=[&](int c,int r) {
            auto index=topology.index(c,r);
            if(index!=std::size_t(-1)) observe_tile(index,topology.at(index));
            return topology.tile(c,r);
        };
        return sample_with_lookup(corner,observe_node,lookup,patch);
    }
    // Production supplies its per-tile exact lookup cache. It records each
    // authoritative input on first access, without repeated wrap/hash work.
    template<class ObserveNode,class Lookup>
    ShoreSample sample_with_lookup(Point corner,ObserveNode observe_node,Lookup lookup,Patch const* patch=nullptr) const {
        if(!ready) throw std::runtime_error("world coast field is not ready");
        World dimensions=topology.dimensions(); Point p=corner-Point{.5,.5};
        CoastIndex::Nearest nearest;Point foot=p;
        if(patch && patch->contains(corner)) {
            unsigned best_copy=~0u;std::size_t best_index=0;
            auto distance=[&](Patch::Node const& node) {
                double dx=std::max({node.low.x-p.x,0.,p.x-node.high.x});
                double dy=std::max({node.low.y-p.y,0.,p.y-node.high.y});return dx*dx+dy*dy;
            };
            auto visit=[&](auto&& self,std::size_t index)->void {
                auto const& node=patch->nodes[index];
                if(distance(node)>nearest.squared+1e-10)return;
                if(node.left) {
                    auto first=node.left,second=node.right;
                    if(distance(patch->nodes[second])<distance(patch->nodes[first]))std::swap(first,second);
                    self(self,first);self(self,second);return;
                }
                for(auto i=node.begin;i<node.end;++i) {
                    auto edge_index=patch->order[i];auto const& edge=patch->edges[edge_index];
                    Point query=p+edge.offset,ab=edge.segment.b-edge.segment.a;
                    double t=std::clamp(dot(query-edge.segment.a,ab)/std::max(1e-15,dot(ab,ab)),0.,1.);
                    Point q=edge.segment.a+ab*t;double d=dot(query-q,query-q);
                    if(d<nearest.squared || (d==nearest.squared &&
                        (edge.copy<best_copy || (edge.copy==best_copy &&
                         (edge.cell<nearest.cell || (edge.cell==nearest.cell && edge_index<best_index)))))) {
                        nearest={d,q,edge.cell};foot=q-edge.offset;best_copy=edge.copy;best_index=edge_index;
                    }
                }
            };
            visit(visit,0);
        } else {
        std::vector<CoastIndex::Certificate> certificates;certificates.reserve(64);
        nearest=coast.nearest(p,[](auto,auto){},std::numeric_limits<double>::infinity(),&certificates);
        foot=nearest.foot;
        int nx=dimensions.wrap_x ? 1 : 0,ny=dimensions.wrap_y ? 1 : 0;
        // Retain visited leaf/empty-subtree certificates with their distance.
        // Filtering against the final nearest distance is exactly the old
        // second traversal, without repeating its tree/hash/segment work.
        for(int y=-ny;y<=ny;y++) for(int x=-nx;x<=nx;x++) {
            if(x==0 && y==0)continue;
            Point offset{(x*dimensions.width+y*dimensions.height)*.5,
                         (x*dimensions.width-y*dimensions.height)*.5};
            auto candidate=coast.nearest(p+offset,[](auto,auto){},nearest.squared,&certificates);
            if(candidate.cell && candidate.squared<nearest.squared) {
                nearest=candidate; foot=candidate.foot-offset;
            }
        }
        for(auto const& certificate:certificates)
            if(certificate.squared<=nearest.squared)observe_node(certificate.id,certificate.revision);
        }
        ShoreField field(dimensions,lookup);
        double rocky=field.rockiness(foot);
        double distance=std::isfinite(nearest.squared) ? std::sqrt(nearest.squared) : 1e6;
        distance*=field.coverage(p)>=0 ? 1 : -1;
        return {distance,(.065+.18*field.world_noise(foot,.66,628))*(1-rocky),rocky,
            .46*(1-std::exp(-std::max(0.,-distance)/.85))};
    }
};
} }
