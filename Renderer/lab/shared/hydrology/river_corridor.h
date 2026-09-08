#pragma once
// Opt-in Lab river geometry. Native connectivity is unchanged. Curves, terminal
// presentation and spatial queries share one field in the tile-corner lattice.
#include "field.h"

namespace river {
using hydro::P;
struct Edge { uint64_t id; P a,b; std::vector<P> points; double cost=0,original_cost=0; };
struct Terminal { P p; bool mouth; unsigned profile=0; double yaw=0; };
struct Segment { P a,b; };
struct Sample { double distance=1000, source=1000, mouth=1000; };
inline P screen(P p) { return {64*(p.x+p.y),32*(p.x-p.y)}; }
inline P from_screen(P p) { return {p.x/128+p.y/64,p.x/128-p.y/64}; }
inline double distance(P p,P a,P b) {
    P d=b-a; double t=hydro::sat(hydro::dot(p-a,d)/std::max(1e-15,hydro::dot(d,d)));
    return hydro::length(p-(a+d*t));
}
struct Corridor {
    std::vector<Edge> edges;
    std::vector<Terminal> terminals;
    std::vector<std::vector<double>> pool_profiles;
    std::map<std::pair<int,int>,std::vector<Segment>> buckets;
    std::map<std::pair<int,int>,std::vector<unsigned>> terminal_buckets;

    void load_pool_profiles(char const* path) {
        std::ifstream in(path);std::string line,magic;unsigned count=0,samples=0;
        if(!std::getline(in,line))throw std::runtime_error("Missing river pool profile");
        std::replace(line.begin(),line.end(),',',' ');std::istringstream header(line);
        header>>magic>>count>>samples;
        if(magic!="C3X_RIVER_POOL_PROFILE_V1" || count<1 || count>16 || samples<8 || samples>256)
            throw std::runtime_error("Invalid river pool profile header");
        pool_profiles.clear();
        for(unsigned i=0;i<count;++i) {
            if(!std::getline(in,line))throw std::runtime_error("Missing river pool profile row");
            std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
            std::vector<double> radii(samples);
            for(double& r:radii)if(!(row>>r) || !std::isfinite(r) || r<.2 || r>2)
                throw std::runtime_error("Invalid river pool radius");
            pool_profiles.push_back(radii);
        }
    }

    double pool_radius(Terminal const& t,P delta) const {
        if(pool_profiles.empty())return .22;
        auto const& radii=pool_profiles[t.profile];
        double angle=(std::atan2(delta.y,delta.x)-t.yaw)/6.283185307179586;
        double u=(angle-std::floor(angle))*radii.size();
        unsigned i=unsigned(u);double fraction=u-i;
        return .22*(radii[i%radii.size()]*(1-fraction)+radii[(i+1)%radii.size()]*fraction);
    }

    void insert(P a,P b) {
        for(int y=int(std::floor(std::min(a.y,b.y)-.65));y<=int(std::floor(std::max(a.y,b.y)+.65));++y)
        for(int x=int(std::floor(std::min(a.x,b.x)-.65));x<=int(std::floor(std::max(a.x,b.x)+.65));++x)
            buckets[{x,y}].push_back({screen(a),screen(b)});
    }
    bool affects(int c,int r) const { return buckets.count({c,r}) || terminal_buckets.count({c,r}); }
    bool bank_point(P near,double margin,double side,P& result) const {
        auto found=buckets.find({int(std::floor(near.x)),int(std::floor(near.y))});
        if(found==buckets.end())return false;
        P p=screen(near),foot,normal;double best=1e9;
        for(auto const& s:found->second) {
            P d=s.b-s.a;double length=hydro::length(d);if(length<1e-10)continue;
            P q=s.a+d*hydro::sat(hydro::dot(p-s.a,d)/(length*length));
            double distance=hydro::length(p-q);
            if(distance<best) { best=distance;foot=q;normal={-d.y/length,d.x/length}; }
        }
        if(best==1e9)return false;
        result=from_screen(foot+normal*(margin*side));
        return true;
    }
    Sample sample(P p) const {
        Sample result;
        auto key=std::make_pair(int(std::floor(p.x)),int(std::floor(p.y)));
        auto found=buckets.find(key);
        if(found!=buckets.end())for(auto const& s:found->second)
            result.distance=std::min(result.distance,distance(screen(p),s.a,s.b));
        auto nodes=terminal_buckets.find(key);
        if(nodes!=terminal_buckets.end())for(unsigned index:nodes->second) {
            auto const& t=terminals[index];
            double d=hydro::length(p-t.p)*64;
            if(t.mouth)result.mouth=std::min(result.mouth,d);
            else {
                result.source=std::min(result.source,d);
                // Small inland pool, continuous with the incident channel.
                // Presentation only: no water tile or gameplay terrain change.
                result.distance=std::min(result.distance,d-(pool_radius(t,p-t.p)*64-6));
            }
        }
        return result;
    }

    template<class Height> void build(char const* csv,Height height) {
        edges.clear();terminals.clear();buckets.clear();terminal_buckets.clear();
        hydro::Field field;field.load(csv);field.wraps=true;
        std::map<uint64_t,Edge> unique;
        // Full supplied halo supplies incident edges at viewport boundaries.
        for(auto const& item:field.tiles) {
            auto const& t=item.second;
            for(unsigned bit:{2u,8u,32u,128u})if(t.river&bit) {
                P a,b;
                if(bit==2){a={double(t.c),double(t.r+1)};b={double(t.c+1),double(t.r+1)};}
                if(bit==8){a={double(t.c+1),double(t.r)};b={double(t.c+1),double(t.r+1)};}
                if(bit==32){a={double(t.c),double(t.r)};b={double(t.c+1),double(t.r)};}
                if(bit==128){a={double(t.c),double(t.r)};b={double(t.c),double(t.r+1)};}
                auto id=field.edge_id(a-P{.5,.5},b-P{.5,.5});
                unique.emplace(id,Edge{id,a,b,{}});
            }
        }
        for(auto const& item:unique)edges.push_back(item.second);
        std::map<std::pair<int,int>,std::vector<unsigned>> incident;
        for(unsigned i=0;i<edges.size();++i)for(P p:{edges[i].a,edges[i].b})
            incident[{int(p.x),int(p.y)}].push_back(i);
        auto tangent=[&](P p,P toward,unsigned index) {
            auto const& neighbors=incident.at({int(p.x),int(p.y)});
            P d=toward-p;
            if(neighbors.size()==2)for(unsigned other:neighbors)if(other!=index) {
                auto const& e=edges[other];
                P end=hydro::length(e.a-p)<1e-9?e.b:e.a;
                d=d-(end-p);
            }
            return d*(1/std::max(1e-12,hydro::length(d)));
        };
        for(unsigned i=0;i<edges.size();++i) {
            auto& e=edges[i];P d=e.b-e.a,n{-d.y,d.x};
            P c1=e.a+tangent(e.a,e.b,i)*.34,c2=e.b+tangent(e.b,e.a,i)*.34;
            auto point=[&](double u,double bend) {
                double v=1-u;
                return e.a*(v*v*v)+c1*(3*v*v*u)+c2*(3*v*u*u)+e.b*(u*u*u)+n*(bend*16*u*u*v*v);
            };
            // Endpoint derivatives remain shared while the interior bow can
            // move around a source hill/mountain. Heights are sampled BEFORE
            // river carving, so flattening cannot bias the route selection.
            auto score=[&](double bend) {
                double cost=0;
                for(int j=1;j<12;++j) {
                    P p=point(j/12.,bend);
                    double h=std::max(0.,double(height(p.x,p.y))-2.5);
                    cost+=h*h;
                }
                return cost;
            };
            double preferred=(double(hydro::hash(uint32_t(e.id)^uint32_t(e.id>>32)))/4294967295.-.5)*.16;
            double best=preferred, best_score=score(best);
            e.original_cost=best_score;
            for(double bend:{-.24,-.16,-.08,0.,.08,.16,.24}) {
                double cost=score(bend);
                // On almost flat land keep the source-stable organic bend.
                if(cost+2*std::abs(bend-preferred)<best_score) { best=bend;best_score=cost; }
            }
            e.cost=best_score;
            for(int j=0;j<=32;++j)e.points.push_back(point(j/32.,best));
            for(unsigned j=1;j<e.points.size();++j)insert(e.points[j-1],e.points[j]);
        }
        for(auto const& item:incident)if(item.second.size()==1) {
            P p{double(item.first.first),double(item.first.second)};
            bool complete=true;std::vector<P> water;
            for(int y=int(p.y)-1;y<=int(p.y);++y)for(int x=int(p.x)-1;x<=int(p.x);++x) {
                auto t=field.tiles.find({x,y});
                if(t==field.tiles.end()){complete=false;continue;}
                if(hydro::water(t->second.base))water.push_back({x+.5,y+.5});
            }
            if(!complete)continue; // Unknown outside halo is not a headwater.
            bool mouth=!water.empty();
            if(mouth) {
                auto const& e=edges[item.second.front()];
                P away=p-(hydro::length(e.a-p)<1e-9?e.points[1]:e.points[e.points.size()-2]);
                auto chosen=std::max_element(water.begin(),water.end(),[&](P a,P b){return hydro::dot(a-p,away)<hydro::dot(b-p,away);});
                // The water-tile center lies beyond the optical shore, allowing
                // the same corridor to cut through the beach and enter the sea.
                insert(p,*chosen);
            }
            auto id=edges[item.second.front()].id;
            auto const& edge=edges[item.second.front()];
            uint32_t endpoint=hydro::length(edge.a-p)<1e-9?0x736f7572u:0x706f6f6cu;
            uint32_t seed=hydro::hash(uint32_t(id)^uint32_t(id>>32)^endpoint);
            unsigned profile=pool_profiles.empty()?0:seed%pool_profiles.size();
            double yaw=6.283185307179586*double(hydro::hash(seed))/4294967295.;
            unsigned index=unsigned(terminals.size());terminals.push_back({p,mouth,profile,yaw});
            for(int y=int(p.y)-1;y<=int(p.y);++y)for(int x=int(p.x)-1;x<=int(p.x);++x)
                terminal_buckets[{x,y}].push_back(index);
        }
    }
};
}
