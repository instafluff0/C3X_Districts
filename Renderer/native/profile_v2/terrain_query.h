#pragma once
// Production queries adapted from the pinned terrain/shoreline profile 2.
// Callers supply authoritative tile access with dependency tracking. No fixture
// origin, camera, file IO or mutable scene state belongs in these kernels.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace c3x_renderer { namespace profile_v2 {
constexpr unsigned visual_profile_revision = 1;
constexpr double mountain_scale = 1.30;
constexpr double volcano_scale = 1.60;
constexpr double volcano_footprint = .62 / volcano_scale;
struct Tile { int base = 2, real = 2; bool present = false; };
struct World { int width = 100, height = 100; bool wrap_x = false, wrap_y = false; };
struct Point {
    double x = 0, y = 0;
    Point operator+(Point b) const { return {x+b.x, y+b.y}; }
    Point operator-(Point b) const { return {x-b.x, y-b.y}; }
    Point operator*(double k) const { return {x*k, y*k}; }
};
inline double dot(Point a, Point b) { return a.x*b.x+a.y*b.y; }
inline double length(Point p) { return std::sqrt(dot(p,p)); }
inline double smooth(double a, double b, double x) {
    x=std::clamp((x-a)/(b-a),0.,1.); return x*x*(3-2*x);
}
inline double smoother(double x) {
    x=std::clamp(x,0.,1.); return x*x*x*(x*(x*6-15)+10);
}
inline int mod(int x, int n) { return (x%n+n)%n; }
inline std::uint32_t hash(std::uint32_t x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
inline bool water(Tile t) { return t.base>=11 && t.base<=13; }
inline int material(Tile t) {
    return t.real==9 ? 3 : t.real==4 ? 1 : t.base==0 ? 2 : t.base==1 ? 1 : t.base==3 ? 4 : 0;
}
inline double periodic(double x, double period) { return x-std::floor(x/period)*period; }
inline double wave(double x, double y, World world, int nx, int ny, double phase) {
    return std::sin(6.283185307179586*(nx*periodic(x,world.width)/world.width+
        ny*periodic(y,world.height)/world.height)+phase);
}
template<class Lookup>
std::array<double,5> material_weights(Point p, World world, Lookup const& lookup) {
    int nx=std::max(1,int(std::round(world.width/7.)));
    int ny=std::max(1,int(std::round(world.height/6.)));
    double raw_x=p.x+p.y-1, raw_y=p.x-p.y;
    double wx=.12*wave(raw_x,raw_y,world,nx,ny,.4)+.035*wave(raw_x,raw_y,world,nx*2,-ny,.7);
    double wy=.12*wave(raw_x,raw_y,world,nx,-ny,.9)+.035*wave(raw_x,raw_y,world,nx,ny*2,1.7);
    double gx=p.x-.5+wx, gy=p.y-.5+wy;
    int ix=int(std::floor(gx)), iy=int(std::floor(gy));
    double tx=smoother(gx-ix), ty=smoother(gy-iy);
    auto center=[&](int c,int r) {
        std::array<double,5> w{}; Tile t=lookup(c,r);
        if(!water(t)) { w[material(t)]=1; return w; }
        double total=0;
        for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
            Tile n=lookup(c+x,r+y);
            if(n.present && !water(n)) {
                double k=x && y ? .7 : 1.; w[material(n)]+=k; total+=k;
            }
        }
        if(total==0) w[0]=1; else for(auto& k:w) k/=total;
        return w;
    };
    std::array<double,5> result{};
    for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++) {
        auto w=center(ix+dx,iy+dy); double k=(dx?tx:1-tx)*(dy?ty:1-ty);
        for(int i=0;i<5;i++) result[i]+=w[i]*k;
    }
    return result;
}
inline double relief_support(double local_x,double local_y) {
    return 1-smooth(.52,.75,std::max(std::abs(local_x-.5),std::abs(local_y-.5)));
}
inline std::array<double,2> volcano_uv(double local_x,double local_y) {
    return {.5+(local_x-.5)*volcano_footprint,.5+(local_y-.5)*volcano_footprint};
}
template<class Height>
std::array<double,3> continuous_normal(Point p,double step,double units,Height const& height) {
    double dx=(height(p.x+step,p.y)-height(p.x-step,p.y))/(2*step*units);
    double dy=(height(p.x,p.y+step)-height(p.x,p.y-step))/(2*step*units);
    double n=std::sqrt(dx*dx+dy*dy+1); return {-dx/n,-dy/n,1/n};
}

struct ShoreSample { double distance=0, beach_width=0, rocky=0, depth=0; };
struct CoastSegment { Point a,b; double rocky=0; };

// Center-lattice coordinates: native corner queries subtract (.5,.5) once.
// The memoized contour cells are scratch data owned by a mesh compilation.
// Every lookup, including absent input, reaches the caller's dependency ledger.
template<class Lookup> class ShoreField {
    World world;
    Lookup lookup;
    std::map<std::pair<int,int>,double> coverage_nodes;
    std::map<std::pair<int,int>,std::vector<CoastSegment>> cells;
public:
    ShoreField(World world_,Lookup lookup_) : world(world_),lookup(lookup_) {}
    void clear_scratch() { coverage_nodes.clear(); cells.clear(); }
    double world_noise(Point p,double frequency,unsigned salt) const {
        // For two-axis wrapping choose a common integral period. This agrees
        // exactly with the retained horizontal-wrap profile and closes both
        // seams for production maps that also wrap vertically.
        int period=world.wrap_x ? world.width : world.wrap_y ? world.height : 0;
        if(world.wrap_x && world.wrap_y) {
            int a=world.width,b=world.height;
            while(b) { int r=a%b; a=b; b=r; } period=a;
        }
        if(period>0) frequency=std::max(1.,std::round(frequency*period*.5))/(period*.5);
        double gx=p.x*frequency,gy=p.y*frequency;
        int ix=int(std::floor(gx)),iy=int(std::floor(gy));
        double u=smooth(0,1,gx-ix),v=smooth(0,1,gy-iy);
        auto value=[&](int x,int y) {
            int rx=x+y,ry=x-y;
            if(world.wrap_x) rx=mod(rx,int(std::round(world.width*frequency)));
            if(world.wrap_y) ry=mod(ry,int(std::round(world.height*frequency)));
            return double(hash(std::uint32_t(rx)*73856093u^
                std::uint32_t(ry)*19349663u^salt))/4294967295.;
        };
        return (1-v)*((1-u)*value(ix,iy)+u*value(ix+1,iy))+
            v*((1-u)*value(ix,iy+1)+u*value(ix+1,iy+1));
    }
    double occupancy(Point p,bool hill=false) const {
        auto kernel=[](double t) {
            t=std::abs(t); return t<1 ? (4-6*t*t+3*t*t*t)/6 :
                t<2 ? (2-t)*(2-t)*(2-t)/6 : 0;
        };
        double sum=0,total=0; int x=int(std::floor(p.x)),y=int(std::floor(p.y));
        for(int j=y-1;j<=y+2;j++) for(int i=x-1;i<=x+2;i++) {
            double w=kernel((p.x-i)*1.25)*kernel((p.y-j)*1.25);
            if(w==0) continue;
            Tile t=lookup(i,j); if(!t.present) continue;
            total+=w; sum+=w*(hill ? t.real==5 : !water(t));
        }
        // Missing capture does not invent a coast. The caller records it and
        // refuses publication if the required authoritative halo is incomplete.
        return total>0 ? sum/total : 0;
    }
    double rockiness(Point p) const {
        return smooth(.10,.80,occupancy(p,true)/std::max(.001,occupancy(p)));
    }
    double coverage(Point p) const {
        auto displacement=[&](unsigned seed) {
            return (world_noise(p,.24,seed)-.5)*.68+
                (world_noise(p,.82,seed+1)-.5)*.38+
                (world_noise(p,2.40,seed+2)-.5)*.16;
        };
        Point q=p+Point{displacement(781),displacement(2183)};
        double shape=occupancy(q)-(.525+(world_noise(p,.38,8191)-.5)*.08);
        int x=int(std::floor(p.x+.5)),y=int(std::floor(p.y+.5));
        Tile t=lookup(x,y);
        if(t.present) {
            double core=1-smooth(.16,.36,length(p-Point{double(x),double(y)}));
            shape=shape*(1-core)+(water(t)?-.475:.475)*core;
        }
        return shape;
    }
    std::vector<CoastSegment> const& cell(int x,int y) {
        auto key=std::make_pair(x,y); auto existing=cells.find(key);
        if(existing!=cells.end()) return existing->second;
        std::vector<CoastSegment> result;
        auto node=[&](int i,int j) {
            auto k=std::make_pair(i,j); auto f=coverage_nodes.find(k);
            if(f!=coverage_nodes.end()) return f->second;
            return coverage_nodes.emplace(k,coverage({i*.125,j*.125})).first->second;
        };
        auto triangle=[&](int ax,int ay,int bx,int by,int cx,int cy) {
            Point p[]={{ax*.125,ay*.125},{bx*.125,by*.125},{cx*.125,cy*.125}},cut[3];
            double v[]={node(ax,ay),node(bx,by),node(cx,cy)}; int n=0;
            for(int i=0;i<3;i++) {
                int j=(i+1)%3;
                if((v[i]>0)!=(v[j]>0)) cut[n++]=p[i]+(p[j]-p[i])*(v[i]/(v[i]-v[j]));
            }
            if(n==2) result.push_back({cut[0],cut[1],rockiness((cut[0]+cut[1])*.5)});
        };
        for(int j=y*8;j<(y+1)*8;j++) for(int i=x*8;i<(x+1)*8;i++) {
            triangle(i,j,i+1,j,i+1,j+1); triangle(i,j,i+1,j+1,i,j+1);
        }
        return cells.emplace(key,std::move(result)).first->second;
    }
    // Caller supplies the complete relevant contour domain. It is intentionally
    // explicit: searching a changing visible crop would invalidate pixel reuse.
    ShoreSample sample(Point p,std::vector<CoastSegment> const& coast) const {
        double nearest=1e6; Point foot=p;
        for(auto const& e:coast) {
            Point ab=e.b-e.a;
            double t=std::clamp(dot(p-e.a,ab)/std::max(1e-15,dot(ab,ab)),0.,1.);
            Point q=e.a+ab*t; double distance=length(p-q);
            if(distance<nearest) { nearest=distance; foot=q; }
        }
        double rocky=rockiness(foot);
        double distance=nearest*(coverage(p)>=0 ? 1 : -1);
        return {distance,(.065+.18*world_noise(foot,.66,628))*(1-rocky),rocky,
            .46*(1-std::exp(-std::max(0.,-distance)/.85))};
    }
};
} } // namespace c3x_renderer::profile_v2
