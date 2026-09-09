#pragma once
#include "terrain_query.h"

namespace c3x_renderer { namespace render_core {
struct ReliefSample { float height=0,blend=0,displacement=0; };
struct GroundSample {
    float height=0,authored_height=0,authored_blend=0;
    std::array<float,4> owner{};
};
// Exact zero-height certificate for one tile plus the normal-sampling collar.
// Every inspected tile still enters the caller's dependency ledger, so adding
// relief/dunes or bringing a coast closer invalidates the cached mesh.
struct FlatGroundRegion {
    float u, v;
    bool certified = false;
    template<class Lookup>
    FlatGroundRegion(int c, int r, double center_shore_distance, Lookup const& lookup)
        : u(float(c)), v(float(r)) {
        // Farthest query is < .722 tiles from center. At distance > .85,
        // the cliff shoulder is exactly zero; use a conservative 1.6 bound.
        if (center_shore_distance <= 1.6) return;
        certified = true;
        for (int y=-2;y<=2;y++) for (int x=-2;x<=2;x++) {
            Tile t=lookup(c+x,r+y);
            if (!t.present || t.real==5 || t.real==6 || t.real==10 ||
                (t.base==0 && t.real==0)) certified=false;
        }
    }
    bool contains(float x, float y) const {
        return certified && x>=u-.01f && x<=u+1.01f && y>=v-.01f && y<=v+1.01f;
    }
};
inline float smooth01(float x) { x=std::clamp(x,0.f,1.f); return x*x*(3-2*x); }
inline float smooth_max(float a,float b) {
    if(a<=.001f || b<=.001f) return std::max(a,b);
    float w=std::clamp(.5f+.5f*(b-a)/12.f,0.f,1.f);
    return a*(1-w)+b*w+12*w*(1-w);
}
inline float repeated_frequency(float requested,World world) {
    int period=world.wrap_x ? world.width : world.wrap_y ? world.height : 0;
    if(world.wrap_x && world.wrap_y) {
        int a=world.width,b=world.height;
        while(b) {int r=a%b; a=b; b=r;} period=a;
    }
    return period ? std::max(1.f,std::round(requested*period*.5f))/(period*.5f) : requested;
}

// Source(kind,variant,channel,u,v) samples the normalized selected source field.
// Kind 5 is the selected direct hill; 6 uses five mountain variants; 10 is the
// aligned ordinary volcano. Channel 0 is height, channel 1 is authored coverage.
template<class Lookup,class Source,class Shore,class River,class Dune,class Activity>
class ReliefQuery {
    World world;
    Lookup lookup; Source source; Shore shore; River river; Dune dune; Activity activity;
    mutable bool quiet_ready=false,quiet=false;
    mutable int quiet_c=0,quiet_r=0;
    bool quiet_neighborhood(int c,int r) const {
        if(quiet_ready && c==quiet_c && r==quiet_r)return quiet;
        quiet=true;quiet_c=c;quiet_r=r;quiet_ready=true;
        for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
            Tile t=lookup(c+dx,r+dy);
            if(t.real==5 || t.real==6 || t.real==10 || (t.present && t.base==0 && t.real==0))quiet=false;
        }
        return quiet;
    }
    std::array<int,2> raw(int c,int r) const {
        int x=c+r,y=c-r;
        if(world.wrap_x) x=mod(x,world.width);
        if(world.wrap_y) y=mod(y,world.height);
        return {x,y};
    }
public:
    ReliefQuery(World w,Lookup l,Source s,Shore h,River r,Dune d,Activity a)
        :world(w),lookup(l),source(s),shore(h),river(r),dune(d),activity(a) {}
    ReliefSample body(int c,int r,float x,float y) const {
        Tile tile=lookup(c,r); ReliefSample result;
        if(!tile.present || (tile.real!=6 && tile.real!=10)) return result;
        auto coordinate=raw(c,r);
        unsigned seed=unsigned(coordinate[0]*73+coordinate[1]*151);
        unsigned variant=0;
        float scale=tile.real==6 ? float(mountain_scale) : float(volcano_scale);
        float footprint=float(volcano_footprint);
        if(tile.real==6) {
            variant=(seed>>3)%5;
            if(seed&1) std::swap(x,y);
            if(seed&2) x=1-x;
            if(seed&4) y=1-y;
            bool connected=false;
            for(auto offset:{std::array<int,2>{-1,0},{1,0},{0,-1},{0,1}}) {
                int real=lookup(c+offset[0],r+offset[1]).real;
                connected=connected || real==6 || real==10;
            }
            footprint=(connected ? .50f : .68f)/scale;
        }
        float u=.5f+(x-.5f)*footprint,v=.5f+(y-.5f)*footprint;
        if(u<0 || v<0 || u>1 || v>1) return result;
        float edge=smooth01(std::min({u,v,1-u,1-v})/.055f);
        result.height=source(tile.real,variant,0,u,v);
        result.blend=source(tile.real,variant,1,u,v)*edge;
        return result;
    }
    ReliefSample chain(float x,float y,bool include_volcano,bool* contains_volcano=nullptr) const {
        int c=int(std::floor(x)),r=int(std::floor(y)); ReliefSample result;
        int candidates[][2]={{0,0},{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
        for(auto const& offset:candidates) {
            int owner_c=c+offset[0],owner_r=r+offset[1];
            Tile tile=lookup(owner_c,owner_r);
            if(contains_volcano && tile.present && tile.real==10)*contains_volcano=true;
            if(!tile.present || (tile.real!=6 && (tile.real!=10 || !include_volcano))) continue;
            float u=x-owner_c,v=1-(y-owner_r);
            float support=1-smooth01((std::max(std::abs(u-.5f),std::abs(v-.5f))-.52f)/.23f);
            if(support<=0) continue;
            auto candidate=body(owner_c,owner_r,u,v); candidate.blend*=support;
            auto coordinate=raw(owner_c,owner_r);
            unsigned seed=unsigned(coordinate[0])*73856093u^unsigned(coordinate[1])*19349663u;
            float scale=tile.real==6 ? float(mountain_scale) : float(volcano_scale);
            candidate.displacement=candidate.height*smooth01(candidate.blend/.34f)*
                (tile.real==6 ? 104.f : ((seed>>3)&1) ? 104.f : 88.f)*scale*support;
            if(candidate.displacement>result.displacement) result=candidate;
        }
        return result;
    }
    GroundSample sample(float x,float y,bool with_material=true) const {
        int c=int(std::floor(x)),r=int(std::floor(y)); Tile tile=lookup(c,r);
        GroundSample result;
        if(!tile.present || water(tile)) return result;
        float u=x-c,v=1-(y-r);
        float distances[]={u,v,1-u,1-v};
        int offsets[][2]={{-1,0},{0,1},{1,0},{0,-1}};
        float compatibility=1;
        for(int i=0;i<4;i++) {
            Tile neighbor=lookup(c+offsets[i][0],r+offsets[i][1]);
            if(!neighbor.present || water(neighbor)) compatibility*=smooth01(distances[i]/.22f);
        }
        auto hydrology=shore(x,y);
        // Beyond the cliff shoulder, a neighborhood without relief or dunes
        // contributes exactly zero. Observe its full support once per integer
        // cell; normal samples may differ by tiny fractions but share this fact.
        if(hydrology.distance>1.0 && quiet_neighborhood(c,r))return result;
        float signed_shore=float(std::clamp(-hydrology.distance/.65,-1.,1.));
        float coastal=smooth01((-signed_shore-.02f)/.42f);
        float rocky=smooth01((float(hydrology.rocky)-.55f)/.40f);
        float cliff=smooth01((float(hydrology.distance)-.01f)/.06f);
        float shoulder=1-smooth01((float(hydrology.distance)-.20f)/.65f);
        result.height=(5.f/12.f)*112.f*rocky*cliff*shoulder;
        float support=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
            if(lookup(c+dx,r+dy).real!=5) continue;
            float ox=(x-(c+dx+.5f))/.92f,oy=(y-(r+dy+.5f))/.78f;
            support=std::max(support,smooth01((1-std::sqrt(ox*ox+oy*oy))/.42f));
        }
        float frequency=repeated_frequency(.08f,world);
        float hill=support>0 ? source(5,0,0,.11f+x*frequency,.19f+y*frequency)*52.f*
            support*compatibility*(coastal*(1-rocky)+cliff*rocky)*(5.f/7.f) : 0;
        bool contains_volcano=false;
        auto relief=chain(x,y,true,&contains_volcano);
        result.height+=smooth_max(hill,relief.displacement*coastal*compatibility);
        // Existing analytic dune body is retained as a diagnostic proxy. Only
        // its continuous ownership/collar is ported; no source recovery claimed.
        int left=int(std::floor(x-.5f)),top=int(std::floor(y-.5f));
        float sx=smooth01(x-.5f-left),sy=smooth01(y-.5f-top),desert=0;
        for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++) {
            Tile t=lookup(left+dx,top+dy);
            if(t.present && t.base==0 && t.real==0) desert+=(dx?sx:1-sx)*(dy?sy:1-sy);
        }
        if(desert>0)result.height+=dune(x,y)*desert*smooth01((-signed_shore-.20f)/.42f)*compatibility;
        float valley=1-smooth01((river(c,r,u,v)-4.f)/16.f);
        result.height*=1-valley*.92f*.90f;
        // Normal finite differences consume height only. Root material/owner
        // sampling is unrelated work, and absent volcanoes make both chains equal.
        if(!with_material)return result;
        auto material=(tile.real==10 || !contains_volcano) ? relief : chain(x,y,false);
        result.authored_height=material.height;
        result.authored_blend=material.blend*coastal*compatibility;
        float nearest=1e6f;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
            int owner_c=c+dx,owner_r=r+dy;
            if(lookup(owner_c,owner_r).real!=10) continue;
            float ox=x-owner_c-.5f,oy=y-owner_r-.5f,d=ox*ox+oy*oy;
            if(d>=nearest) continue;
            float owner_u=x-owner_c,owner_v=1-(y-owner_r);
            auto body_sample=body(owner_c,owner_r,owner_u,owner_v);
            float mask=body_sample.blend*float(relief_support(owner_u,owner_v))*coastal*compatibility;
            nearest=d;
            result.owner={.5f+ox*float(volcano_footprint),.5f-oy*float(volcano_footprint),
                          mask,activity(owner_c,owner_r)};
        }
        return result;
    }
};
} }
