#pragma once
// Civ III mountain relief, shared by the relief mesh, route surfaces and
// resource seating so routes and objects sit exactly on the rendered rock.
// Every mountain tile places one authored macro stamp, stretched along its
// range and turned so the stamp's main ridge follows it; every pair of
// adjacent mountains (edge or corner) adds a lower saddle stamp between them,
// so ranges join into one massif. Pieces combine with an order-independent
// smooth maximum of the two highest.
//
// Volcanoes are part of the same relief: a clean stratovolcano cone (concave
// flanks, shallow crater) carrying the gullies, ridges and footprint of Civ
// VI's dedicated volcano element (when the pack provides it), rising little
// above its own tile on screen, turned and mirrored per tile, and joined to neighbouring
// mountains and volcanoes by the same saddles.
// Samples carry the volcano's texture position, coverage, lava-channel mask
// and Civ III activity (0 dormant, 1 smoldering, 2 erupting) for its material.
//
// A query for tile (c,r) uses the main stamps of its 3x3 window and the
// saddles between members of that window. Mountain stamps reach at most 1.06,
// volcano stamps .88 and saddles .96 tiles from their centre, so no other
// stamp touches the tile or its one-step mesh halo: neighbouring tiles agree
// on shared edges. A tile without a mountain or volcano in its 3x3 window
// reads nothing further.
#include <array>
#include <cmath>
#include <cstdint>
#include "data.h"
namespace c3x_renderer { namespace fidelity {
// Topology flags a MountainShape reads besides the terrain category.
enum MountainFlag : unsigned { mountain_snow_cap=1u, volcano_active=2u, volcano_erupting=4u };
struct MountainCell { bool mountain=false,volcano=false,snow=false; unsigned activity=0; std::uint32_t seed=0; };
struct MountainPiece {
    HeightField const* height_field=nullptr;HeightField const* blend_field=nullptr;HeightField const* channel_field=nullptr;
    float center_x=0,center_y=0,range_cos=1,range_sin=0,turn_cos=1,turn_sin=0,inv_long=1,inv_cross=1,
        height=0,reach=0,snow=0;
    unsigned activity=0;
    bool mirror=false;
};
struct MountainSample {
    float displacement=0,dominant=0,height=0,u=0,v=0,snow=0;
    // The highest volcano stamp here: its coverage over other pieces, its
    // texture position, authored lava channel and Civ III activity.
    float volcano=0,volcano_u=.5f,volcano_v=.5f,channel=0;unsigned activity=0;
};
struct MountainShape {
    // Main stamp span and its stretch along a straight range. Height is the
    // accepted lower shape (165 x .68); saddles are lower, shorter stamps.
    static constexpr float span=2.3f,range_stretch=1.15f,height=112.2f,joined_height=1.03f,
        height_jitter=.14f,saddle_height=.50f,corner_saddle=.90f,saddle_long=1.7f,saddle_cross=1.6f,
        blend_k=14.f,reach_scale=.40f,
        // Volcano: on screen the rim stays near its tile diamond's top corner
        // (84 units x .47 px = 39 px above the centre at 1x; the corner is at
        // 32 px). Profile radii are in element units (span 2.2 puts the foot
        // .64 tiles out): crater rim,
        // foot, flank concavity, crater depth, the share of authored detail
        // kept about its radial mean, and the authored footprint softness.
        volcano_span=2.2f,volcano_height=84.f,volcano_jitter=.04f,volcano_rim=.055f,volcano_foot=.29f,
        volcano_concave=1.4f,volcano_crater=.22f,volcano_detail=.55f,volcano_footprint=.35f;
    std::array<MountainPiece,29> pieces{};
    unsigned count=0;
    static std::uint32_t mix(std::uint32_t x){
        x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;
    }
    static float unit(std::uint32_t seed,std::uint32_t salt){
        return float(mix(seed*0x9e3779b1u+salt*0x85ebca6bu)>>8)*(1.f/16777216.f);
    }
    // theta: range direction in world (column,row). The stamp turns so its
    // principal ridge (natural.macro_axis, stamp frame u right, v up) lies on
    // theta; a seeded mirror and half turn add variety.
    void add(NaturalData const& natural,unsigned variant,float x,float y,float theta,
             float long_span,float cross_span,float piece_height,float snow,std::uint32_t seed){
        auto& p=pieces[count++];
        p.mirror=unit(seed,11)<.5f;
        float axis=natural.macro_axis[variant];
        float turn=(p.mirror?-axis:axis)+(unit(seed,12)<.5f?3.14159265f:0.f);
        p.height_field=&natural.fields[natural.macro[variant][0]];p.blend_field=&natural.fields[natural.macro[variant][1]];
        p.center_x=x;p.center_y=y;p.range_cos=std::cos(theta);p.range_sin=std::sin(theta);
        p.turn_cos=std::cos(turn);p.turn_sin=std::sin(turn);
        p.inv_long=1/long_span;p.inv_cross=1/cross_span;p.height=piece_height;
        p.reach=reach_scale*std::max(long_span,cross_span);p.snow=snow;
    }
    // A volcano: the dedicated stamp, uniform span, any seeded turn.
    void add_volcano(NaturalData const& natural,float x,float y,unsigned activity,std::uint32_t seed){
        auto& p=pieces[count++];
        p.mirror=unit(seed,11)<.5f;
        float turn=unit(seed,12)*6.2831853f;
        p.height_field=&natural.volcano_height;p.blend_field=&natural.volcano_blend;p.channel_field=&natural.volcano_channel;
        p.center_x=x;p.center_y=y;p.turn_cos=std::cos(turn);p.turn_sin=std::sin(turn);
        p.inv_long=p.inv_cross=1/volcano_span;
        p.height=volcano_height*(1-volcano_jitter+2*volcano_jitter*unit(seed,3));
        p.reach=reach_scale*volcano_span;p.activity=activity;
    }
    MountainShape()=default;
    // lookup(c,r) returns the natural Tile at any coordinate; flags(c,r) its
    // MountainFlag bits (read only for mountains and volcanoes).
    template<class Lookup,class Flags> MountainShape(NaturalData const& natural,int nc,int nr,Lookup lookup,Flags flags){
        std::array<MountainCell,25> cells{};std::array<bool,25> ready{};
        auto cell=[&](int dc,int dr)->MountainCell const&{
            unsigned i=unsigned((dr+2)*5+dc+2);
            if(!ready[i]){
                auto t=lookup(nc+dc,nr+dr);bool volcano=t.real==10,mountain=t.real==6 || volcano;
                unsigned bits=mountain?unsigned(flags(nc+dc,nr+dr)):0u;
                cells[i]=MountainCell{mountain,volcano,!volcano && (bits&mountain_snow_cap),
                    volcano?((bits&volcano_erupting)?2u:(bits&volcano_active)?1u:0u):0u,mountain_seed(t)};
                ready[i]=true;
            }
            return cells[i];
        };
        bool any=false;
        for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc)any=cell(dc,dr).mountain||any;
        if(!any)return;
        constexpr int around[8][2]={{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
        for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){
            auto const owner=cell(dc,dr);if(!owner.mountain)continue;
            float sx=0,sy=0;bool joined=false;
            for(auto const& n:around){
                if(!cell(dc+n[0],dr+n[1]).mountain)continue;
                float a=2*std::atan2(float(n[1]),float(n[0]));sx+=std::cos(a);sy+=std::sin(a);joined=true;
            }
            std::uint32_t seed=mix(owner.seed);
            if(owner.volcano && natural.volcano_ready){
                add_volcano(natural,float(nc+dc)+.5f,float(nr+dr)+.5f,owner.activity,seed);
                continue;
            }
            bool aligned=std::hypot(sx,sy)>.9f;
            float theta=aligned?.5f*std::atan2(sy,sx):unit(seed,7)*6.2831853f;
            float jitter=1-height_jitter+2*height_jitter*unit(seed,3);
            add(natural,seed%5u,float(nc+dc)+.5f,float(nr+dr)+.5f,theta,
                span*(aligned?range_stretch:1.f),span,height*(joined?joined_height:1.f)*jitter,
                owner.snow?1.f:0.f,seed);
        }
        constexpr int forward[4][2]={{1,0},{0,1},{1,1},{1,-1}};
        for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){
            auto const a=cell(dc,dr);if(!a.mountain)continue;
            for(auto const& f:forward){
                int bc=dc+f[0],br=dr+f[1];
                if(std::abs(bc)>1 || std::abs(br)>1)continue;
                auto const b=cell(bc,br);if(!b.mountain)continue;
                std::uint32_t seed=mix(a.seed^mix(b.seed+0x2c1bu));
                bool corner=f[0]!=0 && f[1]!=0;
                float jitter=1-height_jitter+2*height_jitter*unit(seed,5);
                add(natural,seed%5u,float(nc+dc)+.5f+.5f*float(f[0]),float(nr+dr)+.5f+.5f*float(f[1]),
                    std::atan2(float(f[1]),float(f[0])),(corner?1.41421356f:1.f)*saddle_long,saddle_cross,
                    height*saddle_height*(corner?corner_saddle:1.f)*jitter,
                    .5f*(float(a.snow)+float(b.snow)),seed);
            }
        }
    }
    // Pieces point into the NaturalData they were built from.
    MountainSample sample(NaturalData const& natural,float x,float y) const {
        MountainSample result;float second=0,volcano=0,other=0;
        for(unsigned i=0;i<count;++i){
            auto const& p=pieces[i];
            float dx=x-p.center_x,dy=y-p.center_y;
            if(std::abs(dx)>=p.reach || std::abs(dy)>=p.reach)continue;
            float distance=std::sqrt(dx*dx+dy*dy);
            if(distance>=p.reach)continue;
            // Range frame (stretched), then the stamp's own turn.
            float a=(dx*p.range_cos+dy*p.range_sin)*p.inv_long;
            float b=(-dx*p.range_sin+dy*p.range_cos)*p.inv_cross;
            float lx=a*p.turn_cos-b*p.turn_sin,ly=a*p.turn_sin+b*p.turn_cos;
            float u=.5f+lx,v=p.mirror?.5f+ly:.5f-ly;
            if(u<0||u>1||v<0||v>1)continue;
            float h=p.height_field->sample(u,v);
            float blend=p.blend_field->sample(u,v);
            float shape=std::max(0.f,h)*smooth01((blend-.28f)/.34f);
            if(p.channel_field){
                // The authored element alone reads as a lumpy, flat-topped
                // mound at Civ III scale; keep its detail on a clean cone
                // whose foot eases into the ground.
                float r=std::sqrt(lx*lx+ly*ly);
                float t=std::clamp((r-volcano_rim)/(volcano_foot-volcano_rim),0.f,1.f);
                float cone=r<volcano_rim?1-volcano_crater*(1-smooth01(r/volcano_rim)):std::pow(1-t,volcano_concave);
                float at=std::clamp(r*128-.5f,0.f,62.999f);unsigned bin=unsigned(at);
                float trend=natural.volcano_trend[bin]+(natural.volcano_trend[bin+1]-natural.volcano_trend[bin])*(at-float(bin));
                shape=std::max(0.f,cone+volcano_detail*(h-trend))*smooth01(blend/volcano_footprint);
            }
            // The reach fade only enforces the window bound; authored blend
            // is already zero there for the shipped stamps.
            float displacement=shape*p.height*smooth01((p.reach-distance)/(.1f*p.reach));
            if(displacement<=0)continue;
            if(p.channel_field){
                if(displacement>volcano){
                    volcano=displacement;result.volcano_u=u;result.volcano_v=v;
                    result.channel=p.channel_field->sample(u,v);result.activity=p.activity;
                }
            }else other=std::max(other,displacement);
            if(displacement>result.dominant){
                second=result.dominant;result.dominant=displacement;
                result.u=u;result.v=v;result.snow=p.snow;
            }else second=std::max(second,displacement);
        }
        if(result.dominant<=0)return result;
        // Polynomial smooth maximum of the two highest pieces: saddles rise
        // where two stamps meet at similar height, without a crease, and the
        // result does not depend on piece order. The rise eases in with the
        // second stamp so its footprint edge cannot step.
        float t=std::max(0.f,blend_k-(result.dominant-second))/blend_k;
        result.displacement=result.dominant+blend_k*t*t*.25f*smooth01(second/blend_k);
        result.height=std::min(1.f,result.displacement/height);
        // The volcano owns its material where it stands above the mountain
        // stamps and saddles around it, easing across their meeting line.
        if(volcano>0)result.volcano=smooth01((volcano-other)/14.f+.5f)*smooth01(volcano/6.f);
        return result;
    }
};
} }
