#pragma once
// Combat effect director: which combat effects exist, where and since when.
//
// Inputs are Civ III's authoritative facts (a bombard hit or miss effect, a
// bomber's revealed bomb FLC, HP lost in a combat round) and the units as
// drawn (attack-clip progress crossing pack release markers). Each effect is
// a pack profile at a map anchor with a start time; particles are sampled
// statelessly (effect_sampler.h), so skipped frames need no simulation.
// Units carry generic pack metadata only (releases, munition, bearing,
// domain); nothing here names a unit. No GPU code: the sandbox draws `live()`.
#include "effect_pack.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace c3x_renderer::effects {

// Pack metadata per unit body (unit pack bindings).
struct Release { float phase=0; std::string profile; float x=0,y=0,z=0; };
struct Armament {
    std::string impact_set;          // munition family: profile impact/<set>/<outcome>
    float bearing=0;                 // line of fire off the bow, degrees (broadsides)
    float sync=-1,first=-1;          // native report phase; clip phase of the first release
    float bomb_blast_s=-1;           // first blast after the bomb FLC is revealed
    float flight_lift=0,turn_scale=1;
    bool sea=false,air=false;
    std::vector<Release> releases;
};

// Native action phase -> attack-clip phase so the clip's first release lands
// on Civ III's report: a later release skips into the clip (inside the action
// blend), an earlier one slows the wind-up.
inline float clip_phase(float phase,float sync,float first){
    if(!(sync>0&&sync<1)||!(first>=0&&first<=1))return phase;
    if(first>=sync)return 1-(1-phase)*(1-first)/(1-sync);
    return phase<sync?phase*first/sync:first+(phase-sync)*(1-first)/(1-sync);
}

enum : unsigned { fact_impact=3,fact_bomb_release=4,fact_standalone=5,fact_round_hit=16 };
struct Fact { unsigned kind=0; int unit_id=-1,tile_x=0,tile_y=0,code=0; long long ticks=0,frequency=0; };

// A unit as drawn this frame. Offsets are in tile units of the tile frame
// (x down-right, y down-left on screen, z up), relative to its tile centre.
struct UnitView {
    int unit_id=0,tile_x=0,tile_y=0,action=0;
    float x=0,y=0,yaw=0,phase=0,scale=1,offset_z=0;
    Armament const* arms=nullptr;
};

struct Live {
    std::string key,profile;
    long long start=0,frequency=1;
    int tile_x=0,tile_y=0;
    float x=0,y=0,z=0,yaw=0;
    float bias=0;                    // depth bias toward the camera, tile units
    double duration_ms=0;
};

// Civ III doubled tile coordinates -> tile-frame vector.
inline void tile_delta(int dx,int dy,float& u,float& v){u=(dx+dy)*.5f;v=(dy-dx)*.5f;}

class Director {
public:
    static constexpr std::size_t max_live=96;
    static constexpr float overshoot=.45f;     // a near miss lands beyond the winner
    static constexpr float impact_bias=.3f;    // impacts on a unit draw in front of it
    static constexpr float intercept_height=.6f; // standalone interceptions burst in the air (tile units)

    void push(Fact const& fact){
        std::lock_guard<std::mutex> lock(mutex);
        if(inbox.size()<256){inbox.push_back(fact);active.store(1,std::memory_order_release);}
    }
    // Fact-time ownership: once effects are loaded every native combat
    // effect is drawn here (unknown shooters use the default munition), so
    // no native 2D effect pixels remain.
    bool ready() const{std::lock_guard<std::mutex> lock(mutex);return enabled;}
    void enable(bool value){std::lock_guard<std::mutex> lock(mutex);enabled=value;}
    // Live effects or pending facts keep frames coming (any thread).
    bool animating() const{return active.load(std::memory_order_acquire)!=0;}
    std::vector<Live> const& live() const{return effects;}
    void clear(){std::lock_guard<std::mutex> lock(mutex);inbox.clear();effects.clear();memory.clear();recent.clear();
        active.store(0,std::memory_order_release);}

    // Render thread, once per frame. `water(tile_x,tile_y)` answers terrain.
    void update(Pack const& pack,std::vector<UnitView> const& units,long long now,long long frequency,
                std::function<bool(int,int)> const& water){
        if(frequency<=0)return;
        std::vector<Fact> facts;
        {std::lock_guard<std::mutex> lock(mutex);facts.swap(inbox);}
        if(now<last_now)effects.clear();
        last_now=now;
        auto age=[&](Live const& e){return double(now-e.start)*1000./double(e.frequency);};
        effects.erase(std::remove_if(effects.begin(),effects.end(),[&](Live const& e){return age(e)>=e.duration_ms;}),effects.end());
        for(auto const& unit:units)releases(pack,unit,now,frequency);
        for(auto const& fact:facts){
            if(fact.kind==fact_impact)impact(pack,fact,units,water);
            else if(fact.kind==fact_bomb_release)bomb(pack,fact,units,water);
            else if(fact.kind==fact_standalone)standalone(pack,fact);
            else if(fact.kind==fact_round_hit)combat_round(pack,fact,units,water);
        }
        for(auto it=memory.begin();it!=memory.end();)
            if(now-it->second.ticks>frequency*4)it=memory.erase(it);else ++it;
        for(auto it=recent.begin();it!=recent.end();)
            if(now-it->second>frequency)it=recent.erase(it);else ++it;
        std::lock_guard<std::mutex> lock(mutex);
        active.store(unsigned(effects.size()+inbox.size()),std::memory_order_release);
    }

private:
    struct Memory { int action=-1;float phase=0;long long ticks=0;unsigned serial=0; };
    mutable std::mutex mutex;
    std::vector<Fact> inbox;
    bool enabled=false;
    std::atomic<unsigned> active{0};
    std::unordered_map<int,Memory> memory;
    std::unordered_map<long long,long long> recent;   // tile -> last native impact ticks
    std::vector<Live> effects;
    long long last_now=0;

    static long long tile_key(int x,int y){return (static_cast<long long>(x)<<32)^static_cast<unsigned>(y);}
    static UnitView const* find(std::vector<UnitView> const& units,int id){
        for(auto const& unit:units){if(unit.unit_id==id)return &unit;}
        return nullptr;
    }
    bool spawn(Pack const& pack,Live live){
        auto const* profile=pack.find(live.profile);
        if(!profile||effects.size()>=max_live)return false;
        for(auto const& e:effects)if(e.key==live.key)return false;
        live.duration_ms=profile->duration_ms;effects.push_back(std::move(live));return true;
    }
    static std::string outcome_profile(Pack const& pack,std::string const& set,std::string const& outcome){
        std::string id="impact/"+set+"/"+outcome;
        return pack.find(id)?id:std::string();
    }

    // Muzzle and launch effects at the pack's release markers, as the drawn
    // attack clip crosses them; one per marker per pass of the clip.
    void releases(Pack const& pack,UnitView const& unit,long long now,long long frequency){
        if(!unit.arms||unit.arms->releases.empty())return;
        bool attacking=unit.action>=3&&unit.action<=5;
        auto& m=memory[unit.unit_id];
        if(!attacking){m.action=unit.action;m.phase=unit.phase;m.ticks=now;return;}
        float before=m.action==unit.action&&now-m.ticks<frequency/2?m.phase:std::max(0.f,unit.phase-.05f);
        if(m.action!=unit.action)++m.serial;
        bool wrapped=m.action==unit.action&&unit.phase+.25f<before;
        if(wrapped)++m.serial;
        m.action=unit.action;m.phase=unit.phase;m.ticks=now;
        float c=std::cos(unit.yaw),s=std::sin(unit.yaw);
        float fire=unit.yaw+unit.arms->bearing*3.14159265f/180.f;
        for(std::size_t i=0;i<unit.arms->releases.size();++i){
            auto const& r=unit.arms->releases[i];
            bool crossed=wrapped?(r.phase>before||r.phase<=unit.phase):(r.phase>before&&r.phase<=unit.phase);
            if(!crossed)continue;
            Live live;live.key="r/"+std::to_string(unit.unit_id)+"/"+std::to_string(m.serial)+"/"+std::to_string(i);
            live.profile=r.profile;live.start=now;live.frequency=frequency;
            live.tile_x=unit.tile_x;live.tile_y=unit.tile_y;
            live.x=unit.x+(r.x*c-r.y*s)*unit.scale;live.y=unit.y+(r.x*s+r.y*c)*unit.scale;
            live.z=std::max(0.f,(r.z+unit.offset_z)*unit.scale);live.yaw=fire;
            spawn(pack,std::move(live));
        }
    }

    // A bombard hit or miss: Civ III chose the outcome and the tile.
    void impact(Pack const& pack,Fact const& fact,std::vector<UnitView> const& units,std::function<bool(int,int)> const& water){
        auto const* source=find(units,fact.unit_id);
        bool known=source&&source->arms&&!source->arms->impact_set.empty();
        bool wet=water(fact.tile_x,fact.tile_y);
        std::string outcome=fact.code==7?"miss":fact.code==8?"water":wet?"ship":"hit";
        recent[tile_key(fact.tile_x,fact.tile_y)]=fact.ticks;
        Live live;live.key="i/"+std::to_string(fact.ticks)+"/"+std::to_string(fact.tile_x)+","+std::to_string(fact.tile_y);
        live.profile=outcome_profile(pack,known?source->arms->impact_set:"default",outcome);
        if(live.profile.empty())return;
        float u=0,v=0;
        if(source)tile_delta(fact.tile_x-source->tile_x,fact.tile_y-source->tile_y,u,v);
        live.yaw=(u!=0||v!=0)?std::atan2(v,u):source?source->yaw:0.f;
        live.start=fact.ticks;live.frequency=fact.frequency;live.tile_x=fact.tile_x;live.tile_y=fact.tile_y;
        live.bias=impact_bias;
        spawn(pack,std::move(live));
    }

    // The bomber's bomb FLC was revealed over the target: the stick's first
    // blast follows at the bomb sound's onset, marching along its heading.
    void bomb(Pack const& pack,Fact const& fact,std::vector<UnitView> const& units,std::function<bool(int,int)> const& water){
        auto const* bomber=find(units,fact.unit_id);
        bool known=bomber&&bomber->arms&&!bomber->arms->impact_set.empty();
        std::string set=known?bomber->arms->impact_set:"default";
        if(pack.find("impact/"+set+"_drop/hit"))set+="_drop";
        bool ship=false;
        for(auto const& unit:units){if(unit.tile_x==fact.tile_x&&unit.tile_y==fact.tile_y&&unit.arms&&unit.arms->sea)ship=true;}
        std::string outcome=ship?"ship":water(fact.tile_x,fact.tile_y)?"water":"hit";
        Live live;live.key="b/"+std::to_string(fact.ticks)+"/"+std::to_string(fact.unit_id);
        live.profile=outcome_profile(pack,set,outcome);
        if(live.profile.empty())return;
        double delay=known&&bomber->arms->bomb_blast_s>0?bomber->arms->bomb_blast_s:0;
        live.start=fact.ticks+static_cast<long long>(delay*double(fact.frequency));live.frequency=fact.frequency;
        // Heading: the bomber as drawn, else Civ III's flight direction (unit yaw basis).
        live.tile_x=fact.tile_x;live.tile_y=fact.tile_y;
        live.yaw=bomber?bomber->yaw:(225.f+45.f*float(((fact.code%8)+8)%8))*.01745329252f;live.bias=impact_bias;
        recent[tile_key(fact.tile_x,fact.tile_y)]=live.start;
        spawn(pack,std::move(live));
    }

    // A standalone native effect FLC (SAM shoot-down, SDI interception): an
    // interception burst in the air over its tile.
    void standalone(Pack const& pack,Fact const& fact){
        Live live;live.key="s/"+std::to_string(fact.ticks)+"/"+std::to_string(fact.tile_x)+","+std::to_string(fact.tile_y);
        live.profile=outcome_profile(pack,"intercept","air");
        if(live.profile.empty())return;
        live.start=fact.ticks;live.frequency=fact.frequency;live.tile_x=fact.tile_x;live.tile_y=fact.tile_y;
        live.z=intercept_height;live.bias=impact_bias;
        spawn(pack,std::move(live));
    }

    // HP lost in a combat round: the opponent's munition strikes the loser
    // and the loser's misses land beyond the winner. Bombard damage is the
    // native impact's, so a tile with a recent native impact is skipped.
    void combat_round(Pack const& pack,Fact const& fact,std::vector<UnitView> const& units,std::function<bool(int,int)> const& water){
        auto const* victim=find(units,fact.unit_id);
        if(!victim)return;
        auto hit=recent.find(tile_key(victim->tile_x,victim->tile_y));
        if(hit!=recent.end()&&std::llabs(fact.ticks-hit->second)<fact.frequency*3/2)return;
        UnitView const* opponent=nullptr;int best=1<<30;
        for(auto const& other:units){
            if(other.unit_id==victim->unit_id||other.action<3||other.action>5||!other.arms)continue;
            int d=std::abs(other.tile_x-victim->tile_x)+std::abs(other.tile_y-victim->tile_y);
            if(d<=4&&d<best){best=d;opponent=&other;}
        }
        if(!opponent)return;
        float u=0,v=0;tile_delta(victim->tile_x-opponent->tile_x,victim->tile_y-opponent->tile_y,u,v);
        float length=std::sqrt(u*u+v*v);
        if(length>0){u/=length;v/=length;}else{u=std::cos(opponent->yaw);v=std::sin(opponent->yaw);}
        std::string serial=std::to_string(fact.ticks)+"/"+std::to_string(victim->unit_id);
        bool victim_wet=victim->arms?victim->arms->sea:water(victim->tile_x,victim->tile_y);
        std::string outcome=victim->arms&&victim->arms->air?"air":victim_wet?"ship":"hit";
        Live strike;strike.key="h/"+serial;strike.profile=outcome_profile(pack,opponent->arms->impact_set,outcome);
        strike.start=fact.ticks;strike.frequency=fact.frequency;strike.tile_x=victim->tile_x;strike.tile_y=victim->tile_y;
        strike.x=victim->x;strike.y=victim->y;strike.yaw=std::atan2(v,u);strike.bias=impact_bias;
        if(victim->arms&&victim->arms->air)strike.z=std::max(0.f,victim->offset_z*victim->scale);
        if(!strike.profile.empty())spawn(pack,std::move(strike));
        if(!victim->arms||victim->arms->impact_set.empty()||(opponent->arms&&opponent->arms->air))return;
        bool winner_wet=opponent->arms->sea||water(opponent->tile_x,opponent->tile_y);
        Live miss;miss.key="m/"+serial;miss.profile=outcome_profile(pack,victim->arms->impact_set,winner_wet?"water":"miss");
        miss.start=fact.ticks;miss.frequency=fact.frequency;miss.tile_x=opponent->tile_x;miss.tile_y=opponent->tile_y;
        miss.x=opponent->x-u*overshoot;miss.y=opponent->y-v*overshoot;miss.yaw=std::atan2(-v,-u);
        if(!miss.profile.empty())spawn(pack,std::move(miss));
    }
};

// One director per renderer process: facts arrive through the unit-state
// input path; the scene pass updates and draws it.
inline Director& combat_director(){static Director director;return director;}

} // namespace c3x_renderer::effects
