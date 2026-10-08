"""The combat effect director turns Civ III facts and drawn units into effects.

`render_core/combat_effects.h` decides which pack profiles play, where and
from when; nothing names a unit. This host test drives it with synthetic
units and facts:

- release markers fire once per pass of the attack clip, at the socket point
  rotated with the drawn facing, oriented along the line of fire (bearing);
- a bombard hit/miss picks the source unit's munition and the native outcome
  (land hit, land miss, water miss, a hit on water strikes a ship);
- a revealed bomb FLC starts the stick at the bomb sound's onset along the
  bomber's heading, using the munition's drop variant;
- HP lost in a combat round strikes the loser with the opponent's munition
  and lands the loser's miss beyond the winner, but never duplicates a native
  bombard impact; an aircraft takes the air outcome;
- once effects load every fact is drawn (no native 2D effect remains): an
  unknown shooter uses the default munition, a standalone native effect FLC
  (SAM, SDI) an interception burst in the air; effects expire with their profile.
"""
import subprocess
import unittest

from Renderer.native.test_effect_sampler import run_host

PROGRAM = r'''
#include "Renderer/native/render_core/combat_effects.h"
#include <cstdio>
using namespace c3x_renderer::effects;
static int failures=0;
#define CHECK(x) do{if(!(x)){std::printf("FAIL line %d: %s\n",__LINE__,#x);++failures;}}while(0)
static bool near(float a,float b){return std::fabs(a-b)<1e-4f;}
static Live const* by_profile(Director const& d,char const* id){
    for(auto const& e:d.live())if(e.profile==id)return &e;return nullptr;}
static int count(Director const& d,char const* id){int n=0;for(auto const& e:d.live())n+=e.profile==id;return n;}
int main(){
    Pack pack;
    for(char const* id:{"combat/muzzle_cannon","impact/shell/hit","impact/shell/miss","impact/shell/water",
                        "impact/shell/ship","impact/shell/air","impact/bomb/hit","impact/bomb_drop/hit",
                        "impact/bomb_drop/water","impact/melee/miss","impact/melee/hit","impact/default/hit",
                        "impact/default_drop/hit","impact/intercept/air"}){
        Profile p;p.id=id;p.duration_ms=1000;pack.profiles.push_back(p);}
    auto water=[](int x,int y){return x>=20&&y>=0;};
    long long f=1000;
    Armament gun;gun.impact_set="shell";gun.bearing=90;
    Release r0;r0.phase=.25f;r0.profile="combat/muzzle_cannon";r0.x=1;r0.y=0;r0.z=.2f;
    Release r1=r0;r1.phase=.6f;gun.releases={r0,r1};
    Armament sword;sword.impact_set="melee";
    Armament bomber;bomber.impact_set="bomb";bomber.bomb_blast_s=.638f;bomber.air=true;
    Armament ship;ship.impact_set="shell";ship.sea=true;

    // Releases: once per marker per clip pass, socket rotated with facing.
    Director d;
    UnitView g;g.unit_id=7;g.tile_x=10;g.tile_y=10;g.action=3;g.yaw=1.5707963f;g.scale=.5f;g.arms=&gun;
    auto frame=[&](float phase,long long now,std::vector<UnitView> extra={}){
        g.phase=phase;std::vector<UnitView> units{g};units.insert(units.end(),extra.begin(),extra.end());
        d.update(pack,units,now,f,water);};
    frame(.1f,100);CHECK(d.live().empty());
    frame(.3f,130);CHECK(d.live().size()==1);
    auto const* m=by_profile(d,"combat/muzzle_cannon");
    CHECK(m&&near(m->x,0)&&near(m->y,.5f)&&near(m->z,.1f)&&near(m->yaw,3.1415926f)&&m->start==130);
    frame(.5f,160);CHECK(d.live().size()==1);
    frame(.65f,190);CHECK(d.live().size()==2);
    frame(.05f,220);frame(.3f,250);CHECK(count(d,"combat/muzzle_cannon")==3);   // the loop fires again
    g.action=1;frame(.3f,280);CHECK(count(d,"combat/muzzle_cannon")==3);        // idle never fires

    // Impacts: the source's munition and Civ III's outcome; ownership.
    Director i;UnitView src=g;src.action=1;std::vector<UnitView> units{src};
    i.update(pack,units,1000,f,water);
    CHECK(!i.ready());i.enable(true);   // nothing is owned before the effects load
    CHECK(i.ready());
    i.push({fact_impact,7,12,14,4,1000,f});i.push({fact_impact,7,12,16,7,1001,f});
    i.push({fact_impact,7,22,10,8,1002,f});i.push({fact_impact,7,22,12,3,1003,f});
    i.push({fact_impact,8,18,14,4,1004,f});   // unknown shooter: default munition
    CHECK(i.animating());i.update(pack,units,1005,f,water);
    CHECK(i.live().size()==5&&i.animating());
    CHECK(by_profile(i,"impact/default/hit")&&by_profile(i,"impact/default/hit")->tile_x==18);
    CHECK(by_profile(i,"impact/shell/hit")&&by_profile(i,"impact/shell/miss")&&
          by_profile(i,"impact/shell/water")&&by_profile(i,"impact/shell/ship"));
    auto const* h=by_profile(i,"impact/shell/hit");
    CHECK(h&&h->tile_x==12&&h->tile_y==14&&near(h->yaw,std::atan2(1.f,3.f))&&h->bias>0);  // doubled (2,4) -> tile (3,1)

    // Bomb release: drop variant, delayed to the bomb sound's onset, heading.
    Director b;UnitView plane;plane.unit_id=9;plane.tile_x=4;plane.tile_y=4;plane.action=2;plane.yaw=.7f;plane.arms=&bomber;
    std::vector<UnitView> sky{plane};b.update(pack,sky,0,f,water);
    b.push({fact_bomb_release,9,6,6,5,2000,f});b.update(pack,sky,2001,f,water);
    auto const* stick=by_profile(b,"impact/bomb_drop/hit");
    CHECK(stick&&stick->start==2638&&near(stick->yaw,.7f));

    // Unknown bomber: default stick along Civ III's flight direction; a
    // standalone native effect bursts in the air over its tile.
    b.push({fact_bomb_release,99,8,8,5,3000,f});b.push({fact_standalone,-1,9,9,0,3000,f});b.update(pack,sky,3001,f,water);
    auto const* other=by_profile(b,"impact/default_drop/hit");auto const* burst=by_profile(b,"impact/intercept/air");
    CHECK(other&&other->start==3000&&near(other->yaw,(225.f+225.f)*.01745329252f));
    CHECK(burst&&burst->tile_x==9&&burst->z>0);

    // Combat round: loser struck by the opponent, near miss beyond the winner.
    Director c;UnitView a=g;a.action=3;a.phase=.1f;UnitView v;v.unit_id=11;v.tile_x=11;v.tile_y=11;v.action=3;v.arms=&sword;
    std::vector<UnitView> fight{a,v};c.update(pack,fight,0,f,water);
    c.push({fact_round_hit,11,11,11,0,3000,f});c.update(pack,fight,3001,f,water);
    auto const* strike=by_profile(c,"impact/shell/hit");auto const* miss=by_profile(c,"impact/melee/miss");
    CHECK(strike&&strike->tile_x==11&&strike->tile_y==11);
    CHECK(miss&&miss->tile_x==10&&miss->tile_y==10&&miss->x<0);
    // A native bombard impact on the tile owns that damage.
    c.push({fact_impact,7,11,11,3,5000,f});c.push({fact_round_hit,11,11,11,0,5100,f});c.update(pack,fight,5101,f,water);
    CHECK(count(c,"impact/shell/hit")==1);
    // An aircraft victim takes the air outcome; no opponent, no effect.
    Director air;UnitView flyer=v;flyer.arms=&bomber;flyer.offset_z=.4f;flyer.scale=.75f;
    std::vector<UnitView> dogfight{a,flyer};air.update(pack,dogfight,0,f,water);
    air.push({fact_round_hit,11,11,11,0,10,f});air.update(pack,dogfight,11,f,water);
    CHECK(by_profile(air,"impact/shell/air")&&near(by_profile(air,"impact/shell/air")->z,.3f));
    Director alone;std::vector<UnitView> solo{v};alone.update(pack,solo,0,f,water);
    alone.push({fact_round_hit,11,11,11,0,10,f});alone.update(pack,solo,11,f,water);CHECK(alone.live().empty());

    // Expiry and clock reset.
    i.update(pack,units,2100,f,water);CHECK(i.live().empty()&&!i.animating());
    c.update(pack,fight,10,f,water);CHECK(c.live().empty());
    std::printf(failures?"FAILED %d\n":"PASS combat effect director\n",failures);
    return failures!=0;
}
'''


class CombatEffectDirectorTests(unittest.TestCase):
    def test_director_from_facts_and_drawn_units(self):
        try:
            output = run_host(PROGRAM, "")
        except subprocess.CalledProcessError as failure:
            self.fail(failure.stdout)
        self.assertIn("PASS combat effect director", output)


if __name__ == "__main__":
    unittest.main()
