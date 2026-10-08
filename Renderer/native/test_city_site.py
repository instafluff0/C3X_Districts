"""Version-five city packs: site-optional bodies and ground plates yield to rivers,
water, mountains and steep relief; earlier packs keep every authored body."""
import unittest

from Renderer.native.native_cpp_test import run_cpp


class CitySiteTests(unittest.TestCase):
    def test_marked_bodies_and_plate_yield_to_the_site(self):
        run_cpp(r'''
#include "Renderer/native/city_fidelity/compiler.h"
#include "Renderer/lab/shared/natural/ground.h"
#include <cassert>
#include <cstring>
using namespace c3x_renderer;
int main(){
 city_fidelity::Library library;library.materials.resize(2);
 library.materials[0].channels=4;library.materials[1].channels=4;library.materials[1].ground=1;
 city_fidelity::Model model;model.low[2]=0;model.high[2]=.3f;
 city_fidelity::Part part;city_fidelity::Vertex vertex{};
 vertex.normal[2]=1;vertex.tangent[0]=1;vertex.bitangent[1]=1;
 part.vertices={vertex,vertex,vertex};part.vertices[1].position[0]=.05f;
 part.vertices[2].position[1]=.05f;part.indices={0,1,2};model.parts.push_back(part);
 library.models.push_back(model);
 city_fidelity::Composition city;city.authority="lab-readable-test";
 city.clearance[0]=.04f;city.clearance[1]=24;city.clearance[3]=10;
 city.anchor_layout=1;city.owns_walls=1;
 // Body 0 is the unmarked core at the tile centre; body 1 is a marked
 // outskirt in the next column (world x 11.04-11.16); each carries a light.
 city_fidelity::Light light{};light.range=.3f;light.intensity=1;
 for(float x:{0.f,.6f}){
  city_fidelity::Instance body;body.scale=1;body.offset[0]=x;
  body.bounds[0]=body.bounds[1]=-.06f;body.bounds[2]=body.bounds[3]=.06f;
  body.lights={light};city.instances.push_back(body);
 }
 city.instances[1].flags=city_fidelity::instance_site_optional;city.site_aware=true;
 city.paving.material=1;city.paving.period[0]=city.paving.period[1]=.45f;
 city.paving.vertices={{-.1f,0,1},{.6f,0,1},{.0f,.1f,1}};city.paving.indices={0,1,2};
 library.compositions.push_back(city);
 struct Land{int base=2,real=2;};struct Shore{double distance=100;};
 auto flat=[](float,float){return 2.5f;};
 fidelity::GroundProjection projection{10,12,64,32,1,480};
 // The river runs along x=11.1 under the outskirt; the core is 0.6 away.
 auto river=[](float x,float){return double(std::abs(x-11.1f)*64.f);};
 auto dry=[](float,float){return 1000.;};
 auto land=[](int,int){return Land{};};auto shore=[](float,float){return Shore{};};
 auto const& c=library.compositions[0];
 auto count_bodies=[](city_fidelity::Surfaces const& s){
  unsigned n=0;for(auto const& chunk:s.chunks)n+=!chunk.terrain_conforming;return n;};
 auto plate_at=[](city_fidelity::Surfaces const& s,unsigned vertex){
  for(auto const& chunk:s.chunks)if(chunk.terrain_conforming)return chunk.vertices[vertex].base_terrain-62.f;
  return -1.f;};
 // Default compilation (earlier packs and callers without a site) keeps all.
 city_fidelity::Surfaces every;
 assert(city_fidelity::compile(library,c,10,12,flat,projection,every));
 assert(count_bodies(every)==2 && every.chunks.front().lighting->lights.size()==2);
 // A river under the marked body removes it, its light and its blocker.
 city_fidelity::Surfaces wet;
 assert(city_fidelity::compile(library,c,10,12,flat,projection,wet,city_fidelity::ContinueCompilation{},true,
   city_fidelity::site_filter(c,10,12,land,shore,river,flat)));
 assert(count_bodies(wet)==1);
 auto const& lit=*wet.chunks.front().lighting;
 assert(lit.lights.size()==1 && lit.blockers.size()==1 && lit.lights[0].owner==0.f);
 // The plate fades over the channel but stays under the core.
 assert(plate_at(wet,0)>.99f && plate_at(wet,1)<.01f);
 // A dry site keeps the outskirt and the full plate.
 city_fidelity::Surfaces dry_site;
 assert(city_fidelity::compile(library,c,10,12,flat,projection,dry_site,city_fidelity::ContinueCompilation{},true,
   city_fidelity::site_filter(c,10,12,land,shore,dry,flat)));
 assert(count_bodies(dry_site)==2 && plate_at(dry_site,1)>.99f);
 assert(dry_site.chunks.front().lighting->lights[1].owner==1.f);
 // Water and mountain tiles under the marked body remove it; the core stays.
 for(int kind:{0,1}){
  auto rough=[kind](int column,int){Land l;if(column>=11){if(kind==0)l.base=12;else l.real=6;}return l;};
  assert(!city_fidelity::site_keeps(c,c.instances[1],10,12,rough,shore,dry,flat));
  assert(city_fidelity::site_keeps(c,c.instances[0],10,12,rough,shore,dry,flat));
 }
 // Steep ground under the marked body removes it.
 auto steep=[](float x,float){return x*400.f;};
 assert(!city_fidelity::site_keeps(c,c.instances[1],10,12,land,shore,dry,steep));
 // The shore clearance applies to marked bodies only.
 auto beach=[](float,float){Shore s;s.distance=.01;return s;};
 assert(!city_fidelity::site_keeps(c,c.instances[1],10,12,land,beach,dry,flat));
 assert(city_fidelity::site_keeps(c,c.instances[0],10,12,land,beach,dry,flat));
 // Without the version-five flag nothing yields and the plate is complete.
 library.compositions[0].instances[1].flags=0;library.compositions[0].site_aware=false;
 city_fidelity::Surfaces legacy;
 assert(city_fidelity::compile(library,c,10,12,flat,projection,legacy,city_fidelity::ContinueCompilation{},true,
   city_fidelity::site_filter(c,10,12,land,shore,river,flat)));
 assert(count_bodies(legacy)==2 && plate_at(legacy,1)>.99f);
 std::puts("PASS city site filter: bodies, lights, blockers and plate yield; legacy unchanged");
}
''')

    def test_attached_effects_become_upright_quads_with_their_bodies(self):
        run_cpp(r'''
#include "Renderer/native/city_fidelity/compiler.h"
#include "Renderer/lab/shared/natural/ground.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer;
int main(){
 city_fidelity::Library library;library.materials.resize(2);
 library.materials[0].channels=4;library.materials[1].channels=4;library.materials[1].ground=1;
 city_fidelity::Model model;model.high[2]=.3f;city_fidelity::Part part;city_fidelity::Vertex vertex{};
 vertex.normal[2]=1;part.vertices={vertex,vertex,vertex};part.vertices[1].position[0]=.05f;
 part.vertices[2].position[1]=.05f;part.indices={0,1,2};model.parts.push_back(part);library.models.push_back(model);
 city_fidelity::Composition city;city.clearance[1]=24;city.clearance[3]=10;
 city_fidelity::Instance body;body.scale=2;body.bounds[0]=body.bounds[1]=-.06f;body.bounds[2]=body.bounds[3]=.06f;
 body.effects={{{.01f,0,.1f},float(city_fidelity::effect_flame),.05f,.09f,3,1},
               {{0,.01f,.2f},float(city_fidelity::effect_smoke),.16f,.6f,5,1}};
 city.instances.push_back(body);body.offset[0]=.6f;body.flags=city_fidelity::instance_site_optional;
 body.effects={{{0,0,.1f},float(city_fidelity::effect_night_light),.07f,.07f,9,.8f}};
 city.instances.push_back(body);city.site_aware=true;library.compositions.push_back(city);
 auto flat=[](float,float){return 2.5f;};fidelity::GroundProjection projection{10,12,64,32,1,480};
 auto effect_chunk=[&](city_fidelity::Surfaces const& s)->city_fidelity::Chunk const*{
  for(auto const& c:s.chunks)if(c.material==1 && !c.vertices.empty() && c.vertices[0].base_terrain>=89.5f)return &c;return nullptr;};
 // No effect material: effects are ignored, the bodies are unchanged.
 city_fidelity::Surfaces plain;assert(city_fidelity::compile(library,library.compositions[0],10,12,flat,projection,plain));
 assert(!effect_chunk(plain));
 library.effect_material=1;
 city_fidelity::Surfaces all;assert(city_fidelity::compile(library,library.compositions[0],10,12,flat,projection,all));
 auto const* e=effect_chunk(all);assert(e && e->vertices.size()==18 && e->terrain_conforming && e->effect);
 // Only effect quads leave the static layer for the per-frame pass.
 for(auto const& c:all.chunks)assert(c.effect==(&c==e));
 assert(all.chunks.back().vertices.size()==18); // effects draw after their bodies
 for(unsigned q=0;q<3;q++){
  auto const* v=&e->vertices[q*6];float kind=v[0].base_terrain-90;
  assert(kind==0 || kind==1 || kind==2);
  // Corners: (-1,0),(1,0),(1,1) then (-1,0),(1,1),(-1,1). Across-screen
  // corners share a depth row; the upper edge rises in world height only.
  assert(v[0].u==-1 && v[1].u==1 && v[2].v==1);
  assert(std::abs((v[1].world_x-v[0].world_x)-(v[1].world_y-v[0].world_y))<1e-5f);
  assert(v[2].world_z>v[1].world_z && v[2].world_x==v[1].world_x && v[2].world_y==v[1].world_y);
  assert(v[0].macro_v>0 && v[0].relief_owner_u>0);
 }
 // A marked body that yields to a river takes its night light with it.
 auto river=[](float x,float){return double(std::abs(x-11.1f)*64.f);};
 struct Land{int base=2,real=2;};struct Shore{double distance=100;};
 auto land=[](int,int){return Land{};};auto shore=[](float,float){return Shore{};};
 city_fidelity::Surfaces wet;assert(city_fidelity::compile(library,library.compositions[0],10,12,flat,projection,wet,
   city_fidelity::ContinueCompilation{},false,city_fidelity::site_filter(library.compositions[0],10,12,land,shore,river,flat)));
 auto const* w=effect_chunk(wet);assert(w && w->vertices.size()==12);
 for(auto const& v:w->vertices)assert(v.base_terrain<91.5f);
 // An active volcano's plume: one quad rising from just above its crater,
 // kind 93, thickened (strength 1.6) by an eruption; none without the material.
 city_fidelity::Surfaces plume;
 assert(city_fidelity::volcano_plume(library,10,12,40.f,true,projection,false,plume) && plume.chunks.size()==1);
 auto const* pv=effect_chunk(plume);assert(pv && pv->vertices.size()==6 && pv->effect && pv->terrain_conforming);
 for(auto const& v:pv->vertices)assert(v.base_terrain==93 && v.macro_v==1.6f && v.macro_u>=0 && v.macro_u<64);
 assert(std::abs(pv->vertices[0].world_z-43.f/112)<1e-5f && pv->vertices[2].world_z>pv->vertices[1].world_z+1);
 city_fidelity::Surfaces smoldering;assert(city_fidelity::volcano_plume(library,10,12,40.f,false,projection,false,smoldering));
 assert(effect_chunk(smoldering)->vertices[0].macro_v==1);
 library.effect_material=7;city_fidelity::Surfaces none;
 assert(!city_fidelity::volcano_plume(library,10,12,40.f,true,projection,false,none) && none.chunks.empty());
 std::puts("PASS city effects: upright screen-aligned quads after bodies, site-aware, gated by material");
}
''')

    def test_version_five_decode_reads_look_and_flags(self):
        run_cpp(r'''
#include "Renderer/native/city_fidelity/runtime.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::city_fidelity;
std::vector<std::uint8_t> pack(bool five,float gain){
 std::vector<std::uint8_t> b;auto u=[&](unsigned v){for(int i=0;i<4;i++)b.push_back(std::uint8_t(v>>(8*i)));};
 auto f=[&](float v){unsigned x;std::memcpy(&x,&v,4);u(x);};
 auto s=[&](char const* t){u(unsigned(std::strlen(t)));for(char const* p=t;*p;++p)b.push_back(std::uint8_t(*p));};
 for(char ch:std::string(five?"C3XCITY5":"C3XCITY4"))b.push_back(std::uint8_t(ch));
 u(1);u(1);u(1);
 if(five){f(gain);f(.3f);f(.1f);for(int i=0;i<5;i++)f(0);u(0);}
 u(0);u(4);u(0);s("Renderer/packs/x/a.dds");for(int i=0;i<6;i++)s("");
 u(1);for(float v:{0.f,0.f,0.f,1.f,1.f,1.f})f(v);u(3);for(float v:{0.f,0.f,1.f,0.f,0.f,1.f})f(v);
 u(0);u(3);u(3);for(int k=0;k<3;k++)for(int i=0;i<18;i++)f(0);u(0);u(1);u(2);
 for(unsigned v:{0u,0u,0u,0u,0u,0u,0u,0u,1u})u(v);s("t");for(int i=0;i<4;i++)f(0);
 u(1);u(0);u(0);for(float v:{1.f,0.f,0.f,0.f,-.1f,-.1f,.1f,.1f})f(v);if(five)u(instance_site_optional);u(0);
 if(five){u(1);for(float v:{.01f,.02f,.03f,float(effect_smoke),.1f,.3f,7.f,1.f})f(v);}
 u(0);u(0);
 return b;
}
int main(){
 Library five;assert(five.decode(pack(true,.5f)));
 assert(five.look[0]==.5f && five.look[1]==.3f && five.look[2]==.1f);
 assert(five.compositions[0].instances[0].flags==instance_site_optional && five.compositions[0].site_aware);
 assert(five.effect_material==0 && five.compositions[0].instances[0].effects.size()==1);
 assert(five.compositions[0].instances[0].effects[0].kind==float(effect_smoke));
 Library four;assert(four.decode(pack(false,0)));
 assert(four.look[0]==0 && four.compositions[0].instances[0].flags==0 && !four.compositions[0].site_aware);
 Library bad;assert(!bad.decode(pack(true,9.f)));
 std::puts("PASS version-five city decode: look and flags; version four unchanged");
}
''')

    def test_effects_redraw_in_their_own_layer_not_the_water_scene(self):
        from pathlib import Path
        source = (Path(__file__).parent.parent / "sandbox/fresh_pipeline.h").read_text()
        self.assertTrue(effects_routed_first(source))
        # Static preview frames skip the effect layer.
        self.assertIn("if(!static_preview && !effect_visible[geometry_city].empty()", source)
        # The reflection contributors skip attached effects.
        self.assertIn("record.content().animation_texture || record.content().city_effect ||", source)
        # The earlier routing sent a water-dependent smoke quad to the water
        # scene, which drew it differently while the camera moved (in-game
        # purple and white flashes).
        old = source.replace(
            "auto& output=record.content().city_effect?effect_visible:\n"
            "                    renderer.water_scene_active && record.water_dependent?water_visible:static_visible;",
            "auto& output=renderer.water_scene_active && record.water_dependent?\n"
            "                    water_visible:record.content().city_effect?effect_visible:static_visible;")
        self.assertFalse(effects_routed_first(old))


def effects_routed_first(source):
    """Selection tests city_effect before water dependency."""
    start = source.find("auto& output=")
    end = source.find(";", start)
    statement = source[start:end]
    effect, water = statement.find("city_effect"), statement.find("water_dependent")
    return start >= 0 and 0 <= effect < water


if __name__ == "__main__":
    unittest.main()
