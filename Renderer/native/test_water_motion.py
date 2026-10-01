"""Visual flow follows connectivity, wraps and remote outlet invalidation."""
from pathlib import Path
import unittest
from Renderer.native.source_fidelity.prepare import function
from Renderer.native.native_cpp_test import run_cpp


class WaterMotionTests(unittest.TestCase):
    def test_explored_only_view_advances_copied_clock_and_keeps_fog(self):
        source=Path('Renderer/native/c3x_renderer.cpp').read_text()
        eligibility=function(source,'frame_has_resource_animation')
        clock=function(source,'resource_clock')
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include <vector>
#include "Renderer/native/render_core/water_material_frame.h"
#include "Renderer/native/render_core/dynamic_scene_input.h"
#include "Renderer/native/render_core/visibility_coverage.h"
struct Renderer {
 bool water_scene_active=false,wave_ready=true,visibility_pass=true;
 std::vector<int> resource_animations;
 int resource_animation_for(c3x_renderer_tile_v1 const&)const{return -1;}
''' + eligibility + clock + r'''
};
int main(){
 using namespace c3x_renderer::render_core;
 c3x_renderer_tile_v1 tile{};tile.terrain_type=tile.real_terrain_type=12;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 c3x_renderer_frame_v1 frame{};frame.api_version=C3X_RENDERER_API_VERSION;frame.struct_size=sizeof(frame);
 frame.target_width=frame.tile_width=128;frame.target_height=frame.tile_height=64;
 frame.tiles=&tile;frame.tile_count=1;frame.presentation_frequency=1000;frame.presentation_time_ticks=1000;
 DynamicSceneInputs owner;auto captured=owner.capture(frame,{});assert(captured);
 Renderer renderer;assert(renderer.frame_has_resource_animation(frame));
 auto first=water_material_frame(frame);auto initial_clock=Renderer::resource_clock(frame,60);
 c3x_renderer_frame_v1 later{};assert(captured->sample(3000,1000,1000,later));
 assert(later.presentation_time_ticks==3000 && !(later.tiles[0].tile_flags&C3X_RENDERER_TILE_VISIBLE));
 assert(Renderer::resource_clock(later,60)>initial_clock && renderer.frame_has_resource_animation(later));
 auto advanced=water_material_frame(later);assert(advanced.time==3 && advanced.time>first.time);
 assert(advanced.drift[0]!=first.drift[0]);
 // Copied visibility stays fogged while the cosmetic material sample advances.
 VisibilityCoverage coverage;assert(coverage.capture(later));assert(coverage.state(0,0)==1);
 std::vector<unsigned> pixels(128*64,0xaabb8844),out;coverage.apply(pixels.data(),out);
 assert(out[32*128+64]==0xaa745b39);
 // Sight loss/reveal and hidden on-screen content do not consult a prior clock.
 for(unsigned flags:std::vector<unsigned>{C3X_RENDERER_TILE_VISIBILITY_BITS,
      C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED,
      C3X_RENDERER_TILE_VISIBILITY_KNOWN,C3X_RENDERER_TILE_VISIBILITY_BITS}){
  tile.tile_flags=C3X_RENDERER_TILE_RENDER|flags;
  assert(renderer.frame_has_resource_animation(frame)==bool(flags&C3X_RENDERER_TILE_EXPLORED));
  assert(water_material_frame(frame).time==1);
 }
 tile.tile_flags=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_VISIBILITY_BITS;
 assert(!renderer.frame_has_resource_animation(frame));
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_BITS;
 tile.anchor_x=128;assert(!renderer.frame_has_resource_animation(frame));
}
''')

    def test_native_and_fresh_water_bind_current_fogged_material(self):
        native=Path('Renderer/native/c3x_renderer.cpp').read_text()
        fresh=Path('Renderer/sandbox/fresh_pipeline.h').read_text()
        blocks=[]
        for source,start,end in [(native,'if(environment_profile && (layer==geometry_water || layer==geometry_river)){','if(layer==geometry_wave)'),
                                 (fresh,'if(renderer.environment_profile && (layer==geometry_water || layer==geometry_river)){','if(layer==geometry_wave)')]:
            block=source.split(start,1)[1].split(end,1)[0]
            blocks.append(start+block)
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include <cstring>
#include "Renderer/native/render_core/water_material_frame.h"
using c3x_renderer::render_core::WaterMaterialFrame;
struct Context {WaterMaterialFrame copied;unsigned updates=0;
 void UpdateSubresource(int,int,void*,void const* input,int,int){copied=*static_cast<WaterMaterialFrame const*>(input);++updates;}
 void PSSetConstantBuffers(int,int,int*){}
} context_value,*context=&context_value;
struct Work {void upload_buffer(int){}} work;
struct Renderer {bool environment_profile=true,water_scene_active=true;WaterMaterialFrame water_material;int water_frame=1;} renderer;
struct Mesh {float visual_time=-1;} mesh;
struct Chunk {bool explored=true;bool water_visible()const{return explored;}Mesh const& content()const{return mesh;}} chunk;
enum {geometry_water,geometry_river};
bool environment_profile=true,water_scene_active=true;WaterMaterialFrame water_material;int water_frame=1;
void native_bind(int layer){
''' + blocks[0] + r'''
}
void fresh_bind(int layer){
''' + blocks[1] + r'''
}
int main(){
 c3x_renderer_frame_v1 frame{};frame.presentation_time_ticks=7000;frame.presentation_frequency=1000;
 renderer.water_material=water_material=c3x_renderer::render_core::water_material_frame(frame);
 for(int layer:{geometry_water,geometry_river}){
  chunk.explored=true;native_bind(layer);assert(!std::memcmp(&context->copied,&water_material,sizeof(water_material)));
  fresh_bind(layer);assert(!std::memcmp(&context->copied,&water_material,sizeof(water_material)));
  chunk.explored=false;native_bind(layer);assert(context->copied.time==0 && context->copied.drift[0]==0);
  fresh_bind(layer);assert(context->copied.time==0 && context->copied.drift[0]==0);
  chunk.explored=true;renderer.water_scene_active=water_scene_active=false;
  native_bind(layer);assert(context->copied.time==0);fresh_bind(layer);assert(context->copied.time==0);
  renderer.water_scene_active=water_scene_active=true;
 }
 assert(context->updates==12);
}
''')

    def test_explored_scene_consumes_current_shared_environment(self):
        source=Path('Renderer/sandbox/fresh_pipeline.h').read_text()
        prefix='void update_environment(c3x_renderer_frame_v1 const& frame) {'+source.split(
            'void update_environment(c3x_renderer_frame_v1 const& frame) {',1)[1].split(
            'visual_sun_intensity=',1)[0]
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <cstring>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/environment_runtime.h"
bool cycle_enabled=false;
unsigned GetEnvironmentVariableA(char const*,char* output,unsigned){if(cycle_enabled){output[0]='1';output[1]=0;return 1;}return 0;}
namespace c3x_renderer {namespace render_core {enum {raster_environment};}}
struct Pipeline {
 float visual_hour=0,previous_hour=-1;int previous_season=-1;
 unsigned lighting_revision=0;bool reflection_valid=true,reflected_terrain_material_valid=true;
 struct Rasters {unsigned invalidations=0;void invalidate_all(int){++invalidations;}} static_rasters;
 c3x_renderer::EnvironmentState copied{};
''' + prefix + r'''
 copied=environment;
 }
};
int main(){
 c3x_renderer_tile_v1 tile{};tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;frame.hour=12;frame.season=0;
 frame.presentation_frequency=1000;frame.presentation_time_ticks=1000;
 Pipeline pipeline;pipeline.update_environment(frame);auto noon=pipeline.copied;
 assert(noon.sun_intensity>0 && pipeline.lighting_revision==1);
 frame.hour=0;pipeline.update_environment(frame);auto night=pipeline.copied;
 assert(night.moon_intensity>0 && night.sun_intensity<noon.sun_intensity);
 assert(pipeline.static_rasters.invalidations==2 && pipeline.lighting_revision==2);
 assert(!pipeline.reflection_valid && !pipeline.reflected_terrain_material_valid);
 // Revealing the same permitted scene does not change or duplicate lighting.
 tile.tile_flags|=C3X_RENDERER_TILE_VISIBLE;pipeline.update_environment(frame);
 assert(pipeline.lighting_revision==2 && !std::memcmp(&night,&pipeline.copied,sizeof(night)));
 tile.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;
 frame.season=3;pipeline.update_environment(frame);assert(pipeline.lighting_revision==3);
 // The existing diagnostic environment cycle also uses current presentation
 // time under explored fog; it never promotes gameplay visibility or hour.
 cycle_enabled=true;pipeline.update_environment(frame);auto first=pipeline.copied;
 frame.presentation_time_ticks=15000;pipeline.update_environment(frame);
 assert(pipeline.visual_hour==24 && pipeline.lighting_revision==5);
 assert(std::memcmp(&first,&pipeline.copied,sizeof(first)) && frame.hour==0);
 assert(!(tile.tile_flags&C3X_RENDERER_TILE_VISIBLE));
}
''',sources=('Renderer/native/environment_runtime.cpp',))

    def test_explored_water_clock_does_not_admit_hidden_unit_contributors(self):
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include "Renderer/native/render_core/water_material_frame.h"
#include "Renderer/native/render_core/unit_instances.h"
using namespace c3x_renderer::render_core;
struct Clip {std::string name="idle";bool ambient=true,loop=true;double duration=1;unsigned frames=16;};
struct Unit {std::vector<std::string> keys={"warrior"};std::vector<Clip> actions={Clip{}};};
int main(){
 UnitInstances units;std::vector<Unit> catalog(1);
 c3x_renderer_unit_state_v1 state{};state.struct_size=sizeof(state);state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;
 state.unit_id=7;state.tile_x=state.tile_y=4;state.action=1;state.max_hp=3;state.visible=1;state.presentation_frequency=1000;
 assert(units.state(state));
 c3x_renderer_unit_v1 body{};body.struct_size=sizeof(body);body.unit_id=7;body.action=1;body.frame_count=16;
 body.sprite_width=body.sprite_height=191;body.projection_scale_milli=1000;body.presentation_frequency=1000;
 std::strcpy(body.unit_key,"warrior");UnitInstances::Selection selection;
 assert(units.capture(body,C3X_RENDERER_UNIT_STATE_CAPTURED,catalog,[](int){return "idle";},selection));
 c3x_renderer_tile_v1 tile{};tile.tile_x=tile.tile_y=4;tile.terrain_type=tile.real_terrain_type=12;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_BITS;
 c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;
 frame.target_width=frame.tile_width=128;frame.target_height=frame.tile_height=64;frame.presentation_frequency=1000;
 assert(units.scene_poses(frame,1000,1000,catalog).size()==1);
 tile.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;
 assert(water_scene_tile(tile,frame,true));
 // Main bodies, reflections and shadows consume this same admitted pose list.
 assert(units.scene_poses(frame,7000,1000,catalog).empty());
 tile.tile_flags|=C3X_RENDERER_TILE_VISIBLE;
 assert(units.scene_poses(frame,7000,1000,catalog).size()==1);
}
''')

    def test_bounded_motion_and_authoritative_camera_basis(self):
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include "Renderer/native/render_core/water_material_frame.h"
int main(){
 using namespace c3x_renderer::render_core;
 static_assert(sizeof(WaterMaterialFrame)==32,"two constant-buffer vectors");
 c3x_renderer_tile_v1 tile={};tile.tile_x=10;tile.tile_y=12;
 tile.anchor_x=100;tile.anchor_y=200;tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 tile.terrain_type=2;tile.real_terrain_type=11;
 c3x_renderer_frame_v1 frame={};frame.target_width=640;frame.target_height=480;
 frame.tile_width=128;frame.tile_height=64;frame.tiles=&tile;frame.tile_count=1;
 assert(water_scene_tile(tile,frame,false)); // Civ III coast: land-like m49, water m50.
 assert(!water_scene_tile(tile,frame,true)); // Unknown water cannot start the clock.
 tile.tile_flags|=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 assert(water_scene_tile(tile,frame,true)); // Explored fog retains current cosmetic motion.
 tile.tile_flags|=C3X_RENDERER_TILE_VISIBLE;
 assert(water_scene_tile(tile,frame,true));
 tile.real_terrain_type=2;assert(!water_scene_tile(tile,frame,true));
 tile.river_code=170;assert(water_scene_tile(tile,frame,true));
 tile.real_terrain_type=11;tile.river_code=0;
 auto flags=tile.tile_flags;
 tile.tile_flags=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_EXPLORED;
 assert(!water_scene_tile(tile,frame,false));tile.tile_flags=flags;
 auto x=tile.anchor_x,y=tile.anchor_y;
 for(int offscreen:{-128,640}){tile.anchor_x=offscreen;assert(!water_scene_tile(tile,frame,true));}
 tile.anchor_x=x;
 for(int offscreen:{-64,480}){tile.anchor_y=offscreen;assert(!water_scene_tile(tile,frame,true));}
 tile.anchor_y=y;
 frame.presentation_frequency=1000;frame.world_width_tiles=100;frame.world_height_tiles=80;
 frame.world_wrap_x=frame.world_wrap_y=1;
 auto initial=water_material_frame(frame);
 for(float v:initial.drift)assert(v==0);
 assert(initial.camera[2]==100 && initial.camera[3]==80);
 // An equivalent anchor must not move the optical eye or form a tile seam.
 tile.tile_x+=2;tile.anchor_x+=128;auto equivalent=water_material_frame(frame);
 for(unsigned i=0;i<4;++i)assert(initial.camera[i]==equivalent.camera[i]);
 tile.anchor_x-=64;auto pan=water_material_frame(frame);
 assert(pan.camera[0]==initial.camera[0]+.5f && pan.camera[1]==initial.camera[1]+.5f);
 // Bounded phases return/reverse rather than accumulating a sheet translation.
 bool positive=false,negative=false;float last=0;
 for(int second=1;second<=240;++second){frame.presentation_time_ticks=second*1000;
  auto sample=water_material_frame(frame);
  assert(std::abs(sample.drift[0])<=.11f && std::abs(sample.drift[1])<=.18f && std::abs(sample.drift[2])<=.20f);
  positive=positive || sample.drift[0]>last;negative=negative || sample.drift[0]<last;last=sample.drift[0];
 }
 assert(positive && negative);
}
''')

    def test_drainage_and_curve_inputs(self):
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include "Renderer/native/render_core/world_topology.h"
#include "Renderer/lab/shared/natural/world.h"
int main(){
 using namespace c3x_renderer::render_core;
 WorldTopology world;World dims{32,32,true,false};
 std::vector<std::uint32_t> tiles(512,2u|(2u<<8));world.update(dims,tiles.data(),tiles.size());
 for(int c=20;c<=30;++c)tiles[world.index(c,8)]|=32u<<16;
 auto mouth=world.index(31,7);tiles[mouth]=12u|(12u<<8);
 world.update(dims,tiles.data(),tiles.size());
 for(int c=20;c<=30;++c){
  assert(((world.river_flow(world.index(c,8))>>4)&3)==1);
  assert(world.river_flow(world.index(c,8))==world.river_flow(world.index(c-16,-8)));
 }
 hydro::Field field;field.map_width=32;field.map_height=32;field.wraps=true;
 for(int r=5;r<11;++r)for(int c=18;c<33;++c){auto i=world.index(c,r);auto v=world.at(i);
  field.tiles[{c,r}]={c,r,c+r,c-r,int(v&255),int((v>>8)&255),unsigned((v>>16)&255),world.river_flow(i)};
 }
 river::Corridor corridor;corridor.build(field,[](double,double){return 0.;});
 for(int c=21;c<30;++c){auto sample=corridor.sample({c+.5,8});assert(sample.distance<12);assert(sample.flow.x>.8);}
 // A far-away outlet changes the direction of unchanged upstream terrain.
 auto upstream=world.index(28,8);auto before=world.at(upstream);auto direction=world.river_flow(upstream);
 c3x_renderer::fidelity::NaturalWorld::PageInputs proof;
 proof.values.push_back({upstream,before});proof.flow.push_back(static_cast<unsigned char>(direction));
 assert(proof.valid(world,1));
 tiles[mouth]=2u|(2u<<8);world.update(dims,tiles.data(),tiles.size());
 assert(!proof.valid(world,2));
 assert(world.at(upstream)==before);assert(world.river_flow(upstream)!=direction);
 // A second outlet creates a deterministic watershed, independent of view.
 tiles[world.index(19,7)]=12u|(12u<<8);world.update(dims,tiles.data(),tiles.size());
 assert(((world.river_flow(upstream)>>4)&3)==2);
 assert(world.river_flow(world.index(10,10))==0);
}
''')


if __name__ == '__main__':
    unittest.main()
