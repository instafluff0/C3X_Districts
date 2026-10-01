"""Visibility dependencies and the native fog ownership boundary."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class VisibilityTests(unittest.TestCase):
    def test_fresh_map_without_resources_refreshes_visibility(self):
        text=Path('Renderer/native/c3x_renderer.cpp').read_text()
        prefix=text.split('bool compose_resource_animations(',1)[1].split(
            'if (!shared_scene_surface',1)[0]
        body=prefix.split('{',1)[1]
        run_cpp(r'''
#include "Renderer/native/render_core/visibility_coverage.h"
#include "Renderer/native/render_core/scene_membership.h"
#include <cassert>
#include <vector>
using namespace c3x_renderer::render_core;
struct Renderer {
 VisibilityCoverage visibility_coverage;bool visibility_pass=true;
 long long resource_composite_ticks=1;
 std::vector<int> resource_anchors,sandbox_resource_poses,sandbox_aquatic_resource_poses,sandbox_pose_chunks;
 enum {geometry_water,geometry_river};
 struct Chunk {struct {int left=0,top=0,right=0,bottom=0;} bounds;
  int translation_x=0,translation_y=0;float natural_projection[4]={};};
 Chunk sources[2];SceneMembership<Chunk,2> geometry_vertex_buffers;
 Renderer(){for(unsigned i=0;i<2;++i)geometry_vertex_buffers.edit(i).emplace_back(sources[i]);}
 bool sample(c3x_renderer_frame_v1 const& frame,bool pose_only){
''' + body + r'''
 return false;
 }
};
int main(){
 Renderer renderer;c3x_renderer_frame_v1 frame{};
 frame.target_width=frame.tile_width=128;frame.target_height=frame.tile_height=64;
 c3x_renderer_tile_v1 tile{};frame.tiles=&tile;frame.tile_count=1;
 for(unsigned flags:std::vector<unsigned>{C3X_RENDERER_TILE_VISIBILITY_BITS,
      C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED,
      C3X_RENDERER_TILE_VISIBILITY_KNOWN}){
  tile.tile_flags=C3X_RENDERER_TILE_RENDER|flags;
  assert(renderer.sample(frame,true));
  unsigned wanted=(flags&C3X_RENDERER_TILE_VISIBLE)?2:(flags&C3X_RENDERER_TILE_EXPLORED)?1:0;
  assert(renderer.visibility_coverage.state(0,0)==wanted);
  for(auto const& layer:renderer.geometry_vertex_buffers)
   assert(layer[0].water_visible==(wanted==2));
  assert(!renderer.visibility_coverage.tiles.empty());
 }
 tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 assert(!renderer.sample(frame,true)); // never publish an unclassified map
}
''')

    def test_fresh_output_masks_the_actual_publication_target(self):
        text=Path('Renderer/sandbox/resident_scene.cpp').read_text()
        body=text.split('bool presented=sandbox_backbuffer_output.draw(',1)[1].split(
            'renderer.trace.write("fresh-callback",presented?',1)[0]
        run_cpp(r'''
#include <cassert>
#define SUCCEEDED(x) ((x)>=0)
struct ID3D11Texture2D {int id;};
namespace Microsoft {namespace WRL {
template<class T>struct ComPtr {T* value=nullptr;T** operator&(){return &value;}
 T* Get(){return value;}T* operator->(){return value;}
 template<class U>int As(U** output){return value->As(output);}};
}}
struct ID3D11Resource {ID3D11Texture2D* value;
 int As(ID3D11Texture2D** output){*output=value;return 0;}};
struct Target {ID3D11Resource resource;
 void GetResource(ID3D11Resource** output){*output=&resource;}};
bool drawn=false,draw_ok=true,apply_ok=true,overlay_ok=true;int masks=0,overlays=0;
struct Fresh {struct {struct {ID3D11Texture2D* depth_texture=nullptr;} linear;} glow;
 struct Overlay {template<class D,class C,class F>
 bool draw(D,C,ID3D11Texture2D* texture,F const&,ID3D11Texture2D* actors,unsigned guard,float zoom){
  assert(drawn&&texture->id==42&&actors==nullptr&&guard==4&&zoom==1.25f);
  ++overlays;return overlay_ok;
 }} city_site_overlay;
} sandbox_fresh;
struct Output {bool draw(Target*,int,int){drawn=true;return draw_ok;}} sandbox_backbuffer_output;
struct Renderer {bool visibility_pass=true;int device=0,context=0,visibility_coverage=123;
 struct Mask {bool apply(int,int,ID3D11Texture2D* texture,int coverage,ID3D11Texture2D* actors,unsigned guard,float zoom){
  assert(actors==sandbox_fresh.glow.linear.depth_texture&&guard==4&&zoom==1.25f);
  assert(drawn&&texture->id==42&&coverage==123);++masks;return apply_ok;}} visibility_gpu;
} renderer;
struct Frame {int target_width=128,target_height=64;} frame;
bool sample(Target* target,float zoom=1.25f){
 bool presented=sandbox_backbuffer_output.draw(''' + body + r'''
 return presented;
}
int main(){
 ID3D11Texture2D texture{42};Target target{{&texture}};
 for(int i=0;i<60;++i){drawn=false;assert(sample(&target));}
 assert(masks==60&&overlays==60); // one overlay and fog pass per frame
 renderer.visibility_pass=false;assert(sample(&target)&&masks==60&&overlays==61);
 renderer.visibility_pass=true;draw_ok=false;assert(!sample(&target)&&masks==60&&overlays==61);
 draw_ok=true;overlay_ok=false;assert(!sample(&target)&&masks==60&&overlays==62);
 overlay_ok=true;apply_ok=false;assert(!sample(&target)&&masks==61&&overlays==63);
}
''')

    def test_coverage_and_capture(self):
        run_cpp(r'''
#include "Renderer/native/render_core/visibility_coverage.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_frame_v1 f={};f.target_width=128;f.target_height=64;f.tile_width=128;f.tile_height=64;
 c3x_renderer_tile_v1 t={};t.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 f.tiles=&t;f.tile_count=1;VisibilityCoverage c;
 std::vector<unsigned> source(128*64,0xaabb8844),out;
 assert(c.capture(f));assert(!c.may_contribute(0,0,128,64));c.apply(source.data(),out);assert(out[32*128+64]==0xaa000000);
 t.tile_flags|=C3X_RENDERER_TILE_EXPLORED;assert(c.capture(f));c.apply(source.data(),out);
 assert(out[32*128+64]==0xaa745b39); // 50% gray over each original channel
 t.tile_flags|=C3X_RENDERER_TILE_VISIBLE;assert(c.capture(f));c.apply(source.data(),out);
 assert(out[32*128+64]==source[32*128+64]);assert(out[1*128+64]!=source[1*128+64]);
 auto again=out;c.apply(source.data(),out);assert(again==out && source[0]==0xaabb8844);
 t.tile_flags&=~C3X_RENDERER_TILE_EXPLORED;assert(!c.capture(f));
 t.tile_flags= C3X_RENDERER_TILE_RENDER;assert(!c.capture(f));
 t.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;assert(c.capture(f)&&c.tiles.empty());
 std::vector<c3x_renderer_tile_v1> records;
 for(int v=-1;v<=1;++v)for(int u=-1;u<=1;++u){auto n=t;n.tile_x=u-v;n.tile_y=u+v;
 n.tile_flags=C3X_RENDERER_TILE_VISIBILITY_BITS|((u||v)?C3X_RENDERER_TILE_TOPOLOGY_HALO:C3X_RENDERER_TILE_RENDER);records.push_back(n);}
 f.tiles=records.data();f.tile_count=unsigned(records.size());assert(c.capture(f)&&c.tiles.empty());assert(c.may_contribute(0,0,128,64));assert(!c.may_contribute(128,64,256,128));
 records.push_back(records[4]);records.back().tile_x+=20;records.back().anchor_x+=128;
 f.world_width_tiles=20;f.world_wrap_x=1;f.tiles=records.data();f.tile_count=unsigned(records.size());assert(c.capture(f));
 records.back().tile_flags&=~C3X_RENDERER_TILE_VISIBLE;assert(!c.capture(f));
 f.tile_count=8193;assert(!c.capture(f));
}
''')

    def test_native_config_off_forwarding(self):
        text=Path('injected_code.c').read_text()
        start=text.index('void __fastcall\npatch_Map_Renderer_draw_fog')
        body=text[start:text.index('\n}\n',start)+3]
        run_cpp('''
#include "Renderer/native/c3x_renderer_api.h"
#define __fastcall
#define __ 0
struct Map_Renderer{};struct PCX_Image{};struct RECT{};
struct State {struct {bool enable_custom_rendering;} current_config;
 bool custom_renderer_frame_active=false,custom_renderer_draw_in_progress=false,custom_renderer_redraw_pending=false;
 int custom_renderer_tile_count=0;unsigned custom_renderer_dirty_flags=0;c3x_renderer_tile_v1* custom_renderer_tiles=nullptr;
} state,*is=&state;
struct Bic {struct {Map_Renderer Renderer;} Map;} bic,*p_bic_data=&bic;
int tile_at(int,int){return 0;}unsigned capture_custom_renderer_visibility(int,int,int,int){return 0;}
int calls=0;Map_Renderer renderer;PCX_Image image;RECT clip;
void Map_Renderer_draw_fog(Map_Renderer* r,int,int v,PCX_Image* i,RECT* c){
 if(r!=&renderer || v!=7 || i!=&image || c!=&clip)__builtin_abort();++calls;
}
''' + body.replace('this', 'self') + '''
int main(){patch_Map_Renderer_draw_fog(&renderer,0,7,&image,&clip);
 state.current_config.enable_custom_rendering=true;patch_Map_Renderer_draw_fog(&renderer,0,7,&image,&clip);
 return calls!=1;}
''')

    def test_authoritative_native_capture(self):
        text=Path('injected_code.c').read_text()
        def function(start):
            a=text.index(start);return text[a:text.index('\n}\n',a)+3].replace('this','self')
        program=r'''#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#define __fastcall
#define __ 0
struct Tile {struct {unsigned Fog_Of_War,FOWStatus,V3,Visibility,field_D0_Visibility;} Body;} tile,null_tile,*p_null_tile=&null_tile;
struct Leader {int ID;} leaders[32];
struct City {struct {int X,Y;} Body;} city;
struct Map {struct {City* spotlight_on_city=nullptr;} Renderer;};
struct Bic {struct Map Map;} bic,*p_bic_data=&bic;
struct State {struct {bool share_visibility_in_hotseat=false;} current_config;} state,*is=&state;
unsigned debug=0,human=6;unsigned* p_debug_mode_bits=&debug;unsigned* p_human_player_bits=&human;
bool offline=false,pbem=false,online=false,controls=false;bool* p_is_offline_mp_game=&offline;bool* p_is_pbem_game=&pbem;
bool is_online_game(){return online;}Tile* tile_at(int,int){return &tile;}
using Tile_Body=decltype(tile.Body);
int patch_Map_compute_ni_for_work_area(Map*,int,int,int,int x,int y,int){return (x==2&&y==4)?7:-1;}
bool patch_City_controls_tile(City*,int,int n,bool){assert(n==7);return controls;}
''' + function('bool\nis_explored (') + function('bool __fastcall\npatch_Leader_is_tile_visible (') + function('unsigned int\ncapture_custom_renderer_visibility (') + r'''int main(){
 leaders[1].ID=1;unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,visible=C3X_RENDERER_TILE_VISIBLE;
 auto get=[](){return capture_custom_renderer_visibility(&tile,1,2,4);};
 assert(get()==known);tile.Body.Fog_Of_War=2;assert(get()==(known|explored));
 for(auto field:{&tile.Body.FOWStatus,&tile.Body.V3,&tile.Body.Visibility,&tile.Body.field_D0_Visibility}){*field=2;assert(get()==(known|explored|visible));*field=0;}
 tile.Body.Fog_Of_War=4;state.current_config.share_visibility_in_hotseat=true;offline=true;tile.Body.V3=4;assert(get()==(known|explored|visible));
 pbem=true;assert(get()==known);pbem=false;offline=false;
 debug=8;assert(get()==(known|explored|visible));online=true;assert(get()==known && !(debug&12));online=false;
 tile.Body.Fog_Of_War=2;debug=1;assert(get()==(known|explored|visible));debug=0;
 bic.Map.Renderer.spotlight_on_city=&city;controls=false;assert(get()==(known|explored));controls=true;assert(get()==(known|explored|visible));
 assert(capture_custom_renderer_visibility(&tile,1,0,0)==(known|explored));
 assert(capture_custom_renderer_visibility(nullptr,1,0,0)==0);assert(capture_custom_renderer_visibility(&tile,32,0,0)==0);
 assert(capture_custom_renderer_visibility(&tile,0,0,0)==(known|explored|visible));
}
'''
        run_cpp('#include <initializer_list>\n'+program)

    def test_gpu_oracle(self):
        run_cpp('int test_gpu_visibility();int main(){return test_gpu_visibility();}',
                sources=('Renderer/native/test_gpu_visibility.cpp',),timeout=90)

if __name__=='__main__':unittest.main()
