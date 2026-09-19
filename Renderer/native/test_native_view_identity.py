"""Execute the injected native identity observations and ordinary-call dispatch."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class NativeViewIdentityTests(unittest.TestCase):
    def test_default_handoff_requires_capabilities_and_allows_diagnostic_opt_out(self):
        source = (ROOT / 'injected_code.c').read_text()
        selection = source.split('char async_option[8] = {0};', 1)[1]
        selection = '#ifdef Main_Screen_Form_move_camera' + selection.split('#ifdef Main_Screen_Form_move_camera', 1)[1].split('#endif', 1)[0] + '#endif'
        run_cpp(r"""
#include <cassert>
#include <initializer_list>
#include <cstring>
struct State {
 bool custom_renderer_async_enabled=false;
 void *custom_renderer_render_view=this, *custom_renderer_camera_begin=this,
      *custom_renderer_camera_poll=this, *custom_renderer_camera_present=this,
      *custom_renderer_camera_cancel=this;
};
#define Main_Screen_Form_move_camera available
void select(State* is, char const* async_option) {
""" + selection + r"""
}
#undef Main_Screen_Form_move_camera
void unsupported(State* is, char const* async_option) {
""" + selection + r"""
}
int main() {
 State state;select(&state, "");assert(state.custom_renderer_async_enabled);
 select(&state, "0");assert(!state.custom_renderer_async_enabled);
 select(&state, "1");assert(state.custom_renderer_async_enabled);
 for(auto member:{&State::custom_renderer_render_view,&State::custom_renderer_camera_begin,
                  &State::custom_renderer_camera_poll,&State::custom_renderer_camera_present,
                  &State::custom_renderer_camera_cancel}) {
  State missing;missing.*member=nullptr;select(&missing, "");assert(!missing.custom_renderer_async_enabled);
 }
 unsupported(&state, "");assert(!state.custom_renderer_async_enabled);
}
""")

    def test_capture_lifecycle_visibility_and_legacy_dispatch(self):
        source = (ROOT / 'injected_code.c').read_text()
        world = 'bool\ncapture_custom_renderer_world_topology ()' + source.split('capture_custom_renderer_world_topology ()', 1)[1].split('\nvoid\n', 1)[0]
        viewer = source.split('\t// A complete capture belongs to one authoritative native viewer.', 1)[1].split('\tint const max_tiles', 1)[0]
        retire = source.split('\t// No publication survives unload;', 1)[1].split('\tis->custom_renderer_capture_world_topology = false;', 1)[0]
        retire = retire.split('\n', 1)[1]
        dispatch = '\tstruct c3x_renderer_camera_request_v1 request = {0};' + source.split('\tstruct c3x_renderer_camera_request_v1 request = {0};', 1)[1].split('\tif (is->custom_renderer_presented_frames == 0)', 1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
struct LARGE_INTEGER {long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* p){p->QuadPart=1000;}
void debug(char const*){} auto p_OutputDebugStringA=&debug;
std::size_t fail_at=~std::size_t(0),largest_request=0;
unsigned fail_nth=0;
struct Memory {void* p;template<class T>operator T*()const{return static_cast<T*>(p);}};
Memory allocate(void* p,std::size_t bytes){largest_request=std::max(largest_request,bytes);return {bytes>=fail_at || (fail_nth && !--fail_nth)?nullptr:std::realloc(p,bytes)};}
#define realloc allocate
#define malloc(bytes) allocate(nullptr,bytes)
struct Tile;
struct Vtable {int(*m49_Get_Square_RealType)(Tile*);int(*m50_Get_Square_BaseType)(Tile*);int(*m37_Get_River_Code)(Tile*);};
struct Tile {Vtable* vtable;struct {int FOWStatus=0,Visibility=0,Fog_Of_War=0,V3=0,field_D0_Visibility=0;void* active_tile_effect=nullptr;}Body;int ground=2,base=2,river=0;};
Vtable vtable{[](Tile*t){return t->ground;},[](Tile*t){return t->base;},[](Tile*t){return t->river;}};
struct MapData{int Width=4,Height=4;};using Map=MapData;
struct Bic {MapData Map;} bic;Bic* p_bic_data=&bic;
Tile null_tile{&vtable};Tile* p_null_tile=&null_tile;
std::vector<Tile> tiles(5000,Tile{&vtable});int absent=-1;
Tile* tile_at(int x,int y){int at=(y*bic.Map.Width+x)/2;return at==absent?p_null_tile:&tiles.at(at);}
unsigned modern_calls=0,legacy_calls=0,resident_calls=0;c3x_renderer_camera_identity_v1 received{};
int resident_result=C3X_RENDERER_RESULT_OK;bool probe=true;
bool custom_renderer_native_probe_on(){return probe;}
int resident(int action,void* image,c3x_renderer_camera_request_v1 const* r,c3x_renderer_camera_view_v1* v){
 v->frame.presentation_time_ticks=55;
 assert(action==C3X_NATIVE_MAP_PREPARE&&image&&r);++resident_calls;received=r->identity;return resident_result;
}
int modern(c3x_renderer_camera_request_v1 const* r,c3x_renderer_output_v1*){
 assert(r->version==C3X_RENDERER_CAMERA_VIEW_VERSION && r->struct_size==sizeof(*r));received=r->identity;++modern_calls;return 7;
}
int legacy(c3x_renderer_frame_v1 const*,c3x_renderer_output_v1*){++legacy_calls;return 9;}
struct State {
 bool custom_renderer_async_drawing=false;
 c3x_renderer_camera_present_view_fn custom_renderer_camera_present=nullptr;
 long long custom_renderer_display_clock=0,custom_renderer_camera_ticket=0;
 c3x_renderer_render_view_fn custom_renderer_render_view=modern;
 c3x_renderer_render_fn custom_renderer_render=legacy;
 c3x_renderer_native_map_view_fn custom_renderer_native_map=nullptr;
 c3x_renderer_native_lifetime_fn custom_renderer_native_lifetime=nullptr;
 unsigned* custom_renderer_world_topology=nullptr;
 unsigned long long* custom_renderer_world_visibility=nullptr;
 int custom_renderer_world_topology_count=0,custom_renderer_tile_count=0,custom_renderer_viewer_civ_id=-1;
 long long custom_renderer_world_topology_revision=0,custom_renderer_visibility_revision=0;
 long long custom_renderer_map_epoch=0,custom_renderer_viewer_epoch=0;
 unsigned custom_renderer_requested_frames=0;LARGE_INTEGER custom_renderer_qpc_frequency{1000};
} state;State* is=&state;
''' + world + '\nbool viewer(int visible_to_civ_id){\n' + viewer + '\nreturn true;}\nvoid retire(){\n' + retire + r'''
}
int demand(){void* image=is;c3x_renderer_frame_v1 frame={};frame.presentation_time_ticks=77;frame.world_topology_revision=is->custom_renderer_world_topology_revision;c3x_renderer_output_v1 output={};
''' + dispatch + r'''
 return render_result;
}
int main(){
 assert(viewer(3) && state.custom_renderer_viewer_epoch==1);
 assert(viewer(3) && state.custom_renderer_viewer_epoch==1);
 state.custom_renderer_tile_count=3;assert(!viewer(4) && state.custom_renderer_viewer_civ_id==3);
 state.custom_renderer_tile_count=0;assert(viewer(4) && state.custom_renderer_viewer_epoch==2);
 assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_count==8 && state.custom_renderer_world_topology_revision==1 && state.custom_renderer_visibility_revision==1);
 auto topology=state.custom_renderer_world_topology_revision,visibility=state.custom_renderer_visibility_revision;
 assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_revision==topology && state.custom_renderer_visibility_revision==visibility);
 tiles[7].Body.Visibility=1;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_revision==topology && state.custom_renderer_visibility_revision==++visibility);
 tiles[7].Body.FOWStatus=0x12345678;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_visibility_revision==++visibility); // All sources of current visibility and explored state are observed.
 tiles[6].Body.V3=8;assert(capture_custom_renderer_world_topology());assert(state.custom_renderer_visibility_revision==++visibility);
 tiles[6].Body.field_D0_Visibility=16;assert(capture_custom_renderer_world_topology());assert(state.custom_renderer_visibility_revision==++visibility);
 tiles[6].Body.Fog_Of_War=32;assert(capture_custom_renderer_world_topology());assert(state.custom_renderer_visibility_revision==++visibility);
 tiles[2].ground=4;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_revision==++topology && state.custom_renderer_visibility_revision==visibility);
 assert(demand()==7 && modern_calls==1 && !legacy_calls && state.custom_renderer_map_epoch==1);
 assert(received.map_epoch==1 && received.viewer_epoch==2 && received.visibility_epoch==visibility && received.scene_epoch==topology);
 bic.Map={100,100};assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_count==5000 && largest_request==5000*sizeof(unsigned long long));
 // Either allocation may fail during shrink. Returning to the old map must
 // retain both complete old owners; reallocating topology first used to shrink it.
 auto* old_topology=state.custom_renderer_world_topology;
 auto* old_visibility=state.custom_renderer_world_visibility;
 for(unsigned allocation: {1u,2u}){
  bic.Map={4,4};fail_nth=allocation;
  assert(!capture_custom_renderer_world_topology());
  assert(state.custom_renderer_world_topology==old_topology && state.custom_renderer_world_visibility==old_visibility);
  assert(state.custom_renderer_world_topology_count==5000);
  bic.Map={100,100};assert(capture_custom_renderer_world_topology());
 }
 // An allocation failure cannot publish a new count with an undersized visibility owner.
 bic.Map={2048,2048};fail_at=16u*1024u*1024u;
 assert(!capture_custom_renderer_world_topology() && largest_request==fail_at && state.custom_renderer_world_topology_count==5000);
 fail_at=~std::size_t(0);bic.Map={4,4};assert(capture_custom_renderer_world_topology());
 absent=3;assert(!capture_custom_renderer_world_topology() && !state.custom_renderer_world_topology_count);
 absent=-1;assert(capture_custom_renderer_world_topology());
 auto count=state.custom_renderer_world_topology_count;
 for(auto dims: {MapData{2049,4},MapData{4,2049},MapData{3,4},MapData{0,4}}){
  bic.Map=dims;assert(!capture_custom_renderer_world_topology() && state.custom_renderer_world_topology_count==count);
 }
 retire();assert(!state.custom_renderer_world_visibility && !state.custom_renderer_visibility_revision);
 assert(state.custom_renderer_map_epoch==2 && !state.custom_renderer_viewer_epoch && state.custom_renderer_viewer_civ_id==-1);
 state.custom_renderer_render_view=nullptr;assert(demand()==9 && legacy_calls==1 && modern_calls==1);
 bic.Map={4,4};assert(capture_custom_renderer_world_topology() && !state.custom_renderer_world_visibility); // Older DLL compatibility.
 // The real map dispatch carries the same epochs through the resident owner,
 // and only an admission rejection permits the existing CPU path.
 state.custom_renderer_render_view=modern;state.custom_renderer_native_map=resident;
 state.custom_renderer_native_lifetime=[](int,void*,int){return 1;};
 assert(demand()==C3X_RENDERER_RESULT_OK&&resident_calls==1&&modern_calls==1&&legacy_calls==1);
 assert(state.custom_renderer_display_clock==55&&received.map_epoch==state.custom_renderer_map_epoch);
 resident_result=C3X_RENDERER_RESULT_DEVICE_ERROR;
 assert(demand()==C3X_RENDERER_RESULT_DEVICE_ERROR&&modern_calls==1&&legacy_calls==1);
 resident_result=C3X_RENDERER_RESULT_BAD_ARGUMENT;
 assert(demand()==7&&modern_calls==2&&resident_calls==3);
 probe=false;assert(demand()==7&&modern_calls==3&&resident_calls==3);
 std::free(state.custom_renderer_world_topology);
}
''')

    def test_current_camera_publication_and_request_only_capture(self):
        source = (ROOT / 'injected_code.c').read_text()
        header = (ROOT / 'C3X.h').read_text()
        view = 'struct custom_renderer_native_view {' + header.split('struct custom_renderer_native_view {', 1)[1].split('};', 1)[0] + '};'
        helpers = 'struct custom_renderer_native_view\ncustom_renderer_native_view' + source.split('struct custom_renderer_native_view\ncustom_renderer_native_view', 1)[1].split('void __fastcall\npatch_Map_Renderer_m71_Draw_Tiles', 1)[0]
        body = source.split('void __fastcall\npatch_Map_Renderer_m71_Draw_Tiles', 1)[1]
        start = '\tif (custom_renderer_zoom_enabled ()) sync_custom_renderer_zoom_to_native ();' + body.split('\tif (custom_renderer_zoom_enabled ()) sync_custom_renderer_zoom_to_native ();', 1)[1].split('\tis->custom_renderer_draw_in_progress = true;', 1)[0]
        finish = '\tif (async_view) {' + body.split('\tif (async_view) {', 1)[1].split('\tis->custom_renderer_frame_active = false;', 1)[0]
        program = r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstring>
#include <algorithm>
#define __fastcall
#define __ 0
#define Main_Screen_Form_move_camera native_move
struct RECT {int left=0,top=0,right=2240,bottom=1192;};
struct JGL_Image;
struct ImageVtable {int(*m54_Get_Width)(JGL_Image*);int(*m55_Get_Height)(JGL_Image*);};
struct JGL_Image {ImageVtable* vtable;RECT Clip_Rect;};
ImageVtable image_vtable{[](JGL_Image*){return 2240;},[](JGL_Image*){return 1192;}};
JGL_Image image{&image_vtable};
struct JGL {JGL_Image* Image=&image;};
struct Map_Renderer;
struct Vtable {void* m21_Draw_Tiles_by_Flags;};
struct Map_Renderer {Vtable* vtable;struct JGL JGL;};
struct PCX_Image {void* vtable;struct JGL JGL;};
struct MapData {Map_Renderer Renderer;};
struct Bic {MapData Map;bool is_zoomed_out=false;} bic;
Bic* p_bic_data=&bic;
struct Main_Screen_Form {int camera_x=0,camera_y=0,TileX_Min=0,TileX_Max=20,TileY_Min=0,TileY_Max=20;} screen;
Main_Screen_Form* p_main_screen_form=&screen;
struct Clock {long long QuadPart=0;};
''' + view + r'''
struct State {
 bool custom_renderer_async_enabled=true,custom_renderer_display_valid=false,custom_renderer_nearby_preparing=false;
 bool custom_renderer_draw_in_progress=false,custom_renderer_async_drawing=false,custom_renderer_async_presented=false;
 bool custom_renderer_capture_only=false,custom_renderer_capture_failed=false,custom_renderer_capture_world_topology=true;
 int custom_renderer_zoom_tile_width=128,custom_renderer_tile_count=0;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
 long long custom_renderer_camera_ticket=0,custom_renderer_display_clock=0;
 long long custom_renderer_map_epoch=1,custom_renderer_viewer_epoch=2,custom_renderer_visibility_revision=3;
 struct custom_renderer_native_view custom_renderer_display_view{},custom_renderer_queued_view{};
 c3x_renderer_tile_v1 storage[2]{},*custom_renderer_tiles=storage;
 unsigned custom_renderer_visible_animation_count=0;
 Clock custom_renderer_animation_timestamp,custom_renderer_qpc_frequency;
 c3x_renderer_camera_begin_view_fn custom_renderer_camera_begin;
 c3x_renderer_camera_poll_view_fn custom_renderer_camera_poll;
 c3x_renderer_camera_cancel_fn custom_renderer_camera_cancel;
} state;State* is=&state;
bool custom_renderer_zoom_enabled(){return true;}
void sync_custom_renderer_zoom_to_native(){}
void native_move(Main_Screen_Form* s,int,int x,int y,int,bool){
 s->camera_x=(x%8192+8192)%8192;s->camera_y=(y%4096+4096)%4096;
 s->TileX_Min=s->camera_x/64;s->TileX_Max=s->TileX_Min+20;
 s->TileY_Min=s->camera_y/32;s->TileY_Max=s->TileY_Min+20;
}
unsigned captures=0,begins=0,cancels=0;int queued_x=-1,poll_status=C3X_RENDERER_RESULT_PENDING;
void capture(Map_Renderer* target,int,int viewer,int,int,Map_Renderer* output,void* clip,int x,int y,int flags){
 assert(state.custom_renderer_capture_only && !clip && x==-1 && y==-1 && flags==9 && output==target && viewer==2);
 ++captures;state.custom_renderer_tile_count=2;
 state.storage[0]={};state.storage[0].visibility_mask=8;state.storage[0].anchor_x=-screen.camera_x;
 image.Clip_Rect.left=7; // The real traversal may change clipping; the bridge restores it.
}
void capture_custom_renderer_topology(int viewer,int mask){assert(viewer==2 && mask==8);}
bool prepare_custom_renderer_frame(c3x_renderer_frame_v1* frame){
 *frame={};frame->tiles=state.storage;frame->tile_count=2;
 frame->world_topology_revision=4;frame->presentation_time_ticks=state.custom_renderer_animation_timestamp.QuadPart;return true;
}
int begin(c3x_renderer_camera_request_v1 const* r,long long* ticket){
 assert(r->identity.map_epoch==1 && r->identity.viewer_epoch==2 && r->identity.visibility_epoch==3 && r->identity.scene_epoch==4);
 queued_x=-r->frame->tiles[0].anchor_x;*ticket=++begins;return C3X_RENDERER_RESULT_PENDING;
}
int poll(long long,c3x_renderer_camera_view_v1*){return poll_status;}
int cancel(long long){++cancels;return C3X_RENDERER_RESULT_OK;}
''' + helpers.replace('this', 'screen_arg') + r'''
int displayed_x=-1,animator_x=-1,animator_y=-1;
void call(bool success=true){Map_Renderer* screen_arg=&bic.Map.Renderer;int param_1=2;
 // Native Animator_update computes canvas and wrap copies before calling m71.
 animator_x=screen.camera_x;animator_y=screen.camera_y;
''' + start.replace('this', 'screen_arg') + r'''
 assert(screen.camera_x==animator_x && screen.camera_y==animator_y);
 state.custom_renderer_async_presented=success;
 if(success){displayed_x=screen.camera_x;state.custom_renderer_display_clock=state.custom_renderer_animation_timestamp.QuadPart;}
''' + finish.replace('this', 'screen_arg') + r'''
 assert(screen.camera_x==animator_x && screen.camera_y==animator_y);
}
int main(){
 Vtable vt{reinterpret_cast<void*>(&capture)};bic.Map.Renderer.vtable=&vt;
 state.custom_renderer_camera_begin=begin;state.custom_renderer_camera_poll=poll;state.custom_renderer_camera_cancel=cancel;
 call();assert(displayed_x==0 && state.custom_renderer_display_valid && begins==0);
 // Every native movement is now an exact barrier. The real animator observes
 // these fields between movement and m71; the previous fixture omitted it.
 patch_Main_Screen_Form_move_camera(&screen,0,32,0,1,false);
 assert(screen.camera_x==32 && !state.custom_renderer_display_valid);
 call();assert(displayed_x==32 && begins==0 && captures==0);
 for(int i=0;i<20;++i){
  patch_Main_Screen_Form_move_camera(&screen,0,screen.camera_x+32,0,1,false);
  assert(screen.camera_x==64+i*32); // Input cannot be hidden behind publication.
  call();assert(displayed_x==screen.camera_x && begins==0);
 }
 // Stationary ambient updates still use the bounded caller-driven queue.
 state.custom_renderer_qpc_frequency.QuadPart=150;
 state.custom_renderer_visible_animation_count=1;
 state.custom_renderer_animation_timestamp.QuadPart=30;
 queue_custom_renderer_native_view(&bic.Map.Renderer,2,&state.custom_renderer_display_view);
 assert(begins==1 && state.custom_renderer_camera_ticket && queued_x==672);
 // Movement cancels old ambient work even if it is never ready. Native bounds,
 // picking and unit culling immediately see the new native camera.
 patch_Main_Screen_Form_move_camera(&screen,0,640,0,1,false);
 assert(cancels==1 && !state.custom_renderer_camera_ticket && screen.camera_x==640);
 assert(screen.TileX_Min==10 && !state.custom_renderer_display_valid);
 state.custom_renderer_visible_animation_count=0;
 call();assert(displayed_x==640 && !state.custom_renderer_camera_ticket);
 // Camera changes that bypass the move hook still invalidate old queued output.
 // A completed old ticket cannot rewrite the native animator's current camera.
 state.custom_renderer_visible_animation_count=1;
 state.custom_renderer_animation_timestamp.QuadPart+=30;
 queue_custom_renderer_native_view(&bic.Map.Renderer,2,&state.custom_renderer_display_view);
 assert(state.custom_renderer_camera_ticket);
 native_move(&screen,0,704,64,1,false);poll_status=C3X_RENDERER_RESULT_OK;
 state.custom_renderer_visible_animation_count=0;
 call();assert(cancels==2 && displayed_x==704 && screen.camera_y==64);
 assert(!state.custom_renderer_camera_ticket && screen.TileX_Min==11);
 // Projection changes likewise reject a ready old-view publication.
 state.custom_renderer_visible_animation_count=1;
 state.custom_renderer_animation_timestamp.QuadPart+=30;
 queue_custom_renderer_native_view(&bic.Map.Renderer,2,&state.custom_renderer_display_view);
 assert(state.custom_renderer_camera_ticket);
 state.custom_renderer_zoom_tile_width=144;state.custom_renderer_visible_animation_count=0;
 call();assert(cancels==3 && !state.custom_renderer_camera_ticket && displayed_x==704);
 // Exact programmatic recenter, zoom, wrap and repeated reversal remain native.
 patch_Main_Screen_Form_move_camera(&screen,0,1000,320,0,true);
 call();assert(screen.camera_x==1000 && displayed_x==1000);
 state.custom_renderer_zoom_tile_width=160;call();
 assert(state.custom_renderer_display_view.tile_width==160);
 patch_Main_Screen_Form_move_camera(&screen,0,8180,0,0,true);call();
 patch_Main_Screen_Form_move_camera(&screen,0,screen.camera_x+32,0,1,false);
 assert(screen.camera_x==20);call();assert(displayed_x==20);
 patch_Main_Screen_Form_move_camera(&screen,0,screen.camera_x-32,0,1,false);
 assert(screen.camera_x==8180);call();assert(displayed_x==8180);
 unsigned before=begins;call(false);assert(!state.custom_renderer_display_valid && begins==before);
 state.custom_renderer_async_enabled=false;call();assert(!state.custom_renderer_display_valid && !state.custom_renderer_camera_ticket);
}
'''
        run_cpp(program)
