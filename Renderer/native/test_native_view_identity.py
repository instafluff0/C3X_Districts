"""Execute the injected native identity observations and ordinary-call dispatch."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class NativeViewIdentityTests(unittest.TestCase):
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
struct Tile {Vtable* vtable;struct {int FOWStatus=0,Visibility=0;void* active_tile_effect=nullptr;}Body;int ground=2,base=2,river=0;};
Vtable vtable{[](Tile*t){return t->ground;},[](Tile*t){return t->base;},[](Tile*t){return t->river;}};
struct MapData{int Width=4,Height=4;};using Map=MapData;
struct Bic {MapData Map;} bic;Bic* p_bic_data=&bic;
Tile null_tile{&vtable};Tile* p_null_tile=&null_tile;
std::vector<Tile> tiles(5000,Tile{&vtable});int absent=-1;
Tile* tile_at(int x,int y){int at=(y*bic.Map.Width+x)/2;return at==absent?p_null_tile:&tiles.at(at);}
unsigned modern_calls=0,legacy_calls=0;c3x_renderer_camera_identity_v1 received{};
int modern(c3x_renderer_camera_request_v1 const* r,c3x_renderer_output_v1*){
 assert(r->version==C3X_RENDERER_CAMERA_VIEW_VERSION && r->struct_size==sizeof(*r));received=r->identity;++modern_calls;return 7;
}
int legacy(c3x_renderer_frame_v1 const*,c3x_renderer_output_v1*){++legacy_calls;return 9;}
struct State {
 c3x_renderer_render_view_fn custom_renderer_render_view=modern;
 c3x_renderer_render_fn custom_renderer_render=legacy;
 unsigned* custom_renderer_world_topology=nullptr;
 unsigned long long* custom_renderer_world_visibility=nullptr;
 int custom_renderer_world_topology_count=0,custom_renderer_tile_count=0,custom_renderer_viewer_civ_id=-1;
 long long custom_renderer_world_topology_revision=0,custom_renderer_visibility_revision=0;
 long long custom_renderer_map_epoch=0,custom_renderer_viewer_epoch=0;
 unsigned custom_renderer_requested_frames=0;LARGE_INTEGER custom_renderer_qpc_frequency{1000};
} state;State* is=&state;
''' + world + '\nbool viewer(int visible_to_civ_id){\n' + viewer + '\nreturn true;}\nvoid retire(){\n' + retire + r'''
}
int demand(){c3x_renderer_frame_v1 frame={};frame.world_topology_revision=is->custom_renderer_world_topology_revision;c3x_renderer_output_v1 output={};
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
 assert(state.custom_renderer_visibility_revision==++visibility); // Both complete 32-bit native fields are observed.
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
 std::free(state.custom_renderer_world_topology);
}
''')
