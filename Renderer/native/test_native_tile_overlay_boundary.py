"""Custom rendering captures maps without legacy highlights; vanilla keeps both."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class TileOverlayBoundaryTests(unittest.TestCase):
    def test_pending_and_completed_views_preserve_vanilla(self):
        source = (Path(__file__).resolve().parents[2] / 'injected_code.c').read_text()
        body = source.split('patch_Map_Renderer_m19_Draw_Tile_by_XY_and_Flags (', 1)[1].split('{', 1)[1]
        body = body.split('\tif ((is->city_loc_display_perspective >= 0)', 1)[0]
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include <initializer_list>
#define __ 0
#define __fastcall
#define C3X_RENDERER_RESULT_ERROR 0
struct Tile {} tile;
struct Map_Renderer;
struct Vtable {void (*m21_Draw_Tiles_by_Flags)(Map_Renderer*,int,int,int,int,Map_Renderer*,void*,int,int,int);};
struct Map_Renderer {Vtable* vtable=nullptr;} renderer;
struct Map {};
struct Bic {struct Map Map;} bic;auto p_bic_data=&bic;
struct State {
 struct {bool enable_custom_rendering=false;} current_config;
 bool custom_renderer_capture_only=false,custom_renderer_capture_failed=false;
 bool custom_renderer_frame_active=false,custom_renderer_composited=false,custom_renderer_async_presented=false;
 Tile* current_render_tile=nullptr;int current_render_tile_x=-1,current_render_tile_y=-1;
 void* current_render_tile_district=nullptr;
} state;auto is=&state;
int captured=0,native=0,overlays=0,composites=0;bool ready=false;
Tile* tile_at(int,int){return &tile;}
void* get_district_instance(Tile*){return nullptr;}
bool capture_custom_renderer_tile(int,int,int,Map_Renderer*,int,int,int,Tile*,bool){++captured;return true;}
void capture_custom_renderer_topology(int,int){}
void composite_custom_renderer_frame(){++composites;state.custom_renderer_async_presented=ready;}
void log_custom_renderer_event(char const*,int){}
void Map_Renderer_m19_Draw_Tile_by_XY_and_Flags(Map_Renderer*,int,int,int,int,Map_Renderer*,int,int,int,int){++native;}
void tile_draw(Map_Renderer* self,int param_1,int pixel_x,int pixel_y,Map_Renderer* map_renderer,int param_5,int tile_x,int tile_y,int param_8){
''' + body.replace('this', 'self') + r'''
 ++overlays; // Existing C3X tile highlights begin after the extracted gate.
}
void pass(int flags){for(int n=0;n<6;++n)tile_draw(&renderer,1,n*64,0,&renderer,0,n*2,0,flags);}
void traversal(){for(int flags:{9,4,0x1FEF0,2,0x100})pass(flags);}
int main(){
 Vtable vt{[](Map_Renderer*,int,int,int,int,Map_Renderer*,void*,int,int,int flags){pass(flags);}};
 renderer.vtable=&vt;
 state.custom_renderer_frame_active=true;
 // Even stale renderer bookkeeping must not suppress config-off native work.
 state.custom_renderer_capture_only=true;traversal();
 assert(native==30&&overlays==30&&captured==0&&composites==0);
 state.current_config.enable_custom_rendering=true;native=overlays=0;
 pass(9);assert(captured==6&&native==0&&overlays==0);
 state.custom_renderer_capture_only=false;captured=0;
 for(int attempt=0;attempt<3;++attempt){state.custom_renderer_composited=false;traversal();}
 assert(captured==18&&composites==3&&overlays==0&&native==0);
 state.custom_renderer_composited=false;ready=true;captured=composites=0;traversal();
 assert(captured==6&&composites==1&&overlays==0&&native==0);
 assert(!state.current_render_tile&&state.current_render_tile_x==-1&&state.current_render_tile_y==-1);
 // Partial custom calls still cannot reintroduce legacy highlight sprites.
 state.custom_renderer_frame_active=false;native=overlays=0;pass(2);
 assert(native==6&&overlays==0);
 state.current_config.enable_custom_rendering=false;native=overlays=0;pass(2);
 assert(native==6&&overlays==6);
}
''')


if __name__ == '__main__':
    unittest.main()
