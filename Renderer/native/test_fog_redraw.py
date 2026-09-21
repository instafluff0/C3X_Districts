"""Native fog-only changes must request one authoritative map capture."""
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
import unittest
import re

class FogRedrawTests(unittest.TestCase):
    def test_native_visibility_invalidation(self):
        source=(ROOT/'injected_code.c').read_text()
        body=source[source.index('void __fastcall\npatch_Map_Renderer_draw_fog'):source.index('// Direct grid seam;')]
        body=re.sub(r'\bthis\b','context',body)
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstddef>
#define __fastcall
#define __ 0
struct PCX_Image{};struct RECT{};struct Map_Renderer{};
struct {struct {Map_Renderer Renderer;}Map;} bic;auto p_bic_data=&bic;
struct {struct {bool enable_custom_rendering=true;}current_config;
 bool custom_renderer_frame_active=false,custom_renderer_draw_in_progress=false,custom_renderer_redraw_pending=false;
 unsigned custom_renderer_dirty_flags=0;int custom_renderer_tile_count=2;
 c3x_renderer_tile_v1 custom_renderer_tiles[2]={};} state;auto is=&state;
unsigned visibility[2]={C3X_RENDERER_TILE_VISIBILITY_BITS,C3X_RENDERER_TILE_VISIBILITY_KNOWN};
unsigned* tile_at(int x,int){return &visibility[x];}
unsigned capture_custom_renderer_visibility(unsigned* t,int,int,int){return *t;}
int native=0;
void Map_Renderer_draw_fog(Map_Renderer*,int,int,PCX_Image*,RECT*){++native;}
'''+body+r'''
int main(){
 for(int i=0;i<2;++i){state.custom_renderer_tiles[i].tile_x=i;state.custom_renderer_tiles[i].tile_flags=C3X_RENDERER_TILE_RENDER|visibility[i];}
 auto draw=[&]{patch_Map_Renderer_draw_fog(&bic.Map.Renderer,0,1,nullptr,nullptr);};
 draw();assert(!state.custom_renderer_redraw_pending && !native);
 visibility[1]|=C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
 draw();assert(state.custom_renderer_redraw_pending && (state.custom_renderer_dirty_flags&C3X_RENDERER_DIRTY_SCENE));
 state.custom_renderer_redraw_pending=false;state.custom_renderer_tiles[1].tile_flags|=visibility[1];draw();assert(!state.custom_renderer_redraw_pending);
 visibility[0]&=~C3X_RENDERER_TILE_VISIBLE;draw();assert(state.custom_renderer_redraw_pending); // conceal also invalidates
 state.custom_renderer_redraw_pending=false;state.custom_renderer_frame_active=true;draw();assert(!state.custom_renderer_redraw_pending);
 state.current_config.enable_custom_rendering=false;draw();assert(native==1);
}
''')
