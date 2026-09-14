"""Execute the native adapter's prospective zoom capture without game state mutation."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ZoomPreparationTests(unittest.TestCase):
    def test_native_fixed_point_anchors_and_admission(self):
        source = Path('injected_code.c').read_text()
        inverse = source[source.index('int\ncustom_renderer_zoom_inverse_coordinate'):source.index('bool\ncustom_renderer_zoom_transform_active')]
        prepare = source[source.index('void\nprepare_custom_renderer_zoom_views'):source.index('bool\ncomposite_custom_renderer_frame')]
        prepare = prepare.replace('tiles = malloc (', 'tiles = (c3x_renderer_tile_v1*)malloc (')
        run_cpp(r'''
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
#define ARRAY_LEN(x) (sizeof(x)/sizeof((x)[0]))
struct State {
 int custom_renderer_zoom_native_tile_width=128,custom_renderer_zoom_tile_width=128;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
 c3x_renderer_prepare_view_fn custom_renderer_prepare_view=nullptr;
} storage,*is=&storage;
struct Game {int ScreenWidth=1119,ScreenHeight=900;} game,*p_bic_data=&game;
bool enabled=true;bool custom_renderer_zoom_enabled(){return enabled;}
std::vector<c3x_renderer_frame_v1> frames;
std::vector<std::vector<c3x_renderer_tile_v1>> snapshots;
bool admitted=false;
int receive(c3x_renderer_camera_request_v1 const* request,int query){
 if(query)return admitted?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_PENDING;
 frames.push_back(*request->frame);
 snapshots.emplace_back(request->frame->tiles,request->frame->tiles+request->frame->tile_count);
 return C3X_RENDERER_RESULT_OK;
}
''' + inverse + prepare + r'''
int projected(int raw,int width,int native,long long translation){
 long long value=(long long)raw*width*65536/native+translation;
 return int((value+(value>=0?32768:-32768))/65536);
}
int main(){
 is->custom_renderer_prepare_view=receive;
 for(int native:{64,128})for(int current:{128,160,192}){
  is->custom_renderer_zoom_native_tile_width=native;is->custom_renderer_zoom_tile_width=current;
  is->custom_renderer_zoom_translate_x_fp=-175*65536LL+12345;
  is->custom_renderer_zoom_translate_y_fp=37*65536LL+456;
  std::vector<c3x_renderer_tile_v1> tiles;
  for(int raw=-600;raw<=1600;raw+=7){c3x_renderer_tile_v1 tile={};tile.tile_x=raw;
   tile.anchor_x=projected(raw,current,native,is->custom_renderer_zoom_translate_x_fp);
   tile.anchor_y=projected(raw,current,native,is->custom_renderer_zoom_translate_y_fp);
   tile.tile_flags=C3X_RENDERER_TILE_RENDER;tiles.push_back(tile);}
  auto original=tiles;
  c3x_renderer_frame_v1 frame={};frame.tile_width=current;frame.tile_height=current/2;
  frame.target_width=1119;frame.target_height=900;frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
  c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,{1,2,3,4}};
  frames.clear();snapshots.clear();prepare_custom_renderer_zoom_views(&request);assert(frames.size()==2);
  assert(!std::memcmp(original.data(),tiles.data(),tiles.size()*sizeof(tiles[0])));
  for(unsigned i=0;i<frames.size();++i){int width=frames[i].tile_width;
   long long cx=(game.ScreenWidth/2)*65536LL,cy=(game.ScreenHeight/2)*65536LL;
   long long tx=cx-(cx-is->custom_renderer_zoom_translate_x_fp)*width/current;
   long long ty=cy-(cy-is->custom_renderer_zoom_translate_y_fp)*width/current;
   for(auto const& tile:snapshots[i]){
    assert(tile.anchor_x==projected(tile.tile_x,width,native,tx));
    assert(tile.anchor_y==projected(tile.tile_x,width,native,ty));
   }
  }
  admitted=true;prepare_custom_renderer_zoom_views(&request);assert(frames.size()==2);admitted=false;
 }
 enabled=false;frames.clear();c3x_renderer_camera_request_v1 empty={};prepare_custom_renderer_zoom_views(&empty);assert(frames.empty());
}
''')
