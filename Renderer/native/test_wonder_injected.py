"""Execute C3X's constructed-wonder facts (tile record, frame API 24) with fake state.

Injected code only reports a completed Wonder District's state, wonder, build progress,
facing and art era; Renderer64 draws it. A district still being built, a disabled feature
and any other tile report nothing.
"""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
ROOT = Path(__file__).resolve().parents[2]


def between(source: str, start: str, stop: str) -> str:
    begin = source.index(start)
    return source[begin:source.index(stop, begin)]


class WonderInjectedTests(unittest.TestCase):
    def test_wonder_district_facts(self):
        source = (ROOT / 'injected_code.c').read_text(encoding='latin-1')
        record = between(source, '\t\t\t// API 24: a completed Wonder District', '\n\t\t\tcapture_custom_renderer_city_body')
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include "Renderer/native/c3x_renderer_api.h"
#define WONDER_DISTRICT_ID 1
enum wonder_district_state {WDS_UNUSED=0,WDS_UNDER_CONSTRUCTION,WDS_COMPLETED,WDS_RUINED};
struct City {int cost,progress;};
struct Tile {int district=-1;bool complete=true;int Territory_OwnerID=0;};
struct Leader {int Era;};
struct wonder_district_info {wonder_district_state state;City* city;int city_id;int wonder_index;};
struct district_instance {int district_id;int built_by_civ_id;wonder_district_info wonder_info;};
struct wonder_district_config {char const* wonder_name;bool enable_img_alt_dir;};
struct Config {bool enable_districts=true,enable_wonder_districts=true;};
struct State {Config current_config;wonder_district_config* wonder_district_configs;int wonder_district_count;Tile* current_render_tile=nullptr;}state,*is=&state;
wonder_district_config configs[2]={{"The Pyramids",false},{"The Colossus",true}};
Leader leaders[4]={{0},{2},{3},{1}};
district_instance instance{};City city{400,300};int building=-1;bool alternate=false;
district_instance* get_district_instance(Tile* tile){return tile->district<0?nullptr:&instance;}
wonder_district_info* get_wonder_district_info(Tile*){return &instance.wonder_info;}
bool district_is_complete(Tile* tile,int){return tile->complete;}
bool wonder_district_tile_under_construction(Tile*,int,int,int* out){
 if(instance.wonder_info.state!=WDS_UNDER_CONSTRUCTION || building<0)return false;*out=building;return true;}
int City_get_order_cost(City* c){return c->cost;}
int City_get_order_progress(City* c){return c->progress;}
bool wonder_should_use_alternative_direction_image(int,int,int,wonder_district_config const*){
 assert(is->current_render_tile);return alternate;}
int not_above(int a,int b){return b>a?a:b;}
int not_below(int a,int b){return b<a?a:b;}
c3x_renderer_tile_v1 capture(Tile* tile,int tile_x=3,int tile_y=5){
 c3x_renderer_tile_v1 storage{},*record=&storage;
''' + record + r'''
 return storage;
}
int main(){
 is->wonder_district_configs=configs;is->wonder_district_count=2;
 Tile plain{},district{WONDER_DISTRICT_ID,true,2};
 // Not a Wonder District, or one still being built: nothing.
 assert(capture(&plain).wonder_state==C3X_RENDERER_WONDER_NONE);
 Tile unfinished{WONDER_DISTRICT_ID,false,2};instance={WONDER_DISTRICT_ID,1,{WDS_UNUSED,nullptr,-1,-1}};
 assert(capture(&unfinished).wonder_state==C3X_RENDERER_WONDER_NONE);
 // Unused: its state and the territory owner's era, no wonder.
 auto r=capture(&district);
 assert(r.wonder_state==C3X_RENDERER_WONDER_UNUSED && !r.wonder[0] && r.wonder_era==3 && r.wonder_flags==0);
 // Outside any territory, the builder's era.
 Tile wild{WONDER_DISTRICT_ID,true,-1};assert(capture(&wild).wonder_era==2);
 // Building: the wonder the reserving city builds and its progress over the cost.
 instance.wonder_info={WDS_UNDER_CONSTRUCTION,&city,7,-1};building=0;
 r=capture(&district);
 assert(r.wonder_state==C3X_RENDERER_WONDER_BUILDING && !std::strcmp(r.wonder,"The Pyramids") && r.wonder_progress==191);
 city.progress=900;assert(capture(&district).wonder_progress==255);city.progress=300;
 // A reservation whose city stopped building it reads as unused.
 building=-1;r=capture(&district);assert(r.wonder_state==C3X_RENDERER_WONDER_UNUSED && !r.wonder[0]);
 // Completed: the stored wonder; the alternate facing only where its config allows it,
 // with C3X's facing rule reading the captured tile, restored after.
 instance.wonder_info={WDS_COMPLETED,&city,7,1};alternate=true;
 r=capture(&district);
 assert(r.wonder_state==C3X_RENDERER_WONDER_COMPLETE && !std::strcmp(r.wonder,"The Colossus") &&
        r.wonder_flags==C3X_RENDERER_WONDER_ALTERNATE && r.wonder_progress==0 && is->current_render_tile==nullptr);
 instance.wonder_info.wonder_index=0;assert(capture(&district).wonder_flags==0);
 instance.wonder_info.wonder_index=5;assert(capture(&district).wonder_state==C3X_RENDERER_WONDER_UNUSED);
 // Wonder Districts off: nothing.
 instance.wonder_info.wonder_index=1;is->current_config.enable_wonder_districts=false;
 assert(capture(&district).wonder_state==C3X_RENDERER_WONDER_NONE);
}
''')


if __name__ == "__main__":
    unittest.main()
