"""Execute C3X's natural-wonder facts (tile record) and its native-label delegation with fake state.

Injected code only reports the wonder's name and show-name setting and skips its own
label while Renderer64 draws it; with custom rendering off the native label is unchanged.
"""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
ROOT = Path(__file__).resolve().parents[2]


def between(source: str, start: str, stop: str) -> str:
    begin = source.index(start)
    return source[begin:source.index(stop, begin)]


class NaturalWonderInjectedTests(unittest.TestCase):
    def test_record_facts_and_config_off_label_delegation(self):
        source = (ROOT / 'injected_code.c').read_text(encoding='latin-1')
        label = between(source, '\t// Renderer64 draws natural-wonder names itself', '\tbool draw_named_tiles')
        record = between(source, '\t\t\t// API 23: the natural wonder here', '\n\t\t\t// API 24:')
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include "Renderer/native/c3x_renderer_api.h"
#define NATURAL_WONDER_DISTRICT_ID 4
struct Config {bool enable_natural_wonders=true,show_natural_wonder_name_on_map=true,enable_custom_rendering=true;};
struct State {Config current_config;void* custom_renderer_module=nullptr;}state,*is=&state;
struct Tile {int district=-1,wonder=0;};
struct natural_wonder_info_t {int natural_wonder_id;};
struct district_instance {int district_id;natural_wonder_info_t natural_wonder_info;};
struct natural_wonder_district_config {char const* name;};
district_instance instance;natural_wonder_district_config configs[2]={{"Old Faithful"},{nullptr}};
district_instance* get_district_instance(Tile* tile){
 if(tile->district<0)return nullptr;instance={tile->district,{tile->wonder}};return &instance;}
natural_wonder_district_config const* get_natural_wonder_config_by_id(int id){
 return is->current_config.enable_natural_wonders && id>=0 && id<2?&configs[id]:nullptr;}
bool label(){
''' + label + r'''
 return draw_natural_wonders;
}
c3x_renderer_tile_v1 capture(Tile* tile){
 c3x_renderer_tile_v1 storage{},*record=&storage;
''' + record + r'''
 return storage;
}
int main(){
 // Custom rendering off: the native label draws as before, DLL loaded or not.
 is->current_config.enable_custom_rendering=false;assert(label());
 int module=0;is->custom_renderer_module=&module;assert(label());
 // On with the paired DLL loaded: Renderer64 draws the names, the native draw is skipped.
 is->current_config.enable_custom_rendering=true;assert(!label());
 // On without a renderer module (declined or failed load): native.
 is->custom_renderer_module=nullptr;assert(label());
 // C3X's own settings still decide whether any name shows.
 is->current_config.show_natural_wonder_name_on_map=false;assert(!label());
 is->current_config.show_natural_wonder_name_on_map=true;is->current_config.enable_natural_wonders=false;assert(!label());
 is->current_config.enable_natural_wonders=true;
 // Facts: the wonder's config name and the show-name setting; nothing for other tiles.
 Tile wonder{NATURAL_WONDER_DISTRICT_ID,0},district{2,0},plain{},nameless{NATURAL_WONDER_DISTRICT_ID,1};
 auto r=capture(&wonder);assert(!std::strcmp(r.natural_wonder,"Old Faithful") && r.natural_wonder_flags==C3X_RENDERER_NATURAL_WONDER_NAME);
 assert(!capture(&district).natural_wonder[0] && !capture(&plain).natural_wonder[0] && !capture(&nameless).natural_wonder[0]);
 is->current_config.show_natural_wonder_name_on_map=false;r=capture(&wonder);
 assert(!std::strcmp(r.natural_wonder,"Old Faithful") && r.natural_wonder_flags==0);
 is->current_config.enable_natural_wonders=false;assert(!capture(&wonder).natural_wonder[0]);
}
''')


if __name__ == "__main__":
    unittest.main()
