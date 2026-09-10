"""Execute the injected effect adapter against a recording Windows setter."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class EffectConfigTests(unittest.TestCase):
    def test_production_defaults_and_explicit_pressure_controls(self):
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include <map>
#include <string>
#include "Renderer/native/navigation_options.h"
std::map<std::string,std::string> values;
unsigned reader(char const* name,char* buffer,unsigned capacity) {
 auto it=values.find(name);if(it==values.end())return 0;
 auto size=unsigned(it->second.size());
 if(size>=capacity)return size+1;
 std::memcpy(buffer,it->second.c_str(),size+1);return size;
}
int main(){
 using O=c3x_renderer::NavigationOptions;
 char const* options[]={"C3X_RENDERER_THREE_ZOOM_MEMORY","C3X_RENDERER_WORLD_BACKDROPS",
  "C3X_RENDERER_WORLD_WAVES","C3X_RENDERER_BACKDROP_DEPENDENCIES","C3X_RENDERER_COMPOSITION_RECEIVER_INDEX"};
 for(auto cache:{"0","1"}) {
  values.clear();values["C3X_RENDERER_WORLD_REGIONS"]=cache;
  bool on=cache[0]=='1';
  for(auto name:options) {
   assert(O::retained(reader,name)==on);
   for(auto override_value:{"0","1","invalid-value-over-capacity"}) {
    values[name]=override_value;
    assert(O::retained(reader,name)==(std::strcmp(override_value,"1")==0));
   }
   values.erase(name);assert(O::retained(reader,name)==on);
  }
  assert(O::unit_pose_mib(reader)==(on?512u:8u));
  for(auto setting:{"0","1","512","invalid-value-over-capacity"}) {
   values["C3X_RENDERER_UNIT_POSE_MEMORY"]=setting;
   assert(O::unit_pose_mib(reader)==(std::strcmp(setting,"512")==0?512u:std::strcmp(setting,"1")==0?256u:8u));
  }
 }
 values.clear();assert(O::unit_pose_mib(reader)==8u);
 for(auto name:options)assert(!O::retained(reader,name));
}
''')

    def test_injected_values_and_failure_propagation(self):
        source = (ROOT / 'injected_code.c').read_text()
        start = source.index('bool\nconfigure_custom_renderer_effects ')
        end = source.index('bool\nensure_custom_renderer_loaded', start)
        run_cpp(r'''
#include <cassert>
#include <map>
#include <string>
#define WINAPI
using BOOL=int;
struct Config { bool enable_custom_rendering_cache, enable_custom_rendering_reflections, enable_custom_rendering_waves; };
struct State { Config current_config; } state;
State *is=&state;
std::map<std::string,std::string> values;
int calls=0,fail=0;
BOOL setter(char const *k,char const *v){if(++calls==fail)return 0;values[k]=v;return 1;}
''' + source[start:end] + r'''
int main(){
 for(int mask=0;mask<8;++mask){
  state.current_config={bool(mask&1),bool(mask&2),bool(mask&4)};
  values.clear();calls=0;fail=0;
  values["C3X_RENDERER_REFLECTION_CONTROL"]="stale";
  values["C3X_RENDERER_WAVES"]="stale";
  assert(configure_custom_renderer_effects(setter));
  assert(values["C3X_RENDERER_REFLECTION_CONTROL"]==((mask&2)?"0":"1"));
  assert(values["C3X_RENDERER_WAVES"]==((mask&4)?"1":"0"));
  assert(values["C3X_RENDERER_WORLD_REGIONS"]==((mask&1)?"1":"0"));
  assert(values["C3X_RENDERER_REGION_INPUT_RING"]==((mask&1)?"4":"2"));
  if(mask&1)assert(values["C3X_RENDERER_VISUAL_PROFILE"]=="city-fidelity");
  int count=calls;
  for(int i=1;i<=count;++i){calls=0;fail=i;assert(!configure_custom_renderer_effects(setter));assert(calls==i);}
 }
 assert(!configure_custom_renderer_effects(nullptr));
}
''')
        loader = source[end:source.index('is->custom_renderer_init_state = IS_OK;', end)]
        self.assertLess(loader.index('configure_custom_renderer_effects (set_environment)'), loader.index('LoadLibraryA (path)'))
        for name, default in [('reflections', 'true'), ('waves', 'true'), ('cache', 'false')]:
            self.assertIn(f'{{"enable_custom_rendering_{name}", {default}, offsetof', source)
            self.assertIn(f'enable_custom_rendering_{name} = {default}', (ROOT / 'default.c3x_config.ini').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
