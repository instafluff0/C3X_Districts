"""Shared scene assets must finish loading before a map can be admitted."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class SceneAssetLoadingTests(unittest.TestCase):
    def test_loading_failure_and_device_recovery(self):
        source = Path(__file__).with_name('c3x_renderer.cpp').read_text()
        body = source.split('    bool ensure_scene_assets() {', 1)[1].split('\n    bool ensure_dds_texture(', 1)[0]
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
using DWORD=unsigned;using c3x_renderer_i64=long long;
constexpr unsigned MAX_PATH=260;
struct LARGE_INTEGER {long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER*p){static long long now=0;p->QuadPart=++now;}
template<std::size_t N,class...Args>void sprintf_s(char(&s)[N],char const*f,Args...args){std::snprintf(s,N,f,args...);}
DWORD GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
namespace c3x_renderer {constexpr int terrain_type_count=2;}
int fail=0;std::vector<int> loaded;
bool step(int id){loaded.push_back(id);return fail!=id;}
struct Natural {bool ready=false;std::string failure="test";
 template<class R,class U>bool load(int,std::string const&,R,U,char const*){
  if(ready)return true;return ready=step(2);}};
struct Reflection {bool ready=false;
 bool ensure(int,std::string const&,char const*,unsigned){if(ready)return true;return ready=step(3);}};
struct Cities {bool ready=false;
 template<class R,class U>bool load(int,std::string const&,R,U,char const*){
  if(ready)return true;return ready=step(4);}};
struct State {
 bool cache_valid=false,environment_profile=true,fidelity_profile=true,city_profile=true;
 bool wave_attempted=false,wave_ready=false;
 int device=1,wave_views[3]={};std::string fidelity_root="packs",natural_pack_root="natural",shader_root="shaders";
 struct {bool configured=true;}terrain_textures[2];
 struct Trace {double milliseconds(long long n){return double(n);}void write(char const*,char const*,bool){}}trace;
 Natural natural;Reflection reflection;Cities cities;
 bool initialize(){return step(1);}
 bool ensure_pack_texture(int){return true;}
 bool pack_path(char const*,char const*,char*,std::size_t){return false;}
 bool read_file(char const*,std::vector<std::uint8_t>&){return false;}
 void mix_content_revision(std::vector<std::uint8_t>const&){}
 bool ensure_dds_texture(std::vector<std::uint8_t>const&,int&,bool,bool){return true;}
 unsigned read_u32(std::vector<std::uint8_t>const&,unsigned){return 0;}
 bool ensure_terrain_textures(){return step(5);}
 bool ensure_scene_assets(){
''' + body + r'''
};
int main(){
 for(int failing=1;failing<=5;++failing){
  State state;fail=failing;loaded.clear();assert(!state.ensure_scene_assets());
  assert(loaded.back()==failing);
  for(int id:loaded)assert(id<=failing); // A failed dependency cannot report ready.
 }
 State state;fail=0;loaded.clear();assert(state.ensure_scene_assets());
 assert((loaded==std::vector<int>{1,2,3,4,5}));
 loaded.clear();assert(state.ensure_scene_assets());
 assert((loaded==std::vector<int>{1,5})); // First map reuses loaded scene resources.
 state.natural.ready=state.reflection.ready=state.cities.ready=false;
 loaded.clear();assert(state.ensure_scene_assets());
 assert((loaded==std::vector<int>{1,2,3,4,5})); // Device recovery rebuilds the same set.
 State compatibility;compatibility.environment_profile=compatibility.fidelity_profile=compatibility.city_profile=false;
 loaded.clear();assert(compatibility.ensure_scene_assets());
 assert((loaded==std::vector<int>{1,5}));
}
''')
        loading = source.split('// Production definitions are loaded under the native loading bar.', 1)[1]
        loading = loading.split('\n    bool ensure_scene_assets()', 1)[0]
        self.assertIn('bool ready=ensure_scene_assets();', loading)


if __name__ == '__main__':
    unittest.main()
