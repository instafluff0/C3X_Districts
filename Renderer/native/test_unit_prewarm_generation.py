"""Prewarm survives current-frame preparation and genuine owner replacement."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method as function


STUB = r'''
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#define __declspec(x)
enum {C3X_RENDERER_RESULT_OK=0,C3X_RENDERER_RESULT_PENDING=1,C3X_RENDERER_RESULT_ERROR=2};
unsigned GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
struct Part {unsigned mesh=0,texture=0;};
struct Action {std::string name;std::vector<Part> parts;};
struct Unit {std::vector<Action> actions;};
struct SourceMesh {bool animation=false;};
struct Texture {bool view=false;};
struct Renderer {
 unsigned device_generation=1;
 struct Bodies {
  std::uint64_t catalogue_generation=1;
  std::vector<SourceMesh> meshes;
  std::vector<Texture> textures;
  std::vector<Unit> units;
 } unit_bodies;
 std::vector<bool> frame_mesh_leases;
 struct Trace {void write(char const*,char const*,bool){}} trace;
 bool prepare_unit_action(Action const& action){
  for(auto const& part:action.parts){
   unit_bodies.meshes[part.mesh].animation=true;
   unit_bodies.textures[part.texture].view=true;
  }
  return true;
 }
} renderer;
struct SandboxBackbufferOutput {} sandbox_backbuffer_output;
struct SandboxFreshPipeline {} sandbox_fresh;
struct SandboxDirectUnits {
 struct Mesh {bool vertices=false;};
 std::vector<Mesh> meshes;
 std::size_t mesh_bytes=0,mesh_peak_bytes=0;
 std::uint64_t mesh_evictions=0;
 unsigned moving_subject=0,adoptions=0;
 std::vector<int> prepared_units;
 struct Transitions {void clear(){}} transitions;
 bool initialize(){return true;}
 bool reserve_mesh_bytes(std::size_t){return true;} // Actual budget/pin contract has its own executable fixture.
 Unit const* unit_for(int subject){
  return unsigned(subject)<renderer.unit_bodies.units.size()?&renderer.unit_bodies.units[subject]:nullptr;
 }
 Action const* action_named(Unit const& unit,char const* name){
  auto found=std::find_if(unit.actions.begin(),unit.actions.end(),
    [&](auto const& action){return action.name==name;});
  return found==unit.actions.end()?nullptr:&*found;
 }
 bool prepare_mesh(unsigned index){
  if(index>=renderer.unit_bodies.meshes.size()||!renderer.unit_bodies.meshes[index].animation)return false;
  meshes.resize(renderer.unit_bodies.meshes.size());
  if(!meshes[index].vertices){meshes[index].vertices=true;++adoptions;++mesh_bytes;}
  return true;
 }
'''


class PrewarmGenerationContracts(unittest.TestCase):
    def program(self, enabled=True):
        fresh = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        resident = (ROOT / "Renderer/sandbox/resident_scene.cpp").read_text()
        direct = (ROOT / "Renderer/sandbox/direct_units.h").read_text()
        device = function(resident, "void c3x_renderer64_frame_device()")
        begin = function(resident, "void c3x_renderer64_begin_unit_assets()")
        prewarm = function(direct, "    bool prewarm(int,int)")
        adoption = function(direct, "    int prepare_frame_meshes()")
        wrapper = function(fresh, 'extern "C" __declspec(dllexport) int c3x_sandbox_prewarm_units(')
        # This is the exact synthetic draw's bounded CPU/material/GPU-ready guard.
        guard_start = direct.index("                if(part.mesh>=bodies.meshes.size()", direct.index("    template<class Target>bool draw("))
        guard_end = direct.index("                auto& gpu=meshes[part.mesh];", guard_start)
        guard = direct[guard_start:guard_end]
        code = ("#define C3X_RENDERER64_FRESH\n" if enabled else "") + STUB
        code += prewarm + "\n" + adoption + r'''
 bool ready(){
  auto& bodies=renderer.unit_bodies;
  for(auto const& unit:bodies.units)for(auto const& action:unit.actions)for(auto const& part:action.parts){
''' + guard + r'''
  }
  return true;
 }
} sandbox_direct_units;
'''
        code += device + "\n" + begin + "\n" + wrapper
        code += r'''
int main(){
 unsigned index=0;
 for(auto names:std::vector<std::vector<char const*>>{
     {"idle","move"},{"idle","attack","defend"},{"idle","road"},{"idle","attack","defend"}}){
  Unit unit;
  for(auto name:names){unit.actions.push_back(Action{name,{{index,index}}});++index;}
  renderer.unit_bodies.units.push_back(unit);
 }
 renderer.unit_bodies.meshes.resize(index);renderer.unit_bodies.textures.resize(index);
 assert(c3x_sandbox_prewarm_units(12,0)==0);
 assert(sandbox_direct_units.ready());
 unsigned uploads=sandbox_direct_units.adoptions;
 assert(uploads==index);
'''
        if not enabled:
            code += r'''
 // The ordinary non-FRESH wrapper leaves the caller's prewarm behavior intact.
 assert(c3x_sandbox_prewarm_units(12,0)==0);
 assert(sandbox_direct_units.ready() && sandbox_direct_units.adoptions==uploads);
}
'''
            return code
        code += r'''
 // No real captured units: frame preparation pins no mesh or texture.
 renderer.frame_mesh_leases.assign(index,false);
 c3x_renderer64_begin_unit_assets();
 assert(sandbox_direct_units.prepare_frame_meshes()==C3X_RENDERER_RESULT_OK);
'''
        code += r'''
 assert(sandbox_direct_units.ready());
 assert(sandbox_direct_units.adoptions==uploads);
 // Every unchanged frame retains these immutable resources without more uploads.
 for(unsigned turn=0;turn<1000;++turn){
  c3x_renderer64_begin_unit_assets();
  assert(sandbox_direct_units.prepare_frame_meshes()==C3X_RENDERER_RESULT_OK);
  assert(sandbox_direct_units.ready() && sandbox_direct_units.adoptions==uploads);
 }
 // A genuine catalogue revision must invalidate before the next prewarm adopts.
 ++renderer.unit_bodies.catalogue_generation;
 assert(c3x_sandbox_prewarm_units(12,0)==0);
 assert(sandbox_direct_units.adoptions==2*uploads);
 c3x_renderer64_begin_unit_assets();
 assert(sandbox_direct_units.ready() && sandbox_direct_units.adoptions==2*uploads);
 // A device replacement must reconstruct the owner before adoption too.
 ++renderer.device_generation;
 assert(c3x_sandbox_prewarm_units(12,0)==0);
 assert(sandbox_direct_units.ready() && sandbox_direct_units.adoptions==uploads);
 c3x_renderer64_begin_unit_assets();
 assert(sandbox_direct_units.ready() && sandbox_direct_units.adoptions==uploads);
}
'''
        return code

    def test_fresh_prewarm_survives_preparation_and_genuine_replacement(self):
        run_cpp(self.program())

    def test_non_fresh_wrapper_preserves_existing_prewarm_behavior(self):
        run_cpp(self.program(enabled=False))


if __name__ == "__main__":
    unittest.main()
