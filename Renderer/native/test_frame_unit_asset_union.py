"""Execute the production completed unit-asset union proof and recovery path."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class FrameUnitAssetUnionTests(unittest.TestCase):
    def test_exact_selected_action_union_skips_warm_work_and_preserves_recovery(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        prepare = method(source, "    int prepare_frame_unit_assets(")
        reserve = method(source, "    bool reserve_frame_unit_asset(")
        fields_start = source.index("    std::vector<bool> frame_mesh_leases,frame_texture_leases;")
        fields_end = source.index("\n    // Immutable selections", fields_start)
        fields = source[fields_start:fields_end]
        reset_start = source.index("        unit_asset_preparation.clear();frame_mesh_leases.clear();frame_texture_leases.clear();")
        reset_end = source.index("        unit_bodies.reset_gpu();", reset_start) + len("        unit_bodies.reset_gpu();")
        reset = source[reset_start:reset_end]
        run_cpp(r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#define C3X_RENDERER64_FRESH 1
enum {C3X_RENDERER_RESULT_OK=0,C3X_RENDERER_RESULT_PENDING=1,C3X_RENDERER_RESULT_ERROR=-1,
 DXGI_FORMAT_BC3_UNORM_SRGB=78,DXGI_FORMAT_BC1_UNORM_SRGB=72};
unsigned begins=0;
void c3x_renderer64_begin_unit_assets(){++begins;}
struct View {void Release(){}} view;
namespace c3x_renderer {
struct AnimationMesh {std::size_t bytes=128;};
struct UnitAssetInput {std::string path;bool mesh=false,move=false;};
struct UnitAssetContent {std::shared_ptr<AnimationMesh const> mesh;std::vector<std::uint8_t> dds;bool failed=false;
 static std::size_t mesh_bytes(AnimationMesh const& value){return value.bytes;}};
struct UnitAssetPreparation {
 static constexpr std::size_t byte_limit=16u*1024u*1024u;
 struct Job {std::uint64_t key=0;UnitAssetInput input;};
 struct Statistics {std::size_t pending=0,bytes=0;};
 std::map<std::uint64_t,std::unique_ptr<UnitAssetContent>> ready;std::vector<std::uint64_t> needed;
 unsigned schedules=0,polls=0,clears=0;std::vector<Job> submitted;
 template<class Compile>void schedule(std::deque<Job> jobs,Compile,unsigned count,std::vector<std::uint64_t> keys,std::size_t budget,bool bounded){
  assert(count==4 && budget==64u*1024u*1024u && bounded);++schedules;needed=std::move(keys);submitted.assign(jobs.begin(),jobs.end());}
 std::unique_ptr<UnitAssetContent> take_ready(std::uint64_t key){++polls;auto found=ready.find(key);
  if(found==ready.end())return {};auto value=std::move(found->second);ready.erase(found);return value;}
 Statistics statistics(){return {needed.size(),0};}
 void clear(){++clears;ready.clear();needed.clear();submitted.clear();}
};
template<class Read>std::unique_ptr<UnitAssetContent> compile_unit_asset(UnitAssetInput const&,std::atomic<bool> const&,Read){return {};}
namespace render_core {struct UnitInstances {struct ScenePose {std::size_t unit=0,action=0;int cursor=0,x=0,y=0;std::uint64_t clock=0;};};}
}
struct Bodies {
 struct Mesh {std::shared_ptr<c3x_renderer::AnimationMesh const> animation;View* indices=nullptr;std::string path;
  std::size_t bytes=0;std::uint64_t used=0;bool failed=false;};
 struct Texture {std::vector<std::uint8_t> dds;View* view=nullptr;std::string path;std::size_t bytes=0;std::uint64_t used=0;bool failed=false;};
 struct Part {unsigned mesh=0,texture=0;unsigned material_textures[4]={UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX};};
 struct Action {std::string name="idle";std::vector<Part> parts;};struct Unit {std::vector<Action> actions;};
 std::vector<Mesh> meshes;std::vector<Texture> textures;std::vector<Unit> units;
 std::uint64_t catalogue_generation=1,payload_serial=0;std::size_t resident_bytes=0;
 void remember_contribution_bounds(unsigned){}
 void release(View*& value){value=nullptr;}
 void reset_gpu(){for(auto& texture:textures){texture.view=nullptr;if(texture.dds.empty()){resident_bytes-=texture.bytes;texture.bytes=0;}}}
};
struct RendererState {
 Bodies unit_bodies;c3x_renderer::UnitAssetPreparation unit_asset_preparation;unsigned device_generation=1;
 struct Trace {void write(char const*,char const*,bool){}}trace;
 unsigned uploads=0;
 static bool read_file(char const*,std::vector<std::uint8_t>&,std::size_t){return false;}
 unsigned read_u32(std::vector<std::uint8_t> const&,unsigned){return DXGI_FORMAT_BC3_UNORM_SRGB;}
 bool ensure_dds_texture(std::vector<std::uint8_t> const&,View*& output,bool,bool){output=&view;++uploads;return true;}
''' + fields + '\n' + reserve + '\n' + prepare + r'''
 void reset_assets(){
''' + reset + r'''
 }
};
int main(){
 using Pose=c3x_renderer::render_core::UnitInstances::ScenePose;
 RendererState h;auto& b=h.unit_bodies;b.meshes.resize(3);b.textures.resize(3);b.units.resize(2);
 for(auto& mesh:b.meshes){mesh.animation=std::make_shared<c3x_renderer::AnimationMesh>();mesh.bytes=128;b.resident_bytes+=128;}
 for(auto& texture:b.textures){texture.view=&view;texture.bytes=156;b.resident_bytes+=156;}
 Bodies::Part first;first.material_textures[0]=2;Bodies::Part second;second.mesh=1;second.texture=1;
 Bodies::Part alternative;alternative.mesh=2;alternative.texture=2;
 b.units[0].actions={{"idle",{first}},{"move",{alternative}}};b.units[1].actions={{"idle",{second}}};
 std::vector<Pose> poses={{0,0},{1,0}};
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK && begins==1);
 assert(h.frame_unit_asset_union_valid && h.frame_unit_asset_actions.size()==2 && h.frame_unit_asset_resources.size()==5);
 assert(h.unit_asset_union_builds==1 && h.unit_asset_schedules==1 && h.unit_asset_union_bytes<=96u*1024u);
 auto lease_mesh=h.frame_mesh_leases,lease_texture=h.frame_texture_leases;
 auto prepared_bytes=h.unit_asset_frame_bytes,metadata_bytes=h.unit_asset_union_bytes;
 for(unsigned tick=0;tick<500;++tick){std::reverse(poses.begin(),poses.end());
  for(auto& pose:poses){++pose.cursor;++pose.clock;pose.x+=17;pose.y-=31;}
  auto repeated=poses;repeated.push_back(poses.front());assert(h.prepare_frame_unit_assets(repeated)==C3X_RENDERER_RESULT_OK);
  assert(h.frame_mesh_leases==lease_mesh && h.frame_texture_leases==lease_texture && h.unit_asset_frame_bytes==prepared_bytes);
 }
 assert(begins==501 && h.unit_asset_union_builds==1 && h.unit_asset_schedules==1 && h.unit_asset_preparation.schedules==1);
 assert(h.unit_asset_union_reuses==500 && h.unit_asset_union_probes==2500 && h.unit_asset_union_bytes==metadata_bytes);
 // Different actions and selected sets rebuild their exact lease masks.
 poses={{0,1}};assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK);
 assert(h.frame_mesh_leases[2] && !h.frame_mesh_leases[0] && !h.frame_mesh_leases[1]);
 poses={{0,0},{1,0}};assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK);
 auto builds=h.unit_asset_union_builds;++b.catalogue_generation;
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK && h.unit_asset_union_builds==builds+1);
 builds=h.unit_asset_union_builds;++h.device_generation;
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK && h.unit_asset_union_builds==builds+1);
 // Lost payloads cannot pass the completed proof. Missing jobs still poll and
 // reschedule through the original bounded adoption/recovery path each turn.
 b.textures[0].view=nullptr;auto schedules=h.unit_asset_schedules;
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_PENDING && !h.frame_unit_asset_union_valid);
 assert(h.unit_asset_preparation.needed==std::vector<std::uint64_t>{1});
 auto polls=h.unit_asset_preparation.polls;
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_PENDING && h.unit_asset_schedules==schedules+2 && h.unit_asset_preparation.polls>polls);
 auto failed=std::make_unique<c3x_renderer::UnitAssetContent>();failed->failed=true;h.unit_asset_preparation.ready[1]=std::move(failed);
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_ERROR && !h.frame_unit_asset_union_valid);
 auto recovered=std::make_unique<c3x_renderer::UnitAssetContent>();recovered->dds.resize(156);h.unit_asset_preparation.ready[1]=std::move(recovered);
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK && h.uploads==1 && h.unit_asset_adoptions==1);
 schedules=h.unit_asset_schedules;assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK && h.unit_asset_schedules==schedules);
 b.meshes[0].failed=true;assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_ERROR && !h.frame_unit_asset_union_valid);
 b.meshes[0].failed=false;assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK);
 assert(h.prepare_frame_unit_assets({{99,0}})==C3X_RENDERER_RESULT_ERROR && !h.frame_unit_asset_union_valid);
 assert(h.prepare_frame_unit_assets({{0,99}})==C3X_RENDERER_RESULT_ERROR);
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK);
 // Reset invalidates masks/proof and preserves GPU-only payload recovery.
 h.reset_assets();assert(!h.frame_unit_asset_union_valid && h.frame_mesh_leases.empty() && !h.unit_asset_union_bytes);
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_PENDING && h.unit_asset_preparation.clears==1);
 for(auto key:h.unit_asset_preparation.needed){auto content=std::make_unique<c3x_renderer::UnitAssetContent>();content->dds.resize(156);
  h.unit_asset_preparation.ready[key]=std::move(content);}
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK && h.frame_unit_asset_union_valid);
 // CPU payload disappearance also invalidates the proof, and demanded
 // adoption cannot exceed the unchanged 96 MiB residency allowance.
 b.resident_bytes-=b.meshes[0].bytes;b.meshes[0].bytes=0;b.meshes[0].animation.reset();
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_PENDING);
 assert(h.unit_asset_preparation.needed==std::vector<std::uint64_t>{0});
 auto huge=std::make_shared<c3x_renderer::AnimationMesh>();huge->bytes=97u*1024u*1024u;
 auto denied=std::make_unique<c3x_renderer::UnitAssetContent>();denied->mesh=huge;h.unit_asset_preparation.ready[0]=std::move(denied);
 auto resident=b.resident_bytes;
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_ERROR && b.resident_bytes==resident && !h.frame_unit_asset_union_valid);
 auto mesh_ready=std::make_unique<c3x_renderer::UnitAssetContent>();mesh_ready->mesh=std::make_shared<c3x_renderer::AnimationMesh>();
 h.unit_asset_preparation.ready[0]=std::move(mesh_ready);
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_OK && b.meshes[0].animation && h.frame_unit_asset_union_valid);
 // Oversized identity requests keep the original admission semantics and
 // cannot publish a truncated or unbounded proof.
 std::vector<Pose> large(4097,Pose{0,0});
 assert(h.prepare_frame_unit_assets(large)==C3X_RENDERER_RESULT_OK && !h.frame_unit_asset_union_valid);
 assert(h.unit_asset_union_bytes<=96u*1024u && b.resident_bytes<=96u*1024u*1024u);
 b.units[0].actions[0].parts[0].material_textures[0]=999;++b.catalogue_generation;
 assert(h.prepare_frame_unit_assets(poses)==C3X_RENDERER_RESULT_ERROR && !h.frame_unit_asset_union_valid);
 // begin_unit_assets remains the first action even for malformed requests.
 assert(begins==h.unit_asset_turns+6);
}
''')

    def test_all_assigned_reset_boundaries_invalidate_the_completed_proof(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        self.assertEqual(source.count("frame_unit_asset_union_valid=false;std::vector<std::uint64_t>().swap(frame_unit_asset_actions);"), 2)
        self.assertIn("if(command==Command::unit){renderer_state.frame_unit_asset_union_valid=false;renderer_state.unit_bodies.reset_gpu();}", source)


if __name__ == "__main__":
    unittest.main()
