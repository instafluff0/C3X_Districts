"""Unit types load on demand instead of the whole catalogue at game load.

Loading used to pin all 78 unit types (2,648 meshes, 317 textures): 1.25 GB
of process memory and about 0.6 GB of GPU memory whatever was on the map
(performance review, section 20). Loading now pins none; each camera job
offers every captured unit's missing assets to idle workers, so a type is
usually decoded before its unit enters the view, and frames still wait only
for the units they draw.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class UnitAssetsOnDemandTests(unittest.TestCase):
    def test_loading_pins_no_catalogue_and_jobs_warm_captured_units(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        start = source.index("    bool prepare_loading_sources(")
        loading = source[start:source.index("    // One GPU owner", start)]
        self.assertTrue("renderer_state.prepare_known_unit_sources({},false,cpu_allowance,gpu_allowance,cancel);" in loading)
        # The catalogue preload remains only behind the A/B switch.
        self.assertTrue("assets=catalogue?renderer_state.prepare_known_unit_sources({},true," in loading)
        job = source[source.index("            auto candidates=fresh_unit_poses;"):source.index("            LARGE_INTEGER draw_start={},draw_end={};")]
        self.assertLess(job.index("warm_unit_assets(candidates);"), job.index("prepare_frame_unit_assets(fresh_unit_poses);"))
        warm = method(source, "    void warm_unit_assets(")
        run_cpp(r'''
#include <cassert>
#include <climits>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace c3x_renderer {
struct UnitAssetInput {std::string path;bool mesh=false,move=false,bounds=false;};
namespace render_core {struct UnitInstances {struct ScenePose {std::size_t unit=0,action=0;};};}
}
struct Preparation {
 struct Job {std::uint64_t key=0;c3x_renderer::UnitAssetInput input;};
 std::vector<Job> offered;std::vector<std::size_t> limits;
 bool offer(Job job,std::size_t limit,bool urgent=false){assert(!urgent);offered.push_back(job);limits.push_back(limit);return true;}
};
struct Bodies {
 struct Mesh {std::shared_ptr<int> animation;std::string path;bool failed=false;};
 struct Texture {int* view=nullptr;std::string path;bool failed=false;};
 struct Part {unsigned mesh=0,texture=0;unsigned material_textures[4]={UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX};};
 struct Action {std::string name;std::vector<Part> parts;};struct Unit {std::vector<Action> actions;};
 std::vector<Mesh> meshes;std::vector<Texture> textures;std::vector<Unit> units;
};
struct RendererState {
 Bodies unit_bodies;Preparation unit_asset_preparation;
''' + warm + r'''
};
int main(){
 using Pose=c3x_renderer::render_core::UnitInstances::ScenePose;
 RendererState h;auto& b=h.unit_bodies;int loaded=0;
 b.meshes={{nullptr,"m0"},{std::make_shared<int>(1),"m1"},{nullptr,"m2",true},{nullptr,"m3"}};
 b.textures={{nullptr,"t0"},{&loaded,"t1"},{nullptr,"t2"},{nullptr,"t3",true}};
 Bodies::Part idle;idle.mesh=0;idle.texture=1;idle.material_textures[0]=2;
 Bodies::Part resident;resident.mesh=1;resident.texture=0;
 Bodies::Part failed;failed.mesh=2;failed.texture=3;
 Bodies::Part walk;walk.mesh=3;walk.texture=0;
 b.units={{{{"idle",{idle,resident}},{"move",{walk}}}},{{{"idle",{failed}}}}};
 // Two captured units: unit 0 walking and idle, unit 1 (failed assets),
 // plus an out-of-range pose that must be ignored.
 h.warm_unit_assets({{0,0},{0,1},{1,0},{7,0},{0,9}});
 std::vector<std::uint64_t> keys;for(auto const& job:h.unit_asset_preparation.offered)keys.push_back(job.key);
 // Missing mesh 0 (idle) and 3 (move); missing textures 0 and 2. Resident
 // mesh 1 and texture 1, and failed mesh 2 and texture 3, are not offered.
 assert((keys==std::vector<std::uint64_t>{0*2,2*2+1,0*2+1,3*2,0*2+1}));
 auto const& offered=h.unit_asset_preparation.offered;
 assert(offered[0].input.mesh && !offered[0].input.move && offered[0].input.path=="m0" && !offered[0].input.bounds);
 assert(!offered[1].input.mesh && offered[1].input.path=="t2" && !offered[2].input.mesh && offered[2].input.path=="t0");
 assert(offered[3].input.mesh && offered[3].input.move && offered[3].input.path=="m3");
 for(auto limit:h.unit_asset_preparation.limits)assert(limit==64);
 return 0;
}
''')


if __name__ == '__main__':
    unittest.main()
