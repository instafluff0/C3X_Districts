"""Bounded catalogue-source admission without a pose or native visibility product."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class UnitSourcePreparationTests(unittest.TestCase):
    def test_absolute_owner_allowance_retains_survivors_without_double_charge(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_source_plan.h"
#include <cassert>
#include <limits>
int main(){using Plan=c3x_renderer::render_core::UnitSourcePlan;
 constexpr std::size_t mib=1024u*1024u;
 assert(Plan::owner_allowance(700*mib,0,1536*mib)==700*mib);
 assert(Plan::owner_allowance(700*mib,900*mib,1536*mib)==1536*mib);
 assert(Plan::owner_allowance(100*mib,900*mib,1536*mib)==1000*mib);
 assert(Plan::owner_allowance(0,900*mib,1536*mib)==900*mib);
 assert(Plan::owner_allowance(0,2000*mib,1536*mib)==1536*mib); // Existing oversized owner still refuses admission.
 assert(Plan::owner_allowance(std::numeric_limits<std::uint64_t>::max(),900*mib,1536*mib)==1536*mib);
 assert(Plan::owner_allowance(0,450*mib,768*mib)==450*mib);
 assert(Plan::owner_allowance(200*mib,450*mib,768*mib)==650*mib);
}
''')
    def test_exact_catalogue_deduplication_and_invalid_metadata_refusal(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_source_plan.h"
#include <cassert>
#include <string>
struct Catalogue {
 struct Part {unsigned mesh=0,texture=0,material_textures[4]={UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX};};
 struct Action {std::string name;std::vector<Part> parts;};struct Unit {std::vector<Action> actions;};
 std::vector<int> meshes,textures;std::vector<Unit> units;
};
int main(){using c3x_renderer::render_core::UnitSourcePlan;
 Catalogue c;c.meshes.resize(3);c.textures.resize(4);c.units.resize(2);
 Catalogue::Part p;p.material_textures[0]=3;Catalogue::Part q;q.mesh=1;q.texture=1;
 c.units[0].actions={{"idle",{p}},{"move",{p,q}}};c.units[1].actions={{"attack",{q}}};
 UnitSourcePlan plan;assert(plan.prepare(c,{0,0},false));
 assert(plan.types==std::vector<std::size_t>{0} && plan.actions.size()==2);
 assert(plan.meshes==std::vector<bool>({true,true,false}));
 assert(plan.textures==std::vector<bool>({true,true,false,true}) && plan.mixed_motion[0] && !plan.mixed_motion[1]);
 assert(plan.prepare(c,{},true) && plan.types.size()==2 && plan.actions.size()==3 && plan.mixed_motion[1]);
 auto retained=plan.types;assert(!plan.prepare(c,{2},false) && plan.types==retained);
 c.units[0].actions[0].parts[0].material_textures[3]=4;assert(!plan.prepare(c,{0},false));
 c.units[0].actions[0].parts[0].material_textures[3]=UINT32_MAX;
 c.units[0].actions[0].parts.clear();assert(!plan.prepare(c,{0},false));
 assert(plan.prepare(c,{},false) && plan.actions.empty());
 c.meshes.resize(UnitSourcePlan::resource_limit);assert(!plan.prepare(c,{},true));
 c.meshes.resize(3);c.units.resize(UnitSourcePlan::type_limit+1);assert(!plan.prepare(c,{},true));
 c.units.resize(1);c.units[0].actions.assign(UnitSourcePlan::action_limit+1,{"idle",{p}});
 assert(!plan.prepare(c,{},true));
}
''')

    def test_actual_gpu_source_budget_preserves_pins_and_checks_warm_residency(self):
        source = (ROOT / "Renderer/sandbox/direct_units.h").read_text()
        reserve = method(source, "    bool reserve_mesh_bytes(")
        run_cpp(r'''
#include <vector>
#include <cassert>
#include <cstdint>
#include <climits>
struct Owner {
 struct Mesh {std::size_t bytes=0;std::uint64_t used=0;};
 struct Renderer {struct Bodies {struct Source {bool source_pinned=false;};
  std::vector<Source> meshes;std::size_t source_gpu_limit=192u*1024u*1024u;}unit_bodies;
  std::vector<bool> frame_mesh_leases;}renderer;
 std::vector<Mesh> meshes;std::size_t mesh_bytes=0;unsigned mesh_evictions=0;
''' + reserve + r'''
};
int main(){Owner o;o.renderer.unit_bodies.meshes.resize(3);
 o.renderer.unit_bodies.meshes[0].source_pinned=true;
 o.renderer.frame_mesh_leases={false,true,false};o.meshes={{80,0},{70,1},{40,2}};o.mesh_bytes=190;
 o.renderer.unit_bodies.source_gpu_limit=170;assert(o.reserve_mesh_bytes(0));
 assert(o.mesh_bytes==150 && o.meshes[0].bytes==80 && o.meshes[1].bytes==70 && !o.meshes[2].bytes);
 o.renderer.unit_bodies.source_gpu_limit=149;assert(!o.reserve_mesh_bytes(0) && o.mesh_bytes==150);
 o.renderer.frame_mesh_leases[1]=false;assert(o.reserve_mesh_bytes(0) && o.mesh_bytes==80);
 assert(!o.reserve_mesh_bytes(70));
 o.renderer.unit_bodies.source_gpu_limit=300;assert(o.reserve_mesh_bytes(200));
 assert(o.meshes[0].bytes==80 && o.mesh_evictions==2);
}
''')

    def test_worker_certifies_authored_bounds_before_move_stripping(self):
        fixture = (ROOT / "Renderer/native/test_animation_runtime.cpp").read_text().split("void append_u32(", 1)[1].split("\nint main(", 1)[0]
        run_cpp(r'''
#include "Renderer/native/unit_asset_content.h"
#include <cassert>
void append_u32(''' + fixture + r'''
int main(){using namespace c3x_renderer;auto data=fixture();
 float travel=1;std::uint32_t bits;std::memcpy(&bits,&travel,4);
 auto end_translation=data.size()-16;replace_u32(data,end_translation,bits);
 std::atomic<bool> cancel{false};unsigned reads=0;
 auto read=[&](char const*,std::vector<std::uint8_t>& bytes){++reads;bytes=data;return true;};
 auto raw=compile_unit_asset({"clip",true,false,true},cancel,read);
 auto move=compile_unit_asset({"clip",true,true,true},cancel,read);
 assert(raw && move && !raw->failed && !move->failed && raw->bounds.known && move->bounds.known);
 assert(raw->move_variant_differs && !raw->move_stripped && move->move_variant_differs && move->move_stripped);
 assert(raw->mesh->palettes[28]==1 && move->mesh->palettes[28]==0);
 assert(raw->bounds.baked_radius==move->bounds.baked_radius && raw->bounds.weight_sum==move->bounds.weight_sum);
 assert(raw->bytes()>=UnitAssetContent::mesh_bytes(*raw->mesh)+raw->bounds.bytes());
 auto normal=compile_unit_asset({"clip",true,true,false},cancel,read);
 assert(normal && !normal->failed && !normal->bounds.known && normal->mesh->palettes[28]==0);
 cancel=true;assert(!compile_unit_asset({"unused",true,false,true},cancel,read) && reads==3);
}
''')


if __name__ == "__main__":
    unittest.main()
