"""World validity and spatial selection use production owners, without a GPU."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


class WorldViewSubmissionTests(unittest.TestCase):
    def test_one_frame_proof_rechecks_anchors_epochs_and_gpu_residency(self):
        source=(ROOT/"Renderer/native/c3x_renderer.cpp").read_text()
        validate="    bool tile_content_valid("+source.split("    bool tile_content_valid(",1)[1].split("    bool restore_viewport_geometry(",1)[0]
        run_cpp(r'''
#include <cassert>
#include <array>
#include <cstdint>
#include <vector>
struct c3x_renderer_tile_v1 {int anchor_x=3,anchor_y=4;};
struct Handle {int generation=1;};
struct CachedTileGeometry {
 bool shared_natural=false,world_ground=false;int source_tile_width=128;Handle natural_content;
 std::uint64_t validity_epoch=0;int validity_anchor_x=0,validity_anchor_y=0;bool validity=false;
 std::vector<std::pair<unsigned,unsigned>> appearance_dependencies{{0,1}},dependencies{{0,42}},coast_dependencies{{0,1}},world_dependencies{{0,7}};
 std::vector<std::pair<unsigned,std::array<int,2>>> anchor_dependencies{{0,{10,20}}};int river_dependencies=1;
};
struct State {
 int shadow_tile_width=128;
 std::uint64_t tile_geometry_epoch=1;
 struct Residents {CachedTileGeometry shared;bool alive=true;Residents(){shared.shared_natural=true;}
  CachedTileGeometry* resolve(Handle){return alive?&shared:nullptr;}}resident_content;
 struct Topology {struct Record{unsigned semantic=42;struct {int anchor_x=13,anchor_y=24;}occurrence;}record;
  unsigned appearance_revision(unsigned){return 1;}Record*current(unsigned){return &record;}}topology_cache;
 struct World {unsigned value=7,reads=0;unsigned node_revision(unsigned){++reads;return 1;}
  World&world(){return *this;}unsigned at(unsigned){++reads;return value;}}world_coast;
 struct Rivers {int calls=0;bool valid(int){++calls;return true;}}natural;
'''+validate+r'''
};
int main(){
 State state;CachedTileGeometry cached;c3x_renderer_tile_v1 tile;
 assert(state.tile_content_valid(cached,tile));auto reads=state.world_coast.reads;
 assert(state.tile_content_valid(cached,tile));assert(state.world_coast.reads==reads && state.natural.calls==1);
 ++tile.anchor_x;assert(!state.tile_content_valid(cached,tile));
 --tile.anchor_x;assert(state.tile_content_valid(cached,tile));assert(state.natural.calls==2);
 state.resident_content.alive=false;assert(!state.tile_content_valid(cached,tile));
 state.resident_content.alive=true;assert(state.tile_content_valid(cached,tile));
 ++state.tile_geometry_epoch;state.world_coast.value=8;assert(!state.tile_content_valid(cached,tile));
 ++state.tile_geometry_epoch;state.world_coast.value=7;assert(state.tile_content_valid(cached,tile));
 state.tile_geometry_epoch=0;reads=state.world_coast.reads;
 assert(state.tile_content_valid(cached,tile));assert(state.tile_content_valid(cached,tile));assert(state.world_coast.reads>reads);
 // A world binding follows exact native anchor ratios across projection changes.
 cached.world_ground=true;state.tile_geometry_epoch=10;state.shadow_tile_width=192;
 state.topology_cache.record.occurrence.anchor_x=18;state.topology_cache.record.occurrence.anchor_y=34;
 assert(state.tile_content_valid(cached,tile));
 ++state.tile_geometry_epoch;++state.topology_cache.record.occurrence.anchor_x;
 assert(!state.tile_content_valid(cached,tile));
}
''')

    def test_river_cell_proofs_preserve_exact_values_and_local_validity(self):
        run_cpp(r'''
#include "Renderer/lab/shared/natural/world.h"
#include <cassert>
using namespace c3x_renderer;
using fidelity::NaturalWorld;
int main(){
 render_core::WorldTopology topology;
 std::vector<std::uint32_t> bits(2048,2|(2<<8));
 auto update=[&]{topology.update({64,64,true,true},bits.data(),bits.size());};update();
 NaturalWorld world;world.fields.resize(1);world.fields[0].width=world.fields[0].height=2;
 world.fields[0].pixels={0,64,128,255};world.update_rivers(topology,1);
 NaturalWorld::CellInputs inputs;
 {
  NaturalWorld::DependencyScope scope(world,&inputs);
  auto held=world.bind_river_page(12.5,4.5);
  auto a=held.sample({12.5,4.5});assert(a.distance==1000);
  hydro::P bank;world.river_affects(12,4);world.river_page(12.5,4.5).bank_point({12.5,4.5},1,1,bank);
  assert(inputs.size()==1); // Repeated consumers share an exact empty-cell proof.
  // Holding a page remains safe across the bounded page cache's eviction.
  for(int i=0;i<30;++i)world.river_sample({i*8+.5,4.5});
  auto b=held.sample({12.5,4.5});assert(a.distance==b.distance && a.source==b.source && a.mouth==b.mouth);
 }
 assert(!world.consumer && world.river_pages.size()==16);
 NaturalWorld::CellProof proof;
 auto key=NaturalWorld::CellKey{1,0,12,4};proof.push_back({key,inputs.at(key)});
 auto epoch=world.river_epoch;assert(world.valid(proof));assert(world.river_epoch==epoch);
 // A revision alone is not a changed river query result, even in this page.
 bits[topology.index(10,3)]=2|(5<<8);update();world.update_rivers(topology,2);
 assert(world.valid(proof));
 // Add a local river: absence is observed and outer compiled reuse is rejected.
 bits[topology.index(12,4)]|=10u<<16;update();world.update_rivers(topology,3);
 assert(!world.valid(proof));
 NaturalWorld::CellInputs changed;
 {NaturalWorld::DependencyScope scope(world,&changed);world.river_sample({12.5,4.5});}
 NaturalWorld::CellProof current(changed.begin(),changed.end());assert(world.valid(current));
 // Exact values survive independent page construction and wrapped lookup.
 NaturalWorld cold;cold.fields=world.fields;cold.update_rivers(topology,3);assert(cold.valid(current));
 for(auto const& p:world.river_pages){assert(p.cells->inputs->values.size()<=4096);for(auto const& input:p.cells->inputs->values)assert(topology.at(input.first)==input.second);}
 bits[topology.index(12,4)]&=~(255u<<16);update();world.update_rivers(topology,4);
 assert(world.valid(proof) && !world.valid(current));
 world.reset_world();assert(world.river_pages.empty() && !world.valid(proof));
}
''')

    def test_ground_and_cliff_private_scratches_never_share_consumer_state(self):
        """Milestone 1.2: ground and cliff generation each bind their own
        source_fidelity/surface_query_scratch.h SurfaceQueryScratch (ground_
        query_scratch/cliff_query_scratch in c3x_renderer.cpp) instead of the
        shared per-tile queries/pickup_surface/natural. This proves the exact
        mechanism that makes that safe: two independently-bound scratches'
        NaturalWorld river state never share DependencyScope consumer even
        when interleaved, a proof captured through one instance still
        validates through a completely different NaturalWorld sharing the
        same field/revision (what the ground call site's merge of private
        river_dependencies into the shared, natural-validated map relies on),
        and mutating one scratch's private ExactPointCaches never reaches
        another scratch instance."""
        run_cpp(r'''
#include "Renderer/native/source_fidelity/surface_query_scratch.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 render_core::WorldTopology topology;
 std::vector<std::uint32_t> bits(2048,2|(2<<8));
 auto update=[&]{topology.update({64,64,true,true},bits.data(),bits.size());};update();
 fidelity::NaturalWorld shared_data;
 shared_data.fields.resize(1);shared_data.fields[0].width=shared_data.fields[0].height=2;
 shared_data.fields[0].pixels={0,64,128,255};
 shared_data.update_rivers(topology,1);
 // Ground and cliffs each bind a private scratch to the same shared payload,
 // exactly as c3x_renderer.cpp's ground_query_scratch/cliff_query_scratch do.
 fidelity::SurfaceQueryScratch ground_scratch,cliff_scratch;
 ground_scratch.bind(shared_data,topology,1);
 cliff_scratch.bind(shared_data,topology,1);
 fidelity::NaturalWorld::CellInputs ground_dependencies,cliff_dependencies;
 {
  // Interleave both instances' calls the way two tiles' ground/cliff
  // compiles would if ever scheduled concurrently -- a real single-thread
  // interleaving is enough to prove state does not leak between the two
  // owning objects, since consumer/last_cell live on each NaturalWorld, not
  // on any shared/global/thread-local storage.
  fidelity::NaturalWorld::DependencyScope ground_scope(ground_scratch.rivers,&ground_dependencies);
  fidelity::NaturalWorld::DependencyScope cliff_scope(cliff_scratch.rivers,&cliff_dependencies);
  auto ground_sample=ground_scratch.rivers.river_sample({12.5,4.5});
  auto cliff_sample=cliff_scratch.rivers.river_sample({20.5,4.5});
  ground_scratch.rivers.river_sample({12.5,4.5}); // repeat; must not grow either proof again
  cliff_scratch.rivers.river_sample({20.5,4.5});
  assert(ground_dependencies.size()==1 && cliff_dependencies.size()==1);
  assert(ground_dependencies.begin()->first!=cliff_dependencies.begin()->first);
  assert(ground_sample.distance==1000 && cliff_sample.distance==1000);
 }
 assert(!ground_scratch.rivers.consumer && !cliff_scratch.rivers.consumer);
 // A proof captured via one private scratch still validates through the
 // shared natural object and through a completely different NaturalWorld
 // sharing the same field/revision -- exactly what merging ground's private
 // river_dependencies into the shared river_dependencies map at the ground
 // call site in c3x_renderer.cpp relies on for correct cache invalidation.
 fidelity::NaturalWorld::CellProof ground_proof(ground_dependencies.begin(),ground_dependencies.end());
 assert(shared_data.valid(ground_proof));
 fidelity::NaturalWorld independent_reader;
 independent_reader.fields=shared_data.fields;independent_reader.update_rivers(topology,1);
 assert(independent_reader.valid(ground_proof));
 // Mutating one scratch's private ExactPointCaches/reset never reaches
 // another scratch instance's cached content.
 ground_scratch.pickup_ground_samples.get(0.f,0.f,[]{return render_core::GroundSample{};});
 assert(ground_scratch.pickup_ground_samples.misses==1);
 cliff_scratch.reset_tile();
 assert(ground_scratch.pickup_ground_samples.misses==1);
}
''')

    def test_shared_patch_topology_and_explicit_detail_policy(self):
        run_cpp(r'''
#include "Renderer/lab/shared/natural/ground.h"
#include <cassert>
#include <cstring>
using namespace c3x_renderer::fidelity;
int main(){
 PatchLayouts layouts;
 for(unsigned n:{8u,16u,32u,48u,64u}){
  auto const& layout=layouts.get(n);assert(&layout==&layouts.get(n));
  assert(layout.corners.size()==(n+1)*(n+1) && layout.indices.size()==6*n*n);
  std::vector<MapVertex> grid((n+1)*(n+1));
  for(unsigned i=0;i<grid.size();++i){grid[i].x=float(i);grid[i].base_terrain=-9;}
  for(bool partial:{false,true}){
   if(partial)for(unsigned i=0;i<grid.size()/2;++i)grid[i].base_terrain=-10;
   std::vector<MapVertex> a,b;std::vector<unsigned> ia,ib;
   append_surface_grid(a,grid,n,true,&ia);
   append_surface_grid(b,grid,n,true,&ib,&layout);
   assert(ia==ib && a.size()==b.size());assert(!std::memcmp(a.data(),b.data(),a.size()*sizeof(MapVertex)));
  }
 }
 assert(PatchDetail(128,0)==PatchDetail());
 assert(PatchDetail(128,3).mountain==32 && PatchDetail(192,3).mountain==64);
 assert(PatchDetail(128,5).mountain==16);
 bool failed=false;try{layouts.get(65);}catch(std::length_error const&){failed=true;}assert(failed);
}
''')

    def test_production_batches_compatible_layers_and_splits_only_at_capacity(self):
        source=(ROOT/"Renderer/native/c3x_renderer.cpp").read_text()
        submit="    bool submit_scene_pass("+source.split("    bool submit_scene_pass(",1)[1].split("    std::vector<unsigned> water_scene_order()",1)[0]
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <chrono>
#include <set>
#include "Renderer/native/render_core/geometry_draws.h"
#include "Renderer/native/render_core/scene_surface.h"
#include "Renderer/native/render_core/region_contributor_index.h"
using LONG=int;
struct D3D11_RECT {int left,top,right,bottom;};
struct ViewportShaderSettings {float translation[2]={};};
namespace c3x_renderer {namespace render_core {
struct SourceShadow {
 struct Caster{};struct PreparedCasters{};using Bounds=std::set<std::pair<int,int>>;
 static Bounds required_pages(std::vector<Bounds> const& inputs,int){Bounds out;for(auto const& in:inputs)out.insert(in.begin(),in.end());return out;}
};
struct LinearTarget {int target=0,depth=0;};
}namespace city_fidelity {struct Glow{};}}
struct Chunk {D3D11_RECT bounds={0,0,128,128};int translation_x=0,translation_y=0;float natural_projection[4]={};int id=0;
 c3x_renderer::render_core::SourceShadow::Bounds world_bounds;};
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<Chunk,4>;
using GeometryDrawRecord=GeometryDrawView::Record;
using Shadow=c3x_renderer::render_core::SourceShadow;
struct State {
 int region_origin_x=0,region_origin_y=0,shadow_basis=0,geometry_shadow=3;bool retained_world=false;
 bool water_scene_active=false;
 GeometryDrawView::Records geometry_vertex_buffers;
 c3x_renderer::render_core::RegionContributorIndex region_contributors;
 double frame_scene_execute_ms=0,frame_scene_select_ms=0;
 std::vector<int> issued;std::vector<std::size_t> page_counts;unsigned submissions=0;
 Shadow::PreparedCasters* prepare_shadow_submission(GeometryDrawView,std::vector<Shadow::Caster>&,Shadow::PreparedCasters& p){return &p;}
 bool query_region_inputs(int,int,int,int,int,std::vector<c3x_renderer::render_core::RegionContributorIndex::Item>&){return false;}
 bool chunk_intersects_region(GeometryDrawView::Reference item,ViewportShaderSettings const&,D3D11_RECT const&,bool){return item.content().id>=0;}
 template<class T> bool prepare_receiver_shadows(GeometryDrawView,ViewportShaderSettings const&,std::vector<D3D11_RECT> const&,bool,std::vector<Shadow::Caster> const&,T*,std::nullptr_t,Shadow::Bounds const* pages){page_counts.push_back(pages->size());return true;}
 template<class... T> bool submit_geometry(GeometryDrawView::Records const& selected,T const&...){
  ++submissions;for(auto const& layer:selected)for(auto const& item:layer)issued.push_back(item.content().id);return true;}
 bool submit_prepared_resource_region(GeometryDrawView::Records const& selected,ViewportShaderSettings const&,D3D11_RECT const&,c3x_renderer::city_fidelity::Glow*){return submit_geometry(selected);}
'''+submit+r'''
};
int main(){
 GeometryDrawView::Chunks inputs;
 for(int n=0;n<16;++n){Chunk c;c.id=n;c.world_bounds={{n,0}};inputs[0].push_back(c);}
 Chunk shared;shared.id=16;shared.world_bounds={{0,0}};inputs[1].push_back(shared);
 for(int n=16;n<33;++n){Chunk c;c.id=n+1;c.world_bounds={{n,0}};inputs[2].push_back(c);}
 Chunk shadow;shadow.id=34;inputs[3].push_back(shadow);Chunk hidden;hidden.id=-1;inputs[1].push_back(hidden);
 State state;ViewportShaderSettings settings;c3x_renderer::render_core::LinearTarget target;c3x_renderer::city_fidelity::Glow glow;
 auto run=[&](bool dynamic){unsigned batches=0,selected=0,animated=0,candidates=0,scans=0;
  bool ok=state.submit_scene_pass(inputs,{0,1,2,3},dynamic,target,glow,settings,128,128,{{0,0,128,128}},batches,selected,animated,candidates,scans);
  assert(ok && batches==2 && state.submissions==2 && (dynamic?animated:selected)==35);
  assert(state.page_counts==std::vector<std::size_t>({32,1}));
  for(int i=0;i<35;++i)assert(state.issued[i]==i);
 };
 run(false);state=State{};run(true);
 // A pass containing only nonreceivers still executes exactly once.
 for(auto& layer:inputs)layer.clear();inputs[3].push_back(shadow);state=State{};
 unsigned a=0,b=0,c=0,d=0,e=0;
 assert(state.submit_scene_pass(inputs,{3},false,target,glow,settings,128,128,{{0,0,128,128}},a,b,c,d,e));
 assert(a==1 && state.issued==std::vector<int>{34} && state.page_counts.empty());
}
''')

    def test_selected_pass_membership_borrows_exact_occurrences(self):
        run_cpp(r'''
#include "Renderer/native/render_core/geometry_draws.h"
#include <cassert>
struct Chunk {std::array<int,4> bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};int mesh=0;};
using View=c3x_renderer::render_core::GeometryDrawView<Chunk,4>;
int main(){
 assert(!View{}.pass().any());View::Chunks owners;owners[2].push_back({});owners[2][0].mesh=7;
 View owned(owners);assert(owned.pass().count()==1 && owned.pass().has(2));
 View::Records selected;auto record=View::Record(owners[2][0]);record.translation_x=128;
 selected[2].push_back(record);record.translation_x=-128;selected[2].push_back(record);
 View view(selected);assert(view.pass().count()==1 && !view.pass().has(0));
 assert(&view[2][0].content()==&owners[2][0] && &view[2][1].content()==&owners[2][0]);
 assert(view[2][0].translation_x()==128 && view[2][1].translation_x()==-128);
 // Rebuild selection within its lease. Membership follows content, never a
 // stale retained visibility flag; wrapped occurrences preserve native order.
 selected[0].push_back(record);assert(view.pass().count()==2);
 selected[2].clear();assert(view.pass().count()==1 && !view.pass().has(2));
 selected[0].clear();assert(!view.pass().any());owners[2].clear();
}
''')

    def test_rectangular_pass_selection_preserves_order_and_all_overhangs(self):
        run_cpp(r'''
#include "Renderer/native/render_core/region_contributor_index.h"
#include <cassert>
using Index=c3x_renderer::render_core::RegionContributorIndex;
int main(){
 Index index;std::vector<std::array<int,4>> bounds;
 for(unsigned i=0;i<600;++i){int x=int(i%40)*64-500,y=int(i/40)*80-300;
  bounds.push_back({x,y,x+180,y+240});assert(index.add(0,{i%8,i},x,y,x+180,y+240));}
 index.ready=true;
 for(int x:{-540,-1,100,700})for(int y:{-340,0,450})for(int w:{1,17,256,2248})for(int h:{1,31,1200}){
  std::vector<Index::Item> selected;assert(index.query_rectangle(0,x,y,w,h,selected));
  assert(std::is_sorted(selected.begin(),selected.end()));
  assert(std::adjacent_find(selected.begin(),selected.end())==selected.end());
  std::vector<Index::Item> exact,brute;
  auto hit=[&](unsigned i){auto b=bounds[i];return b[0]<x+w && b[2]>x && b[1]<y+h && b[3]>y;};
  for(auto item:selected)if(hit(item.second))exact.push_back(item);
  for(unsigned i=0;i<bounds.size();++i)if(hit(i))brute.push_back({i%8,i});std::sort(brute.begin(),brute.end());
  assert(exact==brute); // Same material/layer and occurrence order, no duplicates.
 }
 assert(index.bytes<=index.budget);index.clear();std::vector<Index::Item> out;
 assert(!index.query_rectangle(0,0,0,2248,1200,out)); // Caller must use the complete scan.
}
''')


if __name__ == '__main__':
    unittest.main()
