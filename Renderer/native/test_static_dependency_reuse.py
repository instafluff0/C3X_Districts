"""Exact unchanged-frame proofs and independent static coverage contracts."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import GPU_STUB


def method(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


class StaticDependencyReuseTests(unittest.TestCase):
    def test_production_registration_keeps_missing_and_river_input_keys(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        proof = method(source, "struct CachedGeometryProof {") + ";"
        registration = method(source, "    template<class Inputs>bool watch_raster_dependencies(")
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include "Renderer/native/render_core/resident_content.h"
#include "Renderer/lab/shared/natural/world.h"
#include <cassert>
using namespace c3x_renderer::render_core;
using NaturalWorld=c3x_renderer::fidelity::NaturalWorld;
using Domain=RasterDependencyRevisions::Domain;
''' + proof + r'''
struct State {
''' + registration + r'''
};
struct RecordingInputs {
 unsigned calls=0,refuse=~0u;
 bool watch(Domain,std::uint64_t){return ++calls!=refuse;}
};
int main(){
 State state;auto proof=std::make_shared<CachedGeometryProof>();
 // Neither an absent value nor a zero identity may discard a dependency.
 proof->appearance_dependencies={{0,0},{101,7},{102,0}};
 proof->dependencies={{0,0},{201,9},{202,0}};
 proof->coast_dependencies={{0,0},{301,13},{302,0}};
 proof->world_dependencies={{0,0xffffffffu},{401,42},{402,0xffffffffu}};
 proof->tile=0;
 auto first=std::make_shared<NaturalWorld::CellContent>();
 first->inputs=std::make_shared<NaturalWorld::PageInputs>();
 first->inputs->values={{0,0},{501,12},{502,0xffffffffu}};
 first->inputs->flow={2,1,0};
 auto second=std::make_shared<NaturalWorld::CellContent>();
 second->inputs=std::make_shared<NaturalWorld::PageInputs>();
 second->inputs->values={{503,0}};second->inputs->flow={0};
 proof->river_dependencies={{{1,2,3,4},first},{{5,6,7,8},second},{{9,10,11,12},first}};
 RasterContributors<CachedGeometryProof> pixels;RasterContributors<CachedGeometryProof>::Key draw{};
 draw[0]=7;assert(pixels.add(draw,proof,proof->tile,0));
 assert(state.watch_raster_dependencies(*proof,pixels));pixels.finish_dependencies();
 decltype(pixels.dependencies) expected={
  {Domain::appearance,0},{Domain::appearance,101},{Domain::appearance,102},
  {Domain::semantic,0},{Domain::semantic,201},{Domain::semantic,202},
  {Domain::coast,0},{Domain::coast,301},{Domain::coast,302},
  {Domain::world,0},{Domain::world,401},{Domain::world,402},{Domain::world,501},{Domain::world,502},{Domain::world,503},
  {Domain::flow,0},{Domain::flow,501},{Domain::flow,502},{Domain::flow,503},{Domain::visibility,0}};
 assert(pixels.dependencies==expected&&pixels.complete&&pixels.dependencies_complete);
 assert(state.watch_raster_dependencies(*proof,pixels));assert(pixels.dependencies==expected);
 RasterDependencyRevisions revisions;RasterContributors<CachedGeometryProof>::ValidationKey key{};
 unsigned full=0;bool permitted=true;auto exact=[&]{++full;return permitted;};
 assert(pixels.validate(revisions,key,exact));
 for(unsigned i=0;i<1000;++i)assert(pixels.validate(revisions,key,exact));assert(full==1);
 // A different domain or unrelated hidden key cannot invalidate these pixels.
 revisions.touch(Domain::flow,401);revisions.touch(Domain::semantic,501);revisions.touch(Domain::appearance,999);
 assert(pixels.validate(revisions,key,exact));assert(full==1);
 for(auto const& dependency:expected){auto before=full;
  revisions.touch(dependency.domain,dependency.id);assert(pixels.validate(revisions,key,exact));assert(full==before+1);
 }
 // Missing-to-present mutations must reach the exact validator, which refuses
 // the stale proof. Recovering it requires another complete validation.
 for(auto dependency:{RasterDependencyRevisions::Key{Domain::appearance,102},
       {Domain::semantic,202},{Domain::coast,302},{Domain::world,402},{Domain::flow,503},{Domain::visibility,0}}){
  permitted=false;auto before=full;revisions.touch(dependency.domain,dependency.id);
  assert(!pixels.validate(revisions,key,exact)&&full==before+1);
  permitted=true;assert(pixels.validate(revisions,key,exact)&&full==before+2);
 }
 // Every rejected metadata admission propagates immediately, including the
 // last visibility key and the second (flow) registration for each river input.
 RecordingInputs accepted;assert(state.watch_raster_dependencies(*proof,accepted));
 for(unsigned at=1;at<=accepted.calls;++at){RecordingInputs refused;refused.refuse=at;
  assert(!state.watch_raster_dependencies(*proof,refused)&&refused.calls==at);
 }
 CachedGeometryProof incomplete;incomplete.river_dependencies={{{0,0,0,0},nullptr}};
 RasterContributors<CachedGeometryProof> refused;
 assert(!state.watch_raster_dependencies(incomplete,refused));
 auto empty=std::make_shared<NaturalWorld::CellContent>();incomplete.river_dependencies[0].second=empty;
 assert(!state.watch_raster_dependencies(incomplete,refused));
 // A valid empty query is distinct from a missing PageInputs owner.
 empty->inputs=std::make_shared<NaturalWorld::PageInputs>();
 assert(state.watch_raster_dependencies(incomplete,refused));
 assert(refused.dependencies.size()==1&&refused.dependencies.count({Domain::visibility,0})==1);
}
''')

    def test_exact_local_changes_missing_inputs_and_bounded_revision_window(self):
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Proof {unsigned revision=7;};
int main(){
 RasterDependencyRevisions revisions;using D=RasterDependencyRevisions::Domain;
 RasterContributors<Proof> pixels;auto proof=std::make_shared<Proof>();
 RasterContributors<Proof>::Key draw{};draw[0]=7;
 assert(pixels.add(draw,proof,55,3));
 for(auto domain:{D::appearance,D::semantic,D::visibility,D::coast,D::world,D::flow})assert(pixels.watch(domain,55));
 pixels.finish_dependencies();RasterContributors<Proof>::ValidationKey key{};
 unsigned content=7,visibility=3,membership=0;
 auto exact=[&]{++membership;return pixels.valid([&](auto const& p){return p.revision==content;},[&](auto){return visibility;});};
 assert(pixels.validate(revisions,key,exact));
 auto counts=pixels.validation_counts;
 for(unsigned i=0;i<1000;++i)assert(pixels.validate(revisions,key,exact));
 assert(membership==1&&pixels.validation_counts.content==counts.content&&pixels.validation_counts.visibility==counts.visibility);
 assert(pixels.validation_counts.reused==1000&&pixels.validation_counts.changes==0);
 // An unrelated hidden edit advances this checkpoint, without visiting a proof.
 revisions.touch(D::appearance,999);assert(pixels.validate(revisions,key,exact));assert(membership==1);
 assert(pixels.validation_counts.changes==1);
 for(auto domain:{D::appearance,D::semantic,D::visibility,D::coast,D::world,D::flow}){
  auto before=membership;revisions.touch(domain,55);assert(pixels.validate(revisions,key,exact));assert(membership==before+1);
 }
 // Absence is a watched key too; insertion must run and fail its exact proof.
 bool absent=true;assert(pixels.watch(D::semantic,77));pixels.finish_dependencies();
 auto missing=[&]{++membership;return absent&&exact();};assert(pixels.validate(revisions,key,missing));
 absent=false;revisions.touch(D::semantic,77);assert(!pixels.validate(revisions,key,missing));
 absent=true;revisions.touch(D::semantic,77);assert(pixels.validate(revisions,key,missing));
 ++visibility;revisions.touch(D::visibility,55);assert(!pixels.validate(revisions,key,exact));--visibility;
 revisions.touch(D::visibility,55);assert(pixels.validate(revisions,key,exact));
 // Every guard word is exact, including caller-defined view/light/device facts.
 for(unsigned i=0;i<key.size();++i){auto changed=key;++changed[i];auto before=membership;
  assert(pixels.validate(revisions,changed,exact));assert(membership==before+1);key=changed;
 }
 revisions.invalidate();auto before=membership;assert(pixels.validate(revisions,key,exact));assert(membership==before+1);
 for(unsigned i=0;i<RasterDependencyRevisions::capacity+1;++i)revisions.touch(D::world,999);
 before=membership;assert(pixels.validate(revisions,key,exact));assert(membership==before+1);
 RasterDependencyRevisions replacement;before=membership;assert(pixels.validate(replacement,key,exact));assert(membership==before+1);
 // A writer during full validation prevents certification for a later frame.
 auto writes=[&]{revisions.touch(D::world,999);return exact();};key[0]++;
 assert(pixels.validate(revisions,key,writes));before=membership;assert(pixels.validate(revisions,key,exact));assert(membership==before+1);
 // An incomplete or unregistered set cannot take the successful shortcut.
 pixels.dependencies_complete=false;before=membership;assert(pixels.validate(revisions,key,exact));assert(membership==before+1);
 before=membership;assert(pixels.validate(revisions,key,exact));assert(membership==before+1);
 pixels.complete=false;assert(!pixels.validate(revisions,key,exact));
}
''')

    def test_production_static_raster_runs_no_steady_proof_or_membership_visits(self):
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        methods = "\n".join(method(source, signature) for signature in [
            "    RasterInputs::ValidationKey raster_validation_key(",
            "    RasterInputs::Key contributor_key(",
            "    bool raster_dependencies(",
        ])
        run_cpp(r'''
#define C3X_RENDERER64_FRESH
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
#include <cstring>
#include <vector>
#include <map>
using c3x_renderer::render_core::RasterDependencyRevisions;
struct D3D11_RECT {long left,top,right,bottom;};
struct ViewportShaderSettings {float translation[2]={},depth_translation=0,padding=0,inverse_size[2]={1,1},reserved[2]={},natural_projection[4]={};};
struct GeometryDrawRecord {
 struct {std::uint64_t generation=9;}owner;unsigned ordinal=1;int tile_x=2,tile_y=4,translation_x=128,translation_y=64;
 float natural_projection[4]={3,-1,128,1260};struct {int left=0,top=0,right=10,bottom=20;}bounds;
 unsigned territory_rgb=17,territory_edges=3;bool water_dependent=false,water_visible=true;
};
struct GeometryDrawReference {GeometryDrawRecord const& value;explicit GeometryDrawReference(GeometryDrawRecord const& v):value(v){}};
struct CachedGeometryProof {std::uint64_t semantic=7;};
struct CachedMeshGeneration {std::shared_ptr<CachedGeometryProof> proof=std::make_shared<CachedGeometryProof>();};
struct State {
 using RasterInputs=c3x_renderer::render_core::RasterContributors<CachedGeometryProof>;
 struct Renderer {
  bool water_scene_active=true,geometry_canonical_world=true;unsigned content_revision=7,device_generation=1,scene_depth_origin=8;
  struct Topology {struct Record {unsigned visibility_revision=3;}record;
   unsigned scope_sequence()const{return 1;}std::uint64_t key(int,int)const{return 55;}
   Record const* retained(std::uint64_t)const{return &record;}}topology_cache;
  RasterDependencyRevisions raster_dependency_revisions;unsigned semantic=7,proof_visits=0;
  mutable unsigned intersections=0;unsigned raster_proof_rejections[7]={};
  bool raster_content_valid(CachedGeometryProof const& proof){++proof_visits;return semantic==proof.semantic;}
  bool chunk_intersects_region(GeometryDrawReference const&,ViewportShaderSettings const&,D3D11_RECT,bool)const{++intersections;return true;}
  template<class Inputs>bool watch_raster_dependencies(CachedGeometryProof const&,Inputs& inputs){
   return inputs.watch(RasterDependencyRevisions::Domain::semantic,77);}
 }renderer_state;Renderer& renderer=renderer_state;
 struct Content {std::shared_ptr<CachedMeshGeneration> mesh=std::make_shared<CachedMeshGeneration>();
  template<class Handle>std::shared_ptr<void> get(Handle)const{return mesh;}};
 struct Lease {Content content;};std::shared_ptr<Lease> resident_lease=std::make_shared<Lease>();
 float projection_zoom=1,resident_basis_x=0,resident_basis_y=0;int wrap_pixels=0;
 unsigned membership=1;mutable unsigned contributor_visits=0;std::vector<GeometryDrawRecord> records={GeometryDrawRecord{}};
 unsigned view_revision()const{return membership;}
 D3D11_RECT source_bounds(ViewportShaderSettings const&,D3D11_RECT rect,bool)const{return rect;}
 template<class Visit>void contributors(ViewportShaderSettings const&,D3D11_RECT,bool,Visit visit)const{
  for(auto const& record:records){++contributor_visits;visit(0,record);}}
''' + methods + r'''
};
int main(){State state;State::RasterInputs pixels;ViewportShaderSettings settings;D3D11_RECT region={0,0,48,32};
 assert(state.raster_dependencies(pixels,settings,region,true));assert(state.raster_dependencies(pixels,settings,region,false));
 auto visits=state.contributor_visits,proofs=state.renderer.proof_visits,intersections=state.renderer.intersections;
 for(unsigned i=0;i<1000;++i)assert(state.raster_dependencies(pixels,settings,region,false));
 assert(state.contributor_visits==visits&&state.renderer.proof_visits==proofs&&state.renderer.intersections==intersections);
 state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::semantic,888);
 assert(state.raster_dependencies(pixels,settings,region,false));assert(state.contributor_visits==visits);
 state.renderer.semantic=8;state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::semantic,77);
 assert(!state.raster_dependencies(pixels,settings,region,false));assert(state.renderer.proof_visits==proofs+1);
 state.renderer.semantic=7;state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::semantic,77);
 assert(state.raster_dependencies(pixels,settings,region,false));
 // A new generation in the covered view requires exact membership, even if
 // retained topology revisions did not change.
 ++state.membership;++state.records[0].owner.generation;assert(!state.raster_dependencies(pixels,settings,region,false));
 pixels.clear();assert(state.raster_dependencies(pixels,settings,region,true));assert(state.raster_dependencies(pixels,settings,region,false));
 auto dirty=[&](auto change){auto before=state.contributor_visits;change();assert(state.raster_dependencies(pixels,settings,region,false));assert(state.contributor_visits==before+1);};
 dirty([&]{settings.translation[0]=.25f;});dirty([&]{settings.depth_translation=3;});dirty([&]{settings.natural_projection[2]=64;});
 dirty([&]{state.projection_zoom=1.5f;});dirty([&]{state.wrap_pixels=4096;});dirty([&]{++state.renderer.device_generation;});
 dirty([&]{++state.renderer.content_revision;});dirty([&]{region.left=1;});dirty([&]{state.renderer.water_scene_active=false;});
 // Local visibility is separately watched, even with otherwise valid content.
 ++state.renderer.topology_cache.record.visibility_revision;
 state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::visibility,55);
 assert(!state.raster_dependencies(pixels,settings,region,false));
}
''')

    def test_production_receiver_grid_reuses_exact_static_view_and_light(self):
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        identity = method(source, "    std::array<std::uint64_t,9> receiver_identity(")
        render = method(source, "    template<class BodyInputs,class RetireCompletedPlans> bool render(")
        coverage = render[render.index("        float needed[4]"):render.index("        receiver_revision=revision;")]
        run_cpp(r'''
#include "Renderer/native/render_core/shadow_sampling_grid.h"
#include <cassert>
#include <array>
#include <vector>
#include <limits>
#include <cmath>
constexpr unsigned geometry_layer_count=2,geometry_shadow=1;
struct Shadow {struct Bounds {float low[3]={1,2,0},high[3]={2,3,4};};
 static std::array<float,4> project(Bounds const& b,float const* offset,std::array<float,12> const&){return {b.low[0]+offset[0],b.low[1]+offset[1],b.high[0]+offset[0],b.high[1]+offset[1]};}};
struct Record {Shadow::Bounds bounds;int tile_x=2,tile_y=4;Record const& content()const{return *this;}Shadow::Bounds world_bounds=bounds;};
struct GeometryDrawView {using Records=std::array<std::vector<Record>,geometry_layer_count>;};
struct State {
 using Grid=c3x_renderer::render_core::ShadowSamplingGrid;
 struct Renderer {std::array<float,12> shadow_basis={1,0,0,0,0,1,0,0,0,0,1,0};
  unsigned content_revision=7,device_generation=1;bool geometry_canonical_world=true;
  struct World {struct Dims {int width=64,height=64;bool wrap_x=true,wrap_y=true;}dims;
   World const& world()const{return *this;}Dims dimensions()const{return dims;}}world_coast;}renderer;
 Grid receiver_grid;std::array<float,4> receiver_wrap{};std::array<std::uint64_t,9> receiver_key{};
 std::array<float,12> receiver_light{};bool receiver_grid_valid=false;
 std::uint64_t receiver_visits=0,receiver_builds=0,receiver_reuses=0;
''' + identity + r'''
 bool prepare(GeometryDrawView::Records const& receivers,std::uint64_t revision=1,std::uint64_t scene=1){
''' + coverage + r'''
 return true;}
};
int main(){State state;GeometryDrawView::Records records;records[0].resize(100);
 assert(state.prepare(records));assert(state.receiver_builds==1&&state.receiver_visits==200);
 auto first=state.receiver_grid;for(unsigned i=0;i<1000;++i)assert(state.prepare(records));
 assert(state.receiver_visits==200&&state.receiver_builds==1&&state.receiver_reuses==1000);
 assert(state.receiver_grid.low==first.low&&state.receiver_grid.count==first.count);
 // Changed pose/water clocks are absent from the static receiver key.
 for(unsigned i=0;i<12;++i){state.renderer.shadow_basis[i]+=.01f;auto before=state.receiver_visits;
  assert(state.prepare(records));assert(state.receiver_visits==before+200);}
 auto dirty=[&](auto change){auto before=state.receiver_builds;change();assert(state.prepare(records));assert(state.receiver_builds==before+1);};
 dirty([&]{++state.renderer.device_generation;});dirty([&]{++state.renderer.content_revision;});
 dirty([&]{++state.renderer.world_coast.dims.width;});dirty([&]{++state.renderer.world_coast.dims.height;});
 dirty([&]{state.renderer.world_coast.dims.wrap_x=false;});dirty([&]{state.renderer.world_coast.dims.wrap_y=false;});
 dirty([&]{state.renderer.geometry_canonical_world=false;});
 auto before=state.receiver_builds;assert(state.prepare(records,2));assert(state.receiver_builds==before+1);
 before=state.receiver_builds;assert(state.prepare(records,2,2));assert(state.receiver_builds==before+1);
}
''')

    def test_production_atlas_skips_unchanged_static_casters_and_proofs(self):
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        methods = "\n".join(method(source, signature) for signature in [
            "    AtlasInputs::Key caster_key(", "    bool atlas_dependencies(",
        ])
        run_cpp(r'''
#define C3X_RENDERER64_FRESH
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
#include <cstring>
#include <vector>
using c3x_renderer::render_core::RasterDependencyRevisions;
struct Shadow {struct Caster {
 std::uint64_t content_generation=7,version=2;unsigned layer=3,binding=40,count=6,vertex_offset=0,index_offset=0,index_format=42,stride=80;
 struct Bounds {float low[3]={},high[3]={1,1,1};}bounds;float offset[3]={};
 void* vertices=nullptr;void* indices=nullptr;std::vector<int> const* instances=nullptr;
 float instance_material=40;bool rigid=false;
 };static std::array<float,4> project(Caster::Bounds const& b,float const* offset,std::array<float,12> const&){
  return {b.low[0]+offset[0],b.low[1]+offset[1],b.high[0]+offset[0],b.high[1]+offset[1]};}};
struct CachedGeometryProof {unsigned tile=55,semantic=7;};
struct CachedMeshGeneration {std::shared_ptr<CachedGeometryProof> proof=std::make_shared<CachedGeometryProof>();};
struct State {using AtlasInputs=c3x_renderer::render_core::RasterContributors<CachedGeometryProof,20>;
 struct Renderer {
  bool geometry_canonical_world=true;unsigned content_revision=7,device_generation=1,semantic=7,proof_visits=0,raster_proof_rejections[7]={};
  std::array<float,12> shadow_basis={};RasterDependencyRevisions raster_dependency_revisions;
  struct Topology {struct Record {unsigned visibility_revision=3;}record;
   unsigned scope_sequence()const{return 1;}Record const* retained(std::uint64_t)const{return &record;}}topology_cache;
  bool raster_content_valid(CachedGeometryProof const& p){++proof_visits;return p.semantic==semantic;}
  template<class Inputs>bool watch_raster_dependencies(CachedGeometryProof const& p,Inputs& inputs){
   return inputs.watch(RasterDependencyRevisions::Domain::semantic,77)&&inputs.watch(RasterDependencyRevisions::Domain::visibility,p.tile);}
 }renderer;
 struct Content {std::shared_ptr<CachedMeshGeneration> mesh=std::make_shared<CachedMeshGeneration>();
  std::shared_ptr<void> get(std::array<std::uint64_t,2>)const{return mesh;}};
 struct Lease {Content content;};std::shared_ptr<Lease> caster_lease=std::make_shared<Lease>();
 AtlasInputs atlas_inputs;std::vector<Shadow::Caster> caster_inputs={Shadow::Caster{}};
 std::uint64_t caster_signature=1;float box[4]={0,0,48,32};
''' + methods + r'''
};
int main(){State state;assert(state.atlas_dependencies(true));assert(state.atlas_dependencies(false));
 auto counts=state.atlas_inputs.validation_counts;auto proofs=state.renderer.proof_visits;
 for(unsigned i=0;i<1000;++i)assert(state.atlas_dependencies(false));
 assert(state.atlas_inputs.validation_counts.membership==counts.membership&&state.atlas_inputs.validation_counts.content==counts.content&&state.renderer.proof_visits==proofs);
 state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::world,888);assert(state.atlas_dependencies(false));assert(state.renderer.proof_visits==proofs);
 for(unsigned i=0;i<12;++i){auto before=state.atlas_inputs.validation_counts.full;++state.renderer.shadow_basis[i];assert(state.atlas_dependencies(false));assert(state.atlas_inputs.validation_counts.full==before+1);}
 state.renderer.semantic=8;state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::semantic,77);assert(!state.atlas_dependencies(false));
 state.renderer.semantic=7;state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::semantic,77);assert(state.atlas_dependencies(false));
 ++state.caster_signature;++state.caster_inputs[0].index_offset;assert(!state.atlas_dependencies(false));
 state.atlas_inputs.clear();assert(state.atlas_dependencies(true));assert(state.atlas_dependencies(false));
 ++state.renderer.topology_cache.record.visibility_revision;state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::visibility,55);assert(!state.atlas_dependencies(false));
}
''')

    def test_coverage_memo_is_weak_and_invalidated_by_requirement_admission(self):
        run_cpp(GPU_STUB + r'''
#include "Renderer/native/render_core/body_placement_requirements.h"
#include <cassert>
struct Mesh {std::array<int,4> bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};
 std::shared_ptr<std::vector<Owner::Instance> const> instances;float instance_material=1;};
int main(){Owner owner;ID3D11Device device;
 using Inputs=c3x_renderer::render_core::BodyPlacementRequirements<Mesh>;Inputs inputs;
 Mesh mesh;mesh.instances=std::make_shared<std::vector<Owner::Instance>>(2);
 c3x_renderer::render_core::GeometryDrawView<Mesh,2>::Reference draw(mesh);
 Owner::Key key{};key[0]=17;
 auto builder=owner.begin_retained(Owner::Key{});assert(builder);Owner::Range range;
 assert(owner.append(builder,key,mesh.instances.get(),mesh.instances->data(),2,mesh.natural_projection,0,0,0,mesh.instance_material,range));
 auto candidate=owner.upload(builder,&device);builder.reset();assert(candidate);
 assert(inputs.add(owner,0,draw,key));assert(inputs.covers(candidate));auto probes=inputs.coverage_probes;
 for(unsigned i=0;i<1000;++i)assert(inputs.covers(candidate));assert(inputs.coverage_probes==probes&&inputs.coverage_reuses==1000);
 // Duplicate admission does not change the exact requirement list.
 assert(inputs.add(owner,0,draw,key));assert(inputs.covers(candidate));assert(inputs.coverage_probes==probes);
 auto extra=key;extra[0]=18;assert(inputs.add(owner,0,draw,extra));assert(!inputs.covers(candidate));assert(inputs.coverage_probes>probes);
 probes=inputs.coverage_probes;assert(!inputs.covers(candidate));assert(inputs.coverage_probes==probes);
 std::weak_ptr<Owner::Generation const> weak=candidate;candidate.reset();owner.clear();assert(weak.expired());
 inputs.clear();assert(inputs.entries.empty()&&inputs.coverage_probes==0&&inputs.coverage_reuses==0);
}
''')

    def test_production_receiver_revision_tracks_actual_ordered_membership(self):
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        key = method(source, "    RasterInputs::Key contributor_key(")
        start = source.index("        bool same_receivers=true;")
        end = source.index("        visibility_scene_key=scene_key;", start)
        comparison = source[start:end]
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
#include <cstring>
#include <vector>
constexpr unsigned geometry_layer_count=2,geometry_shadow=1;
struct Proof {};
struct GeometryDrawRecord {
 struct {std::uint64_t generation=9;}owner;unsigned ordinal=1;int tile_x=2,tile_y=4,translation_x=128,translation_y=64;
 float natural_projection[4]={3,-1,128,1260};struct {int left=0,top=0,right=10,bottom=20;}bounds;
 unsigned territory_rgb=17,territory_edges=3;bool water_dependent=false,water_visible=true;
};
struct State {
 using RasterInputs=c3x_renderer::render_core::RasterContributors<Proof>;
 using Records=std::array<std::vector<GeometryDrawRecord>,geometry_layer_count>;
 Records all_visible;std::uint64_t static_receiver_revision=0;
''' + key + r'''
 void compare(Records const& prior_receivers){
''' + comparison + r'''
 }
};
int main(){State state;State::Records prior;
 state.all_visible[0].push_back({});state.compare(prior);assert(state.static_receiver_revision==1);
 prior=state.all_visible;for(unsigned i=0;i<1000;++i)state.compare(prior);assert(state.static_receiver_revision==1);
 // A full capture repeated for an unrelated visibility publication preserves
 // receiver identity; unit-only shadow content also remains separate.
 state.all_visible[1].push_back({});state.compare(prior);assert(state.static_receiver_revision==1);
 auto dirty=[&](auto change){prior=state.all_visible;change();auto before=state.static_receiver_revision;
  state.compare(prior);assert(state.static_receiver_revision==before+1);};
 dirty([&]{++state.all_visible[0][0].owner.generation;});
 dirty([&]{++state.all_visible[0][0].tile_x;});dirty([&]{++state.all_visible[0][0].translation_y;});
 dirty([&]{state.all_visible[0][0].natural_projection[2]=64;});dirty([&]{state.all_visible[0][0].water_visible=false;});
 dirty([&]{state.all_visible[0].push_back({});});dirty([&]{std::swap(state.all_visible[0][0],state.all_visible[0][1]);});
 dirty([&]{state.all_visible[0].pop_back();});dirty([&]{state.all_visible[0].clear();});
}
''')


if __name__ == "__main__":
    unittest.main()
