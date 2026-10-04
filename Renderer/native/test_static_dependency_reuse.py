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
    def test_empty_shadow_receiver_extent_is_finite_and_can_resume(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        body = method(source, '    template<class BodyInputs,class RetireCompletedPlans> bool render(')
        body = body[body.index('{') + 1:body.index('        receiver_revision=revision;')]
        run_cpp(r'''
#include <array>
#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cassert>
constexpr unsigned geometry_layer_count=2,geometry_shadow=1;
struct Bounds{float low[2]={10,20},high[2]={30,40};};
struct Record{int tile_x=0,tile_y=0;struct Content{Bounds world_bounds;}value;
 Content const& content()const{return value;}};
struct Shadow{static std::array<float,4> project(Bounds const& b,float const*,std::array<float,12> const&){
 return {b.low[0],b.low[1],b.high[0],b.high[1]};}};
struct SandboxPassWorkload{enum{shadow};struct Scope{template<class T>Scope(T&,int){}};};
struct Test {
 struct Grid{std::array<float,4> bounds{};};
 struct {bool geometry_canonical_world=false;std::array<float,12> shadow_basis{};
  struct {struct World{struct Dims{bool wrap_x=false,wrap_y=false;int width=60,height=60;};
   Dims dimensions()const{return {};}};World world()const{return {};}}world_coast;}renderer;
 int workload=0;int* work=&workload;bool ready=true;
 bool receiver_grid_valid=false;Grid receiver_grid;std::array<float,4> receiver_wrap{};
 std::array<std::uint64_t,2> receiver_key{};std::array<float,12> receiver_light{};
 unsigned receiver_reuses=0,receiver_builds=0,receiver_visits=0;
 bool ensure(){return ready;}bool refresh_casters(std::uint64_t){return true;}
 auto receiver_identity(std::uint64_t r,std::uint64_t s){return std::array<std::uint64_t,2>{r,s};}
 bool configure_stable(Grid& grid,float const* needed){
  for(int i=0;i<4;++i)assert(std::isfinite(needed[i]));
  assert(needed[2]>needed[0]&&needed[3]>needed[1]);
  std::copy(needed,needed+4,grid.bounds.begin());return true;}
 bool receivers(std::array<std::vector<Record>,2> const& receivers,std::uint64_t revision){
  std::uint64_t scene=1,membership=1;
''' + body + r'''
  return true;
 }
};
int main(){Test p;std::array<std::vector<Record>,2> records;
 assert(p.receivers(records,1)&&p.receiver_grid_valid&&p.receiver_builds==1);
 assert((p.receiver_grid.bounds==std::array<float,4>{0,0,1,1}));
 assert(p.receivers(records,1)&&p.receiver_reuses==1&&p.receiver_builds==1);
 records[0].push_back({});assert(p.receivers(records,2)&&p.receiver_builds==2);
 assert((p.receiver_grid.bounds==std::array<float,4>{10,20,30,40}));
 records[0].clear();assert(p.receivers(records,3)&&p.receiver_builds==3);
 assert((p.receiver_grid.bounds==std::array<float,4>{0,0,1,1}));
 p.ready=false;assert(!p.receivers(records,4));
}
''')

    def test_scene_reassembly_preserves_pixels_until_local_order_proof(self):
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        body = method(source, "    bool capture(ViewportShaderSettings const& settings,")
        # The production capture must leave local invalidation to its ordered
        # raster proof, even when full assembly/sorting advances scene order.
        self.assertNotIn("order_revision()", body)
        self.assertNotIn("resident_order_revision", source)
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Proof{};
int main(){
 RasterContributors<Proof> pixels;auto proof=std::make_shared<Proof>();
 RasterContributors<Proof>::Key a{},b{},c{},d{};a[0]=1;b[0]=2;c[0]=3;d[0]=4;
 auto append=[&](std::initializer_list<decltype(a)> keys){pixels.begin_append();
  for(auto const& key:keys)assert(pixels.add(key,proof,key[0],1));};
 auto validate=[&](std::initializer_list<decltype(a)> keys){pixels.begin_membership();bool ok=true;
  for(auto const& key:keys)ok=pixels.visit_membership(key)&&ok;return ok&&pixels.exact_membership();};
 append({a,c});assert(validate({a,c}));
 // An entering strip inserts new contributors between retained contributors.
 append({a,b,c});assert(validate({a,b,c}));
 // A disjoint strip does not constrain its order relative to old pixels.
 append({d});assert(validate({d,a,b,c}));assert(validate({a,b,c,d}));
 // Genuine overlap-order reversals are rejected, as are missing/new draws.
 assert(!validate({a,c,b,d}));assert(!validate({c,a,b,d}));assert(!validate({a,b,c}));
 // Removing an intermediate draw must not hide a reversal of its neighbors.
 assert(!validate({a,c,d}));assert(pixels.remaining_order_preserved());
 assert(!validate({c,a,d}));assert(!pixels.remaining_order_preserved());
 assert(!validate({a,d}));assert(pixels.remaining_order_preserved());
 pixels.clear();append({c,b,a});assert(validate({c,b,a}));assert(!validate({a,b,c}));
 // Hundreds of camera recaptures neither grow metadata nor weaken order.
 auto bytes=pixels.bytes();for(int i=0;i<1000;++i){assert(validate({c,b,a}));assert(pixels.bytes()==bytes);}
}
''')

    def test_empty_viewport_is_a_successful_selection_and_can_resume(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        capture = method(source, '    bool capture(ViewportShaderSettings const& settings,')
        run_cpp(r'''
#include <array>
#include <vector>
#include <cstdint>
#include <cassert>
using LONG=int;
struct D3D11_RECT{int left,top,right,bottom;};
struct ViewportShaderSettings{float translation[2]={};};
constexpr unsigned geometry_layer_count=9,geometry_underlay=0,geometry_bed=1,geometry_water=2,
 geometry_river=3,geometry_route=4,geometry_shadow=5,geometry_wave=6;
namespace c3x_renderer{namespace render_core{constexpr unsigned raster_scene=0,raster_classification=1;}}
struct Record{unsigned id=1;bool water_dependent=false;struct Content{bool animation_texture=false;}value;
 Content const& content()const{return value;}};
Record const& GeometryDrawReference(Record const&r){return r;}
struct SandboxPassWorkload{enum{selection,screen=0};};
using Records=std::array<std::vector<Record>,geometry_layer_count>;
struct Pipeline {
 struct Renderer {Records geometry_vertex_buffers;bool water_scene_active=false,hidden=false;
  struct{float height_pixels=0;}reflection;
  bool chunk_intersects_region(Record const&,ViewportShaderSettings const&,D3D11_RECT,bool){return !hidden;}
 }renderer;
 struct {void invalidate_all(unsigned){}}static_rasters;
 struct Preview{bool valid=false;};std::array<Preview,2> bootstrap;
 struct Work {bool enabled=true;struct Counts{unsigned reuses=0,rebuilds=0,tested_records=0,accepted_records=0;};
 std::array<std::array<Counts,9>,1> counts;}work;
 struct RasterInputs{using Key=unsigned;};
 Records resident,static_visible,water_visible,reflection_visible,all_visible;
 std::uint64_t resident_signature=0,visibility_revision=0,reflection_revision=0,static_receiver_revision=0;
 bool visibility_valid=false,reflection_valid=false,reflected_terrain_material_valid=false,resident_water_scene=false;
 int wrap_pixels=0;unsigned resident_builds=0,visible=0,culled=0,reflection_count=0;
 float projection_zoom=1;
 std::array<std::uint64_t,6> visibility_scene_key={};std::array<float,6> visibility_view_key={};
 std::vector<unsigned> reflection_inputs;
 unsigned contributor_key(unsigned layer,Record const&r){return layer*10+r.id;}
 std::uint64_t view_revision()const{return 1;}
 template<class Visit>void contributors(ViewportShaderSettings const&,D3D11_RECT,bool,Visit visit){
  for(unsigned i=0;i<9;++i)for(auto&r:resident[i])visit(i,r);}
 D3D11_RECT source_bounds(ViewportShaderSettings const&,D3D11_RECT r,bool){return r;}
 D3D11_RECT reflected_water_bounds(ViewportShaderSettings const&,int w,int h){return {0,0,w,h};}
'''+capture+r'''
};
int main(){Pipeline p;ViewportShaderSettings settings;
 // A fresh entirely hidden/empty viewport is a successful capture and reuse.
 assert(p.capture(settings,settings,2248,1268,3840)&&p.visible==0&&p.visibility_valid);
 assert(p.capture(settings,settings,2248,1268,3840)&&p.visible==0&&p.visibility_revision==1);
 Pipeline populated;populated.renderer.geometry_vertex_buffers[0].push_back({});
 assert(populated.capture(settings,settings,2248,1268,3840)&&populated.visible==1);
 populated.renderer.hidden=true;populated.projection_zoom=3;
 assert(populated.capture(settings,settings,2248,1268,3840)&&populated.visible==0);
 assert(populated.all_visible[0].empty()&&populated.static_visible[0].empty());
 auto revision=populated.visibility_revision;
 assert(populated.capture(settings,settings,2248,1268,3840)&&populated.visible==0&&populated.visibility_revision==revision);
 populated.renderer.hidden=false;populated.projection_zoom=1;
 assert(populated.capture(settings,settings,2248,1268,3840)&&populated.visible==1);
 assert(populated.static_receiver_revision==3);
}
''')

    def test_bounded_exact_membership_rejects_removals_duplicate_visits_and_stamp_wrap(self):
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Proof {};
int main(){
 using Inputs=RasterContributors<Proof>;Inputs pixels;auto proof=std::make_shared<Proof>();
 Inputs::Key first{},second{},strip{};first[0]=second[0]=strip[0]=17;second[1]=1;strip[1]=2;
 assert(pixels.add(first,proof,55,3) && pixels.add(second,proof,55,3));
 auto bytes=pixels.bytes();pixels.begin_membership();
 assert(pixels.visit_membership(first) && pixels.visit_membership(first));
 assert(!pixels.exact_membership()); // duplicates cannot hide the removed second key.
 assert(pixels.visit_membership(second) && pixels.exact_membership() && pixels.bytes()==bytes);
 // Strip append admits a union of logical contributors, not another count for
 // each draw intersecting multiple raster writes. Complete covered validation
 // must still visit every member of that admitted union.
 assert(pixels.add(first,proof,55,3) && pixels.add(strip,proof,55,3));
 assert(pixels.draws.size()==3);pixels.begin_membership();
 assert(pixels.visit_membership(first) && pixels.visit_membership(second));
 assert(!pixels.exact_membership());assert(pixels.visit_membership(strip) && pixels.exact_membership());
 // A wrap cannot confuse an ancient visitation stamp with the current scan.
 pixels.membership_epoch=UINT64_MAX;pixels.begin_membership();assert(pixels.membership_epoch==1);
 assert(pixels.visit_membership(first) && !pixels.exact_membership());
 assert(pixels.visit_membership(second) && pixels.visit_membership(strip) && pixels.exact_membership());
 pixels.clear();assert(!pixels.membership_epoch && !pixels.membership_seen && pixels.draws.empty());
 assert(!pixels.visit_membership(first) && !pixels.exact_membership());
 pixels.begin_membership();assert(pixels.exact_membership());
 assert(pixels.add(second,proof,55,3));pixels.begin_membership();
 assert(!pixels.visit_membership(first) && !pixels.exact_membership());
 assert(pixels.visit_membership(second) && pixels.exact_membership());
 assert(!pixels.add(first,nullptr,55,3));pixels.begin_membership();
 assert(!pixels.visit_membership(second) && !pixels.exact_membership());
}
''')

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
 unsigned calls=0,refuse=~0u;bool complete=true;
 bool watch(Domain,std::uint64_t){return ++calls!=refuse;}
 template<class Source,class Register>bool watch_source(std::shared_ptr<Source> const& source,Register register_inputs){return source&&register_inputs(*source);}
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
 refused.clear();auto empty=std::make_shared<NaturalWorld::CellContent>();incomplete.river_dependencies[0].second=empty;
 assert(!state.watch_raster_dependencies(incomplete,refused));
 // A valid empty query is distinct from a missing PageInputs owner.
 refused.clear();empty->inputs=std::make_shared<NaturalWorld::PageInputs>();
 assert(state.watch_raster_dependencies(incomplete,refused));
 assert(refused.dependencies.size()==1&&refused.dependencies.count({Domain::visibility,0})==1);
}
''')

    def test_registration_visits_unique_proofs_and_shared_pages_once(self):
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
int main(){
 State state;using Inputs=RasterContributors<CachedGeometryProof>;Inputs pixels;
 decltype(pixels.dependencies) expected;std::weak_ptr<NaturalWorld::PageInputs> lifetime;
 {
  std::vector<std::shared_ptr<NaturalWorld::PageInputs>> pages;
  for(unsigned page=0;page<4;++page){auto p=std::make_shared<NaturalWorld::PageInputs>();
   for(unsigned i=0;i<512;++i){auto id=page*512+i;p->values.push_back({id,0});
    expected.insert({Domain::world,id});expected.insert({Domain::flow,id});}
   pages.push_back(p);
  }
  lifetime=pages.front();std::vector<std::shared_ptr<CachedGeometryProof>> proofs;
  for(unsigned i=0;i<100;++i){auto p=std::make_shared<CachedGeometryProof>();
   p->tile=55;p->appearance_dependencies={{0,0}};p->dependencies={{3000+i,0}};
   for(unsigned cell=0;cell<8;++cell){auto content=std::make_shared<NaturalWorld::CellContent>();
    content->inputs=pages[cell%4];p->river_dependencies.push_back({{int(cell),0,0,0},content});}
   proofs.push_back(p);expected.insert({Domain::semantic,3000+i});
  }
  expected.insert({Domain::appearance,0});expected.insert({Domain::visibility,55});
  for(unsigned occurrence=0;occurrence<10000;++occurrence){auto& p=proofs[occurrence%100];
   Inputs::Key draw{};draw[0]=1+occurrence%100;draw[1]=occurrence;
   bool first=false;assert(pixels.add(draw,p,55,0,&first));
   if(first)assert(state.watch_raster_dependencies(*p,pixels));
   // Occurrence visibility is independent of shared immutable mesh identity.
   assert(pixels.watch(Domain::visibility,55));
  }
 }
 pixels.finish_dependencies();assert(pixels.complete&&pixels.dependencies_complete);
 assert(pixels.draws.size()==10000&&pixels.proofs.size()==100&&pixels.dependencies==expected);
 auto const& counts=pixels.validation_counts;
 assert(counts.proof_registrations==100&&counts.source_expansions==4&&counts.source_reuses==796);
 assert(counts.dependency_watch_calls==10000+100*3+4*512*2);
 assert(!lifetime.expired());
 RasterDependencyRevisions revisions;Inputs::ValidationKey key{};unsigned calls=0;bool current=true;
 auto exact=[&]{++calls;return current;};assert(pixels.validate(revisions,key,exact));
 for(unsigned i=0;i<1000;++i)assert(pixels.validate(revisions,key,exact));assert(calls==1);
 // Missing->present insertion and present->missing removal both reach exact validation.
 for(auto domain:{Domain::world,Domain::flow,Domain::semantic,Domain::appearance,Domain::visibility}){
  auto id=domain==Domain::semantic?3000:domain==Domain::visibility?55:0;
  current=false;revisions.touch(domain,id);auto prior=calls;assert(!pixels.validate(revisions,key,exact)&&calls==prior+1);
  current=true;revisions.touch(domain,id);assert(pixels.validate(revisions,key,exact));
 }
 pixels.clear();assert(lifetime.expired());assert(pixels.dependency_sources.empty());
 // A replacement owner is expanded anew, with no old registration left behind.
 auto replacement=std::make_shared<CachedGeometryProof>();replacement->tile=56;
 auto cell=std::make_shared<NaturalWorld::CellContent>();cell->inputs=std::make_shared<NaturalWorld::PageInputs>();
 cell->inputs->values={{9000,1}};replacement->river_dependencies={{{0,0,0,0},cell}};
 Inputs::Key draw{};draw[0]=101;bool first=false;assert(pixels.add(draw,replacement,56,1,&first)&&first);
 auto expansions=counts.source_expansions;assert(state.watch_raster_dependencies(*replacement,pixels));
 assert(counts.source_expansions==expansions+1&&pixels.dependencies.count({Domain::world,9000}));
 assert(!pixels.dependencies.count({Domain::world,0}));
 // Failed source expansion cannot be certified, cached or retried as success.
 Inputs failed;unsigned attempts=0;
 assert(!failed.watch_source(cell->inputs,[&](auto const&){++attempts;return false;}));
 assert(!failed.complete&&!failed.watch_source(cell->inputs,[&](auto const&){++attempts;return true;}));
 assert(attempts==1);failed.finish_dependencies();assert(!failed.dependencies_complete);
 assert(!failed.validate(revisions,key,[]{return true;}));
 failed.clear();assert(failed.watch_source(cell->inputs,[](auto const&){return true;}));
 std::shared_ptr<NaturalWorld::PageInputs> missing;assert(!failed.watch_source(missing,[](auto const&){return true;}));
 // Metadata budget refusal remains sticky and cannot acquire another source owner.
 failed.clear();for(unsigned i=0;failed.complete&&i<500000;++i)failed.watch(Domain::world,i);
 assert(!failed.complete);auto owners=failed.dependency_sources.size();
 assert(!failed.watch_source(cell->inputs,[](auto const&){return true;}));assert(failed.dependency_sources.size()==owners);
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
 // Add-only strip expansion becomes reusable after its exact union is known.
 auto extra=state.records.front();++extra.ordinal;state.records.push_back(extra);++state.membership;
 assert(!state.raster_dependencies(pixels,settings,region,false));
 assert(state.raster_dependencies(pixels,settings,region,true));assert(state.raster_dependencies(pixels,settings,region,false));
 visits=state.contributor_visits;for(unsigned i=0;i<1000;++i)assert(state.raster_dependencies(pixels,settings,region,false));
 assert(state.contributor_visits==visits);
 // Duplicate validation visits cannot disguise a dropped retained contribution.
 state.records.insert(state.records.begin(),state.records.front());++state.membership;
 assert(state.raster_dependencies(pixels,settings,region,false));
 assert(pixels.draws.size()==2);state.records.pop_back();++state.membership;
 assert(!state.raster_dependencies(pixels,settings,region,false));
 state.records.push_back(extra);++state.membership;assert(state.raster_dependencies(pixels,settings,region,false));
 auto saved=state.records;state.records.clear();++state.membership;
 assert(!state.raster_dependencies(pixels,settings,region,false));
 state.records=saved;++state.membership;assert(state.raster_dependencies(pixels,settings,region,false));
 // Local visibility is separately watched, even with otherwise valid content.
 ++state.renderer.topology_cache.record.visibility_revision;
 state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::visibility,55);
 assert(!state.raster_dependencies(pixels,settings,region,false));
}
''')

    def test_production_receiver_grid_reuses_exact_static_view_and_light(self):
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        identity = method(source, "    std::array<std::uint64_t,9> receiver_identity(")
        stable = method(source, "    bool configure_stable(Grid& grid,float const* needed){")
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
struct PerfOptions {bool shadow_tight=false;};
PerfOptions& sandbox_perf_options(){static PerfOptions options;return options;}
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
 std::array<float,2> stable_span{};std::array<float,12> stable_light{};std::uint64_t span_refits=0;
''' + identity + stable + r'''
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
 // A scrolling receiver set rebuilds the window but keeps the sampling span,
 // so retained static pixels and unchanged shadow pages stay valid.
 auto span=state.receiver_grid.quality_span;auto refits=state.span_refits;
 for(unsigned step=0;step<40;++step){
  for(auto& record:records[0]){record.world_bounds.low[0]+=.25f;record.world_bounds.high[0]+=.25f;}
  if(step%7==3)records[0].back().world_bounds.high[1]+=.05f;
  assert(state.prepare(records,100+step,2));
  assert(state.receiver_grid.quality_span==span);
 }
 assert(state.span_refits==refits);
 // A genuinely larger receiver extent (zoom out) refits once.
 for(auto& record:records[0])record.world_bounds.high[1]+=20.f;
 assert(state.prepare(records,200,2));assert(state.span_refits==refits+1&&state.receiver_grid.quality_span!=span);
 sandbox_perf_options().shadow_tight=true;assert(state.prepare(records,201,2));
}
''')

    def test_production_atlas_skips_unchanged_static_casters_and_proofs(self):
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        methods = "\n".join(method(source, signature) for signature in [
            "    AtlasInputs::Key caster_key(", "    bool atlas_dependencies(",
        ])
        run_cpp(r'''
#define C3X_RENDERER64_FRESH
#include "Renderer/native/render_core/shadow_page_contents.h"
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
struct State {using AtlasInputs=c3x_renderer::render_core::ShadowCasterProofs<CachedGeometryProof>;
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
 AtlasInputs atlas_inputs;c3x_renderer::render_core::ShadowPageContents<AtlasInputs::Key> page_contents;
 std::uint64_t proof_membership_signature=~std::uint64_t(0);bool shadow_metadata_admit(std::size_t,std::size_t){return true;}
 std::vector<Shadow::Caster> caster_inputs={Shadow::Caster{}};
 std::uint64_t caster_signature=1;float box[4]={0,0,48,32};
''' + methods + r'''
};
int main(){State state;assert(state.atlas_dependencies(true));assert(state.atlas_dependencies(false));
 auto counts=state.atlas_inputs.validation_counts;auto proofs=state.renderer.proof_visits;
 for(unsigned i=0;i<1000;++i)assert(state.atlas_dependencies(false));
 assert(state.atlas_inputs.validation_counts.membership==counts.membership&&state.atlas_inputs.validation_counts.content==counts.content&&state.renderer.proof_visits==proofs);
 state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::world,888);assert(state.atlas_dependencies(false));assert(state.renderer.proof_visits==proofs);
 for(unsigned i=0;i<12;++i){auto before=state.atlas_inputs.validation_counts.full;++state.renderer.shadow_basis[i];assert(state.atlas_dependencies(false));assert(state.atlas_inputs.validation_counts.full==before);}
 state.renderer.semantic=8;state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::semantic,77);assert(!state.atlas_dependencies(false));
 state.renderer.semantic=7;state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::semantic,77);assert(state.atlas_dependencies(false));
 ++state.caster_signature;++state.caster_inputs[0].index_offset;assert(!state.atlas_dependencies(false));
 state.atlas_inputs.clear();assert(state.atlas_dependencies(true));assert(state.atlas_dependencies(false));
 // Dropping an exact, still-permitted caster must invalidate its old maximum-
 // height pixels even when every remaining caster is contained in the atlas.
 auto extra=state.caster_inputs.front();++extra.index_offset;state.caster_inputs.push_back(extra);++state.caster_signature;
 assert(!state.atlas_dependencies(false));assert(state.atlas_dependencies(true));assert(state.atlas_dependencies(false));
 state.caster_inputs.push_back(state.caster_inputs.front());++state.caster_signature;
 assert(!state.atlas_dependencies(false));assert(state.atlas_dependencies(true));
 state.caster_inputs.erase(state.caster_inputs.begin()+1);++state.caster_signature;
 assert(!state.atlas_dependencies(false));assert(state.atlas_dependencies(true));
 // A membership edit preserves the unchanged producer's dependency expansion.
 assert(state.atlas_inputs.validation_counts.proof_registrations==2);
 auto saved=state.caster_inputs;state.caster_inputs.clear();++state.caster_signature;
 assert(!state.atlas_dependencies(false));assert(state.atlas_dependencies(true));assert(state.atlas_inputs.producers.empty());
 state.caster_inputs=saved;++state.caster_signature;assert(!state.atlas_dependencies(false));assert(state.atlas_dependencies(true));
 ++state.renderer.topology_cache.record.visibility_revision;state.renderer.raster_dependency_revisions.touch(RasterDependencyRevisions::Domain::visibility,55);assert(!state.atlas_dependencies(false));
 state.caster_lease->content.mesh->proof.reset();++state.caster_signature;assert(!state.atlas_dependencies(true));
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
