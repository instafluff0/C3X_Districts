"""Execute production placement closure and borrowed shadow lifetime contracts."""
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_shared_instance_submission import GPU_STUB
from Renderer.lab.platform import ROOT

def method(text, signature):
    start = text.index(signature)
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class ReflectionClosureTests(unittest.TestCase):
    def test_production_list_deduplicates_real_consumers_and_reuses_unchanged_selection(self):
        fresh = (ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        cpp = (ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        key = method(cpp, '    c3x_renderer::render_core::SharedInstanceSubmission::Key shared_instance_draw_key(')
        roi = method(fresh, '    bool update_roi(ViewportShaderSettings const& settings,int w,int h){')
        run_cpp(GPU_STUB + r'''
#define C3X_RENDERER64_FRESH 1
#include <chrono>
#include <climits>
#include "Renderer/native/scene_projection.h"
#include <cstring>
#include "Renderer/native/render_core/body_placement_requirements.h"
using LONG=int;struct D3D11_RECT {LONG left,top,right,bottom;};
struct ViewportShaderSettings {float translation[2]={},inverse_size[2]={};};
struct Mesh {std::array<int,4> bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};
 std::uint64_t version=1;std::shared_ptr<std::vector<Owner::Instance> const> instances;float instance_material=40;};
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<Mesh,2>;
using GeometryDrawReference=GeometryDrawView::Reference;
using GeometryDrawRecord=GeometryDrawView::Record;
constexpr unsigned geometry_layer_count=2,geometry_shadow=0;
struct Renderer {Owner shared_instances;bool water_scene_active=true;unsigned content_revision=1,content_view_width=1000,content_view_height=800;
 struct Topology {std::uint64_t sequence=1;std::uint64_t visibility_sequence()const{return sequence;}} topology_cache;
 unsigned tests=0;
 bool chunk_intersects_region(GeometryDrawReference const& draw,ViewportShaderSettings const&,D3D11_RECT clip,bool){++tests;
  return draw.bounds()[0]<clip.right && draw.bounds()[2]>clip.left;}
''' + key + r'''
};
struct StaticRasters {static unsigned lane_of(float zoom){return zoom==1.f?0u:1u;}};
struct Harness {
 Renderer renderer;GeometryDrawView::Records all_visible,guard,roi_records,roi_shadow_records;
 c3x_renderer::render_core::BodyPlacementRequirements<Mesh> body_requirements;
 bool body_requirements_valid=false;unsigned body_requirement_builds=0,body_requirement_reuses=0,body_requirement_visits=0;
 double body_requirement_ms=0;float projection_zoom=1;std::array<float,2> lane_projection{};
 int camera_x=0,camera_y=0,wrap_pixels=0;
 static constexpr int region_margin_x=320,region_margin_y=192,roi_quantum_x=512,roi_quantum_y=256;
 std::array<std::int64_t,11> roi_key{};std::uint64_t roi_revision=1,roi_receiver_check=0,static_receiver_revision=0,membership=1;
 unsigned queries=0;
 struct StaticRect {int left=0,top=0,right=0,bottom=0;} shadow_field;
 float tile_half_width=64,tile_half_height=32;struct {std::array<float,2> receiver_area{};} shadow;
 struct Options {bool shadow_tight=false;};Options sandbox_perf_options()const{return {};}
 float zoom_destination()const{return 1.f;}
 bool canonical_hidden()const{return false;}
 std::uint64_t view_revision()const{return membership;}
 template<class Visit>void contributors(ViewportShaderSettings const&,D3D11_RECT,bool,Visit visit){++queries;
  for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& draw:guard[layer])visit(layer,draw);}
''' + roi + r'''
};
int main(){
 Harness h;Mesh mesh;mesh.instances=std::make_shared<std::vector<Owner::Instance>>(3);mesh.bounds={0,0,100,100};
 GeometryDrawRecord original(mesh);original.owner={1,9};original.translation_x=-6400;
 auto main=original;main.translation_x=0;auto reflected=main;reflected.translation_x=6400;
 auto guarded=main;guarded.translation_x=12800;auto irrelevant=guarded;irrelevant.translation_x=25600;irrelevant.bounds={4000,0,4100,100};
 auto aquatic=guarded;aquatic.translation_x=19200;aquatic.water_dependent=true;
 h.all_visible[1]={main,reflected,main};h.guard[1]={main,guarded,irrelevant,aquatic};ViewportShaderSettings settings;
 // One region-of-interest walk owns body placements for every lane.
 assert(h.update_roi(settings,1000,800) && h.queries==1);
 // The receiver field is recorded in source pixels (screen minus translation).
 assert(h.shadow_field.left<0 && h.shadow_field.right>1000 && h.shadow_field.top<0 && h.shadow_field.bottom>800);
 auto has=[&](GeometryDrawRecord const& draw){auto expected=h.renderer.shared_instance_draw_key(1,GeometryDrawReference(draw));
  for(auto const& entry:h.body_requirements.entries)if(entry.key==expected)return true;return false;};
 assert(has(main) && has(reflected) && has(guarded) && has(aquatic) && !has(original) && !has(irrelevant));
 assert(h.roi_records[1].size()==3 && h.roi_revision==2);
 // Unchanged views and camera steps within the quantum reuse it exactly.
 auto retained=h.renderer.shared_instances.bytes();
 assert(h.update_roi(settings,1000,800) && h.queries==1 && h.body_requirement_reuses==1);
 h.camera_x=100;h.camera_y=-1;h.camera_y=0;assert(h.update_roi(settings,1000,800) && h.queries==1);
 assert(h.renderer.shared_instances.bytes()==retained && h.roi_revision==2);
 // Vanilla scroll steps (128 px in x, 64 px in y at 1x) inside a world-window
 // block keep the region: a 128 px quantum rebuilt it at every step (review 41).
 for(int step=1;step<4;++step){h.camera_x=128*step;assert(h.update_roi(settings,1000,800) && h.queries==1);}
 for(int step=1;step<4;++step){h.camera_y=64*step;assert(h.update_roi(settings,1000,800) && h.queries==1);}
 h.camera_y=0;assert(h.update_roi(settings,1000,800) && h.queries==1 && h.roi_revision==2);
 // Crossing a block, membership or visibility changes rebuild exactly once.
 h.camera_x=520;assert(h.update_roi(settings,1000,800) && h.queries==2 && h.roi_revision==3);
 assert(h.update_roi(settings,1000,800) && h.queries==2);
 ++h.membership;assert(h.update_roi(settings,1000,800) && h.queries==3);
 ++h.renderer.topology_cache.sequence;assert(h.update_roi(settings,1000,800) && h.queries==4);
 // A captured receiver inside the region does not rebuild; one outside does.
 ++h.static_receiver_revision;assert(h.update_roi(settings,1000,800) && h.queries==4);
 h.all_visible[1].push_back(irrelevant);++h.static_receiver_revision;
 assert(h.update_roi(settings,1000,800) && h.queries==5 && has(irrelevant));
 // Contradictory counts under one exact key fail, and admission is charged
 // against the existing owner allowance rather than an independent cache.
 auto mutable_values=std::make_shared<std::vector<Owner::Instance>>(3);Mesh unstable=mesh;unstable.instances=mutable_values;
 GeometryDrawRecord draw(unstable);auto exact=h.renderer.shared_instance_draw_key(1,GeometryDrawReference(draw));
 c3x_renderer::render_core::BodyPlacementRequirements<Mesh> list;
 assert(list.add(h.renderer.shared_instances,1,GeometryDrawReference(draw),exact));mutable_values->push_back({});
 assert(!list.add(h.renderer.shared_instances,1,GeometryDrawReference(draw),exact));list.clear();
 auto pressure=h.renderer.shared_instances.retain_metadata(Owner::budget-h.renderer.shared_instances.bytes()-256);assert(pressure);
 assert(!list.add(h.renderer.shared_instances,1,GeometryDrawReference(main),h.renderer.shared_instance_draw_key(1,GeometryDrawReference(main))));
 assert(h.renderer.shared_instances.bytes()<=Owner::budget);
 pressure.reset();h.body_requirements.clear();assert(!h.renderer.shared_instances.bytes());
}
''')

    def test_shift_wrap_guard_and_unchanged_lighting_use_exact_persistent_ranges(self):
        fresh = (ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        cpp = (ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        key = method(cpp, '    c3x_renderer::render_core::SharedInstanceSubmission::Key shared_instance_draw_key(')
        covered = method(fresh, '    template<class BodyInputs> bool body_placements_covered(')
        append = method(fresh, '    template<class BodyInputs> bool append_body_placements(')
        run_cpp(GPU_STUB + r'''
#include "Renderer/native/render_core/geometry_draws.h"
#include "Renderer/native/render_core/body_placement_requirements.h"
#include "Renderer/native/render_core/resident_content.h"
struct Proof {};
struct Content {struct Generation {std::shared_ptr<Proof> proof;};std::shared_ptr<Generation> mesh;};
struct Mesh {
 std::array<int,4> bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};
 std::uint64_t version=0;std::shared_ptr<std::vector<Owner::Instance> const> instances;float instance_material=40;
};
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<Mesh,2>;
using GeometryDrawReference=GeometryDrawView::Reference;
using GeometryDrawRecord=GeometryDrawView::Record;
struct Renderer {
 Owner shared_instances;
 c3x_renderer::render_core::ResidentContent<Content> resident_content{1};
 bool raster_content_valid(Proof const&){return true;}
''' + key + r'''
};
struct Harness {
 using Submission=Owner;Renderer& renderer;
''' + covered + '\n' + append + r'''
};
int main(){
 ID3D11Device device;Renderer renderer;Harness harness{renderer};Mesh mesh;
 mesh.instances=std::make_shared<std::vector<Owner::Instance>>(3);mesh.version=89;mesh.instance_material=21.18f;
 GeometryDrawRecord canonical(mesh);canonical.owner={3,712};canonical.ordinal=47;
 canonical.translation_x=-771;canonical.translation_y=281;
 float projection[]={17,23,128,1260};std::memcpy(canonical.natural_projection,projection,sizeof(projection));
 GeometryDrawRecord shifted=canonical;shifted.translation_x+=17;shifted.translation_y-=31;
 GeometryDrawRecord reflected=shifted;reflected.translation_x-=6400;
 GeometryDrawRecord guarded=shifted;guarded.translation_x+=6400;
 std::vector<GeometryDrawRecord> required={canonical,shifted,reflected,guarded,reflected};
 c3x_renderer::render_core::BodyPlacementRequirements<Mesh> inputs;
 auto select=[&]{inputs.clear();for(auto const& record:required)
  assert(inputs.add(renderer.shared_instances,1,GeometryDrawReference(record),renderer.shared_instance_draw_key(1,GeometryDrawReference(record))));};
 select();assert(inputs.entries.size()==4 && inputs.visits==5 && inputs.duplicates==1 && inputs.bytes()>0);
 auto builder=renderer.shared_instances.begin(Owner::Key{1});Owner::Range range;
 assert(renderer.shared_instances.append(builder,renderer.shared_instance_draw_key(1,GeometryDrawReference(canonical)),mesh.instances.get(),mesh.instances->data(),3,projection,-771,281,281,21.18f,range));
 auto old=renderer.shared_instances.upload(builder,&device);builder.reset();
 // The original-only union rejects both shifted reflection and strip ranges.
 assert(!harness.body_placements_covered(*old,inputs));
 builder=renderer.shared_instances.begin(Owner::Key{2});assert(harness.append_body_placements(builder,inputs));
 assert(builder->records==12 && builder->ranges.size()==4);
 auto front=renderer.shared_instances.upload(builder,&device);builder.reset();assert(harness.body_placements_covered(*front,inputs));
 for(auto const& record:required){auto found=front->find(renderer.shared_instance_draw_key(1,GeometryDrawReference(record)));
  assert(found.count==3);Owner::Instance packed;
  std::memcpy(&packed,front->buffer->data.data()+found.first*sizeof(packed),sizeof(packed));
  assert(packed.view[0]==record.translation_x && packed.view[1]==record.translation_y && packed.view[2]==record.translation_y);
 }
 auto uploads=renderer.shared_instances.uploads;
 std::reverse(required.begin(),required.end());++required[0].ordinal;
 select();
 // Reordering and light/page changes do not change the placement owner.
 assert(renderer.shared_instances.find_covering([&](auto const& generation){return harness.body_placements_covered(generation,inputs);})==front);
 assert(renderer.shared_instances.uploads==uploads);
 ++required[0].translation_y;select();assert(!harness.body_placements_covered(*front,inputs));
 inputs.clear();
 renderer.shared_instances.clear();old.reset();front.reset();assert(!renderer.shared_instances.bytes());
}
''')

    def test_borrowed_sampling_never_releases_new_generation_or_double_releases_old(self):
        shadow = (ROOT/'Renderer/native/render_core/source_shadow.h').read_text()
        fresh = (ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        accessor = method(shadow, '    ID3D11ShaderResourceView* const& sampled_view()')
        clear = method(shadow, '    void clear(){')
        destructor = method(fresh, '    ~SandboxSceneShadow() {')
        pin_start = fresh.index('        production_view=renderer.source_shadow.view;', fresh.index('    bool ensure() {', fresh.index('struct SandboxSceneShadow')))
        pin_end = fresh.index('        production_field_bytes=0;', pin_start) + len('        production_field_bytes=0;')
        pin = fresh[pin_start:pin_end]
        run_cpp(r'''
#include <cassert>
#include <array>
#include <vector>
#include <cstddef>
#include <memory>
struct Resource {
 unsigned references=1,releases=0;
 void AddRef(){assert(references);++references;}
 void Release(){assert(references);--references;++releases;}
};
using ID3D11ShaderResourceView=Resource;
struct Stream {void clear(){}};
using Lease=std::shared_ptr<Resource>;
struct SourceShadow {
 Resource* view=nullptr;Resource* borrowed_view=nullptr;
 Resource *instance_vertex=nullptr,*resident_instance_vertex=nullptr,*instance_layout=nullptr,*resident_instance_layout=nullptr,*rigid_vertex=nullptr,*resident_rigid_vertex=nullptr;
 Resource *texture=nullptr,*vertex=nullptr,*opaque=nullptr,*cutout=nullptr,*layout=nullptr,*feature_layout=nullptr,*natural_layout=nullptr,*city_layout=nullptr,*caster_settings=nullptr,*table=nullptr,*raster=nullptr,*maximum=nullptr;
 std::array<Resource*,32> targets{};Stream instance_stream;
 template<class T> void drop(T*& pointer){if(pointer)pointer->Release();pointer=nullptr;}
 void clear_cached_pages(){}
''' + accessor + '\n' + clear + r'''
};
struct Renderer {SourceShadow source_shadow;std::size_t fresh_shadow_working_bytes=0;} renderer;
struct SandboxSceneShadow {
 Resource *production_view=nullptr,*texture=nullptr,*view=nullptr,*vertex=nullptr,*instance_vertex=nullptr,*rigid_vertex=nullptr,*opaque=nullptr,*cutout=nullptr,*layout=nullptr,*feature_layout=nullptr,*natural_layout=nullptr,*city_layout=nullptr,*instance_layout=nullptr,*constants=nullptr,*maximum=nullptr,*raster=nullptr;
 std::array<Resource*,25> targets{};std::vector<Resource*> patch_buffers;Lease shared_front,shadow_front;
 std::vector<Lease> terrain_batches;
 std::size_t production_field_bytes=0;
 template<class T> static void drop(T*& pointer){if(pointer)pointer->Release();pointer=nullptr;}
 void retain_source(){
''' + pin + r'''
 }
''' + destructor + r'''
};
int main(){
 Resource original,fresh,current,body,caster,terrain;
 renderer.source_shadow.view=&original;
 {
  SandboxSceneShadow owner;owner.retain_source();owner.view=&fresh;
  auto lease=[](Resource& value){return Lease(&value,[](Resource* pointer){pointer->Release();});};
  owner.shared_front=lease(body);owner.shadow_front=lease(caster);owner.terrain_batches.push_back(lease(terrain));
  renderer.source_shadow.borrowed_view=&fresh;renderer.fresh_shadow_working_bytes=228;
  assert(renderer.source_shadow.sampled_view()==&fresh && original.references==2);
  // Reset clears the borrowed pointer, releasing only the original ownership.
  renderer.source_shadow.clear();assert(!renderer.source_shadow.sampled_view());
  assert(original.references==1 && fresh.references==1 && renderer.fresh_shadow_working_bytes==228);
  assert(body.references==1&&caster.references==1&&terrain.references==1);
  // New device source ownership exists before the old FRESH owner is retired.
  renderer.source_shadow.view=&current;assert(renderer.source_shadow.sampled_view()==&current);
 }
 assert(!original.references && !fresh.references && current.references==1);
 assert(!body.references&&!caster.references&&!terrain.references);
 assert(renderer.source_shadow.view==&current && !renderer.source_shadow.borrowed_view && !renderer.fresh_shadow_working_bytes);
 renderer.source_shadow.clear();assert(!current.references);
 Resource partial;
 {
  SandboxSceneShadow owner;renderer.source_shadow.view=&partial;owner.retain_source();
  renderer.fresh_shadow_working_bytes=228;
  renderer.source_shadow.clear();assert(partial.references==1);
 }
 assert(!partial.references && !renderer.fresh_shadow_working_bytes);
}
''')


class TileContentIdentityTests(unittest.TestCase):
    def test_compiler_policy_quality_and_publication_refresh_preserve_exact_content(self):
        source = (ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        key_start = source.index('            int const shadow_grid = canonical_world_content?')
        key_end = source.index('            auto validation_started=', key_start)
        key = source[key_start:key_end]
        admission_start = source.index('            auto reuse_tile = [&](CachedTileGeometry& cached) {')
        admission_end = source.index('                    auto append_started=', admission_start)
        admission = source[admission_start:admission_end] + 'return true;}return false;};'
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <array>
#include <vector>
#include <cstdint>
#include <cassert>
namespace c3x_renderer {namespace render_core {constexpr unsigned render_core_revision=91;}}
struct Node {int lattice_x=0,lattice_y=0;unsigned degree=0;bool touches_water=false;};
struct CachedTileGeometry {std::array<std::uint64_t,20> compile_context{};};
struct Harness {
 c3x_renderer_tile_v1 tile{};c3x_renderer_frame_v1 frame{};
 bool world_ground=true,pickup_profile=true,dependency_valid=true;
 int content_view_width=960,content_view_height=640,tile_ground_grid=12,flat_grid=8;
 unsigned content_revision=7,device_generation=42,draw_record_count=963;
 std::vector<Node*> local_river_nodes;
 std::array<std::uint64_t,20> compile_context{};
 std::uint64_t tile_content_signature(c3x_renderer_tile_v1 const&)const{return 311;}
 bool tile_content_valid(CachedTileGeometry const&,c3x_renderer_tile_v1 const&)const{return dependency_valid;}
 std::uint64_t content_key(bool canonical_world_content){
''' + key + r'''
  return tile_signature;
 }
 bool admits(CachedTileGeometry& item){
''' + admission + r'''
  return reuse_tile(item);
 }
};
int main(){
 Harness h;h.frame.tile_width=128;h.frame.tile_height=64;
 h.frame.world_width_tiles=h.frame.world_height_tiles=100;h.frame.world_wrap_x=1;
 h.tile.tile_x=50;h.tile.tile_y=50;
 // CPU legacy preparation and canonical fresh policy cannot share a key,
 // even when both use the same near shadow grid and world projection.
 assert(h.content_key(false)!=h.content_key(true));
 h.draw_record_count=500;assert(h.content_key(false)!=h.content_key(true));
 auto near=h.content_key(false);h.draw_record_count=963;
 assert(near!=h.content_key(false));
 auto canonical=h.content_key(true);h.draw_record_count=500;
 assert(canonical==h.content_key(true));h.frame.hour=22;h.frame.season=3;
 assert(canonical==h.content_key(true));
 h.tile_ground_grid=24;assert(canonical!=h.content_key(true));h.tile_ground_grid=12;
 h.flat_grid=16;assert(canonical!=h.content_key(true));h.flat_grid=8;
 ++h.content_revision;assert(canonical!=h.content_key(true));--h.content_revision;
 h.compile_context[14]=(1ull<<48)|5;h.compile_context[15]=4;h.compile_context[17]=92;
 CachedTileGeometry legacy;legacy.compile_context=h.compile_context;
 legacy.compile_context[14]=5;legacy.compile_context[15]=10;
 auto legacy_facts=legacy.compile_context;assert(!h.admits(legacy));assert(legacy.compile_context==legacy_facts);
 // Every concrete context fact remains necessary, including terrain/detail,
 // device/content generations and projection policy. A rejected proof is
 // never rewritten into this request's namespace.
 for(unsigned changed=0;changed<20;++changed){if(changed==17)continue;
  CachedTileGeometry different;different.compile_context=h.compile_context;++different.compile_context[changed];
  auto before=different.compile_context;assert(!h.admits(different));assert(different.compile_context==before);
 }
 // Unchanged, dependency-validated appearance can renew its publication
 // revision while retaining the same actual compiler content.
 CachedTileGeometry refreshed;refreshed.compile_context=h.compile_context;refreshed.compile_context[17]=91;
 assert(h.admits(refreshed));assert(refreshed.compile_context==h.compile_context);
 CachedTileGeometry invalid;invalid.compile_context=h.compile_context;invalid.compile_context[17]=90;
 auto invalid_facts=invalid.compile_context;h.dependency_valid=false;
 assert(!h.admits(invalid));assert(invalid.compile_context==invalid_facts);
}
''')


class TerrainProducerPolicyTests(unittest.TestCase):
    def test_exact_policy_and_canonical_coordinates_control_async_admission(self):
        source = (ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        producer = source.index('        if(cpu_terrain_enabled && !prewarming && !world_batch_enabled){')
        start = source.index('            for(unsigned i=0;i<frame.tile_count;++i){', producer)
        end = source.index('                auto input=terrain_compile_input', start)
        admission = source[start:end] + 'prepared.push_back(tile);}\nreturn prepared;'
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <array>
#include <vector>
#include <cassert>
#include <cstdint>
struct Cached {std::array<std::uint64_t,20> compile_context{};};
struct Observation {unsigned compiled=1;};
struct Topology {
 Observation value;bool known=true;std::uint64_t requested=0;
 Observation* retained(std::uint64_t coordinate){requested=coordinate;return known?&value:nullptr;}
};
struct Content {Cached value;bool known=true;Cached* resolve(unsigned){return known?&value:nullptr;}};
struct Harness {
 c3x_renderer_frame_v1 frame{};Topology topology_cache;Content resident_content;
 std::array<std::uint64_t,20> expected{};bool dependencies=true;
 std::uint64_t coordinate_key(int x,int y){return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);}
 c3x_renderer_tile_v1 content_tile_for(c3x_renderer_tile_v1 tile){tile.tile_x=(tile.tile_x%8+8)%8;return tile;}
 std::vector<int> select_river_nodes(c3x_renderer_tile_v1 const&){return {};}
 unsigned river_context_for(std::vector<int> const&){return 0;}
 auto compile_context_for(c3x_renderer_tile_v1 const&,unsigned){return expected;}
 bool tile_content_valid(Cached const&,c3x_renderer_tile_v1 const&){return dependencies;}
 std::vector<c3x_renderer_tile_v1> admitted(){std::vector<c3x_renderer_tile_v1> prepared;
''' + admission + r'''
 }
};
int main(){
 Harness h;c3x_renderer_tile_v1 tile{};tile.tile_x=-2;tile.tile_y=2;tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 h.frame.tile_count=1;h.frame.tiles=&tile;h.expected[14]=(1ull<<48)|5;h.expected[15]=4;h.expected[17]=92;
 // A wrapped native occurrence uses the same canonical producer namespace
 // as the foreground content request. Legacy policy cannot suppress it.
 h.resident_content.value.compile_context=h.expected;h.resident_content.value.compile_context[14]=5;
 auto selected=h.admitted();assert(selected.size()==1 && selected[0].tile_x==6);
 assert(h.topology_cache.requested==h.coordinate_key(6,2));
 // This early producer has no complete appearance-signature proof yet;
 // every context fact, including the publication revision, remains exact.
 for(unsigned changed=0;changed<20;++changed){
  h.resident_content.value.compile_context=h.expected;++h.resident_content.value.compile_context[changed];
  assert(h.admitted().size()==1);
 }
 h.resident_content.value.compile_context=h.expected;assert(h.admitted().empty());
 h.dependencies=false;assert(h.admitted().size()==1);h.dependencies=true;
 h.resident_content.known=false;assert(h.admitted().size()==1);h.resident_content.known=true;
 h.topology_cache.known=false;assert(h.admitted().size()==1);h.topology_cache.known=true;
 tile.tile_flags=C3X_RENDERER_TILE_PREFETCH;h.dependencies=false;assert(h.admitted().size()==1);
 tile.tile_flags=0;assert(h.admitted().empty());
}
''')


if __name__ == '__main__':
    unittest.main()
