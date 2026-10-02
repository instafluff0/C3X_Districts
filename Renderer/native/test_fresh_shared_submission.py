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
    def test_shift_wrap_guard_and_unchanged_lighting_use_exact_persistent_ranges(self):
        fresh = (ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        cpp = (ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        key = method(cpp, '    c3x_renderer::render_core::SharedInstanceSubmission::Key shared_instance_draw_key(')
        covered = method(fresh, '    template<class BodyInputs> bool body_placements_covered(')
        append = method(fresh, '    template<class BodyInputs> bool append_body_placements(')
        run_cpp(GPU_STUB + r'''
#include "Renderer/native/render_core/geometry_draws.h"
struct Mesh {
 std::array<int,4> bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};
 std::uint64_t version=0;std::shared_ptr<std::vector<Owner::Instance> const> instances;float instance_material=40;
};
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<Mesh,2>;
using GeometryDrawReference=GeometryDrawView::Reference;
using GeometryDrawRecord=GeometryDrawView::Record;
struct Renderer {
 Owner shared_instances;
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
 auto inputs=[&](auto visit){for(auto const& record:required)visit(1,GeometryDrawReference(record));};
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
 // Reordering and light/page changes do not change the placement owner.
 assert(renderer.shared_instances.find_covering([&](auto const& generation){return harness.body_placements_covered(generation,inputs);})==front);
 assert(renderer.shared_instances.uploads==uploads);
 ++required[0].translation_y;assert(!harness.body_placements_covered(*front,inputs));
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
struct Resource {
 unsigned references=1,releases=0;
 void AddRef(){assert(references);++references;}
 void Release(){assert(references);--references;++releases;}
};
using ID3D11ShaderResourceView=Resource;
struct Stream {void clear(){}};
struct Lease {void reset(){}};
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
 std::array<Resource*,25> targets{};std::vector<Resource*> patch_buffers;Lease shared_front;
 std::size_t production_field_bytes=0;
 template<class T> static void drop(T*& pointer){if(pointer)pointer->Release();pointer=nullptr;}
 void retain_source(){
''' + pin + r'''
 }
''' + destructor + r'''
};
int main(){
 Resource original,fresh,current;
 renderer.source_shadow.view=&original;
 {
  SandboxSceneShadow owner;owner.retain_source();owner.view=&fresh;
  renderer.source_shadow.borrowed_view=&fresh;renderer.fresh_shadow_working_bytes=228;
  assert(renderer.source_shadow.sampled_view()==&fresh && original.references==2);
  // Reset clears the borrowed pointer, releasing only the original ownership.
  renderer.source_shadow.clear();assert(!renderer.source_shadow.sampled_view());
  assert(original.references==1 && fresh.references==1 && renderer.fresh_shadow_working_bytes==228);
  // New device source ownership exists before the old FRESH owner is retired.
  renderer.source_shadow.view=&current;assert(renderer.source_shadow.sampled_view()==&current);
 }
 assert(!original.references && !fresh.references && current.references==1);
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
