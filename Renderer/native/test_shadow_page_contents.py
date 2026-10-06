"""Exact resource-free shadow-page ownership and production mapping contracts."""
from pathlib import Path
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class ShadowPageContentsTests(unittest.TestCase):
    def test_production_page_publication_never_samples_unfinished_shadows(self):
        # Execute the production page scheduling/completion/table code, replacing
        # only GPU caster draws with a recognizable shadow value per page.
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        schedule=source[source.index('        draws=0;',source.index('    template<class BodyInputs')):
            source.index('        context->OMSetRenderTargets(0,nullptr,nullptr);',source.index('        draws=0;',source.index('    template<class BodyInputs')))]
        draw_start=schedule.index('        auto* target=targets[')
        draw_end=schedule.index('        if(!page_contents.complete_incremental(',draw_start)
        schedule=schedule[:draw_start]+"        ++draws; pixels[page_contents.slots[page_slot]]=100+page_slot;\n"+schedule[draw_end:]
        table=source[source.index('        std::array<std::array<float,4>,64> table{};',source.index('    template<class BodyInputs')):
            source.index('        wrap_basis=wrap_query;',source.index('    template<class BodyInputs'))]
        run_cpp(r'''
#include "Renderer/native/render_core/shadow_page_contents.h"
#include <cassert>
#include <cstdlib>
#include <cstdio>
namespace c3x_renderer {namespace render_core {
unsigned cached_environment(char const*,char*,unsigned){return 0;}
}}
struct Options{bool legacy=false;};Options sandbox_perf_options(){return {};}
using Grid=c3x_renderer::render_core::ShadowSamplingGrid;
using Pages=c3x_renderer::render_core::ShadowPageContents<unsigned>;
int main(){unsigned cases=0;
 for(unsigned missing=0;missing<=25;++missing){
  Grid sampling_grid;sampling_grid.valid=true;sampling_grid.low={-2,-2};
  sampling_grid.count={5,5};sampling_grid.quality_span={40,40};
  Pages page_contents;Pages::Context context={1,2,3};Pages::Inputs inputs;
  std::array<unsigned,25> pixels{};std::array<float,4> wrap_query{};
  bool proved=true;unsigned draws=0;
  for(unsigned i=0;i<25;++i)inputs[i].push_back(i);
  page_contents.select(sampling_grid,context,inputs,true);
  for(unsigned i=0;i<25;++i){assert(page_contents.complete(i,sampling_grid,context,inputs[i]));pixels[page_contents.slots[i]]=100+i;}
  // A reveal, city edit or page-window shift may invalidate any subset.
  for(unsigned i=0;i<missing;++i)inputs[(i*7)%25].push_back(99);
  page_contents.select(sampling_grid,context,inputs,true);
auto publish=[&]()->bool{
'''+schedule+table+r'''
  assert(draws==missing); // unchanged pages do no GPU work
  for(unsigned i=0;i<25;++i){
   int layer=int(table[3+i][0]);
   // -1 samples unshadowed white in the real shader: the observed forest pop.
   unsigned shaded=layer<0?255:pixels[layer];
   assert(shaded==100+i);
  }
  return true;};assert(publish());++cases;
 }
 std::printf("PASS shadow publication: cases=%u complete_first_image=1 unchanged_pages_reused=1\n",cases);
}
''')

    def test_partial_page_completion_survives_frames_and_rejects_changed_casters(self):
        run_cpp(r'''
#include "Renderer/native/render_core/shadow_page_contents.h"
#include <cassert>
using Grid=c3x_renderer::render_core::ShadowSamplingGrid;
using Pages=c3x_renderer::render_core::ShadowPageContents<unsigned>;
int main(){
 Grid grid;grid.valid=true;grid.low={-2,-2};grid.count={5,5};grid.quality_span={40,40};
 Pages pages;Pages::Context context={1,2,3};std::array<float,12> light{};
 auto update=[&](unsigned caster){
  pages.begin_incremental(grid,context,light);pages.mark(caster);
  assert(pages.retire_missing([](auto){return true;}));
  assert(pages.update(caster,grid,[]{return std::array<float,4>{-100,-100,100,100};},[](auto){return true;}));
  assert(pages.finish_incremental(grid,true,[](auto){return true;}));
 };
 for(unsigned frame=0;frame<13;++frame){
  update(1);unsigned complete=0,draws=0;
  for(unsigned page=0;page<grid.pages();++page){
   complete+=pages.reused[page];
   if(!pages.reused[page]&&draws<2){pages.complete_incremental(page);++draws;}
  }
  assert(complete==std::min(frame*2,25u));assert(draws<=2);
 }
 update(1);for(unsigned page=0;page<grid.pages();++page)assert(pages.reused[page]);
 update(2);for(unsigned page=0;page<grid.pages();++page)assert(!pages.reused[page]);
 pages.complete_incremental(7);update(2);assert(pages.reused[7]);
 ++context[2];update(2);for(unsigned page=0;page<grid.pages();++page)assert(!pages.reused[page]);
}
''')

    def test_exact_completed_pages_survive_shift_and_local_contributor_changes(self):
        run_cpp(r'''
#include "Renderer/native/render_core/shadow_page_contents.h"
#include <cassert>
#include <set>
using Grid=c3x_renderer::render_core::ShadowSamplingGrid;
using Key=std::array<std::uint64_t,20>;
using Pages=c3x_renderer::render_core::ShadowPageContents<Key>;
Grid grid(int x,int y){Grid value;value.valid=true;value.low={x,y};value.count={5,5};value.quality_span={40,40};return value;}
Pages::Inputs inputs(Grid const& value){Pages::Inputs result;
 for(unsigned i=0;i<value.pages();++i){auto p=value.page(i);result[i].push_back(Key{unsigned(p[0]+100),unsigned(p[1]+100),7});}return result;}
unsigned reuse(Pages const& pages,Grid const& value){unsigned n=0;std::set<unsigned> physical;
 for(unsigned i=0;i<value.pages();++i){n+=pages.reused[i];assert(physical.insert(pages.slots[i]).second);}return n;}
void finish(Pages& pages,Grid const& value,Pages::Context const& context,Pages::Inputs const& proof){
 for(unsigned i=0;i<value.pages();++i)if(!pages.reused[i])assert(pages.complete(i,value,context,proof[i]));}
int main(){
 Pages pages;Pages::Context context={1,2,3,1};auto a=grid(-2,-2);auto facts=inputs(a);
 pages.select(a,context,facts,true);assert(!reuse(pages,a));finish(pages,a,context,facts);assert(pages.rebuilt==25);
 pages.select(a,context,facts,true);assert(reuse(pages,a)==25 && pages.hits==25);
 auto b=grid(-1,-2);auto moved=inputs(b);pages.select(b,context,moved,true);assert(reuse(pages,b)==20);
 bool different=false;for(unsigned i=0;i<25;++i)if(pages.reused[i])different|=i!=pages.slots[i];assert(different);
 finish(pages,b,context,moved);assert(pages.rebuilt==30);
 // A caster entering from outside the old camera changes only intersected
 // canonical pages. Its removal and a cutout/placement variant do likewise.
 moved[7].push_back(Key{900,1,2});pages.select(b,context,moved,true);assert(reuse(pages,b)==24);finish(pages,b,context,moved);
 moved[7].pop_back();pages.select(b,context,moved,true);assert(reuse(pages,b)==24);finish(pages,b,context,moved);
 moved[11][0][18]=41;pages.select(b,context,moved,true);assert(reuse(pages,b)==24);finish(pages,b,context,moved);
 // Device, asset/config, scope, density, light/season and wrap are exact
 // caller facts. No old pixel survives an incompatible raster context.
 for(unsigned field:{0u,1u,2u,3u,5u,6u,7u,18u,19u,22u}){
  auto changed=context;++changed[field];pages.select(b,changed,moved,true);assert(!reuse(pages,b));finish(pages,b,changed,moved);
  pages.select(b,context,moved,true);assert(!reuse(pages,b));finish(pages,b,context,moved);
 }
 // Unknown producer proof never authorizes reuse, including an empty page.
 pages.select(b,context,moved,false);assert(!reuse(pages,b));
 for(unsigned i=0;i<25;++i)assert(pages.complete(i,b,context,moved[i],false));
 pages.select(b,context,moved,true);assert(!reuse(pages,b));finish(pages,b,context,moved);
 // An interrupted page is not completed, even when every other page is.
 moved[4][0][1]++;pages.select(b,context,moved,true);assert(reuse(pages,b)==24);
 pages.select(b,context,moved,true);assert(reuse(pages,b)==24);finish(pages,b,context,moved);
 // Bound history to the same 25 physical slots across 2000 camera jumps.
 for(int step=0;step<2000;++step){auto next=grid(step%91-45,step%79-39);auto proof=inputs(next);
  pages.select(next,context,proof,true);reuse(pages,next);finish(pages,next,context,proof);}
 assert(pages.bytes()<=sizeof(Pages)+25*2*sizeof(Key));pages.clear();assert(pages.bytes()==sizeof(Pages));
}
''')

    def test_incremental_proof_and_projection_work_witness(self):
        witness=(ROOT/'Renderer/native/test_shadow_preparation.cpp').read_text()
        projection=(ROOT/'Renderer/native/render_core/source_shadow.h').read_text()
        start=projection.index('    static std::array<float,4> project(Bounds')
        self.assertIn(projection[start:projection.index('    std::array<float,4> projected(',start)],witness)
        run_cpp(witness)

    def test_backing_decoder_distinct_river_owners_share_only_exact_watch_storage(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_backing_codec.h"
#include "Renderer/native/render_core/shadow_page_contents.h"
#include <cassert>
#include <set>
using namespace c3x_renderer;
using R=render_core::RasterDependencyRevisions;
using Proof=fidelity::NaturalWorld::CellContent;
int main(){
 PreparedWorld world;world.ground=std::make_unique<fidelity::PreparedGround>();
 world.terrain=std::make_unique<fidelity::TerrainSurfaces>();world.objects=std::make_unique<objects::PreparedObjects>();
 auto original=std::make_shared<fidelity::NaturalWorld::PageInputs>();
 for(unsigned n=0;n<285;++n){original->values.emplace_back(n,7);original->flow.push_back(n%4);}
 original->checked_revision=123;original->current=true;
 for(int n=0;n<1049;++n){auto cell=std::make_shared<Proof>();cell->values={unsigned(n),7,99};cell->inputs=original;
  world.ground->rivers.emplace(fidelity::NaturalWorld::CellKey{1,2,n,0},std::move(cell));}
 auto encoded=WorldBackingCodec::encode(world);assert(!encoded.empty());
 auto restored=WorldBackingCodec::decode(encoded);assert(restored&&restored->ground->rivers.size()==1049);
 render_core::ShadowCasterProofs<Proof> proofs;R revisions;std::set<void const*> identities;
 proofs.begin({1,2,3,1},revisions,[](auto bytes){return bytes<=16u*1024u*1024u;});
 unsigned generation=0,checks=0;
 auto exact=[&](Proof const& cell){++checks;return cell.inputs&&cell.inputs->values==original->values&&cell.inputs->flow==original->flow;};
 for(auto const& item:restored->ground->rivers){auto const& cell=item.second;
  assert(cell->inputs!=original&&identities.insert(cell->inputs.get()).second);
  assert(!cell->inputs->current&&cell->inputs->checked_revision==-1&&!cell->inputs->checked_world);
  ++generation;assert(proofs.add(generation,cell,generation,1,[](auto const& proof,auto& target){
   return target.watch_source(proof.inputs,[&](auto const& source){
    for(auto const& value:source.values)if(!target.watch(R::Domain::world,value.first)||!target.watch(R::Domain::flow,value.first))return false;return true;});},exact));
 }
 proofs.finish();assert(checks==1049&&proofs.sources.size()==1049&&proofs.dependency_lists.size()==1);
 assert(proofs.dependency_lists.begin()->second.keys.size()==570&&proofs.dependency_lists.begin()->second.refs==1049);
 assert(proofs.dependency_lists.begin()->second.keys.capacity()==1024&&proofs.bytes()<1024u*1024u);
 auto checked=checks;revisions.touch(R::Domain::world,42);
 assert(proofs.validate(revisions,exact,[](auto){return 1;})&&checks==checked+1049);
 checked=checks;revisions.touch(R::Domain::flow,42);
 assert(proofs.validate(revisions,exact,[](auto){return 1;})&&checks==checked+1049);
 proofs.clear();assert(proofs.sources.empty()&&proofs.dependency_lists.empty()&&proofs.bytes()==sizeof(proofs));
}
''')

    def test_shadow_proof_and_page_metadata_share_existing_joint_admission(self):
        from Renderer.native.test_shared_instance_submission import GPU_STUB
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        method=source[source.index('    bool shadow_metadata_admit('):source.index('    AtlasInputs::Key caster_key(')]
        run_cpp(GPU_STUB+r'''
#include "Renderer/native/render_core/shadow_page_contents.h"
struct Proof {};
struct State {using AtlasInputs=c3x_renderer::render_core::ShadowCasterProofs<Proof>;
 struct {Owner shared_instances;}renderer;Owner::CpuLease page_metadata;
 AtlasInputs atlas_inputs;c3x_renderer::render_core::ShadowPageContents<AtlasInputs::Key> page_contents;
'''+method+r'''
};
int main(){State state;using R=c3x_renderer::render_core::RasterDependencyRevisions;R revisions;
 auto pressure=state.renderer.shared_instances.retain_metadata(Owner::budget-2048);assert(pressure);
 auto begin=[&]{state.atlas_inputs.begin({1,2,3,1},revisions,[&](auto bytes){return state.shadow_metadata_admit(bytes,state.page_contents.bytes());});};
 auto proof=std::make_shared<Proof>();auto add=[&]{return state.atlas_inputs.add(1,proof,1,1,[](auto const&,auto& inputs){return inputs.watch(R::Domain::visibility,1);},[](auto const&){return true;});};
 begin();assert(!add()&&state.atlas_inputs.producers.empty()&&!state.page_metadata);pressure.reset();begin();assert(add());state.atlas_inputs.finish();
 auto& pages=state.page_contents;c3x_renderer::render_core::ShadowSamplingGrid grid;grid.valid=true;grid.low={0,0};grid.count={1,1};grid.quality_span={40,40};
 pages.begin_incremental(grid,{1,2,3,1},{});assert(pages.update(State::AtlasInputs::Key{1},grid,[]{return std::array<float,4>{1,1,2,2};},
  [&](auto bytes){return state.shadow_metadata_admit(state.atlas_inputs.bytes(),bytes);}));assert(pages.finish_incremental(grid,true,[](auto){return true;}));
 assert(state.shadow_metadata_admit(state.atlas_inputs.bytes(),pages.bytes()));
 assert(state.page_metadata->bytes()==state.atlas_inputs.bytes()+pages.bytes()+sizeof(Owner::CpuAllocation)&&state.renderer.shared_instances.bytes()==state.page_metadata->bytes());
 auto old=state.page_metadata->bytes();assert(!state.shadow_metadata_admit(State::AtlasInputs::limit,1)&&state.page_metadata->bytes()==old);
 state.atlas_inputs.clear();pages.clear();state.page_metadata.reset();assert(!state.renderer.shared_instances.bytes());
}
''')

    def test_production_region_batches_retain_unchanged_regions_and_reject_stale_capture_bytes(self):
        from Renderer.native.test_shared_instance_submission import GPU_STUB
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        fields=source[source.index('    struct TerrainBatch {'):source.index('    std::uint64_t terrain_batch_builds=')]
        function=source[source.index('    void batch_terrain_casters() {'):source.index('    bool bind_cutout(')]
        key='    AtlasInputs::Key caster_key('+source.split('    AtlasInputs::Key caster_key(',1)[1].split('    bool atlas_dependencies(',1)[0]
        stub=GPU_STUB.replace('struct ID3D11Buffer {std::vector<unsigned char> data;unsigned* freed;',
            'struct ID3D11Buffer {std::vector<unsigned char> data;unsigned* freed; void GetDesc(D3D11_BUFFER_DESC* d){d->ByteWidth=unsigned(data.size());}')
        stub=stub.replace('void Unmap(ID3D11Buffer*,UINT){}',
            'void Unmap(ID3D11Buffer*,UINT){} void CopyResource(ID3D11Buffer* a,ID3D11Buffer* b){a->data=b->data;}')
        stub=stub.replace('struct D3D11_SUBRESOURCE_DATA {void const* pSysMem=nullptr;};',
            'struct D3D11_SUBRESOURCE_DATA {void const* pSysMem=nullptr;unsigned pitch=0,slice=0;};')
        stub=stub.replace('bool fail=false,view_fail=false;', 'bool fail=false,view_fail=false,fail_index=false;unsigned failed_creates=0;')
        stub=stub.replace('++creates;if(fail)return -1;', '++creates;if(fail || (fail_index && d->BindFlags==94)){++failed_creates;return -1;}')
        run_cpp(stub+r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include "Renderer/native/render_core/prepared_mesh.h"
#include <map>
#include <set>
#define SUCCEEDED(value) ((value)>=0)
enum {geometry_land=0,geometry_natural_terrain=1,geometry_natural_mountain=2,
 D3D11_USAGE_STAGING=91,D3D11_CPU_ACCESS_READ=92,D3D11_MAP_READ=93,D3D11_BIND_INDEX_BUFFER=94,DXGI_FORMAT_R16_UINT=95};
struct CachedGeometryProof {std::uint64_t tile=0;bool valid=true;};
struct CachedMeshGeneration {std::shared_ptr<CachedGeometryProof> proof=std::make_shared<CachedGeometryProof>();};
struct Shadow {struct Bounds {float low[3]{},high[3]{1,1,1};};struct Caster {
 std::uint64_t content_generation=7,version=2;unsigned layer=0,binding=~0u,count=3,vertex_offset=0,index_offset=0,stride=4;
 decltype(DXGI_FORMAT_R32_UINT) index_format=DXGI_FORMAT_R32_UINT;Bounds bounds;float offset[3]{};
 ID3D11Buffer *vertices=nullptr,*indices=nullptr;std::vector<int> const* instances=nullptr;float instance_material=40;bool rigid=false;
};};
struct Membership {struct Content {std::map<std::uint64_t,std::shared_ptr<CachedMeshGeneration>> entries;
 std::shared_ptr<void> get(std::array<std::uint64_t,2> key){auto found=entries.find(key[1]);return found==entries.end()?nullptr:found->second;}}content;};
struct Renderer {
 struct Tile {struct {int tile_x=0,tile_y=0;}appearance;};
 struct Topology {std::map<std::uint64_t,Tile> tiles;Tile const* retained(std::uint64_t id){auto i=tiles.find(id);return i==tiles.end()?nullptr:&i->second;}unsigned scope_sequence(){return 1;}}topology_cache;
 ID3D11Device* device;ID3D11DeviceContext* context;Owner shared_instances;
 unsigned content_revision=1,device_generation=1,frame_content_uploads=0;std::size_t fresh_shadow_working_bytes=0,frame_upload_bytes=0;
 std::map<std::tuple<int,int,unsigned>,c3x_renderer::render_core::PreparedMesh> sandbox_shadow_meshes;
 bool raster_content_valid(CachedGeometryProof const& proof){return proof.valid;}
};
struct State {using Submission=Owner;using AtlasInputs=c3x_renderer::render_core::RasterContributors<CachedGeometryProof,20>;
 Renderer& renderer;std::shared_ptr<Membership> caster_lease=std::make_shared<Membership>();std::vector<Shadow::Caster> casters;
 struct Work {struct {unsigned copies=0;}calls;void upload(std::size_t,unsigned){}}counts;Work* work=&counts;
 std::uint64_t terrain_batch_builds=0,terrain_batch_reuses=0;
 template<class T>static void drop(T*& value){if(value)value->Release();value=nullptr;}
 std::size_t bytes()const{std::size_t n=0;for(auto const& x:terrain_batches)n+=x.second->gpu_bytes;return n;}
'''+fields+key+function+r'''
};
int main(){ID3D11Device device;ID3D11DeviceContext context{&device};Renderer renderer{ {},&device,&context};State state{renderer};
 std::vector<ID3D11Buffer*> source_buffers;
 auto make=[&](unsigned generation,int x,int y,bool capture){
  auto proof=std::make_shared<CachedMeshGeneration>();proof->proof->tile=generation;state.caster_lease->content.entries[generation]=proof;
  renderer.topology_cache.tiles[generation].appearance={x,y};Shadow::Caster caster;caster.content_generation=generation;caster.bounds.low[0]=float(x);caster.bounds.high[0]=float(x+1);
  c3x_renderer::render_core::PreparedMesh cpu;cpu.vertex_stride=4;cpu.index_stride=4;cpu.index_count=3;cpu.vertices.resize(8);cpu.indices.resize(12);
  float data[]={float(generation),float(generation+1)};unsigned index[]={0,1,0};std::memcpy(cpu.vertices.data(),data,8);std::memcpy(cpu.indices.data(),index,12);
  D3D11_BUFFER_DESC desc{};desc.ByteWidth=8;D3D11_SUBRESOURCE_DATA initial{cpu.vertices.data()};assert(!device.CreateBuffer(&desc,&initial,&caster.vertices));
  desc.ByteWidth=12;initial.pSysMem=cpu.indices.data();assert(!device.CreateBuffer(&desc,&initial,&caster.indices));
  source_buffers.push_back(caster.vertices);source_buffers.push_back(caster.indices);
  if(capture)renderer.sandbox_shadow_meshes[{x,y,0}]=cpu;return caster;
 };
 auto a=make(1,0,0,true),b=make(2,8,0,true),c=make(3,16,0,true);state.casters={a,b,c};state.batch_terrain_casters();
 assert(state.terrain_batches.size()==3 && state.terrain_batch_builds==3 && state.bytes()>0);
 auto retained_b=state.terrain_batches.at({1,0,0}).get(),retained_c=state.terrain_batches.at({2,0,0}).get();
 auto buffer_b=retained_b->vertices,buffer_c=retained_c->vertices;assert(buffer_b && buffer_c);
 auto entering=make(4,2,0,false);state.casters={a,b,c,entering};state.batch_terrain_casters();
 assert(state.terrain_batch_builds==4 && state.terrain_batch_reuses==2 && state.terrain_batches.size()==3);
 assert(state.terrain_batches.at({1,0,0}).get()==retained_b && state.terrain_batches.at({2,0,0}).get()==retained_c);
 assert(state.terrain_batches.at({1,0,0})->vertices==buffer_b && state.terrain_batches.at({2,0,0})->vertices==buffer_c);
 assert(!state.terrain_batches.at({0,0,0})->vertices); // no capture: exact resident fallback
 auto freed=device.freed;state.casters={a,b,entering};state.batch_terrain_casters();
 assert(state.terrain_batches.size()==2 && device.freed==freed+2 && state.terrain_batch_reuses==4);
 // A stale CPU observation with the same bounds/count cannot replace the
 // authoritative resident bytes. Its rejected scratch/merged owners retire.
 auto changed=make(5,8,0,true);renderer.sandbox_shadow_meshes[{8,0,0}].vertices[0]^=1;
 state.casters={a,changed,entering};state.batch_terrain_casters();assert(!state.terrain_batches.at({1,0,0})->vertices);
 ++renderer.content_revision;state.casters={a,changed,entering};auto builds=state.terrain_batch_builds;state.batch_terrain_casters();assert(state.terrain_batch_builds==builds+2);
 ++renderer.device_generation;state.casters={a,changed,entering};builds=state.terrain_batch_builds;state.batch_terrain_casters();assert(state.terrain_batch_builds==builds+2);
 // A successful vertex upload is counted even if its following index
 // allocation is rejected. The optional batch returns exact resident draws.
 auto failed=make(6,24,0,true);device.fail_index=true;auto uploads=renderer.frame_content_uploads;auto bytes=renderer.frame_upload_bytes;
 state.casters={a,changed,entering,failed};state.batch_terrain_casters();device.fail_index=false;
 assert(renderer.frame_content_uploads==uploads+1 && renderer.frame_upload_bytes==bytes+8);
 assert(!state.terrain_batches.at({3,0,0})->vertices && !state.terrain_batches.at({3,0,0})->gpu_charge);
 auto pressure=renderer.shared_instances.retain_metadata(Owner::budget-renderer.shared_instances.bytes()-256);assert(pressure);
 // Admission rollback leaves all original current resident draws intact.
 state.casters.clear();for(unsigned i=0;i<100;++i)state.casters.push_back(make(100+i,int(i*8),8,false));auto original=state.casters.size();state.batch_terrain_casters();
 assert(state.terrain_batches.empty() && state.casters.size()==original && !state.terrain_metadata);
 pressure.reset();state.caster_lease.reset();for(auto* buffer:source_buffers)buffer->Release();assert(device.creates==device.freed+device.failed_creates && !renderer.shared_instances.bytes());
}
''')

    def test_production_visibility_renewal_and_optional_page_refusal_preserve_exact_source_owners(self):
        from Renderer.native.test_shared_instance_submission import GPU_STUB
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        metadata=source[source.index('    bool shadow_metadata_admit('):source.index('    AtlasInputs::Key caster_key(')]
        dependencies=source[source.index('    bool atlas_dependencies('):source.index('    std::array<std::uint64_t,9> receiver_identity(')]
        fallback=source[source.index('        if(!proved){page_contents.clear();'):source.index('        casters.clear();',source.index('        if(!proved){page_contents.clear();'))]
        run_cpp(GPU_STUB+r'''
#include "Renderer/native/render_core/shadow_page_contents.h"
#include <chrono>
#include <map>
#define C3X_RENDERER64_FRESH
using R=c3x_renderer::render_core::RasterDependencyRevisions;
struct Input {std::vector<unsigned> values;};
struct CachedGeometryProof {std::uint64_t tile=1;unsigned value=7;std::weak_ptr<Input> input;};
struct CachedMeshGeneration {std::shared_ptr<CachedGeometryProof> proof;};
struct Membership {struct Content {std::map<std::uint64_t,std::shared_ptr<CachedMeshGeneration>> entries;
 std::shared_ptr<void> get(std::array<std::uint64_t,2> key){auto i=entries.find(key[1]);return i==entries.end()?nullptr:i->second;}}content;};
struct Renderer {
 struct Tile {std::uint64_t visibility_revision=3;};
 struct Topology {std::map<std::uint64_t,Tile> tiles;std::uint64_t scope=1;
  Tile const* retained(std::uint64_t tile){auto i=tiles.find(tile);return i==tiles.end()?nullptr:&i->second;}
  auto scope_sequence()const{return scope;}}topology_cache;
 unsigned content_revision=2,device_generation=3,geometry_canonical_world=1,current=7,checks=0;
 Owner shared_instances;R raster_dependency_revisions;
 bool raster_content_valid(CachedGeometryProof const& proof){++checks;return proof.value==current;}
 template<class Inputs>bool watch_raster_dependencies(CachedGeometryProof const& proof,Inputs& owner){
  auto input=proof.input.lock();return owner.watch(R::Domain::visibility,proof.tile)&&owner.watch(R::Domain::semantic,proof.tile)&&
   owner.watch_source(input,[&](auto const& source){for(auto id:source.values)if(!owner.watch(R::Domain::world,id))return false;return true;});
 }
};
struct State {
 using AtlasInputs=c3x_renderer::render_core::ShadowCasterProofs<CachedGeometryProof>;
 using Pages=c3x_renderer::render_core::ShadowPageContents<AtlasInputs::Key>;
 Renderer renderer;AtlasInputs atlas_inputs;Pages page_contents;Owner::CpuLease page_metadata;
 std::uint64_t proof_membership_signature=~std::uint64_t(0),caster_signature=1;
 struct Caster {std::uint64_t content_generation=1;};std::vector<Caster> caster_inputs{{1}};
 std::shared_ptr<Membership> caster_lease=std::make_shared<Membership>();
 c3x_renderer::render_core::ShadowSamplingGrid sampling_grid;
'''+metadata+dependencies+r'''
 void fallback(bool dependency_proved,bool proved){Pages::Context page_context{1,2,3,1};
'''+fallback+r'''
 }
};
int main(){
 State state;state.renderer.topology_cache.tiles[1]={3};
 auto input=std::make_shared<Input>();input->values={9,10,11};std::weak_ptr<Input> live=input;
 auto mesh=std::make_shared<CachedMeshGeneration>();mesh->proof=std::make_shared<CachedGeometryProof>();mesh->proof->input=input;
 state.caster_lease->content.entries[1]=mesh;
 assert(state.atlas_dependencies(true)&&state.atlas_dependencies(false));
 auto registrations=state.atlas_inputs.validation_counts.proof_registrations;
 auto watches=state.atlas_inputs.validation_counts.dependency_watch_calls,expansions=state.atlas_inputs.validation_counts.source_expansions;
 // A rejected whole-atlas visibility snapshot must enter actual registration
 // renewal even when the caster membership and all exact owners are unchanged.
 state.renderer.topology_cache.tiles[1].visibility_revision=4;
 state.renderer.raster_dependency_revisions.touch(R::Domain::visibility,1);
 assert(!state.atlas_dependencies(false)&&!state.atlas_inputs.valid_all);
 auto checks=state.renderer.checks;assert(state.atlas_dependencies(true)&&state.renderer.checks==checks+1);
 assert(state.atlas_inputs.producers.at(1).visibility==4&&state.atlas_inputs.valid_all);
 assert(state.atlas_inputs.validation_counts.proof_registrations==registrations&&state.atlas_inputs.validation_counts.dependency_watch_calls==watches&&state.atlas_inputs.validation_counts.source_expansions==expansions);
 input.reset();assert(!live.expired());
 state.sampling_grid.valid=true;state.sampling_grid.low={0,0};state.sampling_grid.count={1,1};state.sampling_grid.quality_span={40,40};
 auto& pages=state.page_contents;State::Pages::Context context{1,2,3,1};
 pages.begin_incremental(state.sampling_grid,context,{});
 auto admit=[&](auto bytes){return state.shadow_metadata_admit(state.atlas_inputs.bytes(),bytes);};
 assert(pages.update(State::AtlasInputs::Key{1,1},state.sampling_grid,[]{return std::array<float,4>{1,1,2,2};},admit));
 assert(pages.finish_incremental(state.sampling_grid,true,admit)&&pages.complete_incremental(0));
 assert(state.shadow_metadata_admit(state.atlas_inputs.bytes(),pages.bytes()));
 auto spare=Owner::budget-state.renderer.shared_instances.bytes();
 auto pressure=state.renderer.shared_instances.retain_metadata(spare-sizeof(Owner::CpuAllocation)-128);assert(pressure);
 assert(!pages.update(State::AtlasInputs::Key{1,2},state.sampling_grid,[]{return std::array<float,4>{1,1,2,2};},admit));
 // Optional page failure invalidates every page; current exact dependencies
 // and their strong source owners survive only under the minimum joint charge.
 state.fallback(true,false);assert(pages.occurrences.empty()&&!pages.reused[0]&&!pages.pages[pages.slots[0]].valid);
 assert(state.proof_membership_signature==state.caster_signature&&state.atlas_inputs.producers.size()==1&&state.atlas_inputs.sources.size()==1&&!live.expired());
 assert(state.page_metadata&&state.page_metadata->bytes()==state.atlas_inputs.bytes()+pages.bytes()+sizeof(Owner::CpuAllocation));
 assert(state.renderer.shared_instances.bytes()<=Owner::budget&&state.renderer.shared_instances.peak_bytes()<=Owner::budget);
 for(unsigned n=0;n<100;++n)assert(state.atlas_dependencies(false));
 assert(state.atlas_inputs.validation_counts.proof_registrations==registrations&&state.atlas_inputs.validation_counts.dependency_watch_calls==watches&&state.atlas_inputs.validation_counts.source_expansions==expansions);
 pressure.reset();pages.begin_incremental(state.sampling_grid,context,{});
 assert(pages.update(State::AtlasInputs::Key{1,2},state.sampling_grid,[]{return std::array<float,4>{1,1,2,2};},admit)&&pages.finish_incremental(state.sampling_grid,true,admit));
 assert(!pages.reused[0]);
 // A genuine content mismatch has no fallback proof authority. It clears
 // registrations/charge and releases the last immutable source-owner pin.
 state.renderer.current=8;state.renderer.raster_dependency_revisions.touch(R::Domain::semantic,1);
 assert(!state.atlas_dependencies(false));state.fallback(false,false);
 assert(state.atlas_inputs.producers.empty()&&state.atlas_inputs.sources.empty()&&live.expired()&&!state.page_metadata);
 assert(state.proof_membership_signature==~std::uint64_t(0)&&!state.renderer.shared_instances.bytes());
 // Even the empty retained owner may not bypass admission if the shared
 // ledger is occupied entirely by a separate live owner.
 auto full=state.renderer.shared_instances.retain_metadata(Owner::budget-sizeof(Owner::CpuAllocation));assert(full);
 state.fallback(true,false);assert(!state.page_metadata&&state.atlas_inputs.producers.empty());full.reset();
 assert(!state.renderer.shared_instances.bytes());
}
''')

    def test_production_whole_atlas_gate_rejects_incomplete_draw_and_retry(self):
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        start=source.index('        unsigned reuse_failures=unsigned(!atlas_complete)')
        end=source.index('        if(work->enabled)++work->row().rebuilds;',start)
        gate=source[start:end]
        reusable=source[source.index('    bool atlas_reusable(unsigned reuse_failures){'):source.index('    template<class BodyInputs,class RetireCompletedPlans> bool render(')]
        run_cpp(r'''
#include <array>
#include <cassert>
#include <memory>
#include <cstdint>
struct State {
 bool atlas_complete=false;std::shared_ptr<int> shadow_front=std::make_shared<int>(1);
 std::uint64_t prepared_signature=3,signature=7;
 std::array<float,4> wrap_basis{};std::array<float,12> light_basis{};
 struct {struct {bool valid(std::shared_ptr<int> const& lease){return bool(lease);}}shared_instances;
  std::array<float,12> shadow_basis{};bool borrowed_scene_frame=false;}renderer;
 struct Grid {bool covers(Grid const&)const{return true;}}sampling_grid;
 struct Work {bool enabled=false;struct {unsigned reuses=0;}count;auto& row(){return count;}}counts;
 Work* work=&counts;unsigned validations=0;
 bool atlas_dependencies(bool){++validations;return true;}
'''+reusable+r'''
 bool reused(std::uint64_t membership,std::uint64_t scene,std::array<float,4> wrap_query,Grid query_grid){
'''+gate+r'''
 return false;
 }
};
int main(){State state;State::Grid grid;std::array<float,4> wrap{};
 assert(!state.reused(3,7,wrap,grid) && !state.validations);
 state.atlas_complete=true;assert(state.reused(3,7,wrap,grid) && state.validations==1);
 // All high-level identities still match after a page draw fails. Actual
 // production completion gating must suppress this otherwise valid fast path.
 state.atlas_complete=false;assert(!state.reused(3,7,wrap,grid) && state.validations==1);
 state.atlas_complete=true;assert(state.reused(3,7,wrap,grid) && state.validations==2);
}
''')
        body=source[source.index('        atlas_complete=false;',start):source.index('\n};',start)]
        self.assertLess(body.index('page_contents.complete_incremental('),body.index('prepared_signature=membership;'))
        self.assertLess(body.index('UpdateSubresource(renderer.source_shadow.table'),body.index('atlas_complete=std::all_of('))

    def test_production_sampling_uses_retained_physical_slots_and_asset_context(self):
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        self.assertIn('int layer=int(pickup_pages[3+logical].x);',source)
        self.assertIn('table[3+slot][0]=page_contents.reused[slot]?float(page_contents.slots[slot]):-1.f;',source)
        self.assertIn('if(layer<0)return -1e6;',source)
        self.assertIn('auto* target=targets[page_contents.slots[page_slot]];',source)
        self.assertIn('if(page_contents.reused[page_slot])continue;',source)
        self.assertIn('Pages::Context page_context={renderer.topology_cache.scope_sequence(),renderer.content_revision,',source)
        self.assertIn('shadow_front->view',source)
        self.assertIn('renderer.shared_instances.valid(front)?front:Submission::Lease{}',source)


if __name__=='__main__':
    unittest.main()
