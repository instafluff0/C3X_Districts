"""Exact resource-free shadow-page ownership and production mapping contracts."""
from pathlib import Path
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class ShadowPageContentsTests(unittest.TestCase):
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

    def test_production_whole_atlas_gate_rejects_incomplete_draw_and_retry(self):
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        start=source.index('        if (atlas_complete && renderer.shared_instances.valid(shadow_front)')
        end=source.index('        if(work->enabled)++work->row().rebuilds;',start)
        gate=source[start:end]
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
  std::array<float,12> shadow_basis{};}renderer;
 struct Grid {bool covers(Grid const&)const{return true;}}sampling_grid;
 struct Work {bool enabled=false;struct {unsigned reuses=0;}count;auto& row(){return count;}}counts;
 Work* work=&counts;unsigned validations=0;
 bool atlas_dependencies(bool){++validations;return true;}
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
        self.assertLess(body.index('page_contents.complete('),body.index('prepared_signature=membership;'))
        self.assertLess(body.index('UpdateSubresource(renderer.source_shadow.table'),body.index('atlas_complete=true;'))

    def test_production_sampling_uses_retained_physical_slots_and_asset_context(self):
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        self.assertIn('int layer=int(pickup_pages[3+logical].x);',source)
        self.assertIn('table[3+slot][0]=float(page_contents.slots[slot])',source)
        self.assertIn('auto* target=targets[page_contents.slots[page_slot]];',source)
        self.assertIn('if(page_contents.reused[page_slot])continue;',source)
        self.assertIn('Pages::Context page_context={renderer.topology_cache.scope_sequence(),renderer.content_revision,',source)
        self.assertIn('shadow_front->view',source)
        self.assertIn('renderer.shared_instances.valid(front)?front:Submission::Lease{}',source)


if __name__=='__main__':
    unittest.main()
