"""Immutable generations, weak history, bounded leases, reset and wrap identity."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT=Path(__file__).resolve().parents[2]


class ResidentMeshOwnership(unittest.TestCase):
    def test_generation_outlives_metadata_without_stale_lookup_or_leaks(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        generation=source[source.index('struct CachedMeshGeneration {'):source.index('\nstruct CachedTileGeometry {')]
        run_cpp(r'''
#include "Renderer/native/render_core/resident_content.h"
#include <array>
#include <vector>
#include <cassert>
#include <thread>
constexpr unsigned geometry_layer_count=2;
struct Resource {unsigned references=1;void Release(){assert(references);--references;}};
struct CachedVertexChunk {Resource* buffer=nullptr;Resource* indices=nullptr;};
struct CachedGeometryProof {};
''' + generation + r'''
struct Metadata {std::shared_ptr<CachedMeshGeneration> mesh=std::make_shared<CachedMeshGeneration>();};
using namespace c3x_renderer::render_core;
int main(){
 auto ledger=std::make_shared<ResidentRetirement>();ResidentContent<Metadata> registry(2);
 Resource resource,indices;Metadata old;old.mesh->layers[0].push_back({&resource,&indices});
 auto stale=registry.bind(old,old.mesh);ResidentSelection active(ledger);
 for(unsigned occurrence=0;occurrence<8192;++occurrence)assert(active.retain(stale,registry.lease(stale)));
 assert(active.size()==1 && resource.references==1 && indices.references==1);
 auto weak_history=stale;auto pending=active;auto shared_selection_bytes=ledger->bytes.load();
 registry.release(stale);assert(!registry.resolve(weak_history) && !registry.lease(weak_history));
 old.mesh->retirement.retire(ledger,600);old.mesh.reset();
 assert(resource.references==1 && ledger->bytes==shared_selection_bytes+600);
 Metadata replacement;auto fresh=registry.bind(replacement,replacement.mesh);
 assert(fresh.slot==stale.slot && fresh.generation!=stale.generation);
 assert(!registry.resolve(stale) && registry.resolve(fresh)==&replacement);
 assert(pending.retain(fresh,registry.lease(fresh))); // supersession detaches selection metadata
 assert(active.size()==1 && pending.size()==2);
 active.clear();assert(resource.references==1);
 // Cancellation drops pending leases; the immutable old buffer retires once.
 pending.clear();assert(!resource.references && !indices.references && ledger->bytes==0);
 assert(registry.resolve(fresh)==&replacement);
 ResidentSelection selected(ledger);assert(selected.retain(fresh,registry.lease(fresh)));
 registry.clear();assert(!registry.resolve(fresh) && !registry.lease(fresh));
 replacement.mesh->retirement.retire(ledger,700);replacement.mesh.reset();assert(ledger->bytes>=700);
 std::thread release([lease=std::move(selected)]()mutable{lease.clear();});release.join();assert(ledger->bytes==0);
 // Device/world/viewer resets never revive a weak generation.
 for(unsigned view=0;view<1000;++view){
  Metadata tile;auto handle=registry.bind(tile,tile.mesh);assert(handle.generation>fresh.generation);
  ResidentSelection camera(ledger);assert(camera.retain(handle,registry.lease(handle)));
  registry.release(handle);tile.mesh->retirement.retire(ledger,4096);tile.mesh.reset();
  assert(ledger->bytes>=4096);camera.clear();assert(ledger->bytes==0);
 }
 assert(registry.bytes()>0 && ledger->peak>=4096);
}
''')

    def test_pending_epoch_and_old_view_retirement_use_the_same_budget(self):
        run_cpp(r'''
#include "Renderer/native/render_core/residency_candidates.h"
#include <unordered_map>
#include <cassert>
using namespace c3x_renderer::render_core;
struct Payload {unsigned* freed;ResidentRetirementToken charge;explicit Payload(unsigned* p):freed(p){}~Payload(){++*freed;}};
struct Item {ContentHandle binding;unsigned last_used=0;std::shared_ptr<Payload> mesh;};
int main(){
 auto ledger=std::make_shared<ResidentRetirement>();ResidentContent<Item> registry(3);
 std::unordered_map<int,Item> cache;unsigned freed[3]={};
 for(int i=0;i<3;++i){auto& tile=cache[i];tile.mesh=std::make_shared<Payload>(&freed[i]);tile.binding=registry.bind(tile,tile.mesh);}
 cache[0].last_used=2;cache[1].last_used=5;cache[2].last_used=1;
 ResidentSelection displayed(ledger);auto stale=cache[0].binding;
 assert(displayed.retain(stale,registry.lease(stale)));
 ResidencyCandidates order;auto priority=[](auto const&){return false;};
 auto victim=order.next(cache,registry,5,priority);assert(victim==cache[2].binding);
 registry.release(victim);cache.erase(2);assert(freed[2]==1);
 victim=order.next(cache,registry,5,priority);assert(victim==stale);
 // Free weak cache metadata and proofs while the displayed generation remains
 // valid. Its retained payload stays charged; the pending epoch cannot evict.
 cache[0].mesh->charge.retire(ledger,600);registry.release(stale);cache.erase(0);
 assert(!registry.resolve(stale) && freed[0]==0 && ledger->bytes>=600);
 assert(!order.next(cache,registry,5,priority).generation);
 displayed.clear();assert(freed[0]==1 && ledger->bytes==0);
 registry.release(cache[1].binding);cache.clear();assert(freed[1]==1);
}
''')

    def test_canonical_content_keeps_occurrence_and_native_representation(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        body='auto content_tile_for='+source.split('auto content_tile_for=',1)[1].split('        auto select_river_nodes=',1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
int main(){
 c3x_renderer_frame_v1 frame={};frame.world_width_tiles=100;frame.world_height_tiles=80;frame.world_wrap_x=1;
 bool canonical_world_content=true;
 auto canonical_component=[](int value,int extent,unsigned wraps){if(!wraps)return value;auto x=value%extent;return x<0?x+extent:x;};
''' + body + r'''
 c3x_renderer_tile_v1 occurrence={};occurrence.tile_x=-176;occurrence.tile_y=56;occurrence.anchor_x=384;occurrence.anchor_y=192;
 auto world=content_tile_for(occurrence);assert(world.tile_x==24 && world.tile_y==56);
 assert(world.anchor_x==384 && occurrence.tile_x==-176);assert(world.anchor_y==192);
 canonical_world_content=false;assert(content_tile_for(occurrence).tile_x==-176); // native 64 remains distinct
 canonical_world_content=true;frame.world_wrap_x=0;assert(content_tile_for(occurrence).tile_x==-176);
}
''')

    def test_river_receiver_uses_queried_neighbor_basis_and_captured_anchor(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        assignment=source.split('owner_record=receiving->occurrence;',1)[1].split('local_u=u;',1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
int main(){
 c3x_renderer_tile_v1 owner_record={};owner_record.tile_x=-100;owner_record.tile_y=56;
 owner_record.anchor_x=1376;owner_record.anchor_y=624;
 bool canonical_world_content=true;
 // The bank lies just over the canonical seam. Modulo would incorrectly
 // move it 100 tiles away from its generating tile at x=99.
 int c=78,r=22;
''' + assignment + r'''
 assert(owner_record.tile_x==100 && owner_record.tile_y==56);
 assert(owner_record.anchor_x==1376 && owner_record.anchor_y==624);
 assert((owner_record.tile_x+owner_record.tile_y)*.5f==78);
}
''')

    def test_shadow_query_wrap_is_continuous_and_preserves_nonshadow_coordinates(self):
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        helper='float3 sandbox_shadow_world('+source.split('float3 sandbox_shadow_world(',1)[1].split('float c3x_paged_visibility(',1)[0]
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cassert>
#include <cmath>
#include <cstring>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
char const* shader=R"shader(
cbuffer Table : register(b4) {float4 pickup_pages[64];};
''' + helper + r'''
StructuredBuffer<float4> Src : register(t0);
RWStructuredBuffer<float4> Dst : register(u0);
[numthreads(1,1,1)]void CS(uint3 id:SV_DispatchThreadID){Dst[id.x]=float4(sandbox_shadow_world(Src[id.x].xyz),Src[id.x].w);}
)shader";
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL feature;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context)));
 float input[8][4]={{49.5f,49.5f,.3f,1},{.5f,.5f,.3f,1},{-50.5f,-50.5f,.3f,1},
  {100.5f,100.5f,.3f,1},{40.5f,-39.5f,.3f,1},{-39.5f,40.5f,.3f,1},
  {35.f,35.f,.3f,1},{30.f,-30.f,.3f,1}};
 D3D11_BUFFER_DESC d={};d.ByteWidth=sizeof(input);d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 d.Usage=D3D11_USAGE_IMMUTABLE;d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride=16;
 D3D11_SUBRESOURCE_DATA data={input,0,0};ComPtr<ID3D11Buffer> src,dst,table,read;
 assert(SUCCEEDED(device->CreateBuffer(&d,&data,&src)));d.BindFlags=D3D11_BIND_UNORDERED_ACCESS;d.Usage=D3D11_USAGE_DEFAULT;
 assert(SUCCEEDED(device->CreateBuffer(&d,nullptr,&dst)));
 D3D11_SHADER_RESOURCE_VIEW_DESC sv={};sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;sv.Buffer.NumElements=8;
 ComPtr<ID3D11ShaderResourceView> srv;assert(SUCCEEDED(device->CreateShaderResourceView(src.Get(),&sv,&srv)));
 D3D11_UNORDERED_ACCESS_VIEW_DESC uv={};uv.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;uv.Buffer.NumElements=8;
 ComPtr<ID3D11UnorderedAccessView> uav;assert(SUCCEEDED(device->CreateUnorderedAccessView(dst.Get(),&uv,&uav)));
 d={};d.ByteWidth=64*16;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;assert(SUCCEEDED(device->CreateBuffer(&d,nullptr,&table)));
 d={};d.ByteWidth=sizeof(input);d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 assert(SUCCEEDED(device->CreateBuffer(&d,nullptr,&read)));
 ComPtr<ID3DBlob> code,errors;assert(SUCCEEDED(D3DCompile(shader,std::strlen(shader),nullptr,nullptr,nullptr,"CS","cs_5_0",0,0,&code,&errors)));
 ComPtr<ID3D11ComputeShader> cs;assert(SUCCEEDED(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&cs)));
 for(unsigned mode=0;mode<8;++mode){
  float constants[64][4]={};constants[1][0]=(mode&4)?25.f:1.f;constants[1][1]=(mode&4)?40.f:0.f;constants[1][2]=(mode&1)?100.f:0;constants[1][3]=(mode&2)?80.f:0;
  context->UpdateSubresource(table.Get(),0,nullptr,constants,0,0);
  auto t=table.Get();auto s=srv.Get();auto u=uav.Get();context->CSSetShader(cs.Get(),nullptr,0);
  context->CSSetConstantBuffers(4,1,&t);context->CSSetShaderResources(0,1,&s);context->CSSetUnorderedAccessViews(0,1,&u,nullptr);
  context->Dispatch(8,1,1);u=nullptr;context->CSSetUnorderedAccessViews(0,1,&u,nullptr);
  context->CopyResource(read.Get(),dst.Get());D3D11_MAPPED_SUBRESOURCE mapped={};assert(SUCCEEDED(context->Map(read.Get(),0,D3D11_MAP_READ,0,&mapped)));
  float expected[8][4];std::memcpy(expected,input,sizeof(input));
  for(auto& p:expected){if(mode&1){auto turn=std::floor((p[0]+p[1]-constants[1][0]+50)/100)*50;p[0]-=turn;p[1]-=turn;}
   if(mode&2){auto turn=std::floor((p[0]-p[1]-constants[1][1]+40)/80)*40;p[0]-=turn;p[1]+=turn;}}
  assert(!std::memcmp(expected,mapped.pData,sizeof(expected)));context->Unmap(read.Get(),0);
 }
}
''',timeout=60)

    def test_shadow_center_follows_authoritative_occurrence_window(self):
        source=(ROOT/'Renderer/sandbox/fresh_pipeline.h').read_text()
        center='auto center=[&]'+source.split('auto center=[&]',1)[1].split('if(dims.wrap_x',1)[0]
        run_cpp(r'''
#include <cmath>
#include <cassert>
int main(){
 float occurrence_low[2]={20,10},occurrence_high[2]={80,50};
''' + center + r'''
 assert(center(0,100)==50 && center(1,80)==30);
 // Wide ordinary receivers stay around their own world window. Seam windows
 // have the same shadow basis in either order and either wrapped occurrence.
 occurrence_low[0]=-2;occurrence_high[0]=2;assert(center(0,100)==0);
 occurrence_low[0]=198;occurrence_high[0]=202;assert(center(0,100)==0);
 occurrence_low[0]=-102;occurrence_high[0]=-98;assert(center(0,100)==0);
}
''')


if __name__=='__main__':unittest.main()
