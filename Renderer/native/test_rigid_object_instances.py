"""Shared rigid placements preserve the existing CPU transform and bounds."""
import unittest
import json
import os
from Renderer.lab.platform import ROOT, windows_root
from Renderer.native.native_cpp_test import run_cpp


class RigidObjectTests(unittest.TestCase):
    def test_gpu_transform_matches_cpu_pack_placements(self):
        if not (ROOT / 'Renderer/packs/ImprovementsNormalized/mine_runtime.bin').is_file():
            self.skipTest('Local generic mine pack unavailable')
        shader=(ROOT / 'Renderer/native/render_core/rigid_instance_geometry.hlsl').read_text()+r'''
StructuredBuffer<RigidInput> inputs:register(t0);
struct Result {float3 world;float3 position;float3 normal;};
RWStructuredBuffer<Result> results:register(u0);
[numthreads(1,1,1)] void CS(uint3 id:SV_DispatchThreadID){
 RigidPoint p=rigid_point(inputs[id.x]);Result o;
 o.world=p.world;o.position=p.position;o.normal=p.normal;results[id.x]=o;
}
'''
        program=r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
#include "Renderer/native/rigid_object_instance.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer;
int main(){
 FeatureBundle bundle;assert(load_feature_bundle(C3X_TEST_PACK,bundle));
 objects::Assets assets{{&bundle,&bundle,&bundle,&bundle,&bundle,&bundle}};
 struct Input {FeatureSourceVertex vertex;fidelity::MeshInstance instance;};
 struct Result {float world[3],position[3],normal[3];};
 std::vector<Input> inputs;std::vector<Result> expected;
 for(int tile:{-17,1,139})for(float scale:{.94f,1.05f,2.32f})for(float ground:{0.f,9.f,17.125f})
 for(unsigned index=0;index<bundle.assets.size();++index){
  auto const& asset=bundle.assets[index];
  auto relief=[&](float,float){return std::array<float,3>{ground,0,0};};auto height=[&](float,float){return ground+2.5f;};
  objects::Projection p;p.tile.tile_x=tile;p.tile.tile_y=3;p.tile_width=128;p.content_view_height=1192;
  p.half_w=64;p.half_h=32;p.relief_projection_scale=128.f/224.f*.82f;p.feature_projection_scale=128.f/224.f;
  p.pickup_profile=p.world_objects=true;
  objects::Instance source{objects::farm_family,index,objects::farm_layer,.5f,.5f,0.f,scale,21,.01f,false};
  auto instance=objects::prepare_rigid(source,p,assets,relief,height);
  objects::Plan plan;plan.instances.push_back(source);objects::Surfaces expanded;
  objects::compile(plan,p,assets,relief,height,expanded,true);
  for(unsigned n=0;n<asset.vertices.size();n+=17){auto const& v=expanded.layers[objects::farm_layer][n];
   inputs.push_back({asset.vertices[n],instance.instance});
   expected.push_back({{v.world_x,v.world_y,v.world_z},{v.x/128,v.y/128,v.z},{v.normal_x,v.normal_y,v.normal_z}});
  }
 }
 assert(inputs.size()<65536);
 ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context)));
 char const* source=C3X_TEST_SHADER;ID3DBlob *code=nullptr,*errors=nullptr;
 auto hr=D3DCompile(source,std::strlen(source),nullptr,nullptr,nullptr,"CS","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
 if(errors)std::printf("%s",static_cast<char const*>(errors->GetBufferPointer()));assert(SUCCEEDED(hr));
 ID3D11ComputeShader* compute=nullptr;assert(SUCCEEDED(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&compute)));
 D3D11_BUFFER_DESC desc{};desc.ByteWidth=unsigned(inputs.size()*sizeof(Input));desc.Usage=D3D11_USAGE_DEFAULT;
 desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;desc.StructureByteStride=sizeof(Input);
 D3D11_SUBRESOURCE_DATA initial{};initial.pSysMem=inputs.data();ID3D11Buffer *input=nullptr,*output=nullptr,*staging=nullptr;
 assert(SUCCEEDED(device->CreateBuffer(&desc,&initial,&input)));ID3D11ShaderResourceView* view=nullptr;
 assert(SUCCEEDED(device->CreateShaderResourceView(input,nullptr,&view)));
 desc.ByteWidth=unsigned(expected.size()*sizeof(Result));desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;desc.StructureByteStride=sizeof(Result);
 assert(SUCCEEDED(device->CreateBuffer(&desc,nullptr,&output)));ID3D11UnorderedAccessView* writable=nullptr;
 assert(SUCCEEDED(device->CreateUnorderedAccessView(output,nullptr,&writable)));
 desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=desc.MiscFlags=desc.StructureByteStride=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 assert(SUCCEEDED(device->CreateBuffer(&desc,nullptr,&staging)));
 context->CSSetShader(compute,nullptr,0);context->CSSetShaderResources(0,1,&view);context->CSSetUnorderedAccessViews(0,1,&writable,nullptr);
 context->Dispatch(unsigned(inputs.size()),1,1);context->CopyResource(staging,output);D3D11_MAPPED_SUBRESOURCE mapped{};
 assert(SUCCEEDED(context->Map(staging,0,D3D11_MAP_READ,0,&mapped)));
 unsigned mismatches[9]={};float maximum[9]={};
 for(unsigned n=0;n<inputs.size();++n)for(unsigned field=0;field<9;++field){
  float a=reinterpret_cast<float const*>(&expected[n])[field],b=static_cast<float const*>(mapped.pData)[n*9+field];
  auto delta=std::abs(a-b);if(delta){++mismatches[field];maximum[field]=std::max(maximum[field],delta);}
 }
 context->Unmap(staging,0);
 for(unsigned field=0;field<9;++field){std::printf("RIGID_TRANSFORM field=%u mismatches=%u max_delta=%.9g\n",field,mismatches[field],maximum[field]);
  assert(maximum[field]<(field<3?3e-5f:field==5?1e-5f:3e-6f));}
 context->ClearState();staging->Release();writable->Release();output->Release();view->Release();input->Release();
 compute->Release();code->Release();if(errors)errors->Release();context->Release();device->Release();
}
'''
        target=(ROOT if os.name=='nt' else windows_root()) / 'Renderer/packs/ImprovementsNormalized/mine_runtime.bin'
        run_cpp(program.replace('C3X_TEST_PACK',json.dumps(str(target))).replace('C3X_TEST_SHADER',json.dumps(shader)),
                sources=('Renderer/native/terrain_scene_runtime.cpp',),timeout=90)

    def test_source_transform_projection_material_and_exact_bounds(self):
        run_cpp(r'''
#include "Renderer/native/rigid_object_instance.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 FeatureBundle bundle;bundle.assets.resize(1);auto& asset=bundle.assets[0];asset.texture_index=2;
 asset.vertices={{{-1,.5f,.2f},{.2f,.3f,.9f},{.1f,.2f}},{{.3f,-.8f,.7f},{.7f,.1f,.2f},{.9f,.6f}},{{.5f,.5f,0},{0,0,1},{0,0}}};
 asset.indices={2,0,1};objects::Assets assets{{&bundle,&bundle,&bundle,&bundle,&bundle,&bundle}};
 assert(objects::shared_rigid_mesh(asset));auto flat=asset;
 for(auto& vertex:flat.vertices)vertex.position[2]=.002f;
 assert(!objects::shared_rigid_mesh(flat));flat.vertices[2].position[2]=.003f;
 assert(objects::shared_rigid_mesh(flat));flat.indices.clear();assert(!objects::shared_rigid_mesh(flat));
 auto relief=[](float,float){return std::array<float,3>{9,0,0};};auto height=[](float,float){return 17.f;};
 for(int tile:{-17,1,139})for(float rotation:{-.24f,0.f,1.57079632679f})for(float scale:{.85f,1.f,2.32f})
 for(auto family:{objects::mine_family,objects::site_family}){
  objects::Projection projection;projection.tile.tile_x=tile;projection.tile.tile_y=3;
  projection.tile_width=128;projection.content_view_height=1192;projection.half_w=64;projection.half_h=32;
  projection.relief_projection_scale=128.f/224.f*.82f;projection.feature_projection_scale=128.f/224.f;
  projection.pickup_profile=projection.world_objects=true;
  objects::Instance source{family,0,objects::mine_layer,.3f,.7f,rotation,scale,21,.18f,false};
  auto rigid=objects::prepare_rigid(source,projection,assets,relief,height);
  assert(rigid.material==float(asset.texture_index)+source.material+source.owner);
  objects::Plan plan;plan.instances.push_back(source);objects::Surfaces expanded;
  objects::compile(plan,projection,assets,relief,height,expanded,true);
  auto const* place=rigid.instance.place;
  for(unsigned i=0;i<asset.vertices.size();++i){auto const& s=asset.vertices[i];auto const& expected=expanded.layers[objects::mine_layer][i];
   float x=(s.position[0]*place[4]-s.position[1]*place[5])*place[6];
   float y=(s.position[0]*place[5]+s.position[1]*place[4])*place[6],z=s.position[2]*place[6];
   float h=z*150.f*(128.f/224.f)/(128.f/224.f*.82f);
   float world[]={place[0]+place[2]+x,place[1]+1-place[3]-y,(place[7]+2.5f+h)/112.f};
   assert(world[0]==expected.world_x && world[1]==expected.world_y && world[2]==expected.world_z);
   float sx=64+(place[2]-place[3])*64+(x-y)*64;
   float sy=((place[2]+place[3])*32-place[7]*(128.f/224.f*.82f))+(x+y)*32-z*150.f*(128.f/224.f);
   assert(sx==expected.x && sy==expected.y && h==expected.z);
   assert(std::floor(sx)>=rigid.bounds[0] && std::ceil(sy)<=rigid.bounds[3]);
   for(unsigned axis=0;axis<3;++axis)assert(world[axis]>=rigid.low[axis] && world[axis]<=rigid.high[axis]);
  }
 }
}
''')


if __name__ == '__main__':
    unittest.main()
