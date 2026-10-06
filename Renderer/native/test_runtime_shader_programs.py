"""Compile the actual runtime shader adapters against the prepared local pack."""
import os
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


def program():
    source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
    # Keep the real loader, adaptations, entries and compiler. Only the owning
    # renderer state is replaced; no shader equations are copied into this test.
    methods = source[source.index('    static std::string source('):source.index('    bool install()')]
    return r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <iterator>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <sstream>
#include <wrl/client.h>
#include <d3d11shader.h>
#include "Renderer/native/render_core/process_environment.h"
#include "Renderer/sandbox/terrain_material_fast.h"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
struct State {
 ID3D11Device* device=nullptr;bool city_profile=true,environment_profile=true;
 std::string shader_root,fidelity_root;
 struct Trace {void write(char const* stage,char const* message,bool=true){std::printf("%s %s\n",stage,message);}} trace;
} renderer;
struct Programs {
''' + methods + r'''
};
int main(){
 char root[32768]={},pack[32768]={};
 std::filesystem::path checkout;
 if(GetEnvironmentVariableA("C3X_TEST_MOD_ROOT",root,sizeof(root)))checkout=root;
 else {
  checkout=std::filesystem::current_path();
  while(!std::filesystem::exists(checkout/"Renderer/sandbox/fresh_pipeline.h")){
   auto parent=checkout.parent_path();if(parent==checkout)return 2;checkout=parent;
  }
 }
 renderer.fidelity_root=checkout.string();
 renderer.shader_root=GetEnvironmentVariableA("C3X_RENDERER_SHADER_SOURCE_ROOT",pack,sizeof(pack))?
     pack:(checkout/"Renderer/packs/Renderer64ResidentRuntime").string();
 if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,
     D3D11_SDK_VERSION,&renderer.device,nullptr,nullptr)))return 3;
 struct Entry {char const* file;char const* entry;};
 Entry entries[]={
  {"hydrology.hlsl","PSIntegrated"},{"hydrology.hlsl","PSCoastalWave"},
  {"feature.hlsl","PSIntegratedFeature"},{"terrain.hlsl","PSFeature"},
  {"mountain.hlsl","PSFeature"},{"objects.hlsl","PSFeature"},
  {"city.hlsl","PSNativeCity"},{"city.hlsl","PSNativeCityEmission"},
  {"city.hlsl","PSNativeCityReflection"},{"city.hlsl","PSNativeCityReflectionEmission"},
  {"water_surface.hlsl","PSWaterSurface"},{"water_surface.hlsl","PSWaterLighting"},
  {"water_surface.hlsl","PSRiverSurface"},{"hydrology.hlsl","PSSandboxUnderlay"},
  {"terrain.hlsl","PSSandboxTerrainMaterial"},{"terrain.hlsl","PSSandboxReflectionTerrainMaterial"},
  {"mountain.hlsl","PSSandboxTerrainMaterial"},{"mountain.hlsl","PSSandboxReflectionTerrainMaterial"},
  {"terrain.hlsl","PSSandboxTerrainRelight"}};
 for(auto const& entry:entries){ID3D11PixelShader* pixel=nullptr;
  if(!Programs::compile(entry.file,entry.entry,&pixel)){renderer.device->Release();return 1;}
  pixel->Release();std::printf("SHADER_PASS %s %s\n",entry.file,entry.entry);std::fflush(stdout);
 }
 // Batched (pulled) feature vertices must link to the feature pixel shader
 // by register; D3D11 does not match interpolants by name.
 ID3D11VertexShader *pulled=nullptr,*pulled_feature=nullptr;
 if(!Programs::compile_pulled_vertex(&pulled,&pulled_feature)||!pulled||!pulled_feature){renderer.device->Release();return 4;}
 pulled->Release();pulled_feature->Release();std::printf("SHADER_PASS pulled VSIntegratedPulled VSIntegratedFeaturePulled\n");
 auto reflect=[](std::string const& text,char const* entry,char const* target,bool output,std::vector<std::string>& out){
  Microsoft::WRL::ComPtr<ID3DBlob> code,errors;
  if(FAILED(D3DCompile(text.data(),text.size(),entry,nullptr,nullptr,entry,target,0,0,&code,&errors)))return false;
  Microsoft::WRL::ComPtr<ID3D11ShaderReflection> r;
  if(FAILED(D3DReflect(code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(&r))))return false;
  D3D11_SHADER_DESC d={};r->GetDesc(&d);
  for(UINT i=0;i<(output?d.OutputParameters:d.InputParameters);++i){D3D11_SIGNATURE_PARAMETER_DESC p={};
   if(FAILED(output?r->GetOutputParameterDesc(i,&p):r->GetInputParameterDesc(i,&p)))return false;
   char line[96];std::snprintf(line,sizeof(line),"%s%u@%u",p.SemanticName,p.SemanticIndex,p.Register);out.push_back(line);}
  return true;
 };
 auto features=Programs::source("feature.hlsl");std::vector<std::string> vs,ps;
 if(!reflect(features,"VSIntegratedFeature","vs_5_0",true,vs)||!reflect(features,"PSIntegratedFeature","ps_5_0",false,ps)){renderer.device->Release();return 5;}
 for(std::size_t i=0;i<ps.size();++i)if(i>=vs.size()||vs[i]!=ps[i]){std::printf("LINK_MISMATCH %zu\n",i);renderer.device->Release();return 6;}
 std::printf("SHADER_PASS feature linkage registers=%zu\n",ps.size());
 renderer.device->Release();return 0;
}
'''


@unittest.skipUnless(os.environ.get('C3X_RENDERER_GPU_TESTS') == '1',
                     'requires the prepared local shader pack and Windows GPU')
class RuntimeShaderPrograms(unittest.TestCase):
    def test_current_adapters_compile(self):
        run_cpp(program(), timeout=240)
