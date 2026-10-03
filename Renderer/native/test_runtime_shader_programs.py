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
#include "Renderer/native/render_core/process_environment.h"
#include "Renderer/sandbox/terrain_material_fast.h"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
struct State {
 ID3D11Device* device=nullptr;bool city_profile=true,environment_profile=true;
 std::string shader_root,fidelity_root;
 struct Trace {void write(char const* stage,char const* message,bool){std::printf("%s %s\n",stage,message);}} trace;
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
 renderer.device->Release();return 0;
}
'''


@unittest.skipUnless(os.environ.get('C3X_RENDERER_GPU_TESTS') == '1',
                     'requires the prepared local shader pack and Windows GPU')
class RuntimeShaderPrograms(unittest.TestCase):
    def test_current_adapters_compile(self):
        run_cpp(program(), timeout=240)
