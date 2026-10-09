"""The memory census sizes live D3D11 resources from their descriptions.

Block-compressed and linear formats, mip chains, texture arrays and multisample
counts are charged exactly, and a resource reached through several views or
owners counts once (performance goals, G1: Civ VI memory direction).
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ResourceCensusTests(unittest.TestCase):
    def test_sizes_formats_mips_arrays_and_counts_each_resource_once(self):
        run_cpp(r'''
#include <windows.h>
#include "Renderer/native/render_core/resource_census.h"
#include <cassert>
#include <cstdio>
#pragma comment(lib,"d3d11.lib")
using c3x_renderer::render_core::ResourceCensus;
int main(){
 // 256x256 BC1 with a full mip chain: 32768 bytes at level 0, then a quarter
 // each level down to the 1x1 level, which still occupies one 8-byte block.
 std::size_t bc1=ResourceCensus::surface(DXGI_FORMAT_BC1_UNORM,256,256,9);
 assert(bc1==32768+8192+2048+512+128+32+8+8+8);
 assert(ResourceCensus::surface(DXGI_FORMAT_BC7_UNORM,4,4,1)==16);
 assert(ResourceCensus::surface(DXGI_FORMAT_R16G16B16A16_FLOAT,10,10,1)==800);
 assert(ResourceCensus::surface(DXGI_FORMAT_D24_UNORM_S8_UINT,10,10,1)==400);
 assert(ResourceCensus::surface(DXGI_FORMAT_R8_UNORM,3,3,1)==9);
 ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;
 if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context))){
  std::puts("WARP unavailable");return 0;}
 D3D11_TEXTURE2D_DESC desc={};desc.Width=64;desc.Height=32;desc.MipLevels=0;desc.ArraySize=3;
 desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.SampleDesc.Count=1;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 ID3D11Texture2D* texture=nullptr;assert(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&texture)));
 ID3D11ShaderResourceView* view=nullptr;assert(SUCCEEDED(device->CreateShaderResourceView(texture,nullptr,&view)));
 // 64x32 down to 1x1 (7 levels) of 4-byte pixels, three slices.
 std::size_t expected=(2048+512+128+32+8+2+1)*4*3;
 ResourceCensus census;
 assert(census.add(view)==expected);
 assert(census.add(texture)==0&&census.add(view)==0); // already counted
 D3D11_BUFFER_DESC buffer={};buffer.ByteWidth=4096;buffer.BindFlags=D3D11_BIND_VERTEX_BUFFER;
 ID3D11Buffer* vertices=nullptr;assert(SUCCEEDED(device->CreateBuffer(&buffer,nullptr,&vertices)));
 assert(census.add(vertices)==4096);
 vertices->Release();view->Release();texture->Release();context->Release();device->Release();
 return 0;
}
''')


if __name__ == '__main__':
    unittest.main()
