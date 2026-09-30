"""Exercise scene-sized facade lights and the selected city shadow connection."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

class CityLighting(unittest.TestCase):
    def test_scene_lights_exceed_old_regional_capacity_and_reset(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
#include "Renderer/native/city_fidelity/scene_lights.h"
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 using namespace c3x_renderer::city_fidelity;
 Lighting city;city.lights.resize(80);city.blockers.resize(42);
 for(unsigned i=0;i<80;++i){city.lights[i].range=1;city.lights[i].intensity=float(i);city.lights[i].owner=float(i%42);}
 for(unsigned i=0;i<42;++i)city.blockers[i].low[0]=float(i);
 std::vector<Lighting const*> cities(30,&city);SceneLights gpu;
 assert(gpu.upload(context.Get(),cities,1,1));
 assert(gpu.capacity>=2400*3+1260*2);
 auto read=[&](ID3D11Buffer* buffer){
  D3D11_BUFFER_DESC desc={};buffer->GetDesc(&desc);unsigned bytes=desc.ByteWidth;
  desc.BindFlags=0;desc.MiscFlags=0;desc.StructureByteStride=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
  ComPtr<ID3D11Buffer> copy;checked(device->CreateBuffer(&desc,nullptr,&copy));context->CopyResource(copy.Get(),buffer);
  D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(copy.Get(),0,D3D11_MAP_READ,0,&m));
  std::vector<float> out(bytes/4);std::memcpy(out.data(),m.pData,bytes);context->Unmap(copy.Get(),0);return out;
 };
 auto field=read(gpu.frame.Get());assert(field[0]==2400&&field[1]==1260&&field[2]==1);
 auto data=read(gpu.data.Get());assert(data[(2399*3+1)*4+3]==79);assert(data[(2399*3+2)*4+3]==float(79%42+29*42));
 assert(data[(2400*3+1259*2)*4]==41);
 auto before=gpu.cached;
 assert(gpu.upload(context.Get(),cities,1,.5f));assert(gpu.cached==before);
 auto list_before=gpu.spatial.records;
 city.lights[0].position[0]=.25f;
 assert(gpu.upload(context.Get(),cities,1,1));assert(gpu.cached!=before);assert(gpu.spatial.records!=list_before);
 // Full-scan fallback for a grid too large, without any dropped lights.
 city.lights[0].range=10000;
 assert(gpu.upload(context.Get(),cities,1,1));assert(!gpu.indexed);
 field=read(gpu.frame.Get());assert(field[0]==2400&&field[19]==0);
 city.lights[0].range=1;
 assert(gpu.upload(context.Get(),cities,1,1));assert(gpu.indexed);
 auto capacity=gpu.capacity;assert(gpu.upload(context.Get(),cities,0,1));field=read(gpu.frame.Get());
 assert(field[0]==0&&field[1]==0&&field[2]==0&&gpu.capacity==capacity);
 assert(gpu.upload(context.Get(),{},1,1));field=read(gpu.frame.Get());assert(field[0]==0&&field[1]==0);
 assert(gpu.upload(context.Get(),{&city},1,1));field=read(gpu.frame.Get());assert(field[0]==80&&field[1]==42);
 Lighting layout_a,layout_b;layout_a.lights.resize(3);layout_b.lights.resize(1);layout_b.blockers.resize(3);
 SceneLights layout_gpu;assert(layout_gpu.upload(context.Get(),{&layout_a},1,1));
 auto counts_before=layout_gpu.cached_lights;auto old_records=layout_gpu.spatial.records;
 assert(layout_gpu.upload(context.Get(),{&layout_b},1,1));assert(layout_gpu.cached_lights!=counts_before);assert(layout_gpu.spatial.records!=old_records);
 assert(layout_gpu.cached_lights==1&&layout_gpu.cached_blockers==3);
 std::puts("PASS city lighting: 2400 lights, 1260 blockers, complete owner indices, daylight/empty/reuse");
}
''',timeout=120)

    def test_all_city_passes_use_current_shadow_field(self):
        source=(Path(__file__).parent.parent/'sandbox/fresh_pipeline.h').read_text()
        self.assertIn('compile("city.hlsl",entries[i],&replacement)',source)
        for entry in ['PSNativeCity','PSNativeCityEmission','PSNativeCityReflection','PSNativeCityReflectionEmission']:
            self.assertIn('"'+entry+'"',source)
        self.assertIn('float texel=max(box.z,box.w)/4096.',source)

if __name__=='__main__':unittest.main()
