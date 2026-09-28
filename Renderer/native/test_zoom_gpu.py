import unittest
from Renderer.native.native_cpp_test import run_cpp


class ZoomGpuTests(unittest.TestCase):
    def test_fullscreen_transform_cost_and_admission(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/gpu_image_compositor.h"
#include <cassert>
#include <cstdio>
#include <limits>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_gpu_images;
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL feature;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context));
 Compositor gpu(device.Get(),context.Get());constexpr unsigned w=2240,h=1260,count=32;
 auto source=gpu.create(w,h,Format::bgra32),destination=gpu.create(w,h,Format::bgra32);
 auto native=gpu.create(w,h,Format::rgb555),wrong_extent=gpu.create(1,1,Format::bgra32);
 assert(source&&destination&&native&&wrong_extent);
 assert(!gpu.transform_view(source,source,1.25f));
 assert(!gpu.transform_view(native,source,1.25f));
 assert(!gpu.transform_view(wrong_extent,source,1.25f));
 for(float invalid:{.9f,3.01f,std::numeric_limits<float>::quiet_NaN()})assert(!gpu.transform_view(destination,source,invalid));
 auto uploads=gpu.stats().uploads,snapshots=gpu.stats().snapshots;
 for(int n=0;n<4;++n)assert(gpu.transform_view(destination,source,1.25f));
 auto allocations=gpu.stats().allocations;
 D3D11_QUERY_DESC desc={D3D11_QUERY_TIMESTAMP_DISJOINT,0};ComPtr<ID3D11Query> disjoint,begin,end;
 checked(device->CreateQuery(&desc,&disjoint));desc.Query=D3D11_QUERY_TIMESTAMP;
 checked(device->CreateQuery(&desc,&begin));checked(device->CreateQuery(&desc,&end));
 context->Begin(disjoint.Get());context->End(begin.Get());
 for(unsigned n=0;n<count;++n)assert(gpu.transform_view(destination,source,1.f+2.f*float(n+1)/float(count)));
 context->End(end.Get());context->End(disjoint.Get());context->Flush();
 auto wait=[&](ID3D11Query* query,void* result,UINT bytes){
  auto deadline=GetTickCount64()+10000;HRESULT status;
  while((status=context->GetData(query,result,bytes,0))==S_FALSE){assert(GetTickCount64()<deadline);Sleep(1);}
  checked(status);
 };
 D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing={};UINT64 start=0,stop=0;
 wait(disjoint.Get(),&timing,sizeof(timing));wait(begin.Get(),&start,sizeof(start));wait(end.Get(),&stop,sizeof(stop));
 assert(gpu.stats().uploads==uploads&&gpu.stats().snapshots==snapshots&&gpu.stats().allocations==allocations);
 std::printf("PASS resolved-world zoom: %ux%u frames=%u uploads=0 readbacks=0 hot_allocations=0\n",w,h,count);
 double seconds=timing.Frequency&&stop>start?double(stop-start)/double(timing.Frequency)/count:0.;
 // Some VM drivers return zero or synthetic near-zero timestamp intervals.
 // A fullscreen multi-fetch pass below one microsecond is not usable timing
 // evidence. Keep the correctness result, but never report that as 0 ms.
 if(timing.Disjoint||!timing.Frequency||stop<=start||seconds<.000001)
  std::puts("MEASURE zoom GPU duration unavailable: driver returned disjoint, invalid or implausibly short timestamps");
 else std::printf("MEASURE isolated zoom pass mean_GPU_ms=%.3f (not game FPS)\n",
  1000.*seconds);
}
''', timeout=90)


if __name__ == "__main__":
    unittest.main()
