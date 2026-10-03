"""Execute the production loading stage before its first unit-consuming render."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_preparation_cancellation import block_at


class UnitSourceLoadingIntegrationTests(unittest.TestCase):
    def test_cold_union_and_retained_retry_admit_before_render_and_refuse_missing_capacity(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        start = source.index('    bool prepare_loading_sources(')
        end = source.index('    // One GPU owner', start)
        method = source[start:end].strip()
        program = r'''
#define C3X_RENDERER64_FRESH 1
#include "Renderer/native/render_core/unit_source_plan.h"
#include <cassert>
#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
constexpr std::uint64_t mib=1024u*1024u;
enum {C3X_RENDERER_RESULT_OK=0,C3X_RENDERER_RESULT_PENDING=1,C3X_RENDERER_RESULT_ERROR=-1,C3X_RENDERER_API_VERSION=1};
struct c3x_renderer_output_v1 {unsigned version=0,size=0;};
std::uint64_t available=4096*mib,total=8192*mib,gpu_budget=2048*mib,gpu_usage=0,gpu_owned=0;
bool memory_ok=true,gpu_ok=true,assets_ok=true,fresh_ok=true,composition_ok=true,rigid_ok=true;
std::vector<int> order;
struct MEMORYSTATUSEX {unsigned dwLength=0;std::uint64_t ullTotalPhys=0,ullAvailPhys=0;};
bool GlobalMemoryStatusEx(MEMORYSTATUSEX* m){m->ullTotalPhys=total;m->ullAvailPhys=available;return memory_ok;}
bool SUCCEEDED(int result){return result==0;}
template<class T>T* IID_PPV_ARGS(T* output){return output;}
void Sleep(unsigned time){assert(time==1);}
enum {DXGI_MEMORY_SEGMENT_GROUP_LOCAL=0};
struct DXGI_QUERY_VIDEO_MEMORY_INFO {std::uint64_t Budget=0,CurrentUsage=0;};
namespace Microsoft {namespace WRL {template<class T>struct ComPtr {T* p=nullptr;T* operator->(){return p;}
 template<class U>int As(ComPtr<U>* out){static U value;out->p=&value;return 0;}};}}
struct IDXGIAdapter3 {int QueryVideoMemoryInfo(unsigned,unsigned,DXGI_QUERY_VIDEO_MEMORY_INFO* info){
 info->Budget=gpu_budget;info->CurrentUsage=gpu_usage;return gpu_ok?0:-1;}};
struct IDXGIAdapter {};
struct IDXGIDevice {int GetAdapter(Microsoft::WRL::ComPtr<IDXGIAdapter>* out){static IDXGIAdapter value;out->p=&value;return 0;}};
struct Device {int QueryInterface(Microsoft::WRL::ComPtr<IDXGIDevice>* out){static IDXGIDevice value;out->p=&value;return 0;}};
bool c3x_renderer64_prepare_scene_assets(){order.push_back(5);return fresh_ok;}
std::size_t c3x_renderer64_unit_source_mesh_bytes(){order.push_back(2);return gpu_owned;}
namespace c3x_gpu_images {struct Session {
 template<class D>Session(D*,void*){}
 bool prepare_assets(std::function<bool()> cancel){order.push_back(6);return composition_ok && !cancel();}
};}
struct State {
 std::size_t unit_source_cpu_allowance=0,unit_source_gpu_allowance=0;
 bool unit_sources_ready=false,unit_rendering_enabled=true;unsigned unit_source_device=0,device_generation=1;
 struct {std::size_t resident_bytes=0,contribution_bytes=0;}unit_bodies;
 struct {unsigned finishes=0;void finish_lease(){++finishes;}}unit_asset_preparation;
 struct {void write(char const*,char const*,bool){}}trace;
 Device owned_device;Device* device=&owned_device;void* context=nullptr;
 std::unique_ptr<c3x_gpu_images::Session> gpu_composition;
 std::size_t required_cpu=128*mib,required_gpu=64*mib;unsigned source_calls=0,renders=0;
 bool ensure_scene_assets(){order.push_back(1);return assets_ok;}
 bool ensure_ordered_rigid_layout(){order.push_back(7);return rigid_ok;}
 int prepare_known_unit_sources(std::vector<std::size_t> const& indices,bool all,
  std::size_t cpu,std::size_t gpu,std::atomic<bool> const*){
  assert(indices.empty() && all);order.push_back(3);++source_calls;
  unit_source_cpu_allowance=cpu;unit_source_gpu_allowance=gpu;
  if(cpu<required_cpu || gpu<required_gpu)return C3X_RENDERER_RESULT_ERROR;
  if(source_calls==1)return C3X_RENDERER_RESULT_PENDING;
  unit_sources_ready=true;unit_source_device=device_generation;return C3X_RENDERER_RESULT_OK;
 }
 template<class... Args>bool render(Args...){order.push_back(4);++renders;assert(unit_sources_ready);return true;}
};
struct Harness {
 State renderer_state;std::atomic<bool> camera_cancelled{false};bool initial_world=true,cancel_on_service=false;
 unsigned services=0,job_frame=0;
 void service_camera_preparation(){++services;if(cancel_on_service)camera_cancelled=true;}
''' + method + r'''
 bool run(){bool complete=prepare_loading_sources(&camera_cancelled,[this]{service_camera_preparation();});
 c3x_renderer_output_v1 output{};return complete && renderer_state.render(job_frame,output);}
};
void reset(){available=4096*mib;total=8192*mib;gpu_budget=2048*mib;gpu_usage=gpu_owned=0;
 memory_ok=gpu_ok=assets_ok=fresh_ok=composition_ok=rigid_ok=true;order.clear();}
int main(){reset();Harness cold;
 assert(cold.run() && cold.renderer_state.renders==1 && cold.renderer_state.source_calls==2);
 assert((order==std::vector<int>{1,7,5,6,2,3,3,4})); // A >96 MiB cold unit union is admitted first.
 assert(cold.renderer_state.unit_source_cpu_allowance==768*mib && cold.renderer_state.unit_source_gpu_allowance==768*mib);
 reset();Harness retry;retry.renderer_state.unit_bodies.resident_bytes=900*mib;
 retry.renderer_state.unit_bodies.contribution_bytes=4*mib;retry.renderer_state.required_cpu=904*mib;
 gpu_owned=gpu_usage=450*mib;gpu_budget=1024*mib;available=2560*mib;retry.renderer_state.required_gpu=450*mib;
 assert(retry.run() && retry.renderer_state.unit_source_cpu_allowance==904*mib && retry.renderer_state.unit_source_gpu_allowance==450*mib);
 // The retained union needs no new source allocation even with zero measured growth headroom.
 reset();Harness denied;available=2560*mib;gpu_budget=1024*mib;gpu_usage=400*mib;
 bool failed=false;try{denied.run();}catch(std::runtime_error const&){failed=true;}
 assert(failed && !denied.renderer_state.renders && denied.renderer_state.unit_asset_preparation.finishes==1 && !denied.renderer_state.unit_sources_ready);
 reset();Harness unavailable;gpu_ok=false;failed=false;try{unavailable.run();}catch(std::runtime_error const&){failed=true;}
 assert(failed && !unavailable.renderer_state.renders && !unavailable.renderer_state.unit_sources_ready);
 reset();Harness init_failed;assets_ok=false;failed=false;try{init_failed.run();}catch(std::runtime_error const&){failed=true;}
 assert(failed && (order==std::vector<int>{1}) && !init_failed.renderer_state.source_calls && !init_failed.renderer_state.renders);
 reset();Harness rigid_failed;rigid_ok=false;failed=false;try{rigid_failed.run();}catch(std::runtime_error const&){failed=true;}
 assert(failed && (order==std::vector<int>{1,7}) && !rigid_failed.renderer_state.source_calls && !rigid_failed.renderer_state.renders);
 reset();Harness disabled;disabled.renderer_state.unit_rendering_enabled=false;
 assert(disabled.prepare_loading_sources() && (order==std::vector<int>{1,7,5,6}) && !disabled.renderer_state.source_calls);
 reset();Harness shader_failed;fresh_ok=false;failed=false;try{shader_failed.run();}catch(std::runtime_error const&){failed=true;}
 assert(failed && (order==std::vector<int>{1,7,5}) && !shader_failed.renderer_state.source_calls && !shader_failed.renderer_state.renders);
 reset();Harness composition_failed;composition_ok=false;failed=false;try{composition_failed.run();}catch(std::runtime_error const&){failed=true;}
 assert(failed && (order==std::vector<int>{1,7,5,6}) && !composition_failed.renderer_state.source_calls && !composition_failed.renderer_state.renders);
 reset();Harness cancelled;cancelled.cancel_on_service=true;failed=false;try{cancelled.run();}catch(std::runtime_error const&){failed=true;}
 assert(failed && cancelled.camera_cancelled && cancelled.renderer_state.source_calls==1 && !cancelled.renderer_state.renders && cancelled.renderer_state.unit_asset_preparation.finishes==1);
}
'''
        run_cpp(program)
        # The bridge build has no fresh source-owner symbol; its shared helper must compile.
        bridge = program.replace('#define C3X_RENDERER64_FRESH 1\n', '')
        bridge = bridge.replace('std::size_t c3x_renderer64_unit_source_mesh_bytes(){order.push_back(2);return gpu_owned;}', '')
        bridge = bridge.replace('bool c3x_renderer64_prepare_scene_assets(){order.push_back(5);return fresh_ok;}', '')
        bridge = bridge[:bridge.index('int main(){reset();Harness cold;')] + r'''
int main(){reset();Harness cold;
 assert(cold.run() && cold.renderer_state.source_calls==2);
 assert((order==std::vector<int>{1,7,6,3,3,4}));
}
'''
        run_cpp(bridge)


if __name__ == "__main__":
    unittest.main()
