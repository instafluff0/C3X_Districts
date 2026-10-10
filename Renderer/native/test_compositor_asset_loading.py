"""Execute source-only compositor preparation without a GPU or a view.

Shader compilation and device allocations are counted host stubs. Native D3D
shader validity and pixels remain the existing Windows compositor oracle.
"""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp
from Renderer.native.test_fresh_preparation_cancellation import block_at


def method(source, signature):
    start = source.index(signature)
    # The cancellation callback has a braced default argument before the body.
    signature_end = source.index('){', start)
    opening = signature_end + 1
    return source[start:opening] + block_at(source, opening)


def fixture():
    base = ROOT / 'Renderer/native'
    compositor = (base / 'gpu_image_compositor.h').read_text()
    spatial = (base / 'gpu_spatial_composition.h').read_text()
    retained = (base / 'retained_composition.h').read_text()
    session = (base / 'gpu_composition_session.h').read_text()
    classes = []
    for filename in ('gpu_image_display.h', 'gpu_view_transform.h',
                     'gpu_unit_scene.h', 'gpu_projected_layer.h'):
        text = (base / filename).read_text()
        classes.append('\n'.join(line for line in text.splitlines()
                                 if not line.startswith(('#include', '#pragma'))))
    ctor = block_at(compositor, compositor.index('    Compositor(ID3D11Device*'))
    ctor = ctor[ctor.index('        char const* source='):ctor.rfind('}')]
    image_constants_start = compositor.index('    struct ImageConstants')
    image_constants = block_at(compositor, image_constants_start) + ';'
    helpers = '\n'.join(method(compositor, '    void ' + name + '()')
                        for name in ('prepare_unit_program', 'prepare_lookup_program',
                                     'prepare_blend_program', 'prepare_image_program',
                                     'prepare_import_program'))
    return r'''
#include <cassert>
#include <array>
#include <cstring>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using HRESULT=int;using LONGLONG=long long;
constexpr int TRUE=1,D3D11_FILL_SOLID=1,D3D11_CULL_NONE=0,
 D3D11_USAGE_DEFAULT=0,D3D11_BIND_CONSTANT_BUFFER=1,
 D3DCOMPILE_ENABLE_STRICTNESS=1,D3DCOMPILE_IEEE_STRICTNESS=2,
 D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST=0;
bool FAILED(HRESULT h){return h<0;}
int operations=0,fail_at=0,compiled=0,draws=0,textures=0;
std::vector<std::string> programs;
int operation(){return ++operations==fail_at?-1:0;}
struct ID3D11Device;
struct Resource {ID3D11Device* device=nullptr;virtual ~Resource()=default;
 void GetDevice(ID3D11Device** result){*result=device;}};
struct ID3D11ComputeShader:Resource{};struct ID3D11VertexShader:Resource{};
struct ID3D11PixelShader:Resource{};struct ID3D11RasterizerState:Resource{};
struct ID3D11Buffer:Resource{};struct ID3D11ShaderResourceView:Resource{};
struct ID3D11RenderTargetView:Resource{};struct ID3D11UnorderedAccessView:Resource{};
struct ID3DBlob:Resource{void* GetBufferPointer(){return this;}unsigned GetBufferSize(){return 4;}};
namespace Microsoft {namespace WRL {
template<class T>struct ComPtr{
 T* value=nullptr;T* Get()const{return value;}T* operator->()const{return value;}
 explicit operator bool()const{return value!=nullptr;}void Reset(){value=nullptr;}
 T** operator&(){value=nullptr;return &value;}
};}}
template<class T>using Ptr=Microsoft::WRL::ComPtr<T>;
template<class T>using ComPtr=Microsoft::WRL::ComPtr<T>;
// Owned fake resources keep the fixture bounded even across allocation retries.
std::vector<std::unique_ptr<Resource>> owned;
template<class T>T* resource(ID3D11Device* device=nullptr){auto next=std::make_unique<T>();
 next->device=device;auto result=next.get();owned.push_back(std::move(next));return result;}
HRESULT D3DCompile(char const*,std::size_t,char const* label,void*,void*,char const* entry,
                  char const*,int,int,ID3DBlob** code,ID3DBlob**){
 if(operation()<0)return -1;++compiled;programs.push_back(std::string(label)+":"+entry);*code=resource<ID3DBlob>();return 0;}
struct D3D11_BUFFER_DESC{unsigned ByteWidth=0,Usage=0,BindFlags=0;};
struct D3D11_RASTERIZER_DESC{int FillMode=0,CullMode=0,ScissorEnable=0,DepthClipEnable=0;};
struct RECT{long left=0,top=0,right=0,bottom=0;};
struct D3D11_VIEWPORT{float a,b,c,d,e,f;};
struct LARGE_INTEGER{long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* value){static long long ticks=0;value->QuadPart=++ticks;}
void OutputDebugStringA(char const*){}
struct ID3D11Device{
 template<class T>int create(T** target){if(operation()<0)return -1;*target=resource<T>(this);return 0;}
 int CreateComputeShader(void*,unsigned,void*,ID3D11ComputeShader** r){return create(r);}
 int CreateVertexShader(void*,unsigned,void*,ID3D11VertexShader** r){return create(r);}
 int CreatePixelShader(void*,unsigned,void*,ID3D11PixelShader** r){return create(r);}
 int CreateRasterizerState(D3D11_RASTERIZER_DESC*,ID3D11RasterizerState** r){return create(r);}
 int CreateBuffer(D3D11_BUFFER_DESC*,void*,ID3D11Buffer** r){return create(r);}
};
struct ID3D11DeviceContext{
 void ClearState(){}
 template<class... T>void RSSetViewports(T...){}template<class... T>void RSSetScissorRects(T...){}
 template<class... T>void RSSetState(T...){}template<class... T>void OMSetRenderTargets(T...){}
 template<class... T>void IASetPrimitiveTopology(T...){}template<class... T>void VSSetShader(T...){}
 template<class... T>void PSSetShader(T...){}template<class... T>void PSSetShaderResources(T...){}
 template<class... T>void CSSetShaderResources(T...){}template<class... T>void CSSetUnorderedAccessViews(T...){}
 template<class... T>void CSSetShader(T...){}template<class... T>void CSSetConstantBuffers(T...){}
 template<class... T>void UpdateSubresource(T...){}
 void Draw(unsigned,unsigned){++draws;}void Dispatch(unsigned,unsigned,unsigned){++draws;}
};
namespace c3x_renderer {inline std::string scene_detail_filter(){return "";}}
namespace c3x_gpu_images {struct Rect{int left=0,top=0,right=0,bottom=0;};
inline void checked(HRESULT result){if(FAILED(result))throw std::runtime_error("allocation");}}
''' + '\n'.join(classes) + r'''
namespace c3x_gpu_images {
struct SpatialComposition{
 ID3D11Device* device;Ptr<ID3D11ComputeShader> shader,fused_shader;Ptr<ID3D11Buffer> constants,fused_constants;
 SpatialComposition(ID3D11Device* d):device(d){}
 static void check(HRESULT result){checked(result);}
''' + method(spatial, '    void program()') + method(spatial, '    void prepare_assets()') + r'''
};
struct Compositor{
 struct Constants{int area[4],offset[2];unsigned mode,color;};
''' + image_constants + r'''
 ID3D11Device* device;ComPtr<ID3D11ComputeShader> shader,import_shader,unit_shader,image_shader,blend_shader,lookup_shader;
 ComPtr<ID3D11Buffer> constants,image_constants;ImageDisplay display_program;ViewTransform view_program;
 c3x_renderer::GpuUnitScene unit_scene;SpatialComposition spatial;
 Compositor(ID3D11Device* d):device(d),spatial(d){
''' + ctor + '\n}\n' + helpers + method(compositor, '    bool prepare_assets(') + r'''
};
struct RetainedComposition{
 ID3D11Device* device;Compositor replay;ProjectedLayer projected_layer;
 RetainedComposition(ID3D11Device* d):device(d),replay(d){}
''' + method(retained, '    bool prepare_assets(') + r'''
};
struct Session{
 Compositor gpu;RetainedComposition layers;
 Session(ID3D11Device* d):gpu(d),layers(d){}
''' + method(session, '    bool prepare_assets(') + r'''
};
}
void reset(){operations=fail_at=compiled=draws=textures=0;programs.clear();owned.clear();}
'''


class CompositorAssetLoadingTests(unittest.TestCase):
    def test_complete_without_view_then_warm_reuse_and_real_draw(self):
        run_cpp(fixture() + r'''
int main(){
 reset();ID3D11Device device;ID3D11DeviceContext context;c3x_gpu_images::Session session(&device);
 // 29: each Compositor's spatial owner also compiles the fused interface
 // entry (a0fa2441, performance review section 35).
 assert(session.prepare_assets());assert(compiled==29 && !draws && !textures);
 auto before=operations;assert(session.prepare_assets());assert(operations==before);
 // Both owners and all packed native display formats are ready. Their actual
 // first draw bodies reuse source programs and allocate no new shader/buffer.
 ID3D11ShaderResourceView input;ID3D11RenderTargetView target;ID3D11UnorderedAccessView output;
 for(auto* gpu:{&session.gpu,&session.layers.replay}){
  for(unsigned format=0;format<3;++format)assert(gpu->display_program.draw(&device,&context,&input,&target,640,480,{0,0,640,480},format));
  gpu->view_program.draw(&device,&context,&input,&output,640,480,1.);
  c3x_renderer::UnitSceneSample sample;gpu->unit_scene.draw(&device,&context,sample,12,16);
 }
 session.layers.projected_layer.draw(&device,&context,&input,nullptr,&output,{0,0,12,16},1.,0,0,0,0);
 assert(operations==before && compiled==29 && draws==11 && !textures);
 // Existing device replacement rebuilds only the final-display source set;
 // the Session itself is recreated by the existing owner reset path.
 ID3D11Device replacement;session.gpu.display_program.prepare_assets(&replacement);
 assert(compiled==33);before=operations;session.gpu.display_program.prepare_assets(&replacement);assert(operations==before);
}
''')

    def test_all_source_allocation_failures_refuse_then_retry_complete(self):
        run_cpp(fixture() + r'''
int main(){
 reset();ID3D11Device device;int total;
 {c3x_gpu_images::Session session(&device);assert(session.prepare_assets());total=operations;}
 for(int failed=1;failed<=total;++failed){
  reset();fail_at=failed;bool threw=false;std::unique_ptr<c3x_gpu_images::Session> session;
  try{session=std::make_unique<c3x_gpu_images::Session>(&device);session->prepare_assets();}
  catch(std::runtime_error const&){threw=true;}
  assert(threw && !draws && !textures);fail_at=0;
  if(!session)session=std::make_unique<c3x_gpu_images::Session>(&device);
  assert(session->prepare_assets());auto before=operations;
  assert(session->prepare_assets() && operations==before && !draws && !textures);
 }
}
''')

    def test_cancel_each_stage_never_reports_partial_ready(self):
        run_cpp(fixture() + r'''
int main(){
 for(unsigned stop=1;stop<=23;++stop){
  reset();ID3D11Device device;c3x_gpu_images::Session session(&device);unsigned checks=0;
  bool ready=session.prepare_assets([&]{return ++checks==stop;});
  if(stop<=22)assert(!ready);else assert(ready);
  assert(!draws && !textures);assert(session.prepare_assets());
  auto before=operations;assert(session.prepare_assets() && operations==before);
 }
}
''')


if __name__ == '__main__':
    unittest.main()
