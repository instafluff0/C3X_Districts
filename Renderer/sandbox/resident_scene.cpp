// Sandbox-only access to production visual preparation and immutable assets.
// The client-owned pipeline below draws the retained scene directly.
#include "../native/c3x_renderer.cpp"
#include <dxgi1_2.h>
#include <wincodec.h>
#include <new>
#pragma comment(lib,"ole32.lib")
#pragma comment(lib,"windowscodecs.lib")

#include "direct_units.h"
#include "bloom.h"
#include "fresh_pipeline.h"

bool sandbox_capture_jpeg(char const* path,D3D11_MAPPED_SUBRESOURCE const& mapped,
        int width,int height){
    static Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    if(!factory){
        CoInitializeEx(nullptr,COINIT_MULTITHREADED);
        if(FAILED(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&factory))))return false;
    }
    wchar_t wide[4*MAX_PATH]{};
    if(!MultiByteToWideChar(CP_ACP,0,path,-1,wide,4*MAX_PATH))return false;
    Microsoft::WRL::ComPtr<IWICStream> stream;
    Microsoft::WRL::ComPtr<IWICBitmapEncoder> encoder;
    Microsoft::WRL::ComPtr<IWICBitmapFrameEncode> frame;
    Microsoft::WRL::ComPtr<IPropertyBag2> options;
    Microsoft::WRL::ComPtr<IWICBitmap> bitmap;
    Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
    if(FAILED(factory->CreateStream(&stream)) ||
        FAILED(stream->InitializeFromFilename(wide,GENERIC_WRITE)) ||
        FAILED(factory->CreateEncoder(GUID_ContainerFormatJpeg,nullptr,&encoder)) ||
        FAILED(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache)) ||
        FAILED(encoder->CreateNewFrame(&frame,&options)))return false;
    if(options){
        PROPBAG2 property={};property.pstrName=const_cast<LPOLESTR>(L"ImageQuality");
        VARIANT quality={};quality.vt=VT_R4;quality.fltVal=.92f;
        options->Write(1,&property,&quality);
    }
    if(FAILED(frame->Initialize(options.Get())) ||
        FAILED(frame->SetSize(UINT(width),UINT(height))))return false;
    GUID format=GUID_WICPixelFormat24bppBGR;
    if(FAILED(frame->SetPixelFormat(&format)) ||
        FAILED(factory->CreateBitmapFromMemory(UINT(width),UINT(height),
            GUID_WICPixelFormat32bppBGRA,mapped.RowPitch,
            UINT(std::size_t(mapped.RowPitch)*height),
            static_cast<BYTE*>(mapped.pData),&bitmap)) ||
        FAILED(factory->CreateFormatConverter(&converter)) ||
        FAILED(converter->Initialize(bitmap.Get(),GUID_WICPixelFormat24bppBGR,
            WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom)) ||
        FAILED(frame->WriteSource(converter.Get(),nullptr)) ||
        FAILED(frame->Commit()) || FAILED(encoder->Commit()))return false;
    return true;
}

// The production display transfer, with its source address offset by the
// scene's four-pixel guard. Its destination is the swapchain backbuffer.
struct SandboxBackbufferOutput {
    ID3D11VertexShader* vertex=nullptr;
    ID3D11PixelShader* pixel=nullptr;
    ID3D11Buffer* settings=nullptr;
    ID3D11RasterizerState* rasterizer=nullptr;
    ~SandboxBackbufferOutput() {
        if(vertex)vertex->Release();
        if(pixel)pixel->Release();
        if(settings)settings->Release();
        if(rasterizer)rasterizer->Release();
    }
    bool ensure() {
        if(pixel)return true;
        char const* source=R"(
Texture2D<float4> scene : register(t0);
Texture2D<float4> moving : register(t1);
Texture2D<float4> bloom : register(t2);
SamplerState linear_clamp : register(s0);
cbuffer OutputSettings : register(b0) {
 float exposure;float gain;float2 inverse_scene;
 float zoom;float3 padding;
 float4 ray_source_direction;
 float4 ray_shape;
};
float4 VSOutput(uint id : SV_VertexID) : SV_Position {
 float2 p=float2((id<<1)&2,id&2); return float4(p*float2(2,-2)+float2(-1,1),0,1);
}
float4 PSOutput(float4 position : SV_Position) : SV_Target {
 // Zoom the resolved scene and moving layer together in the tone-map pass.
 float2 center=(1/inverse_scene-8)*.5;
 float2 source_position=(position.xy-center)/zoom+center+4;
 float2 uv=source_position*inverse_scene;
 float4 dynamic_color,c;
 if(zoom<=1.00001){
  int3 at=int3(int2(position.xy)+int2(4,4),0);
  dynamic_color=moving.Load(at);
  c=dynamic_color+scene.Load(at)*(1-dynamic_color.a);
 }else{
  dynamic_color=moving.SampleLevel(linear_clamp,uv,0);
  c=dynamic_color+scene.SampleLevel(linear_clamp,uv,0)*(1-dynamic_color.a);
 }
 if(c.a<=.000001) return 0;
 // A single sunward source fans across the whole scene.
 // Modulate the composed HDR surfaces instead of laying pale air-colored
 // polygons over them: this preserves terrain texture and object silhouettes.
 if(ray_shape.x>.001){
  float2 delta=source_position-ray_source_direction.xy;
  float distance=max(length(delta),1);
  float2 direction=ray_source_direction.zw;
  float forward=dot(delta,direction);
  float across=1.7*(direction.x*delta.y-direction.y*delta.x)/distance;
  // Peak close to the shared sunward origin, then nearly vanish after
  // travelling forty percent of the viewport width.
  float screen_width=1/inverse_scene.x-8;
  float visible_distance=max(distance-ray_shape.w,0);
  float radial=smoothstep(0,screen_width*.12,visible_distance)*
      exp(-3.2*visible_distance/(screen_width*.40))*
      smoothstep(0,ray_shape.y*.30,forward);
  float soft=4.0/distance;
  float shafts=
      .35*exp(-pow((across+.82)/(.013+soft),2))+
      .62*exp(-pow((across+.67)/(.017+soft),2))+
      .44*exp(-pow((across+.55)/(.012+soft),2))+
      .82*exp(-pow((across+.41)/(.019+soft),2))+
      .57*exp(-pow((across+.29)/(.014+soft),2))+
      .76*exp(-pow((across+.18)/(.016+soft),2))+
      .49*exp(-pow((across+.06)/(.011+soft),2))+
     1.00*exp(-pow((across-.08)/(.018+soft),2))+
      .58*exp(-pow((across-.21)/(.013+soft),2))+
      .87*exp(-pow((across-.34)/(.017+soft),2))+
      .41*exp(-pow((across-.49)/(.012+soft),2))+
      .68*exp(-pow((across-.63)/(.018+soft),2))+
      .37*exp(-pow((across-.78)/(.014+soft),2));
  float light=saturate(shafts*1.3)*radial*ray_shape.x;
  float3 surface=max(c.rgb/c.a,0);
  // Blue water gets its own moving low-sun glint in the water pass.
  float water_hint=smoothstep(.025,.12,surface.b-surface.r);
  light*=1-.82*water_hint;
  float brightness=dot(surface,float3(.25,.62,.13));
  float evening=smoothstep(12,17,ray_shape.z);
  float3 warmth=lerp(float3(.84,.61,.20),float3(.95,.52,.09),evening);
  // Preserve texture contrast: bright surfaces catch more warm light while
  // dark creases and cast shadows remain recognizably dark.
  surface=surface*(1+light*.35)+
      warmth*light*.70*sqrt(saturate(brightness));
  c.rgb=surface*c.a;
 }
 c.rgb+=bloom.SampleLevel(linear_clamp,(source_position+.5)*inverse_scene,0).rgb*gain*c.a;
 float3 rgb=max(0,c.rgb/c.a*exposure);
 rgb/=1+max(rgb.r,max(rgb.g,rgb.b));
 rgb=float3(rgb.r<=.0031308?rgb.r*12.92:1.055*pow(rgb.r,1/2.4)-.055,
            rgb.g<=.0031308?rgb.g*12.92:1.055*pow(rgb.g,1/2.4)-.055,
            rgb.b<=.0031308?rgb.b*12.92:1.055*pow(rgb.b,1/2.4)-.055);
 return float4(saturate(rgb),saturate(c.a));
})";
        auto compile=[&](char const* entry,char const* target,ID3DBlob** blob) {
            ID3DBlob* errors=nullptr;
            HRESULT result=D3DCompile(source,std::strlen(source),"sandbox_backbuffer_output",
                nullptr,nullptr,entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,blob,&errors);
            if(errors){if(FAILED(result))std::printf("SANDBOX_OUTPUT_SHADER %s\n",
                static_cast<char const*>(errors->GetBufferPointer()));errors->Release();}
            return result;
        };
        ID3DBlob *vs=nullptr,*ps=nullptr;
        HRESULT result=compile("VSOutput","vs_4_0",&vs);
        if(SUCCEEDED(result))result=compile("PSOutput","ps_4_0",&ps);
        if(SUCCEEDED(result))result=renderer.device->CreateVertexShader(vs->GetBufferPointer(),
            vs->GetBufferSize(),nullptr,&vertex);
        if(SUCCEEDED(result))result=renderer.device->CreatePixelShader(ps->GetBufferPointer(),
            ps->GetBufferSize(),nullptr,&pixel);
        if(vs)vs->Release();if(ps)ps->Release();
        D3D11_BUFFER_DESC buffer{};buffer.ByteWidth=64;
        buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(result))result=renderer.device->CreateBuffer(&buffer,nullptr,&settings);
        D3D11_RASTERIZER_DESC raster{};raster.FillMode=D3D11_FILL_SOLID;
        raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=true;
        if(SUCCEEDED(result))result=renderer.device->CreateRasterizerState(&raster,&rasterizer);
        return SUCCEEDED(result);
    }
    bool draw(ID3D11RenderTargetView* target,int width,int height) {
        if(!ensure())return false;
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&target,nullptr);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(nullptr,0);
        context->RSSetState(rasterizer);
        D3D11_VIEWPORT viewport={0,0,float(width),float(height),0,1};
        context->RSSetViewports(1,&viewport);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        float values[16]={renderer.display_exposure,sandbox_fresh.glow.gain,
            1.f/sandbox_fresh.glow.native_extent,1.f/sandbox_fresh.glow.native_height,
            sandbox_fresh.display_zoom,0,0,0};
        if(sandbox_fresh.shafts.count){
            auto const& ray=sandbox_fresh.shafts.bands[0];
            values[8]=ray.x+4;
            values[9]=ray.y+4;
            values[10]=ray.dx;values[11]=ray.dy;
            values[12]=ray.phase;values[13]=ray.tile;
            values[14]=sandbox_fresh.visual_hour;
            values[15]=ray.offscreen;
        }
        context->UpdateSubresource(settings,0,nullptr,values,0,0);
        context->PSSetConstantBuffers(0,1,&settings);
        ID3D11ShaderResourceView* sources[]={sandbox_fresh.static_cache.view,
            sandbox_fresh.glow.linear.view,sandbox_fresh.bloom.view[0]};
        context->PSSetShaderResources(0,3,sources);
        context->PSSetSamplers(0,1,&sandbox_fresh.bloom.sampler);
        context->Draw(3,0);
        ID3D11ShaderResourceView* empty[]={nullptr,nullptr,nullptr};
        context->PSSetShaderResources(0,3,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        return true;
    }
};
static SandboxBackbufferOutput sandbox_backbuffer_output;

#ifdef C3X_RENDERER64_FRESH
// Renderer64 supplies the copied Civ III frame and owns GPU publication. This
// adapter writes the sandbox scene into its map image, never an HWND or RPC page.
bool c3x_renderer64_render_fresh(c3x_renderer_frame_v1 const& frame,
        ID3D11RenderTargetView* target) {
    renderer.trace.write("fresh-callback","enter",true);
    static unsigned device_generation=0;
    if(device_generation!=renderer.device_generation) {
        if(device_generation) {
            sandbox_backbuffer_output.~SandboxBackbufferOutput();
            new(&sandbox_backbuffer_output) SandboxBackbufferOutput();
            sandbox_fresh.~SandboxFreshPipeline();
            new(&sandbox_fresh) SandboxFreshPipeline();
            sandbox_direct_units.~SandboxDirectUnits();
            new(&sandbox_direct_units) SandboxDirectUnits();
        }
        device_generation=renderer.device_generation;
    }
    int camera_x=0,camera_y=0;
    if(frame.tile_count && frame.tiles) {
        auto const& first=frame.tiles[0];
        camera_x=first.anchor_x-first.tile_x*frame.tile_width/2;
        camera_y=first.anchor_y-first.tile_y*frame.tile_height/2;
    }
    if(!sandbox_fresh.draw(frame,camera_x,camera_y,0,0,0,0,false,1.f))return false;
    renderer.trace.write("fresh-callback","scene-ready",true);
    if(renderer.trace.level) {
        char detail[256];
        sprintf_s(detail,"prepare=%.3f reflection=%.3f static=%.3f water=%.3f units=%.3f reconstruct=%.3f shadow_builds=%u reflection_draws=%u static_draws=%u",
            sandbox_fresh.phases[0],sandbox_fresh.phases[1],sandbox_fresh.phases[2],
            sandbox_fresh.phases[3],sandbox_fresh.phases[4],sandbox_fresh.phases[5],
            sandbox_fresh.shadow.builds,sandbox_fresh.reflection_draws,
            sandbox_fresh.cache_full_draws);
        renderer.trace.write("fresh-scene-phases",detail,true);
    }
    bool presented=sandbox_backbuffer_output.draw(target,frame.target_width,frame.target_height);
    renderer.trace.write("fresh-callback",presented?"map-ready":"map-failed",true);
    return presented;
}
#endif
static LONG sandbox_swapchain_observed = -1;
static double sandbox_present_phases[4] = {};

extern "C" __declspec(dllexport) LONG c3x_sandbox_swapchain_observed() {
    return sandbox_swapchain_observed;
}
extern "C" __declspec(dllexport) void c3x_sandbox_present_metrics(double* phases) {
    if (phases) std::copy(sandbox_present_phases,sandbox_present_phases+4,phases);
}
extern "C" __declspec(dllexport) void c3x_sandbox_inspect_scene() {
    std::size_t records = 0, nonempty_layers = 0, gpu_bytes = 0;
    for (auto const& layer : renderer.geometry_vertex_buffers) {
        records += layer.size();
        nonempty_layers += !layer.empty();
    }
    for (auto const& pair : renderer.tile_geometry_cache)
        gpu_bytes += pair.second.byte_count;
    std::size_t material_bytes = 0;
    for (auto const& terrain : renderer.terrain_textures) {
        material_bytes += terrain.dds.size() + terrain.material_height_dds.size() +
            terrain.specular_dds.size() + terrain.elevated_dds.size() +
            terrain.elevated_height_dds.size() + terrain.elevated_specular_dds.size();
        for (auto const& layer : terrain.relief_layer_dds) material_bytes += layer.size();
        for (auto const& layer : terrain.water_surface_dds) material_bytes += layer.size();
    }
    std::printf("SANDBOX_PREPARATION tiles=%zu draw_records=%zu layers=%zu gpu_bytes=%zu material_bytes=%zu feature_assets=%zu resource_assets=%d resource_anchors=%zu resource_animations=%zu feature_records=%zu\n",
        renderer.tile_geometry_cache.size(), records, nonempty_layers, gpu_bytes, material_bytes,
        renderer.feature_bundle.assets.size(),int(renderer.resource_assets_ready),
        renderer.resource_anchors.size(),renderer.resource_animations.size(),
        renderer.geometry_vertex_buffers[geometry_feature].size());
    std::fflush(stdout);
    IDXGIDevice* dxgi = nullptr; IDXGIAdapter* adapter = nullptr;
    DXGI_ADAPTER_DESC description{};
    if (renderer.device && SUCCEEDED(renderer.device->QueryInterface(__uuidof(IDXGIDevice),
            reinterpret_cast<void**>(&dxgi))) &&
        SUCCEEDED(dxgi->GetAdapter(&adapter)) && SUCCEEDED(adapter->GetDesc(&description))) {
        char name[256]{};
        WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, name, sizeof(name), nullptr, nullptr);
        std::printf("SANDBOX_DEVICE vendor=0x%x device=0x%x feature_level=0x%x adapter=%s\n",
            description.VendorId, description.DeviceId, unsigned(renderer.device->GetFeatureLevel()), name);
        IDXGIAdapter3* budget_adapter=nullptr;
        if(SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3),
                reinterpret_cast<void**>(&budget_adapter)))){
            DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
            if(SUCCEEDED(budget_adapter->QueryVideoMemoryInfo(0,
                    DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&memory)))
                std::printf("SANDBOX_VIDEO_MEMORY local_budget_bytes=%llu local_usage_bytes=%llu\n",
                    static_cast<unsigned long long>(memory.Budget),
                    static_cast<unsigned long long>(memory.CurrentUsage));
            budget_adapter->Release();
        }
    }
    if (adapter) adapter->Release(); if (dxgi) dxgi->Release();
}

// The client owns the HWND and asks the same D3D11 device that owns prepared
// scene resources to publish to its swap chain. No CPU bitmap is in this path.
extern "C" __declspec(dllexport) int c3x_sandbox_present(HWND window,
        c3x_renderer_frame_v1 const* frame, int unit_x, int unit_y,
        int incarnation,int viewer,int visible,int camera_x, int camera_y) {
    static HWND bound_window = nullptr;
    static IDXGISwapChain1* swap = nullptr;
    static ID3D11Texture2D* back = nullptr;
    static ID3D11RenderTargetView* target = nullptr;
    char flat_option[8]{};
    bool flat_present=GetEnvironmentVariableA("C3X_SANDBOX_FLAT_PRESENT",
        flat_option,sizeof(flat_option)) && std::strcmp(flat_option,"1")==0;
    if (!window || !renderer.device || !renderer.context ||
        (!flat_present && (!sandbox_fresh.active || !sandbox_fresh.static_cache.view ||
            !sandbox_fresh.glow.linear.view ||
            !sandbox_fresh.bloom.view[0])))
        return 1;
    LARGE_INTEGER frequency{},ticks[5]={};
    QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&ticks[0]);
    if (window != bound_window) {
        if (target) { target->Release(); target = nullptr; }
        if (back) { back->Release(); back = nullptr; }
        if (swap) { swap->Release(); swap = nullptr; }
        IDXGIDevice* dxgi_device = nullptr;
        IDXGIAdapter* adapter = nullptr;
        IDXGIFactory2* factory = nullptr;
        HRESULT result = renderer.device->QueryInterface(__uuidof(IDXGIDevice),
            reinterpret_cast<void**>(&dxgi_device));
        if (SUCCEEDED(result)) result = dxgi_device->GetAdapter(&adapter);
        if (SUCCEEDED(result)) result = adapter->GetParent(__uuidof(IDXGIFactory2),
            reinterpret_cast<void**>(&factory));
        if (SUCCEEDED(result)) {
            DXGI_SWAP_CHAIN_DESC1 description{};
            description.Width = unsigned(renderer.content_view_width);
            description.Height = unsigned(renderer.content_view_height);
            description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            description.SampleDesc.Count = 1;
            description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            description.BufferCount = 2;
            description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
            description.Scaling = DXGI_SCALING_STRETCH;
            result = factory->CreateSwapChainForHwnd(renderer.device, window, &description,
                nullptr, nullptr, &swap);
        }
        if (factory) factory->Release();
        if (adapter) adapter->Release();
        if (dxgi_device) dxgi_device->Release();
        if (FAILED(result) || !swap) {
            std::printf("SANDBOX_SWAPCHAIN_ERROR hresult=0x%08lx\n", result);
            std::fflush(stdout);
            return 2;
        }
        result=swap->GetBuffer(0,__uuidof(ID3D11Texture2D),
            reinterpret_cast<void**>(&back));
        if(SUCCEEDED(result))result=renderer.device->CreateRenderTargetView(back,nullptr,&target);
        if(FAILED(result))return 3;
        bound_window = window;
    }
    int width = renderer.content_view_width, height = renderer.content_view_height;
    if(flat_present) {
        float color[4]={0.18f,0.24f,0.31f,1.0f};
        renderer.context->ClearRenderTargetView(target,color);
    } else {
        bool composed=sandbox_backbuffer_output.draw(target,width,height);
        if(!composed)return 6;
    }
    QueryPerformanceCounter(&ticks[1]);
    char unit_option[8]{};
    bool units = GetEnvironmentVariableA("C3X_SANDBOX_UNITS", unit_option, sizeof(unit_option)) &&
        std::strcmp(unit_option, "1") == 0;
    (void)incarnation;(void)viewer;(void)visible;
    QueryPerformanceCounter(&ticks[2]);
    static bool captured_initial = false, captured_moved = false;
    static bool captured_jump=false,captured_return=false,captured_wrap=false;
    static bool captured_scroll=false;
    bool initial_capture = unit_x == 19 && unit_y == 47 && !captured_initial;
    bool moved_capture = unit_x == 20 && unit_y == 48 && !captured_moved && frame &&
        frame->presentation_time_ticks>=2000;
    bool jump_capture=camera_x==-640 && camera_y==-256 && !captured_jump;
    bool scroll_capture=camera_x>=128 && camera_x<144 &&
        camera_y>=64 && camera_y<80 && !captured_scroll;
    bool return_capture=camera_x==0 && camera_y==0 && !captured_return && frame &&
        frame->presentation_time_ticks>=8000;
    bool wrap_capture=camera_x==6400 && camera_y==0 && !captured_wrap;
    char capture_path[4 * MAX_PATH]{};
    char sequence_prefix[3 * MAX_PATH]{};
    static unsigned sequence_frame=0;
    bool sequence_capture=frame && GetEnvironmentVariableA(
        "C3X_SANDBOX_CAPTURE_SEQUENCE",sequence_prefix,sizeof(sequence_prefix));
    char const* capture_variable = initial_capture ? "C3X_SANDBOX_CAPTURE" :
        moved_capture ? "C3X_SANDBOX_CAPTURE_MOVED" :
        scroll_capture ? "C3X_SANDBOX_CAPTURE_SCROLL" :
        jump_capture ? "C3X_SANDBOX_CAPTURE_JUMP" :
        return_capture ? "C3X_SANDBOX_CAPTURE_RETURN" :
        wrap_capture ? "C3X_SANDBOX_CAPTURE_WRAP" : nullptr;
    bool named_capture=units && capture_variable && GetEnvironmentVariableA(
        capture_variable,capture_path,sizeof(capture_path));
    if (sequence_capture) {
        sprintf_s(capture_path,"%s-%04u.jpg",sequence_prefix,sequence_frame++);
    }
    if (sequence_capture || named_capture) {
        D3D11_TEXTURE2D_DESC description{}; back->GetDesc(&description);
        description.Usage = D3D11_USAGE_STAGING;
        description.BindFlags = 0; description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        description.MiscFlags = 0;
        ID3D11Texture2D* staging = nullptr;
        if (SUCCEEDED(renderer.device->CreateTexture2D(&description, nullptr, &staging))) {
            renderer.context->CopyResource(staging, back);
            D3D11_MAPPED_SUBRESOURCE mapped{};
            if (SUCCEEDED(renderer.context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped))) {
                bool written=false;
                if(sequence_capture){
                    written=sandbox_capture_jpeg(capture_path,mapped,width,height);
                }else{
                    FILE* file = nullptr;
                    if (!fopen_s(&file, capture_path, "wb") && file) {
                        BITMAPFILEHEADER header{}; BITMAPINFOHEADER info{};
                        header.bfType = 0x4d42; header.bfOffBits = sizeof(header) + sizeof(info);
                        header.bfSize = header.bfOffBits + unsigned(width * height * 4);
                        info.biSize = sizeof(info); info.biWidth = width; info.biHeight = -height;
                        info.biPlanes = 1; info.biBitCount = 32; info.biCompression = BI_RGB;
                        written = fwrite(&header, sizeof(header), 1, file) == 1 &&
                            fwrite(&info, sizeof(info), 1, file) == 1;
                        for (int y = 0; y < height && written; ++y)
                            written = fwrite(static_cast<unsigned char const*>(mapped.pData) +
                                std::size_t(y) * mapped.RowPitch, width * 4, 1, file) == 1;
                        fclose(file);
                    }
                }
                if (written) {
                    if (initial_capture) captured_initial = true;
                    if (moved_capture) captured_moved = true;
                    if (jump_capture) captured_jump=true;
                    if (scroll_capture) captured_scroll=true;
                    if (return_capture) captured_return=true;
                    if (wrap_capture) captured_wrap=true;
                }
                if(!sequence_capture || sequence_frame%30==0)std::printf("CLIENT_CAPTURE result=%d time_ms=%lld path=%s\n",
                    int(written),frame?static_cast<long long>(frame->presentation_time_ticks):0,
                    capture_path);
                renderer.context->Unmap(staging, 0);
            }
            staging->Release();
        }
    }
    QueryPerformanceCounter(&ticks[3]);
    char present_mode[16]{};
    GetEnvironmentVariableA("C3X_SANDBOX_PRESENT_MODE",present_mode,sizeof(present_mode));
    HRESULT result=std::strcmp(present_mode,"skip")==0?S_OK:
        swap->Present(std::strcmp(present_mode,"immediate")==0?0:1,0);
    QueryPerformanceCounter(&ticks[4]);
    for (int i=0;i<4;++i) sandbox_present_phases[i]=
        1000.0*double(ticks[i+1].QuadPart-ticks[i].QuadPart)/double(frequency.QuadPart);
    DXGI_FRAME_STATISTICS statistics{};
    if (std::strcmp(present_mode,"skip")!=0 && SUCCEEDED(result) &&
        SUCCEEDED(swap->GetFrameStatistics(&statistics)))
        sandbox_swapchain_observed = LONG(statistics.PresentCount);
    return SUCCEEDED(result) ? 0 : 4;
}
