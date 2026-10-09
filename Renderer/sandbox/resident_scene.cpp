// Sandbox-only access to production visual preparation and immutable assets.
// The client-owned pipeline below draws the retained scene directly.
#include "../native/c3x_renderer.cpp"
#include <dxgi1_2.h>
#include <wincodec.h>
#include <new>
#pragma comment(lib,"ole32.lib")
#pragma comment(lib,"windowscodecs.lib")

#include "direct_units.h"
#include "direct_effects.h"
#include "bloom.h"
#include "fresh_pipeline.h"
#include "world_content_receipt.h"
#include "../native/render_core/resource_census.h"

extern "C" __declspec(dllexport) int c3x_sandbox_world_content(char const* path){
    return sandbox_world_content_receipt(renderer,path,[](auto const& owner)->auto const& {return owner.mesh->layers;},
        renderer.retired_content->bytes.load(),renderer.retired_content->peak.load(),renderer.geometry_vertex_buffers.content_size());
}

float sandbox_scene_filmic(){
    char value[32]{};
    if(!GetEnvironmentVariableA("C3X_RENDERER_SCENE_FILMIC",value,sizeof(value)))return .5f;
    char* end=nullptr;float amount=std::strtof(value,&end);
    return end!=value && *end==0 && std::isfinite(amount)?std::clamp(amount,0.f,1.f):.5f;
}

// Explicit standalone diagnostic only; never called by the game draw callback.
// Keep FP16 values for repeatable Mac analysis without asset or image copies.
bool sandbox_capture_linear(char const* path) {
    FILE* file=nullptr;
    if(fopen_s(&file,path,"wb") || !file)return false;
    unsigned header[]={0x32485243u,3}; // CRH2, three RGBA16F surfaces
    bool ok=fwrite(header,sizeof(header),1,file)==1;
    float parameters[]={renderer.display_exposure,sandbox_fresh.glow.gain,sandbox_scene_filmic(),0.f};
    ok=ok && fwrite(parameters,sizeof(parameters),1,file)==1;
    ID3D11Texture2D* surfaces[]={sandbox_fresh.static_cache.resolved,
        sandbox_fresh.glow.linear.resolved,sandbox_fresh.bloom.color[0]};
    for(auto* surface:surfaces){
        if(!ok || !surface){ok=false;break;}
        D3D11_TEXTURE2D_DESC desc{};surface->GetDesc(&desc);
        if(desc.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT || desc.SampleDesc.Count!=1){ok=false;break;}
        unsigned extent[]={desc.Width,desc.Height};
        ok=fwrite(extent,sizeof(extent),1,file)==1;
        desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=desc.MiscFlags=0;
        desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
        if(FAILED(renderer.device->CreateTexture2D(&desc,nullptr,&staging))){ok=false;break;}
        renderer.context->CopyResource(staging.Get(),surface);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if(FAILED(renderer.context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped))){ok=false;break;}
        for(unsigned y=0;y<desc.Height && ok;++y)
            ok=fwrite(static_cast<char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,
                std::size_t(desc.Width)*8,1,file)==1;
        renderer.context->Unmap(staging.Get(),0);
    }
    ok=fclose(file)==0 && ok;
    return ok;
}

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

#include "../native/scene_display.h"

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
 float zoom;float filmic;float2 padding;
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
 c.rgb+=bloom.SampleLevel(linear_clamp,(source_position+.5)*inverse_scene,0).rgb*gain*c.a;
 float3 rgb=scene_display_srgb(c.rgb/c.a*exposure,filmic);
 return float4(saturate(rgb),saturate(c.a));
})";
        std::string combined=c3x_renderer::scene_display_shader();
        combined+=source;
        auto compile=[&](char const* entry,char const* target,ID3DBlob** blob) {
            ID3DBlob* errors=nullptr;
            HRESULT result=D3DCompile(combined.data(),combined.size(),"sandbox_backbuffer_output",
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
        D3D11_BUFFER_DESC buffer{};buffer.ByteWidth=32;
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
        float values[8]={renderer.display_exposure,sandbox_fresh.glow.gain,
            1.f/sandbox_fresh.glow.native_extent,1.f/sandbox_fresh.glow.native_height,
            sandbox_fresh.display_zoom,sandbox_scene_filmic(),0,0};
        context->UpdateSubresource(settings,0,nullptr,values,0,0);
        context->PSSetConstantBuffers(0,1,&settings);
        ID3D11ShaderResourceView* sources[]={sandbox_fresh.static_cache.view,
            sandbox_fresh.glow.linear.view,sandbox_fresh.bloom.view[0]};
        context->PSSetShaderResources(0,3,sources);
        context->PSSetSamplers(0,1,&sandbox_fresh.bloom.sampler);
        context->Draw(3,0);
        {SandboxPassWorkload::Scope pass(sandbox_fresh.work,SandboxPassWorkload::publication);
         sandbox_fresh.work.draw(3);sandbox_fresh.work.upload(sizeof(values));
         if(sandbox_fresh.work.enabled)sandbox_fresh.work.row().target_pixels+=std::uint64_t(width)*height;}
        ID3D11ShaderResourceView* empty[]={nullptr,nullptr,nullptr};
        context->PSSetShaderResources(0,3,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        return true;
    }
};
static SandboxBackbufferOutput sandbox_backbuffer_output;

// Explicit untimed witness capture only. The color view uses the ordinary final
// output shader; depth is exact packed D24/S8 from the completed scene target.
extern "C" __declspec(dllexport) int c3x_sandbox_witness_capture(char const* prefix) {
    if(!prefix)return 1;
    D3D11_TEXTURE2D_DESC d={};d.Width=renderer.content_view_width;d.Height=renderer.content_view_height;
    d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
    d.BindFlags=D3D11_BIND_RENDER_TARGET;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> color;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> target;
    if(FAILED(renderer.device->CreateTexture2D(&d,nullptr,&color)) ||
       FAILED(renderer.device->CreateRenderTargetView(color.Get(),nullptr,&target)) ||
       !sandbox_backbuffer_output.draw(target.Get(),d.Width,d.Height))return 2;
    renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
    ID3D11Texture2D* textures[]={color.Get(),sandbox_fresh.glow.linear.depth_texture};
    for(unsigned surface=0;surface<2;++surface){
        textures[surface]->GetDesc(&d);
        if(d.SampleDesc.Count!=1)return 3;
        d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
        if(FAILED(renderer.device->CreateTexture2D(&d,nullptr,&staging)))return 4;
        renderer.context->CopyResource(staging.Get(),textures[surface]);
        D3D11_MAPPED_SUBRESOURCE mapped={};
        if(FAILED(renderer.context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped)))return 5;
        char path[4*MAX_PATH]={};sprintf_s(path,"%s.%s",prefix,surface?"depth":"bmp");
        FILE* file=nullptr;bool ok=!fopen_s(&file,path,"wb") && file;
        if(ok && !surface){
            BITMAPFILEHEADER header={};BITMAPINFOHEADER info={};
            header.bfType=0x4d42;header.bfOffBits=sizeof(header)+sizeof(info);header.bfSize=header.bfOffBits+d.Width*d.Height*4;
            info.biSize=sizeof(info);info.biWidth=d.Width;info.biHeight=-int(d.Height);info.biPlanes=1;info.biBitCount=32;
            ok=fwrite(&header,sizeof(header),1,file)==1 && fwrite(&info,sizeof(info),1,file)==1;
        }else if(ok){unsigned extent[]={d.Width,d.Height,unsigned(d.Format)};ok=fwrite(extent,sizeof(extent),1,file)==1;}
        for(unsigned y=0;y<d.Height && ok;++y)
            ok=fwrite(static_cast<char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,std::size_t(d.Width)*4,1,file)==1;
        if(file)ok=fclose(file)==0 && ok;
        renderer.context->Unmap(staging.Get(),0);if(!ok)return 6;
    }
    std::printf("CAMERA_CAPTURE prefix=%s translation=%.6f,%.6f depth_translation=%.6f geometry_epoch=%llu projection_zoom=%.6f\n",
        prefix,renderer.geometry_viewport_settings.translation[0],renderer.geometry_viewport_settings.translation[1],
        renderer.geometry_viewport_settings.depth_translation,static_cast<unsigned long long>(renderer.tile_geometry_epoch),sandbox_fresh.projection_zoom);
    std::fflush(stdout);
    return 0;
}

#ifdef C3X_RENDERER64_FRESH
// Renderer64 supplies the copied Civ III frame and owns GPU publication. This
// adapter writes the sandbox scene into its map image, never an HWND or RPC page.
void c3x_renderer64_frame_device(){
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
}
bool c3x_renderer64_prepare_scene_assets(){
    c3x_renderer64_frame_device();
    bool cold=!sandbox_fresh.visual.installed || !sandbox_fresh.shadow.ready || !sandbox_fresh.bloom.blur ||
        !sandbox_fresh.static_restore.pixel || !sandbox_direct_units.pixel || !sandbox_backbuffer_output.pixel;
    LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);
    bool ready=sandbox_fresh.prepare_assets() && sandbox_direct_units.initialize() && sandbox_backbuffer_output.ensure();
    QueryPerformanceCounter(&end);
    char detail[160];sprintf_s(detail,"ready=%u cold=%u shadow_bytes=%zu ms=%.3f",unsigned(ready),unsigned(cold),
        sandbox_fresh.shadow.bytes(),renderer.trace.milliseconds(end.QuadPart-begin.QuadPart));
    renderer.trace.write("load-fresh-programs",detail,true);
    return ready;
}
void c3x_renderer64_retire_geometry_selection(){
    sandbox_fresh.retire_geometry_selection();
}
void c3x_renderer64_begin_unit_assets(){
    c3x_renderer64_frame_device();
    // Completed fronts own pixels, not these borrowed per-pass palette leases.
    // Release them before the next frame can evict or adopt any GPU mesh.
    sandbox_direct_units.prepared_units.clear();
    static std::uint64_t revision=0;
    if(revision!=renderer.unit_bodies.catalogue_generation){
        sandbox_direct_units.meshes.clear();sandbox_direct_units.mesh_bytes=0;
        sandbox_direct_units.transitions.clear();revision=renderer.unit_bodies.catalogue_generation;
    }
}
bool c3x_renderer64_select_units(c3x_renderer_frame_v1 const& frame,
        std::vector<c3x_renderer::render_core::UnitInstances::ScenePose> const& candidates,
        std::vector<c3x_renderer::render_core::UnitInstances::ScenePose>& required,float zoom){
    c3x_renderer64_frame_device();return sandbox_fresh.select_unit_contributors(frame,candidates,required,zoom);
}
std::uint64_t c3x_renderer64_unit_selection_revision(){return sandbox_fresh.unit_contribution_revision;}
int c3x_renderer64_prepare_unit_meshes(){
    c3x_renderer64_frame_device();return sandbox_direct_units.prepare_frame_meshes();
}
std::size_t c3x_renderer64_unit_source_meshes_ready(){
    auto const& sources=renderer.unit_bodies.meshes;auto const& meshes=sandbox_direct_units.meshes;
    std::size_t ready=0;
    for(std::size_t i=0;i<sources.size() && i<meshes.size();++i)if(sources[i].source_pinned &&
            meshes[i].vertices && meshes[i].indices && meshes[i].palette_view)++ready;
    return ready;
}
std::size_t c3x_renderer64_unit_source_mesh_bytes(){
    c3x_renderer64_frame_device();return sandbox_direct_units.mesh_bytes;
}
bool c3x_renderer64_static_refinement_pending(float zoom){
    c3x_renderer64_frame_device();
    return sandbox_fresh.static_refinement_pending(zoom);
}
// GPU memory by owner (performance goals, G1: Civ VI memory direction).
// Textures and targets are sized from their descriptions, each resource once;
// owners that keep their own byte ledgers report those. The remainder of the
// adapter's usage is reported as unattributed.
void c3x_renderer64_memory_census(unsigned long long dxgi_usage){
    c3x_renderer64_frame_device();
    c3x_renderer::render_core::ResourceCensus census;
    auto& f=sandbox_fresh;
    auto linear=[&](c3x_renderer::render_core::LinearTarget const& t){
        return census.add(t.color)+census.add(t.resolved)+census.add(t.depth_texture);};
    std::size_t slots=0,overlays=0,previews=0;
    for(auto const& state:f.static_rasters.states)slots+=linear(state.region);
    for(auto const& slot:f.overlay_slots)overlays+=linear(slot.layer);
    for(auto const& image:f.bootstrap)previews+=linear(image.region);
    std::size_t cache=linear(f.static_cache);
    std::size_t targets=linear(f.material_albedo)+linear(f.terrain_albedo)+linear(f.reflected_terrain_albedo)+
        census.add(f.reflection.color)+census.add(f.reflection.depth_texture)+
        census.add(f.reflection_static.color)+census.add(f.reflection_static.depth_texture)+
        census.add(f.bloom.color[0])+census.add(f.bloom.color[1])+
        census.add(f.glow.color)+census.add(f.glow.validity)+census.add(f.glow.native)+linear(f.glow.linear);
    for(auto const* channel:{&f.material_normal,&f.material_world,&f.terrain_normal,&f.terrain_world,&f.terrain_properties,
            &f.reflected_terrain_normal,&f.reflected_terrain_world,&f.reflected_terrain_properties,&f.aquatic_scene,&f.water_lighting})
        targets+=census.add(channel->texture);
    std::size_t shadow=census.add(f.shadow.texture)+f.shadow.production_field_bytes;
    auto& r=renderer;
    std::size_t frame=linear(r.unit_scene_work)+linear(r.reflection.linear)+linear(r.region_glow.linear)+
        census.add(r.region_glow.color)+census.add(r.region_glow.validity)+census.add(r.region_glow.native)+
        linear(r.city_glow.linear)+census.add(r.city_glow.color)+census.add(r.city_glow.validity)+census.add(r.city_glow.native);
    std::size_t textures=0;
    auto terrain=[&](TerrainTexture const& t){
        textures+=census.add(t.view)+census.add(t.material_height_view)+census.add(t.specular_view)+
            census.add(t.elevated_view)+census.add(t.elevated_height_view)+census.add(t.elevated_specular_view);
        for(auto* view:t.relief_layer_views)textures+=census.add(view);
        for(auto* view:t.water_surface_views)textures+=census.add(view);};
    for(auto const& t:r.terrain_textures)terrain(t);
    terrain(r.dune_surface);
    for(auto* view:r.compiled_material_views())textures+=census.add(view);
    for(auto* view:r.feature_texture_views)textures+=census.add(view);
    for(auto* view:r.resource_texture_views)textures+=census.add(view);
    for(auto* view:r.terrain_extra_views)textures+=census.add(view);
    for(auto* view:r.wave_views)textures+=census.add(view);
    std::size_t geometry=r.tile_geometry_cache_bytes+r.retired_content->bytes.load()+r.terrain_patch_index_bytes;
    std::size_t instances=r.shared_instances.gpu_bytes()+r.ordered_rigid_packets.gpu_bytes()+r.wave_geometry_bytes;
    std::size_t units=sandbox_direct_units.gpu_preparation_bytes()+r.unit_bodies.gpu_content_bytes;
    std::size_t regions=r.render_regions.gpu_bytes+r.reflection_regions.gpu_bytes;
    std::size_t composition=r.composition_working_bytes();
    std::size_t sum=slots+overlays+previews+cache+targets+shadow+frame+textures+geometry+instances+units+regions+composition;
    auto mb=[](std::size_t value){return double(value)/(1024.*1024.);};
    char detail[640];sprintf_s(detail,
        "adapter_mb=%.1f attributed_mb=%.1f unattributed_mb=%.1f static_slots_mb=%.1f static_overlays_mb=%.1f static_previews_mb=%.1f static_cache_mb=%.1f "
        "scene_targets_mb=%.1f shadow_mb=%.1f frame_targets_mb=%.1f textures_mb=%.1f geometry_mb=%.1f instances_mb=%.1f units_mb=%.1f "
        "region_caches_mb=%.1f composition_mb=%.1f",
        mb(std::size_t(dxgi_usage)),mb(sum),dxgi_usage>sum?mb(std::size_t(dxgi_usage)-sum):0.,mb(slots),mb(overlays),mb(previews),mb(cache),
        mb(targets),mb(shadow),mb(frame),mb(textures),mb(geometry),mb(instances),mb(units),mb(regions),mb(composition));
    renderer.trace.write("memory-census",detail,true);
}
bool c3x_renderer64_render_fresh(c3x_renderer_frame_v1 const& frame,
        ID3D11RenderTargetView* target,float zoom) {
    // Per-frame records cost OutputDebugString + formatting on every visual
    // frame; keep them for the explicit detailed trace level only.
    bool frame_trace=renderer.trace.level>=2;
    if(frame_trace)renderer.trace.write("fresh-callback","enter",true);
    c3x_renderer64_frame_device();
    int camera_x=0,camera_y=0;
    if(frame.tile_count && frame.tiles) {
        auto const& first=frame.tiles[0];
        camera_x=first.anchor_x-first.tile_x*frame.tile_width/2;
        camera_y=first.anchor_y-first.tile_y*frame.tile_height/2;
    }
    if(!sandbox_fresh.draw(frame,camera_x,camera_y,0,0,0,0,false,zoom))return false;
    if(frame_trace)renderer.trace.write("fresh-callback","scene-ready",true);
    ++renderer.route_frame_sequence;
    char route[8]={};
    if(c3x_renderer::render_core::cached_environment("C3X_RENDERER_ROUTE_WITNESS",route,sizeof(route))&&route[0]=='1'){
        std::uint64_t facts=14695981039346656037ull,poses=facts,ordered=facts,source=facts;
        auto hash=[](std::uint64_t& value,void const* data,std::size_t size){
            auto bytes=static_cast<unsigned char const*>(data);
            for(std::size_t i=0;i<size;++i){value^=bytes[i];value*=1099511628211ull;}
        };
        // The copied frame/ordered occurrences certify supplied source facts.
        // Pointer addresses and asynchronous scheduling are not source content.
        auto captured=frame;captured.tiles=nullptr;captured.world_topology=nullptr;
        captured.presentation_time_ticks=captured.presentation_frequency=0;
        captured.dirty_flags=captured.visible_animation_count=0;
        bool source_complete=frame.tile_count<=8192&&(!frame.tile_count||frame.tiles)&&
            frame.world_topology_count<=1024u*1024u&&(!frame.world_topology_count||frame.world_topology);
        hash(source,&captured,sizeof(captured));
        if(source_complete){
            hash(source,frame.tiles,std::size_t(frame.tile_count)*sizeof(*frame.tiles));
            hash(source,frame.world_topology,std::size_t(frame.world_topology_count)*sizeof(*frame.world_topology));
        }
        auto const& records=sandbox_direct_units.prepared_units;
        for(auto const& unit:records){
            auto draw=unit.draw;draw.presentation_time_ticks=draw.presentation_frequency=0;
            hash(poses,&draw,sizeof(draw));
            // Sampled cursor/clock vary with asynchronous display opportunities.
            // The representative proof keeps identity, action, anchors, art,
            // palette and pass membership; sampled pose has its own digest.
            draw.action_cursor=0;hash(facts,&draw,sizeof(draw));
            // Version 1 preserves the existing authoritative draw/action
            // normalization and native occurrence order BEFORE pass membership.
            hash(ordered,&draw,sizeof(draw));
            hash(ordered,&unit.instance.unit,sizeof(unit.instance.unit));
            hash(ordered,&unit.instance.action,sizeof(unit.instance.action));
            unsigned mask=(unit.main?1u:0u)|(unit.shadow?2u:0u)|(unit.reflected?4u:0u);
            hash(facts,&mask,sizeof(mask));hash(facts,&unit.instance.unit,sizeof(unit.instance.unit));
            hash(facts,&unit.instance.action,sizeof(unit.instance.action));
        }
        char detail[896];sprintf_s(detail,
            "source_serial=%lld source_generation=%llu complete=1 overflow=%u count=%zu main_units=%u reflected_units=%u shadow_units=%u part_samples=%u facts_digest=%016llx pose_digest=%016llx ordered_facts_version=1 ordered_facts_digest=%016llx source_complete=%u source_digest=%016llx camera_x=%d camera_y=%d zoom=%.6f",
            renderer.route_map_serial,static_cast<unsigned long long>(renderer.route_frame_sequence),unsigned(records.size()>4096),records.size(),
            sandbox_direct_units.main_contributors,sandbox_direct_units.reflection_contributors,sandbox_direct_units.shadow_contributors,
            sandbox_direct_units.part_samples,static_cast<unsigned long long>(facts),static_cast<unsigned long long>(poses),static_cast<unsigned long long>(ordered),unsigned(source_complete),static_cast<unsigned long long>(source),camera_x,camera_y,zoom);
        renderer.trace.write("route-workload",detail,true);
        unsigned segments=unsigned((std::min<std::size_t>(records.size(),4096)+31)/32);
        for(unsigned segment=0;segment<segments;++segment){
            std::string ids,masks;auto end=std::min<std::size_t>(records.size(),(segment+1)*32);
            for(std::size_t i=segment*32;i<end;++i){auto const& unit=records[i];
                if(!ids.empty()){ids+=",";masks+=",";}
                ids+=std::to_string(unit.draw.unit_id);masks+=std::to_string((unit.main?1u:0u)|(unit.shadow?2u:0u)|(unit.reflected?4u:0u));
            }
            sprintf_s(detail,"source_serial=%lld source_generation=%llu segment=%u total_segments=%u unit_ids=%s pass_masks=%s",
                renderer.route_map_serial,static_cast<unsigned long long>(renderer.route_frame_sequence),segment,segments,ids.c_str(),masks.c_str());
            renderer.trace.write("route-workload-members",detail,true);
        }
    }
    // These passes share the immediate context. Resource dependencies are
    // ordered there; submit once at publication instead of flushing mid-frame.
    if(frame_trace) {
        char detail[1024];
        auto const& rejects=renderer.raster_proof_rejections;
        sprintf_s(detail,"prepare=%.3f reflection=%.3f static=%.3f water=%.3f units=%.3f reconstruct=%.3f shadow_builds=%u reflection_draws=%u static_draws=%u visible=%u culled=%u camera=%d,%d translation=%.1f,%.1f pose_samples=%u part_samples=%u self_shadow_samples=%u self_shadow_reuses=%u self_shadow_overflow=%u main_units=%u reflected_units=%u palette_uploads=%u instance_builds=%u instance_reuses=%u instance_bytes=%zu unit_preparation_gpu_bytes=%zu proof_rejects=%u,%u,%u,%u,%u,%u,%u",
            sandbox_fresh.phases[0],sandbox_fresh.phases[1],sandbox_fresh.phases[2],
            sandbox_fresh.phases[3],sandbox_fresh.phases[4],sandbox_fresh.phases[5],
            sandbox_fresh.shadow.builds,sandbox_fresh.reflection_draws,
            sandbox_fresh.cache_full_draws,sandbox_fresh.visible,sandbox_fresh.culled,
            camera_x,camera_y,renderer.geometry_viewport_settings.translation[0],
            renderer.geometry_viewport_settings.translation[1],
            sandbox_direct_units.required_samples,sandbox_direct_units.part_samples,
            sandbox_direct_units.shadow_samples,sandbox_direct_units.shadow_reuses,
            sandbox_direct_units.shadow_overflow,sandbox_direct_units.main_contributors,
            sandbox_direct_units.reflection_contributors,sandbox_direct_units.palette_uploads,
            sandbox_fresh.instance_plan_builds,sandbox_fresh.instance_plan_reuses,sandbox_fresh.instance_plan_bytes,
            sandbox_direct_units.gpu_preparation_bytes(),
            rejects[0],rejects[1],rejects[2],rejects[3],rejects[4],rejects[5],rejects[6]);
        renderer.trace.write("fresh-scene-phases",detail,true);
        if(sandbox_fresh.work.enabled){
            // Heavy passes only: per-layer draw calls (C3X_SANDBOX_PASS_COUNTS=1).
            for(unsigned pass=0;pass<SandboxPassCounts::pass_count;++pass){
                std::uint64_t total=0;for(auto const& row:sandbox_fresh.work.counts[pass])total+=row.draws;
                if(total<100)continue;
                std::string line="pass="+std::to_string(pass)+" draws="+std::to_string(total)+" layers=";
                for(unsigned layer=0;layer<SandboxPassCounts::layers;++layer){auto const& row=sandbox_fresh.work.counts[pass][layer];
                    if(row.draws)line+=std::to_string(layer)+":"+std::to_string(row.draws)+"/"+std::to_string(row.accepted_records)+",";}
                renderer.trace.write("pass-census",line.c_str(),true);
            }
        }
        sprintf_s(detail,"part_samples=%u main_units=%u reflected_units=%u material_buffer_builds=%u material_buffer_reuses=%u material_buffer_uploads=%u material_upload_fallbacks=%u unit_preparation_gpu_bytes=%zu self_shadow_ms=%.3f",
            sandbox_direct_units.part_samples,sandbox_direct_units.main_contributors,sandbox_direct_units.reflection_contributors,
            sandbox_direct_units.material_buffer_builds,sandbox_direct_units.material_buffer_reuses,
            sandbox_direct_units.material_buffer_uploads,sandbox_direct_units.material_upload_fallbacks,
            sandbox_direct_units.gpu_preparation_bytes(),sandbox_direct_units.shadow_ms);
        renderer.trace.write("fresh-unit-material-work",detail,true);
        sprintf_s(detail,"bounds_builds=%u bounds_reuses=%u reflections_removed=%u main_units=%u shadow_units=%u reflected_units=%u",
            sandbox_direct_units.reflection_bounds_builds,sandbox_direct_units.reflection_bounds_reuses,
            sandbox_direct_units.reflection_bounds_rejected,sandbox_direct_units.main_contributors,
            sandbox_direct_units.shadow_contributors,sandbox_direct_units.reflection_contributors);
        renderer.trace.write("fresh-unit-reflection-selection",detail,true);
        auto const& spans=sandbox_fresh.prepare_subspans;auto const& placements=renderer.shared_instances;
        auto validation=sandbox_fresh.static_validation_counts();char validation_detail[2048];
        sprintf_s(validation_detail,"raster_full=%llu raster_content=%llu raster_visibility=%llu raster_membership=%llu raster_regions=%llu raster_reuses=%llu raster_changes=%llu raster_visibility_rejects=%llu atlas_full=%llu atlas_content=%llu atlas_membership=%llu atlas_regions=%llu atlas_reuses=%llu receiver_visits=%llu receiver_builds=%llu receiver_reuses=%llu placement_probes=%llu placement_reuses=%llu raster_registrations=%llu raster_watches=%llu raster_source_expansions=%llu raster_source_reuses=%llu raster_append_ms=%.3f atlas_registrations=%llu atlas_watches=%llu atlas_source_expansions=%llu atlas_source_reuses=%llu atlas_append_ms=%.3f",
            validation.raster.full,validation.raster.content,validation.raster.visibility,validation.raster.membership,validation.raster.regions,validation.raster.reused,validation.raster.changes,validation.raster.visibility_rejects,
            validation.atlas.full,validation.atlas.content,validation.atlas.membership,validation.atlas.regions,validation.atlas.reused,
            validation.receiver_visits,validation.receiver_builds,validation.receiver_reuses,validation.placement_probes,validation.placement_reuses,
            validation.raster.proof_registrations,validation.raster.dependency_watch_calls,validation.raster.source_expansions,validation.raster.source_reuses,validation.raster.append_ms,
            validation.atlas.proof_registrations,validation.atlas.dependency_watch_calls,validation.atlas.source_expansions,validation.atlas.source_reuses,validation.atlas.append_ms);
        renderer.trace.write("static-validation-counts",validation_detail,true);
        sprintf_s(validation_detail,"zoom=%.6f previews=%llu bootstraps=%llu bootstrap_scale=%.3f repairs=%llu repair_pixels=%llu promotions=%llu slices=%llu restarts=%u recenter_copies=%llu",
            sandbox_fresh.projection_zoom,sandbox_fresh.preview_frames,sandbox_fresh.bootstrap_draws,sandbox_perf_options().bootstrap_scale,
            sandbox_fresh.partial_repairs,sandbox_fresh.partial_repair_pixels,sandbox_fresh.refine_promotions,
            sandbox_fresh.refine_slices,sandbox_fresh.refine_restarts,sandbox_fresh.recenter_copies);
        renderer.trace.write("static-presentation",validation_detail,true);
        sprintf_s(validation_detail,"caster_collections=%llu caster_reuses=%llu group_builds=%llu group_reuses=%llu body_placement_builds=%llu body_placement_reuses=%llu terrain_batch_builds=%llu terrain_batch_reuses=%llu page_reuses=%llu page_rebuilds=%llu page_refused=%llu latest_rebuild_draws=%llu projections=%llu projection_reuses=%llu page_tests=%llu contributor_edits=%llu instance_projections=%llu page_sorts=%llu draw_projections=%llu proof_bytes=%zu page_bytes=%zu producers=%zu dependency_sources=%zu",
            validation.caster_collections,validation.caster_reuses,validation.shadow_group_builds,validation.shadow_group_reuses,
            validation.body_placement_builds,validation.body_placement_reuses,validation.terrain_batch_builds,validation.terrain_batch_reuses,
            validation.shadow_page_reuses,validation.shadow_page_rebuilds,validation.shadow_page_refused,validation.shadow_draws,
            validation.shadow_projections,validation.shadow_projection_reuses,validation.shadow_page_tests,validation.shadow_contributor_edits,
            validation.shadow_instance_projections,validation.shadow_page_sorts,validation.shadow_draw_projections,
            validation.shadow_proof_bytes,validation.shadow_page_bytes,
            validation.shadow_producers,validation.shadow_dependency_sources);
        renderer.trace.write("fresh-shadow-retention",validation_detail,true);
        auto const& requirements=sandbox_fresh.body_requirements;
        sprintf_s(detail,"setup_resources_ms=%.3f capture_ms=%.3f raster_proof_ms=%.3f body_requirements_ms=%.3f city_shadow_ms=%.3f unit_selection_ms=%.3f unit_pose_ms=%.3f body_builds=%u body_reuses=%u body_visits=%u body_unique=%zu body_duplicates=%u coverage_probes=%u requirement_bytes=%zu unit_plan_reused=%u unit_reselected=%u range_reuses=%llu carry_visits=%llu carried_ranges=%llu packed_records=%llu gpu_copies=%llu copied_bytes=%llu allocated_bytes=%llu host_uploaded_bytes=%llu",
            spans[0],spans[1],spans[2],spans[3],spans[4],spans[5],spans[6],
            sandbox_fresh.body_requirement_builds,sandbox_fresh.body_requirement_reuses,sandbox_fresh.body_requirement_visits,
            requirements.entries.size(),requirements.duplicates,requirements.coverage_probes,requirements.bytes(),
            sandbox_fresh.prepare_unit_plan_reused,sandbox_fresh.prepare_unit_reselected,
            static_cast<unsigned long long>(placements.range_reuses),static_cast<unsigned long long>(placements.carry_visits),
            static_cast<unsigned long long>(placements.carried_ranges),static_cast<unsigned long long>(placements.packed_records),
            static_cast<unsigned long long>(placements.gpu_copies),static_cast<unsigned long long>(placements.copied_bytes),
            static_cast<unsigned long long>(placements.allocated_bytes),static_cast<unsigned long long>(placements.uploaded_bytes));
        renderer.trace.write("fresh-prepare-work",detail,true);
        auto const& dynamic=sandbox_fresh.dynamic_subspans;auto const& calls=sandbox_fresh.dynamic_calls;
        auto const& constants=sandbox_fresh.phase_constant_counts;
        sprintf_s(detail,"depth_setup_ms=%.3f aquatic_ms=%.3f water_scene_ms=%.3f resources_ms=%.3f waves_ms=%.3f depth_draws=%llu depth_copies=%llu depth_uploads=%llu aquatic_draws=%llu aquatic_copies=%llu aquatic_uploads=%llu water_draws=%llu water_copies=%llu water_uploads=%llu resource_draws=%llu resource_copies=%llu resource_uploads=%llu wave_draws=%llu wave_copies=%llu wave_uploads=%llu water_records=%llu water_constant_updates=%llu water_constant_hits=%llu wave_records=%llu wave_constant_updates=%llu wave_constant_hits=%llu wave_chunks=%zu wave_geometry_bytes=%zu",
            dynamic[0],dynamic[1],dynamic[2],dynamic[3],dynamic[4],
            calls[0].draws,calls[0].copies,calls[0].uploads,calls[1].draws,calls[1].copies,calls[1].uploads,
            calls[2].draws,calls[2].copies,calls[2].uploads,calls[3].draws,calls[3].copies,calls[3].uploads,
            calls[4].draws,calls[4].copies,calls[4].uploads,
            constants.water_records,constants.water_updates,constants.water_hits,constants.wave_records,constants.wave_updates,constants.wave_hits,renderer.wave_chunks.size(),renderer.wave_geometry_bytes);
        renderer.trace.write("fresh-dynamic-work",detail,true);
        auto const& prepared=sandbox_fresh.water_parameters;
        sprintf_s(detail,"prepared_misses=%llu prepared_builds=%llu prepared_reuses=%llu prepared_fallbacks=%llu prepared_batches=%u prepared_records=%llu reused_records=%llu prepared_uploads=%llu prepared_bytes=%llu resident_bytes=%zu metadata_bytes=%zu cold_upload_ms=%.3f",
            static_cast<unsigned long long>(prepared.misses),static_cast<unsigned long long>(prepared.builds),static_cast<unsigned long long>(prepared.reuses),
            static_cast<unsigned long long>(prepared.fallbacks),prepared.batch_count,
            static_cast<unsigned long long>(prepared.prepared_records),static_cast<unsigned long long>(prepared.reused_records),
            static_cast<unsigned long long>(prepared.uploads),static_cast<unsigned long long>(prepared.uploaded_bytes),
            prepared.gpu_bytes(),prepared.metadata_bytes(),prepared.cold_upload_ms);
        renderer.trace.write("fresh-water-submission",detail,true);
        auto const& pulled=sandbox_fresh.pulled_pages;
        sprintf_s(detail,"draws=%llu records=%llu fallbacks=%llu builds=%llu reuses=%llu rebuilds=%llu refusals=%llu retired=%llu copied_bytes=%llu resident_bytes=%zu entries=%zu",
            static_cast<unsigned long long>(sandbox_fresh.pulled_draws),static_cast<unsigned long long>(sandbox_fresh.pulled_records),
            static_cast<unsigned long long>(sandbox_fresh.pulled_fallbacks),static_cast<unsigned long long>(pulled.builds),
            static_cast<unsigned long long>(pulled.reuses),static_cast<unsigned long long>(pulled.rebuilds),
            static_cast<unsigned long long>(pulled.refusals),static_cast<unsigned long long>(pulled.retired),
            static_cast<unsigned long long>(pulled.copied_bytes),pulled.bytes(),pulled.entry_count());
        renderer.trace.write("fresh-pulled-submission",detail,true);
        sprintf_s(detail,"hits=%llu misses=%llu strips=%llu refusals=%llu miss_unshifted=%llu miss_revision=%llu miss_layer=%llu miss_last_layer=%u retired_mismatch=%llu retired_failure=%llu preview_hits=%llu",
            static_cast<unsigned long long>(sandbox_fresh.overlay_hits),static_cast<unsigned long long>(sandbox_fresh.overlay_misses),
            static_cast<unsigned long long>(sandbox_fresh.overlay_strips),static_cast<unsigned long long>(sandbox_fresh.overlay_refusals),
            static_cast<unsigned long long>(sandbox_fresh.overlay_miss_unshifted),static_cast<unsigned long long>(sandbox_fresh.overlay_miss_revision),
            static_cast<unsigned long long>(sandbox_fresh.overlay_miss_layer),sandbox_fresh.overlay_miss_last_layer,
            static_cast<unsigned long long>(sandbox_fresh.overlay_retired_mismatch),static_cast<unsigned long long>(sandbox_fresh.overlay_retired_failure),
            static_cast<unsigned long long>(sandbox_fresh.overlay_preview_hits));
        renderer.trace.write("fresh-overlay-layer",detail,true);
        auto const& packets=renderer.ordered_rigid_packets;
        sprintf_s(detail,"builds=%llu reuses=%llu refusals=%llu draws=%llu submitted_records=%llu uploaded_bytes=%llu placement_copies=%llu pages=%u resident_bytes=%zu metadata_bytes=%zu peak_bytes=%zu",
            static_cast<unsigned long long>(packets.builds),static_cast<unsigned long long>(packets.reuses),
            static_cast<unsigned long long>(packets.refusals),static_cast<unsigned long long>(packets.draws),
            static_cast<unsigned long long>(packets.drawn_records),static_cast<unsigned long long>(packets.uploaded_bytes),
            static_cast<unsigned long long>(packets.placement_copies),packets.page_count(),packets.gpu_bytes(),packets.metadata_bytes(),packets.peak_bytes());
        renderer.trace.write("fresh-ordered-rigid-submission",detail,true);
        sprintf_s(detail,"builds=%llu reuses=%llu schedules=%llu probes=%llu proof_bytes=%zu proof_cap_bytes=98304",
            static_cast<unsigned long long>(renderer.unit_asset_union_builds),static_cast<unsigned long long>(renderer.unit_asset_union_reuses),
            static_cast<unsigned long long>(renderer.unit_asset_schedules),static_cast<unsigned long long>(renderer.unit_asset_union_probes),
            renderer.unit_asset_union_bytes);
        renderer.trace.write("unit-asset-union-work",detail,true);
    }
    bool presented=sandbox_backbuffer_output.draw(target,frame.target_width,frame.target_height);
    char const* failed_step=presented?nullptr:"output";
    if(presented){
        Microsoft::WRL::ComPtr<ID3D11Resource> resource;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        target->GetResource(&resource);
        presented=SUCCEEDED(resource.As(&texture)) &&
            sandbox_fresh.city_site_overlay.draw(renderer.device,renderer.context,
                texture.Get(),frame,sandbox_fresh.glow.linear.depth_texture,4,zoom);
    }
    if(!presented && !failed_step)failed_step="site_overlay";
    // The sandbox draws a fully visible scene. Game frames finish with the
    // existing GPU fog pass, including independent animation frames. Apply it
    // to this publication target, preserving the reusable HDR scene underneath.
    if(presented && renderer.visibility_pass){
        auto visible=renderer.arrival_visibility.sample(frame,renderer.topology_cache.scope_sequence());
        // Newly shown sight, timed from Civ III's visibility capture (QPC).
        if(renderer.arrival_visibility.revealed&&renderer.trace.level&&renderer.arrival_visibility.revealed_frequency>0){
            LARGE_INTEGER now={};QueryPerformanceCounter(&now);char detail[160];
            std::snprintf(detail,sizeof(detail),"cells=%u capture_age_ms=%.1f",renderer.arrival_visibility.revealed,
                1000.*double(now.QuadPart-renderer.arrival_visibility.revealed_capture)/double(renderer.arrival_visibility.revealed_frequency));
            renderer.trace.write("reveal-shown",detail,true);
        }
        presented=renderer.arrival_coverage.capture(visible,renderer.topology_cache.scope_sequence(),
            renderer.topology_cache.visibility_sequence(),renderer.content_revision,renderer.device_generation);
        Microsoft::WRL::ComPtr<ID3D11Resource> resource;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        target->GetResource(&resource);
        presented=presented && SUCCEEDED(resource.As(&texture)) &&
            renderer.visibility_gpu.apply(renderer.device,renderer.context,
                texture.Get(),renderer.arrival_coverage,
                sandbox_fresh.glow.linear.depth_texture,4,zoom);
    }
    if(!presented && !failed_step)failed_step="fog";
    if(failed_step)renderer.trace.write("fresh-callback-failed-step",failed_step,true);
    if(presented && renderer.trace.level && sandbox_fresh.work.enabled){
        char census[8]{};
        bool requested=GetEnvironmentVariableA("C3X_RENDERER_SUBMISSION_CENSUS",census,sizeof(census)) && census[0]=='1';
        unsigned zoom_bits=0;std::memcpy(&zoom_bits,&zoom,sizeof(zoom_bits));
        std::array<std::uint64_t,12> key={std::uint64_t(renderer.route_map_serial),
            sandbox_fresh.view_revision(),sandbox_fresh.visibility_revision,renderer.device_generation,
            renderer.content_revision,std::uint32_t(camera_x),std::uint32_t(camera_y),
            (std::uint64_t(frame.target_width)<<32)|frame.target_height,zoom_bits,
            std::uint64_t(renderer.water_scene_active),sandbox_fresh.reflection_revision,
            std::uint64_t(renderer.city_profile)*2+unsigned(renderer.pickup_profile)};
        if(sandbox_fresh.submission_census.observe(requested,key)){
            char detail[640];
            sprintf_s(detail,"source_serial=%lld source_generation=%llu camera_x=%d camera_y=%d zoom=%.6f view_revision=%llu visibility_revision=%llu complete=1 stable_callbacks=3 water_active=%u wave_pack_ready=%u reflection_enabled=%u",
                renderer.route_map_serial,static_cast<unsigned long long>(renderer.route_frame_sequence),camera_x,camera_y,zoom,
                static_cast<unsigned long long>(sandbox_fresh.view_revision()),static_cast<unsigned long long>(sandbox_fresh.visibility_revision),
                unsigned(renderer.water_scene_active),unsigned(renderer.wave_ready),unsigned(renderer.reflection.enabled));
            renderer.trace.write("submission-census",detail,true);
            for(unsigned pass=0;pass<SandboxPassCounts::pass_count;++pass)
                for(unsigned layer=0;layer<SandboxPassCounts::layers;++layer){
                    auto const& row=sandbox_fresh.work.counts[pass][layer];
                    if(!row.tested_records && !row.tested_instances && !row.draws && !row.upload_bytes &&
                       !row.target_pixels && !row.copy_pixels && !row.rebuilds && !row.reuses)continue;
                    sprintf_s(detail,"source_serial=%lld source_generation=%llu pass=%u layer=%u tested=%llu accepted=%llu tested_instances=%llu accepted_instances=%llu submitted_instances=%llu index_vertices=%llu triangles=%llu draws=%llu upload_bytes=%llu target_pixels=%llu copy_pixels=%llu rebuilds=%llu reuses=%llu",
                        renderer.route_map_serial,static_cast<unsigned long long>(renderer.route_frame_sequence),pass,layer,
                        row.tested_records,row.accepted_records,row.tested_instances,row.accepted_instances,
                        row.submitted_instances,row.index_vertices,row.triangles,row.draws,row.upload_bytes,
                        row.target_pixels,row.copy_pixels,row.rebuilds,row.reuses);
                    renderer.trace.write("submission-census-layer",detail,true);
                }
        }
    }
    if(frame_trace || !presented)renderer.trace.write("fresh-callback",presented?"map-ready":"map-failed",true);
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
        DXGI_SWAP_CHAIN_DESC1 actual={};BOOL fullscreen=FALSE;
        if(FAILED(swap->GetDesc1(&actual)) || FAILED(swap->GetFullscreenState(&fullscreen,nullptr)))return 3;
        std::printf("SANDBOX_SWAPCHAIN width=%u height=%u samples=%u buffers=%u format=%u swap_effect=%u scaling=%u exclusive=%d\n",
            actual.Width,actual.Height,actual.SampleDesc.Count,actual.BufferCount,unsigned(actual.Format),
            unsigned(actual.SwapEffect),unsigned(actual.Scaling),int(fullscreen));
        std::fflush(stdout);
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
    // Capture the first requested view, including standalone custom fixtures.
    bool initial_capture = !captured_initial;
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
        char linear_path[4*MAX_PATH]{};
        if(initial_capture && GetEnvironmentVariableA("C3X_SANDBOX_HDR_CAPTURE",linear_path,sizeof(linear_path))) {
            if(!sandbox_capture_linear(linear_path))return 1;
            std::puts("CLIENT_HDR_CAPTURE pass");
        }
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
