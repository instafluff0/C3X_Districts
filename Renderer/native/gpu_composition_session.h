#pragma once
#include "gpu_frame_api.h"
#include "gpu_image_compositor.h"
#include "retained_composition.h"
namespace c3x_gpu_images {
// Lives exclusively on RendererWorker, with its existing immediate context.
// A map is immutable; native composition writes separately owned images.
class Session {
    static constexpr unsigned live_image_budget=256u*1024u*1024u;
    ID3D11Device* device;ID3D11DeviceContext* context;
    Compositor gpu;RetainedComposition layers;Id map=0;std::int64_t ticket=0,identity=0;std::uint64_t readbacks=0;
    Id resident_unit=0;ID3D11Texture2D* resident_unit_texture=nullptr;
    bool map_animation_expected=false;
public:
    Session(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c),gpu(d,c,live_image_budget,true),layers(d,c){}
    // Eight fullscreen packed/full-color native pairs and old/new immutable
    // maps require about 194 MiB at 2240x1260. Bound live images at 256 MiB,
    // including small UI sources; retained replay has its separate budget.
    bool publish(ID3D11Texture2D* texture,std::int64_t serial,int x=0,int y=0,int width=0,int height=0,RetainedComposition::Sample sample={}){
        if(!texture||serial<=ticket)return false;
        D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
        if(!width)width=int(d.Width);if(!height)height=int(d.Height);
        auto next=gpu.create(width,height,Format::bgra32);
        if(!next){char message[224];auto counts=gpu.stats();
            sprintf_s(message,"[C3X renderer] stage=map-publication-rejected reason=canvas-admission width=%d height=%d resident_bytes=%llu cap_bytes=%u\n",
                width,height,counts.resident_bytes,live_image_budget);OutputDebugStringA(message);return false;}

        if(!gpu.import_bgra(next,texture,x,y)){gpu.destroy(next);return false;}
        // Admission failure leaves the previous immutable map and UI handles
        // usable. Publish the new identity only after its import succeeds.
        if(map){layers.destroy(map);gpu.destroy(map);}map=next;map_animation_expected=bool(sample);
        try{
            // A rejected history is rebuilt only at fresh authoritative map
            // demand. Current native images become immutable static inputs;
            // subsequent map writes restore their dynamic dependencies.
            if(!layers.accepting()){
                layers.clear();gpu.visit_images([&](Id id,unsigned w,unsigned h,Format format,ID3D11Texture2D* source){
                    layers.create(id,w,h,format);layers.source(id,source);
                });
            }
            layers.create(map,width,height,Format::bgra32);layers.source(map,gpu.texture(map),std::move(sample),true,true);}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
        ticket=serial;if(!identity)identity=serial;return true;
    }
    std::int64_t session_identity()const{return identity;}
    Id map_image()const{return map;}
    std::uint64_t upload_count()const{return gpu.stats().uploads;}
    std::int64_t current_ticket()const{return ticket;}
    bool display_to(std::int64_t requested,Id image,ID3D11RenderTargetView* target,ID3D11Texture2D* retained,ID3D11Texture2D* buffer,unsigned w,unsigned h,Rect area){
        if(requested!=ticket||!gpu.display(image,target,w,h,area))return false;
        context->CopyResource(buffer,retained);context->Flush();
        try{layers.commit(image,area);}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}return true;
    }
    int compose_resident_unit(c3x_renderer_gpu_unit_v1 const& request,ID3D11Texture2D* texture,unsigned width,unsigned height,int x,int y,RetainedComposition::Sample sample={}){
        if(request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        if(request.destination==std::int64_t(map)||request.detail==std::int64_t(map))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(!texture||!width||!height||width>1024||height>1024)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(texture!=resident_unit_texture){
            if(resident_unit){layers.destroy(resident_unit);gpu.destroy(resident_unit);}resident_unit=0;resident_unit_texture=nullptr;
            resident_unit=gpu.attach_source(texture);if(!resident_unit)return C3X_RENDERER_RESULT_BAD_ARGUMENT;resident_unit_texture=texture;
        }else gpu.record_external(resident_unit);
        try{layers.create(resident_unit,width,height,Format::bgra32);layers.source(resident_unit,texture,std::move(sample),true);}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
        Command draw={Kind::unit_over,Id(request.destination),resident_unit,{x,y,x+int(width),y+int(height)},
            {request.clip[0],request.clip[1],request.clip[2],request.clip[3]},0,0,0,Id(request.background),Id(request.detail),Id(request.background_detail)};
        bool ok=gpu.submit(&draw,1);if(ok)try{layers.record(draw);}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
        return ok?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
    int draw_dynamic(c3x_renderer_gpu_unit_v1 const& request,unsigned width,unsigned height,int x,int y,RetainedComposition::Direct operation){
        if(request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        if(request.destination==std::int64_t(map)||request.detail==std::int64_t(map)||!operation.draw)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        Command draw={Kind::unit_over,Id(request.destination),0,{x,y,x+int(width),y+int(height)},
            {request.clip[0],request.clip[1],request.clip[2],request.clip[3]},0,0,0,Id(request.background),Id(request.detail),Id(request.background_detail)};
        bool ok=operation.draw(gpu,draw);
        if(ok)try{layers.record(draw,std::move(operation));}catch(std::exception const& e){OutputDebugStringA(e.what());layers.discard();}
        return ok?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
    RetainedComposition::Texture snapshot_bgra(ID3D11Texture2D* source,int x,int y,unsigned w,unsigned h){return layers.snapshot_bgra(source,x,y,w,h);}
    std::uint64_t visual_bytes()const{return layers.bytes();}
    std::size_t allocation_bytes()const{return std::size_t(gpu.stats().resident_bytes+layers.bytes());}
    std::size_t visual_nodes()const{return layers.node_count();}
    std::size_t visual_sources()const{return layers.sampled_sources();}
    // Correct static pixels alone do not certify ambient delivery. A CPU
    // snapshot can sever map samples while unit animation remains reachable.
    // Let the existing native recovery demand run until map writes restore it.
    bool visual_ready()const{return layers.ready()&&(!map_animation_expected||layers.animated_map());}
    bool visual_active()const{return layers.ready()&&layers.animated();}
    void stop_visuals(){layers.uncommit();}
    int visual_frame(long long ticks,long long frequency,ID3D11RenderTargetView* target,ID3D11Texture2D* display,ID3D11Texture2D* buffer){
        try{auto result=layers.draw(ticks,frequency,target,display,buffer);
            c3x_recording::event(c3x_recording::visual,0,[&](auto& b){using namespace c3x_recording;u64(b,std::uint64_t(ticks));u64(b,std::uint64_t(frequency));u32(b,unsigned(result));u64(b,layers.bytes());u64(b,layers.node_count());u64(b,layers.sampled_sources());u32(b,visual_ready());});
            return result;}catch(std::exception const& e){
            // A failed recipe cannot produce a frame. Release its outputs now;
            // the last completed display stays intact, and the next native map
            // rebuilds retained history from authoritative current images.
            MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);GlobalMemoryStatusEx(&memory);
            char status[224];sprintf_s(status,"[C3X renderer] stage=visual-failure-memory device_reason=0x%08lx available_virtual=%llu available_pagefile=%llu\n",
                device->GetDeviceRemovedReason(),memory.ullAvailVirtual,memory.ullAvailPageFile);
            OutputDebugStringA(status);OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();return false;}
    }
    int execute(c3x_renderer_gpu_images_v1 const& request,std::vector<Command> const& commands,
                std::vector<unsigned> const& pixels,c3x_renderer_gpu_result_v1& result,std::vector<unsigned>& output){
        output.clear();result={sizeof(result)};
        if(!ticket||request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        // Draws and the immediately following resource boundary share one
        // owner handoff. Submission order and explicit CPU readback stay exact.
        if(!commands.empty()){
            for(auto const& c:commands)if(c.destination==map||c.detail==map)return request.action==C3X_GPU_SUBMIT?C3X_RENDERER_RESULT_BAD_ARGUMENT:C3X_RENDERER_RESULT_ERROR;
            if(!gpu.submit(commands.data(),commands.size()))return request.action==C3X_GPU_SUBMIT?C3X_RENDERER_RESULT_BAD_ARGUMENT:C3X_RENDERER_RESULT_ERROR;
            try{for(auto const& command:commands)layers.record(command);}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
        }
        bool ok=false;Id image=Id(request.image);
        if(request.action==C3X_GPU_CREATE){image=gpu.create(request.width,request.height,request.format==C3X_GPU_RGB555?Format::rgb555:request.format==C3X_GPU_RGB565?Format::rgb565:Format::bgra32);ok=image!=0;if(ok)layers.create(image,request.width,request.height,request.format==C3X_GPU_RGB555?Format::rgb555:request.format==C3X_GPU_RGB565?Format::rgb565:Format::bgra32);}
        else if(request.action==C3X_GPU_UPLOAD){auto before=gpu.stats().uploads;ok=image!=map&&request.revision>0&&gpu.upload(image,request.revision,pixels.data(),pixels.size());
            if(ok&&gpu.stats().uploads!=before)try{layers.source(image,gpu.texture(image));}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}}
        else if(request.action==C3X_GPU_DESTROY){ok=image!=map&&gpu.destroy(image);if(ok)layers.destroy(image);}
        else if(request.action==C3X_GPU_SUBMIT)ok=true;
        else if(request.action==C3X_GPU_READBACK){
            auto texture=gpu.texture(image);if(!texture)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
            if(std::uint64_t(d.Width)*d.Height>request.pixel_count)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            output.resize(std::size_t(d.Width)*d.Height);
            d.BindFlags=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
            D3D11_MAPPED_SUBRESOURCE data={};checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&data));
            for(unsigned y=0;y<d.Height;++y)std::memcpy(output.data()+std::size_t(y)*d.Width,static_cast<char*>(data.pData)+std::size_t(y)*data.RowPitch,d.Width*4);
            context->Unmap(stage.Get(),0);++readbacks;ok=true;gpu.record_readback(image,output.data(),output.size());
        }
        auto counts=gpu.stats();result.image=std::int64_t(image);result.pixel_count=unsigned(output.size());
        result.resident_bytes=std::int64_t(counts.resident_bytes);result.uploads=std::int64_t(counts.uploads);
        result.commands=std::int64_t(counts.commands);result.readbacks=std::int64_t(readbacks);
        return ok?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
};
}
