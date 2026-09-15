#pragma once
#include "gpu_frame_api.h"
#include "gpu_image_compositor.h"
namespace c3x_gpu_images {
// Lives exclusively on RendererWorker, with its existing immediate context.
// A map is immutable; native composition writes separately owned images.
class Session {
    ID3D11Device* device;ID3D11DeviceContext* context;
    Compositor gpu;Id map=0;std::int64_t ticket=0,identity=0;std::uint64_t readbacks=0;
    Id resident_unit=0;ID3D11Texture2D* resident_unit_texture=nullptr;
public:
    Session(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c),gpu(d,c){}
    bool publish(ID3D11Texture2D* texture,std::int64_t serial,int x=0,int y=0,int width=0,int height=0){
        if(!texture||serial<=ticket)return false;
        D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
        if(!width)width=int(d.Width);if(!height)height=int(d.Height);
        auto next=gpu.create(width,height,Format::bgra32);
        if(!next)return false;
        if(!gpu.import_bgra(next,texture,x,y)){gpu.destroy(next);return false;}
        // Admission failure leaves the previous immutable map and UI handles
        // usable. Publish the new identity only after its import succeeds.
        if(map)gpu.destroy(map);map=next;
        ticket=serial;if(!identity)identity=serial;return true;
    }
    std::int64_t session_identity()const{return identity;}
    Id map_image()const{return map;}
    std::uint64_t upload_count()const{return gpu.stats().uploads;}
    std::int64_t current_ticket()const{return ticket;}
    bool display_to(std::int64_t requested,Id image,ID3D11RenderTargetView* target,ID3D11Texture2D* retained,ID3D11Texture2D* buffer,unsigned w,unsigned h,Rect area){
        if(requested!=ticket||!gpu.display(image,target,w,h,area))return false;
        context->CopyResource(buffer,retained);context->Flush();return true;
    }
    int compose_resident_unit(c3x_renderer_gpu_unit_v1 const& request,ID3D11Texture2D* texture,unsigned width,unsigned height,int x,int y){
        if(request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        if(request.destination==std::int64_t(map)||request.detail==std::int64_t(map))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(!texture||!width||!height||width>1024||height>1024)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(texture!=resident_unit_texture){
            if(resident_unit)gpu.destroy(resident_unit);resident_unit=0;resident_unit_texture=nullptr;
            resident_unit=gpu.attach_source(texture);if(!resident_unit)return C3X_RENDERER_RESULT_BAD_ARGUMENT;resident_unit_texture=texture;
        }
        Command draw={Kind::unit_over,Id(request.destination),resident_unit,{x,y,x+int(width),y+int(height)},
            {request.clip[0],request.clip[1],request.clip[2],request.clip[3]},0,0,0,Id(request.background),Id(request.detail),Id(request.background_detail)};
        return gpu.submit(&draw,1)?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
    int execute(c3x_renderer_gpu_images_v1 const& request,std::vector<Command> const& commands,
                std::vector<unsigned> const& pixels,c3x_renderer_gpu_result_v1& result,std::vector<unsigned>& output){
        output.clear();result={sizeof(result)};
        if(!ticket||request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        bool ok=false;Id image=Id(request.image);
        if(request.action==C3X_GPU_CREATE){image=gpu.create(request.width,request.height,request.format==C3X_GPU_RGB555?Format::rgb555:request.format==C3X_GPU_RGB565?Format::rgb565:Format::bgra32);ok=image!=0;}
        else if(request.action==C3X_GPU_UPLOAD){ok=image!=map&&request.revision>0&&gpu.upload(image,request.revision,pixels.data(),pixels.size());}
        else if(request.action==C3X_GPU_DESTROY){ok=image!=map&&gpu.destroy(image);}
        else if(request.action==C3X_GPU_SUBMIT){
            for(auto const& c:commands)if(c.destination==map||c.detail==map)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            ok=gpu.submit(commands.data(),commands.size());
        }else if(request.action==C3X_GPU_READBACK){
            auto texture=gpu.texture(image);if(!texture)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
            if(std::uint64_t(d.Width)*d.Height>request.pixel_count)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            output.resize(std::size_t(d.Width)*d.Height);
            d.BindFlags=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
            D3D11_MAPPED_SUBRESOURCE data={};checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&data));
            for(unsigned y=0;y<d.Height;++y)std::memcpy(output.data()+std::size_t(y)*d.Width,static_cast<char*>(data.pData)+std::size_t(y)*data.RowPitch,d.Width*4);
            context->Unmap(stage.Get(),0);++readbacks;ok=true;
        }
        auto counts=gpu.stats();result.image=std::int64_t(image);result.pixel_count=unsigned(output.size());
        result.resident_bytes=std::int64_t(counts.resident_bytes);result.uploads=std::int64_t(counts.uploads);
        result.commands=std::int64_t(counts.commands);result.readbacks=std::int64_t(readbacks);
        return ok?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
};
}
