#define NOMINMAX
#include <windows.h>
#include "gpu_composition_session.h"
#include <cassert>
#include <cstdio>
#include <chrono>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_gpu_images;
std::vector<unsigned> hud_read(ID3D11Device* d,ID3D11DeviceContext* c,ID3D11Texture2D* t){
    D3D11_TEXTURE2D_DESC desc={};t->GetDesc(&desc);unsigned w=desc.Width,h=desc.Height;
    desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> stage;checked(d->CreateTexture2D(&desc,nullptr,&stage));c->CopyResource(stage.Get(),t);
    D3D11_MAPPED_SUBRESOURCE m={};checked(c->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));std::vector<unsigned> out(w*h);
    for(unsigned y=0;y<h;++y)std::memcpy(out.data()+y*w,static_cast<char*>(m.pData)+y*m.RowPitch,w*4);
    c->Unmap(stage.Get(),0);return out;
}
int test_fullscreen_hud_recipe(){
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
    constexpr unsigned w=2240,h=1260;Rect full={0,0,w,h};
    for(auto format:{Format::rgb555,Format::rgb565}){
        std::vector<unsigned> pixels(w*h),output;
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)pixels[y*w+x]=0xff000000|((x*13&255)<<16)|((y*7&255)<<8)|(x+y)%256;
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={pixels.data(),w*4,0};ComPtr<ID3D11Texture2D> source;
        checked(device->CreateTexture2D(&desc,&initial,&source));
        bool frozen=false;unsigned active_camera=0;
        Session session(device.Get(),context.Get());
        auto make_sample=[&](unsigned camera){return [&,camera](long long,long long){return frozen||camera!=active_camera?RetainedComposition::SampledImage::frozen():RetainedComposition::SampledImage::bgra(source.Get(),full);};};
        assert(session.publish(source.Get(),1,0,0,w,h,make_sample(0)));
        c3x_renderer_gpu_images_v1 request={};c3x_renderer_gpu_result_v1 result={};
        auto create=[&](unsigned width,unsigned height,int f){request={};request.struct_size=sizeof(request);request.ticket=session.current_ticket();request.action=C3X_GPU_CREATE;
            request.width=width;request.height=height;request.format=f;assert(session.execute(request,{}, {},result,output)==1);return Id(result.image);};
        auto send=[&](std::vector<Command> const& commands){request={};request.struct_size=sizeof(request);request.ticket=session.current_ticket();request.action=C3X_GPU_SUBMIT;
            assert(session.execute(request,commands,{},result,output)==1);};
        auto upload=[&](Id id,std::vector<unsigned> const& values){request={};request.struct_size=sizeof(request);request.ticket=session.current_ticket();request.action=C3X_GPU_UPLOAD;
            request.image=id;request.revision=1;assert(session.execute(request,{},values,result,output)==1);};
        int f=format==Format::rgb555?C3X_GPU_RGB555:C3X_GPU_RGB565;unsigned key=format==Format::rgb555?0x7c1f:0xf81f;
        auto map_words=create(w,h,f),units=create(w,h,f),unit_detail=create(w,h,C3X_GPU_BGRA32);
        auto screen=create(w,h,f),detail=create(w,h,C3X_GPU_BGRA32),saved=create(w,h,f),saved_detail=create(w,h,C3X_GPU_BGRA32);
        auto sprite=create(80,24,C3X_GPU_BGRA32),textlut=create(17,256,C3X_GPU_BGRA32);
        std::vector<unsigned> sprite_pixels(80*24),lookup(17*256);
        for(unsigned i=0;i<sprite_pixels.size();++i)sprite_pixels[i]=0x80201008u+i%16;
        for(unsigned i=0;i<lookup.size();++i)lookup[i]=(i/17+i%17)%256;
        upload(sprite,sprite_pixels);upload(textlut,lookup);
        send({{Kind::quantize,map_words,session.map_image(),full,full},
              {Kind::fill,units,0,full,full,0,0,key},{Kind::fill,unit_detail,0,full,full,0,0,0xffff00ff}});
        struct Placed {Command command;int x,y;};std::vector<Placed> draws;
        for(unsigned i=0;i<140;++i){
            int x=16+(i%14)*155,y=16+(i/14)*122;Rect box={x,y,x+80,y+24};
            std::vector<Command> label={{Kind::fill,units,0,box,full,0,0,0x3def},
                {Kind::fill,unit_detail,0,box,full,0,0,0xff406080},
                {Kind::native_text,units,sprite,box,box,0,0,0,textlut},
                {Kind::native_blend,units,sprite,box,box,0,0,1,units,unit_detail,unit_detail},
                {Kind::native_blend,units,units,box,box,0,0,2,units,unit_detail,unit_detail,1234,141}};
            send({{Kind::hud_begin,units,0,{}, {},x+40,y+12,i+1,0,unit_detail,0,int(key)}});
            send(label);send({{Kind::hud_end}});
            for(auto c:label)draws.push_back({c,x+40,y+12});
        }
        auto boundary=[&]{send({{Kind::world_begin,screen,map_words,full,full,0,0,65536,0,detail,session.map_image(),int(w),int(h)},
                               {Kind::world_end,screen,units,full,full,0,0,key,0,detail,unit_detail,int(w),int(h)}});};
        boundary();assert(session.commit_display(session.current_ticket(),detail,w,h,full));
        desc.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> display,buffer;
        checked(device->CreateTexture2D(&desc,nullptr,&display));checked(device->CreateTexture2D(&desc,nullptr,&buffer));
        ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(display.Get(),nullptr,&target));
        LARGE_INTEGER now={},frequency={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&now);
        // Independent native compositor oracle: transform the complete map
        // once, then submit each native primitive in the same order.
        Compositor oracle(device.Get(),context.Get(),128u*1024u*1024u);
        auto ground=oracle.create(w,h,Format::bgra32);assert(oracle.upload(ground,1,pixels.data(),pixels.size()));
        auto color_id=oracle.create(w,h,Format::bgra32),word_id=oracle.create(w,h,Format::bgra32);
        auto color=oracle.release_import_target(color_id),words=oracle.release_import_target(word_id);
        auto os=oracle.create(80,24,Format::bgra32),ol=oracle.create(17,256,Format::bgra32);
        assert(oracle.upload(os,1,sprite_pixels.data(),sprite_pixels.size()));assert(oracle.upload(ol,1,lookup.data(),lookup.size()));
        double total_ms=0;unsigned frames=0;std::uint64_t first_peak=0;
        for(unsigned camera=0;camera<5;++camera){
            if(camera){
                active_camera=camera;
                for(auto& pixel:pixels)pixel=0xff000000|((pixel+0x17230bu)&0xffffff);
                context->UpdateSubresource(source.Get(),0,nullptr,pixels.data(),w*4,0);
                assert(session.publish(source.Get(),camera+1,0,0,w,h,make_sample(camera)));
                assert(oracle.upload(ground,camera+1,pixels.data(),pixels.size()));
                send({{Kind::quantize,map_words,session.map_image(),full,full}});
                boundary();assert(session.commit_display(session.current_ticket(),detail,w,h,full));
            }
            for(unsigned frame=0;frame<6;++frame){
                send({{Kind::zoom_target,0,0,{}, {},0,0,frame%2?65536u:98304u}});QueryPerformanceCounter(&now);
                auto tick=now.QuadPart+frequency.QuadPart*(frame==2?40:280)/1000;
                auto start=std::chrono::steady_clock::now();
                assert(session.visual_frame(tick,frequency.QuadPart,target.Get(),display.Get(),buffer.Get())!=0);
                auto actual=hud_read(device.Get(),context.Get(),display.Get());
                total_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();++frames;
                double scale=session.visual_scale();
                assert(oracle.transform_view(color,ground,float(scale),&words,format));
                auto ow=oracle.attach_target_unrecorded(words.texture.Get(),format),od=oracle.attach_target_unrecorded(color.texture.Get(),Format::bgra32);
                for(auto const& placed:draws){auto c=placed.command;
                    auto rebind=[&](Id id){return id==units?ow:id==unit_detail?od:id==sprite?os:id==textlut?ol:Id(0);};
                    c.destination=rebind(c.destination);c.source=rebind(c.source);c.background=rebind(c.background);
                    c.detail=rebind(c.detail);c.background_detail=rebind(c.background_detail);
                    int dx=int(std::lround((placed.x-int(w/2))*(scale-1))),dy=int(std::lround((placed.y-int(h/2))*(scale-1)));
                    c.area={c.area.left+dx,c.area.top+dy,c.area.right+dx,c.area.bottom+dy};
                    c.clip={c.clip.left+dx,c.clip.top+dy,c.clip.right+dx,c.clip.bottom+dy};
                    assert(oracle.submit(&c,1));
                }
                auto expected=hud_read(device.Get(),context.Get(),color.texture.Get());
                if(actual!=expected){unsigned at=0;while(actual[at]==expected[at])++at;
                    std::fprintf(stderr,"fullscreen HUD mismatch camera=%u frame=%u scale=%.9f pixel=%u actual=%08x expected=%08x\n",camera,frame,scale,at,actual[at],expected[at]);assert(false);}
                oracle.destroy(ow);oracle.destroy(od);
                if(!camera&&!frame)first_peak=session.allocation_peak();
                // Capture both immutable output versions through native copies.
                if(frame==0)send({{Kind::copy,saved,screen,full,full},{Kind::copy,saved_detail,detail,full,full}});
                assert(session.visual_bytes()<160u*1024u*1024u);
                assert(session.allocation_bytes()<=session.allocation_peak());
            }
        }
        std::printf("PASS fullscreen HUD recipe: format=%u dimensions=%ux%u labels=140 commands=700 frames=%u exact=1 retained=%llu reachable=%zu peak=%llu first_peak=%llu submit_plus_readback_mean_ms=%.3f operations=%u copies=%u copied_pixels=%llu\n",
            unsigned(format),w,h,frames,session.visual_bytes(),session.allocation_bytes(),session.allocation_peak(),first_peak,total_ms/frames,
            session.visual_work().operations,session.visual_work().copies,session.visual_work().copied_pixels);
        frozen=true;QueryPerformanceCounter(&now);
        assert(session.visual_frame(now.QuadPart+frequency.QuadPart,frequency.QuadPart,target.Get(),display.Get(),buffer.Get())!=0);
        // A retired batch becomes completed immutable pixels, without future
        // placements or a retained map callback. Existing small native oracles
        // cover partial commits and old/new map camera replacement as well.
        auto last=hud_read(device.Get(),context.Get(),display.Get());
        send({{Kind::zoom_target,0,0,{}, {},0,0,196608}});
        assert(session.visual_frame(now.QuadPart+frequency.QuadPart*2,frequency.QuadPart,target.Get(),display.Get(),buffer.Get())!=0);
        assert(hud_read(device.Get(),context.Get(),display.Get())==last);
    }
    {
        // Actual batch versions, including one absent from the current front.
        // Replacing the source and reversing the shared zoom must not mutate
        // its completed old packed/detail pixels or its paired self-copy.
        Compositor native(device.Get(),context.Get(),128u*1024u*1024u);
        RetainedComposition retained(device.Get(),context.Get());
        auto map=native.create(w,h,Format::bgra32),words=native.create(w,h,Format::rgb565),detail=native.create(w,h,Format::bgra32);
        std::vector<unsigned> pixels(w*h,0xff345678);assert(native.upload(map,1,pixels.data(),pixels.size()));
        for(auto id:{map,words,detail})retained.create(id,w,h,id==words?Format::rgb565:Format::bgra32);
        bool expired=false;
        retained.source(map,native.texture(map),[&](long long,long long){return expired?RetainedComposition::SampledImage::frozen():RetainedComposition::SampledImage(native.texture(map));},true,true);
        auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();
        Rect label={200,140,280,164},copy={310,180,390,204};
        std::vector<RetainedComposition::Placed> draws={
            {{Kind::fill,words,0,label,full,0,0,0x03ef},240,152},
            {{Kind::fill,detail,0,label,full,0,0,0xff123456},240,152},
            {{Kind::copy,detail,detail,copy,full,label.left,label.top},350,192},
            {{Kind::copy,words,words,copy,full,label.left,label.top},350,192}};
        retained.view(detail,map,zoom,words);retained.placed_batch(words,detail,draws,zoom);
        Id saved_words=10000,saved_detail=10001;
        retained.snapshot(saved_words,words);retained.snapshot(saved_detail,detail);
        retained.commit(detail,full);auto old_color=hud_read(device.Get(),context.Get(),retained.sample(1,1000).Get());
        retained.commit(words,full);auto old_words=hud_read(device.Get(),context.Get(),retained.sample(2,1000).Get());
        expired=true;
        auto next=native.create(w,h,Format::bgra32);std::fill(pixels.begin(),pixels.end(),0xffa0b0c0);
        assert(native.upload(next,1,pixels.data(),pixels.size()));
        retained.source(map,native.texture(next),[](long long,long long){return RetainedComposition::SampledImage{};},true,true);
        zoom->target(1.5,3,1000);retained.view(detail,map,zoom,words);
        draws[0].command.color=0x7fff;draws[1].command.color=0xff8899aa;
        retained.placed_batch(words,detail,draws,zoom);retained.commit(detail,full);
        auto current=hud_read(device.Get(),context.Get(),retained.sample(500,1000).Get());assert(current!=old_color);
        Rect dirty={160,100,500,300};retained.commit(saved_detail,dirty);
        auto partial=hud_read(device.Get(),context.Get(),retained.sample(501,1000).Get());
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
            assert(partial[y*w+x]==(x>=unsigned(dirty.left)&&x<unsigned(dirty.right)&&y>=unsigned(dirty.top)&&y<unsigned(dirty.bottom)?old_color[y*w+x]:current[y*w+x]));
        retained.commit(saved_detail,full);
        assert(hud_read(device.Get(),context.Get(),retained.sample(502,1000).Get())==old_color);
        retained.commit(saved_words,full);
        assert(hud_read(device.Get(),context.Get(),retained.sample(503,1000).Get())==old_words);
        std::printf("PASS fullscreen HUD saved generations: absent_front=1 paired_self_copy=1 source_replaced=1 reversed_zoom=1 partial_commit_exact=1 bytes=%llu unique_peak=%llu\n",retained.bytes(),retained.allocation_peak());
        retained.clear();assert(!retained.node_count()&&!retained.bytes()&&!retained.allocation_bytes());
    }
    return 0;
}
