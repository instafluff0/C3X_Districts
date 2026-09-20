#define NOMINMAX
#include <windows.h>
#include "gpu_composition_session.h"
#include <cassert>
#include <cstdio>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_gpu_images;
std::vector<unsigned> retained_read(ID3D11Device* d,ID3D11DeviceContext* c,ID3D11Texture2D* t){
    D3D11_TEXTURE2D_DESC desc={};t->GetDesc(&desc);auto w=desc.Width,h=desc.Height;
    desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> read;checked(d->CreateTexture2D(&desc,nullptr,&read));c->CopyResource(read.Get(),t);
    D3D11_MAPPED_SUBRESOURCE m={};checked(c->Map(read.Get(),0,D3D11_MAP_READ,0,&m));std::vector<unsigned> out(w*h);
    for(unsigned y=0;y<h;++y)std::memcpy(out.data()+y*w,static_cast<char*>(m.pData)+y*m.RowPitch,w*4);
    c->Unmap(read.Get(),0);return out;
}
int test_retained_composition(){
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
    constexpr unsigned w=48,h=32;Rect full={0,0,w,h},part={5,3,38,26};unsigned checks=0;
    for(auto format:{Format::rgb555,Format::rgb565}){
        Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto create=[&](unsigned x,unsigned y,Format f){auto id=live.create(x,y,f);assert(id);retained.create(id,x,y,f);return id;};
        auto map=create(w,h,Format::bgra32),unit=create(w,h,Format::bgra32),native=create(w,h,format),detail=create(w,h,Format::bgra32),
            back=create(w,h,format),bd=create(w,h,Format::bgra32),sprite=create(w,h,Format::bgra32),words=create(w,h,format),
            lut=create(1024,1024,Format::bgra32),textlut=create(17,256,Format::bgra32),final=create(w,h,Format::bgra32);
        unsigned key=format==Format::rgb555?0x7c1f:0xf81f;
        std::vector<unsigned> pixels(w*h);auto upload=[&](Id id,unsigned salt,bool packed=false){
            for(unsigned i=0;i<pixels.size();++i)pixels[i]=packed?(i%3?((i*salt)&(format==Format::rgb555?32767:65535)):key):0xff000000u|((i*salt)&0xffffff);
            assert(live.upload(id,1,pixels.data(),pixels.size()));retained.source(id,live.texture(id));};
        upload(map,3211);upload(unit,111);upload(words,31,true);upload(sprite,73);
        std::vector<unsigned> lookup(1024*1024);for(unsigned i=0;i<lookup.size();++i)lookup[i]=(i^31)&(format==Format::rgb555?32767:65535);
        assert(live.upload(lut,1,lookup.data(),lookup.size()));retained.source(lut,live.texture(lut));
        lookup.resize(17*256);for(unsigned i=0;i<lookup.size();++i)lookup[i]=(i/17+i%17)%256;
        assert(live.upload(textlut,1,lookup.data(),lookup.size()));retained.source(textlut,live.texture(textlut));
        // Immutable dynamic sources, replaced by the renderer clock alone.
        ComPtr<ID3D11Texture2D> current_map=live.texture(map),current_unit=live.texture(unit);unsigned samples=0;
        retained.source(map,current_map.Get(),[&](long long,long long){++samples;return current_map;});
        retained.source(unit,current_unit.Get(),[&](long long,long long){++samples;return current_unit;});
        std::vector<Command> commands;
        auto draw=[&](Command c){assert(live.submit(&c,1));retained.record(c);commands.push_back(c);};
        draw({Kind::quantize,native,map,full,full,0,0,0});draw({Kind::copy,detail,map,full,full});
        draw({Kind::copy,back,native,full,full});draw({Kind::copy,bd,detail,full,full});
        draw({Kind::unit_over,native,unit,full,part,0,0,0,back,detail,bd});
        draw({Kind::native_image,native,words,{2,1,42,31},part,0,0,65536,0,detail,0,48,32});
        draw({Kind::native_image,native,words,full,part,0,0,key,0,detail,0,48,32});
        draw({Kind::native_image,native,native,{3,2,43,30},full,1,1,65536,0,detail,detail,40,28});
        draw({Kind::native_blend,native,sprite,full,part,0,0,1,native,detail,detail});
        draw({Kind::native_blend,native,native,full,part,0,0,2,native,detail,detail,1234,141});
        draw({Kind::native_lookup,native,lut,full,part,0,0,2,native,detail,detail});
        draw({Kind::native_sprite,native,sprite,full,part});
        draw({Kind::native_text,native,sprite,full,part,0,0,0,textlut});
        draw({Kind::invert,native,0,full,part,0,0,32767});
        draw({Kind::unit_over,native,unit,full,{39,0,48,32},0,0,0,back,detail,bd});
        draw({Kind::expand,final,native,full,full,0,0,65536});
        draw({Kind::copy,final,detail,{0,0,4,32},full});
        retained.commit(final,full);
        auto compare=[&](){auto out=retained.sample(checks+1,1000);assert(out);
            auto expected=retained_read(device.Get(),context.Get(),live.texture(final)),actual=retained_read(device.Get(),context.Get(),out.Get());
            for(unsigned i=0;i<expected.size();++i)if(expected[i]!=actual[i]){std::printf("retained mismatch check=%u pixel=%u expected=%x actual=%x\n",checks,i,expected[i],actual[i]);std::fflush(stdout);assert(false);}++checks;};
        compare();auto baseline=retained.bytes();
        for(unsigned tick=0;tick<60;++tick){
            // A fresh texture represents a finished renderer pose/map sample.
            auto m=live.create(w,h,Format::bgra32),u=live.create(w,h,Format::bgra32);assert(m&&u);
            for(unsigned i=0;i<pixels.size();++i)pixels[i]=0xff000000|((i*3127+tick*13)&0xffffff);
            assert(live.upload(m,1,pixels.data(),pixels.size()));assert(live.upload(map,tick+2,pixels.data(),pixels.size()));
            for(unsigned i=0;i<pixels.size();++i){auto a=(i+tick)%256;pixels[i]=(a<<24)|((a/2)<<8)|(a/3);}
            assert(live.upload(u,1,pixels.data(),pixels.size()));assert(live.upload(unit,tick+2,pixels.data(),pixels.size()));
            current_map=live.texture(m);current_unit=live.texture(u);assert(live.submit(commands.data(),commands.size()));compare();
            live.destroy(m);live.destroy(u);assert(retained.bytes()<=baseline+48*32*8);
        }
        assert(samples==122); // no native capture, upload, or primitive record during sixty visual frames
        // Completed front is immutable until the native transfer; opaque UI
        // overwrites retire dynamic dependencies exactly where they cover them.
        Command fill={Kind::fill,final,0,full,full,0,0,0xff345678};assert(live.submit(&fill,1));retained.record(fill);
        auto before=retained_read(device.Get(),context.Get(),retained.sample(999,1000).Get());assert(before[0]!=fill.color);
        retained.commit(final,part);auto partial=retained_read(device.Get(),context.Get(),retained.sample(1000,1000).Get());
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)assert(partial[y*w+x]==(x>=5&&x<38&&y>=3&&y<26?fill.color:before[y*w+x]));
        retained.commit(final,full);auto old_samples=samples;compare();assert(samples==old_samples);
        auto bounded=retained.node_count();for(unsigned i=0;i<1000;++i){retained.record(fill);retained.commit(final,full);}compare();assert(retained.node_count()<=bounded+1);
        // Procedural unit-style writes sample state before executing against
        // retained underlays. Repeated phases reuse the completed composition;
        // a new phase updates once, and an opaque native UI write retires it.
        std::uint64_t phase=1;unsigned direct_draws=0;
        RetainedComposition::Direct procedural;procedural.animated=true;
        procedural.revision=[&](long long,long long){return phase;};
        procedural.draw=[&](Compositor& target,Command const& input){++direct_draws;auto command=input;
            command.color=0xff000000u|unsigned(phase);return target.submit(&command,1);};
        retained.record(fill,procedural);retained.commit(final,full);
        auto first=retained_read(device.Get(),context.Get(),retained.sample(1001,1000).Get());assert(first[0]==0xff000001u && direct_draws==1);
        retained.sample(1002,1000);assert(direct_draws==1);
        ++phase;auto second=retained_read(device.Get(),context.Get(),retained.sample(1003,1000).Get());assert(second[0]==0xff000002u && direct_draws==2);
        retained.record(fill);retained.commit(final,full);retained.sample(1004,1000);assert(direct_draws==2);
        retained.clear();
        for(auto id:{map,native,detail})retained.create(id,w,h,id==native?format:Format::bgra32);
        retained.source(map,live.texture(map));
        retained.record({Kind::quantize,native,map,full,full});
        retained.record({Kind::copy,detail,map,full,full});
        unsigned scene_draws=0;RetainedComposition::Direct scene_pass;scene_pass.animated=true;
        scene_pass.revision=[&](long long,long long){return phase;};
        scene_pass.draw=[&](Compositor& target,Command const& c){
            ++scene_draws; // Direct bodies need only the ordered native underlay.
            Command write={Kind::fill,c.detail,0,c.area,c.clip,0,0,0xff123456};return target.submit(&write,1);
        };
        retained.record({Kind::unit_over,native,0,part,part,0,0,0,native,detail,detail},scene_pass);
        retained.commit(detail,full);retained.sample(1005,1000);assert(scene_draws==1);
        ++phase;retained.sample(1006,1000);assert(scene_draws==2);
        // Large copied tactical payloads share the retained memory ceiling;
        // rejected admission cannot alter the existing picture or byte charge.
        auto charge=retained.bytes();RetainedComposition::Direct oversized;oversized.input_bytes=129ull*1024*1024;
        bool rejected=false;try{retained.record({Kind::fill,native,0,full,full},oversized);}catch(...){rejected=true;}
        assert(rejected&&retained.bytes()==charge);
        retained.clear();assert(!retained.ready()&&retained.bytes()==0&&retained.node_count()==0);
        retained.create(final,w,h,Format::bgra32);
        unsigned collected=0,executed=0;
        for(unsigned i=0;i<2;++i){
            auto operation=procedural;
            operation.revision=[&](long long,long long){++collected;return phase;};
            operation.draw=[&](Compositor& target,Command const& input){
                assert(collected==2);++executed;return target.submit(&input,1);};
            auto write=fill;write.area=write.clip={int(i*w/2),0,int((i+1)*w/2),h};
            retained.record(write,operation);
        }
        retained.commit(final,full);retained.sample(1007,1000);assert(collected==2&&executed==2);
        collected=0;retained.sample(1008,1000);assert(collected==2&&executed==2);
        collected=0;++phase;retained.sample(1009,1000);assert(collected==2&&executed==4);
        retained.clear();assert(retained.bytes()==0&&retained.node_count()==0);
    }
    // Native map/screen/save copies refer to immutable content versions. A
    // fullscreen copy chain must not allocate one replay texture per transfer.
    {
        constexpr unsigned width=2240,height=1260;Rect bounds={0,0,width,height};
        Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto map=live.create(width,height,Format::bgra32);assert(map);
        std::vector<unsigned> pixels(width*height);
        for(unsigned i=0;i<pixels.size();++i)pixels[i]=0xff000000u|(i&0xffffffu);
        assert(live.upload(map,1,pixels.data(),pixels.size()));
        retained.create(1,width,height,Format::bgra32);retained.source(1,live.texture(map),[&](long long,long long){return RetainedComposition::Texture(live.texture(map));});
        for(Id id=2;id<=18;++id){retained.create(id,width,height,Format::bgra32);retained.record({Kind::copy,id,id-1,bounds,bounds});}
        retained.commit(18,bounds);
        RetainedComposition::Texture result;
        try{result=retained.sample(1,1000);}catch(std::exception const& e){std::fprintf(stderr,"FAIL fullscreen copy chain: %s bytes=%llu nodes=%zu\n",e.what(),retained.bytes(),retained.node_count());throw;}
        auto actual=retained_read(device.Get(),context.Get(),result.Get());assert(actual==pixels);
        assert(retained.bytes()==std::uint64_t(width)*height*4&&retained.node_count()==1);
        // Replacing the source image must not rewrite the completed front.
        retained.record({Kind::fill,1,0,bounds,bounds,0,0,0xffabcdef});
        assert(retained_read(device.Get(),context.Get(),retained.sample(2,1000).Get())==pixels);
        retained.clear();assert(retained.bytes()==0&&retained.node_count()==0);
        std::printf("PASS fullscreen retained copy chain: transfers=17 pixels=%u retained_bytes=%u nodes=1 immutable_front=1\n",width*height,width*height*4);
    }
    // Fullscreen native map, screen, saved UI and staging pairs remain alive
    // while the next immutable map is published. Eight current canvases plus
    // the temporary next map exceed the old 96 MiB live-image ceiling.
    {
        constexpr unsigned width=2240,height=1260;
        D3D11_TEXTURE2D_DESC desc={};desc.Width=width;desc.Height=height;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        ComPtr<ID3D11Texture2D> source;checked(device->CreateTexture2D(&desc,nullptr,&source));
        Session session(device.Get(),context.Get());assert(session.publish(source.Get(),1));
        std::vector<Command> commands;std::vector<unsigned> pixels,output;std::vector<Id> canvases;
        c3x_renderer_gpu_result_v1 result={};
        for(unsigned i=0;i<7;++i){c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);request.ticket=1;
            request.action=C3X_GPU_CREATE;request.width=width;request.height=height;request.format=C3X_GPU_BGRA32;
            assert(session.execute(request,commands,pixels,result,output)==C3X_RENDERER_RESULT_OK);canvases.push_back(Id(result.image));}
        if(!session.publish(source.Get(),2)){std::fprintf(stderr,"FAIL fullscreen publication: old map plus seven native canvases\n");throw std::runtime_error("fullscreen map publication budget");}
        // Headroom remains a hard limit. Rejection leaves the old map usable;
        // releasing optional canvases lets the same publication retry succeed.
        c3x_renderer_gpu_images_v1 allocate={};allocate.struct_size=sizeof(allocate);allocate.ticket=2;
        allocate.action=C3X_GPU_CREATE;allocate.width=width;allocate.height=height;allocate.format=C3X_GPU_BGRA32;
        std::vector<Id> optional;
        for(unsigned i=0;i<3;++i){assert(session.execute(allocate,commands,pixels,result,output)==C3X_RENDERER_RESULT_OK);optional.push_back(Id(result.image));}
        assert(session.execute(allocate,commands,pixels,result,output)==C3X_RENDERER_RESULT_BAD_ARGUMENT);
        assert(!session.publish(source.Get(),3)&&session.current_ticket()==2);
        for(auto canvas:optional){auto release=allocate;release.action=C3X_GPU_DESTROY;release.image=std::int64_t(canvas);
            assert(session.execute(release,commands,pixels,result,output)==C3X_RENDERER_RESULT_OK);}
        assert(session.publish(source.Get(),3));
        for(auto canvas:canvases){c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);request.ticket=3;
            request.action=C3X_GPU_DESTROY;request.image=std::int64_t(canvas);assert(session.execute(request,commands,pixels,result,output)==C3X_RENDERER_RESULT_OK);}
        assert(result.resident_bytes==std::int64_t(width)*height*4);
        std::puts("PASS fullscreen publication: native_canvases=7 next_map=1 old_map_retired=1 no_CPU_fallback=1");
    }
    std::printf("PASS retained composition: %u exact GPU oracles, 120 independent clock frames, aliasing, paired 555/565/full color, UI versioning, partial publication, bounded overwrite and reset\n",checks);return 0;
}
