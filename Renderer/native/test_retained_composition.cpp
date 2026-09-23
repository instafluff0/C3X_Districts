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
        auto charge=retained.bytes();RetainedComposition::Direct oversized;oversized.input_bytes=257ull*1024*1024;
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
    // while the next immutable map is published. Eight packed/full-color pairs plus
    // old/new map overlap exercise the complete fullscreen family.
    {
        constexpr unsigned width=2240,height=1260;
        D3D11_TEXTURE2D_DESC desc={};desc.Width=width;desc.Height=height;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        ComPtr<ID3D11Texture2D> source;checked(device->CreateTexture2D(&desc,nullptr,&source));
        Session session(device.Get(),context.Get());assert(session.publish(source.Get(),1));
        std::vector<Command> commands;std::vector<unsigned> pixels,output;std::vector<Id> canvases;
        c3x_renderer_gpu_result_v1 result={};
        for(unsigned i=0;i<16;++i){c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);request.ticket=1;
            request.action=C3X_GPU_CREATE;request.width=width;request.height=height;request.format=C3X_GPU_BGRA32;
            assert(session.execute(request,commands,pixels,result,output)==C3X_RENDERER_RESULT_OK);canvases.push_back(Id(result.image));}
        if(!session.publish(source.Get(),2)){std::fprintf(stderr,"FAIL fullscreen publication: old map plus eight packed/full-color native pairs\n");throw std::runtime_error("fullscreen map publication budget");}
        // Headroom remains a hard limit. Rejection leaves the old map usable;
        // releasing optional canvases lets the same publication retry succeed.
        c3x_renderer_gpu_images_v1 allocate={};allocate.struct_size=sizeof(allocate);allocate.ticket=2;
        allocate.action=C3X_GPU_CREATE;allocate.width=width;allocate.height=height;allocate.format=C3X_GPU_BGRA32;
        std::vector<Id> optional;
        for(unsigned i=0;i<32;++i){if(session.execute(allocate,commands,pixels,result,output)!=C3X_RENDERER_RESULT_OK)break;optional.push_back(Id(result.image));}
        assert(!optional.empty()&&optional.size()<32);
        assert(session.execute(allocate,commands,pixels,result,output)==C3X_RENDERER_RESULT_BAD_ARGUMENT);
        assert(!session.publish(source.Get(),3)&&session.current_ticket()==2);
        for(auto canvas:optional){auto release=allocate;release.action=C3X_GPU_DESTROY;release.image=std::int64_t(canvas);
            assert(session.execute(release,commands,pixels,result,output)==C3X_RENDERER_RESULT_OK);}
        assert(session.publish(source.Get(),3));
        for(auto canvas:canvases){c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);request.ticket=3;
            request.action=C3X_GPU_DESTROY;request.image=std::int64_t(canvas);assert(session.execute(request,commands,pixels,result,output)==C3X_RENDERER_RESULT_OK);}
        assert(result.resident_bytes==std::int64_t(width)*height*4);
        std::puts("PASS fullscreen publication: native_pairs=8 next_map=1 old_map_retired=1 no_CPU_fallback=1");
    }
    // A partial publication from an untouched image has actual zero pixels.
    // Reusing a broad underlay must not paint through those zero patches or
    // through holes in a newly-created, partially copied destination.
    {
        Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto map=live.create(w,h,Format::bgra32);std::vector<unsigned> pixels(w*h,0xff123456);assert(live.upload(map,1,pixels.data(),pixels.size()));
        retained.create(1,w,h,Format::bgra32);retained.source(1,live.texture(map));
        retained.create(2,w,h,Format::bgra32);retained.commit(1,full);retained.commit(2,part);
        auto actual=retained_read(device.Get(),context.Get(),retained.sample(1,1000).Get());
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
            assert(actual[y*w+x]==(x>=5&&x<38&&y>=3&&y<26?0:0xff123456u));
        retained.record({Kind::copy,2,1,part,part,part.left,part.top});retained.commit(2,full);
        actual=retained_read(device.Get(),context.Get(),retained.sample(2,1000).Get());
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
            assert(actual[y*w+x]==(x>=5&&x<38&&y>=3&&y<26?0xff123456u:0));
        std::puts("PASS retained underlay compaction: zero patches and sparse coverage preserved");
    }
    {
        Compositor scratch(device.Get(),context.Get(),4096);
        auto first=scratch.create(16,16,Format::rgb555);assert(first);
        std::vector<unsigned> words(256,0x1234);assert(scratch.upload(first,1,words.data(),words.size()));
        assert(scratch.recycle(first)&&!scratch.texture(first));
        auto reused=scratch.create(16,16,Format::bgra32);assert(reused&&scratch.stats().reuses==1);
        assert(retained_read(device.Get(),context.Get(),scratch.texture(reused))==std::vector<unsigned>(256,0));
        assert(scratch.recycle(reused));auto larger=scratch.create(32,32,Format::bgra32);assert(larger);
        assert(scratch.stats().resident_bytes==4096);assert(scratch.recycle(larger));scratch.clear_recycled();
        assert(scratch.stats().resident_bytes==0);
        std::puts("PASS replay scratch pool: stale handles rejected, storage cleared, format reset, capacity eviction and reset");
    }
    // Native gameplay mutates HUD/outline versions between presentations.
    // A stationary-only replay cannot expose the live family's source overlap
    // or the cost of rebuilding hundreds of small GPU working surfaces.
    {
        constexpr unsigned width=2240,height=1260;Rect bounds={0,0,width,height};
        Compositor live(device.Get(),context.Get(),256u*1024u*1024u);
        RetainedComposition retained(device.Get(),context.Get());
        auto map=live.create(width,height,Format::bgra32),screen=live.create(width,height,Format::bgra32);
        assert(map&&screen);std::vector<unsigned> pixels(width*height,0xff123456);
        assert(live.upload(map,1,pixels.data(),pixels.size()));
        // Eight native packed/full-color pairs can coexist even when only
        // one completed screen is displayed. They remain independently owned.
        for(Id id=100;id<116;++id){retained.create(id,width,height,Format::bgra32);
            try{retained.source(id,live.texture(map));}catch(std::exception const& e){
                std::fprintf(stderr,"FAIL dense HUD source admission: pair_textures=%u bytes=%llu reason=%s\n",unsigned(id-100),retained.bytes(),e.what());throw;}}
        retained.create(map,width,height,Format::bgra32);retained.create(screen,width,height,Format::bgra32);
        RetainedComposition::Texture sample=live.texture(map);
        retained.source(map,sample.Get(),[&](long long,long long){return sample;},true,true);
        std::vector<Command> commands={{Kind::copy,screen,map,bounds,bounds}};
        for(unsigned i=0;i<1000;++i){Rect area={int(16+(i%40)*50),int(16+(i/40)*40),int(24+(i%40)*50),int(26+(i/40)*40)};
            commands.push_back({Kind::invert,screen,0,area,area});}
        assert(live.submit(commands.data(),commands.size()));
        for(auto const& command:commands)retained.record(command);
        retained.commit(screen,bounds);
        auto expected=retained_read(device.Get(),context.Get(),live.texture(screen));
        assert(retained_read(device.Get(),context.Get(),retained.sample(1,1000).Get())==expected);
        auto initial_bytes=retained.bytes();auto warm=retained.replay_stats();double elapsed=0;
        LARGE_INTEGER frequency;QueryPerformanceFrequency(&frequency);
        for(unsigned tick=0;tick<8;++tick){
            auto next=live.create(width,height,Format::bgra32);assert(next);
            std::fill(pixels.begin(),pixels.end(),0xff123456u+tick+1);assert(live.upload(next,1,pixels.data(),pixels.size()));
            sample=live.texture(next);commands[0].source=next;assert(live.submit(commands.data(),commands.size()));
            expected=retained_read(device.Get(),context.Get(),live.texture(screen));
            LARGE_INTEGER begin,end;QueryPerformanceCounter(&begin);
            auto actual=retained_read(device.Get(),context.Get(),retained.sample(tick+2,1000).Get());
            QueryPerformanceCounter(&end);elapsed+=1000.*double(end.QuadPart-begin.QuadPart)/frequency.QuadPart;
            assert(actual==expected);assert(retained.bytes()==initial_bytes);live.destroy(next);
        }
        auto hot=retained.replay_stats();assert(hot.allocations==warm.allocations&&hot.reuses>warm.reuses);
        std::printf("PASS dense fullscreen retained HUD: native_pairs=8 operations=1000 frames=8 exact=1 bytes=%llu mean_ms=%.3f hot_allocations=%llu reused=%llu\n",retained.bytes(),elapsed/8,hot.allocations-warm.allocations,hot.reuses-warm.reuses);
        // Also replace native pictures repeatedly; stationary sampling alone
        // cannot prove retirement when the game redraws UI or publishes maps.
        for(unsigned tick=0;tick<32;++tick){
            auto next=live.create(width,height,Format::bgra32);assert(next);
            std::fill(pixels.begin(),pixels.end(),0xff345678u+tick);assert(live.upload(next,1,pixels.data(),pixels.size()));
            sample=live.texture(next);commands[0].source=next;assert(live.submit(commands.data(),commands.size()));
            retained.source(map,sample.Get(),[texture=sample](long long,long long){return texture;},true,true);
            auto transfer=commands[0];transfer.source=map;retained.record(transfer);
            for(std::size_t i=1;i<commands.size();++i)retained.record(commands[i]);
            retained.commit(screen,bounds);
            assert(retained_read(device.Get(),context.Get(),retained.sample(100+tick,1000).Get())==
                   retained_read(device.Get(),context.Get(),live.texture(screen)));
            assert(retained.bytes()==initial_bytes);live.destroy(next);
        }
        std::puts("PASS dense native replacement: map_publications=32 UI_writes=32000 exact=1 retained_bytes_stable=1");
        retained.clear();assert(retained.bytes()==0&&retained.node_count()==0);
    }
    // Native saved views may retain multiple animated map versions. Their
    // outputs use the retained budget, not the smaller temporary-work budget.
    {
        constexpr unsigned width=2240,height=1260;Rect bounds={0,0,width,height};
        Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto original=live.create(width,height,Format::bgra32);assert(original);
        std::vector<unsigned> pixels(width*height,0xff123456);
        D3D11_TEXTURE2D_DESC desc={};desc.Width=width;desc.Height=height;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={pixels.data(),width*4,0};ComPtr<ID3D11Texture2D> sampled;
        checked(device->CreateTexture2D(&desc,&initial,&sampled));
        retained.create(100,width,height,Format::bgra32);
        for(Id id=1;id<=16;++id){
            retained.create(id,width,height,Format::bgra32);
            retained.source(id,live.texture(original),[&](long long,long long){
                return RetainedComposition::SampledImage::bgra(sampled.Get(),bounds);},true,true);
            int left=int(id-1)*140;Rect strip={left,0,left+140,int(height)};
            retained.record({Kind::copy,100,id,strip,bounds,left,0});
        }
        retained.commit(100,bounds);
        for(unsigned tick=1;tick<=3;++tick){
            std::fill(pixels.begin(),pixels.end(),0xff123456+tick);
            context->UpdateSubresource(sampled.Get(),0,nullptr,pixels.data(),width*4,0);
            assert(retained_read(device.Get(),context.Get(),retained.sample(tick,1000).Get())==pixels);
            assert(retained.sampling_allocations()==16 && retained.sampling_imports()==16*tick);
            assert(retained.bytes()==std::uint64_t(width)*height*4*16);
        }
        retained.clear();assert(!retained.bytes()&&!retained.node_count()&&!retained.replay_stats().resident_bytes);
        std::puts("PASS fullscreen sampled versions: 16 independent owners, three exact frames, 16 total allocations, reset releases all storage");
    }
    // A CPU snapshot can retain correct pixels while losing the map's sample
    // callback. Even an independently animated unit must not certify that
    // frozen map as ready and disable the native recovery scheduler.
    {
        std::vector<unsigned> pixels(w*h,0xff123456),output;
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={pixels.data(),w*4,0};ComPtr<ID3D11Texture2D> source;
        checked(device->CreateTexture2D(&desc,&initial,&source));
        Session session(device.Get(),context.Get());
        assert(session.publish(source.Get(),1,0,0,w,h,
            [&](long long,long long){return RetainedComposition::Texture(source.Get());}));
        c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);request.ticket=1;
        request.action=C3X_GPU_CREATE;request.width=w;request.height=h;request.format=C3X_GPU_BGRA32;
        c3x_renderer_gpu_result_v1 result={};std::vector<Command> commands;
        assert(session.execute(request,commands,{},result,output)==C3X_RENDERER_RESULT_OK);Id canvas=Id(result.image);
        desc.BindFlags=D3D11_BIND_RENDER_TARGET;
        ComPtr<ID3D11Texture2D> display,buffer;
        checked(device->CreateTexture2D(&desc,nullptr,&display));checked(device->CreateTexture2D(&desc,nullptr,&buffer));
        ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(display.Get(),nullptr,&target));
        auto show=[&]{assert(session.display_to(1,canvas,target.Get(),display.Get(),buffer.Get(),w,h,full));};
        auto copy_map=[&]{request.action=C3X_GPU_SUBMIT;commands={{Kind::copy,canvas,session.map_image(),full,full}};
            assert(session.execute(request,commands,{},result,output)==C3X_RENDERER_RESULT_OK);};
        copy_map();show();assert(session.visual_ready()&&session.visual_active());
        // Native transfers must sample the same clock as autonomous frames.
        // The immutable original map stays unchanged; only its sampled source advances.
        ComPtr<ID3D11Texture2D> sampled;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        checked(device->CreateTexture2D(&desc,&initial,&sampled));
        long long last_tick=0;unsigned samples=0;
        assert(session.publish(source.Get(),2,0,0,w,h,[&](long long ticks,long long){
            if(ticks==last_tick)return RetainedComposition::SampledImage{};
            last_tick=ticks;++samples;return RetainedComposition::SampledImage::bgra(sampled.Get(),full);}));
        request.ticket=2;request.action=C3X_GPU_SUBMIT;commands={{Kind::copy,canvas,session.map_image(),full,full}};
        assert(session.execute(request,commands,{},result,output)==C3X_RENDERER_RESULT_OK);
        for(unsigned tick=1;tick<=8;++tick){
            std::fill(pixels.begin(),pixels.end(),0xff123400u+tick);
            context->UpdateSubresource(sampled.Get(),0,nullptr,pixels.data(),w*4,0);
            assert(session.display_to(2,canvas,target.Get(),display.Get(),buffer.Get(),w,h,full,tick*66,1000));
            assert(last_tick==tick*66 && samples==tick);
            assert(retained_read(device.Get(),context.Get(),display.Get())==pixels);
            assert(session.visual_sample_allocations()==1 && session.visual_sample_imports()==tick);
            assert(session.display_to(2,canvas,target.Get(),display.Get(),buffer.Get(),w,h,full,tick*66,1000));
            assert(samples==tick && session.visual_sample_imports()==tick);
            // Native working images are not modified by animation presentation.
            request.action=C3X_GPU_READBACK;request.image=std::int64_t(canvas);request.pixel_count=w*h;
            assert(session.execute(request,{}, {},result,output)==C3X_RENDERER_RESULT_OK);
            assert(output==retained_read(device.Get(),context.Get(),source.Get()));
        }
        std::puts("PASS native transfer animation: eight native-only samples, one allocation, unchanged clocks reuse pixels, native versions preserved");
        assert(session.publish(source.Get(),3,0,0,w,h,[&](long long,long long){return RetainedComposition::Texture(source.Get());}));
        request.ticket=3;
        auto show_current=[&]{assert(session.display_to(3,canvas,target.Get(),display.Get(),buffer.Get(),w,h,full));};
        request.action=C3X_GPU_UPLOAD;request.image=std::int64_t(canvas);request.revision=1;
        assert(session.execute(request,{},pixels,result,output)==C3X_RENDERER_RESULT_OK);show_current();
        if(session.visual_ready()){std::fprintf(stderr,"FAIL frozen CPU map wrongly disables ambient recovery\n");return 1;}
        RetainedComposition::Direct unit;unit.animated=true;unit.revision=[](long long ticks,long long){return std::uint64_t(ticks);};
        unit.draw=[](Compositor& gpu,Command const& input){auto c=input;c.kind=Kind::fill;c.color=0xffabcdef;return gpu.submit(&c,1);};
        c3x_renderer_gpu_unit_v1 draw={};draw.ticket=3;draw.destination=std::int64_t(canvas);draw.clip[2]=w;draw.clip[3]=h;
        assert(session.draw_dynamic(draw,8,8,0,0,std::move(unit))==C3X_RENDERER_RESULT_OK);show_current();
        if(session.visual_ready()){std::fprintf(stderr,"FAIL animated unit masks missing ambient map source\n");return 1;}
        copy_map();show_current();assert(session.visual_ready()&&session.visual_active());
        assert(session.publish(source.Get(),4)); // genuinely static map is ready too
        request.ticket=4;request.action=C3X_GPU_SUBMIT;commands={{Kind::copy,canvas,session.map_image(),full,full}};
        assert(session.execute(request,commands,{},result,output)==C3X_RENDERER_RESULT_OK);
        assert(session.display_to(4,canvas,target.Get(),display.Get(),buffer.Get(),w,h,full));
        assert(session.visual_ready()&&!session.visual_active()); // fog/static: no animation callbacks
        // A failed optional visual sample must not reject a completed native
        // transfer or discard the caller's current GPU canvases. This models an
        // allocation failure without consuming the host's real address space.
        unsigned failed_samples=0;
        assert(session.publish(source.Get(),5,0,0,w,h,[&](long long,long long)->RetainedComposition::Texture{
            ++failed_samples;throw std::bad_alloc();}));
        request.ticket=5;copy_map();
        assert(session.display_to(5,canvas,target.Get(),display.Get(),buffer.Get(),w,h,full,660,1000));
        assert(failed_samples==1 && !session.visual_ready());
        assert(retained_read(device.Get(),context.Get(),display.Get())==retained_read(device.Get(),context.Get(),source.Get()));
        request.action=C3X_GPU_READBACK;request.image=std::int64_t(canvas);request.pixel_count=w*h;
        assert(session.execute(request,{}, {},result,output)==C3X_RENDERER_RESULT_OK);
        assert(output==retained_read(device.Get(),context.Get(),source.Get()));
        assert(session.publish(source.Get(),6,0,0,w,h,[&](long long,long long){
            return RetainedComposition::SampledImage::bgra(sampled.Get(),full);}));
        request.ticket=6;copy_map();
        assert(session.display_to(6,canvas,target.Get(),display.Get(),buffer.Get(),w,h,full,726,1000));
        assert(session.visual_ready() && retained_read(device.Get(),context.Get(),display.Get())==pixels);
        std::puts("PASS native visual allocation failure: completed transfer preserved, CPU ownership exact, fresh publication resumes animation");
        std::puts("PASS ambient ownership recovery: frozen CPU snapshot rejected, unit-only animation rejected, copied map restores readiness, static maps remain ready");
    }
    std::printf("PASS retained composition: %u exact GPU oracles, 120 independent clock frames, aliasing, paired 555/565/full color, UI versioning, partial publication, bounded overwrite and reset\n",checks);return 0;
}
