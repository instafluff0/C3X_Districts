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
    setvbuf(stdout,nullptr,_IONBF,0);setvbuf(stderr,nullptr,_IONBF,0);
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
    constexpr unsigned w=48,h=32;Rect full={0,0,int(w),int(h)},part={5,3,38,26};unsigned checks=0;
    {
        // One captured world, including its map-attached marker, moves under
        // an opaque fixed panel. No new native command is needed for any of
        // the intermediate frames or the rapid reversal.
        Compositor native(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto world=native.create(w,h,Format::bgra32),screen=native.create(w,h,Format::bgra32);
        std::vector<unsigned> pixels(w*h);
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
            pixels[y*w+x]=0xff000000u|(x*4<<16)|(y*7<<8)|((x+y)*3);
        for(unsigned y=10;y<14;++y)for(unsigned x=29;x<33;++x)pixels[y*w+x]=0xffffe040;
        assert(native.upload(world,1,pixels.data(),pixels.size()));
        retained.create(world,w,h,Format::bgra32);retained.source(world,native.texture(world));
        retained.create(screen,w,h,Format::bgra32);
        auto transition=std::make_shared<c3x_renderer::ZoomTransition>();
        retained.view(screen,world,transition);
        Rect panel={0,0,9,7};unsigned panel_color=0xff17385a;
        retained.record({Kind::fill,screen,0,panel,full,0,0,panel_color});retained.commit(screen,full);
        bool rejected=false;try{retained.view(screen,screen,transition);}catch(std::invalid_argument const&){rejected=true;}
        assert(rejected); // a repeated boundary must not zoom an already zoomed picture
        auto expected_pixel=[&](unsigned x,unsigned y,double scale){
            double px=(double(x)+.5-double(w/2))/scale+double(w/2)-.5;
            double py=(double(y)+.5-double(h/2))/scale+double(h/2)-.5;
            int left=int(std::floor(px)),top=int(std::floor(py));double fx=px-left,fy=py-top;
            unsigned out=0;
            for(unsigned shift:{0,8,16,24}){
                auto channel=[&](int dx,int dy){return double((pixels[std::clamp(top+dy,0,int(h)-1)*w+
                    std::clamp(left+dx,0,int(w)-1)]>>shift)&255);};
                double a=channel(0,0)*(1-fx)+channel(1,0)*fx,b=channel(0,1)*(1-fx)+channel(1,1)*fx;
                out|=unsigned(std::floor(a*(1-fy)+b*fy+.5))<<shift;
            }return out;
        };
        transition->target(1.5,0,1000);
        std::uint64_t warm_bytes=0,warm_allocations=0;
        for(int tick=0;tick<=420;tick+=7){
            if(tick==70)transition->target(1.,tick,1000);
            if(tick==112)transition->target(1.25,tick,1000);
            auto output=retained_read(device.Get(),context.Get(),retained.sample(tick,1000).Get());
            double scale=float(transition->current());
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
                unsigned expected=x<9&&y<7?panel_color:expected_pixel(x,y,scale),actual=output[y*w+x];
                for(unsigned shift:{0,8,16,24})assert(std::abs(int((actual>>shift)&255)-int((expected>>shift)&255))<=1);
            }
            if(tick==14){warm_bytes=retained.bytes();warm_allocations=retained.replay_stats().allocations;}
            if(tick>14){assert(retained.bytes()==warm_bytes);assert(retained.replay_stats().allocations==warm_allocations);}
        }
        assert(retained_read(device.Get(),context.Get(),native.texture(world))==pixels);
        // Actual Civ III UI uses paired packed words and full-color pixels.
        // Transparent pixels in a fixed native form must reveal the zoomed
        // world; opaque pixels must remain fixed in both 555 and 565 modes.
        for(auto format:{Format::rgb555,Format::rgb565}){
            auto words=native.create(w,h,format),detail=native.create(w,h,Format::bgra32),sprite=native.create(8,6,format);
            retained.create(words,w,h,format);retained.create(detail,w,h,Format::bgra32);retained.create(sprite,8,6,format);
            unsigned key=format==Format::rgb555?0x7c1f:0xf81f,ink=format==Format::rgb555?0x03e0:0x07e0;
            std::vector<unsigned> form(48);for(unsigned i=0;i<form.size();++i)form[i]=i%3?key:ink;
            assert(native.upload(sprite,1,form.data(),form.size()));retained.source(sprite,native.texture(sprite));
            auto ui_zoom=std::make_shared<c3x_renderer::ZoomTransition>();ui_zoom->target(1.5,0,1000);
            retained.view(detail,world,ui_zoom,words);
            retained.record({Kind::native_image,words,sprite,{2,2,10,8},full,0,0,key,0,detail,0,8,6});
            retained.commit(detail,full);
            for(int tick:{0,35,70,140,280}){
                auto output=retained_read(device.Get(),context.Get(),retained.sample(tick,1000).Get());
                for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
                    bool fixed=x>=2&&x<10&&y>=2&&y<8&&form[(y-2)*8+x-2]==ink;
                    unsigned expected=fixed?0xff00ff00:expected_pixel(x,y,float(ui_zoom->current())),actual=output[y*w+x];
                    for(unsigned shift:{0,8,16,24})assert(std::abs(int((actual>>shift)&255)-int((expected>>shift)&255))<=1);
                }
            }
            retained.commit(detail,full);
            auto color=retained_read(device.Get(),context.Get(),retained.sample(300,1000).Get());
            retained.commit(words,full);
            auto packed=retained_read(device.Get(),context.Get(),retained.sample(300,1000).Get());
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
                unsigned threshold=0;for(unsigned bit=0;bit<3;++bit){auto a=(x>>bit)&1,b=(y>>bit)&1;threshold=(threshold<<2)|((a^b)<<1)|b;}
                unsigned expected=0;for(unsigned channel=0;channel<3;++channel){
                    unsigned levels=channel==1&&format==Format::rgb565?63:31;
                    unsigned scaled=((color[y*w+x]>>(channel*8))&255)*levels;
                    unsigned q=scaled/255+((scaled%255)*128>(threshold*2+1)*255);
                    expected|=q<<(channel==0?0:channel==1?5:format==Format::rgb565?11:10);
                }assert(packed[y*w+x]==expected);
            }
            retained.destroy(words);retained.destroy(detail);retained.destroy(sprite);
            native.destroy(words);native.destroy(detail);native.destroy(sprite);
        }
        std::puts("PASS zoom with native keyed UI: 555/565 pairs, fixed ink and transparent holes over ten intermediate views");
        // A new camera source remains animated through the view operation.
        // Retiring that camera must also retire its map-animation readiness.
        bool frozen=false;unsigned samples=0;
        retained.source(world,native.texture(world),[&](long long,long long){
            ++samples;return frozen?RetainedComposition::SampledImage::frozen():RetainedComposition::SampledImage(native.texture(world));
        },true,true);
        retained.view(screen,world,transition);retained.commit(screen,full);
        retained.sample(421,1000);assert(retained.animated_map()&&samples==1);
        frozen=true;retained.sample(422,1000);assert(!retained.animated_map()&&samples==2);
        retained.sample(423,1000);assert(samples==2);
        // Erasing a fixed panel and replacing the world version cannot leave
        // its old rectangle behind as the map continues to zoom.
        std::fill(pixels.begin(),pixels.end(),0xff345678);
        assert(native.upload(world,2,pixels.data(),pixels.size()));retained.source(world,native.texture(world));
        retained.view(screen,world,transition);retained.commit(screen,full);
        auto erased=retained_read(device.Get(),context.Get(),retained.sample(427,1000).Get());
        assert(erased==pixels);
        // A subsequent canonical upload does not mutate the selected version.
        std::fill(pixels.begin(),pixels.end(),0xffabcdef);
        assert(native.upload(world,3,pixels.data(),pixels.size()));retained.source(world,native.texture(world));
        assert(retained_read(device.Get(),context.Get(),retained.sample(434,1000).Get())==erased);
        retained.clear();assert(!retained.ready()&&retained.node_count()==0&&retained.bytes()==0);
        std::puts("PASS retained zoom: 61 GPU samples, fixed UI, reversal, bounded storage, erasure and native version isolation");
    }
    {
        // A native unit is first submitted in screen coordinates, then replayed
        // into a rectangle-local image when the animated map changes. Its body
        // must stay above the new map sample at its original screen location.
        Compositor gpu(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        Id map=10000;auto screen=gpu.create(w,h,Format::bgra32);
        assert(screen);std::vector<unsigned> pixels(w*h,0xff183040u);
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={pixels.data(),w*4,0};ComPtr<ID3D11Texture2D> source;
        checked(device->CreateTexture2D(&desc,&initial,&source));
        retained.create(map,w,h,Format::bgra32);retained.create(screen,w,h,Format::bgra32);
        retained.source(map,source.Get(),[&](long long,long long){
            return RetainedComposition::SampledImage::bgra(source.Get(),full);},true,true);
        retained.record({Kind::copy,screen,map,full,full});
        Rect envelope={18,8,36,26},body={24,13,29,19};
        RetainedComposition::Direct unit;unit.draw=[=](Compositor& target,Command const& command){
            Command fill={Kind::fill,command.destination,0,rebase_direct_rect(body,envelope,command.area),
                command.clip,0,0,0xffe4b05au};
            return target.submit(&fill,1);
        };
        retained.record({Kind::unit_over,screen,0,envelope,full},std::move(unit));retained.commit(screen,full);
        for(unsigned tick=0;tick<2;++tick){
            unsigned ground=tick?0xff52728cu:0xff183040u;
            if(tick){std::fill(pixels.begin(),pixels.end(),ground);context->UpdateSubresource(source.Get(),0,nullptr,pixels.data(),w*4,0);}
            RetainedComposition::Texture result;
            try{result=retained.sample(tick+1,1000);}catch(std::exception const& error){
                std::fprintf(stderr,"direct unit sample failed: %s\n",error.what());std::fflush(stderr);return 1;
            }
            assert(result);
            auto image=retained_read(device.Get(),context.Get(),result.Get());
            assert(image[15*w+26]==0xffe4b05au && image[12*w+24]==ground && image[20*w+34]==ground);
        }
        std::puts("PASS retained direct unit: body survives two animated map samples in screen position");
    }
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
        assert(retained.replay_stats().allocations==0&&retained.replay_stats().resident_bytes==0);
        // Replacing the source image must not rewrite the completed front.
        retained.record({Kind::fill,1,0,bounds,bounds,0,0,0xffabcdef});
        assert(retained_read(device.Get(),context.Get(),retained.sample(2,1000).Get())==pixels);
        retained.clear();assert(retained.bytes()==0&&retained.node_count()==0);
        std::printf("PASS fullscreen retained copy chain: transfers=17 pixels=%u retained_bytes=%u nodes=1 immutable_front=1 assembly_scratch=0\n",width*height,width*height*4);
    }
    // JGL opaque paired transfers select versions; keyed full-window forms
    // retain only painted regions. Compare both native words and color detail
    // against the ordinary GPU interpreter across later animated map samples.
    for(auto format:{Format::rgb555,Format::rgb565})for(bool scaled:{false,true}){
        constexpr unsigned width=256,height=144;Rect bounds={0,0,width,height};unsigned key=0x7c1f;
        Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto make=[&](Format f){auto id=live.create(width,height,f);assert(id);retained.create(id,width,height,f);return id;};
        auto map=make(Format::bgra32),words=make(format),detail=make(Format::bgra32),
            saved=make(format),saved_detail=make(Format::bgra32),hud=make(format),hud_detail=make(Format::bgra32);
        auto draw=[&](Command command){assert(live.submit(&command,1));retained.record(command);};
        std::vector<unsigned> pixels(width*height);RetainedComposition::Texture current;
        auto next_map=[&](unsigned tick){for(unsigned i=0;i<pixels.size();++i)pixels[i]=0xff000000|((i*3127+tick*771)&0xffffff);
            assert(live.upload(map,tick+1,pixels.data(),pixels.size()));
            auto next=live.create(width,height,Format::bgra32);assert(next&&live.upload(next,1,pixels.data(),pixels.size()));
            current=live.texture(next);live.destroy(next);};
        next_map(0);retained.source(map,current.Get(),[&](long long,long long){return current;});
        std::vector<Command> recipe={{Kind::quantize,words,map,bounds,bounds},{Kind::copy,detail,map,bounds,bounds}};
        for(auto c:recipe)draw(c);
        auto transfer=Command{Kind::native_image,saved,words,bounds,bounds,0,0,65536,0,saved_detail,detail,int(width),int(height)};
        for(int i=0;i<20;++i)draw(transfer);
        retained.commit(saved_detail,bounds);assert(retained.node_count()==2);
        assert(retained_read(device.Get(),context.Get(),retained.sample(1,1000).Get())==pixels);
        draw({Kind::fill,hud,0,bounds,bounds,0,0,key});
        draw({Kind::fill,hud_detail,0,bounds,bounds,0,0,0xff00ff00});
        // These fills remain known constants even after an earlier display
        // has materialized and retired their static recipes.
        retained.commit(hud,bounds);retained.sample(2,1000);
        draw({Kind::fill,hud,0,{77,38,103,49},bounds,0,0,0x1234});
        draw({Kind::fill,hud_detail,0,{77,38,103,49},bounds,0,0,0xff123abc});
        Command keyed={Kind::native_image,saved,hud,{9,7,249,137},{20,14,232,120},0,0,scaled?65536u:key,0,saved_detail,hud_detail,scaled?120:240,scaled?65:130};
        recipe.push_back(transfer);recipe.push_back(keyed);draw(keyed);
        for(unsigned tick=1;tick<=12;++tick){next_map(tick);assert(live.submit(recipe.data(),recipe.size()));
            for(auto id:{saved,saved_detail}){retained.commit(id,bounds);
                auto actual=retained_read(device.Get(),context.Get(),retained.sample(tick+2,1000).Get());
                auto expected=retained_read(device.Get(),context.Get(),live.texture(id));assert(actual==expected);}
        }
        // The large transparent field must not become another viewport-sized
        // dynamic replay allocation. Source fills remain static and shared.
        assert(retained.bytes()<std::uint64_t(width)*height*(scaled?36:24));
        std::printf("PASS native form transfer simplification: format=%u scaled=%u paired_copies=20 keyed_offset=9,7 animation_samples=12 exact_words_and_color=1 bytes=%llu\n",
            unsigned(format),unsigned(scaled),retained.bytes());
    }
    // Live HUD buttons read tiny regions of the animated full-screen map.
    // Their coordinate domain stays full-screen; their actual read does not
    // require clearing/reconstructing a full-screen source for each button.
    {
        constexpr unsigned width=2240,height=1260,count=32;Rect bounds={0,0,width,height};
        Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        std::vector<unsigned> pixels(width*height,0xff123456);
        auto original=live.create(width,height,Format::bgra32);assert(original&&live.upload(original,1,pixels.data(),pixels.size()));
        D3D11_TEXTURE2D_DESC desc={};desc.Width=width;desc.Height=height;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={pixels.data(),width*4,0};ComPtr<ID3D11Texture2D> source;
        checked(device->CreateTexture2D(&desc,&initial,&source));
        retained.create(1,width,height,Format::bgra32);retained.source(1,live.texture(original),[&](long long,long long){
            return RetainedComposition::SampledImage::bgra(source.Get(),bounds);},true,true);
        retained.create(2,width,height,Format::bgra32);retained.record({Kind::copy,2,1,bounds,bounds});
        for(unsigned i=0;i<count;++i){Id id=10+i;Rect button={0,0,32,32};int x=10+int(i)*60,y=700;
            retained.create(id,32,32,Format::bgra32);
            retained.record({Kind::copy,id,1,button,button,x,y});
            retained.record({Kind::invert,id,0,button,button,0,0,0xffffff});
            retained.record({Kind::copy,2,id,{x,y,x+32,y+32},bounds});}
        retained.commit(2,bounds);std::uint64_t warm_views=0;
        for(unsigned tick=1;tick<=5;++tick){for(unsigned i=0;i<pixels.size();++i)pixels[i]=0xff000000|((i*3127+tick*771)&0xffffff);
            context->UpdateSubresource(source.Get(),0,nullptr,pixels.data(),width*4,0);
            auto expected=pixels;
            for(unsigned i=0;i<count;++i)for(unsigned y=700;y<732;++y)for(unsigned x=10+i*60;x<42+i*60;++x)expected[y*width+x]^=0xffffff;
            assert(retained_read(device.Get(),context.Get(),retained.sample(tick,1000).Get())==expected);
            assert(retained.last_work().assembly_pixels<std::uint64_t(width)*height*2);
            if(tick==1)warm_views=retained.source_view_creations();else assert(retained.source_view_creations()==warm_views);
        }
        std::printf("PASS tiny HUD background reads: buttons=32 full_screen_sources=0 exact_frames=5 assembly_pixels=%llu source_views_reused=1\n",retained.last_work().assembly_pixels);
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
    // Transparent HUD holes can keep prior camera versions alive even after
    // a new full-map copy. Freeze retired sources and collapse their completed
    // recipes without changing any pixels or retaining every camera texture.
    {
        constexpr unsigned width=128,height=96;Rect bounds={0,0,width,height},area={29,41,93,73};
        Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto create=[&](unsigned x,unsigned y,Format f){auto id=live.create(x,y,f);assert(id);retained.create(id,x,y,f);return id;};
        auto screen=create(width,height,Format::rgb555),hud=create(width,height,Format::rgb555),sprite=create(64,32,Format::bgra32);
        std::vector<unsigned> pixels(width*height),program(64*32);
        for(unsigned i=0;i<program.size();++i)program[i]=i%3==0?0xff000000u:i%3==1?0x8000001fu:0x00000400u;
        assert(live.upload(sprite,1,program.data(),program.size()));retained.source(sprite,live.texture(sprite));
        auto draw=[&](Command command){assert(live.submit(&command,1));retained.record(command);};
        draw({Kind::fill,hud,0,bounds,bounds});
        unsigned current=0;std::size_t peak=0;
        for(unsigned view=1;view<=80;++view){
            current=view;auto map=create(width,height,Format::rgb555);
            std::fill(pixels.begin(),pixels.end(),0x1234u+view);assert(live.upload(map,1,pixels.data(),pixels.size()));
            auto texture=RetainedComposition::Texture(live.texture(map));
            retained.source(map,texture.Get(),[&,view,texture](long long,long long){
                return current==view?RetainedComposition::SampledImage(texture):RetainedComposition::SampledImage::frozen();},true,true);
            draw({Kind::copy,screen,map,bounds,bounds});
            draw({Kind::native_blend,hud,sprite,area,bounds,0,0,0,screen});
            draw({Kind::copy,screen,hud,area,bounds,area.left,area.top});
            retained.commit(screen,bounds);
            assert(retained_read(device.Get(),context.Get(),retained.sample(view,1000).Get())==
                   retained_read(device.Get(),context.Get(),live.texture(screen)));
            assert(retained.animated_map());peak=std::max(peak,std::size_t(retained.bytes()));
            assert(retained.bytes()<width*height*4*8);
            retained.destroy(map);live.destroy(map);
        }
        std::printf("PASS retired camera HUD feedback: views=80 exact=1 current_map_animated=1 peak_bytes=%zu\n",peak);
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
        long long last_tick=0;unsigned samples=0;bool freeze=false;
        assert(session.publish(source.Get(),2,0,0,w,h,[&](long long ticks,long long){
            if(freeze)return RetainedComposition::SampledImage::frozen();
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
        // Publishing UI versions must not sample, wait for a frame or expose
        // subsequent uncommitted writes. The cadence sees the latest complete
        // version, including ordered partial updates, exactly once.
        auto before_commit_samples=samples;
        for(unsigned n=0;n<1000;++n)assert(session.commit_display(2,canvas,w,h,full));
        assert(samples==before_commit_samples);
        request.action=C3X_GPU_SUBMIT;
        Rect patch{1,1,4,4};
        commands={{Kind::fill,canvas,0,patch,full,0,0,0xffabcdef}};
        assert(session.execute(request,commands,{},result,output)==C3X_RENDERER_RESULT_OK);
        assert(session.commit_display(2,canvas,w,h,patch));
        commands[0].color=0xff010203;
        assert(session.execute(request,commands,{},result,output)==C3X_RENDERER_RESULT_OK);
        assert(retained_read(device.Get(),context.Get(),display.Get())==pixels);
        assert(session.visual_frame(528,1000,target.Get(),display.Get(),buffer.Get())==1);
        auto committed=pixels;
        for(int y=patch.top;y<patch.bottom;++y)for(int x=patch.left;x<patch.right;++x)committed[y*w+x]=0xffabcdef;
        assert(retained_read(device.Get(),context.Get(),display.Get())==committed);
        assert(samples==before_commit_samples);
        commands={{Kind::copy,canvas,session.map_image(),full,full}};
        assert(session.execute(request,commands,{},result,output)==C3X_RENDERER_RESULT_OK);
        assert(session.commit_display(2,canvas,w,h,full));
        assert(session.visual_frame(528,1000,target.Get(),display.Get(),buffer.Get())==1);
        assert(retained_read(device.Get(),context.Get(),display.Get())==pixels);
        std::puts("PASS deferred UI publication: 1000 commits, zero map samples; partial and uncommitted versions remain ordered");
        // A static tactical texture is copied once. Reusing its renderer
        // scratch cannot mutate the committed version; transparent pixels
        // must still follow every new animated map sample beneath it.
        Compositor overlay_gpu(device.Get(),context.Get());
        auto overlay=overlay_gpu.create(8,8,Format::bgra32);
        std::vector<unsigned> ink(64,0);for(unsigned i=0;i<64;i+=2)ink[i]=0xffabcdef;
        assert(overlay_gpu.upload(overlay,1,ink.data(),ink.size()));
        request.action=C3X_GPU_CREATE;request.width=w;request.height=h;request.format=C3X_GPU_RGB555;
        assert(session.execute(request,{}, {},result,output)==C3X_RENDERER_RESULT_OK);auto native_route=result.image;
        request.action=C3X_GPU_SUBMIT;commands={{Kind::quantize,Id(native_route),session.map_image(),full,full}};
        assert(session.execute(request,commands,{},result,output)==C3X_RENDERER_RESULT_OK);
        c3x_renderer_gpu_unit_v1 route={};route.ticket=2;route.destination=native_route;route.background=native_route;
        route.detail=canvas;route.background_detail=canvas;route.clip[2]=w;route.clip[3]=h;
        assert(session.draw_overlay(route,overlay_gpu.texture(overlay),8,8,2,3)==C3X_RENDERER_RESULT_OK);
        assert(session.commit_display(2,canvas,w,h,full));
        std::fill(ink.begin(),ink.end(),0xff010203);assert(overlay_gpu.upload(overlay,2,ink.data(),ink.size()));
        for(unsigned tick=0;tick<10;++tick){
            std::fill(pixels.begin(),pixels.end(),0xff345600+tick);
            context->UpdateSubresource(sampled.Get(),0,nullptr,pixels.data(),w*4,0);
            assert(session.visual_frame(529+tick,1000,target.Get(),display.Get(),buffer.Get())==1);
            auto actual=retained_read(device.Get(),context.Get(),display.Get());
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
                assert(actual[y*w+x]==(x>=2&&x<10&&y>=3&&y<11&&x%2==0?0xffabcdef:pixels[y*w+x]));
        }
        copy_map();assert(session.commit_display(2,canvas,w,h,full));
        assert(session.visual_frame(539,1000,target.Get(),display.Get(),buffer.Get())==1);
        std::puts("PASS retained static overlay: scratch reuse isolated; ten animated underlays exact");
        // Camera preparation retires the old map's callback before adoption.
        // Subsequent native transfers must keep its last sampled pose, even
        // though the retained recipe has become entirely static.
        freeze=true;
        for(unsigned tick=9;tick<=12;++tick){
            assert(session.display_to(2,canvas,target.Get(),display.Get(),buffer.Get(),w,h,full,tick*66,1000));
            assert(!session.visual_active());
            assert(retained_read(device.Get(),context.Get(),display.Get())==pixels);
        }
        std::puts("PASS camera handoff: repeated native transfers preserve the last animated pose after freezing");
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
        // The asynchronous presenter only commits here. A discarded visual
        // recipe must not poison transport before the next map rebuilds it.
        assert(session.commit_display(5,canvas,w,h,full));
        assert(!session.commit_display(4,canvas,w,h,full));
        assert(!session.commit_display(5,0,w,h,full));
        assert(!session.visual_ready());
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
    {
        // Re-presenting the same completed native screen is common around UI
        // messages. It must not resubmit a full-screen GPU composition when no
        // source, operation or displayed rectangle changed.
        constexpr unsigned width=640,height=480;Rect bounds={0,0,width,height};
        Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto image=live.create(width,height,Format::bgra32);assert(image);
        std::vector<unsigned> pixels(width*height,0xff264c72u);
        assert(live.upload(image,1,pixels.data(),pixels.size()));
        retained.create(image,width,height,Format::bgra32);retained.source(image,live.texture(image));
        D3D11_TEXTURE2D_DESC desc={};desc.Width=width;desc.Height=height;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_RENDER_TARGET;
        ComPtr<ID3D11Texture2D> display,buffer;ComPtr<ID3D11RenderTargetView> target;
        checked(device->CreateTexture2D(&desc,nullptr,&display));checked(device->CreateTexture2D(&desc,nullptr,&buffer));
        checked(device->CreateRenderTargetView(display.Get(),nullptr,&target));
        retained.commit(image,bounds);assert(retained.draw(1,1000,target.Get(),display.Get(),buffer.Get())==1);
        assert(retained.replay_stats().allocations==0);
        LARGE_INTEGER frequency={},begin={},end={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&begin);
        unsigned unchanged=0;
        for(unsigned tick=2;tick<=81;++tick){retained.commit(image,bounds);
            unchanged+=retained.draw(tick,1000,target.Get(),display.Get(),buffer.Get())==2;}
        context->Flush();QueryPerformanceCounter(&end);
        assert(retained_read(device.Get(),context.Get(),display.Get())==pixels);
        assert(unchanged==80);
        retained.record({Kind::fill,image,0,bounds,bounds,0,0,0xff876543u});
        retained.commit(image,bounds);
        assert(retained.draw(82,1000,target.Get(),display.Get(),buffer.Get())==1);
        assert(retained_read(device.Get(),context.Get(),display.Get())==
               std::vector<unsigned>(width*height,0xff876543u));
        std::printf("MEASURE unchanged full-screen native transfers: noops=%u/80 submit_ms=%.3f\n",
            unchanged,1000.*double(end.QuadPart-begin.QuadPart)/frequency.QuadPart);
    }
    {
        // Native glyphs move with a map attachment but never grow, including
        // during reversal. Their old rectangles reveal the current world.
        Compositor native(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto world=native.create(w,h,Format::bgra32),screen=native.create(w,h,Format::bgra32);
        std::vector<unsigned> bg(w*h,0xff123456);assert(native.upload(world,1,bg.data(),bg.size()));
        retained.create(world,w,h,Format::bgra32);retained.source(world,native.texture(world));retained.create(screen,w,h,Format::bgra32);
        auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();zoom->target(1.5,0,1000);
        retained.view(screen,world,zoom);
        Rect glyph={32,20,39,23};retained.record({Kind::fill,screen,0,glyph,full,0,0,0xfff0e0d0},{},zoom,36,22);
        retained.commit(screen,full);
        for(int tick=0;tick<=400;tick+=11){
            if(tick==110)zoom->target(1.,tick,1000);
            auto pixels=retained_read(device.Get(),context.Get(),retained.sample(tick,1000).Get());
            int dx=int(std::lround(12*(zoom->current()-1))),dy=int(std::lround(6*(zoom->current()-1))),ink=0;
            for(int y=0;y<int(h);++y)for(int x=0;x<int(w);++x){
                bool marked=x>=glyph.left+dx&&x<glyph.right+dx&&y>=glyph.top+dy&&y<glyph.bottom+dy;
                assert(pixels[y*w+x]==(marked?0xfff0e0d0:0xff123456));ink+=marked;
            }
            assert(ink==21);
        }
        std::puts("PASS native HUD placement: 37 intermediate/reversed views, invariant 7x3 ink, exact moving anchor and erased old positions");
    }
    {
        // Exercise the production Session boundary, not just its view helper.
        // Native dirty copies stay canonical; full map/unit sources are selected
        // before fixed UI and remain coherent while no game commands arrive.
        std::vector<unsigned> pixels(w*h),output;
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)pixels[y*w+x]=0xff000000u|(x*4<<16)|(y*7<<8)|((x+y)*3);
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={pixels.data(),w*4,0};ComPtr<ID3D11Texture2D> source;
        checked(device->CreateTexture2D(&desc,&initial,&source));
        Session session(device.Get(),context.Get());assert(session.publish(source.Get(),1));
        c3x_renderer_gpu_images_v1 request={};request.struct_size=sizeof(request);request.ticket=1;
        c3x_renderer_gpu_result_v1 result={};
        auto create=[&](int format){request.action=C3X_GPU_CREATE;request.width=w;request.height=h;request.format=format;
            assert(session.execute(request,{}, {},result,output)==1);return Id(result.image);};
        auto map_words=create(C3X_GPU_RGB565),units=create(C3X_GPU_RGB565),unit_detail=create(C3X_GPU_BGRA32);
        auto screen=create(C3X_GPU_RGB565),detail=create(C3X_GPU_BGRA32);
        auto gui=create(C3X_GPU_RGB565),gui_detail=create(C3X_GPU_BGRA32);
        auto blank=create(C3X_GPU_RGB565),blank_detail=create(C3X_GPU_BGRA32);
        auto send=[&](std::vector<Command> const& commands){request={};request.struct_size=sizeof(request);request.ticket=1;request.action=C3X_GPU_SUBMIT;
            assert(session.execute(request,commands,{},result,output)==1);};
        Rect marker={29,10,33,14},panel={0,0,9,7};
        send({{Kind::fill,blank,0,full,full,0,0,0x7c1f},{Kind::fill,blank_detail,0,full,full,0,0,0xffff00ff},
              {Kind::fill,gui,0,full,full,0,0,0x7c1f},{Kind::fill,gui_detail,0,full,full,0,0,0xffff00ff},
              {Kind::fill,gui,0,panel,full,0,0,0x1738},{Kind::fill,gui_detail,0,panel,full,0,0,0xff17385a}});
        send({{Kind::quantize,map_words,session.map_image(),full,full},
              {Kind::fill,units,0,full,full,0,0,0x7c1f},{Kind::fill,unit_detail,0,full,full,0,0,0xffff00ff},
              {Kind::fill,units,0,marker,full,0,0,0x07e0},{Kind::fill,unit_detail,0,marker,full,0,0,0xff00ff00}});
        for(int y=marker.top;y<marker.bottom;++y)for(int x=marker.left;x<marker.right;++x)pixels[y*w+x]=0xff00ff00;
        Rect native_hud={22,14,27,17};
        send({{Kind::hud_begin,units,0,{}, {},24,18,42,0,unit_detail,0,0x7c1f},
              {Kind::fill,units,0,native_hud,full,0,0,0xffff},{Kind::fill,unit_detail,0,native_hud,full,0,0,0xffffffff},
              {Kind::hud_end}});
        unsigned boundaries=0;auto boundary=[&]{++boundaries;send({{Kind::copy,screen,map_words,full,part},
            {Kind::world_begin,screen,map_words,full,full,0,0,65536,0,detail,session.map_image(),int(w),int(h)},
            {Kind::native_image,screen,units,full,part,0,0,0x7c1f,0,detail,unit_detail,int(w),int(h)},
            {Kind::world_end,screen,units,full,full,0,0,0x7c1f,0,detail,boundaries%2?unit_detail:0,int(w),int(h)},
            {Kind::native_image,screen,gui,full,boundaries==1?full:part,0,0,0x7c1f,0,detail,gui_detail,int(w),int(h)}});assert(session.commit_display(1,detail,w,h,boundaries==1?full:part));};
        desc.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> display,buffer;
        checked(device->CreateTexture2D(&desc,nullptr,&display));checked(device->CreateTexture2D(&desc,nullptr,&buffer));
        ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(display.Get(),nullptr,&target));
        LARGE_INTEGER now={},frequency={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&now);
        send({{Kind::fill,detail,0,full,full,0,0,0xff123456},{Kind::zoom_target,0,0,{}, {},0,0,196608}});
        assert(session.commit_display(1,detail,w,h,full));
        assert(session.visual_frame(now.QuadPart+frequency.QuadPart,frequency.QuadPart,target.Get(),display.Get(),buffer.Get())==1);
        session.did_present();assert(session.presented_zoom()==65536); // target without an actual view must never change picking
        boundary();QueryPerformanceCounter(&now);
        send({{Kind::zoom_target,0,0,{}, {},0,0,196608}});
        for(unsigned frame=1;frame<=16;++frame){
            auto tick=now.QuadPart+frequency.QuadPart*frame/60;
            assert(session.visual_frame(tick,frequency.QuadPart,target.Get(),display.Get(),buffer.Get())==1);
            unsigned prior=session.presented_zoom();
            if(frame==1)assert(prior==65536); // Sampling alone cannot move picking.
            session.did_present();double scale=session.presented_zoom()/65536.;assert(scale>=1&&scale<=3);
            auto actual=retained_read(device.Get(),context.Get(),display.Get());
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
                if(x<9&&y<7){assert(actual[y*w+x]==0xff17385a);continue;}
                int hx=int(std::lround((24-int(w/2))*(scale-1))),hy=int(std::lround((18-int(h/2))*(scale-1)));
                if(int(x)>=native_hud.left+hx&&int(x)<native_hud.right+hx&&int(y)>=native_hud.top+hy&&int(y)<native_hud.bottom+hy){
                    if(actual[y*w+x]!=0xffffffff){
                        std::fprintf(stderr,"HUD mismatch frame=%u scale=%.8f pixel=%u,%u actual=%08x expected_rect=%d,%d,%d,%d\n",frame,scale,x,y,actual[y*w+x],native_hud.left+hx,native_hud.top+hy,native_hud.right+hx,native_hud.bottom+hy);
                        for(unsigned yy=0;yy<h;++yy)for(unsigned xx=0;xx<w;++xx)if(actual[yy*w+xx]==0xffffffff)std::fprintf(stderr,"ink %u,%u\n",xx,yy);
                        assert(false);
                    }continue;
                }
                double sx=std::clamp(double(w/2)+(double(x)+.5-w/2)/scale-.5,0.,double(w-1));
                double sy=std::clamp(double(h/2)+(double(y)+.5-h/2)/scale-.5,0.,double(h-1));
                unsigned ix=unsigned(sx),iy=unsigned(sy),jx=std::min(ix+1,w-1),jy=std::min(iy+1,h-1);
                for(unsigned shift:{0u,8u,16u}){auto q=[&](unsigned xx,unsigned yy){return double((pixels[yy*w+xx]>>shift)&255);};
                    double upper=q(ix,iy)+(q(jx,iy)-q(ix,iy))*(sx-ix),lower=q(ix,jy)+(q(jx,jy)-q(ix,jy))*(sx-ix);
                    int expected=int(std::round(upper+(lower-upper)*(sy-iy)));
                    if(std::abs(int((actual[y*w+x]>>shift)&255)-expected)>1){
                        std::fprintf(stderr,"ZOOM mismatch frame=%u scale=%.8f pixel=%u,%u channel=%u actual=%08x expected=%d\n",frame,scale,x,y,shift,actual[y*w+x],expected);assert(false);
                    }
                }
            }
            // Repeated native partial draws must never select the old zoomed
            // screen as their world input or multiply its scale again.
            if(frame%3==0)boundary();
        }
        // A city repaint can arrive with a narrow native dirty rectangle.
        // The new map view retires every placement from the previous view,
        // including labels outside that native rectangle.
        send({{Kind::zoom_target,0,0,{}, {},0,0,98304}});
        QueryPerformanceCounter(&now);
        assert(session.visual_frame(now.QuadPart+frequency.QuadPart,frequency.QuadPart,target.Get(),display.Get(),buffer.Get())==1);
        session.did_present();
        send({{Kind::copy,units,blank,full,full},
              {Kind::copy,unit_detail,blank_detail,full,full},
              {Kind::hud_begin,units,0,{}, {},30,20,43,0,unit_detail,0,0x7c1f},
              {Kind::fill,units,0,{28,20,33,23},full,0,0,0xffff},
              {Kind::fill,unit_detail,0,{28,20,33,23},full,0,0,0xffffffff},{Kind::hud_end},
              {Kind::world_begin,screen,map_words,full,full,0,0,65536,0,detail,session.map_image(),int(w),int(h)},
              {Kind::world_end,screen,units,full,full,0,0,0x7c1f,0,detail,unit_detail,int(w),int(h)},
              {Kind::native_image,screen,gui,full,{28,20,33,23},0,0,0x7c1f,0,detail,gui_detail,int(w),int(h)}});
        assert(session.commit_display(1,detail,w,h,{28,20,33,23}));
        assert(session.visual_frame(now.QuadPart+frequency.QuadPart,frequency.QuadPart,target.Get(),display.Get(),buffer.Get())==1);
        auto moved=retained_read(device.Get(),context.Get(),display.Get());
        unsigned ink=0;for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
            if(moved[y*w+x]==0xffffffff){++ink;assert(x>=31&&x<36&&y>=22&&y<25);}
            if(x<9&&y<7)assert(moved[y*w+x]==0xff17385a);
        }assert(ink==15);
        std::puts("PASS camera HUD replacement: partial native publication, one current label, erased old position, fixed panel preserved");
        request={};request.struct_size=sizeof(request);request.ticket=1;request.action=C3X_GPU_READBACK;
        request.image=unit_detail;request.pixel_count=w*h;
        assert(session.execute(request,{}, {},result,output)==1);
        assert(output[0]==0xffff00ff&&output[20*w+28]==0xffffffff);
        std::puts("PASS live zoom Session: 16 intermediate GPU frames, partial native copies, fixed panel, marker alignment, present-only picking and canonical source preservation");
    }
    // Independent native targets can select the same complete before-images.
    // Share an exact keyed recipe, while keeping genuinely different saved
    // versions, source mappings and alias relationships independent.
    for(auto format:{Format::rgb555,Format::rgb565}){
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());
        auto retained_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());
        auto& live=*live_owner;auto& retained=*retained_owner;
        std::vector<Id> ids;unsigned revision=0;long long clock=10000;
        auto make=[&](Format f,unsigned width=0,unsigned height=0){if(!width)width=w;if(!height)height=h;
            auto id=live.create(width,height,f);assert(id);
            retained.create(id,width,height,f);ids.push_back(id);return id;};
        struct Pair {Id words,detail;};
        auto pair=[&](){return Pair{make(format),make(Format::bgra32)};};
        auto map=make(Format::bgra32),base=make(format),base_detail=make(Format::bgra32),
            artwork=make(format),artwork_detail=make(Format::bgra32),old_artwork=make(format),old_detail=make(Format::bgra32);
        unsigned key=format==Format::rgb555?0x7c1f:0xf81f,ink=format==Format::rgb555?0x03e0:0x07e0;
        std::vector<unsigned> pixels(w*h),words(w*h),colors(w*h);
        for(unsigned i=0;i<words.size();++i){words[i]=i%3?key:ink;colors[i]=0xff12a456u+i;}
        assert(live.upload(artwork,1,words.data(),words.size()));assert(live.upload(old_artwork,1,words.data(),words.size()));
        assert(live.upload(artwork_detail,1,colors.data(),colors.size()));assert(live.upload(old_detail,1,colors.data(),colors.size()));
        retained.source(artwork,live.texture(artwork));retained.source(artwork_detail,live.texture(artwork_detail));
        RetainedComposition::Texture current;
        auto next_map=[&](){++revision;for(unsigned i=0;i<pixels.size();++i)pixels[i]=0xff000000u|((i*3127+revision*771)&0xffffff);
            assert(live.upload(map,revision,pixels.data(),pixels.size()));auto next=live.create(w,h,Format::bgra32);assert(next);
            assert(live.upload(next,1,pixels.data(),pixels.size()));current=live.texture(next);live.destroy(next);};
        next_map();retained.source(map,current.Get(),[&](long long,long long){return current;},true,true);
        Command quantize={Kind::quantize,base,map,full,full},copy_detail={Kind::copy,base_detail,map,full,full};
        retained.record(quantize);retained.record(copy_detail);
        auto restore=[&](Pair p,Id detail_base=0){retained.snapshot(p.words,base);retained.snapshot(p.detail,detail_base?detail_base:base_detail);};
        auto recipe=[&](Pair p,unsigned color){return Command{Kind::native_image,p.words,artwork,full,full,0,0,color,0,p.detail,artwork_detail,int(w),int(h)};};
        auto compare=[&](Pair p,Command c,Id detail_base=0,Id word_base=0){
            assert(live.submit(&quantize,1));assert(live.submit(&copy_detail,1));
            Command reset_words={Kind::copy,p.words,word_base?word_base:base,full,full};
            Command reset_detail={Kind::copy,p.detail,detail_base?detail_base:base_detail,full,full};
            assert(live.submit(&reset_words,1));assert(live.submit(&reset_detail,1));
            c.destination=p.words;if(c.detail)c.detail=p.detail;assert(live.submit(&c,1));
            for(auto id:{p.words,p.detail}){retained.commit(id,full);
                assert(retained_read(device.Get(),context.Get(),retained.sample(++clock,1000).Get())==
                       retained_read(device.Get(),context.Get(),live.texture(id)));++checks;}
        };
        auto a=pair(),b=pair(),again=pair(),saved=pair();restore(a);restore(b);restore(again);
        retained.record(recipe(a,key));retained.record(recipe(b,key^1u));auto nodes=retained.node_count();
        retained.record(recipe(again,key));assert(retained.node_count()==nodes);
        auto reuse=retained.recipe_reuse();assert(reuse.eligible==3&&reuse.reused==1&&reuse.probed<=reuse.eligible*8);
        retained.snapshot(saved.words,a.words);retained.snapshot(saved.detail,a.detail);
        compare(a,recipe(a,key));compare(b,recipe(b,key^1u));compare(again,recipe(again,key));
        auto warm_bytes=retained.bytes(),warm_allocations=retained.replay_stats().allocations;nodes=retained.node_count();
        for(unsigned tick=0;tick<24;++tick){next_map();restore(again);auto before=retained.recipe_reuse();retained.record(recipe(again,key));
            assert(retained.recipe_reuse().reused==before.reused+1);compare(again,recipe(again,key));
            assert(retained.bytes()==warm_bytes&&retained.node_count()==nodes&&retained.replay_stats().allocations==warm_allocations);}
        auto miss=[&](Pair p,Command c,Id detail_base=0,Id word_base=0){auto before=retained.recipe_reuse();
            retained.record(c);assert(retained.recipe_reuse().reused==before.reused);compare(p,c,detail_base,word_base);};
        // A raw clip change remains a miss even when its effective extent is full.
        auto raw_clip=pair();restore(raw_clip);auto c=recipe(raw_clip,key);c.clip={-1,-1,int(w)+1,int(h)+1};miss(raw_clip,c);
        auto clipped=pair();restore(clipped);c=recipe(clipped,key);c.clip={1,0,int(w),int(h)};miss(clipped,c);
        auto changed_key=pair();restore(changed_key);miss(changed_key,recipe(changed_key,key^2u));
        auto no_detail=pair();restore(no_detail);c=recipe(no_detail,key);c.detail=c.background_detail=0;miss(no_detail,c);
        auto other_detail=make(Format::bgra32);std::fill(colors.begin(),colors.end(),0xff765432u);
        assert(live.upload(other_detail,1,colors.data(),colors.size()));retained.source(other_detail,live.texture(other_detail));
        auto changed_underlay=pair();restore(changed_underlay,other_detail);miss(changed_underlay,recipe(changed_underlay,key),other_detail);
        auto wider=make(format,w+1,h);std::vector<unsigned> wide((w+1)*h,ink);for(unsigned i=0;i<wide.size();i+=3)wide[i]=key;
        assert(live.upload(wider,1,wide.data(),wide.size()));retained.source(wider,live.texture(wider));
        auto shifted=pair();restore(shifted);c=recipe(shifted,key);c.source=wider;c.source_x=1;c.background_detail=0;miss(shifted,c);
        // Equal copied pictures do not make self-aliasing the same operation.
        auto cloned=pair();retained.snapshot(cloned.words,base);retained.snapshot(cloned.detail,base_detail);
        assert(live.submit(&quantize,1));assert(live.submit(&copy_detail,1));
        Command clone_words={Kind::copy,cloned.words,base,full,full},clone_detail={Kind::copy,cloned.detail,base_detail,full,full};
        assert(live.submit(&clone_words,1));assert(live.submit(&clone_detail,1));
        auto separate=pair();restore(separate);c=recipe(separate,key);c.source=cloned.words;c.background_detail=cloned.detail;
        retained.record(c);compare(separate,c);
        auto aliased=pair();restore(aliased);c=recipe(aliased,key);c.source=aliased.words;c.background_detail=aliased.detail;miss(aliased,c);
        // Packed format is part of every operand proof, including before-images.
        auto alternate=format==Format::rgb555?Format::rgb565:Format::rgb555;
        auto alternate_base=make(alternate),alternate_artwork=make(alternate);Pair alternate_target{make(alternate),make(Format::bgra32)};
        Command alternate_quantize={Kind::quantize,alternate_base,map,full,full};assert(live.submit(&alternate_quantize,1));retained.record(alternate_quantize);
        std::fill(words.begin(),words.end(),0x1234u);assert(live.upload(alternate_artwork,1,words.data(),words.size()));
        retained.source(alternate_artwork,live.texture(alternate_artwork));retained.snapshot(alternate_target.words,alternate_base);retained.snapshot(alternate_target.detail,base_detail);
        c=recipe(alternate_target,key);c.source=alternate_artwork;c.background_detail=0;miss(alternate_target,c,0,alternate_base);
        // Republish the same artwork handle. The saved old result still owns
        // its original pixels, while its shared map underlay keeps animating.
        // Put its matching live proof back in the recent index before changing
        // the source; a miss must not merely be caused by index eviction.
        restore(again);retained.record(recipe(again,key));compare(again,recipe(again,key));
        for(unsigned i=0;i<words.size();++i)words[i]=i%2?key:0x1234u;
        assert(live.upload(artwork,2,words.data(),words.size()));retained.source(artwork,live.texture(artwork));
        auto republished=pair();restore(republished);miss(republished,recipe(republished,key));
        next_map();c=recipe(saved,key);c.source=old_artwork;c.background_detail=old_detail;compare(saved,c);
        retained.uncommit();for(auto id:ids)retained.destroy(id);
        assert(!retained.node_count()&&!retained.bytes()); // weak recipe index remains alive
        std::printf("PASS keyed recipe reuse: format=%u dynamic_pairs=2 repeats=24 exact_words_and_detail=1 saved_source_version=1 divergence=9 weak_expiry=1\n",unsigned(format));
    }
    // The static expand is shared while its complete proof is still present.
    // Once evaluated, its frozen output cannot serve as a recipe proof.
    for(auto format:{Format::rgb555,Format::rgb565}){
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());
        auto retained_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());
        auto& live=*live_owner;auto& retained=*retained_owner;
        auto source=live.create(w,h,format),oracle=live.create(w,h,Format::bgra32);assert(source&&oracle);
        std::vector<unsigned> pixels(w*h);for(unsigned i=0;i<pixels.size();++i)pixels[i]=(i*31)&32767;
        assert(live.upload(source,1,pixels.data(),pixels.size()));retained.create(source,w,h,format);retained.source(source,live.texture(source));
        constexpr Id first=100001,second=100002,third=100003;
        for(auto id:{first,second,third})retained.create(id,w,h,Format::bgra32);
        Command expand={Kind::expand,first,source,full,full,0,0,65536};retained.record(expand);expand.destination=second;retained.record(expand);
        assert(retained.recipe_reuse().eligible==2&&retained.recipe_reuse().reused==1&&retained.node_count()==2);
        expand.destination=oracle;assert(live.submit(&expand,1));auto expected=retained_read(device.Get(),context.Get(),live.texture(oracle));
        for(auto id:{first,second}){retained.commit(id,full);assert(retained_read(device.Get(),context.Get(),retained.sample(1,1000).Get())==expected);++checks;}
        auto before=retained.recipe_reuse();auto nodes=retained.node_count();expand.destination=third;retained.record(expand);
        assert(retained.recipe_reuse().reused==before.reused&&retained.node_count()==nodes+1);
        retained.commit(third,full);assert(retained_read(device.Get(),context.Get(),retained.sample(2,1000).Get())==expected);++checks;
        retained.uncommit();for(auto id:{source,first,second,third})retained.destroy(id);assert(!retained.node_count()&&!retained.bytes());
        std::printf("PASS expand recipe reuse: format=%u queued_shared=1 retired_proof_rejected=1 exact=1 weak_expiry=1\n",unsigned(format));
    }
    // Thirty-two legitimate saved native pairs would demand 256 MiB of
    // duplicate outputs alone. Keep their exact shared recipe under the same
    // production cap; the live oracle owns only one working destination pair.
    {
        constexpr unsigned width=1024,height=1024,count=32;Rect bounds={0,0,int(width),int(height)};
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());
        auto retained_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());
        auto& live=*live_owner;auto& retained=*retained_owner;
        auto make=[&](Format f){auto id=live.create(width,height,f);assert(id);retained.create(id,width,height,f);return id;};
        auto map=make(Format::bgra32),base=make(Format::rgb555),base_detail=make(Format::bgra32),
            artwork=make(Format::rgb555),artwork_detail=make(Format::bgra32),oracle=live.create(width,height,Format::rgb555),oracle_detail=live.create(width,height,Format::bgra32);
        assert(oracle&&oracle_detail);std::vector<unsigned> pixels(width*height),words(width*height),colors(width*height);
        for(unsigned i=0;i<pixels.size();++i){pixels[i]=0xff000000u|((i*3127)&0xffffff);words[i]=i%3?0x7c1fu:0x03e0u;colors[i]=0xff12a456u;}
        assert(live.upload(map,1,pixels.data(),pixels.size()));assert(live.upload(artwork,1,words.data(),words.size()));
        assert(live.upload(artwork_detail,1,colors.data(),colors.size()));
        retained.source(map,live.texture(map),[&](long long,long long){return RetainedComposition::Texture(live.texture(map));},true,true);
        retained.source(artwork,live.texture(artwork));retained.source(artwork_detail,live.texture(artwork_detail));
        Command quantize={Kind::quantize,base,map,bounds,bounds},detail_copy={Kind::copy,base_detail,map,bounds,bounds};
        assert(live.submit(&quantize,1));assert(live.submit(&detail_copy,1));retained.record(quantize);retained.record(detail_copy);
        Command oracle_words={Kind::copy,oracle,base,bounds,bounds},oracle_color={Kind::copy,oracle_detail,base_detail,bounds,bounds};
        assert(live.submit(&oracle_words,1));assert(live.submit(&oracle_color,1));
        Command keyed={Kind::native_image,oracle,artwork,bounds,bounds,0,0,0x7c1f,0,oracle_detail,artwork_detail,int(width),int(height)};
        assert(live.submit(&keyed,1));auto expected_words=retained_read(device.Get(),context.Get(),live.texture(oracle));
        auto expected_color=retained_read(device.Get(),context.Get(),live.texture(oracle_detail));
        for(unsigned i=0;i<count;++i){Id destination=200000+i*2,detail=destination+1;
            retained.create(destination,width,height,Format::rgb555);retained.create(detail,width,height,Format::bgra32);
            retained.snapshot(destination,base);retained.snapshot(detail,base_detail);auto command=keyed;command.destination=destination;command.detail=detail;retained.record(command);}
        auto reuse=retained.recipe_reuse();assert(reuse.eligible==count&&reuse.reused==count-1&&reuse.probed<=count*8);
        auto nodes=retained.node_count();std::uint64_t warm_bytes=0;
        for(unsigned i=0;i<count;++i)for(unsigned output=0;output<2;++output){Id id=200000+i*2+output;retained.commit(id,bounds);
            assert(retained_read(device.Get(),context.Get(),retained.sample(i+1,1000).Get())==(output?expected_color:expected_words));++checks;
            if(!warm_bytes)warm_bytes=retained.bytes();assert(retained.bytes()==warm_bytes&&retained.node_count()==nodes);}
        assert(warm_bytes==std::uint64_t(width)*height*4*6); // three sources, quantize and one output pair
        retained.uncommit();for(unsigned i=0;i<count*2;++i)retained.destroy(200000+i);
        for(auto id:{map,base,base_detail,artwork,artwork_detail})retained.destroy(id);
        assert(!retained.node_count()&&!retained.bytes());
        std::printf("PASS paired recipe budget: targets=%u size=%ux%u eligible=%llu reused=%llu retained_bytes=%llu exact_oracles=64 weak_expiry=1 cap=268435456\n",
            count,width,height,reuse.eligible,reuse.reused,warm_bytes);
    }
    // The production HUD stores thousands of placed commands over an animated
    // packed/detail pair. Compare its compiled path with the retained interpreter
    // at exactly the same clock, including response text and alias boundaries.
    for(auto format:{Format::rgb555,Format::rgb565}){
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());auto& live=*live_owner;
        auto fast_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& fast=*fast_owner;
        auto oracle_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& oracle=*oracle_owner;
        oracle.set_compiled_enabled(false);
        constexpr unsigned width=128,height=96;Rect bounds={0,0,width,height};
        auto create=[&](unsigned x,unsigned y,Format f){auto id=live.create(x,y,f);assert(id);fast.create(id,x,y,f);oracle.create(id,x,y,f);return id;};
        auto map=create(width,height,Format::bgra32),words=create(width,height,format),detail=create(width,height,Format::bgra32),
            ink=create(8,8,format),ink_detail=create(8,8,Format::bgra32),glyph=create(8,8,Format::bgra32),
            curves=create(17,32,Format::bgra32),blend=create(8,8,Format::bgra32);
        auto upload=[&](Id id,std::vector<unsigned> const& pixels){assert(live.upload(id,1,pixels.data(),pixels.size()));
            fast.source(id,live.texture(id));oracle.source(id,live.texture(id));};
        unsigned key=format==Format::rgb555?0x7c1f:0xf81f;std::vector<unsigned> pixels(64);
        for(unsigned i=0;i<64;++i)pixels[i]=i%3?key:(i*731u)&65535;upload(ink,pixels);
        for(unsigned i=0;i<64;++i)pixels[i]=0xff000000u|((i*5371)&0xffffff);upload(ink_detail,pixels);
        for(unsigned i=0;i<64;++i)pixels[i]=(i%32)|((i*3%32)<<10)|((i*7%32)<<20)|(i%2?0x40000000u:0);upload(glyph,pixels);
        pixels.resize(17*32);for(unsigned i=0;i<pixels.size();++i)pixels[i]=(i/17*3+i%17*13)%256;upload(curves,pixels);
        pixels.resize(64);for(unsigned i=0;i<64;++i)pixels[i]=((i*41%256)<<24)|((i*3719)&0xffffff);upload(blend,pixels);
        RetainedComposition::Texture current;std::vector<unsigned> ground(width*height);unsigned map_revision=0;
        auto advance=[&]{++map_revision;for(unsigned i=0;i<ground.size();++i)ground[i]=0xff000000u|((i*3127+map_revision*771)&0xffffff);
            auto next=live.create(width,height,Format::bgra32);assert(next&&live.upload(next,1,ground.data(),ground.size()));current=live.texture(next);live.destroy(next);};
        advance();for(auto retained:{&fast,&oracle})retained->source(map,current.Get(),[&](long long,long long){return current;},true,true);
        auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();
        std::vector<RetainedComposition::Placed> placed;
        for(unsigned i=0;i<320;++i){int x=12+int(i*7%101),y=12+int(i*11%69);Rect area={x,y,x+8,y+8};Rect clip={x+int(i%2),y,x+7,y+7};
            auto add=[&](Command c){placed.push_back({c,x+3,y+3});};
            add({Kind::native_text,words,glyph,area,clip,0,0,0,curves});
            add({Kind::native_text,detail,glyph,area,clip,0,0,0,curves});
            add({Kind::native_image,words,ink,area,clip,0,0,key,0,detail,ink_detail,8,8});
            add({Kind::native_blend,words,blend,area,clip,0,0,i%5==2?1u:i%5,words,detail,detail});
            add({Kind::invert,words,0,area,clip,0,0,0x421});
            add({Kind::native_blend,words,words,area,clip,0,0,2,words,detail,detail,0x1234,141});
        }
        // A cross-position pair read is a real boundary. Commands after it
        // must observe its completed words and full-color companion.
        placed.push_back({{Kind::native_image,words,words,{20,20,28,28},bounds,10,10,65536,0,detail,detail,8,8},23,23});
        placed.push_back({{Kind::fill,detail,0,{19,19,24,24},bounds,0,0,0xffa13758},21,21});
        for(auto retained:{&fast,&oracle}){retained->view(detail,map,zoom,words);retained->placed_batch(words,detail,placed,zoom);retained->commit(detail,bounds);}
        std::uint64_t bound=0,plan_builds=0;
        for(unsigned tick=0;tick<24;++tick){
            if(tick==6)zoom->target(1.5,tick*17,1000);if(tick==12)zoom->target(1.,tick*17,1000);if(tick==18)zoom->target(1.25,tick*17,1000);
            advance();for(auto id:{words,detail}){
                fast.commit(id,bounds);oracle.commit(id,bounds);
                auto actual=retained_read(device.Get(),context.Get(),fast.sample(tick*17+1,1000).Get());
                auto expected=retained_read(device.Get(),context.Get(),oracle.sample(tick*17+1,1000).Get());
                if(actual!=expected){for(unsigned p=0;p<actual.size();++p)if(actual[p]!=expected[p]){
                    std::fprintf(stderr,"compiled HUD mismatch format=%u tick=%u target=%llu pixel=%u actual=%08x expected=%08x\n",unsigned(format),tick,id,p,actual[p],expected[p]);break;}assert(false);}++checks;
            }
            auto plan=fast.plan_reuse();if(tick==2){bound=plan.source_binds;plan_builds=plan.builds;}
            if(tick>2)assert(plan.source_binds==bound); // map/zoom never rebind immutable glyphs and tables
        }
        auto before=fast.plan_reuse();fast.sample(410,1000);fast.sample(411,1000);auto after=fast.plan_reuse();
        assert(after.builds==before.builds&&after.reuses>=before.reuses+2&&after.source_binds==bound&&plan_builds>0);
        // Updating a working source cannot rewrite captured HUD operands or
        // an old saved reader. A newly captured batch sees the new generation.
        fast.snapshot(500,detail);oracle.snapshot(500,detail);
        std::fill(pixels.begin(),pixels.end(),0xff543210u);assert(live.upload(blend,2,pixels.data(),pixels.size()));
        for(auto retained:{&fast,&oracle})retained->source(blend,live.texture(blend));
        for(auto retained:{&fast,&oracle})retained->commit(500,bounds);
        assert(retained_read(device.Get(),context.Get(),fast.sample(412,1000).Get())==retained_read(device.Get(),context.Get(),oracle.sample(412,1000).Get()));++checks;
        auto old_program=fast.plan_reuse();
        for(auto retained:{&fast,&oracle}){retained->view(detail,map,zoom,words);retained->placed_batch(words,detail,placed,zoom);retained->commit(detail,bounds);}
        assert(fast.plan_reuse().batch_builds==old_program.batch_builds+1&&fast.plan_reuse().batch_reuses==old_program.batch_reuses);
        assert(retained_read(device.Get(),context.Get(),fast.sample(413,1000).Get())==retained_read(device.Get(),context.Get(),oracle.sample(413,1000).Get()));++checks;
        fast.clear();oracle.clear();assert(!fast.bytes()&&!fast.node_count()&&!oracle.bytes()&&!oracle.node_count());
        std::printf("PASS compiled retained HUD: format=%u commands=%zu same_clock_pairs=24 text_response=1 blends=5 alias_boundary=1 zoom_reversal=1 immutable_binds=%llu old_reader=1 reset=1\n",unsigned(format),placed.size(),bound);
    }
    // A normal busy native HUD has 750 independently captured sources. Source
    // identities must not consume the compositor's 512 live image handles.
    for(auto format:{Format::rgb555,Format::rgb565}){
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());auto& live=*live_owner;
        auto fast_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& fast=*fast_owner;
        auto oracle_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& oracle=*oracle_owner;
        oracle.set_compiled_enabled(false);
        constexpr unsigned width=128,height=96,sources=900;Rect bounds={0,0,width,height};
        std::vector<Id> captured_ids;
        auto create=[&](unsigned x,unsigned y,Format f){auto id=live.create(x,y,f);assert(id);fast.create(id,x,y,f);oracle.create(id,x,y,f);captured_ids.push_back(id);return id;};
        auto map=create(width,height,Format::bgra32),words=create(width,height,format),detail=create(width,height,Format::bgra32);
        RetainedComposition::Texture current;std::vector<unsigned> pixels(width*height);unsigned revision=0;
        auto advance=[&]{++revision;for(unsigned i=0;i<pixels.size();++i)pixels[i]=0xff000000u|((i*3127+revision*771)&0xffffff);
            auto next=live.create(width,height,Format::bgra32);assert(next&&live.upload(next,1,pixels.data(),pixels.size()));current=live.texture(next);live.destroy(next);};
        advance();for(auto retained:{&fast,&oracle})retained->source(map,current.Get(),[&](long long,long long){return current;},true,true);
        std::vector<RetainedComposition::Placed> placed;
        auto capture=[&](std::vector<unsigned> const& body){auto id=create(2,2,Format::bgra32);assert(live.upload(id,1,body.data(),body.size()));
            fast.source(id,live.texture(id));oracle.source(id,live.texture(id));assert(live.destroy(id));return id;};
        for(unsigned index=0;index<sources;++index){unsigned ink=(index*731+17)&(format==Format::rgb565?65535:32767);
            auto source=capture({ink|65536u,0,(ink^0x421u)|65536u,ink|65536u});
            int x=8+int(index*7%108),y=8+int(index*11%76);Rect area={x,y,x+2,y+2};
            placed.push_back({{Kind::native_sprite,words,source,area,bounds},x+1,y+1});
            placed.push_back({{Kind::native_sprite,detail,source,area,bounds,0,0,format==Format::rgb565?2u:1u},x+1,y+1});
        }
        // External ground is a real interpreter boundary. Its source and
        // both ground handles retire before replay; captured versions remain.
        auto fallback=capture({0xff443388u,0x80204060u,0xff123456u,0});
        auto ground_words=create(width,height,format),ground_detail=create(width,height,Format::bgra32);
        for(auto id:{ground_words,ground_detail}){std::fill(pixels.begin(),pixels.end(),id==ground_words?0x1234u:0xff2468acu);
            assert(live.upload(id,1,pixels.data(),pixels.size()));fast.source(id,live.texture(id));oracle.source(id,live.texture(id));assert(live.destroy(id));}
        placed.push_back({{Kind::unit_over,words,fallback,{31,27,33,29},bounds,0,0,0,ground_words,detail,ground_detail},32,28});
        placed.push_back({{Kind::native_image,words,words,{35,29,39,33},bounds,30,25,65536,0,detail,detail,4,4},37,31});
        placed.push_back({{Kind::fill,detail,0,{35,29,37,31},bounds,0,0,0xffa13758},36,30});
        auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();
        for(auto retained:{&fast,&oracle}){retained->view(detail,map,zoom,words);retained->placed_batch(words,detail,placed,zoom);retained->commit(detail,bounds);}
        std::uint64_t bindings=0,source_copies=0,compilations=0;
        for(unsigned tick=0;tick<8;++tick){advance();
            // Every native world-end constructs a fresh view/batch Node. Its
            // changed before-image must reuse the exact immutable HUD program.
            if(tick)for(auto retained:{&fast,&oracle}){retained->view(detail,map,zoom,words);retained->placed_batch(words,detail,placed,zoom);}
            for(auto id:{words,detail}){
            fast.commit(id,bounds);oracle.commit(id,bounds);
            assert(retained_read(device.Get(),context.Get(),fast.sample(tick+1,1000).Get())==
                   retained_read(device.Get(),context.Get(),oracle.sample(tick+1,1000).Get()));++checks;
            auto counters=fast.replay_stats();assert(counters.spatial_commands>=sources*2&&counters.spatial_dispatches&&counters.interpreter_dispatches);
        }
            auto plan=fast.plan_reuse();auto counters=fast.replay_stats();
            if(tick==1){bindings=plan.source_binds;source_copies=counters.spatial_source_copies;compilations=counters.spatial_compilations;}
            if(tick>1)assert(plan.source_binds==bindings&&counters.spatial_source_copies==source_copies&&counters.spatial_compilations==compilations);
        }
        assert(bindings==sources+3&&source_copies==sources&&fast.plan_reuse().batch_builds==1&&fast.plan_reuse().batch_reuses==7);
        fast.snapshot(50000,detail);oracle.snapshot(50000,detail);
        for(auto retained:{&fast,&oracle})retained->commit(50000,bounds);
        assert(retained_read(device.Get(),context.Get(),fast.sample(9,1000).Get())==retained_read(device.Get(),context.Get(),oracle.sample(9,1000).Get()));++checks;
        // A different native target pair can share the program: every paired
        // alias is rebound while the saved reader keeps its own before-image.
        auto other_words=create(width,height,format),other_detail=create(width,height,Format::bgra32);auto other=placed;
        for(auto& draw:other){auto& c=draw.command;Id* operands[]={&c.destination,&c.source,&c.background,&c.detail,&c.background_detail,&c.program};
            for(auto id:operands)if(*id==words)*id=other_words;else if(*id==detail)*id=other_detail;}
        auto warm=fast.replay_stats();auto reused=fast.plan_reuse();
        for(auto retained:{&fast,&oracle}){retained->view(other_detail,map,zoom,other_words);retained->placed_batch(other_words,other_detail,other,zoom);retained->commit(other_detail,bounds);}
        assert(retained_read(device.Get(),context.Get(),fast.sample(10,1000).Get())==retained_read(device.Get(),context.Get(),oracle.sample(10,1000).Get()));++checks;
        auto reused_now=fast.plan_reuse();auto warm_now=fast.replay_stats();
        assert(reused_now.batch_reuses==reused.batch_reuses+1&&reused_now.source_binds==bindings&&warm_now.spatial_compilations==warm.spatial_compilations&&warm_now.spatial_source_copies==source_copies);
        // An equal scale with a different live placement owner is a miss.
        auto other_zoom=std::make_shared<c3x_renderer::ZoomTransition>();
        for(auto retained:{&fast,&oracle}){retained->view(other_detail,map,other_zoom,other_words);retained->placed_batch(other_words,other_detail,other,other_zoom);retained->commit(other_detail,bounds);}
        assert(fast.plan_reuse().batch_builds==reused_now.batch_builds+1);
        assert(retained_read(device.Get(),context.Get(),fast.sample(11,1000).Get())==retained_read(device.Get(),context.Get(),oracle.sample(11,1000).Get()));++checks;
        for(auto retained:{&fast,&oracle}){retained->destroy(fallback);retained->commit(50000,bounds);}
        assert(retained_read(device.Get(),context.Get(),fast.sample(12,1000).Get())==retained_read(device.Get(),context.Get(),oracle.sample(12,1000).Get()));++checks;
        fast.uncommit();oracle.uncommit();for(auto id:captured_ids)for(auto retained:{&fast,&oracle})retained->destroy(id);
        fast.destroy(50000);oracle.destroy(50000);
        assert(!fast.bytes()&&!fast.node_count()&&!oracle.bytes()&&!oracle.node_count()); // latest weak artifact cannot pin sources
        fast.clear();oracle.clear();assert(!fast.bytes()&&!fast.node_count()&&!oracle.bytes()&&!oracle.node_count());
        std::printf("PASS many-source retained HUD: format=%u unique_sources=%u commands=%zu world_publications=8 bound=%llu source_copies=%llu warm_rebinds=0 pair_rebind=1 placement_owner_miss=1 external_fallback_after_retirement=1 old_reader=1 weak_expiry=1 reset=1\n",
            unsigned(format),sources+3,placed.size(),bindings,source_copies);
    }
    // Sharing a HUD artifact must not retain an obsolete immutable base. The
    // current artifact stays live while successive unsaved base nodes expire.
    for(auto format:{Format::rgb555,Format::rgb565}){
        constexpr unsigned width=32,height=24;Rect bounds={0,0,width,height};
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());auto& live=*live_owner;
        auto fast_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& fast=*fast_owner;
        auto oracle_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& oracle=*oracle_owner;oracle.set_compiled_enabled(false);
        constexpr Id map=60000,words=60001,detail=60002,saved_words=60003,saved_detail=60004;
        auto sprite=live.create(2,2,Format::bgra32);std::vector<unsigned> ink={0x11234u,0,0x10421u,0x107e0u};assert(live.upload(sprite,1,ink.data(),ink.size()));
        for(auto retained:{&fast,&oracle}){retained->create(map,width,height,Format::bgra32);retained->create(words,width,height,format);
            retained->create(detail,width,height,Format::bgra32);retained->create(sprite,2,2,Format::bgra32);retained->source(sprite,live.texture(sprite));}
        auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();
        std::vector<RetainedComposition::Placed> placed={{{Kind::native_sprite,words,sprite,{12,10,14,12},bounds},13,11},
            {{Kind::native_sprite,detail,sprite,{12,10,14,12},bounds,0,0,format==Format::rgb555?1u:2u},13,11}};
        std::vector<unsigned> pixels(width*height),saved[2];std::uint64_t warm_bytes=0;std::size_t warm_nodes=0;
        for(unsigned publication=0;publication<10;++publication){for(unsigned i=0;i<pixels.size();++i)pixels[i]=0xff000000u|((i*3127+publication*771)&0xffffff);
            auto id=live.create(width,height,Format::bgra32);assert(id&&live.upload(id,1,pixels.data(),pixels.size()));RetainedComposition::Texture texture=live.texture(id);live.destroy(id);
            for(auto retained:{&fast,&oracle}){retained->source(map,texture.Get(),{},true);retained->view(detail,map,zoom,words);retained->placed_batch(words,detail,placed,zoom);}
            for(unsigned output=0;output<2;++output){auto target=output?detail:words;fast.commit(target,bounds);oracle.commit(target,bounds);
                auto actual=retained_read(device.Get(),context.Get(),fast.sample(publication+1,1000).Get());
                assert(actual==retained_read(device.Get(),context.Get(),oracle.sample(publication+1,1000).Get()));++checks;if(publication==9)saved[output]=actual;}
            if(!publication){warm_bytes=fast.bytes();warm_nodes=fast.node_count();}
            else assert(fast.bytes()==warm_bytes&&fast.node_count()==warm_nodes);
            assert(fast.plan_reuse().source_binds==1&&fast.replay_stats().spatial_compilations==1&&fast.replay_stats().spatial_source_copies==1);
        }
        assert(fast.plan_reuse().batch_builds==1&&fast.plan_reuse().batch_reuses==9);
        for(auto retained:{&fast,&oracle}){retained->snapshot(saved_words,words);retained->snapshot(saved_detail,detail);}
        std::fill(pixels.begin(),pixels.end(),0xffabcdefu);auto replacement=live.create(width,height,Format::bgra32);assert(replacement&&live.upload(replacement,1,pixels.data(),pixels.size()));
        for(auto retained:{&fast,&oracle}){retained->source(map,live.texture(replacement),{},true);retained->view(detail,map,zoom,words);retained->placed_batch(words,detail,placed,zoom);}
        live.destroy(replacement);
        for(unsigned output=0;output<2;++output){auto target=output?detail:words;fast.commit(target,bounds);oracle.commit(target,bounds);
            auto current=retained_read(device.Get(),context.Get(),fast.sample(11,1000).Get());assert(current!=saved[output]);
            assert(current==retained_read(device.Get(),context.Get(),oracle.sample(11,1000).Get()));++checks;
            target=output?saved_detail:saved_words;fast.commit(target,bounds);oracle.commit(target,bounds);
            assert(retained_read(device.Get(),context.Get(),fast.sample(12,1000).Get())==saved[output]);
            assert(retained_read(device.Get(),context.Get(),oracle.sample(12,1000).Get())==saved[output]);++checks;
        }
        assert(fast.plan_reuse().source_binds==1&&fast.replay_stats().spatial_compilations==1);
        fast.uncommit();oracle.uncommit();for(auto id:{map,words,detail,saved_words,saved_detail,sprite})for(auto retained:{&fast,&oracle})retained->destroy(id);
        assert(!fast.bytes()&&!fast.node_count()&&!oracle.bytes()&&!oracle.node_count());
        std::printf("PASS shared HUD immutable bases: format=%u publications=11 steady_bytes=%llu steady_nodes=%zu base_expiry=1 independent_saved_pair=1 source_rebinds=0 metadata_rebuilds=0 weak_expiry=1\n",unsigned(format),warm_bytes,warm_nodes);
    }
    // The real native world-end path creates fresh retained nodes while the
    // lexical HUD is redrawn from the same immutable sources. New snapshot
    // wrappers must reuse exact preparation, with changing unit before-images.
    for(int native_format:{C3X_GPU_RGB555,C3X_GPU_RGB565}){
        constexpr unsigned width=32,height=32,sources=150;Rect bounds={0,0,width,height};
        std::vector<unsigned> pixels(width*height,0xff2468acu),output;
        D3D11_TEXTURE2D_DESC desc={};desc.Width=width;desc.Height=height;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initial={pixels.data(),width*4,0};ComPtr<ID3D11Texture2D> map_texture;
        checked(device->CreateTexture2D(&desc,&initial,&map_texture));
        auto owner=std::make_unique<Session>(device.Get(),context.Get());auto& session=*owner;assert(session.publish(map_texture.Get(),1));
        c3x_renderer_gpu_images_v1 request={};c3x_renderer_gpu_result_v1 result={};
        auto execute=[&](unsigned action,Id id,std::vector<Command> const& commands={},std::vector<unsigned> const& body={}){
            request.struct_size=sizeof(request);request.ticket=1;request.action=action;request.image=std::int64_t(id);
            assert(session.execute(request,commands,body,result,output)==1);
        };
        auto create=[&](unsigned x,unsigned y,int format){request={};request.width=x;request.height=y;request.format=format;
            execute(C3X_GPU_CREATE,0);return Id(result.image);};
        auto map_words=create(width,height,native_format),units=create(width,height,native_format),unit_detail=create(width,height,C3X_GPU_BGRA32),
            screen=create(width,height,native_format),detail=create(width,height,C3X_GPU_BGRA32);
        unsigned key=native_format==C3X_GPU_RGB555?0x7c1f:0xf81f;
        execute(C3X_GPU_SUBMIT,0,{{Kind::quantize,map_words,session.map_image(),bounds,bounds},
            {Kind::fill,units,0,bounds,bounds,0,0,key},{Kind::fill,unit_detail,0,bounds,bounds,0,0,0xffff00ff},
            {Kind::hud_begin,units,0,{}, {},16,16,42,0,unit_detail,0,int(key)}});
        std::vector<Id> hud_sources;std::vector<Command> redraw;
        for(unsigned index=0;index<sources;++index){auto source=create(2,2,C3X_GPU_BGRA32);hud_sources.push_back(source);
            unsigned ink=(index*731+17)&(native_format==C3X_GPU_RGB555?32767:65535);
            request.revision=1;execute(C3X_GPU_UPLOAD,source,{}, {ink|65536u,0,(ink^0x421u)|65536u,ink|65536u});
            int x=1+int(index*7%23),y=1+int(index*11%23);Rect area={x,y,x+2,y+2};
            std::vector<Command> draws={{Kind::native_sprite,units,source,area,bounds},
                {Kind::native_sprite,unit_detail,source,area,bounds,0,0,native_format==C3X_GPU_RGB555?1u:2u}};
            redraw.insert(redraw.end(),draws.begin(),draws.end());execute(C3X_GPU_SUBMIT,0,draws);
        }
        execute(C3X_GPU_SUBMIT,0,{{Kind::hud_end}});
        desc.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> display,buffer;
        checked(device->CreateTexture2D(&desc,nullptr,&display));checked(device->CreateTexture2D(&desc,nullptr,&buffer));
        ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(display.Get(),nullptr,&target));
        Counts warm={};RetainedComposition::PlanReuse plan={};
        for(unsigned publication=0;publication<8;++publication){
            if(publication){execute(C3X_GPU_SUBMIT,0,{{Kind::hud_begin,units,0,{}, {},16,16,42,0,unit_detail,0,int(key)}});
                execute(C3X_GPU_SUBMIT,0,redraw);execute(C3X_GPU_SUBMIT,0,{{Kind::hud_end}});}
            execute(C3X_GPU_SUBMIT,0,{{Kind::fill,units,0,{29,29,31,31},bounds,0,0,publication+0x1234u},
                {Kind::fill,unit_detail,0,{29,29,31,31},bounds,0,0,0xff345670u+publication},
                {Kind::copy,screen,map_words,bounds,bounds},
                {Kind::copy,detail,session.map_image(),bounds,bounds},
                {Kind::world_begin,screen,map_words,bounds,bounds,0,0,65536,0,detail,session.map_image(),int(width),int(height)},
                {Kind::native_image,screen,units,bounds,bounds,0,0,key,0,detail,unit_detail,int(width),int(height)},
                {Kind::world_end,screen,units,bounds,bounds,0,0,key,0,detail,unit_detail,int(width),int(height)}});
            assert(session.commit_display(1,detail,width,height,bounds));
            assert(session.visual_frame(publication+1,1000,target.Get(),display.Get(),buffer.Get())==1);
            request.pixel_count=width*height;execute(C3X_GPU_READBACK,detail);
            assert(retained_read(device.Get(),context.Get(),display.Get())==output);++checks;
            auto current=session.visual_gpu_counts();auto current_plan=session.visual_plan_reuse();
            if(!publication){warm=current;plan=current_plan;assert(plan.source_binds==sources&&warm.spatial_source_copies==sources&&plan.batch_builds==1);}
            else assert(current_plan.source_binds==plan.source_binds&&current_plan.batch_builds==plan.batch_builds&&
                current_plan.batch_reuses==publication&&current.spatial_source_copies==warm.spatial_source_copies&&current.spatial_compilations==warm.spatial_compilations);
        }
        for(auto id:hud_sources)execute(C3X_GPU_DESTROY,id);
        std::printf("PASS Session world-end HUD reuse: format=%d publications=8 lexical_redraws=8 sources=%u commands=300 exact=8 source_rebinds=0 atlas_recopies=0 metadata_rebuilds=0 distinct_snapshot_versions=1 distinct_before_images=1\n",native_format,sources);
    }
    // Exact live selection can share complete source planes. Its first
    // multipart write must acquire private storage, preserving the old source.
    for(auto format:{Format::rgb555,Format::rgb565}){
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());auto& live=*live_owner;
        auto fast_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& fast=*fast_owner;
        auto oracle_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& oracle=*oracle_owner;oracle.set_compiled_enabled(false);
        auto create=[&](Format f){auto id=live.create(w,h,f);assert(id);fast.create(id,w,h,f);oracle.create(id,w,h,f);return id;};
        auto base=create(format),base_detail=create(Format::bgra32),selected=create(format),selected_detail=create(Format::bgra32),
            old=create(format),old_detail=create(Format::bgra32);
        std::vector<unsigned> words(w*h),colors(w*h);for(unsigned i=0;i<words.size();++i){words[i]=(i*31+7)&(format==Format::rgb555?32767:65535);colors[i]=0xff000000u|((i*731+17)&0xffffff);}
        assert(live.upload(base,1,words.data(),words.size())&&live.upload(base_detail,1,colors.data(),colors.size()));
        for(auto retained:{&fast,&oracle}){retained->source(base,live.texture(base));retained->source(base_detail,live.texture(base_detail));
            retained->snapshot(old,base);retained->snapshot(old_detail,base_detail);retained->select_world(selected,selected_detail,base,base_detail);}
        auto compare=[&](Id id,std::vector<unsigned> const& expected,unsigned tick){fast.commit(id,full);oracle.commit(id,full);
            assert(retained_read(device.Get(),context.Get(),fast.sample(tick,1000).Get())==expected);
            assert(retained_read(device.Get(),context.Get(),oracle.sample(tick,1000).Get())==expected);++checks;};
        compare(selected,words,1);assert(fast.last_work().selected_borrows==2&&fast.last_work().selected_owned==0);
        assert(fast.last_work().avoided_copy_pixels==std::uint64_t(w)*h*2);
        assert(oracle.last_work().copied_pixels-fast.last_work().copied_pixels==std::uint64_t(w)*h*2);
        compare(selected_detail,colors,2);
        Rect edit={7,9,19,23};auto changed_words=words,changed_colors=colors;
        for(unsigned y=edit.top;y<unsigned(edit.bottom);++y)for(unsigned x=edit.left;x<unsigned(edit.right);++x){changed_words[y*w+x]=0x1234;changed_colors[y*w+x]=0xffabcdef;}
        for(auto retained:{&fast,&oracle}){retained->record({Kind::fill,base,0,edit,full,0,0,0x1234});
            retained->record({Kind::fill,base_detail,0,edit,full,0,0,0xffabcdef});retained->select_world(selected,selected_detail,base,base_detail);}
        compare(selected,changed_words,3);assert(fast.last_work().selected_owned==2&&!fast.last_work().selected_borrows);
        compare(selected_detail,changed_colors,4);compare(old,words,5);compare(old_detail,colors,6);
        // Return from owned multipart storage to exact source leases.
        for(auto retained:{&fast,&oracle}){retained->source(base,live.texture(base));retained->source(base_detail,live.texture(base_detail));
            retained->select_world(selected,selected_detail,base,base_detail);}
        compare(selected,words,7);assert(fast.last_work().selected_borrows==2);compare(selected_detail,colors,8);
        fast.clear();oracle.clear();assert(!fast.bytes()&&!fast.node_count()&&!oracle.bytes()&&!oracle.node_count());
        std::printf("PASS selected-world leases: format=%u borrowed_to_owned=1 owned_to_borrowed=1 old_source_preserved=1 exact_planes=8 saved_source=1 reset=1\n",unsigned(format));
    }
    // Independent keyed native images write their admitted generation pair
    // directly. Compare both output planes with the ordinary interpreter,
    // including clipped extents, changing source revisions and saved readers.
    for(auto format:{Format::rgb555,Format::rgb565})for(bool source_detail:{false,true}){
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());auto& live=*live_owner;
        auto fast_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& fast=*fast_owner;
        auto oracle_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& oracle=*oracle_owner;oracle.set_compiled_enabled(false);
        auto create=[&](Format f){auto id=live.create(w,h,f);assert(id);fast.create(id,w,h,f);oracle.create(id,w,h,f);return id;};
        auto base=create(format),base_color=create(Format::bgra32),ink=create(format),ink_color=create(Format::bgra32),
            screen=create(format),detail=create(Format::bgra32),saved=create(format),saved_color=create(Format::bgra32),
            checkpoint=create(format),checkpoint_color=create(Format::bgra32);
        auto old_ink=live.create(w,h,format),old_ink_color=live.create(w,h,Format::bgra32);assert(old_ink&&old_ink_color);
        unsigned key=format==Format::rgb555?0x7c1f:0xf81f;
        std::vector<unsigned> words(w*h),colors(w*h),art(w*h),art_colors(w*h);
        // Packed uploads admit native 16-bit words. Exercise the upper native
        // bit without supplying out-of-domain uint32 data to that API.
        for(unsigned i=0;i<art.size();++i){art[i]=(i%3?key:0x1234u)|((i%7==0)?0x8000u:0);art_colors[i]=0xff000000u|((i*771+0x34567)&0xffffff);}
        assert(live.upload(ink,1,art.data(),art.size())&&live.upload(ink_color,1,art_colors.data(),art_colors.size()));
        assert(live.upload(old_ink,1,art.data(),art.size())&&live.upload(old_ink_color,1,art_colors.data(),art_colors.size()));
        RetainedComposition::Texture current[2];
        auto map=[&](unsigned tick){for(unsigned i=0;i<words.size();++i){words[i]=(i*31+tick*47)&(format==Format::rgb555?32767:65535);colors[i]=0xff000000u|((i*3127+tick*771)&0xffffff);}
            assert(live.upload(base,tick,words.data(),words.size())&&live.upload(base_color,tick,colors.data(),colors.size()));
            for(unsigned i=0;i<2;++i){auto id=live.create(w,h,i?Format::bgra32:format);assert(id);
                auto const& pixels=i?colors:words;assert(live.upload(id,1,pixels.data(),pixels.size()));current[i]=live.texture(id);live.destroy(id);}};
        map(1);
        for(auto retained:{&fast,&oracle}){retained->source(base,current[0].Get(),[&](long long,long long){return current[0];},true,true);
            retained->source(base_color,current[1].Get(),[&](long long,long long){return current[1];},true,true);
            retained->source(ink,live.texture(ink));retained->source(ink_color,live.texture(ink_color));
            retained->select_world(screen,detail,base,base_color);}
        Command recipe={Kind::native_image,screen,ink,full,full,0,0,key,0,detail,source_detail?ink_color:0,int(w),int(h)};
        for(auto retained:{&fast,&oracle})retained->record(recipe);
        auto compare=[&](Id actual,Command command,unsigned tick,bool color){
            Command reset[2]={{Kind::copy,screen,base,full,full},{Kind::copy,detail,base_color,full,full}};
            assert(live.submit(reset,2)&&live.submit(&command,1));auto expected=retained_read(device.Get(),context.Get(),live.texture(color?detail:screen));
            fast.commit(actual,full);oracle.commit(actual,full);
            assert(retained_read(device.Get(),context.Get(),fast.sample(tick,1000).Get())==expected);
            assert(retained_read(device.Get(),context.Get(),oracle.sample(tick,1000).Get())==expected);++checks;
        };
        compare(screen,recipe,1,false);assert(fast.last_work().selected_borrows==2&&fast.last_work().direct_native_images==1);
        assert(fast.last_work().avoided_copy_pixels==std::uint64_t(w)*h*4);
        assert(oracle.last_work().copied_pixels-fast.last_work().copied_pixels==std::uint64_t(w)*h*4);
        compare(detail,recipe,2,true);
        for(auto retained:{&fast,&oracle}){retained->snapshot(saved,screen);retained->snapshot(saved_color,detail);}
        // Replacing artwork creates a new recipe generation; the saved native
        // reader retains the old artwork while its live world still animates.
        for(unsigned i=0;i<art.size();++i){art[i]=i%2?key:0x0421u;art_colors[i]=0xffa45612u;}
        assert(live.upload(ink,2,art.data(),art.size())&&live.upload(ink_color,2,art_colors.data(),art_colors.size()));
        map(3);
        auto clipped=recipe;clipped.clip=part;
        for(auto retained:{&fast,&oracle}){retained->source(ink,live.texture(ink));retained->source(ink_color,live.texture(ink_color));
            retained->select_world(screen,detail,base,base_color);retained->record(clipped);}
        compare(screen,clipped,3,false);assert(fast.last_work().direct_native_images==1); // current generation; saved reader is evaluated separately
        compare(detail,clipped,4,true);
        auto old_recipe=recipe;old_recipe.source=old_ink;old_recipe.background_detail=source_detail?old_ink_color:0;
        compare(saved,old_recipe,5,false);compare(saved_color,old_recipe,6,true);
        // Cross-position aliased reads retain the original assembled interpreter.
        Command aliased={Kind::native_image,screen,screen,{3,2,43,30},full,1,1,65536,0,detail,detail,40,28};
        for(auto retained:{&fast,&oracle}){retained->select_world(screen,detail,base,base_color);retained->record(aliased);}
        compare(screen,aliased,7,false);assert(!fast.last_work().direct_native_images);
        compare(detail,aliased,8,true);
        // A new full pair cannot be partially admitted when only one plane's
        // capacity remains. Synthetic direct payload is an existing budget
        // charge, kept outside the front so no extra GPU allocations are used.
        // Remove the earlier cold recipes: those are now valid eviction
        // candidates. This case must exhaust non-evictable storage to exercise
        // atomic refusal; the cold-form regression covers successful eviction.
        for(auto retained:{&fast,&oracle}){
            for(auto id:{screen,detail,saved,saved_color})retained->destroy(id);
            retained->create(screen,w,h,format);retained->create(detail,w,h,Format::bgra32);
            retained->select_world(screen,detail,base,base_color);
            retained->snapshot(checkpoint,screen);retained->snapshot(checkpoint_color,detail);}
        fast.commit(checkpoint,full);assert(retained_read(device.Get(),context.Get(),fast.sample(9,1000).Get())==words);
        auto stable_bytes=fast.bytes();constexpr Id pressure=900001;fast.create(pressure,1,1,Format::bgra32);
        RetainedComposition::Direct charge;charge.input_bytes=256u*1024u*1024u-stable_bytes-std::uint64_t(w)*h*4;
        fast.record({Kind::fill,pressure,0,{0,0,1,1},{0,0,1,1}},charge);
        auto charged_bytes=fast.bytes();auto refused=recipe;refused.color=0x4567;
        fast.record(refused);fast.commit(screen,full);bool rejected=false;
        try{fast.sample(10,1000);}catch(std::runtime_error const&){rejected=true;}
        assert(rejected&&fast.bytes()==charged_bytes);fast.destroy(pressure);assert(fast.bytes()==stable_bytes);
        fast.commit(checkpoint,full);assert(retained_read(device.Get(),context.Get(),fast.sample(11,1000).Get())==words);++checks;
        oracle.record(refused);compare(screen,refused,12,false);compare(detail,refused,13,true);
        assert(fast.last_work().direct_native_images==0); // already completed same generation
        fast.clear();oracle.clear();assert(!fast.bytes()&&!fast.node_count()&&!oracle.bytes()&&!oracle.node_count());
        std::printf("PASS owned native-image outputs: format=%u source_detail=%u paired_exact=12 keyed_holes=1 clipped=1 saved_reader=1 source_revision=1 alias_fallback=1 atomic_pair_refusal=1 reset=1\n",unsigned(format),unsigned(source_detail));
    }
    // Fragmented pre-HUD worlds can assemble directly into the already
    // admitted generation pair. Each saved reader still owns its before-image.
    for(auto format:{Format::rgb555,Format::rgb565}){
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());auto& live=*live_owner;
        auto fast_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& fast=*fast_owner;
        auto oracle_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& oracle=*oracle_owner;oracle.set_compiled_enabled(false);
        auto create=[&](Format f){auto id=live.create(w,h,f);assert(id);fast.create(id,w,h,f);oracle.create(id,w,h,f);return id;};
        auto base=create(format),base_color=create(Format::bgra32),words=create(format),detail=create(Format::bgra32),saved=create(format),saved_color=create(Format::bgra32);
        RetainedComposition::Texture current[2];
        auto update=[&](unsigned tick){for(unsigned output=0;output<2;++output){auto id=live.create(w,h,output?Format::bgra32:format);assert(id);std::vector<unsigned> pixels(w*h);
            for(unsigned i=0;i<pixels.size();++i)pixels[i]=output?0xff000000u|((i*173+tick*977)&0xffffffu):(i*41+tick*29)&(format==Format::rgb555?32767u:65535u);
            assert(live.upload(id,1,pixels.data(),pixels.size()));current[output]=live.texture(id);live.destroy(id);}};
        update(1);auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();Rect panel={0,0,8,6},ink={17,13,22,18};
        std::vector<RetainedComposition::Placed> commands={{{Kind::fill,words,0,ink,full,0,0,0x1234},19,15},{{Kind::fill,detail,0,ink,full,0,0,0xff456789},19,15}};
        for(auto retained:{&fast,&oracle}){
            retained->source(base,current[0].Get(),[&](long long,long long){return current[0];},true,true);
            retained->source(base_color,current[1].Get(),[&](long long,long long){return current[1];},true,true);
            retained->record({Kind::copy,words,base,full,full});retained->record({Kind::copy,detail,base_color,full,full});
            retained->record({Kind::fill,words,0,panel,full,0,0,0x0421});retained->record({Kind::fill,detail,0,panel,full,0,0,0xff708090});
            retained->snapshot(saved,words);retained->snapshot(saved_color,detail);retained->placed_batch(words,detail,commands,zoom);
        }
        for(unsigned tick=1;tick<=6;++tick){if(tick>1)update(tick);
            for(unsigned output=0;output<2;++output){auto image=output?detail:words;fast.commit(image,full);oracle.commit(image,full);
                auto actual=retained_read(device.Get(),context.Get(),fast.sample(tick,1000).Get());
                assert(actual==retained_read(device.Get(),context.Get(),oracle.sample(tick,1000).Get()));++checks;
                if(!output)assert(oracle.last_work().copied_pixels-fast.last_work().copied_pixels==std::uint64_t(w)*h*2);
            }
        }
        for(auto image:{saved,saved_color}){fast.commit(image,full);oracle.commit(image,full);
            assert(retained_read(device.Get(),context.Get(),fast.sample(7,1000).Get())==retained_read(device.Get(),context.Get(),oracle.sample(7,1000).Get()));++checks;}
        fast.clear();oracle.clear();assert(!fast.bytes()&&!fast.node_count()&&!oracle.bytes()&&!oracle.node_count());
        std::printf("PASS direct HUD before-image assembly: format=%u exact=14 avoided_full_pair_copies=6 immutable_saved_inputs=1 reset=1\n",unsigned(format));
    }
    // The assembled display is optional, never a native version. Animate the
    // map underneath a fixed panel, then change topology, restore a saved
    // image and exercise shifted self-copy and sparse/admission fallbacks.
    // Display accepts BGRA only. Native word surfaces are verified through
    // sample() and paired HUD assembly above, before their display conversion.
    for(auto format:{Format::bgra32}){
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());auto& live=*live_owner;
        auto fast_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& fast=*fast_owner;
        auto oracle_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());auto& oracle=*oracle_owner;
        oracle.set_compiled_enabled(false);
        auto create=[&]{auto id=live.create(w,h,format);assert(id);fast.create(id,w,h,format);oracle.create(id,w,h,format);return id;};
        auto map=create(),screen=create(),saved=create(),sparse=create();
        RetainedComposition::Texture current;
        auto change_map=[&](unsigned tick){auto id=live.create(w,h,format);assert(id);std::vector<unsigned> pixels(w*h);
            for(unsigned i=0;i<pixels.size();++i)pixels[i]=format==Format::bgra32?0xff000000u|((i*711+tick*571)&0xffffffu):(i*31+tick*47)&(format==Format::rgb555?32767u:65535u);
            assert(live.upload(id,1,pixels.data(),pixels.size()));current=live.texture(id);live.destroy(id);};
        change_map(1);Rect panel={0,0,8,6};unsigned color=format==Format::bgra32?0xff718294u:0x1234u;
        for(auto retained:{&fast,&oracle}){retained->source(map,current.Get(),[&](long long,long long){return current;},true,true);
            retained->record({Kind::copy,screen,map,full,full});retained->record({Kind::fill,screen,0,panel,full,0,0,color});retained->commit(screen,full);}
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_RENDER_TARGET;
        ComPtr<ID3D11Texture2D> display[3],buffer[3];ComPtr<ID3D11RenderTargetView> target[3];
        for(unsigned i=0;i<3;++i){checked(device->CreateTexture2D(&desc,nullptr,&display[i]));checked(device->CreateTexture2D(&desc,nullptr,&buffer[i]));
            checked(device->CreateRenderTargetView(display[i].Get(),nullptr,&target[i]));}
        // Alternate the fast path's targets as a flip swap chain does. Only
        // the retained assembled canvas has the immediately previous frame.
        auto compare=[&](unsigned tick){unsigned output=(tick&1)?0u:2u;
            assert(fast.draw(tick,1000,target[output].Get(),display[output].Get(),buffer[output].Get())==1);
            assert(oracle.draw(tick,1000,target[1].Get(),display[1].Get(),buffer[1].Get())==1);
            auto actual=retained_read(device.Get(),context.Get(),display[output].Get());
            assert(actual==retained_read(device.Get(),context.Get(),display[1].Get())&&actual==retained_read(device.Get(),context.Get(),buffer[output].Get()));++checks;};
        assert(fast.draw(1,1000,nullptr,display[0].Get(),buffer[0].Get())==0);
        for(unsigned tick=1;tick<=8;++tick){if(tick>1)change_map(tick);compare(tick);
            if(tick>1){assert(fast.last_work().assembly_pixels==std::uint64_t(w)*h-48);
                assert(fast.last_work().copied_pixels<oracle.last_work().copied_pixels);}}
        assert(fast.draw(9,1000,target[0].Get(),display[0].Get(),buffer[0].Get())==2);
        for(auto retained:{&fast,&oracle}){retained->snapshot(saved,screen);Rect edit={11,8,17,13};
            retained->record({Kind::fill,screen,0,edit,full,0,0,color^0x421});retained->commit(screen,edit);}
        compare(10);
        Command shifted={Kind::copy,screen,screen,{4,5,22,20},full,1,2};
        for(auto retained:{&fast,&oracle}){retained->record(shifted);retained->commit(screen,full);}compare(11);
        for(auto retained:{&fast,&oracle})retained->commit(saved,full);compare(12);
        // A mandatory input charge may evict the optional front before its
        // reserve rejects. Leave less than one canvas of available capacity.
        constexpr Id pressure=900002;fast.create(pressure,1,1,Format::bgra32);auto before=fast.bytes();
        RetainedComposition::Direct charge;charge.input_bytes=256u*1024u*1024u-before+1;
        fast.record({Kind::fill,pressure,0,{0,0,1,1},{0,0,1,1}},std::move(charge));
        assert(fast.bytes()==256u*1024u*1024u-std::uint64_t(w)*h*4+1);
        for(auto retained:{&fast,&oracle}){Rect edit={2,2,3,3};retained->record({Kind::fill,saved,0,edit,full,0,0,color^0x842});retained->commit(saved,full);}compare(13);
        fast.destroy(pressure);
        for(auto retained:{&fast,&oracle}){retained->record({Kind::fill,sparse,0,{3,4,9,10},full,0,0,color});retained->commit(sparse,full);}compare(14);
        fast.clear();oracle.clear();assert(!fast.bytes()&&!fast.node_count()&&!oracle.bytes()&&!oracle.node_count());
        std::printf("PASS retained damaged front: format=%u exact=14 rotating_targets=1 failed_display_retry=1 unchanged_fixed_pixels=1 partial_commit=1 shifted_self_copy=1 saved_version=1 sparse_fallback=1 optional_admission_eviction=1 reset=1\n",unsigned(format));
    }
    // Cached sparse HUD artwork is optional. A fresh native output pair has
    // priority under the same hard cap, including when another saved pair
    // shares the compiled artwork. Keep both generations pixel-exact.
    {
        constexpr unsigned width=64,height=64;Rect bounds={0,0,width,height};
        constexpr std::uint64_t plane=std::uint64_t(width)*height*4;
        auto live_owner=std::make_unique<Compositor>(device.Get(),context.Get());
        auto retained_owner=std::make_unique<RetainedComposition>(device.Get(),context.Get());
        auto& live=*live_owner;auto& retained=*retained_owner;
        auto make=[&](Format format){auto id=live.create(width,height,format);assert(id);retained.create(id,width,height,format);return id;};
        auto base=make(Format::rgb555),base_color=make(Format::bgra32),art=make(Format::rgb555),
            words=make(Format::rgb555),detail=make(Format::bgra32),saved=make(Format::rgb555),saved_color=make(Format::bgra32);
        std::vector<unsigned> pixels(width*height,0x1234);
        for(auto id:{base,base_color,art}){assert(live.upload(id,1,pixels.data(),pixels.size()));retained.source(id,live.texture(id));}
        retained.record({Kind::fill,art,0,{7,9,37,41},bounds,0,0,0x0421}); // requires a private assembled source
        auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();
        std::vector<RetainedComposition::Placed> batch={{{Kind::copy,words,art,bounds,bounds},32,32}};
        auto publish=[&]{retained.snapshot(words,base);retained.snapshot(detail,base_color);retained.placed_batch(words,detail,batch,zoom);retained.commit(words,bounds);};
        publish();auto expected=retained_read(device.Get(),context.Get(),retained.sample(1,1000).Get());
        for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)
            assert(expected[y*width+x]==(x>=7&&x<37&&y>=9&&y<41?0x0421u:0x1234u));
        retained.snapshot(saved,words);retained.snapshot(saved_color,detail);
        constexpr Id pressure=900003;retained.create(pressure,1,1,Format::bgra32);
        RetainedComposition::Direct charge;charge.input_bytes=256u*1024u*1024u-retained.bytes()-plane*2+plane/2;
        retained.record({Kind::fill,pressure,0,{0,0,1,1},{0,0,1,1}},charge);
        publish();RetainedComposition::Texture admitted;
        try{admitted=retained.sample(2,1000);}catch(std::exception const& e){
            std::fprintf(stderr,"FAIL mandatory HUD admission: %s\n",e.what());return 1;
        }
        assert(retained_read(device.Get(),context.Get(),admitted.Get())==expected);
        assert(retained.bytes()<=256u*1024u*1024u);
        retained.commit(detail,bounds);assert(retained_read(device.Get(),context.Get(),retained.sample(3,1000).Get())==pixels);
        retained.commit(saved,bounds);assert(retained_read(device.Get(),context.Get(),retained.sample(4,1000).Get())==expected);
        retained.commit(saved_color,bounds);assert(retained_read(device.Get(),context.Get(),retained.sample(5,1000).Get())==pixels);
        retained.destroy(pressure);publish();assert(retained_read(device.Get(),context.Get(),retained.sample(6,1000).Get())==expected);
        retained.clear();assert(!retained.bytes()&&!retained.node_count());checks+=6;
        std::puts("PASS mandatory HUD admission: optional atlas evicted; shared saved pair exact; cache rebind; cap unchanged");
    }
    std::printf("PASS retained composition: %u exact GPU oracles, 120 independent clock frames, aliasing, paired 555/565/full color, UI versioning, partial publication, bounded overwrite and reset\n",checks);return 0;
}
