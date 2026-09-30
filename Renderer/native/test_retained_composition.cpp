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
    std::printf("PASS retained composition: %u exact GPU oracles, 120 independent clock frames, aliasing, paired 555/565/full color, UI versioning, partial publication, bounded overwrite and reset\n",checks);return 0;
}
