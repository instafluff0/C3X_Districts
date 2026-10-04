#define NOMINMAX
#include <windows.h>
#include "gpu_composition_session.h"
#include <cassert>
#include <cstdio>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_gpu_images;
std::vector<unsigned> ready_read(ID3D11Device* d,ID3D11DeviceContext* c,ID3D11Texture2D* t){
    D3D11_TEXTURE2D_DESC desc={};t->GetDesc(&desc);auto w=desc.Width,h=desc.Height;
    desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> read;checked(d->CreateTexture2D(&desc,nullptr,&read));c->CopyResource(read.Get(),t);
    D3D11_MAPPED_SUBRESOURCE m={};checked(c->Map(read.Get(),0,D3D11_MAP_READ,0,&m));std::vector<unsigned> out(w*h);
    for(unsigned y=0;y<h;++y)std::memcpy(out.data()+y*w,static_cast<char*>(m.pData)+y*m.RowPitch,w*4);
    c->Unmap(read.Get(),0);return out;
}
int test_ready_frame_composition(){
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
    constexpr unsigned w=48,h=32;Rect full={0,0,int(w),int(h)};
    for(bool map_source:{false,true}){
        Compositor native(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
        auto map=native.create(w,h,Format::bgra32),screen=native.create(w,h,Format::bgra32);
        std::vector<unsigned> pixels(w*h,0xff123456);assert(native.upload(map,1,pixels.data(),pixels.size()));
        retained.create(map,w,h,Format::bgra32);retained.create(screen,w,h,Format::bgra32);
        ComPtr<ID3D11Texture2D> texture;D3D11_TEXTURE2D_DESC desc={};
        desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data={pixels.data(),w*4,0};checked(device->CreateTexture2D(&desc,&data,&texture));
        unsigned prepared=0,sampled=0;bool ready=false,retired=false;
        RetainedComposition::Sample source=[&](long long,long long){
            assert(prepared>sampled);++sampled;
            if(retired)return RetainedComposition::SampledImage::frozen();
            return ready?RetainedComposition::SampledImage::bgra(texture.Get(),full):RetainedComposition::SampledImage::held();
        };
        source.projected=[&](long long tick,long long frequency,float){return source(tick,frequency);};
        source.prepare=[&](long long,long long,float){
            ++prepared;ready=prepared>=3;
            if(ready){std::fill(pixels.begin(),pixels.end(),0xffabcdef);context->UpdateSubresource(texture.Get(),0,nullptr,pixels.data(),w*4,0);}
        };
        retained.source(map,native.texture(map),source,true,map_source);
        auto zoom=std::make_shared<c3x_renderer::ZoomTransition>();zoom->target(1.25,0,1000);
        retained.view(screen,map,zoom);
        Rect panel={0,0,8,6};retained.record({Kind::fill,screen,0,panel,full,0,0,0xff778899});retained.commit(screen,full);
        for(int tick=1;tick<=3;++tick){
            RetainedComposition::Texture output;try{output=retained.sample(tick,1000);}catch(std::exception const& e){std::printf("prep exception: %s\n",e.what());return 1;}
            auto image=ready_read(device.Get(),context.Get(),output.Get());
            assert(image[1]==0xff778899 && image[16*w+24]==(tick<3?0xff123456:0xffabcdef));
        }
        assert(prepared==3 && sampled==3);retired=true;
        auto image=ready_read(device.Get(),context.Get(),retained.sample(4,1000).Get());
        assert(image[16*w+24]==0xffabcdef);
        image=ready_read(device.Get(),context.Get(),retained.sample(5,1000).Get());
        assert(prepared==4&&sampled==4&&image[1]==0xff778899&&image[16*w+24]==0xffabcdef);
        // Native world composition can be rebuilt while a replacement map is
        // pending. The old source has already retired, but its projected pose
        // is newer than its original canonical publication.
        retained.view(screen,map,zoom);
        retained.record({Kind::fill,screen,0,panel,full,0,0,0xff556677});retained.commit(screen,full);
        image=ready_read(device.Get(),context.Get(),retained.sample(6,1000).Get());
        if(image[16*w+24]!=0xffabcdef){std::puts("FAIL rebuilt native view rewound retired projected pose");return 1;}
        assert(prepared==4&&sampled==4&&image[1]==0xff556677);
        for(auto kind:{Kind::unit_over,Kind::native_image}){
            auto words=native.create(w,h,Format::rgb555),detail=native.create(w,h,Format::bgra32);
            auto ink=native.create(w,h,Format::bgra32),keys=native.create(w,h,Format::rgb555);
            for(auto id:{words,keys})retained.create(id,w,h,Format::rgb555);
            for(auto id:{detail,ink})retained.create(id,w,h,Format::bgra32);
            std::vector<unsigned> overlay(w*h,0),packed(w*h,0x7c1f);
            for(unsigned y=8;y<12;++y)for(unsigned x=8;x<12;++x){overlay[y*w+x]=0xff00ff00;packed[y*w+x]=0x03e0;}
            assert(native.upload(ink,1,overlay.data(),overlay.size()));retained.source(ink,native.texture(ink));
            assert(native.upload(keys,1,packed.data(),packed.size()));retained.source(keys,native.texture(keys));
            retained.record({Kind::copy,detail,map,full,full});
            retained.record({Kind::fill,words,0,full,full});
            if(kind==Kind::unit_over)retained.record({kind,words,ink,full,full,0,0,0,words,detail,detail});
            else retained.record({kind,words,keys,full,full,0,0,0x7c1f,words,detail,ink,int(w),int(h)});
            retained.view(screen,detail,zoom);retained.commit(screen,full);
            image=ready_read(device.Get(),context.Get(),retained.sample(7,1000).Get());
            if(image[16*w+24]!=0xffabcdef){std::printf("FAIL native overlay kind=%u rewound retired projected pose\n",unsigned(kind));return 1;}
            assert(prepared==4&&sampled==4&&image[10*w+10]==0xff00ff00);
        }
        std::puts("PASS ready frame: pending preparation retains pixels, native UI advances, rebuilt views and both overlay paths retain the final retired pose");
    }
    return 0;
}
