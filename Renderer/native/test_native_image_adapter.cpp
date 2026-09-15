#define C3X_NATIVE_ADAPTER_TEST
#include "test_native_observation.cpp"
#include "native_image_adapter.h"
#ifdef C3X_NATIVE_WORKER_TEST
#include "gpu_image_worker_client.h"
#include "color_quantization.h"
using Backend=c3x_gpu_images::WorkerClient;
#else
#include "test_local_image_backend.h"
using Backend=c3x_gpu_images::LocalBackend;
#endif
using namespace c3x_gpu_images;
c3x_native_images::Adapter<Backend>* adapter=nullptr;
int translate(int op,void* object,void* source,void const* from,void const* to,unsigned color){return adapter->operation(op,object,source,from,to,color);}
int native_adapter_contract(char const* path,Backend& gpu,Id map=0,unsigned const* map_pixels=nullptr,int map_width=0,int phase_x=0,int phase_y=0){
    try{
        HMODULE jgl=LoadLibraryA(path);verify(c3x_native_observation::verified_module(jgl),"audited JGL");
        auto graph=reinterpret_cast<void*(__cdecl*)()>(GetProcAddress(jgl,"get_graphsy_object_ptr"))();auto gt=*reinterpret_cast<void***>(graph);
        auto create=reinterpret_cast<Create>(gt[31]);
        constexpr int w=64,h=48;RECT full={0,0,w,h};
        auto root=create(graph,nullptr,1);verify(reinterpret_cast<Init>(root->vtable[1])(root,w,h,16,1)==0,"root init");
        void* original[60];std::memcpy(original,root->vtable,sizeof original);
        auto init=reinterpret_cast<Init>(original[1]);auto fill=reinterpret_cast<Fill>(original[17]);auto copy=reinterpret_cast<Copy>(original[16]);
        auto get=reinterpret_cast<Get>(original[3]);auto release=reinterpret_cast<Release>(original[9]);
        JGL_Image* control[3];JGL_Image* target[3];
        for(auto& image:control){image=create(graph,nullptr,1);verify(init(image,w,h,16,1)==0,"control init");verify(fill(image,&full,int(0x80000000u))==0,"control clear");}
        // Preexisting CPU UI is never assumed GPU-owned. Keep a raw pointer across
        // every GPU use to witness writes that have no getter/release event.
        auto ui=create(graph,nullptr,1);verify(init(ui,w,h,16,1)==0,"UI init");verify(fill(ui,&full,int(0x80000123u))==0,"UI clear");
        auto retained=get(ui,0,0);verify(retained!=nullptr,"retained UI pointer");release(ui,1);
        c3x_native_images::Adapter<Backend> backend(gpu,original[4],original[9]);adapter=&backend;
        capture.write=log_line;state.custom_renderer_native_observe=observe;set_custom_renderer_native_probe(root);
        verify(state.custom_renderer_native_probe_active,"actual hook attach");state.custom_renderer_native_image=translate;
        for(auto& image:target){image=create(graph,nullptr,1);verify(reinterpret_cast<Init>(image->vtable[1])(image,w,h,16,1)==0,"admitted init");
            verify(backend.owns(image),"fresh destination admitted");verify(reinterpret_cast<Fill>(image->vtable[17])(image,&full,int(0x80000000u))==0,"GPU clear native return");}
        auto compare=[&](int index,bool cpu=false){GdiFlush();auto expected=get(control[index],0,0);verify(expected!=nullptr,"oracle lease");
            std::vector<std::uint32_t> observed;
            if(cpu){auto bits=get(target[index],0,0);verify(bits!=nullptr,"restored native lease");auto stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(target[index])+0x40);
                observed.resize(w*h);for(int y=0;y<h;++y)for(int x=0;x<w;++x)observed[y*w+x]=bits[y*stride+x];release(target[index],1);
            }else {observed.resize(w*h);verify(gpu.readback(backend.image(target[index]),observed.data(),observed.size()),"GPU oracle read");}
            auto stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(control[index])+0x40);
            bool equal=true;for(int y=0;y<h;++y)for(int x=0;x<w;++x)if(observed[y*w+x]!=expected[y*stride+x]){std::fprintf(stderr,"mismatch image=%d x=%d y=%d native=%x GPU=%x\n",index,x,y,expected[y*stride+x],observed[y*w+x]);equal=false;goto done;}
            done:release(control[index],1);verify(equal,"exact native adapter parity");};
#ifdef C3X_NATIVE_WORKER_TEST
        verify(map&&map_pixels&&map_width>=w,"resident map fixture");
        verify(backend.insert_map(target[0],map,{0,0,w,h},0,0,phase_x,phase_y),"resident map insertion into native image");
        auto seed=get(control[0],0,0);auto seed_stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(control[0])+0x40);
        auto rounding=c3x_renderer::color_rounding_table(31);
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){auto c=map_pixels[y*map_width+x];
            auto t=c3x_renderer::color_threshold(unsigned(x-phase_x),unsigned(y-phase_y));
            seed[y*seed_stride+x]=std::uint16_t((rounding[t][c&255]>>3)|((rounding[t][(c>>8)&255]>>3)<<5)|((rounding[t][(c>>16)&255]>>3)<<10));}
        release(control[0],1);compare(0);
#endif
        auto both_fill=[&](int index,RECT area,unsigned color){verify(fill(control[index],&area,int(color))==0,"oracle fill");verify(reinterpret_cast<Fill>(target[index]->vtable[17])(target[index],&area,int(color))==0,"hooked fill");};
        auto both_copy=[&](int from,int to,RECT a,RECT b){verify(copy(control[from],control[to],&a,&b)==0,"oracle copy");verify(reinterpret_cast<Copy>(target[from]->vtable[16])(target[from],target[to],&a,&b)==0,"hooked copy");};
        for(int phase=0;phase<6;++phase){
            RECT clip={7,8,59,44},area={-4,3,44,39};
            verify(reinterpret_cast<Clip>(control[0]->vtable[13])(control[0],&clip)==0,"oracle clip");verify(reinterpret_cast<Clip>(target[0]->vtable[13])(target[0],&clip)==0,"GPU clip metadata");
            both_fill(0,area,0x80001800u+phase);compare(0);
            verify(reinterpret_cast<Clip>(control[0]->vtable[13])(control[0],&full)==0,"oracle unclip");verify(reinterpret_cast<Clip>(target[0]->vtable[13])(target[0],&full)==0,"GPU unclip");
            RECT popup={10,12,40,30};both_copy(0,1,popup,popup);both_fill(0,popup,0x80007fff);both_copy(1,0,popup,popup);compare(0);compare(1);
            RECT from={3,5,45,35},to={9,11,51,41};both_copy(0,0,from,to);compare(0);
        }
        verify(backend.stats().readbacks==0,"translated chain performs zero execution readbacks");
        auto upload_before=gpu.stats().uploads;
        for(int n=0;n<3;++n){verify(copy(ui,control[2],&full,&full)==0,"oracle CPU UI copy");verify(reinterpret_cast<Copy>(ui->vtable[16])(ui,target[2],&full,&full)==0,"GPU CPU UI copy");compare(2);}
        verify(gpu.stats().uploads==upload_before+1,"unchanged CPU source uploads once");
        retained[5]=0x4567;
        verify(copy(ui,control[2],&full,&full)==0,"oracle retained pointer edit");verify(reinterpret_cast<Copy>(ui->vtable[16])(ui,target[2],&full,&full)==0,"GPU retained pointer edit");compare(2);
        verify(gpu.stats().uploads==upload_before+2,"complete source validity detects retained-pointer writes");
        // Actual image slot 33 now also skips JGL rasterization for the audited
        // transparent mode; native palette/shadow variants still fall back.
        verify(fill(ui,&full,int(0x80007c1fu))==0,"native key background");
        RECT foreground={5,6,28,29};verify(fill(ui,&foreground,int(0x800003e0u))==0,"native key foreground");
        using Draw=int(__thiscall*)(JGL_Image*,JGL_Image*,int,int);
        for(int phase=0;phase<3;++phase){
            verify(reinterpret_cast<Draw>(original[33])(ui,control[0],-3,2)==0,"oracle image draw");
            verify(reinterpret_cast<Draw>(ui->vtable[33])(ui,target[0],-3,2)==0,"hooked GPU image draw");compare(0);
        }
        verify(backend.stats().readbacks==0&&gpu.stats().uploads==upload_before+3,"keyed image chain reuses CPU upload without readback");
        // Unsupported stretch restores both GPU images before native StretchBlt.
        RECT reduced={0,0,32,24};both_copy(0,1,full,reduced);verify(!backend.owns(target[0])&&!backend.owns(target[1]),"stretch returns involved images to CPU ownership");compare(0,true);compare(1,true);
        verify(backend.stats().readbacks==2,"one readback for each dirty native fallback image");
        // Public bits access must restore current output before returning. A later
        // native write remains native even after releasing the original pointer.
        auto public_bits=reinterpret_cast<Get>(target[2]->vtable[7])(target[2],0,0);verify(public_bits!=nullptr,"public lease");release(target[2],1);compare(2,true);
        auto reads=backend.stats().readbacks;both_fill(2,{0,0,2,2},0x80000456);compare(2,true);
        verify(!backend.owns(target[2])&&backend.stats().readbacks==reads,"CPU exposure permanently demotes lifetime");
        auto old=backend.image(target[2]);verify(reinterpret_cast<Init>(target[2]->vtable[1])(target[2],w,h,16,1)==0,"reinitialize lifetime");
        verify(backend.owns(target[2])&&backend.image(target[2])!=old,"reinitialize retires GPU identity");
        both_fill(2,full,0x80000567);compare(2);
        using Dc=HDC(__thiscall*)(JGL_Image*);
        auto native_dc=reinterpret_cast<Dc>(control[2]->vtable[10])(control[2]);
        auto gpu_dc=reinterpret_cast<Dc>(target[2]->vtable[10])(target[2]);
        verify(native_dc&&gpu_dc,"public HDC leases");
        verify(PatBlt(native_dc,3,4,12,8,DSTINVERT)&&PatBlt(gpu_dc,3,4,12,8,DSTINVERT),"CPU destination-dependent GDI drawing");GdiFlush();
        reinterpret_cast<Release>(control[2]->vtable[11])(control[2],1);reinterpret_cast<Release>(target[2]->vtable[11])(target[2],1);
        compare(2,true);verify(!backend.owns(target[2]),"public HDC receives current GPU pixels before native invert");
        verify(reinterpret_cast<Init>(target[2]->vtable[1])(target[2],w,h,16,1)==0,"new lifetime before config off");both_fill(2,full,0x80000567);
        // Config-off restores all remaining dirty images before returning to JGL.
        state.current_config.enable_custom_rendering=false;both_fill(2,{0,0,2,2},0x80000111);compare(2,true);
        verify(state.custom_renderer_native_image==nullptr,"config-off drains and unbinds backend");state.current_config.enable_custom_rendering=true;
        state.custom_renderer_native_image=translate;
        auto temporary=create(graph,nullptr,1);verify(reinterpret_cast<Init>(temporary->vtable[1])(temporary,w,h,16,1)==0,"temporary GPU lifetime");
        auto temporary_id=backend.image(temporary);verify(temporary_id!=0,"temporary GPU admitted");
        verify(reinterpret_cast<Fill>(temporary->vtable[17])(temporary,&full,int(0x80000000u))==0,"temporary GPU clear");
        verify(fill(root,&full,int(0x80000000u))==0,"null-fill control clear");
        RECT tiny_clip={1,1,2,2};verify(reinterpret_cast<Clip>(temporary->vtable[13])(temporary,&tiny_clip)==0,"temporary clip");
        verify(reinterpret_cast<Clip>(root->vtable[13])(root,&tiny_clip)==0,"null-fill control clip");
        auto expected_return=fill(root,nullptr,int(0x80000456u));
        verify(reinterpret_cast<Fill>(temporary->vtable[17])(temporary,nullptr,int(0x80000456u))==expected_return,"null fill retains native return value");
        verify(!backend.owns(temporary),"untranslated null fill stays native");
        auto cleared=get(temporary,0,0),expected_clear=get(root,0,0);verify(cleared&&expected_clear,"null fill native leases");
        for(int pixel=0;pixel<w*h;++pixel)verify(cleared[pixel]==expected_clear[pixel],"null fill exact native pixels");
        release(temporary,1);release(root,1);
        reinterpret_cast<Destroy>(temporary->vtable[0])(temporary,1);verify(temporary_id&&!backend.image(temporary),"native destruction retires GPU ownership");
        verify(reinterpret_cast<Init>(target[2]->vtable[1])(target[2],w,h,16,1)==0,"new lifetime before detach");both_fill(2,full,0x80000567);
        reads=backend.stats().readbacks;set_custom_renderer_native_probe(nullptr);compare(2,true);
        verify(backend.stats().readbacks==reads+1&&state.custom_renderer_native_image==nullptr,"detach drains current pixels before removing hooks");
        for(auto image:target)reinterpret_cast<Destroy>(image->vtable[0])(image,1);
        for(auto image:control)reinterpret_cast<Destroy>(image->vtable[0])(image,1);
        reinterpret_cast<Destroy>(ui->vtable[0])(ui,1);reinterpret_cast<Destroy>(root->vtable[0])(root,1);
        verify(gpu.stats().resident_bytes>0,"drain releases all images, only shared overlap scratch remains");
        auto stats=backend.stats();std::printf("PASS hooked native GPU adapter: 6 phases exact native pixels; translated=%llu source_checks=%llu uploads=%llu fallback_readbacks=%llu bytes=%llu; zero readbacks in admitted GPU chain; retained-pointer edits, stretch, CPU access, reinit, config-off and detach pass\n",stats.translated,stats.source_checks,std::uint64_t(gpu.stats().uploads),stats.readbacks,stats.readback_bytes);
        reinterpret_cast<void(__thiscall*)(void*,unsigned)>(gt[0])(graph,1);FreeLibrary(jgl);return 0;
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

#ifndef C3X_NATIVE_WORKER_TEST
int main(int argc,char** argv){
    if(argc!=2)return 2;
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context));
    Backend gpu(device.Get(),context.Get());return native_adapter_contract(argv[1],gpu);
}
#endif
