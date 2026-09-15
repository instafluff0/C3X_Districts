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
int (*native_present_image)(void*,void*,void const*)=nullptr;
int translate(int op,void* object,void* source,void const* from,void const* to,unsigned color){
    if(op==C3X_NATIVE_IMAGE_PRESENT&&native_present_image)return native_present_image(object,source,from);
    if(op==C3X_NATIVE_IMAGE_DRAIN&&native_present_image)native_present_image(nullptr,nullptr,nullptr);
    return adapter->operation(op,object,source,from,to,color);
}
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
        RECT foreground={5,6,28,29};verify(fill(ui,&foreground,int(0x80002345u))==0,"native key foreground");
        using Draw=int(__thiscall*)(JGL_Image*,JGL_Image*,int,int);
        for(int phase=0;phase<3;++phase){
            verify(reinterpret_cast<Draw>(original[33])(ui,control[0],-3,2)==0,"oracle image draw");
            verify(reinterpret_cast<Draw>(ui->vtable[33])(ui,target[0],-3,2)==0,"hooked GPU image draw");compare(0);
        }
        verify(backend.stats().readbacks==0&&gpu.stats().uploads==upload_before+3,"keyed image chain reuses CPU upload without readback");
#ifdef C3X_NATIVE_WORKER_TEST
        // Full-color destination: only native UI is expanded. The map beneath
        // transparent pixels retains every BGRA bit from the production renderer.
        auto full_color=gpu.create(w,h,Format::bgra32);
        Command base={Kind::copy,full_color,map,{0,0,w,h},{0,0,w,h}};gpu.submit(&base,1);
        Command overlay={Kind::expand,full_color,backend.image(ui),{0,0,w,h},{0,0,w,h},0,0,0x7c1f};gpu.submit(&overlay,1);
        BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=w;info.bmiHeader.biHeight=-h;
        info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;void* rgb=nullptr;
        auto dc=CreateCompatibleDC(nullptr);auto bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&rgb,nullptr,0);verify(dc&&bitmap&&rgb,"native full-color UI oracle");
        auto previous=SelectObject(dc,bitmap);using UiDc=HDC(__thiscall*)(JGL_Image*);
        auto source_dc=reinterpret_cast<UiDc>(original[10])(ui);verify(BitBlt(dc,0,0,w,h,source_dc,0,0,SRCCOPY)!=FALSE,"GDI native UI color expansion");GdiFlush();
        reinterpret_cast<Release>(original[11])(ui,1);
        std::vector<unsigned> composed(w*h);gpu.readback(full_color,composed.data(),composed.size());
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){auto expected=x>=foreground.left&&x<foreground.right&&y>=foreground.top&&y<foreground.bottom?
            (static_cast<unsigned*>(rgb)[y*w+x]|0xff000000u):map_pixels[y*map_width+x];verify(composed[y*w+x]==expected,"full-color map plus native UI exact");}
        SelectObject(dc,previous);DeleteObject(bitmap);DeleteDC(dc);gpu.destroy(full_color);
#endif
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
        // Native sprite operations consume CPU source data, never destination bits.
        JGLSprite sprite={};auto base_address=reinterpret_cast<char*>(jgl);
        reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(base_address+0x7e80)(&sprite,nullptr);
        sprite.d=*reinterpret_cast<int*>(reinterpret_cast<char*>(ui)+0x7c);sprite.f28=*reinterpret_cast<int*>(reinterpret_cast<char*>(ui)+0x4d0);
        sprite.bits=get(ui,0,0);sprite.bit_count=16;sprite.stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(ui)+0x40);sprite.width=w;sprite.height=h;
        auto palette=reinterpret_cast<void*(__thiscall*)(JGL_Image*)>(original[58])(ui);
        using SpriteDraw=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*);
        auto original_sprite=reinterpret_cast<SpriteDraw>(state.custom_renderer_jgl_sprite_original);
        auto sprite_readbacks=backend.stats().readbacks;
        auto draw_sprite=[&](int x,int y,void* selected_palette){
            auto result=original_sprite(&sprite,control[2],x,y,selected_palette);
            verify(reinterpret_cast<SpriteDraw>(sprite.vtable[17])(&sprite,target[2],x,y,selected_palette)==result,"native sprite return");
            compare(2);verify(backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"native sprite has no destination readback");
        };
        auto sprite_uploads=gpu.stats().uploads;
        draw_sprite(3,4,palette);draw_sprite(-9,7,palette);draw_sprite(300,400,palette);
        verify(gpu.stats().uploads==sprite_uploads+1,"native sprite reuses unchanged source at new anchors");
        reinterpret_cast<unsigned short*>(sprite.bits)[9]=0x4567;draw_sprite(3,4,palette);
        verify(gpu.stats().uploads==sprite_uploads+2,"native sprite retained-pointer mutation updates source");
        auto source_bits=sprite.bits;
        auto sprite_palette=reinterpret_cast<void*(__thiscall*)(void*,void*)>(gt[30])(graph,nullptr);
        verify(sprite_palette!=nullptr,"indexed sprite palette");
        auto palette_table=*static_cast<void***>(sprite_palette);
        auto colors=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(palette_table[6])(sprite_palette);
        verify(colors!=nullptr,"555 palette words");
        for(unsigned n=0;n<256;++n)colors[n]=static_cast<unsigned short>(n*127);colors[0]=0x7c1f;
        std::vector<unsigned char> indexed(w*h);for(unsigned n=0;n<indexed.size();++n)indexed[n]=n%256;
        sprite.bit_count=8;sprite.bits=indexed.data();sprite.stride=w;
        draw_sprite(-3,2,sprite_palette);draw_sprite(5,-4,sprite_palette);
        colors[17]^=0x7fff;draw_sprite(0,0,sprite_palette);
        indexed[12]=42;draw_sprite(0,0,sprite_palette);
        verify(backend.stats().readbacks==sprite_readbacks,"all palette indices including opaque magenta and transparent 254/255 avoid readback");
        // Actual native row-trimmed layout: left/count bytes and a 16-bit
        // offset into the source stream. Descriptor edits are content changes.
        std::vector<unsigned char> row_data,row_headers(h*4);
        for(int y=0;y<h;++y){unsigned left=3+y%4,count=w-left-5,offset=unsigned(row_data.size());
            row_headers[y*4]=static_cast<unsigned char>(left);row_headers[y*4+1]=static_cast<unsigned char>(count);row_headers[y*4+2]=static_cast<unsigned char>(offset&255);row_headers[y*4+3]=static_cast<unsigned char>(offset>>8);
            row_data.insert(row_data.end(),indexed.begin()+y*w+left,indexed.begin()+y*w+left+count);
        }
        sprite.f18=1;sprite.f1c=reinterpret_cast<int>(row_headers.data());sprite.bits=row_data.data();
        draw_sprite(0,0,sprite_palette);draw_sprite(-7,9,sprite_palette);
        row_headers[1]-=3;draw_sprite(4,3,sprite_palette);
        sprite.f18=0;sprite.f1c=0;
        // Unsupported 16-bit scaled program retains the native no-draw behavior
        // and synchronizes before it can acquire destination bytes.
        sprite.f28=0x80007c1f;sprite.bit_count=16;sprite.bits=source_bits;sprite.stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(ui)+0x40);
        auto scale=reinterpret_cast<int*>(base_address+0x6c0fc);int saved_scale=*scale;*scale=saved_scale*2;
        auto result=original_sprite(&sprite,control[2],3,4,palette);
        verify(reinterpret_cast<SpriteDraw>(sprite.vtable[17])(&sprite,target[2],3,4,palette)==result,"unsupported native sprite return");
        *scale=saved_scale;compare(2,true);verify(!backend.owns(target[2]),"unsupported sprite restores complete destination");
        reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(base_address+0x3cf10)(sprite_palette,1);
        sprite.bits=nullptr;reinterpret_cast<void(__thiscall*)(JGLSprite*)>(base_address+0x7ed0)(&sprite);release(ui,1);
        verify(reinterpret_cast<Init>(target[2]->vtable[1])(target[2],w,h,16,1)==0,"fresh palette destination");both_fill(2,full,0x80000567);
        both_fill(2,{3,4,17,19},2);compare(2,true);verify(!backend.owns(target[2]),"palette-index fill retains native semantics after CPU barrier");
        verify(reinterpret_cast<Init>(target[2]->vtable[1])(target[2],w,h,16,1)==0,"fresh config-off destination");both_fill(2,full,0x80000567);
        RECT outside_source={-2,0,62,48};
        verify(copy(ui,control[2],&outside_source,&full)==0,"native clipped-source control");
        verify(reinterpret_cast<Copy>(ui->vtable[16])(ui,target[2],&outside_source,&full)==0,"queued adapter rejects unsupported source before skipping native draw");
        compare(2,true);verify(!backend.owns(target[2]),"unsupported source bounds restore CPU before native copy");
        verify(reinterpret_cast<Init>(target[2]->vtable[1])(target[2],w,h,16,1)==0,"fresh config-off destination after source fallback");both_fill(2,full,0x80000567);
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
        #ifndef C3X_NATIVE_WORKER_TEST
        verify(gpu.stats().resident_bytes==w*h*4,"drain releases all images, only shared overlap scratch remains");
#endif
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
