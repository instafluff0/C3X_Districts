#define C3X_NATIVE_ADAPTER_TEST
#include "test_native_observation.cpp"
#include "native_image_adapter.h"
#include "native_sprite_diagnostics.h"
#include "test_native_ui_assets.h"
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
        // A real game has already configured its graphics owner. The misleadingly
        // named export is a factory, only appropriate for this isolated setup.
        auto module=reinterpret_cast<char*>(jgl);
        *reinterpret_cast<int*>(static_cast<char*>(graph)+0x134)=16;
        auto same_owner=[&]{return *reinterpret_cast<void**>(module+0x70d30)==graph&&
            *reinterpret_cast<void**>(module+0x70f30)==graph;};
        auto create=reinterpret_cast<Create>(gt[31]);
        constexpr int w=64,h=48;RECT full={0,0,w,h};
        auto root=create(graph,nullptr,1);verify(reinterpret_cast<Init>(root->vtable[1])(root,w,h,16,1)==0,"root init");
        void* original[60];std::memcpy(original,root->vtable,sizeof original);
        auto init=reinterpret_cast<Init>(original[1]);auto fill=reinterpret_cast<Fill>(original[17]);auto copy=reinterpret_cast<Copy>(original[16]);
        auto get=reinterpret_cast<Get>(original[3]);auto release=reinterpret_cast<Release>(original[9]);
        auto replace_storage=[&](JGL_Image* image){auto fn=reinterpret_cast<Init>(image->vtable[1]);
            int code=fn(image,w+1,h,16,1);return code?code:fn(image,w,h,16,1);};
        JGL_Image* control[3];JGL_Image* target[3];
        for(auto& image:control){image=create(graph,nullptr,1);verify(init(image,w,h,16,1)==0,"control init");verify(fill(image,&full,int(0x80000000u))==0,"control clear");}
        // Preexisting CPU UI is never assumed GPU-owned. Keep a raw pointer across
        // every GPU use to witness writes that have no getter/release event.
        auto ui=create(graph,nullptr,1);verify(init(ui,w,h,16,1)==0,"UI init");verify(fill(ui,&full,int(0x80000123u))==0,"UI clear");
        auto retained=get(ui,0,0);verify(retained!=nullptr,"retained UI pointer");release(ui,1);
        c3x_native_images::Adapter<Backend> backend(gpu,original[4],original[9]);adapter=&backend;
        capture.write=log_line;state.custom_renderer_native_observe=observe;set_custom_renderer_native_probe(root);
        verify(state.custom_renderer_native_probe_active,"actual hook attach");
        verify(same_owner(),"hook attachment preserves the configured JGL owner");
        auto default_ui=create(graph,nullptr,1);
        verify(reinterpret_cast<Init>(default_ui->vtable[1])(default_ui,w,h,0,1)==0&&
            *reinterpret_cast<int*>(reinterpret_cast<char*>(default_ui)+0x24)==16,"new UI inherits native 16-bit mode after hook attachment");
        reinterpret_cast<Destroy>(original[0])(default_ui,1);
        state.custom_renderer_native_image=translate;
        for(auto& image:target){image=create(graph,nullptr,1);verify(reinterpret_cast<Init>(image->vtable[1])(image,w,h,16,1)==0,"admitted init");
            verify(backend.owns(image),"fresh destination admitted");verify(reinterpret_cast<Fill>(image->vtable[17])(image,&full,int(0x80000000u))==0,"GPU clear native return");}
        // Long-lived CPU UI sources must not exhaust destination slots. Their
        // mirrors are disposable; real CPU pixels remain authoritative.
        std::vector<JGL_Image*> churn;
        auto before_churn=backend.stats().readbacks;
        for(int n=0;n<48;++n){auto image=create(graph,nullptr,1);churn.push_back(image);
            verify(init(image,w,h,16,1)==0&&fill(image,&full,int(0x80000000u|unsigned(n)))==0,"CPU source churn init");
            verify(reinterpret_cast<Copy>(image->vtable[16])(image,target[0],&full,&full)==0&&backend.owns(target[0]),"CPU source churn preserves GPU destination");
        }
        verify(backend.stats().readbacks==before_churn,"CPU source cache eviction never drains owned maps");
        for(auto image:churn)reinterpret_cast<Destroy>(image->vtable[0])(image,1);
        verify(reinterpret_cast<Fill>(target[0]->vtable[17])(target[0],&full,int(0x80000000u))==0,"restore zero parity baseline after churn");
        std::puts("PASS native CPU source churn: 48 sources, owned destination preserved, zero readbacks");
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
        // Real label backgrounds and border/indicator lines, through the same
        // paired owner. Native rasterization supplies the independent oracle.
        using Tint=int(__thiscall*)(JGL_Image*,RECT*,int,int);
        using Line=int(__thiscall*)(JGL_Image*,int,int,int,int,int,int);
        auto original_tint=reinterpret_cast<Tint>(original[18]);auto original_line=reinterpret_cast<Line>(original[25]);
        auto shape_readbacks=backend.stats().readbacks;
        RECT shape_clip={3,4,61,45};
        for(auto image:{control[0],target[0]})verify(reinterpret_cast<Clip>(image->vtable[13])(image,&shape_clip)==0,"native shape clip");
        auto shape_detail=backend.display_image(target[0]);verify(shape_detail!=0,"shape full-color destination");
        std::vector<unsigned> detail_seed(w*h),detail_result(w*h);std::uint64_t shape_revision=0;
        for(int percent:{-1,0,1,25,50,99,100,101})for(unsigned color:{0x80000000u,0x80007fffu,0x80004c96u}){
            both_fill(0,full,0x80004c96u);
            std::fill(detail_seed.begin(),detail_seed.end(),0xffa52cb9u);verify(gpu.upload(shape_detail,++shape_revision,detail_seed.data(),detail_seed.size()),"independent full-color shape background");
            RECT area={-3,7,72,33};
            verify(original_tint(control[0],&area,int(color),percent)==0,"native tint oracle");
            verify(reinterpret_cast<Tint>(target[0]->vtable[18])(target[0],&area,int(color),percent)==0,"GPU tint hook");compare(0);
            verify(gpu.readback(shape_detail,detail_result.data(),detail_result.size()),"tint detail oracle");
            auto words=get(control[0],0,0);int pitch=*reinterpret_cast<int*>(reinterpret_cast<char*>(control[0])+0x40);
            // Native expansion replicates low bits, rather than rounding 255/31.
            auto native_rgb=[](unsigned c){unsigned b=c&31,g=(c>>5)&31,r=(c>>10)&31;return ((b<<3)|(b>>2))|((g<<3)|(g>>2))<<8|((r<<3)|(r>>2))<<16;};
            int weight=std::clamp(percent,0,100)*256/100;
            for(int y=0;y<h;++y)for(int x=0;x<w;++x){unsigned expected=detail_seed[y*w+x];
                if(x>=shape_clip.left&&x<shape_clip.right&&y>=area.top&&y<area.bottom){unsigned native=native_rgb(words[y*pitch+x]),below=native_rgb(0x4c96);expected=0xff000000u;
                    for(unsigned shift:{0u,8u,16u}){int value=int((native>>shift)&255)+(int((detail_seed[y*w+x]>>shift)&255)-int((below>>shift)&255))*weight/256;
                        expected|=unsigned(std::clamp(value,0,255))<<shift;}}
                verify(detail_result[y*w+x]==expected,"native tint preserves independent full-color map contribution");}
            release(control[0],1);
        }
        for(int x0:{-4,0,7,32,63,68})for(int y0:{-3,0,8,47,52})for(int x1:{0,31,63})for(int y1:{0,23,47}){
            both_fill(0,full,0x80000567u);
            verify(original_line(control[0],x0,y0,x1,y1,int(0x80002defu),1)==0,"native line oracle");
            verify(reinterpret_cast<Line>(target[0]->vtable[25])(target[0],x0,y0,x1,y1,int(0x80002defu),1)==0,"GPU line hook");
            try{compare(0);}catch(...){std::fprintf(stderr,"LINE_CASE %d,%d -> %d,%d\n",x0,y0,x1,y1);throw;}
        }
        for(auto image:{control[0],target[0]})verify(reinterpret_cast<Clip>(image->vtable[13])(image,&full)==0,"native shape unclip");
        verify(backend.owns(target[0])&&backend.stats().readbacks==shape_readbacks,"label background/line chain stays resident");
        using Lookup=int(__thiscall*)(JGL_Image*,RECT*,JGL_Image*,int,void*);
        auto original_lookup=reinterpret_cast<Lookup>(original[21]);
        std::vector<unsigned short> lookup_table(524288);
        for(unsigned n=0;n<lookup_table.size();++n)lookup_table[n]=static_cast<unsigned short>((n*109+(n>>15)*977)^(n>>7));
        auto lookup_readbacks=backend.stats().readbacks;
        RECT lookup_area={-3,2,72,46},lookup_clip={12,12,40,30};
        both_fill(1,full,0x80003579u);
        for(auto image:{control[0],target[0]})verify(reinterpret_cast<Clip>(image->vtable[13])(image,&lookup_clip)==0,"lookup clip metadata");
        std::vector<unsigned> lookup_seed(w*h);std::uint64_t lookup_seed_revision=0;
        auto lookup_seed_image=gpu.create(w,h,Format::rgb555);verify(lookup_seed_image!=0,"lookup oracle seed image");
        for(int percent:{0,1,6,7,20,25,39,40,41,50,60,75,80,93,99,100}){
            auto bits=get(control[0],0,0);auto pitch=*reinterpret_cast<int*>(reinterpret_cast<char*>(control[0])+0x40);
            for(unsigned n=0;n<lookup_seed.size();++n){lookup_seed[n]=n%7==0?0x7c1f:((n*313+percent*101)&65535);bits[(n/w)*pitch+n%w]=static_cast<unsigned short>(lookup_seed[n]);}
            release(control[0],1);
            verify(gpu.upload(lookup_seed_image,++lookup_seed_revision,lookup_seed.data(),lookup_seed.size()),"lookup native seed");
            Command seed_command={Kind::copy,backend.image(target[0]),lookup_seed_image,{0,0,w,h},{0,0,w,h}};
            verify(gpu.submit(&seed_command,1),"copy native seed without altering adapter upload revision");
            auto links=control[0]->Bits_Data_Links,bg_links=control[1]->Bits_Data_Links;
            verify(original_lookup(control[0],&lookup_area,control[1],percent,lookup_table.data())==0,"native lookup oracle");
            control[0]->Bits_Data_Links=links;control[1]->Bits_Data_Links=bg_links;
            verify(reinterpret_cast<Lookup>(target[0]->vtable[21])(target[0],&lookup_area,target[1],percent,lookup_table.data())==0,"resident lookup hook");compare(0);
        }
        verify(gpu.destroy(lookup_seed_image),"retire independent lookup seed source");
        // Mutable table bytes, not pointer identity, define cache validity.
        auto lookup_uploads=gpu.stats().uploads;
        for(int repeat=0;repeat<2;++repeat){
            verify(reinterpret_cast<Lookup>(target[0]->vtable[21])(target[0],&lookup_area,target[0],40,lookup_table.data())==0,"self lookup table reuse");}
        verify(gpu.stats().uploads==lookup_uploads,"unchanged lookup table stays uploaded");
        lookup_table[10*32768+0x1234]^=0x7fff;
        verify(reinterpret_cast<Lookup>(target[0]->vtable[21])(target[0],&lookup_area,target[0],40,lookup_table.data())==0,"lookup table mutation");
        verify(gpu.stats().uploads==lookup_uploads+1,"same-pointer lookup edit is uploaded");
        // Identity tables must preserve independent full-color map pixels exactly.
        for(unsigned n=0;n<lookup_table.size();++n)lookup_table[n]=static_cast<unsigned short>(n&32767);
        for(auto image:{control[0],target[0]})verify(reinterpret_cast<Clip>(image->vtable[13])(image,&full)==0,"lookup unclip");
        both_fill(0,full,0x80001234u);
        for(unsigned n=0;n<detail_seed.size();++n)detail_seed[n]=0xff000000u|((n*37177)&0xffffff);
        verify(gpu.upload(shape_detail,++shape_revision,detail_seed.data(),detail_seed.size()),"lookup independent full-color seed");
        verify(reinterpret_cast<Lookup>(target[0]->vtable[21])(target[0],&full,target[0],40,lookup_table.data())==0,"identity lookup full-color hook");
        verify(gpu.readback(shape_detail,detail_result.data(),detail_result.size())&&detail_result==detail_seed,"identity lookup preserves full map precision");
        // A channel permutation is a non-identity, independently predictable LUT.
        for(unsigned n=0;n<lookup_table.size();++n)lookup_table[n]=static_cast<unsigned short>(((n&31)<<10)|(n&0x3e0)|((n>>10)&31));
        verify(reinterpret_cast<Lookup>(target[0]->vtable[21])(target[0],&full,target[0],40,lookup_table.data())==0,"channel-remapping lookup full-color hook");
        verify(gpu.readback(shape_detail,detail_result.data(),detail_result.size()),"lookup color-remapping oracle");
        for(unsigned n=0;n<detail_result.size();++n){auto c=detail_seed[n];verify(detail_result[n]==(0xff000000u|((c&255)<<16)|(c&0xff00)|((c>>16)&255)),"lookup transforms full-color map channels without truncation");}
        verify(backend.stats().readbacks==lookup_readbacks&&backend.owns(target[0]),"lookup chain never acquires CPU destination pixels");
        both_fill(0,full,0x80001234u); // Restore paired fixture state after independent color remapping.
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
        // Native text uses resident font responses across anchors. Metadata DC
        // inspection is private; a translated label never restores map pixels.
        auto text_dc=*reinterpret_cast<HDC*>(reinterpret_cast<char*>(target[2])+0x4bc);
        SetBkMode(text_dc,TRANSPARENT);SetTextAlign(text_dc,TA_LEFT|TA_TOP);SetTextColor(text_dc,RGB(233,175,77));
        auto text_before=backend.stats();auto text_uploads=gpu.stats().uploads;
        for(int x:{2,8,11})verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,int,int,char const*,int)>(target[2]->vtable[46])(target[2],x,3,"Civ III",7)==0,"hooked cached text");
        verify(backend.owns(target[2])&&backend.stats().readbacks==text_before.readbacks&&backend.stats().text_builds==text_before.text_builds+1&&backend.stats().text_hits==text_before.text_hits+2&&gpu.stats().uploads==text_uploads+2,"text cache retains one glyph/response pair, no background readback or repeat upload");
        both_fill(2,full,0x80000000);compare(2);
        // Native keyed copies must test packed words while preserving the
        // source's independent full-color pixels.
        backend.display_image(target[0]);backend.display_image(target[1]);
        for(auto key:{0x7c1fu,0x2345u,0u}){
            both_fill(0,{8,9,22,21},0x80000000u|key);
            *reinterpret_cast<unsigned*>(reinterpret_cast<char*>(control[0])+0x4d0)=0x80000000u|key;
            *reinterpret_cast<unsigned*>(reinterpret_cast<char*>(target[0])+0x4d0)=0x80000000u|key;
            for(int to:{1}){
                std::vector<unsigned> native(w*h),source_rgb(w*h),expected(w*h),actual(w*h);
                gpu.readback(backend.image(target[0]),native.data(),native.size());
                gpu.readback(backend.display_image(target[0]),source_rgb.data(),source_rgb.size());
                gpu.readback(backend.display_image(target[to]),expected.data(),expected.size());
                for(int y=2;y<h;++y)for(int x=0;x<w-3;++x)if(native[(y-2)*w+x+3]!=key)expected[y*w+x]=source_rgb[(y-2)*w+x+3];
                verify(reinterpret_cast<Draw>(original[33])(control[0],control[to],-3,2)==0,"native resident keyed copy");
                verify(reinterpret_cast<Draw>(target[0]->vtable[33])(target[0],target[to],-3,2)==0,"GPU resident keyed copy");compare(to);
                gpu.readback(backend.display_image(target[to]),actual.data(),actual.size());verify(actual==expected,"native key preserves independent full-color source");
            }
        }
        // Actual JGL StretchBlt is the native oracle. A separate 32-bit GDI
        // surface checks that the same transfer preserves full-color precision.
        unsigned detail_revision=100;
        for(auto extent:std::array<std::array<int,2>,8>{{{{32,24}},{{43,31}},{{61,43}},{{91,73}},{{7,5}},{{13,17}},{{97,19}},{{23,79}}}}){
            for(int to:{1,0}){
                // A fresh varied source prevents earlier shrink results from
                // making later cases pass with an accidentally uniform image.
                for(int n=0;n<w*h;++n)retained[n]=std::uint16_t((n*193u+79u)&0x7fff);
                verify(copy(ui,control[0],&full,&full)==0,"fresh stretch source oracle");
                verify(reinterpret_cast<Copy>(ui->vtable[16])(ui,target[0],&full,&full)==0,"fresh stretch source GPU");
                std::vector<unsigned> rich(w*h);for(int n=0;n<w*h;++n){unsigned c=retained[n];rich[n]=0xff000000u|((c&31)<<3)|(((c>>5)&31)<<11)|(((c>>10)&31)<<19)|(unsigned(n*137)&0x070707);}
                verify(gpu.upload(backend.display_image(target[0]),++detail_revision,rich.data(),rich.size()),"independent full-color source detail");
                RECT from={5,3,62,46},scaled={-3,-2,extent[0]-3,extent[1]-2};
                std::vector<unsigned> source_rgb(w*h),destination_rgb(w*h),actual(w*h);
                gpu.readback(backend.display_image(target[0]),source_rgb.data(),source_rgb.size());
                gpu.readback(backend.display_image(target[to]),destination_rgb.data(),destination_rgb.size());
                HDC stretch_dc[2]={CreateCompatibleDC(nullptr),CreateCompatibleDC(nullptr)};HBITMAP stretch_bitmap[2]={};HGDIOBJ old_bitmap[2]={};void* stretch_pixels[2]={};
                BITMAPINFO stretch_info={};stretch_info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);stretch_info.bmiHeader.biWidth=w;stretch_info.bmiHeader.biHeight=-h;stretch_info.bmiHeader.biPlanes=1;stretch_info.bmiHeader.biBitCount=32;
                for(int n=0;n<2;++n){stretch_bitmap[n]=CreateDIBSection(stretch_dc[n],&stretch_info,DIB_RGB_COLORS,&stretch_pixels[n],nullptr,0);verify(stretch_dc[n]&&stretch_bitmap[n]&&stretch_pixels[n],"full-color stretch oracle surface");old_bitmap[n]=SelectObject(stretch_dc[n],stretch_bitmap[n]);}
                std::memcpy(stretch_pixels[0],source_rgb.data(),source_rgb.size()*4);std::memcpy(stretch_pixels[1],destination_rgb.data(),destination_rgb.size()*4);
                verify(StretchBlt(stretch_dc[1],scaled.left,scaled.top,extent[0],extent[1],stretch_dc[0],from.left,from.top,from.right-from.left,from.bottom-from.top,SRCCOPY)!=FALSE,"independent GDI full-color stretch");GdiFlush();
                both_copy(0,to,from,scaled);compare(to);
                gpu.readback(backend.display_image(target[to]),actual.data(),actual.size());
                for(int n=0;n<w*h;++n)verify((actual[n]&0xffffff)==(static_cast<unsigned*>(stretch_pixels[1])[n]&0xffffff),"full-color stretch exact GDI channels");
                for(int n=0;n<2;++n){SelectObject(stretch_dc[n],old_bitmap[n]);DeleteObject(stretch_bitmap[n]);DeleteDC(stretch_dc[n]);}
            }
        }
        verify(backend.owns(target[0])&&backend.owns(target[1])&&backend.stats().readbacks==0,"scaled and keyed native/full-color chains remain GPU-owned");
        // Unlike StretchBlt, transparent Sprite::draw traverses its source in
        // place. Preserve that native self-overlap behavior through one barrier.
        verify(reinterpret_cast<Draw>(original[33])(control[0],control[0],-3,2)==0,"native transparent self-draw");
        verify(reinterpret_cast<Draw>(target[0]->vtable[33])(target[0],target[0],-3,2)==0,"transparent self-draw fallback");
        compare(0,true);verify(!backend.owns(target[0])&&backend.stats().readbacks==1,"native traversal restored before self-draw");
        // Explicit public access still restores each dirty native image once.
        for(int n=0;n<2;++n){verify(reinterpret_cast<Get>(target[n]->vtable[7])(target[n],0,0)!=nullptr,"public image lease");release(target[n],1);compare(n,true);}
        verify(backend.stats().readbacks==2,"one readback for each explicitly exposed image");
        // Public bits access must restore current output before returning. A later
        // native write remains native even after releasing the original pointer.
        auto public_bits=reinterpret_cast<Get>(target[2]->vtable[7])(target[2],0,0);verify(public_bits!=nullptr,"public lease");release(target[2],1);compare(2,true);
        auto reads=backend.stats().readbacks;both_fill(2,{0,0,2,2},0x80000456);compare(2,true);
        verify(!backend.owns(target[2])&&backend.stats().readbacks==reads,"CPU exposure permanently demotes lifetime");
        verify(reinterpret_cast<Init>(target[2]->vtable[1])(target[2],w,h,16,1)==0&&!backend.owns(target[2]),"no-op native init cannot reacquire an escaped image");
        auto old=backend.image(target[2]);verify(replace_storage(target[2])==0,"reinitialize lifetime");
        verify(backend.owns(target[2])&&backend.image(target[2])!=old,"reinitialize retires GPU identity");
        both_fill(2,full,0x80000567);compare(2);
        using Dc=HDC(__thiscall*)(JGL_Image*);
        auto native_dc=reinterpret_cast<Dc>(control[2]->vtable[10])(control[2]);
        auto gpu_dc=reinterpret_cast<Dc>(target[2]->vtable[10])(target[2]);
        verify(native_dc&&gpu_dc,"public HDC leases");
        verify(PatBlt(native_dc,3,4,12,8,DSTINVERT)&&PatBlt(gpu_dc,3,4,12,8,DSTINVERT),"CPU destination-dependent GDI drawing");GdiFlush();
        reinterpret_cast<Release>(control[2]->vtable[11])(control[2],1);reinterpret_cast<Release>(target[2]->vtable[11])(target[2],1);
        compare(2,true);verify(!backend.owns(target[2]),"public HDC receives current GPU pixels before native invert");
        verify(replace_storage(target[2])==0,"new lifetime before config off");both_fill(2,full,0x80000567);
        // Native sprite operations consume CPU source data, never destination bits.
        JGLSprite sprite={};auto base_address=reinterpret_cast<char*>(jgl);
        reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(base_address+0x7e80)(&sprite,nullptr);
        sprite.d=*reinterpret_cast<int*>(reinterpret_cast<char*>(ui)+0x7c);sprite.f28=*reinterpret_cast<int*>(reinterpret_cast<char*>(ui)+0x4d0);
        sprite.bits=get(ui,0,0);sprite.bit_count=16;sprite.stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(ui)+0x40);sprite.width=w;sprite.height=h;
        auto palette=reinterpret_cast<void*(__thiscall*)(JGL_Image*)>(original[58])(ui);
        using SpriteDraw=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*);
        auto original_sprite=reinterpret_cast<SpriteDraw>(state.custom_renderer_jgl_sprite_original);
        JGLSprite empty_sprite={};reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(base_address+0x7e80)(&empty_sprite,nullptr);
        auto empty_readbacks=backend.stats().readbacks;bool empty_background_owned=backend.owns(target[0]);
        verify(original_sprite(&empty_sprite,control[2],0,0,nullptr)==7,"native empty sprite returns 7");
        for(int n=0;n<24;++n)verify(reinterpret_cast<SpriteDraw>(empty_sprite.vtable[17])(&empty_sprite,target[2],n,n,nullptr)==7,"hook preserves native empty sprite return");
        using EmptyBlend=int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*);
        using EmptyBlendOnto=int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,int,int,void*);
        verify(reinterpret_cast<EmptyBlend>(state.custom_renderer_jgl_blend_original[0])(&empty_sprite,&empty_sprite,control[0],control[2],0,0,nullptr)==23,"native empty blend returns 23");
        verify(reinterpret_cast<EmptyBlend>(empty_sprite.vtable[20])(&empty_sprite,&empty_sprite,target[0],target[2],0,0,nullptr)==23,"hook preserves native empty blend return");
        verify(reinterpret_cast<EmptyBlendOnto>(empty_sprite.vtable[21])(&empty_sprite,&empty_sprite,target[2],0,0,nullptr)==23,"hook preserves native empty blend-onto return");
        verify(reinterpret_cast<EmptyBlendOnto>(empty_sprite.vtable[22])(&empty_sprite,&empty_sprite,target[2],0,0,nullptr)==23,"hook preserves native empty alpha return");
        verify(reinterpret_cast<SpriteDraw>(empty_sprite.vtable[23])(&empty_sprite,target[2],0,0,nullptr)==7,"empty keyed sprite keeps native return");
        using EmptyMask=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,int,void*);
        verify(reinterpret_cast<EmptyMask>(empty_sprite.vtable[29])(&empty_sprite,target[2],0,0,0,nullptr)==reinterpret_cast<EmptyMask>(state.custom_renderer_jgl_blend_original[7])(&empty_sprite,control[2],0,0,0,nullptr),"empty mask keeps native return");
        using EmptyLookup=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*,void*);
        for(int slot:{31,33})verify(reinterpret_cast<EmptyLookup>(empty_sprite.vtable[slot])(&empty_sprite,target[2],0,0,nullptr,nullptr)==7,"empty shadow/lookup keeps native return");
        using EmptyOpacity=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,float,void*,int);
        verify(reinterpret_cast<EmptyOpacity>(empty_sprite.vtable[37])(&empty_sprite,target[2],0,0,.5f,nullptr,0)==7,"empty opacity keeps native return");
        verify(backend.owns(target[0])==empty_background_owned&&backend.owns(target[2])&&backend.stats().readbacks==empty_readbacks,"empty native draws preserve resident underlay without readbacks");
        compare(2);
        reinterpret_cast<void(__thiscall*)(JGLSprite*)>(base_address+0x7ed0)(&empty_sprite);
        std::puts("PASS empty native sprites: exact error returns, no destination readback or ownership loss");
        auto sprite_readbacks=backend.stats().readbacks;
        auto draw_sprite=[&](int x,int y,void* selected_palette){
            auto key=sprite.f28;auto result=original_sprite(&sprite,control[2],x,y,selected_palette);
            auto expected_key=sprite.f28;sprite.f28=key;
            verify(reinterpret_cast<SpriteDraw>(sprite.vtable[17])(&sprite,target[2],x,y,selected_palette)==result,"native sprite return");
            verify(sprite.f28==expected_key,"native source-key side effect");
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
        // The real game can stay on CPU composition for its entire session.
        // Its diagnostics must work without constructing a map/Adapter owner.
        c3x_native_images::SpriteDiagnostics cpu_diagnostics;
        cpu_diagnostics.write=log_line;
        auto saved_colors=std::vector<unsigned short>(colors,colors+256);
        auto saved_indices=indexed;
        cpu_diagnostics.cpu_operation(C3X_NATIVE_SPRITE,&sprite,sprite_palette);
        verify(cpu_diagnostics.cpu_samples()==1,"CPU-only source sampled without GPU admission");
        cpu_diagnostics.cpu_operation(C3X_NATIVE_SPRITE,&sprite,sprite_palette);
        verify(cpu_diagnostics.cpu_samples()==1,"identical CPU source deduplicated");
        colors[17]^=0x7fff;
        cpu_diagnostics.cpu_operation(C3X_NATIVE_SPRITE,&sprite,sprite_palette);
        verify(cpu_diagnostics.cpu_samples()==2,"same-pointer palette change sampled");
        colors[17]^=0x7fff;
        c3x_renderer_native_sprite_style diagnostic_style={sprite_palette,nullptr,0,1};
        cpu_diagnostics.cpu_operation(C3X_NATIVE_SPRITE_STYLE,&sprite,&diagnostic_style);
        verify(cpu_diagnostics.cpu_samples()==3,"keyed UI has independent sampling budget");
        verify(std::equal(saved_colors.begin(),saved_colors.end(),colors)&&indexed==saved_indices,"CPU diagnostics preserve sprite and palette bytes");
        std::puts("PASS CPU-only diagnostics: independent of map admission, palette changes sampled, source bytes preserved");
        // Check actual CPU destination words before any later image transfer.
        RECT cpu_anchor={0,0,0,0};
        verify(original_sprite(&sprite,ui,0,0,sprite_palette)==0,"native CPU sprite for completion oracle");
        cpu_diagnostics.cpu_result(ui,&sprite,sprite_palette,&cpu_anchor,0);
        verify(cpu_diagnostics.completed_samples()==1&&cpu_diagnostics.completed_mismatches==0,"native CPU word oracle matches actual JGL draw");
        auto cpu_words=get(ui,0,0);verify(cpu_words!=nullptr,"CPU completion negative control");
        auto saved_word=cpu_words[0];cpu_words[0]^=1;release(ui,1);
        c3x_native_images::SpriteDiagnostics bad_cpu;bad_cpu.write=log_line;
        bad_cpu.cpu_result(ui,&sprite,sprite_palette,&cpu_anchor,0);
        verify(bad_cpu.completed_mismatches==1,"native CPU completion oracle detects wrong destination word");
        cpu_diagnostics.cpu_result(ui,&sprite,sprite_palette,&cpu_anchor,0);
        verify(cpu_diagnostics.completed_samples()==2&&cpu_diagnostics.completed_mismatches==1,"later bad use of the same source is not deduplicated away");
        cpu_words=get(ui,0,0);cpu_words[0]=saved_word;release(ui,1);
        auto indexed_destination=create(graph,nullptr,1);
        verify(init(indexed_destination,w,h,8,1)==0,"indexed intermediate init");
        verify(original_sprite(&sprite,indexed_destination,0,0,sprite_palette)==0,"native indexed intermediate draw");
        cpu_diagnostics.cpu_result(indexed_destination,&sprite,sprite_palette,&cpu_anchor,0);
        verify(cpu_diagnostics.completed_samples()==3&&cpu_diagnostics.completed_mismatches==1,"indexed intermediate oracle checks raw indices");
        reinterpret_cast<Destroy>(original[0])(indexed_destination,1);
        std::puts("PASS native CPU completion oracle: 8-bit indices, 16-bit palette words, deliberately corrupted destination detected");
        // Advisor calls commonly omit the explicit palette. Exercise both
        // source-local and Graphsy-default selection against the native helper.
        auto saved_palette=sprite.d;
        auto global_owner=reinterpret_cast<void**>(base_address+0x70f48);
        auto saved_global_owner=*global_owner;
        void* fixture_owner[2]={nullptr,sprite_palette};
        *global_owner=fixture_owner;
        sprite.d=int(reinterpret_cast<std::uintptr_t>(sprite_palette));
        draw_sprite(2,3,nullptr);
        sprite.d=0;
        draw_sprite(-3,2,nullptr);
        *global_owner=saved_global_owner;sprite.d=saved_palette;
        std::puts("PASS omitted sprite palette: source-local and Graphsy default match native pixels");
        // Use JGL's actual sprite allocator/compactor, as PCX slicing does.
        auto compression=reinterpret_cast<int*>(base_address+0x70d94);int saved_compression=*compression;*compression=1;
        for(int packing:{0,1})for(auto extent:std::array<std::array<int,2>,4>{{{{16,16}},{{26,30}},{{148,109}},{{708,140}}}}){
            int sw=extent[0],sh=extent[1];std::vector<unsigned char> art(sw*sh,255);
            for(int y=0;y<sh;++y)for(int x=0;x<sw;++x)if(x>y%9&&x<sw-y%7&&y%17!=0)art[y*sw+x]=static_cast<unsigned char>((x*79+y*23)%255);
            JGLSprite sliced={};
            reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(base_address+0x7e80)(&sliced,nullptr);
            using SpriteInit=int(__thiscall*)(JGLSprite*,void*,int,int,int,int,void*);
            verify(reinterpret_cast<SpriteInit>(sliced.vtable[1])(&sliced,art.data(),sw,sh,8,packing,sprite_palette)==0,"native sprite source construction");
            for(int slot:{17,23}){
                both_fill(2,full,0x80000567u);
                auto original_draw=slot==17?original_sprite:reinterpret_cast<SpriteDraw>(state.custom_renderer_jgl_blend_original[6]);
                auto expected=original_draw(&sliced,control[2],2,3,nullptr);
                verify(reinterpret_cast<SpriteDraw>(sliced.vtable[slot])(&sliced,target[2],2,3,nullptr)==expected,"native constructed sprite return");
                compare(2);
                auto detail=backend.display_image(target[2]);std::vector<unsigned> sprite_rgb(w*h);
                verify(detail&&gpu.readback(detail,sprite_rgb.data(),sprite_rgb.size()),"constructed sprite display oracle");
                auto words=get(control[2],0,0);int pitch=*reinterpret_cast<int*>(reinterpret_cast<char*>(control[2])+0x40);
                for(int y=0;y<h;++y)for(int x=0;x<w;++x){unsigned c=words[y*pitch+x];
                    unsigned expected_rgb=0xff000000u|((c&31)<<3)|((c&31)>>2)|(((c>>5&31)<<3)|(c>>7&7))<<8|(((c>>10&31)<<3)|(c>>12&7))<<16;
                    verify(sprite_rgb[y*w+x]==expected_rgb,"constructed sprite full-color display matches native");}
                release(control[2],1);
            }
            reinterpret_cast<void(__thiscall*)(JGLSprite*)>(base_address+0x7ed0)(&sliced);
        }
        *compression=saved_compression;
        c3x_native_images::SpriteDiagnostics diagnostic_probe;c3x_native_images::SpriteDiagnostics::Sample wrong;
        wrong.width=w;wrong.height=h;wrong.before.resize(w*h);wrong.expected.resize(w*h);
        auto oracle_words=get(control[2],0,0);int oracle_stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(control[2])+0x40);
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)wrong.expected[y*w+x]=oracle_words[y*oracle_stride+x];
        release(control[2],1);wrong.expected[0]^=1;
        diagnostic_probe.finish(gpu,backend.image(target[2]),0,wrong);
        verify(diagnostic_probe.samples==1&&diagnostic_probe.mismatches==1,"diagnostic detects deliberately mismatched expected pixel");compare(2);
        std::puts("PASS native-constructed UI sprites: raw/compacted, default palettes and full-color displayed pixels");
        auto scales=reinterpret_cast<int*>(base_address+0x6c0fc);
        std::array<int,3> saved_scales={scales[0],scales[1],scales[2]};
        for(auto extent:std::array<std::array<int,3>,7>{{{{1,1,2}},{{3,2,2}},{{2,3,2}},{{7,5,8}},{{13,17,11}},{{2,1,1}},{{1,3,2}}}}){
            for(unsigned i=0;i<3;++i)scales[i]=extent[i];
            draw_sprite(3,4,sprite_palette);draw_sprite(-9,-7,sprite_palette);
        }
        for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
        // Native single-key artwork, solid picking/selection masks and map
        // shadows share source sampling but preserve distinct index semantics.
        using SpriteMask=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,int,void*);
        using SpriteShadow=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*,void*);
        std::vector<unsigned short> shadow_table(4*32768);
        for(unsigned n=0;n<shadow_table.size();++n)shadow_table[n]=static_cast<unsigned short>((n*109+(n>>15)*977)&32767);
        auto check_style=[&](int slot,int x,int y,unsigned color){
            auto key=sprite.f28;int expected,actual;
            if(slot==23){expected=reinterpret_cast<SpriteDraw>(state.custom_renderer_jgl_blend_original[6])(&sprite,control[2],x,y,sprite_palette);
                auto key_after=sprite.f28;sprite.f28=key;actual=reinterpret_cast<SpriteDraw>(sprite.vtable[23])(&sprite,target[2],x,y,sprite_palette);verify(sprite.f28==key_after,"single-key sprite metadata");}
            else if(slot==29){expected=reinterpret_cast<SpriteMask>(state.custom_renderer_jgl_blend_original[7])(&sprite,control[2],x,y,int(color),sprite_palette);
                auto key_after=sprite.f28;sprite.f28=key;actual=reinterpret_cast<SpriteMask>(sprite.vtable[29])(&sprite,target[2],x,y,int(color),sprite_palette);verify(sprite.f28==key_after,"solid mask metadata");}
            else {expected=reinterpret_cast<SpriteShadow>(state.custom_renderer_jgl_blend_original[8])(&sprite,control[2],x,y,shadow_table.data(),sprite_palette);
                auto key_after=sprite.f28;sprite.f28=key;actual=reinterpret_cast<SpriteShadow>(sprite.vtable[31])(&sprite,target[2],x,y,shadow_table.data(),sprite_palette);verify(sprite.f28==key_after,"shadow sprite metadata");}
            if(actual!=expected)std::fprintf(stderr,"STYLE slot=%d flags=%d x=%d y=%d expected=%d actual=%d\n",slot,sprite.f18,x,y,expected,actual);
            verify(actual==expected,"native sprite-style return");compare(2);
            verify(backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"complete sprite-style chain remains GPU resident");
        };
        for(int slot:{23,29,31}){
            for(unsigned n=0;n<indexed.size();++n)indexed[n]=static_cast<unsigned char>(slot==31?248+n%6+(n%6>=4?2:0):n%256);
            for(auto extent:std::array<std::array<int,3>,4>{{{{1,1,1}},{{1,1,2}},{{3,2,2}},{{7,5,8}}}}){
                for(unsigned i=0;i<3;++i)scales[i]=extent[i];
                for(unsigned color:{17u,0x80003579u}){check_style(slot,3,4,color);check_style(slot,-9,-7,color);}
            }
        }
        for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
        for(unsigned n=0;n<indexed.size();++n)indexed[n]=static_cast<unsigned char>(n%256);
        using SpriteOpacity=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,float,void*,int);
        auto check_opacity=[&](float opacity,int flags,int x,int y){
            int expected=reinterpret_cast<SpriteOpacity>(state.custom_renderer_jgl_blend_original[9])(&sprite,control[2],x,y,opacity,sprite_palette,flags);
            int actual=reinterpret_cast<SpriteOpacity>(sprite.vtable[37])(&sprite,target[2],x,y,opacity,sprite_palette,flags);
            if(actual!=expected)std::fprintf(stderr,"OPACITY %.9g flags=%d x=%d y=%d scale=%d,%d/%d expected=%d actual=%d\n",opacity,flags,x,y,scales[0],scales[1],scales[2],expected,actual);
            verify(actual==expected,"opacity native return");compare(2);
            verify(backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"opacity transitions remain GPU resident");
        };
        for(auto extent:std::array<std::array<int,3>,4>{{{{1,1,1}},{{1,1,2}},{{3,2,2}},{{7,5,8}}}}){
            for(unsigned i=0;i<3;++i)scales[i]=extent[i];
            for(int flags:{0,1,256})for(unsigned n=0;n<=32;++n){
                check_opacity(float(n)/32,flags,3,4);check_opacity(float(n)/32,flags,-9,-7);
            }
        }
        for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
        for(unsigned n=1;n<32;n+=2)for(float delta:{-0.000001f,0.000001f})check_opacity(float(n)/32+delta,0,3,4);
        std::puts("PASS native opacity: all 16 steps, threshold neighbors, low-byte flags, indexed keys, clipping and positive scaling; zero destination readbacks");
        using SpriteLookup=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*,void*);
        auto original_sprite_lookup=reinterpret_cast<SpriteLookup>(state.custom_renderer_jgl_blend_original[3]);
        for(unsigned n=0;n<lookup_table.size();++n)lookup_table[n]=static_cast<unsigned short>((n*109+(n>>15)*977)^(n>>7));
        for(unsigned n=0;n<indexed.size();++n)indexed[n]=static_cast<unsigned char>(n%32);
        auto check_lookup_sprite=[&](int x,int y){
            both_fill(2,full,0x80003579u);
            auto result=original_sprite_lookup(&sprite,control[2],x,y,lookup_table.data(),sprite_palette);
            verify(reinterpret_cast<SpriteLookup>(sprite.vtable[33])(&sprite,target[2],x,y,lookup_table.data(),sprite_palette)==result,"GPU lookup sprite native return");compare(2);
            verify(backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"lookup sprite has no destination readback");
        };
        for(auto extent:std::array<std::array<int,3>,5>{{{{1,1,1}},{{1,1,2}},{{3,2,2}},{{2,3,2}},{{7,5,8}}}}){
            for(unsigned i=0;i<3;++i)scales[i]=extent[i];
            check_lookup_sprite(3,4);check_lookup_sprite(-9,-7);
        }
        for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
        indexed[4]=255;indexed[5]=254;lookup_table[0]^=0x7fff;check_lookup_sprite(0,0);
        // Native FLC drawing combines palette pixels and two lookup ranges,
        // using a separate map underlay only where the destination is magenta.
        using SpriteLookupOver=int(__thiscall*)(JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*,void*);
        auto original_lookup_over=reinterpret_cast<SpriteLookupOver>(state.custom_renderer_jgl_blend_original[4]);
        lookup_table.resize(31*32768);
        for(unsigned n=0;n<lookup_table.size();++n)lookup_table[n]=static_cast<unsigned short>((n*109+(n>>15)*977)&32767);
        for(unsigned n=0;n<indexed.size();++n)indexed[n]=static_cast<unsigned char>(n%256);
        auto check_lookup_over=[&](int x,int y,bool same,unsigned below){
            both_fill(0,full,0x80003579u);both_fill(2,full,0x80000000u|below);
            auto native_background=same?control[2]:control[0],gpu_background=same?target[2]:target[0];
            auto links=control[2]->Bits_Data_Links,bg_links=native_background->Bits_Data_Links;
            auto result=original_lookup_over(&sprite,native_background,control[2],x,y,lookup_table.data(),sprite_palette);
            control[2]->Bits_Data_Links=links;native_background->Bits_Data_Links=bg_links;
            verify(reinterpret_cast<SpriteLookupOver>(sprite.vtable[35])(&sprite,gpu_background,target[2],x,y,lookup_table.data(),sprite_palette)==result,"GPU native FLC lookup return");
            verify(backend.owns(target[2]),"FLC lookup admitted destination");compare(2);
            verify(backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"FLC lookup has no destination readback");
        };
        for(unsigned below:{0x7c1fu,0x1234u})for(bool same:{false,true})for(auto anchor:std::array<std::array<int,2>,3>{{{{0,0}},{{-9,7}},{{8,-5}}}})check_lookup_over(anchor[0],anchor[1],same,below);
        for(auto extent:std::array<std::array<int,3>,3>{{{{1,1,2}},{{-1,1,1}},{{1,-1,1}}}}){
            for(unsigned i=0;i<3;++i)scales[i]=extent[i];check_lookup_over(3,4,false,0x1234);}
        for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
        colors[17]^=0x7fff;lookup_table[30*32768+0x3579]^=0x7fff;check_lookup_over(0,0,false,0x7c1f);
        // The zoomed-out FLC entry is a distinct native source program. It
        // starts at source zero after clipping and selects every second byte.
        using SpriteLookupScaled=int(__thiscall*)(void*,JGL_Image*,JGL_Image*,int,int,int,int,int,void*,void*);
        auto original_lookup_scaled=reinterpret_cast<SpriteLookupScaled>(state.custom_renderer_jgl_blend_original[5]);
        for(auto extent:std::array<std::array<int,3>,4>{{{{1,1,2}},{{2,2,4}},{{1,1,3}},{{-1,-1,2}}}}){
            for(unsigned i=0;i<3;++i)scales[i]=extent[i];
            for(unsigned below:{0x7c1fu,0x1234u})for(bool same:{false,true})
            for(auto anchor:std::array<std::array<int,2>,4>{{{{0,0}},{{-9,7}},{{8,-5}},{{50,40}}}}){
                both_fill(2,full,0x80000000u|below);both_fill(1,full,0x80003579u);
                auto native_background=same?control[2]:control[1];auto gpu_background=same?target[2]:target[1];
                auto result=original_lookup_scaled(&sprite,native_background,control[2],anchor[0],anchor[1],extent[0],extent[1],extent[2],lookup_table.data(),sprite_palette);
                native_background->Bits_Data_Links=0;native_background->Current_Bits_Data=0;
                control[2]->Bits_Data_Links=0;control[2]->Current_Bits_Data=0;
                verify(reinterpret_cast<SpriteLookupScaled>(sprite.vtable[34])(&sprite,gpu_background,target[2],anchor[0],anchor[1],extent[0],extent[1],extent[2],lookup_table.data(),sprite_palette)==result,"GPU scaled FLC native return");
                verify(backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"scaled FLC composition stays resident");compare(2);
            }
        }
        for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
        auto lookup_background=create(graph,nullptr,1);
        verify(reinterpret_cast<Init>(lookup_background->vtable[1])(lookup_background,w,h,16,1)==0,"fresh full-color FLC underlay");
        verify(reinterpret_cast<Fill>(lookup_background->vtable[17])(lookup_background,&full,int(0x80002345u))==0,"FLC underlay native words");
        auto background_detail=backend.display_image(lookup_background),flc_detail=backend.display_image(target[2]);
        std::vector<unsigned> background_seed(w*h),flc_seed(w*h),flc_result(w*h);
        for(unsigned n=0;n<background_seed.size();++n){background_seed[n]=0xff000000u|((n*15719)&0xffffff);flc_seed[n]=0xff000000u|((n*73117)&0xffffff);}
        verify(gpu.upload(background_detail,1,background_seed.data(),background_seed.size()),"independent FLC full-color underlay");
        RECT flc_clip={3,4,61,45};verify(reinterpret_cast<Clip>(lookup_background->vtable[13])(lookup_background,&flc_clip)==0,"FLC background defines clip");
        for(unsigned n=0;n<lookup_table.size();++n)lookup_table[n]=static_cast<unsigned short>(n&32767);
        std::fill(indexed.begin(),indexed.end(),static_cast<unsigned char>(254));indexed[4*w+3]=17;indexed[4*w+4]=255;
        unsigned flc_revision=0;
        for(bool reduced:{false,true})for(unsigned native_below:{0x7c1fu,0x1234u}){
            scales[0]=scales[1]=1;scales[2]=reduced?2:1;indexed[2]=17;indexed[4]=255;
            both_fill(2,full,0x80000000u|native_below);
            verify(gpu.upload(flc_detail,++flc_revision,flc_seed.data(),flc_seed.size()),"independent FLC destination color");
            auto result=reduced?reinterpret_cast<SpriteLookupScaled>(sprite.vtable[34])(&sprite,lookup_background,target[2],0,0,1,1,2,lookup_table.data(),sprite_palette):
                reinterpret_cast<SpriteLookupOver>(sprite.vtable[35])(&sprite,lookup_background,target[2],0,0,lookup_table.data(),sprite_palette);
            verify(result==0,"ordinary/scaled FLC full-color shadow hook");
            verify(gpu.readback(flc_detail,flc_result.data(),flc_result.size()),"FLC full-color shadow oracle");
            for(int y=0;y<h;++y)for(int x=0;x<w;++x){unsigned n=y*w+x,expected=flc_seed[n];
                unsigned source_index=reduced&&x>=flc_clip.left&&y>=flc_clip.top?unsigned((y-flc_clip.top)*2*w+(x-flc_clip.left)*2):n;
                if(x>=flc_clip.left&&x<std::min(int(flc_clip.right),reduced?w/2:w)&&y>=flc_clip.top&&y<std::min(int(flc_clip.bottom),reduced?h/2:h)&&indexed[source_index]!=255){
                    expected=native_below==0x7c1f?background_seed[n]:flc_seed[n];
                    if(indexed[source_index]==17){unsigned c=colors[17],b=c&31,g=(c>>5)&31,r=(c>>10)&31;expected=0xff000000u|((b<<3)|(b>>2))|(((g<<3)|(g>>2))<<8)|(((r<<3)|(r>>2))<<16);}}
                verify(flc_result[n]==expected,"FLC native key selects correct full-color source; only palette pixels replace map precision");}
        }
        for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
        // Shadow lookup transforms retain full-color map channels; alternating
        // the native shadow/FLC tables reuses both resident assets.
        for(unsigned n=0;n<shadow_table.size();++n)shadow_table[n]=static_cast<unsigned short>(n&32767);
        for(unsigned n=0;n<indexed.size();++n)indexed[n]=static_cast<unsigned char>(248+n%4);
        both_fill(2,full,0x80001234u);
        verify(gpu.upload(flc_detail,++flc_revision,flc_seed.data(),flc_seed.size()),"independent map before native shadow");
        check_style(31,0,0,0);
        verify(gpu.readback(flc_detail,flc_result.data(),flc_result.size())&&flc_result==flc_seed,"native shadow identity table preserves all map color bits");
        auto alternate_tables=[&](){
            verify(reinterpret_cast<SpriteLookupOver>(sprite.vtable[35])(&sprite,target[2],target[2],0,0,lookup_table.data(),sprite_palette)==0,"FLC table stays resident beside shadow table");
            verify(reinterpret_cast<SpriteShadow>(sprite.vtable[31])(&sprite,target[2],0,0,shadow_table.data(),sprite_palette)==0,"shadow table stays resident beside FLC table");
        };
        alternate_tables();auto alternating_uploads=gpu.stats().uploads;alternate_tables();
        verify(gpu.stats().uploads==alternating_uploads+2,"alternating tables upload only changed sprite programs, not lookup assets");
        reinterpret_cast<Destroy>(lookup_background->vtable[0])(lookup_background,1);
        both_fill(2,full,0x80000567u);for(unsigned n=0;n<indexed.size();++n)indexed[n]=static_cast<unsigned char>(n%256);
        verify(backend.stats().readbacks==sprite_readbacks,"complete native FLC chain remains resident");
        // Blend into independent full-color map data. Native output words are
        // the oracle; only the map's extra precision follows the background weight.
        auto expand_opacity=[](unsigned c){unsigned b=c&31,g=(c>>5)&31,r=(c>>10)&31;return 0xff000000u|((b<<3)|(b>>2))|(((g<<3)|(g>>2))<<8)|(((r<<3)|(r>>2))<<16);};
        for(unsigned step=1;step<=15;++step){
            both_fill(2,full,0x80001234u);
            verify(gpu.upload(flc_detail,++flc_revision,flc_seed.data(),flc_seed.size()),"independent full-color opacity background");
            check_opacity(float(step)/16,0,0,0);
            verify(gpu.readback(flc_detail,flc_result.data(),flc_result.size()),"opacity full-color oracle read");
            auto native_words=get(control[2],0,0);auto native_stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(control[2])+0x40);
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){unsigned at=y*w+x,expected=flc_seed[at];
                if(indexed[at]!=255){expected=0xff000000u;auto native_color=expand_opacity(native_words[y*native_stride+x]),below=expand_opacity(0x1234);
                    for(unsigned channel=0;channel<3;++channel){unsigned shift=channel*8;int delta=int(flc_seed[at]>>shift&255)-int(below>>shift&255);
                        int value=int(native_color>>shift&255)+delta*int(16-step)/16;expected|=unsigned(std::max(0,std::min(255,value)))<<shift;}}
                verify(flc_result[at]==expected,"opacity preserves independent full-color map contribution");}
            release(control[2],1);
        }
        // Actual native row-trimmed layout: left/count bytes and a 16-bit
        // offset into the source stream. Descriptor edits are content changes.
        std::vector<unsigned char> row_data,row_headers(h*4);
        for(int y=0;y<h;++y){unsigned left=3+y%4,count=w-left-5,offset=unsigned(row_data.size());
            row_headers[y*4]=static_cast<unsigned char>(left);row_headers[y*4+1]=static_cast<unsigned char>(count);row_headers[y*4+2]=static_cast<unsigned char>(offset&255);row_headers[y*4+3]=static_cast<unsigned char>(offset>>8);
            row_data.insert(row_data.end(),indexed.begin()+y*w+left,indexed.begin()+y*w+left+count);
        }
        sprite.f18=1;sprite.f1c=reinterpret_cast<int>(row_headers.data());sprite.bits=row_data.data();
        check_opacity(0.5f,0,0,0);check_opacity(1.f,1,-7,9);
        for(int slot:{23,29,31}){check_style(slot,0,0,0x80003579u);check_style(slot,-7,9,17);}
        draw_sprite(0,0,sprite_palette);draw_sprite(-7,9,sprite_palette);
        row_headers[1]-=3;draw_sprite(4,3,sprite_palette);
        for(auto extent:std::array<std::array<int,3>,4>{{{{1,1,2}},{{3,2,2}},{{2,3,2}},{{7,5,8}}}}){
            for(unsigned i=0;i<3;++i)scales[i]=extent[i];
            // Exclude the unsafe final source-row edge of the native program.
            RECT trim_clip={2,3,59,19};
            reinterpret_cast<Clip>(control[2]->vtable[13])(control[2],&trim_clip);
            reinterpret_cast<Clip>(target[2]->vtable[13])(target[2],&trim_clip);
            draw_sprite(3,4,sprite_palette);draw_sprite(-9,-7,sprite_palette);
            check_style(29,3,4,17);check_style(29,-9,-7,0x80003579u);
            auto row_count=row_headers[4*4+1];row_headers[4*4+1]=0;
            draw_sprite(-2,0,sprite_palette);check_style(29,-2,0,17);row_headers[4*4+1]=row_count;
        }
        for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
        reinterpret_cast<Clip>(control[2]->vtable[13])(control[2],&full);
        reinterpret_cast<Clip>(target[2]->vtable[13])(target[2],&full);
        sprite.f18=0;sprite.f1c=0;
        // Main-screen chrome uses JGL's separate premultiplied/straight alpha
        // sprite slots. Exercise actual native routines with every alpha value,
        // including index 254 (drawn here) and 255 (skipped by these programs).
        verify(replace_storage(target[0])==0,"resident HUD underlay after earlier CPU-escape test");both_fill(0,full,0x80004567);
        JGLSprite alpha_sprite={};reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(base_address+0x7e80)(&alpha_sprite,nullptr);
        std::vector<unsigned char> alpha_bytes(256);for(unsigned n=0;n<256;++n)alpha_bytes[n]=static_cast<unsigned char>(n);
        alpha_sprite.bit_count=8;alpha_sprite.bits=alpha_bytes.data();alpha_sprite.stride=alpha_sprite.width=alpha_sprite.height=16;
        sprite.bit_count=8;sprite.bits=indexed.data();sprite.stride=sprite.width=sprite.height=16;
        for(unsigned n=0;n<256;++n){auto hud_rgb_entry=reinterpret_cast<unsigned char*(__thiscall*)(void*,unsigned)>(palette_table[8])(sprite_palette,n);
            verify(hud_rgb_entry!=nullptr,"native RGB palette entry");hud_rgb_entry[0]=static_cast<unsigned char>(n*71);hud_rgb_entry[1]=static_cast<unsigned char>(n*117);hud_rgb_entry[2]=static_cast<unsigned char>(n*173);}
        using HudDraw=int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*);
        using AlphaDraw=int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,int,int,void*);
        backend.display_image(target[0]);backend.display_image(target[2]);
        std::vector<unsigned> hud_detail(w*h);for(int n=0;n<w*h;++n)hud_detail[n]=0xff917a38u+unsigned(n%5);
        verify(gpu.upload(backend.display_image(target[0]),1,hud_detail.data(),hud_detail.size()),"independent full-color HUD background");
        auto expand_hud=[](unsigned word){unsigned b=word&31,g=(word>>5)&31,r=(word>>10)&31;
            return 0xff000000u|((b<<3)|(b>>2))|(((g<<3)|(g>>2))<<8)|(((r<<3)|(r>>2))<<16);};
        for(int slot:{20,21,22}){
            auto prior_uploads=gpu.stats().uploads;
            for(auto anchor:std::array<std::array<int,2>,4>{{{{0,0}},{{3,4}},{{33,24}},{{-5,-3}}}}){
                if(slot==22&&anchor[0]<0)continue;
                RECT blend_clip={2,3,51,39};
                for(int i:{0,2}){reinterpret_cast<Clip>(control[i]->vtable[13])(control[i],&blend_clip);reinterpret_cast<Clip>(target[i]->vtable[13])(target[i],&blend_clip);}
                if(anchor[0]==3){alpha_bytes[33]^=64;colors[17]^=0x3ff;
                    auto hud_rgb_entry=reinterpret_cast<unsigned char*(__thiscall*)(void*,unsigned)>(palette_table[8])(sprite_palette,17);hud_rgb_entry[1]^=16;}
                int background_index=slot==20?0:2;
                std::vector<unsigned> background_words(w*h),background_rgb(w*h),expected_full(w*h),actual_full(w*h);
                gpu.readback(backend.image(target[background_index]),background_words.data(),background_words.size());
                gpu.readback(backend.display_image(target[background_index]),background_rgb.data(),background_rgb.size());
                gpu.readback(backend.display_image(target[2]),expected_full.data(),expected_full.size());
                int native_result,gpu_result;
                if(slot==20){native_result=reinterpret_cast<HudDraw>(state.custom_renderer_jgl_blend_original[0])(&sprite,&alpha_sprite,control[0],control[2],anchor[0],anchor[1],sprite_palette);
                    gpu_result=reinterpret_cast<HudDraw>(sprite.vtable[20])(&sprite,&alpha_sprite,target[0],target[2],anchor[0],anchor[1],sprite_palette);}
                else {native_result=reinterpret_cast<AlphaDraw>(state.custom_renderer_jgl_blend_original[slot-20])(&sprite,&alpha_sprite,control[2],anchor[0],anchor[1],sprite_palette);
                    gpu_result=reinterpret_cast<AlphaDraw>(sprite.vtable[slot])(&sprite,&alpha_sprite,target[2],anchor[0],anchor[1],sprite_palette);}
                verify(native_result==gpu_result,"native alpha sprite return");compare(2);
                auto native_after=get(control[2],0,0);auto native_stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(control[2])+0x40);
                for(int y=0;y<16;++y)for(int x=0;x<16;++x){int dx=x+anchor[0],dy=y+anchor[1];unsigned weight=alpha_bytes[y*16+x];
                    if(dx<0||dy<0||dx>=w||dy>=h||(slot!=22&&(dx<blend_clip.left||dx>=blend_clip.right||dy<blend_clip.top||dy>=blend_clip.bottom))||indexed[y*16+x]==255||weight==255)continue;
                    unsigned at=dy*w+dx,native_hud_color=expand_hud(native_after[dy*native_stride+dx]),native_background=expand_hud(background_words[at]);
                    if(slot==22&&weight)++weight;
                    unsigned corrected=0xff000000u;for(unsigned channel=0;channel<3;++channel){int shift=channel*8;
                        int delta=int(background_rgb[at]>>shift&255)-int(native_background>>shift&255);
                        int value=int(native_hud_color>>shift&255)+delta*int(weight)/256;
                        corrected|=unsigned(std::max(0,std::min(255,value)))<<shift;}
                    expected_full[at]=corrected;
                }
                release(control[2],1);gpu.readback(backend.display_image(target[2]),actual_full.data(),actual_full.size());
                verify(actual_full==expected_full,"HUD preserves native UI colors and independent full-color map contribution");
                verify(backend.owns(target[0])&&backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"HUD blending keeps native/full-color background and destination resident");
            }
            verify(gpu.stats().uploads<=prior_uploads+2,"HUD anchors reuse source uploads; alpha/palette edits replace them");
        }
        for(int i:{0,2}){reinterpret_cast<Clip>(control[i]->vtable[13])(control[i],&full);reinterpret_cast<Clip>(target[i]->vtable[13])(target[i],&full);}
        alpha_sprite.bits=nullptr;reinterpret_cast<void(__thiscall*)(JGLSprite*)>(base_address+0x7ed0)(&alpha_sprite);
        sprite.stride=sprite.width=w;sprite.height=h;
        { // Actual Main_Screen_Form HUD pairs, clipped at all viewport edges.
            NativeUiAssets hud(graph,base_address);
            for(unsigned index=0;index<hud.pairs.size();++index)for(auto anchor:std::array<std::array<int,2>,4>{{{{0,0}},{{-17,-9}},{{w-12,h-8}},{{-300,0}}}}){
                both_fill(0,full,0x80003256u);both_fill(2,full,0x80001234u);
                auto expected_return=hud.draw(index,control[0],control[2],anchor[0],anchor[1],true);
                verify(hud.draw(index,target[0],target[2],anchor[0],anchor[1])==expected_return,"local HUD preserves native return");
                compare(2);verify(backend.owns(target[0])&&backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"local HUD stays resident");
            }
            if(!hud.pairs.empty())std::puts("PASS local HUD art: six panel/button pairs, native allocator/palettes, clipped edges, exact pixels and returns, no CPU fallback");
        }
        // The 16-bit scaled helper is a native no-op; retain the GPU canvas.
        // Its positive key still resolves/mutates through the destination palette.
        sprite.f28=0x80007c1f;sprite.bit_count=16;sprite.bits=source_bits;sprite.stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(ui)+0x40);
        auto scale=reinterpret_cast<int*>(base_address+0x6c0fc);int saved_scale=*scale;*scale=saved_scale*2;
        auto result=original_sprite(&sprite,control[2],3,4,palette);
        verify(reinterpret_cast<SpriteDraw>(sprite.vtable[17])(&sprite,target[2],3,4,palette)==result,"unsupported native sprite return");
        *scale=saved_scale;compare(2);verify(backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"native no-draw keeps resident destination");
        sprite.f28=255;draw_sprite(3,4,palette);draw_sprite(-3,2,palette);
        auto image_key=reinterpret_cast<unsigned*>(reinterpret_cast<char*>(ui)+0x4d0);auto saved_key=*image_key;*image_key=255;
        for(int x:{0,-3,7}){
            verify(reinterpret_cast<Draw>(original[33])(ui,control[2],x,2)==0,"palette-index image control");
            verify(reinterpret_cast<Draw>(ui->vtable[33])(ui,target[2],x,2)==0,"palette-index image GPU draw");
            compare(2);verify(*image_key==255&&backend.owns(target[2])&&backend.stats().readbacks==sprite_readbacks,"image key stays on temporary descriptor, no readback");
        }
        *image_key=saved_key;
        // Palette fills read the current table on every call, without treating
        // its pointer as a revision or applying sprite transparency to fills.
        verify(replace_storage(target[2])==0,"palette fill lifetime");both_fill(2,full,0x80000567);
        auto control_palette=reinterpret_cast<void**>(reinterpret_cast<char*>(control[2])+0x7c),target_palette=reinterpret_cast<void**>(reinterpret_cast<char*>(target[2])+0x7c);
        auto saved_control_palette=*control_palette,saved_target_palette=*target_palette;*control_palette=*target_palette=sprite_palette;
        for(unsigned color:{0u,17u,254u,255u,511u}){
            colors[color&255]^=0x7fff;both_fill(2,{3,4,17,19},color);compare(2);
            RECT panel={1,2,43,35};
            verify(original_tint(control[2],&panel,int(color),25)==0&&reinterpret_cast<Tint>(target[2]->vtable[18])(target[2],&panel,int(color),25)==0,"tint observes live native palette edits");compare(2);
            verify(original_line(control[2],1,2,37,29,int(color),1)==0&&reinterpret_cast<Line>(target[2]->vtable[25])(target[2],1,2,37,29,int(color),1)==0,"line observes live native palette edits");compare(2);
        }
        verify(backend.owns(target[2]),"live palette mutation retains GPU fill ownership");
        *control_palette=saved_control_palette;*target_palette=saved_target_palette;
        reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(base_address+0x3cf10)(sprite_palette,1);
        sprite.bits=nullptr;reinterpret_cast<void(__thiscall*)(JGLSprite*)>(base_address+0x7ed0)(&sprite);release(ui,1);
        verify(replace_storage(target[2])==0,"fresh palette destination");both_fill(2,full,0x80000567);
        for(unsigned color:{2u,255u,0x80012345u,0xff002345u}){both_fill(2,{3,4,17,19},color);compare(2);}
        verify(backend.owns(target[2]),"palette and packed native fills preserve GPU ownership");
        verify(replace_storage(target[2])==0,"fresh config-off destination");both_fill(2,full,0x80000567);
        RECT outside_source={-2,0,62,48};
        verify(copy(ui,control[2],&outside_source,&full)==0,"native clipped-source control");
        verify(reinterpret_cast<Copy>(ui->vtable[16])(ui,target[2],&outside_source,&full)==0,"queued adapter rejects unsupported source before skipping native draw");
        compare(2,true);verify(!backend.owns(target[2]),"unsupported source bounds restore CPU before native copy");
        verify(replace_storage(target[2])==0,"fresh config-off destination after source fallback");both_fill(2,full,0x80000567);
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
        verify(backend.owns(temporary),"null fill retains GPU ownership");
        std::vector<unsigned> cleared(w*h);gpu.readback(backend.image(temporary),cleared.data(),cleared.size());
        auto expected_clear=get(root,0,0);verify(expected_clear!=nullptr,"null fill oracle lease");
        for(int pixel=0;pixel<w*h;++pixel)verify(cleared[pixel]==expected_clear[pixel],"null fill exact native pixels");release(root,1);
        reinterpret_cast<Destroy>(temporary->vtable[0])(temporary,1);verify(temporary_id&&!backend.image(temporary),"native destruction retires GPU ownership");
#ifndef C3X_NATIVE_WORKER_TEST
        // The live recording repeatedly copies CPU-owned fullscreen canvases.
        // Keep an escaped pointer, including edits in the last row: validation
        // must inspect current words even when no lease event announces a write.
        for(int width:{2240,2239}){
            constexpr int height=1260;RECT area={0,0,width,height};
            auto source=create(graph,nullptr,1),destination=create(graph,nullptr,1);
            verify(init(source,width,height,16,1)==0,"fullscreen CPU source init");
            verify(fill(source,&area,int(0x80000123u))==0,"fullscreen CPU source fill");
            verify(reinterpret_cast<Init>(destination->vtable[1])(destination,width,height,16,1)==0,"fullscreen GPU destination init");
            auto held=get(source,0,0);verify(held!=nullptr,"fullscreen escaped pointer");release(source,1);
            auto stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(source)+0x40);
            auto transfer=[&]{verify(reinterpret_cast<Copy>(source->vtable[16])(source,destination,&area,&area)==0&&backend.owns(destination),"fullscreen CPU-to-GPU copy");};
            auto uploads=gpu.stats().uploads;transfer();verify(gpu.stats().uploads==uploads+1,"fullscreen initial upload");
            std::vector<unsigned> observed(std::size_t(width)*height);
            verify(gpu.readback(backend.image(destination),observed.data(),observed.size()),"fullscreen warm completion");
            LARGE_INTEGER frequency,began,ended;QueryPerformanceFrequency(&frequency);
            for(bool changed:{false,true}){
                uploads=gpu.stats().uploads;auto expanded=backend.stats().source_expanded_bytes;QueryPerformanceCounter(&began);
                for(unsigned n=0;n<96;++n){
                    if(changed&&n%12==0)held[(height-1)*stride+width-1]=static_cast<unsigned short>(0x200+n);
                    transfer();
                }
                verify(gpu.readback(backend.image(destination),observed.data(),observed.size()),"fullscreen measured completion");QueryPerformanceCounter(&ended);
                verify(gpu.stats().uploads==uploads+(changed?8:0),"fullscreen content changes alone upload");
                verify(backend.stats().source_expanded_bytes-expanded==std::uint64_t(changed?8:0)*width*height*4,"only changed sources allocate expanded upload pixels");
                for(int y=0;y<height;++y)for(int x=0;x<width;++x)
                    verify(observed[std::size_t(y)*width+x]==held[y*stride+x],"fullscreen strided source exact pixels");
                std::printf("PASS fullscreen CPU source width=%d changed=%u copies=96 uploads=%llu complete_ms=%.3f\n",width,unsigned(changed),gpu.stats().uploads-uploads,1000.*double(ended.QuadPart-began.QuadPart)/frequency.QuadPart);
            }
            reinterpret_cast<Destroy>(destination->vtable[0])(destination,1);
            reinterpret_cast<Destroy>(source->vtable[0])(source,1);
        }
#endif
        verify(replace_storage(target[2])==0,"new lifetime before detach");both_fill(2,full,0x80000567);
        reads=backend.stats().readbacks;set_custom_renderer_native_probe(nullptr);compare(2,true);
        verify(backend.stats().readbacks==reads+1&&state.custom_renderer_native_image==nullptr,"detach drains current pixels before removing hooks");
        for(auto image:target)reinterpret_cast<Destroy>(image->vtable[0])(image,1);
        for(auto image:control)reinterpret_cast<Destroy>(image->vtable[0])(image,1);
        reinterpret_cast<Destroy>(ui->vtable[0])(ui,1);reinterpret_cast<Destroy>(root->vtable[0])(root,1);
        #ifndef C3X_NATIVE_WORKER_TEST
        verify(gpu.stats().resident_bytes==w*h*8,"drain releases all images, only shared native/full-color overlap scratch remains");
#endif
        verify(same_owner(),"drawing, diagnostics and detach preserve the native graphics owner");
        std::puts("PASS JGL owner continuity: hook attachment, default UI bit depth, drawing, diagnostics and detach");
        auto stats=backend.stats();std::printf("PASS hooked native GPU adapter: 6 phases exact native pixels; translated=%llu source_checks=%llu uploads=%llu fallback_readbacks=%llu bytes=%llu; zero fallback readbacks in admitted GPU chain (explicit diagnostic/oracle reads excluded); retained-pointer edits, stretch, CPU access, reinit, config-off and detach pass\n",stats.translated,stats.source_checks,std::uint64_t(gpu.stats().uploads),stats.readbacks,stats.readback_bytes);
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
