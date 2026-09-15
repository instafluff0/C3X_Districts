#include <algorithm>
#define C3X_NATIVE_ADAPTER_TEST
#include "test_native_observation.cpp"
#include "test_native_bootstrap.h"

DWORD WINAPI escape_lifetime(void* object){RECT area={0,0,16,16};auto image=static_cast<JGL_Image*>(object);
    return DWORD(reinterpret_cast<Fill>(image->vtable[17])(image,&area,int(0x80001234u)));}
int deny_native_access(int,void*,void*,void const*,void const*,unsigned){return -1;}
int main(int argc,char** argv){
    if(argc!=3)return 2;
    try {
        tracking_candidate=argv[2];state.current_config.enable_custom_rendering=false;
        verify(!GetModuleHandleA("jgl.dll"),"startup precedes DLL loading");
        start_custom_renderer_native_tracking();
        verify(!state.custom_renderer_native_module,"early bootstrap cannot attach before DLL exists");
        verify(!patch_load_jgl_lib(nullptr)&&!state.custom_renderer_native_module,"failed native load does not attach");
        auto graph=patch_load_jgl_lib(argv[1]);auto jgl=bootstrap_jgl;
        verify(graph&&state.custom_renderer_native_module&&state.custom_renderer_native_lifetime&&state.custom_renderer_native_probe_active,"post-load bootstrap attaches before the first native image");
        auto table=*static_cast<void***>(graph);auto create=reinterpret_cast<Create>(table[31]);
        void* original[60];std::memcpy(original,state.custom_renderer_jgl_original,sizeof(original));
        auto older=create(graph,nullptr,1);
        verify(reinterpret_cast<Init>(original[1])(older,16,16,16,1)==0,"unobserved init cannot claim lifetime");
        auto query=[&](void* image){return state.custom_renderer_native_lifetime(C3X_NATIVE_MAP,image,0)!=0;};
        verify(!query(older),"preexisting image is never admitted by a later draw");
        JGL_Image* images[2];RECT area={0,0,16,16};
        for(auto& image:images){image=create(graph,nullptr,1);verify(reinterpret_cast<Init>(image->vtable[1])(image,16,16,16,1)==0,"tracked native init");verify(query(image),"fresh native lifetime recorded while rendering is disabled");}
        verify(reinterpret_cast<Fill>(images[0]->vtable[17])(images[0],&area,int(0x80001234u))==0,"native fill");
        verify(reinterpret_cast<Copy>(images[0]->vtable[16])(images[0],images[1],&area,&area)==0,"native copy");
        GdiFlush();auto native_pixels=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[1]);
        verify(native_pixels&&std::all_of(native_pixels,native_pixels+256,[](unsigned short c){return c==0x1234;}),"config-off native copied pixels exact");
        reinterpret_cast<Release>(original[9])(images[1],1);
        verify(query(images[0])&&query(images[1]),"audited private fill/copy leases preserve lifetime evidence");
        auto metadata_palette=reinterpret_cast<void*(__thiscall*)(void*,void*)>(table[30])(graph,nullptr);
        using Palette=void(__thiscall*)(JGL_Image*,void*);
        reinterpret_cast<Palette>(images[0]->vtable[59])(images[0],metadata_palette);
        verify(query(images[0]),"16-bit palette metadata preserves lifetime evidence");
        using DefaultFont=int(__thiscall*)(JGL_Image*);using TextRgb=int(__thiscall*)(JGL_Image*,int,int,int);
        verify(reinterpret_cast<DefaultFont>(images[0]->vtable[43])(images[0])==0,"native default font metadata");
        verify(reinterpret_cast<TextRgb>(images[0]->vtable[45])(images[0],17,83,149)==0,"native RGB text state");
        struct Font {char pad[0x18];HFONT handle;} font={};font.handle=static_cast<HFONT>(GetStockObject(SYSTEM_FONT));void* native_font=&font;
        verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,void*)>(images[0]->vtable[42])(images[0],&native_font)==0,"native explicit font metadata");
        verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,int)>(images[0]->vtable[44])(images[0],7)==0,"native indexed text state");
        verify(query(images[0]),"font/color state does not expose pixels");
        verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,int,int,char const*,int)>(images[0]->vtable[46])(images[0],1,1,"Hi",2)==0&&query(images[0]),"scoped native TextOut preserves lifetime evidence");
        // HUD alpha calls before GPU admission are private native drawing too.
        // Their internal leases must not permanently disqualify these surfaces.
        JGLSprite hud_color={},hud_alpha={};auto jgl_base=reinterpret_cast<char*>(jgl);
        unsigned char hud_indices[16]={},hud_weights[16]={};
        for(auto sprite:{&hud_color,&hud_alpha}){reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(jgl_base+0x7e80)(sprite,nullptr);
            sprite->bit_count=8;sprite->width=sprite->stride=sprite->height=4;}
        hud_color.bits=hud_indices;hud_alpha.bits=hud_weights;
        auto hud_palette=reinterpret_cast<void*(__thiscall*)(void*,void*)>(table[30])(graph,nullptr);verify(hud_palette!=nullptr,"startup HUD palette");
        verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*)>(hud_color.vtable[20])(&hud_color,&hud_alpha,images[0],images[1],0,0,hud_palette)==0,"config-off HUD background blend");
        verify(query(images[0])&&query(images[1])&&state.custom_renderer_native_operation==0&&images[0]->Bits_Data_Links==0&&images[1]->Bits_Data_Links==0,"HUD source and destination private leases preserve lifetime and scope");
        for(int slot:{21,22}){
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,int,int,void*)>(hud_color.vtable[slot])(&hud_color,&hud_alpha,images[1],0,0,hud_palette)==0,"config-off HUD destination blend");
            verify(query(images[1])&&state.custom_renderer_native_operation==0&&images[1]->Bits_Data_Links==0,"HUD private destination lease preserves lifetime and scope");
            auto held=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[1]);
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,int,int,void*)>(hud_color.vtable[slot])(&hud_color,&hud_alpha,images[1],0,0,hud_palette)==0,"HUD destination blend with caller lease");
            verify(images[1]->Bits_Data_Links==1&&images[1]->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(held)),"HUD destination blend preserves existing caller lease");
            reinterpret_cast<Release>(original[9])(images[1],1);
        }
        for(bool alias:{false,true})for(int x:{0,32}){
            auto target=images[alias?0:1];
            auto held_background=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[0]);
            auto held_destination=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(target);
            int before_background=images[0]->Bits_Data_Links,before_destination=target->Bits_Data_Links;
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*)>(hud_color.vtable[20])(&hud_color,&hud_alpha,images[0],target,x,0,hud_palette)==0,"HUD blend with caller-held leases");
            verify(images[0]->Bits_Data_Links==before_background&&target->Bits_Data_Links==before_destination&&
                images[0]->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(held_background))&&target->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(held_destination)),"HUD exact lease preservation for separate/aliased and clipped native calls");
            reinterpret_cast<Release>(original[9])(images[0],1);reinterpret_cast<Release>(original[9])(target,1);
        }
        using Tint=int(__thiscall*)(JGL_Image*,RECT*,int,int);
        using Lookup=int(__thiscall*)(JGL_Image*,RECT*,JGL_Image*,int,void*);
        std::vector<unsigned short> lookup_table(524288);for(unsigned n=0;n<lookup_table.size();++n)lookup_table[n]=static_cast<unsigned short>(n&32767);
        for(auto background:{images[0],images[1]}){
            auto bits=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[0]);
            auto links=images[0]->Bits_Data_Links;
            verify(reinterpret_cast<Lookup>(images[0]->vtable[21])(images[0],&area,background,40,lookup_table.data())==0,"native lookup private borrows");
            verify(query(images[0])&&query(background)&&images[0]->Bits_Data_Links==links&&images[0]->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(bits)),"lookup preserves entry pointer lease");
            reinterpret_cast<Release>(original[9])(images[0],1);
        }
        lookup_table.resize(31*32768);
        for(auto background:{images[0],images[1]}){
            auto held=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[1]);
            auto links=images[1]->Bits_Data_Links;
            auto flc_result=reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*,void*)>(hud_color.vtable[35])(&hud_color,background,images[1],0,0,lookup_table.data(),hud_palette);
            if(flc_result)std::fprintf(stderr,"FLC_STARTUP result=%d format=%d bits=%d source_present=%d background_links=%d destination_links=%d\n",flc_result,*reinterpret_cast<int*>(reinterpret_cast<char*>(background)+0x28),background->BitCount,hud_color.bits!=nullptr,background->Bits_Data_Links,images[1]->Bits_Data_Links);
            verify(flc_result==0,"native FLC lookup private borrows");
            verify(query(images[1])&&query(background)&&images[1]->Bits_Data_Links==links&&images[1]->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(held)),"FLC lookup preserves caller entry lease");
            reinterpret_cast<Release>(original[9])(images[1],1);
        }
        auto native_scales=reinterpret_cast<int*>(jgl_base+0x6c0fc);int saved_lookup_scales[3]={native_scales[0],native_scales[1],native_scales[2]};
        native_scales[0]=native_scales[1]=1;native_scales[2]=2;
        for(auto background:{images[0],images[1]})for(int x:{-2,0,32}){
            auto held=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[1]);
            auto links=images[1]->Bits_Data_Links;
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,JGL_Image*,int,int,int,int,int,void*,void*)>(hud_color.vtable[34])(&hud_color,background,images[1],x,0,1,1,2,lookup_table.data(),hud_palette)==0,"native scaled FLC with caller lease");
            verify(query(images[1])&&query(background)&&images[1]->Bits_Data_Links==links&&images[1]->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(held)),"scaled FLC preserves caller entry lease");
            reinterpret_cast<Release>(original[9])(images[1],1);
        }
        for(unsigned i=0;i<3;++i)native_scales[i]=saved_lookup_scales[i];
        std::vector<unsigned short> map_shadow_table(4*32768);for(unsigned n=0;n<map_shadow_table.size();++n)map_shadow_table[n]=static_cast<unsigned short>(n&32767);
        for(int slot:{23,29,31})for(int x:{-2,0,32}){
            std::fill(std::begin(hud_indices),std::end(hud_indices),static_cast<unsigned char>(slot==31?248:17));
            auto held=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[1]);auto links=images[1]->Bits_Data_Links;
            int result=slot==23?reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*)>(hud_color.vtable[23])(&hud_color,images[1],x,0,hud_palette):
                slot==29?reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,int,void*)>(hud_color.vtable[29])(&hud_color,images[1],x,0,17,hud_palette):
                reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*,void*)>(hud_color.vtable[31])(&hud_color,images[1],x,0,map_shadow_table.data(),hud_palette);
            verify(result==0&&query(images[1])&&images[1]->Bits_Data_Links==links&&images[1]->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(held)),"single-key/mask/shadow calls preserve native caller leases");
            reinterpret_cast<Release>(original[9])(images[1],1);
        }
        for(float opacity:{0.f,0.03125f,0.5f,1.f})for(int x:{-2,0,32}){
            auto held=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[1]);auto links=images[1]->Bits_Data_Links;
            auto result=reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,float,void*,int)>(hud_color.vtable[37])(&hud_color,images[1],x,0,opacity,hud_palette,0);
            verify(result==0&&query(images[1])&&images[1]->Bits_Data_Links==links&&images[1]->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(held)),"opacity transitions preserve native caller leases");
            reinterpret_cast<Release>(original[9])(images[1],1);
        }
        for(auto sprite:{&hud_color,&hud_alpha}){sprite->bits=nullptr;reinterpret_cast<void(__thiscall*)(JGLSprite*)>(jgl_base+0x7ed0)(sprite);}
        reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(jgl_base+0x3cf10)(hud_palette,1);
        using Line=int(__thiscall*)(JGL_Image*,int,int,int,int,int,int);
        verify(reinterpret_cast<Tint>(images[0]->vtable[18])(images[0],&area,int(0x80000000u),50)==0&&query(images[0])&&images[0]->Bits_Data_Links==0,"native label panel preserves private lifetime and leases");
        for(RECT segment:{RECT{1,2,12,9},RECT{12,9,1,2},RECT{-20,-20,-2,-1},RECT{2,2,2,2}}){
            auto held=reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(original[4])(images[0]);
            verify(reinterpret_cast<Line>(images[0]->vtable[25])(images[0],segment.left,segment.top,segment.right,segment.bottom,int(0x80001234u),1)==0,"native private line");
            verify(query(images[0])&&images[0]->Bits_Data_Links==1&&images[0]->Current_Bits_Data==int(reinterpret_cast<std::uintptr_t>(held)),"native line preserves caller entry lease");
            reinterpret_cast<Release>(original[9])(images[0],1);
        }
        auto dc=reinterpret_cast<HDC(__thiscall*)(JGL_Image*)>(images[0]->vtable[10])(images[0]);verify(dc!=nullptr,"public native DC");
        reinterpret_cast<Release>(images[0]->vtable[11])(images[0],1);verify(!query(images[0]),"DC release cannot undo a CPU escape");
        verify(reinterpret_cast<Init>(images[0]->vtable[1])(images[0],16,16,0,1)==0&&!query(images[0]),"no-op init cannot erase a CPU escape");
        verify(reinterpret_cast<Init>(images[0]->vtable[1])(images[0],17,16,16,1)==0&&query(images[0]),"storage replacement starts a new lifetime");
        auto pixel=reinterpret_cast<Get>(images[0]->vtable[3])(images[0],0,0);verify(pixel!=nullptr,"public native pixel");
        reinterpret_cast<Release>(images[0]->vtable[9])(images[0],1);verify(!query(images[0]),"raw pointer escape remains recorded");
        verify(reinterpret_cast<Init>(images[0]->vtable[1])(images[0],16,16,16,1)==0&&query(images[0]),"reinit before scenario transition");
        set_custom_renderer_native_probe(nullptr);state.custom_renderer_native_observe=nullptr;
        verify(state.custom_renderer_native_probe_active&&query(images[0]),"scene detach preserves process lifetime observation");
        verify(reinterpret_cast<Fill>(images[0]->vtable[17])(images[0],&area,int(0x80005678u))==0&&query(images[0]),"native fill remains native after scene detach");
        auto thread=CreateThread(nullptr,0,escape_lifetime,images[1],0,nullptr);verify(thread!=nullptr,"foreign native caller");
        verify(WaitForSingleObject(thread,10000)==WAIT_OBJECT_0,"foreign draw completion");CloseHandle(thread);verify(!query(images[1]),"foreign drawing invalidates exclusive ownership");
        for(auto image:images){reinterpret_cast<Destroy>(image->vtable[0])(image,1);verify(!query(image),"destroyed lifetime retired");}
        auto denied=create(graph,nullptr,1);verify(reinterpret_cast<Init>(denied->vtable[1])(denied,16,16,16,1)==0,"barrier-failure native image");
        state.custom_renderer_native_image=deny_native_access;
        verify(!reinterpret_cast<Get>(denied->vtable[3])(denied,0,0),"failed GPU barrier cannot expose a stale pixel pointer");
        verify(!reinterpret_cast<unsigned short*(__thiscall*)(JGL_Image*)>(denied->vtable[4])(denied),"failed GPU barrier cannot expose stale bits");
        verify(!reinterpret_cast<HDC(__thiscall*)(JGL_Image*)>(denied->vtable[10])(denied),"failed GPU barrier cannot expose a stale native DC");
        verify(reinterpret_cast<Init>(denied->vtable[1])(denied,32,32,16,1)==-1,"failed drain defers native reinitialization");
        verify(reinterpret_cast<Destroy>(denied->vtable[0])(denied,1)==denied,"failed drain defers destruction while owner retains image identity");
        state.custom_renderer_native_image=nullptr;reinterpret_cast<Destroy>(denied->vtable[0])(denied,1);
        // Exhaustion cannot evict proof for an existing image or invent it for
        // the unrecorded extra object. Registry records never dereference IDs.
        for(unsigned i=1;i<=1024;++i)verify(state.custom_renderer_native_lifetime(C3X_NATIVE_INIT,reinterpret_cast<void*>(std::uintptr_t(i)),0)==1,"bounded registry admission");
        verify(state.custom_renderer_native_lifetime(C3X_NATIVE_INIT,reinterpret_cast<void*>(std::uintptr_t(1025)),0)==0&&query(reinterpret_cast<void*>(std::uintptr_t(1))),"registry overflow rejects instead of replacing live evidence");
        for(unsigned i=1;i<=1024;++i)state.custom_renderer_native_lifetime(C3X_NATIVE_DESTROY,reinterpret_cast<void*>(std::uintptr_t(i)),0);
        reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(reinterpret_cast<char*>(jgl)+0x3cf10)(metadata_palette,1);
        auto module=state.custom_renderer_native_module;
        reinterpret_cast<Destroy>(older->vtable[0])(older,1);
        // Keep one test reference so native table restoration can be inspected.
        auto retained_jgl=LoadLibraryA(argv[1]);
        state.custom_renderer_native_image=deny_native_access;patch_unload_jgl_lib();
        verify(state.custom_renderer_native_module==module&&bootstrap_graph==graph,"failed barrier defers native DLL unload");
        state.custom_renderer_native_image=nullptr;patch_unload_jgl_lib();
        verify(!state.custom_renderer_native_module&&!state.custom_renderer_native_lifetime&&!state.custom_renderer_native_probe_active&&
            std::equal(original,original+60,reinterpret_cast<void**>(reinterpret_cast<char*>(retained_jgl)+0x68238)),"native unload restores slots and clears tracking owner");
        FreeLibrary(retained_jgl);
        graph=patch_load_jgl_lib(argv[1]);
        verify(graph&&state.custom_renderer_native_lifetime,"tracking reattaches after native DLL reload");
        verify(!state.custom_renderer_native_lifetime(C3X_NATIVE_MAP,reinterpret_cast<void*>(1),0),"previous epoch evidence cannot survive reload");
        patch_unload_jgl_lib();
        std::puts("PASS native startup lifetimes: actual bootstrap/hooks, config-off native pixels, private leases, font state, persistent scene boundary, CPU/foreign escapes, reinit, destroy and bounded admission");return 0;
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL native startup lifetimes: %s\n",e.what());return 1;}
}
