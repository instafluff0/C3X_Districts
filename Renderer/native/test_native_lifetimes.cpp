#include <algorithm>
#define C3X_NATIVE_ADAPTER_TEST
#include "test_native_observation.cpp"
char const* tracking_candidate=nullptr;
HMODULE tracking_load(char const*){return LoadLibraryA(tracking_candidate);}
#define LoadLibraryA tracking_load
#include "build/native_tracking_bootstrap.h"
#undef LoadLibraryA

DWORD WINAPI escape_lifetime(void* object){RECT area={0,0,16,16};auto image=static_cast<JGL_Image*>(object);
    return DWORD(reinterpret_cast<Fill>(image->vtable[17])(image,&area,int(0x80001234u)));}
int main(int argc,char** argv){
    if(argc!=3)return 2;
    try {
        auto jgl=LoadLibraryA(argv[1]);verify(jgl!=nullptr,"load audited JGL");
        auto graph=reinterpret_cast<void*(__cdecl*)()>(GetProcAddress(jgl,"get_graphsy_object_ptr"))();auto table=*reinterpret_cast<void***>(graph);
        auto create=reinterpret_cast<Create>(table[31]);auto older=create(graph,nullptr,1);
        verify(reinterpret_cast<Init>(older->vtable[1])(older,16,16,16,1)==0,"pre-bootstrap native image");
        void* original[60];std::memcpy(original,older->vtable,sizeof(original));
        tracking_candidate=argv[2];state.current_config.enable_custom_rendering=false;
        start_custom_renderer_native_tracking();
        verify(state.custom_renderer_native_module&&state.custom_renderer_native_lifetime&&state.custom_renderer_native_probe_active,"actual startup bootstrap attaches before map/configuration");
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
        using DefaultFont=int(__thiscall*)(JGL_Image*);using TextRgb=int(__thiscall*)(JGL_Image*,int,int,int);
        verify(reinterpret_cast<DefaultFont>(images[0]->vtable[43])(images[0])==0,"native default font metadata");
        verify(reinterpret_cast<TextRgb>(images[0]->vtable[45])(images[0],17,83,149)==0,"native RGB text state");
        struct Font {char pad[0x18];HFONT handle;} font={};font.handle=static_cast<HFONT>(GetStockObject(SYSTEM_FONT));void* native_font=&font;
        verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,void*)>(images[0]->vtable[42])(images[0],&native_font)==0,"native explicit font metadata");
        verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,int)>(images[0]->vtable[44])(images[0],7)==0,"native indexed text state");
        verify(query(images[0]),"font/color state does not expose pixels");
        verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,int,int,char const*,int)>(images[0]->vtable[46])(images[0],1,1,"Hi",2)==0&&query(images[0]),"scoped native TextOut preserves lifetime evidence");
        auto dc=reinterpret_cast<HDC(__thiscall*)(JGL_Image*)>(images[0]->vtable[10])(images[0]);verify(dc!=nullptr,"public native DC");
        reinterpret_cast<Release>(images[0]->vtable[11])(images[0],1);verify(!query(images[0]),"DC release cannot undo a CPU escape");
        verify(reinterpret_cast<Init>(images[0]->vtable[1])(images[0],16,16,16,1)==0&&query(images[0]),"successful reinit starts a new lifetime");
        auto pixel=reinterpret_cast<Get>(images[0]->vtable[3])(images[0],0,0);verify(pixel!=nullptr,"public native pixel");
        reinterpret_cast<Release>(images[0]->vtable[9])(images[0],1);verify(!query(images[0]),"raw pointer escape remains recorded");
        verify(reinterpret_cast<Init>(images[0]->vtable[1])(images[0],16,16,16,1)==0&&query(images[0]),"reinit before scenario transition");
        set_custom_renderer_native_probe(nullptr);state.custom_renderer_native_observe=nullptr;
        verify(state.custom_renderer_native_probe_active&&query(images[0]),"scene detach preserves process lifetime observation");
        verify(reinterpret_cast<Fill>(images[0]->vtable[17])(images[0],&area,int(0x80005678u))==0&&query(images[0]),"native fill remains native after scene detach");
        auto thread=CreateThread(nullptr,0,escape_lifetime,images[1],0,nullptr);verify(thread!=nullptr,"foreign native caller");
        verify(WaitForSingleObject(thread,10000)==WAIT_OBJECT_0,"foreign draw completion");CloseHandle(thread);verify(!query(images[1]),"foreign drawing invalidates exclusive ownership");
        for(auto image:images){reinterpret_cast<Destroy>(image->vtable[0])(image,1);verify(!query(image),"destroyed lifetime retired");}
        // Exhaustion cannot evict proof for an existing image or invent it for
        // the unrecorded extra object. Registry records never dereference IDs.
        for(unsigned i=1;i<=1024;++i)verify(state.custom_renderer_native_lifetime(C3X_NATIVE_INIT,reinterpret_cast<void*>(std::uintptr_t(i)),0)==1,"bounded registry admission");
        verify(state.custom_renderer_native_lifetime(C3X_NATIVE_INIT,reinterpret_cast<void*>(std::uintptr_t(1025)),0)==0&&query(reinterpret_cast<void*>(std::uintptr_t(1))),"registry overflow rejects instead of replacing live evidence");
        for(unsigned i=1;i<=1024;++i)state.custom_renderer_native_lifetime(C3X_NATIVE_DESTROY,reinterpret_cast<void*>(std::uintptr_t(i)),0);
        auto module=state.custom_renderer_native_module;state.custom_renderer_native_lifetime=nullptr;set_custom_renderer_native_probe(nullptr);
        verify(!state.custom_renderer_native_probe_active&&std::equal(original,original+60,older->vtable),"all native slots restored at process-owner shutdown");
        reinterpret_cast<Destroy>(older->vtable[0])(older,1);FreeLibrary(module);FreeLibrary(jgl);
        std::puts("PASS native startup lifetimes: actual bootstrap/hooks, config-off native pixels, private leases, font state, persistent scene boundary, CPU/foreign escapes, reinit, destroy and bounded admission");return 0;
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL native startup lifetimes: %s\n",e.what());return 1;}
}
