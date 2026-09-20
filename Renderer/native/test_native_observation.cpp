// Executes the actual injected hook block against the installed, hash-pinned JGL.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include "native_observation.h"
#define __ 0
struct JGL_Image {void** vtable;char pad[0x20];int BitCount;char dimensions[0x1c];RECT Clip_Rect,Image_Rect;char rest[0x4c8-0x64];int Bits_Data_Links,Current_Bits_Data;};
static_assert(offsetof(JGL_Image,BitCount)==0x24 && offsetof(JGL_Image,Image_Rect)==0x54 && offsetof(JGL_Image,Bits_Data_Links)==0x4c8);
struct JGLSprite {void** vtable;int a,b,c,d;void* bits;int f18,f1c,bit_count,f24,f28,stride,width,height;CRITICAL_SECTION lock;};
struct PCX_Image {struct {JGL_Image* Image;} JGL;};
struct OpenGLRenderer;
enum {LDO_NEVER,LDO_WINE,LDO_ALWAYS,IS_OK};
struct State {
    char const* mod_rel_dir=".";
    bool custom_renderer_modal=false,paused_for_popup=false;int saved_tile_count=-1;
    struct {bool enable_custom_rendering=true;int draw_lines_using_gdi_plus=LDO_NEVER;} current_config;
    bool running_on_wine=false;unsigned ogl_color=0xffffffff;int ogl_line_width=1;bool ogl_line_stipple_enabled=false;
    struct {int init_state=IS_OK;void* gp_graphics=nullptr;
        int (__stdcall *CreateFromHDC)(HDC,void**)=nullptr;
        int (__stdcall *DeleteGraphics)(void*)=nullptr;
        int (__stdcall *SetSmoothingMode)(void*,int)=nullptr;
        int (__stdcall *SetPenDashStyle)(void*,int)=nullptr;
        int (__stdcall *CreatePen1)(unsigned,float,int,void**)=nullptr;
        int (__stdcall *DeletePen)(void*)=nullptr;
        int (__stdcall *DrawLineI)(void*,void*,int,int,int,int)=nullptr;
    } gdi_plus;
    HMODULE kernel32=GetModuleHandleA("kernel32.dll");
#include "build/native_probe_state.h"
};
State state={};State* is=&state;
struct {bool is_now_loading_game=false;} main_screen_fixture;auto p_main_screen_form=&main_screen_fixture;
unsigned player_bits=1;unsigned* p_player_bits=&player_bits;
auto p_GetModuleHandleA=&GetModuleHandleA;auto p_GetProcAddress=&GetProcAddress;
PCX_Image screen;PCX_Image* screen_canvas=&screen;
#define p_jgl_screen_canvas screen_canvas
std::vector<c3x_renderer_native_observation> events;
c3x_native_observation::Capture capture;
std::vector<std::string> lines;
void log_line(char const* line){lines.emplace_back(line);}
c3x_renderer_native_observe_fn staged_observe=nullptr;
int observe(c3x_renderer_native_observation const* event){events.push_back(*event);
    int external=staged_observe?staged_observe(event):1;
    return capture.observe(event)&&external;
}
void verify(bool result,char const* message){if(!result)throw std::runtime_error(message);}
bool native_called=false;
void __cdecl native_present(RECT*){native_called=true;}
auto present_fn=&native_present;
#define JGL_present_screen present_fn
// The camera bridge itself is executed by test_native_view_identity.py.
void settle_custom_renderer_navigation(int){}
#include "build/native_probe_hooks.h"
using Create=JGL_Image*(__thiscall*)(void*,void*,int);
using Init=int(__thiscall*)(JGL_Image*,int,int,int,int);
using Fill=int(__thiscall*)(JGL_Image*,RECT*,int);
using Copy=int(__thiscall*)(JGL_Image*,JGL_Image*,RECT*,RECT*);
using Clip=int(__thiscall*)(JGL_Image*,RECT*);
using Get=std::uint16_t*(__thiscall*)(JGL_Image*,int,int);
using Release=void(__thiscall*)(JGL_Image*,int);
using Destroy=void*(__thiscall*)(JGL_Image*,unsigned);
void check_pixels(JGL_Image* image){
    for(int y=0;y<48;++y)for(int x=0;x<64;++x){auto p=reinterpret_cast<Get>(image->vtable[7])(image,x,y);
        verify(p!=nullptr,"pixel lease");auto value=*p;reinterpret_cast<Release>(image->vtable[9])(image,1);
        verify(value==(x>=7&&x<40&&y>=5&&y<33?0x03e0:0),"exact native pixels");}
}
DWORD WINAPI foreign(void* object){RECT full={0,0,64,48};auto image=static_cast<JGL_Image*>(object);
    return DWORD(reinterpret_cast<Fill>(image->vtable[17])(image,&full,int(0x80000000u)));}
#ifndef C3X_NATIVE_ADAPTER_TEST
int main(int argc,char** argv){
    if(argc!=2&&argc!=3)return 2;
    try {
        HMODULE staged=nullptr;
        if(argc==3){staged=LoadLibraryA(argv[2]);verify(staged!=nullptr,"load staged observer DLL");
            staged_observe=reinterpret_cast<c3x_renderer_native_observe_fn>(GetProcAddress(staged,"c3x_renderer_native_observe"));verify(staged_observe!=nullptr,"staged observer export");}
        capture.write=log_line;
        verify(!c3x_native_observation::verified_module(GetModuleHandleA("kernel32.dll")),"unrecognized module rejected");
        HMODULE module=LoadLibraryA(argv[1]);verify(module!=nullptr,"load JGL");
        verify(c3x_native_observation::verified_module(module),"audited JGL hash");
        auto factory=reinterpret_cast<void*(__cdecl*)()>(GetProcAddress(module,"get_graphsy_object_ptr"));
        void* graph=factory();auto gt=*reinterpret_cast<void***>(graph);
        auto create=reinterpret_cast<Create>(gt[31]);auto a=create(graph,nullptr,1),b=create(graph,nullptr,1);
        void* original[60];std::memcpy(original,a->vtable,sizeof original);
        auto sprite_table=reinterpret_cast<void**>(reinterpret_cast<char*>(module)+0x68440);void* original_sprite=sprite_table[17];
        void* original_blends[3]={sprite_table[20],sprite_table[21],sprite_table[22]};
        verify(reinterpret_cast<Init>(a->vtable[1])(a,64,48,16,1)==0,"init A");
        verify(reinterpret_cast<Init>(b->vtable[1])(b,64,48,16,1)==0,"init B");
        // An unsupported root must never cause writes, even with an observer installed.
        set_custom_renderer_native_probe(a);verify(!state.custom_renderer_native_probe_active,"missing observer preserves native path");
        state.custom_renderer_native_observe=observe;
        present_fn=nullptr;set_custom_renderer_native_probe(a);verify(!state.custom_renderer_native_probe_active,"missing executable capability preserves native path");present_fn=native_present;
        auto root_table=a->vtable;void* fake[60];std::memcpy(fake,original,sizeof fake);a->vtable=fake;
        set_custom_renderer_native_probe(a);verify(!state.custom_renderer_native_probe_active,"unknown table rejected");a->vtable=root_table;
        set_custom_renderer_native_probe(a);verify(state.custom_renderer_native_probe_active,"attach actual injected hooks");
        verify(capture.images.at(a).roles==1,"map root identity");
        RECT full={0,0,64,48},clip={7,5,40,33};
        verify(reinterpret_cast<Fill>(a->vtable[17])(a,&full,int(0x80000000u))==0,"clear A");
        verify(reinterpret_cast<Clip>(a->vtable[13])(a,&clip)==0,"clip A");events.clear();
        verify(reinterpret_cast<Fill>(a->vtable[17])(a,&full,int(0x800003e0u))==0,"clipped fill");
        unsigned pixels=0,dc=0;for(auto const& e:events){if(e.operation==C3X_NATIVE_PIXEL){++pixels;verify(e.context==C3X_NATIVE_FILL,"fill access context");}if(e.operation==C3X_NATIVE_DC)++dc;}
        verify(pixels==1&&dc==0,"one underlying pixel acquire; aliases not double counted");check_pixels(a);
        verify(reinterpret_cast<Clip>(a->vtable[13])(a,&full)==0,"unclip A");events.clear();
        verify(reinterpret_cast<Copy>(a->vtable[16])(a,b,&full,&full)==0,"copy A to B");GdiFlush();
        dc=0;for(auto const& e:events)if(e.operation==C3X_NATIVE_DC){++dc;verify(e.context==C3X_NATIVE_COPY,"copy access context");}
        verify(dc==2 && capture.edge_count==1,"copy source and destination ownership");check_pixels(b);
        // Image draw uses its internal sprite helper; the separate sprite call tests the virtual hook.
        verify(reinterpret_cast<Fill>(b->vtable[17])(b,&full,int(0x80000000u))==0,"clear destination before image draw");
        events.clear();using Draw=int(__thiscall*)(JGL_Image*,JGL_Image*,int,int);
        verify(reinterpret_cast<Draw>(a->vtable[33])(a,b,0,0)==0,"image draw native sprite path");
        bool image_access=false;for(auto const& e:events)if((e.operation==C3X_NATIVE_PIXEL||e.operation==C3X_NATIVE_BITS)&&e.context==C3X_NATIVE_IMAGE_DRAW)image_access=true;
        verify(image_access,"direct native sprite helper stays attributed to image draw");check_pixels(b);
        JGLSprite sprite={};auto base=reinterpret_cast<char*>(module);
        reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(base+0x7e80)(&sprite,nullptr);
        sprite.d=*reinterpret_cast<int*>(reinterpret_cast<char*>(a)+0x7c);
        sprite.f28=*reinterpret_cast<int*>(reinterpret_cast<char*>(a)+0x4d0);
        sprite.bits=reinterpret_cast<Get>(a->vtable[7])(a,0,0);sprite.bit_count=16;sprite.stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(a)+0x40);sprite.width=64;sprite.height=48;
        verify(reinterpret_cast<Fill>(b->vtable[17])(b,&full,int(0x80000000u))==0,"clear destination before sprite draw");
        events.clear();using SpriteDraw=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*);
        verify(reinterpret_cast<SpriteDraw>(sprite.vtable[17])(&sprite,b,0,0,reinterpret_cast<void*(__thiscall*)(JGL_Image*)>(a->vtable[58])(a))==0,"actual sprite draw");
        bool seen_sprite=false;for(auto const& e:events)if(e.operation==C3X_NATIVE_SPRITE)seen_sprite=true;
        verify(seen_sprite,"sprite method observed");check_pixels(b);
        sprite.bits=nullptr;reinterpret_cast<void(__thiscall*)(JGLSprite*)>(base+0x7ed0)(&sprite);
        reinterpret_cast<Release>(a->vtable[9])(a,1);
        auto id=capture.images.at(b).id;verify(reinterpret_cast<Init>(b->vtable[1])(b,64,48,16,1)==0,"reinitialize image");
        verify(capture.images.at(b).id!=id,"new content lifetime on reinit");
        state.current_config.enable_custom_rendering=false;events.clear();foreign(a);verify(events.empty(),"config off passthrough");state.current_config.enable_custom_rendering=true;
        events.clear();HANDLE thread=CreateThread(nullptr,0,foreign,a,0,nullptr);verify(thread!=nullptr,"foreign thread");WaitForSingleObject(thread,INFINITE);CloseHandle(thread);
        verify(events.empty(),"no foreign-thread observer calls");
        verify(lines.empty(),"no logging during native drawing or image destruction");
        screen.JGL.Image=b;events.clear();patch_JGL_present_screen(&full);
        verify(!lines.empty(),"buffered diagnostics flush after native transfer");
        verify(lines.front().find("surface="+std::to_string(id)+" roles=0 copy=")!=std::string::npos,"retired image access counts survive until publication");
        verify(native_called&&events.size()==2&&events.front().operation==C3X_NATIVE_SCREEN&&events.back().operation==C3X_NATIVE_PRESENT,"native final transfer order");
        verify(capture.images.at(b).roles==2,"screen identity");
        reinterpret_cast<Destroy>(b->vtable[0])(b,1);verify(!capture.images.count(b),"destruction retires pointer");
        set_custom_renderer_native_probe(nullptr);
        verify(!state.custom_renderer_native_probe_active&&!std::memcmp(a->vtable,original,sizeof original)&&sprite_table[17]==original_sprite&&std::equal(original_blends,original_blends+3,sprite_table+20),"restore all slots");
        set_custom_renderer_native_probe(a);verify(state.custom_renderer_native_probe_active,"reattach");
        screen.JGL.Image=a;capture.presents=8191;patch_JGL_present_screen(&full);
        verify(capture.ended&&!state.custom_renderer_native_probe_active&&!std::memcmp(a->vtable,original,sizeof original),"budget exhaustion detaches hooks");
        reinterpret_cast<Destroy>(a->vtable[0])(a,1);reinterpret_cast<void(__thiscall*)(void*,unsigned)>(gt[0])(graph,1);FreeLibrary(module);
        // Collector exhaustion is explicit; no false complete-coverage assertion.
        c3x_native_observation::Capture bounded;bounded.owner=GetCurrentThreadId();bounded.write=log_line;
        c3x_renderer_native_observation event={};event.struct_size=sizeof event;event.operation=C3X_NATIVE_MAP;
        for(unsigned n=1;n<=300;++n){event.object=reinterpret_cast<void*>(n);bounded.observe(&event);}
        verify(bounded.images.size()==256&&bounded.lost==44,"bounded image ownership");
        if(staged){staged_observe=nullptr;FreeLibrary(staged);}
        std::puts("PASS actual injected JGL hooks: exact fill/copy/sprite pixels, contexts, lifetime, native transfer order, config off, thread isolation, detach/reattach, bounded capture");return 0;
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

#endif
