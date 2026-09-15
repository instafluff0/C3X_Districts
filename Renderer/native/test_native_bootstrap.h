#pragma once
// Native loader boundary fixture: JGL's real factory runs exactly once, then
// the extracted production wrapper attaches tracking before any image init.
char const* tracking_candidate=nullptr;
HMODULE bootstrap_jgl=nullptr;void* bootstrap_graph=nullptr;
void* test_load_jgl(char const* path){
    if(!path)return nullptr;
    bootstrap_jgl=LoadLibraryA(path);if(!bootstrap_jgl)return nullptr;
    auto factory=reinterpret_cast<void*(__cdecl*)()>(GetProcAddress(bootstrap_jgl,"get_graphsy_object_ptr"));
    bootstrap_graph=factory?factory():nullptr;return bootstrap_graph;
}
void test_unload_jgl(){
    if(bootstrap_graph)reinterpret_cast<void*(__thiscall*)(void*,unsigned)>((*static_cast<void***>(bootstrap_graph))[0])(bootstrap_graph,1);
    bootstrap_graph=nullptr;if(bootstrap_jgl)FreeLibrary(bootstrap_jgl);bootstrap_jgl=nullptr;
}
void unload_custom_renderer(){
    // The real scene-unload body has separate scheduler/ownership tests. Here
    // retain the exact native drain/detach boundary without destroying test assets.
    set_custom_renderer_native_probe(nullptr);state.custom_renderer_native_observe=nullptr;
}
HMODULE tracking_load(char const*){return LoadLibraryA(tracking_candidate);}
#define LoadLibraryA tracking_load
#define load_jgl_lib test_load_jgl
#define unload_jgl_lib test_unload_jgl
#include "build/native_tracking_bootstrap.h"
#undef LoadLibraryA
#undef load_jgl_lib
#undef unload_jgl_lib
