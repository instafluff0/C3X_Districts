#ifndef C3X_NAVIGATION_OPTIONS_H
#define C3X_NAVIGATION_OPTIONS_H

#include <cstring>

namespace c3x_renderer {
// The injected bridge already owns WORLD_REGIONS through the user's cache
// switch. Missing newer options inherit that policy; explicit benchmark and
// pressure-test values retain their original meaning. No environment mutation.
struct NavigationOptions {
    template<class Read> static bool enabled(Read read,char const* name,bool fallback=false) {
        char value[16]={};
        auto size=read(name,value,sizeof(value));
        return size==0?fallback:size<sizeof(value) && std::strcmp(value,"1")==0;
    }
    template<class Read> static bool retained(Read read,char const* name) {
        return enabled(read,name,enabled(read,"C3X_RENDERER_WORLD_REGIONS"));
    }
    template<class Read> static unsigned unit_pose_mib(Read read) {
        char value[16]={};
        auto size=read("C3X_RENDERER_UNIT_POSE_MEMORY",value,sizeof(value));
        if(size==0)return enabled(read,"C3X_RENDERER_WORLD_REGIONS")?512u:8u;
        if(size>=sizeof(value))return 8u;
        return std::strcmp(value,"512")==0?512u:std::strcmp(value,"1")==0?256u:8u;
    }
};
}
#endif
