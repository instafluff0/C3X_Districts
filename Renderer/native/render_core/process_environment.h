#pragma once
#include <windows.h>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Diagnostic switches that are fixed for the process lifetime. The draw paths
// queried them per layer and per record; GetEnvironmentVariableA locks the
// process environment and scans it on every call (hundreds of times a frame).
// Same contract as GetEnvironmentVariableA: 0 when unset, the copied length
// when it fits, otherwise the required size including the terminator.
// Only use this for variables that are never changed while the process runs.
inline DWORD cached_environment(char const* name,char* buffer,DWORD size){
    struct Entry {std::string name,value;bool present=false;};
    static std::mutex mutex;
    static std::vector<Entry> entries;
    std::lock_guard<std::mutex> lock(mutex);
    Entry const* found=nullptr;
    for(auto const& entry:entries)if(entry.name==name){found=&entry;break;}
    if(!found){
        Entry entry;entry.name=name;
        // Avoid the Windows SDK's lowercase type macros in local names.
        char inline_value[256]={};
        DWORD length=GetEnvironmentVariableA(name,inline_value,DWORD(sizeof(inline_value)));
        if(length>=sizeof(inline_value)){
            std::string heap_value(length,'\0');
            length=GetEnvironmentVariableA(name,heap_value.data(),length);
            heap_value.resize(length);entry.value=std::move(heap_value);entry.present=true;
        }else if(length || GetLastError()!=ERROR_ENVVAR_NOT_FOUND){
            entry.value.assign(inline_value,length);entry.present=length!=0;
        }
        entries.push_back(std::move(entry));found=&entries.back();
    }
    if(!found->present)return 0;
    DWORD length=DWORD(found->value.size());
    if(!buffer || length>=size)return length+1;
    std::memcpy(buffer,found->value.c_str(),std::size_t(length)+1);
    return length;
}
}}
