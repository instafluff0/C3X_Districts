#pragma once
// Diagnostic sampling of Civ III's thread (C3X_RENDERER_SAMPLE_GAME=1 with
// C3X_RENDERER_TRACE_FILE): about every millisecond the sampler suspends that
// thread and records the QPC time, its instruction pointer, the nearest
// validated call sites in up to four successive modules (with the import slot
// the first calls through, if any) and up to eight return addresses inside the
// game executable, to <trace file>.samples. Module snapshots and the import tables
// of the executable and jgl.dll let the report name modules and the imported
// APIs being waited in. Renderer/tools/game_samples_report.py reads the file.
// x86 bridge only: Renderer64's own threads are never sampled.
#include <windows.h>
#include <tlhelp32.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace c3x_game_sampler {
#if defined(_M_IX86)
struct Image {std::uint32_t base=0,end=0;};
inline bool readable(std::uint32_t address,void* out,std::size_t size){
    SIZE_T got=0;
    return ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<void const*>(std::uintptr_t(address)),out,size,&got)&&got==size;
}
// A return address follows a direct (E8 rel32) or indirect (FF /2) call.
// Module images stay mapped while loaded, so their bytes are read directly.
inline bool call_site(std::uint32_t address,Image const& image){
    if(address<image.base+0x1000||address>=image.end)return false;
    auto before=reinterpret_cast<unsigned char const*>(std::uintptr_t(address-6));
    return before[1]==0xE8||(before[0]==0xFF&&(before[1]&0x38)==0x10)||
        (before[3]==0xFF&&(before[4]&0x38)==0x10)||(before[4]==0xFF&&(before[5]&0x38)==0x10);
}
inline Image module_of(std::vector<Image> const& modules,std::uint32_t address){
    for(auto const& m:modules)if(address>=m.base&&address<m.end)return m;
    return {};
}
// Import slots of one loaded module: kind 2 records (slot, "dll!function").
inline void write_imports(std::FILE* out,HMODULE module){
    if(!module)return;
    auto base=reinterpret_cast<unsigned char const*>(module);
    auto nt=reinterpret_cast<IMAGE_NT_HEADERS const*>(base+reinterpret_cast<IMAGE_DOS_HEADER const*>(base)->e_lfanew);
    auto const& directory=nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if(!directory.VirtualAddress)return;
    for(auto d=reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR const*>(base+directory.VirtualAddress);d->Name;++d){
        auto names=reinterpret_cast<IMAGE_THUNK_DATA32 const*>(base+(d->OriginalFirstThunk?d->OriginalFirstThunk:d->FirstThunk));
        for(std::uint32_t n=0;names[n].u1.AddressOfData;++n){
            char label[64]={};
            if(names[n].u1.Ordinal&IMAGE_ORDINAL_FLAG32)
                std::snprintf(label,sizeof(label),"%s!#%u",reinterpret_cast<char const*>(base+d->Name),unsigned(names[n].u1.Ordinal&0xffff));
            else std::snprintf(label,sizeof(label),"%s!%s",reinterpret_cast<char const*>(base+d->Name),
                reinterpret_cast<IMAGE_IMPORT_BY_NAME const*>(base+names[n].u1.AddressOfData)->Name);
            std::uint32_t kind=2,slot=std::uint32_t(std::uintptr_t(base+d->FirstThunk+n*4));
            std::fwrite(&kind,4,1,out);std::fwrite(&slot,4,1,out);std::fwrite(label,1,sizeof(label),out);
        }
    }
}
inline void write_modules(std::FILE* out,std::vector<Image>* modules=nullptr){
    HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,0);
    if(snapshot==INVALID_HANDLE_VALUE)return;
    MODULEENTRY32W entry={};entry.dwSize=sizeof(entry);
    for(BOOL more=Module32FirstW(snapshot,&entry);more;more=Module32NextW(snapshot,&entry)){
        std::uint32_t kind=1,base=std::uint32_t(std::uintptr_t(entry.modBaseAddr)),size=entry.modBaseSize;
        char name[64]={};WideCharToMultiByte(CP_UTF8,0,entry.szModule,-1,name,int(sizeof(name))-1,nullptr,nullptr);
        std::fwrite(&kind,4,1,out);std::fwrite(&base,4,1,out);std::fwrite(&size,4,1,out);std::fwrite(name,1,sizeof(name),out);
        if(modules)modules->push_back({base,base+size});
    }
    CloseHandle(snapshot);
}
inline void run(DWORD game_thread,std::FILE* out){
    HANDLE target=OpenThread(THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT|THREAD_QUERY_INFORMATION,FALSE,game_thread);
    HANDLE timer=CreateWaitableTimerExW(nullptr,nullptr,0x2/*CREATE_WAITABLE_TIMER_HIGH_RESOLUTION*/,TIMER_ALL_ACCESS);
    if(!timer)timer=CreateWaitableTimerExW(nullptr,nullptr,0,TIMER_ALL_ACCESS);
    if(!target||!timer){std::fclose(out);return;}
    auto exe_base=std::uint32_t(std::uintptr_t(GetModuleHandleA(nullptr)));
    auto dos=reinterpret_cast<IMAGE_DOS_HEADER const*>(std::uintptr_t(exe_base));
    auto nt=reinterpret_cast<IMAGE_NT_HEADERS const*>(std::uintptr_t(exe_base)+std::uintptr_t(dos->e_lfanew));
    Image exe{exe_base,exe_base+nt->OptionalHeader.SizeOfImage};
    std::vector<Image> modules;write_modules(out,&modules);
    write_imports(out,GetModuleHandleA(nullptr));write_imports(out,GetModuleHandleA("jgl.dll"));
    LARGE_INTEGER due={};due.QuadPart=-10000; // 1 ms, relative
    std::uint32_t stack[1024];
    for(unsigned sample=0;;++sample){
        if(!SetWaitableTimer(timer,&due,0,nullptr,nullptr,FALSE)||WaitForSingleObject(timer,1000)!=WAIT_OBJECT_0)break;
        if(SuspendThread(target)==DWORD(-1))break;
        CONTEXT context={};context.ContextFlags=CONTEXT_CONTROL|CONTEXT_INTEGER;
        LARGE_INTEGER now={};QueryPerformanceCounter(&now);
        // Suspended: copy registers and stack only (no allocation, no output).
        std::uint32_t record[17]={0};unsigned found=0,words=0;
        if(GetThreadContext(target,&context)){
            record[3]=context.Eip;
            // Read the stack in 1 KB pieces; it may end within the window.
            for(;words<1024;words+=256)
                if(!readable(context.Esp+words*4,stack+words,256*4))break;
        }
        ResumeThread(target);
        // Call sites in successive modules (e.g. ntdll <- win32u <- gdi32 <-
        // C3XRenderer.dll) and the import slot the first one calls through
        // (FF 15 disp32) name the work being executed.
        auto previous=module_of(modules,record[3]);unsigned chain=0;
        for(unsigned n=0;n<words&&chain<4;++n){
            auto m=module_of(modules,stack[n]);
            if(m.base&&m.base!=previous.base&&call_site(stack[n],m)){
                if(!chain){auto at=reinterpret_cast<unsigned char const*>(std::uintptr_t(stack[n]-6));
                    if(at[0]==0xFF&&at[1]==0x15)std::memcpy(&record[8],at+2,4);}
                record[4+chain++]=stack[n];previous=m;
            }
        }
        for(unsigned n=0;n<words&&found<8;++n)
            if(call_site(stack[n],exe))record[9+found++]=stack[n];
        record[0]=4;std::memcpy(&record[1],&now.QuadPart,8);
        std::fwrite(record,4,17,out);
        if(sample%10000==9999){modules.clear();write_modules(out,&modules);std::fflush(out);}
        else if(sample%1000==999)std::fflush(out);
    }
    std::fclose(out);CloseHandle(timer);CloseHandle(target);
}
inline void start(DWORD game_thread){
    char option[4]={},path[MAX_PATH]={};
    if(GetEnvironmentVariableA("C3X_RENDERER_SAMPLE_GAME",option,sizeof(option))!=1||option[0]!='1')return;
    auto length=GetEnvironmentVariableA("C3X_RENDERER_TRACE_FILE",path,MAX_PATH-10);
    if(!length||length>=MAX_PATH-10)return;
    strcat_s(path,".samples");
    std::FILE* out=nullptr;
    if(fopen_s(&out,path,"wb")||!out)return;
    // Detached: process exit ends it; joining under the loader lock could hang.
    std::thread([game_thread,out]{run(game_thread,out);}).detach();
}
#else
inline void start(DWORD){}
#endif
}
