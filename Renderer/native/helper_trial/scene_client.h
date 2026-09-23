#pragma once
#define NOMINMAX
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "scene_wire.h"

namespace c3x_helper_trial {
// Bounded, one-operation diagnostic transport shared by Gate 2 and the full
// interleaved replay. Both payload and response contain values, never native
// pointers; the helper owns the renderer DLL and all of its scene allocations.
class SceneClient {
    HANDLE mapping=nullptr,request=nullptr,response=nullptr,process=nullptr;
    Wire* wire=nullptr;
    unsigned sequence=0;
    std::wstring base;
    static std::wstring name(std::wstring const& value,wchar_t const* suffix){return value+suffix;}
    void stop()noexcept{
        if(process){
            if(WaitForSingleObject(process,0)==WAIT_TIMEOUT){
                if(wire&&request){wire->magic=wire_magic;wire->version=wire_version;
                    wire->sequence=++sequence;wire->kind=0;SetEvent(request);}
                if(WaitForSingleObject(process,5000)==WAIT_TIMEOUT){TerminateProcess(process,1);WaitForSingleObject(process,5000);}
            }
            CloseHandle(process);process=nullptr;
        }
        if(wire){UnmapViewOfFile(wire);wire=nullptr;}
        if(response){CloseHandle(response);response=nullptr;}
        if(request){CloseHandle(request);request=nullptr;}
        if(mapping){CloseHandle(mapping);mapping=nullptr;}
    }
public:
    SceneClient(SceneClient const&)=delete;
    SceneClient& operator=(SceneClient const&)=delete;
    SceneClient(std::wstring const& helper,std::wstring const& dll){
        try{
            if(helper.find(L'"')!=std::wstring::npos||dll.find(L'"')!=std::wstring::npos)
                throw std::runtime_error("invalid helper path");
            base=L"Local\\C3XScene_"+std::to_wstring(GetCurrentProcessId())+L"_"+std::to_wstring(GetTickCount64());
            mapping=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,sizeof(Wire),name(base,L"_map").c_str());
            request=CreateEventW(nullptr,FALSE,FALSE,name(base,L"_request").c_str());
            response=CreateEventW(nullptr,FALSE,FALSE,name(base,L"_response").c_str());
            if(!mapping||!request||!response)throw std::runtime_error("x64 scene IPC creation failed");
            wire=static_cast<Wire*>(MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(Wire)));
            if(!wire)throw std::runtime_error("x64 scene IPC view failed");
            std::wstring command=L"\""+helper+L"\" --child \""+base+L"\" \""+dll+L"\"";
            std::vector<wchar_t> writable(command.begin(),command.end());writable.push_back(0);
            STARTUPINFOW startup={};startup.cb=sizeof(startup);PROCESS_INFORMATION child={};
            if(!CreateProcessW(helper.c_str(),writable.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,
                               nullptr,nullptr,&startup,&child))throw std::runtime_error("x64 scene helper start failed");
            process=child.hProcess;CloseHandle(child.hThread);
            HANDLE ready[2]={response,process};
            if(WaitForMultipleObjects(2,ready,FALSE,120000)!=WAIT_OBJECT_0)
                throw std::runtime_error("x64 scene helper startup failed");
        }catch(...){stop();throw;}
    }
    ~SceneClient(){stop();}
    Wire const& call(unsigned kind,unsigned subtype,unsigned char const* bytes,unsigned count,
                     unsigned expected_code=0,std::int64_t recorded_ticket=0,std::int64_t recorded_image=0,
                     std::int64_t clock_ticks=0,std::int64_t clock_frequency=0,bool final_image=false,
                     bool live=false,bool raw_shared=false){
        if(!wire||!process||count>wire_capacity||(!bytes&&count))throw std::runtime_error("invalid scene request");
        wire->magic=wire_magic;wire->version=wire_version;wire->sequence=++sequence;
        wire->kind=kind;wire->subtype=subtype;wire->size=count;wire->reply_size=0;
        wire->shared_raw=raw_shared?1u:0u;wire->live=live?1u:0u;
        wire->consumer_pid=final_image?GetCurrentProcessId():0;
        wire->expected_code=expected_code;wire->recorded_ticket=recorded_ticket;wire->recorded_image=recorded_image;
        wire->replay_clock=live?0u:1u;wire->clock_ticks=clock_ticks;wire->clock_frequency=clock_frequency;
        if(count)std::memcpy(wire->payload,bytes,count);
        HANDLE ready[2]={response,process};
        if(!SetEvent(request)||WaitForMultipleObjects(2,ready,FALSE,120000)!=WAIT_OBJECT_0)
            throw std::runtime_error("x64 scene helper response failed");
        if(wire->magic!=wire_magic||wire->version!=wire_version||wire->sequence!=sequence||
           wire->kind!=kind||wire->subtype!=subtype||wire->size!=count||wire->reply_size>wire_capacity)
            throw std::runtime_error("x64 scene helper response header changed");
        if(wire->status)throw std::runtime_error(std::string("x64 scene helper: ")+wire->error);
        return *wire;
    }
    Wire const& call_live(unsigned kind,unsigned subtype,unsigned char const* bytes,unsigned count,
                          bool shared_frame=false,bool raw_shared=false,
                          std::int64_t ticks=0,std::int64_t frequency=0){
        return call(kind,subtype,bytes,count,0,0,0,ticks,frequency,shared_frame,true,raw_shared);
    }
};
}
