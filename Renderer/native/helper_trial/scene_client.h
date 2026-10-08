#pragma once
#define NOMINMAX
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>
#include "scene_wire.h"
#include "../remote_scene_output.h"
#include "../scene_projection.h"

namespace c3x_helper_trial {
// Bounded, one-operation diagnostic transport shared by Gate 2 and the full
// interleaved replay. Both payload and response contain values, never native
// pointers; the helper owns the renderer DLL and all of its scene allocations.
class SceneClient {
    HANDLE mapping=nullptr,request=nullptr,response=nullptr,control=nullptr,image_completed=nullptr,camera_mutex=nullptr,process=nullptr;
    HANDLE image_interrupt=nullptr; // local: a camera request was posted during an image wait
    Wire* wire=nullptr;
    unsigned sequence=0;
    std::atomic<std::int64_t> admitted_camera{0};
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
        if(image_completed){CloseHandle(image_completed);image_completed=nullptr;}
        if(image_interrupt){CloseHandle(image_interrupt);image_interrupt=nullptr;}
        if(camera_mutex){CloseHandle(camera_mutex);camera_mutex=nullptr;}
        if(control){CloseHandle(control);control=nullptr;}
        if(request){CloseHandle(request);request=nullptr;}
        if(mapping){CloseHandle(mapping);mapping=nullptr;}
    }
public:
    struct Stats {unsigned sequence=0;std::uint64_t service_us=0,private_bytes=0;};
    Stats stats()const{return wire?Stats{wire->sequence,wire->service_us,wire->private_bytes}:Stats{};}
    bool alive()const{return process&&WaitForSingleObject(process,0)==WAIT_TIMEOUT;}
    unsigned frames()const{return wire?unsigned(InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&wire->visual_frames),0,0)):0;}
    void publication_pressure(std::size_t records){
        if(wire)InterlockedExchange(reinterpret_cast<volatile LONG*>(&wire->native_queue_records),LONG(records));
    }
    void begin_image_receipt(){if(!ResetEvent(image_completed))throw std::runtime_error("image receipt reset failed");}
    void prepare_camera_receipt(){
        if(wire)InterlockedExchange(reinterpret_cast<volatile LONG*>(&wire->camera_receiver_thread),LONG(GetCurrentThreadId()));
    }
    bool camera_completion(std::int64_t ticket,std::vector<unsigned char>& bytes,int& code){
        if(!wire||!camera_mutex||!alive()||!InterlockedCompareExchange(
            reinterpret_cast<volatile LONG*>(&wire->camera_completion_available),0,0))return false;
        auto acquired=WaitForSingleObject(camera_mutex,0);
        if(acquired==WAIT_TIMEOUT){code=C3X_RENDERER_RESULT_PENDING;return true;}
        if(acquired!=WAIT_OBJECT_0){if(acquired==WAIT_ABANDONED)ReleaseMutex(camera_mutex);return false;}
        struct Unlock {HANDLE mutex;~Unlock(){ReleaseMutex(mutex);}} unlock{camera_mutex};
        return c3x_remote_scene::snapshot_camera_completion(wire->camera_completion,ticket,bytes,code);
    }
    void wait_image_receipt(){
        HANDLE ready[2]={image_completed,process};
        if(WaitForMultipleObjects(2,ready,FALSE,120000)!=WAIT_OBJECT_0)
            throw std::runtime_error("image execution receipt unavailable");
    }
    // As wait_image_receipt, but returns false early when interrupt_image_wait
    // was called (from any thread) so the waiting thread can send other work.
    bool wait_image_receipt_or_interrupt(){
        if(!image_interrupt){wait_image_receipt();return true;}
        HANDLE ready[3]={image_completed,process,image_interrupt};
        auto result=WaitForMultipleObjects(3,ready,FALSE,120000);
        if(result==WAIT_OBJECT_0)return true;
        if(result==WAIT_OBJECT_0+2)return false;
        throw std::runtime_error("image execution receipt unavailable");
    }
    void interrupt_image_wait(){if(image_interrupt)SetEvent(image_interrupt);}
    unsigned presented_zoom()const{
        auto value=wire?unsigned(InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&wire->presented_zoom_q16),0,0)):0;
        return value>=c3x_renderer::SceneProjection::minimum_q16&&value<=c3x_renderer::SceneProjection::maximum_q16?
            value:65536u;
    }
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
            control=CreateEventW(nullptr,FALSE,FALSE,name(base,L"_control").c_str());
            image_completed=CreateEventW(nullptr,FALSE,FALSE,name(base,L"_images_complete").c_str());
            image_interrupt=CreateEventW(nullptr,FALSE,FALSE,nullptr);
            camera_mutex=CreateMutexW(nullptr,FALSE,name(base,L"_camera_complete_mutex").c_str());
            if(!mapping||!request||!response||!control||!image_completed||!camera_mutex)throw std::runtime_error("x64 scene IPC creation failed");
            wire=static_cast<Wire*>(MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(Wire)));
            if(!wire)throw std::runtime_error("x64 scene IPC view failed");
            std::wstring command=L"\""+helper+L"\" --child \""+base+L"\" \""+dll+L"\" "+std::to_wstring(GetCurrentProcessId());
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
    void remember_camera(std::int64_t ticket){admitted_camera.store(ticket,std::memory_order_release);}
    void retire_camera_receipt(){
        admitted_camera.store(0,std::memory_order_release);
        if(wire)InterlockedExchange64(&wire->obsolete_camera_through,0);
    }
    void supersede_pending_camera(){
        auto ticket=admitted_camera.load(std::memory_order_acquire);
        if(ticket>0 && wire && control){
            InterlockedExchange64(&wire->obsolete_camera_through,ticket);
            SetEvent(control);
        }
    }
    std::uint64_t duplicate_into_helper(HANDLE source){
        if(!source||!process)throw std::runtime_error("invalid surface handle");
        HANDLE remote=nullptr;
        if(!DuplicateHandle(GetCurrentProcess(),source,process,&remote,0,FALSE,DUPLICATE_SAME_ACCESS))
            throw std::runtime_error("surface handle duplication failed");
        return std::uint64_t(reinterpret_cast<std::uintptr_t>(remote));
    }
    Wire const& call(unsigned kind,unsigned subtype,unsigned char const* bytes,unsigned count,
                     unsigned expected_code=0,std::int64_t recorded_ticket=0,std::int64_t recorded_image=0,
                     std::int64_t clock_ticks=0,std::int64_t clock_frequency=0,bool final_image=false,
                     bool live=false,bool raw_shared=false,bool replay_override=false,bool required_loading=false){
        if(!wire||!process||count>wire_capacity||(!bytes&&count))throw std::runtime_error("invalid scene request");
        wire->magic=wire_magic;wire->version=wire_version;wire->sequence=++sequence;
        wire->kind=kind;wire->subtype=subtype;wire->size=count;wire->reply_size=0;
        wire->shared_raw=raw_shared?1u:0u;wire->live=live?1u:0u;
        wire->consumer_pid=final_image?GetCurrentProcessId():0;
        wire->expected_code=expected_code;wire->recorded_ticket=recorded_ticket;wire->recorded_image=recorded_image;
        wire->replay_clock=!live||replay_override;wire->clock_ticks=clock_ticks;wire->clock_frequency=clock_frequency;
        if(count)std::memcpy(wire->payload,bytes,count);
        HANDLE ready[2]={response,process};
        if(!SetEvent(request))throw std::runtime_error("x64 scene helper request signal failed");
        DWORD const wait=WaitForMultipleObjects(2,ready,FALSE,required_loading?INFINITE:120000);
        if(wait==WAIT_OBJECT_0+1){
            DWORD exit_code=0;
            GetExitCodeProcess(process,&exit_code);
            throw std::runtime_error("x64 scene helper exited before response, code="+
                std::to_string(static_cast<unsigned long>(exit_code)));
        }
        if(wait!=WAIT_OBJECT_0)
            throw std::runtime_error("x64 scene helper response wait failed, code="+
                std::to_string(static_cast<unsigned long>(wait)));
        if(wire->magic!=wire_magic||wire->version!=wire_version||wire->sequence!=sequence||
           wire->kind!=kind||wire->subtype!=subtype||wire->size!=count||wire->reply_size>wire_capacity)
            throw std::runtime_error("x64 scene helper response header changed");
        if(wire->status)throw std::runtime_error(std::string("x64 scene helper: ")+wire->error);
        return *wire;
    }
    Wire const& call_live(unsigned kind,unsigned subtype,unsigned char const* bytes,unsigned count,
                          bool shared_frame=false,bool raw_shared=false,
                          std::int64_t ticks=0,std::int64_t frequency=0,bool replay_override=false,bool required_loading=false){
        return call(kind,subtype,bytes,count,0,0,0,ticks,frequency,shared_frame,true,raw_shared,replay_override,required_loading);
    }
};
}
