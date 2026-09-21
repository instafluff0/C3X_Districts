#pragma once
// Opt-in diagnostic journal. Fixed-width little-endian values, no native
// pointers or C++ object layouts on disk. A stopped capture is a valid prefix,
// never a silently sampled stream. Normal gameplay does not open a file.
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <atomic>
#include "gpu_image_commands.h"

namespace c3x_recording {
enum Event : std::uint32_t { begin=1,end=2,create=3,destroy=4,upload=5,submit=6,
    external=7,checkpoint=8,display=9,native_begin=10,native_end=11,lifetime=12,
    visual=13,stop=14 };
enum Stop : unsigned { closed=0,byte_limit=1,time_limit=2,io_error=3,
    allocation_failure=4,low_address_space=5,unsupported=6 };
using Bytes=std::vector<unsigned char>;
inline void u32(Bytes& out,std::uint32_t value){for(unsigned n=0;n<4;++n)out.push_back(static_cast<unsigned char>(value>>(n*8)));}
inline void u64(Bytes& out,std::uint64_t value){u32(out,std::uint32_t(value));u32(out,std::uint32_t(value>>32));}
inline std::uint32_t checksum(Bytes const& data){std::uint32_t h=2166136261u;for(auto b:data)h=(h^b)*16777619u;return h;}
inline void command(Bytes& out,c3x_gpu_images::Command const& c){
    u32(out,unsigned(c.kind));for(auto id:{c.destination,c.source,c.background,c.detail,c.background_detail,c.program})u64(out,id);
    for(auto v:{c.area.left,c.area.top,c.area.right,c.area.bottom,c.clip.left,c.clip.top,c.clip.right,c.clip.bottom,
        c.source_x,c.source_y,c.source_width,c.source_height})u32(out,unsigned(v));u32(out,c.color);
}
// Lossless word runs. Literal blocks also bound worst-case storage; no image
// hashing can accidentally substitute a different source on collision.
inline void pixels(Bytes& out,unsigned const* words,std::size_t count){
    u32(out,unsigned(count));std::size_t at=0;
    while(at<count){std::size_t run=1;while(at+run<count&&words[at+run]==words[at]&&run<0x7fffffffu)++run;
        if(run>=3){u32(out,0x80000000u|unsigned(run));u32(out,words[at]);at+=run;continue;}
        auto first=at;at+=run;
        while(at<count&&at-first<65536){if(at+2<count&&words[at]==words[at+1]&&words[at]==words[at+2])break;++at;}
        u32(out,unsigned(at-first));for(auto n=first;n<at;++n)u32(out,words[n]);
    }
}
struct Cursor {
    Bytes const& data;std::size_t at=0;
    std::uint32_t u32(){if(data.size()-at<4)throw std::runtime_error("truncated recording value");std::uint32_t v=0;for(unsigned n=0;n<4;++n)v|=std::uint32_t(data[at++])<<(n*8);return v;}
    std::uint64_t u64(){auto lo=u32();return lo|(std::uint64_t(u32())<<32);}
    void done(){if(at!=data.size())throw std::runtime_error("trailing recording data");}
    c3x_gpu_images::Command command(){using namespace c3x_gpu_images;Command c={};auto kind=u32();if(kind>unsigned(Kind::native_lookup))throw std::runtime_error("unknown image command");c.kind=Kind(kind);
        c.destination=u64();c.source=u64();c.background=u64();c.detail=u64();c.background_detail=u64();c.program=u64();
        c.area={int(u32()),int(u32()),int(u32()),int(u32())};c.clip={int(u32()),int(u32()),int(u32()),int(u32())};
        c.source_x=int(u32());c.source_y=int(u32());c.source_width=int(u32());c.source_height=int(u32());c.color=u32();return c;}
    std::vector<unsigned> pixels(){auto count=u32();if(count>2240u*1260u)throw std::runtime_error("recorded image too large");std::vector<unsigned> out;out.reserve(count);
        while(out.size()<count){auto token=u32(),n=token&0x7fffffffu;if(!n||n>count-out.size())throw std::runtime_error("invalid pixel run");
            if(token&0x80000000u){auto value=u32();out.insert(out.end(),n,value);}else for(unsigned i=0;i<n;++i)out.push_back(u32());}return out;}
};
class Journal {
    std::mutex mutex;FILE* file=nullptr;bool finished=false;std::atomic<bool> enabled{false};
    std::uint64_t written=0,sequence=0,stream=0,token=0,native_serial=0;
    std::uint64_t limit=512ull*1024*1024;long long start_ticks=0,frequency=1,capture_start=0;
    std::unordered_map<void*,std::uint64_t> native_ids;
    void raw(Event kind,std::uint64_t owner,Bytes const& payload){
        LARGE_INTEGER now={};QueryPerformanceCounter(&now);Bytes header;
        u32(header,0x31523343);u32(header,unsigned(kind));u32(header,unsigned(payload.size()));u32(header,checksum(payload));
        u64(header,++sequence);u64(header,owner);u64(header,std::uint64_t(now.QuadPart-start_ticks));
        if(fwrite(header.data(),1,header.size(),file)!=header.size()||(!payload.empty()&&fwrite(payload.data(),1,payload.size(),file)!=payload.size())||fflush(file)!=0){finished=true;enabled=false;OutputDebugStringA("[C3X renderer] recording stopped: disk error\n");}
        written+=header.size()+payload.size();
    }
    void finish_locked(unsigned reason){if(!file||finished)return;Bytes b;u32(b,reason);raw(stop,0,b);finished=true;enabled=false;fclose(file);file=nullptr;}
public:
    Journal()noexcept{try{
        wchar_t path[32768]={};auto length=GetEnvironmentVariableW(L"C3X_RENDERER_RECORD_FILE",path,32768);if(!length||length>=32768)return;
        if(_wfopen_s(&file,path,L"wb")||!file)return;
        LARGE_INTEGER f={},q={};QueryPerformanceFrequency(&f);QueryPerformanceCounter(&q);frequency=f.QuadPart;start_ticks=q.QuadPart;
        Bytes header;u32(header,0x52433343);u32(header,2);u64(header,std::uint64_t(frequency));
        if(fwrite(header.data(),1,header.size(),file)!=header.size()){fclose(file);file=nullptr;return;}written=header.size();
        enabled=true;OutputDebugStringA("[C3X renderer] composition recording enabled: diagnostic readbacks; not an FPS baseline\n");
    }catch(...){if(file)fclose(file);file=nullptr;enabled=false;}}
    ~Journal(){
        // Process detach can terminate another thread while it owns this lock.
        // Every complete record is already flushed; an absent footer is safer
        // than blocking game exit to wait for a terminated writer.
        if(!mutex.try_lock())return;
        try{finish_locked(closed);if(file)fclose(file);}catch(...){}
        mutex.unlock();
    }
    bool active()const noexcept{return enabled.load(std::memory_order_relaxed);}
    void finish(unsigned reason)noexcept{try{std::lock_guard<std::mutex> lock(mutex);finish_locked(reason);}catch(...){enabled=false;}}
    void emit(Event kind,std::uint64_t owner,Bytes const& payload)noexcept{
        try{std::lock_guard<std::mutex> lock(mutex);if(!file||finished)return;LARGE_INTEGER now={};QueryPerformanceCounter(&now);
            if(written+payload.size()+128>limit){finish_locked(byte_limit);return;}
            if(capture_start&&now.QuadPart-capture_start>180*frequency){finish_locked(time_limit);return;}
            raw(kind,owner,payload);
        }catch(...){finish(allocation_failure);}
    }
    std::uint64_t open(std::uint64_t budget)noexcept{try{if(!active())return 0;std::uint64_t id;{std::lock_guard<std::mutex> lock(mutex);id=++stream;
        if(!capture_start){LARGE_INTEGER now={};QueryPerformanceCounter(&now);capture_start=now.QuadPart;}}
        Bytes b;u64(b,budget);emit(begin,id,b);return id;}catch(...){finish(allocation_failure);return 0;}}
    bool snapshot_allowed()noexcept{if(!active())return false;MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);
        if(!GlobalMemoryStatusEx(&memory)||memory.ullAvailVirtual<256ull*1024*1024){finish(low_address_space);return false;}return true;}
    std::uint64_t native_id(void* pointer){if(!pointer)return 0;auto found=native_ids.find(pointer);if(found!=native_ids.end())return found->second;
        if(native_ids.size()>=8192)throw std::runtime_error("native recording identity capacity");auto id=++native_serial;native_ids.emplace(pointer,id);return id;}
    std::uint64_t native(Event kind,int operation,void* image,void* source,unsigned value,int result=0,bool retire=false)noexcept{
        try{if(!active())return 0;Bytes b;std::uint64_t call;
            {std::lock_guard<std::mutex> lock(mutex);call=++token;u64(b,call);u32(b,unsigned(operation));u64(b,native_id(image));u64(b,native_id(source));u32(b,value);u32(b,unsigned(result));
                if(retire)native_ids.erase(image);}
            emit(kind,0,b);return call;
        }catch(...){finish(allocation_failure);return 0;}
    }
};
inline Journal& journal(){static Journal value;return value;}
struct FailureGuard {
    bool recording;int exceptions=std::uncaught_exceptions();
    ~FailureGuard(){if(recording&&std::uncaught_exceptions()>exceptions)journal().finish(unsupported);}
};
// No allocation and no file I/O after the recorder has stopped.
template<class Build> inline void event(Event kind,std::uint64_t owner,Build build)noexcept{
    auto& j=journal();if(!j.active())return;try{Bytes b;build(b);j.emit(kind,owner,b);}catch(...){j.finish(allocation_failure);}
}
}
