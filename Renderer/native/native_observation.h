#pragma once
// Caller-thread diagnostics only. No native image, HDC or pointer reaches a worker.
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "c3x_renderer_api.h"

namespace c3x_native_observation {
inline bool verified_module(HMODULE module) {
    if (!module || module != GetModuleHandleA("jgl.dll")) return false;
    wchar_t path[MAX_PATH];
    DWORD length=GetModuleFileNameW(module,path,MAX_PATH);
    if (!length || length>=MAX_PATH) return false;
    HANDLE file=CreateFileW(path,GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if (file==INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER size={};
    bool valid=GetFileSizeEx(file,&size) && size.QuadPart>0 && size.QuadPart<1024*1024;
    BCRYPT_ALG_HANDLE algorithm=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    valid=valid && BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0;
    valid=valid && BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0)>=0;
    unsigned char block[16384],digest[32]; DWORD count=0;
    while (valid) {
        if (!ReadFile(file,block,sizeof block,&count,nullptr)) {valid=false;break;}
        if (!count) break;
        valid=BCryptHashData(hash,block,count,0)>=0;
    }
    valid=valid && BCryptFinishHash(hash,digest,sizeof digest,0)>=0;
    unsigned char const expected[32]={0x0b,0x0c,0xd5,0x14,0xde,0x0d,0x95,0xb9,0x3d,0x20,0x65,0x5f,0x4b,0x51,0x94,0x17,
        0x3f,0xe2,0x57,0x32,0x5a,0xf8,0x2e,0x55,0x8a,0x15,0x23,0x05,0xff,0x0d,0xbd,0xf2};
    valid=valid && std::memcmp(digest,expected,sizeof digest)==0;
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm,0);
    CloseHandle(file); return valid;
}

struct Surface {
    unsigned id=0,roles=0;
    std::array<unsigned,C3X_NATIVE_OPERATION_COUNT> calls={};
    unsigned raw_pixels=0,raw_dc=0;
};
struct Edge {unsigned source=0,destination=0;};
struct Capture {
    // Fixed capture limits; exhaustion is reported, never treated as complete coverage.
    std::unordered_map<void*,Surface> images;
    std::array<Edge,512> edges={};
    unsigned next_id=0,edge_count=0,presents=0,window_presents=0,depth=0,nested=0;
    unsigned lost=0; bool ended=false;
    DWORD owner=0; LARGE_INTEGER begin={},frequency={};
    double transfer_ms=0;
    std::array<char,65536> pending={};std::size_t pending_size=0;
    void (*write)(char const*)=[](char const* text){OutputDebugStringA(text);};

    void emit(char const* text) {
        std::size_t length=std::strlen(text);
        if (length>=pending.size()-pending_size) {++lost;return;}
        std::memcpy(pending.data()+pending_size,text,length);pending_size+=length;
    }
    Surface* surface(void* object,c3x_renderer_native_observation const& event) {
        if (!object) return nullptr;
        auto found=images.find(object);
        if (found!=images.end()) return &found->second;
        if (images.size()>=256) {++lost;return nullptr;}
        Surface value;value.id=++next_id;
        auto* result=&images.emplace(object,value).first->second;
        char line[256];std::snprintf(line,sizeof line,"[C3X native] surface=%u width=%d height=%d bits=%d\n",
            result->id,event.width,event.height,event.bit_count);
        emit(line);return result;
    }
    void record_surface(Surface& s) {
        char line[512];
        unsigned total=0;for (auto n:s.calls)total+=n;if (!total)return;
        std::snprintf(line,sizeof line,"[C3X native] surface=%u roles=%u copy=%u fill=%u image=%u sprite=%u pixel=%u bits=%u dc=%u raw_pixels=%u raw_dc=%u\n",
            s.id,s.roles,s.calls[C3X_NATIVE_COPY],s.calls[C3X_NATIVE_FILL],s.calls[C3X_NATIVE_IMAGE_DRAW],s.calls[C3X_NATIVE_SPRITE],
            s.calls[C3X_NATIVE_PIXEL],s.calls[C3X_NATIVE_BITS],s.calls[C3X_NATIVE_DC],s.raw_pixels,s.raw_dc);emit(line);
        s.calls={};s.raw_pixels=s.raw_dc=0;
    }
    void flush() {
        pending[pending_size]=0;if(pending_size)write(pending.data());pending_size=0;
        for (auto& item:images) record_surface(item.second);
        pending[pending_size]=0;if(pending_size)write(pending.data());pending_size=0;
        char line[512];
        std::snprintf(line,sizeof line,"[C3X native] presents=%u window=%u native_transfer_ms=%.3f nested=%u dropped=%u edges=%u\n",
            presents,window_presents,transfer_ms,nested,lost,edge_count);write(line);
        transfer_ms=0;window_presents=0;nested=0;
    }
    int observe(c3x_renderer_native_observation const* event) {
        if (!event || event->struct_size!=sizeof *event || event->operation<0 || event->operation>=C3X_NATIVE_OPERATION_COUNT) return 0;
        auto const& e=*event;
        if (e.operation==C3X_NATIVE_VERIFY) {
            if (ended || !verified_module(static_cast<HMODULE>(e.object))) return 0;
            owner=GetCurrentThreadId();QueryPerformanceFrequency(&frequency);images.reserve(256);
            emit("[C3X native] probe=attached mode=pass-through roles=1:map,2:screen max_surfaces=256 max_edges=512 max_presents=8192\n");return 1;
        }
        if (ended || owner!=GetCurrentThreadId()) return 0;
        if (e.operation==C3X_NATIVE_INIT || e.operation==C3X_NATIVE_DESTROY) {
            auto found=images.find(e.object);
            if (found!=images.end()) {
                record_surface(found->second);
                char line[160];std::snprintf(line,sizeof line,"[C3X native] retire=%u reason=%s\n",found->second.id,
                    e.operation==C3X_NATIVE_INIT?"reinit":"destroy");emit(line);images.erase(found);
            }
            if (e.operation==C3X_NATIVE_DESTROY) return 1;
        }
        Surface* s=surface(e.object,e);
        if (s) {
            ++s->calls[e.operation];
            unsigned role=e.operation==C3X_NATIVE_MAP?1u:e.operation==C3X_NATIVE_SCREEN?2u:0u;
            if (role && !(s->roles&role)) {s->roles|=role;char line[96];std::snprintf(line,sizeof line,"[C3X native] surface=%u roles=%u\n",s->id,s->roles);emit(line);}
            if (!e.context) {
                if (e.operation==C3X_NATIVE_PIXEL || e.operation==C3X_NATIVE_BITS) ++s->raw_pixels;
                if (e.operation==C3X_NATIVE_DC) ++s->raw_dc;
            }
        }
        // Directed attempted copy dependencies, not proof of successful writes or coverage.
        if (s && (e.operation==C3X_NATIVE_COPY || e.operation==C3X_NATIVE_IMAGE_DRAW) && e.peer) {
            c3x_renderer_native_observation unknown=e;unknown.width=unknown.height=unknown.bit_count=0;
            Surface* source=surface(e.peer,unknown);
            if (source) {
                unsigned n=0;for (;n<edge_count;++n)if(edges[n].source==source->id&&edges[n].destination==s->id)break;
                if(n==edge_count) {
                    if(edge_count==edges.size())++lost;
                    else {edges[edge_count++]={source->id,s->id};char line[128];std::snprintf(line,sizeof line,"[C3X native] copy_dependency=%u->%u\n",source->id,s->id);emit(line);}
                }
            }
        }
        if (e.operation==C3X_NATIVE_SCREEN) {if (depth++)++nested;else QueryPerformanceCounter(&begin);}
        if (e.operation==C3X_NATIVE_PRESENT) {
            ++presents;++window_presents;
            if (depth && !--depth) {LARGE_INTEGER end;QueryPerformanceCounter(&end);transfer_ms+=1000.*double(end.QuadPart-begin.QuadPart)/double(frequency.QuadPart);}
            // Flush only after the outer native transfer has returned, never during its drawing.
            if (!depth && (presents==1 || window_presents>=120 || presents>=8192)) flush();
            if (!depth && presents>=8192) {ended=true;write("[C3X native] probe=complete hooks=detach\n");return 0;}
        }
        return 1;
    }
};
} // namespace c3x_native_observation
