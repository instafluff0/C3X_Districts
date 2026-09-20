#pragma once
#include <windows.h>
#include <compressapi.h>
#include <array>
#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <cstdint>

namespace c3x_renderer { namespace render_core {
// A bounded, delete-on-close session file backs compiler output. The existing
// worker pool performs compression and reads; there is no extra thread, GPU
// readback, cross-session identity or dependency on a source art format.
template<class Key> class CompressedWorldStore {
    struct Entry {std::uint64_t offset=0,hash=0;unsigned packed=0,raw=0;};
    std::mutex mutex;
    std::map<Key,Entry> entries;
    HANDLE file=INVALID_HANDLE_VALUE;
    HMODULE library=nullptr;
    decltype(&CreateCompressor) create_compressor=nullptr;
    decltype(&CloseCompressor) close_compressor=nullptr;
    decltype(&Compress) compress=nullptr;
    decltype(&CreateDecompressor) create_decompressor=nullptr;
    decltype(&CloseDecompressor) close_decompressor=nullptr;
    decltype(&Decompress) decompress=nullptr;
    std::uint64_t used=0,stored_raw=0,read_count=0,write_count=0;
    static constexpr unsigned blob_limit=16u*1024u*1024u;
    static constexpr std::uint64_t disk_limit=1024ull*1024u*1024u;
    bool attempted=false;
    static std::uint64_t checksum(std::vector<unsigned char> const& bytes){
        std::uint64_t hash=1469598103934665603ull;
        for(auto byte:bytes)hash=(hash^byte)*1099511628211ull;return hash;
    }
    bool ensure_locked(){
        if(file!=INVALID_HANDLE_VALUE)return true;
        if(attempted)return false;attempted=true;
        wchar_t system[MAX_PATH]={};auto length=GetSystemDirectoryW(system,MAX_PATH);
        if(!length || length+13>=MAX_PATH)return false;
        std::wstring path(system,length);path+=L"\\cabinet.dll";library=LoadLibraryW(path.c_str());
        if(!library)return false;
        create_compressor=reinterpret_cast<decltype(create_compressor)>(GetProcAddress(library,"CreateCompressor"));
        close_compressor=reinterpret_cast<decltype(close_compressor)>(GetProcAddress(library,"CloseCompressor"));
        compress=reinterpret_cast<decltype(compress)>(GetProcAddress(library,"Compress"));
        create_decompressor=reinterpret_cast<decltype(create_decompressor)>(GetProcAddress(library,"CreateDecompressor"));
        close_decompressor=reinterpret_cast<decltype(close_decompressor)>(GetProcAddress(library,"CloseDecompressor"));
        decompress=reinterpret_cast<decltype(decompress)>(GetProcAddress(library,"Decompress"));
        if(!create_compressor || !close_compressor || !compress || !create_decompressor || !close_decompressor || !decompress)return false;
        wchar_t directory[MAX_PATH]={},temporary[MAX_PATH]={};length=GetTempPathW(MAX_PATH,directory);
        if(!length || length>=MAX_PATH || !GetTempFileNameW(directory,L"C3R",0,temporary))return false;
        file=CreateFileW(temporary,GENERIC_READ|GENERIC_WRITE,0,nullptr,CREATE_ALWAYS,
            FILE_ATTRIBUTE_TEMPORARY|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
        if(file==INVALID_HANDLE_VALUE)DeleteFileW(temporary);
        return file!=INVALID_HANDLE_VALUE;
    }
public:
    struct Stats {std::uint64_t bytes=0,raw=0,reads=0,writes=0;unsigned records=0;};
    ~CompressedWorldStore(){clear();if(library)FreeLibrary(library);}
    void clear(){std::lock_guard<std::mutex> lock(mutex);
        if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);file=INVALID_HANDLE_VALUE;
        entries.clear();used=stored_raw=read_count=write_count=0;attempted=false;
        // No borrowed compression call may survive the existing worker lease.
        if(library){FreeLibrary(library);library=nullptr;}
    }
    Stats statistics(){std::lock_guard<std::mutex> lock(mutex);return {used,stored_raw,read_count,write_count,unsigned(entries.size())};}
    bool contains(Key const& key){std::lock_guard<std::mutex> lock(mutex);return entries.count(key)!=0;}
    void invalidate(Key const& key){std::lock_guard<std::mutex> lock(mutex);auto found=entries.find(key);
        if(found!=entries.end()){stored_raw-=found->second.raw;entries.erase(found);}}
    bool put(Key const& key,std::vector<unsigned char> const& raw){
        if(raw.empty() || raw.size()>blob_limit)return false;
        {std::lock_guard<std::mutex> lock(mutex);if(entries.count(key))return true;
            if(used>=disk_limit || entries.size()>=131072u || !ensure_locked())return false;}
        // Each operation owns its compressor; concurrent lanes share only the
        // short file/index lock. Reset joins the workers before closing handles.
        COMPRESSOR_HANDLE handle=nullptr;
        if(!create_compressor(COMPRESS_ALGORITHM_XPRESS_HUFF,nullptr,&handle))return false;
        struct Close {decltype(close_compressor) fn;COMPRESSOR_HANDLE handle;~Close(){fn(handle);}} close{close_compressor,handle};
        SIZE_T size=0;compress(handle,raw.data(),raw.size(),nullptr,0,&size);
        if(!size || size>blob_limit)return false;
        std::vector<unsigned char> packed(size);
        if(!compress(handle,raw.data(),raw.size(),packed.data(),packed.size(),&size))return false;
        auto hash=checksum(raw);std::lock_guard<std::mutex> lock(mutex);
        if(entries.count(key))return true;
        if(size>disk_limit-used || file==INVALID_HANDLE_VALUE)return false;
        LARGE_INTEGER offset{};offset.QuadPart=static_cast<LONGLONG>(used);DWORD written=0;
        if(!SetFilePointerEx(file,offset,nullptr,FILE_BEGIN) ||
           !WriteFile(file,packed.data(),DWORD(size),&written,nullptr) || written!=size)return false;
        entries.emplace(key,Entry{used,hash,unsigned(size),unsigned(raw.size())});
        used+=size;stored_raw+=raw.size();++write_count;return true;
    }
    std::vector<unsigned char> get(Key const& key){
        Entry entry;std::vector<unsigned char> packed;
        {std::lock_guard<std::mutex> lock(mutex);auto found=entries.find(key);
            if(found==entries.end() || file==INVALID_HANDLE_VALUE)return {};
            entry=found->second;packed.resize(entry.packed);LARGE_INTEGER offset{};offset.QuadPart=static_cast<LONGLONG>(entry.offset);DWORD read=0;
            if(!SetFilePointerEx(file,offset,nullptr,FILE_BEGIN) ||
               !ReadFile(file,packed.data(),entry.packed,&read,nullptr) || read!=entry.packed)return {};
            ++read_count;}
        DECOMPRESSOR_HANDLE handle=nullptr;if(!create_decompressor(COMPRESS_ALGORITHM_XPRESS_HUFF,nullptr,&handle))return {};
        struct Close {decltype(close_decompressor) fn;DECOMPRESSOR_HANDLE handle;~Close(){fn(handle);}} close{close_decompressor,handle};
        std::vector<unsigned char> raw(entry.raw);SIZE_T size=0;
        if(!decompress(handle,packed.data(),packed.size(),raw.data(),raw.size(),&size) || size!=raw.size() || checksum(raw)!=entry.hash)return {};
        return raw;
    }
};
}}
