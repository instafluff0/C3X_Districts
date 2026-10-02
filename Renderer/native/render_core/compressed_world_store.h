#pragma once
#include <windows.h>
#include <compressapi.h>
#include <array>
#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <cstdint>
#include <atomic>
#include <chrono>
#include "backing_allocation.h"

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
    std::uint64_t used=0,stored_raw=0,read_count=0,write_count=0,compaction_count=0;
    std::atomic<std::uint64_t> compress_us{0},lock_wait_us{0},optional_skips{0};
    std::uint64_t write_us=0,read_us=0,compaction_us=0,optional_writes=0;
    static constexpr unsigned blob_limit=16u*1024u*1024u;
    static constexpr std::uint64_t disk_limit=1024ull*1024u*1024u;
    BackingAllocation allocation;
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
    void trim_locked(){
        auto end=allocation.high_water();if(end>=used || file==INVALID_HANDLE_VALUE)return;
        LARGE_INTEGER offset{};offset.QuadPart=static_cast<LONGLONG>(end);
        if(SetFilePointerEx(file,offset,nullptr,FILE_BEGIN) && SetEndOfFile(file))used=end;
    }
    bool compact_locked(){
        auto begin=std::chrono::steady_clock::now();
        std::vector<typename std::map<Key,Entry>::iterator> ordered;ordered.reserve(entries.size());
        for(auto it=entries.begin();it!=entries.end();++it)ordered.push_back(it);
        std::sort(ordered.begin(),ordered.end(),[](auto const& a,auto const& b){return a->second.offset<b->second.offset;});
        std::uint64_t destination=0;bool ok=true;
        for(auto it:ordered){auto& entry=it->second;
            if(entry.offset!=destination){
                // Read the whole bounded record before overwriting a lower
                // range. Ascending moves cannot overwrite any later source.
                std::vector<unsigned char> packed(entry.packed);DWORD count=0;
                LARGE_INTEGER offset{};offset.QuadPart=static_cast<LONGLONG>(entry.offset);
                if(!SetFilePointerEx(file,offset,nullptr,FILE_BEGIN) ||
                   !ReadFile(file,packed.data(),entry.packed,&count,nullptr) || count!=entry.packed){ok=false;break;}
                offset.QuadPart=static_cast<LONGLONG>(destination);count=0;
                if(!SetFilePointerEx(file,offset,nullptr,FILE_BEGIN) ||
                   !WriteFile(file,packed.data(),entry.packed,&count,nullptr) || count!=entry.packed){
                    // An overlapping partial write may have damaged this one
                    // record. Remove its proof so recovery uses the compiler.
                    stored_raw-=entry.raw;entries.erase(it);ok=false;break;
                }
                entry.offset=destination;
            }
            destination+=entry.packed;
        }
        std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges;ranges.reserve(entries.size());
        for(auto const& item:entries)ranges.emplace_back(item.second.offset,item.second.packed);
        bool layout=allocation.reset_layout(std::move(ranges));
        if(layout){trim_locked();if(ok)++compaction_count;}
        compaction_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();
        return layout && ok;
    }
public:
    struct Stats {std::uint64_t bytes=0,raw=0,reads=0,writes=0;unsigned records=0;
        std::uint64_t live=0,limit=0,compactions=0;
        std::uint64_t compress_us=0,lock_wait_us=0,write_us=0,read_us=0,compaction_us=0,optional_skips=0,optional_writes=0;};
    explicit CompressedWorldStore(std::uint64_t capacity=disk_limit):allocation((std::min)(capacity,disk_limit)){}
    ~CompressedWorldStore(){clear();if(library)FreeLibrary(library);}
    void clear(){std::lock_guard<std::mutex> lock(mutex);
        if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);file=INVALID_HANDLE_VALUE;
        entries.clear();allocation.clear();used=stored_raw=read_count=write_count=compaction_count=0;attempted=false;
        compress_us=lock_wait_us=optional_skips=0;write_us=read_us=compaction_us=optional_writes=0;
        // No borrowed compression call may survive the existing worker lease.
        if(library){FreeLibrary(library);library=nullptr;}
    }
    Stats statistics(){std::lock_guard<std::mutex> lock(mutex);return {used,stored_raw,read_count,write_count,unsigned(entries.size()),
        allocation.live_bytes(),allocation.capacity(),compaction_count,compress_us.load(),lock_wait_us.load(),write_us,read_us,compaction_us,optional_skips.load(),optional_writes};}
    bool contains(Key const& key){std::lock_guard<std::mutex> lock(mutex);return entries.count(key)!=0;}
    void invalidate(Key const& key){std::lock_guard<std::mutex> lock(mutex);auto found=entries.find(key);
        if(found!=entries.end()){stored_raw-=found->second.raw;allocation.release(found->second.offset,found->second.packed);
            entries.erase(found);trim_locked();}}
private:
    bool put_impl(Key const& key,std::vector<unsigned char> const& raw,bool optional,std::atomic<bool> const* stop){
        if(raw.empty() || raw.size()>blob_limit){if(optional)++optional_skips;return false;}
        auto stopped=[&]{return stop && stop->load(std::memory_order_relaxed);};
        auto acquire=[&](std::unique_lock<std::mutex>& lock){
            auto begin=std::chrono::steady_clock::now();
            if(optional){if(stopped() || !lock.try_lock()){++optional_skips;return false;}}
            else lock.lock();
            lock_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();
            return true;
        };
        {std::unique_lock<std::mutex> lock(mutex,std::defer_lock);if(!acquire(lock))return false;
            if(entries.count(key))return true;
            if(entries.size()>=131072u || !allocation.capacity() || !ensure_locked()){if(optional)++optional_skips;return false;}}
        // Each operation owns its compressor; concurrent lanes share only the
        // short file/index lock. Reset joins the workers before closing handles.
        COMPRESSOR_HANDLE handle=nullptr;
        if(!create_compressor(COMPRESS_ALGORITHM_XPRESS_HUFF,nullptr,&handle))return false;
        struct Close {decltype(close_compressor) fn;COMPRESSOR_HANDLE handle;~Close(){fn(handle);}} close{close_compressor,handle};
        auto compression_begin=std::chrono::steady_clock::now();
        SIZE_T size=0;compress(handle,raw.data(),raw.size(),nullptr,0,&size);
        if(!size || size>blob_limit)return false;
        std::vector<unsigned char> packed(size);
        if(!compress(handle,raw.data(),raw.size(),packed.data(),packed.size(),&size))return false;
        compress_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-compression_begin).count();
        auto hash=checksum(raw);std::unique_lock<std::mutex> lock(mutex,std::defer_lock);if(!acquire(lock))return false;
        if(entries.count(key))return true;
        if(file==INVALID_HANDLE_VALUE || entries.size()>=131072u || size>allocation.capacity()-allocation.live_bytes()){
            if(optional)++optional_skips;return false;
        }
        std::uint64_t location=0;
        if(!allocation.allocate(size,location)){
            // Optional backing never holds the shared file lock for compaction.
            // Demand can compile again; explicit backing-only jobs may compact.
            if(optional){++optional_skips;return false;}
            if(!compact_locked() || !allocation.allocate(size,location))return false;
        }
        LARGE_INTEGER offset{};offset.QuadPart=static_cast<LONGLONG>(location);DWORD written=0;
        auto write_begin=std::chrono::steady_clock::now();
        if(!SetFilePointerEx(file,offset,nullptr,FILE_BEGIN) ||
           !WriteFile(file,packed.data(),DWORD(size),&written,nullptr) || written!=size){
            // Conservatively account any partially extended file until its tail
            // is truncated. A failed record is never published into the index.
            used=(std::max)(used,location+written);allocation.release(location,size);trim_locked();return false;
        }
        entries.emplace(key,Entry{location,hash,unsigned(size),unsigned(raw.size())});
        write_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-write_begin).count();
        used=(std::max)(used,location+size);stored_raw+=raw.size();++write_count;if(optional)++optional_writes;return true;
    }
public:
    bool put(Key const& key,std::vector<unsigned char> const& raw){return put_impl(key,raw,false,nullptr);}
    bool put_optional(Key const& key,std::vector<unsigned char> const& raw,std::atomic<bool> const* stop=nullptr){return put_impl(key,raw,true,stop);}
    std::vector<unsigned char> get(Key const& key){
        Entry entry;std::vector<unsigned char> packed;
        {auto begin=std::chrono::steady_clock::now();std::lock_guard<std::mutex> lock(mutex);
            lock_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();
            auto found=entries.find(key);
            if(found==entries.end() || file==INVALID_HANDLE_VALUE)return {};
            entry=found->second;packed.resize(entry.packed);LARGE_INTEGER offset{};offset.QuadPart=static_cast<LONGLONG>(entry.offset);DWORD read=0;
            auto read_begin=std::chrono::steady_clock::now();
            if(!SetFilePointerEx(file,offset,nullptr,FILE_BEGIN) ||
               !ReadFile(file,packed.data(),entry.packed,&read,nullptr) || read!=entry.packed)return {};
            read_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-read_begin).count();
            ++read_count;}
        DECOMPRESSOR_HANDLE handle=nullptr;if(!create_decompressor(COMPRESS_ALGORITHM_XPRESS_HUFF,nullptr,&handle))return {};
        struct Close {decltype(close_decompressor) fn;DECOMPRESSOR_HANDLE handle;~Close(){fn(handle);}} close{close_decompressor,handle};
        std::vector<unsigned char> raw(entry.raw);SIZE_T size=0;
        if(!decompress(handle,packed.data(),packed.size(),raw.data(),raw.size(),&size) || size!=raw.size() || checksum(raw)!=entry.hash)return {};
        return raw;
    }
};
}}
