#pragma once
#ifdef _WIN32
#include <windows.h>
#include <compressapi.h>
#endif
#include <array>
#include <algorithm>
#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <limits>

namespace c3x_renderer { namespace render_core {
// Requested allocation bytes, including map/control nodes, packed capacities
// and temporary codec/decode buffers. Caller-owned encoded input is excluded.
struct WorldStoreMemory {
    std::mutex mutex;
    std::uint64_t resident=0,in_flight=0,pinned=0,peak=0,limit=0,refusals=0;
    bool capacity_pressure=false;
    explicit WorldStoreMemory(std::uint64_t capacity):limit(capacity){}
    void sample(){peak=(std::max)(peak,resident+in_flight);}
    bool reserve(std::uint64_t bytes){std::lock_guard<std::mutex> lock(mutex);
        if(bytes>limit || resident>limit-bytes){++refusals;return false;}resident+=bytes;sample();return true;}
    void release(std::uint64_t bytes){std::lock_guard<std::mutex> lock(mutex);resident-=bytes;}
    void flight(std::uint64_t bytes){std::lock_guard<std::mutex> lock(mutex);in_flight+=bytes;sample();}
    void unflight(std::uint64_t bytes){std::lock_guard<std::mutex> lock(mutex);in_flight-=bytes;}
    bool publish(std::uint64_t bytes){std::lock_guard<std::mutex> lock(mutex);
        if(bytes>limit || resident>limit-bytes){++refusals;return false;}
        in_flight-=bytes;resident+=bytes;sample();return true;}
    void retire(std::uint64_t bytes){std::lock_guard<std::mutex> lock(mutex);pinned+=bytes;}
    void release_payload(std::uint64_t bytes,bool retired){std::lock_guard<std::mutex> lock(mutex);
        resident-=bytes;if(retired)pinned-=bytes;}
};
template<class T> struct WorldStoreAllocator {
    using value_type=T;std::shared_ptr<WorldStoreMemory> memory;
    explicit WorldStoreAllocator(std::shared_ptr<WorldStoreMemory> value={}):memory(std::move(value)){}
    template<class U> WorldStoreAllocator(WorldStoreAllocator<U> const& other):memory(other.memory){}
    T* allocate(std::size_t count){
        if(count>(std::numeric_limits<std::size_t>::max)()/sizeof(T))throw std::bad_alloc();
        auto bytes=count*sizeof(T);if(memory && !memory->reserve(bytes))throw std::bad_alloc();
        try{return static_cast<T*>(::operator new(bytes));}catch(...){if(memory)memory->release(bytes);throw;}}
    void deallocate(T* value,std::size_t count){::operator delete(value);if(memory)memory->release(count*sizeof(T));}
    template<class U> bool operator==(WorldStoreAllocator<U> const& other)const{return memory==other.memory;}
    template<class U> bool operator!=(WorldStoreAllocator<U> const& other)const{return !(*this==other);}
};
struct WorldStoreBuffer {
    std::vector<unsigned char> bytes;std::shared_ptr<WorldStoreMemory> memory;std::uint64_t charged=0;
    explicit WorldStoreBuffer(std::shared_ptr<WorldStoreMemory> value):memory(std::move(value)){}
    ~WorldStoreBuffer(){memory->unflight(charged);}
    void resize(std::size_t size){bytes.resize(size);auto next=bytes.capacity();
        memory->flight(next-charged);charged=next;}
    void compact(){
        if(bytes.capacity()==bytes.size())return;
        auto previous=charged;
        {std::vector<unsigned char> exact(bytes.begin(),bytes.end());
            memory->flight(exact.capacity());bytes.swap(exact);charged=bytes.capacity();}
        // The old codec reservation remains charged until its allocation dies.
        memory->unflight(previous);}
};
class WorldStoreCompression {
#ifdef _WIN32
    HMODULE library=nullptr;
    decltype(&CreateCompressor) create_compressor=nullptr;
    decltype(&CloseCompressor) close_compressor=nullptr;
    decltype(&Compress) compress=nullptr;
    decltype(&CreateDecompressor) create_decompressor=nullptr;
    decltype(&CloseDecompressor) close_decompressor=nullptr;
    decltype(&Decompress) decompress=nullptr;
    union Allocation {std::size_t bytes;std::max_align_t alignment;unsigned char span[MEMORY_ALLOCATION_ALIGNMENT];};
    static_assert(sizeof(Allocation)%MEMORY_ALLOCATION_ALIGNMENT==0,"codec allocation alignment");
    static PVOID __cdecl allocate(PVOID context,SIZE_T size){
        if(size>(std::numeric_limits<std::size_t>::max)()-sizeof(Allocation))return nullptr;
        auto bytes=size+sizeof(Allocation);auto value=static_cast<Allocation*>(std::malloc(bytes));
        if(!value)return nullptr;value->bytes=bytes;static_cast<WorldStoreMemory*>(context)->flight(bytes);return value+1;}
    static VOID __cdecl free(PVOID context,PVOID value){if(!value)return;auto allocation=static_cast<Allocation*>(value)-1;
        static_cast<WorldStoreMemory*>(context)->unflight(allocation->bytes);std::free(allocation);}
    bool available()const{return create_compressor && close_compressor && compress &&
        create_decompressor && close_decompressor && decompress;}
#endif
public:
    WorldStoreCompression(){
#ifdef _WIN32
        wchar_t system[MAX_PATH]={};auto length=GetSystemDirectoryW(system,MAX_PATH);
        if(!length || length+13>=MAX_PATH)return;
        std::wstring path(system,length);path+=L"\\cabinet.dll";library=LoadLibraryW(path.c_str());if(!library)return;
        create_compressor=reinterpret_cast<decltype(create_compressor)>(GetProcAddress(library,"CreateCompressor"));
        close_compressor=reinterpret_cast<decltype(close_compressor)>(GetProcAddress(library,"CloseCompressor"));
        compress=reinterpret_cast<decltype(compress)>(GetProcAddress(library,"Compress"));
        create_decompressor=reinterpret_cast<decltype(create_decompressor)>(GetProcAddress(library,"CreateDecompressor"));
        close_decompressor=reinterpret_cast<decltype(close_decompressor)>(GetProcAddress(library,"CloseDecompressor"));
        decompress=reinterpret_cast<decltype(decompress)>(GetProcAddress(library,"Decompress"));
#endif
    }
    ~WorldStoreCompression(){
#ifdef _WIN32
        if(library)FreeLibrary(library);
#endif
    }
    bool encode(std::vector<unsigned char> const& raw,WorldStoreBuffer& packed){
#ifdef _WIN32
        if(!available())return false;
        COMPRESS_ALLOCATION_ROUTINES routines={allocate,free,packed.memory.get()};
        COMPRESSOR_HANDLE handle=nullptr;if(!create_compressor(COMPRESS_ALGORITHM_XPRESS_HUFF,&routines,&handle))return false;
        struct Close {decltype(close_compressor) fn;COMPRESSOR_HANDLE handle;~Close(){fn(handle);}} close{close_compressor,handle};
        SIZE_T size=0;compress(handle,raw.data(),raw.size(),nullptr,0,&size);
        if(!size || size>16u*1024u*1024u)return false;packed.resize(size);
        if(!compress(handle,raw.data(),raw.size(),packed.bytes.data(),packed.bytes.size(),&size))return false;
        packed.bytes.resize(size);return true;
#else
        (void)raw;(void)packed;return false; // Portable contracts inject their own lossless codec.
#endif
    }
    bool decode(std::vector<unsigned char> const& packed,std::vector<unsigned char>& raw,WorldStoreMemory& memory){
#ifdef _WIN32
        if(!available())return false;
        COMPRESS_ALLOCATION_ROUTINES routines={allocate,free,&memory};
        DECOMPRESSOR_HANDLE handle=nullptr;if(!create_decompressor(COMPRESS_ALGORITHM_XPRESS_HUFF,&routines,&handle))return false;
        struct Close {decltype(close_decompressor) fn;DECOMPRESSOR_HANDLE handle;~Close(){fn(handle);}} close{close_decompressor,handle};
        SIZE_T size=0;return decompress(handle,packed.data(),packed.size(),raw.data(),raw.size(),&size) && size==raw.size();
#else
        (void)packed;(void)raw;(void)memory;return false;
#endif
    }
};
// Session-owned compressed RAM only. Existing producer lanes call this store;
// no generated-world file, fallback, arbitrary eviction or additional worker.
template<class Key,class Codec=WorldStoreCompression> class CompressedWorldStore {
    struct Blob {
        std::vector<unsigned char> packed;std::shared_ptr<WorldStoreMemory> memory;
        std::uint64_t hash=0;unsigned raw=0;mutable std::atomic<bool> retired{false};
        Blob(WorldStoreBuffer& buffer,std::uint64_t checksum,unsigned size):memory(buffer.memory),hash(checksum),raw(size){
            if(!memory->publish(buffer.charged))throw std::bad_alloc();
            packed=std::move(buffer.bytes);buffer.charged=0;}
        ~Blob(){memory->release_payload(packed.capacity(),retired.load());}
        void retire()const{if(!retired.exchange(true))memory->retire(packed.capacity());}
    };
    struct Entry {std::shared_ptr<Blob const> blob;std::uint64_t key_bytes=0;};
    using Pair=std::pair<Key const,Entry>;
    std::shared_ptr<WorldStoreMemory> memory;
    std::mutex mutex;
    std::map<Key,Entry,std::less<Key>,WorldStoreAllocator<Pair>> entries;
    std::shared_ptr<Codec> codec;
    std::function<std::uint64_t(Key const&)> key_owned_bytes;
    std::uint64_t epoch=0,stored_raw=0,live=0,read_count=0,write_count=0;
    std::uint64_t compress_us=0,lock_wait_us=0,write_us=0,read_us=0,optional_writes=0;
    std::atomic<std::uint64_t> optional_skips{0};
    static constexpr unsigned blob_limit=16u*1024u*1024u;
    static std::uint64_t checksum(std::vector<unsigned char> const& bytes){
        std::uint64_t hash=1469598103934665603ull;for(auto byte:bytes)hash=(hash^byte)*1099511628211ull;return hash;}
    void remove(typename decltype(entries)::iterator found){
        live-=found->second.blob->packed.capacity();stored_raw-=found->second.blob->raw;
        found->second.blob->retire();memory->release(found->second.key_bytes);entries.erase(found);}
    bool put_impl(Key const& key,std::vector<unsigned char> const& raw,bool optional,std::atomic<bool> const* stop){
        if(raw.empty() || raw.size()>blob_limit)return false;
        auto stopped=[&]{return stop && stop->load(std::memory_order_relaxed);};
        auto acquire=[&](std::unique_lock<std::mutex>& lock){
            auto begin=std::chrono::steady_clock::now();
            if(optional){if(stopped() || !lock.try_lock())return false;}else lock.lock();
            lock_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();return true;};
        std::shared_ptr<Codec> owned_codec;std::uint64_t generation=0;
        {std::unique_lock<std::mutex> lock(mutex,std::defer_lock);if(!acquire(lock))return false;
            if(entries.count(key))return true;if(entries.size()>=131072u)return false;
            if(!codec)codec=std::allocate_shared<Codec>(WorldStoreAllocator<Codec>(memory));
            owned_codec=codec;generation=epoch;}
        WorldStoreBuffer packed(memory);auto begin=std::chrono::steady_clock::now();
        if(!owned_codec->encode(raw,packed) || packed.bytes.empty() || packed.bytes.size()>blob_limit)return false;
        if(stopped())return false;packed.compact();
        auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();
        auto hash=checksum(raw);std::unique_lock<std::mutex> lock(mutex,std::defer_lock);if(!acquire(lock))return false;
        if(generation!=epoch || stopped())return false;
        compress_us+=elapsed;if(entries.count(key))return true;if(entries.size()>=131072u)return false;
        auto publication=std::chrono::steady_clock::now();
        auto blob=std::allocate_shared<Blob>(WorldStoreAllocator<Blob>(memory),packed,hash,unsigned(raw.size()));
        Key owned_key(key);auto extra=key_owned_bytes?key_owned_bytes(owned_key):0;
        if(!memory->reserve(extra))return false;
        try{entries.emplace(std::move(owned_key),Entry{blob,extra});}catch(...){memory->release(extra);throw;}
        live+=blob->packed.capacity();stored_raw+=raw.size();++write_count;if(optional)++optional_writes;
        write_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-publication).count();return true;
    }
public:
    struct Stats {
        std::uint64_t bytes=0,raw=0,reads=0,writes=0;unsigned records=0;
        std::uint64_t live=0,limit=0,compactions=0;
        std::uint64_t compress_us=0,lock_wait_us=0,write_us=0,read_us=0,compaction_us=0,optional_skips=0,optional_writes=0;
        std::uint64_t resident_bytes=0,in_flight_bytes=0,peak_bytes=0,pinned_bytes=0,capacity_refusals=0;
        std::uint64_t generated_world_file_reads=0,generated_world_file_writes=0;
        bool capacity_pressure=false;
    };
    explicit CompressedWorldStore(std::uint64_t capacity=1024ull*1024u*1024u,std::shared_ptr<Codec> injected={})
        :memory(std::make_shared<WorldStoreMemory>(capacity)),entries(WorldStoreAllocator<Pair>(memory)),codec(std::move(injected)){}
    ~CompressedWorldStore(){clear();}
    bool configure(std::uint64_t capacity,std::function<std::uint64_t(Key const&)> owned_bytes={}){
        std::lock_guard<std::mutex> lock(mutex);std::lock_guard<std::mutex> account(memory->mutex);
        memory->capacity_pressure=capacity<memory->resident;
        // A producer may publish after the caller samples resident bytes. Freeze
        // growth at the actual locked value instead of retaining an older cap.
        memory->limit=(std::max)(capacity,memory->resident);
        if(memory->capacity_pressure || (owned_bytes && !entries.empty() && !key_owned_bytes)){++memory->refusals;return false;}
        // A populated store retains its established key accounting contract.
        if(owned_bytes && entries.empty())key_owned_bytes=std::move(owned_bytes);return true;}
    void clear(){
        std::lock_guard<std::mutex> lock(mutex);++epoch;
        while(!entries.empty())remove(entries.begin());codec.reset();
        stored_raw=live=read_count=write_count=compress_us=lock_wait_us=write_us=read_us=optional_writes=0;optional_skips=0;
        std::lock_guard<std::mutex> account(memory->mutex);memory->peak=memory->resident+memory->in_flight;memory->refusals=0;memory->capacity_pressure=false;
    }
    Stats statistics(){
        std::lock_guard<std::mutex> lock(mutex);std::lock_guard<std::mutex> account(memory->mutex);
        Stats out;out.bytes=out.resident_bytes=memory->resident;out.raw=stored_raw;out.reads=read_count;out.writes=write_count;
        out.records=unsigned(entries.size());out.live=live;out.limit=memory->limit;
        out.compress_us=compress_us;out.lock_wait_us=lock_wait_us;out.write_us=write_us;out.read_us=read_us;
        out.optional_skips=optional_skips.load();out.optional_writes=optional_writes;
        out.in_flight_bytes=memory->in_flight;out.peak_bytes=memory->peak;out.pinned_bytes=memory->pinned;out.capacity_refusals=memory->refusals;
        out.capacity_pressure=memory->capacity_pressure;return out;}
    bool contains(Key const& key){std::lock_guard<std::mutex> lock(mutex);return entries.count(key)!=0;}
    template<class Visitor> void inspect_keys(Visitor visitor){
        std::lock_guard<std::mutex> lock(mutex);for(auto const& item:entries)visitor(item.first);}
    void invalidate(Key const& key){
        std::lock_guard<std::mutex> lock(mutex);auto found=entries.find(key);if(found!=entries.end())remove(found);}
    bool put(Key const& key,std::vector<unsigned char> const& raw){
        try{return put_impl(key,raw,false,nullptr);}catch(...){return false;}}
    bool put_optional(Key const& key,std::vector<unsigned char> const& raw,std::atomic<bool> const* stop=nullptr){
        try{if(put_impl(key,raw,true,stop))return true;}catch(...){}
        ++optional_skips;return false;}
    std::vector<unsigned char> get(Key const& key){
        std::shared_ptr<Blob const> blob;std::shared_ptr<Codec> owned_codec;std::uint64_t generation=0;
        {auto begin=std::chrono::steady_clock::now();std::lock_guard<std::mutex> lock(mutex);
            lock_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();
            auto found=entries.find(key);if(found==entries.end())return {};
            blob=found->second.blob;owned_codec=codec;generation=epoch;}
        if(!owned_codec)return {};
        auto begin=std::chrono::steady_clock::now();std::vector<unsigned char> raw(blob->raw);
        memory->flight(raw.capacity());
        struct Flight {std::shared_ptr<WorldStoreMemory> memory;std::size_t bytes;~Flight(){memory->unflight(bytes);}} flight{memory,raw.capacity()};
        if(!owned_codec->decode(blob->packed,raw,*memory) || checksum(raw)!=blob->hash)return {};
        {std::lock_guard<std::mutex> lock(mutex);if(generation==epoch){
            ++read_count;read_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();}}
        return raw; // Returned decoded bytes transfer to caller ownership.
    }
};
}}
