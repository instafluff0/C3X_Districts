#pragma once
#include "codec.h"
#include "../asset_content_hash.h"
#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <chrono>
#include <functional>

namespace c3x_inputs {
enum class Kind:std::uint32_t { manifest=1,configuration,scene,world_page,unit,unit_forget,
    camera,native_operation,native_snapshot,image_commands,tactical,visual,presentation,result,reset,footer,asset,settings };
enum class Stop:std::uint32_t { closed=0,duration,queue_full,byte_limit,io_error,unsupported,allocation_failure,producer_limit };
struct Limits {
    std::size_t queue_bytes=32u*1024u*1024u;
    std::uint64_t segment_bytes=128ull*1024*1024,total_bytes=8ull*1024*1024*1024;
    // Test-only deterministic slow-writer fault; production keeps zero.
    unsigned writer_delay_ms=0;
};
struct Event {Kind kind;std::uint32_t flags=0;std::uint64_t sequence=0,ticks=0;Bytes payload;};
class Journal {
    std::filesystem::path directory;Limits limits;std::uint64_t frequency;
    std::mutex mutex;std::condition_variable wake;std::deque<Event> queue;std::thread writer;
    bool stopping=false;Stop reason=Stop::closed;
    std::uint64_t sequence=0,queued_bytes=0,high_water=0,written=0,events=0,segments=0;
    std::atomic<bool> enabled{true};
    std::function<void(Journal&)> idle_watchdog;
    void write_loop()noexcept{
        try{
            std::ofstream stream,index(directory/"index.jsonl",std::ios::out|std::ios::trunc);
            require(bool(index),"input index open failed");std::uint64_t extent=0,last_second=UINT64_MAX,last_sequence=0;
            auto segment=[&]{
                if(stream.is_open()){stream.flush();require(bool(stream),"input segment flush failed");stream.close();}
                char name[64];std::snprintf(name,sizeof(name),"segment-%06llu.c3xi",static_cast<unsigned long long>(segments));
                stream.open(directory/name,std::ios::binary|std::ios::out|std::ios::trunc);require(bool(stream),"input segment open failed");
                Writer head;head.u32(0x49583343);head.u32(1);head.u64(segments++);head.u64(frequency);head.u64(last_sequence);
                stream.write(reinterpret_cast<char const*>(head.bytes.data()),std::streamsize(head.bytes.size()));extent=head.bytes.size();written+=extent;
            };
            auto write=[&](Event const& item){
                if(!stream.is_open()||extent+48+item.payload.size()>limits.segment_bytes)segment();
                Writer head;head.u32(0x45583343);head.u32(std::uint32_t(item.kind));head.u32(std::uint32_t(item.payload.size()));head.u32(item.flags);
                head.u64(item.sequence);head.u64(item.ticks);
                auto hash=c3x_renderer::asset_content_hash(item.payload.data(),item.payload.size());
                auto metadata=c3x_renderer::asset_content_hash(head.bytes.data(),head.bytes.size());
                for(unsigned i=0;i<4;++i)head.u32(hash[i]^metadata[i]);
                auto offset=extent;stream.write(reinterpret_cast<char const*>(head.bytes.data()),std::streamsize(head.bytes.size()));
                if(!item.payload.empty())stream.write(reinterpret_cast<char const*>(item.payload.data()),std::streamsize(item.payload.size()));
                require(bool(stream),"input segment write failed");extent+=head.bytes.size()+item.payload.size();written+=head.bytes.size()+item.payload.size();
                last_sequence=item.sequence;++events;auto second=item.ticks/frequency;
                if(item.kind!=Kind::footer&&(second!=last_second||item.kind==Kind::presentation)){
                    index<<"{\"sequence\":"<<item.sequence<<",\"ticks\":"<<item.ticks<<",\"segment\":"<<(segments-1)<<",\"offset\":"<<offset<<",\"kind\":"<<std::uint32_t(item.kind)<<"}\n";
                    last_second=second;
                }
                // Flush on the writer, never on the game caller. A crash can
                // lose the incomplete tail; checksummed prior records survive.
                stream.flush();index.flush();require(bool(stream)&&bool(index),"input flush failed");
            };
            for(;;){
                Event item;bool idle=false;
                {std::unique_lock<std::mutex> lock(mutex);
                    if(idle_watchdog)wake.wait_for(lock,std::chrono::milliseconds(100),[&]{return stopping||!queue.empty();});
                    else wake.wait(lock,[&]{return stopping||!queue.empty();});
                    if(queue.empty()){if(stopping)break;idle=true;}
                    else{item=std::move(queue.front());queue.pop_front();}}
                // The runtime admission gate may call stop(). Never hold the
                // writer mutex across it, and never add another recorder thread.
                if(idle){idle_watchdog(*this);continue;}
                if(limits.writer_delay_ms)std::this_thread::sleep_for(std::chrono::milliseconds(limits.writer_delay_ms));
                if(written+item.payload.size()+256>limits.total_bytes){
                    std::lock_guard<std::mutex> lock(mutex);reason=Stop::byte_limit;stopping=true;enabled=false;queue.clear();queued_bytes=0;break;
                }
                write(item);
                {std::lock_guard<std::mutex> lock(mutex);queued_bytes-=48+item.payload.capacity();}
            }
            Writer end;
            {std::lock_guard<std::mutex> lock(mutex);end.u32(std::uint32_t(reason));end.u64(events);end.u64(high_water);}
            // Footer sequence follows actually persisted input, including when
            // quota rejection discarded queued input. Its reason denies completeness.
            write({Kind::footer,0,last_sequence+1,0,std::move(end.bytes)});
            stream.close();index.close();
            std::ofstream manifest(directory/"finished.json",std::ios::out|std::ios::trunc);
            manifest<<"{\"schema\":1,\"segments\":"<<segments<<",\"events\":"<<events<<",\"bytes\":"<<written
                <<",\"stop_reason\":"<<std::uint32_t(reason)<<",\"queue_high_water_bytes\":"<<high_water<<"}\n";
            manifest.flush();require(bool(manifest),"input completion manifest failed");
        }catch(...){std::lock_guard<std::mutex> lock(mutex);reason=Stop::io_error;stopping=true;enabled=false;queue.clear();queued_bytes=0;}
    }
public:
    Journal(std::filesystem::path root,std::uint64_t ticks_per_second,Limits options={},std::function<void(Journal&)> watchdog={}):directory(std::move(root)),limits(options),frequency(ticks_per_second),idle_watchdog(std::move(watchdog)){
        require(frequency&&frequency<1000000000000ull,"invalid input clock");
        require(limits.queue_bytes>=48&&limits.segment_bytes>=128&&limits.total_bytes>=limits.segment_bytes,"invalid recording limits");
        require(!std::filesystem::exists(directory)&&std::filesystem::create_directories(directory),"input capture directory already exists or cannot be created");
        require(std::filesystem::space(directory).available>=limits.total_bytes+64u*1024u*1024u,"insufficient input recording disk space");
        std::ofstream manifest(directory/"started.json");manifest<<"{\"schema\":1,\"frequency\":"<<frequency<<",\"queue_limit_bytes\":"<<limits.queue_bytes<<",\"segment_limit_bytes\":"<<limits.segment_bytes<<",\"session_limit_bytes\":"<<limits.total_bytes<<"}\n";
        manifest.flush();require(bool(manifest),"input manifest write failed");writer=std::thread([this]{write_loop();});
    }
    Journal(Journal const&)=delete;Journal& operator=(Journal const&)=delete;
    ~Journal(){finish(Stop::closed);}
    bool active()const{return enabled.load(std::memory_order_relaxed);}
    bool emit(Kind kind,std::uint64_t ticks,Bytes payload,std::uint32_t flags=0)noexcept{
        try{
            std::lock_guard<std::mutex> lock(mutex);if(stopping)return false;
            if(kind==Kind::footer||std::uint32_t(kind)<1||std::uint32_t(kind)>std::uint32_t(Kind::settings)){reason=Stop::unsupported;stopping=true;enabled=false;wake.notify_one();return false;}
            auto bytes=48+payload.capacity();
            if(payload.size()>payload_limit||bytes>limits.queue_bytes-queued_bytes){reason=Stop::queue_full;stopping=true;enabled=false;wake.notify_one();return false;}
            queue.push_back({kind,flags,++sequence,ticks,std::move(payload)});queued_bytes+=bytes;high_water=std::max(high_water,queued_bytes);wake.notify_one();return true;
        }catch(...){stop(Stop::allocation_failure);return false;}
    }
    void stop(Stop why)noexcept{std::lock_guard<std::mutex> lock(mutex);if(!stopping){reason=why;stopping=true;enabled=false;}wake.notify_one();}
    void finish(Stop why){stop(why);if(writer.joinable())writer.join();}
    Stop stop_reason(){std::lock_guard<std::mutex> lock(mutex);return reason;}
};
struct SegmentReader {
    std::filesystem::path directory;std::ifstream stream;std::uint64_t segment=0,sequence=0,frequency=0,total_segments=0,current_segment=0;
    bool footer=false;Stop reason=Stop::unsupported;
    explicit SegmentReader(std::filesystem::path root):directory(std::move(root)){
        require(std::filesystem::is_directory(directory),"missing input directory");
        std::uint64_t count=0;for(auto const& entry:std::filesystem::directory_iterator(directory))
            if(entry.path().extension()==".c3xi")++count;
        require(count>0&&count<=65536,"missing or excessive input segments");total_segments=count;
        for(std::uint64_t n=0;n<count;++n){char name[64];std::snprintf(name,sizeof(name),"segment-%06llu.c3xi",static_cast<unsigned long long>(n));
            require(std::filesystem::is_regular_file(directory/name),"missing input segment");}
        require(open(),"missing first input segment");
    }
    void read(Bytes& bytes,std::size_t count){bytes.resize(count);stream.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(count));require(std::size_t(stream.gcount())==count,"truncated input segment");}
    bool open(){
        char name[64];std::snprintf(name,sizeof(name),"segment-%06llu.c3xi",static_cast<unsigned long long>(segment));
        auto path=directory/name;if(!std::filesystem::exists(path))return false;
        current_segment=segment;stream.close();stream.clear();stream.open(path,std::ios::binary);require(bool(stream),"cannot open input segment");
        Bytes bytes;read(bytes,32);Reader header{bytes};require(header.u32()==0x49583343&&header.u32()==1,"unsupported input journal");
        require(header.u64()==segment++,"input segment order mismatch");auto f=header.u64();require(f&&(!frequency||f==frequency),"input clock mismatch");frequency=f;
        require(header.u64()==sequence,"input segment chain mismatch");return true;
    }
    bool next_verified(Event& item,bool allow_prefix,bool& truncated){
        try{return next(item);}catch(std::runtime_error const& error){
            if(allow_prefix&&current_segment+1==total_segments&&std::string(error.what())=="truncated input segment"){truncated=true;return false;}throw;
        }
    }
    bool next(Event& item){
        require(!footer,"read after input footer");
        if(stream.peek()==std::char_traits<char>::eof()&&!open())return false;
        Bytes bytes;read(bytes,48);Reader header{bytes};require(header.u32()==0x45583343,"invalid input event marker");
        auto kind=header.u32();require(kind>=1&&kind<=std::uint32_t(Kind::settings),"unknown input event");item.kind=Kind(kind);
        auto count=header.u32();require(count<=payload_limit,"oversized input event");item.flags=header.u32();item.sequence=header.u64();item.ticks=header.u64();
        require(item.sequence==++sequence,"input sequence gap");std::array<std::uint32_t,4> hash;for(auto& part:hash)part=header.u32();
        read(item.payload,count);auto actual=c3x_renderer::asset_content_hash(item.payload.data(),item.payload.size());
        auto metadata=c3x_renderer::asset_content_hash(bytes.data(),32);for(unsigned i=0;i<4;++i)actual[i]^=metadata[i];
        require(actual==hash,"input checksum mismatch");
        if(item.kind==Kind::footer){Reader end{item.payload};auto why=end.u32();require(why<=std::uint32_t(Stop::producer_limit),"invalid input footer");reason=Stop(why);
            require(end.u64()+1==sequence,"input footer count mismatch");end.u64();end.done();footer=true;
            require(stream.peek()==std::char_traits<char>::eof(),"data after input footer");
            char name[64];std::snprintf(name,sizeof(name),"segment-%06llu.c3xi",static_cast<unsigned long long>(segment));
            require(!std::filesystem::exists(directory/name),"segment after input footer");}
        return true;
    }
};
}
