#pragma once
#include <windows.h>
#include "journal.h"
#include "assets.h"
#include <memory>
#include <map>

namespace c3x_inputs {
// The capture service is process-lifetime only. Pin this DLL while its optional
// writer thread exists; normal rendering neither pins a module nor starts I/O.
class ReplayAssets {
    std::mutex mutex;std::map<std::string,Asset> expected;std::string failure;std::size_t names=0;
public:
    std::atomic<bool> enabled{false};
    void begin(){std::lock_guard<std::mutex> lock(mutex);require(!enabled,"replay assets already initialized");enabled=true;}
    void add(std::string name,Asset value){std::lock_guard<std::mutex> lock(mutex);require(expected.size()<8192&&name.size()<4u*1024u*1024u-names,"replay asset budget");
        names+=name.size();require(expected.emplace(std::move(name),value).second,"duplicate replay asset");}
    void verify(std::string const& name,Asset const& value){std::lock_guard<std::mutex> lock(mutex);auto found=expected.find(name);
        require(found!=expected.end(),"consumed asset missing from capture");require(found->second==value,"consumed asset differs from capture");}
    void fail(char const* message){std::lock_guard<std::mutex> lock(mutex);if(failure.empty())failure=message;}
    void check(){std::lock_guard<std::mutex> lock(mutex);if(!failure.empty())throw std::runtime_error(failure);}
};
inline ReplayAssets& replay_assets(){static ReplayAssets value;return value;}
inline void runtime_anchor(){}
using Settings=std::vector<std::pair<std::string,std::string>>;
inline Settings input_settings(){
    Settings result;auto environment=GetEnvironmentStringsA();require(environment!=nullptr,"input settings unavailable");
    try{for(auto item=environment;*item;item+=std::strlen(item)+1){std::string setting=item;auto equal=setting.find('=');
        if(setting.rfind("C3X_RENDERER_",0)==0&&equal!=std::string::npos&&setting.rfind("C3X_RENDERER_INPUT_",0)!=0)
            result.emplace_back(setting.substr(0,equal),setting.substr(equal+1));}}
    catch(...){FreeEnvironmentStringsA(environment);throw;}FreeEnvironmentStringsA(environment);
    require(result.size()<=512,"input environment limit");std::sort(result.begin(),result.end());return result;
}
inline void settings_fields(Writer& out,Settings const& settings){out.u32(unsigned(settings.size()));
    for(auto const& setting:settings){out.string(setting.first.c_str(),256);out.string(setting.second.c_str(),32768);}}
class Runtime {
    Settings settings;std::mutex settings_mutex;
    std::unique_ptr<Journal> journal;std::atomic<std::uint64_t> tokens{0},gameplay_start{0};
    std::mutex assets_mutex;std::map<std::string,Asset> assets;std::size_t asset_names=0;
    std::mutex identities;std::map<void*,std::uint32_t> objects;std::map<DWORD,std::uint32_t> threads;std::uint32_t next_object=0;
    std::uint64_t origin=0,frequency=1;std::atomic<unsigned> producers{0};std::mutex admission;unsigned open_calls=0;
public:
    Runtime()noexcept{try{
        wchar_t path[32768]={};auto n=GetEnvironmentVariableW(L"C3X_RENDERER_INPUT_RECORD_DIR",path,32768);
        if(!n||n>=32768)return;
        LARGE_INTEGER q={},f={},after={};FILETIME utc={};QueryPerformanceCounter(&q);
        auto precise=reinterpret_cast<void(WINAPI*)(LPFILETIME)>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"),"GetSystemTimePreciseAsFileTime"));
        if(precise)precise(&utc);else GetSystemTimeAsFileTime(&utc);
        QueryPerformanceCounter(&after);QueryPerformanceFrequency(&f);origin=std::uint64_t(q.QuadPart);frequency=std::uint64_t(f.QuadPart);
        ClockOrigin correlation{origin,std::uint64_t(utc.dwLowDateTime)|(std::uint64_t(utc.dwHighDateTime)<<32),std::uint64_t(after.QuadPart),precise?1u:0u};
        HMODULE module=nullptr;
        require(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<wchar_t const*>(&runtime_anchor),&module)!=0,"cannot pin input recorder module");
        journal=std::make_unique<Journal>(path,frequency,Limits{},[this](Journal& writer){
            std::lock_guard<std::mutex> lock(admission);
            if(!open_calls&&expired())writer.stop(Stop::duration);
        });
        Writer manifest;manifest.u32(protocol_version);manifest.u32(C3X_RENDERER_API_VERSION);
        // Coverage is declared explicitly; adding a packet is not proof that
        // every native/ambient producer has reached the replay boundary.
        manifest.u64(0);
        clock_origin(manifest,correlation);
        settings=input_settings();settings_fields(manifest,settings);
        journal->emit(Kind::manifest,0,std::move(manifest.bytes));
        wchar_t binary[32768]={};auto length=GetModuleFileNameW(module,binary,32768);require(length&&length<32768,"input module path failed");asset(binary,nullptr,0,false,1);
        OutputDebugStringA("[C3X renderer] input recording started; qualification required\n");
    }catch(...){journal.reset();OutputDebugStringA("[C3X renderer] input recording could not start\n");}}
    void configuration()noexcept{if(!active())return;try{std::lock_guard<std::mutex> lock(settings_mutex);auto current=input_settings();
        if(current!=settings){emit(Kind::settings,0,[&](Writer& out){settings_fields(out,current);});settings=std::move(current);}}
        catch(std::bad_alloc const&){stop(Stop::allocation_failure);}catch(...){stop(Stop::unsupported);}}
    bool active()const{return journal&&journal->active();}
    std::uint64_t now()const{LARGE_INTEGER q={};QueryPerformanceCounter(&q);return std::uint64_t(q.QuadPart)-origin;}
    std::uint64_t token(){return ++tokens;}
    std::uint32_t object(void* pointer){if(!pointer)return 0;std::lock_guard<std::mutex> lock(identities);
        auto found=objects.find(pointer);if(found!=objects.end())return found->second;
        require(objects.size()<8192&&next_object<UINT32_MAX,"native input identity limit");auto id=++next_object;objects.emplace(pointer,id);return id;}
    std::uint32_t thread(){std::lock_guard<std::mutex> lock(identities);auto current=GetCurrentThreadId();auto found=threads.find(current);if(found!=threads.end())return found->second;
        require(threads.size()<64,"native input thread limit");auto id=unsigned(threads.size()+1);threads.emplace(current,id);return id;}
    void retire(void* pointer){if(!active())return;std::lock_guard<std::mutex> lock(identities);objects.erase(pointer);}
    bool expired()const{auto start=gameplay_start.load();if(!start)return false;auto tick=now();return tick>=start-1&&tick-(start-1)>=600*frequency;}
    bool begin_call(){if(!active())return false;std::lock_guard<std::mutex> lock(admission);if(!active())return false;
        if(!open_calls&&expired()){journal->stop(Stop::duration);return false;}++open_calls;return true;}
    void end_call(){std::lock_guard<std::mutex> lock(admission);if(open_calls)--open_calls;
        if(!open_calls&&active()&&expired())journal->stop(Stop::duration);}
    void gameplay(){if(!active())return;std::uint64_t empty=0;gameplay_start.compare_exchange_strong(empty,now()+1);}
    template<class Encode>void emit(Kind kind,std::uint32_t flags,Encode encode)noexcept{
        if(!active())return;
        unsigned active=producers.fetch_add(1);struct Producer {std::atomic<unsigned>& count;~Producer(){--count;}} producer{producers};
        if(active>=2){journal->stop(Stop::producer_limit);return;}
        try{auto ticks=now();
            Writer out;encode(out);journal->emit(kind,ticks,std::move(out.bytes),flags);
        }catch(std::bad_alloc const&){journal->stop(Stop::allocation_failure);}
        catch(...){journal->stop(Stop::unsupported);OutputDebugStringA("[C3X renderer] input recording stopped at unsupported payload\n");}
    }
    template<class Path>void asset(Path path,void const* data,std::size_t bytes,bool consumed,unsigned purpose=0)noexcept{
        if(!active()&&!replay_assets().enabled.load(std::memory_order_relaxed))return;
        try{auto name=asset_path(path);Asset value;
            if(consumed){value.exists=true;value.size=bytes;Sha256 digest;digest.add(data,bytes);value.hash=digest.finish();}
            else value=asset_file(name);
            if(replay_assets().enabled){replay_assets().verify(name,value);return;}
            std::lock_guard<std::mutex> lock(assets_mutex);auto found=assets.find(name);
            if(found!=assets.end()){require(found->second==value,"asset changed during input capture");return;}
            require(assets.size()<8192&&name.size()<4u*1024u*1024u-asset_names,"asset manifest budget");asset_names+=name.size();assets.emplace(name,value);
            emit(Kind::asset,purpose,[&](Writer& out){asset_fields(out,name,value);});
        }catch(std::exception const& error){if(replay_assets().enabled)replay_assets().fail(error.what());else stop(Stop::unsupported);}
        catch(...){if(replay_assets().enabled)replay_assets().fail("asset verification failed");else stop(Stop::unsupported);}
    }
    void stop(Stop why){if(journal)journal->stop(why);}
    void finish(){if(journal)journal->finish(Stop::closed);}
};
struct ReplayExecution {
    bool performance=false;int result=0;double service_ms=0;bool reused_adoption=false;
    long long last_ticks=0,last_frequency=1;
};
inline ReplayExecution& replay_execution(){thread_local ReplayExecution state;return state;}
template<class F>int measure_replay(F run){LARGE_INTEGER start={},end={},frequency={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&start);int result=run();QueryPerformanceCounter(&end);replay_execution().service_ms+=1000.*double(end.QuadPart-start.QuadPart)/double(frequency.QuadPart);return result;}
struct ReplayClock {
    std::vector<std::pair<std::int64_t,std::int64_t>> values;std::size_t at=0;
    char const* failure=nullptr;
    // Missing diagnostic input must not throw through production clock callers.
    // Preserve their current values and reject at the replay entry boundary.
    bool sample(long long& ticks,long long& frequency){
        if(replay_execution().performance){auto& execution=replay_execution();if(!values.empty()){execution.last_ticks=values.front().first;execution.last_frequency=values.front().second;}ticks=execution.last_ticks;frequency=execution.last_frequency;at=values.size();return true;}
        if(at>=values.size()){failure="replay missing consumed clock input";return false;}
        ticks=values[at].first;frequency=values[at++].second;require(ticks>=0&&frequency>0,"invalid replay clock");return true;}
};
inline ReplayClock*& replay_clock(){thread_local ReplayClock* clock=nullptr;return clock;}
inline Runtime& runtime(){static Runtime* service=new Runtime;return *service;}
// Parent tokens distinguish native API calls from nested GPU/clock calls. The
// timestamp belongs to input arrival, never asynchronous file completion.
inline std::uint64_t& parent_token(){thread_local std::uint64_t parent=0;return parent;}
inline bool& native_root(){thread_local bool value=false;return value;}
struct Call {
    std::uint64_t id=0,parent=0;bool completed=false;
    template<class Encode>Call(Kind kind,std::uint32_t subtype,Encode encode){
        if(native_root())return;
        auto& service=runtime();if(!parent_token())service.configuration();if(!service.begin_call())return;
        id=service.token();parent=parent_token();parent_token()=id;
        service.emit(kind,subtype,[&](Writer& out){out.u64(id);out.u64(parent);encode(out);});
    }
    ~Call(){if(id){if(!completed)runtime().stop(Stop::unsupported);parent_token()=parent;runtime().end_call();}}
    template<class Encode>int result(int code,Encode encode){
        if(id)runtime().emit(Kind::result,0,[&](Writer& out){out.u64(id);out(std::int32_t(code));encode(out);});completed=true;return code;
    }
    int result(int code){return result(code,[](Writer&){});}
};
}
