// External diagnostic for Civ III music skips (Renderer/docs/audio_skips.md).
// Civ III's sound.dll mixes all audio (music included) in a 30 ms periodic
// multimedia-timer callback that holds one critical section and stores the
// timeGetTime() of each service; it keeps ~270 ms mixed ahead of playback.
// This x86 process polls that timestamp every millisecond from outside the
// game and writes each service interval longer than the threshold with the
// probe's own longest poll delay over it: a pause that also stopped this
// time-critical thread is guest-wide (VM/host scheduling), one that did not is
// specific to the game. Stalls past 75 ms and callbacks holding the lock for
// 15 ms or more record thread states, the wait chain and validated call sites
// of the timer thread and Civ III's main thread. Per second it writes CPU and
// memory of the system, the game and the Renderer64 helper.
// It never writes game memory; threads are suspended only for context reads,
// except --hold-test, which holds game threads on purpose (disposable games).
//   OUT.txt [seconds] [threshold_ms]              monitor the running game
//   OUT.txt seconds threshold --no-game           only this thread's poll delays
//   OUT.txt seconds threshold --buffers           list sound.dll's DirectSound buffers
//   OUT.txt seconds threshold --hold-test         hold main/timer threads for fixed times
//   OUT.txt seconds threshold --record            loopback-record the default output
// Renderer/tools/sound_probe_report.py summarizes the log and a recording.
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#include <share.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <winternl.h>
#include <pdh.h>
#include <wct.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#pragma comment(lib,"winmm.lib")
#pragma comment(lib,"psapi.lib")
#pragma comment(lib,"ole32.lib")
#pragma comment(lib,"pdh.lib")
#pragma comment(lib,"advapi32.lib")

// sound.dll 2003-09-26 (TimeDateStamp 0x3f74c977): the timer object pointer
// and the callback's critical section, followed by its reentrancy flag.
constexpr std::uint32_t sound_stamp=0x3f74c977,timer_object=0xb3a64,timer_lock=0xb3a08;
struct Module {std::uint32_t base=0,end=0;std::string name;};
HANDLE game=nullptr;std::vector<Module> modules;
bool read(std::uint32_t address,void* out,std::size_t size){
    SIZE_T got=0;return ReadProcessMemory(game,reinterpret_cast<void const*>(std::uintptr_t(address)),out,size,&got)&&got==size;
}
DWORD find_process(wchar_t const* name){
    DWORD pid=0;HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);PROCESSENTRY32W entry={sizeof(entry)};
    for(BOOL more=Process32FirstW(snapshot,&entry);more;more=Process32NextW(snapshot,&entry))
        if(!_wcsicmp(entry.szExeFile,name)){pid=entry.th32ProcessID;break;}
    CloseHandle(snapshot);return pid;
}
void load_modules(){
    modules.clear();HMODULE handles[512];DWORD bytes=0;
    if(!EnumProcessModulesEx(game,handles,sizeof(handles),&bytes,LIST_MODULES_ALL))return;
    for(DWORD n=0;n<bytes/sizeof(HMODULE)&&n<512;++n){
        MODULEINFO info={};char name[MAX_PATH]={};
        if(!GetModuleInformation(game,handles[n],&info,sizeof(info)))continue;
        GetModuleBaseNameA(game,handles[n],name,MAX_PATH);
        auto base=std::uint32_t(std::uintptr_t(info.lpBaseOfDll));modules.push_back({base,base+info.SizeOfImage,name});
    }
}
Module const* module_of(std::uint32_t address){
    for(auto const& m:modules)if(address>=m.base&&address<m.end)return &m;
    return nullptr;
}
std::string label(std::uint32_t address){
    char text[96];auto m=module_of(address);
    if(m)std::snprintf(text,sizeof(text),"%s+0x%x",m->name.c_str(),unsigned(address-m->base));
    else std::snprintf(text,sizeof(text),"0x%08x",unsigned(address));
    return text;
}
// A return address follows a direct (E8 rel32) or indirect (FF /2) call.
bool call_site(std::uint32_t address){
    auto m=module_of(address);unsigned char b[6];
    if(!m||address<m->base+0x1000||!read(address-6,b,6))return false;
    return b[1]==0xE8||(b[0]==0xFF&&(b[1]&0x38)==0x10)||(b[3]==0xFF&&(b[4]&0x38)==0x10)||(b[4]==0xFF&&(b[5]&0x38)==0x10);
}
// EIP and up to 14 validated call sites from the first 4 KB of the stack.
std::string sample_thread(DWORD tid){
    HANDLE thread=OpenThread(THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT|THREAD_QUERY_INFORMATION,FALSE,tid);
    if(!thread)return "open-failed";
    std::uint32_t stack[1024]={};unsigned words=0;CONTEXT context={};context.ContextFlags=CONTEXT_CONTROL;
    bool ok=SuspendThread(thread)!=DWORD(-1);
    if(ok){
        ok=GetThreadContext(thread,&context)!=FALSE;
        if(ok)for(;words<1024;words+=128)if(!read(context.Esp+words*4,stack+words,128*4))break;
        ResumeThread(thread);
    }
    CloseHandle(thread);
    if(!ok)return "context-failed";
    std::string text="eip="+label(context.Eip);unsigned found=0;
    for(unsigned n=0;n<words&&found<14;++n)if(call_site(stack[n])){text+=" <"+label(stack[n]);++found;}
    return text;
}
DWORD main_thread(DWORD pid){
    DWORD best=0;ULONGLONG earliest=~0ull;HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0);THREADENTRY32 entry={sizeof(entry)};
    for(BOOL more=Thread32First(snapshot,&entry);more;more=Thread32Next(snapshot,&entry)){
        if(entry.th32OwnerProcessID!=pid)continue;
        HANDLE thread=OpenThread(THREAD_QUERY_LIMITED_INFORMATION,FALSE,entry.th32ThreadID);if(!thread)continue;
        FILETIME created,exited,kernel,user;
        if(GetThreadTimes(thread,&created,&exited,&kernel,&user)){
            ULONGLONG at=(ULONGLONG(created.dwHighDateTime)<<32)|created.dwLowDateTime;
            if(at<earliest){earliest=at;best=entry.th32ThreadID;}}
        CloseHandle(thread);
    }
    CloseHandle(snapshot);return best;
}
// Scheduler state, wait reason and current priority of two game threads
// (KTHREAD_STATE 1 ready, 2 running, 5 waiting; KWAIT_REASON 9 WrPageIn,
// 5 Suspended, 6 UserRequest, 15 WrEventPair/queue..., from the NT snapshot).
std::string thread_states(DWORD pid,DWORD first,DWORD second){
    using Query=LONG(NTAPI*)(ULONG,void*,ULONG,ULONG*);
    static auto query_system=reinterpret_cast<Query>(GetProcAddress(GetModuleHandleA("ntdll.dll"),"NtQuerySystemInformation"));
    std::vector<unsigned char> buffer(1<<20);ULONG got=0;LONG status;
    while((status=query_system(5,buffer.data(),ULONG(buffer.size()),&got))==LONG(0xC0000004))buffer.resize(buffer.size()*2);
    if(status<0)return "query-failed";
    std::string text;
    for(auto p=reinterpret_cast<SYSTEM_PROCESS_INFORMATION const*>(buffer.data());;){
        if(DWORD(std::uintptr_t(p->UniqueProcessId))==pid){
            auto threads=reinterpret_cast<SYSTEM_THREAD_INFORMATION const*>(p+1);
            for(ULONG n=0;n<p->NumberOfThreads;++n){
                auto tid=DWORD(std::uintptr_t(threads[n].ClientId.UniqueThread));
                if(tid!=first&&tid!=second)continue;
                char t[96];std::snprintf(t,sizeof(t),"%s%lu:state=%lu,wait=%lu,priority=%ld",text.empty()?"":" ",tid,
                    threads[n].ThreadState,threads[n].WaitReason,threads[n].Priority);text+=t;
            }
            break;
        }
        if(!p->NextEntryOffset)break;
        p=reinterpret_cast<SYSTEM_PROCESS_INFORMATION const*>(reinterpret_cast<unsigned char const*>(p)+p->NextEntryOffset);
    }
    return text;
}
// Wait Chain Traversal: what a blocked thread waits on (critical section,
// SendMessage, mutex, ALPC, COM...) and which thread owns it.
std::string wait_chain(DWORD tid){
    HWCT session=OpenThreadWaitChainSession(0,nullptr);if(!session)return "wct-open-failed";
    WAITCHAIN_NODE_INFO nodes[WCT_MAX_NODE_COUNT];DWORD count=WCT_MAX_NODE_COUNT;BOOL cycle=FALSE;
    std::string text;
    if(!GetThreadWaitChain(session,0,WCTP_GETINFO_ALL_FLAGS,tid,&count,nodes,&cycle))text="wct-failed-"+std::to_string(GetLastError());
    else for(DWORD n=0;n<count;++n){
        char t[192];auto const& node=nodes[n];
        if(node.ObjectType==WctThreadType)
            std::snprintf(t,sizeof(t),"%sthread(pid=%lu,tid=%lu,status=%d)",n?" > ":"",node.ThreadObject.ProcessId,node.ThreadObject.ThreadId,int(node.ObjectStatus));
        else{
            char name[64]={};WideCharToMultiByte(CP_UTF8,0,node.LockObject.ObjectName,-1,name,sizeof(name)-1,nullptr,nullptr);
            std::snprintf(t,sizeof(t),"%sobject(type=%d,status=%d,name=%s)",n?" > ":"",int(node.ObjectType),int(node.ObjectStatus),name);
        }
        text+=t;
    }
    CloseThreadWaitChainSession(session);return text;
}
ULONGLONG filetime(FILETIME t){return (ULONGLONG(t.dwHighDateTime)<<32)|t.dwLowDateTime;}
ULONGLONG process_cpu(HANDLE process){
    FILETIME c,e,k,u;return process&&GetProcessTimes(process,&c,&e,&k,&u)?filetime(k)+filetime(u):0;
}
long long qpc(){LARGE_INTEGER v;QueryPerformanceCounter(&v);return v.QuadPart;}
// Suspends one game thread for the given time and reports the QPC interval.
void hold_thread(std::FILE* out,char const* name,DWORD tid,DWORD milliseconds){
    HANDLE thread=OpenThread(THREAD_SUSPEND_RESUME,FALSE,tid);if(!thread)return;
    auto begin=qpc();SuspendThread(thread);Sleep(milliseconds);ResumeThread(thread);auto end=qpc();CloseHandle(thread);
    std::fprintf(out,"hold thread=%s tid=%lu ms=%lu begin_qpc=%lld end_qpc=%lld\n",name,tid,milliseconds,begin,end);std::fflush(out);
}
// --record: shared-mode loopback capture of the default render endpoint.
// OUT.raw holds the interleaved mix-format samples; OUT lists the format and,
// per packet, its first frame, QPC position (100 ns) and buffer flags.
int record(char const* path,double seconds){
    if(FAILED(CoInitializeEx(nullptr,COINIT_MULTITHREADED)))return 1;
    IMMDeviceEnumerator* devices=nullptr;IMMDevice* device=nullptr;IAudioClient* client=nullptr;IAudioCaptureClient* capture=nullptr;WAVEFORMATEX* format=nullptr;
    if(FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator),nullptr,CLSCTX_ALL,IID_PPV_ARGS(&devices)))||
       FAILED(devices->GetDefaultAudioEndpoint(eRender,eConsole,&device))||
       FAILED(device->Activate(__uuidof(IAudioClient),CLSCTX_ALL,nullptr,reinterpret_cast<void**>(&client)))||
       FAILED(client->GetMixFormat(&format))||
       FAILED(client->Initialize(AUDCLNT_SHAREMODE_SHARED,AUDCLNT_STREAMFLAGS_LOOPBACK,10000000,0,format,nullptr))||
       FAILED(client->GetService(IID_PPV_ARGS(&capture)))||FAILED(client->Start()))return 1;
    std::FILE* text=_fsopen(path,"w",_SH_DENYNO);std::string raw_path=std::string(path)+".raw";std::FILE* raw=_fsopen(raw_path.c_str(),"wb",_SH_DENYNO);
    if(!text||!raw)return 1;
    LARGE_INTEGER frequency;QueryPerformanceFrequency(&frequency);
    std::fprintf(text,"format tag=%u channels=%u rate=%lu bits=%u block=%u qpc_frequency=%lld\n",format->wFormatTag,format->nChannels,
        format->nSamplesPerSec,format->wBitsPerSample,format->nBlockAlign,frequency.QuadPart);
    auto end=qpc()+(long long)(seconds*double(frequency.QuadPart));unsigned long long frames_total=0;
    while(qpc()<end){
        Sleep(5);UINT32 packet=0;
        while(SUCCEEDED(capture->GetNextPacketSize(&packet))&&packet){
            BYTE* data=nullptr;UINT32 frames=0;DWORD flags=0;UINT64 position=0,qpc_position=0;
            if(FAILED(capture->GetBuffer(&data,&frames,&flags,&position,&qpc_position)))break;
            std::vector<BYTE> silence;if(flags&AUDCLNT_BUFFERFLAGS_SILENT){silence.assign(std::size_t(frames)*format->nBlockAlign,0);data=silence.data();}
            std::fwrite(data,format->nBlockAlign,frames,raw);
            std::fprintf(text,"packet frame=%llu frames=%u qpc100ns=%llu flags=%lu\n",frames_total,frames,qpc_position,flags);
            frames_total+=frames;capture->ReleaseBuffer(frames);
        }
    }
    client->Stop();std::fclose(raw);std::fclose(text);return 0;
}

int main(int argc,char** argv){
    if(argc<2){std::fprintf(stderr,"usage: sound_service_probe OUT.txt [seconds] [threshold_ms] [--no-game|--buffers|--hold-test|--record]\n");return 2;}
    double seconds=argc>2?std::atof(argv[2]):420;unsigned threshold=argc>3?unsigned(std::atoi(argv[3])):60;
    bool no_game=argc>4&&!std::strcmp(argv[4],"--no-game");
    if(argc>4&&!std::strcmp(argv[4],"--record"))return record(argv[1],seconds);
    std::FILE* out=_fsopen(argv[1],"w",_SH_DENYNO);if(!out)return 2;
    LARGE_INTEGER frequency;QueryPerformanceFrequency(&frequency);
    auto ms=[&](long long ticks){return double(ticks)*1000.0/double(frequency.QuadPart);};
    auto start=qpc();auto elapsed=[&]{return ms(qpc()-start)/1000.0;};
    DWORD pid=0,game_main=0;std::uint32_t sound=0,object=0;
    if(!no_game){
        while(!(pid=find_process(L"Civ3Conquests.exe"))&&elapsed()<seconds)Sleep(250);
        if(!pid){std::fprintf(out,"error no-game\n");std::fclose(out);return 1;}
        game=OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_INFORMATION|SYNCHRONIZE,FALSE,pid);
        if(!game){std::fprintf(out,"error open-game %lu\n",GetLastError());std::fclose(out);return 1;}
        while(elapsed()<seconds&&WaitForSingleObject(game,0)==WAIT_TIMEOUT){
            load_modules();
            for(auto const& m:modules)if(!_stricmp(m.name.c_str(),"sound.dll"))sound=m.base;
            IMAGE_DOS_HEADER dos={};std::uint32_t stamp=0;
            if(sound&&read(sound,&dos,sizeof(dos))&&read(sound+dos.e_lfanew+8,&stamp,4)){
                if(stamp!=sound_stamp){std::fprintf(out,"error sound-stamp 0x%08x\n",stamp);std::fclose(out);return 1;}
                if(read(sound+timer_object,&object,4)&&object)break;
            }
            Sleep(250);
        }
        if(!object){std::fprintf(out,"error no-sound-timer\n");std::fclose(out);return 1;}
        game_main=main_thread(pid);
    }
    std::fprintf(out,"start qpc=%lld frequency=%lld pid=%lu main=%lu sound=0x%08x object=0x%08x threshold=%u\n",
        start,frequency.QuadPart,pid,game_main,sound,object,threshold);
    // --buffers: list sound.dll's DirectSound buffer objects (a DSBUFFERDESC
    // {0x24, flags, bytes, 0, &format} at +0x70 with the format at +0xa8) and
    // their ring positions/lead fields at +0x10..+0x44, then exit.
    if(argc>4&&!std::strcmp(argv[4],"--buffers")){
        std::vector<std::uint32_t> page(0x10000/4);
        for(std::uint32_t at=0x10000;at<0xfff00000u;){
            MEMORY_BASIC_INFORMATION info={};
            if(!VirtualQueryEx(game,reinterpret_cast<void const*>(std::uintptr_t(at)),&info,sizeof(info)))break;
            auto region=std::uint32_t(std::uintptr_t(info.BaseAddress)),size=std::uint32_t(info.RegionSize);
            if(info.State==MEM_COMMIT&&info.Type==MEM_PRIVATE&&!(info.Protect&(PAGE_GUARD|PAGE_NOACCESS)))
                for(std::uint32_t chunk=region;chunk<region+size;chunk+=0x10000){
                    auto bytes=std::min<std::uint32_t>(0x10000,region+size-chunk);
                    if(!read(chunk,page.data(),bytes))continue;
                    for(std::uint32_t n=0;n+12<bytes/4;++n){
                        auto self=chunk+n*4-0x70;
                        if(page[n]!=0x24||page[n+4]!=self+0xa8||!page[n+2]||page[n+2]>0x400000)continue;
                        std::uint32_t fields[16]={},format[4]={};
                        if(!read(self+0x10,fields,sizeof(fields))||!read(self+0xa8,format,sizeof(format)))continue;
                        std::fprintf(out,"buffer object=0x%08x flags=0x%x bytes=%u rate=%u bytes_per_second=%u",self,page[n+1],page[n+2],format[1],format[2]);
                        for(unsigned f=0;f<16;++f)std::fprintf(out," +0x%x=%u",0x10+f*4,fields[f]);
                        std::fprintf(out,"\n");
                    }
                }
            at=region+size;if(at<region)break;
        }
        std::fclose(out);return 0;
    }
    // --hold-test: learn the mixer's timer thread as the lock owner, then hold
    // Civ III's main thread or the timer thread for fixed times. Pair it with
    // a --record capture to hear which holds glitch the music.
    if(argc>4&&!std::strcmp(argv[4],"--hold-test")){
        std::map<DWORD,unsigned> owners;
        for(auto until=qpc()+3*frequency.QuadPart;qpc()<until;){
            std::uint32_t owner=0;Sleep(1);if(read(sound+timer_lock+0x0c,&owner,4)&&owner)++owners[owner];}
        DWORD timer=0;unsigned best=0;for(auto const& [tid,count]:owners)if(count>best){best=count;timer=tid;}
        std::fprintf(out,"timer-thread tid=%lu lock_polls=%u\n",timer,best);
        if(!timer){std::fclose(out);return 1;}
        struct Hold {char const* name;DWORD tid,ms;};
        Hold const holds[]={{"main",game_main,1000},{"timer",timer,150},{"timer",timer,300},{"main",game_main,3000},{"timer",timer,600}};
        for(auto const& hold:holds){Sleep(4000);hold_thread(out,hold.name,hold.tid,hold.ms);}
        Sleep(4000);std::fclose(out);return 0;
    }
    // The callback's timer clients: list nodes {?, next, client} at object+0x4c.
    std::uint32_t node=0;if(object)read(object+0x4c,&node,4);
    for(unsigned n=0;node&&n<16;++n){
        std::uint32_t link[3]={},vtable=0,methods[2]={};
        if(!read(node,link,12))break;
        if(link[2]&&read(link[2],&vtable,4)&&read(vtable+0x38,&methods[0],4))read(vtable+0x44,&methods[1],4);
        std::fprintf(out,"timer-client node=0x%08x client=0x%08x vtable=%s m38=%s m44=%s\n",node,link[2],
            label(vtable).c_str(),label(methods[0]).c_str(),label(methods[1]).c_str());
        node=link[1];
    }
    timeBeginPeriod(1);SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_TIME_CRITICAL);
    // PDH reads the 64-bit helper's memory without 32-bit truncation.
    HANDLE helper=nullptr;PDH_HQUERY query=nullptr;PDH_HCOUNTER page_reads_counter=nullptr,helper_ws_counter=nullptr,helper_private_counter=nullptr;DWORD game_faults0=0;
    if(PdhOpenQueryW(nullptr,0,&query)!=ERROR_SUCCESS||PdhAddEnglishCounterW(query,L"\\Memory\\Page Reads/sec",0,&page_reads_counter)!=ERROR_SUCCESS)query=nullptr;
    else {
        PdhAddEnglishCounterW(query,L"\\Process(C3XRendererHelper64)\\Working Set",0,&helper_ws_counter);
        PdhAddEnglishCounterW(query,L"\\Process(C3XRendererHelper64)\\Private Bytes",0,&helper_private_counter);
        PdhCollectQueryData(query);
    }
    unsigned const edges[]={40,60,90,150,300,600,1000000};
    unsigned long long histogram[7]={},poll_histogram[7]={},guest_wide=0,game_only=0;
    std::map<DWORD,unsigned long long> owners;unsigned long long polls=0,services=0;
    std::uint32_t last=0;long long previous=qpc(),second_start=previous,changed_at=previous;
    FILETIME idle0,kernel0,user0;GetSystemTimes(&idle0,&kernel0,&user0);
    ULONGLONG game0=process_cpu(game),helper0=0;
    unsigned second_max_interval=0;double second_max_poll=0;
    // Per service interval: the probe's longest poll delay, the polls that saw
    // the reentrancy flag, the lock owners and (once past 75 ms) thread samples.
    double interval_poll=0;unsigned interval_flag=0;std::map<DWORD,unsigned> interval_owners;
    bool sampled=false;std::string samples;
    long long busy_since=0;std::string busy_sample;unsigned long long slow_callbacks=0;
    while(elapsed()<seconds&&(no_game||WaitForSingleObject(game,0)==WAIT_TIMEOUT)){
        Sleep(1);
        auto now=qpc();double poll_ms=ms(now-previous);previous=now;++polls;
        if(poll_ms>second_max_poll)second_max_poll=poll_ms;
        if(poll_ms>interval_poll)interval_poll=poll_ms;
        {unsigned bucket=0;while(poll_ms>=edges[bucket])++bucket;++poll_histogram[bucket];}
        if(!no_game){
            std::uint32_t serviced=0,flag=0,owner=0;unsigned char lock[0x1c]={};
            if(!read(object+4,&serviced,4)||!read(sound+timer_lock,lock,sizeof(lock)))break;
            std::memcpy(&owner,lock+0x0c,4);std::memcpy(&flag,lock+0x18,4);
            if(owner){++owners[owner];++interval_owners[owner];}
            if(flag)++interval_flag;
            // A callback in progress (flag set, lock held by the timer thread)
            // for 15 ms or more is sampled once and reported when it ends.
            DWORD timer=0;{unsigned long long best=0;for(auto const& [tid,count]:owners)if(tid!=game_main&&count>best){best=count;timer=tid;}}
            if(flag&&owner&&owner==timer){
                if(!busy_since)busy_since=now;
                else if(busy_sample.empty()&&ms(now-busy_since)>=15){
                    load_modules();
                    busy_sample=" states=["+thread_states(pid,timer,game_main)+"] timer_wait=["+wait_chain(timer)+"] timer_stack=["+sample_thread(timer)+"]";
                }
            }
            else if(busy_since){
                double held=ms(now-busy_since);
                if(held>=15){++slow_callbacks;std::fprintf(out,"slow-callback t=%.3f held_ms=%.1f%s\n",ms(busy_since-start)/1000.0,held,busy_sample.c_str());}
                busy_since=0;busy_sample.clear();
            }
            if(serviced&&serviced!=last){
                if(last){
                    unsigned interval=serviced-last;++services;
                    unsigned bucket=0;while(interval>=edges[bucket])++bucket;++histogram[bucket];
                    if(interval>second_max_interval)second_max_interval=interval;
                    if(interval>=threshold){
                        // A pause that stopped this thread for most of the interval was guest-wide.
                        bool wide=interval_poll+30.0>=double(interval);(wide?guest_wide:game_only)++;
                        std::string who;for(auto const& [tid,count]:interval_owners){
                            char t[48];std::snprintf(t,sizeof(t),"%s%lu%s:%u",who.empty()?"":",",tid,tid==game_main?"(main)":"",count);who+=t;}
                        std::fprintf(out,"stall t=%.3f qpc=%lld interval_ms=%u probe_max_poll_ms=%.1f %s flag_polls=%u owners=%s%s\n",
                            ms(now-start)/1000.0,now,interval,interval_poll,wide?"guest-wide":"game-only",interval_flag,
                            who.empty()?"-":who.c_str(),samples.c_str());
                    }
                }
                last=serviced;changed_at=now;interval_poll=0;interval_flag=0;interval_owners.clear();sampled=false;samples.clear();
            }
            else if(last&&!sampled&&ms(now-changed_at)>=75){
                sampled=true;load_modules();
                char head[80];std::snprintf(head,sizeof(head)," sampled_at_ms=%.0f flag=%u owner=%lu",ms(now-changed_at),flag,owner);samples=head;
                samples+=" states=["+thread_states(pid,timer,game_main)+"]";
                if(timer)samples+=" timer_wait=["+wait_chain(timer)+"] timer_stack=["+sample_thread(timer)+"]";
                if(owner&&owner!=game_main&&owner!=timer)samples+=" owner_stack=["+sample_thread(owner)+"]";
                samples+=" main_stack=["+sample_thread(game_main)+"]";
            }
        }
        if(now-second_start>=frequency.QuadPart){
            if(!no_game&&!helper){DWORD h=find_process(L"C3XRendererHelper64.exe");if(h){helper=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,FALSE,h);helper0=process_cpu(helper);}}
            FILETIME idle,kernel,user;GetSystemTimes(&idle,&kernel,&user);
            double wall=ms(now-second_start)*1e4;
            double total=double(filetime(kernel)-filetime(kernel0)+filetime(user)-filetime(user0));
            double busy=total>0?1.0-double(filetime(idle)-filetime(idle0))/total:0;
            ULONGLONG game1=process_cpu(game),helper1=process_cpu(helper);
            // Memory: guest available, working sets, the game's page faults
            // (soft and hard) and the system's hard page reads per second.
            MEMORYSTATUSEX status={sizeof(status)};GlobalMemoryStatusEx(&status);
            PROCESS_MEMORY_COUNTERS game_memory={sizeof(game_memory)};
            if(game)GetProcessMemoryInfo(game,&game_memory,sizeof(game_memory));
            double page_reads=0,helper_ws=0,helper_private=0;
            if(query&&PdhCollectQueryData(query)==ERROR_SUCCESS){
                auto value=[](PDH_HCOUNTER counter){PDH_FMT_COUNTERVALUE v={};
                    return counter&&PdhGetFormattedCounterValue(counter,PDH_FMT_DOUBLE,nullptr,&v)==ERROR_SUCCESS?v.doubleValue:0.0;};
                page_reads=value(page_reads_counter);helper_ws=value(helper_ws_counter)/1048576.0;helper_private=value(helper_private_counter)/1048576.0;
            }
            std::fprintf(out,"second t=%.1f max_interval_ms=%u max_poll_ms=%.1f system_busy=%.2f game_cpus=%.2f helper_cpus=%.2f"
                " avail_mb=%llu game_ws_mb=%zu game_faults=%lu helper_ws_mb=%.0f helper_private_mb=%.0f page_reads=%.0f\n",
                elapsed(),second_max_interval,second_max_poll,busy,game?double(game1-game0)/wall:0.0,helper?double(helper1-helper0)/wall:0.0,
                status.ullAvailPhys>>20,game_memory.WorkingSetSize>>20,game_memory.PageFaultCount-game_faults0,
                helper_ws,helper_private,page_reads);
            game_faults0=game_memory.PageFaultCount;
            idle0=idle;kernel0=kernel;user0=user;game0=game1;helper0=helper1;
            second_start=now;second_max_interval=0;second_max_poll=0;std::fflush(out);
        }
    }
    std::fprintf(out,"summary seconds=%.1f polls=%llu services=%llu long_intervals_guest_wide=%llu long_intervals_game_only=%llu slow_callbacks=%llu\nintervals",
        elapsed(),polls,services,guest_wide,game_only,slow_callbacks);
    for(unsigned n=0;n<7;++n)std::fprintf(out," <%u:%llu",edges[n],histogram[n]);
    std::fprintf(out,"\npoll_delays");
    for(unsigned n=0;n<7;++n)std::fprintf(out," <%u:%llu",edges[n],poll_histogram[n]);
    std::fprintf(out,"\nowners");
    for(auto const& [tid,count]:owners)std::fprintf(out," %lu%s:%llu",tid,tid==game_main?"(main)":"",count);
    std::fprintf(out,"\n");std::fclose(out);timeEndPeriod(1);
    return 0;
}
