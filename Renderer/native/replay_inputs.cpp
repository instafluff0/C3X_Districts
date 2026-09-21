#define NOMINMAX
#include <windows.h>
#include "input_recording/journal.h"
#include "input_recording/assets.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
using namespace c3x_inputs;
using Replay=int(*)(unsigned char const*,unsigned,void*,char*,unsigned);
struct Pending {Event input;std::vector<std::pair<std::uint64_t,std::uint64_t>> clocks;};
void apply_settings(Reader& header){
    auto environment=GetEnvironmentStringsW();require(environment!=nullptr,"cannot inspect replay environment");
    for(auto item=environment;*item;item+=wcslen(item)+1){std::wstring setting=item;auto equal=setting.find(L'=');
        if(setting.rfind(L"C3X_RENDERER_",0)==0&&equal!=std::wstring::npos)SetEnvironmentVariableW(setting.substr(0,equal).c_str(),nullptr);}
    FreeEnvironmentStringsW(environment);SetEnvironmentVariableW(L"C3X_RENDERER_MANUAL_VISUAL",L"1");
    auto count=header.u32();require(count<=512,"input settings limit");
    for(unsigned n=0;n<count;++n){auto key=header.string(256),value=header.string(32768);require(key.rfind("C3X_RENDERER_",0)==0&&key.rfind("C3X_RENDERER_INPUT_",0)!=0,"invalid recorded setting");
        if(key.find("TRACE")==std::string::npos&&key.find("RECORD")==std::string::npos&&key!="C3X_RENDERER_MANUAL_VISUAL")SetEnvironmentVariableA(key.c_str(),value.c_str());}header.done();
}
int wmain(int argc,wchar_t** argv){
    HMODULE module=nullptr;HWND window=nullptr;int exit_code=0;
    try{
        require(argc>=4&&std::wstring(argv[1])==L"--development","usage: replay_inputs --development DLL INPUT_DIRECTORY [--frames DIR LAST | --range DIR FIRST LAST | --seconds DIR START END] [--timeline NEW_FILE] [--fingerprints NEW_FILE] [--trace NEW_FILE] [--allow-prefix] [--compare-candidate]");
        std::filesystem::path frames,timeline_path,trace_path,fingerprints_path;std::uint64_t first_frame=1,frame_limit=0,frame_number=0,exported=0;double first_second=0,last_second=0;bool seconds=false,allow_prefix=false,tail_truncated=false,compare_candidate=false,binary_matches_capture=true;
        auto integer=[](wchar_t const* raw){std::size_t used=0;std::wstring text(raw);auto value=std::stoull(text,&used);require(used==text.size()&&value&&value<=1000000,"invalid frame bound");return value;};
        auto second=[](wchar_t const* raw){std::size_t used=0;std::wstring text(raw);auto value=std::stod(text,&used);require(used==text.size()&&std::isfinite(value)&&value>=0&&value<=86400,"invalid second bound");return value;};
        for(int i=4;i<argc;){std::wstring flag=argv[i++];
            if(flag==L"--compare-candidate"){compare_candidate=true;continue;}
            if(flag==L"--allow-prefix"){allow_prefix=true;continue;}
            if(flag==L"--timeline"){require(i<argc&&timeline_path.empty(),"invalid timeline option");timeline_path=argv[i++];continue;}
            if(flag==L"--trace"){require(i<argc&&trace_path.empty(),"invalid trace option");trace_path=argv[i++];continue;}
            if(flag==L"--fingerprints"){require(i<argc&&fingerprints_path.empty(),"invalid fingerprint option");fingerprints_path=argv[i++];continue;}
            require(frames.empty()&&i<argc,"invalid frame export option");frames=argv[i++];
            if(flag==L"--frames"){require(i<argc,"missing frame limit");frame_limit=integer(argv[i++]);}
            else if(flag==L"--range"){require(i+1<argc,"missing frame range");first_frame=integer(argv[i++]);frame_limit=integer(argv[i++]);require(first_frame<=frame_limit&&frame_limit-first_frame<10000,"invalid frame range");}
            else if(flag==L"--seconds"){require(i+1<argc,"missing second range");first_second=second(argv[i++]);last_second=second(argv[i++]);require(first_second<last_second,"empty second range");seconds=true;}
            else throw std::runtime_error("unknown replay option");
        }
        if(!frames.empty())require(!std::filesystem::exists(frames)&&std::filesystem::create_directories(frames),"frame directory must be new");
        if(!trace_path.empty())require(!std::filesystem::exists(trace_path),"trace must be new");
        auto configure=[&](Reader& settings){apply_settings(settings);if(!trace_path.empty()){
            SetEnvironmentVariableW(L"C3X_RENDERER_TRACE",L"2");SetEnvironmentVariableW(L"C3X_RENDERER_TRACE_MIB",L"32");
            SetEnvironmentVariableW(L"C3X_RENDERER_TRACE_BUFFERED",L"1");SetEnvironmentVariableW(L"C3X_RENDERER_TRACE_FILE",trace_path.c_str());}};
        std::ofstream timeline;if(!timeline_path.empty()){require(!std::filesystem::exists(timeline_path),"timeline must be new");timeline.open(timeline_path);require(bool(timeline),"cannot open replay timeline");}
        std::ofstream fingerprints;if(!fingerprints_path.empty()){require(!std::filesystem::exists(fingerprints_path),"fingerprints must be new");fingerprints.open(fingerprints_path);require(bool(fingerprints),"cannot open replay fingerprints");}
        // Replay owns offered ticks. An independent cadence would add inputs
        // that were never present in the recording.
        require(GetEnvironmentVariableW(L"C3X_RENDERER_INPUT_RECORD_DIR",nullptr,0)==0,"disable input recording before replay");
        {SegmentReader probe(argv[3]);Event first;require(probe.next(first)&&first.kind==Kind::manifest,"missing input manifest");Reader header{first.payload};
            require(header.u32()==protocol_version&&header.u32()==C3X_RENDERER_API_VERSION,"input API mismatch");header.u64();ClockOrigin origin;clock_origin(header,origin);configure(header);}
        // Resolve and verify the complete observed asset closure before loading
        // the DLL. Optional missing assets are pinned too; source data stays local.
        {SegmentReader probe(argv[3]);Event item;while(!probe.footer&&probe.next_verified(item,allow_prefix,tail_truncated))if(item.kind==Kind::asset){Reader data{item.payload};std::string path;auto recorded=asset_fields(data,path);data.done();require(item.flags<=1,"unknown asset role");auto matches=asset_file(item.flags==1?asset_path(argv[2]):path)==recorded;
            if(item.flags==1){binary_matches_capture=matches;require(matches||compare_candidate,"replay DLL does not match capture (candidate comparison must be explicit)");}
            else if(!matches){std::cerr<<"asset="<<path<<'\n';throw std::runtime_error("asset manifest mismatch");}}}
        module=LoadLibraryW(argv[2]);require(module!=nullptr,"cannot load replay DLL");
        auto replay=reinterpret_cast<Replay>(GetProcAddress(module,"c3x_renderer_input_replay"));require(replay!=nullptr,"DLL lacks production input replay entry");
        auto asset_entry=reinterpret_cast<int(*)(unsigned char const*,unsigned)>(GetProcAddress(module,"c3x_renderer_input_replay_asset"));require(asset_entry&&asset_entry(nullptr,0)==1,"DLL lacks replay asset verification");
        {SegmentReader probe(argv[3]);Event item;while(!probe.footer&&probe.next_verified(item,allow_prefix,tail_truncated))if(item.kind==Kind::asset&&item.flags==0)require(asset_entry(item.payload.data(),unsigned(item.payload.size()))==1,"replay asset import failed");}
        WNDCLASSW wc={};wc.lpfnWndProc=DefWindowProcW;wc.hInstance=GetModuleHandleW(nullptr);wc.lpszClassName=L"C3XInputReplay";RegisterClassW(&wc);
        window=CreateWindowW(wc.lpszClassName,L"C3X input replay",WS_POPUP,0,0,2240,1260,nullptr,nullptr,wc.hInstance,nullptr);require(window!=nullptr,"cannot create replay target");
        SegmentReader reader(argv[3]);Event event;std::map<std::uint64_t,Pending> pending;
        std::uint64_t calls=0,clocks=0,queued=0,peak=0,last=0;bool manifest=false;LARGE_INTEGER frequency={},started={},ended={};QueryPerformanceFrequency(&frequency);double work_ms=0;
        while(!reader.footer&&reader.next_verified(event,allow_prefix,tail_truncated)){
            last=event.sequence;Reader payload{event.payload};
            if(event.kind==Kind::manifest){require(!manifest&&event.sequence==1,"duplicate/misplaced input manifest");
                require(payload.u32()==protocol_version&&payload.u32()==C3X_RENDERER_API_VERSION,"input API mismatch");payload.u64();ClockOrigin origin;clock_origin(payload,origin);auto count=payload.u32();require(count<=512,"input settings limit");for(unsigned n=0;n<count;++n){payload.string(256);payload.string(32768);}payload.done();manifest=true;continue;}
            require(manifest,"missing input manifest");if(event.kind==Kind::footer)break;
            if(event.kind==Kind::asset)continue;
            if(event.kind==Kind::settings){require(pending.empty(),"settings changed during a pending input call");configure(payload);continue;}
            if(event.kind==Kind::visual&&event.flags==2){auto parent=payload.u64(),tick=payload.u64(),freq=payload.u64();payload.done();
                require(freq&&tick<=INT64_MAX&&freq<=INT64_MAX,"invalid sampled clock");++clocks;
                if(parent){auto it=pending.find(parent);require(it!=pending.end(),"clock has missing parent");require(it->second.clocks.size()<64,"excessive call clocks");it->second.clocks.emplace_back(tick,freq);}continue;}
            auto token=payload.u64();
            if(event.kind!=Kind::result){require(token&&pending.size()<256&&!pending.count(token),"invalid call token");
                require(event.payload.size()<=32u*1024u*1024u-queued,"replay pending input budget");queued+=event.payload.size();peak=std::max(peak,queued);
                pending.emplace(token,Pending{std::move(event),{}});continue;}
            auto it=pending.find(token);require(it!=pending.end(),"result has missing input");auto& call=it->second;
            // A bounded tool envelope carries owned scalar records to the DLL;
            // it is decoded before any production renderer invocation.
            Writer envelope;envelope.u32(unsigned(call.input.kind));envelope.u32(call.input.flags);
            envelope.u32(unsigned(call.input.payload.size()));envelope.reserve(call.input.payload.size());envelope.bytes.insert(envelope.bytes.end(),call.input.payload.begin(),call.input.payload.end());
            envelope.u32(unsigned(event.payload.size()));envelope.reserve(event.payload.size());envelope.bytes.insert(envelope.bytes.end(),event.payload.begin(),event.payload.end());
            envelope.u32(unsigned(call.clocks.size()));for(auto const& sample:call.clocks){envelope.u64(sample.first);envelope.u64(sample.second);}
            MSG message;while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageW(&message);}
            char error[512]={};QueryPerformanceCounter(&started);int ok=replay(envelope.bytes.data(),unsigned(envelope.bytes.size()),window,error,sizeof(error));QueryPerformanceCounter(&ended);
            work_ms+=1000.*double(ended.QuadPart-started.QuadPart)/double(frequency.QuadPart);
            if(!ok){std::cerr<<"event="<<last<<" call="<<token<<" family="<<unsigned(call.input.kind)<<" subtype="<<call.input.flags<<" error="<<error<<'\n';throw std::runtime_error("production input replay mismatch");}
            Reader request{call.input.payload};request.u64();request.u64();Reader completion{event.payload};completion.u64();auto code=completion.u32();
            bool boundary=(call.input.kind==Kind::presentation&&request.u32()==0)||
                (call.input.kind==Kind::visual&&call.input.flags==1)||
                (call.input.kind==Kind::native_snapshot&&call.input.flags==0&&request.u32()==1);
            if(boundary&&code==C3X_RENDERER_RESULT_OK){++frame_number;auto time=double(event.ticks)/double(reader.frequency);
                if(fingerprints.is_open()){
                    auto fingerprint=reinterpret_cast<int(*)(unsigned*,unsigned*)>(GetProcAddress(module,"c3x_renderer_input_replay_fingerprint"));
                    unsigned extent[2]={},hash[4]={};require(fingerprint&&fingerprint(extent,hash)==1,"display fingerprint failed");
                    fingerprints<<"{\"frame\":"<<frame_number<<",\"call\":"<<token<<",\"sequence\":"<<event.sequence
                        <<",\"width\":"<<extent[0]<<",\"height\":"<<extent[1]<<",\"bgra_hash128\":\""<<std::hex<<std::setfill('0');
                    for(auto word:hash)fingerprints<<std::setw(8)<<word;fingerprints<<std::dec<<"\"}\n";
                    require(bool(fingerprints),"replay fingerprint write failed");
                }
                bool selected=!frames.empty()&&(seconds?(time>=first_second&&time<last_second):(frame_number>=first_frame&&frame_number<=frame_limit));
                if(selected){
                    require(exported<10000,"frame export count limit; select a smaller range");
                    require(std::filesystem::space(frames).available>=2240ull*1260*4+64ull*1024*1024,"insufficient frame export disk space");++exported;
                    auto export_frame=reinterpret_cast<int(*)(wchar_t const*)>(GetProcAddress(module,"c3x_renderer_input_replay_frame"));
                    wchar_t name[64];swprintf_s(name,L"frame-%06llu.bmp",static_cast<unsigned long long>(frame_number));require(export_frame&&export_frame((frames/name).c_str())==1,"frame export failed");}
                if(timeline.is_open())timeline<<"{\"frame\":"<<frame_number<<",\"call\":"<<token<<",\"sequence\":"<<event.sequence<<",\"seconds\":"<<time
                    <<",\"family\":"<<unsigned(call.input.kind)<<",\"exported\":"<<(selected?"true":"false")<<"}\n";
            }
            ++calls;queued-=call.input.payload.size();pending.erase(it);
        }
        if(timeline.is_open()){timeline.flush();require(bool(timeline),"replay timeline write failed");}
        if(fingerprints.is_open()){fingerprints.flush();require(bool(fingerprints),"replay fingerprint flush failed");}
        bool complete=reader.footer&&(reader.reason==Stop::closed||reader.reason==Stop::duration)&&pending.empty()&&!tail_truncated;
        require(complete||allow_prefix,"input capture incomplete (use --allow-prefix to inspect verified completed calls)");
        std::cout<<"{\"status\":\"development_input_replay\",\"qualified\":false,\"timing_scope\":\"forensic_calls_not_performance\",\"calls\":"<<calls<<",\"binary_matches_capture\":"<<(binary_matches_capture?"true":"false")<<",\"complete\":"<<(complete?"true":"false")<<",\"unfinished_calls\":"<<pending.size()<<",\"accepted_presentations\":"<<frame_number<<",\"clock_samples\":"<<clocks<<",\"pending_peak_bytes\":"<<peak<<",\"production_call_ms\":"<<work_ms<<"}\n";
    }catch(std::exception const& error){std::cerr<<error.what()<<'\n';exit_code=1;}
    // Finish exception unwinding before unloading a DLL that may have thrown.
    if(module){auto reset=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_reset"));if(reset)reset();}
    if(window)DestroyWindow(window);
    if(module)FreeLibrary(module);
    return exit_code;
}
