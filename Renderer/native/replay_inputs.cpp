#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include "input_recording/journal.h"
#include "input_recording/assets.h"
#include "helper_trial/scene_client.h"
#include "helper_trial/shared_frame_reader.h"
#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <cmath>
using namespace c3x_inputs;
using Replay=int(*)(unsigned char const*,unsigned,void*,char*,unsigned);
using Execution=int(*)(unsigned,double*,int*,unsigned*);
struct MemorySample {std::uint64_t private_bytes=0,free_bytes=0,largest_free=0;};
MemorySample memory_sample(){
    MemorySample result;PROCESS_MEMORY_COUNTERS_EX memory={};memory.cb=sizeof(memory);
    require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),sizeof(memory))!=FALSE,"cannot sample replay process");result.private_bytes=memory.PrivateUsage;
    MEMORY_BASIC_INFORMATION region={};std::uintptr_t address=0;
    while(VirtualQuery(reinterpret_cast<void*>(address),&region,sizeof(region))){if(region.State==MEM_FREE){result.free_bytes+=region.RegionSize;result.largest_free=std::max(result.largest_free,std::uint64_t(region.RegionSize));}
        auto next=reinterpret_cast<std::uintptr_t>(region.BaseAddress)+region.RegionSize;if(next<=address)break;address=next;}return result;
}
struct Pending {Event input;std::vector<std::pair<std::uint64_t,std::uint64_t>> clocks;std::size_t settings=0;bool skipped=false;};
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
    HMODULE module=nullptr;HWND window=nullptr;void* reservation=nullptr;int exit_code=0;
    try{
        require(argc>=4&&std::wstring(argv[1])==L"--development","usage: replay_inputs --development DLL INPUT_DIRECTORY [--frames DIR LAST | --range DIR FIRST LAST | --seconds DIR START END] [--timeline NEW_FILE] [--fingerprints NEW_FILE] [--trace NEW_FILE] [--allow-prefix] [--before-event N] [--compare-candidate] [--performance NEW_FILE] [--reserve-mib N] [--watch] [--realtime NEW_FILE] [--x64-scene HELPER DLL NEW_REPORT]");
        std::wstring label=L"C3X input replay";
        std::filesystem::path frames,timeline_path,trace_path,fingerprints_path,performance_path,shadow_path;
        std::wstring shadow_helper,shadow_dll;unsigned reserve_mib=0;std::uint64_t before_event=0;bool stopped_before_event=false;std::uint64_t first_frame=1,frame_limit=0,frame_number=0,exported=0;double first_second=0,last_second=0;bool seconds=false,allow_prefix=false,tail_truncated=false,compare_candidate=false,binary_matches_capture=true,watch=false,realtime=false;
        auto integer=[](wchar_t const* raw){std::size_t used=0;std::wstring text(raw);auto value=std::stoull(text,&used);require(used==text.size()&&value&&value<=1000000,"invalid frame bound");return value;};
        auto second=[](wchar_t const* raw){std::size_t used=0;std::wstring text(raw);auto value=std::stod(text,&used);require(used==text.size()&&std::isfinite(value)&&value>=0&&value<=86400,"invalid second bound");return value;};
        for(int i=4;i<argc;){std::wstring flag=argv[i++];
            if(flag==L"--label"){require(i<argc,"missing playback label");label=argv[i++];require(label.size()<=120,"playback label limit");continue;}
            if(flag==L"--realtime"){require(i<argc&&performance_path.empty(),"invalid realtime option");realtime=true;performance_path=argv[i++];continue;}
            if(flag==L"--watch"){watch=true;continue;}
            if(flag==L"--performance"){require(i<argc&&performance_path.empty(),"invalid performance option");performance_path=argv[i++];continue;}
            if(flag==L"--x64-scene"){require(i+2<argc&&shadow_path.empty(),"invalid x64 scene option");shadow_helper=argv[i++];shadow_dll=argv[i++];shadow_path=argv[i++];continue;}
            if(flag==L"--reserve-mib"){require(i<argc&&!reserve_mib,"invalid reservation option");auto n=integer(argv[i++]);require(n<=2048,"reservation limit");reserve_mib=unsigned(n);continue;}
            if(flag==L"--compare-candidate"){compare_candidate=true;continue;}
            if(flag==L"--allow-prefix"){allow_prefix=true;continue;}
            if(flag==L"--before-event"){
                require(i<argc&&!before_event,"invalid event cutoff");std::wstring raw=argv[i++];std::size_t used=0;
                before_event=std::stoull(raw,&used);require(used==raw.size()&&before_event>1&&raw[0]!=L'-',"invalid event cutoff");continue;
            }
            if(flag==L"--timeline"){require(i<argc&&timeline_path.empty(),"invalid timeline option");timeline_path=argv[i++];continue;}
            if(flag==L"--trace"){require(i<argc&&trace_path.empty(),"invalid trace option");trace_path=argv[i++];continue;}
            if(flag==L"--fingerprints"){require(i<argc&&fingerprints_path.empty(),"invalid fingerprint option");fingerprints_path=argv[i++];continue;}
            require(frames.empty()&&i<argc,"invalid frame export option");frames=argv[i++];
            if(flag==L"--frames"){require(i<argc,"missing frame limit");frame_limit=integer(argv[i++]);}
            else if(flag==L"--range"){require(i+1<argc,"missing frame range");first_frame=integer(argv[i++]);frame_limit=integer(argv[i++]);require(first_frame<=frame_limit&&frame_limit-first_frame<10000,"invalid frame range");}
            else if(flag==L"--seconds"){require(i+1<argc,"missing second range");first_second=second(argv[i++]);last_second=second(argv[i++]);require(first_second<last_second,"empty second range");seconds=true;}
            else throw std::runtime_error("unknown replay option");
        }
        require(!before_event||allow_prefix,"explicit event cutoff requires --allow-prefix");
        bool performance=!performance_path.empty();require(!performance||(frames.empty()&&fingerprints_path.empty()&&!watch),"performance runs cannot export, fingerprint or pace playback");require(!reserve_mib||performance,"reservation requires performance mode");
        require(shadow_path.empty()||!realtime,"x64 scene diagnostic is unpaced only");
        std::ofstream measurements;if(performance){require(!std::filesystem::exists(performance_path),"performance output must be new");measurements.open(performance_path);require(bool(measurements),"cannot open performance output");}
        if(reserve_mib){reservation=VirtualAlloc(nullptr,std::size_t(reserve_mib)*1024*1024,MEM_RESERVE,PAGE_NOACCESS);require(reservation!=nullptr,"cannot reserve declared address-space pressure");}
        if(!frames.empty())require(!std::filesystem::exists(frames)&&std::filesystem::create_directories(frames),"frame directory must be new");
        if(!trace_path.empty())require(!std::filesystem::exists(trace_path),"trace must be new");
        auto configure=[&](Reader& settings){apply_settings(settings);if(realtime)SetEnvironmentVariableW(L"C3X_RENDERER_MANUAL_VISUAL",L"0");if(!trace_path.empty()){
            SetEnvironmentVariableW(L"C3X_RENDERER_TRACE",L"2");SetEnvironmentVariableW(L"C3X_RENDERER_TRACE_MIB",L"32");
            SetEnvironmentVariableW(L"C3X_RENDERER_TRACE_BUFFERED",L"1");SetEnvironmentVariableW(L"C3X_RENDERER_TRACE_FILE",trace_path.c_str());}};
        std::vector<Bytes> settings_versions;std::size_t active_settings=0,applied_settings=0;
        std::ofstream timeline;if(!timeline_path.empty()){require(!std::filesystem::exists(timeline_path),"timeline must be new");timeline.open(timeline_path);require(bool(timeline),"cannot open replay timeline");}
        std::ofstream fingerprints;if(!fingerprints_path.empty()){require(!std::filesystem::exists(fingerprints_path),"fingerprints must be new");fingerprints.open(fingerprints_path);require(bool(fingerprints),"cannot open replay fingerprints");}
        // Replay owns offered ticks. An independent cadence would add inputs
        // that were never present in the recording.
        require(GetEnvironmentVariableW(L"C3X_RENDERER_INPUT_RECORD_DIR",nullptr,0)==0,"disable input recording before replay");
        {SegmentReader probe(argv[3]);Event first;require(probe.next(first)&&first.kind==Kind::manifest,"missing input manifest");Reader header{first.payload};
            require(header.u32()==protocol_version&&header.u32()==C3X_RENDERER_API_VERSION,"input API mismatch");header.u64();ClockOrigin origin;clock_origin(header,origin);settings_versions.emplace_back(first.payload.begin()+header.at,first.payload.end());configure(header);}
        // Resolve and verify the complete observed asset closure before loading
        // the DLL. Optional missing assets are pinned too; source data stays local.
        {SegmentReader probe(argv[3]);Event item;while(!probe.footer&&probe.next_verified(item,allow_prefix,tail_truncated))if(item.kind==Kind::asset){Reader data{item.payload};std::string path;auto recorded=asset_fields(data,path);data.done();require(item.flags<=1,"unknown asset role");auto matches=asset_file(item.flags==1?asset_path(argv[2]):path)==recorded;
            if(item.flags==1){binary_matches_capture=matches;require(matches||compare_candidate,"replay DLL does not match capture (candidate comparison must be explicit)");}
            else if(!matches){std::cerr<<"asset="<<path<<'\n';throw std::runtime_error("asset manifest mismatch");}}}
        module=LoadLibraryW(argv[2]);require(module!=nullptr,"cannot load replay DLL");
        auto replay=reinterpret_cast<Replay>(GetProcAddress(module,"c3x_renderer_input_replay"));require(replay!=nullptr,"DLL lacks production input replay entry");
        auto execution=reinterpret_cast<Execution>(GetProcAddress(module,"c3x_renderer_input_replay_execution"));if(performance)require(execution&&execution(1,nullptr,nullptr,nullptr)==1,"DLL lacks performance replay entry");
        using LiveStatus=int(*)(unsigned,unsigned,double*,unsigned*);
        auto live_status=reinterpret_cast<LiveStatus>(GetProcAddress(module,"c3x_renderer_input_replay_live_status"));
        if(realtime)require(live_status,"DLL lacks independent real-time playback; use a compatible candidate");
        auto asset_entry=reinterpret_cast<int(*)(unsigned char const*,unsigned)>(GetProcAddress(module,"c3x_renderer_input_replay_asset"));require(asset_entry&&asset_entry(nullptr,0)==1,"DLL lacks replay asset verification");
        {SegmentReader probe(argv[3]);Event item;while(!probe.footer&&probe.next_verified(item,allow_prefix,tail_truncated))if(item.kind==Kind::asset&&item.flags==0)require(asset_entry(item.payload.data(),unsigned(item.payload.size()))==1,"replay asset import failed");}
        WNDCLASSW wc={};wc.lpfnWndProc=DefWindowProcW;wc.hInstance=GetModuleHandleW(nullptr);wc.lpszClassName=L"C3XInputReplay";RegisterClassW(&wc);
        window=CreateWindowW(wc.lpszClassName,label.c_str(),WS_POPUP,0,0,2240,1260,nullptr,nullptr,wc.hInstance,nullptr);require(window!=nullptr,"cannot create replay target");
        if(performance||watch)ShowWindow(window,SW_SHOWNOACTIVATE);
        std::unique_ptr<c3x_helper_trial::SceneClient> shadow;
        std::unique_ptr<c3x_helper_trial::SharedFrameReader> shadow_frames;
        std::ofstream shadow_log;
        if(!shadow_path.empty()){
            require(!std::filesystem::exists(shadow_path),"x64 scene report must be new");
            shadow_log.open(shadow_path);require(bool(shadow_log),"cannot open x64 scene report");
            shadow=std::make_unique<c3x_helper_trial::SceneClient>(shadow_helper,shadow_dll);
            shadow_frames=std::make_unique<c3x_helper_trial::SharedFrameReader>();
        }
        SegmentReader reader(argv[3]);Event event;std::map<std::uint64_t,Pending> pending;
        std::uint64_t calls=0,clocks=0,queued=0,peak=0,last=0;bool manifest=false;LARGE_INTEGER frequency={},started={},ended={};QueryPerformanceFrequency(&frequency);double work_ms=0;
        LARGE_INTEGER playback_start={};QueryPerformanceCounter(&playback_start);double maximum_playback_lag_ms=0;std::uint64_t late_playback_frames=0,skipped_offers=0;
        unsigned live_counts[5]={};std::vector<double> ambient_times;double next_status=1,maximum_input_lateness_ms=0;
        if(realtime)require(execution(2,nullptr,nullptr,nullptr)==1,"DLL lacks realtime execution mode");
        auto elapsed=[&]{LARGE_INTEGER now={};QueryPerformanceCounter(&now);return double(now.QuadPart-playback_start.QuadPart)/double(frequency.QuadPart);};
        auto live_update=[&]{if(!realtime)return;double samples[4096];int count=live_status(unsigned(ambient_times.size()),4096,samples,live_counts);
            require(count>=0&&!live_counts[3]&&!live_counts[4],"autonomous playback failed or exhausted its bounded telemetry");
            for(int n=0;n<count;++n){measurements<<"{\"row\":\"ambient-present\",\"seconds\":"<<samples[n]<<"}\n";ambient_times.push_back(samples[n]);}
            if(elapsed()>=next_status){std::cout<<"realtime seconds="<<elapsed()<<" ambient_presentations="<<live_counts[1]<<" pending="<<live_counts[2]<<" max_input_lag_ms="<<maximum_input_lateness_ms<<std::endl;next_status=elapsed()+1;}
        };
        auto messages=[&]{MSG message;while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)){
            require(!(watch||realtime)||!(message.message==WM_QUIT||(message.message==WM_KEYDOWN&&message.wParam==VK_ESCAPE)),"playback stopped by user");
            TranslateMessage(&message);DispatchMessageW(&message);}live_update();require(!(watch||realtime)||IsWindow(window),"playback window closed");};
        while(!reader.footer&&reader.next_verified(event,allow_prefix,tail_truncated)){
            if(before_event && event.sequence>=before_event){stopped_before_event=true;break;}
            last=event.sequence;Reader payload{event.payload};
            if(event.kind==Kind::manifest){require(!manifest&&event.sequence==1,"duplicate/misplaced input manifest");
                require(payload.u32()==protocol_version&&payload.u32()==C3X_RENDERER_API_VERSION,"input API mismatch");payload.u64();ClockOrigin origin;clock_origin(payload,origin);auto count=payload.u32();require(count<=512,"input settings limit");for(unsigned n=0;n<count;++n){payload.string(256);payload.string(32768);}payload.done();manifest=true;continue;}
            require(manifest,"missing input manifest");if(event.kind==Kind::footer)break;
            if(event.kind==Kind::asset)continue;
            if(event.kind==Kind::settings){require(settings_versions.size()<4096,"settings epoch limit");settings_versions.push_back(event.payload);active_settings=settings_versions.size()-1;continue;}
            if(event.kind==Kind::visual&&event.flags==2){auto parent=payload.u64(),tick=payload.u64(),freq=payload.u64();payload.done();
                require(freq&&tick<=INT64_MAX&&freq<=INT64_MAX,"invalid sampled clock");++clocks;
                if(parent){auto it=pending.find(parent);require(it!=pending.end(),"clock has missing parent");require(it->second.clocks.size()<64,"excessive call clocks");it->second.clocks.emplace_back(tick,freq);}continue;}
            auto token=payload.u64();
            if(event.kind!=Kind::result){require(token&&pending.size()<256&&!pending.count(token),"invalid call token");
                require(event.payload.size()<=32u*1024u*1024u-queued,"replay pending input budget");queued+=event.payload.size();peak=std::max(peak,queued);
                auto parent=payload.u64();bool skip=realtime&&((event.kind==Kind::visual&&event.flags==1)||(pending.count(parent)&&pending.at(parent).skipped));
                pending.emplace(token,Pending{std::move(event),{},active_settings,skip});continue;}
            auto it=pending.find(token);require(it!=pending.end(),"result has missing input");auto& call=it->second;
            // A bounded tool envelope carries owned scalar records to the DLL;
            // it is decoded before any production renderer invocation.
            Writer envelope;envelope.u32(unsigned(call.input.kind));envelope.u32(call.input.flags);
            envelope.u32(unsigned(call.input.payload.size()));envelope.reserve(call.input.payload.size());envelope.bytes.insert(envelope.bytes.end(),call.input.payload.begin(),call.input.payload.end());
            envelope.u32(unsigned(event.payload.size()));envelope.reserve(event.payload.size());envelope.bytes.insert(envelope.bytes.end(),event.payload.begin(),event.payload.end());
            envelope.u32(unsigned(call.clocks.size()));for(auto const& sample:call.clocks){envelope.u64(sample.first);envelope.u64(sample.second);}
            messages();
            if(watch||realtime){for(;;){LARGE_INTEGER now={};QueryPerformanceCounter(&now);
                auto remaining=1000.*double(call.input.ticks)/double(reader.frequency)-1000.*double(now.QuadPart-playback_start.QuadPart)/double(frequency.QuadPart);
                if(remaining<=0)break;Sleep(DWORD(std::min(10.,std::max(1.,remaining))));messages();}}
            if(applied_settings!=call.settings){Reader settings{settings_versions[call.settings]};configure(settings);applied_settings=call.settings;}
            if(call.skipped){++skipped_offers;++calls;queued-=call.input.payload.size();pending.erase(it);continue;}
            auto dispatch_seconds=elapsed();auto lateness_ms=std::max(0.,1000.*(dispatch_seconds-double(call.input.ticks)/double(reader.frequency)));
            maximum_input_lateness_ms=std::max(maximum_input_lateness_ms,lateness_ms);
            char error[512]={};QueryPerformanceCounter(&started);int ok=replay(envelope.bytes.data(),unsigned(envelope.bytes.size()),window,error,sizeof(error));QueryPerformanceCounter(&ended);
            auto envelope_ms=1000.*double(ended.QuadPart-started.QuadPart)/double(frequency.QuadPart);work_ms+=envelope_ms;
            if(!ok){std::cerr<<"event="<<last<<" call="<<token<<" family="<<unsigned(call.input.kind)<<" subtype="<<call.input.flags<<" error="<<error<<'\n';throw std::runtime_error("production input replay mismatch");}
            Reader request{call.input.payload};request.u64();request.u64();Reader completion{event.payload};completion.u64();auto code=completion.u32();
            bool native_visual_policy=false;
            if(shadow&&call.input.kind==Kind::native_bridge&&call.input.flags==1){
                Reader visual_request{call.input.payload};visual_request.u64();visual_request.u64();
                int operation=0;visual_request(operation);native_visual_policy=operation==C3X_NATIVE_VISUAL_POLICY;
            }
            if(shadow&&(call.input.kind==Kind::scene||call.input.kind==Kind::image_commands||
                        (call.input.kind==Kind::configuration&&call.input.flags==3)||
                        (call.input.kind==Kind::unit&&call.input.flags==1)||
                        call.input.kind==Kind::presentation||
                        (call.input.kind==Kind::visual&&call.input.flags==1)||
                        native_visual_policy||
                        (call.input.kind==Kind::native_bridge&&(call.input.flags==6||call.input.flags==8)))){
                Reader identities{event.payload};identities.u64();identities.u32();
                std::int64_t old_ticket=0,old_image=0;
                unsigned pixel_witness=0;unsigned pixel_hash[4]={};
                int old_bounds[4]={};
                if(call.input.kind==Kind::scene&&call.input.flags==3){old_ticket=std::int64_t(identities.u64());old_image=std::int64_t(identities.u64());}
                else if(call.input.kind==Kind::image_commands){
                    old_image=std::int64_t(identities.u64());identities.u32();pixel_witness=identities.u32();
                    if(pixel_witness)for(auto& word:pixel_hash)word=identities.u32();
                }
                else if(call.input.kind==Kind::unit)for(auto& bound:old_bounds)identities(bound);
                bool final_image=false;
                if(call.input.kind==Kind::presentation){Reader value{call.input.payload};value.u64();value.u64();
                    int action=0;value(action);final_image=action==0&&code==C3X_RENDERER_RESULT_OK;}
                if(call.input.kind==Kind::visual)final_image=!call.clocks.empty();
                LARGE_INTEGER shadow_begin={},shadow_end={};QueryPerformanceCounter(&shadow_begin);
                auto const& remote=shadow->call(unsigned(call.input.kind),call.input.flags,
                    call.input.payload.data()+request.at,unsigned(call.input.payload.size()-request.at),
                    code,old_ticket,old_image,
                    call.clocks.empty()?0:std::int64_t(call.clocks.front().first),
                    call.clocks.empty()?0:std::int64_t(call.clocks.front().second),final_image);
                QueryPerformanceCounter(&shadow_end);
                bool final_valid=false,final_match=false,diff_available=false;
                unsigned final_hash[4]={};std::uint64_t different_pixels=0;
                int difference_bounds[4]={},max_channel_delta=0;
                if(remote.shared_handle){
                    auto frame=shadow_frames->read(remote.shared_handle,remote.width,remote.height);
                    std::copy(frame.hash.begin(),frame.hash.end(),final_hash);
                    unsigned local_extent[2]={},local_hash[4]={};
                    auto fingerprint=reinterpret_cast<int(*)(unsigned*,unsigned*)>(GetProcAddress(module,"c3x_renderer_input_replay_fingerprint"));
                    final_valid=fingerprint&&fingerprint(local_extent,local_hash)==1&&
                        local_extent[0]==frame.width&&local_extent[1]==frame.height;
                    final_match=final_valid&&std::equal(std::begin(local_hash),std::end(local_hash),std::begin(final_hash));
                    if(final_valid&&!final_match){
                        auto pixels=reinterpret_cast<int(*)(unsigned*,unsigned,unsigned,unsigned)>(
                            GetProcAddress(module,"c3x_renderer_input_replay_pixels"));
                        std::vector<unsigned> local(frame.pixels.size());
                        diff_available=pixels&&pixels(local.data(),unsigned(local.size()),frame.width,frame.height)==1;
                        if(diff_available){difference_bounds[0]=int(frame.width);difference_bounds[1]=int(frame.height);
                            for(unsigned y=0;y<frame.height;++y)for(unsigned x=0;x<frame.width;++x){
                                auto index=std::size_t(y)*frame.width+x;
                                auto a=local[index],b=frame.pixels[index];if(a==b)continue;
                                ++different_pixels;difference_bounds[0]=std::min(difference_bounds[0],int(x));
                                difference_bounds[1]=std::min(difference_bounds[1],int(y));
                                difference_bounds[2]=std::max(difference_bounds[2],int(x)+1);
                                difference_bounds[3]=std::max(difference_bounds[3],int(y)+1);
                                for(unsigned shift=0;shift<32;shift+=8)
                                    max_channel_delta=std::max(max_channel_delta,std::abs(int((a>>shift)&255)-int((b>>shift)&255)));
                            }}
                    }
                }
                bool pixels_match=!pixel_witness||(remote.gpu_hash_valid&&
                    std::equal(std::begin(pixel_hash),std::end(pixel_hash),std::begin(remote.gpu_hash)));
                bool bounds_match=call.input.kind!=Kind::unit||code!=C3X_RENDERER_RESULT_OK||
                    std::equal(std::begin(old_bounds),std::end(old_bounds),std::begin(remote.bounds));
                shadow_log<<"{\"sequence\":"<<call.input.sequence<<",\"kind\":"<<unsigned(call.input.kind)
                    <<",\"subtype\":"<<call.input.flags<<",\"recorded_result\":"<<code
                    <<",\"helper_result\":"<<remote.code<<",\"matches\":"<<(remote.code==code?"true":"false")
                    <<",\"bytes\":"<<remote.size<<",\"width\":"<<remote.width
                    <<",\"height\":"<<remote.height<<",\"rendered\":"<<remote.rendered
                    <<",\"fallback\":"<<remote.fallback<<",\"result_image\":"<<remote.result_image
                    <<",\"result_pixels\":"<<remote.result_pixels<<",\"pixel_witness\":"<<pixel_witness
                    <<",\"pixels_match\":"<<(pixels_match?"true":"false")
                    <<",\"bounds_match\":"<<(bounds_match?"true":"false")
                    <<",\"final_image\":"<<(remote.shared_handle?"true":"false")
                    <<",\"final_valid\":"<<(final_valid?"true":"false")
                    <<",\"final_match\":"<<(final_match?"true":"false")
                    <<",\"diff_available\":"<<(diff_available?"true":"false")
                    <<",\"different_pixels\":"<<different_pixels
                    <<",\"max_channel_delta\":"<<max_channel_delta
                    <<",\"difference_bounds\":["<<difference_bounds[0]<<","<<difference_bounds[1]
                    <<","<<difference_bounds[2]<<","<<difference_bounds[3]<<"]"
                    <<",\"executed\":"<<(remote.executed?"true":"false")
                    <<",\"service_ms\":"<<double(remote.service_us)/1000.
                    <<",\"roundtrip_ms\":"<<1000.*double(shadow_end.QuadPart-shadow_begin.QuadPart)/double(frequency.QuadPart)
                    <<",\"x64_private_bytes\":"<<remote.private_bytes<<"}\n";
                require(bool(shadow_log),"x64 scene report write failed");
            }
            if(performance){double service_ms=0;int actual=0;unsigned reused=0;require(execution(realtime?2:1,&service_ms,&actual,&reused)==1,"performance measurement missing");
                measurements<<"{\"call\":"<<token<<",\"sequence\":"<<event.sequence<<",\"input_seconds\":"<<double(call.input.ticks)/double(reader.frequency)<<",\"dispatch_seconds\":"<<dispatch_seconds<<",\"input_lateness_ms\":"<<lateness_ms<<",\"family\":"<<unsigned(call.input.kind)<<",\"subtype\":"<<call.input.flags<<",\"recorded_result\":"<<code<<",\"actual_result\":"<<actual<<",\"production_service_ms\":"<<service_ms<<",\"envelope_ms\":"<<envelope_ms<<",\"reused_adoption\":"<<(reused?"true":"false");
                if(calls%100==0){auto memory=memory_sample();measurements<<",\"private_bytes\":"<<memory.private_bytes<<",\"free_va_bytes\":"<<memory.free_bytes<<",\"largest_free_va_bytes\":"<<memory.largest_free;}
                measurements<<"}\n";require(bool(measurements),"performance measurement write failed");code=unsigned(actual);
            }
            bool boundary=(call.input.kind==Kind::native_bridge&&call.input.flags==1&&request.u32()==C3X_NATIVE_IMAGE_PRESENT)||
                (call.input.kind==Kind::presentation&&request.u32()==0)||
                (call.input.kind==Kind::visual&&call.input.flags==1)||
                (call.input.kind==Kind::native_snapshot&&call.input.flags==0&&request.u32()==1);
            if(boundary&&code==C3X_RENDERER_RESULT_OK){++frame_number;auto time=double(event.ticks)/double(reader.frequency);
                if(watch){LARGE_INTEGER now={};QueryPerformanceCounter(&now);auto lag=1000.*double(now.QuadPart-playback_start.QuadPart)/double(frequency.QuadPart)-time*1000;
                    maximum_playback_lag_ms=std::max(maximum_playback_lag_ms,lag);if(lag>33)++late_playback_frames;}
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
                    <<",\"dispatch_seconds\":"<<dispatch_seconds<<",\"input_lateness_ms\":"<<lateness_ms<<",\"family\":"<<unsigned(call.input.kind)<<",\"exported\":"<<(selected?"true":"false")<<"}\n";
            }
            ++calls;queued-=call.input.payload.size();pending.erase(it);
        }
        if(realtime){
            // Preserve a trailing interval containing no external calls.
            while(elapsed()<double(event.ticks)/double(reader.frequency)){messages();Sleep(5);}
            auto shutdown=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_input_replay_shutdown"));require(shutdown,"realtime shutdown entry missing");shutdown();live_update();
            double maximum_gap=0;for(std::size_t n=1;n<ambient_times.size();++n)maximum_gap=std::max(maximum_gap,1000.*(ambient_times[n]-ambient_times[n-1]));
            std::cout<<"{\"viewing_mode\":\"realtime_candidate\",\"recorded_ambient_calls_replaced\":"<<skipped_offers<<",\"autonomous_presentations\":"<<live_counts[1]<<",\"autonomous_pending\":"<<live_counts[2]<<",\"maximum_ambient_gap_ms_including_lifecycle\":"<<maximum_gap<<",\"maximum_input_lateness_ms\":"<<maximum_input_lateness_ms<<",\"wall_seconds\":"<<elapsed()<<"}\n";
        }
        if(timeline.is_open()){timeline.flush();require(bool(timeline),"replay timeline write failed");}
        if(fingerprints.is_open()){fingerprints.flush();require(bool(fingerprints),"replay fingerprint flush failed");}
        if(measurements.is_open()){measurements.flush();require(bool(measurements),"performance output flush failed");}
        require(!before_event||stopped_before_event,"requested event cutoff was not reached");
        bool complete=!stopped_before_event&&reader.footer&&(reader.reason==Stop::closed||reader.reason==Stop::duration)&&pending.empty()&&!tail_truncated;
        require(complete||allow_prefix,"input capture incomplete (use --allow-prefix to inspect verified completed calls)");
        if(watch)std::cout<<"{\"viewing_mode\":\"paced_forensic_playback\",\"dropped_by_player\":0,\"frames_late_over_33ms\":"<<late_playback_frames<<",\"maximum_playback_lag_ms\":"<<maximum_playback_lag_ms<<"}\n";
        std::cout<<"{\"status\":\"development_input_replay\",\"qualified\":false,\"timing_scope\":\""<<(realtime?"paced_candidate_recorded_native_consumption":performance?"unpaced_native_service_not_live_fps":"forensic_calls_not_performance")<<"\",\"reserved_va_mib\":"<<reserve_mib<<",\"calls\":"<<calls<<",\"binary_matches_capture\":"<<(binary_matches_capture?"true":"false")<<",\"complete\":"<<(complete?"true":"false")<<",\"before_event\":"<<before_event<<",\"last_event\":"<<last<<",\"unfinished_calls\":"<<pending.size()<<",\"accepted_presentations\":"<<frame_number<<",\"clock_samples\":"<<clocks<<",\"pending_peak_bytes\":"<<peak<<",\"replay_envelope_ms\":"<<work_ms<<"}\n";
    }catch(std::exception const& error){std::cerr<<error.what()<<'\n';exit_code=1;}
    // Finish exception unwinding before unloading a DLL that may have thrown.
    if(module){auto reset=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_input_replay_shutdown"));if(!reset)reset=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_reset"));if(reset)reset();}
    if(window)DestroyWindow(window);
    if(module)FreeLibrary(module);
    if(reservation)VirtualFree(reservation,0,MEM_RELEASE);
    return exit_code;
}
