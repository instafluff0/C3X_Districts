#pragma once
#include <atomic>
#include <mutex>
#include <set>
#include <thread>
#include "zoom_preview_state.h"

namespace c3x_renderer { namespace sandbox {
// The producer's serial is distinct from the coalesced policy serial. A full
// projection can still be stale when an input arrives during drawing/Present.
inline bool zoom_preview_present_is_current(std::uint64_t source_input,std::uint64_t latest_input,
        double source_zoom,double shown_zoom,double latest_zoom,double source_time_ms,double latest_input_time_ms){
    double target=double(float(latest_zoom));
    return source_input==latest_input&&std::isfinite(source_zoom)&&std::isfinite(shown_zoom)&&
        std::isfinite(target)&&target>=1.&&target<=3.&&source_zoom==target&&shown_zoom==target&&
        std::isfinite(source_time_ms)&&std::isfinite(latest_input_time_ms)&&
        source_time_ms>=0&&latest_input_time_ms>=0&&source_time_ms>=latest_input_time_ms;
}
}}

// Private standalone scheduling witness, called only after native128 adoption.
// Input has its own wall-clock producer; completed renders never advance it.
int sandbox_zoom_preview_witness(HMODULE module,HWND window,c3x_renderer_frame_v1 frame){
    using namespace c3x_renderer::sandbox;
    using Draw=int(*)(c3x_renderer_frame_v1 const*,char const*,int,int,int,int,int,int,int,float);
    using Present=int(*)(HWND,c3x_renderer_frame_v1 const*,int,int,int,int,int,int,int);
    using Commit=int(*)(unsigned,float,std::uint64_t);
    using Select=int(*)(unsigned,float,std::uint64_t);
    auto draw=reinterpret_cast<Draw>(GetProcAddress(module,"c3x_sandbox_draw_fresh"));
    auto present=reinterpret_cast<Present>(GetProcAddress(module,"c3x_sandbox_present"));
    auto commit=reinterpret_cast<Commit>(GetProcAddress(module,"c3x_sandbox_zoom_commit"));
    auto ready=reinterpret_cast<int(*)()>(GetProcAddress(module,"c3x_sandbox_zoom_ready"));
    auto select=reinterpret_cast<Select>(GetProcAddress(module,"c3x_sandbox_zoom_select"));
    auto bytes=reinterpret_cast<std::uint64_t(*)()>(GetProcAddress(module,"c3x_sandbox_zoom_bytes"));
    auto observed=reinterpret_cast<LONG(*)()>(GetProcAddress(module,"c3x_sandbox_swapchain_observed"));
    auto busy_update=reinterpret_cast<int(*)(c3x_renderer_frame_v1 const*)>(GetProcAddress(module,"c3x_sandbox_zoom_busy"));
    auto busy_metrics=reinterpret_cast<void(*)(unsigned*)>(GetProcAddress(module,"c3x_sandbox_zoom_busy_metrics"));
    auto pass_counts=reinterpret_cast<void(*)(SandboxPassCounts*)>(GetProcAddress(module,"c3x_sandbox_pass_counts"));
    char mode[32]{},gesture[32]{},visual[4*MAX_PATH]{},busy_text[8]{};
    GetEnvironmentVariableA("C3X_ZOOM_PREVIEW_MODE",mode,sizeof(mode));
    GetEnvironmentVariableA("C3X_ZOOM_PREVIEW_GESTURE",gesture,sizeof(gesture));
    GetEnvironmentVariableA("C3X_ZOOM_PREVIEW_VISUAL",visual,sizeof(visual));
    GetEnvironmentVariableA("C3X_ZOOM_PREVIEW_BUSY",busy_text,sizeof(busy_text));
    bool control=!std::strcmp(mode,"control"),preview_only=!std::strcmp(mode,"preview");
    bool busy=busy_text[0]=='1';
    if(!draw||!present||(!control&&(!commit||!select||!bytes||!ready))||(busy&&(!busy_update||!busy_metrics)))return 40;
    char counts_text[8]{};GetEnvironmentVariableA("C3X_SANDBOX_PASS_COUNTS",counts_text,sizeof(counts_text));
    if(counts_text[0]!='1')pass_counts=nullptr;else if(!pass_counts)return 50;
    ZoomPreviewIdentity identity{};identity.map_epoch=1;identity.viewer_epoch=2;
    identity.visibility_epoch=3;identity.scene_epoch=4;identity.native_width=frame.tile_width;
    identity.target_width=frame.target_width;identity.target_height=frame.target_height;
    std::set<int> cities;for(unsigned n=0;n<frame.tile_count;++n)if(frame.tiles[n].city_id>=0)cities.insert(frame.tiles[n].city_id);
    LARGE_INTEGER frequency{},start{};QueryPerformanceFrequency(&frequency);
    auto tick=[](){LARGE_INTEGER t{};QueryPerformanceCounter(&t);return t.QuadPart;};
    auto ms=[&](LONGLONG t){return double(t)*1000./frequency.QuadPart;};
    // Initial source was drawn at29500 by the adoption path. Snapshot it before
    // starting interaction measurements; GPU readiness follows ordered Present.
    if(busy){if(busy_update(&frame)||draw(&frame,nullptr,0,0,24,56,1,47,1,1.f))return 41;}
    if(!control&&(commit(0,1.f,1)||select(0,1.f,1)||present(window,&frame,24,56,1,47,1,0,0)))return 42;
    std::array<unsigned,3> initial_busy{};if(busy)busy_metrics(initial_busy.data());
    if(!control){auto deadline=GetTickCount64()+10000;int pending=ready();
        while(pending==1&&GetTickCount64()<deadline){Sleep(1);pending=ready();}
        if(pending)return 50;
    }
    // Establish the same GPU-ready initial map in control and candidate.
    // This untimed original color/depth readback is excluded from interaction.
    char seed_path[4*MAX_PATH]{};
    auto capture=reinterpret_cast<int(*)(char const*)>(GetProcAddress(module,"c3x_sandbox_witness_capture"));
    if(!capture||!GetEnvironmentVariableA("C3X_ZOOM_PREVIEW_SEED_CAPTURE",seed_path,sizeof(seed_path))||capture(seed_path))return 53;
    QueryPerformanceCounter(&start);
    auto now=[&](){return ms(tick()-start.QuadPart);};
    ZoomPreviewState policy;policy.request(1.,identity,0);
    ZoomPreviewSource initial{};initial.id=1;initial.identity=identity;policy.seed(initial);
    struct Input{double time=0,zoom=1;std::uint64_t serial=0;};
    Input latest{};std::vector<Input> events;std::mutex input_mutex;
    std::atomic<bool> stop{false};std::atomic<double> close_started{-1};
    bool sustained=!std::strcmp(gesture,"sustained"),resume=!std::strcmp(gesture,"resume");
    double duration=sustained?3600.:2200.;
    auto path=[&](double t){
        if(t<200)return 1.;
        if(sustained){
            if(t>=2700)return 1.;
            double u=std::fmod(t-200,1000.);return 1.+.25*(u<=500?u/500:(1000-u)/500);
        }
        if(t<550)return 1.+.25*(t-200)/350.;
        double reverse=1000.;double begun=close_started.load();
        if(resume&&begun>=0)reverse=begun+5.;
        if(t<reverse)return 1.25;
        if(t<reverse+350)return 1.25-.25*(t-reverse)/350.;
        return 1.;
    };
    std::thread producer([&](){
        double previous=1.;std::uint64_t serial=0;
        while(!stop.load()){
            double t=now(),zoom=path(t);
            if(zoom!=previous){std::lock_guard<std::mutex> lock(input_mutex);
                latest={now(),zoom,++serial};events.push_back(latest);previous=zoom;}
            Sleep(1);
        }
    });
    auto sample=[&](){std::lock_guard<std::mutex> lock(input_mutex);return latest;};
    struct Record{double start=0,done=0,draw=0,present=0,shown=1,requested_done=1,completed_zoom=1,age=0;
        double source_time=0;std::uint64_t input=0,latest_input=0,source_input=0,completed_input=0,source=0,memory=0;
        unsigned redraw=0,wide=0,poses=0,moving=0,parts=0;LONG observed=-1;bool quality=false;};
    std::vector<Record> records;
    struct UnitSubmission{std::uint64_t draws=0,triangles=0,submitted_instances=0,index_vertices=0;};
    struct JobRecord{std::uint64_t id=0,input=0;double start=0,submitted=0,presented=0,gpu_observed=-1,gpu_last_pending=-1,zoom=1;bool wide=false,counts=false;
        std::array<UnitSubmission,3> units{};};
    std::vector<JobRecord> jobs;
    struct SourceStamp{std::uint64_t id=0,input=0;double time=0;std::array<unsigned,3> busy{};};
    SourceStamp stamps[2]={{1,0,0,initial_busy},{}};
    auto last_generated_busy=initial_busy;
    std::uint64_t consumed=0,source_serial=1;double last_visual=-1000;int result=0;
    double completed_zoom=1.;std::uint64_t completed_input=0;
    struct Pending{bool valid=false;unsigned slot=0;std::size_t record=0;
        ZoomPreviewRefinement job{};ZoomPreviewSource source{};SourceStamp stamp{};}pending;
    auto source_slot=[&](std::uint64_t id){return stamps[0].id==id?0:stamps[1].id==id?1:-1;};
    while((now()<duration||pending.valid)&&now()<duration+2000&&records.size()<600){
        MSG message{};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
        Input input=sample();double entered=now();
        if(input.serial!=consumed){policy.request(float(input.zoom),identity,std::uint64_t(input.time));consumed=input.serial;}
        bool adopted=false;
        if(pending.valid){
            int status=ready();double observed_time=now();
            if(status<0){result=51;break;}
            if(status==1)jobs[pending.record].gpu_last_pending=observed_time;
            else{
                pending.source.completed_time_ms=std::uint64_t(observed_time);
                if(!policy.complete(pending.job,pending.source)){result=47;break;}
                stamps[pending.slot]=pending.stamp;completed_zoom=pending.source.zoom;completed_input=pending.stamp.input;
                jobs[pending.record].gpu_observed=observed_time;pending.valid=false;adopted=true;
            }
        }
        ZoomPreviewRefinement job{};
        bool render=control||(!preview_only&&entered<duration&&!pending.valid&&!adopted&&policy.begin(std::uint64_t(entered),job));
        double render_zoom=control?input.zoom:job.zoom;
        std::uint64_t source_time=control?std::uint64_t(entered):job.source_time_ms;
        frame.presentation_frequency=1000;frame.presentation_time_ticks=29500+source_time;
        double submitted=entered;SourceStamp completed_stamp{};JobRecord job_record{};
        if(render){
            if(resume&&render_zoom==1.25&&close_started.load()<0)close_started.store(entered);
            // Preserve the only wide coverage through arbitrary input reversal.
            // Pending refinement always writes the other slot, even if a close
            // source was previously displayed. Never spend unrendered margins.
            auto const& wide=policy.wide_source();
            int old_slot=control?0:wide.id&&wide.coverage_min_zoom==1.?source_slot(wide.id):-1;
            if(!control&&old_slot<0){result=52;break;}
            unsigned destination=unsigned(1-old_slot);
            if(!control)policy.discard_source(stamps[destination].id);
            if(busy&&busy_update(&frame)){result=43;break;}
            if(draw(&frame,nullptr,0,0,24,56,1,47,1,float(render_zoom))){result=44;break;}
            if(busy)busy_metrics(last_generated_busy.data());
            completed_stamp={control?records.size()+1:source_serial+1,input.serial,double(source_time),last_generated_busy};
            if(pass_counts){SandboxPassCounts work{};pass_counts(&work);job_record.counts=true;
                unsigned const selected[]={SandboxPassCounts::units,SandboxPassCounts::reflected_units,SandboxPassCounts::unit_shadow};
                for(unsigned q=0;q<3;++q)for(auto const& row:work.counts[selected[q]]){
                    auto& total=job_record.units[q];total.draws+=row.draws;total.triangles+=row.triangles;
                    total.submitted_instances+=row.submitted_instances;total.index_vertices+=row.index_vertices;
                }
            }
            if(!control){
                ++source_serial;
                if(commit(destination,float(render_zoom),source_serial)){result=45;break;}
                pending.valid=true;pending.slot=destination;pending.record=jobs.size();pending.job=job;pending.stamp=completed_stamp;
                pending.source.id=source_serial;pending.source.identity=identity;pending.source.zoom=render_zoom;
                pending.source.coverage_min_zoom=render_zoom;pending.source.source_time_ms=source_time;
            }
            submitted=now();
            job_record.id=control?records.size()+1:job.id;job_record.input=input.serial;
            job_record.start=entered;job_record.submitted=submitted;job_record.zoom=render_zoom;
            job_record.wide=!control&&job.refresh_wide;jobs.push_back(job_record);
        }
        Input before_present=sample();
        if(!control&&before_present.serial!=consumed){policy.request(float(before_present.zoom),identity,std::uint64_t(before_present.time));consumed=before_present.serial;}
        auto shown=policy.preview(std::uint64_t(submitted));
        int selected_slot=control?0:shown.valid?source_slot(shown.source.id):-1;
        if(!control&&(selected_slot<0||select(unsigned(selected_slot),float(shown.absolute_zoom),shown.source.id))){result=46;break;}
        if(visual[0]&&submitted-last_visual>=100){SetEnvironmentVariableA("C3X_SANDBOX_CAPTURE_SEQUENCE",visual);last_visual=submitted;}
        double presenting=now();
        result=present(window,&frame,24,56,1,47,1,0,0);
        double done=0;Input at_done{};{std::lock_guard<std::mutex> lock(input_mutex);done=now();at_done=latest;}
        SetEnvironmentVariableA("C3X_SANDBOX_CAPTURE_SEQUENCE",nullptr);
        if(result){policy.present_failed();break;}
        if(at_done.serial!=consumed){policy.request(float(at_done.zoom),identity,std::uint64_t(at_done.time));consumed=at_done.serial;}
        if(!control&&!policy.present_success(shown)){result=48;break;}
        auto displayed_stamp=control?completed_stamp:stamps[selected_slot];
        Record record{};record.start=entered;record.done=done;record.draw=submitted-entered;record.present=done-presenting;
        record.shown=control?float(input.zoom):shown.absolute_zoom;record.requested_done=at_done.zoom;
        record.completed_zoom=control?input.zoom:completed_zoom;record.input=control?input.serial:before_present.serial;record.latest_input=at_done.serial;
        record.completed_input=control?input.serial:completed_input;record.source_input=displayed_stamp.input;record.source_time=displayed_stamp.time;
        record.source=displayed_stamp.id;record.age=done-displayed_stamp.time;
        record.memory=bytes?bytes():0;record.redraw=render;record.wide=!control&&render&&job.refresh_wide;
        record.poses=displayed_stamp.busy[0];record.moving=displayed_stamp.busy[1];record.parts=displayed_stamp.busy[2];
        record.observed=observed?observed():-1;
        record.quality=zoom_preview_present_is_current(displayed_stamp.input,at_done.serial,
            control?double(float(input.zoom)):shown.source.zoom,record.shown,at_done.zoom,displayed_stamp.time,at_done.time);
        records.push_back(record);
        if(render)jobs.back().presented=done;
    }
    stop.store(true);producer.join();
    std::printf("ZOOM_WORKLOAD mode=%s gesture=%s cities=%zu tiles=%u native_width=%d busy=%u identity=1,2,3,4 debounce_ms=60 refresh_due_ms=250 input_clock=independent_QPC wall_duration_ms=%.6f result=%d gpu_completion_gate=%u pending_at_end=%u\n",
        mode,gesture,cities.size(),frame.tile_count,frame.tile_width,unsigned(busy),now(),result,unsigned(!control&&!preview_only),unsigned(pending.valid));
    for(auto const& input:events)std::printf("ZOOM_INPUT serial=%llu time_ms=%.6f zoom=%.9f\n",static_cast<unsigned long long>(input.serial),input.time,input.zoom);
    for(std::size_t index=0;index<records.size();++index){auto const& r=records[index];
        std::printf("ZOOM_FRAME index=%zu start_ms=%.6f done_ms=%.6f draw_cpu_ms=%.6f present_ms=%.6f shown=%.9f requested_done=%.9f completed_zoom=%.9f input=%llu latest_input=%llu source_input=%llu completed_input=%llu source=%llu source_time_ms=%.6f source_age_ms=%.6f redraw=%u wide_refresh=%u quality=%u bytes=%llu poses=%u moving=%u parts=%u observed=%ld\n",
            index,r.start,r.done,r.draw,r.present,r.shown,r.requested_done,r.completed_zoom,
            static_cast<unsigned long long>(r.input),static_cast<unsigned long long>(r.latest_input),static_cast<unsigned long long>(r.source_input),
            static_cast<unsigned long long>(r.completed_input),static_cast<unsigned long long>(r.source),r.source_time,r.age,r.redraw,r.wide,unsigned(r.quality),
            static_cast<unsigned long long>(r.memory),r.poses,r.moving,r.parts,r.observed);}
    for(auto const& j:jobs){std::printf("ZOOM_REDRAW id=%llu input=%llu start_ms=%.6f submitted_ms=%.6f presented_ms=%.6f zoom=%.9f wide=%u gpu_observed_ms=%.6f gpu_last_pending_ms=%.6f\n",
        static_cast<unsigned long long>(j.id),static_cast<unsigned long long>(j.input),j.start,j.submitted,j.presented,j.zoom,unsigned(j.wide),j.gpu_observed,j.gpu_last_pending);
        if(j.counts)for(unsigned p=0;p<3;++p){auto const& c=j.units[p];char const* names[]={"units","reflected_units","unit_shadow"};
            std::printf("ZOOM_UNIT_SUBMISSION redraw=%llu pass=%s draws=%llu triangles=%llu submitted_instances=%llu index_vertices=%llu\n",
                static_cast<unsigned long long>(j.id),names[p],static_cast<unsigned long long>(c.draws),static_cast<unsigned long long>(c.triangles),
                static_cast<unsigned long long>(c.submitted_instances),static_cast<unsigned long long>(c.index_vertices));}
    }
    if(busy){std::array<unsigned,3> final_busy{};busy_metrics(final_busy.data());
        std::printf("ZOOM_BUSY_FINAL poses=%u moving=%u parts=%u agrees_last_full_source=%u full_redraws=%zu\n",
            final_busy[0],final_busy[1],final_busy[2],unsigned(final_busy==last_generated_busy),jobs.size());}
    std::fflush(stdout);return result;
}
