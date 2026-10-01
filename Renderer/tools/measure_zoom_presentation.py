"""Private HWND frame admission comparison from the frozen zoom prototype."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
from Renderer.lab.platform import windows_root, native_command_result
from Renderer.tools import measure_redraw_navigation as navigation
from Renderer.tools import measure_zoom_preview as preview
from Renderer.tools.measure_redraw_submission import archive_outputs

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'Renderer/.cache/zoom-preview-step'
BASE = ROOT / 'Renderer/.cache/redraw-underlay-step'
OUT = ROOT / 'Renderer/.cache/zoom-presentation-step'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if OUT.exists():
        raise ValueError('Preserve existing zoom-presentation evidence')
    manifest = json.loads((FROZEN / 'source-manifest.json').read_text())
    OUT.mkdir()
    source = OUT / 'sources'
    for relative, expected in manifest['after'].items():
        original = FROZEN / 'sources' / relative
        if sha(original) != expected:
            raise ValueError('Frozen preview source changed: ' + relative)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    for name in ('zoom_presentation_permit.h', 'zoom_presentation_probe.h'):
        relative = 'Renderer/sandbox/' + name
        shutil.copy2(ROOT / relative, source / relative)
    before = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    path = source / 'Renderer/sandbox/resident_scene.cpp'
    text = path.read_text()

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Presentation insertion changed: ' + old[:90])
        text = text.replace(old, new)

    replace('static double sandbox_present_phases[4] = {};', r'''static double sandbox_present_phases[4] = {};
#include "zoom_presentation_permit.h"
#include "zoom_presentation_probe.h"
static c3x_renderer::sandbox::ZoomPresentationPermit sandbox_presentation_permit;
static c3x_renderer::sandbox::ZoomPresentationProbe sandbox_presentation_probe;
static IDXGISwapChain1* sandbox_presentation_swap=nullptr;
static bool sandbox_waitable_mode(){
    char value[16]{};GetEnvironmentVariableA("C3X_ZOOM_PRESENT_MODE",value,sizeof(value));
    return std::strcmp(value,"waitable")==0;
}
extern "C" __declspec(dllexport) void c3x_sandbox_presentation_probe(c3x_renderer::sandbox::ZoomPresentationProbe* value){
    if(sandbox_presentation_swap){
        DXGI_FRAME_STATISTICS statistics{};UINT counter=0;
        sandbox_presentation_probe.counter_status=int(sandbox_presentation_swap->GetLastPresentCount(&counter));
        sandbox_presentation_probe.last_present=counter;
        sandbox_presentation_probe.statistics_status=int(sandbox_presentation_swap->GetFrameStatistics(&statistics));
        sandbox_presentation_probe.present_count=statistics.PresentCount;
        sandbox_presentation_probe.present_refresh=statistics.PresentRefreshCount;
        sandbox_presentation_probe.sync_refresh=statistics.SyncRefreshCount;
        sandbox_presentation_probe.sync_qpc=statistics.SyncQPCTime.QuadPart;
        sandbox_presentation_probe.sync_gpu_qpc=statistics.SyncGPUTime.QuadPart;
        LARGE_INTEGER observation{};QueryPerformanceCounter(&observation);
        sandbox_presentation_probe.observation_qpc=observation.QuadPart;
    }
    if(value)*value=sandbox_presentation_probe;
}
// Called on the HWND owner before sampling input or submitting scene work.
// No renderer transaction or input mutex is held across this message-aware wait.
extern "C" __declspec(dllexport) int c3x_sandbox_present_admit(DWORD timeout){
    using c3x_renderer::sandbox::ZoomPresentationWait;
    if(!sandbox_waitable_mode())return 0;
    if(!sandbox_presentation_permit.valid())return -1;
    ++sandbox_presentation_probe.admissions;
    try{
    if(sandbox_presentation_permit.ready())return 0;
    ++sandbox_presentation_probe.denials;++sandbox_presentation_probe.waits;
    auto result=sandbox_presentation_permit.wait(timeout);
    if(result==ZoomPresentationWait::granted)return 0;
    if(result==ZoomPresentationWait::message){++sandbox_presentation_probe.message_wakes;return 1;}
    if(result==ZoomPresentationWait::timeout){++sandbox_presentation_probe.timeouts;return 2;}
    return -1;
    }catch(std::exception const&){return -1;}
}''')
    replace('    if (window != bound_window) {', '''    if (window != bound_window) {
        sandbox_presentation_permit.reset();sandbox_presentation_probe={};
        sandbox_presentation_swap=nullptr;
        sandbox_presentation_probe.waitable=sandbox_waitable_mode()?1:0;''')
    replace('            description.Scaling = DXGI_SCALING_STRETCH;', '''            description.Scaling = DXGI_SCALING_STRETCH;
            if(sandbox_presentation_probe.waitable)
                description.Flags=DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;''')
    replace('        result=swap->GetBuffer(0,__uuidof(ID3D11Texture2D),', r'''        if(sandbox_presentation_probe.waitable){
            Microsoft::WRL::ComPtr<IDXGISwapChain2> latency;
            result=swap->QueryInterface(IID_PPV_ARGS(&latency));
            if(SUCCEEDED(result))result=latency->SetMaximumFrameLatency(1);
            if(SUCCEEDED(result)){
                HANDLE signal=latency->GetFrameLatencyWaitableObject();
                if(signal)sandbox_presentation_permit.reset(signal);else result=E_FAIL;
            }
            if(FAILED(result)){
                std::printf("SANDBOX_ADMISSION_ERROR hresult=0x%08lx fallback=none\n",result);
                std::fflush(stdout);return 7;
            }
        }
        result=swap->GetBuffer(0,__uuidof(ID3D11Texture2D),''')
    replace('        bound_window = window;', '        bound_window = window;sandbox_presentation_swap=swap;')
    replace('    int width = renderer.content_view_width, height = renderer.content_view_height;', r'''    if(sandbox_presentation_probe.waitable){
        int admitted=c3x_sandbox_present_admit(2000);
        while(admitted==1){
            MSG message{};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){
                if(message.message==WM_QUIT)return 8;
                TranslateMessage(&message);DispatchMessageA(&message);
            }
            admitted=c3x_sandbox_present_admit(2000);
        }
        if(admitted){std::printf("SANDBOX_ADMISSION_ERROR status=%d fallback=none\n",admitted);return 8;}
    }
    QueryPerformanceCounter(&ticks[1]);
    int width = renderer.content_view_width, height = renderer.content_view_height;''')
    replace('    QueryPerformanceCounter(&ticks[1]);\n    char unit_option[8]{};', '    char unit_option[8]{};')
    replace('        swap->Present(std::strcmp(present_mode,"immediate")==0?0:1,0);', '''        swap->Present(sandbox_presentation_probe.waitable?0:
            (std::strcmp(present_mode,"immediate")==0?0:1),0);
    sandbox_presentation_probe.present_status=int(result);
    if(result==S_OK && std::strcmp(present_mode,"skip")!=0 && sandbox_presentation_probe.waitable)
        sandbox_presentation_permit.presented();''')
    replace('    DXGI_FRAME_STATISTICS statistics{};\n    if (std::strcmp(present_mode,"skip")!=0', '''    DXGI_FRAME_STATISTICS statistics{};
    UINT counter=0;
    sandbox_presentation_probe.counter_status=int(swap->GetLastPresentCount(&counter));
    sandbox_presentation_probe.last_present=counter;
    sandbox_presentation_probe.statistics_status=int(swap->GetFrameStatistics(&statistics));
    sandbox_presentation_probe.present_count=statistics.PresentCount;
    sandbox_presentation_probe.present_refresh=statistics.PresentRefreshCount;
    sandbox_presentation_probe.sync_refresh=statistics.SyncRefreshCount;
    sandbox_presentation_probe.sync_qpc=statistics.SyncQPCTime.QuadPart;
    sandbox_presentation_probe.sync_gpu_qpc=statistics.SyncGPUTime.QuadPart;
    LARGE_INTEGER observation{};QueryPerformanceCounter(&observation);
    sandbox_presentation_probe.observation_qpc=observation.QuadPart;
    if (std::strcmp(present_mode,"skip")!=0''')
    replace('        SUCCEEDED(swap->GetFrameStatistics(&statistics)))',
        '        sandbox_presentation_probe.statistics_status==0)')
    # Retain the exact S_OK contract; an occluded status must not advance policy.
    replace('    return SUCCEEDED(result) ? 0 : 4;', '    return result==S_OK ? 0 : 4;')
    path.write_text(text)

    path = source / 'Renderer/sandbox/zoom_preview_witness.h'
    text = path.read_text()
    def witness_replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Admission witness changed: ' + old[:90])
        text = text.replace(old, new)
    witness_replace('#include "zoom_preview_state.h"', '#include "zoom_preview_state.h"\n#include "zoom_presentation_probe.h"')
    witness_replace('records.size()<600', 'records.size()<100000')
    witness_replace('    auto draw=reinterpret_cast<Draw>', '''    auto admit=reinterpret_cast<int(*)(DWORD)>(GetProcAddress(module,"c3x_sandbox_present_admit"));
    auto phases=reinterpret_cast<void(*)(double*)>(GetProcAddress(module,"c3x_sandbox_present_metrics"));
    auto probe=reinterpret_cast<void(*)(ZoomPresentationProbe*)>(GetProcAddress(module,"c3x_sandbox_presentation_probe"));
    auto fresh_metrics=reinterpret_cast<void(*)(double*,unsigned*,unsigned*,unsigned*,std::size_t*,float*)>(GetProcAddress(module,"c3x_sandbox_fresh_metrics"));
    char presentation_mode[16]{};GetEnvironmentVariableA("C3X_ZOOM_PRESENT_MODE",presentation_mode,sizeof(presentation_mode));
    bool waitable=std::strcmp(presentation_mode,"waitable")==0;
    if(waitable&&(!admit||!probe||!phases))return 54;
    auto draw=reinterpret_cast<Draw>''')
    witness_replace('unsigned redraw=0,wide=0,poses=0,moving=0,parts=0;LONG observed=-1;bool quality=false;};', '''unsigned redraw=0,wide=0,poses=0,moving=0,parts=0;LONG observed=-1;bool quality=false;
        double opportunity=0,wait=0;std::array<double,4> phases{};ZoomPresentationProbe probe{};};''')
    witness_replace('''        MSG message{};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
        Input input=sample();double entered=now();''', '''        double opportunity=now();
        MSG message{};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
        if(admit){
            int admission=admit(1000);
            while(admission==1||admission==2){
                while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){
                    if(message.message==WM_QUIT){result=55;break;}
                    TranslateMessage(&message);DispatchMessageA(&message);
                }
                if(result||now()>=duration+2000)break;
                admission=admit(1000);
            }
            if(result||admission){result=56;break;}
        }
        Input input=sample();double entered=now();''')
    witness_replace('''        Record record{};record.start=entered;record.done=done;record.draw=submitted-entered;record.present=done-presenting;''', '''        Record record{};record.start=entered;record.done=done;record.draw=submitted-entered;record.present=done-presenting;
        record.opportunity=opportunity;record.wait=entered-opportunity;
        if(phases)phases(record.phases.data());if(probe)probe(&record.probe);''')
    witness_replace('''        std::array<UnitSubmission,3> units{};};''', '''        std::array<UnitSubmission,3> units{};std::array<double,6> phases{};double drawn=0;};''')
    witness_replace('''            if(busy)busy_metrics(last_generated_busy.data());''', '''            job_record.drawn=now();
            if(fresh_metrics)fresh_metrics(job_record.phases.data(),nullptr,nullptr,nullptr,nullptr,nullptr);
            if(busy)busy_metrics(last_generated_busy.data());''')
    witness_replace('''        if(j.counts)for(unsigned p=0;p<3;++p)''', r'''        std::printf("ZOOM_REFINEMENT id=%llu prepare_ms=%.6f reflection_ms=%.6f static_ms=%.6f water_ms=%.6f units_ms=%.6f reconstruct_ms=%.6f snapshot_ms=%.6f\n",
            static_cast<unsigned long long>(j.id),j.phases[0],j.phases[1],j.phases[2],j.phases[3],j.phases[4],j.phases[5],j.submitted-j.drawn);
        if(j.counts)for(unsigned p=0;p<3;++p)''')
    witness_replace('''    stop.store(true);producer.join();''', r'''    stop.store(true);producer.join();
    if(!result&&now()<duration)result=58;
    ZoomPresentationProbe tail{};
    if(probe){
        double deadline=now()+100.;probe(&tail);
        while(now()<deadline&&tail.statistics_status==0&&tail.counter_status==0&&tail.present_count<tail.last_present){
            MSG message{};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
            DWORD remaining=DWORD(std::max(1.,std::ceil(deadline-now())));
            MsgWaitForMultipleObjectsEx(0,nullptr,remaining,QS_ALLINPUT,MWMO_INPUTAVAILABLE);probe(&tail);
        }
        std::printf("ZOOM_DELIVERY_TAIL time_ms=%.6f statistics_status=%d counter_status=%d last_present=%u present_count=%u present_refresh=%u sync_refresh=%u sync_qpc=%lld observation_qpc=%lld\n",
            now(),tail.statistics_status,tail.counter_status,tail.last_present,tail.present_count,tail.present_refresh,tail.sync_refresh,tail.sync_qpc,tail.observation_qpc);
    }''')
    witness_replace('''    for(auto const& j:jobs){''', r'''    std::printf("ZOOM_CLOCK frequency=%lld start_qpc=%lld\n",frequency.QuadPart,start.QuadPart);
    for(std::size_t index=0;index<records.size();++index){auto const& r=records[index];auto const& p=r.probe;
        std::printf("ZOOM_PRESENT index=%zu opportunity_ms=%.6f admitted_ms=%.6f wait_ms=%.6f setup_ms=%.6f output_ms=%.6f capture_ms=%.6f dxgi_present_ms=%.6f waitable=%d present_status=%d statistics_status=%d counter_status=%d last_present=%u present_count=%u present_refresh=%u sync_refresh=%u sync_qpc=%lld sync_gpu_qpc=%lld observation_qpc=%lld admissions=%llu denials=%llu waits=%llu message_wakes=%llu timeouts=%llu\n",
            index,r.opportunity,r.start,r.wait,r.phases[0],r.phases[1],r.phases[2],r.phases[3],p.waitable,p.present_status,p.statistics_status,p.counter_status,
            p.last_present,p.present_count,p.present_refresh,p.sync_refresh,p.sync_qpc,p.sync_gpu_qpc,p.observation_qpc,
            static_cast<unsigned long long>(p.admissions),static_cast<unsigned long long>(p.denials),static_cast<unsigned long long>(p.waits),
            static_cast<unsigned long long>(p.message_wakes),static_cast<unsigned long long>(p.timeouts));
    }
    std::printf("ZOOM_PRESENTER mode=%s waitable=%u present_interval=%u max_latency=%u lock_during_wait=0\n",
        presentation_mode,unsigned(waitable),waitable?0u:1u,waitable?1u:0u);
    for(auto const& j:jobs){''')
    path.write_text(text)
    for arm in ('candidate', 'client'):
        (OUT / arm / 'obj').mkdir(parents=True)
        shutil.copy2(FROZEN / arm / 'build.bat', OUT / arm / 'build.bat')
    (OUT / 'frozen').mkdir()
    os.link(FROZEN / 'candidate/C3XRenderer_x64.dll', OUT / 'frozen/C3XRenderer_x64.dll')
    (OUT / 'accepted').mkdir()
    (OUT / 'accepted/shaders').symlink_to(BASE / 'accepted/shaders', target_is_directory=True)
    shutil.copy2(FROZEN / 'keep-awake.ps1', OUT / 'keep-awake.ps1')
    after = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    protected = ['C3X.h', 'injected_code.c', 'ep.c', 'civ_prog_objects.csv', 'ref/Civ3Conquests.h']
    (OUT / 'source-manifest.json').write_text(json.dumps(dict(frozen_preview=manifest['after'],
        runtime=manifest['c7_runtime'], before=before, after=after,
        changed=[n for n in before if before[n] != after[n]],
        protected={n: sha(ROOT / n) if (ROOT / n).is_file() else None for n in protected}), indent=2) + '\n')


def build(arm):
    batch = windows_root() / OUT.relative_to(ROOT).as_posix() / arm / 'build.bat'
    result = native_command_result(str((OUT / 'sources/Renderer/native').relative_to(ROOT)),
        'call "' + str(batch) + '"', timeout_seconds=600)
    (OUT / arm / 'build-result.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'pass':
        raise ValueError('Build failed: ' + arm)
    manifest = json.loads((OUT / 'source-manifest.json').read_text())
    manifest.setdefault('binaries', {})[arm + ('/client_x64.exe' if arm == 'client' else '/C3XRenderer_x64.dll')] = sha(
        OUT / arm / ('client_x64.exe' if arm == 'client' else 'C3XRenderer_x64.dll'))
    (OUT / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run(label, presentation='vsync', mode='refine', gesture='ordinary', hour=12, busy=False, frozen=False, counts=False):
    navigation.OUT = OUT
    directory = windows_root() / OUT.relative_to(ROOT).as_posix() / label
    options = dict(C3X_ZOOM_PREVIEW_MODE=mode, C3X_ZOOM_PREVIEW_GESTURE=gesture,
        C3X_ZOOM_PRESENT_MODE=presentation, C3X_ZOOM_PREVIEW_SEED_CAPTURE=str(directory / 'seed'),
        C3X_ZOOM_PREVIEW_BUSY='1' if busy else '0', C3X_ZOOM_PREVIEW_VISUAL='',
        C3X_RENDERER_PATCH_PIXELS='0', C3X_SANDBOX_PASS_COUNTS='1' if counts else '0',
        C3X_RENDERER_TRACE='0', C3X_RENDERER_PROFILE='0', C3X_SANDBOX_PROFILE_COMPLETION='0',
        C3X_RENDERER_OUTPUT_COMPLETION_PROBE='0', C3X_SANDBOX_GPU_TIMESTAMPS='0',
        C3X_SANDBOX_FRAME_TIMINGS='0', C3X_SANDBOX_WITNESS_SCENARIO='zoom',
        C3X_SANDBOX_ZOOM_CHECKPOINTS='0', C3X_SANDBOX_CAUSAL_MODE='0', C3X_SANDBOX_CAUSAL_LEDGER='0',
        C3X_SANDBOX_WORLD_CONTENT_DIR='', C3X_SANDBOX_UNDERLAY_PROBE='0',
        C3X_RENDERER_SHADER_SOURCE_ROOT=str(windows_root() / BASE.relative_to(ROOT).as_posix() / 'accepted/shaders'))
    result = navigation.run(label, 'frozen' if frozen else 'candidate', 'navigation', hour, 1,
        options, False, 'full_guest', 'client')
    archive_outputs(OUT / label)
    analyze(label)
    return result


def analyze(label):
    preview.OUT = OUT
    preview.analyze(label)
    directory = OUT / label
    groups = {}
    for line in (directory / 'run.log').read_text(errors='replace').splitlines():
        if not line.startswith(('ZOOM_PRESENT ', 'ZOOM_PRESENTER ', 'ZOOM_REFINEMENT ', 'ZOOM_DELIVERY_TAIL ', 'ZOOM_CLOCK ')):
            continue
        name, rest = line.split(' ', 1)
        fields = dict(re.findall(r'(\w+)=(\S+)', rest))
        for key, value in fields.items():
            try:
                fields[key] = float(value) if '.' in value else int(value)
            except ValueError:
                pass
        groups.setdefault(name, []).append(fields)
    summary = json.loads((directory / 'zoom-summary.json').read_text())
    frames = summary['records']['ZOOM_FRAME']
    observations = groups.get('ZOOM_PRESENT', [])
    intervals = [frames[n]['done_ms'] - frames[n-1]['done_ms'] for n in range(1, len(frames))]
    def dist(values):
        d = navigation.distribution(values)
        d['over_25ms'] = sum(v > 25 for v in values)
        d['over_33_33ms'] = sum(v > 2000/60 for v in values)
        d['over_50ms'] = sum(v > 50 for v in values)
        d['over_100ms'] = sum(v > 100 for v in values)
        return d
    result = dict(records=groups, intervals=dist(intervals),
        phases={name: dist([x[name] for x in observations]) for name in
            ('wait_ms', 'setup_ms', 'output_ms', 'capture_ms', 'dxgi_present_ms')},
        successful_present_returns=sum(x['present_status'] == 0 for x in observations),
        valid_statistics=sum(x['statistics_status'] == 0 for x in observations),
        valid_counters=sum(x['counter_status'] == 0 for x in observations),
        statistics_statuses=sorted(set(x['statistics_status'] for x in observations)),
        counter_statuses=sorted(set(x['counter_status'] for x in observations)))
    valid = [x for x in observations if x['statistics_status'] == 0]
    advancing = [x for n, x in enumerate(valid) if n == 0 or x['present_count'] > valid[n-1]['present_count']]
    result['statistics_advancing_samples'] = len(advancing)
    result['display_observation_intervals'] = dist([
        (advancing[n]['sync_qpc'] - advancing[n-1]['sync_qpc']) * 1000 /
        float(re.search(r'ZOOM_CLOCK frequency=(\d+)', (directory / 'run.log').read_text()).group(1))
        for n in range(1, len(advancing))]) if 'ZOOM_CLOCK frequency=' in (directory / 'run.log').read_text() else {}
    result['statistics_count_delta'] = valid[-1]['present_count'] - valid[0]['present_count'] if len(valid) > 1 else None
    result['submission_count_delta'] = observations[-1]['last_present'] - observations[0]['last_present'] if len(observations) > 1 else None
    id_spans = [x['last_present'] - x['present_count'] for x in valid if x['counter_status'] == 0]
    result['submitted_minus_last_observed_id'] = dict(samples=len(id_spans),
        minimum=min(id_spans) if id_spans else None, maximum=max(id_spans) if id_spans else None,
        mean=sum(id_spans)/len(id_spans) if id_spans else None,
        units='present_ID_span', actual_queue_length_known=False)
    result['readiness'] = {name: observations[-1][name] - observations[0][name] if len(observations) > 1 else 0
        for name in ('admissions', 'denials', 'waits', 'message_wakes', 'timeouts')}
    clock = groups.get('ZOOM_CLOCK', [{}])[0]
    if clock:
        frequency, origin = clock['frequency'], clock['start_qpc']
        all_observations = observations + groups.get('ZOOM_DELIVERY_TAIL', [])
        # PresentCount is the ID of the most recently reported delivered image,
        # not a count of all delivered images. Never label its delta as FPS.
        known = {}
        for x in all_observations:
            if x['statistics_status'] != 0 or x['sync_qpc'] < origin or x['sync_qpc'] > x['observation_qpc']:
                continue
            if x['present_refresh'] != x['sync_refresh']:
                continue  # No assumed refresh-rate timestamp extrapolation.
            known[x['present_refresh']] = x
        deliveries = sorted(known.values(), key=lambda x: x['sync_qpc'])
        matched = {x['last_present']: f for x, f in zip(observations, frames) if x['counter_status'] == 0}
        timeline = []
        for x in deliveries:
            f = matched.get(x['present_count'])
            timeline.append(dict(time_ms=(x['sync_qpc']-origin)*1000/frequency,
                refresh=x['present_refresh'], present_id=x['present_count'], matched=bool(f),
                frame=f if f else None))
        delivered_frames = [x for x in timeline if x['matched']]
        steps = [deliveries[n]['present_refresh']-deliveries[n-1]['present_refresh'] for n in range(1,len(deliveries))]
        id_steps = [deliveries[n]['present_count']-deliveries[n-1]['present_count'] for n in range(1,len(deliveries))]
        result['guest_delivery'] = dict(observations=len(deliveries), matched_images=len(delivered_frames),
            unobserved_ids_remain_unknown=True,
            intervals=dist([timeline[n]['time_ms']-timeline[n-1]['time_ms'] for n in range(1,len(timeline))]),
            refresh_steps=steps, present_id_steps=id_steps, multi_refresh_observation_gaps=sum(x>1 for x in steps),
            unobserved_refresh_opportunities=sum(max(0,x-1) for x in steps),
            minimum_without_new_image=sum(max(0,r-max(0,i)) for r,i in zip(steps,id_steps)),
            gaps_with_unobserved_intermediate_images=sum(r>1 and i>1 for r,i in zip(steps,id_steps)),
            first_time_ms=timeline[0]['time_ms'] if timeline else None,
            last_time_ms=timeline[-1]['time_ms'] if timeline else None,
            tail=groups.get('ZOOM_DELIVERY_TAIL', []), timeline=timeline)
        events = summary['records'].get('ZOOM_INPUT', [])
        if events:
            first = events[0];last = events[-1]
            changed = next((x for x in delivered_frames if x['time_ms']>=first['time_ms'] and abs(x['frame']['shown']-1)>1e-8),None)
            quality = next((x for x in delivered_frames if x['time_ms']>=last['time_ms'] and x['frame']['quality'] and
                x['frame']['source_input']>=last['serial'] and abs(x['frame']['shown']-last['zoom'])<1e-6),None)
            result['guest_delivery']['first_input_to_changed_ms'] = changed['time_ms']-first['time_ms'] if changed else None
            result['guest_delivery']['final_quality_after_input_ms'] = quality['time_ms']-last['time_ms'] if quality else None
        result['refinement_phases'] = {name: dist([x[name] for x in groups.get('ZOOM_REFINEMENT',[])]) for name in
            ('prepare_ms','reflection_ms','static_ms','water_ms','units_ms','reconstruct_ms','snapshot_ms')}
    (directory / 'presentation-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(label=label, intervals=result['intervals'], phases=result['phases'],
        readiness=result['readiness'], statistics_statuses=result['statistics_statuses'],
        last_observed_ID_delta=result['statistics_count_delta'], submitted_ID_delta=result['submission_count_delta']), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'build', 'run', 'analyze'])
    p.add_argument('label', nargs='?')
    p.add_argument('--presentation', choices=['vsync', 'waitable'], default='vsync')
    p.add_argument('--mode', choices=['preview', 'refine'], default='refine')
    p.add_argument('--gesture', choices=['ordinary', 'sustained', 'resume'], default='ordinary')
    p.add_argument('--hour', type=int, default=12)
    p.add_argument('--busy', action='store_true')
    p.add_argument('--frozen', action='store_true')
    p.add_argument('--counts', action='store_true')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare()
    elif a.action == 'build':
        build(a.label)
    elif a.action == 'analyze':
        analyze(a.label)
    else:
        run(a.label, a.presentation, a.mode, a.gesture, a.hour, a.busy, a.frozen, a.counts)
