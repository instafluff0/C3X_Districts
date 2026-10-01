"""Private C7 retained-image preview scheduling gate; no staging or promotion."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from Renderer.tools import measure_redraw_navigation as navigation
from Renderer.tools.measure_redraw_submission import archive_outputs
from Renderer.lab.platform import windows_root, native_command_result

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'Renderer/.cache/redraw-underlay-step'
OUT = ROOT / 'Renderer/.cache/zoom-preview-step'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if OUT.exists():
        raise ValueError('Preserve existing zoom-preview evidence')
    OUT.mkdir()
    baseline = json.loads((ROOT / 'Renderer/.cache/redraw-submission-step/compiled-input-verification.json').read_text())['runtime']
    source = OUT / 'sources'
    for relative, expected in baseline.items():
        original = ROOT / relative
        if sha(original) != expected:
            raise ValueError('C7 source changed: ' + relative)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    extra = ['Renderer/native/c3x_renderer.def']
    for name in ('pass_counts.h', 'capture_model.h', 'reference_x64.cpp', 'client_x64.cpp', 'camera_witness.h'):
        relative = Path('Renderer/sandbox') / name
        target = source / relative
        frozen = ROOT / 'Renderer/.cache/redraw-submission-step/client-source' / relative
        shutil.copy2(frozen, target)
    for original in (ROOT / 'Renderer/native').glob('*.h'):
        if not (source / original.relative_to(ROOT)).exists():
            extra.append(original.relative_to(ROOT).as_posix())
    extra += ['Renderer/sandbox/' + name for name in (
        'zoom_preview_gpu.h', 'zoom_preview_state.h', 'zoom_preview_witness.h', 'zoom_preview_busy_units.h')]
    for relative in extra:
        shutil.copy2(ROOT / relative, source / relative)
    before = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}

    path = source / 'Renderer/sandbox/resident_scene.cpp'
    text = path.read_text()

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Preview insertion changed: ' + old[:70])
        text = text.replace(old, new)

    replace('static SandboxBackbufferOutput sandbox_backbuffer_output;', '''static SandboxBackbufferOutput sandbox_backbuffer_output;
#include "zoom_preview_gpu.h"
#include "zoom_preview_busy_units.h"
static c3x_renderer::sandbox::CompletedMapPreview sandbox_zoom_preview;
static bool sandbox_zoom_selected=false;
static std::uint64_t sandbox_zoom_ids[2]{};
static Microsoft::WRL::ComPtr<ID3D11Query> sandbox_zoom_completion;
// GPU serial is the fixed fixture visibility/view scope. Unique completed
// output IDs are validated separately, so a close output retains the wide one.
extern "C" __declspec(dllexport) int c3x_sandbox_zoom_commit(unsigned slot,float zoom,std::uint64_t id){
    auto width=unsigned(renderer.content_view_width),height=unsigned(renderer.content_view_height);
    if(slot>=2 || !id || !sandbox_zoom_preview.prepare(renderer.device,slot,width,height))return 1;
    float previous=sandbox_fresh.display_zoom;sandbox_fresh.display_zoom=1.f;
    bool ok=sandbox_backbuffer_output.draw(sandbox_zoom_preview.target(slot),int(width),int(height));
    sandbox_fresh.display_zoom=previous;
    if(!ok || !sandbox_zoom_preview.commit(renderer.context,slot,sandbox_fresh.glow.linear.depth_texture,zoom,zoom,1))return 2;
    if(!sandbox_zoom_completion){D3D11_QUERY_DESC d{};d.Query=D3D11_QUERY_EVENT;
        if(FAILED(renderer.device->CreateQuery(&d,&sandbox_zoom_completion)))return 3;}
    renderer.context->End(sandbox_zoom_completion.Get());
    sandbox_zoom_ids[slot]=id;sandbox_zoom_selected=false;return 0;
}
extern "C" __declspec(dllexport) int c3x_sandbox_zoom_ready(){
    if(!sandbox_zoom_completion)return -1;BOOL done=FALSE;
    HRESULT h=renderer.context->GetData(sandbox_zoom_completion.Get(),&done,sizeof(done),D3D11_ASYNC_GETDATA_DONOTFLUSH);
    return h==S_FALSE?1:FAILED(h)?-1:done?0:1;
}
extern "C" __declspec(dllexport) int c3x_sandbox_zoom_select(unsigned slot,float zoom,std::uint64_t id){
    if(slot>=2 || sandbox_zoom_ids[slot]!=id || !sandbox_zoom_preview.select(slot,zoom,1))return 1;
    sandbox_zoom_selected=true;return 0;
}
extern "C" __declspec(dllexport) std::uint64_t c3x_sandbox_zoom_bytes(){return sandbox_zoom_preview.bytes();}
extern "C" __declspec(dllexport) int c3x_sandbox_zoom_busy(c3x_renderer_frame_v1 const* frame){
    return !frame || !sandbox_zoom_preview_busy_update(*frame);
}
extern "C" __declspec(dllexport) void c3x_sandbox_zoom_busy_metrics(unsigned* values){
    if(!values)return;auto m=sandbox_zoom_preview_busy_metrics(false);
    values[0]=m.visible;values[1]=m.travelling;values[2]=m.parts;
}''')
    replace('if(zoom<=1.00001){', 'if(abs(zoom-1.f)<=.00001){')
    replace('!sandbox_backbuffer_output.draw(target.Get(),d.Width,d.Height)', '''!(sandbox_zoom_selected?
            sandbox_zoom_preview.draw(renderer.device,renderer.context,target.Get(),d.Width,d.Height):
            sandbox_backbuffer_output.draw(target.Get(),d.Width,d.Height))''')
    replace('bool composed=sandbox_backbuffer_output.draw(target,width,height);', '''bool composed=sandbox_zoom_selected?
            sandbox_zoom_preview.draw(renderer.device,renderer.context,target,unsigned(width),unsigned(height)):
            sandbox_backbuffer_output.draw(target,width,height);''')
    path.write_text(text)
    path = source / 'Renderer/sandbox/client_x64.cpp'
    text = path.read_text()
    text = text.replace('#include "camera_witness.h"', '#include "zoom_preview_witness.h"\n#include "camera_witness.h"')
    path.write_text(text)
    path = source / 'Renderer/sandbox/camera_witness.h'
    text = path.read_text()
    text = text.replace('    if(zoom_sequence)views={{0,0,1,"capture_zoom"}};', '''    if(zoom_sequence)views={{0,0,1,"capture_zoom"}};
    char preview_mode[32]{};GetEnvironmentVariableA("C3X_ZOOM_PREVIEW_MODE",preview_mode,sizeof(preview_mode));
    if(preview_mode[0])views={{0,0,1,"preview_origin"}};''')
    old = '        char receipts[4*MAX_PATH]={};'
    if text.count(old) != 1:
        raise ValueError('Camera adoption boundary changed')
    text = text.replace(old, '''        if(preview_mode[0]){
            int preview_result=sandbox_zoom_preview_witness(module,window,frame);
            if(!preview_result)std::printf("CAMERA_WITNESS pass preview=1\\n");
            return preview_result;
        }
''' + old)
    old = '        if(captures[0]){\n            if(!capture)return 29;'
    oracle = '''        char oracle_text[8]{};
        if(GetEnvironmentVariableA("C3X_ZOOM_PREVIEW_ORACLE",oracle_text,sizeof(oracle_text))&&oracle_text[0]=='1'){
            auto commit=reinterpret_cast<int(*)(unsigned,float,std::uint64_t)>(GetProcAddress(module,"c3x_sandbox_zoom_commit"));
            auto select=reinterpret_cast<int(*)(unsigned,float,std::uint64_t)>(GetProcAddress(module,"c3x_sandbox_zoom_select"));
            auto oracle_ready=reinterpret_cast<int(*)()>(GetProcAddress(module,"c3x_sandbox_zoom_ready"));
            if(!commit||!select||!oracle_ready||commit(0,view.zoom,99)||select(0,view.zoom,99)||
               present(window,&frame,24,56,1,47,1,view.x,view.y))return 54;
            auto oracle_deadline=GetTickCount64()+10000;int status=oracle_ready();
            while(status==1&&GetTickCount64()<oracle_deadline){Sleep(1);status=oracle_ready();}
            if(status)return 55;
            std::printf("ZOOM_ORACLE source=original_full_projection relative=1 gpu_ready=1 zoom=%.9f ticks=%lld\\n",view.zoom,frame.presentation_time_ticks);
        }
'''
    if text.count(old) != 1:
        raise ValueError('Oracle capture boundary changed')
    text = text.replace(old, oracle + old)
    path.write_text(text)

    for arm, template in (('candidate', 'candidate'), ('client', 'client')):
        (OUT / arm / 'obj').mkdir(parents=True)
        batch = (ROOT / 'Renderer/.cache/redraw-submission-step' / template / 'build.bat').read_text()
        batch = batch.replace('..\\.cache\\redraw-submission-step\\' + template, '..\\..\\..\\' + arm)
        (OUT / arm / 'build.bat').write_text(batch)
    (OUT / 'control').mkdir()
    os.link(BASE / 'control/C3XRenderer_x64.dll', OUT / 'control/C3XRenderer_x64.dll')
    path = OUT / 'accepted/shaders'
    path.parent.mkdir()
    path.symlink_to(BASE / 'accepted/shaders', target_is_directory=True)
    shutil.copy2(BASE / 'keep-awake.ps1', OUT / 'keep-awake.ps1')
    after = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    (OUT / 'source-manifest.json').write_text(json.dumps(dict(c7_runtime=baseline, before=before, after=after,
        changed=[name for name in before if before[name] != after[name]], extras=extra), indent=2) + '\n')


def build(arm):
    batch = windows_root() / OUT.relative_to(ROOT).as_posix() / arm / 'build.bat'
    sources = 'qa-sources' if arm == 'qa-client' else 'sources'
    result = native_command_result(str((OUT / sources / 'Renderer/native').relative_to(ROOT)),
        'call "' + str(batch) + '"', timeout_seconds=600)
    (OUT / arm / 'build-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['output_tail'])
    if result['status'] != 'pass':
        raise ValueError('Build failed: ' + arm)


def run(label, mode='refine', gesture='ordinary', hour=12, busy=False, visual=False, counts=False, controls=None):
    navigation.OUT = OUT
    directory = windows_root() / OUT.relative_to(ROOT).as_posix() / label
    options = dict(C3X_ZOOM_PREVIEW_MODE=mode, C3X_ZOOM_PREVIEW_GESTURE=gesture,
        C3X_ZOOM_PREVIEW_SEED_CAPTURE=str(directory / 'seed'),
        C3X_ZOOM_PREVIEW_BUSY='1' if busy else '0',
        C3X_ZOOM_PREVIEW_VISUAL=str(directory / 'visual') if visual else '',
        C3X_RENDERER_PATCH_PIXELS='0', C3X_SANDBOX_PASS_COUNTS='1' if counts else '0',
        C3X_RENDERER_TRACE='0', C3X_RENDERER_PROFILE='0', C3X_SANDBOX_PROFILE_COMPLETION='0',
        C3X_RENDERER_OUTPUT_COMPLETION_PROBE='0', C3X_SANDBOX_GPU_TIMESTAMPS='0',
        C3X_SANDBOX_FRAME_TIMINGS='0', C3X_SANDBOX_WITNESS_SCENARIO='zoom',
        C3X_SANDBOX_ZOOM_CHECKPOINTS='0', C3X_SANDBOX_CAUSAL_MODE='0', C3X_SANDBOX_CAUSAL_LEDGER='0',
        C3X_SANDBOX_WORLD_CONTENT_DIR='', C3X_SANDBOX_UNDERLAY_PROBE='0',
        C3X_RENDERER_SHADER_SOURCE_ROOT=str(windows_root() / BASE.relative_to(ROOT).as_posix() / 'accepted/shaders'))
    options.update(controls or {})
    arm = 'control' if mode == 'control' and not busy else 'candidate'
    result = navigation.run(label, arm, 'navigation', hour, 1, options, False, 'full_guest', 'client')
    archive_outputs(OUT / label)
    analyze(label)
    return result


def analyze(label):
    directory = OUT / label
    text = (directory / 'run.log').read_text(errors='replace')
    groups = {}
    for line in text.splitlines():
        if not line.startswith(('ZOOM_FRAME ', 'ZOOM_INPUT ', 'ZOOM_REDRAW ', 'ZOOM_WORKLOAD ', 'ZOOM_UNIT_SUBMISSION ', 'ZOOM_BUSY_FINAL ', 'ZOOM_BUSY ')):
            continue
        name, rest = line.split(' ', 1)
        fields = dict(re.findall(r'(\w+)=(\S+)', rest))
        for key, value in fields.items():
            try:
                fields[key] = float(value) if '.' in value else int(value)
            except ValueError:
                pass
        groups.setdefault(name, []).append(fields)
    frames = groups.get('ZOOM_FRAME', [])
    inputs = groups.get('ZOOM_INPUT', [])
    jobs = groups.get('ZOOM_REDRAW', [])
    if not frames:
        raise ValueError('Missing executed zoom preview frames')
    intervals = [frames[n]['done_ms'] - frames[n-1]['done_ms'] for n in range(1, len(frames))]
    def dist(values):
        result = navigation.distribution(values)
        result['over_16_67ms_budget'] = result.pop('missed_16_67ms', 0)
        result['over_25ms'] = sum(v > 25 for v in values)
        result['over_100ms'] = sum(v > 100 for v in values)
        return result
    latency = []
    changed_frames = [f for n, f in enumerate(frames) if n and abs(f['shown'] - frames[n-1]['shown']) > 1e-8]
    changed_latency = []
    for event in inputs:
        changed = next((f for f in frames if f['input'] >= event['serial'] and f['done_ms'] >= event['time_ms']), None)
        if changed:
            latency.append(changed['done_ms'] - event['time_ms'])
        changed = next((f for f in changed_frames if f['input'] >= event['serial'] and f['done_ms'] >= event['time_ms']), None)
        if changed:
            changed_latency.append(changed['done_ms'] - event['time_ms'])
    final = inputs[-1] if inputs else dict(time_ms=0, serial=0, zoom=1)
    first_input = inputs[0]['time_ms'] if inputs else 0
    first_changed = next((f for f in changed_frames if f['done_ms'] >= first_input), None)
    quality = next((f for f in frames if f['done_ms'] >= final['time_ms'] and f['quality'] and
        f.get('source_input', f['input']) >= final['serial'] and
        abs(f['shown'] - final['zoom']) < 1e-6), None)
    previous = sum(f['done_ms'] < final['time_ms'] for f in frames)
    result = dict(records=groups, present_intervals=dist(intervals),
        interaction_intervals=dist([intervals[n-1] for n in range(1, len(frames))
            if frames[n]['done_ms'] >= first_input and frames[n-1]['done_ms'] <= final['time_ms']]),
        preview_only_intervals=dist([intervals[n-1] for n in range(1, len(frames)) if not frames[n]['redraw']]),
        redraw_intervals=dist([intervals[n-1] for n in range(1, len(frames)) if frames[n]['redraw']]),
        input_to_coalesced_present_return=dist(latency),
        input_to_changed_present_return=dist(changed_latency), changed_scale_presents=len(changed_frames),
        first_input_to_changed_present_ms=first_changed['done_ms'] - first_input if first_changed else None,
        draw_submission=dist([f['draw_cpu_ms'] for f in frames if f['redraw']]),
        full_redraw_through_present=dist([j['presented_ms'] - j['start_ms'] for j in jobs]),
        refinement_submission_to_gpu_ready_observation=dist([j['gpu_observed_ms'] - j['submitted_ms'] for j in jobs if j.get('gpu_observed_ms', -1) >= 0]),
        full_refinement_to_gpu_ready_observation=dist([j['gpu_observed_ms'] - j['start_ms'] for j in jobs if j.get('gpu_observed_ms', -1) >= 0]),
        gpu_completed_refinements=sum(j.get('gpu_observed_ms', -1) >= 0 for j in jobs),
        present_call=dist([f['present_ms'] for f in frames]),
        source_age=dist([f['source_age_ms'] for f in frames]),
        final_input=final, current_quality_after_final_ms=quality['done_ms'] - final['time_ms'] if quality else None,
        displayed_frames_to_final_quality=frames.index(quality) + 1 - previous if quality else None,
        maximum_preview_bytes=max(f['bytes'] for f in frames), full_redraws=len(jobs),
        presentation_evidence='QPC successful DXGI Present returns; scanout not measured. Candidate EVENT completion is observed without blocking, between last-pending and first-ready polls.',
        notes=['Coalesced event latency includes superseded events; input clock is independent of rendered frames.',
               'Preview-only mode freezes dynamics and is an invalid quality target.',
               'Readback visual runs are diagnostic and excluded from quiet timing comparisons.'])
    (directory / 'zoom-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('records', 'notes')}, indent=2))
    return result


def oracle(label, arm, hour=12, endpoint='zoom_settle', client_arm='client'):
    navigation.OUT = OUT
    result = navigation.run(label, arm, 'navigation', hour, 1, dict(
        C3X_ZOOM_PREVIEW_MODE='', C3X_ZOOM_PREVIEW_ORACLE='1' if arm == 'candidate' else '0',
        C3X_SANDBOX_WITNESS_SCENARIO='', C3X_SANDBOX_WITNESS_ENDPOINT=endpoint,
        C3X_SANDBOX_PASS_COUNTS='0', C3X_SANDBOX_GPU_TIMESTAMPS='0',
        C3X_RENDERER_PROFILE='0', C3X_RENDERER_TRACE='0',
        C3X_RENDERER_SHADER_SOURCE_ROOT=str(windows_root() / BASE.relative_to(ROOT).as_posix() / 'accepted/shaders')),
        True, 'full_guest', client_arm)
    archive_outputs(OUT / label)
    return result


def prepare_qa():
    if (OUT / 'qa-sources').exists():
        raise ValueError('Preserve existing QA snapshot')
    shutil.copytree(OUT / 'sources', OUT / 'qa-sources')
    path = OUT / 'qa-sources/Renderer/sandbox/camera_witness.h'
    text = path.read_text()
    old = '            auto commit=reinterpret_cast<int(*)(unsigned,float,std::uint64_t)>'
    if text.count(old) != 1:
        raise ValueError('Original-vs-retained oracle insertion changed')
    text = text.replace(old, '''            char original_prefix[4*MAX_PATH]{};
            sprintf_s(original_prefix,"%s\\\\%s-original",captures,view.name);
            if(!capture||capture(original_prefix))return 56;
''' + old)
    path.write_text(text)
    (OUT / 'qa-client/obj').mkdir(parents=True)
    text = (OUT / 'client/build.bat').read_text().replace('..\\..\\..\\client', '..\\..\\..\\qa-client')
    (OUT / 'qa-client/build.bat').write_text(text)
    files = {p.relative_to(OUT / 'qa-sources').as_posix(): sha(p) for p in (OUT / 'qa-sources').rglob('*') if p.is_file()}
    (OUT / 'qa-sources.json').write_text(json.dumps(files, indent=2) + '\n')


def visual(label):
    from PIL import Image, ImageDraw, ImageFont
    directory = OUT / label
    summary = json.loads((directory / 'zoom-summary.json').read_text())
    frames = summary['records']['ZOOM_FRAME']
    sampled, previous = [], -1000
    for frame in frames:
        submitted = frame['start_ms'] + frame['draw_cpu_ms']
        if submitted - previous >= 100:
            sampled.append(frame)
            previous = submitted
    images = sorted(directory.glob('visual-*.jpg'))
    if len(images) != len(sampled):
        raise ValueError('Visual readback/source timestamp count mismatch')
    try:
        font = ImageFont.truetype('Arial.ttf', 28)
    except OSError:
        font = ImageFont.load_default()
    annotated = directory / 'annotated'
    annotated.mkdir(exist_ok=True)
    receipts, concat = [], []
    for index, (path, frame) in enumerate(zip(images, sampled)):
        image = Image.open(path).convert('RGB')
        ImageDraw.Draw(image).rectangle((0, 0, image.width, 92), fill=(10, 10, 10))
        text = (f"Present return {frame['done_ms']:.1f} ms | shown {frame['shown']:.4f} | requested {frame['requested_done']:.4f}\n"
                f"Source {frame['source']} at {frame['source_time_ms']:.1f} ms | age {frame['source_age_ms']:.1f} ms | full quality {frame['quality']}")
        ImageDraw.Draw(image).text((16, 8), text, fill='white', font=font)
        destination = annotated / f'frame-{index:03d}.png'
        image.save(destination)
        duration = (sampled[index+1]['done_ms'] - frame['done_ms']) / 1000 if index+1 < len(sampled) else .1
        concat += [f"file 'annotated/{destination.name}'", 'option framerate 1000', f'duration {duration:.9f}']
        receipts.append(dict(jpeg=path.name, sha256=sha(path), present_return_ms=frame['done_ms'],
            duration_seconds=duration, shown=frame['shown'], source=frame['source'], source_time_ms=frame['source_time_ms']))
    concat.append(f"file 'annotated/frame-{len(sampled)-1:03d}.png'")
    concat.append('option framerate 1000')
    (directory / 'visual-concat.txt').write_text('\n'.join(concat) + '\n')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('ffmpeg unavailable; annotated frames and timestamps preserved')
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
        '-i', 'visual-concat.txt', '-vsync', 'vfr', '-c:v', 'libx264', '-preset', 'veryfast',
        '-crf', '18', '-pix_fmt', 'yuv420p', 'normal-speed.mp4'], cwd=directory, check=True)
    (directory / 'visual-timestamps.json').write_text(json.dumps(dict(frames=receipts,
        evidence='10 Hz diagnostic readbacks; original QPC dwell times, no timeline compression; Present return is not scanout.'), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'prepare-qa', 'build', 'run', 'analyze', 'oracle', 'visual'))
    p.add_argument('--arm', default='candidate', choices=('candidate', 'client', 'control', 'qa-client'))
    p.add_argument('--endpoint', default='zoom_settle', choices=('origin', 'zoom_settle'))
    p.add_argument('--label')
    p.add_argument('--mode', default='refine', choices=('control', 'preview', 'refine'))
    p.add_argument('--gesture', default='ordinary', choices=('ordinary', 'sustained', 'resume'))
    p.add_argument('--hour', type=int, default=12)
    p.add_argument('--busy', action='store_true')
    p.add_argument('--visual', action='store_true')
    p.add_argument('--counts', action='store_true')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare()
    elif a.action == 'prepare-qa':
        prepare_qa()
    elif a.action == 'build':
        build(a.arm)
    elif a.action == 'analyze':
        analyze(a.label)
    elif a.action == 'oracle':
        oracle(a.label, a.arm, a.hour, a.endpoint)
    elif a.action == 'visual':
        visual(a.label)
    else:
        run(a.label, a.mode, a.gesture, a.hour, a.busy, a.visual, a.counts)
