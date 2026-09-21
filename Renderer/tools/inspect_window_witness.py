"""Review sampled window evidence and optionally correlate a verified input journal.

Correlation identifies the most recent completed presentation candidate. It does
not prove that candidate was displayed or recover uncaptured compositor frames.
"""
import argparse
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


def load(path, limit=64 * 1024 * 1024):
    if path.stat().st_size > limit:
        raise ValueError('Evidence file exceeds review bound')
    return path.read_text(encoding='utf-8')


def rows(path):
    return [json.loads(line) for line in load(path).splitlines() if line.strip()]


def window_rows(path, interrupted):
    lines = load(path).splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if interrupted and index == len(lines) - 1 and not line.endswith('\n'):
                return result, True
            raise ValueError('Corrupt window timeline before a verified prefix boundary') from None
    return result, False


def positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError('Expected positive integer timestamp or identity')
    return value


def correlate(frames, started, inspection):
    report = json.loads(load(inspection / 'report.json'))
    if not report['verified_prefix'] or report['qpc_origin'] <= 0:
        raise ValueError('Correlation requires a verified input prefix and clock origin')
    frequency = positive(report['frequency'])
    if frequency != started['qpc_frequency']:
        raise ValueError('Window and input clocks use different frequencies')
    # UTC is an independent sanity check against selecting another machine/boot.
    # Use integer subtraction before division: FILETIME exceeds JS exact integers.
    predicted_delta = (started['started_qpc'] - report['qpc_origin']) / frequency
    utc_delta = (started['utc_filetime_100ns'] - report['utc_filetime_100ns']) / 10_000_000
    bracket = ((started['qpc_after_utc'] - started['started_qpc']) + report['qpc_utc_bracket_ticks']) / frequency
    if abs(predicted_delta - utc_delta) > .1 + bracket:
        raise ValueError('Input and window clock correlations disagree; no alignment assumed')
    first = report['qpc_origin'] / frequency
    last = first + report['duration_seconds']
    if not any(first <= row['compositor_100ns'] / 10_000_000 <= last for row in frames):
        raise ValueError('Window and input recording intervals do not overlap')
    presents = [row for row in rows(inspection / 'timeline.jsonl') if row['frame']]
    presents.sort(key=lambda row: (row['result_ticks'], row['sequence']))
    times = [row['result_ticks'] for row in presents]
    for frame in frames:
        ticks = frame['compositor_100ns'] * frequency // 10_000_000 - report['qpc_origin']
        frame['input_seconds'] = ticks / frequency
        index = bisect_right(times, ticks) - 1
        if index >= 0 and ticks <= report['duration_seconds'] * frequency:
            candidate = presents[index]
            frame['preceding_presentation_candidate'] = candidate['frame']
            frame['candidate_call'] = candidate['call']
            frame['candidate_age_ms'] = (ticks - candidate['result_ticks']) * 1000 / frequency
    return {'input_complete': report['complete'], 'input_frames': report['accepted_presentations'],
            'utc_correlation_error_ms': (predicted_delta - utc_delta) * 1000,
            'scope': 'preceding completed presentation candidate, not proof of display'}


def inspect(witness, inspection=None):
    started = json.loads(load(witness / 'started.json'))
    frequency = positive(started['qpc_frequency'])
    origin = positive(started['started_qpc'])
    finished = json.loads(load(witness / 'finished.json')) if (witness / 'finished.json').exists() else None
    timeline, truncated_tail = window_rows(witness / 'timeline.jsonl', finished is None)
    if len(timeline) > 20000:
        raise ValueError('Window timeline exceeds bounded duration')
    frames = [dict(row) for row in timeline if 'frame' in row]
    memory = [dict(row) for row in timeline if row.get('event') == 'process_memory']
    if not frames:
        raise ValueError('No window images to review')
    last_source = 0
    total = 0
    for n, row in enumerate(frames, 1):
        if row['frame'] != n or positive(row['compositor_100ns']) < last_source:
            raise ValueError('Window frame identity/time is not ordered')
        if row['previous_compositor_100ns'] != last_source:
            raise ValueError('Broken window frame predecessor')
        last_source = row['compositor_100ns']
        if not 0 < row['width'] <= 2400 or not 0 < row['height'] <= 1400:
            raise ValueError('Unsupported window extent')
        if row['saved_qpc'] < row['arrival_qpc']:
            raise ValueError('Negative window capture span')
        image = witness / f'window-{n:06}.jpg'
        if image.stat().st_size != row['bytes']:
            raise ValueError('Missing or changed window image length')
        total += row['bytes']
        row['seconds'] = row['compositor_100ns'] / 10_000_000 - origin / frequency
        row['capture_ms'] = (row['saved_qpc'] - row['arrival_qpc']) * 1000 / frequency
    if finished and (finished['frames'] != len(frames) or finished['bytes'] != total):
        raise ValueError('Window completion receipt disagrees with timeline')
    if total > 2 * 1024 * 1024 * 1024:
        raise ValueError('Window evidence exceeds storage bound')
    for row in memory:
        row['seconds'] = (row['qpc'] - origin) / frequency
        if row['free_bytes'] < row['largest_free_bytes']:
            raise ValueError('Invalid address-space sample')
        if 'address_limit' in row and row['enumerated_through'] > row['address_limit']:
            raise ValueError('Memory sample exceeds target address space')
    summary = {'complete': bool(finished and finished['complete']), 'qualified_for_gameplay': False,
               'scope': 'lossy sampled window evidence; uncaptured frames unknown; not scanout or FPS',
               'frames': len(frames), 'image_bytes': total, 'memory_samples': len(memory),
               'truncated_final_timeline_record': truncated_tail,
               'events': {kind: sum(row.get('event') == kind for row in timeline)
                          for kind in ('no_frame', 'resize_gap')},
               'dequeued_unsaved_frames': finished['dequeued_unsaved_frames'] if finished else None,
               'min_private_bytes': min((row['private_bytes'] for row in memory), default=None),
               'max_private_bytes': max((row['private_bytes'] for row in memory), default=None)}
    resources = [row for row in memory if 'observer_cpu_100ns' in row]
    if len(resources) >= 2:
        a, b = resources[0], resources[-1]
        if b['observer_cpu_100ns'] < a['observer_cpu_100ns'] or b['qpc'] <= a['qpc']:
            raise ValueError('Observer resource counters moved backward')
        wall = (b['qpc'] - a['qpc']) / frequency
        cpu = (b['observer_cpu_100ns'] - a['observer_cpu_100ns']) / 10_000_000
        summary['observer_resources'] = {
            'sample_interval_seconds': wall, 'cpu_seconds': cpu, 'mean_cpu_cores': cpu / wall,
            'peak_private_bytes': max(row['observer_private_bytes'] for row in resources),
            'max_memory_scan_ms': max((row['qpc'] - row['begin_qpc']) * 1000 / frequency for row in resources),
            'scope': 'observer process CPU/memory; excludes GPU contention and is not frame overhead calibration'}
    if inspection:
        summary['alignment'] = correlate(frames, started, inspection)
    return summary, frames, memory


def render_review(witness, output, summary, frames, memory):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    for row in frames:
        path = witness / f'window-{row["frame"]:06}.jpg'
        row['image'] = quote(os.path.relpath(path, output).replace(os.sep, '/'), safe='/')
    data = json.dumps({'summary': summary, 'frames': frames, 'memory': memory}).replace('<', '\\u003c')
    page = '''<!doctype html><meta charset="utf-8"><title>Renderer window evidence</title>
<style>body{font:16px system-ui;background:#14181d;color:#e4ebf3;margin:24px auto;max-width:1200px}h1{font-size:24px}p{line-height:1.5}img{max-width:100%;max-height:72vh;background:#000}input[type=range]{width:100%}pre{white-space:pre-wrap;font:14px monospace;color:#bfd1e3}.controls{display:flex;gap:12px;align-items:center}button,input{font:inherit;padding:6px}#label{flex:1}</style>
<h1>Recorded window evidence</h1><p>Lossy samples of one Windows window. Missing compositor frames are unknown. A preceding renderer presentation is a candidate, not proof that Windows displayed it. Capture time below measures the helper's work, not game frame time.</p>
<div class="controls"><button id="prev">Previous</button><button id="next">Next</button><span id="label"></span><label>Window second <input id="second" type="number" step=".1" style="width:100px"></label></div>
<input id="position" type="range" min="0"><img id="picture" alt="Captured game or test window"><pre id="details"></pre><details><summary>Coverage and completeness</summary><pre id="coverage"></pre></details>
<script id="evidence" type="application/json">DATA</script><script>
const data=JSON.parse(document.getElementById('evidence').textContent), frames=data.frames;
const position=document.getElementById('position'),second=document.getElementById('second');position.max=frames.length-1;
function show(index){index=Math.max(0,Math.min(frames.length-1,index));const f=frames[index];position.value=index;second.value=f.seconds.toFixed(3);
document.getElementById('picture').src=f.image;document.getElementById('label').textContent=`Sample ${f.frame} of ${frames.length} · ${f.width}×${f.height}`;
const memory=data.memory.filter(m=>m.seconds<=f.seconds).at(-1);document.getElementById('details').textContent=JSON.stringify({sample:f,memory_before_sample:memory||null},null,2);}
position.oninput=()=>show(Number(position.value));document.getElementById('prev').onclick=()=>show(Number(position.value)-1);document.getElementById('next').onclick=()=>show(Number(position.value)+1);
second.onchange=()=>{const t=Number(second.value);let at=0;for(let i=0;i<frames.length;i++){if(frames[i].seconds<=t)at=i;else break;}show(at);};
document.getElementById('coverage').textContent=JSON.stringify(data.summary,null,2);show(0);</script>'''
    (output / 'review.html').write_text(page.replace('DATA', data), encoding='utf-8')
    # A small independent contact sheet makes motion, native UI and resizing
    # inspectable without downloading or decoding a whole movie.
    from PIL import Image, ImageDraw
    count = min(12, len(frames))
    selected = sorted({round(n * (len(frames) - 1) / max(1, count - 1)) for n in range(count)})
    sheet = Image.new('RGB', (4 * 320, math.ceil(len(selected) / 4) * 220), '#202832')
    draw = ImageDraw.Draw(sheet)
    for n, index in enumerate(selected):
        frame = frames[index]
        with Image.open(witness / f'window-{frame["frame"]:06}.jpg') as image:
            if image.size != (frame['width'], frame['height']):
                raise ValueError('Window image extent differs from timeline')
            image.thumbnail((312, 192));x, y = n % 4 * 320 + 4, n // 4 * 220 + 4
            sheet.paste(image, (x, y));draw.text((x, y + 195), f'Sample {frame["frame"]}  {frame["seconds"]:.3f}s', fill='white')
    sheet.save(output / 'contact-sheet.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--witness', required=True, type=Path)
    parser.add_argument('--inputs', type=Path, help='Verified inspect_inputs output directory')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    summary, frames, memory = inspect(args.witness, args.inputs)
    render_review(args.witness.resolve(), args.out.resolve(), summary, frames, memory)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
