"""Where Civ III's thread spends time: a sampled profile, per near segment.

Usage: game_samples_report.py CAPTURE_DIR [--top N]

CAPTURE_DIR holds renderer-core.log.samples, written by a scripted run with
-ProfileRenderer and -RendererOptions "C3X_RENDERER_SAMPLE_GAME=1"
(Renderer/native/game_thread_sampler.h). Each sample has the QPC time, the
instruction pointer, the nearest call site in another module with the import
slot it calls through, and up to eight return addresses inside
Civ3Conquests.exe found on the stack (validated call sites; a stale one can
appear, so inclusive counts are upper bounds). C3X's injected code is mapped
inside the executable image above Civ III's own code and is shown as c3x+offset.

Function starts come from ref/Civ3Conquests-Unmodded.exe.c (FUN_xxxxxxxx) and
names from civ_prog_objects.csv where it defines one. An unnamed function keeps
its FUN_ address for lookup in the decompiled sources.

Per segment the report prints:
  self       the sampled instruction: an executable function, or the module
             (jgl.dll, C3XRenderer.dll, system DLLs) or `injected` (code
             outside any module: C3X's compiled injected code)
  callers    for samples outside the executable, the nearest Civ III function
             on the stack: which game code is calling that module or patch
  api        for samples outside the executable, the imported function the
             nearest calling module invoked (e.g. jgl.dll calling gdi32!...)
  inclusive  every executable function on the stack, counted once per sample
"""
import bisect
import collections
import json
import pathlib
import re
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def functions():
    starts = set()
    text = (ROOT / 'ref/Civ3Conquests-Unmodded.exe.c').read_text(errors='replace')
    for match in re.finditer(r'(?m)^(?:[A-Za-z_][\w *]*\s)?(?:\w+::)?FUN_([0-9a-f]{8})\(', text):
        starts.add(int(match[1], 16))
    names = {}
    for line in (ROOT / 'civ_prog_objects.csv').read_text(errors='replace').splitlines():
        cells = [c.strip().strip('"') for c in line.split(',')]
        if len(cells) >= 6 and cells[0] == 'define' and '(' in ','.join(cells[5:]):
            try:
                address = int(cells[1], 16)
            except ValueError:
                continue
            names[address] = cells[4]
            starts.add(address)
    ordered = sorted(starts)
    return ordered, names


def read(path):
    data = pathlib.Path(path).read_bytes()
    at, samples, modules, imports = 0, [], {}, {}
    while at + 4 <= len(data):
        (kind,) = struct.unpack_from('<I', data, at)
        if kind == 0:
            if at + 48 > len(data):
                break
            values = struct.unpack_from('<IQ9I', data, at)
            samples.append((values[1], values[2], [v for v in values[3:] if v], 0, 0))
            at += 48
        elif kind == 3:
            if at + 56 > len(data):
                break
            values = struct.unpack_from('<IQ11I', data, at)
            samples.append((values[1], values[2], [v for v in values[5:] if v], values[3], values[4]))
            at += 56
        elif kind == 2:
            if at + 72 > len(data):
                break
            (slot,) = struct.unpack_from('<I', data, at + 4)
            imports[slot] = data[at + 8:at + 72].split(b'\0', 1)[0].decode(errors='replace')
            at += 72
        elif kind == 1:
            if at + 76 > len(data):
                break
            base, size = struct.unpack_from('<II', data, at + 4)
            name = data[at + 12:at + 76].split(b'\0', 1)[0].decode(errors='replace')
            modules[base] = (base + size, name)
            at += 76
        else:
            raise SystemExit(f'unknown record kind {kind} at byte {at}')
    return samples, modules, imports


def segments(capture):
    path = capture / 'mouse-events.json'
    if not path.exists():
        return [('all', 0, 1 << 62)]
    raw = json.loads(path.read_text())
    q = [e['qpc'] for e in (raw['events'] if isinstance(raw, dict) else raw)]
    if len(q) < 25:
        return [('all', 0, 1 << 62)]
    return [('1x idle', q[0] - (q[1] - q[0]), q[0]), ('1x scroll', q[0], q[3]), ('2x scroll', q[13], q[14]),
            ('3x scroll', q[16], q[17]), ('3x idle', q[17], q[18]), ('1x idle end', q[19], q[24])]


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    capture = pathlib.Path(argv[1])
    top = int(argv[argv.index('--top') + 1]) if '--top' in argv else 15
    path = capture / 'renderer-core.log.samples' if capture.is_dir() else capture
    samples, modules, imports = read(path)
    starts, names = functions()
    code_end = starts[-1] + 0x10000  # C3X sections follow Civ III's code
    exe = next(((b, e) for b, (e, n) in modules.items() if n.lower().startswith('civ3conquests')), None)
    module_ranges = sorted((b, e, n) for b, (e, n) in modules.items())
    module_starts = [m[0] for m in module_ranges]

    def function(address):
        if address >= code_end:
            return f'c3x+{address - code_end:x}'
        i = bisect.bisect_right(starts, address) - 1
        if i < 0:
            return f'exe+{address:08x}'
        start = starts[i]
        return names.get(start, f'FUN_{start:08x}')

    def place(address):
        if exe and exe[0] <= address < exe[1]:
            return function(address), True
        i = bisect.bisect_right(module_starts, address) - 1
        if i >= 0 and address < module_ranges[i][1]:
            return module_ranges[i][2], False
        return 'injected', False

    for name, a, b in (segments(capture) if capture.is_dir() else [('all', 0, 1 << 62)]):
        chosen = [s for s in samples if a <= s[0] <= b]
        if not chosen:
            continue
        self_counts, callers, inclusive = collections.Counter(), collections.Counter(), collections.Counter()
        apis = collections.Counter()
        for _, eip, returns, caller, slot in chosen:
            leaf, in_exe = place(eip)
            self_counts[leaf] += 1
            stack = [function(r) for r in returns]
            if not in_exe:
                callers[f'{leaf} <- {stack[0] if stack else "?"}'] += 1
                if caller:
                    apis[f'{leaf} via {imports.get(slot, "direct call")} from {place(caller)[0]}'] += 1
            for f in set(([leaf] if in_exe else []) + stack):
                inclusive[f] += 1
        n = len(chosen)
        print(f'== {name}: {n} samples')
        for title, counter in (('self', self_counts), ('callers', callers), ('api', apis), ('inclusive', inclusive)):
            print(f'  {title}:')
            for key, count in counter.most_common(top):
                print(f'    {100 * count / n:5.1f}%  {key}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
