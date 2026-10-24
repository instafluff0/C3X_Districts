# Music skips and blips with Renderer64

Measured 2026-10-18 in the Parallels ARM VM (8 vCPUs on a 4P+6E Mac, 14 GB),
with the probe in `Renderer/tools/sound_service_probe.cpp`.

## How Civ III plays audio

Civ III's `sound.dll` (TimeDateStamp `0x3f74c977`) does its own DirectSound
mixing. Miles (`Mss32.dll`) only reads files and decodes MP3. Everything,
music included, is mixed in one 30 ms periodic `timeSetEvent` callback.
The callback runs on winmm's timer thread at priority 15 and holds one
critical section (`sound.dll+0xb3a08`) while it mixes.

The output is a 262,144-byte DirectSound ring (1.49 s at 44.1 kHz stereo). The
mixer keeps 8 periods (240 ms) mixed past the write cursor, which is about
270 ms ahead of playback. Its latency setter (`sound.dll+0x11120`, period ms ×
count) stores that lead at buffer `+0x38`. New sounds are appended at the end
of the mixed region. Only construction and reset restart mixing at the write
cursor. So sound-effect latency equals the lead.

If the callback misses about 280 ms, playback runs past the mixed audio and
DirectSound replays ring contents from 1.49 s earlier. That replay is the
audible skip or blip.

## Controlled check

The probe's `--hold-test` held game threads in a disposable session, and a
loopback recording captured the output:

| Held thread | Hold | Audio |
| --- | --- | --- |
| Civ III main thread | 1 s, 3 s | clean |
| sound timer thread | 150 ms | clean |
| sound timer thread | 300 ms | ring replay from ~290 ms (0.30 s) |
| sound timer thread | 600 ms | ring replay from ~300 ms (0.64 s) |

Music does not depend on Civ III's main thread. Renderer64 waits on that
thread (camera jobs, native calls) therefore cannot cause skips, however long
they are. Only the timer thread matters.

## Renderer64 measurements

There were 8 scripted runs (`near` ×2, `zoom-out` ×2, `turn-stress`,
`settler`, `city`, `units`), about 28 minutes in total, busy and early saves:

- Mixer service intervals were 30 ms almost always. The longest was 140 ms,
  well below 280 ms. No run produced a ring replay.
- Civ III's main thread never held the sound lock while the callback waited.
  The guest was 15–60% busy, and the timer thread runs at priority 15. Guest
  CPU load does not starve it.
- Nearly every interval of 60 ms or more was guest-wide: the probe's own
  time-critical thread stopped for the same time. These are VM pauses, not
  Civ III or Renderer64 blocking.
- There were no hard page faults in the guest. Guest available memory was
  3.3 GB or more.
- In every run, Parallels logged `[BALLOON] Memory pressure level changed: 0=>1`
  (`~/Parallels/<VM>.pvm/parallels.log`) 60–80 s after launch. That is when
  the helper's private memory reaches about 4 GB, on top of its GPU memory,
  which the host backs.
- The worst cluster (95 ms and 122 ms) fell in the second of one such
  transition.
- With no game running, putting the Mac under memory pressure (6 GB allocated
  and touched) produced a 92 ms whole-VM pause. The idle baseline is 10–17 ms.

## Memory during a busy-save `near` run

The Mac has 24 GB; the VM has 14 GB. These were sampled every 2 s with
`vm_stat`, `kern.memorystatus_vm_pressure_level` and `sample_memory.ps1`.

| | Before launch | About 80 s in, to the end | After exit |
| --- | --- | --- | --- |
| macOS pressure level | 1 (warn) | 2 (critical) | 1 |
| Parallels resident | 11–12 GB | 3.2–5.5 GB | 11.7 GB |
| Compressor (occupied / stored) | 3.7 / 16 GB | 8–11 / 22–26 GB | 3.3 / 14 GB |
| Swap used | 4.16 GB | 4.7 GB | 4.7 GB |

- The helper grew from 1.5 GB to a 5.7 GB private peak, with 3.4 GB of GPU
  memory (Parallels has no dedicated VRAM, so this is system memory). It then
  settled to about 4.6 + 2.0–2.6 GB. Civ III used 0.3 GB.
- The goal in `performance_goals.md` is about 4 GB process and 2 GB GPU on
  the busy save.
- The Mac mirrors the helper's GPU memory almost 1:1, outside the 14 GB of
  guest RAM. Parallels' `IOAccelerator (graphics)` footprint went from
  0.38 GB idle to 3.24 GB at first view and 3.67 GB at the peak. At the same
  moments the helper's GPU memory was 2.88 and 3.45 GB. Parallels' total
  footprint went from 15.4 to 19.5 GB.
  - Guest RAM is fully dirty and constant, so GPU cuts reduce the Mac's
    footprint directly. Cuts to the helper's other private memory help only
    by making guest RAM easier to compress.
- Windows still had 3.1 GB or more available. So Renderer64's own pressure
  release, which triggers below 1.5–2 GB available in the guest, never ran.
  The guest cannot see that the Mac is short.

## Conclusion

Skips are not caused by the 32-bit process being blocked. They come from
whole-VM pauses of about 280 ms or more. Host memory pressure is the measured
trigger, and Renderer64's memory footprint brings the Mac into that state.
Scripted runs reached about half the audible length. Longer play, more Mac
apps or concurrent host work make longer pauses more likely.

## Remedies

- Bring Renderer64's peak toward its 4 GB / 2 GB goal (performance goals
  G1). The peak, not the steady state, is what tips the Mac into critical
  pressure.
- Reduce host memory pressure while playing: quit memory-heavy Mac apps and
  avoid host test sweeps. The Mac is already compressing about 16 GB before
  the game starts.
- Not taken (user decision, 2026-10-18): raising the music ring's lead. The
  `push 8` at `sound.dll+0x33971` feeds the latency setter, and 16 would hold
  480 ms ahead. Memory reduction is the chosen fix.

## Probe

Build it in the VM with `Renderer\tools\BUILD_SOUND_SERVICE_PROBE.bat`, then
start it elevated before a scripted run:

```
sound_service_probe.exe C:\ProgramData\c3x-sound-probe\run.txt 420 60
```

Run it as the user with `... rec.txt 40 0 --record` for a loopback recording.
Summarize on the Mac with
`python3 Renderer/tools/sound_probe_report.py run.txt [--record rec.txt]`.
The report exits 1 for an interval of 280 ms or more, or for a ring replay.
