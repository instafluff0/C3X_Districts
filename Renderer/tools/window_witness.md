# Sampled window evidence (development)

`window_witness.cpp` is an optional **external** Windows helper. It targets one
process's existing main window through Windows Graphics Capture. It neither
injects nor creates a game presenter, and captures no desktop or other application
window. The system capture border remains enabled. It is wired into the opt-in game capture launcher behind a matching-build
admission receipt. This does not establish the complete live-workload fidelity gate.

Build with `Renderer/tools/BUILD_WINDOW_WITNESS.bat`. It builds a 64-bit x64
helper (using Windows emulation on the ARM VM); no renderer build, staging, installation
or game launch occurs. The development CLI is:

```
window_witness PID NEW_DIRECTORY 600 5 sampled-window-evidence
window_witness self-test NEW_DIRECTORY 5 5 sampled-window-evidence
```

The helper waits up to 30 seconds for the target's largest visible, unowned
window, then keeps that exact capture item. It does not follow a new window after
the old one closes. Extent is bounded at 2400×1400, storage at 2 GiB, and the frame
pool at two surfaces. Processing holds at most one chosen frame plus one being
dequeued; there is one staging texture and one packed CPU image. Encoding and
readback cost belong to this process, but still compete for the machine's GPU and
CPU and require an on/off overhead comparison before game use.

JPEG frames are a **lossy sampled window witness**, not replacement renderer
inputs or exact pixel oracles. The chosen cadence is a maximum, not a promise.
The timeline records compositor timestamps, receipt/save QPC brackets, size,
foreground state, no-frame opportunities and resize gaps. It counts dequeued but
unsaved frames and explicitly leaves other missed compositor frames unknown.
A complete duration does not mean every game frame was captured. Native popups
outside the chosen window, physical scanout, HDR fidelity and automatic recovery
onto a new window are not covered. Exact retained-renderer BMPs remain a separate
replay oracle.

Once per second the helper records target-process private/working-set bytes and
virtual-address region metadata. It reads no process contents. The executable's
PE header supplies the large-address-aware limit, so observing a 32-bit process
from a 64-bit helper cannot accidentally report terabytes of usable game space.
The address scan records its bound and whether enumeration reached it; reserved
capacity remains distinct from committed memory or heap fragmentation.

The five-second GDI motion/resize self-test passed with 23 JPEGs, a resize gap
and five process-memory samples. The production native/DirectComposition fixture
also passed with 119 window samples and two matching 970-frame input replays.
The native fixture has an opt-in integration:

```
python -m Renderer.native.record_gpu_frame --scene SCENE --out NEW_DIRECTORY --record-inputs --visual-only --input-soak-seconds 30 --window-witness-seconds 25
```

It builds/freezes the helper, targets only the launched test process by PID,
preserves both process exit results and marks the receipt diagnostic. It cannot
be combined with the frame benchmark. No game launcher or installed DLL changes.
Windows' capture border changes desktop edge pixels: an initial uncoordinated
run correctly failed the native fixture's exact desktop oracle. A named event
now confines observation to the soak interval, with a completion barrier before
the exact comparisons resume. Capture must be at least five seconds shorter than
the soak. No border suppression or pixel masking is used.

The 2240×1260 pressure fixture passed both process exit checks and all native
pixel comparisons. Window/input clock correlation differed by 0.044 ms. The
observer peaked at 32.2 MiB private memory and consumed 2.22 CPU seconds over
24.13 seconds; GPU contention and live-game overhead remain unqualified. The
target's minimum available address space was 282.2 MiB despite a 1 GiB virtual
reservation: this does not reproduce the earlier live 120 MiB envelope or heap
fragmentation. Local evidence is under
`Renderer/native/build/input-recording/window-native-pressure-20260921-c`,
`window-native-pressure-aligned-20260921-c` and
`window-native-pressure-replay-20260921-c`.

Review existing evidence without the VM:

```
python -m Renderer.tools.inspect_window_witness --witness CAPTURE_DIRECTORY --out NEW_REVIEW_DIRECTORY
```

This validates frame ordering and storage lengths, writes a bounded contact sheet
and produces a local `review.html` with sample/time selection and memory context.
A killed helper's last torn timeline record is recoverable when no completion
receipt exists; it remains incomplete. Add `--inputs INSPECTION_DIRECTORY` to
align a verified input journal's inspection output. Alignment requires overlapping
QPC intervals and consistent QPC/UTC brackets. It reports only the most recent
completed presentation **candidate** and its age, never equality with that
window image. Wrong-clock and non-overlapping sessions fail instead of appearing
aligned. `python -m unittest Renderer.tools.test_window_witness` exercises these
controls and interrupted evidence recovery.

Microsoft documents [window targeting](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.capture.interop/nf-windows-graphics-capture-interop-igraphicscaptureiteminterop-createforwindow),
[frame lifetime and resizing](https://learn.microsoft.com/en-us/windows/apps/develop/media-authoring-processing/screen-capture),
and [the compositor QPC timestamp](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.direct3d11captureframe.systemrelativetime).
`compositor_100ns` uses the system-relative QPC timeline in 100 ns units; renderer
input journals separately store QPC frequency/origin and the UTC correlation
bracket. The reviewer aligns the two without assuming frame numbers or input-call
completion equal what Windows composed. Address limits follow Microsoft's
[process memory limits](https://learn.microsoft.com/en-us/windows/win32/memory/memory-limits-for-windows-releases).

The launcher requests an orderly stop by creating `stop.txt` in the observer
directory. A successful requested stop is complete for the sampled interval, not
proof of the full requested duration. The launcher parser/quoting/stop control
passes against an owned test window with 13 samples; no game is launched by
`test_capture_launcher.ps1`.
