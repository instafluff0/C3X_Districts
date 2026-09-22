# Gate 1: isolated x64 render-to-x86 presentation trial

Run from the repository root:

```sh
python3 Renderer/native/helper_trial/run_gate1.py --visible
```

The runner builds a standalone x64 producer and x86 consumer on the Windows VM,
then runs a bounded self-test at 2240 × 1260. It does not stage C3X, call
`INSTALL.bat`, or launch Civ III. Add `--frames N`, `--width W`, and `--height H`
for a bounded diagnostic run. `--visible` requests a desktop pixel witness;
an unavailable interactive desktop makes the visible gate **unconfirmed**, while
a wrong desktop pixel is a failure. A run without `--visible` may check the
graphics import and retained pixels but cannot certify actual display.

The producer and consumer must select the same hardware D3D11 adapter. The
producer owns two single-sample BGRA frames behind keyed mutexes; the consumer
imports them and uses the production `NativePresenter` DirectComposition path on
its own HWND. The native self-test checks frame identity and sampled pixels,
partial native overlays and untouched pixels, resize, stale/busy frames,
producer termination and restart, independent presentation while the HWND
thread is blocked, and GDI restoration after detaching the visual. It records
phase timing, both processes' virtual-address headroom, and driver-reported GPU
budget/usage. A separate 64-byte shared-memory/event round-trip measures the
control-path primitive across x86 and x64; the graphics self-test deliberately
uses a simple synchronous command pipe, whose round-trip time is not the proposed
production IPC latency.
Diagnostic readback is permitted for pixel verification; a production transfer
that requires a CPU framebuffer copy does not pass this gate.

Each run writes a small report in ignored `Renderer/native/build/helper_trial/`
under a unique invocation ID. `completion.txt` comes from the Windows child
process. A missing or mismatched completion means the process state is
**unconfirmed**: inspect the existing VM invocation before retrying. The runner
checks PE headers for actual x64 and x86 binaries and rejects a different
presentation backend even if the native program exits successfully.

The Windows 11 Parallels VM is ARM64, so both trial binaries run under Windows
emulation. Its timing is evidence for this supported test machine, not an
estimate of native x64 performance. A Gate 1 pass proves the cross-process
graphics/presentation boundary on this VM. It does not establish a gameplay
speedup, Standard-map latency, or that moving the full renderer is worthwhile;
those belong to Gate 2's same-input workload comparison.
