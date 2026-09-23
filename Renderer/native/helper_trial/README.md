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

## Gate 2: real renderer scene and GPU map transfer

Run from the repository root with a retained local input journal:

```sh
python3 Renderer/native/helper_trial/run_gate2.py --capture Renderer/native/build/input-recording/<capture>/inputs
```

The runner builds the current renderer core in both x86 and x64, extracts the
same recorded scene values, compares result identities and CPU pixel witnesses,
and transfers real x64 GPU map images to an x86 D3D11 importer. Add
`--verify-pixels` for diagnostic GPU readback and exact pixel comparison; add
`--crash-after 27` to test helper restart mid-workload. The ignored receipt
under `native/build/helper_trial/gate2/runs/` records source, binary and capture
hashes, completion, parity, timing and process memory. A scene-only pass is not
a native-interleaved gameplay speed or presentation result.

Use `--raw-shared --verify-pixels` to exercise the zero-readback `R32_UINT`
map transfer and exact imported pixels. For the complete recorded call stream,
run `python3 Renderer/native/helper_trial/run_interleaved.py --capture
Renderer/native/build/input-recording/<capture>/inputs`. That diagnostic mirrors
GPU/native visual policy, image and unit lifecycles, final-image composition,
and ambient frames; it still renders the x86 control simultaneously and does
not qualify production FPS or the post-reset CPU/native route.

The [Gate 2 results](../../docs/helper64_gate2_results.md) close the x64
architecture/capacity decision. The [migration plan](../../docs/helper64_migration_plan.md)
defines the complete replay, native composition and cutover acceptance still
required. This trial never stages a DLL, calls `INSTALL.bat`, or launches Civ III.

## Direct cross-process composition surface

`call helper_trial\probe_direct_surface.bat 1120 1192` from `Renderer/native`
builds a separate x86-window/x64-producer proof. Omit the dimensions for a
320×240 smoke test. The x86 owner creates one DirectComposition surface handle,
duplicates it into x64, and binds a wrapper to its own HWND once. The x64
process owns the BGRA swap chain and presents ten subsequent frames. The x86
window thread samples desktop pixels without pumping messages or adopting a
per-frame shared image. This is a graphics and window-layer feasibility test,
not a Civ III launch, renderer-core benchmark, frame-rate test or staging step.

The Windows VM showed all ten x64 presents at both sizes, with nine map-color
transitions seen while the owner thread was unpumped. A native child-window
pixel remained on top. The composition map covered a native pixel drawn directly
to the parent HWND; Civ III's same-HWND labels and HUD therefore need a separate
upper composition plane (or an equivalent exact ownership solution) before this
path can replace the current presenter. See
[the direct-surface findings](../../docs/direct_surface_trial.md).
