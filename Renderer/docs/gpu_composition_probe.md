# GPU composition feasibility

## Decision and bounded implementation

The user authorized a prototype and necessary verified patch-table entries on
September 14, 2026. The production pixel path remains the retained renderer. The question
is whether image-operation translation can keep a completed map on the GPU through
ordered native composition and one native-driven presentation boundary.

Audit the installed JGL implementation, not just Civ III's wrappers. Classify image
copies/fills, sprite/palette drawing, text, clipping, CPU pixel access and screen
publication. CPU source-asset processing is compatible with upload-on-change;
live destination reads require an explicit synchronization or translated operation.
Do not infer backend coverage from GDI imports or a successful HDC substitution.

The isolated x86 probe compares three downstream paths with identical resident map
pixels and ordered operations: staging readback plus CPU/GDI composition, DXGI GDI
composition, and GPU image primitives with cached CPU-rasterized text. Include
clipping, transparent sprites, destination-dependent operations and background
save/restore, then present only the completed image. Check against an independent
CPU result outside timing. Use CPU wall time, separating submission/presentation
from an equal diagnostic completion barrier; Parallels GPU timestamps are unusable.

This measures composition feasibility and its cost, not world rendering, native
call coverage, physical display latency or gameplay speed. It does not install a
second game presenter or alter the current DLL. Promote only after an audited
native surface-to-screen path can preserve complete UI transactions and exact
camera/visibility/picking identity. No timing-based screen exceptions.

## Audited native responsibilities

The installed GOG `jgl.dll` hash is
`0b0cd514de0d95b93d20655f4b5194173fe257325af82e558a152305ff0dbdf2`.
The following are **DLL-relative RVAs**, not executable patch-table addresses.
They apply only to this binary. Exported factory construction installs Graphsy
vtable RVA `0x685f8`; image construction installs `0x68238`; the concrete sprite
vtable is `0x68440`. Base-class tables contain stubs and are not the live methods.

| Operation | Concrete evidence | Consequence |
| --- | --- | --- |
| Image creation | Graphsy slot 31, RVA `0x3bff0`; image init slot 1, `0x1800`, creates a DIB/DC. | CPU image storage has its own lifetime; replacing only its DC does not replace its bits. |
| DC acquire/release | Image slots 10/11, `0x1b20` / `0x1b40`, return the stored DC and maintain lease counts. | An HDC adapter must preserve native ownership and nesting. |
| Clipped fill | Image slot 17, `0x2270`, dispatches the 16-bit helper `0x3f70`; that obtains slot-7 pixel data and writes rows. | HDC substitution cannot redirect this draw. |
| Image copy/stretch | Image slot 16, `0x1ec0`, acquires both DCs, calls `StretchBlt`, releases both. | Copying into an ordinary CPU image brings the GPU-to-CPU boundary back. |
| Sprites | `Sprite::draw` calls sprite slot 17, `0x8180`; its 16-bit path reaches `0xa1d0`, which obtains destination slot-7 pixels. | GPU sprite operations must preserve palette, clipping, transparency and destination-dependent variants. This is static call-path evidence, not complete live coverage. |
| Text | `PCX_Image::do_draw_text` retains native markup/layout and ultimately invokes JGL operations; the DLL also imports `TextOutA`. | Cached text images are plausible; the synthetic font witness is not complete native text parity. |
| Screen transfer | Graphsy slot 41, `0x3baa0`, gets the retained screen image DC and calls `BitBlt` into the window DC at member `0x138`. The EXE wrapper `FUN_00606780(RECT*)` draws final tooltip/cursor material before this call. | There is a concrete native-driven final transfer seam to investigate. Its full caller/lifecycle coverage is not yet proven. |

`test_jgl_image_operations.cpp` loads this hash-pinned DLL in an isolated x86
process, without the game. Per-instance observation wrappers confirm a clipped
16-bit fill uses **zero DC acquisitions and one pixel acquisition**. A copy acquires
one source and one destination DC. Both compare all 3,072 native pixels exactly;
leases and original vtables are restored before destruction. This initial witness does not patch a running game. The next integration is below.

## Reproduction and interpretation

```sh
python3 -m Renderer.native.record_gpu_composition --map Renderer/native/build/world-content/control-final-zoom/zoom.bmp --jgl Renderer/native/build/gpu-composition/audit/jgl.dll
```

Use an existing production-rendered BMP under the checkout. `--jgl` is optional;
the local DLL is deliberately not distributed. The driver builds through the
existing native build owner and records invocation-specific logs, source/input
hashes, exact RGB parity, samples, completion and a retained executable. No staging,
installation or game launch. `--present` includes DXGI presentation in timing;
otherwise the swap chain is created only for an untimed final presentation witness.

The CPU control includes staging → publication bytes → DIB → CPU canvas, then an
upload into the common GPU destination. That final upload is **not the installed
native screen-transfer path**. It makes destination comparisons explicit but cannot
establish a production speedup. GPU/GDI internals may transfer data even when there
is no application readback. A shared 1-pixel staging completion probe is reported
separately, adds synchronization, and is not proposed for the shipping GPU path.
It is not a physical presentation timestamp or calibrated GPU execution time.

The scene is a retained production map image with 64 synthetic transparent sprites,
cached non-antialiased GDI text, clip edges, popup save/restore and a destination
invert operation. Source asset processing is excluded equally. Six phases compare
exact RGB against CPU/GDI rasterization; alpha is excluded because the native final
image is opaque. This does not validate native 555/565 rounding, palette-shadow
rules, all font effects, or real Advisor/tooltip/button lifecycle behavior.

## Decision

GPU operation translation is technically viable for the tested semantics. HDC-only
replacement is rejected as a complete JGL backend because actual native fills and
sprite paths bypass it. Keep the existing pixel path as control. The user clarified the intended hybrid: CPU drawing/upload costs for infrequently
changed UI are acceptable; repeated map readback is the constraint to remove.
Keep native UI layout/rasterization where useful, upload changed opaque panels or
faithfully represented transparent elements, and retain them on the GPU. Scrolling
can change label placement without re-rasterizing the text. GPU operations own
map-dependent composition, background copies/restoration and required palette/shadow
semantics. A 20 ms one-time update is not a 20 ms per-frame budget; it can still be
visible if frequent or if it blocks the native caller.

The next integration responsibility is to collect the complete operation/read set
of the native map-to-screen surface family at the existing final transfer boundary.
Separate CPU-owned source images from operations that read/write the animated map.
Translate only the required destination operations; a wholesale UI rewrite is not
the objective. Unknown destination reads must remain explicit; do not hide them
behind stale CPU mirrors or per-screen timers.

Local binary/disassembly and run artifacts remain under
`native/build/gpu-composition/`. Timing results below distinguish the standalone
mechanism from game performance; licensed DLLs and map images stay ignored.


## Measured checkpoint

Final source passes exact RGB comparisons at 1440×900 and 2240×1192, six phases
per size, with 72 timed calls per arm in three rotated blocks. One completed
presentation is executed outside off-screen timing. Minimum sampled largest free
32-bit region is 2,037 MiB. This is a small isolated process, not a combined
renderer/game memory measurement. Target dimensions are capped at 2240×1192;
textures and CPU buffers have fixed per-process owners and RAII cleanup.

| Viewport | CPU readback/GDI/upload | DXGI/GDI | GPU primitives |
| --- | --- | --- | --- |
| 1440×900, mean including diagnostic completion | 11.95 ms | 3.70 ms | 5.30 ms |
| 2240×1192, mean including diagnostic completion | 12.86 ms | 9.22 ms | 6.08 ms |
| 1440×900, submission alone | 4.18 ms | 1.43 ms | 0.47 ms |
| 2240×1192, submission alone | 5.56 ms | 3.62 ms | 0.46 ms |

**The speedup magnitude and interop-versus-primitives ranking are not established.**
Completion waits vary heavily: the three 1440 GPU block means are 1.44 / 13.06 /
1.42 ms. Initial presented runs gave 4.00 / 2.64 / 1.41 ms, while a repeat gave
17.00 / 17.11 / 10.90 ms. Deferring swap-chain creation did not eliminate the long
completion tails, so presentation pacing alone is not a sufficient attribution.
A process-name check found no concurrently running game or renderer after the
repeat; this is not continuous GPU exclusivity evidence. Do not reuse the fastest
run as a promised game gain. Existing unreliable timestamp/event findings remain.

The demonstrated elimination is application map readback and CPU map copies on
the two GPU-destination arms: 5.18 MB or 10.68 MB per frame, excluding the equal
4-byte diagnostic probe. The tested CPU arm redraws overlays each call; it is not
a comparison against cached static UI. This checkpoint supports the hybrid's
technical feasibility, not a claim that CPU static-UI generation must be replaced.

Evidence directories: `3ecf8f9c5c4b471cbc1ade88d59a7eee` (1440),
`b6d0be3b3d97456fb0221f357fdc2a6b` (2240), and
`b26d50f0a10c43a899bf6b8c4d45bc46` (actual JGL witness and earlier presented repeat).
Failed builds and earlier measurements remain preserved. The preserved control DLL is `55276aec…f196ca`. The initial probe did not change
injected source or patch entries; the subsequent observation integration does.


## Native integration: observe before substituting destinations

`injected_code.c` now supplies permanent, typed pass-through wrappers for image
lifetime, direct pixel/bits/DC access, copy, fill, image drawing and sprite drawing.
The map-capture target is interpreted through the same PCX view used by the current
compositor. The two authorized GOG entries in the patch ledger identify the native
final transfer and its screen image. Its original tooltip/cursor/transfer body runs
unchanged. This is an optional DLL export, not a changed required renderer ABI.

Only the audited JGL hash and expected concrete method addresses attach. In
addition to the table above: deleting destructor is slot 0 / `0x15d0`
(`image, unsigned flags -> image`); pixel core is slot 3 / `0x1bc0`
(`image, x, y -> pointer`); bits core is slot 4 / `0x1b70`
(`image -> pointer`); image draw is slot 33 / `0x2410`
(`image, destination, x, y -> int`). The actual sprite witness copies JGL row stride and color-mode metadata; its
16-bit stride is in native pixel units, not DIB bytes. Native methods use thiscall, bridged by
injected fastcall wrappers with the unused EDX argument where needed. Init accepts
width/height/bits/mode; copy accepts destination/source rectangle/target rectangle;
fill accepts rectangle/color; sprite draw accepts destination/x/y/palette.
JGL aliases 5/7 and 8 call core slots; direct internal sprite helpers remain
attributed to their enclosing image operation rather than counted as virtual calls.

The caller-thread collector holds at most 256 live identities, 512 directed copy
edges and 64 KiB of pending log text. Reinitialization/destruction retires identities.
It records attempts and access counts, **not** replayable commands, modified pixel
areas, successful-copy proofs or GPU work. Role bits 1/2 identify native map/screen
roots; edges connect other surfaces without claiming that every CPU image belongs
on the GPU. Dimensions can initially be unknown for a copy source. Unattributed
access remains explicit. No pointers, paths, text contents or pixels enter logs or
workers. Writes are buffered until the outer native transfer returns; summaries
cover 120 native transfers. At 8,192 transfers the hook set detaches. Overflow is
reported as `dropped`, and an incomplete capture cannot establish coverage.

The observer runs only on the attaching native thread. Existing pointer retention,
other threads, internal calls and other JGL methods can still evade attribution;
absence of an observed access is not proof that CPU access is absent. Reset/unload
and config-off preserve native fallback. Even if restoring page protection fails,
wrappers live in permanent injected code and stop calling the optional DLL.
No timer, delayed UI paint, new presenter or GPU destination replacement is added.

Reproduce the extracted injected-hook contract without the game:

```sh
python3 -m Renderer.tools.audit_native_composition
python3 -m Renderer.native.record_native_observation --jgl Renderer/native/build/gpu-composition/audit/jgl.dll
```

The strategic live checkpoint is one capture covering idle animation, scrolling,
Advisor open/page switch/close, a popup and unit command-button refresh. Logs begin
`[C3X native]`; after evaluation staging, use ordinary `INSTALL.bat`, without
environment settings. The next
implementation uses the observed map-to-screen chain to admit a GPU destination
and translate its map-dependent operations, retaining CPU source UI where useful.
Unknown destination access must be resolved or retain the whole existing path;
partial GPU/CPU mirrors would reintroduce stale backgrounds and flicker.


Validation checkpoint: the exact extracted injected hooks pass against the audited
DLL, including 3,072-pixel fill/copy/image/sprite checks, clipping, access attribution,
reinit/destruction, temporary-image counts, deferred logging, config-off, foreign
thread exclusion, missing capabilities, detach/reattach and bounded ownership.
Receipt: `native/build/gpu-composition/354612ea2cad4b15b4e6602acc0e9d28/`.
The GOG byte audit, required injected compilation and 45 existing UI/cadence/GDI/
bridge/view-identity contracts pass. The normal production city scene (263 visible
tiles, no fallback) matches the preserved control BMP byte for byte. The older
synthetic renderer smoke reaches the same camera-ticket assertion failure on both
control and candidate; it is not reported as a passing test or a new regression.

Normal candidate `038faec0…0be2e2` is preserved under
`native/build/native-observation/evaluation/`, with receipts in its parent.
The user granted staging permission and this exact observation DLL is now in
`Renderer/bin`; control `55276aec…f196ca` is preserved in
`native/build/native-observation/control/`. No installation or game launch occurred.
The user is away; the strategic native capture remains pending without blocking
independent backend implementation.

## Packed GPU executor

`native/gpu_image_compositor.h` now replaces the probe's earlier GPU drawing arm.
It borrows a device/context and owns bounded image storage, revision-based CPU
uploads and ordered transactions: fill, copy, color key, invert and overlapping
save/restore. Compatible copies and complete clears use hardware operations;
keyed/partial/destination-dependent work uses integer compute passes. R32_UINT
storage preserves native words exactly; RGB555, RGB565 and BGRA32 have distinct
compatibility identities. Cross-encoding copies and unsupported commands reject
before any transaction draw. Reused handles cannot revive destroyed images, and
GPU writes prevent a stale CPU revision from overwriting a newer image.

Limits are 32 images, 2,048 commands per transaction, 2240×1192 per image and
64 MiB of owned texture storage including overlap scratch/replacement peaks.
Driver-retained allocations and the rest of the renderer are additional. There
is no presenter, native pointer, thread creation or execution readback in this
owner. CPU-source revisions remain the adapter's responsibility; these tests do
not establish complete source validity in the game.

The actual JGL oracle validates native 16-bit clipping, copies, keyed image drawing,
background save/restore, overlap and GDI inversion. Native inversion flips all
16 stored bits, including RGB555's unused high bit; RGB color keys ignore alpha.
Revision, lifetime, format, rejected-transaction and scratch-budget contracts pass.
The existing 32-bit composition fixture retains CPU/GDI controls, exact six-phase
RGB checks, native-size cases and the separately reported completion barrier.
Run the native oracle with the existing driver plus `--gpu`.
Evidence: `903200226b2c4cfb877146caa2bfce3f` (native oracle) and
`110f19e09ee84e29b29cd272aeb0e05a` (final 2240×1192 executor, six phases,
72 timed calls/arm). CPU-control / GDI-interop / GPU means including diagnostic
completion were 8.21 / 4.85 / 2.84 ms; GPU submission alone was 0.27 ms.
The prior 1440 executor checkpoint `d00ab196edef4b258957a5c771179ab5` also passes
exact RGB. These are downstream fixture results, subject to the existing control,
completion-wait and Parallels limitations above, not measured game acceleration.

## Native image adapter checkpoint

`native/native_image_adapter.h` now executes actual injected copy/fill/image-draw
hooks against JGL images, through the same packed GPU executor. The harness calls
native methods; it does not issue a second manually matched GPU command sequence.
Admission is limited to successfully initialized 16-bit image lifetimes observed
after attachment. Preexisting surfaces stay CPU-owned. Native DIB masks distinguish
555/565; unsupported formats, stretch, palette modes and raw CPU/GDI operations
retain native behavior. Null-rectangle fills use a different native clear
helper and also fall back. The direct clip hook (slot 13 / RVA `0x1a40`) preserves
JGL's clipping body without treating its private metadata HDC as pixel access.

Admitted chains keep output on the GPU. Before public bits/pixel/HDC access or an
unsupported operation, dirty images synchronize back to their native DIB. Once a
pointer/DC escapes, that lifetime never becomes a GPU destination again. This
preserves later retained-pointer writes. CPU source images use exact full-word
comparisons on every use, uploading only changed content; pointer identity and
lease counters are not revisions. This scan has a caller cost and is not claimed
as a final fast source policy. Native init/destruction retire GPU identities;
config-off and detach drain dirty images before removing the backend.

Limits remain 32 images and 64 MiB GPU storage; the adapter additionally caps
retained CPU comparison bytes at 64 MiB, with one image-sized comparison temporary
or staging readback at a time. Native DIBs, driver allocations and the rest of the
renderer are additional. A synchronization/device failure terminates the isolated
backend; production device-loss recovery is not implemented. No stale-pixel
fallback, second presenter or thread is introduced.

Reproduce with `record_native_observation --adapter --jgl` and the same local DLL
path above. The six-phase native oracle covers clipping, fill/copy/key, overlap,
popup save/restore, retained-pointer source edits, stretch, CPU destination GDI
inversion, lifetime reuse/destruction, config-off and detach. Admitted rendering
chains perform zero execution readbacks; oracle readbacks are separate. Explicit
CPU handoffs do read back and are counted. Pass-through observation still passes
its own extracted-hook contract, including the exact staged DLL export.
Receipts: `0f0a3e92c9b347228916f81a5865e8c5` (adapter: 44 translated operations,
14 source checks, 10 uploads, seven explicit fallback readbacks / 86,016 bytes)
and `f684b988aa0344cda21ee179a6ec9fbd` (staged observer compatibility).
The normal renderer candidate and required
injected compilation pass; 45 existing UI/cadence/GDI/bridge/view tests pass.

## Resident map → existing worker → composition

The normal renderer DLL now exposes `c3x_renderer_gpu_render` and
`c3x_renderer_gpu_images` using `native/gpu_frame_api.h`. The first copies the
captured request, renders full detail through the existing world/view/pass owners,
unwraps the circular finished surface into a sampleable GPU texture and imports
it directly into the packed-image compositor. It returns an opaque map ticket and
native replacement metadata, with no CPU pixel pointer. Producer readbacks are
counted and must be zero for success. CPU bitmap validity is explicitly separate;
returning to CPU rendering rebuilds current pixels, including after GPU failure.

Composition packets create/destroy images, upload revised CPU sources, and submit
ordered copy/fill/key/invert commands on that same `RendererWorker` and immediate
context. Caller arrays are copied before queueing; native pointers, HDCs and COM
objects never cross threads. Maps are immutable. A new GPU frame retires the old
map/ticket while retaining UI images; CPU demand/configuration/reset retires the
session. Image and whole-command validation precede mutation. Readback requires an
explicit request and is reserved here for the oracle or a future native CPU barrier.
Success establishes GPU command order, not physical completion or scanout.

This path admits the existing bounded city profile without waves/reflections.
Other profiles retain the current native CPU path. The compositor cap is 64 MiB
including map/UI images and overlap scratch, plus one BGRA view (at most 10.2 MiB).
Copied upload/readback buffers each cap at one 2240×1192 image; metadata and command
queues are bounded. Existing renderer/native/driver allocations are additional.
No second GPU device, worker pool or presenter is created.

`record_gpu_frame --scene <existing scene.csv>` runs the existing capture/production
harness against a candidate DLL. Stationary, animation, scrolling and local-change
cases compare exact pixels and replacement flags, then check GPU UI composition,
upload reuse, immutable-map rejection, ticket/image lifetimes and exact CPU fallback.
The test's explicit readback occurs after GPU composition and is counted separately.
Normal build, matching TCC C/MSVC C++ packet layouts, the existing native JGL
adapter and 45 UI/cadence/GDI/bridge contracts pass. The 1440×900 checkpoint
`36e73263463d455080a0490d61fd4a7a` matches all pixels and ownership in four cases;
528 tiles are visible, 784 captured. The matching 640×480 checkpoint is
`80da53f379664d76bf34488e4f185751` (263 visible / 592 captured). At 1440×900,
this avoids 5.18 MB of map readback per refreshed
frame in this route, while adding bounded GPU unwrap/import work. The isolated
32-bit process retains a largest free address region of about 2.0 GiB; this is not
an in-game memory measurement. This validates a real
renderer-to-compositor path, replacing the earlier CPU-uploaded map seed for this
fixture; it does not yet measure a native final-presentation speedup.

The staged observation DLL remains intact. The native JGL adapter still needs its
packet transport connected to these worker exports, complete sprite/palette/text
and CPU-access coverage, device-loss recovery, and one native-driven GPU final
transfer. The current GDI final transfer requires CPU pixels and is unchanged.
Live native capture remains pending. Static UI can remain CPU-rasterized and
uploaded on change throughout this architecture.
