# GPU composition implementation and evidence

The final section and [retained plan](retained_renderer_plan.md) describe current
status. Earlier sections preserve checkpoint findings, including superseded next
steps; device-loss reconstruction is now user-deferred.

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

## Resident map → native operations → native display

Optional `c3x_renderer_gpu_render`, `c3x_renderer_gpu_images` and
`c3x_renderer_gpu_present` connect the existing world/view/pass owners to an
immutable BGRA map, native operation packets and actual JGL final transfer.
`native_image_adapter.h` keeps native-word/full-color pairs for admitted fresh
map/screen/save lifetimes. Copies propagate both; fills and ordinary keyed UI
expand native colors over unchanged map pixels. Unsupported sprite modes/GDI/raw access
restores native words and relinquishes ownership. Native pointers stay on the
caller; up to 2048 commands coalesce on the existing GPU worker.

`patch_JGL_Graphsy_present` runs after native tooltip/cursor drawing and preserves
palette binding. It uses the supplied native HWND without creating a window or
requesting redraws. DXGI lifecycle/Present stay on that window's thread to avoid
[DXGI/window-thread deadlock](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/dxgi-best-practices).
A retained display preserves partial updates; recreation requires a full transfer.
The executor caps images/maps/scratch at 32 images / 64 MiB, caller packets at
176 KiB and CPU comparisons at 64 MiB plus one temporary image. Native DIBs,
world targets and the separate presenter budget below are additional.

Reproduce with `python3 -m Renderer.native.record_gpu_frame --scene <scene.csv>`.
Preserved receipts under `native/build/gpu-composition/`: `608e7d25ec424d7ba276d36fbcfd5a8c`
(1440×900, 528 visible/784 captured), `0ede3a048c024fb9afcc026d4fe499d7`
(640×480), `898b44c431e44ddaa57e05ea2b27819c` (observer compatibility).
Stationary/animation/scroll/edit, native pixels, full-color display, palette,
popup restore, partial transfer, recreation and config-off pass. The admitted
chain performs zero execution reads; explicit config-off restores three images.
The separate adapter stress fixture translates 47 operations with ten intentional
fallback reads / 122,880 bytes. Oracle reads are additional. The map producer
eliminates 5.18 MB/readback at 1440×900 in this fixture, not in the live bridge.

Preserve expensive diagnostics: `07b5131e653d4889be201dcad2211a65` found incorrect
RGB565 shader expansion, fixed with explicit bit extraction; its compiler cause
is unproven. `40aca9375e854f4c84459402247cf5ed` records a failed Parallels dispatch.
`c61ff1963f5c4f4d912a37a47a217bfc` / `b247c0ae7dba4dc4adff705eb999384c`
established that JGL's final transfer needs a palette-owner interface, not a bare
palette. The fixture now initializes actual native palettes and owner wrappers.
`6c7ba2a4b68347db8d76587076ab6828` attributes the large display mismatch to a desktop
narrower than the test window; moving that same window verifies every pixel without
changing desktop settings. Earlier controls and unreliable GPU timestamps remain
preserved. None of these capture diagnostics is a presentation-latency benchmark.

## Live final-transfer compatibility integration

The loader now resolves `c3x_renderer_native_image` behind the audited GOG/JGL
hooks. This callback consumes only final transfer/drain; **existing map and UI
images remain CPU-owned**. `native_screen_bridge.h` borrows current JGL bits and
stride, copies the requested native words, and returns the lease before worker
submission. Retained-pointer changes are observed on each request. No pixel
comparison, screen-type guess or timing-based UI completion rule is used.

`NativePresenter` uploads packed R16 words and uses the shared `ImageDisplay` pass
to expand 555/565 UI colors. Full-color map display still uses unchanged BGRA.
Partial updates retain untouched display pixels. The compatibility route keeps a
packed CPU copy of the displayed image so returning to GDI can restore exactly
that image before the original partial transfer; uploading the entire current
native screen instead could reveal unfinished UI outside that rectangle.
Shaders survive window handoffs and invalidate on device replacement. Presentation
stays on the native window thread; a queue barrier protects device access without
retiring map publications, preparation or unit poses. Active prospective work
causes immediate native fallback rather than cancellation or an added frame wait.
Unsupported windows/formats, partial first frames and GPU failures also fall back.
Config-off/reset release window ownership while native CPU pixels remain current.

At 1440×900, the first correct BGRA compatibility path cost about 7.2 ms/callback;
a bounded snapshot diagnostic attributed 3.68 ms to native capture/CPU expansion.
Uploading native words instead reduced the callback to a 2.42–3.49 ms block mean across repeated runs,
versus native GDI 1.92–2.72 ms in the same alternating blocks (about 0.5–0.8 ms
additional callback cost).
Both arms including the common desktop completion barrier were about 16.7 ms.
This is a **final-transfer fixture**, not whole-request/game speedup evidence.
Uploads halve to 2.59 MB for a full 1440×900 transfer; partial transfers upload
only their rectangle. Existing map readback remains in the live path.
The compatibility presenter adds ten GPU bytes/pixel (native source, retained
BGRA display, swap buffer; at most 25.5 MiB), plus bounded packed CPU snapshot,
worker packet and displayed shadow (at most 15.3 MiB combined). Existing native
DIBs, world targets and driver allocations are additional.

Current-code fixtures check the actual live export with preexisting CPU surfaces,
retained-pointer edits, both native color formats, full/partial display, interleaved
map publication, reset/recreation, observation expiry and partial config-off/GDI
fallback. RGB565 GDI fallback is checked with the test window fully onscreen;
large-window GPU checks expose every pixel by moving only that test window.
Receipts and candidate/control hashes are indexed in
`native/build/gpu-composition/live-screen-checkpoint.json`.

Preserve `f15f19de14934e37bd9ea1a2956e6ae9` / `1f01f884cc7c45ac8044df8872912b74`:
GetObject height did not preserve JGL's logical row order; its native lease/stride
fixed the inversion. `3c320c66097f4fb090116c249ee1d003` found the off-desktop GDI
clipping limit in the capture fixture; the tested dirty rectangle is now exposed
before native transfer. `add975dd282446a6b3a36c3b4a34a218` preserves the slower
CPU-expansion path and its attribution diagnostic. These findings do not justify
repeating closed timing experiments or treating Parallels GPU timestamps as valid.

The prior observer `038faec0…0be2e2` remains the rollback control. The current build
is an evaluation of GPU presentation through normal INSTALL.bat; no game launch
or installation is performed by the tooling. The pending strategic game checkpoint
covers scrolling/zoom, Advisors/popups/buttons, overlays/picking/unit actions and
window restore together, with `stage=native-screen` route/callback counters.
The remaining architectural connection is exclusive map-to-screen ownership and
map-dependent sprite/unit/access coverage, so the resident full-color map can
replace the live CPU bitmap path. This compatibility integration does not remove
map readback or prove that ownership. No new CSV entries are needed.

## Sprite and session ownership checkpoint

Existing sprite slot 17 now translates ordinary 8/16-bit and row-trimmed 8-bit
sources. Coverage is explicit: indices 254/255 skip; opaque entries with the same
RGB still draw. Palette/source/row-header edits are checked on use. One bounded
decoded source reuses uploads; this historical checkpoint retained native scaling
and unsupported key modes (coverage was subsequently extended below). Paired native words/BGRA keep the map's full color.
CPU map/unit jobs no longer retire the GPU session, and image/present packets
preserve prepared views and unit playback. Map admission publishes transactionally:
image-budget rejection preserves the prior map, ticket and native images.

Current receipts `34673598b4cf460ea73472fd87a657f2` (1440×900) and
`2671be8b0b084bcfad84504c72e42afe` (640×480) pass actual hooks, every native color
expansion threshold, clipped/trimmed sprites, palette edits, actual custom-unit
and CPU-map interleaving, saturated image admission and full/partial display.
The sprite fixture translates 57 operations, with ten intentional fallback reads
(122,880 bytes); the admitted map/display chain has zero execution reads before
explicit fallback. Map oracle reads remain separately counted. These are
correctness/work-elimination checks, not a live latency improvement. The staged
compatibility DLL remains `a79fb579…dec552`; this checkpoint is not an all-GPU
game build. Live admission, map-dependent unit/GDI access and recovery of dirty
GPU-only native images after device loss remain unfinished.

Preserve `087a5838f6504b009355309adce95cb4` and
`c0680459a50144ff87a5cb08531e02ef`: nested command initialization produced an
observed kind 0, rejecting the sprite copy format. Explicitly initializing a
single command before copying the pair passes; the compiler cause is unproven.
`7941ccb0301c4a0bb2d176588084edb4` exposed disabled unit assets in the GPU fixture;
it now enables the production unit definitions before testing real interleaving.

## Unit composition connection

`c3x_renderer_gpu_unit` shares `RendererWorker::draw_unit` playback and pose
selection. Prepared body/shadow pixels feed a bounded reusable GPU source; paired
native-word/full-color outputs blend against resident destination and underlay
images. Only the selected body rectangle is copied on the GPU. Native key tests,
clipping, aliased underlays, erase bounds and actual unit display pass against the
existing CPU blitter at 1440×900 (`3754bdcdcce948d0b1c5d6846d175b58`). The same 640×480
chain passes (`884eb54116a94c3a85a90c72978c6cac`). These checks
cover all 256 alpha values and both native formats, with zero background reads.
The native unit hook now offers this operation before obtaining either DC.

The compositor's 64 MiB cap includes the source and two destination snapshots;
source comparison and the worker pose copy add at most 8 MiB of CPU storage beyond
the existing pose publication/cache. Optional command packets now include paired
image handles and verify the command struct size. Configuration/reset own cleanup.
Cold pose rendering still finishes through the CPU cache. Live surface admission,
GDI access and device-loss restoration remain required; this is not an all-GPU
game build or whole-request speedup claim. The staged DLL is unchanged.

Preserve `b99b94f78ba14e6ab137400704e8c1df`: the initial unit shader produced an
incorrect RGB565 green channel. Explicit bit extraction, matching the validated
native expansion pass, fixes the oracle mismatch; the compiler cause is unproven.
The next GDI boundary is concrete: audited JGL text slot 46 (`0x1de0`) calls
`TextOutA` (IAT `0x6802c`), then releases its DC; slot 42 (`0x1d10`) selects the
font through `SelectObject` (IAT `0x68054`). These are verified native operations,
not newly enabled hooks or permission to infer arbitrary DC lease lifetimes.

## Startup evidence and destination demand

`patch_init_floating_point` loads the optional lifetime service before native
constructors, with a pinned module reference separate from scenario renderer
ownership. The service records successful INIT and invalidates on destruction,
reinit, public pointer/DC escape or foreign-thread access. It never dereferences
native pointers; its 1024-entry limit rejects overflow without evicting live proof.
Audited private copy/fill/sprite/text/metadata leases do not escape. Font/color/text
slots are listed in the patch ledger; text still executes its original GDI body.

`bbcdc78503ff4d72b1b0f8d37e3c9288` passes the actual extracted bootstrap and lifetime
contracts, including config-off native pixels and scene detach. The injected smoke
and 44 native contracts pass. `321cb836e6b94cd7850c2a33576c34c6` connects this evidence
to demand-based adapter admission at 1440×900: canvases exist and receive native
drawing before GPU attachment, then map/copy/unit/popup/final-display operations
remain resident. Unused surfaces allocate nothing; older unobserved and escaped
surfaces are rejected. This closes the fixture's attachment-order assumption; it
does not enable the live resident map endpoint or prove game performance.

The GPU presentation API now separates intentional window discard (action 1)
from return to native GDI (action 2). The latter snapshots the **last displayed**
GPU surface on the existing worker, releases DXGI on the native thread, and restores
that image before native partial drawing resumes. Normal GPU presentation keeps
no CPU shadow and incurs no new readback. This fallback uses one bounded BGRA
snapshot plus staging surface (at most 21.4 MB transient at the supported maximum
extent); compatibility presentation reuses its existing CPU display shadow.
Final connected receipts `0aed12fe601840cea167343110f2cf5c` (1440×900) and
`08ff79542d404f6fa17db3a66f1ab5e8` (640×480) verify map-family-only admission and
preservation of displayed pixels despite newer working-canvas content, followed
by a native partial GDI update. Device-loss reconstruction remains unresolved: a
failed readback is not permission to publish stale native pixels.

The broad build's base native smoke now passes ticket reuse for identical camera
requests, adoption by synchronous demand, and fencing of changed requests; its
older always-new-ticket assertions were obsolete. The separate approved-terrain
boundary witness still fails (2,755 pixels above its tolerance); the preserved
`a79fb579…dec552` control also fails (2,226 pixels). Counts differ, so this is not a
byte-parity proof or a clean full-suite result. The visual threshold is unchanged;
logs are preserved with the admission/handoff checkpoint.

## Production native map owner

`native_composition_owner.h` connects the tested adapter/worker to the live DLL
exports. `composite_custom_renderer_frame` prepares the resident map, applies its
existing replacement validator, then commits without a destination HDC or CPU
bitmap. Rejection cancels the pending insertion. Existing JGL hooks route copy,
fill, sprite, unit and final transfer operations through this owner. Configuration,
reset and detach drain while native images and the GPU session still exist.

`707e940f1cd74782aaa7f96b56489b57` (640×480) and
`06afb80d775e496fa9898860d5f89144` (1440×900) execute the production exports through
real JGL images: pre-map initialization, prepare/cancel, wrong-target rejection,
current metadata, unchanged CPU map storage after commit, copies, actual units,
next-view reuse, exact displayed pixels and reset handoff. Unsupported surface
sizes and GPU image pressure reject admission for the existing CPU path, preserving
the prior ticket. They retain the existing
producer's stationary/animation/scroll/local-edit checks. These are work-removal
and correctness witnesses, not game cadence or whole-request performance evidence.

Ordinary bounded native labels now use `native_text_raster.h` and the same adapter:
Windows rasterizes the selected font on synthetic backgrounds once; cached response
data blends over the resident map on the GPU. A 32-run / 8 MiB LRU shares the existing
64 MiB image budget; preparation uses under 2 MiB scratch. GPU image handles expand
from 32 to 128 to accommodate small glyph assets without changing the byte cap.
No map pixels, HDCs or game pointers cross to the GPU worker. Native text-edge color
differences are user-authorized; font choice, layout, clipping and smoothing remain
native. Unsupported transforms/complex clips, current-position/RTL text and runs
above 16K raster pixels retain the explicit CPU fallback.

The bounded diagnostic rejects direct 16-bit/32-bit GDI parity: default smoothed
text differs. Exact per-channel native/full-color response compilation took about
34–37 ms for the sample. Following the user's tolerance clarification, 17 background
samples reduce this to about 1 ms and match 32-bit Windows text within two channel
levels in the tested cases. This is preparation timing, not whole-request latency.
CPU UI source generation is allowed; repeating labels reuses resident GPU data.
Explicit RGB565 packing is required by the shader oracle; a combined conditional
shift/mask expression failed, as previously found for unit composition.

Unit finishing now uses the resident path described below. Asynchronous GPU map
publication and device-loss reconstruction remain incomplete. Failed native barriers return a
negative result; hooks deny stale leases rather than allowing native drawing from
old pixels. The older staged DLL is unchanged. No CSV addition or game launch.

The final candidate compiles; injected compilation and all 44 native contracts
pass. The extracted dispatch contract exercises resident success, CPU admission
fallback, hard-error propagation and unchanged identity epochs. Lifetime receipt
`ffae5513509f4ac69d604d8040ff4f59` additionally checks that failed barriers deny
pixel/bits/DC access and defer native reinit/destruction. These guards are not
reconstruction after device loss. No broader terrain-suite reclassification or
new visual acceptance is implied.

Text-connected receipts `afcc6b9316e841208a1c749a752e2982` (640×480) and
`1899844fc07042349bf5c442fcf0096c` (1440×900), candidate `16aae829…06f8cb`,
verify actual slot-46 calls through final display with untouched CPU screen storage.
The adapter proves one build/upload pair across three label anchors, two cache hits,
and no background readbacks. `BUILD.bat native-text` independently compares actual
D3D output with GDI over four backgrounds, both native formats and three font
qualities, with transparent/opaque drawing. Comparison rows are native 555, GDI 32,
and GPU in `native/build/gpu-composition/native-text-checkpoint/`. No injected source,
CSV, staged DLL, installation or game launch changed in this checkpoint.

## Resident unit finishing and publication

`UnitPoseContent` now owns translation-free ground-shadow coverage, using the
previous finishing formula. CPU helpers prepare it alongside posed geometry;
CPU compatibility and GPU finishing consume the same bytes. `GpuUnitFinish` combines
coverage with rendered alpha and premultiplies body color on the existing GPU worker.
Its integer output matches the previous finishing/blending path in the tested cases.

`UnitBodyRenderer` owns a 64 MiB / 512-entry resident pose cache with the existing
appearance/pose key; native placement is excluded. The compositor's last borrowed
source can retain one additional texture (up to 4 MiB) after cache eviction. Borrowing
replaces the old CPU body upload cache. Shadow upload scratch is at most 1 MiB;
a new cache allocation can transiently exceed the retained cap by 4 MiB before eviction. Unit target/material budgets are unchanged.
CPU/GPU cache usage counters are separate because CPU bitmap hits can run concurrently
with GPU preparation. Reset releases resident poses and finishing resources.

The existing upcoming-pose queue selects GPU publication after GPU unit demand;
normal authored timing, frozen-unit eligibility and caller-driven delivery are unchanged.
Optional preparation retains the existing memory-pressure guard and never notifies Civ III.
CPU fallback keeps its explicit bitmap path; ordinary GPU unit demand no longer enters it.
Candidate builds and CPU pose/playback/cache tests pass. Connected tests exercise cold
and cached poses, changed anchors/directions/hours/seasons/zoom, native 555/565 words,
full-color backgrounds, clipping and final display. Detailed harness logging checks
actual body-readback and composition-upload counters; normal trace filtering is not
used as evidence of eliminated work. No whole-request speedup is claimed here.

Final resident-unit receipts `4f4705429d244e6db155c7d099ad0d19` (640×480) and
`21d630d9401340529f461a032647e770` (1440×900) each verify 15 real GPU unit
requests, cold and cached, with zero body readbacks/composition uploads. Thirteen
CPU pose/playback/preparation/cache tests also pass. Candidate `23063ef9…c986ed`
and logs are preserved under `native/build/gpu-composition/resident-unit-checkpoint/`.
No new unit visual difference, injected patch, staging, installation or game launch.


## Resident map preparation and caller adoption

`PreparedViewArea` now owns the same complete capture/dependency proof for CPU
pixels or immutable GPU storage. GPU projection selects an offset view, without a
CPU crop; the existing worker imports that rectangle into the native image session.
Cold demand uses the same working extent as preparation and establishes coverage.
The existing 32 MiB per-area / 64 MiB retained-view limits count texture bytes
conservatively even when views share storage. The current publication remains
bounded to 32 MiB; render scratch and composition budgets are unchanged.

Native image/unit/presentation calls preserve queued area preparation. Incompatible
map demand cancels it at existing safe points; no callback or redraw request is
introduced. `c3x_renderer_native_map_view` returns the actual adopted ambient clock.
The original metadata-only export remains available to explicit controls.

Validation exposed and fixed two ownership problems: CPU fallback must publish a
temporary area crop before returning it, and an unchanged GPU scene still requires
one complete CPU transfer when its CPU mirror is stale. Demand/preparation now use
the same working extent, preserving the existing native CPU output contract.
The 32×32 fixture's duplicate wrapped occurrences intentionally reject area reuse;
a 128×128 periodic extension of its unchanged tile attributes exercises admission.

Receipts `d4adf124b06c45afa277bc13852a354b` (640×480) and
`34b1a57664d943ad877a66b039ddafa5` (1440×900) pass connected native composition,
background refresh adoption, actual sample clocks, exact CPU-reference pixels,
local changes, explicit CPU fallback and reset. Both retain the 15-request resident
unit proof. The injected smoke and native dispatch/publication contracts pass.
Candidate `5ee204b4…12d9e0`, inputs and logs are in
`native/build/gpu-composition/resident-map-checkpoint/`. Staging is unchanged.

Nine prepared map-call samples per size had median CPU return times 0.810 / 2.316 ms;
the larger-size maximum was 17.365 ms while preparation could delay demand. These
are diagnostic call timings, not completed-display latency or an overall speedup.
The final section records the complete request comparison and the ported
full-worker test, preserving its camera obligations against current interfaces.


## Native image-transfer completion

The existing image owner now submits positive, in-bounds native stretching,
keyed full-color copies and current-palette/null fills. One paired command tests
transparency against native words and writes native/full-color results together;
overlap snapshots count against the existing 64 MiB composition cap. Opaque
same-size copies keep their direct GPU-copy path. No new hook or native address.

Actual JGL uses `BLACKONWHITE`: enlargement selects the center sample; shrink
ANDs the source interval ending at that sample, beginning after the preceding
sample (the first interval starts at zero). Ordinary nearest sampling and simple
floor/ceil interval boundaries were rejected by native pixel evidence. Eight
clipped scale cases and independent RGB555/RGB565 GDI DIBs now pass exactly.
Native transparent self-draw instead traverses in place; it retains an explicit
CPU barrier, rather than incorrectly borrowing StretchBlt's overlap semantics.

Positive indexed sprite scaling now follows JGL's separate ordinary/row-trimmed
programs: truncated float extents and 16.16 source increments, with the ordinary
origin offset. Disassembly/oracle evidence confirms trimmed scaling includes the
byte after its count and stops advancing at an empty row. The adapter declines
unproven terminal reads and mirrored sources. Destination palettes resolve positive
16-bit keys; Sprite mutates its key, Image uses a temporary descriptor. Native
scaled 16-bit drawing is a no-op. Bounded CPU decoding uses at most 2240×1192 words
per scratch/cache vector; only source data uploads, never destination readback.

Connected receipts `8ba6aa46112f4878a615e44b7f09b833` (640×480) and
`420e34758a8d4d118d3e1ae0de6a3d2e` (1440×900) pass actual hooks, paired color
precision, live palette edits, CPU barriers, bounded ownership, prepared maps,
units and final display. Each retains 15 cold/warm resident-unit requests with
zero body readbacks/composition uploads. The admitted native image chain performs
zero execution readbacks; independent oracle reads are excluded from that claim.
Candidate `27409085…7a2a7c`, exact inputs, rejected mapping diagnostics and logs
are preserved in `native/build/gpu-composition/native-transfer-checkpoint/`.
Staging remains unchanged; whole-request speedup is still open. The user has
deferred device-loss reconstruction; retain the existing failure guards and focus
on normal-path performance and drawing coverage.


## Complete native rendering-request comparison

`record_gpu_frame.py --benchmark` extends the existing native-screen fixture with
fresh captures, eight animated units, JGL HUD/labels and final transfer. CPU/GPU/GPU/CPU
blocks each have eight warmup and 32 measured demands. Both arms use the same DLL,
local cache-enabled policy and full detail; input capture and pixel oracles are
outside timing. Cold pose transitions remain included. A separate desktop boundary
measures completion waiting, not physical scanout. Static map samples can retain an
old clock legitimately; units receive the current clock.

| Sequence | 640×480 CPU → GPU request mean | 1440×900 CPU → GPU request mean | 1440×900 with desktop boundary |
| --- | --- | --- | --- |
| Stationary animation | 35.70 → 33.64 ms | 43.28 → 35.11 ms | 49.42 → 46.28 ms |
| Dense scrolling | 119.74 → 45.29 ms | 166.92 → 51.51 ms | 176.76 → 61.89 ms |
| Local-change sequence | 41.59 → 32.78 ms | 48.68 → 35.64 ms | 54.91 → 46.83 ms |

Readiness-checkpoint receipts `0262a84bb2e342638b954534fa809fe8` / `77edf290622c4a49a9a78459f31db07a`
pass connected correctness and confirm untouched CPU map/screen storage throughout
GPU blocks. Candidate `4ef83685…483a2`, full distributions, sources and staging/rollback
receipts are in `native/build/gpu-composition/ready-content-checkpoint/`. The prior
`62faa141…5a2be1` candidate and both resolution runs remain in `whole-frame-control/`.
It established the complete benchmark and removed an unnecessary prior-CPU-call
prerequisite from native GPU startup. No new native hook or CSV entry is required.

The correction keeps unfinished unit predictions queued and adopts only ready CPU
pose content for speculative GPU submission. Helper completion wakes the internal
GPU owner; no polling or Civ III notification. GPU preparation median falls from
41.586 to 1.132 ms at 640×480 (139 prepared adoptions), and full GPU scrolling from
61.93 to 45.29 ms; at 1440×900, 76.69 to 51.51 ms. Idle still spends about 31 ms
per request in the unit group on average; p95 remains 130.66 ms at 1440×900.
Cold pose construction remains the dominant idle responsibility.

The initial `3f4f589b…a9b254b` run omitted the game's worker/cache policy and remains
nonrepresentative. `19e011f5…11d5d04` records the first readiness implementation;
an ownership test caught input leases surviving into notification. The final code
releases those leases before completion, and callback removal joins notification.
The full-worker test now retains its camera/cancellation/priority assertions
against the current interfaces. Pixel oracles remain unchanged; legacy terrain
boundary evidence is not reclassified. Device recovery stays deferred.

## Current native UI completion

One final presenter now accepts both resident images and completed CPU UI sources.
Partial CPU UI updates retain untouched full-color display pixels without restoring
the map; CPU fallback storage is valid only when complete. The final native Graphsy
hook binds this presenter from the process-owned module on configured UI demand,
before the first map and after scene unload. Previously the export existed but
its game binding depended on the first map load. The connected fixture now enters
through this hook and uses a distinct native-return sentinel to prove GPU
substitution, including config-off drain and reenable. UI-only demand creates the
device without map materials; startup menus retain native behavior until configured.
The actual-hook lifecycle checks pass at 640×480 (`5375f9cd23174597aa3119e8c8abc146`)
and 1440×900 (`1a0f410d02304019bfa086977c5d1a39`), followed by injected compilation.
`native-ui-lifecycle-checkpoint/` preserves current sources and these receipts with
the unchanged `deb24c11…514cec` DLL. Its performance reference remains the opacity
checkpoint; this connection is not claimed as an additional speedup. Foreground transfer preserves interrupted camera
input/tickets; background activity no longer selects GDI. No timing rules.

Normal HUD chrome/buttons also use JGL sprite slots 20/21/22, beyond ordinary
sprite slot 17. The production owner now submits all three through the existing
paired compositor. Slots 20/21 add premultiplied palette color to background ×
alpha/256 with native 555 arithmetic; slot 22 applies its distinct (alpha+1)/256
rule with separately truncated products. Actual JGL oracles cover all alpha values,
background aliasing, clipping, and palette/alpha edits. These programs key index
255 only; ordinary PCX sprite drawing still skips both 254/255. Full-color output
preserves the independent map contribution while retaining native UI colors.
Source decoding is bounded and reused; neither background nor destination is read
back. Straight-alpha out-of-bounds traversal and unsupported layouts stay native.
The same hooks preserve native private-lease scope before GPU admission. Slot 20's
native helper releases both borrowed links against background even when destination
is distinct; slot 21 over-releases and slot 22 omits its release. Each wrapper
restores the affected entry lease state after the scalar call.
This prevents false outstanding destination borrows without consuming caller-held
pointers. Actual startup/config-off tests cover held leases, clipping and aliasing.
The patch ledger records the verified runtime slots; no CSV entries were changed.

Connected receipts `d301d0077d9c413a9294d56607b951c6` (640×480) and
`5a0313cb04f44daa8c936ea0a9af55a7` (expanded 1440×900 benchmark)
pass with actual HUD slots 20/21/22 inside every timed request. Startup receipt
`6e70d5644c884e5e8635e5096ce65ad8` proves native lease preservation; injected
compilation and 23 portable contracts pass. The paired benchmark measures CPU →
GPU request means of 164.90 → 53.07 ms scrolling, 40.97 → 35.66 ms idle animation
and 47.65 → 35.13 ms local change; desktop-boundary means are 174.15 → 61.13,
47.87 → 47.07 and 54.41 → 46.79 ms. GPU idle p95 is 133.12 ms versus CPU 108.83 ms;
cold unit poses remain dominant. This preserves the established gain and extends
coverage; it does not establish another speedup over the readiness checkpoint.

The incomplete runs `217058ed…`, `8576c37e…`, `3913f228…`, and `2f2f0180…`
preserve the HUD lease diagnostics. The final run proves zero outstanding private
borrows and unchanged CPU map/screen pixels for every GPU block. No reset bypass,
benchmark-only lease release or timing workaround was used. The record tool now
preserves failed receipts even when stderr interrupts a sample line.

Prior staged `66351828…49fd30` and its exact sources remain in
`native/build/gpu-composition/native-ui-completion-checkpoint/`. The previous HUD candidate
`c0e9afbe…544193` was staged for ordinary `INSTALL.bat`; exact sources, receipts and
rollback identity are in `native-hud-completion-checkpoint/`.
No install or game launch occurred. Native UI/scroll/picking validation and
unsupported drawing coverage remain open; CPU source preparation is intentional.

## Native label panels and border lines

Normal map-label call sites use image slot 18 for 25/50-percent translucent
backgrounds. `PCX_Image` borders/indicators use image slot 25. Both are now routed
through the same production owner; no native destination pixels are acquired.
Tint extends the paired blend program with a constant native color and bounded
percentage. Native 555 word output matches JGL, while the independent full-color
map contribution follows the same background weight. Axis-aligned lines use fill
commands; other lines compile bounded coverage into the existing reusable sprite
source. JGL's X ordering, strict Bresenham tie rule, inclusive endpoints, clipping,
degenerate no-op and ignored final argument are preserved.

The ordinary native fallback remains available before GPU admission. Its diagonal
helper leaves imbalanced temporary borrows on some exits; the wrapper restores
entry lease state without consuming caller-held pointers. Startup tests exercise
those leases. The pinned-DLL slot checks and detach cover both added hooks; no
CSV entries changed. Device-loss reconstruction remains deferred.

The isolated native oracle passes 270 clipped line cases, translucent panels across
clamped percentages and colors, and full-color background contribution. The
connected benchmark now includes eight translucent map labels and border outlines
inside each timed frame. This is coverage completion, not a claim of a separate
speedup. Connected receipts `83805961e37447e792e874078f897c54` (640×480),
`062724e2db76480aa9ebf8d1b69ce369` (1440×900 paired benchmark), and
`2725454c44654ee894423b3e46fdb76e` (current DLL startup lifetimes) pass.
Injected compilation and 23 portable contracts pass. Live palette edits also match
native output. Each GPU benchmark block preserves CPU map/screen storage.

CPU → GPU whole-request means are 164.74 → 52.48 ms scrolling, 41.20 → 35.78 ms
idle animation, and 47.92 → 36.78 ms local change. Desktop-completion means are
174.16 → 61.91, 48.12 → 46.57, and 54.65 → 47.83 ms. Idle p95 regresses from
105.64 to 131.50 ms; unit work averages 31.38 ms of GPU's 35.78 ms request.
Cold pose/shadow work remains the performance priority. The same-DLL paired test
includes capture-free measurement and cold transitions; it is not live gameplay.

Candidate `70fa7eb6…995556` is staged for ordinary `INSTALL.bat`, with exact inputs,
logs, executables and receipts in `native-label-completion-checkpoint/` and the
previous DLL preserved in `native-hud-completion-checkpoint/`. No install/launch.
The subsequent lookup integration below addresses image slot 21 and sprite slot
33. CPU minimap/source-image preparation remains intentional and does not by itself
require map readback. Device-loss reconstruction stays outside this work.

## Lookup panels and indexed effects

Image panels, lookup sprites and native FLC bodies/shadows now submit through one paired GPU lookup program.
Image slot 21 occurs in native unit-control/panel painting; sprite slot 33 occurs
in native UI effects. Slot 35 combines palette pixels and two lookup ranges in
native FLC map/unit-control drawing. Slot 34 supplies the scaled FLC route used
by zoomed-out unit cursors and animations. All depend on existing destination pixels. Native fallback
would otherwise revoke resident ownership. CPU preparation copies caller-owned
lookup/source data; no destination pixels are acquired for these effects.

Two content-validated lookup assets coexist: the full FLC/UI table and native
four-block shadow table. Together they retain under 2.25 MiB CPU data and 4.5 MiB
GPU storage inside the existing composition budget; replacement needs under 6 MiB
additional CPU scratch. Same-pointer edits invalidate content, while alternating
shadow/FLC draws reuse both assets. Sprite coverage
reuses the existing bounded source cache and positive scaling program. Image
clipping, magenta background substitution, float-percent truncation, sprite index
rules and native no-ops follow the pinned JGL binary. Native scalar fallback
preserves image/FLC routines' unreleased entry leases. No CSV edits or timers.

Native-word output is exact; independent full-color pixels interpolate the same
lookup lattice. Identity and channel-remapping tables preserve full map precision.
This extends color handling for these UI effects; actual effect appearance still
belongs to the strategic game checkpoint. Mirrored plain lookup sprites and unsupported
FLC source formats retain fallback; the tested native FLC form preserves its own
mirror/scale no-op semantics. Scaled FLC preserves native clipping followed by
source traversal from byte zero and every-second-byte selection. It feeds the same
lookup submission; CPU preparation never reads either destination or underlay.
Single-key Advisor/UI artwork, solid masks and native map shadows also feed the
same sprite/lookup owner through slots 23/29/31. CPU-owned picking masks remain
native. The final native oracle (`6618b7addf1d422daf7e1186b9a9f528`) passes raw
and trimmed sources, scaling/clipping, native return/metadata behavior, full-color
shadow/fade preservation and simultaneous lookup reuse without destination readbacks.
Connected 640×480 (`383b5296928d48bf84937b74fcf9a984`), startup/config-off leases
(`a4f2211559a648b6bdc94a5475fc8b5e`), injected compilation and 23 portable
contracts pass. The 1440×900 benchmark (`268f78b251f94adc9c8a9124032c6a26`)
includes these forms plus opacity transitions each request. CPU → GPU request
means are 163.07 → 65.11 ms scrolling, 41.60 → 38.13 ms stationary animation and
47.79 → 38.82 ms local change; desktop-boundary means are 173.13 → 75.44,
48.11 → 47.06 and 54.38 → 47.10 ms. GPU idle p95 remains 131.23 ms
(CPU 109.23), dominated by unit work. These compare complete delivery routes,
not an isolated gain over the previous DLL. GPU UI/output costs 5.64–7.50 ms mean
versus CPU 1.53–1.55 ms; all remain included. No execution readbacks occur in the
admitted chain; contiguous free virtual space stays above 1.49 GB.
Candidate `deb24c11…514cec` is staged for ordinary `INSTALL.bat`; exact source,
DLL and receipts are in `native-opacity-completion-checkpoint/`. The prior sprite
family checkpoint is preserved. No install or game launch was performed; actual
gameplay coverage remains unverified.

Sprite slot 37 (`FUN_005f8a70`, pinned JGL RVA `0x9220`) now supplies command-panel
icons and UI fades through the existing sprite owner and blend shader. Its native
sixteenth-step weights are exact, including threshold neighbors and the unusual
15/16 result at opacity 1 (the unreachable opaque branch compares against 100).
Low-byte flags select either index 255 alone or 248–255 as transparent. The native
oracle verifies clipping, positive scaling, trimmed no-ops and full-color map
contributions with zero destination readbacks. Native fallback preserves existing
caller leases. No new renderer owner, timer or device-loss work is introduced.
PCX wrappers for image slots 34/36–41 resolve to pinned JGL RVA `0x3be40`, a shared
unsupported stub. Native minimap/source-image preparation remains intentional;
neither is an additional map composition responsibility. Actual game-screen
coverage is still pending and cannot be inferred from isolated/native replay tests.
