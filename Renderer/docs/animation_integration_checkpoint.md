# Resource and unit animation integration

The user authorized animated resources facing southeast and enabled unit bodies
following native Civ III movement. Start with a small functional checkpoint;
the user launches the game. Keep renderer-specific code in the DLL, retain the
normal `INSTALL.bat` entry point and ordinary `OutputDebugStringA` diagnostics.
Edge speckling remains deferred. This work does not advance unrelated Lab gates
or M9–M11.

## Current implementation

`native/animation_runtime.h` provides a source-independent binary skin-palette
decoder and evaluator for the DLL. It accepts one material part per payload,
validates dimensions and exact byte length before allocation, bounds each file
to 64 MiB, rejects nonfinite/non-affine matrices and invalid indices or weights,
and leaves the previous decoded asset unchanged on rejection. Playback samples
absolute time, interpolates adjacent imported pose palettes, transforms normals
with the inverse transpose and preserves the final pose for native one-shot
actions. It does not own gameplay state, a timer, or terrain invalidation.

`tools/asset_compiler/build_resource_animation_runtime.py` compiles existing
validated model-aware resources. The ignored `ResourceAnimationRuntime` pack
contains 26 subjects across bananas, cattle, furs, game, horses, ivory, rubber,
wheat, fish and whales. Skin-palette payloads total 8,946,048 bytes, plus
deduplicated textures. All imported samples remain available; this is not the
Lab's sparse action-pose mesh bundle. The runtime manifest remains explicitly
disabled until the dynamic layer and orientation calibration are verified.

The binary format is little-endian: eight-byte `C3XANM1\0`, version u32, vertex
count u32, triangle-index count u32, bone count u32, frame count u32, duration
f32; followed by vertices (position3, normal3, UV2 f32; joints4 u32; weights4
f32), u32 triangle indices, then frame-major/bone-major row-vector 4×4 f32 skin
palettes. Offline binding multiplies each inverse bind matrix by that frame's
validated world matrix. Payloads contain no paths or source-engine types.

## Verified evidence

`python3 -m unittest Renderer.native.test_animation_runtime -v` executes the
actual C++ evaluator. The local licensed witness covers all 26 subjects at four
authored times each, including both endpoints. All 104 poses agree with the
independent normalized CPU skinning implementation within
`3.363e-8` tile units after the translation correction below. Every subject
changes its body positions across the sampled clip. UVs are unchanged and output normals remain unit length.
The synthetic executable checks interpolation, nonuniform-scale normal
transforms, one-shot endpoints, loop wrapping, repeated/skipped callbacks,
invalid timing and malformed/truncated/oversized payload rejection.

Windows x86 `/W4 /WX` compilation and the executable animation checks passed
through `renderer_dev.windows_command_result` and `BUILD.bat candidate-compile`.
The normal build workflow now runs these executable checks. No injected source,
CSV address, installed game executable, or staged production DLL was changed
for this candidate. The game was not launched. Native temporal rendering now
passes too; in-game movement and redraw scheduling still require the later
manual checkpoint.

## Dynamic resource candidate

The DLL now loads one calibrated primary subject for each of the ten resource
families when the verification switch `C3X_RENDERER_RESOURCE_ANIMATION=1` is set.
No special user launcher is planned: this switch is temporary until the full
checkpoint is staged. The production DLL in `Renderer/bin` is unchanged.

Animated bodies are absent from static meshes and retained pixel caches. Tile
geometry retains only immutable resource anchors. A separate posed vertex-buffer
pool (32 MiB maximum) and completed bitmap handle animation. Repeated redraws
within the same 15 Hz absolute-time quantum reuse the published image without
cancelling background terrain preparation. New phases drive the existing visible
animation demand. Resource ownership is retained on mesh-cache hits, including
fixed scrolling; units remain outside this map plane.

The dynamic pass restores exact MSAA4 linear color and depth behind each body,
then draws just the new poses with native terrain occlusion. A maximum of 32
128×128 background blocks costs 24 MiB, independent of window size. At capacity,
uncached regions redraw normally rather than dropping bodies. These blocks are
invalidated by the complete authoritative scene signature. Static terrain, mesh,
pixel-block, recent-view and source-shadow budgets remain unchanged. Resource
bodies receive static source shadows but do not animate their own cast shadows
in this first checkpoint; they never invalidate static source-shadow pages.

The first actual renders caught an importer defect that numeric palette parity
alone could not detect: clutter animation translations had been exported at
1/12 while their skeletons were normalized at 1/100. Clips also retained constant
source-scene root placement. The runtime compiler now matches the exact clip
hash to extraction metadata, converts authored translation channels by the
proven 0.12 ratio, preserves already-normalized missing tracks, and removes only
the initial root-placement offset. Animated root deltas, rotations and scale
remain intact. A synthetic executable calibration test checks all three rules.
Source packs and active Lab code are not rewritten. Corrected renders show intact
horses, cattle, deer, foxes, elephants and plants. Generic bindings rotate −Y
forward models by +90° and +Y cattle by −90°, projecting southeast. The fish
school retains its authored internal member headings within the resource body.

`python3 Renderer/native/verify_animation.py` runs the candidate offscreen in the
Windows VM. Current evidence in `verification/animation/temporal.json` covers:

- Six timed frames at normal/reduced zoom and noon/night; all five phase changes
  change pixels, while repeated callbacks at the same time are byte-identical.
- All ten synthetic resource families claim replacement ownership. Animated
  updates build zero terrain meshes and upload zero terrain bytes.
- A four-raw-tile camera jump compared with a reset and fresh render: zero pixels
  differing by more than two channel levels, total channel error 0/0/1.
- Resource removal stops animation demand and restores the base map; comparison
  with fresh rendering has zero differing pixels and total error 2/1/1.
- Background caching matches the preceding uncached dynamic implementation with
  total channel error 0/1/2 over the three inspected frames, no difference above
  one channel level. The reduced-zoom compositor takes about 4–5 ms, with roughly
  9–11 ms API-call time in the latest headless run. This is not gameplay evidence.
- The native portable smoke passes with animation disabled, including actual
  RGB555/RGB565 destination copies. Thirty-seven focused renderer/playback and
  calibration checks pass. No injected compilation is required for DLL-only work.

Usual `OutputDebugStringA` records include `animation-bind` and `animation-frame`:
resource identity, load outcome, southeast facing, clock quantum, visible bodies,
dirty rectangles/pixels, pose uploads, buffer/backdrop bytes, cache hits/misses,
terrain rebuild count and elapsed time. Runtime does not open a log file. The
headless verifier explicitly opts into its local trace files.

## Unit playback data and native pose adapter (2026-09-07)

`tools/asset_compiler/build_unit_animation_runtime.py` now exports complete clips
for Warrior, Archer, Swordsman, Infantry, Fighter and Galley: 48 actions, 177
deduplicated payloads, 19,192,048 bytes. Source packs remain unchanged. Skinned
parts and rigid socket attachments use the same bounded DLL palette evaluator;
weapons are never recentered independently. All source texture channels, source
tints and existing owner-mask contracts are preserved as data. Warrior's source
owner-tint contract remains unresolved and must be addressed before enabling its
custom body. Missing actions are not invented or silently aliased.

Whole-kit planar root travel is removed while joint and vertical movement are
retained. This aligns the mixed input packs: the older Warrior model-aware cache
already strips root travel, whereas family exports retain it. Native screen
anchors remain the only source of gameplay movement. A comparison against raw
authored clips, with this explicit planar normalization, passes 576 part poses,
including 192 socket poses, within 2.256e-7 tiles using the actual C++ evaluator.
See `verification/animation/unit-payloads-portable_cpp.json`. This is numerical
pose/material preservation evidence, not a rendered or in-game unit gate.

`native/unit_animation_runtime.h` derives pose phase directly from the captured
native action cursor, preserves one-shot endpoints, and reconstructs the native
anchor with exact normal/reduced integer arithmetic. It rejects unrelated
Sprite/canvas identities and unsupported actions. The Windows x86 `/W4 /WX`
candidate build and executable cursor/anchor/rejection checks pass. Full source
pose comparisons ran on the portable host; a separate attempt to run Python
tests inside Windows could not run because that VM has no `py` launcher.
The adapter is not connected to live hooks yet; no unit draw ownership is claimed.

The next implementation must connect complete-kit rendering and the minimal
native body bridge, then verify source tint/owner color, directions, native
movement and retained overlays. Do not substitute the old sparse baked Lab
action samples for these full clips. Animation remains unstaged, and Civ III has
not been launched.

## Work still required before the user-run checkpoint

### Body renderer and bridge increment

`native/unit_body_renderer.h` now renders complete unit bodies offscreen on the
existing D3D worker, then copies premultiplied pixels to the native Animator
canvas. Five families currently qualify: Archer, Swordsman, Infantry, Fighter
and Galley. Warrior remains native pending its missing owner-material contract;
Armies and other unsupported families remain entirely native. A single fixed
idle-stance fit gives readable normal-zoom bodies without per-pose resizing.

The independent sprite cache is capped at 8 MiB / 128 entries and excludes
unit identity and pixel position. Its key includes action/cursor, direction,
source dimensions, zoom, effective color and lighting. Repeated poses and new
screen occurrences reuse the same bitmap. Unit jobs retain the terrain worker's
completed publication and queued preparation. Bounded unit scratch/mesh buffers
never enter terrain cache keys or geometry ownership.

`C3X.h` adds only the optional DLL entry and scoped unit/canvas identities.
The three wrappers in `injected_code.c` capture the existing current action,
cursor, exact Sprite/canvas, palette and coordinates and call the original
native functions. Rendering, skinning, lighting, sprite caching and body logging
stay in the DLL. The approved injected compile passes. The actual extracted
wrappers also pass executable x86 checks for both zooms, native fallback,
disabled/invisible/Army controls, scoped restoration and retained underlay/HUD.

The GOG disassembly proved the current reduced CSV declaration is wrong:
`RET 0x24` consumes nine stack arguments. The concrete three-row human pickup is
`handoffs/animation_unit_hooks_gog.md`; other-build addresses remain unverified.
Until those rows become inleads, no live body replacement occurs. Both animation
loader switches are still verification-only and the candidate remains unstaged.

`python3 Renderer/native/verify_units.py` exercises the real DLL body renderer,
eight directions, both zooms, temporal changes, exact repeated/translated sprite
reuse, native fallback, resource animation and terrain reconstruction after unit
draws. Reports are in `verification/animation/units.json`. `unit-body` debug
records include unit/key, native current/queued action, cursor/count, direction,
anchor, zoom, effective color, result, sprite-cache hit/bytes and elapsed time.
The normal runtime continues to use OutputDebugStringA only.

1. Finish the enabled-unit body path on the native Animator canvas, including
   authoritative pixel anchors, direction, native action/progress, retained HUD,
   movement timing and focused action-transition logging. Tile unit fields are
   descriptive and cannot substitute for this boundary.
2. Finish southeast calibration for the marine school members, then validate
   the unit and resource scene together, fog/visibility transitions,
   overlap with retained native objects, interrupted actions and disabled-unit
   controls. Prove native map redraw scheduling with the user at the checkpoint.
3. Resolve any concrete native hook changes, compile the minimal capture/forwarding
   bridge with the approved smoke if needed, remove the temporary resource test
   switch and stage the verified DLL for normal `INSTALL.bat`. Do not launch the
   game. The complete animation goal remains active.

## Native dependencies

No new patch capability is needed by the current offline compiler/evaluator.
The ordinary map boundary and scheduler already support resource demand.
`Unit_tick_anim`, `Sprite_draw_unit_body_normal`, and
`Sprite_draw_unit_body_reduced` now have callable GOG-only CSV definitions;
they are not inleads. The old unit-body audit and current signatures must be
reconciled against the executable before a concrete hook request. Other builds
remain unaudited for this boundary. Do not edit CSV entries or suppress native unit pixels
before the replacement path succeeds.
