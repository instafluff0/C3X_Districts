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
`6.575e-7` tile units. UVs are unchanged and output normals remain unit length.
The synthetic executable checks interpolation, nonuniform-scale normal
transforms, one-shot endpoints, loop wrapping, repeated/skipped callbacks,
invalid timing and malformed/truncated/oversized payload rejection.

Windows x86 `/W4 /WX` compilation and the executable animation checks passed
through `renderer_dev.windows_command_result` and `BUILD.bat candidate-compile`.
The normal build workflow now runs these executable checks. No injected source,
CSV address, installed game executable, or staged production DLL was changed
for this foundation. The game was not launched. This evidence establishes
playback math and payload safety, not rendered animation or in-game movement.

## Work still required before the user-run checkpoint

1. Bind resource assets and calibrate southeast orientation. Normalized assets
   do not all share a forward axis: rest-pose bone evidence places horse heads
   toward −Y and cattle heads toward +Y. Deer, foxes and elephants also point
   toward −Y. Fish are an authored school with differing member headings;
   validate its intended resource orientation visually. Do not apply one random
   or guessed yaw to all assets. Runtime +X projects southeast in the existing
   Civ III isometric basis.
2. Add a separate resource dynamic layer. Keep static terrain pixels, meshes,
   prepared blocks and shadow pages independent of presentation time. Preserve
   foreground occlusion, fog, ownership and dirty-region restoration. Both
   native worker fast-publication and viewport-cache hits currently bypass
   dynamic work; both must route to composition when visible ambient resources
   need a new frame. Extend animation demand only for visible replacements.
3. Deliver enabled units on the native Animator canvas, with authoritative
   pixel anchors, direction, action/progress, selection/HUD and movement timing.
   `docs/i20_native_unit_animation_handoff.md` and the body-boundary spike remain
   the native contracts. The tile-capture unit fields are descriptive and cannot
   substitute for this dynamic boundary. Do not bake units into terrain.
4. Add bounded debug summaries for asset binding, orientation, action transitions,
   visible instance count, pose reuse, dirty bounds, timing and failure. Use the
   existing debug output; no per-vertex or automatic file logging.
5. Run temporal renders, camera/fog/overlap and cache witnesses, disabled-unit
   controls, Windows native builds, and the approved injected compile only if a
   minimal capture/forwarding bridge changes. Stage the tested checkpoint for
   normal `INSTALL.bat`, then stop for user gameplay testing without launching.

## Native dependencies

No new patch capability is needed by the current offline compiler/evaluator.
The ordinary map boundary and scheduler already support resource demand.
`Unit_tick_anim` currently has a callable GOG-only CSV definition, not an
inlead. The old unit-body audit identifies GOG normal/reduced draw sites, but
the reduced ABI and other supported builds still need proof before any concrete
CSV change request. Do not edit CSV entries or suppress native unit pixels
before the replacement path succeeds.
